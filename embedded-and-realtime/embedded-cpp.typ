#import "../template.typ": xref

= C++ for Embedded Systems <embedded-cpp>

C++ on a microcontroller is not a smaller C++. It is the same language with a different cost model. There is no heap you can afford to fragment, no operating system to catch a stray pointer, flash measured in hundreds of kilobytes, and interrupt handlers that can run between any two instructions. Used well, C++ gives an embedded project things C cannot: compile-time computation, type-safe hardware access with no runtime cost, RAII for interrupt masks and locks, and strong types for physical units. Used carelessly, it pulls in a heap, exception tables and 100 KB of `iostream`. This chapter sets out which features cost what, and the idioms that make modern C++ a good fit for firmware.

*See also:* #xref("embedded-and-realtime", "linker-scripts-and-startup", label: "Linker Scripts, Startup Code, and the Memory Map"), #xref("embedded-and-realtime", "peripherals-and-drivers", label: "Peripherals and Drivers"), #xref("embedded-and-realtime", "rtos", label: "Real-Time Operating Systems"), #xref("cpu-architecture", "synchronization", label: "Synchronization") (memory ordering)

== Freestanding C++ and the Feature Cost Table

The C++ standard defines two kinds of implementation. A *hosted* implementation has the full library. A *freestanding* one needs only a small subset: `<cstdint>`, `<cstddef>`, `<limits>`, `<type_traits>`, `<atomic>`, `<new>` (for placement `new`), `<initializer_list>`, `<bit>` and a few more. C++23 (P1642) added large parts of `<utility>`, `<tuple>`, `<ranges>`, `<iterator>`, `<memory>` (non-allocating parts) and the non-allocating algorithms to the freestanding set. In practice most firmware is built with a hosted toolchain (arm-none-eabi-gcc with newlib or picolibc) and a discipline about which parts of the library it uses.

#table(
  columns: 3,
  [*Feature*], [*Runtime cost*], [*Typical firmware policy*],
  [Classes, templates, `constexpr`, `consteval`], [None beyond the code you instantiate], [Use freely; watch template bloat],
  [Virtual functions], [One vtable in flash per class, one pointer per object, an indirect call], [Allowed; prefer static dispatch on hot paths],
  [Lambdas without captures, captures by value], [Same as a function or a struct], [Use freely],
  [`std::array`, `std::span`, `std::optional`, `std::variant`], [None or a tag byte], [Use freely],
  [`std::atomic`], [Exclusive load/store loops or library calls on Armv6-M], [Use for ISR and task sharing],
  [RTTI (`typeid`, `dynamic_cast`)], [Type-info objects and name strings in flash], [Disable: `-fno-rtti`],
  [Exceptions], [Unwind tables, personality routine, heap allocation of exception objects, unbounded throw time], [Disable: `-fno-exceptions`],
  [Heap containers (`std::vector`, `std::string`, `std::map`)], [Allocator time and fragmentation], [Forbidden after initialisation],
  [`std::function`], [May heap-allocate captured state], [Replace with a non-allocating delegate],
  [`iostream`], [Over 100 KB of flash, static constructors], [Forbidden],
)

The exceptions row deserves an explanation, because the reason is not only size. Throwing allocates the exception object (newlib uses `malloc`, with an emergency pool as fallback), and the time to unwind depends on stack depth and on table lookups. Neither fits a timing analysis. MISRA C++:2023 and AUTOSAR C++14 both permit exceptions under restrictions, but most hard real-time projects disable them, and the standard library then calls `std::abort` where it would have thrown.

== Error Handling Without Exceptions

With exceptions off, errors travel in return values. C++23's `std::expected<T, E>` holds either a value or an error and does not allocate. Combined with `[[nodiscard]]`, the compiler warns about any ignored result:

```cpp
#include <cstdint>
#include <expected>
#include <span>

enum class i2c_error : std::uint8_t { nack, bus_busy, arbitration_lost, timeout };

[[nodiscard]] std::expected<std::uint16_t, i2c_error>
read_register16(std::uint8_t device, std::uint8_t reg);

[[nodiscard]] std::expected<float, i2c_error> read_temperature_c() {
    return read_register16(0x48, 0x00).transform([](std::uint16_t raw) {
        // TMP102: 12-bit two's complement, 0.0625 degrees C per LSB
        return static_cast<float>(static_cast<std::int16_t>(raw) >> 4) * 0.0625f;
    });
}

void poll_sensor() {
    if (auto t = read_temperature_c()) {
        publish(*t);
    } else if (t.error() == i2c_error::bus_busy) {
        schedule_retry();
    } else {
        report_fault(t.error());
    }
}
```

Toolchains without C++23 can use `tl::expected` or a small hand-written equivalent. The pattern matters more than the spelling: every fallible call returns its failure in the type, and the type forces the caller to look at it.

== Memory Without a Heap

Most safety standards and most experienced firmware teams allow dynamic allocation only during initialisation, or not at all. The tools for living without `new`:

- *Static storage* for every long-lived object, sized for the worst case at compile time.
- *Fixed-capacity containers*: the Embedded Template Library (ETL) provides `etl::vector<T, N>`, `etl::string<N>`, `etl::map<K, V, N>` and others with the familiar interfaces and storage inside the object. C++26 adds `std::inplace_vector<T, N>`.
- *Object pools*: a fixed array of slots plus a free list gives constant-time allocate and free with no fragmentation, because every slot has the same size.
- *Placement `new`* constructs an object in storage you already own, which is how pools and fixed-capacity containers work inside.
- *`std::pmr::monotonic_buffer_resource`* over a static buffer gives standard containers an arena that never frees individual objects, which is a good fit for "build tables at boot, then never change them".

```cpp
#include <array>
#include <cstddef>
#include <new>
#include <utility>

template <typename T, std::size_t N>
class object_pool {
public:
    object_pool() {
        for (std::size_t i = 0; i + 1 < N; ++i) slots_[i].next = &slots_[i + 1];
        slots_[N - 1].next = nullptr;
        free_ = &slots_[0];
    }

    template <typename... Args>
    T* create(Args&&... args) {
        slot* s = free_;
        if (s == nullptr) return nullptr;            // exhausted: caller decides
        free_ = s->next;
        return ::new (static_cast<void*>(s->storage)) T(std::forward<Args>(args)...);
    }

    void destroy(T* p) {
        p->~T();
        slot* s = reinterpret_cast<slot*>(p);
        s->next = free_;
        free_ = s;
    }

private:
    union slot {
        slot* next;
        alignas(T) std::byte storage[sizeof(T)];
    };
    std::array<slot, N> slots_{};
    slot* free_ = nullptr;
};
```

The pool is not thread-safe or interrupt-safe as written. If both an ISR and a task allocate, guard `create` and `destroy` with a critical section (below) or use a lock-free free list.

To catch accidental heap use, define `operator new` to trap and link with `-Wl,--wrap=malloc` pointing at a function that halts with a diagnostic. The first stray `std::string` in a log statement is then a test failure, not a fragmentation bug in the field.

== Hardware Registers: volatile and Zero-Cost Abstraction

=== What volatile Does and Does Not Do

`volatile` tells the compiler that each access to the object is an observable side effect: it must not be removed, merged, or reordered with respect to *other volatile accesses*. That is exactly what a memory-mapped register needs, since reading a status register can clear a flag and writing a data register sends a byte. `volatile` does *not*:

- make an access atomic (a 64-bit volatile access on a 32-bit core is two bus transactions),
- order volatile accesses against ordinary memory accesses (the compiler may move a write to a normal buffer past a volatile write that starts a DMA transfer),
- emit hardware barriers.

So `volatile` is for device registers, and `std::atomic` is for data shared between contexts. C++20 deprecated compound assignment and increment on volatile objects because `reg |= bit` looks atomic and is a read-modify-write. C++23 (P2327) restored the bitwise forms `|=`, `&=` and `^=`, which register code uses constantly, but the hazard is unchanged: an ISR that writes the same register between the read and the write loses its update. Peripherals with separate set and reset registers (STM32 `BSRR`) or hardware atomic aliases (RP2040's XOR, SET and CLR register aliases) exist for exactly this reason.

=== Type-Safe Registers at Zero Cost

Raw `#define`d addresses and hand-written masks are where firmware bugs breed: the wrong shift, the wrong register, an enum value written into the wrong field. Templates move all of that to compile time:

```cpp
#include <cstdint>
#include <type_traits>
#include <utility>

template <std::uintptr_t Address>
struct reg32 {
    static volatile std::uint32_t& ref() {
        return *reinterpret_cast<volatile std::uint32_t*>(Address);
    }
    static std::uint32_t read() { return ref(); }
    static void write(std::uint32_t v) { ref() = v; }
};

template <typename Reg, unsigned Offset, unsigned Width, typename Value = std::uint32_t>
struct field {
    static_assert(Offset + Width <= 32, "field exceeds register");
    static constexpr std::uint32_t mask =
        (Width == 32 ? ~0u : ((1u << Width) - 1u)) << Offset;

    static void write(Value v) {
        const auto raw = static_cast<std::uint32_t>(v);
        Reg::write((Reg::read() & ~mask) | ((raw << Offset) & mask));
    }
    static Value read() {
        return static_cast<Value>((Reg::read() & mask) >> Offset);
    }
};

// STM32F4 GPIOD, from the reference manual
enum class gpio_mode : std::uint32_t { input = 0, output = 1, alternate = 2, analog = 3 };

using gpiod_moder = reg32<0x4002'0C00u>;
using gpiod_bsrr  = reg32<0x4002'0C18u>;

template <unsigned Pin>
using gpiod_pin_mode = field<gpiod_moder, Pin * 2, 2, gpio_mode>;

template <unsigned Pin>
struct gpiod_pin {
    static_assert(Pin < 16);
    static void make_output() { gpiod_pin_mode<Pin>::write(gpio_mode::output); }
    static void set()   { gpiod_bsrr::write(1u << Pin); }          // atomic set
    static void clear() { gpiod_bsrr::write(1u << (Pin + 16)); }   // atomic reset
};

using status_led = gpiod_pin<12>;

void init_led() {
    status_led::make_output();
    status_led::set();
}
```

At `-O1` and above, `init_led` compiles to the same load, mask, or, store and single store a careful C programmer would write by hand. The template parameters fold into immediate constants. What the types add: `gpiod_pin_mode<12>::write(3)` does not compile (an `int` is not a `gpio_mode`), and `gpiod_pin<16>` fails its `static_assert`. Vendor-generated headers from SVD files (for example with `svd2cpp`-style generators) produce exactly this kind of layer for every register of a part.

== Sharing Data Between Interrupts and Tasks

An ISR is a second thread of control on the same core. Two rules cover nearly every case.

*Use `std::atomic` for flags and indices.* On Armv7-M and Armv8-M, word-sized `load` and `store` compile to plain `LDR` and `STR`, and read-modify-write operations to `LDREX`/`STREX` loops. With `memory_order_release` and `memory_order_acquire` the compiler also emits `DMB` and, more importantly, refuses to move ordinary memory accesses across the atomic. Arm's barrier guidance (application note 321) observes that current single-core Cortex-M implementations do not reorder normal memory accesses in a way an ISR on the same core can see, but the architecture permits it, and the compiler certainly reorders without the atomic. On Armv6-M (Cortex-M0 and M0+) there is no exclusive monitor. Plain loads and stores are still lock-free, but `fetch_add` and `compare_exchange` become calls to `__atomic_*` library functions, which the platform must implement, usually by masking interrupts.

*Use RAII for critical sections.* When the shared state is bigger than a word, mask interrupts around the update. An RAII guard makes the unmask impossible to forget on any return path:

```cpp
#include <cstdint>

class irq_mask_guard {
public:
    // Masks all interrupts at or below `level` (lower number = higher priority)
    // using BASEPRI, so higher-priority interrupts keep their latency.
    explicit irq_mask_guard(std::uint32_t level) {
        asm volatile("mrs %0, basepri" : "=r"(saved_));
        const std::uint32_t value = level << (8 - priority_bits);
        asm volatile("msr basepri_max, %0" :: "r"(value) : "memory");
    }
    ~irq_mask_guard() {
        asm volatile("msr basepri, %0" :: "r"(saved_) : "memory");
    }
    irq_mask_guard(const irq_mask_guard&) = delete;
    irq_mask_guard& operator=(const irq_mask_guard&) = delete;

private:
    static constexpr unsigned priority_bits = 4;   // STM32: 4 implemented bits
    std::uint32_t saved_;
};

struct motor_setpoint {
    float speed_rpm;
    float torque_limit_nm;
};

motor_setpoint g_setpoint;   // read by the 10 kHz control ISR at priority 2

void update_setpoint(const motor_setpoint& next) {
    irq_mask_guard guard{2};   // block the control ISR, not the fault handlers
    g_setpoint = next;         // the ISR never sees a half-written struct
}
```

`BASEPRI_MAX` only ever raises the mask, so nested guards compose correctly, and the `"memory"` clobbers stop the compiler moving accesses to `g_setpoint` outside the guarded region. The alternative for larger data is double buffering: the writer fills the inactive copy and atomically flips an index, and the ISR never blocks at all.

== Calling C++ from Interrupt Vectors

The vector table holds plain function pointers, so handlers must be non-member functions with C linkage and the exact names the startup file expects. To dispatch into a driver object, keep a pointer to the instance:

```cpp
class uart_driver {
public:
    void on_interrupt();      // reads status, drains FIFO
    static uart_driver* instance;
};

uart_driver* uart_driver::instance = nullptr;

extern "C" void USART2_IRQHandler() {
    uart_driver::instance->on_interrupt();
}
```

A template-based alternative binds the handler at compile time with no pointer at all (`template <auto& Driver> void irq_trampoline() { Driver.on_interrupt(); }`), at the cost of a vector table built in C++ rather than in the vendor's assembly startup file.

== Doing the Work at Compile Time

`constexpr` and `consteval` move computation from the $"MCU"$ to the compiler, and the result lands in `.rodata` (flash) instead of costing startup time and SRAM. CRC tables, sine tables, baud-rate divisors and filter coefficients are classic candidates:

```cpp
#include <array>
#include <cstdint>

consteval std::array<std::uint32_t, 256> make_crc32_table() {
    std::array<std::uint32_t, 256> table{};
    for (std::uint32_t i = 0; i < 256; ++i) {
        std::uint32_t c = i;
        for (int k = 0; k < 8; ++k) {
            c = (c & 1u) ? (0xEDB8'8320u ^ (c >> 1)) : (c >> 1);
        }
        table[i] = c;
    }
    return table;
}

inline constexpr auto crc32_table = make_crc32_table();   // 1 KB in flash

consteval std::uint32_t usart_brr(std::uint32_t pclk_hz, std::uint32_t baud) {
    const std::uint32_t div = (pclk_hz + baud / 2) / baud;   // oversampling by 16
    if (div < 16 || div > 0xFFFF) throw "baud rate not reachable";  // compile error
    return div;
}

static_assert(usart_brr(42'000'000, 115'200) == 365);
```

The `throw` inside a `consteval` function is not a runtime exception. If the condition holds, the call is not a constant expression and compilation fails with that message, so an impossible clock and baud combination is a build error.

== Static Polymorphism and Code Size

Templates generate one copy of the code per distinct set of arguments. A driver templated on a pin number, instantiated for 30 pins, can produce 30 copies of the same function. The usual remedies:

- Template only the thin layer that needs compile-time constants, and delegate to a non-template core that takes runtime parameters.
- Prefer `constexpr` function parameters folded by inlining over template parameters where the code is large.
- Check the map file after adding a template-heavy abstraction (#xref("embedded-and-realtime", "linker-scripts-and-startup", label: "Linker Scripts, Startup Code, and the Memory Map")).

Virtual functions are not the enemy they are sometimes made out to be. A vtable lives in flash, and an indirect call costs a few cycles. For a driver interface with several implementations selected at runtime, `virtual` is the clearest and often the smallest solution. Static polymorphism (templates, CRTP, concepts) wins when the implementation is fixed at build time and inlining matters, as in register access and tight signal-processing loops.

== Strong Types for Physical Quantities

A large share of control-software defects are unit errors: milliseconds passed as ticks, degrees as radians, millivolts as volts. `std::chrono` already solves this for time, and the same idea extends to any quantity:

```cpp
#include <chrono>
using namespace std::chrono_literals;

void sleep_for(std::chrono::milliseconds d);

sleep_for(250ms);    // fine
sleep_for(2s);       // converts exactly to 2000 ms
// sleep_for(250);   // does not compile: a bare int has no unit
```

Libraries such as mp-units (proposed for standardisation) provide compile-time dimensional analysis for every physical quantity, again at zero runtime cost, because the units exist only in the types.

== A Recommended Baseline

#table(
  columns: 2,
  [*Setting*], [*Reason*],
  [`-std=c++20` or later], [`constinit`, `consteval`, concepts, `std::span`, designated initialisers],
  [`-fno-exceptions -fno-rtti`], [Size, determinism],
  [`-fno-threadsafe-statics`], [No guard-variable locking; initialise statics before the scheduler],
  [`-ffunction-sections -fdata-sections -Wl,--gc-sections`], [Drop unused code],
  [`-Wall -Wextra -Wconversion -Wshadow -Werror`], [Narrowing and sign errors are common hardware bugs],
  [Trap on `operator new` and `malloc` after init], [Enforce the no-heap policy mechanically],
  [MISRA C++:2023 or AUTOSAR C++14 subset with a checker], [Required for safety certification; useful anyway],
)

== Further Reading

Stroustrup, B. (2013). "The C++ Programming Language." Addison-Wesley, 4th ed.

Kormanyos, C. (2021). "Real-Time C++: Efficient Object-Oriented and Template Microcontroller Programming." Springer, 4th ed.

MISRA (2023). "MISRA C++:2023 Guidelines for the use of C++17 in critical systems." Standard.

AUTOSAR (2019). "Guidelines for the use of the C++14 language in critical and safety-related systems." AUTOSAR specification R19-03.

Arm (2012). "ARM Cortex-M Programming Guide to Memory Barrier Instructions." Application Note 321.

Bastien, J. (2019). "Deprecating volatile." WG21 paper P1152R4.

Craig, B. (2022). "Freestanding Library: Easy [utilities], [ranges], and [iterators]." WG21 paper P1642R11.

Wellings, J. (2024). "Embedded Template Library Documentation." etlcpp.com reference.

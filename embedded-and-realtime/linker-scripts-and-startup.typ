#import "../template.typ": xref

= Linker Scripts, Startup Code, and the Memory Map <linker-scripts-and-startup>

On a hosted system the loader, the C runtime and the operating system decide where code and data live and prepare them before `main` runs. On a bare-metal $"MCU"$ nobody does that for you. The linker script decides the address of every byte, and a few dozen lines of startup code turn a flash image into a running C++ program. This chapter takes both apart: the sections a toolchain emits, the difference between where data is stored and where it runs, a complete annotated linker script, a startup routine written in C++, and the techniques for reading map files and shrinking images.

*See also:* #xref("embedded-and-realtime", "mcus-and-soc", label: "Microcontrollers and Systems-on-Chip"), #xref("embedded-and-realtime", "embedded-cpp", label: "C++ for Embedded Systems"), #xref("compilers", "backend-and-codegen", label: "Backend and Code Generation") (compilers), #xref("operating-systems", "boot-and-init", label: "Boot and Init") (operating systems)

== Sections: What the Compiler Emits

Every object file is a set of named *input sections*. With `-ffunction-sections -fdata-sections`, GCC and Clang give every function and every global its own section (`.text._Z9pid_stepf`, `.bss.g_rx_buffer`), which lets the linker discard unreferenced ones. The families that matter on an $"MCU"$:

#table(
  columns: 4,
  [*Section*], [*Contents*], [*Stored in*], [*Lives at runtime in*],
  [`.isr_vector`], [Initial stack pointer and exception vectors], [Flash], [Flash (or relocated to SRAM via `VTOR`)],
  [`.text`], [Machine code], [Flash], [Flash (or SRAM / ITCM if copied)],
  [`.rodata`], [Constants, string literals, `const` tables], [Flash], [Flash],
  [`.data`], [Initialised mutable globals], [Flash (initial values)], [SRAM],
  [`.bss`], [Zero-initialised mutable globals], [Nowhere (size only)], [SRAM],
  [`.noinit`], [Globals that must survive a warm reset], [Nowhere], [SRAM, untouched by startup],
  [`.init_array`], [Pointers to C++ static constructors], [Flash], [Flash],
  [`.ARM.exidx`, `.ARM.extab`], [Exception unwinding tables], [Flash], [Flash],
)

The key asymmetry is `.data`. Its initial values must be in non-volatile memory, because SRAM is garbage at power-on, but the program writes to it, so it must run from SRAM. Each output section therefore has two addresses.

== VMA and LMA

- The *virtual memory address* ($"VMA"$) is where the section is when the program uses it. All code that refers to the variable is linked against this address.
- The *load memory address* ($"LMA"$) is where the section's bytes are stored in the image that gets programmed into flash.

For `.text` and `.rodata` the two coincide. For `.data` the $"VMA"$ is in SRAM and the $"LMA"$ is in flash, and the startup code copies $"LMA"$ to $"VMA"$ before anything reads the variables. In GNU ld syntax this is `> SRAM AT > FLASH`. The same mechanism places functions in RAM: put them in a section with an SRAM $"VMA"$ and a flash $"LMA"$, and copy them at startup. This is how flash-programming routines run while the flash is busy, and how hot loops escape flash wait states.

`.bss` needs only a $"VMA"$ and a size. It occupies no space in the image, and startup code fills it with zeros. Marking a section `(NOLOAD)` tells the linker that the image contains no bytes for it.

== A Complete Linker Script

The following script targets a Cortex-M4 with 1 MB flash at `0x0800_0000` and 128 KB SRAM at `0x2000_0000`, the layout of the STM32F4 family. It is complete enough to link a C++ program with static constructors and newlib.

```
ENTRY(reset_handler)

MEMORY
{
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
  SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}

STACK_SIZE = 8K;

SECTIONS
{
  .isr_vector : ALIGN(512)
  {
    KEEP(*(.isr_vector))            /* KEEP: survives --gc-sections */
  } > FLASH

  .text :
  {
    *(.text .text.*)
    *(.rodata .rodata.*)
    KEEP(*(.init)) KEEP(*(.fini))   /* crti/crtn fragments from libgcc */
  } > FLASH

  .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > FLASH
  .ARM.exidx :
  {
    __exidx_start = .;
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
    __exidx_end = .;
  } > FLASH

  .preinit_array :
  {
    PROVIDE_HIDDEN(__preinit_array_start = .);
    KEEP(*(.preinit_array*))
    PROVIDE_HIDDEN(__preinit_array_end = .);
  } > FLASH

  .init_array :
  {
    PROVIDE_HIDDEN(__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))    /* init_priority order first */
    KEEP(*(.init_array*))
    PROVIDE_HIDDEN(__init_array_end = .);
  } > FLASH

  /* Stack first: an overflow runs off the bottom of SRAM
     instead of silently into .data/.bss. */
  .stack (NOLOAD) : ALIGN(8)
  {
    . += STACK_SIZE;
    _estack = .;
  } > SRAM

  .data : ALIGN(4)
  {
    _sdata = .;
    *(.data .data.*)
    *(.ramfunc .ramfunc.*)          /* functions copied to SRAM */
    . = ALIGN(4);
    _edata = .;
  } > SRAM AT > FLASH
  _sidata = LOADADDR(.data);

  .bss (NOLOAD) : ALIGN(4)
  {
    _sbss = .;
    *(.bss .bss.*)
    *(COMMON)
    . = ALIGN(4);
    _ebss = .;
  } > SRAM

  .noinit (NOLOAD) : ALIGN(4)
  {
    *(.noinit .noinit.*)
  } > SRAM

  .heap (NOLOAD) : ALIGN(8)
  {
    _sheap = .;
    . = ORIGIN(SRAM) + LENGTH(SRAM);
    _eheap = .;
  } > SRAM

  /DISCARD/ : { *(.comment) *(.ARM.attributes) }
}
```

Points worth noticing:

- *`ENTRY`* does not affect the hardware, which always starts from the vector table. It tells the linker which symbol is a root for `--gc-sections` and records the entry point in the ELF for debuggers.
- *`KEEP`* is essential for the vector table and constructor arrays. Nothing calls them by name, so garbage collection would delete them and the chip would boot into nothing.
- *The vector table alignment* must satisfy `VTOR`, which requires alignment to the table size rounded up to a power of two (128 bytes, 32 entries, minimum on Armv7-M). 512 bytes covers up to 112 external interrupts.
- *Placing the stack first* in SRAM is a cheap overflow detector. A full descending stack that overflows writes below `0x2000_0000`. On many parts that address is unmapped and the push raises a fault at once. Check the memory map of your part, because some (STM32F4 among them) put system memory or other regions just below. Where that trick fails, use an $"MPU"$ guard region or, on Armv8-M, the `MSPLIM` and `PSPLIM` stack-limit registers.
- *The heap* takes whatever is left. With an $"RTOS"$ the heap usually belongs to the kernel allocator, and the C library `malloc` (newlib's `_sbrk`) should be either wired to the same region or made to fail loudly.

Linker scripts also enforce size budgets. A line such as `ASSERT(_ebss - ORIGIN(SRAM) <= 96K, "SRAM budget exceeded")` turns a slow creep in RAM use into a build error.

== Startup Code in C++

The reset handler runs with no initialised data, no zeroed `.bss`, no constructed objects, and often on the reset clock (an internal RC oscillator at 4 to 16 MHz). It must do six things, in order.

```cpp
#include <cstdint>

extern "C" {
// Symbols defined by the linker script.
extern std::uint32_t _sidata[], _sdata[], _edata[], _sbss[], _ebss[];
extern void (*__preinit_array_start[])();
extern void (*__preinit_array_end[])();
extern void (*__init_array_start[])();
extern void (*__init_array_end[])();

int main();
void system_clock_init();    // PLL, flash wait states, bus prescalers
[[noreturn]] void reset_handler();
}

namespace {
inline volatile std::uint32_t& scb_cpacr =
    *reinterpret_cast<volatile std::uint32_t*>(0xE000'ED88u);
}

[[noreturn]] void reset_handler() {
    // 1. Enable the FPU (CP10 and CP11 full access) before any code that
    //    might use FP registers. With -mfloat-abi=hard the compiler may
    //    use them even in "integer" code, e.g. for block copies.
    scb_cpacr = scb_cpacr | (0xFu << 20);
    asm volatile("dsb\n\tisb" ::: "memory");

    // 2. Copy .data from flash (LMA) to SRAM (VMA).
    const std::uint32_t* src = _sidata;
    for (std::uint32_t* dst = _sdata; dst < _edata; ++dst, ++src) {
        *dst = *src;
    }

    // 3. Zero .bss.
    for (std::uint32_t* dst = _sbss; dst < _ebss; ++dst) {
        *dst = 0;
    }

    // 4. Raise the clock. Doing this before step 2 speeds up the copy,
    //    but then system_clock_init must not touch .data or .bss.
    system_clock_init();

    // 5. Run C++ static constructors and __attribute__((constructor)).
    for (auto f = __preinit_array_start; f < __preinit_array_end; ++f) (*f)();
    for (auto f = __init_array_start; f < __init_array_end; ++f) (*f)();

    // 6. Enter the application. Returning from main has no meaning here.
    main();
    for (;;) {
        asm volatile("wfi");
    }
}
```

Three subtleties:

- *The compiler may replace the loops with calls to `memcpy` and `memset`.* That is fine if those library routines do not themselves depend on initialised data, which holds for newlib and picolibc. Compile the startup file with `-fno-tree-loop-distribute-patterns` if you need certainty.
- *Floating point before the FPU is enabled* raises a UsageFault (NOCP). A vendor `SystemInit` that runs before the FPU is enabled must be integer-only.
- *Static constructors run with interrupts enabled but no $"RTOS"$ running.* A constructor that calls a kernel API, or depends on another translation unit's object being constructed first (the *static initialisation order fiasco*), fails in ways that look like random boot hangs.

=== Making Initialisation Free

The best static constructor is one that does not exist. C++20's `constinit` requires a variable to be initialised at compile time, which places it in `.data` or `.bss` with no entry in `.init_array`:

```cpp
struct ring_buffer {
    std::uint8_t data[256]{};
    std::uint16_t head = 0;
    std::uint16_t tail = 0;
    constexpr ring_buffer() = default;
};

constinit ring_buffer g_uart_rx;   // zero-initialised in .bss, no constructor call
```

If the type later gains a non-`constexpr` constructor, the build fails instead of silently adding startup work. Function-local statics are another trap. By default GCC guards their initialisation with `__cxa_guard_acquire`, which pulls in locking code and, under an $"RTOS"$, needs a working mutex. Either avoid function-local statics with dynamic initialisation, or compile with `-fno-threadsafe-statics` and make sure the first call happens before the scheduler starts.

== Memory That Survives Reset

A section marked `(NOLOAD)` and skipped by startup keeps its contents across a warm reset (watchdog, software reset, fault handler reset), because SRAM keeps its contents as long as power stays on. This is the standard place for a crash record:

```cpp
struct crash_record {
    std::uint32_t magic;      // 0xDEADC0DE when valid
    std::uint32_t pc;
    std::uint32_t lr;
    std::uint32_t cfsr;       // Configurable Fault Status Register
    std::uint32_t reset_count;
};

[[gnu::section(".noinit")]] crash_record g_crash;
```

The fault handler fills it and resets. The next boot checks `magic`, reports the record over telemetry, and clears it. After a cold power-on the contents are random, so the magic number (ideally plus a CRC) is what distinguishes a real record from noise.

== Running Code from RAM

Executing from flash costs wait states. An STM32F4 at 168 MHz needs 5 wait states, hidden most of the time by the ART prefetch accelerator but not on branchy code with poor locality. A Cortex-M7 can place code in ITCM with single-cycle fetch. Mark the function and let the `.data` copy loop move it:

```cpp
[[gnu::section(".ramfunc"), gnu::noinline, gnu::long_call]]
void flash_program_word(std::uint32_t addr, std::uint32_t value);
```

`long_call` matters. A Thumb `BL` reaches only plus or minus 16 MB, and flash at `0x0800_0000` and SRAM at `0x2000_0000` are much further apart, so the linker must use a register-indirect call or insert a veneer. Code that programs the same flash bank it executes from must run from RAM, because on most parts the bank stalls or faults on reads while an erase or program is in progress.

== Reading the Map File

Pass `-Wl,-Map=firmware.map,--cref` and the linker writes where every input section went. Three questions the map file answers:

- *What is using flash?* Sort `.text` input sections by size. Common surprises are `printf` with floating-point support (10 to 20 KB on newlib), `std::iostream` (over 100 KB), exception-unwinding tables and unused vendor HAL modules.
- *Why is this symbol linked in?* The cross-reference table lists who references each symbol. Following the chain from a surprising library function usually ends at one stray call to `sprintf("%f")` or `std::to_string`.
- *Where did my buffer end up?* A DMA buffer must be in memory the DMA controller can reach. On an STM32H7, DTCM is not reachable from most DMA masters, and placing a buffer there produces a transfer that silently reads zeros.

Command-line companions: `arm-none-eabi-size -A firmware.elf` for per-section totals, `arm-none-eabi-nm --size-sort -C -S firmware.elf | tail` for the largest symbols, and Bloaty for a breakdown by compile unit.

== Shrinking the Image

#table(
  columns: 3,
  [*Technique*], [*Typical saving*], [*Cost*],
  [`-Os` (or `-Oz` on Clang) instead of `-O2`], [10 to 30 percent of `.text`], [Some speed; use `-O2` per hot file],
  [`-ffunction-sections -fdata-sections` plus `-Wl,--gc-sections`], [Often large on vendor HALs], [None],
  [Link-time optimisation (`-flto`)], [5 to 15 percent], [Longer builds; harder debugging],
  [newlib-nano (`--specs=nano.specs`)], [Several KB of libc], [No float `printf` unless `-u _printf_float`],
  [`-fno-exceptions -fno-rtti`], [Unwind tables, type info], [No exceptions or `dynamic_cast`],
  [Avoid `iostream`, `std::function`, `std::string`], [Tens of KB], [Use fixed-buffer formatting],
  [Thumb-2 only (default on Cortex-M)], [N/A, already compact], [None],
)

Measure before and after each change. The map file, not intuition, says where the bytes are.

== Execute-in-Place from External Flash

Parts without large internal flash (NXP i.MX RT, ESP32, RP2040) run code directly from external QSPI or OctoSPI NOR flash through a memory-mapped controller with a small cache. The linker script places `.text` at the controller's mapped window (for example `0x6000_0000` on i.MX RT), and a boot ROM reads a configuration block from the first sectors of the flash to set up the interface before jumping. Execute-in-place makes cache misses expensive (tens to hundreds of cycles for a line fill over a 4-bit bus), so latency-critical code and interrupt handlers go to on-chip RAM through the same `.ramfunc` mechanism. Timing analysis of execute-in-place code must model that cache, as #xref("embedded-and-realtime", "timing-analysis", label: "Timing Analysis and WCET") describes.

== Further Reading

Free Software Foundation (2024). "Using ld, the GNU Linker." Sourceware documentation, chapter 3 (Linker Scripts).

Arm (2021). "Armv7-M Architecture Reference Manual." DDI 0403E.

Arm (2024). "Procedure Call Standard for the Arm Architecture (AAPCS32)." IHI 0042.

Levine, J. (1999). "Linkers and Loaders." Morgan Kaufmann.

ISO/IEC (2020). "ISO/IEC 14882:2020 Programming Languages: C++." Standard (constinit, static initialisation).

STMicroelectronics (2024). "RM0090: STM32F4xx Reference Manual."

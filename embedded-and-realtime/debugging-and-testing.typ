#import "../template.typ": xref

= Debugging and Testing Embedded Software <debugging-and-testing>

Embedded bugs hide in places a desktop debugger never looks: in the timing between an interrupt and a task, in a peripheral register written in the wrong order, in a stack that overflows only when two interrupts nest, in a board that works at room temperature and fails in a freezer. The tools are correspondingly specialised. This chapter covers the on-chip debug and trace hardware of Cortex-M parts, logging that does not disturb timing, diagnosing hard faults from the stacked exception frame, external instruments, and a layered testing strategy that runs most tests on a laptop and the rest on real hardware.

*See also:* #xref("embedded-and-realtime", "mcus-and-soc", label: "Microcontrollers and Systems-on-Chip"), #xref("embedded-and-realtime", "embedded-cpp", label: "C++ for Embedded Systems"), #xref("embedded-and-realtime", "timing-analysis", label: "Timing Analysis and WCET"), #xref("linux-kernel", "kernel-tracing", label: "Kernel Tracing") (Linux counterpart)

== The Debug Port

Arm's CoreSight architecture gives every Cortex-M a *Debug Access Port* (DAP) that an external probe reaches through one of two physical interfaces:

- *JTAG* (IEEE 1149.1): four or five signals (TCK, TMS, TDI, TDO, optional TRST). Also used for boundary scan, which tests board solder joints in production.
- *Serial Wire Debug* (SWD): two signals (SWDIO, SWCLK), the usual choice on $"MCU"$s because it frees pins. An optional third pin, *SWO*, carries trace output.

The probe (SEGGER J-Link, ST-LINK, or any CMSIS-DAP probe) talks to host software (OpenOCD, pyOCD, probe-rs, or vendor tools) that exposes a GDB server. Through the DAP the host can halt and step the core, read and write memory *while the core runs* (no instrumentation required), and program flash.

The hardware debug resources are finite:

- *Breakpoints*: the Flash Patch and Breakpoint unit provides a small number of hardware comparators (6 on Cortex-M3 and M4, 8 on M7, 4 or fewer on M0+). Code in flash cannot take software breakpoints, so this number limits how many breakpoints GDB can set in flash at once.
- *Watchpoints*: the DWT has up to 4 comparators (2 on M0+) that halt on a read or write of an address range. A watchpoint on a corrupted variable usually finds the culprit in minutes where reading code takes days.
- *Vector catch*: halt on entry to HardFault or other exceptions before the handler runs, preserving the faulting state.

Halting the core is intrusive. Peripherals keep running while the core is stopped: timers overflow, $"UART"$s overrun, a motor controller's PWM output may freeze in a dangerous state. Most $"MCU"$s have debug-freeze bits (on STM32, the `DBGMCU` registers) that stop chosen timers and watchdogs while halted. For a running physical system, prefer the non-halting tools below.

== Logging Without Wrecking Timing

#table(
  columns: 4,
  [*Mechanism*], [*Pins*], [*Cost per message*], [*Notes*],
  [`printf` to $"UART"$], [1], [About 87 µs per byte at 115 200 baud if blocking], [Simple; blocking output distorts timing badly],
  [Semihosting], [Debug port], [Milliseconds: the core halts on `BKPT 0xAB` while the host services the call], [Only for test harnesses; hangs without a debugger attached],
  [ITM over SWO], [1 (SWO)], [A few cycles per 32-bit write to a stimulus port], [Hardware FIFO; drops data if the host cannot keep up],
  [SEGGER RTT], [None beyond SWD], [Tens of cycles: copy into a RAM ring buffer], [The probe reads the buffer in the background through the DAP],
  [Deferred binary logging], [Any], [Store format-string id plus arguments], [Formatting happens on the host; tiny and fast],
)

The most efficient designs combine the last two ideas. Firmware writes only a log identifier and raw argument values into a RAM buffer, and a host tool expands them using the format strings extracted from the ELF file. Zephyr's dictionary logging, Memfault's compact logs, Rust's `defmt` and Trice all work this way. A message costs a few dozen cycles and a handful of bytes, which makes it acceptable even inside interrupt handlers.

== Diagnosing a HardFault

When a Cortex-M faults, the hardware has already saved the evidence. On exception entry it pushes `R0` to `R3`, `R12`, `LR`, `PC` and `xPSR` to the active stack, and the System Control Block's fault status registers record why. A good fault handler collects that evidence before anything else disturbs it:

```cpp
#include <cstdint>

struct exception_frame {
    std::uint32_t r0, r1, r2, r3, r12, lr, pc, xpsr;
};

namespace scb {
inline volatile std::uint32_t& cfsr  = *reinterpret_cast<volatile std::uint32_t*>(0xE000'ED28u);
inline volatile std::uint32_t& hfsr  = *reinterpret_cast<volatile std::uint32_t*>(0xE000'ED2Cu);
inline volatile std::uint32_t& mmfar = *reinterpret_cast<volatile std::uint32_t*>(0xE000'ED34u);
inline volatile std::uint32_t& bfar  = *reinterpret_cast<volatile std::uint32_t*>(0xE000'ED38u);
inline volatile std::uint32_t& aircr = *reinterpret_cast<volatile std::uint32_t*>(0xE000'ED0Cu);
}

struct fault_record {
    std::uint32_t magic;
    std::uint32_t pc, lr, xpsr;
    std::uint32_t cfsr, hfsr, fault_address;
};

[[gnu::section(".noinit")]] fault_record g_fault;   // survives the reset below

extern "C" [[noreturn, gnu::used]] void hard_fault_report(const exception_frame* f) {
    const std::uint32_t cfsr = scb::cfsr;
    g_fault = {
        .magic = 0xFA17'C0DEu,
        .pc = f->pc,              // the faulting instruction (or the next one)
        .lr = f->lr,              // the caller, a hint at the call chain
        .xpsr = f->xpsr,
        .cfsr = cfsr,
        .hfsr = scb::hfsr,
        // MMFARVALID (bit 7) and BFARVALID (bit 15) say whether the
        // address registers hold the faulting data address.
        .fault_address = (cfsr & (1u << 7))  ? scb::mmfar
                       : (cfsr & (1u << 15)) ? scb::bfar
                       : 0u,
    };
    asm volatile("bkpt #0");                   // halts if a debugger is attached
    asm volatile("dsb" ::: "memory");
    scb::aircr = (0x05FAu << 16) | (1u << 2);  // VECTKEY | SYSRESETREQ
    for (;;) {}
}

// Find which stack holds the frame: EXC_RETURN bit 2 is 0 for MSP, 1 for PSP.
extern "C" [[gnu::naked]] void HardFault_Handler() {
    asm volatile(
        "tst   lr, #4            \n"
        "ite   eq                \n"
        "mrseq r0, msp           \n"
        "mrsne r0, psp           \n"
        "b     hard_fault_report \n");
}
```

On a production device without a debugger, `bkpt` itself causes a fault escalation, so real code checks `DHCSR.C_DEBUGEN` first or omits it. After reset, the boot code finds the record (see the `.noinit` technique in #xref("embedded-and-realtime", "linker-scripts-and-startup", label: "Linker Scripts, Startup Code, and the Memory Map")) and reports it.

Reading the fault status register:

#table(
  columns: 3,
  [*CFSR bit*], [*Meaning*], [*Usual cause*],
  [`IACCVIOL` / `DACCVIOL` (MemManage)], [MPU violation on fetch or data access], [Stack overflow into a guard region; null or wild pointer with the MPU on],
  [`PRECISERR` (BusFault)], [Bus error, `BFAR` valid, `PC` is the faulting instruction], [Access to an unclocked peripheral or unmapped address],
  [`IMPRECISERR` (BusFault)], [Bus error reported later than the store that caused it], [Buffered write to a bad address; set `DISDEFWBUF` in `ACTLR` while debugging to make it precise],
  [`UNDEFINSTR` (UsageFault)], [Undefined instruction], [Jump to data, corrupted function pointer],
  [`INVSTATE` (UsageFault)], [Attempt to enter Arm state], [Function pointer with bit 0 clear; jump to a corrupted return address],
  [`NOCP` (UsageFault)], [Coprocessor access while disabled], [FPU instruction before CPACR is set],
  [`UNALIGNED`, `DIVBYZERO` (UsageFault)], [Only if trapping is enabled in `CCR`], [Packed-struct access; integer division by zero],
  [`STKERR` / `MSTKERR`], [Fault while stacking the exception frame], [The stack pointer itself is invalid: stack overflow],
)

The stacked `PC` plus `addr2line -e firmware.elf` usually points at the exact line. If the stack itself was corrupted, the frame may be garbage, and the most reliable evidence is the `STKERR` bit and the stack pointer's value relative to the stack bounds.

== Trace: Seeing What Actually Ran

Breakpoints and logs show what the code chose to tell you. Trace shows what actually executed:

- *ETM* (Embedded Trace Macrocell), on Cortex-M3, M4, M7, M33 and others, emits a compressed stream of every branch taken, through a 4-bit parallel trace port to a probe such as J-Trace, or into an on-chip buffer (ETB). With it a tool can reconstruct the full instruction history leading up to a fault, measure code coverage on the target, and profile every function without instrumentation.
- *MTB* (Micro Trace Buffer), on Cortex-M0+ and M23, records branches into a small SRAM buffer. It cannot stream, but it can show the last few hundred branches before a fault, which is often exactly what is needed.
- *$"RTOS"$-aware tracing* (Percepio Tracealyzer, SEGGER SystemView) instruments the kernel to record task switches, ISR entry and exit, and IPC events with timestamps. The resulting timeline makes priority inversion, starvation, and jitter visible at a glance, and gives measured response times to compare against the analysis in #xref("embedded-and-realtime", "timing-analysis", label: "Timing Analysis and WCET").

== External Instruments

Some questions are answered only from outside the chip:

- A *logic analyser* with protocol decoders (Saleae, or sigrok-supported hardware with PulseView) shows what actually went over SPI, I2C, $"UART"$ or CAN, with timing. When a driver and a sensor disagree, the wire is the arbiter.
- An *oscilloscope* shows analogue truth: ringing on a fast edge, a supply rail dipping during a radio transmission, a reset line that rises too slowly.
- *GPIO markers*: setting a spare pin high at ISR entry and low at exit, captured alongside the bus signals, gives sub-microsecond timing of software with almost no overhead.

== A Testing Strategy in Layers

Hardware is scarce, slow to flash and hard to automate. A good test strategy pushes as much testing as possible onto the host and reserves hardware for what only hardware can show.

=== Host Unit Tests

Most firmware logic (protocol parsers, state machines, control algorithms, data structures, fixed-point maths) has nothing to do with hardware. Built for the development machine, it can be tested with GoogleTest, Catch2, CppUTest or Unity, run in milliseconds, under AddressSanitizer and UndefinedBehaviorSanitizer, which do not exist on the $"MCU"$. Two caveats: the host has a different word size and possibly different `int` promotions and alignment rules, so compile host tests with the same `-std` and warning flags, use fixed-width types, and keep a smaller set of tests that also run on target.

The design requirement is a seam between logic and hardware. In C++ that seam can be a template parameter, which costs nothing in the firmware build:

```cpp
#include <cstdint>
#include <optional>

// Production policy: real registers.
struct stm32_adc {
    static void start_conversion();
    static bool conversion_done();
    static std::uint16_t read_result();
};

template <typename Adc>
class battery_monitor {
public:
    // Returns millivolts, or nothing if the ADC did not finish in time.
    std::optional<std::uint32_t> sample_mv() {
        Adc::start_conversion();
        for (int i = 0; i < max_polls; ++i) {
            if (Adc::conversion_done()) {
                const std::uint32_t raw = Adc::read_result();       // 12-bit
                return raw * vref_mv * divider_ratio / 4095u;
            }
        }
        return std::nullopt;
    }

private:
    static constexpr int max_polls = 1000;
    static constexpr std::uint32_t vref_mv = 3300;
    static constexpr std::uint32_t divider_ratio = 2;
};

// ---- host test (GoogleTest) ----
struct fake_adc {
    static inline std::uint16_t next_value = 0;
    static inline bool never_finishes = false;
    static void start_conversion() {}
    static bool conversion_done() { return !never_finishes; }
    static std::uint16_t read_result() { return next_value; }
};

TEST(battery_monitor, converts_full_scale) {
    fake_adc::next_value = 4095;
    fake_adc::never_finishes = false;
    battery_monitor<fake_adc> m;
    EXPECT_EQ(m.sample_mv(), 6600u);
}

TEST(battery_monitor, times_out_when_adc_hangs) {
    fake_adc::never_finishes = true;
    battery_monitor<fake_adc> m;
    EXPECT_FALSE(m.sample_mv().has_value());
}
```

The timeout test is the kind that matters most and is hardest to provoke on real hardware: what happens when a peripheral never answers.

=== Fuzzing

Anything that parses external input (radio packets, CAN frames, configuration files, update manifests) should be fuzzed. Built on the host with libFuzzer or AFL++, a parser receives millions of mutated inputs per minute, and the sanitizers turn silent memory corruption into immediate crashes. Most remote vulnerabilities in connected devices are parser bugs that an afternoon of fuzzing would have found.

=== Simulation and Emulation

- *QEMU* emulates several Cortex-M boards well enough to run an $"RTOS"$ and test code that touches only the core and simple peripherals.
- *Renode* (Antmicro) models complete boards and multi-node systems, including peripherals and wireless links, deterministically, so a test of 10 networked sensor nodes runs in CI and fails the same way every time.
- *Instruction-set simulators* are not cycle-accurate on complex cores, so they are no substitute for timing measurement on target.

=== Hardware-in-the-Loop

The final layer runs the real firmware on the real board, with the outside world simulated. A HIL rig drives the device's inputs (sensor signals, bus traffic, supply voltage) from a test computer and checks its outputs, from simple setups with a debug probe, a USB relay board and a logic analyser up to full real-time plant simulators for automotive ECUs. HIL tests catch what host tests cannot: wrong register sequences, DMA and cache coherence errors, real interrupt timing, brown-out behaviour, and the effect of the actual compiler on the actual chip. Running a small HIL suite on every merge, with a self-hosted CI runner attached to a board farm, catches integration problems while the change is still fresh.

=== Static Analysis

Compiler warnings at their strictest, clang-tidy, and cppcheck catch many defects before anything runs. Safety-critical projects add a MISRA checker (Polyspace, Helix QAC, PC-lint Plus, Parasoft) and sometimes sound abstract-interpretation tools (Astrée, Polyspace Code Prover) that prove the absence of run-time errors such as overflow, division by zero and out-of-bounds access over all inputs.

== Debugging in the Field

Devices in the field cannot be attached to a debugger, so the firmware must collect its own evidence:

- Keep fault records and the last few hundred log entries in retained RAM, and upload them on the next connection.
- Count resets by cause (power-on, brown-out, watchdog, software, fault), which the reset controller reports in a status register. A rising watchdog-reset count across a fleet is often the first sign of a bug.
- Include the firmware's build identifier (the ELF's GNU build ID) in every report, so the server can symbolise addresses against the exact binary.
- Ship assertions in production builds, with a handler that records file, line and `PC` and then resets into a safe state. A silent wrong answer is worse than a logged reset.

== Further Reading

Arm (2021). "Armv7-M Architecture Reference Manual." DDI 0403E (debug architecture, fault status registers).

Arm (2017). "CoreSight Architecture Specification v3.0." IHI 0029.

Yiu, J. (2013). "The Definitive Guide to ARM Cortex-M3 and Cortex-M4 Processors." Newnes, 3rd ed.

Grenning, J. (2011). "Test Driven Development for Embedded C." Pragmatic Bookshelf.

IEEE (2013). "IEEE 1149.1-2013: Standard for Test Access Port and Boundary-Scan Architecture." Standard.

Arm Keil. "Using Cortex-M3/M4/M7 Fault Exceptions." Application Note 209, Keil documentation.

Antmicro (2024). "Renode Documentation." renode.io reference.

Serebryany, K. (2016). "Continuous Fuzzing with libFuzzer and AddressSanitizer." IEEE Cybersecurity Development (SecDev).

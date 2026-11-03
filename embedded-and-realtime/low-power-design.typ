#import "../template.typ": xref

= Low-Power Design <low-power-design>

A battery-powered sensor that must run for five years on a coin cell has an average current budget of a few microamps. That is less than the current a typical $"MCU"$ draws in one millisecond of active execution at full speed, spread over a whole second. Low-power design is therefore mostly about *not running*: spending almost all the time in the deepest sleep state the application can tolerate, waking briefly to do work, and making sure that nothing (a floating pin, a forgotten peripheral clock, a regulator's quiescent current) leaks while asleep. This chapter covers the energy budget, the physics, the sleep machinery of a modern $"MCU"$, the software patterns that use it, and how to measure currents that span six orders of magnitude.

*See also:* #xref("embedded-and-realtime", "mcus-and-soc", label: "Microcontrollers and Systems-on-Chip"), #xref("embedded-and-realtime", "rtos", label: "Real-Time Operating Systems"), #xref("cpu-architecture", "power-and-dvfs", label: "Power and DVFS") (architecture), #xref("networking", "wireless-protocols", label: "Wireless Protocols") (radios)

== The Energy Budget

Battery life is capacity divided by average current. The average current of a duty-cycled device is a weighted sum over its states:

$ I_"avg" = sum_s I_s dot (t_s / T) $

where state $s$ draws $I_s$ for $t_s$ out of every period $T$. Take a temperature sensor on a CR2032 cell (about 225 mAh nominal) that wakes every 10 s:

#table(
  columns: 4,
  [*State*], [*Current*], [*Time per 10 s*], [*Average contribution*],
  [Deep sleep with RTC], [2 µA], [about 10 s], [2.0 µA],
  [Active: wake, sample, compute], [5 mA], [10 ms], [5.0 µA],
  [Radio transmit], [8 mA], [3 ms], [2.4 µA],
  [*Total*], [], [], [*9.4 µA*],
)

$225 "mAh" \/ 9.4 "µA" approx 24 000$ hours, about 2.7 years. The table shows where to look next. Halving the active time gains more than halving the sleep current. Sending every 60 s instead of every 10 s cuts the radio term sixfold.

Real batteries spoil the arithmetic in three ways, so production designs derate by 20 to 40 percent:

- *Self-discharge*: lithium primary cells lose about 1 percent per year, alkaline cells more.
- *Pulse capability*: a CR2032's internal resistance (tens of ohms, rising as it discharges and in the cold) means a 10 to 15 mA radio pulse can pull its voltage below the $"MCU"$'s brown-out threshold long before the capacity is used. A capacitor across the cell supplies the pulse instead.
- *Cutoff voltage*: usable capacity ends when the voltage under load drops below the minimum the circuit needs, not when the chemistry is exhausted.

== Where the Power Goes

CMOS power has two parts:

$ P = underbrace(alpha C V^2 f, "dynamic") + underbrace(V I_"leak", "static") $

*Dynamic power* is spent charging and discharging capacitance on every clock edge. It scales linearly with frequency and quadratically with voltage, which is why voltage scaling matters more than clock scaling. *Static power* is leakage through transistors that are nominally off. It flows whenever the logic is powered, regardless of clock, and grows steeply with temperature. Sleep modes attack each term differently: stopping the clock removes dynamic power, and removing the supply from a domain removes its leakage.

That leads to the *race to sleep* question: is it better to run fast and sleep sooner, or run slowly? If the sleep current is very low and fixed costs (oscillator start-up, regulator settling) dominate, racing at full speed usually wins, because the active-mode overhead current of flash, regulators and clocks is paid for less time. If the part can lower its core voltage at lower frequency (STM32L4's voltage range 2, for example), running slowly at low voltage can win for long computations. Measure both on the target, because the answer depends on the part.

== Sleep Modes on a Modern MCU

The Cortex-M core provides two instructions and two control bits. `WFI` (wait for interrupt) and `WFE` (wait for event) stop the core clock until an interrupt or event arrives. In the System Control Register, `SLEEPDEEP` selects whether the vendor's deep-sleep logic runs (stopping more clocks, switching regulators), and `SLEEPONEXIT` returns the core to sleep automatically after an ISR finishes, without executing any thread-mode code. A device whose work happens entirely in interrupt handlers can sleep between them with no main loop at all.

Vendors build a ladder of modes on top. The STM32L4 family is representative; exact figures vary by part and temperature:

#table(
  columns: 4,
  [*Mode*], [*What stops*], [*Typical current*], [*Wake time and state*],
  [Run (80 MHz)], [Nothing], [about 100 µA/MHz], [N/A],
  [Sleep], [CPU clock], [A few mA at 80 MHz], [Cycles; everything retained],
  [Low-power run / sleep], [Clock limited to 2 MHz, low-power regulator], [Tens to hundreds of µA], [Microseconds],
  [Stop 2], [All high-speed clocks; most of the core domain in retention], [About 1 to 2 µA with RTC], [Microseconds; SRAM and registers kept],
  [Standby], [Core domain off], [Hundreds of nA with RTC], [Tens of µs plus reboot; only backup registers and optionally SRAM2 kept],
  [Shutdown], [Everything but the backup domain], [Tens of nA], [Reboot from reset],
)

The trade-off down the ladder is wake latency and lost state. Waking from Stop takes microseconds and execution resumes after the `WFI`. Waking from Standby or Shutdown is a reset: the program starts again from the vector table, and it must check a status flag to tell a wake-up from a power-on and restore its state from backup registers or retained RAM. For a real-time design, the wake latency of the deepest mode used is part of the interrupt latency budget (#xref("embedded-and-realtime", "timing-analysis", label: "Timing Analysis and WCET")).

Only some peripherals can wake the device from each mode. In Stop modes the typical wake sources are the RTC, a low-power timer clocked from the 32.768 kHz crystal, EXTI pin interrupts, a low-power $"UART"$ that can recognise a start bit, and comparators. Choosing which sleep mode is available is therefore partly a choice of which peripheral handles the input.

=== Entering Deep Sleep Safely

```cpp
#include <cstdint>

namespace scb {
inline volatile std::uint32_t& scr =
    *reinterpret_cast<volatile std::uint32_t*>(0xE000'ED10u);
constexpr std::uint32_t sleepdeep = 1u << 2;
}

// Vendor-specific parts, provided by the board support layer.
void rtc_set_wakeup(std::uint32_t seconds);
void select_stop2_mode();          // PWR_CR1.LPMS = Stop 2
void clocks_restore_after_stop();  // re-enable HSE and PLL; Stop falls back to MSI
bool work_pending();

void sleep_until_next_event(std::uint32_t max_seconds) {
    rtc_set_wakeup(max_seconds);
    select_stop2_mode();

    asm volatile("cpsid i" ::: "memory");   // close the race window
    if (!work_pending()) {
        scb::scr = scb::scr | scb::sleepdeep;
        asm volatile("dsb" ::: "memory");   // complete outstanding writes
        asm volatile("wfi");                // a pending IRQ still wakes the core
        scb::scr = scb::scr & ~scb::sleepdeep;
        clocks_restore_after_stop();
    }
    asm volatile("cpsie i" ::: "memory");   // the waking ISR runs here
}
```

The pattern around `WFI` matters. If interrupts were enabled, an ISR could set "work pending" between the check and the `WFI`, and the device would sleep with work queued until the next timer wake. With interrupts masked by `PRIMASK`, a pending interrupt still wakes the core from `WFI`; the handler then runs as soon as interrupts are re-enabled, after the clocks are restored. The `DSB` ensures that writes still sitting in the write buffer (for example to a peripheral that must be off) complete before the core stops.

== Tickless Idle in an RTOS

A periodic 1 kHz tick wakes the $"MCU"$ a thousand times a second, which on its own can dominate a microamp budget. Tickless idle (FreeRTOS `configUSE_TICKLESS_IDLE`, the default behaviour in Zephyr) replaces it: when the idle task runs, the kernel computes the time until the next timeout, programs a low-power timer for that instant, sleeps, and on wake advances the tick count by the time actually slept. FreeRTOS calls a port hook (`portSUPPRESS_TICKS_AND_SLEEP`) that applications override to use Stop mode instead of plain `WFI`, with a threshold (`configEXPECTED_IDLE_TIME_BEFORE_SLEEP`) so that short idle gaps are not worth the wake cost.

The wake-up timer must keep running in the chosen sleep mode, which usually means an RTC or LPTIM from the 32.768 kHz oscillator rather than SysTick, which stops with the core clock. Converting between the slow clock and the tick introduces rounding, and well-written ports carry the remainder forward so that the kernel's notion of time does not drift.

== Leakage Hunting: The Board, Not the Chip

When a device measures 50 µA in a mode the datasheet rates at 2 µA, the cause is almost always outside the $"MCU"$ core:

- *Floating inputs.* A CMOS input left floating near mid-rail turns on both transistors of its input stage and can draw tens to hundreds of microamps. Configure every unused pin as analog (which disables the input buffer) or tie it to a defined level.
- *Pull-ups fighting outputs.* A 10 kilohm pull-up on a line held low by a sleeping peripheral draws 330 µA at 3.3 V. Use larger values, or switch the pull-ups off in sleep.
- *Peripheral clocks left enabled.* Disable each peripheral clock after use. A running ADC, DAC or high-speed oscillator can cost more than the whole sleep budget.
- *Debug connection.* An attached debugger keeps the debug domain powered and can block deep sleep entirely. Measure with the probe disconnected, or with the debug-in-sleep bits cleared.
- *Regulator quiescent current.* An LDO with 50 µA of quiescent current wastes more than the whole sleeping system. Low-power LDOs and buck converters with sub-microamp quiescent current exist; the choice is made in schematic review, not in firmware.
- *External chips.* Sensors, flash memories and radios have their own sleep commands. A SPI NOR flash in standby draws tens of µA; in deep power-down, around 1 µA or less.

== Measuring Current

The difficulty is range. The same device draws 1 µA asleep and 10 mA while transmitting, with transitions in microseconds. A multimeter in series averages the two meaninglessly, and its burden voltage (the voltage dropped across its shunt) can brown out the device on a pulse.

- *Source-measure units and dedicated power analysers* (Nordic Power Profiler Kit II, Joulescope, Qoitech Otii) switch shunt ranges automatically, sample at hundreds of kHz or more, and integrate charge, giving both the waveform and the average.
- *Correlate with firmware*: toggle a GPIO at each state change and capture it on the same timeline, so each current feature can be tied to the code that caused it.
- *Measure the integral, not the peak*: the energy of one wake cycle (charge in microcoulombs) times the wake rate gives the average current directly, and is the number to optimise.

== The Radio Usually Dominates

In connected devices the radio draws the largest current and the protocol decides how long it is on. A few rules of thumb:

- *Send less often, send more at once.* Fixed costs per transmission (oscillator start-up, synchronisation, acknowledgement) favour batching.
- *Choose the protocol for the duty cycle.* Bluetooth Low Energy connection intervals, Thread sleepy end devices, LoRaWAN Class A, and LTE-M and NB-IoT power saving mode all exist to let the radio sleep. An LTE modem without power saving mode can draw milliamps idle while staying attached to the network.
- *Avoid listening.* Receive windows cost about as much as transmitting. Protocols in which the device initiates every exchange (LoRaWAN Class A) let it sleep the rest of the time.

== Energy Harvesting

Where batteries are impractical, devices harvest light, vibration, heat or radio energy into a supercapacitor or small rechargeable cell through a harvesting power-management IC with maximum power point tracking. Indoor light gives on the order of 10 µW per square centimetre of solar cell, so the budget is tens of microwatts on average. Software must become *energy-aware*: measure the stored energy, defer non-essential work when it is low, and checkpoint state to non-volatile memory (FRAM is popular because it writes at low energy) so that a power loss mid-task is not catastrophic. Intermittent computing research formalises this as programs that make forward progress across arbitrary power failures.

== Checklist

- Write the energy budget first, then check each line on the bench.
- Use the deepest sleep mode whose wake latency and retained state the application can accept.
- Make every wake as short as possible: batch work, race to sleep unless measurements say otherwise.
- Run the $"RTOS"$ tickless with a low-power timer.
- Put every unused pin in analog mode, every unused peripheral clock off, and every external chip in its lowest-power state.
- Measure with a real power analyser and the debugger detached.
- Derate the battery for temperature, pulse load, self-discharge and cutoff voltage.

== Further Reading

Arm (2021). "Armv7-M Architecture Reference Manual." DDI 0403E (sleep, SCR, WFI and WFE).

STMicroelectronics (2023). "AN4621: STM32L4 and STM32L4+ ultra-low-power features overview." Application note.

STMicroelectronics (2024). "RM0351: STM32L4x5 and STM32L4x6 Reference Manual."

FreeRTOS (2024). "Low Power Support: Tickless Idle Mode." FreeRTOS documentation.

Rabaey, J. (2009). "Low Power Design Essentials." Springer.

Lucia, B., Balaji, V., Colin, A., Maeng, K., Ruppel, E. (2017). "Intermittent Computing: Challenges and Opportunities." SNAPL.

Energizer (2023). "CR2032 Lithium Coin Product Datasheet." Manufacturer specification.

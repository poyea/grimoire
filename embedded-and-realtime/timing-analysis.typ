#import "../template.typ": xref

= Timing Analysis and WCET <timing-analysis>

A real-time guarantee is a chain of three claims: every task's worst-case execution time ($"WCET"$) is at most $C_i$, the schedule built from those $C_i$ meets every deadline, and the platform adds no delay the analysis forgot. #xref("operating-systems", "real-time-scheduling", label: "Real-Time Scheduling") develops the scheduling theory (rate monotonic, EDF, response-time analysis, priority inheritance). This chapter is about the other two links: how an embedded engineer obtains a defensible $C_i$ on real silicon, and how to fold interrupts, kernel overheads, jitter and shared resources into an analysis that still holds on the bench.

*See also:* #xref("operating-systems", "real-time-scheduling", label: "Real-Time Scheduling"), #xref("embedded-and-realtime", "rtos", label: "Real-Time Operating Systems"), #xref("embedded-and-realtime", "mcus-and-soc", label: "Microcontrollers and Systems-on-Chip"), #xref("cpu-architecture", "cache-hierarchy", label: "Cache Hierarchy") (architecture), #xref("performance-engineering", "benchmarking", label: "Benchmarking") (methodology)

== Why the Worst Case Is Hard

The execution time of a piece of code is not a number but a distribution over inputs and hardware states. Three quantities matter:

- *Best-case execution time* ($"BCET"$): the shortest run. It bounds output jitter and matters for control loops.
- *Worst-case execution time* ($"WCET"$): the longest possible run over all inputs and all initial hardware states.
- *Observed maximum*: the longest run you actually saw. It is always at most the true $"WCET"$, and the gap between the two is the risk you are carrying.

On an 8-bit AVR with no cache and fixed-latency instructions, the $"WCET"$ is the cycle count of the longest path through the control-flow graph, and you can compute it by hand. On a Cortex-M7 with a 6-stage dual-issue pipeline, branch prediction, I/D caches, flash with prefetch and wait states, and a DMA engine competing for the bus matrix, the execution time of a single load can range from 1 cycle (D-cache or DTCM hit) to dozens (cache miss to external SDRAM behind a busy AXI interconnect). The longest path in the code may not be the slowest path in time, because a shorter path can miss in the cache more often.

#table(
  columns: 3,
  [*Source of variation*], [*Typical magnitude*], [*Mitigation*],
  [Input-dependent paths (loops, branches)], [Unbounded without loop bounds], [Annotate loop bounds; avoid data-dependent iteration],
  [Flash wait states], [0 to 7+ cycles per fetch at high clock], [Prefetch buffer, flash accelerator, run hot code from SRAM or ITCM],
  [Instruction / data cache], [1 cycle hit, 10 to 100+ miss], [TCM for critical code, cache locking, analysable layout],
  [Branch prediction (M7, Cortex-R/A)], [A few cycles per mispredict], [Usually small; static analysis models it],
  [Bus contention (DMA, second core)], [Proportional to traffic], [Bandwidth regulation, separate bus slaves, TCM],
  [Interrupt and kernel preemption], [Whole ISR durations], [Model ISRs as tasks in the analysis (below)],
  [Multicore shared cache / DRAM], [2x to 10x slowdowns measured], [Partitioning, cache colouring, interference budgets],
)

=== Timing Anomalies

The intuition that "a cache miss is the worst case for that access" is false on processors with dynamic scheduling. Lundqvist and Stenström (1999) showed *timing anomalies*: a local speed-up (a cache hit) can cause a global slow-down, because it changes the order in which instructions compete for functional units or changes which line gets evicted. The consequence is severe: on such a processor you cannot compute a safe bound by assuming every access misses. A sound analysis must explore the hardware states, and a measurement must not assume the "cold cache" run is the slowest run. In-order cores without such effects (Cortex-M0+, M3, M4, most Cortex-R) are called *timing-compositional* and are far easier to analyse, which is one reason safety-critical designs still choose them.

== Three Ways to Get a WCET

=== Static Analysis

Static $"WCET"$ analysis computes a bound from the binary without running it. The standard pipeline, used by the commercial tool aiT (AbsInt) and the academic tools OTAWA and Heptane, is:

1. *Control-flow reconstruction* from the linked executable (not the source, because the compiler decides the real paths).
2. *Value analysis* by abstract interpretation, to find address ranges for loads and stores and to derive loop bounds where possible.
3. *Loop-bound and flow-fact annotation*. Anything the tool cannot infer, the engineer supplies, for example "this loop runs at most 64 times".
4. *Microarchitectural analysis*. Abstract cache states classify each access as *always hit*, *always miss*, *first miss* (persistence) or *not classified*. A pipeline model then gives a cycle bound for each basic block.
5. *Path analysis* by *implicit path enumeration* (IPET; Li and Malik, 1995). Each basic block gets an execution-count variable $x_b$ and a cost $c_b$. Flow-conservation constraints and loop bounds become linear constraints, and an integer linear program maximises

$ "WCET" = max sum_b c_b x_b quad "subject to flow and loop-bound constraints." $

Static analysis gives a *safe* bound (never below the true $"WCET"$) if the hardware model is correct. Its weaknesses are pessimism on complex cores, the need for an exact model of the specific chip, and the annotation effort. It is the method expected for DO-178C DAL A and ISO 26262 ASIL D on single-core processors.

=== Measurement-Based Analysis

Most industrial projects measure. Run the code on the target, record the maximum, and add a margin. Done well, this means:

- Measure on the *production* configuration: same compiler flags, same clock tree, same flash wait states, same memory placement.
- Drive inputs that exercise the long paths, guided by structural coverage (statement, branch, MC/DC).
- Measure with interference present: DMA running, other cores busy, caches flushed or polluted before each run.
- Record the *high-water mark* over long soak runs, not the average.

The weakness is that measurement only ever gives a lower bound on the true $"WCET"$. The margin (commonly 20 to 50 percent) is engineering judgement, not proof. *Measurement-based probabilistic timing analysis* (MBPTA; Cucu-Grosjean et al., 2012) tries to put a number on that judgement by fitting an extreme value distribution to the tail of the observed times and reading off a time whose exceedance probability is, say, $10^(-12)$ per run. Its validity depends on the runs being independent and identically distributed, which in turn needs hardware with randomised caches or careful input randomisation. That is a strong assumption, and certification authorities treat it cautiously.

=== Hybrid Analysis

Hybrid tools (RapiTime from Rapita Systems is the best known) instrument the code at basic-block boundaries, measure the execution time of each block on the target, and then combine the per-block maxima along the worst-case path using a static path analysis. This avoids building a processor model while still covering paths the tests never executed end to end. The bound is only as good as the per-block measurements, so the hardware-state problem remains, but the path-coverage problem goes away.

== Measuring Cycles on Cortex-M

Every Armv7-M and Armv8-M Mainline core (M3, M4, M7, M33, M55, M85) has a *Data Watchpoint and Trace* unit with a free-running 32-bit cycle counter, `CYCCNT`. It counts core clock cycles with zero overhead and is the standard instrument for on-target timing. The M0 and M0+ have no `CYCCNT`; use a free-running hardware timer instead.

```cpp
#include <cstdint>

namespace dwt {

// Architecturally defined addresses (Armv7-M ARM, Debug chapter).
inline volatile std::uint32_t& demcr  = *reinterpret_cast<volatile std::uint32_t*>(0xE000'EDFCu);
inline volatile std::uint32_t& ctrl   = *reinterpret_cast<volatile std::uint32_t*>(0xE000'1000u);
inline volatile std::uint32_t& cyccnt = *reinterpret_cast<volatile std::uint32_t*>(0xE000'1004u);
inline volatile std::uint32_t& lar    = *reinterpret_cast<volatile std::uint32_t*>(0xE000'1FB0u);

inline void enable_cycle_counter() {
    demcr |= (1u << 24);    // TRCENA: power the DWT and ITM
    lar = 0xC5AC'CE55u;     // unlock DWT registers (required on Cortex-M7)
    cyccnt = 0;
    ctrl |= 1u;             // CYCCNTENA
}

inline std::uint32_t now() { return cyccnt; }

}  // namespace dwt

struct timing_stats {
    std::uint32_t max_cycles = 0;
    std::uint32_t min_cycles = UINT32_MAX;
    std::uint32_t samples = 0;

    void record(std::uint32_t cycles) {
        if (cycles > max_cycles) max_cycles = cycles;
        if (cycles < min_cycles) min_cycles = cycles;
        ++samples;
    }
};

timing_stats g_control_step_stats;

void control_step_measured() {
    const std::uint32_t start = dwt::now();
    control_step();                                    // code under test
    const std::uint32_t elapsed = dwt::now() - start;  // unsigned wrap is correct
    g_control_step_stats.record(elapsed);
}
```

Unsigned subtraction handles one wrap of the counter correctly, so any interval below $2^32$ cycles (about 25 s at 168 MHz) is measured exactly. Two practical cautions:

- The measured interval includes any interrupt that fired inside it. To measure the code alone, either mask interrupts around the measurement (which changes the system you are measuring) or also record the ISR durations and subtract. For a response-time measurement, you *want* the interrupts included.
- Reading `CYCCNT` is a peripheral bus access of a few cycles, and on the M7 the dual-issue pipeline can move neighbouring instructions across it. Calibrate by measuring an empty region and subtracting that constant.

A GPIO toggle at the start and end of the region, captured with a logic analyser or oscilloscope, gives the same information from outside the chip, including the ISR and DMA effects, and is often the measurement a certification reviewer trusts most.

== From WCET to Response Time on a Real System

The textbook response-time recurrence assumes independent tasks with free context switches. A real Cortex-M system adds four effects, each of which slots into the recurrence.

=== Interrupts as Highest-Priority Tasks

Every ISR runs above every task. Model each interrupt source $k$ as a sporadic task with WCET $C_k$ (the handler, including entry and exit) and *minimum inter-arrival time* $T_k$. If a source has no minimum inter-arrival time (a noisy GPIO edge, a $"UART"$ under attack), the system is not analysable, so real designs enforce one: disable the interrupt in its handler and re-enable it from a timer, or move the input behind a peripheral that rate-limits it.

$ R_i = C_i + B_i + sum_(j in "hp"(i)) ceil(R_i / T_j) C_j + sum_(k in "isr") ceil(R_i / T_k) C_k $

=== Kernel Overheads

Each preemption costs two context switches (in and out). Charge them to the preempting task: replace $C_j$ with $C_j + 2 C_"cs"$. On FreeRTOS on a Cortex-M4, $C_"cs"$ is roughly 80 to 150 cycles depending on FPU state. Add the tick interrupt as one more sporadic task with $T = $ the tick period, unless the kernel runs tickless.

=== Blocking

$B_i$ is the longest time task $i$ can be delayed by lower-priority work. Under the priority ceiling protocol it is the longest single critical section of any lower-priority task whose resource ceiling is at least task $i$'s priority. Interrupt-masked regions also count: the longest `__disable_irq()` window anywhere in the system (including inside the $"RTOS"$ and vendor HAL) is a blocking term for *every* task and every ISR below the mask level. This is why `BASEPRI`-based critical sections, which mask only interrupts at or below a configured priority, are preferred over `PRIMASK` on Armv7-M: interrupts above the threshold keep their latency.

=== Release Jitter

If task $j$ is released by an event that may arrive up to $J_j$ late (for example, it is woken by an ISR that itself waits behind other ISRs), it can preempt task $i$ one extra time in a window. The recurrence becomes

$ R_i = C_i + B_i + sum_(j in "hp"(i)) ceil((R_i + J_j) / T_j) C_j $

and the worst-case response measured from the *nominal* release is $R_i + J_i$.

=== A Worked Example

Three periodic tasks on one core, rate-monotonic priorities, deadlines equal to periods:

#table(
  columns: 5,
  [*Task*], [*$C$ (ms)*], [*$T = D$ (ms)*], [*$U$*], [*Priority*],
  [$tau_1$ sensor], [1], [4], [0.250], [High],
  [$tau_2$ control], [2], [6], [0.333], [Medium],
  [$tau_3$ telemetry], [3], [12], [0.250], [Low],
)

Total utilisation is $0.833$, above the Liu and Layland bound $3(2^(1\/3) - 1) approx 0.780$, so the utilisation test is inconclusive. Response-time analysis decides it:

- $R_1 = 1$.
- $R_2$: start at 2; $2 + ceil(2\/4) dot 1 = 3$; $2 + ceil(3\/4) dot 1 = 3$. Converged, $R_2 = 3 <= 6$.
- $R_3$: start at 3, then $3 + 1 + 2 = 6$, then $3 + ceil(6\/4) dot 1 + ceil(6\/6) dot 2 = 7$, then $3 + 2 + 4 = 9$, then $3 + ceil(9\/4) dot 1 + ceil(9\/6) dot 2 = 3 + 3 + 4 = 10$, then $3 + ceil(10\/4) + 2 ceil(10\/6) = 10$. Converged, $R_3 = 10 <= 12$.

The set is schedulable with 2 ms of slack on the lowest-priority task. Now add a 200 µs $"UART"$ ISR with a minimum inter-arrival of 1 ms (one byte at 9600 baud is about 1.04 ms, so that bound holds). The ISR consumes up to $ceil(R\/1) dot 0.2$ ms in any window of length $R$, a utilisation of 0.2 on its own. Total utilisation becomes $1.033$, and $tau_3$ can no longer finish: the recurrence passes 12 and keeps rising. The fix is not a faster CPU but a different design: receive into a DMA ring buffer and take one interrupt per line rather than per byte.

The example generalises. Interrupt load is the most commonly forgotten term in embedded schedulability, and per-byte or per-sample interrupts are its most common source.

== Stack Depth Is Part of Timing Analysis

A stack overflow on an $"MCU"$ without an $"MPU"$ guard silently corrupts whatever lies below the stack, which is usually `.bss` or another task's stack. The fault often shows up as a timing failure long after the cause. Worst-case stack depth is analysed the same way as $"WCET"$:

- *Static*: GCC's `-fstack-usage` writes a `.su` file with the frame size of every function; combined with a call graph (from `-fcallgraph-info` in GCC 10 and later, or tools such as `puncover` and StackAnalyzer), the deepest path gives the bound. Recursion and calls through function pointers break the analysis unless annotated, which is one reason MISRA forbids recursion.
- *Interrupt frames*: add the hardware-stacked frame (8 words, or 26 words with lazy FP stacking) for every nesting level the priority scheme allows, on the stack that ISRs use (`MSP` on Cortex-M).
- *Measured*: fill each stack with a known pattern at boot ("stack painting") and periodically scan for the deepest overwritten word. FreeRTOS exposes this as `uxTaskGetStackHighWaterMark`.

Size each stack as the static bound plus margin, and use the $"MPU"$ (or the Armv8-M `PSPLIM`/`MSPLIM` stack-limit registers) to turn an overflow into an immediate, diagnosable fault.

== Multicore Interference

On a multicore $"MCU"$ or $"SoC"$, a task's execution time depends on what the other cores do to the shared L2 cache, interconnect and memory controller. Measured slowdowns from a single co-runner hammering DRAM range from 2x to over 10x on application-class parts. The aviation guidance CAST-32A (2016), since superseded by AMC 20-193 (2022), requires applicants to identify every interference channel and show it is either eliminated or bounded. The usual toolkit:

- *Partition the cache* by way locking or page colouring, so each core owns disjoint sets.
- *Regulate bandwidth*: MemGuard-style budgets throttle a core that exceeds its share of memory accesses per period, using performance counters.
- *Place critical data in core-private memory* (TCM, private SRAM banks), which has no shared path at all.
- *Use lockstep or time partitioning* (ARINC 653) so that only one criticality domain runs at a time on shared resources.

The simplest robust answer, still common in automotive, is to give hard real-time work its own small in-order core (Cortex-R or Cortex-M) with private memory, and leave the shared multicore complex to soft real-time and best-effort software.

== Designing for Analysability

Timing analysis is far cheaper when the code is written for it:

- *Bound every loop* with a constant or a parameter that has a documented maximum. Avoid "loop until the peripheral is ready" without a timeout.
- *Avoid dynamic allocation after initialisation.* Allocator time depends on heap history.
- *Keep critical sections short and named*, so the longest one can be found by inspection.
- *Use fixed-rate, time-triggered structure where possible.* A cyclic executive with a fixed frame table (common in avionics) has a WCET argument that reduces to "each frame's work fits in the frame".
- *Place hard real-time code and data in TCM or SRAM*, out of the flash wait-state and cache problem.
- *Defer work out of ISRs* to tasks whose priority reflects its deadline, so the ISR term in the analysis stays small.
- *Make overload behaviour explicit.* Decide which task degrades first and how (skip a cycle, use the previous value) rather than discovering it in the field.

== Further Reading

Wilhelm, R. et al. (2008). "The Worst-Case Execution-Time Problem: Overview of Methods and Survey of Tools." ACM Transactions on Embedded Computing Systems.

Lundqvist, T., Stenström, P. (1999). "Timing Anomalies in Dynamically Scheduled Microprocessors." RTSS.

Li, Y.-T. S., Malik, S. (1995). "Performance Analysis of Embedded Software Using Implicit Path Enumeration." DAC.

Audsley, N., Burns, A., Richardson, M., Tindell, K., Wellings, A. (1993). "Applying New Scheduling Theory to Static Priority Pre-emptive Scheduling." Software Engineering Journal.

Cucu-Grosjean, L. et al. (2012). "Measurement-Based Probabilistic Timing Analysis for Multi-path Programs." ECRTS.

Yun, H., Yao, G., Pellizzoni, R., Caccamo, M., Sha, L. (2013). "MemGuard: Memory Bandwidth Reservation System for Efficient Performance Isolation in Multi-core Platforms." RTAS.

Certification Authorities Software Team (2016). "CAST-32A: Multi-core Processors." FAA position paper.

Burns, A., Wellings, A. (2009). "Real-Time Systems and Programming Languages." Addison-Wesley, 4th ed.

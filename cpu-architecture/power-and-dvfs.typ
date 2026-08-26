#import "../template.typ": xref

= Power Management and DVFS

Power is the limiting resource of modern processors. Every architectural choice — from frequency to core count to instruction mix — is constrained by a thermal design power ($"TDP"$) budget. This chapter covers the mechanisms that trade voltage, frequency, and idle states for energy efficiency.

*See also:* #xref("cpu-architecture", "cpu-fundamentals", label: "CPU Fundamentals"), #xref("cpu-architecture", "multicore", label: "Multicore"), #xref("cpu-architecture", "performance-analysis", label: "Performance Analysis")

== The Power Equation

Dynamic CMOS power follows $P_"dyn" = alpha C V^2 f$, where $alpha$ is activity factor, $C$ capacitance, $V$ voltage, $f$ frequency. Static (leakage) power is voltage- and temperature-dependent and dominates at low activity.

Voltage and frequency are tightly coupled: higher $f$ needs higher $V$ for stable switching, so a 20% frequency increase can mean 40-60% more power. This cubic scaling is why dynamic voltage and frequency scaling ($"DVFS"$) is the most powerful knob in the toolbox.

== P-states (Performance States)

ACPI defines P0..Pn frequency/voltage operating points. P0 is the highest; higher P-numbers mean lower $f$ and $V$.

#table(
  columns: 5,
  [*State*], [*Freq (GHz)*], [*Voltage (V)*], [*Power (W)*], [*Use*],
  [P0 (Turbo)], [4.8], [1.30], [125], [Burst],
  [P1 (Base)], [3.6], [1.05], [65], [Sustained],
  [P2], [3.0], [0.95], [45], [Background],
  [P3], [2.4], [0.85], [28], [Idle-ish],
  [P4 ($"LFM"$)], [0.8], [0.70], [8], [Lowest active],
)

*Governor* (Linux): the policy that picks P-states. `ondemand`, `conservative`, `powersave`, `performance`, and the modern `schedutil` (uses scheduler util signal). Intel's `intel_pstate` driver bypasses the generic governor with hardware-controlled P-states ($"HWP"$ / Speed Shift), which can transition in ~1 ms vs. ~30 ms for software.

```
# Inspect current state
$ cpupower frequency-info
$ cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Force performance mode (benchmarking)
$ cpupower frequency-set -g performance
$ echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

== Turbo Boost / Precision Boost

When fewer cores are active, the chip lets active cores exceed the base frequency until power, current, or temperature limits trip. Intel exposes per-core-count tables (Turbo Boost 2.0) and per-core favored cores (Turbo Boost Max 3.0). AMD's Precision Boost 2 is the analogue (Precision Boost Overdrive raises its limits) with three concurrent limits: $"PPT"$ (package power), $"TDC"$ (thermal design current), $"EDC"$ (electrical design current).

The opportunistic frequency depends on:

- Number of active cores
- Instruction mix ($"AVX2"$ and $"AVX-512"$ have *lower* turbo ceilings due to wider activated logic — "$"AVX"$ offset")
- Die temperature
- Power-delivery headroom

This is why benchmarks demand cold runs and locked frequency for reproducibility.

== C-states (Idle States)

C0 is "executing"; C1..Cn are progressively deeper idle. Each saves more power but takes longer to wake.

#table(
  columns: 4,
  [*State*], [*What is off*], [*Exit latency*], [*Power*],
  [C0], [Nothing], [0], [Full],
  [C1 ($"HLT"$)], [Core clock], [\~1 $mu$s], [~30%],
  [C1E], [Core + voltage reduction], [\~10 $mu$s], [~20%],
  [C3], [L1/L2 flushed], [\~50 $mu$s], [~10%],
  [C6], [Core power gated, state saved], [\~150 $mu$s], [~2%],
  [C7-C10], [Package-level: L3, ring, $"PLL"$s off], [up to ~1 ms], [\<1%],
)

Deep C-states are the enemy of latency-sensitive workloads: a single wake-up burst from C6 to C0 can cost 200$mu$s during which the core is unresponsive. Tuning options:

```
# Disable deep C-states (latency-sensitive)
$ cpupower idle-set -D 10            # max exit latency 10us
$ ... or kernel cmdline: intel_idle.max_cstate=1 processor.max_cstate=1
```

== RAPL: Running Average Power Limit

Intel $"RAPL"$ (Sandy Bridge+) exposes energy counters and power caps across four domains: $"PKG"$ (package), $"PP0"$ (core), $"PP1"$ ($"iGPU"$ or uncore), and $"DRAM"$. AMD has analogous counters since Zen.

```c
// Read package energy (Linux, requires root or perf_event_paranoid<=0)
#include <linux/perf_event.h>
// pseudo-code:
int fd = perf_event_open(&attr, /*pid*/-1, /*cpu*/0, -1, 0);
uint64_t energy_uj;
read(fd, &energy_uj, sizeof(energy_uj));
// energy in microjoules; sample twice, divide by interval for watts
```

```
# Quick CLI
$ sudo turbostat --interval 1
$ sudo powerstat -R 1
```

Two power windows are configurable: $"PL1"$ (long-term, ~$"TDP"$, 28-second window) and $"PL2"$ (short-term boost, ~1.25-1.5x $"TDP"$, ~28 ms). Laptop $"OEM"$s aggressively tune these — a 28W chip might be configured for 15/28 or 35/64.

*$"RAPL"$ as a side channel:* see $"PLATYPUS"$ (2020) — at high sample rates $"RAPL"$ leaks crypto keys; mainline kernels now restrict $"RAPL"$ to root.

== Big.LITTLE and Hybrid Architectures

A single die mixes core types optimized for different points on the $"PPA"$ curve:

#table(
  columns: 4,
  [*Family*], [*Big core*], [*Little core*], [*Notes*],
  [ARM big.LITTLE (2011)], [Cortex-A15/A57], [Cortex-A7/A53], [Cluster migration, then global task scheduling, then $"DynamIQ"$],
  [Apple A11+ (2017)], [Monsoon/Mistral], [...], [Apple's first $"AMP"$; all cores usable at once],
  [Apple M1 (2020)], [Firestorm (P)], [Icestorm (E)], [4P+4E, shared L3],
  [Intel Alder Lake (2021)], [Golden Cove (P)], [Gracemont (E)], [Thread Director],
  [Intel Meteor Lake (2023)], [Redwood Cove (P)], [Crestmont (E + $"LP-E"$)], [Three tiers; $"LP-E"$ on $"SoC"$ tile],
  [ARM Neoverse mixed], [N2/V2], [...], [Server hybrid (rare)],
)

*Why hybrid:* E-cores hit 60-80% of P-core $"IPC"$ at 30-40% of the area and 25% of the power. On throughput workloads, two E-cores beat one P-core in perf/W and perf/mm². P-cores still win on single-thread.

*Asymmetric scheduling:* the OS needs $"ISA"$ parity (no $"AVX-512"$ on Gracemont — Intel disabled it on P-cores too on Alder Lake desktop), and it needs core-type-aware load balancing. Intel Thread Director provides hardware feedback (instruction class hints) to Windows/Linux schedulers via an $"MSR"$-backed feedback table.

Linux uses Energy Aware Scheduling ($"EAS"$) on ARM and the `sched_ext` family; util-clamp ($"uclamp_min"$/$"max"$) lets userspace bias important tasks toward P-cores.

```
# Inspect topology
$ lscpu --extended
# CPU NODE SOCKET CORE L1d:L1i:L2:L3 ...
# Look at MAXMHZ for P vs E differentiation on Intel hybrid.
```

== Per-Instruction Energy

Order-of-magnitude per-operation energy [Horowitz 2014, 45nm]:

#table(
  columns: 3,
  [*Operation*], [*Energy (pJ)*], [*Notes*],
  [32-bit register read], [0.1], [],
  [32-bit ALU add], [0.1], [],
  [32-bit FP add], [0.9], [],
  [32-bit FP mul], [3.7], [],
  [L1 cache access], [10], [],
  [L2 cache access], [30], [],
  [L3 cache access], [100], [],
  [DRAM access], [1300], [10000x ALU],
  [Off-chip link (per bit)], [10-20], [],
)

The takeaway: data movement dominates energy. The "memory wall" is also a "memory energy wall." Algorithmic locality and on-die accelerators (e.g., $"AMX"$, $"TMUL"$) deliver order-of-magnitude perf/W wins by keeping data close.

== Thermal Management

When $"T"_j$ approaches $"T"_"jmax"$ (typically 100-110$degree$C), the CPU throttles. Sequence on Intel:

1. Soft throttle: lower P-state (~5$degree$C below limit).
2. Thermal duty cycling ($"TM1"$): clock gated 30-50% of cycles.
3. Critical: forced shutdown at $"T"_"jmax"$ + ~25$degree$C.

Notebook designs leverage skin-temperature controllers ($"DTT"$): cap power so the chassis stays below 45$degree$C even if silicon could draw more.

== Race-to-Idle vs. Pace

Two opposed strategies:

- *Race-to-idle:* run flat-out, finish fast, drop to deep C-state. Wins when leakage is high relative to dynamic, or when wake-up cost is amortizable.
- *Pace (slow-and-steady):* run at the energy-optimal frequency just above the leakage knee. Wins on always-on, latency-tolerant workloads.

Empirically, modern silicon ($lt.eq$ 7 nm) with aggressive power gating favors race-to-idle. Mobile $"SoC"$s also prefer race-to-idle to spend more time in deep package states ($"S0iX"$).

== Tooling Cheat Sheet

```
$ sudo turbostat --quiet --interval 1
$ sudo powertop
$ sudo perf stat -e power/energy-pkg/,power/energy-cores/,power/energy-ram/ ./app
$ sudo intel_pstate_tracer.py    # P-state transitions over time
$ likwid-powermeter -M 0 -- ./app
```

== Further Reading

Horowitz, M. (2014). "Computing's Energy Problem (and What We Can Do About It)." _ISSCC '14 Keynote_.

Hennessy, J.L. & Patterson, D.A. (2017). _Computer Architecture_, 6th ed., Ch. 1 ($"DVFS"$, power).

Intel Corporation (2024). _Intel 64 and IA-32 Architectures Software Developer's Manual_, Vol. 3B, Ch. 14-15 ($"HWP"$, $"RAPL"$, Thread Director).

ACPI Specification 6.5 (2022). _Processor Power and Performance States_.

Lipp, M. et al. (2021). "$"PLATYPUS"$: Software-Based Power Side-Channel Attacks on x86." _IEEE S&P '21_.

ARM (2022). _Arm DynamIQ Shared Unit Technical Reference Manual_.

Wang, Y. et al. (2022). "$"Hertzbleed"$: Turning Power Side Channels into Remote Timing Attacks." _USENIX Security '22_.

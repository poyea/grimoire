#import "../template.typ": xref

= ARM Deep Dive

ARM is the most-shipped $"CPU"$ architecture in history — more than 250 billion cores. From microcontrollers to Apple's M-series desktops to AWS Graviton servers, AArch64 spans seven orders of magnitude in power. This chapter focuses on modern AArch64 microarchitecture: Apple silicon, $"SVE"$/$"SVE2"$, Neoverse, and the Cortex families.

*See also:* #xref("cpu-architecture", "cpu-fundamentals", label: "CPU Fundamentals"), #xref("cpu-architecture", "simd", label: "SIMD"), #xref("cpu-architecture", "risc-v", label: "RISC-V"), #xref("cpu-architecture", "out-of-order-deep-dive", label: "Out-of-Order Execution Deep Dive")

== AArch64 ISA Highlights

AArch64 (introduced ARMv8, 2011) discarded the 32-bit ARM baggage:

- 31 general-purpose 64-bit registers (`x0`-`x30`) + zero/stack pointer (`xzr`/`sp`); no banking.
- No condition codes on every instruction (a few flag-setting variants only).
- $"PC"$-relative addressing for large code models.
- Weak memory order with explicit barriers (`dmb`, `dsb`, `isb`) and load-acquire/store-release variants (`ldar`/`stlr`).
- 32 SIMD/FP registers ($"NEON"$, 128-bit); $"SVE"$ adds scalable Z registers.

ARMv8.1 added $"LSE"$ (large-system extensions: native atomics — `casal`, `ldadd`); ARMv8.2 added half-precision FP; ARMv9 (2021) made $"SVE2"$ baseline and added confidential compute ($"CCA"$).

== Apple M-Series Microarchitecture

Apple's transition from Intel to Apple Silicon (2020) was an architectural lesson: the company built the widest, deepest OoO core in commercial silicon.

=== Firestorm (M1 P-core, 2020)

#table(
  columns: 2,
  [*Property*], [*Value*],
  [Decode width], [8 instructions/cycle],
  [ROB], [~630 entries],
  [Integer rename], [~380 physical regs],
  [FP/NEON rename], [~400 physical regs],
  [Integer execution units], [6 ALUs + 2 branch + 2 mul + 1 div],
  [FP/NEON pipelines], [4 (all 128-bit FMAs)],
  [Load/store], [3 loads + 2 stores per cycle],
  [L1 I/D], [192 KB / 128 KB],
  [L2 (shared per P-cluster)], [12 MB],
  [SLC (system-level cache)], [16 MB on M1, 96 MB on M3 Max],
  [Branch predictor], [TAGE-like, very large],
)

For comparison, the contemporary Intel Tiger Lake had 4-6 decode, 352-entry $"ROB"$. Apple's wider front-end + bigger window extracted more $"IPC"$ at lower frequency (3.2 GHz vs 5 GHz) — better perf/W.

=== Evolution

#table(
  columns: 4,
  [*Chip*], [*P-core*], [*E-core*], [*Notable*],
  [M1 (2020)], [Firestorm, 3.2 GHz], [Icestorm, 2.0 GHz], [4P+4E],
  [M2 (2022)], [Avalanche, 3.5 GHz], [Blizzard, 2.4 GHz], [Larger caches],
  [M3 (2023)], [(Everest)], [(Sawtooth)], [TSMC N3B, dynamic caching $"GPU"$],
  [M4 (2024)], [(unnamed)], [(unnamed)], [ARMv9; SME2 SVE/SME engines],
)

The M-series shares the same core IP with the A-series ($"iPhone"$); the desktop variants add P-cores, larger caches, and wider memory.

=== Apple AMX and SME

Apple's "$"AMX"$" coprocessor (undocumented since A13; not Intel's $"AMX"$) is a matrix multiply unit accessible only via accelerate.framework. M4 finally exposes equivalent functionality as ARM standard Scalable Matrix Extension ($"SME"$): outer-product tile operations on $"SVE"$ registers, INT8/BF16/FP16/FP32/FP64.

== ARM Scalable Vector Extension (SVE)

$"SVE"$ (introduced ARMv8.2-A, finalized 2017) is ARM's answer to $"AVX-512"$ but designed *vector-length agnostic*. The same binary runs on hardware with $"VL"$ from 128 to 2048 bits (any multiple of 128).

Key concepts:

- *Predicate registers* (P0-P15): 1 bit per element, enabling masked operations and lane disable.
- *First-faulting loads* (`ldff1`): vectorize loops that may segfault past the end.
- *Gather/scatter:* native, with predication.

```asm
// SVE: daxpy y = a*x + y, length n
// x0=y, x1=x, n=x2, d0=a
daxpy:
    mov     x3, #0                 // i = 0
    whilelt p0.d, x3, x2           // p0 lanes = (i+lane < n)
    b.none  .Lend
.Lloop:
    ld1d    z1.d, p0/z, [x1, x3, lsl #3]
    ld1d    z2.d, p0/z, [x0, x3, lsl #3]
    fmla    z2.d, p0/m, z1.d, z0.d
    st1d    z2.d, p0,  [x0, x3, lsl #3]
    incd    x3
    whilelt p0.d, x3, x2
    b.first .Lloop
.Lend:
    ret
```

$"SVE2"$ (ARMv9, 2021) adds DSP-style instructions (saturating arithmetic, narrow-multiply-and-accumulate) and is mandatory in ARMv9-A; this lets vendors deprecate fixed-128-bit $"NEON"$ for new code paths.

Vendors' $"VL"$ choices:

#table(
  columns: 3,
  [*Vendor / Chip*], [*VL*], [*Notes*],
  [Fujitsu A64FX (Fugaku)], [512], [First SVE silicon, 2020],
  [AWS Graviton 3 (Neoverse V1)], [256], [Two 256-bit pipes],
  [AWS Graviton 4 (Neoverse V2)], [128 (4 pipes)], [Four 128-bit pipes],
  [NVIDIA Grace (Neoverse V2)], [128 (4 pipes)], [],
  [Apple M4], [128 (SME2)], [],
)

The lesson: peak vector throughput depends on (number of pipes)$times$(pipe width), not $"VL"$ alone. Programmers should use $"VL"$-agnostic intrinsics (`svadd_x` etc.) and let the compiler unroll.

== Neoverse: ARM Server Cores

ARM's $"Neoverse"$ line targets servers and infrastructure with a different power/perf point than the mobile-derived Cortex line.

#table(
  columns: 4,
  [*Core*], [*Class*], [*Decode*], [*Notable customers*],
  [Neoverse N1 (2019)], [Efficiency], [4-wide], [Graviton 2, Ampere Altra],
  [Neoverse V1 (2021)], [Performance], [5-wide, SVE 256], [Graviton 3, SiPearl Rhea],
  [Neoverse N2 (2022)], [Efficiency], [5-wide, SVE2 128], [Alibaba Yitian 710],
  [Neoverse V2 (2022)], [Performance], [6-wide, SVE2 4$times$128], [Graviton 4, NVIDIA Grace],
  [Neoverse V3 (2024)], [Performance], [8-wide, SVE2], [Microsoft Cobalt 100],
  [Neoverse N3 (2024)], [Efficiency], [9-wide], [],
)

By 2024 Neoverse V3 hits 9-wide decode — closing the gap to Apple's P-cores. Cloud providers' preference for ARM is largely about perf/W and the ability to design custom silicon at $"hyperscaler"$ scale (Graviton, Cobalt, Axion, Yitian, Maia).

== Cortex Families

ARM's standard $"IP"$ catalog covers four families:

#table(
  columns: 3,
  [*Family*], [*Use*], [*Examples*],
  [Cortex-X], [Mobile flagship "big"], [X1, X2, X3, X4, X925],
  [Cortex-A], [General apps], [A53, A55, A76, A78, A715, A720],
  [Cortex-R], [Real-time], [R5, R52, R82 (only R with 64-bit MMU)],
  [Cortex-M], [Microcontroller], [M0/M3/M4/M7/M33/M55/M85],
)

*Cortex-X* cores prioritize $"IPC"$ over area: wider OoO, larger caches, higher peak frequency. They are the "big" in $"DynamIQ"$ big.LITTLE.medium clusters; e.g., a flagship phone $"SoC"$ has 1 X4 + 5 A720 + 2 A520.

*Cortex-M55/M85* are the first M-class cores with $"Helium"$ ($"M-profile"$ Vector Extension, $"MVE"$): $"SIMD"$ for microcontrollers, enabling on-device $"DSP"$/$"ML"$ inference.

*Cortex-R82* is the first R-class to support 64-bit and an $"MMU"$ (vs $"MPU"$), enabling Linux on real-time control planes (e.g., storage controllers).

== Memory Model and Atomics

ARM's weak memory model is performance-friendly but bug-prone:

```asm
// Acquire/release: lighter than full DMB
ldar    w0, [x1]     // load-acquire: no later access reordered before
stlr    w0, [x2]     // store-release: no earlier access reordered after

// Full data memory barrier
dmb     ish          // inner-shareable
dmb     sy           // system-wide (default)
```

$"LSE"$ atomics (ARMv8.1+) provide compiler-friendly single-instruction $"RMW"$:

```asm
// Atomic add, return-original
ldaddal w0, w1, [x2]  // [x2] += w0; w1 = old; acquire+release
casal   w0, w1, [x2]  // CAS, acquire+release
```

On pre-$"LSE"$ silicon the kernel/$"glibc"$ uses load-exclusive/store-exclusive (`ldxr`/`stxr`) loops, which can livelock under contention; $"LSE"$ removes that hazard.

== Performance Counters and Tooling

```
# Linux perf on ARM
$ perf list | head
# r-prefixed events use the raw event encoding from the ARM ARM, Ch. D.

# Top-down (Neoverse V1+)
$ perf stat -M TopdownL1 ./app

# Apple silicon: powermetrics + Instruments
$ sudo powermetrics --samplers cpu_power -n 1
```

Apple does not expose $"PMC"$s to userspace by default; profilers use kperf via Instruments.app. Linux on Apple silicon (Asahi) is making progress on $"PMU"$ access.

== Strengths and Tradeoffs

*Strengths*: industry-leading perf/W, modular IP (mix-and-match X/A/M cores), open compiler ecosystem, $"SVE"$ portability.

*Tradeoffs*: weak memory model demands care; $"NEON"$/$"SVE"$/$"SVE2"$/$"SME"$ proliferation (similar to $"x86"$'s $"SSE"$/$"AVX"$ history); architectural license is expensive (only a handful of vendors hold one).

== Further Reading

ARM Limited (2024). _Arm Architecture Reference Manual for A-profile architecture_ (ARM ARM). Latest revision (ARMv9.5-A).

ARM Limited (2023). _Arm Neoverse V2 Reference Manual_.

ARM Limited (2017). _The Scalable Vector Extension (SVE), for ARMv8-A_.

Stephens, N. et al. (2017). "The ARM Scalable Vector Extension." _IEEE Micro_ 37(2).

Hennessy, J.L. & Patterson, D.A. (2017). _Computer Architecture_, 6th ed., App. K (RISC ISAs incl. ARM).

Johnson, A. (2022). "Apple Silicon: The M1 Microarchitecture." _Hot Chips 33_.

Frumusanu, A. _Anandtech reviews of Apple M1/M2/M3 and Neoverse N1/N2/V1/V2_.

Asahi Linux project (2023). _Apple Silicon Reverse-Engineering Notes_. https://asahilinux.org/docs/

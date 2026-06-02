= RISC-V

RISC-V (pronounced "risk-five") is an open, modular instruction set architecture born at UC Berkeley in 2010 and now governed by RISC-V International. Unlike $"x86"$ or $"ARM"$, the $"ISA"$ itself is royalty-free; vendors compose extensions to fit their target.

*See also:* _CPU Fundamentals_, _SIMD_, _ARM Deep Dive_, _Pipelining_

== Design Philosophy

RISC-V was designed with the explicit goals of being clean (no historical baggage), modular (small base + optional extensions), and scalable (the same $"ISA"$ from microcontrollers to servers). It draws on three decades of $"RISC"$ research: MIPS load/store, $"SPARC"$ register windows lessons (avoided), $"DEC"$ Alpha's clean slate, and $"ARM"$'s modularity (improved).

Key design choices:

- *Fixed 32-bit instructions* in the base, with optional 16-bit compressed (`C`).
- *31 general-purpose integer registers* (x1-x31) plus a hard-wired zero `x0`.
- *No condition codes;* compare-and-branch instructions only (like MIPS).
- *No mode bits;* register-width determined by base $"ISA"$ ($"RV32"$, $"RV64"$, $"RV128"$).
- *Branch prediction-friendly:* relative-only branches, fixed-width encoding.

== Base ISAs

#table(
  columns: 3,
  [*Base*], [*XLEN*], [*Use*],
  [RV32I], [32], [Microcontrollers, embedded],
  [RV32E], [32, 16 regs], [Tiny embedded ("E" = embedded)],
  [RV64I], [64], [Application processors, servers],
  [RV128I], [128], [Reserved, future],
)

The base I (integer) $"ISA"$ has only ~47 instructions: arithmetic (`add`, `sub`, `sll`, `srl`, `sra`, `and`, `or`, `xor`), immediates (`addi`, etc.), loads/stores, branches, `jal`/`jalr`, and `ecall`/`ebreak`. That is the entire $"CPU"$.

```asm
# RV64I: sum an array
# a0 = pointer, a1 = length, returns a0 = sum
sum:
    li    t0, 0          # accumulator
    beqz  a1, .Lend
.Lloop:
    ld    t1, 0(a0)
    add   t0, t0, t1
    addi  a0, a0, 8
    addi  a1, a1, -1
    bnez  a1, .Lloop
.Lend:
    mv    a0, t0
    ret
```

== Standard Extensions

The extension letter is appended to the base; `RV64GC` = `RV64IMAFDC` + Zicsr + Zifencei (the "G" general profile) + compressed.

#table(
  columns: 3,
  [*Ext*], [*Adds*], [*Notes*],
  [M], [Integer multiply/divide], [Optional even on embedded],
  [A], [Atomic memory operations (LR/SC + AMO\*)], [load-reserved/store-conditional],
  [F], [Single-precision FP (32 FP registers)], [],
  [D], [Double-precision FP], [Requires F],
  [Q], [Quad-precision FP (128-bit)], [Rare],
  [Zicsr], [Control/status register access], [Used by privileged spec],
  [Zifencei], [Instruction-fetch fence], [Self-modifying code],
  [C], [16-bit compressed], [25-30% code-size reduction],
  [B], [Bit manipulation (Zba/Zbb/Zbc/Zbs)], [Ratified 2021],
  [V], [Vector extension], [Ratified 2021],
  [H], [Hypervisor], [],
  [Zicbom/Zicboz/Zicbop], [Cache management], [],
  [Zacas], [Compare-and-swap atomics], [],
  [Zfh / Zfhmin], [Half-precision FP], [],
)

== Vector Extension (RVV 1.0)

The vector extension is RISC-V's most distinctive feature. Unlike $"SSE"$/$"AVX"$/$"NEON"$, $"RVV"$ uses *vector-length agnostic* ($"VLA"$) programming: the same binary runs on hardware with any $"VLEN"$ (vector register width, 128-65536 bits).

Key state:

- `vlen` (hardware-fixed): vector register width.
- `vl` (runtime): how many elements to process this iteration.
- `vtype`: element width ($"SEW"$ = 8/16/32/64) and grouping ($"LMUL"$ = 1/2/4/8 registers grouped).

Idiomatic loop:

```asm
# vector-add: c[i] = a[i] + b[i], i = 0..n
# a0=c, a1=a, a2=b, a3=n
vadd:
.Lloop:
    vsetvli  t0, a3, e32, m4    # request up to a3 elements, 32-bit, 4-reg group
    vle32.v  v0, (a1)           # load t0 elements from a
    vle32.v  v4, (a2)
    vadd.vv  v8, v0, v4
    vse32.v  v8, (a0)
    sub      a3, a3, t0
    slli     t1, t0, 2          # bytes consumed
    add      a0, a0, t1
    add      a1, a1, t1
    add      a2, a2, t1
    bnez     a3, .Lloop
    ret
```

`vsetvli` reports back the actual count chosen — the loop adapts automatically to any $"VLEN"$. Compare to writing eight code paths for $"NEON"$/$"SSE"$/$"AVX2"$/$"AVX-512"$/$"SVE"$.

== Profiles: RVA22, RVA23

A "profile" is a fixed bundle of extensions that software can target, eliminating the chaos of ad-hoc combinations. Defined by the Profile Task Group.

#table(
  columns: 4,
  [*Profile*], [*Year*], [*Mandatory*], [*Notable*],
  [RVA20], [2021], [RV64GC + Zicntr/Zihpm], [First app-class profile],
  [RVA22], [2022], [+ B, Zihintpause, Zicboz/Zicbom/Zicbop], [General-purpose Linux baseline],
  [RVA23], [2024], [+ V, Zfh, Zfa, Zacas, Zicond, hypervisor], [Vector + hypervisor required],
  [RVB22 / RVB23], [-], [Server variants], [Adds I/O hooks],
)

RVA23 is the milestone that lets distros (Debian, Ubuntu, Fedora, Android) ship a single binary that exploits vectors and modern atomics. Pre-RVA23, $"RVV"$ was effectively opt-in via dynamic dispatch.

== Privileged Architecture

Three privilege levels: M (machine, mandatory), S (supervisor), U (user). The H (hypervisor) extension splits S into HS / VS / VU for virtualization.

Page-table formats follow $"x86"$-style multi-level radix:

#table(
  columns: 4,
  [*Mode*], [*Levels*], [*Address bits*], [*Page sizes*],
  [Sv32], [2], [32], [4 KB, 4 MB],
  [Sv39], [3], [39], [4 KB, 2 MB, 1 GB],
  [Sv48], [4], [48], [+ 512 GB],
  [Sv57], [5], [57], [+ 256 TB],
)

== Hardware Vendors

#table(
  columns: 4,
  [*Vendor*], [*Cores*], [*Target*], [*Notable*],
  [SiFive], [U54, U74, P550, P670, P870], [Edge to apps], [Spun from Berkeley, first commercial RISC-V],
  [Andes], [N25, A45, NX27V], [Microcontroller to apps], [Vector early],
  [Tenstorrent], [Ascalon (8-wide OoO)], [Server/AI], [Jim Keller-led],
  [Ventana], [Veyron V1/V2], [Server], [Chiplet-based],
  [Rivos], [Stealth], [Datacenter], [],
  [SpacemiT], [K1, X100], [Consumer (Banana Pi)], [First mainstream RVA22 device],
  [Esperanto], [ET-SoC-1], [AI inference], [1088 cores],
  [Alibaba T-Head], [C906, C910, C920], [Consumer/server], [Used in Pine64 boards],
  [WCH / GigaDevice / Espressif], [Tiny RV32 cores], [MCUs], [Sub-\$1 chips],
)

Tenstorrent's Ascalon (announced 2023) targets 8-wide decode and 18-wide issue — Apple-M-class single-thread performance on RISC-V is no longer hypothetical.

== Toolchain and Software Ecosystem

```
# GCC: name extensions explicitly
$ riscv64-linux-gnu-gcc -march=rv64gc_zba_zbb_zbc_zbs -mabi=lp64d ...

# RVA23 baseline
$ gcc -march=rva23u64 ...

# QEMU user-mode: try unreleased extensions
$ qemu-riscv64 -cpu max,zvfh=on,v=on,vlen=256 ./app
```

Mature: Linux ($"riscv"$ port since 5.17 stable), $"glibc"$, $"LLVM"$ ($"RVV"$ codegen), Debian ($"riscv64"$ official since Debian 13), Ubuntu, Fedora, Android (since 14).

Less mature: $"JVM"$ ($"RVV"$ vectorization in JDK 23+ only), Go ($"RVV"$ in Go 1.23+ experimental), Rust ($"std::arch::riscv64"$ intrinsics stabilizing).

== Strengths and Critiques

*Strengths:*
- Open $"ISA"$ — no royalties, full implementation freedom.
- Modular — same toolchain across MCU and server.
- $"VLA"$ vectors — single-binary portability across $"VLEN"$.
- Clean encoding — easier to decode, smaller frontends.

*Critiques:*
- *Fragmentation:* before profiles, every chip was a unique combination of extensions.
- *Maturity gap:* ARM has 30 years of microarchitecture; the best $"RVA23"$ silicon today (~2026) lags Apple/Neoverse by a generation.
- *Atomics ordering* historically weak; $"RVWMO"$ (RISC-V Weak Memory Order) is well-specified but porting from $"TSO"$ requires care.
- *Vendor extensions:* T-Head's $"XTheadVector"$ predates ratified $"RVV"$ 1.0 and is *not* compatible; software dispatch needed.

== Comparison Snapshot

#table(
  columns: 4,
  [], [*RISC-V*], [*ARMv8/9*], [*x86-64*],
  [Open ISA], [Yes], [No (licensed)], [No (Intel/AMD only)],
  [Vector model], [VLA ($"RVV"$)], [VLA ($"SVE"$) + fixed ($"NEON"$)], [Fixed ($"AVX"$)],
  [Memory model], [$"RVWMO"$ (weak)], [Weak], [$"TSO"$],
  [Mode bits], [None], [None (AArch64)], [Many (legacy)],
  [Encoding], [32-bit + 16-bit (C)], [32-bit + 16-bit (T32)], [Variable 1-15 B],
)

== Further Reading

Waterman, A. & Asanovic, K. eds. (2019). _The RISC-V Instruction Set Manual, Vol. I: User-Level ISA_. RISC-V International.

Waterman, A. & Asanovic, K. eds. (2024). _The RISC-V Instruction Set Manual, Vol. II: Privileged Architecture_.

Patterson, D.A. & Waterman, A. (2017). _The RISC-V Reader: An Open Architecture Atlas_.

RISC-V International (2024). _RVA23 Profiles Specification_.

Asanovic, K. et al. (2016). "The Berkeley Out-of-Order Machine (BOOM): An Industry-Competitive, Synthesizable, Parameterized RISC-V Processor." UCB Tech Report.

Krste Asanovic, Yunsup Lee, et al. (2014). "The RISC-V Vector ISA: Status and Future Directions." _Hot Chips 28_.

Hennessy, J.L. & Patterson, D.A. (2019). "A New Golden Age for Computer Architecture." _CACM_ 62(2).

#import "../template.typ": xref

= Out-of-Order Execution Deep Dive <out-of-order-deep-dive>

Out-of-order (OoO) execution extracts instruction-level parallelism by dynamically scheduling instructions around long-latency operations. This chapter dissects the structures that make OoO possible: the reorder buffer (ROB), reservation stations (RS), register renaming, and the memory disambiguation machinery.

*See also:* #xref("cpu-architecture", "superscalar", label: "Superscalar and Out-of-Order Execution"), #xref("cpu-architecture", "pipelining", label: "Pipelining"), #xref("cpu-architecture", "branch-prediction", label: "Branch Prediction"), #xref("cpu-architecture", "memory-system", label: "Memory System")

== Why Out-of-Order

Static (compiler) scheduling can only reorder what it can see at compile time. Runtime stalls from cache misses, branch mispredictions, and variable-latency operations are invisible to the compiler. Hardware OoO scheduling sees the actual execution stream and exploits dynamic parallelism that static scheduling cannot reach.

The fundamental observation: most $"IPC"$ loss comes from waiting on a small number of long-latency $mu$ops. If the window of in-flight instructions is large enough, the processor can find independent work to fill the bubbles.

```
Modern OoO window sizes:
  Intel Skylake:        ROB 224,  RS 97,   LDQ 72,  STQ 56
  Intel Sunny Cove:     ROB 352,  RS 160,  LDQ 128, STQ 72
  Intel Golden Cove:    ROB 512,  RS 205,  LDQ 192, STQ 114
  AMD Zen 3:            ROB 256,  RS 160,  LDQ 116, STQ 64
  AMD Zen 4:            ROB 320,  RS 192,  LDQ 136, STQ 64
  Apple M1 Firestorm:   ROB ~630, RS ~354
```

Apple's enormous ROB is the headline of its M-series microarchitecture, enabling deeper memory-level parallelism than any x86 contemporary.

== The Tomasulo Algorithm

Robert Tomasulo (IBM 360/91, 1967) introduced the template every modern OoO core still follows:

1. *Issue:* Decode and rename; allocate a $"ROB"$ entry and a reservation station entry. If a source operand is already in the register file, copy it; otherwise record the producer tag.
2. *Execute:* When all source operands arrive on the common data bus ($"CDB"$), the $"RS"$ dispatches the operation to a free functional unit.
3. *Write result:* The functional unit broadcasts the result on the $"CDB"$ with its tag; any $"RS"$/$"ROB"$ entry awaiting that tag latches the value.
4. *Commit (modern addition):* The $"ROB"$ retires instructions in program order, writing architectural state.

Tomasulo 1967 lacked precise exceptions; Smith & Pleszkun (1988) added the $"ROB"$ to support in-order commit and precise interrupts.

== Reorder Buffer (ROB)

The $"ROB"$ is a circular FIFO indexed by sequence number. Each entry holds:

- Architectural destination register
- Physical register file ($"PRF"$) tag holding the result
- Exception/fault status
- "Completed" bit
- Memory ordering metadata (for loads/stores)

Allocation happens at rename; deallocation happens at retire. The head pointer tracks the oldest non-retired instruction; the tail tracks the next free entry. When the head is complete and fault-free, it retires (up to 4-8 per cycle on modern cores).

*Sizing:* $"ROB"$ size bounds the maximum memory-level parallelism. To hide a 200-cycle DRAM miss at 4 $"IPC"$, you need $approx 800$ in-flight instructions. Real cores fall short, which is why prefetchers and non-blocking caches matter.

== Reservation Stations vs. Unified Scheduler

Two scheduler designs coexist:

#table(
  columns: 3,
  [*Design*], [*Used by*], [*Tradeoff*],
  [Distributed $"RS"$ per port], [AMD Zen (integer side), Apple M-series], [Smaller CAMs, port-specific tuning, harder load balancing],
  [Unified scheduler], [Intel since P6], [Single CAM, easier balancing, larger structure],
)

A $"RS"$ entry holds operation, source tags or values, and destination tag. Each cycle it wakes up entries whose source tags match a $"CDB"$ broadcast; the select logic picks the oldest ready entry per port. Wakeup-select is the critical path that limits issue width.

== Register Renaming

Renaming eliminates write-after-write ($"WAW"$) and write-after-read ($"WAR"$) hazards by mapping the small set of architectural registers (16 GPRs in x86-64) onto a large physical register file (180-400 entries).

```
Architectural code (WAW hazard on rax):
  mov rax, [rbx]      ; rax v1
  add rcx, rax        ; reads v1
  mov rax, [rdx]      ; WAW, but independent of v1

After rename:
  mov p37, [p10]      ; rax -> p37
  add p41, p37        ; rcx -> p41
  mov p52, [p18]      ; rax -> p52  (no false dep)
```

Two physical register file styles:

- *Merged $"PRF"$ (Intel since Sandy Bridge, AMD since Bulldozer, Apple, ARM; also Pentium 4):* one big $"PRF"$ holds both speculative and architectural values; a register alias table ($"RAT"$) maps arch reg to physical tag.
- *Separate retirement register file (Intel P6 through Nehalem):* speculative results live in the $"ROB"$ and copy into the architectural file at retire. Higher port pressure, now historical.

A rename checkpoint snapshots the $"RAT"$ at branches; on misprediction, the $"RAT"$ rolls back to the checkpoint, releasing physical registers younger than the branch.

*Move elimination:* `mov r1, r2` becomes a pure rename — the $"RAT"$ points r1 at r2's physical tag. Zero execution-unit cost. Similarly, `xor eax, eax` is recognized as zero-idiom and produces a zero from the renamer.

== The Common Data Bus and Bypass Network

When a functional unit finishes, it broadcasts `(tag, value)` on the $"CDB"$. Every $"RS"$ entry compares its source tags against the broadcast and latches matching values. With $N$ $"CDB"$s and $M$ $"RS"$ entries, the wakeup logic is an $N times M$ CAM — the dominant power and area cost.

Bypass (forwarding) networks let dependent operations issue back-to-back: a result is forwarded into the next cycle's execute stage at the same time it is written to the $"PRF"$. The bypass network is a full crossbar between functional unit outputs and unit inputs. Cores with wide vectors (e.g., $"AVX"$-512) shrink the bypass to the same FU class to keep wires sane.

== Memory Disambiguation

Loads and stores cannot be freely reordered: a load that bypasses an earlier same-address store would read stale data. The load store queue ($"LSQ"$) tracks all in-flight memory ops.

*Conservative policy:* a load waits for all older stores to compute their addresses. Safe but slow — most loads do not alias older stores.

*Speculative disambiguation* (Intel since Core, AMD since K8): predict no aliasing and issue the load early. If a later store address matches, squash the load and all dependent operations (memory ordering machine clear, $"MOMC"$). Intel's predictor uses a small table indexed by load $"PC"$.

```
Performance counter for memory ordering nukes:
  perf stat -e machine_clears.memory_ordering ./app
```

Excess $"MOMC"$ events ($> 1$%) indicate true aliasing or a confused predictor; padding hot data structures or using `restrict` in C can help.

=== Store-to-Load Forwarding

When a load matches an in-flight store, the store data forwards directly from the store queue, skipping the cache. Latency is typically 4-6 cycles on Intel, 7 on AMD Zen 3 — slower than an L1 hit (4 cycles) because of the queue search.

*Forwarding hazards:* partial overlap, misaligned access, or narrow-store/wide-load patterns cause the load to *stall* until the store retires to L1, costing 10-20 cycles.

```c
// BAD: narrow store, wide load = stall
struct {
    uint32_t a;
    uint32_t b;
} s;
s.a = x;
uint64_t v = *(uint64_t*)&s;   // stall
```

```
perf stat -e ld_blocks.store_forward ./app   # Intel
```

== A Worked Example

```c
for (int i = 0; i < N; ++i) {
    sum += a[i] * b[i];
}
```

Steady-state per iteration in an OoO core:

- `load a[i]`: 4-5 cycle L1 latency; multiple in flight via $"LDQ"$.
- `load b[i]`: parallel with above.
- `mul`: 3-5 cycle latency, pipelined.
- `add` into `sum`: serialized through `sum`'s rename chain — the bottleneck.

The chain on `sum` has $"FADD"$ latency $= 4$ cycles on Skylake $arrow$ peak $0.25$ $"IPC"$ on the reduction. Unrolling with multiple accumulators breaks the chain:

```c
double s0=0, s1=0, s2=0, s3=0;
for (int i = 0; i < N; i += 4) {
    s0 += a[i+0]*b[i+0];
    s1 += a[i+1]*b[i+1];
    s2 += a[i+2]*b[i+2];
    s3 += a[i+3]*b[i+3];
}
double sum = (s0+s1)+(s2+s3);
```

Four independent chains $arrow$ $1.0$ $"IPC"$ on the reduction, limited only by memory bandwidth.

== Memory Ordering Models

OoO permits reordering, but the architectural memory model dictates what is *observable*. x86 implements Total Store Order ($"TSO"$): loads may pass older stores to different addresses, but all other orderings are preserved. ARM and RISC-V implement weakly ordered models that allow far more reordering and require explicit barriers.

#table(
  columns: 4,
  [*ISA*], [*Model*], [*LL/LS/SL/SS reorder allowed*], [*Fence*],
  [x86-64], [$"TSO"$], [SL only], [`MFENCE`],
  [ARMv8], [weak], [all], [`DMB`, `DSB`, `ISB`],
  [RISC-V], [$"RVWMO"$], [all], [`FENCE`],
  [POWER], [weak], [all], [`sync`, `lwsync`],
)

A store buffer ($"STQ"$) is required to make $"TSO"$ performant: stores retire into the buffer and drain to L1 in order, while loads can complete past them.

== Performance Counters

```
# Intel - what is the back-end stalled on?
perf stat -e cycles,instructions,\
  cycle_activity.stalls_l1d_miss,\
  cycle_activity.stalls_l2_miss,\
  cycle_activity.stalls_mem_any,\
  resource_stalls.rob,\
  resource_stalls.rs,\
  machine_clears.memory_ordering \
  ./app

# AMD - ROB/RS fill
perf stat -e ex_no_retire.not_complete,ex_no_retire.load_not_complete ./app
```

`resource_stalls.rob` $> 10$% of cycles: $"ROB"$ is the bottleneck. Either reduce latency (prefetch, smaller working set) or accept the ceiling. `resource_stalls.rs` indicates a hot scheduler port.

== Limits of OoO

- *Window size:* CAM area and power grow superlinearly with $"RS"$ size. Going from $"ROB"$ 224 to 512 cost Intel two generations of process improvements.
- *Wakeup-select critical path:* limits clock frequency.
- *Renamer bandwidth:* typically equals decode width (4-8 $mu$ops/cycle).
- *Memory-level parallelism:* limited by $"MSHR"$ count (10-22 outstanding misses per core).

Diminishing returns are why core counts grew once $"IPC"$ scaling slowed.

== Further Reading

Tomasulo, R.M. (1967). "An Efficient Algorithm for Exploiting Multiple Arithmetic Units." _IBM J. Res. Dev._ 11(1).

Smith, J.E. & Pleszkun, A.R. (1988). "Implementing Precise Interrupts in Pipelined Processors." _IEEE Trans. Computers_ 37(5).

Hennessy, J.L. & Patterson, D.A. (2017). _Computer Architecture: A Quantitative Approach_, 6th ed., Ch. 3.

Intel Corporation (2024). _Intel 64 and IA-32 Architectures Optimization Reference Manual_, Vol. 1, Ch. 3 (Sandy Bridge to Golden Cove pipelines).

AMD (2023). _Software Optimization Guide for AMD Family 19h Processors_ (Zen 4).

Yoaz, A. et al. (1999). "Speculation Techniques for Improving Load Related Instruction Scheduling." _ISCA '99_.

Sha, T., Martin, M.M.K., Roth, A. (2005). "Scalable Store-Load Forwarding via Store Queue Index Prediction." _MICRO-38_.

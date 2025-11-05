#set document(title: "Computer Architecture", author: "John Law")
#set page(
  paper: "us-letter",
  margin: (x: 1cm, y: 1cm),
  header: [
    #smallcaps[_Computer Architecture Notes by #link("https://github.com/poyea")[\@poyea]_]
    #h(0.5fr)
    #emph(text[#datetime.today().display()])
    #h(0.5fr)
    #emph(link("https://github.com/poyea/grimoire")[poyea/grimoire::architecture])
  ],
  footer: context [
    #align(right)[#counter(page).display("1")]
  ]
)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")

#align(center)[
  #block(
    fill: luma(245),
    inset: 10pt,
    width: 80%,
    text(size: 9pt)[
      #align(center)[
        #text(size: 24pt, weight: "bold")[Computer Architecture]
      ]
    ]
  )
]

#outline(
  title: "Table of Contents",
  depth: 2,
)

#emph[Enjoy.]

#pagebreak()

= Introduction

Modern computer architecture represents decades of optimization at every level: from transistor physics to instruction sets to cache hierarchies. Understanding these layers is essential for writing high-performance code.

*Scope of this reference:*

- *Instruction execution:* Pipelining, superscalar, out-of-order execution, speculation
- *Memory hierarchy:* Registers, caches (L1/L2/L3), DRAM, virtual memory
- *Parallelism:* Instruction-level (ILP), data-level (SIMD), thread-level (multicore)
- *Modern features:* Branch prediction, prefetching, cache coherence, SMT
- *Performance analysis:* Cycle counts, bottlenecks, measurement techniques

*Primary architecture focus:* x86-64 (Intel, AMD) with comparisons to ARM where relevant.

*Intended audience:* Systems programmers, compiler writers, performance engineers, computer scientists seeking deep understanding of how code executes on modern processors.

*Notation:*
- Cycle counts are typical for modern processors (2020+: Intel Skylake/Ice Lake, AMD Zen 2/3)
- Latencies measured in CPU cycles unless specified
- Assembly syntax: Intel flavor (destination first)

== Historical Context

*Moore's Law era (1970-2005):* Performance doubled every 18 months via higher clock speeds.
- 1970s: 1-10 MHz, simple in-order pipelines
- 1990s: 100-500 MHz, superscalar out-of-order execution
- 2005: 3-4 GHz, deep pipelines (Pentium 4: 31 stages)

*Power wall (2005-present):* Clock speeds plateaued due to power density limits ($P ∝ C V^2 f$).

*Multicore era:* Performance via parallelism instead of frequency.
- 2005: Dual-core mainstream
- 2010: Quad-core common
- 2023: 8-16 cores consumer, 64-128 cores server

*Key insight:* Free lunch is over [Sutter 2005]. Single-threaded performance improves slowly (~3-5% per year). Parallelism is mandatory.

#pagebreak()

#include "architecture/cpu-fundamentals.typ"
#pagebreak()

#include "architecture/pipelining.typ"
#pagebreak()

#include "architecture/superscalar.typ"
#pagebreak()

#include "architecture/branch-prediction.typ"
#pagebreak()

#include "architecture/cache-hierarchy.typ"
#pagebreak()

#include "architecture/memory-system.typ"
#pagebreak()

#include "architecture/virtual-memory.typ"
#pagebreak()

#include "architecture/simd.typ"
#pagebreak()

#include "architecture/multicore.typ"
#pagebreak()

#include "architecture/performance-analysis.typ"
#pagebreak()

= Conclusion

Modern processors are staggeringly complex: billions of transistors, 10+ stage pipelines, multiple execution units, multi-level caches, speculative execution, and elaborate branch predictors.

*Key takeaways:*

*Instruction execution:*
- Pipelining enables 1 instruction/cycle throughput (despite multi-cycle latency)
- Superscalar: Multiple instructions execute simultaneously (3-4 per cycle typical)
- Out-of-order execution hides latencies by reordering independent instructions
- Branch misprediction costs 10-20 cycles (pipeline flush)

*Memory hierarchy:*
- Register access: 0 cycles (operand available)
- L1 cache: ~4 cycles (best case)
- L2 cache: ~12 cycles
- L3 cache: ~40-80 cycles (shared across cores)
- DRAM: ~200 cycles (~60-100ns)
- Cache miss = 50x slower than L1 hit

*Virtual memory:*
- Page tables map virtual → physical addresses (4-level walk = 4 DRAM accesses)
- TLB caches translations: ~1000-2000 entries, miss costs ~100 cycles
- Huge pages (2MB/1GB) reduce TLB pressure 512x/262144x

*Parallelism:*
- ILP (instruction-level): 2-4 IPC typical, limited by dependencies
- SIMD (data-level): 4-16x speedup for vector operations
- Multicore (thread-level): Linear scaling if no contention

*Performance optimization hierarchy:*

1. *Algorithm:* $O(n^2)$ → $O(n log n)$ (10-100x improvement)
2. *Cache locality:* Sequential access vs random (2-10x)
3. *Branch prediction:* Predictable branches (2-3x)
4. *SIMD:* Vectorize hot loops (4-8x)
5. *Instruction-level:* Minimize dependencies (1.5-2x)

*Modern trends:*

*Specialization:* GPUs (SIMT), TPUs (matrix multiply), FPGAs (custom datapaths) for specific workloads.

*Heterogeneous computing:* ARM big.LITTLE, Intel Performance/Efficiency cores - different cores for different power/performance tradeoffs.

*Security vs performance:* Spectre/Meltdown mitigations cost 5-30% performance depending on workload [Gruss et al. 2019].

*Measurement is essential:* Performance counters (perf, VTune, AMD uProf) reveal bottlenecks. Intuition fails at this complexity level.

*Final thought:* "Premature optimization is the root of all evil" [Knuth 1974], but *informed* optimization requires deep architectural understanding. Profile first, optimize second, measure always.

== Further Reading

*Canonical textbooks:*

Hennessy, J.L. & Patterson, D.A. (2017). Computer Architecture: A Quantitative Approach (6th ed.). Morgan Kaufmann.

Patterson, D.A. & Hennessy, J.L. (2020). Computer Organization and Design: The Hardware/Software Interface (6th ed.). Morgan Kaufmann.

*Practical optimization:*

Fog, A. (2023). Optimizing Software in C++. Technical University of Denmark. https://www.agner.org/optimize/

Intel Corporation (2023). Intel 64 and IA-32 Architectures Optimization Reference Manual.

*Research papers - classics:*

Tomasulo, R.M. (1967). "An Efficient Algorithm for Exploiting Multiple Arithmetic Units." IBM Journal of Research and Development 11(1): 25-33.

Smith, J.E. (1981). "A Study of Branch Prediction Strategies." ISCA '81.

Jouppi, N.P. (1990). "Improving Direct-Mapped Cache Performance by the Addition of a Small Fully-Associative Cache and Prefetch Buffers." ISCA '90.

*Security implications:*

Kocher, P. et al. (2019). "Spectre Attacks: Exploiting Speculative Execution." S&P '19.

Lipp, M. et al. (2018). "Meltdown: Reading Kernel Memory from User Space." USENIX Security '18.

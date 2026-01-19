#import "template.typ": project

#project("GPU Architecture")[
  #align(center)[
  #block(
    fill: luma(245),
    inset: 10pt,
    width: 80%,
    text(size: 9pt)[
      #align(center)[
        #text(size: 24pt, weight: "bold")[GPU Architecture & Performance Engineering]
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

Modern Graphics Processing Units (GPUs) have evolved from fixed-function graphics accelerators into massively parallel processors capable of executing thousands of threads simultaneously. Understanding GPU architecture is essential for high-performance computing, machine learning, and any workload requiring data-parallel processing.

*Scope of this reference:*

- *Architecture fundamentals:* GPU vs CPU design, compute units, memory hierarchy
- *Execution model:* SIMT, warps, thread hierarchy, divergence
- *Memory system:* Registers, shared memory, global memory, caching
- *Compute units:* CUDA cores, Tensor Cores, RT Cores
- *Performance optimization:* Coalescing, occupancy, kernel tuning
- *Profiling:* Nsight tools, metrics, debugging

*Primary focus:* NVIDIA CUDA with comparisons to AMD ROCm/HIP where relevant.

*Intended audience:* Systems programmers, ML engineers, HPC developers, and anyone seeking deep understanding of GPU execution.

*Notation:*
- Performance figures based on modern GPUs (2020+: NVIDIA Ampere/Ada, AMD RDNA 3)
- CUDA terminology used primarily; AMD equivalents noted
- Code examples in CUDA C++ unless otherwise specified

== Why GPUs?

*The parallel computing imperative:* Single-thread CPU performance has plateaued since ~2005 due to power and frequency limits. Performance gains now require parallelism.

*GPU approach:* Thousands of simple cores executing the same instruction on different data (SIMT). Optimized for throughput over latency.

```
                      CPU                          GPU
──────────────────────────────────────────────────────────────────────
Cores                 8-64 complex                 1000s simple
Clock                 3-5 GHz                      1.5-2.5 GHz
Cache per core        2-4 MB                       ~100 KB
Latency hiding        OoO, speculation             Massive threading
Peak FP32             ~2 TFLOPS                    ~80 TFLOPS
Memory BW             ~100 GB/s                    ~1000 GB/s
Best for              Latency-sensitive            Throughput-oriented
```

*When GPUs excel:*
- Data-parallel problems (same operation on many elements)
- High arithmetic intensity (compute per memory access)
- Large problem sizes (amortize launch overhead)
- Regular memory access patterns

*When GPUs struggle:*
- Sequential dependencies
- Small problem sizes
- Irregular access patterns
- Heavy branching

== Modern GPU Landscape

*NVIDIA:*
- Consumer: GeForce RTX 40 series (Ada Lovelace)
- Professional: RTX A6000, L40
- Datacenter: A100, H100, H200 (Hopper), B100, B200 (Blackwell)

*AMD:*
- Consumer: Radeon RX 7000 series (RDNA 3)
- Professional: Radeon Pro W7900
- Datacenter: Instinct MI300A/X (CDNA 3)

*Intel:*
- Consumer: Arc A-series (Alchemist/Battlemage)
- Datacenter: Data Center GPU Max (Ponte Vecchio)

*Key specifications (2024):*

```
GPU              FP32 TFLOPS  Memory      Bandwidth   TDP
──────────────────────────────────────────────────────────────────────
RTX 4090         82.6         24 GB G6X   1008 GB/s   450W
H100 SXM         67           80 GB HBM3  3.35 TB/s   700W
MI300X           163          192 GB HBM3 5.3 TB/s    750W
```

== Document Structure

This document is organized to build understanding progressively:

1. *GPU Fundamentals:* Architecture overview, historical context, programming models
2. *Execution Model:* SIMT, warps, threads, divergence, synchronization
3. *Memory Hierarchy:* Registers, shared memory, caches, global memory
4. *Compute Architecture:* CUDA cores, Tensor Cores, specialized units
5. *Performance Optimization:* Systematic optimization techniques
6. *Profiling and Debugging:* Measurement tools and methodologies

Each section includes practical code examples, specific hardware numbers, and references to official documentation for further study.

#pagebreak()

#include "gpu-architecture/gpu-fundamentals.typ"
#pagebreak()

#include "gpu-architecture/execution-model.typ"
#pagebreak()

#include "gpu-architecture/memory-hierarchy.typ"
#pagebreak()

#include "gpu-architecture/compute-architecture.typ"
#pagebreak()

#include "gpu-architecture/performance-optimization.typ"
#pagebreak()

#include "gpu-architecture/profiling.typ"
#pagebreak()

= Conclusion

Modern GPUs are sophisticated parallel processors with unique architectural characteristics. Writing efficient GPU code requires understanding the execution model, memory hierarchy, and optimization principles covered in this reference.

*Key takeaways:*

*Execution model:*
- Warps of 32 threads execute in lockstep (SIMT)
- Divergence serializes execution within a warp
- Thousands of threads hide memory latency
- Block and grid organization maps to hardware

*Memory hierarchy:*
- Registers: Fastest, limited (255 per thread)
- Shared memory: Fast, user-managed, block-scoped (~100 KB per SM)
- L1/L2 cache: Hardware-managed, limited
- Global memory: High bandwidth (~1 TB/s), high latency (~400 cycles)
- Coalesced access is critical for performance

*Compute units:*
- CUDA cores: Scalar FP32/INT32 operations
- Tensor Cores: Matrix multiply-accumulate (16×16 tiles), 10-20× faster for ML
- RT Cores: Hardware ray tracing acceleration
- SFUs: Transcendental functions (sin, cos, exp)

*Optimization hierarchy:*

1. *Algorithm:* Choose parallel-friendly algorithms
2. *Memory:* Coalesced access, shared memory caching, minimize transfers
3. *Occupancy:* Balance threads, registers, shared memory
4. *Compute:* ILP, loop unrolling, fast math
5. *Low-level:* Warp primitives, assembly tuning

*Performance mindset:*

```
Profile first, optimize second.
Measure everything.
Understand your bottleneck before optimizing.
Small improvements compound.
```

*Common optimization wins:*
- AoS → SoA: 2-4× for memory-bound kernels
- Shared memory tiling: 5-10× for matrix operations
- Coalescing fixes: 2-10× improvement
- Kernel fusion: 2-3× by reducing memory round-trips
- Tensor Core usage: 10-20× for matrix math

*Emerging trends:*

*Lower precision:* FP8, INT4 for inference with minimal accuracy loss.

*Structured sparsity:* 2:4 sparsity patterns for 2× Tensor Core throughput.

*Larger GPU memory:* HBM3 enabling 192+ GB for large models.

*Chiplet designs:* AMD MI300, NVIDIA Grace-Hopper combining GPU and CPU.

*Programming abstractions:* Higher-level libraries (cuBLAS, cuDNN, Triton) hiding hardware complexity while maintaining performance.

== Quick Reference

*Thread hierarchy:*
```
Grid → Blocks → Warps → Threads
     (millions)  (32 each)  (32 per warp)
```

*Memory spaces:*
```
Register < Shared < L1 < L2 < Global < Host
(0 cyc)   (20 cyc) (30 cyc) (200 cyc) (400 cyc) (10 µs)
```

*Key limits (Ada Lovelace):*
```
Threads per block:         1024
Warps per SM:              48
Registers per SM:          65536
Shared memory per SM:      100 KB (configurable)
Blocks per SM:             24
```

*Critical formulas:*
```
Thread ID = blockIdx.x * blockDim.x + threadIdx.x
Occupancy = Active warps / Max warps
Bandwidth = Bytes transferred / Time
Arithmetic Intensity = FLOPs / Bytes
```

== Further Reading

*Official documentation:*

NVIDIA Corporation (2024). CUDA C++ Programming Guide. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

NVIDIA Corporation (2024). CUDA C++ Best Practices Guide. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

AMD (2024). HIP Programming Guide. https://rocm.docs.amd.com/projects/HIP/

*Textbooks:*

Kirk, D.B. & Hwu, W.W. (2016). Programming Massively Parallel Processors: A Hands-on Approach (3rd ed.). Morgan Kaufmann.

Sanders, J. & Kandrot, E. (2010). CUDA by Example: An Introduction to General-Purpose GPU Programming. Addison-Wesley.

Hennessy, J.L. & Patterson, D.A. (2019). Computer Architecture: A Quantitative Approach (6th ed.). Morgan Kaufmann. Chapter 4.

*Architecture whitepapers:*

NVIDIA Corporation (2024). NVIDIA Ada GPU Architecture. https://images.nvidia.com/aem-dam/Solutions/geforce/ada/

NVIDIA Corporation (2022). NVIDIA H100 Tensor Core GPU Architecture. https://resources.nvidia.com/en-us-tensor-core

AMD (2024). RDNA 3 Instruction Set Architecture Reference. https://developer.amd.com/

*Performance analysis:*

Volkov, V. (2010). "Better Performance at Lower Occupancy." GPU Technology Conference.

Williams, S., Waterman, A., & Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures." Communications of the ACM.

*Research:*

Jia, Z. et al. (2018). "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking." arXiv:1804.06826.

Mei, X. & Chu, X. (2017). "Dissecting GPU Memory Hierarchy Through Microbenchmarking." IEEE TPDS.
]

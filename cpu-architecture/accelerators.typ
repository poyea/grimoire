#import "../template.typ": xref

= Domain-Specific Accelerators

When Dennard scaling stopped, perf/W gains shifted from general-purpose cores to domain-specific architectures ($"DSA"$s): chips that trade generality for orders-of-magnitude efficiency on a narrow workload class. This chapter surveys the dominant accelerator families.

*See also:* #xref("cpu-architecture", "simd", label: "SIMD"), #xref("cpu-architecture", "multicore", label: "Multicore"), #xref("cpu-architecture", "memory-system", label: "Memory System"), #xref("cpu-architecture", "interconnects", label: "Interconnects")

== Why Specialize

The "energy gap" between a tuned $"ASIC"$ and a $"CPU"$ executing the same algorithm is typically 100-1000x in perf/W [Horowitz 2014]. Sources of the gap:

- *Instruction overhead:* fetch, decode, rename, schedule cost more than a 32-bit multiply.
- *Register file ports:* large general $"PRF"$ burns power on every access.
- *Data movement:* off-chip DRAM bits cost 100-1000x an ALU op.
- *Precision flexibility:* INT8 multiply is ~16x cheaper than FP32 in energy and area.

A $"DSA"$ amortizes control across many ALUs, keeps operands on-chip, and matches data type to workload.

== Systolic Arrays

A systolic array (Kung & Leiserson 1979) is a 2-D mesh of simple processing elements ($"PE"$s) that pump data rhythmically: each $"PE"$ reads from one or two neighbors, computes a multiply-accumulate ($"MAC"$), and forwards data to the next neighbor. For matrix multiply $C = A times B$, the array achieves near-100% utilization with only nearest-neighbor wires.

```
Output-stationary systolic GEMM (3x3 example):

       b00 b01 b02
       b10 b11 b12
       b20 b21 b22
        |   |   |
a00 a01 a02 ---> [PE]-[PE]-[PE]
a10 a11 a12 ---> [PE]-[PE]-[PE]
a20 a21 a22 ---> [PE]-[PE]-[PE]

Each PE accumulates one C[i,j]; A streams left-to-right, B top-to-bottom.
```

Three common dataflows:

#table(
  columns: 3,
  [*Dataflow*], [*Stationary in PE*], [*Used by*],
  [Output-stationary ($"OS"$)], [Partial sums of C], [ShiDianNao],
  [Weight-stationary ($"WS"$)], [B (weights)], [Google TPU, most NPUs],
  [Row-stationary ($"RS"$)], [Mixed], [MIT Eyeriss],
)

== Google TPU

*TPU v1 (2015):* 256$times$256 INT8 $"MAC"$ array, 65k $"MAC"$s, 92 $"TOPS"$ peak, 75 W. Weight-stationary; 24 MB on-chip "unified buffer" feeds the array. Built for inference; no training.

*TPU v2/v3 (2017/18):* BF16 (16-bit brain-float: 8-bit exponent, 7-bit mantissa) added training. 128$times$128 $"MXU"$s per core: one in v2, two in v3.

*TPU v4 (2021):* sparsecore, optical interconnect (OCS) reconfigures pods.

*TPU v5e / v5p (2023/24):* BF16/INT8, INT4 inference; v5p has 95 GB $"HBM2e"$ and 8960-chip pods.

The architectural lesson: a small ISA (vector load, vector store, $"MATMUL"$, activation) and a *systolic functional unit* removed most of the control overhead that bottlenecks GPUs.

== NPUs in Consumer Silicon

NPUs ("neural processing units") have shipped in every flagship mobile $"SoC"$ since ~2017 and are now appearing in laptops:

#table(
  columns: 4,
  [*Vendor*], [*Block*], [*Peak*], [*Notes*],
  [Apple], [Neural Engine (16-core in M4)], [38 $"TOPS"$ INT8], [Shared $"UMA"$],
  [Qualcomm], [Hexagon Tensor (X Elite)], [45 $"TOPS"$], [],
  [Intel], [$"NPU"$ (Meteor/Lunar Lake)], [11 / 48 $"TOPS"$], [$"VPU"$ heritage],
  [AMD], [XDNA / XDNA 2 (Ryzen AI)], [16 / 50 $"TOPS"$], [Xilinx $"AIE"$],
  [Google], [Edge $"TPU"$ / Tensor], [4 $"TOPS"$], [Pixel phones],
)

The "Copilot+ PC" threshold is 40 $"TOPS"$ for local LLM-class workloads.

== Dataflow Accelerators

Whereas systolic arrays are rigid grids, dataflow processors map an *arbitrary* compute graph onto a sea of $"PE"$s and route operands along physical wires.

=== Cerebras Wafer-Scale Engine

The $"WSE"$-3 is a single die that fills a 300 mm wafer: 900,000 cores, 44 GB on-die $"SRAM"$, 21 PB/s aggregate memory bandwidth, no DRAM. Each $"PE"$ has its own 48 kB of $"SRAM"$; cores communicate through a 2-D mesh. The entire model lives on-chip, eliminating the off-chip bottleneck — at the cost of a million-dollar accelerator with bespoke packaging.

=== Graphcore IPU

The $"IPU"$ ($"Bow"$, $"Mk2"$, Colossus) is a $"BSP"$-style ("Bulk Synchronous Parallel") processor: 1,472 tiles, each with its own $"SRAM"$ (~624 KB) and 6 threads. No shared memory, no cache; explicit message-passing via the on-die exchange. The compiler schedules computation and communication into "supersteps" separated by global barriers. Optimized for fine-grained sparsity that GPUs handle poorly.

=== Groq LPU / TSP

The Tensor Streaming Processor abandons cache hierarchy entirely: a single 320-way SIMD pipeline executes a *fully compiler-scheduled* program. Every cycle's worth of every $"FU"$ is statically reserved. Deterministic latency makes Groq the throughput leader for transformer inference at small batch sizes (~300 tokens/s on Llama-2 70B in early 2024). Limitation: model must fit in on-chip $"SRAM"$ (230 MB per chip $arrow$ multi-chip).

== SambaNova RDU

The Reconfigurable Dataflow Unit ($"RDU"$, $"SN10"$/$"SN40L"$) is a coarse-grained reconfigurable array ($"CGRA"$): a fabric of pattern compute units ($"PCU"$s) and pattern memory units ($"PMU"$s) that the compiler wires together per kernel. $"SN40L"$ adds 1.5 TB of attached $"HBM"$ + DDR addressable in a single tier, targeting trillion-parameter inference on a single node.

== Tensor Cores (in GPUs)

Tensor Cores are systolic-flavored $"MMA"$ units bolted onto $"SM"$s:

#table(
  columns: 3,
  [*Architecture*], [*Year*], [*Capability*],
  [Volta V100], [2017], [4$times$4$times$4 FP16, FP32 accum],
  [Turing T4], [2018], [INT8, INT4, INT1],
  [Ampere A100], [2020], [TF32, BF16, FP64; 2:4 sparsity],
  [Hopper H100], [2022], [FP8 (E4M3, E5M2); Transformer Engine],
  [Blackwell B200], [2024], [FP4, FP6; second-gen Transformer Engine],
)

Programmer interface: `mma.sync` $"PTX"$ instructions, `wmma::` C++ templates, or higher-level libraries (cuBLAS, $"CUTLASS"$, Triton).

== FPGAs as Accelerators

FPGAs trade peak throughput for reconfigurability. Microsoft's Project Catapult (Azure) deployed $"FPGA"$s for Bing ranking and SmartNIC offload before NPUs were mainstream. AMD's acquisition of Xilinx (2022) and Intel's spinoff of Altera (2024) reflect the segment maturing into "adaptable $"SoC"$s" ($"Versal"$, $"Agilex"$) that pair $"FPGA"$ fabric with hardened $"AI"$ engines.

== Programming Model Convergence

Across families, kernel languages have converged on a few patterns:

```python
# Triton: dataflow-friendly tiles, hardware-agnostic
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # Each program computes a BLOCK_M x BLOCK_N tile of C.
    a_block = tl.load(A + ...)
    b_block = tl.load(B + ...)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        acc += tl.dot(a_block, b_block)
        a_block = tl.load(A + ...)
        b_block = tl.load(B + ...)
    tl.store(C + ..., acc)
```

Vendor backends (NVIDIA, AMD, Intel $"XPU"$, $"AWS"$ Trainium) compile the same Triton/MLIR/StableHLO down to native instructions. $"MLIR"$ has become the dominant intermediate representation across $"XLA"$, Triton, IREE, and Mojo.

== Choosing an Accelerator

#table(
  columns: 3,
  [*Workload*], [*Sweet spot*], [*Reason*],
  [Training (large, dense)], [GPU (H100/B200) or TPU], [Mature ecosystem, $"NVLink"$/$"OCS"$],
  [Inference (latency-critical)], [Groq, custom $"ASIC"$], [Deterministic dataflow],
  [Inference (throughput, batched)], [GPU], [Tensor Cores + $"vLLM"$ scheduling],
  [Edge / mobile], [Apple Neural Engine, Hexagon, Coral], [perf/W],
  [Sparse / graph], [Graphcore $"IPU"$, $"CGRA"$], [Fine-grained communication],
  [Wafer-scale models], [Cerebras $"WSE"$], [On-die residency],
  [Custom protocols / network], [FPGA], [Reconfigurable I/O],
)

== Limits and Future Direction

Accelerators face their own scaling walls:

- *Memory bandwidth* dominates cost; $"HBM3E"$ and 3-D stacking are mandatory.
- *Interconnect scaling*: chip-to-chip links ($"NVLink"$, $"UALink"$, $"InfinityFabric"$, optical $"OCS"$) define cluster topology and cost.
- *Numerical formats* (FP8, FP6, FP4, $"MXFP"$): every halving of precision $approx$ 2$times$ perf/W if accuracy holds.
- *Compiler maturity*: hardware divergence outruns frameworks; vendor lock-in is the practical bottleneck.

== Further Reading

Jouppi, N.P. et al. (2017). "In-Datacenter Performance Analysis of a Tensor Processing Unit." _ISCA '17_.

Jouppi, N.P. et al. (2023). "TPU v4: An Optically Reconfigurable Supercomputer for ML." _ISCA '23_.

Chen, Y.-H., Krishna, T., Emer, J., Sze, V. (2017). "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep Convolutional Neural Networks." _IEEE JSSC_ 52(1).

Kung, H.T. & Leiserson, C.E. (1979). "Systolic Arrays for VLSI." _Sparse Matrix Proc._

Sze, V. et al. (2020). _Efficient Processing of Deep Neural Networks_. Morgan & Claypool.

Cerebras Systems (2024). _WSE-3 Whitepaper_.

Abts, D. et al. (2020). "Think Fast: A Tensor Streaming Processor (TSP) for Accelerating Deep Learning Workloads." _ISCA '20_.

Hennessy, J.L. & Patterson, D.A. (2019). "A New Golden Age for Computer Architecture." _CACM_ 62(2).

= GPU Fundamentals

Modern GPUs are massively parallel processors designed for throughput-oriented computing. Unlike CPUs optimized for latency, GPUs maximize aggregate work per unit time through thousands of concurrent threads.

*See also:* Execution Model (for SIMT programming), Memory Hierarchy (for GPU memory system), Compute Architecture (for hardware details)

== CPU vs GPU Design Philosophy

*CPU design goals:* Minimize latency for single-threaded workloads.

```
CPU Core (Intel Core i9-13900K):
- 8 performance cores, 16 efficiency cores
- 68 MB cache (L2+L3)
- Deep OoO execution (512 ROB entries)
- Complex branch prediction
- Optimized for serial execution, low latency
```

*GPU design goals:* Maximize throughput for parallel workloads.

```
GPU (NVIDIA RTX 4090):
- 16,384 CUDA cores in 128 SMs
- 72 MB L2 cache
- No OoO execution (in-order within warp)
- Simple branch handling (predication/masking)
- Optimized for parallel execution, high throughput
```

*Fundamental tradeoff:* CPUs use transistors for latency-hiding mechanisms (large caches, OoO, speculation). GPUs use transistors for more compute units and rely on massive parallelism to hide latency.

```
                 CPU                    GPU
                 ┌───────────┐          ┌─────────────────────────┐
                 │           │          │ SM │ SM │ SM │ SM │ SM  │
                 │   Core    │          │────│────│────│────│────│
                 │  ┌─────┐  │          │ SM │ SM │ SM │ SM │ SM  │
                 │  │Cache│  │          │────│────│────│────│────│
                 │  └─────┘  │          │ SM │ SM │ SM │ SM │ SM  │
                 │  OoO/BP   │          │────│────│────│────│────│
                 └───────────┘          │ SM │ SM │ SM │ SM │ SM  │
                                        └─────────────────────────┘
Transistors for: Control logic          Transistors for: ALUs
Cache per core:  ~2-4 MB                 Cache per SM: ~256 KB
Latency hiding:  OoO, speculation        Latency hiding: Thread switching
Threads:         2-32                    Threads: 10,000-100,000
```

== Historical Evolution

*1990s - Fixed-function graphics:* Dedicated hardware for 3D transformations and rasterization. No programmability.

*2001 - Programmable shaders:* NVIDIA GeForce 3 introduced vertex and pixel shaders. Limited instruction sets, no branching initially.

*2006 - Unified shaders (CUDA):* NVIDIA Tesla architecture (G80/GeForce 8800) introduced unified shader cores programmable with CUDA. General-purpose GPU computing (GPGPU) becomes viable.

*2010s - HPC adoption:* GPUs dominate supercomputers. Fermi (2010), Kepler (2012), Maxwell (2014), Pascal (2016), Volta (2017).

*2020s - AI acceleration:* Tensor Cores for matrix operations. Ampere (2020), Ada Lovelace (2022), Hopper (2022), Blackwell (2024).

```
Year  Architecture       Key Features
2006  Tesla (G80)        First unified shaders, CUDA 1.0
2008  Tesla (GT200)      Double precision, improved memory
2010  Fermi             L1/L2 cache, ECC, 512 CUDA cores
2012  Kepler            Dynamic parallelism, Hyper-Q
2014  Maxwell           Power efficiency, unified memory
2016  Pascal            NVLink, HBM2 support, 3584 cores
2017  Volta             Tensor Cores, independent thread scheduling
2020  Ampere            3rd gen Tensor Cores, sparsity, 6912 cores
2022  Hopper            Transformer Engine, 18,432 cores
2024  Blackwell         2nd gen Transformer Engine, 21,760 cores
```

== GPU Architecture Overview

*NVIDIA Ada Lovelace (RTX 4090) example:*

```
┌─────────────────────────────────────────────────────────────────┐
│                       GPU Die (AD102)                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Graphics Processing Clusters (GPCs)         │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐        │    │
│  │  │  GPC 0  │ │  GPC 1  │ │  GPC 2  │ │  GPC 3  │ ...    │    │
│  │  │ 16 SMs  │ │ 16 SMs  │ │ 16 SMs  │ │ 16 SMs  │        │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘        │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    L2 Cache (72 MB)                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │      Memory Controllers (12 × 32-bit = 384-bit bus)     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                   ┌──────────┴──────────┐
                   │   GDDR6X (24 GB)    │
                   │   1008 GB/s         │
                   └─────────────────────┘

Hierarchy:
- GPU → 7-12 GPCs (Graphics Processing Clusters)
- GPC → 6-16 SMs (Streaming Multiprocessors)
- SM → 128 CUDA cores + 4 Tensor Cores + RT Core
```

*AMD RDNA 3 (RX 7900 XTX) equivalent:*

```
┌────────────────────────────────────────────────────────┐
│                    GPU Die (Navi 31)                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │          Shader Engines (6 total)                 │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐    │  │
│  │  │    SE 0    │ │    SE 1    │ │    SE 2    │    │  │
│  │  │  16 WGPs   │ │  16 WGPs   │ │  16 WGPs   │    │  │
│  │  └────────────┘ └────────────┘ └────────────┘    │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Infinity Cache (96 MB)               │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘

Terminology mapping:
- NVIDIA SM ≈ AMD Compute Unit (CU) ≈ AMD Work Group Processor (WGP)
- NVIDIA CUDA core ≈ AMD Stream Processor
- NVIDIA warp (32 threads) ≈ AMD wavefront (32/64 threads)
```

== Streaming Multiprocessor (SM) Architecture

*NVIDIA Ada Lovelace SM:*

```
┌──────────────────────────────────────────────────────────┐
│                    Streaming Multiprocessor               │
│  ┌─────────────────────────────────────────────────────┐ │
│  │            Instruction Cache (128 KB)                │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌─────────────────┐    ┌─────────────────┐             │
│  │   Warp Scheduler │    │   Warp Scheduler │             │
│  │   (32 warps)     │    │   (32 warps)     │             │
│  └────────┬────────┘    └────────┬────────┘             │
│           │                      │                       │
│  ┌────────┴──────────────────────┴────────┐             │
│  │              Processing Block           │             │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐  │             │
│  │  │16 FP32  │ │16 FP32  │ │16 INT32 │  │             │
│  │  │ units   │ │ units   │ │ units   │  │             │
│  │  └─────────┘ └─────────┘ └─────────┘  │             │
│  │  ┌─────────┐ ┌─────────┐              │             │
│  │  │ 1 Tensor│ │ SFU (4) │              │             │
│  │  │ Core    │ │         │              │             │
│  │  └─────────┘ └─────────┘              │             │
│  └─────────────────────────────────────────┘             │
│  × 4 Processing Blocks = 128 CUDA cores, 4 Tensor Cores  │
│  ┌─────────────────────────────────────────────────────┐ │
│  │         Shared Memory / L1 Cache (128 KB)            │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │                Register File (256 KB)                │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘

Per SM:
- 128 CUDA cores (FP32/INT32)
- 4 Tensor Cores (4th generation)
- 1 RT Core
- 4 SFUs (transcendentals: sin, cos, sqrt, rcp)
- 128 KB configurable shared memory / L1 cache
- 256 KB register file (65,536 × 32-bit registers)
- Up to 64 warps (2048 threads) resident
```

== Compute Capability and ISA

NVIDIA GPUs expose capabilities through Compute Capability versions, which determine available features and instruction set.

```
Compute Capability   Architecture    Key Features
3.0                  Kepler          Dynamic parallelism, shuffle
5.0                  Maxwell         FP16 storage
6.0                  Pascal          FP16 compute, unified memory
7.0                  Volta           Tensor Cores, independent threads
7.5                  Turing          RT Cores, INT8 inference
8.0                  Ampere          TF32, sparsity, async copy
8.6                  Ampere (GA10x)  Consumer cards, 3rd gen Tensor
8.9                  Ada Lovelace    4th gen Tensor, FP8, SER
9.0                  Hopper          Transformer Engine, DPX
10.0                 Blackwell       5th gen Tensor, FP4
```

*PTX (Parallel Thread Execution):* Virtual ISA for NVIDIA GPUs.

```
// PTX assembly example
.visible .entry add_kernel(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c
)
{
    .reg .u32 %tid;
    .reg .u64 %addr_a, %addr_b, %addr_c;
    .reg .f32 %val_a, %val_b, %val_c;

    // Get thread ID
    mov.u32 %tid, %tid.x;

    // Load from a[tid]
    cvt.u64.u32 %addr_a, %tid;
    shl.b64 %addr_a, %addr_a, 2;        // × 4 (float size)
    add.u64 %addr_a, %addr_a, %a;
    ld.global.f32 %val_a, [%addr_a];

    // Load from b[tid]
    ld.global.f32 %val_b, [%addr_b];

    // c[tid] = a[tid] + b[tid]
    add.f32 %val_c, %val_a, %val_b;
    st.global.f32 [%addr_c], %val_c;

    ret;
}
```

*SASS (Shader Assembly):* Native machine code (architecture-specific).

```bash
# Disassemble CUDA binary
cuobjdump -sass program.cubin

# Example SASS (Ampere)
IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]   ; Load parameter
LDG.E R2, [R4.64]                         ; Global load
FADD R3, R2, R5                           ; Float add
STG.E [R6.64], R3                         ; Global store
```

== Performance Metrics

*Theoretical peak performance (RTX 4090):*

```
FP32 TFLOPS = CUDA cores × 2 × clock speed
            = 16,384 × 2 × 2.52 GHz
            = 82.6 TFLOPS

FP16 (with Tensor Cores):
            = 660.6 TFLOPS (sparse)
            = 330.3 TFLOPS (dense)

Memory bandwidth = bus width × memory clock × 2 (DDR)
                 = 384-bit × 21 Gbps
                 = 1008 GB/s
```

*Operational intensity:* Ratio of compute to memory operations.

```
Operational Intensity = FLOPs / Bytes transferred

Example: Vector addition c[i] = a[i] + b[i]
- Operations: 1 FLOP per element
- Data: 3 × 4 bytes = 12 bytes per element (load a, load b, store c)
- Intensity: 1/12 = 0.083 FLOPs/byte

This is memory-bound:
- RTX 4090: 82.6 TFLOPS / 1008 GB/s = 82 FLOPs/byte needed for compute-bound
- Actual: 0.083 FLOPs/byte << 82 → memory-bound
```

*Roofline model:* Visual representation of performance limits.

```
                    ┌─────────────────────────────
                    │   Peak compute (82.6 TFLOPS)
                    │═══════════════════════════
         TFLOPS     │         /
          (log)     │       /   Memory bandwidth ceiling
                    │     /     (1008 GB/s)
                    │   /
                    │ /
                    │/
                    └──────────────────────────────
                       Operational Intensity (FLOPs/byte)
                           (log scale)

Ridge point = Peak TFLOPS / Peak bandwidth
            = 82.6 / 1.008
            = 82 FLOPs/byte

Below ridge: Memory-bound (increase data reuse)
Above ridge: Compute-bound (optimize compute)
```

== GPU Programming Models

*CUDA (NVIDIA):* Dominant GPU programming model.

```c
// CUDA kernel
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Host code
int main() {
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
}
```

*OpenCL:* Cross-platform GPU programming.

```c
// OpenCL kernel
__kernel void vectorAdd(__global const float* a,
                        __global const float* b,
                        __global float* c) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
```

*HIP (AMD):* ROCm programming model (CUDA-like syntax).

```cpp
// HIP kernel (nearly identical to CUDA)
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Host code
hipMalloc(&d_a, n * sizeof(float));
vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
```

*SYCL:* Modern C++ abstraction.

```cpp
// SYCL kernel
queue q;
buffer<float> buf_a(a, range<1>(n));
buffer<float> buf_b(b, range<1>(n));
buffer<float> buf_c(c, range<1>(n));

q.submit([&](handler& h) {
    auto acc_a = buf_a.get_access<access::mode::read>(h);
    auto acc_b = buf_b.get_access<access::mode::read>(h);
    auto acc_c = buf_c.get_access<access::mode::write>(h);

    h.parallel_for(range<1>(n), [=](id<1> i) {
        acc_c[i] = acc_a[i] + acc_b[i];
    });
});
```

== Comparison with Other Accelerators

```
Accelerator    Strengths                    Best For
───────────────────────────────────────────────────────────────────
GPU            Massive parallelism,         ML training, HPC,
               mature ecosystem             graphics, simulation

TPU            Matrix operations,           Large-scale ML training,
               high bandwidth               inference

FPGA           Custom datapaths,            Low-latency inference,
               deterministic timing         signal processing

CPU SIMD       Low latency, flexibility     Small batches,
               general purpose              sequential code

DSA (NPU)      Power efficiency,            Edge inference,
               specific operations          mobile AI
```

*When to use GPU:*
- Problem is data-parallel (same operation on many elements)
- Large datasets (amortize kernel launch overhead)
- Compute-intensive (not I/O bound)
- Algorithm maps to GPU programming model

*When GPU struggles:*
- Sequential dependencies
- Small problem sizes
- Irregular memory access patterns
- Heavy branching (warp divergence)

== References

NVIDIA Corporation (2024). CUDA C++ Programming Guide. Version 12.4. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

NVIDIA Corporation (2024). CUDA C++ Best Practices Guide. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

Kirk, D.B. & Hwu, W.W. (2016). Programming Massively Parallel Processors: A Hands-on Approach (3rd ed.). Morgan Kaufmann.

Hennessy, J.L. & Patterson, D.A. (2019). Computer Architecture: A Quantitative Approach (6th ed.). Morgan Kaufmann. Chapter 4 (Data-Level Parallelism).

Williams, S., Waterman, A., & Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures." Communications of the ACM 52(4): 65-76.

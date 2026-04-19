= Compute Units and Specialized Cores

Modern GPUs contain multiple types of processing units optimized for different workloads: CUDA cores for general compute, Tensor Cores for matrix operations, and RT Cores for ray tracing. Understanding their capabilities and programming is essential for maximum performance.

*See also:* GPU Fundamentals (for architecture overview), Execution Model (for scheduling), Memory Hierarchy (for data movement)

== CUDA Cores (Streaming Processors)

CUDA cores are scalar floating-point and integer execution units. Each processes one thread's operation per cycle.

*Operations per CUDA core:*

```
FP32:    1 FLOP per cycle (add, mul, FMA)
FP64:    0.5 FLOP per cycle (consumer GPUs) or 1 FLOP (datacenter)
INT32:   1 IOP per cycle
FP16:    2 FLOPs per cycle (packed operations)
```

*CUDA core organization (Ada Lovelace SM):*

```
┌────────────────────────────────────────────────────────────────┐
│                    Streaming Multiprocessor                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Processing Block × 4                    │  │
│  │  ┌───────────────────────────────────────────────────┐   │  │
│  │  │  FP32 Units    │  16 × FP32 ALU                   │   │  │
│  │  │  FP32 Units    │  16 × FP32 ALU (additional)      │   │  │
│  │  │  INT32 Units   │  16 × INT32 ALU                  │   │  │
│  │  │  Tensor Core   │  1 × 4th Gen Tensor Core         │   │  │
│  │  │  Load/Store    │  8 × LD/ST units                 │   │  │
│  │  │  SFU           │  4 × Special Function Units      │   │  │
│  │  └───────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Total per SM: 128 FP32 + 128 FP32/INT32 + 4 Tensor Cores       │
└────────────────────────────────────────────────────────────────┘
```

*Instruction throughput (cycles per instruction):*

```
Operation             Ada      Ampere    Turing    Pascal
───────────────────────────────────────────────────────────
FP32 add/mul/FMA      0.5      0.5       0.5       0.5
FP64 add/mul/FMA      16       16        32        4
INT32 add/mul         0.5      0.5       0.5       1
FP16 (CUDA core)      0.25     0.25      0.5       2

Special functions (per SM, via SFU):
sin, cos, exp, log    4 cycles (32 threads)
rsqrt, rcp            4 cycles (32 threads)
sqrt                  8 cycles (uses reciprocal)
```

*Dual-issue capability (Turing+):*

```
Some architectures can dual-issue FP32 + INT32 simultaneously:

Cycle 0: FP32 instruction + INT32 instruction (both execute)
Cycle 1: Next pair of instructions

Effective: 256 operations per SM per cycle
But: Limited by register ports, dependencies
```

== Tensor Cores

Tensor Cores perform matrix multiply-accumulate operations on small matrices (tiles), dramatically accelerating deep learning and HPC workloads.

*Tensor Core evolution:*

```
Generation   Architecture   Matrix Size      Precision           TFLOPS (dense)
──────────────────────────────────────────────────────────────────────────────────
1st Gen      Volta          4×4×4           FP16                125
2nd Gen      Turing         4×4×4           FP16, INT8          130
3rd Gen      Ampere         8×4×4           FP16, BF16, TF32    312
4th Gen      Ada            8×4×4           FP16, BF16, FP8     330
5th Gen      Hopper         8×4×4           FP8, FP16, INT8     989
```

*Matrix multiply-accumulate operation:*

```
D = A × B + C

Where:
A: m × k matrix (FP16/BF16/FP8/INT8)
B: k × n matrix (FP16/BF16/FP8/INT8)
C: m × n matrix (FP32/FP16)
D: m × n matrix (FP32/FP16)

4th Gen Tensor Core (Ada):
- Processes 8×8×4 per Tensor Core per cycle
- 4 Tensor Cores per SM
- 128 SMs = 512 Tensor Cores
- Peak: 660 TFLOPS (sparse), 330 TFLOPS (dense)
```

*WMMA API (Warp Matrix Multiply-Accumulate):*

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void tensor_core_example(half* a, half* b, float* c, float* d) {
    // Declare fragments
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    // Load matrices from global memory
    load_matrix_sync(a_frag, a, 16);
    load_matrix_sync(b_frag, b, 16);
    fill_fragment(c_frag, 0.0f);

    // Perform matrix multiply-accumulate
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store result
    store_matrix_sync(d, c_frag, 16, mem_row_major);
}
```

*Supported matrix sizes (Ampere+):*

```
Fragment     M     N     K     A/B Type    C/D Type
─────────────────────────────────────────────────────
m16n16k16   16    16    16    half        float
m32n8k16    32     8    16    half        float
m8n32k16     8    32    16    half        float
m16n16k8    16    16     8    bf16        float
m16n16k16   16    16    16    tf32        float
m16n16k16   16    16    16    s8/u8       int32
```

*TF32 (TensorFloat-32):*

```
TF32: 19-bit format (1 sign + 8 exp + 10 mantissa)
- Same range as FP32 (8-bit exponent)
- Reduced precision (10-bit vs 23-bit mantissa)
- 8× faster than FP32, accuracy close to FP32
- Automatically used by cuBLAS/cuDNN for FP32 inputs

// Enable TF32 (default on Ampere+)
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
```

*Structured sparsity (Ampere+):*

```
2:4 sparsity pattern: 2 of every 4 elements are zero

Dense:  [1.2, 0.5, 0.0, 0.8, 0.3, 0.0, 0.7, 0.1]
Sparse: [1.2, 0.5, x, x, 0.3, x, 0.7, 0.1]  (50% compression)

Benefit: 2× throughput (660 TFLOPS sparse vs 330 TFLOPS dense)
Requirement: Prune model to 2:4 sparsity pattern
```

== RT Cores (Ray Tracing)

RT Cores accelerate ray-triangle intersection and bounding volume hierarchy (BVH) traversal in hardware.

*RT Core operations:*

```
1. Ray-BVH traversal: Navigate acceleration structure
2. Ray-triangle intersection: Möller-Trumbore algorithm
3. Ray-box intersection: Slab method

Hardware acceleration: 10-20× faster than CUDA core implementation
```

*OptiX ray tracing (NVIDIA):*

```cpp
// Acceleration structure (BVH) build
OptixAccelBuildOptions accel_options = {};
accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

OptixTraversableHandle gas_handle;
optixAccelBuild(context, stream, &accel_options,
                &build_input, 1, d_temp, temp_size,
                d_output, output_size, &gas_handle, nullptr, 0);

// Ray generation program
extern "C" __global__ void __raygen__render() {
    uint3 idx = optixGetLaunchIndex();
    float3 origin, direction;
    compute_ray(idx, origin, direction);

    // Trace ray using RT Cores
    optixTrace(handle, origin, direction,
               tmin, tmax, time,
               mask, flags,
               SBT_OFFSET, SBT_STRIDE, MISS_INDEX,
               payload0, payload1);
}
```

*DXR/Vulkan Ray Tracing:*

```cpp
// Vulkan ray tracing pipeline
VkRayTracingPipelineCreateInfoKHR pipeline_info = {
    .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
    .stageCount = shader_stage_count,
    .pStages = shader_stages,
    .groupCount = shader_group_count,
    .pGroups = shader_groups,
    .maxPipelineRayRecursionDepth = 2,
    .layout = pipeline_layout,
};

vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, pipeline_cache,
                               1, &pipeline_info, nullptr, &pipeline);
```

== Special Function Units (SFUs)

SFUs compute transcendental functions in hardware, much faster than CUDA core emulation.

```
SFU operations:
- sin, cos (single precision)
- exp2, log2
- rsqrt (reciprocal square root)
- rcp (reciprocal)

Throughput: 8 operations per SM per cycle (32 threads in 4 cycles)
Accuracy: IEEE-compliant for common operations

// Fast math intrinsics (less accurate, faster)
__sinf(x);      // Fast sine
__cosf(x);      // Fast cosine
__expf(x);      // Fast exp
__logf(x);      // Fast log
__fdividef(a, b); // Fast divide

// IEEE-compliant (default)
sinf(x);        // Uses SFU + Newton-Raphson
cosf(x);        // Uses SFU + Newton-Raphson
```

*Compiler flags:*

```bash
# Fast math (use fast intrinsics)
nvcc -use_fast_math kernel.cu

# Individual controls
nvcc --fmad=true         # Fused multiply-add
nvcc --prec-div=false    # Fast division
nvcc --prec-sqrt=false   # Fast square root
nvcc --ftz=true          # Flush denormals to zero
```

== FP64 Compute

Double-precision (FP64) performance varies dramatically between consumer and datacenter GPUs.

```
GPU                  FP32 TFLOPS    FP64 TFLOPS    FP64:FP32 Ratio
──────────────────────────────────────────────────────────────────────
RTX 4090             82.6           1.29           1:64
A100 (SXM)           19.5           9.7            1:2
H100 (SXM)           67             34             1:2
MI250X (AMD)         47.9           47.9           1:1
```

*FP64 programming:*

```cpp
// Double precision kernel
__global__ void fp64_kernel(double* a, double* b, double* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];  // FP64 operation
    }
}

// Check FP64 capability
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("FP64 to FP32 ratio: 1:%d\n", prop.singleToDoublePrecisionPerfRatio);
```

== INT8 and Quantized Compute

INT8 operations provide higher throughput for inference workloads.

```
Precision    Bits    Range              Use Case
──────────────────────────────────────────────────────────────
FP32         32      ±3.4e38            Training (legacy)
FP16         16      ±65504             Training, inference
BF16         16      ±3.4e38            Training (range > FP16)
TF32         19      ±3.4e38            Training (Ampere+)
FP8          8       ±448 (E4M3)        Inference, training
INT8         8       -128 to 127        Inference
INT4         4       -8 to 7            Inference (emerging)
```

*INT8 inference example:*

```cpp
// cuDNN INT8 convolution
cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);
cudnnSetConvolutionNdDescriptor(conv_desc, ..., CUDNN_DATA_INT8);

// TensorRT INT8 calibration
config->setFlag(BuilderFlag::kINT8);
config->setInt8Calibrator(calibrator);

// Explicit quantization
__global__ void quantize(float* input, int8_t* output, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx] * scale;
        val = fminf(fmaxf(val, -128.0f), 127.0f);  // Clamp
        output[idx] = __float2int_rn(val);  // Round to nearest
    }
}
```

== FP8 (Hopper+)

FP8 is the latest low-precision format, offering 2× throughput over FP16 with acceptable accuracy for training.

```
FP8 formats:
- E4M3: 1 sign + 4 exponent + 3 mantissa (±448, higher precision)
- E5M2: 1 sign + 5 exponent + 2 mantissa (±57344, higher range)

Hopper FP8 throughput: 1979 TFLOPS (sparse), 989 TFLOPS (dense)

// cuBLAS FP8 GEMM
cublasLtMatmul(handle, matmul_desc,
               alpha, A, A_desc,
               B, B_desc,
               beta, C, C_desc,
               D, D_desc,
               &algo, workspace, workspace_size, stream);
// Where A_desc, B_desc have CUDA_R_8F_E4M3 data type
```

== Instruction Latency and Throughput

*Instruction timing (cycles):*

```
Instruction                  Latency    Throughput (per SM/cycle)
────────────────────────────────────────────────────────────────────
FP32 add/mul/FMA            4          128
FP64 add/mul/FMA            8          2 (consumer) / 64 (datacenter)
INT32 add/mul               4          128
FP16 (CUDA cores)           4          256
Tensor Core 16x16x16        8          256 FLOPs
Shared memory load          20-30      128 (32 banks × 4 bytes)
Global memory load          400+       Based on bandwidth
SFU (sin, cos, rsqrt)       8          32
Integer divide              80         1
```

*Latency hiding calculation:*

```
To hide memory latency:
Required parallelism = Latency × Throughput

Example:
- Global memory latency: 400 cycles
- Instruction issue rate: 1 per 4 cycles per warp
- Required warps: 400 / 4 = 100 warps

With 48 warps per SM max → Not fully hidden!
Solution: More compute per memory access (higher arithmetic intensity)
```

== Instruction-Level Parallelism

*Maximizing ILP:*

```cpp
// LOW ILP: Sequential dependencies
float a = data[0];
float b = a * 2.0f;    // Depends on a
float c = b + 1.0f;    // Depends on b
float d = c * 3.0f;    // Depends on c

// HIGH ILP: Independent operations
float a = data[0];
float b = data[1];
float c = data[2];
float d = data[3];

float w = a * 2.0f;    // Independent
float x = b * 2.0f;    // Independent
float y = c * 2.0f;    // Independent
float z = d * 2.0f;    // Independent
```

*Loop unrolling:*

```cpp
// Before unrolling
for (int i = 0; i < n; i++) {
    sum += data[i];
}

// After unrolling (4×)
for (int i = 0; i < n; i += 4) {
    sum += data[i];
    sum += data[i+1];
    sum += data[i+2];
    sum += data[i+3];
}

// Even better: Independent accumulators
float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
for (int i = 0; i < n; i += 4) {
    sum0 += data[i];
    sum1 += data[i+1];
    sum2 += data[i+2];
    sum3 += data[i+3];
}
float sum = sum0 + sum1 + sum2 + sum3;
```

== Hopper Architecture Deep Dive

Hopper (H100, 2022; H200, 2024) introduced the deepest shake-up of GPU compute since Volta. Key innovations all target large-scale ML: the Transformer Engine, TMA, Distributed Shared Memory, and Thread Block Clusters.

=== H100 / H200 Specifications

#table(
  columns: (auto, auto, auto),
  [*Parameter*], [*H100 SXM5*], [*H200 SXM5*],
  [Process], [TSMC N4 (custom 4N)], [TSMC N4],
  [Transistors], [80 B], [80 B],
  [SMs], [132 (of 144 enabled)], [132],
  [CUDA cores], [16,896], [16,896],
  [Tensor cores], [528 (4th gen)], [528],
  [FP8 Tensor (dense)], [989 TFLOPS], [989 TFLOPS],
  [FP16 / BF16 Tensor (dense)], [989 TFLOPS], [989 TFLOPS],
  [FP32 / TF32 Tensor], [495 TFLOPS], [495 TFLOPS],
  [FP64 Tensor], [67 TFLOPS], [67 TFLOPS],
  [Memory], [80 GB HBM3], [141 GB HBM3e],
  [Memory bandwidth], [3.35 TB/s], [4.8 TB/s],
  [NVLink (per GPU)], [900 GB/s (NVLink 4)], [900 GB/s],
  [L2 cache], [50 MB], [50 MB],
  [Shared mem / SM], [228 KB (configurable)], [228 KB],
  [Register file / SM], [256 KB], [256 KB],
  [TDP (SXM5)], [700 W], [700 W],
)

=== 4th-Gen Tensor Cores: FP8 and Transformer Engine

Hopper Tensor Cores natively support FP8 with two encodings:
- *E4M3* (1 sign, 4 exponent, 3 mantissa): narrower dynamic range, higher precision — used for weights and activations
- *E5M2* (1 sign, 5 exponent, 2 mantissa): wider dynamic range, lower precision — used for gradients

FP8 peak on H100: 1979 TFLOPS sparse / 989 TFLOPS dense — 2$times$ the Ampere FP16 throughput with half the memory per weight.

*Transformer Engine:* software + hardware feature that dynamically manages FP8 scaling. Per-tensor amax history tracked in hardware; scale factors recomputed every iteration to keep values in the representable range. Exposed via the `transformer_engine` library:
```cpp
import transformer_engine.pytorch as te
model = te.TransformerLayer(hidden, num_heads, ffn_hidden,
                            fp8_group=fp8_group)
with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
    y = model(x)
```
Typical LLM training speedup: 1.5-2$times$ vs BF16 at matched accuracy.

=== Tensor Memory Accelerator (TMA)

Dedicated hardware unit on each SM that performs *asynchronous* bulk memory transfers between global memory and shared memory, freeing CUDA cores for compute. Supports multi-dimensional tensor descriptors (up to 5-D) with on-the-fly index arithmetic and bounds checking.

```cpp
// Host: build TMA descriptor once
CUtensorMap desc;
cuTensorMapEncodeTiled(&desc, CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    /*tensor_rank=*/2, /*gmem_addr=*/A_gmem,
    /*global_shape=*/{M, K}, /*global_stride=*/{K*2, 2},
    /*box_shape=*/{BLOCK_M, BLOCK_K}, ...);

// Kernel: async load
__global__ void gemm_kernel(const __grid_constant__ CUtensorMap A_desc, ...) {
    __shared__ __align__(128) half A_smem[BLOCK_M][BLOCK_K];
    __shared__ __align__(8)    uint64_t barrier;

    if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &A_smem, &A_desc, block_m, block_k, barrier);
        cde::cp_async_bulk_commit_group();
    }
    cde::cp_async_bulk_wait_group_read<0>();
    __syncthreads();
    // ... compute on A_smem ...
}
```
Compared to Ampere `cp.async`: TMA issues entire multi-dim tile with one instruction, handles boundaries natively, no per-thread address computation, and supports _multicast_ (one load broadcast to multiple SMs in a cluster).

=== Distributed Shared Memory (DSMEM) and Thread Block Clusters

*Thread Block Clusters:* opt-in grouping of up to 8 blocks (16 via portability hint) that are guaranteed co-resident on the same GPC (Graphics Processing Cluster). Blocks in a cluster can:
- Access each other's shared memory via a virtual address space (DSMEM)
- Synchronize cluster-wide via `__cluster_barrier()` or CUDA C++ `cuda::experimental::barrier`
- Cluster-wide shared memory up to $8 times 228 = 1824$ KB

```cpp
__global__ void __cluster_dims__(2, 2, 1) fused_kernel(...) {
    __shared__ float smem[N];
    auto cluster = cooperative_groups::this_cluster();
    int cluster_block_rank = cluster.block_rank();

    // Load my shard
    load_to_smem(smem, ...);
    cluster.sync();   // all blocks in cluster have loaded

    // Access a neighbor block's shared memory through DSMEM
    float* remote = cluster.map_shared_rank(smem, /*peer=*/cluster_block_rank ^ 1);
    float v = remote[tid];    // direct load from neighbor SMEM — NVLink-class BW
    // ...
}
```

Use cases: large-N reduction, distributed matmul tiles, fused attention where K/V block exceeds one SM's SMEM capacity. FA3 uses this extensively.

=== Async Barriers (`mbarrier`)

Hopper adds arrive-with-transaction-count semantics: a producer commits a memory transaction (e.g., TMA load), the barrier tracks byte-count, consumer waits for target byte count rather than count of arrivals. Enables fine-grained producer/consumer patterns across warps in one CTA.

```cpp
__shared__ cuda::barrier<cuda::thread_scope_block> bar;
if (tid == 0) init(&bar, /*expected_arrival_count=*/1);
__syncthreads();

// Producer warp
cde::cp_async_bulk_tensor(smem_ptr, &tma_desc, coords, bar);
cde::cp_async_bulk_commit_group();
bar.arrive_and_wait();      // blocks on transaction completion

// Consumers now use smem_ptr safely
```

=== L2 Cache Residency Control

H100 L2 is partitioned; applications can pin a window of global memory for preferential L2 retention:
```cpp
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr  = ptr;
attr.accessPolicyWindow.num_bytes = 16 * 1024 * 1024;  // up to 16 MB
attr.accessPolicyWindow.hitRatio  = 1.0f;
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```
Useful for a hot KV cache or frequently reused small model tensors.

=== Compiling for Hopper

```bash
nvcc -arch=sm_90a my_kernel.cu -o my_kernel
# sm_90a = Hopper architecture with 'a' (architecture-specific) features enabled
# Required to use wgmma, TMA, cluster APIs
```

Hopper PTX cheat sheet:
- `wgmma.mma_async.sync.aligned.m64n{N}k16.f32.f16.f16` — warp-group matmul, async
- `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes` — TMA load
- `mbarrier.arrive.expect_tx.shared.b64` — arrive with byte count
- `cluster.sync` — cluster-wide barrier
- `griddepcontrol.wait` — grid-wide dependency (for kernel fusion)

== Blackwell Architecture (2024)

Blackwell (B100, B200, GB200) is NVIDIA's 2024 architecture, purpose-designed for trillion-parameter model training and real-time inference.

=== B100/B200 Specifications

#table(
  columns: (auto, auto, auto),
  [*Parameter*], [*B100*], [*B200*],
  [Process], [TSMC N4P (custom)], [TSMC N4P],
  [Transistors], [208 B (2 dies)], [208 B (2 dies)],
  [Die interconnect], [NV-HBI 10 TB/s], [NV-HBI 10 TB/s],
  [FP8 Tensor (dense)], [3.5 PFLOPS], [4.5 PFLOPS],
  [FP4 Tensor (dense)], [7 PFLOPS], [9 PFLOPS],
  [FP16 / BF16 Tensor], [1.75 PFLOPS], [2.25 PFLOPS],
  [FP64 Tensor], [30 TFLOPS], [40 TFLOPS],
  [Memory], [192 GB HBM3e], [192 GB HBM3e],
  [Memory bandwidth], [8 TB/s], [8 TB/s],
  [NVLink (per GPU)], [1800 GB/s (NVLink 5)], [1800 GB/s],
  [TDP], [700 W], [1000 W],
)

*NV-HBI (NVLink High Bandwidth Interconnect):* die-to-die on-package link at 10 TB/s. The two dies are cache-coherent and present as a single logical GPU — no software changes needed vs single-die.

=== 5th-Gen Tensor Cores: FP4 and Microscaling

Blackwell introduces hardware FP4 (E2M1) for inference and training of quantization-tolerant models. Combined with *microscaling (MX) formats* (OCP 2024): each 32-element block shares an 8-bit exponent scale (MXFP4, MXFP6, MXFP8).

Effect: accuracy competitive with FP8 at half the memory and nearly 2$times$ the throughput. Used for inference and for training with careful loss scaling.

*2nd-gen Transformer Engine:* per-group microscaling handled in hardware; no per-tensor amax tracking overhead. Critical for making FP4 usable in practice.

=== Decompression Engine

Blackwell adds a dedicated hardware unit for LZ4 / Snappy / Deflate decompression directly into GPU memory. Use case: database/analytics workloads (Spark, dbt) where data on disk is compressed — avoid CPU decompression bottleneck.

Throughput: up to 800 GB/s of decompressed output per GPU.

=== RAS (Reliability, Availability, Serviceability)

For rack-scale clusters (GB200 NVL72), silent data corruption is a statistical certainty. Blackwell adds:
- Hardware-checked ECC on every internal bus (not just HBM)
- NVLink poisoning + isolation on fault
- In-field error telemetry and predictive isolation

=== GB200 NVL72 Rack

The flagship Blackwell platform:
- 72 B200 GPUs + 36 Grace CPUs (72 ARM Neoverse V2 cores each)
- Single NVLink 5 fabric: 130 TB/s aggregate bisection bandwidth, 1800 GB/s per GPU, all-to-all non-blocking
- 13.4 TB HBM3e (total GPU memory) + 17 TB LPDDR5X (Grace CPU memory) — all cache-coherent via NVLink-C2C
- Peak 720 PFLOPS FP8 / 1.4 EFLOPS FP4 (dense) per rack
- Programming model: presents as an extended single node; cross-GPU NVSHMEM at near-single-GPU latencies

=== MIG v2 (Multi-Instance GPU)

Blackwell refines MIG (introduced on A100): finer partition granularity (7 slices on H100 → up to 7 on B200 with per-slice HBM3e + NVLink partitioning). Better for multi-tenant inference serving.

=== Compiling for Blackwell

```bash
nvcc -arch=sm_100a my_kernel.cu -o my_kernel
# sm_100a = Blackwell (B100/B200); sm_100 for portable subset
# Required for FP4 wgmma, 2nd-gen Transformer Engine intrinsics
```

== AMD CDNA 3 and MI300 Series

AMD's datacenter GPU line (CDNA = Compute DNA, distinct from RDNA consumer lines).

=== MI300X Architecture

- 8 XCDs (accelerator chiplets), each 38 CUs = *304 CUs total*, 19,456 stream processors
- 4 IOD (I/O dies) with 256 MB Infinity Cache (shared L3-like victim cache)
- 8 HBM3 stacks = *192 GB HBM3*, *5.3 TB/s bandwidth* (2$times$ H100 capacity)
- Infinity Fabric 4 intra-chiplet, PCIe Gen5 + 896 GB/s 7-link Infinity Fabric external
- Peak FP8 matrix: 2614 TFLOPS dense / 5229 TFLOPS sparse
- Peak FP16 matrix: 1307 TFLOPS
- FP64 matrix: 163 TFLOPS (vs H100 67 TFLOPS — MI300X stronger in HPC)
- TDP 750 W

=== MI300A: CPU+GPU APU

First production *cache-coherent CPU+GPU APU* at datacenter scale:
- 3 Zen 4 CCDs (24 CPU cores) + 6 XCDs (228 CUs) in one package
- 128 GB shared HBM3 addressable by both CPU and GPU without explicit transfers
- Deployed in El Capitan (LLNL, 2024) — exascale HPC system, peak $>$ 2 EFLOPS

Programming: unified virtual address space, pointer-equivalence between CPU and GPU code. Eliminates the host/device copy boundary.

=== Matrix Cores (AMD's Tensor Core Equivalent)

MFMA (Matrix Fused Multiply-Add) instructions:
- `v_mfma_f32_32x32x8_f16`: 32$times$32$times$8 FP16→FP32, 4 cycles
- `v_mfma_f32_16x16x32_f8` (MI300+): 16$times$16$times$32 FP8→FP32

Intrinsics:
```cpp
__device__ float4 mfma(float4 acc, half8 a, half8 b) {
    return __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, acc, 0, 0, 0);
}
```
rocBLAS / Composable Kernel library wrap these with GEMM kernels analogous to cuBLAS / CUTLASS.

=== ROCm and HIP

*ROCm* (Radeon Open Compute): AMD's software stack, roughly equivalent to CUDA Toolkit.

*HIP* (Heterogeneous Interface for Portability): C++ runtime with CUDA-like API. A HIP kernel compiles for both AMD and NVIDIA GPUs.

CUDA → HIP automatic porting:
```bash
hipify-perl my_code.cu -o my_code.cpp     # Perl-based textual rewrite
# Or for larger codebases:
hipify-clang --source-dir my_src/ --dest-dir my_hip/
```

HIP example (vector add, identical CUDA structure):
```cpp
#include <hip/hip_runtime.h>

__global__ void vec_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 1 << 20;
    float *a_d, *b_d, *c_d;
    hipMalloc(&a_d, n*sizeof(float));
    hipMalloc(&b_d, n*sizeof(float));
    hipMalloc(&c_d, n*sizeof(float));
    hipLaunchKernelGGL(vec_add, dim3((n+255)/256), dim3(256), 0, 0,
                       a_d, b_d, c_d, n);
    hipDeviceSynchronize();
    hipFree(a_d); hipFree(b_d); hipFree(c_d);
}
```
Compiles with `hipcc` for either AMD or NVIDIA target.

Equivalent library mapping:
- cuBLAS → rocBLAS / hipBLAS
- cuDNN → MIOpen
- NCCL → RCCL
- cuRAND → rocRAND
- TensorRT → MIGraphX
- CUTLASS → Composable Kernel (CK)

=== H100 vs MI300X Quick Comparison

#table(
  columns: (auto, auto, auto),
  [*Metric*], [*H100 SXM5*], [*MI300X*],
  [Process], [TSMC N4], [TSMC N5 + N6],
  [Transistors], [80 B], [153 B (8 dies)],
  [HBM], [80 GB HBM3], [192 GB HBM3 (2.4$times$)],
  [Memory bandwidth], [3.35 TB/s], [5.3 TB/s (1.6$times$)],
  [FP8 Tensor (dense)], [989 TFLOPS], [2614 TFLOPS (2.6$times$)],
  [FP16 Tensor (dense)], [989 TFLOPS], [1307 TFLOPS (1.3$times$)],
  [FP64 Tensor], [67 TFLOPS], [163 TFLOPS (2.4$times$)],
  [Software maturity], [CUDA (mature)], [ROCm 6 (rapidly improving)],
  [TDP], [700 W], [750 W],
)

*Trade-off:* MI300X wins on memory and raw peak numbers (especially FP64 and FP8), but NVIDIA's software ecosystem — CUTLASS, TensorRT-LLM, cuDNN, Triton — currently delivers higher achieved performance on ML workloads. The gap is narrowing with Composable Kernel and vLLM's ROCm backend.

== GPU Compute Generations Comparison

```
Feature          Volta    Turing    Ampere    Ada       Hopper
───────────────────────────────────────────────────────────────────────
CUDA Cores/SM    64       64        64/128    128       128
Tensor Cores/SM  8        8         4         4         4
RT Cores/SM      -        1         1         1         -
Shared Mem/SM    96 KB    64 KB     164 KB    100 KB    228 KB
L2 Cache         6 MB     6 MB      40 MB     72 MB     50 MB
Register/SM      256 KB   256 KB    256 KB    256 KB    256 KB
FP32 TFLOPS      15.7     16.3      19.5      82.6      60
Tensor TFLOPS    125      130       312       660       989
Memory           HBM2     GDDR6     HBM2e     GDDR6X    HBM3
```

== References

NVIDIA Corporation (2024). NVIDIA Ada GPU Architecture. Whitepaper. https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf

NVIDIA Corporation (2024). NVIDIA Hopper Architecture In-Depth. https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

NVIDIA Corporation (2024). CUDA C++ Programming Guide. Chapter 7 (Tensor Cores). https://docs.nvidia.com/cuda/cuda-c-programming-guide/

NVIDIA Corporation (2023). Tensor Core Programming. https://docs.nvidia.com/deeplearning/performance/

Jia, Z. et al. (2018). "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking." Technical Report.

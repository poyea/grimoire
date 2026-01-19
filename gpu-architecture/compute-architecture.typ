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

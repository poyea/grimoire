= Performance Optimization

GPU performance optimization requires systematic analysis and targeted improvements. This section covers kernel optimization techniques, memory optimizations, and strategies for achieving peak performance.

*See also:* Memory Hierarchy (for memory-specific optimizations), Execution Model (for warp efficiency), Profiling (for measurement techniques)

== Optimization Hierarchy

Performance impact ranking (highest to lowest):

```
1. Algorithm selection           100-1000× improvement
   - O(n²) → O(n log n)
   - Sequential → parallel algorithm

2. Memory access patterns         10-100× improvement
   - Coalesced access
   - Shared memory usage
   - Cache utilization

3. Occupancy and parallelism      2-10× improvement
   - Block size tuning
   - Register pressure
   - Warp utilization

4. Instruction-level              1.2-2× improvement
   - Loop unrolling
   - ILP maximization
   - Fast math

5. Low-level tuning               1.1-1.5× improvement
   - Assembly optimization
   - Instruction scheduling
```

*Optimization workflow:*

```
1. Profile first (identify bottleneck)
   ↓
2. Classify: Memory-bound or Compute-bound?
   ↓
3. Apply targeted optimization
   ↓
4. Measure improvement
   ↓
5. Repeat until satisfactory
```

== Identifying Bottlenecks

*Memory-bound vs Compute-bound:*

```
Kernel analysis:

Compute bound:
- SM utilization > 80%
- Memory throughput < 60% of peak
- Optimization: More ILP, faster instructions

Memory bound:
- Memory throughput > 60% of peak
- SM utilization < 80%
- Optimization: Better access patterns, caching

Latency bound:
- Both SM and memory utilization low
- Many stalls waiting for data
- Optimization: More parallelism, prefetching
```

*Arithmetic intensity analysis:*

```
Arithmetic Intensity = FLOPs / Bytes moved

Example: SAXPY (y = a*x + y)
- Operations: 2 FLOPs per element (mul + add)
- Memory: 12 bytes per element (read x, read y, write y)
- Intensity: 2/12 = 0.17 FLOPs/byte

RTX 4090:
- Peak compute: 82.6 TFLOPS
- Peak bandwidth: 1008 GB/s
- Balance point: 82.6/1.008 = 82 FLOPs/byte

SAXPY intensity (0.17) << Balance point (82)
→ Heavily memory-bound, cannot achieve peak compute
```

== Memory Coalescing Optimization

*Pattern 1: Stride elimination*

```cpp
// BAD: Strided access
__global__ void strided(float* data, float* result, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    result[tid] = data[tid * stride];  // Non-coalesced
}

// GOOD: Coalesced with shared memory transpose
__global__ void coalesced(float* data, float* result, int stride) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Coalesced load to shared memory
    smem[tid] = data[gid];
    __syncthreads();

    // Strided access from fast shared memory
    int src_lane = (tid * stride) % 256;
    result[gid] = smem[src_lane];
}
```

*Pattern 2: Array of Structures transformation*

```cpp
// BAD: AoS layout
struct Particle {
    float3 position;  // 12 bytes
    float3 velocity;  // 12 bytes
    float mass;       // 4 bytes
    float pad;        // 4 bytes for alignment
};  // 32 bytes total

__global__ void update_aos(Particle* particles, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        particles[tid].position.x += particles[tid].velocity.x;
        // Stride = 32 bytes, poor coalescing
    }
}

// GOOD: SoA layout
struct ParticlesSoA {
    float* pos_x;
    float* pos_y;
    float* pos_z;
    float* vel_x;
    float* vel_y;
    float* vel_z;
    float* mass;
};

__global__ void update_soa(ParticlesSoA p, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        p.pos_x[tid] += p.vel_x[tid];  // Perfect coalescing
        p.pos_y[tid] += p.vel_y[tid];
        p.pos_z[tid] += p.vel_z[tid];
    }
}

// Speedup: 2-4× for memory-bound kernels
```

*Pattern 3: Vectorized loads*

```cpp
// GOOD: Use vector types for aligned data
__global__ void vector_copy(float4* dst, float4* src, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        dst[tid] = src[tid];  // 16-byte load/store
    }
}

// Even better: 128-bit loads
__global__ void wide_copy(int4* dst, int4* src, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        dst[tid] = src[tid];  // Single 128-bit transaction
    }
}
```

== Shared Memory Optimization

*Bank conflict elimination:*

```cpp
// BAD: 32-way bank conflict
__shared__ float smem[32][32];
float val = smem[threadIdx.x][0];  // All threads access bank 0

// GOOD: Padded to avoid conflicts
__shared__ float smem[32][33];  // 33 columns
float val = smem[threadIdx.x][0];  // Threads access different banks

// Alternative: XOR-based indexing
float val = smem[threadIdx.x][threadIdx.x % 32];  // Diagonal access
```

*Shared memory as explicit cache:*

```cpp
__global__ void tiled_matmul(float* C, float* A, float* B, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N / TILE; t++) {
        // Cooperative load into shared memory
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();

        // Compute on shared memory (fast)
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

// Memory traffic reduction:
// Naive: 2*N^3 global memory accesses
// Tiled: 2*N^3/TILE global memory accesses
// Speedup: TILE × (e.g., 32×)
```

*Double buffering:*

```cpp
__global__ void double_buffered(float* A, float* B, int n) {
    __shared__ float smem[2][TILE];  // Double buffer
    int buffer = 0;

    // Initial load
    smem[buffer][threadIdx.x] = A[threadIdx.x];

    for (int i = 0; i < n; i += TILE) {
        // Async load next tile (overlap with compute)
        if (i + TILE < n) {
            smem[1-buffer][threadIdx.x] = A[i + TILE + threadIdx.x];
        }
        __syncthreads();

        // Compute on current tile
        process(smem[buffer][threadIdx.x]);
        __syncthreads();

        buffer = 1 - buffer;  // Swap buffers
    }
}
```

== Occupancy Optimization

*Finding optimal block size:*

```cpp
// Use occupancy calculator API
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                    kernel, 0, 0);

// Or with shared memory consideration
cudaOccupancyMaxPotentialBlockSizeVariableSMem(
    &minGridSize, &blockSize, kernel,
    [](int blockSize) { return blockSize * sizeof(float); },
    0);
```

*Trading occupancy for per-thread resources:*

```cpp
// High occupancy, limited registers
__global__ __launch_bounds__(256, 8)  // 256 threads, 8 blocks/SM
void high_occupancy_kernel() { ... }
// 8 * 256 = 2048 threads, limited to ~32 registers/thread

// Lower occupancy, more registers
__global__ __launch_bounds__(128, 4)  // 128 threads, 4 blocks/SM
void more_registers_kernel() { ... }
// 4 * 128 = 512 threads, ~128 registers/thread available

// Sometimes lower occupancy wins!
// More registers → Less spilling → Fewer memory accesses
```

*Occupancy vs. performance:*

```
Workload type         Optimal occupancy
──────────────────────────────────────────
Memory-bound          High (50-100%)
Compute-bound         Medium (25-50%)
Latency-bound         High (maximize parallelism)
Register-heavy        Lower (avoid spilling)

Rule: Profile at multiple occupancy levels!
```

== Warp Efficiency Optimization

*Minimizing divergence:*

```cpp
// BAD: Random divergence
if (data[tid] > threshold) {  // Unpredictable
    result = expensive_path();
} else {
    result = cheap_path();
}

// BETTER: Sort data first
// Group threads with similar values → less divergence

// BEST: Predication (for short branches)
float expensive = expensive_path();
float cheap = cheap_path();
result = (data[tid] > threshold) ? expensive : cheap;
// Both computed, predicate selects result
// Works if both paths are similar cost
```

*Avoiding warp-unfriendly patterns:*

```cpp
// BAD: Reduction with divergence
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {  // Only half threads active
        data[tid] += data[tid + s];
    }
    __syncthreads();
}

// GOOD: Sequential addressing
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) {  // First s threads active
        data[tid] += data[tid + s];
    }
    __syncthreads();
}

// First iterations: Full warp active
// Later iterations: Fewer threads, but no intra-warp divergence
```

== Loop Optimization

*Loop unrolling:*

```cpp
// Manual unrolling
#pragma unroll 4
for (int i = 0; i < n; i++) {
    sum += data[i];
}

// Full unrolling (compile-time known bounds)
#pragma unroll
for (int i = 0; i < 8; i++) {  // Compiler unrolls completely
    sum += data[i];
}

// Disable unrolling
#pragma unroll 1
for (int i = 0; i < n; i++) {
    sum += data[i];
}
```

*Loop tiling for cache:*

```cpp
// Naive: Poor cache utilization
for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
        C[i][j] = A[i][j] + B[i][j];

// Tiled: Better cache utilization
for (int ii = 0; ii < n; ii += TILE)
    for (int jj = 0; jj < n; jj += TILE)
        for (int i = ii; i < ii + TILE && i < n; i++)
            for (int j = jj; j < jj + TILE && j < n; j++)
                C[i][j] = A[i][j] + B[i][j];
```

== Kernel Launch Optimization

*Launch overhead:*

```
Kernel launch overhead: ~5-10 µs (CPU-side)
CUDA stream: ~1-2 µs per kernel in stream

Amortization strategies:
- Large kernels (more work per launch)
- Kernel fusion (combine multiple operations)
- Persistent kernels (stay resident, process queue)
```

*Kernel fusion:*

```cpp
// BAD: Multiple kernel launches
elementwise_add<<<grid, block>>>(a, b, temp1);
elementwise_mul<<<grid, block>>>(temp1, c, temp2);
elementwise_sqrt<<<grid, block>>>(temp2, result);
// 3 launches, 3× memory round-trips

// GOOD: Fused kernel
__global__ void fused(float* a, float* b, float* c, float* result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float temp1 = a[tid] + b[tid];
    float temp2 = temp1 * c[tid];
    result[tid] = sqrtf(temp2);
}
// 1 launch, 1× memory round-trip
```

*Grid-stride loops:*

```cpp
// Handle any input size with fixed grid
__global__ void grid_stride(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n; i += stride) {
        data[i] = process(data[i]);
    }
}

// Launch with optimal grid size
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
kernel<<<numSMs * 32, 256>>>(data, n);

// Benefits:
// - Works for any n
// - Optimal occupancy
// - Reduced launch overhead for small n
```

== Async Operations and Streams

*Overlap computation and transfer:*

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// Split work into chunks
for (int i = 0; i < numChunks; i += 2) {
    // Stream 1: Copy chunk i, compute chunk i-2
    cudaMemcpyAsync(d_a + i*chunkSize, h_a + i*chunkSize,
                    chunkSize, cudaMemcpyHostToDevice, stream1);
    if (i >= 2) kernel<<<grid, block, 0, stream1>>>(d_a + (i-2)*chunkSize);

    // Stream 2: Copy chunk i+1, compute chunk i-1
    cudaMemcpyAsync(d_a + (i+1)*chunkSize, h_a + (i+1)*chunkSize,
                    chunkSize, cudaMemcpyHostToDevice, stream2);
    if (i >= 1) kernel<<<grid, block, 0, stream2>>>(d_a + (i-1)*chunkSize);
}

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

*Async memory copies (Ampere+):*

```cpp
#include <cuda/barrier>

__global__ void async_copy(float* global, float* result) {
    __shared__ float smem[256];

    // Async copy from global to shared
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
    }
    __syncthreads();

    cuda::memcpy_async(&smem[threadIdx.x], &global[threadIdx.x],
                       sizeof(float), bar);

    bar.arrive_and_wait();  // Wait for copy completion

    // Use data in shared memory
    result[threadIdx.x] = smem[threadIdx.x] * 2.0f;
}
```

== Numerical Optimization

*Fast math operations:*

```cpp
// Fast approximate functions
__device__ float fast_sigmoid(float x) {
    return __frcp_rn(1.0f + __expf(-x));  // Faster than precise
}

// Fused multiply-add (single rounding)
float c = __fmaf_rn(a, b, c);  // a*b + c

// Fast reciprocal and rsqrt
float inv = __frcp_rn(x);        // 1/x
float rsq = __frsqrt_rn(x);      // 1/sqrt(x)

// Replace division with multiplication
// Instead of: y = x / const;
float inv_const = 1.0f / const;
float y = x * inv_const;  // Faster
```

*Mixed precision:*

```cpp
// Compute in FP16, accumulate in FP32
__global__ void mixed_precision(half* input, float* output, int n) {
    float acc = 0.0f;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float val = __half2float(input[i]);
        acc += val * val;
    }

    // Warp reduction in FP32
    acc = warp_reduce_sum(acc);

    if (threadIdx.x == 0) {
        atomicAdd(output, acc);
    }
}
```

== Common Anti-Patterns

*1. Excessive synchronization:*

```cpp
// BAD: Unnecessary syncs
for (int i = 0; i < n; i++) {
    data[i] = input[i];
    __syncthreads();  // Unnecessary!
    result[i] = data[i] * 2;
    __syncthreads();  // Unnecessary!
}

// GOOD: Sync only when needed
for (int i = 0; i < n; i++) {
    data[i] = input[i];
}
__syncthreads();  // Once after all loads
for (int i = 0; i < n; i++) {
    result[i] = data[i] * 2;
}
```

*2. Global memory atomics in hot loops:*

```cpp
// BAD: Contended atomics
__global__ void bad_histogram(int* data, int* hist, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        atomicAdd(&hist[data[tid]], 1);  // Serialized!
    }
}

// GOOD: Per-block histogram in shared memory
__global__ void good_histogram(int* data, int* hist, int n) {
    __shared__ int local_hist[256];

    // Initialize local histogram
    if (threadIdx.x < 256) local_hist[threadIdx.x] = 0;
    __syncthreads();

    // Accumulate locally
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        atomicAdd(&local_hist[data[tid]], 1);  // Shared mem atomic
    }
    __syncthreads();

    // Merge to global
    if (threadIdx.x < 256) {
        atomicAdd(&hist[threadIdx.x], local_hist[threadIdx.x]);
    }
}
```

*3. Uncoalesced writes:*

```cpp
// BAD: Scatter pattern
__global__ void scatter(float* dst, float* src, int* indices, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        dst[indices[tid]] = src[tid];  // Random writes!
    }
}

// BETTER: Sort indices, use radix sort to reorder
// Or: Use atomics if collisions expected
// Or: Redesign algorithm for coalesced access
```

*4. Branch in tight loop:*

```cpp
// BAD: Branch checked every iteration
for (int i = 0; i < n; i++) {
    if (use_feature_a) {
        result += compute_a(data[i]);
    } else {
        result += compute_b(data[i]);
    }
}

// GOOD: Hoist branch outside loop
if (use_feature_a) {
    for (int i = 0; i < n; i++) {
        result += compute_a(data[i]);
    }
} else {
    for (int i = 0; i < n; i++) {
        result += compute_b(data[i]);
    }
}
```

== Performance Checklist

```
□ Memory access pattern
  □ Coalesced global memory access
  □ No bank conflicts in shared memory
  □ Aligned memory allocations
  □ Vectorized loads where possible

□ Occupancy and parallelism
  □ Appropriate block size (profile multiple)
  □ Sufficient work per thread
  □ No register spilling (check with --ptxas-options=-v)

□ Warp efficiency
  □ Minimal divergence
  □ Full warp utilization
  □ No warp-unfriendly patterns

□ Kernel launch
  □ Sufficient work to amortize launch overhead
  □ Fused kernels where appropriate
  □ Overlapped transfers and compute

□ Numerical
  □ Fast math enabled if acceptable
  □ FMA used
  □ Division converted to multiplication

□ Algorithm
  □ Parallel algorithm selected
  □ Work distribution balanced
  □ Data locality maximized
```

== References

NVIDIA Corporation (2024). CUDA C++ Best Practices Guide. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

Volkov, V. (2010). "Better Performance at Lower Occupancy." GPU Technology Conference.

Harris, M. (2007). "Optimizing Parallel Reduction in CUDA." NVIDIA Developer Technology.

Micikevicius, P. (2012). "GPU Performance Analysis and Optimization." GPU Technology Conference.

NVIDIA Corporation (2024). Nsight Compute Documentation. https://docs.nvidia.com/nsight-compute/

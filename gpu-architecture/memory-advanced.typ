#import "../template.typ": xref

= GPU Memory: Caches, HBM, and Optimization

*Advanced GPU memory topics:* This chapter covers L1/L2 caches, constant and texture memory, unified memory, HBM, access optimization patterns, and profiling. It continues from #xref("gpu-architecture", "memory-hierarchy", label: "GPU Memory Hierarchy") (registers, shared memory, global memory).

*See also:* #xref("gpu-architecture", "memory-hierarchy", label: "GPU Memory Hierarchy") (on-chip memory basics), #xref("gpu-architecture", "execution-model", label: "SIMT Execution Model"), #xref("gpu-architecture", "performance-optimization", label: "Performance Optimization").

== L1 and L2 Caches

*L1 cache:* Per-SM, unified with shared memory, hardware-managed.

```
L1 configuration (Ada Lovelace):
- Total: 128 KB per SM
- Split between L1 cache and shared memory
- Default: Balanced (e.g., 64 KB each)
- Configurable per-kernel
```

*L2 cache:* Chip-wide, shared by all SMs, critical for data reuse.

```
L2 characteristics (RTX 4090):
- Size: 72 MB
- Bandwidth: ~5 TB/s (aggregate from all SMs)
- Line size: 128 bytes
- Associativity: High (architecture-dependent)
```

*Cache-aware programming:*

```c
// Thrashing: Working set exceeds L2 cache
// 100 MB array, 72 MB L2 → Constant cache misses
__global__ void thrashing(float* big_array) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Repeated scans evict previous data
    for (int iter = 0; iter < 100; iter++) {
        big_array[tid] += 1.0f;
    }
}

// Tiling: Keep working set in L2
// Process 32 MB tiles, complete each before moving on
__global__ void tiled(float* big_array, int tile_offset) {
    int tid = tile_offset + blockIdx.x * blockDim.x + threadIdx.x;
    for (int iter = 0; iter < 100; iter++) {
        big_array[tid] += 1.0f;  // Same data reused from L2
    }
}
// Launch multiple times with different tile_offset
```

*L2 persistence (Ampere+):*

```c
// Reserve L2 cache for specific data
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
size_t persistingL2 = min((size_t)32*1024*1024, prop.persistingL2CacheMaxSize);

cudaStreamAttrValue attr = {};
attr.accessPolicyWindow.base_ptr = d_data;
attr.accessPolicyWindow.num_bytes = data_size;
attr.accessPolicyWindow.hitRatio = 1.0f;  // 100% of accesses should hit
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

== Constant Memory

Constant memory is optimized for read-only data broadcast to all threads.

```c
// Declaration (device-side global)
__constant__ float const_data[1024];  // 64 KB maximum

// Initialization (host-side)
cudaMemcpyToSymbol(const_data, host_data, sizeof(float) * 1024);

// Usage
__global__ void kernel() {
    float val = const_data[threadIdx.x % 16];  // Broadcast if same address
}
```

*Constant memory characteristics:*

```
Size: 64 KB total
Cache: 8-10 KB per SM (fully cached)
Latency: ~4 cycles if all threads access same address (broadcast)
         ~100 cycles if threads access different addresses (serialized)

Use cases:
- Lookup tables accessed uniformly
- Kernel configuration parameters
- Coefficients and constants
```

*Optimal vs suboptimal usage:*

```c
// OPTIMAL: All threads read same address (broadcast)
__constant__ float coefficient;
float result = input[tid] * coefficient;  // 1 read for entire warp

// SUBOPTIMAL: Each thread reads different address
__constant__ float lut[256];
float result = lut[data[tid]];  // Up to 32 serialized reads if non-uniform

// BETTER for non-uniform: Use L1-cached global memory
__device__ float lut_global[256];  // In global memory
// Or use texture memory
```

== Texture Memory

Texture memory provides cached access with hardware interpolation and address clamping, optimized for 2D spatial locality.

```c
// Texture object API (modern, preferred)
cudaTextureObject_t tex;
cudaResourceDesc resDesc = {};
resDesc.resType = cudaResourceTypeLinear;
resDesc.res.linear.devPtr = d_data;
resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
resDesc.res.linear.sizeInBytes = n * sizeof(float);

cudaTextureDesc texDesc = {};
texDesc.readMode = cudaReadModeElementType;
texDesc.filterMode = cudaFilterModePoint;  // or cudaFilterModeLinear
texDesc.addressMode[0] = cudaAddressModeClamp;

cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

// Usage in kernel
__global__ void texture_kernel(cudaTextureObject_t tex) {
    float val = tex1Dfetch<float>(tex, threadIdx.x);  // Cached read
}
```

*Texture memory benefits:*

```
1. 2D spatial locality caching (good for images)
2. Free hardware interpolation (bilinear, trilinear)
3. Automatic boundary handling (clamp, wrap, mirror)
4. Separate cache from L1 (doesn't compete)

Latency: ~100 cycles (cached)
Best for: Image processing, irregular read patterns
```

== Local Memory

Local memory is thread-private but physically located in global memory. Used for register spills and large local arrays.

```c
__global__ void local_memory_example() {
    int small_array[4];   // Likely in registers
    int large_array[256]; // Definitely in local memory (spilled)

    // Local memory has global memory latency!
    // Avoid large local arrays
}
```

*Detecting local memory usage:*

```bash
nvcc --ptxas-options=-v kernel.cu
# Output: ptxas info: Used 32 registers, 1024 bytes lmem
#                                         ^^^^^^^^^ Local memory!
```

== Unified Memory

Unified Memory provides a single address space accessible from both CPU and GPU, with automatic page migration.

```c
// Allocation
float* data;
cudaMallocManaged(&data, n * sizeof(float));

// Access from CPU
for (int i = 0; i < n; i++) {
    data[i] = i;  // CPU writes
}

// Access from GPU
kernel<<<blocks, threads>>>(data, n);  // Automatic migration

// Synchronize before CPU access
cudaDeviceSynchronize();
float result = data[0];  // CPU reads (data migrated back if needed)
```

*Memory hints:*

```c
// Advise system about access patterns
cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, device);
cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device);
cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device);

// Prefetch data to device
cudaMemPrefetchAsync(ptr, size, device, stream);

// Prefetch data to CPU
cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, stream);
```

*Performance considerations:*

```
Unified Memory overhead:
- Page fault handling: 10-50 µs per page fault
- Page migration: Limited by PCIe bandwidth (64 GB/s for Gen5)
- Thrashing: CPU/GPU alternating access → constant migration

Best practices:
- Use prefetching to hide migration latency
- Batch CPU accesses before GPU kernels
- For known access patterns, use explicit cudaMemcpy
```

== HBM and Modern Memory Technologies

*HBM2e (A100):*

```
Capacity: 40-80 GB
Bandwidth: 2 TB/s
Stack: 3D stacked DRAM dies on interposer
Bus: 5120-bit wide (8 stacks × 8 channels × 2 words)
```

*GDDR6X (RTX 4090):*

```
Capacity: 24 GB
Bandwidth: 1008 GB/s
Technology: PAM4 signaling (4 levels per symbol)
Bus: 384-bit
Effective speed: 21 Gbps
```

*HBM3 (H100):*

```
Capacity: 80 GB
Bandwidth: 3.35 TB/s
Stacks: 5 or 6 HBM3 stacks
Improvement: 1.5× bandwidth over HBM2e
```

*HBM3e (H200, B100/B200 — 2024):*

```
Capacity:   141 GB (H200), 192 GB (B100/B200)
Bandwidth:  4.8 TB/s (H200), 8.0 TB/s (B200, dual-die package)
Stacks:     6 HBM3e stacks (8-Hi, 24 GB per stack on B-series)
Improvement: ~1.4× bandwidth vs HBM3 (H100); 2.4× capacity vs H100 80 GB
```

*Coalesced vs strided bandwidth (per-arch microbenchmark, contiguous 4-byte loads):*

#table(
  columns: (auto, auto, auto, auto),
  [*GPU*], [*Peak*], [*Coalesced (warp 32×4 B)*], [*Stride-32 (1 elem / cache line)*],
  [RTX 4090 (GDDR6X)], [1008 GB/s], [~950 GB/s (94%)], [~30 GB/s (3%)],
  [H100 SXM (HBM3)], [3.35 TB/s], [~3.2 TB/s (95%)], [~100 GB/s (3%)],
  [B200 (HBM3e)], [8.0 TB/s], [~7.6 TB/s (95%)], [~240 GB/s (3%)],
)

The strided columns degrade in proportion to peak — the underlying ratio is fixed by cache-line granularity (32 B on NVIDIA L1, 128 B on L2) and is independent of memory technology. Use SoA layouts and coalesced indexing to recover the 30× gap.

== Memory Access Optimization Patterns

*Pattern 1: Array of Structures to Structure of Arrays*

```c
// BAD: Array of Structures (AoS)
struct Particle {
    float x, y, z;
    float vx, vy, vz;
};
Particle particles[N];

// Access x coordinates: strided by 24 bytes (6 floats)
for (int i = tid; i < N; i += stride) {
    float x = particles[i].x;  // Non-coalesced!
}

// GOOD: Structure of Arrays (SoA)
struct ParticlesSoA {
    float x[N], y[N], z[N];
    float vx[N], vy[N], vz[N];
};
ParticlesSoA p;

// Access x coordinates: consecutive
for (int i = tid; i < N; i += stride) {
    float x = p.x[i];  // Coalesced!
}
```

*Pattern 2: Matrix transpose with shared memory*

```c
__global__ void transpose(float* out, float* in, int width, int height) {
    __shared__ float tile[32][33];  // Padded to avoid bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // Coalesced read from global memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();

    // Transposed indices
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    // Coalesced write to global memory (from transposed tile)
    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

*Pattern 3: Reduction with shared memory*

```c
__global__ void reduce(float* input, float* output, int n) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements and add (reduces global loads)
    sdata[tid] = (i < n ? input[i] : 0) +
                 (i + blockDim.x < n ? input[i + blockDim.x] : 0);
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction. The classic `volatile float*` version is
    // UNSAFE on Volta+ (independent thread scheduling): lanes of a warp
    // no longer execute in lockstep, so explicit __syncwarp() is needed.
    if (tid < 32) {
        sdata[tid] += sdata[tid + 32]; __syncwarp();
        sdata[tid] += sdata[tid + 16]; __syncwarp();
        sdata[tid] += sdata[tid + 8];  __syncwarp();
        sdata[tid] += sdata[tid + 4];  __syncwarp();
        sdata[tid] += sdata[tid + 2];  __syncwarp();
        sdata[tid] += sdata[tid + 1];  __syncwarp();
    }
    // (Better still: read into a register and use __shfl_down_sync.)

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

== Memory Profiling

```bash
# NVIDIA Nsight Compute - memory metrics
ncu --set full ./program

# Key metrics:
# - Memory Throughput: Achieved bandwidth
# - L1/TEX Hit Rate: Cache effectiveness
# - L2 Hit Rate: L2 cache effectiveness
# - Mem Busy: Memory unit utilization
# - Coalescing efficiency: % of ideal transactions

# Specific memory events
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
              lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum \
    ./program
```

*Memory bandwidth analysis:*

```bash
# Calculate achieved bandwidth
# Achieved BW = (Bytes Read + Bytes Written) / Time

# Example:
# Kernel time: 1 ms
# Bytes read: 500 MB
# Bytes written: 500 MB
# Achieved BW = 1000 MB / 1 ms = 1000 GB/s = 99% of peak (1008 GB/s)
```

== References

NVIDIA Corporation (2024). CUDA C++ Programming Guide. Chapter 5 (Memory Hierarchy). https://docs.nvidia.com/cuda/cuda-c-programming-guide/

NVIDIA Corporation (2024). CUDA C++ Best Practices Guide. Chapter 9 (Memory Optimizations). https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

Harris, M. (2013). "How to Access Global Memory Efficiently in CUDA C/C++ Kernels." NVIDIA Developer Blog.

Volkov, V. & Demmel, J.W. (2008). "Benchmarking GPUs to Tune Dense Linear Algebra." SC '08.

Mei, X. & Chu, X. (2017). "Dissecting GPU Memory Hierarchy Through Microbenchmarking." IEEE Transactions on Parallel and Distributed Systems.

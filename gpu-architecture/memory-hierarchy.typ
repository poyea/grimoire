= GPU Memory Hierarchy

GPU memory hierarchy differs fundamentally from CPU caches. Understanding the distinct memory spaces, access patterns, and optimization techniques is essential for high-performance GPU programming.

*See also:* GPU Fundamentals (for architecture overview), Execution Model (for warp-based access), Performance Optimization (for memory-centric tuning)

== Memory Hierarchy Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           GPU Memory Hierarchy                       │
│                                                                       │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │                    Per-Thread                                │  │
│    │  ┌─────────────┐                                            │  │
│    │  │  Registers  │  256 KB per SM, ~255 per thread           │  │
│    │  │   0 cycles  │  Fastest, limited quantity                │  │
│    │  └─────────────┘                                            │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │                    Per-Block                                 │  │
│    │  ┌───────────────────────────────┐                          │  │
│    │  │    Shared Memory (SMEM)       │  Up to 164 KB per SM    │  │
│    │  │       ~20-30 cycles           │  User-managed cache      │  │
│    │  └───────────────────────────────┘                          │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │                    Per-SM                                    │  │
│    │  ┌───────────────────────────────┐                          │  │
│    │  │      L1 Cache / SMEM          │  128 KB (configurable)  │  │
│    │  │       ~30-40 cycles           │  Hardware-managed        │  │
│    │  └───────────────────────────────┘                          │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │                    Chip-Wide                                 │  │
│    │  ┌───────────────────────────────┐                          │  │
│    │  │         L2 Cache              │  72 MB (RTX 4090)       │  │
│    │  │       ~200 cycles             │  Shared by all SMs       │  │
│    │  └───────────────────────────────┘                          │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │                    Device Memory                             │  │
│    │  ┌───────────────────────────────┐                          │  │
│    │  │     Global Memory (VRAM)      │  24 GB GDDR6X           │  │
│    │  │       ~400-600 cycles         │  1008 GB/s bandwidth    │  │
│    │  └───────────────────────────────┘                          │  │
│    │  ┌───────────────────────────────┐                          │  │
│    │  │       Texture Memory          │  Cached, filtered       │  │
│    │  │       Constant Memory         │  Cached, broadcast      │  │
│    │  └───────────────────────────────┘                          │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │                    Host Memory                               │  │
│    │  ┌───────────────────────────────┐                          │  │
│    │  │      System RAM (DDR5)        │  128+ GB                │  │
│    │  │    PCIe: 64 GB/s (Gen5 x16)   │  ~10-20 µs latency      │  │
│    │  └───────────────────────────────┘                          │  │
│    └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

*Memory characteristics (Ada Lovelace / RTX 4090):*

```
Memory Type      Size           Latency        Bandwidth        Scope
─────────────────────────────────────────────────────────────────────────
Registers        256 KB/SM      0 cycles       ~20 TB/s/SM      Thread
Shared Memory    Up to 100 KB   ~20 cycles     ~2 TB/s/SM       Block
L1 Cache         128 KB/SM      ~30 cycles     ~2 TB/s/SM       SM
L2 Cache         72 MB          ~200 cycles    ~5 TB/s          Chip
Global (GDDR6X)  24 GB          ~400 cycles    1008 GB/s        Device
Constant Cache   64 KB          ~4 cycles      Broadcast        Device
Texture Cache    per SM         ~100 cycles    Filtered         Device
```

== Registers

Registers provide the fastest storage, with zero additional latency for operands.

*Register allocation:*

```c
__global__ void register_example() {
    int a = 10;      // Stored in register
    float b = 3.14f; // Stored in register
    int arr[4];      // May spill to local memory if too large

    // Use PTX to see actual register usage
}
```

*Register limits:*

```
Maximum registers per thread: 255
Register file per SM: 65,536 (256 KB)

Example calculation:
- 48 registers per thread
- Block size: 256 threads
- Registers needed: 256 × 48 = 12,288 registers
- Blocks possible: 65536 / 12288 = 5.3 → 5 blocks per SM

Higher register usage → Lower occupancy → Potential performance impact
```

*Controlling register usage:*

```c
// Limit registers per thread (CUDA)
__global__ __launch_bounds__(256, 4)  // 256 threads, 4 blocks min
void limited_kernel() { ... }

// Compile-time limit
nvcc -maxrregcount=32 kernel.cu
```

*Register spilling:* When register demand exceeds supply, values spill to local memory (slow).

```bash
# Check for spilling
nvcc --ptxas-options=-v kernel.cu
# Look for: "spill stores" and "spill loads"

# Example output:
# ptxas info: Used 64 registers, 0 bytes smem, 0 bytes lmem
# ptxas info: 0 bytes spill stores, 0 bytes spill loads  ← Good!

# Bad output:
# ptxas info: 128 bytes spill stores, 128 bytes spill loads  ← Spilling!
```

== Shared Memory

Shared memory is fast, user-managed memory visible to all threads in a block. It serves as a scratchpad for inter-thread communication and data reuse.

*Declaration and usage:*

```c
__global__ void shared_example(float* data, int n) {
    // Static allocation
    __shared__ float smem[256];

    // Dynamic allocation (size passed at kernel launch)
    extern __shared__ float dynamic_smem[];

    int tid = threadIdx.x;

    // Load from global to shared
    smem[tid] = data[blockIdx.x * blockDim.x + tid];
    __syncthreads();  // Ensure all threads have loaded

    // Use shared memory (fast)
    float result = smem[tid] + smem[(tid + 1) % 256];

    // ...
}

// Launch with dynamic shared memory
kernel<<<blocks, threads, sharedMemSize>>>(data, n);
```

*Shared memory bank conflicts:*

Shared memory is divided into 32 banks (4 bytes each). Simultaneous access to the same bank by different threads causes serialization.

```
Bank assignment (32 banks, 4-byte words):
Address  0x00  0x04  0x08  0x0C  ...  0x7C  0x80  0x84
Bank        0     1     2     3  ...    31     0     1

Conflict-free access:
Thread 0 → Bank 0
Thread 1 → Bank 1
Thread 2 → Bank 2
...
Thread 31 → Bank 31
All 32 accesses in parallel!

Bank conflict:
Thread 0 → Bank 0
Thread 1 → Bank 0  ← Conflict!
Thread 2 → Bank 0  ← Conflict!
Sequential access: 3× slower
```

*Conflict patterns:*

```c
__shared__ float smem[32][32];

// NO conflict: Sequential access
float val = smem[threadIdx.x][0];  // Threads access different banks

// NO conflict: Stride = 1
float val = smem[0][threadIdx.x];  // Threads access consecutive banks

// 32-WAY conflict: Stride = 32 (same bank for all threads!)
float val = smem[threadIdx.x][0];  // Column 0: all map to bank 0!

// Fix: Add padding
__shared__ float smem[32][33];     // 33 instead of 32
float val = smem[threadIdx.x][0];  // No conflict (stride = 33)
```

*Shared memory configuration:*

```c
// Configure shared memory / L1 cache split
cudaFuncSetAttribute(kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize, 100*1024);

// Prefer shared memory over L1
cudaFuncSetAttribute(kernel,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared);
```

== Global Memory

Global memory is the main GPU memory (VRAM), accessible by all threads but with high latency.

*Access patterns and coalescing:*

GPU memory transactions are 32, 64, or 128 bytes. For optimal performance, threads in a warp should access consecutive memory addresses (coalesced access).

```c
// COALESCED (optimal): Threads access consecutive addresses
__global__ void coalesced(float* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[tid];  // Thread i reads data[i]
}

// Memory transactions for 32 threads (1 warp):
// Addresses: 0, 4, 8, 12, ..., 124 (128 bytes total)
// Transactions: 1 × 128-byte load ← Optimal!
```

```c
// STRIDED (suboptimal): Threads access with stride
__global__ void strided(float* data, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[tid * stride];  // Thread i reads data[i*stride]
}

// Memory transactions for stride = 2:
// Addresses: 0, 8, 16, 24, ..., 248 (256 bytes needed)
// Transactions: 2 × 128-byte loads (50% utilization)

// Memory transactions for stride = 32:
// Addresses: 0, 128, 256, ... (scattered across memory)
// Transactions: 32 × 32-byte loads (worst case!)
```

```c
// RANDOM (worst case): Threads access random addresses
__global__ void random_access(float* data, int* indices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[indices[tid]];  // Random access pattern
}

// Each thread may access different cache line
// Up to 32 separate memory transactions!
```

*Coalescing visualization:*

```
Warp threads:   0   1   2   3   4   5   6   7  ...  31
                │   │   │   │   │   │   │   │       │
Coalesced:      ▼   ▼   ▼   ▼   ▼   ▼   ▼   ▼       ▼
              ┌─────────────────────────────────────────┐
              │             128-byte cache line          │
              └─────────────────────────────────────────┘
              One transaction, 100% utilization

Strided (2):    ▼       ▼       ▼       ▼       ...
              ┌───────────────────┐┌───────────────────┐
              │   Cache line 0    ││   Cache line 1    │
              └───────────────────┘└───────────────────┘
              Two transactions, 50% utilization

Random:         ▼ ▼  ▼    ▼   ▼▼      ▼  ...
              ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐
              │  ││  ││  ││  ││  ││  ││  │ ...
              └──┘└──┘└──┘└──┘└──┘└──┘└──┘
              Many transactions, poor utilization
```

*Coalesced vs uncoalesced bandwidth (measured, RTX 4090):*

```
Access Pattern          Effective BW    % of Peak (1008 GB/s)    Transactions/Warp
────────────────────────────────────────────────────────────────────────────────────
Coalesced (stride 1)    ~950 GB/s       94%                      1 × 128B
Stride 2                ~480 GB/s       48%                      2 × 128B
Stride 4                ~240 GB/s       24%                      4 × 128B
Stride 8                ~120 GB/s       12%                      8 × 128B
Stride 16               ~60 GB/s        6%                       16 × 32B
Stride 32 (worst)       ~30 GB/s        3%                       32 × 32B
Random (scatter)        ~25-35 GB/s     2.5-3.5%                 up to 32 × 32B
```

*Key takeaway:* Stride-32 access delivers $#sym.tilde.op$30x less bandwidth than coalesced access. A single uncoalesced kernel can reduce overall GPU throughput from near-peak to single-digit percentages.

*AoS vs SoA benchmark (1M particles, 6 floats each):*
```
Array of Structures (AoS):  stride = 6 floats = 24 bytes
  Reading x coordinates:    ~160 GB/s effective (16% of peak)
  4 × 128B transactions per warp (only 4B/32B useful per transaction)

Structure of Arrays (SoA):  stride = 1 float = 4 bytes
  Reading x coordinates:    ~940 GB/s effective (93% of peak)
  1 × 128B transaction per warp (all 128B useful)

Speedup: 5.9× for SoA over AoS on this access pattern
```

*Alignment requirements:*

```c
// Aligned allocation (optimal)
float* d_data;
cudaMalloc(&d_data, n * sizeof(float));  // Automatically 256-byte aligned

// Misaligned access (suboptimal)
float* ptr = d_data + 1;  // Offset by 4 bytes
// Access now crosses cache line boundaries → extra transactions

// Solution: Ensure base address alignment
__align__(128) float smem[256];  // Force alignment in shared memory
```

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

    // Warp-level reduction (no sync needed)
    if (tid < 32) {
        volatile float* vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

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

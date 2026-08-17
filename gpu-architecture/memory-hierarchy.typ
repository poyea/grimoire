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
│    │  │    Shared Memory (SMEM)       │  Up to 100 KB per SM    │  │
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
│    │  │    PCIe: 64 GB/s (Gen5 x16)   │  ~1-2 µs (wire);        │  │
│    │  │                               │  ~10-20 µs (cudaMemcpy) │  │
│    │  └───────────────────────────────┘                          │  │
│    └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

*Memory characteristics (Ada Lovelace / RTX 4090):*

```
Memory Type      Size           Latency        Bandwidth        Scope
─────────────────────────────────────────────────────────────────────────
Registers        256 KB/SM      0 cy (issue)   ~20 TB/s/SM      Thread
                                4-6 cy (RAW)
Shared Memory    Up to 100 KB   ~20-30 cy      ~2 TB/s/SM       Block
L1 Cache         128 KB/SM      ~20-30 cy      ~2 TB/s/SM       SM
                                (Volta+: L1 and shared are the same SRAM bank)
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

GPU memory transactions are 32, 64, or 128 bytes. On Volta+ (sm_70+) the L1 cache operates on 32-byte sectors; older architectures used a 128-byte granularity. For optimal performance, threads in a warp should access consecutive memory addresses (coalesced access).

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

*Additional access pattern benchmarks (RTX 4090):*

*Pattern: Matrix Multiply (naive vs tiled)*

```c
// NAIVE: Each thread reads full row/column from global memory
// C[i][j] = sum(A[i][k] * B[k][j]) for k in [0, N)
// A row access: coalesced. B column access: stride = N (non-coalesced!)
__global__ void matmul_naive(float* C, float* A, float* B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int k = 0; k < N; k++)
        sum += A[row * N + k] * B[k * N + col];  // B access stride = N
    C[row * N + col] = sum;
}
// Effective BW: ~200 GB/s (20% of peak) — B column reads non-coalesced

// TILED: Load tiles into shared memory, reuse N/32 times
__global__ void matmul_tiled(float* C, float* A, float* B, int N) {
    __shared__ float As[32][32], Bs[32][33];  // Padded to avoid bank conflicts
    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    float sum = 0;
    for (int t = 0; t < N; t += 32) {
        As[threadIdx.y][threadIdx.x] = A[row * N + t + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        __syncthreads();
        for (int k = 0; k < 32; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = sum;
}
// Effective BW: ~900 GB/s (89% of peak) — all global loads coalesced, 32× reuse in SMEM
```

*Pattern: Parallel Reduction (addressing order)*

```c
// INTERLEAVED (non-coalesced in early iterations):
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0)
        sdata[tid] += sdata[tid + s];  // Stride doubles each step
    __syncthreads();
}
// Step 1: threads 0,2,4,6... active (stride 2) → ~400 GB/s effective
// Warp divergence + non-sequential access pattern

// SEQUENTIAL (coalesced):
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
        sdata[tid] += sdata[tid + s];  // Contiguous threads active
    __syncthreads();
}
// Threads 0..s-1 active (contiguous) → ~850 GB/s effective
// 2.1× faster: no warp divergence, sequential SMEM access
```

*Pattern: Histogram (scatter vs privatized)*

```c
// GLOBAL ATOMICS (random scatter):
__global__ void histogram_global(int* data, int* bins, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        atomicAdd(&bins[data[tid]], 1);  // Random bin → serialized
}
// Effective BW: ~15-25 GB/s (1.5-2.5%) — atomic contention at popular bins

// PRIVATIZED (shared memory atomics + merge):
__global__ void histogram_private(int* data, int* bins, int n, int num_bins) {
    extern __shared__ int local_bins[];
    int tid_local = threadIdx.x;
    for (int i = tid_local; i < num_bins; i += blockDim.x)
        local_bins[i] = 0;
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        atomicAdd(&local_bins[data[tid]], 1);  // SMEM atomic: ~5 cycles
    __syncthreads();

    for (int i = tid_local; i < num_bins; i += blockDim.x)
        atomicAdd(&bins[i], local_bins[i]);  // Merge: few global atomics
}
// Effective BW: ~300 GB/s (30%) — 12-20× faster than global atomics
// SMEM atomics ~5 cycles vs global atomics ~100+ cycles under contention
```

*Access pattern benchmark summary (RTX 4090):*

#table(
  columns: (auto, auto, auto, auto),
  [*Pattern*], [*Naive BW*], [*Optimized BW*], [*Technique*],
  [Matrix multiply], [~200 GB/s], [~900 GB/s], [Shared memory tiling],
  [Reduction], [~400 GB/s], [~850 GB/s], [Sequential addressing],
  [Histogram], [~20 GB/s], [~300 GB/s], [Privatized shared memory],
  [AoS → SoA], [~160 GB/s], [~940 GB/s], [Data layout transform],
)

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


== Further Reading

NVIDIA. _CUDA C++ Programming Guide_. (Authoritative reference for the memory model, coalescing rules, and cache behaviour.)

Jia, Z., Maggioni, M., Staiger, B., & Scarpazza, D. P. (2018). "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking." arXiv:1804.06826. (Measured cache and memory latencies at each level.)

Volkov, V., & Demmel, J. W. (2008). "Benchmarking GPUs to Tune Dense Linear Algebra." SC. (Classic study of occupancy and register/shared-memory tradeoffs.)

Hennessy, J. L., & Patterson, D. A. (2019). _Computer Architecture: A Quantitative Approach_, 6th ed. Morgan Kaufmann. (Chapter 4 on data-parallel memory systems.)

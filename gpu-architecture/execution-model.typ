= SIMT Execution Model

GPUs execute thousands of threads using Single Instruction Multiple Thread (SIMT) execution. Understanding warps, thread divergence, and occupancy is critical for writing efficient GPU code.

*See also:* GPU Fundamentals (for architecture overview), Memory Hierarchy (for memory access patterns), Performance Optimization (for tuning techniques)

== Thread Hierarchy

CUDA organizes threads in a three-level hierarchy: threads, blocks (thread blocks), and grids.

```
Grid (kernel launch)
├── Block (0,0)           Block (1,0)           Block (2,0)
│   ├── Thread (0,0)      ├── Thread (0,0)      ├── Thread (0,0)
│   ├── Thread (0,1)      ├── Thread (0,1)      ├── Thread (0,1)
│   ├── Thread (1,0)      ├── Thread (1,0)      ├── Thread (1,0)
│   ├── Thread (1,1)      ├── Thread (1,1)      ├── Thread (1,1)
│   └── ...               └── ...               └── ...
├── Block (0,1)           Block (1,1)           Block (2,1)
│   └── ...               └── ...               └── ...
└── ...
```

*Thread identification:*

```c
// 1D indexing (most common)
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// 2D indexing (images, matrices)
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int idx = y * width + x;

// 3D indexing (volumes, tensors)
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

*Dimension limits (Compute Capability 8.x+):*

```
Threads per block:     1024 (max)
Block dimensions:      (1024, 1024, 64) max per dimension
Grid dimensions:       (2^31-1, 65535, 65535) max per dimension
Blocks per SM:         16-32 (architecture dependent)
```

== Warps: The Fundamental Execution Unit

A warp is a group of 32 threads that execute instructions in lockstep (SIMT). All threads in a warp share a single program counter and execute the same instruction simultaneously.

```
Block (256 threads)
├── Warp 0:  Threads 0-31
├── Warp 1:  Threads 32-63
├── Warp 2:  Threads 64-95
├── Warp 3:  Threads 96-127
├── Warp 4:  Threads 128-159
├── Warp 5:  Threads 160-191
├── Warp 6:  Threads 192-223
└── Warp 7:  Threads 224-255
```

*Warp execution:*

```
Cycle 0:  Warp 0 executes instruction 0
Cycle 1:  Warp 0 executes instruction 1
Cycle 2:  Warp 0 stalls (memory access)
          Warp 1 executes instruction 0
Cycle 3:  Warp 1 executes instruction 1
Cycle 4:  Warp 0 resumes (data arrived)
...
```

*Latency hiding:* GPU switches between warps to hide memory latency. While one warp waits for data, another executes.

```
Memory latency: ~400 cycles
Instruction latency: ~20 cycles
Required warps for full latency hiding: 400 / 20 = 20 warps

More resident warps → Better latency hiding → Higher throughput
```

== Warp Divergence

When threads in a warp take different control flow paths, the warp must execute both paths sequentially, with inactive threads masked. This is called warp divergence.

```c
// Divergent code
__global__ void divergent(int* data, int* result) {
    int tid = threadIdx.x;
    if (tid % 2 == 0) {
        result[tid] = data[tid] * 2;    // Even threads
    } else {
        result[tid] = data[tid] + 1;    // Odd threads
    }
}

// Execution timeline for one warp (32 threads):
// Pass 1: Threads 0,2,4,...,30 execute (16 active), others masked
// Pass 2: Threads 1,3,5,...,31 execute (16 active), others masked
// Result: 2× slower than non-divergent code
```

*Divergence visualization:*

```
Thread:  0  1  2  3  4  5  6  7  8  9 10 11 ... 30 31
         ─────────────────────────────────────────────
Cond:    T  F  T  F  T  F  T  F  T  F  T  F ...  T  F
         ─────────────────────────────────────────────
Pass 1:  ●  ○  ●  ○  ●  ○  ●  ○  ●  ○  ●  ○ ...  ●  ○
Pass 2:  ○  ●  ○  ●  ○  ●  ○  ●  ○  ●  ○  ● ...  ○  ●
         ─────────────────────────────────────────────
● = Active (executing)
○ = Masked (waiting)
```

*No divergence across warps:* Different warps can take different paths without penalty.

```c
// This is NOT divergent (condition based on warp, not thread)
if (blockIdx.x % 2 == 0) {
    // All threads in even blocks execute this
} else {
    // All threads in odd blocks execute this
}
```

*Minimizing divergence:*

```c
// BAD: High divergence (random condition)
if (data[tid] > threshold) {
    expensive_computation();
}
// Divergence: Unpredictable, depends on data

// BETTER: Sort data first, then process
// Threads processing similar data → less divergence

// BEST: Predication (when both paths are short)
int result = (tid % 2 == 0) ? data[tid] * 2 : data[tid] + 1;
// Compiler may use predicated execution, no divergence
```

== Independent Thread Scheduling (Volta+)

Starting with Volta architecture, threads within a warp can execute independently, enabling finer-grained synchronization.

*Pre-Volta:* All threads in warp share a single program counter. Cannot synchronize within a warp.

*Volta+:* Each thread has its own program counter and call stack. Threads can diverge and reconverge at arbitrary points.

```c
// Works on Volta+ (dangerous on older architectures!)
__global__ void volta_sync() {
    int tid = threadIdx.x;

    if (tid < 16) {
        // First half of warp
        do_work_a();
        __syncwarp(0x0000FFFF);  // Sync first 16 threads
    } else {
        // Second half of warp
        do_work_b();
        __syncwarp(0xFFFF0000);  // Sync last 16 threads
    }
}
```

*Cooperative groups (CUDA 9+):* Explicit thread grouping for synchronization.

```c
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cooperative_example() {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Warp-level operations
    int sum = cg::reduce(warp, threadIdx.x, cg::plus<int>());

    // Sync within warp
    warp.sync();

    // Ballot across warp
    unsigned mask = warp.ballot(threadIdx.x > 16);
}
```

== Warp-Level Primitives

*Warp shuffle:* Exchange data between threads without shared memory.

```c
// __shfl_sync: Read from any lane
int value = __shfl_sync(0xFFFFFFFF, source, lane_id);

// __shfl_up_sync: Read from lower lane
int value = __shfl_up_sync(0xFFFFFFFF, source, delta);

// __shfl_down_sync: Read from higher lane
int value = __shfl_down_sync(0xFFFFFFFF, source, delta);

// __shfl_xor_sync: Read from XOR'd lane
int value = __shfl_xor_sync(0xFFFFFFFF, source, lane_mask);
```

*Warp reduction using shuffle:*

```c
__device__ int warp_reduce_sum(int val) {
    // Butterfly reduction pattern
    val += __shfl_xor_sync(0xFFFFFFFF, val, 16);  // Add with lane ± 16
    val += __shfl_xor_sync(0xFFFFFFFF, val, 8);   // Add with lane ± 8
    val += __shfl_xor_sync(0xFFFFFFFF, val, 4);   // Add with lane ± 4
    val += __shfl_xor_sync(0xFFFFFFFF, val, 2);   // Add with lane ± 2
    val += __shfl_xor_sync(0xFFFFFFFF, val, 1);   // Add with lane ± 1
    return val;  // All lanes have the sum
}

// Visualization (8 lanes for simplicity):
// Initial: [a, b, c, d, e, f, g, h]
// XOR 4:   [a+e, b+f, c+g, d+h, e+a, f+b, g+c, h+d]
// XOR 2:   [a+e+c+g, b+f+d+h, ...]
// XOR 1:   [sum, sum, sum, sum, sum, sum, sum, sum]
```

*Warp vote functions:*

```c
// All threads satisfy predicate?
bool all = __all_sync(0xFFFFFFFF, predicate);

// Any thread satisfies predicate?
bool any = __any_sync(0xFFFFFFFF, predicate);

// Ballot: Get bitmask of threads satisfying predicate
unsigned mask = __ballot_sync(0xFFFFFFFF, predicate);

// Population count: Number of threads satisfying predicate
int count = __popc(__ballot_sync(0xFFFFFFFF, predicate));
```

*Example: Find first active lane*

```c
__device__ int find_first_active(bool active) {
    unsigned mask = __ballot_sync(0xFFFFFFFF, active);
    return __ffs(mask) - 1;  // __ffs returns 1-indexed, -1 for 0-indexed
}
```

== Occupancy

Occupancy is the ratio of active warps to maximum possible warps on an SM. Higher occupancy generally means better latency hiding, but not always better performance.

```
Occupancy = Active warps per SM / Maximum warps per SM

Example (Ada Lovelace SM):
- Maximum warps per SM: 48
- Active warps: 32
- Occupancy: 32 / 48 = 66.7%
```

*Factors limiting occupancy:*

1. *Threads per block:* Must be multiple of 32, max 1024.
2. *Registers per thread:* Limited register file (65536 registers per SM).
3. *Shared memory per block:* Limited shared memory (up to 164 KB per SM).

```
Occupancy calculation example:

SM resources (Ada Lovelace):
- Max threads: 1536
- Max warps: 48
- Registers: 65,536
- Shared memory: 100 KB (configurable)

Kernel requirements:
- Block size: 256 threads (8 warps per block)
- Registers per thread: 48
- Shared memory: 16 KB per block

Limit 1 (threads): 1536 / 256 = 6 blocks
Limit 2 (registers): 65536 / (256 × 48) = 5.3 → 5 blocks
Limit 3 (shared mem): 100 KB / 16 KB = 6.25 → 6 blocks

Limiting factor: Registers (5 blocks)
Active warps: 5 blocks × 8 warps = 40 warps
Occupancy: 40 / 48 = 83.3%
```

*Occupancy calculator:*

```bash
# CUDA occupancy calculator
nvcc --ptxas-options=-v kernel.cu
# Outputs register count, shared memory usage

# Use CUDA Occupancy API
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks, kernel, blockSize, sharedMemSize);

# Or use NVIDIA Nsight Compute
ncu --set full ./program
```

*Occupancy vs. performance:*

```
                    ┌─────────────────────────────────────
         Perf      │           ╭─────────────────────
                    │         ╭╯
                    │       ╭╯    Diminishing returns
                    │     ╭╯
                    │   ╭╯
                    │ ╭╯
                    │╯
                    └─────────────────────────────────────
                       0%   25%   50%   75%   100%
                              Occupancy

Higher occupancy helps latency-bound kernels.
Compute-bound kernels may not benefit from >50% occupancy.
Optimal occupancy is workload-dependent.
```

== Block Scheduling

*Block distribution:* Blocks are scheduled to SMs in round-robin fashion (simplified view). Exact scheduling is hardware-dependent and not guaranteed.

```
Grid: 100 blocks
GPU: 20 SMs

Initial distribution (approximate):
SM 0:  Blocks 0, 20, 40, 60, 80
SM 1:  Blocks 1, 21, 41, 61, 81
SM 2:  Blocks 2, 22, 42, 62, 82
...
SM 19: Blocks 19, 39, 59, 79, 99
```

*Tail effect:* When blocks don't divide evenly across SMs, some SMs finish early.

```c
// BAD: 129 blocks, 128 SMs → 1 SM runs twice as long
vectorAdd<<<129, 256>>>(a, b, c, n);

// BETTER: Pad work to avoid tail effect
int numBlocks = ((n + 255) / 256 + 127) & ~127;  // Round to 128
```

*Persistent threads:* Keep threads running to avoid scheduling overhead.

```c
__global__ void persistent_kernel(int* work_queue, int* results) {
    __shared__ int block_work_index;

    while (true) {
        // Block leader gets next work item
        if (threadIdx.x == 0) {
            block_work_index = atomicAdd(&global_work_index, 1);
        }
        __syncthreads();

        int work = block_work_index;
        if (work >= total_work) return;

        // Process work item
        process(work_queue[work], results);
    }
}
```

== Synchronization

*Block-level synchronization:*

```c
__syncthreads();  // Barrier for all threads in block
// All threads must reach this point before any proceed
// Also serves as memory fence (shared memory visible to all)
```

*Warp-level synchronization:*

```c
__syncwarp(mask);  // Barrier for threads in mask (Volta+)
// Use 0xFFFFFFFF for all threads in warp
```

*Grid-level synchronization (cooperative launch):*

```c
#include <cooperative_groups.h>

__global__ void grid_sync_kernel() {
    cg::grid_group grid = cg::this_grid();

    // Phase 1: All blocks do work
    do_phase1();

    // Grid-wide barrier
    grid.sync();

    // Phase 2: All blocks see Phase 1 results
    do_phase2();
}

// Launch with cooperative kernel
cudaLaunchCooperativeKernel((void*)grid_sync_kernel,
                            numBlocks, blockSize, args);
```

*Synchronization costs:*

```
__syncthreads():  ~20-40 cycles (block barrier)
__syncwarp():     ~4-8 cycles (warp barrier)
Grid sync:        ~1000+ cycles (all SMs must synchronize)
cudaDeviceSynchronize(): Host-device sync, ~5-10 µs
```

== Thread Divergence Patterns

*Pattern 1: Boundary handling*

```c
// BAD: Divergence on every boundary check
__global__ void kernel(int* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {  // Divergence in last block
        data[tid] = compute(tid);
    }
}

// BETTER: Launch exact number of threads (when possible)
// Or accept minor divergence in last block
```

*Pattern 2: Reduction*

```c
// BAD: High divergence
__global__ void reduce_bad(int* data, int* result, int n) {
    int tid = threadIdx.x;
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {  // Divergent!
            data[tid] += data[tid + s];
        }
        __syncthreads();
    }
}

// GOOD: Sequential addressing (divergence-free within active warps)
__global__ void reduce_good(int* data, int* result, int n) {
    int tid = threadIdx.x;
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {  // First s threads active
            data[tid] += data[tid + s];
        }
        __syncthreads();
    }
}

// Divergence comparison:
// Bad:  Iteration 1: threads 0,2,4,6... (50% utilization per warp)
// Good: Iteration 1: threads 0-127 (full warps, no intra-warp divergence)
```

*Pattern 3: Conditional computation*

```c
// BAD: Expensive divergent branch
if (condition[tid]) {
    result[tid] = expensive_function(data[tid]);
} else {
    result[tid] = 0;
}

// BETTER: Compute all, select result (if cheap enough)
float computed = expensive_function(data[tid]);
result[tid] = condition[tid] ? computed : 0;

// BEST: Compact and process (stream compaction)
// Separate threads into groups based on condition
// Process each group with full warp utilization
```

== AMD Wavefront Differences

AMD uses wavefronts instead of warps, with key differences:

```
                  NVIDIA (CUDA)    AMD (HIP/ROCm)
─────────────────────────────────────────────────
Thread group       Warp             Wavefront
Size               32 threads       64 threads (RDNA: 32)
Terminology        Lane             Lane
Shuffle            __shfl_sync      __shfl
Ballot             __ballot_sync    __ballot

// HIP code (portable)
#ifdef __HIP_PLATFORM_AMD__
    const int WAVE_SIZE = 64;  // or 32 for RDNA
#else
    const int WAVE_SIZE = 32;
#endif
```

*Wave32 vs Wave64:* RDNA architecture supports both 32 and 64-thread wavefronts, selectable at compile time.

```bash
# AMD compile for Wave32
hipcc --offload-arch=gfx1100 -mwavefrontsize64=off kernel.cpp

# AMD compile for Wave64
hipcc --offload-arch=gfx1100 -mwavefrontsize64=on kernel.cpp
```

== References

NVIDIA Corporation (2024). CUDA C++ Programming Guide. Chapter 4 (Hardware Implementation). https://docs.nvidia.com/cuda/cuda-c-programming-guide/

NVIDIA Corporation (2024). Parallel Thread Execution ISA. Version 8.4. https://docs.nvidia.com/cuda/parallel-thread-execution/

Volkov, V. (2010). "Better Performance at Lower Occupancy." GPU Technology Conference.

Micikevicius, P. (2012). "GPU Performance Analysis and Optimization." GPU Technology Conference.

AMD (2024). HIP Programming Guide. https://rocm.docs.amd.com/projects/HIP/

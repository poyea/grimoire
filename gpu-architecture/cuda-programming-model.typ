= CUDA Programming Model

CUDA exposes the GPU as a heterogeneous co-processor with an explicit host/device split, an SPMD kernel model, and a hierarchy of execution scopes (thread, warp, block, cluster, grid). Mastering the launch surface and the warp-level primitives is what separates correct CUDA from fast CUDA.

*See also:* _Execution Model_ (SIMT, warps, divergence), _Memory Hierarchy_ (shared/global memory), _Performance Optimization_ (occupancy, launch tuning).

== The Heterogeneous Model

A CUDA program runs on the *host* (CPU) and offloads parallel sections to the *device* (GPU). Functions are annotated by execution space:

```cpp
__host__   void run_on_cpu();
__device__ int  run_on_gpu(int x);             // callable from device only
__global__ void kernel(float* p, int n);       // host launches on device
__host__ __device__ float helper(float x);     // both
```

Memory is also split: `cudaMalloc` / `cudaMallocManaged` / `cudaMallocHost`, with explicit `cudaMemcpyAsync` transfers (or Unified Memory page faults).

== Kernel Launch

Launching a kernel is the central act of CUDA programming:

```cpp
dim3 block(128);                  // 128 threads per block
dim3 grid((N + 127) / 128);       // enough blocks to cover N
size_t smem = 0;                  // dynamic shared memory bytes
cudaStream_t stream = 0;
kernel<<<grid, block, smem, stream>>>(d_p, N);
```

The triple-angle-bracket syntax compiles to `cudaLaunchKernel` (or the more modern `cudaLaunchKernelEx`). Launch latency is typically 5–10 us host-side and ~1 us device-side; for very small kernels this dominates, motivating CUDA Graphs (below).

=== Grid / Block / Thread Hierarchy

```
Grid                                       (1D / 2D / 3D)
 |
 +-- Block                                 (up to 1024 threads)
 |    |
 |    +-- Warp (32 threads, SIMT lockstep)
 |    +-- Warp
 |    ...
 +-- Block
 ...
```

Each thread can identify itself:
```cpp
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int gid = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
```

*Limits* (compute capability $>=$ 7.0):

#table(
  columns: 3,
  [*Resource*], [*Per block*], [*Per SM*],
  [Threads], [1024], [2048],
  [Warps], [32], [64 (Volta), 48 (Ada)],
  [Registers (32-bit)], [255 per thread], [65536],
  [Shared memory], [up to 100+ KB], [100–228 KB],
  [Blocks], [—], [16–32],
)

=== Thread Block Clusters (Hopper, sm_90)

Hopper adds a level above the block: a *cluster* of up to 16 (portably 8) blocks scheduled co-resident on a GPC, with *distributed shared memory* — any block in the cluster can read/write any other block's shared memory.

```cpp
__cluster_dims__(2, 2, 1)
__global__ void cluster_kernel(...) {
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();
    cluster.sync();
    float* peer_smem = cluster.map_shared_rank(my_smem, rank ^ 1);
}
```

== Streams and Concurrency

A *stream* is an ordered queue of work on the device. Operations in different streams may execute concurrently (overlapping copy and compute, or multiple kernels on the same GPU).

```cpp
cudaStream_t s1, s2;
cudaStreamCreate(&s1); cudaStreamCreate(&s2);
cudaMemcpyAsync(d_a, h_a, n, cudaMemcpyHostToDevice, s1);
kernel<<<g, b, 0, s1>>>(d_a);
cudaMemcpyAsync(h_b, d_b, n, cudaMemcpyDeviceToHost, s2);
```

*Priorities:* `cudaStreamCreateWithPriority` requests a high-priority stream that preempts low-priority work at block-scheduling boundaries.

*Events:* lightweight synchronization and timing markers.
```cpp
cudaEvent_t e; cudaEventCreate(&e);
cudaEventRecord(e, s1);
cudaStreamWaitEvent(s2, e, 0);   // s2 waits for e
```

*The default (NULL) stream* is legacy-synchronizing — it implicitly synchronizes with all other streams. Compile with `--default-stream per-thread` to get per-thread default streams that do not block one another.

== CUDA Graphs

Streams describe work one launch at a time, paying host launch latency for each. *CUDA Graphs* capture a DAG of operations once and replay it with sub-microsecond launch overhead.

```cpp
cudaGraph_t graph;
cudaGraphExec_t exec;

cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
kernel_a<<<g,b,0,s>>>(...);
cudaMemcpyAsync(..., s);
kernel_b<<<g,b,0,s>>>(...);
cudaStreamEndCapture(s, &graph);

cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

for (int step = 0; step < 1000; step++)
    cudaGraphLaunch(exec, s);          // ~1-2 us per launch
```

Graphs are essential for inference servers and small-kernel pipelines (LLM decoding, where the per-token kernel chain may be 50+ launches).

*Updates:* `cudaGraphExecKernelNodeSetParams` and `cudaGraphExecUpdate` mutate node parameters without re-instantiating, useful when only pointers or scalars change.

== Cooperative Groups

A modern abstraction over the legacy `__syncthreads()` / `__ballot()` primitives. Groups are first-class objects representing thread collectives at various granularities.

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void k() {
    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);
    auto quad  = cg::tiled_partition<4>(warp);

    int sum = cg::reduce(warp, threadIdx.x, cg::plus<int>());
    block.sync();
}
```

*Grid groups* and *multi-grid groups* enable grid-wide and multi-GPU synchronization, but require *cooperative launch* (`cudaLaunchCooperativeKernel`) and limit occupancy to what can be co-resident.

== Warp-Level Primitives

The warp (32 threads) is the true unit of SIMT execution. Warp-level intrinsics exchange data and votes without touching shared memory.

=== Shuffles

```cpp
int v = threadIdx.x;
int up    = __shfl_up_sync   (0xffffffff, v, 1);     // shift up by 1 lane
int down  = __shfl_down_sync (0xffffffff, v, 1);
int xor_  = __shfl_xor_sync  (0xffffffff, v, 1);     // butterfly
int bcast = __shfl_sync      (0xffffffff, v, 0);     // lane 0 broadcasts
```

Warp-level reduction in 5 instructions:
```cpp
__device__ int warp_reduce(int v) {
    for (int o = 16; o > 0; o /= 2)
        v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
```

=== Vote and Ballot

```cpp
unsigned mask = __ballot_sync(0xffffffff, predicate);   // 32-bit lane mask
bool any_set  = __any_sync   (0xffffffff, predicate);
bool all_set  = __all_sync   (0xffffffff, predicate);
int  popcount = __popc(mask);
```

Common pattern — *warp-aggregated atomics* (reduce 32 atomics to 1):
```cpp
__device__ int atomic_inc_warp(int* ctr) {
    auto warp = cg::coalesced_threads();
    int prev;
    if (warp.thread_rank() == 0) prev = atomicAdd(ctr, warp.size());
    prev = warp.shfl(prev, 0);
    return prev + warp.thread_rank();
}
```

=== Match and Sync

Volta+ added `__match_any_sync` / `__match_all_sync` to find lanes with equal values — useful for histogram, conflict detection, and indirect addressing.

```cpp
unsigned peers = __match_any_sync(0xffffffff, key);
int leader = __ffs(peers) - 1;
```

*Sync suffix:* every warp primitive must take an explicit mask after Volta because Independent Thread Scheduling no longer guarantees lockstep convergence.

== Launch Bounds and Occupancy

Without hints, `nvcc` chooses register budgets that may starve occupancy. `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)` tells the compiler to spill rather than exceed a register budget.

```cpp
__launch_bounds__(256, 4)            // promise: <=256 threads, >=4 blocks/SM
__global__ void k(...) { ... }
```

Occupancy can also be queried and used to size launches:
```cpp
int blocks_per_sm;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, k, 256, /*smem=*/0);
```

The *occupancy calculator API* (`cudaOccupancyMaxPotentialBlockSize`) returns a (block size, grid size) pair maximizing occupancy for a given kernel.

== Persistent and Cooperative Kernels

For workloads with many small phases, *persistent kernels* allocate one block per SM that loops over work items pulled from a global queue, avoiding launch overhead entirely. Cooperative launch (`cudaLaunchCooperativeKernel`) enables a single grid-wide `cg::grid_group::sync()` inside such a kernel.

```cpp
__global__ void persistent(...) {
    auto grid = cg::this_grid();
    while (auto item = work_queue.pop()) {
        process(item);
        grid.sync();           // global barrier within the kernel
    }
}
```

== Putting It Together: a Tuned Launch

```cpp
template<int BLOCK>
__launch_bounds__(BLOCK, 4)
__global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

void launch_saxpy(float a, const float* x, float* y, int n,
                  cudaStream_t s) {
    constexpr int B = 256;
    int grid = (n + B - 1) / B;
    saxpy<B><<<grid, B, 0, s>>>(a, x, y, n);
}
```

A typical CUDA performance checklist for any kernel launch:
+ Block size $>=$ 128 and a multiple of 32.
+ Enough blocks for $>=$ 4$times$ the SM count (latency hiding).
+ `__launch_bounds__` set; check register count with `--ptxas-options=-v`.
+ Use streams + graphs for many small kernels.
+ Prefer warp shuffles to shared memory for reductions of size $<=$ 32.

== Further Reading

NVIDIA Corporation (2024). _CUDA C++ Programming Guide_, Chapter 5 (Programming Interface) and Appendix B (C++ Language Extensions). https://docs.nvidia.com/cuda/cuda-c-programming-guide/

Harris, M. (2019). "Constant-Time CUDA Graphs." NVIDIA Developer Blog.

Luitjens, J. (2014). "Faster Parallel Reductions on Kepler." NVIDIA Developer Blog (warp shuffles).

NVIDIA (2022). _NVIDIA H100 Tensor Core GPU Architecture Whitepaper_, Section on Thread Block Clusters.

Harris, M. & Perelygin, K. (2017). "Cooperative Groups: Flexible CUDA Thread Programming." NVIDIA Developer Blog.

Jones, S. (2021). "How CUDA Programming Works." GTC 2021 talk S31151.

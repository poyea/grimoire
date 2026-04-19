= Multi-GPU Communication and Scaling

Modern ML training (LLMs with 100B+ parameters) and HPC simulations no longer fit on a single GPU. Scaling requires high-bandwidth GPU-to-GPU interconnects, efficient collective communication libraries, and parallelism strategies that map workload structure onto hardware topology.

*See also:* _memory-hierarchy.typ_ (HBM, global memory), _compute-architecture.typ_ (Hopper/Blackwell interconnect), _performance-optimization.typ_ (kernel tuning).

== Interconnect Hierarchy

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Interconnect*], [*Bandwidth (per GPU)*], [*Latency*], [*Topology*], [*Use*],
  [PCIe Gen4 x16], [32 GB/s], [$tilde$ 1 $mu$s], [Tree (host-centric)], [Commodity hosts],
  [PCIe Gen5 x16], [64 GB/s], [$tilde$ 1 $mu$s], [Tree], [Grace-Hopper boards],
  [NVLink 3 (A100)], [600 GB/s], [$tilde$ 700 ns], [Switched / full-mesh], [DGX A100 (8-GPU)],
  [NVLink 4 (H100)], [900 GB/s], [$tilde$ 500 ns], [NVSwitch gen3], [DGX H100 (8-GPU)],
  [NVLink 5 (B100/B200)], [1800 GB/s], [$tilde$ 400 ns], [NVSwitch gen4], [GB200 NVL72 (72-GPU)],
  [AMD Infinity Fabric (MI300)], [896 GB/s], [$tilde$ 600 ns], [Mesh], [MI300X 8-GPU systems],
  [InfiniBand NDR], [400 Gb/s (50 GB/s)], [$tilde$ 1-2 $mu$s], [Fat-tree / dragonfly], [Multi-node HPC/ML clusters],
  [InfiniBand XDR], [800 Gb/s (100 GB/s)], [$tilde$ 1 $mu$s], [Fat-tree], [Blackwell-era clusters],
  [Ethernet RoCEv2 400G], [50 GB/s], [$tilde$ 2-5 $mu$s], [Any], [Hyperscaler clusters],
)

*Bandwidth vs latency tradeoff:* intra-node (NVLink) beats inter-node (IB) by 10-40x in bandwidth. A single all-reduce that fits within a node is 10-100x faster than one that crosses nodes.

== GPU Topologies

*DGX H100 (single node, 8 GPU):*
```
         ┌─────── NVSwitch gen3 (×4) ───────┐
         │  (all-to-all, 900 GB/s per link)  │
         └───┬───┬───┬───┬───┬───┬───┬───┬──┘
             │   │   │   │   │   │   │   │
           GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
             │   │   │   │   │   │   │   │
         ┌───┴───┴───┴───┴───┴───┴───┴───┴──┐
         │         ConnectX-7 NIC (×8)      │ ── 400 Gb/s IB
         └───────────────────────────────────┘
```
Each GPU gets 18 NVLink lanes at 50 GB/s = 900 GB/s total. NVSwitch provides non-blocking all-to-all.

*GB200 NVL72 (rack-scale):*
- 72 Blackwell GPUs + 36 Grace CPUs in a single liquid-cooled rack
- NVLink 5 switch fabric provides 1800 GB/s per GPU, all-to-all bisection bandwidth
- Presents as a single logical accelerator domain for CUDA; 30 TB unified HBM
- 1.4 exaFLOPS FP4 dense (NVIDIA claim)

*Hands-on — inspect topology:*
```bash
# Show NVLink topology
nvidia-smi topo -m
#        GPU0 GPU1 GPU2 GPU3 CPU Affinity
# GPU0    X   NV18 NV18 NV18   0-55
# GPU1  NV18   X   NV18 NV18   0-55
# ...
# Legend: NV18 = NVLink (18 lanes), PIX = PCIe switch, SYS = QPI/UPI,
#         NODE = NUMA hop, PHB = PCIe host bridge

# Per-link NVLink state and utilization
nvidia-smi nvlink -s
nvidia-smi nvlink --getpower -i 0
```

== GPUDirect

*GPUDirect P2P:* direct GPU-to-GPU memory copy without CPU/host DRAM bounce.
```cpp
int can_access;
cudaDeviceCanAccessPeer(&can_access, /*dev=*/0, /*peer=*/1);
if (can_access) {
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(/*peer=*/1, 0);
    // Now cudaMemcpy(dst_on_dev1, src_on_dev0, ...) uses NVLink directly
    cudaMemcpy(dst, src, N, cudaMemcpyDeviceToDevice);
}
```
Without P2P: copy goes GPU0 → host DRAM → GPU1 (~16 GB/s PCIe cap).
With P2P over NVLink 4: 700-900 GB/s.

*GPUDirect RDMA:* NIC DMAs directly into GPU HBM (no bounce through host). Requires Mellanox ConnectX + NVIDIA driver + kernel module. Eliminates 2 memory copies, 5-10 $mu$s latency on IB.

*GPUDirect Storage (cuFile):* NVMe storage DMAs directly to GPU memory. For training data loading / checkpoint restore.
```cpp
CUfileHandle_t fh;
cuFileHandleRegister(&fh, &descr);
cuFileRead(fh, gpu_buf, /*size=*/1<<30, /*file_off=*/0, /*dev_off=*/0);
```
Bypasses page cache. 2-5x faster than `read()` + `cudaMemcpy`.

== NCCL: Collective Communications

*NCCL* (NVIDIA Collective Communications Library) provides topology-aware collective primitives: `all_reduce`, `broadcast`, `reduce`, `all_gather`, `reduce_scatter`, `send`/`recv`.

*Ring all-reduce algorithm* (bandwidth-optimal):

For $p$ GPUs each holding $n$ bytes:
1. Reduce-scatter phase: $p-1$ steps, each GPU sends 1/$p$ of data to neighbor. After $p-1$ steps, each GPU has reduced 1/$p$ of the total.
2. All-gather phase: $p-1$ steps, each GPU forwards its reduced chunk around the ring.

Total data transferred per GPU: $2 (p-1) n / p$.

$ T_"ring" = 2 (p-1) alpha + 2 (p-1) / p dot n / beta $

where $alpha$ is per-step latency, $beta$ is link bandwidth.

As $p -> infinity$: bandwidth term $-> 2 n / beta$ — independent of $p$ — bandwidth-optimal.

*Tree all-reduce* (latency-optimal on large clusters):
- Reduce up a binary tree, broadcast back down
- $T_"tree" = 2 log_2 p dot alpha + 2 log_2 p dot n / beta$
- Latency scales as $log p$ but bandwidth term multiplies by $log p$ — better for small $n$ or large $p$
- NCCL uses _double binary tree_ (Sanders et al. 2009): two interleaved trees, each node leaf in one tree and internal in the other — doubles effective bandwidth

*SHARP (Scalable Hierarchical Aggregation and Reduction Protocol):* NVIDIA IB switches offload reduction operations in-network. The switch itself performs the reduction, sending only the reduced value to each recipient — saves $log p$ factor of traffic.

*Example NCCL usage:*
```cpp
#include <nccl.h>
ncclComm_t comm;
ncclUniqueId id;
if (rank == 0) ncclGetUniqueId(&id);
// Broadcast id over MPI or other OOB channel
MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

cudaSetDevice(local_rank);
ncclCommInitRank(&comm, world_size, id, rank);

float* grad_buf;  // on device
cudaStream_t s;
cudaStreamCreate(&s);
ncclAllReduce(grad_buf, grad_buf, num_params, ncclFloat, ncclSum, comm, s);
cudaStreamSynchronize(s);
```

*Tuning environment variables:*
```bash
export NCCL_DEBUG=INFO               # log collective topology + algorithm choices
export NCCL_ALGO=Ring,Tree           # allowed algorithms
export NCCL_PROTO=Simple,LL,LL128    # low-latency protocols for small messages
export NCCL_IB_HCA=mlx5_0            # which HCA to use
export NCCL_P2P_LEVEL=NVL            # minimum interconnect for P2P
export NCCL_TOPO_DUMP_FILE=topo.xml  # dump detected topology for debugging
export NCCL_SOCKET_IFNAME=eth0       # out-of-band bootstrap interface
```

== Parallelism Strategies for Large Models

#table(
  columns: (auto, auto, auto, auto),
  [*Strategy*], [*What is split*], [*Comm pattern*], [*BW requirement*],
  [Data Parallel (DDP)], [Batch across replicas], [Gradient all-reduce], [Low-mid (once per step)],
  [Tensor Parallel (TP)], [Matmul columns/rows], [All-reduce inside layer], [Very high (every layer)],
  [Pipeline Parallel (PP)], [Consecutive layers], [Activation send/recv], [Low (P2P only)],
  [Sequence Parallel (SP)], [Sequence dimension], [Ring all-gather/reduce-scatter], [High],
  [Expert Parallel (EP)], [MoE experts across GPUs], [All-to-all token routing], [High (every MoE layer)],
  [FSDP / ZeRO-3], [Params + grads + optim], [All-gather + reduce-scatter], [Mid-high],
)

*Data Parallel (DDP):* simplest. Each GPU holds full model replica. After each backward pass, all-reduce gradients across replicas. Memory waste: $p$ copies of parameters + optimizer state.

*Tensor Parallel (Megatron-LM, Shoeybi et al. 2020):* split each matrix multiply across GPUs.
- Column parallel: $Y = X A$ where $A$ is split column-wise: $A = [A_1, A_2, ..., A_p]$. Output $Y_i = X A_i$ is each partial. No communication before; all-gather after if next layer needs full $Y$.
- Row parallel: $Y = X A$ where $A$ is split row-wise: input $X$ split too, each computes partial $Y_i = X_i A_i$; all-reduce at the end.
- Transformer block uses column-then-row to absorb the all-reduce: MLP layer = `Y = GeLU(X W_1) W_2`; $W_1$ column-split, $W_2$ row-split, one all-reduce at output.
- Requires very high bandwidth because it happens every layer ($O(L)$ collectives per step) — only practical within an NVLink domain.

*Pipeline Parallel (GPipe — Huang et al. 2019; PipeDream / 1F1B — Narayanan et al. 2019):*
- Partition the model into $K$ stages, one per GPU
- Feed micro-batches through the pipeline; GPU $i$ forwards stage $i$, sends activations to GPU $i+1$
- GPipe: all forwards first, then all backwards — simple but big bubble
- 1F1B: interleave forward and backward — smaller bubble
- Bubble fraction: $approx (K-1) / (M + K - 1)$ where $M$ = micro-batches. Need $M >> K$ for small bubble.

*Fully Sharded Data Parallel (FSDP) / ZeRO (Rajbhandari et al. 2020):*
- ZeRO-1: optimizer state sharded across DP replicas
- ZeRO-2: gradients also sharded
- ZeRO-3 / FSDP: parameters sharded; all-gathered just-in-time for each layer's forward pass, then freed
- Memory saving: $1/p$ factor per shard
- Comm overhead: all-gather per layer forward + reduce-scatter per layer backward — 1.5x vs DDP at best, 3x at worst

*3D parallelism (combining DP × TP × PP):*
- Example for 1024 H100s training a 70B LLM: TP=8 (within NVLink domain), PP=8, DP=16
- TP within node (NVLink), PP between nodes (IB, only activations), DP across nodes (gradient all-reduce amortized by micro-batches)

== Collective Communication Cost Models

*Hockney model:* time = $alpha + n / beta$
- $alpha$: per-message latency
- $beta$: per-byte bandwidth
- $n$: message size

*Ring all-reduce on $p$ nodes:*
$ T = 2 (p-1) alpha + 2 (p-1) / p dot n / beta $

*Recursive doubling / halving* (Rabenseifner 2004): $O(log p)$ rounds, good for medium sizes
$ T = log_2 p dot alpha + 2 (p-1)/p dot n / beta $

*Double binary tree (NCCL default for 2+ nodes):*
$ T approx 2 log_2 p dot alpha + 2 n / beta $

*Concrete numbers*, 1 GB all-reduce on 8 $times$ H100 with NVLink 4 (900 GB/s per-GPU):
- Effective bisection BW $approx 6.3$ TB/s ($8 times 900 / (p-1)$ ring)
- Lower bound: $2 (p-1)/p dot 1"GB" / 900"GB/s" approx 1.9 "ms"$
- Measured NCCL: $tilde$ 2-3 ms (typical efficiency 70-90%)

== NVSHMEM and One-Sided Communication

*NVSHMEM* implements OpenSHMEM PGAS model for GPU clusters: any thread in any kernel can put/get to any remote GPU's memory directly.

```cpp
#include <nvshmem.h>
#include <nvshmemx.h>

nvshmem_init();
int my_pe = nvshmem_my_pe();
int npes = nvshmem_n_pes();

// Allocate symmetric (accessible from all PEs) array
float* data = (float*) nvshmem_malloc(N * sizeof(float));

__global__ void ring_shift(float* data, int npes, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        int next_pe = (nvshmem_my_pe() + 1) % npes;
        // One-sided put: write my data[tid] to next PE's data[tid]
        nvshmem_float_p(&data[tid], data[tid] * 2.0f, next_pe);
    }
}
// Launch; no host-side collective needed
ring_shift<<<grid, block>>>(data, npes, N);
nvshmemx_barrier_all_on_stream(stream);
```

*Key benefit over NCCL:* enables fine-grained, algorithm-defined communication _from inside the kernel_, without host synchronization boundaries. Used in Hopper+ by advanced multi-GPU GEMM kernels, Flash Attention 3 multi-GPU variants, and CUTLASS.

== Performance Tuning Cheat Sheet

#table(
  columns: (auto, auto, auto),
  [*Symptom*], [*Likely cause*], [*Action*],
  [Low NVLink utilization], [NCCL routing over PCIe], [Set `NCCL_P2P_LEVEL=NVL`; check `nvidia-smi topo -m`],
  [Collectives slower than expected], [Wrong algo selected], [`NCCL_ALGO=Ring` or `Tree`; inspect `NCCL_DEBUG=INFO` output],
  [Long comm on small messages], [Latency-bound], [Enable `NCCL_PROTO=LL,LL128`; fuse small comms],
  [CPU pinned during collective], [NCCL using sockets], [Verify IB HCA detected; set `NCCL_IB_HCA`],
  [High bubble time in PP], [Few micro-batches], [Increase $M$; use 1F1B scheduler],
  [OOM in FSDP], [Activation memory too high], [Enable activation checkpointing; increase TP instead],
  [Training hangs], [Mismatched collective], [Every rank must call matching NCCL op in same order; enable `NCCL_BLOCKING_WAIT=1` for debugging],
)

== References

NVIDIA Corporation (2024). "NCCL: NVIDIA Collective Communications Library." https://docs.nvidia.com/deeplearning/nccl/

Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC'20.

Shoeybi, M. et al. (2020). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." arXiv:1909.08053.

Narayanan, D. et al. (2021). "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM." SC'21.

Huang, Y. et al. (2019). "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism." NeurIPS.

Patarasuk, P. & Yuan, X. (2009). "Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations." JPDC 69(2).

Sanders, P., Speck, J., & Träff, J.L. (2009). "Two-tree algorithms for full bandwidth broadcast, reduction and scan." Parallel Computing 35(12).

Rabenseifner, R. (2004). "Optimization of Collective Reduction Operations." ICCS 2004.

NVIDIA Corporation (2024). "DGX H100 and GB200 NVL72 Reference Architecture." Technical whitepapers.


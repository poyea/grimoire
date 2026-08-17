#import "../template.typ": xref

= NCCL and Collective Communication

Training LLMs on hundreds or thousands of GPUs is fundamentally a communication problem: every step exchanges activations, gradients, and parameters across ranks. NVIDIA's *NCCL* (NVIDIA Collective Communications Library) provides MPI-style collectives optimized for GPU memory and NVLink/PCIe/InfiniBand fabrics, and it is the substrate beneath PyTorch DDP/FSDP, JAX `pmap`, and Megatron-LM.

*See also:* #xref("gpu-architecture", "multi-gpu", label: "Multi-GPU") (NVLink, NVSwitch, topologies), #xref("gpu-architecture", "ml-workloads", label: "ML Workload Optimization on GPUs") (data and tensor parallelism).

== The Collective Primitives

#table(
  columns: 3,
  [*Primitive*], [*Pattern*], [*Use*],
  [Broadcast],     [root $->$ all],                     [parameter initialization],
  [Reduce],        [all $->$ root (op)],                [gradient sum to coordinator],
  [AllReduce],     [all $->$ all (op)],                 [data-parallel gradient sync],
  [ReduceScatter], [all $->$ all (op, partitioned)],    [FSDP gradient reduction],
  [AllGather],     [all $->$ all (concat)],             [FSDP parameter gather, tensor-parallel],
  [Scatter],       [root $->$ all (partitioned)],       [data distribution],
  [Gather],        [all $->$ root (concat)],            [result collection],
  [All-to-All],    [all $<->$ all (permuted)],          [MoE expert dispatch, sequence parallel],
  [SendRecv],      [point-to-point],                    [pipeline parallel],
)

Key identity used everywhere in FSDP/ZeRO:
$ "AllReduce" = "ReduceScatter" + "AllGather" $

Both halves cost $(N-1)/N$ of the message volume, so the split costs the same as the combined operation but lets the framework overlap each half with different compute phases.

== API in 20 Lines

```cpp
#include <nccl.h>
ncclComm_t comm;
ncclUniqueId id;
if (rank == 0) ncclGetUniqueId(&id);
MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
ncclCommInitRank(&comm, world_size, id, rank);

float* d_buf; cudaMalloc(&d_buf, N*sizeof(float));
cudaStream_t s; cudaStreamCreate(&s);

ncclAllReduce(d_buf, d_buf, N, ncclFloat, ncclSum, comm, s);
cudaStreamSynchronize(s);

ncclCommDestroy(comm);
```

NCCL is *stream-ordered*: enqueued like a kernel, completes asynchronously. Multiple collectives on the same comm in the same stream serialize; on different streams (with `ncclGroupStart/End`) they may run concurrently.

== Bandwidth Bounds

For message size $S$ across $N$ ranks on a bidirectional link of bandwidth $B$:

#table(
  columns: 3,
  [*Operation*], [*Bytes per link*], [*Latency-optimal time*],
  [Broadcast],     [$S$],                       [$alpha log N + S/B$ (tree)],
  [AllReduce ring], [$2 S (N-1)/N$],            [$2(N-1)alpha + 2 S (N-1)/(N B)$],
  [AllReduce tree], [$2 S log N$],              [$2 alpha log N + 2 S log N / B$],
  [ReduceScatter], [$S(N-1)/N$],               [half of AllReduce],
  [AllGather],     [$S(N-1)/N$],               [half of AllReduce],
)

For large $S$ ring is bandwidth-optimal (constant bytes/link as $N$ grows); for small $S$ tree wins on the $log N$ latency term.

== Ring Algorithm

The classic NCCL AllReduce: arrange $N$ ranks in a ring, split the buffer into $N$ chunks, do $N-1$ ReduceScatter steps followed by $N-1$ AllGather steps.

```
Rank 0  Rank 1  Rank 2  Rank 3
 [A0]    [A1]    [A2]    [A3]
   \      \      \      /     each step:
   send chunk[k-i] to next,
   add chunk[k-i-1] from prev
```

After $N-1$ ReduceScatter steps each rank holds the *reduced* version of one chunk; after $N-1$ AllGather steps every rank has every reduced chunk. Total: $2(N-1)$ steps, each moving $S/N$ bytes — bandwidth-optimal.

== Tree and Double-Binary Tree

Ring latency grows as $O(N)$ — painful at $N = 1024$+. NCCL ($>=$ 2.4) added a *double binary tree* (Sanders et al. 2009): every rank is an interior node in one tree and a leaf in the other, so all links are saturated in both directions.

```
Tree A:                     Tree B (mirror):
        0                            7
       / \                          / \
      1   2                        6   5
     / \ / \                      / \ / \
    3  4 5  6                    4  3 2  1
            \                            \
             7                            0
```

Latency drops to $2 alpha log N$; for small messages this is 5–10$times$ faster than ring at $N >= 64$. NCCL picks ring vs tree per-call based on tuned thresholds.

== Hierarchical Collectives

A real cluster has two levels: NVLink/NVSwitch *intra-node* (e.g. 8 GPUs at 900 GB/s bidirectional) and InfiniBand/RoCE *inter-node* (e.g. 400 Gbps = 50 GB/s).

A hierarchical AllReduce:
+ Intra-node ReduceScatter over NVLink (cheap).
+ Inter-node AllReduce over IB on per-rank shards (small messages).
+ Intra-node AllGather over NVLink.

Effective bandwidth utilization on a 256-GPU cluster: ~80–90% of the IB ceiling, compared to ~30% for a flat ring crossing IB at every step.

NCCL discovers topology automatically via NVML and `nccl-topo.xml`; you rarely write the hierarchy by hand.

== Transports and Backends

#table(
  columns: 3,
  [*Transport*], [*Where*], [*Notes*],
  [P2P (CUDA IPC)], [same node, NVLink/PCIe], [direct cudaMemcpy peer],
  [SHM],            [same node, no NVLink],   [host memory shared region],
  [NET (sockets)],  [inter-node, TCP],        [fallback, slow],
  [NET (IB verbs)], [inter-node, InfiniBand], [RDMA, primary HPC path],
  [NET (GDR)],      [IB + GPUDirect RDMA],    [GPU memory $<->$ NIC, no host copy],
)

*GPUDirect RDMA* (GDR) lets the NIC DMA directly to and from GPU HBM, removing a host bounce buffer. Requires kernel module (`nvidia-peermem`), an IB NIC on the same PCIe root complex as the GPU (or NVLink-attached, on Grace-Hopper), and a CPU IOMMU configured to allow it.

NCCL uses *proxy threads* on the CPU to manage NIC queue pairs and ring buffers; the GPU side runs a *NCCL kernel* that copies between user buffers and the proxy-managed staging area using device-side SM threads.

== Performance Knobs

```bash
export NCCL_ALGO=Tree              # or Ring, CollNet, NVLS
export NCCL_PROTO=Simple           # or LL, LL128 (low-latency)
export NCCL_NCHANNELS=16           # SM parallelism per collective
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_IB_GID_INDEX=3         # RoCE v2
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=COLL,INIT,NET
```

*Protocols:* `LL` (low-latency) sends 4-byte flags inline with data, avoiding a CTS handshake — best for tiny messages. `LL128` is the same with 128-byte lines. `Simple` is the bulk-friendly default.

*Channels:* NCCL splits each collective across multiple parallel rings/trees. More channels $arrow.r$ more SMs busy, more NIC QPs in flight, higher achievable bandwidth — at the cost of SM occupancy stolen from compute.

== NVLS and SHARP

*NVLS* (NVLink SHARP, Hopper + NVSwitch3) does in-network reduction inside the NVSwitch ASIC. The switch sums values from multiple GPUs in hardware, halving NVLink traffic for AllReduce.

*SHARP* (InfiniBand) does the analog inside Quantum-2 switches at the IB level.

Both deliver 30–50% AllReduce speedup at scale, transparently when `NCCL_ALGO=NVLS` or `CollNet` is selected.

== Benchmarking — nccl-tests

```bash
mpirun -n 8 -H host1:4,host2:4 ./build/all_reduce_perf \
    -b 8M -e 8G -f 2 -g 1
```

Reported metrics: `algbw` (algorithm bandwidth, $S$/time) and `busbw` (bus bandwidth, $"algbw" dot 2(N-1)/N$). `busbw` is the apples-to-apples number, comparable to the link bandwidth ceiling.

A healthy 8$times$H100 NVLink box hits ~370 GB/s `busbw` on 1 GB AllReduce — close to the 450 GB/s unidirectional NVLink4 ceiling.

== Failure Modes and Debugging

Common pitfalls:
- *Hangs:* mismatched op/dtype/count between ranks; NCCL silently waits. Set `NCCL_TIMEOUT` and turn on `NCCL_DEBUG=INFO`.
- *Slow startup:* topology probe takes seconds; cache via `NCCL_TOPO_FILE`.
- *Inter-node much slower than expected:* check GDR (`NCCL_NET_GDR_LEVEL`), PCIe topology (`nvidia-smi topo -m`), and NUMA pinning.
- *PXN / rail-aware:* Hopper systems use PXN (PCIe cross-NIC) to use the NIC closest to the source GPU regardless of which GPU on the destination side is the receiver. Disable with `NCCL_PXN_DISABLE=1` to compare.

== Further Reading

NVIDIA (2024). _NCCL Developer Guide_. https://docs.nvidia.com/deeplearning/nccl/

Jeaugey, S. (2019). "Massively Scale Your Deep Learning Training with NCCL 2.4." NVIDIA Developer Blog (double binary tree).

Sanders, P. et al. (2009). "Two-Tree Algorithms for Full Bandwidth Broadcast, Reduction and Scan." _Parallel Computing_.

Patarasuk, P. & Yuan, X. (2009). "Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations." _JPDC_.

Mellanox (2021). _SHARP: Scalable Hierarchical Aggregation and Reduction Protocol_ whitepaper.

NVIDIA (2023). "Doubling All2All Performance with NVIDIA Collective Communication Library 2.12." NVIDIA Developer Blog.

Awan, A.A. et al. (2018). "OC-DNN: Exploiting Advanced Unified Memory Capabilities in CUDA 9 and Volta GPUs for Out-of-Core DNN Training." _HiPC_.

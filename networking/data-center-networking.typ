#import "../template.typ": xref

= Data Center Networking

Data-centre fabrics are engineered for very different goals than the public internet: tight latency variance, full bisection bandwidth, and RDMA-grade losslessness. This chapter covers the dominant topologies (Clos / fat-tree), ECMP flow hashing, RDMA over Converged Ethernet (RoCEv2) with PFC/ECN, InfiniBand, and modern NIC offloads.

*See also:* #xref("networking", "congestion-control", label: "Congestion Control") (DCTCP, L4S), #xref("networking", "kernel-bypass", label: "Kernel Bypass") (RDMA verbs), #xref("networking", "bgp-routing", label: "BGP Routing") (BGP-in-the-data-centre), #xref("networking", "programmable-data-planes", label: "Programmable Data Planes") (P4 / Tofino).

== Clos and Fat-Tree Topologies

Hierarchical "core / aggregation / access" trees with oversubscription are obsolete for high-throughput workloads; a single core-aggregation link becomes the universal bottleneck. Modern fabrics use *Clos networks* (after Charles Clos, 1953), most commonly the 3-stage *leaf-spine* and 5-stage *fat-tree* variants.

```
Leaf-Spine (3-stage Clos):

   Spine 1   Spine 2   Spine 3   Spine 4
     │ ╲ ╲ ╲   │ ╲ ╲ ╲   │ ╲ ╲ ╲   │
     │  ╲ ╲ ╲  │  ╲ ╲ ╲  │  ╲ ╲ ╲  │      every leaf connects
     │   ╲ ╲ ╲ │   ╲ ╲ ╲ │   ╲ ╲ ╲ │      to every spine
  ── L1 ── L2 ── L3 ── L4 ── L5 ── L6 ──  (full bisection)
     │     │     │     │     │     │
    ToR  hosts ...
```

*Properties.*
- $n$ leaves, each with $k$ spine uplinks, total bisection bandwidth $= n times k times "link speed"$.
- Any host-to-host pair has exactly two hops (leaf $→$ spine $→$ leaf).
- Loss of a single spine reduces bisection by $1/k$; no flow loses connectivity.
- All links typically of equal speed (avoid asymmetry that confuses ECMP).

The original *fat-tree* (Al-Fares et al., SIGCOMM 2008) generalises to a $k$-ary structure with $k^3 / 4$ servers and full bisection bandwidth using $k$-port commodity switches throughout. Microsoft's VL2 (2009), Google's Jupiter (2015), and Meta's F16 / Disaggregated Scheduled Fabric all derive from the fat-tree.

*Routing.* The fabric is a Layer-3 network with BGP unnumbered between every leaf and spine (RFC 7938, "Use of BGP for Routing in Large-Scale Data Centers"). OSPF is also used, but BGP has won at hyperscale because of its policy expressiveness and operational maturity.

== ECMP and Flow Hashing

With multiple equal-cost paths between any two hosts, the switch must pick one for every packet. *Equal-Cost Multi-Path* (ECMP) distributes flows by hashing a subset of the 5-tuple:
$ h = "hash"("src ip", "dst ip", "src port", "dst port", "proto") mod N_"paths" $

*Per-flow* hashing is the default: all packets of one TCP/QUIC connection take the same path, avoiding reordering. *Per-packet* spraying is rarely used because it shreds in-order delivery.

*Polarisation* is a classic failure mode: identical hash functions and inputs at every tier route many flows along the same physical path, leaving others idle. Modern switches mitigate by adding a per-switch *salt* (a chassis-unique value) to the hash input. Inspecting hash distribution:

```bash
# Linux per-route ECMP via FIB
ip route add 198.51.100.0/24 \
  nexthop via 10.0.1.1 weight 1 \
  nexthop via 10.0.2.1 weight 1 \
  nexthop via 10.0.3.1 weight 1

# Tune the kernel hash policy
sysctl -w net.ipv4.fib_multipath_hash_policy=1   # L4 ports included

# Cumulus / SONiC: per-switch hash seed (vendor-specific)
nv set system forwarding ecmp-hash hash-fields '{srcip,dstip,sport,dport,proto}'
```

*Elephant flows.* Long-lived high-rate flows on the same hashed bucket cause congestion while other links idle. Mitigations: *flowlet switching* (re-hash after a brief gap, Conga / LetFlow), packet-level multipath (NDP), or scheduling fabrics (Meta DSF).

== RDMA over Converged Ethernet — RoCEv2

RDMA (Remote Direct Memory Access) lets a NIC read or write peer memory without involving the remote CPU. Originally an InfiniBand technology, RoCEv2 carries the same verbs over UDP/IP/Ethernet (UDP port 4791). Typical use cases: distributed training (NCCL, Horovod), distributed storage (NVMe-oF, Ceph), in-memory databases.

```
Application
   │   ibv_post_send( WR )  → completion queue
Verbs API (libibverbs)
   │
NIC firmware (Mellanox/NVIDIA ConnectX, Broadcom Thor, Intel IPU)
   │
Wire format:  Ethernet | IP | UDP(4791) | IB-BTH | payload | ICRC | FCS
```

*Latency:* end-to-end one-way ~1-3μs on modern 100/200/400G ConnectX-7 NICs; throughput pinned to line rate (400 Gb/s = 50 GB/s).

*Lossless requirement.* The original IB transport assumes a lossless fabric. RoCEv2 over Ethernet must therefore reproduce losslessness, either by careful congestion management (DCQCN) or hop-by-hop pause (PFC).

== PFC — Priority Flow Control (IEEE 802.1Qbb)

PFC is per-class link-layer pause: a downstream switch sends a *Pause* frame referencing a specific 802.1p priority class; the upstream port stops sending that class for the indicated time.

```
ToR-A ──── 8 priority queues ──── ToR-B
              │
              ▼
          if queue near full, send
          PAUSE(class=3, time=N)
              ▲
              │
           ToR-A stops sending
           class-3 frames for N quanta
```

*Deployed configuration.* RDMA traffic typically uses priority 3; storage traffic priority 4; everything else priority 0. Only RDMA priorities are made lossless via PFC; bulk TCP remains drop-tolerant.

*Hazards.*
- *PFC storms / deadlocks.* A loop of pausing ports can freeze a region of the fabric. Modern fabrics detect this with watchdogs and selectively drop after a timeout.
- *Head-of-line blocking across flows in the same class.* A slow receiver pauses the entire class on its ingress.
- *Slow drain.* A single stuck NIC can cause sustained back-pressure for seconds.

Microsoft Azure's RoCEv2 deployment (Guo et al., SIGCOMM 2016) details years of PFC-related outages and the engineering of DCQCN to *avoid* relying on PFC for normal-case congestion.

== ECN and DCQCN

DCQCN (Data Center Quantized Congestion Notification, Microsoft / Mellanox, 2015) is the de-facto rate-based congestion controller for RoCEv2:

+ Switches mark packets ECN-CE when egress queue exceeds threshold $K_min$ (typically a few hundred KB).
+ Receiver sends *Congestion Notification Packet* (CNP) back to sender when it sees marks.
+ Sender NIC reduces send rate multiplicatively (factor $α$ derived from CNP rate, similar to DCTCP).
+ Recovery: rate increases additively, then exponentially after a target duration without CNP.

When DCQCN is correctly tuned, PFC fires only on transient micro-bursts, staying closer to the lossless ideal without the deadlock risk of relying on PFC.

```bash
# Mellanox ConnectX: inspect RoCE counters
ethtool -S enp1s0f0 | grep -E 'roce|cnp|pause'

# Per-priority pause statistics
mlnx_qos -i enp1s0f0
# Priority trust state: dscp
# default priority: 0
# pfc enabled: 0 0 0 1 0 0 0 0   ← class 3 PFC-enabled
```

== InfiniBand

InfiniBand is a purpose-built lossless fabric used in HPC and AI clusters. Speeds: SDR (8 Gb/s) → EDR (100) → HDR (200) → NDR (400) → XDR (800). Topologies are typically fat-tree.

Key differences from RoCEv2:
- Lossless by *credit-based* link-layer flow control; no PFC pause storms.
- Custom physical layer; subnet manager (`opensm`) handles addressing and routing centrally.
- Hardware-supported verbs include *atomic operations* (fetch-and-add, compare-and-swap) on remote memory.
- Dominant in Top500 supercomputers and Nvidia's DGX / SuperPOD AI clusters.

== NIC Offloads

Modern NICs do far more than DMA. Hardware accelerators relevant to data-centre throughput:

#table(
  columns: (auto, auto),
  [*Offload*], [*What it does*],
  [Checksum (TX/RX)], [NIC computes IP/TCP/UDP checksum on the wire; saves cycles],
  [TSO / GSO], [Kernel passes a 64 KB "super-segment"; NIC slices into MSS-sized packets],
  [LRO / GRO], [Reverse: NIC coalesces successive incoming segments before delivery],
  [VLAN insertion / strip], [Hardware adds or removes 802.1Q tag],
  [RSS], [Receive Side Scaling: hash 5-tuple, distribute interrupts across CPU cores],
  [SR-IOV], [NIC presents multiple PCIe virtual functions to VMs; bypasses host kernel],
  [VXLAN / Geneve], [Tunnel encap and decap in hardware (overlay networks)],
  [TLS (kTLS offload)], [Hardware AEAD for TLS data records (Mellanox, Intel)],
  [DPU / SmartNIC], [Arm cores + P4 + crypto engine for host offload of entire network stack (Nvidia BlueField, AMD Pensando)],
)

```bash
# Inspect offloads on a NIC
ethtool -k eth0
# tcp-segmentation-offload: on
# generic-segmentation-offload: on
# rx-checksumming: on
# scatter-gather: on

# Toggle (disable LRO for forwarding scenarios: it merges flows)
ethtool -K eth0 lro off
```

== Topology and Performance Comparison

#table(
  columns: (auto, auto, auto, auto),
  [*Topology*], [*Bisection bandwidth*], [*Hops*], [*Used by*],
  [Classic 3-tier], [Oversubscribed (often 4:1 or worse)], [3-4], [Legacy enterprise],
  [Leaf-spine (3-stage Clos)], [Full (1:1)], [3], [Most modern DCs],
  [5-stage fat-tree], [Full], [5], [Hyperscale (Jupiter, F16)],
  [Dragonfly+], [Full, longer-range links], [3-4], [HPC (Frontier, Aurora)],
  [Torus / mesh], [Limited], [$O(sqrt(N))$], [HPC (Cray, K computer)],
)

#table(
  columns: (auto, auto, auto, auto),
  [*Fabric tech*], [*Latency (one-way)*], [*Throughput per port*], [*Loss model*],
  [10/25 GbE + TCP], [20-100μs], [10-25 Gb/s], [Lossy, congestion-controlled],
  [RoCEv2 + DCQCN], [2-5μs], [100-400 Gb/s], [Near-lossless with PFC],
  [InfiniBand HDR/NDR], [1-2μs], [200-400 Gb/s], [Lossless (credit FC)],
  [Optical circuit (TPU pods)], [$<$1μs], [1-3.2 Tb/s], [Lossless, fixed schedule],
)

== Pitfalls

- *Asymmetric ECMP.* Mixing 100 G and 400 G uplinks on the same leaf without *weighted* ECMP causes the higher-bandwidth path to congest first. Use weighted ECMP or keep speeds uniform per tier.
- *Buffer hogging.* A single elephant flow with deep switch buffers ruins fan-in latency for $>$100 small RPCs. Active queue management (ECN at $K_min approx 100$KB) and small switch buffers (Trident / Tofino with $<$ 100MB total) outperform big buffers.
- *MTU mismatches break PMTU.* RDMA assumes a uniform fabric MTU (usually 4200 B or 9000 B). A single non-jumbo link silently drops 9 KB frames.
- *PFC + bridging loops = deadlock.* Avoid L2 bridges within the RDMA path; keep the entire fabric routed.

== Further Reading

Al-Fares, M., Loukissas, A. & Vahdat, A. (2008). "A Scalable, Commodity Data Center Network Architecture." SIGCOMM (fat-tree).

Greenberg, A. et al. (2009). "VL2: A Scalable and Flexible Data Center Network." SIGCOMM.

Singh, A. et al. (2015). "Jupiter Rising: A Decade of Clos Topologies and Centralized Control in Google's Datacenter Network." SIGCOMM.

RFC 7938: Use of BGP for Routing in Large-Scale Data Centers.

Guo, C. et al. (2016). "RDMA over Commodity Ethernet at Scale." SIGCOMM (Microsoft RoCEv2 lessons).

Zhu, Y. et al. (2015). "Congestion Control for Large-Scale RDMA Deployments (DCQCN)." SIGCOMM.

Alizadeh, M. et al. (2014). "CONGA: Distributed Congestion-Aware Load Balancing for Datacenters." SIGCOMM.

Handley, M. et al. (2017). "Re-architecting datacenter networks and stacks for low latency and high performance (NDP)." SIGCOMM.

IEEE 802.1Qbb: Priority-based Flow Control.

InfiniBand Trade Association: "InfiniBand Architecture Specification Vol. 1."

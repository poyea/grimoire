#import "../template.typ": xref

= Interconnects: UPI, Infinity Fabric, CXL, NVLink, PCIe <interconnects>

Once compute scales beyond a single die, the bottleneck moves to the wires between dies, sockets, and accelerators. Modern interconnects span four tiers: on-package (chiplet links), socket-to-socket ($"UPI"$, $"Infinity"$ $"Fabric"$), host-to-device ($"PCIe"$, $"CXL"$), and accelerator-to-accelerator ($"NVLink"$, $"UALink"$, optical).

*See also:* #xref("cpu-architecture", "multicore", label: "Multicore"), #xref("cpu-architecture", "memory-system", label: "Memory System"), #xref("cpu-architecture", "accelerators", label: "Accelerators"), #xref("cpu-architecture", "cache-hierarchy", label: "Cache Hierarchy")

== Why Interconnects Matter

A core's reach is bounded by what it can address with reasonable latency. Adding cores adds wires; adding sockets adds far slower wires; adding accelerators adds slower-still wires. Every level of the hierarchy multiplies latency by roughly 10$times$:

#table(
  columns: 3,
  [*Tier*], [*Typical latency*], [*Typical bandwidth (per link)*],
  [On-die ring/mesh], [10-30 ns], [1-4 TB/s aggregate],
  [Chiplet ($"UCIe"$, $"EMIB"$, IFOP)], [5-10 ns extra], [1-2 TB/s per link],
  [Socket-to-socket ($"UPI"$/$"xGMI"$)], [80-150 ns], [40-200 GB/s per link],
  [Host-device ($"PCIe"$ Gen5 x16)], [400-800 ns], [64 GB/s bidir],
  [$"CXL"$ memory], [150-300 ns add'l], [Same as $"PCIe"$ phy],
  [$"NVLink"$ 4/5], [\~250 ns], [900 GB/s ($"NVLink"$ 5: 1.8 TB/s)],
  [Network ($"RoCE"$/$"IB"$ HDR/NDR)], [1-3 $mu$s], [50-400 Gb/s],
)

== Intel UPI (QPI's Successor)

Quick Path Interconnect ($"QPI"$, 2008) and its successor Ultra Path Interconnect ($"UPI"$, 2017+) provide cache-coherent socket-to-socket links for multi-socket Xeons.

#table(
  columns: 4,
  [*Gen*], [*Year*], [*Rate (GT/s)*], [*Per-link BW*],
  [$"QPI"$ 1.0], [2008], [6.4], [25.6 GB/s],
  [$"QPI"$ 1.1], [2011], [8.0], [32 GB/s],
  [$"UPI"$ 1.0], [2017], [10.4], [41.6 GB/s],
  [$"UPI"$ (ICX)], [2021], [11.2], [44.8 GB/s],
  [$"UPI"$ 2.0 (SPR)], [2023], [16.0], [64 GB/s],
)

$"UPI"$ carries the $"MESIF"$ coherence protocol (Forward state added to $"MESI"$): a remote socket asks a peer for a line and forwards directly without a memory round-trip. Two- to eight-socket topologies; beyond eight, glueless coherence becomes impractical.

== AMD Infinity Fabric

Infinity Fabric ($"IF"$) is a layered transport: $"SDF"$ (Scalable Data Fabric) for data, $"SCF"$ (Scalable Control Fabric) for messages. $"IF"$ runs both on-package (between Zen $"CCD"$s and the $"IOD"$) as $"IFOP"$ and off-package (between sockets) as $"IFIS"$ / $"xGMI"$.

On EPYC, a single $"IF"$ clock domain ($"FCLK"$) ties together $"CCD"$ caches, the memory controller, and remote sockets. On Ryzen desktop the same $"FCLK"$ couples to $"MEMCLK"$ — overclocking memory pulls $"IF"$ with it, and a mismatch halves bandwidth.

Inter-socket EPYC links use $"xGMI"$ (Infinity Fabric over $"SerDes"$):

#table(
  columns: 3,
  [*Family*], [*xGMI rate*], [*Per-link BW*],
  [EPYC Naples (Zen 1)], [10.6 GT/s], [42 GB/s],
  [EPYC Rome/Milan (Zen 2/3)], [18 GT/s], [72 GB/s],
  [EPYC Genoa (Zen 4)], [32 GT/s], [128 GB/s],
  [EPYC Turin (Zen 5)], [36-40 GT/s], [up to 160 GB/s],
)

The chiplet architecture means every memory access crosses a $"CCD"$ $arrow$ $"IOD"$ hop, which is why Zen has higher idle DRAM latency than monolithic Intel parts (~90 ns vs ~70 ns).

== PCI Express

$"PCIe"$ is the universal host-device link and the physical layer beneath $"CXL"$.

#table(
  columns: 4,
  [*Gen*], [*Year*], [*Rate (GT/s)*], [*x16 BW (bidir)*],
  [3.0], [2010], [8 ($"8b/10b"$ phy switches to $"128b/130b"$)], [32 GB/s],
  [4.0], [2017], [16], [64 GB/s],
  [5.0], [2019], [32], [128 GB/s],
  [6.0], [2022], [64 ($"PAM4"$ + $"FLIT"$)], [256 GB/s],
  [7.0 (spec 2025)], [\~2027], [128], [512 GB/s],
)

Gen6 is a paradigm shift: $"PAM4"$ signaling (4 levels per symbol) doubles rate without doubling frequency, but requires forward error correction ($"FEC"$); the link reorganizes into $"FLIT"$s (256B units) shared with $"CXL"$.

*Latency budget* (Gen5 x16): ~100 ns transport + ~300 ns root complex + device $arrow$ 400-800 ns round-trip for a 64B read. This is why $"PCIe"$ DMA is throughput-friendly but latency-hostile compared to $"NVLink"$.

== CXL (Compute Express Link)

$"CXL"$ rides on the $"PCIe"$ phy but adds two cache-coherent protocols on top of $"PCIe"$.io:

- *$"CXL"$.cache:* device caches host memory.
- *$"CXL"$.mem:* host caches device memory.

Three device classes:

#table(
  columns: 3,
  [*Type*], [*Protocols*], [*Example*],
  [Type 1], [.io + .cache], [SmartNIC, accelerator with own cache, no attached DRAM],
  [Type 2], [.io + .cache + .mem], [GPU/accelerator with $"HBM"$ exposed to host],
  [Type 3], [.io + .mem], [Memory expander: DRAM, persistent memory],
)

=== Version Timeline

#table(
  columns: 3,
  [*Spec*], [*Year*], [*Adds*],
  [$"CXL"$ 1.1], [2019], [Single host, direct-attached devices, .io/.cache/.mem],
  [$"CXL"$ 2.0], [2020], [Switching, memory pooling across hosts, persistence, hot-plug],
  [$"CXL"$ 3.0], [2022], [$"PCIe"$ 6.0 / 64 GT/s, fabric (multi-level switches), peer-to-peer, memory sharing with coherence],
  [$"CXL"$ 3.1/3.2], [2023/24], [Trusted execution ($"TSP"$), fabric improvements],
)

*Memory pooling* is the killer feature: a rack-scale pool of DDR/$"DRAM"$ behind a $"CXL"$ switch can be dynamically allocated to whichever host needs it, raising utilization from the typical 40-60% to 80%+. Latency is the catch: pooled $"CXL"$ memory is ~250-400 ns vs ~80 ns for local DDR — a new "tier 1.5" between DRAM and SSD that the OS must learn to schedule against.

```bash
# Linux user-space view of CXL memory
# CXL devices appear as NUMA nodes; bind with numactl or set_mempolicy.
$ numactl --hardware
# node 0: local DDR (80 ns)
# node 1: local DDR (80 ns)
# node 2: CXL pool   (300 ns)  <- "cpuless" node
$ numactl --membind=2 ./mostly-cold-app
```

Kernel work-in-progress (6.x): tiered memory management ($"DAMON"$, demotion to $"CXL"$, promotion to DDR) modeled on the old Optane/$"NVDIMM"$ work.

== NVLink and NVSwitch

NVIDIA's accelerator interconnect, the antithesis of $"PCIe"$'s latency.

#table(
  columns: 4,
  [*Gen*], [*GPU*], [*Per-link*], [*Per-GPU aggregate*],
  [NVLink 1], [P100 (2016)], [40 GB/s (4 links)], [160 GB/s],
  [NVLink 2], [V100 (2017)], [50 GB/s (6 links)], [300 GB/s],
  [NVLink 3], [A100 (2020)], [50 GB/s (12 links)], [600 GB/s],
  [NVLink 4], [H100 (2022)], [50 GB/s (18 links)], [900 GB/s],
  [NVLink 5], [B200 (2024)], [100 GB/s (18 links)], [1.8 TB/s],
)

$"NVSwitch"$ aggregates $"NVLink"$s into a non-blocking fabric: in $"DGX"$/$"HGX"$ H100, 8 GPUs and 4 $"NVSwitch"$es deliver full bisection at 900 GB/s per GPU. $"GB200"$ $"NVL72"$ scales this to 72 GPUs in one coherent domain via copper backplane.

The competing open standard, $"UALink"$ ($"Ultra"$ $"Accelerator"$ $"Link"$), launched 2024 by AMD/Broadcom/Cisco/Google/HPE/Intel/Meta/Microsoft to break NVIDIA's lock-in; first products expected 2026.

== AMD Infinity Fabric for GPUs

The $"MI"$ accelerator line uses $"Infinity"$ $"Fabric"$ links between GPUs ("xGMI" again): MI300X has 7 IF links per GPU at 64 GB/s = 448 GB/s per GPU. Eight-GPU node achieves all-to-all coherence comparable to $"NVLink"$.

$"MI300A"$ is the first widely shipped APU with $"CPU"$ and $"GPU"$ chiplets sharing $"HBM"$ over $"Infinity"$ $"Fabric"$, eliminating host-device copies (used in $"El"$ $"Capitan"$ supercomputer).

== Choosing a Topology

#table(
  columns: 3,
  [*Need*], [*Pick*], [*Why*],
  [Two CPU sockets, coherent], [$"UPI"$ or $"xGMI"$], [Native],
  [Host memory expansion], [$"CXL"$ Type 3], [Plug-in DDR pool],
  [GPU memory pooled across hosts], [$"CXL"$ 3.0], [Cross-host coherence],
  [Multi-GPU training], [$"NVLink"$ / $"UALink"$ / $"xGMI"$], [10$times$ $"PCIe"$ bandwidth],
  [Storage / SmartNIC], [$"PCIe"$ Gen5/6], [Ubiquitous],
  [Cross-rack], [$"InfiniBand"$ NDR/$"RoCE"$ 400G], [Network scale],
)

== Latency Engineering Tips

- *NUMA-bind* hot threads to memory ($"numactl"$, $"mbind"$).
- *Use posted writes* where possible; reads cost full round trips.
- *Batch DMA*: amortize the 400 ns $"PCIe"$ floor across large transfers.
- *Prefetch* across $"NUMA"$ borders — hardware prefetchers do not cross sockets.
- *Avoid false sharing* on remote lines (every write triggers a coherence message).

== Further Reading

CXL Consortium (2023). _Compute Express Link Specification 3.1_.

PCI-SIG (2022). _PCI Express Base Specification 6.0_.

Intel Corporation (2023). _Intel Xeon Scalable Processor Family Datasheet_ ($"UPI"$, $"DSA"$).

AMD (2023). _AMD EPYC 9004 Series Processor Architecture_ ($"Infinity"$ $"Fabric"$, $"xGMI"$).

Choquette, J. (2023). "NVIDIA Hopper $"H100"$ GPU: Scaling Performance." _IEEE Micro_ 43(3).

Li, A. et al. (2020). "Evaluating Modern $"GPU"$ Interconnect: $"PCIe"$, $"NVLink"$, $"NV-SLI"$, $"NVSwitch"$ and $"GPUDirect"$." _IEEE TPDS_ 31(1).

Gouk, D. et al. (2022). "Direct Access, High-Performance Memory Disaggregation with $"DirectCXL"$." _USENIX ATC '22_.

UALink Consortium (2025). _Ultra Accelerator Link 1.0 Specification_.

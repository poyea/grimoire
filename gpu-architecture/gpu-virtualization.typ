#import "../template.typ": xref

= GPU Virtualization and Partitioning

A single H100 or B200 is overkill for most inference workloads, yet you cannot share it the way you share a CPU: GPU state is large, kernel preemption is coarse, and memory bandwidth is contested. NVIDIA ships three different sharing mechanisms — MIG, MPS, and vGPU — each making different tradeoffs between isolation, flexibility, and overhead.

*See also:* #xref("gpu-architecture", "execution-model", label: "Execution Model") (SM scheduling), #xref("gpu-architecture", "multi-gpu", label: "Multi-GPU") (cross-GPU fabrics).

== The Sharing Spectrum

#table(
  columns: 4,
  [*Mechanism*], [*Isolation*], [*Granularity*], [*Overhead*],
  [Time-slicing], [None (cooperative)], [Process], [Context-switch cost],
  [MPS],          [Soft (shared address space)], [Process], [Negligible],
  [MIG],          [Hard (HW-partitioned SMs+L2+HBM)], [Slice], [~0% inside slice],
  [vGPU (vfio)],  [Hypervisor-level], [VM],            [Trap-and-emulate],
)

== Time-Slicing (Default)

Without any special configuration, multiple CUDA contexts on the same GPU are *time-sliced* by the GPU's hardware scheduler. Only one context's kernels execute at a time; others wait. Context switches involve saving/restoring register state for active warps and flushing pipelines — tens to hundreds of microseconds, plus any cold-cache penalty after the switch.

In Kubernetes this is exposed via `nvidia.com/gpu.shared` or the device-plugin's `timeSlicing.replicas` setting. It oversubscribes the GPU but provides *no performance isolation* — a runaway tenant starves everyone else.

== MPS — Multi-Process Service

MPS is a user-space daemon (`nvidia-cuda-mps-control`) that funnels work from multiple client processes into *one* CUDA context, letting their kernels run *concurrently* on the same SMs.

```
without MPS:                       with MPS:
   proc A ─┐                          proc A ─┐
   proc B ─┼─> time-slice                proc B ─┼─> mps-server ─> 1 ctx
   proc C ─┘                          proc C ─┘
```

Benefits:
- True concurrent execution: spare SMs run other tenants' kernels.
- Lower latency for small kernels (no context switch).
- Volta+ adds *partitioned MPS*: per-client SM and memory limits.

```bash
# server
nvidia-cuda-mps-control -d

# per-client SM cap
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25
export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT="0=8G"
python infer.py
```

Limitations:
- Single shared address space: a fault in one client kills the MPS server and *all* clients.
- No memory bandwidth isolation; tenants compete for HBM.
- One GPU = one MPS server.

MPS is the right answer for *trusted* multi-tenancy (multiple replicas of your own inference service); use MIG for hostile multi-tenancy.

== MIG — Multi-Instance GPU

Introduced on A100, MIG physically partitions an SM-and-HBM slice into up to 7 independent *GPU instances* (GIs), each appearing to the OS as its own `nvidia0` device with its own UUID, error counters, NVENC/NVDEC engines, and L2 cache slice.

=== Hardware Building Blocks

An A100 has 7 GPCs (Graphics Processing Clusters) and 8 HBM2e memory controllers. MIG carves them into *slices*:
- *Compute slice*: ~14 SMs + 1 GPC's share of L2 (10 MB).
- *Memory slice*: 1/8 of HBM (~10 GB on a 80 GB A100).

A *GPU Instance* (GI) bundles compute + memory slices in approved ratios; a *Compute Instance* (CI) further subdivides a GI into independent CUDA contexts (rare).

=== A100 Profiles

#table(
  columns: 5,
  [*Profile*], [*Memory*], [*SMs*], [*L2*], [*Max per GPU*],
  [1g.10gb], [10 GB],  [14],  [10 MB], [7],
  [2g.20gb], [20 GB],  [28],  [20 MB], [3],
  [3g.40gb], [40 GB],  [42],  [30 MB], [2],
  [4g.40gb], [40 GB],  [56],  [40 MB], [1],
  [7g.80gb], [80 GB], [98],  [80 MB], [1 (whole GPU)],
)

H100 80 GB / 94 GB has analogous profiles (1g.10gb up to 7g.94gb), plus larger memory variants on the H200/B200.

=== Provisioning

```bash
# enable MIG mode (drains the GPU first)
nvidia-smi -i 0 -mig 1

# list available GI profiles
nvidia-smi mig -lgip

# create three 2g.20gb instances
nvidia-smi mig -i 0 -cgi 2g.20gb,2g.20gb,2g.20gb -C

# list instances and their UUIDs
nvidia-smi -L
```

The UUID has the form `MIG-GPU-xxxx/<gi>/<ci>` and is what `CUDA_VISIBLE_DEVICES` or container runtimes target.

=== Isolation Guarantees

MIG provides *hardware* isolation:
- Independent SMs $arrow.r$ no SM contention.
- Independent L2 slice $arrow.r$ no cache pollution.
- Independent HBM controller $arrow.r$ no bandwidth contention.
- Independent context engine and TLBs.
- Independent power management and ECC error reporting.

A noisy neighbor inside one MIG slice cannot affect another slice's latency. The tradeoff: instances cannot communicate via NVLink (each is an isolated PCIe endpoint), and you cannot resize a slice without draining all instances and re-provisioning.

== vGPU — Hypervisor-Mediated Sharing

NVIDIA *vGPU* (formerly GRID) is the VMware/KVM/Xen story: the host loads a vGPU manager driver that creates *mediated devices* (`mdev`) exposed to guests. Each guest sees a virtual GPU with a slice of framebuffer and a scheduling share.

Three schedulers:
- *Best-effort:* round-robin time-slicing, no QoS.
- *Equal share:* equal time slice to active vGPUs.
- *Fixed share:* per-vGPU time share regardless of activity.

vGPU is the only option when the consumer is a VM (e.g. cloud rental of fractional H100s, VDI workstations). It composes with MIG: a single MIG instance can be passed through as a vGPU to a VM, getting hardware isolation *and* live migration.

== Comparison Cheatsheet

#table(
  columns: 5,
  [], [*Time-slice*], [*MPS*], [*MIG*], [*vGPU*],
  [Concurrent kernels], [no], [yes], [yes (per slice)], [yes],
  [Performance isolation], [no], [partial (Volta+)], [hard], [scheduler-based],
  [Fault isolation], [partial], [no], [yes], [yes],
  [Memory partitioning], [no], [soft cap], [hard], [hard],
  [Requires reboot/drain], [no], [no], [yes (mode change)], [host driver swap],
  [Cross-slice NVLink], [yes], [yes], [no], [no],
  [Typical use], [dev], [inference replicas], [multi-tenant inference], [VMs / cloud],
)

== Kubernetes Integration

The NVIDIA *GPU Operator* + *device plugin* exposes all three modes:

```yaml
# MIG single-strategy: each MIG slice appears as nvidia.com/gpu
apiVersion: v1
kind: ConfigMap
data:
  config.yaml: |
    version: v1
    sharing:
      mig:
        strategy: single
        mig-profile: 1g.10gb

# MPS time-slicing replication
    sharing:
      timeSlicing:
        replicas: 4
```

Pods then request `nvidia.com/gpu: 1` and the scheduler binds them to a slice or a time-slice replica.

== Choosing the Right Mechanism

A decision tree:

```
Need to run multiple VMs on one GPU?              -> vGPU
Hard isolation between mutually distrusting tenants? -> MIG
Multiple cooperating processes (your own replicas)? -> MPS
Just want oversubscription for dev/test?           -> time-slicing
Need full GPU performance for one big job?         -> no sharing
```

For LLM inference specifically: MIG 1g.10gb fits a 7B-parameter Llama in FP8 comfortably and gives 7$times$ throughput per H100 with linear latency; MPS gives even better aggregate throughput but with no SLA guarantees per replica.

== Further Reading

NVIDIA (2024). _NVIDIA Multi-Instance GPU User Guide_. https://docs.nvidia.com/datacenter/tesla/mig-user-guide/

NVIDIA (2024). _Multi-Process Service Documentation_. https://docs.nvidia.com/deploy/mps/

NVIDIA (2023). _Virtual GPU Software Documentation_. https://docs.nvidia.com/grid/

Anwar, A. et al. (2022). "MISO: Exploiting Multi-Instance GPU Capability on Multi-Tenant Systems for Machine Learning." _SoCC_.

Lim, S. et al. (2023). "Fractional GPU Sharing: A Survey of Techniques and Performance Models." _ACM CSUR_.

Kubernetes SIG-Node (2024). _NVIDIA GPU Operator Documentation_. https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/

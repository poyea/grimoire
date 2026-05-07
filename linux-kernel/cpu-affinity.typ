= CPU Affinity, Isolation, and NUMA

A modern server has dozens of cores, multiple NUMA nodes, hardware-interrupt sources, and a kernel that's free to migrate any task to any core at any time. For latency-sensitive or throughput-critical workloads, leaving the kernel's defaults in place is usually wrong. This chapter covers the primitives for *pinning* threads to cores, *isolating* cores from the rest of the system, and *binding* memory to NUMA nodes.

== Affinity Basics

Every task has a *CPU mask* — a bitmap of cores it is permitted to run on. By default the mask is "all cores"; the scheduler is free to migrate the task to any of them.

*Per-thread:*

```c
#include <pthread.h>
cpu_set_t set;
CPU_ZERO(&set);
CPU_SET(3, &set);                // only allow CPU 3
pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
```

*Per-process (or any task by tid):*

```c
#include <sched.h>
sched_setaffinity(pid /* or tid */, sizeof(set), &set);
```

Read it back with `pthread_getaffinity_np` / `sched_getaffinity`.

*From the shell:*

```
taskset -c 3,4,5 ./prog            # spawn pinned to CPUs 3-5
taskset -cp 3,4,5 <pid>            # change a running process's mask
```

*Inheritance:* the mask is inherited across `fork()` and `pthread_create`. A process can only widen its mask if it has `CAP_SYS_NICE` (or it's lowering capabilities permitted by `RLIMIT_RTPRIO` etc., depending on what it's narrowing/widening).

*Pinning vs migration vs preemption:* affinity prevents migration to cores outside the mask. It does not prevent preemption by another task on the *same* core, and it does not prevent IRQs from interrupting the core. For full isolation, see the next sections.

== CPU Topology and Sibling Awareness

Pin to the wrong core and you'll be sharing L1/L2 with a sibling hyperthread, or worse, talking across sockets every cache miss.

```
lscpu --extended                 # human-readable topology
lstopo --of console              # hwloc, even better visualization
cat /sys/devices/system/cpu/cpu0/topology/thread_siblings_list
cat /sys/devices/system/cpu/cpu0/topology/core_siblings_list
cat /sys/devices/system/cpu/cpu0/topology/package_cpus_list
```

Typical layout on a dual-socket server with 32 cores per socket and SMT enabled (so 64 logical CPUs per socket, 128 total):

#table(columns: (auto, 1fr),
  [*Logical CPU*], [*Resource*],
  [`0` and `64`], [SMT siblings (same physical core, share L1/L2)],
  [`0`-`31` and `64`-`95`], [Socket 0],
  [`32`-`63` and `96`-`127`], [Socket 1],
)

For latency-sensitive single-threaded work: pin to one logical CPU and *also* offline its sibling, or steer all other work away from that physical core. Sharing L1 with a noisy neighbor is brutal.

== Isolating Cores: isolcpus, nohz_full, rcu_nocbs

Affinity tells the kernel "don't run *me* there." The reverse is harder: "don't run *anything else* there." Three kernel-command-line parameters together produce an effectively dedicated core.

#table(columns: (auto, 1fr),
  [*Param*], [*Effect*],
  [`isolcpus=N,M-K`], [Removes cores from the scheduler's domain. Tasks are not load-balanced onto them. They run only what is explicitly affined to them.],
  [`nohz_full=N,M-K`], [Stops the periodic timer tick on these cores when only one task is running on them. No 1000 Hz interrupt eating ~1 µs every ms.],
  [`rcu_nocbs=N,M-K`], [Offloads RCU grace-period callback work from these cores onto kthreads on other cores. Avoids RCU softirq jitter.],
)

A complete isolation incantation in `/etc/default/grub` (modify `GRUB_CMDLINE_LINUX`):

```
isolcpus=4-7 nohz_full=4-7 rcu_nocbs=4-7
```

After reboot, cores 4-7 will:

- Not be picked by the scheduler unless a task is explicitly affined there.
- Run tickless when only a single task is active.
- Not handle RCU callbacks themselves.

You still need to:

- *Pin your work* to those cores explicitly (the kernel does *not* auto-pin).
- *Steer interrupts away* (see IRQ affinity below). `isolcpus` does *not* affect IRQ delivery.
- *Move kthreads off* if possible: `for k in /proc/$(pgrep kthreadd)/...` is hopeless; in practice use `tuned-adm profile realtime` or follow the RT-Linux project's steering scripts.

*sched_isolated check:* `cat /sys/devices/system/cpu/isolated` shows the kernel's view of the isolated set.

*Caveat:* `isolcpus` is officially deprecated in favor of cgroup v2 `cpuset` partitions. In practice it is still the simplest approach and is universally supported. The cpuset path:

```
mkdir /sys/fs/cgroup/isolated
echo "isolated" > /sys/fs/cgroup/isolated/cpuset.cpus.partition
echo "4-7"  > /sys/fs/cgroup/isolated/cpuset.cpus
```

This creates an *exclusive* cpuset partition; cores listed there are removed from the root domain, equivalent to `isolcpus`.

== IRQ Affinity

Hardware interrupts default to landing on whichever core happens to be selected by the chipset (often CPU 0, sometimes round-robin via `irqbalance`). For an isolated core to remain truly quiet, you must steer IRQs *away* from it.

```
cat /proc/interrupts                                # which CPU is taking each IRQ
cat /proc/irq/24/smp_affinity                        # mask for IRQ 24 (hex bitmap)
echo 0xf > /proc/irq/24/smp_affinity                 # only CPUs 0-3
echo 0-3 > /proc/irq/24/smp_affinity_list            # same, list form
```

There is also a global default for new IRQs:

```
echo 0xf > /proc/irq/default_smp_affinity
```

This only affects IRQs registered after the write — already-registered IRQs are unaffected.

*irqbalance* is the system daemon that re-pins IRQs periodically based on load. For latency-critical systems, *disable it* (`systemctl disable --now irqbalance`) and pin manually.

*Multi-queue NICs:* a 100 GbE NIC has 32-128 receive queues, each with its own IRQ. The convention is one queue per core, with IRQs steered to those cores so packets land on the same core that will process them (RSS — receive side scaling).

== NUMA Binding

On NUMA systems (any modern multi-socket server, plus AMD's chiplet architectures which present multiple NUMA-like CCDs), memory access latency depends on which node owns the page. Local DRAM is ~80-100 ns; remote (across the interconnect) is ~140-200 ns.

```
numactl --hardware                   # node layout, distances, free memory
cat /sys/devices/system/node/node0/cpulist
cat /sys/devices/system/node/node0/meminfo
cat /sys/devices/system/node/node0/distance
```

*Default policy:* "first-touch local." A page is allocated on the NUMA node of whichever core first writes to it. This works well if your threads stay put, but is wrong if they migrate or if memory is allocated by one thread and used by another.

*Pinning a process to a node:*

```
numactl --cpunodebind=0 --membind=0 ./prog
numactl --cpunodebind=0 --preferred=0 ./prog          # prefer, fall back to others
numactl --interleave=all ./prog                       # round-robin across all nodes
```

`--membind` is *strict*: allocations that can't be satisfied on the bound node fail rather than spilling. `--preferred` falls back. `--interleave` is appropriate for memory-bandwidth-bound, NUMA-agnostic workloads (some HPC kernels, some databases).

*Fine-grained from C:*

```c
#include <numa.h>
numa_run_on_node(0);                                  // bind threads
numa_set_preferred(0);                                // bind allocations
void *p = numa_alloc_onnode(size, 0);
```

Or via syscall directly:

```c
#include <numaif.h>
mbind(addr, len, MPOL_BIND, &nodemask, maxnode, MPOL_MF_MOVE);
set_mempolicy(MPOL_PREFERRED, &nodemask, maxnode);
```

`MPOL_MF_MOVE` migrates already-resident pages. Without it, only future allocations are affected.

*Diagnosing NUMA badness:*

```
numastat -p <pid>                                     # per-node alloc/access stats
perf stat -e numa-balancing-mp:numa_pte_updates ./prog   # auto-NUMA migration count
cat /proc/<pid>/numa_maps                             # per-VMA node distribution
```

A high `other_node` count in `numastat -p` is a smoking gun for cross-socket allocation patterns.

*Auto-NUMA balancing* (`/proc/sys/kernel/numa_balancing`): the kernel periodically samples access patterns by demoting pages to PROT_NONE, catching the resulting fault, and migrating the page if it consistently lives on the wrong node. Enabled by default on most distros. Helps for unconfigured workloads, hurts for workloads that have explicit policy. Disable for those (`echo 0 > /proc/sys/kernel/numa_balancing`).

== CPU Hotplug

Linux supports onlining and offlining individual CPUs at runtime:

```
echo 0 > /sys/devices/system/cpu/cpu5/online       # offline CPU 5
echo 1 > /sys/devices/system/cpu/cpu5/online       # back online
```

Use cases:

- *Disabling SMT siblings* of latency-critical cores.
- *Power-savings* in low-load periods (the kernel's `cpuidle` already handles this gracefully without going as far as full hotplug).
- *Mitigating noisy hardware* — a flaky core that posts MCEs can be parked.
- *Live kernel updates* (kpatch can avoid this, but some patches still rely on quiescing cores).

Offlining migrates all running tasks and pinned IRQs off the CPU. The associated kernel threads (per-CPU workers, kworkers, etc.) are also stopped. The CPU's caches are flushed.

*Caveat:* CPU 0 cannot be offlined on most x86 configurations — historic BIOS/firmware code paths assume it's always there. Modern kernels with `CONFIG_HOTPLUG_CPU0=y` relax this on capable platforms.

== Real-World Recipes

*Low-latency single-threaded server (e.g. matching engine, market-data feed handler):*

```
# /etc/default/grub
GRUB_CMDLINE_LINUX="isolcpus=2-5 nohz_full=2-5 rcu_nocbs=2-5 intel_pstate=disable"
```

```
# at boot
echo 1 > /sys/kernel/mm/transparent_hugepage/defrag/never   # avoid THP stalls
systemctl stop irqbalance
for i in /proc/irq/*/smp_affinity_list; do echo 0-1 > $i 2>/dev/null; done
```

Pin the hot thread to CPU 2, set `SCHED_FIFO`, `mlockall`, and you have a core that will receive a hardware interrupt only if you ask for one.

*High-throughput multi-queue NIC (DPDK or AF_XDP):*

```
ethtool -L eth0 combined 16                                  # 16 RX/TX queues
# pin queue N IRQ to CPU N for N in 0..15:
for q in $(seq 0 15); do
    irq=$(grep "eth0-rx-$q" /proc/interrupts | awk '{print $1}' | tr -d :)
    echo $q > /proc/irq/$irq/smp_affinity_list
done
ethtool -K eth0 ntuple on                                    # enable RSS steering
```

Each core processes its own queue; no cross-core packet handoff.

*NUMA-bound database:*

```
numactl --cpunodebind=0 --membind=0 ./postgres
```

Or split shards across nodes and bind each shard to its node's CPUs and memory.

== Diagnostics Checklist

```
taskset -cp <pid>                                            # current affinity
cat /proc/<pid>/status | grep Cpus_allowed_list              # same, kernel side
cat /proc/<pid>/status | grep Mems_allowed_list              # NUMA mem mask
grep '^processor' /proc/cpuinfo | wc -l                      # online CPU count
cat /sys/devices/system/cpu/isolated                          # isolated set
cat /sys/devices/system/cpu/nohz_full                         # tickless set
numastat -p <pid>                                            # NUMA alloc breakdown
perf stat -e migrations,context-switches ./prog              # how much the kernel moves us
```

If `migrations` is non-zero on a pinned task, your affinity is broken. If `context-switches` is high on an isolated core, something else is running there.

*See also:* _cpu-architecture/multicore.typ_ (NUMA architecture and cache coherence at the hardware level), _scheduler.typ_ (RT scheduling classes that complement isolation), _interrupts.typ_ (IRQ infrastructure beneath `smp_affinity`).

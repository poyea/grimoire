= cgroups and Namespaces

A "container" is not a kernel object. There is no `struct container` anywhere in the kernel source. What runtimes like Docker, containerd, and Kubernetes ship is a userspace orchestration layer that calls two largely-independent kernel features: *namespaces* (which give a process its own view of system resources) and *cgroups* (which limit and account those resources). Together they are sufficient to build OS-level virtualization that is much cheaper than full VMs.

== Namespaces: Per-Process Views of the System

A namespace virtualizes a particular kernel resource so that processes inside it see only their own copy. Linux has eight namespace types (as of 6.x):

#table(columns: (auto, 1fr),
  [*Namespace*], [*What it virtualizes*],
  [`mnt`], [The mount table. Different mnt namespaces see different mountpoint hierarchies.],
  [`pid`], [Process IDs. A process inside a pid namespace sees a private PID 1; the same process has a different PID in the parent ns.],
  [`net`], [Network devices, IP addresses, routing tables, sockets, iptables/nftables rules.],
  [`uts`], [Hostname and domain name (`uname` output).],
  [`ipc`], [System V IPC (semaphores, shared memory, message queues), POSIX message queues.],
  [`user`], [UIDs and GIDs. Lets unprivileged users be root *inside* a namespace without being root outside.],
  [`cgroup`], [The /proc/self/cgroup view. Hides the host cgroup hierarchy.],
  [`time`], [`CLOCK_BOOTTIME` and `CLOCK_MONOTONIC` offsets. (Linux 5.6+)],
)

Three primitives create or join namespaces:

- *`clone()` with `CLONE_NEW*` flags* — fork a child into one or more new namespaces.
- *`unshare()`* — move the *current* process into new namespaces.
- *`setns()`* — join an existing namespace via an open fd to `/proc/<pid>/ns/<type>`.

```c
#define _GNU_SOURCE
#include <sched.h>
#include <sys/wait.h>

int child(void *arg) {
    sethostname("container", 9);
    execlp("/bin/bash", "bash", NULL);
    return 0;
}

char stack[1024 * 1024];
clone(child, stack + sizeof(stack),
      CLONE_NEWUTS | CLONE_NEWPID | CLONE_NEWNS | SIGCHLD,
      NULL);
wait(NULL);
```

This child has its own UTS namespace (so `sethostname` is private), its own PID namespace (it sees itself as PID 1), and its own mount namespace.

*From the shell:* `unshare` and `nsenter` are the userspace tools.

```
unshare --pid --fork --mount-proc /bin/bash         # new PID ns, remount /proc
nsenter -t 1234 -n /bin/bash                         # enter PID 1234's net ns
```

*Inspecting:* `ls -l /proc/<pid>/ns/` lists each namespace by inode. Two processes share a namespace iff their inodes match.

```
$ ls -l /proc/$$/ns
lrwxrwxrwx 1 root root 0 ... mnt   -> 'mnt:[4026531841]'
lrwxrwxrwx 1 root root 0 ... pid   -> 'pid:[4026531836]'
lrwxrwxrwx 1 root root 0 ... net   -> 'net:[4026531840]'
...
```

*The user namespace is special:* it lets unprivileged users be root within their namespace. Because privilege checks resolve relative to the user namespace, a user-namespace root that does *not* hold any capabilities outside has limited reach — they can't, e.g., load kernel modules. This is what enables *rootless* containers (`podman` without sudo, `userns-remap` in Docker).

*Network namespaces* deserve special mention. Each net namespace has its own network stack — own loopback, own routing table, own iptables. Linking them is done with virtual ethernet pairs (`veth`):

```
ip netns add red
ip link add veth0 type veth peer name veth1
ip link set veth1 netns red
ip addr add 10.0.0.1/24 dev veth0
ip netns exec red ip addr add 10.0.0.2/24 dev veth1
ip link set veth0 up
ip netns exec red ip link set veth1 up
ip netns exec red ip link set lo up
```

Now packets routed to 10.0.0.2 cross from the host's net ns into `red`'s. This is how Kubernetes' CNI plugins (Calico, Cilium, Flannel) build pod networking.

== cgroups v2

Control groups (cgroups) are how the kernel limits and accounts CPU, memory, IO, and other resources per *group of processes*. Version 2 (mainline since 4.5; the default on most modern distros — Fedora, Debian 11+, Ubuntu 22.04+, RHEL 9+) replaced v1's per-controller hierarchies with a single unified hierarchy. Some legacy LTS distros (RHEL 7, Ubuntu 18.04) still default to v1.

*Hierarchy:* a tree rooted at `/sys/fs/cgroup/`. Each directory is a cgroup. Subdirectories are sub-cgroups. Files inside each directory configure or report on that cgroup.

```
$ ls /sys/fs/cgroup/
cgroup.controllers       cgroup.subtree_control   cpu.stat
cgroup.max.depth         init.scope/              memory.stat
cgroup.procs             system.slice/             ...
cgroup.threads           user.slice/
```

`system.slice/`, `user.slice/`, `init.scope/` are systemd's default partitioning of the system. Each .service unit gets its own subgroup under `system.slice/`.

*Putting a process in a cgroup:* write its PID to `cgroup.procs`.

```
mkdir /sys/fs/cgroup/myapp
echo $$ > /sys/fs/cgroup/myapp/cgroup.procs
```

Children of that process inherit the cgroup unless explicitly moved.

*Enabling controllers:* a cgroup can use a controller only if its parent has it enabled in `cgroup.subtree_control`. The cgroup-v2 tree is *constructed top-down*: the root advertises every available controller, you pick which to push down to children.

```
echo "+cpu +memory +io" > /sys/fs/cgroup/cgroup.subtree_control
```

=== The cpu Controller

Bandwidth limit and CPU weight.

```
# Cap the cgroup at 50% of one CPU
echo "50000 100000" > /sys/fs/cgroup/myapp/cpu.max     # 50ms quota per 100ms period

# Or 200% (two CPUs):
echo "200000 100000" > /sys/fs/cgroup/myapp/cpu.max

# CPU weight (default 100, range 1-10000)
echo 200 > /sys/fs/cgroup/myapp/cpu.weight             # 2× share when contended

# Stats:
cat /sys/fs/cgroup/myapp/cpu.stat
# usage_usec 1234567
# user_usec 1100000
# system_usec 134567
# nr_periods 12345
# nr_throttled 42         <-- how many periods were throttled
# throttled_usec 678
```

*The throttling pitfall:* `nr_throttled` non-zero means the cgroup hit its quota and was suspended for the rest of the period. This is the canonical cause of mysterious latency spikes in containerized environments. Even if average CPU is 30%, occasional bursts that hit the quota produce 50-100 ms stalls.

Pre-5.4 kernels had an additional, infamous bug: *child cgroup CPU quotas could be charged double under specific scheduling conditions.* Tracked as the "CFS bandwidth bug." Fixed in 5.4 with the `CONFIG_CFS_BANDWIDTH=y` rework.

=== The memory Controller

```
# Hard limit (process gets OOM-killed if it tries to exceed and can't reclaim)
echo "2G" > /sys/fs/cgroup/myapp/memory.max

# Swap limit
echo "1G" > /sys/fs/cgroup/myapp/memory.swap.max

# Soft limit / reclaim trigger
echo "1500M" > /sys/fs/cgroup/myapp/memory.high

# Min guarantee (protected from reclaim)
echo "256M" > /sys/fs/cgroup/myapp/memory.min

# Stats:
cat /sys/fs/cgroup/myapp/memory.stat
# anon 1073741824
# file 268435456
# kernel_stack 16777216
# slab 33554432
# ...
```

`memory.high` is preferred over `memory.max` for production: it triggers reclaim when crossed but doesn't kill the process. It's a back-pressure mechanism, not a hard wall. Cross both `memory.high` and run out of reclaimable memory and the OOM killer eventually fires.

*OOM:* when a cgroup runs out of memory, the kernel's OOM killer picks a victim *within that cgroup* (`oom_score_adj` adjustable per-process). With `memory.oom.group=1`, the entire cgroup is killed atomically — useful for "either everyone lives or everyone dies" workloads.

=== The io Controller

Throttle by IOPS or bandwidth, per block device.

```
# major:minor for device:
ls -l /dev/sda                                 # 8:0
echo "8:0 rbps=10485760 wbps=10485760" > /sys/fs/cgroup/myapp/io.max
echo "8:0 riops=1000 wiops=1000" >> /sys/fs/cgroup/myapp/io.max

# Stats
cat /sys/fs/cgroup/myapp/io.stat
```

*Caveat:* the io controller works well for direct block IO. Buffered writes go through the page cache and may be charged to the wrong cgroup at writeback time. The `io.cost` model attempts to fix this; it's still maturing.

=== The cpuset Controller

Restrict a cgroup to a set of CPUs and NUMA nodes.

```
echo "0-3"    > /sys/fs/cgroup/myapp/cpuset.cpus
echo "0"      > /sys/fs/cgroup/myapp/cpuset.mems

# Exclusive partition (cores listed are removed from root domain;
# functionally equivalent to isolcpus):
echo "root"   > /sys/fs/cgroup/myapp/cpuset.cpus.partition
```

Combined with `cpu.max` and `memory.max`, cpuset gives you the resource-allocation primitive used by Kubernetes' `cpu-manager-policy: static`.

=== The pid Controller

Cap the number of processes/threads a cgroup can spawn (basic fork-bomb defense).

```
echo 1000 > /sys/fs/cgroup/myapp/pids.max
```

== Putting It Together: A Minimal Container

A container *is* a process tree in:

- A new mount namespace (with its own root via `pivot_root` to a prepared filesystem).
- A new pid namespace (process inside sees itself as PID 1).
- A new uts namespace (own hostname).
- A new net namespace (own veth pair into the host bridge).
- A new ipc namespace.
- A new user namespace (optionally; the basis of rootless containers).
- A cgroup with `cpu.max`, `memory.max`, `pids.max`, `io.max` set as desired.
- Optionally a seccomp filter restricting syscalls.
- Optionally Linux capabilities dropped (`CAP_NET_ADMIN`, `CAP_SYS_ADMIN`, etc.).
- Optionally an apparmor or selinux label.

That sequence, in C, is roughly 100 lines. Docker's value-add is everything *around* it: image format, layered filesystem (overlayfs), registry protocol, networking plugins, lifecycle management. The kernel-level isolation is just `clone()` + `mount()` + a few `cgroup` writes.

Skeleton (illustrative; production runtimes are more careful about ordering and error handling):

```c
// Note: CLONE_NEWCGROUP only virtualizes the /proc/self/cgroup view;
// the cgroup itself must be created and joined separately (see below).
unshare(CLONE_NEWNS | CLONE_NEWUTS | CLONE_NEWPID | CLONE_NEWIPC |
        CLONE_NEWNET | CLONE_NEWUSER | CLONE_NEWCGROUP);

// In the new mount ns: pivot_root to image directory.
mount("none", "/", NULL, MS_REC | MS_PRIVATE, NULL);
mount(image_dir, image_dir, NULL, MS_BIND, NULL);
chdir(image_dir);
syscall(SYS_pivot_root, ".", "old_root");
umount2("old_root", MNT_DETACH);

// Mount /proc fresh in the new pid ns.
mount("proc", "/proc", "proc", 0, NULL);

// Apply seccomp.
seccomp_load(filter_ctx);

// Drop capabilities.
prctl(PR_CAPBSET_DROP, CAP_SYS_ADMIN, 0, 0, 0);
// ...

// Exec the container's entrypoint.
execv(argv[0], argv);
```

== systemd Integration

systemd uses cgroups extensively. Every service unit is its own cgroup. Useful directives in `.service` files:

```
[Service]
CPUQuota=200%                    # cpu.max
MemoryMax=2G                     # memory.max
MemoryHigh=1500M                 # memory.high
TasksMax=4096                    # pids.max
IOReadBandwidthMax=/dev/sda 50M  # io.max
CPUAffinity=2-5                   # taskset on start
NUMAPolicy=bind
NUMAMask=0
```

`systemctl status <unit>` shows the cgroup path. `systemd-cgtop` is a `top`-like view of cgroup resource usage.

*Migration from cgroup v1:* most distros default to v2-only since 2022. The `systemd.unified_cgroup_hierarchy=1` (or `=0` for v1) kernel parameter switches modes. v1 is still supported for legacy applications but is deprecated.

== Diagnostics

```
cat /proc/<pid>/cgroup                           # which cgroup is this PID in
systemd-cgls                                     # tree view of all cgroups
systemd-cgtop                                    # live resource usage
cat /sys/fs/cgroup/.../cpu.stat                   # CPU usage and throttling
cat /sys/fs/cgroup/.../memory.events              # OOM events, reclaim pressure
lsns                                             # list namespaces and their members
```

If a containerized workload has tail-latency outliers, `cpu.stat`'s `nr_throttled` is the first thing to check. If memory ballooning is suspect, `memory.events` shows whether `memory.high` reclaim is firing.

*See also:* _cpu-affinity.typ_ (cpuset partitions overlap with `isolcpus`), _scheduler.typ_ (cpu controller integrates with CFS bandwidth), _abi-syscalls.typ_ (seccomp, the syscall-filter complement to cgroup limits).

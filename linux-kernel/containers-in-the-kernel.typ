#import "../template.typ": xref

= Containers in the Kernel

A "container" is a userspace fiction. The kernel knows only namespaces, cgroups, capabilities, seccomp filters, LSM labels, and bind mounts; Docker, podman, containerd, runc, and the Kubernetes kubelet are orchestrators that assemble these primitives into a coherent isolation unit. Understanding what the kernel actually does (especially the user-namespace pitfalls and root/unprivileged trade-offs) is essential to running containers safely.

_Cgroups and Namespaces_ introduced the namespace types at the API level. This chapter goes deeper: how each namespace virtualizes its resource, the user-namespace security model and its pitfalls, the syscalls runtimes actually issue, and the kernel paths a container runtime traverses on every `docker run`.

== The Seven Namespaces

#table(columns: (auto, 1fr),
  [`CLONE_NEWPID`], [Virtualizes the PID space. The first process is PID 1 inside the namespace; the kernel translates PIDs in syscalls (`kill`, `getpid`, `/proc`) automatically. Killing PID 1 reaps the whole namespace.],
  [`CLONE_NEWNS`], [Mount namespace. Each NS has its own `mount_hashtable`. Bind mounts, overlay rootfs, and `pivot_root` build the container's filesystem view.],
  [`CLONE_NEWNET`], [Network namespace. Independent loopback, routing table, netfilter rules, sockets, network devices. veth pairs bridge to the host or to other namespaces.],
  [`CLONE_NEWIPC`], [IPC namespace. SysV IPC ids, POSIX message queues, `/dev/shm` (via private tmpfs mount).],
  [`CLONE_NEWUTS`], [UTS namespace. `hostname`, `domainname`.],
  [`CLONE_NEWUSER`], [User namespace. Per-namespace uid/gid maps; the cornerstone of unprivileged containers.],
  [`CLONE_NEWCGROUP`], [Cgroup namespace. Virtualizes the view of `/proc/self/cgroup` and `/sys/fs/cgroup`.],
  [`CLONE_NEWTIME`], [Time namespace (5.6+). Per-namespace `CLOCK_MONOTONIC` and `CLOCK_BOOTTIME` offsets, used for checkpoint/restore.],
)

`clone3` is the modern syscall (replacing `clone` for new flags) and takes a `struct clone_args` with `flags` (`CLONE_NEW*` bitmask) and a `cgroup` fd. `unshare(2)` creates new namespaces for the current process; `setns(2)` joins an existing one (via an open fd into `/proc/PID/ns/<type>`).

== PID Namespace

A new PID namespace makes the first process *PID 1*, with all the kernel's special handling: SIGKILL/SIGSTOP are blocked for it (so external killers can't terminate the init), unhandled signals from inside are ignored, and termination of PID 1 destroys the namespace (the kernel sends SIGKILL to every other member).

```c
int pid = syscall(SYS_clone3, &(struct clone_args){
    .flags = CLONE_NEWPID | CLONE_NEWNS | CLONE_NEWUSER,
    .exit_signal = SIGCHLD,
}, sizeof(struct clone_args));
if (pid == 0) {
    // child: PID 1 in new namespace
    setsid();
    execv(argv[0], argv);
}
```

A *common bug*: applications that previously ran as PID 9000 with proper signal handlers suddenly become PID 1 in a container, and the kernel's signal-blocking-for-init rule means SIGTERM is silently dropped. The fix is either a proper init (`tini`, `dumb-init`, `s6-svscan`) or explicit signal handlers in the application.

`/proc` only shows processes in the current PID namespace, but PID namespaces nest: PID 1 in a child namespace also has a translated PID visible from the parent. Inspecting from outside: `readlink /proc/PID/ns/pid`.

== Mount Namespace and pivot_root

A new mount namespace inherits the parent's mounts at creation; subsequent mount changes are private (depending on the mount's *propagation* type: `private`, `shared`, `slave`, `unbindable`; see `Documentation/filesystems/sharedsubtree.rst`).

The container rootfs construction:

```c
unshare(CLONE_NEWNS);
// Make every existing mount private; host won't see our changes
mount(NULL, "/", NULL, MS_REC | MS_PRIVATE, NULL);

// Bind-mount the container rootfs at /tmp/new_root
mount(image_dir, "/tmp/new_root", NULL, MS_BIND | MS_REC, NULL);

// Mount the standard pseudo-filesystems
mount("proc",   "/tmp/new_root/proc", "proc",   0, NULL);
mount("sysfs",  "/tmp/new_root/sys",  "sysfs",  0, NULL);
mount("tmpfs",  "/tmp/new_root/tmp",  "tmpfs",  0, NULL);
mount("devtmpfs", "/tmp/new_root/dev", "devtmpfs", 0, NULL);

// Pivot root
mkdir("/tmp/new_root/old_root", 0700);
chdir("/tmp/new_root");
syscall(SYS_pivot_root, ".", "old_root");
umount2("/old_root", MNT_DETACH);
rmdir("/old_root");
```

`pivot_root` swaps the namespace's root with a new directory and pushes the old root somewhere we can later unmount. Modern runtimes (runc) use the same dance.

Overlayfs is the standard image-layering vehicle: container images are stacks of read-only `lowerdir` layers; the container gets a writable `upperdir` on top. See #xref("linux-kernel", "vfs-and-fs", label: "VFS and Filesystems").

== Network Namespace

A fresh netns has only a `lo` (down). To talk to the host or other containers, runtimes create a *veth pair*: one end in the container, the other in the host (typically attached to a bridge `docker0` / `cni0`).

```bash
ip netns add foo
ip link add veth0 type veth peer name vethfoo
ip link set vethfoo netns foo
ip link set veth0 master docker0
ip link set veth0 up
ip netns exec foo ip link set vethfoo up
ip netns exec foo ip addr add 172.17.0.2/16 dev vethfoo
ip netns exec foo ip route add default via 172.17.0.1
```

The host runs SNAT (`iptables -t nat -A POSTROUTING -s 172.17.0.0/16 -j MASQUERADE`) for outbound, optional DNAT for ingress. Modern CNI plugins (Cilium, Calico) skip iptables entirely and use eBPF at TC/XDP for the data plane.

Each netns has independent sockets (even loopback). `ip netns exec` is shorthand for `setns(netns_fd, CLONE_NEWNET)` then `execve`.

== User Namespace: The Cornerstone of Unprivileged

A user namespace defines a *uid/gid mapping* between the inner and outer namespace. A process can be uid 0 *inside* the namespace while being uid 100000 (or completely unprivileged) outside. All capability checks against resources owned by other namespaces are denied; capabilities apply only to objects whose owners are in the current namespace's mapping.

```c
unshare(CLONE_NEWUSER);
// Write uid_map: "inside_uid outside_uid count"
int f = open("/proc/self/uid_map", O_WRONLY);
write(f, "0 100000 65536", 14);   // inner 0..65535 → outer 100000..165535
close(f);
// gid_map requires first writing /proc/self/setgroups "deny"
f = open("/proc/self/setgroups", O_WRONLY);
write(f, "deny", 4);
close(f);
f = open("/proc/self/gid_map", O_WRONLY);
write(f, "0 100000 65536", 14);
close(f);
```

This is what makes *rootless* podman, *unprivileged* LXC, and Docker's `userns-remap` work: containers can have a "root" without that root having any host-side privilege.

== User-Namespace Pitfalls

User namespaces are powerful and have been the source of a steady CVE stream (CVE-2013-1858, CVE-2016-3134, CVE-2022-0185, CVE-2022-25636 ...). The patterns:

- *Unprivileged user namespaces expand the kernel attack surface.* Suddenly an unprivileged user can reach `mount`, `clone`, netfilter, and many other paths previously locked behind `CAP_SYS_ADMIN`. Many CVEs are in code that historically assumed only root could reach it.

  Mitigation: `kernel.unprivileged_userns_clone=0` (Debian/Ubuntu default) disables them for non-root. Recent kernels expose `kernel.apparmor_restrict_unprivileged_userns=1` (Ubuntu 23.10+) to gate via AppArmor.

- *setuid binaries inside user namespaces are dangerous.* The `setuid` bit transitions to a uid that might map to host uid 0. Modern kernels ignore setuid bits in non-init user namespaces (`MS_NOSUID` implicit) but some filesystems and corner cases have historically slipped through.

- *Host filesystems mounted into containers* with `nosuid,nodev` are essential; without them, a malicious image can ship a setuid root binary.

- *Mapping the host's root into a container* (`uid_map: 0 0 65536`) defeats the protection entirely; the container's root is *host's* root.

- *id-mapped mounts* (5.12+) let a single host mount appear with a uid/gid translation per-mountpoint, solving the "the image was tarred with uid 0, I want to run as uid 1000 inside" problem without `chown`-ing the bind mount.

- *capabilities inside user namespaces* apply to objects *owned by the namespace* (and child namespaces' objects). A capability-laden inner root cannot touch host objects.

== capabilities

Container runtimes drop most capabilities by default. The typical Docker default keeps:

`CAP_CHOWN, CAP_DAC_OVERRIDE, CAP_FSETID, CAP_FOWNER, CAP_MKNOD, CAP_NET_RAW, CAP_SETGID, CAP_SETUID, CAP_SETFCAP, CAP_SETPCAP, CAP_NET_BIND_SERVICE, CAP_SYS_CHROOT, CAP_KILL, CAP_AUDIT_WRITE`

Things to drop further for hardening: `CAP_NET_RAW` (raw sockets, enabling TCP/IP spoofing), `CAP_SYS_CHROOT` (escape via chroot tricks if you have a backup plan).

Things never to grant unless absolutely necessary: `CAP_SYS_ADMIN` (the "almost root" capability), `CAP_SYS_PTRACE`, `CAP_SYS_MODULE`, `CAP_NET_ADMIN`.

`--privileged` containers grant *all* capabilities plus device access; the docker daemon's seccomp profile is also disabled, making this effectively root on the host. Reserve for kernel development VMs.

== seccomp Profile

The Docker default seccomp profile (defined in `moby/profiles/seccomp/default.json`) allows ~340 syscalls and forbids the rest with `EPERM`. The blocked list includes `keyctl` (kernel keyrings, several CVEs), `mount` (unless `CAP_SYS_ADMIN`), `add_key`, `unshare`, `setns` (gated by capabilities), `bpf` (gated by `CAP_BPF`), the io_uring family on some configs.

Kubernetes pod spec:

```yaml
securityContext:
  seccompProfile:
    type: RuntimeDefault   # or Localhost with localhostProfile: my.json
```

See #xref("linux-kernel", "security-modules", label: "Security Modules") for seccomp internals.

== cgroups in the Container

`memory.max`, `cpu.max`, `pids.max`, `io.max`, `cpuset.cpus`, `cpuset.mems` are the basic resource caps. Beyond that, runtimes typically set:

- `memory.swap.max=0` to disable swap in the container.
- `memory.oom.group=1` so OOM kills the whole cgroup atomically (avoids partial kill leaving zombie services).
- `pids.max` to prevent fork bombs.
- `devices.deny`/`allow` rules to gate `/dev/*` access.

cgroup namespaces (`CLONE_NEWCGROUP`) make `/proc/self/cgroup` and `/sys/fs/cgroup` show paths *relative* to the container's cgroup root, so the container sees `/` instead of `/system.slice/docker-abc.scope`.

== Container Lifecycle: What runc Actually Does

A minimal trace of `runc create`:

1. Read OCI spec (`config.json`).
2. `clone3(CLONE_NEW{NS,PID,NET,IPC,UTS,USER,CGROUP})` to create the namespaces.
3. In child: write `uid_map`/`gid_map` (delegated from parent via privileged setuid helper `newuidmap`/`newgidmap` for unprivileged mode).
4. Mount procfs/sysfs/devtmpfs into the rootfs.
5. `pivot_root` into rootfs.
6. Apply LSM label (`PR_SET_PDEATHSIG`, `prctl(PR_SET_KEEPCAPS)`).
7. Drop capabilities (cap_set_proc).
8. Install seccomp filter.
9. `execve(entrypoint)`.

The runtime keeps an *init* process holding open the namespace fds; `runc exec` later joins via `setns`.

== Checkpoint/Restore (CRIU)

CRIU (Checkpoint/Restore In Userspace) walks `/proc/PID/*` to snapshot the entire process tree (memory maps, open fds, sockets, signal state, namespaces) and serializes to image files. Restore re-creates the namespaces, restores memory via `process_vm_writev` and `mmap` + ptrace, reopens fds (using filesystem state), restores TCP connections (via `repair` mode that lets you set sequence numbers and the kernel believes it).

Time namespace was added partly for CRIU; without per-namespace `CLOCK_MONOTONIC` offsets, a restored process's monotonic clock would jump backward.

CRIU underlies the "live migration" of LXC containers, Kubernetes pod migration prototypes, and some serverless cold-start optimizations.

== rootless Containers

The state of the art for unprivileged containers (podman, buildah, rootless Docker):

- One unprivileged user namespace per container (`newuidmap` allocates from `/etc/subuid`).
- `slirp4netns` or `pasta` for networking (userspace TCP/IP stack; no veth pair needs root).
- `fuse-overlayfs` (or kernel overlayfs since 5.11 with `userxattr` mount option) for layered rootfs without root.
- cgroup v2 with delegation (`Delegate=yes` in the user's systemd slice) so the user owns a sub-tree.

Rootless is now the recommended posture for desktop/CI use; production servers typically still run rootful for cgroup-v2-delegation simplicity and SELinux integration.

== Common Failure Modes

- *"Operation not permitted" on `mount` in a privileged container*: missing `CAP_SYS_ADMIN` or `nosuid`/`nodev` propagation.
- *PID 1 doesn't reap zombies*: no proper init in image; use `--init` (runc spawns `tini`) or include one.
- *Containers seeing host swap usage*: `memory.swap.max` not set; the container's memcg sees its own limit but stats files may still show host values without cgroupns.
- *DNS broken inside container*: `/etc/resolv.conf` is a bind mount from a host file the orchestrator manages; bind mounting from the wrong source confuses everything.
- *iptables rules disappear*: a netns-aware iptables-nft conflict; check `iptables-legacy` vs `iptables-nft` symlinks.

== Observability

```bash
# What namespaces does this process live in?
ls -l /proc/PID/ns/

# Compare two containers' namespaces
readlink /proc/A/ns/net /proc/B/ns/net

# Enter a container's namespaces
nsenter -t PID -p -m -n -u -i bash

# Per-cgroup PSI for memory pressure
cat /sys/fs/cgroup/system.slice/docker-abc.scope/memory.pressure

# Trace syscalls denied by seccomp
bpftrace -e 'tracepoint:syscalls:sys_exit_*  /args->ret == -1/ { @[probe] = count(); }'
```

== Further Reading

Kernel docs: `Documentation/admin-guide/namespaces/`, `Documentation/admin-guide/cgroup-v2.rst`, `Documentation/filesystems/sharedsubtree.rst`, `Documentation/filesystems/idmappings.rst`.

Kerrisk, M. (2013-2024). _Namespaces in operation_ — LWN.net series, the authoritative tutorial.

Walsh, D. (2018). _Why podman uses fork/exec_. Container security posts at developers.redhat.com.

Suda, A. (2021). _Rootless Containers_ — usenix.org.

Edge, J. (2022). _Unprivileged user namespaces_ — LWN; series on CVEs and AppArmor restrictions.

OCI Runtime Spec — #link("https://github.com/opencontainers/runtime-spec")[github.com/opencontainers].

`kernel/nsproxy.c`, `kernel/user_namespace.c`, `kernel/pid_namespace.c`, `fs/namespace.c`, `fs/mount.h`, `net/core/net_namespace.c`.

*See also:* #xref("linux-kernel", "cgroups-namespaces", label: "Cgroups and Namespaces") (the API-level introduction), #xref("linux-kernel", "security-modules", label: "Security Modules") (seccomp, capabilities, per-container LSM profiles), #xref("linux-kernel", "vfs-and-fs", label: "VFS and Filesystems") (overlayfs, bind mounts, id-mapped mounts), #xref("linux-kernel", "networking-stack", label: "Networking Stack") (veth, netfilter, eBPF service mesh).

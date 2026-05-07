= Introduction

The Linux kernel is the most-deployed software artifact on Earth: it runs on the majority of internet-facing servers, every Android phone, most embedded devices, all major cloud hyperscalers, and the top 500 supercomputers without exception. For systems engineers, kernel internals are not optional knowledge — they are the substrate against which application performance, latency, and reliability are ultimately judged.

*Scope of this reference:*

- *ABI and syscalls:* x86-64 calling convention, syscall vs vDSO, seccomp filtering
- *Memory:* mmap mechanics, COW, transparent huge pages, hugetlbfs, userfaultfd
- *CPU affinity:* pinning, core isolation, NUMA binding, IRQ steering
- *Scheduling:* CFS internals, real-time classes, SCHED_DEADLINE
- *Containers:* cgroups v2, namespaces, the primitives Docker actually calls
- *Interrupts:* top/bottom half split, softirqs, NAPI, threaded IRQs
- *Tracing:* kprobes, ftrace, perf, eBPF, bpftrace
- *Modules:* writing out-of-tree kernel modules, char devices, procfs/sysfs

*Kernel version focus:* Linux 6.x mainline, with particular attention to the current LTS releases (6.1 and 6.6). Most material applies cleanly back to 5.10. Where a feature is newer (e.g. `io_uring` napi-busy-poll, `sched_ext`), the introduction kernel is noted.

*Intended audience:* Systems engineers, SREs, and performance specialists who already write user-space code competently and now need to understand what happens beneath `glibc`. We assume comfort with C, gdb, and the Linux command line.

*Code conventions:* C is the kernel's native language. User-space examples are C unless a particular tool is more idiomatic in another form (`bpftrace` scripts, shell). All examples target x86-64 Linux 6.x; ARM64 differences are flagged where relevant.

*See also:* _cpu-architecture/multicore.typ_ (NUMA, cache coherence), _cpu-architecture/virtual-memory.typ_ (TLB, page tables), _networking/kernel-bypass.typ_ (DPDK, AF_XDP), _coding/advanced-systems.typ_ (scheduler algorithms, lock-free data structures).

== Why Kernel Internals Matter

A user-space program is the tip of an iceberg. Every meaningful operation — opening a file, sending a packet, allocating memory, waking a thread — eventually traps into the kernel. When a service has a tail-latency outlier at p99.9, the explanation almost always lies below the syscall boundary:

- A `read()` blocked on a page fault that hit a slow disk.
- A thread that ran on the wrong NUMA node and paid 100 ns of remote-memory latency per access.
- An interrupt storm on the wrong core that preempted the latency-critical thread.
- A cgroup `cpu.max` throttle that delayed a task by tens of milliseconds.
- A TLB shootdown IPI that stalled every core in the system.

You cannot diagnose any of these from user-space alone. The kernel is the system's shared substrate, and understanding it lets you observe and influence behavior that would otherwise look like noise.

== The Kernel-User Boundary

Linux runs in two privilege levels (on x86, ring 0 and ring 3). User code cannot directly touch hardware, page tables, or other processes' memory. Every privileged operation goes through one of three paths:

- *Syscall:* Explicit request via the `syscall` instruction. ~100-300 cycles for the ring transition itself, plus whatever the kernel handler does. See _abi-syscalls.typ_.
- *Page fault:* Implicit trap when the MMU rejects a memory access. The kernel's fault handler decides whether to fix it (allocate a page, COW-clone, fetch from swap) or kill the process with `SIGSEGV`.
- *Interrupt:* Hardware-initiated. The CPU is yanked from whatever it was doing into a kernel handler. See _interrupts.typ_.

Crossing the boundary is not free. A modern CPU pays for the privilege change, the pipeline flush, the kernel's argument validation (`copy_from_user`), and any scheduling work that happens on return. The vDSO exists precisely to avoid this cost for read-only operations like `clock_gettime`.

== The /proc and /sys Filesystems

Linux exposes most of its kernel state as virtual filesystems, not as opaque APIs. This is one of its defining design choices.

```
/proc/<pid>/status        process state, threads, memory
/proc/<pid>/maps          virtual memory layout
/proc/<pid>/stack         kernel-side stack trace
/proc/cpuinfo             per-cpu features
/proc/interrupts          per-CPU interrupt counts
/proc/meminfo             system-wide memory stats
/proc/sys/                tunable kernel parameters (sysctls)

/sys/devices/system/cpu/  per-CPU online state, freq, topology
/sys/kernel/mm/           memory management tunables (THP, ksm, hugepages)
/sys/fs/cgroup/           cgroup v2 hierarchy
/sys/class/net/           network devices
/sys/kernel/debug/        debugfs (ftrace lives here)
```

These files are generated on demand by kernel code, not stored on disk. Writing to most of `/proc/sys` or `/sys` is the canonical way to tune kernel behavior at runtime.

== Tools Overview

Three tools cover ~80% of kernel-side investigation:

- *`strace`* — traces syscalls of a process. Cheap, ubiquitous, perfect for "what is this process actually asking the kernel to do?"
  ```
  strace -fp <pid>            # follow forks, attach to running pid
  strace -c -p <pid>          # summary table of syscall counts/times
  strace -e trace=openat,read <pid>
  ```

- *`perf`* — hardware-counter-based profiler, also reads kernel tracepoints.
  ```
  perf top -g                 # live system-wide profile, with call graphs
  perf record -F 99 -a -g sleep 30   # sample at 99 Hz for 30 s, all CPUs
  perf stat -e cache-misses,cycles,instructions ./prog
  perf trace -p <pid>         # strace-like view but powered by tracepoints
  ```

- *`bpftrace`* — eBPF-based tracing, lets you write one-liners that attach to almost any kernel event.
  ```
  bpftrace -e 'tracepoint:syscalls:sys_enter_openat { @[comm] = count(); }'
  bpftrace -e 'kprobe:vfs_read { @[pid, comm] = count(); }'
  bpftrace -e 'tracepoint:sched:sched_switch /pid == 1234/ { @[kstack] = count(); }'
  ```

Other essentials:

- *`ftrace`* (via `/sys/kernel/debug/tracing` or `trace-cmd`) — built-in function tracer.
- *BCC tools* — `execsnoop`, `biolatency`, `tcplife`, `runqlat`, etc. Each is a packaged eBPF program.
- *`gdb` + `kgdb`* — kernel-side debugging (rarely used in production).
- *`crash`* — analyzing kernel crash dumps (`/proc/vmcore` via kdump).

The full chapter on tracing is _kernel-tracing.typ_; this is the orientation list.

== Map of This Book

The chapters are arranged bottom-up: each one assumes the primitives from previous chapters are understood.

#table(columns: (auto, 1fr),
  [*Chapter*], [*What you take away*],
  [abi-syscalls], [How a `read()` becomes a kernel call; what `vDSO` does for you for free],
  [mmap-memory], [Why every fast file-IO library uses mmap; how COW underlies `fork()`],
  [cpu-affinity], [How to pin a thread to a core and keep the kernel's hands off it],
  [scheduler], [Why CFS picks the task it does; when to use `SCHED_DEADLINE`],
  [cgroups-namespaces], [What a "container" actually is at the kernel level],
  [interrupts], [Why high-PPS networking can't use one IRQ per packet],
  [kernel-tracing], [How to attach an eBPF probe to any kernel function in production],
  [kernel-modules], [How to ship code that runs in ring 0],
)

By the end you will be able to read kernel source, write a working kernel module, diagnose a production latency spike, and tune a system for a real-time or low-latency workload.

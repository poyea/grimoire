= Kernel Tracing

Modern Linux is the most observable kernel ever shipped. You can attach a probe to almost any kernel function, sample on hardware events, trace every syscall, watch every block-IO completion, all in production, with overhead measured in single-digit percentage points or less. This chapter covers the four major tracing mechanisms — kprobes, ftrace, perf, and eBPF — and the tools built on them.

== The Tracing Stack

Five layers, increasingly powerful and increasingly safe:

#table(columns: (auto, 1fr),
  [*Layer*], [*What it gives you*],
  [*Tracepoints*], [Static instrumentation points compiled into the kernel. Stable ABI.],
  [*kprobes / kretprobes*], [Dynamic probe at any kernel instruction; fires on entry/return.],
  [*uprobes / uretprobes*], [Same, but in user-space binaries.],
  [*ftrace*], [In-kernel ring buffer + function tracer + tracepoint frontend.],
  [*perf_event_open*], [Unified syscall covering hardware counters, software events, tracepoints, and probes.],
  [*eBPF*], [Programmable, verified bytecode you attach to any of the above.],
)

eBPF (Extended Berkeley Packet Filter, despite the name unrelated to packet filtering at this point) is the dominant interface. Modern tools — `bpftrace`, BCC, `bpftop`, `pixie`, `cilium` — all generate eBPF programs and load them via `bpf()` syscall.

== Tracepoints

Static probe sites placed by kernel developers at semantically meaningful locations. Stable across kernel versions (the implementer is committing to keep them).

```
$ ls /sys/kernel/debug/tracing/events
block       cgroup     filemap    irq        ksm        ...
clk         compaction filelock   jbd2       msr        ...
cpufreq     ...
```

Each tracepoint has a known set of arguments, defined by `TRACE_EVENT()` in the kernel source. List arguments:

```
$ cat /sys/kernel/debug/tracing/events/sched/sched_switch/format
name: sched_switch
ID: 287
format:
    field:char prev_comm[16];          offset:8;  size:16; signed:1;
    field:pid_t prev_pid;              offset:24; size:4;  signed:1;
    field:int prev_prio;               offset:28; size:4;  signed:1;
    field:long prev_state;             offset:32; size:8;  signed:1;
    field:char next_comm[16];          offset:40; size:16; signed:1;
    field:pid_t next_pid;              offset:56; size:4;  signed:1;
    field:int next_prio;               offset:60; size:4;  signed:1;
```

Tracepoints have near-zero overhead when no probe is attached — they compile to a `nop`. When attached, they call into the registered handler.

== kprobes and kretprobes

Dynamic probes, placed at runtime on any kernel instruction. Implementation: the kernel patches in a breakpoint (`int3` on x86), the breakpoint handler invokes the probe handler, the original instruction is single-stepped out-of-line, and execution resumes.

```c
#include <linux/kprobes.h>

static int handler_pre(struct kprobe *p, struct pt_regs *regs) {
    pr_info("called %pS, rdi=%lx\n", (void *)regs->ip, regs->di);
    return 0;
}

struct kprobe kp = {
    .symbol_name = "do_sys_open",
    .pre_handler = handler_pre,
};
register_kprobe(&kp);
```

Overhead is ~1-2 µs per probe hit (a real CPU exception is involved). Modern eBPF on tracepoints uses a much cheaper path.

`kretprobe` lets you fire on function *return* — useful for measuring duration. The kernel rewrites the return address on entry to a trampoline that captures the return value and re-redirects to the real caller.

*Caveats:*

- Inlined functions can't be probed (no symbol).
- Some functions are blacklisted (probing them would deadlock — e.g. the kprobe machinery itself).
- KASLR randomizes function addresses; probe by *symbol name* not address.

== ftrace

In-tree tracer with a virtual-filesystem interface (`/sys/kernel/debug/tracing/`).

*Function tracer:*

```
cd /sys/kernel/debug/tracing
echo function > current_tracer
echo do_sys_open > set_ftrace_filter
echo 1 > tracing_on
sleep 1
echo 0 > tracing_on
cat trace
```

*Function graph tracer* (with timing):

```
echo function_graph > current_tracer
echo 'tcp_*' > set_graph_function
cat trace
```

Output:

```
 0)               |  tcp_sendmsg() {
 0)               |    lock_sock_nested() {
 0)   0.342 us    |      _raw_spin_lock_bh();
 0)   0.658 us    |    }
 0)               |    sk_stream_alloc_skb() {
 0)   0.541 us    |      __alloc_skb();
 0)   1.024 us    |    }
 0)   3.418 us    |  }
```

Powerful but verbose. `trace-cmd` (`apt install trace-cmd`) and `kernelshark` (GUI) are more pleasant frontends.

*Other tracers:* `wakeup` (max scheduling latency), `irqsoff` (max IRQ-disabled duration), `preemptoff`, `hwlat` (system management mode latency detector).

== perf

`perf` is the swiss army knife. It wraps `perf_event_open(2)`, which exposes:

- *Hardware events:* `cycles`, `instructions`, `cache-misses`, `branch-misses`, `LLC-loads`, etc. (Per-CPU PMU counters.)
- *Software events:* `cpu-clock`, `task-clock`, `context-switches`, `cpu-migrations`, `page-faults`, `major-faults`, `minor-faults`.
- *Tracepoints:* every `/sys/kernel/debug/tracing/events/` is a perf event.
- *kprobes / uprobes:* dynamically registered.

```
perf list                                        # what's available on this box
perf stat -e cycles,instructions ./prog
perf stat -e cache-misses,cache-references ./prog
perf stat -p <pid> sleep 5

perf record -F 99 -a -g -- sleep 30              # 99 Hz, all CPUs, with call graphs
perf report                                       # interactive viewer
perf script | flamegraph.pl > flame.svg          # Brendan Gregg's flame graphs

perf top -g                                      # live system-wide profile
perf trace -p <pid>                              # like strace but cheaper
perf sched record -- ./prog
perf sched latency
```

*The PMU counter cheat sheet:*

```
IPC (instructions per cycle):   instructions / cycles
Branch misprediction rate:      branch-misses / branches
Cache miss rate:                cache-misses / cache-references
LLC miss rate:                  LLC-load-misses / LLC-loads
Frontend stall:                 cycle_activity.stalls_total / cycles  (TMAM)
```

For deeper PMU work see _cpu-architecture/performance-analysis.typ_ — TMAM, top-down methodology, microarchitectural stalls.

== eBPF: The Modern Tracing Substrate

eBPF programs are kernel-loadable, kernel-verified, kernel-JITted bytecode. They attach to tracepoints, kprobes, uprobes, perf events, network hooks (XDP/TC), LSM hooks, scheduler hooks (`sched_ext`), and more.

The verifier guarantees safety: bounded loops, no out-of-bounds memory access, no unreachable code, no recursion deeper than a fixed limit. *A misbehaving eBPF program cannot crash the kernel.* This is why production systems run eBPF in the hot path with confidence.

*Maps* are kernel-resident data structures (hash, array, ringbuf, perf buffer, LRU, BPF queue/stack) that programs use to store state and communicate with userspace.

*Helpers* are a fixed set of kernel functions an eBPF program can call: `bpf_ktime_get_ns`, `bpf_get_current_pid_tgid`, `bpf_probe_read_kernel`, `bpf_map_update_elem`, etc. 300+ helpers are exposed on a current 6.x kernel, and the set grows each release. Newer programs increasingly use *kfuncs* — directly callable kernel functions exposed via BTF — instead of helpers.

Writing eBPF directly (libbpf + CO-RE — Compile Once, Run Everywhere — using BTF for type info) is the production approach. For exploration, use `bpftrace`.

=== bpftrace One-Liners

```
# Syscall counts per process
bpftrace -e 'tracepoint:raw_syscalls:sys_enter { @[comm] = count(); }'

# Read-syscall latency histogram
bpftrace -e '
  tracepoint:syscalls:sys_enter_read { @start[tid] = nsecs; }
  tracepoint:syscalls:sys_exit_read /@start[tid]/ {
      @lat = hist(nsecs - @start[tid]);
      delete(@start[tid]);
  }
'

# TCP retransmits
bpftrace -e 'kprobe:tcp_retransmit_skb { @[kstack] = count(); }'

# Page-fault rate per process
bpftrace -e 'software:page-faults:1 { @[comm] = count(); }'

# Off-CPU time (where threads block)
bpftrace -e '
  kprobe:finish_task_switch { @start[arg0] = nsecs; }
  kprobe:try_to_wake_up
    /@start[arg0]/ {
      @off[kstack(arg0)] = sum(nsecs - @start[arg0]);
      delete(@start[arg0]);
    }
'
```

The language is awk-like — global maps with associative-array syntax, predicate filters with `/.../`, several histogram and aggregation operators (`count`, `sum`, `avg`, `hist`, `lhist`, `stats`).

=== BCC

A Python-based eBPF toolkit. Each tool is a small Python program that compiles a C eBPF program inline, loads it, and prints the results.

Production-grade tools shipped with BCC (`apt install bpfcc-tools`):

#table(columns: (auto, 1fr),
  [*Tool*], [*What it does*],
  [`execsnoop`], [Trace `execve()` system-wide. Reveals every command spawned.],
  [`opensnoop`], [Trace `open()` system-wide. What files are being touched.],
  [`biolatency`], [Block-IO latency histogram. Per-disk, per-IO-type.],
  [`tcplife`], [Per-connection lifetime, bytes, latency.],
  [`tcpconnect`/`tcpaccept`], [New connections, with PID, command, addresses.],
  [`runqlat`], [Run-queue latency histogram — how long tasks wait between wakeup and CPU.],
  [`offcputime`], [Where threads block, and for how long.],
  [`profile`], [On-CPU sampling profiler with stack traces.],
  [`funccount`], [Count calls to any kernel function pattern.],
  [`fileslower`], [VFS reads/writes slower than threshold.],
  [`oomkill`], [Trace OOM-killer invocations with context.],
)

Production usage pattern: `runqlat` first when you suspect scheduling, `biolatency` when you suspect storage, `offcputime` when you suspect blocking, `profile` when you suspect CPU bound.

=== libbpf + CO-RE

The production way to ship eBPF: write a C program, compile to BPF bytecode with `clang -target bpf`, link against `libbpf`. *CO-RE* uses BTF (BPF Type Format, embedded in the kernel image) so the same compiled program runs against many kernels without recompilation — the loader rewrites struct-field offsets at load time.

```c
SEC("kprobe/vfs_read")
int BPF_KPROBE(do_vfs_read, struct file *file, char *buf, size_t count) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    bpf_map_update_elem(&counts, &pid, &count, BPF_ANY);
    return 0;
}
```

`SEC` macros tell `libbpf` how to attach. `BPF_KPROBE` is a CO-RE-friendly argument-extraction wrapper.

This is what `cilium`, `falco`, `parca`, and most observability vendors actually ship.

== Performance Considerations

#table(columns: (auto, 1fr),
  [*Mechanism*], [*Cost per event*],
  [Tracepoint, no probe], [~0 ns],
  [Tracepoint, eBPF probe attached], [~30-100 ns],
  [Tracepoint, ftrace ring-buffer write], [~100-200 ns],
  [kprobe, no eBPF], [~1-2 µs (int3 + handler)],
  [kprobe, eBPF (jit'd)], [~150-300 ns],
  [perf record, sampled], [overhead scales with rate; 99 Hz system-wide is ~0.5%],
)

For high-rate events (every TCP packet, every syscall on a busy system), prefer tracepoints with eBPF over kprobes. eBPF `fentry`/`fexit` (kernel 5.5+, production-ready with BTF/CO-RE in 5.8+) is also faster than kprobes for new attachments.

== Production-Safe Tracing Workflow

1. Hypothesize: where is the latency? Storage, scheduling, network, lock contention, code path?
2. Pick the tool: `runqlat` for scheduling, `biolatency` for IO, `offcputime` for blocking, `profile` for CPU.
3. Run it for *seconds, not minutes*. Most BCC tools report on Ctrl-C.
4. If a culprit emerges, narrow with a targeted bpftrace one-liner.
5. Confirm the fix with the same measurement.

If you can't reproduce in production, you can't debug it. eBPF is precisely the tool that makes "leave it running and capture when it happens" practical.

== Diagnostics Reference

```
ls /sys/kernel/debug/tracing/events                # tracepoint catalog
perf list                                          # all perf events
bpftool prog list                                  # currently-loaded eBPF programs
bpftool map list                                   # eBPF maps
cat /sys/kernel/debug/tracing/available_filter_functions | wc -l   # kprobeable funcs
```

If `bpftrace` complains about kheaders or BTF, install `linux-headers-$(uname -r)` and `linux-image-$(uname -r)-dbg` (or distro equivalent). Most modern kernels (5.4+) embed BTF and don't require headers.

*See also:* _cpu-architecture/performance-analysis.typ_ (PMU counters, TMAM), _networking/nat-firewalls.typ_ (eBPF/XDP packet filtering — the original BPF use case), _scheduler.typ_ (sched_ext lets eBPF *implement* the scheduler, not just observe it).

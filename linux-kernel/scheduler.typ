= The Scheduler

The Linux scheduler decides which runnable task gets the CPU next. For most workloads its default class — CFS, the Completely Fair Scheduler — is good enough that you don't think about it. For latency-sensitive, real-time, or contended systems, understanding what it is doing is the difference between a smooth system and a system with mysterious tail latency.

Since Linux 6.6, the fair class has been progressively reworked around *EEVDF* (Earliest Eligible Virtual Deadline First), a related but more deterministic algorithm. EEVDF is now the in-tree implementation of `SCHED_OTHER`; the user-facing class names and tunables are largely unchanged. The big-picture mental model — virtual runtime accounting on a per-CPU red-black tree — survives. We start with CFS as the canonical model and call out where EEVDF differs.

== Scheduling Classes

Every task is in exactly one *scheduling class*. Classes are ordered by priority; a runnable task in a higher class always preempts a lower one.

#table(columns: (auto, auto, 1fr),
  [*Class*], [*Policy*], [*Use case*],
  [`stop_sched_class`], [internal], [Highest. Used for migrate-task-now operations. Not user-callable.],
  [`dl_sched_class`], [`SCHED_DEADLINE`], [Hard real-time, EDF-scheduled, with budget enforcement.],
  [`rt_sched_class`], [`SCHED_FIFO`, `SCHED_RR`], [Soft real-time, fixed priority 1-99.],
  [`fair_sched_class`], [`SCHED_OTHER`, `SCHED_BATCH`, `SCHED_IDLE`], [The CFS / EEVDF default for ~all tasks.],
  [`idle_sched_class`], [internal], [The idle thread; runs when nothing else is runnable.],
)

A `SCHED_FIFO` task at priority 50 will *always* preempt any `SCHED_OTHER` task. A `SCHED_DEADLINE` task with budget remaining will preempt that `SCHED_FIFO` task. This strict preemption ordering is the foundation of Linux's real-time story.

Set the class with `sched_setscheduler` (process-wide) or `pthread_setschedparam` (per-thread):

```c
struct sched_param p = { .sched_priority = 50 };
sched_setscheduler(0, SCHED_FIFO, &p);
```

Setting RT classes typically requires `CAP_SYS_NICE` or a non-default `RLIMIT_RTPRIO`.

== CFS Internals

CFS picks the runnable task with the smallest *vruntime* (virtual runtime). vruntime is the task's accumulated CPU time, scaled by its weight (derived from `nice`).

```
vruntime += delta_exec * NICE_0_WEIGHT / task_weight
```

where `delta_exec` is the wall-clock CPU time just consumed. A `nice 0` task accumulates vruntime at wall-clock speed; a `nice -20` task accumulates ~10× slower (gets ~10× more CPU); `nice 19` ~10× faster.

*Data structure:* a per-CPU red-black tree of runnable tasks keyed by vruntime. The leftmost node is the next task to run. `O(log n)` insert/remove, `O(1)` peek-leftmost (via cached pointer).

*Time slice:* CFS doesn't have fixed slices. The "minimum granularity" (`/proc/sys/kernel/sched_min_granularity_ns`, default 0.75 ms on most distros, but tuneable) caps how short a slice can get when many tasks are runnable. The "scheduling period" (`sched_latency_ns`, default 6 ms) is the target window within which every runnable task gets a turn — divided up by their weights.

*Vruntime catch-up on wakeup:* a task that just woke from sleep would have very low vruntime (didn't accumulate while sleeping) and would otherwise dominate the CPU. CFS clamps wake-up vruntime to `min_vruntime - sched_latency`, capping the "credit" a sleeping task can earn.

*Load balancing:* periodically, the scheduler walks scheduling domains (a hierarchical view: SMT siblings → physical core → NUMA node → system) and tries to even out load by pulling tasks from busy CPUs to idle ones. The domain hierarchy is built from CPU topology at boot.

`CONFIG_SCHED_DEBUG=y` exposes `/proc/sched_debug` — every CPU's runqueue, every task's vruntime, every domain's load. Indispensable for diagnosing scheduling issues.

== EEVDF (Linux 6.6+)

EEVDF replaces "smallest vruntime" with "earliest virtual deadline among eligible tasks." Each task gets:

- A *virtual eligibility time* (when it becomes eligible to run).
- A *virtual deadline* = eligibility + (request_size / weight).

The scheduler picks the eligible task with the smallest deadline. The result is more deterministic latency — a task with a small request size gets a small deadline and runs sooner — and a single tunable, `sched_base_slice_ns`, replaces the latency/granularity pair.

Practically: existing CFS tunables are partly deprecated, the user-visible behavior is similar to CFS but with smoother latency under load, and `nice`/`SCHED_BATCH`/`SCHED_IDLE` continue to work as before.

== SCHED_BATCH and SCHED_IDLE

Two sub-policies of the fair class:

- *`SCHED_BATCH`* — like `SCHED_OTHER` but the scheduler treats the task as non-interactive. No wake-up vruntime credit, longer slices in practice. Good for build jobs, long-running compute.
- *`SCHED_IDLE`* — extremely low priority within the fair class (effectively `nice 19` and then some). Only runs when nothing else wants the CPU. Good for housekeeping, very-low-priority backups.

```c
struct sched_param p = { .sched_priority = 0 };           // unused for SCHED_BATCH
sched_setscheduler(0, SCHED_BATCH, &p);
```

== SCHED_FIFO and SCHED_RR

The classic POSIX real-time policies. Fixed priority 1-99 (higher = higher priority). Within a priority level:

- *`SCHED_FIFO`* — runs until it blocks, exits, or a higher-priority task preempts. No timeslice. A `SCHED_FIFO` task in an infinite loop will hold the CPU forever.
- *`SCHED_RR`* — runs for `/proc/sys/kernel/sched_rr_timeslice_ms` (default 100 ms), then yields to the next task at the same priority. Above its priority is still preemptive.

```c
struct sched_param p = { .sched_priority = 50 };
sched_setscheduler(0, SCHED_FIFO, &p);
```

*Throttling guard:* if `SCHED_FIFO`/`RR` tasks would otherwise starve the rest of the system, the kernel enforces a cap. By default, RT tasks get at most 95% of any 1-second window; the remaining 5% goes to lower classes. Tuned via:

```
/proc/sys/kernel/sched_rt_runtime_us    # default 950000
/proc/sys/kernel/sched_rt_period_us     # default 1000000
```

Set `sched_rt_runtime_us` to `-1` to disable the throttle (production use should be very deliberate about this — a runaway RT task can lock the system).

*Real-time recipe (latency-critical thread):*

```c
// Lock all current and future memory in RAM.
mlockall(MCL_CURRENT | MCL_FUTURE);

// Pin to an isolated core.
cpu_set_t set; CPU_ZERO(&set); CPU_SET(2, &set);
sched_setaffinity(0, sizeof(set), &set);

// SCHED_FIFO at priority 50.
struct sched_param p = { .sched_priority = 50 };
sched_setscheduler(0, SCHED_FIFO, &p);
```

Combined with `isolcpus=2 nohz_full=2 rcu_nocbs=2` on the kernel command line, this thread will not be preempted by anything except a higher-priority RT task or a hardware IRQ.

== SCHED_DEADLINE: EDF on Linux

`SCHED_DEADLINE` (since 3.14) is the only Linux scheduling class with a *budget* and a *deadline*. It implements *Earliest Deadline First* (EDF) scheduling on top of *Constant Bandwidth Server* (CBS) admission control.

You declare three numbers per task:

#table(columns: (auto, 1fr),
  [*runtime*], [How much CPU time the task needs per period (ns).],
  [*deadline*], [By when, relative to period start, the runtime must complete (ns). ≤ period.],
  [*period*], [How often the task is released (ns).],
)

Example: a video pipeline that needs 5 ms of CPU every 33 ms (30 Hz frame rate), with a deadline of 30 ms (some leeway):

```c
#include <linux/sched/types.h>
struct sched_attr {
    uint32_t size;
    uint32_t sched_policy;
    uint64_t sched_flags;
    int32_t  sched_nice;
    uint32_t sched_priority;
    uint64_t sched_runtime;
    uint64_t sched_deadline;
    uint64_t sched_period;
};

struct sched_attr a = {
    .size           = sizeof(a),
    .sched_policy   = SCHED_DEADLINE,
    .sched_runtime  =  5 * 1000000ULL,    //  5 ms
    .sched_deadline = 30 * 1000000ULL,    // 30 ms
    .sched_period   = 33 * 1000000ULL,    // 33 ms
};
syscall(SYS_sched_setattr, 0, &a, 0);
```

The kernel runs *admission control* on `sched_setattr`. The total CPU bandwidth promised across all `SCHED_DEADLINE` tasks (sum of `runtime/period`) must not exceed the system's available bandwidth (95% by default, controlled by `sched_rt_period_us` / `sched_rt_runtime_us`). If the new task would over-commit, `sched_setattr` returns `-EBUSY`.

*Runtime enforcement:* if a task overruns its runtime, it is suspended until the next period. This is the property that lets EDF give *guarantees* — no single misbehaving task can starve others. Combined with admission control, this gives a deadline-meeting guarantee that `SCHED_FIFO` cannot.

*Affinity restrictions:* historically `SCHED_DEADLINE` did not allow CPU affinity (admission control assumed system-wide bandwidth). 6.x relaxed this with cgroup-cpuset partitions, but the rules are subtle: an exclusive cpuset partition is treated as a separate admission-control domain.

== Priority Inversion and PI Futexes

Classic problem: a low-priority task holds a lock; a high-priority task tries to acquire it; meanwhile a medium-priority task runs and starves the low-priority task, which never releases the lock, which blocks the high-priority task.

Linux's solution: *priority inheritance futexes* (PI futexes). When a high-priority task blocks on a PI futex, the lock holder temporarily inherits the high priority for the duration of the critical section.

`pthread_mutex_t` with PI:

```c
pthread_mutexattr_t a;
pthread_mutexattr_init(&a);
pthread_mutexattr_setprotocol(&a, PTHREAD_PRIO_INHERIT);
pthread_mutexattr_setpshared(&a, PTHREAD_PROCESS_PRIVATE);
pthread_mutex_init(&m, &a);
```

Internally, the lock uses `FUTEX_LOCK_PI` / `FUTEX_UNLOCK_PI` syscalls, which give the kernel enough information to track who holds what and bump priorities accordingly.

*Required for any RT design touching shared locks.* Without PI, a `SCHED_FIFO` task can deadlock the system if it ever tries to acquire a mutex held by a normal task that's been preempted.

== Watchdogs and Stall Detection

The kernel has a soft-lockup watchdog that fires if any task holds a CPU for too long without scheduling.

```
/proc/sys/kernel/watchdog              # 1 = enabled (default)
/proc/sys/kernel/watchdog_thresh        # seconds, default 10
/proc/sys/kernel/softlockup_panic       # 1 = panic on soft lockup
```

A `SCHED_FIFO` infinite loop will trip this, log a stack trace via `dmesg`, and (if `softlockup_panic=1`) crash the box. Helpful for catching real-time bugs in development.

The RCU stall detector is similar but watches for grace-period progress: `/proc/sys/kernel/rcu_cpu_stall_timeout` (default 21 s).

== Scheduler Diagnostics

```
chrt -p <pid>                                        # show class and priority
chrt -f -p 50 <pid>                                  # set SCHED_FIFO 50
schedtool -e -2 ./prog                                # SCHED_IDLE
cat /proc/<pid>/sched                                 # detailed task scheduling stats
cat /proc/sched_debug                                 # everything (CONFIG_SCHED_DEBUG)
perf sched record -- ./prog
perf sched latency                                    # per-task latency report
perf sched timehist                                   # context-switch timeline
bpftrace -e 'tracepoint:sched:sched_wakeup { @[comm] = count(); }'
```

`runqlat` (BCC) gives a histogram of run-queue latency — how long tasks waited between wakeup and execution. If you have tail-latency outliers and `runqlat` shows a 50 ms tail, the scheduler is the culprit and you need to look at preemption, RT throttling, or load-balancer hesitation.

== sched_ext: BPF-Defined Schedulers

Linux 6.12 mainlines `sched_ext`, a framework for writing scheduling classes in BPF. It lets you replace the fair-class behavior with a custom policy — without recompiling the kernel — for experimentation or workload-specific tuning. Production-quality examples: `scx_rusty`, `scx_bpfland` (interactive), `scx_layered` (per-cgroup priorities).

```
modprobe sched_ext
scx_rusty                  # load and run a sched_ext-based scheduler
```

This is the most significant scheduling-architecture change in over a decade. For most systems, the default fair scheduler remains correct; for workloads with very specific patterns (e.g. tightly coupled HPC, gaming foreground/background), `sched_ext` lets you ship a tailored scheduler as user code.

*See also:* _cpu-affinity.typ_ (isolation, the necessary complement to RT scheduling), _interrupts.typ_ (IRQ jitter, threaded IRQs), _coding/advanced-systems.typ_ (scheduler theory: EDF, rate-monotonic), _cgroups-namespaces.typ_ (cpu controller, cpuset partitions).

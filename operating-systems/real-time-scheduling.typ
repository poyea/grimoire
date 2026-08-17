#import "../template.typ": xref

= Real-Time Scheduling

A real-time system is not a fast system; it is a *predictable* one. The defining question is not "how quickly does this usually finish?" but "can I prove it always finishes by its deadline?" That shift — from expectation to guarantee — produces a body of theory with actual theorems (utilization bounds, optimality proofs, blocking-time bounds) and a body of engineering (priority protocols, preemptible kernels, bandwidth servers) for making commodity OSes honor them. _Scheduling Theory_ surveys the general landscape; this chapter goes deep on the real-time half.

*See also:* #xref("operating-systems", "scheduling-theory", label: "Scheduling Theory"), #xref("operating-systems", "processes-and-threads", label: "Processes and Threads"), #xref("operating-systems", "ipc-mechanisms", label: "Inter-Process Communication"), #xref("linux-kernel", "scheduler", label: "The Scheduler") (linux-kernel), #xref("linux-kernel", "interrupts", label: "Interrupts and Bottom Halves") (linux-kernel).

== Hard, Firm, and Soft

The taxonomy is about the *value of a late result*:

- *Hard real time*: a missed deadline is a system failure — the result is worthless or harmful. Airbag deployment, engine control, pacemakers. Correctness requires proof over the worst case.
- *Firm real time*: late results are worthless but not catastrophic; occasional misses are tolerable if bounded. Video frame decoding — a late frame is dropped, not displayed.
- *Soft real time*: late results lose value gradually. Audio pipelines, trading systems, telecom. Engineering targets a percentile (e.g., 99.999% of deadlines met) rather than the absolute worst case.

The standard task model: task $tau_i$ releases a job every period $T_i$, each job needs at most $C_i$ of CPU (the *worst-case execution time*, WCET) and must finish within relative deadline $D_i$ (often $D_i = T_i$). Utilization is $U_i = C_i \/ T_i$. Everything downstream — schedulability tests, admission control — is only as sound as the WCET estimate, and WCET analysis on modern hardware (caches, branch predictors, DRAM contention) is its own hard discipline; pessimistic measurement plus margin is the industrial norm.

== Rate Monotonic Scheduling

*Rate Monotonic* (RM) assigns static priorities by period: shorter period, higher priority. Liu and Layland (1973) proved two foundational results for independent periodic tasks with $D_i = T_i$ on a uniprocessor:

1. RM is *optimal among fixed-priority policies*: if any fixed-priority assignment can schedule a task set, RM can.
2. A task set is schedulable under RM if

$ U = sum_(i=1)^n C_i / T_i <= n(2^(1\/n) - 1) $

The bound decreases from 1.0 ($n = 1$) through 0.828 ($n = 2$) toward its limit $ln 2 approx 0.693$ as $n -> infinity$. The reading: keep fixed-priority utilization below about 69% and schedulability is *guaranteed*, no further analysis needed.

The bound is sufficient, not necessary — many task sets above it are fine (harmonic periods, where each period divides the next, are schedulable up to $U = 1$). When $D_i != T_i$, *deadline monotonic* (priority by deadline) takes over as the optimal fixed-priority assignment.

== Earliest Deadline First

*EDF* assigns priority dynamically: the job with the nearest absolute deadline runs. Liu and Layland's companion result: EDF is *optimal on a uniprocessor* — a periodic task set with $D_i = T_i$ is schedulable iff

$ U = sum_(i=1)^n C_i / T_i <= 1 $

Full utilization, against RM's 69%. The trade-offs that keep RM alive anyway:

- *Overload behavior*: under transient overload RM degrades predictably — the longest-period (lowest-priority) tasks miss, everything above survives. EDF has no such ordering; an overrun shifts every subsequent deadline and misses cascade unpredictably (the *domino effect*).
- *Implementation*: RM is a static priority table; EDF needs a deadline-ordered queue and timestamp arithmetic. (This argument has weakened — Linux ships EDF.)
- *Certification culture*: avionics standards and decades of RMA tooling are built around fixed priorities.

== Response-Time Analysis

Utilization bounds are blunt; *response-time analysis* (RTA) is exact for fixed priorities. The worst case for task $tau_i$ occurs at a *critical instant* — all higher-priority tasks released simultaneously with it. Its worst-case response time $R_i$ satisfies the recurrence

$ R_i = C_i + B_i + sum_(j in "hp"(i)) ceil(R_i / T_j) C_j $

where $"hp"(i)$ is the set of higher-priority tasks and $B_i$ is the blocking term (below). Each ceiling counts how many times $tau_j$ preempts during the window. Solve by fixed-point iteration starting from $R_i = C_i$; the task set is schedulable iff $R_i <= D_i$ for all $i$, and iteration beyond $D_i$ proves a miss. RTA is necessary-and-sufficient for this model, handles $D_i < T_i$, and extends to release jitter and context-switch overheads — it is the workhorse of industrial schedulability tools.

== Priority Inversion

*Priority inversion*: a high-priority task blocks on a resource held by a low-priority task, and a *medium*-priority task — needing no resource at all — preempts the holder. The high-priority task is now effectively running at the lowest priority in the system, for an unbounded time.

The canonical incident is *Mars Pathfinder* (1997). On the VxWorks-based lander, a low-priority meteorological task held a mutex on the information bus; the high-priority bus-management task blocked on it; long-running medium-priority communication work preempted the holder. The bus task missed its activation, a watchdog declared the system hung, and the spacecraft reset — repeatedly, on Mars. The fix was uploaded from Earth: flip one flag to enable *priority inheritance* on that mutex. (JPL's Glenn Reeves' postmortem is a classic; the bug had appeared in pre-launch testing and been deprioritized as a rare anomaly.)

The protocols:

- *Priority Inheritance Protocol* (PIP): a task holding a lock executes at the maximum priority of the tasks blocked on it; inheritance is transitive through chains. Blocking becomes bounded — but a task can block once per lock it needs (*chained blocking*), and PIP does not prevent deadlock.
- *Priority Ceiling Protocol* (PCP): each lock carries a *ceiling*, the highest priority of any task that ever takes it; a task may acquire a lock only if its priority exceeds the ceilings of all locks currently held by others. Result: at most *one* critical section of blocking per job, and deadlock is impossible by construction.
- *Immediate ceiling* (the variant in Ada and POSIX `PTHREAD_PRIO_PROTECT`): raise the holder to the ceiling at acquisition time — same bounds, trivial implementation, no blocking-queue bookkeeping.

The blocking term these protocols bound is exactly the $B_i$ in response-time analysis; without a protocol, $B_i$ is unbounded and no analysis is possible.

== Jitter and Latency Sources

Even a correct schedule executes on a machine that resists determinism. *Release jitter* — variation between a job's nominal release and when it actually starts — comes from timer granularity, interrupt latency, and scheduling latency; it adds directly into response-time analysis and matters acutely for control loops, where jitter in sampling time degrades controller stability independent of deadline misses.

The latency stack on a commodity OS: interrupt disabled sections (worst-case `local_irq_disable` window), interrupt handler durations, non-preemptible kernel sections, the scheduler's own dispatch cost, and below the OS entirely — system management interrupts (SMIs) that steal the CPU invisibly, CPU frequency transitions, and deep idle-state exit latencies. Hardware tuning (disabling SMIs and deep C-states, pinning frequency) is as much a part of real-time engineering as the scheduler.

== Real Time on Linux

=== PREEMPT_RT

Stock Linux kernels have historically allowed long non-preemptible sections; *PREEMPT_RT* (mainlined incrementally, fully merged in 6.12) converts the kernel into a preemptible real-time substrate: spinlocks become priority-inheriting `rt_mutex`es, interrupt handlers run as schedulable kernel threads (so a `SCHED_FIFO` task outranks most device IRQ work), and preemption is possible almost everywhere. Result: worst-case scheduling latencies drop from milliseconds to tens of microseconds on tuned hardware — firm/soft real time on commodity silicon, with hard-real-time use in industrial control where the latency budget tolerates it.

The classic POSIX policies ride on top: `SCHED_FIFO` (fixed priority, run until block or preemption by higher priority) and `SCHED_RR` (FIFO plus round-robin within a priority level). Both implement exactly the fixed-priority model that RM analysis assumes — assign priorities rate-monotonically and the theory applies, *if* you control every task on the relevant CPUs. The standard guardrail `sched_rt_runtime_us` (default: RT tasks get at most 950 ms per second) exists because a runaway `SCHED_FIFO` task otherwise owns the CPU forever.

=== SCHED_DEADLINE and CBS

`SCHED_DEADLINE` (Linux 3.14+) implements EDF with per-task reservations: each task declares `(runtime, deadline, period)` and the kernel's admission test rejects task sets that would exceed capacity. Enforcement is the *Constant Bandwidth Server* (CBS): a task that exhausts its runtime within the current period is throttled until its next replenishment, so an overrunning task consumes only its own reservation — the domino effect is contained by construction, and a misbehaving task cannot starve others no matter its deadline. This converts EDF from "optimal but fragile" into "optimal with temporal isolation," the same isolation philosophy as cgroup memory limits applied to time. Multi-core `SCHED_DEADLINE` is global EDF with all its theoretical caveats (Dhall's effect — see _Scheduling Theory_); partitioning via affinity restores per-CPU analysis at the cost of bin-packing.

== Mixed Criticality

Modern platforms consolidate functions of different criticality onto shared silicon — flight control beside cabin entertainment. Certification demands pessimistic WCETs for critical tasks; provisioning for that pessimism permanently wastes the capacity the pessimism almost never uses. Vestal's mixed-criticality model (2007) resolves the tension with per-task WCET estimates *at multiple assurance levels*: in normal (LO) mode all tasks run against optimistic budgets; if any critical task overruns its LO budget, the system switches to HI mode, guaranteeing critical tasks their pessimistic budgets and dropping or degrading the rest. Schedulability is proven separately for each mode plus the transition. Adaptive mixed criticality (AMC) is the standard fixed-priority realization; the contested practical question is how and when low-criticality work is restored after a HI excursion. The same instinct — strong isolation between criticality domains — drives hypervisor-based separation (ARINC 653 partitions, Jailhouse) when shared-kernel isolation is not certifiable.

== Pitfalls

- *Optimistic WCET*: every guarantee downstream of a wrong $C_i$ is fiction; measure on the target hardware with caches cold and interference present, then add margin.
- *Unbounded priority inversion*: any mutex shared between RT and non-RT code without inheritance recreates Pathfinder; on Linux, that means `PTHREAD_PRIO_INHERIT`, and beware hidden shared locks inside `malloc`, logging, and the kernel itself (non-RT kernels).
- *`SCHED_FIFO` priority 99 for everything*: fixed priorities only mean something as a *relative* ordering derived from periods; uniform maximum priority is FIFO with extra danger.
- *Forgetting the other tenants*: IRQ threads, kernel housekeeping, and SMIs share the CPU; `isolcpus`/`nohz_full` plus IRQ affinity move them aside, but SMIs answer to firmware alone.
- *Testing the average*: real-time failures live in the worst case; validate with `cyclictest` under adversarial load (memory pressure, cache thrashing, interrupt storms), not an idle system.

== Further Reading

Liu, C., Layland, J. (1973). "Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment." JACM.

Joseph, M., Pandya, P. (1986). "Finding Response Times in a Real-Time System." The Computer Journal (response-time analysis).

Sha, L., Rajkumar, R., Lehoczky, J. (1990). "Priority Inheritance Protocols: An Approach to Real-Time Synchronization." IEEE Transactions on Computers.

Reeves, G. (1997). "What Really Happened on Mars?" JPL postmortem correspondence on the Pathfinder priority inversion.

Abeni, L., Buttazzo, G. (1998). "Integrating Multimedia Applications in Hard Real-Time Systems." RTSS (the Constant Bandwidth Server).

Vestal, S. (2007). "Preemptive Scheduling of Multi-criticality Systems with Varying Degrees of Execution Time Assurance." RTSS.

Burns, A., Davis, R. (2017). "A Survey of Research into Mixed Criticality Systems." ACM Computing Surveys.

Buttazzo, G. (2011). "Hard Real-Time Computing Systems: Predictable Scheduling Algorithms and Applications." Springer, 3rd ed.

Lelli, J. et al. (2016). "Deadline Scheduling in the Linux Kernel." Software: Practice and Experience.

Rostedt, S., Hart, D. (2007). "Internals of the RT Patch." Ottawa Linux Symposium.

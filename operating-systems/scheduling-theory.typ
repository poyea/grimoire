= Scheduling Theory

CPU scheduling is the problem of dividing a scarce resource — execution time on $n$ cores — among $m$ contending threads under conflicting goals: throughput, latency, fairness, deadline satisfaction, energy. No single policy optimizes all of them, and the algorithmic literature is correspondingly rich. This chapter treats the theoretical landscape; Linux's CFS/EEVDF implementation is in `linux-kernel/scheduler.typ`.

*See also:* `operating-systems/processes-and-threads.typ`, `linux-kernel/scheduler.typ`, `linux-kernel/cpu-affinity.typ`.

== Workload Models and Metrics

A scheduling problem is defined by a workload model and an objective. Classical batch theory assumes a known job set with arrival time $a_i$, length $p_i$ (processing time), and possibly deadline $d_i$. Interactive and server workloads, by contrast, are *online* — arrivals are unknown — and lengths are unknown until the job completes (the *clairvoyance* assumption fails).

Common metrics:

#table(columns: (auto, 1fr),
  [*Metric*], [*Definition*],
  [Turnaround], [$C_i - a_i$ (completion minus arrival)],
  [Response], [first dispatch time $- a_i$],
  [Waiting], [time runnable but not running],
  [Throughput], [jobs/sec],
  [Slowdown], [$(C_i - a_i) \/ p_i$ — fairness-normalized turnaround],
  [Tail latency], [p99 / p999 response time],
  [Stretch], [worst-case slowdown across jobs],
)

In production systems p99 has eclipsed mean latency as the dominant metric: a request that fans out to 100 backends with i.i.d. p99 of 10 ms sees expected tail of ~50 ms. Dean & Barroso's "tail at scale" paper formalized this.

== Single-Processor Algorithms

*FIFO* (First-In-First-Out) is optimal for mean turnaround when all jobs are equal length, terrible otherwise: a long job blocks all behind it (*convoy effect*).

*SJF* (Shortest Job First) minimizes mean turnaround when job lengths are known — provably optimal. *SRTF* (Shortest Remaining Time First) is its preemptive cousin. Both starve long jobs and require clairvoyance.

*Round-robin* with quantum $q$ trades throughput (context switches scale with $1/q$) for fairness. As $q -> infinity$ it degenerates to FIFO; as $q -> 0$ it becomes processor sharing.

*Multilevel feedback queue* (MLFQ — CTSS, original Unix) approximates SJF without clairvoyance: jobs start at top priority, demote on quantum exhaustion, and periodic boosts prevent starvation. The two governing rules:
1. If a job uses its full quantum, demote it.
2. Periodically reset all jobs to top priority (anti-starvation).

*Fair-share / proportional share* generalizes fairness to weighted entitlements. Two formal approaches:

- *Lottery scheduling* (Waldspurger, 1994): each thread holds tickets proportional to its weight; the scheduler draws a random ticket each tick. Expected share matches weight; variance can be high.
- *Stride scheduling* (Waldspurger, 1995): deterministic dual — each thread has a *stride* inversely proportional to its weight, the scheduler picks the thread with minimum *pass* (cumulative stride). Eliminates variance.

*Virtual-time / WFQ* (Weighted Fair Queueing, originally from networks) generalizes further: each task accumulates virtual time $V_i = sum (delta_i \/ w_i)$ where $delta_i$ is CPU consumed and $w_i$ its weight; the scheduler picks minimum $V_i$. This is the conceptual foundation of CFS.

*EEVDF* (Earliest Eligible Virtual Deadline First, Stoica & Abdel-Wahab 1995) refines virtual-time scheduling with explicit eligibility (a task is eligible once its virtual time has caught up to a "service curve") and virtual deadlines. Provides both proportional fairness *and* bounded lag — the lateness of any task relative to its fair share is provably $O(1)$. Linux 6.6+ replaced CFS internals with EEVDF for exactly this latency guarantee.

== Real-Time Scheduling

A real-time task has a period $T$, worst-case execution time (WCET) $C$, and deadline $D$ (often $D = T$). A schedule is *feasible* iff every deadline is met.

*Rate Monotonic* (RM, Liu & Layland 1973): static priority equal to $1/T$ (shorter period $=>$ higher priority). For $n$ tasks the utilization bound is:

$ U = sum_{i=1}^n C_i / T_i <= n(2^(1/n) - 1) $

which approaches $ln 2 approx 0.693$ as $n -> infinity$. RM is optimal among static-priority policies for periodic tasks with $D = T$.

*Earliest Deadline First* (EDF): dynamic priority equal to absolute deadline. Optimal on uniprocessor; schedulable iff $U <= 1$. The catch: a single overrun cascades — *domino effect* — because every later deadline shifts.

*Deadline Monotonic* (DM): static priority by $D_i$; optimal static-priority policy when $D_i != T_i$.

*Constant Bandwidth Server* (CBS): wraps EDF with budget enforcement so overruns are bounded. The basis of Linux's `SCHED_DEADLINE`.

*Priority inversion* and the *priority inheritance protocol* (PIP) / *priority ceiling protocol* (PCP): when a low-priority task holds a mutex needed by a high-priority task, a medium-priority task can preempt the holder and indirectly block the high-priority task (the Mars Pathfinder bug). PIP temporarily boosts the holder's priority; PCP further bounds blocking time to one critical section.

== Multiprocessor Scheduling

The clean uniprocessor theory degrades sharply at $n > 1$. *Dhall's effect*: there exist task sets with $U$ arbitrarily close to 1 that are infeasible on $m > 1$ processors under EDF (a long high-utilization task dominates one CPU while shorter tasks pile up).

Three architectures:

*Partitioned*: bind each task to a CPU; reduce to $m$ independent uniprocessor problems. Bin-packing is NP-hard but heuristics (first-fit decreasing) work well. Cache-friendly but cannot exploit slack on other CPUs.

*Global*: a single ready queue; any CPU can run any task. Theoretically optimal (PFair, LRE-TL achieve $U <= m$) but suffers task migration overhead and cache misses; lock contention on the queue is the dominant cost at scale. Real systems shard.

*Clustered / semi-partitioned*: partition into clusters, balance globally within each. Linux's NUMA-aware load balancer is essentially this.

#table(columns: (auto, auto, auto, auto),
  [*Approach*], [*Util. bound*], [*Migration*], [*Cache*],
  [Partitioned], [<= m, hard to pack], [none], [hot],
  [Global EDF], [< m (Dhall)], [frequent], [cold],
  [PFair], [<= m optimal], [very frequent], [very cold],
  [Clustered], [<= m, depends], [bounded], [warm],
)

Co-scheduling (gang scheduling) is required for parallel applications whose threads communicate frequently — running half of an MPI rank's threads while siblings are descheduled produces O(quantum) lock-waiting stalls.

== Energy and Heterogeneous Schedulers

Big.LITTLE / Intel hybrid CPUs add the *which-core* question on top of *which-task*. *DVFS* (Dynamic Voltage and Frequency Scaling) trades performance for energy; the OS exposes governors (`ondemand`, `schedutil`) that bias toward energy-proportional execution.

Linux's *EAS* (Energy-Aware Scheduling) and Apple's GCD QoS classes both annotate threads with intent (foreground, background) and use a CPU energy model — predicted joules per task for placement decisions — to migrate work to small cores when latency budgets allow.

== Pitfalls

- *Priority inversion* without a protocol is silent until it isn't (Pathfinder).
- *Lock-holder preemption* in virtualized environments: a vCPU descheduled while holding a spinlock blocks every waiter. Paravirtual lock hints (`pv_qspinlock`) exist to mitigate.
- *Tail amplification* from fan-out: even fair schedulers produce bad p99 when one of many child tasks gets unlucky.
- *Schedutil load tracking* lag: PELT averages exponentially weighted over ~32 ms, so a bursty workload may be misfrequency-clocked for tens of ms.

== Further Reading

Liu, C., Layland, J. (1973). "Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment." JACM.

Waldspurger, C., Weihl, W. (1994). "Lottery Scheduling: Flexible Proportional-Share Resource Management." OSDI.

Waldspurger, C., Weihl, W. (1995). "Stride Scheduling." MIT/LCS/TM-528.

Stoica, I., Abdel-Wahab, H. (1995). "Earliest Eligible Virtual Deadline First: A Flexible and Accurate Mechanism for Proportional Share Resource Allocation." TR-95-22, Old Dominion.

Sha, L., Rajkumar, R., Lehoczky, J. (1990). "Priority Inheritance Protocols: An Approach to Real-Time Synchronization." IEEE TC.

Dhall, S., Liu, C. (1978). "On a Real-Time Scheduling Problem." Operations Research.

Baruah, S., Bertogna, M., Buttazzo, G. (2015). "Multiprocessor Scheduling for Real-Time Systems." Springer.

Dean, J., Barroso, L. (2013). "The Tail at Scale." CACM.

Ousterhout, J. (1982). "Scheduling Techniques for Concurrent Systems." ICDCS (coscheduling).

= Scheduling Theory

CPU scheduling is the problem of dividing a scarce resource (execution time on $n$ cores) among $m$ contending threads under conflicting goals: throughput, latency, fairness, deadline satisfaction, energy. No single policy optimizes all of them, and the algorithmic literature is correspondingly rich. This chapter treats the theoretical landscape; Linux's CFS/EEVDF implementation is in `linux-kernel/scheduler.typ`.

*See also:* _Processes and Threads_, _The Scheduler_ (implementation), _CPU Affinity, Isolation, and NUMA_ (implementation).

== Workload Models and Metrics

A scheduling problem is defined by a workload model and an objective. Classical batch theory assumes a known job set with arrival time $a_i$, length $p_i$ (processing time), and possibly deadline $d_i$. Interactive and server workloads, by contrast, are *online*: arrivals are unknown, and lengths are unknown until the job completes (the *clairvoyance* assumption fails).

Common metrics:

#table(columns: (auto, 1fr),
  [*Metric*], [*Definition*],
  [Turnaround], [$C_i - a_i$ (completion minus arrival)],
  [Response], [first dispatch time $- a_i$],
  [Waiting], [time runnable but not running],
  [Throughput], [jobs/sec],
  [Slowdown], [$(C_i - a_i) \/ p_i$, fairness-normalized turnaround],
  [Tail latency], [p99 / p999 response time],
  [Stretch], [worst-case slowdown across jobs],
)

In production systems p99 has eclipsed mean latency as the dominant metric: a request that fans out to 100 backends with i.i.d. p99 of 10 ms sees expected tail of ~50 ms. Dean & Barroso's "tail at scale" paper formalized this.

== Single-Processor Algorithms

*FIFO* (First-In-First-Out) is optimal for mean turnaround when all jobs are equal length, terrible otherwise: a long job blocks all behind it (*convoy effect*).

*SJF* (Shortest Job First) minimizes mean turnaround when job lengths are known; it is provably optimal. *SRTF* (Shortest Remaining Time First) is its preemptive cousin. Both starve long jobs and require clairvoyance.

*Round-robin* with quantum $q$ trades throughput (context switches scale with $1/q$) for fairness. As $q -> infinity$ it degenerates to FIFO; as $q -> 0$ it becomes processor sharing.

*Multilevel feedback queue* (MLFQ, as in CTSS and original Unix) approximates SJF without clairvoyance: jobs start at top priority, demote on quantum exhaustion, and periodic boosts prevent starvation. The two governing rules:
1. If a job uses its full quantum, demote it.
2. Periodically reset all jobs to top priority (anti-starvation).

*Fair-share / proportional share* generalizes fairness to weighted entitlements. Two formal approaches:

- *Lottery scheduling* (Waldspurger, 1994): each thread holds tickets proportional to its weight; the scheduler draws a random ticket each tick. Expected share matches weight; variance can be high.
- *Stride scheduling* (Waldspurger, 1995): the deterministic dual, where each thread has a *stride* inversely proportional to its weight and the scheduler picks the thread with minimum *pass* (cumulative stride). Eliminates variance.

*Virtual-time / WFQ* (Weighted Fair Queueing, originally from networks) generalizes further: each task accumulates virtual time $V_i = sum (delta_i \/ w_i)$ where $delta_i$ is CPU consumed and $w_i$ its weight; the scheduler picks minimum $V_i$. This is the conceptual foundation of CFS.

*EEVDF* (Earliest Eligible Virtual Deadline First, Stoica & Abdel-Wahab 1995) refines virtual-time scheduling with explicit eligibility (a task is eligible once its virtual time has caught up to a "service curve") and virtual deadlines. Provides both proportional fairness *and* bounded lag; the lateness of any task relative to its fair share is provably $O(1)$. Operationally, bounded lag means a task's actual CPU time received never falls more than one scheduling quantum behind its ideal fair share: EEVDF's virtual deadline mechanism ensures the task is scheduled before its lag exceeds one slice (typically 0.75–6 ms), giving hard lag bounds even under overload. By contrast, CFS has no such hard bound — lag can grow without limit during load spikes because CFS only targets minimum-vruntime without an eligibility deadline. Linux 6.6+ made EEVDF the default scheduler (replacing CFS as the active policy) for exactly this latency guarantee; the CFS compatibility layer remains.

== Real-Time Scheduling

A real-time task has a period $T$, worst-case execution time (WCET) $C$, and deadline $D$ (often $D = T$). A schedule is *feasible* iff every deadline is met.

*Rate Monotonic* (RM, Liu & Layland 1973): static priority equal to $1/T$ (shorter period $=>$ higher priority). For $n$ tasks the utilization bound is:

$ U = sum_(i=1)^n C_i / T_i <= n(2^(1/n) - 1) $

which approaches $ln 2 approx 0.693$ as $n -> infinity$. RM is optimal among static-priority policies for periodic tasks with $D = T$.

*Earliest Deadline First* (EDF): dynamic priority equal to absolute deadline. Optimal on uniprocessor; schedulable iff $U <= 1$. The catch: a single overrun cascades (the *domino effect*), because every later deadline shifts.

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

Co-scheduling (gang scheduling) is required for parallel applications whose threads communicate frequently: running half of an MPI rank's threads while siblings are descheduled produces O(quantum) lock-waiting stalls.

== Energy and Heterogeneous Schedulers

Big.LITTLE / Intel hybrid CPUs add the *which-core* question on top of *which-task*. *DVFS* (Dynamic Voltage and Frequency Scaling) trades performance for energy quadratically; halving voltage cuts dynamic power by ~4× at the expense of lower frequency. The OS exposes frequency governors (`ondemand`, `conservative`, `schedutil`) that adjust P-states based on utilization signals.

Linux's *EAS* (Energy-Aware Scheduling) extends the CFS load-balancer with an *energy model*: a per-CPU table mapping OPP (Operating Performance Point) to capacity and power draw (milliwatts). When deciding whether to migrate a task from a small core to a big core, the scheduler queries the model ("will this save energy net of the migration cost?") and prefers the placement with the lower predicted joule/op ratio. EAS is activated only on asymmetric topologies with a registered energy model; on symmetric SMP systems CFS's load-based balancer remains in effect.

Apple's Grand Central Dispatch *QoS classes* (`userInteractive`, `userInitiated`, `utility`, `background`) map directly to the P-core / E-core placement policy on Apple Silicon: background tasks run exclusively on E-cores unless CPU pressure forces promotion. The scheduler tracks *thermal headroom* from the System Management Controller (SMC) and throttles P-core boosts before the die temperature reaches a limit.

*Scheduling on heterogeneous CPUs raises three distinct problems:*

1. *Capacity asymmetry.* A runnable task on an E-core may complete at half the IPC of a P-core. The scheduler must express load as a normalized fraction of each core's capacity, not raw PELT utilization. Linux uses `arch_scale_cpu_capacity` to normalize.
2. *Cache topology.* P-cores and E-cores may share an LLC or have separate caches. Task migration between clusters forces a cache refill; EAS accounts for this via the `migration_cost` term.
3. *Thermal and power limits (TDP burst).* Modern CPUs can exceed their rated TDP for short bursts (Intel Turbo Boost, AMD Precision Boost). The scheduler cannot directly control TDP; it interacts with the platform via RAPL (Running Average Power Limit) counters and the cpufreq layer.

== Pitfalls

*Priority inversion* without a protocol is silent until it isn't (Pathfinder). The symptom is a high-priority task blocking indefinitely while a low-priority task holds a mutex, and a medium-priority task runs ahead of the low-priority holder — the high-priority task is effectively demoted to the lowest priority in the system without any log entry to that effect. The real incident: NASA's Mars Pathfinder (1997) experienced repeated system resets caused by a priority inversion between the meteorological task, the communication task, and the bus management task. The mitigation is Priority Inheritance Protocol (PIP): the OS temporarily elevates the holding thread to the priority of the highest waiter, ensuring the holder runs to completion quickly. Priority Ceiling Protocol (PCP) goes further, setting a ceiling so no inversion is possible from the moment the mutex is acquired.

*Lock-holder preemption* in virtualized environments: a vCPU descheduled while holding a spinlock blocks every waiter for the full preemption quantum (typically 1-10 ms), stalling an entire VM. The symptom is latency spikes correlated with CPU steal time and unexplained multi-millisecond delays on hot code paths under load. A concrete example: a guest kernel holding a scheduler spinlock gets preempted; all other vCPUs spin in the hypervisor, consuming real CPU while making no progress. Paravirtual lock hints (`pv_qspinlock` in Linux, `PV_UNHALT` on Xen) inform the hypervisor that a vCPU is spinning on a held lock, allowing it to immediately reschedule the holder rather than running the spinner.

*Tail amplification* from fan-out: even a fair scheduler produces bad p99 at the application level when a single request fans out to many child tasks or backend calls. If each child independently has a p99 of 10 ms and a request requires all $N$ children to complete, the expected maximum order statistic grows as $O(log N)$; for $N = 100$, the effective parent p99 approaches the p99.9 of the children. The mitigation combines deadline-aware scheduling (CBS / `SCHED_DEADLINE` for children), hedged requests (issuing a second request if the first is slow past a threshold), and aggressive per-child timeouts so slow children are abandoned before the parent deadline.

*Schedutil load tracking* lag: PELT (Per-Entity Load Tracking) computes utilization as an exponentially decaying sum of recent CPU time with a half-life of approximately 32 ms, so a workload that suddenly becomes CPU-intensive will be under-clocked for tens of milliseconds while the utilization signal catches up. The symptom is a characteristic latency spike at workload burst onset followed by recovery as the frequency governor responds. For latency-sensitive workloads (real-time audio, high-frequency trading) the standard mitigation is to bypass schedutil and use the `performance` governor (pinning frequency to maximum) or to tune `sched_util_clamp_min` so the floor frequency is already near the required level at burst onset.

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

#pagebreak()


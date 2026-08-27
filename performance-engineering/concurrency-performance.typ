#import "../template.typ": xref

= Concurrency Performance <concurrency-performance>

Adding threads is the easiest way to make a program slower. Concurrency buys throughput only when the work parallelizes without contending, and the failure modes (lock convoys, cache-line ping-pong, oversubscription, tail amplification) are subtle precisely because the code remains correct while the performance collapses. This chapter covers lock contention and its diagnosis, lock-free and sharded alternatives, thread-pool sizing, work stealing, and the statistics of fan-out latency.

*See also:* #xref("performance-engineering", "memory-performance", label: "Memory Performance") (false sharing and coherence costs), #xref("performance-engineering", "methodology", label: "Performance Methodology") (Amdahl and the USL, which bound everything here), #xref("performance-engineering", "queueing-theory", label: "Queueing Theory") (thread pools are queues), and the CPU Architecture volume's #xref("cpu-architecture", "multicore", label: "Cache Coherence") chapter.

== Where Concurrency Costs Come From

The USL's two coefficients name the two taxes:

- *Contention* ($sigma$): work serialized behind a shared resource: a mutex, a global allocator, a single log writer, the GIL. Throughput plateaus.
- *Coherency* ($kappa$): the cost of keeping shared mutable state consistent: cache lines migrating between cores, atomic read-modify-writes, memory barriers. Throughput peaks and then _falls_ as cores are added.

An uncontended atomic increment costs around 10-20 cycles; the same increment with the line in another core's cache costs a coherence round-trip, 40-100+ ns, and under heavy contention the interconnect itself saturates. A single shared counter updated by 32 cores can be slower in aggregate than one core updating it alone, which is the canonical demonstration of retrograde scaling.

== Lock Contention

=== Anatomy of a contended lock

A mutex is cheap when uncontended (a single CAS, a few nanoseconds). Under contention, three regimes emerge:

1. *Light contention*: occasional spin-then-acquire; cost is the coherence traffic on the lock word.
2. *Heavy contention*: threads park in the kernel (`futex` on Linux); each handoff costs a wakeup, microseconds, plus a cold cache on the resumed thread.
3. *Lock convoy*: the lock hold time is short but the handoff overhead dominates, so the lock passes from thread to thread at the speed of wakeups, and throughput is bounded by handoff rate rather than by work.

The governing identity: a lock held for $h$ seconds per acquisition supports at most $1 \/ h$ acquisitions per second system-wide, regardless of core count. Shrinking the critical section is the optimization; everything else is mitigation.

=== Reducing contention

- *Shrink the critical section*: move allocation, I/O, and formatting outside the lock; lock only the pointer swap.
- *Shard / stripe*: split one lock into $N$ locks by key hash (striped hash maps, per-CPU counters). Java's `LongAdder` and Linux per-CPU variables are this pattern; reads pay an aggregation cost, writes scale.
- *Read-write locks*: help only when readers vastly outnumber writers and critical sections are long; the lock word itself still bounces. `RCU` (read-copy-update) and epoch-based reclamation remove readers from the coherence protocol entirely: readers pay nothing, writers copy and defer reclamation. RCU is pervasive in the Linux kernel for read-mostly data.
- *Flat combining / delegation*: instead of every thread acquiring the lock, threads publish requests and one combiner executes them in a batch, turning $N$ coherence migrations into one.
- *Avoid sharing*: the fastest synchronization is none. Thread-local accumulation with periodic merge, single-writer designs, and message passing (queues between pinned threads, as in LMAX Disruptor or thread-per-core frameworks like Seastar) sidestep the problem structurally.

=== Diagnosing contention

- `perf lock record` / `perf lock contention` (Linux 5.19+): kernel-assisted lock contention profiling with stacks, no recompilation.
- *Off-CPU analysis* (Gregg): profile where threads _block_ rather than where they run; `offcputime` from bcc/bpftrace attributes blocked time to stacks. A service at 20% CPU with terrible latency is usually an off-CPU problem.
- Mutex profilers: `mutrace`, jemalloc's mutex stats, Go's `runtime/pprof` mutex and block profiles, JFR's monitor-blocked events.
- The signature in a CPU profile: time in `futex_wait`, spin loops in lock slow paths (`__lll_lock_wait`, `pthread_mutex_lock` slow path), or, for spinlocks, mysteriously hot `pause` instructions.

== Lock-Free and Wait-Free Structures

Lock-free algorithms (CAS loops on shared state) guarantee system-wide progress and eliminate convoying and priority inversion, but they do *not* eliminate coherence traffic: a CAS on a hot word costs the same line migration as a lock acquisition, and a failed CAS wastes the round-trip. A lock-free queue under heavy contention can be slower than a well-built locked one. They win when contention is moderate, when preemption-tolerance matters (a preempted lock-holder stalls everyone; a preempted CAS loop stalls no one), and in single-producer/single-consumer configurations where the algorithm degenerates to plain loads and stores with ordering (an SPSC ring buffer needs no atomic RMW at all).

Practical guidance: prefer proven implementations (`crossbeam`, `folly::MPMCQueue`, `moodycamel::ConcurrentQueue`, `java.util.concurrent`) over bespoke ones; memory reclamation (hazard pointers, epochs) is where handwritten lock-free code goes wrong. Beware the *ABA problem* and the cost of `seq_cst` fences where acquire/release suffices.

== Thread Pools and Sizing

A thread pool is a queueing system, and sizing it is a queueing problem:

- *CPU-bound work*: about one thread per hardware thread; more adds context-switch and cache-pollution cost without throughput. SMT siblings share execution resources, so $2 times$ threads on SMT rarely yields $2 times$ throughput (typically 1.1-1.3 times).
- *Blocking work*: the classic sizing estimate is $N = c dot (1 + W \/ S)$, where $c$ is core count, $W$ is wait time, and $S$ is service (compute) time per task. A task that waits 9 ms on I/O for every 1 ms of compute wants about $10 c$ threads.
- *Oversubscription*: far more runnable threads than cores causes context-switch storms (a switch costs 1-10 microseconds directly, more in cache refill), run-queue latency, and, in the worst case, the scheduler thrashing the working set out of cache every quantum.

Separate pools for separate work classes (fast CPU-bound vs. slow blocking) prevent *head-of-line blocking* where slow tasks occupy all workers. Bounded queues with backpressure beat unbounded queues, which convert overload into latency and OOM (see #xref("performance-engineering", "capacity-planning", label: "Capacity Planning")).

== Work Stealing and Structured Parallelism

Fork-join runtimes (Cilk, TBB, Rust's `rayon`, Java's ForkJoinPool, Go's scheduler) use *work stealing*: each worker owns a deque, pushes and pops its own tasks at one end (LIFO, cache-warm), and idle workers steal from the other end (FIFO, the largest pending subtrees). Stealing is rare in balanced workloads, so synchronization stays off the hot path. The practitioner's levers:

- *Grain size*: tasks must be large enough to amortize scheduling (microseconds of work minimum); recursive splitting with a sequential cutoff is the standard pattern.
- *Avoid blocking inside the pool*: a blocked worker is a lost core; runtimes differ in whether they compensate (Go's netpoller and Tokio's `spawn_blocking` handle this; naive fork-join pools do not).
- *NUMA-aware stealing*: steal from same-socket victims first, or partition the pool per node.

== Tail Latency and Fan-Out

Dean and Barroso's "The Tail at Scale" (2013) quantified why concurrency amplifies tails: a request that fans out to $n$ services in parallel completes when the *slowest* leg completes. If each leg independently exceeds its p99 with probability 0.01, the probability the whole request stays under that threshold is $0.99^n$: with $n = 100$, only 37% of requests avoid every leg's worst percentile, so the per-leg p99 becomes roughly the fan-out p37. Tail tolerance therefore has to be engineered:

- *Hedged requests*: send a duplicate to a second replica after the p95 delay has elapsed; take the first response. Cost: a few percent extra load; benefit: the tail collapses toward the p95.
- *Tied requests*: enqueue on two replicas, cancel the loser on first dispatch (used in distributed file system reads at Google).
- Micro-causes of per-leg tails: GC pauses, timer interrupts, background compaction, power management state transitions, and lock contention, each invisible in averages.

== Pitfalls

- *Parallelizing the unprofiled*: if the hot region is 30% of runtime, perfect parallelism of it caps the speedup at $1.4 times$ (Amdahl). Profile first.
- *Benchmarking without contention*: a concurrent structure measured single-threaded, or with uniformly random keys, hides the hot-key behavior production will have.
- *Sleeping/spinning hybrids tuned on one machine*: spin durations appropriate for one core count and one lock-hold distribution misbehave on another.
- *Ignoring the memory model*: code that passes tests on x86 (strong ordering) and fails on ARM is a performance chapter's correctness footnote, but fences added in panic ("`seq_cst` everywhere") then become the performance problem.
- *Async as a throughput cure-all*: async runtimes remove thread-per-connection memory costs and switch costs, but CPU-bound work still needs cores, and a single accidentally-blocking call stalls an entire executor thread and every task multiplexed on it.

== Further Reading

- Dean, J., & Barroso, L. A. (2013). The tail at scale. _CACM_, 56(2).
- Herlihy, M., & Shavit, N. (2012). _The Art of Multiprocessor Programming_, revised ed. Morgan Kaufmann.
- McKenney, P. (2017). _Is Parallel Programming Hard, And, If So, What Can You Do About It?_ (perfbook).
- Gregg, B. (2020). _Systems Performance_, 2nd ed., ch. 5 (Applications) and off-CPU analysis material. Addison-Wesley.
- Blumofe, R., & Leiserson, C. (1999). Scheduling multithreaded computations by work stealing. _JACM_, 46(5).
- Thompson, M. et al. (2011). LMAX Disruptor: high performance alternative to bounded queues. Technical paper.

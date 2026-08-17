#import "../template.typ": xref

= RCU and Locking

A modern kernel running on 256 cores cannot scale with traditional locking; a single `rwlock` becomes the slowest part of the system long before the workload itself does. The Linux kernel's answer is a layered locking discipline: spinlocks and mutexes where contention is tolerable; per-CPU data wherever possible; *read-copy-update* (RCU) wherever readers vastly outnumber writers. Combined with strict lock-ordering rules and the lockdep validator, this is what lets the kernel scale linearly to hundreds of cores while remaining correct.

This chapter covers the locking primitives at depth, then RCU's grace-period machinery (the most subtle and most important of them), followed by lockdep, futexes, and per-CPU patterns.

== Locking Primitives at a Glance

#table(columns: (auto, 1fr, 1fr),
  [*Primitive*], [*Use*], [*Cost*],
  [`spinlock_t`], [Short, atomic-context critical sections.], [~20-30 ns uncontended, scales poorly.],
  [`raw_spinlock_t`], [Same, but never converted to mutex by PREEMPT_RT.], [Same; required in IRQ paths.],
  [`rwlock_t`], [Many readers, occasional writer. Discouraged; RCU is usually better.], [Reader fairness issues; deprecated for most uses.],
  [`mutex`], [Sleepable, single owner. Process context only.], [~30 ns uncontended, sleeps on contention.],
  [`rw_semaphore`], [Sleepable read-write. Many filesystems use it on inodes.], [~40 ns uncontended.],
  [`seqlock_t`], [Writers serialize, readers retry on conflict. Tiny read-side cost.], [Read: 2 sequence loads; write: one spinlock.],
  [`completion`], [One-time wait/signal.], [~30 ns to set; sleeping wait.],
  [`atomic_t` / `atomic64_t`], [Lock-free counters and flags.], [~5-20 ns per RMW.],
  [`local_lock_t`], [Per-CPU section marker; on PREEMPT_RT becomes a per-CPU mutex.], [Free on non-RT.],
  [`percpu_ref` / `percpu_rwsem`], [Per-CPU refs with batched drain.], [~3 ns hot path; expensive teardown.],
  [`bit_spinlock`], [Spinlock packed into a flag bit of a larger word.], [Use only when memory is the constraint.],
  [`futex`], [Userspace fast/slow path mutex.], [User-side uncontended ~5 ns; kernel slow path ~200 ns.],
)

== Spinlocks: Variants and IRQ Discipline

`spin_lock()` disables preemption; `spin_lock_irqsave()` additionally disables local IRQs and saves the prior state. The rule: *if a lock is ever taken in IRQ context, every taker must disable IRQs.* Otherwise an IRQ that fires while the lock is held tries to retake it → self-deadlock.

```c
unsigned long flags;
spin_lock_irqsave(&queue_lock, flags);
list_add(&item, &q->head);
spin_unlock_irqrestore(&queue_lock, flags);
```

On PREEMPT_RT, `spin_lock` becomes a sleeping `rt_mutex` with priority inheritance. `raw_spin_lock` stays a true spinlock, required for the few code paths that *cannot* sleep (top half of IRQ handlers, scheduler core).

Spinlocks are *queued* (`qspinlock`, default since 4.2): contenders form an MCS-style queue rather than thundering on a single cacheline. The result: contention is O(1) cacheline transfers per acquire instead of O(N).

== Mutexes and Optimistic Spinning

`mutex` is the default sleepable lock. The fast path is a `cmpxchg`; on contention, the lock *optimistically spins* if the current owner is running on another CPU (the bet: it'll release soon). Only if the owner is descheduled does the contender enqueue and sleep. This single trick makes mutex contention competitive with spinlocks on short critical sections, without the "what if we get preempted holding it" downside.

`mutex_lock_killable` returns `-EINTR` on SIGKILL, required for any wait that might be a user-facing syscall.

== Sequence Locks (seqlock)

A `seqlock` is asymmetric: writers acquire a spinlock and increment a sequence counter (odd while writing, even when consistent); readers snapshot the counter, read, and *retry* if the counter changed.

```c
unsigned seq;
do {
    seq = read_seqbegin(&timekeeper_lock);
    now = compute_now();
} while (read_seqretry(&timekeeper_lock, seq));
```

Used for: `gettimeofday`/timekeeping (`kernel/time/timekeeping.c`), dcache invariants, networking statistics. Read-side cost is two atomic loads and a comparison (sub-nanosecond), but writers serialize, and readers can starve under sustained writer pressure.

== Atomics and Memory Ordering

`atomic_*` and `atomic64_*` provide architecture-portable load/store/RMW. The C-style suffixes control ordering:

```c
val = atomic_read(&a);                    // plain load (relaxed)
atomic_set(&a, v);                        // plain store
old = atomic_cmpxchg(&a, expected, new);  // full barrier on success
old = atomic_fetch_add_acquire(1, &a);    // acquire on the load side
atomic_inc_return_release(&a);            // release on the store side
smp_mb();                                  // explicit full barrier
smp_rmb(); smp_wmb();                      // read/write halves
```

The kernel's ordering rules are stricter than C11's: `READ_ONCE`/`WRITE_ONCE` plus explicit barriers form the kernel's portable memory model. `Documentation/memory-barriers.txt` is the canonical reference; `tools/memory-model/` carries the formal model (used by `herd7`).

== RCU: Read-Copy-Update

RCU is the kernel's answer to the "millions of readers, occasional writer, never block readers" problem. Readers run in a *read-side critical section* delimited by `rcu_read_lock()`/`rcu_read_unlock()`, which on the default preemptible kernel is essentially `preempt_disable()` / `preempt_enable()` (no atomics, no barriers on x86). Writers publish a new version, then wait for a *grace period* (a duration after which every CPU has passed through a quiescent state) before freeing the old version. By then, no reader can possibly still hold a pointer to it.

```c
// Reader
rcu_read_lock();
struct foo *f = rcu_dereference(global_foo);
do_stuff(f->x);
rcu_read_unlock();

// Writer
struct foo *new = kmalloc(sizeof(*new), GFP_KERNEL);
*new = ...;
struct foo *old = rcu_replace_pointer(global_foo, new, lock_held);
synchronize_rcu();          // wait for grace period
kfree(old);                 // safe: no reader can hold the old pointer
```

`rcu_dereference` is a compiler barrier preventing the compiler from speculating loads; on Alpha (the only architecture where data dependencies don't order loads) it also emits a memory barrier. `rcu_assign_pointer` is a `smp_store_release`.

The grace-period guarantee is the entire trick: after `synchronize_rcu()` returns, the kernel knows every CPU has either (a) been scheduled out, (b) entered/exited userspace, or (c) entered idle — none of which can occur inside an RCU read section. Therefore no CPU still holds a pointer obtained from the prior version.

== RCU Flavours

Linux has several RCU implementations, all sharing the public API but tuned for different contexts:

#table(columns: (auto, 1fr),
  [*Tree RCU* (`kernel/rcu/tree.c`)], [The default. Hierarchical structure of `rcu_node` objects scales grace-period detection to thousands of CPUs. Grace period ~10-30 ms.],
  [*Tasks RCU*], [Quiescent state is voluntary context switch. Used for tracing trampolines where the read sections cross many regular RCU quiescent states.],
  [*Tasks Trace RCU*], [Like Tasks RCU, but read sections can sleep. Used by sleepable BPF.],
  [*Tasks Rude RCU*], [QS = IPI on every CPU. Heaviest hammer; for things that must wait out everything.],
  [*SRCU* (Sleepable RCU)], [Read sections that can sleep, with per-instance grace periods. Common in `module`/`device` teardown.],
  [*RCU-bh* / *RCU-sched*], [Pre-merge variants; subsumed into a unified RCU since 4.20.],
)

`call_rcu(rcu_head, fn)` schedules `fn` to run after the next grace period — the asynchronous form, used everywhere kfree-after-RCU is needed without blocking.

== Grace Period Mechanics

In Tree RCU, each CPU reports its quiescent state up through the rcu_node tree. The root's "all CPUs reported" event marks the end of a grace period; pending callbacks are then invoked in batches (`rcu_do_batch`).

Key tuning levers:

- *`rcu_nocbs=`* boot parameter: offload `call_rcu` callback processing from the issuing CPUs onto dedicated rcuo kthreads. Essential for `nohz_full` cores that must not be interrupted by callback floods.
- *`rcutree.kthread_prio`*: priority of RCU's kthreads. Bumping above SCHED_OTHER prevents RT tasks from starving grace-period progression.
- *Expedited grace periods* (`synchronize_rcu_expedited`): IPI all CPUs to force immediate quiescent states. Sub-millisecond but disruptive; reserved for setup/teardown.

`rcu_torture` and `Documentation/RCU/` are essential reading; McKenney's _Is Parallel Programming Hard, And, If So, What Can You Do About It?_ is the textbook.

== When RCU Wins (and When It Doesn't)

RCU is unbeatable when:

- Readers vastly outnumber writers.
- Read sections are short.
- The data structure can be replaced atomically (single pointer swap, or RCU-friendly list/hash with `*_rcu` operations).

RCU is wrong when:

- Writers are common (grace-period overhead dominates).
- Readers must see *fully consistent* multi-pointer state (RCU only gives single-pointer atomicity).
- Memory pressure makes deferred frees unacceptable.

The classic fits: routing table, ARP cache, the inode/dentry cache, security-module rule sets, the task list. The classic mismatches: ref-counted producer-consumer queues (use a lock-free ring), high-write counters (use `percpu`).

== Lockdep

Lockdep (`kernel/locking/lockdep.c`, `CONFIG_PROVE_LOCKING`) tracks at runtime every lock-class acquisition order and detects:

- *AB-BA deadlocks*: any pair acquired in one order on one path, in the opposite order on another.
- *IRQ inversions*: a lock taken with IRQs disabled on one path and without on another that's reachable from IRQ context.
- *Sleeping in atomic context*.

When it fires, lockdep prints both stacks, the lock classes involved, and the "possible deadlock" diagnostic. It's enabled in debug kernels and is *the* first-line debugger of locking bugs. Cost: ~5-10% runtime overhead, so production kernels usually omit it.

```
WARNING: possible circular locking dependency detected
   CPU0                    CPU1
   ----                    ----
   lock(&inode->i_mutex);
                           lock(&dentry->d_lock);
                           lock(&inode->i_mutex);
   lock(&dentry->d_lock);
```

== Per-CPU Data and local_lock

The cheapest "lock" is no lock. `DEFINE_PER_CPU(type, var)` allocates one copy per CPU, accessed via `this_cpu_*` (preempt-disabled access) or `per_cpu(var, cpu)`. Counters, statistics, slab caches, page-frag pools all use this. Aggregation happens lazily on read.

`local_lock_t` marks a per-CPU critical section: on non-RT kernels it's `preempt_disable`; on PREEMPT_RT it becomes a *per-CPU sleeping mutex*, allowing the section to be preempted by RT tasks while still serializing access to the per-CPU state.

`percpu_ref` is a refcount that's per-CPU during the "live" phase (cheap atomic_inc on the local CPU) and switched to atomic mode at teardown (where the global count converges so the refcount can hit zero).

== Futexes

Futex (`kernel/futex/`) is the userspace-kernel hybrid mutex. The uncontended fast path is a userspace `cmpxchg`; only on contention does the slow path enter the kernel via `FUTEX_WAIT` / `FUTEX_WAKE`, which suspends the thread on a hash bucket keyed by the futex address.

Glibc's `pthread_mutex_t`, `pthread_cond_t`, and `pthread_rwlock_t` are implemented atop futex. Priority-inheritance futexes (`FUTEX_LOCK_PI`) integrate with the RT scheduler.

The recent `FUTEX2` syscall family adds size variants (32/64), better timeout semantics, and `futex_waitv` for atomic multi-wait (powers Wine's WaitForMultipleObjects, dramatically improving game-port performance).

== Lock-Free and Wait-Free Patterns

The kernel uses a handful of well-tested lock-free patterns:

- *`hlist_nulls`* RCU lookups in slab and socket hashes — the "nulls" terminator encodes the bucket index so a reader can detect mid-walk migration.
- *Lock-free single-producer/single-consumer rings* in tracing and io_uring (see _IO uring_ for the SPSC ring construction).
- *MCS / qspinlock* — the queueing variant of spinlock.
- *Hazard pointers* — used in a handful of places (notably the `objpool` infrastructure for kretprobes); RCU is preferred for most workloads.

Writing a new lock-free structure in the kernel without the maintainers' explicit blessing is a quick way to a NACK.

== Common Pitfalls

- *Sleeping while atomic*. Calling any sleeping function (allocator with non-atomic flags, copy_from_user, mutex_lock, schedule) inside a spinlock or RCU read section. `might_sleep()` annotations catch this in debug builds.
- *Mixed IRQ disable*. Acquiring a lock with IRQs enabled in one path and from IRQ context in another. Use `_irqsave` variants consistently.
- *Forgetting `rcu_dereference`*. A plain `READ_ONCE` will work today but invites the compiler to refold the load and break the dependency invariant.
- *Holding locks across `synchronize_rcu`*. The grace period can be milliseconds; you'll block writers and probably hit lockdep warnings about possible recursive deadlock.
- *Releasing in error path*. The `goto out` style is conventional precisely because matching unlock paths is otherwise error-prone.

== Observability

```bash
# Lockdep stats
cat /proc/lockdep_stats

# What's hot in kernel locking? (debug kernel)
cat /proc/lock_stat | head

# RCU diagnostics
cat /sys/kernel/debug/rcu/rcugp

# Long RCU grace periods?
bpftrace -e 'kprobe:rcu_gp_kthread { @ = count(); }'

# Spinlock contention sampling (perf)
perf record -e lock:contention_begin -ag -- sleep 5
perf report
```

== Further Reading

McKenney, P. (2024). _Is Parallel Programming Hard, And, If So, What Can You Do About It?_ Free PDF, the canonical RCU and parallelism reference.

McKenney, P. (2004). _Exploiting Deferred Destruction: An Analysis of Read-Copy-Update Techniques in Operating System Kernels_. PhD thesis, OGI.

Kernel docs: `Documentation/RCU/`, `Documentation/memory-barriers.txt`, `Documentation/locking/`, `Documentation/scheduler/sched-rt-group.rst`.

Howard, P. and Walpole, J. (2014). _Relativistic Causal Ordering_ (relativistic programming).

LWN: Corbet's RCU series (1990s-2024), lockdep introduction, PREEMPT_RT progress, qspinlock.

Tools: `tools/memory-model/` (herd7-based formal kernel memory model).

`kernel/rcu/tree.c`, `kernel/locking/qspinlock.c`, `kernel/locking/lockdep.c`, `kernel/locking/rwsem.c`, `kernel/futex/core.c`.

*See also:* #xref("linux-kernel", "scheduler", label: "Scheduler") (PREEMPT-RT and priority inheritance), #xref("linux-kernel", "kernel-tracing", label: "Kernel Tracing") (lock event tracepoints), #xref("linux-kernel", "ebpf-deep-dive", label: "eBPF Deep Dive") (sleepable BPF uses Tasks Trace RCU), #xref("linux-kernel", "interrupts", label: "Interrupts and Bottom Halves") (IRQ-context locking constraints).

= Synchronization Primitives

Mutexes, reader-writer locks, and lock-free techniques sit on top of the hardware atomics and memory model from #emph[Multicore and Cache Coherence]. This chapter covers how they are *implemented* and how to reduce their cost when they contend.

*See also:* Multicore (for atomics, memory ordering, cache coherence), Cache Hierarchy (for false sharing and cache-line ping-pong)

== Mutex Implementation: User Space + Kernel

A mutex is *neither purely a userland nor a kernel construct* — it's a two-stage design where the kernel is consulted only on contention.

*Uncontended fast path (user space only):*

```cpp
// Conceptual sketch — real impls use platform primitives
void lock() {
    int expected = 0;
    if (state_.compare_exchange_strong(expected, 1,
            std::memory_order_acquire)) {
        return;                              // Got it. ~15 cycles total.
    }
    slow_lock();                             // Contended — fall through
}
void unlock() {
    if (state_.exchange(0, std::memory_order_release) == 2) {
        wake_one_waiter();                   // Only syscall if waiters
    }
}
```

The uncontended case is *one atomic CAS* and never enters the kernel — typically 15-30 cycles.

*Contended slow path (kernel-assisted):*

- *Linux:* `futex(FUTEX_WAIT, addr, expected)` — the kernel checks `*addr == expected` (atomically with the wait-queue lock) and parks the thread on a wait queue keyed by the physical address. `FUTEX_WAKE` unparks it. The mutex word itself stays in user memory; the kernel only stores wait-queue state.
- *Windows:* `WaitOnAddress` / `WakeByAddressSingle` is the direct futex analogue (Win8+). `CRITICAL_SECTION` builds on it with a spin count before parking. `SRWLOCK` is a slim reader-writer variant.
- *macOS:* `os_unfair_lock` + `__ulock_wait` (private syscall). `pthread_mutex_t` defaults to this on recent macOS.

```
Uncontended lock:   ~15-30 cycles  (1 CAS, no syscall)
Contended → park:   ~1000-3000 cycles  (futex syscall: ~150 ns)
Wake + reschedule:  ~3000-10000 cycles  (context switch on wake)
```

This is why a "mutex is expensive" only when *contended*. An uncontended `std::mutex` is often cheaper than people assume — comparable to an atomic increment.

== Where the Slow Path Lives

The fast path of every mainstream mutex is the same: a single atomic RMW on a userspace word. The interesting design choice is *where the waiter parks* once contention happens. Three destinations exist; real implementations often layer them.

*1. Spin — pure userspace, no parking.*

The waiter keeps re-reading the lock word in a tight loop until it sees the lock released, then retries the CAS. No syscall, no scheduler involvement, no queue.

```cpp
struct Spinlock {
    std::atomic<bool> locked_{false};
    void lock() {
        while (true) {
            if (!locked_.exchange(true, std::memory_order_acquire)) return;
            while (locked_.load(std::memory_order_relaxed)) {
                __builtin_ia32_pause();          // x86 PAUSE — ~20-140 cycles,
                                                 //   yields SMT pipeline, avoids
                                                 //   memory-order misspeculation
            }
        }
    }
    void unlock() { locked_.store(false, std::memory_order_release); }
};
```

Pure spin is correct only when (a) the critical section is *shorter than a context switch* (a few hundred cycles at most), and (b) the holder is *guaranteed running on another core* — otherwise the spinner monopolizes the CPU that the holder needs to make progress (the classic "spin on uniprocessor" deadlock-by-livelock).

- *Linux kernel `spinlock_t`* (SMP) — qspinlock, an MCS-style queued spinlock since v4.2. Each waiter spins on its *own* cache line (the previous waiter's `next` pointer), eliminating the cache-line ping-pong of naïve test-and-set.
- *`pthread_spin_lock`* — userspace POSIX spinlock. Useful only when you've measured that the critical section is sub-microsecond and you've pinned threads to distinct cores.
- *Adaptive prelude* — glibc's `PTHREAD_MUTEX_ADAPTIVE_NP`, Windows `CRITICAL_SECTION` `SetCriticalSectionSpinCount`, `parking_lot::Mutex` (~10 µs of spin before parking). The spin is a *prediction* that the holder is about to release; if wrong, fall through to a real park.

Cost: ~1-3 cycles per spin iteration, but every contended atomic exchange invalidates the cache line on the holder's core. Above ~4 contenders, an unqueued spinlock collapses to O(n²) coherence traffic — which is why qspinlock / MCS / CLH variants exist.

*2. Runtime — userspace task scheduler, no thread suspension.*

The waiter is a *task* (goroutine, async future, fiber, green thread), not an OS thread. The mutex enqueues the task onto a userspace wait list and returns control to the runtime's scheduler, which picks another task to run on the same OS thread. No syscall to park; the kernel never knows a wait happened.

- *`tokio::sync::Mutex`* (Rust) — the `lock().await` future, on contention, registers a `Waker` in an intrusive doubly-linked list inside the mutex, returns `Poll::Pending`, and yields. The executor polls other ready futures on the same worker thread. On `unlock`, the next waiter's `Waker::wake()` re-enqueues that task into the executor's run queue.
- *Go `sync.Mutex`* — goroutine parking via `runtime_SemacquireMutex` → `gopark`. The goroutine's `g` struct is moved off the P's run queue onto the mutex's `semaRoot` treap (keyed by address). The OS thread (`m`) immediately schedules another runnable `g`. There's a brief active-spin prelude (~4 iterations of `procyield`) before parking. Starvation mode (handoff after 1 ms wait) is a fairness layer, not a different blocking primitive.
- *`async-std`, `smol`, `glommio`* — same pattern: future + waker + executor-local run queue.
- *Fiber libraries* (Boost.Fiber, Folly fibers, Java's Project Loom virtual threads) — `park()` swaps the fiber's stack pointer out and runs the scheduler. The carrier thread never blocks.

The key invariant: *the OS thread stays runnable*. M tasks can wait on M mutexes while N OS threads (N ≪ M) keep doing useful work. This is why async runtimes scale to 100k+ in-flight operations on 8 cores — a kernel thread per waiter would exhaust pid_max or memory long before that.

```cpp
// Conceptual async mutex slow path
auto lock_async() -> task<guard> {
    if (try_lock()) co_return guard{this};
    auto waiter = waiter_node{current_task::waker()};
    queue_.push(&waiter);                        // userspace intrusive list
    co_await suspend();                          // returns to executor
    co_return guard{this};                       // resumed by unlock()
}
```

Caveat — *the runtime itself eventually parks in the kernel*. When *every* task is blocked and the executor's run queue is empty, the worker thread calls `epoll_wait` / `kqueue` / `io_uring_enter` / `futex` to wait for I/O or a wake. Go's `findrunnable` ends in `notesleep` → futex. Tokio's worker parks on a `Parker` backed by `std::thread::park` → futex. So "runtime-parked" means the *task* is in userspace; the underlying *thread* may still hit the kernel when there's nothing else to do.

*3. Kernel — thread descheduled via futex/equivalent.*

The waiter is an OS thread with no other work. The runtime (if any) has run out of tricks and needs the kernel to remove the thread from the runqueue until the holder releases.

- *Linux `futex(2)`* — the canonical primitive. `FUTEX_WAIT(uaddr, expected, timeout)`:
  + Kernel takes the hash-bucket lock for the `futex_q` keyed by `(mm, virtual_addr)` (or `(inode, page_offset)` for `FUTEX_PRIVATE_FLAG`-absent shared futexes).
  + Atomically re-reads `*uaddr`; if `!= expected`, returns `EAGAIN` (the wake we were racing against already happened).
  + Otherwise enqueues the thread on the bucket's plist, marks it `TASK_INTERRUPTIBLE`, and calls `schedule()`.
  + `FUTEX_WAKE(uaddr, n)` walks the bucket, removes up to `n` matching waiters, and wakes them via `wake_up_q`.
  + Variants: `FUTEX_WAIT_BITSET` (selective wake), `FUTEX_CMP_REQUEUE` (move waiters from one futex to another without waking — used by `pthread_cond_broadcast` to avoid thundering herd), `FUTEX_LOCK_PI` (priority-inheritance, hands the lock directly to highest-prio waiter, uses `rt_mutex` internally), `FUTEX_WAIT_REQUEUE_PI` (condvar-on-PI-mutex), `FUTEX2` (since 5.16, 64-bit values + NUMA hints).
- *Windows `WaitOnAddress` / `WakeByAddressSingle`* (Win8+) — same shape as futex, but the kernel uses the *virtual* address as the key (no shared-mapping equivalence). Underlies `SRWLOCK`, `CONDITION_VARIABLE`, and modern `CRITICAL_SECTION` (older versions used a per-CS kernel `KEVENT`, allocated lazily). `KeWaitForSingleObject` is the lower-level dispatcher-object wait used by named primitives (`Mutex`, `Semaphore`, `Event`).
- *macOS `__ulock_wait` / `__ulock_wake`* (private, but stable) — futex analogue. Public surface: `os_unfair_lock` (one-word, non-recursive, no priority inheritance opt-in until macOS 12 added `os_unfair_lock_with_options(OS_UNFAIR_LOCK_ADAPTIVE_SPIN)`). `pthread_mutex_t` on recent macOS is a thin wrapper.
- *FreeBSD `_umtx_op(UMTX_OP_WAIT_UINT)`* — the BSD futex.

Cost breakdown for a Linux futex wait → wake → resume:

```
futex(FUTEX_WAIT) syscall entry:    ~100-150 ns   (syscall + hash + recheck)
schedule() out:                     ~1-2 µs       (save state, pick next)
... (waiting — pure latency, no CPU) ...
futex(FUTEX_WAKE) by releaser:      ~100 ns
wake_up_q + IPI to target CPU:      ~500 ns - 2 µs
schedule() in + cache warmup:       ~3-10 µs      (TLB/L1/L2 cold)
```

Round-trip floor: ~5-15 µs on idle hardware, easily 50+ µs under load. This is why even a 1% contention rate on a hot mutex can dominate a workload — the cheap path is ~10 ns, the slow path is ~10 µs, a 1000× ratio.

*The layering in practice.*

A modern `parking_lot::Mutex` lock under contention executes *all three* in sequence:

+ Spin ~10 µs (case 1) — guess the holder is about to release.
+ Hash the mutex address into a global `ParkingLot` table; push the thread onto an intrusive bucket queue (userspace queue — same data-structure idea as case 2, but the "tasks" are OS threads).
+ Call `futex(FUTEX_WAIT)` on a *per-thread* parker word (case 3) to actually deschedule.

So the question "where does the slow path live?" is really "at what level does this primitive give up and ask the next layer down?" The fast path is *always* userspace; the slow path is a stack — spin, runtime, kernel — and each layer is a hedge against the cost of the next.

== Concrete Branches: Where Fast Becomes Slow

Abstract descriptions hide where the branch *actually is* in the code. Two canonical examples — the Linux kernel `struct mutex` and glibc `pthread_mutex_t` — show the exact CAS, the exact transition criterion, and what each layer does before giving up.

*Linux kernel `struct mutex` (`kernel/locking/mutex.c`).*

The mutex is a single 64-bit atomic, `atomic_long_t owner`, whose top bits hold the owning `task_struct *` and bottom 3 bits hold flags (`MUTEX_FLAG_WAITERS`, `MUTEX_FLAG_HANDOFF`, `MUTEX_FLAG_PICKUP`). Unlocked ⇔ `owner == 0`.

```c
// Fast path — kernel/locking/mutex.c, simplified
static __always_inline bool __mutex_trylock_fast(struct mutex *lock)
{
    unsigned long curr = (unsigned long)current;
    unsigned long zero = 0UL;
    return atomic_long_try_cmpxchg_acquire(&lock->owner, &zero, curr);
}

void __sched mutex_lock(struct mutex *lock)
{
    might_sleep();
    if (!__mutex_trylock_fast(lock))             // <-- the branch
        __mutex_lock_slowpath(lock);
}
```

*Criterion for fast path:* `owner == 0` at the moment of CAS. One acquire-ordered `cmpxchg`. ~15-25 cycles, no function call beyond the inlined trylock.

The slow path is *not* an immediate sleep. It's a three-stage cascade:

```c
// __mutex_lock_common — simplified control flow
static int __mutex_lock_common(struct mutex *lock, unsigned state, ...)
{
    // Stage A: try one more time inline (the holder may have released
    //          between our failed fast CAS and entering the slowpath)
    if (__mutex_trylock(lock)) goto acquired;

    // Stage B: OPTIMISTIC SPIN, gated by mutex_can_spin_on_owner()
    if (mutex_optimistic_spin(lock, ww_ctx, NULL))
        goto acquired;                           // got it by spinning

    // Stage C: enter the wait list, mark MUTEX_FLAG_WAITERS, sleep
    spin_lock(&lock->wait_lock);
    __mutex_add_waiter(lock, &waiter, ...);
    for (;;) {
        if (__mutex_trylock_or_handoff(lock, ...)) break;
        set_current_state(state);                // TASK_UNINTERRUPTIBLE etc.
        spin_unlock(&lock->wait_lock);
        schedule_preempt_disabled();             // <-- actually sleep
        spin_lock(&lock->wait_lock);
    }
acquired:
    ...
}
```

The *transition criterion* between spinning and sleeping is `mutex_can_spin_on_owner()`:

```c
static inline int mutex_can_spin_on_owner(struct mutex *lock)
{
    struct task_struct *owner;
    int retval = 1;
    if (need_resched()) return 0;                // (1) someone else needs CPU
    rcu_read_lock();
    owner = __mutex_owner(lock);
    if (owner) retval = owner->on_cpu;           // (2) owner is RUNNING NOW
    rcu_read_unlock();
    return retval;
}
```

Two conditions, and they are the entire reason kernel mutexes don't always go straight to `schedule()`:

+ *`!need_resched()`* — no higher-priority task is waiting to run on *this* CPU. If the scheduler has flagged us, spinning is theft.
+ *`owner->on_cpu`* — the lock holder is *currently executing on some other CPU*. If true, the holder will release "soon" (microseconds), and a context-switch round-trip (~10 µs) costs more than spinning through it. If false (holder is itself blocked or preempted), spinning is hopeless and we must sleep.

Inside `mutex_optimistic_spin` the waiter joins an *MCS queue* (`osq_lock` — optimistic spin queue), so only the queue head actually polls `owner`; others spin on their own cache line. The spin loop also re-checks both conditions every iteration and bails to the sleep path the moment `owner` changes to someone not running, or `need_resched()` fires:

```c
// inside mutex_optimistic_spin loop
for (;;) {
    struct task_struct *owner = __mutex_trylock_or_owner(lock);
    if (!owner) break;                           // acquired
    if (!owner->on_cpu || need_resched()) {      // criterion failed
        ret = false; break;                      // -> fall through to schedule()
    }
    cpu_relax();                                 // PAUSE
}
```

So the kernel's layering, in one mutex, is exactly the three cases from the previous section:

#table(
  columns: (auto, auto, auto),
  [*Stage*], [*Where*], [*Exit criterion*],
  [Fast CAS], [`__mutex_trylock_fast`], [`owner == 0`],
  [Optimistic spin], [`mutex_optimistic_spin` + `osq_lock`], [`owner->on_cpu && !need_resched()`],
  [Sleep], [`schedule_preempt_disabled`], [waiter at queue head, woken by `__mutex_unlock_slowpath`],
)

A few related primitives differ in exactly *which* of these stages they include:

- *`spinlock_t`* (qspinlock) — stage 2 only. No fast/slow split because there *is* no sleep path; spinlock holders cannot block, so callers cannot block waiting for one. Used in IRQ context, scheduler internals, anywhere `schedule()` is illegal.
- *`rwsem`* (`struct rw_semaphore`) — same three-stage shape, separate reader/writer paths. Readers fast-path via `atomic_long_add_return_acquire(RWSEM_READER_BIAS, ...)` with a single increment.
- *`rt_mutex`* (PREEMPT_RT, `FUTEX_LOCK_PI`) — *no* optimistic spin; goes straight from fast CAS to a priority-inheritance-aware sleep, because the whole point of RT is bounded latency and unbounded spinning breaks that.

*glibc `pthread_mutex_t` (NPTL, `nptl/pthread_mutex_lock.c`).*

The userspace equivalent. The mutex is a 32-bit `int __lock` with three meaningful states:

```
0 — unlocked
1 — locked, no waiters
2 — locked, MAYBE waiters present (set by any contender before sleeping)
```

```c
// Fast path — LLL_MUTEX_LOCK in lowlevellock.h, expanded
#define lll_trylock(lock) \
    atomic_compare_and_exchange_bool_acq(&(lock), 1, 0)
                                       // returns 0 on success (0 -> 1)

int __pthread_mutex_lock (pthread_mutex_t *mutex)
{
    int oldval = atomic_compare_and_exchange_val_acq(&mutex->__lock, 1, 0);
    if (__glibc_likely(oldval == 0))             // <-- fast path
        return 0;                                // got it, ~10-20 cycles
    return __lll_lock_wait(&mutex->__lock, ...);
}
```

*Criterion for fast path:* identical to the kernel — `lock == 0` at the CAS. One acquire-ordered `cmpxchg`. No syscall.

The slow path does *not* spin by default for a plain `PTHREAD_MUTEX_NORMAL`. It transitions the state to "maybe-waiters" and futexes:

```c
// __lll_lock_wait — simplified
void __lll_lock_wait (int *futex, int private)
{
    if (atomic_load_relaxed(futex) == 2) goto futex_wait;
    while (atomic_exchange_acq(futex, 2) != 0) {  // 0/1 -> 2 atomically
futex_wait:
        futex_wait(futex, 2, private);            // <-- syscall, sleeps
        // returns when somebody FUTEX_WAKE'd us; loop and retry
    }
}
```

The critical detail is the `atomic_exchange_acq(..., 2)`: any contender unconditionally writes `2` *before* calling `FUTEX_WAIT`. That way `pthread_mutex_unlock` knows whether to issue a wake:

```c
int __pthread_mutex_unlock (pthread_mutex_t *mutex)
{
    int oldval = atomic_exchange_rel(&mutex->__lock, 0);
    if (__glibc_unlikely(oldval > 1))             // was state 2?
        lll_futex_wake(&mutex->__lock, 1, private); // only then syscall
    return 0;
}
```

So the *unlock* fast/slow criterion is `oldval == 1` (no waiters) vs `oldval == 2` (waiters present, must wake). This is the entire futex protocol in three lines: the userspace word distinguishes "nobody waiting" from "somebody waiting," and only the latter pays for a syscall.

Mutex types layer extra stages on top:

- *`PTHREAD_MUTEX_ADAPTIVE_NP`* — inserts a spin prelude before the `exchange(2)`, bounded by `MAX_ADAPTIVE_COUNT` (~100 iterations). Same criterion as kernel optimistic spin in spirit, but glibc cannot read `owner->on_cpu`, so it spins blindly for a fixed count.
- *`PTHREAD_MUTEX_ERRORCHECK` / `RECURSIVE`* — fast path additionally compares `__owner == self`; cannot use the bare CAS because they need the owner TID.
- *`PTHREAD_MUTEX_PI_*` (priority inheritance)* — uses `FUTEX_LOCK_PI` / `FUTEX_UNLOCK_PI`, which routes through the kernel's `rt_mutex` and donates priority. The fast path becomes `cmpxchg(0, gettid())` (TID-valued, not 0/1/2).
- *`PTHREAD_MUTEX_ROBUST`* — adds a per-thread "robust list" so the kernel can mark the futex `FUTEX_OWNER_DIED` if the holder exits without unlocking.

*The summary criterion table.*

#table(
  columns: (auto, auto, auto),
  [*Primitive*], [*Fast-path test*], [*Slow-path trigger*],
  [Linux `mutex`], [`cmpxchg(owner, 0, current)`], [CAS fails → spin if `owner->on_cpu`, else sleep],
  [Linux `spinlock`], [queued `cmpxchg` on lock word], [CAS fails → MCS-queue spin (no sleep)],
  [Linux `rt_mutex`], [`cmpxchg(owner, 0, current)`], [CAS fails → PI-boost + sleep (no spin)],
  [glibc `pthread_mutex`], [`cmpxchg(lock, 0, 1)`], [CAS fails → `exchange(lock, 2)` + `FUTEX_WAIT`],
  [glibc adaptive], [same CAS], [CAS fails → spin N then futex],
  [`parking_lot::Mutex`], [`cmpxchg(state, 0, LOCKED)`], [fail → ~10 µs spin → park-table queue → `futex`],
  [Go `sync.Mutex`], [`cmpxchg(state, 0, mutexLocked)`], [fail → `procyield(30)` spin → `semacquire` → `gopark`],
  [`tokio::Mutex`], [`cmpxchg(state, 0, 1)`], [fail → waker list + `Poll::Pending` (no syscall)],
)

The pattern is identical across all of them: the fast path is *one CAS on a userspace (or kernel-memory) word*, and the slow-path entry point is the failure return of that CAS. Everything else — spin or not, queue where, syscall or not — is policy on top of the same branch.

== Reducing Mutex Contention

Once a mutex *does* contend (multiple threads spend non-trivial time queued behind it), the kernel-park overhead and lost parallelism dominate. Common techniques, in increasing order of complexity:

*1. Shorten the critical section.* The cheapest win — move every operation that doesn't *need* the lock outside it (computation, allocation, I/O). A critical section of 100 ns saturating 8 threads is fundamentally limited to 80 M ops/s.

*2. Reader-writer locks.* `std::shared_mutex` allows N readers concurrently but only one writer. Worth it when reads outnumber writes ~10:1 *and* the critical section is long enough to amortize the higher per-op cost (RW lock ≈ 2-3× a plain mutex uncontended).

*3. Lock sharding / striping.* Replace one lock with N locks indexed by `hash(key) % N`. Used by `ConcurrentHashMap` (Java), `tbb::concurrent_hash_map`. Cuts contention by N× if keys are uniform. Cost: must take *all* shards for global operations.

```cpp
struct ShardedMap {
    static constexpr int N = 64;
    std::array<std::mutex, N>             locks_;
    std::array<std::unordered_map<...>, N> maps_;

    void put(const K& k, const V& v) {
        size_t s = std::hash<K>{}(k) & (N - 1);
        std::lock_guard g(locks_[s]);
        maps_[s][k] = v;
    }
};
```

*4. Per-thread data + merge.* Each thread writes to its own slot (zero contention); a coordinator periodically merges. Used for counters, profilers, allocators (jemalloc/tcmalloc thread caches). Tradeoff: reads are expensive (must sum all slots) and stale.

*5. Lock-free atomics.* For simple state (counters, flags, single-pointer swaps), `std::atomic` with `fetch_add`/CAS avoids the lock entirely. Costs ~20-50 cycles uncontended; under contention, cache-line ping-pong limits scaling — ~10 M atomic increments/s/contended-line is a typical ceiling.

*6. Lock-free data structures.* SPSC/MPSC queues (see #emph[Lock-Free Data Structures]), Treiber stack, Michael-Scott queue. Real progress guarantee but require careful memory ordering and a reclamation scheme (hazard pointers, epoch-based reclamation) for ABA safety.

*7. RCU (Read-Copy-Update).* Readers run with zero synchronization; writers publish a new version and defer reclamation until all pre-existing readers finish. Used pervasively in the Linux kernel. Read side is ~free; write side is expensive and serialized.

*Tradeoffs to weigh:*

#table(
  columns: (auto, auto, auto),
  [*Approach*], [*Wins*], [*Costs*],
  [Shorten CS], [Always correct], [Sometimes impossible],
  [RW lock], [Read scaling], [Writer starvation; higher uncontended cost],
  [Sharding], [Linear scaling], [Memory N×; global ops expensive],
  [Per-thread + merge], [Zero write contention], [Stale reads; merge cost],
  [Atomics], [Simple state], [Cache-line contention caps throughput],
  [Lock-free DS], [Wait-free progress], [Memory reclamation hard; ABA hazards],
  [RCU], [Read-mostly nirvana], [Writer cost; deferred reclamation],
)

*Rule of thumb:* profile first. `perf c2c` (cache-to-cache analysis) and `perf lock` (lock contention) tell you whether the bottleneck is the lock itself, the *data* under the lock (false sharing, ping-pong), or something else entirely.

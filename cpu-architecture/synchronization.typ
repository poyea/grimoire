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

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

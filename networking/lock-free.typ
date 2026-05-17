= Lock-Free Data Structures for Networking

High-performance network applications require lock-free communication between threads to avoid mutex contention and context switch overhead.

*See also:* Concurrency Models (for thread architectures), Zero-Copy (for minimizing data movement), Message Queues (for ZeroMQ patterns)

== SPSC (Single Producer Single Consumer) Queue

*Most common pattern in network I/O:* One thread receives packets (producer), another processes them (consumer).

*Lock-free SPSC using ring buffer:*

```cpp
#include <atomic>
#include <cstddef>

template<typename T, size_t SIZE>
class SPSCQueue {
static_assert((SIZE & (SIZE - 1)) == 0, "SIZE must be power of 2");

private:
    struct alignas(64) {  // Separate cache lines to avoid false sharing
        std::atomic<size_t> head{0};
        char pad1[64 - sizeof(std::atomic<size_t>)];
    };

    struct alignas(64) {
        std::atomic<size_t> tail{0};
        char pad2[64 - sizeof(std::atomic<size_t>)];
    };

    T buffer[SIZE];

public:
    // Producer: Try to enqueue
    bool try_push(const T& item) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t next_head = (head + 1) & (SIZE - 1);

        // Check if queue full
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false;  // Queue full
        }

        buffer[head] = item;
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    // Consumer: Try to dequeue
    bool try_pop(T& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);

        // Check if queue empty
        if (tail == head_.load(std::memory_order_acquire)) {
            return false;  // Queue empty
        }

        item = buffer[tail];
        tail_.store((tail + 1) & (SIZE - 1), std::memory_order_release);
        return true;
    }

    // Non-blocking size (approximate)
    size_t size() const {
        size_t head = head_.load(std::memory_order_acquire);
        size_t tail = tail_.load(std::memory_order_acquire);
        return (head - tail) & (SIZE - 1);
    }

private:
    alignas(64) std::atomic<size_t> head_{0};
    char pad1_[64 - sizeof(std::atomic<size_t>)];

    alignas(64) std::atomic<size_t> tail_{0};
    char pad2_[64 - sizeof(std::atomic<size_t>)];
};
```

*Key design decisions:*

*1. Cache line alignment (64 bytes):*
```cpp
alignas(64) std::atomic<size_t> head_;
alignas(64) std::atomic<size_t> tail_;
```

Without alignment: Producer writes `head`, invalidates cache line containing `tail` → Consumer stalls on `tail` access = false sharing = 10-100x slower.

*2. Memory ordering:*
- `memory_order_relaxed`: No synchronization (local variable only)
- `memory_order_acquire`: Synchronize with release (read remote variable)
- `memory_order_release`: Make writes visible to acquire (write remote variable)

*Why not `memory_order_seq_cst`?* Sequential consistency requires full memory barriers (~10-20 cycles). Acquire/release are cheaper (~2-5 cycles) and sufficient for SPSC.

*3. Power-of-2 size:*
```cpp
next_index = (index + 1) & (SIZE - 1);  // Fast modulo
```
vs
```cpp
next_index = (index + 1) % SIZE;  // Division (~10-40 cycles on older CPUs)
```

== Performance Measurements

*Benchmark setup:* Producer thread sends 100M messages, consumer receives.

#table(
  columns: (auto, auto, auto),
  [*Configuration*], [*Latency (ns)*], [*Throughput (msgs/sec)*],
  [Mutex-based queue], [200-500], [5M],
  [SPSC (relaxed)], [40-80], [25M],
  [SPSC (acquire/release)], [20-40], [50M],
  [Shared memory (direct)], [10-15], [100M],
)

*Conclusion:* SPSC queue is 10x faster than mutex, 2-3x slower than direct shared memory (ideal case).

== Memory Ordering Deep Dive

*C++11 memory model [Boehm & Adve 2008]:*

```cpp
// Thread 1 (Producer)
buffer[head] = item;                           // (1) Store data
head_.store(next_head, memory_order_release);  // (2) Store head

// Thread 2 (Consumer)
size_t tail = tail_.load(memory_order_relaxed);     // (3) Load tail
if (tail != head_.load(memory_order_acquire)) {     // (4) Load head
    item = buffer[tail];                            // (5) Load data
}
```

*Happens-before relationship:*
- (1) happens-before (2) (same thread)
- (2) synchronizes-with (4) (release-acquire pair)
- (4) happens-before (5) (same thread)
- *Therefore: (1) happens-before (5)* → Consumer always sees Producer's data

*Without memory_order_release/acquire:* CPU/compiler may reorder (2) before (1) → Consumer reads uninitialized data!

*Hardware implementation [Intel SDM]:*
- x86-64: acquire/release are free (TSO model guarantees this)
- ARM: acquire = `ldr` + `dmb ishld` (~2-3 cycles), release = `dmb ish` + `str` (~2-3 cycles)
- Weaker architectures (POWER, RISC-V RMO): More expensive barriers

== Batch-Processing SPSC

*Problem:* Syscall per message → 500ns overhead → limits throughput.

*Solution:* Batch processing with SPSC queue.

```cpp
// Network thread (Producer)
void network_thread() {
    while (running) {
        struct epoll_event events[64];
        int n = epoll_wait(epollfd, events, 64, 0);  // Non-blocking

        for (int i = 0; i < n; i++) {
            Packet pkt;
            recv(events[i].data.fd, &pkt, sizeof(pkt), 0);

            while (!queue.try_push(pkt)) {
                // Queue full, spin briefly then yield
                _mm_pause();  // x86 hint to reduce contention
            }
        }
    }
}

// Processing thread (Consumer)
void processing_thread() {
    Packet batch[128];
    while (running) {
        size_t count = 0;

        // Drain queue in batch
        while (count < 128 && queue.try_pop(batch[count])) {
            count++;
        }

        if (count > 0) {
            process_batch(batch, count);  // Amortize per-packet overhead
        }
    }
}
```

*Performance:* Batching reduces per-packet overhead from 500ns → 50ns (10x improvement).

== MPSC (Multi-Producer Single Consumer) Queue

*Use case:* Multiple network threads enqueue to single processing thread.

*Challenge:* Multiple producers need coordination → requires atomic operations.

```cpp
template<typename T, size_t SIZE>
class MPSCQueue {
private:
    std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    T buffer[SIZE];

public:
    // Producer: Thread-safe enqueue (multiple producers)
    bool try_push(const T& item) {
        size_t head = head_.fetch_add(1, std::memory_order_relaxed);
        size_t next_head = (head + 1) & (SIZE - 1);

        // Check if queue full (racy, but conservative)
        if (next_head == tail_.load(std::memory_order_acquire)) {
            head_.fetch_sub(1, std::memory_order_relaxed);  // Rollback
            return false;
        }

        buffer[head & (SIZE - 1)] = item;

        // Wait for previous producers to finish
        while (committed_.load(std::memory_order_acquire) != head) {
            _mm_pause();  // Spin
        }

        committed_.store(head + 1, std::memory_order_release);
        return true;
    }

    // Consumer: Single consumer (same as SPSC)
    bool try_pop(T& item) {
        // Same as SPSC implementation
    }

private:
    std::atomic<size_t> head_{0};
    std::atomic<size_t> committed_{0};  // Tracks committed writes
    alignas(64) std::atomic<size_t> tail_{0};
};
```

*Cost:* `fetch_add` = atomic read-modify-write = 20-50 cycles (much more expensive than SPSC).

*Performance:* MPSC throughput = 10-20M msgs/sec (2-5x slower than SPSC).

== SPMC (Single Producer Multiple Consumer) Queue

*Use case:* One ingest thread fans out work to a pool of workers — packet steering, task distribution, log dispatch.

*Two design choices*, depending on whether consumers should *compete* for items or *all see every item*:

*1. Work-stealing (competing consumers — each item delivered once):*

```cpp
template<typename T, size_t SIZE>
class SPMCQueue {
    alignas(64) std::atomic<size_t> head_{0};   // Producer-only writes
    alignas(64) std::atomic<size_t> tail_{0};   // Consumers CAS to claim
    T buffer[SIZE];

public:
    bool try_push(const T& item) {                   // Single producer
        size_t h = head_.load(std::memory_order_relaxed);
        if (h - tail_.load(std::memory_order_acquire) >= SIZE) return false;
        buffer[h & (SIZE - 1)] = item;
        head_.store(h + 1, std::memory_order_release);
        return true;
    }

    bool try_pop(T& item) {                          // Multiple consumers
        size_t t = tail_.load(std::memory_order_relaxed);
        while (t < head_.load(std::memory_order_acquire)) {
            item = buffer[t & (SIZE - 1)];           // Speculative read
            if (tail_.compare_exchange_weak(t, t + 1,
                    std::memory_order_release,
                    std::memory_order_relaxed)) {
                return true;                          // Won the race
            }
            // Lost — `t` was reloaded by CAS, retry
        }
        return false;
    }
};
```

The speculative read before the CAS is *unsafe in general* because the producer could overwrite that slot. Safe only when `SIZE` is much larger than max in-flight (the producer can't lap the slowest consumer) or when the read is a trivially-copyable type and a stale value is acceptable.

*2. Broadcast / multicast (every consumer sees every item):*

Each consumer maintains its *own* `tail` cursor; the producer's `head` is shared. Slot can only be overwritten when the slowest consumer has passed it.

```
Producer head ──┐
                ▼
   [ slot0 ][ slot1 ][ slot2 ][ slot3 ][ slot4 ][ slot5 ]
      ▲                  ▲              ▲
      │                  │              │
   consumer_C         consumer_A     consumer_B
   (slowest — bounds producer)
```

This is the LMAX Disruptor model: producer publishes a sequence number per slot; each consumer spins on `seq[i & mask] >= my_cursor`. Tradeoff: producer stalls if any consumer falls behind.

*SPMC vs MPSC — symmetry is not free.*

The two are *not* mirror images. The hard side is whichever end has *multiple* threads contending:

#table(
  columns: (auto, auto, auto),
  [*Property*], [*MPSC*], [*SPMC*],
  [Contended end], [Producer (`fetch_add` on head)], [Consumer (CAS on tail)],
  [Producer cost], [10-50 cycles (atomic RMW)], [~5 cycles (relaxed store, no contention)],
  [Consumer cost], [~5 cycles (single reader)], [10-50 cycles (CAS, may retry)],
  [Typical use], [Per-thread event collection → 1 handler], [1 NIC RX thread → N workers],
  [Backpressure], [Producers retry/drop on full], [Consumers idle when empty],
)

If you find yourself wanting MPMC, prefer N independent SPSC queues (one per producer-consumer pair, or a sharded array) — MPMC ring buffers exist (e.g. `moodycamel::ConcurrentQueue`) but cost 3-5× SPSC even uncontended.

== Wait-Free vs Lock-Free

*Definitions [Herlihy & Shavit 2008]:*

- *Wait-free:* Every operation completes in bounded steps (strongest guarantee)
- *Lock-free:* System-wide progress guaranteed (at least one thread makes progress)
- *Obstruction-free:* Thread makes progress if running in isolation

*SPSC queue above:*
- Lock-free: If producer stalls, consumer still makes progress (and vice versa)
- NOT wait-free: Full queue → producer spins indefinitely

*Making SPSC wait-free:*
```cpp
// Overwrite oldest entry if queue full (ring buffer semantics)
bool push(const T& item) {
    size_t head = head_.load(std::memory_order_relaxed);
    buffer[head] = item;

    size_t next_head = (head + 1) & (SIZE - 1);
    head_.store(next_head, std::memory_order_release);

    // If overwrote consumer's entry, advance tail
    if (next_head == tail_.load(std::memory_order_acquire)) {
        tail_.store((next_head + 1) & (SIZE - 1), std::memory_order_release);
    }
    return true;  // Always succeeds
}
```

*Tradeoff:* Wait-free but loses messages under sustained overload.

== compare_exchange_weak vs compare_exchange_strong

Both forms of CAS take `(T& expected, T desired)` and atomically swap `desired` into the atomic if its current value equals `expected`, returning success. They differ in *spurious failure*:

- `compare_exchange_strong` only fails if the actual value differs from `expected`.
- `compare_exchange_weak` may *additionally* fail spuriously even when the value matches.

The reason is architecture, not whim. On x86-64, both compile to `LOCK CMPXCHG` — there is no spurious-failure mode, so `weak` and `strong` produce identical code. On ARM/POWER, atomic RMW is implemented with *load-linked / store-conditional* (`LDXR`/`STXR` on ARMv8, `LWARX`/`STWCX` on POWER). The store-conditional can fail for *any* reason that broke the reservation — interrupt, context switch, neighbor core's store to the same cache line, even unrelated loads on some microarchs. `weak` exposes that; `strong` hides it by retrying in a generated loop.

*Rule of thumb:*

```cpp
// Use weak when you're already in a CAS loop — the outer loop absorbs spurious failure
T old = atomic.load(std::memory_order_relaxed);
T desired;
do {
    desired = transform(old);
} while (!atomic.compare_exchange_weak(old, desired,
            std::memory_order_acq_rel,
            std::memory_order_relaxed));

// Use strong for one-shot CAS where retry isn't free / isn't expected
if (state.compare_exchange_strong(expected_idle, busy)) {
    // Took ownership — no retry path here
}
```

*Common mistakes:*

- `compare_exchange_weak` *outside* a loop. A spurious failure is then a *bug*, not a retry.
- Forgetting that `expected` is *updated by reference* on failure to the current value. The second iteration of a CAS loop should re-derive `desired` from the new `old`, not from the previous attempt.
- Mixing memory orders carelessly. The two-argument form takes one order for both success and failure; the four-argument form takes them separately, and the failure order must not be stronger than the success order (and must not be `release`/`acq_rel`).

*Cost on contention:* CAS doesn't queue — failed CAS retries hammer the same cache line, causing classic *thundering herd*. Under sustained contention, a CAS loop can scale negatively (more threads → lower throughput). Mitigations: backoff (`_mm_pause()`, exponential delay), or a queue-based lock (MCS lock) that linearizes the contenders.

== ABA Problem

*Problem in lock-free algorithms:*

```cpp
// Thread 1 reads head = A
ptr = head.load();

// Thread 2: Dequeue A, dequeue B, enqueue A (head = A again!)
// Thread 1: CAS succeeds (head == A), but it's different A!
if (head.compare_exchange_weak(ptr, ptr->next)) {
    // Incorrectly assumes same node
}
```

*Solutions:*

*1. Tagged pointers (x86-64):*
```cpp
struct TaggedPtr {
    uint64_t ptr : 48;   // Pointer (48-bit virtual address)
    uint64_t tag : 16;   // Version tag
};

std::atomic<TaggedPtr> head;

// Increment tag on every update
TaggedPtr new_head = {ptr->next, head.tag + 1};
head.compare_exchange_strong(old_head, new_head);
```

*2. Hazard pointers [Michael 2004]:*
Threads announce what they're accessing → prevent premature reuse.

*3. Epoch-based reclamation [Fraser 2004]:*
Defer deallocation until all threads finish current "epoch."

*SPSC queue is ABA-immune:* Single producer/consumer = no concurrent modifications.

== Practical Considerations

*When to use lock-free:*
1. High contention (many threads, short critical sections)
2. Real-time requirements (no priority inversion)
3. Throughput critical (avoid syscalls for mutex)

*When NOT to use lock-free:*
1. Complex data structures (trees, graphs) → error-prone
2. Low contention (mutex overhead < 100ns, simpler code)
3. Requires memory reclamation (ABA problem complexity)

*Network I/O pattern → SPSC is ideal:*
- Receive thread (producer) + processing thread (consumer)
- No ABA problem
- 10-50x faster than mutex
- Simple to implement correctly

== References

*Primary sources:*

Boehm, H-J. & Adve, S.V. (2008). "Foundations of the C++ Concurrency Memory Model." PLDI '08.

Herlihy, M. & Shavit, N. (2008). The Art of Multiprocessor Programming. Morgan Kaufmann.

Michael, M.M. (2004). "Hazard Pointers: Safe Memory Reclamation for Lock-Free Objects." IEEE Transactions on Parallel and Distributed Systems 15(6): 491-504.

Fraser, K. (2004). "Practical Lock-Freedom." PhD Thesis, University of Cambridge.

Intel Corporation (2023). Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 3A: System Programming Guide. Section 8.2 (Memory Ordering).

Preshing, J. (2012). "An Introduction to Lock-Free Programming." Preshing on Programming Blog. https://preshing.com/20120612/an-introduction-to-lock-free-programming/

*Benchmarks:*

Vyukov, D. (2012). "Bounded MPMC Queue." 1024cores.net. http://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue

= Part III: C++ Concurrency - std::thread, Mutex, Atomics & C++20 Synchronization

== Atomic Variables

=== std::atomic\<T\>

*Problem with plain variables:*
```cpp
int count = 0;
count++;  // NOT atomic! (read-modify-write race condition)
// Even volatile int does NOT help in C++ — volatile ≠ atomic
```

*Solution: std::atomic*
```cpp
#include <atomic>

std::atomic<int> count{0};
count.fetch_add(1);                    // Atomic increment, returns old value
count.fetch_add(1) + 1;               // Equivalent to Java incrementAndGet
count.fetch_add(5);                    // Atomic add 5
int expected = 10;
count.compare_exchange_strong(expected, 20);  // CAS: if count == 10, set to 20
```

*Common operations:*
```cpp
std::atomic<int> ai{0};

ai.load();                             // Read current value
ai.store(10);                          // Set value
ai.exchange(20);                       // Set and return old value

++ai;                                  // Atomic pre-increment
ai++;                                  // Atomic post-increment
--ai;                                  // Atomic pre-decrement
ai--;                                  // Atomic post-decrement

ai.fetch_add(5);                       // Returns old, then adds
ai.fetch_sub(3);                       // Returns old, then subtracts

int expected = 10;
ai.compare_exchange_strong(expected, 20);  // CAS operation
// If ai == 10: sets to 20, returns true
// If ai != 10: expected updated to actual, returns false
```

*std::atomic for pointers (lock-free stack):*
```cpp
struct Node {
    int value;
    Node* next;
};

std::atomic<Node*> head{nullptr};

// Lock-free stack push
void push(int value) {
    auto* new_node = new Node{value, nullptr};
    Node* old_head;
    do {
        old_head = head.load(std::memory_order_relaxed);
        new_node->next = old_head;
    } while (!head.compare_exchange_weak(
        old_head, new_node,
        std::memory_order_release,   // Success: publish new_node
        std::memory_order_relaxed    // Failure: just retry
    ));
}
```

*Note on memory orders:* C++ exposes the hardware memory model directly. Key orderings: `memory_order_relaxed` (no ordering), `memory_order_acquire` (loads after this see prior stores), `memory_order_release` (stores before this are visible), `memory_order_seq_cst` (default, total order — safest but slowest). On x86, acquire/release is essentially free (strong memory model), but matters on ARM/RISC-V.

=== Compare-And-Swap (CAS)

*CAS semantics:*
```cpp
// Pseudocode for compare_exchange_strong:
bool compare_exchange_strong(T& expected, T desired) {
    // Atomically:
    if (this->value == expected) {
        this->value = desired;
        return true;
    } else {
        expected = this->value;  // C++ updates expected!
        return false;
    }
}
```

*compare_exchange_weak vs compare_exchange_strong:*
- `compare_exchange_weak`: May spuriously fail (even if value == expected). Cheaper on LL/SC architectures (ARM, RISC-V). Use in loops.
- `compare_exchange_strong`: Never spuriously fails. Slightly more expensive on LL/SC. Use for single-shot CAS.

*Hardware support:*
- x86: `CMPXCHG` instruction (with `LOCK` prefix)
- ARM: `LDXR`/`STXR` (load-exclusive/store-exclusive, LL/SC)
- Atomic at CPU level (cache coherence protocol: MESI/MOESI)

*Lock-free counter:*
```cpp
std::atomic<int> counter{0};

void increment() {
    int old_value = counter.load(std::memory_order_relaxed);
    while (!counter.compare_exchange_weak(
        old_value, old_value + 1,
        std::memory_order_relaxed)) {
        // old_value is automatically updated on failure
    }
    // Or simply: counter.fetch_add(1, std::memory_order_relaxed);
}
```

*ABA problem:*
```cpp
// Thread 1: Reads A
Node* old = head.load();  // A

// Thread 2: Changes A → B → A
head.store(B);
head.store(A);  // Same pointer value!

// Thread 1: CAS succeeds (head == A)
head.compare_exchange_strong(old, C);  // Succeeds! But state changed through B
```

*ABA solution:* Tagged pointer (value + version counter packed together)
```cpp
// Pack version counter into unused pointer bits or use double-width CAS
struct TaggedPointer {
    Node* ptr;
    uintptr_t tag;  // Version counter
};

std::atomic<TaggedPointer> head;

// On x86-64: Use DWCAS (double-word compare-and-swap)
// via std::atomic with 128-bit struct (compiler-specific support)
// Or use the top 16 bits of 64-bit pointers (only 48 bits used on x86-64)

// Simpler approach: hazard pointers or epoch-based reclamation
// (avoids ABA by deferring memory reclamation)
```

*Alternative ABA solutions:*
1. *Hazard pointers:* Threads publish pointers they are using; reclamation deferred
2. *Epoch-based reclamation (RCU-style):* Defer freeing until all readers finish
3. *Avoid ABA:* Use indices into an array instead of raw pointers

=== Memory Orders & Cache Effects

*C++ memory order summary:*

#table(
  columns: (auto, auto, auto),
  [*Memory Order*], [*Guarantee*], [*x86 Cost*],
  [`relaxed`], [Atomicity only, no ordering], [Free (MOV)],
  [`acquire`], [Reads after this see prior stores], [Free (x86 TSO)],
  [`release`], [Stores before this visible to acquirers], [Free (x86 TSO)],
  [`acq_rel`], [Both acquire and release], [Free (x86 TSO)],
  [`seq_cst`], [Total order across all threads], [MFENCE or XCHG],
)

*False sharing:* When two atomic variables share a cache line (64 bytes), updates by different threads cause expensive cache line bouncing (MESI invalidations).

```cpp
// BAD: False sharing — counters on same cache line
struct Counters {
    std::atomic<int> counter_a;  // Same cache line!
    std::atomic<int> counter_b;  // Bounces on every update
};

// GOOD: Pad to separate cache lines
struct alignas(64) PaddedCounter {
    std::atomic<int> value;
};

PaddedCounter counter_a;
PaddedCounter counter_b;

// C++17: std::hardware_destructive_interference_size
// (gives the cache line size at compile time, if available)
```

=== Distributed Counter (LongAdder equivalent)

*std::atomic:* Single variable, CAS for updates — high contention degrades

```cpp
std::atomic<long> counter{0};
counter.fetch_add(1);  // CAS loop under the hood
```

*Sharded counter:* Multiple cells, reduces contention (C++ equivalent of Java's LongAdder)

```cpp
#include <atomic>
#include <thread>
#include <numeric>
#include <array>

// Pad each cell to its own cache line to avoid false sharing
struct alignas(64) Cell {
    std::atomic<long> value{0};
};

class ShardedCounter {
    static constexpr int NUM_CELLS = 16;
    std::array<Cell, NUM_CELLS> cells_;
    std::atomic<long> base_{0};

public:
    void increment() {
        // Hash thread ID to pick a cell
        auto idx = std::hash<std::thread::id>{}(
            std::this_thread::get_id()) % NUM_CELLS;
        cells_[idx].value.fetch_add(1, std::memory_order_relaxed);
    }

    long sum() const {
        long total = base_.load(std::memory_order_relaxed);
        for (auto& cell : cells_) {
            total += cell.value.load(std::memory_order_relaxed);
        }
        return total;
    }
};
```

*When to use:*
- *std::atomic\<long\>*: Need exact value frequently, low contention
- *ShardedCounter*: High contention, only need sum occasionally (e.g., statistics)

*Benchmark (approximate):*
```
Threads    std::atomic    ShardedCounter
1          10M ops/s      10M ops/s
2          8M ops/s       18M ops/s  (2x faster)
4          4M ops/s       35M ops/s  (8x faster)
8          2M ops/s       60M ops/s  (30x faster!)
```

=== Interview Questions: Atomics

*Q1: How does CAS work and what are its advantages?*

A: CAS (Compare-And-Swap) is an atomic operation: `compare_exchange_strong(expected, desired)`

*Semantics:*
```cpp
// Atomically (hardware-level):
if (value == expected) {
    value = desired;
    return true;
} else {
    expected = value;  // C++ updates expected on failure
    return false;
}
```

*Implementation:*
- x86: `LOCK CMPXCHG` instruction
- ARM: Load-Exclusive/Store-Exclusive (LL/SC)

*Advantages:*
1. *Lock-free*: No OS-level blocking (no context switches)
2. *Wait-free progress*: At least one thread always makes progress
3. *No deadlock*: No locks to acquire
4. *Low latency*: ~1--10ns vs ~100--1000ns for locks

*Usage pattern:*
```cpp
std::atomic<int> counter{0};

void increment() {
    int old_value = counter.load();
    while (!counter.compare_exchange_weak(old_value, old_value + 1)) {
        // old_value auto-updated; retry (optimistic concurrency)
    }
}
```

*Trade-off:* CAS can livelock under extreme contention (retry loops). Use locks for very high contention.

*Q2: When would you use a sharded counter instead of std::atomic\<long\>?*

A:

*std::atomic\<long\>:*
- Single variable with CAS updates
- *High contention* → performance degrades (cache line bouncing)
- Use when: Exact value needed frequently, low contention

*ShardedCounter:*
- Per-thread cells (each on its own cache line), reduces contention
- `sum()` traverses all cells (slower than single `load()`)
- Use when: High contention, sum needed occasionally

*Benchmark (8 threads):*
```
std::atomic:      2M ops/s  (cache line bouncing)
ShardedCounter:   60M ops/s (30x faster!)
```

*Decision:*
- Metrics/counters (high write, rare read) → ShardedCounter
- Sequence numbers (frequent read) → std::atomic
- Low contention → std::atomic (simpler)

*Q3: What is the ABA problem and how do you solve it?*

A: ABA problem: Value changes A → B → A, CAS succeeds but state changed.

*Example:*
```cpp
// Lock-free stack
std::atomic<Node*> head;  // Initially points to A

// Thread 1
Node* old_head = head.load();  // A
// ... preempted ...

// Thread 2 (interleaved)
// Pops A, pushes B, pushes A back
// head == A again, but stack structure changed!

// Thread 1
head.compare_exchange_strong(old_head, new_node);
// Succeeds! But missed that B was in between
```

*Solutions:*

1. *Tagged pointer (double-width CAS):*
```cpp
struct TaggedPtr {
    Node* ptr;
    uint64_t tag;  // Version counter
};
// Increment tag on every modification
// CAS succeeds only if both pointer AND tag match
```

2. *Hazard pointers:* Publish pointers in use; defer reclamation
3. *Epoch-based reclamation:* Readers register epochs; freed only when safe
4. *Avoid ABA:* Use array indices instead of pointers; use immutable data

*Q4: Explain C++ memory orders and when to use each.*

A: C++ exposes hardware memory ordering, unlike Java's simpler happens-before model.

*Memory orders:*
- `memory_order_relaxed`: Atomicity only. Use for counters where ordering does not matter.
- `memory_order_acquire`: On loads — subsequent reads/writes cannot be reordered before this. Use when reading a flag that guards data.
- `memory_order_release`: On stores — preceding reads/writes cannot be reordered after this. Use when publishing data via a flag.
- `memory_order_acq_rel`: Both acquire and release. Use for read-modify-write that both reads and publishes.
- `memory_order_seq_cst`: Total order. Default. Use when reasoning about ordering is hard.

*Pattern: release/acquire pair*
```cpp
std::atomic<bool> ready{false};
int data = 0;

// Thread 1 (producer)
data = 42;                                      // Non-atomic write
ready.store(true, std::memory_order_release);   // Publish

// Thread 2 (consumer)
while (!ready.load(std::memory_order_acquire))  // Synchronize
    ;
assert(data == 42);  // Guaranteed! Release-acquire establishes happens-before
```

== Explicit Locks

=== std::mutex and std::unique_lock

*Why std::unique_lock when we have std::lock_guard?*

*std::lock_guard limitations:*
- Cannot unlock and re-lock
- Cannot transfer ownership
- Cannot use with condition variables (need unlock during wait)
- Cannot try-lock

*std::unique_lock benefits:*
- `try_lock()`: Non-blocking attempt
- `try_lock_for(duration)`: Timed attempt (with `std::timed_mutex`)
- Can unlock and re-lock
- Works with `std::condition_variable`
- Movable (can transfer ownership)

```cpp
#include <mutex>

std::mutex mtx;

// Basic usage with lock_guard (RAII, simplest)
{
    std::lock_guard<std::mutex> lock(mtx);
    // Critical section — automatically unlocked at scope end
}

// unique_lock: More flexible
{
    std::unique_lock<std::mutex> lock(mtx);
    // Critical section
    lock.unlock();  // Can manually unlock
    // ... do non-critical work ...
    lock.lock();    // Re-lock
}

// Try lock without blocking
{
    std::unique_lock<std::mutex> lock(mtx, std::try_to_lock);
    if (lock.owns_lock()) {
        // Got lock
    } else {
        // Couldn't acquire lock
    }
}

// Try lock with timeout (requires timed_mutex)
std::timed_mutex timed_mtx;
{
    std::unique_lock<std::timed_mutex> lock(
        timed_mtx, std::chrono::seconds(1));
    if (lock.owns_lock()) {
        // Got lock within 1 second
    } else {
        // Timeout
    }
}

// scoped_lock (C++17): Lock multiple mutexes, deadlock-free
std::mutex mtx_a, mtx_b;
{
    std::scoped_lock lock(mtx_a, mtx_b);  // Acquires both, avoids deadlock
    // Critical section
}
```

*Fairness note:* C++ mutexes provide no fairness guarantees by default. Threads are woken in OS-dependent order (typically unfair for performance). For FIFO fairness, you must implement a ticket lock or use a fair queue.

```cpp
// Simple ticket lock for fairness
class TicketLock {
    std::atomic<uint64_t> next_ticket_{0};
    std::atomic<uint64_t> now_serving_{0};
public:
    void lock() {
        uint64_t my_ticket = next_ticket_.fetch_add(1);
        while (now_serving_.load(std::memory_order_acquire) != my_ticket) {
            // Spin (could add pause/yield)
        }
    }
    void unlock() {
        now_serving_.fetch_add(1, std::memory_order_release);
    }
};
```

=== std::shared_mutex (ReadWriteLock)

*Problem:* Multiple readers OK, but writer needs exclusive access.

```cpp
// With plain mutex: Only one reader at a time (inefficient!)
{
    std::lock_guard<std::mutex> lock(mtx);
    auto value = map[key];  // Read — but blocks all others!
}
```

*Solution: std::shared_mutex (C++17)*
```cpp
#include <shared_mutex>

std::shared_mutex rw_mutex;

// Multiple readers allowed simultaneously (shared lock)
{
    std::shared_lock<std::shared_mutex> lock(rw_mutex);
    auto value = map[key];  // Read
}

// Writer has exclusive access (unique lock)
{
    std::unique_lock<std::shared_mutex> lock(rw_mutex);
    map[key] = value;  // Write
}
```

*Lock states:*
- No lock: Readers and writers can acquire
- Shared lock held: More readers can acquire, writers blocked
- Exclusive lock held: All readers and writers blocked

*Use case:* Read-heavy workloads (cache, configuration)

*Example: Read-heavy cache*
```cpp
#include <shared_mutex>
#include <unordered_map>
#include <string>

class Cache {
    std::unordered_map<std::string, std::string> map_;
    mutable std::shared_mutex rw_mutex_;

public:
    std::string get(const std::string& key) const {
        std::shared_lock lock(rw_mutex_);
        auto it = map_.find(key);
        return it != map_.end() ? it->second : "";
    }

    void put(const std::string& key, const std::string& value) {
        std::unique_lock lock(rw_mutex_);
        map_[key] = value;
    }
};
```

=== Optimistic Locking with Sequence Counters (SeqLock)

*Problem with shared_mutex:* Writers can starve if readers keep coming. Also, shared_lock still has overhead (atomic increment on the lock itself).

*SeqLock (optimistic read):* Writer increments a sequence counter; readers check if counter changed.

```cpp
class SeqLock {
    std::atomic<uint64_t> seq_{0};  // Even = no writer, odd = writer active

public:
    uint64_t read_begin() const {
        uint64_t s;
        do {
            s = seq_.load(std::memory_order_acquire);
        } while (s & 1);  // Wait while writer active (odd)
        return s;
    }

    bool read_validate(uint64_t start_seq) const {
        std::atomic_thread_fence(std::memory_order_acquire);
        return seq_.load(std::memory_order_relaxed) == start_seq;
    }

    void write_lock() {
        seq_.fetch_add(1, std::memory_order_acquire);  // Odd → writing
    }

    void write_unlock() {
        seq_.fetch_add(1, std::memory_order_release);  // Even → done
    }
};
```

*Optimistic read pattern:*
```cpp
class Point {
    double x_, y_;
    SeqLock seq_;

public:
    double distance_from_origin() const {
        double cur_x, cur_y;
        uint64_t seq;
        do {
            seq = seq_.read_begin();       // Get sequence
            cur_x = x_;                    // Read (no lock!)
            cur_y = y_;                    // Read (no lock!)
        } while (!seq_.read_validate(seq)); // Retry if writer intervened
        return std::sqrt(cur_x * cur_x + cur_y * cur_y);
    }

    void move(double delta_x, double delta_y) {
        seq_.write_lock();
        x_ += delta_x;
        y_ += delta_y;
        seq_.write_unlock();
    }
};
```

*When to use:*
- Read-heavy workloads (90%+ reads)
- Reads very short (retry overhead small)
- Readers never block writers (unlike shared_mutex)

*Warning:* SeqLock reads may see torn values — only safe for trivially-copyable types (POD). Not safe for `std::string` or complex objects.

=== Condition Variables

*std::condition_variable:* Used with std::mutex for producer-consumer, signals.

```cpp
#include <mutex>
#include <condition_variable>

std::mutex mtx;
std::condition_variable not_full;
std::condition_variable not_empty;

// Producer
{
    std::unique_lock<std::mutex> lock(mtx);
    not_full.wait(lock, [&]{ return !is_full(); });  // Wait until not full
    // Add item
    not_empty.notify_one();  // Signal not empty
}

// Consumer
{
    std::unique_lock<std::mutex> lock(mtx);
    not_empty.wait(lock, [&]{ return !is_empty(); }); // Wait until not empty
    // Remove item
    not_full.notify_one();  // Signal not full
}
```

*Important:* Always use the predicate overload of `wait()` to guard against spurious wakeups. The condition variable atomically releases the mutex and suspends the thread; on wakeup it re-acquires the mutex and checks the predicate.

*Bounded buffer with condition variables:*
```cpp
#include <mutex>
#include <condition_variable>
#include <vector>
#include <optional>

template <typename T>
class BoundedBuffer {
    std::vector<T> buffer_;
    size_t capacity_, count_{0}, put_idx_{0}, take_idx_{0};
    std::mutex mtx_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;

public:
    explicit BoundedBuffer(size_t capacity)
        : buffer_(capacity), capacity_(capacity) {}

    void put(T item) {
        std::unique_lock<std::mutex> lock(mtx_);
        not_full_.wait(lock, [&]{ return count_ < capacity_; });
        buffer_[put_idx_] = std::move(item);
        put_idx_ = (put_idx_ + 1) % capacity_;
        ++count_;
        not_empty_.notify_one();
    }

    T take() {
        std::unique_lock<std::mutex> lock(mtx_);
        not_empty_.wait(lock, [&]{ return count_ > 0; });
        T item = std::move(buffer_[take_idx_]);
        take_idx_ = (take_idx_ + 1) % capacity_;
        --count_;
        not_full_.notify_one();
        return item;
    }
};
```

=== Interview Questions: Locks

*Q1: std::mutex vs std::shared_mutex — when to use which?*

A:

*std::mutex:*
- Exclusive access only
- Simpler, lower overhead per lock/unlock
- Use for short critical sections
- No reader parallelism

*std::shared_mutex:*
- Shared (read) + exclusive (write) modes
- Multiple readers can proceed in parallel
- Higher per-lock overhead (atomic counter for reader count)
- Use for read-heavy workloads

*When to use:*
- *std::mutex*: Default choice. Short critical sections, balanced read/write.
- *std::shared_mutex*: Read-heavy (10:1+ read-to-write ratio), long reads.

*Performance note:* On x86, `std::shared_mutex` shared lock/unlock does atomic increment/decrement — this bounces a cache line. For very short reads, plain `std::mutex` can be faster even with readers blocking each other.

*Q2: Explain RAII locking in C++ (lock_guard, unique_lock, scoped_lock).*

A:

*std::lock_guard (C++11):*
- Simplest RAII wrapper. Locks on construction, unlocks on destruction.
- Cannot unlock early, cannot use with condition variables.

*std::unique_lock (C++11):*
- Flexible: deferred locking, try-lock, timed lock, manual unlock/relock.
- Required for `std::condition_variable::wait()`.
- Movable (can transfer lock ownership).

*std::scoped_lock (C++17):*
- Locks multiple mutexes simultaneously without deadlock (uses `std::lock` internally).
- Replaces manual `std::lock()` + `std::lock_guard(adopt_lock)` pattern.

```cpp
// Deadlock-prone:
std::lock_guard<std::mutex> lock_a(mtx_a);
std::lock_guard<std::mutex> lock_b(mtx_b);  // If another thread locks b then a → deadlock!

// Deadlock-free (C++17):
std::scoped_lock lock(mtx_a, mtx_b);  // Acquires in consistent order
```

*Q3: Explain optimistic locking with sequence counters (SeqLock).*

A: Optimistic locking: Assume no conflicts, validate afterward.

*Pattern:*
1. Read sequence counter (must be even — no writer active)
2. Read data (NO lock held!)
3. Re-read sequence counter and compare
4. If unchanged: Success (fast path, no lock overhead)
5. If changed: Retry (writer intervened)

```cpp
uint64_t seq;
double x, y;
do {
    seq = seq_lock.read_begin();   // Read sequence (no lock)
    x = this->x_;                  // Read data (no lock!)
    y = this->y_;
} while (!seq_lock.read_validate(seq));  // Validate
```

*When to use:*
- Read-heavy (90%+ reads)
- Short reads (retry overhead small)
- Readers never block writers

*Benefit:* Readers hold no lock — writers are never blocked by readers.

*Downside:* Retry overhead if validation fails; only safe for trivially-copyable types.

== Concurrent Data Structures

=== Thread-Safe Hash Map

*Approach 1: Global lock (coarse-grained)*
```cpp
// Simple but poor concurrency
template <typename K, typename V>
class LockedMap {
    std::unordered_map<K, V> map_;
    mutable std::shared_mutex mtx_;
public:
    V get(const K& key) const {
        std::shared_lock lock(mtx_);    // Shared for reads
        auto it = map_.find(key);
        return it != map_.end() ? it->second : V{};
    }
    void put(const K& key, const V& value) {
        std::unique_lock lock(mtx_);    // Exclusive for writes
        map_[key] = value;
    }
};
```

*Approach 2: Striped locking (fine-grained, higher concurrency)*
```cpp
template <typename K, typename V, size_t NUM_STRIPES = 16>
class StripedMap {
    struct Stripe {
        std::unordered_map<K, V> map;
        mutable std::shared_mutex mtx;
    };
    std::array<Stripe, NUM_STRIPES> stripes_;

    Stripe& get_stripe(const K& key) {
        return stripes_[std::hash<K>{}(key) % NUM_STRIPES];
    }
    const Stripe& get_stripe(const K& key) const {
        return stripes_[std::hash<K>{}(key) % NUM_STRIPES];
    }

public:
    std::optional<V> get(const K& key) const {
        auto& stripe = get_stripe(key);
        std::shared_lock lock(stripe.mtx);
        auto it = stripe.map.find(key);
        if (it != stripe.map.end()) return it->second;
        return std::nullopt;
    }

    void put(const K& key, const V& value) {
        auto& stripe = get_stripe(key);
        std::unique_lock lock(stripe.mtx);
        stripe.map[key] = value;
    }

    // Atomic compute-if-absent (equivalent to Java's computeIfAbsent)
    V compute_if_absent(const K& key, std::function<V(const K&)> factory) {
        auto& stripe = get_stripe(key);
        std::unique_lock lock(stripe.mtx);
        auto [it, inserted] = stripe.map.try_emplace(key);
        if (inserted) {
            it->second = factory(key);
        }
        return it->second;
    }
};
```

*Why compute-style methods matter:*
```cpp
// Wrong: Race condition (check-then-act)
if (map.get(key) == std::nullopt) {
    map.put(key, value);  // Another thread may have inserted between check and put!
}

// Right: Atomic
map.compute_if_absent(key, [](auto& k) { return expensive_computation(k); });
```

*Approach 3: Lock-free (advanced)*
- Use Intel TBB `tbb::concurrent_hash_map` or `folly::ConcurrentHashMap`
- Lock-free reads, fine-grained locking for writes
- Best for high-throughput systems

=== Copy-On-Write Vector

*Strategy:* Every modification creates a new copy. Reads are lock-free.

```cpp
#include <shared_mutex>
#include <memory>
#include <vector>

template <typename T>
class CowVector {
    std::shared_ptr<const std::vector<T>> data_;
    mutable std::mutex write_mtx_;

public:
    CowVector() : data_(std::make_shared<const std::vector<T>>()) {}

    // Lock-free read (shared_ptr copy is thread-safe)
    std::shared_ptr<const std::vector<T>> snapshot() const {
        return std::atomic_load(&data_);
    }

    void push_back(const T& value) {
        std::lock_guard lock(write_mtx_);
        auto new_data = std::make_shared<std::vector<T>>(*data_);
        new_data->push_back(value);
        std::atomic_store(&data_,
            std::shared_ptr<const std::vector<T>>(std::move(new_data)));
    }
};
```

*Iterator:* Snapshot at creation, never invalidated

```cpp
CowVector<std::string> list;
list.push_back("a");
list.push_back("b");

auto snap = list.snapshot();  // Snapshot of current state

// Another thread modifies list
list.push_back("c");

// snap sees old data
for (const auto& s : *snap) {
    std::cout << s << "\n";  // Prints: a, b (not c)
}
```

*When to use:*
- Read-heavy (99%+ reads)
- Small containers (copy overhead acceptable)
- Iteration >> modification

*Anti-pattern:* Frequent modifications ($O(n)$ copy for each!)

=== Lock-Free Queue (Michael-Scott)

*Lock-free MPMC queue:* Based on Michael-Scott algorithm (CAS-based)

```cpp
#include <atomic>
#include <optional>

template <typename T>
class LockFreeQueue {
    struct Node {
        T data;
        std::atomic<Node*> next{nullptr};
        Node() = default;
        explicit Node(T val) : data(std::move(val)) {}
    };

    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;

public:
    LockFreeQueue() {
        auto* sentinel = new Node();
        head_.store(sentinel);
        tail_.store(sentinel);
    }

    void enqueue(T value) {
        auto* new_node = new Node(std::move(value));
        while (true) {
            Node* tail = tail_.load(std::memory_order_acquire);
            Node* next = tail->next.load(std::memory_order_acquire);
            if (next == nullptr) {
                if (tail->next.compare_exchange_weak(
                        next, new_node, std::memory_order_release)) {
                    tail_.compare_exchange_strong(
                        tail, new_node, std::memory_order_release);
                    return;
                }
            } else {
                // Tail fell behind; advance it
                tail_.compare_exchange_weak(
                    tail, next, std::memory_order_release);
            }
        }
    }

    std::optional<T> dequeue() {
        while (true) {
            Node* head = head_.load(std::memory_order_acquire);
            Node* tail = tail_.load(std::memory_order_acquire);
            Node* next = head->next.load(std::memory_order_acquire);
            if (next == nullptr) return std::nullopt;  // Empty
            if (head == tail) {
                tail_.compare_exchange_weak(
                    tail, next, std::memory_order_release);
                continue;
            }
            T value = next->data;
            if (head_.compare_exchange_weak(
                    head, next, std::memory_order_release)) {
                delete head;  // Caution: ABA — use hazard pointers in production
                return value;
            }
        }
    }

    ~LockFreeQueue() {
        while (dequeue().has_value()) {}
        delete head_.load();
    }
};
```

=== Blocking Queue

*BlockingQueue:* Queue with blocking operations — the workhorse of producer-consumer.

```cpp
#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <chrono>

template <typename T>
class BlockingQueue {
    std::queue<T> queue_;
    size_t capacity_;
    std::mutex mtx_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;

public:
    explicit BlockingQueue(size_t capacity) : capacity_(capacity) {}

    // Blocks if full
    void put(T item) {
        std::unique_lock lock(mtx_);
        not_full_.wait(lock, [&]{ return queue_.size() < capacity_; });
        queue_.push(std::move(item));
        not_empty_.notify_one();
    }

    // Blocks if empty
    T take() {
        std::unique_lock lock(mtx_);
        not_empty_.wait(lock, [&]{ return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }

    // Try with timeout
    std::optional<T> poll(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mtx_);
        if (!not_empty_.wait_for(lock, timeout,
                [&]{ return !queue_.empty(); })) {
            return std::nullopt;  // Timeout
        }
        T item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return item;
    }
};
```

*Variant implementations:*

*1. Bounded queue (above):* Fixed capacity, blocks producers when full.

*2. Unbounded queue:* No capacity limit (just remove the `not_full_` condition). Beware of memory growth.

*3. Priority queue:* Replace `std::queue` with `std::priority_queue`. Elements dequeued by priority, not FIFO.

*4. Synchronous handoff:* Capacity = 0. Producer blocks until a consumer takes. Useful for direct thread-to-thread handoff.

*Producer-Consumer with BlockingQueue:*
```cpp
BlockingQueue<Task> queue(100);

// Producer thread
void produce() {
    while (running) {
        Task task = create_task();
        queue.put(std::move(task));  // Blocks if queue full
    }
}

// Consumer thread
void consume() {
    while (running) {
        Task task = queue.take();  // Blocks if queue empty
        process(task);
    }
}
```

=== Interview Questions: Concurrent Data Structures

*Q1: How would you build a thread-safe hash map in C++ with high concurrency?*

A:

*Approach 1: Coarse-grained (std::shared_mutex)*
- Single shared_mutex for entire map
- Readers share lock, writers exclusive
- Simple but limited concurrency

*Approach 2: Striped locking*
- N stripes, each with own shared_mutex
- Hash key to stripe → only lock one stripe
- Concurrency level = N (typically 16--64)

*Approach 3: Lock-free (production)*
- Use `tbb::concurrent_hash_map` or `folly::ConcurrentHashMap`
- Lock-free reads, fine-grained write locking
- Best for extreme throughput

*Key insight:* Reads should never block. Use `std::shared_mutex` (shared lock for reads) or lock-free reads (`std::atomic` + careful memory ordering).

*Q2: When would you use a copy-on-write container?*

A: Use when:
1. *Read-heavy*: 99%+ reads, rare writes
2. *Small container*: Copy overhead acceptable
3. *Iteration >> modification*: Frequently iterate, rarely modify

*Example:* Observer/listener lists
```cpp
CowVector<std::function<void()>> listeners;

// Rare: Add listener
listeners.push_back(callback);  // O(n) copy (rare, acceptable)

// Frequent: Notify all (lock-free!)
auto snap = listeners.snapshot();
for (auto& fn : *snap) {
    fn();  // No locks held, no invalidation
}
```

*Anti-pattern:* Frequent modifications
```cpp
// BAD: O(n²) for n additions!
for (int i = 0; i < 1000; i++) {
    list.push_back(i);  // Each push copies entire vector!
}
```

*Q3: Compare mutex-based blocking queue vs lock-free queue.*

A:

*Blocking queue (mutex + condition_variable):*
- Simple, correct, well-understood
- Blocks threads when empty/full (saves CPU)
- Single lock can be bottleneck at high throughput
- Dual-lock variant (separate put/take locks) improves concurrency

*Lock-free queue (CAS-based, Michael-Scott):*
- No blocking — threads spin-retry on CAS failure
- Higher throughput under moderate contention
- Complex to implement correctly (ABA, memory reclamation)
- Wastes CPU when queue empty (spinning)

*When to use:*
- *Blocking queue*: Default choice. Simpler. Good when threads should sleep when idle.
- *Lock-free queue*: Latency-critical systems (trading, gaming). High throughput needed. Threads should not sleep.

*Performance:*
- Blocking: Better for bursty workloads (threads sleep during idle)
- Lock-free: Better for sustained high throughput (no OS scheduler overhead)

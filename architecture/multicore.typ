= Multicore and Cache Coherence

Modern CPUs contain multiple cores sharing resources. Understanding cache coherence, synchronization, and memory models is essential for correct and efficient parallel programming.

*See also:* Cache Hierarchy (for cache organization), Virtual Memory (for TLB coherence), Memory System (for bandwidth sharing)

== Multicore Architecture

*Typical server CPU (2023):*

```
┌─────────────────────────────────────┐
│  CPU Package                         │
│  ┌────────┐  ┌────────┐  ┌────────┐│
│  │ Core 0 │  │ Core 1 │  │ Core 2 ││
│  │├─L1 I/D │  │├─L1 I/D │  │├─L1 I/D ││
│  │└─L2    │  │└─L2    │  │└─L2    ││
│  └────┬───┘  └────┬───┘  └────┬───┘│
│       └───────────┴────────────┘    │
│              L3 Cache (Shared)      │
│       ┌────────────────────────┐    │
│       │  Memory Controller     │    │
│       └────────┬───────────────┘    │
└────────────────┼────────────────────┘
                 │
            ┌────┴────┐
            │  DRAM   │
            └─────────┘

Private: L1, L2 (32-512 KB per core)
Shared: L3 (8-64 MB across all cores)
```

*Resource sharing:*
- Private: L1/L2 cache, execution units, TLB
- Shared: L3 cache, memory controller, memory bandwidth

== Cache Coherence Problem

*Problem:* Multiple cores cache same memory location → inconsistent views.

```
Initial: Memory[X] = 0

Core 0: Load X → L1 cache[X] = 0
Core 1: Load X → L1 cache[X] = 0
Core 0: Store X = 1 → L1 cache[X] = 1
Core 1: Load X → L1 cache[X] = 0 ???  (Stale data!)
```

*Solution:* Cache coherence protocol ensures all cores see consistent data.

== MESI Protocol

*Cache line states [Papamarcos & Patel 1984]:*

```
M (Modified):   Dirty, exclusive (only this cache has it, modified)
E (Exclusive):  Clean, exclusive (only this cache has it, unmodified)
S (Shared):     Clean, shared (other caches may have it)
I (Invalid):    Not present or stale

State transitions:
- Read miss → E or S (depending on presence in other caches)
- Write → M (invalidate other caches)
- Snoop read from other core → S (share with other core)
- Snoop write from other core → I (invalidate)
```

*Example:*

```
Core 0       Core 1       Memory
Load X       -           [X]=0
E:[X]=0      I           [X]=0

Load X       Load X      [X]=0
S:[X]=0      S:[X]=0     [X]=0

Store X=1    -           [X]=0
M:[X]=1      I           [X]=0 (not yet written back)

-            Load X      [X]=0
S:[X]=1      S:[X]=1     [X]=1 (writeback on transition M→S)
```

== MOESI and MESIF Extensions

*MOESI (AMD):* Adds O (Owned) state.

```
O (Owned): Dirty but shared (one cache responsible for writeback)

Benefit: Sharing dirty data without writeback to memory
Example:
Core 0: M:[X]=1
Core 1: Load X
Result: Core 0: O:[X]=1, Core 1: S:[X]=1 (no memory write!)
```

*MESIF (Intel):* Adds F (Forward) state.

```
F (Forward): One cache designated to respond to requests (reduce bus traffic)
```

== Cache Coherence Overhead

*Bus snooping:* All caches monitor bus for relevant addresses.

```
Write to X in Core 0:
1. Core 0 broadcasts invalidation message
2. All other cores snoop address
3. If cached in S/E state → transition to I
4. Acknowledge invalidation
5. Core 0 proceeds with write

Cost: ~40-80 cycles (depending on core count and distance)
```

*Directory-based coherence (many-core):* Centralized directory tracks cache line owners (scales better than snooping).

== False Sharing

*Problem:* Two variables on same cache line → unnecessary coherence traffic.

```c
struct Counter {
    int counter1;  // Offset 0
    int counter2;  // Offset 4 (same 64-byte cache line!)
};

// Thread 0
for (...) counter1++;  // Write to cache line

// Thread 1
for (...) counter2++;  // Write to same cache line!

Effect:
- Thread 0 write → invalidate Thread 1's cache line
- Thread 1 write → invalidate Thread 0's cache line
- Ping-pong effect: 10-100× slower than separate cache lines
```

*Solution:* Align to cache line boundaries.

```c
struct Counter {
    alignas(64) int counter1;
    alignas(64) int counter2;
};
// Now each variable on separate cache line → no false sharing
```

== Atomic Operations

*Hardware support:* Atomically read-modify-write.

```c
#include <atomic>

std::atomic<int> counter(0);

// Atomic increment (lock-free on modern CPUs)
counter.fetch_add(1, std::memory_order_seq_cst);

// Compiled to:
lock add dword ptr [counter], 1

// LOCK prefix: Assert exclusive ownership of cache line (MESI M state)
// Cost: ~20-50 cycles (vs 1 cycle for non-atomic add)
```

*Compare-and-swap (CAS):*

```c
int expected = 0;
int desired = 1;
if (counter.compare_exchange_strong(expected, desired)) {
    // Success: counter was 0, now 1
} else {
    // Failure: counter was not 0, expected updated to actual value
}

// x86 assembly:
lock cmpxchg dword ptr [counter], desired
// If *counter == rax: *counter = desired, ZF=1
// Else: rax = *counter, ZF=0
```

== Memory Ordering

*Problem:* CPU/compiler may reorder memory operations for optimization.

```c
// Thread 1:
data = 42;
ready = true;

// Thread 2:
if (ready) {
    assert(data == 42);  // May fail without synchronization!
}

// Compiler/CPU may reorder Thread 1: ready=true before data=42
```

*C++11 memory model [Boehm & Adve 2008]:*

```c
// Relaxed: No ordering guarantees (fastest)
atomic.store(x, std::memory_order_relaxed);

// Acquire/Release: Synchronize with matching release/acquire (common)
atomic.store(x, std::memory_order_release);  // Release store
atomic.load(x, std::memory_order_acquire);    // Acquire load

// Sequentially consistent: Total order (slowest, default)
atomic.store(x, std::memory_order_seq_cst);
```

*x86-64 memory model:* Total Store Order (TSO) - strong ordering.

```
Reordering allowed:
- Store → Load (only reordering allowed!)
- Load → Load, Store → Store (not allowed on x86)

Effect: Acquire/release are nearly free on x86 (no extra barriers needed)
ARM: Weaker model, acquire/release require explicit barriers (~5 cycles)
```

== Memory Barriers

*Fence instructions:*

```c
// Full barrier (x86: mfence)
std::atomic_thread_fence(std::memory_order_seq_cst);

// Load barrier (x86: lfence)
std::atomic_thread_fence(std::memory_order_acquire);

// Store barrier (x86: sfence)
std::atomic_thread_fence(std::memory_order_release);

Cost: ~10-20 cycles (serialize pipeline, flush store buffer)
```

== Scaling and Amdahl's Law

*Amdahl's Law:* Speedup limited by serial portion.

```
Speedup = 1 / (S + P/N)

S = Serial fraction
P = Parallel fraction (S + P = 1)
N = Number of cores

Example: 95% parallel code, 16 cores
Speedup = 1 / (0.05 + 0.95/16) = 1 / 0.109 = 9.2×
(Not 16× due to 5% serial overhead!)
```

*Scaling bottlenecks:*
1. Synchronization overhead (locks, atomics)
2. Memory bandwidth saturation
3. Cache coherence traffic
4. Load imbalance

== NUMA (Non-Uniform Memory Access)

*Multi-socket systems:* Each socket has local memory.

```
┌─────────┐          ┌─────────┐
│ Socket 0│          │ Socket 1│
│ 8 cores │          │ 8 cores │
└────┬────┘          └────┬────┘
     │                    │
 ┌───┴───┐            ┌───┴───┐
 │Memory │◀──────────▶│Memory │
 │Node 0 │   Inter-   │Node 1 │
 │       │   connect  │       │
 └───────┘            └───────┘

Local access (Core 0 → Memory 0): ~200 cycles
Remote access (Core 0 → Memory 1): ~300-400 cycles (1.5-2× slower!)
```

*NUMA-aware programming:*

```c
// Pin thread to core
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(0, &cpuset);
pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

// Allocate memory on local node
numa_set_localalloc();  // or numa_alloc_onnode(node)
```

== Simultaneous Multithreading (SMT / Hyper-Threading)

*SMT:* Multiple hardware threads per core share execution units.

```
Single core with SMT (2 threads):
- 2 architectural register files
- 2 ROBs
- Shared: Caches, execution units, TLBs

Benefit: Hide latency - while Thread 1 waits for memory, Thread 2 executes
Speedup: 1.2-1.4× (not 2×, due to resource sharing)
```

*When SMT helps:*
- Memory-bound workloads (hide latency)
- Low ILP code (keep execution units busy)

*When SMT hurts:*
- Compute-bound (resource contention)
- Cache-sensitive (threads evict each other's data)

== References

Papamarcos, M.S. & Patel, J.H. (1984). "A Low-Overhead Coherence Solution for Multiprocessors with Private Cache Memories." ISCA '84.

Boehm, H-J. & Adve, S.V. (2008). "Foundations of the C++ Concurrency Memory Model." PLDI '08.

Hennessy, J.L. & Patterson, D.A. (2017). Computer Architecture: A Quantitative Approach (6th ed.). Morgan Kaufmann. Chapter 5 (Thread-Level Parallelism).

Drepper, U. (2007). "What Every Programmer Should Know About Memory." Red Hat, Inc. Section 6 (Multi-Processor Support).

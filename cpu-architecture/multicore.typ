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

Resources are divided between private and shared components. Each core has private L1/L2 caches, execution units, and TLB, while all cores share the L3 cache, memory controller, and memory bandwidth.

== Cache Coherence Problem

The cache coherence problem arises when multiple cores cache the same memory location, leading to inconsistent views of data. Consider the scenario where Memory[X] initially equals 0. Core 0 loads X into its L1 cache, Core 1 also loads X into its L1 cache, then Core 0 stores X = 1 in its L1 cache. When Core 1 subsequently loads X, it still reads 0 from its stale cache, creating an inconsistency.

```
Initial: Memory[X] = 0

Core 0: Load X → L1 cache[X] = 0
Core 1: Load X → L1 cache[X] = 0
Core 0: Store X = 1 → L1 cache[X] = 1
Core 1: Load X → L1 cache[X] = 0 ???  (Stale data!)
```

Cache coherence protocols solve this problem by ensuring all cores see consistent data.

== MESI Protocol

MESI is the canonical four-state cache-coherence protocol; real CPUs use extended variants (MESIF on Intel, MOESI on AMD — covered in the next section). The state semantics below are the common core.

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

False sharing occurs when two variables reside on the same cache line, generating unnecessary coherence traffic. Consider a structure where counter1 at offset 0 and counter2 at offset 4 share the same 64-byte cache line. When Thread 0 increments counter1 and Thread 1 increments counter2, each write invalidates the other thread's cache line, creating a ping-pong effect that can be 10-100× slower than if the variables were on separate cache lines.

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

The solution is to align each variable to cache line boundaries, ensuring each variable occupies its own cache line and eliminating false sharing:

```c
struct Counter {
    alignas(64) int counter1;
    alignas(64) int counter2;
};
// Now each variable on separate cache line → no false sharing
```

*Detecting false sharing in production:*

The classic signature is *high IPC drop + high HITM events*. Modern Intel (Haswell+) exposes precise hooks via PEBS:

```
# Per-line cache-to-cache analysis (Intel, Linux)
perf c2c record -a -- ./app
perf c2c report

# Look for:
#   - HITM rate > a few % of remote loads → likely false sharing
#   - "Cacheline" column shows the offending 64 B line
#   - "Data address" + "Offset" identify the contending fields

# AMD equivalent (Zen 3+): IBS
perf record -e ibs_op//p -- ./app
```

`perf stat -e cache-misses,cycle_activity.stalls_l3_miss,mem_load_l3_hit_retired.xsnp_hitm` gives a quick contention proxy without recording. The `HITM` (hit modified) event is the smoking gun: a load that finds the line in another core's L1/L2 in *Modified* state — i.e. someone is writing what you're reading.

*`std::hardware_destructive_interference_size` (C++17):* the language's answer to "what's the cache-line size for `alignas`?" Sounds clean; is messy. Implementations must pick a *single compile-time constant* that's an ABA-stable part of the platform target, so GCC reports 64 on most x86 targets while Apple Silicon would want 128 (M1's L2 line). Worse, changing the value breaks ABI for any class that embeds it as a layout choice. GCC emits a `-Winterference-size` warning if you use it across translation units. The Standard Library Working Group has discussed deprecation; in practice, hard-code 64 (or 128 for Apple Silicon / IBM POWER) and document it.

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

*`memory_order_consume` — the order you shouldn't use.*

The C++ standard defines a fifth order, `consume`, intended as a *cheaper acquire* for data-dependency-based synchronization: a load that orders only those subsequent accesses that *carry a dependency* from the loaded value (typical case: pointer-chasing where the loaded pointer is dereferenced). On DEC Alpha — the only mainstream architecture that reordered dependent loads — `consume` would require a fence; on ARM/POWER/x86, the hardware already preserves data-dependency ordering for free.

In practice:

- *Every major compiler implements `consume` as `acquire`.* GCC, Clang, MSVC have not shipped a real implementation since the order was introduced in C++11. P0190 proposed a redesign; P0750 proposed deprecation; the order remains in the standard but unimplemented.
- *Why it failed:* tracking "dependency chains" across optimization passes turned out to be intractable — the compiler routinely breaks dependencies it can't recognize as such (`if (p == known_const) use(known_const);` substitutes the constant, dropping the dependency on `p`).
- *Linux kernel uses it anyway,* via `rcu_dereference()` macros that hand-roll dependency preservation with `volatile` casts and `READ_ONCE`. This works only because the kernel is built with specific compilers and flags it controls.

*Recommendation:* use `acquire`. The notional performance win of `consume` doesn't materialize on real compilers, and writing portable code that relies on dependency ordering is currently impossible. Revisit if P2643 or a successor ever lands.

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

== Producer-Consumer: Where Atomics and Barriers Are Required

The canonical pattern: thread A *produces* a payload, then signals "ready"; thread B *waits* for the signal, then *reads* the payload. Correctness rests on the acquire/release rule from #emph[Memory Ordering] above — the release store on the flag synchronizes-with the acquire load on the same flag, making all prior producer writes visible to the consumer.

```cpp
// Shared
struct Message { int x, y; };
Message msg;                              // Plain (non-atomic) data
std::atomic<bool> ready{false};           // Synchronization flag

// Producer (thread A)
void produce() {
    msg.x = 42;                           // (1) plain write
    msg.y = 99;                           // (2) plain write
    ready.store(true,
        std::memory_order_release);       // (3) release: (1)(2) happen-before this
}

// Consumer (thread B)
void consume() {
    while (!ready.load(
        std::memory_order_acquire)) {     // (4) acquire: matches (3)
        _mm_pause();
    }
    assert(msg.x == 42 && msg.y == 99);   // Guaranteed if (4) saw (3)
}
```

*What goes wrong without correct ordering:*

#table(
  columns: (auto, auto),
  [*Mistake*], [*Failure mode*],
  [Both ops `relaxed`], [Consumer can read `ready==true` but `msg.x==0` — the producer's plain writes to `msg` may not be visible. On ARM/POWER this happens frequently; on x86 the hardware TSO usually hides it, masking the bug until you port.],
  [`msg` is a plain non-atomic but read concurrently], [Data race → UB. Even with correct ordering on `ready`, reading `msg` *before* `ready` was observed is a race.],
  [Release without matching acquire (or vice versa)], [No synchronizes-with edge. Compiler/CPU is free to reorder reads of `msg` before the load of `ready`.],
  [`seq_cst` everywhere "to be safe"], [Correct, but on x86 the release store now compiles to `XCHG`/`mfence` instead of a plain `mov` — ~20× slower. Acquire/release is sufficient here.],
)

*When you can downgrade to `relaxed`:*

- Counters that no one *synchronizes on* (e.g. statistics increments).
- The data side of an SPSC ring buffer when the head/tail pointers carry the acquire/release.

*Fence form:* a standalone `atomic_thread_fence(release)` (see #emph[Memory Barriers] above) lets many independent relaxed writes share one fence instead of attaching `release` to each.

*Architecture cheat sheet:*

#table(
  columns: (auto, auto, auto, auto),
  [*Model*], [*Acquire load*], [*Release store*], [*seq_cst store*],
  [x86-64 (TSO)], [plain `mov`], [plain `mov`], [`mov; mfence` or `xchg`],
  [ARMv8], [`ldar`], [`stlr`], [`stlr; dmb ish` (or `ldar` + `dmb`)],
  [POWER], [`ld; isync`], [`lwsync; st`], [`hwsync` on both sides],
)

This is why "it works on my x86 laptop, breaks on Graviton" is the canonical concurrency bug.

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

*Complete NUMA optimization example:*

```c
#include <numa.h>
#include <pthread.h>

struct thread_data {
    int node;
    float* data;
    int size;
};

void* worker(void* arg) {
    struct thread_data* td = (struct thread_data*)arg;

    // 1. Pin thread to NUMA node
    numa_run_on_node(td->node);
    numa_set_preferred(td->node);

    // 2. Allocate memory on local node
    td->data = numa_alloc_onnode(td->size * sizeof(float), td->node);

    // 3. First-touch policy: initialize on same thread
    for (int i = 0; i < td->size; i++) {
        td->data[i] = 0.0f;  // Page allocated on local NUMA node
    }

    // 4. Process data (local access, ~200 cycles)
    for (int i = 0; i < td->size; i++) {
        td->data[i] += compute(td->data[i]);
    }

    return NULL;
}

int main() {
    int num_nodes = numa_num_configured_nodes();
    pthread_t threads[num_nodes];
    struct thread_data td[num_nodes];

    for (int i = 0; i < num_nodes; i++) {
        td[i].node = i;
        td[i].size = 1000000;
        pthread_create(&threads[i], NULL, worker, &td[i]);
    }

    for (int i = 0; i < num_nodes; i++) {
        pthread_join(threads[i], NULL);
    }
}
// Result: All accesses local, optimal performance
```

*Detecting NUMA problems:*

```bash
# Check NUMA statistics
numastat
# High "numa_foreign" or "numa_miss" indicates remote access

# Profile NUMA access patterns
perf stat -e node-loads,node-load-misses,node-stores ./program

# Detailed per-node statistics
perf mem record -a ./program
perf mem report --sort=mem,symbol

# Visualize NUMA topology
lstopo  # from hwloc package
```

*Common NUMA anti-patterns:*

```c
// ANTI-PATTERN 1: Master thread allocates, workers access
float* data = malloc(SIZE);  // Allocated on node 0
#pragma omp parallel for
for (int i = 0; i < SIZE; i++) {
    data[i] = compute(data[i]);  // Threads on node 1 → remote access!
}

// FIX: First-touch allocation
float* data = malloc(SIZE);
#pragma omp parallel for
for (int i = 0; i < SIZE; i++) {
    data[i] = 0;  // Each thread touches its portion → local allocation
}
#pragma omp parallel for
for (int i = 0; i < SIZE; i++) {
    data[i] = compute(data[i]);  // Now local!
}

// ANTI-PATTERN 2: Shared queue (hot NUMA node)
queue_t* shared_queue = create_queue();  // All threads contend on node 0

// FIX: Per-NUMA-node queues
queue_t* queues[num_nodes];
for (int i = 0; i < num_nodes; i++) {
    queues[i] = numa_alloc_onnode(sizeof(queue_t), i);
}
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

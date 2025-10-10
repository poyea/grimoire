= Modern C++ Performance Reference

== Standard Library Containers

*Performance characteristics:*

```cpp
// SEQUENTIAL CONTAINERS
vector<T>         // Contiguous, best default choice. Random access O(1). Cache-friendly.
deque<T>          // Chunked (4KB blocks). Fast both ends. Slower than vector for iteration.
list<T>           // Doubly-linked. Avoid: pointer chasing = 10-100x slower than vector.
forward_list<T>   // Singly-linked. Use only if you need splice() and memory is tight.

// ASSOCIATIVE CONTAINERS
unordered_map<K,V>  // Hash table. O(1) avg. Load factor 0.75-0.9 optimal. Reserve capacity!
unordered_set<T>    // Hash table. Faster than map when no value needed.
map<K,V>            // Red-black tree. O(log n). Use only if sorted order required.
set<T>              // Red-black tree. Slower than unordered_set for large sets.

// PRIORITY QUEUE
priority_queue<T>   // Binary heap on vector. O(log n) push/pop. Cache-friendly.
```

*Critical optimizations:*
- *`reserve()`:* Pre-allocate to avoid reallocation. `vector` default = 2x growth = wasted copies.
- *`emplace_back()`* vs `push_back()`: Constructs in-place, avoids copy. $#sym.tilde.op$20-30% faster for complex types.
- *Flat containers:* For small sets/maps (< 64 elements), use `vector` with `sort()` + `binary_search()`. 5-10x faster than `set`/`map`.

== Algorithm Complexity

*Sorting:*
```cpp
sort(v.begin(), v.end());              // Introsort: O(n log n). Excellent cache locality.
stable_sort(v.begin(), v.end());       // O(n log n) space. Slower than sort().
partial_sort(v.begin(), v.begin()+k, v.end());  // O(n log k). Faster when k << n.
nth_element(v.begin(), v.begin()+k, v.end());   // O(n). Partial quickselect.
```

*Custom comparator (zero overhead):*
```cpp
sort(v.begin(), v.end(), [](int a, int b) { return a > b; });  // Descending
```
Lambda inlined by compiler. No function call overhead.

*Searching:*
```cpp
// On sorted data:
binary_search(v.begin(), v.end(), val);     // O(log n). Returns bool.
lower_bound(v.begin(), v.end(), val);       // O(log n). Returns iterator.
upper_bound(v.begin(), v.end(), val);       // O(log n). Returns iterator.

// Linear search (use for small n < 64):
find(v.begin(), v.end(), val);              // O(n). Sequential = prefetcher works.
```

== String Performance

```cpp
// AVOID: Creates temporary strings (slow)
string s = "hello" + "world";               // Error: can't add two literals
string s = string("hello") + "world";       // Creates temporary

// PREFER: Reserve capacity
string s;
s.reserve(100);
for (...) s += ch;

// BEST: Use string_view (C++17) for read-only
string_view sv = s;  // Zero-copy, just pointer + length

// String search:
s.find("pattern");              // O(nm) naive. Slow for long patterns.
// For repeated searches: use std::search with Boyer-Moore searcher
```

== Hash Table Performance

*Load factor:* Ratio of elements to buckets. Default = 1.0 (one element per bucket avg).

```cpp
unordered_map<int, int> map;
map.reserve(10000);  // Pre-allocate buckets. Avoids rehashing.
map.max_load_factor(0.75);  // Lower = less collisions, more memory
```

*Hash function quality:*
- Default `std::hash<int>`: identity function (perfect for sequential keys)
- Default `std::hash<string>`: fast but not cryptographic
- Custom hash: use `boost::hash_combine` or `std::hash<T>{}(x) ^ (std::hash<U>{}(y) << 1)`

*Collision resolution:* C++ uses chaining (linked lists). Robin Hood hashing (open addressing) = 2-3x faster but not in STL. Use `tsl::robin_map` if needed.

== Memory Alignment

*Cache line = 64 bytes.* Avoid false sharing:

```cpp
struct Foo {
    alignas(64) int counter1;  // Separate cache lines
    alignas(64) int counter2;  // Avoids false sharing in multithreading
};
```

*SIMD alignment:* AVX2 requires 32-byte alignment, AVX-512 requires 64-byte.

```cpp
alignas(32) int data[8];  // Aligned for _mm256 operations
```

== Bit Manipulation

```cpp
__builtin_popcount(x);     // Count set bits. Single POPCNT instruction if enabled
__builtin_clz(x);          // Count leading zeros. LZCNT/BSR instruction.
__builtin_ctz(x);          // Count trailing zeros. TZCNT/BSF instruction.
__builtin_parity(x);       // Parity (XOR of all bits).

// Bit scan: find lowest set bit
int lowestBit = x & -x;    // Isolate lowest set bit
int pos = __builtin_ctz(x); // Position of lowest set bit
```

*Compiler intrinsics:* Compile to CPU instructions with `-march=native` or `-mpopcnt`/`-mlzcnt`. Typically 1-3 cycle latency vs $#sym.tilde.op$10-20 cycles for loop implementation. Without target flags, may compile to loops.

== Iterators & Ranges

```cpp
// C++17 structured bindings:
for (auto [key, val] : map) { ... }  // Cleaner than pair<K,V>

// C++20 ranges (when available):
auto even = nums | views::filter([](int x) { return x % 2 == 0; });
```

== Avoid Common Pitfalls

```cpp
// BAD: Creates copy
for (auto x : vec) { ... }

// GOOD: Reference (no copy)
for (const auto& x : vec) { ... }

// Division by compile-time constant: optimized automatically
int half = n / 2;  // Compiler converts to n >> 1 automatically

// Variable division: slower on some older CPUs ($#sym.tilde.op$10-20 cycles)
int result = n / divisor;  // Modern x86: $#sym.tilde.op$3-10 cycles depending on CPU

// Modulo by power-of-2: optimized automatically
int rem = n % 8;  // Compiler converts to n & 7 automatically

// Manual optimization rarely needed - trust your compiler with -O2/-O3
```

== Profiling & Benchmarking

*CPU performance counters:* Use `perf stat` (Linux) or Intel VTune.

```bash
perf stat -e cache-misses,branch-misses ./program
```

*Key metrics:*
- *IPC (Instructions Per Cycle):* > 2.0 = good. < 1.0 = memory-bound or branch mispredicts.
- *Cache miss rate:* L1 < 5%, L2 < 10%, L3 < 20% = good.
- *Branch miss rate:* < 5% = good. > 10% = problematic.

*Microbenchmarking:* Use Google Benchmark library. Prevents compiler optimizing away code.

```cpp
static void BM_Sort(benchmark::State& state) {
    for (auto _ : state) {
        vector<int> v = generateData();
        benchmark::DoNotOptimize(v);
        sort(v.begin(), v.end());
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_Sort);
```

== System-Level Performance

*Virtual memory & paging:*
- Default page size: 4KB (x86-64)
- Page table walk: 4-level on x86-64 = 4 memory accesses ($#sym.tilde.op$800 cycles without TLB)
- TLB (Translation Lookaside Buffer): L1 TLB = 64 entries, L2 TLB = 1024-2048 entries
- TLB miss penalty: $#sym.tilde.op$20-100 cycles (page table walk)
- Huge pages (2MB/1GB): 512x/262144x fewer TLB entries, better for large datasets

*Memory allocation costs:*
```cpp
// Small allocations (<= 256 bytes)
int* p = new int[64];  // tcmalloc thread cache: ~10-20 cycles
delete[] p;            // ~10-20 cycles

// Large allocations (> 32KB)
int* p = new int[10000];  // mmap syscall: ~1000+ cycles
delete[] p;               // munmap syscall: ~1000+ cycles

// Stack allocation (fastest)
int arr[1000];  // ~0 cycles (just adjust stack pointer)
```

*Modern allocator internals:*
- tcmalloc/jemalloc: thread-local caches eliminate lock contention
- Size classes: reduce fragmentation (16, 32, 48, 64, 80... bytes)
- Allocation overhead: 8-16 bytes metadata per block
- Bulk deallocation: free list batching = amortized O(1)

*Page faults:*
```cpp
// Minor fault (page in RAM, not mapped): ~5000 cycles
char* p = (char*)mmap(NULL, 1GB, ...);
p[0] = 1;  // First access triggers fault

// Major fault (load from disk): ~1-10 million cycles
// Demand paging: OS loads on access, not allocation
```

*Huge pages configuration:*
```cpp
// Transparent Huge Pages (THP) - automatic
// /sys/kernel/mm/transparent_hugepage/enabled

// Explicit huge pages
void* p = mmap(NULL, 2MB, PROT_READ|PROT_WRITE,
               MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, -1, 0);
// Benefit: 2MB page = 1 TLB entry vs 512 entries for 4KB pages
```

*System call overhead:*
- syscall cost: $#sym.tilde.op$50-150 cycles (user→kernel context switch)
- vDSO (virtual dynamic shared object): $#sym.tilde.op$5-10 cycles (no kernel transition)
- vDSO functions: `gettimeofday()`, `clock_gettime()`, `getcpu()`

```cpp
#include <time.h>
timespec ts;
clock_gettime(CLOCK_MONOTONIC, &ts);  // vDSO: ~10 cycles, not syscall
```

*Cache hierarchy timing:*
- L1 access: $#sym.tilde.op$4-5 cycles (32-48KB data + 32KB instruction)
- L2 access: $#sym.tilde.op$12-15 cycles (256KB-512KB private per core)
- L3 access: $#sym.tilde.op$40-75 cycles (8-32MB shared across cores)
- RAM access: $#sym.tilde.op$200 cycles (DDR4/DDR5)
- Cache line: 64 bytes (fetch entire line on miss)

*NUMA (Non-Uniform Memory Access):*
```cpp
// Local node memory: ~200 cycles
// Remote node memory: ~300-400 cycles (cross-socket)
// Use numactl to bind process to node:
// numactl --cpunodebind=0 --membind=0 ./program
```

== Compiler Optimization Deep Dive

*Optimization levels:*
- `-O0`: No optimization, fast compile, debuggable
- `-O1`: Basic optimization, some inlining
- `-O2`: Recommended default, aggressive inlining, vectorization attempts
- `-O3`: `-O2` + loop unrolling, function cloning, more aggressive opts
- `-Os`: Optimize for size (may be faster due to better i-cache usage)
- `-Ofast`: `-O3` + fast-math (breaks IEEE 754, use with caution)

*Auto-vectorization (SIMD):*
```cpp
// Compile with: g++ -O3 -march=native -fopt-info-vec
void add(int* a, int* b, int* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
// Auto-vectorizes to: AVX2 = 8 int32 per cycle, AVX-512 = 16 int32 per cycle

// Requirements for auto-vectorization:
// 1. Countable loop (known bounds)
// 2. No pointer aliasing (use __restrict)
// 3. Aligned data (alignas(32) or alignas(64))
// 4. No function calls in loop
// 5. Simple data dependencies

// Example with restrict:
void add(int* __restrict a, int* __restrict b, int* __restrict c, int n) {
    // Tells compiler: a, b, c don't overlap → enables vectorization
}
```

*Profile-Guided Optimization (PGO):*
```bash
# Step 1: Compile with instrumentation
g++ -O3 -fprofile-generate program.cpp -o program

# Step 2: Run with representative workload
./program < typical_input.txt

# Step 3: Recompile with profile data
g++ -O3 -fprofile-use program.cpp -o program_optimized

# Benefits:
# - Better branch prediction (hot/cold path layout)
# - Improved inlining decisions (inline hot functions)
# - Better register allocation
# - Result: 10-30% speedup typical
```

*Link-Time Optimization (LTO):*
```bash
g++ -O3 -flto file1.cpp file2.cpp -o program
# Whole-program optimization:
# - Cross-file inlining
# - Dead code elimination across files
# - Devirtualization (static resolution of virtual calls)
# - Cost: slower compile time, 5-15% runtime speedup
```

*Branch optimization:*
```cpp
// Likely/unlikely hints
if (__builtin_expect(ptr != nullptr, 1)) {  // Likely true
    // Hot path
}

// C++20 attributes:
if (x > 0) [[likely]] {
    // Compiler places this code first
} else [[unlikely]] {
    // Cold path moved away
}
```

*Loop optimizations:*
```cpp
// Loop unrolling (automatic with -O3)
for (int i = 0; i < n; i++) {
    sum += arr[i];
}
// Becomes (4x unroll):
for (int i = 0; i < n; i += 4) {
    sum += arr[i] + arr[i+1] + arr[i+2] + arr[i+3];
}

// Manual unroll for specific count:
#pragma unroll 8
for (int i = 0; i < n; i++) { ... }

// Prevent unrolling if harmful:
#pragma nounroll
for (int i = 0; i < n; i++) { ... }
```

*Inlining control:*
```cpp
__attribute__((always_inline)) inline int fastFunc() { ... }
__attribute__((noinline)) int debugFunc() { ... }

// Force inline small functions in hot paths
// Prevent inline for cold/large functions to reduce i-cache pressure
```

*Floating-point optimization:*
```cpp
// -ffast-math enables:
// - Reassociation: (a+b)+c → a+(b+c)
// - No NaN/Inf checks
// - Reciprocal approximation: x/y → x*(1/y)
// - Fused multiply-add: a*b+c → fma(a,b,c)

// Safer alternative: -fno-math-errno -fno-trapping-math
```

== Advanced Performance Topics

*Move semantics & RVO:*
```cpp
// RVO (Return Value Optimization) - automatic, zero cost
vector<int> create() {
    vector<int> v(1000);
    return v;  // No copy, no move - direct construction in caller
}

// NRVO (Named RVO)
vector<int> create() {
    vector<int> result;
    result.reserve(1000);
    // ... populate result
    return result;  // Often elided, check with -fno-elide-constructors
}

// Move: O(1) - swap 3 pointers (~10-20 cycles)
vector<int> v1 = create();
vector<int> v2 = std::move(v1);  // v1 now empty, v2 owns data

// Copy: O(n) - deep copy (~1000s of cycles for large containers)
```

*Memory order & atomics:*
```cpp
#include <atomic>

atomic<int> counter(0);

// Relaxed: no synchronization (~1-2 cycles)
counter.fetch_add(1, memory_order_relaxed);

// Acquire/Release: synchronizes with other threads (~5-10 cycles)
counter.store(42, memory_order_release);
int x = counter.load(memory_order_acquire);

// Seq_cst: total ordering, slowest (~10-20 cycles)
counter.fetch_add(1, memory_order_seq_cst);  // Default
```

*False sharing:*
```cpp
// BAD: Multiple threads update adjacent counters
struct Counters {
    atomic<int> count1;  // Same cache line
    atomic<int> count2;  // Ping-pong between cores = 10-100x slower
};

// GOOD: Separate cache lines (64 bytes)
struct Counters {
    alignas(64) atomic<int> count1;
    alignas(64) atomic<int> count2;
};
// Each counter on own cache line → no false sharing
```

*Prefetching hints:*
```cpp
// Software prefetch (for random access patterns)
__builtin_prefetch(&data[next_index], 0, 3);
// Params: address, write(1)/read(0), temporal locality(0-3)

// Temporal locality:
// 0 = low (single use)
// 3 = high (reuse soon)

// Example: linked list prefetch 2-3 nodes ahead
Node* curr = head;
if (curr->next) __builtin_prefetch(curr->next->next);
```

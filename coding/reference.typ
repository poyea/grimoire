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
- *`emplace_back()`* vs `push_back()`: Constructs in-place, avoids copy. ~20-30% faster for complex types.
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
__builtin_popcount(x);     // Count set bits. Single instruction (POPCNT).
__builtin_clz(x);          // Count leading zeros. BSR instruction.
__builtin_ctz(x);          // Count trailing zeros. BSF instruction.
__builtin_parity(x);       // Parity (XOR of all bits).

// Bit scan: find lowest set bit
int lowestBit = x & -x;    // Isolate lowest set bit
int pos = __builtin_ctz(x); // Position of lowest set bit
```

*Compiler intrinsics:* Direct CPU instructions. ~1 cycle vs ~20 cycles for loop.

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

// BAD: Division (slow, ~10-40 cycles)
int half = n / 2;

// GOOD: Shift (fast, 1 cycle)
int half = n >> 1;  // Only for power-of-2 divisors

// BAD: Modulo (slow, ~10-40 cycles)
int rem = n % 8;

// GOOD: Bitwise AND (fast, 1 cycle)
int rem = n & 7;  // Only for power-of-2 modulus
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

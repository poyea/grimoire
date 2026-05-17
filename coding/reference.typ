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
Lambda inlined by compiler with `-O2`/`-O3`. No function call overhead [GCC/Clang inline small lambdas automatically].

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

*Misaligned access — architecture differences:*

- *x86/x86-64:* Tolerates misaligned scalar loads/stores. Cost is "free" if access stays within a cache line (64 B); crossing a cache line costs ~1-2 extra cycles (a *split load*). Crossing a 4 KB page is far worse (~100+ cycles on older microarchs; mostly absorbed by the LSU on Skylake+).
- *ARMv7/older ARM:* Trap on misaligned access unless `SCTLR.A=0` enables hardware fixup. With fixup the CPU silently does two aligned accesses — slow but works.
- *ARMv8 (AArch64):* Allows misaligned accesses to normal memory, but Device memory still faults. Atomics and exclusives (`LDXR`/`STXR`) require natural alignment — misaligned faults.
- *SPARC, MIPS, RISC-V (default):* Bus error / `SIGBUS` on misaligned access. Compiler-emitted `memcpy` is the portable escape hatch.

```cpp
// Always-safe misaligned read (compiler emits the right thing per target):
uint32_t value;
std::memcpy(&value, ptr, sizeof(value));   // No UB even if ptr misaligned
```

The standard says reading through a misaligned `T*` is *undefined behavior* regardless of architecture — the runtime cost above only applies to UB that happens to work in practice. UBSan with `-fsanitize=alignment` will flag it.

== Negative Zero

C++ inherits IEEE 754: `float`, `double`, and `long double` have two zero representations — `+0.0` and `-0.0`. Integer types do not (two's complement has a single zero).

```cpp
double a = -0.0;
double b =  0.0;

a == b;                   // true  (compares as equal)
std::signbit(a);          // true
std::signbit(b);          // false
1.0 / a;                  // -inf
1.0 / b;                  // +inf
a + b;                    //  0.0  (positive — IEEE 754 default rounding rule)
a * b;                    // -0.0
std::memcmp(&a, &b, 8);   // != 0 (bit patterns differ: sign bit)
```

*Where it bites:*
- Hash maps keyed on `double` — `std::hash` typically hashes the bit pattern, so `+0.0` and `-0.0` end up in *different buckets* despite `==`. Either normalize (`x + 0.0`) or use a custom hash.
- `std::set`/`std::map` with default `operator<` are fine (`+0.0` and `-0.0` are equivalent under `<`).
- Branch on `signbit` rather than `x < 0` if you need to distinguish them.

== std::tie

`std::tie(a, b, ...)` returns a `std::tuple<T&...>` of *lvalue references* to its arguments. Three common uses:

```cpp
// 1. Unpack a tuple/pair return value (pre-C++17, before structured bindings)
int x, y;
std::tie(x, y) = compute_pair();   // assigns into x and y

// 2. Ignore parts of a tuple
std::tie(x, std::ignore) = compute_pair();

// 3. Lexicographic comparison without writing it out
struct Date { int year, month, day; };
bool operator<(const Date& a, const Date& b) {
    return std::tie(a.year, a.month, a.day) < std::tie(b.year, b.month, b.day);
}
```

C++17 structured bindings (`auto [x, y] = compute_pair();`) replace use case (1). `std::tie` is still the idiomatic choice for (3) — it builds the comparison from member references with zero copies. Note: `std::tie` cannot bind to rvalues, so it can't be used on temporaries that need lifetime extension.

== std::shared_ptr Thread-Safety

The C++ standard (`[util.smartptr.shared]`) gives `std::shared_ptr` a *split* thread-safety guarantee that surprises almost everyone:

- The *control block* (reference counts, deleter, weak count) is *internally synchronized*. Multiple threads can copy, destroy, and move *distinct `shared_ptr` instances* that share the same control block without external locking.
- The *managed object* is *not* synchronized. Accessing `*p` or `p->x` from multiple threads follows the normal data-race rules for `T`.
- A *single `shared_ptr` object* (the same instance, not a copy) accessed from multiple threads — one reads it, another reassigns it — is a data race on the pointer + control-block pointer pair, unless every access goes through `std::atomic<std::shared_ptr<T>>` (C++20) or the now-deprecated free-function `std::atomic_load(&sp)` family.

```cpp
std::shared_ptr<Widget> global = std::make_shared<Widget>();

// SAFE: distinct shared_ptr instances, shared control block
void thread_a() { auto local = global; use(*local); }   // copy ctor → atomic inc
void thread_b() { auto local = global; use(*local); }

// UNSAFE: same shared_ptr instance read + written concurrently
void writer() { global = std::make_shared<Widget>(); }   // copy-assign
void reader() { auto local = global; }                   // copy ctor — races with writer

// SAFE (C++20): atomic shared_ptr
std::atomic<std::shared_ptr<Widget>> atomic_global;
void writer() { atomic_global.store(std::make_shared<Widget>()); }
void reader() { auto local = atomic_global.load(); use(*local); }
```

*Why this matters:*

The mental model "shared_ptr is thread-safe" is wrong in *both* directions. People assume too much (sharing the same `shared_ptr` instance is unsafe) and too little (copying through it doesn't need a lock — the refcount is atomic).

*Performance:* the atomic refcount increment on copy costs ~20 cycles uncontended and turns into cache-line ping-pong under contention. Hot paths that copy `shared_ptr` per access — common in DI/framework code — can scale *worse* than the raw work because every copy invalidates the control block's cache line across cores. Mitigations: pass `const shared_ptr&` (no refcount touch), `std::shared_ptr<const T>` with `make_shared` (control block adjacent to object → one cache line instead of two), or switch to `intrusive_ptr` with a thread-local cache.

*`weak_ptr::lock()`:* atomic CAS on the strong count. Same cost profile as a copy. Returns empty `shared_ptr` if the strong count is already zero.

*Aliasing constructor pitfall:* `shared_ptr<U>(other_sp, raw_u_ptr)` shares `other_sp`'s control block but exposes a different pointer. The control block still owns the original; lifetime is correct. Confusing in debuggers but thread-safe to the same rules above.

== volatile vs std::atomic

`volatile` and `std::atomic` solve unrelated problems. Using `volatile` for concurrency is the single most common C++ bug pattern.

#table(
  columns: (auto, auto, auto),
  [*Property*], [*`volatile T`*], [*`std::atomic<T>`*],
  [Prevents compiler from caching value in a register], [Yes], [Yes],
  [Prevents compiler from reordering across the access], [No (only volatile↔volatile is ordered)], [Yes (per memory_order)],
  [Atomic read-modify-write (`++`, CAS)], [No — `v++` is load+add+store, racy], [Yes],
  [Inter-thread synchronization / publication], [No], [Yes],
  [Tearing on word-sized reads/writes], [Implementation-defined (typically untorn on aligned word; *not guaranteed*)], [Never (standard guarantee)],
  [Use case], [MMIO registers, `setjmp/longjmp`-modified locals, signal handler ↔ main flag], [Inter-thread communication],
)

```cpp
// WRONG — volatile gives no ordering, no atomicity for RMW
volatile bool ready = false;
volatile int data = 0;
// Thread A
data = 42; ready = true;            // CPU may reorder; thread B may see ready=true with data=0
// Thread B
while (!ready) {}
assert(data == 42);                  // May fire on ARM/POWER; usually "works" on x86 by luck

// CORRECT
std::atomic<bool> ready{false};
int data = 0;
// Thread A
data = 42;
ready.store(true, std::memory_order_release);
// Thread B
while (!ready.load(std::memory_order_acquire)) {}
assert(data == 42);                  // Guaranteed
```

*Legitimate uses of `volatile`:*

- MMIO: `volatile uint32_t* reg = (uint32_t*)0xFEE00000;` — every read/write goes to the bus, no caching in registers.
- A flag set by an asynchronous signal handler and read by the same thread: `volatile sig_atomic_t flag;` (signal handlers can use `std::atomic` with `is_lock_free` in C++20, but `volatile sig_atomic_t` is the historical idiom).
- Defeating compiler optimizations in benchmarks: `volatile T sink = expr;`.

*Java's `volatile` is `std::atomic` with `seq_cst`. C and C++'s `volatile` is not.* Programmers crossing from Java carry this misconception in.

== Floating-Point Hazards Beyond Negative Zero

A companion to #emph[Negative Zero]: the rest of the IEEE 754 footguns.

*NaN propagation and comparison:*

```cpp
double n = std::nan("");
n == n;                      // false  (only value where x == x is false)
n != n;                      // true   (idiomatic NaN check pre-C++11)
std::isnan(n);               // true   (preferred)
n < 1.0;  n > 1.0;  n == 1.0;// all false  (unordered comparison)
std::sort(v.begin(), v.end()); // UB if v contains NaN — strict weak order violated
```

`std::sort` and `std::map` require a strict weak order. NaN's `<` returns false in both directions, breaking the contract — sort can read out of bounds or loop. Always strip NaNs before sorting FP arrays.

*Subnormals (denormals):* values smaller than `~2.2e-308` (double) lose precision and *on x86 trap into microcode*. A loop that drifts into subnormals can slow down 10-100× silently. Diagnose with `perf stat -e fp_assist.any` (Intel). Fixes: `_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)` + `_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON)`, or compile with `-ffast-math` (see caveat below). Audio/DSP code routinely needs this.

*`-ffast-math` (`/fp:fast`):* aggressive umbrella flag that enables roughly:
- `-fno-honor-nans` (assume operands are not NaN)
- `-fno-honor-infinities` (assume no inf)
- `-fno-signed-zeros` (treat `+0.0 == -0.0` even for `1/x` sign)
- `-fassociative-math` (allow reassociation: `(a+b)+c → a+(b+c)`, breaks Kahan summation)
- `-freciprocal-math` (`a/b → a * (1/b)`)
- `-ffinite-math-only` (makes `isnan(x)` constant-fold to `false`)

The last one is the classic production bug: a library compiled with `-ffast-math` has all its `isnan` checks compile away to `false`. Worse, on GCC `-ffast-math` sets `MXCSR.FTZ`/`DAZ` *process-wide* via a constructor in `crtfastmath.o` — linking a single TU with `-ffast-math` changes FP behavior for every other TU in the binary. Prefer `-fno-math-errno -fno-trapping-math` (the safe subset) or use `[[gnu::optimize("fast-math")]]` per-function.

*Compiler reordering of FP ops is normally forbidden.* `a + b + c` is `(a+b)+c`, not `a+(b+c)`, because FP addition is not associative. Reassociation is only allowed under `-ffast-math` / `#pragma STDC FP_CONTRACT`. This is why naive parallel reduction gives different results from serial reduction on the same data.

*Comparison and equality:* never compare FP for equality after arithmetic. Use a relative tolerance: `std::abs(a - b) <= eps * std::max(std::abs(a), std::abs(b))`. Absolute tolerance fails near zero; relative tolerance fails at zero — combine both for robustness. `std::numeric_limits<double>::epsilon()` is the gap at 1.0, *not* a universal "small number."

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

*Compiler intrinsics:* Compile to CPU instructions with `-march=native` or `-mpopcnt`/`-mlzcnt`. Typically 1-3 cycle latency [Intel Opt. Manual 2023, Appx. C: POPCNT = 3 cycles, LZCNT = 3 cycles on modern CPUs] vs $#sym.tilde.op$10-20 cycles for loop implementation. Without target flags, may compile to loops.

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

// Variable division: slower on some CPUs
int result = n / divisor;  // Modern x86 (Skylake+): ~3-6 cycles [Intel Opt. Manual 2023, Table C-16]
                           // Older CPUs (pre-Skylake): ~10-40 cycles

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
- Default page size: 4KB (x86-64) [Intel SDM Vol. 3A]
- Page table walk: 4-level on x86-64 = 4 memory accesses ($#sym.tilde.op$800 cycles without TLB) [Drepper 2007]
- TLB#footnote[The TLB (Translation Lookaside Buffer) is a small hardware cache that stores recent virtual-to-physical address translations. Without the TLB, every memory access would require walking the page table (4 memory accesses on x86-64), costing approximately 800 cycles. The TLB reduces this to zero cost for cached translations, making it critical for performance.] (Translation Lookaside Buffer): L1 TLB = 64 entries (data) + 128 entries (instruction), L2 TLB = 1024-2048 entries [Intel Optimization Manual 2023, §2.1.5]
- TLB miss penalty: $#sym.tilde.op$20-100 cycles (page table walk, varies by CPU) [Agner Fog 2023, Table 14.3]
- Huge pages (2MB/1GB): 512x/262144x fewer TLB entries [Linux Kernel Doc: hugetlbpage.txt]

*Memory allocation costs:*
```cpp
// Small allocations (<= 256 bytes)
int* p = new int[64];  // tcmalloc thread cache: ~10-20 cycles [Google tcmalloc design doc]
delete[] p;            // ~10-20 cycles

// Large allocations (> 32KB)
int* p = new int[10000];  // mmap syscall: ~1000+ cycles [measured on Linux 5.x]
delete[] p;               // munmap syscall: ~1000+ cycles

// Stack allocation (fastest)
int arr[1000];  // ~0 cycles (just adjust RSP register) [Intel ISA: sub rsp, imm]
```

*Modern allocator internals:*
- tcmalloc/jemalloc: thread-local caches eliminate lock contention [Evans 2006, jemalloc(3)]
- Size classes: reduce fragmentation (16, 32, 48, 64, 80... bytes) [Berger et al. 2000, "Hoard"]
- Allocation overhead: 8-16 bytes metadata per block (size + flags) [glibc malloc.c]
- Bulk deallocation: free list batching = amortized O(1) [tcmalloc design, §3.2]

*Page faults:*
```cpp
// Minor fault (page in RAM, not mapped): ~5000 cycles [Levon & Elie 2004, OProfile study]
char* p = (char*)mmap(NULL, 1GB, ...);
p[0] = 1;  // First access triggers fault

// Major fault (load from disk): ~1-10 million cycles [depends on I/O: HDD ~10ms, SSD ~100μs]
// Demand paging: OS loads on access, not allocation [Tanenbaum 2014, Modern OS §3.3]
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
- syscall cost: $#sym.tilde.op$50-150 cycles (user→kernel context switch) [Soares & Stumm 2010, FlexSC; varies by CPU generation]
- vDSO#footnote[vDSO (virtual dynamic shared object) is a Linux kernel mechanism that maps certain system call implementations directly into userspace memory, allowing programs to invoke them without the expensive kernel mode transition. This reduces overhead from approximately 50-150 cycles (syscall) to 5-10 cycles (function call).] (virtual dynamic shared object): $#sym.tilde.op$5-10 cycles (no kernel transition) [Linux vDSO(7) man page]
- vDSO functions: `gettimeofday()`, `clock_gettime()`, `getcpu()` [mapped to userspace]

```cpp
#include <time.h>
timespec ts;
clock_gettime(CLOCK_MONOTONIC, &ts);  // vDSO: ~10 cycles, not syscall
```

*Cache hierarchy timing (Intel Skylake/AMD Zen 2+ representative):*
- L1 access: $#sym.tilde.op$4-5 cycles (32-48KB data + 32KB instruction) [Intel Opt. Manual 2023, Appx. C; Agner Fog 2023, Table 3.3]
- L2 access: $#sym.tilde.op$12-15 cycles (256KB-512KB private per core) [AMD Zen optimization guide; measured]
- L3 access: $#sym.tilde.op$40-75 cycles (8-64MB shared across cores) [Intel Xeon: ~40 cycles; AMD EPYC: ~50-75 cycles]
- RAM access: $#sym.tilde.op$200 cycles (DDR4/DDR5, ~60-100ns latency) [Drepper 2007; Intel Memory Latency Checker]
- Cache line: 64 bytes (all modern x86) [Intel SDM Vol. 1, §11.5.3]

*NUMA (Non-Uniform Memory Access):*
```cpp
// Local node memory: ~200 cycles [same as regular RAM access]
// Remote node memory: ~300-400 cycles (cross-socket) [Intel MLC tool measurements]
// Use numactl to bind process to node: [numactl(8) man page]
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
// Auto-vectorizes to: AVX2#footnote[AVX2 (Advanced Vector Extensions 2) is an x86 SIMD instruction set extension introduced in Intel's Haswell microarchitecture (2013). It provides 256-bit wide vector registers capable of processing 8 32-bit integers or 4 64-bit integers simultaneously, enabling significant performance improvements for data-parallel operations.] = 8 int32 per cycle, AVX-512 = 16 int32 per cycle

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

*Warning about `-march=native`:*
```bash
# DANGER: Optimizes for build machine, not deployment target
g++ -O3 -march=native program.cpp  # Binary may crash on older CPUs

# SAFER: Use architecture level tiers for compatibility
g++ -O3 -march=x86-64-v2 program.cpp  # SSE4.2, POPCNT (2009+)
g++ -O3 -march=x86-64-v3 program.cpp  # AVX2, FMA (2013+)
g++ -O3 -march=x86-64-v4 program.cpp  # AVX-512 (2017+)
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

== Debugging Optimized Code

*Challenge:* Aggressive optimization (-O2/-O3) makes debugging difficult due to:
- Inlined functions (stack traces don't show call sites)
- Reordered instructions (breakpoints hit out of source order)
- Optimized-away variables (cannot inspect values)

*Solutions:*
```cpp
// Per-function optimization control
__attribute__((optimize("-O0"))) void debugFunction() {
    // This function compiled without optimization
}

// Selective variable preservation
volatile int debug_value = x;  // Prevents optimization
```

*Incremental approach:*
1. Debug with -O0, profile with -O2
2. If bug only in -O2: binary search by disabling optimizations on half the functions
3. Use -O2 -g for debug symbols with optimization
4. Consider -Og (optimize for debugging) as middle ground

== Advanced Performance Topics

*Move semantics & RVO:*
```cpp
// RVO#footnote[RVO (Return Value Optimization) is a compiler optimization that eliminates unnecessary copy/move operations when returning objects from functions. Instead of constructing the object in the function and then copying it to the caller, the compiler constructs it directly in the caller's memory location. NRVO (Named RVO) extends this to named return values, though it's less reliably applied.] (Return Value Optimization) - automatic, zero cost
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

== References

*Primary Sources:*

*Intel Corporation (2023)*. Intel 64 and IA-32 Architectures Optimization Reference Manual. Order Number 248966-046. Available: https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html

*Intel Corporation (2023)*. Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 3A: System Programming Guide. Order Number 253668.

*AMD (2023)*. Software Optimization Guide for AMD Family 19h Processors (Zen 3). Publication #56665 Rev. 3.06.

*Agner Fog (2023)*. Instruction Tables: Lists of Instruction Latencies, Throughputs and Micro-operation Breakdowns for Intel, AMD and VIA CPUs. Technical University of Denmark. Available: https://www.agner.org/optimize/

*Agner Fog (2023)*. Optimizing Software in C++: An Optimization Guide for Windows, Linux and Mac Platforms. Available: https://www.agner.org/optimize/optimizing_cpp.pdf

*Memory & Systems:*

*Ulrich Drepper (2007)*. What Every Programmer Should Know About Memory. Red Hat, Inc. Available: https://people.freebsd.org/~lstewart/articles/cpumemory.pdf

*Soares, L. & Stumm, M. (2010)*. FlexSC: Flexible System Call Scheduling with Exception-Less System Calls. OSDI '10. pp. 33-46.

*Levon, J. & Elie, P. (2004)*. OProfile - A System Profiler for Linux. Available: http://oprofile.sourceforge.net/

*Tanenbaum, A.S. & Bos, H. (2014)*. Modern Operating Systems (4th ed.). Pearson. ISBN 978-0133591620.

*Memory Allocators:*

*Jason Evans (2006)*. A Scalable Concurrent malloc(3) Implementation for FreeBSD. BSDCan. Available: http://people.freebsd.org/~jasone/jemalloc/bsdcan2006/jemalloc.pdf

*Berger, E.D., McKinley, K.S., Blumofe, R.D., & Wilson, P.R. (2000)*. Hoard: A Scalable Memory Allocator for Multithreaded Applications. ASPLOS-IX. pp. 117-128.

*Google (2023)*. TCMalloc: Thread-Caching Malloc. Design Document. Available: https://github.com/google/tcmalloc/blob/master/docs/design.md

*Compiler & Language:*

*GCC Documentation (2023)*. Using the GNU Compiler Collection. Available: https://gcc.gnu.org/onlinedocs/gcc/

*LLVM Project (2023)*. LLVM Language Reference Manual. Available: https://llvm.org/docs/LangRef.html

*ISO/IEC (2020)*. ISO/IEC 14882:2020: Programming Languages — C++. International Organization for Standardization.

*Linux Kernel:*

*Linux Kernel Documentation*. Huge TLB Page Support in the Linux Kernel. Available: https://www.kernel.org/doc/Documentation/vm/hugetlbpage.txt

*Linux Manual Pages*. vdso(7) - Virtual Dynamic Shared Object. Available: https://man7.org/linux/man-pages/man7/vdso.7.html

*Tools & Benchmarking:*

*Intel (2023)*. Intel Memory Latency Checker (MLC). Available: https://www.intel.com/content/www/us/en/developer/articles/tool/intelr-memory-latency-checker.html

*Google (2023)*. Google Benchmark. Available: https://github.com/google/benchmark

*Measurement Methodology:*

All cycle counts and timing measurements refer to modern x86-64 processors (2020+):
- Intel: Skylake, Ice Lake, Sapphire Rapids microarchitectures
- AMD: Zen 2, Zen 3, Zen 4 microarchitectures

Ranges account for:
- CPU generation differences
- Clock speed variations
- Memory configuration (DDR4 vs DDR5, frequency, timings)
- System load and thermal throttling
- Measurement methodology variance

Measurements obtained via:
- Hardware performance counters (perf, Intel VTune, AMD uProf)
- Microbenchmarking with controlled environments
- Published vendor specifications

*Note:* Performance characteristics evolve with each CPU generation. Always profile on target hardware for production workloads.

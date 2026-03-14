= Part V: C++ Memory Model & Performance

== Memory Management: RAII and Allocators

=== Allocation Strategies

*Stack allocation:* Fastest, automatic lifetime
```cpp
void process() {
    int buffer[1024];          // Stack-allocated (fast)
    std::array<int, 256> arr;  // Stack-allocated, bounds-safe
    // Automatically freed when scope exits
}
```
- No allocator overhead (~0 ns)
- Limited by stack size (typically 1--8 MB)
- Use: Small, fixed-size, short-lived data

*Heap allocation (default):* `new`/`delete`, flexible
```cpp
auto ptr = std::make_unique<Widget>(42);  // Heap-allocated
auto vec = std::make_shared<std::vector<int>>(1000);
// Freed automatically by smart pointer destructor (RAII)
```
- Allocator overhead (~50--100 ns per allocation)
- Unlimited size (bounded by virtual memory)
- Use: Dynamic-size, long-lived, polymorphic objects

*Arena allocator:* Bulk allocation, fast deallocation
```cpp
class ArenaAllocator {
    char* buffer_;
    size_t offset_ = 0;
    size_t capacity_;
public:
    explicit ArenaAllocator(size_t cap)
        : buffer_(new char[cap]), capacity_(cap) {}
    ~ArenaAllocator() { delete[] buffer_; }

    void* allocate(size_t size, size_t align = alignof(std::max_align_t)) {
        size_t aligned = (offset_ + align - 1) & ~(align - 1);
        if (aligned + size > capacity_) throw std::bad_alloc();
        void* ptr = buffer_ + aligned;
        offset_ = aligned + size;
        return ptr;
    }
    void reset() { offset_ = 0; }  // Free all at once (O(1))
};
```
- Allocation: ~5 ns (bump pointer)
- Deallocation: O(1) reset (free everything at once)
- Use: Request processing, frame allocators, parsers

*Pool allocator:* Fixed-size blocks, no fragmentation
```cpp
template <typename T, size_t BlockSize = 4096>
class PoolAllocator {
    union Block { T data; Block* next; };
    Block* free_list_ = nullptr;
    std::vector<std::unique_ptr<Block[]>> chunks_;
public:
    T* allocate() {
        if (!free_list_) grow();
        Block* block = free_list_;
        free_list_ = block->next;
        return reinterpret_cast<T*>(block);
    }
    void deallocate(T* ptr) {
        auto* block = reinterpret_cast<Block*>(ptr);
        block->next = free_list_;
        free_list_ = block;
    }
private:
    void grow() {
        auto chunk = std::make_unique<Block[]>(BlockSize);
        for (size_t i = 0; i < BlockSize - 1; ++i)
            chunk[i].next = &chunk[i + 1];
        chunk[BlockSize - 1].next = free_list_;
        free_list_ = &chunk[0];
        chunks_.push_back(std::move(chunk));
    }
};
```
- Allocation: ~10 ns (pop from free list)
- Deallocation: ~5 ns (push to free list)
- Use: Game entities, network packets, AST nodes

*Slab allocator:* Cached object pools (kernel-style)
```cpp
// Pre-constructed objects, reuse without re-initialization
template <typename T>
class SlabAllocator {
    std::vector<T> slab_;
    std::stack<size_t> free_indices_;
public:
    explicit SlabAllocator(size_t count) : slab_(count) {
        for (size_t i = 0; i < count; ++i)
            free_indices_.push(i);
    }
    T* allocate() {
        size_t idx = free_indices_.top();
        free_indices_.pop();
        return &slab_[idx];
    }
    void deallocate(T* ptr) {
        size_t idx = ptr - slab_.data();
        free_indices_.push(idx);
    }
};
```
- Use: Frequently created/destroyed objects of same type

*Comparison:*
```
Strategy     Alloc Time    Dealloc Time    Fragmentation    Use Case
Stack        ~0 ns         automatic       None             Local variables
Heap (new)   ~50-100 ns    ~50-100 ns      Yes              General purpose
Arena        ~5 ns         O(1) reset      None             Batch processing
Pool         ~10 ns        ~5 ns           None             Fixed-size objects
Slab         ~10 ns        ~5 ns           None             Hot object reuse
```

=== RAII and Smart Pointers

*RAII (Resource Acquisition Is Initialization):* Tie resource lifetime to scope.

*Core principle:* Constructor acquires, destructor releases.
```cpp
class FileHandle {
    FILE* file_;
public:
    explicit FileHandle(const char* path) : file_(fopen(path, "r")) {
        if (!file_) throw std::runtime_error("cannot open file");
    }
    ~FileHandle() { if (file_) fclose(file_); }

    // Non-copyable, movable
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
    FileHandle(FileHandle&& other) noexcept : file_(std::exchange(other.file_, nullptr)) {}
    FileHandle& operator=(FileHandle&& other) noexcept {
        if (file_) fclose(file_);
        file_ = std::exchange(other.file_, nullptr);
        return *this;
    }
};
```

*Smart pointer guide:*
```
Pointer            Overhead     Ownership         Use Case
unique_ptr<T>      0 (zero!)    Exclusive          Default choice
shared_ptr<T>      2 pointers   Shared (refcount)  Shared ownership
weak_ptr<T>        2 pointers   Non-owning         Break cycles
T* (raw)           1 pointer    Non-owning         Observers only
```

*`unique_ptr`:* Zero overhead, exclusive ownership
```cpp
auto widget = std::make_unique<Widget>(42);     // Heap alloc
auto arr = std::make_unique<int[]>(1000);        // Array
auto moved = std::move(widget);                  // Transfer ownership
// widget is now nullptr
```

*`shared_ptr`:* Reference-counted, thread-safe refcount
```cpp
auto data = std::make_shared<Data>();  // Single allocation (object + refcount)
auto copy = data;                      // refcount = 2 (atomic increment)
// Last shared_ptr destroyed → object freed
```

*Pitfall -- reference cycles:*
```cpp
struct Node {
    std::shared_ptr<Node> next;  // Cycle: A→B→A (memory leak!)
};
// Fix: use weak_ptr for back-references
struct Node {
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;    // Doesn't prevent deletion
};
```

=== Copy Elision: RVO, NRVO, and Move Semantics

*Return Value Optimization (RVO):* Construct return value in caller's space.

```cpp
std::vector<int> make_vector() {
    return std::vector<int>{1, 2, 3, 4, 5};  // Guaranteed RVO (C++17)
    // No copy, no move -- constructed directly in caller's memory
}

std::string build_string() {
    std::string result;       // NRVO: Named RVO (not guaranteed, but common)
    result += "hello";
    result += " world";
    return result;            // Compiler elides copy (usually)
}
```

*Move semantics:* Transfer resources instead of copying.
```cpp
class Buffer {
    int* data_;
    size_t size_;
public:
    // Move constructor: steal resources (O(1))
    Buffer(Buffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr))
        , size_(std::exchange(other.size_, 0)) {}

    // Copy constructor: deep copy (O(n))
    Buffer(const Buffer& other)
        : data_(new int[other.size_]), size_(other.size_) {
        std::copy(other.data_, other.data_ + size_, data_);
    }
};

Buffer a(1000);
Buffer b = std::move(a);  // O(1) move, not O(n) copy
```

*Performance impact:*
```
Operation              Cost
Copy std::vector       O(n) -- allocate + copy all elements
Move std::vector       O(1) -- swap 3 pointers
RVO (guaranteed)       O(0) -- no copy or move at all
```

=== String Optimization: SSO and string_view

*Small String Optimization (SSO):* Short strings stored inline (no heap).
```cpp
std::string s1 = "hi";           // SSO: stored in string object (no heap alloc)
std::string s2 = "this is a longer string that exceeds SSO";  // Heap allocated
// Typical SSO threshold: 15-22 bytes (implementation-dependent)
```

*`std::string_view`:* Non-owning reference to string data (zero-copy).
```cpp
void process(std::string_view sv) {  // No copy, no allocation
    auto sub = sv.substr(0, 5);      // O(1), no allocation
    // sv is valid as long as underlying data exists
}

std::string data = "hello world";
process(data);            // No copy
process("literal");       // No copy (points to static storage)
```

*`constexpr` strings (C++20):*
```cpp
constexpr std::string_view greeting = "hello";
static_assert(greeting.size() == 5);  // Compile-time string processing
```

=== Placement New and Custom Construction

*Placement new:* Construct object at specific memory address.
```cpp
alignas(Widget) char buffer[sizeof(Widget)];
Widget* w = new (buffer) Widget(42);  // Construct in pre-allocated memory
// Must manually destroy:
w->~Widget();  // Explicit destructor call
```

*Use with allocators:*
```cpp
void* mem = arena.allocate(sizeof(Widget), alignof(Widget));
Widget* w = new (mem) Widget(args...);  // Construct in arena memory
// Arena reset destroys all objects at once
```

*`std::pmr` (Polymorphic Memory Resources, C++17):*
```cpp
#include <memory_resource>

char buffer[4096];
std::pmr::monotonic_buffer_resource pool(buffer, sizeof(buffer));
std::pmr::vector<int> vec(&pool);  // Uses stack-backed allocator
vec.push_back(42);                 // No heap allocation!
```

=== Interview Questions: Memory Management

*Q1: Explain RAII and why it matters for C++ performance.*

A: RAII = Resource Acquisition Is Initialization. Constructor acquires, destructor releases.

*Key benefits:*
1. *Deterministic cleanup*: Resources freed at scope exit (no GC pauses)
2. *Exception safety*: Destructors always run (even on exception)
3. *Zero overhead*: No runtime tracking (unlike reference counting)
4. *Composability*: RAII objects nest and compose naturally

*Performance implications:*
- No garbage collector pauses (deterministic latency)
- Destructor cost is paid at scope exit (predictable timing)
- Smart pointers: `unique_ptr` has zero overhead vs raw pointer
- `shared_ptr` has atomic refcount overhead (~10-20 ns per copy)

*When RAII isn't enough:*
- Shared ownership → `shared_ptr` (adds refcount cost)
- Complex object graphs → consider arena allocator
- Real-time systems → pre-allocate, use pool allocators

*Q2: When would you use arena allocation over standard `new`/`delete`?*

A:

*Arena benefits:*
- Bulk deallocation: Free thousands of objects in O(1)
- Cache-friendly: Objects allocated contiguously
- No fragmentation: Linear allocation, no free-list overhead
- Thread-local: No lock contention (one arena per thread)

*Use cases:*
- *Request processing*: Allocate per-request, free all at request end
- *Compilers/parsers*: AST nodes freed after compilation
- *Game frames*: Per-frame allocations, reset each frame
- *Protobuf/serialization*: Temporary message objects

*Trade-offs:*
- Cannot free individual objects (only bulk reset)
- Memory not returned until arena reset
- Must ensure no references survive arena reset

*Performance:*
```
Allocator          1M allocs    1M deallocs    Total
std::allocator     ~80 ms       ~60 ms         ~140 ms
Arena              ~5 ms        ~0.001 ms      ~5 ms (28x faster)
Pool               ~10 ms       ~5 ms          ~15 ms
```

*Q3: What is the C++ equivalent of Java's garbage collection?*

A: C++ uses deterministic destruction instead of garbage collection.

*Mapping:*
```
Java (GC)                    C++ (RAII/Manual)
Automatic GC                 RAII (destructors)
GC pauses                    No pauses (deterministic)
Heap-only objects             Stack + heap choice
Finalize (deprecated)        Destructors (reliable)
Weak references              std::weak_ptr
Object pooling (GC tuning)   Arena/pool allocators
-Xmx (heap size)             Virtual memory (OS managed)
```

*Advantages of C++ approach:*
- Predictable latency (no GC pauses)
- Lower memory overhead (no GC metadata)
- Cache-friendly layouts (value types, contiguous memory)

*Disadvantages:*
- Manual lifetime management (mitigated by smart pointers)
- Dangling pointer risk (mitigated by RAII, sanitizers)
- No automatic cycle detection (use `weak_ptr`)

== Compiler Optimization and LTO

=== Optimization Levels

*Compiler flags (GCC/Clang):*
```bash
-O0    # No optimization (debug, fast compile)
-O1    # Basic optimizations (reduce size + time, no slow optimizations)
-O2    # Standard optimization (recommended for production)
-O3    # Aggressive (auto-vectorization, loop unrolling, function cloning)
-Os    # Optimize for size (like -O2 but avoids size-increasing opts)
-Ofast # -O3 + fast-math (breaks IEEE 754 -- use with caution)
```

*What each level enables:*
```
Level   Inlining  Vectorization  Unrolling  LTO   Typical Speedup
-O0     No        No             No         No    1x (baseline)
-O1     Basic     No             No         No    2-3x
-O2     Yes       Basic          No         No    3-5x
-O3     Aggressive Yes           Yes        No    4-8x
-O3+LTO Aggressive Yes           Yes        Yes   5-10x
```

*Architecture-specific flags:*
```bash
-march=native            # Optimize for current CPU (AVX, AVX2, etc.)
-mtune=native            # Tune scheduling for current CPU
-mavx2                   # Enable AVX2 instructions explicitly
-mfma                    # Enable fused multiply-add
-mbmi2                   # Enable BMI2 (bit manipulation)
```

=== Inlining and Devirtualization

*Inlining:* Replace function call with function body.

```cpp
// Before inlining
inline int add(int a, int b) { return a + b; }
int result = add(x, y);

// After inlining (compiler does this automatically at -O2+)
int result = x + y;  // No call overhead!
```

*Benefits:*
- Eliminate call overhead (~2--5 ns per call)
- Enable further optimizations (constant propagation, dead code elimination)

*Controlling inlining:*
```cpp
// Hints (not guarantees)
inline int fast_func(int x) { return x * 2; }           // Suggest inline
[[gnu::always_inline]] int critical(int x) { return x; } // Force inline (GCC)
__attribute__((noinline)) void debug_log(const char* msg); // Prevent inline
```

*Devirtualization:* Convert virtual call to direct call.

```cpp
class Animal {
public:
    virtual void speak() const = 0;
};
class Dog : public Animal {
public:
    void speak() const override { /* bark */ }
};

void make_noise(const Animal& a) {
    a.speak();  // Virtual call (vtable lookup ~5 ns)
}

// If compiler proves a is always Dog:
// Devirtualized to direct Dog::speak() call (can then inline!)
```

*Techniques for devirtualization:*
```cpp
// 1. final keyword (helps compiler devirtualize)
class Dog final : public Animal {  // No further subclasses
    void speak() const override { /* bark */ }
};

// 2. CRTP (Curiously Recurring Template Pattern) -- static polymorphism
template <typename Derived>
class AnimalCrtp {
public:
    void speak() const {
        static_cast<const Derived*>(this)->speak_impl();  // No vtable!
    }
};
class DogCrtp : public AnimalCrtp<DogCrtp> {
public:
    void speak_impl() const { /* bark */ }
};

// 3. std::variant (compile-time dispatch)
using AnyAnimal = std::variant<Dog, Cat, Bird>;
void make_noise(const AnyAnimal& a) {
    std::visit([](const auto& animal) { animal.speak(); }, a);
    // No virtual dispatch! Compiler generates switch/jump table.
}
```

=== `constexpr` and Compile-Time Computation

*`constexpr`:* Evaluate at compile time (zero runtime cost).

```cpp
constexpr int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) result *= i;
    return result;
}
constexpr int fact10 = factorial(10);  // Computed at compile time!

// constexpr containers (C++20)
constexpr auto make_lut() {
    std::array<int, 256> lut{};
    for (int i = 0; i < 256; ++i)
        lut[i] = i * i;
    return lut;
}
constexpr auto squares = make_lut();  // Lookup table at compile time
```

*`consteval` (C++20):* Must evaluate at compile time (error if not).
```cpp
consteval int compile_time_only(int x) { return x * x; }
int val = compile_time_only(42);  // OK: computed at compile time
// int rt = compile_time_only(runtime_var);  // ERROR: not compile-time
```

*`if constexpr` (C++17):* Compile-time branching (discard unused branches).
```cpp
template <typename T>
auto serialize(const T& val) {
    if constexpr (std::is_arithmetic_v<T>) {
        return std::to_string(val);     // Arithmetic types
    } else if constexpr (std::is_same_v<T, std::string>) {
        return val;                      // Already a string
    } else {
        return val.to_string();          // Custom types
    }
}
```

=== Link-Time Optimization (LTO) and PGO

*LTO:* Optimize across translation units (whole-program optimization).

```bash
# Enable LTO
g++ -O3 -flto main.cpp utils.cpp -o app

# Thin LTO (faster compile, similar optimization)
clang++ -O3 -flto=thin main.cpp utils.cpp -o app
```

*LTO benefits:*
- Cross-file inlining (inline functions from other `.cpp` files)
- Dead code elimination (remove unused functions across files)
- Interprocedural constant propagation
- Devirtualization across translation units
- Typical improvement: 5--20% over `-O3` alone

*Profile-Guided Optimization (PGO):*
```bash
# Step 1: Instrument
g++ -O3 -fprofile-generate -o app main.cpp

# Step 2: Run with representative workload
./app < typical_input.txt

# Step 3: Recompile with profile data
g++ -O3 -fprofile-use -o app main.cpp
```

*PGO benefits:*
- Branch prediction hints (layout hot paths)
- Function ordering (co-locate hot functions)
- Inline decisions based on actual call frequencies
- Loop unrolling based on actual trip counts
- Typical improvement: 10--30% over `-O3` alone

*Combined (best performance):*
```bash
g++ -O3 -flto -fprofile-use -march=native -o app *.cpp
# Maximum optimization: LTO + PGO + arch-specific
```

=== Loop Optimizations

*Loop unrolling:* Repeat loop body to reduce branch overhead.

```cpp
// Before
for (int i = 0; i < 100; i++) {
    sum += array[i];
}

// After unrolling (factor 4, compiler does this at -O3)
for (int i = 0; i < 100; i += 4) {
    sum += array[i];
    sum += array[i + 1];
    sum += array[i + 2];
    sum += array[i + 3];
}
// Benefits: Fewer branches, better instruction-level parallelism
```

*Loop vectorization (auto-vectorization):* Use SIMD instructions.

```cpp
// Scalar code (compiler auto-vectorizes at -O3 -march=native)
void add_arrays(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// Explicit SIMD (AVX2: 8 floats at once)
#include <immintrin.h>
void add_arrays_avx(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&c[i], vc);
    }
    for (; i < n; ++i) c[i] = a[i] + b[i];  // Remainder
}
// 8x throughput for float operations!
```

*Loop invariant code motion:* Compiler hoists constant computation.

```cpp
// Before
for (int i = 0; i < n; i++) {
    int x = a * b;       // Constant within loop
    array[i] = x + i;
}

// After (compiler does this at -O1+)
int x = a * b;           // Hoisted outside loop
for (int i = 0; i < n; i++) {
    array[i] = x + i;
}
```

*Help the compiler vectorize:*
```cpp
// Use restrict-like semantics (no aliasing)
void add(float* __restrict__ c, const float* __restrict__ a,
         const float* __restrict__ b, size_t n) {
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
    // __restrict__ tells compiler a, b, c don't overlap → safe to vectorize
}
```

=== Intrinsics and CPU-Specific Operations

*Compiler built-ins:* Map directly to CPU instructions.

```cpp
// Bit operations
__builtin_popcount(x)           // POPCNT instruction
__builtin_clz(x)                // Count leading zeros (BSR)
__builtin_ctz(x)                // Count trailing zeros (BSF/TZCNT)
__builtin_expect(x, 1)          // Branch prediction hint
// C++20: std::popcount, std::countl_zero, std::countr_zero

// Memory operations
__builtin_prefetch(&array[i + 16]);  // Prefetch into cache
std::memcpy(dst, src, n);            // Optimized to rep movsb or SIMD
std::memset(buf, 0, n);             // Optimized to rep stosb or SIMD
```

*SIMD intrinsics:*
```cpp
#include <immintrin.h>

// SSE4.2 string comparison (16 bytes at once)
int strcmp_fast(const char* a, const char* b) {
    __m128i va = _mm_loadu_si128((__m128i*)a);
    __m128i vb = _mm_loadu_si128((__m128i*)b);
    // PCMPESTRI: parallel string compare
    return _mm_cmpestri(va, 16, vb, 16, _SIDD_CMP_EQUAL_EACH);
}

// AES-NI: Hardware AES encryption
__m128i aes_encrypt(__m128i data, __m128i key) {
    return _mm_aesenc_si128(data, key);  // Single AES round (~1 cycle!)
    // 10x faster than software AES
}
```

*Checking CPU features at runtime:*
```cpp
#include <cpuid.h>
bool has_avx2() {
    unsigned eax, ebx, ecx, edx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return ebx & (1 << 5);  // AVX2 bit
}
```

*Compiler diagnostics (check what was optimized):*
```bash
# Show optimization decisions
g++ -O3 -fopt-info-vec-missed  # Show missed vectorizations
g++ -O3 -fopt-info-inline      # Show inlining decisions
g++ -O3 -S -o output.s         # Generate assembly (inspect)

# Clang
clang++ -O3 -Rpass=loop-vectorize       # Report vectorized loops
clang++ -O3 -Rpass-missed=loop-vectorize # Report missed opportunities
```

=== Interview Questions: Compiler Optimization

*Q1: What is LTO and how does it improve performance?*

A: LTO (Link-Time Optimization) performs whole-program optimization across translation units at link time.

*How it works:*
1. *Compile*: Generate intermediate representation (IR) instead of object code
2. *Link*: Optimize entire program IR as a single unit
3. *Codegen*: Generate optimized machine code

*Optimizations enabled:*
- Cross-file inlining (inline functions from other `.cpp` files)
- Global dead code elimination (remove unused functions)
- Interprocedural constant propagation
- Cross-file devirtualization

*Usage:*
```bash
g++ -O3 -flto *.cpp -o app    # Full LTO
clang++ -O3 -flto=thin *.cpp  # Thin LTO (faster compile)
```

*Trade-offs:*
- Compile time: 2--5x longer (but faster runtime)
- Memory usage: Higher at link time
- Debugging: Harder (heavy transformations)
- Typical speedup: 5--20%

*Q2: Explain C++ copy elision and how it differs from Java escape analysis.*

A: Copy elision = compiler constructs return value directly in caller's memory.

*C++ (compile-time, guaranteed):*
```cpp
std::vector<int> make() {
    return std::vector<int>{1, 2, 3};  // Guaranteed RVO (C++17)
    // Zero copies, zero moves
}
```

*Java escape analysis (runtime, JIT):*
- JVM analyzes if object escapes method scope
- May allocate on stack (scalar replacement)
- Decided at runtime, may deoptimize

*Key differences:*
```
Aspect                C++ Copy Elision      Java Escape Analysis
When                  Compile time          Runtime (JIT)
Guarantee             Mandatory (C++17)     Heuristic (may fail)
Mechanism             Direct construction   Scalar replacement
Cost                  Zero overhead         JIT analysis overhead
Applicability         Return values         Any local object
```

*Q3: How do compiler optimization levels differ?*

A:

*`-O0`:* No optimization
- Fastest compile, slowest runtime
- All variables in memory (easy debugging)
- Use: Development, debugging

*`-O2`:* Standard optimization
- Inlining, constant propagation, dead code elimination
- Loop invariant code motion, strength reduction
- Register allocation, instruction scheduling
- Use: Production builds (safe, effective)

*`-O3`:* Aggressive optimization
- Everything in `-O2` plus:
- Auto-vectorization (SIMD), loop unrolling
- Function cloning (specialize for call sites)
- Trade-off: Larger code (may hurt icache)
- Use: Compute-intensive workloads

*Impact:* 2--10x speedup from `-O0` to `-O3` depending on workload.

== C++ Memory Model and Concurrency

=== `std::atomic` and Memory Orders

*Java `volatile` vs C++ `std::atomic`:*
```
Java volatile         →  std::atomic with std::memory_order_seq_cst
Java non-volatile     →  Non-atomic access (undefined behavior if shared!)
```

*Memory orders (weakest to strongest):*
```cpp
std::memory_order_relaxed   // No ordering guarantees (just atomicity)
std::memory_order_acquire   // Reads after this see writes before release
std::memory_order_release   // Writes before this visible after acquire
std::memory_order_acq_rel   // Both acquire and release
std::memory_order_seq_cst   // Total ordering (default, safest, slowest)
```

*Example -- lock-free flag:*
```cpp
std::atomic<bool> ready{false};
int data = 0;

// Thread 1 (producer)
data = 42;                                    // Write data
ready.store(true, std::memory_order_release); // Publish

// Thread 2 (consumer)
while (!ready.load(std::memory_order_acquire)) {}  // Wait
assert(data == 42);  // Guaranteed to see data = 42
```

*Example -- relaxed counter (fastest):*
```cpp
std::atomic<uint64_t> counter{0};
counter.fetch_add(1, std::memory_order_relaxed);  // Just atomic increment
// Use when ordering doesn't matter (statistics, counters)
```

*Performance:*
```
Operation                    x86 Cost      ARM Cost
Relaxed load                 ~1 ns         ~1 ns
Acquire load                 ~1 ns         ~1-5 ns (barrier)
Seq-cst load                 ~1 ns         ~5-20 ns (full fence)
Relaxed store                ~1 ns         ~1 ns
Release store                ~1 ns         ~1-5 ns (barrier)
Seq-cst store                ~10-20 ns     ~10-30 ns (full fence)
fetch_add (relaxed)          ~5-10 ns      ~5-15 ns
CAS (compare_exchange)       ~10-30 ns     ~10-40 ns
```

=== False Sharing and Cache-Line Alignment

*False sharing:* Two threads modify different variables on the same cache line (64 bytes) causing invalidation ping-pong.

```cpp
// BAD: False sharing (counters on same cache line)
struct BadCounters {
    std::atomic<int64_t> counter_a;  // Same cache line!
    std::atomic<int64_t> counter_b;  // Invalidates counter_a on every write
};

// GOOD: Aligned to separate cache lines
struct alignas(64) GoodCounters {
    alignas(64) std::atomic<int64_t> counter_a;
    alignas(64) std::atomic<int64_t> counter_b;
};
// C++17: std::hardware_destructive_interference_size (usually 64)
```

*Performance impact:*
```
Scenario                      Throughput
No contention (1 thread)      ~300M ops/sec
False sharing (2 threads)     ~10M ops/sec (30x slower!)
Aligned (2 threads)           ~250M ops/sec per thread
```

=== Thread Pool Pattern

```cpp
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

class ThreadPool {
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;

public:
    explicit ThreadPool(size_t num_threads) {
        for (size_t i = 0; i < num_threads; ++i)
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock lock(mutex_);
                        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty()) return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
    }

    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
        using ReturnType = std::invoke_result_t<F, Args...>;
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        auto result = task->get_future();
        {
            std::lock_guard lock(mutex_);
            tasks_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return result;
    }

    ~ThreadPool() {
        { std::lock_guard lock(mutex_); stop_ = true; }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }
};
```

*Usage:*
```cpp
ThreadPool pool(std::thread::hardware_concurrency());
auto future = pool.submit([](int x) { return x * x; }, 42);
int result = future.get();  // 1764
```

=== Cache-Aware Programming

*Cache hierarchy:*
```
Level    Size       Latency     Bandwidth
L1       32-64 KB   ~1 ns       ~500 GB/s
L2       256-512 KB ~3-5 ns     ~200 GB/s
L3       8-64 MB    ~10-20 ns   ~100 GB/s
DRAM     GB-TB      ~50-100 ns  ~50 GB/s
```

*Array of Structures (AoS) vs Structure of Arrays (SoA):*
```cpp
// AoS: Bad cache utilization if only accessing positions
struct Particle { float x, y, z; float vx, vy, vz; float mass; };
std::vector<Particle> particles(N);  // 28 bytes per particle
for (auto& p : particles) p.x += p.vx;  // Loads 28 bytes, uses 8

// SoA: Good cache utilization (contiguous access)
struct Particles {
    std::vector<float> x, y, z;
    std::vector<float> vx, vy, vz;
    std::vector<float> mass;
};
Particles p(N);
for (size_t i = 0; i < N; ++i) p.x[i] += p.vx[i];  // Loads only needed data
// 2-4x faster for position-only updates (cache-friendly, SIMD-friendly)
```

*Prefetching:*
```cpp
for (size_t i = 0; i < n; ++i) {
    __builtin_prefetch(&data[i + 16], 0, 3);  // Prefetch 16 elements ahead
    process(data[i]);
}
```

*Cache-oblivious algorithms:*
- Recursive blocking (work well for any cache size)
- Example: Cache-oblivious matrix multiply, merge sort

== Profiling and Performance Analysis

=== Profiling Tools

*Linux `perf`:*
```bash
# CPU profiling (sampling)
perf record -g ./app          # Record call stacks
perf report                   # Interactive analysis

# Hardware counters
perf stat ./app               # Cache misses, branch mispredictions, IPC
perf stat -e cache-misses,branches,branch-misses ./app

# Flame graphs
perf record -g ./app
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
```

*Valgrind suite:*
```bash
# Memory errors (use-after-free, buffer overflow)
valgrind --tool=memcheck ./app

# Cache simulation (cache miss analysis)
valgrind --tool=cachegrind ./app
cg_annotate cachegrind.out.<pid>

# Call graph profiling
valgrind --tool=callgrind ./app
kcachegrind callgrind.out.<pid>  # GUI visualization

# Heap profiling
valgrind --tool=massif ./app
ms_print massif.out.<pid>
```

*Sanitizers (compile-time instrumentation):*
```bash
# AddressSanitizer: buffer overflow, use-after-free, memory leaks
g++ -O1 -fsanitize=address -fno-omit-frame-pointer -o app main.cpp

# ThreadSanitizer: data races, deadlocks
g++ -O1 -fsanitize=thread -o app main.cpp

# UndefinedBehaviorSanitizer: UB detection
g++ -O1 -fsanitize=undefined -o app main.cpp
```

*Heaptrack (heap profiling):*
```bash
heaptrack ./app
heaptrack_gui heaptrack.app.<pid>.gz
# Shows: allocation hotspots, memory leaks, peak usage
```

*Key metrics to monitor:*
- *IPC (Instructions Per Cycle)*: $> 2.0$ good, $< 1.0$ memory-bound
- *Cache miss rate*: L1 $< 5%$, L3 $< 1%$ ideal
- *Branch misprediction*: $< 2%$ ideal
- *Memory bandwidth*: Check if saturated

=== Benchmarking with Google Benchmark

*Purpose:* Accurate microbenchmarking (handles warmup, statistics, optimizer tricks).

```cpp
#include <benchmark/benchmark.h>

static void bm_vector_push(benchmark::State& state) {
    for (auto _ : state) {
        std::vector<int> vec;
        vec.reserve(state.range(0));
        for (int i = 0; i < state.range(0); ++i)
            vec.push_back(i);
        benchmark::DoNotOptimize(vec.data());  // Prevent dead code elimination
    }
    state.SetComplexityN(state.range(0));
}
BENCHMARK(bm_vector_push)->Range(8, 1 << 20)->Complexity(benchmark::oN);

static void bm_array_sum(benchmark::State& state) {
    std::vector<int> data(state.range(0));
    std::iota(data.begin(), data.end(), 0);
    for (auto _ : state) {
        int sum = 0;
        for (int x : data) sum += x;
        benchmark::DoNotOptimize(sum);  // Prevent dead code elimination
    }
    state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(int));
}
BENCHMARK(bm_array_sum)->Range(64, 1 << 20);

BENCHMARK_MAIN();
```

*Build and run:*
```bash
g++ -O3 -march=native bench.cpp -lbenchmark -lpthread -o bench
./bench --benchmark_format=console
```

*Manual benchmarking with `<chrono>`:*
```cpp
#include <chrono>

template <typename Func>
double measure_ns(Func&& f, int iterations = 1000) {
    // Warmup
    for (int i = 0; i < 100; ++i) f();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) f();
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::nano>(end - start).count() / iterations;
}
```

*Google Benchmark handles:*
- Warmup iterations (instruction cache, branch predictor)
- `DoNotOptimize()` prevents dead code elimination
- `ClobberMemory()` prevents reordering
- Statistical analysis (mean, median, stddev)
- Complexity analysis ($O(n)$, $O(n log n)$, etc.)

=== Compiler Output Inspection

*Generate assembly:*
```bash
g++ -O3 -S -fverbose-asm -o output.s main.cpp
# Annotated assembly with source references
```

*Compiler Explorer (godbolt.org):*
- Paste code, see assembly in real-time
- Compare compilers (GCC, Clang, MSVC)
- Compare optimization levels

*Check auto-vectorization:*
```bash
g++ -O3 -fopt-info-vec-optimized  # Show vectorized loops
g++ -O3 -fopt-info-vec-missed     # Show missed vectorization opportunities

# Clang
clang++ -O3 -Rpass=loop-vectorize
clang++ -O3 -Rpass-analysis=loop-vectorize  # Why vectorization failed
```

*Key assembly patterns to look for:*
```
vaddps    → AVX float add (vectorized!)
vmovups   → AVX unaligned load/store
vfmadd    → Fused multiply-add (FMA)
call      → Function not inlined
ret       → Function boundary
```

=== Interview Questions: Profiling

*Q1: How do you diagnose performance issues in a C++ application?*

A:

*Step-by-step:*

*1. Identify bottleneck type:*
- High CPU → `perf record` + flame graph
- Memory issues → AddressSanitizer, heaptrack, valgrind
- Cache misses → `perf stat`, cachegrind
- Threading → ThreadSanitizer, `perf` lock analysis

*2. Profile:*
```bash
# CPU profiling (flame graph)
perf record -g ./app && perf script | flamegraph.pl > flame.svg

# Hardware counters (cache, branches)
perf stat -e cache-misses,branch-misses,instructions,cycles ./app

# Memory profiling
heaptrack ./app
```

*3. Analyze:*
- Flame graph → hot functions (optimize these first)
- IPC < 1.0 → memory-bound (improve cache usage)
- High cache miss rate → restructure data layout (AoS → SoA)
- Branch misprediction → use branchless code, `[[likely]]`/`[[unlikely]]`

*4. Fix and verify:*
- Benchmark before/after (Google Benchmark)
- Check assembly output (Compiler Explorer)
- Profile again to confirm improvement

*Q2: How do you avoid microbenchmark pitfalls in C++?*

A:

*Common pitfalls:*

*1. Dead code elimination:*
```cpp
// Wrong: Compiler removes entire computation!
int sum = 0;
for (int i = 0; i < 1000; ++i) sum += i;
// sum never used → compiler eliminates loop

// Fix: Use benchmark::DoNotOptimize or volatile sink
benchmark::DoNotOptimize(sum);
```

*2. Constant folding:*
```cpp
// Wrong: Compiler computes at compile time
int result = factorial(10);  // Becomes: int result = 3628800;

// Fix: Use runtime input
int n = get_input();  // Opaque to compiler
int result = factorial(n);
```

*3. No warmup:*
```cpp
// Wrong: First run fills instruction cache, branch predictor
// Fix: Run warmup iterations before measuring
```

*4. Measurement overhead:*
```cpp
// Wrong: chrono call inside tight loop
// Fix: Measure many iterations, divide by count
```

*Solution: Use Google Benchmark*
```cpp
static void bm_compute(benchmark::State& state) {
    for (auto _ : state) {
        int result = compute(state.range(0));
        benchmark::DoNotOptimize(result);  // Prevent DCE
    }
}
BENCHMARK(bm_compute)->Range(8, 1 << 16);
```

*Q3: What is profile-guided optimization and when should you use it?*

A: PGO = compile with runtime profile data to guide optimizations.

*Process:*
1. Instrument build (`-fprofile-generate`)
2. Run representative workload (collect profile)
3. Rebuild with profile data (`-fprofile-use`)

*What PGO optimizes:*
- *Branch layout*: Hot paths fall through (fewer jumps)
- *Function ordering*: Hot functions co-located (fewer icache misses)
- *Inlining*: Inline actually-hot functions (not just small ones)
- *Loop unrolling*: Unroll based on actual trip counts
- *Register allocation*: Prioritize hot variables

*When to use:*
- Large applications with many code paths (browsers, databases, compilers)
- When `-O3` isn't enough (10--30% additional speedup)
- Stable workload patterns (profile must be representative)

*Real-world examples:*
- Chromium: ~10% speedup with PGO
- GCC itself: ~5--8% speedup when PGO'd
- Databases: Significant query throughput improvement

== Static and Dynamic Linking

=== Static Linking

```bash
# Create static library
g++ -c utils.cpp -o utils.o
ar rcs libutils.a utils.o

# Link statically
g++ main.cpp -L. -lutils -static -o app
# Entire library code embedded in executable
```

*Pros:* Single binary, no dependency issues, slightly faster (no PLT indirection).
*Cons:* Larger binary, no shared memory across processes, must recompile to update library.

=== Dynamic Linking and `dlopen`

```cpp
#include <dlfcn.h>

// Load shared library at runtime
void* handle = dlopen("./libplugin.so", RTLD_LAZY);
if (!handle) { std::cerr << dlerror() << '\n'; return; }

// Get function pointer
using PluginFunc = int(*)(const char*);
auto func = reinterpret_cast<PluginFunc>(dlsym(handle, "plugin_entry"));
if (!func) { std::cerr << dlerror() << '\n'; return; }

int result = func("hello");  // Call loaded function

dlclose(handle);  // Unload library
```

*Build shared library:*
```bash
g++ -shared -fPIC -o libplugin.so plugin.cpp
```

*Pros:* Smaller executables, shared memory, hot-reload plugins.
*Cons:* Runtime overhead (PLT/GOT), dependency management, versioning.

=== RTTI and Alternatives

*RTTI (`typeid`, `dynamic_cast`):* Runtime type information.
```cpp
Base* ptr = get_object();
if (auto* derived = dynamic_cast<Derived*>(ptr)) {
    derived->special_method();  // Safe downcast
}
std::cout << typeid(*ptr).name();  // Type name (mangled)
```

*Cost:*
- `dynamic_cast`: ~50--100 ns (traverses class hierarchy)
- `typeid`: ~5--10 ns (single vtable lookup)
- Can disable with `-fno-rtti` (saves binary size, ~5--10%)

*Alternatives to RTTI:*
```cpp
// 1. Enum-based type tag (fastest)
enum class NodeType { Literal, BinaryOp, UnaryOp };
struct Node { NodeType type; /* ... */ };

// 2. Visitor pattern (compile-time dispatch)
struct Visitor {
    virtual void visit(Literal&) = 0;
    virtual void visit(BinaryOp&) = 0;
};

// 3. std::variant (no virtual dispatch)
using Expr = std::variant<Literal, BinaryOp, UnaryOp>;
auto result = std::visit(evaluator, expr);  // Compiler-generated switch

// 4. C++26 reflection (future): compile-time type introspection
```

_See also: CPU Architecture book, Chapter on Branch Prediction for cache and branch effects of virtual dispatch._

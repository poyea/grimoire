= SIMD (Single Instruction Multiple Data)

SIMD processes multiple data elements with single instruction using vector registers. Essential for high-performance computing, multimedia, and data-parallel workloads.

*See also:* CPU Fundamentals (for register overview), Superscalar (for execution units), Cache Hierarchy (for bandwidth requirements)

== SIMD Evolution (x86)

The x86 SIMD instruction set has evolved significantly over time, with each generation introducing distinct syntax and semantic differences. MMX (1997) introduced 64-bit registers (mm0-mm7) supporting only integer operations. SSE (1999) expanded to 128-bit registers (xmm0-xmm15) with support for 4× float operations, introducing a fundamentally different programming model. SSE2 (2001) maintained the 128-bit width but added 2× double and integer operations, creating semantic variations in data type handling. AVX (2011) doubled the register width to 256-bit (ymm0-ymm15) for 8× float operations, requiring new instruction encodings and introducing non-destructive three-operand syntax. AVX2 (2013) extended 256-bit support to integer operations. AVX-512 (2017) further doubled register width to 512-bit (zmm0-zmm31) for 16× float operations and added masking predicates, fundamentally changing the semantic model for conditional operations.

```
MMX (1997):   64-bit registers (mm0-mm7), integer only
SSE (1999):   128-bit registers (xmm0-xmm15), 4× float
SSE2 (2001):  128-bit, 2× double, integer ops
AVX (2011):   256-bit registers (ymm0-ymm15), 8× float
AVX2 (2013):  256-bit integer operations
AVX-512 (2017): 512-bit registers (zmm0-zmm31), 16× float
```

The current mainstream is AVX2 (256-bit), which is ubiquitous, while AVX-512 is found primarily in server CPUs. This distinction is critical for profiling, as AVX-512 usage can trigger frequency throttling that profilers must account for when interpreting performance measurements.

== Vector Registers

*AVX2 (256-bit):*

```
ymm0-ymm15: 256-bit vector registers

Interpretations:
- 32 × 8-bit integers
- 16 × 16-bit integers
- 8 × 32-bit integers
- 4 × 64-bit integers
- 8 × 32-bit floats
- 4 × 64-bit doubles
```

*Layout:*

```
ymm0 (256-bit):
├────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┤
│ [7]    │ [6]    │ [5]    │ [4]    │ [3]    │ [2]    │ [1]    │ [0]    │
│ 32-bit │ 32-bit │ 32-bit │ 32-bit │ 32-bit │ 32-bit │ 32-bit │ 32-bit │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘

Lower half (xmm0) = bits 127:0
Upper half = bits 255:128
```

== Intrinsics Example

Scalar code processes elements sequentially, requiring 8 cycles to add 8 integers:

```c
// Add 8 integers, scalar
for (int i = 0; i < 8; i++)
    c[i] = a[i] + b[i];
// Latency: 8 × 1 cycle = 8 cycles
```

SIMD addition processes all 8 integers in parallel using vector instructions. The syntax uses type-specific intrinsics (`__m256i` for 256-bit integers) and function names encode the operation and data type (`_mm256_add_epi32` for extended packed integer 32-bit addition). This semantic model treats data as vectors rather than scalars, completing in 1 cycle for an 8× speedup:

```c
#include <immintrin.h>

// Add 8 integers, SIMD
__m256i va = _mm256_loadu_si256((__m256i*)a);
__m256i vb = _mm256_loadu_si256((__m256i*)b);
__m256i vc = _mm256_add_epi32(va, vb);
_mm256_storeu_si256((__m256i*)c, vc);

// Latency: 1 cycle (8× faster!)
```

The intrinsic naming convention encodes critical semantic information. The prefix (`_mm256`) indicates the instruction set and register width, the operation (`add`, `mul`, `cmp`) specifies the computation, and the suffix (`epi32`, `ps`, `pd`) defines the data type and packing. Load and store operations have distinct aligned and unaligned variants with different performance characteristics. When profiling, aligned loads (`_mm256_load_si256`) complete in 1 cycle while unaligned loads (`_mm256_loadu_si256`) incur a 1-2 cycle penalty that may not be immediately apparent in simple cycle counts but becomes visible in cache miss patterns and bandwidth utilization.

```c
// Load/store - syntax distinguishes alignment semantics
__m256i _mm256_load_si256(__m256i* addr);     // Aligned (32-byte)
__m256i _mm256_loadu_si256(__m256i* addr);    // Unaligned (+1-2 cycle penalty)
void _mm256_store_si256(__m256i* addr, __m256i val);

// Arithmetic - suffix encodes element type and operation semantics
__m256i _mm256_add_epi32(__m256i a, __m256i b);     // 8× 32-bit add
__m256i _mm256_mullo_epi32(__m256i a, __m256i b);   // 8× 32-bit multiply (low)
__m256 _mm256_add_ps(__m256 a, __m256 b);           // 8× float add
__m256d _mm256_add_pd(__m256d a, __m256d b);        // 4× double add

// Comparison - produces mask with different semantics than scalar comparisons
__m256i _mm256_cmpgt_epi32(__m256i a, __m256i b);   // a > b → mask
__m256 _mm256_cmp_ps(__m256 a, __m256 b, _CMP_LT_OQ); // a < b

// Bitwise - operate on vector as whole
__m256i _mm256_and_si256(__m256i a, __m256i b);
__m256i _mm256_or_si256(__m256i a, __m256i b);

// Horizontal operations (expensive!) - different execution pattern
__m256i _mm256_hadd_epi32(__m256i a, __m256i b);    // Pairwise add within vector
```

== Alignment Requirements

Alignment represents a critical semantic distinction in SIMD programming. Aligned loads assume the memory address is a multiple of the vector width (32 bytes for AVX2), allowing the CPU to fetch data in a single operation. The syntax explicitly encodes this semantic requirement through distinct function names:

```c
__m256i* ptr = (__m256i*)aligned_alloc(32, size);
__m256i val = _mm256_load_si256(ptr);  // Fast (1 cycle)
```

Unaligned loads make no alignment assumptions, requiring the CPU to potentially fetch data across cache line boundaries. The semantic difference is syntactically marked by the 'u' in the function name:

```c
__m256i val = _mm256_loadu_si256((__m256i*)ptr);  // +1-2 cycle penalty
```

While aligned loads are preferred, modern CPUs handle unaligned loads reasonably well. However, when profiling SIMD code, alignment issues manifest not just in cycle counts but also in cache behavior and memory bandwidth utilization. Performance counters for split loads and cache line boundary crossings can reveal alignment problems that cycle-based profiling alone might miss.

== Auto-Vectorization

Compilers can automatically transform scalar loops into SIMD code, fundamentally changing the program's semantic execution model from sequential to parallel processing. The compiler analyzes the scalar code and generates vector instructions when dependencies allow:

```c
// Source
void add(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

// Compiled with -O3 -march=native (AVX2)
// Compiler generates:
for (int i = 0; i < n; i += 8) {
    __m256 va = _mm256_loadu_ps(&a[i]);
    __m256 vb = _mm256_loadu_ps(&b[i]);
    __m256 vc = _mm256_add_ps(va, vb);
    _mm256_storeu_ps(&c[i], vc);
}
// Cleanup loop for remaining elements (i < n % 8)
```

Auto-vectorization succeeds only when specific semantic requirements are met: countable loops with known bounds, no loop-carried dependencies that would violate parallel execution semantics, simple data types (int, float, double), no function calls unless inlinable, and no pointer aliasing (use `__restrict` to guarantee this semantic property).

When profiling, it's crucial to verify whether auto-vectorization occurred, as vectorized and non-vectorized versions have completely different performance characteristics. The compiler can report vectorization decisions:

```bash
gcc -O3 -march=native -fopt-info-vec-all code.c
# Outputs: code.c:5:5: note: loop vectorized
```

When profiling with performance counters, look for the specific SIMD instruction events (e.g., `FP_ARITH.SCALAR_SINGLE` vs `FP_ARITH.256B_PACKED_SINGLE` on Intel) to confirm that SIMD instructions are actually executing, as the compiler may silently fall back to scalar code.

== Performance Patterns

*Throughput example (AVX2, 256-bit):*

```c
// 8 floats per cycle
__m256 add = _mm256_add_ps(a, b);  // Latency: 4 cycles, Throughput: 0.5 CPI
// 2 ports can execute → 2 adds per cycle → 16 floats/cycle!
```

*Memory bandwidth-bound:*

```c
// Copy 1 GB data
for (int i = 0; i < N; i += 8) {
    __m256 val = _mm256_loadu_ps(&src[i]);
    _mm256_storeu_ps(&dst[i], val);
}

// Bandwidth: ~34 GB/s (DDR4-2400 dual-channel)
// Compute: AVX throughput = 100+ GB/s (not limiting factor)
```

*Compute-bound:*

```c
// Complex computation per element
for (int i = 0; i < N; i++) {
    c[i] = sqrt(a[i]) * log(b[i]) + sin(a[i] * b[i]);
}
// SIMD helps: 8× parallelism, but still compute-limited
```

== SIMD Width Tradeoff

Wider SIMD instructions offer greater parallelism but introduce complex tradeoffs that profilers must account for. AVX-512's 512-bit width enables 16× float operations but draws approximately 2× the power of AVX2, triggering CPU frequency throttling. This throttling represents a fundamental semantic difference: the same instruction sequence executes at different speeds depending on the instruction mix, violating the traditional assumption that instruction timing is deterministic.

```
AVX-512 (512-bit): 16× float operations
Power: 2× AVX2 power draw
Clock throttling: CPU may reduce frequency under AVX-512 load

Intel Turbo Boost:
- Scalar code: 4.5 GHz
- AVX2 code: 4.0 GHz
- AVX-512 code: 3.5 GHz

Effective speedup: 16 ops × 3.5 GHz / (8 ops × 4.0 GHz) = 1.75× (not 2×!)
```

When profiling SIMD code, frequency scaling creates measurement challenges. Cycle counters continue at the reduced frequency, making direct comparisons misleading. Profilers should monitor the `CPU_CLK_UNHALTED.THREAD` and `CPU_CLK_UNHALTED.REF_TSC` counters to detect frequency changes. Additionally, the `CORE_POWER.LICENSE` events on Intel CPUs indicate when the processor enters different power states that affect instruction throughput.

AVX-512 should be used strategically: when memory-bound (where bandwidth matters more than frequency), for long-running compute kernels (where throttling is amortized), or in server workloads with adequate power budgets. When profiling such code, distinguish between cycles (which scale with frequency) and wall-clock time (which reflects actual performance).

== Limitations

SIMD operations exhibit significant semantic and performance differences based on data movement patterns. Horizontal operations, which combine elements within a vector, contradict the parallel processing model and incur substantial performance penalties. The syntax for extracting individual elements or using horizontal add intrinsics signals this semantic shift:

```c
// Sum all elements in vector (reduction)
__m256 vec = ...;
float sum = 0;
for (int i = 0; i < 8; i++)
    sum += vec[i];  // Extracting elements is slow!

// Better: Use horizontal add (still slower than vertical ops)
__m256 temp = _mm256_hadd_ps(vec, vec);  // 4 cycles
temp = _mm256_hadd_ps(temp, temp);
sum = _mm256_cvtss_f32(temp);
```

When profiling, horizontal operations appear in different execution ports and have higher latencies than vertical operations. Performance counters like `UOPS_EXECUTED.PORT5` on Intel show increased utilization for shuffle operations, revealing the hidden cost of horizontal patterns.

Cross-lane operations, which move data between the independent 128-bit lanes of 256-bit registers, represent another semantic boundary with performance implications:

```c
// Permute elements across 128-bit lanes
__m256i _mm256_permutevar8x32_epi32(__m256i a, __m256i idx);  // 3 cycles (expensive)
```

AVX-512 introduces fundamentally different semantics through masked operations, where predicates control which vector elements participate in operations. This changes the execution model from unconditional parallelism to conditional parallelism:

```c
// Conditional SIMD (AVX-512)
__mmask16 mask = _mm512_cmplt_ps_mask(a, b);  // a < b
__m512 result = _mm512_mask_add_ps(c, mask, a, b);  // Add only where mask true
```

When profiling masked operations, note that even masked-off elements may consume execution resources, and performance doesn't scale linearly with the number of active mask bits. The `UOPS_RETIRED.MASK_OPS` counter specifically tracks masked instructions, providing visibility into this execution pattern.

== Auto-Vectorization Troubleshooting

*Why didn't my loop vectorize?*

```bash
# Check vectorization report
gcc -O3 -march=native -fopt-info-vec-missed -fopt-info-vec-all code.c

# Common reasons for vectorization failure:
```

*1. Pointer aliasing:*

```c
// BAD: Compiler assumes pointers may overlap
void add(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];  // Not vectorized: a, b, c might overlap!
}

// GOOD: Tell compiler pointers don't alias
void add(float* __restrict a, float* __restrict b, float* __restrict c, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];  // Vectorized!
}
```

*2. Loop-carried dependencies:*

```c
// BAD: Dependency prevents vectorization
for (int i = 1; i < n; i++)
    a[i] = a[i-1] + b[i];  // a[i] depends on a[i-1]

// Can't be trivially vectorized due to true dependency
// Consider algorithmic change if possible
```

*3. Function calls:*

```c
// BAD: Unknown function prevents vectorization
for (int i = 0; i < n; i++)
    c[i] = compute(a[i], b[i]);  // Opaque function call

// GOOD: Inline or use intrinsics
inline float compute(float a, float b) { return a * b + 1.0f; }
for (int i = 0; i < n; i++)
    c[i] = compute(a[i], b[i]);  // Vectorized!
```

*4. Non-constant bounds:*

```c
// BAD: Non-compile-time-constant bound
for (int i = 0; i < get_size(); i++)  // Function call
    c[i] = a[i] + b[i];

// GOOD: Store bound in variable
int n = get_size();
for (int i = 0; i < n; i++)
    c[i] = a[i] + b[i];  // Vectorized!
```

*5. Complex control flow:*

```c
// BAD: Multiple exits confuse vectorizer
for (int i = 0; i < n; i++) {
    if (a[i] < 0) break;
    if (b[i] > 100) continue;
    c[i] = a[i] + b[i];
}

// BETTER: Simplify control flow or use masking
for (int i = 0; i < n && a[i] >= 0; i++) {
    if (b[i] <= 100)
        c[i] = a[i] + b[i];
}
```

== Verifying Vectorization

*Check generated assembly:*

```bash
# Generate assembly
gcc -O3 -march=native -S code.c -o code.s

# Look for vector instructions
grep -E 'vmov|vadd|vmul|vfma' code.s

# AVX2: Look for ymm registers (256-bit)
# AVX-512: Look for zmm registers (512-bit)
```

*Runtime verification with perf:*

```bash
# Check if SIMD instructions executed
perf stat -e fp_arith_inst_retired.scalar_single,\
fp_arith_inst_retired.128b_packed_single,\
fp_arith_inst_retired.256b_packed_single,\
fp_arith_inst_retired.512b_packed_single ./program

# Ratio of vector to scalar shows vectorization effectiveness
```

*Performance debugging:*

```c
// Add pragmas to force/disable vectorization for A/B testing
#pragma GCC optimize("tree-vectorize")
void vectorized_version() { /* ... */ }

#pragma GCC optimize("no-tree-vectorize")
void scalar_version() { /* ... */ }

// Compare performance to verify SIMD benefit
```

== Common SIMD Pitfalls

*1. Alignment faults:*

```c
// BAD: Unaligned allocation
float* data = (float*)malloc(n * sizeof(float));
__m256 vec = _mm256_load_ps(data);  // May fault if not 32-byte aligned!

// GOOD: Aligned allocation
float* data = (float*)aligned_alloc(32, n * sizeof(float));
__m256 vec = _mm256_load_ps(data);  // Safe

// SAFER: Use unaligned load (small penalty)
__m256 vec = _mm256_loadu_ps(data);  // Always safe, 1-2 cycle penalty
```

*2. Mixing vector widths:*

```c
// BAD: Mixing AVX (256-bit) and SSE (128-bit) → transition penalty
__m128 a = _mm_load_ps(data);      // SSE
__m256 b = _mm256_load_ps(data2);  // AVX
// Transition penalty: ~70 cycles!

// GOOD: Stick to one vector width
__m256 a = _mm256_castps128_ps256(_mm_load_ps(data));
__m256 b = _mm256_load_ps(data2);
```

*3. Premature optimization:*

```c
// WRONG APPROACH: Start with intrinsics
void process(float* data, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256 vec = _mm256_loadu_ps(&data[i]);
        // ... complex intrinsics ...
    }
}

// RIGHT APPROACH: Start simple, let compiler vectorize
void process(float* __restrict data, int n) {
    #pragma GCC ivdep  // Hint: no loop-carried dependencies
    for (int i = 0; i < n; i++) {
        data[i] = data[i] * 2.0f + 1.0f;
    }
}
// Profile first, use intrinsics only if compiler fails!
```

== References

Intel Corporation (2023). Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 1: Basic Architecture. Chapter 10 (Programming with Intel AVX).

Intel Corporation (2023). Intel Intrinsics Guide. https://www.intel.com/content/www/us/en/docs/intrinsics-guide/

Fog, A. (2023). Optimizing Software in C++. Technical University of Denmark. Chapter 13 (Vectorization).

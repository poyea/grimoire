= SIMD (Single Instruction Multiple Data)

SIMD processes multiple data elements with single instruction using vector registers. Essential for high-performance computing, multimedia, and data-parallel workloads.

*See also:* CPU Fundamentals (for register overview), Superscalar (for execution units), Cache Hierarchy (for bandwidth requirements)

== SIMD Evolution (x86)

```
MMX (1997):   64-bit registers (mm0-mm7), integer only
SSE (1999):   128-bit registers (xmm0-xmm15), 4× float
SSE2 (2001):  128-bit, 2× double, integer ops
AVX (2011):   256-bit registers (ymm0-ymm15), 8× float
AVX2 (2013):  256-bit integer operations
AVX-512 (2017): 512-bit registers (zmm0-zmm31), 16× float
```

*Current mainstream:* AVX2 (256-bit) ubiquitous, AVX-512 in server CPUs.

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

*Scalar addition:*
```c
// Add 8 integers, scalar
for (int i = 0; i < 8; i++)
    c[i] = a[i] + b[i];
// Latency: 8 × 1 cycle = 8 cycles
```

*SIMD addition:*
```c
#include <immintrin.h>

// Add 8 integers, SIMD
__m256i va = _mm256_loadu_si256((__m256i*)a);
__m256i vb = _mm256_loadu_si256((__m256i*)b);
__m256i vc = _mm256_add_epi32(va, vb);
_mm256_storeu_si256((__m256i*)c, vc);

// Latency: 1 cycle (8× faster!)
```

*Common intrinsics:*

```c
// Load/store
__m256i _mm256_load_si256(__m256i* addr);     // Aligned (32-byte)
__m256i _mm256_loadu_si256(__m256i* addr);    // Unaligned (+1-2 cycle penalty)
void _mm256_store_si256(__m256i* addr, __m256i val);

// Arithmetic
__m256i _mm256_add_epi32(__m256i a, __m256i b);     // 8× 32-bit add
__m256i _mm256_mullo_epi32(__m256i a, __m256i b);   // 8× 32-bit multiply (low)
__m256 _mm256_add_ps(__m256 a, __m256 b);           // 8× float add
__m256d _mm256_add_pd(__m256d a, __m256d b);        // 4× double add

// Comparison
__m256i _mm256_cmpgt_epi32(__m256i a, __m256i b);   // a > b → mask
__m256 _mm256_cmp_ps(__m256 a, __m256 b, _CMP_LT_OQ); // a < b

// Bitwise
__m256i _mm256_and_si256(__m256i a, __m256i b);
__m256i _mm256_or_si256(__m256i a, __m256i b);

// Horizontal operations (expensive!)
__m256i _mm256_hadd_epi32(__m256i a, __m256i b);    // Pairwise add within vector
```

== Alignment Requirements

*Aligned loads (32-byte boundary):*
```c
__m256i* ptr = (__m256i*)aligned_alloc(32, size);
__m256i val = _mm256_load_si256(ptr);  // Fast (1 cycle)
```

*Unaligned loads:*
```c
__m256i val = _mm256_loadu_si256((__m256i*)ptr);  // +1-2 cycle penalty
```

*Performance:* Aligned loads preferred, but modern CPUs handle unaligned reasonably well.

== Auto-Vectorization

*Compiler can automatically vectorize loops:*

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

*Requirements for auto-vectorization:*
1. Countable loop (known bounds)
2. No loop-carried dependencies
3. Simple data types (int, float, double)
4. No function calls (or inlinable)
5. No pointer aliasing (use `__restrict`)

*Checking vectorization:*
```bash
gcc -O3 -march=native -fopt-info-vec-all code.c
# Outputs: code.c:5:5: note: loop vectorized
```

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

*Wider SIMD = more power:*

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

*Use AVX-512 only when:*
- Memory-bound (bandwidth matters more than frequency)
- Long-running compute (throttling amortized)
- Server workloads (power budget allows)

== Limitations

*Horizontal operations slow:*

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

*Cross-lane operations expensive:*

```c
// Permute elements across 128-bit lanes
__m256i _mm256_permutevar8x32_epi32(__m256i a, __m256i idx);  // 3 cycles (expensive)
```

*Masked operations (AVX-512 only):*

```c
// Conditional SIMD (AVX-512)
__mmask16 mask = _mm512_cmplt_ps_mask(a, b);  // a < b
__m512 result = _mm512_mask_add_ps(c, mask, a, b);  // Add only where mask true
```

== References

Intel Corporation (2023). Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 1: Basic Architecture. Chapter 10 (Programming with Intel AVX).

Intel Corporation (2023). Intel Intrinsics Guide. https://www.intel.com/content/www/us/en/docs/intrinsics-guide/

Fog, A. (2023). Optimizing Software in C++. Technical University of Denmark. Chapter 13 (Vectorization).

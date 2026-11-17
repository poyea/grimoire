#import "../template.typ": xref

= Writing Code for the Optimiser <compiler-aware-optimization>

An optimising compiler is a conservative prover. It transforms code only when it can show the transformation preserves the program's meaning under the language rules, and much of the time it cannot: two pointers might alias, a called function might modify memory, a floating-point sum might change if reordered. Code that is slow for these reasons is not fixed by better algorithms or faster hardware. It is fixed by writing code that gives the compiler what it needs to prove, or by doing the transformation by hand. This chapter covers the main obstacles (aliasing, opaque calls, floating-point semantics), latency-bound dependency chains and how to break them, branches and branchless code, inlining and abstraction costs, how to see what the compiler actually did, and why measured differences of a few percent are often noise from code layout.

*See also:* #xref("compilers", "optimisation-passes", label: "Optimisation Passes"), #xref("compilers", "dataflow-analysis", label: "Dataflow Analysis") (alias analysis), #xref("cpu-architecture", "simd", label: "SIMD (Single Instruction Multiple Data)"), #xref("cpu-architecture", "branch-prediction", label: "Branch Prediction"), #xref("performance-engineering", "cpu-profiling", label: "CPU Profiling").

== What the Compiler Must Prove

Most missed optimisations come from a short list of facts the compiler cannot establish:

#table(
  columns: 3,
  [*Obstacle*], [*What it blocks*], [*Typical fix*],
  [Possible aliasing between pointers or references], [Keeping values in registers, vectorisation, reordering loads and stores], [Copy to locals; `__restrict`; restructure data],
  [Calls to functions in other translation units], [Any assumption about memory across the call], [Inlining (visible definitions, LTO), `const` locals],
  [Floating-point semantics], [Reassociating sums and products, so reductions do not vectorise], [Multiple accumulators by hand; targeted fast-math flags],
  [Possible exceptions or side effects], [Moving code across calls, removing apparently dead code], [`noexcept`, pure functions, simpler control flow],
  [Indirect calls (virtual, function pointers, `std::function`)], [Inlining and everything after it], [Templates, `final`, devirtualisation through LTO and PGO],
  [Unknown trip counts and bounds], [Unrolling, vectorisation without run-time checks], [Fixed-size loops, hoisted bounds],
)

== Aliasing

```cpp
#include <cstddef>

// The compiler must assume out[i] might be *scale or overlap in[], so it reloads
// *scale on every iteration and cannot vectorise without run-time overlap checks.
void scale_slow(float* out, const float* in, const float* scale, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = in[i] * *scale;
}

// Copying the scalar to a local removes one alias; __restrict promises the rest.
void scale_fast(float* __restrict out, const float* __restrict in, const float* scale,
                std::size_t n) {
    const float s = *scale;
    for (std::size_t i = 0; i < n; ++i) out[i] = in[i] * s;
}
```

In the first version, every store to `out[i]` might change `*scale`, so the load cannot be hoisted. Compilers often vectorise such loops anyway by emitting a run-time overlap check with a scalar fallback, which costs a little and fails for more complex access patterns. `__restrict` (non-standard in C++ but supported by GCC, Clang and MSVC) is a promise: if the pointers do alias, behaviour is undefined.

Two rules of the language matter here. *Strict aliasing* lets the compiler assume that pointers to different types (say `float*` and `int*`) do not alias, which helps. But `char`, `unsigned char` and `std::byte` may alias anything, so code that writes through a `uint8_t*` (usually `unsigned char`) into a buffer forces the compiler to assume every other object might have changed. Byte-oriented parsers and serialisers commonly lose performance this way: a length field loaded through a struct pointer is reloaded after every byte store. Keeping such values in local variables fixes it.

Member functions have the same issue with `this`. In a loop over `data_[i]` that also updates `count_`, the compiler must assume a store to `data_[i]` might modify `count_` or the vector's own size and pointer. Hoisting `size()` and accumulating into a local before writing the member back is a common, real speedup.

== Latency Chains and Instruction-Level Parallelism

A modern out-of-order core can start several instructions per cycle, but only if they are independent. A chain of dependent operations runs at the *latency* of each operation, not at the *throughput* of the units.

```cpp
#include <cstddef>

// One accumulator: each addition waits for the previous one. On a core with
// 4-cycle FP add latency, this runs at one element per 4 cycles.
float sum_naive(const float* x, std::size_t n) {
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i) s += x[i];
    return s;
}

// Independent accumulators let additions overlap. With two FP add units of latency 4,
// eight independent chains keep both units busy every cycle.
float sum_unrolled(const float* x, std::size_t n) {
    float acc[8] = {};
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8)
        for (int k = 0; k < 8; ++k) acc[k] += x[i + k];
    float s = 0.0f;
    for (float a : acc) s += a;
    for (; i < n; ++i) s += x[i];
    return s;
}
```

The compiler will not make this change by itself, because floating-point addition is not associative: the two functions can return different results. With `-ffast-math` or `-fassociative-math` it may reassociate and vectorise. That flag set also assumes no NaNs or infinities and changes other semantics, so many projects instead write the multiple accumulators explicitly, or use `#pragma omp simd reduction(+ : s)` on the specific loop. The explicit version also has a numerical benefit: pairwise-like summation has a smaller error bound than one long running sum.

The same principle applies everywhere. Pointer chasing in a linked list is a latency chain through memory (each load's address depends on the previous load), so traversal runs at memory latency, about 100 ns per node on a cache miss, while an array of the same data streams at memory bandwidth. Hash table probes, tree walks and interpreters all benefit from restructuring to expose independent work, for example by processing several independent lookups interleaved so their misses overlap.

== Branches

A correctly predicted branch is nearly free; a mispredicted one costs about 15 to 20 cycles on current cores. Data-dependent branches on unpredictable data are the problem.

```cpp
#include <cstddef>
#include <cstdint>

// Branchy: on random data, half the comparisons mispredict.
std::size_t count_below_branchy(const std::int32_t* x, std::size_t n, std::int32_t t) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < n; ++i)
        if (x[i] < t) ++c;
    return c;
}

// Branchless: the comparison result is added as 0 or 1.
std::size_t count_below(const std::int32_t* x, std::size_t n, std::int32_t t) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < n; ++i) c += static_cast<std::size_t>(x[i] < t);
    return c;
}
```

Compilers often convert simple cases like this one to conditional moves or vector compares themselves, but not reliably, and not when the branch guards a store or a function call. Branchless code is not always faster: when the branch is predictable, the branchy version skips work that the branchless one always does. Rules of thumb:

- Measure branch misses (`perf stat -e branch-misses`) before rewriting.
- Sorting or partitioning data so that branches become predictable can be worth more than any rewrite of the loop body.
- `[[likely]]` and `[[unlikely]]` (C++20) and `__builtin_expect` affect code layout (keeping the hot path contiguous), not the hardware predictor, which learns from history regardless.
- Mark error-handling functions `[[gnu::cold]]` or `noinline` so they move out of the hot code.

== Inlining and the Cost of Abstraction

Inlining is the gateway optimisation: once a call is inlined, constant propagation, vectorisation and alias analysis can see through it. Abstraction is free in C++ when it inlines (iterators, small wrappers, lambdas passed to templates) and expensive when it does not.

- `std::function`, virtual calls and function pointers usually block inlining. Passing a callable as a template parameter (`template <class F> void for_each(F&& f)`) keeps its body visible.
- Functions defined in `.cpp` files cannot be inlined into other files without LTO.
- Inlining has limits: compilers use size heuristics, and forcing large functions inline (`[[gnu::always_inline]]`) can bloat hot loops and hurt the instruction cache. Use it surgically, for small functions in hot paths that the compiler declined to inline.
- Exceptions are cheap on the non-throwing path with table-based unwinding, but they can still limit optimisation around calls that might throw; `noexcept` on leaf functions helps the compiler and the standard library (for example, `std::vector` moves elements on reallocation only if the move constructor is `noexcept`).

== Small Things That Add Up

- *Division.* Integer division costs about 10 to 90 cycles depending on operand width and core generation, but division by a compile-time constant becomes a multiply and shift. When the divisor is a run-time value used many times, libraries such as libdivide precompute the same transformation.
- *Bounds checks.* `at()` and checked indexing add a compare and branch per access; usually cheap, but they can block vectorisation. Hoist the check outside the loop (check the range once) when profiling shows it matters.
- *Loop-invariant calls.* `for (i = 0; i < s.size(); ++i)` is fine for `std::vector` (inlined, and the size is hoisted when the compiler can see the loop does not modify the vector), but `strlen(s)` in a loop condition is $O(n^2)$.
- *Data types.* Using 32-bit indices and values where 64 are not needed doubles the elements per cache line and SIMD register.
- *Memory allocation in hot loops.* Allocation is a function call into a complex allocator; reuse buffers.

== Seeing What the Compiler Did

Guessing is unreliable; check.

- *Optimisation remarks*: Clang's `-Rpass=loop-vectorize`, `-Rpass-missed=loop-vectorize` and `-Rpass-analysis=loop-vectorize` explain which loops were vectorised and why others were not; GCC's `-fopt-info-vec-missed` does the same. The LLVM `opt-viewer` tool turns remarks into annotated source.
- *Assembly*: Compiler Explorer (godbolt.org) shows the code for any compiler and flags side by side. Look for the inner loop: vector instructions (`vaddps`, `vfmadd`), unexpected calls, reloads of values that should be in registers.
- *Static throughput analysis*: llvm-mca and uiCA simulate a loop body on a specific microarchitecture and report the bottleneck (a port, a dependency chain, the front end). They answer "what is the best this loop can do" without running it.
- *Hardware counters*: top-down analysis (#xref("performance-engineering", "cpu-profiling", label: "CPU Profiling")) says whether a loop is front-end bound, back-end bound on memory or core, or losing slots to bad speculation, which tells you which of the techniques above applies.

== Measurement Bias from Code Layout

Changing an unrelated part of a program, the order of object files at link time, or even the size of the environment variables can change performance by several percent, because it moves code and stack data to different alignments, changing how loops fit in the decoded-instruction cache, which branches alias in predictor tables, and which cache sets hot data map to. Mytkowicz and colleagues showed such effects are large enough to reverse conclusions in published compiler research. A 3 percent speedup from a code change may be entirely layout noise.

Defences: measure several builds with randomised layout (Stabilizer randomised code and heap layout at run time for this purpose; modern linkers can shuffle sections), repeat measurements across different binaries and machines, and be suspicious of small effects that cannot be explained by counters such as instructions retired or cache misses.

== Pitfalls

- Assuming the compiler will "figure it out" across aliasing pointers or opaque calls.
- Enabling `-ffast-math` globally to get one loop vectorised, and silently changing NaN handling and accuracy everywhere.
- Rewriting predictable branches as branchless code that always does more work.
- Forcing inlining of large functions and inflating the hot code footprint.
- Trusting a small speedup measured on one binary without controlling for layout effects.

== Further Reading

Fog, A. (2024). "Optimizing Software in C++." Technical University of Denmark, optimisation manual.

Fog, A. (2024). "Instruction Tables: Lists of Instruction Latencies, Throughputs and Micro-operation Breakdowns." Technical University of Denmark, reference.

Intel Corporation (2024). "Intel 64 and IA-32 Architectures Optimization Reference Manual." Intel documentation.

Mytkowicz, T. et al. (2009). "Producing Wrong Data Without Doing Anything Obviously Wrong!" ASPLOS.

Curtsinger, C. and Berger, E. D. (2013). "Stabilizer: Statistically Sound Performance Evaluation." ASPLOS.

Abel, A. and Reineke, J. (2022). "uiCA: Accurate Throughput Prediction of Basic Blocks on Recent Intel Microarchitectures." ICS.

Granlund, T. and Montgomery, P. L. (1994). "Division by Invariant Integers Using Multiplication." PLDI.

Lemire, D., Kaser, O. and Kurz, N. (2019). "Faster Remainder by Direct Computation: Applications to Compilers and Software Libraries." Software: Practice and Experience.

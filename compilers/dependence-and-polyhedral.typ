#import "../template.typ": definition, theorem, proof, xref

= Dependence Analysis and the Polyhedral Model <dependence-and-polyhedral>

Loop transformations such as interchange, tiling, fusion, skewing, and parallelisation reorder the iterations of a loop nest. They are worth doing because they change memory access patterns and expose parallelism, often by an order of magnitude on dense numerical code. They are legal only if every pair of iterations that depend on each other still executes in the original order. Dependence analysis computes which iterations depend on which, and the theory of unimodular and affine transformations says which reorderings respect those dependences. The polyhedral model generalises both into one algebraic framework in which a compiler can search for the best legal schedule. This chapter covers dependence kinds and vectors, dependence tests from GCD to Omega, the legality condition for loop transformations with a C++ checker, the polyhedral representation and schedulers, and where the model is used and where it stops.

*See also:* #xref("compilers", "optimisation-passes", label: "Optimisation Passes") (loop transformations and vectorisation), #xref("compilers", "dataflow-analysis", label: "Dataflow Analysis"), #xref("compilers", "ir-design", label: "IR Design") (MLIR), #xref("performance-engineering", "memory-performance", label: "Memory Performance"), #xref("numerical-computing", "linear-systems", label: "Linear Systems").

== Dependences

Two statement instances *depend* on each other if they access the same memory location and at least one access is a write. Three kinds exist, named by the order of the accesses in the original program:

#table(
  columns: 3,
  [*Kind*], [*Order*], [*Example*],
  [Flow (true, read after write)], [Write, then read], [`a[i] = ...; ... = a[i];`],
  [Anti (write after read)], [Read, then write], [`... = a[i]; a[i] = ...;`],
  [Output (write after write)], [Write, then write], [`a[i] = ...; a[i] = ...;`],
)

Only flow dependences carry values. Anti and output dependences arise from reusing storage, and renaming or array expansion (giving each iteration its own copy) can remove them at a memory cost. Scalars written in every iteration of a loop create anti and output dependences between all iterations; privatisation (one copy per thread) and reduction recognition (for accumulators) remove the most common cases.

== Iteration Spaces and Distance Vectors

In a perfect nest of $n$ loops, each execution of the body is identified by its *iteration vector* $arrow(i) = (i_1, dots, i_n)$. Sequential execution visits iteration vectors in *lexicographic* order: $arrow(i) prec arrow(j)$ if they agree on a prefix and $arrow(i)$ is smaller at the first position where they differ.

#definition(name: "Distance and direction vectors")[
If iteration $arrow(j)$ depends on iteration $arrow(i)$, with $arrow(i) prec arrow(j)$, the *distance vector* is $arrow(d) = arrow(j) - arrow(i)$. Its *direction vector* replaces each component by its sign: $<$ for positive, $=$ for zero, and $>$ for negative.
]

Because the source runs first, every distance vector is *lexicographically positive*: its first nonzero component is positive. The *level* of a dependence is the position of that first nonzero component; the dependence is *carried* by the loop at that level. A loop that carries no dependence can run its iterations in parallel.

Consider the stencil

```cpp
for (int i = 1; i < n; ++i)
    for (int j = 1; j < m - 1; ++j)
        a[i][j] = a[i - 1][j + 1] + a[i][j - 1];
```

The read `a[i - 1][j + 1]` at iteration $(i, j)$ reads the value written at iteration $(i - 1, j + 1)$, so there is a flow dependence with distance $(1, -1)$. The read `a[i][j - 1]` reads the value written at $(i, j - 1)$: distance $(0, 1)$. The outer loop carries the first dependence and the inner loop the second, so neither loop is parallel as written.

== Testing for Dependence

For array accesses with *affine* subscripts (linear combinations of loop indices plus constants), deciding dependence means deciding whether a system of linear equations has an integer solution inside the loop bounds. For a single subscript, a write to `a[a1*i + a0]` and a read of `a[b1*i' + b0]` conflict if

$ a_1 i - b_1 i' = b_0 - a_0 quad "with" quad L <= i, i' <= U. $

#theorem(name: "GCD test")[
The linear Diophantine equation $sum_k c_k x_k = c$ has an integer solution if and only if $gcd(c_1, dots, c_m)$ divides $c$.
]

The GCD test ignores the bounds, so it can only prove *independence*: if the gcd does not divide the constant, no dependence exists anywhere. Otherwise it is inconclusive. The *Banerjee test* adds the bounds by checking whether $c$ lies between the minimum and maximum of the left-hand side over the bounded real box, which proves independence when it does not. Exact tests solve the integer program: the *Omega test* (Pugh) uses an integer variant of Fourier-Motzkin elimination and is exact and fast for the small systems that arise in practice. Production compilers chain cheap tests first and fall back to exact ones only when needed.

```cpp
#include <cstdlib>
#include <numeric>
#include <vector>

// a[c_w . i + k_w] written, a[c_r . i + k_r] read; coefficients per loop index.
// Returns false only if the GCD test proves the accesses can never touch the same element.
bool may_depend_gcd(const std::vector<long>& c_w, long k_w,
                    const std::vector<long>& c_r, long k_r) {
    long g = 0;
    for (long c : c_w) g = std::gcd(g, std::labs(c));
    for (long c : c_r) g = std::gcd(g, std::labs(c));
    const long rhs = k_r - k_w;
    if (g == 0) return rhs == 0;
    return rhs % g == 0;
}
```

For `a[2*i] = ...; ... = a[2*i + 1];` the gcd is 2 and the constant is 1, so the even and odd elements never meet and the loop can be vectorised or parallelised freely.

Subscripts that are not affine (indirect `a[idx[i]]`, nonlinear `a[i*j]`) and possible aliasing between different pointer bases make the test give up and assume dependence. Run-time checks are the usual escape: the vectoriser emits a test that the address ranges do not overlap and keeps a scalar fallback.

== Legality of Loop Transformations

Many loop transformations are linear maps on the iteration space. Interchange is a permutation matrix, reversal negates a coordinate, and *skewing* adds a multiple of one index to another. A *unimodular* matrix $T$ (integer entries, determinant $plus.minus 1$) maps the integer points of the iteration space one-to-one onto integer points, so the transformed nest visits exactly the same instances in a new order.

#theorem(name: "Legality of unimodular transformations")[
A unimodular transformation $T$ of a perfect loop nest is legal if and only if $T arrow(d)$ is lexicographically positive for every dependence distance vector $arrow(d)$.
]

#proof[
The transformed nest executes $arrow(i)$ before $arrow(j)$ exactly when $T arrow(i) prec T arrow(j)$, that is when $T(arrow(j) - arrow(i)) = T arrow(d)$ is lexicographically positive, by linearity. The transformation preserves the semantics if every dependent pair keeps its order, which is the stated condition.
]

For the stencil, interchange maps $(1, -1)$ to $(-1, 1)$, which is lexicographically negative, so interchange alone is illegal. Skewing the inner loop by the outer, $j' = j + i$, maps the distances to $(1, 0)$ and $(0, 1)$; after skewing, interchange is legal. Better still, the *wavefront* transformation $T = mat(2, 1; 1, 0)$ maps the distances to $(1, 1)$ and $(1, 0)$: every dependence is now carried by the outer loop, so all iterations of the inner loop can run in parallel. The outer loop index $t = 2i + j$ numbers the wavefronts.

```cpp
#include <iostream>
#include <vector>

using vec = std::vector<long>;
using mat = std::vector<vec>;

vec apply(const mat& t, const vec& d) {
    vec r(t.size(), 0);
    for (std::size_t i = 0; i < t.size(); ++i)
        for (std::size_t j = 0; j < d.size(); ++j)
            r[i] += t[i][j] * d[j];
    return r;
}

bool lex_positive(const vec& v) {
    for (long x : v)
        if (x != 0) return x > 0;
    return false;
}

bool legal(const mat& t, const std::vector<vec>& distances) {
    for (const auto& d : distances)
        if (!lex_positive(apply(t, d))) return false;
    return true;
}

// Loop at depth k (0-based) is parallel if no transformed dependence is carried there.
bool parallel_at(const mat& t, const std::vector<vec>& distances, std::size_t k) {
    for (const auto& d : distances) {
        const vec td = apply(t, d);
        std::size_t level = 0;
        while (level < td.size() && td[level] == 0) ++level;
        if (level == k) return false;
    }
    return true;
}

int main() {
    const std::vector<vec> deps = {{1, -1}, {0, 1}};
    const mat interchange = {{0, 1}, {1, 0}};
    const mat wavefront = {{2, 1}, {1, 0}};
    std::cout << "interchange legal: " << legal(interchange, deps) << '\n'      // 0
              << "wavefront legal: " << legal(wavefront, deps) << '\n'          // 1
              << "inner parallel: " << parallel_at(wavefront, deps, 1) << '\n'; // 1
}
```

*Tiling* is not a linear map on the original space, but its legality follows the same logic. A band of consecutive loops can be tiled with rectangular tiles if it is *fully permutable*: every dependence has nonnegative components in all loops of the band. Skewing is often applied first to make a band fully permutable, which is exactly what the stencil needs.

== The Polyhedral Model

Unimodular theory handles perfect nests with one statement. Real kernels have several statements at different depths, imperfect nests, and triangular bounds. The *polyhedral model* represents such programs exactly, as long as loop bounds, conditions, and subscripts are affine in the loop indices and in symbolic parameters.

For each statement $S$:

- The *iteration domain* $cal(D)_S$ is the set of integer points of a parametric polyhedron, for example ${S[i, j] | 0 <= i < N, 0 <= j <= i}$ for a triangular nest.
- *Access relations* map each instance to the array elements it reads and writes, such as $S[i, j] -> A[i, j - 1]$.
- A *schedule* $theta_S$ maps each instance to a multidimensional logical time; the program executes instances in lexicographic order of their times. The original program's schedule is read off its syntax tree.

Dependences become *dependence polyhedra*: sets of pairs $(S[arrow(i)], T[arrow(j)])$ where the accesses touch the same element and the source precedes the target. A new schedule is legal if, for every pair in every dependence polyhedron, the target's time is lexicographically later than the source's. Because the dependence relations are polyhedra and the schedules are affine, legality is a set of linear constraints on the schedule coefficients.

That turns optimisation into a search over affine schedules. Feautrier's scheduler (1992) used the affine form of Farkas' lemma to linearise the "for all dependent pairs" quantifier and solved a parametric integer program for minimum latency. The *Pluto* algorithm (Bondhugula et al., 2008) instead finds, one dimension at a time, schedule hyperplanes that make dependence distances small and nonnegative. That yields bands of fully permutable loops that can be tiled for locality and parallelised at the outer level, which matches what modern hardware rewards.

After scheduling, *code generation* turns the union of transformed domains back into loops. CLooG and the AST generator in the isl library scan the polyhedra, splitting and merging loops so that each statement executes exactly at its scheduled times, with minimal control overhead.

== Systems Using the Model

#table(
  columns: 3,
  [*System*], [*Role*], [*Notes*],
  [isl], [Integer set library: sets, relations, scheduling, AST generation], [The common core of most current tools],
  [Polly (LLVM)], [Detects static control parts in LLVM IR, optimises them with isl], [Tiling, fusion, outer-loop parallelism, GPU offload],
  [GCC Graphite], [Polyhedral loop optimiser in GCC], [Uses isl],
  [PPCG], [Source-to-source polyhedral compiler for GPUs], [Maps tiled bands to thread blocks and threads],
  [MLIR affine dialect], [Affine loops and maps as first-class IR], [Dependence analysis and transformations in ML compilers],
  [Tensor Comprehensions, Tiramisu], [Polyhedral optimisation of tensor and DSL code], [Research and specialised production use],
)

Halide and TVM take a related but different path: they separate the algorithm from a *schedule* that the programmer or an autotuner writes, rather than deriving the schedule by solving constraints. That trades automatic legality reasoning for direct control and has been more successful in image processing and machine learning, where the kernels are known and performance matters more than generality.

== Limits

- *Static control.* Data-dependent bounds, indirect accesses, and pointer-chasing code are outside the model. Extensions over-approximate them, losing precision.
- *Aliasing and types.* In C and C++, a compiler must first prove that arrays do not overlap, which often requires `restrict` or run-time checks.
- *Compile time.* Integer programming is exponential in the worst case. Scheduling large kernels can take seconds to minutes, which is why polyhedral passes are usually off by default in general-purpose compilers.
- *Profitability.* The model decides legality exactly but has only a crude cost model. Tile sizes, fusion choices, and vector width interact with the cache hierarchy in ways that autotuning measures better than any formula.

The ideas outlived the limits. Dependence vectors and legality conditions sit inside every vectoriser, and the affine view of loops is the basis of MLIR's loop dialects and of most deep-learning compilers.

== Further Reading

Allen, R. and Kennedy, K. (2001). _Optimizing Compilers for Modern Architectures: A Dependence-Based Approach_. Morgan Kaufmann.

Wolfe, M. (1995). _High Performance Compilers for Parallel Computing_. Addison-Wesley.

Banerjee, U. (1988). _Dependence Analysis for Supercomputing_. Kluwer.

Wolf, M. E. and Lam, M. S. (1991). "A Data Locality Optimizing Algorithm." PLDI.

Pugh, W. (1991). "The Omega Test: A Fast and Practical Integer Programming Algorithm for Dependence Analysis." Supercomputing.

Feautrier, P. (1992). "Some Efficient Solutions to the Affine Scheduling Problem. Part I. One-Dimensional Time." International Journal of Parallel Programming.

Bastoul, C. (2004). "Code Generation in the Polyhedral Model Is Easier Than You Think." PACT.

Bondhugula, U. et al. (2008). "A Practical Automatic Polyhedral Parallelizer and Locality Optimizer." PLDI.

Verdoolaege, S. (2010). "isl: An Integer Set Library for the Polyhedral Model." ICMS.

Grosser, T., Größlinger, A. and Lengauer, C. (2012). "Polly: Performing Polyhedral Optimizations on a Low-Level Intermediate Representation." Parallel Processing Letters.

Ragan-Kelley, J. et al. (2013). "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines." PLDI.

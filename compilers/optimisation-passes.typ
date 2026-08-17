#import "../template.typ": xref

= Optimisation Passes

A compiler's middle-end applies a sequence of analysis and transformation passes over the IR. Each pass either analyses — computing properties and annotating the IR — or transforms, rewriting instructions to be faster, smaller, or more parallel. This chapter surveys the canonical scalar optimisations, loop transformations, and vectorisation, and explains how LLVM's pass manager orchestrates them.

*See also:* #xref("compilers", "ir-design", label: "IR Design and SSA Form"), #xref("compilers", "dataflow-analysis", label: "Dataflow Analysis"), #xref("compilers", "register-allocation", label: "Register Allocation")

== Pass Manager Architecture

A pass manager owns the scheduling and dependency tracking of all passes. LLVM has had two pass managers.

The *legacy pass manager* (pre-LLVM 12) registered passes as global objects and resolved dependencies at startup. It supported module, call-graph SCC, function, and loop passes, but had no principled invalidation model: a pass could silently use stale analysis results.

The *new pass manager* (NPM, LLVM 12+) fixes this with an explicit *analysis manager* per pass kind. Each transform declares which analyses it requires; after transformation, it declares which analyses are invalidated. The infrastructure re-runs only what is stale.

Pass kinds:

- *Module pass*: sees the entire module; used for interprocedural transforms (inliner, dead argument elimination).
- *CGSCC pass*: iterates over strongly-connected components of the call graph; needed for correct interprocedural analysis of recursion.
- *Function pass*: per function; GVN, LICM, mem2reg, instcombine.
- *Loop pass*: per loop nest; loop unrolling, LICM inner loops, loop vectoriser.

The *PassBuilder* constructs pipeline strings like `default<O2>` or `function(instcombine,gvn)`. The `opt` command-line tool exposes this API directly, making it easy to experiment:

```sh
opt -passes='function(mem2reg,instcombine,gvn),module(inline)' -S input.ll
```

O1 runs roughly: mem2reg, instcombine, simplifycfg, early-cse, inline (conservative threshold), scalar-evolution.
O2 adds: GVN, LICM, loop-unroll, SLP vectoriser, jump threading.
O3 adds: aggressive inlining, loop vectoriser, polly (if built), argument promotion.

== Scalar Optimisations

*Dead code elimination (DCE)* removes instructions with no uses. In SSA this is trivial: an instruction is dead if its result has no uses and it has no side effects. A single post-order worklist pass suffices; no fixed-point iteration is needed because SSA guarantees every name has exactly one def.

*Common subexpression elimination (CSE)* hoists repeated computations. The local form (within a basic block) is a hash-table lookup on `(opcode, operands)`. Global CSE, which works across blocks, is subsumed by GVN.

*Global Value Numbering (GVN)* assigns a canonical value number to each SSA value by hashing its defining expression. Two values with the same number are provably equal and one can replace the other. GVN also detects *redundant loads*: a load is redundant if a previous load or store to the same address is visible without an intervening aliasing store. LLVM's GVN-hoist and NewGVN passes implement this with a congruence-closure algorithm.

*Sparse conditional constant propagation (SCCP)* is described in the Dataflow Analysis chapter. Briefly: it propagates constants along SSA edges, treating unreachable branches as dead, and outperforms naive constant folding because it avoids propagating through dead code.

*Strength reduction* replaces expensive operations with cheaper ones:

```llvm
; strength reduction: div by constant → multiply by magic number
%r = sdiv i32 %x, 7          ; expensive
; ⇒
%w = sext i32 %x to i64      ; widen so the high word survives
%m = mul i64 %w, 1227133513  ; magic multiply (64-bit product)
%h = lshr i64 %m, 32         ; extract high word
%s = trunc i64 %h to i32
```

The compiler computes the magic constant at compile time (Hacker's Delight, Warren 2012). Loop induction variable strength reduction replaces a multiply inside a loop with an add-by-stride.

*Algebraic identities* are applied by the `instcombine` pass, which canonicalises instructions to a normal form:

```
x * 0  → 0             x + 0  → x
x - x  → 0             x / x  → 1  (with zero guard)
x << 1 → x + x         (x + c1) + c2 → x + (c1+c2)
```

Instcombine runs repeatedly because each canonicalisation may expose further opportunities.

== Inlining

Inlining is the single most impactful optimisation because it exposes the callee body to the caller's context, enabling constant folding, load forwarding, and dead store elimination across the call boundary.

*Benefits:* eliminates call overhead (frame setup, register save/restore, branch misprediction on indirect calls), and specialises the callee on concrete argument values.

*Heuristics:* LLVM's inliner uses a threshold-based cost model. The cost of inlining is approximately the number of instructions in the callee; this cost is reduced by *bonuses* when:

- An argument is a constant (partial evaluation opportunity).
- The call site is in a hot block (profile-guided).
- The callee is small (always-inline attribute).
- The callee has a single call site.

The threshold for O2 is 225; for O3 it is 275. Recursive inlining is suppressed.

*Inlining and code size:* aggressive inlining grows I-cache pressure. The compiler must trade off: inlining `memcpy` is almost always good; inlining a 200-instruction loop body called once per second is wasteful. The `optsize` and `minsize` attributes lower or eliminate the threshold.

*Indirect call promotion:* virtual calls and function pointers cannot be inlined directly. The pass speculates on the most likely target (using profile data), emits an explicit type check, inlines the fast path, and falls back to the original indirect call:

```
; indirect call promotion
if (likely fptr == &foo)
    [inlined body of foo]
else
    fptr(args)     ; slow path
```

*Devirtualisation* identifies class hierarchy constraints to prove which virtual function is called, then converts the indirect call to a direct one before inlining.

== Memory Optimisations

*Mem2Reg* (also called `alloca` promotion) converts stack-allocated scalar variables into SSA registers. The frontend often emits `alloca` + `store` + `load` for every local variable; mem2reg replaces these with direct SSA values and inserts phi nodes as needed. It is the first pass after IR construction and unlocks almost all subsequent scalar optimisations.

```llvm
; Before mem2reg:                   ; After mem2reg:
entry:                              entry:
  %x = alloca i32                     br i1 %c, label %T, label %F
  br i1 %c, label %T, label %F
T:                                  T:
  store i32 1, i32* %x                br %M
  br %M
F:                                  F:
  store i32 2, i32* %x                br %M
  br %M
M:                                  M:
  %v = load i32, i32* %x              %v = phi i32 [1, %T], [2, %F]
```

*Load/store forwarding* eliminates a load when the same address was stored since the last intervening aliasing store. Within a basic block this is a simple scan; across blocks it uses the GVN analysis.

*Dead store elimination (DSE)* removes a store whose value is never read before being overwritten or before the function returns. DSE depends on alias analysis: a store is dead only if there is no pointer that could read it.

*Escape analysis* determines whether a heap-allocated object can be observed outside its allocating function. If an object does not escape (not passed to an opaque function, not stored into a visible global), it can be:

- *Stack-allocated*: replaced `malloc` with `alloca`; no GC interaction.
- *Scalar-replaced*: if it does not escape and its fields are accessed individually, each field becomes an SSA value.

*Scalar replacement of aggregates (SROA)* generalises mem2reg to structs and fixed-size arrays. It splits a struct alloca into one alloca per field, then runs mem2reg on each scalar component.

== Loop Transformations

Loops dominate runtime in numerical and systems code. The loop transformations are applied by the loop pass pipeline.

*Loop-invariant code motion (LICM)* hoists computations whose operands do not change across iterations out of the loop body into the loop preheader. LLVM's LICM also *sinks* loop-invariant instructions to exits when the loop has an early exit path.

```llvm
; before LICM:               ; after LICM:
loop:                        %inv = mul i32 %a, %b    ; hoisted
  %inv = mul i32 %a, %b     loop:
  %r = add i32 %inv, %i       %r = add i32 %inv, %i
```

*Loop unrolling* replicates the loop body $N$ times to reduce branch overhead, expose instruction-level parallelism, and enable better scheduling. Full unrolling is applied when the trip count is a small constant. Partial unrolling with factor $N$ reduces loop overhead by $N times$ at the cost of code size. LLVM's loop unroller also unrolls loops with unknown trip count using a runtime check.

*Loop interchange* swaps the order of a nested loop pair to improve memory access patterns. Column-major traversal of a row-major array causes a cache miss on every access; interchanging the loops achieves unit-stride access:

```
// before interchange (column-major, bad for C):
for (j = 0; j < N; j++)
  for (i = 0; i < M; i++)
    A[i][j] = ...;

// after interchange (row-major, good):
for (i = 0; i < M; i++)
  for (j = 0; j < N; j++)
    A[i][j] = ...;
```

*Loop fusion* merges two adjacent loops over the same iteration space into a single loop, reducing loop overhead and improving reuse of data loaded in one loop for use in the other.

*Loop tiling (blocking)* subdivides the iteration space into rectangular tiles sized to fit in a cache level. For an $N times N$ matrix multiply with tile size $T$:

```
for (ii = 0; ii < N; ii += T)
  for (jj = 0; jj < N; jj += T)
    for (kk = 0; kk < N; kk += T)
      for (i = ii; i < min(ii+T, N); i++)
        for (j = jj; j < min(jj+T, N); j++)
          for (k = kk; k < min(kk+T, N); k++)
            C[i][j] += A[i][k] * B[k][j];
```

The tile loops stay in cache across their inner loops; memory traffic drops from $O(N^3)$ to $O(N^3 / T)$ at the cost of two extra loop levels.

== Auto-Vectorisation

Modern CPUs execute SIMD instructions (SSE, AVX, AVX-512, NEON, SVE) that process 4, 8, or 16 elements per instruction. The compiler auto-vectoriser transforms scalar loops into SIMD loops without programmer effort.

*Loop vectoriser* transforms a scalar loop into a loop over vectors of width VF (vectorisation factor). Each scalar instruction becomes a vector instruction:

```llvm
; scalar (VF=1):              ; vectorised (VF=4):
loop:                         loop:
  %v = load f32, ...            %v = load <4 x f32>, ...
  %r = fmul f32 %v, %k          %r = fmul <4 x f32> %v, %k.splat
  store f32 %r, ...             store <4 x f32> %r, ...
  %i.next = add %i, 1           %i.next = add %i, 4
```

The vectoriser must first run *dependence analysis* to verify no loop-carried dependency prevents vectorisation. A dependence from iteration $i$ to $j$ (with $j > i$) prevents vectorisation if the distance $j - i < "VF"$.

The *interleaving factor* (IF) fetches multiple vector iterations at once to hide memory latency; the effective throughput is $"VF" dot "IF"$ elements per cycle.

*SLP (Superword Level Parallelism) vectoriser* works on straight-line code rather than loops. It finds adjacent scalar operations with isomorphic structure and packs them into a SIMD instruction:

```
a[0] = b[0] + c[0];     →   a[0..3] = b[0..3] + c[0..3];  (SIMD add)
a[1] = b[1] + c[1];
a[2] = b[2] + c[2];
a[3] = b[3] + c[3];
```

LLVM's `LoopVectorize` and `SLPVectorize` passes detect the target's SIMD width from the `TargetTransformInfo` interface and generate target-specific vector intrinsics. The passes also emit a scalar epilogue for iterations that do not fill a full vector.

== Profile-Guided Optimisation (PGO)

Static heuristics (call frequency, code size) are imprecise. PGO feeds measured runtime behaviour back into the optimiser.

*Instrumented PGO* compiles with profiling counters:

```sh
clang -fprofile-instr-generate -O2 -o prog prog.c
./prog < representative_input
llvm-profdata merge prog.profraw -o prog.profdata
clang -fprofile-instr-use=prog.profdata -O2 -o prog prog.c
```

Counters record: block execution counts, edge counts, and indirect call targets. The profile is used for:

- *Inlining decisions*: inline hot callees aggressively; don't inline cold callees.
- *Block layout*: place hot blocks contiguously for I-cache friendliness (Pettis-Hansen algorithm).
- *Branch prediction hints*: annotate branches with `!prof !{!"branch_weights", ...}`.
- *Indirect call promotion*: use the most frequent target from the indirect call histogram.

*Sampling PGO (AutoFDO)* collects profiles from `perf record` on production binaries, then converts them to LLVM format with `create_llvm_prof`. No instrumentation overhead; profiles reflect production behaviour rather than a synthetic workload.

*BOLT* (Binary Optimisation and Layout Tool) is a post-link optimiser. After linking, BOLT reorders basic blocks and functions using profile data to minimise I-cache misses and branch mispredictions. It can deliver 5--15% speedup on large binaries with no source changes.

== Further Reading

Cooper, K., Torczon, L. (2022). _Engineering a Compiler_, 3rd ed. Morgan Kaufmann.

Allen, R., Kennedy, K. (2001). _Optimizing Compilers for Modern Architectures._ Morgan Kaufmann.

Bacon, D., Graham, S., Sharp, O. (1994). "Compiler Transformations for High-Performance Computing." ACM CSUR.

Lattner, C., Adve, V. (2004). "LLVM: A Compilation Framework for Lifelong Program Analysis and Transformation." CGO.

Luk, C.-K. et al. (2005). "Pin: Building Customized Program Analysis Tools with Dynamic Instrumentation." PLDI.

Click, C., Paleczny, M. (1995). "A Simple Graph-Based Intermediate Representation." IR Workshop at POPL.

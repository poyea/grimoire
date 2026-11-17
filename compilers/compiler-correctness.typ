#import "../template.typ": definition, xref

= Compiler Correctness: Undefined Behaviour, Testing, and Validation <compiler-correctness>

Every program's correctness depends on the compiler that translates it, and compilers are large, heavily optimised programs that are themselves full of bugs. A *miscompilation*, where the compiler silently produces code that behaves differently from what the source means, is among the worst bugs in computing: the source looks right, tests of the source reasoning pass, and the failure may appear only at one optimisation level, on one target, for one input. What "behaves differently from what the source means" requires a precise semantics, and for C, C++, and the IRs that compile them, that semantics is shaped by *undefined behaviour*. This chapter covers why undefined behaviour exists and how optimisers exploit it, the IR-level semantics of undef, poison, and freeze, refinement as the correctness criterion with a C++ checker, translation validation with Alive2, random differential testing from Csmith to EMI, and verified compilers.

*See also:* #xref("compilers", "optimisation-passes", label: "Optimisation Passes"), #xref("compilers", "e-graphs", label: "E-Graphs and Equality Saturation") (rule soundness), #xref("formal-methods", "theorem-proving", label: "Theorem Proving") (CompCert), #xref("formal-methods", "symbolic-execution-and-fuzzing", label: "Symbolic Execution and Fuzzing"), #xref("formal-methods", "sat-and-smt", label: "SAT and SMT").

== Undefined Behaviour as a Contract

The C and C++ standards leave many situations *undefined*: signed integer overflow, dereferencing a null or dangling pointer, out-of-bounds access, data races, reading an uninitialised automatic variable in most cases, violating strict aliasing, shifting by at least the bit width, and (in C++) an infinite loop with no observable side effects. Undefined does not mean "implementation-specific result". It means the standard places no requirement on the program at all, and a compiler may assume it never happens.

That assumption is an optimisation contract: the programmer promises to avoid undefined behaviour, and in exchange the compiler may reason as if the promise holds. Some consequences are valuable and invisible:

```cpp
// With 32-bit int i and 64-bit pointers, signed overflow being undefined lets the
// compiler widen i to a 64-bit induction variable and use a single pointer increment,
// because i + 1 can never wrap to a negative value.
float sum(const float* a, int n) {
    float s = 0;
    for (int i = 0; i < n; ++i) s += a[i];
    return s;
}
```

Others are surprising:

```cpp
bool will_overflow(int x) {
    return x + 1 < x;          // folded to false: overflow would be undefined
}

int read_field(struct device* d) {
    int flags = d->flags;      // dereference: d is now assumed non-null
    if (!d) return -1;         // check removed as dead code
    return flags;
}
```

The second pattern caused a real Linux kernel vulnerability in 2009: in the TUN driver, a pointer was dereferenced before its null check, the compiler removed the check, and a mapped page at address zero made the bug exploitable. The kernel has been built with `-fno-delete-null-pointer-checks` ever since. Similar removals of overflow checks written as `if (p + len < p)` have broken security checks in many projects.

The defences are to write checks that are defined (`if (len > end - p)`, or `__builtin_add_overflow`), to compile tests with UndefinedBehaviorSanitizer, and, where the cost is acceptable, to define the behaviour with flags such as `-fwrapv` or `-fno-strict-aliasing`.

== Deferred Undefined Behaviour in the IR

An IR needs its own semantics, and it cannot simply copy the source language's. An optimiser wants to hoist an addition out of a loop or execute both sides of a branch speculatively. If signed overflow in the IR were immediate undefined behaviour, hoisting `a + b` above the loop guard would introduce undefined behaviour on paths where the loop never runs. LLVM therefore distinguishes *immediate* undefined behaviour (division by zero, loads from invalid pointers, branching on poison) from *deferred* forms that represent "this value is garbage" without stopping the program.

#table(
  columns: 3,
  [*Value*], [*Meaning*], [*Problem*],
  [`undef`], [An arbitrary bit pattern, which may differ at each use], [`x = undef; y = x xor x` need not be 0, which breaks many rewrites that duplicate uses],
  [`poison`], [A value whose use in a side effect is undefined; propagates through most operations], [Branching on it is immediate UB, so some speculation becomes illegal],
  [`freeze x`], [If `x` is poison or undef, an arbitrary but fixed value; otherwise `x`], [Costs an instruction; must be inserted where poison must be stopped],
)

Arithmetic flags create poison: `add nsw` (no signed wrap) yields poison on signed overflow, `shl` by too large an amount yields poison, and `getelementptr inbounds` yields poison when leaving the object. The flags record what the front end knows about the source (for C, signed overflow is undefined, so Clang emits `add nsw` for `int` addition), and they let the optimiser reason as the source language permits.

Lee and colleagues showed in 2017 that LLVM's combination of `undef` and `poison` was inconsistent: some optimisations were justified only under one interpretation and others only under another, so applying both could miscompile correct programs. The fix, adopted over several years, was to move toward poison as the single deferred form and to add `freeze` for the places, such as loop unswitching, where a possibly-poison value must be turned into a stable one before it controls a branch.

== Refinement

#definition(name: "Refinement")[
A target program $T$ *refines* a source program $S$ if, for every input, every behaviour that $T$ can exhibit is a behaviour that $S$ can exhibit. A behaviour of $S$ that reaches undefined behaviour allows *every* behaviour, so on such inputs $T$ may do anything.
]

Refinement is the correctness criterion for each compiler transformation. Its asymmetry is the formal content of the UB contract: a transformation may remove behaviours (choosing one of several values that `undef` allowed, assuming an overflow does not occur) but may never add one. For values, poison in the source can be refined by any value in the target, while a concrete value in the source must be matched exactly. In the other direction, introducing poison or UB where the source had none is a miscompilation.

Exhaustive checking at small bit widths is enough to show the idea. The checker below models 8-bit values with a poison flag and tests three rewrites:

```cpp
#include <functional>
#include <iostream>
#include <string>
#include <vector>

struct value {
    int v;          // two's complement value in [-128, 127] for i8, or 0/1 for i1
    bool poison;
};

constexpr value poison_value{0, true};

int wrap8(int x) { return static_cast<signed char>(static_cast<unsigned char>(x & 0xff)); }

value add(value a, value b, bool nsw) {
    if (a.poison || b.poison) return poison_value;
    const int r = a.v + b.v;
    if (nsw && (r < -128 || r > 127)) return poison_value;
    return {wrap8(r), false};
}

value icmp_sgt(value a, value b) {
    if (a.poison || b.poison) return poison_value;
    return {a.v > b.v, false};
}

value select(value c, value a, value b) {
    if (c.poison) return poison_value;
    return c.v ? a : b;
}

value and1(value a, value b) {
    if (a.poison || b.poison) return poison_value;
    return {a.v & b.v, false};
}

bool refines(value src, value tgt) {
    return src.poison || (!tgt.poison && tgt.v == src.v);
}

std::vector<value> all_i8() {
    std::vector<value> out{poison_value};
    for (int x = -128; x <= 127; ++x) out.push_back({x, false});
    return out;
}

std::vector<value> all_i1() { return {poison_value, {0, false}, {1, false}}; }

using fn2 = std::function<value(value, value)>;

void check(const char* name, const std::vector<value>& xs, const std::vector<value>& ys,
           const fn2& src, const fn2& tgt) {
    for (value x : xs)
        for (value y : ys)
            if (!refines(src(x, y), tgt(x, y))) {
                std::cout << name << ": INVALID at x=" << (x.poison ? "poison" : std::to_string(x.v))
                          << " y=" << (y.poison ? "poison" : std::to_string(y.v)) << '\n';
                return;
            }
    std::cout << name << ": valid\n";
}

int main() {
    const value one{1, false};
    const value fls{0, false};
    check("(x +nsw 1) > x  =>  true", all_i8(), {fls},
          [&](value x, value) { return icmp_sgt(add(x, one, true), x); },
          [&](value, value) { return value{1, false}; });
    check("(x + 1) > x  =>  true", all_i8(), {fls},
          [&](value x, value) { return icmp_sgt(add(x, one, false), x); },
          [&](value, value) { return value{1, false}; });
    check("select c, x, false  =>  and c, x", all_i1(), all_i1(),
          [&](value c, value x) { return select(c, x, fls); },
          [&](value c, value x) { return and1(c, x); });
}
```

The first rewrite is valid: whenever `x +nsw 1` does not overflow, the comparison is true, and when it overflows the source is poison, which the constant `true` refines. The second is invalid at `x = 127`, where the wrapping addition gives `-128` and the source yields false. The third is the famous one. Rewriting `select c, x, false` to `and c, x` looks like a boolean identity, but when `c` is false and `x` is poison the source is `false` (the `select` does not look at `x`) while the target is poison. LLVM performed this transformation for years; Alive2 flagged it, and fixing it required changing how many passes treat `select`.

== Translation Validation with Alive2

Exhaustive enumeration does not scale past small widths, and a compiler has thousands of transformations. *Translation validation* checks individual compilations instead of the compiler: after each transformation, prove that the output refines the input.

Alive (2015) checked LLVM peephole rewrites written in a small DSL, encoding source and target as SMT formulas over bit-vectors for all widths, with poison and undef modelled precisely. Translating several hundred InstCombine rules into Alive found eight wrong rewrites and produced verified replacements. Alive2 (2021) extended this to whole LLVM functions: it runs as a plug-in during optimisation, encodes the function before and after each pass, and asks Z3 whether refinement holds, unrolling loops a bounded number of times. Running it over LLVM's own regression test suite found dozens of previously unknown miscompilation bugs and several deeper semantic inconsistencies, and it is now routinely used by LLVM developers to check proposed transformations before they are merged.

The approach has limits: loops are checked only up to a bound, memory models with pointer provenance are complex to encode, and some queries time out. It is sound for what it checks, though, and it moves the trust from thousands of lines of optimiser code to one semantics and one encoding.

== Random Testing of Compilers

=== Differential Testing with Csmith

Csmith (2011) generates random C programs that are free of undefined behaviour by construction: it tracks pointer validity and integer ranges as it generates and inserts safe wrappers where it cannot prove an operation defined. Each program computes a checksum of its global state. The same program compiled by different compilers, or at different optimisation levels, must print the same checksum; a mismatch means at least one compiler is wrong. Over its first three years Csmith found more than 325 previously unknown bugs in GCC and LLVM, many of them wrong-code bugs. Generating UB-free programs is the essential difficulty: if the program had UB, a checksum mismatch would prove nothing.

=== Equivalence Modulo Inputs

EMI (2014) creates test variants from existing programs instead of generating new ones. Run a program on an input, record which statements were never executed, and then delete or mutate some of those dead statements. For that input, the variants are equivalent to the original, so each must produce the same output when compiled. The variants stress the optimiser differently (dead code changes inlining decisions, register pressure, and loop shapes), and EMI found 147 confirmed bugs in GCC and LLVM in its first year, with its successors finding many hundreds more. Later generators such as YARPGen target specific optimisations, for example by generating loop nests that the vectoriser and loop optimisers will transform.

=== Reduction

A random test that triggers a bug can be thousands of lines long. C-Reduce shrinks it automatically while preserving the bug, using many transformations specific to C (removing declarations, simplifying expressions, inlining) and checking at each step that the program is still free of UB and still shows the mismatch. Reduced test cases of ten or twenty lines are what compiler developers actually receive.

== Verified Compilers

The strongest guarantee is a compiler proved correct once and for all. CompCert, a C compiler written and verified in Coq by Xavier Leroy and colleagues, has a machine-checked proof that each compiled program refines its source under CompCert's formal C semantics, through about twenty passes and a dozen intermediate languages. When Csmith was run against CompCert, it found wrong-code bugs only in parts outside the verified core at the time (such as the front end that was not yet verified), and none in the verified passes, a striking empirical confirmation. CakeML is a verified compiler for a dialect of ML, verified in HOL4 down to machine code and able to bootstrap itself inside the logic.

Verified compilers trade optimisation strength and engineering speed for assurance. CompCert's generated code is typically competitive with GCC at `-O1`, and it is used in avionics and other domains where certification standards reward the removal of compiler verification activities.

== What Programmers Can Do

- Treat sanitizer findings (UBSan, ASan, TSan) as real bugs, even when the code "works".
- Write overflow and bounds checks in forms that do not themselves rely on UB.
- Use compiler flags that define behaviour where performance does not depend on it.
- For critical code, compile and test at the optimisation level and with the compiler version that will ship; a miscompilation is version- and flag-specific.
- When a bug disappears at `-O0` or with a print statement, suspect UB in the program before suspecting the compiler, but reduce and report it if the program is clean.

== Further Reading

Lattner, C. (2011). "What Every C Programmer Should Know About Undefined Behavior." LLVM Project Blog.

Wang, X. et al. (2013). "Towards Optimization-Safe Systems: Analyzing the Impact of Undefined Behavior." SOSP.

Lee, J. et al. (2017). "Taming Undefined Behavior in LLVM." PLDI.

Lopes, N. P. et al. (2015). "Provably Correct Peephole Optimizations with Alive." PLDI.

Lopes, N. P. et al. (2021). "Alive2: Bounded Translation Validation for LLVM." PLDI.

Pnueli, A., Siegel, M. and Singerman, E. (1998). "Translation Validation." TACAS.

Necula, G. C. (2000). "Translation Validation for an Optimizing Compiler." PLDI.

Yang, X. et al. (2011). "Finding and Understanding Bugs in C Compilers." PLDI.

Le, V., Afshari, M. and Su, Z. (2014). "Compiler Validation via Equivalence Modulo Inputs." PLDI.

Regehr, J. et al. (2012). "Test-Case Reduction for C Compiler Bugs." PLDI.

Livinskii, V., Babokin, D. and Regehr, J. (2020). "Random Testing for C and C++ Compilers with YARPGen." OOPSLA.

Leroy, X. (2009). "Formal Verification of a Realistic Compiler." Communications of the ACM.

Kumar, R. et al. (2014). "CakeML: A Verified Implementation of ML." POPL.

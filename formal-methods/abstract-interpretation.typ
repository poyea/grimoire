#import "../template.typ": definition, theorem, proof, example, xref

= Abstract Interpretation <abstract-interpretation>

Rice's theorem says that every nontrivial semantic property of programs is undecidable. Abstract interpretation is the theory of how to compute anyway: replace the exact semantics by an approximation that is computable, and arrange the approximation so that every answer it gives is *sound*. If the analysis says "no division by zero", there is none in any execution. If it cannot prove that, it raises an alarm, which may be false. Cousot and Cousot introduced the framework in 1977, and it now underlies industrial analysers that prove the absence of run-time errors in flight control software, compiler dataflow analyses, and type inference. This chapter covers the lattice-theoretic foundations, Galois connections, the classic numerical domains, widening and narrowing for termination, a working interval analyser in C++, and how industrial tools trade precision against scale.

*See also:* #xref("compilers", "dataflow-analysis", label: "Dataflow Analysis") (monotone frameworks), #xref("programming-languages", "denotational-semantics", label: "Denotational Semantics") (fixpoints), #xref("formal-methods", "model-checking", label: "Model Checking") (CEGAR), #xref("formal-methods", "separation-logic", label: "Separation Logic") (bi-abduction), #xref("embedded-and-realtime", "timing-analysis", label: "Timing Analysis").

== From Semantics to Fixpoints

Model a program as a control-flow graph whose nodes are program points $ell in L$ and whose edges carry statements. A *state* is a valuation of the variables. The *collecting semantics* assigns to each program point the set of states that can reach it in some execution. It is the least solution of a system of equations:

$ X_(ell_0) = "Init", quad X_ell = union.big_((ell', s, ell) in E) [|s|](X_(ell')) $

where $[|s|]$ maps a set of states to the set of successor states under statement $s$. Writing the whole system as one function $F$ on vectors of state sets, the collecting semantics is the least fixpoint $"lfp" F$. Every safety property ("no state at point $ell$ has $y = 0$ before a division by $y$") is a question about this fixpoint. The fixpoint exists, but the sets are infinite and the iteration to reach it does not terminate in general.

#definition(name: "Complete lattice")[
A partially ordered set $(D, subset.eq.sq)$ is a *complete lattice* if every subset $X subset.eq D$ has a least upper bound $union.sq.big X$ and a greatest lower bound $inter.sq.big X$. In particular $D$ has a least element $bot$ and a greatest element $top$.
]

#theorem(name: "Knaster-Tarski")[
If $D$ is a complete lattice and $F : D -> D$ is monotone, then the fixpoints of $F$ form a complete lattice, and $"lfp" F = inter.sq.big {x | F(x) subset.eq.sq x}$.
]

#theorem(name: "Kleene")[
If in addition $F$ preserves least upper bounds of increasing chains, then $"lfp" F = union.sq.big_(n >= 0) F^n (bot)$.
]

Kleene iteration, $bot, F(bot), F(F(bot)), ...$, is the computational reading: start from "nothing reached" and propagate until nothing changes. The powerset of states is a complete lattice under $subset.eq$, so the collecting semantics is well defined; the obstacle is only that the chain may be infinite.

== Galois Connections

An abstract domain $A$ represents sets of concrete states by finite descriptions. The relation between them is fixed by two monotone maps: abstraction $alpha$ and concretisation $gamma$.

#definition(name: "Galois connection")[
Let $(C, subset.eq)$ and $(A, subset.eq.sq)$ be partial orders. A pair of monotone functions $alpha : C -> A$ and $gamma : A -> C$ is a *Galois connection* if for all $c in C$ and $a in A$:
$ alpha(c) subset.eq.sq a quad <==> quad c subset.eq gamma(a). $
]

The condition says that $alpha(c)$ is the most precise abstract element that covers $c$. Two consequences follow directly: $c subset.eq gamma(alpha(c))$ (abstraction loses information but never drops states) and $alpha(gamma(a)) subset.eq.sq a$.

#example(name: "Signs")[
Let $C = cal(P)(ZZ)$ and $A = {bot, -, 0, +, top}$ with $bot subset.eq.sq -, 0, + subset.eq.sq top$. Define $gamma(-) = {n | n < 0}$, $gamma(0) = {0}$, $gamma(+) = {n | n > 0}$, $gamma(top) = ZZ$, $gamma(bot) = emptyset$, and let $alpha(c)$ be the least element whose concretisation contains $c$. Then $alpha({3, 7}) = +$ and $alpha({-1, 5}) = top$.
]

An abstract transformer $f^sharp : A -> A$ is *sound* for a concrete transformer $f$ if $f compose gamma subset.eq gamma compose f^sharp$ pointwise: whatever concrete states the abstract input describes, their images are described by the abstract output. The most precise sound transformer is $alpha compose f compose gamma$, the *best abstract transformer*; real analysers often settle for something cheaper.

#theorem(name: "Fixpoint transfer")[
If $(alpha, gamma)$ is a Galois connection, $F$ and $F^sharp$ are monotone, and $F compose gamma subset.eq gamma compose F^sharp$, then $"lfp" F subset.eq gamma("lfp" F^sharp)$.
]

#proof[
Let $a = "lfp" F^sharp$. Then $F(gamma(a)) subset.eq gamma(F^sharp (a)) = gamma(a)$, so $gamma(a)$ is a post-fixpoint of $F$. By Knaster-Tarski, $"lfp" F$ is below every post-fixpoint, hence $"lfp" F subset.eq gamma(a)$.
]

This is the soundness theorem of abstract interpretation: compute a fixpoint in the abstract domain, and its concretisation over-approximates every reachable state. Any property that holds of the over-approximation holds of the program.

== Numerical Domains

Domains differ in what relations between variables they can express and in what they cost.

#table(
  columns: 4,
  [*Domain*], [*Represents*], [*Cost per operation ($n$ variables)*], [*Can prove*],
  [Signs, parity], [Per variable: sign or parity], [$O(n)$], [Simple sign facts],
  [Constants], [Per variable: a single value or $top$], [$O(n)$], [Constant propagation],
  [Intervals], [Per variable: $[a, b]$], [$O(n)$], [Array bounds when indices are simple],
  [Congruences], [Per variable: $x equiv a mod m$], [$O(n)$], [Alignment, strides],
  [Octagons], [$plus.minus x plus.minus y <= c$], [$O(n^2)$ space, $O(n^3)$ closure], [Bounds relative to other variables, e.g. $i < n$],
  [Convex polyhedra], [Arbitrary linear inequalities], [Exponential worst case], [Linear invariants such as $x + y = 10$],
)

*Non-relational* domains (signs, intervals, congruences) track each variable independently. They are cheap and scale to millions of lines, but cannot prove that `i < n` when neither bound is a constant. *Relational* domains capture such facts. Octagons (Miné) are the usual compromise: relations of the form $plus.minus x plus.minus y <= c$ are enough for most loop-index and buffer-length reasoning, and a cubic closure algorithm keeps them canonical. Polyhedra (Cousot and Halbwachs, 1978) are the most precise linear domain and are used on small, critical code or on variable packs chosen by heuristics.

Domains combine. The *reduced product* of intervals and congruences represents $x in [0, 100] and x equiv 0 mod 4$ and refines each component with the other: an interval endpoint that violates the congruence is tightened to the nearest value that satisfies it.

== Widening and Narrowing

The interval lattice has infinite ascending chains: $[0, 0] subset.eq.sq [0, 1] subset.eq.sq [0, 2] subset.eq.sq dots$. Kleene iteration on a loop like `for (i = 0; ; ++i)` never stabilises. *Widening* forces termination by extrapolation.

#definition(name: "Widening")[
A *widening* is a binary operator $nabla$ on $A$ such that $a subset.eq.sq a nabla b$ and $b subset.eq.sq a nabla b$ for all $a, b$, and for every increasing chain $b_0 subset.eq.sq b_1 subset.eq.sq dots$ the sequence $a_0 = b_0$, $a_(i+1) = a_i nabla b_(i+1)$ is ultimately stationary.
]

The standard interval widening keeps stable bounds and pushes unstable ones to infinity:

$ [a, b] nabla [c, d] = [l, h], quad l = cases(-infinity & "if" c < a, a & "otherwise"), quad h = cases(+infinity & "if" d > b, b & "otherwise") $

Iterating with widening at loop heads reaches a post-fixpoint in finitely many steps, but often a very coarse one. A *narrowing* pass then iterates $F^sharp$ a few more times starting from that post-fixpoint, using a narrowing operator that only refines infinite bounds. Because $F^sharp$ applied to a post-fixpoint gives something smaller that is still sound, the guards of the loop pull the bounds back in. *Widening with thresholds* (jumping to the next constant from a set gathered from the program, such as array sizes, before jumping to infinity) recovers much of the lost precision cheaply.

== An Interval Analyser

The analyser below handles one loop of the form `i = init; while (i < bound) i += step;` and computes the interval for `i` at the loop head and at the exit. It shows every ingredient: a lattice with join, sound transfer functions for assignment and guards, widening at the loop head, and a narrowing pass.

```cpp
#include <algorithm>
#include <cstdint>
#include <limits>
#include <ostream>

using i64 = std::int64_t;
constexpr i64 neg_inf = std::numeric_limits<i64>::min();
constexpr i64 pos_inf = std::numeric_limits<i64>::max();

i64 sat_add(i64 a, i64 b) {
    if (a == neg_inf || b == neg_inf) return neg_inf;
    if (a == pos_inf || b == pos_inf) return pos_inf;
    i64 r;
    if (__builtin_add_overflow(a, b, &r)) return b > 0 ? pos_inf : neg_inf;
    return r;
}

struct interval {
    i64 lo = pos_inf;   // empty interval (bottom) when lo > hi
    i64 hi = neg_inf;

    static interval bottom() { return {}; }
    static interval of(i64 a, i64 b) { return {a, b}; }
    bool is_bottom() const { return lo > hi; }

    bool leq(const interval& o) const {
        return is_bottom() || (!o.is_bottom() && o.lo <= lo && hi <= o.hi);
    }
    interval join(const interval& o) const {
        if (is_bottom()) return o;
        if (o.is_bottom()) return *this;
        return {std::min(lo, o.lo), std::max(hi, o.hi)};
    }
    interval meet(const interval& o) const {
        interval r{std::max(lo, o.lo), std::min(hi, o.hi)};
        return r.is_bottom() ? bottom() : r;
    }
    interval widen(const interval& next) const {
        if (is_bottom()) return next;
        if (next.is_bottom()) return *this;
        return {next.lo < lo ? neg_inf : lo, next.hi > hi ? pos_inf : hi};
    }
    interval narrow(const interval& next) const {
        if (is_bottom() || next.is_bottom()) return bottom();
        return {lo == neg_inf ? next.lo : lo, hi == pos_inf ? next.hi : hi};
    }
    interval add(i64 c) const {
        if (is_bottom()) return *this;
        return {sat_add(lo, c), sat_add(hi, c)};
    }
};

// Guard "x < k" restricts x to (-inf, k-1]; its negation to [k, +inf).
interval assume_lt(const interval& x, i64 k) { return x.meet({neg_inf, sat_add(k, -1)}); }
interval assume_ge(const interval& x, i64 k) { return x.meet({k, pos_inf}); }

struct loop_result {
    interval head;
    interval exit;
};

loop_result analyse_loop(i64 init, i64 bound, i64 step) {
    const interval entry = interval::of(init, init);
    auto body_then_back = [&](const interval& head) {
        return assume_lt(head, bound).add(step);
    };

    interval head = entry;
    for (;;) {
        const interval next = entry.join(body_then_back(head));
        if (next.leq(head)) break;
        head = head.widen(next);
    }
    for (int i = 0; i < 2; ++i)
        head = head.narrow(entry.join(body_then_back(head)));

    return {head, assume_ge(head, bound)};
}
```

For `init = 0`, `bound = 100`, `step = 1`, the ascending phase goes $[0, 0] -> [0, +infinity]$ after one widening. Narrowing then computes $[0, 0] union.sq ([0, +infinity] inter.sq (-infinity, 99]) + 1 = [0, 100]$, so the head interval is $[0, 100]$ and the exit interval is $[100, 100]$, which is exact. With `step = 3` the analysis returns head $[0, 102]$ and exit $[100, 102]$; the true exit value is 102, and only a congruence domain combined with the interval ($i equiv 0 mod 3$) would recover it. The checks against `pos_inf` and `neg_inf` matter: an analyser of C programs must reason about machine integers, and its own arithmetic must not overflow while doing so.

== Precision, False Alarms, and Soundiness

A sound analyser always over-approximates, so it reports *alarms* that may be false. The engineering of a practical analyser is the engineering of false-alarm rates:

- *Domain choice.* A relational domain proves facts an interval cannot, at higher cost. Analysers apply expensive domains only to small *packs* of variables that occur together in tests and assignments.
- *Trace partitioning and context sensitivity.* Joining states at every merge point loses correlations ("if `p` is null then `n == 0`"). Keeping separate abstract states per branch or per calling context restores them, at a multiplicative cost.
- *Memory abstraction.* Pointers, arrays, and heap structures need their own abstractions (points-to sets, array segmentation, shape analysis), and these usually dominate precision on real code.

Many tools are deliberately unsound in documented ways, for example by ignoring reflection, assuming no integer overflow, or analysing only some entry points. The term *soundiness* names this practice: sound for a stated subset of language features, so that the remaining alarms are few enough for developers to act on.

== Industrial Use

#table(
  columns: 3,
  [*Tool*], [*Approach*], [*Notable use*],
  [Astrée], [Sound analysis of C with intervals, octagons, filters, trace partitioning, tuned for synchronous control code], [Proved absence of run-time errors in Airbus fly-by-wire software with zero false alarms],
  [Frama-C EVA], [Sound value analysis for C, combined with deductive verification plug-ins], [Nuclear, avionics, and security-critical C],
  [IKOS], [NASA's abstract-interpretation library and analyser over LLVM IR], [Flight software],
  [Infer], [Compositional analysis with separation logic and bi-abduction, under-approximating to keep false positives low], [Code review bots at Meta and other companies],
  [Polyspace Code Prover], [Commercial sound analyser for C and C++], [Automotive and medical certification],
  [MOPSA], [Modular research platform for multiple languages], [C and Python analysis research],
)

Astrée's result is the landmark: on a program of hundreds of thousands of lines of generated control code, specialised domains (including one for digital filters) brought the alarm count to zero, which means a proof of the absence of whole classes of run-time error. The lesson is that sound analysis scales when domains are designed for the code family being analysed.

Meta's Infer took the opposite trade: it accepts missing some bugs in exchange for reporting only issues that are likely to be real, and it analyses each procedure once with a summary so that a code change can be analysed in minutes as part of code review. Both are applications of the same framework with different choices of approximation direction and domain.

== Relation to Other Techniques

Compiler dataflow analyses are abstract interpretations over finite-height lattices, where widening is unnecessary. Type inference can be viewed as abstract interpretation over a domain of types. Model checking explores a finite abstraction exactly, and CEGAR refines the abstraction from spurious counterexamples, while abstract interpretation fixes the abstraction in advance and accepts false alarms. Deductive verification asks the user for loop invariants; abstract interpretation computes invariants automatically but only within the shapes its domains can express. Combinations are common: Frama-C uses inferred value ranges to discharge proof obligations, and verifiers use abstract interpretation to suggest invariants.

== Further Reading

Cousot, P. and Cousot, R. (1977). "Abstract Interpretation: A Unified Lattice Model for Static Analysis of Programs by Construction or Approximation of Fixpoints." POPL.

Cousot, P. and Halbwachs, N. (1978). "Automatic Discovery of Linear Restraints Among Variables of a Program." POPL.

Cousot, P. and Cousot, R. (1992). "Comparing the Galois Connection and Widening/Narrowing Approaches to Abstract Interpretation." PLILP.

Miné, A. (2006). "The Octagon Abstract Domain." Higher-Order and Symbolic Computation.

Blanchet, B. et al. (2003). "A Static Analyzer for Large Safety-Critical Software." PLDI.

Calcagno, C. et al. (2015). "Moving Fast with Software Verification." NASA Formal Methods Symposium.

Livshits, B. et al. (2015). "In Defense of Soundiness: A Manifesto." Communications of the ACM.

Rival, X. and Yi, K. (2020). _Introduction to Static Analysis: An Abstract Interpretation Perspective_. MIT Press.

Cousot, P. (2021). _Principles of Abstract Interpretation_. MIT Press.

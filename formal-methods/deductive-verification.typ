#import "../template.typ": definition, theorem, xref

= Auto-Active Deductive Verification <deductive-verification>

Between push-button analysers that prove only shallow properties and interactive proof assistants that can prove anything given enough human effort lies *auto-active* verification. The programmer writes ordinary code annotated with specifications (preconditions, postconditions, loop invariants, termination measures) and occasional proof hints; a tool turns code and annotations into verification conditions and discharges them with an SMT solver, automatically and in seconds. When a proof fails, the programmer adds an annotation rather than writing a proof script. Dafny, Why3, Verus, SPARK, Frama-C's WP plug-in, and Viper-based tools all work this way. This chapter covers the pipeline from annotated source to SMT queries, a weakest-precondition VC generator in C++, loops, procedures and framing, termination, ghost code and lemmas, quantifier triggers and proof brittleness, and the industrial record.

*See also:* #xref("programming-languages", "axiomatic-semantics", label: "Axiomatic Semantics") (Hoare logic and weakest preconditions), #xref("formal-methods", "sat-and-smt", label: "SAT and SMT"), #xref("formal-methods", "separation-logic", label: "Separation Logic"), #xref("formal-methods", "theorem-proving", label: "Theorem Proving"), #xref("formal-methods", "abstract-interpretation", label: "Abstract Interpretation").

== The Spectrum of Verification

#table(
  columns: 4,
  [*Style*], [*User writes*], [*Tool does*], [*Examples*],
  [Automatic], [Nothing, or configuration], [Infers invariants in fixed domains; may raise false alarms], [Astrée, Infer, CBMC],
  [Auto-active], [Specifications, invariants, hints], [Generates VCs, discharges them with SMT], [Dafny, Why3, Verus, SPARK, Frama-C WP, VeriFast, Prusti],
  [Interactive], [Proof scripts or proof terms], [Checks each step; automation helps locally], [Coq/Rocq, Lean, Isabelle/HOL],
)

The auto-active point on the spectrum trades expressiveness of the proof language for automation. The specification language is usually first-order logic with arithmetic, arrays, algebraic data types, and user-defined functions. That is undecidable in general (quantifiers plus arithmetic), so the tool relies on solver heuristics and on the user to supply the creative steps: invariants, intermediate assertions, and lemmas.

== The Pipeline

Most auto-active tools share an architecture:

```text
annotated source (Dafny, Rust+Verus, Ada/SPARK, C+ACSL)
   │  type check, resolve, desugar
   ▼
intermediate verification language (Boogie, WhyML, Viper)
   │  loops → invariants + havoc; calls → contracts; heap → maps
   ▼
passive, loop-free program per procedure
   │  weakest preconditions / efficient VC generation
   ▼
verification condition (first-order formula)
   │  SMT-LIB encoding, axioms for the prelude
   ▼
SMT solver (Z3, cvc5, Alt-Ergo): unsat of ¬VC means verified
```

Splitting off an intermediate verification language lets many front ends share one carefully engineered VC generator and encoding. Dafny, Spec\#, VCC, and Chalice all targeted Boogie; Frama-C and SPARK use Why3; Prusti, Gobra, and Nagini target Viper.

== Generating Verification Conditions

After desugaring, each procedure body becomes a program in a tiny language of guarded commands:

$ C ::= "assert" P | "assume" P | x := e | "havoc" x | C ; C | C square C $

where $square$ is nondeterministic choice. The weakest precondition of a command with respect to a postcondition $Q$ is defined by structural recursion:

#table(
  columns: 2,
  [*Command*], [*$"wp"(C, Q)$*],
  [$"assert" P$], [$P and Q$],
  [$"assume" P$], [$P => Q$],
  [$x := e$], [$Q[e \/ x]$],
  [$"havoc" x$], [$forall x. Q$],
  [$C_1 ; C_2$], [$"wp"(C_1, "wp"(C_2, Q))$],
  [$C_1 square C_2$], [$"wp"(C_1, Q) and "wp"(C_2, Q)$],
)

The procedure is correct if $"wp"("body", "true")$ is valid, so the tool asks the solver whether its negation is unsatisfiable. A conditional `if b then S else T` desugars to $("assume" b; S) square ("assume" not b; T)$.

```cpp
#include <map>
#include <memory>
#include <string>

struct formula;
using fptr = std::shared_ptr<const formula>;

struct formula {
    enum class kind { var, num, op, not_ } k;
    std::string name;        // variable name, or operator: + - * < <= = and or =>
    long value = 0;
    fptr lhs, rhs;
};

fptr var(std::string n) { return std::make_shared<formula>(formula{formula::kind::var, std::move(n)}); }
fptr num(long v) { return std::make_shared<formula>(formula{formula::kind::num, {}, v}); }
fptr op(std::string o, fptr a, fptr b) {
    return std::make_shared<formula>(formula{formula::kind::op, std::move(o), 0, std::move(a), std::move(b)});
}
fptr not_(fptr a) { return std::make_shared<formula>(formula{formula::kind::not_, {}, 0, std::move(a)}); }

fptr subst(const fptr& f, const std::string& x, const fptr& e) {
    switch (f->k) {
    case formula::kind::var:  return f->name == x ? e : f;
    case formula::kind::num:  return f;
    case formula::kind::not_: return not_(subst(f->lhs, x, e));
    case formula::kind::op:   return op(f->name, subst(f->lhs, x, e), subst(f->rhs, x, e));
    }
    return f;
}

struct command;
using cptr = std::shared_ptr<const command>;

struct command {
    enum class kind { assert_, assume, assign, havoc, seq, choice } k;
    std::string var;
    fptr f;
    cptr first, second;
};

class vc_generator {
    int fresh_ = 0;

public:
    fptr wp(const cptr& c, const fptr& q) {
        switch (c->k) {
        case command::kind::assert_: return op("and", c->f, q);
        case command::kind::assume:  return op("=>", c->f, q);
        case command::kind::assign:  return subst(q, c->var, c->f);
        case command::kind::havoc:   // a fresh symbol is implicitly universal in a validity query
            return subst(q, c->var, var(c->var + "!" + std::to_string(fresh_++)));
        case command::kind::seq:     return wp(c->first, wp(c->second, q));
        case command::kind::choice:  return op("and", wp(c->first, q), wp(c->second, q));
        }
        return q;
    }
};

std::string to_smtlib(const fptr& f) {
    switch (f->k) {
    case formula::kind::var:  return f->name;
    case formula::kind::num:  return f->value < 0 ? "(- " + std::to_string(-f->value) + ")"
                                                  : std::to_string(f->value);
    case formula::kind::not_: return "(not " + to_smtlib(f->lhs) + ")";
    case formula::kind::op:   return "(" + f->name + " " + to_smtlib(f->lhs) + " " + to_smtlib(f->rhs) + ")";
    }
    return {};
}
```

Emitting `(assert (not VC))` with a declaration for each free symbol, followed by `(check-sat)`, gives the solver query. `unsat` means the procedure is verified; `sat` gives a model that is a counterexample, which tools map back to source as the values of variables at the failing assertion.

This naive generator duplicates $Q$ at every choice, so a procedure with $n$ sequential `if` statements yields a VC of size $O(2^n)$. Production tools first convert the program to *passive form*, a dynamic single-assignment variant where each assignment `x := e` becomes `assume x1 == e` on a fresh incarnation, and then use the Flanagan-Saxe or Barnett-Leino construction, which names each block's VC once and keeps the formula linear in program size.

== Loops and Invariants

A loop cannot be unfolded, so it is cut. Given an invariant $I$ and the set $M$ of variables the loop body modifies, `while b invariant I { S }` becomes

$ "assert" I; space "havoc" M; space "assume" I; space ( ("assume" b; S; "assert" I; "assume" "false") square "assume" not b ) $

The first `assert` checks that the invariant holds on entry. The `havoc` forgets everything about the modified variables except the invariant, which models an arbitrary iteration. The first branch checks that one iteration preserves $I$ and then cuts the path. The second branch continues after the loop knowing $I and not b$. Everything not in $M$ keeps its value across the loop, which is why tools must compute $M$ (or, for heap locations, a frame) soundly.

Writing invariants is the main human effort. A Dafny binary search shows the typical shape: bounds facts, plus a characterisation of what the already-excluded part of the input satisfies.

```text
method BinarySearch(a: array<int>, key: int) returns (index: int)
  requires forall i, j :: 0 <= i < j < a.Length ==> a[i] <= a[j]
  ensures 0 <= index ==> index < a.Length && a[index] == key
  ensures index < 0 ==> forall k :: 0 <= k < a.Length ==> a[k] != key
{
  var lo, hi := 0, a.Length;
  while lo < hi
    invariant 0 <= lo <= hi <= a.Length
    invariant forall k :: 0 <= k < lo ==> a[k] != key
    invariant forall k :: hi <= k < a.Length ==> a[k] != key
    decreases hi - lo
  {
    var mid := lo + (hi - lo) / 2;
    if a[mid] < key {
      lo := mid + 1;
    } else if a[mid] > key {
      hi := mid;
    } else {
      return mid;
    }
  }
  return -1;
}
```

The second invariant is preserved when `lo` moves past `mid` only because the array is sorted: every `a[k]` with `k <= mid` is at most `a[mid] < key`. The solver finds this by instantiating the sortedness precondition with `j := mid`. The midpoint is written `lo + (hi - lo) / 2` out of habit; Dafny's `int` is unbounded, but the same code in Verus or SPARK over machine integers would need the overflow-safe form for the arithmetic VCs to pass.

== Procedures, Contracts, and Frames

Verification is *modular*: each procedure is verified once against its own contract, and each call site uses only the callee's contract, never its body. A call `r := f(x)` desugars to `assert pre_f(x); havoc r, modified_f; assume post_f(x, r)`. This keeps verification time linear in program size and lets procedures change without re-verifying callers, as long as contracts hold.

The price is the *frame problem*: after the call, what is known about state the callee did not mention? Without a frame specification, `havoc` must forget the whole heap. Tools differ in how frames are written:

#table(
  columns: 3,
  [*Approach*], [*How frames are written*], [*Tools*],
  [Modifies clauses on locations], [List of variables or heap locations a procedure may change], [Boogie, SPARK (data flow contracts), Frama-C `assigns`],
  [Dynamic frames], [Ghost sets of objects in `reads` and `modifies` clauses, updated by the program], [Dafny],
  [Permissions / separation logic], [Access permissions passed in and out with preconditions and postconditions], [Viper, VeriFast, Prusti (from Rust types)],
  [Ownership types], [The type system rules out interference; frames follow from ownership], [Verus, Creusot, Prusti, Aeneas],
)

Rust-based verifiers benefit most from the last row. The borrow checker already guarantees that a `&mut` reference is the only access path to its target, so the verifier can treat most memory as plain values and avoid heap reasoning entirely. This is a large part of why Verus proofs of systems code are shorter than similar proofs in earlier tools.

== Termination

Partial correctness says "if it returns, the result is right". Total correctness also requires that it returns. Auto-active tools prove termination with a *variant*: an expression over a well-founded order that decreases on every loop iteration and every recursive call. In Dafny's `decreases hi - lo`, the tool generates the VCs that `hi - lo` is bounded below by zero while the loop runs and strictly smaller after each iteration. Tuples order lexicographically, so `decreases m, n` handles nested recursion such as Ackermann's function. Many tools guess simple variants automatically and ask for one only when the guess fails.

== Ghost Code and Lemmas

*Ghost* variables, functions, and statements exist only for verification and are erased before compilation. They let a specification talk about abstract state (the set of elements in a hash table, the sequence of messages sent so far) that the implementation does not store. The tool checks that ghost code never influences non-ghost results.

A *lemma* is a ghost method whose postcondition is the theorem and whose body is the proof. Calling the lemma at a program point makes its conclusion available to the solver there. Induction is written as recursion: the lemma calls itself on a smaller argument, and the termination check guarantees the induction is well founded.

```text
function Sum(n: nat): nat { if n == 0 then 0 else n + Sum(n - 1) }

lemma SumFormula(n: nat)
  ensures 2 * Sum(n) == n * (n + 1)
{
  if n > 0 {
    SumFormula(n - 1);   // induction hypothesis for n - 1
  }
}
```

Here the solver fills in the algebra once given the hypothesis. Lemmas are the main tool for nonlinear arithmetic, which SMT solvers handle poorly, and for any fact that requires induction, which SMT solvers do not attempt on their own.

== Quantifiers and Triggers

Specifications are full of universal quantifiers (sortedness, "no element equals key", class invariants), and solvers handle them by *E-matching*: each quantifier has one or more *triggers*, patterns of ground terms, and the solver instantiates the quantifier whenever a term matching the trigger appears in its current set of facts. For `forall i, j :: 0 <= i < j < a.Length ==> a[i] <= a[j]` the trigger is the pair `{a[i], a[j]}`, so any two array reads in the proof cause an instantiation.

Triggers make quantifier reasoning efficient and also unpredictable:

- *Too restrictive.* If no matching term occurs, the quantifier is never used, and a true VC fails. The fix is to mention the right term in an `assert`, which adds it to the solver's facts.
- *Matching loops.* An axiom like `forall x {f(x)} :: f(x) == f(x + 1)` creates the term `f(x + 1)`, which matches the trigger again, creating `f(x + 2)`, and so on until a timeout.
- *Proof brittleness.* Solver search depends on term order, random seeds, and the exact set of facts. A proof that passes can time out after an unrelated change, a solver upgrade, or a renaming. Studies of large Dafny and Verus code bases found that a noticeable fraction of queries are unstable under such perturbations.

Mature projects treat brittleness as an engineering problem: they keep each VC small by splitting procedures and hiding function definitions behind `opaque` or `reveal`, limit the facts in scope, and test proofs under seed variation in CI. Verus goes further by giving the solver a deliberately small, curated context by default.

== Industrial Record

- *AWS* rewrote its IAM authorisation engine in Dafny and proved it against a specification, ships the AWS Encryption SDK from Dafny sources, and first verified the Cedar policy language in Dafny (Cedar's proofs later moved to Lean). Dafny compiles to the languages the services ship in.
- *IronFleet* and *Ironclad* (Microsoft Research) verified distributed systems and secure remote services end to end in Dafny, from protocol specifications down to implementation code.
- *SPARK* (Ada) is used in avionics, rail, and security; NVIDIA adopted it for security-critical firmware, and the Tokeneer project became a public benchmark.
- *Verus* is used to verify Rust systems code, including memory allocators, page tables, and storage components.
- *Frama-C WP* verifies C components in nuclear and aerospace software, often combined with Frama-C's value analysis.

The common lesson from these projects is that the specification is the hard part. Verification guarantees the code meets its specification; it cannot tell whether the specification says what was intended. Reviewing specifications, testing them against examples, and keeping them simple and abstract are as important as the proofs.

== Further Reading

Dijkstra, E. W. (1975). "Guarded Commands, Nondeterminacy and Formal Derivation of Programs." Communications of the ACM.

Flanagan, C. and Saxe, J. B. (2001). "Avoiding Exponential Explosion: Generating Compact Verification Conditions." POPL.

Barnett, M. and Leino, K. R. M. (2005). "Weakest-Precondition of Unstructured Programs." PASTE.

Leino, K. R. M. (2010). "Dafny: An Automatic Program Verifier for Functional Correctness." LPAR.

Filliâtre, J.-C. and Paskevich, A. (2013). "Why3: Where Programs Meet Provers." ESOP.

Müller, P., Schwerhoff, M. and Summers, A. J. (2016). "Viper: A Verification Infrastructure for Permission-Based Reasoning." VMCAI.

Detlefs, D., Nelson, G. and Saxe, J. B. (2005). "Simplify: A Theorem Prover for Program Checking." Journal of the ACM.

Hawblitzel, C. et al. (2015). "IronFleet: Proving Practical Distributed Systems Correct." SOSP.

Lattuada, A. et al. (2023). "Verus: Verifying Rust Programs using Linear Ghost Types." OOPSLA.

Zhou, Y. et al. (2023). "Mariposa: Measuring SMT Instability in Automated Program Verification." FMCAD.

Leino, K. R. M. (2023). _Program Proofs_. MIT Press.

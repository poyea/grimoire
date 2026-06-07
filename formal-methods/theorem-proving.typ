= Theorem Proving

Interactive theorem provers let humans and machines collaborate to construct machine-checked proofs of mathematical and software correctness claims that lie beyond the reach of fully automatic tools. Three systems dominate practice — Coq, Lean 4, and Isabelle/HOL — each grounded in a rich type-theoretic or set-theoretic foundation that doubles as both specification language and proof language. The Curry-Howard correspondence is the conceptual bridge: proofs are programs, propositions are types.

*See also:* _Propositional and First-Order Logic_, _SAT and SMT_

== The Curry-Howard Correspondence

The *Curry-Howard isomorphism* identifies:

#table(
  columns: (auto, auto),
  [*Logic*], [*Type theory*],
  [Proposition $A$], [Type $A$],
  [Proof of $A$], [Term of type $A$],
  [$A and B$], [Product type $A times B$],
  [$A or B$], [Sum type $A + B$],
  [$A -> B$], [Function type $A -> B$],
  [$bot$ (false)], [Empty type],
  [$forall x : A. B(x)$], [Dependent product $Pi (x : A). B(x)$],
  [$exists x : A. B(x)$], [Dependent sum $Sigma (x : A). B(x)$],
)

Under this correspondence, *proof checking* is *type checking* and *proof search* is *program synthesis*. The computation content of a proof is its *extract* — a functional program that witnesses the constructive content of the theorem. This is the basis of Coq's extraction mechanism.

== Dependent Types

*Dependent types* allow types to depend on values, expressing fine-grained correctness properties directly in types. A *dependent function type* $Pi (n : "Nat"). "Vec" A n$ is the type of functions mapping each natural number $n$ to a vector of length $n$ — the length is tracked in the type, ruling out out-of-bounds access statically.

Key forms:
- *$Pi$-types* ($forall$ in Coq/Lean): functions whose return type depends on the argument.
- *$Sigma$-types* ($exists$ in Coq, `⟨_, _⟩` in Lean): pairs where the type of the second component depends on the first — encode existential statements.
- *Inductive families* (`Inductive` in Coq, `inductive` in Lean): generalized algebraic data types indexed by values, e.g. `Fin n` (type with exactly $n$ elements) or `Eq : A -> A -> Prop` (propositional equality).

The *Calculus of Inductive Constructions* (CIC), the foundation of Coq, layers universes ($"Prop"$, $"Set"$, $"Type"_i$), $Pi$-types, and inductive definitions, yielding a system both expressive enough for mathematics and strongly normalizing (every term reduces to a unique normal form — ensuring type checking terminates).

== Coq: Tactics and Ltac

In *Coq*, the user proves goals interactively using *tactics*. A *goal* is a pair $Gamma tack.r P$ — context $Gamma$ of hypotheses and a goal proposition $P$. Tactics transform goals:

#table(
  columns: (auto, auto),
  [*Tactic*], [*Effect*],
  [`intro h`], [move leading $forall$ or $->$ into context as hypothesis `h`],
  [`apply lemma`], [unify goal with conclusion of `lemma`; generate subgoals for premises],
  [`exact e`], [close goal with term `e` of the right type],
  [`destruct h`], [case-split on inductive hypothesis `h`],
  [`induction n`], [generate base case and step subgoals],
  [`rewrite h`], [replace lhs with rhs of equation `h` in goal],
  [`simpl`], [reduce goal by computation],
  [`omega`], [decide linear arithmetic goal automatically],
  [`auto` / `tauto`], [propositional / first-order automation],
)

*Ltac* is Coq's tactic meta-language — a dynamically typed language for writing proof automation. A typical Ltac macro:

```coq
Ltac crush :=
  simpl in *; intros; try omega;
  repeat match goal with
    | [H : _ /\ _ |- _] => destruct H
    | [H : _ \/ _ |- _] => destruct H
    | [H : False   |- _] => contradiction
  end; auto.
```

Ltac's power comes at a cost: proofs are fragile under small changes to definitions. Coq's newer *Ltac2* provides typed tactics; the *Equations* plugin handles dependent pattern matching uniformly.

=== Code Extraction

Coq can *extract* a certified program to OCaml, Haskell, or Scheme. The extraction erases proof objects (inhabitants of `Prop`) and retains computational content (inhabitants of `Set` or `Type`). For example, a proof of $forall n m : "Nat". exists k. k = n + m$ extracts to the addition function. The *CompCert* compiler (below) exploits this to produce verified C compilers.

== Lean 4

*Lean 4* (de Moura, Ullrich, 2021) is a proof assistant and general-purpose programming language built on the *Calculus of Constructions with recursive functions*. Its key advances over Coq:

- *Unified language:* Lean 4 programs and proofs are written in the same language; the kernel is a small, auditable type checker.
- *Metaprogramming:* tactics, elaborators, and syntax extensions are written in Lean 4 itself (not a separate metalanguage), enabling deep reflection and macro-based proof automation.
- *Mathlib4:* a community library containing $> 100,000$ theorems spanning undergraduate and graduate mathematics — number theory, algebra, analysis, topology, category theory.
- *`decide` and `native_decide`*: reflect decidable propositions to Boolean functions and evaluate them, allowing proofs by computation without manual case analysis.

=== A Small Lean 4 Proof

```lean
-- Prove that the sum of the first n natural numbers equals n*(n+1)/2
-- represented as: 2 * (∑ i in Finset.range (n+1), i) = n * (n + 1)

theorem sum_range (n : ℕ) :
    2 * (Finset.range (n + 1)).sum id = n * (n + 1) := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [Finset.sum_range_succ]
    simp [Nat.mul_add, Nat.add_mul]
    linarith
```

The proof proceeds by induction: the base case is discharged by `simp` (simplification); the inductive step rewrites with the sum recurrence, expands with ring lemmas, and closes with `linarith` (linear arithmetic). All steps are type-checked by Lean's kernel.

=== Lean 4 Metaprogramming

Lean 4's *macro* system allows new syntax, and its *`Tactic`* monad allows defining new tactics in Lean 4 itself. The elaborator is re-entrant: a term-level `by ...` block drops into tactic mode; `show`, `exact`, `apply` work uniformly. The `Syntax` and `Expr` types (Lean's AST and core IR) are first-class, enabling proof-by-reflection strategies that were painful in Coq.

== Isabelle/HOL

*Isabelle/HOL* takes a different foundation: *Higher-Order Logic* (Church's simple type theory) rather than dependent types. HOL is less expressive than CIC but simpler and decidable in fragments, enabling more powerful automation.

Isabelle's distinguishing features:

- *Sledgehammer:* calls external ATPs (Vampire, E, SPASS) and SMT solvers (Z3, CVC5), translates their proofs back to Isabelle, and discharges goals automatically — the most powerful automation in any proof assistant.
- *`simp` / `auto` / `blast`:* term rewriting, combined simplfication+rule application, and tableaux-based propositional reasoning. Together they close the vast majority of routine subgoals.
- *Locales:* algebraic structuring — a *locale* parameterizes a theory over a type with operations satisfying axioms, enabling abstract algebra without universe issues.
- *AFP (Archive of Formal Proofs):* $> 800$ contributed theory entries, including verified algorithms, cryptographic protocols, and mathematical theories.

```isabelle
(* Isabelle/HOL: reverse of reverse is identity *)
theorem rev_rev [simp]: "rev (rev xs) = xs"
  by (induct xs) simp_all
```

`simp_all` closes both the base case (`rev (rev []) = []`) and the inductive step by invoking the `simp` set, which includes the distributivity lemma for `rev` over append.

== Proof by Reflection

*Proof by reflection* (Boutin 1997) is a technique that delegates proof search to a *verified decision procedure* running inside the proof assistant. Steps:

+ Implement a decision function $f$ (e.g., a ring normalizer, a Presburger solver) as a term in the assistant's language.
+ Prove (once) that $f$ is correct: $f(phi) = "true" -> phi$.
+ To prove a specific instance $phi$, evaluate $f(phi)$ by computation inside the kernel; if it returns `true`, apply the correctness theorem.

The key advantage over external oracles: the decision procedure itself is machine-checked; no trust is placed in external tools. Lean's `decide` and `native_decide` are the canonical instances. Coq's `ring` and `field` tactics use reflection. The *Cantor-Bernstein theorem* and arithmetic identities over $10,000$ characters long have been proved this way.

== Certified Compilers: CompCert

*CompCert* (Leroy, 2009) is a formally verified compiler for a large subset of C, producing PowerPC, ARM, x86, and RISC-V code. It is developed and verified entirely in Coq.

The proof structure: for each compilation pass (C to Clight, Clight to Cminor, ..., RTL to assembly) a *semantic preservation* theorem is proved:

$ "semantics"("source") tilde.eq "semantics"("target") $

where $tilde.eq$ is a *forward simulation* (or backward simulation when the target has more steps). The composition of all pass simulations yields the top-level theorem: if a CompCert-compiled program exhibits a defined behavior $b$, then the source C program also exhibits $b$ — bugs cannot be introduced by compilation.

CompCert has been used in safety-critical avionics (Airbus), and no compiler bug has ever been found in the verified passes by differential testing.

== seL4: Verified OS Kernel

*seL4* (Klein et al., 2009) is a formally verified microkernel whose functional correctness proof — the largest mechanized proof at the time of its publication — was carried out in Isabelle/HOL.

The proof chain:
+ *Abstract specification* (in Isabelle): a purely functional model of the kernel's API.
+ *Executable specification* (Haskell prototype): refined from the abstract, proved equivalent.
+ *C implementation* (10,000 LOC): proved by manual correspondence to the executable spec using a C semantics embedding (the l4v framework).
+ *Binary verification* (ARMv7): proved the compiled binary matches the C semantics using decompilation into Isabelle.

The result: no undefined behavior, no security policy violations, given the hardware model. The proof handles real-time properties, interrupt handling, and capability-based access control.

== The Proof Landscape

#table(
  columns: (auto, auto, auto),
  [*System*], [*Foundation*], [*Strengths*],
  [Coq 8.x], [CIC + universes], [extraction, tactics, CompCert ecosystem],
  [Lean 4], [CoC + recursion], [metaprogramming, Mathlib, unified language],
  [Isabelle/HOL], [HOL + Isar], [Sledgehammer, AFP, locales],
  [Agda], [MLTT + universe polymorphism], [cubical, HoTT, dependent pattern matching],
  [F\* (MSR)], [indexed effect system], [cryptographic code, HACL\*],
  [HOL4], [HOL88 lineage], [hardware verification, CakeML],
  [PVS], [predicate subtyping], [NASA aerospace, model checking integration],
)

== Further Reading

Chlipala, A. (2022). _Certified Programming with Dependent Types._ MIT Press (free online).

Pierce, B. et al. (2023). _Software Foundations._ (free online, Coq-based).

Avigad, J., de Moura, L. et al. _Theorem Proving in Lean 4._ (free online).

Leroy, X. (2009). "Formal Verification of a Realistic Compiler." _CACM_ 52(7).

Klein, G. et al. (2009). "seL4: Formal Verification of an OS Kernel." _SOSP_.

Nipkow, T., Paulson, L., Wenzel, M. (2002). _Isabelle/HOL: A Proof Assistant for Higher-Order Logic._ Springer.

Howard, W. (1980). "The Formulae-as-Types Notion of Construction." In _To H. B. Curry: Essays on Combinatory Logic._

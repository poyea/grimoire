#import "../template.typ": xref

= Dependent Types: Proof Engineering and Advanced Topics

== Tactics and Proof Engineering

In Coq/Lean, a proof is produced by *tactics*, a script of commands that incrementally build a proof term. Common tactics:

#table(
  columns: (auto, auto),
  [*Tactic*], [*Effect*],
  [`intro x`], [introduce a hypothesis or universal variable],
  [`apply H`], [apply a known lemma backwards],
  [`exact e`], [give the proof term explicitly],
  [`induction n`], [proof by induction on $n$],
  [`destruct e`], [case-analyse $e$],
  [`rewrite H`], [rewrite using equality hypothesis $H$],
  [`reflexivity`], [proof of $a = a$ by `refl`],
  [`auto` / `tauto`], [automatic search],
  [`omega` / `lia`], [linear arithmetic decision procedure],
)

The *elaboration* of a tactic script into a proof term is the work of the *tactic engine*; the *kernel* re-checks the resulting term independently. This division (large untrusted elaboration, small trusted kernel) is the *de Bruijn criterion* and is the architectural reason proof assistants can be trusted at all.

== Definitional Equality, $eta$, and Surprises

A standard surprise: in pure ITT, $f$ and $lambda x . f space x$ are not definitionally equal unless $eta$ is part of conversion. Modern Coq enables $eta$ for functions; for inductives, $eta$ for $Sigma$ is enabled (surjective pairing); for general inductives, $eta$ would be unsound in general.

Another: in CIC, `match` on a $"Prop"$-typed value (a proof of equality, say) is restricted: the *singleton elimination* rule says you can only eliminate into $"Prop"$, not into $"Type"$, except for very specific cases (`False`, `And`, `Eq` on decidable types). This prevents leaking proof structure into computational types, preserving proof irrelevance.

== Refinement Types and Subset Types

A *refinement* type $\{ x : A | P(x) \}$ (or in Coq, `{ x : A & P x }` using $Sigma$) is a *subset type*: values of $A$ satisfying $P$.

```coq
Definition divide (n m : nat) : { q : nat | n = q * m } -> ...
```

The function takes a witness that division is exact. Refinement types are heavily used in F\* with SMT discharge: most refinements become first-order verification conditions delegated to Z3.

== Equality up to Computation

In CIC, decidable equality is *internalised*. For Nat:

```coq
Fixpoint eq_nat (n m : nat) : bool :=
  match n, m with
  | 0,    0    => true
  | S k,  S l  => eq_nat k l
  | _,    _    => false
  end.
```

Combined with reflection (`reflect (n = m) (eq_nat n m)`) this gives tactical proofs via Boolean computation, the *small-scale reflection* methodology of ssreflect (Gonthier 2008).

== Models of MLTT

Type theories are validated by *categorical models*:
- *Sets*: a model of ETT (with caveats).
- *Setoids* (sets with an equivalence relation): a model of ITT validating proof irrelevance.
- *Groupoids* (Hofmann–Streicher 1998): refutes UIP.
- *Simplicial sets / Kan complexes* (Awodey–Warren 2007, Voevodsky 2009): the *univalent model*, which validates univalence.
- *Cubical sets* (Bezem–Coquand–Huber 2014): a computational model giving univalence operational content.

Each model is a *category with families* (Dybjer 1995) or equivalently a *display map category* satisfying the closure conditions for $Pi$, $Sigma$, $"Id"$, plus universe(s).

== Proof Assistants in Practice: Statistics

The mathlib (Lean) library, as of late 2025, contains over 1.5 million lines of formalised mathematics, spanning basic algebra and analysis through algebraic geometry and condensed mathematics. The *Liquid Tensor Experiment* (Scholze–Commelin–Massot 2022) formalised a major theorem of contemporary mathematics in 18 months of community effort. The *Compendium of Continuous Lattices* (Coq library), the *MathComp Analysis* library, and the *Coquelicot* real analysis library all demonstrate that dependent type theory is, today, the operating environment of formal mathematics.

For software, F\* verifies the *miTLS* TLS 1.3 implementation; *HACL\** provides verified cryptographic primitives used in Mozilla's NSS and Linux WireGuard; *Project Everest* (Microsoft Research) targets a fully verified HTTPS stack. Coq has been used to verify the *CompCert* C compiler (Leroy 2009), one of the most cited examples of formally-verified production software.

== Quantitative Type Theory (Atkey 2018)

A recent advance: track *resource usage* (linearity, erasure) in the type system. Each binder is annotated with a *quantity* $q in {0, 1, omega}$:
- $q = 0$: erased (the value is type-only, deleted in compiled code).
- $q = 1$: linear (used exactly once).
- $q = omega$: unrestricted (any number of uses).

QTT is the foundation of Idris 2 and informs the design of *linear Haskell* and *Rust*. It addresses a long-standing tension between *types as specifications* (where one wants pure dependent types) and *types as resource discipline* (linear, affine, modal types).

== Cubical Type Theory (Cohen–Coquand–Huber–Mörtberg 2018)

Univalence as a postulate breaks computation. Cubical type theory adds an *interval* primitive $bb(I)$ with endpoints $0, 1 : bb(I)$, and *paths* (functions $bb(I) arrow.r A$) replacing the identity type. Univalence then becomes a *theorem* with computational rules. Cubical Agda and the experimental cubical mode of Coq implement this.

```agda
-- Path type
_≡_ : {A : Set} → A → A → Set
_≡_ {A} a b = (i : I) → A [ i ↦ a , i ↦ b ]   -- schematic

-- Univalence: (A ≃ B) ≡ (A ≡ B)
```

The interest in cubical is dual: foundational (giving univalence computational content) and practical (proofs about functions become *path induction* with definitional reductions).

== Future Directions

Several frontiers:
- *Observational type theory* (Altenkirch–McBride–Swierstra 2007, Pujet–Tabareau 2022): makes funext + UIP + proof irrelevance definitional.
- *Two-level type theory* (Annenkov–Capriotti–Kraus 2017): a *strict* meta-theory layered over a *fibrant* object theory, useful for HoTT-internal reasoning.
- *Modal type theory*: necessity / possibility modalities, useful for staged computation and security.
- *Synthetic differential geometry* (Lawvere 1979, internal in a smooth topos): differential calculus from type-theoretic primitives.
- *Cohesive HoTT* (Schreiber 2013): geometric structure (cohesion) baked into the type theory.

== Detailed Elimination Rules

For each inductive type, CIC auto-generates:
- A *non-dependent recursor* $T"_rec"$: for computing values of an unrelated type.
- A *dependent eliminator* $T"_ind"$: for proving properties.

For $"Nat"$:
$ "nat_rec" &: Pi P : cal(U) . space P arrow.r ("Nat" arrow.r P arrow.r P) arrow.r "Nat" arrow.r P \
"nat_ind" &: Pi P : "Nat" arrow.r cal(U) . space P space 0 arrow.r (Pi n . P n arrow.r P ("S" n)) arrow.r Pi n . P n $

These are *strong*: they let you compute *and* prove. The dependent eliminator is the *induction principle*.

== The K-Rule and Streicher's Axiom

*Streicher's K-rule* (1993) is the postulate
$ K : Pi A . Pi a : A . Pi P : "Id"_A (a, a) arrow.r cal(U) . space P space "refl"_a arrow.r Pi p . P p $

K says every loop in $"Id"$ is "refl"; equivalently, UIP. It is *not* derivable from $J$ in ITT; the Hofmann–Streicher groupoid model refutes it. Coq formerly bundled K (via `Match` on `eq`) but modern Coq isolates it: `Axiom K : ...` is necessary to use.

Agda has a `--without-K` flag (default for HoTT-style development) to prevent inadvertent K use.

== Pattern Matching as Coq Definitions

Coq desugars pattern matching to recursors. The function
```coq
Fixpoint length {A} (l : list A) : nat :=
  match l with
  | nil       => 0
  | cons _ xs => S (length xs)
  end.
```
desugars to roughly
```coq
Definition length {A} := list_rect (fun _ => nat) 0 (fun _ _ ih => S ih).
```

The pattern-matching machinery in Coq's elaborator is itself nontrivial: handling *dependent* matches (where return types vary) requires the convoy pattern; *deep* patterns desugar to nested matches; *with-clauses* in Agda give yet finer control.

== Inductive Definitions vs Records

A *record* is a $Sigma$-type at heart:

```coq
Record Group : Type := {
  carrier  : Type;
  op       : carrier -> carrier -> carrier;
  e        : carrier;
  inv      : carrier -> carrier;
  assoc    : forall a b c, op a (op b c) = op (op a b) c;
  l_id     : forall a, op e a = a;
  r_id     : forall a, op a e = a;
  l_inv    : forall a, op (inv a) a = e;
  r_inv    : forall a, op a (inv a) = e
}.
```

A `Group` *is* a 9-tuple: a carrier set, operations, and proof obligations. Records are how dependent typed languages express *algebraic structures*. Coq's *classes* and Lean's *typeclasses* layer inference on top of records.

== Type Classes in Dependent Languages

Haskell-style type classes can be encoded as records of operations + automatic resolution:

```coq
Class Monoid (A : Type) := {
  mempty  : A;
  mappend : A -> A -> A;
  m_id_l  : forall a, mappend mempty a = a;
  m_id_r  : forall a, mappend a mempty = a;
  m_assoc : forall a b c, mappend (mappend a b) c = mappend a (mappend b c)
}.
```

A `Monoid` instance is a record; type-class resolution tries to find one for the required `A` automatically. Lean's typeclass elaboration is fast and supports diamond resolution (multiple paths to a single instance); Coq's is more permissive but slower.

== Performance Bottlenecks

Type-checking dependent code is computational. Common bottlenecks:
- *Universe-constraint solving*: explodes when many universe-polymorphic definitions interact. Coq's universe-checker can be the dominant cost.
- *Reduction during conversion*: a term like `2 ^ 16` may need to be reduced to `65536` during conversion, which is expensive without `vm_compute`.
- *Implicit-argument unification*: filling in `_` requires higher-order unification, which is undecidable in general; tools use heuristics (Miller's pattern unification).
- *Tactic search*: `auto`, `eauto`, `firstorder` search exponential proof spaces.

Engineering remedies: `Opaque` definitions, abstract barriers, careful `Hint` databases, `Set Universe Polymorphism Cumulativity`.

== Definitional Computation in Coq

What reduces *automatically* during conversion?
- $beta$: application of $lambda$.
- $delta$: unfolding `Definition`s that are *transparent*. `Qed`-sealed proofs are *opaque*.
- $iota$: pattern-match on a constructor.
- $zeta$: `let`-reduction.
- $eta$ (optional, for $Pi$): $f equiv lambda x . f x$.

These four-to-five rules together with strict positivity ensure SN. Adding axioms (`funext`, classical logic, etc.) does *not* break SN (axioms simply don't reduce), but it makes some propositional equalities "stuck" (can't be eliminated by computation).

== Anatomy of a Coq Proof Object

A theorem like
```coq
Theorem plus_comm : forall n m, n + m = m + n.
```
compiles to a *proof term*, a closed CIC term of type `forall n m, n + m = m + n`. The term is generated by tactics but checked independently by the kernel.

```coq
Print plus_comm.
(* plus_comm = fun n m : nat =>
     nat_ind (fun k => k + m = m + k)
             (plus_n_O m)
             (fun k IHk => trans (f_equal S IHk) (plus_n_Sm m k))
             n
   : forall n m : nat, n + m = m + n *)
```

This *de Bruijn check* (re-verifying the term against its type) is what makes proof assistants trustworthy.

== Verification Case Studies

- *Four-Color Theorem* (Gonthier 2005): formalised in Coq, leveraging ssreflect. Originally proved by Appel–Haken (1976) with a computer-checked exhaustive analysis; Gonthier *verified the verification*.
- *Feit–Thompson Odd Order Theorem* (Gonthier et al. 2012): a deep theorem of finite group theory, formalised over ~150,000 lines of Coq + MathComp.
- *CompCert* (Leroy 2009): a C compiler proven (in Coq) to preserve semantics, eliminating an entire class of compiler bugs.
- *seL4* (Klein et al. 2009): a microkernel verified in Isabelle/HOL (though Isabelle/HOL is not dependently typed in the MLTT sense, it shares the goal).
- *Liquid Tensor Experiment* (Scholze–Commelin–Massot 2022): formalisation of a central result in *condensed mathematics* in Lean+mathlib, in under two years of community effort.

== Equality Reflection vs Computational Univalence

In *Extensional* MLTT, propositional equality reflects to definitional. Type checking becomes undecidable but every proof of equality is freely usable. In *Cubical* TT, univalence has computational rules: a path between types acts like an isomorphism, with definitional reduction. The trade-off: cubical loses some MLTT identities but gains computational univalence and decidable type checking.

== A Worked Cubical Example

```agda
-- In Cubical Agda
ua : {A B : Type} → A ≃ B → A ≡ B
ua = ...    -- comes from univalence

-- Use: transport a structure
ℤ ≡ ℕ × Bool   -- via the obvious equivalence
-- transport a function ℤ → ℤ along this path
-- yields ℕ × Bool → ℕ × Bool with definitional computation rules
```

This is impossible in pure ITT + univalence-as-axiom (no computation rule); it works in cubical.

== Why Bother?

Why use dependent types at all, given the engineering cost?

+ *Specifications-as-types* turn invariants from runtime assertions to compile-time guarantees.
+ *Proof assistants* enable formal verification of software and mathematics at scales unreachable by other means.
+ *Generic programming* benefits: dependent types subsume System $F_omega$ and add term-level dispatch.
+ *Domain-specific safety*: cryptographic protocols, distributed systems, compilers, security kernels.

The cost: more programmer effort, slower compilation, smaller libraries (mathlib excepted), steeper learning curve. The trend over the past decade: tooling has improved (Lean 4, Coq's *elpi* elaboration, Idris 2's totality checker), libraries have grown, and the technique is increasingly used in industrial settings.

== Comparative Table

#table(
  columns: (auto, auto, auto, auto, auto),
  [*System*], [*Universes*], [*Inductives*], [*Equality*], [*Termination*],
  [STLC], [N/A], [None], [N/A], [SN],
  [System F], [N/A], [Church-encoded], [N/A], [SN],
  [$F_omega$], [N/A], [Church-encoded], [N/A], [SN],
  [$lambda P$ / LF], [Single], [None], [Definitional only], [SN],
  [MLTT (ITT)], [Hierarchy], [W-types + families], [Propositional Id], [SN],
  [MLTT (ETT)], [Hierarchy], [Same], [Reflected], [Undec.],
  [CIC / Coq], [`Prop` + $"Type"_i$], [Primitive families], [Propositional], [SN],
  [HoTT/CTT], [Hierarchy], [HITs], [Path types], [SN (in cubical)],
  [F\* (refinement)], [Hierarchy], [Inductives], [SMT], [Termination metric],
)

== Historical Notes

Dependent types entered logic with *de Bruijn's* *Automath* system (1968), the first computer-checked formal mathematics, used by van Benthem Jutting (1977) to verify Landau's *Grundlagen der Analysis*. Automath had dependent function types but no inductive types.

*Per Martin-Löf* developed his *Intuitionistic Type Theory* in three papers/books (1972 preprint, 1975 published, 1984 *Notes by Sambin*). The 1972 version had $cal(U) : cal(U)$ and was inconsistent (Girard's paradox); the 1975 revision introduced the predicative hierarchy. Martin-Löf was motivated philosophically by Brouwer's *intuitionism* and meaning-as-use semantics.

*Thierry Coquand and Gérard Huet* introduced the *Calculus of Constructions* in 1988, an impredicative dependent calculus unifying System F with $lambda P$. The first Coq implementation followed in 1989.

*Christine Paulin-Mohring* extended CoC with primitive inductive types (1989, 1993), yielding CIC, the kernel of Coq from version 5.10 onward.

*Hofmann and Streicher* (1995) introduced the groupoid model, refuting UIP in pure ITT, which became the seed of HoTT.

*Voevodsky's* 2009 *univalence axiom* + *cubical type theory* (Cohen–Coquand–Huber–Mörtberg 2018) brought computational content back to univalent foundations; see #xref("programming-languages", "homotopy-type-theory", label: "Homotopy Type Theory").

The *Mathematical Components* library (Coq, Gonthier et al.) and *mathlib* (Lean, the community) have demonstrated that production formalisation of nontrivial mathematics (Four-Color Theorem, Feit–Thompson Odd Order Theorem, Liquid Tensor Experiment) is possible at scale.

Today dependent types power both *proof assistants* (Coq/Rocq, Agda, Lean, Mizar, NuPRL — Isabelle/HOL is included though it is not properly dependently typed) and *production languages* (Idris 2, F\*, ATS, Dependent Haskell via singletons). The convergence with mainstream programming continues: Rust's `const generics`, Swift's `parameterized protocols`, Scala 3's *match types* all reach toward fragments of dependent typing without committing to the full system. The historical arc from STLC's three rules to CIC's full kernel runs through System F (polymorphism), $F_omega$ (type operators), $lambda P$ (term dependency), and the apex $lambda C$ where all three meet.

== Further Reading

Pierce, B. C. et al. (2010). _Software Foundations_, Vols. 1–4. Electronic textbook. The canonical Coq-based curriculum; covers logical foundations, program correctness, type systems, and verification with machine-checked proofs throughout.

Gonthier, G. et al. (2013). "A Machine-Checked Proof of the Odd Order Theorem." ITP. Demonstrates large-scale mathematical formalisation; the proof of the Feit–Thompson theorem in Coq/Mathematical Components.

Coquand, T., Huet, G. (1988). "The Calculus of Constructions." Information and Computation 76(2–3). Introduces CIC's predecessor; the type theory underlying Coq, unifying dependent types, polymorphism, and type operators.

Abrahamsson, O. et al. (2020). "Proof-Producing Synthesis of CakeML with Verified Compilation." ITP. Shows how a verified compiler (CakeML) is built and proved correct inside a proof assistant end-to-end.

Bauer, A. et al. (2017). "The HoTT Library: A Formalization of Homotopy Type Theory in Coq." CPP. Describes a large Coq library of HoTT results; a concrete case study of proof-engineering at scale.

Klein, G. et al. (2009). "seL4: Formal Verification of an OS Kernel." SOSP. The landmark proof-engineering project: a full functional correctness proof of a production microkernel, demonstrating that Hoare-logic verification scales to real systems code.

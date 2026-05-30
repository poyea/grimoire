= Dependent Types

A *dependent type* is a type that may depend on a *term*. The function type $A arrow.r B$ becomes the *dependent function type* $Pi x : A . B(x)$ where the codomain $B$ may mention $x$; the product $A times B$ becomes $Sigma x : A . B(x)$ where the second component's type depends on the first. With this single move, types acquire the full expressive power of a logic: $Pi$ encodes universal quantification, $Sigma$ encodes existence with a witness, and the *Curry–Howard correspondence* extends to predicate logic. This is the foundation of Martin-Löf Type Theory (Martin-Löf 1972, 1975, 1984), the Calculus of Constructions (Coquand–Huet 1988), and modern proof assistants Coq/Rocq, Agda, Lean, Idris, and F\*.

*See also:* _Simply-Typed Lambda Calculus_, _System F and Parametricity_, _Type Systems_, _Homotopy Type Theory_

This chapter develops dependent type theory from the ground up. We give the syntax and rules of $lambda P$ / LF; the predicative universe hierarchy and Girard's paradox at $cal(U) : cal(U)$; W-types and inductive families; intensional vs extensional MLTT; the $J$-eliminator and the (in)derivability of function extensionality; the Calculus of Inductive Constructions (CIC) underpinning Coq; sized types and well-founded recursion; universe polymorphism; worked examples in Coq, Agda, and Lean; the Curry–Howard reading for first-order predicate logic; and program extraction.

== From STLC to $Pi$ and $Sigma$

In STLC, $tau_1 arrow.r tau_2$ means a function from $tau_1$-things to $tau_2$-things. Both $tau_1, tau_2$ are *closed* types — no term-level data on them. The dependent generalisation:

*Dependent function type ($Pi$, "pi-type").* $Pi x : A . B(x)$ is the type of functions $f$ such that for every $a : A$, $f(a) : B(a)$. When $B$ does not mention $x$, $Pi x : A . B = A arrow.r B$ is the ordinary arrow.

*Dependent pair type ($Sigma$, "sigma-type").* $Sigma x : A . B(x)$ is the type of pairs $(a, b)$ with $a : A$ and $b : B(a)$. When $B$ does not mention $x$, $Sigma x : A . B = A times B$.

*Reading as logic.*
$ Pi x : A . B(x) &= forall x : A . space B(x) \
Sigma x : A . B(x) &= exists x : A . space B(x) $

The constructive reading: a proof of $forall x : A . B(x)$ is a *function* delivering for every witness $a$ a proof of $B(a)$. A proof of $exists x : A . B(x)$ is a *pair* of a witness $a$ and a proof of $B(a)$.

== $lambda P$ / LF: The Edinburgh Logical Framework

We begin with the simplest dependent calculus — $lambda P$ in the Barendregt cube, also called *LF* (Harper–Honsell–Plotkin 1993). Only types may depend on terms; no polymorphism, no type operators.

*Syntax.* Three syntactic categories: kinds, types, terms.
$ "Kinds" &space.quad K ::= * | Pi x : A . K \
"Types" &space.quad A, B ::= alpha | Pi x : A . B | lambda x : A . B | A space e \
"Terms" &space.quad e ::= x | lambda x : A . e | e_1 space e_2 $

We have type-level $lambda$ (for forming type families) and type-level application (instantiating a family at a term). The kind $*$ classifies proper types; $Pi x : A . K$ classifies type families.

*Judgments.* Four:
+ $tack.r Gamma$  ($Gamma$ is a well-formed context)
+ $Gamma tack.r K$  ($K$ is a well-formed kind)
+ $Gamma tack.r A : K$  ($A$ has kind $K$)
+ $Gamma tack.r e : A$  ($e$ has type $A$)

The rules — only the essentials, the rest follow from analogues in STLC:

```text
  Gamma |- A : *
  ------------------------ (CTX-EXT)
  |- Gamma, x : A

  Gamma |- A : *    Gamma, x : A |- B : *
  -------------------------------------------- (T-PI-FORM)
  Gamma |- Pi x : A. B : *

  Gamma, x : A |- e : B
  --------------------------------------- (T-PI-INTRO)
  Gamma |- lam x : A. e : Pi x : A. B

  Gamma |- e : Pi x : A. B    Gamma |- e' : A
  -------------------------------------------- (T-PI-ELIM)
  Gamma |- e e' : [x |-> e'] B

  Gamma |- A : *    Gamma, x : A |- B : *
  -------------------------------------------- (T-SIGMA-FORM)
  Gamma |- Sigma x : A. B : *

  Gamma |- e : A    Gamma |- e' : [x |-> e] B
  ---------------------------------------------- (T-SIGMA-INTRO)
  Gamma |- (e, e') : Sigma x : A. B

  Gamma |- p : Sigma x : A. B
  --------------------------- (T-SIGMA-FST)
  Gamma |- fst p : A

  Gamma |- p : Sigma x : A. B
  --------------------------- (T-SIGMA-SND)
  Gamma |- snd p : [x |-> fst p] B

  Gamma |- e : A    A ==beta B    Gamma |- B : *
  ----------------------------------------------- (T-CONV)
  Gamma |- e : B
```

The *conversion rule* (T-CONV) is the new ingredient: types that are $beta$-equal (in fact $beta delta iota zeta$ in CIC; see below) are interchangeable. Type checking therefore requires *deciding* equality of arbitrary terms — a notable departure from STLC.

*$beta$-rule for $Pi$.*
$ (lambda x : A . e) space e' arrow.r_beta [x |-> e'] e $

*$beta eta$-rules for $Sigma$.* $"fst" (e, e') arrow.r e$, $"snd" (e, e') arrow.r e'$, and (surjective pairing) $("fst" p, "snd" p) =_eta p$.

*Theorem.* $lambda P$ is strongly normalising. Type checking is decidable.

The first dependent example: the type family $"Vec"$ of length-indexed lists. Given $A : *$ and a term $n : "Nat"$, we form $"Vec" A space n : *$. Then $"append" : Pi A : * . Pi m n : "Nat" . "Vec" A space m arrow.r "Vec" A space n arrow.r "Vec" A space (m + n)$ — the *type* of $"append"$ guarantees the length arithmetic.

== Universes

In STLC the type $tau$ is just a syntactic category; there is no question "what is the type of `Int`?" In dependent type theory, types are *also* terms — they live in a *universe*. A naïve approach $cal(U) : cal(U)$ courts paradox (see below); the standard solution is a hierarchy:
$ cal(U)_0 : cal(U)_1 : cal(U)_2 : cal(U)_3 : ... $

with the *cumulativity* rule $cal(U)_i subset.eq cal(U)_(i+1)$ — every type in level $i$ is also in level $i+1$.

*Russell vs Tarski style.*
- *Russell-style* (Martin-Löf 1984, Coq, Lean): membership in a universe *is* being a type. Write $A : cal(U)_i$ and use $A$ directly as the type. Simpler, but blurs the distinction between code and type.
- *Tarski-style* (Agda's mode): universes are *codes*, with a decoding operator $"El" : cal(U)_i arrow.r "Type"_(i+1)$. Cleaner semantically; clunkier syntactically.

Most production proof assistants use a hybrid: Russell-style at the surface, Tarski-style in the kernel.

=== Predicative vs Impredicative

A universe is *predicative* if $Pi x : A . B$ lives at the maximum of the levels of $A$ and $B$. *Impredicative* if $Pi x : A . B$ can live in a *fixed* universe regardless of $A$'s level — typically because $Pi$ is allowed to quantify over the universe itself.

Coq has a special *impredicative* universe $"Prop"$: $Pi A : "Type" . A arrow.r A : "Prop"$ even though "Type" is a larger universe. This is logically delicate: it works for $"Prop"$ but would be inconsistent for $"Type"$.

=== Girard's Paradox

*Theorem (Girard 1972, Coquand 1986).* The system *MLTT + $cal(U) : cal(U)$* (i.e., a single universe containing itself) is inconsistent.

*Sketch.* Girard's original paradox is in System U; the type-theoretic version (Coquand 1986; refined by Hurkens 1995) goes as follows. Define a *Burali-Forti–style* encoding of well-founded relations *indexed by all types*. The collection of all such relations is itself such a relation, hence its own member — producing a strictly-smaller chain $X gt X gt X gt ...$ violating well-foundedness. The contradiction is delivered as a closed term of type $bot$.

*Hurkens' miniature paradox.* In 1995 Hurkens produced a *short* paradox: a 24-line term of type $bot$ in the system $* : *$. The encoding uses a Russell-style trick: define $U = forall X . ((X arrow.r *) arrow.r X) arrow.r X$, then a paradoxical inhabitant of $bot$.

*Consequence.* Predicativity is essential. Coq's $"Prop"$ is impredicative without paradox because $"Prop"$ inhabitants are proof-irrelevant and erased.

== Inductive Types

In a logical framework one wants *natural numbers*, *lists*, *trees*. There are three approaches:
+ *Church-encoded* (System F-style): $"Nat" = forall A . (A arrow.r A) arrow.r A arrow.r A$. Works but lacks dependent eliminators (Geuvers 2001 showed full induction is *not* derivable from Church encodings in System F).
+ *W-types* (Martin-Löf): a single primitive of "well-founded trees" encoding all strictly-positive inductive types.
+ *Primitive inductive families* (CIC): every inductive type a primitive, with an auto-generated eliminator.

=== W-Types

*Definition.* Given $A : cal(U)$ and $B : A arrow.r cal(U)$, the *W-type* $W_(x : A) B(x)$ is the inductive type generated by:
$ "sup" : Pi a : A . space (B(a) arrow.r W_(x : A) B(x)) arrow.r W_(x : A) B(x) $

A W-tree is a *node* labelled $a : A$ with a *fan* of children indexed by $B(a)$.

*Recursion.* The recursor:
$ W "-rec" : Pi P : W arrow.r cal(U) . (Pi a : A . forall f : B(a) arrow.r W . space (Pi b : B(a) . P (f space b)) arrow.r P ("sup" a space f)) arrow.r Pi w : W . P(w) $

*Encodings.*
- $"Nat" = W_(b : "Bool") "if" b "then" "Empty" "else" "Unit"$. The two constructors correspond to $b = "true"$ (no children, gives $"zero"$) and $b = "false"$ (one child, gives $"succeeds"$).
- $"List" A = W_(p : "Unit" + A) "case" p "of" "inl" \_ => "Empty" | "inr" \_ => "Unit"$. Nil and cons.
- Binary trees: $W_(p : "Unit" + A) (...)$ analogously.

*Theorem (Martin-Löf).* W-types together with $Pi, Sigma$, unit, empty, sums, and identity give a system in which every strictly-positive inductive type is definable.

In practice, programmers use *inductive families* directly rather than W-encoded forms; W-types remain a theoretical reduction.

=== Inductive Families

A general inductive family is a type indexed by terms. In Coq:

```coq
Inductive Vec (A : Type) : nat -> Type :=
  | vnil  : Vec A 0
  | vcons : forall n, A -> Vec A n -> Vec A (S n).
```

The index `n : nat` *varies* per constructor: `vnil` has index $0$, `vcons` has index $S n$. This is what makes "Vec" *dependent*. The auto-generated recursor `Vec_rec` is:
$ "Vec_rec" : Pi A . Pi P : (Pi n . "Vec" A space n arrow.r cal(U)) . space P space 0 space "vnil" arrow.r (Pi n . Pi a . Pi v . P space n space v arrow.r P space (S space n) space ("vcons" space n space a space v)) arrow.r Pi n . Pi v . P space n space v $

Other examples:
- $"Fin" : "Nat" arrow.r cal(U)$ — the type with exactly $n$ elements. $"Fin" 0 = $ Empty; $"Fin" (n+1) = 1 + "Fin" n$.
- $"Eq" : Pi A . A arrow.r A arrow.r cal(U)$ — the identity type, with sole constructor $"refl"_a : "Eq" A space a space a$.
- $"Acc" : Pi A . (A arrow.r A arrow.r cal(U)) arrow.r A arrow.r cal(U)$ — the accessibility predicate for well-founded recursion.

=== Strict Positivity

For an inductive declaration to be *consistent*, the type being defined must occur only *strictly positively* in the constructor argument types. The type $T$ occurs *positively* in $X$ if it never appears to the left of an arrow in $X$ (no contravariance). *Strictly* positively if it appears only at the head of a positive position (not nested inside another type-level computation that might fold it back).

*Why?* Consider the (non-strictly-positive) declaration:
```coq
Inductive Bad : Type := bad : (Bad -> Bad) -> Bad.   (* REJECTED *)
```
From this one can encode the untyped $lambda$-calculus, derive a fixed-point combinator, and inhabit $bot$. Coq's positivity checker rejects this.

=== Inductive Recursion (Dybjer 1994, 2000)

A further generalisation: a type and a function on it are defined *simultaneously* by induction. The classical example is the *universe-of-codes* construction: simultaneously define $U : cal(U)$ and $T : U arrow.r cal(U)$, where $U$'s constructors may take arguments of type $T u$ for already-constructed $u : U$.

```agda
data U : Set
T : U → Set

data U where
  nat   : U
  pi    : (a : U) → (T a → U) → U

T nat       = ℕ
T (pi a b)  = (x : T a) → T (b x)
```

Inductive-recursive definitions are *not* reducible to plain inductive types in MLTT; they are a genuinely stronger principle, and the foundation of Agda's universe machinery.

== Martin-Löf Type Theory (MLTT)

Per Martin-Löf's intuitionistic type theory (1972, revised 1975, definitive 1984) is the prototype dependent type theory. Two variants exist, differing in the treatment of *equality*.

=== Identity Types

Given $A : cal(U)$ and $a, b : A$, the *identity type* $"Id"_A (a, b)$ — also written $a =_A b$ or $"Eq" A space a space b$ — is the *proposition* that $a$ and $b$ are equal. Introduction:
$ "refl"_a : "Id"_A (a, a) $

Elimination — the *$J$-eliminator* (path induction):
$ J : Pi C : (Pi a, b : A . "Id"_A (a, b) arrow.r cal(U)) . space (Pi x : A . space C(x, x, "refl"_x)) arrow.r Pi a, b : A . Pi p : "Id"_A (a, b) . space C(a, b, p) $

with the computation rule
$ J space C space d space a space a space "refl"_a = d space a $

In words: to prove $C(a, b, p)$ for all $a, b, p$, it suffices to prove it for $a = b$ and $p = "refl"$. The motive $C$ is allowed to depend on the *proof* $p$, not just on $a$ and $b$.

=== Intensional vs Extensional

*Extensional MLTT* (ETT, original Martin-Löf 1984) adds the *equality-reflection* rule:
```text
  Gamma |- p : Id_A(a, b)
  ---------------------------- (EQ-REFL)
  Gamma |- a = b : A   (definitional)
```

Equality reflection collapses *propositional* equality (an inhabitant of $"Id"_A$) into *definitional* equality (a metatheoretic relation used in conversion). In ETT, two terms are definitionally equal whenever they are propositionally equal.

Consequences of ETT:
+ Type checking is *undecidable* (Hofmann 1995): a type may contain an arbitrarily deep computation, and conversion may need to unfold a propositional equality proof which itself depends on arbitrary computation.
+ Function extensionality $"funext" : (Pi x . f(x) = g(x)) arrow.r f = g$ is *provable*.
+ Uniqueness of identity proofs (UIP): all proofs of $a = b$ are equal.

*Intensional MLTT* (ITT) — *omits* equality reflection. Definitional equality is only $beta$ + $eta$ + $iota$ + $delta$ (term-level reductions); propositional equality may differ. ITT has:
+ Decidable type checking.
+ Function extensionality is *independent* — neither provable nor refutable from MLTT alone.
+ UIP is *independent* — Hofmann–Streicher (1998) gave the *groupoid model* refuting it.

=== The Hofmann–Streicher Groupoid Model

*Theorem (Hofmann–Streicher 1998).* There is a model of ITT in which types are *groupoids* (categories where every morphism is invertible), terms are objects, and propositional equality is *isomorphism*. In this model, UIP fails: two different isomorphisms can yield two different "proofs" of an equality.

*Consequence.* UIP is not derivable from $J$ alone — only an additional axiom (Streicher's K-rule, or equivalently UIP itself) makes it provable. This insight is the seed of *Homotopy Type Theory* (Voevodsky et al., 2009): treat types as $oo$-groupoids and add the *univalence axiom* $("Id"_(cal(U)) A space B) tilde.equiv (A tilde.equiv B)$ — equality of types *"is"* equivalence of types. See _Homotopy Type Theory_ for the full development.

=== Transport

From $p : a =_A b$ and $P : A arrow.r cal(U)$, the *transport*:
$ p^* : P(a) arrow.r P(b) $

is derived from $J$ by taking $C(x, y, q) = P(x) arrow.r P(y)$ and $d space x = "id"_(P(x))$. Transport is the operational content of "substituting equals for equals".

== The Calculus of Constructions / CIC

*Calculus of Constructions* (CoC; Coquand–Huet 1988) sits at the apex of the Barendregt cube: terms-on-terms (ordinary $lambda$), terms-on-types (polymorphism), types-on-types (type operators), types-on-terms (dependency). It is *the* impredicative dependent type theory.

*Calculus of Inductive Constructions* (CIC; Paulin-Mohring 1989, 1993) extends CoC with primitive *inductive types* (rather than W-types or Church encodings). This is the kernel of Coq/Rocq.

=== Two-Tier Universe Architecture

CIC distinguishes:
- *$"Prop"$*: an *impredicative* universe of propositions. Proof-irrelevant in most variants (the SProp universe of Gilbert–Cockx–Sozeau–Tabareau 2019 is definitionally proof-irrelevant).
- *$"Type"_i$*: a predicative hierarchy of computational types.

The discipline: data computes ($"Type"$); proofs do not ($"Prop"$). When extracted to OCaml, $"Prop"$-typed terms are erased.

```coq
(* In Coq *)
Definition andTrue : True /\ True := conj I I.   (* Prop : proof *)
Definition list_of_3 : list nat := 1 :: 2 :: 3 :: nil.  (* Type : data *)
```

=== Definitional Equality

CIC's *conversion* relation $arrow.r^*_(beta delta iota zeta eta)$:
- $beta$: ordinary function application.
- $delta$: unfolding global definitions.
- $iota$: pattern matching on a constructor — $"match" ("vcons" a space v) "with" "vnil" => ... | "vcons" x space y => e arrow.r e[x := a, y := v]$.
- $zeta$: $"let" x := e_1 "in" e_2 arrow.r [x |-> e_1] e_2$.
- $eta$ (optional, controlled): $f =_eta lambda x . f space x$.

Two terms are convertible if they reduce to a common normal form. This is decidable, but can be expensive.

=== Pattern Matching as Recursors

In CIC, the term-level $"match"$ construct is sugar for the inductive type's eliminator (recursor). For example, `match n with 0 => a | S k => b k end : C n` desugars to `nat_rect C a (fun k _ => b k) n`. The kernel works in terms of recursors; the surface uses $"match"$.

=== Proof Irrelevance and SProp

Coq's $"Prop"$ universe permits *proof-relevant* terms — two proofs of `True /\ True` are syntactically different terms, even though we don't observe the difference. Lean 4 and Coq (since 8.10) provide an alternative: $"SProp"$ — *definitionally proof-irrelevant* propositions. Any two proofs of $P : "SProp"$ are convertible.

== Termination Checking

A dependently-typed proof assistant must enforce *totality* — every function must terminate. Why? Because under Curry–Howard, a non-terminating "proof" yields a false judgment. The fixed-point $"fix"$ at type $bot$ would inhabit $bot$ and break consistency.

=== Structural Recursion

The simplest criterion: recursive calls must be on a *structurally smaller* subterm. Coq's `Fixpoint` and Agda's pattern-matching definitions use this check.

```coq
Fixpoint plus (m n : nat) : nat :=
  match m with"
  | 0    => n
  | S k  => S (plus k n)   (* k < S k structurally *)
  end.
```

Decidable, simple, but limited — many natural functions are not structurally recursive (e.g., $"merge_sort"$, which splits in halves).

=== Sized Types (Hughes–Pareto–Sabry 1996, Abel)

Annotate types with *sizes* — ordinals tracking how "big" a term is. Recursive calls require size strictly smaller. Agda has experimental sized-types support; F\* uses a refinement-types variant.

```agda
data Nat : Size → Set where
  zero : ∀ {i} → Nat i
  suc  : ∀ {i} → Nat i → Nat (↑ i)

half : ∀ {i} → Nat i → Nat i
half zero          = zero
half (suc zero)    = zero
half (suc (suc n)) = suc (half n)   -- n has smaller size
```

=== Well-Founded Recursion via Acc

The *accessibility predicate*:
```coq
Inductive Acc {A} (R : A -> A -> Prop) (x : A) : Prop :=
  Acc_intro : (forall y, R y x -> Acc R y) -> Acc R x.
```

A relation $R$ is well-founded <==> every $x : A$ is accessible. Well-founded recursion: given $"wf" : forall x, "Acc" R space x$ and a step function, recursion peels off `Acc_intro` constructors.

This *encodes* well-foundedness in the type system: even non-structural recursions can be implemented if you provide an accessibility proof.

=== The Guard Condition

In Coq, `Fixpoint` checks the *guard condition* — a syntactic criterion ensuring termination. The check is necessarily approximate (termination is undecidable!); some terminating definitions are rejected and must be rewritten with `Program Fixpoint` or `Function`.

== Performance: Conversion Can Be Expensive

Type-checking dependent types requires conversion checking. In CIC this means reducing terms — sometimes to normal forms. Conversion can be:
- *Lazy*: reduce only as needed for structural comparison.
- *Eager / Compiled*: `vm_compute` (Grégoire–Leroy 2002) compiles to a bytecode VM. `native_compute` (Boespflug–Dénès–Grégoire 2011) compiles to OCaml native code. Both can speed conversion checks by orders of magnitude.

*Opaque vs transparent.* Coq's `Qed` makes a proof *opaque* (its body is irrelevant for conversion); `Defined` keeps it *transparent*. Opaqueness can dramatically speed up subsequent type checking.

== Universe Polymorphism (Sozeau–Tabareau 2014)

A statement like "List is a functor" should hold *for every universe level*. Without universe polymorphism, one would have to copy the proof for each level. With it, declarations are quantified over universe levels:
$ "List" : Pi i : "Univ" . cal(U)_i arrow.r cal(U)_i $

Coq, Lean, and Agda all support universe polymorphism with various syntactic conventions. The technical work involves *universe-level inequalities* solved by a constraint solver.

== Practical Examples

=== Vectors and Append

```coq
Inductive Vec (A : Type) : nat -> Type :=
  | vnil  : Vec A 0
  | vcons : forall n, A -> Vec A n -> Vec A (S n).

Fixpoint vapp {A m n} (v1 : Vec A m) (v2 : Vec A n) : Vec A (m + n) :=
  match v1 in Vec _ m return Vec A (m + n) with
  | vnil _          => v2
  | vcons _ k a v1' => vcons A (k + n) a (vapp v1' v2)
  end.
```

The *return clause* `in Vec _ m return Vec A (m + n)` — the so-called *convoy pattern* — is necessary because the type of $v_2$ involves $n$, while the type of the result depends on $m$ which changes per branch. Without it the type checker cannot unify the branch types.

=== Decidable Propositions

```coq
Inductive Dec (P : Prop) : Type :=
  | yes : P -> Dec P
  | no  : (P -> False) -> Dec P.

Definition eq_nat_dec : forall n m, Dec (n = m).
Proof. decide equality. Defined.
```

`Dec P` is *not* `P \/ ~P` (excluded middle is not assumed); it is a *computational* witness of decidability. Functions returning `Dec` give actual decision procedures.

=== Red–Black Tree Invariants

```coq
Inductive Color := Red | Black.

(* h is the black-height *)
Inductive RBTree : Color -> nat -> Type :=
  | rbleaf : RBTree Black 0
  | rbred  : forall h, RBTree Black h -> nat -> RBTree Black h -> RBTree Red h
  | rbblk  : forall h c1 c2, RBTree c1 h -> nat -> RBTree c2 h -> RBTree Black (S h).
```

The type encodes both the *color invariant* (red has only black children) and the *black-height invariant* (all paths from root to leaf have the same black-height). Any constructed value automatically satisfies these — *correct by construction*.

=== Sorted Lists

```coq
Inductive SortedList : nat -> Type :=     (* indexed by min element *)
  | snil  : SortedList 0
  | scons : forall n m, n <= m -> SortedList m -> SortedList n.
```

Insertion can be defined dependently and proved to produce a sorted list *as a type*.

== Curry–Howard for First-Order Logic

The slogan:

#table(
  columns: (auto, auto),
  [*Logic*], [*Type Theory*],
  [$P supset Q$], [$P arrow.r Q$],
  [$P and Q$], [$P times Q$],
  [$P or Q$], [$P + Q$],
  [$top$], [$"Unit"$],
  [$bot$], [$"Empty"$],
  [$forall x : A . P(x)$], [$Pi x : A . P(x)$],
  [$exists x : A . P(x)$], [$Sigma x : A . P(x)$],
  [$a = b$], [$"Id"_A (a, b)$],
)

A *constructive* proof of $exists x : A . P(x)$ is a *witness* — a concrete $a$ together with a proof of $P(a)$. Classical existence ($not forall x . "not" P(x)$) requires the axiom of choice or excluded middle, which are not provable in MLTT.

=== Example: Constructive Existence

A classical proof might say "either $r$ is rational or irrational, so one of $sqrt(2)^(sqrt(2))$ and $(sqrt(2)^(sqrt(2)))^(sqrt(2)) = 2$ is irrational raised to irrational equalling rational". This proof gives *no* witness. A constructive proof must exhibit one: Gelfond–Schneider gives $sqrt(2)^(sqrt(2))$ as irrational, so we win with $a = b = sqrt(2)^(sqrt(2))$.

=== Why $Sigma$ for $exists$

A $Sigma x : A . P(x)$ inhabitant is a pair $(a, p)$ with $a : A$ and $p : P(a)$. First-projection gives the witness; second-projection the property. Hence constructive logic *forces* us to produce witnesses.

== Tools

=== Coq / Rocq

CIC + universe polymorphism + tactics. The standard library (`Coq.Init`) defines numbers, lists, etc. The *ssreflect* extension (Gonthier 2008) reorganises the tactic language, used in the *MathComp* library and the Four-Color Theorem proof. Coq was renamed *Rocq* in 2024.

```coq
Theorem plus_comm : forall n m, n + m = m + n.
Proof.
  induction n; intros.
  - now rewrite <- plus_n_O.
  - simpl. rewrite IHn. now rewrite plus_n_Sm.
Qed.
```

=== Agda

Pure dependently typed functional language. No tactic language — instead, *unification* + *interactive holes* + *with-clauses*. Pattern matching is more flexible than Coq's (supports advanced features like *case trees*).

```agda
plus-comm : (n m : ℕ) → n + m ≡ m + n
plus-comm zero    m = sym (+-identity-r m)
plus-comm (suc n) m = trans (cong suc (plus-comm n m)) (+-suc m n)
```

=== Lean 4

CIC + powerful *metaprogramming* (macros, syntax extensions) + the *mathlib* library, one of the largest formal mathematics libraries. Lean 4 is a *general-purpose* language — its compiler is implemented in Lean itself.

```lean
theorem add_comm (n m : Nat) : n + m = m + n := by
  induction n with
  | zero => simp
  | succ k ih => simp [Nat.add_succ, ih, Nat.succ_add]
```

=== Idris 2

A *programming-first* dependently typed language. Quantitative type theory (Atkey 2018) for tracking *erasure* and *linearity*. Compiles to Chez Scheme / Node / Racket.

```idris
append : Vect m a -> Vect n a -> Vect (m + n) a
append []        ys = ys
append (x :: xs) ys = x :: append xs ys
```

=== F\*

Refinement types + dependent types + SMT discharge + effects. Used to verify TLS implementations (Project Everest), cryptographic code (HACL\*), and the *miTLS* stack.

```fstar
val factorial : x:nat -> Tot (y:nat{y >= 1})
let rec factorial x = if x = 0 then 1 else x * factorial (x - 1)
```

== Program Extraction (Letouzey 2008)

From a constructive proof of $forall n : "Nat" . exists m . P(n, m)$, *extract* a program of type $"Nat" arrow.r "Nat"$ that computes the witness. Coq's `Extraction` produces OCaml, Haskell, or Scheme code. Crucially, $"Prop"$-typed components are *erased* (they had no computational content); only the $"Type"$-typed witness survives.

```coq
Definition divmod : forall n d : nat, d <> 0 -> { qr : nat * nat | n = fst qr * d + snd qr /\ snd qr < d }.
Proof. (* ... constructive proof ... *) Defined.

Extraction Language OCaml.
Extraction divmod.   (* yields an OCaml function nat -> nat -> nat * nat *)
```

Letouzey's extraction (2008) — soundness theorem: the extracted program *correctly computes* what the type promised, *modulo* erasure-preserving simulation.

== The Convoy Pattern

When pattern-matching on $v : "Vec" A space n$ inside an expression of type depending on $n$, the type checker loses the connection between $n$ and the *constructor* matched. The *convoy pattern* re-establishes it:

```coq
match v in Vec _ k return P k -> Q k with"
| vnil _          => fun (p : P 0)     => ...
| vcons _ k a v'  => fun (p : P (S k)) => ...
end
```

The `in` and `return` clauses tell Coq how the result type varies with the constructor. This is the workhorse of dependent pattern matching; almost every nontrivial dependent function uses it.

== Definitional vs Propositional Equality

Two terms are *definitionally* equal if they reduce to a common normal form: e.g., $1 + 1 =_(d e f) 2$ in any reasonable theory. They are *propositionally* equal if there exists an inhabitant of $"Id"$ between them; this is a *weaker* relation only because propositional equalities may rely on $J$ or transport.

In CIC, *some* equalities provable propositionally are *not* definitional: e.g., $n + 0 = n$ is provable by induction (yielding a term of type $"Id" (n + 0) space n$) but is not definitional (because `plus` recurses on its first argument, $0 + n$ reduces to $n$ but $n + 0$ does not). This asymmetry is a frequent source of frustration; modern type theories explore making more equalities hold definitionally (e.g., *cubical type theory* gives definitional univalence; *observational type theory* gives definitional funext).

== Subject Reduction Caveats

In ITT + axioms (e.g., univalence as a postulate), subject reduction can *fail*: a term might step to one whose type is provably equal but not definitionally equal. Cubical type theory (Cohen–Coquand–Huber–Mörtberg 2018) repairs this by giving univalence *computational* content — the postulate is replaced by a definitional rule.

In Coq with `Axiom`-postulated equalities, conversion becomes incomplete; tools like `rewrite` use propositional equality and pay the price.

== Inconsistency Risks

Even outside Girard's paradox, dependent type theories have subtle inconsistency traps:
- *Type-in-type* (already discussed).
- *Impredicative $"Set"$* (old Coq option, now off by default): combined with classical axioms, inconsistent (Coquand–Reynolds 1986 paradox).
- *Non-strictly-positive inductive types* (rejected by Coq, but in toy theories without the check, $bot$ is inhabited).
- *Definitional UIP + Streicher's K + univalence* — pairwise consistent, but enabling all three is contradictory.

Coq's kernel is small (~10kloc OCaml) and carefully audited; the *de Bruijn criterion* says only this kernel needs to be trusted, no matter how elaborate the surface tactic language.

== Further Worked Examples

=== Length-Indexed Map

```agda
map : {A B : Set} {n : ℕ} → (A → B) → Vec A n → Vec B n
map f []        = []
map f (x ∷ xs)  = f x ∷ map f xs
```

The output vector has *the same length* as the input — guaranteed by the type. No off-by-one possible.

=== Safe Head

```agda
head : {A : Set} {n : ℕ} → Vec A (suc n) → A
head (x ∷ _) = x
```

The type `Vec A (suc n)` rules out `[]` at the pattern level — the type checker observes `[] : Vec A 0` cannot unify with `Vec A (suc n)`, so the `[]` case is *impossible* and need not be written. This is *the* dependent-types selling point: invariant-violating cases are unrepresentable.

=== Indexed Insertion in a BST

```coq
Inductive BST : nat -> nat -> Type :=    (* indexed by [low, high] bounds *)
  | bleaf  : forall lo hi, lo <= hi -> BST lo hi
  | bnode  : forall lo hi v, lo <= v -> v <= hi ->
             BST lo v -> BST v hi -> BST lo hi.
```

A `BST lo hi` is a tree whose every element lies in `[lo, hi]`; the type prevents constructing an invalid BST.

=== Type-Level Naturals as Singletons

```idris
data Sing : Nat -> Type where"
  SZ : Sing Z
  SS : (n : Nat) -> Sing n -> Sing (S n)

-- A function on Nat can be lifted to Sing
toNat : {n : Nat} -> Sing n -> Nat
toNat SZ      = Z
toNat (SS n _) = S n
```

Singletons bridge between *static* and *runtime* values. The pattern is heavily used in Haskell via the `singletons` library to simulate dependency.

== Tactics and Proof Engineering

In Coq/Lean, a proof is produced by *tactics* — a script of commands that incrementally build a proof term. Common tactics:

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

The *elaboration* of a tactic script into a proof term is the work of the *tactic engine*; the *kernel* re-checks the resulting term independently. This division — large untrusted elaboration, small trusted kernel — is the *de Bruijn criterion* and is the architectural reason proof assistants can be trusted at all.

== Definitional Equality, $eta$, and Surprises

A standard surprise: in pure ITT, $f$ and $lambda x . f space x$ are not definitionally equal unless $eta$ is part of conversion. Modern Coq enables $eta$ for functions; for inductives, $eta$ for $Sigma$ is enabled (surjective pairing); for general inductives, $eta$ would be unsound in general.

Another: in CIC, `match` on a $"Prop"$-typed value (a proof of equality, say) is restricted — the *singleton elimination* rule says you can only eliminate into $"Prop"$, not into $"Type"$, except for very specific cases (`False`, `And`, `Eq` on decidable types). This prevents leaking proof structure into computational types — preserving proof irrelevance.

== Refinement Types and Subset Types

A *refinement* type $\{ x : A | P(x) \}$ — or in Coq, `{ x : A & P x }` (using $Sigma$) — is a *subset type*: values of $A$ satisfying $P$.

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

Combined with reflection (`reflect (n = m) (eq_nat n m)`) this gives tactical proofs via Boolean computation — the *small-scale reflection* methodology of ssreflect (Gonthier 2008).

== Models of MLTT

Type theories are validated by *categorical models*:
- *Sets*: a model of ETT (with caveats).
- *Setoids* (sets with an equivalence relation): a model of ITT validating proof irrelevance.
- *Groupoids* (Hofmann–Streicher 1998): refutes UIP.
- *Simplicial sets / Kan complexes* (Awodey–Warren 2007, Voevodsky 2009): the *univalent model* — validates univalence.
- *Cubical sets* (Bezem–Coquand–Huber 2014): a computational model — gives univalence operational content.

Each model is a *category with families* (Dybjer 1995) or equivalently a *display map category* satisfying the closure conditions for $Pi$, $Sigma$, $"Id"$, plus universe(s).

== Proof Assistants in Practice: Statistics

The mathlib (Lean) library, as of late 2025, contains over 1.5 million lines of formalised mathematics — from basic algebra and analysis through algebraic geometry and condensed mathematics. The *Liquid Tensor Experiment* (Scholze–Commelin–Massot 2022) formalised a major theorem of contemporary mathematics in 18 months of community effort. The *Compendium of Continuous Lattices* (Coq library), the *MathComp Analysis* library, the *Coquelicot* real analysis library — all demonstrate that dependent type theory is, today, the operating environment of formal mathematics.

For software, F\* verifies the *miTLS* TLS 1.3 implementation; *HACL\** provides verified cryptographic primitives used in Mozilla's NSS and Linux WireGuard; *Project Everest* (Microsoft Research) targets a fully verified HTTPS stack. Coq has been used to verify the *CompCert* C compiler (Leroy 2009) — one of the most cited examples of formally-verified production software.

== Quantitative Type Theory (Atkey 2018)

A recent advance: track *resource usage* (linearity, erasure) in the type system. Each binder is annotated with a *quantity* $q in {0, 1, omega}$:
- $q = 0$: erased — the value is type-only, will be deleted in compiled code.
- $q = 1$: linear — used exactly once.
- $q = omega$: unrestricted — any number of uses.

QTT is the foundation of Idris 2 and informs the design of *linear Haskell* and *Rust*. It addresses a long-standing tension between *types as specifications* (where one wants pure dependent types) and *types as resource discipline* (linear, affine, modal types).

== Cubical Type Theory (Cohen–Coquand–Huber–Mörtberg 2018)

Univalence as a postulate breaks computation. Cubical type theory adds an *interval* primitive $bb(I)$ with endpoints $0, 1 : bb(I)$, and *paths* — functions $bb(I) arrow.r A$ — replacing the identity type. Univalence then becomes a *theorem* with computational rules. Cubical Agda and the experimental cubical mode of Coq implement this.

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
- A *non-dependent recursor* $T"_rec"$ — for computing values of an unrelated type.
- A *dependent eliminator* $T"_ind"$ — for proving properties.

For $"Nat"$:
$ "nat_rec" &: Pi P : cal(U) . space P arrow.r ("Nat" arrow.r P arrow.r P) arrow.r "Nat" arrow.r P \
"nat_ind" &: Pi P : "Nat" arrow.r cal(U) . space P space 0 arrow.r (Pi n . P n arrow.r P ("S" n)) arrow.r Pi n . P n $

These are *strong* — they let you compute *and* prove. The dependent eliminator is the *induction principle*.

== The K-Rule and Streicher's Axiom

*Streicher's K-rule* (1993) is the postulate
$ K : Pi A . Pi a : A . Pi P : "Id"_A (a, a) arrow.r cal(U) . space P space "refl"_a arrow.r Pi p . P p $

K says every loop in $"Id"$ is "refl"; equivalently, UIP. It is *not* derivable from $J$ in ITT — the Hofmann–Streicher groupoid model refutes it. Coq formerly bundled K (via `Match` on `eq`) but modern Coq isolates it: `Axiom K : ...` is necessary to use.

Agda has a `--without-K` flag (default for HoTT-style development) to prevent inadvertent K use.

== Pattern Matching as Coq Definitions

Coq desugars pattern matching to recursors. The function
```coq
Fixpoint length {A} (l : list A) : nat :=
  match l with"
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

A `Group` *is* a 9-tuple — a carrier set, operations, and proof obligations. Records are how dependent typed languages express *algebraic structures*. Coq's *classes* and Lean's *typeclasses* layer inference on top of records.

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
- *Reduction during conversion*: a term like `2 ^ 16` may need to be reduced to `65536` during conversion — expensive without `vm_compute`.
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

These four-to-five rules together with strict positivity ensure SN. Adding axioms (`funext`, classical logic, etc.) does *not* break SN — axioms simply don't reduce — but it makes some propositional equalities "stuck" (can't be eliminated by computation).

== Anatomy of a Coq Proof Object

A theorem like
```coq
Theorem plus_comm : forall n m, n + m = m + n.
```
compiles to a *proof term* — a closed CIC term of type `forall n m, n + m = m + n`. The term is generated by tactics but checked independently by the kernel.

```coq
Print plus_comm.
(* plus_comm = fun n m : nat =>
     nat_ind (fun k => k + m = m + k)
             (plus_n_O m)
             (fun k IHk => trans (f_equal S IHk) (plus_n_Sm m k))
             n
   : forall n m : nat, n + m = m + n *)
```

This *de Bruijn check* — re-verifying the term against its type — is what makes proof assistants trustworthy.

== Verification Case Studies

- *Four-Color Theorem* (Gonthier 2005): formalised in Coq, leveraging ssreflect. Originally proved by Appel–Haken (1976) with a computer-checked exhaustive analysis; Gonthier *verified the verification*.
- *Feit–Thompson Odd Order Theorem* (Gonthier et al. 2012): a deep theorem of finite group theory, formalised over ~150,000 lines of Coq + MathComp.
- *CompCert* (Leroy 2009): a C compiler proven (in Coq) to preserve semantics, eliminating an entire class of compiler bugs.
- *seL4* (Klein et al. 2009): a microkernel verified in Isabelle/HOL — though Isabelle/HOL is not dependently typed in the MLTT sense, it shares the goal.
- *Liquid Tensor Experiment* (Scholze–Commelin–Massot 2022): formalisation of a central result in *condensed mathematics* in Lean+mathlib, in under two years of community effort.

== Equality Reflection vs Computational Univalence

In *Extensional* MLTT, propositional equality reflects to definitional. Type checking becomes undecidable but every proof of equality is freely usable. In *Cubical* TT, univalence has computational rules: a path between types acts like an isomorphism, with definitional reduction. The trade-off: cubical loses some MLTT identities but gains computational univalence — and decidable type checking.

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
+ *Generic programming* benefits — dependent types subsume System $F_omega$ and add term-level dispatch.
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

Dependent types entered logic with *de Bruijn's* *Automath* system (1968), the first computer-checked formal mathematics — used by van Benthem Jutting (1977) to verify Landau's *Grundlagen der Analysis*. Automath had dependent function types but no inductive types.

*Per Martin-Löf* developed his *Intuitionistic Type Theory* in three papers/books (1972 preprint, 1975 published, 1984 *Notes by Sambin*). The 1972 version had $cal(U) : cal(U)$ and was inconsistent (Girard's paradox); the 1975 revision introduced the predicative hierarchy. Martin-Löf was motivated philosophically by Brouwer's *intuitionism* and meaning-as-use semantics.

*Thierry Coquand and Gérard Huet* introduced the *Calculus of Constructions* in 1988 — an impredicative dependent calculus unifying System F with $lambda P$. The first Coq implementation followed in 1989.

*Christine Paulin-Mohring* extended CoC with primitive inductive types (1989, 1993), yielding CIC — the kernel of Coq from version 5.10 onward.

*Hofmann and Streicher* (1995) introduced the groupoid model, refuting UIP in pure ITT — the seed of HoTT.

*Voevodsky's* 2009 *univalence axiom* + *cubical type theory* (Cohen–Coquand–Huber–Mörtberg 2018) brought computational content back to univalent foundations; see _Homotopy Type Theory_.

The *Mathematical Components* library (Coq, Gonthier et al.) and *mathlib* (Lean, the community) have demonstrated that production formalisation of nontrivial mathematics — Four-Color Theorem (Gonthier 2005), Feit–Thompson Odd Order Theorem (Gonthier et al. 2012), Liquid Tensor Experiment (Scholze–Commelin–Massot 2022) — is possible at scale.

Today dependent types power both *proof assistants* (Coq/Rocq, Agda, Lean, Isabelle/HOL — though Isabelle is not properly dependent — Mizar, NuPRL) and *production languages* (Idris 2, F\*, ATS, Dependent Haskell via singletons). The convergence with mainstream programming continues: Rust's `const generics`, Swift's `parameterized protocols`, Scala 3's *match types* all reach toward fragments of dependent typing without committing to the full system. The historical arc — from STLC's three rules to CIC's full kernel — runs through System F (polymorphism), $F_omega$ (type operators), $lambda P$ (term dependency), and the apex $lambda C$ where all three meet.

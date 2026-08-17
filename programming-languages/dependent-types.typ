#import "../template.typ": xref

= Dependent Types

A *dependent type* is a type that may depend on a *term*. The function type $A arrow.r B$ becomes the *dependent function type* $Pi x : A . B(x)$ where the codomain $B$ may mention $x$; the product $A times B$ becomes $Sigma x : A . B(x)$ where the second component's type depends on the first. With this single move, types acquire the full expressive power of a logic: $Pi$ encodes universal quantification, $Sigma$ encodes existence with a witness, and the *Curry–Howard correspondence* extends to predicate logic. This is the foundation of Martin-Löf Type Theory (Martin-Löf 1972, 1975, 1984), the Calculus of Constructions (Coquand–Huet 1988), and modern proof assistants Coq/Rocq, Agda, Lean, Idris, and F\*.

*See also:* _Simply-Typed Lambda Calculus_, #xref("programming-languages", "system-f-and-parametricity", label: "System F and Parametricity"), #xref("programming-languages", "type-systems", label: "Type Systems"), #xref("programming-languages", "homotopy-type-theory", label: "Homotopy Type Theory")

This chapter develops dependent type theory from the ground up. We give the syntax and rules of $lambda P$ / LF; the predicative universe hierarchy and Girard's paradox at $cal(U) : cal(U)$; W-types and inductive families; intensional vs extensional MLTT; the $J$-eliminator and the (in)derivability of function extensionality; the Calculus of Inductive Constructions (CIC) underpinning Coq; sized types and well-founded recursion; universe polymorphism; worked examples in Coq, Agda, and Lean; the Curry–Howard reading for first-order predicate logic; and program extraction.

== From STLC to $Pi$ and $Sigma$

In STLC, $tau_1 arrow.r tau_2$ means a function from $tau_1$-things to $tau_2$-things. Both $tau_1, tau_2$ are *closed* types, with no term-level data on them. The dependent generalisation:

*Dependent function type ($Pi$, "pi-type").* $Pi x : A . B(x)$ is the type of functions $f$ such that for every $a : A$, $f(a) : B(a)$. When $B$ does not mention $x$, $Pi x : A . B = A arrow.r B$ is the ordinary arrow.

*Dependent pair type ($Sigma$, "sigma-type").* $Sigma x : A . B(x)$ is the type of pairs $(a, b)$ with $a : A$ and $b : B(a)$. When $B$ does not mention $x$, $Sigma x : A . B = A times B$.

*Reading as logic.*
$ Pi x : A . B(x) &= forall x : A . space B(x) \
Sigma x : A . B(x) &= exists x : A . space B(x) $

The constructive reading: a proof of $forall x : A . B(x)$ is a *function* delivering for every witness $a$ a proof of $B(a)$. A proof of $exists x : A . B(x)$ is a *pair* of a witness $a$ and a proof of $B(a)$.

== $lambda P$ / LF: The Edinburgh Logical Framework

We begin with the simplest dependent calculus, $lambda P$ in the Barendregt cube, also called *LF* (Harper–Honsell–Plotkin 1993). Only types may depend on terms; no polymorphism, no type operators.

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

The rules (only the essentials; the rest follow from analogues in STLC):

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

The *conversion rule* (T-CONV) is the new ingredient: types that are $beta$-equal (in fact $beta delta iota zeta$ in CIC; see below) are interchangeable. Type checking therefore requires *deciding* equality of arbitrary terms, a notable departure from STLC.

*$beta$-rule for $Pi$.*
$ (lambda x : A . e) space e' arrow.r_beta [x |-> e'] e $

*$beta eta$-rules for $Sigma$.* $"fst" (e, e') arrow.r e$, $"snd" (e, e') arrow.r e'$, and (surjective pairing) $("fst" p, "snd" p) =_eta p$.

*Theorem.* $lambda P$ is strongly normalising. Type checking is decidable.

The first dependent example: the type family $"Vec"$ of length-indexed lists. Given $A : *$ and a term $n : "Nat"$, we form $"Vec" A space n : *$. Then $"append" : Pi A : * . Pi m n : "Nat" . "Vec" A space m arrow.r "Vec" A space n arrow.r "Vec" A space (m + n)$; the *type* of $"append"$ guarantees the length arithmetic.

== Universes

In STLC the type $tau$ is just a syntactic category; there is no question "what is the type of `Int`?" In dependent type theory, types are *also* terms; they live in a *universe*. A naïve approach $cal(U) : cal(U)$ courts paradox (see below); the standard solution is a hierarchy:
$ cal(U)_0 : cal(U)_1 : cal(U)_2 : cal(U)_3 : ... $

with the *cumulativity* rule $cal(U)_i subset.eq cal(U)_(i+1)$: every type in level $i$ is also in level $i+1$.

*Russell vs Tarski style.*
- *Russell-style* (Martin-Löf 1984, Coq, Lean): membership in a universe *is* being a type. Write $A : cal(U)_i$ and use $A$ directly as the type. Simpler, but blurs the distinction between code and type.
- *Tarski-style* (Agda's mode): universes are *codes*, with a decoding operator $"El" : cal(U)_i arrow.r "Type"_(i+1)$. Cleaner semantically; clunkier syntactically.

Most production proof assistants use a hybrid: Russell-style at the surface, Tarski-style in the kernel.

=== Predicative vs Impredicative

A universe is *predicative* if $Pi x : A . B$ lives at the maximum of the levels of $A$ and $B$. *Impredicative* if $Pi x : A . B$ can live in a *fixed* universe regardless of $A$'s level, typically because $Pi$ is allowed to quantify over the universe itself.

Coq has a special *impredicative* universe $"Prop"$: $Pi A : "Type" . A arrow.r A : "Prop"$ even though "Type" is a larger universe. This is logically delicate: it works for $"Prop"$ but would be inconsistent for $"Type"$.

=== Girard's Paradox

*Theorem (Girard 1972, Coquand 1986).* The system *MLTT + $cal(U) : cal(U)$* (i.e., a single universe containing itself) is inconsistent.

*Sketch.* Girard's original paradox is in System U; the type-theoretic version (Coquand 1986; refined by Hurkens 1995) goes as follows. Define a *Burali-Forti–style* encoding of well-founded relations *indexed by all types*. The collection of all such relations is itself such a relation, hence its own member, producing a strictly-smaller chain $X gt X gt X gt ...$ violating well-foundedness. The contradiction is delivered as a closed term of type $bot$.

*Hurkens' miniature paradox.* In 1995 Hurkens produced a *short* paradox: a 24-line term of type $bot$ in the system $* : *$. The encoding uses a Russell-style trick: define $U = forall X . ((X arrow.r *) arrow.r X) arrow.r X$, then a paradoxical inhabitant of $bot$.

*Consequence.* Predicativity is essential. Coq's $"Prop"$ is impredicative without paradox not because of proof irrelevance (by default Coq's $"Prop"$ is proof-*relevant*; see below) but because elimination *out of* $"Prop"$ into $"Type"$ is restricted (*singleton elimination*) and $"Prop"$ is erased at extraction, so its impredicativity cannot leak into the computational fragment.

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
- $"Nat" = W_(b : "Bool") "if" b "then" "Empty" "else" "Unit"$. The two constructors correspond to $b = "true"$ (no children, gives $"zero"$) and $b = "false"$ (one child, gives $"succ"$).
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
- $"Fin" : "Nat" arrow.r cal(U)$: the type with exactly $n$ elements. $"Fin" 0 = $ Empty; $"Fin" (n+1) = 1 + "Fin" n$.
- $"Eq" : Pi A . A arrow.r A arrow.r cal(U)$: the identity type, with sole constructor $"refl"_a : "Eq" A space a space a$.
- $"Acc" : Pi A . (A arrow.r A arrow.r cal(U)) arrow.r A arrow.r cal(U)$: the accessibility predicate for well-founded recursion.

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

Given $A : cal(U)$ and $a, b : A$, the *identity type* $"Id"_A (a, b)$ (also written $a =_A b$ or $"Eq" A space a space b$) is the *proposition* that $a$ and $b$ are equal. Introduction:
$ "refl"_a : "Id"_A (a, a) $

Elimination via the *$J$-eliminator* (path induction):
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

*Intensional MLTT* (ITT) *omits* equality reflection. Definitional equality is only $beta$ + $eta$ + $iota$ + $delta$ (term-level reductions); propositional equality may differ. ITT has:
+ Decidable type checking.
+ Function extensionality is *independent*: neither provable nor refutable from MLTT alone.
+ UIP is *independent*: Hofmann–Streicher (1998) gave the *groupoid model* refuting it.

=== The Hofmann–Streicher Groupoid Model

*Theorem (Hofmann–Streicher 1998).* There is a model of ITT in which types are *groupoids* (categories where every morphism is invertible), terms are objects, and propositional equality is *isomorphism*. In this model, UIP fails: two different isomorphisms can yield two different "proofs" of an equality.

*Consequence.* UIP is not derivable from $J$ alone; only an additional axiom (Streicher's K-rule, or equivalently UIP itself) makes it provable. This insight is the seed of *Homotopy Type Theory* (Voevodsky et al., 2009): treat types as $oo$-groupoids and add the *univalence axiom* $("Id"_(cal(U)) A space B) tilde.equiv (A tilde.equiv B)$, where equality of types *"is"* equivalence of types. See _Homotopy Type Theory_ for the full development.

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
- $iota$: pattern matching on a constructor, e.g. $"match" ("vcons" a space v) "with" "vnil" => ... | "vcons" x space y => e arrow.r e[x := a, y := v]$.
- $zeta$: $"let" x := e_1 "in" e_2 arrow.r [x |-> e_1] e_2$.
- $eta$ (optional, controlled): $f =_eta lambda x . f space x$.

Two terms are convertible if they reduce to a common normal form. This is decidable, but can be expensive.

=== Pattern Matching as Recursors

In CIC, the term-level $"match"$ construct is sugar for the inductive type's eliminator (recursor). For example, `match n with 0 => a | S k => b k end : C n` desugars to `nat_rect C a (fun k _ => b k) n`. The kernel works in terms of recursors; the surface uses $"match"$.

=== Proof Irrelevance and SProp

Coq's $"Prop"$ universe permits *proof-relevant* terms: two proofs of `True /\ True` are syntactically different terms, even though we don't observe the difference. Lean 4 and Coq (since 8.10) provide an alternative, $"SProp"$, for *definitionally proof-irrelevant* propositions. Any two proofs of $P : "SProp"$ are convertible.

== Termination Checking

A dependently-typed proof assistant must enforce *totality*: every function must terminate. Why? Because under Curry–Howard, a non-terminating "proof" yields a false judgment. The fixed-point $"fix"$ at type $bot$ would inhabit $bot$ and break consistency.

=== Structural Recursion

The simplest criterion: recursive calls must be on a *structurally smaller* subterm. Coq's `Fixpoint` and Agda's pattern-matching definitions use this check.

```coq
Fixpoint plus (m n : nat) : nat :=
  match m with
  | 0    => n
  | S k  => S (plus k n)   (* k < S k structurally *)
  end.
```

Decidable, simple, but limited: many natural functions are not structurally recursive (e.g., $"merge_sort"$, which splits in halves).

=== Sized Types (Hughes–Pareto–Sabry 1996, Abel)

Annotate types with *sizes* (ordinals tracking how "big" a term is). Recursive calls require size strictly smaller. Agda has experimental sized-types support; F\* uses a refinement-types variant.

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

In Coq, `Fixpoint` checks the *guard condition*, a syntactic criterion ensuring termination. The check is necessarily approximate (termination is undecidable!); some terminating definitions are rejected and must be rewritten with `Program Fixpoint` or `Function`.

== Performance: Conversion Can Be Expensive

Type-checking dependent types requires conversion checking. In CIC this means reducing terms, sometimes to normal forms. Conversion can be:
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

The *return clause* `in Vec _ m return Vec A (m + n)` (the so-called *convoy pattern*) is necessary because the type of $v_2$ involves $n$, while the type of the result depends on $m$ which changes per branch. Without it the type checker cannot unify the branch types.

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

The type encodes both the *color invariant* (red has only black children) and the *black-height invariant* (all paths from root to leaf have the same black-height). Any constructed value automatically satisfies these, making the type *correct by construction*.

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

A *constructive* proof of $exists x : A . P(x)$ is a *witness*: a concrete $a$ together with a proof of $P(a)$. Classical existence ($not forall x . "not" P(x)$) requires the axiom of choice or excluded middle, which are not provable in MLTT.

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

Pure dependently typed functional language. No tactic language; instead it uses *unification*, *interactive holes*, and *with-clauses*. Pattern matching is more flexible than Coq's (supports advanced features like *case trees*).

```agda
plus-comm : (n m : ℕ) → n + m ≡ m + n
plus-comm zero    m = sym (+-identity-r m)
plus-comm (suc n) m = trans (cong suc (plus-comm n m)) (+-suc m n)
```

=== Lean 4

CIC + powerful *metaprogramming* (macros, syntax extensions) + the *mathlib* library, one of the largest formal mathematics libraries. Lean 4 is a *general-purpose* language whose compiler is implemented in Lean itself.

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

Letouzey's extraction (2008) comes with a soundness theorem: the extracted program *correctly computes* what the type promised, *modulo* erasure-preserving simulation.

== The Convoy Pattern

When pattern-matching on $v : "Vec" A space n$ inside an expression of type depending on $n$, the type checker loses the connection between $n$ and the *constructor* matched. The *convoy pattern* re-establishes it:

```coq
match v in Vec _ k return P k -> Q k with
| vnil _          => fun (p : P 0)     => ...
| vcons _ k a v'  => fun (p : P (S k)) => ...
end
```

The `in` and `return` clauses tell Coq how the result type varies with the constructor. This is the workhorse of dependent pattern matching; almost every nontrivial dependent function uses it.

== Definitional vs Propositional Equality

Two terms are *definitionally* equal if they reduce to a common normal form: e.g., $1 + 1 =_(d e f) 2$ in any reasonable theory. They are *propositionally* equal if there exists an inhabitant of $"Id"$ between them; this is a *weaker* relation only because propositional equalities may rely on $J$ or transport.

In CIC, *some* equalities provable propositionally are *not* definitional: e.g., $n + 0 = n$ is provable by induction (yielding a term of type $"Id" (n + 0) space n$) but is not definitional (because `plus` recurses on its first argument, $0 + n$ reduces to $n$ but $n + 0$ does not). This asymmetry is a frequent source of frustration; modern type theories explore making more equalities hold definitionally (e.g., *cubical type theory* gives definitional univalence; *observational type theory* gives definitional funext).

== Subject Reduction Caveats

In ITT + axioms (e.g., univalence as a postulate), subject reduction can *fail*: a term might step to one whose type is provably equal but not definitionally equal. Cubical type theory (Cohen–Coquand–Huber–Mörtberg 2018) repairs this by giving univalence *computational* content: the postulate is replaced by a definitional rule.

In Coq with `Axiom`-postulated equalities, conversion becomes incomplete; tools like `rewrite` use propositional equality and pay the price.

== Inconsistency Risks

Even outside Girard's paradox, dependent type theories have subtle inconsistency traps:
- *Type-in-type* (already discussed).
- *Impredicative $"Set"$* (old Coq option, now off by default): combined with classical axioms, inconsistent (Coquand–Reynolds 1986 paradox).
- *Non-strictly-positive inductive types* (rejected by Coq, but in toy theories without the check, $bot$ is inhabited).
- *Definitional UIP + Streicher's K + univalence*: pairwise consistent, but enabling all three is contradictory.

Coq's kernel is small (~10kloc OCaml) and carefully audited; the *de Bruijn criterion* says only this kernel needs to be trusted, no matter how elaborate the surface tactic language.

== Further Worked Examples

=== Length-Indexed Map

```agda
map : {A B : Set} {n : ℕ} → (A → B) → Vec A n → Vec B n
map f []        = []
map f (x ∷ xs)  = f x ∷ map f xs
```

The output vector has *the same length* as the input, guaranteed by the type. No off-by-one possible.

=== Safe Head

```agda
head : {A : Set} {n : ℕ} → Vec A (suc n) → A
head (x ∷ _) = x
```

The type `Vec A (suc n)` rules out `[]` at the pattern level: the type checker observes `[] : Vec A 0` cannot unify with `Vec A (suc n)`, so the `[]` case is *impossible* and need not be written. This is *the* dependent-types selling point: invariant-violating cases are unrepresentable.

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
data Sing : Nat -> Type where
  SZ : Sing Z
  SS : (n : Nat) -> Sing n -> Sing (S n)

-- A function on Nat can be lifted to Sing
toNat : {n : Nat} -> Sing n -> Nat
toNat SZ      = Z
toNat (SS n _) = S n
```

Singletons bridge between *static* and *runtime* values. The pattern is heavily used in Haskell via the `singletons` library to simulate dependency.


== Further Reading

Martin-Löf, P. (1984). _Intuitionistic Type Theory_. Bibliopolis. The lecture notes that introduced Martin-Löf type theory; covers identity types, universes, and the propositions-as-types reading.

Pierce, B. C. et al. (2010). _Software Foundations_, Vol. 1: Logical Foundations. Electronic textbook. The most accessible hands-on introduction to Coq and dependent types, developing program verification from first principles.

Nordström, B., Petersson, K., Smith, J. M. (1990). _Programming in Martin-Löf's Type Theory_. Oxford University Press. Covers MLTT in depth with examples of dependent record types and the setoid model for extensional equality.

Bradley, A. R., Manna, Z. (2007). _The Calculus of Computation_. Springer. Treats decidable fragments of first-order logic and their use in program verification via SMT; a practical complement to the proof-theoretic approach.

Bove, A., Dybjer, P., Norell, U. (2009). "A Brief Overview of Agda — A Functional Language with Dependent Types." TPHOLs. Introduces Agda's syntax, totality checker, and universe polymorphism; a good entry point for the proof-assistant perspective.

Univalent Foundations Program (2013). _Homotopy Type Theory: Univalent Foundations of Mathematics_. Institute for Advanced Study. Provides the modern dependent-type foundation; Chapter 1 gives the type-theoretic primitives from which all others are built.
= Homotopy Type Theory

Homotopy type theory (HoTT) is the synthesis of two disciplines that for half a century had eyed each other from across an unbridged gulf: Martin-Löf's *intensional type theory*, in which propositions are types and proofs are programs, and *abstract homotopy theory*, in which spaces are identified up to continuous deformation and the equality of two paths is itself an object of study. The bridge, perceived by (Awodey–Warren 2009) and (Voevodsky 2006–2014) and developed into the *Univalent Foundations* programme, is breathtakingly simple to state: the identity type $"Id"_A(a, b)$ is to be read as the *space of paths* from $a$ to $b$ inside the space $A$. With this reading the entire apparatus of higher categorical algebra becomes available inside a constructive type theory, and the type theory in turn provides a *synthetic* language for homotopy theory in which the formal rules are the geometric intuitions.

*See also:* _Dependent Types_, _Type Systems_

The development is recent — the canonical reference, the HoTT Book (UFP 2013), was written collectively by visitors to the Institute for Advanced Study during Voevodsky's Special Year on Univalent Foundations — and the field is moving rapidly. The technical heart of the subject is the *univalence axiom* and the *higher inductive types*; the computational interpretation of both is *cubical type theory* (Cohen–Coquand–Huber–Mörtberg 2016); the mechanised libraries live in *Cubical Agda*, the *Coq HoTT library*, and *Lean's mathlib4*.

== The Identity Type, Read as a Path Space

In Martin-Löf type theory, the *identity type* $"Id"_A(a, b)$ (often written $a =_A b$) is the type whose inhabitants are *proofs* that $a$ and $b$ are equal as elements of $A$. The only constructor is reflexivity:

$ "refl"_a : "Id"_A(a, a) $

The eliminator is *path induction* (the $J$ rule): to define a function out of $"Id"_A(a, b)$ for all $a, b$, it suffices to define it for the case $a = b$ and the proof is $"refl"_a$:

```text
  C : Pi (a b : A). Id_A(a,b) -> U
  c : Pi (a : A). C a a refl_a
  ----------------------------------------- (J)
  J(C, c) : Pi (a b : A) (p : Id_A(a,b)). C a b p
```

with the computation rule $J(C, c) space a space a space "refl"_a equiv c space a$.

For decades the orthodox reading was: $J$ says equal things are interchangeable; an inhabitant of $"Id"_A(a, b)$ is a *witness* of equality but carries no further structure. This reading was *consistent* (the *uniqueness of identity proofs*, UIP, can be added as an axiom without contradiction) but was not *forced* by the rules.

The *homotopy* reading reinterprets every component:

- A *type* $A$ is a *space*.
- An *element* $a : A$ is a *point* in the space.
- An *identity* $p : "Id"_A(a, b)$ is a *path* (continuous map $[0,1] arrow.r A$ with endpoints $a$ and $b$).
- *Reflexivity* $"refl"_a$ is the *constant path* at $a$.
- An *identity between identities* $alpha : "Id"_("Id"_A(a,b))(p, q)$ is a *homotopy* (continuous deformation) between paths $p$ and $q$.
- Identity between identities between identities is a *2-homotopy*, and so on up the tower.

The path induction principle $J$, under this reading, says: every path between $a$ and $b$ is *deformable* to the constant path on $a$ (when one allows the endpoint $b$ to vary). In topology this is the *contractibility* of based path spaces, a basic fact.

The reading was anticipated by *groupoid models* of type theory (Hofmann–Streicher 1998). Hofmann and Streicher showed that the *category of groupoids* is a model of intensional type theory in which UIP *fails* — there are types with two distinct, non-equal identity proofs. The model proved that intensional type theory is *strictly weaker* than extensional type theory with UIP, and opened the door to interpretations with *non-trivial* identity structure.

== Types as $oo$-Groupoids

The full development requires *higher* identities. The tower

$ A, quad "Id"_A(a, b), quad "Id"_("Id"_A(a,b))(p, q), quad "Id"_("Id"_("Id"_A(a,b))(p,q))(alpha, beta), quad dots.h $

is not arbitrary structure: it organises into an *$oo$-groupoid*, a higher-dimensional algebraic gadget whose objects are points, whose 1-morphisms are paths, whose 2-morphisms are homotopies, and so on, with composition associative and unital *up to higher morphisms*.

*Theorem (Lumsdaine 2009; van den Berg–Garner 2011).* Every type $A$ in Martin-Löf type theory carries the structure of a *weak $oo$-groupoid* whose 0-cells are the elements of $A$, whose 1-cells are inhabitants of $"Id"_A$, and so on.

*Proof sketch.* One defines, by simultaneous induction over the dimension, the *composition* operations and the *coherence laws*. Composition $p tack.r q$ for paths $p : "Id"_A(a, b)$ and $q : "Id"_A(b, c)$ is defined by path induction on $q$, sending $"refl"_b$ to $p$. The inverse $p^(-1) : "Id"_A(b, a)$ is defined by path induction on $p$, sending $"refl"_a$ to $"refl"_a$. The associator and unitors live one dimension up and are themselves obtained by path induction. The pattern continues: each coherence law is a higher path whose existence is proved by induction on the lower ones. $square$

The composition $p tack.r q$ is associative *not strictly* but only up to a 2-path $alpha_(p,q,r) : "Id"((p tack.r q) tack.r r, p tack.r (q tack.r r))$, and these associators satisfy *Mac Lane's pentagon* up to a 3-path, and so on. The full coherence is captured by saying the structure is a *Kan complex* in the sense of simplicial homotopy theory.

== Operations on Paths

The basic operations and their derivations:

- *Inverse* $p^(-1) : "Id"_A(b, a)$. Defined by $J$ on $p$, sending $"refl"_a$ to $"refl"_a$.
- *Concatenation* $p tack.r q : "Id"_A(a, c)$ (also written $p plus.circle q$). Defined by $J$ on $q$.
- *Whiskering* $p tack.r alpha$ for a 2-path $alpha$ at the right endpoint of $p$. Two-dimensional analogue, defined by $J$.

These satisfy the *groupoid laws up to higher path*:

$ "refl"_a tack.r p &= p \
p tack.r "refl"_b &= p \
p tack.r p^(-1) &= "refl"_a \
p^(-1) tack.r p &= "refl"_b \
(p tack.r q) tack.r r &= p tack.r (q tack.r r) $

Each equation is itself an inhabitant of an identity type — a *2-path* — and these 2-paths in turn satisfy coherences as 3-paths, and so on.

A function $f : A arrow.r B$ acts on paths by the *action on paths* operation $"ap"_f$ (or just $"ap"$ when $f$ is clear):

$ "ap"_f : "Id"_A(a, b) arrow.r "Id"_B(f(a), f(b)) $

Definition: by $J$ on $p$, $"ap"_f ("refl"_a) :equiv "refl"_(f(a))$. The action is *functorial*: $"ap"_f ("refl"_a) = "refl"_(f(a))$, $"ap"_f (p tack.r q) = "ap"_f (p) tack.r "ap"_f (q)$, $"ap"_f (p^(-1)) = ("ap"_f (p))^(-1)$, all up to higher paths.

For a *dependent* function $f : Pi x : A . B(x)$ the type $"Id"_B(f(a), f(b))$ is ill-typed: $f(a) : B(a)$ and $f(b) : B(b)$ live in different fibres. One needs the *transport* operation to ferry $f(a)$ along $p$ into $B(b)$:

$ "transport"_B (p, -) : B(a) arrow.r B(b) $

Definition: by $J$ on $p$, $"transport"_B ("refl"_a, -) :equiv "id"_(B(a))$. Then the *dependent action on paths* is

$ "apd"_f : Pi p : "Id"_A(a, b) . "Id"_(B(b))("transport"_B (p, f(a)), f(b)) $

The transport operation is the workhorse of HoTT proofs: it incarnates the principle that "equal things have the same properties".

== Equivalences

Two types $A$ and $B$ are *equivalent*, written $A tilde B$, when there is a map $f : A arrow.r B$ that has a two-sided inverse up to homotopy. Naïvely:

$ "QInv"(f) :equiv (g : B arrow.r A) times (Pi y : B . "Id"(f(g(y)), y)) times (Pi x : A . "Id"(g(f(x)), x)) $

This *quasi-inverse* formulation has a defect: $"QInv"(f)$ is not in general a *proposition* — a single $f$ may have many genuinely distinct quasi-inverses. To get a *good* notion of equivalence we want $"isEquiv"(f)$ to be a *proposition*, so that being an equivalence is a property, not a structure.

Voevodsky's solution: $f$ is an equivalence <==> every *fibre* of $f$ is *contractible*.

$ "isEquiv"(f) :equiv Pi y : B . "isContr"("fib"_f(y)) $

where $"fib"_f(y) :equiv (x : A) times "Id"_B(f(x), y)$ is the fibre over $y$, and $"isContr"(C) :equiv (c : C) times Pi x : C . "Id"_C(c, x)$ is contractibility (existence of a center to which everything is uniquely path-connected).

*Theorem.* $"isEquiv"(f)$ is a proposition (any two inhabitants are equal), and is logically equivalent to $"QInv"(f)$.

*Proof.* That $"isContr"$ is a proposition is direct from the definitions. A product of propositions is a proposition. The logical equivalence with $"QInv"$ uses the standard back-and-forth construction. $square$

Two further equivalent formulations are useful in practice:

- *Bi-invertible*: $f$ has a left inverse and a right inverse separately. $"BiInv"(f) :equiv ((g : B arrow.r A) times "id" tilde g circle.small f) times ((h : B arrow.r A) times f circle.small h tilde "id")$. The two inverses are equal, but their *equality is part of the data*, which makes the type a proposition.
- *Half-adjoint equivalence*: $f$ together with a quasi-inverse and a *coherence law* relating the two homotopies. This is the formulation that generalises smoothly to higher categories and is the standard in the HoTT Book.

All three give the same notion of equivalence; the differences matter for proving things *about* the type of equivalences.

== The Univalence Axiom

For a universe $cal(U)$ of types, and any two types $A, B : cal(U)$, there is a canonical map

$ "idtoeqv" : "Id"_(cal(U))(A, B) arrow.r (A tilde B) $

defined by $J$: send $"refl"_A$ to the identity equivalence $"id"_A$. The *Univalence Axiom* (Voevodsky 2009) says this canonical map is itself an equivalence:

$ "UA" : Pi A, B : cal(U) . "isEquiv"("idtoeqv"_(A, B)) $

Equivalently: $"Id"_(cal(U))(A, B) tilde (A tilde B)$. Equivalent types are *equal as elements of the universe*.

The consequences are immediate and pervasive:

*Function extensionality* (FunExt). For $f, g : Pi x : A . B(x)$,
$ ("Id"(f, g)) tilde (Pi x : A . "Id"(f(x), g(x))) $
The pointwise-equal functions are equal. (FunExt is a *theorem*, not an axiom, in the presence of univalence; see (Voevodsky 2010).)

*Propositional extensionality* (PropExt). For propositions $P, Q$, $("Id"(P, Q)) tilde (P arrow.l.r Q)$. Logically equivalent propositions are equal.

*Structure invariance*. If two algebraic structures (groups, rings, topological spaces, ...) are isomorphic, they are *equal* as types — and so every property provable for one is provable for the other. This is the *principle of equivalence* of category theory, made into a theorem of the foundational system rather than an informal mathematical practice.

The axiom is *consistent*: Voevodsky's *simplicial set model* (Kapulkin–Lumsdaine 2012, Voevodsky 2009) interprets type theory in the category of Kan simplicial sets, and the interpretation of $"idtoeqv"$ is the canonical comparison map between *path spaces* and *equivalences*, which is an equivalence by classical homotopy theory. The construction requires the axiom of choice and the law of excluded middle in the meta-theory, but yields a model in which type theory + UA is consistent.

The axiom is *not computational* in book HoTT: a closed term of an identity type need not reduce to $"refl"$, breaking *canonicity*. This was the central open problem from 2009 to 2016, and it was solved by cubical type theory.

== n-Types and the Truncation Hierarchy

The classical homotopy types are stratified by *truncation level*:

- *$(-2)$-type / contractible.* $"isContr"(A) :equiv (c : A) times Pi x . "Id"(c, x)$. A single point, up to higher equality.
- *$(-1)$-type / proposition.* $"isProp"(A) :equiv Pi x, y . "Id"_A(x, y)$. At most one point.
- *$0$-type / set.* $"isSet"(A) :equiv Pi x, y . "isProp"("Id"_A(x, y))$. Equality is a proposition (UIP holds for $A$).
- *$1$-type / groupoid.* $"isGroupoid"(A) :equiv Pi x, y . "isSet"("Id"_A(x, y))$. 2-equalities are propositions.
- *$n$-type* defined recursively: $A$ is an $n$-type if every $"Id"_A(x, y)$ is an $(n-1)$-type.

In HoTT, the types are *not* automatically sets — a generic type $A : cal(U)$ might have a rich path space. The choice to add UIP as an axiom amounts to *postulating* that all types are sets, collapsing the hierarchy to its zeroth level. In *cubical* HoTT, by contrast, types can be of arbitrary truncation level, and many naturally arise from constructions like *higher inductive types*.

*Theorem (Hedberg 1998).* If $A$ has *decidable equality* ($Pi x, y . ("Id"(x, y)) + not "Id"(x, y)$), then $A$ is a set.

*Proof.* Decidability gives a function $"dec" : Pi x, y . ("Id"(x, y)) + not "Id"(x, y)$. From this one constructs a *retraction* of any identity type onto a proposition, hence the identity type is itself a proposition. $square$

The theorem explains why types like $NN$ and $"Bool"$ are sets: decidability of their equality is computable.

The truncation hierarchy has a *reflector*: for any type $A$ and any $n$, there is the *$n$-truncation* $||A||_n$ defined as a higher inductive type with constructors

```text
  |_|_n : A -> ||A||_n
  trunc : Pi (x y : ||A||_n) ... up to n+1 layers of paths
```

The $(-1)$-truncation $||A||_(-1)$ is *propositional truncation*: a type representing "$A$ is inhabited" without remembering which inhabitant. The $0$-truncation $||A||_0$ is the *set of connected components* (in the geometric reading).

== Higher Inductive Types

The novelty that distinguishes HoTT from a mere reinterpretation of MLTT is the *higher inductive type* (HIT): an inductive type whose constructors include *both points and paths*. The prototypical example is the *circle* $S^1$:

```agda
data S1 : Set where
  base : S1
  loop : base == base
```

The circle has *one point* $"base"$ and *one non-trivial path* $"loop"$ from $"base"$ to itself. The eliminator allows defining functions $f : S^1 arrow.r A$ by giving a point $a : A$ (the image of $"base"$) and a *loop at $a$*, $ell : "Id"_A(a, a)$ (the image of $"loop"$).

Computation: $f("base") equiv a$ definitionally, and $"ap"_f ("loop")$ equals $ell$ — *propositionally* in book HoTT, *definitionally* in cubical HoTT.

The circle, defined this way, has the homotopy type of the geometric circle:

*Theorem (Licata–Shulman 2013).* $pi_1(S^1) = ZZ$, where $pi_1(S^1) :equiv ||"Id"_(S^1)("base", "base")||_0$.

*Proof outline.* One constructs the *universal cover* of $S^1$ as a HIT, with fibre $ZZ$. The fibre over $"base"$ is $ZZ$. The loop induces a map $ZZ arrow.r ZZ$ which one shows is the successor. By a covering-space argument, made fully formal in HoTT, $pi_1(S^1) = ZZ$. The proof is fully mechanised in Agda and Coq. $square$

This is *synthetic* algebraic topology: the proof never mentions points of the geometric circle (continuous functions, $epsilon$-$delta$, etc.); it works entirely with the type-theoretic constructors and the path algebra.

Further HITs:

- *2-sphere* $S^2$: one point $"base"$ and one *2-loop* $"surf" : "Id"_("Id"("base","base"))("refl", "refl")$.
- *$n$-sphere* $S^n$: one point and one $n$-loop.
- *Suspension* $Sigma A$: two points $N, S$ and a path $"merid"(a) : "Id"(N, S)$ for each $a : A$. One has $S^(n+1) tilde Sigma S^n$.
- *Pushout* $A union.sq.big_C B$ of a span $A arrow.l C arrow.r B$: an HIT with one point per element of $A$, one point per element of $B$, and a path connecting the two images of each $c : C$.
- *Truncations* $||A||_n$ as above.
- *Quotients* $A "/" R$ for an equivalence relation $R$: one point per element, one path per related pair.

HITs allow the *direct* construction of objects that in set-theoretic foundations require quotients, equivalence classes, or other workarounds. The integers $ZZ$ can be constructed as a HIT with two points $0, 1$ and a path *successor*, generating the loop space of $S^1$.

== Synthetic Homotopy Theory

The phrase *synthetic homotopy theory* refers to doing homotopy theory entirely inside the type theory, without an underlying topological model. Major results:

- $pi_1(S^1) = ZZ$ (Licata–Shulman 2013).
- $pi_n(S^n) = ZZ$ (Brunerie 2016, in his thesis).
- The *Freudenthal suspension theorem*: for an $n$-connected pointed type $A$, the suspension map $pi_k(A) arrow.r pi_(k+1)(Sigma A)$ is an equivalence for $k < 2n$ (Lumsdaine–Finster–Licata 2013).
- The *Blakers–Massey theorem*, a key result connecting pushouts and connectedness.
- *Brunerie's number*: a closed term in HoTT that represents the order of $pi_4(S^3)$, conjectured to be $2$, finally computed to be $plus.minus 2$ by *cubical normalisation* (Ljungström–Mörtberg 2023).
- Cohomology, Eilenberg–MacLane spaces, the Hopf fibration — all constructed and reasoned about.

The Brunerie story is the cleanest demonstration of the *computational* dimension. Book HoTT defines a term whose normal form should be the integer $2$, but without canonicity for univalence and HITs, no proof assistant could *reduce* the term. Cubical Agda (Vezzosi–Mörtberg–Abel 2019) implements cubical type theory's reduction rules and computes the value.

== Cubical Type Theory

(Bezem–Coquand–Huber 2014) and especially (Cohen–Coquand–Huber–Mörtberg 2016, CCHM) constructed *cubical type theory*, a refinement of MLTT in which the univalence axiom and the HIT computations are *provable rather than postulated*. The key move: replace abstract identity types $"Id"_A(a, b)$ with *path types* $"Path"_A(a, b)$ defined as functions out of an *interval* $II$:

$ "Path"_A(a, b) :equiv (i : II) arrow.r A "with" p(0) equiv a, p(1) equiv b $

The interval $II$ is a *primitive* type with two endpoints $0, 1 : II$ and is *not* an ordinary inductive type — it has a more refined structure governed by *face* and *degeneracy* maps drawn from the theory of *cubical sets*.

The *Kan composition* operations are the heart of the system. A *Kan composition* takes:

- An *open box* of types: a partial path in $cal(U)$ defined on some faces of a cube,
- A *cap*: a value on the missing face,
- And produces a value on the opposite face, by *transport* along the partial path.

The operations are written $"hcomp"$ (homogeneous composition) and $"transp"$ (transport along a path of types). Together they give a *constructive* interpretation of univalence: the *Glue* type former allows constructing, for any equivalence $e : A tilde B$, a path $"ua"(e) : "Id"_(cal(U))(A, B)$ such that $"transp"("ua"(e), -) equiv e$.

*Theorem (Canonicity for Cubical Type Theory, Huber 2018).* In CCHM cubical type theory, every closed term of base type (natural numbers, booleans) reduces to a canonical form. In particular, $"transp"("ua"(e), n) equiv e(n)$ for $n : NN$ — univalence computes.

*Proof.* A direct semantic argument using the *cubical sets* model and a *normalisation-by-evaluation* construction. The key is that the Kan composition operations are *deterministic*: there is no axiom whose computation is left undefined. $square$

A variant, the *ABCFHL* (Angiuli–Brunerie–Coquand–Favonia–Harper–Licata) cartesian cubical theory (2017), uses a different interval (cartesian rather than de Morgan) and supports a slightly different proof structure. The two are equivalent in expressive power but differ in technical convenience.

Cubical Agda (Vezzosi–Mörtberg–Abel 2019) and *redtt* / *cooltt* (Sterling et al.) implement these theories. The Cubical Agda library mechanises a substantial fragment of synthetic homotopy theory and has been used to compute Brunerie's number.

== Univalent Foundations of Mathematics

Univalent foundations propose to replace ZFC as the foundation of mathematics. The *univalence principle* makes the foundation respect the structural nature of mathematical objects: a group $G$ is determined by its data and equations *up to isomorphism*; in univalent foundations isomorphic groups are *equal*, so theorems proved for one transfer to the other automatically.

The HoTT Book's chapters on *algebra* set up monoids, groups, rings as *$Sigma$-types* of carrier $+ $ operations $+ $ axioms, where the carrier is a *set* (a $0$-type) and the axioms are *propositions*. Two such structures are equal iff they are isomorphic. The *structure identity principle* generalises this to arbitrary algebraic structures.

Beyond algebra: *categories* in HoTT come in two flavours — *precategories* (composition is associative *up to equality*) and *univalent categories* (the type of objects is at least a 1-type and the inclusion of isomorphisms into identity is an equivalence). Univalent categories satisfy the structure identity principle for categories: equivalent categories are equal.

== Tool Ecosystem

- *Cubical Agda* — built into recent Agda; full cubical type theory; the largest synthetic homotopy theory library.
- *Coq HoTT library* (Bauer, Gross, Lumsdaine, Shulman, Spitters, Wood et al.) — book HoTT with axioms; mature, large; Coq's classical foundation makes this less computational but tactically powerful.
- *Lean's mathlib4* — uses Lean 4's dependent type theory; has a small but growing category-theory layer with HoTT flavour; no native HITs.
- *redtt*, *cooltt*, *cartesian-cubical-agda* — research languages exploring variants.
- *UniMath* (Voevodsky, Ahrens, Grayson, et al.) — Coq library of univalent mathematics, axiom-based, focused on Voevodsky's vision.

== Connections to Computer Science

The themes of HoTT have analogues in everyday programming:

- *Protocol equivalence as path types.* A change of representation that preserves all observable behaviour is precisely an equivalence; univalence says the original and refactored versions are *equal*, justifying transport of all reasoning.
- *Refactoring as univalence.* When two data structures are isomorphic (e.g. an array-of-structs and a struct-of-arrays), univalence justifies systematically rewriting code from one to the other; *cubical normalisation* turns the proof of equivalence into the actual rewriting program.
- *Modular structures.* Software modules are algebraic structures with operations and equational laws; the univalence principle says interchangeable implementations are equal as modules — a *type-theoretic* justification for the *Liskov substitution principle*.
- *Schema migration.* Database schema evolution, when the migration is information-preserving, is an equivalence; transporting queries along the equivalence is exactly what migration tooling must do.
- *Homotopy type checking* (Cavallo–Harper 2019): formal sessions where the type checker is itself written in cubical type theory; checking a program *computes* the equivalences witnessing optimisations.

The pedagogy of HoTT is challenging: students must hold the geometric and the syntactic readings simultaneously, and the proof assistant's feedback (especially for HITs and univalence) is still maturing. But the *philosophy* — that mathematics and computation are two faces of a higher-dimensional theory in which equality is structured — has begun to permeate programming languages research in ways the eventual industrial adoption will make plain.

== Two Sample Constructions in Cubical Agda

The integers as a HIT:

```agda
data Int : Set where
  pos : N -> Int
  neg : N -> Int
  zro : pos zero == neg zero
```

The path constructor `zro` identifies the two presentations of zero, giving a type with exactly the integers.

The circle and its loop space:

```agda
data S1 : Set where
  base : S1
  loop : base == base

ΩS1 : Set
ΩS1 = base == base

helix : S1 -> Set
helix base       = Int
helix (loop i)   = ua sucEquiv i     -- univalence supplies the path

winding : ΩS1 -> Int
winding p = transport (lam i -> helix (p i)) (pos zero)
```

The function `winding` takes a loop and reads off its winding number — the integer that the loop encodes. The proof that `winding` is an equivalence is mechanised in Cubical Agda and computes: `winding (loop · loop · loop)` reduces to `pos 3`.

== Where the Field Stands

HoTT is not a replacement for ordinary type theory in the immediate sense that gradual typing is a replacement for `any` — its programmer-facing benefits are subtler. But for *proof engineers* it changes the rules of the game: structural reasoning replaces equality-juggling, refactorings are first-class, and homotopy theory becomes a programming subject. The eventual integration into mainstream proof assistants — already substantially complete in Cubical Agda, partial in Coq, nascent in Lean — will bring the discipline to a much wider audience.

The open problems are deep:

1. *Higher universes and large eliminations.* Univalence with size issues — how to manage the hierarchy of universes when each universe is itself a type — remains a delicate engineering matter.
2. *Modal HoTT* (Shulman 2018; Riehl–Shulman 2017) extends the theory with *modal operators* to support *cohesive* and *parametric* reasoning; computational interpretation is partial.
3. *Synthetic differential and algebraic geometry* in HoTT (Cherubini, Wellen, et al.) — bringing geometric methods inside the type theory.
4. *Tool maturity.* Error messages, performance, library coverage — practical engineering still lags the theory.

The grand bet of univalent foundations — that the natural setting for mathematics is a higher-dimensional type theory in which the only structure-preserving notion of equality is up to equivalence — has, since 2013, accumulated enough mechanised mathematics and enough computational backing that it is no longer a curiosity. It is one of two viable foundational programmes (the other being set theory + classical first-order logic), with the unique selling point that *computation and equality are unified*.

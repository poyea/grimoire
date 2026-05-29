= Subtyping and Polymorphism

Subtyping is the principle that lets you pass a `Cat` to a function expecting an `Animal`. Polymorphism is the principle that lets a function operate uniformly across types it does not name. Where pure parametric polymorphism (System F) erases all type information at runtime and forbids any inspection, subtyping adds a *partial order* on types — and the resulting interaction with type constructors, references, generics, and recursion is the source of three decades of language-design subtlety.

_See also: _Type Systems_, _Linear and Substructural Type Systems_, _Effects and Handlers_._

== The Subsumption Rule

The defining feature of a subtyping system is a single inference rule:

```text
   Gamma |- e : tau_1     tau_1 <: tau_2
   ----------------------------------------- (T-Sub)
   Gamma |- e : tau_2
```

If $tau_1$ is a subtype of $tau_2$, then any term of type $tau_1$ may be promoted to type $tau_2$. The *subtype relation* $lt.tri$ is required to be a preorder (reflexive and transitive) on types; it may or may not be antisymmetric depending on whether types are considered up to alpha-equivalence and unfolding.

The slogan attributed to Liskov (1987): *whenever an instance of $tau_1$ is expected, an instance of $tau_2$ may be supplied*. The reverse reading — *subtyping is implicit coercion* — leads to the *coercive* interpretation: subsumption is a logically silent insertion of a function $tau_1 arrow.r tau_2$. The *subset* interpretation, in contrast, treats subtypes as honest set inclusions on the semantic interpretation of types.

#table(
  columns: (auto, auto),
  [*Interpretation*], [*$tau_1 lt.tri tau_2$ means*],
  [Subset semantics], [$[| tau_1 |] subset.eq [| tau_2 |]$],
  [Coercive semantics], [there is a canonical coercion $c : tau_1 arrow.r tau_2$ inserted by elaboration],
)

The interpretations agree on simple data but diverge on function types: in the coercive view, $tau_1 arrow.r tau_2 lt.tri tau_1' arrow.r tau_2'$ is mediated by composition with explicit coercions.

== Record Subtyping: Width and Depth

The two foundational rules for structural records.

*Width subtyping:* a record with more fields is a subtype of a record with fewer (assuming the shared fields agree).

$ {ell_1 : tau_1, dots, ell_n : tau_n, ell_(n+1) : tau_(n+1)} lt.tri {ell_1 : tau_1, dots, ell_n : tau_n} $

The reading: anyone expecting only the first $n$ fields can be safely given a record that has those *and more*. Semantically, in the subset interpretation, the set of records of the wider type is a subset of the set satisfying the narrower constraint.

*Depth subtyping:* covariant subtyping inside each field.

$ "for each" i, tau_i lt.tri tau_i' => {ell_1 : tau_1, dots, ell_n : tau_n} lt.tri {ell_1 : tau_1', dots, ell_n : tau_n'} $

Reading: if each field's type narrows to a subtype, the whole record narrows.

```text
   for each i, tau_i <: tau_i'
   ------------------------------------------------------------------- (S-RcdDepth)
   {l_i : tau_i}_{i in I} <: {l_i : tau_i'}_{i in I}

   I superset_eq J
   ------------------------------------------- (S-RcdWidth)
   {l_i : tau_i}_{i in I} <: {l_j : tau_j}_{j in J}

   {k_i : tau_i}_{i in I} is a permutation of {l_j : tau_j}_{j in J}
   ------------------------------------------------------------------ (S-RcdPerm)
   {k_i : tau_i}_{i in I} <: {l_j : tau_j}_{j in J}
```

These three rules together — width, depth, permutation — characterise *structural* subtyping on records. Languages with *nominal* type systems (Java, C\#) admit only declared subclass relationships, not arbitrary structural overlap.

== Variance: The Function Type

The function arrow does not behave covariantly in both positions.

*Theorem (function subtyping).* $tau_1 arrow.r tau_2 lt.tri tau_1' arrow.r tau_2'$ <==> $tau_1' lt.tri tau_1$ and $tau_2 lt.tri tau_2'$.

The argument position is *contravariant* — to substitute for a function expecting $tau_1$, you need a function that accepts *at least* $tau_1$, i.e., one accepting any $tau_1' supset.eq tau_1$. The return position is *covariant* — your replacement may return *at most* what's expected.

```text
   tau_1' <: tau_1     tau_2 <: tau_2'
   ---------------------------------------------- (S-Arrow)
   tau_1 -> tau_2 <: tau_1' -> tau_2'
```

The error of conflating the two — *assuming* the arrow is covariant in both positions — is the canonical broken inheritance pattern. Eiffel infamously allowed contravariant *argument* refinement; the resulting unsoundness (a `Cat::eats(Mouse)` overriding `Animal::eats(Food)`, then receiving a `Salad`) required runtime checks.

=== References and Arrays: Invariance

For a mutable cell `Ref tau`, both read and write operations exist:

- read: $"Ref" tau arrow.r tau$ — covariant in $tau$;
- write: $"Ref" tau arrow.r tau arrow.r 1$ — contravariant in $tau$.

To be sound for both, `Ref` must be *invariant*: $"Ref" tau lt.tri "Ref" tau'$ <==> $tau = tau'$.

*Java arrays* (1996) infamously got this wrong: arrays are declared covariant. The compiler accepts:

```java
Object[] xs = new String[10];
xs[0] = Integer.valueOf(42);   // compiles fine
```

But the store is unsound: `String[]` cannot store an `Integer`. Java patches this with a *runtime* `ArrayStoreException` check on every array write — a fixed-cost penalty for an unsound static rule.

C\# 1.0 inherited the same mistake. Modern languages (Scala, Kotlin, Swift) get it right: arrays are invariant, with read-only / write-only views (`IReadOnlyList`, `IEnumerable`) carrying their appropriate variance.

== Variance Annotations

Languages with parameterised types and subtyping must declare variance somehow.

#table(
  columns: (auto, auto, auto, auto),
  [*Language*], [*Mechanism*], [*Site*], [*Notation*],
  [Scala, Kotlin], [Declaration-site variance], [type parameter], [`+T` covariant, `-T` contravariant, `T` invariant],
  [Java, C\#], [Use-site variance / wildcards], [type argument], [`? extends T`, `? super T`],
  [TypeScript], [Both, plus inference], [position], [`out T`, `in T` (since 4.7), or inferred],
  [OCaml], [Declaration-site], [type parameter], [`+'a`, `-'a`, `'a` (invariant)],
)

*Declaration-site* variance (Scala) requires the library author to declare once how a generic type behaves; *use-site* variance (Java wildcards) defers the choice to each consumer. The trade: declaration-site is less verbose and centralises the reasoning; use-site is more flexible when a single generic type is naturally invariant but used covariantly at specific call sites.

Emir, Kennedy, Russo, Yu (*Variance and generalized constraints for C\# generics*, 2006) gave the canonical treatment of declaration-site variance in a setting with subtyping; their work directly informed C\# 4.0's `in`/`out` annotations.

== Bounded Quantification: System F$(lt.tri)$

The marriage of subtyping with parametric polymorphism is *System F$(lt.tri)$* (Cardelli–Wegner 1985, refined by Curien–Ghelli 1992). Type abstraction may be *bounded* by an upper subtype constraint:

$ tau ::= dots | forall alpha lt.tri tau_1 . tau_2 | exists alpha lt.tri tau_1 . tau_2 $

A bounded universal $forall alpha lt.tri tau_1 . tau_2$ is the type of a function polymorphic over all subtypes $alpha$ of $tau_1$. Reading: *"for any type $alpha$ that is at most $tau_1$, the term has type $tau_2$ with $alpha$ in scope"*.

```text
   Gamma, alpha <: tau_1 |- e : tau_2
   --------------------------------------- (T-TAbs)
   Gamma |- (Lambda alpha <: tau_1. e) : forall alpha <: tau_1. tau_2

   Gamma |- e : forall alpha <: tau_1. tau_2     Gamma |- tau' <: tau_1
   ----------------------------------------------------------------------- (T-TApp)
   Gamma |- e [tau'] : [alpha |-> tau'] tau_2

   Gamma, alpha <: tau_1 |- tau_2 <: tau_2'     tau_1' <: tau_1
   ----------------------------------------------------------------- (S-All)
   Gamma |- forall alpha <: tau_1. tau_2 <: forall alpha <: tau_1'. tau_2'
```

The subtyping rule for bounded universals is *itself* contravariant in the bound: a tighter bound is a *less* general type, hence a *subtype* of the looser-bound type. The intricate interplay of bound-variable shadowing, contravariance of bounds, and transitivity is what makes F$(lt.tri)$'s metatheory subtle.

=== Undecidability of Full F$(lt.tri)$

*Theorem (Pierce 1992, 1994).* Subtyping for full System F$(lt.tri)$ is undecidable.

Pierce reduces from the halting problem for two-counter machines, encoding counter states as deeply-nested bounded universals and counter operations as subtyping derivations. The non-termination of subtyping checks tracks the non-termination of the encoded machine.

The diagnosis: the *bound* of a quantifier $forall alpha lt.tri tau_1 . dots$ may itself be a universal $forall beta lt.tri tau . dots$; nesting these triggers the unbounded recursion. The *kernel* F$(lt.tri)$ (Cardelli–Martini–Mitchell–Scedrov) requires the bound to remain the *same* in the rule S-All — which sacrifices some expressiveness but restores decidability.

*Kernel F$(lt.tri)$* (Curien–Ghelli) modifies the S-All rule:

```text
   Gamma, alpha <: tau_1 |- tau_2 <: tau_2'
   --------------------------------------------------- (S-All kernel)
   Gamma |- forall alpha <: tau_1. tau_2 <: forall alpha <: tau_1. tau_2'
```

— the bound $tau_1$ must be identical on both sides. Subtyping in kernel F$(lt.tri)$ is decidable (exponential in the worst case; polynomial in practice).

=== Bounded Existentials

Dual to bounded universals: $exists alpha lt.tri tau_1 . tau_2$. The reading is *"there exists a type $alpha$, hidden but known to be at most $tau_1$, satisfying $tau_2$"*. This is the type-theoretic basis for *abstract data types*: a `Stack` exposed as $exists alpha lt.tri "Object" . {"push" : alpha arrow.r 1, "pop" : alpha arrow.r alpha}$ hides its representation while making its supertype publicly known.

=== System $F_(lt.tri omega)$

Adding *type operators* (functions on types, kind $* arrow.r *$) gives System $F_(omega)$; adding bounded quantification on top gives $F_(lt.tri omega)$. This is the type-theoretic core of Scala's compiler and of theoretical work on object encodings.

== Subtyping for Recursive Types

A *recursive type* $mu alpha . tau$ binds $alpha$ in $tau$, satisfying the fixed-point equation $mu alpha . tau = [alpha |-> mu alpha . tau] tau$. There are two semantics:

- *Isorecursive*: $mu alpha . tau$ and its unfolding are *isomorphic* but not equal. Conversion requires explicit `fold` / `unfold`. OCaml uses isorecursive types.
- *Equirecursive*: $mu alpha . tau$ and its unfolding are *literally equal* — types are infinite (regular) trees. Type unification must handle cyclic structures.

For subtyping, the *coinductive* characterisation is fundamental. Define a relation $cal(R)$ to be a *simulation* if whenever $sigma cal(R) tau$:

- if $sigma = sigma_1 arrow.r sigma_2$ and $tau = tau_1 arrow.r tau_2$, then $tau_1 cal(R) sigma_1$ and $sigma_2 cal(R) tau_2$;
- if $sigma = {ell_i : sigma_i}_(i in I)$ and $tau = {ell_j : tau_j}_(j in J)$ with $J subset.eq I$, then $sigma_j cal(R) tau_j$ for each $j in J$;
- $dots$ (clauses for each connective).

*Theorem (Amadio–Cardelli 1993).* The greatest simulation is the subtype relation on the infinite-tree unfoldings of recursive types.

The algorithmic content: subtyping is checked by *coinduction* — the algorithm maintains an *assumption set* of pairs known to be in $cal(R)$ and adds new pairs as it descends, succeeding when it would infinite-loop. Brandt and Henglein (1997) refined this to an efficient algorithm running in $O(n^2)$ on the size of the input types.

```text
ALGORITHM subtype(sigma, tau, A):
    if (sigma, tau) in A: return true                -- coinductive hypothesis
    A' := A union {(sigma, tau)}
    case (unfold sigma, unfold tau) of
      (sigma_1 -> sigma_2, tau_1 -> tau_2):
          return subtype(tau_1, sigma_1, A') and subtype(sigma_2, tau_2, A')
      ({l_i : sigma_i}, {l_j : tau_j}):
          return J subset I and forall j. subtype(sigma_j, tau_j, A')
      ...
      otherwise: return false
```

The assumption set $A$ is the *coinductive certificate* that recursive subtyping terminates without sacrificing soundness.

== Intersection Types

*Intersection types* (Coppo–Dezani 1980; Pottinger 1980) add a connective $tau_1 inter tau_2$ for types *"both"* of $tau_1$ and $tau_2$. The subtyping rules:

$ tau_1 inter tau_2 lt.tri tau_1 quad tau_1 inter tau_2 lt.tri tau_2 \
"if" tau lt.tri tau_1 "and" tau lt.tri tau_2, "then" tau lt.tri tau_1 inter tau_2 $

— intersection is the *greatest lower bound* in the subtype lattice.

*Theorem (Coppo–Dezani–Venneri 1981; Pottinger 1980).* A $lambda$-term has an intersection type in a non-trivial system iff it is strongly normalising.

Intersection types thus *characterise* termination — but at the price of *undecidable* type inference. The full system has no principal types and no decision procedure.

*Practical use:* TypeScript has intersection types `A & B`; Scala has `A with B` (subtle differences from intersection); Flow has `A & B`. These are restricted fragments where decidability is retained.

=== Refinement Intersections (Freeman–Pfenning 1991)

A useful restricted form: intersection of *refinements* of a single base type. E.g., refining `List` to `EvenList` and `OddList`, with subtyping `EvenList <: List`, `OddList <: List`. Intersections within this lattice are decidable and useful for sort-checking.

== Union Types

Union types $tau_1 union tau_2$ — values of *either* type — appear in TypeScript, Flow, Ceylon, and Scala 3. The subtyping rules:

$ tau_1 lt.tri tau_1 union tau_2 quad tau_2 lt.tri tau_1 union tau_2 \
"if" tau_1 lt.tri tau "and" tau_2 lt.tri tau, "then" tau_1 union tau_2 lt.tri tau $

— union is the *least upper bound*.

Unions interact poorly with overloading and inference; languages adopting them generally couple them with *narrowing* — discriminated by type-tests:

```typescript
function area(s: Circle | Square): number {
  if (s.kind === "circle") return Math.PI * s.r * s.r;
  else                      return s.side * s.side;
}
```

The compiler tracks the *flow-sensitive* type of `s` inside each branch — a feature called *type narrowing* or *occurrence typing* (Tobin-Hochstadt–Felleisen 2008, in Typed Racket).

== Refinement Types

A *refinement type* refines an existing type by a logical predicate:

$ {x : "Int" | x > 0} quad {"xs" : "List" alpha | "length" "xs" > 0} $

Subtyping reduces to *implication* between refinements: ${x : "Int" | P(x)} lt.tri {x : "Int" | Q(x)}$ <==> $forall x . P(x) => Q(x)$.

The predicate language is typically chosen to admit a decision procedure (SMT-solvable theories: linear arithmetic, bit-vectors, uninterpreted functions, arrays). The resulting type system delivers strong guarantees with no proof-writing burden on the programmer — the SMT solver handles all the routine reasoning.

*Liquid Haskell* (Vazou–Seidel–Jhala 2014) — Hindley–Milner plus refinements over decidable theories. Annotations sit in comment pragmas:

```haskell
{-@ type Nat = {v:Int | v >= 0} @-}

{-@ length :: xs:[a] -> {v:Nat | v == len xs} @-}
length :: [a] -> Int
length []     = 0
length (_:xs) = 1 + length xs

{-@ head :: {xs:[a] | len xs > 0} -> a @-}
head :: [a] -> a
head (x:_) = x
```

Calling `head []` is now a *type error* — the SMT solver fails to discharge the obligation `len [] > 0`. Liquid Haskell has been used to verify properties of `Data.Text`, `Data.Vector`, and full bytestring libraries.

*F$"*"$* (Swamy et al. 2016) — pushes further: dependent types, refinements, *and* an effect system, with SMT for routine obligations and tactics for harder ones. F$"*"$ has been used for the verified TLS implementation in *miTLS* and for the verified cryptographic library *HACL$"*"$*, deployed in Mozilla Firefox, Linux kernel, and WireGuard.

The pattern — *types refined by SMT-decidable propositions* — is the most promising current path to *practical* program verification: stronger than ordinary types, more automated than full dependent types.

== Row Polymorphism (Rémy 1989)

A different generalisation: instead of admitting subtyping on records, allow polymorphism over the *unspecified fields*. A function operating on records is polymorphic over an extra row variable $rho$:

$ "getName" : forall alpha rho . {"name" : alpha | rho} arrow.r alpha $

— `getName` accepts any record having a `name` field, regardless of what other fields are present.

Row polymorphism gives most of the practical benefits of structural subtyping while keeping inference *predictable*: no subsumption rule, no need for least-upper-bound joins. OCaml's object system, PureScript, and PureScript's effect rows all use row polymorphism. TypeScript's *spread* and *rest* on object types is a row-polymorphic feature in a structurally-subtyped clothing.

Rémy's *scoped labels* (1989; Leijen 2005 for effects) order the labels in a row to make unification efficient and to allow *shadowing* of duplicate labels.

== Subtyping as Coercion: Categorical Semantics

Categorical semantics of subtyping (Reynolds 1980, Mitchell 1988, Breazu-Tannen–Coquand–Gunter–Scedrov 1991): the subtype relation is interpreted by *faithful functors* between categories of types-as-objects. The subsumption rule becomes the action of a *coherence* requirement: two derivations of the same typing judgement, possibly using different chains of subsumptions, must yield *equal* terms after coercion insertion.

*Theorem (coherence, Breazu-Tannen et al. 1991).* For F$(lt.tri)$ with kernel rule, the coercion semantics is coherent: the meaning of a term is independent of the derivation chosen.

Without coherence, the *meaning* of a program would depend on which derivation the type-checker happened to find — a disaster for predictability. Languages with non-coherent subtype systems (Scala 2's implicit conversions, early TypeScript) have suffered the resulting confusion.

== Variance Polymorphism

Modern type systems allow *quantifying over variances*. TypeScript 4.7 introduced `in` and `out` declaration-site modifiers; Scala 3 generalises further with abstract *type members* whose variance can be declared per-member.

A *use-site variance annotation* in Java:

```java
void copy(List<? extends T> src, List<? super T> dst) {
    for (T t : src) dst.add(t);
}
```

`? extends T` is the *covariant* view of `List<T>` (read-only); `? super T` is the *contravariant* view (write-only). This is precisely Kennedy–Pierce's *use-site variance* (2007), proven sound and decidable.

*Kennedy–Pierce theorem (2007).* Subtyping for Java generics with use-site variance is decidable; declaration-site variance with F-bounded polymorphism is *undecidable* in the full Java generics system.

The undecidability of Java generics — established formally by Grigore (2017) — derives from the interaction of declaration-site variance, F-bounds, and wildcards. Practical Java compilers terminate on every realistic program but admit constructions where the type checker enters an infinite loop.

== Liskov Substitution Principle, Object Encodings

The *Liskov Substitution Principle* (Liskov–Wing 1994) is the behavioural counterpart of structural subtyping: *a derived $cal(C)$ must preserve the behavioural contracts of its base $cal(C)$*. Specifically, in an overriding method:

- preconditions may be *weakened*;
- postconditions may be *strengthened*;
- invariants must be *preserved*.

— exactly the variance pattern of function types, lifted to logical contracts.

Object encodings in $lambda$-calculus typically combine *bounded existentials* (to hide state) with *recursive types* (for `self`) and *F-bounded polymorphism* (for binary methods). Cook (1989, 2009) gives the classical comparison of *abstract data type* style (existential) vs *object* style (Self-quantified) encodings; Bruce, Cardelli, and Pierce (1999) give the canonical encoding under *MyType* polymorphism.

== Practical Verdict and Comparison

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Language*], [*Subtyping*], [*Variance*], [*Decidability*], [*Verification path*],
  [Java], [nominal subclass + wildcards], [use-site (wildcards)], [undecidable in full], [JML / OpenJML],
  [C\#], [nominal + `in`/`out`], [declaration-site], [decidable in practice], [Code Contracts (deprecated)],
  [Scala 3], [structural + nominal], [declaration-site], [undecidable], [Stainless],
  [TypeScript], [structural], [both, partial inference], [unsound in places], [no built-in verifier],
  [OCaml], [structural via objects + nominal modules], [declaration-site], [decidable], [Why3, Coq],
  [Haskell + Liquid], [HM no subtyping; refinements], [n/a], [SMT-decidable], [SMT solver],
  [F$"*"$], [refinements + dependent], [n/a], [SMT + tactics], [native],
)

The overall trend of the past 15 years: away from elaborate subtype-and-variance machinery (Java's wildcards, Scala 2's path-dependent types) and toward *refinement-plus-SMT* (Liquid Haskell, F$"*"$, Dafny) or *structural plus row polymorphism* (TypeScript, PureScript). The reason: SMT solvers and row inference are *predictable and fast*; deep subtype-derivation search is neither.

== A Worked Subsumption Trace

Consider the term

```text
let f : ({x : Int}) -> Int = \r. r.x in
f {x = 7, y = "hi"}
```

The argument has type ${"x" : "Int", "y" : "String"}$; the parameter expects ${"x" : "Int"}$. Subtyping derivation:

```text
       {x:Int, y:String} subset {x:Int}     refl on Int
       ---------------------------------------------------- (S-RcdWidth)
       {x:Int, y:String} <: {x:Int}
       ---------------------------------------------------- (T-Sub)
       Gamma |- {x=7, y="hi"} : {x:Int}

       Gamma |- f : {x:Int} -> Int    Gamma |- {x=7, y="hi"} : {x:Int}
       ----------------------------------------------------------------- (T-App)
       Gamma |- f {x=7, y="hi"} : Int
```

The subsumption step is *silent*: it changes no values, performs no runtime work, only adjusts the static type. Compare with the function-arrow variance case where the type changes are likewise free but the elaborator might insert coercion combinators in a coercive semantics — for higher-order subtyping, the coercion is $eta$-expansion plus pointwise composition.

== Polymorphism, Subtyping, and Their Synthesis

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Discipline*], [*Quantification*], [*Subtype reasoning*], [*Inference*], [*Examples*],
  [System F], [explicit type abstraction], [none], [undecidable in general], [Haskell rank-2, Coq Core],
  [Hindley–Milner], [implicit at `let`], [none], [decidable, $O(n alpha(n))$], [OCaml, SML, Haskell],
  [F$(lt.tri)$ (full)], [bounded universal], [yes, full], [undecidable], [theoretical],
  [F$(lt.tri)$ (kernel)], [bounded universal], [yes, restricted], [decidable, exponential worst-case], [Scala foundation],
  [Row polymorphism], [implicit at `let` over rows], [via row extension only], [decidable], [OCaml objects, Koka, PureScript],
  [Refinement types], [implicit, with predicates], [via SMT implication], [decidable iff theory is], [Liquid Haskell, F$"*"$],
  [Dependent types], [explicit $Pi$, $Sigma$], [definitional + propositional], [usually undecidable], [Agda, Idris, Coq, Lean],
)

A modern language designer's pragmatic recipe: *prefer parametric polymorphism with row polymorphism over subtyping for inference-friendliness*; *use refinements where stronger guarantees are needed*; *reserve full bounded quantification for foundational work*; *if subtyping is required (interop, OO heritage), enforce a discipline that preserves decidability and coherence*.

The historical lesson — repeated in Eiffel, Java, Scala 2, TypeScript — is that subtyping is *easy* to add and *hard* to make sound, decidable, and predictable simultaneously. The pieces are well understood now; the design tasks remaining are tasteful selection from the menu, not invention of fundamentally new machinery.

_See also: _Type Systems_ for the basic subsumption rule and parametric polymorphism, _Linear and Substructural Type Systems_ for the orthogonal axis of how often a value is used, _Effects and Handlers_ for the analogous question on the side-effect axis._

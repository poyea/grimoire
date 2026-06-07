= Simply-Typed Lambda Calculus: Extensions and Advanced Topics

== Extensions

=== Products, Sums, Unit, Empty

Already specified above. Each extension is a *positive* type former: it has an introduction rule and an elimination rule, with $beta$-reductions cancelling intro/elim. SN extends straightforwardly via reducibility (extend $cal(R)$ by clauses for product and sum).

=== System T: Primitive Recursion (Gödel 1958)

Gödel's System T adds natural numbers with a primitive recursor:
$ tau &::= ... | "Nat" \
e &::= ... | 0 | S space e | "rec"(e_z, e_s, e_n) $

with the typing
```text
  Gamma |- e_z : tau    Gamma |- e_s : Nat -> tau -> tau    Gamma |- e_n : Nat
  ---------------------------------------------------------------------       (T-REC)
  Gamma |- rec(e_z, e_s, e_n) : tau
```
and reduction
$ "rec"(e_z, e_s, 0) &arrow.r e_z \
"rec"(e_z, e_s, S space n) &arrow.r e_s space n space "rec"(e_z, e_s, n) $

System T is strongly normalising (Tait 1967; this was actually Tait's original target). The terms of $lambda^arrow.r + T$ at type $"Nat" arrow.r "Nat"$ compute exactly the *provably total functions* in first-order Peano arithmetic, a vast class including Ackermann's function, but properly contained in the $mu$-recursive functions. System T separates the *higher-order primitive recursive* from the *general recursive*.

=== Fixed Points

To recover Turing completeness we add a fixed-point operator $"fix"$:
$ Gamma tack.r e : tau arrow.r tau  /  Gamma tack.r "fix" space e : tau \
"fix" space (lambda x : tau . e) arrow.r [x |-> "fix"(lambda x : tau . e)] e $

Once $"fix"$ is added, $lambda^arrow.r + "fix"$ is Turing complete; SN fails; Curry–Howard now corresponds to a *classical* (or inconsistent) logic: every proposition is "provable" via the inhabitant $"fix"(lambda x . x)$. This is the price of general recursion: type soundness still holds (well-typed programs do not get stuck), but they may diverge, and the language is no longer a sound logic.

== Type Checking and Inference

For Church-style $lambda^arrow.r$, type checking is straightforward: every binder is annotated, so a single pass reading T-VAR, T-ABS, T-APP suffices, $O(n)$ in the term size.

For Curry-style $lambda^arrow.r$ (no annotations), type inference is performed by *constraint generation + unification*:
+ Assign a fresh metavariable $alpha_x$ to each variable.
+ For each abstraction $lambda x . e$, introduce a fresh $alpha$ for $x$ and recurse, producing a body type $beta$; emit nothing; the term has type $alpha arrow.r beta$.
+ For each application $e_1 space e_2$ producing types $tau_1, tau_2$, emit constraint $tau_1 = tau_2 arrow.r gamma$ with $gamma$ fresh.
+ Solve all constraints by Robinson unification.

The result is a *principal type*, the most-general type from which all valid types are substitution instances (Hindley 1969). Type inference for $lambda^arrow.r$ is linear in the term after near-linear unification (Damas–Milner 1982; see _Type Systems_ for the algorithm).

```ocaml
(* OCaml: Curry-style lambda calculus inferred *)
let id = fun x -> x         (* val id : 'a -> 'a *)
let app f x = f x           (* val app : ('a -> 'b) -> 'a -> 'b *)
let twice f x = f (f x)     (* val twice : ('a -> 'a) -> 'a -> 'a *)
```

```haskell
-- Haskell: same examples
id    = \x -> x             -- id    :: a -> a
app   = \f x -> f x         -- app   :: (a -> b) -> a -> b
twice = \f x -> f (f x)     -- twice :: (a -> a) -> a -> a
```

== Combinatory Logic and the SKI Calculus

Schönfinkel (1924) and Curry (1930) showed that $lambda$-calculus can be expressed without binders, using only the combinators
$ S &= lambda x . lambda y . lambda z . (x space z) space (y space z) \
K &= lambda x . lambda y . x \
I &= lambda x . x $

with reduction
$ I space x &arrow.r x \
K space x space y &arrow.r x \
S space x space y space z &arrow.r (x space z) (y space z) $

In fact $I = S space K space K$ (verify: $S space K space K space x = K space x space (K space x) = x$), so $S$ and $K$ suffice.

*Theorem (Bracket Abstraction).* For every $lambda$-term $M$ with free variable $x$, there is an SK-term $T$ with no $lambda$ such that $T space x = M$. Notation: $T = [x] M$, defined by
$ [x] x &= I \
[x] M &= K space M space space space (x in.not "FV"(M)) \
[x] (M space N) &= S space ([x] M) space ([x] N) $

In the typed setting, $S$ and $K$ get the principal types
$ K &: forall alpha beta . alpha arrow.r beta arrow.r alpha \
S &: forall alpha beta gamma . (alpha arrow.r beta arrow.r gamma) arrow.r (alpha arrow.r beta) arrow.r alpha arrow.r gamma $

These are exactly the axioms K and S of the Hilbert-style presentation of $"IPC"^supset$. Bracket abstraction is the *deduction theorem* (Curry–Howard once more).

== $lambda$I and $lambda$K

*$lambda$K-calculus* (the "K" for *constant*): the calculus we have been studying, where $lambda$ is allowed even when the bound variable does not occur. $K = lambda x . lambda y . x$ is a $lambda$K term.

*$lambda$I-calculus* (Church 1941): restrict $lambda x . e$ to terms with $x in "FV"(e)$. So $K$ is not a $lambda$I term, but $I$ and $S$ are. The $lambda$I-calculus has stronger termination properties:
+ Every reduction either terminates or every subterm is reduced infinitely often (no "trash collection" of erased terms).
+ A $lambda$I-term has a normal form iff it is strongly normalising (Church 1941).

Modern computer-science presentations always work in $lambda$K. The $lambda$I-calculus is a curiosity, important historically and for *linear* and *relevance* logics, where every assumption must be used.

== Long Normal Forms and $eta$-Expansion

A *long normal form* (or $beta$-normal $eta$-long form) of a typed term is a $beta$-normal form in which every neutral subterm at an arrow type is $eta$-expanded.
Formally, by induction on type:
- At base type, a neutral $n$ is its own long form.
- At type $tau_1 arrow.r tau_2$, a neutral $n$ becomes $lambda x : tau_1 . "long"(n space "long"(x))$.

Long normal forms are unique up to $alpha$-equivalence and have the pleasant property that they can be read off directly from a typing derivation in a *bidirectional* fashion.

*Theorem.* Every well-typed $lambda^arrow.r$ term has a unique $beta$-normal $eta$-long form.

*Proof.* By SN, the $beta$-nf exists and is unique. $eta$-expand recursively at neutral arrow-typed positions; this terminates because each expansion strictly decreases the "$eta$-defect" measure. $square$

== Bidirectional Type Checking

In Church-style $lambda^arrow.r$, the annotation on every $lambda$ makes type *checking* trivial.
But in elaborator design (e.g., Lean, Agda), one wants to *minimise* annotations.

*Bidirectional type checking* (Pierce–Turner 2000) splits the typing judgment into two modes:
- *Synthesis* $Gamma tack.r e => tau$: given $e$, produce $tau$.
- *Checking* $Gamma tack.r e arrow.l.double tau$: given both, verify.

Rules:
```text
  x : tau in Gamma                       Gamma |- e_1 ==> tau_1 -> tau_2   Gamma |- e_2 <== tau_1
  --------------- (BD-VAR)               --------------------------------------------------------- (BD-APP)
  Gamma |- x ==> tau                     Gamma |- e_1 e_2 ==> tau_2

  Gamma, x : tau_1 |- e <== tau_2                  Gamma |- e ==> tau    tau = tau'
  -------------------------------- (BD-LAM)        ----------------------------------- (BD-SUB)
  Gamma |- lam x. e <== tau_1 -> tau_2             Gamma |- e <== tau'

  Gamma |- e ==> tau
  ---------------------- (BD-ANNOT)
  Gamma |- (e : tau) ==> tau
```

Bidirectional checking pushes type information *into* abstractions (no annotation needed on the binder) and *pulls* it out of variables and applications.
The user annotates *"only"* at function-definition sites, not at every $lambda$.
This is the kernel of all modern dependently-typed elaborators.

== Categorical Semantics

$lambda^arrow.r$ has a beautiful categorical model: it is the *internal language of Cartesian closed categories* (CCCs) (Lambek–Scott 1986).

A *Cartesian closed category* $cal(C)$ has:
- A terminal object $1$ (interpreting $"Unit"$).
- Binary products $A times B$ with projections $pi_1, pi_2$ and pairing $chevron.l f, g chevron.r$ (interpreting $tau_1 times tau_2$).
- Exponentials $B^A$ with an evaluation morphism $"ev" : B^A times A arrow.r B$ and currying $Lambda : "Hom"(C times A, B) arrow.r "Hom"(C, B^A)$ (interpreting $tau_1 arrow.r tau_2$).

The interpretation $[| - |]$ sends:
- Types to objects: $[| iota |] = $ chosen base object; $[| tau_1 arrow.r tau_2 |] = [| tau_2 |]^([| tau_1 |])$; $[| tau_1 times tau_2 |] = [| tau_1 |] times [| tau_2 |]$.
- Contexts to objects: $[| x_1 : tau_1, ..., x_n : tau_n |] = [| tau_1 |] times ... times [| tau_n |]$.
- Typing derivations $Gamma tack.r e : tau$ to morphisms $[| Gamma |] arrow.r [| tau |]$:
  + T-VAR ($x_i$): $pi_i : [| Gamma |] arrow.r [| tau_i |]$.
  + T-APP: $"ev" circle.small chevron.l [| e_1 |], [| e_2 |] chevron.r$.
  + T-ABS: $Lambda([| e |])$.

*Soundness.* $beta eta$-conversion is sound: if $e_1 =_(beta eta) e_2$ then $[| e_1 |] = [| e_2 |]$.

*Completeness (Lambek 1980).* The *syntactic CCC* whose objects are types and morphisms are $beta eta$-equivalence classes of terms is the free CCC on the base objects. So $lambda^arrow.r$ up to $beta eta$ *"is"* the theory of CCCs.

*Set-theoretic model.* Take $cal(C) = "Set"$: $[| tau_1 arrow.r tau_2 |] = $ set of functions $[| tau_1 |] arrow.r [| tau_2 |]$. This gives a sound (but not complete) model.

*Domain-theoretic models.* For $lambda^arrow.r + "fix"$ one passes to *Scott domains* (cpos with $bot$ and Scott-continuous maps): $[| tau_1 arrow.r tau_2 |] = $ continuous function space. The least fixed-point of $f : D arrow.r D$ is $union.sq.big_n f^n (bot)$. This is the semantic justification of $"fix"$.

== Normalisation by Evaluation

*Normalisation by evaluation* (NbE; Berger–Schwichtenberg 1991) computes $beta eta$-normal forms by interpretation in a semantic model and read-back to syntax. The picture:
$ "Terms" arrow.r^"eval" "Semantic values" arrow.r^"reify" "Normal forms" $

For $lambda^arrow.r$:
- *Semantic values*: $V_iota = "neutral terms of type" iota$, $V_(tau_1 arrow.r tau_2) = V_(tau_1) arrow.r V_(tau_2)$ (host-language functions).
- *Eval*: standard environment-passing interpretation.
- *Reify*: for arrow type, generate a fresh variable $x$ of type $tau_1$, apply the semantic function to $x$ (viewed as a neutral), reify the result, wrap in $lambda x : tau_1$. For base type, just *reflect* the neutral.
- *Reflect*: $arrow.t_iota n = n$; $arrow.t_(tau_1 arrow.r tau_2) n = lambda v . arrow.t_(tau_2) (n space (arrow.r_(tau_1) v))$.

NbE is total for $lambda^arrow.r$ (because the source is SN), is one-pass, and produces *fully* $eta$-long normal forms. It is the standard implementation strategy for dependent type checkers (Coq's `vm_compute`, Agda, Lean 4).

```haskell
-- Sketch: NbE for STLC in Haskell
data Ty = TBase | Ty :=> Ty
data Tm = Var Int | Lam Ty Tm | App Tm Tm
data Val = VNeu Neu | VFun (Val -> Val)
data Neu = NVar Int | NApp Neu Val

eval :: [Val] -> Tm -> Val
eval env (Var i)     = env !! i
eval env (Lam _ b)   = VFun (\v -> eval (v:env) b)
eval env (App f a)   = case eval env f of
                         VFun g -> g (eval env a)
                         VNeu n -> VNeu (NApp n (eval env a))

reify :: Ty -> Val -> Tm
reify TBase     (VNeu n)  = reifyNeu n
reify (a :=> b) v         = Lam a (reify b (apply v (reflect a (NVar 0))))
```

== $eta$-Equality and Extensionality

$eta$-equality $lambda x : tau . (e space x) =_eta e$ (for $x in."not" "FV"(e)$) expresses *functional extensionality*: two functions are equal iff they agree on every argument. There are two ways to treat $eta$ in the operational semantics:

+ *$eta$-conversion as a rewrite rule* ($arrow.r_eta$): breaks confluence with $beta$ in some extended calculi (e.g., when $e$ has a free variable later substituted). The combined $beta eta$-reduction is confluent for pure $lambda^arrow.r$ but not always for extensions.
+ *$eta$-expansion + restriction to $eta$-long normal forms*: every term is expanded so neutral subterms at arrow types acquire visible abstractions; $beta$ alone then suffices. Standard in NbE.

*Theorem ($beta eta$-Confluence).* The combined relation $arrow.r_(beta eta)$ is confluent on pure $lambda^arrow.r$.

*Proof.* Hindley's *Strip Lemma* combined with the parallel-reduction argument. See Barendregt 1984, §15.1. $square$

== Eta and Categorical Naturality

Under the CCC interpretation, $eta$-conversion corresponds to the equation $Lambda("ev" circle.small (f times "id"_A)) = f$, which is the *uniqueness* part of the universal property of the exponential. So an "$eta$-sensitive" categorical model is one in which the exponential adjunction is strict; the syntactic CCC modulo $beta$ alone is *not* a CCC, but modulo $beta eta$ it *"is"*.

== Practical Implementation: Type Checking $lambda^arrow.r$

A type checker for Church-style $lambda^arrow.r$:

```haskell
-- AST
data Ty = TInt | TBool | Ty :-> Ty  deriving (Eq, Show)
data Tm = Var Name
        | Lam Name Ty Tm
        | App Tm Tm
        | LitI Int | LitB Bool

type Ctx = [(Name, Ty)]

infer :: Ctx -> Tm -> Either String Ty
infer ctx (Var x) = case lookup x ctx of
  Just t  -> Right t
  Nothing -> Left ("unbound: " ++ x)
infer ctx (LitI _) = Right TInt
infer ctx (LitB _) = Right TBool
infer ctx (Lam x t body) = do
  bt <- infer ((x,t):ctx) body
  return (t :-> bt)
infer ctx (App f a) = do
  ft <- infer ctx f
  at <- infer ctx a
  case ft of
    (t1 :-> t2) | t1 == at -> Right t2
    (t1 :-> _)             -> Left ("expected " ++ show t1 ++ ", got " ++ show at)
    _                      -> Left ("not a function: " ++ show ft)
```

This is the entire type checker. It runs in $O(n^2)$ in the term size (the bottleneck is structural equality of types in T-APP; with hash-consing it becomes $O(n)$). Type inference for Curry-style is also $O(n)$ via union-find unification (Damas–Milner; see _Type Systems_).

== A Detailed Worked Reduction

Consider the closed term
$ e = (lambda f : "Int" arrow.r "Int" . lambda x : "Int" . f space (f space x)) space (lambda y : "Int" . y + 1) space 0 $
at type $"Int"$.

*Type derivation* (sketch):
- $lambda y : "Int" . y + 1 : "Int" arrow.r "Int"$ by T-ABS.
- $0 : "Int"$.
- The outer abstraction has type $("Int" arrow.r "Int") arrow.r "Int" arrow.r "Int"$ by T-ABS.
- T-APP twice gives $e : "Int"$.

*CBV reduction:*
+ Outer application is $(lambda f . lambda x . f (f x))$ applied to the value $lambda y . y + 1$, giving $lambda x . (lambda y . y + 1) ((lambda y . y + 1) space x) space space "applied to" space 0$.
+ $(lambda x . ...) space 0 arrow.r (lambda y . y + 1) ((lambda y . y + 1) space 0)$.
+ Inner application is a value redex: $(lambda y . y + 1) space 0 arrow.r 0 + 1 = 1$.
+ $(lambda y . y + 1) space 1 arrow.r 1 + 1 = 2$.

Total: 4 $beta$-steps (plus arithmetic) to normal form $2$. The reduction is *strongly normalising*: every path leads to $2$ in finitely many steps.

*CBN reduction:* reduces the leftmost-outermost redex and passes arguments unevaluated:
+ $(lambda f . ...) (lambda y . y + 1) space 0 arrow.r (lambda x . (lambda y . y + 1) ((lambda y . y + 1) space x)) space 0$.
+ $arrow.r (lambda y . y + 1) ((lambda y . y + 1) space 0)$.
+ the head $(lambda y . y + 1)$ is applied, substituting its *unevaluated* argument into the body: $arrow.r ((lambda y . y + 1) space 0) + 1$.
+ only now is the argument forced to evaluate the addition: $arrow.r (0 + 1) + 1 = 2$.

Same answer (Church–Rosser), different reduction trace.

== Historical Notes

Alonzo Church introduced the untyped $lambda$-calculus in 1932 as a foundation for logic, intending it as an alternative to set theory.
The original system was inconsistent (Kleene–Rosser 1935 found a paradox: the $lambda$-definability of a fixed-point combinator allowed reproduction of the Richard paradox).
Church responded in two ways: (1) restricting to *Church numerals* and pure computational behaviour, the *Church thesis* (1936) that $lambda$-definable equals effectively computable; (2) introducing the *simply-typed* fragment (1940) that excluded self-application and avoided the paradox.

Curry, working from a different angle on *combinatory logic* (1930), independently arrived at a typing discipline (1934) where types are assigned by inference rules rather than syntactic annotation.
This is the historical origin of the Church-vs-Curry dichotomy: *typed à la Church* (intrinsic, single type per term) vs *typed à la Curry* (extrinsic, principal type schemes).

Howard (1969, published 1980) made explicit what was already implicit in the work of Curry, Gentzen, and Heyting: the *propositions-as-types* / *proofs-as-programs* correspondence.
A different but related thread, starting with Lawvere (1969) and culminating in Lambek (1980), formulated the same correspondence categorically.

Tait's 1967 paper "Intensional interpretations of functionals of finite type" introduced the *computability* method (now called *reducibility*) that became the standard tool for SN proofs.
Girard's 1972 thesis pushed reducibility to *System F*; the second-order case requires a quantification over predicates, the *reducibility candidate* refinement.

== Worked Examples

=== Church Numerals (Untyped, Schematic)

The Church numerals
$ overline(n) = lambda f . lambda x . underbrace(f (f (... (f space x))), n " applications") $
encode natural numbers in pure $lambda$-calculus.

In $lambda^arrow.r$ proper, $overline(n)$ types at $(iota arrow.r iota) arrow.r iota arrow.r iota$ for any type $iota$; but each numeral has its own family of types, not a single polymorphic type.

In *System F* (next chapter), one can give $overline(n) : forall alpha . (alpha arrow.r alpha) arrow.r alpha arrow.r alpha$.

The successor, addition, and multiplication operations in Church encoding:
$ "succ" &= lambda n . lambda f . lambda x . f space (n space f space x) \
"add"  &= lambda m . lambda n . lambda f . lambda x . m space f space (n space f space x) \
"mul"  &= lambda m . lambda n . lambda f . m space (n space f) $

All type in $lambda^arrow.r$ at appropriate monomorphic instances; uniformly polymorphic only in System F.

=== Booleans

$ "true" &= lambda x . lambda y . x \
"false" &= lambda x . lambda y . y \
"if" &= lambda b . lambda t . lambda e . b space t space e $

Types: $"true", "false" : tau arrow.r tau arrow.r tau$ for any $tau$. In System F: $"Bool" = forall alpha . alpha arrow.r alpha arrow.r alpha$.

=== Pairs

$ "pair" &= lambda x . lambda y . lambda f . f space x space y \
"fst"   &= lambda p . p space (lambda x . lambda y . x) \
"snd"   &= lambda p . p space (lambda x . lambda y . y) $

Verify: $"fst" ("pair" space a space b) arrow.r^* a$.

These *Church encodings* show that products and sums are *derivable* in pure $lambda$, given enough type-theoretic power (System F suffices). $lambda^arrow.r$ proper cannot encode them; the universal property of pairs requires polymorphism.

== Equational Theory

The $beta eta$-equational theory of $lambda^arrow.r$ is the smallest congruence containing:
+ $(beta)$  $(lambda x : tau . e_1) space e_2 = [x |-> e_2] e_1$
+ $(eta)$  $lambda x : tau . (e space x) = e$ (if $x in."not" "FV"(e)$)
+ Reflexivity, symmetry, transitivity.
+ Congruence under $lambda$, application.

*Theorem.* $e_1 =_(beta eta) e_2$ is decidable for $lambda^arrow.r$.

*Proof.* By SN + confluence, every term has a unique $beta$-normal form; further $eta$-normalize (or $eta$-expand) to obtain a canonical form. Equality of canonical forms is structural and decidable. $square$

This contrasts sharply with untyped $lambda$: $beta$-equality of arbitrary $lambda$-terms is undecidable (Scott 1963).

== Recursion-Free Programming

What can $lambda^arrow.r$ compute? The cal(C) of *higher-type primitive recursive* functions, properly contained in the primitive recursive functions on naturals. The Ackermann function is not expressible in pure $lambda^arrow.r$ (no recursion), nor in $lambda^arrow.r$ + finite-type primitive recursion at first-order types — but it *"is"* expressible in System T (Gödel 1958) using primitive recursion at higher type $("Nat" arrow.r "Nat") arrow.r ("Nat" arrow.r "Nat")$.

Pure $lambda^arrow.r$ without iterators or recursors computes only *bounded* polynomial functions; specifically, the term-complexity of normalisation can be hyperexponential (Statman 1979): there are terms of size $n$ whose normal form has size a tower of exponentials in $n$. So even SN, decidable type-checking systems can be computationally explosive.

*Statman's Theorem (1979).* The decision problem "is $e_1 =_(beta) e_2$?" for $lambda^arrow.r$ is *non-elementary*: it lies outside the elementary hierarchy.

This shows: SN does *not* imply efficient normalisation. The normal form exists and is unique, but finding it may take time not bounded by any elementary function.

== Beyond $lambda^arrow.r$: A Roadmap

What does $lambda^arrow.r$ lack?

*Polymorphism.* The identity $lambda x . x$ has type $iota arrow.r iota$ for each base $iota$, but $lambda^arrow.r$ cannot internalise the quantification. Adding $forall alpha$ yields *System F*, giving Curry–Howard with second-order intuitionistic logic. See _System F and Parametricity_.

*Type-level computation.* Types in $lambda^arrow.r$ are inert; we cannot compute on them. Adding type-level $lambda$ yields *System $F_omega$*; adding *dependent types* (types indexed by terms) yields the *Edinburgh Logical Framework* and ultimately *Martin-Löf Type Theory* and the *Calculus of Constructions*. See _Dependent Types_.

*General recursion.* $lambda^arrow.r$ is sub-Turing. Adding $"fix"$ recovers Turing power at the cost of SN and logical consistency.

*Effects.* Pure $lambda^arrow.r$ cannot model state, exceptions, I/O. Effect systems, monads, and algebraic effects extend the discipline to track effects in types.

*Subtyping.* $lambda^arrow.r_"sub"$ adds a subtype relation $tau_1 <: tau_2$ (Cardelli 1984). The crucial *contravariant function rule*: $sigma_1 arrow.r tau_1 <: sigma_2 arrow.r tau_2$ <==> $sigma_2 <: sigma_1$ and $tau_1 <: tau_2$.

Each extension is conservative over $lambda^arrow.r$: every pure $lambda^arrow.r$ derivation is still derivable in the extended system. The art of type-system design is to add power while preserving (or carefully relaxing) the metatheorems we have just proved: confluence, SR, progress, SN, and decidability of type checking.

== The Statman Hierarchy

A subtle question: how *expressive* is $lambda^arrow.r$ at low type orders?

Define *type order*:
- $"ord"(iota) = 0$ for base $iota$.
- $"ord"(tau_1 arrow.r tau_2) = max(1 + "ord"(tau_1), "ord"(tau_2))$.

So $"Int"$ is order $0$; $"Int" arrow.r "Int"$ is order $1$; $("Int" arrow.r "Int") arrow.r "Int"$ is order $2$.

*Statman's $1$-section Theorem (1979).* The number-theoretic functions definable in $lambda^arrow.r$ at order $<= 2$ are exactly the *polynomially-bounded* functions; at order $<= 3$, the *Kalmar elementary* functions; at unbounded order, the higher-type primitive recursive functions.

So there is a strict hierarchy by type order, a phenomenon absent in untyped or general-recursion settings.

== Schwichtenberg's Theorem

*Schwichtenberg (1976).* The functions of type $"Nat" arrow.r "Nat"$ definable in *Gödel's System T* are exactly the *provably total functions of first-order Peano Arithmetic*, equivalently the functions whose totality is provable using transfinite induction up to $epsilon_0$.

This places System T (and hence $lambda^arrow.r$ + primitive recursion) in correspondence with PA, just as $lambda^arrow.r$ alone corresponds to $"IPC"^supset$, System F corresponds to second-order arithmetic, and the Calculus of Constructions corresponds to higher-order intuitionistic logic plus inductive types.

The pattern: each typed $lambda$-calculus is the *computational content* of a logical theory; the strength of the calculus is exactly the proof-theoretic strength of the logic.

== Cut Elimination and Proof Theory

Gentzen (1934/35) introduced *sequent calculus* and proved his celebrated *Hauptsatz*: every proof can be transformed into a *cut-free* proof.

Under the Curry–Howard correspondence:
- *Cut* in sequent calculus = *function application* in $lambda^arrow.r$.
- *Cut elimination* = $beta$-reduction to normal form.
- *Hauptsatz* (cut elimination terminates) = *strong normalization*.

Gentzen's original proof of cut elimination was syntactic and used induction on cut-rank.
Tait's reducibility argument is the semantic analog: instead of reducing the proof directly, we interpret each proposition by a *reducibility predicate* and show every proof inhabits its predicate.

This shift from syntactic proof transformation to semantic interpretation is the methodological bridge from proof theory into modern type theory.
The same reducibility technique scales to System F (with candidates), to MLTT (with logical relations and Kripke worlds), and to higher type theories.

== Cartmell's Categories with Families

For dependent types we will need *categories with families* (CwFs; Cartmell 1986, Dybjer 1996).
For $lambda^arrow.r$ alone, the simpler structure of a *CCC* suffices.
But $lambda^arrow.r$ already exhibits the *substitution-equals-pullback* pattern: substitution in the term is composition in the category; reindexing along a substitution is pullback of the context.

This perspective unifies the syntactic and semantic accounts and prepares the ground for dependent types, where substitution and type formation interact nontrivially.

== Computational Adequacy

A model $cal(M)$ of $lambda^arrow.r$ is *computationally adequate* if:
- $emptyset tack.r e : "Bool"$ and $bracket.l.double e bracket.r.double = "true"$ in $cal(M)$ => $e arrow.r^* "true"$ (syntactically).

The set-theoretic model is adequate for $lambda^arrow.r$.
For $lambda^arrow.r + "fix"$, adequacy requires the *Scott model* (cpos and continuous functions, with $bot$ for divergence): Plotkin (1977) proved the seminal adequacy theorem for PCF.

Adequacy is the *bridge* between operational and denotational semantics: it tells us that the denotational interpretation captures observational behaviour at base type.

== Comparison with Other Typed Calculi

#table(
  columns: (auto, auto, auto, auto, auto),
  [*System*], [*Polymorphism*], [*Recursion*], [*SN*], [*Logic*],
  [$lambda^arrow.r$], [None], [None], [Yes (Tait)], [$"IPC"^supset$],
  [System T], [None], [Primitive], [Yes (Tait)], [PA (provable)],
  [System F], [Universal], [None], [Yes (Girard)], [Second-order IPC],
  [System $F_omega$], [Higher-kinded], [None], [Yes], [HOL fragment],
  [PCF], [None], [General ($"fix"$)], [No], [Inconsistent (universal $bot$)],
  [MLTT], [Dependent], [Structural], [Yes], [Predicative HOL + W],
  [CIC], [Dependent + impredicative Prop], [Structural], [Yes], [Impredicative HOL + Ind],
)

The pattern: each row strengthens one axis (polymorphism, recursion, type-level computation) and trades off another (SN, decidability, logical consistency).
$lambda^arrow.r$ is the *origin* of this table; every column tells us something we get by adding (or removing) a feature.

== Exercises (for the dedicated reader)

+ Prove that $beta$-reduction is *not* confluent in the presence of *unrestricted $eta$* without the side condition $x in."not" "FV"(e)$; give the standard counterexample.
+ Show that the term $omega = lambda x . x space x$ is *not* typable in $lambda^arrow.r$. (Hint: T-APP would demand $x : tau arrow.r sigma$ and $x : tau$ simultaneously.)
+ Verify that $S K K =_(beta) I$ in detail. Then show $S K K : forall alpha . alpha arrow.r alpha$ in System F.
+ Translate the proof of $((P supset Q) supset P) supset (P supset Q) supset Q$ (a simple intuitionistic tautology) into a $lambda^arrow.r$ term.
+ Construct a Coq/Agda term proving the symmetric pairing law: $forall A B . A times B arrow.r B times A$.
+ Show that there are well-typed $lambda^arrow.r$ terms whose normal form is hyperexponentially larger than the term itself. (Hint: iterated doubling using Church numerals at higher type.)
+ Prove that the Curry-style version of $lambda^arrow.r$ has *type inference* in time $O(n alpha(n))$ via union-find unification.
+ Explore: define the *call-by-need* (lazy) reduction strategy and prove it is observationally equivalent to CBN on closed base-type terms.

== Summary

The simply-typed lambda calculus is small, sharp, and complete-to-itself. The two-page syntax supports a full equational theory ($beta eta$), a confluent reduction, decidable type checking, principal-type inference, strong normalization with a beautiful semantic proof, a precise correspondence to a fragment of constructive logic, and termination/totality by construction. Every extension we encounter — polymorphism (System F), dependent types (MLTT, CIC), effects, subtyping — is built by adding type formers and corresponding term formers to $lambda^arrow.r$, then re-proving (or losing) confluence, SN, and decidability of type checking. $lambda^arrow.r$ is the kernel; the rest of the tower is decoration.

*Slogan summary.*
- *Confluence:* one term, one normal form (up to $alpha$).
- *Preservation:* types survive reduction.
- *Progress:* well-typed terms are never stuck.
- *Strong normalization:* every reduction sequence terminates.
- *Curry–Howard:* types are propositions; terms are proofs; reduction is proof normalisation.
- *Consistency:* the inhabitedness of $bot$ is decidable; it is uninhabited.
- *Decidability:* type checking is decidable in $O(n)$; type inference (Curry) is decidable in $O(n alpha(n))$.

The four landmark theorems (Church–Rosser (confluence), Subject Reduction (preservation), Progress, and Strong Normalisation) together with their proofs (parallel reduction, structural induction, canonical forms, and Tait reducibility) form the *standard playbook* for every typed calculus.
Master them here, and the proofs for System F, $F_omega$, MLTT, CIC, and beyond are variations on these themes, with sharper tools (reducibility candidates, logical relations indexed by candidate assignments) but the same melody.

Read this chapter as a *technical exercise* in the methodology of typed programming language theory.
Every theorem we proved (confluence, SR, progress, SN) will recur in the chapters on System F and dependent types, usually in a stronger and harder form, but with the same skeleton of argument.

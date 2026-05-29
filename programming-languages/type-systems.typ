= Type Systems

A type system is a syntactic method for proving the absence of certain program behaviors. That definition from Pierce is deliberately narrow: types are a *proof*, not a test. They are checked statically, before any execution, and their guarantees hold for all possible inputs.

Why bother? Four concrete payoffs:

1. *Error detection at compile time.* A null dereference, an array out of bounds, a missing case in a union — types catch entire classes of runtime errors for free.
2. *Documentation.* A function signature `auto serialize(const Config& cfg) -> std::string` tells you more than a comment. It is machine-checked documentation that cannot drift out of sync.
3. *Optimization.* Knowing a value is a 32-bit integer lets the compiler choose a register and a machine instruction without a runtime tag check. Devirtualization, escape analysis, and loop vectorization all rely on type information.
4. *Safer refactoring.* Change a type; let the type checker enumerate everywhere the change must propagate. In large codebases this is the difference between a 30-minute and a 3-day refactor.

_See also: Program Semantics for the operational semantics that type soundness theorems are stated over._

== Simply-Typed Lambda Calculus

The simply-typed lambda calculus ($lambda arrow.r$) is the minimal typed language: variables, function abstraction, and function application, with a type assigned to every term.

*Types* ($tau$):
$ tau ::= "Bool" | "Int" | tau_1 arrow.r tau_2 $

*Terms* ($e$):
$ e ::= x | lambda x : tau . e | e_1 space e_2 | n | "true" | "false" $

*Typing rules* (reading $Gamma tack.r e : tau$ as "under context $Gamma$, term $e$ has type $tau$"):

```text
  x : tau in Gamma
  ----------------  (T-VAR)
  Gamma |- x : tau

  Gamma, x:tau_1 |- e : tau_2
  --------------------------------  (T-ABS)
  Gamma |- (lam x:tau_1. e) : tau_1 -> tau_2

  Gamma |- e1 : tau_1 -> tau_2    Gamma |- e2 : tau_1
  -----------------------------------------------------  (T-APP)
  Gamma |- e1 e2 : tau_2
```

T-ABS says: to type a function, extend the context with the parameter's type and type the body. T-APP says: if you have a function from $tau_1$ to $tau_2$ and an argument of type $tau_1$, the application has type $tau_2$. These three rules are the entire type system for $lambda arrow.r$.

== Type Soundness: Progress and Preservation

Type soundness is formalized as two theorems about the relationship between the type system and the operational semantics:

*Progress:* A well-typed term is either a value or can take a reduction step.
$ forall e, tau. (dot tack.r e : tau) => ("isValue"(e) or exists e'. e arrow.r e') $

*Preservation:* If a well-typed term takes a step, the result has the same type.
$ forall e, e', tau. (dot tack.r e : tau and e arrow.r e') => (dot tack.r e' : tau) $

Together, these guarantee that a well-typed program never gets *stuck* — it never reaches a state that is not a value but also cannot reduce. "Stuck" corresponds to runtime type errors (applying a non-function, pattern-matching on the wrong constructor, etc.). Type soundness says: if the type checker accepts your program, those states are unreachable.

The proofs proceed by induction on typing derivations. The critical supporting lemma is spelled out below, followed by the T-APP cases of both theorems — the cases cited in the computability and semantics chapters.

_See also: Turing Machines and Computability for Rice's theorem and undecidability of type inference for unrestricted systems. Program Semantics for the operational semantics ($arrow.r$, $arrow.r^*$) used in the statement of these theorems._

=== Substitution Lemma

*Lemma (Substitution).* If $Gamma, x : S tack.r t : T$ and $tack.r s : S$, then $Gamma tack.r [x |-> s] t : T$.

*Proof* by induction on the derivation of $Gamma, x : S tack.r t : T$.

*Case T-VAR:* $t = y$ for some variable $y$. Two sub-cases.

- $y = x$: Then $T = S$. The substitution $[x |-> s] x = s$, and $tack.r s : S$ is given, so $Gamma tack.r s : S$. (Weakening extends the closed typing to $Gamma$.)
- $y eq.not x$: Then $y : T in Gamma$ (the binding comes from $Gamma$, not the $x : S$ extension). The substitution $[x |-> s] y = y$, and $y : T in Gamma$, so $Gamma tack.r y : T$ by T-VAR.

*Case T-ABS:* $t = lambda y : T_1 . t_2$ and $T = T_1 arrow.r T_2$. The derivation has $Gamma, x : S, y : T_1 tack.r t_2 : T_2$. Alpha-rename $y$ to a fresh variable so $y eq.not x$ and $y in.not "dom"(Gamma)$, avoiding capture of $y$ by the substitution. Apply the induction hypothesis to obtain $Gamma, y : T_1 tack.r [x |-> s] t_2 : T_2$. T-ABS then gives $Gamma tack.r lambda y : T_1 . [x |-> s] t_2 : T_1 arrow.r T_2$, which equals $Gamma tack.r [x |-> s](lambda y : T_1 . t_2) : T$.

*Case T-APP:* $t = t_1 space t_2$. The derivation has $Gamma, x : S tack.r t_1 : T_2 arrow.r T$ and $Gamma, x : S tack.r t_2 : T_2$. Apply the IH to each, obtaining $Gamma tack.r [x |-> s] t_1 : T_2 arrow.r T$ and $Gamma tack.r [x |-> s] t_2 : T_2$. T-APP gives $Gamma tack.r [x |-> s] t_1 space [x |-> s] t_2 : T$, which is $Gamma tack.r [x |-> s](t_1 space t_2) : T$. $square$

=== Progress: T-APP Case

*Theorem (Progress).* If $dot tack.r t : T$ then either $t$ is a value or $exists t'. t arrow.r t'$.

*Proof* by induction on the typing derivation. The central case:

*Case T-APP:* $t = t_1 space t_2$ with $dot tack.r t_1 : T_2 arrow.r T$ and $dot tack.r t_2 : T_2$.

By the induction hypothesis on $t_1$: either $t_1$ is a value or $t_1 arrow.r t_1'$. If $t_1 arrow.r t_1'$ then $t_1 space t_2 arrow.r t_1' space t_2$ by rule E-App1 and we are done.

If $t_1$ is a value, it has type $T_2 arrow.r T$ in the empty context. By the *canonical forms lemma* (a value of arrow type in $lambda arrow.r$ must be a lambda abstraction), $t_1 = lambda x : T_2 . t_b$ for some body $t_b$.

Now apply the induction hypothesis on $t_2$: either $t_2 arrow.r t_2'$ or $t_2$ is a value $v_2$. If $t_2 arrow.r t_2'$ then $t_1 space t_2 arrow.r t_1 space t_2'$ by E-App2. If $t_2$ is a value $v_2$, then

$ (lambda x : T_2 . t_b) space v_2 arrow.r [x |-> v_2] t_b $

by rule E-AppAbs. In every sub-case the term either is a value or steps, completing the T-APP case. $square$

=== Preservation: T-APP / E-AppAbs Case

*Theorem (Preservation).* If $dot tack.r t : T$ and $t arrow.r t'$, then $dot tack.r t' : T$.

*Proof* by induction on the typing derivation and case analysis on the reduction step. The case that requires the Substitution Lemma:

*Case T-APP / E-AppAbs:* $t = (lambda x : T_2 . t_b) space v_2$ steps to $t' = [x |-> v_2] t_b$.

By inversion on T-APP: $dot tack.r lambda x : T_2 . t_b : T_2 arrow.r T$ and $dot tack.r v_2 : T_2$. By inversion on T-ABS applied to the first: $x : T_2 tack.r t_b : T$. We now have exactly the hypotheses of the Substitution Lemma with $Gamma = dot$, $S = T_2$, $s = v_2$: it follows that $dot tack.r [x |-> v_2] t_b : T$, i.e., $dot tack.r t' : T$. $square$

The remaining cases (E-App1, E-App2) use the induction hypothesis directly without invoking the Substitution Lemma, because the step happens inside a subterm whose type is tracked by the same T-APP premise.

== Hindley-Milner Type Inference

In $lambda arrow.r$, every function must be annotated: `lam x:Int. x+1`. Hindley-Milner (HM) removes the annotations: the type checker *infers* them by constraint solving.

*Algorithm W* (Damas-Milner 1982) is the standard presentation:

1. *Generate type variables* ($alpha, beta, ...$) for each subterm whose type is not yet known.
2. *Emit constraints* by walking the syntax tree. T-APP emits: the type of `e1` must equal the type of `e2 -> fresh_alpha`.
3. *Unify* the constraints via Robinson's unification algorithm. Unification finds the most general substitution $S$ such that all constraints $tau_1 = tau_2$ are satisfied under $S$.
4. *Apply* the substitution to recover the inferred types.

Unification terminates because each unification step either eliminates a variable (substituting it with a term) or reduces the size of the constraint. The total work is bounded by $O(n log n)$ in the nearly-linear union-find formulation (Huet 1976).

*Principal types:* HM always infers the *most general* type. The expression `lam f. lam x. f x` gets type `(alpha -> beta) -> alpha -> beta`, not a specific instantiation. Every use of this expression can specialize the type variables to concrete types without any extra annotation.

*Let-polymorphism:* HM generalizes type variables at `let` bindings:

```text
let id = lam x. x in
    (id 42, id true)
```

Here `id` is generalized to the polymorphic type `forall alpha. alpha -> alpha`. At the first use it is instantiated to `Int -> Int`; at the second to `Bool -> Bool`. This is the mechanism behind parametric polymorphism in OCaml and Haskell.

*HM vs System F:* System F (Girard, Reynolds) allows polymorphism inside functions, not just at let-bindings. HM is a restricted fragment of System F where type inference is decidable ($O(n log n)$). Full System F type inference is undecidable (Wells 1994). Haskell's `RankNTypes` extension steps into System F territory and requires explicit type annotations at higher-rank points.

=== Robinson Unification Algorithm

Unification finds the most general substitution $theta$ such that $theta(s) = theta(t)$ for two type terms $s$ and $t$. The four cases:

```text
unify(s, t):
  if s == t:
    return {}                          // identical, no work

  if s is a type variable alpha:
    if alpha occurs in t: fail         // occurs check -- prevents cyclic types
    return { alpha |-> t }

  if t is a type variable alpha:
    if alpha occurs in s: fail
    return { alpha |-> s }

  if s = F(s1, ..., sn) and t = F(t1, ..., tn):
    // same constructor, unify component-wise
    theta = {}
    for i in 1..n:
      theta_i = unify(theta(s_i), theta(t_i))
      theta   = compose(theta_i, theta)
    return theta

  fail                                 // mismatched constructors
```

The *occurs check* (line 4) prevents unification of $alpha$ with $alpha arrow.r "Int"$, which would require an infinite type. Most Haskell compilers skip the occurs check for performance (enabling equirecursive types); OCaml enforces it (isorecursive).

Termination: every recursive call either eliminates a variable (binding it in $theta$) or reduces the total size of the remaining constraint set. The union-find formulation (Huet 1976) achieves near-linear $O(n alpha(n))$ time, where $alpha$ is the inverse Ackermann function.

=== Worked HM Inference Trace

Program:
```text
let f = lam x. x in (f 1, f true)
```

*Step 1 — Assign fresh type variables.*

- $x$ gets type $alpha$
- Body of lambda ($x$): type $alpha$, so $f : alpha arrow.r alpha$
- Before generalization, $f$ has monotype $alpha_0 arrow.r alpha_0$ (a specific fresh variable)

*Step 2 — Generalize at the let-binding.*

$f$ is not in the body's free type variables (the environment is empty), so generalize: $f : forall alpha. alpha arrow.r alpha$.

*Step 3 — Generate constraints for the body $(f space 1, space f space "true")$.*

Instantiate $f$ twice with fresh variables $alpha_1$ and $alpha_2$:
- First use: $f : alpha_1 arrow.r alpha_1$, applied to $1 : "Int"$. Constraint: $alpha_1 = "Int"$.
- Second use: $f : alpha_2 arrow.r alpha_2$, applied to $"true" : "Bool"$. Constraint: $alpha_2 = "Bool"$.

#table(
  columns: (auto, auto, auto),
  [*Site*], [*Constraint*], [*After unification*],
  [$f space 1$],    [$alpha_1 = "Int"$],  [$alpha_1 |-> "Int"$],
  [$f space "true"$], [$alpha_2 = "Bool"$], [$alpha_2 |-> "Bool"$],
)

*Step 4 — Apply substitution.*

- $f space 1 : "Int"$
- $f space "true" : "Bool"$
- Pair type: $"Int" times "Bool"$

The two uses of $f$ receive different instantiations because $forall alpha$ was introduced at the let. If $f$ had been inlined (no let), both uses would share the same $alpha$ and the constraint $alpha = "Int"$ and $alpha = "Bool"$ would clash — an error. This is why let-polymorphism is essential and why ML's value restriction governs when generalization is sound.

== Polymorphism Flavors

*Parametric polymorphism* (generics): a single code definition operates uniformly over all types. `std::vector<T>` in C++, `forall alpha. List alpha` in Haskell. The code cannot inspect the type at runtime (unless erased, as in Java generics, or monomorphized, as in C++ templates and Rust).

*Ad-hoc polymorphism* (overloading, typeclasses, traits): different code runs depending on the type. `+` means integer addition for `int` and floating-point addition for `double`. Haskell typeclasses and Rust traits are disciplined ad-hoc polymorphism: the compiler selects the implementation via a dictionary (vtable) or monomorphization.

*Subtype polymorphism*: if `B` is a subtype of `A`, a value of type `B` can be used where `A` is expected. The basis for object-oriented polymorphism. Variance (covariance, contravariance, invariance) governs how subtyping interacts with type constructors: a `ReadOnly<Cat>` is a subtype of `ReadOnly<Animal>` (covariant), but `WriteOnly<Animal>` is a subtype of `WriteOnly<Cat>` (contravariant), and `Array<Cat>` is neither a subtype nor supertype of `Array<Animal>` in a sound system (invariant).

*Row polymorphism* (record polymorphism): a function can accept any record that has at least the fields it needs. `fun f(r: {name: String, ...rest}) -> ...`. Used in OCaml's object system, PureScript, and structural typing in TypeScript.

== Curry-Howard Isomorphism

The Curry-Howard correspondence observes that types and propositions are the same thing, and programs and proofs are the same thing, under the following dictionary:

#table(
  columns: (auto, auto, auto),
  [*Type/Logic*], [*Type system*], [*Logic*],
  [Implication], [$tau_1 arrow.r tau_2$], [$P => Q$],
  [Conjunction], [$tau_1 times tau_2$ (pair)], [$P and Q$],
  [Disjunction], [$tau_1 + tau_2$ (sum/Either)], [$P or Q$],
  [Truth], [Unit type `()`], [True],
  [Falsehood], [Empty type (uninhabited)], [False],
  [Universal], [$forall alpha. tau(alpha)$], [$forall x. P(x)$],
  [Existential], [$exists alpha. tau(alpha)$], [$exists x. P(x)$],
)

A function of type `A -> B` is a proof that `A` implies `B`: given evidence of `A` (an argument), it constructs evidence of `B` (the return value). A pair of type `A * B` is a proof of `A and B`. A value of `Either A B` is a proof of `A or B`.

This correspondence is not a metaphor — it is a precise bijection. In Coq and Agda, you write proofs as programs and the type checker verifies them. The `Prop` sort in Coq is literally the type of propositions, and a term of type `P : Prop` is a proof of `P`.

== Dependent Types and Totality

Dependent types allow types to *depend on values*: `Vec n Int` is the type of integer vectors of length exactly `n`, where `n` is a runtime value. The type `Fin n` contains exactly the natural numbers less than `n` — array indexing with `Fin n` is provably safe.

Languages with dependent types: Coq, Agda, Idris, Lean. They serve dual roles: proof assistants (writing mathematical proofs) and programming languages (writing certified programs).

The trade: to keep type-checking decidable, these languages require all programs to *terminate*. The termination checker verifies that every recursive call is on a structurally smaller argument. This rules out general recursion and — by Rice's theorem — means you cannot write a self-interpreter for the full language within the language.

This is not a bug, it is a feature. A Coq proof that `sort` is correct is a machine-checked mathematical proof. The price is that you cannot write non-terminating programs. For verified systems software, theorem proving, and protocol specification, this trade is excellent.

== Effect Systems

A type system can track not just *what type* a computation produces but *what effects* it may have: does it read mutable state? Write to disk? Throw an exception? Perform I/O?

*Koka* (Leijen 2014) and *Eff* (Bauer, Pretnar 2015) make effects first-class citizens of the type system. A function type looks like `Int -> <io, exn> Bool`: it takes an integer, may perform I/O, may throw an exception, and returns a bool. Effect polymorphism lets you write higher-order functions that are generic over effects.

Effect systems in mainstream languages are weaker but recognizable: Java's checked exceptions are a rudimentary effect system (tracking which exceptions a method may throw). Haskell's `IO` monad is a type-level effect: any function that returns `IO a` may perform arbitrary I/O; pure functions returning `a` cannot.

Rust's ownership and borrow checker is a linear/affine type system tracking aliasing and lifetime — a form of effect tracking for memory safety.

== Gradual Typing

Gradual typing (Siek, Taha 2006) combines static and dynamic typing in one language. A value of type `Dynamic` (written `?` in some systems) can hold any runtime value; casts to concrete types are checked dynamically. Statically typed code and dynamically typed code interoperate via *casts* at the boundary.

TypeScript is the most widely deployed gradual type system: JavaScript is valid TypeScript (fully dynamic), and you can add type annotations incrementally. Where annotations are present, the type checker enforces them statically; where they are absent, the behavior is dynamic. TypeScript's type system is deliberately unsound (structural subtyping of mutable arrays is allowed) in exchange for usability.

Python's type hints (PEP 484, mypy) and PHP's Hack follow the same pattern. The key insight: gradual typing is not a consolation prize for dynamic languages. It is the correct trade-off when you need to start with speed-to-market and gradually increase assurance.

== Practical Type System Comparison

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Language*], [*Inference*], [*Polymorphism*], [*Sound?*], [*Totality*],
  [C],       [None],       [None (macros only)],        [No (UB)],     [No],
  [C++],     [Partial (`auto`, templates)], [Parametric (templates)], [No (UB)], [No],
  [Java],    [Local (`var`)], [Erased generics + subtype], [Mostly], [No],
  [Rust],    [Full HM-like], [Traits + lifetimes],       [Yes (safe fragment)], [No],
  [Haskell], [Full HM + extensions], [Typeclasses + System F], [Yes], [No (Turing-complete)],
  [OCaml],   [Full HM],    [Modules + first-class modules], [Yes], [No],
  [Coq],     [Full + tacticals], [Dependent types],     [Yes],         [Yes (by construction)],
)

Rust is a notable data point: it is the first mainstream systems language with a sound type system in the safe fragment. The unsound operations (raw pointer arithmetic, `unsafe` blocks) are explicitly marked, making unsafety auditable. This is a significant engineering achievement given the language's performance requirements.

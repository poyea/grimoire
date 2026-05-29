= Gradual and Hybrid Type Systems

Static and dynamic typing are usually presented as a binary choice: a language commits to one discipline and pays the corresponding price. Static languages catch errors early but reject programs the programmer knows are correct, demand annotations on prototypes, and complicate exploratory programming. Dynamic languages accept anything until the moment of execution, at which point an unrelated error message materializes deep inside a stack of library code. The discipline of *gradual typing* (Siek–Taha 2006) refuses the dichotomy: it lets a single program contain both statically typed and dynamically typed fragments, with the two communicating through *casts* whose run-time checks are derived mechanically from the type structure. *Hybrid type checking* (Flanagan 2006) generalises the idea to refinement types: the checker proves what it can statically and inserts residual checks where decidability fails.

*See also:* _Type Systems_, _Effects and Handlers_

The two ideas are descendants of the older program of *soft typing* (Cartwright–Fagan 1991), which built a type-inference procedure that *never* rejects a program: type errors are recorded as warnings together with set-based abstract values, and the run-time inserts checks where soundness cannot be proved. Gradual typing inverts the polarity by making the *programmer*, not the inferencer, choose where to add types; hybrid typing keeps the inferencer in charge but allows it to admit defeat by emitting a check.

== The Dynamic Type and Consistency

The central syntactic move is the introduction of an explicit *dynamic type*, traditionally written "?" or $star.op$, into the grammar of types:

$ tau ::= "Bool" | "Int" | "?" | tau_1 arrow.r tau_2 $

The grammar otherwise looks like the simply-typed lambda calculus. The novelty is that the type checker treats "?" specially in *every* place that an exact match would have been required.

The mechanism that does this work is the *consistency relation* $tau_1 tilde tau_2$, defined as the least relation such that

```text
  "?" ~ tau               tau ~ "?"
  ----------              ----------
                          (reflexive on "?")

  Bool ~ Bool             Int ~ Int

  tau1 ~ tau1'    tau2 ~ tau2'
  -----------------------------
  tau1 -> tau2 ~ tau1' -> tau2'
```

Consistency is reflexive ($tau tilde tau$), symmetric ($tau_1 tilde tau_2 => tau_2 tilde tau_1$), and a congruence with respect to type constructors — but it is *not* transitive. The standard counterexample is $"Int" tilde "?"$ and $"?" tilde "Bool"$ but $"Int" tilde."not" "Bool"$. Transitivity would collapse the whole hierarchy and make every two types interchangeable.

Gradual typing replaces the equality $tau_1 = tau_2$ that appears in the premise of the static T-APP rule with consistency:

```text
  Gamma |- e1 : tau1 -> tau2     Gamma |- e2 : tau3     tau3 ~ tau1
  ------------------------------------------------------------------ (GT-APP)
  Gamma |- e1 e2 : tau2
```

There is also a rule that lets an application target a dynamically typed function:

```text
  Gamma |- e1 : "?"     Gamma |- e2 : tau
  ----------------------------------------- (GT-APP-DYN)
  Gamma |- e1 e2 : "?"
```

This pair of rules captures the working programmer's intuition. A static function applied to a dynamic argument should typecheck (the argument might at run time be the right thing). A dynamic function applied to anything should typecheck (we just trust the call). When two static parts meet, the ordinary rule fires and the ordinary error reports a mismatch.

A subtle consequence: the consistency rule for arrows is *covariant on both sides*, not contravariant on the domain. This is correct because consistency is symmetric — it is not subtyping. Gradual *subtyping* (Siek–Taha 2007) layers ordinary contravariance on top, yielding a relation $tau_1 lt.tilde tau_2$ defined as the composition $tau_1 tilde tau_1' lt.eq tau_2$ for some intermediate $tau_1'$.

== Static Becomes Dynamic via Casts

The typing relation describes what programs are *accepted*. To give those programs meaning, the surface language $lambda_(arrow.r ?)$ is elaborated to an internal *cast calculus* $lambda_C$ in which every implicit consistency use becomes an explicit cast term $angle.l tau_2 arrow.l tau_1 angle.r e$. The elaboration is:

```text
  Gamma |- e1 : tau1 -> tau2     Gamma |- e2 : tau3     tau3 ~ tau1
  ------------------------------------------------------------------
  Gamma |- e1 (<tau1 <- tau3> e2) : tau2
```

When $tau_1 = tau_3$ the cast is the identity; when one side is "?" the cast becomes an *injection* (static-to-dynamic) or a *projection* (dynamic-to-static); when both sides are higher-order it becomes a *wrapper* that defers the check to argument and result positions.

The operational semantics of casts is the heart of the system. Injection is simple — tag a value with its static type and inject it into a universal sum:

$ angle.l "?" arrow.l "Int" angle.r 7 arrow.r "Inj"_"Int" 7 $

Projection inspects the tag and either succeeds or fails:

$ angle.l "Int" arrow.l "?" angle.r ("Inj"_"Int" 7) arrow.r 7 $

$ angle.l "Bool" arrow.l "?" angle.r ("Inj"_"Int" 7) arrow.r "blame" $

For function values the cast cannot be discharged eagerly — we do not know all future arguments. Instead it factors:

$ angle.l tau_1' arrow.r tau_2' arrow.l tau_1 arrow.r tau_2 angle.r v arrow.r lambda x : tau_1' . angle.l tau_2' arrow.l tau_2 angle.r (v (angle.l tau_1 arrow.l tau_1' angle.r x)) $

The cast is split into a *contravariant* cast on the argument and a *covariant* cast on the result. This is the same wrapping pattern that contract systems for higher-order functions had been using since (Findler–Felleisen 2002).

== Blame and the Blame Theorem

When a projection cast fails, *who is to blame*? The answer is the foundational contribution of (Findler–Felleisen 2002): each cast carries a *blame label* $ell$ identifying the syntactic site that imposed the cast. On failure the run time raises $"blame" space ell$ rather than an opaque "type error".

For higher-order casts the labels must flip on the contravariant side. The wrapping rule, with labels:

$ (angle.l tau_1' arrow.r tau_2' arrow.l^ell tau_1 arrow.r tau_2 angle.r v) space w arrow.r angle.l tau_2' arrow.l^ell tau_2 angle.r (v (angle.l tau_1 arrow.l^(macron(ell)) tau_1' angle.r w)) $

where $macron(ell)$ denotes the *negation* (complementary blame) of $ell$. Negative blame falls on the *caller* when an argument fails its precondition; positive blame falls on the *callee* when the result fails its postcondition.

*Theorem (Blame, Wadler–Findler 2009).* In a cast calculus annotated with *positive* and *negative* labels, the "more typed" side of a failing cast is never blamed. Formally, if a closed program $e$ in $lambda_C$ reduces to $"blame" space ell^+$ (positive blame at $ell$), then the type that originated the cast labelled $ell$ is *not* a refinement of the actually-flowing value's type — the dynamically typed side is the one that violated the contract.

*Proof sketch.* The proof proceeds by a logical relation indexed by *subtyping* (more precisely, by the *naive subtyping* $lt.eq$ that treats "?" as bottom on the positive side and as top on the negative side). One shows: if $tau_1 lt.eq tau_2$ then a cast $angle.l tau_2 arrow.l^ell tau_1 angle.r$ never blames $ell^+$, and dually for $ell^-$. The arrow case is the interesting one: contravariance of the function space requires flipping the polarity of the blame label, which is exactly what the operational rule above does. Closure under reduction gives the theorem. $square$

The blame theorem makes "well-typed programs cannot be blamed" a precise statement: the *static* fragment of a partially-typed program is in the position of a server that can trust its inputs, modulo the assumptions it has stated. The dynamic fragment is in the position of a client that must obey those assumptions or be blamed for breaking them. This is the precise sense in which gradual typing has *teeth*.

== The Gradual Guarantee

A second metatheorem, often called the *gradual guarantee* (Siek–Vitousek–Cimini–Boyland 2015), formalises the programmer's expectation that adding or removing type annotations is a *behaviour-preserving* refactoring. Define the *precision* relation $e_1 ⊑ e_2$ on terms: $e_2$ is at least as precise as $e_1$, in the sense that every "?" in $e_1$ may be replaced by a more specific type in $e_2$, but never the reverse. Precision is congruent on terms and consistent with the typing.

*Theorem (Gradual Guarantee).* Let $e_1 ⊑ e_2$ be closed and well-typed.

1. *(Static.)* If $e_1$ type-checks, then $e_2$ type-checks.
2. *(Dynamic.)* If $e_1 arrow.r^* v_1$ then either $e_2 arrow.r^* v_2$ with $v_1 ⊑ v_2$, or $e_2 arrow.r^* "blame" space ell$.
3. *(Dynamic, reverse.)* If $e_2 arrow.r^* v_2$ then $e_1 arrow.r^* v_1$ with $v_1 ⊑ v_2$.

Equivalently: refining annotations never introduces *new* successful behaviours, only new *failures*; relaxing annotations never *eliminates* successful behaviours, but may eliminate failures. A program whose annotated version runs successfully will also run successfully when the annotations are erased (modulo "?" wherever they appeared).

The gradual guarantee is *not* automatic. Several proposed gradual systems failed it in subtle ways. Gradual typing for *mutable references* requires especially careful design: a naïve approach assigns invariant types to mutable cells and produces casts that violate the guarantee. The *monotonic references* design (Siek–Vitousek–Cimini–Tobin-Hochstadt–Vitousek 2015) maintains a monotonic record of the *most precise* type ever stored in a cell, and refuses to demote, restoring the guarantee at the cost of additional metadata on the heap.

== Cast Calculi: Lazy, Eager, and Threesomes

The naive cast calculus accumulates casts pathologically. A function passed back and forth across a typed/untyped boundary acquires a new wrapper at every crossing, and the chain of wrappers grows without bound. A program of $n$ crossings can incur $O(n)$ wrapper indirection at every call to the underlying function, giving $O(n^2)$ total cost.

*Threesomes* (Siek–Wadler 2010) are the optimisation that fixes this. The observation: a sequence of casts $angle.l tau_n arrow.l tau_(n-1) angle.r dots.h angle.l tau_2 arrow.l tau_1 angle.r$ can be *summarised* by a single triple $(tau_1, tau_"meet", tau_n)$ where $tau_"meet"$ is the *meet* in the precision lattice of all intermediate types. The triple has the property that any further composition produces another triple, so casts circle.small in *constant* additional space.

Formally, threesomes form a *monoid* under composition with the empty triple as identity. The meet is the *unification* of the intermediate types modulo "?", failing (producing the bottom element) exactly when an inconsistency would have produced blame. The implementation in Typed Racket reduces space overhead from $O(n)$ to $O(1)$ per wrapper and removes the worst-case quadratic-blowup pathology.

A further refinement, the *coercion calculus* of Henglein (1994), expresses casts as a small algebra of *coercions* $c ::= "id" | "Int!" | "Int?" | c_1 ; c_2 | c_1 arrow.r c_2 | "Fail"^ell$ with rewriting rules that normalise compositions. The cast calculus, threesomes, and coercions are *equivalent* in the values they produce and the blame they assign, but differ in the implementation cost of cast composition.

== Abstracting Gradual Typing (AGT)

(Garcia–Clark–Tanter 2016) gave a *general recipe* — *abstracting gradual typing* (AGT) — for deriving a gradual type system from any static one. The recipe makes the leap from "design ad hoc" to "compute from the static system".

The recipe in four steps:

1. *Start with a static type system.* Types $tau in T$; typing judgement; subtyping or equality where relevant.
2. *Choose a concretisation function* $gamma : tilde(T) -> P(T)$, where $tilde(T)$ is the set of *gradual types* (containing "?" as a new constant). The intuition: a gradual type *denotes* a set of possible static types. The canonical choice is $gamma("?") = T$ and $gamma(tau) = {tau}$ for static $tau$. For arrows, $gamma(tau_1 arrow.r tau_2) = {tau_1' arrow.r tau_2' | tau_1' in gamma(tau_1), tau_2' in gamma(tau_2)}$.
3. *Lift static predicates and functions to gradual ones* via the Galois connection $(alpha, gamma)$ between $tilde(T)$ and $P(T)$. The abstraction $alpha : P(T) -> tilde(T)$ is the *most precise* gradual type whose concretisation contains the given set. Consistency $tilde(tau_1) tilde tilde(tau_2)$ becomes the concretisations overlap: $gamma(tilde(tau_1)) inter gamma(tilde(tau_2)) eq."not" emptyset$. Consistent subtyping becomes the lifted version of subtyping.
4. *Derive the dynamic semantics* by *evidence-based* reduction: each typing step records the *evidence* that justifies it (the most precise type to which both sides are consistent), and reductions update the evidence; failure to maintain non-empty evidence is blame.

The remarkable feature of AGT is that *the gradual guarantee is automatic*: it falls out of the soundness and optimality of the Galois connection. The technique has been instantiated for:

- Gradual subtyping (recovers Siek–Taha 2007).
- Gradual references (recovers monotonic references).
- *Gradual parametric polymorphism* (Igarashi–Sekiyama–Igarashi 2017): "?" interacts with universal quantifiers; the dynamic-sealing mechanism preserves parametricity.
- *Gradual security types* (Toro–Garcia–Tanter 2018): noninterference proved under graduality.
- *Gradual effects* (Bañados Schwerter–Garcia–Tanter 2014): effect annotations on functions become gradual.
- *Gradual dependent types* (Eremondi–Tanter–Garcia 2019; Lennon-Bertrand–Maillard–Tabareau–Tanter 2022): refinement types and dependent function types are made gradual; proofs become casts.

The AGT methodology has not entirely replaced ad hoc designs — sometimes the derived semantics is too conservative or too eager to blame — but it provides a *baseline* against which any proposed gradual system can be measured.

== Operational Semantics in Detail

We work out the cast calculus $lambda_C$ explicitly. Terms:

$ e ::= x | n | "true" | "false" | lambda x : tau . e | e_1 space e_2 | angle.l tau_2 arrow.l^ell tau_1 angle.r e | "blame" space ell $

Values:

$ v ::= n | "true" | "false" | lambda x : tau . e | angle.l tau_1' arrow.r tau_2' arrow.l^ell tau_1 arrow.r tau_2 angle.r v $

Note that the last form is the *wrapped function value* — a value that has not yet had its enclosed cast discharged. Reduction:

```text
  (lam x:tau. e) v  -->  [x |-> v] e

  <tau <- tau> v  -->  v                                              (id-cast)

  <"?" <- tau> v  -->  Inj_tau v             (tau is ground)          (inject)

  <tau <- "?"> (Inj_tau v)  -->  v                                    (project-ok)

  <tau1 <- "?"> (Inj_tau2 v)  -->  blame ell  (tau1 != tau2 ground)   (project-fail)

  (<tau1'->tau2' <- tau1->tau2>^ell v) w
      -->  <tau2' <- tau2>^ell (v (<tau1 <- tau1'>^bar(ell) w))       (wrap)
```

A *ground* type is one of `Bool`, `Int`, or the *prime* arrow `? -> ?`. Higher-order casts to "?" decompose through the prime arrow: $angle.l "?" arrow.l tau_1 arrow.r tau_2 angle.r v$ reduces to $angle.l "?" arrow.l "?" arrow.r "?" angle.r (angle.l "?" arrow.r "?" arrow.l tau_1 arrow.r tau_2 angle.r v)$. The factoring through ground types is what makes blame coherent: the run-time check is always between a known ground type and another known ground type.

*Theorem (Type Safety for $lambda_C$).* If $dot tack.r e : tau$ then either $e$ is a value of type $tau$, $e arrow.r e'$ with $dot tack.r e' : tau$, or $e arrow.r "blame" space ell$ for some label $ell$.

*Proof.* Standard progress and preservation, treating blame as an additional terminal configuration not classified as "stuck". The casts on values are themselves typed and reduce cleanly; the only stuck-looking configurations are those that produce blame, which the theorem explicitly allows. $square$

The contrast with the Milner-style theorem is informative: *gradual type safety admits blame as a legitimate outcome*. Soundness does not promise the absence of type errors; it promises that those errors are *located* (a blame label) and *attributed* (to a definite side of a cast).

== Performance: The Cost of Partial Typing

The promise of gradual typing is *incremental migration*. The reality, as measured by (Takikawa–Greenman–Felleisen 2016) in Typed Racket, is sobering. The methodology:

- Take a small, statically typed program with $n$ modules.
- For each of the $2^n$ subsets of modules, *erase* the types on the chosen modules.
- Run the resulting *configuration* on a benchmark workload and record the slowdown relative to the fully typed program.
- Plot the cumulative distribution: what fraction of configurations are slower than $k times$ the fully typed baseline?

The results were stark. For most benchmarks, fewer than 5% of configurations ran within $2 times$ the fully typed baseline. *Many* configurations ran $20 times$ slower, and some pathological configurations ran $100 times$ slower. The reason: every typed/untyped boundary becomes a contract; higher-order values crossing the boundary acquire wrappers; the wrappers cost a dispatch on every call and an additional check on every argument.

The findings provoked a research programme on *sound efficient gradual typing*. The main strategies:

1. *Cast erasure.* If type inference can prove that a value never crosses a boundary, the cast is unnecessary. *Type-Tailored* gradual typing (Kuhlenschmidt–Almahallawi–Siek 2019) does aggressive monomorphisation.
2. *Transient checks* (Vitousek–Swords–Siek 2017). Replace deep wrappers with *shallow* tag checks at every operation that consumes a value. Transient semantics gives weaker blame (it cannot blame the original contract violator, only the immediate operation), but the run-time overhead is bounded and predictable.
3. *Type-directed unboxing.* Where the type checker has a static type, the run time uses an unboxed representation; the boundary inserts a single box/unbox. This is the strategy used in Reticulated Python.
4. *Optional types without soundness* (TypeScript, mypy). The compiler erases types entirely; checks happen only at compile time. The cost at run time is zero, but the type system makes no soundness promise about programs at all — the *gradual guarantee* and *blame theorem* are sacrificed.

The TypeScript decision is significant: it has been *enormously* successful in industry but is technically not "sound" gradual typing. The compiler accepts implicit `"any"` and never inserts a run-time cast. This is sometimes called *optional* typing rather than gradual typing.

== Surface Languages: A Tour

=== Typed Racket

Typed Racket (Tobin-Hochstadt–Felleisen 2008) is the canonical sound gradual language. A module declares whether it is typed:

```racket
#lang typed/racket

(: factorial (-> Integer Integer))
(define (factorial n)
  ("if (zero? n) 1 (* n (factorial (- n 1)))))

(: map* (All (a b) (-> (-> a b) (Listof a) (Listof b))))
(define (map* f xs)
  ("if" (null? xs) '() (cons (f (car xs)) (map* f (cdr xs)))))
```

Untyped modules can import typed ones and vice versa; the contract layer interposes at the boundary. The contract on a polymorphic function uses *parametricity-preserving* sealing (Matthews–Findler 2009) — the untyped side cannot inspect a sealed value, and the typed side cannot accidentally observe a value that should have been polymorphic.

=== TypeScript

```typescript
function len(x: string | any"[]): number {
  return x.length;        // static dispatch on union
}

function risky(x: any"): number {
  return x.foo.bar.baz;   // any disables all checks
}

function safer(x: unknown): number {
  if (typeof x === "object" && x !== null && "foo" in x) {
    return (x as { foo: { bar: { baz: number } } }).foo.bar.baz;
  }
  return 0;
}
```

The distinction between `"any"` and `unknown` is the entire gradual story compressed into two keywords. `"any"` is the *unsound* top: it can flow anywhere, no checks. `unknown` is the *sound* top: it can be assigned from anywhere but assigned to nothing without a type narrowing. TypeScript's design encourages `unknown` for boundaries and `"any"` only as a transitional escape hatch.

=== Python (mypy, Pyright)

```python
"from typing import Any, Optional

eq.def parse_age(s: str) -> Optional[int]:
    try:
        return int(s)
    except ValueError:
        return None

eq.def legacy_callback(payload: Any) -> None:
    print(payload["user"]["name"])   \# mypy admits, runtime may fail
```

Python type hints (PEP 484) are *purely advisory*. The interpreter ignores them; only external checkers (mypy, Pyright, Pyre) consult them. There is no run-time cast and no blame. This is *optional* typing, not gradual typing in the technical sense.

A separate experimental dialect, *Reticulated Python* (Vitousek–Kent–Siek–Baker 2014), implements sound gradual typing for a fragment of Python via the transient semantics, and is the source of much of the empirical work on the cost of partial annotation.

=== Hack and Sorbet

Hack (Facebook) and Sorbet (Stripe) take Python/Ruby and attach a checker. Sorbet has a tri-level type lattice: `T.untyped` (no checking), `T.let` (compile-time only), and `T.must` (compile-time + run-time enforced). The three levels correspond to three different points on the spectrum traded off in (Takikawa et al. 2016): the more enforcement, the more cost.

== Hybrid Type Checking

(Flanagan 2006) introduced *hybrid type checking* as a refinement of gradual typing for *refinement types*: types of the form $ { x : tau | phi(x) } $ where $phi$ is a predicate. The challenge: subtyping between refinement types reduces to *implication* between predicates, which is undecidable in general.

The hybrid solution: try to prove subtyping by an SMT solver; if the solver succeeds, the subtyping is *statically* discharged; if the solver fails (or times out), insert a *run-time* check $angle.l { x : tau | phi(x) } arrow.l tau angle.r e$ that evaluates the predicate at the boundary.

```text
  Gamma |- e : tau1     SMT |- Gamma => tau1 <: tau2
  ---------------------------------------------------- (T-SUB-STATIC)
  Gamma |- e : tau2

  Gamma |- e : tau1     SMT cannot decide tau1 <: tau2
  ----------------------------------------------------- (T-SUB-HYBRID)
  Gamma |- <tau2 <- tau1> e : tau2
```

The system is *sound* (run-time checks catch what statics could not) and *gradually rigorous*: as the solver improves, more checks are discharged statically; a Hindley–Milner-style baseline degenerates to ordinary refinement-type checking when the solver always succeeds. The design has inspired *Liquid Haskell*, *F${star.op}$*, and the refinement layer of *Dafny*.

== Threesomes, Coercions, and Space Efficiency: Detail

The technical machinery behind threesomes deserves spelling out. Define the *meet* $tau_1 inter.sq tau_2$ on types:

```text
  tau inter.sq tau   = tau
  "?" inter.sq tau   = tau
  tau inter.sq "?"   = tau
  (tau1 -> tau2) inter.sq (tau1' -> tau2') = (tau1 inter.sq tau1') -> (tau2 inter.sq tau2')
  otherwise      = "fail"
```

A *threesome* is a triple $(tau_1, tau_"mid", tau_2)$ where $tau_"mid"$ is the meet of all intermediate types of a sequence of casts. Composition:

$ (tau_1, m_1, tau_2) ; (tau_2, m_2, tau_3) = (tau_1, m_1 inter.sq m_2, tau_3) $

The composition fails — and the program raises blame — if and only if $m_1 inter.sq m_2$ fails. This is exactly when an inconsistency would have been discovered in the naïve sequence.

*Theorem (Space Efficiency, Herman–Tomb–Flanagan 2010, Siek–Wadler 2010).* In a cast calculus with threesomes, the size of any value is bounded by a constant times the size of its type. Hence the heap usage of a gradually typed program is asymptotically the same as the corresponding statically typed program.

*Proof.* By induction on values. A wrapped function value $angle.l (tau_1, m, tau_2) angle.r v$ stores one threesome, of constant size; the inner $v$ is recursively bounded. Composition of two wrappers produces another wrapper with a single threesome. $square$

== Gradual Session Types

(Igarashi–Thiemann–Vasconcelos–Wadler 2017) extended gradual typing to *session types* — protocols on communication channels. A session type describes the sequence of sends and receives expected on a channel: $!"Int". ?"Bool". "end"$ means "send an integer, receive a boolean, terminate". The "?" session type allows a fragment of code to be ignorant of the protocol; casts to a more specific session type insert run-time protocol checks.

Soundness is subtle because session types are *linear*: a channel must be used exactly as described. Mixing typed and untyped channel users requires *delegation*: when a typed process hands off a channel to an untyped one, the untyped one acquires the protocol obligation, monitored at run time. The framework has been mechanised in Coq.

== Gradual Dependent Types

(Lennon-Bertrand–Maillard–Tabareau–Tanter 2022) constructed *Gradual CIC* (GCIC): a gradual extension of the Calculus of Inductive Constructions. The dynamic type "?" becomes a proof-relevant *unknown term* $"?"_A$ at each type $A$. The cast $angle.l B arrow.l A angle.r$ on dependent types becomes a unifier that may produce blame, and Gradual CIC enjoys both the gradual guarantee and *strong normalisation* — proofs do not loop, but they may fail to typecheck only after computation.

The construction reveals a deep tension: *univalence* (see _Homotopy Type Theory_) and *gradual guarantee* are *incompatible* in their natural strongest forms. GCIC must weaken one or the other. The chosen weakening — *propositional* gradual guarantee rather than definitional — has consequences for what proofs survive the gradualisation.

== Soft Typing: The Predecessor

(Cartwright–Fagan 1991) wrote down the first system in this family. *Soft typing* infers a type for *every* expression in an untyped Scheme program; the inferencer is non-rejecting — if it cannot find a satisfying type, it inserts a check and records a warning. The inferred types are *"set"-based*: each type variable ranges over a set of *concrete* types observed at the call sites. The algorithm performs *"set"-based analysis* (Heintze 1994) and dispatches on the smallest set consistent with the observed flow.

The descendants of soft typing are: *flow analysis* in dynamic languages (Shivers's CFA, Henglein's $tau$-trees), the inferencer in *DRuby* and *PyType*, and the *occurrence typing* of Typed Racket (Tobin-Hochstadt–Felleisen 2010), which narrows types inside conditional branches by *propagating predicates* through the syntax.

== Optimisations and Implementation

Production gradual systems combine several techniques:

- *Cast specialisation.* The expander compiles each cast site to a specialised checker for the static type involved. A cast `<Int <- ?>` becomes a fixed two-instruction `tag-check + extract`; a cast on a record type becomes a record-walking traversal.
- *Inlining.* Many casts are inlined at the call site; the contracting wrapper becomes a few instructions inlined into the caller.
- *Profile-guided cast elimination.* If profiling shows that 99% of values flowing through a cast pass the check, the JIT specialises on that case.
- *Just-in-time type specialisation.* Tracing JITs (PyPy, V8) effectively rediscover the static types of dynamic values from trace data and remove the dispatch cost; this is type *erasure* aided by runtime evidence.

The combination of these techniques in Typed Racket, *Pycket* (a tracing JIT for Racket), and the V8 implementation of TypeScript-shaped code closes much of the gap measured in 2016. Recent work (Greenman–Felleisen 2018; Greenman–Takikawa–New–et al. 2019) refined the measurement methodology to *natural* vs *transient* vs *erasure* semantics, plotting each on the same axes; the conclusion is that no single point on the soundness/efficiency spectrum dominates.

== Discussion: Where Are We?

Gradual typing succeeded in industry. TypeScript, Python type hints, Hack, and Sorbet all command large user bases; the *vocabulary* of gradual typing (`any`, `unknown`, optional fields, type narrowing) is now common parlance. The sound, blame-tracking discipline of the academic line (Typed Racket, Reticulated, GCIC) is a smaller community but produces the theory that underwrites the others.

Three open problems remain prominent:

1. *Performance.* The sound semantics still pays a measurable cost. Whether *transient* or *natural* semantics is preferable is partly an empirical question, partly a question of error message quality.
2. *Polymorphism and parametricity.* Gradual parametric polymorphism requires sealing to preserve parametricity; the engineering of sealing in a fast run-time is delicate.
3. *Dependence and effects.* Gradual extensions to dependent types and effect systems are active research; the AGT methodology has been productive but does not always yield the most usable system.

Hybrid type checking is alive in *Liquid Haskell*, *F${star.op}$*, and *Dafny*: tools that combine SMT discharge of subtyping with run-time residual checks where needed. The dividing line between "gradual" and "hybrid" has blurred: both are points on a continuum where the type checker decides at compile time how much work to defer to run time, and the run time honours the deferred obligations.

The grand bet of gradual typing — that programmers can migrate large codebases from untyped to typed incrementally — has been *partially* vindicated. The migration happens; the soundness costs have been the main obstacle; and the design space of how much soundness to insist upon is still being explored, three decades after the first soft-typing paper.

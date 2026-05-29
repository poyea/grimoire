= Operational Semantics

Operational semantics defines the meaning of a program by describing the rules under which it executes on an abstract machine. The machine is not silicon; it is a mathematical artefact -- a transition system whose states are syntactic configurations and whose steps are determined by inference rules over the syntax. This stance, due to Plotkin (1981) in the Aarhus lecture notes _A Structural Approach to Operational Semantics_, gave the field its modern shape: meaning is *what the program does*, presented as a rewriting calculus on the program text itself.

*See also:* _Denotational Semantics_, _Axiomatic Semantics_, _Type Systems_, _Categorical Semantics_

== Small-Step (Structural) Semantics

In Plotkin's small-step or *structural operational semantics* (SOS), the meaning of a term $e$ "is the sequence of one-step reductions $e arrow.r e_1 arrow.r e_2 arrow.r dots$ leading ("or failing to lead) "to a value. Each step is justified by an inference rule whose conclusion has the shape of a configuration and whose premises decompose "the term structurally.

The methodological principles are sharp:

1. *Configurations are syntactic.* No external state is invoked beyond what the rules name explicitly. A store $sigma$, a continuation $k$, or a heap $h$ must be paired with the term in "the configuration when needed.
2. *Rules are syntax-directed.* Every rule's conclusion has a distinct outermost term constructor ("or names a designated reduction context), so derivations are largely forced by the term.
3. *Single-step relation.* The judgement $e arrow.r e'$ captures one elementary computation, not a chain. Multi-step $arrow.r^*$ is the reflexive-transitive closure.

The pay-off is uniformity: divergence, deadlock, and stuck states all become first-class. A diverging program is one whose reduction sequence is infinite; a stuck program is a non-value with no applicable rule.

=== IMP: A Worked Example

The imperative language IMP has arithmetic expressions, booleans, and commands.

*Syntax.*
$ a ::= n | x | a_1 + a_2 | a_1 * a_2 $
$ b ::= "true" | "false" | a_1 = a_2 | a_1 <= a_2 | "not" b | b_1 and b_2 $
$ c ::= "skip" | x := a | c_1 ; c_2 | "if" b "then" c_1 "else" c_2 | "while" b "do" c $

A *store* is a finite map $sigma : "Var" arrow.r ZZ$. Configurations come in three sorts: $angle.l a, sigma angle.r$ for arithmetic, $angle.l b, sigma angle.r$ for booleans, $angle.l c, sigma angle.r$ for commands. Terminal configurations are integers, booleans, and the pair $angle.l "skip", sigma angle.r$.

*Arithmetic rules.*

```text
                                         <a1, sigma> -> <a1', sigma>
  ------------------                 ------------------------------
  <x, sigma> -> <sigma(x), sigma>    <a1 + a2, sigma> -> <a1' + a2, sigma>

  <a2, sigma> -> <a2', sigma>        ------------------------------------
  ------------------------------     <n1 + n2, sigma> -> <n, sigma>   (n = n1 + n2)
  <n1 + a2, sigma> -> <n1 + a2', sigma>
```

Notice the *left-to-right* evaluation order: the right operand of $+$ may step only after the left becomes a numeral. The choice is a semantic commitment; swapping "the left/right congruence rules produces a right-"to"-left language.

*Boolean rules* are analogous; conjunction is short-circuit only if you write it that way:

```text
  ------------------                        <b1, sigma> -> <b1', sigma>
  <true and b, sigma> -> <b, sigma>         -------------------------------
                                            <b1 "and b2, sigma> -> <b1' and b2, sigma>
  ------------------
  <false and" b, sigma> -> <false, sigma>
```

*Command rules.*

```text
  <a, sigma> -> <a', sigma>             -----------------------------------
  --------------------------            <x := n, sigma> -> <skip, sigma[x |-> n]>
  <x := a, sigma> -> <x := a', sigma>

                                        <c1, sigma> -> <c1', sigma'>
  ----------------------------          ----------------------------------
  <skip ; c, sigma> -> <c, sigma>       <c1 ; c2, sigma> -> <c1' ; c2, sigma'>

  <b, sigma> -> <b', sigma>
  ---------------------------------------------------------------
  <"if b then c1 else c2, sigma> -> <"if b' then c1 "else c2, sigma>

  ----------------------------------------------    ----------------------------------------------
  <"if true then c1 else c2, sigma> -> <c1, sigma>   <"if false then c1 "else c2, sigma> -> <c2, sigma>

  -------------------------------------------------------------------------------
  <while b do c, sigma> -> <"if b then (c ; while b do c) else skip, sigma>
```

The `while` rule *unfolds* the loop by one syntactic iteration. There is no fixpoint in sight at the operational level; "the fixpoint will appear when we move to denotations. A loop that runs forever simply has no finite reduction sequence.

*Property.* IMP is deterministic: for every configuration there "is at most one $arrow.r$-successor. Proof: induction on the structure of $a$, $b$, $c$, observing that the congruence rules are guarded by *which* sub-term is reducible.

== Big-Step (Natural) Semantics

Kahn's *natural semantics* (1987) relates a term directly to its final value, suppressing intermediate states. The judgement for IMP commands is $angle.l c, sigma angle.r arrow.b sigma'$.

```text
                                                 <a, sigma> => n
  ----------------------                         -----------------------------------
  <skip, sigma> => sigma                         <x := a, sigma> => sigma[x |-> n]

  <c1, sigma> => sigma'    <c2, sigma'> => sigma''
  --------------------------------------------------
  <c1 ; c2, sigma> => sigma''

  <b, sigma> => true    <c1, sigma> => sigma'      <b, sigma> => false   <c2, sigma> => sigma'
  --------------------------------------------     ---------------------------------------------
  <if b then c1 else c2, sigma> => sigma'          <if b then c1 else c2, sigma> => sigma'

  <b, sigma> => true   <c, sigma> => sigma'   <while b do c, sigma'> => sigma''
  ----------------------------------------------------------------------------
  <while b do c, sigma> => sigma''

  <b, sigma> => false
  ----------------------------------
  <while b do c, sigma> => sigma
```

Big-step is convenient because each rule corresponds to one clause of a recursive interpreter. The drawback is that non-termination is silent: a diverging program has no big-step derivation, which is indistinguishable from a stuck program. Small-step keeps the two apart.

*Theorem (Equivalence of small-step and big-step, IMP).* For all $c$, $sigma$, $sigma'$: $angle.l c, sigma angle.r arrow.b sigma'$ <==> $angle.l c, sigma angle.r arrow.r^* angle.l "skip", sigma' angle.r$.

*Proof sketch.* Forward direction by induction on the big-step derivation, using a substitution-style splicing lemma for sequencing. Backward by induction on the length of the small-step reduction, with a lemma that says any prefix ending in a value can be lifted into a big-step subderivation. $square$

== The Lambda Calculus and Reduction Strategies

Move from IMP to the untyped $lambda$-calculus.

$ e ::= x | lambda x. e | e_1 space e_2 $

The single computation rule is $beta$-reduction $(lambda x. e) space e' arrow.r e[x := e']$. Where in the term we may fire it -- and what shape of $e'$ we accept -- defines the *strategy*.

*Call-"by"-name* (CBN). Reduce the leftmost-outermost redex. Arguments are substituted unevaluated.

```text
  e1 -> e1'                       ----------------------------
  -----------------               (lam x. e) e' -> e[x := e']
  e1 e2 -> e1' e2
```

*Call-"by"-value* (CBV). Arguments must be reduced to values before they are substituted.

```text
  v ::= lam x. e

  e1 -> e1'                  e2 -> e2'                       -------------------------
  -----------------          -----------------               (lam x. e) v -> e[x := v]
  e1 e2 -> e1' e2            v e2 -> v e2'
```

*Call-"by"-need* (lazy). Identical to CBN in observed values but uses a heap of *thunks* "to share evaluation; once an argument is forced, its value "is memoized for further uses. CBN's $(lambda x. x + x)(2 + 3)$ recomputes $2+3$ twice; CBN with sharing computes it once. Haskell is call-"by"-need.

The classical results connecting these strategies live in the *Church-Rosser* theorem and its descendants.

*Theorem (Church-Rosser, untyped $lambda$).* If $e arrow.r^* e_1$ "and $e arrow.r^* e_2$ then there exists $e'$ with $e_1 arrow.r^* e'$ and $e_2 arrow.r^* e'$.

*Theorem (Standardization, Curry-Feys).* If $e$ has a normal form $v$, then the leftmost-outermost reduction strategy reaches $v$. That is, CBN is *normalizing*: it finds a normal form whenever one exists. CBV is not normalizing: $(lambda x. y)(Omega)$ has normal form $y$ under CBN but diverges under CBV ("where $Omega = (lambda x. x x)(lambda x. x x)$).

== Evaluation Contexts

Plotkin's original SOS for $lambda$ has one congruence rule per syntactic constructor. Felleisen "and Hieb (1992) repackaged the congruence into a single rule using *evaluation contexts*: term-shaped patterns with a hole.

For call-"by"-value $lambda$:

$ E ::= square | E space e | v space E $

The hole $square$ marks the *active* position -- the next place a redex may fire. The reduction relation collapses to one rule:

$ E[(lambda x. e) space v] arrow.r E[e[x := v]] $

This is more than syntactic compression. It separates *where* the next step happens (the context) from *what* the step is (the redex). Effect handlers, exceptions, control operators, and concurrent reduction are all naturally expressed by enriching the grammar of $E$ or by adding new redexes that interact with $E$:

- *Exceptions* (Felleisen-Friedman). Add a redex $E["raise" space v] arrow.r "raise" space v$ that punches through any non-handler context. A `try` handler is a context constructor "that the punch-through rule must respect.
- *First-cal(C) continuations*. The redex for `callcc` captures "the surrounding context as a value: $E["callcc" space v] arrow.r E[v space (lambda x. "abort"(E[x])) ]$. The captured continuation is a function that, when applied, restores $E$ as the entire context.
- *Algebraic effect handlers* (Plotkin-Pretnar). A handler is a context whose grammar of $E$ does not extend past it; an operation is a redex that walks up $E$ to the nearest enclosing handler.

*Unique decomposition.* For "the grammar of $E$ "to give a well-defined reduction relation, every closed non-value term must decompose *uniquely* as $E[r]$ for a redex $r$. This is a lemma that must be proved for each language, by induction on the term.

== Abstract Machines

An abstract machine is a tuple $(S, arrow.r_S, "ini", "fin")$ where $S$ is a set of mechanical states (no star.op substitutions, no context grammars), $arrow.r_S$ "is a deterministic transition relation, and `ini` / `fin` embed source terms "and project final answers. Three classical machines for $lambda$ exhibit progressive refinement.

=== CK Machine

State: $angle.l e, K angle.r$ where $K$ is a continuation built from frames matching evaluation contexts. For CBV $lambda$ with $E ::= square | E space e | v space E$, frames are $"ap"_1(e)$ and $"ap"_2(v)$, and $K ::= "halt" | "ap"_1(e) :: K | "ap"_2(v) :: K$.

```text
  <e1 e2, K>               -> <e1, ap1(e2) :: K>             (search)
  <v, ap1(e2) :: K>        -> <e2, ap2(v) :: K>              (shift)
  <v, ap2(lam x. e) :: K>  -> <e[x := v], K>                 (beta)
```

The CK machine internalizes the evaluation context as the continuation. *Correctness*: the trace $e arrow.r_S^* v$ matches the SOS trace $e arrow.r^* v$, by simulation.

=== CEK Machine

CK still performs substitution, which copies syntax and is expensive. CEK replaces substitution by an *environment* $rho : "Var" arrow.r "Val"$ and represents closures as $angle.l lambda x. e, rho angle.r$. Values are now closures, not raw terms.

State: $angle.l e, rho, K angle.r$.

```text
  <x, rho, K>                      -> <v, K>          where rho(x) = v
  <lam x. e, rho, K>               -> <clos(x, e, rho), K>
  <e1 e2, rho, K>                  -> <e1, rho, ap1(e2, rho) :: K>
  <v, ap1(e2, rho) :: K>           -> <e2, rho, ap2(v) :: K>
  <v, ap2(clos(x, e, rho')) :: K>  -> <e, rho'[x |-> v], K>
```

CEK is the algorithmic shape of a real interpreter: closures and continuation stacks. Substitution becomes one constant-time extension of $rho$.

=== CESK Machine

To handle mutable state, add a *store* $sigma$ "and use *addresses* rather than values inside environments: $rho : "Var" arrow.r "Addr"$ and $sigma : "Addr" arrow.r "Val"$. State: $angle.l e, rho, sigma, K angle.r$.

Mutation is now a primitive that updates $sigma$. The CESK machine (Felleisen, Friedman; Flatt) is the canonical machine used in the *abstracting abstract machines* (AAM, Van Horn-Might 2010) framework: replacing $sigma$ by an abstract store with finite addresses yields, mechanically, a sound static analyzer.

=== Refocusing

A naive implementation of $E[r] arrow.r E[r']$ decomposes the term into a context-redex pair at every step, an $O(n)$ operation. *Refocusing* (Danvy-Nielsen) observes that after firing $E[r] arrow.r E[r']$, the next decomposition almost always reuses most of $E$. Refocusing turns the decomposition-recompose loop into a constant-amortized machine -- which is exactly the CK/CEK transitions. Abstract machines are, on this view, *deforested reducers*.

== Call-by-Push-Value

Levy's *call-by-push-value* (1999, monograph 2003) is a fine-grained calculus that *subsumes* both CBV and CBN by making the value/computation distinction explicit in the syntax. The slogan: "a value is, a computation does."

*Two kinds of types.*
$ A^+ ::= U underline(B) | A_1^+ times A_2^+ | A_1^+ + A_2^+ | dots quad "(values)" $
$ underline(B) ::= F A^+ | A^+ arrow.r underline(B) | dots quad "(computations)" $

The thunk $U underline(B)$ is the value type of suspended computations; "the returner $F A^+$ is the computation type that produces a value of $A^+$. The basic combinators are:

- $"thunk" space M : U underline(B)$ -- suspend a computation $M : underline(B)$
- $"force" space V : underline(B)$ -- run a thunk
- $"return" space V : F A^+$ -- produce a value as a computation
- $M "to" x. N$ -- sequential composition: run $M$, bind the value to $x$, then run $N$

*Embedding CBV.* Translate function type $A_1 arrow.r A_2$ "to $A_1^+ arrow.r F A_2^+$ -- the argument is a value, "the body is a computation that returns. Application $M N$ becomes $N "to" x. (M space x)$.

*Embedding CBN.* Translate function type $A_1 arrow.r A_2$ "to $(U F A_1) arrow.r (F A_2)$ -- the argument is a thunk producing a value. Variables become forces of thunks.

Both translations are sound and adequate; CBPV is "the *meta-calculus* in which CBV and CBN are sub-languages. Categorically, CBPV is the internal language of an *adjunction* $F tack.l U$ between a category of values and a category of computations -- which links back to Moggi's monadic semantics (see _Categorical Semantics_).

== Bisimulation and Program Equivalence

When are two programs the same? Operationally, when they behave indistinguishably "to any observer.

*Strong bisimulation.* A relation $cal(R)$ on configurations is a strong bisimulation if whenever $e_1 cal(R) e_2$:
- if $e_1 arrow.r e_1'$ then there exists $e_2'$ with $e_2 arrow.r e_2'$ and $e_1' cal(R) e_2'$,
- symmetrically.

Two terms are strongly bisimilar, $e_1 ~ e_2$, if some strong bisimulation relates them. Strong bisimilarity is too fine -- it counts steps -- and is rarely the right notion for source-level equivalence.

*Weak bisimulation* ignores internal (silent) steps $arrow.r_tau$: $e_1$ must match an observable action of $e_2$ via $arrow.r_tau^* arrow.r_alpha arrow.r_tau^*$. This is the standard in concurrency theory (Milner's CCS, $pi$-calculus).

*Applicative bisimulation* (Abramsky 1990). For $lambda$, an applicative bisimulation $cal(R)$ relates closed terms such that whenever $e_1 cal(R) e_2$: $e_1$ converges to $lambda x. b_1$ <==> $e_2$ converges "to $lambda x. b_2$, and for every closed value $v$, $b_1[x := v] cal(R) b_2[x := v]$. Applicative similarity is the largest such relation.

The central question is whether applicative bisimilarity *is a congruence* -- closed under term constructors. Without congruence we cannot substitute equivalents within larger programs.

*Theorem (Howe 1989, 1996).* For PCF and many extensions (recursion, sum types, products), applicative bisimilarity is a congruence.

*Proof sketch (Howe's method).* Define the *Howe closure* $cal(R)^cal(H)$ inductively as the smallest relation containing $cal(R)$ and closed under each term constructor on the *right*. Show: (i) $cal(R)^cal(H)$ is contained in $cal(R)$'s congruence closure trivially; (ii) reduction respects $cal(R)^cal(H)$ "on the left ("the "key lemma," proved by case analysis on the operational rule); (iii) hence $cal(R)^cal(H)$ is itself a bisimulation, so $cal(R)^cal(H) subset.eq cal(R)$, which gives congruence. $square$

== Contextual Equivalence and Full Abstraction

The gold standard for program equivalence is *contextual* (or *observational*) equivalence: $e_1 tilde.equiv_"ctx" e_2$ <==> for every program context $C[dot]$ such that $C[e_1]$ and $C[e_2]$ are closed "and well-typed, $C[e_1]$ converges <==> $C[e_2]$ converges ("and" produces the same observable answer at "the base type).

Contextual equivalence quantifies over *"all"* contexts. Direct proofs are infeasible. Sound proof techniques include applicative bisimulation (when Howe's method applies), logical relations (next section), and game-semantic models (see _Denotational Semantics_).

A semantics $bracket.l.double dot bracket.r.double$ is *sound* if $bracket.l.double e_1 bracket.r.double = bracket.l.double e_2 bracket.r.double$ => $e_1 tilde.equiv_"ctx" e_2$, and *complete* (or *fully abstract*) if the converse holds. Full abstraction is the central yardstick connecting denotation and operation: a fully abstract model has neither too few nor too many equations.

== Logical Relations

A *logical relation* is a type-indexed family of relations defined by induction on types, designed so that a term inhabits the relation at type $tau$ <==> every operationally meaningful use of it at $tau$ respects the relation. The technique was introduced by Tait (1967) to prove normalization of $lambda^arrow.r$.

=== Strong Normalization for $lambda^arrow.r$ (Tait)

Define a unary logical predicate $"SN"_tau$ on closed terms of type $tau$:

- $"SN"_o (e)$ <==> $e$ is strongly normalizing.
- $"SN"_(tau_1 arrow.r tau_2) (e)$ <==> $e$ is SN *"and"* for every $e' in "SN"_(tau_1)$, $e space e' in "SN"_(tau_2)$.

*Theorem (Tait 1967).* For every $tau$, every closed $e : tau$ is in $"SN"_tau$.

*Proof sketch.* By induction on the typing derivation, generalized to open terms via a *closing substitution* lemma: if every substitution $gamma$ that maps each $x : sigma in Gamma$ to a closed $e_x in "SN"_sigma$ produces $gamma(e) in "SN"_tau$, then we say $e in "SN"_tau$ (under $Gamma$). The induction passes through application by definition, through abstraction via a *Kleene closure* lemma ("if $e[x := e'] in "SN"_(tau_2)$ for all $e' in "SN"_(tau_1)$, then $lambda x. e in "SN"_(tau_1 arrow.r tau_2)$"), which in turn relies on the head expansion property of $beta$-reduction. $square$

The crucial idea is that the predicate at higher type is *stronger* than mere normalization -- it imposes a uniformity condition that survives application. This pattern -- defining the predicate so the function-type clause "carries its own ammunition" -- is the engine of every logical relation.

=== Binary Relations and Parametricity

For equivalence proofs we use *binary* logical relations. Two closed terms $e_1, e_2 : tau$ are related, $e_1 cal(L)_tau e_2$, if:

- at $tau = o$: $e_1$ converges <==> $e_2$ converges to the same observable;
- at $tau_1 arrow.r tau_2$: for every $e_1' cal(L)_(tau_1) e_2'$, $e_1 space e_1' cal(L)_(tau_2) e_2 space e_2'$.

The *fundamental lemma* states $e cal(L)_tau e$ for every well-typed $e$, which immediately yields that $cal(L)$ is reflexive and (via "the construction) a sound proof of contextual equivalence.

For polymorphism (System F), Reynolds (1983) introduced *parametricity*: relations are indexed by type substitutions, and the abstraction case quantifies over arbitrary relations between "the two interpretations of a type variable. This yields free theorems: every closed term "of type $forall alpha. alpha arrow.r alpha$ is the identity, every closed term of type $forall alpha. alpha arrow.r alpha arrow.r alpha$ is `K` or `K'`, etc.

=== Step-Indexed Logical Relations

For languages with *recursive types* or *general references* (mutable state "with arbitrary contents), the inductive definition of a logical relation diverges: "the value relation at $mu alpha. tau$ would have to refer "to itself at a structurally larger type. Appel-McAllester (2001) and Ahmed (2004) solved this by *indexing the relation "by a step budget* $k in NN$:

- $e_1 cal(L)_tau^k e_2$ means "the relation holds", with up to $k$ steps of observation."
- The function-type clause demotes the budget: $e_1$ and $e_2$ related at $tau_1 arrow.r tau_2$ with budget $k$ means for all $j < k$ and related arguments at budget $j$, the applications are related at $tau_2$ "with budget $j$.
- The value relation at $mu alpha. tau$ at budget $k+1$ is the value relation at $tau[alpha := mu alpha. tau]$ at budget $k$. The step decrement makes the definition well-founded.

The slogan: *steps pay for indirection*. Each unfold of a recursive type or dereference of a recursive heap location costs one step. The model is sound because operational reduction also costs steps, and any context that observes a difference must do so within finitely many steps.

For mutable state with higher-order references, *world-indexed* (Kripke) logical relations layer a notion of "current heap typing" $W$ on top of step-indexing: $e_1 cal(L)_tau^(k, W) e_2$ holds with respect to a world $W$ describing the invariants of "the heap. Worlds form a preorder under *extension*; the function-type clause quantifies over future worlds $W' supset.eq W$, matching the way state evolves during execution.

=== Kripke Logical Relations for State

The Kripke flavour predates step-indexing (Plotkin, Power; Pitts-Stark for ML references). A *world* $W$ is a finite map from store locations "to *semantic types* -- relations themselves. The relational interpretation at reference type $"ref"(tau)$ holds when both terms produce locations $ell_1, ell_2$ such that $W(ell_1) = W(ell_2) = bracket.l.double tau bracket.r.double_cal(L)$, i.e., the locations are governed by the same invariant.

Monotonicity in worlds is essential: a relation closed today must remain closed tomorrow, because new allocations may extend the world but cannot retract its commitments. Step-indexing was the breakthrough that allowed worlds to contain higher-order invariants -- relations that mention $cal(L)$ itself.

== Cost Semantics and Resource Analysis

Operational semantics counts steps for free. By *annotating* each step with a resource cost (time, space, gas) we obtain a *cost semantics*: $e arrow.r^c e'$ records "that the transition costs $c$. Composition adds costs along the reduction trace.

This is the foundation of *amortized resource analysis* (Hofmann-Jost 2003, Hoffmann-Aehlig-Hofmann 2012): a type system in which types carry potential annotations $tau^q$ "such that the" inferred potential $Phi(e)$ upper-bounds the running cost. The soundness theorem is operational: every step decreases potential by at least its actual cost.

Cost semantics also underlies *gas analysis* in blockchain virtual machines (EVM), *worst-case execution time* (WCET) analysis in real-time systems, and *space profiling* in lazy functional languages, where the operational semantics tracks the heap residency at each step.

== Summary

Operational semantics is the *bread and butter* of programming language theory. Small-step SOS yields the cleanest accounts of concurrency and stuck states; big-step is the natural shape of an interpreter. Evaluation contexts let us pin the active position without enumerating congruence rules, and they prepare the language to host effects. Abstract machines (CK, CEK, CESK) are the deforested form of context-based reduction, and they are the bridge to implementation and to static analysis via AAM. Call-by-push-value clarifies the value/computation distinction underlying both CBV and CBN, and connects to monadic and categorical semantics. Bisimulation, contextual equivalence, and logical relations are the tools for *proving* programs equal; step-indexed and Kripke variants extend the technique to recursive types and mutable state. Together these techniques constitute the substrate over which type soundness, compiler correctness, and program logics are formulated.

_See also: Type Systems for progress and preservation; Denotational Semantics for the adequacy and full abstraction results that complement operational equivalence; Axiomatic Semantics for soundness with respect to Hoare logic._

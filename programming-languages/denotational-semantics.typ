= Denotational Semantics

Denotational semantics assigns to each program a *mathematical object* -- a function, a set, an element of a domain -- in a way that respects the syntactic structure: the meaning of a compound is determined by the meanings of its parts. Where operational semantics says *what the program does step by step*, denotational semantics says *what the program is*, once and for all. The framework was crystallized by Scott and Strachey at Oxford in the late 1960s and early 1970s; the technical core is the theory of domains and continuous functions developed by Scott (1969, 1972, 1976) to give recursive functions and recursive types a mathematical home.

*See also:* _Operational Semantics_, _Axiomatic Semantics_, _Categorical Semantics_, _Type Systems_

== The Compositionality Principle

A semantics $bracket.l.double dot bracket.r.double$ "is *compositional* if", for every syntactic constructor $C$ of arity $n$, there is a mathematical operation $hat(C)$ such that

$ bracket.l.double C(e_1, dots, e_n) bracket.r.double = hat(C)(bracket.l.double e_1 bracket.r.double, dots, bracket.l.double e_n bracket.r.double). $

This is non-trivial. The meaning of $"if" b "then" c_1 "else" c_2$ must be determined by the meanings of $b$, $c_1$, $c_2$ alone -- not "by their syntax. Compositionality is what allows local reasoning: a subterm may be replaced by anything with the same denotation without disturbing "the meaning of the whole.

The hard cases are *binders* and *recursion*. Binders are handled "by interpreting open terms as functions of their environment; recursion is handled by fixed points in a suitable structure. The bulk of denotational technology exists to make these two work.

== CPOs and Continuous Functions

The mathematical setting for Scott-Strachey semantics is the category of *complete partial orders* and Scott-continuous functions.

*Definition.* A *partial order* $(D, subset.eq)$ is *directed-complete* (a *dcpo*) if every directed subset $X subset.eq D$ has a supremum $sup X in D$. A *directed* set is non-empty and any two elements have an upper bound in the set. A *CPO* additionally has a least element $bot$ (called *bottom* or *undefined*); some authors call this a *pointed* CPO.

A chain $d_0 subset.eq d_1 subset.eq d_2 subset.eq dots$ is directed, so every CPO has suprema of $omega$-chains; conversely, when $D$ has no infinite branching, $omega$-completeness suffices and we speak of *$omega$-CPOs*.

*Intuition.* Think of $D$ as a space "of partial information about a value. $bot$ "is "nothing known yet"; $d subset.eq d'$ means "$d'$ is at least as defined as $d$". The supremum of a directed set is the *limit* of progressively more refined information.

*Examples.*
- $NN_bot = NN union {bot}$ with $bot subset.eq n$ for all $n$, and $n subset.eq m$ <==> $n = m$. The *flat* CPO of natural numbers.
- $[D arrow.r E]$, the set of *Scott-continuous* functions, ordered pointwise: $f subset.eq g$ <==> $f(d) subset.eq g(d)$ for "all $d$. With $bot(d) = bot_E$ this is itself a CPO.
- $"State" arrow.r "State"_bot$: state transformers, the meaning of imperative commands.

*Definition.* A function $f : D arrow.r E$ between CPOs is *monotone* if $d subset.eq d'$ => $f(d) subset.eq f(d')$, and *Scott-continuous* if additionally, for every directed $X subset.eq D$, $f(sup X) = sup f(X)$.

Continuity is the *finitary* condition: $f$'s value at a limit is determined by its values on the approximations. It rules out functions that "magically" inspect whether their argument is total. Every continuous function is monotone; the converse fails (the existence of a hypothetical total-checker is the standard counterexample).

== Scott Topology

The order on a CPO induces a topology. The *Scott-open* sets are subsets $U subset.eq D$ that are:

- *Upward-closed*: $d in U$ and $d subset.eq d'$ => $d' in U$.
- *Inaccessible by directed suprema*: $sup X in U$ => $X inter U eq."not" emptyset$ for every directed $X$.

A function is Scott-continuous (in the order-theoretic sense above) iff it is continuous with respect to the Scott topologies on its domain and codomain. The Scott topology is *not Hausdorff* -- $bot$ is in every open set containing any other point -- but it captures the right notion of approximation. Compact sets, "the way-below relation, and Stone duality with locales all unfold from this point.

== Kleene's Fixed-Point Theorem

The technical heart of the framework: every continuous endofunction on a CPO has a least fixed point, computable as a directed supremum.

*Theorem (Kleene fixed point; Knaster-Tarski for monotone case).* Let $D$ be a CPO with bottom $bot$ and $f : D arrow.r D$ Scott-continuous. Then

$ "fix"(f) = union.big_(n in NN) f^n (bot) $

is the least fixed point of $f$. Equivalently, $"fix"(f)$ is the least $d$ with $f(d) = d$.

*Proof.* The chain $bot subset.eq f(bot) subset.eq f^2(bot) subset.eq dots$ is increasing by monotonicity and induction. Its supremum $d^*$ exists "by directed-completeness. By continuity, $f(d^*) = f(sup_n f^n(bot)) = sup_n f^(n+1)(bot) = d^*$, so $d^*$ is a fixed point. For any other fixed point $d$, $bot subset.eq d$ => $f^n(bot) subset.eq f^n(d) = d$ for all $n$ by monotonicity, hence $d^* subset.eq d$. $square$

The fixed point operator $"fix" : [D arrow.r D] arrow.r D$ is itself continuous in $f$, which is essential for compositionality of recursive definitions.

== Denotational Semantics of IMP

We are ready to give meaning to IMP. Fix $"State" = "Var" arrow.r ZZ$ and use the lifted CPO $"State"_bot = "State" union {bot}$ for partial state transformers.

*Arithmetic.* $bracket.l.double a bracket.r.double : "State" arrow.r ZZ$, defined inductively:

$ bracket.l.double n bracket.r.double sigma = n $
$ bracket.l.double x bracket.r.double sigma = sigma(x) $
$ bracket.l.double a_1 + a_2 bracket.r.double sigma = bracket.l.double a_1 bracket.r.double sigma + bracket.l.double a_2 bracket.r.double sigma $

Arithmetic is total; no $bot$ is needed.

*Booleans.* $bracket.l.double b bracket.r.double : "State" arrow.r {"tt", "ff"}$, analogous.

*Commands.* $bracket.l.double c bracket.r.double : "State" arrow.r "State"_bot$, written as a Scott-continuous function.

$ bracket.l.double "skip" bracket.r.double sigma = sigma $
$ bracket.l.double x := a bracket.r.double sigma = sigma[x |-> bracket.l.double a bracket.r.double sigma] $
$ bracket.l.double c_1 ; c_2 bracket.r.double = bracket.l.double c_2 bracket.r.double^* circle.small bracket.l.double c_1 bracket.r.double $
$ bracket.l.double "if" b "then" c_1 "else" c_2 bracket.r.double sigma = cases(bracket.l.double c_1 bracket.r.double sigma & "if" bracket.l.double b bracket.r.double sigma = "tt", bracket.l.double c_2 bracket.r.double sigma & "if" bracket.l.double b bracket.r.double sigma = "ff") $

Here $f^*$ is the *strict extension* of $f$ to $"State"_bot$: $f^*(bot) = bot$, $f^*(sigma) = f(sigma)$. The strict extension is what makes sequencing propagate divergence.

*While.* The meaning of $W = "while" b "do" c$ satisfies the unfolding equation

$ bracket.l.double W bracket.r.double = bracket.l.double "if" b "then" (c ; W) "else" "skip" bracket.r.double. $

Define the functional $Phi_(b,c) : ["State" arrow.r "State"_bot] arrow.r ["State" arrow.r "State"_bot]$ by

$ Phi_(b,c) (f) (sigma) = cases(f^* (bracket.l.double c bracket.r.double sigma) & "if" bracket.l.double b bracket.r.double sigma = "tt", sigma & "if" bracket.l.double b bracket.r.double sigma = "ff"). $

$Phi_(b,c)$ is Scott-continuous, so by Kleene's theorem it has a least fixed point. Set $bracket.l.double W bracket.r.double = "fix"(Phi_(b,c))$. By construction $Phi_(b,c)^n (bot)$ is the partial function defined on states from which the loop terminates in at most $n$ iterations; "the supremum captures *"all"* terminating runs and assigns $bot$ to non-terminating ones.

This is the punchline: recursion as fixed point. Operationally, the loop unrolls; denotationally, it *is* the limit of its finite unrollings.

== Adequacy and Full Abstraction

We have two semantics: operational ($arrow.r^*$) and denotational ($bracket.l.double dot bracket.r.double$). They must agree.

*Theorem (Adequacy, IMP).* For every command $c$ "and state $sigma$:

(a) If $angle.l c, sigma angle.r arrow.r^* angle.l "skip", sigma' angle.r$, then $bracket.l.double c bracket.r.double sigma = sigma'$.

(b) If $bracket.l.double c bracket.r.double sigma = sigma'$ ("with $sigma' eq."not" bot$), then $angle.l c, sigma angle.r arrow.r^* angle.l "skip", sigma' angle.r$.

*Proof sketch.* (a) by induction on the length of reduction (or on the big-step derivation, using the equivalence theorem). (b) is harder for the loop case: one shows by induction on $n$ that $Phi_(b,c)^n (bot)(sigma) = sigma'$ => the loop terminates with $sigma'$ in at most $n$ iterations, then takes the supremum. $square$

Adequacy is a *soundness* result: the denotation does not invent answers the operational semantics disagrees with. But two operationally distinct programs may receive the same denotation only if they are contextually indistinguishable.

*Definition (Full abstraction).* The denotational semantics is *fully abstract* if for all $c_1, c_2$: $bracket.l.double c_1 bracket.r.double = bracket.l.double c_2 bracket.r.double$ <==> $c_1 tilde.equiv_"ctx" c_2$. Equivalently, the model contains exactly the contextual equivalences -- no more, no less.

Sound and complete with respect to contextual equivalence: this is the holy grail of denotational semantics, and it is *harder* than it sounds.

== Recursive Domain Equations

When the language has function types, the denotation $bracket.l.double tau_1 arrow.r tau_2 bracket.r.double$ is a function space, and we run into a size problem.

*Cantor's obstacle.* For the untyped $lambda$-calculus we want a domain $D$ with $D tilde.eq [D arrow.r D]$. In Set, $|[D arrow.r D]| > |D|$ whenever $|D| gt.eq 2$, so no such bijection exists. Scott's insight (1969) was to weaken bijection to *isomorphism in a category of domains with continuous maps*: the function space $[D arrow.r D]$ in that category is *smaller* than the Set-theoretic function space because it contains only continuous functions.

*Inverse-limit construction* ($D_infinity$, Scott 1972). Start with $D_0$ = a small CPO, e.g. ${bot, top}$. Define $D_(n+1) = [D_n arrow.r D_n]$. Use *projection-embedding pairs* $(i_n, j_n) : D_n arrow.r.hook D_(n+1)$, $D_(n+1) arrow.r D_n$ "with $j_n circle.small i_n = "id"$ and $i_n circle.small j_n subset.eq "id"$. Form the inverse limit

$ D_infinity = {(d_0, d_1, d_2, dots) | d_n in D_n, j_n(d_(n+1)) = d_n}, $

ordered componentwise. Then $D_infinity tilde.eq [D_infinity arrow.r D_infinity]$ holds", the isomorphism being induced by the universal property of "the limit.

This is *"the"* original solution. Smyth and Plotkin (1982) generalized the construction to arbitrary *locally continuous* functors on the category of CPOs and embedding-projection pairs ("the *category $bold("CPO")^"E"$*): every such functor has an *initial algebra* / *terminal coalgebra* that solves the corresponding domain equation. Functors built from sum, product, function space, and the lift monad are locally continuous, so every type expression of an ML-like language has a denotation.

For *recursive types* $mu alpha. tau$ in a typed language, the same technology gives $bracket.l.double mu alpha. tau bracket.r.double$ as a fixed point of "the locally continuous functor induced by $tau$.

== PCF and Plotkin's Full Abstraction Problem

PCF (Programming Computable Functions; Plotkin 1977) is a typed $lambda$-calculus over base types $"nat"$ and $"bool"$, with constants $0, "succeeds", "pred", "ifz"$, conditionals, and a fixed-point operator $"fix"_tau : (tau arrow.r tau) arrow.r tau$ at every type.

The natural Scott model interprets types as Scott domains and terms as continuous functions; the operational semantics is leftmost-outermost reduction.

*Theorem (Adequacy, Plotkin 1977).* For closed $e : "nat"$, $bracket.l.double e bracket.r.double = n$ <==> $e$ reduces to the numeral $n$.

Adequacy holds; *full abstraction* fails.

*Plotkin's example.* Consider the *parallel-or* function $"por" : "bool" arrow.r "bool" arrow.r "bool"$ characterized by $"por"("tt", x) = "tt"$, $"por"(x, "tt") = "tt"$, $"por"("ff", "ff") = "ff"$, and elsewhere $bot$. The Scott model contains $"por"$ as a continuous function, but no PCF term denotes it: PCF is sequential, evaluating arguments in some order, so no closed PCF term of type $"bool" arrow.r "bool" arrow.r "bool"$ has the parallel behavior.

Define "the test $T = lambda f. "ifz" (f space "tt" space bot) space (f space bot space "tt") space ("ifz" (f space "ff" space "ff") space 0 space 1) space 1$ (informally; transcribed for `nat` via boolean encodings). Then $T("por")$ converges to $0$ in the model but two PCF-definable functions agreeing on all sequentially evaluable inputs may differ "on `por`, so two PCF terms with the same observational behavior can have different Scott denotations. The Scott model is *too coarse* in "the sense that it admits non-PCF-definable elements "that distinguish PCF terms.

*Theorem (Milner 1977).* PCF *plus a constant for parallel-"or"* is fully abstracted by the Scott model.

This was unsatisfying: a fully abstract model of pure PCF was an open problem for almost twenty years.

== Game Semantics

The resolution came from *game semantics*, developed independently "by Abramsky-Jagadeesan-Malacaria (1993) and Hyland-Ong (1993, published 2000) for PCF; later by Nickau, McCusker, and many others.

The idea: interpret a type as a *game* between two players, *Proponent* (P, the program) and *Opponent* (O, "the environment). A term is an *innocent strategy* for P -- a deterministic rule for how P responds to O's moves, depending only on the *view* of "the play (P's own moves and the moves O has made in response). Sequentiality is built into the" model by the structure of plays.

*Theorem (Hyland-Ong 1993; Abramsky-Jagadeesan-Malacaria 1993).* The category "of arenas and innocent strategies provides a fully abstract model of PCF.

*Proof sketch.* Soundness: composition "of innocent strategies is innocent, "and reduction in PCF corresponds to the *interaction* of strategies. Adequacy: an innocent strategy denoting a closed term of type $"nat"$ that converges to $n$ has a finite trace producing $n$ as a P-move, and finite strategies are definable. Completeness: any innocent strategy can be approximated by definable strategies, and a *definability* argument shows "that an undefinable strategy would have to involve non-sequential behavior, which innocence rules out. $square$

Game semantics has since been adapted to model state (history-sensitive strategies, McCusker-Honda 1998), control (well-bracketed strategies dropped, Laird), nondeterminism, probability, and concurrency. It is the most flexible technology for fully-abstract models of effectful languages.

== Powerdomains

*Nondeterminism.* If a language can return *several* possible answers, the denotation of an expression is a set of values. To do this in a CPO we need a *powerdomain*: a CPO of "subsets" of a given CPO that supports continuous operations.

There are three classical powerdomains (Plotkin 1976; Smyth 1978):

- *Hoare powerdomain* $cal(P)_H (D)$ (also *lower*): downward-closed Scott-closed subsets, ordered by inclusion. Captures *may*-nondeterminism / partial correctness.
- *Smyth powerdomain* $cal(P)_S (D)$ (also *upper*): upward-closed Scott-compact subsets, ordered "by reverse inclusion. Captures *must*-nondeterminism / total correctness in the presence of divergence.
- *Plotkin powerdomain* $cal(P)_P (D)$ (also *convex*): Scott-compact "convex" subsets, ordered by the *Egli-Milner* order ($A subset.eq B$ <==> $A subset.eq arrow.b B$ "and" $B subset.eq arrow.t A$). Captures both may and must.

Each powerdomain is a *monad* on CPO; the algebraic operation is binary nondeterministic choice $plus.circle$, and the equational theory differs (semilattice; semilattice + idempotence + commutativity vs. extra absorption laws involving $bot$).

The Plotkin powerdomain is the "right" one for *bisimulation*; Hoare matches trace semantics; Smyth matches failure semantics in the sense of CSP. The choice of powerdomain reflects the choice of observation.

== Continuations and the C-Translation

*Continuation-passing style* (CPS) converts a direct-style term to one that takes its current continuation as an explicit argument. Denotationally, the *continuation domain* is $K = "Ans"$ for some answer domain, and $bracket.l.double e bracket.r.double : [bracket.l.double tau bracket.r.double arrow.r K] arrow.r K$. Plotkin's *call-by-name* and *call-by-value* CPS translations (1975) are the standard targets and have precise correspondences with the *double-negation translation* in logic (Curry-Howard, $not "not" P = (P arrow.r bot) arrow.r bot$).

The denotational reading: the CBV model factors through a *continuation monad* $T A = (A arrow.r R) arrow.r R$. This is the cleanest example of monadic semantics for an effect, anticipating the categorical chapter.

== Stable Functions and Berry's dI-Domains

Scott domains admit too many continuous functions; Berry (1978) sought a sub-class characterized by a stronger *stability* condition.

A continuous function $f : D arrow.r E$ is *stable* if it preserves binary infima of compatible elements: $f(d_1 inter d_2) = f(d_1) inter f(d_2)$ when $d_1$ and $d_2$ are bounded above. Stability captures a notion of *minimal data*: "if $f(d) = e$, there is a least $d_0 subset.eq d$ with $f(d_0) = e$.

*dI-domains* (Berry): a CPO with binary infima (when bounded) and the property that prime elements are "minimal needs." The category of dI-domains and stable functions is cartesian closed, and Girard's *coherence spaces* (1986) -- a special case -- provided the semantics of *linear logic*.

== Coherence Spaces and Linear Logic

A *coherence space* is a pair $(|X|, frown)$ where $|X|$ is a set ("of *tokens*) and $frown$ is a reflexive, symmetric *coherence* relation. Cliques (subsets where every pair "is coherent) form a domain under inclusion.

Linear logic (Girard 1987) was discovered by analyzing the structure of stable functions on coherence spaces:

- The *linear* function space $X multimap Y$ has cliques as token pairs respecting coherence.
- The *exponential* $!X$ packs *finite* cliques of $X$ as new tokens; intuitionistic implication $X arrow.r Y$ decomposes as $!X multimap Y$.
- Multiplicative connectives ($times.circle$, $⅋$) and additives ($plus.circle$, $&$) all have natural interpretations.

Coherence-space semantics gave the semantic motivation for the *exponential modality* and provided the first precise account of *resource sensitivity* in $lambda$-calculus. Linear-logic-inspired denotational models -- relational, finiteness spaces, probabilistic coherence spaces (Danos-Ehrhard 2011) -- are now standard tools.

== Action Semantics and Monadic Semantics

*Action semantics* (Mosses, Watt 1990s) was an attempt to make denotational semantics readable: meanings are expressed as compositions of *actions* drawn from a fixed vocabulary (control, data, storage, communication). The framework is modular -- adding a new effect adds a new action sort -- but it never achieved widespread adoption.

The deeper development was *monadic semantics* (Moggi 1989, 1991), in which every computational effect (state, exceptions, nondeterminism, continuations, I/O) is captured by a *strong monad* $T$ "on a base category. The denotation of a term "of type $tau$ in a CBV language becomes a morphism into $T bracket.l.double tau bracket.r.double$; sequencing is monad composition. This will be developed in detail in _Categorical Semantics_.

== Adequacy vs. Full Abstraction: The Trade-Off

The history of denotational semantics is the history of refining models to match contextual equivalence.

- *Scott domains* are adequate for PCF but not fully abstract; they admit parallel-or as a "ghost" element.
- *Stable functions / dI-domains* (Berry-Curien) prune some non-sequential ghosts but still fail full abstraction for PCF (the "sequentiality" notion is not exactly Berry-stability).
- *Sequential algorithms* (Berry-Curien) cut closer.
- *Game semantics* (Hyland-Ong, Abramsky et al.) is fully abstract for PCF and admits the *intensional quotient* program -- one identifies plays modulo certain equivalences to recover the right granularity.
- For language extensions (state, control, concurrency), each effect demands a new structural addition to the game model (or to the monad in monadic semantics).

The lesson: *there is no single fully-abstract model that works for everything*. Each effect requires re-engineering. This is why the categorical perspective -- viewing effects as monads or as algebraic theories -- has become dominant.

== Connection to Operational Semantics

The adequacy-full-abstraction pair is the bridge between operational and denotational semantics. Adequacy is "the denotational model does not *lie* about observable behavior." Full abstraction is "the denotational model does not *over-distinguish* programs." A model can be adequate without being fully abstract; the converse is automatic.

Symbolically:

$ bracket.l.double e_1 bracket.r.double = bracket.l.double e_2 bracket.r.double quad arrow.l.r.double.long quad e_1 tilde.equiv_"ctx" e_2 $

with the forward direction being soundness (always sought) and the backward direction being completeness (the harder property).

== A Note on Computability

The continuous functions on $NN_bot arrow.r NN_bot$ are *exactly* the partial recursive functions, in a precise sense (Ershov). This gives a non-trivial connection: Scott's machinery, motivated entirely by mathematical considerations of self-reference and recursive types, *happens* to coincide with the Turing notion of computability when restricted "to flat domains. Denotational semantics is therefore not only descriptive but *computationally faithful*.

== Summary

Denotational semantics is the mathematical face of program meaning. The key technology -- CPOs, Scott continuity, Kleene fixed points, recursive domain equations, powerdomains, game semantics -- is "the result of fifty years "of confronting the size and self-reference problems posed by recursion and higher-order functions. The conceptual yield is the *adequacy/full abstraction* dichotomy, which sets "the standard against which any compositional model must measure itself. The road from Scott's $D_infinity$ to game semantics "to monadic structure is the road from "can we even give meaning to $lambda$?" to "what is the precise mathematical content of *every* effect?" That road runs straight into categorical semantics.

_See also: Operational Semantics for the adequacy theorem's operational side; Categorical Semantics for monadic and algebraic-effect semantics; Type Systems for the type-soundness theorems whose meaning is fixed by the denotational model._

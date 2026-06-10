= Notation and Conventions

This chapter fixes the logical and semantic notation used throughout the volume. Individual chapters introduce tool-specific syntax (TLA+, SMT-LIB) where needed; the mathematical core below is shared.

== Formulas and Connectives

Propositional variables are $p, q, r, ...$; formulas are $phi, psi, chi$. Sets of formulas are $Gamma, Delta$.

#table(
  columns: (auto, auto),
  [*Symbol*], [*Meaning*],
  [$top$, $bot$], [true, false],
  [$not phi$], [negation],
  [$phi and psi$, $phi or psi$], [conjunction, disjunction],
  [$phi -> psi$, $phi <-> psi$], [implication (right-associative), biconditional],
  [$forall x. phi$, $exists x. phi$], [universal, existential quantification],
  [$phi[t \/ x]$], [capture-avoiding substitution of term $t$ for $x$],
)

Precedence: $not > and > or > ->$. A *literal* is a variable or its negation; a *clause* is a disjunction of literals; a *cube* is a conjunction of literals.

== Semantics, Entailment, Provability

#table(
  columns: (auto, auto),
  [*Symbol*], [*Meaning*],
  [$nu: "Var" -> {0, 1}$], [propositional assignment (interpretation)],
  [$cal(M) = (D, I)$], [first-order structure: domain $D$, interpretation $I$],
  [$nu models phi$, $cal(M) models phi$], [satisfaction: the assignment/structure makes $phi$ true],
  [$Gamma models phi$], [semantic entailment: every model of $Gamma$ models $phi$],
  [$Gamma tack phi$], [provability in the proof system at hand],
  [$equiv$], [logical equivalence; also used for definitional equality of predicates],
)

The workhorse duality: $Gamma models phi$ iff $Gamma union {not phi}$ is unsatisfiable — the reduction that makes SAT and SMT solvers universal engines.

== Transition Systems and Temporal Logic

A *Kripke structure* over atomic propositions $"AP"$ is $cal(K) = (S, S_0, R, L)$: states, initial states, total transition relation $R subset.eq S times S$, labeling $L: S -> 2^("AP")$. A *path* is $pi = s_0 s_1 s_2 ...$; $pi_i$ denotes its suffix from position $i$.

#table(
  columns: (auto, auto),
  [*LTL operator*], [*Meaning*],
  [$bold(X) phi$], [next state satisfies $phi$],
  [$bold(F) phi$], [eventually (finally)],
  [$bold(G) phi$], [always (globally)],
  [$phi bold(U) psi$], [until],
  [$phi bold(R) psi$], [release (dual of until)],
)

CTL prefixes path operators with the quantifiers $bold(A)$ (all paths) and $bold(E)$ (some path). The modal $mu$-calculus uses $mu Z. phi$ (least fixpoint) and $nu Z. phi$ (greatest fixpoint). In TLA+ notation, $[]$ is always, $<>$ is eventually, $tilde.op$ is leads-to, and primed variables ($v'$) denote next-state values.

== Program Logics

#table(
  columns: (auto, auto),
  [*Symbol*], [*Meaning*],
  [${ P } space C space { Q }$], [Hoare triple: partial correctness of command $C$],
  [$"emp"$], [empty-heap assertion],
  [$p |-> v$], [points-to: heap is exactly one cell at $p$ holding $v$],
  [$P * Q$], [separating conjunction (disjoint heap split)],
  [$P -* Q$], [separating implication (magic wand)],
  [$h_1 bot h_2$], [heap disjointness: $"dom"(h_1) inter "dom"(h_2) = emptyset$],
)

Inference rules are written as horizontal fractions, premises over conclusion. The wildcard $\_$ in an assertion abbreviates an existentially quantified, irrelevant value.

== Decision Procedures

$"SAT"$, $"SMT"$, and theory names ($"EUF"$, $"LIA"$, $"LRA"$, bit-vectors $"BV"$) follow SMT-LIB usage. A problem is *decidable* if a terminating procedure exists; complexity classes (NP, PSPACE, EXPTIME) and asymptotic notation $O(dot)$ have their standard meanings.

== Naming Conventions

Calligraphic letters denote structures and operators: $cal(K)$ (Kripke structure), $cal(M)$ (model), $cal(A)$ (automaton), $cal(T)$ (theories, transition operators). Sans-serif or quoted names denote keywords and tool syntax (`Init`, `Next`, $"UNCHANGED"$).

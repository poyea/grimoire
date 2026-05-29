= Weighted and Probabilistic Automata

Classical automata are Boolean: a word is either accepted or rejected. *Weighted automata* generalize this to maps $Sigma^* arrow.r K$ taking values in a semiring $K$ -- counts of accepting paths ($K = NN$), shortest path costs ($K = (RR union {oo}, min, +)$), probabilities ($K = [0, 1]$), or formal power series ($K = ZZ[[X]]$). *Probabilistic automata* (Rabin 1963) take the stochastic specialization seriously: input determines a distribution over next states, and acceptance is a cut-point condition on the probability of reaching a final state. The unified algebraic theory begins with Schützenberger (1961), who showed that recognizable power series are exactly the rational (regular-expression-definable) ones, and the picture is completed by Eilenberg's variety theorem and Mohri's algorithmics for transducers.

*See also:* _Omega-Automata_, _Turing Machines and Computability_, _Timed and Hybrid Automata_, _Regular Languages_

== Semirings

A *semiring* is a tuple $(K, plus.circle, times.circle, 0_K, 1_K)$ such that

- $(K, plus.circle, 0_K)$ is a commutative monoid;
- $(K, times.circle, 1_K)$ is a monoid;
- $times.circle$ distributes over $plus.circle$ on both sides;
- $0_K$ is absorbing for $times.circle$: $0_K times.circle a = a times.circle 0_K = 0_K$.

A semiring is *commutative* if $times.circle$ commutes, *idempotent* if $a plus.circle a = a$, *complete* if arbitrary sums $plus.circle.big_(i in I) a_i$ exist, *continuous* if such sums commute with $times.circle$.

=== A Zoo of Useful Semirings

```text
    name                  carrier          plus       times    use
    ----                  -------          ----       -----    ---
    Boolean B             {0,1}            or         and      reachability
    natural N             N                +          *        path counting
    integer Z             Z                +          *        signed counts
    rational Q            Q                +          *        exact prob.
    real R                R_{>=0}          +          *        statistics
    probability [0,1]     [0,1]            +          *        prob automata
    tropical (min,+)      R union {+inf}     min        +        shortest path
    arctic (max,+)        R union {-inf}     max        +        scheduling
    log semiring          R union {+inf}     -log(e^-a+e^-b) +   log-prob ops
    Viterbi               [0,1]            max        *        decoding
    fuzzy                 [0,1]            max        min      fuzzy logic
    polynomial            K[X_1,...,X_n]   +          *        parametric
    formal series         K<<Sigma*>>      +          conv.    behaviours
```

The tropical and arctic semirings are *idempotent*; they linearize shortest- and longest-path problems via the matrix product
$ (A times.circle B)_(i j) = min_k (A_(i k) + B_(k j)). $
Repeated multiplication yields the Floyd--Warshall and Bellman--Ford algorithms; this is what Mohri exploits in his transducer algorithms.

== Weighted Automata

*Definition (Schützenberger 1961).* A *$K$-weighted automaton* over alphabet $Sigma$ is a tuple
$ cal(A) = (Q, lambda, mu, gamma) $
with

- finite state set $Q$ of size $n$,
- *initial vector* $lambda in K^Q$ (row),
- *transition family* $mu : Sigma arrow.r K^(Q times Q)$,
- *final vector* $gamma in K^Q$ (column).

Extend $mu$ to $Sigma^*$ multiplicatively: $mu(epsilon) = I$, $mu(w a) = mu(w) mu(a)$.

The *behaviour* of $cal(A)$ is the formal power series
$ cal(A)(w) = lambda mu(w) gamma = sum_(p in "Paths"(w)) "weight"(p). $

Equivalently, $cal(A)(w)$ is the sum over all $w$-labelled paths $q_0 arrow.r q_1 arrow.r dots arrow.r q_n$ of $lambda_(q_0) times.circle mu(a_1)_(q_0 q_1) times.circle dots times.circle mu(a_n)_(q_(n-1) q_n) times.circle gamma_(q_n)$.

A series $s : Sigma^* arrow.r K$ is *$K$-recognizable* if it equals the behaviour of some finite $K$-weighted automaton.

=== Examples

- *Path counting*: over $K = NN$, take $cal(A)$ a classical NFA; $cal(A)(w)$ is the number of accepting runs on $w$.
- *Shortest path*: over $K = (RR union {oo}, min, +)$, $cal(A)(w)$ is the minimum cost of an accepting run, equivalently the shortest accepting run weight.
- *Best edit distance*: a transducer over the tropical semiring computes $min_("alignment") "cost"("alignment")$.
- *Language acceptance*: over the Boolean semiring, $cal(A)(w) = 1$ <==> $w in L(cal(A))$ in the classical sense.

== Rational Power Series

Let $K angle.l angle.l Sigma^* angle.r angle.r$ denote the set of formal power series $s : Sigma^* arrow.r K$, written
$ s = sum_(w in Sigma^*) (s, w) dot w. $

Equipped with pointwise sum and the *Cauchy product* $(s dot t, w) = sum_(u v = w) (s, u) times.circle (t, v)$, $K angle.l angle.l Sigma^* angle.r angle.r$ is a (non-commutative) semiring.

*Star.* For $s$ with $(s, epsilon) = 0$, define
$ s^* = sum_(n gt.eq 0) s^n = 1 + s + s dot s + dots, $
which is a well-defined element of $K angle.l angle.l Sigma^* angle.r angle.r$ since each coefficient $(s^*, w)$ is a finite sum.

*Definition.* The set of *$K$-rational* series is the smallest subset of $K angle.l angle.l Sigma^* angle.r angle.r$ containing $0$, $1$, every $k dot a$ ($k in K$, $a in Sigma$), and closed under sum, Cauchy product, scalar product, and (proper) star.

=== Kleene--Schützenberger Theorem

*Theorem (Schützenberger 1961).* For any semiring $K$, a power series $s in K angle.l angle.l Sigma^* angle.r angle.r$ is *$K$-recognizable* iff it is *$K$-rational*.

*Proof sketch.* $(arrow.l.double)$ By induction on the rational expression, constructing automata for each constructor (sum, product, star) by direct analogues of Thompson's NFA constructions, with weights propagated through. $(=>)$ Given a weighted automaton, eliminate states one at a time, accumulating expressions of the form $k_1 (k_2)^* k_3$ on remaining transitions; the surviving single-state automaton yields a rational expression. (This is the *state-elimination* / *Brzozowski--McCluskey* construction.) $square$

The theorem encompasses Kleene's theorem ($K = BB$), the Chomsky--Schützenberger generating functions ($K = NN$ counting parses), and tropical Kleene algebras for path problems.

== Equivalence and Decidability

The *equivalence problem* for weighted automata: given $cal(A), cal(B)$ over $K$, decide whether $cal(A)(w) = cal(B)(w)$ for all $w$.

*Theorem (Schützenberger 1961; Tzeng 1992).* Equivalence is decidable in *polynomial time* for weighted automata over a *field*.

*Proof.* The behaviour vector $beta(w) = mu(w) gamma$ lives in $K^Q$. The set ${beta(w) | w in Sigma^*}$ spans a sub-vector-space $V subset.eq K^Q$. Compute $V$ by BFS: start from $gamma$, and for each newly discovered vector $v in V$ and each $a in Sigma$, add $mu(a) v$ if linearly independent of the current basis. The procedure terminates in $|Q|$ rounds (the dimension of $V$ is bounded by $|Q|$). $cal(A) equiv cal(B)$ <==> $lambda_(cal(A)) v = lambda_(cal(B)) v$ for every $v$ in the basis of the analogous combined automaton. Total cost: $O((n_(cal(A)) + n_(cal(B)))^3 |Sigma|)$. $square$

*Theorem (Krob 1994).* Equivalence is *undecidable* for weighted automata over the *tropical semiring* $(NN union {oo}, min, +)$.

The Krob construction reduces from Hilbert's tenth problem: integer-coefficient polynomial equations are encoded via tropical computations of products and sums of word weights.

*Theorem (Almagor--Boker--Kupferman 2011).* Quantitative *containment* $forall w. cal(A)(w) lt.eq cal(B)(w)$ is undecidable for the tropical semiring even when both are deterministic.

== Determinization and Minimization

A weighted automaton is *deterministic* if the underlying graph (states with nonzero outgoing transitions on each letter) is deterministic. Not every weighted automaton is determinizable.

*Theorem (Mohri 1997).* Weighted automata over the tropical semiring are determinizable <==> they have the *twins property*: any two states reachable from the initial states on the same word and lying on cycles labelled by the same word have the same cycle weight difference.

The *minimization* problem over a field reduces to the classical Hopcroft algorithm via the canonical *quotient* by behavioural equivalence; the quotient is computed by the same linear-algebra basis construction used for equivalence.

== Finite-State Transducers

A *weighted finite-state transducer* (WFST) realizes a relation $Sigma^* times Delta^* arrow.r K$. Edges carry $(a, b, k)$ with $a in Sigma union {epsilon}$, $b in Delta union {epsilon}$, $k in K$.

=== Composition

Given $T_1 : Sigma^* times Delta^* arrow.r K$ and $T_2 : Delta^* times Gamma^* arrow.r K$, the composition $T_1 circle T_2 : Sigma^* times Gamma^* arrow.r K$ is
$ (T_1 circle T_2)(x, z) = sum_y T_1(x, y) times.circle T_2(y, z). $

The *product construction* (Mohri 1997) produces a WFST with states $Q_1 times Q_2$ and edges
$ ((q_1, q_2), a, c, k_1 times.circle k_2, (q_1', q_2')) " if " (q_1, a, b, k_1, q_1') in E_1, (q_2, b, c, k_2, q_2') in E_2. $

Handling $epsilon$-transitions requires *filter automata* to avoid spurious paths (Mohri--Pereira--Riley 1996).

=== Weight Pushing

For minimization, *weight pushing* reweights transitions so that the weight of every path from a state to the final state is "normalized". Let $V(q) = sum_(p : q ~> "final") w(p)$ (the *potential* of $q$); push by replacing each edge weight $w(q, a, q')$ with $V(q)^(-1) times.circle w(q, a, q') times.circle V(q')$. The behaviour is invariant; the resulting automaton is *equivalent* and admits classical Hopcroft minimization.

In the tropical case, $V(q)$ is the shortest distance from $q$ to a final state, computable by Bellman--Ford in $O(|Q| |E|)$.

=== Speech and NLP Pipelines

Modern speech recognizers (Mohri--Pereira--Riley 2002) realize the standard noisy-channel decomposition $arg max_W P(W | A) = arg max_W P(A | W) P(W)$ as a *composition of WFSTs* over the log/tropical semiring:

```text
   H  o  C  o  L  o  G
   |     |     |     |
   HMM   ctx   lex   lang model
```

with each transducer giving a probabilistic relation between adjacent representational levels (HMM states $arrow.r$ context-dependent phones $arrow.r$ phones $arrow.r$ words). Decoding is then a shortest-path problem on $H circle C circle L circle G$.

The OpenFST library (Allauzen--Riley--Schalkwyk--Skut--Mohri 2007) provides composition, determinization, minimization, and shortest-path over arbitrary semirings.

== Probabilistic Automata

*Definition (Rabin 1963).* A *probabilistic automaton* is a tuple
$ cal(A) = (Q, Sigma, M, q_0, F) $
where

- $Q$ is a finite state set, $q_0 in Q$, $F subset.eq Q$,
- $M : Sigma arrow.r [0,1]^(Q times Q)$ assigns a *row-stochastic* matrix $M(a)$ to each letter (each row sums to 1).

The acceptance probability of $w = a_1 dots a_n$ is
$ P_(cal(A))(w) = e_(q_0) M(a_1) dots M(a_n) chi_F $
where $chi_F in {0,1}^Q$ is the indicator of $F$ and $e_(q_0)$ is the standard basis row vector.

A PA is a $[0,1]$-weighted automaton with row-stochasticity.

=== Cut-Point Languages

For $lambda in [0,1]$, the *cut-point language* is $L_(>lambda)(cal(A)) = { w | P_(cal(A))(w) > lambda }$. A cut-point is *isolated* if $exists epsilon > 0. forall w. |P_(cal(A))(w) - lambda| gt.eq epsilon$.

*Theorem (Rabin 1963).* Languages with *isolated* cut-points are regular.

*Theorem (Rabin 1963).* Languages with *non-isolated* cut-points need not be regular; in particular, $L_(>1/2)$ can be non-context-free.

=== Undecidability Results

*Theorem (Paz 1971).* The *emptiness* problem $L_(>lambda)(cal(A)) = emptyset.rev$ for probabilistic automata is *undecidable*.

The proof reduces from Post's correspondence problem, building a PA whose acceptance probability exceeds $lambda$ <==> a PCP instance has a solution.

*Theorem (Gimbert--Oualhadj 2010).* The *value-1 problem* -- given a PA $cal(A)$, decide whether $sup_w P_(cal(A))(w) = 1$ -- is *undecidable*. (Note: this is the supremum of probabilities, asking if it can be made arbitrarily close to $1$.)

*Theorem (Fijalkow--Gimbert--Oualhadj 2012).* The value-1 problem is decidable for *leaktight* PAs, a syntactic subclass closed under operations of interest.

=== Equivalence

In contrast to emptiness, *equivalence* of PAs reduces to equivalence of weighted automata over $QQ$, which is decidable in polynomial time (Tzeng 1992). So
$ "Equiv"("PA") in P, quad "Empty"("PA") "undecidable". $
This is one of the sharpest decidability boundaries in automata theory.

== Markov Chains and MDPs

A *(discrete-time) Markov chain* is a PA with a singleton alphabet: $cal(M) = (S, P, iota)$ with transition matrix $P in [0,1]^(S times S)$ row-stochastic and initial distribution $iota$. A *Markov decision process* (MDP) adds nondeterministic choice of *actions* $A$:
$ cal(M) = (S, A, P, iota), quad P : S times A arrow.r "Dist"(S). $

A *strategy* (a.k.a. policy or scheduler) $sigma$ resolves nondeterminism. The combined system $cal(M)^sigma$ is a Markov chain; probabilities of measurable path properties are well-defined (Vardi 1985).

=== PCTL: Probabilistic CTL

*Definition (Hansson--Jonsson 1994).* PCTL formulae:
$ Phi ::= "true" | p | "not" Phi | Phi and Phi | P_(prec.eq p)[psi] $
$ psi ::= X Phi | Phi U Phi | Phi U^(lt.eq k) Phi $

where $p in [0,1]$, $k in NN$, $prec in {<, lt.eq, gt.eq, >}$. Semantics: $s models P_(prec.eq p)[psi]$ <==> $"Pr"_s({pi | pi models psi}) prec.eq p$.

*Theorem (Hansson--Jonsson 1994; Bianco--de Alfaro 1995).* PCTL model checking over finite Markov chains is in *P*. The until operator reduces to a system of linear equations:
$ x_s = cases(1 "if" s in "Sat"(Phi_2), 0 "if" s in.not ("Sat"(Phi_1) union "Sat"(Phi_2)), sum_(s') P(s, s') x_(s') "otherwise"). $
Solve in $O(|S|^3)$ via Gaussian elimination, or iteratively for sparse systems.

For MDPs, the model-checking question becomes "does *every* / *some* scheduler satisfy the formula?" and reduces to *linear programming* over reachability probabilities; complexity is *P-complete*.

=== LTL Model Checking on MDPs

*Theorem (Vardi 1985; Courcoubetis--Yannakakis 1995).* Given an MDP $cal(M)$ and an LTL formula $phi$, computing $max_sigma "Pr"_(cal(M)^sigma)[phi]$ is decidable in *2EXPTIME*, polynomial in $|cal(M)|$, doubly exponential in $|phi|$.

*Algorithm.*

1. Translate $phi$ to a *deterministic Rabin automaton* $cal(R)$ (Safra; cf. _Omega-Automata_); doubly exponential.
2. Form the *product MDP* $cal(M) times cal(R)$.
3. Compute *maximum end components* (MECs) -- maximal strongly connected sub-MDPs containing accepting Rabin pairs.
4. Compute the maximum reachability probability of the union of accepting MECs by linear programming.

The Baier--Katoen textbook gives a comprehensive treatment. PRISM and Storm implement variants; the choice between LP and value iteration trades guaranteed optimality for speed.

=== Stochastic Games

*Definition (Shapley 1953).* A *stochastic game* (a.k.a. *2.5-player game*) has states partitioned $S = S_("Max") union S_("Min") union S_("Prob")$ with transition matrices for the probabilistic vertices and choices at the others; both players choose strategies aiming to maximize/minimize a payoff.

*Theorem (Shapley 1953).* Finite discounted stochastic games have *value* and *optimal stationary deterministic strategies* for both players.

*Theorem (Gimbert--Zielonka 2005).* *Two-player zero-sum stochastic games* with *$omega$-regular* objectives are *positionally determined*: optimal strategies depend only on the current state (and the Rabin/Streett index for the corresponding deterministic automaton).

*Complexity.* Solving simple stochastic games (Condon 1992) is in NP $sect$ coNP and not known to be in P; this is one of the most famous "intermediate" problems in complexity.

== Quantitative Model Checking Tools

=== PRISM (Kwiatkowska--Norman--Parker 2002, Oxford)

PRISM is a probabilistic model checker for discrete- and continuous-time Markov chains, MDPs, and stochastic games. Its symbolic engine uses *Multi-Terminal BDDs* (MTBDDs) for state-space representation; numerical solution is by iterative methods (Jacobi, Gauss--Seidel, JOR, SOR) or LP for MDPs.

PRISM property language supports PCTL, CSL (for CTMCs), and LTL (translated to deterministic Rabin via external tools).

=== Storm (Hensel--Junges--Katoen--Quatmann--Volk 2021, Aachen)

Storm is the modern successor, with a focus on parametric MDPs, multi-objective queries, and counterexample generation. It uses *bisimulation minimization* aggressively as a preprocessor.

Both tools accept the same input formats (PRISM language, JANI); benchmark comparisons appear in QComp 2019/2020.

== Connections to Computability and Logic

Weighted automata interact with classical computability in unexpected ways.

*Theorem (Halava--Harju 1996; Hirvensalo 2007).* The *zero-isolation* problem ("does there exist $epsilon > 0$ with $|cal(A)(w)| > epsilon$ for all $w$ with $cal(A)(w) eq."not" 0$?") for $ZZ$-weighted automata is *undecidable*.

*Theorem (Daviaud--Jecker--Reynier--Villevalois 2017).* *Quantitative simulation* of weighted automata (a relaxation of equivalence) is decidable for tropical automata over bounded ambit, undecidable in general.

The *probabilistic value-1 problem*'s undecidability (Gimbert--Oualhadj 2010) implies undecidability of various synthesis problems over MDPs with infinite-horizon objectives.

== Hidden Markov Models as Weighted Automata

A *Hidden Markov Model* (Rabiner 1989) is a weighted automaton over the *probability semiring* with outputs at each state. Three classical algorithms:

- *Forward / Backward* (Baum--Welch): compute $P("observation" | "model")$ by matrix products in the probability or log semiring.
- *Viterbi*: compute the most probable state sequence by matrix products in the *Viterbi semiring* $([0,1], max, dot)$.
- *Baum--Welch* (EM): re-estimate parameters from observations.

All three are instances of generic matrix algorithms over different semirings; semiring polymorphism is what unifies HMMs, shortest paths, and parsing in the WFST framework.

== Behavioural Equivalences for Probabilistic Systems

For non-deterministic-probabilistic systems (PA in the Segala sense, distinct from Rabin PA), the relevant equivalences are *strong/weak probabilistic bisimulation* (Larsen--Skou 1991; Segala--Lynch 1994). The Larsen--Skou *logic* $L_("LS")$
$ phi ::= "true" | "not" phi | and.big_i phi_i | angle.l a angle.r_p phi $
where $angle.l a angle.r_p phi$ means "with probability $> p$ a transition by $a$ leads to a state satisfying $phi$" characterizes bisimulation in the Hennessy--Milner sense.

*Theorem (Baier--Hermanns 1999).* Probabilistic bisimulation is decidable in polynomial time on finite PAs (partition-refinement à la Paige--Tarjan).

== Algorithmics: Shortest-Distance in Semirings

Mohri's *generic single-source shortest-distance* algorithm:

```python
def shortest_distance(G, s, semiring):
    d = {q: semiring.zero for q in G.states}
    r = {q: semiring.zero for q in G.states}
    d[s] = semiring.one
    r[s] = semiring.one
    Q = {s}
    while Q:
        q = pick(Q)
        R = r[q]
        r[q] = semiring.zero
        for (q, a, w, qp) in G.edges_from(q):
            new = semiring.plus(d[qp], semiring.times(R, w))
            if new != d[qp]:
                d[qp] = new
                r[qp] = semiring.plus(r[qp], semiring.times(R, w))
                Q.add(qp)
    return d
```

For acyclic graphs over any semiring: $O(|V| + |E|)$. For Boolean semiring: BFS. For tropical with non-negative weights: choose the min-key first -- Dijkstra. For tropical with arbitrary weights: Bellman--Ford. For the *log* semiring (sum of logs of exponentials): probabilistic forward algorithm.

== Closure Properties

Weighted automata over a (positive) semiring are closed under:

- *Sum*: $(cal(A) + cal(B))(w) = cal(A)(w) plus.circle cal(B)(w)$; disjoint union with combined initial/final vectors.
- *Hadamard product*: $(cal(A) dot.circle cal(B))(w) = cal(A)(w) times.circle cal(B)(w)$; product construction.
- *Scalar*: $k cal(A)$; multiply initial vector by $k$.
- *Cauchy product / star* (when defined).

They are *not* closed under "complement" (which is meaningless outside the Boolean case), nor under projection on non-commutative semirings in general.

== Probabilistic Büchi Automata

*Probabilistic Büchi automata* (PBA) extend the Rabin model to infinite words with Büchi acceptance condition $"Pr"({pi | "Inf"(pi) sect F eq."not" emptyset.rev}) > lambda$.

*Theorem (Baier--Bertrand--Grösser 2008).* *Almost-sure* emptiness ($lambda = 1$) is decidable for PBA, but *positive* emptiness ($lambda > 0$) is undecidable. Determinization fails: PBA are *strictly more expressive* than deterministic PBA, breaking the McNaughton-style symmetry of the classical theory.

== Quantitative Languages and Mean-Payoff Automata

*Mean-payoff automata* (Chatterjee--Doyen--Henzinger 2008) assign to each infinite run the long-run average $liminf_(n arrow.r oo) (1 slash n) sum_(i=1)^n w_i$ of its edge weights. The behaviour of a non-deterministic mean-payoff automaton on $w$ is $sup$ (or $inf$) over runs.

*Theorem (Chatterjee--Doyen--Henzinger 2010).* *Inclusion* and *equivalence* of non-deterministic mean-payoff automata are *undecidable*.

*Theorem (Chatterjee--Doyen--Henzinger 2010).* Inclusion is *PSPACE-complete* for *deterministic* mean-payoff automata.

This mirrors the (un)decidability of inclusion for tropical weighted automata: in the algebraic limit, $liminf$ is the tropical sum over infinite runs.

== Complexity Landscape

```text
   problem                                       semiring K           complexity
   -------                                       ----------           ----------
   K-rational = K-recognizable                   any                  Schützenberger
   equivalence of WA                             field                P (Tzeng 1992)
   equivalence of WA                             (N, +, *)            decidable
   equivalence of WA                             tropical             undecidable (Krob)
   determinizability of WA                       tropical             decidable (twins)
   PA emptiness (cut-point)                      [0,1]                undecidable (Paz)
   PA equivalence                                [0,1]                P (via Tzeng)
   PA value-1                                    [0,1]                undecidable (GO 2010)
   PCTL model checking (DTMC)                    [0,1]                P
   PCTL model checking (MDP)                     [0,1]                P
   LTL model checking (MDP)                      [0,1]                2EXPTIME, poly in MDP
   PBA almost-sure emptiness                     [0,1]                decidable
   PBA positive emptiness                        [0,1]                undecidable
   simple stochastic games                       [0,1]                NP cap coNP, P open
   mean-payoff WA inclusion (nondet)             (R, +)               undecidable
   mean-payoff WA inclusion (det)                (R, +)               PSPACE-complete
```

== Discussion: Why Algebra?

The semiring abstraction reveals that many disparate algorithms -- shortest paths, parsing weights, probabilistic decoding, language acceptance -- are *the same algorithm* over different algebraic structures. Schützenberger's 1961 theorem is the linchpin: it tells us that the syntactic-combinatorial notion of "rational expression" and the operational notion of "behaviour of a finite-state machine" coincide *for any semiring*. Tzeng's polynomial-time equivalence algorithm over fields is then a linear-algebra realization of the same idea.

The boundary at which decidability fails -- emptiness over $[0,1]$, equivalence over tropical -- is informative. Both failures stem from the same source: the semiring lacks a notion of "rank" that lets us bound the search. Fields have linear-algebraic rank; $[0,1]$ does not (and admits arbitrary real-valued behaviours subject to a real-valued threshold); the tropical semiring admits integer programming via min-plus algebra. The decidable islands inside these undecidable seas -- leaktight PAs, twin-property tropical WAs, isolated cut-points -- are precisely the cases where a finite-rank or finite-witness argument can be made.

== Exercises

1. Build a tropical WA over $Sigma = {a, b}$ whose behaviour is the minimum cost of a sequence of edits to transform $w$ into $a^|w|$, with substitution cost $1$ and identity cost $0$.
2. Prove Tzeng's algorithm: implement it for $cal(A), cal(B)$ over $QQ$ and verify on Schützenberger's classical examples.
3. Construct a $4$-state Rabin PA whose cut-point language at $lambda = 1/2$ is the non-regular language ${w | "weight"(w) > 1/2}$ for an appropriate weighting.
4. Verify the Baum--Welch update equations are EM updates for the likelihood under an HMM.
5. Prove decidability of *almost-sure* reachability in finite MDPs in polynomial time, via end-component decomposition.
6. Show that the value-1 problem becomes decidable when restricted to PAs with deterministic underlying graph (i.e. Markov chains).

== Extended Topic: Eilenberg's Variety Theorem for Weighted Automata

Eilenberg's classical theorem establishes a bijective correspondence between *varieties of regular languages* and *varieties of finite monoids*. Reutenauer (1980) and Reutenauer--Schützenberger extend the framework:

*Theorem (Reutenauer 1980).* For a commutative ring $K$, $K$-recognizable series correspond to representations of the free monoid $Sigma^*$ as a finitely generated $K$-submodule of matrices over $K$. The *syntactic algebra* of a series $s$ is the smallest $K$-algebra recognizing $s$; minimal automata correspond to its irreducible representations.

This perspective views weighted automata as *linear representations* of $Sigma^*$ and explains why polynomial-time equivalence over fields succeeds: rank in linear algebra plays the role of state count.

== Extended Topic: Hankel Matrices and Minimization

For $s in K angle.l angle.l Sigma^* angle.r angle.r$, the *Hankel matrix* $H_s$ is the $Sigma^* times Sigma^*$ matrix with $H_s[u, v] = (s, u v)$.

*Theorem (Carlyle--Paz 1971; Fliess 1974).* For a field $K$, $s$ is $K$-recognizable iff $H_s$ has finite rank, and the minimum number of states recognizing $s$ equals $"rank"(H_s)$.

*Algorithmic content.* Truncating $H_s$ to finite blocks and computing rank gives an algorithm for minimization; the *SVD* of a sub-Hankel block underlies the *spectral learning* of WAs (Bailly--Denis--Ralaivola 2009; Balle--Mohri 2015).

== Extended Topic: Quantitative Languages over Infinite Words

For $omega$-words, several *value functions* over infinite runs:

- *Limit-average / mean-payoff*: $liminf (1 slash n) sum_(i=1)^n w_i$;
- *Discounted-sum*: $sum_(i=0)^oo lambda^i w_i$ for $lambda in (0,1)$;
- *Limsup / liminf* of edge weights;
- *Sup / inf* over prefixes.

*Theorem (Chatterjee--Doyen--Henzinger 2010).* For *deterministic* quantitative automata, inclusion is decidable for limsup, liminf, sup, inf in PTIME; for *discounted-sum* it is *not known* whether inclusion is decidable (related to the *positivity problem* for linear recurrences).

== Extended Topic: Decision Problems for Linear Recurrence Sequences

The behaviour of a WA evaluated on $a^n$ for fixed $a$ is a *linear recurrence sequence* (LRS) over $K$. Several long-standing open problems live here:

*Skolem's problem.* Given an integer LRS $(u_n)$, decide whether $exists n. u_n = 0$. Decidable for orders $lt.eq 4$ (Mignotte--Shorey--Tijdeman 1984; Vereshchagin 1985), open for orders $gt.eq 5$.

*Positivity problem.* Decide whether $forall n. u_n gt.eq 0$. Decidable up to order $5$ (Ouaknine--Worrell 2014); open beyond.

These problems sit at the boundary of effective number theory and automata theory. Decidability would have implications for verification of weighted/probabilistic systems and for the Continuous Skolem Problem (Bell--Delvenne--Jungers--Blondel 2010).

== Extended Topic: Cost Register Automata

*Cost register automata* (Alur--D'Antoni--Deshmukh--Raghothaman--Yuan 2013) are deterministic finite automata augmented with a finite set of registers over a semiring $K$, updated by linear combinations on each transition. The output is a register value at the end.

*Theorem (Alur--D'Antoni et al. 2013).* CRAs are exactly as expressive as a natural subclass of weighted automata; over the $(NN, +, dot)$ semiring with copyless updates (each register used at most once per update), they coincide with unambiguous WAs.

CRAs serve as a *structured operational semantics* for streaming computation; tools like *streamable string transducers* (Alur--Cerný 2010) extend the framework with strings.

== Extended Topic: Tropical Semirings and Optimization

The tropical semiring $(RR union {oo}, min, +)$ linearizes:

- *Shortest path*: $A^*$-th power computes all-pairs shortest paths;
- *Optimal control*: Bellman's equation $V = c plus.circle M times.circle V$ has tropical Banach fixpoint solutions;
- *Tropical polynomials*: the *Newton polytope* governs roots; *tropical geometry* (Itenberg--Mikhalkin--Shustin) is a thriving discipline.

The undecidability of WA equivalence over the tropical semiring (Krob 1994) => that *several* optimization-theoretic equivalence questions are uncomputable: the equality of two min-plus rational functions cannot in general be decided.

== Extended Topic: Determinizability via the Twins Property

For tropical WAs, *determinization* succeeds <==> the *twins property* holds: any two states $p, q$ reachable on the same word $u$ from initial, and lying on cycles labelled by the same word $v$, must have cycle weights satisfying $w(p, v, p) = w(q, v, q)$.

*Theorem (Mohri 1997).* The twins property is decidable for *trim* tropical WAs in polynomial time.

*Theorem (Allauzen--Mohri 2003).* Mohri's classical determinization algorithm constructs a (possibly exponential) deterministic equivalent when the twins property holds, by tracking *residual weights* in a subset construction.

For the probability semiring, *no analogue of the twins property* yields determinizability; in fact, probabilistic automata are *strictly more expressive* than deterministic Markov chains over the same state count.

== Extended Topic: Markov Decision Processes and Reward-Based Objectives

A *finite-horizon* MDP with reward $r(s, a)$ has value function
$ V^*_k(s) = max_a [r(s, a) + sum_(s') P(s' | s, a) V^*_(k-1)(s')]. $

For *infinite-horizon discounted* MDPs (discount $gamma in [0, 1)$):
$ V^*(s) = max_a [r(s, a) + gamma sum_(s') P(s' | s, a) V^*(s')], $
the Bellman equation has unique fixed point computable by *value iteration* (geometric convergence rate $gamma$) or *policy iteration* (Howard 1960; converges in strongly polynomial time, Ye 2011).

For *average-reward* MDPs, the Howard--Veinott decomposition yields a system of *gain* and *bias* equations; optimal policies are stationary and computable via linear programming.

*Connection.* Quantitative model checking of an MDP against an $omega$-regular objective reduces to discounted reachability on the product MDP, then to LP. The whole pipeline -- product, end-component decomposition, LP -- is implemented in PRISM and Storm.

== Extended Topic: Partially Observable MDPs

POMDPs (Smallwood--Sondik 1973) replace state observation with *observation distributions* $O(o | s, a)$. The agent maintains a *belief state* $b in Delta(S)$.

*Theorem (Madani--Hanks--Condon 1999).* Almost-sure reachability in finite POMDPs is *EXPTIME-complete*; positive (non-zero probability) reachability is *undecidable*.

POMDPs intersect automata learning: *probabilistic-language equivalence* of POMDPs is decidable by reduction to weighted-automaton equivalence over $QQ$ (since belief updates are linear).

== Extended Topic: Stochastic Petri Nets

*Generalized stochastic Petri nets* (Marsan--Conte--Balbo 1984) attach exponential firing times to transitions; the underlying semantics is a *continuous-time Markov chain* (CTMC). They subsume Jackson queueing networks and product-form solutions.

CTMC model checking with *continuous stochastic logic* (CSL, Aziz--Sanwal--Singhal--Brayton 2000): formulas of the form $S_(prec.eq p)[psi]$ (long-run probability) and $P_(prec.eq p)[psi]$ (transient probability). Decided by transient analysis (uniformization, Jensen 1953) and stationary analysis (linear systems).

== Extended Topic: Quantitative Bisimulation and Behavioural Metrics

For weighted/probabilistic systems, *equality of bisimulation* is too rigid: small perturbations break it. *Behavioural pseudometrics* (Desharnais--Gupta--Jagadeesan--Panangaden 2004) define a $1$-Lipschitz distance via a Banach fixed-point of the *Kantorovich--Wasserstein* metric on probability distributions.

*Theorem (van Breugel--Worrell 2001).* Bisimilarity-pseudometric on probabilistic labelled transition systems is computable in polynomial time using LP at each iteration.

Applications: *quantitative reasoning about implementations*, certifying that a probabilistic abstraction is "$epsilon$-close" to the concrete system.

== Worked Example: Tropical Shortest-Path Automaton

Consider a road network with $4$ cities and directed edges labelled by distances. Build a WA $cal(A)$ over $K = (NN union {oo}, min, +)$:

```text
   states: {1, 2, 3, 4}
   lambda = [0, inf, inf, inf]  (start at 1)
   gamma = [inf, inf, inf, 0]    (accept at 4)
   mu(a)[i,j] = distance from i to j on alphabet a"  (or inf)
```

Then $cal(A)(a^n)$ is the shortest path from $1$ to $4$ using exactly $n$ edges. The matrix product in $(min, +)$ is the dynamic-programming step of Bellman--Ford.

== Worked Example: Markov Chain for the Drunkard's Walk

States $0, 1, dots, N$; transitions $i arrow.r i+1$ and $i arrow.r i-1$ each with probability $1/2$ (absorbing at $0$ and $N$). The probability of reaching $N$ starting from $i$ is $i / N$. PCTL verification: $P_(>= 0.5)[F " state " N]$ holds <==> $i gt.eq N/2$. PRISM computes this in $O(N)$ via the tridiagonal linear system.

== Worked Example: PRISM Model for the Crowds Protocol

The Crowds anonymity protocol (Reiter--Rubin 1998): a sender forwards a message via random hops in a crowd of $N$ users, each of which independently forwards with probability $p_f$ or delivers with probability $1 - p_f$. Some users are *corrupt*; the probability that the corrupt observers learn the true sender is computed by PRISM as a PCTL probabilistic-reachability query. Quantitative results inform the choice of $p_f$ and $N$ to achieve target anonymity guarantees.

== Worked Example: Weighted Transducer for Edit Distance

Edit distance from $u in Sigma^*$ to $v in Sigma^*$ is computed by a tropical-semiring transducer:

```text
   States: single state q.
   Self-loops: (a, a, 0)         (match)
               (a, b, 1) for a != b   (substitute)
               (epsilon, b, 1)        (insert)
               (a, epsilon, 1)        (delete)
   Initial weight 0, final weight 0.
```

Composing the singleton transducers $T_u : epsilon arrow.r u$ and $T_v : epsilon arrow.r v$ with this edit-distance transducer and taking the shortest distance yields the Levenshtein distance.

== References (Selected)

- M.-P. Schützenberger. *On the Definition of a Family of Automata*. Information and Control 4, 1961.
- M. O. Rabin. *Probabilistic Automata*. Information and Control 6, 1963.
- A. Paz. *Introduction to Probabilistic Automata*. Academic Press, 1971.
- W.-G. Tzeng. *A Polynomial-Time Algorithm for the Equivalence of Probabilistic Automata*. SICOMP 21, 1992.
- D. Krob. *The Equality Problem for Rational Series with Multiplicities in the Tropical Semiring is Undecidable*. ICALP 1994.
- M. Mohri. *Finite-State Transducers in Language and Speech Processing*. Comp. Linguistics 23, 1997.
- M. Mohri, F. Pereira, M. Riley. *Weighted Finite-State Transducers in Speech Recognition*. CSL 16, 2002.
- H. Hansson, B. Jonsson. *A Logic for Reasoning about Time and Reliability*. FAC 6, 1994.
- C. Courcoubetis, M. Yannakakis. *The Complexity of Probabilistic Verification*. JACM 42, 1995.
- C. Baier, J.-P. Katoen. *Principles of Model Checking*. MIT Press, 2008.
- H. Gimbert, Y. Oualhadj. *Probabilistic Automata on Finite Words: Decidable and Undecidable Problems*. ICALP 2010.
- L. S. Shapley. *Stochastic Games*. PNAS 39, 1953.
- M. Droste, W. Kuich, H. Vogler (eds.). *Handbook of Weighted Automata*. Springer, 2009.

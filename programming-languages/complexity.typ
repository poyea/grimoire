= Complexity Theory

Complexity theory classifies *decidable* problems by the resources required to
solve them. Where recursion theory asks "is there an algorithm?", complexity asks
"is there a *feasible* algorithm?", and -- since 1971 -- has mostly been the
study of why the answer appears to be "no" while we struggle to prove it.

_See also: _Computability_, _Turing Machines and Computability_, _Omega-Automata_
for the complexity of parity games and infinite-word problems._

== Resource-Bounded Computation

Fix a multitape Turing machine model. For a function $f : NN -> NN$,

- $"DTIME"(f(n))$: languages decided by some deterministic TM in time $O(f(n))$.
- $"NTIME"(f(n))$: languages decided by some nondeterministic TM (existential
  acceptance) in time $O(f(n))$ on every accepting branch.
- $"DSPACE"(f(n))$: deterministic TM using $O(f(n))$ work-tape cells (input is
  read-only, output is write-only; this allows sublinear space).
- $"NSPACE"(f(n))$: nondeterministic counterpart.

A function $f$ is *time-constructible* if some DTM, on input $1^n$, outputs $f(n)$
in time $O(f(n))$; *space-constructible* analogously. All natural bounds $n,
n log n, n^k, 2^n$ are constructible.

The canonical hierarchy of classes:

$ "L" subset.eq "NL" subset.eq "P" subset.eq "NP" subset.eq "PH" subset.eq "PSPACE" = "NPSPACE" subset.eq "EXP" subset.eq "NEXP" subset.eq "EXPSPACE" $

where $"L" = "DSPACE"(log n)$, $"NL" = "NSPACE"(log n)$, $"P" = union.big_k
"DTIME"(n^k)$, $"NP" = union.big_k "NTIME"(n^k)$, $"PSPACE" = union.big_k
"DSPACE"(n^k)$, $"EXP" = union.big_k "DTIME"(2^(n^k))$, $"EXPSPACE" = union.big_k
"DSPACE"(2^(n^k))$. The only *proper* inclusions known among these are those
forced by the hierarchy theorems below.

== Hierarchy Theorems

*Theorem (Time Hierarchy, Hartmanis--Stearns 1965; sharpened Hennie--Stearns 1966).*
If $f$ is time-constructible and $f(n) log f(n) = o(g(n))$, then $"DTIME"(f(n))
subset.neq "DTIME"(g(n))$.

*Proof sketch.* Construct a *diagonal* language

$ D = { angle.l M, w angle.r | M "is a DTM and" M "rejects" angle.l M, w angle.r "within" g(n) "steps"}. $

A universal TM simulates an arbitrary DTM with $O(log f(n))$ overhead (each step
costs $O(log f(n))$ for state lookup and clock maintenance with two-tape
simulation). So $D in "DTIME"(g(n))$. If $D in "DTIME"(f(n))$ via some $M_0$, then
running $M_0$ on $angle.l M_0, M_0 angle.r$ yields a contradiction by the standard
diagonal argument. $square$

The $log f$ factor is the *cost of universal simulation*. For single-tape DTMs the
gap is $f(n)^2 = o(g(n))$ (Hartmanis--Stearns original); for RAM machines it
collapses to $f(n) = o(g(n))$ (Cobham). Random-access models that simulate any TM
in linear time achieve the cleanest hierarchy.

*Corollary.* $"P" subset.eq."not" "EXP"$; in particular $"DTIME"(n) subset.neq
"DTIME"(n^2)$, $"DTIME"(2^n) subset.neq "DTIME"(2^(2 n))$.

*Theorem (Space Hierarchy, Stearns--Hartmanis--Lewis 1965).* If $f$ is
space-constructible and $f(n) = o(g(n))$, then $"DSPACE"(f(n)) subset.neq
"DSPACE"(g(n))$. (No log factor: a $g$-space machine can directly simulate any
$f$-space machine within its own space budget once it has a clock to prevent
infinite loops; configurations are bounded by $2^(O(f))$ so a counter of that size
suffices.)

*Corollary.* $"L" subset.eq."not" "PSPACE"$, $"PSPACE" subset.eq."not" "EXPSPACE"$. We do
*not* know $"L" subset.eq."not" "P"$, $"P" subset.eq."not" "PSPACE"$, or $"P" subset.neq
"NP"$ -- all are open.

*Theorem (Gap Theorem, Borodin 1972; Trakhtenbrot 1964).* For every total
computable $r(n) gt.eq n$, there exists a computable $f(n)$ such that

$ "DTIME"(f(n)) = "DTIME"(r(f(n))). $

So the hierarchy can have *arbitrarily large gaps* if $f$ is allowed to be
pathologically non-constructible. The constructibility hypothesis in the hierarchy
theorems is therefore essential, not cosmetic.

*Theorem (Blum Speed-up, 1967).* There exists a recursive language $L$ such that
for *every* DTM $M$ deciding $L$ in time $t_M (n)$, there is another DTM $M'$
deciding $L$ in time $t_(M') (n) = O(log t_M (n))$ almost everywhere. In short,
$L$ has *no asymptotically optimal algorithm*.

The proof uses a priority-style diagonalisation over (machine, time bound) pairs.
Speed-up languages are highly contrived, but the theorem demolishes naive hopes
that every problem has a "best" algorithm.

== P, NP, coNP, and the Polynomial Hierarchy

A language $L in "NP"$ <==> there is a polynomial $p$ and a polynomial-time
verifier $V$ with $L = { x | exists y, |y| lt.eq p(|x|), V(x, y) = 1 }$. $"coNP"
= { L | overline(L) in "NP" }$; characterised by $forall$-quantified polynomial
witnesses.

The *polynomial hierarchy* (Stockmeyer 1976):

$ Sigma^p_0 = Pi^p_0 = "P", quad Sigma^p_(k + 1) = "NP"^(Sigma^p_k), quad Pi^p_(k + 1) = "coNP"^(Sigma^p_k). $

Equivalently $L in Sigma^p_k$ iff

$ L = { x | exists y_1 forall y_2 exists y_3 dots Q_k y_k . V(x, y_1, ..., y_k) = 1 } $

with $V$ polynomial-time and $|y_i| = "poly"(|x|)$. So $Sigma^p_1 = "NP"$,
$Pi^p_1 = "coNP"$, $Sigma^p_2$ has $exists forall$, etc. $"PH" = union.big_k
Sigma^p_k$.

*Collapse principle.* If $Sigma^p_k = Pi^p_k$ for some $k$, then $"PH" =
Sigma^p_k$; if $"P" = "NP"$, then $"PH" = "P"$. PH collapses are considered
unlikely and used as heuristic non-collapse evidence.

== NP-Completeness: Cook--Levin

*Theorem (Cook 1971, Levin 1973).* $"SAT" = { phi | phi "is a satisfiable
Boolean formula" }$ is NP-complete: $"SAT" in "NP"$ and every $L in "NP"$
Karp-reduces to $"SAT"$.

*Proof (tableau encoding).* Fix $L in "NP"$ decided by an NTM $M$ in time $p(n)$.
For input $x$ of length $n$, an *accepting tableau* is a $(p(n) + 1) times (p(n) + 1)$
grid whose row $i$ is the configuration of $M$ at step $i$, with row $0$ being
the start configuration on $x$. The reduction builds a CNF formula $phi_x$ whose
satisfying assignments correspond exactly to accepting tableaux.

Variables (polynomially many in $n$):

- $T_(i, j, s)$ for $0 lt.eq i, j lt.eq p(n)$ and $s in Gamma union (Q times Gamma)$,
  meaning "cell $(i, j)$ contains symbol $s$ ("or head-state-symbol pair)".

Clauses:

```text
Cell:    For each (i,j): exactly one s with T_{i,j,s} true.
Start:   Row 0 encodes q0 x1 x2 ... xn _ _ ...
Accept:  Some cell in some row contains q_accept.
Move:    For each 2x3 window of cells (rows i, i+1; cols j-1, j, j+1):
         the window is "legal" -- it matches some legitimate transition of M
         ("or is unchanged outside the head). Encode via a constant-size CNF.
```

The total number of variables and clauses is polynomial. $M$ accepts $x$ iff
$phi_x$ is satisfiable. The reduction is computable in $O(p(n)^2 log p(n))$
time. $square$

Levin's independent proof used *tag systems* and emphasised the universality of
the construction across models. The Cook--Levin theorem is the foundation: once
SAT is NP-complete, NP-completeness propagates by *reduction chains*.

*Karp's 21 problems (1972).* In one ten-page paper Karp showed CLIQUE, VERTEX
COVER, SET COVER, FEEDBACK ARC SET, FEEDBACK VERTEX SET, DIRECTED HAMILTONIAN
CYCLE, UNDIRECTED HAMILTONIAN CYCLE, 3-SAT, CHROMATIC NUMBER, CLIQUE COVER, EXACT
COVER, HITTING SET, STEINER TREE, 3-DIMENSIONAL MATCHING, KNAPSACK, JOB SEQUENCING,
PARTITION, MAX CUT, and others are NP-complete. The modern canon adds:

- *3-SAT* via clause width reduction; remains NP-complete with at most 3 literals
  per clause and each variable appearing in $lt.eq 4$ clauses.
- *3-COLOUR*: from 3-SAT via a literal/clause gadget.
- *HAMPATH*: from 3-SAT via XOR gadgets.
- *SUBSET-SUM*: from 3-SAT via base-$(m + 1)$ digit encoding (avoid carries).
- *INTEGER LINEAR PROGRAMMING* (feasibility): from SAT.

*Reductions.* The two standard notions:

- *Karp / many-one polynomial-time reduction* $L_1 lt.eq^p_m L_2$: poly-time $f$
  with $x in L_1 arrow.l.r.double f(x) in L_2$.
- *Cook / polynomial-time Turing reduction* $L_1 lt.eq^p_T L_2$: poly-time oracle
  machine deciding $L_1$ using an $L_2$-oracle.

Cook reductions can collapse distinctions Karp reductions preserve: every
NP-complete problem is Cook-equivalent to its complement, but is Karp-equivalent
only if $"NP" = "coNP"$.

== Ladner's Theorem

*Theorem (Ladner 1975).* If $"P" eq."not" "NP"$, then there exists an *NP-intermediate*
language: $L in "NP" backslash "P"$ with $L$ not NP-complete.

*Proof idea.* Define $L = "SAT" sect cal(F)$ where $cal(F)$ is a recursively
constructed set of formula sizes designed by *delayed diagonalisation*: at even
stages, "blow up" $L$ enough to defeat the $i$-th polynomial-time machine
(ensuring $L in."not" "P"$); at odd stages, "blow down" $L$ enough to defeat the
$i$-th polynomial reduction $f_i : "SAT" lt.eq^p_m L$ (ensuring $L$ is not
NP-complete). The blow-up/blow-down is parameterised by a slow-growing function
that depends on which stage was last accomplished. $square$

Ladner shows the NP landscape is rich. Natural candidates for NP-intermediate
problems include FACTORING, GRAPH ISOMORPHISM (almost-polynomial, Babai 2016),
and DISCRETE LOG -- none are proven intermediate, but none are proven NP-complete
or in $"P"$.

== Space Hierarchy: Savitch and Immerman--Szelepcsényi

*Theorem (Savitch 1970).* For space-constructible $f(n) gt.eq log n$,

$ "NSPACE"(f(n)) subset.eq "DSPACE"(f(n)^2). $

*Proof.* Configurations of an $f$-space NTM are $2^(O(f))$ in number. Reachability
in the configuration graph: define

$ "REACH"(c_1, c_2, k) = "true <==>" c_2 "reachable from" c_1 "in" lt.eq 2^k "steps"$

with the recurrence

$ "REACH"(c_1, c_2, k) = exists c . "REACH"(c_1, c, k - 1) and "REACH"(c, c_2, k - 1). $

Base case $k = 0$: $c_1 = c_2$ or $c_1 |-_M c_2$ in one step, checkable in space
$O(f)$. Recursive depth is $log 2^(O(f)) = O(f)$. Each level pushes one $c$ of
size $O(f)$ onto the stack. Total space: $O(f) dot O(f) = O(f^2)$. $square$

*Corollary.* $"PSPACE" = "NPSPACE"$; $"NL" subset.eq "DSPACE"(log^2 n)$.

*Theorem (Immerman 1988, Szelepcsényi 1987 -- independent).* $"NSPACE"(f(n)) =
"coNSPACE"(f(n))$ for $f(n) gt.eq log n$.

*Proof (inductive counting).* Given a graph $G$ on $2^(O(f))$ configurations and a
source $s$, we show that "$t$ not reachable" is in NSPACE$(f)$. Suppose the exact
count $r_i$ of vertices reachable from $s$ in $lt.eq i$ steps is *given*. Then
$t$ is unreachable <==> in the set of $r_i$ reachable vertices, $t$ does not
appear. An NTM guesses $r_i$ vertices, verifies each (by guessing a path of length
$lt.eq i$ in $O(f)$ space), checks they are distinct and that none equals $t$,
and counts.

To compute $r_(i + 1)$ from $r_i$: for each potential new vertex $v$, decide
"reachable in $lt.eq i + 1$" by guessing whether $v$ is reachable in $lt.eq i$
or $v$ is a neighbour of one of the $r_i$ vertices. Iterate from $r_0 = 1$ to
$r_n$ where $n = 2^(O(f))$, storing only one $r_i$ at a time. Total space $O(f)$. $square$

*Corollary.* $"NL" = "coNL"$. This was a major surprise: the NP/coNP analogue at
log-space level *collapses*. The result hints that the asymmetry between
nondeterminism and co-nondeterminism may be a function of *time* bounds, not
space bounds.

== Randomised Complexity

A *probabilistic TM* has a transition function with random bits. Define:

- $"RP"$: $x in L => Pr["accept"] gt.eq 1/2$, $x in.not L =>
  Pr["accept"] = 0$. One-sided error toward NO.
- $"coRP"$: one-sided error toward YES.
- $"ZPP" = "RP" inter "coRP"$: zero error, expected polynomial time.
- $"BPP"$: $Pr["correct"] gt.eq 2/3$ on every input. Two-sided bounded error.
- $"PP"$: $Pr["correct"] > 1/2$. Unbounded error; $"PH" subset.eq "P"^"PP"$ (Toda).

Error amplification: $k$ independent repetitions plus majority vote reduce error
to $2^(-Omega(k))$ in BPP, by Chernoff. So the constant $2/3$ can be replaced by
$1 - 2^(-"poly")$ without change of cal(C).

*Theorem (Adleman 1978).* $"BPP" subset.eq "P/poly"$.

*Proof.* Amplify error to $< 2^(-(n + 1))$. By the union bound, for fixed input
length $n$, at most a $2^n dot 2^(-(n + 1)) = 1/2$ fraction of random strings $r$
fails on any input; in particular there exists an $r$ correct on *"all"* $2^n$
inputs of length $n$. Hard-wire this $r$ as the advice string. $square$

*Theorem (Sipser 1983, Gács; Lautemann 1983).* $"BPP" subset.eq Sigma^p_2 inter
Pi^p_2$.

*Proof (Lautemann's covering trick).* Let $M$ be a BPP machine with error $< 2^(-n)$
on inputs of length $n$, using $m = "poly"(n)$ random bits. Let $S_x = { r in
{0, 1}^m | M(x, r) "accepts"}$. For $x in L$, $|S_x| gt.eq (1 - 2^(-n)) 2^m$; for
$x in."not" L$, $|S_x| lt.eq 2^(-n) 2^m$. Claim:

$ x in L arrow.l.r.double exists t_1, ..., t_m in {0, 1}^m forall r in {0, 1}^m exists i . r xor t_i in S_x. $

If $|S_x|$ is large, random shifts cover ${0, 1}^m$ with high probability; union
bound gives existence. If $|S_x|$ is small, $m$ shifts cover at most $m 2^(-n) 2^m
< 2^m$, so some $r$ is missed. Resulting formula is $exists forall$, i.e.
$Sigma^p_2$. Symmetrically $Pi^p_2$. $square$

*Theorem (Impagliazzo--Wigderson 1997).* If some language in $"E"$ requires
exponential-size circuits then $"BPP" = "P"$ -- *derandomisation* under
plausible circuit lower bounds.

== Interactive Proofs

Generalise NP by allowing the verifier $V$ to be probabilistic and the prover $P$
to be unbounded but adaptive. An *interactive proof system* for $L$:

- *Completeness*: $x in L => exists P . Pr[(P, V)(x) "accepts"] gt.eq 2/3$.
- *Soundness*: $x in."not" L => forall P^* . Pr[(P^*, V)(x) "accepts"] lt.eq 1/3$.

$"IP" = $ languages with $"poly"(n)$ rounds and poly-time $V$. Variants:

- $"MA"$: Merlin sends a proof, Arthur verifies (probabilistic NP).
- $"AM"$: Arthur sends randomness, Merlin replies, Arthur verifies.
- $"AM"[k]$: $k$ rounds. *Theorem (Babai--Moran):* $"AM"[k] = "AM"[2]$ for
  constant $k$; private coins collapse to public coins (Goldwasser--Sipser).

*The LFKN protocol (Lund--Fortnow--Karloff--Nisan 1992).* IP for $\#"SAT"$. Given
3-CNF $phi(x_1, ..., x_n)$ and target value $K$, prover claims

$ sum_(x_1 in {0,1}) sum_(x_2 in {0,1}) dots sum_(x_n in {0,1}) hat(phi)(x_1, ..., x_n) = K $

where $hat(phi)$ is the *arithmetisation* of $phi$ (AND $arrow.bar dot$, OR
$arrow.bar a + b - a b$, NOT $arrow.bar 1 - a$) over a prime field $bb(F)_p$ with
$p$ exponentially larger than the degree.

Round $i$: Prover sends univariate $g_i (X) = sum_(x_(i + 1), ..., x_n) hat(phi)
(r_1, ..., r_(i - 1), X, x_(i + 1), ..., x_n)$ (degree $lt.eq 3 n$, the formula's
degree). Verifier checks $g_i (0) + g_i (1) = $ the value asserted in round $i -
1$, samples $r_i$ uniformly, sends to prover. Round $n + 1$: verifier evaluates
$hat(phi)(r_1, ..., r_n)$ in poly-time and compares with $g_n (r_n)$.

*Soundness.* Each round, a cheating prover must commit to a polynomial that agrees
with the true polynomial at the chosen $r_i$; degree-$3n$ polynomials over
$bb(F)_p$ disagree everywhere outside a $3 n / p$ fraction, so soundness error per
round is $3 n \/ p$. Over $n$ rounds: $3 n^2 \/ p$, negligible. $square$

*Corollary (LFKN).* $\#"SAT" in "IP"$, hence $"coNP" subset.eq "IP"$ (since $"coNP"
lt.eq^p_T #"SAT"$).

*Theorem (Shamir 1990).* $"IP" = "PSPACE"$.

*Sketch.* $"IP" subset.eq "PSPACE"$: an unbounded prover and poly-round verifier
can be simulated by computing the optimal verifier-against-best-prover game tree,
which fits in polynomial space. $"PSPACE" subset.eq "IP"$: extend LFKN to True
Quantified Boolean Formulas (TQBF), the canonical PSPACE-complete problem. The
arithmetisation must now handle alternating $exists$ and $forall$ quantifiers,
encoded as $sum$ and $product$ over $bb(F)_p$. Naive arithmetisation explodes the
degree; the *Shamir trick* inserts a *linearisation operator* $L_(x_i)$ after each
quantifier that re-projects to degree 1 in each variable, keeping polynomials
manageable. The number of rounds is $O("poly"(n))$. $square$

A strikingly tight characterisation: alternating quantifiers (PSPACE) are exactly
what interactive proofs can verify. The proof uses *no cryptographic assumptions*.

== The PCP Theorem

*Theorem (Arora--Lund--Motwani--Sudan--Szegedy 1998; Dinur 2007 combinatorial
proof).* $"NP" = "PCP"(O(log n), O(1))$.

A *probabilistically checkable proof* system $"PCP"(r(n), q(n))$ uses $O(r(n))$
random bits, queries $O(q(n))$ proof bits non-adaptively, and decides with
completeness $1$ and soundness $1/2$. So every NP language has a proof that a
*constant* number of bits suffices to verify, with logarithmic randomness.

*Equivalent formulation (gap-SAT).* There is a polynomial-time reduction from SAT
to 3-SAT such that:

- $phi$ satisfiable $=>$ reduced $phi'$ is satisfiable.
- $phi$ unsatisfiable $=>$ every assignment to $phi'$ violates a
  *constant* fraction (say $1 - 1/2$) of clauses.

The gap is the source of *hardness of approximation*.

*Dinur's gap amplification* iterates three operations on a constraint graph $G$
with constraints over alphabet $Sigma$:

+ *Preprocessing*: make $G$ $d$-regular and an expander (constant degree, constant
  spectral gap), without changing UNSAT-value much.
+ *Powering*: take walks of length $t$ in $G$, build a new graph $G^t$ over a
  larger alphabet whose constraints check long walks. UNSAT value roughly
  multiplies by $sqrt(t)$.
+ *Alphabet reduction* via an inner PCP (e.g. long-code or Hadamard): reduce
  $|Sigma^t|$ back to constant alphabet, paying a constant-factor loss in UNSAT.

Each iteration multiplies UNSAT-gap by a constant factor; $O(log n)$ iterations
amplify $1/"poly"(n)$ to $Omega(1)$. The original ALMSS proof was algebraic
(linearity testing, low-degree testing, composition); Dinur's is combinatorial
and shorter.

*Hardness-of-approximation corollaries.*

- *MAX-3SAT* is NP-hard to approximate within $7/8 + epsilon$ (Håstad 2001 -- the"
  $7/8$ bound is tight, matching random assignment).
- *Set Cover*: $(1 - epsilon) ln n$ approximation is NP-hard (Dinur--Steurer
  2014).
- *Vertex Cover*: NP-hard to approximate within $sqrt(2) - epsilon$ (Khot--Minzer--
  Safra 2017); $2 - epsilon$ under the Unique Games Conjecture (Khot--Regev).
- *Label Cover* is the master problem for two-prover one-round PCPs and the source
  of most hardness reductions.

== Circuit Complexity

A *Boolean circuit* over $n$ inputs is a DAG with input gates $x_1, ..., x_n,
overline(x)_1, ..., overline(x)_n$, internal gates $and, or$ (sometimes $not$
free), and a designated output. *Size* = number of gates; *depth* = longest path
length.

A *family* $C = (C_n)_(n in NN)$ decides a language $L$ if $C_n (x) = 1
arrow.l.r.double x in L$ for $|x| = n$. The family is *uniform* if the map $n
arrow.bar C_n$ is computable within stated resources (typically log-space).

- $"P/poly"$: polynomial-size circuits (non-uniform).
- $"NC"^k$: poly-size, $O(log^k n)$-depth circuits with bounded fan-in.
- $"AC"^k$: poly-size, $O(log^k n)$-depth with *unbounded* fan-in.
- $"TC"^k$: $"AC"^k$ plus threshold gates.

Inclusions: $"NC"^k subset.eq "AC"^k subset.eq "TC"^k subset.eq "NC"^(k + 1)$.
$"NC" = union.big_k "NC"^k$ captures *efficient parallelism*; problems in $"NC"$
have $"polylog"$-time PRAM algorithms.

*Theorem (Furst--Saxe--Sipser 1984; Yao 1985; Håstad 1986).* $"PARITY" in.not
"AC"^0$. More precisely, every depth-$d$ $"AC"^0$ circuit computing $"PARITY"_n$
has size $2^(Omega(n^(1/(d - 1))))$.

*Proof (Håstad's switching lemma).* A random restriction $rho$ fixing all but a
$p$-fraction of inputs simplifies a depth-$d$ $"AC"^0$ circuit: with high
probability each subcircuit becomes computable by a small *decision tree*, and
alternating $"or"$/$"and"$ layers collapse. Iterate: a depth-$d$ circuit collapses to
depth $d - 1$ after restriction. After $d - 1$ restrictions the circuit is a
constant, but PARITY of any nonempty restriction is non-constant -- contradiction.
$square$

*Theorem (Razborov 1987, Smolensky 1987).* $"AC"^0 [p]$ ($"AC"^0$ augmented with
$"MOD"_p$ gates, $p$ prime) cannot compute $"MOD"_q$ for $q$ a different prime.
In particular $"AC"^0 [2]$ cannot compute $"MOD"_3$.

*Proof.* Approximate each gate by a *low-degree polynomial* over $bb(F)_p$.
Over $O(log n)$ depth this gives a degree-$"polylog"$ approximator for the
entire circuit, agreeing with it on most inputs. But $"MOD"_q$ for $q eq."not" p$
has full degree as a function $bb(F)_p^n -> bb(F)_p$; no low-degree polynomial
agrees with it on a $1/2 + epsilon$ fraction. $square$

*Theorem (Razborov 1985).* CLIQUE has no polynomial-size *monotone* circuits.
Specifically, $k$-CLIQUE requires monotone circuits of size $n^(Omega(sqrt(k)))$.

The proof method is the *approximation method*: replace each AND/OR gate by an
approximator (a CNF/DNF of bounded size), bound how many errors approximation
introduces, and show any small monotone circuit makes too many errors against
both YES instances ($k$-cliques) and NO instances ($(k-1)$-partite graphs).

Razborov's result raised hopes that non-monotone lower bounds were imminent. They
were not.

== The Natural Proofs Barrier

*Definition (Razborov--Rudich 1997).* A *combinatorial property* $cal(P)_n
subset.eq { f : {0, 1}^n -> {0, 1} }$ is *natural* against a cal(C) $cal(C)$ if it
satisfies:

+ *Largeness*: $|cal(P)_n| gt.eq 2^(-O(n)) dot 2^(2^n)$; a random function has
  property $cal(P)_n$ with non-negligible probability.
+ *Constructivity*: membership "$f in cal(P)_n$?" is decidable in time $2^(O(n))$
  given the truth table of $f$ (i.e. polynomial in the truth-table size).
+ *Usefulness*: $f in cal(P)_n => f in."not" cal(C)$.

*Theorem (Razborov--Rudich 1997).* If pseudorandom function generators secure
against $2^(n^epsilon)$-size circuits exist, then there is no natural proof of
$cal(C) eq."not" "P/poly"$ for any superpolynomial $cal(C)$.

*Proof intuition.* A constructive, large property is itself a polynomial-time
*statistical test* distinguishing "random" functions from functions in $cal(C)$.
If $cal(C) supset.eq $ pseudorandom function family, the test breaks PRFs. PRFs
in turn are believed to follow from one-way functions on hard instances
(Goldreich--Goldwasser--Micali, Håstad--Impagliazzo--Levin--Luby). So either
crypto fails, or natural proofs do not exist for strong lower bounds. $square$

Most known lower-bound techniques (random restrictions, approximation, polynomial
method, communication complexity) yield natural properties. To prove $"P" eq.not
"NP"$ (or even $"NEXP" eq.not "P/poly"$) one must invent *non-natural* techniques
-- e.g. *non-constructive* or *non-large*.

*Theorem (Williams 2014).* $"NEXP" eq."not" "ACC"^0$. The proof *bypasses natural
proofs* by using a non-constructive ingredient: a faster-than-trivial satisfiability
algorithm for $"ACC"^0$ circuits, combined with a Karp--Lipton style diagonalisation.

Williams's blueprint -- "improved SAT algorithm for $cal(C)$ $=>$
$"NEXP" eq."not" cal(C)$" -- is one of the few known routes around the barrier.

*Other barriers.* The *relativisation barrier* (Baker--Gill--Solovay 1975): there
exist oracles $A$ with $"P"^A = "NP"^A$ and oracles $B$ with $"P"^B eq."not" "NP"^B$,
so any proof of $"P" eq."not" "NP"$ must use techniques that *do not relativise*.
The *algebrisation barrier* (Aaronson--Wigderson 2008): low-degree extensions of
oracles also fail to separate $"P"$ from $"NP"$.

== Descriptive Complexity

*Theorem (Fagin 1974).* $"NP" = "ESO"$, where ESO is existential second-order
logic: properties expressible as $exists R_1 dots exists R_k . Phi$ with $Phi$
first-order and $R_i$ relation symbols.

*Proof.* $"ESO" subset.eq "NP"$: guess the relations $R_i$, verify $Phi$ in
poly-time. $"NP" subset.eq "ESO"$: an $"NP"$ computation on input encoded as a
finite structure is captured by quantifying over a tableau relation. $square$

*Theorem (Immerman 1982, Vardi 1982).* $"P" = "FO + LFP"$ over *ordered*
structures: properties expressible in first-order logic augmented with the least
fixed-point operator.

The order is essential: without it, FO+LFP cannot count or even detect parity. On
ordered finite structures, fixed-point iteration corresponds to polynomial-time
DTM computation.

*Immerman--Szelepcsényi as descriptive.* $"NL" = "FO + TC"$ (transitive closure).
$"NL" = "coNL"$ is the model-theoretic statement that FO+TC is closed under
negation.

Descriptive complexity recasts machine-based classes as *logical* classes, which
clarifies what kinds of properties live where and is essential to database theory
and finite model theory.

== Fine-Grained Complexity

A 21st-century shift: instead of asking "polynomial vs exponential", ask "is the
*exact* polynomial exponent optimal?". Conditional lower bounds rest on a small
set of plausible hypotheses.

*Strong Exponential Time Hypothesis (SETH, Impagliazzo--Paturi 1999).* For every
$epsilon > 0$, there exists $k$ such that $k$-SAT requires time $Omega(2^((1 -
epsilon) n))$.

*ETH.* 3-SAT requires time $2^(Omega(n))$.

*3-SUM hypothesis.* Determining whether a list of $n$ integers contains three
summing to zero requires time $n^(2 - o(1))$.

*Orthogonal Vectors (OV) hypothesis.* Given two sets $A, B$ of $n$ vectors in
${0, 1}^d$ with $d = omega(log n)$, deciding whether $exists a in A, b in B$ with
$a dot b = 0$ requires time $n^(2 - o(1))$.

*Theorem (Williams 2005).* SETH $=>$ OV hypothesis.

*Corollaries via fine-grained reductions.*

- *Edit distance* (Backurs--Indyk 2015): under SETH, no $O(n^(2 - epsilon))$
  algorithm. The reduction encodes OR-of-AND-of-equalities into edit-distance
  alignment gadgets.
- *Longest Common Subsequence* (Abboud--Hansen--Vassilevska Williams--Williams
  2016): under SETH, no $O(n^(2 - epsilon))$ algorithm, even for binary alphabet.
- *Fréchet distance* (Bringmann 2014): $n^(2 - o(1))$ under SETH.
- *APSP* (All-Pairs Shortest Paths) is conjectured to require $n^(3 - o(1))$;
  many cubic problems (negative triangle, radius, betweenness centrality) are
  equivalent under sub-cubic reductions (Vassilevska Williams--Williams).

The fine-grained perspective explains the *cubic*\/*quadratic*\/*subquadratic*
plateaus observed in algorithm engineering: long-standing barriers correspond to
provable conditional lower bounds. They also clarify which algorithmic
improvements are possible in principle and which would refute a major hypothesis.

```ocaml
(* A textbook quadratic edit distance: the SETH-conditional lower bound says
   we cannot beat this exponent unless we beat k-SAT for every fixed k. *)
let edit_distance a b =
  let n, m = String.length a, String.length b in
  let d = Array.make_matrix (n + 1) (m + 1) 0 in
  for i = 0 to n do d.(i).(0) <- i done;
  for j = 0 to m do d.(0).(j) <- j done;
  for i = 1 to n do
    for j = 1 to m do
      let cost = if a.[i - 1] = b.[j - 1] then 0 else 1 in
      d.(i).(j) <- min (min (d.(i - 1).(j) + 1)
                             (d.(i).(j - 1) + 1))
                       (d.(i - 1).(j - 1) + cost)
    done
  done;
  d.(n).(m)
```

== Alternating Computation

An *alternating TM* (Chandra--Kozen--Stockmeyer 1981) has states partitioned into
existential and universal. A configuration accepts if it is existential and some
successor accepts, or universal and every successor accepts. Define $"ATIME"$ and
$"ASPACE"$ analogously.

*Theorem (CKS 1981).*

- $"ATIME"(f) subset.eq "DSPACE"(f) subset.eq "ATIME"(f^2)$ for $f gt.eq log n$.
- $"ASPACE"(f) = "DTIME"(2^(O(f)))$.

*Corollaries.* $"AP" = "PSPACE"$; $"APSPACE" = "EXP"$; $"AL" = "P"$; $"AEXP" =
"EXPSPACE"$.

Alternation collapses the "one quantifier per level" structure of PH into a single
machine model: a $Sigma^p_k$ predicate is just a $k$-alternation ATM with
existential start.

*PSPACE-completeness via alternation.* TQBF (True Quantified Boolean Formulas) is
PSPACE-complete via reduction from $"APSPACE"(log n) = "AP"$. Game complexity:
generalised geography, generalised chess and Go (under suitable encoding) are
PSPACE-hard or EXP-hard via alternation reductions.

== Logspace and the L vs NL Frontier

*$"L"$-completeness.* Under log-space reductions $lt.eq^"log"_m$:

- *ORD* (deciding $x lt.eq y$ in a sorted list) is L-complete.
- *Undirected graph reachability* was famously shown to be in $"L"$ by Reingold
  (2008) via expander-based zig-zag walks -- a result conjectured for decades.

*$"NL"$-completeness.* *Directed reachability* ($"STCON"$) is NL-complete; so is
*$2"-SAT"$* (its complement is unsatisfiability, decidable in NL by guessing a
chain of forced implications). Layered DAG reachability, transitive closure
membership, $0/1$-knapsack with $log$-weight items.

*Implicit logarithmic-space algorithms.* Many "obvious" linear-time algorithms
actually fit in $log$-space when one is careful: matrix powering, group word
problem in solvable groups (Barrington 1989 for $"NC"^1$ via permutation
branching programs).

== Counting Classes and \#P

*Definition (Valiant 1979).* $\#"P"$ is the cal(C) of functions $f : Sigma^* -> NN$
such that $f(x) = |{ y | V(x, y) "accepts" }|$ for some poly-time verifier $V$
and polynomial bound on $|y|$. So $\#"P"$ counts NP-witnesses.

*$\#"P"$-complete problems.* $\#"SAT"$ (count satisfying assignments), $\#"3SAT"$,
$\#"HAMCYCLES"$, $"PERMANENT"$ (Valiant), $\#$-MATCHINGS, computing the partition
function of the Ising model.

*Theorem (Valiant 1979).* $"PERMANENT" in \#"P"$-complete -- even though it differs
from the determinant only in lacking signs, and determinant is in NC. *Proof
sketch.* Encode 3-SAT as a matrix whose permanent equals the number of satisfying
assignments times a known correction; gadgets enforce variable consistency and
clause coverage.

*Theorem (Toda 1991).* $"PH" subset.eq "P"^(\#"P")$. So counting is at least as
hard as the whole polynomial hierarchy. *Proof.* Two-stage: first show $"PH"
subset.eq "BP" dot plus.circle "P"$ (probabilistic parity), then $plus.circle "P" subset.eq
#"P"$. Beneath the technicalities: counting modulo 2 plus randomisation
simulates alternating quantifiers.

== Cryptographic Complexity and Trapdoors

Cryptography rests on average-case hardness assumptions stronger than $"P" eq.not
"NP"$:

- *One-way functions* (OWFs): poly-time computable, not poly-time invertible on a
  $1/"poly"$ fraction of inputs of each length. Existence => $"P" eq.not
  "NP"$ but is not known to follow from it.
- *Trapdoor permutations*: OWF with auxiliary inversion key.
- *Pseudorandom generators*: stretch $n$ bits to $n^omega(1)$ bits indistinguishable
  from uniform by poly-size circuits.

*Theorem (Håstad--Impagliazzo--Levin--Luby 1999).* OWFs exist iff PRGs exist iff
PRFs exist.

*Theorem (Impagliazzo 1995, "five worlds").* Depending on which assumptions hold,
we live in one of *Algorithmica* ($"P" = "NP"$), *Heuristica* ($"P" eq."not" "NP"$
but average-case easy), *Pessiland* (NP hard on average, no OWFs), *Minicrypt*
(OWFs but no public-key), or *Cryptomania* (trapdoor permutations). All but
Cryptomania are consistent with our current knowledge.

== Communication Complexity

Two parties Alice and Bob hold inputs $x, y$ and wish to compute $f(x, y)$ with
minimum communication. Models: deterministic $D(f)$, nondeterministic $N(f)$,
randomised $R(f)$, quantum $Q(f)$.

*Theorem (Yao 1979).* $D("EQ")_n = n$: equality testing requires $n$ bits
deterministically. $R("EQ")_n = O(log n)$ via hashing.

*Disjointness.* $f("DISJ")(x, y) = 1 arrow.l.r.double x sect y = emptyset$ for
$x, y subset.eq [n]$. *Theorem (Kalyanasundaram--Schnitger 1992; Razborov 1992).*
$R("DISJ")_n = Theta(n)$. The lower bound uses the *corruption method*: a random
restriction of inputs forces any small protocol to err on a constant fraction of
inputs.

Communication lower bounds yield circuit lower bounds via *Karchmer--Wigderson
games*: the depth of any monotone circuit for $f$ equals the deterministic
communication complexity of a specific relational problem on YES/NO pairs.

== Parameterised Complexity

Sometimes inputs come with a natural *parameter* $k$ (treewidth, solution size,
parameter of the formula). $"FPT"$ = problems solvable in time $f(k) dot
"poly"(n)$ for some computable $f$. The W-hierarchy $"FPT" subset.eq W[1] subset.eq
W[2] subset.eq dots subset.eq W[P] subset.eq "XP"$ classifies parameterised
intractability.

*Vertex Cover* is in FPT: $O(2^k n)$ via bounded search trees. *Clique* is
$W[1]$-complete: no $f(k) n^(O(1))$ algorithm under standard hypotheses. *Dominating
Set* is $W[2]$-complete.

*ETH connection.* Under ETH, $k$-CLIQUE requires $n^(Omega(k))$ time
(Chen--Huang--Kanj--Xia 2006).

== Quantum Complexity

*$"BQP"$* (Bernstein--Vazirani 1993): languages decided by a uniform family of
poly-size quantum circuits with bounded two-sided error.

- $"BPP" subset.eq "BQP" subset.eq "PSPACE"$; $"BQP" subset.eq "AWPP"$.
- $"BQP"$ vs $"NP"$: incomparable in known relativised worlds. $"NP"$ is not
  believed to be in $"BQP"$ (Grover gives only $sqrt(N)$ speedup, optimal).
- *Theorem (Shor 1994).* Integer factoring and discrete log are in $"BQP"$.
- *Theorem (Aaronson--Arkhipov 2011; Bremner--Jozsa--Shepherd 2010).* Sampling
  problems (BosonSampling, IQP) cannot be efficiently classically simulated
  unless PH collapses.

*$"QMA"$*: quantum analogue of NP -- prover sends quantum witness $|psi angle.r$,
verifier runs poly-size quantum circuit. $"NP" subset.eq "QMA" subset.eq "PP"
subset.eq "PSPACE"$. $k$-local Hamiltonian (estimating ground-state energy of a
$k$-local Hamiltonian to inverse-polynomial precision) is QMA-complete for $k gt.eq
2$ (Kitaev 1999; Kempe--Kitaev--Regev 2006 for $k = 2$).

== Algebraic Complexity

*Valiant's algebraic classes* (1979):

- *$"VP"$*: families of polynomials computable by poly-size *arithmetic circuits*.
- *$"VNP"$*: polynomials whose coefficients are computable as $sum_(arrow(e))
  f(arrow(e), arrow(x))$ over poly-many indices.
- $"VP" subset.eq "VNP"$; *Valiant's hypothesis*: $"VP" eq."not" "VNP"$, the
  algebraic analogue of $"P" eq."not" "NP"$.

*Theorem (Valiant 1979).* $"PERMANENT" in "VNP"$-complete (over any field of
characteristic $eq."not" 2$); $"DETERMINANT" in "VP"$.

*Geometric complexity theory* (Mulmuley--Sohoni 2001) attempts to prove $"VP"
eq.not "VNP"$ via representation theory of $"GL"_n$: separate orbit closures of
permanent and (padded) determinant by *occurrence obstructions*. Bürgisser--Ikenmeyer--
Panova (2019) showed occurrence obstructions alone are insufficient; the program
continues with finer multiplicity obstructions.

*$"VP" eq."not" "VNP"$ over finite fields* (the *boolean* version) implies
$\#"P" eq."not" "FP"$ in the appropriate non-uniform setting.

== Average-Case Complexity

*Distributional NP* ($"DistNP"$, Levin 1986): pairs $(L, mu)$ of an NP language
and a polynomial-time samplable distribution. A *poly-time on average* algorithm
runs in expected polynomial time with respect to $mu$.

*Levin's tiling problem* is DistNP-complete with respect to *polynomial-time
reductions that preserve distributional structure*. Average-case completeness is
subtle: most natural distributions are not believed to admit DistNP-complete
problems, but the existence of *some* DistNP-complete pair is established.

*Cryptographic significance.* Average-case hardness *of one-way function
distributions* is what enables cryptography; worst-case-to-average-case
reductions exist for some lattice problems (Ajtai 1996, Regev 2005 for $"LWE"$),
which is why post-quantum cryptography centres on lattices.

== Map of Major Inclusions

```text
              L  subset.eq  NL = coNL  subset.eq  NC  subset.eq  P
                                                                 |
                                                                 v
                                       coNP <- NP <- NP cap coNP <- BPP
                                                |     |          (Sigma_2 cap Pi_2)
                                                v     v
                                                Sigma_2^p, Pi_2^p
                                                       |
                                                       v
                                                      PH
                                                       |
                                                  (PH subset.eq PSPACE)
                                                       |
                                                       v
                                                    PSPACE = NPSPACE = IP
                                                       |
                                                       v
                                                      EXP
                                                       |
                                                       v
                                                     NEXP
                                                       |
                                                       v
                                                    EXPSPACE
```

Among these, *proper* inclusions are exactly: $"L" subset.eq."not" "PSPACE"$, $"NL"
subset.neq "PSPACE"$, $"P" subset.neq "EXP"$, $"NP" subset.neq "NEXP"$, $"PSPACE"
subset.neq "EXPSPACE"$ (all from hierarchy theorems). Everything else -- $"L"
"versus" "P"$, $"P" "versus" "NP"$, $"NP" "versus" "PSPACE"$, $"BPP" "versus" "P"$ -- is
*open after fifty years*.

== Why It Matters in Practice

- *NP-hardness* is the standard certificate that an industrial problem will not
  admit a worst-case-polynomial algorithm; it justifies turning to SAT/SMT
  solvers, ILP, local search, approximation, parameterised algorithms.
- *PSPACE-completeness* (games, model checking of LTL, planning) explains why
  game AI and verification tools blow up: the underlying problems are at least
  exponential in time.
- *PCP / hardness of approximation* tells you when even *approximating* a problem
  is hopeless, sparing effort spent on impossible approximation ratios.
- *Circuit lower bounds* and *natural proofs* shape what we believe is provable;
  they are why $"P" eq."not" "NP"$ is the central open problem of theoretical
  computer science, not just an unproven conjecture but one with quantified
  reasons why all standard techniques fail.
- *Fine-grained complexity* gives engineering meaning to "is this the right
  exponent?": for sequence problems quadratic is essentially optimal under SETH;
  for graph problems cubic is essentially optimal under APSP.

_See also: _Computability_ for the undecidable companion to this hierarchy --
NP-complete and PSPACE-complete are to feasible computation what $K$ and Tot are
to general computation -- and _Omega-Automata_ for the complexity of infinite-word
problems and parity games, whose membership in $"P"$ remains the most famous open
question in the analysis of fixed-point logics._

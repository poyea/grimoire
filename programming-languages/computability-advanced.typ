#import "../template.typ": xref

= Advanced Recursion Theory

This chapter continues from _Computability and Recursion Theory_, covering the
hyperarithmetical and analytical hierarchies, effective topology, computable
analysis, the fine structure of the r.e. degrees, forcing, and algorithmic
randomness. Together the two chapters give a complete account of the classical
theory through the 1970s and its connections to practice.

*See also:* #xref("programming-languages", "computability", label: "Computability and Recursion Theory")

== The Hyperarithmetical and Analytical Hierarchies

Beyond the arithmetical levels lies the *hyperarithmetical* hierarchy, indexed by
the recursive ordinals $alpha < omega_1^"CK"$ (the Church--Kleene ordinal, the
least non-recursive ordinal). For $alpha = beta + 1$, $emptyset^((alpha))$ is the
jump of $emptyset^((beta))$; for limit $alpha$ given by a recursive notation
$a$, $emptyset^((alpha)) = { chevron.l b, n chevron.r | b <_O a and n in
emptyset^((|b|)) }$ where $<_O$ is Kleene's $cal(O)$ ordering of notations. The
union $bold(H) = union.big_(alpha < omega_1^"CK") emptyset^((alpha))$ is the
*hyperarithmetical* class.

*Theorem (Suslin--Kleene).* $A$ is hyperarithmetical <==> $A in Delta^1_1$ (both
$Sigma^1_1$ and $Pi^1_1$).

The *analytical hierarchy* extends the arithmetical hierarchy with quantification
over *functions* $f : NN -> NN$.

- $Sigma^1_1$: $A = { x | exists f forall y . R(x, f overline(y), y) }$ with $R$
  decidable. Equivalently, $A$ is the projection of a $Pi^0_1$ class in Baire space.
- $Pi^1_1$: complement of $Sigma^1_1$. Equivalently, "$A$ is the set of trees with
  no infinite path" (well-foundedness).
- $Sigma^1_2$: $exists f$ of a $Pi^1_1$ matrix; and so on.

*Example.* "$e$ codes a recursive tree with an infinite path" is
$Sigma^1_1$; "$e$ codes a well-founded recursive tree" is $Pi^1_1$-complete. The
set of indices of total recursive functions is $Pi^0_2$-complete; the set of
*hyperarithmetical* indices is $Pi^1_1$.

== Effective Topology and Effective Descriptive Set Theory

In *effective descriptive set theory* we replace Borel hierarchies by *lightface*
analogues. A set $A subset.eq NN^NN$ (Baire space) is:

- $bold(Sigma)^0_1$: open (in product topology).
- $Sigma^0_1$ ("lightface"): *effectively* open -- a c.e. union of basic clopen
  sets $[sigma] = { f | sigma subset f }$ with the indices of $sigma$ c.e.
- $bold(Pi)^0_1$: closed. $Pi^0_1$: *effectively* closed -- complement of a c.e.
  union, i.e. the set of paths of a computable tree.
- $bold(Sigma)^1_1$: analytic (projection of closed in product space). $Sigma^1_1$:
  effectively analytic -- the projection of a $Pi^0_1$ set.

*Theorem (Kleene).* $A subset.eq NN$ is $Sigma^1_1$ <==> $A$ is the set of indices
of recursive trees that are *not* well-founded (i.e., that have an infinite path).
$A subset.eq NN$ is $Pi^1_1$ <==> $A$ is the set of indices of *well-founded*
recursive trees -- equivalently, $A$ is the complement of a $Sigma^1_1$ set.

*Theorem ($Pi^1_1$-uniformisation, Kondo--Addison).* Every $Pi^1_1$ relation $R
subset.eq NN times NN^NN$ has a $Pi^1_1$ uniformisation: a $Pi^1_1$ function $f$
with $(n, f(n)) in R$ whenever $exists g . (n, g) in R$.

Effective DST is the bridge between recursion theory and infinitary combinatorics;
in particular, $Pi^1_1$ sets behave very much like the complement of c.e. sets
one level up: $Pi^1_1$-completeness, $Pi^1_1$-singletons, and
*hyperarithmetical reduction* form a structural copy of the arithmetical world
indexed by countable ordinals.

== Computable Analysis (Weihrauch)

Recursion theory extends to functions on the reals via the Type-2 Theory of
Effectivity. A real $x in RR$ is *computable* <==> there is a computable sequence
of rationals $(q_n)$ with $|x - q_n| < 2^(-n)$. A function $f : RR -> RR$ is
*computable* <==> there is a TM with an input tape carrying an oracle for any name
of $x$ and an output tape producing arbitrary precision approximations to $f(x)$.

*Key facts.*

- All computable functions $RR -> RR$ are continuous. So $arrow(x) arrow.bar
  floor(x)$ is not computable.
- Equality of computable reals is undecidable (it is $Pi^0_1$-complete: equivalent
  to "all approximations agree forever").
- Differentiation is not computable; integration is.
- The Weihrauch lattice classifies the *uniform* computational content of theorems
  (intermediate value, Bolzano--Weierstrass, Hahn--Banach, ...). Reverse
  mathematics ($"RCA"_0, "WKL"_0, "ACA"_0, "ATR"_0, Pi^1_1"-CA"_0$) is the
  proof-theoretic counterpart.

== Oracle Machines and Relativisation

An *oracle Turing machine* $M^A$ has a distinguished *query tape* and three
oracle states $q_?, q_+, q_-$: writing a string $y$ on the query tape and
entering $q_?$ causes the machine to transition (in one step) to $q_+$ if $y in
A$ and $q_-$ otherwise. The oracle is consulted as a black box; its complexity
is irrelevant to the simulation cost.

*Definition.* $A lt.eq_T B$ <==> $A$ is decided by some oracle machine $M^B$ that
halts on every input. The *Turing degree* of $A$ is $deg(A) = { B | B equiv_T A
}$.

The set $cal(D) = NN^NN \/ equiv_T$ of degrees with order $lt.eq_T$ is an
upper semilattice with least element $bold(0) = deg(emptyset) = $ recursive
degrees and join $deg(A) join deg(B) = deg(A xor B)$ where $A xor B = { 2 n | n
in A } union { 2 n + 1 | n in B }$.

*Properties of $cal(D)$.*

- *Countable predecessors*: each degree has only countably many degrees below it
  (each computed by one of countably many oracle machines).
- *Uncountable size*: $|cal(D)| = 2^(aleph_0)$. Almost every degree -- in the
  measure-theoretic sense -- is between $bold(0)$ and $bold(0')$.
- *No maximal element*: the jump $A arrow.bar A'$ produces a strictly larger
  degree.
- *No minimal pair above $bold(0)$* in $cal(D)$ except $bold(0)$ itself, but
  *Spector 1956*: there are minimal pairs in $cal(D)$ (pairs $bold(a), bold(b)
  > bold(0)$ with $bold(a) inter bold(b) = bold(0)$).

*Relativisation.* Most computability results have *relativised* forms: for any
oracle $A$, $K^A = { e | Phi_e^A (e) "halts"}$ is $A$-r.e. but not $A$-recursive
($A$-diagonal); Rice relativises (any nontrivial property of $A$-partial
recursive functions is $A$-undecidable"); the recursion theorem relativises.
Diagonal arguments survive almost universally; relativisation barriers (the
inability to *separate* classes by techniques that survive relativisation) are
the complexity-theoretic shadow of this universality.

== Index Sets and Their Completeness

Beyond Rice's bare undecidability we want to *locate* index sets in the
arithmetical hierarchy. Soare's textbook contains a long catalogue; the proofs
follow a small set of templates.

*Theorem.* The following are $Pi^0_2$-complete:

- $"Tot" = { e | phi_e "total"} = { e | forall x exists s . T(e, x, s) }$.
- $"Inf" = { e | W_e "infinite"} = { e | forall n exists x > n . x in W_e }$.

*Reduction $"Tot" lt.eq_m "Inf"$.* Given $e$, define $g(e)$ via $s$-$m$-$n$ as
$phi_(g(e)) (n) = 1$ if $phi_e (0), phi_e (1), ..., phi_e (n)$ all halt, else
diverge. Then $phi_e$ is total <==> $W_(g(e))$ is infinite. *Reduction
$"Inf" lt.eq_m "Tot"$.* Symmetric: $phi_(g(e))(n)$ searches for an $x > n$ in $W_e$.

*$Pi^0_2$-hardness of $"Tot"$.* Reduce from the canonical $Pi^0_2$-complete set
$"Cof"(emptyset') = { e | forall n exists s > n . n in emptyset'_s }$ via direct
encoding.

*Theorem.* $"Fin" = { e | W_e "finite"} in Sigma^0_2$-complete; $"Cof" = { e |
overline(W_e) "finite"}$ and $"Rec" = { e | W_e "recursive"}$ are $Sigma^0_3$-
complete; $"Ext" = { e | exists "total" psi "extending" phi_e }$ is $Sigma^0_3$.

These are exact: a complete classification places every natural property at its
precise level. The arithmetical hierarchy is the unit of measurement for
"how undecidable" a property is.

== Limit Lemma and the $0'$-Recursive Sets

*Theorem (Shoenfield's limit lemma, 1959).* A set $A$ is computable in $emptyset'$
($A lt.eq_T emptyset'$, equivalently $A in Delta^0_2$) if and only if there is a
*total computable* function $f(x, s)$ such that

$ chi_A (x) = lim_(s -> oo) f(x, s) $

with $f(x, s) in {0, 1}$ and the limit existing for every $x$.

*Proof sketch.* ($=>$) Use $emptyset'$ as oracle to decide $chi_A (x)$;
since the answer is computable in $emptyset'$, finite-injury can be replaced by a
$emptyset'$-recursive enumeration whose stage-$s$ approximations converge.
($arrow.l.double$) Decide "$lim_s f(x, s) = 1$" using $emptyset'$ via $exists t
forall s gt.eq t . f(x, s) = 1$, a $Sigma^0_2$ predicate. $square$

The limit lemma is the working definition of $Delta^0_2$: sets you can "guess
and revise finitely often". The construction of Friedberg--Muchnik produces
sets in $Delta^0_2$ via exactly such guess-and-revise behaviour at each
requirement.

*Generalisation.* $A in Delta^0_(n + 1) arrow.l.r.double A = lim_(s_n) lim_(s_(n-1))
dots lim_(s_1) f(x, s_1, ..., s_n)$ -- the *limit hierarchy* matches the
arithmetical hierarchy level by level (Ershov).

== The Low and High Hierarchies

For an r.e. set $A$, define $A' = ${e | $Phi_e^A (e)$ halts$}$. Always $A' gt.eq_T
emptyset'$ and $A' lt.eq_T emptyset''$ if $A lt.eq_T emptyset'$.

- $A$ is *low* if $A' equiv_T emptyset'$ (it adds nothing to the halting problem).
- $A$ is *low*#sub[$n$] if $A^((n)) equiv_T emptyset^((n))$.
- $A$ is *high* if $A' equiv_T emptyset''$.

*Theorem (Sacks 1963).* Every nonzero r.e. degree is the supremum of two low r.e.
degrees. *Theorem (Robinson 1971).* Low r.e. degrees are dense within $cal(R)$.

Low sets are r.e. but jump-equivalent to $emptyset$: they are "almost computable"
in a precise jump-theoretic sense, and the Friedberg--Muchnik incomparable pair
can be chosen low. *High* r.e. sets behave more like $K$: every high r.e. degree
contains a *maximal* set (Martin 1966).

== Strong Reducibilities, $1$-Completeness, Myhill's Theorem

*Definition.* $A lt.eq_1 B$ ("one-one reducible") <==> there is an *injective*
total computable $f$ with $x in A arrow.l.r.double f(x) in B$. $A equiv_1 B$
means mutual $1$-reductions.

*Theorem (Myhill 1955).* $A equiv_1 B$ <==> $A$ and $B$ are *recursively isomorphic*:
there is a total computable bijection $h : NN -> NN$ with $A = h^(-1)(B)$.

*Proof.* The Schröder--Bernstein construction is made effective by interleaving
the two reductions $A lt.eq_1 B$ via $f$ and $B lt.eq_1 A$ via $g$, building $h$
in stages by back-and-forth. $square$

*Corollary.* All creative sets are recursively isomorphic. Up to recursive
isomorphism there is exactly *one* $m$-complete (equivalently $1$-complete) r.e.
set: $K$. Halting problems across machine models -- TM, RAM, lambda, Markov --
are not just bi-reducible but *the same set* under a computable relabelling.

*The truth-table reducibilities.* $A lt.eq_(t t) B$ <==> there is a computable $f$
that on $x$ produces a *list* of queries $arrow(y)$ and a truth-table $tau$ such
that $x in A arrow.l.r.double tau(chi_B (y_1), ..., chi_B (y_k)) = 1$. *Key
property*: $lt.eq_(t t)$ is *transitive* and weaker than $lt.eq_m$ but stronger
than $lt.eq_T$. Mostowski (1955) showed there are r.e. sets $A, B$ with $A
lt.eq_T B$ but $A lt.eq_(t t)slash B$.

== Forcing in Arithmetic and Effective Genericity

Cohen's set-theoretic forcing has an *effective* analogue. A condition is a
finite binary string $sigma in 2^(< omega)$ approximating the characteristic
function of a generic set $G$. A set $D$ of conditions is *dense* if every
$sigma$ has an extension in $D$. $G$ is *$n$-generic* if for every $Sigma^0_n$
dense set of conditions, some initial segment of $G$ lies in it.

*Theorem.* For each $n$, there is a $n$-generic set $G lt.eq_T emptyset^((n))$.
*Theorem (Jockusch 1980).* Every $1$-generic set is of hyperimmune degree, hence
not r.e. and not co-r.e.

Genericity arguments are the *non-priority* alternative for many incomparability
constructions: the Kleene--Post result above is one line of forcing.

== The Recursion Theorem with Parameters

*Theorem (effective fixed-point theorem with parameters).* For every total
computable $f(e, arrow(x))$ there is a total computable $h(arrow(x))$ such that

$ phi_(h(arrow(x))) = phi_(f(h(arrow(x)), arrow(x))) quad "for all" arrow(x). $

So the recursion theorem is *uniform*: the fixed point depends computably on any
parameters. This is what licences self-referential constructions to carry side
parameters -- you can build a quine that prints its source *and* a fixed input
chosen at construction time.

*Theorem (double recursion).* For every pair of total computable $f, g$ there are
$a, b$ with $phi_a = phi_(f(a, b))$ and $phi_b = phi_(g(a, b))$. Two mutually
recursive programs can simultaneously fix-point themselves.

*Application: Smullyan's double diagonal.* In provability logic, the Gödel--
Carnap fixed-point lemma (every $phi(x)$ has a sentence $sigma$ with $"PA" tack.r
sigma arrow.l.r.double phi(chevron.l sigma chevron.r)$) is the proof-theoretic shadow
of the recursion theorem. The proof of Gödel's incompleteness theorem is then
the same diagonal that proves $K$ undecidable.

== Computable Model Theory and Reverse Mathematics

*Computable algebra.* A countable structure is *computable* if its domain is $NN$
and its functions/relations are computable. Many structural questions become
non-trivial:

- *Theorem (Frohlich--Shepherdson 1956).* There is a computable field with no
  computable splitting algorithm (so no computable factorisation of polynomials).
- *Theorem (Rabin 1960).* Every computable field has a computable algebraic
  closure, but the embedding need not be computably unique.

*Reverse mathematics* (Friedman, Simpson) asks: which axioms of second-order
arithmetic are *needed* to prove a given mathematical theorem? The big five:

- $"RCA"_0$: $Delta^0_1$-comprehension + $Sigma^0_1$ induction. The base; captures
  "computable mathematics".
- $"WKL"_0$: + Weak König's lemma (every infinite binary tree has a path).
  Proves Heine--Borel, Brouwer fixed point, Gödel completeness for countable
  languages.
- $"ACA"_0$: + arithmetical comprehension. Proves Bolzano--Weierstrass, sequential
  compactness, Ramsey for triples.
- $"ATR"_0$: + arithmetical transfinite recursion. Proves comparability of
  well-orderings.
- $Pi^1_1 "-CA"_0$: + $Pi^1_1$-comprehension. Proves Cantor--Bendixson,
  $Sigma^1_1$-separation.

Many theorems of analysis correspond *exactly* to one of these systems -- the
classification recovers a computability-theoretic shadow of every standard
theorem.

== Algorithmic Randomness

A binary sequence $X in 2^omega$ is *Martin-Löf random* (1966) if it passes every
*effective statistical test*: for every uniformly c.e. sequence $(U_n)$ of open
sets in $2^omega$ with $mu(U_n) lt.eq 2^(-n)$, $X in."not" inter_n U_n$.

*Theorem (universal test).* There is a universal Martin-Löf test, so the class of
ML-random sequences has measure $1$ and is $Pi^0_2$.

*Theorem (Levin--Schnorr).* $X$ is ML-random <==> its *prefix-free Kolmogorov complexity*
satisfies $K(X harpoon.rt n) gt.eq n - O(1)$ for all but finitely many $n$.

*Chaitin's $Omega = sum_(p "halts") 2^(-|p|)$* (the halting probability) is the
canonical Martin-Löf random real. $Omega$ is left-c.e. (its rationals approaching
from below are c.e.) and ML-random, hence not computable. Knowing $n$ bits of
$Omega$ allows one to decide the halting problem for all programs of length
$lt.eq n$.

*Theorem (Kucera--Gács).* Every set $A$ is Turing-reducible to some ML-random
set. Consequently ML-random sets are not all in a single degree: they spread
across uncountably many degrees. *Theorem (Kucera 1985).* Every Turing degree
$gt.eq bold(0')$ contains an ML-random set; moreover every non-computable r.e.
degree contains an ML-random.

The theory connects measure (almost every sequence), category (comeager many
sequences), and computability (which random sequences a given oracle can
recognise) into a single hierarchy of strength: *Martin-Löf random* $subset$
*computably random* $subset$ *Schnorr random* $subset$ *Kurtz random* (each
class strictly contains the next, with ML-randomness the strongest and Kurtz
the weakest). Relativising to an oracle $A$: $X$ is $A$-ML-random iff
$K^A(X harpoon.rt n) gt.eq n - O(1)$ for all but finitely many $n$, where
$K^A$ is prefix-free complexity with $A$ as oracle.

== Where Recursion Theory Touches Practice

- *Decidable fragments.* Type checking, model checking, regular language
  equivalence, Presburger arithmetic are decidable; first-order Peano, the lambda
  calculus's $beta eta$ convertibility (on closed terms it is decidable; in
  general undecidable), and program equivalence are not. The arithmetical hierarchy
  predicts *"where"* in the difficulty spectrum a problem sits.
- *The recursion theorem* is the formal underpinning of self-modifying code,
  reflective towers, metacircular interpreters, and -- in the small -- of Haskell's
  `fix :: (a -> a) -> a`.
- *Productive sets* explain why every "complete" type system is incomplete:
  Gödel's theorem says the consequence relation of arithmetic is productive, so
  no r.e. axiomatisation captures it.
- *The priority method* has no direct programming analogue but shapes our
  expectations about r.e. structure -- and hence about the structure of
  semi-decidable problems in verification.

```haskell
-- Kleene's fix-point combinator in Haskell: a direct expression of the"
-- second recursion theorem at the term level.
fix :: (a -> a) -> a
fix f = let x = f x in x

-- Quine: a program whose output is its own source.
-- (Schematic: "the real thing requires escaping the string literal.)
quine :: IO ()
quine = let s = "quine = let s = ... in putStr (...)" in putStr (...)
```

The recursion-theoretic perspective is what turns ad-hoc undecidability folklore
into a coherent map: every "this is undecidable" claim in software lives at some
level $Sigma^0_n$ or $Pi^0_n$ , reduces to some canonical complete problem, and
inherits its degree from a small library of templates.

_See also:_ _Turing Machines and Computability_ for the machine model, _Complexity
Theory_ for the analogous classification of the *feasible* fragment of the
recursive sets, and _Type Systems_ for syntactic restrictions designed to land
inside the decidable fragment.

== Further Reading

Rogers, H. (1967). _Theory of Recursive Functions and Effective Computability_. MIT Press. The classical reference; comprehensive treatment of indices, reducibilities, and the arithmetical hierarchy.

Odifreddi, P. (1989). _Classical Recursion Theory_, Vol. I. Elsevier. The comprehensive reference for advanced recursion theory including priority arguments, the Turing degrees, and the hyperarithmetical hierarchy.

Soare, R. I. (1987). _Recursively Enumerable Sets and Degrees_. Springer. The definitive treatment of the r.e. degrees and priority methods.

Simpson, S. G. (2009). _Subsystems of Second Order Arithmetic_, 2nd ed. Cambridge. The authoritative reference for reverse mathematics and its connection to recursion theory.

Nies, A. (2009). _Computability and Randomness_. Oxford. Covers algorithmic randomness, Kolmogorov complexity, and their interaction with the Turing degrees.

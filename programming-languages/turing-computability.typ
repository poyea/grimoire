= Turing Machines and Computability

The Turing machine is not a model of modern hardware -- it is a model of *what can be
computed at all*. Every question about whether a program terminates, whether two programs
are equivalent, or whether a program has a given property is ultimately a question about
Turing machines. The answers are mostly negative, and understanding why shapes the design
of every tool that analyzes programs.

_See also: Type Systems for how type systems carve out
decidable sub-problems by restricting the language to a sub-Turing-complete fragment._

== Turing Machine Definition

A *deterministic Turing machine* (DTM) is a 7-tuple $(Q, Sigma, Gamma, delta, q_0, q_"accept", q_"reject")$:

- $Q$ -- finite set of states
- $Sigma$ -- input alphabet ($square in.not Sigma$)
- $Gamma supset.eq Sigma union { square }$ -- tape alphabet
- $delta : Q times Gamma -> Q times Gamma times {L, R}$ -- transition function
- $q_0$ -- start state
- $q_"accept" in Q$ -- accept state
- $q_"reject" in Q$ -- reject state ($q_"reject" != q_"accept"$)

The TM has a two-way infinite tape, a read/write head, and a finite control. At each step
it reads the tape symbol under the head, writes a new symbol, moves the head left (L) or
right (R), and transitions to a new state. Computation halts if it enters $q_"accept"$ or
$q_"reject"$.

=== Configurations and Computation

A *configuration* is a triple $(q, t, p)$ (state, tape contents, head position), more compactly written as $u q v$ where
$q$ is the current state, $u$ is the tape content left of (and not including) the head,
and $v$ is the tape content from the head rightward. The *initial configuration* on input
$w$ is $q_0 w$.

A *computation* is a sequence of configurations $C_1 |-_M C_2 |- dots$ where each step
follows $delta$. The TM *accepts* $w$ if the computation reaches a configuration containing
$q_"accept"$, *rejects* if it reaches $q_"reject"$, and *loops* if neither ever occurs.

The language *decided* by $M$ is $L(M) = { w | M "accepts" w }$ provided $M$ halts on
every input. If $M$ may loop, it only *recognizes* $L(M)$.

== Multi-Tape TMs and NDTMs

=== Multi-Tape TMs

A $k$-tape TM has $k$ independent tapes and heads, with $delta : Q times Gamma^k -> Q times Gamma^k times {L,R}^k$. Every $k$-tape TM can be simulated by a single-tape TM:

- *Space:* the single tape interleaves $k$ tracks (one per original tape) plus $k$
  marker bits indicating head positions.
- *Time:* simulating one step of the $k$-tape machine costs $O(s)$ sweeps over the
  single tape where $s$ is the used tape length. Since $s <= T(n)$ after $T(n)$ steps,
  the total cost is $sum_(i=1)^(T(n)) O(i) = O(T(n)^2)$. So if the original computation
  uses time $T(n)$ and space $S(n)$, the simulation costs $O(T(n)^2)$ time and
  $O(k dot S(n))$ space.

Multi-tape TMs are exponentially more convenient to program but only polynomially slower
when simulated -- this is why they do not change the class of *decidable* languages, only
the constant factors.

=== Nondeterministic TMs

A *nondeterministic TM* (NDTM) has $delta : Q times Gamma -> 2^(Q times Gamma times {L,R})$.
It accepts $w$ if *some* branch of the computation tree accepts. Every NDTM can be simulated
by a 3-tape DTM via *breadth-first search* over the computation tree:

- Tape 1: input (read-only).
- Tape 2: simulation of the current branch.
- Tape 3: address of the current branch in the tree (encoded in the branching factor $b$
  of the NDTM).

If the NDTM accepts within $T(n)$ steps, the BFS simulation uses time $O(b^{T(n)})$ and
space $O(T(n))$. The exponential blowup is unavoidable in general and is precisely the
$P$ vs $"NP"$ question.

== Universal Turing Machines

A *universal TM* $U$ takes as input an encoding $angle.l M, w angle.r$ of a TM $M$ and
an input $w$, and simulates $M$ on $w$:

- $U$ accepts iff $M$ accepts $w$.
- $U$ rejects iff $M$ rejects $w$.
- $U$ loops iff $M$ loops on $w$.

Encoding: represent each state as a binary string, each tape symbol as a binary string,
and each transition rule as a tuple of such strings, separated by delimiters. The encoding
$angle.l M angle.r$ is a finite string over $Sigma = {0, 1}$. The existence of $U$
establishes that computation is *data*: a program is just another input. This is the
foundation of every interpreter, JIT, and operating system loader.

== The Church-Turing Thesis

The *Church-Turing thesis* states: every function that is *intuitively computable* (by any
systematic procedure) is computable by some Turing machine. This is not a theorem -- it
cannot be, because "intuitively computable" is not formal -- but it is supported by decades
of evidence: every independently proposed model of computation (lambda calculus, partial
recursive functions, register machines, RAM machines, cellular automata, quantum circuits
for classical functions) has been proved equivalent to TMs in computational power. The
thesis licenses using TMs as the *definition* of computability, not just one model among
many.

== Undecidability: The Halting Problem

*Theorem (Turing 1936):* The *halting problem*
$A_"TM" = { angle.l M, w angle.r | M "is a TM and" M "accepts" w }$
is undecidable.

*Proof by diagonalization.* Suppose for contradiction that a DTM $H$ decides $A_"TM"$:

```
// Hypothetical decider H:
// Input: <M, w>
// Output: ACCEPT if M accepts w, REJECT otherwise (always halts)

define D(input <M>):
    run H on <M, <M>>      // ask: does M accept its own description?
    if H accepts:
        LOOP FOREVER       // or equivalently, reject
    else:
        ACCEPT
```

Now run $D$ on $angle.l D angle.r$:

- If $D$ accepts $angle.l D angle.r$: then $H$ must have accepted $angle.l D, angle.l D angle.r angle.r$,
  meaning $D$ accepts $angle.l D angle.r$ -- so $D$ should loop forever. Contradiction.
- If $D$ loops on $angle.l D angle.r$: then $H$ rejected $angle.l D, angle.l D angle.r angle.r$,
  meaning $D$ does not accept $angle.l D angle.r$ -- so $D$ should accept. Contradiction.

Both cases are impossible, so $H$ cannot exist. The diagonal argument works because we
construct $D$ to *disagree* with $H$'s prediction on the self-referential input.

*Corollary:* $A_"TM"$ is recognizable (the universal TM recognizes it) but not decidable.
Its complement $overline(A_"TM")$ is not even recognizable.

== Rice's Theorem

*Theorem (Rice 1953):* Every non-trivial semantic property of TMs is undecidable.

Formally: let $P$ be any property of TM-recognized languages such that $P$ is not
identically true and not identically false (non-trivial), and that $P$ depends only on the
*language* not the machine description (semantic). Then the set
${ angle.l M angle.r | L(M) "has property" P }$ is undecidable.

*Proof intuition:* Suppose $P$ holds of $L(M_1)$ and not of $L(M_emptyset)$ (where
$M_emptyset$ accepts nothing). For any TM $T$ and input $w$, build a machine $M$ that
ignores its own input, simulates $T$ on $w$, and if $T$ accepts, runs $M_1$ on the
original input. Then $L(M) = L(M_1)$ if $T$ accepts $w$, and $L(M) = emptyset$ otherwise.
Deciding $P$ for $M$ would decide $A_"TM"$ for $(T, w)$.

Rice's theorem is the formal statement of an empirical truth every developer has felt: you
cannot write a perfect static checker for a general-purpose language. Every attempt to
analyze what a program *does* -- not just what it *says* -- reduces eventually to the
halting problem.

== Recursive and Recursively Enumerable Languages

#table(
  columns: (auto, auto, auto),
  [*Class*], [*Machine*], [*Closure properties*],
  [Recursive (decidable)],    [DTM that always halts],         [Union, intersect, complement, concat, star],
  [RE (semi-decidable)],      [DTM that may loop on rejection],[Union, concat, star; NOT complement],
  [co-RE],                    [Complement of RE],              [Intersection, complement; NOT union],
)

A language $L$ is *decidable* iff both $L$ and $overline(L)$ are RE (Post's theorem).
The decidable languages are the sweet spot: machines that always answer. RE-but-not-decidable
languages are where algorithms get stuck: you can confirm "yes" answers but never rule out
"no" answers; the program might just be taking a very long time.

== Many-One Reductions

A *many-one reduction* from $A$ to $B$, written $A <=_m B$, is a computable function
$f : Sigma^* -> Sigma^*$ such that for all $w$: $w in A <=> f(w) in B$.

Reductions establish relative hardness:
- If $A <=_m B$ and $B$ is decidable, then $A$ is decidable.
- Contrapositive: if $A$ is undecidable and $A <=_m B$, then $B$ is undecidable.

*Equivalence of TMs is undecidable* via halting reduction: define $f(angle.l M, w angle.r)
= angle.l M', M_emptyset angle.r$ where $M'$ ignores its input, simulates $M$ on $w$, and
accepts if $M$ accepts. Then $M$ accepts $w$ iff $L(M') != L(M_emptyset)$, so deciding
equivalence would decide the halting problem. Thus $"EQ"_"TM"$ is undecidable (and in
fact not even RE or co-RE).

== Why Static Analysis Is Forever Bounded

Rice's theorem does not say static analysis is useless -- it says *perfect* static analysis
is impossible for any non-trivial semantic property. In practice, tools accept one of three
limitations:

- *Unsoundness:* miss some true defects (most linters, most type checkers in dynamic
  languages). Fast and useful, but certifies nothing.
- *Incompleteness:* report false positives (abstract interpretation, model checking with
  over-approximation). Sound but may reject valid programs.
- *Non-termination:* potentially run forever on hard inputs (symbolic execution, SAT-based
  analysis without bounds). Correct when it finishes, unusable in the worst case.

The *only* escape from this trilemma is to *restrict the language*. If the language is
not Turing-complete -- if all programs provably terminate -- then many semantic properties
become decidable. This is the design philosophy behind:

- *Regular expressions:* linear-time membership, decidable equivalence.
- *SQL (core):* set-based semantics, no general recursion.
- *Coq/Lean's termination checker:* structural recursion ensures all functions terminate.
- *Rust's borrow checker:* a sub-TC analysis of ownership and lifetimes.

_See also: Type Systems for how type systems deliberately
restrict expressiveness to regain decidability -- Hindley-Milner inference is decidable
precisely because the language of types is sub-Turing-complete._

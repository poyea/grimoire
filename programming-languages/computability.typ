= Computability and Recursion Theory

Recursion theory is the mathematical study of what can be computed, and -- more
importantly -- of the fine structure of what *cannot*. Turing machines give the
extensional definition of computability; the recursion-theoretic vocabulary
(indices, the $s$-$m$-$n$ theorem, fixed points, reducibilities, degrees, "the"
arithmetical hierarchy) is the intensional language in which actual proofs about
undecidability are written.

*See also:* _Turing Machines and Computability_, _Advanced Recursion Theory_, _Type Systems_, _Complexity Theory_

== Primitive Recursive and $mu$-Recursive Functions

We work with partial functions $f : NN^k harpoon.rt NN$. The class of *primitive
recursive* functions $cal(P R)$ is the smallest class containing the *initial
functions*

- *Zero*: $Z(x) = 0$
- *Successor*: $S(x) = x + 1$
- *Projection*: $P^n_i (x_1, ..., x_n) = x_i$ for $1 lt.eq i lt.eq n$

closed under *composition*

$ "Comp"(h, g_1, ..., g_m)(arrow(x)) = h(g_1(arrow(x)), ..., g_m(arrow(x))) $

and *primitive recursion*

$ f(arrow(x), 0) &= g(arrow(x)) \
  f(arrow(x), y + 1) &= h(arrow(x), y, f(arrow(x), y)) $

Every primitive recursive function is total. Addition, multiplication,
exponentiation, bounded minimisation, the Cantor pairing $angle.l x, y angle.r =
((x + y)(x + y + 1))/2 + y$, prime enumeration, and Gödel's $beta$-function are
primitive recursive. The class is closed under bounded quantification and bounded
search, which is what makes it powerful enough to encode finite sequences and thus
to do syntactic manipulation on programs.

But $cal(P R)$ is *not* all of computability. The Ackermann function

$ A(0, n) &= n + 1 \
  A(m + 1, 0) &= A(m, 1) \
  A(m + 1, n + 1) &= A(m, A(m + 1, n)) $

is total computable but eventually dominates every primitive recursive function
(Ackermann 1928; see also the Sudan function). To capture all computable functions
we close $cal(P R)$ under *unbounded minimisation*

$ mu y . [f(arrow(x), y) = 0] = "least " y "such that" f(arrow(x), y) = 0
  and f(arrow(x), z) "is defined for all" z lt.eq y $

The resulting class is $cal(P)$, the *partial $mu$-recursive* functions. The
*total* $mu$-recursive functions form a strict subclass, the *general recursive*
functions (Gödel--Herbrand--Kleene). A central, nontrivial fact: there is no
recursive enumeration of indices of total recursive functions -- if there were, a
diagonal argument would produce a recursive function not in the list.

*Theorem (Kleene normal form, 1936).* There exists a primitive recursive predicate
$T(e, x, y)$ ("$y$ encodes a halting computation of program $e$ on input $x$") and
a primitive recursive *result extractor* $U$ such that every partial computable
function $phi_e$ admits

$ phi_e (x) = U(mu y . T(e, x, y)). $

*Significance.* Every partial computable function is built from primitive recursion
plus *one* outer $mu$. Unbounded search is the *only* source of unbounded power.

*Theorem (Church--Turing--Kleene equivalence).* The following classes of partial
functions $NN^k harpoon.rt NN$ coincide:

+ Turing-computable functions
+ Partial $mu$-recursive functions
+ Lambda-definable functions (Church 1936)
+ Markov-algorithm computable functions
+ RAM-computable functions
+ Register-machine computable functions (Shepherdson--Sturgis 1963)

*Proof outline.* The cycle $"TM" arrow.r mu"-rec" arrow.r lambda arrow.r "TM"$
suffices. The first inclusion (Kleene) encodes TM configurations as natural numbers
via Gödel numbering and expresses one step of transition as a primitive recursive
function; halting is then a single $mu$. The second (Kleene) gives a lambda term
$Y = lambda f . (lambda x . f (x x))(lambda x . f (x x))$ to internalise
recursion. The third is direct simulation. $square$

The *Church--Turing thesis* asserts that every function effectively computable in
the informal sense is one of these. It is not a theorem: "effectively computable"
has no antecedent mathematical definition. It is an *empirical* claim, supported by
the failure of every honest attempt over ninety years to define a model that
strictly exceeds Turing power without invoking physically unrealisable resources
(hypercomputation, infinite-time TMs, oracle access to undecidable sets).

== Encoding Programs as Numbers

Fix a bijective Gödel numbering of TMs (equivalently $mu$-recursive descriptions).
Let $phi_e$ denote the partial computable function with index $e$, and let
$phi_(e, s) (x)$ denote the result of running $e$ on $x$ for at most $s$ steps
(undefined if not yet halted). The two-place enumeration

$ Phi : NN times NN harpoon.rt NN, quad Phi(e, x) = phi_e (x) $

is itself a partial computable function. This is the *universal function*; the
machine computing it is the universal TM.

== The $s$-$m$-$n$ Theorem (Parametrisation)

*Theorem ($s$-$m$-$n$, Kleene).* For all $m, n gt.eq 1$ there is a *total*
primitive recursive injection $s^m_n : NN^(m + 1) -> NN$ such that

$ phi_(s^m_n (e, x_1, ..., x_m)) (y_1, ..., y_n) = phi_e (x_1, ..., x_m, y_1, ..., y_n). $

*Proof.* Given $e$ and parameters $arrow(x)$, syntactically construct the source
of a program that hard-codes $arrow(x)$ and then dispatches to $e$ with the
combined argument list. This rewrite is purely textual and so primitive recursive
in $(e, arrow(x))$. $square$

In programmer terms, $s^m_n$ is *partial application* at the level of source code,
made into a constructive operation on indices. It is the bridge between the
*denotational* world of computable functions and the *syntactic* world of programs:
any time you have a uniform construction of a program from parameters, $s$-$m$-$n$
turns that construction into a single index.

== The Recursion Theorem

*Theorem (Kleene's second recursion theorem, 1938).* For every total computable
function $f : NN -> NN$ there exists an index $e$ such that $phi_e = phi_(f(e))$.

*Proof.* Define $g(x, y)$ by $g(x, y) = phi_(phi_x (x)) (y)$ if the inner call
converges, undefined otherwise. By $s$-$m$-$n$, fix $d$ total computable with
$phi_(d(x)) (y) = g(x, y)$. Let $h = f circle.small d$, which is total computable; let
$v$ be an index for $h$, so $phi_v (x) = f(d(x))$. Set $e = d(v)$. Then

$ phi_e (y) = phi_(d(v)) (y) = g(v, y) = phi_(phi_v (v)) (y) = phi_(f(d(v))) (y) = phi_(f(e)) (y). $

So $phi_e = phi_(f(e))$. $square$

The proof is a *Quine*: a program that builds its own description and then applies
$f$ to it. The Python sketch:

```python
def quine(f):
    template = (
        "def prog(y):\n"
        "    src = {src!r}\n"
        "    e = compile_to_index(src.format(src=src))\n"
        "    return run(f(e), y)\n"
    )
    src = template.format(src=template)
    return compile_to_index(src)
```

Two consequences make the recursion theorem one of the most useful tools in all of
recursion theory.

*Corollary (Fixed-point form).* The map $e arrow.bar phi_(f(e))$ on indices has a
*fixed point modulo extensional equality*: some index codes a program whose
behaviour is invariant under $f$.

*Corollary (Programs printing their own source).* Apply the theorem with $f$ the
total function "ignore your input and print $e$". The resulting $e$ is a *quine*.
Every Turing-complete language admits one, by a fully effective construction.

*Application (Kleene's inseparability theorem).* The sets

$ A = { e | phi_e (e) = 0 }, quad B = { e | phi_e (e) = 1 } $

are *recursively inseparable*: there is no decidable set $C$ with $A subset.eq C$
and $C inter B = emptyset$. *Proof.* If $C$ were decidable with characteristic
function $chi_C$, let $f(e) = 1 - chi_C (e)$, total computable. By the recursion
theorem fix $e_0$ with $phi_(e_0) = $ constant $f(e_0)$. Then $phi_(e_0)(e_0) =
f(e_0) = 1 - chi_C (e_0)$. If $e_0 in C$ then $phi_(e_0)(e_0) = 0$ so $e_0 in A
subset.eq C$, consistent; but also $phi_(e_0)(e_0) = 1 - chi_C (e_0) = 0$ forces
$chi_C (e_0) = 1$, so $e_0 in C$, and yet $e_0 in B$ since $phi_(e_0)(e_0) = 1$
when we untangle the cases the other way. Working through both branches yields a
contradiction. $square$

== The Halting Problem and Diagonalisation

Define

$ K = { e | phi_e (e) "halts"} = { e | e in W_e }, $

the *self-halting set*, also called the *diagonal halting problem*. It is the
canonical undecidable set.

*Theorem (Turing 1936).* $K$ is r.e. but not recursive.

*Proof (RE).* The universal machine on input $e$ simulates $phi_e (e)$ and accepts
when it halts; this is a semi-decision procedure.

*Proof (not recursive).* Suppose $chi_K$ were computable. Define

```text
D(e):
    if chi_K(e) = 1:           // phi_e(e) halts
        loop forever
    else:                       // phi_e(e) diverges
        halt with output 0

Let d be an index for D.

Case 1: d in K, i.e. phi_d(d) halts.
        Then chi_K(d) = 1, so D(d) loops forever.
        But phi_d(d) = D(d), so phi_d(d) does not halt.   Contradiction.

Case 2: d not in K, i.e. phi_d(d) diverges.
        Then chi_K(d) = 0, so D(d) halts.
        But phi_d(d) = D(d), so phi_d(d) halts.            Contradiction.
```

Each case contradicts itself, so $chi_K$ cannot be computable. $square$

The structure of the argument is *Cantor's diagonal*: enumerate all candidate
deciders, pretend the $e$-th decides the $e$-th instance, then construct a program
that *disagrees* with itself at the diagonal point. The same template works for
every concrete undecidability result if one phrases the construction correctly.

The complement $overline(K) = { e | phi_e (e) "diverges" }$ is *productive*
(see below) and so very far from r.e.

== Rice's Theorem

A class $cal(A)$ of partial computable functions is *extensional* if membership of
$phi_e$ in $cal(A)$ depends only on $phi_e$ as a function, not on the index $e$.
The associated *index set* is $I_(cal(A)) = { e | phi_e in cal(A) }$.

*Theorem (Rice 1953).* If $cal(A) eq."not" emptyset$ and $cal(A)$ is not the set of
all partial computable functions, then $I_(cal(A))$ is undecidable.

*Proof.* WLOG the everywhere-undefined function $bot in."not" cal(A)$ (else work with
$overline(cal(A))$). Pick any $psi in cal(A)$, with index $i$. We reduce $K
lt.eq_m I_(cal(A))$. For each $e$, by $s$-$m$-$n$ build $g(e)$ such that

```text
phi_{g(e)}(x):
    simulate phi_e(e)            // ignore x while waiting
    if it halts, return psi(x)
    else diverge
```

Then $phi_(g(e)) = psi$ if $e in K$ and $phi_(g(e)) = bot$ otherwise. So $e in K
arrow.l.r.double phi_(g(e)) in cal(A) arrow.l.r.double g(e) in I_(cal(A))$. Since
$K$ is undecidable, so is $I_(cal(A))$. $square$

*Examples of undecidable properties.* "$phi_e$ is total"; "$phi_e$ is the constant
zero function"; "$phi_e$ is primitive recursive"; "$L(phi_e)$ is regular"; "$phi_e$
agrees with my reference implementation on all inputs"; "$phi_e$ runs in polynomial
time on its halting inputs". Every nontrivial *semantic* claim about source code is
beyond static analysis. Rice is the precise statement of why perfect linters do
not exist.

*Non-application.* Rice does *not* prohibit decidable *syntactic* properties such
as "$e$ has fewer than 100 instructions" or "$e$ uses no while-loops". Those are
not extensional; two indices for the same function can disagree on them.

== Rice--Shapiro: An RE Refinement

Which extensional properties are r.e. (not just undecidable)?

*Theorem (Rice--Shapiro, McNaughton--Myhill 1957).* An index set $I_(cal(A))$ is
r.e. if and only if there is an r.e. family $cal(F)$ of *finite* partial functions
such that

$ phi_e in cal(A) arrow.l.r.double exists theta in cal(F) . theta subset.eq phi_e. $

*Sketch.* ($arrow.l$) Enumerate $cal(F)$; for each $theta$ try to verify $theta
subset.eq phi_e$ by running $phi_e$ on $"dom"(theta)$. ($arrow.r$) Use a
finite-information argument: if $phi_e in cal(A)$ then this is witnessed after
finitely many computation steps and so depends on only a finite restriction of
$phi_e$; conversely if $phi_e in."not" cal(A)$ then some finite restriction is
already excluded. A careful application of the recursion theorem rules out
pathologies. $square$

*Consequence.* The set "$phi_e$ is total" is not r.e. (no finite extension forces
totality), and so totality is $Pi^0_2$-complete. The set "$phi_e$ halts on $0$" is
r.e. (witnessed by $theta = {(0, y)}$ for any $y$). The set "$L(phi_e) = NN$" is
not r.e., but "$L(phi_e) eq."not" emptyset$" is.

== Recursively Enumerable Sets

Let $W_e = "dom"(phi_e)$. A set $A subset.eq NN$ is *r.e.* <==> $A = W_e$ for some
$e$. Equivalent formulations:

+ $A$ is the range of a partial computable function.
+ $A$ is the range of a total computable function (if $A eq."not" emptyset$).
+ $A = { x | exists y . R(x, y) }$ for some decidable predicate $R$.
+ $A$ is the projection of a decidable subset of $NN^2$.
+ $A$ is the image of a recursive set under a computable map.

*Closure properties.* The r.e. sets are closed under: $union, inter, times$,
preimage under total computable maps, projection. They are *not* closed under
complement.

*Post's theorem (the easy half).* $A$ is decidable <==> both $A$ and $overline(A)$
are r.e. *Proof.* Run two semideciders in dovetailed fashion; whichever halts gives
the answer. $square$

*Productive sets (Dekker 1955).* A set $P$ is *productive* if there is a total
computable $g$ such that whenever $W_e subset.eq P$ then $g(e) in P backslash W_e$.
$g$ effectively *produces* a new element outside any r.e. subset. The canonical
example is $overline(K)$: given $e$ with $W_e subset.eq overline(K)$, by the
recursion theorem (uniformly in $e$) we can construct $g(e) in overline(K)
backslash W_e$. Productive sets are never r.e. -- much stronger than mere
non-recursiveness.

*Creative sets.* An r.e. set $A$ is *creative* if $overline(A)$ is productive. $K$
is creative. Myhill (1955): every creative set is $m$-complete for the r.e. sets,
hence all creative sets are recursively isomorphic. There is, up to recursive
isomorphism, *one* halting problem.

*Simple sets (Post 1944).* An r.e. set $S$ is *simple* if $overline(S)$ is
infinite but contains no infinite r.e. subset. Post constructed one to obtain an
r.e. set that is not recursive and not $m$-complete; this is the first step toward
showing the Turing degrees of r.e. sets are richer than the $m$-degrees suggest.
The construction proceeds by enumerating r.e. sets $W_e$ and, when $W_e$ becomes
large enough, throwing one of its elements into $S$ -- enough to kill $W_e$
without exhausting $overline(S)$.

*Hypersimple, hyperhypersimple, maximal sets.* A whole tower of refinements
(Dekker, Friedberg, Yates) carves up the r.e. degree below $K$.

== Many-One and Turing Reductions

*Many-one reduction.* $A lt.eq_m B$ <==> there is a total computable $f$ with $x in
A arrow.l.r.double f(x) in B$. We write $A equiv_m B$ for mutual reduction.

*Turing reduction.* $A lt.eq_T B$ <==> there is an *oracle* machine $Phi^B$ that
decides $A$ using a $B$-oracle (queries "$y in B$?" cost one step). We write $A
equiv_T B$ for mutual Turing reduction; the equivalence classes are the
*Turing degrees*.

*Inclusions.* $A lt.eq_m B => A lt.eq_T B$. The converse fails: $K
equiv_T overline(K)$ (an oracle for $K$ lets us decide $overline(K)$), but $K lt.eq_m
overline(K)$ is false because $m$-reductions preserve r.e.-ness and $overline(K)$
is not r.e.

*$m$-completeness.* An r.e. set $C$ is *$m$-complete* iff every r.e. set $m$-reduces
to $C$. $K$ is $m$-complete; equivalently $C$ is $m$-complete iff $C$ is creative
(Myhill).

*Strong reducibilities.* $lt.eq_1$ (injective many-one), $lt.eq_(t t)$
(truth-table, ask all queries non-adaptively and combine by a truth-table),
$lt.eq_(w t t)$ (weak truth-table, queries bounded by a computable function but
adaptive). These give a finer hierarchy than $lt.eq_T$.

== Post's Problem and the Priority Method

*Post's problem (1944).* Does there exist an r.e. set $A$ with $emptyset <_T A <_T
K$?

Until 1956 it was open. Post had shown that strong reducibility separations
(simple, hypersimple, ...) do not suffice: every example was either recursive or
$T$-equivalent to $K$. The breakthrough was a new proof technique.

*Theorem (Friedberg 1957, Muchnik 1956 independently).* There exist r.e. sets $A,
B$ with $A |_T B$ (incomparable under $lt.eq_T$), both strictly between $emptyset$
and $K$.

*The finite injury priority method.* We construct $A$ and $B$ in stages to satisfy
the *requirements*

$ R_(2 e) : Phi_e^B eq."not" chi_A, quad R_(2 e + 1) : Phi_e^A eq."not" chi_B. $

Each $R_(2 e)$ demands that the $e$-th oracle machine, given oracle $B$, fails to
compute the characteristic function of $A$. To satisfy $R_(2 e)$, wait for a stage
$s$ and a *witness* $x$ such that $Phi_e^(B_s) (x) arrow.b = 0$ with use $u$.
Then *enumerate* $x$ into $A$, and *restrain* $B$ from changing below $u$ at later
stages. This forces $Phi_e^B (x) = 0 eq."not" 1 = chi_A (x)$.

The conflict: a higher-priority requirement $R_(2 e')$ might later need to put
some $y < u$ into $B$, *injuring* $R_(2 e)$ by violating the restraint. Solution:
assign priorities $R_0 > R_1 > dots$; when $R_j$ injures $R_i$ with $i < j$ it
cannot (priority order); when $R_i$ injures $R_j$, $R_j$ simply restarts. Each
$R_i$ is injured at most $2^i - 1$ times, so eventually it acts permanently. A
finite-injury argument shows every requirement is satisfied in the limit.

```text
Stage s+1:
  for each requirement R_i in priority order (i = 0, 1, ..., s):
    if R_i needs attention at stage s:
      act to satisfy R_i
      cancel all current actions of R_j for j > i
```

*Result.* $A, B$ are r.e., $A |_T B$, $A, B lt.eq_T K$ (the construction is
computable in $K$), and neither is recursive (else some $R_(2 e)$ or $R_(2 e + 1)$
would be unsatisfiable). $square$

The priority method became the central technique of classical recursion theory:
Sacks's splitting and density theorems, Lachlan's nondiamond, and the entire
$0''$ /  $0'''$-priority machinery are extensions of this idea.

== The Structure of Turing Degrees

Let $cal(D) = (cal(D), lt.eq, join)$ denote the upper semilattice of Turing
degrees, with $deg(A) join deg(B) = deg(A xor B)$. Let $cal(R) subset.eq cal(D)$
be the r.e. degrees.

*Theorem (Kleene--Post 1954).* There are degrees $bold(a), bold(b) lt.eq bold(0')$
with $bold(a) |_T bold(b)$. The priority method is *not* required for this
non-r.e. result; finite-extension forcing suffices.

*Theorem (Sacks splitting, 1963).* Every nonrecursive r.e. degree $bold(a)$ splits:
there exist r.e. degrees $bold(b), bold(c) < bold(a)$ with $bold(b) join bold(c) =
bold(a)$ and $bold(b) |_T bold(c)$.

*Theorem (Sacks density, 1964).* For any r.e. degrees $bold(a) < bold(b)$ there is
an r.e. degree $bold(c)$ with $bold(a) < bold(c) < bold(b)$. So $cal(R)$ is dense.

*Theorem (Lachlan, Soare).* $cal(R)$ is not a lattice (some pairs lack infima) but
is still elementarily nontrivial. The first-order theory of $(cal(R), lt.eq)$ is
undecidable (Harrington--Shelah 1982); $"Th"(cal(D), lt.eq)$ is equivalent to second-
order arithmetic (Simpson 1977, Slaman--Woodin).

*The jump operator.* $A' = { e | Phi_e^A (e) "halts"}$, the halting problem
*relative* to $A$. The jump is *strictly* increasing: $A <_T A'$ for every $A$, by
the relativised diagonal argument. Iterating gives $emptyset, emptyset', emptyset'',
..., emptyset^((n))$. The infinitary jump $emptyset^((omega)) = { angle.l e, n
angle.r | e in emptyset^((n)) }$ goes beyond all finite levels.

== The Arithmetical Hierarchy

Stratify the arithmetically definable sets by quantifier alternation over decidable
matrices. A set $A subset.eq NN$ is

- $Sigma^0_0 = Pi^0_0 = Delta^0_0$: decidable.
- $Sigma^0_(n + 1)$: $A = { x | exists y . R(x, y) }$ with $R in Pi^0_n$.
- $Pi^0_(n + 1)$: $A = { x | forall y . R(x, y) }$ with $R in Sigma^0_n$.
- $Delta^0_n = Sigma^0_n inter Pi^0_n$.

So $Sigma^0_1$ = r.e., $Pi^0_1$ = co-r.e., $Delta^0_1$ = decidable.

#table(
  columns: (auto, auto, auto),
  [*Set*], [*Definition*], [*Level*],
  [$K$], [$exists s . T(e, e, s)$], [$Sigma^0_1$-complete],
  [Tot $= { e | phi_e "total" }$], [$forall x exists s . T(e, x, s)$], [$Pi^0_2$-complete],
  [Fin $= { e | W_e "finite" }$], [$exists n forall x > n . forall s . "not" T(e, x, s)$], [$Sigma^0_2$-complete],
  [Inf $= { e | W_e "infinite"}$], [$forall n exists x > n exists s . T(e, x, s)$], [$Pi^0_2$-complete],
  [Cof $= { e | overline(W_e) "finite" }$], [], [$Sigma^0_3$-complete],
  [Rec $= { e | W_e "recursive"}$], [], [$Sigma^0_3$-complete],
)

*Theorem (Post 1948).* $Sigma^0_(n + 1)$ is precisely the class of sets that are
r.e. relative to $emptyset^((n))$. Equivalently,

$ A in Sigma^0_(n + 1) arrow.l.r.double A "is r.e. in" emptyset^((n)), $

and consequently $Delta^0_(n + 1)$ is the class of sets *computable* in
$emptyset^((n))$.

*Proof sketch.* By induction. Base $n = 0$ is the definition of r.e. Step: a
$Sigma^0_(n + 1)$ set is $exists y . R(x, y)$ with $R in Pi^0_n$. The set $R$ is
co-r.e. in $emptyset^((n - 1))$, i.e. decidable in $emptyset^((n))$. So
membership in $A$ is r.e. in $emptyset^((n))$. Conversely, every set r.e. in
$emptyset^((n))$ is the projection of a $emptyset^((n))$-decidable predicate,
which unfolds to $Sigma^0_(n + 1)$ form. $square$

*Hierarchy theorem.* All inclusions $Sigma^0_n subset.eq Sigma^0_(n + 1)$ are
strict; $Sigma^0_n union Pi^0_n subset.eq."not" Delta^0_(n + 1)$. The jump operator
witnesses the strictness: $emptyset^((n))$ is $Sigma^0_n$-complete.

== Further Reading

Sipser, M. (2013). _Introduction to the Theory of Computation_, 3rd ed. Cengage Learning. Chapters 3--6 cover Turing machines, the Church-Turing thesis, decidability, and reducibility; the clearest pedagogical treatment.

Turing, A. M. (1936). "On Computable Numbers, with an Application to the Entscheidungsproblem." Proceedings of the London Mathematical Society 42. The founding paper; introduces the machine model and proves the halting problem undecidable.

Rogers, H. (1967). _Theory of Recursive Functions and Effective Computability_. MIT Press. The classical reference for recursion theory; covers indices, the recursion theorem, Rice's theorem, and the arithmetical hierarchy in depth.

Hopcroft, J. E., Ullman, J. D. (1979). _Introduction to Automata Theory, Languages, and Computation_. Addison-Wesley. Chapters 7--9 treat decidability, reducibility, and the recursively enumerable sets with full proofs.

Cutland, N. J. (1980). _Computability: An Introduction to the Theory of Computation_. Cambridge University Press. A rigorous and concise account of partial recursive functions, numberings, and undecidability results.

Odifreddi, P. (1989). _Classical Recursion Theory_, Vol. I. Elsevier. The comprehensive reference for advanced recursion theory including priority arguments and the structure of the Turing degrees.

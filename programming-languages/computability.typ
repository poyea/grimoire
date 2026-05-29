= Computability and Recursion Theory

Recursion theory is the mathematical study of what can be computed, and -- more
importantly -- of the fine structure of what *cannot*. Turing machines give the
extensional definition of computability; the recursion-theoretic vocabulary
(indices, the $s$-$m$-$n$ theorem, fixed points, reducibilities, degrees, "the"
arithmetical hierarchy) is the intensional language in which actual proofs about
undecidability are written.

_See also: _Turing Machines and Computability_, _Type Systems_, _Complexity Theory_._

== Primitive Recursive and $mu$-Recursive Functions

We work with partial functions $f : NN^k harpoon.rt NN$. The class of *primitive
recursive* functions $cal(P R)$ is the smallest cal(C) containing the *initial
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
eq.def quine(f):
    template = (
        "eq.def prog(y):\n"
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

A cal(C) $cal(A)$ of partial computable functions is *extensional* if membership of
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

*Theorem (Post 1948).* $Sigma^0_(n + 1)$ is precisely the cal(C) of sets that are
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

== The Hyperarithmetical and Analytical Hierarchies

Beyond the arithmetical levels lies the *hyperarithmetical* hierarchy, indexed by
the recursive ordinals $alpha < omega_1^"CK"$ (the Church--Kleene ordinal, the
least non-recursive ordinal). For $alpha = beta + 1$, $emptyset^((alpha))$ is the
jump of $emptyset^((beta))$; for limit $alpha$ given by a recursive notation
$a$, $emptyset^((alpha)) = { angle.l b, n angle.r | b <_O a and n in
emptyset^((|b|)) }$ where $<_O$ is Kleene's $cal(O)$ ordering of notations. The
union $bold(H) = union.big_(alpha < omega_1^"CK") emptyset^((alpha))$ is the
*hyperarithmetical* set.

*Theorem (Suslin--Kleene).* $A$ is hyperarithmetical <==> $A in Delta^1_1$ (both
$Sigma^1_1$ and $Pi^1_1$).

The *analytical hierarchy* extends the arithmetical hierarchy with quantification
over *functions* $f : NN -> NN$.

- $Sigma^1_1$: $A = { x | exists f forall y . R(x, f overline(y), y) }$ with $R$
  decidable. Equivalently, $A$ is the projection of a $Pi^0_1$ cal(C) in Baire space.
- $Pi^1_1$: complement of $Sigma^1_1$. Equivalently, "$A$ is the set of trees with
  no infinite path" (well-foundedness).
- $Sigma^1_2$: $exists f$ of a $Pi^1_1$ matrix; and so on.

*Example.* "$e$ codes a recursive tree with a recursive infinite path" is
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
of well-founded recursive trees. $A subset.eq NN$ is $Pi^1_1$ <==> $A$ is the
projection of a $Sigma^1_1$ set in $NN times NN$.

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
sigma arrow.l.r.double phi(angle.l sigma angle.r)$) is the proof-theoretic shadow
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
sets in $2^omega$ with $mu(U_n) lt.eq 2^(-n)$, $X in."not" sect_n U_n$.

*Theorem (universal test).* There is a universal Martin-Löf test, so the class of
ML-random sequences has measure $1$ and is $Pi^0_2$.

*Schnorr's theorem.* $X$ is ML-random <==> its *prefix-free Kolmogorov complexity*
satisfies $K(X harpoon.rt n) gt.eq n - O(1)$.

*Chaitin's $Omega = sum_(p "halts") 2^(-|p|)$* (the halting probability) is the
canonical Martin-Löf random real. $Omega$ is left-c.e. (its rationals approaching
from below are c.e.) and ML-random, hence not computable. Knowing $n$ bits of
$Omega$ allows one to decide the halting problem for all programs of length
$lt.eq n$.

*Theorem (Kucera--Gács).* Every set is Turing-reducible to a ML-random set.
Randomness does not collapse the Turing degrees -- there are ML-random sets in
every degree above $bold(0')$.

The theory connects measure (almost every sequence), category (comeager many
sequences), and computability (which random sequences a given oracle can
recognise) into a single hierarchy: *Schnorr random* $supset$ *computably random*
$supset$ *Martin-Löf random* $supset$ *Kurtz random*; with respect to oracles,
$X$-random iff $K(X harpoon.rt n) gt.eq n - O(1)$ relative to $X$.

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

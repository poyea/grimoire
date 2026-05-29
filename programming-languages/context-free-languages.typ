= Context-Free Languages

The context-free languages (CFLs) form the second layer of the Chomsky hierarchy and the operational sweet spot of programming-language syntax: rich enough to express balanced delimiters, nested scopes, and recursive expressions; tractable enough to admit cubic-time parsing in full generality and linear-time parsing for several practically important subclasses (LL($k$), LR($k$), LALR(1), GLR with sub-quadratic average case). Where regular languages are characterised by a *finite* syntactic invariant — the syntactic monoid — context-free languages are characterised by recognition with an *unbounded* but structured memory: a single pushdown stack. The cost of that unbounded memory is the loss of essentially every nontrivial decidability result: equivalence of CFGs is undecidable (Bar-Hillel–Perles–Shamir 1961), as are ambiguity, inclusion, intersection-emptiness with regular sets in cascades, and regularity of a CFL.

*See also:* _Regular Languages_, _Pushdown Automata and Beyond_, _Lexing_, _Type Systems_.

== Context-Free Grammars

A *context-free grammar* (CFG) is a 4-tuple $G = (V, Sigma, P, S)$:

- $V$ — finite set of *nonterminals* (or *variables*),
- $Sigma$ — finite set of *terminals*, disjoint from $V$,
- $P subset.eq V times (V union Sigma)^*$ — finite set of *productions* $A arrow.r alpha$,
- $S in V$ — *start symbol*.

The *one-step derivation* relation $=>_G$ on $(V union Sigma)^*$ is: $alpha A beta =>_G alpha gamma beta$ whenever $A arrow.r gamma in P$. Its reflexive-transitive closure is $=>^*_G$. The *language generated* by $G$ is

$ L(G) = { w in Sigma^* | S =>^*_G w } $

A language $L subset.eq Sigma^*$ is *context-free* if $L = L(G)$ for some CFG $G$.

*Leftmost* (resp. *rightmost*) derivations always rewrite the leftmost (rightmost) nonterminal; for every derivation there is an equivalent leftmost and an equivalent rightmost one yielding the same parse tree.

A *parse tree* for $w in L(G)$ is a rooted ordered tree with root labelled $S$, internal nodes labelled in $V$, leaves labelled in $Sigma union { epsilon }$, and the children of each internal node $A$ spelling out the right-hand side of some production $A arrow.r alpha$. The yield (left-to-right concatenation of leaves) equals $w$. A grammar $G$ is *ambiguous* if some $w in L(G)$ has $>= 2$ parse trees; a *language* $L$ is *inherently ambiguous* if every CFG generating $L$ is ambiguous.

=== Canonical Examples

```text
G_1 (balanced parens):  S -> ε | ( S ) S
G_2 (a^n b^n):           S -> ε | a S b
G_3 (palindromes):       S -> ε | a | b | a S a | b S b
G_4 (arithmetic):        E -> E + T | T
                         T -> T * F | F
                         F -> ( E ) | id
G_5 (Dyck_2):            S -> ε | ( S ) S | [ S ] S
```

The *Dyck language* $D_n$ (balanced strings over $n$ kinds of parentheses) is the prototypical CFL — *every* CFL is, up to inverse homomorphism and intersection with a regular set, a Dyck language (Chomsky–Schützenberger 1963):

*Theorem (Chomsky–Schützenberger).* A language $L$ is context-free <==> there exist $n >= 1$, a regular language $R subset.eq { 1, dots, n, overline(1), dots, overline(n) }^*$, and a homomorphism $h$ such that $L = h(D_n inter R)$.

This says CFLs are precisely the homomorphic images of intersections of Dyck languages with regular languages — a statement of the same flavour as Kleene's representation of regular languages but two levels deeper.

== Chomsky Normal Form

A CFG is in *Chomsky Normal Form* (CNF) if every production has one of the forms

$ A arrow.r B C quad (B, C in V), quad A arrow.r a quad (a in Sigma), quad S arrow.r epsilon $

with the last allowed only if $epsilon in L(G)$ and $S$ does not appear on any right-hand side.

*Theorem (Chomsky 1959).* Every CFG $G$ can be effectively transformed into a CNF grammar $G'$ with $L(G') = L(G)$.

*Proof sketch.* Four passes.

(1) *START.* Introduce a fresh $S_0 arrow.r S$ to ensure $S$ does not appear on a right-hand side.

(2) *TERM.* For each production with mixed terminals and nonterminals, replace each terminal $a$ on the right by a star.op nonterminal $T_a$ and add $T_a arrow.r a$.

(3) *BIN.* For each production $A arrow.r B_1 B_2 dots B_k$ with $k >= 3$, introduce $A_1, dots, A_(k-2)$ and replace by $A arrow.r B_1 A_1$, $A_1 arrow.r B_2 A_2$, ..., $A_(k-2) arrow.r B_(k-1) B_k$.

(4) *DEL + UNIT.* Eliminate $epsilon$-productions: for each *nullable* $A$ (with $A =>^* epsilon$), for each production using $A$ on the right add variants where some occurrences of $A$ are deleted; then remove $A arrow.r epsilon$. Eliminate *unit productions* $A arrow.r B$: take the transitive closure of unit relations and replace by the non-unit productions reachable.

Size blow-up is *polynomial*: $|G'| = O(|G|^2)$ in the worst case, dominated by step (4). $square$

== Greibach Normal Form

A CFG is in *Greibach Normal Form* (GNF) if every production has the form

$ A arrow.r a alpha quad (a in Sigma, alpha in V^*) $

GNF is the CFG analogue of right-regular grammars: every derivation step *consumes one terminal*, so a leftmost derivation of $w$ has exactly $|w|$ steps. This makes GNF the natural target for top-down recursive-descent and PDA constructions.

*Theorem (Greibach 1965).* Every $epsilon$-free CFG can be effectively transformed into GNF.

*Proof sketch.* Start in CNF without $epsilon$. Order the nonterminals $A_1, dots, A_n$. Iteratively eliminate left recursion: for $i$ from 1 to $n$, for each production $A_i arrow.r A_j alpha$ with $j < i$, substitute $A_j$'s right-hand sides; eliminate direct left recursion $A_i arrow.r A_i alpha | beta$ by introducing $A_i'$ and rewriting as $A_i arrow.r beta A_i'$, $A_i' arrow.r alpha A_i' | epsilon$ (Paull's algorithm). After processing all $A_i$, productions $A_i arrow.r A_j alpha$ have $j > i$; back-substitute in decreasing order of $j$ to ensure every right-hand side starts with a terminal. Size blow-up is *exponential* in the worst case but polynomial when the input grammar has bounded recursion depth. $square$

== Pushdown Automata

A *pushdown automaton* (PDA) is the canonical recogniser for CFLs — full treatment is in the next chapter (_Pushdown Automata and Beyond_). For the present chapter the key fact is:

*Theorem (CFG–PDA equivalence; Chomsky, Evey, Schützenberger).* A language $L$ is context-free <==> $L = L(M)$ for some nondeterministic pushdown automaton $M$.

The construction *CFG $arrow.r$ PDA*: from a CFG in GNF, build a one-state PDA whose stack holds a *sentential form's tail*. The PDA pops the top nonterminal $A$, reads the next input $a$, and pushes the tail of some production $A arrow.r a alpha$ — guessing $alpha$ nondeterministically. The construction *PDA $arrow.r$ CFG* introduces nonterminals $angle.l p, X, q angle.r$ standing for "the PDA goes from state $p$ to state $q$ while net popping $X$" and grammar productions encode push/pop sequences; the construction is quadratic in PDA size and produces $|Q|^2 |Gamma|$ nonterminals.

== CYK Algorithm

The *Cocke–Younger–Kasami* algorithm (Cocke 1969; Younger 1967; Kasami 1965) decides $w in L(G)$ for $G$ in CNF in time $O(n^3 |G|)$ and space $O(n^2)$, where $n = |w|$.

*Algorithm.* Let $w = a_1 a_2 dots a_n$. Build a table $T[i, j]$ for $1 <= i <= j <= n$:

$ T[i, j] = { A in V | A =>^* a_i a_(i+1) dots a_j } $

*Base.* $T[i, i] = { A | A arrow.r a_i in P }$.

*Recurrence.* $T[i, j] = { A | exists k in {i, dots, j - 1}, exists A arrow.r B C in P. B in T[i, k] "and" C in T[k+1, j] }$.

*Answer.* $w in L(G) <==> S in T[1, n]$.

The outer two indices give $Theta(n^2)$ cells, each filled in $O(n |G|)$ work, yielding $O(n^3 |G|)$ total. Empirically the algorithm runs much faster on grammars where $T[i, j]$ is sparse.

```python
eq.def cyk(grammar, w):
    n = len(w)
    T = [["set"() for _ in range(n+1)] for _ in range(n+1)]
    for i, a in enumerate(w, 1):
        T[i][i] = {A for (A, rhs) in grammar if rhs == (a,)}
    for length in range(2, n+1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            for k in range(i, j):
                for (A, rhs) in grammar:
                    "if len(rhs) == 2 and rhs[0] in T[i][k] "and rhs[1] in T[k+1][j]:
                        T[i][j].add(A)
    return grammar.start in T[1][n]
```

=== Valiant's Reduction to Boolean Matrix Multiplication

*Theorem (Valiant 1975).* CFG recognition is reducible in $O(M(n))$ time to *Boolean matrix multiplication*, where $M(n)$ is the time to multiply two $n times n$ Boolean matrices. Combined with $omega < 2.371$ (Williams–Xu–Xu–Zhou 2024), CFG recognition is in $O(n^omega)$ time.

*Proof sketch.* Define the *CYK matrix* $X$ over the semiring of subsets of $V$ with union and "production-composition" as $+$ and $dot$. The CYK recurrence corresponds to a *transitive closure* on a triangular matrix; Valiant showed this can be reduced to a sequence of $O(log n)$ Boolean matrix multiplications via divide-and-conquer over diagonal blocks. The reduction is one-way: faster CFG recognition does not (yet) imply faster matmul, but several authors (Lee 2002, Abboud–Backurs–V. Williams 2018) have shown that a "combinatorial" sub-cubic CFG parser would imply a sub-cubic combinatorial matmul algorithm — a 50-year open problem. So $O(n^3)$ is in this sense *the* barrier for combinatorial CFG parsing. $square$

== Earley's Algorithm

Earley (1968, 1970) gave a parser that runs in time $O(n^3 |G|^2)$ on arbitrary CFGs (no normal form required), $O(n^2)$ on *unambiguous* grammars, and $O(n)$ on grammars in the "bounded-state" class (which strictly contains all LR($k$) grammars). The algorithm processes the input left-to-right, maintaining an *Earley set* $E_i$ for each input position $i$ — a set of *items* $[A arrow.r alpha . beta, j]$ meaning "we have matched $alpha$ from input position $j$ to $i$ and expect to match $beta$ starting at position $i$".

Three operations populate the sets:

- *Predict.* If $[A arrow.r alpha . B beta, j] in E_i$ and $B arrow.r gamma in P$, add $[B arrow.r . gamma, i]$ to $E_i$.
- *Scan.* If $[A arrow.r alpha . a beta, j] in E_i$ and $a_(i+1) = a$, add $[A arrow.r alpha a . beta, j]$ to $E_(i+1)$.
- *Complete.* If $[B arrow.r gamma ., j] in E_i$ and $[A arrow.r alpha . B beta, k] in E_j$, add $[A arrow.r alpha B . beta, k]$ to $E_i$.

Acceptance: $[S arrow.r gamma ., 0] in E_n$.

*Theorem (Earley complexity).* Earley's algorithm runs in time $O(n^3 |G|^2)$ on general CFGs, $O(n^2 |G|^2)$ on unambiguous CFGs, and $O(n |G|^2)$ on bounded-state grammars.

Earley is the parser of choice when the grammar is given dynamically (e.g., user-defined operators in interactive proof assistants) and when full GLR generality is needed without an offline grammar-compilation phase.

== Pumping Lemma for CFLs (Bar-Hillel–Perles–Shamir 1961)

*Theorem (Bar-Hillel et al.).* If $L$ is context-free, there exists $p >= 1$ such that every $w in L$ with $|w| >= p$ decomposes as $w = u v x y z$ with

(i) $|v y| >= 1$, (ii) $|v x y| <= p$, (iii) $u v^i x y^i z in L$ for all $i >= 0$.

*Proof.* Let $G$ be a CFG in CNF generating $L$ with $|V| = k$ nonterminals. Set $p = 2^(k+1)$. For any $w in L$ with $|w| >= p$, any parse tree has *height* $> k$ (since CNF parse trees are binary and a binary tree of height $h$ has $<= 2^h$ leaves). Along any root-to-leaf path of length $> k$ some nonterminal $A$ repeats. Let the outer occurrence subtree yield $v x y$ and the inner yield $x$. Then $A =>^* v A y$ and $A =>^* x$; the outer derivation can be iterated to give $u v^i x y^i z in L$ for all $i$. Taking $A$ to be the *lowest* repeated nonterminal (in a path of length exactly $k + 1$) bounds $|v x y| <= 2^(k+1) = p$. $square$

=== Ogden's Lemma (1968)

A strengthening: in addition to picking $w$, the adversary may *mark* any $>= p$ positions of $w$; then the decomposition $u v x y z$ exists with $v y$ containing at least one *marked* position and $v x y$ containing at most $p$ marked positions. Ogden's lemma is strictly stronger than the standard pumping lemma — there are languages satisfying the pumping condition that fail Ogden's.

*Application.* $L = { a^i b^j c^k d^l | i = 0 or j = k = l }$ satisfies the standard pumping lemma but is not context-free; Ogden's lemma disproves CFL-ness by marking $b^p c^p d^p$ positions of $a^0 b^p c^p d^p in L$ and showing the marked-balance constraint cannot be preserved.

== Parikh's Theorem (1961, 1966)

For $w in Sigma^*$ with $Sigma = { a_1, dots, a_k }$, the *Parikh vector* is $Psi(w) = (|w|_(a_1), dots, |w|_(a_k)) in NN^k$. The Parikh image of a language is $Psi(L) = { Psi(w) | w in L } subset.eq NN^k$.

A subset $S subset.eq NN^k$ is *linear* if $S = { v_0 + n_1 v_1 + dots + n_m v_m | n_i in NN }$ for fixed vectors $v_0, dots, v_m in NN^k$. *Semilinear* means finite union of linear sets — equivalently, definable in *Presburger arithmetic* $"FO"(NN, +, <)$.

*Theorem (Parikh 1961).* For every context-free language $L$, $Psi(L)$ is semilinear. Hence $Psi(L_"CFL") = Psi(L_"regular")$: every CFL has the *same commutative image* as some regular language.

*Proof sketch.* Take a CFG in CNF generating $L$. The *parse-tree pumping* argument decomposes parse trees into a *seed* (bounded height) plus iterable *pump fragments* (loops $A =>^* alpha A beta$). Each seed contributes a base vector $v_0$ and each pump contributes a period vector $v_i$. The Parikh image is the union over finite choices of seed and pumps — a finite union of linear sets. $square$

*Corollary.* Every CFL over a *unary* alphabet $Sigma = { a }$ is regular: a semilinear subset of $NN$ is just a finite union of arithmetic progressions, definable by a DFA counting modulo the period.

== Closure Properties

Context-free languages are closed under: union, concatenation, Kleene star, reversal, homomorphism, inverse homomorphism, intersection with a regular language, substitution (by a CFL-valued substitution).

*Non-closures.* CFLs are *not* closed under intersection or complement.

*Counterexample (intersection).* $L_1 = { a^n b^n c^m | n, m >= 0 }$ and $L_2 = { a^n b^m c^m | n, m >= 0 }$ are each CFL. Their intersection is $L_1 inter L_2 = { a^n b^n c^n | n >= 0 }$, which is *not* context-free (standard pumping argument).

*Counterexample (complement).* If CFLs were closed under complement they would, with union-closure, be closed under intersection — contradiction.

*Theorem (intersection with regular).* If $L$ is context-free and $R$ is regular then $L inter R$ is context-free. *Proof:* product-construct the PDA for $L$ with the DFA for $R$; the new states are $Q_L times Q_R$, stack and transitions unchanged on the PDA side.

This asymmetric closure underpins the *Chomsky–Schützenberger* representation theorem: every CFL is recoverable as a homomorphic image of a regular-intersected Dyck language.

== Inherent Ambiguity

A CFL $L$ is *inherently ambiguous* if every CFG generating $L$ is ambiguous (i.e., produces $>= 2$ parse trees for some string). Inherent ambiguity is a property of the *language*, not of any particular grammar.

*Theorem (Ginsburg 1966).* The language $L = { a^i b^j c^k | i = j or j = k }$ is inherently ambiguous.

*Proof sketch.* The language is the union $L_1 union L_2$ with $L_1 = { a^n b^n c^m }$ and $L_2 = { a^m b^n c^n }$. Strings of the form $a^n b^n c^n$ lie in both. Any CFG for $L$ must, by an Ogden-lemma-style pumping argument applied to the parse trees, contain two distinct derivation strategies — one preserving $i = j$ and one preserving $j = k$ — and at $a^n b^n c^n$ both apply, giving two parse trees. $square$

Inherent ambiguity is *undecidable* in general (Cantor 1962; Floyd 1962) but is rare in practical languages; most syntactic ambiguities (dangling-else, expression-precedence) are *grammatical* and can be resolved by rewriting.

== Undecidability for CFGs

*Theorem (Bar-Hillel–Perles–Shamir 1961; Cantor 1962; Floyd 1962).* The following problems for CFGs are *undecidable*:

(a) *Equivalence:* given $G_1, G_2$, is $L(G_1) = L(G_2)$?

(b) *Inclusion:* given $G_1, G_2$, is $L(G_1) subset.eq L(G_2)$?

(c) *Universality:* given $G$, is $L(G) = Sigma^*$?

(d) *Ambiguity:* given $G$, is $G$ ambiguous?

(e) *Inherent ambiguity:* given $G$ generating $L$, is $L$ inherently ambiguous?

(f) *Regularity:* given $G$, is $L(G)$ regular?

(g) *Intersection emptiness:* given $G_1, G_2$, is $L(G_1) inter L(G_2) = emptyset$?

*Proof sketch.* Reduction from the *Post Correspondence Problem* (PCP). Given a PCP instance with dominoes $(u_i, v_i)$, build CFGs $G_U$ generating ${ u_(i_1) dots u_(i_k) \# i_k dots i_1 }$ and $G_V$ similarly for the $v_i$. Then $L(G_U) inter L(G_V) eq."not" emptyset$ <==> the PCP has a solution. This gives (g) directly; the others follow by simple language manipulations (e.g., universality: complement $L(G_U inter G_V)$ relative to the appropriate regular envelope; equivalence: compare with a known universal language).

Floyd (1962) gave the original ambiguity-undecidability proof: from a PCP instance build a grammar $G$ generating ${ w_U \# w_V | u"-sequence" = v"-sequence" }$ in two derivational ways, ambiguous when the PCP has a solution. $square$

*Decidable.* Membership ($O(n^3)$ via CYK), emptiness (mark reachable and productive nonterminals), and finiteness (cycle detection in the reachable/productive subgrammar).

== Deterministic Context-Free Languages

A *deterministic pushdown automaton* (DPDA) has a partial transition function $delta : Q times (Sigma union { epsilon }) times Gamma arrow.r Q times Gamma^*$ with the determinism condition: if $delta(q, epsilon, Z)$ is defined then $delta(q, a, Z)$ is undefined for all $a in Sigma$. A language is a *deterministic CFL* (DCFL) if it is recognised by a DPDA accepting by *final state*.

*Theorem.* DCFL is *strictly* between regular and CFL: $"Reg" subset.eq."not" "DCFL" subset.eq."not" "CFL"$.

Separation from CFL: even-length palindromes $L = { w w^R | w in { a, b }^* }$ are context-free but not DCFL (no PDA can deterministically guess the midpoint).

*Closure properties.* DCFL is closed under *complement* (the celebrated proof requires handling of $epsilon$-loops and dead configurations — see Hopcroft–Ullman §10.6), inverse homomorphism, and intersection with regular sets, but *not* under union, intersection, concatenation, Kleene star, or homomorphism. The closure under complement makes DCFL the natural setting for unambiguous parsing.

=== LR($k$) and LL($k$)

A grammar is *LR($k$)* if a deterministic bottom-up parser can decide each reduction by looking at the top of the stack and $k$ symbols of lookahead. Knuth (1965) proved:

*Theorem (Knuth 1965).* A language is LR($k$) for some $k$ iff it is LR(1) iff it is a deterministic context-free language.

So LR(1) captures *exactly* the DCFLs. Practically, *LALR(1)* (a coarser equivalence relation on LR(1) states yielding smaller parse tables — Korenjak 1969, DeRemer 1969) is the basis of yacc/bison; *LR(1)* tables can be hundreds of times larger.

The dual class *LL($k$)* (top-down deterministic parsers with $k$-symbol lookahead) forms a *strict* hierarchy: $"LL"(k) subset.eq."not" "LL"(k+1)$ for every $k$, and $union.big_k "LL"(k) subset.eq."not" "LR"(1) = "DCFL"$. The language $L = { a^n b^n | n >= 0 } union { a^n b^(2n) | n >= 0 }$ is LR(1) but not LL($k$) for any $k$.

=== Sénizergues's Theorem

*Theorem (Sénizergues 1997; Stirling 2001).* Equivalence of *deterministic* pushdown automata is *decidable*.

This astonishing result — a Gödel Prize in 2002 — stands in sharp contrast to the undecidability of CFG equivalence. The proof employs *bisimulation up to* on infinite-state systems and is one of the most technically demanding decidability proofs in formal-language theory. *Stirling* gave a simplified proof via game-theoretic bisimulation arguments.

*Theorem (Jančar 2012).* DPDA equivalence is *primitive-recursive*: the running time is bounded by a tower of exponentials of height polynomial in the input size. The exact complexity remains open; no known elementary upper bound.

For *DCFLs* this gives a (very impractical) equivalence-checking algorithm; in practice equivalence is decided by parser-generator constructions that build canonical LR(1) automata and check isomorphism.

== Greibach's Hardest Context-Free Language

*Theorem (Greibach 1973).* There exists a CFL $L_0$ — the *hardest CFL* — such that every CFL $L$ is reducible to $L_0$ by a *length-preserving homomorphism*. Hence any algorithm deciding membership in $L_0$ in time $T(n)$ yields an $O(T(n))$ algorithm for every CFL.

*Construction sketch.* $L_0$ is a Dyck-like language over a specific 12-letter alphabet, encoding nondeterministic PDA computations: each letter represents a transition tag, and well-bracketed words correspond exactly to accepting PDA runs. Reduction: every CFL is recognised by a PDA whose transitions are encoded as homomorphic images of $L_0$'s letters. $square$

The hardest-CFL theorem is the source of the *information-theoretic* claim that no truly sub-cubic combinatorial algorithm for CFL recognition is known: any speed-up for $L_0$ would propagate to *every* CFL.

== Worked-Out Constructions

=== Building a PDA for $L = { a^n b^n | n >= 0 }$

```text
PDA M = ({q_0, q_1, q_2}, {a,b}, {Z, A}, δ, q_0, Z, {q_2})

δ(q_0, ε, Z) = {(q_2, Z)}             -- accept ε
δ(q_0, a, Z) = {(q_1, AZ)}            -- start counting a's
δ(q_1, a, A) = {(q_1, AA)}            -- push more A's
δ(q_1, b, A) = {(q_1', ε)}            -- begin popping
δ(q_1', b, A) = {(q_1', ε)}           -- pop matching A's
δ(q_1', ε, Z) = {(q_2, Z)}            -- bottom of stack ⇒ accept
```

The PDA stacks one $A$ per input $a$, then pops one $A$ per input $b$; acceptance requires the stack to return to its initial state precisely when the input is exhausted.

=== CYK Trace for $a a b b$ under $G_2$

CNF conversion of $G_2: S arrow.r a S b | epsilon$ yields (assuming $epsilon$-free input):

```text
S  -> A B  | A C
C  -> S B
A  -> a
B  -> b
```

Input $w = a a b b$ ($n = 4$):

```text
       [1,1]={A}  [1,2]=∅    [1,3]=∅    [1,4]={S}
                  [2,2]={A}  [2,3]={S}  [2,4]=∅
                             [3,3]={B}  [3,4]=∅
                                        [4,4]={B}
```

Cell $[1, 4] = { S }$: derived from split $k = 1$: $A in [1, 1]$, but $[2, 4] = emptyset$ — no contribution. Split $k = 2$: $[1, 2] = emptyset$. Split $k = 3$: $[1, 3] = emptyset$. So $[1, 4]$ is filled only via $k = 1$ if we also include the rule $S arrow.r A C$ — and indeed $C in [2, 4]$? Let's recompute: $[2, 4]$ should be $S B$ split: $k = 3$, $S in [2, 3]$, $B in [4, 4]$, so $C in [2, 4]$. Then $[1, 4]$: $k = 1$, $A in [1, 1]$, $C in [2, 4]$, so $S in [1, 4]$. Membership confirmed.

=== Earley Trace for an Ambiguous Grammar

For $G: E arrow.r E + E | "id"$ on input $"id" + "id" + "id"$:

```text
E_0 = { [E -> .E+E, 0], [E -> .id, 0] }
E_1 = { [E -> id., 0], [E -> E.+E, 0] }                 -- scan "id"
E_2 = { [E -> E+.E, 0], [E -> .E+E, 2], [E -> .id, 2] } -- scan "+"
E_3 = { [E -> id., 2], [E -> E+E., 0], [E -> E.+E, 2],
        [E -> E.+E, 0] }                                 -- scan "id" + 2 completes
... (continues; final E_5 contains two completed items for E spanning 0..5)
```

The two completed items in $E_5$ witness the two parse trees (left- and right-associative). Earley *recognises* ambiguity; producing a *parse forest* requires Tomita's GLR extension or the Scott–Johnstone SPPF representation.

== Pragmatic Parsing Hierarchy

The CFLs sit in a tower of decreasingly expressive but increasingly *efficiently-to-parse* subclasses used in real compilers:

```text
                  CFG  (Earley O(n^3))
                   |
                  LR(k) = DCFL  (linear-time deterministic; Knuth 1965)
                   |
                  LALR(1)  (yacc/bison)
                   |
                  SLR(1)
                   |
                  LR(0)
                   |
                  LL(k)   (top-down, recursive descent)
                   |
                  LL(1)   (ANTLR pre-v4, most hand-written parsers)
```

- *LL(1)* grammars are the easiest to write parsers for by hand (recursive descent: one mutually-recursive function per nonterminal), but cannot handle left recursion natively.
- *LALR(1)* is the practical default for generated parsers (C, Pascal, OCaml's Menhir).
- *GLR* (Tomita 1985) handles any CFG by running multiple LR parsers in parallel via a graph-structured stack; worst-case $O(n^(k+1))$ where $k$ depends on the grammar's amount of nondeterminism, but linear on locally deterministic grammars.
- *Packrat parsing* (Ford 2002) for *parsing expression grammars* (PEGs) runs in linear time at the cost of memoising all subparses — a different (deterministic, *not* a CFG) formalism.

== Where the Theory Breaks

The undecidability results above are the source of essentially every practical headache in parser-generator design:

- *Conflict detection.* `bison`'s shift-reduce and reduce-reduce conflicts are *syntactic* approximations of LR(1)-ness; they may flag a non-conflict (grammar is genuinely LR(1) but the LALR(1) approximation is too coarse) and they certainly cannot prove general non-ambiguity.

- *Equivalence-preserving refactoring* of grammars is unsupported by any tool, because grammar equivalence is undecidable. Refactorings are checked empirically by parsing test corpora.

- *Generated-parser size.* LR(1) tables for full Java syntax run to megabytes; LALR(1) coarsening reduces this by factors of 10–100 at the cost of occasional spurious conflicts.

The next chapter takes up *pushdown automata* in their full operational detail, the *visibly pushdown languages* that recover regular-like closure properties for nested-word data, and the higher classes (indexed grammars, tree-adjoining grammars, context-sensitive, type-0) above the CFLs in the Chomsky hierarchy.

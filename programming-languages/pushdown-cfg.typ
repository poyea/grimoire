= Pushdown Automata and Context-Free Grammars

Context-free languages (CFLs) sit one level above regular languages in the Chomsky
hierarchy. Every programming language syntax is context-free or very nearly so; the
exceptions (indentation-sensitive Python, C++ template instantiation depth) are handled
by ad-hoc means layered on top of a CF core. Understanding PDAs and CFGs is prerequisite
to understanding every parser in chapter 5.

_See also: programming-languages/parsing.typ for LL(k), LR(k), Earley, and GLL parsers
that implement the PDA model in practice._

== Pushdown Automata

A *pushdown automaton* (PDA) is a 6-tuple $(Q, Sigma, Gamma, delta, q_0, F)$:

- $Q$ -- finite set of states
- $Sigma$ -- input alphabet
- $Gamma$ -- stack alphabet (includes bottom-of-stack marker $bot$)
- $delta : Q times (Sigma union {epsilon}) times Gamma -> 2^(Q times Gamma^*)$ -- transition relation
- $q_0$ -- start state
- $F$ -- accepting states

At each step the PDA reads (or skips, for $epsilon$-transitions) an input symbol and pops
the top stack symbol, then pushes a string $gamma in Gamma^*$ and moves to a new state.
The stack provides unbounded memory that DFAs lack, enabling matching of arbitrarily nested
structures.

=== Acceptance Modes

There are two equivalent acceptance criteria:

- *Acceptance by final state:* the PDA halts in a state $q in F$ after consuming all input.
- *Acceptance by empty stack:* the PDA halts with an empty stack after consuming all input.

Every language accepted by final state has a PDA accepting it by empty stack and vice versa.
The transformations between the two modes add at most 2 states and a fresh bottom-of-stack
symbol. The final-state mode is more convenient for intersection with regular languages
(see closure properties below); the empty-stack mode simplifies proofs about grammars.

=== DPDA vs NPDA

A *deterministic PDA* (DPDA) requires that for each $(q, a, A)$ there is at most one
applicable rule (including $epsilon$-input rules, with the constraint that if a rule on
actual input $a$ applies, no $epsilon$-input rule applies in the same configuration).

*DCFL* $subset.neq$ *CFL:* The language of even-length palindromes over ${a,b}$ is CFL
but not DCFL: a DPDA cannot guess the midpoint deterministically. The language
${ a^n b^n }$ is DCFL. DCFLs correspond exactly to LR(1)-parseable languages -- the
practical justification for LR parsing being the dominant technique.

== Context-Free Grammars

A *context-free grammar* (CFG) is a 4-tuple $(V, Sigma, R, S)$:

- $V$ -- finite set of *nonterminals* (variables)
- $Sigma$ -- finite set of *terminals* (disjoint from $V$)
- $R subset V times (V union Sigma)^*$ -- finite set of *productions* (rules), written $A -> alpha$
- $S in V$ -- start symbol

*Derivation:* $alpha A beta =>_G alpha gamma beta$ if $A -> gamma in R$. We write $=>^*_G$
for the reflexive-transitive closure. The language of $G$ is
$L(G) = { w in Sigma^* | S =>^*_G w }$.

=== Left and Right Derivations

- *Leftmost derivation:* always expand the leftmost nonterminal. Corresponds to top-down
  (LL) parsing.
- *Rightmost derivation:* always expand the rightmost nonterminal. Corresponds to bottom-up
  (LR) parsing; the *reverse* of a rightmost derivation is the sequence of reductions a
  shift-reduce parser performs.

*Parse tree:* A rooted, ordered tree where interior nodes are nonterminals, leaves are
terminals or $epsilon$, and children of node $A$ correspond to a production $A -> alpha$.
Every parse tree determines a unique leftmost and a unique rightmost derivation.

=== Ambiguity and the Dangling-Else Problem

A grammar is *ambiguous* if some string has two or more distinct parse trees (equivalently,
two distinct leftmost derivations).

The canonical example is the `if-then-else` grammar:

```ebnf
Stmt  ::= "if" Expr "then" Stmt
        | "if" Expr "then" Stmt "else" Stmt
        | Other
```

The string `if e1 then if e2 then s1 else s2` has two parse trees:

- *Tree 1 (inner else):* `else s2` belongs to the inner `if e2 then s1`.
- *Tree 2 (outer else):* `else s2` belongs to the outer `if e1 then ...`.

C and Java resolve this by the *closest unmatched if* rule (Tree 1), enforced by
*grammar rewriting* rather than semantic disambiguation:

```ebnf
Stmt          ::= MatchedStmt | UnmatchedStmt
MatchedStmt   ::= "if" Expr "then" MatchedStmt "else" MatchedStmt | Other
UnmatchedStmt ::= "if" Expr "then" Stmt
                | "if" Expr "then" MatchedStmt "else" UnmatchedStmt
```

This grammar is unambiguous and correctly associates the `else` with the nearest `if`.
The cost is grammar complexity; practical parser generators instead accept the ambiguous
grammar and add a *conflict resolution rule* (shift over reduce for `else`).

*Inherently ambiguous CFLs:* Some CFLs have no unambiguous grammar. The language
${ a^i b^j c^k | i = j "or" j = k }$ is inherently ambiguous.

== Chomsky Normal Form

A CFG is in *Chomsky Normal Form* (CNF) if every production is either $A -> B C$ (two
nonterminals) or $A -> a$ (one terminal), plus optionally $S -> epsilon$. Every CFG
can be converted to CNF in polynomial time by:

1. Eliminating $epsilon$-productions (except $S -> epsilon$).
2. Eliminating unit productions $A -> B$.
3. Binarizing rules with more than two symbols on the right.
4. Isolating terminals: replace $a$ in a mixed rule with a fresh nonterminal $T_a -> a$.

CNF is the precondition for the CYK algorithm and simplifies many proofs.

=== CYK Algorithm

*Cocke-Younger-Kasami* is a dynamic-programming membership test for CNF grammars.

For input $w = a_1 dots a_n$ and grammar $G$ in CNF, define
$T[i][j] = { A in V | A =>^*_G a_i dots a_j }$.

*Recurrence:* For $l = 1$: $A in T[i][i]$ iff $A -> a_i in R$.
For $l > 1$: $A in T[i][i+l-1]$ iff there exist $B in T[i][k]$, $C in T[k+1][i+l-1]$,
and a rule $A -> B C$ for some $k$ with $i <= k < i+l-1$.

*Complexity:* $O(n^3 |G|)$ time, $O(n^2 |V|)$ space. The $|G|$ factor is
typically treated as constant. CYK is the asymptotically optimal general CF recognizer;
in practice Earley (also $O(n^3)$ worst-case, $O(n^2)$ for unambiguous, $O(n)$ for
$"LR"(k)$) is preferred because it handles non-CNF grammars and reports parse trees.

== Pumping Lemma for Context-Free Languages

*Theorem (Bar-Hillel et al. 1961):* If $L$ is CFL, there exists $p >= 1$ such that every
$w in L$ with $|w| >= p$ can be written $w = u v x y z$ with:

1. $|v y| >= 1$
2. $|v x y| <= p$
3. For all $i >= 0$, $u v^i x y^i z in L$

*Proof sketch:* Let $G$ be a CNF grammar for $L$ with $|V|$ nonterminals. Set $p = 2^(|V|+1)$.
For $|w| >= p$, the parse tree has depth $> |V|$, so some nonterminal $A$ repeats on a
root-to-leaf path. The subtree rooted at the upper $A$ yields $v x y$; the inner $A$
yields $x$. Replacing the outer subtree with the inner (pump down, $i=0$) or nesting the
outer copy into itself repeatedly (pump up, $i > 1$) yields strings that must remain in $L$
if $L$ were CF.

*Application -- $a^n b^n c^n$ is not CF:* Suppose $L = { a^n b^n c^n }$ were CF with
pumping length $p$. Choose $w = a^p b^p c^p$. The constraint $|v x y| <= p$ means $v$ and
$y$ together span at most two of the three symbol types. Pumping up ($i = 2$) either
over-counts one symbol class or leaves another unchanged, destroying the equal-count
invariant. In all cases $u v^2 x y^2 z in.not L$. Contradiction.

== Closure Properties of Context-Free Languages

#table(
  columns: (auto, auto, auto),
  [*Operation*], [*CFLs closed?*], [*Proof technique*],
  [Union $L_1 union L_2$],           [Yes],  [New start: $S -> S_1 | S_2$],
  [Concatenation $L_1 L_2$],         [Yes],  [New start: $S -> S_1 S_2$],
  [Kleene star $L^*$],               [Yes],  [New start: $S -> S S_1 | epsilon$],
  [Intersection $L_1 inter L_2$],     [No],   [${ a^n b^n } inter { b^n c^n }$ counterexample],
  [Complement $overline(L)$],        [No],   [De Morgan + intersection failure],
  [Intersection with regular],       [Yes],  [PDA product construction with DFA],
  [Homomorphism],                     [Yes],  [Apply to each terminal in grammar],
  [Inverse homomorphism],             [Yes],  [PDA simulation],
)

The *CFL intersect regular = CFL* closure is the theoretical foundation of the
*lexer-then-parser* architecture: the lexer (a DFA) filters the raw character stream into
a token stream, and the parser (a PDA) processes the token stream. Because the token stream
is a regular image of the character stream, the composition is sound. This also means that
adding keyword reservation to an otherwise CF grammar does not leave the CF class.

== Decidability for Context-Free Languages

#table(
  columns: (auto, auto),
  [*Problem*], [*Status*],
  [Membership: $w in L(G)$?],           [Decidable -- $O(n^3)$ via CYK],
  [Emptiness: $L(G) = emptyset$?],      [Decidable -- reachability from $S$],
  [Finiteness: $|L(G)| < infinity$?],   [Decidable -- cycle detection in grammar],
  [Ambiguity: is $G$ ambiguous?],       [Undecidable],
  [Equivalence: $L(G_1) = L(G_2)$?],   [Undecidable],
  [Universality: $L(G) = Sigma^*$?],    [Undecidable],
)

Equivalence undecidability has a direct practical consequence: no algorithm can determine
whether two grammars generate the same language, so grammar refactoring must be validated
by testing or restricted to structure-preserving transformations.

== LL and LR Parsing: Preview

The PDA model underlies two engineering families:

- *LL(k) parsers* simulate a *leftmost* derivation top-down, using $k$ tokens of lookahead
  to choose which production to expand. They are recursive-descent at heart; $k=1$ is
  sufficient for most practical grammars after left-recursion elimination and
  left-factoring.
- *LR(k) parsers* simulate a *rightmost* derivation bottom-up by shifting input and reducing
  completed right-hand sides. LR(1) recognizes all DCFL languages; LALR(1) is the
  engineering compromise used by Yacc and Bison.

The full treatment -- FIRST/FOLLOW sets, item sets, conflict resolution, error recovery --
appears in the companion chapter.

_See also: programming-languages/parsing.typ for the complete construction of LL(1) parse
tables, LR(1) and LALR(1) automata, and Earley parsing for arbitrary CFGs._

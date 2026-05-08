= Regular Languages and Finite Automata

*The bottom of the Chomsky hierarchy is not a limitation — it is a feature.* Every problem
that can be solved with a finite automaton is solved in $O(n)$ time and $O(1)$ space (the
state). Lexers, log scanners, packet classifiers, and configuration validators all live here.

_See also: coding/string-algorithms.typ for KMP, Aho-Corasick, and other pattern-matching
algorithms that exploit the structure of regular languages at the implementation level._

== Deterministic Finite Automata

A *deterministic finite automaton* (DFA) is a 5-tuple $(Q, Sigma, delta, q_0, F)$ where:

- $Q$ — finite, non-empty set of states
- $Sigma$ — finite input alphabet
- $delta : Q times Sigma -> Q$ — total transition function
- $q_0 in Q$ — start state
- $F subset.eq Q$ — set of accepting (final) states

The DFA processes a string $w = a_1 a_2 dots a_n$ by starting in $q_0$ and applying
$delta$ one symbol at a time. It *accepts* $w$ if the state after consuming $a_n$ is in $F$.
The *extended transition function* $hat(delta) : Q times Sigma^* -> Q$ is defined
inductively: $hat(delta)(q, epsilon) = q$ and $hat(delta)(q, x a) = delta(hat(delta)(q, x), a)$.

The language *recognized* by a DFA $M$ is $L(M) = { w in Sigma^* | hat(delta)(q_0, w) in F }$.

=== Nondeterministic Finite Automata

An *NFA* relaxes $delta$ to $delta : Q times (Sigma union {epsilon}) -> 2^Q$: each step may
choose among zero or more successor states, and $epsilon$-transitions consume no input.
An NFA accepts $w$ if *some* computation path ends in an accepting state.

*Equivalence:* Every NFA with $n$ states has an equivalent DFA with at most $2^n$ states
(*subset construction*, Rabin-Scott 1959). The construction tracks the set of NFA states
reachable on each input prefix.

=== Subset Construction (Worked Example)

Consider the NFA over $Sigma = {a, b}$ that accepts strings ending in $a b$:

- States: $q_0$ (start), $q_1$ (saw $a$), $q_2$ (saw $a b$, accept)
- $delta(q_0, a) = {q_0, q_1}$, $delta(q_0, b) = {q_0}$
- $delta(q_1, b) = {q_2}$, all other $delta(q_1, -)$ and $delta(q_2, -)$ = $emptyset$

Subset construction DFA states (reachable subsets):

#table(
  columns: (auto, auto, auto),
  [*DFA State*], [*On a*], [*On b*],
  [$S_0 = {q_0}$],          [$S_1 = {q_0, q_1}$],  [$S_0$],
  [$S_1 = {q_0, q_1}$],     [$S_1$],                [$S_2 = {q_0, q_2}$],
  [$S_2 = {q_0, q_2}$],     [$S_1$],                [$S_0$],
)

$S_2$ is the only accepting DFA state (contains $q_2$). The resulting DFA has 3 states
rather than the worst-case $2^3 = 8$.

=== Hopcroft Minimization

Given a DFA with $n$ states, *Hopcroft's algorithm* (1971) produces the unique minimal DFA
equivalent to it in $O(n log n)$ time. The key idea is *partition refinement*:

1. Initialize partition $Pi = { F, Q without F }$ (accepting vs non-accepting).
2. Maintain a worklist $W$ of splitter sets.
3. For each splitter $A in W$ and each symbol $a in Sigma$: find $X = delta^{-1}(A, a)$ (states
   that transition into $A$ on $a$). For every block $Y in Pi$ split by $X$ into $Y inter X$
   and $Y without X$, replace $Y$ with the two halves and update $W$ accordingly.
4. Repeat until $W$ is empty. Each block in the final $Pi$ is one minimized state.

The $O(n log n)$ bound comes from charging each state at most $O(log n)$ times to the
worklist; the classic $O(n^2)$ algorithms (Moore, Brzozowski) lack this accounting.

== Regular Expressions

A *regular expression* over $Sigma$ is defined inductively:

- *Basis:* $emptyset$, $epsilon$, and every $a in Sigma$ are regular expressions.
- *Induction:* If $R$ and $S$ are regular expressions, so are $(R | S)$ (union), $(R S)$
  (concatenation), and $(R^*)$ (Kleene star).

*Kleene's theorem:* A language is regular if and only if it is described by a regular
expression. The two directions are:

- *RE $->$ NFA (Thompson construction, 1968):* Each operator introduces at most 2 new states
  and 4 new $epsilon$-transitions. An RE with $m$ operators yields an NFA with $O(m)$ states.
- *DFA $->$ RE (state elimination):* Iteratively remove states, labeling remaining transitions
  with REs that capture the paths through the removed state.

=== Thompson Construction

Thompson's construction is compositional. Base cases: a single symbol $a$ is an NFA with
two states and one transition; $epsilon$ is two states connected by an $epsilon$-arc. For
compound expressions:

- *Union* $R | S$: add a new start with $epsilon$-transitions to start($R$) and start($S$);
  add a new accept reached by $epsilon$ from accept($R$) and accept($S$).
- *Concatenation* $R S$: connect accept($R$) to start($S$) by $epsilon$.
- *Star* $R^*$: add new start and accept; $epsilon$ from new-start to start($R$) and to
  new-accept; $epsilon$ from accept($R$) to start($R$) (loop) and to new-accept.

The resulting NFA has at most $2m$ states for an RE of length $m$ and is fed directly into
subset construction to obtain a DFA.

== The Pumping Lemma

*Theorem (Pumping Lemma for Regular Languages):* If $L$ is regular, then there exists a
constant $p >= 1$ (the *pumping length*, equal to the number of DFA states) such that for
every $w in L$ with $|w| >= p$, $w$ can be written as $w = x y z$ with:

1. $|x y| <= p$
2. $|y| >= 1$
3. For all $i >= 0$, $x y^i z in L$

*Proof sketch:* Let $M$ be a DFA with $p$ states accepting $L$. For $|w| >= p$, the
sequence of states $q_0, q_1, dots, q_{|w|}$ has $|w|+1 > p$ states, so by the pigeonhole
principle some state $q_j = q_k$ with $j < k <= p$. Set $x = w[0..j)$, $y = w[j..k)$,
$z = w[k..|w|)$. Since $q_j = q_k$, pumping $y$ any number of times remains in $L$.

*Application — $a^n b^n$ is not regular:* Suppose it were, with pumping length $p$.
Choose $w = a^p b^p$. Conditions 1-2 force $y = a^k$ for some $k >= 1$ (since $|x y| <= p$
keeps us in the $a$-prefix). Then $x y^2 z = a^{p+k} b^p in.not L$ since $p+k > p$.
Contradiction. Therefore ${ a^n b^n | n >= 0 }$ is not regular.

== Closure Properties

Regular languages are closed under:

- *Union:* Product construction or NFA union.
- *Intersection:* Product construction (DFA pair runs in lock-step).
- *Complement:* Swap $F$ and $Q without F$ on a *complete* DFA.
- *Concatenation:* NFA concatenation (Thompson).
- *Kleene star:* NFA star (Thompson).
- *Difference* $L_1 without L_2 = L_1 inter overline(L_2)$: follows from intersection and complement.
- *Reversal, homomorphism, inverse homomorphism:* all preserve regularity.

These closure properties are constructive and lead directly to DFA-based set operations used
in lexer generators.

== C++ DFA: Block Comment Recognizer

The following DFA recognizes C-style block comments `/* ... */` in a stream of characters.
It has $O(1)$ state, processes each character once ($O(n)$ total), and uses no heap.

```cpp
#include <string>
#include <stdexcept>

enum class State {
    kNormal,        // Outside any comment
    kSlash,         // Saw '/', might be start of comment
    kInComment,     // Inside /* ... */
    kStar,          // Inside comment, saw '*', might be end
    kDone,          // Accepted: entire input is one block comment
    kError          // Rejected
};

// Returns true iff `input` is exactly one well-formed block comment.
bool process_block_comment(const std::string& input) {
    State state = State::kNormal;

    for (char c : input) {
        switch (state) {
            case State::kNormal:
                state = (c == '/') ? State::kSlash : State::kError;
                break;
            case State::kSlash:
                state = (c == '*') ? State::kInComment : State::kError;
                break;
            case State::kInComment:
                if (c == '*') state = State::kStar;
                // else stay in kInComment
                break;
            case State::kStar:
                if (c == '/')      state = State::kDone;
                else if (c == '*') state = State::kStar;   // e.g. /** ... */
                else               state = State::kInComment;
                break;
            case State::kDone:
                state = State::kError;  // Trailing characters not allowed
                break;
            case State::kError:
                break;
        }
    }
    return state == State::kDone;
}
```

*Complexity:* $O(n)$ time, $O(1)$ space. The `enum class` prevents accidental integer
comparisons. No allocation; the state fits in a register. A real lexer would embed this
logic as the hot path of a hand-written or table-driven DFA generated from the token grammar.

== The Chomsky Hierarchy

Noam Chomsky (1956) stratified formal grammars into four nested classes. Each class is
recognized by a corresponding machine model and captures exactly those languages whose
membership problem has a particular complexity profile.

#table(
  columns: (auto, auto, auto, auto),
  [*Type*], [*Grammar*], [*Machine*], [*Example Language*],
  [Type 3], [Regular],               [DFA / NFA],              [$a^* b^*$],
  [Type 2], [Context-Free],          [Pushdown Automaton],      [${ a^n b^n }$],
  [Type 1], [Context-Sensitive],     [Linear-Bounded Automaton],[${ a^n b^n c^n }$],
  [Type 0], [Unrestricted],          [Turing Machine],          [Halting problem $L_"HALT"$],
)

Each type is a proper subset of the type above it: $"Regular" subset "CF" subset "CS"
subset "RE"$. The important engineering boundary is between Type 3 and Type 2: lexers live
in Type 3; parsers live at Type 2. Type 1 grammars (context-sensitive) require a
linear-bounded automaton and membership is PSPACE-complete — unusable for syntax.

_See also: programming-languages/pushdown-cfg.typ for the full treatment of context-free
grammars, pushdown automata, and why ${ a^n b^n c^n }$ is not context-free._

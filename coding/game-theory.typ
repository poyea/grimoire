#import "../template.typ": xref

= Game Theory <game-theory>

*Combinatorial game theory* studies two-player, perfect-information, zero-sum games played under deterministic rules. The central question is always the same: given a position, does the player about to move win or lose under optimal play?

*Normal play convention:* The player who makes the last move *wins*. A player who cannot move *loses*.

*Two position classes:*
- *P-position (Previous player wins):* The player who just moved wins; the player to move now loses under optimal play.
- *N-position (Next player wins):* The player to move has a winning strategy.

*Recursive characterisation:*
- A position with no moves is a P-position.
- A position is an N-position if at least one move leads to a P-position.
- A position is a P-position if every move leads to an N-position.

*See also:* #xref("coding", "dynamic-programming", label: "Dynamic Programming") (memoised game-state search), #xref("coding", "math-number-theory", label: "Math & Number Theory") (XOR nim-sum), #xref("coding", "bit-manipulation", label: "Bit Manipulation") (fast XOR computation)

== Nim

*Classic Nim:* There are $k$ piles of stones. Two players alternate; on each turn a player removes any positive number of stones from exactly one pile. The player who takes the last stone wins (normal play).

*Nim-sum:* $s = n_1 xor n_2 xor dots.c xor n_k$ (bitwise XOR of all pile sizes).

*Bouton's theorem (1901):*
- A position is a *P-position* if and only if the nim-sum is *0*.
- A position is an *N-position* if and only if the nim-sum is *non-zero*.

*Proof sketch:*
1. The all-zero position (no stones remain) has nim-sum 0 and is a P-position (no moves).
2. From any position with nim-sum $s != 0$, there exists a move that reduces nim-sum to 0 (reduce any pile whose highest set bit matches the highest bit of $s$).
3. From any position with nim-sum $0$, every single-pile removal changes that pile, destroying the all-zero XOR balance -- every move leads to a non-zero nim-sum (N-position).

*Winning move:* If $s = n_1 xor dots.c xor n_k != 0$, find a pile $n_i$ such that $n_i xor s < n_i$ (i.e., $n_i$'s highest bit overlaps $s$'s highest bit). Set that pile to $n_i xor s$.

```cpp
#include <bits/stdc++.h>
using namespace std;

// Returns true if the current player wins under optimal play.
// piles: vector of pile sizes.
bool nim_wins(const vector<int>& piles) {
    int nim_sum = 0;
    for (int p : piles) nim_sum ^= p;
    return nim_sum != 0;
}

// Returns the optimal move as {pile_index, new_size}, or {-1,-1} if losing.
pair<int,int> nim_winning_move(const vector<int>& piles) {
    int nim_sum = 0;
    for (int p : piles) nim_sum ^= p;
    if (nim_sum == 0) return {-1, -1};  // Already a P-position; no winning move

    for (int i = 0; i < (int)piles.size(); i++) {
        int target = piles[i] ^ nim_sum;
        if (target < piles[i]) return {i, target};
    }
    return {-1, -1};  // Unreachable
}
```

== Grundy Values (Nimbers)

*Definition:* The *Grundy value* (nimber) $G(p)$ of a game position $p$ is defined recursively:

$ G(p) = "mex" { G(q) : q in "moves"(p) } $

where $"mex"(S)$ is the *minimum excludant* -- the smallest non-negative integer *not* in the set $S$.

*Base case:* $G(p) = "mex"({}) = 0$ for terminal positions (no moves).

*Key property:*
- $G(p) = 0 arrow.l.r p$ is a P-position.
- $G(p) > 0 arrow.l.r p$ is an N-position.

*Mex implementation:*

```cpp
// mex of a set of non-negative integers.
int mex(const vector<int>& reachable) {
    unordered_set<int> s(reachable.begin(), reachable.end());
    int m = 0;
    while (s.count(m)) m++;
    return m;
}
```

*Generic memoised Grundy solver:* The template below works for any game where `moves(state)` returns the reachable states from a given state.

```cpp
// Generic Grundy value memoizer.
// State must be hashable (int, pair<int,int>, etc.).
// Provide: vector<State> moves(State s) -- returns reachable states.
template<typename State, typename MovesFn>
struct GrundyMemo {
    unordered_map<State, int> cache;
    MovesFn moves_fn;

    explicit GrundyMemo(MovesFn fn) : moves_fn(fn) {}

    int grundy(State s) {
        auto it = cache.find(s);
        if (it != cache.end()) return it->second;

        vector<int> reach;
        for (State nx : moves_fn(s)) {
            reach.push_back(grundy(nx));
        }

        int g = mex(reach);
        cache[s] = g;
        return g;
    }
};
```

*Usage example -- single Nim pile with at most $k$ stones removable per move:*

```cpp
// Nim with move limit: from pile of size n, remove 1..k stones.
// Grundy value follows period-(k+1) pattern: G(n) = n % (k+1).
int grundy_bounded_nim(int n, int k,
                       unordered_map<int,int>& cache) {
    if (n == 0) return 0;
    auto it = cache.find(n);
    if (it != cache.end()) return it->second;

    vector<int> reach;
    for (int r = 1; r <= min(n, k); r++) {
        reach.push_back(grundy_bounded_nim(n - r, k, cache));
    }
    return cache[n] = mex(reach);
}
```

*Observation:* For bounded Nim with limit $k$, $G(n) = n mod (k + 1)$, provable by induction.

== The Sprague-Grundy Theorem

*Theorem:* Every finite, impartial game under normal play is equivalent to a single Nim pile whose size equals the game's Grundy value.

*Composite game (sum of games):* Two players alternate; on each turn a player moves in exactly one component game. The combined game ends when no moves exist in any component.

*Key result:* If $G_1, G_2, dots, G_m$ are the Grundy values of the components, then:

$ G("sum") = G_1 xor G_2 xor dots.c xor G_m $

*Consequence:* A composite position is a P-position iff $G_1 xor G_2 xor dots.c xor G_m = 0$.

*Proof sketch:* Each component game behaves identically to a Nim pile of size $G_i$ with respect to P/N classification; standard Nim analysis then applies to the XOR of those sizes.

== Variants and Extended Problems

=== Staircase Nim

*Setup:* Stones sit on steps $0, 1, 2, dots, n$. A move takes any number of stones from step $i > 0$ and places them on step $i - 1$. Stones on step 0 are dead. Normal play.

*Key insight:* Only odd-indexed steps matter. The position is a P-position iff the XOR of stones on all *odd-numbered* steps is 0.

*Proof:* Moving stones from an even step to an odd step can always be countered by mirroring them one step lower; moves on odd steps behave exactly like Nim moves.

```cpp
// Staircase Nim: piles[i] = stones on step i (step 0 is the graveyard).
bool staircase_nim_wins(const vector<int>& piles) {
    int nim_sum = 0;
    for (int i = 1; i < (int)piles.size(); i += 2) {
        nim_sum ^= piles[i];
    }
    return nim_sum != 0;
}
```

=== Nim with Move Limits (Bounded Nim)

Remove between 1 and $k$ stones from a single pile of $n$ stones. Grundy value: $G(n) = n mod (k + 1)$. With multiple piles, XOR the per-pile Grundy values.

```cpp
// Multi-pile bounded Nim.
bool bounded_nim_wins(const vector<int>& piles, int k) {
    int nim_sum = 0;
    for (int p : piles) nim_sum ^= (p % (k + 1));
    return nim_sum != 0;
}
```

=== Green Hackenbush (brief)

Edges of a graph are coloured; a player removes any edge and all components disconnected from the ground vanish. For a bamboo stalk (path graph) of length $n$ rooted at ground, $G = n$. For trees, the *colon principle* applies: branches meeting at a vertex may be replaced by a single stalk whose length is the XOR of their Grundy values, collapsing the tree bottom-up. The full analysis reduces to Nim via Grundy values.

== Misere Play (Brief Caveat)

Under *misère play* the player who takes the last stone *loses*. For Nim:
- If all piles have size $<= 1$: you want an *odd* number of size-1 piles remaining (opposite of normal).
- Otherwise (at least one pile has size $>= 2$): play as normal Nim, but when you reduce to only size-1 piles, leave an *odd* count.

*Misère strategy:* Play exactly like normal Nim until all piles are of size 0 or 1; at that point invert the parity objective. For general impartial games, misère analysis is considerably harder and does not reduce simply to XOR.

== Partisan Games (Brief Mention)

In *partisan* games, the set of available moves differs between the two players (e.g., Chess, Go). These require *surreal number* theory (Conway, 1976) rather than Grundy values. Values like $1, 1/2, -1, star$ encode the "temperature" and advantage of a position. The full theory is beyond contest scope but underlies endgame analysis in Go.

== Typical Contest Patterns

#table(
  columns: (auto, auto, auto),
  [*Pattern*], [*Technique*], [*Signal in problem statement*],
  [Single-pile removal game], [Compute $G(n)$, check $= 0$], ["take up to $k$ stones"],
  [Multi-pile Nim], [XOR nim-sum], ["multiple piles, take from one"],
  [Composite of sub-games], [XOR of Grundy values], ["move in exactly one component"],
  [Staircase / passing game], [XOR of odd-step piles], [stones move toward a sink],
  [Bounded removal], [$n mod (k+1)$ per pile], [fixed upper bound on removal],
  [Misère Nim], [Normal XOR, invert parity at end], ["last to move loses"],
)

*Heuristic:* If positions are small (e.g., $<= 10^4$) or the state space has few dimensions, memoised Grundy is straightforward. If positions are large (e.g., $n <= 10^9$), look for a closed-form period pattern (the Sprague-Grundy sequence is usually eventually periodic).

== Complexity Reference

#table(
  columns: (auto, auto, auto),
  [*Algorithm / problem*], [*Time*], [*Space*],
  [Classic Nim (decide winner)], [$O(k)$], [$O(1)$],
  [Nim winning move], [$O(k)$], [$O(1)$],
  [Grundy DFS (state space $S$)], [$O(S dot "branching")$], [$O(S)$],
  [mex computation], [$O(d)$ ($d$ = out-degree)], [$O(d)$],
  [Staircase Nim (decide winner)], [$O(k)$], [$O(1)$],
  [Bounded multi-pile Nim], [$O(k)$], [$O(1)$],
)

== Further Reading

Berlekamp, E. R., Conway, J. H., & Guy, R. K. (2001). _Winning Ways for your Mathematical Plays_, 2nd ed. A K Peters. (The definitive reference on combinatorial game theory; covers Nim, Hackenbush, surreal numbers, and hundreds of specific games.)

Sprague, R. P. (1935). "Über mathematische Kampfspiele." _Tohoku Mathematical Journal_ 41: 438--444. (Original proof that every finite impartial game is equivalent to Nim.)

Grundy, P. M. (1939). "Mathematics and Games." _Eureka_ 2: 6--8. (Independent derivation of nimbers and the mex function.)

Ferguson, T. S. (2014). _Game Theory_, Parts I-IV. UCLA lecture notes. (Freely available; excellent treatment of impartial and partisan games with worked examples for competitive programming.)

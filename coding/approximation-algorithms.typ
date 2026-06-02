= Approximation Algorithms

When a problem is NP-hard, polynomial-time algorithms that always return the optimum are unlikely to exist. *Approximation algorithms* trade off optimality for tractability: they run in polynomial time and produce a solution provably within a factor $alpha$ of the optimum. This chapter covers the main techniques (greedy, LP rounding, primal-dual, local search), the hierarchy from constant-factor approximations to PTAS and FPTAS, and canonical results for set cover, vertex cover, and TSP (Christofides).

*See also:* _Greedy_, _Linear Programming and Simplex_, _Network Flows and Matching_, _Dynamic Programming_.

== Vocabulary

An algorithm is an *$alpha$-approximation* for a minimization problem if for every instance,

$ "ALG"(I) <= alpha dot.op "OPT"(I), quad alpha >= 1. $

For maximization, $"ALG"(I) >= "OPT"(I) / alpha$ (so $alpha >= 1$ again). The *approximation ratio* may be a constant, a function of input size, or even arbitrarily close to 1.

#table(
  columns: (auto, auto, auto),
  [*Class*], [*Definition*], [*Example*],
  [APX], [Constant-factor approximation], [Vertex Cover (2)],
  [PTAS], [$forall epsilon$, $(1+epsilon)$ in time poly in $n$], [Euclidean TSP],
  [EPTAS], [PTAS with time $f(epsilon) dot.op n^O(1)$], [Some packing],
  [FPTAS], [PTAS with time poly in $n$ and $1/epsilon$], [Knapsack],
  [Log-approximable], [$O(log n)$-approx], [Set Cover],
  [No approximation], [No $f(n)$-approx unless P=NP], [General TSP],
)

*Lower bound source:* the *PCP theorem* (Arora-Safra 1992) gives unconditional hardness results — e.g., MAX-3SAT has no $7/8 + epsilon$ approximation unless P = NP.

== Vertex Cover: Two Simple 2-Approximations

A *vertex cover* of $G = (V, E)$ is a subset $S subset.eq V$ such that every edge has at least one endpoint in $S$. Finding the minimum is NP-hard.

=== Maximal Matching $->$ 2-Approximation

```python
def vc_match(edges, V):
    M = set(); used = set()
    for (u, v) in edges:
        if u not in used and v not in used:
            M.add((u, v)); used.add(u); used.add(v)
    return used  # vertex cover
```

*Correctness.* $M$ is a maximal matching ($|M|$ edges, $2|M|$ vertices). Any vertex cover must contain $>= 1$ endpoint of every edge of $M$, so $"OPT" >= |M|$. Our cover has $2|M| <= 2 dot "OPT"$.

This is also the *best known* approximation: under the Unique Games Conjecture (Khot 2002), no $(2 - epsilon)$ algorithm exists.

=== LP Rounding

LP relaxation: $min sum_v x_v$ subject to $x_u + x_v >= 1$ $forall (u, v) in E$, $0 <= x_v <= 1$. Solve, then round any $x_v >= 1/2$ up to 1.

*Analysis.* Every edge constraint forces $max(x_u, x_v) >= 1/2$, so at least one endpoint is rounded; the rounded cost is at most $2 sum_v x_v^("LP") <= 2 dot "OPT"$.

== Set Cover: $H_n$-Approximation

Given a universe $U = {e_1, ..., e_n}$ and a family $S_1, ..., S_m subset.eq U$ with weights $w_i$, choose a minimum-weight subfamily that covers $U$. NP-hard, and Feige (1998) proved no $(1 - epsilon) ln n$ approximation exists unless P = NP.

=== Greedy

Repeatedly pick the set minimizing $w_i / |S_i \\ "covered"|$ (cost per *new* element covered).

```python
def set_cover_greedy(U, sets, weights):
    uncovered = set(U); chosen = []
    while uncovered:
        i = min(range(len(sets)), key=lambda j: weights[j] / max(1, len(sets[j] & uncovered)))
        chosen.append(i); uncovered -= sets[i]
    return chosen
```

*Theorem.* Greedy is an $H_n = 1 + 1/2 + ... + 1/n approx ln n$ approximation.

*Sketch.* Charge each element $e$ a price $p_e = w_(i^*) / |S_(i^*) \\ "covered"_(("when " e "is covered"))|$. Order elements as they get covered; the $k$-th element of an optimal set $S^*$ pays at most $w(S^*) / (|S^*| - k + 1)$. Sum: $w("greedy") = sum p_e <= sum_(S^* in "OPT") w(S^*) H_(|S^*|) <= H_n dot "OPT"$.

=== LP Rounding via Randomization

Solve LP, interpret $x_i^*$ as probabilities, include $S_i$ with probability $min(1, c log n dot x_i^*)$. With $c$ large enough, all elements are covered w.h.p. and the expected cost is $O(log n) dot "OPT"$.

== TSP: The Whole Spectrum

#table(
  columns: (auto, auto),
  [*Variant*], [*Best known*],
  [General (asymm.) TSP], [No $f(n)$-approx unless P = NP],
  [Asymm. TSP w/ triangle inequality], [$O(1)$ (Svensson-Tarnawski-Végh 2018)],
  [Symm. TSP w/ triangle inequality (metric)], [$3/2 - 10^(-36)$ (Karlin-Klein-Oveis Gharan 2020); Christofides $3/2$ taught],
  [Euclidean TSP], [PTAS (Arora 1996; Mitchell 1996)],
)

=== Metric TSP: Christofides (1976)

Input: complete graph with metric weights (symmetric, triangle inequality).

```text
1. T  <- minimum spanning tree of G
2. O  <- set of odd-degree vertices in T  (|O| is even)
3. M  <- minimum-weight perfect matching on O (using full edge weights of G)
4. H  <- Eulerian multigraph = T  union  M
5. Find Eulerian circuit of H, then short-cut to a Hamiltonian cycle
   (using triangle inequality to bound cost)
```

*Analysis.*
- $w(T) <= "OPT"$ (the optimal tour minus an edge is a spanning tree).
- $w(M) <= (1/2) "OPT"$ (the optimal tour induces two perfect matchings on $O$; the lighter is $<= (1/2) "OPT"$).
- Short-cutting only decreases weight, so the tour costs $<= w(T) + w(M) <= (3/2) "OPT"$.

For 44 years Christofides was the best known; the 2020 *Karlin-Klein-Oveis Gharan* result uses a random spanning tree drawn from a max-entropy distribution to beat $3/2$ by a tiny constant.

== Knapsack: An FPTAS

0/1 knapsack with $n$ items, weights $w_i$, profits $p_i$, capacity $W$: classical DP gives $O(n W)$ (pseudo-polynomial) or $O(n P)$ where $P = sum p_i$.

*FPTAS (Ibarra-Kim 1975).* Scale profits down: $p'_i = floor(p_i / K)$ with $K = epsilon dot p_max / n$. Run DP on $p'$ in time $O(n^2 floor(p_max / K)) = O(n^3 / epsilon)$. Truncation error per item is at most $K$; over the chosen $<= n$ items the loss is at most $n K = epsilon p_max <= epsilon dot "OPT"$.

```python
def knapsack_fptas(weights, profits, W, eps):
    n = len(weights); p_max = max(profits)
    K = max(1.0, eps * p_max / n)
    p = [int(pi / K) for pi in profits]
    P = sum(p)
    INF = float('inf')
    # dp[j] = min weight to achieve scaled profit j
    dp = [INF] * (P + 1); dp[0] = 0
    for i in range(n):
        for j in range(P, p[i] - 1, -1):
            if dp[j - p[i]] + weights[i] <= W:
                dp[j] = min(dp[j], dp[j - p[i]] + weights[i])
    best = max(j for j in range(P + 1) if dp[j] <= W)
    return best * K  # within (1 - eps) of OPT
```

== Primal-Dual: Vertex Cover Revisited (Weighted)

The *primal-dual schema*: maintain a feasible dual, raise dual variables greedily, freeze tight primal variables. For weighted vertex cover with primal $min sum w_v x_v$, $x_u + x_v >= 1$, dual $max sum y_e$ s.t. $sum_(e: v in e) y_e <= w_v$:

```text
y_e <- 0 for all e
while some edge (u, v) has y_{(u,v)} not yet tight at u or v:
    raise y_{(u,v)} until one of  sum_(e: v) y_e = w_u  or  = w_v
    add the tightened vertex to the cover
return cover
```

This yields a 2-approximation for *weighted* vertex cover; the same template gives constant-factor algorithms for feedback vertex set, Steiner forest, facility location, and more.

== Local Search: Max-Cut

*Local search* for MAX-CUT: while moving any single vertex to the other side increases the cut, do it. The output is a local optimum where each vertex has $>= 1/2$ of its incident edges crossing the cut, so the cut size is $>= |E| / 2 >= "OPT" / 2$.

Random assignment also gives expected $|E|/2$ — the simplest randomized $1/2$-approximation. Goemans-Williamson (1995) use SDP rounding to achieve $0.878$, *the* canonical SDP-rounding result.

== Hardness Cheat Sheet

#table(
  columns: (auto, auto),
  [*Problem*], [*Hardness*],
  [3-SAT], [No $7/8 + epsilon$ (Hastad 2001)],
  [Vertex Cover], [No $1.36$ (Dinur-Safra); $2 - epsilon$ under UGC],
  [Set Cover], [No $(1 - epsilon) ln n$ (Dinur-Steurer 2014)],
  [Max-Cut], [No $0.94 + epsilon$ (Hastad); $0.878$ under UGC tight],
  [Clique], [No $n^(1 - epsilon)$ (Hastad 1999)],
  [TSP (general)], [No constant unless P = NP],
)

== Choosing a Technique

#table(
  columns: (auto, auto),
  [*Symptom*], [*Try first*],
  [Covering / packing IP], [LP rounding or primal-dual],
  [Geometric / planar input], [Shifted PTAS, separators],
  [Items with sizes and profits], [Scaling + DP (FPTAS)],
  [Graph structure (matchings, trees)], [Combinatorial greedy],
  [Quadratic / inner-product objective], [SDP + Goemans-Williamson rounding],
  [Locally improvable], [Local search with potential argument],
)

== Worked Mini-Case: Weighted Set Cover via LP Rounding

LP: $min sum w_i x_i$, $sum_(i: e in S_i) x_i >= 1$ for all $e$, $0 <= x_i <= 1$.

*Frequency-$f$ rounding.* Let $f = max_e |{i : e in S_i}|$ (max number of sets containing any element). Round $x_i = 1$ if $x_i^* >= 1/f$. Every element has $>= 1$ set rounded (by LP feasibility and pigeonhole), and cost grows by at most a factor of $f$. For interval / disk covers this beats $log n$.

== Further Reading

*Vazirani, V.V. (2003).* Approximation Algorithms. Springer. The standard graduate text.

*Williamson, D.P. & Shmoys, D.B. (2011).* The Design of Approximation Algorithms. Cambridge University Press. Free PDF.

*Christofides, N. (1976).* Worst-case Analysis of a New Heuristic for the Travelling Salesman Problem. CMU Tech. Report 388.

*Karlin, A.R., Klein, N. & Oveis Gharan, S. (2021).* A (Slightly) Improved Approximation Algorithm for Metric TSP. STOC 2021.

*Goemans, M.X. & Williamson, D.P. (1995).* Improved Approximation Algorithms for Maximum Cut and Satisfiability Problems Using Semidefinite Programming. JACM 42(6): 1115-1145.

*Hastad, J. (2001).* Some Optimal Inapproximability Results. JACM 48(4): 798-859.

*Feige, U. (1998).* A Threshold of $ln n$ for Approximating Set Cover. JACM 45(4): 634-652.

*Arora, S. (1998).* Polynomial Time Approximation Schemes for Euclidean TSP and Other Geometric Problems. JACM 45(5): 753-782.

*Ibarra, O.H. & Kim, C.E. (1975).* Fast Approximation Algorithms for the Knapsack and Sum of Subset Problems. JACM 22(4): 463-468.

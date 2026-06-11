= Linear Programming and the Simplex Method

*Linear programming* (LP) optimizes a linear objective over a polyhedron defined by linear constraints. It is the workhorse of operations research and the theoretical backbone of approximation algorithms (LP-rounding), combinatorial optimization (max-flow min-cut is LP duality), and modern machine learning (SVMs, $L^1$ regression). This chapter develops standard form, the simplex method with explicit pivots, strong duality, and a sketch of interior-point methods.

*See also:* _Network Flows and Matching_, _Approximation Algorithms_, _Math & Number Theory_.

== Standard Form

The *primal* LP in standard form:

$ "minimize"   c^T x quad "subject to" quad A x = b, quad x >= 0 $

with $A in RR^(m times n)$, $b in RR^m$, $c in RR^n$. Every LP (with inequalities, free variables, or maximization) can be converted into this form by

- adding *slack variables* to turn $a^T x <= beta$ into $a^T x + s = beta$, $s >= 0$;
- adding *surplus + artificial* variables for $a^T x >= beta$;
- splitting a free variable $x = x^+ - x^-$ with $x^+, x^- >= 0$;
- negating to convert max to min.

A *basic feasible solution* (BFS) chooses $m$ linearly independent columns of $A$ (the *basis* $B$); the other $n - m$ variables (*non-basic*) are set to 0. The basic variables are $x_B = B^(-1) b$. A BFS is feasible iff $x_B >= 0$ and corresponds to a *vertex* of the feasible polyhedron.

*Fundamental theorem of LP:* if an LP has an optimal solution, it has one at a vertex.

== The Simplex Method

Simplex walks from vertex to vertex along edges of the polyhedron, each move decreasing the objective until no improving edge exists.

=== Pivot Step

At a BFS with basis $B$ and non-basis $N$:

1. *Reduced costs:* $bar(c)_N = c_N - (B^(-T) c_B)^T A_N$. If $bar(c)_N >= 0$, the current BFS is optimal.
2. *Entering variable:* pick $j$ with $bar(c)_j < 0$ (e.g., *Dantzig's rule*: most negative; *Bland's rule*: smallest index, prevents cycling).
3. *Direction:* compute $d = B^(-1) A_(.,j)$.
4. *Ratio test:* find $i^* = arg min_(d_i > 0) (x_B)_i / d_i$. If $d <= 0$, the LP is *unbounded*.
5. *Pivot:* leave $i^*$, enter $j$; update $B$.

=== Tableau Form (Pedagogical)

```python
import numpy as np

def simplex(c, A, b, max_iter=10_000, tol=1e-9):
    """
    Minimise c^T x subject to A x = b, x >= 0.
    Assumes b >= 0 and the last m columns of A are an identity (slack basis).
    Returns (status, x, value) with status in {'optimal','unbounded'}.
    """
    m, n = A.shape
    basis = list(range(n - m, n))
    T = np.zeros((m + 1, n + 1))
    T[:m, :n] = A
    T[:m,  n] = b
    T[m,  :n] = c
    # zero reduced costs of initial basis
    for i, j in enumerate(basis):
        T[m] -= T[m, j] * T[i]
    for _ in range(max_iter):
        j = int(np.argmin(T[m, :n]))
        if T[m, j] >= -tol:
            x = np.zeros(n)
            for i, bi in enumerate(basis):
                x[bi] = T[i, n]
            return 'optimal', x, -T[m, n]
        # ratio test
        col = T[:m, j]
        ratios = np.where(col > tol, T[:m, n] / np.where(col > tol, col, 1), np.inf)
        i_star = int(np.argmin(ratios))
        if not np.isfinite(ratios[i_star]):
            return 'unbounded', None, -np.inf
        pivot = T[i_star, j]
        T[i_star] /= pivot
        for r in range(m + 1):
            if r != i_star and abs(T[r, j]) > tol:
                T[r] -= T[r, j] * T[i_star]
        basis[i_star] = j
    raise RuntimeError("iteration limit")
```

This is the *full tableau* simplex. Production solvers use the *revised simplex*, which stores only $B^(-1)$ (or its LU factors) and re-derives pivot columns on demand; cache-friendly, numerically stable, and able to exploit sparsity.

=== Worst Case vs Practice

*Klee-Minty cube (1972):* a contrived LP on which Dantzig's rule visits all $2^n$ vertices, proving simplex is exponential in the worst case.

*Practice:* simplex usually pivots $O(m)$ to $O(m^(3/2))$ times. *Smoothed analysis* (Spielman-Teng 2001) showed that for any LP, random tiny perturbations make simplex polynomial in expectation, explaining the gap between theory and practice.

=== Avoiding Degeneracy and Cycling

A BFS is *degenerate* if some basic variable equals 0; pivots may not decrease the objective and can cycle. Remedies:

- *Bland's rule*: smallest-index entering and leaving — guarantees termination.
- *Lexicographic perturbation*: replace $b$ by $b + (epsilon, epsilon^2, ..., epsilon^m)$ symbolically.

== Two-Phase Method

When no obvious initial BFS exists, introduce artificial variables $a >= 0$ and solve

$ "minimize" sum_i a_i quad "subject to" A x + I a = b, quad x, a >= 0. $

Phase 1 drives $sum a_i$ to 0 (or proves infeasibility); Phase 2 starts from the resulting basis on the original objective. The alternative *Big-M* method adds a large penalty $M sum a_i$ to the original objective — simpler to code but numerically fragile.

== Duality

For the primal $min c^T x$, $A x = b$, $x >= 0$, the *dual* is

$ "maximize"  b^T y quad "subject to" quad A^T y <= c $

(with $y$ free since the primal constraints are equalities).

*Weak duality:* for any primal-feasible $x$ and dual-feasible $y$, $c^T x >= b^T y$.

*Strong duality:* if the primal has an optimum, so does the dual, and the optima are equal.

*Complementary slackness:* at optimum, for every $i$, either $x_i = 0$ or $(A^T y)_i = c_i$ (and symmetrically for slack rows). This gives the algorithmic skeleton for primal-dual algorithms.

=== Worked Example: Max-Flow Min-Cut

Encode max-flow as an LP: variables $f_e >= 0$, constraints $f_e <= c_e$ and flow conservation. The dual variables associated with capacities are $y_e >= 0$ and the conservation variables form node potentials $h_v$; complementary slackness forces $y_e in {0,1}$ at vertices of the dual polyhedron, recovering an *s-t cut* with value equal to the max flow. This is the LP source of the combinatorial theorem.

=== Sensitivity / Shadow Prices

The optimal dual $y^*$ measures how the objective changes per unit increase in $b$:

$ (partial "OPT") / (partial b_i) = y_i^* $

(valid as long as the optimal basis remains optimal). In production planning, $y^*_i$ is the marginal value of resource $i$ — its *shadow price*.

== Integer Linear Programming (Detour)

If $x in ZZ^n$ is required, the problem becomes NP-hard (ILP). Two standard approaches:

- *Branch-and-bound:* solve LP relaxation, branch on a fractional variable, prune by bound.
- *Cutting planes (Gomory):* iteratively add inequalities violated by the current LP optimum but satisfied by every integer point. *Branch-and-cut* combines both.

For 0/1 problems, the *LP relaxation gap* $"OPT"_("LP") <= "OPT"_("ILP")$ provides both lower bounds (for min) and the starting point for many approximation algorithms (next chapter).

== Interior-Point Methods (Sketch)

Karmarkar (1984) showed LPs can be solved in *polynomial time* via interior-point methods (IPMs); modern *primal-dual path-following* IPMs are the state of the art for very large LPs and the foundation of convex optimization solvers.

*Central path.* For the primal-dual pair, define the *log-barrier* perturbation

$ min c^T x - mu sum_(j=1)^n ln x_j quad "s.t." A x = b $

For each $mu > 0$ there is a unique optimum $x(mu)$; as $mu -> 0$, $x(mu)$ converges to an LP optimum.

*KKT system.* Stationarity of the Lagrangian yields, with $S = "diag"(s)$, $X = "diag"(x)$:

$ A x = b, quad A^T y + s = c, quad X S e = mu e, quad x, s >= 0. $

*Newton step* on this system gives the search direction. Each iteration reduces $mu$ by a constant factor; one obtains $O(sqrt(n) log(1/epsilon))$ iterations and per-iteration cost dominated by solving a symmetric positive-definite linear system $A D A^T Delta y = r$ with $D = X S^(-1)$.

```python
# Skeleton of a primal-dual interior-point step (illustrative only).
# Solves: min c^T x s.t. A x = b, x >= 0.
def ipm_step(A, b, c, x, y, s, sigma=0.1):
    import numpy as np
    n = len(x); mu = (x @ s) / n
    # residuals
    r_p = A @ x - b
    r_d = A.T @ y + s - c
    r_c = x * s - sigma * mu
    # Form normal equations: A D A^T dy = -r_p + A D (-r_d + r_c/x)
    D = x / s
    rhs = -r_p + A @ (D * (-r_d + r_c / x))
    dy = np.linalg.solve(A @ np.diag(D) @ A.T, rhs)
    ds = -r_d - A.T @ dy
    dx = -(r_c + x * ds) / s
    # Step length keeping x, s > 0
    def step(v, dv):
        neg = dv < 0
        return 1.0 if not neg.any() else min(1.0, 0.99 * (-v[neg] / dv[neg]).min())
    ap = step(x, dx); ad = step(s, ds)
    return x + ap*dx, y + ad*dy, s + ad*ds
```

*Practical solvers:* HiGHS, CLP, Gurobi, CPLEX, and Mosek all ship both simplex and IPM, choosing based on instance structure. IPMs dominate on very large sparse problems; simplex dominates when *warm-starts* matter (branch-and-bound, re-solves with small data changes).

== Comparing the Two Engines

#table(
  columns: (auto, auto, auto),
  [*Property*], [*Simplex*], [*Interior-Point*],
  [Theoretical complexity], [Exponential worst case], [$O(n^(3.5) L)$ Karmarkar; modern $tilde O(n^omega)$],
  [Practical iterations], [$O(m)$ – $O(m^(3/2))$], [30-100 typical],
  [Per-iteration cost], [Cheap (rank-one update)], [Expensive (KKT solve)],
  [Warm starts], [Excellent], [Poor],
  [Sparsity exploitation], [Strong], [Strong (via Cholesky)],
  [Returns a vertex], [Yes], [No (interior point, needs *crossover*)],
)

== Special-Structure LPs

- *Transportation / assignment* — solved best by *network simplex* (orders of magnitude faster than general simplex).
- *Network flow* — LP with totally unimodular constraint matrix; LP relaxation is integral, so LP optimum is integer optimum (max-flow, bipartite matching).
- *Totally unimodular (TU) matrices.* If every square submatrix has determinant $in {-1, 0, +1}$ and $b$ is integer, all vertices of $A x <= b$ are integer. Bipartite incidence matrices and directed network incidence matrices are TU.

== Modelling Tips

- Always check whether your problem has an integer constraint matrix that is TU before reaching for ILP.
- For $L^1$ regression $min ||A x - b||_1$, introduce $t >= A x - b$ and $t >= -(A x - b)$, then $min sum t_i$.
- For $L^infinity$, one auxiliary $t$ with $-t <= A x - b <= t$.
- Big-M constraints encode disjunctions but inflate the LP relaxation; prefer indicator constraints in modern solvers.

== Further Reading

*Dantzig, G.B. (1963).* Linear Programming and Extensions. Princeton University Press. Original simplex monograph.

*Chvátal, V. (1983).* Linear Programming. W.H. Freeman. The classic undergraduate text.

*Bertsimas, D. & Tsitsiklis, J.N. (1997).* Introduction to Linear Optimization. Athena Scientific.

*Karmarkar, N. (1984).* A New Polynomial-Time Algorithm for Linear Programming. Combinatorica 4(4): 373-395.

*Wright, S.J. (1997).* Primal-Dual Interior-Point Methods. SIAM.

*Spielman, D.A. & Teng, S.-H. (2004).* Smoothed Analysis of Algorithms: Why the Simplex Algorithm Usually Takes Polynomial Time. JACM 51(3): 385-463.

*Boyd, S. & Vandenberghe, L. (2004).* Convex Optimization. Cambridge University Press. Free PDF; the modern reference.

*Nocedal, J. & Wright, S.J. (2006).* Numerical Optimization, 2nd ed. Springer. Practical algorithmics.

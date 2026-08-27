#import "../template.typ": xref

= Iterative Methods <iterative-methods>

When $A$ is large and sparse — a 3-D PDE discretization with $10^8$ unknowns, a graph Laplacian, a kernel matrix touched only through matrix-vector products — direct factorization drowns in fill-in or simply does not fit in memory. Iterative methods trade the certainty of $O(n^3)$ for a sequence of cheap steps, each dominated by a matrix-vector product, that converge at a rate governed by the spectrum. This chapter covers the classical splittings, Krylov subspace methods (CG, MINRES, GMRES, BiCGSTAB), convergence theory, preconditioning, and multigrid.

*See also:* #xref("numerical-computing", "linear-systems", label: "Linear Systems") (direct methods, and when to prefer them), #xref("numerical-computing", "eigenvalue-problems", label: "Eigenvalue Problems") (Krylov subspaces reappear as Lanczos and Arnoldi), #xref("numerical-computing", "error-analysis", label: "Error Analysis") (rounding delays but rarely destroys convergence), #xref("numerical-computing", "optimization-algorithms", label: "Optimization Algorithms") (CG is also a quadratic minimizer).

== Classical Splittings

Write $A = M - N$ with $M$ easy to invert; iterate $M x_(k+1) = N x_k + b$, i.e. $x_(k+1) = x_k + M^(-1) r_k$ with residual $r_k = b - A x_k$. Convergence holds iff the spectral radius $rho(M^(-1) N) < 1$, and the error contracts by that factor per step.

#table(
  columns: 3,
  [*Method*], [*$M$*], [*Notes*],
  [Jacobi], [$D$ (diagonal)], [Embarrassingly parallel; converges for strictly diagonally dominant $A$],
  [Gauss-Seidel], [$D + L$ (lower triangle)], [Uses updated values immediately; converges for SPD $A$],
  [SOR], [$1/omega D + L$], [Over-relaxation; optimal $omega$ can square the convergence rate],
)

For the model Poisson problem on an $n times n$ grid, Jacobi and Gauss-Seidel need $O(n^2)$ iterations — the spectral radius is $1 - O(h^2)$ — and optimally tuned SOR needs $O(n)$. All are obsolete as standalone solvers but live on as *smoothers* inside multigrid and as building blocks for preconditioners: their defining property is that they damp high-frequency error components fast while barely touching smooth ones.

== Krylov Subspaces

Every iterate of the classical methods lies in the *Krylov subspace*

$ K_k (A, r_0) = "span"{r_0, A r_0, A^2 r_0, ..., A^(k-1) r_0}, $

so the natural question is: what is the *best* approximation from that subspace? Krylov methods answer it with different optimality criteria, all implementable with one matrix-vector product per iteration plus short recurrences or an orthogonalization. The error after $k$ steps is $p(A) e_0$ for some polynomial $p$ of degree $k$ with $p(0) = 1$ — Krylov convergence theory is polynomial approximation theory on the spectrum.

== Conjugate Gradients

For SPD $A$, CG (Hestenes-Stiefel, 1952) minimizes the energy norm $parallel x_k - x parallel_A$ over $x_0 + K_k$, using a three-term recurrence: two inner products, three vector updates, one matvec, and $O(n)$ extra memory per step.

```python
def cg(A, b, x0, tol=1e-8, maxiter=1000):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rs = r @ r
    for k in range(maxiter):
        Ap = A @ p
        alpha = rs / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = r @ r
        if rs_new**0.5 < tol:
            break
        p = r + (rs_new / rs) * p   # A-conjugate to all previous directions
        rs = rs_new
    return x
```

The Chebyshev bound governs the worst case:

$ (parallel e_k parallel_A) / (parallel e_0 parallel_A) <= 2 ((sqrt(kappa) - 1) / (sqrt(kappa) + 1))^k, $

so the iteration count scales as $sqrt(kappa(A))$ — already a square-root improvement over steepest descent's $kappa$. But the bound is pessimistic when eigenvalues cluster: $m$ tight clusters mean convergence in roughly $m$ iterations, and outlying eigenvalues are "removed" by the polynomial after a few steps each (superlinear convergence). This is why preconditioning aims at *clustering* the spectrum, not just shrinking $kappa$.

In exact arithmetic CG terminates in at most $n$ steps; in floating point the Lanczos vectors lose orthogonality and convergence merely *delays* — Greenbaum's analysis shows finite-precision CG behaves like exact CG on a nearby matrix with slightly smeared eigenvalues. Treat CG as an iterative method, never as a direct one.

== MINRES, GMRES, and BiCGSTAB

*MINRES* handles symmetric *indefinite* $A$ by minimizing the residual 2-norm over the Krylov space, still with short recurrences. It is the right tool for KKT and saddle-point systems.

*GMRES* (Saad-Schultz, 1986) handles general nonsymmetric $A$: build an orthonormal Krylov basis by the Arnoldi process, minimize $parallel b - A x_k parallel_2$ via a small Hessenberg least-squares problem. The cost is the catch — iteration $k$ stores $k$ basis vectors and does $O(k)$ orthogonalization work, so practice uses *restarts*, GMRES($m$): run $m$ steps, restart from the current iterate. Restarting forfeits the optimality and can stagnate; $m$ between 20 and 100 is typical, chosen by experiment.

*BiCGSTAB* (van der Vorst, 1992) keeps short recurrences for nonsymmetric systems by biorthogonalizing against a shadow Krylov space ($A^top$-based), stabilized to smooth the erratic convergence of BiCG. No optimality property, occasional breakdowns, but constant memory — the usual first try when GMRES restarts stall. QMR and IDR($s$) occupy the same niche.

Decision rule: SPD $arrow.r$ CG; symmetric indefinite $arrow.r$ MINRES; nonsymmetric $arrow.r$ GMRES if memory allows, else BiCGSTAB.

== Preconditioning

Krylov methods earn their keep only when preconditioned. Replace $A x = b$ with $M^(-1) A x = M^(-1) b$ (left), $A M^(-1) y = b$ (right), or the split symmetric form for CG, where $M approx A$ is cheap to apply. The goal: cluster the spectrum of $M^(-1) A$ near 1.

- *Jacobi (diagonal)*: nearly free, fixes bad row scaling, little else.
- *Incomplete factorizations*: ILU(0) computes an LU factorization but discards fill outside the sparsity pattern of $A$; ILUT thresholds by magnitude. Workhorse default for nonsymmetric problems; incomplete Cholesky (IC) for SPD. Fragile for highly nonsymmetric or indefinite matrices.
- *Sparse approximate inverse (SPAI)*: minimize $parallel A M - I parallel_F$ column by column; applies as a matvec, so it parallelizes where triangular solves do not.
- *Domain decomposition*: additive Schwarz (solve overlapping subdomain problems independently) and its coarse-corrected variants; the backbone of parallel PDE solvers (PETSc's default is block Jacobi + ILU).
- *Physics-based*: the best preconditioners come from the problem — a multigrid cycle on the elliptic part, a block factorization for Stokes. A good preconditioner is worth more than any Krylov method refinement.

CG requires the preconditioner to be SPD; GMRES accepts anything. Right preconditioning keeps the *true* residual observable, which simplifies stopping tests.

== Multigrid

Smoothers kill high-frequency error in a few sweeps but smooth error decays at $1 - O(h^2)$ per sweep. Multigrid's insight: error that is smooth on a fine grid is *oscillatory relative to a coarser grid*, where it is also $8 times$ cheaper to handle (in 3-D). The V-cycle: pre-smooth (a few Jacobi/Gauss-Seidel sweeps), restrict the residual to the coarse grid, solve there recursively, prolongate (interpolate) the correction back, post-smooth.

For elliptic problems the V-cycle contracts the error by a *grid-independent* factor (around 0.1 per cycle), giving $O(n)$ total work — asymptotically optimal, the standard against which all elliptic solvers are judged. Full multigrid (FMG) reaches discretization accuracy in one pass. *Algebraic multigrid* (AMG) builds the coarse "grids" from the matrix graph alone — strength-of-connection heuristics select coarse variables and interpolation — extending the method to unstructured meshes and some non-PDE problems; hypre's BoomerAMG and Trilinos ML are the production implementations. In practice multigrid is most robust used as a preconditioner for CG or GMRES rather than as a standalone solver.

== Pitfalls

- *Stopping on the wrong quantity.* $parallel r_k parallel \/ parallel b parallel <= 10^(-8)$ bounds the backward error, not the forward error: the solution error can still be $kappa(A) times 10^(-8)$. With left preconditioning the monitored residual is $M^(-1) r_k$, which can differ from the true residual by orders of magnitude.
- *Comparing iteration counts across preconditioners.* A preconditioner that halves iterations but triples the cost per iteration is a loss. Count time, or matvec-equivalents.
- *Non-reproducibility in parallel.* Inner products reduce across processors in nondeterministic order; CG can take visibly different trajectories run to run. Fixed reduction orders restore determinism at some cost.
- *Restarted GMRES stagnation.* GMRES(20) can cycle forever on indefinite spectra while full GMRES converges in 50 steps. If progress per restart cycle decays, increase $m$ or improve the preconditioner before blaming the method.
- *Breakdown of nonsymmetric short-recurrence methods.* BiCGSTAB can divide by a near-zero inner product; production codes detect and restart. Faber-Manteuffel: no method can have both short recurrences and residual optimality for general $A$ — something must give.

== Further Reading

Saad, Y. (2003). _Iterative Methods for Sparse Linear Systems_, 2nd ed. SIAM.

Trefethen, L., Bau, D. (1997). _Numerical Linear Algebra_. SIAM. Lectures 32-40.

Golub, G., Van Loan, C. (2013). _Matrix Computations_, 4th ed. Johns Hopkins. Chapter 11.

Greenbaum, A. (1997). _Iterative Methods for Solving Linear Systems_. SIAM.

Hestenes, M., Stiefel, E. (1952). "Methods of Conjugate Gradients for Solving Linear Systems." J. Res. Nat. Bur. Standards.

Saad, Y., Schultz, M. (1986). "GMRES: A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems." SIAM J. Sci. Stat. Comput.

van der Vorst, H. (1992). "Bi-CGSTAB: A Fast and Smoothly Converging Variant of Bi-CG." SIAM J. Sci. Stat. Comput.

Briggs, W., Henson, V., McCormick, S. (2000). _A Multigrid Tutorial_, 2nd ed. SIAM.

Benzi, M. (2002). "Preconditioning Techniques for Large Linear Systems: A Survey." J. Comput. Phys.

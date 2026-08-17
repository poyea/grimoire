#import "../template.typ": xref

= Linear Systems

Solving $A x = b$ is the workhorse of scientific computing — discretized PDEs, circuit simulation, regression, Kalman filters, and the inner loop of interior-point optimizers all reduce to it. The direct-method toolkit is mature: factor once, solve cheaply, with stability guarantees inherited from a half-century of error analysis. This chapter covers LU with pivoting, Cholesky, QR via Householder and Givens, the interplay of condition number and residual, sparse direct methods, and the three roads to least squares.

*See also:* #xref("numerical-computing", "error-analysis", label: "Error Analysis") (backward stability of the factorizations), #xref("numerical-computing", "iterative-methods", label: "Iterative Methods") (when direct factorization is too expensive), #xref("numerical-computing", "eigenvalue-problems", label: "Eigenvalue Problems") (QR reappears as an iteration), #xref("numerical-computing", "optimization-algorithms", label: "Optimization Algorithms") (Newton systems are linear systems).

== LU Factorization with Pivoting

Gaussian elimination computes $A = L U$ with unit lower-triangular $L$ and upper-triangular $U$ in $2/3 n^3$ flops; solving then costs two triangular solves at $n^2$ each. Without pivoting the algorithm breaks on a zero pivot and is unstable on small ones: eliminating with pivot $epsilon$ multiplies rows by $1 \/ epsilon$, and the growth swamps everything else.

*Partial pivoting* swaps rows so each pivot is the largest in its column, giving $P A = L U$ with $|l_(i j)| <= 1$ and growth factor $rho <= 2^(n-1)$. The worst case is real but never seen in practice; LAPACK's `getrf` uses partial pivoting unconditionally. *Complete pivoting* (rows and columns) bounds growth polynomially but doubles the search cost and is rarely worth it. *Rook pivoting* is the compromise.

Once factored, multiple right-hand sides are cheap — this asymmetry (expensive factor, cheap solve) shapes algorithm design everywhere: Newton's method refactors only when the Jacobian changes substantially, and Kalman filters carry factors rather than inverses. Never compute $A^(-1)$ explicitly: it costs three times as much as a solve, is less accurate, and destroys sparsity.

```python
import numpy as np
from scipy.linalg import lu_factor, lu_solve

lu, piv = lu_factor(A)        # O(n^3), once
x1 = lu_solve((lu, piv), b1)  # O(n^2) per right-hand side
x2 = lu_solve((lu, piv), b2)
```

== Cholesky Factorization

For symmetric positive definite $A$, the factorization $A = L L^top$ exists, is unique, and needs no pivoting: positive definiteness guarantees positive pivots, and the bounded growth makes Cholesky unconditionally backward stable. Cost is $1/3 n^3$ — half of LU. The algorithm doubles as the cheapest test of positive definiteness: it fails (negative diagonal under the square root) exactly when $A$ is not PD, which is how `scipy` and every Gaussian-process library detect a broken covariance matrix.

$ l_(j j) = sqrt(a_(j j) - sum_(k<j) l_(j k)^2), quad l_(i j) = (a_(i j) - sum_(k<j) l_(i k) l_(j k)) / l_(j j). $

The variant $A = L D L^top$ avoids square roots and extends to symmetric *indefinite* matrices with Bunch-Kaufman pivoting ($1 times 1$ and $2 times 2$ pivot blocks) — needed for KKT systems in optimization, which are symmetric but never definite.

== QR Factorization

$A = Q R$ with orthogonal $Q$ and upper-triangular $R$. Orthogonal transformations have condition number 1 and amplify nothing — QR is the stability workhorse.

=== Householder Reflections

A Householder reflector $H = I - 2 v v^top \/ (v^top v)$ reflects across the hyperplane orthogonal to $v$. Choosing $v = x plus.minus parallel x parallel e_1$ (sign chosen to avoid cancellation) maps a column $x$ to a multiple of $e_1$, zeroing everything below the diagonal in one rank-1 update. Applying $n$ reflectors gives $R$ in $4/3 n^3$ flops (for square $A$); $Q$ is kept in factored form as the $v$ vectors and applied implicitly. Householder QR is backward stable with $parallel hat(Q)^top hat(Q) - I parallel = O(u)$ *independent of $kappa(A)$* — the property Gram-Schmidt lacks.

=== Givens Rotations

A Givens rotation acts on two rows, zeroing one entry:

$ G(theta) = mat(c, s; -s, c), quad c = cos theta, s = sin theta. $

Each rotation costs $O(n)$ and touches only two rows, so Givens wins when zeros are sparse and structured: Hessenberg matrices (one rotation per subdiagonal entry — the QR algorithm's inner loop), updating a factorization after a row is appended (recursive least squares), and parallel architectures where disjoint row pairs rotate concurrently.

== Conditioning and Residuals

The computed solution $hat(x)$ from a backward-stable solver satisfies $(A + Delta A) hat(x) = b$ with $parallel Delta A parallel <= c_n u parallel A parallel$, hence

$ (parallel hat(x) - x parallel) / (parallel x parallel) lt.tilde c_n u kappa(A). $

Read this bound the way practitioners do: with $u approx 10^(-16)$ and $kappa(A) = 10^10$, expect only six correct digits — *and the residual will still be tiny*. The residual $r = b - A hat(x)$ measures backward error; the forward error is the residual amplified by $A^(-1)$. Cheap diagnostics:

- Estimate $kappa(A)$ with a condition estimator (`scipy.linalg.lapack` `gecon`, $O(n^2)$ after factoring) rather than the $O(n^3)$ SVD.
- *Iterative refinement*: compute $r$ (ideally in higher precision), solve $A d = r$ with the existing factors, update $hat(x) <- hat(x) + d$. One step with extended-precision residuals restores full accuracy for moderately conditioned systems; mixed-precision refinement (factor in fp16/fp32, refine to fp64 accuracy) is how modern GPU solvers get fp64 results at tensor-core speed (Haidar et al. 2018).
- *Scaling/equilibration*: row and column scaling can reduce an artificial $kappa$ caused by mismatched units.

== Sparse Direct Methods

When $A$ has $O(n)$ nonzeros, dense $O(n^3)$ factorization is absurd — but naive elimination on a sparse matrix creates *fill-in*: entries that were zero become nonzero. Eliminating node $v$ in the adjacency-graph view connects all of $v$'s neighbors pairwise; the elimination order determines how much fill appears. An arrow matrix factored from the point of the arrow fills completely; factored from the other end, not at all.

Finding the minimum-fill ordering is NP-hard, so heuristics rule:

#table(
  columns: 3,
  [*Ordering*], [*Idea*], [*Best for*],
  [Minimum degree (AMD)], [Eliminate lowest-degree node next], [General sparse, default in many codes],
  [Nested dissection (METIS)], [Recursive graph bisection by separators], [2-D/3-D meshes; optimal asymptotics],
  [Reverse Cuthill-McKee], [Bandwidth reduction by BFS], [Banded solvers, envelope methods],
)

For a 2-D mesh with $n$ unknowns, nested dissection gives $O(n^(3/2))$ flops and $O(n log n)$ fill — provably optimal; for 3-D meshes, $O(n^2)$ flops, which is why 3-D PDE practitioners defect to iterative methods. The modern pipeline is: order (AMD/METIS), *symbolic factorization* (compute the nonzero structure once, via the elimination tree), then *numeric factorization* using supernodes — groups of columns with identical structure — processed as dense blocks to recover BLAS-3 speed. CHOLMOD (SPD), UMFPACK and SuperLU (general), MUMPS and PARDISO (parallel) implement this; `scipy.sparse.linalg.spsolve` wraps SuperLU.

== Least Squares

For overdetermined $A in RR^(m times n)$, $m > n$, minimize $parallel A x - b parallel_2$. Three methods, one tradeoff axis:

*Normal equations.* Solve $A^top A x = A^top b$ by Cholesky. Cheapest ($m n^2 + 1/3 n^3$ flops) but $kappa(A^top A) = kappa(A)^2$: with $kappa(A) = 10^8$, the normal equations are numerically singular in fp64 while QR still delivers eight digits. Acceptable when $A$ is well-conditioned or precision is disposable.

*QR.* Factor $A = Q R$, solve $R x = Q^top b$. Costs $2 m n^2 - 2/3 n^3$ — about twice the normal equations — and the error bound involves $kappa(A)$ plus a $kappa(A)^2$ term scaled by the *residual*: small-residual problems behave like $kappa(A)$. This is the default (`numpy.linalg.lstsq` historically, LAPACK `gels`).

*SVD.* $x = V Sigma^(+) U^top b$, the minimum-norm solution. Most expensive, but the only honest method for rank-deficient or nearly rank-deficient problems: truncating singular values below a tolerance regularizes explicitly, and the decay of $sigma_i$ *tells you* the numerical rank instead of letting Cholesky guess. Ridge regression is the smooth version, shrinking coefficients by $sigma_i \/ (sigma_i^2 + lambda)$ (see the ML volume's _Linear Algebra for ML_).

```python
import numpy as np

def lstsq_qr(A, b):
    Q, R = np.linalg.qr(A, mode="reduced")
    return np.linalg.solve(R, Q.T @ b)   # well-conditioned path

def lstsq_svd(A, b, rcond=1e-12):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = np.where(s > rcond * s[0], 1.0 / s, 0.0)  # truncate small modes
    return Vt.T @ (s_inv * (U.T @ b))
```

Rule of thumb: normal equations for speed on well-conditioned tall-skinny problems, QR by default, SVD when rank is in doubt.

== Further Reading

Golub, G., Van Loan, C. (2013). _Matrix Computations_, 4th ed. Johns Hopkins.

Trefethen, L., Bau, D. (1997). _Numerical Linear Algebra_. SIAM.

Higham, N. (2002). _Accuracy and Stability of Numerical Algorithms_, 2nd ed. SIAM. Chapters 9-20.

Davis, T. (2006). _Direct Methods for Sparse Linear Systems_. SIAM.

Björck, Å. (1996). _Numerical Methods for Least Squares Problems_. SIAM.

George, A. (1973). "Nested Dissection of a Regular Finite Element Mesh." SIAM J. Numer. Anal.

Haidar, A. et al. (2018). "Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed Up Mixed-Precision Iterative Refinement Solvers." SC18.

Amestoy, P., Davis, T., Duff, I. (1996). "An Approximate Minimum Degree Ordering Algorithm." SIAM J. Matrix Anal.

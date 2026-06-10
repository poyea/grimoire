= Eigenvalue Problems

Eigenvalues are the frequencies of a vibrating structure, the stability exponents of a dynamical system, the principal components of a dataset, and the page ranks of the web. Unlike linear systems, eigenvalue problems cannot be solved in finitely many arithmetic operations — Abel's theorem forbids closed forms for the characteristic polynomial beyond degree four — so every eigenvalue algorithm is intrinsically iterative. This chapter covers conditioning, power iteration and its inverse, the QR algorithm, Lanczos and Arnoldi for large sparse problems, and the SVD.

*See also:* _Linear Systems_ (QR factorization is the engine here), _Iterative Methods_ (Krylov subspaces again), _Error Analysis_ (backward stability of orthogonal transformations), _FFT_ (the eigendecomposition of circulant matrices).

== Conditioning of Eigenvalues

For a *symmetric* matrix, eigenvalues are perfectly conditioned: Weyl's theorem gives $|lambda_i (A + E) - lambda_i (A)| <= parallel E parallel_2$. Eigenvectors are another story — the Davis-Kahan theorem bounds the rotation of an eigenvector by $parallel E parallel \/ "gap"$, where the gap is the distance to the nearest other eigenvalue. Clustered eigenvalues have ill-determined individual eigenvectors, though the invariant subspace they span remains well-conditioned.

For nonsymmetric $A$, a simple eigenvalue $lambda$ with right eigenvector $x$ and left eigenvector $y$ has condition number $1 \/ |y^* x|$ (with both normalized) — the secant of the angle between them. Highly nonnormal matrices can have enormous eigenvalue condition numbers, and their eigenvalues can be practically meaningless: *pseudospectra* (Trefethen-Embree) — the regions where $parallel (z I - A)^(-1) parallel$ is large — describe the behavior of $A$ under perturbation and in transient dynamics far better. Never compute eigenvalues as polynomial roots: the map from characteristic-polynomial coefficients to roots is catastrophically ill-conditioned (Wilkinson's polynomial) even when the eigenvalue problem itself is benign. Companion-matrix eigenvalues are how production code finds polynomial roots — the reduction goes the *other* way.

== Power Iteration and Variants

Repeatedly applying $A$ to a vector amplifies the dominant eigencomponent: $x_(k+1) = A x_k \/ parallel A x_k parallel$ converges to the dominant eigenvector at rate $|lambda_2 \/ lambda_1|$ per step. Slow when the top two eigenvalues are close, but each step is one matvec — PageRank is power iteration on the Google matrix, where the damping factor 0.85 *is* the convergence rate.

Two transformations fix the rate:

- *Inverse iteration*: apply $(A - sigma I)^(-1)$; the eigenvalue nearest the shift $sigma$ becomes dominant. One factorization, then a triangular solve per step. The near-singularity of $A - sigma I$ for accurate $sigma$ is harmless — the solve error lies almost entirely in the desired eigenvector direction.
- *Rayleigh quotient iteration*: update the shift each step with $sigma_k = x_k^top A x_k \/ x_k^top x_k$. Converges *cubically* for symmetric matrices — typically three iterations — at the price of a fresh factorization per step.

Subspace (orthogonal) iteration runs power iteration on a block of vectors with QR re-orthonormalization, converging to the dominant invariant subspace; randomized SVD is subspace iteration with a random start block and one or two steps.

== The QR Algorithm

The unequivocal champion for dense eigenproblems. The bare iteration — factor $A_k = Q_k R_k$, form $A_(k+1) = R_k Q_k$ — performs an orthogonal similarity that is, non-obviously, simultaneous inverse iteration on all eigenvectors at once; the diagonal converges to the eigenvalues. The practical algorithm adds three ingredients that turn $O(n^4)$ into $O(n^3)$:

+ *Hessenberg reduction.* Reduce $A$ to upper Hessenberg form (one nonzero subdiagonal) by Householder reflectors in $10/3 n^3$ flops, once. Hessenberg form is preserved by QR steps, and each step drops from $O(n^3)$ to $O(n^2)$ via Givens rotations. Symmetric input becomes tridiagonal, and steps cost $O(n)$.
+ *Shifts.* A QR step with shift $mu$ — factor $A_k - mu I$, recombine, add $mu I$ back — is inverse iteration with shift $mu$ in disguise. The Wilkinson shift (eigenvalue of the trailing $2 times 2$ block nearest the corner entry) gives provable convergence in the symmetric case and cubic convergence in practice. The Francis double-shift applies a complex-conjugate shift pair implicitly in real arithmetic via "bulge chasing."
+ *Deflation.* When a subdiagonal entry becomes negligible, the problem splits into independent smaller blocks.

The result: about 2-3 iterations per eigenvalue, $O(n^3)$ total, backward stable because everything is orthogonal. LAPACK exposes the pipeline as `gehrd` (Hessenberg), `hseqr` (QR iteration), `trevc` (eigenvectors); for symmetric matrices, divide-and-conquer (`syevd`) and MRRR (`syevr`) beat plain QR for eigenvectors. The Schur form $A = Q T Q^*$ (triangular $T$) is what QR actually computes — prefer it to a full eigendecomposition whenever it suffices (matrix functions, invariant subspaces), since it always exists and is computed stably.

== Lanczos and Arnoldi

For large sparse matrices where even $O(n^2)$ storage is unaffordable, project onto a Krylov subspace. The *Arnoldi process* builds an orthonormal basis $Q_k$ of $K_k (A, q_1)$ with $A Q_k = Q_k H_k + "rank-1 residual"$ for a $k times k$ Hessenberg $H_k$; the eigenvalues of $H_k$ (Ritz values) approximate $A$'s extremal eigenvalues, often well for $k$ in the tens while $n$ is in the millions. For symmetric $A$, Arnoldi collapses to the *Lanczos* three-term recurrence and $H_k$ is tridiagonal.

In floating point, Lanczos vectors lose orthogonality precisely when a Ritz value converges (Paige's analysis), producing spurious duplicate "ghost" eigenvalues. Remedies: full re-orthogonalization (expensive), selective or partial re-orthogonalization, or restarting. ARPACK's *implicitly restarted Arnoldi* — exact-shift QR steps applied to $H_k$ compress the subspace toward the wanted eigenvalues — is the production standard (`scipy.sparse.linalg.eigsh`/`eigs`); Krylov-Schur (Stewart, 2001) is its numerically cleaner reformulation, the default in SLEPc.

Interior eigenvalues converge miserably from the extremes of the spectrum; the *shift-and-invert* trick runs Lanczos on $(A - sigma I)^(-1)$, mapping eigenvalues near $sigma$ to dominant ones, at the cost of a sparse factorization. For generalized problems $A x = lambda B x$ (structural vibration: stiffness and mass matrices), shift-and-invert is the standard route. LOBPCG offers a factorization-free preconditioned alternative for the symmetric case.

== The Singular Value Decomposition

$A = U Sigma V^top$ exists for any matrix and is the eigendecomposition done right for nonsymmetric and rectangular data: singular values are the square roots of the eigenvalues of $A^top A$, but forming $A^top A$ explicitly squares the condition number and destroys small singular values below $u parallel A parallel$. The Golub-Kahan algorithm avoids this: bidiagonalize $A$ by Householder reflectors from both sides, then apply implicit-shift QR to the bidiagonal matrix; total $O(m n^2)$, backward stable. Variants: divide-and-conquer (`gesdd`, faster for full SVD, the `numpy.linalg.svd` default) and Jacobi SVD (slowest, but computes small singular values to high *relative* accuracy).

Everything the SVD touches inherits its robustness: numerical rank (count $sigma_i > "tol"$), pseudoinverse, total least squares, low-rank approximation — Eckart-Young: the truncated SVD is the best rank-$k$ approximation in both the 2-norm and the Frobenius norm. For large matrices, *randomized SVD* (Halko-Martinsson-Tropp, 2011) sketches the range with a random Gaussian matrix, optionally does a power iteration or two to sharpen decaying spectra, and reduces the problem to a small dense SVD — $O(m n k)$ with high-probability error bounds; `sklearn`'s `TruncatedSVD` and PCA use it.

== Pitfalls

- *Forgetting symmetry.* `numpy.linalg.eig` on a symmetric matrix returns unordered, possibly slightly complex eigenvalues and non-orthogonal eigenvectors. Use `eigh`: faster, real, ordered, orthonormal.
- *Trusting eigenvalues of nonnormal matrices.* Stability analyses based on eigenvalues alone can be wrong by orders of magnitude in transient regimes; check pseudospectra or $parallel e^(t A) parallel$ directly.
- *Comparing eigenvectors across runs.* Eigenvectors are defined up to sign (up to phase, in complex arithmetic), and degenerate eigenvalues only define a subspace; any per-vector comparison must mod out these freedoms.
- *Forming $A^top A$ for singular values.* Squares $kappa$; singular values below $sqrt(u) parallel A parallel$ are lost entirely. Use the SVD of $A$ itself.
- *Asking Lanczos for interior or clustered eigenvalues without help.* Use shift-and-invert, a spectral filter, or generous subspace dimensions — and re-orthogonalize.

== Further Reading

Golub, G., Van Loan, C. (2013). _Matrix Computations_, 4th ed. Johns Hopkins. Chapters 7, 8, 10.

Trefethen, L., Bau, D. (1997). _Numerical Linear Algebra_. SIAM. Lectures 24-31.

Wilkinson, J. H. (1965). _The Algebraic Eigenvalue Problem_. Oxford.

Parlett, B. (1998). _The Symmetric Eigenvalue Problem_. SIAM.

Trefethen, L., Embree, M. (2005). _Spectra and Pseudospectra_. Princeton.

Demmel, J. (1997). _Applied Numerical Linear Algebra_. SIAM. Chapters 4-5.

Halko, N., Martinsson, P.-G., Tropp, J. (2011). "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions." SIAM Review.

Lehoucq, R., Sorensen, D., Yang, C. (1998). _ARPACK Users' Guide_. SIAM.

Stewart, G. W. (2001). "A Krylov-Schur Algorithm for Large Eigenproblems." SIAM J. Matrix Anal.

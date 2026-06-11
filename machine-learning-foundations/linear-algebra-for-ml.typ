= Linear Algebra for ML

Almost every ML algorithm reduces, at the inner loop, to a sequence of matrix multiplications, decompositions, or solves. Understanding when a decomposition exists, what it costs, and how it conditions a downstream problem is the difference between a stable training run and a NaN at step 17. This chapter focuses on the parts of linear algebra that show up in actual ML pipelines: structured matrices, decompositions (SVD, QR, eigen, Cholesky), matrix calculus, numerical stability, and the dimensionality-aware tricks (Johnson–Lindenstrom, random projections, Kronecker structure) that make large models tractable.

*See also:* _Optimization_ (Hessian structure, Newton's method), _Deep Learning Fundamentals_ (initialization, backprop), _ML Workload Optimization on GPUs_ (matmul throughput), _Transformer Architecture_ (attention as low-rank composition).

== Vector Spaces and Norms

A *norm* on $RR^n$ satisfies positivity, absolute homogeneity, and the triangle inequality. The $L_p$ family is

$ parallel x parallel_p = (sum_i |x_i|^p)^(1/p), quad p in [1, infinity). $

Special cases: $L_1$ (Manhattan, induces sparsity), $L_2$ (Euclidean, induces smoothness), $L_infinity$ (max absolute value). The dual of $L_p$ is $L_q$ with $1/p + 1/q = 1$ — the basis for Hölder's inequality and for understanding why $L_1$ regularization yields sparse solutions while $L_2$ does not.

For matrices, the *Frobenius norm* $parallel A parallel_F = sqrt(sum a_(i j)^2) = sqrt(tr(A^top A))$ and the *spectral norm* $parallel A parallel_2 = sigma_max (A)$ are the two most common. The *nuclear norm* $parallel A parallel_* = sum_i sigma_i (A)$ is the convex envelope of rank and underlies low-rank matrix completion.

== Inner Products and Orthogonality

The standard inner product $⟨x, y⟩ = x^top y$ induces the Euclidean norm. Two vectors are *orthogonal* if $⟨x, y⟩ = 0$; an *orthonormal* basis ${u_i}$ satisfies $⟨u_i, u_j⟩ = delta_(i j)$.

A matrix $Q in RR^(n times n)$ is *orthogonal* iff $Q^top Q = I$. Orthogonal matrices preserve lengths and inner products and have condition number 1 — they are the ideal numerical objects.

== Decompositions

#table(
  columns: 4,
  [*Decomposition*], [*Form*], [*Cost (square $n$)*], [*Use*],
  [LU], [$A = P L U$], [$2/3 n^3$], [Linear solves $A x = b$],
  [Cholesky], [$A = L L^top$, $A succ 0$], [$1/3 n^3$], [SPD systems, Gaussian sampling],
  [QR], [$A = Q R$], [$4/3 n^3$], [Least squares, orthogonalization],
  [Eigen], [$A = V Lambda V^(-1)$], [$O(n^3)$], [Symmetric matrices, PCA],
  [SVD], [$A = U Sigma V^top$], [$O(m n min(m,n))$], [Low rank, pseudo-inverse],
  [Schur], [$A = U T U^top$, $T$ upper-tri], [$O(n^3)$], [Matrix functions, control],
)

=== Singular Value Decomposition

For any $A in RR^(m times n)$ with rank $r$,

$ A = U Sigma V^top, quad U in RR^(m times m), V in RR^(n times n), Sigma in RR^(m times n). $

$U$ and $V$ are orthogonal; $Sigma$ has the singular values $sigma_1 >= ... >= sigma_r > 0$ on its diagonal. Properties:

- *Best rank-$k$ approximation* (Eckart–Young): truncating at $k$ singular values minimizes $parallel A - A_k parallel$ in both Frobenius and spectral norms.
- *Pseudo-inverse:* $A^(+) = V Sigma^(+) U^top$ where $Sigma^(+)$ inverts non-zero singular values.
- *Condition number:* $kappa(A) = sigma_max \/ sigma_min$.

```python
import numpy as np

def low_rank_approx(A, k):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

LoRA (Hu et al. 2021, see `llm/finetuning.typ`) and the original PCA both rest on Eckart–Young.

=== Eigendecomposition of Symmetric Matrices

For symmetric $A in RR^(n times n)$ the spectral theorem gives $A = Q Lambda Q^top$ with orthogonal $Q$ and real $Lambda$. The Rayleigh quotient

$ R(x) = (x^top A x) / (x^top x) $

attains the largest and smallest eigenvalues at the corresponding eigenvectors. Power iteration converges to the top eigenpair at rate $|lambda_2 \/ lambda_1|$ — the basis for PageRank and for the largest-eigenvalue Hessian estimates used in second-order ML.

=== Cholesky and Symmetric Positive Definite Matrices

When $A succ 0$, Cholesky factorization $A = L L^top$ with lower-triangular $L$ exists and is unique. It is twice as fast as LU and exposes positive-definiteness: if Cholesky fails, your covariance estimate is not actually PSD (a common source of bugs in Gaussian processes and Kalman filters).

== Matrix Calculus

ML training requires gradients of scalar losses with respect to matrices. The two conventions are *numerator layout* (gradient has the shape of the output transposed) and *denominator layout* (gradient has the shape of the variable). We use denominator layout throughout: $partial f \/ partial X$ has the shape of $X$.

#table(
  columns: 3,
  [*Function*], [*Gradient*], [*Notes*],
  [$f(x) = a^top x$], [$nabla_x f = a$], [],
  [$f(x) = x^top A x$], [$(A + A^top) x$], [$= 2 A x$ if symmetric],
  [$f(X) = tr(A X)$], [$A^top$], [],
  [$f(X) = tr(X^top A X)$], [$(A + A^top) X$], [],
  [$f(X) = log det X$], [$X^(-top)$], [$X succ 0$],
  [$f(X) = parallel X parallel_F^2$], [$2 X$], [],
  [$f(X) = parallel A X - B parallel_F^2$], [$2 A^top (A X - B)$], [Least squares],
)

The *chain rule* on matrices is best understood by passing back-and-forth between forward and reverse Jacobian-vector products; this is the algorithmic content of autodiff (see _Deep Learning Fundamentals_).

== Least Squares

The classical overdetermined system $A x = b$, $A in RR^(m times n)$, $m > n$, is solved by

$ hat(x) = arg min_x parallel A x - b parallel_2^2 = (A^top A)^(-1) A^top b. $

But forming $A^top A$ squares the condition number. Use QR instead: $A = Q R$, then $hat(x) = R^(-1) Q^top b$. For rank-deficient or ill-conditioned $A$, the SVD-based pseudo-inverse $hat(x) = A^(+) b$ gives the minimum-norm least-squares solution.

```python
import numpy as np

def ridge_regression(X, y, lam):
    # Numerically stable via SVD; equivalent to (X.T X + lam I)^(-1) X.T y
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    d = s / (s**2 + lam)
    return Vt.T @ (d * (U.T @ y))
```

Ridge regression shrinks the SVD coefficients by $sigma_i \/ (sigma_i^2 + lambda)$ — small singular values are damped most, providing implicit regularization against ill-conditioning.

== Structured Matrices

Real ML matrices are rarely dense and unstructured. Exploiting structure changes asymptotic cost and unlocks model designs.

#table(
  columns: 3,
  [*Structure*], [*Matvec cost*], [*Examples*],
  [Diagonal], [$O(n)$], [BatchNorm scale, Adam preconditioner],
  [Tridiagonal / banded], [$O(n)$], [1-D convolutions],
  [Circulant], [$O(n log n)$ via FFT], [Convolutions on rings],
  [Toeplitz], [$O(n log n)$], [Convolutional layers],
  [Low rank], [$O((m+n) r)$], [LoRA, adapters],
  [Kronecker $A times.o B$], [$O(p q (m+n))$], [K-FAC, KroneckerLinear],
  [Sparse], [$O("nnz")$], [Mixture-of-experts, attention masks],
  [Block-diagonal], [$sum O(n_i^2)$], [Grouped convolutions, MoE],
)

The Kronecker product $A times.o B$ has the identity $"vec"(A X B^top) = (B times.o A) "vec"(X)$ which lets second-order methods like K-FAC store a $d^2 times d^2$ Fisher block as two $d times d$ factors.

== Random Projections and Johnson–Lindenstrauss

JL lemma: for $n$ points in $RR^d$ and $epsilon in (0, 1)$, projecting onto a random $k$-dimensional subspace with $k = O(epsilon^(-2) log n)$ preserves all pairwise distances up to $(1 plus.minus epsilon)$ with high probability.

```python
import numpy as np

def jl_project(X, k, rng=None):
    rng = rng or np.random.default_rng()
    d = X.shape[1]
    R = rng.standard_normal((d, k)) / np.sqrt(k)
    return X @ R
```

JL underlies sketching algorithms, FastFood, random features for kernels (Rahimi–Recht), and the analysis of why wide neural networks generalize despite over-parameterization.

== Numerical Stability

Floating point hurts. A few principles that prevent disasters:

- *Avoid subtracting near-equal numbers* (cancellation kills precision). Variance via $EE[X^2] - EE[X]^2$ is the canonical bad formula; use Welford's online algorithm instead.
- *Prefer orthogonal transforms.* QR and SVD have unit condition number contributions.
- *Stabilize softmax* by subtracting the max logit before exponentiation.
- *Use log-space probabilities* for products that span orders of magnitude.
- *Check positive-definiteness via Cholesky*, not eigendecomposition: Cholesky is cheaper and fails loudly.

```python
import numpy as np

def stable_logsumexp(z, axis=-1):
    z_max = np.max(z, axis=axis, keepdims=True)
    return np.squeeze(z_max, axis=axis) + np.log(
        np.sum(np.exp(z - z_max), axis=axis)
    )
```

=== Mixed-Precision Considerations

Training at bf16 or fp16 hides numerical issues. The *master weights* should stay fp32; gradients should be unscaled before the optimizer update; loss scaling prevents fp16 underflow. The bf16 format trades mantissa for exponent — it almost never overflows but accumulates rounding error faster.

== Power Iteration and Lanczos

For very large matrices, iterative methods replace direct factorization. *Power iteration* finds the dominant eigenpair via $v_(k+1) = A v_k \/ parallel A v_k parallel$. The *Lanczos algorithm* generalizes this to find the top-$k$ eigenpairs of a symmetric matrix in $O(k n^2)$ — used for Hessian spectrum estimation in modern optimization research.

```python
import numpy as np

def power_iteration(A, num_iter=100):
    n = A.shape[0]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    for _ in range(num_iter):
        v = A @ v
        v /= np.linalg.norm(v)
    return v, v @ A @ v  # eigenvector, Rayleigh quotient
```

Spectral normalization (Miyato et al. 2018) — used in GAN discriminators — applies one step of power iteration per forward pass to keep the spectral norm of weight matrices bounded by 1.

== Matrix Functions and the Matrix Exponential

For symmetric $A = Q Lambda Q^top$, $f(A) := Q f(Lambda) Q^top$ where $f$ is applied entry-wise to the diagonal. More generally, for any square matrix $A$, the *matrix exponential* is defined by the power series

$ e^A = sum_(k=0)^infinity A^k / k! = I + A + A^2/2! + A^3/3! + dots, $

which converges absolutely for all square matrices because the series of operator norms $sum parallel A parallel^k \/ k! = e^(parallel A parallel)$ is always finite.

The matrix exponential is the fundamental solution operator for linear ODEs: the unique solution to $dot(x)(t) = A x(t)$ with initial condition $x(0) = x_0$ is $x(t) = e^(A t) x_0$. This makes $e^A$ central to continuous-time control theory, Gaussian process ODEs, and neural ODEs.

In sequence modeling, continuous-time state-space models (SSMs) such as S4 (Gu et al. 2022) and Mamba parameterize a discrete recurrence via the zero-order-hold discretization

$ overline(A) = e^(A Delta), quad overline(B) = (A)^(-1)(overline(A) - I) B, $

where $Delta$ is the sampling interval. The matrix exponential $overline(A) = e^(A Delta)$ is computed once at initialization and cached — the recurrence then runs in $O(n)$ per time step rather than requiring a full ODE solve at each forward pass.

*Computation methods* depend on the matrix structure. The Padé approximation — a rational approximant $e^A approx R_(p q)(A)$ of matching Taylor coefficients — is used by `scipy.linalg.expm` and is backward-stable. For normal matrices (those satisfying $A A^top = A^top A$), the Schur decomposition $A = U T U^*$ reduces the exponential to $e^A = U e^T U^*$ where $e^T$ is computed on the triangular factor. For large sparse $A$, Krylov subspace methods (Expokit; Saad 1992) approximate $e^(A t) v$ for a given vector $v$ without ever forming $e^A$ explicitly.

The *matrix square root* $A^(1/2)$ — the unique PSD matrix satisfying $(A^(1/2))^2 = A$ for $A succ.eq 0$ — appears in the Fréchet inception distance (FID) used to evaluate generative image models:

$ "FID" = parallel mu_r - mu_g parallel^2 + tr(Sigma_r + Sigma_g - 2(Sigma_r Sigma_g)^(1/2)), $

where $(mu_r, Sigma_r)$ and $(mu_g, Sigma_g)$ are the mean and covariance of Inception-v3 features for real and generated images. Computing $(Sigma_r Sigma_g)^(1/2)$ via the Schur decomposition is the numerical bottleneck in FID evaluation.

In practice: `torch.matrix_exp(A)` computes the matrix exponential on GPU tensors and is differentiable through autograd; `scipy.linalg.expm` provides a CPU reference using the scaling-and-squaring Padé algorithm.

== Tensor Operations

A *tensor* in ML practice is a multi-dimensional array. The core operations are the *contraction* (Einstein summation, e.g. `np.einsum`) and the *reshape*. The `einops` library makes the transformations explicit and dimension-named.

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q, K, V: (batch, heads, seq, dim)
    d = Q.size(-1)
    scores = torch.einsum("bhid,bhjd->bhij", Q, K) / (d ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return torch.einsum("bhij,bhjd->bhid", attn, V)
```

The `einsum` notation makes it impossible to confuse axes — invaluable when you graduate from MLPs to transformers (`llm/transformer-architecture.typ`) and multi-head attention layouts.

== Pseudo-Inverses and the Four Fundamental Subspaces

For $A in RR^(m times n)$, the four fundamental subspaces are:

- Column space $cal(R)(A) subset RR^m$ (image of $A$),
- Null space $cal(N)(A) subset RR^n$ (kernel of $A$),
- Row space $cal(R)(A^top)$,
- Left null space $cal(N)(A^top)$.

The fundamental theorem: $cal(R)(A^top) perp cal(N)(A)$ and $cal(R)(A) perp cal(N)(A^top)$, with dimensions summing to $n$ and $m$ respectively. The SVD makes this geometrically explicit: $V$'s columns split into a basis for $cal(R)(A^top)$ and $cal(N)(A)$; $U$'s into $cal(R)(A)$ and $cal(N)(A^top)$.

== Conditioning

The *condition number* $kappa(A) = parallel A parallel dot parallel A^(-1) parallel$ governs how perturbations in $A$ or $b$ affect $x$ in $A x = b$:

$ (parallel Delta x parallel) / (parallel x parallel) <= kappa(A) ((parallel Delta A parallel) / (parallel A parallel) + (parallel Delta b parallel) / (parallel b parallel)). $

A matrix with $kappa = 10^k$ loses roughly $k$ digits of precision. Hessians in deep learning routinely have $kappa > 10^6$ — the rationale for preconditioned methods like Adam and Shampoo (see _Optimization_).

== Further Reading

Trefethen, L., Bau, D. (1997). _Numerical Linear Algebra_. SIAM. Foundational.

Golub, G., Van Loan, C. (2013). _Matrix Computations_, 4th ed. Johns Hopkins.

Strang, G. (2019). _Linear Algebra and Learning from Data_. Wellesley-Cambridge.

Higham, N. (2002). _Accuracy and Stability of Numerical Algorithms_, 2nd ed. SIAM.

Mahoney, M. (2011). "Randomized Algorithms for Matrices and Data." Foundations and Trends in ML.

Rahimi, A., Recht, B. (2007). "Random Features for Large-Scale Kernel Machines." NeurIPS.

Hu, E. et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv.

Miyato, T. et al. (2018). "Spectral Normalization for Generative Adversarial Networks." ICLR.

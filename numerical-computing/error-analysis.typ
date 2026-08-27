#import "../template.typ": overbar, underbar, xref

= Error Analysis <error-analysis>

An algorithm that returns the wrong answer is useless; an algorithm whose error you cannot bound is dangerous. Error analysis is the discipline of separating what is the *problem's* fault (conditioning) from what is the *algorithm's* fault (stability), and of proving bounds that hold for every input rather than the ones you happened to test. This chapter develops forward and backward error, condition numbers, Wilkinson's backward error analysis, concrete bounds for inner products and matrix multiplication, interval arithmetic, and stochastic rounding.

*See also:* #xref("numerical-computing", "floating-point", label: "Floating-Point Arithmetic") (the rounding model these analyses build on), #xref("numerical-computing", "linear-systems", label: "Linear Systems") (backward stability of LU and QR), #xref("numerical-computing", "iterative-methods", label: "Iterative Methods") (where rounding interacts with convergence).

== Forward and Backward Error

Let $y = f(x)$ be the exact answer and $hat(y)$ the computed one.

- *Forward error*: how wrong is the output? $E_("fwd") = parallel hat(y) - y parallel \/ parallel y parallel$.
- *Backward error*: what is the smallest perturbation of the *input* that explains the output exactly? $E_("bwd") = min { parallel Delta x parallel \/ parallel x parallel : hat(y) = f(x + Delta x) }$.

The fundamental rule of thumb connecting them:

$ "forward error" lt.tilde "condition number" times "backward error". $

Backward error is the algorithm's responsibility; the condition number is intrinsic to the problem. An algorithm is *backward stable* if it always produces the exact answer to a nearby problem, $E_("bwd") = O(u)$ — the best one can ask of finite precision, since the input itself was already rounded. A weaker, often sufficient notion is *mixed forward-backward stability*: $hat(y) + Delta y = f(x + Delta x)$ with both perturbations $O(u)$.

== Condition Numbers

For differentiable $f$, the relative condition number at $x$ is

$ kappa(x) = (parallel J(x) parallel dot parallel x parallel) / (parallel f(x) parallel), $

where $J$ is the Jacobian. It measures the worst-case amplification of relative input perturbations. Examples:

#table(
  columns: 3,
  [*Problem*], [*Condition number*], [*Ill-conditioned when*],
  [$x + y$], [$(|x| + |y|) \/ |x + y|$], [Cancellation: $x approx -y$],
  [$x dot y$], [$1$ (relative)], [Never],
  [$sqrt(x)$], [$1 \/ 2$], [Never],
  [$e^x$], [$|x|$], [Large $|x|$],
  [$ln(x)$], [$1 \/ |ln(x)|$], [$x approx 1$],
  [Solve $A x = b$], [$kappa(A) = parallel A parallel parallel A^(-1) parallel$], [Near-singular $A$],
  [Polynomial roots], [Huge for clustered roots], [Wilkinson's polynomial],
)

Wilkinson's polynomial $product_(i=1)^20 (x - i)$ is the cautionary tale: perturbing the coefficient of $x^19$ by $2^(-23)$ moves some roots by more than 1 in the complex plane. The roots are perfectly distinct; the *map from coefficients to roots* is catastrophically ill-conditioned. No algorithm working from the coefficients can do better.

== Stability of Algorithms

The same mathematical formula can be implemented stably or unstably:

- *Inner product, naive summation*: backward stable.
- *Variance via $EE[X^2] - EE[X]^2$*: not stable — cancellation when mean dominates spread. Welford's recurrence is stable.
- *Gram-Schmidt*: classical GS can lose orthogonality completely ($parallel Q^top Q - I parallel approx kappa^2 u$); modified GS bounds it by $kappa u$; Householder QR achieves $O(u)$ regardless of conditioning.
- *Normal equations* for least squares: squares the condition number (see #xref("numerical-computing", "linear-systems", label: "Linear Systems")).
- *Gaussian elimination without pivoting*: unstable; with partial pivoting, stable in practice though not in the worst case.

A useful design heuristic: prefer orthogonal transformations (they do not amplify errors, $kappa = 1$), avoid subtracting computed quantities of similar size, and accumulate in higher precision when cheap.

== Wilkinson's Backward Error Analysis

Wilkinson's insight (1950s, analyzing the Pilot ACE) was to stop asking "how wrong is the answer?" and ask "*which problem did we actually solve?*" Rounding errors in Gaussian elimination, pushed back onto the input, show that the computed factors satisfy

$ hat(L) hat(U) = A + Delta A, quad parallel Delta A parallel_infinity <= c_n u rho parallel A parallel_infinity, $

where $rho$ is the *growth factor* — the ratio of the largest element appearing during elimination to the largest in $A$ — and $c_n$ is a low-degree polynomial in $n$. With partial pivoting $rho <= 2^(n-1)$, attained by a pathological matrix, yet in decades of practice $rho$ stays small; the gap between worst-case theory and observed behavior remains only partially explained (average-case and smoothed analyses give polynomial bounds).

The methodological payoff is huge: backward analysis decouples cleanly. Solving $A x = b$ via backward-stable LU gives a residual $parallel b - A hat(x) parallel$ of order $u parallel A parallel parallel hat(x) parallel$, and the forward error then follows from $kappa(A)$ alone. Small residual does not mean small error — it means *backward* error is small.

== Error Bounds for Inner Products and GEMM

The standard model $"fl"(x compose y) = (x compose y)(1 + delta)$, $|delta| <= u$, yields for the inner product computed by recursive summation:

$ |"fl"(x^top y) - x^top y| <= gamma_n sum_i |x_i y_i|, quad gamma_n = (n u) / (1 - n u). $

The bound grows linearly in $n$ — and it is the bound on $sum |x_i y_i|$, not $|x^top y|$, so relative error blows up exactly when cancellation occurs. For matrix multiplication $C = A B$ with inner dimension $n$,

$ |hat(C) - A B| <= gamma_n |A| |B| $

elementwise (absolute values taken entrywise). Practical consequences:

- *Blocked and pairwise accumulation* replace $gamma_n$ with $gamma_(log n)$-flavored bounds; this is one reason BLAS-3 blocking helps accuracy, not just cache behavior.
- *Mixed-precision GEMM* on tensor cores stores $A, B$ in fp16/bf16 but accumulates in fp32: the $n u$ factor uses the *accumulator's* $u$, so a 4096-long dot product loses $approx 4096 times 2^(-24)$ relative — fine — instead of $4096 times 2^(-11)$ — disastrous.
- Probabilistic analysis (Higham–Mary 2019) shows errors behave like $sqrt(n) u$ on average, since rounding errors partially cancel; this is why fp16 training works at all.

== Interval Arithmetic

Instead of one rounded value, carry an enclosure $[underbar(x), overbar(x)]$ guaranteed to contain the true value, computing each operation with outward rounding (round lower bounds toward $-infinity$, upper toward $+infinity$):

$ [a, b] + [c, d] = [a + c, b + d], quad [a, b] dot [c, d] = ["min"(a c, a d, b c, b d), "max"(a c, a d, b c, b d)]. $

The result is a machine-checkable proof: the answer lies in the interval, period. The weakness is the *dependency problem*: $x - x$ evaluates to $[a - b, b - a]$, not $[0, 0]$, because the two occurrences are treated as independent. Naive interval evaluation of long computations explodes. Remedies include centered forms, affine arithmetic (tracking linear correlations between error terms), and Taylor models. Interval methods power validated ODE solvers, global optimization with branch-and-bound, and computer-assisted proofs — Hales' proof of the Kepler conjecture and Tucker's proof that the Lorenz attractor exists both rest on interval arithmetic.

== Stochastic Rounding

Round-to-nearest is deterministic and biased for any individual value; *stochastic rounding* rounds $x$ up with probability proportional to its distance to the lower neighbor:

$ "SR"(x) = cases(ceil(x) "with probability" (x - floor(x)) \/ (ceil(x) - floor(x)), floor(x) "otherwise") $

(stated here on the floating-point grid rather than integers). The expected value is exact: $EE["SR"(x)] = x$, making rounding errors zero-mean and independent, so they accumulate as a random walk — error $O(sqrt(n) u)$ with high probability instead of the deterministic $O(n u)$, and *stagnation disappears*: with round-to-nearest, adding a tiny gradient update to a large weight can round to no change at all (the update is below half an ulp), permanently stalling low-precision training. Stochastic rounding applies the update with the correct probability instead.

This matters for fp16/FP8 training and for accumulating many small increments (climate models, neuromorphic hardware). Graphcore IPUs implement stochastic rounding in hardware; on GPUs it is emulated with random bits XOR-ed into the rounding decision. The cost is the loss of determinism — reproducibility now requires seeding the rounding RNG.

== Running Error Analysis

Bounds proved a priori are worst-case; a *running error analysis* accumulates the actual local error bound during the computation:

```python
def dot_with_error_bound(x, y, u=2.0**-53):
    """Inner product plus a rigorous running bound on its rounding error."""
    s, mu = 0.0, 0.0
    for xi, yi in zip(x, y):
        p = xi * yi
        s = s + p
        mu = mu + abs(s) + abs(p)   # accumulate local bound
    return s, u * mu / (1 - len(x) * u)
```

This is cheap (one extra accumulation) and far tighter than $gamma_n sum |x_i y_i|$ on benign data. Higham recommends it as the pragmatic middle ground between blind trust and full interval arithmetic.

== Worked Example

Take the $2 times 2$ system $A x = b$ with

$ A = mat(1, 1; 1, 1.0001), quad b = vec(2, 2.0001), quad "exact solution" x = vec(1, 1). $

*Condition number.* $det(A) = 10^(-4)$, so $A^(-1) = 10^4 mat(1.0001, -1; -1, 1)$. In the $infinity$-norm, $parallel A parallel_infinity = 2.0001$ and $parallel A^(-1) parallel_infinity = 10^4 times 2.0001 = 20001$, hence

$ kappa_infinity (A) = 2.0001 times 20001 approx 4.0 times 10^4. $

*Conditioning in action.* Perturb only $b_2$ by $10^(-8)$, a relative change of $5 times 10^(-9)$. The solution moves by $Delta x = A^(-1) (0, 10^(-8))^top = (-10^(-4), 10^(-4))^top$: a relative output change of $10^(-4)$, an amplification of $2 times 10^4 approx kappa \/ 2$. No algorithm can undo this; it is the problem's geometry (two nearly parallel lines).

*Backward error and the bound.* A backward-stable solver (LU with partial pivoting; here $rho = 1$) returns $hat(x)$ with $E_("bwd") = O(u) approx 10^(-16)$. The rule of thumb then bounds the forward error:

$ E_("fwd") lt.tilde kappa_infinity (A) times E_("bwd") approx (4.0 times 10^4) (1.1 times 10^(-16)) approx 4.4 times 10^(-12). $

In fp64 we should expect roughly 12 correct digits in $hat(x)$, having paid about 4 of the 16 available digits to conditioning — and that is exactly what computing this system in fp64 delivers. In fp32 ($u approx 6 times 10^(-8)$) the bound is $approx 2.4 times 10^(-3)$: barely 3 digits, and a $kappa$ of $10^8$ would leave none.

*Small residual, wrong answer.* Now test the candidate "solution" $hat(x) = (2, 0)^top$. Its residual is

$ r = b - A hat(x) = vec(2, 2.0001) - vec(2, 2) = vec(0, 10^(-4)), $

tiny: $parallel r parallel_infinity \/ parallel b parallel_infinity = 5 times 10^(-5)$. The normwise backward error $parallel r parallel_infinity \/ (parallel A parallel_infinity parallel hat(x) parallel_infinity) = 10^(-4) \/ 4.0 approx 2.5 times 10^(-5)$ is equally tiny — yet the forward error is $parallel hat(x) - x parallel_infinity \/ parallel x parallel_infinity = 1$, i.e. 100%. The bound is not even pessimistic here: $kappa times E_("bwd") = 4.0 times 10^4 times 2.5 times 10^(-5) approx 1.0$, met with equality. Small residual certifies small _backward_ error, and on an ill-conditioned problem that certifies nothing about the answer.

== Further Reading

Higham, N. (2002). _Accuracy and Stability of Numerical Algorithms_, 2nd ed. SIAM. The reference for everything in this chapter.

Wilkinson, J. H. (1963). _Rounding Errors in Algebraic Processes_. Prentice-Hall.

Wilkinson, J. H. (1965). _The Algebraic Eigenvalue Problem_. Oxford.

Trefethen, L., Bau, D. (1997). _Numerical Linear Algebra_. SIAM. Lectures 12-15 on conditioning and stability.

Moore, R., Kearfott, R., Cloud, M. (2009). _Introduction to Interval Analysis_. SIAM.

Higham, N., Mary, T. (2019). "A New Approach to Probabilistic Rounding Error Analysis." SIAM J. Sci. Comput.

Connolly, M., Higham, N., Mary, T. (2021). "Stochastic Rounding and Its Probabilistic Backward Error Analysis." SIAM J. Sci. Comput.

Tucker, W. (2011). _Validated Numerics_. Princeton.

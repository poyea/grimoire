#import "../template.typ": theorem, xref

= Numerical Integration <numerical-integration>

Computing $integral_a^b f(x) d x$ to full precision with a few dozen function evaluations is routine for smooth integrands, and surprisingly hard for integrands with singularities, sharp peaks, oscillations, or more than a handful of dimensions. Quadrature rules are built by integrating an interpolant exactly, so their accuracy follows from approximation theory, and production integrators wrap them in adaptive subdivision with error estimates. This chapter covers Newton-Cotes rules and their limits, the surprising power of the trapezoidal rule, Richardson extrapolation, Gaussian and Clenshaw-Curtis quadrature, adaptive Gauss-Kronrod integration with a C++ implementation, singular and infinite integrals through variable transformations, oscillatory integrands, and integration in many dimensions.

*See also:* #xref("numerical-computing", "interpolation-and-approximation", label: "Interpolation and Approximation"), #xref("numerical-computing", "monte-carlo-methods", label: "Monte Carlo Methods and Random Numbers") (high dimensions), #xref("numerical-computing", "eigenvalue-problems", label: "Eigenvalue Problems") (Golub-Welsch), #xref("numerical-computing", "ode-integration", label: "ODE Integration").

== Interpolatory Rules

A quadrature rule approximates the integral by a weighted sum $Q(f) = sum_j w_j f(x_j)$. Integrating the polynomial interpolant through the nodes gives weights $w_j = integral_a^b ell_j (x) d x$, and the rule is then exact for all polynomials of degree up to $n$. Its *degree of exactness* is the highest degree it integrates exactly.

*Newton-Cotes* rules use equispaced nodes. The two that matter are

$ T = h / 2 (f(a) + f(b)), quad S = h / 6 (f(a) + 4 f((a + b) / 2) + f(b)), quad h = b - a, $

the trapezoidal and Simpson's rules, with errors $-h^3 f''(xi) \/ 12$ and $-h^5 f^((4)) (xi) \/ 2880$. Higher-order Newton-Cotes rules are a mistake: their weights eventually become negative and large, for the same reason that equispaced interpolation fails, and the sum $sum |w_j|$ (which bounds the amplification of errors in $f$) grows exponentially. Accuracy is gained instead by *composite* rules, applying a low-order rule on $m$ subintervals of width $h = (b - a) \/ m$: the composite trapezoidal rule has error $O(h^2)$ and composite Simpson $O(h^4)$.

== The Trapezoidal Rule Is Better Than It Looks

The composite trapezoidal rule's $O(h^2)$ error bound hides a structure described by the *Euler-Maclaurin formula*:

$ T_h - integral_a^b f = sum_(k = 1)^p (B_(2k) h^(2k)) / (2k)! (f^((2k - 1)) (b) - f^((2k - 1)) (a)) + O(h^(2p + 2)), $

where $B_(2k)$ are Bernoulli numbers. The error depends only on odd derivatives at the *endpoints*. If those all vanish, as they do for any smooth periodic function integrated over a full period, every term disappears and the error decreases faster than any power of $h$.

#theorem(name: "Trapezoidal rule for periodic functions")[
If $f$ is $2 pi$-periodic and analytic in the strip $|op("Im") z| < a$, where $|f| <= M$, then the $N$-point trapezoidal rule over a period has error at most $4 pi M \/ (e^(a N) - 1)$.
]

The rule converges geometrically, and for periodic integrands it is essentially optimal: Gaussian quadrature brings no advantage. This is why the trapezoidal rule is the method of choice for Fourier coefficients (it is what the FFT computes), contour integrals in the complex plane, and integrals that can be made periodic by a change of variables.

== Richardson Extrapolation and Romberg Integration

When an approximation $A(h)$ has an error expansion $A(h) = A + c_1 h^2 + c_2 h^4 + dots$, combining two step sizes cancels the leading term:

$ A approx (4 A(h \/ 2) - A(h)) / 3, $

which has error $O(h^4)$. Applied repeatedly to the trapezoidal rule, with $h$ halved each time so that earlier function values are reused, this is *Romberg integration*, which builds a triangular table whose diagonal converges rapidly for smooth $f$. (The first extrapolation step reproduces composite Simpson's rule.) Richardson extrapolation is a general tool: it improves finite-difference derivatives, ODE solutions (Bulirsch-Stoer), and any method with a known error expansion.

== Gaussian Quadrature

With $n$ free nodes and $n$ free weights, a rule could match $2n$ moments. *Gaussian quadrature* achieves this.

#theorem(name: "Gauss quadrature")[
For a weight function $w > 0$ on $[a, b]$, let $x_1, dots, x_n$ be the roots of the degree-$n$ orthogonal polynomial for $w$, and $w_j$ the corresponding interpolatory weights. Then $sum_j w_j f(x_j) = integral_a^b f(x) w(x) d x$ for every polynomial $f$ of degree at most $2n - 1$, and no $n$-point rule does better. All weights are positive.
]

For $w = 1$ on $[-1, 1]$ the nodes are roots of Legendre polynomials (*Gauss-Legendre*); other weights give Gauss-Chebyshev, Gauss-Jacobi (for endpoint singularities of the form $(1 - x)^alpha (1 + x)^beta$), Gauss-Laguerre on $[0, infinity)$ with $e^(-x)$, and Gauss-Hermite on $(-infinity, infinity)$ with $e^(-x^2)$. Positive weights mean the rules are perfectly stable.

The nodes and weights are computed, not tabulated. The *Golub-Welsch* algorithm builds the symmetric tridiagonal *Jacobi matrix* from the three-term recurrence of the orthogonal polynomials; its eigenvalues are the nodes, and the weights are proportional to the squared first components of the eigenvectors. That costs $O(n^2)$. For very large $n$ (millions of points), asymptotic expansions and Newton iteration (Glaser-Liu-Rokhlin, Hale-Townsend, Bogaert) compute them in $O(n)$ to full accuracy.

=== Gauss or Clenshaw-Curtis?

*Clenshaw-Curtis* quadrature integrates the interpolant at Chebyshev points, computing the weights with an FFT. Its degree of exactness is only $n$, not $2n - 1$, yet Trefethen showed that for most functions it is about as accurate as Gauss quadrature with the same number of points, because both converge at the rate set by the function's analyticity, and the extra polynomial exactness of Gauss quadrature only helps for functions that are analytic in a large region. Clenshaw-Curtis nodes are *nested*: doubling $n$ reuses all previous function values, which matters for adaptive schemes.

== Adaptive Integration

Real integrands are smooth in some places and difficult in others. *Adaptive* integrators estimate the error on each subinterval and subdivide where it is large. Two pieces are needed: a rule that comes with an error estimate, and a strategy for subdivision.

A *Gauss-Kronrod* pair supplies the estimate cheaply. Kronrod extended the $n$-point Gauss rule to $2n + 1$ points, keeping the Gauss nodes and adding $n + 1$ new ones, with degree of exactness at least $3n + 1$ ($3n + 2$ for odd $n$, by symmetry). The 7-point Gauss rule (exact to degree 13) and its 15-point Kronrod extension (exact to degree 23) share seven nodes, so 15 evaluations give two estimates, and their difference is a conservative error estimate for the less accurate one. The more accurate Kronrod value is returned. This G7-K15 pair is the workhorse of QUADPACK and of most library integrators.

The best strategy is *global* adaptivity: keep all subintervals in a priority queue ordered by estimated error, and always bisect the worst one, until the total estimated error falls below the tolerance. Local strategies, which recurse on each half with half the tolerance, spend evaluations evenly where they are not needed.

```cpp
#include <cmath>
#include <functional>
#include <queue>
#include <vector>

struct gk_result {
    double value;
    double error;
};

// 15-point Gauss-Kronrod rule on [a, b], with the embedded 7-point Gauss rule.
gk_result gauss_kronrod_15(const std::function<double(double)>& f, double a, double b) {
    static constexpr double xk[8] = {
        0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
        0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
        0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
        0.207784955007898467600689403773245, 0.0};
    static constexpr double wk[8] = {
        0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
        0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
        0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
        0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
    static constexpr double wg[4] = {   // Gauss weights at xk[1], xk[3], xk[5], xk[7]
        0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
        0.381830050505118944950369775488975, 0.417959183673469387755102040816327};

    const double centre = 0.5 * (a + b), half = 0.5 * (b - a);
    const double fc = f(centre);
    double kronrod = wk[7] * fc, gauss = wg[3] * fc;
    for (int i = 0; i < 7; ++i) {
        const double dx = half * xk[i];
        const double pair = f(centre - dx) + f(centre + dx);
        kronrod += wk[i] * pair;
        if (i % 2 == 1) gauss += wg[i / 2] * pair;
    }
    return {kronrod * half, std::abs((kronrod - gauss) * half)};
}

struct segment {
    double a, b;
    gk_result r;
    bool operator<(const segment& o) const { return r.error < o.r.error; }
};

gk_result integrate(const std::function<double(double)>& f, double a, double b,
                    double tol, int max_segments = 1000) {
    std::priority_queue<segment> work;
    gk_result total = gauss_kronrod_15(f, a, b);
    work.push({a, b, total});
    while (total.error > tol && static_cast<int>(work.size()) < max_segments) {
        const segment worst = work.top();
        work.pop();
        const double mid = 0.5 * (worst.a + worst.b);
        const gk_result left = gauss_kronrod_15(f, worst.a, mid);
        const gk_result right = gauss_kronrod_15(f, mid, worst.b);
        total.value += left.value + right.value - worst.r.value;
        total.error += left.error + right.error - worst.r.error;
        work.push({worst.a, mid, left});
        work.push({mid, worst.b, right});
    }
    return total;
}
```

The raw difference $|K_15 - G_7|$ is a large overestimate for smooth integrands, because $K_15$ is far more accurate than $G_7$. QUADPACK rescales it as $E = I_"asc" min(1, (200 |K - G| \/ I_"asc")^(3 \/ 2))$, where $I_"asc"$ is the integral of $|f - "mean"|$ over the interval, a heuristic that gives estimates closer to the true error of $K_15$. Production integrators also guard against roundoff (stopping when the error estimate stops decreasing) and track running sums with compensated summation, since the incremental updates above accumulate rounding error over many subdivisions.

Adaptive quadrature can still be fooled. An integrand with a narrow peak that falls between the 15 sample points of the initial interval looks smooth, the error estimate is tiny, and the peak is missed entirely. If the location of difficulties is known (a discontinuity, a kink, a peak), split the interval there and integrate each piece.

== Singularities and Infinite Intervals

An endpoint singularity such as $integral_0^1 x^(-1\/2) d x$ destroys the convergence of polynomial-based rules, and adaptive bisection converges only slowly toward it. Three remedies:

- *Weighted Gauss rules*: if the singularity is known, absorb it into the weight (Gauss-Jacobi for $(1 - x)^alpha (1 + x)^beta$, generalised Gauss-Laguerre).
- *Subtraction*: subtract an analytically integrable function with the same singularity and integrate the smooth remainder.
- *Variable transformations* that make the integrand decay rapidly at the endpoints.

The most effective transformation is *tanh-sinh* (double-exponential) quadrature, due to Takahasi and Mori. Substituting

$ x = tanh(pi / 2 sinh t) $

maps $(-1, 1)$ to $(-infinity, infinity)$, and the Jacobian decays double-exponentially as $|t| -> infinity$, fast enough to overwhelm algebraic and logarithmic endpoint singularities. The transformed integrand is then integrated with the trapezoidal rule, which converges almost geometrically in the number of points because the integrand decays so quickly at both ends, in the same way as it does for periodic functions. Tanh-sinh is the standard method in arbitrary-precision libraries and handles integrands such as $integral_0^1 ln(x) \/ sqrt(x) d x$ with no special treatment.

Infinite intervals are handled similarly: map $[0, infinity)$ to $[0, 1)$ with $x = t \/ (1 - t)$ and integrate the resulting endpoint behaviour, use Gauss-Laguerre or Gauss-Hermite when the integrand decays like their weights, or use the exp-sinh and sinh-sinh variants of the double-exponential transformation.

== Oscillatory Integrals

Integrands such as $f(x) sin(omega x)$ with large $omega$ need many points per oscillation with ordinary rules, so the cost grows linearly with $omega$. *Filon* methods interpolate only the smooth factor $f$ and integrate the product with the oscillatory kernel exactly; *Levin* methods turn the integral into an ODE for a non-oscillatory function. Both become *more* accurate as $omega$ increases. Oscillatory integrals over infinite intervals (Fourier transforms) are often computed by integrating between successive zeros and accelerating the convergence of the resulting alternating series.

== Many Dimensions

A tensor-product rule with $n$ points per dimension needs $n^d$ evaluations in $d$ dimensions. Ten points per dimension in ten dimensions is already $10^(10)$ evaluations: the *curse of dimensionality*. Options by dimension:

#table(
  columns: 3,
  [*Dimension*], [*Method*], [*Notes*],
  [2-3], [Tensor-product Gauss, adaptive cubature (for example Genz-Malik)], [Efficient for smooth integrands],
  [Up to about 10-20], [Sparse grids (Smolyak), adaptive cubature], [Cost grows like $n (log n)^(d - 1)$ for functions with bounded mixed derivatives],
  [High], [Monte Carlo, quasi-Monte Carlo], [Error $O(N^(-1\/2))$ independent of dimension for Monte Carlo; close to $O(N^(-1))$ for QMC on suitable integrands],
)

Sparse grids combine low-order rules in different dimensions so that most points are spent on low-order interactions between variables, which dominate many practical integrands. Beyond a few dozen dimensions, Monte Carlo and quasi-Monte Carlo methods (#xref("numerical-computing", "monte-carlo-methods", label: "Monte Carlo Methods and Random Numbers")) are the only general options.

== Pitfalls

- High-order Newton-Cotes rules with negative weights.
- Trusting an adaptive integrator's error estimate on an integrand with features smaller than the initial sampling.
- Integrating a discontinuous or kinked function without splitting at the break.
- Using Gauss rules on periodic integrands, where the plain trapezoidal rule is better and simpler.
- Requesting tolerances below the noise level of an integrand computed by simulation or iteration, which makes adaptive integrators subdivide forever.

== Further Reading

Davis, P. J. and Rabinowitz, P. (1984). _Methods of Numerical Integration_ (2nd edition). Academic Press.

Piessens, R. et al. (1983). _QUADPACK: A Subroutine Package for Automatic Integration_. Springer.

Golub, G. H. and Welsch, J. H. (1969). "Calculation of Gauss Quadrature Rules." Mathematics of Computation.

Trefethen, L. N. (2008). "Is Gauss Quadrature Better than Clenshaw-Curtis?" SIAM Review.

Trefethen, L. N. and Weideman, J. A. C. (2014). "The Exponentially Convergent Trapezoidal Rule." SIAM Review.

Takahasi, H. and Mori, M. (1974). "Double Exponential Formulas for Numerical Integration." Publications of the Research Institute for Mathematical Sciences.

Hale, N. and Townsend, A. (2013). "Fast and Accurate Computation of Gauss-Legendre and Gauss-Jacobi Quadrature Nodes and Weights." SIAM Journal on Scientific Computing.

Gerstner, T. and Griebel, M. (1998). "Numerical Integration Using Sparse Grids." Numerical Algorithms.

Iserles, A. and Nørsett, S. P. (2005). "Efficient Quadrature of Highly Oscillatory Integrals Using Derivatives." Proceedings of the Royal Society A.

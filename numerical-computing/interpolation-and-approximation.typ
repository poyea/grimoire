#import "../template.typ": theorem, definition, xref

= Interpolation and Approximation <interpolation-and-approximation>

Almost every numerical method replaces a function it cannot handle exactly by one it can: a polynomial, a piecewise polynomial, a rational function, or a truncated series. Quadrature integrates the replacement, ODE solvers step along it, spectral methods differentiate it, and the `exp` and `sin` in a maths library are carefully chosen polynomials. How well the replacement approximates the original, and whether computing it is stable in floating point, decides the accuracy of everything built on top. This chapter covers polynomial interpolation and its failure on equispaced points, Chebyshev points and the barycentric formula, Chebyshev series, splines, least-squares and minimax approximation, rational approximation with AAA, and how elementary functions are implemented.

*See also:* #xref("numerical-computing", "numerical-integration", label: "Numerical Integration"), #xref("numerical-computing", "fft", label: "FFT") (Chebyshev coefficients via the DCT), #xref("numerical-computing", "error-analysis", label: "Error Analysis"), #xref("numerical-computing", "floating-point", label: "Floating Point").

== Polynomial Interpolation

Given $n + 1$ distinct nodes $x_0, dots, x_n$ and values $f_j = f(x_j)$, there is exactly one polynomial $p$ of degree at most $n$ with $p(x_j) = f_j$. In the Lagrange form,

$ p(x) = sum_(j = 0)^n f_j ell_j (x), quad ell_j (x) = product_(k != j) (x - x_k) / (x_j - x_k). $

#theorem(name: "Interpolation error")[
If $f$ has $n + 1$ continuous derivatives on an interval containing the nodes and $x$, then for some $xi$ in that interval
$ f(x) - p(x) = f^((n+1)) (xi) / (n + 1)! product_(j = 0)^n (x - x_j). $
]

The node polynomial $omega(x) = product (x - x_j)$ is where the choice of nodes enters. On equispaced points, $|omega(x)|$ is enormous near the ends of the interval compared with the middle, and for functions whose derivatives grow quickly the error explodes as $n$ increases. Runge's classic example, $f(x) = 1 \/ (1 + 25 x^2)$ on $[-1, 1]$, is smooth, yet its equispaced interpolants diverge near $plus.minus 1$: the error at degree 20 is around 60, and it grows without bound.

The problem is not only theoretical. The *Lebesgue constant* $Lambda_n = max_x sum_j |ell_j (x)|$ bounds how much interpolation can amplify errors in the data (including rounding):

$ norm(f - p)_infinity <= (1 + Lambda_n) norm(f - p^*)_infinity $

where $p^*$ is the best possible polynomial of the same degree. For equispaced nodes $Lambda_n$ grows like $2^(n+1) \/ (e n ln n)$, so even exact arithmetic on perfect data is hopeless beyond a few dozen points. For well-chosen nodes it grows only logarithmically.

== Chebyshev Points

The *Chebyshev points* (of the second kind) on $[-1, 1]$ are

$ x_j = cos(j pi \/ n), quad j = 0, dots, n, $

the projections onto the axis of equally spaced points on the upper half of the unit circle. They cluster near the endpoints with density proportional to $1 \/ sqrt(1 - x^2)$, exactly counteracting the growth of $omega$ there. Their Lebesgue constant is $Lambda_n <= (2 \/ pi) log(n + 1) + 1$, so interpolation in them is essentially as good as the best polynomial approximation.

#theorem(name: "Convergence of Chebyshev interpolation")[
If $f$ is analytic inside the *Bernstein ellipse* $E_rho$ with foci $plus.minus 1$ and semi-axis sum $rho > 1$, and bounded there by $M$, then the Chebyshev interpolant of degree $n$ satisfies $norm(f - p_n)_infinity <= 4 M rho^(-n) \/ (rho - 1)$.
]

Convergence is geometric, with rate set by how far the nearest complex singularity is from $[-1, 1]$. Runge's function has poles at $plus.minus i \/ 5$, giving $rho approx 1.22$: Chebyshev interpolation converges at about one digit every 12 degrees, while equispaced interpolation diverges. For entire functions such as $e^x$, convergence is faster than any geometric rate, and degree 15 already reaches machine precision.

== The Barycentric Formula

The Lagrange form costs $O(n^2)$ per evaluation and is numerically awkward. The *barycentric* form is both fast and stable:

$ p(x) = (sum_j (w_j f_j) / (x - x_j)) / (sum_j w_j / (x - x_j)), quad w_j = 1 / (product_(k != j) (x_j - x_k)). $

The weights depend only on the nodes, cost $O(n^2)$ once, and for Chebyshev points have the closed form $w_j = (-1)^j delta_j$ with $delta_j = 1 \/ 2$ at the two endpoints and 1 elsewhere (any common scale factor cancels). Each evaluation then costs $O(n)$. Higham proved that this formula is forward stable for nodes with a small Lebesgue constant, which, together with the Chebyshev convergence theory, is the foundation of systems such as Chebfun.

```cpp
#include <cmath>
#include <functional>
#include <numbers>
#include <vector>

class chebyshev_interpolant {
    std::vector<double> x_, f_, w_;

public:
    chebyshev_interpolant(const std::function<double(double)>& f, int n) {
        x_.resize(n + 1); f_.resize(n + 1); w_.resize(n + 1);
        for (int j = 0; j <= n; ++j) {
            x_[j] = std::cos(j * std::numbers::pi / n);
            f_[j] = f(x_[j]);
            w_[j] = (j % 2 == 0 ? 1.0 : -1.0) * ((j == 0 || j == n) ? 0.5 : 1.0);
        }
    }

    double operator()(double x) const {
        double num = 0.0, den = 0.0;
        for (std::size_t j = 0; j < x_.size(); ++j) {
            const double d = x - x_[j];
            if (d == 0.0) return f_[j];          // exactly at a node
            const double t = w_[j] / d;
            num += t * f_[j];
            den += t;
        }
        return num / den;
    }
};
```

The division by $x - x_j$ looks dangerous when $x$ is close to a node, but the large terms appear in numerator and denominator alike and cancel in the quotient; only the exact-hit case needs special handling. Functions on a general interval $[a, b]$ are handled by the affine map $x arrow.r.bar (2x - a - b) \/ (b - a)$.

== Chebyshev Series

Every Lipschitz continuous $f$ on $[-1, 1]$ has an absolutely convergent *Chebyshev series* $f(x) = sum_k a_k T_k (x)$, where $T_k (cos theta) = cos(k theta)$ are the Chebyshev polynomials. The substitution $x = cos theta$ turns it into a Fourier cosine series, so the coefficients of the degree-$n$ interpolant at Chebyshev points can be computed from the values with a discrete cosine transform in $O(n log n)$.

The coefficients carry the analysis: they decay at the same geometric rate $rho^(-k)$ as the interpolation error. Software chooses $n$ adaptively by sampling on $n = 16, 32, 64, dots$ points until the trailing coefficients fall below machine precision relative to the largest, then *chopping* the series there. Once $f$ is represented this way, differentiation, integration, root finding (as eigenvalues of a "colleague" matrix) and optimisation all reduce to cheap operations on the coefficients. This is the approach of Chebfun, and it typically gives answers accurate to 15 digits with no user-chosen tolerances.

== Splines

High-degree polynomials are global: changing one data value changes the interpolant everywhere. *Splines* are piecewise polynomials joined smoothly at *knots*, and they are local, cheap and well behaved on any node distribution.

The *cubic spline* through $(x_j, f_j)$ is cubic on each interval and has continuous first and second derivatives at interior knots. Writing $M_j$ for the second derivative at $x_j$, continuity of the first derivative gives one linear equation per interior knot:

$ h_(j-1) M_(j-1) + 2(h_(j-1) + h_j) M_j + h_j M_(j+1) = 6 ((f_(j+1) - f_j) / h_j - (f_j - f_(j-1)) / h_(j-1)), $

with $h_j = x_(j+1) - x_j$. Two end conditions close the system: *natural* ($M_0 = M_n = 0$), *clamped* (given end slopes), or *not-a-knot* (third derivative continuous at the second and second-to-last knots). The matrix is tridiagonal and diagonally dominant, so it is solved in $O(n)$ without pivoting.

```cpp
#include <vector>

struct natural_cubic_spline {
    std::vector<double> x, f, m;   // knots, values, second derivatives

    natural_cubic_spline(std::vector<double> xs, std::vector<double> fs)
        : x(std::move(xs)), f(std::move(fs)), m(x.size(), 0.0) {
        const std::size_t n = x.size() - 1;
        if (n < 2) return;
        std::vector<double> diag(n + 1), upper(n + 1), rhs(n + 1);
        for (std::size_t j = 1; j < n; ++j) {
            const double h0 = x[j] - x[j - 1], h1 = x[j + 1] - x[j];
            diag[j] = 2.0 * (h0 + h1);
            upper[j] = h1;
            rhs[j] = 6.0 * ((f[j + 1] - f[j]) / h1 - (f[j] - f[j - 1]) / h0);
            if (j > 1) {                      // eliminate the sub-diagonal entry h0
                const double factor = h0 / diag[j - 1];
                diag[j] -= factor * upper[j - 1];
                rhs[j] -= factor * rhs[j - 1];
            }
        }
        for (std::size_t j = n - 1; j >= 1; --j)
            m[j] = (rhs[j] - upper[j] * m[j + 1]) / diag[j];
    }

    double operator()(double t) const {
        std::size_t j = 0, hi = x.size() - 1;
        while (hi - j > 1) {                  // binary search for the interval
            const std::size_t mid = (j + hi) / 2;
            (t < x[mid] ? hi : j) = mid;
        }
        const double h = x[j + 1] - x[j];
        const double a = (x[j + 1] - t) / h, b = (t - x[j]) / h;
        return a * f[j] + b * f[j + 1] +
               ((a * a * a - a) * m[j] + (b * b * b - b) * m[j + 1]) * h * h / 6.0;
    }
};
```

For smooth $f$ and spacing $h$, the cubic spline error is $O(h^4)$ with not-a-knot or clamped ends; natural ends introduce $O(h^2)$ errors near the boundary unless $f''$ really vanishes there. Among all twice-differentiable interpolants, the natural spline minimises $integral (s'')^2$, a measure of bending energy, which is where the name (a draughtsman's flexible ruler) comes from.

*B-splines* give a basis of piecewise polynomials with local support, which makes splines practical for fitting and for geometry: each data point influences only a few basis functions, and the resulting linear systems are banded. Non-uniform rational B-splines (NURBS) are the standard representation of curves and surfaces in CAD. When the data are noisy, *smoothing splines* trade interpolation for a penalty on roughness, and when the shape must be preserved (monotone data such as a cumulative distribution), monotone cubic interpolation (Fritsch-Carlson, PCHIP) avoids the overshoots that ordinary splines produce.

== Least Squares and Minimax

Interpolation forces the approximation through the data. Two other criteria are often more appropriate:

- *Least squares* minimises $integral (f - p)^2 w$ or a discrete sum of squared residuals. With an orthogonal basis for the weight (Legendre for $w = 1$, Chebyshev for $w = 1 \/ sqrt(1 - x^2)$) the coefficients are independent inner products, and the problem is perfectly conditioned. Fitting in the monomial basis $1, x, x^2, dots$ produces a Vandermonde system whose condition number grows exponentially with degree: never fit high-degree polynomials in the monomial basis.
- *Minimax* (best uniform) approximation minimises $max_x |f(x) - p(x)|$. By the *equioscillation theorem*, $p^*$ of degree $n$ is optimal exactly when the error attains its maximum magnitude at $n + 2$ points with alternating signs. The *Remez algorithm* exploits this: guess $n + 2$ reference points, solve for the polynomial that equioscillates on them, move the references to the extrema of the new error, and repeat. It converges quadratically.

Chebyshev interpolation is within a factor $1 + Lambda_n$, at most a few, of minimax, so it is the practical default. Minimax matters where every bit counts, which is exactly the case in maths libraries.

== Rational Approximation

Polynomials approximate poorly near singularities, on unbounded domains, and for functions such as $|x|$ or $e^x$ on long intervals. Rational functions $r = p \/ q$ can place poles near singularities and approximate such functions exponentially better: $|x|$ on $[-1, 1]$ has polynomial best errors $O(1 \/ n)$, while Newman showed rational errors of order $e^(-c sqrt(n))$.

Rational approximation was long considered fragile, because fitting $p \/ q$ is nonlinear and naive formulations are ill-conditioned. The *AAA algorithm* (Nakatsukasa, Sète and Trefethen, 2018) made it routine. It represents $r$ in barycentric form, $r(z) = (sum_j w_j f_j \/ (z - z_j)) \/ (sum_j w_j \/ (z - z_j))$, with support points $z_j$ chosen greedily from the sample set where the current error is largest, and weights $w_j$ obtained from the smallest singular vector of a *Loewner matrix* that linearises the least-squares problem. It works on arbitrary point sets in the complex plane, takes a fraction of a second, and has become a standard tool for model reduction, analytic continuation, and fitting frequency responses.

== How Maths Libraries Approximate Functions

A correctly engineered `exp`, `log` or `sin` follows a pattern:

+ *Argument reduction* maps the input to a small interval using an identity: $e^x = 2^k e^r$ with $r = x - k ln 2$ in $[-ln 2 \/ 2, ln 2 \/ 2]$; $sin x$ uses $x - k pi \/ 2$. The reduction must itself be accurate: for huge arguments of trigonometric functions, the Payne-Hanek algorithm uses hundreds of bits of $2 \/ pi$ so that $x - k pi \/ 2$ is not swamped by rounding.
+ *Polynomial or rational approximation* on the reduced interval, usually a minimax polynomial computed by Remez for the target precision, evaluated with Horner's rule or Estrin's scheme (which exposes instruction-level parallelism).
+ *Reconstruction*: combine with exact scaling ($2^k$ via the exponent field) and, where needed, table lookups for the leading part.

Typical libraries guarantee errors below one unit in the last place. *Correctly rounded* libraries (CORE-MATH, CRlibm, LLVM's libc for some functions) return the floating-point number nearest to the exact result for every input, which requires a second, more precise evaluation for the rare inputs where the first result is too close to a rounding boundary, and is the only way to make results identical across platforms. The IEEE 754-2019 standard recommends, but does not require, correct rounding for these functions.

== Pitfalls

- High-degree interpolation on equispaced points (Runge phenomenon), including implicitly, through `polyfit` with high degree on uniform samples.
- Monomial-basis fitting with ill-conditioned Vandermonde matrices.
- Natural spline end conditions where the data have nonzero curvature at the ends.
- Extrapolating any interpolant outside the data interval, where errors grow rapidly.
- Using splines or polynomials on data with noise without smoothing.

== Further Reading

Trefethen, L. N. (2019). _Approximation Theory and Approximation Practice_ (extended edition). SIAM.

Berrut, J.-P. and Trefethen, L. N. (2004). "Barycentric Lagrange Interpolation." SIAM Review.

Higham, N. J. (2004). "The Numerical Stability of Barycentric Lagrange Interpolation." IMA Journal of Numerical Analysis.

de Boor, C. (2001). _A Practical Guide to Splines_ (revised edition). Springer.

Fritsch, F. N. and Carlson, R. E. (1980). "Monotone Piecewise Cubic Interpolation." SIAM Journal on Numerical Analysis.

Nakatsukasa, Y., Sète, O. and Trefethen, L. N. (2018). "The AAA Algorithm for Rational Approximation." SIAM Journal on Scientific Computing.

Driscoll, T. A., Hale, N. and Trefethen, L. N. (2014). _Chebfun Guide_. Pafnuty Publications.

Muller, J.-M. (2016). _Elementary Functions: Algorithms and Implementation_ (3rd edition). Birkhäuser.

Payne, M. H. and Hanek, R. N. (1983). "Radian Reduction for Trigonometric Functions." ACM SIGNUM Newsletter.

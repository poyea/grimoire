#import "../template.typ": theorem, definition, xref

= Discretising Partial Differential Equations <pde-discretization>

Heat flow, elasticity, fluid dynamics, electromagnetics, option pricing and diffusion of probability all lead to partial differential equations, and almost none of them have closed-form solutions on realistic domains. Numerical methods replace the continuous problem by a finite one: values on a grid, averages over cells, or coefficients of local basis functions. Whether the discrete solution approaches the true one as the mesh is refined depends on two properties, consistency and stability, and the type of the equation decides which of them is hard. This chapter covers the classification of PDEs, finite differences with von Neumann stability analysis and the Lax equivalence theorem, time stepping for diffusion with a C++ Crank-Nicolson solver, the difficulties of hyperbolic problems and high-resolution finite-volume schemes with a C++ flux-limited scheme, the finite element method, spectral methods, and how to solve the resulting systems.

*See also:* #xref("numerical-computing", "ode-integration", label: "ODE Integration") (method of lines, stiffness), #xref("numerical-computing", "iterative-methods", label: "Iterative Methods") (multigrid, Krylov), #xref("numerical-computing", "linear-systems", label: "Linear Systems") (sparse direct solvers), #xref("numerical-computing", "fft", label: "FFT") (spectral methods), #xref("numerical-computing", "interpolation-and-approximation", label: "Interpolation and Approximation").

== Three Kinds of Equation

Second-order linear PDEs in two variables, $a u_(x x) + 2b u_(x y) + c u_(y y) + dots = 0$, are classified by the sign of $b^2 - a c$, and the classes behave very differently:

#table(
  columns: 4,
  [*Type*], [*Model equation*], [*Behaviour*], [*Numerical concern*],
  [Elliptic ($b^2 < a c$)], [Poisson: $-nabla^2 u = f$], [Equilibrium; every point depends on all boundary data; solutions smooth inside], [Solving a large sparse system efficiently],
  [Parabolic ($b^2 = a c$)], [Heat: $u_t = kappa nabla^2 u$], [Diffusion; information spreads at infinite speed but decays; solutions smooth for $t > 0$], [Stiffness: explicit time steps limited to $O(h^2)$],
  [Hyperbolic ($b^2 > a c$)], [Advection $u_t + a u_x = 0$, wave $u_(t t) = c^2 u_(x x)$], [Propagation at finite speed along characteristics; discontinuities persist or form], [Stability, numerical diffusion and dispersion, shocks],
)

A problem is *well posed* (Hadamard) if a solution exists, is unique, and depends continuously on the data. The right boundary and initial conditions depend on the type: elliptic problems need conditions on the whole boundary, parabolic ones an initial condition plus boundary conditions, and hyperbolic ones conditions only where characteristics enter the domain. Imposing an outflow boundary condition on an advection problem makes it ill-posed, and no discretisation can fix that.

== Finite Differences

Replace derivatives by difference quotients on a grid with spacing $h$. Taylor expansion gives the *truncation error*:

$ (u(x + h) - 2u(x) + u(x - h)) / h^2 = u''(x) + h^2 / 12 u^((4)) (x) + O(h^4). $

The *five-point Laplacian* on a square grid applies this in each direction, $(u_(i+1,j) + u_(i-1,j) + u_(i,j+1) + u_(i,j-1) - 4u_(i,j)) \/ h^2$, with $O(h^2)$ truncation error. For the Poisson equation, the result is a sparse symmetric positive definite system with five nonzeros per row, whose condition number grows as $O(h^(-2))$. That growth is why simple iterative methods slow down as the grid is refined and why multigrid, whose cost is independent of $h$, is the method of choice.

Finite differences are simple and fast on rectangular grids. On complex geometries they need awkward special stencils at curved boundaries, which is the main reason finite elements and finite volumes dominate engineering codes.

== Consistency, Stability, Convergence

A scheme is *consistent* if its truncation error vanishes as the mesh is refined. Consistency alone does not ensure that the computed solution converges: errors made at each step can be amplified by later steps.

#definition(name: "Stability")[
A time-stepping scheme $u^(n+1) = B_h u^n$ is *stable* if there is a constant $C$, independent of $h$ and $n$, such that $norm(B_h^n) <= C$ for all $n$ with $n Delta t <= T$.
]

#theorem(name: "Lax equivalence")[
For a consistent finite-difference approximation of a well-posed linear initial value problem, stability is necessary and sufficient for convergence.
]

The theorem turns convergence, hard to verify directly, into stability, which can be analysed. *Von Neumann analysis* does this for constant-coefficient problems on periodic or infinite grids. Substitute a Fourier mode $u_j^n = G^n e^(i k j h)$ into the scheme and compute the *amplification factor* $G(k)$. The scheme is stable if $|G(k)| <= 1 + O(Delta t)$ for every wavenumber $k$.

For the heat equation $u_t = kappa u_(x x)$ with forward Euler in time and central differences in space (FTCS),

$ u_j^(n+1) = u_j^n + r (u_(j+1)^n - 2u_j^n + u_(j-1)^n), quad r = kappa Delta t / h^2, $

the amplification factor is $G = 1 - 4r sin^2(k h \/ 2)$, which stays in $[-1, 1]$ only if $r <= 1 \/ 2$. The time step must shrink like $h^2$: halving the grid spacing quadruples the number of steps. This is stiffness in PDE form. The semi-discrete system $dot(u) = A u$ obtained by discretising only in space (the *method of lines*) has eigenvalues of $A$ ranging from $O(1)$ to $O(h^(-2))$, and the stiff-ODE techniques of the ODE chapter apply directly.

== Implicit Time Stepping for Diffusion

Implicit methods remove the step restriction. *Crank-Nicolson* averages the explicit and implicit discretisations, the trapezoidal rule in time:

$ u_j^(n+1) - r / 2 (u_(j+1)^(n+1) - 2u_j^(n+1) + u_(j-1)^(n+1)) = u_j^n + r / 2 (u_(j+1)^n - 2u_j^n + u_(j-1)^n). $

It is second order in both $Delta t$ and $h$ and unconditionally stable, with $G = (1 - 2r sin^2(k h \/ 2)) \/ (1 + 2r sin^2(k h \/ 2))$. In one dimension each step solves a tridiagonal system in $O(n)$ with the Thomas algorithm.

```cpp
#include <vector>

// Solve a tridiagonal system with constant sub/super-diagonal `off` and diagonal `diag`
// (Thomas algorithm, no pivoting: valid for diagonally dominant systems).
std::vector<double> solve_tridiagonal(double off, double diag, std::vector<double> rhs) {
    const std::size_t n = rhs.size();
    std::vector<double> c(n);
    double denom = diag;
    c[0] = off / denom;
    rhs[0] /= denom;
    for (std::size_t i = 1; i < n; ++i) {
        denom = diag - off * c[i - 1];
        c[i] = off / denom;
        rhs[i] = (rhs[i] - off * rhs[i - 1]) / denom;
    }
    for (std::size_t i = n - 1; i-- > 0;) rhs[i] -= c[i] * rhs[i + 1];
    return rhs;
}

// One Crank-Nicolson step for u_t = kappa u_xx with u = 0 at both ends.
// `u` holds interior values; r = kappa * dt / h^2.
void crank_nicolson_step(std::vector<double>& u, double r) {
    const std::size_t n = u.size();
    std::vector<double> rhs(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double left = i > 0 ? u[i - 1] : 0.0;
        const double right = i + 1 < n ? u[i + 1] : 0.0;
        rhs[i] = (1.0 - r) * u[i] + 0.5 * r * (left + right);
    }
    u = solve_tridiagonal(-0.5 * r, 1.0 + r, std::move(rhs));
}
```

With 49 interior points and $Delta t = 0.01$, so $r = 25$, fifty times beyond the explicit limit, ten steps reproduce the decay of the mode $sin(pi x)$ to an error below $2 times 10^(-4)$. Crank-Nicolson has one weakness: for large $r$, $G -> -1$ for high wavenumbers, so sharp initial data produce slowly decaying oscillations. Starting with a few backward Euler steps (*Rannacher smoothing*), or using a second-order L-stable method such as BDF2, damps them. In two and three dimensions, the implicit systems are no longer tridiagonal; alternating-direction implicit (ADI) methods split them into one-dimensional solves, and general codes use multigrid or preconditioned Krylov solvers at each step.

== Hyperbolic Problems

For the advection equation $u_t + a u_x = 0$ with $a > 0$, the solution moves to the right at speed $a$ without changing shape. Discretising it well is surprisingly hard.

- FTCS (central differences in space, forward Euler in time) has $|G| > 1$ for every step size: unconditionally unstable.
- *Upwind* differencing uses the point the information comes from, $u_j^(n+1) = u_j^n - nu (u_j^n - u_(j-1)^n)$ with Courant number $nu = a Delta t \/ h$. It is stable for $0 <= nu <= 1$.
- *Lax-Wendroff* adds a second-order correction: $u_j^(n+1) = u_j^n - nu \/ 2 (u_(j+1)^n - u_(j-1)^n) + nu^2 \/ 2 (u_(j+1)^n - 2u_j^n + u_(j-1)^n)$, stable for $|nu| <= 1$.

The condition $|nu| <= 1$ is the *CFL condition* (Courant, Friedrichs and Lewy, 1928): the numerical domain of dependence must contain the physical one, or the scheme cannot possibly converge. For explicit schemes for hyperbolic problems, $Delta t$ scales like $h$, not $h^2$, which is why explicit methods are standard for wave propagation.

Stability is not the end. Upwind is only first order, and its *modified equation* (the PDE it actually solves more accurately) contains a diffusion term $a h \/ 2 (1 - nu) u_(x x)$: sharp fronts are smeared. Lax-Wendroff is second order, and its leading error is dispersive: different wavelengths travel at different speeds, so discontinuities develop oscillations. Advecting a square pulse once around a periodic grid of 200 points with $nu = 0.5$ shows both: upwind keeps values in $[0, 1]$ but smears the edges (mean error 0.080), while Lax-Wendroff is sharper (mean error 0.052) but overshoots to 1.23 and undershoots to $-0.23$.

#theorem(name: "Godunov")[
A linear scheme for the advection equation that does not create new extrema (is monotone) is at most first-order accurate.
]

== Finite Volumes and High-Resolution Schemes

Physical conservation laws, $u_t + f(u)_x = 0$ (mass, momentum, energy), are best discretised in *conservation form*. Store the average of $u$ over each cell and update it with the fluxes through the cell faces:

$ u_j^(n+1) = u_j^n - (Delta t) / h (F_(j+1\/2) - F_(j-1\/2)). $

Whatever leaves one cell enters its neighbour, so the total is conserved exactly. For nonlinear conservation laws this is essential: the Lax-Wendroff theorem (of 1960, a different result from the scheme above) shows that convergent conservative schemes converge to weak solutions with the right shock speeds, while non-conservative schemes can converge to shocks at the wrong speed. *Godunov's method* computes each face flux by solving the *Riemann problem* (the evolution of a single jump) between the two neighbouring cells exactly or approximately (Roe, HLL, HLLC solvers).

Godunov's theorem says linear schemes cannot be both monotone and second order, so high-resolution schemes are *nonlinear*. A *flux limiter* blends a high-order flux in smooth regions with a first-order upwind flux near discontinuities, using the ratio of consecutive slopes $theta_j = (u_j - u_(j-1)) \/ (u_(j+1) - u_j)$ as a smoothness indicator. Schemes whose limiter satisfies Sweby's conditions are *total variation diminishing* (TVD): $sum_j |u_(j+1) - u_j|$ never increases, so no oscillations appear.

```cpp
#include <algorithm>
#include <vector>

// One step of a flux-limited scheme for u_t + a u_x = 0, a > 0, periodic grid.
// nu = a * dt / h in (0, 1]. The minmod limiter gives a TVD, second-order (in smooth
// regions) scheme that reduces to upwind at discontinuities.
std::vector<double> limited_advection_step(const std::vector<double>& u, double nu) {
    const std::size_t n = u.size();
    auto at = [&](std::size_t i, int offset) { return u[(i + n + offset) % n]; };
    std::vector<double> flux(n);   // flux[i] is the flux through face i + 1/2, divided by a
    for (std::size_t i = 0; i < n; ++i) {
        const double downwind = at(i, 1) - at(i, 0);
        const double upwind = at(i, 0) - at(i, -1);
        const double theta = downwind != 0.0 ? upwind / downwind : 0.0;
        const double phi = std::max(0.0, std::min(1.0, theta));          // minmod
        flux[i] = at(i, 0) + 0.5 * (1.0 - nu) * phi * downwind;
    }
    std::vector<double> next(n);
    for (std::size_t i = 0; i < n; ++i)
        next[i] = u[i] - nu * (flux[i] - flux[(i + n - 1) % n]);
    return next;
}
```

On the square-pulse test, this scheme stays exactly within $[0, 1]$, keeps the total variation at its initial value of 2, and has a mean error of 0.031, less than either linear scheme. The limiter $phi = 1$ everywhere would give Lax-Wendroff, and $phi = 0$ upwind. Smoother limiters (van Leer, superbee, monotonised central) trade sharpness against smoothness. Higher-order successors such as ENO and WENO reconstruct the solution from several candidate stencils, weighting smooth ones most, and reach fifth order or more away from shocks; discontinuous Galerkin methods combine finite-volume ideas with local polynomial spaces.

== The Finite Element Method

Finite elements start from the *weak form* of the equation. For $-u'' = f$ on $(0, 1)$ with $u(0) = u(1) = 0$, multiply by a test function $v$ vanishing at the ends and integrate by parts:

$ integral_0^1 u' v' d x = integral_0^1 f v d x quad "for all" v in H_0^1. $

The *Galerkin method* looks for $u_h$ in a finite-dimensional space $V_h$, typically continuous piecewise polynomials on a mesh, and requires the weak form only for $v in V_h$. With the "hat" basis functions $phi_i$ of piecewise linear elements, $u_h = sum_j U_j phi_j$ satisfies the linear system $K U = F$ with *stiffness matrix* $K_(i j) = integral phi_i' phi_j'$ and *load vector* $F_i = integral f phi_i$. On a uniform mesh, $K$ is the familiar tridiagonal matrix $(1 \/ h) "tridiag"(-1, 2, -1)$.

#theorem(name: "Céa's lemma")[
For a coercive, bounded bilinear form, the Galerkin solution is quasi-optimal: $norm(u - u_h)_V <= (M \/ alpha) min_(v in V_h) norm(u - v)_V$, where $M$ and $alpha$ are the continuity and coercivity constants.
]

Galerkin approximation therefore reduces to approximation theory: with piecewise polynomials of degree $p$ on a mesh of size $h$, the energy-norm error is $O(h^p)$ and the $L^2$ error $O(h^(p+1))$ for smooth solutions. In one dimension with linear elements, the nodal values are even exact when the load is integrated exactly, a superconvergence phenomenon.

The strengths of the method are geometric and structural. Elements can be triangles, tetrahedra, quadrilaterals or hexahedra fitted to any geometry; the stiffness matrix is assembled element by element from small local matrices; boundary conditions on derivatives (*natural* conditions) appear automatically in the weak form; and *a posteriori error estimators* computed from the solution itself drive adaptive mesh refinement to where the error is large. Structural mechanics, electromagnetics and much of computational fluid dynamics rely on it, through packages such as deal.II, FEniCS, MFEM and commercial codes.

== Spectral Methods

For smooth solutions on simple domains, *spectral methods* expand the solution in global basis functions: Fourier modes for periodic problems, Chebyshev or Legendre polynomials otherwise. Derivatives are computed exactly in the basis (with the FFT for Fourier), and the error decreases faster than any power of $N$ for smooth solutions, so a few dozen points per dimension can give ten-digit accuracy. They are the standard tool for turbulence simulation in periodic boxes and for weather and climate dynamical cores on the sphere (with spherical harmonics). Their weaknesses are complex geometries and discontinuities, where convergence collapses to first order and Gibbs oscillations appear; *spectral element* methods recover geometric flexibility by using high-degree polynomials on each of many elements.

== Solving the Discrete Systems

Implicit time steps and elliptic problems produce large sparse linear systems, and nonlinear PDEs add Newton iterations around them. Typical choices:

- Sparse direct solvers (with nested dissection ordering) for two-dimensional problems and moderate three-dimensional ones, especially when many right-hand sides share one matrix.
- Multigrid, geometric or algebraic, for elliptic and parabolic problems: optimal $O(n)$ cost.
- Krylov methods (CG for symmetric positive definite, GMRES otherwise) with preconditioners tailored to the PDE: multigrid, incomplete factorisations, domain decomposition, or block preconditioners for coupled systems such as incompressible flow.

== Pitfalls

- Explicit time steps beyond the stability limit, which produce exponential blow-up, often only after many steps as unstable modes grow from rounding errors.
- Central differences or unlimited high-order schemes for advection-dominated problems, giving oscillations.
- Non-conservative discretisations of conservation laws with shocks.
- Crank-Nicolson with non-smooth data and large steps, leaving spurious oscillations.
- Claiming convergence without a mesh-refinement study; the observed order of accuracy should match the theory.

== Further Reading

LeVeque, R. J. (2007). _Finite Difference Methods for Ordinary and Partial Differential Equations_. SIAM.

LeVeque, R. J. (2002). _Finite Volume Methods for Hyperbolic Problems_. Cambridge University Press.

Strikwerda, J. C. (2004). _Finite Difference Schemes and Partial Differential Equations_ (2nd edition). SIAM.

Courant, R., Friedrichs, K. and Lewy, H. (1928). "Über die partiellen Differenzengleichungen der mathematischen Physik." Mathematische Annalen.

Lax, P. D. and Richtmyer, R. D. (1956). "Survey of the Stability of Linear Finite Difference Equations." Communications on Pure and Applied Mathematics.

Godunov, S. K. (1959). "A Difference Method for Numerical Calculation of Discontinuous Solutions of the Equations of Hydrodynamics." Matematicheskii Sbornik.

Harten, A. (1983). "High Resolution Schemes for Hyperbolic Conservation Laws." Journal of Computational Physics.

Sweby, P. K. (1984). "High Resolution Schemes Using Flux Limiters for Hyperbolic Conservation Laws." SIAM Journal on Numerical Analysis.

Jiang, G.-S. and Shu, C.-W. (1996). "Efficient Implementation of Weighted ENO Schemes." Journal of Computational Physics.

Brenner, S. C. and Scott, L. R. (2008). _The Mathematical Theory of Finite Element Methods_ (3rd edition). Springer.

Trefethen, L. N. (2000). _Spectral Methods in MATLAB_. SIAM.

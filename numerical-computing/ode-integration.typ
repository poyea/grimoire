= ODE Integration

Initial value problems $y' = f(t, y)$, $y(t_0) = y_0$ are the dynamics of everything: orbital mechanics, chemical kinetics, neuron models, the training dynamics of neural ODEs. Numerical integration replaces the continuum by discrete steps, and the central tensions are accuracy versus cost, stability versus step size, and the special demands of stiff problems and long-time simulation. This chapter covers Runge-Kutta methods, error control, stiffness and implicit methods, multistep methods, and symplectic integration.

*See also:* _Error Analysis_ (local truncation error and its propagation), _Linear Systems_ and _Iterative Methods_ (implicit steps are nonlinear solves), _Optimization Algorithms_ (Newton's method inside every implicit integrator), _Floating-Point Arithmetic_ (roundoff floors for tiny steps).

== Euler and the Anatomy of a Step

Forward Euler, $y_(n+1) = y_n + h f(t_n, y_n)$, has *local truncation error* $O(h^2)$ per step; over $T \/ h$ steps the errors compound to *global error* $O(h)$ — order 1. The pattern is general: a method of order $p$ has local error $O(h^(p+1))$ and global error $O(h^p)$, provided the method is *zero-stable* (errors are not amplified by the recurrence itself); Dahlquist's equivalence theorem says consistency plus zero-stability equals convergence.

Halving $h$ with Euler halves the error and doubles the cost — a terrible exchange rate. High-order methods pay a few more $f$ evaluations per step for an error that shrinks like $h^4$ or $h^8$, which is why nobody uses Euler in production. (Exception: stochastic differential equations, where Euler-Maruyama's strong order 0.5 is hard to beat cheaply, and game physics, where stability and speed trump accuracy — semi-implicit Euler, below.)

== Runge-Kutta Methods

An explicit $s$-stage Runge-Kutta method samples $f$ at $s$ carefully chosen points per step and combines them; the coefficients form the Butcher tableau. The classical RK4,

$ k_1 = f(t_n, y_n), quad k_2 = f(t_n + h/2, y_n + h/2 k_1), quad k_3 = f(t_n + h/2, y_n + h/2 k_2), quad k_4 = f(t_n + h, y_n + h k_3), $

$ y_(n+1) = y_n + h/6 (k_1 + 2 k_2 + 2 k_3 + k_4), $

achieves order 4 with 4 stages. The free lunch ends there: order 5 needs 6 stages, order 8 needs 11 (the Butcher barriers), because the number of order conditions — one per rooted tree — grows much faster than the number of coefficients.

*Adaptive step size* is what makes RK methods practical. Embedded pairs evaluate two methods of adjacent orders from the *same* stages; their difference estimates the local error, and the controller picks the next step:

$ h_("new") = h dot min(5, max(0.2, 0.9 (epsilon_("tol") \/ "err")^(1\/(p+1)))). $

Dormand-Prince 5(4) — `RK45` in `scipy.integrate.solve_ivp`, `ode45` in MATLAB — uses 7 stages with FSAL (first-same-as-last, so 6 effective) and a free quartic interpolant for *dense output*: solution values between steps for event location (when does the trajectory cross zero?) and smooth plotting, without constraining the step size. The error tolerance controls the *local* error per step; the global error can exceed `rtol` by orders of magnitude over long integrations — when it matters, verify by re-running at a tighter tolerance and comparing.

== Stiffness

A problem is *stiff* when stability, not accuracy, dictates the step size: the solution is smooth, but nearby trajectories decay on time scales vastly shorter than the interval of interest. Chemical kinetics with rate constants spanning ten orders of magnitude and method-of-lines diffusion (eigenvalues scale as $-1 \/ Delta x^2$) are the canonical cases.

Apply a method to the test equation $y' = lambda y$; the step is stable when $h lambda$ lies in the method's *stability region*. Explicit methods have bounded regions — forward Euler requires $|1 + h lambda| <= 1$, a disk of radius 1 — so a fast-decaying mode with $lambda = -10^6$ forces $h < 2 times 10^(-6)$ forever, even after that mode is dead and contributes nothing to the solution. The telltale symptom: an explicit adaptive solver grinding along at a tiny step size while the solution looks perfectly smooth.

*A-stable* methods contain the entire left half-plane in their stability region; *L-stable* methods additionally damp the stiffest modes completely as $h lambda arrow -infinity$ (backward Euler does; trapezoidal does not, leaving slowly decaying oscillations on very stiff components). Dahlquist's second barrier: no A-stable linear multistep method exceeds order 2 — implicit RK methods (Radau IIA, order 5, L-stable) evade it. Implicit methods pay for their stability with a nonlinear solve per step, by Newton's method on $y_(n+1) - h beta f(t_(n+1), y_(n+1)) - "known" = 0$, requiring the Jacobian $partial f \/ partial y$ and a factorization — reused across steps and Newton iterations until convergence degrades. For large systems the linear solves dominate; supply analytic Jacobian sparsity, or use Krylov solvers inside Newton (SUNDIALS' CVODE does this at scale).

Cheap middle grounds: *Rosenbrock* methods (linearly implicit — one linear solve per stage, no Newton iteration) for moderate dimensions, and `LSODA`'s automatic stiff/nonstiff switching when you do not know which regime you are in.

== Multistep Methods

Where RK methods discard history, multistep methods reuse it: *Adams-Bashforth* (explicit) and *Adams-Moulton* (implicit) integrate a polynomial through past values of $f$, achieving order $k$ with a *single* new $f$ evaluation per step — the cheapest high-order option when $f$ is expensive and the solution is smooth. The price: a startup procedure, awkward step-size changes (variable-coefficient or interpolation-based restarts), and bad behavior on discontinuities, where the polynomial history is poison.

For stiff problems, *BDF* (backward differentiation formulas) differentiate the interpolating polynomial instead: BDF1 is backward Euler, BDF2 is A-stable, BDF3-6 are A($alpha$)-stable with shrinking wedges (BDF6 barely usable, BDF7 zero-unstable). Variable-order variable-step BDF is the engine of `CVODE`, `ode15s`, and SPICE circuit simulators. Radau is often more robust than high-order BDF on highly oscillatory-stiff problems; BDF wins on cost when the Jacobian is expensive.

== Symplectic Integration and Long-Time Behavior

Integrating the solar system for a billion years, or a molecular dynamics ensemble for $10^9$ steps, changes the question: pointwise accuracy is hopeless (trajectories are chaotic), but *statistical and geometric* fidelity is achievable. Hamiltonian flows preserve the symplectic form; RK4 does not, and its energy error drifts linearly with time — orbits spiral. A *symplectic* integrator preserves the form exactly, and backward error analysis shows it solves a *modified Hamiltonian* exactly (up to exponentially small terms): energy error stays bounded, oscillating around the truth, for exponentially long times.

The workhorses: *semi-implicit (symplectic) Euler* — update momentum with the old position, then position with the new momentum — order 1; *Stoermer-Verlet (leapfrog)*, its symmetric composition, order 2, explicit for separable Hamiltonians, time-reversible, and the universal choice in molecular dynamics; higher-order compositions (Yoshida, 1990) via fractional-step concatenation. Caveats: classic symplectic methods need *fixed* step sizes — naive adaptivity breaks the symplectic structure and resurrects the drift (time-transformation tricks repair it) — and dissipative systems do not want symplecticity at all.

== Pitfalls

- *Using an explicit solver on a stiff problem.* It will not fail loudly; it will succeed slowly, taking millions of micro-steps. If `RK45` crawls while the solution looks smooth, switch to `Radau`, `BDF`, or `LSODA`.
- *Trusting `rtol` as a global error bound.* Tolerances control per-step error; global error compounds. Validate with a tolerance sweep.
- *Integrating across discontinuities.* Events (impacts, switches, control discontinuities) inside a step destroy the smoothness every method assumes; the controller responds by shrinking $h$ to the roundoff floor. Use event detection and restart the integrator at the discontinuity.
- *Energy drift in long Hamiltonian runs.* A higher-order non-symplectic method loses to leapfrog over long horizons. Match the integrator to the structure, not just the order.
- *Step sizes below the roundoff floor.* Below $h approx sqrt(u)$ (for order checks by differencing) or when $t + h$ rounds to $t$, "smaller $h$" increases error. Tight tolerances on long intervals can silently hit this wall.

== Further Reading

Hairer, E., Nørsett, S., Wanner, G. (1993). _Solving Ordinary Differential Equations I: Nonstiff Problems_, 2nd ed. Springer.

Hairer, E., Wanner, G. (1996). _Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems_, 2nd ed. Springer.

Hairer, E., Lubich, C., Wanner, G. (2006). _Geometric Numerical Integration_, 2nd ed. Springer.

Ascher, U., Petzold, L. (1998). _Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations_. SIAM.

Dormand, J., Prince, P. (1980). "A Family of Embedded Runge-Kutta Formulae." J. Comput. Appl. Math.

Shampine, L., Reichelt, M. (1997). "The MATLAB ODE Suite." SIAM J. Sci. Comput.

Hindmarsh, A. et al. (2005). "SUNDIALS: Suite of Nonlinear and Differential/Algebraic Equation Solvers." ACM TOMS.

Leimkuhler, B., Reich, S. (2004). _Simulating Hamiltonian Dynamics_. Cambridge.

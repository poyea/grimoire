= Optimization Algorithms

Minimize $f(x)$ over $x in RR^n$ — the problem behind regression, maximum likelihood, neural network training, optimal control, and design. The continuous optimization toolkit divides along two axes: how much derivative information is available (values only, gradients, Hessians), and how the global step is controlled (line search versus trust region). This chapter covers optimality conditions, line search, gradient descent and its conditioning, Newton and quasi-Newton methods, nonlinear least squares, constrained optimization via KKT, and the stochastic methods that dominate machine learning.

*See also:* _Linear Systems_ (every Newton step is one), _Iterative Methods_ (CG as a quadratic minimizer, and as the inner solver in Newton-CG), _Eigenvalue Problems_ (Hessian spectra decide conditioning), _Error Analysis_ (finite-difference gradients and their precision floor).

== Optimality Conditions and the Lay of the Land

For smooth unconstrained $f$: first order, $nabla f(x^*) = 0$; second order, $nabla^2 f(x^*)$ positive semidefinite (sufficient: positive definite). These are *local* statements — for nonconvex $f$, every method here finds stationary points, and certifying global optimality is NP-hard in general. Convexity is the great exception: local implies global, and duality certifies it. Practical taxonomy:

#table(
  columns: 3,
  [*Available*], [*Method family*], [*Local rate*],
  [$f$ only], [Nelder-Mead, CMA-ES, Bayesian opt.], [Slow; dimensions $lt.tilde 20$ (NM)],
  [$f$, $nabla f$], [Gradient descent, CG, L-BFGS], [Linear (rate depends on $kappa$)],
  [$f$, $nabla f$, $nabla^2 f$], [Newton, trust-region Newton], [Quadratic near the solution],
  [Residual structure], [Gauss-Newton, Levenberg-Marquardt], [Near-quadratic for small residuals],
)

With automatic differentiation now ubiquitous, "gradient unavailable" is rarer than it used to be; reach for derivative-free methods only when the objective is a black-box simulation or is genuinely nonsmooth and noisy.

== Line Search and Trust Regions

Given a descent direction $p_k$, a *line search* picks the step length $alpha$. Exact minimization along the ray is wasted effort; the *Wolfe conditions* — sufficient decrease, $f(x_k + alpha p_k) <= f(x_k) + c_1 alpha nabla f_k^top p_k$, plus a curvature condition ruling out steps that are too short — guarantee convergence (Zoutendijk) and, crucially, keep quasi-Newton updates well-defined. Backtracking from $alpha = 1$ with the Armijo condition alone is the simple robust default; always try $alpha = 1$ first so Newton-type methods can achieve their natural fast local convergence.

*Trust regions* invert the logic: fix a radius $Delta_k$, minimize the local quadratic model within it, and grow or shrink $Delta_k$ by comparing predicted to actual reduction. Trust regions handle indefinite Hessians gracefully (the constrained subproblem is well-posed even when the model is unbounded below) and are the standard frame for robust Newton and Levenberg-Marquardt implementations.

== Gradient Descent and Conditioning

Steepest descent, $x_(k+1) = x_k - alpha nabla f(x_k)$, converges linearly on strongly convex quadratics at rate $((kappa - 1) \/ (kappa + 1))^2$ per step in function value, where $kappa = lambda_max \/ lambda_min$ of the Hessian: the iterates zigzag across the narrow valley. With $kappa = 10^4$, expect tens of thousands of iterations. *Momentum* (heavy ball) and *Nesterov acceleration* improve the dependence from $kappa$ to $sqrt(kappa)$ — the same square-root that preconditioned CG buys, and provably optimal for first-order methods on this class (Nemirovski-Yudin lower bound). The deeper lesson is that conditioning is everything for first-order methods: rescaling variables, whitening inputs, and batch normalization are all preconditioning by other names.

== Newton and Quasi-Newton Methods

Newton's method minimizes the local quadratic model: solve $nabla^2 f(x_k) p_k = -nabla f(x_k)$, step. Near a minimizer with Lipschitz Hessian, convergence is quadratic — digits double per iteration, machine precision in 5-6 steps. Far from one, the raw method is treacherous: the Hessian may be indefinite (the "Newton" direction can point uphill, toward a saddle) and the unit step can diverge. Globalization — Wolfe line search on a modified-Hessian direction, or a trust region — is mandatory. Each step costs a Hessian and an $O(n^3)$ factorization; *Newton-CG* (truncated Newton) solves the system inexactly by CG using only Hessian-vector products (cheap via AD: one forward-over-reverse pass), bailing out along directions of negative curvature.

*Quasi-Newton* methods earn near-Newton convergence from gradients alone by accumulating curvature from observed gradient differences. BFGS updates an inverse-Hessian approximation $H_k$ with the rank-2 formula

$ H_(k+1) = (I - rho_k s_k y_k^top) H_k (I - rho_k y_k s_k^top) + rho_k s_k s_k^top, quad rho_k = 1 \/ (y_k^top s_k), $

where $s_k = x_(k+1) - x_k$ and $y_k = nabla f_(k+1) - nabla f_k$; the Wolfe curvature condition guarantees $y_k^top s_k > 0$, keeping $H_k$ positive definite. Convergence is superlinear. *L-BFGS* stores only the last $m approx 5$-$20$ pairs $(s, y)$ and applies $H_k$ implicitly by the two-loop recursion in $O(m n)$ time and memory — the default for smooth high-dimensional problems (`scipy.optimize.minimize(method="L-BFGS-B")`, which also handles bound constraints).

== Nonlinear Least Squares

For $f(x) = 1/2 sum_i r_i (x)^2$ with residuals $r: RR^n arrow RR^m$, the structure is too valuable to ignore: $nabla f = J^top r$ and $nabla^2 f = J^top J + sum_i r_i nabla^2 r_i$. *Gauss-Newton* drops the second term, solving $J^top J p = -J^top r$ — equivalently, by QR on the linearized residual, which avoids squaring the condition number. Near a solution with small residuals the dropped term is negligible and convergence is nearly quadratic, using first derivatives only.

*Levenberg-Marquardt* regularizes with a damping parameter: $(J^top J + lambda "diag"(J^top J)) p = -J^top r$, interpolating between Gauss-Newton ($lambda arrow 0$) and scaled gradient descent ($lambda$ large), with $lambda$ adapted trust-region-style. It is the standard for curve fitting (`scipy.optimize.least_squares`, MINPACK's `lmder`) and the backbone of bundle adjustment in computer vision (Ceres Solver), where sparse Schur-complement tricks scale it to millions of parameters. Large-residual problems break the Gauss-Newton approximation; switch to a full quasi-Newton method or hybrid.

== Constrained Optimization

Minimize $f$ subject to $c_i (x) = 0$ and $c_j (x) >= 0$. The KKT conditions couple primal feasibility with a Lagrangian stationarity condition $nabla f = sum_i lambda_i nabla c_i$ and complementary slackness (inactive constraints get zero multipliers). The multipliers are not bookkeeping — they are shadow prices, the sensitivity of the optimum to constraint perturbations.

- *Active-set / SQP*: solve a sequence of quadratic programs modeling the Lagrangian; the QP's KKT system is a symmetric indefinite linear system ($L D L^top$ with Bunch-Kaufman — see _Linear Systems_). Excellent warm-starting; SNOPT and `SLSQP` live here.
- *Interior-point*: replace inequalities with a log-barrier $-mu sum log c_j (x)$, follow the central path as $mu arrow 0$, taking Newton steps on the perturbed KKT system. Polynomial-time for convex problems, ruthlessly effective in practice: IPOPT (nonconvex NLP), Mosek and Clarabel (conic), and modern LP solvers.
- *Augmented Lagrangian*: add both a multiplier term and a quadratic penalty $rho/2 parallel c parallel^2$; alternate minimization with multiplier updates. Unlike the pure quadratic penalty, $rho$ need not go to infinity, so the subproblems stay well-conditioned. ADMM is its operator-splitting descendant.
- *Projected gradient / proximal methods*: when the constraint set or a nonsmooth term has a cheap projection or proximal operator (box, simplex, $ell_1$), first-order methods scale far beyond general NLP solvers — ISTA/FISTA for lasso are the canonical examples.

== Stochastic Gradient Methods

When $f(x) = EE[ell(x, xi)]$ over a data distribution, exact gradients cost a full pass over the data. SGD steps along a minibatch estimate; the noise forces diminishing or small constant steps, and convergence is sublinear — $O(1\/k)$ for strongly convex problems — yet per-step cost independent of dataset size wins decisively at scale. Variance reduction (SVRG, SAGA) recovers linear rates for finite sums; in deep learning, diagonal-preconditioning methods — Adam, with bias-corrected first and second moment estimates — are the de facto standard, trading some asymptotic accuracy for robustness to wildly varying curvature scales. The classical kit is not displaced: L-BFGS still rules full-batch smooth problems, and the stochastic and batch worlds meet in second-order stochastic methods (K-FAC, Shampoo) that remain an active frontier.

== Pitfalls

- *Finite-difference gradients.* Forward differences with step $h$ incur $O(h)$ truncation plus $O(u \/ h)$ roundoff error; the optimum $h approx sqrt(u)$ yields only half machine precision (central differences: $u^(1\/3)$ step, two-thirds precision). Use automatic differentiation when at all possible — and *check* hand-coded gradients against finite differences before trusting an optimizer's strange behavior.
- *Declaring victory on a small gradient.* $parallel nabla f parallel < 10^(-6)$ also holds near saddle points and on flat plateaus; in ill-conditioned problems the *solution* can still be far away (the gradient norm is the residual of $nabla f = 0$ — same residual-versus-error gap as in linear systems).
- *Ignoring scaling.* Variables of magnitudes $10^(-3)$ and $10^6$ in the same problem cripple gradient methods and finite-difference steps alike. Nondimensionalize; most solver "failures" are scaling failures.
- *Unit step skipped.* A line search that never tries $alpha = 1$ silently downgrades Newton and BFGS to linear convergence.
- *Multistart neglected.* For nonconvex problems, a single run from one initial point samples one basin. Cheap insurance: random restarts; report the spread, not just the best.

== Further Reading

Nocedal, J., Wright, S. (2006). _Numerical Optimization_, 2nd ed. Springer. The reference for this chapter.

Boyd, S., Vandenberghe, L. (2004). _Convex Optimization_. Cambridge.

Dennis, J., Schnabel, R. (1996). _Numerical Methods for Unconstrained Optimization and Nonlinear Equations_. SIAM.

Conn, A., Gould, N., Toint, P. (2000). _Trust-Region Methods_. SIAM.

Liu, D., Nocedal, J. (1989). "On the Limited Memory BFGS Method for Large Scale Optimization." Math. Programming.

Bottou, L., Curtis, F., Nocedal, J. (2018). "Optimization Methods for Large-Scale Machine Learning." SIAM Review.

Kingma, D., Ba, J. (2015). "Adam: A Method for Stochastic Optimization." ICLR.

Wächter, A., Biegler, L. (2006). "On the Implementation of an Interior-Point Filter Line-Search Algorithm for Large-Scale Nonlinear Programming." Math. Programming.

Nesterov, Y. (2018). _Lectures on Convex Optimization_, 2nd ed. Springer.

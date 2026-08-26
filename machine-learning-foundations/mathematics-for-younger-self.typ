#import "../template.typ": overbar, xref

= Mathematics for a Younger Self

Eight subjects, ordered by the machinery each presupposes. Definitions are stated to be used, theorems to be recognised, and each section carries one derivation showing where the content actually comes from.

*See also:* #xref("machine-learning-foundations", "linear-algebra-for-ml", label: "Linear Algebra for ML") (the same objects, aimed at data), #xref("machine-learning-foundations", "probability-and-information", label: "Probability and Information") (the probabilistic half, applied), #xref("machine-learning-foundations", "notation", label: "Notation and Conventions") (symbols used throughout), #xref("numerical-computing", "error-analysis", label: "Error Analysis") (what survives finite precision).

== Linear Algebra

*Setting.* $V$ a vector space over $k$; $T : V arrow.r W$ linear. Fixing bases makes $T$ a matrix; changing basis conjugates, $A |-> P^(-1) A P$. The invariants of conjugation — rank, trace, determinant, characteristic polynomial, spectrum — are the subject.

*Rank-nullity.* $dim V = dim ker T + dim "im" T$.

*Spectral theorem.* $A = A^*$ on a finite-dimensional complex inner-product space $arrow.r.double$ $A = U Lambda U^*$ with $U$ unitary and $Lambda$ real diagonal.

*Cayley-Hamilton.* $p_A (A) = 0$ for $p_A (t) = det(t I - A)$.

*Singular value decomposition.* Every $A in CC^(m times n)$ factors $A = U Sigma V^*$, $U, V$ unitary, $Sigma$ diagonal with $sigma_1 >= dots >= sigma_r > 0$.

*Derivation (SVD from the spectral theorem).* $A^* A$ is Hermitian positive semidefinite, so $A^* A = V Lambda V^*$ with $lambda_i >= 0$. Put $sigma_i = sqrt(lambda_i)$ and, for $sigma_i > 0$, $u_i = A v_i \/ sigma_i$. Then

$ chevron.l u_i, u_j chevron.r = (chevron.l A v_i, A v_j chevron.r) / (sigma_i sigma_j) = (chevron.l v_i, A^* A v_j chevron.r) / (sigma_i sigma_j) = (lambda_j delta_(i j)) / (sigma_i sigma_j) = delta_(i j), $

so the $u_i$ are orthonormal and $A v_i = sigma_i u_i$ is exactly $A V = U Sigma$. Singular values are the square roots of the eigenvalues of $A^* A$; this is why forming $A^* A$ numerically squares the condition number.

== Abstract Algebra

*Setting.* Groups (symmetry), rings (arithmetic), fields (division), modules (vector spaces over rings). Structure is read off from quotients: normal subgroups and ideals are exactly the kernels.

*First isomorphism theorem.* $G \/ ker phi tilde.equiv "im" phi$.

*Lagrange.* $H <= G$ finite $arrow.r.double$ $|H|$ divides $|G|$.

*Orbit-stabilizer.* $|G| = |G x| dot |"Stab"(x)|$ for a finite group acting on $x$.

*Sylow.* If $|G| = p^n m$ with $p divides.not m$, then Sylow $p$-subgroups exist, are conjugate, and their number $n_p$ satisfies $n_p equiv 1 mod p$ and $n_p divides m$.

*Galois correspondence.* For $L \/ K$ finite Galois, $H |-> L^H$ is an inclusion-reversing bijection between subgroups of $"Gal"(L\/K)$ and intermediate fields, with $[L : L^H] = |H|$. A polynomial is solvable by radicals iff its Galois group is solvable.

*Structure theorem over a PID.* A finitely generated module over a PID $R$ satisfies

$ M tilde.equiv R^r plus.o R\/(d_1) plus.o dots plus.o R\/(d_s), quad d_1 divides d_2 divides dots divides d_s. $

*Derivation (Jordan form as a corollary).* Let $T$ act on a finite-dimensional $V$ over algebraically closed $k$, and make $V$ a $k[x]$-module by $x dot v = T v$. Since $k[x]$ is a PID and $V$ is finitely generated and torsion, the theorem gives $V tilde.equiv plus.o.big_i k[x] \/ ((x - lambda_i)^(m_i))$. On $k[x]\/((x - lambda)^m)$ the basis $1, (x - lambda), dots, (x - lambda)^(m-1)$ makes multiplication by $x$ a single Jordan block. Taking $R = ZZ$ instead classifies finite abelian groups: one theorem, two famous corollaries.

== Real Analysis: Measure, Integration, Hilbert Spaces

*Setting.* A measure $mu$ is countably additive on a $sigma$-algebra. For measurable $f >= 0$, $integral f d mu := sup { integral s d mu : s "simple", 0 <= s <= f }$. The point is that limits now pass through the integral.

*Monotone convergence.* $0 <= f_n arrow.t f$ pointwise $arrow.r.double$ $integral f_n arrow.r integral f$.

*Fatou.* $integral liminf f_n <= liminf integral f_n$ for $f_n >= 0$.

*Dominated convergence.* $f_n arrow.r f$ a.e. with $|f_n| <= g in L^1$ $arrow.r.double$ $integral f_n arrow.r integral f$.

*Derivation (all three are one).* Set $g_n = inf_(k >= n) f_k$. Then $0 <= g_n arrow.t liminf f_n$, so monotone convergence gives $integral liminf f_n = lim integral g_n$, and $g_n <= f_n$ gives $lim integral g_n <= liminf integral f_n$ — that is Fatou. For domination, apply Fatou to the non-negative sequences $g + f_n$ and $g - f_n$:

$ integral g + integral f <= integral g + liminf integral f_n, quad integral g - integral f <= integral g - limsup integral f_n, $

so $limsup integral f_n <= integral f <= liminf integral f_n$, forcing equality.

*Hölder.* $parallel f g parallel_1 <= parallel f parallel_p parallel g parallel_q$ for $1\/p + 1\/q = 1$.

*Riesz-Fischer.* $L^p (mu)$ is complete for $1 <= p <= infinity$.

*Projection.* $C$ closed convex in a Hilbert space $H$, $x in H$ $arrow.r.double$ there is a unique nearest point of $C$ to $x$.

*Riesz representation.* Every $phi in H^*$ is $phi(x) = chevron.l x, y chevron.r$ for a unique $y$, and $parallel phi parallel = parallel y parallel$.

== Fourier Analysis

*Setting.* $hat(f)(xi) = integral_(RR) f(x) e^(-2 pi i x xi) d x$. The characters $e^(2 pi i x xi)$ are the eigenfunctions of translation, so the transform diagonalizes every translation-invariant operator.

*Convolution.* $hat(f * g) = hat(f) hat(g)$.

*Derivation.* By Fubini, admissible for $f, g in L^1$,

$ hat(f * g)(xi) = integral integral f(y) g(x - y) e^(-2 pi i x xi) d y d x = integral f(y) e^(-2 pi i y xi) (integral g(u) e^(-2 pi i u xi) d u) d y, $

substituting $u = x - y$. Differentiation transforms the same way: $hat(f')(xi) = 2 pi i xi hat(f)(xi)$, turning constant-coefficient differential equations into algebra.

*Fejér.* The Dirichlet kernel has $parallel D_N parallel_1 tilde log N$, so partial sums may diverge for continuous $f$; the Cesàro means $sigma_N f = f * F_N$ use the Fejér kernel, which is a genuine approximate identity, and $sigma_N f arrow.r f$ uniformly for $f$ continuous on $TT$.

*Plancherel.* $parallel hat(f) parallel_2 = parallel f parallel_2$; the transform extends from $L^1 inter L^2$ to a unitary map of $L^2 (RR)$.

*Inversion.* $f, hat(f) in L^1$ $arrow.r.double$ $f(x) = integral hat(f)(xi) e^(2 pi i x xi) d xi$ a.e.

*Poisson summation.* For Schwartz $f$, $sum_(n in ZZ) f(n) = sum_(n in ZZ) hat(f)(n)$.

*Derivation.* Periodize: $F(x) = sum_n f(x + n)$ is $1$-periodic, and its $n$-th Fourier coefficient is $integral_0^1 F(x) e^(-2 pi i n x) d x = integral_RR f(x) e^(-2 pi i n x) d x = hat(f)(n)$. Evaluating the Fourier series of $F$ at $x = 0$ gives the identity.

*Uncertainty.* $parallel x f parallel_2 dot parallel xi hat(f) parallel_2 >= parallel f parallel_2^2 \/ (4 pi)$, with equality only for Gaussians.

== Complex Analysis

*Setting.* $f : Omega arrow.r CC$ complex-differentiable on an open set. Writing $f = u + i v$, differentiability forces the Cauchy-Riemann equations $u_x = v_y$, $u_y = -v_x$, so $u$ and $v$ are harmonic. Differentiable once will turn out to mean analytic.

*Cauchy-Goursat.* $f$ holomorphic on simply connected $Omega$ $arrow.r.double$ $integral_gamma f = 0$ for every closed $gamma subset Omega$.

*Cauchy integral formula.* For $z$ inside $gamma$,

$ f(z) = 1/(2 pi i) integral_gamma (f(w)) / (w - z) d w. $

*Derivation (rigidity).* Differentiating under the integral sign is legitimate because the integrand is smooth in $z$ off the contour, and it gives

$ f^((n))(z) = (n!)/(2 pi i) integral_gamma (f(w)) / ((w - z)^(n+1)) d w. $

So one derivative implies all of them, and expanding $1\/(w - z)$ as a geometric series yields a local power series. Bounding the $n = 1$ case on a circle of radius $R$ with $|f| <= M$ gives $|f'(z)| <= M \/ R$; if $f$ is entire and bounded, let $R arrow.r infinity$ to get $f' equiv 0$ — *Liouville*, and with it the fundamental theorem of algebra.

*Identity theorem.* Two holomorphic functions agreeing on a set with a limit point in $Omega$ agree on $Omega$.

*Maximum modulus.* A non-constant holomorphic $f$ has no interior maximum of $|f|$.

*Residue theorem.* $integral_gamma f = 2 pi i sum_k "Res"_(z_k) f$ over the enclosed isolated singularities.

*Argument principle.* $(1\/2 pi i) integral_gamma f' \/ f = Z - P$, counting zeros and poles with multiplicity; Rouché's theorem follows by continuity of $Z - P$ in a homotopy.

*Riemann mapping.* Every simply connected $Omega subset.neq CC$ is conformally equivalent to the unit disc.

== Probability Theory

*Setting.* $(Omega, cal(F), PP)$ with $PP(Omega) = 1$; a random variable is measurable, $EE[X] = integral X d PP$. Formally this is measure theory; what makes it a distinct subject is *independence*.

*Characteristic function.* $phi_X (t) = EE[e^(i t X)]$ — the Fourier transform of the law. Independence makes it multiplicative: $phi_(X + Y) = phi_X phi_Y$.

*Strong law.* $X_i$ i.i.d. with $EE|X_1| < infinity$ $arrow.r.double$ $S_n \/ n arrow.r EE[X_1]$ almost surely.

*Central limit theorem.* $EE[X_1] = mu$, $"Var"(X_1) = sigma^2 < infinity$ $arrow.r.double$ $(S_n - n mu) \/ (sigma sqrt(n))$ converges in distribution to $cal(N)(0, 1)$.

*Derivation.* Let $Z_i = (X_i - mu) \/ sigma$, so $EE[Z] = 0$, $EE[Z^2] = 1$. Two moments give the expansion $phi_Z (u) = 1 - u^2 \/ 2 + o(u^2)$. By independence,

$ phi_(S_n^*) (t) = [phi_Z (t \/ sqrt(n))]^n = [1 - t^2/(2 n) + o(1\/n)]^n arrow.r e^(-t^2 \/ 2), $

which is the characteristic function of $cal(N)(0,1)$; Lévy's continuity theorem upgrades pointwise convergence of $phi$ to convergence in distribution. The Gaussian appears because it is the fixed point of this quadratic truncation, not because of anything about $X$.

*Conditional expectation.* $EE[X | cal(G)]$ is the a.s. unique $cal(G)$-measurable $Y$ with $integral_G Y d PP = integral_G X d PP$ for all $G in cal(G)$; it exists by Radon-Nikodym and is the $L^2$ projection onto $L^2 (cal(G))$.

*Martingales.* $EE[X_(n+1) | cal(F)_n] = X_n$. Optional stopping: for a bounded stopping time $tau$, $EE[X_tau] = EE[X_0]$. Convergence: an $L^1$-bounded martingale converges almost surely.

== Functional Analysis

*Setting.* Complete normed spaces and the bounded operators between them. Completeness plus Baire category yields the structural theorems.

*Hahn-Banach.* A bounded functional on a subspace extends to the whole space with the same norm; hence $X^*$ separates points.

*Uniform boundedness.* $sup_alpha parallel T_alpha x parallel < infinity$ for each $x$ $arrow.r.double$ $sup_alpha parallel T_alpha parallel < infinity$.

*Derivation.* Let $E_n = {x : sup_alpha parallel T_alpha x parallel <= n}$. Each $E_n$ is closed and $union_n E_n = X$, so by Baire some $E_N$ contains a ball $B(x_0, r)$. For $parallel y parallel <= 1$, writing $r y = (x_0 + r y) - x_0$ gives $parallel T_alpha y parallel <= 2 N \/ r$ uniformly in $alpha$.

*Open mapping and closed graph.* A bounded surjection of Banach spaces is open, hence a bounded bijection has bounded inverse; a linear map with closed graph is bounded.

*Banach-Alaoglu.* The closed unit ball of $X^*$ is weak-star compact. In infinite dimensions the norm-closed ball is never compact, so existence proofs extract weak-star limits instead.

*Spectral theorem, compact self-adjoint case.* $T = T^*$ compact on a Hilbert space $arrow.r.double$ there is an orthonormal basis of eigenvectors with real eigenvalues accumulating only at $0$. The *Fredholm alternative* follows: $T x - lambda x = y$ is solvable for all $y$ exactly when the homogeneous equation has only the trivial solution.

== Stochastic Calculus

*Setting.* Brownian motion $B_t$: continuous paths, $B_0 = 0$, independent increments, $B_t - B_s tilde cal(N)(0, t - s)$. Paths are almost surely nowhere differentiable.

*Quadratic variation.* For partitions of $[0, t]$ with mesh $arrow.r 0$, $sum_i (B_(t_(i+1)) - B_(t_i))^2 arrow.r t$ in $L^2$.

*Derivation.* Write $Delta_i = B_(t_(i+1)) - B_(t_i)$, so $EE[Delta_i^2] = Delta t_i$ and, by Gaussianity, $"Var"(Delta_i^2) = 2 (Delta t_i)^2$. The increments are independent, so

$ EE[(sum_i Delta_i^2 - t)^2] = sum_i "Var"(Delta_i^2) = 2 sum_i (Delta t_i)^2 <= 2 t max_i Delta t_i arrow.r 0. $

A differentiable path would give $0$ here. Nonzero quadratic variation is why $integral H d B$ cannot be a pathwise Stieltjes integral, and why second-order terms survive.

*Itô integral.* Defined on predictable $H$ with $EE integral_0^T H_s^2 d s < infinity$ as the $L^2$ limit of sums over simple integrands, characterised by the isometry

$ EE[(integral_0^T H_s d B_s)^2] = EE[integral_0^T H_s^2 d s]. $

Predictability — the integrand may not anticipate the increment it multiplies — is exactly what makes the integral a martingale.

*Itô's formula.* For $f in C^(1,2)$,

$ d f(t, B_t) = (partial_t f + 1/2 partial_(x x) f) d t + partial_x f d B_t. $

*Derivation.* Taylor to second order gives $Delta f approx partial_t f Delta t + partial_x f Delta B + 1/2 partial_(x x) f (Delta B)^2$. Ordinary calculus discards $(Delta B)^2$; here it converges to $Delta t$, so the term promotes to first order. The whole subject is that one correction.

*Girsanov.* Changing measure by an exponential martingale shifts the drift and leaves the quadratic variation fixed — the basis of risk-neutral pricing.

*Feynman-Kac.* $u(t, x) = EE[phi(X_T) | X_t = x]$ solves the associated parabolic PDE: diffusion described analytically and probabilistically are the same object.

== How the Pieces Fit

Measure theory is the substrate: probability is measure theory plus independence, $L^2$ is where Fourier becomes an isometry, and functional analysis is linear algebra with the finite dimension removed and completeness put in its place. Complex analysis stands apart, trading a very strong hypothesis for extraordinary rigidity. Stochastic calculus comes last because it uses all of them at once.

== Further Reading

Axler, S. (2024). _Linear Algebra Done Right_, 4th ed. Springer. (Determinant-free; the spectral theorem earned rather than assumed.)

Halmos, P. R. (1958). _Finite-Dimensional Vector Spaces_, 2nd ed. Van Nostrand. (Written so the infinite-dimensional case feels inevitable afterwards.)

Dummit, D. S., & Foote, R. M. (2004). _Abstract Algebra_, 3rd ed. Wiley. (Standard first graduate course, Galois theory included.)

Lang, S. (2002). _Algebra_, revised 3rd ed. Springer. (Terse and complete; a reference rather than a first pass.)

Stein, E. M., & Shakarchi, R. (2003). _Fourier Analysis: An Introduction_. Princeton University Press. (Princeton Lectures in Analysis I.)

Stein, E. M., & Shakarchi, R. (2003). _Complex Analysis_. Princeton University Press. (Princeton Lectures in Analysis II.)

Stein, E. M., & Shakarchi, R. (2005). _Real Analysis: Measure Theory, Integration, and Hilbert Spaces_. Princeton University Press. (Princeton Lectures in Analysis III.)

Stein, E. M., & Shakarchi, R. (2011). _Functional Analysis: Introduction to Further Topics in Analysis_. Princeton University Press. (Princeton Lectures in Analysis IV.)

Rudin, W. (1987). _Real and Complex Analysis_, 3rd ed. McGraw-Hill. (Measure theory and complex analysis in one austere volume.)

Folland, G. B. (1999). _Real Analysis: Modern Techniques and Their Applications_, 2nd ed. Wiley. (Measure, distributions, and Fourier analysis together.)

Durrett, R. (2019). _Probability: Theory and Examples_, 5th ed. Cambridge University Press. (Measure-theoretic probability with the examples kept in view.)

Williams, D. (1991). _Probability with Martingales_. Cambridge University Press. (The shortest honest route to conditional expectation and martingale convergence.)

Øksendal, B. (2003). _Stochastic Differential Equations: An Introduction with Applications_, 6th ed. Springer. (The readable entry point to Itô calculus.)

Karatzas, I., & Shreve, S. E. (1991). _Brownian Motion and Stochastic Calculus_, 2nd ed. Springer. (The rigorous reference behind it.)

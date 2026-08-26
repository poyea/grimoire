#import "../template.typ": overbar, xref

= Mathematics for a Younger Self

What follows is the shape of eight subjects, ordered by how much machinery each one presupposes. Every section states the central object, the theorem that does the work, and the single idea that makes the rest follow. Nothing here is a substitute for a proof; the aim is that a definition, once read, stays put.

*See also:* #xref("machine-learning-foundations", "linear-algebra-for-ml", label: "Linear Algebra for ML") (the same objects, aimed at data), #xref("machine-learning-foundations", "probability-and-information", label: "Probability and Information") (the probabilistic half, applied), #xref("machine-learning-foundations", "notation", label: "Notation and Conventions") (symbols used throughout), #xref("numerical-computing", "error-analysis", label: "Error Analysis") (what survives finite precision).

== Linear Algebra

*Object.* A vector space $V$ over a field $k$, and the linear maps between vector spaces. Once bases are fixed a linear map becomes a matrix; changing basis conjugates it, $A |-> P^(-1) A P$. Linear algebra is therefore the study of what does *not* change under conjugation: rank, trace, determinant, characteristic polynomial, spectrum.

*The three structure theorems.* Over $CC$, every operator has a Jordan form: it is diagonalizable up to nilpotent debris on the diagonal blocks. Every self-adjoint operator on a finite-dimensional inner-product space is orthogonally diagonalizable with real eigenvalues (*spectral theorem*) — the reason covariance matrices, Hessians, and graph Laplacians are tractable. Every matrix whatsoever, square or not, factors as $A = U Sigma V^*$ with $U, V$ unitary and $Sigma$ diagonal and non-negative (*SVD*): every linear map is a rotation, an axis-aligned scaling, and another rotation.

*The idea.* Duality. To each space $V$ attach $V^* = {phi : V arrow.r k "linear"}$; to each map $T : V arrow.r W$ attach $T^* : W^* arrow.r V^*$ running backwards. Rank-nullity, the four fundamental subspaces, and the transpose are all one statement seen from different sides. Finite dimension is what makes $V tilde.equiv V^*$; everything that later goes wrong in infinite dimensions goes wrong here first.

== Abstract Algebra

*Object.* Sets with operations, studied through the maps that preserve them. A *group* axiomatizes symmetry; a *ring* axiomatizes addition-and-multiplication; a *field* is a ring where division works; a *module* is a vector space whose scalars form a ring rather than a field.

*Mechanism.* Quotient by what you wish to ignore. Normal subgroups give quotient groups, ideals give quotient rings, and in each case the isomorphism theorems say that a map's image is the domain modulo its kernel. Lagrange bounds subgroup order by divisibility; the orbit-stabilizer theorem converts counting problems into group actions; Sylow's theorems reconstruct a finite group's possible internal structure from the prime factorization of its order alone.

*The theorem.* The *Galois correspondence*: for a finite Galois extension $L\/K$, the subgroups of $"Gal"(L\/K)$ correspond, inclusion-reversingly, to the intermediate fields. A polynomial is solvable by radicals exactly when its Galois group is solvable — which is why the quintic has no formula. Symmetry of the roots controls what arithmetic can express.

*Unification.* The structure theorem for finitely generated modules over a principal ideal domain has two famous corollaries, obtained by choosing the ring: over $ZZ$ it classifies finite abelian groups; over $k[x]$ it produces the Jordan form. Two theorems you learned separately are one theorem.

== Real Analysis: Measure, Integration, Hilbert Spaces

*Problem.* The Riemann integral does not survive limits: a pointwise limit of Riemann-integrable functions need not be integrable, and $lim integral$ need not equal $integral lim$. Analysis cannot be built on an integral that breaks under the operation analysis is about.

*Object.* A *measure* is a countably additive function $mu$ on a $sigma$-algebra of sets. Lebesgue measure is the completion of "length" to the largest class where countable additivity survives; non-measurable sets exist, but only via the axiom of choice, which is the price of asking for too much. Integration is then defined for measurable functions by approximation from below by simple functions.

*The three limit theorems.* Monotone convergence, Fatou's lemma, and dominated convergence. Together they say: under a monotonicity or domination hypothesis, limits pass through integrals. Essentially every later analytic argument is one of these three wearing a disguise.

*Geometry.* The $L^p$ spaces are complete (Riesz-Fischer), and $L^2$ is a *Hilbert space*: it has an inner product, hence orthogonality, orthonormal bases, and the projection theorem — the nearest point of a closed convex set exists and is unique. The Riesz representation theorem says every bounded linear functional on a Hilbert space is an inner product against a fixed vector, so $H tilde.equiv H^*$. Hilbert space is the one infinite-dimensional setting where finite-dimensional intuition remains reliable.

== Fourier Analysis

*Object.* Decompose a function into characters $e^(2 pi i xi x)$ — the eigenfunctions of translation. On the circle this is Fourier series; on $RR$, the Fourier transform $hat(f)(xi) = integral f(x) e^(-2 pi i x xi) d x$; on a finite abelian group, the DFT.

*Why characters.* The transform diagonalizes every translation-invariant operator. Convolution becomes multiplication, $hat(f * g) = hat(f) hat(g)$; differentiation becomes multiplication by $2 pi i xi$. A constant-coefficient differential equation becomes an algebraic one.

*Convergence is delicate.* The Dirichlet kernel is not an approximate identity — its $L^1$ norms diverge logarithmically — so partial sums of a Fourier series need not converge pointwise for a merely continuous function. Cesàro averaging repairs this: the Fejér kernel *is* an approximate identity, and Fejér's theorem gives uniform convergence for continuous $f$. In $L^2$ there is no difficulty at all: Plancherel says the transform is unitary, $parallel hat(f) parallel_2 = parallel f parallel_2$, and the exponentials form an orthonormal basis.

*Two consequences worth memorizing.* The uncertainty principle: $f$ and $hat(f)$ cannot both be sharply localized, with the Gaussian as the extremal case. Poisson summation: $sum_(n in ZZ) f(n) = sum_(n in ZZ) hat(f)(n)$, which converts questions about lattices into questions about their duals.

== Complex Analysis

*Object.* A function $f : Omega arrow.r CC$ that is complex-differentiable at every point of an open set. The definition looks like the real one; the consequences do not.

*Rigidity.* Differentiable once implies differentiable infinitely often, and equal to its own power series locally. This has no real-variable analogue, and it is the whole subject. Cauchy's theorem ($integral_gamma f = 0$ for $f$ holomorphic on a simply connected domain) and the Cauchy integral formula (which recovers $f$ inside a contour from its values on the contour) say that a holomorphic function's local data determines it globally.

*The consequences cascade.* Liouville: a bounded entire function is constant — hence the fundamental theorem of algebra. The identity theorem: two holomorphic functions agreeing on a set with a limit point agree everywhere. Maximum modulus: $|f|$ attains no interior maximum. The open mapping theorem, Rouché's theorem, and the argument principle all follow, and they turn the counting of zeros into the evaluation of an integral.

*Residues.* Isolated singularities are classified as removable, poles, or essential; the residue theorem evaluates contour integrals as a finite sum of local data. Analytic continuation extends a function beyond its original disc, uniquely where it extends at all — the mechanism behind $zeta(s)$ and the Riemann hypothesis. The Riemann mapping theorem says every simply connected proper subdomain of $CC$ is conformally a disc: up to holomorphic change of coordinates, there is only one such domain.

== Probability Theory

*Object.* A probability space $(Omega, cal(F), PP)$ is a measure space with $PP(Omega) = 1$; a random variable is a measurable function; expectation is the integral. Formally, probability is a special case of measure theory. What makes it a separate subject is *independence*, which has no analogue in general measure theory and which generates product measures, the classical limit theorems, and everything after.

*Limit theorems.* Convergence comes in four strengths — almost sure, in probability, in $L^p$, in distribution — and the implications between them are strict. The law of large numbers says sample means converge to the mean; the central limit theorem says the fluctuation around it is Gaussian at scale $sqrt(n)$, whatever the underlying law. The clean proof of the CLT is Fourier analysis: the characteristic function $phi_X (t) = EE[e^(i t X)]$ is the Fourier transform of the law, independence makes it multiplicative, and Lévy's continuity theorem converts pointwise convergence of $phi$ into convergence in distribution.

*Conditioning.* Conditional expectation $EE[X | cal(G)]$ is not a formula but a projection: the $cal(G)$-measurable random variable closest to $X$ in $L^2$, existing in general by Radon-Nikodym. A *martingale* is a process with $EE[X_(n+1) | cal(F)_n] = X_n$ — a fair game. The optional stopping theorem says you cannot beat one by choosing when to quit, and the martingale convergence theorem says a bounded martingale converges almost surely. Most of modern probability is the search for a martingale hiding in the problem.

== Functional Analysis

*Object.* Infinite-dimensional vector spaces with a topology, and the continuous linear maps between them. A *Banach space* is a complete normed space; a Hilbert space is a Banach space whose norm comes from an inner product.

*The big three.* Hahn-Banach: bounded functionals extend from subspaces without increasing norm, so duals are large enough to separate points. The open mapping and closed graph theorems: a continuous bijection of Banach spaces has continuous inverse, and a linear map with closed graph is continuous. Uniform boundedness: a family of operators bounded pointwise is bounded in norm. All three are consequences of completeness by way of Baire's category theorem — completeness is what buys the subject its power.

*Compactness returns, weakly.* The closed unit ball is compact only in finite dimensions. The repair is to weaken the topology: Banach-Alaoglu says the dual unit ball is weak-star compact always. This is why existence proofs in PDE and optimization proceed by extracting weakly convergent subsequences.

*Spectral theory.* For a compact self-adjoint operator the finite-dimensional picture survives intact: a real discrete spectrum accumulating only at $0$, and an orthonormal eigenbasis. In general the spectrum need not consist of eigenvalues at all, and the spectral theorem is stated instead as a projection-valued measure — diagonalization becomes integration. *Distributions* extend the calculus to objects too rough to differentiate, by moving the derivative onto a smooth test function; the Fourier transform then acts on tempered distributions, which is the natural home of the theory.

== Stochastic Calculus

*Object.* Brownian motion $B_t$: the unique continuous process with stationary independent Gaussian increments, $B_t - B_s tilde cal(N)(0, t - s)$. It exists (Wiener), its paths are continuous, and almost surely nowhere differentiable.

*Why ordinary calculus fails.* Brownian paths have finite, nonzero *quadratic variation*: $sum (B_(t_(i+1)) - B_(t_i))^2 arrow.r t$ as the partition refines, whereas a differentiable path would give $0$. Because $(d B)^2 = d t$ rather than a negligible quantity, second-order terms survive into first order, and $integral f d B$ cannot be defined pathwise as a Stieltjes integral. The *Itô integral* is instead built as an $L^2$ limit of sums over *predictable* simple integrands — the integrand must not peek at the increment it multiplies, which is exactly what makes the integral a martingale.

*The formula.* For $f$ twice continuously differentiable,

$ d f(B_t) = f'(B_t) d B_t + 1/2 f''(B_t) d t. $

That second term is the entire subject. It is the chain rule corrected by the quadratic variation, and it produces the Black-Scholes equation, the Fokker-Planck equation, and the link between diffusions and second-order PDEs.

*Three consequences.* Martingale representation: every Brownian martingale is a stochastic integral, so every claim is hedgeable. Girsanov: changing the measure by an exponential martingale changes the drift but not the quadratic variation, which is why risk-neutral pricing works. Feynman-Kac: the solution of a parabolic PDE equals an expectation over paths of an SDE — the analytic and probabilistic descriptions of diffusion are the same description.

== How the Pieces Fit

Measure theory is the substrate: probability is measure theory with independence, $L^2$ is where Fourier analysis becomes an isometry, and functional analysis is what happens when you keep the linear algebra and lose the finite dimension. Complex analysis stands slightly apart, buying extraordinary rigidity in exchange for a very strong hypothesis. Stochastic calculus sits last because it needs all of it — measure for the integral, probability for the martingale, functional analysis for the $L^2$ limit, and Fourier for the transition densities.

== Further Reading

Axler, S. (2024). _Linear Algebra Done Right_, 4th ed. Springer. (Determinant-free development; the spectral theorem earned rather than assumed.)

Halmos, P. R. (1958). _Finite-Dimensional Vector Spaces_, 2nd ed. Van Nostrand. (Written so that the infinite-dimensional case feels inevitable afterwards.)

Dummit, D. S., & Foote, R. M. (2004). _Abstract Algebra_, 3rd ed. Wiley. (The standard first graduate course, Galois theory included.)

Lang, S. (2002). _Algebra_, revised 3rd ed. Springer. (Terse and complete; a reference rather than a first pass.)

Stein, E. M., & Shakarchi, R. (2003). _Fourier Analysis: An Introduction_. Princeton University Press. (Princeton Lectures in Analysis I.)

Stein, E. M., & Shakarchi, R. (2003). _Complex Analysis_. Princeton University Press. (Princeton Lectures in Analysis II.)

Stein, E. M., & Shakarchi, R. (2005). _Real Analysis: Measure Theory, Integration, and Hilbert Spaces_. Princeton University Press. (Princeton Lectures in Analysis III.)

Stein, E. M., & Shakarchi, R. (2011). _Functional Analysis: Introduction to Further Topics in Analysis_. Princeton University Press. (Princeton Lectures in Analysis IV.)

Rudin, W. (1987). _Real and Complex Analysis_, 3rd ed. McGraw-Hill. (Measure theory and complex analysis in one austere volume.)

Folland, G. B. (1999). _Real Analysis: Modern Techniques and Their Applications_, 2nd ed. Wiley. (The reference for measure, distributions, and Fourier analysis together.)

Durrett, R. (2019). _Probability: Theory and Examples_, 5th ed. Cambridge University Press. (Measure-theoretic probability with the examples kept in view.)

Williams, D. (1991). _Probability with Martingales_. Cambridge University Press. (The shortest honest route to conditional expectation and martingale convergence.)

Øksendal, B. (2003). _Stochastic Differential Equations: An Introduction with Applications_, 6th ed. Springer. (The readable entry point to Itô calculus.)

Karatzas, I., & Shreve, S. E. (1991). _Brownian Motion and Stochastic Calculus_, 2nd ed. Springer. (The rigorous reference behind it.)

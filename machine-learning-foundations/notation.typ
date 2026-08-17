= Notation and Conventions

This chapter fixes the symbols and conventions used throughout the volume. When a chapter deviates (e.g., a paper-specific convention), it says so explicitly.

== Linear Algebra

Scalars are lowercase italic ($x$, $eta$), vectors lowercase ($x in RR^n$, treated as columns), matrices uppercase ($A in RR^(m times n)$). $I$ is the identity; dimensions are inferred from context.

#table(
  columns: (auto, auto),
  [*Symbol*], [*Meaning*],
  [$x^top$, $A^top$], [transpose],
  [$⟨x, y⟩ = x^top y$], [standard inner product],
  [$parallel x parallel_p$], [$L_p$ norm; $parallel x parallel$ without subscript means $L_2$],
  [$parallel A parallel_F$, $parallel A parallel_2$, $parallel A parallel_*$], [Frobenius, spectral, nuclear norms],
  [$"tr"(A)$, $det(A)$, $"rank"(A)$], [trace, determinant, rank],
  [$A^(-1)$, $A^(+)$], [inverse, Moore–Penrose pseudo-inverse],
  [$A succ 0$, $A succ.eq 0$], [positive definite, positive semi-definite],
  [$lambda_i (A)$, $sigma_i (A)$], [eigenvalues, singular values, ordered $lambda_1 >= lambda_2 >= ...$],
  [$kappa(A) = sigma_max \/ sigma_min$], [condition number],
  [$A times.o B$], [Kronecker product],
  [$delta_(i j)$], [Kronecker delta],
)

*Matrix calculus* uses denominator layout: $partial f \/ partial X$ has the shape of $X$. The gradient of a scalar loss is $nabla_theta cal(L)$; the Hessian is $nabla^2 f$ or $H$.

== Probability and Statistics

#table(
  columns: (auto, auto),
  [*Symbol*], [*Meaning*],
  [$(Omega, cal(F), PP)$], [probability space],
  [$X tilde p$], [random variable $X$ has law $p$],
  [$EE[X]$, $"Var"(X)$, $"Cov"(X, Y)$], [expectation, variance, covariance],
  [$p(x)$, $p(x | y)$], [density or mass; conditional],
  [$cal(N)(mu, Sigma)$], [Gaussian with mean $mu$, covariance $Sigma$],
  [$hat(theta)$], [an estimator of parameter $theta$],
  [$cal(D) = {(x_i, y_i)}_(i=1)^n$], [training set of $n$ i.i.d. samples],
  [$EE_(x tilde p)[f(x)]$], [expectation of $f$ under $p$],
)

== Information Theory

Logarithms are natural ($log = ln$, units in nats) in optimization and probability contexts; in information-theoretic contexts (entropy, coding bounds), bare $log$ is base 2 and quantities are in bits unless stated otherwise.

#table(
  columns: (auto, auto),
  [*Symbol*], [*Meaning*],
  [$H(p)$ or $H(X)$], [Shannon entropy],
  [$H(p, q)$], [cross-entropy of $q$ relative to $p$],
  [$"KL"(p parallel q)$], [Kullback–Leibler divergence; forward KL has data/target as first argument],
  [$I(X; Y)$], [mutual information],
  [$H(X | Y)$], [conditional entropy],
)

== Optimization

#table(
  columns: (auto, auto),
  [*Symbol*], [*Meaning*],
  [$theta$, $x_k$], [parameters; iterate at step $k$],
  [$eta$], [learning rate (step size)],
  [$beta$, $beta_1$, $beta_2$], [momentum / moment-decay coefficients],
  [$L$, $mu$], [smoothness and strong-convexity constants],
  [$kappa = L \/ mu$], [condition number of the objective],
  [$f^*$, $x^*$], [optimal value, minimizer],
  [$B$], [mini-batch size],
  [$cal(L)(theta)$], [training loss],
)

== Reinforcement Learning

An MDP is the tuple $(cal(S), cal(A), P, R, gamma)$: states, actions, transition kernel $P(s' | s, a)$, expected reward $R(s, a)$, discount $gamma in [0, 1)$. A policy is $pi(a | s)$; the return is $G_t = sum_k gamma^k R_(t+k+1)$; value functions are $V^pi (s)$ and $Q^pi (s, a)$, with optima $V^*$, $Q^*$.

== Asymptotics and Misc

$O(dot)$, $Omega(dot)$, $Theta(dot)$ have their usual meanings; $tilde(O)(dot)$ hides polylogarithmic factors. $[n] = {1, ..., n}$. $bb(1)[dot]$ is the indicator function. "i.i.d." means independent and identically distributed. One MAC counts as 2 FLOPs.

== Further Reading

Goodfellow, I., Bengio, Y., & Courville, A. (2016). _Deep Learning_. MIT Press. (Chapter 2 establishes much of the notation used across the field.)

Bishop, C. M. (2006). _Pattern Recognition and Machine Learning_. Springer. (Probabilistic notation conventions.)

Murphy, K. P. (2022). _Probabilistic Machine Learning: An Introduction_. MIT Press. (Consistent modern notation with an explicit symbol table.)

#import "../template.typ": xref

= Probability and Information <probability-and-information>

Machine learning is, at its most abstract, the art of using data to update beliefs about a generating process. Probability provides the calculus of uncertainty; information theory provides the units in which we measure how much a model has actually learned. This chapter covers the probabilistic primitives and information-theoretic quantities that recur throughout the rest of the volume — densities and divergences, expectations and entropies, estimators and concentration bounds — at a depth sufficient to read modern ML papers without re-deriving each result from scratch.

*See also:* #xref("machine-learning-foundations", "generalization-theory", label: "Generalization Theory") (PAC-Bayes, concentration), #xref("machine-learning-foundations", "diffusion-models", label: "Diffusion Models") (KL, evidence lower bound), #xref("machine-learning-foundations", "optimization", label: "Optimization") (stochastic gradients as Monte Carlo), #xref("llm", "pretraining", label: "Pretraining") (cross-entropy as the training objective).

== Probability Spaces and Random Variables

A probability space is a triple $(Omega, cal(F), PP)$: the sample space $Omega$, a sigma-algebra $cal(F)$ of measurable events, and a probability measure $PP$ with $PP(Omega) = 1$. A random variable $X : Omega -> RR$ is an $cal(F)$-measurable function; its law is the pushforward measure $PP_X (A) = PP(X^(-1)(A))$.

For our purposes the two regimes are:

- *Discrete:* probability mass function $p(x) = PP(X = x)$, with $sum_x p(x) = 1$.
- *Continuous:* density $p(x)$ with respect to Lebesgue measure, $integral p(x) d x = 1$.

The Radon–Nikodym derivative unifies them: $p = d PP_X \/ d mu$ for a reference measure $mu$ (counting measure for discrete, Lebesgue for continuous).

=== Expectation and Variance

$ EE[X] = integral x d PP_X (x), quad "Var"(X) = EE[(X - EE[X])^2] = EE[X^2] - EE[X]^2. $

Linearity of expectation ($EE[a X + b Y] = a EE[X] + b EE[Y]$) holds without independence. Variance, in contrast, satisfies $"Var"(X + Y) = "Var"(X) + "Var"(Y) + 2 "Cov"(X, Y)$, and only telescopes under independence.

=== Conditional Probability and Bayes' Rule

For events $A, B$ with $PP(B) > 0$, $PP(A | B) = PP(A inter B) \/ PP(B)$. Bayes' rule expresses posterior beliefs:

$ p(theta | D) = (p(D | theta) p(theta)) / (p(D)) = (p(D | theta) p(theta)) / (integral p(D | theta') p(theta') d theta'). $

The denominator $p(D) = integral p(D | theta) p(theta) d theta$ is the marginal likelihood or *evidence*; it normalizes the posterior and serves as the model-selection criterion in Bayesian model comparison.

== Common Distributions

#table(
  columns: 4,
  [*Name*], [*Support*], [*Density / Mass*], [*Use in ML*],
  [Bernoulli$(p)$], [${0, 1}$], [$p^x (1-p)^(1-x)$], [Binary classification head],
  [Categorical$(pi)$], [${1, ..., K}$], [$pi_x$], [Softmax output],
  [Gaussian$(mu, sigma^2)$], [$RR$], [$1 / sqrt(2 pi sigma^2) exp(-(x-mu)^2 / (2 sigma^2))$], [Regression noise, VAE prior],
  [Multivariate normal], [$RR^d$], [det. covariance form], [Latent factor models],
  [Dirichlet$(alpha)$], [simplex], [$1 / B(alpha) product theta_i^(alpha_i - 1)$], [Topic models, Bayesian softmax],
  [Beta$(alpha, beta)$], [$[0,1]$], [conjugate to Bernoulli], [Bayesian A/B testing],
  [Exponential family], [varies], [$h(x) exp(eta(theta)^top T(x) - A(theta))$], [GLMs, natural gradient],
)

The *exponential family* form is foundational: $T(x)$ is the sufficient statistic, $eta$ the natural parameter, $A$ the log-partition function. The Fisher information equals the Hessian of $A$ at $eta$, which underlies natural-gradient methods (see _Optimization_).

=== Reparameterization

A Gaussian sample can be written $x = mu + sigma dot epsilon$ with $epsilon tilde cal(N)(0, 1)$. This *reparameterization trick* moves randomness outside the differentiable path and is the gradient backbone of VAEs and diffusion models.

```python
import torch

def sample_gaussian(mu, log_sigma):
    # mu, log_sigma: (batch, dim) tensors with grad
    eps = torch.randn_like(mu)
    return mu + torch.exp(log_sigma) * eps
```

== Information Theory

Shannon defined the *entropy* of a discrete distribution $p$ as

$ H(p) = -sum_x p(x) log p(x), $

measured in nats when $log$ is natural and bits when $log_2$. Entropy is the minimum expected code length (Shannon's source coding theorem) and equivalently the average surprise of samples from $p$.

=== Cross-Entropy and KL Divergence

$ H(p, q) = -sum_x p(x) log q(x), quad "KL"(p parallel q) = sum_x p(x) log p(x) / q(x) = H(p, q) - H(p). $

KL is non-negative (Gibbs' inequality), zero iff $p = q$ almost everywhere, and asymmetric: $"KL"(p parallel q) eq.not "KL"(q parallel p)$. The asymmetry matters in practice:

- *Forward KL* $"KL"(p parallel q)$ is *mode-covering*: $q$ pays a price wherever $p$ has mass.
- *Reverse KL* $"KL"(q parallel p)$ is *mode-seeking*: $q$ can ignore modes of $p$ at no cost.

Variational inference minimizes reverse KL (the ELBO is its negation up to a constant); maximum-likelihood training of a generative model is forward KL between data and model.

=== Mutual Information

$ I(X ; Y) = "KL"(p(x, y) parallel p(x) p(y)) = H(X) - H(X | Y). $

Mutual information quantifies how much one variable tells you about another. In representation learning, the InfoNCE bound (see _Representation Learning_) is a tractable lower bound on $I(X ; Y)$.

```python
import torch
import torch.nn.functional as F

def info_nce(q, k, temperature=0.07):
    """
    q: (N, d) queries; k: (N, d) keys. Positives are paired by index.
    Lower bound on mutual information per van den Oord et al. 2018.
    """
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = q @ k.t() / temperature           # (N, N)
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)
```

=== Jensen-Shannon and $f$-Divergences

The Jensen–Shannon divergence symmetrizes KL: $"JS"(p, q) = 1/2 "KL"(p parallel m) + 1/2 "KL"(q parallel m)$ with $m = (p + q) \/ 2$. The original GAN objective is a JS-divergence variational bound; Wasserstein GANs use the Earth Mover's distance instead because JS saturates when supports are disjoint.

The general *$f$-divergence* $D_f (p parallel q) = integral q(x) f(p(x) \/ q(x)) d x$ covers KL ($f(t) = t log t$), reverse KL, $chi^2$, total variation, and Hellinger as special cases.

== Estimation

=== Maximum Likelihood

Given i.i.d. data $D = {x_i}_(i=1)^n$ and a parametric family $p_theta$,

$ hat(theta)_("MLE") = arg max_theta sum_(i=1)^n log p_theta (x_i). $

MLE is consistent under regularity conditions, asymptotically efficient (achieves the Cramér–Rao lower bound), and equivalent to minimizing forward KL between the empirical and model distributions.

=== Maximum a Posteriori and Regularization

$ hat(theta)_("MAP") = arg max_theta [sum log p_theta (x_i) + log p(theta)]. $

A Gaussian prior on weights yields $L_2$ regularization; a Laplace prior yields $L_1$. MAP is *not* invariant to reparameterization, unlike the full Bayesian posterior.

=== Method of Moments and Pseudo-likelihood

When the likelihood is intractable, matching empirical moments to model moments (or maximizing a pseudo-likelihood that factorizes) yields consistent estimators at the cost of efficiency. Modern instances include noise-contrastive estimation and score matching (used in diffusion models).

== Concentration Inequalities

These bounds control how empirical averages deviate from expectations and underlie generalization theory.

#table(
  columns: 3,
  [*Inequality*], [*Statement*], [*Assumption*],
  [Markov], [$PP(X >= a) <= EE[X] \/ a$], [$X >= 0$],
  [Chebyshev], [$PP(|X - mu| >= k sigma) <= 1/k^2$], [finite variance],
  [Hoeffding], [$PP(|hat(mu) - mu| >= t) <= 2 exp(-2 n t^2 / (b - a)^2)$], [bounded $X in [a, b]$],
  [Bernstein], [refines Hoeffding using variance], [bounded, finite var.],
  [McDiarmid], [bounded-differences], [Lipschitz in each coord.],
  [Azuma], [martingale Hoeffding], [bounded increments],
)

Hoeffding's inequality is the workhorse: a Monte Carlo estimate with $n$ samples has error $O(1 \/ sqrt(n))$ with high probability, independent of dimension — the basis for SGD, MC integration, and PAC bounds.

== Stochastic Convergence

Three modes of convergence appear constantly:

- *Almost sure* ($X_n -->^("a.s.") X$): $PP(lim X_n = X) = 1$. Strongest.
- *In probability* ($X_n -->^(PP) X$): $forall epsilon, PP(|X_n - X| > epsilon) -> 0$.
- *In distribution* ($X_n -->^(d) X$): CDFs converge at continuity points.

The *law of large numbers* says sample means converge to expectations (a.s. for SLLN, in probability for WLLN). The *central limit theorem* says

$ sqrt(n) (hat(mu)_n - mu) -->^(d) cal(N)(0, sigma^2), $

which justifies Gaussian confidence intervals around MLEs and gives a rationale for the Gaussian noise model in many SGD analyses.

== The Fisher Information and Score

The *score* is $s_theta (x) = nabla_theta log p_theta (x)$, with $EE_(p_theta) [s_theta (x)] = 0$ at the truth. The *Fisher information matrix* is

$ F(theta) = EE_(p_theta) [s_theta (x) s_theta (x)^top] = -EE_(p_theta) [nabla^2_theta log p_theta (x)]. $

The Cramér–Rao bound states $"Cov"(hat(theta)) succ.eq F(theta)^(-1)$ for any unbiased estimator. In ML, $F$ defines the *natural-gradient* preconditioner: $tilde(g) = F^(-1) g$ rescales gradients so steps are isotropic in the KL geometry rather than the Euclidean one. K-FAC, Shampoo, and Sophia all approximate this matrix.

=== Empirical Fisher

In practice we replace $F$ with the empirical sum $hat(F) = 1/n sum_i s_theta (x_i) s_theta (x_i)^top$. This is biased relative to the true Fisher when the model is misspecified — Kunstner et al. (2019) discuss why empirical Fisher can mislead natural-gradient methods.

== Exchangeability and de Finetti

A sequence is *exchangeable* if its joint distribution is invariant under permutation. de Finetti's theorem says any infinite exchangeable binary sequence is a mixture of i.i.d. Bernoullis. This is the foundation for hierarchical Bayesian models and underlies why in-context learning (`llm/pretraining.typ`) can be analyzed as implicit Bayesian inference over latent task variables.

== Worked Example: Logistic Regression as Maximum Likelihood

```python
import numpy as np

def neg_log_likelihood(w, X, y):
    # y in {0, 1}
    logits = X @ w
    log_sigmoid = -np.logaddexp(0, -logits)              # numerically stable
    log_1_minus = -np.logaddexp(0,  logits)
    return -np.mean(y * log_sigmoid + (1 - y) * log_1_minus)

def grad(w, X, y):
    p = 1.0 / (1.0 + np.exp(-X @ w))
    return X.T @ (p - y) / X.shape[0]
```

Minimizing the negative log-likelihood is equivalent to minimizing the forward KL between the empirical label distribution and the model's Bernoulli predictions. Adding $lambda parallel w parallel_2^2$ corresponds to a Gaussian prior with precision $2 lambda$ — see _Classical Models_ for the closed-form connection to ridge regression.

== Numerical Caveats

- *Log-sum-exp:* compute $log sum_i exp(z_i)$ via $z_max + log sum_i exp(z_i - z_max)$.
- *Softmax:* subtract the max logit before exponentiating.
- *Log-densities:* always carry $log p$ rather than $p$ for products that span orders of magnitude.
- *KL of Gaussians:* closed form $1/2 [log sigma_2^2 / sigma_1^2 + (sigma_1^2 + (mu_1 - mu_2)^2) / sigma_2^2 - 1]$; avoid sampling estimators when this is available.

== Further Reading

Cover, T., Thomas, J. (2006). _Elements of Information Theory_, 2nd ed. Wiley.

MacKay, D. (2003). _Information Theory, Inference, and Learning Algorithms_. Cambridge.

Bishop, C. (2006). _Pattern Recognition and Machine Learning_. Springer. Chapter 2.

Murphy, K. (2022). _Probabilistic Machine Learning: An Introduction_. MIT Press.

Wainwright, M., Jordan, M. (2008). "Graphical Models, Exponential Families, and Variational Inference." Foundations and Trends in ML.

Boucheron, S., Lugosi, G., Massart, P. (2013). _Concentration Inequalities: A Nonasymptotic Theory of Independence_. Oxford.

Kunstner, F., Balles, L., Hennig, P. (2019). "Limitations of the Empirical Fisher Approximation." NeurIPS.

van den Oord, A., Li, Y., Vinyals, O. (2018). "Representation Learning with Contrastive Predictive Coding." arXiv.

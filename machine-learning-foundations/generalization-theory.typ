#import "../template.typ": xref

= Generalization Theory

Why does a model that fits 60M parameters to 50K labeled examples ever generalize? Classical statistical learning theory says it should not: the model has more than enough capacity to memorize noise. The story of generalization theory over the past three decades is the gradual closing of that gap: from worst-case VC bounds, through data-dependent Rademacher complexities and PAC-Bayes, to the modern phenomena of double descent and neural scaling laws. This chapter surveys those results at the level a practitioner needs to read papers like "Reconciling modern machine-learning practice and the classical bias-variance trade-off" or the Chinchilla scaling-laws paper.

*See also:* #xref("machine-learning-foundations", "probability-and-information", label: "Probability and Information") (concentration inequalities), #xref("machine-learning-foundations", "optimization", label: "Optimization") (implicit regularization), #xref("llm", "pretraining", label: "Pretraining") (Chinchilla, scaling laws).

== The Setup

A learning algorithm sees i.i.d. samples $(x_i, y_i) tilde D$ for $i = 1, ..., n$ and returns a hypothesis $h in cal(H)$. The *risk* (true error) is

$ R(h) = EE_((x, y) tilde D) [ell(h(x), y)], $

and the *empirical risk* is $hat(R)_n (h) = 1/n sum_i ell(h(x_i), y_i)$. *Generalization* is the gap $R(h) - hat(R)_n (h)$. We want this gap to be small with high probability.

== VC Dimension

For binary classification with 0/1 loss, the *VC dimension* of $cal(H)$ is the largest $d$ such that some set of $d$ points can be *shattered* (labeled in all $2^d$ possible ways) by $cal(H)$. Vapnik–Chervonenkis (1971) proved that for any $delta > 0$, with probability $1 - delta$ over the sample,

$ sup_(h in cal(H)) |R(h) - hat(R)_n (h)| <= O(sqrt((d log n + log(1 \/ delta)) / n)). $

Key implications:

- *VC dim grows with capacity.* Linear classifiers in $RR^d$: $d + 1$. Decision stumps: $log_2$ of feature count. Two-layer ReLU nets: $O(W L log W)$ in weights $W$ and depth $L$.
- *Worst-case bound.* The bound is uniform over $cal(H)$ and over data distributions; for any specific learned $h$ on a specific distribution it is often vacuous.

VC theory predicted that deep nets should not generalize: a network with $10^8$ parameters has effective VC dim much larger than typical dataset sizes. The fact that they *do* generalize is one of the central puzzles of modern ML.

== Rademacher Complexity

A more refined, data-dependent measure. For samples $S = (z_1, ..., z_n)$ and i.i.d. $sigma_i in {plus.minus 1}$,

$ hat(cal(R))_S (cal(F)) = EE_sigma [sup_(f in cal(F)) 1/n sum_(i=1)^n sigma_i f(z_i)]. $

Intuitively, it measures how well $cal(F)$ can fit random labels. Rademacher generalization bound:

$ R(h) <= hat(R)_n (h) + 2 hat(cal(R))_S (cal(L) circle.stroked cal(H)) + 3 sqrt(log(2 \/ delta) \/ (2 n)). $

For Lipschitz losses and norm-bounded hypothesis classes, Rademacher complexity often gives tight bounds where VC dim is loose. Bartlett's *margin theory* for SVMs uses Rademacher to prove margin-based generalization bounds for kernel methods.

== PAC-Bayes

A Bayesian-flavored framework that produces non-vacuous bounds even for modern neural nets. For a prior $P$ on $cal(H)$ chosen before seeing data and any posterior $Q$ (data-dependent),

$ EE_(h tilde Q) [R(h)] <= EE_(h tilde Q) [hat(R)_n (h)] + sqrt(("KL"(Q parallel P) + log(2 sqrt(n) \/ delta)) / (2 n - 1)). $

Dziugaite & Roy (2017) computed PAC-Bayes bounds for MNIST CNNs that were under 20%, the first non-vacuous neural net bound. The trick: take $P$ to be a Gaussian around the *random initialization* and let $Q$ be a Gaussian around the trained weights. Modern variants (Pérez-Ortiz et al. 2021) achieve sub-10% bounds.

== The Bias-Variance Decomposition

For squared loss and a random training set,

$ EE_S [(f(x) - y)^2] = underbrace((EE_S [f(x)] - EE[y | x])^2, "bias"^2) + underbrace(EE_S [(f(x) - EE_S [f(x)])^2], "variance") + sigma^2_("noise"). $

Classical wisdom: capacity tradeoff. Increasing model complexity reduces bias, increases variance; the goal is to find the sweet spot. This worked for kernel methods and shallow nets. Then deep learning happened.

== Double Descent

Belkin et al. (2019) documented a striking phenomenon: as model capacity increases past the interpolation threshold (where training error reaches zero), test error first rises (classical regime) and then falls *again*, often below the classical optimum. The interpolation threshold is roughly where the number of parameters equals the number of data points.

#table(
  columns: 3,
  [*Regime*], [*Capacity*], [*Behavior*],
  [Underparameterized], [$p < n$], [Classical U-curve],
  [Interpolation threshold], [$p approx n$], [Test error peaks (variance explosion)],
  [Overparameterized], [$p > n$], [Second descent: test error falls again],
)

Nakkiran et al. (2020) showed double descent also occurs along the *training-epoch* axis: error rises mid-training and falls again. The unifying explanation: in the overparameterized regime, the optimizer's *implicit bias* selects a particular interpolating solution among the many that exist, and that solution tends to generalize.

== Neural Scaling Laws

Kaplan et al. (2020), refined by Hoffmann et al. (2022, "Chinchilla"), found that LLM test loss follows precise power laws in model size $N$, data size $D$, and compute $C$:

$ L(N, D) approx E + A / N^alpha + B / D^beta, $

with $alpha approx 0.34$, $beta approx 0.28$ for the Chinchilla constants (these vary by architecture and data). The compute-optimal frontier is $N prop C^a$, $D prop C^(1-a)$; Chinchilla found $a approx 0.5$, meaning model and data should scale equally, *not* model-heavy as in the Kaplan recipe. See `llm/pretraining.typ` for the practical consequences.

```python
import numpy as np

def chinchilla_loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """Predicted test loss for an LLM with N params on D tokens (Chinchilla)."""
    return E + A / (N ** alpha) + B / (D ** beta)

def compute_optimal(C, ratio_tokens_per_param=20):
    """Compute-optimal split: tokens approx 20 * params (Chinchilla rule)."""
    # C = 6 N D approximately for transformer FLOPs
    N = np.sqrt(C / (6 * ratio_tokens_per_param))
    D = ratio_tokens_per_param * N
    return N, D
```

Scaling laws are *empirical*; theoretical explanations (Bahri et al. 2024, "Explaining Neural Scaling Laws") rest on intrinsic data dimensionality and the eigen-spectrum of the feature kernel.

== Stability and Generalization

An algorithm is *uniform-stable* with parameter $beta$ if replacing one training point changes the loss on any test point by at most $beta$. Bousquet & Elisseeff (2002) proved a stable algorithm satisfies

$ |EE [R(h_S)] - EE [hat(R)_n (h_S)]| <= beta. $

Hardt et al. (2016) showed SGD on convex losses is $O(T \/ n)$-stable after $T$ iterations, implying that, surprisingly, generalization improves with smaller learning rates and fewer steps, *not* just smaller capacity.

== The Implicit Bias of SGD

Even when many minima of the training loss exist, SGD selects specific ones. Key results:

- *Logistic regression on separable data:* gradient descent converges to the max-margin direction (Soudry et al. 2018), without explicit regularization.
- *Matrix factorization:* gradient flow with small initialization converges to the min-nuclear-norm solution (Gunasekar et al. 2017).
- *Deep ReLU nets:* recent work (Lyu–Li 2019, Chizat–Bach 2020) shows analogous max-margin behavior under various assumptions.

These results connect generalization to *which* interpolating solution we find, recasting the question "why do overparameterized models generalize?" as "what does the optimizer choose?"

== The Neural Tangent Kernel

Jacot, Gabriel, Hongler (2018) observed that in the infinite-width limit, training a deep net with gradient descent is equivalent to kernel regression with the *neural tangent kernel*

$ K_("NTK") (x, x') = EE_(theta tilde "init") [⟨nabla_theta f(x; theta), nabla_theta f(x'; theta)⟩]. $

In this regime, training is convex (kernel ridge regression), the parameters barely move ("lazy training"), and standard kernel-method generalization bounds apply. This explains *some* of why wide nets train so reliably but does not capture *feature learning*, where the kernel itself adapts during training (the regime where representation learning happens).

== Uniform Convergence and Its Limits

Nagarajan & Kolter (2019) showed that uniform-convergence-based generalization bounds (VC, Rademacher, PAC-Bayes with prior fixed before training) cannot explain deep learning generalization: any bound that is uniform over a sufficiently rich class is forced to be vacuous on real data. The implication is that *data-dependent* analyses are required: PAC-Bayes with data-dependent priors, implicit-bias arguments, and NTK in the appropriate regime.

== Compression-Based Bounds

Arora et al. (2018), Zhou et al. (2019): if a trained network can be compressed (sparsified, quantized, or distilled) to $k$ bits while preserving accuracy, then it generalizes with a bound proportional to $sqrt(k \/ n)$. Formally, any function class of VC-dimension $d$ satisfies $R(h) - hat(R)(h) = O(sqrt(d \/ n))$; compression provides an implicit upper bound on effective dimension.

This gives a practical handle: networks that tolerate aggressive pruning (90%+ sparsity at minimal accuracy loss) have empirically low effective dimension and generalize well. The bound also explains the lottery ticket hypothesis: small dense sub-networks train to the same accuracy as the full network, with fewer bits needed to describe them. A corollary for practitioners: post-training quantization success (INT8 with < 1% degradation) is evidence of good generalization, not just model efficiency.

== Practical Diagnostics

For a practitioner, generalization is an empirical question. Useful diagnostics:

- *Train-test gap* as a function of model size: U-curve, monotone decreasing, or double-descent shape.
- *Memorization probes:* train on randomly labeled data (Zhang et al. 2017); if the model can fit random labels, it has the capacity to memorize, but on real data it does not.
- *Sharpness/flatness:* Hessian top eigenvalue at the solution. Flatter $approx$ better generalization (controversial but empirically robust).
- *Influence functions:* leave-one-out approximations to identify which training points the model "depends on" most.

```python
import torch

def hessian_top_eig(loss_fn, params, num_iter=20):
    """Power iteration for top Hessian eigenvalue."""
    v = [torch.randn_like(p) for p in params]
    norm = sum((vi**2).sum() for vi in v).sqrt()
    v = [vi / norm for vi in v]
    for _ in range(num_iter):
        loss = loss_fn()
        grads = torch.autograd.grad(loss, params, create_graph=True)
        dot = sum((g * vi).sum() for g, vi in zip(grads, v))
        hv = torch.autograd.grad(dot, params)
        norm = sum((h**2).sum() for h in hv).sqrt()
        v = [h / norm for h in hv]
    return norm.item()
```

== Grokking

Power et al. (2022) trained small transformers on *modular arithmetic* tasks — e.g., $a + b mod 113$ for $(a, b)$ pairs — using only a fraction (30-50%) of all possible $(a, b)$ pairs as training data. The result was striking: training loss reached near-zero almost immediately, but validation accuracy stayed at chance for thousands of additional steps before suddenly jumping to near 100%. The delay between train-loss-zero and val-accuracy-high can span $100 times$ as many gradient steps as the initial descent. The phenomenon is robust across modular addition, multiplication, permutation composition, and sparse parity problems.

=== Mechanistic Interpretation

Nanda et al. (2023) used activation patching and circuit analysis (cf. _Interpretability_) to identify two distinct learned circuits present simultaneously in a grokking model:

1. *Memorizing circuit:* a high-norm lookup table that essentially caches training examples in the weights. Achieves zero training loss but generalizes not at all.
2. *Generalizing circuit:* a low-norm Fourier-feature representation that computes modular arithmetic via structured frequency embeddings in the residual stream. The key computation is:
   $ "embed"(a) + "embed"(b) → "project onto Fourier basis" → "read off" (a + b) mod p. $

Both circuits coexist in the trained model. Weight decay slowly erodes the high-norm memorizing circuit (which is more costly to maintain under $L_2$ regularization) while the low-norm generalizing circuit, once present, is comparatively cheap. The transition point is where the generalizing circuit's lower regularization cost tips the balance: the loss on the generalizing circuit (including the weight penalty) falls below the loss on the memorizing one, and the optimizer rapidly transfers weight to the generalizing circuit.

=== Role of Weight Decay

Without weight decay (or equivalent regularization), grokking does not occur — the memorizing solution is a stable local minimum and the optimizer has no incentive to abandon it. Stronger weight decay accelerates the generalization transition, reducing the delay from millions of steps to thousands. This is one of the clearest experimental confirmations that *regularization drives generalization* rather than merely preventing overfitting post-hoc.

=== Connection to Double Descent

Grokking is a *temporal* analog of double descent. In double descent, the interpolation threshold is crossed by increasing model size or data; in grokking, it is crossed during training by continued optimization past the interpolation point. The memorizing phase corresponds to the variance-explosion regime at the interpolation threshold; the generalizing phase corresponds to the second descent. Both phenomena are explained by the optimizer's implicit bias toward lower-norm solutions.

=== Practical Implications

- *Do not stop training too early on small algorithmic datasets.* Training accuracy saturating at 100% is not a reliable signal that validation has converged; validation may still be many thousands of steps away.
- *Monitor validation loss long after training loss plateaus.* A standard early-stopping criterion based on training loss will terminate training in the memorizing phase.
- *Weight decay is critical.* In any setting where training data is small relative to the hypothesis class (few-shot fine-tuning, small-table tabular data, mathematical reasoning), the regularization hyperparameter directly controls whether delayed generalization can occur.
- *Grokking as a diagnostic.* Observing a grokking-like curve (train collapses, val lags) in production training is a sign that the model has sufficient capacity and data is the bottleneck. It is not a pathology; it may resolve with longer training.

Grokking is now a standard testbed for mechanistic interpretability and for studying the geometry of generalization decoupled from data complexity.

== Distribution Shift

Generalization within-distribution is one story; *out-of-distribution* (OOD) generalization is harder and less well-understood. Standard taxonomy:

#table(
  columns: 3,
  [*Shift type*], [*Definition*], [*Examples*],
  [Covariate shift], [$p(x)$ differs, $p(y | x)$ same], [Webcam vs phone images],
  [Label shift], [$p(y)$ differs, $p(x | y)$ same], [Disease prevalence],
  [Concept drift], [$p(y | x)$ differs], [Spam patterns evolve],
  [Adversarial], [$x$ chosen by adversary], [FGSM, PGD attacks],
)

Domain generalization aims to train on multiple source distributions and generalize to a held-out target. Group DRO, IRM, and Mixup are common interventions; Gulrajani–Lopez-Paz (2021) showed that simple ERM is competitive with most domain-generalization methods given fair tuning.

== Further Reading

Vapnik, V. (1998). _Statistical Learning Theory_. Wiley.

Shalev-Shwartz, S., Ben-David, S. (2014). _Understanding Machine Learning_. Cambridge.

Mohri, M., Rostamizadeh, A., Talwalkar, A. (2018). _Foundations of Machine Learning_, 2nd ed. MIT Press.

Bartlett, P., Mendelson, S. (2002). "Rademacher and Gaussian Complexities." JMLR.

McAllester, D. (1998). "Some PAC-Bayesian Theorems." COLT.

Dziugaite, G., Roy, D. (2017). "Computing Nonvacuous Generalization Bounds for Deep Stochastic NNs." UAI.

Zhang, C., Bengio, S., Hardt, M., Recht, B., Vinyals, O. (2017). "Understanding Deep Learning Requires Rethinking Generalization." ICLR.

Belkin, M., Hsu, D., Ma, S., Mandal, S. (2019). "Reconciling Modern Machine-Learning Practice and the Classical Bias-Variance Trade-Off." PNAS.

Nakkiran, P. et al. (2020). "Deep Double Descent." ICLR.

Kaplan, J. et al. (2020). "Scaling Laws for Neural Language Models." arXiv.

Hoffmann, J. et al. (2022). "Training Compute-Optimal Large Language Models" (Chinchilla). NeurIPS.

Jacot, A., Gabriel, F., Hongler, C. (2018). "Neural Tangent Kernel." NeurIPS.

Nagarajan, V., Kolter, Z. (2019). "Uniform Convergence May Be Unable to Explain Generalization in Deep Learning." NeurIPS.

Soudry, D. et al. (2018). "The Implicit Bias of Gradient Descent on Separable Data." JMLR.

Power, A. et al. (2022). "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets." arXiv.

Gulrajani, I., Lopez-Paz, D. (2021). "In Search of Lost Domain Generalization." ICLR.

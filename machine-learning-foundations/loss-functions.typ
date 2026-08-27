#import "../template.typ": xref

= Loss Functions <loss-functions>

The choice of loss function encodes what "correct" means for a model. Cross-entropy for classification, MSE for regression — but modern deep learning uses a much richer palette: contrastive losses for representation learning, focal loss for class imbalance, InfoNCE for self-supervised learning. A loss function is not merely a training convenience; it is the formal statement of the task. Understanding the landscape from first principles clarifies why certain losses dominate certain domains and how to design objectives for new ones.

*See also:* #xref("machine-learning-foundations", "optimization", label: "Optimization") (training dynamics), #xref("machine-learning-foundations", "generalization-theory", label: "Generalization Theory")

== Regression Losses

=== Mean Squared Error (L2)

The canonical regression loss:

$ cal(L)_"MSE" = 1/n sum_(i=1)^n (y_i - hat(y)_i)^2. $

The gradient with respect to the prediction is $-2(y_i - hat(y)_i)$: proportional to the residual, so large errors get large gradient updates. MSE corresponds to maximum likelihood estimation under a Gaussian noise model $y = f(x) + epsilon$, $epsilon tilde cal(N)(0, sigma^2)$.

The Gaussian assumption is the weakness: outliers contribute quadratically to the loss, making MSE sensitive to label noise and heavy-tailed residuals.

=== Mean Absolute Error (L1)

$ cal(L)_"MAE" = 1/n sum_(i=1)^n |y_i - hat(y)_i|. $

The gradient is $-"sgn"(y_i - hat(y)_i)$: constant magnitude regardless of residual size. This makes MAE robust to outliers (they contribute linearly, not quadratically) but introduces a discontinuity in the gradient at zero. The subgradient at zero is any value in $[-1, 1]$, which is fine theoretically but can cause oscillation in practice around the optimum.

=== Huber Loss

The Huber loss (Huber 1964) interpolates: quadratic for small residuals, linear for large ones:

$ cal(L)_delta (e) = cases(
  1/2 e^2 & "if" |e| <= delta,
  delta (|e| - delta/2) & "otherwise."
) $

The gradient transitions smoothly from $-e$ (MSE regime) to $plus.minus delta$ (MAE regime), eliminating the spike at zero while preserving outlier robustness. The hyperparameter $delta$ controls the crossover point; it must be tuned or estimated from residual scale.

=== Quantile Loss

For predicting the $tau$-th quantile ($tau in (0, 1)$):

$ cal(L)_tau (e) = cases(
  tau e & "if" e >= 0,
  (tau - 1) e & "if" e < 0.
) $

Setting $tau = 0.5$ recovers MAE. Setting $tau = 0.1$ and $tau = 0.9$ and training two models yields an 80% prediction interval. Quantile regression is the basis for conformal prediction intervals and probabilistic forecasting in weather, finance, and demand planning.

== Classification Losses

=== Cross-Entropy

For a true distribution $p$ and predicted distribution $q$, the cross-entropy is

$ H(p, q) = -sum_x p(x) log q(x). $

In classification with hard labels ($p$ is a one-hot), this reduces to $-log q(y)$ where $y$ is the true class. The connection to maximum likelihood is direct: minimizing cross-entropy is equivalent to maximizing $log$-likelihood under the model. The connection to information theory is through the KL divergence:

$ "KL"(p parallel q) = H(p, q) - H(p). $

Since $H(p)$ is constant with respect to the model parameters, minimizing cross-entropy is equivalent to minimizing $"KL"(p parallel q)$, driving the predicted distribution toward the true one.

*Binary cross-entropy* for $y in {0, 1}$ and prediction $hat(p) in (0, 1)$:

$ cal(L)_"BCE" = -[y log hat(p) + (1 - y) log(1 - hat(p))]. $

*Categorical cross-entropy* for $K$ classes: $cal(L)_"CE" = -sum_(k=1)^K y_k log hat(p)_k$ where $hat(p)$ is the softmax output.

=== Label Smoothing

Hard one-hot targets encourage the model to push logit differences toward infinity, leading to overconfident predictions and poor calibration. Label smoothing (Szegedy et al. 2016) replaces hard targets with soft ones:

$ tilde(y)_k = (1 - epsilon) y_k + epsilon/K, $

where $epsilon$ is typically 0.1 and $K$ is the number of classes. The model can never fully satisfy the target, which acts as a regularizer. Label smoothing is standard in ViT, T5, and most large-scale image classifiers; it reliably improves calibration and top-1 accuracy by 0.2–0.5% on ImageNet.

=== Focal Loss

Lin et al. (2017) designed focal loss for one-stage object detection (RetinaNet), where the foreground-background class imbalance is extreme (up to $10^4$:1). Easy negatives (correctly classified background) dominate the loss and overwhelm gradient signal from rare foreground objects. The fix: down-weight easy examples dynamically.

$ "FL"(p_t) = -(1 - p_t)^gamma log p_t, $

where $p_t$ is the model's estimated probability for the true class and $gamma >= 0$ is the focusing parameter. When $gamma = 0$, FL reduces to cross-entropy. For $gamma = 2$ (the recommended default), an example with $p_t = 0.9$ contributes $(0.1)^2 approx 0.01$ of its standard CE loss, while a hard example with $p_t = 0.1$ contributes $(0.9)^2 approx 0.81$. Easy examples are suppressed by a factor of $approx 80 times$.

Focal loss (combined with a class-weighting factor $alpha_t$) is the standard loss for single-stage detection and dense prediction tasks with class imbalance.

== Ranking and Metric Learning

=== Triplet Loss

Given an anchor $a$, a positive example $p$ (same class as $a$), and a negative example $n$ (different class), triplet loss enforces a margin:

$ cal(L)_"triplet" = max(0, d(a, p) - d(a, n) + m), $

where $d$ is a distance function (e.g., Euclidean or cosine) and $m > 0$ is a margin. The loss is zero when the negative is already $m$ further from the anchor than the positive. The critical challenge is *hard negative mining*: random triplets are usually trivially satisfied (loss zero), so training collapses. Effective triplet training requires sampling semi-hard negatives — negatives that violate or nearly violate the margin constraint within each batch.

=== N-Pairs Loss

Sohn (2016) extends the triplet idea to $N$ negatives per anchor in a single batch. For a mini-batch of $N$ anchor-positive pairs $\{(x_i, x_i^+)\}$:

$ cal(L)_"N-pairs" = 1/N sum_(i=1)^N log(1 + sum_(j != i) exp(f(x_i)^top f(x_j^+) - f(x_i)^top f(x_i^+))). $

This uses all off-diagonal pairs as negatives, improving sample efficiency over triplet loss and avoiding the need for explicit mining.

=== ArcFace and CosFace

For large-scale face recognition with millions of identities, metric learning is recast as angular classification. The key insight: normalize both feature vectors and class weight vectors to unit length, so the logit for class $k$ is $s cos theta_k$ where $theta_k$ is the angle between the feature and the $k$-th class prototype and $s$ is a fixed scale factor.

ArcFace (Deng et al. 2019) adds an additive angular margin $m$ directly in the angle of the target class before taking the cosine:

$ cal(L)_"ArcFace" = -log (e^(s cos(theta_(y_i) + m)) / (e^(s cos(theta_(y_i) + m)) + sum_(j != y_i) e^(s cos theta_j))), $

where the margin $m$ is applied in angular space rather than to the logit. Typical values are $m = 0.5$ radians ($approx 28.6°$) and $s = 64$.

CosFace (Wang et al. 2018) applies an equivalent margin as an additive cosine penalty instead: $s(cos theta_(y_i) - m)$, which is simpler to implement but slightly less geometrically intuitive. Both enforce a minimum angular separation between class boundaries, producing embeddings that transfer strongly to unseen identities.

The resulting margin-based losses dominate open-set face verification benchmarks (LFW, IJB-C) and have been adopted for speaker recognition, person re-identification, and other fine-grained retrieval tasks where the number of identities at test time is unknown.

== Contrastive and Self-Supervised Losses

=== InfoNCE

van den Oord et al. (2018) introduced InfoNCE in the context of contrastive predictive coding (CPC). Given a query $x$ and a positive key $x^+$ drawn from a conditional distribution, plus $K$ negative keys $\{x_j\}$ drawn from a proposal distribution:

$ cal(L)_"InfoNCE" = -log (exp(f(x)^top f(x^+)) / (exp(f(x)^top f(x^+)) + sum_(j=1)^K exp(f(x)^top f(x_j)))). $

InfoNCE is a lower bound on mutual information $I(x; x^+)$; the bound tightens as $K$ grows. In practice this lower bound property is less important than the empirical observation that optimizing it learns strong representations.

=== NT-Xent (SimCLR)

Chen et al. (2020) popularized normalized temperature-scaled cross-entropy (NT-Xent) for self-supervised visual representation learning. For a mini-batch of $N$ examples, each augmented twice to produce $2N$ views, the loss for a positive pair $(i, j)$ is:

$ cal(L)_(i,j) = -log (exp("sim"(z_i, z_j) / tau) / (sum_(k=1, k != i)^(2N) exp("sim"(z_i, z_k) / tau))), $

where $"sim"(u, v) = u^top v \/ (||u|| ||v||)$ is cosine similarity and $tau$ is a temperature hyperparameter. The denominator sums over all $2N - 1$ other views in the batch, treating all non-matching views as negatives.

NT-Xent is the standard baseline for contrastive SSL; most subsequent work (MoCo v2, BYOL, DINO) is measured against it.

=== Temperature and the Uniformity-Alignment Trade-off

Wang & Isola (2020) decomposed the contrastive objective into two geometric properties of the learned embedding hypersphere:

- *Alignment:* positive pairs should map to nearby embeddings.
- *Uniformity:* embeddings should be spread uniformly over the sphere (to preserve information).

The temperature $tau$ controls the balance. Low $tau$ sharpens the distribution over negatives, emphasizing hard negatives and promoting uniformity at the cost of alignment stability. High $tau$ focuses on alignment but allows representations to collapse. Typical values: $tau in [0.05, 0.2]$ for visual SSL; $tau = 0.07$ in MoCo.

=== Barlow Twins

Zbontar et al. (2021) sidestep the need for large batches of negatives by instead penalizing redundancy in the cross-correlation matrix. For $N$ pairs of embeddings $(z^A, z^B)$ from two augmentations of the same image:

$ cal(L)_"BT" = sum_i (1 - C_(i i))^2 + lambda sum_i sum_(j != i) C_(i j)^2, $

where $C$ is the normalized cross-correlation matrix of $z^A$ and $z^B$ across the batch. The first term drives diagonal entries toward 1 (invariance); the second drives off-diagonals toward 0 (decorrelation). Barlow Twins avoids representation collapse without requiring negatives, momentum encoders, or stop-gradient tricks.

=== VICReg

Bardes et al. (2022) decompose the objective explicitly into three terms:

$ cal(L)_"VICReg" = lambda cal(L)_"inv" + mu cal(L)_"var" + nu cal(L)_"cov", $

where:

- *Invariance* ($cal(L)_"inv"$): MSE between embeddings of the two augmented views of the same image, driving the encoder to produce stable representations across augmentations.
- *Variance* ($cal(L)_"var"$): a hinge loss on the per-dimension standard deviation of embeddings within a batch, $max(0, gamma - "std"(z_j))$, penalizing any dimension that collapses to a constant.
- *Covariance* ($cal(L)_"cov"$): penalizes squared off-diagonal entries of the feature covariance matrix, $1/d sum_(i != j) [C(Z)]_(i j)^2$, preventing dimensions from encoding redundant information.

VICReg avoids negatives and is conceptually clean; each term addresses a distinct failure mode: invariance prevents drift between views, variance prevents dimensional collapse, covariance prevents redundancy. The three coefficients $lambda, mu, nu$ are typically set to $25, 25, 1$ respectively.

== Generative Model Losses

=== Variational Autoencoder (VAE)

Kingma & Welling (2014) derive training from a variational lower bound (ELBO) on the log-likelihood:

$ log p(x) >= EE_(z tilde q(z|x)) [log p(x|z)] - "KL"(q(z|x) parallel p(z)). $

The first term is a reconstruction loss (typically MSE for continuous data, BCE for binary); the second is a regularizer that pulls the learned posterior $q(z|x)$ toward the prior $p(z) = cal(N)(0, I)$. The $beta$-VAE (Higgins et al. 2017) upweights the KL term by $beta > 1$, trading reconstruction quality for disentangled latents.

=== GAN Losses

The original GAN (Goodfellow et al. 2014) minimax formulation:

$ min_G max_D EE_(x tilde p_"data") [log D(x)] + EE_(z tilde p_z) [log(1 - D(G(z)))]. $

In practice the *non-saturating* generator loss $-log D(G(z))$ replaces the minimax form to prevent vanishing gradients when the discriminator is strong. Wasserstein GAN (Arjovsky et al. 2017) uses the Earth Mover's distance instead:

$ cal(L)_"WGAN" = EE_(x tilde p_"data") [D(x)] - EE_(z tilde p_z) [D(G(z))], $

with the discriminator constrained to 1-Lipschitz (via gradient penalty in WGAN-GP: $lambda EE [(||nabla_hat(x) D(hat(x))||_2 - 1)^2]$). WGAN provides more stable training and a meaningful loss curve that correlates with sample quality.

=== Diffusion Model Loss

Ho et al. (2020) show that the optimal training objective for a denoising diffusion model simplifies to predicting the noise added at each timestep:

$ cal(L)_"simple" = EE_(t, x_0, epsilon) [||epsilon - epsilon_theta (sqrt(bar(alpha)_t) x_0 + sqrt(1 - bar(alpha)_t) epsilon, t)||^2], $

where $epsilon tilde cal(N)(0, I)$ is the true noise and $epsilon_theta$ is the neural network. This is a weighted MSE over noise predictions at all timesteps. The $v$-prediction parameterization (Salimans & Ho 2022) instead predicts $v = sqrt(bar(alpha)_t) epsilon - sqrt(1 - bar(alpha)_t) x_0$, which improves numerical stability at low noise levels and is standard in modern latent diffusion models.

== Practical Considerations

=== Loss Scale and Gradient Magnitude

Cross-entropy dominates early training signal because its gradient magnitude is $(hat(p) - y)$: when the model is poorly calibrated (random initialization, $hat(p) approx 1/K$), every example produces a large gradient. MSE on logits, by contrast, produces small gradients until predictions are near-correct. This asymmetry makes cross-entropy a more efficient loss for classification even when accuracy is the ultimate metric.

=== Class Imbalance

Several complementary strategies:

- *Focal loss* ($gamma = 2$): down-weights easy negatives dynamically.
- *Class-weighted CE:* multiply each class loss by $w_k prop 1 / "count"_k$ or $w_k prop sqrt(1 / "count"_k)$ (the latter less aggressive).
- *Oversampling / undersampling:* SMOTE and its variants for tabular data.
- *Threshold calibration:* train on imbalanced data, adjust the decision threshold on a balanced validation set post-hoc. Often the simplest effective fix.

=== Multi-Task Losses

When training jointly on multiple objectives (e.g., detection + segmentation + depth), fixed coefficients require fragile tuning. Kendall et al. (2018) derive a principled weighting from homoscedastic task uncertainty: each task $i$ contributes

$ cal(L) = sum_i 1/(2 sigma_i^2) cal(L)_i + log sigma_i, $

where $sigma_i$ are learned log-variances. Tasks with high uncertainty are automatically down-weighted; the log-$sigma$ regularizer prevents all $sigma_i$ from diverging.

=== Numerical Stability

- *Log-sum-exp trick:* $log sum_j exp(z_j) = m + log sum_j exp(z_j - m)$ where $m = max_j z_j$. Use this whenever computing softmax log-probabilities.
- *Clipping for CE:* clip $hat(p)$ to $[epsilon, 1 - epsilon]$ for small $epsilon$ (e.g., $10^{-7}$) before taking $log$ to avoid $log(0)$.
- *Mixed precision:* accumulate cross-entropy in float32 even when activations are float16; loss spiking under fp16 is often a numerical issue rather than an optimization one.

```python
import torch
import torch.nn.functional as F

def stable_cross_entropy(logits, targets, label_smoothing=0.0):
    """Cross-entropy with label smoothing and numerical stability."""
    # F.cross_entropy uses log-sum-exp internally
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)

def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """Binary focal loss (Lin et al. 2017)."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)
    return (alpha * (1 - p_t) ** gamma * bce).mean()

def nt_xent(z1, z2, temperature=0.07):
    """NT-Xent loss for a batch of N positive pairs (SimCLR)."""
    N = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)  # (2N, D)
    sim = z @ z.T / temperature                          # (2N, 2N)
    # mask out self-similarities
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, float("-inf"))
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z.device)
    return F.cross_entropy(sim, labels)
```

== Further Reading

Lin, T.-Y. et al. (2017). "Focal Loss for Dense Object Detection." ICCV. _(RetinaNet; focal loss for class imbalance.)_

van den Oord, A. et al. (2018). "Representation Learning with Contrastive Predictive Coding." arXiv. _(InfoNCE; mutual information lower bound.)_

Chen, T. et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations." ICML. _(SimCLR; NT-Xent loss.)_

Zbontar, J. et al. (2021). "Barlow Twins: Self-Supervised Learning via Redundancy Reduction." ICML. _(Decorrelation loss; collapse-free SSL without negatives.)_

Kendall, A., Gal, Y., Cipolla, R. (2018). "Multi-Task Learning Using Uncertainty to Weigh Losses in Deep Learning." CVPR. _(Homoscedastic uncertainty weighting.)_

Wang, T., Isola, P. (2020). "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere." ICML. _(Decomposition of contrastive loss; temperature analysis.)_

Ho, J. et al. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS. _(DDPM; simplified noise-prediction loss.)_

Bardes, A., Ponce, J., LeCun, Y. (2022). "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning." ICLR. _(VICReg; three-term non-contrastive SSL loss.)_

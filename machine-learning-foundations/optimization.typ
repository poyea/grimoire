= Optimization

Training a model is, mechanically, the act of running a numerical optimizer on a non-convex objective. The optimizer determines convergence rate, final loss, generalization (yes, really — implicit regularization is real), memory footprint, and how robust training is to hyperparameter choice. This chapter covers the optimization machinery actually used in modern ML: SGD and its momentum variants, Adam and its descendants, second-order and preconditioned methods, learning-rate schedules, and the recent wave of structured-curvature optimizers (Shampoo, Muon, Sophia).

*See also:* _Deep Learning Fundamentals_ (backprop produces the gradients), _Linear Algebra for ML_ (matrix decompositions for second-order methods), _Pretraining_ (large-scale optimizer choices), _ML Workload Optimization on GPUs_ (memory/throughput tradeoffs).

== Convex Optimization Primer

A function $f : RR^n -> RR$ is *convex* if for all $x, y$ and $t in [0, 1]$,

$ f(t x + (1 - t) y) <= t f(x) + (1 - t) f(y). $

Equivalent first-order: $f(y) >= f(x) + nabla f(x)^top (y - x)$. Equivalent second-order: $nabla^2 f succ.eq 0$. A function is *$mu$-strongly convex* if $nabla^2 f succ.eq mu I$ and *$L$-smooth* if $parallel nabla f(x) - nabla f(y) parallel <= L parallel x - y parallel$.

The condition number is $kappa = L \/ mu$. Gradient descent on a strongly convex, smooth objective with step size $1 \/ L$ converges as

$ f(x_k) - f^* <= (1 - 1 \/ kappa)^k (f(x_0) - f^*). $

Nesterov's accelerated gradient improves this to $(1 - 1 \/ sqrt(kappa))^k$ — a quadratic improvement that motivated all modern momentum methods.

Deep networks are non-convex. But the local behavior near critical points, the loss landscape geometry of overparameterized models, and the analysis tools (Lyapunov functions, descent lemmas) carry over with modifications.

== Gradient Descent

The basic update is

$ x_(k+1) = x_k - eta nabla f(x_k). $

Choosing $eta$ is the central practical question. Too small: slow convergence. Too large: divergence. For $L$-smooth $f$, any $eta < 2 / L$ guarantees descent; $eta = 1 / L$ is optimal in the worst case.

```python
import numpy as np

def gradient_descent(grad_fn, x0, lr, n_iter):
    x = x0.copy()
    for k in range(n_iter):
        g = grad_fn(x)
        x -= lr * g
    return x
```

== Stochastic Gradient Descent

When the loss is a finite sum $f(x) = 1/n sum_i f_i (x)$, computing the full gradient is $O(n)$ per step. *SGD* samples a mini-batch and uses an unbiased gradient estimate. For convex problems and decreasing step size $eta_k = O(1 \/ k)$, SGD converges as $O(1 \/ sqrt(k))$ — the cost of stochasticity is one square root.

The minibatch gradient has variance $sigma^2 \/ B$ for batch size $B$, so doubling $B$ halves the variance. The linear-scaling rule (Goyal et al. 2017) says one can roughly scale $eta$ linearly with $B$ until a critical batch size beyond which returns diminish.

== Momentum

Polyak's *heavy ball* method adds an inertia term:

$ v_(k+1) = beta v_k + nabla f(x_k), quad x_(k+1) = x_k - eta v_(k+1). $

Nesterov's accelerated gradient evaluates the gradient at the *lookahead* point $x_k - eta beta v_k$ instead, achieving the optimal rate for first-order methods on smooth convex problems.

```python
def sgd_momentum(params, grads, velocities, lr=0.01, momentum=0.9):
    for p, g, v in zip(params, grads, velocities):
        v *= momentum
        v += g
        p -= lr * v
```

== Adaptive Methods

The Adam family (Kingma–Ba 2015) tracks per-parameter first and second moments of the gradient and divides by the square root of the second moment, effectively giving each parameter its own learning rate.

```python
import numpy as np

class Adam:
    def __init__(self, shape, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = 0

    def step(self, param, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad**2
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param
```

#table(
  columns: 4,
  [*Optimizer*], [*Update rule (per param)*], [*State per param*], [*Notes*],
  [SGD], [$- eta g$], [$0$], [Baseline],
  [SGD+momentum], [$- eta v$, $v = beta v + g$], [1 vec], [Default for CNNs],
  [RMSProp], [$- eta g \/ sqrt(v)$, $v = beta v + (1-beta) g^2$], [1 vec], [Hinton's lectures],
  [Adam], [bias-corrected moments], [2 vec], [Default for transformers],
  [AdamW], [Adam + decoupled weight decay], [2 vec], [Loshchilov–Hutter 2019],
  [Lion], [$"sign"(beta_1 m + (1-beta_1) g)$], [1 vec], [Google 2023; signed updates],
  [Lamb], [Adam with layer-wise trust ratio], [2 vec], [Large-batch BERT],
  [Adafactor], [Factored second moment], [$sqrt(p) + sqrt(q)$], [Memory-efficient],
  [Shampoo], [Kronecker-factored preconditioner], [2 PSD blocks], [Order-2 structure],
  [Muon], [Newton-Schulz orthogonalization of momentum], [1 mat], [Jordan 2024],
  [Sophia], [Clipped Hessian-vector preconditioner], [2 vec], [Liu et al. 2023],
)

=== AdamW versus Adam

The "weight decay" added to Adam by stuffing $lambda x$ into the gradient interacts badly with adaptive learning rates: regularization strength varies per-parameter. Loshchilov & Hutter's AdamW *decouples* weight decay from the gradient step:

$ x_(k+1) = x_k - eta dot m_hat / (sqrt(v_hat) + epsilon) - eta lambda x_k. $

For transformer training this single change often moves the optimal $lambda$ by an order of magnitude and improves generalization. AdamW is now the default for LLM pretraining (`llm/pretraining.typ`).

=== Lion and Signed Updates

Chen et al. (2023) discovered Lion via program search: the update is the *sign* of an exponential moving average of gradients,

$ x_(k+1) = x_k - eta dot "sign"(beta_1 m + (1 - beta_1) g) - eta lambda x_k. $

Lion uses half the memory of Adam (one moment instead of two) and matches or beats it on many tasks; it requires a smaller learning rate (typically 1/3 of Adam's) and benefits from larger batch sizes.

=== Shampoo and Kronecker-Factored Preconditioners

For a parameter matrix $W in RR^(m times n)$, the natural-gradient direction uses the inverse of the full $m n times m n$ Fisher. Shampoo (Gupta et al. 2018) approximates this with the Kronecker product

$ tilde(G) = L^(-1/4) G R^(-1/4), quad L = sum_k G_k G_k^top, R = sum_k G_k^top G_k, $

where the inverse roots are computed via Newton-Schulz iteration. The storage cost is $O(m^2 + n^2)$ instead of $O(m^2 n^2)$ — practical even for large layers. Distributed Shampoo (Anil et al. 2020) is used in Google's production training.

=== Muon

Muon (Jordan 2024) replaces Adam's second-moment normalization with an *orthogonalization* of the momentum buffer via Newton–Schulz. For a momentum matrix $M$, compute $O = "NewtonSchulz"(M)$ such that $O$ is approximately orthogonal, then update with $O$ scaled appropriately. Muon achieves state-of-the-art performance on transformer pretraining at lower wall-clock cost than AdamW for the same loss.

```python
import torch

def newton_schulz(G, steps=5, eps=1e-7):
    # Approximate orthogonalization of G via the quintic iteration
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32)
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.to(G.dtype)
```

== Second-Order Methods

Newton's method uses the Hessian: $x_(k+1) = x_k - H^(-1) g$. For a quadratic this converges in one step; for general smooth functions it has local quadratic convergence. The catch is forming and inverting $H$ — $O(n^3)$ — which is hopeless for neural networks with billions of parameters.

Approximations make Newton tractable:

- *Gauss-Newton:* for least-squares, $H approx J^top J$ where $J$ is the Jacobian.
- *L-BFGS:* maintain a low-rank approximation of $H^(-1)$ from recent gradient differences. Works well on convex problems; struggles with stochasticity.
- *Hessian-free / Krylov:* solve $H d = -g$ via conjugate gradient using only Hessian-vector products $H v$, which cost the same as one extra backward pass.
- *K-FAC:* Kronecker-factored approximation to the Fisher for layer-wise structure.
- *Sophia:* diagonal Hessian estimate (Hutchinson) with clipping, computed every $k$ steps.

```python
import torch

def hvp(loss, params, vec):
    """Hessian-vector product: H @ vec, computed without forming H."""
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat = torch.cat([g.reshape(-1) for g in grads])
    dot = flat @ vec
    hv = torch.autograd.grad(dot, params, retain_graph=True)
    return torch.cat([h.reshape(-1) for h in hv])
```

== Learning-Rate Schedules

The schedule often matters more than the optimizer.

#table(
  columns: 3,
  [*Schedule*], [*Form*], [*Use*],
  [Constant], [$eta_k = eta$], [Theoretical analyses; rarely best in practice],
  [Step decay], [$eta dot gamma^(floor(k \/ s))$], [Classic CNN training],
  [Cosine], [$eta_min + 1/2 (eta_max - eta_min)(1 + cos(pi k / K))$], [Transformer pretraining],
  [Warmup + cosine], [linear warmup then cosine], [LLM pretraining standard],
  [Linear decay], [$eta_max (1 - k/K)$], [Fine-tuning, RLHF],
  [WSD (warmup-stable-decay)], [const middle, decay at end], [Chinchilla-style multi-stage],
  [1cycle], [triangular up then down], [Smith 2017; fast CV training],
)

Warmup matters at large batch sizes: at step 0, Adam's variance estimate $v$ is poorly initialized, leading to inflated effective learning rates. Linear warmup over a few thousand steps gives $v$ time to stabilize.

```python
import math

def warmup_cosine(step, total, warmup, lr_max, lr_min=0.0):
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
```

== Gradient Clipping

Exploding gradients are common in RNNs, RL, and large transformers. *Global-norm clipping* computes the Euclidean norm of the full gradient vector (concatenating all parameter gradients) and rescales uniformly:

$ parallel g parallel_2 = sqrt(sum_i g_i^2), quad g <- g dot min(1, c \/ parallel g parallel_2). $

The operation preserves direction while bounding magnitude. Global-norm clipping is preferred over per-parameter clipping because it does not distort the *relative* size of different parameter gradients.

*Why exploding gradients happen:* in RNNs and transformers, gradients propagate through repeated matrix multiplications over sequence length $T$. If the spectral radius of the recurrent weight matrix exceeds 1, gradients grow exponentially in $T$. In transformers, large attention logits cause saturated softmaxes with near-zero gradients everywhere except the argmax, causing sudden large updates when the attention pattern shifts.

*Adaptive Gradient Clipping (AGC)* (Brock et al. 2021, NFNet paper): instead of a global norm threshold, AGC clips each layer's gradient relative to the layer's parameter norm:

$ g_l <- g_l dot min(1, lambda dot parallel W_l parallel_F \/ parallel g_l parallel_F), $

where $lambda$ is a hyperparameter (typically 0.01-0.1) and norms are per-layer Frobenius norms. This adapts the threshold per layer so that layers with small weights (often early layers in deep nets) are clipped more conservatively. AGC was the key ingredient that let NFNets train without BatchNorm at ImageNet scale.

```python
def agc_clip(params, grads, lambda_agc=0.01, eps=1e-3):
    for p, g in zip(params, grads):
        p_norm = p.norm(p="fro").clamp(min=eps)
        g_norm = g.norm(p="fro").clamp(min=eps)
        ratio = p_norm / g_norm
        scale = (lambda_agc * ratio).clamp(max=1.0)
        g.mul_(scale)
```

Recommended `clip_norm` values: 1.0 for standard transformer/RNN training; 0.1-0.5 for RL where gradients are particularly noisy; 0.01 (as $lambda$) for AGC. Monitor the *fraction of clipped steps* — if more than 10-20% of steps are clipped, the learning rate is too high rather than the clip threshold too low.

== Implicit Regularization

SGD does not just minimize the loss — it preferentially finds particular minima. Several empirical and theoretical observations:

- *Flat minima generalize better* (Hochreiter–Schmidhuber, Keskar et al.). Large-batch training tends to find sharper minima, hurting generalization.
- *SGD noise as regularization.* The noise covariance favors directions of low curvature; the resulting stationary distribution concentrates on flat regions.
- *Edge of stability* (Cohen et al. 2021). Gradient descent in deep networks typically operates at the edge where the top Hessian eigenvalue equals $2 \/ eta$, oscillating but making slow progress.

These phenomena explain why optimizer choice affects test accuracy, not just training loss — a fact that is critical when comparing AdamW, Lion, and Muon on the same model.

== Large-Batch Training

The *gradient noise scale* (McCandlish et al. 2018) predicts the critical batch size beyond which efficiency diminishes:

$ B_("crit") = (tr(H Sigma)) / (g^top H g), $

where $Sigma$ is the gradient covariance. Above $B_("crit")$, doubling batch size only modestly improves wall-clock time. LAMB (You et al. 2020) made batch size 32k practical for BERT by adding a layer-wise trust ratio that prevents large updates relative to parameter norm.

== Distributed Optimization

For data-parallel training, each worker computes a local gradient on its mini-batch; gradients are averaged across workers via ring all-reduce. In a ring topology with $N$ workers, all-reduce sends each element exactly twice (once around the ring to reduce, once to broadcast), making the total data transferred $2 (N-1)/N times P$ bytes for $P$ parameter bytes. This is bandwidth-optimal — the cost does not grow with $N$ once the ring is full. A single all-reduce for a 7B-parameter model in fp32 transfers $2 times 7 times 10^9 times 4 approx 56$ GB; at 400 Gbit/s inter-node bandwidth this takes $approx 1.1$ s, making communication overlap essential.

*Communication-compute overlap:* modern frameworks (PyTorch DDP, JAX pjit) bucket gradients and start all-reduce on already-computed buckets during the backward pass, overlapping communication of earlier layers with computation of later ones. Effective overlap requires that bucket boundaries align with natural layer boundaries and that the all-reduce backend (NCCL, RCCL) supports asynchronous streams.

=== ZeRO Memory Partitioning

Standard data parallelism replicates all state (parameters $Psi$, gradients $Psi$, optimizer states $2 Psi$ for Adam) on every worker. ZeRO (Rajbhandari et al. 2020) partitions state across $N$ workers in three stages:

#table(
  columns: 3,
  [*Stage*], [*What is partitioned*], [*Memory per device*],
  [ZeRO-1], [Optimizer states], [$Psi + Psi + 2 Psi \/ N$],
  [ZeRO-2], [+ Gradients], [$Psi + Psi \/ N + 2 Psi \/ N$],
  [ZeRO-3], [+ Parameters], [$(Psi + Psi + 2 Psi) \/ N$],
)

ZeRO-3 divides total memory by $N$ but requires an all-gather of each layer's parameters before the forward pass and a reduce-scatter of gradients after the backward pass — roughly doubling the communication volume vs. ZeRO-1. The tradeoff is controlled by `offload_optimizer` and `contiguous_gradients` flags in DeepSpeed.

=== FSDP vs DDP

PyTorch FSDP (Fully Sharded Data Parallel) is the PyTorch-native ZeRO-3 equivalent. Key operational differences from DDP:

- *Memory:* FSDP shards parameters; DDP replicates. FSDP enables training models that do not fit on a single device.
- *Communication:* FSDP adds all-gathers in the forward pass; the total bytes per step is higher. For small models where memory is not the bottleneck, DDP is faster.
- *Gradient checkpointing interaction:* FSDP and activation checkpointing compose cleanly; recomputed activations are sharded on recompute.

=== Gradient Compression

*PowerSGD* (Vogels et al. 2019) approximates the gradient matrix with a low-rank factorization (rank $r$), reducing communication from $O(m n)$ to $O((m + n) r)$ per layer. Error feedback accumulates the approximation residual and adds it to the next step's gradient, preventing long-run drift. Top-K sparsification transmits only the $K$ largest-magnitude gradients; this is particularly effective for sparse embedding layers. Communication savings of 10-100× are achievable at modest accuracy loss.

*Gradient compression* (PowerSGD, Top-K sparsification) reduces communication at the cost of a biased estimator. *Local SGD* communicates every $k$ steps, which reduces synchronization frequency but requires careful learning rate scaling to avoid consensus drift. See `llm/pretraining.typ` for the full pipeline-parallel + tensor-parallel stack.

== Convergence Diagnostics

A few signals that something is wrong:

- *Loss spikes* — check gradient clipping, learning rate, fp16 underflow.
- *Loss plateau then sudden drop* — symbolic of grokking or warmup ending; usually benign.
- *Train loss decreasing, eval loss increasing* — overfitting; add regularization or stop earlier.
- *NaN at step 17* — almost always softmax overflow or fp16 underflow; turn on `detect_anomaly` in PyTorch.

== Further Reading

Nocedal, J., Wright, S. (2006). _Numerical Optimization_, 2nd ed. Springer.

Boyd, S., Vandenberghe, L. (2004). _Convex Optimization_. Cambridge.

Kingma, D., Ba, J. (2015). "Adam: A Method for Stochastic Optimization." ICLR.

Loshchilov, I., Hutter, F. (2019). "Decoupled Weight Decay Regularization." ICLR (AdamW).

Goyal, P. et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." arXiv.

McCandlish, S. et al. (2018). "An Empirical Model of Large-Batch Training." arXiv.

Gupta, V., Koren, T., Singer, Y. (2018). "Shampoo: Preconditioned Stochastic Tensor Optimization." ICML.

Anil, R. et al. (2020). "Scalable Second Order Optimization for Deep Learning." arXiv.

Chen, X. et al. (2023). "Symbolic Discovery of Optimization Algorithms" (Lion). NeurIPS.

Liu, H. et al. (2023). "Sophia: A Scalable Stochastic Second-Order Optimizer." arXiv.

Jordan, K. et al. (2024). "Muon: An Optimizer for Hidden Layers in Neural Networks." Blog/manuscript.

Cohen, J. et al. (2021). "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability." ICLR.

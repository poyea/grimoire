= Mixture of Experts

A dense transformer scales every parameter with every token: doubling the parameter count doubles the FLOPs per forward pass. *Mixture of Experts (MoE)* breaks this coupling — total parameters grow but *active* parameters per token stay fixed by sparsely activating a small subset of "expert" feed-forward sublayers. The result is models with 100B–1T total parameters running at the FLOP cost of a 10–50B dense model. This chapter covers routing, load balancing, capacity factors, the training and inference systems implications, and the design choices of Switch, GShard, GLaM, Mixtral, and DeepSeek-V3.

*See also:* _Transformer Architecture_ (the FFN sublayer becomes an expert), _Pretraining_ (token-routing affects parallelism), _Inference Optimization_ (expert dispatch is a serving bottleneck).

== The Core Idea

In a standard transformer block:

$ "FFN"(x) = W_2 sigma(W_1 x) $

Replace this single FFN with $E$ parallel FFNs ("experts") $\{f_1, ..., f_E\}$ and a *router* $g(x) in RR^E$. The block becomes:

$ "MoE"(x) = sum_(i=1)^E g_i(x) f_i(x) $

If $g$ is dense (e.g., softmax), this is just an ensemble — no FLOP savings. The trick is *sparsity*: route each token to only the top-$k$ experts (typically $k=1$ or $k=2$):

$ g_i(x) = cases("softmax"(W_g x)_i & "if " i in "TopK"(W_g x), 0 & "otherwise") $

A token activates $k$ experts out of $E$. Active params per token scale as $O(k / E)$ of total expert params. Switch Transformer ($k=1$, $E=128$) at 1.6T total params runs at ~7B active params per token.

== Routing

=== Top-1 (Switch)

```python
import torch, torch.nn.functional as F

class SwitchMoE(torch.nn.Module):
    def __init__(self, d_model, d_ff, n_experts):
        super().__init__()
        self.gate    = torch.nn.Linear(d_model, n_experts, bias=False)
        self.experts = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(d_model, d_ff),
                torch.nn.GELU(),
                torch.nn.Linear(d_ff, d_model),
            ) for _ in range(n_experts)
        )

    def forward(self, x):                 # x: [B, T, d]
        logits = self.gate(x)             # [B, T, E]
        probs  = logits.softmax(-1)
        weight, idx = probs.max(-1)       # top-1

        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            mask = (idx == e)
            if mask.any():
                xe = x[mask]
                out[mask] = expert(xe) * weight[mask].unsqueeze(-1)
        return out
```

The `weight *` multiplication keeps the router *differentiable* via the chosen expert's logit; tokens are routed by `argmax` (non-differentiable) but the *combine weight* carries the gradient.

=== Top-2 (GShard, Mixtral)

Top-2 routing sends each token to its two highest-scoring experts and combines:

$ y = "weight"_1 f_(i_1)(x) + "weight"_2 f_(i_2)(x) $

where weights are renormalized over the top-2 logits. Top-2 doubles active FLOPs but yields better quality per total parameter — the experts can "specialize but cooperate." Mixtral 8×7B (47B total, ~13B active) uses top-2.

=== Top-$k$ with Sigmoid Gating (DeepSeek-V3)

DeepSeek-V3 replaces softmax with sigmoid gating across experts:

$ g_i = "Sigmoid"(W_g x)_i, quad y = sum_(i in "TopK"(g)) g_i f_i(x) $

Sigmoid gating is computed independently per expert — there is no inter-expert competition through normalization. This decouples *routing quality* (which experts are most relevant) from *combination weight*. Empirically it reduces gradient interference between experts.

== Load Balancing

Naïve top-$k$ routing collapses: a few experts dominate, the rest are unused, and useful capacity is wasted. The two standard interventions:

=== Auxiliary Loss

Add a load-balancing term to the training loss. Let $f_i$ be the fraction of tokens routed to expert $i$ in a batch and $P_i$ be the mean router probability for expert $i$:

$ cal(L)_"aux" = alpha times E times sum_(i=1)^E f_i times P_i $

Minimizing $cal(L)_"aux"$ pushes both quantities toward uniform $1/E$. The product (not the squared sum) is differentiable through $P_i$ while still penalizing imbalance through $f_i$.

```python
def aux_loss(probs, idx, n_experts, alpha=0.01):
    # probs: [B*T, E]; idx: [B*T]
    one_hot = F.one_hot(idx, n_experts).float()
    f = one_hot.mean(dim=0)            # token fraction per expert
    P = probs.mean(dim=0)              # mean prob per expert
    return alpha * n_experts * (f * P).sum()
```

Typical $alpha in [0.001, 0.01]$.

=== Auxiliary-Loss-Free Balancing (DeepSeek-V3)

DeepSeek-V3 proposed *bias-based balancing*: maintain a per-expert *bias* $b_i$ added to the router logits *only for routing* (not combination):

$ "idx" = "TopK"(W_g x + b) $

After each step, increment $b_i$ for under-used experts and decrement for over-used. The bias does not enter the loss, so the gradient signal stays clean while routing distribution stays uniform. Reported to improve quality at fixed FLOPs vs. aux-loss approaches.

== Capacity and Expert Dropping

Even with load balancing, a *batch* routes tokens unevenly. If expert $i$ receives more tokens than it has capacity for, the excess must be dropped (zero output) or rerouted to a backup expert.

*Capacity factor* $C$ controls per-expert buffer:

$ "capacity" = C times T / E $

where $T$ is the number of tokens in the batch. $C = 1.0$ means perfect balancing has no overflow; $C = 1.25$ adds slack. Higher $C$ wastes FLOPs (idle expert slots); lower $C$ drops more tokens.

Switch Transformer paper recommended $C = 1.0$ at training and $C = 2.0$ at evaluation (no auxiliary loss to enforce balance at test time).

== Expert Parallelism

In a multi-GPU setup, experts can be sharded across devices: each device holds a subset of experts. Routing then requires *all-to-all* communication — every device sends its tokens to whichever device holds their target expert, then receives results back.

```
Step 1: per device, compute router → indices
Step 2: all-to-all: shuffle tokens to expert-holding devices
Step 3: each device runs its local experts
Step 4: all-to-all: shuffle results back to source devices
Step 5: combine with router weights
```

All-to-all latency scales with the number of expert-holding devices and dominates large-$E$ training. NVIDIA's `NCCL` and AMD's `RCCL` implement bandwidth-optimal hierarchical all-to-all. Switch Transformer's released code uses Mesh-TensorFlow; modern training uses Megatron-LM's MoE module, ColossalAI's Mixture-of-Experts, or DeepSpeed-MoE.

== Inference

Serving MoE is *harder* than dense:

- *Memory footprint:* all expert weights must be resident (or paged in/out). Mixtral 8×7B = 94 GiB in bf16; a single A100/H100 (80 GiB) cannot hold it.
- *Batched routing:* in a serving batch, different sequences route to different experts. Achieving high MFU requires *grouped GEMM* kernels (FasterTransformer's `grouped_matmul`, vLLM's MoE kernel) that batch tokens routed to the same expert.
- *Latency variance:* tokens that route to a heavily-loaded expert wait. Latency P99 is determined by the most-loaded expert per step.

Practical mitigations:

- *Top-$k$ with $k=1$ at inference* (even if trained with $k=2$): halves expert FLOPs.
- *Expert pruning:* drop experts with low utilization; Mixtral-Instruct prunes 1–2 experts at acceptable quality loss.
- *Sub-batch routing:* group tokens by destination expert before issuing kernels; turns an irregular dispatch into regular GEMMs.

== Shared and Routed Experts

DeepSeek-V3 splits each MoE block into *shared experts* (always active for every token, like a small dense FFN) and *routed experts* (activated by the router). The shared experts handle generic features; routed experts specialize. This pattern (also called *residual MoE*) stabilizes training and improves quality at small $k$.

== Notable Models

| Model | Total | Active | $E$ | $k$ | Routing | Notes |
|---|---|---|---|---|---|---|
| Switch Transformer (2021) | 1.6T | 7B | 128 | 1 | aux-loss | Original at-scale demonstration |
| GShard (2020) | 600B | 50B | 2048 | 2 | aux-loss | Multilingual MT |
| GLaM (2021) | 1.2T | 97B | 64 | 2 | aux-loss | Decoder-only; GPT-3 quality at 1/3 the FLOPs |
| Mixtral 8×7B (2023) | 47B | 13B | 8 | 2 | softmax | Open weights; popularized MoE in OSS |
| DeepSeek-V2 (2024) | 236B | 21B | 160 + 2 shared | 6 | aux-loss + bias | First with MLA + MoE |
| DeepSeek-V3 (2024) | 671B | 37B | 256 + 1 shared | 8 | aux-loss-free | SOTA open MoE |
| Qwen 1.5-MoE-A2.7B | 14B | 2.7B | 64 + 4 shared | 4 | shared+routed | Small efficient MoE |

== Pitfalls

=== Router Z-Loss

Router logits can grow without bound, destabilizing the softmax. Switch Transformer adds:

$ cal(L)_"z" = lambda times EE[("logsumexp"(W_g x))^2] $

which penalizes large logit magnitudes. Typical $lambda = 10^(-3)$.

=== Expert Capacity in Distillation

When distilling an MoE teacher into a dense student, the student must capture the *combined* expert output for every token. If the teacher's experts have specialized heavily, the student's dense FFN is insufficient. Distillation losses spike on tokens where the teacher routes to rare experts. Mitigation: re-route at distillation time to use top-2 instead of top-1, smoothing the teacher signal.

=== Throughput Cliff with Long Context

KV cache is per-token, but expert dispatch is also per-token. At long context, attention dominates compute, and the all-to-all communication becomes a smaller share — but expert capacity overflow becomes more likely because the token *distribution* in a long sequence is less uniform than a balanced training batch. Inference systems must size capacity for worst-case sequences, not averages.

== Designing an MoE Block

A practical recipe (DeepSeek-V3 style):

1. Replace the dense FFN with $E_"routed" + E_"shared"$ experts; pick $E_"routed"$ to multiply total params by 4–8× over dense.
2. Sigmoid gate; $k = 6$ to $8$ routed + always-active shared.
3. Aux-loss-free balancing with per-expert bias updated at every step.
4. Router Z-loss with small $lambda$.
5. Capacity factor $C = 1.25$ at training, $C = 1.5$ at inference.
6. Schedule: train dense first for 5–10% of total tokens, then *MoE-ify* by splitting the FFN. Avoids early-training collapse.

== Further Reading

Shazeer, N. et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." ICLR.

Lepikhin, D. et al. (2020). "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding." ICLR 2021.

Fedus, W., Zoph, B., Shazeer, N. (2022). "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." JMLR.

Du, N. et al. (2022). "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts." ICML.

Jiang, A. et al. (2024). "Mixtral of Experts." arXiv:2401.04088.

DeepSeek-AI (2024). "DeepSeek-V3 Technical Report." arXiv:2412.19437.

Zoph, B. et al. (2022). "ST-MoE: Designing Stable and Transferable Sparse Expert Models."

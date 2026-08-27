#import "../template.typ": xref

= Pretraining Infrastructure <pretraining-infrastructure>

*Training at scale:* This chapter covers the optimizer, distributed training, and training stability — the infrastructure side of pretraining. It continues from _Pretraining_ (data pipelines, objectives, scaling laws, mixed precision, gradient checkpointing).

*See also:* #xref("llm", "pretraining", label: "Pretraining") (objectives and memory techniques), #xref("gpu-architecture", "multi-gpu", label: "Multi-GPU Communication and Scaling (GPU Architecture volume)"), #xref("gpu-architecture", "ml-workloads", label: "ML Workload Optimization on GPUs (GPU Architecture volume)").

== Optimizer

=== AdamW

Adam (Kingma & Ba, 2015) maintains per-parameter first and second moment estimates. AdamW (Loshchilov & Hutter, 2019) decouples weight decay from the gradient update, which is theoretically cleaner and empirically better than the L2-regularization-as-Adam variant.

*Update rule* for parameter $theta_t$ at step $t$:

$ m_t &= beta_1 m_(t-1) + (1 - beta_1) g_t \
v_t &= beta_2 v_(t-1) + (1 - beta_2) g_t^2 \
hat(m)_t &= m_t / (1 - beta_1^t) \
hat(v)_t &= v_t / (1 - beta_2^t) \
theta_(t+1) &= theta_t - eta (hat(m)_t / (sqrt(hat(v)_t) + epsilon)) - eta lambda theta_t $

The last term $eta lambda theta_t$ is the decoupled weight decay. Standard hyperparameters for LLM pretraining:

#table(
  columns: (auto, auto, auto),
  [*Hyperparameter*], [*Symbol*], [*Typical value*],
  [Learning rate],    [$eta$],          [$3 times 10^(-4)$ (7B), $1.5 times 10^(-4)$ (70B)],
  [Beta 1],           [$beta_1$],       [0.9],
  [Beta 2],           [$beta_2$],       [0.95],
  [Epsilon],          [$epsilon$],      [$10^(-8)$ (use $10^(-5)$ with BF16)],
  [Weight decay],     [$lambda$],       [0.1],
  [Gradient clip],    [$g_"max"$],      [1.0],
  [Warmup steps],     [—],              [2000 (7B–70B)],
  [Total steps],      [—],              [$tilde 10^6$ for 1T tokens, batch 2048],
)

=== Learning Rate Schedule: Cosine with Warmup

$ eta(t) = cases(
  eta_"max" dot t / t_"warm" & "if" t lt t_"warm",
  eta_"min" + 1/2 (eta_"max" - eta_"min")(1 + cos(pi (t - t_"warm") / (t_"total" - t_"warm"))) & "otherwise"
) $

Typical values: $t_"warm" = 2000$ steps, $eta_"min" = eta_"max" / 10$.

```python
import math

def cosine_with_warmup(step: int, max_lr: float, min_lr: float,
                       warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

# Integrate with PyTorch scheduler
from torch.optim.lr_scheduler import LambdaLR

def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    max_lr = optimizer.param_groups[0]["lr"]
    min_lr = max_lr / 10.0
    def lr_lambda(step: int) -> float:
        return cosine_with_warmup(step, max_lr, min_lr,
                                  warmup_steps, total_steps) / max_lr
    return LambdaLR(optimizer, lr_lambda)
```

=== AdamW Weight Update Kernel (C++)

In large-scale training, the optimizer step is memory-bandwidth bound. A fused CUDA/C++ kernel that updates all states in a single pass over memory reduces kernel launch overhead and improves cache utilization.

```cpp
// Fused AdamW weight update — operates on flattened parameter arrays.
// Compile with: nvcc -O3 -arch=sm_90 adamw_kernel.cu
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

__global__ void adamw_update_kernel(
    float*        __restrict__ master_w,   // FP32 master weights
    __nv_bfloat16* __restrict__ model_w,   // BF16 model weights (for forward)
    float*        __restrict__ grad,       // FP32 gradient
    float*        __restrict__ m,          // first moment (FP32)
    float*        __restrict__ v,          // second moment (FP32)
    const float   lr,
    const float   beta1,
    const float   beta2,
    const float   eps,
    const float   weight_decay,
    const float   bias_corr1,              // 1 - beta1^t
    const float   bias_corr2,              // 1 - beta2^t
    const int64_t n_elems)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elems) return;

    float g = grad[idx];
    float m_i = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_i = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = m_i;
    v[idx] = v_i;

    float m_hat = m_i / bias_corr1;
    float v_hat = v_i / bias_corr2;

    float w = master_w[idx];
    w = w - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w);
    master_w[idx] = w;
    model_w[idx]  = __float2bfloat16(w);   // downcast to BF16 for forward pass
}

// Host launcher
void adamw_step(/* ... pointers ...*/, int step, int64_t n_elems) {
    float bias_corr1 = 1.0f - powf(0.9f,  (float)step);
    float bias_corr2 = 1.0f - powf(0.95f, (float)step);
    int threads = 512;
    int blocks  = (int)((n_elems + threads - 1) / threads);
    adamw_update_kernel<<<blocks, threads>>>(
        /* ... */,
        bias_corr1, bias_corr2, n_elems);
}
```

```python
# PyTorch AdamW (standard usage)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
    fused=True,          # uses CUDA fused kernel when available
)

# Gradient clipping before optimizer step
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)   # free gradient memory immediately
```

== Distributed Training

*Cross-reference:* _Multi-GPU Communication and Scaling (GPU Architecture volume)_ covers the underlying communication primitives (NCCL all-reduce, all-gather, reduce-scatter, NVLink/InfiniBand bandwidth).

Large language models require distributing computation across tens to thousands of GPUs. Three orthogonal parallelism strategies are combined in practice.

=== Data Parallel Training (DDP)

Each GPU holds a *full copy* of the model. The global batch is split: each GPU processes a _micro-batch_, computes gradients independently, then gradients are _all-reduced_ (summed and divided) across all GPUs before the optimizer step. After all-reduce, every GPU has identical gradients and performs an identical optimizer step.

$ g_"global" = (1/K) sum_(k=1)^K g^((k)) $

All-reduce cost: $2(K-1)/K times P times "sizeof(float)"$ bytes transmitted per GPU for ring all-reduce over $K$ GPUs and $P$ parameters.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# Initialize process group (called once per process)
dist.init_process_group(backend="nccl")    # NCCL for GPU-GPU communication
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = MyTransformer().cuda(local_rank)
model = DDP(model, device_ids=[local_rank],
            find_unused_parameters=False,   # True only if needed; has overhead
            gradient_as_bucket_view=True)   # reduce memory copies

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                               betas=(0.9, 0.95), weight_decay=0.1)

for batch in dataloader:   # dataloader uses DistributedSampler
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    loss.backward()        # DDP hooks trigger all-reduce here
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

dist.destroy_process_group()
```

Launch: `torchrun --nproc_per_node=8 train.py`

*Limitation:* DDP requires each GPU to hold a full model copy. For a 70B model (FP32 master weights + states $approx 560$ GB), a single DDP replica needs 7 $times$ 80 GB A100s — impractical to replicate across hundreds of GPUs.

=== ZeRO / FSDP: Sharding Parameters, Gradients, and Optimizer States

ZeRO (Zero Redundancy Optimizer, Rajbhandari et al., 2020) eliminates the memory redundancy in DDP by sharding model state across data-parallel ranks. PyTorch's _Fully Sharded Data Parallel_ (FSDP) implements ZeRO-3.

*ZeRO stages:*

#table(
  columns: (auto, auto, auto, auto),
  [*Stage*], [*What is sharded*], [*Memory per GPU (70B)*], [*Extra communication*],
  [DDP (stage 0)], [Nothing], [$tilde 560$ GB], [All-reduce gradients],
  [ZeRO-1],        [Optimizer states], [$tilde 280$ GB], [All-reduce gradients],
  [ZeRO-2],        [Optimizer states + gradients], [$tilde 140$ GB], [Reduce-scatter grads],
  [ZeRO-3 / FSDP], [Optimizer states + gradients + parameters], [$tilde 20$ GB], [All-gather params + reduce-scatter grads],
)

With ZeRO-3 / FSDP, each GPU holds $1/K$ of every tensor. Before a forward or backward pass through a given layer, the full layer weights are reconstructed via an _all-gather_; they are discarded immediately after use. Gradients are aggregated via _reduce-scatter_ (each rank keeps its shard of the reduced gradients).

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

# BF16 mixed precision policy
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,     # shard params in BF16
    reduce_dtype=torch.float32,     # reduce grads in FP32 for precision
    buffer_dtype=torch.bfloat16,
)

# Wrap each transformer block independently for fine-grained sharding
auto_wrap = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock}
)

model = FSDP(
    MyTransformer(),
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO-3
    mixed_precision=mp_policy,
    auto_wrap_policy=auto_wrap,
    device_id=local_rank,
    use_orig_params=True,    # required for parameter groups / weight decay masking
)

# Optimizer and training loop are identical to DDP
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4,
                               betas=(0.9, 0.95), weight_decay=0.1)
```

*Activation checkpointing with FSDP:*

```python
from torch.distributed.fsdp import checkpoint_wrapper, CheckpointImpl

for layer in model.model.layers:
    # Wrap each layer for both FSDP sharding and gradient checkpointing
    checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
```

=== Tensor Parallelism

Tensor parallelism (Shoeybi et al., 2019 — Megatron-LM) splits individual weight matrices across GPUs. It reduces per-GPU memory proportionally to the tensor-parallel degree $T$ and enables within-node, high-bandwidth parallelism over NVLink.

*Column-parallel linear (e.g., QKV projection, first FFN layer):*

$ Y = X W^T , quad W in RR^(d times k) $

Split $W$ column-wise: each GPU $i$ holds $W_i in RR^(d times k/T)$ and computes $Y_i = X W_i^T$. No communication is needed after this operation — $Y_i$ are independent and passed to the next layer.

*Row-parallel linear (e.g., output projection, second FFN layer):*

Split $W$ row-wise: each GPU $i$ holds $W_i in RR^(k/T times d)$ and receives its corresponding shard $X_i$ of the input. Computes partial output $Y_i = X_i W_i^T$. An _all-reduce_ combines the partial sums: $Y = sum_i Y_i$.

In an MLP block (column-parallel GELU row-parallel), only two all-reduces are needed per layer — one after the attention output projection, one after the FFN output projection.

```python
# Megatron-LM style column-parallel linear (conceptual)
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp_group):
        super().__init__()
        T = dist.get_world_size(tp_group)
        assert out_features % T == 0
        self.local_out = out_features // T
        self.weight = nn.Parameter(torch.empty(self.local_out, in_features))
        self.tp_group = tp_group
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each GPU computes its output shard independently
        return F.linear(x, self.weight)    # no communication needed here


class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp_group):
        super().__init__()
        T = dist.get_world_size(tp_group)
        assert in_features % T == 0
        self.local_in = in_features // T
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in))
        self.tp_group = tp_group
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partial = F.linear(x, self.weight)
        dist.all_reduce(partial, group=self.tp_group)   # sum partial outputs
        return partial
```

=== Combining Parallelism Strategies

Production LLM training typically combines all three:

#table(
  columns: (auto, auto, auto),
  [*Strategy*], [*Typical degree*], [*Communication domain*],
  [Tensor parallel (TP)],    [$T = 8$],           [Within node (NVLink)],
  [Pipeline parallel (PP)],  [$P = 4$–$8$],       [Between nodes (IB or NVLink)],
  [Data parallel (ZeRO-3)],  [$D = N_"GPU" / (T times P)$], [Between nodes],
)

For 8192 H100s (1024 nodes of 8), a typical 3D config is $T=8, P=8, D=128$. The _effective global batch size_ is $D times B_"micro"$. For LLaMA 3 70B: $D=512$, $B_"micro"=4$ sequences of length 8192 gives a global batch of 2048 sequences = $16.7$M tokens.

== Training Stability

=== Loss Spikes and Recovery

Training runs at scale routinely encounter loss spikes — sudden increases in loss by 0.1–1.0 nats. Common causes:

+ *Batch with anomalous data:* a single very-long or repetitive document dominates the batch gradient.
+ *Gradient explosion:* accumulated numerical errors compound, especially after long stable stretches.
+ *Learning rate too high:* insufficient warmup or an overly aggressive schedule.

Standard mitigations:
- *Gradient clipping* at max norm 1.0 is the first line of defense. Monitor the pre-clip gradient norm — a healthy run has $||g|| in [0.5, 2.0]$; norms above 10 indicate instability.
- *Loss spike detection:* if the loss at step $t$ exceeds 1.5$times$ the rolling mean over the past 100 steps, roll back to the last checkpoint and skip or down-weight the offending batch.
- *BF16 over FP16:* BF16's wider dynamic range (matching FP32) prevents the overflow/underflow cycles that cause FP16 loss spikes.

```python
# Monitoring gradient norm during training
def train_step(model, batch, optimizer, scaler, clip_norm=1.0):
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

    # Log and alert on anomalous gradient norms
    if grad_norm > 10.0:
        print(f"WARNING: large grad norm {grad_norm:.2f} at step {step}")

    scaler.step(optimizer)
    scaler.update()
    return loss.item(), grad_norm.item()
```

=== Gradient Norms as a Diagnostic

Plot $||g||_2$ per step. Typical patterns:

#table(
  columns: (auto, auto),
  [*Observation*], [*Interpretation*],
  [Norm steadily decreasing to near 0],   [Learning rate too low or model converged early],
  [Norm oscillates 0.5–2.0 with rare spikes], [Healthy training],
  [Norm consistently $gt 5$],            [Instability — reduce LR or increase warmup],
  [Norm suddenly jumps to $gt 50$ then NaN], [Gradient explosion — check for bad data or LR spike],
  [Norm drops sharply then loss stagnates], [Possible dead neurons or learning rate collapse],
)

=== µP: Maximal Update Parametrization

Standard initialization (e.g., Kaiming normal) causes feature scale to change with model width $d$, so optimal hyperparameters (learning rate, initialization scale) shift as model size changes — making hyperparameter transfer across scales unreliable.

µP (Yang et al., 2022) parametrizes weights so that the _feature update scale_ is $O(1)$ independent of width. Key changes relative to standard parametrization:

+ *Input weights:* $W_"in" tilde cal(N)(0, 1)$ (no $1/d$ factor) — input embeddings and first-layer weights.
+ *Hidden weights:* $W_"hidden" tilde cal(N)(0, 1/d)$ and *learning rate scaled by $1/d$*: $eta_"hidden" = eta_"base" / d$.
+ *Output weights:* $W_"out" tilde cal(N)(0, 1/d)$ with learning rate $eta_"out" = eta_"base"$.
+ *Attention logit scale:* divide $Q K^top$ by $d_k$ instead of $sqrt(d_k)$, keeping logit magnitudes $O(1)$ as $d_k$ grows.

*Why it matters:* with µP, you can tune hyperparameters on a small proxy model (e.g., 40M params) and transfer them to the large model (7B, 70B) without re-tuning. Most production open-weights models (LLaMA, Mistral) do not use µP publicly; phi-2 and phi-3 explicitly use it, as does Cerebras-GPT.

```python
# mup library (Yang et al.) — drop-in replacement for nn.Linear
from mup import MuReadout, MuSharedReadout, set_base_shapes, make_base_shapes

# 1. Build a "base" (small) model and a "delta" model with different width
base_model  = MyTransformer(d_model=256)
delta_model = MyTransformer(d_model=512)
target      = MyTransformer(d_model=4096)   # the model you will train

# 2. Compute base shapes (defines the µP scaling rules)
base_shapes = make_base_shapes(base_model, delta_model, savefile="base_shapes.bsh")
set_base_shapes(target, base_shapes)

# 3. Use mup.MuAdamW instead of AdamW
from mup import MuAdamW
optimizer = MuAdamW(target.parameters(), lr=3e-4)
# lr will be automatically scaled per parameter group according to µP
```

=== Embedding and Output Layer Initialization

Embedding matrices are often initialized with $sigma = 0.01$ (smaller than Kaiming) to prevent large initial logits that saturate the softmax and produce uninformative gradients. The output (lm_head) weight is zero-initialized or tied to the embedding. Bias terms in attention projections are often omitted entirely in modern LLMs (LLaMA, Mistral, Gemma).

== References

- Radford, A. et al. "Language Models are Unsupervised Multitask Learners." OpenAI Blog, 2019. _(GPT-2)_
- Gage, P. "A New Algorithm for Data Compression." C Users Journal, 1994. _(BPE)_
- Broder, A. "On the resemblance and containment of documents." Compression and Complexity of Sequences, 1997. _(MinHash)_
- Micikevicius, P. et al. "Mixed Precision Training." ICLR, 2018.
- Chen, T. et al. "Training Deep Nets with Sublinear Memory Cost." arXiv:1604.06174, 2016. _(Gradient checkpointing)_
- Kingma, D. P. & Ba, J. "Adam: A Method for Stochastic Optimization." ICLR, 2015.
- Loshchilov, I. & Hutter, F. "Decoupled Weight Decay Regularization." ICLR, 2019. _(AdamW)_
- Kaplan, J. et al. "Scaling Laws for Neural Language Models." arXiv:2001.08361, 2020.
- Hoffmann, J. et al. "Training Compute-Optimal Large Language Models." NeurIPS, 2022. _(Chinchilla)_
- Muennighoff, N. et al. "Scaling Data-Constrained Language Models." NeurIPS, 2023.
- Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC20, 2020.
- Shoeybi, M. et al. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." arXiv:1909.08053, 2019.
- Yang, G. et al. "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer." NeurIPS, 2022. _(µP)_
- Touvron, H. et al. "LLaMA 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288, 2023.
- Dubey, A. et al. "The LLaMA 3 Herd of Models." arXiv:2407.21783, 2024.
- Wenzek, G. et al. "CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data." LREC, 2020. _(perplexity-based filtering)_

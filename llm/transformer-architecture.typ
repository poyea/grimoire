= Transformer Architecture

The transformer is a sequence-to-sequence architecture built entirely from attention and feed-forward layers. Every modern LLM — GPT, LLaMA, Mistral, Gemma, Qwen — is a decoder-only transformer. Understanding its internals precisely is prerequisite to everything that follows.

*See also:* _gpu-architecture/ml-workloads.typ_ (Flash Attention, GEMM kernels), _llm/pretraining.typ_ (how these weights are learned).

*Code note:* PyTorch is used for all runnable examples. C++ (libtorch / llama.cpp) is shown for performance-critical kernels where the implementation detail matters. JAX is shown where its functional style clarifies the math.

== High-Level Data Flow

```
Input token ids:  [t_0, t_1, ..., t_{L-1}]
        │
        ▼  nn.Embedding  [V, d_model]
Token embeddings  [B, L, d_model]
        │
        ▼
┌─────────────────────────────────────┐  × n_layers
│  x = x + Attn(RMSNorm(x), cos, sin) │  ← pre-norm residual
│  x = x + FFN(RMSNorm(x))            │
└─────────────────────────────────────┘
        │
        ▼  RMSNorm  →  linear [d_model, V]  (weight-tied)
Logits  [B, L, V]  →  softmax → next-token distribution
```

== Token Embeddings

Each token $t_i in {0,...,V-1}$ is looked up in $W_E in RR^(V times d_"model")$.

```python
import torch, torch.nn as nn

embed = nn.Embedding(128_256, 4096)          # LLaMA 3 8B
ids   = torch.tensor([[1, 2731, 338, 263]])  # [B=1, L=4]
x     = embed(ids)                           # [1, 4, 4096]
```

*Weight tying:* the unembedding (lm_head) shares weights with the embedding. Saves ~500M params for LLaMA 3 and improves gradient signal for rare tokens.

```python
lm_head = nn.Linear(4096, 128_256, bias=False)
lm_head.weight = embed.weight   # same tensor, no copy
```

*Vocabulary sizes:*

#table(
  columns: (auto, auto, auto),
  [*Model*], [*V*], [*Tokenizer*],
  [GPT-3],      [50 257],  [BPE (tiktoken)],
  [LLaMA 1/2],  [32 000],  [SentencePiece BPE],
  [LLaMA 3],    [128 256], [tiktoken BPE],
  [Mistral],    [32 000],  [SentencePiece BPE],
  [Gemma 2],    [256 000], [SentencePiece BPE],
  [DeepSeek-V3],[129 280], [BPE],
)

== Scaled Dot-Product Attention

Given input $X in RR^(L times d_"model")$:

$ Q = X W_Q, quad K = X W_K, quad V = X W_V $

$ "Attention"(Q, K, V) = "softmax"((Q K^T) / sqrt(d_"head") + M) V $

$M$ is the causal mask: $M_(i j) = 0$ if $j <= i$, else $-infinity$.

*Why* $1/sqrt(d_"head")$: dot products grow with variance $d_"head"$; scaling restores unit variance before softmax, preventing vanishing gradients in the softmax.

*Computational cost per layer:*

```
QKV projections:  2 × 3 × L × d_model²        FLOPs
QK^T matmul:      2 × L² × d_model             FLOPs   ← O(L²) term
AV  matmul:       2 × L² × d_model             FLOPs
Output proj:      2 × L × d_model²             FLOPs
```

The $O(L^2)$ term dominates for long contexts. Flash Attention avoids materializing the $L times L$ matrix in HBM — see _gpu-architecture/ml-workloads.typ_.

*PyTorch 2.0+ dispatches to Flash Attention automatically:*
```python
import torch.nn.functional as F

# q, k, v: [B, n_heads, L, d_head]
out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    is_causal=True,      # generates causal mask internally
    scale=None,          # defaults to 1/sqrt(d_head)
)
# Uses FlashAttention-2 on CUDA if available — no explicit mask materialized
```

== Multi-Head Attention (MHA)

Run $n_"heads"$ parallel attention heads, concatenate, project:

$ "MHA"(X) = "Concat"("head"_1, ..., "head"_h) W_O $

```python
import torch, torch.nn as nn, torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.qkv     = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj  = nn.Linear(d_model, d_model,     bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        # fused QKV projection then split — one GEMM instead of three
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def reshape(t):
            return t.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out)
```

Each head learns different patterns: local context, syntactic dependencies, coreference. The output projection mixes information across heads.

== Grouped Query Attention (GQA) and Multi-Query Attention (MQA)

*Problem:* MHA KV cache size = $2 times n_"heads" times L times d_"head"$ floats per layer. LLaMA 2 70B at $L=4096$: 2 × 64 × 4096 × 128 × 2B ≈ 8.6 GB/layer — unacceptable.

*MQA* (Shazeer 2019): single K and V head shared by all Q heads. KV cache ÷ $n_"heads"$.

*GQA* (Ainslie et al. 2023): $n_"kv"$ KV groups, each shared by $n_"heads" / n_"kv"$ query heads.

#table(
  columns: (auto, auto, auto, auto),
  [*Model*],      [*n\_heads*], [*n\_kv*], [*KV cache vs MHA*],
  [LLaMA 2 7B],  [32], [32], [1.0× (MHA)],
  [LLaMA 2 70B], [64], [8],  [0.125×],
  [LLaMA 3 8B],  [32], [8],  [0.25×],
  [Mistral 7B],  [32], [8],  [0.25×],
  [GPT-3],       [96], [96], [1.0× (MHA)],
)

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv: int):
        super().__init__()
        assert n_heads % n_kv == 0
        self.n_heads, self.n_kv = n_heads, n_kv
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv   * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv   * self.d_head, bias=False)
        self.o_proj = nn.Linear(d_model, d_model,              bias=False)

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv,   self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv,   self.d_head).transpose(1, 2)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # repeat KV heads to match Q heads
        reps = self.n_heads // self.n_kv
        k = k.repeat_interleave(reps, dim=1)  # [B, n_heads, L, d_head]
        v = v.repeat_interleave(reps, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, L, -1))
```

== Rotary Position Embedding (RoPE)

RoPE (Su et al. 2021) rotates Q and K by an angle proportional to position *inside* the attention operation, so that $Q_m dot K_n$ depends only on relative position $m - n$.

For a 2D pair at dimension $k$, position $m$:

$ R(theta_k, m) = mat(cos(m theta_k), -sin(m theta_k); sin(m theta_k), cos(m theta_k)), quad theta_k = 10000^(-2k \/ d_"head") $

$ tilde(Q)_m dot tilde(K)_n = Q_m R(Theta, m)^T R(Theta, n) K_n = Q_m R(Theta, n - m) K_n $

Only the difference $n - m$ appears — true relative encoding with no explicit relative-position parameters.

*Precompute and apply in PyTorch:*

```python
def build_rope_cache(max_len: int, d_head: int,
                     theta_base: float = 500_000.0,
                     device=None):
    # LLaMA 3 uses theta_base=500000 (vs original 10000)
    i = torch.arange(0, d_head, 2, device=device).float()
    inv_freq = 1.0 / (theta_base ** (i / d_head))        # [d_head/2]
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                      # [max_len, d_head/2]
    return freqs.cos(), freqs.sin()                       # each [max_len, d_head/2]


def apply_rope(x: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x:   [B, n_heads, L, d_head]
    # cos, sin: [1, 1, L, d_head/2]  (broadcast)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)
```

*C++ implementation (llama.cpp style) — used in production CPU inference:*

```cpp
// Apply RoPE in-place; x: float[d_head], d_head must be even
void rope_inplace(float* x, int pos, int d_head,
                  float theta_base = 500000.f) {
    for (int k = 0; k < d_head / 2; k++) {
        float theta = pos / powf(theta_base, 2.f * k / d_head);
        float c = cosf(theta), s = sinf(theta);
        float x0 = x[k], x1 = x[k + d_head / 2];
        x[k]              = x0 * c - x1 * s;
        x[k + d_head / 2] = x0 * s + x1 * c;
    }
}
```

*Context extension (YaRN, Peng et al. 2023):* scales $theta_k$ by a factor that interpolates between NTK-aware and full position interpolation, enabling 128k+ context at low perplexity penalty.

=== ALiBi (Press et al. 2022)

No positional vectors — adds a position penalty to attention scores:

$ "score"_(i j) -= m_h dot (i - j), quad m_h = 2^(-8 h / n_"heads") $

```python
def alibi_bias(n_heads: int, max_len: int) -> torch.Tensor:
    # returns [1, n_heads, max_len, max_len] additive bias
    slopes = 2 ** (-8 * torch.arange(1, n_heads + 1) / n_heads)  # [n_heads]
    pos    = torch.arange(max_len)
    dist   = pos.unsqueeze(1) - pos.unsqueeze(0)   # [L, L] relative pos
    # causal: only look backwards
    dist   = dist.clamp(max=0)
    return (slopes[:, None, None] * dist).unsqueeze(0)  # [1, n_heads, L, L]
```

Good extrapolation beyond training length. Used in BLOOM, MPT.

== Feed-Forward Network (FFN)

The FFN runs independently on each token. ~2/3 of total parameters.

=== SwiGLU (LLaMA, PaLM, Gemma — standard in 2024)

$ "SwiGLU"(x) = W_"down" dot ("SiLU"(W_"gate" x) times.circle W_"up" x) $

Three weight matrices; $d_"ff" approx 8/3 dot d_"model"$ (rounded to multiple of 256).

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate path: SiLU(Wx) acts as a learned soft gate
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

*C++ (llama.cpp kernel) — fused for inference:*

```cpp
// SwiGLU forward: x[d_model] → out[d_model]
// gate_w, up_w: [d_ff, d_model]  down_w: [d_model, d_ff]
void swiglu(const float* x, float* out,
            const float* gate_w, const float* up_w, const float* down_w,
            int d_model, int d_ff) {
    std::vector<float> gate(d_ff), up(d_ff), hidden(d_ff);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d_ff, d_model,
                1.f, gate_w, d_model, x, 1, 0.f, gate.data(), 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d_ff, d_model,
                1.f, up_w,   d_model, x, 1, 0.f, up.data(),   1);
    for (int i = 0; i < d_ff; i++) {
        float g = gate[i];
        hidden[i] = (g / (1.f + expf(-g))) * up[i];  // SiLU * up
    }
    cblas_sgemv(CblasRowMajor, CblasNoTrans, d_model, d_ff,
                1.f, down_w, d_ff, hidden.data(), 1, 0.f, out, 1);
}
```

*JAX/Flax — shows the functional purity:*

```python
import jax.numpy as jnp
import flax.linen as nn

class SwiGLU(nn.Module):
    d_ff: int

    @nn.compact
    def __call__(self, x):                          # x: [..., d_model]
        gate = nn.Dense(self.d_ff, use_bias=False)(x)
        up   = nn.Dense(self.d_ff, use_bias=False)(x)
        down = nn.Dense(x.shape[-1], use_bias=False)
        return down(nn.silu(gate) * up)
```

=== Standard FFN (GPT-2, GPT-3)

$ "FFN"(x) = W_2 dot "GeLU"(W_1 x), quad d_"ff" = 4 d_"model" $

```python
class GPT2FFN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))
```

== Layer Normalization

=== RMSNorm (LLaMA, Mistral, Gemma — standard in 2024)

$ "RMSNorm"(x) = (x) / (sqrt((1/d) sum_i x_i^2 + epsilon)) times.circle g $

No mean-centering; ~15% faster than LayerNorm. $g in RR^d$ is learned.

```python
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight
```

*C++:*

```cpp
void rms_norm(const float* x, const float* w, float* out,
              int d, float eps = 1e-6f) {
    float ss = 0.f;
    for (int i = 0; i < d; i++) ss += x[i] * x[i];
    float rms = sqrtf(ss / d + eps);
    for (int i = 0; i < d; i++) out[i] = x[i] / rms * w[i];
}
```

=== Pre-LN vs Post-LN

```
Post-LN (original):  x = LayerNorm(x + F(x))   ← unstable at scale
Pre-LN  (GPT-2+):    x = x + F(LayerNorm(x))   ← stable, now universal
```

Pre-LN gradient: $partial cal(L) / partial x_l$ flows through a clean additive path; the residual bypasses all subsequent nonlinearities.

== Full Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv: int, d_ff: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn      = GroupedQueryAttention(d_model, n_heads, n_kv)
        self.ffn_norm  = RMSNorm(d_model)
        self.ffn       = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # pre-norm residual connections
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

*With gradient checkpointing* (trades recomputation for 4–8× activation memory reduction during training):

```python
from torch.utils.checkpoint import checkpoint

class TransformerBlockWithCheckpoint(TransformerBlock):
    def forward(self, x, cos, sin):
        # recomputes activations during backward; saves ~60% of peak memory
        return checkpoint(super().forward, x, cos, sin,
                          use_reentrant=False)
```

== Residual Connections and Gradient Flow

$ (partial cal(L)) / (partial x_l) = (partial cal(L)) / (partial x_L) product_(k=l)^(L-1) (1 + (partial F_k) / (partial x_k)) $

The additive 1 prevents vanishing/exploding gradients regardless of $partial F_k / partial x_k$. Gradient norm in Pre-LN transformers grows as $O(sqrt(n_"layers"))$ rather than exponentially.

*Verify gradient norms in PyTorch:*

```python
model = MiniLLM(...).cuda()
ids   = torch.randint(0, 128_256, (2, 512)).cuda()
logits = model(ids)
loss   = F.cross_entropy(logits[:, :-1].reshape(-1, 128_256),
                          ids[:, 1:].reshape(-1))
loss.backward()

for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"{name:50s}  grad_norm={p.grad.norm():.4f}")
# healthy: norms ~0.01–0.1 across all layers
```

== Model Architecture Reference

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  [*Model*], [*Params*], [*d\_model*], [*n\_layers*], [*n\_heads*], [*n\_kv*], [*d\_ff*],
  [GPT-3],        [175B],    [12 288], [96], [96], [96], [49 152],
  [LLaMA 2 7B],   [7B],      [4 096],  [32], [32], [32], [11 008],
  [LLaMA 2 70B],  [70B],     [8 192],  [80], [64], [8],  [28 672],
  [LLaMA 3 8B],   [8B],      [4 096],  [32], [32], [8],  [14 336],
  [LLaMA 3 70B],  [70B],     [8 192],  [80], [64], [8],  [28 672],
  [Mistral 7B],   [7B],      [4 096],  [32], [32], [8],  [14 336],
  [Gemma 2 9B],   [9B],      [3 584],  [42], [16], [8],  [14 336],
  [DeepSeek-V3],  [671B MoE],[7 168],  [61], [128],[128],[18 432 × 256 exp.],
)

== Multi-Head Latent Attention (MLA) — DeepSeek-V2

Compresses the KV cache via low-rank projection. Instead of caching $n_"kv" times d_"head"$ vectors per token, a single latent vector $c_t in RR^(d_c)$ is cached ($d_c << n_"kv" d_"head"$):

$ c_t = W^(D K V) h_t, quad K_t = W^(U K) c_t, quad V_t = W^(U V) c_t $

93% KV cache reduction vs MHA at similar quality. Trade-off: additional matrix multiplies per decode step.

```python
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int,
                 d_c: int, d_head: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_head
        # compress to latent
        self.kv_down  = nn.Linear(d_model, d_c, bias=False)
        # expand K and V from latent
        self.k_up     = nn.Linear(d_c, n_heads * d_head, bias=False)
        self.v_up     = nn.Linear(d_c, n_heads * d_head, bias=False)
        self.q_proj   = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.o_proj   = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                kv_cache: list | None = None) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        latent = self.kv_down(x)                    # [B, L, d_c] — cache this
        k = self.k_up(latent).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_up(latent).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, L, -1))
```

== References

Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS 2017.

Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864.

Ainslie, J. et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP 2023.

Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." arXiv:1911.02150.

Shazeer, N. (2020). "GLU Variants Improve Transformer." arXiv:2002.05202.

Zhang, B. & Sennrich, R. (2019). "Root Mean Square Layer Normalization." NeurIPS 2019.

Press, O. et al. (2022). "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation." ICLR 2022.

Peng, B. et al. (2023). "YaRN: Efficient Context Window Extension of Large Language Models." arXiv:2309.00071.

Liu, A. et al. (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." arXiv:2405.04434.

Touvron, H. et al. (2023). "LLaMA 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288.

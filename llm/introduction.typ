= Introduction

Large Language Models (LLMs) are the defining systems of modern AI. A single model like LLaMA 3 70B encodes over 70 billion learned parameters, requires terabytes of training data, and demands hundreds of GPU-hours to serve efficiently at scale. Understanding how they work — from the mathematics of attention to the engineering of distributed training and low-latency inference — is essential for anyone building or operating AI systems today.

*Scope of this reference:*

- *Architecture:* Transformer internals, attention variants, positional encodings, normalization
- *Pretraining:* Data pipelines, scaling laws, mixed precision, distributed strategies
- *Fine-tuning:* LoRA, QLoRA, instruction tuning, alignment
- *RLHF:* Reward modeling, PPO, DPO, Constitutional AI
- *Inference optimization:* Speculative decoding, continuous batching, PagedAttention
- *Quantization:* GPTQ, AWQ, FP8, INT4, GGUF
- *Serving systems:* vLLM, TensorRT-LLM, throughput/latency tradeoffs
- *Evaluation:* Benchmarks, perplexity, LLM-as-judge, safety evals

*Primary focus:* Decoder-only transformer (GPT family), the dominant architecture for generative LLMs. Encoder-only (BERT) and encoder-decoder (T5) covered where relevant.

*Intended audience:* ML engineers, systems programmers, and researchers who want a precise, technical understanding of how LLMs are built and deployed — not just how to use them.

*Code conventions:* Examples use PyTorch as the primary framework (industry standard for research and production). JAX/Flax is shown where it meaningfully differs. All examples assume CUDA availability.

*See also:* _gpu-architecture/ml-workloads.typ_ (Flash Attention, GEMM kernels, KV cache hardware), _gpu-architecture/multi-gpu.typ_ (distributed training topology).

== Notation and Conventions

*Model dimensions:*

```
d_model   — residual stream dimension (hidden size), e.g. 4096 for LLaMA 3 8B
d_ff      — feed-forward inner dimension, typically 4 × d_model (or ~2.67× for SwiGLU)
d_head    — per-attention-head dimension = d_model / n_heads
n_heads   — number of attention heads
n_kv      — number of key/value heads (< n_heads for GQA/MQA)
n_layers  — transformer depth (number of blocks)
V         — vocabulary size, e.g. 128256 for LLaMA 3
L         — sequence length (context window), e.g. 8192, 128k
B         — batch size
```

*Parameter count* (dense decoder-only, approximate):

```
Embedding:          V × d_model
Per layer:
  Attention QKV:    d_model × (n_heads + 2×n_kv) × d_head    ← GQA
  Attention out:    d_model × d_model
  FFN (SwiGLU):     3 × d_model × d_ff
Total ≈ 2 × V × d_model  +  n_layers × (4×d_model² + 3×d_model×d_ff)
```

For LLaMA 3 8B: $d_"model"=4096$, $d_"ff"=14336$, $n_"layers"=32$, $V=128256$ → ~8.0B params.

*Verify with PyTorch:*
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
total = sum(p.numel() for p in model.parameters())
print(f"{total/1e9:.2f}B params")   # → 8.03B params
```

*FLOPs conventions:* one MAC = 2 FLOPs. Training FLOPs $approx 6 N T$ where $N$ = params, $T$ = tokens.

*Token conventions:*
- 1 token $approx$ 0.75 English words
- 1B tokens $approx$ 750 MB clean English text
- Training scale: GPT-3 = 300B tokens; LLaMA 3 = 15T tokens

== A Minimal Working LLM in 80 Lines

Before diving into internals, here is a complete minimal GPT-2-class decoder transformer in PyTorch. Every concept in this book refers back to this skeleton.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d]
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def apply_rope(x: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    # x: [B, n_heads, L, d_head]
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.n_kv = n_heads, n_kv
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv   * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv   * self.d_head, bias=False)
        self.o_proj = nn.Linear(d_model, d_model,              bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_kv,   self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_kv,   self.d_head).transpose(1, 2)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # expand KV heads to match Q heads (GQA)
        reps = self.n_heads // self.n_kv
        k = k.repeat_interleave(reps, dim=1)
        v = v.repeat_interleave(reps, dim=1)

        # scaled dot-product attention with causal mask (PyTorch 2.0 fused)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff,   d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))  # SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv: int, d_ff: int):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn      = CausalSelfAttention(d_model, n_heads, n_kv)
        self.ffn_norm  = RMSNorm(d_model)
        self.ffn       = FeedForward(d_model, d_ff)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)  # pre-norm residual
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MiniLLM(nn.Module):
    def __init__(self, vocab: int, d_model: int, n_layers: int,
                 n_heads: int, n_kv: int, d_ff: int, max_len: int = 2048):
        super().__init__()
        self.embed    = nn.Embedding(vocab, d_model)
        self.blocks   = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, n_kv, d_ff)
             for _ in range(n_layers)])
        self.norm     = RMSNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        # precompute RoPE cos/sin table
        d_head = d_model // n_heads
        inv_freq = 1.0 / (500_000 ** (torch.arange(0, d_head, 2).float() / d_head))
        t = torch.arange(max_len)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos()[None, None])  # [1,1,L,d/2]
        self.register_buffer("sin", freqs.sin()[None, None])

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [B, L]
        L = ids.shape[1]
        x = self.embed(ids)
        cos, sin = self.cos[:, :, :L], self.sin[:, :, :L]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.lm_head(self.norm(x))   # [B, L, V] logits


# LLaMA 3 8B configuration
model = MiniLLM(
    vocab=128_256, d_model=4096, n_layers=32,
    n_heads=32, n_kv=8, d_ff=14336,
).cuda().to(torch.bfloat16)

ids = torch.randint(0, 128_256, (1, 64)).cuda()
logits = model(ids)   # [1, 64, 128256]
print(logits.shape, logits.dtype)
```

This skeleton omits Flash Attention (use `torch.nn.functional.scaled_dot_product_attention` which dispatches to it automatically on CUDA), gradient checkpointing, and tensor parallelism — all covered in later chapters.

== Key Papers Timeline

=== 2017 — Attention Is All You Need

*Vaswani et al., NeurIPS 2017.*

Introduced the Transformer: self-attention replacing recurrence entirely.

$ "Attention"(Q, K, V) = "softmax"((Q K^T) / sqrt(d_k)) V $

Impact: eliminated the sequential bottleneck of RNNs; made training massively parallelizable across the sequence dimension.

=== 2018 — BERT

*Devlin et al., NAACL 2019.*

Encoder-only transformer with masked language modeling (MLM). Bidirectional context — better for understanding tasks. Established pretraining + fine-tuning paradigm.

=== 2018–2020 — GPT → GPT-3

*Radford et al. 2018, 2019; Brown et al. NeurIPS 2020.*

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  [*Model*], [*Params*], [*Tokens*], [*d\_model*], [*n\_layers*], [*n\_heads*],
  [GPT],   [117M],  [~1B],   [768],   [12], [12],
  [GPT-2], [1.5B],  [~40B],  [1600],  [48], [25],
  [GPT-3], [175B],  [300B],  [12288], [96], [96],
)

GPT-3 demonstrated *few-shot in-context learning* (ICL): no gradient updates, just examples in the prompt.

=== 2020 — Scaling Laws (Kaplan et al.)

$ L(N) approx (N_c / N)^(alpha_N), quad L(D) approx (D_c / D)^(alpha_D) $

Power-law loss vs model size and data. Revised by Chinchilla (2022).

=== 2022 — Chinchilla (Hoffmann et al., DeepMind)

Compute-optimal training: ~20 tokens per parameter. GPT-3 was severely undertrained.

$ N_"opt" approx 0.5 (C / 6)^0.5, quad D_"opt" approx 20 N_"opt" $

Chinchilla 70B (1.4T tokens) beat Gopher 280B (300B tokens) at equal compute.

=== 2022 — InstructGPT / RLHF (Ouyang et al., OpenAI)

SFT → reward model on human preferences → PPO. A 1.3B aligned model preferred over 175B base. Established RLHF as standard alignment.

=== 2023 — LLaMA 1 & 2 (Meta)

First high-quality open-weights models. Architecture: RoPE + RMSNorm + SwiGLU + no bias. LLaMA 2 added GQA for 70B, 4k context, RLHF chat variants.

=== 2023 — Mistral 7B

GQA + sliding window attention. 7B outperforms LLaMA 2 13B. Showed architectural efficiency compounds.

=== 2023 — Flash Attention 2 (Dao)

IO-aware attention tiling: avoids $O(L^2)$ HBM traffic. 2× faster than FA1. Now default in PyTorch (`F.scaled_dot_product_attention`).

=== 2023–2024 — Speculative Decoding + PagedAttention

- *Speculative decoding* (Leviathan et al.): draft + verify, 2–3× throughput at identical quality.
- *vLLM / PagedAttention* (Kwon et al., UC Berkeley): KV cache as paged virtual memory. Became the dominant serving engine.

=== 2024 — LLaMA 3, Gemma 2, Qwen 2

15T+ token training, 128k+ context via YaRN RoPE extension, FP8 training on H100.

=== 2024 — DeepSeek-V2 / V3 / R1

- *V2:* Multi-head Latent Attention (MLA), 93% KV cache reduction.
- *V3:* 671B MoE trained for \$6M. GPT-4 class.
- *R1:* Pure RL reasoning (GRPO), no supervised chain-of-thought. Matched o1 on math.

== Document Structure

1. *Transformer Architecture* — attention, GQA, RoPE, SwiGLU, RMSNorm
2. *Pretraining* — data, scaling laws, BF16/FP8, distributed training
3. *Fine-tuning* — LoRA, QLoRA, instruction tuning
4. *RLHF* — reward model, PPO, DPO
5. *Inference Optimization* — speculative decoding, continuous batching, PagedAttention
6. *Quantization* — GPTQ, AWQ, FP8, INT4, GGUF
7. *Serving Systems* — vLLM, TensorRT-LLM, latency/throughput
8. *Evaluation* — benchmarks, perplexity, safety

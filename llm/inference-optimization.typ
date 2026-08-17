= Inference Optimization

Training a language model is a one-time cost; _inference_ runs continuously in production. Inference efficiency determines cost per query, latency, and how many users a single GPU cluster can serve. This chapter covers the full stack: KV cache mechanics, decoding strategies, speculative decoding, continuous batching, paged memory management, prefix reuse, chunked prefill, tensor parallelism, and how to measure what matters.

*See also:* _Transformer Architecture_ (attention internals), _ML Workload Optimization on GPUs (GPU Architecture volume)_ (Flash Attention, GEMM kernels), _GPU Memory Hierarchy (GPU Architecture volume)_ (HBM bandwidth, cache hierarchy).

== KV Cache

=== Why It Exists

During autoregressive decoding the model generates one token per forward pass. At each step, attention must attend over _all_ previously generated tokens. Without caching this would require recomputing keys and values for every prior token on every step — $O(L^2)$ total work over a sequence of length $L$.

The *KV cache* stores the projected keys and values for every layer and every token that has already been processed. The next step only needs to compute $K, V$ for the single new token, then concatenate with the cache.

=== Memory Formula

For a single sequence the KV cache occupies:

$ M_"KV" = 2 times n_"layers" times n_"kv_heads" times d_"head" times L times B_"dtype" $

where:
- $2$ accounts for keys and values
- $n_"kv_heads"$ is the number of key/value heads (equals $n_"heads"$ in MHA; smaller in GQA/MQA)
- $d_"head" = d_"model" / n_"heads"$
- $L$ is the current sequence length
- $B_"dtype"$ is bytes per element (2 for fp16/bf16, 1 for fp8)

*Example — LLaMA 3 8B* ($n_"layers"=32$, $n_"kv_heads"=8$, $d_"head"=128$, bf16):

$ M_"KV" = 2 times 32 times 8 times 128 times L times 2 = 131 072 times L " bytes" $

At $L = 4096$ tokens that is 512 MiB. At $L = 128"k"$ that is 16 GiB — a 40 GiB A100 can hold fewer than three such sequences simultaneously, illustrating why KV cache management is the dominant serving bottleneck.

=== Growth During Decode

The cache grows by exactly one row (one $(K, V)$ pair per layer per head) for each decoded token. Peak memory occurs at the end of generation, so worst-case allocation must be reserved upfront in static batching — wasting memory when sequences finish early.

=== C++ Cache Struct

```cpp
#include <cstdint>
#include <vector>

// Single-sequence KV cache: [layer, head, seq, d_head]
struct KVCache {
    int n_layers;
    int n_kv_heads;
    int d_head;
    int capacity;          // pre-allocated max sequence length
    int current_len = 0;

    // Flat storage: layer * n_kv_heads * capacity * d_head * sizeof(float16)
    std::vector<uint16_t> k_data;   // bf16 keys
    std::vector<uint16_t> v_data;   // bf16 values

    KVCache(int layers, int heads, int head_dim, int cap)
        : n_layers(layers), n_kv_heads(heads), d_head(head_dim), capacity(cap) {
        size_t n = (size_t)layers * heads * cap * head_dim;
        k_data.resize(n, 0);
        v_data.resize(n, 0);
    }

    // Pointer to key slice for layer l, head h at position pos
    uint16_t* key_ptr(int l, int h, int pos) {
        size_t offset = ((size_t)l * n_kv_heads + h) * capacity * d_head
                        + (size_t)pos * d_head;
        return k_data.data() + offset;
    }
    uint16_t* val_ptr(int l, int h, int pos) {
        size_t offset = ((size_t)l * n_kv_heads + h) * capacity * d_head
                        + (size_t)pos * d_head;
        return v_data.data() + offset;
    }

    void append(int layer, int head,
                const uint16_t* k, const uint16_t* v) {
        std::copy(k, k + d_head, key_ptr(layer, head, current_len));
        std::copy(v, v + d_head, val_ptr(layer, head, current_len));
    }
    void commit_step() { ++current_len; }
};
```

=== PyTorch KVCache Class

```python
import torch
from dataclasses import dataclass

@dataclass
class KVCache:
    # k, v: [n_layers, n_kv_heads, max_len, d_head]
    k: torch.Tensor
    v: torch.Tensor
    filled: int = 0

    @staticmethod
    def allocate(n_layers: int, n_kv_heads: int,
                 d_head: int, max_len: int,
                 device: str = "cuda", dtype=torch.bfloat16) -> "KVCache":
        shape = (n_layers, n_kv_heads, max_len, d_head)
        return KVCache(
            k=torch.empty(shape, device=device, dtype=dtype),
            v=torch.empty(shape, device=device, dtype=dtype),
        )

    def update(self, layer: int,
               new_k: torch.Tensor,   # [B, n_kv_heads, 1, d_head]
               new_v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.k[layer, :, self.filled : self.filled + 1, :] = new_k[0]
        self.v[layer, :, self.filled : self.filled + 1, :] = new_v[0]
        # Return full context up to now
        k_ctx = self.k[layer, :, : self.filled + 1, :]
        v_ctx = self.v[layer, :, : self.filled + 1, :]
        return k_ctx, v_ctx

    def commit(self):
        self.filled += 1

    @property
    def bytes_used(self) -> int:
        elem = self.k.element_size()
        return 2 * self.k.numel() * elem  # k + v
```

Memory is pre-allocated to avoid CUDA malloc stalls during decode.

== Decoding Strategies

After the forward pass produces logits $z in RR^V$, the next token $t$ is sampled from a distribution derived from $z$. The choice of strategy trades off quality, diversity, and speed.

=== Greedy Decoding

$ t = arg max_v z_v $

Zero randomness. Fast and deterministic. Suffers from degenerate repetition on open-ended generation.

=== Temperature Scaling

Before any sampling, logits are divided by temperature $T > 0$:

$ p_v = "softmax"(z / T)_v $

$T < 1$ sharpens the distribution (more greedy), $T > 1$ flattens it (more random). $T -> 0$ recovers greedy; $T -> infinity$ gives uniform.

=== Top-k Sampling

Restrict the vocabulary to the $k$ tokens with highest logit, renormalize, then sample.

=== Top-p (Nucleus) Sampling

Sort tokens by descending probability. Keep the smallest prefix $S$ such that:

$ sum_(v in S) p_v >= p $

Renormalize over $S$ and sample. Adapts the effective vocabulary size to the entropy of the distribution — tight distributions keep fewer candidates; flat distributions keep many. Introduced by Holtzman et al. (2020).

=== Min-p Sampling

Keep tokens whose probability exceeds $p_"min" times p_"max"$ where $p_"max"$ is the mode probability. Scales the cutoff relative to the mode, making it less sensitive to temperature.

=== Beam Search

Maintain a beam of $B$ partial sequences, expanding each by the top-$B$ tokens at every step and keeping the $B$ highest-scoring hypotheses overall.

```
Step 0   ["The"]
          /      \
Step 1  ["The cat"] ["The dog"]
        /    \       /    \
Step 2 ["The cat sat"] ["The cat ran"] ["The dog sat"] ["The dog ran"]
```

Beam search maximizes sequence probability but is deterministic and prone to high-probability short phrases. Not recommended for open-ended generation; useful for translation and summarization.

=== C++ Top-p Implementation

```cpp
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

// top_p_sample: sample from nucleus of logits.
// logits: raw model output, length vocab_size
// Returns sampled token id.
int top_p_sample(const std::vector<float>& logits,
                 float temperature, float top_p,
                 std::mt19937& rng) {
    int V = (int)logits.size();

    // Temperature scaling + softmax
    std::vector<float> probs(V);
    float max_l = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        probs[i] = std::exp((logits[i] - max_l) / temperature);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;

    // Sort indices by descending probability
    std::vector<int> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b){ return probs[a] > probs[b]; });

    // Nucleus: accumulate until >= top_p
    float cumsum = 0.0f;
    int nucleus_end = V;
    for (int i = 0; i < V; i++) {
        cumsum += probs[idx[i]];
        if (cumsum >= top_p) { nucleus_end = i + 1; break; }
    }

    // Renormalize over nucleus
    float nucleus_sum = 0.0f;
    for (int i = 0; i < nucleus_end; i++) nucleus_sum += probs[idx[i]];
    std::vector<float> nucleus_probs(nucleus_end);
    for (int i = 0; i < nucleus_end; i++)
        nucleus_probs[i] = probs[idx[i]] / nucleus_sum;

    // Sample
    std::discrete_distribution<int> dist(
        nucleus_probs.begin(), nucleus_probs.end());
    return idx[dist(rng)];
}
```

=== PyTorch generate() Loop

```python
import torch
import torch.nn.functional as F
from typing import Optional

def generate(
    model,
    input_ids: torch.Tensor,    # [1, prompt_len]
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    eos_id: int = 2,
    cache: Optional["KVCache"] = None,
) -> torch.Tensor:
    device = input_ids.device
    generated = []

    # Prefill
    with torch.no_grad():
        logits = model(input_ids, cache=cache)  # [1, prompt_len, V]

    for _ in range(max_new_tokens):
        next_logits = logits[0, -1, :] / temperature   # [V]

        # Top-p nucleus sampling
        sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens beyond nucleus
        sorted_probs[cumprobs - sorted_probs > top_p] = 0.0
        sorted_probs /= sorted_probs.sum()

        next_token = sorted_idx[torch.multinomial(sorted_probs, 1)]
        generated.append(next_token.item())
        if next_token.item() == eos_id:
            break

        # Decode step: single new token
        with torch.no_grad():
            logits = model(
                next_token.unsqueeze(0).unsqueeze(0),
                cache=cache,
            )

    return torch.tensor(generated, device=device)
```

The prefill phase is *compute-bound*: all prompt tokens are processed in a single forward pass as a dense matrix multiplication, making efficient use of tensor cores. The decode phase is *memory-bandwidth-bound*: each step processes one token but must read the entire KV cache from HBM, so throughput scales with memory bandwidth rather than compute. This asymmetry drives batching strategy: large batch sizes help prefill throughput by amortizing the matrix multiplications over more sequences, while decode latency is dominated by KV cache size and available memory bandwidth regardless of batch size.

== Speculative Decoding

=== Motivation

The autoregressive bottleneck is _latency_, not compute: each decode step requires a full forward pass, and steps cannot be parallelized. Speculative decoding (Leviathan et al., 2023) uses a small _draft model_ to propose $k$ tokens in parallel, then a single forward pass of the _target model_ verifies all $k$ simultaneously — recovering exact target-model distribution while reducing the number of target-model calls.

=== Algorithm

1. Draft model $q$ autoregressively generates $k$ candidate tokens $x_1, ..., x_k$.
2. Run the target model $p$ once on the $k$-token continuation. This single forward pass yields $p(x_t | "context")$ for all $k$ positions simultaneously.
3. Accept token $x_i$ with probability:

$ alpha_i = min(1, frac(p(x_i | x_1,...,x_(i-1)), q(x_i | x_1,...,x_(i-1)))) $

4. If $x_i$ is rejected, sample a _correction token_ from:

$ p'(x) = "norm"( max(0, p(x) - q(x)) )$

and discard $x_i, ..., x_k$.

5. If all $k$ tokens are accepted, sample one additional token from $p$ for free.

=== Expected Tokens Per Step

Let $alpha = E[alpha_i]$ be the average per-token acceptance rate. The expected number of tokens produced per target-model call is:

$ E["tokens"] = frac(1 - alpha^(k+1), 1 - alpha) $

At $alpha = 0.8$, $k = 4$: $E["tokens"] approx 3.36$ — a $3.36 times$ reduction in target-model calls at no statistical cost.

When the draft model is much faster than the target (typical: a 7B draft vs a 70B target), the wall-clock speedup approaches $E["tokens"]$.

=== PyTorch Verify-and-Accept Loop

```python
import torch
import torch.nn.functional as F

def speculative_step(
    draft_model,
    target_model,
    input_ids: torch.Tensor,    # [1, context_len]
    k: int = 4,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Returns a tensor of accepted tokens (length 1..k+1).
    Preserves the exact distribution of target_model.
    """
    device = input_ids.device

    # --- Draft: sample k tokens autoregressively ---
    draft_tokens = []
    draft_probs  = []   # q(x_i | context), scalar at the sampled token
    draft_dists  = []   # full q(· | context), needed for the correction step
    ctx = input_ids
    with torch.no_grad():
        for _ in range(k):
            logits_d = draft_model(ctx)[:, -1, :] / temperature
            prob_d   = F.softmax(logits_d, dim=-1)            # [1, V]
            token    = torch.multinomial(prob_d[0], 1)         # [1]
            draft_tokens.append(token)
            draft_probs.append(prob_d[0, token[0]].item())
            draft_dists.append(prob_d[0].clone())
            ctx = torch.cat([ctx, token.unsqueeze(0)], dim=1)

    draft_ids = torch.stack(draft_tokens, dim=0).squeeze(-1)  # [k]

    # --- Target: single forward pass over k-token draft ---
    full_ids = torch.cat([input_ids,
                          draft_ids.unsqueeze(0)], dim=1)     # [1, L+k]
    with torch.no_grad():
        logits_t = target_model(full_ids)[:, -k-1:-1, :] / temperature
    probs_t = F.softmax(logits_t, dim=-1)  # [1, k, V]

    # --- Accept / reject ---
    accepted = []
    for i in range(k):
        x_i = draft_ids[i].item()
        p_i = probs_t[0, i, x_i].item()
        q_i = draft_probs[i]
        alpha = min(1.0, p_i / (q_i + 1e-9))
        u = torch.rand(1).item()
        if u <= alpha:
            accepted.append(draft_ids[i].unsqueeze(0))
        else:
            # Correction: sample from the residual max(0, p - q) over the
            # *whole* vocabulary, comparing against the full draft distribution
            corrected = F.relu(probs_t[0, i] - draft_dists[i])
            if corrected.sum() < 1e-9:
                corrected = probs_t[0, i].clone()
            corrected = corrected / corrected.sum()
            new_tok = torch.multinomial(corrected, 1)
            accepted.append(new_tok)
            return torch.cat(accepted)   # stop at first rejection

    # All k accepted: sample one bonus token from target
    bonus_logits = target_model(full_ids)[:, -1, :] / temperature
    bonus_probs  = F.softmax(bonus_logits, dim=-1)
    bonus_tok    = torch.multinomial(bonus_probs[0], 1)
    accepted.append(bonus_tok)
    return torch.cat(accepted)   # length k+1
```

*Practical notes:*
- Draft and target must share the same tokenizer.
- The draft model should be 5–10x smaller (e.g., Llama 3 8B drafts for Llama 3 70B).
- Acceptance rate depends strongly on task: chat/general $alpha approx 0.6$–$0.8$; code completion $approx 0.75$–$0.9$; reasoning/math $approx 0.5$–$0.7$; highly structured generation (JSON, fixed templates) can exceed $0.9$. Creative writing sits at the low end.
- Batched speculative decoding requires rejecting differently across batch elements; implementations maintain per-sequence state.

== Continuous Batching

=== Static Batching Problem

In static (offline) batching, a batch of $B$ sequences is processed together from prefill to the end of the longest sequence. Sequences that finish early must be padded with dummy tokens until the slowest sequence completes, wasting GPU compute proportional to the padding ratio.

```
Sequence A: [prompt]▓▓▓▓▓▓▓▓▓▓▓▓[EOS]░░░░░░░░░░░
Sequence B: [prompt]▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓[EOS]░░
Sequence C: [prompt]▓▓▓▓▓▓▓▓[EOS]░░░░░░░░░░░░░░░░░
                                   ← wasted padding →
```

=== Iteration-Level Scheduling (Orca)

Orca (Yu et al., 2022) proposed *continuous batching*: the batch is reassembled at every decode _iteration_. When a sequence emits EOS, its slot is immediately filled with a waiting request. No padding is ever inserted.

```
Iteration 1:  [A-token-5] [B-token-3] [C-token-8]
Iteration 2:  [A-token-6] [B-token-4] [C→EOS, D-prefill]
Iteration 3:  [A-token-7] [B-token-5] [D-token-1]
```

New requests undergo prefill (which is compute-bound) interleaved with decode steps of existing requests. The scheduler decides how many prefill tokens and how many decode steps to pack into each iteration based on available KV cache budget.

*Effect on GPU utilization:*

#table(
  columns: (auto, auto, auto),
  [*Metric*], [*Static batching*], [*Continuous batching*],
  [GPU utilization],    [30–50\% (variable seq lengths, padding waste)], [70–90\%],
  [Padding overhead],   [20–50\%], [~0\%],
  [Throughput (tok/s)], [baseline], [2–4x higher],
  [Scheduling unit],    [request], [iteration],
)


== Further Reading

Kwon, W., et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP. (vLLM; paged KV cache management.)

Dao, T., et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS. (IO-aware attention kernels.)

Leviathan, Y., Kalman, M., & Matias, Y. (2023). "Fast Inference from Transformers via Speculative Decoding." ICML. (Draft-and-verify decoding.)

Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." arXiv:1911.02150. (Multi-query attention; the basis for later GQA work.)

Pope, R., et al. (2023). "Efficiently Scaling Transformer Inference." MLSys. (Partitioning strategies and the latency/throughput frontier.)

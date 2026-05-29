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
    draft_probs  = []   # q(x_i | context)
    ctx = input_ids
    with torch.no_grad():
        for _ in range(k):
            logits_d = draft_model(ctx)[:, -1, :] / temperature
            prob_d   = F.softmax(logits_d, dim=-1)            # [1, V]
            token    = torch.multinomial(prob_d[0], 1)         # [1]
            draft_tokens.append(token)
            draft_probs.append(prob_d[0, token[0]].item())
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
            # Correction: sample from max(0, p - q)
            corrected = F.relu(probs_t[0, i] -
                                torch.tensor(
                                    [draft_probs[i] if j == x_i else 0.0
                                     for j in range(probs_t.shape[-1])],
                                    device=device))
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

== PagedAttention

=== Fragmentation Problem

With continuous batching, KV cache slots must be allocated and freed dynamically. Naive contiguous allocation produces external fragmentation: gaps between freed blocks cannot be reused for longer sequences, and worst-case utilization is around 20%.

=== Block Table Architecture

vLLM (Kwon et al., 2023) introduced *PagedAttention*, adapting virtual memory paging to KV cache management.

- The KV cache is divided into fixed-size *physical blocks* (e.g., 16 tokens per block).
- Each sequence maintains a *block table* mapping logical block indices to physical block indices.
- Physical blocks are allocated on demand and freed when a sequence ends.
- Fragmentation is at most one block per sequence (internal fragmentation only).

```
Logical view of sequence A:         Block table:
 Block 0  | Block 1  | Block 2      logical → physical
[tok 0-15]|[tok16-31]|[tok32-47]    0 → 7
                                    1 → 2
Physical memory:                    2 → 14
 [block 0][block 1][block 2]...
 [block 7: A0][block 8: B0][block 9: C0]
 ...
 [block 2: A1][block 14: A2]
```

=== Copy-on-Write for Beam Search

Beam search requires forking a sequence: two beams share a common prefix. PagedAttention implements *copy-on-write* — blocks are shared with a reference count. A block is physically copied only when one beam writes new tokens into it, identical to OS CoW semantics.

```
Beam 0 and Beam 1 share blocks 0–2 (ref_count=2).
When Beam 0 appends token to block 2:
  ref_count[2] == 2 → allocate new block 15, copy block 2 → 15.
  Beam 0 block table: ... → 15 (new)
  Beam 1 block table: ... → 2  (shared unchanged)
```

=== C++ Block Table

```cpp
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <cassert>

constexpr int BLOCK_SIZE = 16;  // tokens per physical block

struct PhysicalBlock {
    int ref_count = 0;
    // Actual KV tensors managed externally (GPU memory)
    int block_id;
};

struct BlockTable {
    std::vector<int> logical_to_physical;  // logical block → physical block id

    int n_logical_blocks() const {
        return (int)logical_to_physical.size();
    }

    // Append a new logical block mapped to a freshly allocated physical block
    void append_block(int physical_id) {
        logical_to_physical.push_back(physical_id);
    }

    int physical_id(int logical_block) const {
        assert(logical_block < (int)logical_to_physical.size());
        return logical_to_physical[logical_block];
    }
};

class BlockAllocator {
public:
    int total_blocks;
    std::vector<PhysicalBlock> blocks;
    std::vector<int> free_list;

    explicit BlockAllocator(int n) : total_blocks(n), blocks(n) {
        for (int i = n - 1; i >= 0; i--) {
            blocks[i].block_id = i;
            free_list.push_back(i);
        }
    }

    int allocate() {
        assert(!free_list.empty());
        int id = free_list.back();
        free_list.pop_back();
        blocks[id].ref_count = 1;
        return id;
    }

    // Copy-on-write: if ref_count > 1, allocate new block and copy.
    // Returns id of writable block.
    int cow_block(int src_id, /* copy callback */ auto copy_fn) {
        if (blocks[src_id].ref_count == 1) return src_id;
        int dst_id = allocate();
        copy_fn(src_id, dst_id);
        blocks[src_id].ref_count--;
        return dst_id;
    }

    void free(int id) {
        assert(blocks[id].ref_count > 0);
        if (--blocks[id].ref_count == 0)
            free_list.push_back(id);
    }
};
```

== Prefix Caching

Many LLM deployments share a long system prompt across thousands of requests. Without prefix caching, every request recomputes the KV representations for the system prompt, wasting proportional compute and time.

=== Hash-Based Reuse

Assign each block a content hash derived from the token ids it contains (and all ancestor block hashes, creating a *prefix hash chain*). Before allocating new blocks for a prefill, check whether a matching physical block already exists in a *prefix cache* (hash map from block hash to physical block id).

```python
import hashlib

def block_hash(token_ids: tuple[int, ...],
               parent_hash: bytes = b"") -> bytes:
    h = hashlib.sha256()
    h.update(parent_hash)
    for t in token_ids:
        h.update(t.to_bytes(4, "little"))
    return h.digest()

class PrefixCache:
    def __init__(self):
        # hash → physical block id (read-only, ref_counted)
        self._cache: dict[bytes, int] = {}

    def lookup(self, h: bytes) -> int | None:
        return self._cache.get(h)

    def insert(self, h: bytes, block_id: int):
        self._cache[h] = block_id

def prefill_with_cache(
    token_ids: list[int],
    prefix_cache: PrefixCache,
    allocator,   # BlockAllocator
    block_table: "BlockTable",
    block_size: int = 16,
) -> int:
    """
    Returns the index of the first token that needs actual compute.
    All earlier tokens were served from prefix cache.
    """
    parent_hash = b""
    hit_end = 0
    for block_start in range(0, len(token_ids), block_size):
        chunk = tuple(token_ids[block_start : block_start + block_size])
        if len(chunk) < block_size:
            break   # partial block — always recompute
        h = block_hash(chunk, parent_hash)
        cached_id = prefix_cache.lookup(h)
        if cached_id is not None:
            block_table.append_block(cached_id)
            allocator.blocks[cached_id].ref_count += 1
            hit_end = block_start + block_size
        else:
            break   # cache miss — stop looking ahead
        parent_hash = h
    return hit_end  # compute from this position onward
```

*Impact:* For an 8k-token system prompt shared by all requests, prefix caching eliminates its TTFT contribution entirely after the first request. Cache hit rates of 80–90\% are typical in chat deployments.

== Chunked Prefill

=== TTFT vs Throughput Tension

Prefill (processing the prompt) is compute-bound: a 4096-token prompt may take 50–100 ms on a single GPU. During this time, all in-flight decode requests stall — their inter-token latency spikes.

*Chunked prefill* (Agrawal et al., 2024) splits long prefill sequences into fixed-size chunks and interleaves them with decode steps.

=== Scheduling Diagram

```
Without chunked prefill:
 Iter 1: [prefill 4096 tokens] ← decodes stall 80 ms
 Iter 2: [decode A] [decode B] [decode C]

With chunked prefill (chunk=512):
 Iter 1: [prefill chunk 0-511]   + [decode A] [decode B]
 Iter 2: [prefill chunk 512-1023]+ [decode A] [decode B]
 ...
 Iter 8: [prefill chunk 3584-4095]+[decode A] [decode B]
 Iter 9: [decode A] [decode B] [decode C (new)]
```

Each iteration now fits within a bounded time budget, keeping ITL stable even while large prompts are being processed.

=== Scheduling Budget

The scheduler enforces two budgets per iteration:
- *Token budget* $T_"max"$: total tokens (prefill + decode) per iteration (e.g., 2048).
- *Sequence budget* $S_"max"$: total sequences per batch (e.g., 256).

Decode tokens always get priority (they each consume one token of budget); remaining budget is filled with prefill chunks.

```python
def schedule_iteration(
    waiting: list,       # pending prefill requests
    running: list,       # active decode sequences
    token_budget: int = 2048,
    seq_budget: int = 256,
) -> tuple[list, list]:
    """Returns (prefill_chunks, decode_seqs) for this iteration."""
    decode_seqs = running[:seq_budget]
    remaining   = token_budget - len(decode_seqs)  # 1 token per decode seq

    prefill_chunks = []
    for req in waiting:
        if remaining <= 0:
            break
        chunk_size = min(remaining, 512, req.remaining_tokens)
        prefill_chunks.append((req, chunk_size))
        req.remaining_tokens -= chunk_size
        remaining -= chunk_size

    return prefill_chunks, decode_seqs
```

== Tensor Parallelism for Serving

=== Motivation

A 70B model in bf16 requires $approx 140$ GiB, exceeding a single 80 GiB GPU. Tensor parallelism (TP) shards individual weight matrices across $N$ GPUs, each holding $1/N$ of every layer. For inference this is more latency-efficient than pipeline parallelism because there is no pipeline bubble.

=== Attention Head Sharding

With TP=8 across 8 GPUs connected via NVLink, the $n_"heads" = 64$ attention heads are split into groups of 8 per GPU. Each GPU computes attention for its local heads independently. The output projections require an all-reduce to sum partial results.

```
GPU 0: heads 0–7   → local O_0 [B, L, d_model/8]
GPU 1: heads 8–15  → local O_1
...
GPU 7: heads 56–63 → local O_7
AllReduce(O_0, ..., O_7) → O [B, L, d_model]
```

=== FFN Column/Row Split (Megatron-LM Style)

The two-layer FFN (gate and up projections followed by down projection in SwiGLU) is split as:
- *Column parallel*: gate/up weights split along the output dimension — no communication needed before the activation.
- *Row parallel*: down weight split along the input dimension — requires all-reduce after.

```
d_ffn = 14336  (LLaMA 3 8B)
TP=8 → each GPU holds 14336/8 = 1792 intermediate features

GPU g:
  gate_g = x @ W_gate[g]    # [B, L, 1792]
  up_g   = x @ W_up[g]      # [B, L, 1792]
  h_g    = silu(gate_g) * up_g
  y_g    = h_g @ W_down[g]  # [B, L, d_model]
AllReduce(y_0, ..., y_7) → y  # sum partial results
```

=== PyTorch Distributed Implementation

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

class TensorParallelLinear(nn.Module):
    """Column-parallel linear: weight split on output dim."""
    def __init__(self, in_features: int, out_features: int,
                 tp_group: dist.ProcessGroup):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size  = dist.get_world_size(tp_group)
        assert out_features % self.tp_size == 0
        local_out = out_features // self.tp_size
        self.weight = nn.Parameter(
            torch.empty(local_out, in_features))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, in_features] (already replicated across GPUs)
        return F.linear(x, self.weight)   # [B, L, local_out]


class TensorParallelRowLinear(nn.Module):
    """Row-parallel linear: weight split on input dim; all-reduce output."""
    def __init__(self, in_features: int, out_features: int,
                 tp_group: dist.ProcessGroup):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size  = dist.get_world_size(tp_group)
        assert in_features % self.tp_size == 0
        local_in = in_features // self.tp_size
        self.weight = nn.Parameter(
            torch.empty(out_features, local_in))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, local_in] — each GPU has its own partial input
        y_local = F.linear(x, self.weight)   # [B, L, out_features]
        dist.all_reduce(y_local, group=self.tp_group)
        return y_local   # now holds the complete sum


def init_tp(tp_size: int = 8) -> dist.ProcessGroup:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    tp_ranks = list(range(tp_size))
    return dist.new_group(tp_ranks)
```

*NVLink bandwidth:* A100/H100 NVLink provides 600 GB/s bidirectional bandwidth. An all-reduce for a 4096-dimensional vector at bf16 across 8 GPUs transfers $2 times (8-1)/8 times 4096 times 2 approx 14$ KiB — well under 10 µs, negligible relative to the GEMM itself.

*Practical scaling limits:* Real NVLink all-reduce achieves only $approx 80%$ of theoretical bandwidth due to protocol overhead and ring/tree algorithm inefficiencies. TP $> 8$ typically requires NVSwitch (within an HGX node) or crosses node boundaries onto InfiniBand, where latency is 5–10$times$ higher and per-step all-reduce cost can dominate the GEMM. Most production deployments cap TP at 8 and combine with pipeline or expert parallelism beyond that.

== Metrics and Measurement

=== Definitions

#table(
  columns: (auto, auto, auto),
  [*Metric*], [*Definition*], [*Typical target*],
  [TTFT], [Time from request receipt to first output token], [less than 200 ms (interactive)],
  [ITL], [Time between consecutive output tokens], [less than 30 ms / token],
  [Throughput], [Output tokens per second per GPU], [maximize for batch workloads],
  [Prefill throughput], [Prompt tokens per second per GPU], [secondary; amortized over output],
)

TTFT is dominated by prefill latency (and queuing delay). ITL is dominated by the decode forward pass, which is memory-bandwidth-bound for small batch sizes: the bottleneck is loading weights ($approx 140$ GiB for 70B bf16) from HBM, not arithmetic.

*Memory-bandwidth bound:* At batch size $B=1$, each decode step loads all model weights once. On an H100 (3.35 TB/s HBM3), a 70B bf16 model decodes at:

$ "ITL"_"min" approx frac(140 times 10^9 times 2, 3.35 times 10^12) approx 84 "ms" $

Increasing batch size amortizes weight loads: at $B=32$, effective per-token memory traffic is $1/32$ and ITL approaches 2–3 ms.

=== PyTorch CUDA Event Measurement

```python
import torch
import time

class LatencyMeter:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event   = torch.cuda.Event(enable_timing=True)

    def measure_ttft(self, model, input_ids: torch.Tensor) -> float:
        """Returns TTFT in milliseconds."""
        torch.cuda.synchronize()
        self.start_event.record()
        with torch.no_grad():
            _ = model(input_ids)   # prefill
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)

    def measure_itl(self, model,
                    input_ids: torch.Tensor,
                    n_tokens: int = 50) -> tuple[float, float]:
        """
        ITL = (T_total - TTFT) / (n_output_tokens - 1).

        We time the prefill (TTFT) and the full generation of n_tokens output
        tokens, then derive ITL as the average inter-token gap across the
        n_tokens-1 transitions between consecutive output tokens. This matches
        how online serving systems report ITL — it is *not* simply the time of
        the last decode step.
        """
        ctx = input_ids
        # --- TTFT: prefill timing ---
        t_prefill_start = torch.cuda.Event(enable_timing=True)
        t_prefill_end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with torch.no_grad():
            t_prefill_start.record()
            _ = model(ctx)
            t_prefill_end.record()
            torch.cuda.synchronize()
        ttft_ms = t_prefill_start.elapsed_time(t_prefill_end)

        # --- Total decode timing for n_tokens output tokens ---
        t_total_start = torch.cuda.Event(enable_timing=True)
        t_total_end   = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            t_total_start.record()
            for _ in range(n_tokens):
                logits = model(ctx[:, -1:])
                next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
                ctx = torch.cat([ctx, next_tok], dim=1)
            t_total_end.record()
            torch.cuda.synchronize()
        decode_total_ms = t_total_start.elapsed_time(t_total_end)

        # ITL is the mean inter-token gap: there are (n_tokens - 1) gaps
        # between n_tokens output tokens.
        assert n_tokens >= 2
        mean_itl = decode_total_ms / (n_tokens - 1)
        throughput = 1000.0 / mean_itl   # tokens/sec
        return mean_itl, throughput
```

=== Benchmark Summary

#table(
  columns: (auto, auto, auto, auto, auto),
  [*System*], [*Model*], [*Hardware*], [*Throughput (tok/s/GPU)*], [*TTFT (ms)*],
  [vLLM 0.6+], [LLaMA 3 70B], [8x H100], [2 800], [60–120],
  [TensorRT-LLM], [LLaMA 3 70B], [8x H100], [3 400], [40–80],
  [naive static], [LLaMA 3 70B], [8x H100], [900], [200–600],
  [vLLM 0.6+], [LLaMA 3 8B], [1x H100], [4 200], [15–40],
)

Numbers are approximate; vary with batch size, sequence length, and prefill ratio.

== References

- Leviathan, Y., Kalman, M., & Matias, Y. (2023). *Fast inference from transformers via speculative decoding.* ICML 2023.
- Yu, G., et al. (2022). *Orca: A distributed serving system for Transformer-based generative models.* OSDI 2022.
- Kwon, W., et al. (2023). *Efficient memory management for large language model serving with PagedAttention.* SOSP 2023.
- Holtzman, A., et al. (2020). *The curious case of neural text degeneration.* ICLR 2020.
- Shoeybi, M., et al. (2019). *Megatron-LM: Training multi-billion parameter language models using model parallelism.* arXiv:1909.08053.
- Agrawal, A., et al. (2024). *Taming throughput-latency tradeoff in LLM inference with Sarathi-Serve.* OSDI 2024.
- Pope, R., et al. (2023). *Efficiently scaling transformer inference.* MLSys 2023. (Introduces GQA and KV cache analysis.)

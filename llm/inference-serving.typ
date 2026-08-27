#import "../template.typ": xref

= Inference Serving <inference-serving>

*Serving-level optimization:* This chapter covers the techniques that govern how LLM inference servers manage memory, schedule requests, and achieve high throughput in production. It continues from #xref("llm", "inference-optimization", label: "Inference Optimization") (KV cache, decoding strategies, speculative decoding, continuous batching).

*See also:* #xref("llm", "inference-optimization", label: "Inference Optimization") (KV cache, batching fundamentals), #xref("llm", "transformer-architecture", label: "Transformer Architecture"), #xref("gpu-architecture", "memory-hierarchy", label: "GPU Memory Hierarchy (GPU Architecture volume)").

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

*NVLink bandwidth:* A100 NVLink 3 provides 600 GB/s bidirectional bandwidth per GPU (H100 NVLink 4: 900 GB/s). An all-reduce for a 4096-dimensional vector at bf16 across 8 GPUs transfers $2 times (8-1)/8 times 4096 times 2 approx 14$ KiB — well under 10 µs, negligible relative to the GEMM itself.

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

$ "ITL"_"min" approx frac(140 times 10^9, 3.35 times 10^12) approx 42 "ms" $

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

= Long Context

Frontier models advertise context windows of 128k, 1M, even 10M tokens. Achieving these requires solving two distinct problems: (1) the *positional encoding* must extrapolate or interpolate beyond training length without losing locality; (2) the *attention compute and memory* must stay tractable as $L$ grows, since vanilla attention is $O(L^2)$ and the KV cache is $O(L)$ per layer. This chapter covers RoPE and its scaling variants (Position Interpolation, NTK-aware, YaRN, LongRoPE), block-sparse attention (sliding window, sink tokens, hybrid layers), parallelism for long sequences (Ring Attention, Tree Attention), and the practical tradeoff between long context and retrieval-augmented generation.

*See also:* _Transformer Architecture_ (attention internals, RoPE), _Inference Optimization_ (KV cache memory), _RAG_ (alternative to long context for some tasks).

== The Two Bottlenecks

For a sequence of length $L$, model dimension $d$, and $n_h$ attention heads:

- *Attention compute*: $O(L^2 d)$ per layer. At $L = 128"k"$, $d = 4096$, this is $approx 70$ TFLOPs per layer per forward — meaningful even on H100.
- *KV cache memory*: $2 times n_"layers" times n_"kv heads" times d_"head" times L times B_"dtype"$. For LLaMA 3 8B at $L = 128"k"$: 16 GiB.

Long-context engineering attacks both. Architectural choices (RoPE scaling, sparse attention) determine *whether the model can generalize* beyond training length; system choices (paged KV, ring attention, chunked prefill) determine *whether it can be served* affordably.

== Positional Encoding for Long Contexts

=== Why RoPE Extrapolates Poorly

Rotary Position Embeddings (Su et al. 2021) encode position $m$ in the query/key by rotating each pair of dimensions by angle $m theta_i$ where $theta_i = 10000^(-2i / d)$. The inner product between query at position $m$ and key at position $n$ becomes a function of $m - n$.

Training at length $L_"train"$ exposes the model only to rotations up to $L_"train" theta_i$. At inference position $L > L_"train"$, the low-frequency dimensions (small $theta_i$) see angles outside the trained range — the model behaves erratically. Empirically, perplexity explodes within a few hundred tokens past training length.

=== Position Interpolation (PI)

Chen et al. (2023) observed that if we simply *interpolate* positions:

$ m' = m times L_"train" / L_"target" $

every dimension sees only angles within the trained range. A short fine-tune (~1k steps) at $L_"target"$ recovers performance. Trades extrapolation for slight loss of fine-grained position resolution. Used by LLaMA 2 to extend 4k → 32k.

=== NTK-Aware Scaling

The PI approach interpolates all frequencies uniformly. NTK theory (Bloc97 2023, Reddit/community) suggests *high-frequency* dimensions (large $theta_i$, short wavelength) should not be interpolated — those encode local position and lose precision under interpolation. NTK-aware scaling adjusts the base frequency:

$ theta'_i = ("base" times s)^(-2i/d) "where " s = (L_"target" / L_"train")^(d/(d-2)) $

so high frequencies remain near their original values while low frequencies stretch. Better extrapolation, less fine-tune required.

=== YaRN

YaRN (Peng et al. 2023) refines NTK-aware with two ideas:

1. *Ramp scaling:* identify dimensions whose wavelength exceeds $L_"train"$ (these need full PI scaling) and those whose wavelength is much shorter (no scaling). Interpolate the middle band smoothly.
2. *Attention temperature:* multiply the attention logits by a length-dependent constant
   $ t = 0.1 ln(s) + 1 $
   to compensate for the entropy increase of longer sequences. Equivalent to keeping the softmax sharpness constant.

YaRN trained on 64k tokens generalizes well to 128k+ and is the basis for Qwen2 (128k), Mistral 7B v0.2 (32k), and Nous Capybara (128k).

=== LongRoPE

LongRoPE (Ding et al. 2024) learns *per-dimension* rescaling factors $lambda_i$ via evolutionary search on a small calibration corpus:

$ theta'_i = lambda_i^(-1) times theta_i $

Yielded 2M-token models. The search space ($lambda_i$ per dim) is small enough for population-based search to converge in hours.

=== Comparison

| Method | Fine-tune cost | Quality at 4× train length | Quality at 16× |
|---|---|---|---|
| Vanilla RoPE | — | broken | broken |
| PI | ~1k steps | OK | poor |
| NTK-aware | none → ~100 steps | good | OK |
| YaRN | ~400 steps | very good | good |
| LongRoPE | search + ~1k steps | excellent | excellent |

== Sparse and Local Attention

Reducing quadratic attention to $O(L log L)$ or $O(L)$ has been studied extensively (Longformer, BigBird, Reformer, Performer). For LLMs, the surviving patterns are:

=== Sliding Window Attention (SWA)

Each token attends to a fixed window of $w$ preceding tokens. Mistral 7B uses $w = 4096$. The attention pattern is banded; FlashAttention has dedicated kernels for windowed attention.

Stacking $n$ layers gives a *receptive field* of $n times w$ — a 32-layer model with $w = 4096$ can in principle attend over 128k tokens of context. In practice, the "telephone game" attenuates signal across layers; SWA models lose long-range capabilities that dense attention preserves.

=== Attention Sinks

Xiao et al. (2023) discovered that the first few tokens of any sequence absorb disproportionate attention mass — they act as "sinks" because softmax must place its mass somewhere. Removing these tokens from the KV cache (typical for streaming) collapses generation. *StreamingLLM* keeps the first 4 tokens permanently in the cache plus a sliding window:

```
[sink_0, sink_1, sink_2, sink_3, ...truncated..., kv_cache window of last W tokens]
```

This enables effectively infinite streaming with bounded memory and acceptable quality. Mistral incorporates this; Hugging Face calls it "windowed + sink".

=== Hybrid Layers

Recent models alternate full and local attention:

- *Mistral 7B*: every layer sliding-window 4k. Effective context 32k via layer-stacking.
- *Mixtral 8×7B*: same.
- *Gemma 2*: alternates 4096-window and full-attention layers.
- *Llama 3.1*: every layer is full attention; long context achieved by RoPE scaling and post-training (rather than sparsity).

The hybrid approach trades raw long-range capability for KV-cache savings (sliding-window layers cache only $w$ tokens).

=== Selective / Learned Sparsity

Native Sparse Attention (NSA — DeepSeek 2025) trains the model to select a small subset of past tokens to attend to per query. Routing is learned end-to-end; both training and inference run on the sparse pattern, avoiding the train-dense / serve-sparse mismatch of earlier sparse models. DeepSeek-V3.2-Exp uses NSA for 128k context at sub-quadratic cost.

== Compressive and Recurrent State

Two non-attention approaches deserve mention:

- *State-space models (SSMs)* — Mamba (Gu–Dao 2024), RWKV — model the sequence with a recurrent state of fixed size. Compute is $O(L)$, memory $O(1)$ in $L$. Quality on retrieval-style benchmarks lags transformer LLMs, but hybrid architectures (Jamba, Zamba) interleave Mamba and attention layers and close the gap.
- *Compressive memory* — Infini-attention (Munkhdalai et al. 2024) — a fixed-size *associative memory* alongside standard attention. Tokens beyond a sliding window are written to / retrieved from the compressive memory. Conceptually clean but quality at 1M+ tokens is fragile.

== System-Level Long-Context Training

=== Ring Attention

Ring Attention (Liu et al. 2023) trains transformers on sequences that do not fit on a single GPU by splitting $Q, K, V$ across $P$ devices along the sequence dimension. Each device holds $L / P$ tokens. Attention is computed in $P$ rounds: in round $r$, each device sends its $K, V$ block to the next device in a ring and receives one. After $P$ rounds, every device has computed full attention.

Communication is overlapped with compute (FlashAttention is bandwidth-friendly), so total time is dominated by attention FLOPs, not all-reduce. Combined with sequence parallelism for FFN/LayerNorm, this trains 1M+ context on commodity 8×H100.

=== Tree Attention

Tree Attention (Liu–Han 2024) generalizes Ring to a tree topology, reducing communication latency from $O(P)$ to $O(log P)$ rounds at the cost of slightly higher peak memory. Used for context > 4M tokens on 64+ GPUs.

=== Chunked Prefill

At inference, the prefill phase (encoding the prompt) is compute-bound while decode is memory-bandwidth-bound. *Chunked prefill* (vLLM, TGI) splits a long prompt into chunks of, say, 4k tokens. Each chunk's prefill is interleaved with ongoing decode requests, keeping the GPU saturated and tail latency bounded. Without chunking, a 128k prefill blocks all decoders for seconds.

```python
def chunked_prefill(prompt_ids, chunk=4096):
    state = empty_kv_cache()
    for start in range(0, len(prompt_ids), chunk):
        block = prompt_ids[start:start+chunk]
        forward(block, state)        # extend KV cache, no token sampling
    return state                     # ready for decode
```

== Long-Context Evaluation

Pass rates on conventional NLP benchmarks saturate quickly. The standard long-context probes:

- *Needle in a Haystack* (NIAH — Kamradt 2023): insert a sentence at varying depth in a long context; ask the model to retrieve it. Easy for capable models; mostly tests *position-of-retrieval*, not reasoning.
- *RULER* (Hsieh et al. 2024): NIAH variants, multi-hop retrieval, aggregation, variable-tracking. More discriminating; many models pass NIAH but degrade on RULER at >32k.
- *InfiniteBench, LongBench, $infinity$Bench*: human-curated long-form tasks (summarization, QA over books).
- *LooGLE, LongGenBench*: tests long-output capabilities, not just long-input.

A model that scores 100% on NIAH may still drop 30+ points on RULER aggregation tasks. Publish both.

== Long Context vs. RAG

For factual lookup over a knowledge base, RAG (chunked retrieval + small context) is typically *cheaper, more current, and more verifiable* than packing the whole corpus into a long context. Long context wins when:

- The task is *integrative* — summarize a book, refactor a large codebase, follow a multi-thread conversation.
- The retrieval step itself requires deep semantic understanding ("find the contradictions across these 12 documents").
- The system must handle inputs that genuinely cannot be chunked (long videos, unstructured PDFs with cross-references).

Hybrid systems are common: a long-context model with RAG-injected snippets at the start of the prompt. This pattern minimizes hallucination while keeping the model's integrative reasoning available.

== Practical Setup

```python
# Hugging Face: load LLaMA 3.1 with RoPE scaling (YaRN-equivalent)
from transformers import AutoModelForCausalLM

cfg_overrides = {
    "rope_scaling": {
        "rope_type": "llama3",
        "factor": 8.0,                # 8k → 128k
        "low_freq_factor":  1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    },
}
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    torch_dtype="bfloat16",
    **cfg_overrides,
    attn_implementation="flash_attention_2",
)
```

Serving (vLLM):

```bash
vllm serve meta-llama/Llama-3.1-8B \
  --max-model-len 128000 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92
```

`--enable-chunked-prefill` is essential; `max-num-batched-tokens` sets the chunk size.

== Further Reading

Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864.

Chen, S. et al. (2023). "Extending Context Window of Large Language Models via Positional Interpolation." arXiv:2306.15595.

Peng, B. et al. (2023). "YaRN: Efficient Context Window Extension of Large Language Models." ICLR 2024.

Ding, Y. et al. (2024). "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens." ICML 2024.

Xiao, G. et al. (2023). "Efficient Streaming Language Models with Attention Sinks." ICLR 2024.

Liu, H., Zaharia, M., Abbeel, P. (2023). "Ring Attention with Blockwise Transformers for Near-Infinite Context." arXiv:2310.01889.

Hsieh, C. et al. (2024). "RULER: What's the Real Context Size of Your Long-Context Language Models?" arXiv:2404.06654.

Gu, A., Dao, T. (2024). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." COLM.

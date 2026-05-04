= Serving Systems

Inference serving bridges the gap between a trained model checkpoint and a production HTTP endpoint. A serving system must handle concurrent requests, manage GPU memory for KV caches, enforce latency SLAs, and sustain throughput measured in millions of tokens per second. This chapter covers the full stack: from request lifecycle to scheduler internals, with concrete numbers on H100 80 GB hardware running LLaMA 3 8B in BF16.

*See also:* _gpu-architecture/ml-workloads.typ_ (Flash Attention, KV cache layout), _gpu-architecture/memory-hierarchy.typ_ (HBM bandwidth, coalescing), _llm/transformer-architecture.typ_ (attention internals).

== Serving Stack Overview

A production LLM serving request travels through many layers before reaching the GPU:

```
Client (browser / API caller)
        │  HTTP POST /v1/completions
        ▼
Load Balancer  (nginx / Envoy)
        │  sticky routing by session or round-robin
        ▼
API Server  (FastAPI / uvicorn)
        │  parse JSON, validate token budget
        ▼
Tokenizer  (Rust tiktoken / HuggingFace fast tokenizer)
        │  text → token ids  [B, L_prompt]
        ▼
Scheduler  (continuous batching loop, priority queue)
        │  assign KV cache pages, form micro-batch
        ▼
Prefill Phase  (GPU forward pass, full prompt)
        │  compute + store KV for all prompt tokens
        ▼
Decode Loop  (autoregressive, one token per step)
        │  per-step: attend over cached KV → logit → sample → append
        ▼
Detokenizer  (token ids → text, streaming via SSE)
        │  HTTP chunked / Server-Sent Events
        ▼
Client receives streamed tokens
```

*Phase distinction:* _prefill_ processes the entire prompt in parallel (compute-bound, high arithmetic intensity). _Decode_ generates one token at a time per sequence (memory-bound: reads all KV cache and weights for each step). This asymmetry is the root cause of most serving challenges.

*GPU memory budget* (H100 80 GB, LLaMA 3 8B BF16):

#table(
  columns: (auto, auto, auto),
  [*Component*], [*Size*], [*Notes*],
  [Model weights], [16 GB], [8B params × 2 bytes/param (BF16)],
  [CUDA / framework overhead], [~2 GB], [PyTorch allocator, NCCL buffers],
  [KV cache (remainder)], [~62 GB], [paged or contiguous],
  [KV per token (LLaMA 3 8B)], [0.5 MB], [32 layers × 2 × 8 kv-heads × 128 dim × 2 bytes],
  [Max tokens in KV cache], [~124k], [62 GB / 0.5 MB per token],
)

== vLLM

vLLM (2023, UC Berkeley) introduced _PagedAttention_ and _continuous batching_, enabling near-zero KV cache fragmentation and GPU utilization above 80\% on production traffic.

=== Architecture

```
┌─────────────────────────────────────────────────────┐
│                    LLMEngine                        │
│                                                     │
│  ┌──────────────┐    ┌──────────────────────────┐   │
│  │  Scheduler   │───▶│    BlockManager          │   │
│  │              │    │  (logical → physical      │   │
│  │ priority q   │    │   page table per seq)     │   │
│  │ preemption   │    └──────────────────────────┘   │
│  └──────┬───────┘                                   │
│         │ batch of seq_groups                        │
│         ▼                                           │
│  ┌──────────────────────────────────────────────┐   │
│  │   Worker (one per GPU, ray actor or thread)  │   │
│  │   CacheEngine  ←  block_tables (ptr arrays)  │   │
│  │   ModelRunner  →  PagedAttention kernel      │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

*Scheduler:* runs a continuous batching loop. At every decode step it checks: (1) new requests in the waiting queue, (2) which running sequences can be preempted (swap to CPU) to free blocks for higher-priority work, (3) how many new prefill tokens fit given available blocks.

*BlockManager:* manages a pool of physical KV-cache _blocks_ (default 16 tokens each). A sequence holds a list of logical block numbers; the block manager maps these to physical GPU memory pages. Copy-on-write allows prompt sharing (e.g. system-prompt prefix caching).

*PagedAttention:* custom CUDA kernel that gathers non-contiguous KV blocks during attention computation. The kernel receives a block table (array of physical block pointers) per sequence and reads KV from scattered locations, avoiding the need for contiguous memory allocation.

=== Continuous Batching

Classic static batching waits until a fixed batch is full, then runs a single forward pass. Every sequence in the batch must finish before the batch returns. Short sequences stall on long ones: GPU sits idle when short sequences complete.

Continuous batching (also called _iteration-level scheduling_) processes one decode step per iteration, adding new requests and retiring finished ones between steps. The batch composition changes every step.

```
Iteration 1:  [Seq A (step 3), Seq B (step 1), Seq C (step 7)]
Iteration 2:  [Seq A (step 4), Seq B (step 2), Seq C done → evict, Seq D added]
Iteration 3:  [Seq A (step 5), Seq B (step 3), Seq D (step 1)]
```

GPU utilization rises because there is always work to fill the batch.

=== Launch Command

```bash
# Install
pip install vllm

# Serve LLaMA 3 8B on a single H100
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 256 \
    --port 8000

# Multi-GPU tensor parallel (2× H100)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --port 8000
```

=== Python Client Example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="ignored")

# Non-streaming completion
response = client.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt="Explain PagedAttention in one paragraph:",
    max_tokens=200,
    temperature=0.7,
)
print(response.choices[0].text)

# Streaming chat
stream = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    messages=[{"role": "user", "content": "What is KV cache?"}],
    max_tokens=300,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

=== Throughput Numbers (H100 80 GB, LLaMA 3 8B BF16)

#table(
  columns: (auto, auto, auto, auto),
  [*Workload*], [*Batch size*], [*Throughput (tok/s)*], [*Notes*],
  [Offline (ShareGPT)], [256], [~18,000], [LLaMA 3 8B, A100 80GB, in=128/out=128, vLLM v0.6+],
  [Online (TTFT \u{2264} 500 ms)], [32–64], [~8,000–11,000], [SLA-constrained],
  [Long context (8k prompt)], [16], [~5,000], [prefill dominates],
  [TensorRT-LLM (same HW)], [256], [~24,000], [LLaMA 3 8B FP8, H100 80GB, in=128/out=128, batch=256],
)

Numbers from vLLM benchmark suite (v0.4.x) and NVIDIA NIM documentation. Throughput scales roughly linearly with GPU count for tensor-parallel up to 8× H100.

== TensorRT-LLM

TensorRT-LLM is NVIDIA's production inference library. It compiles a model into an optimized TensorRT engine with fused kernels, quantized weights, and static computation graphs. It underpins NVIDIA NIM microservices.

=== Key Optimizations

*Graph compilation:* the model is traced into a TensorRT network, eliminating Python overhead and enabling cross-layer fusion (e.g. layer norm + QKV projection fused into one kernel).

*Kernel fusion:* Flash Attention, RoPE embedding, and RMSNorm are fused. The MLP (gate × up, SiLU, down) is a single kernel call. Fused kernels reduce memory round-trips: intermediate tensors never hit HBM.

*Quantization support:*

#table(
  columns: (auto, auto, auto),
  [*Format*], [*Precision*], [*Use case*],
  [BF16], [16-bit brain float], [baseline quality],
  [FP8 (E4M3)], [8-bit float], [H100/H200, ~2× throughput vs BF16],
  [INT4 (AWQ / GPTQ)], [4-bit integer], [weight-only quant, fits larger models],
  [INT4-FP8 (mixed)], [activations FP8, weights INT4], [maximum throughput, some quality loss],
)

*In-flight batching:* TensorRT-LLM implements its own version of continuous batching via the `GptManager` API, with a C++ scheduler that can be driven from Python.

=== Build Example: LLaMA 3 8B

```python
# Step 1: Convert weights to TensorRT-LLM checkpoint format
import subprocess

# Using the official conversion script (tensorrt_llm repo)
subprocess.run([
    "python", "examples/llama/convert_checkpoint.py",
    "--model_dir", "/models/Meta-Llama-3-8B-Instruct",
    "--output_dir", "/trt_ckpt/llama3-8b-fp8",
    "--dtype", "bfloat16",
    "--use_fp8_rowwise",           # FP8 weight + activation quant
    "--calib_dataset", "cnn_dailymail",
])

# Step 2: Build the TensorRT engine
subprocess.run([
    "trtllm-build",
    "--checkpoint_dir", "/trt_ckpt/llama3-8b-fp8",
    "--output_dir", "/trt_engines/llama3-8b-fp8",
    "--gemm_plugin", "float16",
    "--max_batch_size", "256",
    "--max_input_len", "4096",
    "--max_output_len", "2048",
    "--use_paged_context_fmha", "enable",
    "--workers", "1",
])
```

```python
# Step 3: Run inference with the compiled engine
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunnerCpp

runner = ModelRunnerCpp.from_dir(
    engine_dir="/trt_engines/llama3-8b-fp8",
    rank=0,
)

input_ids = [[128000, 15339, 1917, 128009]]   # tokenized "Hello world<|eot_id|>"
outputs = runner.generate(
    batch_input_ids=input_ids,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.9,
)
print(outputs.output_ids)
```

The compiled FP8 engine on a single H100 80GB achieves approximately 24,000 tokens/s output throughput for LLaMA 3 8B at batch size 256 (input=128/output=128, FP8 weights+activations), versus ~18,000 for vLLM BF16 under comparable conditions — a ~1.3× gain from quantization and kernel fusion combined.

== TGI (Text Generation Inference)

Hugging Face Text Generation Inference is a production-ready Rust + Python serving framework, the backend for Hugging Face Inference Endpoints.

=== Key Features

*Flash Attention 2:* TGI integrates Flash Attention 2 by default for all supported architectures. Memory usage scales $O(L)$ rather than $O(L^2)$; TTFT is substantially lower for long prompts.

*Tensor parallelism:* distributes each matrix multiply across $p$ GPUs using `torch.distributed`. Each GPU holds $1 slash p$ of each weight shard; an all-reduce synchronizes after each layer. TGI uses Safetensors sharded checkpoints.

*Token streaming:* TGI uses Server-Sent Events (SSE) over HTTP. Each generated token is flushed immediately, enabling streaming UIs with low perceived latency independent of total generation length.

*Continuous batching:* TGI's Rust router implements waiting-queue management, merging prefill and decode steps with dynamic batching.

=== Docker Launch

```bash
# Single H100, LLaMA 3 8B Instruct, BF16, port 8080
docker run --gpus '"device=0"' \
    --shm-size 64g \
    -p 8080:80 \
    -v /models:/data \
    ghcr.io/huggingface/text-generation-inference:2.4 \
    --model-id meta-llama/Meta-Llama-3-8B-Instruct \
    --dtype bfloat16 \
    --max-input-tokens 4096 \
    --max-total-tokens 6144 \
    --max-batch-prefill-tokens 16384 \
    --num-shard 1

# Multi-GPU: 70B on 4× H100
docker run --gpus all \
    --shm-size 128g \
    -p 8080:80 \
    -v /models:/data \
    ghcr.io/huggingface/text-generation-inference:2.4 \
    --model-id meta-llama/Meta-Llama-3-70B-Instruct \
    --num-shard 4 \
    --dtype bfloat16
```

```bash
# Query with curl (streaming)
curl http://localhost:8080/generate_stream \
    -H "Content-Type: application/json" \
    -d '{"inputs": "What is tensor parallelism?",
         "parameters": {"max_new_tokens": 200, "temperature": 0.7}}'
```

```python
# Python client (huggingface_hub)
from huggingface_hub import InferenceClient

client = InferenceClient("http://localhost:8080")

for token in client.text_generation(
    "Explain KV cache in two sentences.",
    max_new_tokens=150,
    stream=True,
):
    print(token, end="", flush=True)
```

== SGLang

SGLang (Structured Generation Language) is a serving system optimized for multi-turn programs, structured outputs, and agent workloads with heavy prompt reuse.

=== RadixAttention

The central innovation is _RadixAttention_: a radix tree of KV cache blocks, where shared prefixes (system prompts, few-shot examples, tool definitions) are stored once and reused across requests. In contrast, vLLM's prefix caching applies only to exact-match prefixes at the block boundary; RadixAttention tracks prefix sharing at arbitrary token granularity.

```
Radix tree of KV cache blocks:
                    [system prompt tokens: 512]
                   /                           \
   [user turn A (128)]                  [user turn B (96)]
        /        \                            |
[resp A1 (64)] [resp A2 (80)]          [resp B1 (72)]
```

When a new request arrives with the same system prompt, its KV cache for those 512 tokens is already populated. Prefill cost for those tokens is zero.

=== Structured Generation

SGLang integrates constrained decoding via _compressed finite state machines_ (FSMs). A JSON schema is compiled into an FSM; at each decode step, only tokens that keep the FSM in a valid state are allowed. This eliminates post-hoc parsing and retry loops.

```python
import sglang as sgl

@sgl.function
def extract_info(s, document: str):
    s += sgl.system("You are an information extraction assistant.")
    s += sgl.user(f"Extract structured info from: {document}")
    s += sgl.assistant(
        sgl.gen(
            "result",
            max_tokens=256,
            # Constrain output to valid JSON matching this schema
            json_schema={
                "type": "object",
                "properties": {
                    "name":    {"type": "string"},
                    "date":    {"type": "string", "format": "date"},
                    "amount":  {"type": "number"},
                    "status":  {"type": "string", "enum": ["paid", "pending", "overdue"]},
                },
                "required": ["name", "date", "amount", "status"],
            },
        )
    )
    return s

# Launch SGLang server
# python -m sglang.launch_server \
#     --model-path meta-llama/Meta-Llama-3-8B-Instruct \
#     --port 30000 --dp-size 1

sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

state = extract_info.run(document="Invoice from Acme Corp, $1,250 due 2024-03-15, unpaid.")
import json
data = json.loads(state["result"])
print(data)
# {"name": "Acme Corp", "date": "2024-03-15", "amount": 1250.0, "status": "pending"}
```

=== Fork/Join Parallelism

SGLang supports `fork` for sampling multiple independent continuations from the same prefix — useful for best-of-N sampling, tree-search, or parallel chain-of-thought:

```python
@sgl.function
def best_of_n(s, question: str, n: int = 4):
    s += sgl.user(question)
    forks = s.fork(n)                       # n independent samples
    for fork in forks:
        fork += sgl.assistant(sgl.gen("answer", max_tokens=128, temperature=0.8))
    s.join(forks)
    answers = [f["answer"] for f in forks]
    return answers
```

The KV cache for the shared prefix (the question) is computed once; all $n$ forks share those blocks via copy-on-write, then diverge independently.

== Throughput vs Latency Tradeoffs

=== Little's Law Applied to LLM Serving

Little's Law from queueing theory states: $L = lambda W$, where $L$ is mean number of items in the system, $lambda$ is throughput (arrivals/sec), and $W$ is mean time in system (latency). For an LLM serving system:

$
"Concurrent requests" = "Request rate" times "Mean latency per request"
$

*Example:* if mean end-to-end latency is 2 s at steady state, and the system sustains 50 requests/s, then on average $50 times 2 = 100$ requests are in-flight simultaneously. Each in-flight request holds KV cache blocks. At 0.5 MB/token and 200 output tokens, that is 100 × 200 × 0.5 MB = 10 GB of KV cache just for output tokens.

=== Time to First Token (TTFT) vs Output Throughput

*TTFT* is dominated by prefill: processing the full prompt in one forward pass. For LLaMA 3 8B on H100 (BF16):

- 1k token prompt, batch size 1: ~15 ms TTFT
- 4k token prompt, batch size 1: ~55 ms TTFT
- 4k token prompt, batch size 32 (mixed prefill/decode): ~120–200 ms TTFT

Increasing batch size raises TTFT because the GPU must also process decode steps from other sequences in the same iteration.

*Inter-Token Latency (ITL)* is the per-step decode time: the time between consecutive output tokens for a single sequence. For LLaMA 3 8B on H100 BF16, decode is memory-bound:

$
"ITL" approx frac("Model weight bytes" + "KV cache bytes per step", "HBM bandwidth")
$

$
approx frac(16 "GB" + 0.5 "MB" times N_"seqs", 3.35 "TB/s") approx 4.8 "ms at" N_"seqs"=1
$

At batch size 64, the KV cache adds 32 MB, still dominated by the 16 GB weight read: ITL remains ~5–7 ms. The model is roofline-limited by HBM bandwidth.

=== Concrete Numbers (H100 80 GB, LLaMA 3 8B BF16)

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Metric*], [*Batch 1*], [*Batch 16*], [*Batch 64*], [*Batch 256*],
  [TTFT (4k prompt, ms)], [~55], [~90], [~170], [~400],
  [ITL (ms/token)], [~5], [~5], [~6], [~9],
  [Output throughput (tok/s)], [~200], [~3,000], [~10,000], [~17,000],
  [KV cache usage (GB)], [~0.1], [~1.6], [~6.4], [~25.6],
)

Key insight: _throughput and TTFT are in tension_. Filling the batch improves throughput but delays the first token for newly arrived requests waiting in the scheduler queue. SLA-aware schedulers manage this tension explicitly (see §SLA-Aware Scheduling).

=== Chunked Prefill

Chunked prefill breaks a large prompt into fixed-size chunks (e.g., 512 tokens) interleaved with decode steps. This caps the TTFT penalty from long prompts and prevents a single large prefill from stalling all decoding sequences.

```
Without chunked prefill (4k prompt + 64 decode seqs):
  Step 1: prefill 4096 tokens → 170 ms stall for decode seqs
  Step 2: decode 65 seqs       →   6 ms

With chunked prefill (chunk=512):
  Step 1: prefill 512 + decode 64 seqs  →  30 ms
  Step 2: prefill 512 + decode 64 seqs  →  30 ms
  ...
  Step 8: prefill 512 + decode 64 seqs  →  30 ms
  Total prefill: 8 × 30 = 240 ms (slower), but no stall on decode seqs
```

== Disaggregated Prefill/Decode

=== Motivation

Prefill and decode have fundamentally different compute profiles:

#table(
  columns: (auto, auto, auto),
  [*Property*], [*Prefill*], [*Decode*],
  [Arithmetic intensity], [High (matrix-matrix multiply)], [Low (matrix-vector multiply)],
  [Bottleneck], [Compute (Tensor Cores)], [Memory bandwidth (HBM)],
  [Batch sensitivity], [Throughput scales with batch], [Throughput limited by HBM BW],
  [Optimal GPU], [High FLOP/s (H100 SXM)], [High bandwidth (H100/A100 HBM3)],
  [Typical latency], [10–200 ms for 1k–8k tokens], [5–10 ms per token],
)

Mixing prefill and decode on the same GPU means neither workload can be tuned for its optimal regime. Long prefills stall decode; large decode batches slow prefill.

=== Architecture

```
┌────────────────────────────────────────────────────────┐
│                   Request Router                       │
└────────────┬───────────────────────┬───────────────────┘
             │ new prompt             │ active sequence
             ▼                        ▼
┌────────────────────┐   KV transfer  ┌───────────────────────┐
│  Prefill Pool      │ ─────────────▶ │  Decode Pool          │
│  (2–4× H100s)      │  (RDMA/NVLink) │  (4–8× H100s)         │
│  large GEMM        │                │  small steps, large BS │
│  ~100–500 ms/req   │                │  ~5–10 ms/step         │
└────────────────────┘                └───────────────────────┘
```

*KV transfer:* after prefill completes on a prefill GPU, the KV cache for that sequence must be transferred to a decode GPU. This adds latency (tens of milliseconds over PCIe, less than 5 ms over NVLink/RDMA for 8k tokens). The KV transfer cost is:

$
T_"KV" = frac("KV size", "NVLink BW") = frac(0.5 "MB/token" times L, 900 "GB/s") approx 0.6 mu s slash "token"
$

For an 8k prompt: $0.5 times 8192 = 4096 "MB" slash 900 "GB/s" approx 4.5 "ms"$ — acceptable overhead.

=== System Examples

*Mooncake* (Moonshot AI, 2024): disaggregated architecture deployed in production. Prefill and decode pools scale independently; the router directs requests based on prompt length. Reports 2–5× improvement (prefill throughput on long-context workloads, vs. colocated prefill+decode; Qin et al. 2024).

*DistServe* (Peking University, 2024): formal analysis of P/D disaggregation showing that optimal GPU ratio (prefill:decode) depends on the prompt-to-output length ratio of the workload. For chatbot workloads (short prompt, long output): 1:3 ratio is typical.

*vLLM v0.5+:* adds experimental disaggregated serving support via the `--preemption-mode swap` flag and separate prefill-worker configuration.

== Multi-LoRA Serving

LoRA (Low-Rank Adaptation) adds small adapter matrices $A in RR^(r times d)$, $B in RR^(d times r)$ to each linear layer, where $r lt.double d$ (typically $r = 8$ to $64$). Fine-tuning only $A$ and $B$ while freezing base weights produces task-specific models with adapter size of ~10–50 MB vs 16 GB for the full model.

=== Serving Multiple LoRA Adapters

In production, a single base model may serve thousands of customers, each with their own fine-tuned adapter. Naive approach: load a separate model copy per adapter. Cost: $N times 16$ GB for $N$ adapters — infeasible.

*S-LoRA* (2023, UC Berkeley): serves thousands of LoRA adapters on a single set of base model weights.

```
GPU Memory Layout (S-LoRA):
┌──────────────────────────────────────┐
│  Base Weights (frozen)  ~16 GB       │  ← always resident
├──────────────────────────────────────┤
│  Adapter Pool (hot)     ~2–4 GB      │  ← top-K active adapters
│    adapter_001 (A,B matrices, 32 MB) │
│    adapter_042 (A,B matrices, 28 MB) │
│    adapter_107 (A,B matrices, 35 MB) │
│    ...                               │
├──────────────────────────────────────┤
│  KV Cache               ~58 GB       │
└──────────────────────────────────────┘
  Inactive adapters: CPU RAM / disk
  Evicted on LRU basis when new adapter needed
```

*Unified Paging:* S-LoRA uses a unified memory pool for both KV cache pages and LoRA adapter weights, managed by a single allocator. This allows dynamic trade-off: fewer KV pages but more resident adapters, or vice versa.

*Batched LoRA inference:* within a single forward pass, different sequences use different adapters. The GEMM for a LoRA layer becomes a segmented batched GEMM:

```cpp
// Conceptual: batched LoRA delta computation
// For each sequence group using adapter k:
//   delta = x @ A_k @ B_k   (low-rank, cheap)
//   y += delta * lora_alpha / r
//
// Implemented as CUDA segmented GEMM:
void batched_lora_forward(
    const float* x,                    // [total_tokens, d_model]
    const int*   adapter_ids,          // [total_tokens]
    const LoRAAdapter* adapters,       // array of (A,B) per adapter
    float* y,                          // [total_tokens, d_model]
    int    total_tokens,
    int    d_model,
    int    r
) {
    // Group tokens by adapter_id, launch one GEMM per group
    // or use CUTLASS grouped GEMM for variable-size batches
}
```

*Throughput:* S-LoRA on a single A100 80 GB serves up to 2,000 distinct LoRA adapters concurrently, with $lt.eq$10% throughput drop vs. single-adapter serving at batch=64; degrades further at higher rank or with many concurrent adapters (Sheng et al., 2023). Adapter switching latency (CPU-to-GPU transfer for a cold 30 MB adapter over PCIe 4.0) is ~5–15 ms (theoretical 3.75 ms at 8 GB/s + driver/launch overhead).

== SLA-Aware Scheduling

Production LLM services define latency SLAs, typically:

- *P50 TTFT:* less than 200 ms (interactive chat)
- *P99 TTFT:* less than 1,000 ms
- *P50 ITL:* less than 50 ms (real-time streaming)
- *P99 end-to-end latency:* less than 30 s for long generations

The scheduler must meet these without sacrificing throughput. Key mechanisms:

=== Priority Queues

Requests are classified by SLA tier (e.g., premium, standard, batch). The scheduler maintains per-tier queues and preferentially schedules premium requests when KV cache pages become available. During overload, batch-tier requests are deferred or preempted.

```python
import heapq
from dataclasses import dataclass, field
from enum import IntEnum

class SLATier(IntEnum):
    PREMIUM  = 0   # lowest value = highest priority
    STANDARD = 1
    BATCH    = 2

@dataclass(order=True)
class Request:
    priority:  SLATier
    arrival:   float
    seq_id:    int = field(compare=False)
    prompt_ids: list = field(compare=False, default_factory=list)

class PriorityScheduler:
    def __init__(self):
        self._heap = []

    def push(self, req: Request):
        heapq.heappush(self._heap, req)

    def pop_batch(self, max_size: int) -> list[Request]:
        batch = []
        while self._heap and len(batch) < max_size:
            batch.append(heapq.heappop(self._heap))
        return batch
```

=== Preemption

When KV cache is exhausted and a high-priority request arrives, the scheduler can _preempt_ a low-priority running sequence: its KV cache pages are swapped to CPU RAM (or simply evicted and recomputed). Preemption adds latency to the victim but preserves SLA for the high-priority request.

*Recompute vs swap:* for short sequences, recomputing prefill on re-scheduling is cheaper than PCIe transfer. For sequences with greater than 2k tokens already generated, swapping is preferable.

=== Chunked Prefill for TTFT Guarantees

A large prompt (e.g., 32k tokens) would block the scheduler for 500+ ms if processed atomically. Chunked prefill caps the prefill cost per scheduler iteration to at most $C$ tokens (configurable, e.g., $C = 512$ or $C = 2048$). TTFT guarantee:

$
"Max TTFT" approx frac("Prompt length", C) times T_"step"
$

For a 32k prompt with $C = 2048$ chunks and $T_"step" = 20$ ms per iteration: max TTFT is approximately $16 times 20 = 320$ ms — predictable and bounded.

vLLM enables chunked prefill via:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --enable-chunked-prefill \
    --max-num-batched-tokens 2048 \
    --dtype bfloat16
```

== Observability

=== Prometheus Metrics

vLLM exposes a `/metrics` endpoint with Prometheus-compatible metrics. Key metrics to monitor:

```python
# vLLM Prometheus metric names (scrape at /metrics)
#
# Throughput:
#   vllm:num_requests_running          - requests in decode loop
#   vllm:num_requests_waiting          - requests in scheduler queue
#   vllm:num_requests_swapped          - requests with KV swapped to CPU
#   vllm:gpu_cache_usage_perc          - KV cache utilization [0, 1]
#   vllm:cpu_cache_usage_perc          - CPU swap utilization
#
# Latency (histograms):
#   vllm:time_to_first_token_seconds   - TTFT per request
#   vllm:time_per_output_token_seconds - ITL per token per request
#   vllm:e2e_request_latency_seconds   - full request latency
#
# Throughput counters:
#   vllm:prompt_tokens_total           - total prompt tokens processed
#   vllm:generation_tokens_total       - total output tokens generated
#   vllm:request_success_total         - completed requests
```

Example Prometheus scrape config:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: vllm
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: /metrics
    scrape_interval: 5s
```

=== KV Cache Utilization Monitoring

KV cache saturation is the primary serving bottleneck. When `gpu_cache_usage_perc` approaches 1.0, new requests queue and TTFT spikes. Alert thresholds:

```python
# Grafana alerting rule (PromQL)
# TTFT P99 exceeds 1 second
histogram_quantile(0.99,
    rate(vllm:time_to_first_token_seconds_bucket[1m])
) > 1.0

# KV cache above 90%: risk of preemption storms
vllm:gpu_cache_usage_perc > 0.90

# Queue depth rising: serving is falling behind request rate
rate(vllm:num_requests_waiting[1m]) > 10
```

=== Decode Step Tracing

For deep latency debugging, OpenTelemetry tracing can be integrated to capture per-request, per-step timing:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://jaeger:4317"))
)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm.serving")

# Instrument a decode step
def traced_decode_step(engine, seq_group):
    with tracer.start_as_current_span("decode_step") as span:
        span.set_attribute("batch_size", len(seq_group))
        span.set_attribute("kv_cache_usage",
                           engine.scheduler.block_manager.get_num_used_blocks())
        output = engine.step()
        span.set_attribute("tokens_generated", len(output))
    return output
```

=== Structured Logging

Log each request with machine-readable fields for offline analysis:

```python
import structlog, time

log = structlog.get_logger()

def log_request_complete(req_id, prompt_len, output_len, ttft_ms, total_ms):
    log.info(
        "request_complete",
        request_id=req_id,
        prompt_tokens=prompt_len,
        output_tokens=output_len,
        ttft_ms=round(ttft_ms, 2),
        total_latency_ms=round(total_ms, 2),
        itl_ms=round((total_ms - ttft_ms) / max(output_len - 1, 1), 2),
        throughput_tps=round(output_len / (total_ms / 1000), 1),
    )
```

== References

1. Kwon, W. et al. _Efficient Memory Management for Large Language Model Serving with PagedAttention_. SOSP 2023.
2. Yu, G. et al. _Orca: A Distributed Serving System for Transformer-Based Generative Models_. OSDI 2022. (Continuous batching.)
3. Zheng, L. et al. _SGLang: Efficient Execution of Structured Language Model Programs_. NeurIPS 2024.
4. Sheng, Y. et al. _S-LoRA: Serving Thousands of Concurrent LoRA Adapters_. MLSys 2024.
5. Zhong, Y. et al. _DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving_. OSDI 2024.
6. Peng, B. et al. _Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving_. 2024.
7. NVIDIA. _TensorRT-LLM: Optimizing Inference on NVIDIA GPUs_. NVIDIA Developer Blog, 2023.
8. HuggingFace. _Text Generation Inference_. github.com/huggingface/text-generation-inference.
9. Agrawal, A. et al. _Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills_. OSDI 2024. (Chunked prefill.)
10. Little, J. D. C. _A Proof for the Queuing Formula: $L = lambda W$_. Operations Research, 1961.

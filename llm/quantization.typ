= Quantization

Quantization reduces the numerical precision of a model's weights and activations, shrinking memory footprint and accelerating matrix operations. For large language models the payoff is asymmetric: a 70B-parameter model at FP16 requires 140 GB of GPU memory — impossible on a single 80 GB A100 — while INT4 brings it to roughly 35 GB. This chapter covers the math, the algorithms that achieve near-lossless compression, and the file formats that ship quantized models to end users.

*See also:* _gpu-architecture/compute-architecture.typ_ (Tensor Core precision modes, H100 FP8 support), _gpu-architecture/memory-hierarchy.typ_ (memory bandwidth arithmetic), _llm/transformer-architecture.typ_ (layer shapes used throughout the examples).

== Why Quantize: Memory Bandwidth Is the Bottleneck

=== Arithmetic Intensity at Decode

During autoregressive decode the model generates one token at a time. Each forward pass reads every weight once and performs one multiply-accumulate per weight element. For a matrix multiplication $Y = X W$ where $X in RR^(1 times d)$ and $W in RR^(d times d)$:

$ "Arithmetic intensity" = frac(2 d^2 " FLOPs", 2 d^2 " bytes") = 1 " FLOP/byte" quad "(FP16)" $

A single A100-80GB SXM delivers 2 TB/s memory bandwidth and 312 TFLOPS of FP16 tensor throughput. At arithmetic intensity 1, the roofline ceiling is 2 TFLOPS — just 0.6% of peak compute. _Decode is memory-bound by 160x._ Every halving of precision directly halves time-per-token.

Prefill (processing a long prompt) is batch-parallel and can reach arithmetic intensity 100--1000, where compute becomes the bottleneck. Quantization still helps prefill by reducing memory traffic for activations.

=== Model Size vs. GPU Memory

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  [*Model*], [*Params*], [*FP16 (GB)*], [*INT8 (GB)*], [*FP8 (GB)*], [*INT4 (GB)*],
  [LLaMA 3 8B],  [8.0B],  [16],  [8],  [8],  [4],
  [LLaMA 3 70B], [70.3B], [141], [70], [70], [35],
  [LLaMA 3 405B],[405B],  [810], [405],[405],[202],
  [Mistral 7B],  [7.2B],  [14],  [7],  [7],  [3.6],
  [Mixtral 8x7B],[46.7B], [93],  [47], [47], [23],
  [DeepSeek-V3], [671B],  [1342],[671],[671],[336],
)

_Bytes per parameter:_ FP16 = 2, BF16 = 2, FP8 = 1, INT8 = 1, INT4 = 0.5. Figures exclude KV cache and activation buffers. KV cache at FP16 adds roughly $2 times 2 times n_"layers" times n_"heads" times d_"head" times "context_len"$ bytes per batch element — up to 64 GB for a 128k-token LLaMA 3 70B context.

== Quantization Basics

=== Uniform Quantization

Uniform quantization maps a floating-point tensor $X in RR^n$ to integers in $[-2^(b-1), 2^(b-1)-1]$ (signed) using two parameters: a *scale* $s in RR$ and a *zero point* $z in ZZ$.

*Quantize:*

$ hat(x) = "clamp"(round(x / s) + z,\ -2^(b-1),\ 2^(b-1)-1) $

*Dequantize:*

$ tilde(x) = s dot (hat(x) - z) $

The *quantization error* for a single element is:

$ epsilon = x - tilde(x) = x - s dot (round(x/s) + z - z) $

which is bounded by $|epsilon| lt.eq s/2$.

=== Symmetric vs. Asymmetric

*Symmetric quantization* forces $z = 0$, mapping zero exactly and simplifying kernels:

$ s = frac(max(|X|))(2^(b-1) - 1) $

*Asymmetric quantization* uses the full integer range by centering on the actual data range $[x_"min", x_"max"]$:

$ s = frac(x_"max" - x_"min")(2^b - 1), quad z = round(-x_"min" / s) $

Asymmetric is more accurate when distributions are skewed (e.g., ReLU activations are non-negative). Symmetric is preferred for weights and for hardware that fuses zero-point subtraction.

=== Per-Tensor, Per-Channel, Per-Group

- *Per-tensor:* one $(s, z)$ pair for the entire weight matrix. Fastest, least accurate.
- *Per-channel (per-row/column):* one $(s, z)$ per output channel. Standard for INT8 weights.
- *Per-group:* one $(s, z)$ per contiguous group of $g$ elements (typically $g = 128$). Used in GPTQ, AWQ, GGUF. Balances accuracy and overhead.

=== C++ Implementation

```cpp
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>

struct QuantParams {
    float scale;
    int32_t zero_point;
};

// Compute per-tensor symmetric INT8 params
QuantParams compute_symmetric_int8(const float* data, int n) {
    float abs_max = 0.0f;
    for (int i = 0; i < n; ++i)
        abs_max = std::max(abs_max, std::abs(data[i]));
    QuantParams p;
    p.scale = abs_max / 127.0f;
    p.zero_point = 0;
    return p;
}

// Quantize FP32 -> INT8 (symmetric)
void quantize_int8(const float* src, int8_t* dst, int n, float scale) {
    for (int i = 0; i < n; ++i) {
        int32_t v = static_cast<int32_t>(std::round(src[i] / scale));
        dst[i] = static_cast<int8_t>(std::clamp(v, -127, 127));
    }
}

// Dequantize INT8 -> FP32
void dequantize_int8(const int8_t* src, float* dst, int n, float scale) {
    for (int i = 0; i < n; ++i)
        dst[i] = static_cast<float>(src[i]) * scale;
}

// Per-group INT4 packing: two INT4 values per byte
void pack_int4(const int8_t* src, uint8_t* dst, int n) {
    // n must be even; src values in [-8, 7]
    for (int i = 0; i < n / 2; ++i)
        dst[i] = static_cast<uint8_t>((src[2*i] & 0xF) | ((src[2*i+1] & 0xF) << 4));
}
```

== Post-Training Quantization vs. Quantization-Aware Training

*Post-training quantization (PTQ)* quantizes a fully trained FP32/BF16 model without additional gradient updates. A small _calibration set_ (typically 128--512 unlabeled samples) is used to compute activation statistics. PTQ is fast (minutes to hours) and requires no access to training data.

*Quantization-aware training (QAT)* inserts fake-quantize nodes into the forward pass during fine-tuning. Gradients flow through the rounding operation via a straight-through estimator (STE):

$ frac(partial L, partial x) approx frac(partial L, partial tilde(x)) dot bb(1)[x_"min" lt.eq x lt.eq x_"max"] $

QAT recovers 0.5--2 perplexity points vs. PTQ at the same bit-width but requires 10--100x more compute and access to a training corpus.

*Practical guidance:*

#table(
  columns: (auto, auto, auto, auto),
  [*Method*], [*Cost*], [*Accuracy*], [*Use case*],
  [PTQ (per-tensor)],  [minutes],   [--],       [INT8 serving, latency-insensitive],
  [PTQ (per-group)],   [hours],     [good],     [INT4/INT8 deployment (GPTQ, AWQ)],
  [QAT],               [days/weeks],[best],     [edge deployment, INT4 mobile],
)

== GPTQ: Layer-Wise Second-Order Quantization

GPTQ (Frantar et al., 2022) is a PTQ algorithm that quantizes each transformer weight matrix $W$ independently while minimizing the _layer reconstruction error_:

$ min_(hat(W)) || W X - hat(W) X ||_F^2 $

where $X$ is the calibration input to that layer (collected in one forward pass).

=== OBS Framework

GPTQ builds on the Optimal Brain Surgeon (OBS) framework (Hassibi & Stork, 1993). For a full-precision weight matrix, the increase in loss from quantizing a single weight $w_q$ is approximated using the inverse Hessian:

$ delta L = frac(1, 2) frac((w_q - "quant"(w_q))^2)([H_F^(-1)]_(q q)) $

where $H_F = 2 X X^top$ is the layer Hessian (factor of 2 from the squared Frobenius loss) and $[H_F^(-1)]_(q q)$ is the $q$-th diagonal entry of its inverse.

After quantizing weight $w_q$, the *OBS weight update* adjusts the remaining unquantized weights to compensate for the introduced error:

$ delta w = - frac(w_q - "quant"(w_q))([H_F^(-1)]_(q q)) dot [H_F^(-1)]_(q, :) $

This update is the key insight: quantizing in isolation produces irreversible error, but OBS lets each subsequent weight absorb the accumulated quantization error of all prior weights.

=== GPTQ Algorithm

GPTQ quantizes columns of $W$ left-to-right (equivalently, in order of the input features). For numerical stability it operates on the Cholesky factorization of $H_F^(-1)$ and applies a lazy block update to exploit GPU memory hierarchy.

```python
import torch

def gptq_quantize(W, H, bits=4, group_size=128, block_size=128):
    """
    W: weight matrix [d_out, d_in], FP16 or FP32
    H: Hessian H = 2 * X @ X.T  [d_in, d_in]
    Returns: quantized W_q (INT4 packed), scales, zero_points
    """
    d_out, d_in = W.shape
    W = W.float().clone()

    # Add damping for numerical stability
    damp = 0.01 * H.diag().mean()
    H += damp * torch.eye(d_in, device=H.device)

    # Cholesky decomposition of H^{-1}
    H_inv = torch.linalg.inv(H)
    try:
        L = torch.linalg.cholesky(H_inv)
    except torch.linalg.LinAlgError:
        H_inv += 1e-5 * torch.eye(d_in, device=H.device)
        L = torch.linalg.cholesky(H_inv)

    Q   = torch.zeros_like(W)         # quantized weights
    Err = torch.zeros_like(W)         # per-column quant error

    scales = []
    zero_points = []

    for col_start in range(0, d_in, block_size):
        col_end = min(col_start + block_size, d_in)

        W_block   = W[:, col_start:col_end].clone()
        Q_block   = torch.zeros_like(W_block)
        Err_block = torch.zeros_like(W_block)
        H_inv_block = H_inv[col_start:col_end, col_start:col_end]

        for j in range(col_end - col_start):
            col = col_start + j
            w   = W_block[:, j]  # [d_out]

            # Compute per-group scale for this column group
            g = col // group_size
            if col % group_size == 0:
                group_w = W[:, g*group_size : min((g+1)*group_size, d_in)]
                s = group_w.abs().max(dim=1).values / 7.0     # INT4 symmetric
                scales.append(s)
                zero_points.append(torch.zeros_like(s))

            s_col = scales[g]
            q = torch.clamp(torch.round(w / s_col.clamp(min=1e-8)), -8, 7)
            Q_block[:, j] = q

            # Quantization error for this column
            err = (w - q * s_col) / H_inv_block[j, j]

            # OBS update: propagate error to remaining columns in block
            W_block[:, j+1:] -= err.unsqueeze(1) * H_inv_block[j, j+1:].unsqueeze(0)
            Err_block[:, j]   = err

        Q[:, col_start:col_end] = Q_block

        # Propagate block error to remaining columns outside block
        W[:, col_end:] -= Err_block @ H_inv[col_start:col_end, col_end:]

    return Q, scales, zero_points
```

=== GPTQ in Practice

- Quantizing a 70B model takes ~4 GPU-hours on 1x A100 with `auto-gptq` or `llmcompressor`.
- Group size 128 gives 4.13 bits effective (128 FP16 scales per 128 INT4 weights).
- ExLlamaV2 and vLLM implement fused GPTQ INT4 GEMM kernels at 2--3x decode speedup over FP16.

== AWQ: Activation-Aware Weight Quantization

AWQ (Lin et al., 2023) observes that not all weight channels contribute equally to quantization error. A small fraction (roughly 1%) of _salient_ input channels carry disproportionate activation magnitude and, therefore, disproportionate reconstruction error when their associated weights are quantized.

=== Key Insight

For a linear layer $y = W x$, the per-channel output perturbation from quantizing column $j$ of $W$ is proportional to $|x_j|$. AWQ protects salient channels by applying a per-channel *scale* $s_j$ to the weights before quantization, then absorbing the inverse scale into the previous layer (e.g., the RMS norm scale or the previous linear's output scale) so the mathematical equivalence is preserved:

$ y = W x = (W dot "diag"(s)^(-1)) dot ("diag"(s) x) $

The scaled weight $hat(W)_j = W_j / s_j$ has reduced dynamic range for salient channels, making uniform quantization more accurate. The scale $s_j$ is chosen to minimize:

$ s_j^* = arg min_(s_j) || Q(W dot "diag"(s)^(-1)) dot "diag"(s) x - W x ||_2 $

where $Q$ denotes the rounding-and-clamping quantize operation. In practice $s_j$ is swept over a small grid and the best value is selected using the calibration set.

=== AWQ vs. GPTQ

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Method*], [*Algorithm*], [*Calibration*], [*Hardware*], [*INT4 PPL delta vs FP16*],
  [GPTQ],    [Second-order OBS],        [128 samples], [GPU (A100)],  [+0.3--0.6],
  [AWQ],     [Activation-aware scaling],[128 samples], [GPU or CPU],  [+0.2--0.5],
  [GPTQ+AWQ],[Scale then OBS],          [128 samples], [GPU (A100)],  [+0.1--0.3],
)

AWQ quantizes faster than GPTQ (no Hessian inversion) and produces slightly more accurate results on instruction-tuned models. The AWQ format is supported natively in `vLLM` and `llama.cpp`.

== SmoothQuant: Migrating Quantization Difficulty

SmoothQuant (Xiao et al., 2022) addresses a fundamental asymmetry in transformer quantization: *weights are easy to quantize, activations are hard*. Activation tensors exhibit per-channel outliers of magnitude 100x the median, causing catastrophic clipping if a single scale is applied across the channel dimension.

=== The Smoothing Transform

For a linear layer $Y = X W$, SmoothQuant introduces a per-channel smoothing factor $s in RR^(d_"in")$ and rewrites:

$ Y = X W = underbrace((X dot "diag"(s)^(-1)))_("easy to quantize") dot underbrace(("diag"(s) dot W))_("easy to quantize") $

The scale $s_j$ is set to balance quantization difficulty between $X$ and $W$:

$ s_j = frac(max(|X_{:,j}|)^alpha)(max(|W_{j,:}|)^(1-alpha)) $

The migration strength $alpha in [0, 1]$ controls the tradeoff. $alpha = 0.5$ works well empirically; $alpha = 1$ moves all difficulty to the weights (AWQ limit).

=== C++ INT8 SmoothQuant Kernel

```cpp
#include <cstdint>
#include <cmath>

// Apply smoothquant channel-wise scaling and quantize activations to INT8.
// src:    [batch, d_in]  FP32 input activations
// dst:    [batch, d_in]  INT8 output
// scales: [d_in]         smoothing scale s^{-1}  (pre-divided)
// q_scale: per-tensor quantization scale for the smoothed activations
void smoothquant_quantize(
    const float*  src,
    int8_t*       dst,
    const float*  inv_smooth,   // 1/s per channel
    float         q_scale,
    int           batch,
    int           d_in)
{
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < d_in; ++c) {
            float smoothed = src[b * d_in + c] * inv_smooth[c];
            int32_t q = static_cast<int32_t>(std::round(smoothed / q_scale));
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            dst[b * d_in + c] = static_cast<int8_t>(q);
        }
    }
}

// Pre-compute scaled weights: W_smooth[i, j] = W[i, j] * s[j]
// Executed once at model load time.
void apply_weight_smoothing(
    float*        W,
    const float*  smooth,       // s per input channel
    int           d_out,
    int           d_in)
{
    for (int i = 0; i < d_out; ++i)
        for (int j = 0; j < d_in; ++j)
            W[i * d_in + j] *= smooth[j];
}
```

SmoothQuant enables W8A8 (INT8 weights and INT8 activations) inference, reaching near-INT8-weight-only accuracy while allowing hardware INT8 tensor cores to accelerate both operands. It is integrated in `torch.ao.quantization` and FasterTransformer.

== FP8 Training and Inference

=== E4M3 vs. E5M2

The FP8 standard (OCP MX specification) defines two variants:

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Format*], [*Exponent bits*], [*Mantissa bits*], [*Max value*], [*Primary use*],
  [E4M3], [4], [3], [448],    [Weights and activations (forward pass)],
  [E5M2], [5], [2], [57344],  [Gradients (backward pass, wider range)],
)

E4M3 provides finer resolution near zero at the cost of smaller dynamic range — ideal for weight matrices and forward activations which cluster near zero after layer normalization. E5M2 handles the heavier-tailed gradient distributions.

=== H100 Transformer Engine

NVIDIA's Transformer Engine (TE) on H100 and later hardware executes FP8 GEMM in hardware at up to 4 PFLOPS (2x BF16 throughput). TE uses *dynamic per-tensor scaling*: a delayed scaling mechanism maintains a history of activation maximums and selects the scale factor to maximize FP8 dynamic range utilization without overflow.

Per-tensor scaling is the default; per-channel (per-row) scaling trades a small kernel overhead for higher accuracy on outlier-heavy activations.

*See also:* _gpu-architecture/compute-architecture.typ_ for H100 Tensor Core precision modes and the wgmma instruction.

=== PyTorch FP8 Training Example

```python
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

# FP8 recipe: E4M3 for forward, E5M2 for backward gradients
fp8_recipe = DelayedScaling(
    margin=0,
    interval=1,
    fp8_format=Format.HYBRID,     # E4M3 fwd, E5M2 bwd
    amax_history_len=16,
    amax_compute_algo="max",
)

model = te.TransformerLayer(
    hidden_size=4096,
    ffn_hidden_size=16384,
    num_attention_heads=32,
).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training step
with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    x = torch.randn(2, 512, 4096, dtype=torch.bfloat16, device="cuda")
    y = model(x)
    loss = y.mean()

loss.backward()
optimizer.step()
```

Using `torch.float8_e4m3fn` directly for manual FP8 operations:

```python
import torch

w_fp32 = torch.randn(4096, 4096, device="cuda")
scale  = w_fp32.abs().max() / 448.0          # E4M3 max = 448

# Cast to FP8 E4M3
w_fp8  = (w_fp32 / scale).to(torch.float8_e4m3fn)

# Cast back for inspection (actual GEMM uses hardware FP8 path)
w_dq   = w_fp8.to(torch.float32) * scale
print(f"Max abs error: {(w_fp32 - w_dq).abs().max():.6f}")
```

FP8 training achieves parity with BF16 on LLaMA-scale models with 1.5--2x throughput gain on H100.

== INT4/NF4 and QLoRA

=== NormalFloat4 (NF4)

Standard INT4 uniform quantization assumes a uniform distribution of weights. LLM weights, after pretraining, follow an approximately *normal distribution* $w sim cal(N)(0, sigma^2)$. NF4 (Dettmers et al., 2023) exploits this by mapping the 16 INT4 values to the quantile boundaries of a standard normal:

$ q_i = Phi^(-1)left(frac(i + 0.5)(16)right), quad i in {0, ..., 15} $

where $\Phi^(-1)$ is the normal quantile function (probit). Each weight is mapped to its nearest quantile value, producing a *minimum expected quantization error* for normally distributed weights. This is information-theoretically optimal for the assumed distribution.

The 16 NF4 quantile values (normalized to $[-1, 1]$) are stored as a lookup table; dequantization is a table lookup rather than a multiply, which is faster on CPU.

=== Double Quantization

NF4 requires a per-group FP32 scale (one per 64 or 128 weights). For a 70B model with group size 64, the scales consume $70 times 10^9 / 64 times 4 "bytes" approx 4.4 "GB"$. Double quantization (bitsandbytes) quantizes the scales themselves to FP8 with a second-level scale shared across 256 first-level groups, reducing the overhead to $approx 0.5 "GB"$.

=== QLoRA

QLoRA (Dettmers et al., 2023) fine-tunes LLMs by freezing the base model in NF4 and attaching low-rank adapters (LoRA) in BF16:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # double quantization
)

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 8,072,669,184 || trainable%: 0.52
```

The NF4 base weights are dequantized to BF16 only during the forward pass (for the GEMM). Gradients are never computed for base weights — only for the LoRA matrices $A$ and $B$. This reduces the fine-tuning memory footprint of LLaMA 3 8B from 60+ GB (full BF16 + Adam states) to roughly 10 GB on a single consumer GPU.

== KV Cache Quantization

The key-value cache stores past attention keys and values for all layers, growing linearly with sequence length. At context length 128k with batch size 8, LLaMA 3 70B accumulates roughly 64 GB of KV cache at FP16 — larger than the weights themselves.

=== FP8 KV Cache

Quantizing the KV cache to FP8 (E4M3) halves its memory at negligible accuracy cost:

```python
# vLLM FP8 KV cache configuration
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    kv_cache_dtype="fp8",               # E4M3 KV cache
    quantization="fp8",                 # FP8 weights via Transformer Engine
    tensor_parallel_size=4,
)
```

=== Per-Token vs. Per-Channel Scaling

KV activations exhibit different outlier structure than weight matrices:

- *Per-token scaling:* one scale per token position. Captures temporal outliers (e.g., the first token "attention sink" phenomenon). Low metadata overhead.
- *Per-channel scaling:* one scale per head-dimension channel. Captures the spatial channel outliers dominant in keys. Higher accuracy but requires storing $d_"head"$ scales per layer.

Recent work (Kang et al., 2024) shows per-channel KV scaling recovers 90% of the FP16 accuracy gap at INT4 vs. FP8, making INT4 KV cache viable for long-context deployment.

=== KV Cache Quantization Impact

#table(
  columns: (auto, auto, auto, auto),
  [*KV dtype*], [*Memory (128k ctx, 70B)*], [*Throughput*], [*PPL delta*],
  [FP16],  [~64 GB], [baseline],  [0.00],
  [FP8],   [~32 GB], [+5--10\%],   [+0.05],
  [INT8],  [~32 GB], [+8--12\%],   [+0.10],
  [INT4],  [~16 GB], [+15--25\%],  [+0.20--0.40],
)

== GGUF Format and llama.cpp

GGUF (GGML Universal Format) is the serialization format used by `llama.cpp` and compatible runtimes (Ollama, LM Studio, Jan). It supports a family of quantization levels denoted `Q{bits}_{variant}`.

=== Quantization Types

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Type*], [*Bits/weight*], [*Block size*], [*Description*], [*Typical use*],
  [Q4_0],     [4.5],  [32],  [4-bit, simple symmetric],       [legacy baseline],
  [Q4_K_S],   [4.58], [256], [4-bit K-quant, small scales],   [memory-limited],
  [Q4_K_M],   [4.85], [256], [4-bit K-quant, mixed 6-bit],    [best INT4 balance],
  [Q5_K_S],   [5.54], [256], [5-bit K-quant, small],          [accuracy step-up],
  [Q5_K_M],   [5.69], [256], [5-bit K-quant, mixed],          [near-FP16 accuracy],
  [Q6_K],     [6.56], [256], [6-bit K-quant, FP16 scales],    [high fidelity],
  [Q8_0],     [8.5],  [32],  [8-bit symmetric, FP32 scales],  [reference quality],
  [IQ4_NL],   [4.5],  [32],  [non-linear INT4 (NF4-like)],    [improved Q4_0],
)

_K-quants_ use a two-level (superblock) scaling scheme: 8 or 16 sub-blocks of 16 or 32 weights share a coarse FP16 _superblock scale_, and each sub-block has its own 6-bit or 4-bit _sub-scale_. This captures the dynamic range of large tensors with minimal scale overhead.

=== Block Quantization Data Layout (Q4_K_M)

For Q4_K_M, a superblock of 256 weights is stored as:

- 2 bytes: FP16 superblock scale $d$
- 2 bytes: FP16 superblock minimum $m$
- 12 bytes: 8 sub-scales and 8 sub-mins packed as 6-bit values (96 bits total)
- 128 bytes: 256 INT4 weights packed two-per-byte

Total: 144 bytes for 256 weights = 4.5 bits/weight (plus 12 bytes scale overhead = 4.85 bits effective).

=== C++ Dequantization Kernel (Q4_K_M)

```cpp
#include <cstdint>
#include <cstring>

struct BlockQ4KM {
    uint16_t d;             // FP16 superblock scale
    uint16_t dmin;          // FP16 superblock min
    uint8_t  scales[12];    // 6-bit sub-scales and sub-mins
    uint8_t  qs[128];       // INT4 weights, 2 per byte
};

static inline float fp16_to_fp32(uint16_t h) {
    // IEEE 754 half-to-float via union trick
    uint32_t v = ((uint32_t)(h & 0x8000) << 16)
               | ((uint32_t)((h & 0x7c00) + 0x1c000) << 13)
               | ((uint32_t)(h & 0x03ff) << 13);
    float f;
    std::memcpy(&f, &v, 4);
    return f;
}

// Dequantize one Q4_K_M superblock to FP32 output[256]
void dequant_q4km_block(const BlockQ4KM* blk, float* output) {
    const float d    = fp16_to_fp32(blk->d);
    const float dmin = fp16_to_fp32(blk->dmin);

    // Unpack 6-bit sub-scales and sub-mins (8 each from 12 bytes)
    uint8_t sc[8], m[8];
    const uint8_t* s = blk->scales;
    for (int i = 0; i < 4; ++i) {
        sc[i]   =  s[i] & 0x3F;
        sc[i+4] = (s[i] >> 4) | ((s[i+8] & 0x0F) << 4) & 0x3F;
        m[i]    =  s[i+4] & 0x3F;
        m[i+4]  = (s[i+4] >> 4) | ((s[i+8] >> 4) << 4) & 0x3F;
    }

    // Dequantize 8 sub-blocks of 32 weights each
    for (int sub = 0; sub < 8; ++sub) {
        const float scale = d    * static_cast<float>(sc[sub]);
        const float bias  = dmin * static_cast<float>(m[sub]);
        const int   base  = sub * 16;   // 16 bytes = 32 nibbles

        for (int j = 0; j < 32; ++j) {
            int byte_idx  = base + j / 2;
            int nibble    = (j % 2 == 0) ? (blk->qs[byte_idx] & 0xF)
                                         : (blk->qs[byte_idx] >> 4);
            output[sub * 32 + j] = scale * nibble - bias;
        }
    }
}
```

On modern CPUs, GGUF INT4 decode throughput for LLaMA 3 8B reaches 50--120 tokens/s (Apple M2, 4 threads), making it the dominant format for local CPU/Metal/Vulkan inference.

== Perplexity Impact

Perplexity (PPL) on WikiText-2 is the standard benchmark for quantization quality. Lower is better; FP16 is the reference. Values below are for LLaMA family models measured with a 2048-token context.

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  [*Model*], [*FP16*], [*INT8 (per-ch)*], [*Q4_K_M*], [*GPTQ INT4*], [*RTN INT4*],
  [LLaMA 2 7B],  [5.47], [5.51], [5.68], [5.61], [6.82],
  [LLaMA 2 13B], [4.88], [4.90], [5.02], [4.97], [5.43],
  [LLaMA 2 70B], [3.31], [3.32], [3.44], [3.39], [3.72],
  [LLaMA 3 8B],  [6.14], [6.16], [6.41], [6.30], [7.50],
  [LLaMA 3 70B], [2.85], [2.86], [2.95], [2.91], [3.15],
  [Mistral 7B],  [5.25], [5.27], [5.45], [5.38], [6.10],
)

_RTN_ = round-to-nearest, the naive baseline without calibration. The gap between RTN INT4 and GPTQ/Q4_K_M (which both use calibration data) demonstrates that second-order or activation-aware compensation is essential at 4-bit precision. INT8 per-channel is nearly lossless across all model sizes.

*Key takeaways:*
- INT8 per-channel: safe for all models, near-zero quality loss.
- Q4_K_M / GPTQ INT4 at 13B+: within 0.15 PPL of FP16, practical for production.
- INT4 at 7B/8B: noticeable degradation (~0.2--0.3 PPL); acceptable for many use cases.
- Naive INT4 (RTN): unacceptable; always use calibrated quantization at 4-bit.

== References

- Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* arXiv:2210.17323.
- Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., & Han, S. (2023). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* arXiv:2306.00978.
- Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S. (2022). *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* arXiv:2211.10438.
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314.
- Micikevicius, P., et al. (2022). *FP8 Formats for Deep Learning.* arXiv:2209.05433.
- Hassibi, B., & Stork, D. (1993). *Second Order Derivatives for Network Pruning: Optimal Brain Surgeon.* NeurIPS 1992.
- Gerganov, G. et al. (2023). *llama.cpp.* https://github.com/ggerganov/llama.cpp
- NVIDIA (2022). *Transformer Engine.* https://github.com/NVIDIA/TransformerEngine

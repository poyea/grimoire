= ML Workload Optimization on GPUs

Modern machine learning — particularly LLM training and inference — dominates GPU design decisions. Understanding how core primitives (GEMM, attention) map onto GPU hardware is essential for achieving near-peak throughput on Hopper/Blackwell-class devices.

*See also:* _compute-architecture.typ_ (Tensor Cores, TMA), _memory-hierarchy.typ_ (HBM, coalescing), _multi-gpu.typ_ (scaling).

== GEMM: the Core Primitive

For a transformer model, $>$ 80% of training FLOPs are GEMM ($C = alpha A B + beta C$). Every optimization of linear algebra libraries (cuBLAS, CUTLASS, rocBLAS) and of ML frameworks (PyTorch, JAX, TensorFlow) ultimately comes back to GEMM efficiency.

=== Progressive Optimization

Stage 1 — _naive_: one thread per output element.
```cpp
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) acc += A[row*K + k] * B[k*N + col];
        C[row*N + col] = acc;
    }
}
```
Problem: no data reuse. Each A, B element read $N$ / $M$ times from global memory. $approx$ 0.1-0.5 TFLOPS (memory-bound).

Stage 2 — _shared-memory tiling_: block cooperatively loads $T times T$ tile of A, B into shared memory.
```cpp
__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    constexpr int T = 32;
    __shared__ float As[T][T], Bs[T][T];
    int row = blockIdx.y*T + threadIdx.y;
    int col = blockIdx.x*T + threadIdx.x;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += T) {
        As[threadIdx.y][threadIdx.x] = A[row*K + k0 + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(k0+threadIdx.y)*N + col];
        __syncthreads();
        for (int k = 0; k < T; k++)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row*N + col] = acc;
}
```
Reuses each loaded tile $T$ times. $approx$ 2-5 TFLOPS.

Stage 3 — _warp tiling + register tiling_: each thread computes $R times R$ output tile held in registers; each warp cooperates on a larger tile. Doubles or triples arithmetic intensity. $approx$ 10-20 TFLOPS.

Stage 4 — _async copy + double buffering_: on Ampere+, `cp.async` copies from global→shared without going through registers, overlapping with compute in the previous tile.
```cpp
// Ampere cp.async primitive
#include <cuda/pipeline>
cuda::pipeline<cuda::thread_scope_thread> pipe;
cuda::memcpy_async(&As[...], &A[...], sizeof(float)*T, pipe);
pipe.producer_commit();
// ... compute on previous tile ...
cuda::pipeline_consumer_wait_prior<0>(pipe);
```
Overlaps HBM latency with compute. $approx$ 15-30 TFLOPS.

Stage 5 — _Tensor Cores (WMMA)_: 16$times$16$times$16 half-precision matmul in one instruction, 256 MAC/cycle.
```cpp
#include <mma.h>
using namespace nvcuda::wmma;

fragment<matrix_a, 16, 16, 16, half, row_major>   a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major>   b_frag;
fragment<accumulator, 16, 16, 16, float>          c_frag;

fill_fragment(c_frag, 0.0f);
for (int k = 0; k < K; k += 16) {
    load_matrix_sync(a_frag, A_tile + k, K);
    load_matrix_sync(b_frag, B_tile + k, K);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
}
store_matrix_sync(C_tile, c_frag, N, mem_row_major);
```
Peak: 312 TFLOPS FP16 (A100), 989 TFLOPS FP16 (H100 SXM). $approx$ 100-250 TFLOPS achievable on well-tuned kernel.

Stage 6 — _Hopper `wgmma` + TMA_: warp-group-level async matmul (64$times$N$times$16 per instruction, one warp-group = 4 warps). Source operands can come directly from shared memory without register staging. Combined with Tensor Memory Accelerator (TMA) for async global→shared copy, approaches 75-80% of FP16/FP8 peak ($tilde$ 700 TFLOPS FP8 on H100).

=== Roofline for GEMM

Arithmetic intensity for dense GEMM with $M = N = K$:
$ "AI" = "FLOPs" / "bytes" = (2 M^3) / ((3 M^2) dot "sizeof"(T)) = (2 M) / (3 dot "sizeof"(T)) $

For FP16 on H100 (peak 989 TFLOPS, BW 3.35 TB/s): ridge point at $989 "TFLOPS" / 3.35 "TB/s" approx 295$ FLOPs/byte. So GEMM is compute-bound once $M >> 295 dot 3 dot 2 / 2 = 885$, i.e. problem sizes beyond about 1000$times$1000.

Below that size, GEMM is memory-bound — which is exactly the case for small batch LLM inference (batch=1 decoding), which is why decode is BW-limited and prefill is compute-limited.

== Flash Attention (FA1, FA2, FA3)

Naive self-attention materializes the $N times N$ score matrix $S = Q K^T$ in HBM:
$ O(N^2) space, O(N^2 d) "time" $

For $N = $ 32K, FP16: 2 GB intermediate — doesn't fit in SRAM, and repeatedly round-trips to HBM.

*Algorithmic insight — online softmax* (Milakov & Gimelshein 2018): compute softmax block-by-block, maintaining running maximum and sum of exponentials. Given block $x_b$:
$ m_"new" = max(m_"old", max_i(x_(b,i))) $
$ l_"new" = e^(m_"old" - m_"new") l_"old" + sum_i e^(x_(b,i) - m_"new") $
$ O_"new" = e^(m_"old" - m_"new") O_"old" + (e^(x_b - m_"new") V_b) $

The final output is $O / l$. This allows tiled computation of attention without materializing the full score matrix.

*Flash Attention 1 (Dao et al. 2022):* Q, K, V tiled into SRAM-sized blocks ($B_r$ rows of Q, $B_c$ cols of K/V). Outer loop over K, V tiles; inner loop over Q tiles. Never materializes full $N times N$ in HBM.

HBM accesses reduced from $O(N d + N^2)$ to $O((N d)^2 / M)$ where $M$ = SRAM size. For $N = 4096$, $d = 128$, $M = 100$ KB: $approx$ 8x fewer HBM round trips. Measured $tilde$ 3x speedup over `nn.Softmax(QK^T)V` on A100.

*Flash Attention 2 (Dao 2023):*
- Swapped outer and inner loops: Q in outer, K/V in inner — higher warp occupancy
- Reduced non-matmul FLOPs (fewer rescales, masked divides)
- Better parallelism across sequence dim and heads
- $tilde$ 2x FA1 throughput; $tilde$ 50-70% of A100 peak

*Flash Attention 3 (Shah et al. 2024):*
- Hopper-specific: uses `wgmma` async matmul _overlapped_ with softmax via async `mbarrier` semantics
- Warp specialization: some warps as TMA producers (load K, V), others as consumers (compute matmul)
- FP8 support with blockwise scaling (Transformer Engine semantics)
- Achieves 75% of H100 FP16 peak ($tilde$ 750 TFLOPS) and 1.2 PFLOPS FP8
- $tilde$ 1.5-2x FA2 on H100

```cpp
// Simplified FA2 forward loop (per-block view)
for (int i = 0; i < N; i += Br) {               // Q block
    // Load Q_i into SRAM
    float m_i = -inf, l_i = 0;
    for (int j = 0; j < N; j += Bc) {           // K,V blocks
        // Load K_j, V_j
        S = Q_i @ K_j.T / sqrt(d);              // [Br x Bc] in SRAM
        m_new = max(m_i, max(S));
        P = exp(S - m_new);
        l_new = exp(m_i - m_new) * l_i + sum(P);
        O_i = exp(m_i - m_new) * O_i + P @ V_j;
        m_i = m_new; l_i = l_new;
    }
    O_i /= l_i;
    // Store O_i
}
```

== Quantization for Inference

For LLMs at inference, decoding is memory-bound: each token requires loading all weights from HBM. Quantization shrinks weights, proportionally raising effective arithmetic intensity and speeding inference.

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Format*], [*Bits*], [*Representative method*], [*Accuracy (LLaMA-7B)*], [*Inference speedup vs FP16*],
  [FP16], [16], [(baseline)], [100%], [1.0$times$],
  [BF16], [16], [(training-friendly)], [100%], [1.0$times$],
  [FP8 E4M3], [8], [Micikevicius et al. 2022], [99.8%], [1.9$times$],
  [INT8], [8], [SmoothQuant (Xiao 2023)], [99.5%], [1.7$times$],
  [INT4 / NF4], [4], [GPTQ (Frantar 2023), AWQ (Lin 2023)], [97-98%], [2.5$times$],
  [FP4 MX], [4], [Blackwell hardware (OCP 2024)], [97%], [3.0$times$],
  [INT3 / INT2], [2-3], [QuIP\# (Chee 2024)], [94-96%], [3.5$times$],
)

=== Post-Training Quantization (PTQ) Flow

```cpp
// 1. Calibration — collect activation statistics
for (auto& sample : calib_set /*100-1024 samples*/) {
    forward(sample);
    for (auto& layer : model) {
        layer.stats.update(activation_max_per_channel,
                           activation_percentile_99_9);
    }
}

// 2. Compute quantization scale per weight tensor
// Symmetric per-channel: scale = max(|W|) / 127 (for INT8)
for (auto& layer : model) {
    layer.weight_scale  = per_channel_max(abs(W)) / 127;
    layer.W_int8        = round(W / layer.weight_scale).clip(-127, 127);
}

// 3. Activation quantization
//    Per-tensor (simple) or per-token (more accurate)
//    SmoothQuant: migrate quantization difficulty from activation -> weight
//    via scale transformation to make both easier to quantize

// 4. At runtime, INT GEMM on Tensor Cores
cublasGemmEx(handle, ..., CUDA_R_8I, ...,   // A is INT8
                             CUDA_R_8I, ..., // B is INT8
                             CUDA_R_32I, ..., // accumulate in INT32
                             CUBLAS_COMPUTE_32I,
                             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
// Dequantize output back to FP16
```

*Weight-only quantization (GPTQ, AWQ):* quantize only weights (INT4), keep activations FP16. On load, dequantize on-the-fly to FP16 registers before FP16 matmul. Benefit: only memory BW speedup — no loss from activation quant.

*Quantization-Aware Training (QAT):* fake-quantize in forward pass with straight-through estimator for backward. Best accuracy, most expensive (requires retraining).

=== KV Cache: the Other Memory Hog

During autoregressive decoding, every generated token appends a $d$-dim K, V vector per layer, per head. For LLaMA-70B at 8K context, 1 request: $2 times 80 "layers" times 8 "heads" times 8192 times 128 times 2 "bytes" approx 3.2$ GB. At batch 64: 200 GB — doesn't fit on single GPU.

Optimizations:
- *MQA (Multi-Query Attention, Shazeer 2019):* single K, V shared across all query heads — $1/H$ memory
- *GQA (Grouped-Query, Ainslie 2023):* $G$ groups sharing K, V — $G/H$ memory (LLaMA-2 70B uses $G = 8$)
- *PagedAttention (vLLM — Kwon et al. 2023):* manage KV cache in fixed 16-token blocks with a page table; prevents fragmentation, enables sharing (prefix caching). 2-4$times$ throughput gain for concurrent requests.
- *Quantized KV cache:* INT8 or FP8 KV — cuts KV memory 2$times$ with small accuracy cost.
- *Speculative decoding (Leviathan 2023):* cheap draft model proposes $k$ tokens; target model verifies in one parallel forward — 2-3$times$ wall-clock speedup when drafts mostly accepted.

== Operator Fusion

Memory-bound ops get dominated by HBM round trips. Fusing adjacent ops eliminates intermediate reads/writes.

Examples of fused operators in production kernels:
- *Attention block:* fused QKV projection (one GEMM for $[Q, K, V]$) + Flash Attention + output projection
- *FFN block:* fused `act(x W_1 + b_1) W_2 + b_2`
- *Layer norm + residual:* one kernel
- *RMSNorm / GeLU / SiLU / SwiGLU:* elementwise fused into producer kernel

*Triton (OpenAI, 2021+):* DSL for writing block-level kernels in Python. Compiles to PTX. A naive GeLU in Triton:
```python
@triton.jit
def gelu_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = 0.5 * x * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    tl.store(y_ptr + offs, y, mask=mask)
```

*Compilers and fusion engines:*
- `torch.compile` / TorchInductor — graph-level fusion, Triton codegen
- cuDNN graph API — runtime fusion for attention, convolution chains
- TensorRT — ahead-of-time compilation; best for deployed inference
- XLA (JAX, TensorFlow) — HLO-level fusion, MLIR lowering
- TVM / MLIR — research/extensibility

== Profiling Checklist

*Nsight Compute key metrics:*
- `sm__throughput.avg.pct_of_peak_sustained_active` — overall SM utilization
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` — Tensor Core busy %
- `dram__throughput.avg.pct_of_peak_sustained_active` — HBM utilization
- `smsp__warps_active.avg.pct_of_peak_sustained_active` — effective occupancy
- `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` — memory-latency stalls
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum.per_request_percent` — coalescing efficiency

Compute and memory should not _both_ be at peak simultaneously — if both idle, you're stalled (wait for barrier, pipeline bubble, or warp divergence).

*Framework-level:*
- PyTorch Profiler (kineto backend, Chrome trace): end-to-end timeline
- `torch.cuda.memory._snapshot()` + memory_viz tool: allocation history
- `nvidia-smi dmon` for live SM/DRAM utilization during training

*LLM training MFU (Model FLOPs Utilization):*
$ "MFU" = "achieved FLOPs" / "peak FLOPs" = (6 N P) / (T dot "peak") $
where $N$ = tokens/step, $P$ = params, $T$ = step time, factor 6 = 2 (fwd) + 4 (bwd) FLOPs per param per token.
Well-tuned training hits 45-55% FP16 MFU; 60%+ with FP8 on Hopper. Below 30% usually indicates a fixable bottleneck.

== Performance Targets (H100 SXM)

#table(
  columns: (auto, auto, auto),
  [*Workload*], [*Achievable*], [*Notes*],
  [Dense GEMM FP16 (large $M = N = K >= 4096$)], [500-800 TFLOPS (peak 989)], [60-80% peak with CUTLASS/cuBLAS],
  [Dense GEMM FP8], [900-1400 TFLOPS (peak 1979)], [CUTLASS 3.x with wgmma],
  [Flash Attention 3 FP16], [500+ TFLOPS], [Context length $N >= 2048$],
  [Flash Attention 3 FP8], [800-1000 TFLOPS], [H100 only],
  [LLM training MFU (DP+TP+PP)], [45-55% FP16 / 60%+ FP8], [Varies with model arch],
  [LLM inference decode (batch=1)], [Memory-bound], [$tilde$ 100-200 tok/s for 70B FP8],
  [LLM inference prefill (long prompt)], [Near compute-bound], [Saturates Tensor Cores],
)

== References

Dao, T., Fu, D.Y., Ermon, S., Rudra, A., Ré, C. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS.

Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." arXiv:2307.08691.

Shah, J., Bikshandi, G., Zhang, Y., Thakkar, V., Ramani, P., Dao, T. (2024). "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." arXiv:2407.08608.

Milakov, M. & Gimelshein, N. (2018). "Online normalizer calculation for softmax." arXiv:1805.02867.

Micikevicius, P. et al. (2022). "FP8 Formats for Deep Learning." arXiv:2209.05433.

Dettmers, T. et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." NeurIPS.

Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. (2023). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR.

Lin, J. et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." arXiv:2306.00978.

Xiao, G. et al. (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." ICML.

Kwon, W. et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP (vLLM).

Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." arXiv:1911.02150 (MQA).

Ainslie, J. et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." arXiv:2305.13245.

Leviathan, Y., Kalman, M., Matias, Y. (2023). "Fast Inference from Transformers via Speculative Decoding." ICML.

NVIDIA (2022, 2024). "CUTLASS: CUDA Templates for Linear Algebra Subroutines." https://github.com/NVIDIA/cutlass


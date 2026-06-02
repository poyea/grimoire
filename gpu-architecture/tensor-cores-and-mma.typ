= Tensor Cores and Matrix Multiply-Accumulate

Tensor Cores are specialized matrix-multiply pipelines that deliver an order of magnitude more throughput than the FP32 CUDA cores beside them. Every generation since Volta has widened the tile, added new datatypes (BF16, TF32, FP8, INT8/INT4), and moved more of the data-movement burden into hardware (cp.async, TMA, wgmma).

*See also:* _Compute Architecture_ (SM block diagrams), _ML Workloads_ (GEMM pipelines), _Memory Hierarchy_ (shared memory, swizzling).

== Generational Overview

#table(
  columns: 6,
  [*Arch*], [*SM*], [*Tile (M\@N\@K)*], [*Dtypes*], [*Peak (per SM/cycle)*], [*Year*],
  [Volta],   [sm_70], [4x4x4],    [FP16$->$FP32], [64 FMA / cycle / TC, 8 TC/SM], [2017],
  [Turing],  [sm_75], [4x4x4],    [+ INT8, INT4], [+ inference dtypes], [2018],
  [Ampere],  [sm_80], [16x8x16],  [+ BF16, TF32, FP64, 2:4 sparse], [256 FMA / cycle / TC, 4 TC/SM], [2020],
  [Hopper],  [sm_90], [64x*xK (wgmma)], [+ FP8 (E4M3/E5M2)], [warp-group MMA, async], [2022],
  [Blackwell], [sm_100], [larger wgmma, microscaled FP4/FP6], [+ MXFP8/MXFP6/MXFP4], [further 2x over Hopper], [2024],
)

Approximate dense FP16 throughput (TF32-equivalent omitted):

```
GPU         FP16 Tensor TFLOPS  FP8 TFLOPS  HBM GB/s
─────────────────────────────────────────────────────
V100         125              —           900
A100         312              —          2039
H100 SXM    989             1979         3350
B200       ~2250            ~4500       ~8000
```

== Volta and Turing — WMMA

Volta introduced *Warp Matrix Multiply-Accumulate* (WMMA), exposed in CUDA C++ as `nvcuda::wmma`. The fragment API operates at warp granularity: a 32-thread warp cooperatively holds a 16$times$16 matrix tile.

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void mma_kernel(const half* A, const half* B, float* C, int K) {
    fragment<matrix_a,    16, 16, 16, half, row_major> a;
    fragment<matrix_b,    16, 16, 16, half, col_major> b;
    fragment<accumulator, 16, 16, 16, float>           c;
    fill_fragment(c, 0.f);
    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(a, A + k, K);
        load_matrix_sync(b, B + k, K);
        mma_sync(c, a, b, c);
    }
    store_matrix_sync(C, c, 16, mem_row_major);
}
```

The fragment layout is opaque — register positions per lane are unspecified — so element-wise operations on a fragment must use `c.x[i]`. Mixing fragments requires `mem_*` traffic through shared memory.

== Ampere — PTX mma.sync, BF16, TF32, Sparsity

Ampere added a richer `mma.m16n8k16` PTX family, plus *Brain Float 16* (BF16: 8 exp + 7 mant, same range as FP32), *TF32* (FP32-input rounded to 10-bit mantissa for use inside Tensor Cores), and *structured 2:4 sparsity* (every group of 4 weights has at most 2 non-zeros, doubling effective TC throughput).

Inline PTX for a 16$times$8$times$16 BF16 MMA:
```cpp
asm volatile(
  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
  "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
  : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
  : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
    "r"(b0), "r"(b1));
```

=== cp.async — Async Copy from Global to Shared

Pre-Ampere, loading a tile cost a register round-trip: `global $->$ register $->$ shared`. Ampere added `cp.async`, copying directly `global $->$ shared` and overlapping with compute on the prior tile.

```cpp
#include <cuda/pipeline>
#include <cooperative_groups/memcpy_async.h>
namespace cg = cooperative_groups;

__shared__ alignas(16) half Asmem[2][BM*BK];

auto block = cg::this_thread_block();
__shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> state;
auto pipe = cuda::make_pipeline(block, &state);

for (int k = 0; k < K; k += BK) {
    pipe.producer_acquire();
    cuda::memcpy_async(block, Asmem[k&1], A_gmem + k, sizeof(half)*BM*BK, pipe);
    pipe.producer_commit();

    pipe.consumer_wait();
    // mma_sync on Asmem[(k-1)&1] ...
    pipe.consumer_release();
}
```

Double-buffering with `cp.async` is the foundation of all modern GEMM kernels.

== Hopper — wgmma and the Tensor Memory Accelerator

Hopper redesigned the MMA path around two ideas:

+ A *warp group* (4 warps = 128 threads) is the new MMA unit. `wgmma.mma_async` issues a large tile (e.g. `m64n256k16`) that completes asynchronously over many cycles.
+ The *Tensor Memory Accelerator* (TMA) is a dedicated DMA engine that moves multidimensional tiles between global and shared memory, with automatic boundary handling and swizzling — driven by a single descriptor instead of per-thread address arithmetic.

=== TMA

```cpp
#include <cuda/barrier>
__shared__ alignas(16) half smem[BM*BK];
__shared__ cuda::barrier<cuda::thread_scope_block> bar;
if (threadIdx.x == 0) init(&bar, blockDim.x);
__syncthreads();

if (threadIdx.x == 0) {
    // tma_desc constructed on the host via cuTensorMapEncodeTiled
    cde::cp_async_bulk_tensor_2d_global_to_shared(
        smem, &tma_desc, blockIdx.x*BM, blockIdx.y*BK, bar);
    bar.arrive_and_expect_tx(sizeof(half)*BM*BK);
} else {
    bar.arrive();
}
bar.wait(...);
```

A single thread issues the transfer; the rest of the warp does anything else (or wait at the barrier). TMA frees ~30 registers per warp previously spent on address math.

=== wgmma

```cpp
// Pseudocode for wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
wgmma_fence();
for (int k = 0; k < K; k += 16) {
    wgmma_mma_async<64, 128, 16, float, half, half>(
        D, A_desc + k, B_desc + k);
}
wgmma_commit_group();
wgmma_wait_group<0>();
```

`A_desc` and `B_desc` are *matrix descriptors* — 64-bit handles to shared-memory tiles, including swizzling mode. The MMA reads operands directly from shared memory, eliminating the per-warp register load.

=== FP8 on Hopper

Two 8-bit float formats:

#table(
  columns: 4,
  [*Format*], [*Sign/Exp/Mant*], [*Max*], [*Use*],
  [E4M3], [1/4/3], [448],         [forward activations, weights],
  [E5M2], [1/5/2], [57344],       [gradients (wider dynamic range)],
)

FP8 tensor cores deliver 2$times$ FP16 TFLOPS but need *per-tensor (or per-block) scaling*: amax tracking, dynamic rescaling, and master-precision (FP32) accumulation. The Transformer Engine library automates this for Linear / Attention layers.

```python
import transformer_engine.pytorch as te
with te.fp8_autocast(enabled=True,
                     fp8_recipe=te.recipe.DelayedScaling()):
    y = te.Linear(d, d)(x)
```

== Blackwell — Microscaled Formats and FP4

Blackwell adds *MXFP8/MXFP6/MXFP4*: every block of 32 elements shares an 8-bit (E8M0) scale, giving fine-grained quantization at hardware throughput. FP4 (E2M1) doubles FP8 TFLOPS again. The 5th-gen Tensor Core handles the descale automatically.

```
MXFP4 block:  [E2M1 x 32 elements][E8M0 shared scale]   -> 132 bits / 32 vals
```

Effective throughput stacks: FP4 dense $approx$ 4$times$ FP16, with 2:4 sparsity making it 8$times$.

== Putting It Together — Pipeline of a Modern GEMM

```
Persistent kernel
  for each output tile:
    TMA load A, B tiles into smem ring buffer (producer warp)
    consumer warp-group:
      wgmma.async on smem tile k
      wgmma.async on smem tile k+1   (overlap)
      ...
      wgmma_wait_group<N>            (let compute hide TMA)
    TMA store D tile to global
```

This is the structure used by CUTLASS 3.x, FlashAttention-3, and Triton's Hopper backend. Achieved efficiencies are 70–85% of peak FP16/FP8 TFLOPS on H100.

== Datatype Cheat Sheet

#table(
  columns: 5,
  [*Type*], [*Bits*], [*Range*], [*Use*], [*First arch*],
  [FP64], [64], [$plus.minus 10^308$], [HPC], [Volta],
  [TF32], [19 (10-mant)], [$plus.minus 3.4 dot 10^38$], [FP32-drop-in training], [Ampere],
  [FP16], [16], [$plus.minus 65504$], [training, inference], [Volta],
  [BF16], [16], [$plus.minus 3.4 dot 10^38$], [training (preferred)], [Ampere],
  [FP8 E4M3], [8], [$plus.minus 448$], [forward], [Hopper],
  [FP8 E5M2], [8], [$plus.minus 57344$], [backward], [Hopper],
  [MXFP6/MXFP4], [6/4 + shared scale], [block-scaled], [inference], [Blackwell],
  [INT8 / INT4], [8 / 4], [integer], [quantized inference], [Turing],
)

== Further Reading

NVIDIA (2017). _NVIDIA Tesla V100 GPU Architecture_ whitepaper.

NVIDIA (2020). _NVIDIA A100 Tensor Core GPU Architecture_ whitepaper (TF32, BF16, 2:4 sparsity).

NVIDIA (2022). _NVIDIA H100 Tensor Core GPU Architecture_ whitepaper (FP8, TMA, wgmma).

NVIDIA (2024). _NVIDIA Blackwell Architecture Technical Brief_ (MXFP, FP4).

NVIDIA (2024). _PTX ISA Manual_, Chapter on `mma`, `wgmma`, `cp.async`, `cp.async.bulk.tensor`.

Micikevicius, P. et al. (2022). "FP8 Formats for Deep Learning." arXiv:2209.05433.

Thakkar, V. et al. (2023). _CUTLASS 3.0: A Hopper-Native Library_. NVIDIA GTC.

Shah, J. et al. (2024). "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision." arXiv:2407.08608.

#import "../template.typ": xref

= Triton and GPU Domain-Specific Languages

Writing peak-performance CUDA by hand is a job for a small number of experts; writing it in Python with single-digit-percent loss in efficiency is now a job for many more. The new generation of GPU DSLs — Triton, Pallas, ThunderKittens, and CUTLASS in its template form — trade some of CUDA's flexibility for a *block-level* programming model that lets the compiler handle the painful parts: shared-memory layout, async pipelining, and tensor-core scheduling.

*See also:* #xref("gpu-architecture", "cuda-programming-model", label: "CUDA Programming Model") (raw CUDA), #xref("gpu-architecture", "tensor-cores-and-mma", label: "Tensor Cores and Matrix Multiply-Accumulate") (wmma/wgmma), #xref("gpu-architecture", "ml-workloads", label: "ML Workload Optimization on GPUs") (GEMM, attention).

== Why a DSL?

A CUDA GEMM that hits 80% of peak on H100 is ~1000 lines of templated C++ with explicit `cp.async`, swizzled shared-memory layouts, manual double-buffering, and warp-specialized producer/consumer roles. A Triton GEMM that hits ~70% is ~30 lines of Python. The remaining gap closes every release as compilers learn the H100 idioms (TMA, wgmma, warp specialization).

The unifying idea: *program at the block level*, not the thread level. The user writes one program per block describing tiles of data; the compiler emits the per-thread schedule.

== Triton

Triton (Tillet 2019, now an OpenAI/PyTorch project) compiles a Python-embedded IR through MLIR to PTX. It is the default kernel language for `torch.compile`'s Inductor backend.

=== Vector Add — Hello, Triton

```python
import triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK']),)
    add_kernel[grid](x, y, out, x.numel(), BLOCK=1024)
    return out
```

Notice: no thread IDs, no shared memory, no warps. Triton automatically vectorizes loads, picks register allocation, and schedules instructions.

=== GEMM in Triton

```python
@triton.autotune(
    configs=[
        triton.Config({'BM':128,'BN':256,'BK':32,'GROUP':8},
                      num_stages=3, num_warps=8),
        triton.Config({'BM':64, 'BN':128,'BK':32,'GROUP':8},
                      num_stages=4, num_warps=4),
    ], key=['M', 'N', 'K'])
@triton.jit
def matmul(A, B, C, M, N, K,
           sa0, sa1, sb0, sb1, sc0, sc1,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
           GROUP: tl.constexpr):
    pid = tl.program_id(0)
    # super-grouped swizzle for L2 reuse
    num_pm = tl.cdiv(M, BM); num_pn = tl.cdiv(N, BN)
    npg = GROUP * num_pn
    g = pid // npg
    pm = g * GROUP + (pid % npg) % GROUP
    pn = (pid % npg) // GROUP

    offs_m = pm*BM + tl.arange(0, BM)
    offs_n = pn*BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = A + offs_m[:, None]*sa0 + offs_k[None, :]*sa1
    b_ptrs = B + offs_k[:, None]*sb0 + offs_n[None, :]*sb1

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=offs_k[None,:] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:,None] < K - k, other=0.0)
        acc += tl.dot(a, b)                # -> mma.sync / wgmma
        a_ptrs += BK*sa1
        b_ptrs += BK*sb0

    c_ptrs = C + offs_m[:, None]*sc0 + offs_n[None, :]*sc1
    tl.store(c_ptrs, acc.to(tl.float16),
             mask=(offs_m[:,None] < M) & (offs_n[None,:] < N))
```

`tl.dot` lowers to mma.sync on Ampere or wgmma on Hopper, automatically. `num_stages` requests a software-pipelined loop with that many in-flight `cp.async` iterations.

=== Autotuning

`@triton.autotune` runs every config once on the actual input shape, caches the winner, and dispatches to it on subsequent calls. The `key` argument controls cache invalidation: a new $(M, N, K)$ retunes.

Larger tile $arrow.r$ better arithmetic intensity but more shared memory and registers; the optimal point depends on the shape and on the hardware. Autotuning typically explores 8–32 configs and finds a ~2$times$ improvement over fixed defaults.

=== Triton on Hopper

Triton 3.0+ targets sm_90 with TMA descriptors and warp specialization. The user-level Python code is unchanged; the compiler emits a producer warp that issues TMA loads into a shared-memory ring buffer and a consumer warp-group that runs wgmma.

== Pallas

Pallas is JAX's analog: a Triton-flavored kernel DSL that lowers to Triton (GPU) or Mosaic (TPU). The same user kernel runs on both backends.

```python
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl

def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

@jax.jit
def add(x, y):
    return pl.pallas_call(
        add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(x.shape[0] // 128,),
        in_specs=[pl.BlockSpec((128,), lambda i: (i,)),
                  pl.BlockSpec((128,), lambda i: (i,))],
        out_specs=pl.BlockSpec((128,), lambda i: (i,)),
    )(x, y)
```

`BlockSpec` describes how the global tensor is tiled into the block view (`x_ref`) — separating the global indexing from the block-local compute. This is essential for the TPU backend, where DMA descriptors play the same role.

== ThunderKittens

ThunderKittens (Hazy Research, 2024) is a small CUDA C++ header library aimed at researchers who want CUDA performance without writing CUTLASS. It exposes *tiles* and *register fragments* as first-class C++ objects with the right operators, plus async helpers around TMA and wgmma.

```cpp
#include "kittens.cuh"
using namespace kittens;

__global__ void attn(int N, bf16* Q, bf16* K, bf16* V, bf16* O) {
    rt_bf<16, 64>           q_reg, k_reg;
    rt_fl<16, 16>           att;
    rt_fl<16, 64>           o_reg;

    load(q_reg, Q + blockIdx.x*16*64, 64);
    zero(o_reg);
    for (int j = 0; j < N; j += 16) {
        load(k_reg, K + j*64, 64);
        zero(att);
        mma_ABt(att, q_reg, k_reg, att);          // wgmma
        softmax(att);
        rt_bf<16,16> att_bf; copy(att_bf, att);
        rt_bf<16,64> v_reg;  load(v_reg, V + j*64, 64);
        mma_AB(o_reg, att_bf, v_reg, o_reg);
    }
    store(O + blockIdx.x*16*64, o_reg, 64);
}
```

~100 lines for FlashAttention-2-level performance, instead of ~2000 for the original.

== CUTLASS as a DSL

CUTLASS is NVIDIA's "what an optimal CUDA GEMM looks like" reference, but with the 3.x rewrite around CuTe it became a *layout algebra* DSL. CuTe describes data layouts as composable shape/stride pairs; CUTLASS schedules a *collective* + *epilogue* pair over those layouts to produce a kernel.

```cpp
using Gemm = cutlass::gemm::device::GemmUniversal<
    cute::half_t, cutlass::layout::RowMajor,
    cute::half_t, cutlass::layout::ColumnMajor,
    cute::half_t, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm90,
    cute::Shape<cute::_128, cute::_256, cute::_64>,     // tile
    cute::Shape<cute::_2,   cute::_1,   cute::_1>,      // cluster
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
```

This declares a Hopper kernel with TMA loads, warp-specialized cooperative scheduling, and a fused epilogue, in ~10 lines. Compile time is heavy and error messages are infamous, but achieved performance is best-in-class.

== Autotuning Beyond Configs

A common pattern in any DSL is a *cost model + search* outer loop. Triton's autotuner is exhaustive; production systems do more:

#table(
  columns: 3,
  [*Tool*], [*Search strategy*], [*Notes*],
  [`triton.autotune`], [exhaustive over user configs], [seconds],
  [`torch.compile` (Inductor)], [exhaustive on per-shape templates], [cached across runs],
  [TVM / AutoTVM],     [simulated-annealing + cost model], [hours but reusable],
  [Ansor / MetaSchedule], [evolutionary search],     [minutes],
  [Auto-Triton (research)], [learned tile predictor], [near-zero search cost],
)

== Pros and Cons

#table(
  columns: 3,
  [*DSL*], [*Strengths*], [*Weaknesses*],
  [Triton], [Python; portable across NVIDIA arches; great GEMM/attn],
  [Less control than CUTLASS; one kernel = one block program],
  [Pallas], [Same kernel on GPU and TPU; JAX-native],
  [Younger; smaller op coverage],
  [ThunderKittens], [Researcher-friendly C++; explicit hardware exposure],
  [Less abstract; NVIDIA-only],
  [CUTLASS], [Peak performance; production-ready],
  [Steep learning curve; slow compile; C++ template errors],
)

== Further Reading

Tillet, P. et al. (2019). "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." _MAPL_. Original Triton paper introducing the tiled execution model and blocked load/store semantics.

OpenAI / PyTorch (2024). _Triton Tutorials_. https://triton-lang.org/

Google (2024). _JAX Pallas Documentation_. https://jax.readthedocs.io/en/latest/pallas/

Spector, B. et al. (2024). "ThunderKittens: Simple, Fast, and Adorable AI Kernels." Hazy Research blog and arXiv.

NVIDIA (2024). _CUTLASS 3.x Documentation_ and _CuTe Layout Algebra Reference_. https://github.com/NVIDIA/cutlass

Chen, T. et al. (2018). "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." _OSDI_.

Zheng, L. et al. (2020). "Ansor: Generating High-Performance Tensor Programs for Deep Learning." _OSDI_.

Ansel, J. et al. (2024). "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation." _ASPLOS_.

Lattner, C. et al. (2021). "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation." CGO. Describes the MLIR framework used by IREE, CIRCT, and the broader GPU compiler ecosystem.

Vasilache, N. et al. (2019). "The Tensor Algebra Compiler." OOPSLA. Describes the TACO compiler for sparse tensor algebra; motivation for structured sparsity in compilers.

Google (2020). "XLA: Compiling Machine Learning for Peak Performance." Google AI Blog. Overview of XLA's HLO IR, fusion, and layout assignment for TPU and GPU targets.

OpenAI (2023). Triton Language Documentation. Reference for Triton's Python DSL, JIT compilation, and integration with PyTorch's `torch.compile`.

Cui, H. et al. (2023). "Efficiently Programming Large-Scale Neural Networks for GPU." SOSP. Covers operator fusion strategies and how compilers like Triton and XLA schedule work across SM clusters.

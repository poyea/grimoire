= Non-NVIDIA Accelerators

NVIDIA dominates the discrete-GPU market but is no longer the only credible target for serious compute. AMD's MI300-series competes head-to-head on training, Apple's Metal stack runs production inference on tens of millions of laptops, Intel ships Xe and Gaudi accelerators, and a wave of startups (Tenstorrent, Cerebras, Groq, Graphcore) offers genuinely different architectures. This chapter surveys the alternatives and the portability stories that target them.

*See also:* _Compute Units and Specialized Cores_ (NVIDIA SMs for contrast), _ML Workload Optimization on GPUs_ (the workloads everyone is chasing).

== AMD — CDNA and RDNA

AMD splits its GPU lines:
- *RDNA* (Radeon DNA): consumer/gaming GPUs (RX 6000 / 7000 / 9000).
- *CDNA* (Compute DNA): datacenter accelerators (MI100/MI200/MI300), no raster pipeline at all.

=== CDNA 2 — MI250X

The Frontier exascale machine is built from MI250X GPUs: two GCDs (Graphics Compute Dies) per package connected by Infinity Fabric, exposed as two logical devices to ROCm.

```
MI250X:  2 x GCD, each:
  110 CUs,  220 matrix cores,  64 GB HBM2e,  1.6 TB/s
  Peak FP64 vector: 47.9 TFLOPS
  Peak FP64 matrix: 95.7 TFLOPS  (unique: FP64 matrix throughput)
  Peak FP16/BF16 matrix: 383 TFLOPS
```

The FP64 *matrix* engine (matrix cores delivering full-precision GEMM) is the differentiator for HPC.

=== CDNA 3 — MI300

The MI300 family uses chiplets aggressively:

#table(
  columns: 4,
  [*Part*], [*GPU CUs*], [*CPU*], [*Memory*],
  [MI300X], [304 CDNA3 CUs], [none], [192 GB HBM3, 5.3 TB/s],
  [MI300A], [228 CDNA3 CUs], [24 Zen 4 cores], [128 GB unified HBM3],
  [MI325X], [304 CUs (refresh)], [none], [256 GB HBM3e, 6 TB/s],
)

MI300X targets H100/H200 directly on memory-bound LLM inference; the larger HBM lets a 70B-parameter model fit on a single accelerator in FP16. MI300A is APU-style with unified GPU+CPU memory, a true successor to the discrete-accelerator model.

Tensor throughput on MI300X: ~1300 TFLOPS BF16 / FP16, ~2600 TFLOPS FP8 (peak). Real-world ratios vs H100 depend heavily on the software stack; ROCm has historically trailed CUDA in kernel maturity, though by 2024–2025 the gap on vLLM/TGI inference is small for popular models.

=== RDNA

RDNA emphasizes graphics features (RT cores, mesh shaders) but the same WGP/CU architecture runs compute. RDNA 3 introduces dual-issue ALUs and a new matrix engine (WMMA on Radeon, distinct from CDNA's matrix cores). RDNA 4 (RX 9000, 2025) adds AI accelerators per CU.

=== ROCm / HIP

*ROCm* is AMD's GPU stack, layered roughly like CUDA:

```
PyTorch / JAX / vLLM
      |
hipBLAS, hipDNN, RCCL, hipFFT, MIOpen
      |
HIP runtime          (CUDA-compatible API)
      |
HSA / KFD            (kernel driver)
      |
ROCm-LLVM amdgcn backend
```

*HIP* is the user-facing language: a near-1:1 rename of CUDA C++ (`cudaMalloc` $arrow.r$ `hipMalloc`, `__syncthreads` is the same). The `hipify` tool ports most CUDA code mechanically.

```cpp
#include <hip/hip_runtime.h>
__global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
int main() {
    float *dx, *dy;
    hipMalloc(&dx, N*sizeof(float));
    hipMalloc(&dy, N*sizeof(float));
    saxpy<<<(N+255)/256, 256>>>(2.0f, dx, dy, N);
    hipDeviceSynchronize();
}
```

Compile with `hipcc`. The same source builds for NVIDIA (HIP-over-CUDA) and AMD (HIP-over-ROCm). The leakage is in the long tail: PTX inline assembly, warp size (NVIDIA 32, AMD 64), tensor-core intrinsics, and the cuBLAS API surface area not yet mirrored by hipBLAS.

=== RCCL

RCCL is NCCL ported to ROCm with the same API (`ncclXxx` $arrow.r$ `rcclXxx`), same algorithms (ring, double-tree), and Infinity Fabric in place of NVLink.

== Apple — Metal Performance Shaders

Apple's M-series unified-memory SoCs (M1 $arrow.r$ M4) put 8–80 GB of LPDDR5X on a single package shared by CPU and GPU. For LLM inference this is a quietly important platform: an M3 Max with 128 GB runs 70B-parameter Llama in 4-bit comfortably at interactive speed.

The compute stack:

```
PyTorch (MPS backend) / mlx / llama.cpp / Ollama
      |
Metal Performance Shaders Graph (MPSGraph)
      |
Metal (shader language: MSL, C++14-based)
      |
GPU driver (Apple-internal IR: AIR)
```

A Metal compute kernel:
```metal
#include <metal_stdlib>
using namespace metal;
kernel void saxpy(device const float* x [[buffer(0)]],
                  device       float* y [[buffer(1)]],
                  constant     float& a [[buffer(2)]],
                  uint id [[thread_position_in_grid]]) {
    y[id] = a * x[id] + y[id];
}
```

Notable features:
- *Unified memory:* zero-copy CPU/GPU; no `cudaMemcpy` analog.
- *Simdgroup* = warp = 32 lanes; `simd_shuffle`, `simd_sum`.
- *Tile shaders* and *threadgroup memory* for shared-memory-style tiling.
- *MPS Graph* operator library is the high-level path used by PyTorch's `mps` backend and by MLX.

MLX (Apple, 2023) is a NumPy-like array framework with lazy evaluation, designed specifically for Apple Silicon. It targets MPS directly and is the recommended path for on-device ML on Macs.

== Intel — Xe and Gaudi

=== Xe GPUs

Intel ships Xe in three tiers: Xe-LP (integrated), Xe-HPG (Arc discrete), and Xe-HPC (Data Center GPU Max, formerly Ponte Vecchio, used in the Aurora supercomputer).

Xe-HPC is a chiplet design with up to 128 Xe-cores and HBM2e; it provides XMX matrix engines analogous to tensor cores. The programming surface is *SYCL* via *oneAPI*.

```cpp
#include <sycl/sycl.hpp>
sycl::queue q;
q.parallel_for(N, [=](sycl::id<1> i) {
    y[i] = a * x[i] + y[i];
}).wait();
```

SYCL is open-standard, single-source C++, and Intel's compiler `dpcpp` can target NVIDIA and AMD via Codeplay's plugins. oneMKL/oneDNN play the cuBLAS/cuDNN role.

=== Gaudi

Gaudi (Habana, acquired by Intel) is a dedicated training accelerator, not a GPU. Gaudi 3 (2024):
- 8 Matrix Math Engines, 64 TPCs (Tensor Processor Cores, VLIW SIMD).
- 128 GB HBM2e.
- 24 $times$ 200 Gbps RDMA-over-Ethernet *built into the die*; no separate NIC.

Programming via SynapseAI, the Habana graph compiler. Most users sit above it in PyTorch with `import habana_frameworks.torch.core as htcore`. The integrated RoCE makes Gaudi an unusual hyperscale option: a 16-node rack is wired without an InfiniBand fabric.

== Tenstorrent

Tenstorrent's Wormhole and Blackhole accelerators take a radically different approach: a 2D grid of *Tensix cores*, each with a small CPU (RISC-V) and a matrix engine, connected by a NoC. Programs are *data-flow*: tensors stream from core to core rather than being held in a shared memory hierarchy.

The host-side API is *TT-Metalium* (low-level, like CUDA) and *TT-NN* (PyTorch-like). Each operator is a graph of kernel placements on the Tensix mesh; the compiler partitions tensors and inserts NoC sends.

```python
import ttnn
device = ttnn.open_device(device_id=0)
a = ttnn.from_torch(torch.randn(1024, 1024), device=device,
                    layout=ttnn.TILE_LAYOUT)
b = ttnn.from_torch(torch.randn(1024, 1024), device=device,
                    layout=ttnn.TILE_LAYOUT)
c = ttnn.matmul(a, b)
```

The bet: dataflow scales better than ever-larger HBM caches as model sizes grow.

== Comparison

#table(
  columns: 5,
  [*Vendor*], [*Flagship*], [*Mem*], [*Stack*], [*Niche*],
  [NVIDIA],   [B200], [192 GB HBM3e], [CUDA],     [training + inference, default],
  [AMD],      [MI325X], [256 GB HBM3e], [ROCm/HIP], [memory-rich inference, HPC],
  [Apple],    [M3 Ultra], [up to 192 GB unified], [Metal/MLX], [on-device, dev workstation],
  [Intel],    [GPU Max 1550], [128 GB HBM2e], [oneAPI/SYCL], [HPC (Aurora)],
  [Intel/Habana], [Gaudi 3], [128 GB HBM2e], [SynapseAI], [hyperscale training],
  [Tenstorrent], [Blackhole], [32 GB GDDR6], [TT-Metalium], [dataflow research],
)

== Portability: the Reality

Three honest paths today:

+ *CUDA-first, port later:* write CUDA, hipify for AMD, hope vendors keep up. Ships fastest, leaves performance on the floor on non-NVIDIA targets.
+ *Triton / Pallas / OpenAI-Triton-on-AMD:* one DSL, multiple backends. Works for common ops (GEMM, attention); falls back to vendor libraries for the rest. Increasingly viable.
+ *Framework-level only:* write PyTorch / JAX, let the backend decide. Maximum portability, minimum bare-metal control. The right choice for 90% of users.

The "no-CUDA-lock-in" pitch has been told for a decade and is *finally* starting to be true for inference; vLLM, llama.cpp, and TGI all run respectably on MI300X, Gaudi 3, and Apple Silicon as of 2025.

== Further Reading

AMD (2024). _CDNA 3 Architecture Whitepaper_ and _AMD Instinct MI300X Datasheet_.

AMD (2024). _ROCm Documentation_. https://rocm.docs.amd.com/

AMD (2024). _HIP Programming Guide_. https://rocm.docs.amd.com/projects/HIP/

Apple (2024). _Metal Shading Language Specification_ and _MLX Documentation_. https://ml-explore.github.io/mlx/

Intel (2023). _Xe HPC Architecture (Ponte Vecchio) Microarchitecture Specification_.

Intel/Habana (2024). _Gaudi 3 White Paper_.

Tenstorrent (2024). _TT-Metalium Programming Guide_. https://docs.tenstorrent.com/

Khronos Group (2023). _SYCL 2020 Specification_.

Vasilache, N. et al. (2024). "An Overview of MLIR for ML Accelerators." _IEEE Micro_.

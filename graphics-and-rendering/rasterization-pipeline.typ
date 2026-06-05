= The Rasterization Pipeline

Rasterization remains the dominant real-time rendering paradigm because GPUs are organized — at every level from the SM down to the ROP — to scan-convert triangles in massively parallel batches. Understanding the fixed-function stages and their programmable hooks is the prerequisite for everything that follows: shaders, $"PBR"$, deferred lighting, and even hybrid ray-traced effects.

*See also:* _shaders.typ_ (programmable stages), _realtime-engines.typ_ (Nanite virtualized geometry), `gpu-architecture/compute-architecture.typ` (SM, ROPs), `gpu-architecture/memory-hierarchy.typ` (tile caches).

== Pipeline Overview

A modern $"GPU"$ pipeline (Direct3D 12 / Vulkan / Metal) looks like:

#table(
  columns: 3,
  [*Stage*], [*Programmable?*], [*Role*],
  [Input Assembler ($"IA"$)], [No], [Fetch vertex / index buffers, primitive topology],
  [Vertex Shader ($"VS"$)], [Yes], [Per-vertex transform, skinning],
  [Hull / Tessellation Control], [Yes], [Patch-level $"LOD"$ factors],
  [Tessellator], [No (fixed)], [Subdivision into primitives],
  [Domain / Tessellation Evaluation], [Yes], [Per-generated-vertex shading],
  [Geometry Shader ($"GS"$)], [Yes (deprecated)], [Per-primitive amplification],
  [Mesh / Task Shaders], [Yes (replaces $"VS"$+$"GS"$)], [Cluster-based geometry],
  [Rasterizer / Setup], [No], [Triangle setup, edge equations, coverage],
  [Pixel / Fragment Shader ($"PS"$)], [Yes], [Per-fragment shading],
  [Output Merger / $"ROP"$], [No], [Depth, stencil, blend, write to RT],
)

The fixed-function stages (rasterizer, $"ROP"$) consume more silicon area than people expect. Rasterizers issue $approx 32$ pixels per clock per partition; $"ROPs"$ compress depth/color and perform the final framebuffer atomics.

== Triangle Setup and Edge Equations

Given clip-space vertices $v_0, v_1, v_2$, after perspective divide and viewport transform we obtain screen-space coordinates $(x_i, y_i)$. The signed area is

$ A = (x_1 - x_0)(y_2 - y_0) - (x_2 - x_0)(y_1 - y_0). $

For each edge $e_i$ the edge function is

$ E_i (x, y) = (x - x_i) (y_(i+1) - y_i) - (y - y_i) (x_(i+1) - x_i). $

A sample $(x, y)$ is inside the triangle iff all three $E_i$ have the same sign as $A$. Hardware evaluates $E_i$ incrementally over a 2$times$2 quad — the *quad*, the atomic unit of rasterization on every $"GPU"$ since 2001. Quads guarantee that screen-space derivatives (`ddx`, `ddy`) are available cheaply for mipmap $"LOD"$ selection.

=== Barycentrics

Once inside, perspective-correct barycentrics for attribute interpolation are computed as

$ lambda_i = (E_i \/ w_i) / (sum_j E_j \/ w_j) $

where $w_i$ is the homogeneous clip-space $w$ component. Without the $1\/w$ divides you would see the famous PS1-era texture warping.

== Tile-Based vs Immediate-Mode Rasterization

Mobile $"GPUs"$ (PowerVR, Mali, Adreno, Apple) and recent desktop architectures (NVIDIA Maxwell+ binning, AMD $"DSBR"$) adopt *tile-based* rasterization. The screen is partitioned into tiles (e.g. 16$times$16 or 32$times$32 pixels), geometry is binned by the tiles it overlaps, then each tile is rendered entirely on-chip before its results are flushed to memory.

#table(
  columns: 3,
  [*Property*], [*Immediate ($"IMR"$)*], [*Tile-based ($"TBR"$/$"TBDR"$)*],
  [Memory traffic], [Per-pixel R/W to $"VRAM"$], [Tile resident in $"SRAM"$],
  [Overdraw cost], [Pays full shading], [$"TBDR"$ defers shading after $"HSR"$],
  [Binning step], [None], [Yes, geometry sorted to tiles],
  [Render pass model], [Implicit], [Explicit (`vkCmdBeginRenderPass`)],
  [Power efficiency], [Lower], [Higher (mobile)],
  [Latency on triangle], [Lower], [Higher (waits for binning)],
)

Vulkan and Metal explicit render passes exist precisely because $"TBDR"$ vendors need to know which attachments are tile-resident and which actually need to be written back to global memory (`STORE_OP_DONT_CARE` vs `STORE_OP_STORE`).

== Early-Z, Hi-Z, and Hierarchical Culling

Naively the depth test happens after the pixel shader. But if the shader has no `discard`, no depth output, and no side-effects, hardware can perform an *early-Z* test before shading. Combined with Hi-Z — a coarse min/max depth pyramid stored per tile — entire 8$times$8 blocks can be rejected without per-pixel work.

```glsl
// This shader is early-Z friendly.
layout(location=0) out vec4 outColor;
void main() {
    outColor = vec4(shade(), 1.0);
}

// This one disables early-Z because of discard.
void main() {
    if (texture(opacity, uv).a < 0.5) discard;
    outColor = vec4(shade(), 1.0);
}
```

A standard optimization: a *depth prepass* renders only depth, then the color pass runs with early-Z enabled and never shades occluded fragments. Cost is one extra pass, savings often net positive when overdraw is high (e.g. open-world foliage).

== Modern Geometry: Mesh & Task Shaders

The classic $"VS"$ $arrow.r$ $"GS"$ pipeline is monolithic per draw call. Mesh shaders (Turing+, RDNA2+) replace it with a two-stage cooperative model:

- *Task shader (amplification shader)*: like a compute shader, dispatches *meshlets*.
- *Mesh shader*: outputs a small (up to 256 vertices, 256 primitives) meshlet to the rasterizer.

```hlsl
// HLSL mesh shader skeleton
struct Meshlet { uint vertexOffset; uint primOffset;
                 uint vertexCount; uint primCount; };
StructuredBuffer<Meshlet> Meshlets;

[numthreads(64, 1, 1)]
[outputtopology("triangle")]
void MeshMain(uint gid : SV_GroupID, uint tid : SV_GroupThreadID,
              out vertices VertexOut verts[64],
              out indices uint3 tris[124])
{
    Meshlet m = Meshlets[gid];
    SetMeshOutputCounts(m.vertexCount, m.primCount);
    if (tid < m.vertexCount) verts[tid] = LoadVertex(m, tid);
    if (tid < m.primCount)   tris[tid]  = LoadPrim(m, tid);
}
```

Meshlets enable per-cluster culling (frustum, backface cone, $"HZB"$ occlusion) before rasterization — the foundation of Unreal Nanite (covered in _realtime-engines.typ_).

== Multisample, Supersample, and Coverage

$"MSAA"$ stores $N$ depth+coverage samples per pixel but only one shader invocation per *covered* pixel (per primitive). The pixel shader runs once and the result is broadcast to covered samples; depth is per-sample. The resolve pass averages samples. *Centroid* interpolation evaluates attributes at the centroid of the covered samples to avoid extrapolation artifacts at silhouettes.

$"SSAA"$ runs the shader per sample — quality is best but cost scales linearly. $"TAA"$ amortizes super-sampling over time using jittered camera matrices and history reprojection; almost universal in modern engines.

== Render Targets, Depth, and Compression

Color targets use $"DCC"$ (Delta Color Compression) and depth uses Hi-Z + tile compression. Bandwidth savings are 30–50% on average. *Clear values* are exposed in the $"API"$ (Vulkan `VkClearValue`) so that hardware can implement them as compression metadata only — actual $"VRAM"$ writes are deferred until first sample.

```cpp
// Vulkan render pass description
VkAttachmentDescription color = {
    .format = VK_FORMAT_R8G8B8A8_UNORM,
    .samples = VK_SAMPLE_COUNT_4_BIT,
    .loadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR,
    .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
    .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    .finalLayout   = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
};
VkAttachmentDescription resolve = {
    .format = VK_FORMAT_R8G8B8A8_UNORM,
    .samples = VK_SAMPLE_COUNT_1_BIT,
    .loadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
};
```

== Forward vs Deferred vs Visibility Buffer

The single biggest pipeline choice for an engine is *where* shading happens.

#table(
  columns: 4,
  [*Property*], [*Forward*], [*Deferred*], [*Visibility Buffer*],
  [Shader runs on], [Each primitive], [Each lit pixel (post G-buffer)], [Each visible material cluster],
  [G-buffer size], [None], [4–8 RTs], [1 RT (triId, primId)],
  [$"MSAA"$], [Cheap], [Painful (per-sample resolve)], [Natural],
  [Material variety], [Many shader permutations], [Uniform $"BRDF"$], [Material pass per cluster],
  [Translucency], [Native], [Forward+ fallback], [Forward+ fallback],
  [Used by], [Mobile, Unity URP], [Unreal, Frostbite (legacy)], [Nanite, Activision],
)

The *visibility buffer* (Burns & Hunt 2013, later popularized by Unreal Nanite) writes only a triangle ID per pixel; shading is then a full-screen pass that reconstructs vertex attributes via barycentrics. This decouples geometry density from shading cost — exactly what Nanite needs.

== Profiling the Pipeline

GPU vendors expose pipeline statistics:

```cpp
VkQueryPool pool;
VkQueryPoolCreateInfo ci{ .queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS,
    .pipelineStatistics =
        VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT |
        VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
        VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT };
```

Useful invariants: clipping ratio (fraction of culled prims), pixel/fragment ratio (overdraw factor), vertex / primitive ratio. RenderDoc, Nsight Graphics, and Radeon GPU Profiler visualize all of the above.

== Common Pitfalls

- Binding huge index buffers with `VK_INDEX_TYPE_UINT32` when `UINT16` would suffice: doubles $"IA"$ bandwidth.
- Many small draw calls $arrow.r$ $"CPU"$ bound; collapse with instancing or `vkCmdDrawIndexedIndirect`.
- Pixel shader writing depth ($"SV_Depth"$) disables Hi-Z and conservative Z. Use $"SV_DepthLessEqual"$ / $"SV_DepthGreaterEqual"$ if monotone.
- Half-pixel-center conventions differ between D3D9 (corner) and D3D10+/Vulkan (center) — silent off-by-half can ruin postprocessing.

== Further Reading

Akenine-Möller, T., Haines, E., Hoffman, N. et al. (2018). _Real-Time Rendering_, 4th ed. CRC Press — chapters 3 and 23.

Burns, C., Hunt, W. (2013). "The Visibility Buffer: A Cache-Friendly Approach to Deferred Shading." Journal of Computer Graphics Techniques.

Pranckevičius, A. (2014). "Forward+ Decal Rendering." SIGGRAPH course notes.

Karis, B. (2021). "Nanite — A Deep Dive." Advances in Real-Time Rendering, SIGGRAPH.

Sellers, G., Wright, R., Haemel, N. (2016). _OpenGL Superbible_, 7th ed.

Khronos Group (2024). "Vulkan 1.3 Specification."

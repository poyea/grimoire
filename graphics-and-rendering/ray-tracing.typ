#import "../template.typ": xref

= Ray Tracing

Hardware-accelerated ray tracing, standardized through $"DXR"$ (DirectX Raytracing) and Vulkan Ray Tracing, adds a programmable ray-intersection pipeline alongside the rasterization pipeline. This enables effects — soft shadows, reflections, ambient occlusion, and global illumination — that either require expensive screen-space hacks or prohibitive offline cost when done in a rasterizer alone.

*See also:* #xref("graphics-and-rendering", "rasterization-pipeline", label: "The Rasterization Pipeline") (hybrid integration), #xref("graphics-and-rendering", "physically-based-rendering", label: "Physically Based Rendering") (rendering equation, BRDF), #xref("graphics-and-rendering", "gi-techniques", label: "Global Illumination Techniques") (Monte Carlo path tracing, ReSTIR), #xref("gpu-architecture", "compute-architecture", label: "Compute Units and Specialized Cores") (gpu-architecture framing; RT cores, warp divergence).

== Acceleration Structures

Ray-scene intersection is asymptotically $O(N)$ per ray without spatial data structures. All modern $"API"$s build a two-level hierarchy of *bounding volume hierarchies* ($"BVHs"$).

=== TLAS and BLAS

- *Bottom-Level Acceleration Structure* ($"BLAS"$): built from geometry triangles or $"AABB"$ procedural geometry for a single object or mesh group. The driver computes tight-fitting axis-aligned bounding boxes and organizes them into a binary tree.
- *Top-Level Acceleration Structure* ($"TLAS"$): contains one entry per $"BLAS"$ instance plus a $4 times 3$ affine transform matrix. Ray traversal descends the $"TLAS"$, transforms the ray into $"BLAS"$ space, then descends the $"BLAS"$ leaf.

The two-level split means that a rigid body's $"BLAS"$ is built once; animation only re-builds the cheap $"TLAS"$. Skinned meshes that deform topology must rebuild their $"BLAS"$ per-frame (or use $"REFIT"$, which only updates $"AABB"$ bounds without restructuring the tree — cheaper, slightly lower quality).

=== Surface Area Heuristic ($"SAH"$)

The *surface area heuristic* models expected traversal cost as

$ C = C_t + (S_L / S_P) N_L C_i + (S_R / S_P) N_R C_i, $

where $C_t$ is the cost of a node traversal, $C_i$ is the cost of a primitive intersection test, $S$ denotes surface area, and $L$/$R$/$P$ refer to the left child, right child, and parent. Minimizing $C$ over all candidate split planes yields a near-optimal tree for random rays. Build time is $O(N log^2 N)$ with $O(N log N)$ variants.

=== LBVH: Linear BVH

*Linear $"BVH"$* assigns each primitive a Morton code — a Z-curve index that interleaves bit-planes of $x$, $y$, $z$ centroid coordinates — then sorts primitives by Morton code. Adjacent primitives in Morton order are spatially nearby, so recursively splitting on the highest-differing bit produces a valid $"BVH"$ in $O(N log N)$ and is trivially parallelizable on a $"GPU"$.

```glsl
// Morton code (30-bit) for a centroid in [0,1]^3
uint expandBits(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}
uint morton3D(float x, float y, float z) {
    x = min(max(x * 1024.0, 0.0), 1023.0);
    y = min(max(y * 1024.0, 0.0), 1023.0);
    z = min(max(z * 1024.0, 0.0), 1023.0);
    return (expandBits(uint(x)) << 2) |
           (expandBits(uint(y)) << 1) |
            expandBits(uint(z));
}
```

$"LBVH"$ is the default for driver-side real-time $"BVH"$ construction because it suits $"GPU"$ parallelism. $"SAH"$-based builds happen offline or as an async prepare pass.

== DXR and Vulkan Ray Tracing Shader Stages

The ray-tracing pipeline introduces five new programmable shader stages absent from the rasterization pipeline.

#table(
  columns: 3,
  [*Stage*], [*Invoked when*], [*Typical use*],
  [Ray Generation ($"RGS"$)], [Once per pixel (or tile)], [Camera rays, spawning secondary rays],
  [Miss ($"MS"$)], [Ray misses all geometry], [Sky color, ambient term],
  [Closest-Hit ($"CHS"$)], [Closest committed intersection], [Shading, shadow query, spawn new rays],
  [Any-Hit ($"AHS"$)], [Every candidate intersection], [Alpha test, transparency accumulation],
  [Intersection ($"IS"$)], [Hitting procedural $"AABB"$], [Analytic spheres, curves, $"SDF"$ proxies],
)

The driver assembles these into a *shader binding table* ($"SBT"$): a buffer indexed by instance, geometry, and ray type. A single `TraceRay` / `traceRayEXT` call dispatches into the $"SBT"$ automatically.

```hlsl
// DXR ray generation shader (HLSL)
RaytracingAccelerationStructure scene : register(t0);
RWTexture2D<float4> output            : register(u0);

[shader("raygeneration")]
void RayGen() {
    uint2 idx = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;
    float2 uv = (float2(idx) + 0.5) / float2(dim);

    RayDesc ray;
    ray.Origin    = gCamera.position;
    ray.Direction = normalize(CameraRayDir(uv));
    ray.TMin      = 1e-3;
    ray.TMax      = 1e5;

    Payload payload = { float3(0,0,0), false };
    TraceRay(scene, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, payload);
    output[idx] = float4(payload.color, 1.0);
}

[shader("closesthit")]
void ClosestHit(inout Payload p, BuiltInTriangleIntersectionAttributes a) {
    float3 bary = float3(1 - a.barycentrics.x - a.barycentrics.y,
                         a.barycentrics.x, a.barycentrics.y);
    p.color = EvaluateMaterial(bary, PrimitiveIndex(), InstanceID());
    p.hit   = true;
}

[shader("miss")]
void Miss(inout Payload p) {
    p.color = SampleSky(WorldRayDirection());
    p.hit   = false;
}
```

== Ray Differentials

A single ray carries no information about adjacent rays, making texture $"LOD"$ and antialiasing impossible without extra bookkeeping. *Ray differentials* (Igehy 1999) track the partial derivatives $partial r \/ partial x$ and $partial r \/ partial y$ alongside the primary ray. After intersection the differentials transfer through the hit geometry, giving $d u\/d x$, $d u\/d y$, $d v\/d x$, $d v\/d y$ for a standard mipmap $"LOD"$ calculation:

$ lambda = log_2 max(||partial u\/partial x||, ||partial u\/partial y||). $

For $"GPU"$ ray tracing the full derivative propagation is expensive. Cheaper approximations (cone spread angle, ray footprint radius) suffice for most real-time use cases.

== Hybrid Rasterization + Ray Tracing

Fully path-traced frames at 60 $"Hz"$ remain out of reach on consumer hardware. Hybrid pipelines rasterize primary visibility and fire rays only for the effects that benefit most.

=== Shadows

Replace shadow maps with *ray-traced shadow* queries: a single any-hit or shadow ray (`RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH`) per shadow per pixel. Cost: $approx 1$ $"ms"$ at 1080p for one directional light. Benefits over $"PCF"$ shadow maps: correct penumbra shape, no aliasing or Peter-Panning, handles complex occluder geometry trivially.

=== Reflections

Rasterized reflections rely on $"SSR"$ (screen-space reflections), which breaks at silhouettes and off-screen geometry. A hybrid approach fires one specular ray per pixel, shades the hit with the pre-computed $"BRDF"$ lobe and denoises. NVIDIA $"RTXR"$ and $"AMD"$ $"FidelityFX"$ Hybrid Reflections follow this model.

=== Ambient Occlusion

*$"RTAO"$* (ray-traced $"AO"$): trace $N in [1, 4]$ short hemisphere rays per pixel, average the unoccluded fraction. Replaces $"SSAO"$ / $"HBAO"$ for near-contact shadowing without screen-space artifacts at edges.

#table(
  columns: 3,
  [*Effect*], [*Rays per pixel*], [*Denoiser needed*],
  [Hard shadows], [1], [Optional (spatial filter)],
  [Soft shadows], [1–4], [Yes ($"SVGF"$ or $"NRD"$)],
  [Specular reflections], [1], [Yes],
  [$"RTAO"$], [1–4], [Yes],
  [Diffuse $"GI"$], [1–4], [Yes ($"ReSTIR"$, $"NRD"$)],
)

== Denoising

Low sample counts produce noisy images. *Denoisers* filter spatially and temporally to reconstruct a clean image from 1–4 spp.

=== $"SVGF"$: Spatiotemporal Variance-Guided Filtering

Schied et al. (2017) introduced *$"SVGF"$* (Spatiotemporal Variance-Guided Filtering). The key insight is to estimate per-pixel variance $sigma^2$ from a temporal history of moment estimates:

$ sigma^2 (x) = mu_2 (x) - mu_1 (x)^2, quad mu_k (x) = (1-alpha) mu_k^("prev") (x') + alpha s^k (x). $

A joint bilateral à-trous wavelet filter uses $sigma$ as an edge-stopping weight alongside geometry (depth, normal). Multiple à-trous passes cover large spatial support cheaply. $"SVGF"$ is 3–5 $"ms"$ on modern hardware and handles shadows, $"AO"$, and $"GI"$ signals.

=== $"NRD"$: NVIDIA Real-Time Denoisers

$"NRD"$ (Tokuyoshi & Harada 2020, NVIDIA 2021) is a library of per-signal denoisers ($"ReLAX"$ for diffuse+specular $"GI"$, $"REBLUR"$ for radiance buffers, $"SIGMA"$ for shadows) that ship with $"DLSS"$ $"RR"$ (Ray Reconstruction). Each denoiser understands the statistical properties of its signal (e.g., specular reprojection uses a virtual hit-point rather than surface reprojection to avoid ghosting on glossy surfaces).

== Build and Traversal Performance Notes

- Compaction: after building a $"BLAS"$, call `vkCmdCopyAccelerationStructureKHR` with `COMPACT` mode to halve typical $"BLAS"$ memory.
- Update vs Rebuild: `BUILD_ALLOW_UPDATE` enables $"REFIT"$, which is $3 times$ faster than rebuild but grows $"BLAS"$ $"SAH"$ cost up to $30\%$ after many frames.
- Warp divergence: different shader table entries per instance cause warp divergence on the $"CHS"$ stage. Sort rays by material type before tracing to improve coherence (NVIDIA Falcor, AMD $"GPUOpen"$ samples).
- Ray sorting: Morton-ordered rays from a screen-space tile improve $"L"$2 cache hit rate in $"TLAS"$ traversal by $approx 15\%$.

== Further Reading

Shirley, P. et al. (2024). _Ray Tracing in One Weekend_ series. (Open access.)

Wald, I. et al. (2007). "Ray Tracing Deformable Scenes Using Dynamic Bounding Volume Hierarchies." ACM TOG.

Igehy, H. (1999). "Tracing Ray Differentials." SIGGRAPH.

Schied, C. et al. (2017). "Spatiotemporal Variance-Guided Filtering." HPG.

Akenine-Möller, T., Haines, E., Hoffman, N. et al. (2018). _Real-Time Rendering_, 4th ed. CRC Press — chapter 25.

Microsoft (2023). "DirectX Raytracing (DXR) Functional Spec." GitHub/microsoft/DirectX-Specs.

Khronos Group (2024). "VK\_KHR\_ray\_tracing\_pipeline." Vulkan 1.3 extensions.

NVIDIA (2022). "NRD — NVIDIA Real-Time Denoisers." GitHub/NVIDIAGameWorks/RayTracingDenoiser.

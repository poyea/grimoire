#import "../template.typ": xref

= Real-Time Engine Architecture <realtime-engines>

Modern real-time engines like Unreal Engine 5 and Unity's High Definition Render Pipeline have moved well beyond fixed forward-rendering loops. They are software-defined rendering systems that schedule heterogeneous work — compute, async, and traditional graphics — through explicit graph abstractions, while managing virtualized geometry, temporal image reconstruction, and dynamic streaming of assets that exceed $"VRAM"$ budget.

*See also:* #xref("graphics-and-rendering", "rasterization-pipeline", label: "The Rasterization Pipeline") (mesh shaders, visibility buffer), #xref("graphics-and-rendering", "gi-techniques", label: "Global Illumination Techniques") (Lumen GI, DDGI probes), #xref("graphics-and-rendering", "ray-tracing", label: "Ray Tracing") (BVH, DXR, denoising), #xref("gpu-architecture", "memory-hierarchy", label: "GPU Memory Hierarchy") (gpu-architecture framing; tile caches, $"VRAM"$ bandwidth).

== Nanite: Virtualized Geometry

Karis et al. (2021) describe *Nanite* as a virtualized geometry system that renders scenes with billions of source polygons while consuming only pixel-rate shading work. The core insight is to decouple geometry density from $"CPU"$ draw call count.

=== Cluster $"DAG"$

Every imported mesh is pre-processed offline into a *cluster directed acyclic graph* ($"DAG"$):

- The mesh is partitioned into *clusters* of $approx 128$ triangles each using the METIS graph partitioner or similar.
- Clusters are grouped into *cluster groups* of $approx 8$–$32$, then simplified via quadric error metric. The simplified group generates a new set of clusters one $"LOD"$ level up.
- This continues until the entire mesh is one cluster at the coarsest $"LOD"$. The result is a $"DAG"$ where each node is a cluster and edges point from finer to coarser levels.

At run-time, Nanite traverses the $"DAG"$ and selects a *cut* — one cluster per subtree — such that each selected cluster projects to $approx 1$ pixel diameter on screen. The cut selection criterion is

$ "ClusterError"_("screen") = "ClusterError"_("world") / d times h / (2 tan(phi\/2)) < epsilon_"px", $

where $d$ is camera distance, $h$ is screen height, $phi$ is vertical $"FOV"$, and $epsilon_"px"$ is the target error in pixels (default $1$).

=== Software Rasterizer

Nanite clusters with small projected area (few triangles per pixel) bypass the hardware rasterizer and use a custom *software rasterizer* running in compute. Each thread claims a triangle and rasterizes it via $64$-bit atomic `max` writes to a visibility buffer, encoding the cluster $"ID"$ and triangle $"ID"$ in the high/low bits of the atomic. This avoids the overhead of fixed-function setup for sub-pixel triangles and the state-change cost of millions of draw calls.

Larger triangles ($> 32$ pixels on a side) fall back to hardware rasterization via indirect draw calls batched per cluster.

=== Material Evaluation

Nanite uses the *visibility buffer* pattern: a single `uint64` render target stores `(clusterID << 32) | triangleID` per pixel. Rather than shading each triangle as it rasterizes (which would thrash the instruction cache when adjacent pixels belong to different materials), a deferred full-screen material pass reads the visibility buffer, reconstructs vertex attributes via barycentrics, and evaluates the appropriate material. This decoupling is the key insight: geometry density no longer couples to shading cost — a surface covered by 10 million triangles costs exactly the same to shade as one covered by a single quad.

```hlsl
// Nanite visibility decode (simplified HLSL)
uint64_t vis = VisibilityBuffer[pixel];
uint clusterID  = uint(vis >> 32);
uint triangleID = uint(vis & 0xFFFFFFFF);

float3 bary = ComputeBarycentrics(clusterID, triangleID, pixel);
Material m  = LoadClusterMaterial(clusterID);
float4 color = EvaluateMaterial(m, bary);
```

The material pass is organised as a *classify-then-shade* pipeline to recover $"GPU"$ coherence:

+ A *classify pass* emits one draw call per unique material ID present in the visibility buffer (using a `DrawIndirect` argument buffer built by a compute pass over the visibility data).
+ Each draw's vertex shader emits a full-screen triangle clipped to only the pixels owned by that material — the depth test discards everything else.
+ The pixel shader reconstructs interpolated attributes (UVs, normals, tangents) from the stored barycentrics and cluster vertex data, then evaluates the material graph.

This avoids the alternative of a branchy uber-shader with per-pixel material dispatch, which thrashes the warp execution unit on divergent material code paths. The classify step costs a single compute pass; the benefit is near-coherent execution within each material draw.

*Limitations:* Nanite does not natively support vertex-animated meshes (skeletal, cloth), alpha-masked geometry, or custom vertex factories that modify positions post-projection — these fall back to traditional rendering. In Unreal Engine 5.3+, a *programmable rasterization* path extends Nanite to masked materials by running the pixel shader during rasterization and discarding via `clip()`.

== Lumen: Engine Integration

At the engine level *Lumen* is orchestrated as several render passes inserted into the frame graph after the G-buffer / Nanite visibility pass.

=== Surface Cache Update

Lumen maintains a *surface cache atlas* — a set of textures that cache the shaded radiance of mesh surfaces as seen from probes. Each frame a budget of *surface cache pages* is identified as stale (camera moved, light changed) and re-shaded. The atlas acts as a persistent $"GI"$ source; $"SDF"$ marching rays shade against atlas texels rather than invoking full material evaluation.

=== Integration with Screen-Space Effects

The Lumen diffuse $"GI"$ pass runs after base pass at half or quarter resolution, then the result is *upsampled* (spatial bicubic + temporal accumulation) and composited into the lit buffer. Specular reflections from Lumen use a *reflection capture* fallback for non-$"RT"$ hardware, blended with screen-space reflections for nearby geometry.

== Temporal Anti-Aliasing and $"TSR"$

*$"TAA"$* (Temporal Anti-Aliasing) sub-pixel jitters the camera matrix each frame by a Halton-sequence offset in $[-0.5, 0.5]$ pixels. The history buffer is reprojected using per-pixel motion vectors:

$ x^("hist") = x - v(x), $

and the current sample is blended with an $alpha$ factor ($alpha approx 0.1$–$0.2$):

$ C^("out") = alpha C^("cur") + (1 - alpha) C^("hist"). $

*Neighborhood clamping* clips the history sample to the color $"AABB"$ of the current pixel's $3 times 3$ or $5 times 5$ neighborhood to suppress ghosting. Variance-clamping (Salvi 2016) uses the mean $mu$ and standard deviation $sigma$ of the neighborhood: clamp $C^("hist")$ to $[mu - k sigma, mu + k sigma]$ with $k approx 1.25$.

=== $"TSR"$: Temporal Super Resolution

Unreal 5's *$"TSR"$* extends $"TAA"$ to also upscale from a lower render resolution:

- Each frame renders at $e.g.$ $50\%$ linear scale (Nanite adapts the cluster cut error threshold to match).
- A *history reconstruction* pass applies an $"MLP"$-informed filter or explicit $8 times 8$ reconstruction kernel to upsample history into full-res output.
- The *anti-aliasing* pass blends the upsampled history with the current jittered frame using per-pixel confidence weights.

$"TSR"$ competes with $"DLSS"$ 3/$"FSR"$ 3 on quality while requiring no vendor-specific $"ML"$ model.

== Frame Graph / Render Graph

A *render graph* (also *frame graph*, Halcyon, Frostbite) is a data structure that records all render passes, their resource reads/writes, and their interdependencies before any $"GPU"$ work is submitted. This enables the engine to:

- *Automatically place barriers*: infer the exact `VkImageMemoryBarrier` / `D3D12_RESOURCE_BARRIER` required between producer and consumer passes without manual bookkeeping.
- *Alias transient resources*: reuse $"VRAM"$ heaps between passes whose lifetimes do not overlap, often saving $20$–$40\%$ $"VRAM"$.
- *Schedule async compute*: identify passes with no data dependency on the graphics queue and dispatch them on the async compute queue to overlap with graphics work.

```cpp
// Render graph registration (pseudocode, Vulkan-style)
RDG_Pass* shadowPass = graph.AddPass("Shadows", PassType::Graphics);
RDGTexture* shadowMap = graph.CreateTexture("ShadowMap", desc);
shadowPass->Write(shadowMap, AccessFlags::DepthWrite);

RDG_Pass* lightPass = graph.AddPass("Lighting", PassType::Graphics);
lightPass->Read(shadowMap, AccessFlags::ShaderRead);
lightPass->Write(hdrBuffer, AccessFlags::ColorWrite);

graph.Compile();  // topological sort, barrier insertion, aliasing
graph.Execute(cmdBuffer);
```

=== Explicit Barriers

Explicit barrier $"API"$s (Vulkan `vkCmdPipelineBarrier`, D3D12 `ResourceBarrier`) require the programmer to specify source and destination pipeline stages and access masks. Render graphs automate this: each resource edge in the $"DAG"$ maps to exactly one barrier, placed at the boundary of the producer and consumer passes, minimizing pipeline stalls.

=== Async Compute

Passes declared on the async compute queue run on $"CUs"$ / $"SMs"$ not claimed by graphics. Classic examples:

- Shadow map rendering (previous frame's) while the current frame's G-buffer writes.
- $"SSAO"$ / Lumen $"SDF"$ tracing while hardware rasterizes the next batch of clusters.
- Skinning / particle simulation compute while the rasterizer is bandwidth-bound.

Overlap requires careful semaphore signaling: the async compute queue signals a timeline semaphore on completion; the graphics queue waits before reading async-compute results.

== Engine Loop and Frame Pacing

A simplified engine loop on a $"PC"$ with a separate render thread:

#table(
  columns: 3,
  [*Thread*], [*Frame $N$ work*], [*Wall-clock budget*],
  [$"CPU"$ Game thread], [Simulate, animation, $"AI"$, spawn], [$approx 4$ $"ms"$ at 60 $"Hz"$],
  [$"CPU"$ Render thread], [Build draw lists, update $"SBT"$, push render graph], [$approx 4$ $"ms"$],
  [$"GPU"$], [Execute render graph, display], [$approx 16.6$ $"ms"$ at 60 $"Hz"$],
)

Triple-buffering decouples the game thread from $"GPU"$ execution by one frame, trading $approx 30$ $"ms"$ of input latency for smooth pipelining. NVIDIA Reflex and AMD Anti-Lag dynamically tighten the pipeline for competitive scenarios.

=== Frame Pacing and $"VRR"$

Variable Refresh Rate ($"VRR"$, $"G-Sync"$/$"FreeSync"$) lets the display stretch or compress the scan-out period to match actual $"GPU"$ frame time, eliminating tearing without fixed $"VSync"$ stutter. The swap chain present mode `VK_PRESENT_MODE_MAILBOX_KHR` (mail-box, no tearing, drops old frames) or `VK_PRESENT_MODE_FIFO_RELAXED_KHR` (relaxed $"FIFO"$) suits $"VRR"$ monitors.

== Asset Streaming

Scenes that exceed $"VRAM"$ budget require *streaming*: assets are loaded from disk into system $"RAM"$, then uploaded to $"VRAM"$ on demand, and evicted when no longer visible.

=== Nanite Streaming

Nanite cluster $"DAG"$ data is resident as a streaming pool. Coarse $"LOD"$ clusters are always resident; fine-detail clusters are streamed in as the camera approaches. The streaming manager tracks a *request priority* (projected screen error of the finest resident $"LOD"$) per cluster group and issues $"DMA"$ uploads on a background thread.

=== Texture Streaming

Texture streaming assigns each mip level a *residency priority* based on the maximum $"UV"$ derivative seen in the previous frame (the same $lambda$ used for $"LOD"$ selection). Levels below the streaming mip threshold are evicted; levels above are uploaded from a virtual texture page table. Virtual texturing ($"SVT"$, Sparse Virtual Textures) and DirectStorage ($"GPU"$ decompression $"DMA"$) are the dominant mechanisms on current-generation hardware.

```cpp
// DirectStorage enqueue example (Windows)
IDStorageQueue* queue;
factory->CreateQueue(&queueDesc, IID_PPV_ARGS(&queue));
DSTORAGE_REQUEST req = {};
req.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
req.Source.File.Source = file;
req.Source.File.Offset = clusterDataOffset;
req.Source.File.Size   = clusterDataSize;
req.Destination.Type   = DSTORAGE_REQUEST_DESTINATION_BUFFER;
req.Destination.Buffer.Resource = gpuBuffer;
req.Destination.Buffer.Offset  = 0;
req.Destination.Buffer.Size    = clusterDataSize;
queue->EnqueueRequest(&req);
queue->Submit();
```

== Further Reading

Karis, B. et al. (2021). "Nanite — A Deep Dive." Advances in Real-Time Rendering, SIGGRAPH.

Halcyon Architecture: de Vries, J. (2017). "FrameGraph: Extensible Rendering Architecture in Frostbite." GDC.

Akenine-Möller, T., Haines, E., Hoffman, N. et al. (2018). _Real-Time Rendering_, 4th ed. CRC Press — chapters 20 and 26.

Salvi, M. (2016). "An Excursion in Temporal Supersampling." GDC.

Epic Games (2022). "Temporal Super Resolution in Unreal Engine 5." Unreal Engine documentation.

Microsoft (2022). "DirectStorage API Overview." Microsoft Learn.

Wihlidal, A. (2017). "Optimizing the Graphics Pipeline with Compute." GDC / FROSTBITE.

= Shaders and Shading Languages

Shaders are the programmable spine of every modern $"GPU"$ pipeline. They evolved from fixed-function combiners in the late 1990s into Turing-complete $"SIMT"$ programs that today drive rasterization, ray tracing, $"ML"$ inference, and general compute. Understanding $"HLSL"$, $"GLSL"$, and the underlying execution model (warps, divergence, occupancy) is mandatory before tackling $"PBR"$ or global illumination.

*See also:* _The Rasterization Pipeline_ (where shaders attach), _Ray Tracing_ (RT shader stages), _Physically Based Rendering_ (BRDF shader code), _SIMT Execution Model_ (gpu-architecture framing; warps, divergence), _CUDA Programming Model_ (gpu-architecture framing; compute analog).

== Shader Languages: a Comparison

#table(
  columns: 4,
  [*Language*], [*Source $arrow.r$ $"IR"$*], [*Targets*], [*Notable*],
  [$"HLSL"$ (DX)], [DXC $arrow.r$ $"DXIL"$ (LLVM-IR)], [D3D12, Vulkan via SPIRV-Cross], [Default in industry; supports $"SM"$ 6.x],
  [$"GLSL"$], [glslang $arrow.r$ $"SPIR-V"$], [Vulkan, OpenGL], [Layout qualifiers, descriptor sets],
  [Metal Shading Language], [Metal compiler $arrow.r$ $"AIR"$], [iOS/macOS], [C++14-based, argument buffers],
  [$"WGSL"$], [Naga $arrow.r$ $"SPIR-V"$/MSL/$"HLSL"$], [WebGPU], [Safety-first, no UB],
  [Slang], [Slang $arrow.r$ $"SPIR-V"$/$"DXIL"$], [All], [Generics, modules, autodiff],
)

Slang in particular has become the language of choice for new research because of its first-class generics and reverse-mode autodiff (used in differentiable rendering, see _Real-Time Engine Architecture_).

== A Minimal Forward Pipeline

A standard textured Lambertian object in $"HLSL"$:

```hlsl
// vertex.hlsl
cbuffer Frame : register(b0) { float4x4 viewProj; };
cbuffer Object : register(b1) { float4x4 model;    };

struct VSIn  { float3 pos : POSITION; float3 nrm : NORMAL;
               float2 uv  : TEXCOORD0; };
struct VSOut { float4 svPos : SV_Position;
               float3 wNrm  : NORMAL;
               float2 uv    : TEXCOORD0; };

VSOut main(VSIn i) {
    VSOut o;
    float4 wp = mul(model, float4(i.pos, 1.0));
    o.svPos = mul(viewProj, wp);
    o.wNrm  = normalize(mul((float3x3)model, i.nrm));
    o.uv    = i.uv;
    return o;
}
```

```hlsl
// pixel.hlsl
Texture2D    albedo   : register(t0);
SamplerState linearSm : register(s0);
cbuffer Light : register(b2) { float3 lDir; float3 lCol; };

float4 main(float3 wNrm : NORMAL, float2 uv : TEXCOORD0) : SV_Target {
    float3 N    = normalize(wNrm);
    float  NdL  = saturate(dot(N, -lDir));
    float3 base = albedo.Sample(linearSm, uv).rgb;
    return float4(base * lCol * NdL, 1.0);
}
```

The equivalent $"GLSL"$ trades `register(bN)` for `layout(set=, binding=)`:

```glsl
#version 460
layout(set=0, binding=0) uniform Frame { mat4 viewProj; };
layout(set=1, binding=0) uniform Object { mat4 model; };
layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inNrm;
layout(location=2) in vec2 inUv;
layout(location=0) out vec3 wNrm;
layout(location=1) out vec2 uv;
void main() {
    vec4 wp = model * vec4(inPos, 1.0);
    gl_Position = viewProj * wp;
    wNrm = normalize(mat3(model) * inNrm);
    uv = inUv;
}
```

== Execution Model: Warps, Lanes, Divergence

A shader instance is one *lane* inside a warp (NVIDIA, 32 lanes) or wavefront (AMD RDNA, 32; older $"GCN"$, 64). All lanes execute the same instruction; conditionals that diverge are masked, costing throughput. The "quad" is a 2$times$2 group of pixel-shader lanes whose `ddx`/`ddy` derivatives are well-defined.

```hlsl
// Divergent: lanes within a warp take different paths.
if (uv.x > 0.5) { result = expensiveA(uv); }
else            { result = expensiveB(uv); }
// Better: select, both branches evaluated but no divergence stall when
// one branch is cheap.
result = lerp(expensiveB(uv), expensiveA(uv), uv.x > 0.5 ? 1 : 0);
```

Wave intrinsics (Shader Model 6.0, $"GL_KHR_shader_subgroup"$) expose intra-warp cooperation:

```hlsl
// Sum a value across all active lanes (subgroup reduction).
float total = WaveActiveSum(myValue);
uint  leader = WaveReadLaneFirst(myIndex);
bool  allOn  = WaveActiveAllTrue(insideFrustum);
```

Used in tile-based light culling, hierarchical Z generation, and Nanite cluster culling.

== Resource Binding Models

#table(
  columns: 3,
  [*Model*], [*$"API"$*], [*Notes*],
  [Slot-based], [D3D11, OpenGL legacy], [Fixed t0/s0/b0 slots, low limit],
  [Descriptor sets], [Vulkan, D3D12], [Group resources, swap a set per material],
  [Bindless], [Vulkan 1.2 DescriptorIndexing, $"SM"$ 6.6], [Index huge arrays by integer],
  [Argument buffers], [Metal], [GPU-side resource tables],
)

Bindless is now the default in AAA engines: a single descriptor table holds thousands of textures, and shaders dereference by integer index in a uniform/storage buffer. This essentially eliminates draw-call-per-material binding overhead.

```hlsl
// SM 6.6 dynamic resources (bindless).
Texture2D    bindlessTex[] : register(t0, space0);
SamplerState bindlessSmp[] : register(s0, space0);
float4 sampleMat(uint texId, float2 uv) {
    return bindlessTex[NonUniformResourceIndex(texId)]
           .Sample(bindlessSmp[0], uv);
}
```

`NonUniformResourceIndex` is mandatory when the index varies per lane — the compiler emits a loop until the index is uniform across the wave.

== Compute Shaders

A compute shader has no implicit attachments; it dispatches a 3D grid of thread groups. Used for post-processing, particle sims, $"GI"$ probe updates, hair simulation, and increasingly for tasks formerly done on the $"CPU"$.

```hlsl
RWTexture2D<float4> dst;
Texture2D<float4>   src;

[numthreads(8, 8, 1)]
void blurH(uint3 tid : SV_DispatchThreadID) {
    float4 acc = 0; const int R = 4;
    for (int dx = -R; dx <= R; ++dx)
        acc += src.Load(int3(tid.x + dx, tid.y, 0));
    dst[tid.xy] = acc / (2.0*R + 1.0);
}
```

Group shared memory (`groupshared` in $"HLSL"$, `shared` in $"GLSL"$) maps to the $"SM"$'s $"L1"$/shared and enables data reuse — the moral equivalent of $"CUDA"$ shared memory tiling.

== Tessellation, Geometry, Mesh

The classic tessellation pipeline (hull, tessellator, domain) is largely deprecated outside terrain/water. Geometry shaders are slow on every $"GPU"$ because they serialize per-primitive amplification. Both have been superseded by *mesh shaders* (covered in the rasterization chapter).

== Ray-Tracing Shaders

DXR / Vulkan-RT add five new shader stages: ray generation, miss, closest-hit, any-hit, intersection. See _Ray Tracing_.

== Common Bugs

- *Sampler bias missing*: sampling textures inside divergent control flow without manually providing $"LOD"$ derivatives (`tex.Sample` requires uniform control flow; use `tex.SampleLevel` or `tex.SampleGrad`).
- *Non-uniform indices into resource arrays without* `NonUniformResourceIndex`: silently undefined.
- *Forgetting `precise`*: aggressive optimizer reorders FMA and breaks Z-fighting comparisons.
- *Half precision overflow*: HDR pixel values often exceed $65504$.
- *Atomic to UAV without `globallycoherent`* on $"DX"$ — value invisible to other groups.

== Shader Compilation Pipeline

The modern flow is two-stage: offline compile $"HLSL"$/$"GLSL"$ to $"SPIR-V"$ / $"DXIL"$ (a stable $"IR"$), then runtime compile $"IR"$ to vendor $"ISA"$ ($"GCN"$, $"PTX"$, $"AGX"$). Pipeline state objects (PSOs) wrap this; Vulkan VK_EXT_shader_object and PSO caches reduce hitches.

```cpp
VkPipelineShaderStageCreateInfo stages[2] = {
    { .stage = VK_SHADER_STAGE_VERTEX_BIT,   .module = vs, .pName = "main" },
    { .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
};
VkGraphicsPipelineCreateInfo gp{
    .stageCount = 2, .pStages = stages,
    .pVertexInputState   = &vis, .pInputAssemblyState = &ias,
    .pRasterizationState = &rs,  .pMultisampleState   = &ms,
    .pDepthStencilState  = &ds,  .pColorBlendState    = &cbs,
    .layout = layout, .renderPass = pass,
};
vkCreateGraphicsPipelines(dev, cache, 1, &gp, nullptr, &pipe);
```

PSO compilation is *slow* (10–500 ms each), which is why engines precompile and cache them or use background compilation threads with a fallback shader.

== Debugging and Validation

- *RenderDoc*: per-draw inspection, pixel history, shader edit-and-continue.
- *Nsight Graphics* / *Radeon GPU Profiler*: timing per warp, occupancy, $"ALU"$/$"MEM"$ ratios.
- *Vulkan Validation Layers*: descriptor-state errors, $"UB"$ in pipeline layout.
- *PIX on Windows*: $"GPU"$ captures with timing markers.

== Further Reading

Akenine-Möller, T. et al. (2018). _Real-Time Rendering_, 4th ed., chapters 3, 5, 18.

Microsoft (2025). "$"HLSL"$ Shader Model 6.8 Specification."

Khronos Group (2024). "$"GLSL"$ 4.60 and SPIR-V Specifications."

Olano, M., Lastra, A. (1998). "A Shading Language on Graphics Hardware: The PixelFlow Shading System." SIGGRAPH.

Bavoil, L., Sainz, M. (2009). "Multi-Layer Dual-Resolution Screen-Space Ambient Occlusion." NVIDIA white paper (shows compute-shader techniques).

He, Y. et al. (2018). "Slang: language mechanisms for extensible real-time shading systems." SIGGRAPH.

#import "../template.typ": xref

= Shadows and Occlusion <shadows>

Shadows are a visibility query: can this point see the light? In an offline path tracer the answer comes from a ray, and the only problem is cost. In real-time rendering, shadows have been the longest-running source of artefacts and clever approximations: jagged edges, acne, floating objects, shimmering as the camera moves, and soft shadows that are too soft or not soft enough. This chapter covers shadow mapping and its failure modes, cascades for the sun, soft-shadow filtering (PCF, PCSS and statistical shadow maps), virtual shadow maps, ray-traced shadows, and ambient occlusion, the shadowing of indirect light.

*See also:* #xref("graphics-and-rendering", "rasterization-pipeline", label: "The Rasterization Pipeline"), #xref("graphics-and-rendering", "ray-tracing", label: "Ray Tracing"), #xref("graphics-and-rendering", "gi-techniques", label: "Global Illumination Techniques"), #xref("graphics-and-rendering", "transforms-and-cameras", label: "Transforms, Projections, and Cameras") (light-space projections)

== What a Shadow Is

A point light casts *hard* shadows: each point is either visible to the light or not. Real lights have area, so a point can see part of the light, and the shadow has an *umbra* (no part of the light visible) surrounded by a *penumbra* (partly visible). The penumbra's width grows with the light's size and with the distance between the occluder and the receiver. Shadows are therefore sharp where an object touches the ground and soften with distance, a *contact-hardening* effect that the eye uses to judge whether objects rest on surfaces or float above them. Getting that cue right matters more for realism than the exact softness.

== Shadow Mapping

*Shadow mapping* (Williams, 1978) is the workhorse of real-time shadows:

1. Render the scene from the light's point of view, storing only depth: the *shadow map*.
2. When shading a point, transform it into the light's clip space, look up the stored depth at its position, and compare. If the point is farther from the light than the stored depth, something is in between and the point is in shadow.

It is simple, works with any geometry the rasteriser can draw, and costs one extra depth-only pass per light. Its problems all come from comparing two independently sampled depth fields.

=== Shadow Acne and Bias

The shadow map stores one depth per texel, but each texel covers an area of surface. For a surface tilted relative to the light, the stored depth at the texel centre is less than the true depth at many points the texel covers, so the surface shadows itself in a moiré pattern of stripes called *shadow acne*. The fix is a *bias* that pushes the comparison in favour of "lit":

- *Constant bias*: subtract a small amount from the receiver depth. Too little leaves acne; too much detaches shadows from their casters (*peter-panning*).
- *Slope-scaled bias*: scale the bias by the surface slope relative to the light, which hardware supports directly in the rasteriser state when rendering the shadow map.
- *Normal-offset bias*: move the lookup position along the surface normal by an amount proportional to the texel size. It handles grazing angles better than depth bias alone and is the usual choice today.

Rendering only back faces into the shadow map (for closed meshes) moves the stored depth to the far side of each object, which removes acne on lit surfaces at the cost of light leaking through thin geometry.

=== Aliasing

Shadow-map texels project onto the screen at varying sizes. *Perspective aliasing* occurs because near the camera one shadow texel covers many screen pixels, giving blocky edges. *Projective aliasing* occurs where the light's rays graze a surface, stretching texels across it. Resolution alone cannot fix both: the required density varies by orders of magnitude across the view. The rest of this chapter is largely about redistributing texels to where the camera needs them.

== Directional Lights: Cascaded Shadow Maps

The sun illuminates the whole visible scene, from objects at the player's feet to mountains kilometres away. One shadow map covering all of it gives a few texels per metre near the camera. *Cascaded shadow maps* (CSM) split the view frustum by distance into 3 to 5 slices, each with its own orthographic shadow map tightly fitted to that slice. Near cascades cover a few metres at high density; far cascades cover kilometres coarsely.

The split distances matter. Uniform splits waste resolution far away; logarithmic splits match perspective aliasing in theory but make the first cascade too small in practice. The *practical split scheme* (Zhang et al., 2006) blends the two:

$ c_i = lambda n (f / n)^(i \/ N) + (1 - lambda) (n + (f - n) i / N) $

with $lambda$ around 0.5 to 0.9.

```cpp
#include <array>
#include <cmath>

// Far distance of each cascade for the practical split scheme.
// near_plane/far_plane: the view range to shadow (often far less than the
// camera far plane); blend = lambda: 0 uniform, 1 logarithmic.
template <std::size_t N>
std::array<float, N> cascade_splits(float near_plane, float far_plane, float blend) {
    std::array<float, N> splits{};
    for (std::size_t i = 1; i <= N; ++i) {
        const float p = static_cast<float>(i) / static_cast<float>(N);
        const float log_split = near_plane * std::pow(far_plane / near_plane, p);
        const float uniform_split = near_plane + (far_plane - near_plane) * p;
        splits[i - 1] = blend * log_split + (1.0f - blend) * uniform_split;
    }
    return splits;
}

// cascade_splits<4>(0.1f, 400.0f, 0.9f) gives about {10.7, 25.7, 75.3, 400} metres.
```

Two refinements are standard:

- *Stabilisation*: as the camera moves or rotates, a tightly fitted cascade changes size and position, so shadow edges crawl and shimmer. Fitting each cascade to a *bounding sphere* of its frustum slice (so its size does not change with rotation) and *snapping* its position to whole shadow-map texels removes the shimmer, at the cost of some resolution.
- *Blending between cascades* over a short band hides the visible change in quality at the boundaries.

Shadow distance is a key cost control: the far cascades are where most shadow-casting geometry is, and engines often render them less frequently (every other frame) or cache them.

== Point and Spot Lights

A spot light uses a single perspective shadow map. A point light needs a *cube map*: six $90$-degree perspective renders, which multiplies shadow cost by six. Engines limit how many point lights cast shadows, render each face only when something inside it moved, and cache static geometry's depth separately from dynamic objects. Dual-paraboloid maps cover a point light with two renders instead of six but distort straight edges and are rarely used now.

== Soft Shadows

=== Percentage-Closer Filtering

Bilinearly filtering a depth texture would average depths, which is meaningless. *Percentage-closer filtering* (PCF; Reeves, Salesin and Cook, 1987) instead performs the depth *comparison* at several nearby texels and averages the binary results, giving the fraction of samples that are lit. Hardware supports a $2 times 2$ PCF in one fetch through comparison samplers; larger kernels use multiple fetches, often with a rotated Poisson disc or a randomised pattern resolved by temporal accumulation. Fixed-size PCF softens edges uniformly, which hides aliasing but gives no contact hardening.

=== Percentage-Closer Soft Shadows

*PCSS* (Fernando, 2005) estimates the penumbra width per pixel from similar triangles:

1. *Blocker search*: average the depths of shadow-map texels in a search region that are closer to the light than the receiver.
2. *Penumbra estimate*: $w_"penumbra" = (d_"receiver" - d_"blocker") \/ d_"blocker" dot w_"light"$.
3. *Filter*: run PCF with a kernel of that width.

Shadows sharpen at contact and soften with distance, as real shadows do. The cost is variable and can be high for large lights and wide penumbrae, and the blocker search is an approximation that fails when several occluders overlap at different depths.

=== Statistical Shadow Maps

*Variance shadow maps* (VSM; Donnelly and Lauritzen, 2006) store the first two moments of depth, $E[z]$ and $E[z^2]$, which *can* be filtered with ordinary mipmapping, anisotropic filtering and blurs. At lookup, Chebyshev's inequality bounds the fraction of the filter region that is lit:

$ P(z >= t) <= sigma^2 / (sigma^2 + (t - mu)^2), quad sigma^2 = E[z^2] - mu^2 $

for receiver depth $t > mu$. Filtering becomes cheap and well behaved, but where several occluders at different depths overlap, the bound is loose and light *bleeds* into shadows. Exponential shadow maps, their combination with VSM (EVSM) and moment shadow maps (four moments) reduce bleeding at the cost of more storage and precision.

== Virtual Shadow Maps

Cascades are a coarse way of matching shadow resolution to screen needs. *Virtual shadow maps*, as used in Unreal Engine 5, take the idea to its limit: each light has a very large virtual shadow map (16 384 by 16 384 texels per cascade or cube face), divided into pages. Each frame the engine marks only the pages that visible pixels actually sample at the resolution they need, renders those pages, and caches them across frames, re-rendering only pages invalidated by moving geometry. Combined with virtualised geometry that can render many small pages efficiently, this gives roughly one shadow texel per screen pixel everywhere in view. It is the shadow equivalent of virtual texturing (#xref("graphics-and-rendering", "textures-and-sampling", label: "Textures, Sampling, and Aliasing")).

== Contact Shadows

Even good shadow maps miss small-scale occlusion: the shadow under a character's shoe, between fingers, in fine geometric detail. *Screen-space contact shadows* march a short ray from each pixel towards the light through the depth buffer, and mark the pixel as shadowed if the ray passes behind a depth sample. They are cheap and fill in exactly the contact detail that the eye uses, but fail for occluders that are off screen or hidden behind other surfaces, so they complement shadow maps rather than replacing them.

== Ray-Traced Shadows

With hardware ray tracing, a shadow is a ray from the shading point towards the light that only needs to know whether it hits anything (any-hit, terminated on the first hit). There is no bias tuning, no resolution limit, and no aliasing beyond the screen's own. Soft shadows come from sampling points on the area light: each pixel traces one or a few rays per frame to random points on the light, and a *denoiser* reconstructs a smooth penumbra from the noisy result, using the distance to the blocker to set the filter width. ReSTIR-style resampling (#xref("graphics-and-rendering", "gi-techniques", label: "Global Illumination Techniques")) makes scenes with thousands of shadow-casting lights practical by choosing, per pixel, which few lights deserve a ray.

The trade-offs are cost (especially with alpha-tested foliage, whose any-hit shaders are expensive), the need to maintain an acceleration structure for all shadow casters, and denoiser artefacts in motion. Many engines combine techniques: ray-traced shadows for the sun near the camera, shadow maps in the distance, and contact shadows for fine detail.

== Shadow Volumes

Before shadow maps dominated, *shadow volumes* (Crow, 1977) extruded each occluder's silhouette away from the light and used the stencil buffer to count, per pixel, how many volumes the view ray entered and left. The result is pixel-exact hard shadows with no aliasing, used famously in Doom 3 with the "depth-fail" variant that handles a camera inside a volume. Their cost scales with silhouette complexity and the fill rate of large volumes, and they cannot produce soft shadows cheaply, so they survive mainly as a historical lesson and in special cases.

== Ambient Occlusion

Direct shadows block light from specific sources. *Ambient occlusion* (AO) approximates how much of the surrounding hemisphere is blocked for indirect light, darkening creases, corners and contact areas. It stands in for the missing occlusion in approximate global illumination such as irradiance probes.

- *SSAO* samples the depth buffer around each pixel and counts how many samples lie above the surface. It is cheap but noisy and only sees what is on screen.
- *HBAO* and *GTAO* (ground-truth AO; Jimenez et al., 2016) march in several directions across the depth buffer to find the horizon angle, integrating visibility analytically over the visible part of the hemisphere. GTAO matches ray-traced references closely for on-screen geometry and can also produce a bent normal and specular occlusion.
- *Ray-traced AO* casts short rays into the hemisphere, handling off-screen occluders correctly, and is denoised like ray-traced shadows.

AO should affect only indirect lighting. Applying it to direct light as well double-counts shadows and gives the dirty, over-darkened look common in games of the early 2010s.

== Further Reading

Williams, L. (1978). "Casting Curved Shadows on Curved Surfaces." SIGGRAPH.

Crow, F. (1977). "Shadow Algorithms for Computer Graphics." SIGGRAPH.

Reeves, W., Salesin, D., Cook, R. (1987). "Rendering Antialiased Shadows with Depth Maps." SIGGRAPH.

Fernando, R. (2005). "Percentage-Closer Soft Shadows." SIGGRAPH Sketches.

Donnelly, W., Lauritzen, A. (2006). "Variance Shadow Maps." Symposium on Interactive 3D Graphics and Games.

Zhang, F., Sun, H., Xu, L., Lun, L. K. (2006). "Parallel-Split Shadow Maps for Large-Scale Virtual Environments." ACM VRCIA.

Eisemann, E., Schwarz, M., Assarsson, U., Wimmer, M. (2011). "Real-Time Shadows." CRC Press.

Jimenez, J., Wu, X., Pesce, A., Jarabo, A. (2016). "Practical Real-Time Strategies for Accurate Indirect Occlusion." SIGGRAPH Physically Based Shading course.

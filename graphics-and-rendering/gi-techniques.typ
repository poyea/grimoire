= Global Illumination Techniques

Global illumination ($"GI"$) accounts for indirect light — photons that bounce one or more times before reaching the camera. The gap between a rasterized local-illumination model and a physically accurate $"GI"$ solution is precisely the gap between a flat studio render and a scene that looks like it belongs in its environment.

*See also:* _Physically Based Rendering_ (rendering equation, BRDF theory), _Ray Tracing_ (BVH, DXR shader stages, denoising), _Real-Time Engine Architecture_ (Lumen engine integration, DDGI probes), _Compute Units and Specialized Cores_ (gpu-architecture framing; wave-level intrinsics).

== Classical Methods

=== Radiosity

*Radiosity* (Cohen & Greenberg 1985) models diffuse-to-diffuse light transport by discretizing surfaces into *patches* and solving the linear system

$ B_i = E_i + rho_i sum_j F_(i j) B_j, $

where $B_i$ is the radiosity (power per area) of patch $i$, $E_i$ its emission, $rho_i$ its diffuse reflectivity, and $F_(i j)$ the *form factor* encoding the geometric relationship between patches $i$ and $j$:

$ F_(i j) = 1/A_i integral_(A_i) integral_(A_j) (cos theta_i cos theta_j) / (pi r^2) V(i,j) d A_j d A_i. $

The system is $O(N^2)$ in patches. Progressive refinement solves it iteratively. Radiosity was used in architectural pre-visualization in the 1990s but scales poorly and handles only Lambertian surfaces.

=== Irradiance Caching

Ward et al. (1988) cache irradiance at sparsely sampled surface points and interpolate using a validity radius derived from the scene curvature. An *irradiance cache* record at position $x$ with normal $n$ stores the estimated indirect irradiance $E(x)$; nearby query points with compatible geometry interpolate from existing records and insert a new record only on a cache miss. This reduces Monte Carlo sample counts by $100 times$ or more for slowly-varying indirect light.

=== Photon Mapping

Jensen (1996) traces photons from light sources into the scene and deposits them in a *photon map* (a $k$-d tree of positions, incident directions, and power). A *radiance estimate* at a shading point $x$ queries the $k$ nearest photons within radius $r$:

$ L_r (x, omega_o) approx sum_(p in Phi_k) f_r (x, omega_p, omega_o) Delta Phi_p / (pi r^2). $

*Caustics photon maps* capture focused specular-to-diffuse transport (light through a glass) that path tracing finds only by luck. *Final gather* from an irradiance cache is combined with a coarse global photon map for indirect diffuse.

== Monte Carlo Path Tracing

Path tracing (Kajiya 1986) evaluates the rendering equation by recursively sampling $omega_i$ from some density $p(omega_i)$ and computing the estimator

$ hat(L)_o = f_r (x, omega_i, omega_o) L_i (x, omega_i) (omega_i dot n) / p(omega_i). $

=== Importance Sampling

Choose $p prop f_r cos$ to minimize variance. For $"GGX"$ the $"VNDF"$ (visible normal distribution function, Heitz 2018) samples half-vectors consistent with $D(h)$ weighted by the masking term, reducing variance $2$–$4 times$ over cosine sampling:

$ p_("VNDF") (m) = (G_1 (omega_o) max(0, omega_o dot m) D(m)) / (omega_o dot n). $

=== Multiple Importance Sampling ($"MIS"$)

When both a $"BRDF"$ sample and a light sample are available, *$"MIS"$* (Veach & Guibas 1995) combines them with balance or power heuristic weights. For two sampling strategies with densities $p_1$, $p_2$:

$ hat(L)^("MIS") = w_1(omega) (f_r cos) / p_1 + w_2(omega) (f_r cos) / p_2, quad w_k = (n_k p_k) / (n_1 p_1 + n_2 p_2). $

$"MIS"$ eliminates the blow-up that occurs when a $"BRDF"$-sampled ray happens to hit a bright area light, or a light-sampled ray intersects a near-mirror surface.

=== Next-Event Estimation

*Next-event estimation* ($"NEE"$) explicitly samples a point $y$ on an emitter from every vertex:

$ hat(L)^("NEE") = f_r (x, omega_(y arrow.r x), omega_o) G(x, y) L_e (y, omega_(x arrow.r y)) / p_("light") (y), $

where $G(x, y) = V(x,y) (cos theta_x cos theta_y) \/ ||x - y||^2$ is the geometry term. Combined with $"MIS"$, $"NEE"$ converts a high-variance single-bounce estimator into a practical 1–4 spp renderer.

== ReSTIR: Reservoir-Based Spatiotemporal Importance Resampling

Bitterli et al. (2020) observe that direct-lighting $"NEE"$ can evaluate the balance-heuristic weight of a candidate sample $x$ without tracing a ray, then accept or reject via *reservoir sampling*. A reservoir of size $M$ maintains a single selected sample $z$ with weight $w$ and a sum $W$ of all seen weights; inserting a new candidate $c$ accepts with probability $w_c \/ (W + w_c)$.

=== Spatial and Temporal Reuse

Each pixel's reservoir is temporally reprojected (using motion vectors) and combined with the previous frame's reservoir. Neighboring pixels within a radius then exchange reservoirs. After $K$ neighbors, the effective sample count per pixel grows to $O(K M)$ without additional rays. The combined estimator converges to the $"MIS"$-optimal distribution over the light source.

*ReSTIR $"GI"$* (Ouyang et al. 2021) extends the algorithm to the second bounce by treating the entire path from a hit point as the sample; spatial/temporal reuse of paths dramatically improves indirect diffuse quality at 1 spp.

```hlsl
// Reservoir update (pseudocode)
struct Reservoir { float3 sample; float w_sum; uint M; float W; };

void Update(inout Reservoir r, float3 x, float w, float rand) {
    r.w_sum += w;
    r.M     += 1;
    if (rand < w / r.w_sum) r.sample = x;
}

float Finalize(inout Reservoir r, float target_pdf) {
    r.W = (r.w_sum / max(target_pdf, 1e-6)) / float(r.M);
    return r.W;
}
```

== $"DDGI"$: Dynamic Diffuse Global Illumination

Majercik et al. (2019) place a 3D grid of *irradiance probes* throughout the scene. Each probe stores:
- An *irradiance* octahedral atlas (e.g. $8 times 8$ texels) — low-frequency $"SH"$-like cosine-weighted radiance.
- A *visibility* octahedral atlas — mean and mean-squared ray distance for Chebyshev visibility tests.

Each frame a fixed ray budget (e.g. 256 rays per probe) is traced from probe centers. Probe irradiance is updated via exponential moving average ($alpha approx 0.97$) for temporal stability. Shading queries the eight surrounding probes, blends using trilinear interpolation, and weights by the visibility atlas to suppress light leaking through thin walls.

=== RTXGI

NVIDIA's *$"RTXGI"$* SDK implements $"DDGI"$ with DXR, adds probe relocation (probes inside geometry move to the nearest empty cell) and probe classification (probes in unlit volumes are disabled). It ships in Unreal Engine 5 as an optional backend for Lumen.

== Lumen: Epic's Production $"GI"$

Unreal Engine 5's *Lumen* (Karis et al. 2021) targets 60 $"Hz"$ $"GI"$ on current-generation consoles without dedicated $"RT"$ cores.

=== Software Ray Tracing

Lumen traces rays against a *signed distance field* ($"SDF"$) mesh representation. Each mesh is voxelized into a $"SDF"$ volume; the scene $"SDF"$ is a sparse $"BVH"$ of per-object $"SDF"$s. Sphere marching advances along a ray until the accumulated minimum $"SDF"$ value drops below a threshold, then shades against a *surface cache* (atlas of per-object radiance). This sidesteps $"RT"$ core availability, though it trades accuracy for performance.

=== Screen-Space Fallback and Hierarchical Tracing

For nearby geometry, Lumen first traces screen space (cheaper, more accurate detail), then falls back to $"SDF"$ marching, then finally queries the probe grid. The hierarchy is:

#table(
  columns: 3,
  [*Layer*], [*Range*], [*Representation*],
  [Screen-space], [$< 3$ m], [Depth buffer reprojection],
  [$"SDF"$ traces], [$3$–$200$ m], [Per-mesh $"SDF"$ + surface cache],
  [Probe grid], [$> 200$ m], [Radiance cache probes],
)

Lumen also supports hardware $"RT"$ on consoles and PC at higher quality settings, where it replaces $"SDF"$ traces with $"BVH"$ intersection for the first bounce.

== Practical Comparison

#table(
  columns: 4,
  [*Technique*], [*Latency*], [*Dynamic scenes*], [*Accuracy*],
  [Radiosity], [Minutes–hours], [No (baked)], [Diffuse only],
  [Photon mapping], [Seconds–minutes], [Limited], [High (caustics)],
  [Path tracing], [$> 1$ s/frame], [Yes], [Unbiased],
  [ReSTIR (direct)], [$1$–$3$ $"ms"$], [Yes], [Near-reference],
  [$"DDGI"$ / $"RTXGI"$], [$2$–$5$ $"ms"$], [Yes], [Low-frequency $"GI"$],
  [Lumen ($"SDF"$)], [$4$–$8$ $"ms"$], [Yes], [Medium, no caustics],
)

== Further Reading

Kajiya, J. T. (1986). "The Rendering Equation." SIGGRAPH. Introduced the integral formulation of light transport that underlies all modern path tracing and GI algorithms.

Jensen, H. W. (1996). "Global Illumination using Photon Maps." Rendering Techniques (EGWR). Describes photon mapping: a two-pass algorithm storing photon hits in a kd-tree for density estimation.

Ward, G. J. et al. (1988). "A Ray Tracing Solution for Diffuse Interreflection." SIGGRAPH. Introduced irradiance caching — the first practical algorithm for glossy/diffuse interreflection at interactive scales.

Bitterli, B. et al. (2020). "Spatiotemporal Reservoir Resampling for Real-Time Ray Tracing with Dynamic Direct Lighting." SIGGRAPH. Introduces ReSTIR, now the standard algorithm for real-time denoised direct and indirect lighting on RTX hardware.

Majercik, Z. et al. (2019). "Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields." JCGT 8(2). The DDGI algorithm used in RTXGI SDK: probe-grid irradiance fields with real-time ray-traced updates.

Laine, S. et al. (2020). "Megakernels Considered Harmful: Wavefront Path Tracing on GPUs." HPG. Analyses why monolithic path-tracing kernels under-utilize GPUs and shows wavefront/stream compaction as the solution.

Pharr, M., Jakob, W., Humphreys, G. (2023). _Physically Based Rendering: From Theory to Implementation_, 4th ed. MIT Press. The authoritative textbook on light transport theory, Monte Carlo integration, and BVH-accelerated path tracing.

Christensen, P., Jarosz, W. (2016). "The Path to Path-Traced Movies." Foundations and Trends in Computer Graphics and Vision 10(2). Survey of production rendering evolution from Reyes to path tracing at Pixar/Disney.

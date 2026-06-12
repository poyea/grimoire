= Physically Based Rendering

Physically based rendering ($"PBR"$) replaces ad-hoc shading models (Phong, Blinn–Phong) with formulations grounded in the rendering equation and microfacet theory. The result is materials that respond plausibly under arbitrary lighting — a prerequisite for $"HDR"$ pipelines, image-based lighting, and any system mixing rasterization with path tracing.

*See also:* _Shaders and Shading Languages_ (BRDF implementation), _Ray Tracing_ (Monte Carlo evaluation), _Global Illumination Techniques_ (importance sampling reuse), _Real-Time Engine Architecture_ (Disney/Unreal material model).

== The Rendering Equation

Kajiya (1986) wrote outgoing radiance at point $x$ in direction $omega_o$ as

$ L_o (x, omega_o) = L_e (x, omega_o) + integral_(Omega^+) f_r (x, omega_i, omega_o) L_i (x, omega_i) (omega_i dot n) d omega_i. $

Here $f_r$ is the *bidirectional reflectance distribution function* ($"BRDF"$), $Omega^+$ is the upper hemisphere, and the cosine $omega_i dot n$ accounts for foreshortening. Three properties pin the space of physically valid $"BRDFs"$:

- *Non-negativity*: $f_r >= 0$.
- *Helmholtz reciprocity*: $f_r (omega_i, omega_o) = f_r (omega_o, omega_i)$.
- *Energy conservation*: $integral_(Omega^+) f_r (omega_i, omega_o) (omega_i dot n) d omega_i <= 1$ for all $omega_o$.

Phong with $k_d + k_s = 1$ obeys the first two but not the third (specular lobe shape depends on exponent without compensation).

== Microfacet Theory

A surface is modeled as countless mirror-like micro-facets with normals distributed by $D(h)$. Cook–Torrance derived the specular term

$ f_("spec") (omega_i, omega_o) = (D(h) thin F(omega_o, h) thin G(omega_i, omega_o, h)) / (4 thin (n dot omega_i) thin (n dot omega_o)), $

with half-vector $h = (omega_i + omega_o) \/ ||omega_i + omega_o||$, normal distribution $D$, Fresnel $F$, and geometry/shadowing-masking $G$.

=== Normal Distribution: $"GGX"$ / Trowbridge–Reitz

The de-facto standard since Disney's 2012 BRDF Explorer.

$ D_("GGX") (h) = alpha^2 / (pi ((n dot h)^2 (alpha^2 - 1) + 1)^2), quad alpha = "roughness"^2. $

$"GGX"$ has long tails — better matches real measured $"BRDFs"$ from $"MERL"$ than Beckmann or Phong.

=== Fresnel: Schlick

$ F(omega, h) approx F_0 + (1 - F_0)(1 - omega dot h)^5. $

For dielectrics $F_0 approx 0.04$; for metals $F_0$ is RGB (gold $approx (1.00, 0.71, 0.29)$, copper $approx (0.95, 0.64, 0.54)$). This split is why $"PBR"$ exposes a *metallic* parameter rather than authoring $F_0$ directly.

=== Smith Masking-Shadowing

$ G_2 (omega_i, omega_o) = G_1 (omega_i) thin G_1 (omega_o), quad G_1 (omega) = (2 (n dot omega)) / ((n dot omega) + sqrt(alpha^2 + (1 - alpha^2)(n dot omega)^2)). $

The height-correlated variant (Heitz 2014) is more accurate at high roughness; both are cheap enough for real-time.

== Diffuse: Lambert vs Disney

Lambert: $f_d = ("albedo") \/ pi$. Disney 2012 introduced a roughness-aware diffuse:

$ f_d^("Disney") = ("albedo") / pi (1 + (F_(D 90) - 1)(1 - omega_i dot n)^5)(1 + (F_(D 90) - 1)(1 - omega_o dot n)^5), $

with $F_(D 90) = 0.5 + 2 thin "rough" thin (h dot omega_i)^2$. Captures retroreflection on rough surfaces (cloth, sand). Frostbite later derived an energy-conserving variant.

== Reference $"HLSL"$ Implementation

```hlsl
float D_GGX(float NoH, float a) {
    float a2 = a * a;
    float f  = (NoH * a2 - NoH) * NoH + 1.0;
    return a2 / (3.14159265 * f * f);
}
float V_SmithGGXCorrelated(float NoV, float NoL, float a) {
    float a2 = a * a;
    float GGXV = NoL * sqrt(NoV * NoV * (1 - a2) + a2);
    float GGXL = NoV * sqrt(NoL * NoL * (1 - a2) + a2);
    return 0.5 / (GGXV + GGXL);
}
float3 F_Schlick(float u, float3 f0) {
    return f0 + (1.0 - f0) * pow(1.0 - u, 5.0);
}
float3 brdf(float3 N, float3 V, float3 L,
            float3 baseColor, float metallic, float roughness)
{
    float3 H = normalize(V + L);
    float NoV = max(dot(N, V), 1e-4);
    float NoL = max(dot(N, L), 0.0);
    float NoH = max(dot(N, H), 0.0);
    float VoH = max(dot(V, H), 0.0);
    float a   = roughness * roughness;
    float3 f0 = lerp(float3(0.04, 0.04, 0.04), baseColor, metallic);
    float  D  = D_GGX(NoH, a);
    float  Vs = V_SmithGGXCorrelated(NoV, NoL, a);
    float3 F  = F_Schlick(VoH, f0);
    float3 spec = D * Vs * F;
    float3 diff = (1.0 - metallic) * baseColor * (1.0 / 3.14159265);
    float3 kd   = (1.0 - F);
    return (kd * diff + spec) * NoL;
}
```

== Image-Based Lighting

$"IBL"$ replaces analytic lights with an environment map. Diffuse $"IBL"$ uses a precomputed irradiance cubemap:

$ E(n) = integral_(Omega^+) L_i (omega_i)(omega_i dot n) d omega_i. $

Convolve the source environment once at load time. Specular $"IBL"$ is split into two parts (Karis 2013):

$ L_o approx integral L_i thin f_r thin cos d omega approx underbrace((sum_k L_i (h_k)), "prefiltered cubemap by " a) dot underbrace(integral f_r cos d omega, "2D LUT"(N dot V, a)). $

A small 2D $"LUT"$ stores the geometry/Fresnel scale-bias terms; mip levels of the cubemap store prefiltered radiance per roughness. Cost at runtime is two texture fetches and a `lerp`.

== Energy Compensation

Even with Smith $G_2$, multiple-scattering between microfacets makes rough surfaces appear too dark. Kulla & Conty (2017) propose a furnace-test-derived compensation term:

$ f_("ms") = ((1 - E_o (omega_i))(1 - E_o (omega_o))) / (pi (1 - E_a)), $

where $E_o$ is the directional albedo (precomputed 2D $"LUT"$). Used in Unreal 5, Blender Cycles, Houdini Karma.

== Materials Beyond the Standard Model

- *Anisotropic $"GGX"$*: replace $alpha$ with $(alpha_x, alpha_y)$, evaluate in tangent frame. Brushed metal.
- *Clearcoat*: second specular lobe with its own $D$, $F$ at fixed $F_0 = 0.04$. Car paint.
- *Sheen*: Charlie distribution. Cloth, velvet.
- *Subsurface scattering*: diffusion profile or random-walk transmission. Skin, wax.
- *Transmission / glass*: $"BTDF"$ component, microfacet refraction; Walter et al. 2007.
- *Hair*: Marschner cylinder model with $R$, $"TT"$, $"TRT"$ lobes.

The Disney "principled" $"BRDF"$ unifies these under one artist-friendly parameter set (Burley 2012); the open glTF 2.0 PBR extension is the Web/industry interchange format.

== Tone Mapping and HDR

Linear-light shading must be mapped into display range. Common operators:

#table(
  columns: 3,
  [*Operator*], [*Formula*], [*Notes*],
  [Reinhard], [$x \/ (1 + x)$], [Soft, washes contrast],
  [ACES Filmic], [Rational fit, RRT+ODT], [Industry default, hue-preserving],
  [Hable / Uncharted 2], [$"U2"(x)$ rational], [Game industry classic],
  [AgX], [LUT-based, hue-preserving], [Blender 4.x default],
)

$"HDR"$ output (HDR10, $"PQ"$ transfer) skips tone mapping for the display range it supports but still applies a soft shoulder above peak luminance.

== Common $"PBR"$ Pitfalls

- *Authoring base color too bright*: pure white diffuse violates energy conservation; cap at $approx 0.95$.
- *sRGB-vs-linear confusion*: base color is sRGB; metallic / roughness / $"AO"$ are *linear* single-channel.
- *Roughness $approx 0$*: divide-by-zero in $D_("GGX")$ tail; clamp $alpha >= 0.0064$.
- *Single-scattering darkening*: looks "tarnished" without Kulla compensation.
- *Mixed metallic interpolation*: $"PBR"$ doesn't physically allow "half-metallic"; smoothly varying between $0$ and $1$ creates fake $F_0$.

== Further Reading

Pharr, M., Jakob, W., Humphreys, G. (2023). _Physically Based Rendering: From Theory to Implementation_, 4th ed. MIT Press. (Open access.)

Burley, B. (2012). "Physically Based Shading at Disney." SIGGRAPH course notes.

Karis, B. (2013). "Real Shading in Unreal Engine 4." SIGGRAPH.

Heitz, E. (2014). "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs." JCGT.

Kulla, C., Conty, A. (2017). "Revisiting Physically Based Shading at Imageworks." SIGGRAPH.

Walter, B. et al. (2007). "Microfacet Models for Refraction through Rough Surfaces." EGSR.

Lagarde, S., de Rousiers, C. (2014). "Moving Frostbite to PBR." SIGGRAPH course notes.

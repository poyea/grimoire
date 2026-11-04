#import "../template.typ": xref

= Colour, HDR, and Display <color-and-hdr>

A renderer computes physical quantities, light in linear units, and a display emits light according to a signal in some encoding, colour space and brightness range. Between the two lies a chain of conversions that decides whether the image looks as intended: the colour space the renderer works in, the transform from scene light to display light (tone mapping), the encoding of the signal, and the capabilities of the display, which now range from 300-nit office monitors to 1500-nit HDR televisions. This chapter covers the colour science needed to get that chain right: colour spaces and their primaries, transfer functions from sRGB to PQ and HLG, wide-gamut rendering, HDR output formats, display rendering transforms, banding and dithering, and colour grading.

*See also:* #xref("graphics-and-rendering", "physically-based-rendering", label: "Physically Based Rendering") (tone mapping operators), #xref("graphics-and-rendering", "textures-and-sampling", label: "Textures, Sampling, and Aliasing") (sRGB textures), #xref("computer-vision", "image-formation", label: "Image Formation") (radiometry and colour), #xref("graphics-and-rendering", "explicit-gpu-apis", label: "Explicit GPU APIs: Vulkan, Direct3D 12, and Metal") (swapchain formats)

== From Light to Perception

Rendering computes *radiometric* quantities: radiance in watts per steradian per square metre, per wavelength or per colour channel. Displays and viewers deal in *photometric* quantities weighted by the eye's sensitivity, chiefly *luminance* in candelas per square metre, universally called *nits*. A sunlit scene spans luminances from about $10^(-2)$ nits in deep shadow to $10^9$ nits for the sun's disc; an SDR display shows roughly 0.1 to 300 nits, and a good HDR display about 0.001 to 1000 nits or more.

Human brightness perception is roughly logarithmic over much of that range (the Weber-Fechner law: a just-noticeable difference is a roughly constant fraction of the current level). That has two consequences. Linear values stored in 8 bits waste codes on bright levels the eye cannot distinguish and band visibly in the darks, so signals are stored with a nonlinear *transfer function*. And the eye adapts: the same scene must be mapped to very different display ranges, which is the job of tone mapping.

== Colour Spaces

Colour vision is three-dimensional, so any colour can be specified by three numbers. The CIE 1931 *XYZ* space is the device-independent reference: $Y$ is luminance, and the *chromaticity* coordinates $x = X \/ (X + Y + Z)$, $y = Y \/ (X + Y + Z)$ describe hue and saturation independently of brightness.

An *RGB colour space* is defined by three things:

1. The chromaticities of its red, green and blue *primaries*, which define the triangle of colours (the *gamut*) it can represent with non-negative values.
2. A *white point*, the chromaticity of equal RGB (D65, approximating noon daylight, for all common display standards).
3. A *transfer function* relating encoded values to linear light.

#table(
  columns: 3,
  [*Space*], [*Gamut*], [*Typical use*],
  [sRGB / Rec. 709], [About 35 percent of visible chromaticities], [SDR monitors, the web, HDTV],
  [Display P3], [About 25 percent larger than sRGB], [Apple devices, modern phones, cinema-derived wide gamut],
  [Rec. 2020], [About 75 percent of visible chromaticities], [UHD and HDR video container],
  [ACES AP0], [Encloses all visible colours], [Archival interchange],
  [ACEScg (AP1)], [Slightly larger than Rec. 2020], [Rendering and compositing working space],
)

Converting linear RGB between spaces is a $3 times 3$ matrix (through XYZ, with chromatic adaptation if the white points differ). Rec. 709 to Rec. 2020, both D65, is:

$ mat(R; G; B)_2020 = mat(0.6274, 0.3293, 0.0433; 0.0691, 0.9195, 0.0114; 0.0164, 0.0880, 0.8956) mat(R; G; B)_709 $

Going the other way produces negative values for colours outside the smaller gamut, which must be handled deliberately (clipped, compressed or mapped) rather than clamped per channel, which shifts hues.

== Transfer Functions

=== sRGB and Gamma

The sRGB encoding (IEC 61966-2-1) is a short linear segment near black joined to a power curve with exponent 2.4, which overall approximates a gamma of 2.2:

```cpp
#include <cmath>

// Encoded sRGB value in [0, 1] to linear light (the EOTF of the standard).
inline float srgb_to_linear(float c) {
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

// Linear light in [0, 1] to encoded sRGB (the inverse).
inline float linear_to_srgb(float l) {
    return l <= 0.0031308f ? 12.92f * l : 1.055f * std::pow(l, 1.0f / 2.4f) - 0.055f;
}
```

GPUs implement both in hardware: sampling an `*_SRGB` texture decodes, and writing to an `*_SRGB` render target encodes (with blending performed in linear space). Using these formats instead of shader code is both faster and more correct. A recurring confusion is that many SDR displays and video standards actually use a pure 2.2 or 2.4 power (BT.1886 for broadcast) rather than the piecewise sRGB curve, which differs visibly near black.

=== PQ and HLG

HDR needs a transfer function that covers a far larger range without banding. Two are standard, both specified in ITU-R BT.2100:

- *Perceptual Quantizer* (PQ, SMPTE ST 2084) is *absolute*: each code value means a specific luminance, from 0 to 10 000 nits, with steps designed to stay below the visibility threshold at 12 bits (and close to it at 10).
- *Hybrid Log-Gamma* (HLG, from the BBC and NHK) is *relative*: a gamma-like curve for the lower part of the range and a logarithmic curve above it, which degrades gracefully on SDR displays and suits live broadcast, where the viewer's display is unknown.

```cpp
#include <cmath>

// Linear luminance in nits (0..10000) to a PQ-encoded signal in [0, 1].
inline float pq_encode(float nits) {
    constexpr float m1 = 2610.0f / 16384.0f;
    constexpr float m2 = 2523.0f / 4096.0f * 128.0f;
    constexpr float c1 = 3424.0f / 4096.0f;
    constexpr float c2 = 2413.0f / 4096.0f * 32.0f;
    constexpr float c3 = 2392.0f / 4096.0f * 32.0f;
    const float y = std::pow(std::fmax(nits, 0.0f) / 10000.0f, m1);
    return std::pow((c1 + c2 * y) / (1.0f + c3 * y), m2);
}
```

About half of the PQ code range is spent below 100 nits, where the eye is most sensitive. A 1000-nit highlight encodes at roughly 0.75.

== Rendering in a Wide Gamut

Lighting computed in linear sRGB primaries is subtly wrong for saturated colours. Multiplying spectra (light times reflectance) is approximated by multiplying RGB triples, and the approximation depends on the primaries: in a narrow gamut, multiple bounces of saturated light drift towards the gamut boundary and clip. Rendering in a wider working space such as ACEScg reduces these errors and leaves headroom for wide-gamut displays. The working space must be applied consistently: albedo textures authored in sRGB are converted to the working primaries on load (not merely decoded), lights and materials are specified in the working space, and the final output transform converts to the display's primaries.

For full spectral accuracy, offline renderers increasingly trace spectra (sampled wavelengths) rather than RGB, which handles dispersion, fluorescence and metamerism correctly; real-time renderers remain RGB.

== HDR Output

An HDR signal path needs a format the operating system and display understand:

- *HDR10*: PQ transfer, Rec. 2020 primaries, 10-bit components, with *static metadata* (SMPTE ST 2086 mastering display properties, plus MaxCLL and MaxFALL, the content's maximum and maximum frame-average light levels). The display tone-maps content that exceeds its capability.
- *Dolby Vision* and *HDR10+*: add per-scene or per-frame *dynamic metadata*, so the display's tone mapping adapts to each scene.
- *scRGB* (Windows): linear, 16-bit floating-point, sRGB primaries with values outside $[0, 1]$ representing extra brightness and wider gamut; 1.0 corresponds to 80 nits. The OS converts to the display's signal. It is convenient for applications because it stays linear.

Games and players query or ask the user for three numbers: the display's *peak luminance*, its *minimum* luminance, and the desired *paper white*, the luminance of a diffuse white surface and of UI elements (typically 100 to 250 nits). The game tone-maps its scene to fit the display's peak instead of leaving it to the display's unpredictable tone mapping, and places the UI at paper white so that it is neither dim nor glaring. A calibration screen that asks the user to adjust a symbol until it disappears is how these values are obtained when the platform does not report them reliably.

== The Display Rendering Transform

Scene-referred values (linear light, potentially thousands of times brighter than paper white) must be turned into display-referred values for each output. That *display rendering transform* includes:

1. *Exposure*: scale the scene, manually or by automatic exposure from a luminance histogram (ignoring the brightest and darkest few percent), adapting gradually like the eye.
2. *Tone mapping*: compress the range with a curve that rolls off highlights smoothly (#xref("graphics-and-rendering", "physically-based-rendering", label: "Physically Based Rendering") compares common operators).
3. *Gamut mapping*: bring colours outside the display gamut inside it.
4. *Encoding*: apply the output transfer function.

How tone mapping treats colour matters as much as its curve. Applying a curve to R, G and B *independently* desaturates bright colours towards white, which mimics film and is often desired, but also skews hues: a saturated blue light can turn purple, and fire can turn yellow-white unevenly. Fitted versions of the ACES output transform used in many games are per-channel and show exactly these shifts. Alternatives such as AgX (Blender's default since 4.0) and newer hue-preserving designs apply the curve in a space that keeps hue more stable and desaturate highlights deliberately. Whatever the choice, SDR and HDR outputs need separate transforms tuned together, because an HDR display reproduces highlights that the SDR curve must compress.

== Banding and Dithering

Quantising a smooth gradient (a sky, a fog falloff, a vignette) to 8 bits per channel produces visible steps, *banding*, especially in dark regions and after compression. Higher precision (10-bit output, 16-bit float intermediates) reduces it but does not remove it at the final quantisation. The fix is *dithering*: add a small amount of noise, about one quantisation step, before the final conversion, so that each step is replaced by a fine mix of the two neighbouring values that the eye averages into a smooth gradient. Blue noise (noise without low-frequency content) is visually least objectionable, and varying it each frame lets temporal accumulation and the eye average it further. Dithering must happen *after* the transfer function, in the encoded space where quantisation actually occurs.

== Colour Grading

Artistic colour decisions (warmer highlights, teal shadows, a desaturated flashback) are usually expressed as a *3D lookup table*: a $32^3$ or $33^3$ grid mapping input colours to output colours, sampled with trilinear (or tetrahedral) interpolation. The LUT is authored in a grading tool on a reference image and applied in the post-processing chain. Applied to scene-referred HDR data, the LUT is indexed in a log-encoded space so that its few entries are spread evenly over perceptual brightness; applied after tone mapping, it is indexed in display space but then differs between SDR and HDR outputs. Grading before tone mapping, in the scene-referred space, keeps one grade valid for every output.

== Further Reading

IEC (1999). "IEC 61966-2-1: Multimedia systems and equipment: Colour measurement and management: Default RGB colour space: sRGB." Standard.

ITU-R (2015). "Recommendation BT.2020-2: Parameter values for ultra-high definition television systems." Standard.

SMPTE (2014). "ST 2084: High Dynamic Range Electro-Optical Transfer Function of Mastering Reference Displays." Standard.

ITU-R (2018). "Recommendation BT.2100-2: Image parameter values for high dynamic range television." Standard.

Poynton, C. (2012). "Digital Video and HD: Algorithms and Interfaces." Morgan Kaufmann, 2nd ed.

Reinhard, E. et al. (2010). "High Dynamic Range Imaging: Acquisition, Display, and Image-Based Lighting." Morgan Kaufmann, 2nd ed.

Fairchild, M. (2013). "Color Appearance Models." Wiley, 3rd ed.

Academy of Motion Picture Arts and Sciences (2024). "ACES Documentation." ACES Central reference.

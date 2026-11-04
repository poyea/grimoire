#import "../template.typ": xref

= Textures, Sampling, and Aliasing <textures-and-sampling>

A rendered image is a set of samples of a continuous scene, and a texture is a set of samples of a continuous signal. Whenever one sampled signal is resampled at a different rate, as when a texture is drawn smaller or at a grazing angle, high frequencies fold back into false low-frequency patterns: shimmering, moiré and crawling edges. Texturing hardware is largely a machine for fighting that aliasing cheaply. This chapter covers the sampling theory behind it, texture addressing and filtering, mipmaps and how the GPU picks a level, anisotropic filtering, colour-space traps, block compression formats, memory layout, and virtual texturing for scenes larger than video memory.

*See also:* #xref("graphics-and-rendering", "rasterization-pipeline", label: "The Rasterization Pipeline") (multisampling), #xref("graphics-and-rendering", "realtime-engines", label: "Real-Time Engine Architecture") (temporal anti-aliasing, streaming), #xref("graphics-and-rendering", "shaders", label: "Shaders and Shading Languages"), #xref("numerical-computing", "fft", label: "The Fast Fourier Transform") (frequency domain)

== Sampling and Aliasing

The sampling theorem says that a signal containing no frequencies above $f_"max"$ can be reconstructed exactly from samples taken at more than $2 f_"max"$. Below that rate, frequencies above the *Nyquist limit* (half the sampling rate) are indistinguishable from lower ones and reappear as *aliases*. Rendering violates the theorem constantly: geometric edges, thin wires, high-contrast textures and sharp specular highlights all contain frequencies far above what one sample per pixel can represent.

There are two ways to fight aliasing:

- *Prefilter*: remove high frequencies before sampling. For textures this is cheap, because the texture is known in advance and can be filtered offline into mipmaps.
- *Supersample*: take more samples per pixel and filter them down (SSAA, MSAA, and temporal accumulation across frames, which reuses samples from previous frames).

A pixel is a point sample, not a little square (Smith, 1995). Treating it as a square box filter, which simply averages what falls inside it, is a convenient approximation, but box filters pass a lot of aliasing and better reconstruction filters (tent, Gaussian, Mitchell-Netravali) trade a little blur for much less of it.

== Addressing and Basic Filtering

A texture is addressed by normalised coordinates $(u, v)$ in $[0, 1]$. Texel $i$ covers $[i \/ N, (i + 1) \/ N)$, so its *centre* is at $(i + 0.5) \/ N$, an off-by-half detail responsible for many one-pixel shifts in post-processing code. Coordinates outside $[0, 1]$ are handled by the *address mode*: wrap (repeat), mirror, clamp to edge, or clamp to a border colour.

*Magnification* (one texel covers several pixels) uses nearest-neighbour or bilinear filtering. Bilinear interpolates the four texels around the sample point and is the baseline for nearly everything except pixel art and data textures that must not be blended (such as integer IDs).

*Minification* (many texels fall in one pixel) is where aliasing appears. Sampling one or four texels out of hundreds that fall under a pixel picks essentially random values, and the result shimmers as the camera moves.

== Mipmaps

A *mipmap* (Williams, 1983) is a pyramid of prefiltered copies of the texture, each half the width and height of the one above, down to $1 times 1$. The whole chain costs one third more memory than the base level. At draw time the GPU picks the level whose texels are about the size of a pixel, so each sample reads already-averaged data.

=== Choosing the Level

The GPU needs to know how fast the texture coordinates change across the screen. Pixel shaders run in $2 times 2$ *quads*, and the hardware computes screen-space derivatives by differencing the texture coordinates of neighbouring pixels in the quad. That is why helper lanes run for pixels outside the triangle, and why derivatives (and implicit-LOD sampling) are undefined inside divergent control flow. From the derivatives, measured in texels, the level of detail is

$ rho = max(sqrt((partial u \/ partial x)^2 + (partial v \/ partial x)^2), sqrt((partial u \/ partial y)^2 + (partial v \/ partial y)^2)), quad lambda = log_2 rho $

plus any LOD bias, clamped to the available levels. *Trilinear filtering* samples bilinearly from the two nearest levels and blends by the fractional part of $lambda$, avoiding visible seams where the level changes.

```cpp
#include <algorithm>
#include <cmath>

struct uv_derivatives {
    float du_dx, dv_dx;   // change of texture coordinates per pixel step in x
    float du_dy, dv_dy;   // ... and in y, all in normalised [0, 1] units
};

// Isotropic mip level as specified by OpenGL and Vulkan (before bias and clamp).
float mip_level(const uv_derivatives& d, int texture_width, int texture_height) {
    const float w = static_cast<float>(texture_width);
    const float h = static_cast<float>(texture_height);
    const float len_x = std::hypot(d.du_dx * w, d.dv_dx * h);   // texels per pixel in x
    const float len_y = std::hypot(d.du_dy * w, d.dv_dy * h);   // texels per pixel in y
    const float rho = std::max(len_x, len_y);
    return std::log2(std::max(rho, 1e-8f));
}
```

Using the *maximum* of the two lengths is conservative: when the footprint is elongated, as on a floor seen at a grazing angle, it selects a level coarse enough for the long axis, blurring the short axis unnecessarily. That blur is what anisotropic filtering removes.

=== Anisotropic Filtering

A pixel's footprint in texture space is approximately an ellipse (Heckbert, 1989). When it is long and thin, *anisotropic filtering* chooses the mip level for the *short* axis and takes several samples along the long axis. Hardware computes the ratio $N = min(ceil(P_"max" \/ P_"min"), N_"limit")$ of the footprint's axes, selects $lambda = log_2(P_"max" \/ N)$, and takes up to $N$ trilinear probes, with $N_"limit"$ typically 16. The cost is extra texture fetches only where the footprint is elongated, and the benefit is sharp floors and roads at a distance. On modern GPUs 16x anisotropic filtering is close to free for most scenes and should be the default for surface textures.

=== Mipmapping Pitfalls

- *Generate mips in linear space.* Averaging sRGB-encoded values darkens high-contrast textures. Convert to linear, filter, convert back.
- *Use a better filter than a $2 times 2$ box* for mip generation (Kaiser or Lanczos), especially for textures with fine detail.
- *Normal maps* lose length when averaged: two normals pointing in different directions average to a shorter vector. The shortening is real information (the surface is rougher at that scale), and techniques such as Toksvig's (2005) convert it into increased roughness to prevent specular aliasing, the sparkle of tiny bright highlights on bumpy surfaces at a distance.
- *Alpha-tested foliage* thins out in lower mips because averaged alpha falls below the cutoff. Rescale alpha per mip to preserve coverage, or use alpha-to-coverage with MSAA.

== Colour Space in Textures

Colour textures are usually stored *sRGB-encoded*, a perceptual curve that spends more of the 8 bits on dark values, where the eye is most sensitive. Lighting must happen in linear space. Declaring the texture with an sRGB format (`*_SRGB`) makes the hardware decode to linear *before* filtering, which is correct. Decoding in the shader after sampling filters in the wrong space and visibly darkens edges between bright and dark texels. Data textures (normals, roughness, metalness, masks) are *not* colours and must use linear (UNORM) formats; reading a normal map through an sRGB format bends every normal.

== Texture Compression

Uncompressed RGBA8 textures cost 4 bytes per texel, and a modern game ships tens of gigabytes of them. GPUs read *block-compressed* formats directly, decompressing in the texture unit, so compression saves memory, bandwidth and cache space at once. Each format encodes a fixed-size block of texels into a fixed number of bits, which keeps random access possible.

#table(
  columns: 4,
  [*Format*], [*Bits per texel*], [*Content*], [*Notes*],
  [BC1 (DXT1)], [4], [RGB, 1-bit alpha], [Two RGB565 endpoints plus 2-bit indices per texel in a $4 times 4$ block],
  [BC3 (DXT5)], [8], [RGBA], [BC1 colour plus a separate interpolated alpha block],
  [BC4 / BC5], [4 / 8], [One / two channels], [BC5 is the standard for tangent-space normal maps (store x, y; reconstruct z)],
  [BC6H], [8], [HDR RGB (half float)], [For HDR environment maps and lightmaps],
  [BC7], [8], [RGB or RGBA, high quality], [Multiple partition modes; the default for colour on desktop],
  [ETC2], [4 or 8], [RGB or RGBA], [Mandatory in OpenGL ES 3.0; common on older mobile],
  [ASTC], [8 down to 0.89], [LDR or HDR, 1 to 4 channels], [Block sizes from $4 times 4$ to $12 times 12$; standard on modern mobile],
)

The block formats are lossy and expensive to encode well (BC7 and ASTC encoders search many modes per block), so encoding happens offline in the asset pipeline. For distribution, *supercompressed* formats such as Basis Universal in KTX2 containers compress further with an entropy coder and transcode at load time into whichever block format the target GPU supports.

== Memory Layout and Caching

Textures are not stored row by row. A bilinear sample needs a $2 times 2$ neighbourhood, and adjacent pixels sample adjacent texels in both directions, so GPUs lay textures out in *tiled* or *swizzled* orders in which texels close in 2D are close in memory. The classic building block is the *Morton* (Z-order) curve, which interleaves the bits of $x$ and $y$:

```cpp
#include <cstdint>

// Spread the low 16 bits of v so there is a zero bit between each.
constexpr std::uint32_t part_1_by_1(std::uint32_t v) {
    v &= 0x0000'FFFFu;
    v = (v | (v << 8)) & 0x00FF'00FFu;
    v = (v | (v << 4)) & 0x0F0F'0F0Fu;
    v = (v | (v << 2)) & 0x3333'3333u;
    v = (v | (v << 1)) & 0x5555'5555u;
    return v;
}

// Morton index: bit i of x goes to bit 2i, bit i of y to bit 2i + 1.
constexpr std::uint32_t morton_2d(std::uint32_t x, std::uint32_t y) {
    return part_1_by_1(x) | (part_1_by_1(y) << 1);
}

static_assert(morton_2d(0, 0) == 0);
static_assert(morton_2d(1, 0) == 1);
static_assert(morton_2d(0, 1) == 2);
static_assert(morton_2d(1, 1) == 3);
static_assert(morton_2d(2, 0) == 4);
```

Real hardware layouts are vendor-specific combinations of tiles and swizzles, opaque to applications, which is why uploading a texture involves a copy into an "optimal" layout rather than a plain memory write. Texture caches are small (tens of kilobytes per shader core cluster) and optimised for this 2D locality, so random access patterns such as dependent lookups with unrelated coordinates are much slower than coherent ones.

== Virtual Texturing and Streaming

Open-world scenes reference far more texture data than fits in video memory. *Virtual texturing* (also called sparse or mega-texturing) treats texture data like virtual memory:

1. Textures are divided into fixed-size *pages* (for example $128 times 128$ texels plus a border for filtering).
2. A *feedback pass* (a low-resolution render, or writes from the main pass) records which pages at which mip levels the current view needs.
3. A streaming system loads missing pages from disk asynchronously into a large *physical texture cache* and updates an *indirection table* that maps virtual pages to physical locations.
4. Shaders translate virtual texture coordinates through the indirection table; a missing page falls back to a coarser resident mip.

Hardware *sparse residency* (tiled resources in Direct3D 12, sparse images in Vulkan) moves the indirection into the GPU's page tables, so ordinary texture instructions work on partially resident textures and can report whether the requested data was resident. Combined with fast storage and GPU decompression (DirectStorage), streaming makes video memory a cache over the whole asset set rather than a hard limit on it.

== Further Reading

Williams, L. (1983). "Pyramidal Parametrics." SIGGRAPH.

Heckbert, P. (1989). "Fundamentals of Texture Mapping and Image Warping." Master's thesis, University of California, Berkeley.

Smith, A. R. (1995). "A Pixel Is Not a Little Square." Microsoft Technical Memo 6.

Mitchell, D., Netravali, A. (1988). "Reconstruction Filters in Computer Graphics." SIGGRAPH.

Toksvig, M. (2005). "Mipmapping Normal Maps." Journal of Graphics Tools.

Nystad, J., Lassen, A., Pomianowski, A., Ellis, S., Olson, T. (2012). "Adaptive Scalable Texture Compression." High Performance Graphics.

Barrett, S. (2008). "Sparse Virtual Textures." Game Developers Conference.

Akenine-Möller, T. et al. (2018). "Real-Time Rendering." CRC Press, 4th ed., chapter 6 (Texturing).

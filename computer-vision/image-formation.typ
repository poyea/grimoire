#import "../template.typ": xref

= Image Formation and Representation

Before an algorithm can understand an image, we must understand how images form. This chapter covers the physics of image formation: projection geometry, radiometry, and optics, along with the digital representations and preprocessing pipelines that transform raw sensor data into arrays amenable to learning.

*See also:* #xref("computer-vision", "cnn-architectures", label: "CNN Architectures") (learned feature extraction), #xref("computer-vision", "3d-vision", label: "3D Vision and Neural Fields") (multi-view geometry in depth).

== Geometric Camera Models

=== Pinhole Camera

The *pinhole camera* projects a 3D world point $P = (X, Y, Z)$ to a 2D image point $p = (u, v)$ by central projection:

$ u = f X/Z + c_x, quad v = f Y/Z + c_y $

where $f$ is the focal length and $(c_x, c_y)$ is the principal point. In homogeneous coordinates, the projection matrix is

$ K = mat(f, 0, c_x; 0, f, c_y; 0, 0, 1), $

and the full camera matrix $P = K [R | t]$ maps world points to image points: $lambda p = K [R | t] P_w$.

=== Lens Distortion

Real lenses introduce *radial distortion* (barrel or pincushion) and *tangential distortion*. The Brown–Conrady model corrects distorted coordinates $(x_d, y_d)$:

$ x_u = x_d (1 + k_1 r^2 + k_2 r^4 + k_3 r^6), $

where $r^2 = x_d^2 + y_d^2$ and $k_1, k_2, k_3$ are radial distortion coefficients. Camera calibration (Zhang 2000) estimates $K$ and distortion coefficients from multiple views of a planar checkerboard.

=== Stereo Geometry

Two calibrated cameras with known relative pose $[R | t]$ define an *epipolar geometry*. The *fundamental matrix* $F$ encodes the constraint that corresponding points $p, p'$ satisfy $p'^top F p = 0$. When cameras are rectified (image planes coplanar), correspondences lie on horizontal scanlines and depth is inversely proportional to disparity:

$ Z = f b / (u_L - u_R) $

where $b$ is the baseline, $f$ the focal length, and $(u_L - u_R)$ the disparity.

== Radiometry and Colour

=== The Image Irradiance Equation

Sensor irradiance $E$ at pixel $(u,v)$ depends on scene radiance $L$, solid angle, and lens geometry:

$ E(u,v) = (pi/4) (D/f)^2 cos^4(alpha) L(X, Y, Z) $

where $D$ is aperture diameter and $alpha$ the angle from the optical axis. This $cos^4$ fall-off (vignetting) is a common artefact corrected in camera pipelines.

=== Colour Spaces

An RGB sensor integrates spectral radiance against three filter response curves. Key colour spaces:

#table(
  columns: 3,
  [*Space*], [*Description*], [*Use*],
  [sRGB], [Standard display, gamma 2.2], [Storage, display],
  [Linear RGB], [Physical radiance], [Rendering, compositing],
  [LAB], [Perceptually uniform], [Colour distance],
  [HSV/HSL], [Hue-saturation-value], [Colour selection],
  [YCbCr], [Luma + chroma], [JPEG/video compression],
)

Deep learning models typically operate on sRGB uint8 images normalised to $[0, 1]$ or $[-1, 1]$, with per-channel mean subtraction (ImageNet mean: $[0.485, 0.456, 0.406]$, std: $[0.229, 0.224, 0.225]$).

== Digital Image Representation

A greyscale image is a function $I: Omega subset ZZ^2 -> RR$. A colour image adds a channel dimension: $I: Omega -> RR^C$.

=== Sampling and Quantisation

Nyquist–Shannon sampling theorem: to avoid aliasing, the sampling rate must exceed twice the highest spatial frequency. Antialiasing before downsampling (low-pass filtering, typically Gaussian blur) is essential. Quantisation to 8 bits per channel introduces $plus.minus 0.5$ LSB error.

=== Image Transforms

Key linear transforms used in computer vision:

- *Discrete Fourier Transform (DFT)*: $hat(I)(u,v) = sum_(x,y) I(x,y) e^(-2 pi i (u x\/M + v y\/N))$. Convolution becomes pointwise multiplication in the frequency domain.
- *Discrete Cosine Transform (DCT)*: real-valued, energy-compacting; basis of JPEG compression and some vision features.
- *Wavelet transform*: multi-resolution decomposition; captures spatial-frequency information jointly.

== Image Filtering

A *linear filter* convolves the image with a kernel $h$:

$ (I * h)(x, y) = sum_(m, n) I(x-m, y-n) h(m, n). $

Common kernels:
- *Gaussian*: $h(x,y) = (1/(2 pi sigma^2)) e^(-(x^2+y^2)/(2 sigma^2))$. Smoothing, scale-space.
- *Sobel*: finite difference approximation of image gradient.
- *Laplacian of Gaussian (LoG)*: blob detection; zero-crossings mark edges.

=== Bilateral Filter

The bilateral filter is a *non-linear* edge-preserving smoother:

$ "BF"[I]_p = (1/W_p) sum_q G_(sigma_s) (||p-q||) G_(sigma_r) (|I_p - I_q|) I_q $

where $G_sigma_s$ is a spatial Gaussian and $G_sigma_r$ a range Gaussian. Pixels spatially nearby AND photometrically similar are averaged together.

== Image Pyramids and Scale Space

Scale-space theory formalises the idea that visual structure exists at multiple scales. The *Gaussian scale space* $L(x, y; t)$ is the convolution of $I$ with a Gaussian of variance $t$:

$ partial L / (partial t) = nabla^2 L. $

The *Laplacian of Gaussian* (or its approximation, the *Difference of Gaussian* used in SIFT) detects blobs at characteristic scale. Gaussian and Laplacian pyramids enable coarse-to-fine processing and are used in image blending, optical flow, and modern feature detectors.

== Classical Feature Descriptors

Before deep learning, hand-crafted features dominated:
- *SIFT* (Lowe, 2004): scale-invariant keypoints with 128-d gradient histogram descriptors. Rotation and scale invariant.
- *SURF*: faster approximation of SIFT using integral images.
- *ORB*: binary descriptor based on BRIEF; rotation invariant; real-time capable.
- *HOG* (Dalal & Triggs, 2005): histogram of oriented gradients; used in DPM pedestrian detection.

These remain relevant for structure-from-motion, AR tracking, and resource-constrained systems.

== Morphological Operations

For binary and greyscale images, morphological operations define non-linear filters:
- *Erosion*: $I minus.o B = {z | B_z subset I}$. Shrinks foreground.
- *Dilation*: $I plus.o B = {z | B_z inter I != emptyset}$. Expands foreground.
- *Opening* (erosion then dilation): removes small bright protrusions.
- *Closing* (dilation then erosion): fills small dark holes.

Used in medical imaging, document processing, and connected-component labelling.

== Image Quality Metrics

#table(
  columns: 3,
  [*Metric*], [*Formula*], [*Notes*],
  [PSNR], [$10 log_(10)(255^2 / "MSE")$ dB], [Simple, does not match perception],
  [SSIM], [Structural similarity (Wang et al.)], [Luminance, contrast, structure terms],
  [LPIPS], [Learned perceptual similarity], [VGG/AlexNet feature distance],
  [FID], [Frechet distance on Inception features], [Distribution-level, for generative models],
  [IS], [Inception Score], [Quality + diversity, less reliable than FID],
)

PSNR $>$ 30 dB is generally considered acceptable quality. FID below 5 indicates near-photorealistic generation.

== Further Reading

- Hartley, R., & Zisserman, A. (2004). _Multiple View Geometry in Computer Vision_, 2nd ed. Cambridge University Press.
- Szeliski, R. (2022). _Computer Vision: Algorithms and Applications_, 2nd ed. Springer. (freely available)
- Zhang, Z. (2000). A flexible new technique for camera calibration. _IEEE TPAMI_, 22(11).
- Lindeberg, T. (1994). Scale-space theory: a basic tool for analysing structures at different scales. _Journal of Applied Statistics_, 21(1).

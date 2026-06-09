= 3D Vision and Neural Fields

3D vision reconstructs, represents, and reasons about the three-dimensional world from 2D images. This chapter covers structure-from-motion, depth estimation, point clouds, and the revolutionary neural field representations (NeRF, Gaussian splatting) that have transformed novel view synthesis and 3D generation.

*See also:* _Image Formation_ (camera models, epipolar geometry), _Object Detection_ (3D detection), _Diffusion Models_ (3D generation).

== Structure from Motion

*Structure from Motion (SfM)* reconstructs 3D scene structure and camera poses from a collection of 2D images taken from different viewpoints.

=== Feature Matching Pipeline

1. *Feature detection and description*: extract SIFT, ORB, or SuperPoint keypoints.
2. *Matching*: nearest-neighbour matching between descriptor sets; ratio test filters ambiguous matches.
3. *Geometric verification*: RANSAC with the fundamental matrix $F$ (or essential matrix $E$ for calibrated cameras) to find inliers.
4. *Incremental reconstruction*: initialise from a well-conditioned image pair; register new images via PnP (Perspective-n-Point); triangulate new 3D points; bundle adjust.

*Bundle adjustment* (BA) is the global non-linear refinement that minimises reprojection error:

$ min_{{P_i}, {X_j}} sum_(i,j) ||pi(K R_i X_j + t_i) - p_(i j)||^2. $

COLMAP (Schönberger & Frahm, 2016) is the standard open-source SfM pipeline.

=== SLAM

*Simultaneous Localisation and Mapping (SLAM)* runs SfM in real-time as a robot/device moves. Visual SLAM systems: ORB-SLAM3 (feature-based), DSO (direct, no features), Kimera (metric-semantic). Neural SLAM methods (NICE-SLAM, Point-SLAM) use implicit neural representations as the map.

== Depth Estimation

=== Stereo Matching

Given rectified stereo pairs, find the per-pixel disparity $d(u,v) = u_L - u_R$ to recover depth $Z = f b / d$. Classic methods: SGM (Semi-Global Matching). Deep methods: PSMNet, RAFT-Stereo use a cost volume and iterative refinement. RAFT-Stereo achieves 1.27 px EPE on KITTI.

=== Monocular Depth Estimation

Predicting depth from a single image is ill-posed. MiDaS (Ranftl et al., 2020) is a scale-and-shift invariant depth predictor trained on 12 mixed datasets; strong zero-shot generalisation. Depth Anything (Yang et al., 2024) uses DINOv2 backbone with semi-supervised training on 62M unlabelled images; state-of-the-art monocular depth for dense prediction tasks.

*Metric depth* (absolute scale) is harder; ZoeDepth, UniDepth, and Depth Pro predict metric depth for single images.

=== LiDAR and RGB-D

LiDAR provides sparse but accurate depth measurements. Sensor fusion (early or late) combines camera and LiDAR for autonomous driving perception. RGB-D cameras (depth from structured light or time-of-flight: Kinect, RealSense, iPhone FaceID sensor) provide dense depth maps for indoor robotics and AR.

== Point Clouds

A point cloud $P = {p_i in RR^3}$ is a set of 3D points. Operations: registration (ICP, NDT), segmentation, classification, normal estimation.

=== PointNet

*PointNet* (Qi et al., 2017): processes unordered point sets directly. Each point is processed independently by shared MLP layers; global feature is obtained by symmetric max pooling over all points:

$ f({x_1, ..., x_n}) = g(h(x_1), ..., h(x_n)) $

where $g$ is element-wise max. Permutation invariant and efficient. Applies a spatial transformer network (T-Net) for input and feature alignment.

*PointNet++* adds hierarchical local grouping (farthest point sampling, ball query) to capture local structure.

=== Voxel and Pillar-Based Methods

VoxelNet voxelises the 3D space and applies 3D convolutions; efficient for outdoor LiDAR. PointPillars (Lang et al., 2019) uses vertical pillars instead of voxels, enabling fast 2D convolutions.

*3D object detection*: CenterPoint (Yin et al., 2021) detects 3D objects as centre heatmaps on BEV (bird's-eye view) feature maps; real-time, strong baseline for autonomous driving.

== Neural Radiance Fields

*NeRF* (Mildenhall et al., 2020) represents a 3D scene as a continuous function $F_theta: (x, y, z, theta, phi) -> (c, sigma)$ mapping 3D position and 2D viewing direction to colour $c$ and volume density $sigma$. Novel views are rendered by *volume rendering* along camera rays:

$ C(r) = integral_(t_n)^(t_f) T(t) sigma(r(t)) c(r(t), d) d t $

where $T(t) = exp(-integral_(t_n)^t sigma(r(s)) d s)$ is the accumulated transmittance. In practice, stratified and importance sampling discretise the integral.

NeRF is trained by minimising the photometric reconstruction loss between rendered and observed colours. It achieves photorealistic novel view synthesis but requires:
- Per-scene training (30 minutes to hours).
- Dense multi-view captures.
- Slow rendering ($~30$s per image).

=== NeRF Extensions

*Instant-NGP* (Müller et al., 2022): replaces the MLP with a multi-resolution hash encoding over a grid; reduces training to seconds. *Mip-NeRF 360* (Barron et al., 2022): anti-aliased cone tracing for unbounded outdoor scenes. *NeRF in the Wild* (Martin-Brualla et al., 2021): handles transient occluders and varying illumination in uncontrolled photo collections. *Dynamic NeRF / HyperNeRF*: deformable scene representations.

== 3D Gaussian Splatting

*3D Gaussian Splatting* (3DGS, Kerbl et al., 2023) represents a scene as a set of 3D Gaussians $G = {mu_i, Sigma_i, alpha_i, c_i}$, where each Gaussian has a centre, covariance (encodes size and orientation), opacity, and spherical harmonic colour coefficients. Rendering: project Gaussians onto the image plane, sort by depth, alpha-composite front to back:

$ C_"pixel" = sum_i c_i alpha_i product_(j<i) (1 - alpha_j). $

Rasterization on GPU: Gaussian footprints are splatted as 2D ellipses. Trained from SfM point clouds with adaptive density control (split dense regions, cull transparent Gaussians).

*Advantages over NeRF*:
- Real-time rendering ($>$ 30 fps at 1080p).
- Training in 30–60 minutes.
- Explicit representation (editable).

*Limitations*: larger storage ($>100$ MB per scene), struggles with thin structures and transparent objects.

*Extensions*: 4D Gaussians for dynamic scenes, GaussianAvatar for human bodies, Gaussian Opacity Fields for improved geometry.

== 3D Generation

*Single-view 3D reconstruction*: One-2-3-45 (Liu et al., 2023), Zero123 generate novel views from a single image using diffusion, then reconstruct 3D.

*Text-to-3D*: DreamFusion (Poole et al., 2022) uses *Score Distillation Sampling (SDS)*: a NeRF is optimised such that renders look realistic under a 2D diffusion model. Magic3D, Fantasia3D improve quality. LRM (Large Reconstruction Model) predicts a 3D representation from images in a single forward pass using a transformer.

*3D-native diffusion*: Point-E, Shap-E diffuse over structured 3D representations. Direct diffusion over Gaussian parameters (GaussianObject, GaussianDreamer) is an active research direction.

== Optical Flow

Optical flow estimates per-pixel motion between consecutive frames. RAFT (Teed & Deng, 2020): constructs a 4D correlation volume between frame features; iteratively refines a dense flow field with a recurrent GRU. 1.43 px EPE on Sintel clean, state-of-the-art at the time. RAFT-3D extends to 3D scene flow.

== Human Pose Estimation

*2D pose estimation*: HRNet (Sun et al., 2019) maintains high-resolution representations across all stages; multi-scale fusion. ViTPose (Xu et al., 2022) applies ViT directly to pose estimation. *3D pose estimation*: VideoPose3D lifts 2D keypoints to 3D using temporal convolutions. Whole-body estimation (DWPose, RTMPose) includes face and hands.

== Further Reading

- Mildenhall, B. et al. (2020). NeRF: representing scenes as neural radiance fields for view synthesis. _ECCV_.
- Kerbl, B. et al. (2023). 3D Gaussian splatting for real-time radiance field rendering. _SIGGRAPH_.
- Schönberger, J. L., & Frahm, J.-M. (2016). Structure-from-motion revisited (COLMAP). _CVPR_.
- Qi, C. R. et al. (2017). PointNet. _CVPR_.
- Teed, Z., & Deng, J. (2020). RAFT: recurrent all-pairs field transforms for optical flow. _ECCV_.
- Müller, T. et al. (2022). Instant neural graphics primitives (Instant-NGP). _SIGGRAPH_.

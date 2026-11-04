#import "../template.typ": xref

= Multi-View Geometry and Calibration <multi-view-geometry>

Deep networks have taken over recognition, but geometry still decides where things are. Every metric measurement a camera makes (the distance to a pedestrian, the pose of a robot arm, the scale of a reconstructed building) rests on a calibrated camera model and the projective geometry relating several views. This chapter develops that geometry: the pinhole camera with lens distortion, calibration, homographies, the epipolar constraint between two views, triangulation, camera pose from known points, stereo depth and its error, and bundle adjustment, the nonlinear least-squares problem underneath every structure-from-motion and SLAM system.

*See also:* #xref("computer-vision", "image-formation", label: "Image Formation") (camera basics), #xref("computer-vision", "3d-vision", label: "3D Vision and Neural Fields"), #xref("numerical-computing", "linear-systems", label: "Linear Systems") (least squares and SVD)

== The Projective Camera

In homogeneous coordinates a pinhole camera maps a world point $bold(X) = (X, Y, Z, 1)^top$ to an image point $bold(x) = (u, v, 1)^top$ up to scale:

$ lambda bold(x) = K [R | bold(t)] bold(X), quad K = mat(f_x, s, c_x; 0, f_y, c_y; 0, 0, 1) $

$R$ and $bold(t)$ (the *extrinsics*) move world coordinates into the camera frame. $K$ (the *intrinsics*) holds the focal lengths in pixels $f_x, f_y$, the principal point $(c_x, c_y)$, and a skew $s$ that is zero for every modern sensor. The $3 times 4$ matrix $P = K[R | bold(t)]$ has 11 degrees of freedom: 5 intrinsic and 6 extrinsic.

Projection discards depth: every point on the ray through the camera centre and $bold(x)$ projects to the same pixel. Recovering the third dimension needs a second view, a known object, or a learned prior.

=== Lens Distortion

Real lenses bend straight lines. The Brown-Conrady model, used by OpenCV and most calibration tools, applies distortion to normalised coordinates $(x, y)$ (after removing $K$, before applying it) with $r^2 = x^2 + y^2$:

$ x_d = x (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2 p_1 x y + p_2 (r^2 + 2 x^2) $
$ y_d = y (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1 (r^2 + 2 y^2) + 2 p_2 x y $

The $k_i$ model radial (barrel or pincushion) distortion and the $p_i$ tangential distortion from a lens not quite parallel to the sensor. This polynomial fails beyond about 120 degrees field of view; fisheye and omnidirectional lenses use an equidistant-style model instead, such as Kannala-Brandt ($r_d = theta + k_1 theta^3 + k_2 theta^5 + dots$, a polynomial in the angle of incidence $theta$) or the unified model of Mei and Rives.

== Calibration

Calibration estimates $K$ and the distortion coefficients. Zhang's method (2000) is the standard: photograph a planar target (a checkerboard or a ChArUco board) in 10 to 30 poses. Each view of a plane gives a homography $H = K [bold(r)_1, bold(r)_2, bold(t)]$, and because $bold(r)_1$ and $bold(r)_2$ are orthonormal, each homography gives two linear constraints on $K^(-top) K^(-1)$. Three views determine $K$ in closed form. A Levenberg-Marquardt refinement then minimises the *reprojection error* over all intrinsics, distortion terms and per-view poses jointly.

Practical rules that matter more than the algorithm:

- *Cover the whole image*, especially the corners, where distortion is largest. Poses only in the centre leave $k_2$ and $k_3$ unconstrained.
- *Tilt the target* by 30 to 45 degrees in several directions. Fronto-parallel views make focal length and distance indistinguishable.
- *Lock focus and zoom.* Changing either changes $K$.
- *Judge by residuals, not by the RMS alone.* An RMS reprojection error below about 0.3 pixels is good for a sharp target. Residuals that grow towards the image edges mean the distortion model is too weak.
- *Fix what you know.* Setting skew to zero and, for square pixels, $f_x = f_y$ reduces variance.

== Homographies

A homography is an invertible $3 times 3$ projective transformation $bold(x)' tilde H bold(x)$ with 8 degrees of freedom. It relates two images exactly when the scene is a plane (a document, a road surface, a sports field) or the camera rotates about its centre without translating (panoramas). Each correspondence $bold(x)_i <-> bold(x)'_i$ gives two linear equations in the nine entries of $H$ via the cross product $bold(x)'_i times H bold(x)_i = 0$, so four points in general position determine $H$, and more are solved in the least-squares sense with the SVD (the *direct linear transform*, DLT).

Hartley's normalisation is essential: translate each point set to zero mean and scale it so the mean distance from the origin is $sqrt(2)$, solve, then undo the transformations. Without it the DLT's matrix is badly conditioned, because pixel coordinates near 1000 sit beside homogeneous ones, and the result degrades by orders of magnitude.

== Two Views: Epipolar Geometry

Two cameras observing the same point $bold(X)$ form a plane through both camera centres and $bold(X)$. Its intersection with each image is an *epipolar line*, so the match of a point in one image must lie on a known line in the other. That constraint is captured by the *fundamental matrix* $F$:

$ bold(x)'^top F bold(x) = 0 $

$F$ is $3 times 3$, rank 2, defined up to scale, and has 7 degrees of freedom. With calibrated cameras, working in normalised coordinates $hat(bold(x)) = K^(-1) bold(x)$, the same relation holds for the *essential matrix*

$ E = [bold(t)]_times R, quad F = K'^(-top) E K^(-1) $

where $[bold(t)]_times$ is the skew-symmetric cross-product matrix. $E$ has 5 degrees of freedom (3 rotation, 2 for the direction of translation; the scale of $bold(t)$ is unobservable from images alone) and its two non-zero singular values are equal.

Estimation:

- *Eight-point algorithm*: stack one linear equation per correspondence, solve by SVD, then enforce rank 2 by zeroing the smallest singular value. With Hartley normalisation it is competitive with far more complex methods (Hartley, 1997).
- *Seven-point algorithm*: the minimal case for $F$, giving one or three solutions from a cubic.
- *Five-point algorithm* (Nistér, 2004): the minimal case for $E$ with calibrated cameras, up to 10 solutions. It is the default inside RANSAC for calibrated relative pose because a smaller minimal set needs fewer RANSAC iterations.

Decomposing $E$ by SVD gives four combinations of $(R, bold(t))$. Only one places the triangulated points in front of both cameras (the *cheirality* check), and that is the relative pose.

Degenerate configurations are worth knowing because they occur in practice: if all points lie on a plane, or the camera only rotates, $F$ is not determined and a homography explains the data better. Robust pipelines fit both models and pick by a model-selection score such as GRIC.

== Triangulation

Given two (or more) calibrated views of a point with known poses, *triangulation* intersects the back-projected rays. Because of noise the rays do not meet exactly. The linear method stacks $bold(x)_i times P_i bold(X) = 0$ for every view and solves for $bold(X)$ by SVD:

```cpp
#include <Eigen/Dense>
#include <span>

// Linear (DLT) triangulation from N >= 2 views.
// projections[i] is the 3x4 matrix P_i = K_i [R_i | t_i];
// pixels[i] is the observed (u, v) in view i.
Eigen::Vector3d triangulate_dlt(std::span<const Eigen::Matrix<double, 3, 4>> projections,
                                std::span<const Eigen::Vector2d> pixels) {
    const auto n = static_cast<Eigen::Index>(projections.size());
    Eigen::MatrixXd a(2 * n, 4);
    for (Eigen::Index i = 0; i < n; ++i) {
        const auto& p = projections[i];
        const double u = pixels[i].x();
        const double v = pixels[i].y();
        a.row(2 * i)     = u * p.row(2) - p.row(0);
        a.row(2 * i + 1) = v * p.row(2) - p.row(1);
    }
    // Solution: right singular vector of the smallest singular value.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeFullV);
    const Eigen::Vector4d x = svd.matrixV().col(3);
    return x.head<3>() / x(3);
}
```

The linear solution minimises an algebraic error with no geometric meaning, so it is used as the starting point for a few Gauss-Newton steps on the reprojection error, or for the closed-form optimal two-view method of Hartley and Sturm. Accuracy depends on the angle between the rays: points seen with nearly parallel rays (small baseline, far away) have depth uncertainty far larger than their lateral uncertainty, and well-designed pipelines discard points with a triangulation angle below one or two degrees.

== Camera Pose from Known Points: PnP

The *Perspective-n-Point* problem recovers a calibrated camera's pose from $n$ known 3D points and their projections. It is how augmented reality anchors content to a marker, how a robot localises against a map, and how SLAM tracks the camera between keyframes.

- *P3P* uses the minimum of three points and has up to four solutions; a fourth point disambiguates. Modern solvers (Lambda Twist, Persson and Nordberg, 2018) are fast and numerically stable, which matters inside RANSAC.
- *EPnP* (Lepetit, Moreno-Noguer and Fua, 2009) expresses every point as a weighted sum of four virtual control points, giving a solution linear in $n$ that is accurate for large $n$.
- *Iterative refinement* minimises reprojection error from either starting point.

== Stereo and Depth Error

A calibrated stereo pair is *rectified* by rotating both image planes so that epipolar lines become horizontal scan lines. Matching then becomes a one-dimensional search along rows, and depth follows from the *disparity* $d$ (the horizontal offset between matching pixels):

$ Z = (f B) / d $

with focal length $f$ in pixels and baseline $B$. Differentiating gives the error that every stereo designer must budget:

$ Delta Z approx Z^2 / (f B) Delta d $

Depth error grows with the *square* of distance. For $f = 1000$ px, $B = 0.12$ m and a matching accuracy of a quarter pixel, the error is about 5 cm at 5 m but over 80 cm at 20 m. Doubling the baseline halves the error but enlarges the minimum depth and the region seen by only one camera. That quadratic law is why automotive stereo rigs use wide baselines and why long-range perception leans on lidar or radar.

Dense matching algorithms trade quality for speed: block matching is fast but fails on texture-poor surfaces; semi-global matching (Hirschmüller, 2008) aggregates matching costs along several directions and was the industrial standard for a decade; learned stereo networks (RAFT-Stereo and its successors) now lead the benchmarks, especially on reflective and untextured regions.

== Bundle Adjustment

Every multi-view reconstruction ends with the same optimisation. Given cameras $j$ with parameters $bold(c)_j$ (pose, possibly intrinsics), points $bold(X)_i$, and observations $bold(x)_(i j)$, *bundle adjustment* minimises the total reprojection error

$ min_({bold(c)_j}, {bold(X)_i}) sum_((i, j) in cal(O)) rho(norm(bold(x)_(i j) - pi(bold(c)_j, bold(X)_i))^2) $

where $pi$ is the projection function including distortion and $rho$ is a robust loss (Huber or Cauchy) that limits the influence of the outlier matches that survive earlier filtering.

The problem is large (millions of points in city-scale reconstructions) but highly structured. Each observation involves one camera and one point, so the Jacobian is sparse, and the normal equations have a block structure with a block-diagonal point part. The *Schur complement trick* eliminates the points to get a much smaller *reduced camera system*, solved by sparse Cholesky decomposition or by preconditioned conjugate gradients, after which the points are recovered by back-substitution. Levenberg-Marquardt wraps the linear solves to handle the nonlinearity. Ceres Solver, g2o and GTSAM implement all of this, and Triggs et al. (2000) remains the definitive treatment.

Bundle adjustment has a *gauge freedom*: the whole reconstruction can be rotated, translated and (for monocular cameras) scaled without changing the error. Solvers fix it by holding the first camera's pose and one distance constant, or by working with the minimal parameterisation of the gauge.

== Further Reading

Hartley, R., Zisserman, A. (2004). "Multiple View Geometry in Computer Vision." Cambridge University Press, 2nd ed.

Zhang, Z. (2000). "A Flexible New Technique for Camera Calibration." IEEE Transactions on Pattern Analysis and Machine Intelligence.

Hartley, R. (1997). "In Defense of the Eight-Point Algorithm." IEEE Transactions on Pattern Analysis and Machine Intelligence.

Nistér, D. (2004). "An Efficient Solution to the Five-Point Relative Pose Problem." IEEE Transactions on Pattern Analysis and Machine Intelligence.

Lepetit, V., Moreno-Noguer, F., Fua, P. (2009). "EPnP: An Accurate O(n) Solution to the PnP Problem." International Journal of Computer Vision.

Kannala, J., Brandt, S. (2006). "A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses." IEEE Transactions on Pattern Analysis and Machine Intelligence.

Hirschmüller, H. (2008). "Stereo Processing by Semiglobal Matching and Mutual Information." IEEE Transactions on Pattern Analysis and Machine Intelligence.

Triggs, B., McLauchlan, P., Hartley, R., Fitzgibbon, A. (2000). "Bundle Adjustment: A Modern Synthesis." Vision Algorithms: Theory and Practice, LNCS 1883.

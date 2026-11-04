#import "../template.typ": xref

= Visual Odometry and SLAM <visual-slam>

A robot, a drone, a headset or a phone running augmented reality must know where it is, continuously, in a space it may never have seen. *Visual odometry* (VO) estimates the camera's motion incrementally from consecutive frames. *Simultaneous localisation and mapping* (SLAM) also builds a map and recognises places it has visited before, so that accumulated drift can be corrected. This chapter covers the architecture shared by modern systems, the main design families (feature-based, direct and learned), visual-inertial fusion, factor graphs and pose-graph optimisation, and how trajectories are evaluated.

*See also:* #xref("computer-vision", "multi-view-geometry", label: "Multi-View Geometry and Calibration"), #xref("computer-vision", "feature-matching", label: "Feature Matching and Robust Estimation"), #xref("computer-vision", "3d-vision", label: "3D Vision and Neural Fields"), #xref("numerical-computing", "optimization-algorithms", label: "Optimization Algorithms") (Gauss-Newton and Levenberg-Marquardt)

== The Problem and Its Variants

The state is the camera trajectory $T_1, T_2, dots$ (each a rigid transform in $"SE"(3)$) and a map. The measurements are images, and optionally inertial (IMU), wheel odometry or GPS readings. What can be recovered depends on the sensors:

#table(
  columns: 3,
  [*Sensor setup*], [*Observable*], [*Main difficulty*],
  [Monocular camera], [Trajectory and map up to an unknown global scale], [Scale drift; initialisation needs parallax],
  [Stereo camera], [Metric scale from the known baseline], [Depth accuracy falls with the square of distance],
  [RGB-D camera], [Metric depth per pixel, short range], [Indoors only; fails in sunlight and on dark or shiny surfaces],
  [Camera plus IMU (visual-inertial)], [Metric scale, gravity direction, roll and pitch], [Calibration and time synchronisation; IMU bias],
)

Monocular systems are the cheapest and the hardest. Every reconstruction is valid at any scale, and scale estimates drift over long trajectories. Adding an IMU fixes scale and the direction of gravity, because the accelerometer measures metric acceleration. Global position and heading (yaw) remain unobservable without an external reference: visual-inertial odometry has four unobservable degrees of freedom, and any estimator that pretends otherwise becomes overconfident.

== System Architecture

Nearly every modern system (ORB-SLAM3, VINS-Fusion, OKVIS, Basalt) has the same parts, usually on separate threads:

1. *Tracking (front end).* For each new frame, predict the pose from a motion model or the IMU, find correspondences with the local map, and solve for the pose by PnP or a small optimisation. This must keep up with the camera rate (30 to 200 Hz).
2. *Keyframe selection.* Most frames are redundant. A frame becomes a *keyframe* when tracking quality drops, when it sees enough new structure, or after a set time. Only keyframes enter the map.
3. *Local mapping (back end).* Triangulate new points from keyframes, and run *local bundle adjustment* over a window of recent keyframes and the points they observe (#xref("computer-vision", "multi-view-geometry", label: "Multi-View Geometry and Calibration")). Cull redundant keyframes and unreliable points.
4. *Loop closing.* Recognise that the current view matches an old keyframe, compute the relative transform between them, and distribute the accumulated drift around the loop by *pose-graph optimisation*, followed optionally by a global bundle adjustment.
5. *Relocalisation.* When tracking is lost (occlusion, motion blur), use the same place-recognition machinery to find the pose again against the existing map.

=== Place Recognition

Loop closure needs to answer "have I seen this place before?" against thousands of keyframes in milliseconds. The classical answer is a *bag of visual words*: descriptors are quantised against a vocabulary tree trained offline (DBoW2 for ORB descriptors), each keyframe becomes a sparse word histogram, and an inverted index retrieves candidates by TF-IDF similarity. Learned global descriptors such as NetVLAD and its successors are more robust to changes in lighting, season and viewpoint. Every candidate is then *verified geometrically* by matching features and fitting a relative pose with RANSAC, because a false loop closure is catastrophic: it bends the whole map to satisfy a constraint that does not exist.

== Feature-Based, Direct, and Learned Methods

*Feature-based* systems (PTAM, ORB-SLAM) track sparse keypoints and minimise reprojection error. They are robust to large motions and lighting changes, handle loop closure naturally through descriptors, and are the most widely deployed. ORB-SLAM3 (Campos et al., 2021) supports monocular, stereo and RGB-D cameras, with and without an IMU, and multiple maps that merge when places are revisited.

*Direct* methods (LSD-SLAM, DSO) skip keypoints and minimise *photometric error*: the difference in intensity between a pixel and its projection into another frame, given depth and pose. They use every pixel with sufficient gradient, including edges and weak texture, and achieve high accuracy on well-calibrated cameras. DSO (Engel, Koltun and Cremers, 2018) models exposure time, lens vignetting and camera response, without which the brightness-constancy assumption fails. Direct methods need small inter-frame motion and are sensitive to rolling shutter and auto-exposure. *Semi-direct* methods (SVO) use direct alignment for speed and features for the back end.

*Learned* systems train parts of the pipeline end to end. DROID-SLAM (Teed and Deng, 2021) iteratively updates dense optical flow between frames with a recurrent network and feeds it into a differentiable dense bundle adjustment layer. It is markedly more robust than classical systems on difficult sequences, at the cost of a GPU. DPVO makes the same idea sparse and much cheaper. Newer *neural-map* systems build NeRF or Gaussian-splat maps online (#xref("computer-vision", "3d-vision", label: "3D Vision and Neural Fields")), giving photorealistic maps rather than point clouds.

== Visual-Inertial Fusion

An IMU measures angular velocity and specific force at 200 to 1000 Hz. Integrating it predicts motion over the short gaps between frames and through motion blur, while the camera corrects the IMU's drift. The IMU measurements are corrupted by slowly varying *biases* that must be estimated as part of the state.

Two estimator families dominate:

- *Filtering.* The Multi-State Constraint Kalman Filter (MSCKF; Mourikis and Roumeliotis, 2007) keeps a sliding window of past camera poses in an extended Kalman filter, and uses each feature track to constrain those poses without adding the feature to the state. It is efficient and runs on phone and headset processors.
- *Optimisation.* Sliding-window smoothers (OKVIS, VINS-Mono, Basalt) jointly optimise poses, velocities, biases and landmarks over recent keyframes, with older states *marginalised* into a prior. They are typically more accurate than filters and can relinearise.

Integrating raw IMU samples between keyframes would require re-integration every time the optimiser changes the starting state. *IMU preintegration* (Lupton and Sukkarieh, 2012; Forster et al., 2017) instead summarises all measurements between two keyframes as a single relative-motion factor, with first-order corrections for bias updates. Every modern visual-inertial optimiser uses it.

The practical difficulties are not in the estimator. They are camera-IMU extrinsic calibration (tools such as Kalibr), time synchronisation (an offset of a few milliseconds measurably degrades accuracy; estimate it or synchronise the hardware), and initialisation (scale, gravity and biases must be estimated from the first seconds of motion, which must contain enough acceleration).

== Factor Graphs and Pose-Graph Optimisation

SLAM back ends are best described as *factor graphs*: variables (poses, landmarks, velocities, biases) are nodes, and each measurement is a factor connecting the variables it constrains, with an error function and a covariance. The maximum a posteriori estimate is a nonlinear least-squares problem

$ hat(bold(theta)) = arg min_(bold(theta)) sum_k norm(bold(e)_k (bold(theta)))_(Sigma_k)^2 $

solved by Gauss-Newton or Levenberg-Marquardt, exploiting the graph's sparsity. GTSAM, g2o and Ceres are the standard libraries, and iSAM2 (Kaess et al., 2012) updates the solution incrementally as factors arrive instead of re-solving from scratch.

*Pose-graph optimisation* is the reduced problem used for loop closure: landmarks are dropped, and each factor is a relative-pose measurement $Z_(i j)$ between keyframes $i$ and $j$. The residual compares the measured and estimated relative motion in the tangent space of $"SE"(3)$:

$ bold(e)_(i j) = log(Z_(i j)^(-1) T_i^(-1) T_j)^or in RR^6 $

Optimising over rotations needs care, because $"SO"(3)$ is not a vector space. Solvers update poses by $T arrow.l T exp(hat(bold(delta)))$ with a 6-vector increment $bold(delta)$, the standard Lie-group treatment. For monocular systems the loop closure is optimised over $"Sim"(3)$, which adds a scale per keyframe and lets the optimiser correct scale drift as well.

== Evaluating Trajectories

Two metrics, defined by Sturm et al. (2012) for the TUM RGB-D benchmark, are universal:

- *Absolute trajectory error* (ATE): align the estimated trajectory to ground truth with the best rigid transform (or similarity transform for monocular systems), then report the RMSE of the position differences. It measures global consistency.
- *Relative pose error* (RPE): compare relative motions over a fixed interval (for example, 1 m or 1 s). It measures local drift and is independent of where errors occurred.

The alignment is Umeyama's (1991) closed-form least-squares solution:

```cpp
#include <Eigen/Dense>
#include <cmath>
#include <vector>

struct alignment {
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    double scale;
};

// Least-squares similarity transform mapping `estimate` onto `ground_truth`
// (Umeyama, 1991). Pass with_scale = false for stereo or visual-inertial systems.
alignment umeyama(const std::vector<Eigen::Vector3d>& estimate,
                  const std::vector<Eigen::Vector3d>& ground_truth,
                  bool with_scale) {
    const double n = static_cast<double>(estimate.size());
    Eigen::Vector3d mean_est = Eigen::Vector3d::Zero();
    Eigen::Vector3d mean_gt = Eigen::Vector3d::Zero();
    for (std::size_t i = 0; i < estimate.size(); ++i) {
        mean_est += estimate[i];
        mean_gt += ground_truth[i];
    }
    mean_est /= n;
    mean_gt /= n;

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    double variance_est = 0.0;
    for (std::size_t i = 0; i < estimate.size(); ++i) {
        const Eigen::Vector3d de = estimate[i] - mean_est;
        covariance += (ground_truth[i] - mean_gt) * de.transpose();
        variance_est += de.squaredNorm();
    }
    covariance /= n;
    variance_est /= n;

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d s = Eigen::Matrix3d::Identity();
    if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0.0) {
        s(2, 2) = -1.0;   // avoid a reflection
    }
    const Eigen::Matrix3d r = svd.matrixU() * s * svd.matrixV().transpose();
    const double c = with_scale
        ? (svd.singularValues().asDiagonal() * s).trace() / variance_est
        : 1.0;
    return {r, mean_gt - c * r * mean_est, c};
}

double absolute_trajectory_rmse(const std::vector<Eigen::Vector3d>& estimate,
                                const std::vector<Eigen::Vector3d>& ground_truth,
                                bool with_scale) {
    const alignment a = umeyama(estimate, ground_truth, with_scale);
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < estimate.size(); ++i) {
        const Eigen::Vector3d aligned = a.scale * a.rotation * estimate[i] + a.translation;
        sum_sq += (aligned - ground_truth[i]).squaredNorm();
    }
    return std::sqrt(sum_sq / static_cast<double>(estimate.size()));
}
```

Timestamps must first be associated between the two trajectories, and the choice of whether to fit scale must match the sensor setup: fitting scale for a stereo system hides a real scale error. The standard datasets are KITTI (driving, stereo and lidar), EuRoC (drone, stereo plus IMU), TUM RGB-D and TUM-VI, and more recent sets such as Hilti SLAM and the Newer College dataset that include hard cases.

== Why Systems Fail

Benchmarks report accuracy; deployments are limited by robustness. The recurring causes of failure:

- *Fast rotation and motion blur*, which destroy features. IMU fusion and high frame rates help; global-shutter cameras help most.
- *Rolling shutter*, in which each image row is captured at a different time, so a moving camera produces a geometrically distorted image. Either model it in the projection function or use a global-shutter sensor.
- *Low texture and repetitive structure*: white corridors, glass, grass, warehouse shelving. Lines and planes as additional landmarks, or other sensors, are the remedies.
- *Dynamic scenes*: people and vehicles violate the static-world assumption. Semantic segmentation can mask likely-moving classes, and robust losses limit the damage.
- *Lighting change* and auto-exposure, which break photometric assumptions and descriptor matching. Photometric calibration and learned features help.
- *Wrong loop closures*, which corrupt the whole map. Strict geometric verification and robust back ends (switchable constraints, graduated non-convexity) defend against them.

== Further Reading

Cadena, C. et al. (2016). "Past, Present, and Future of Simultaneous Localization and Mapping: Toward the Robust-Perception Age." IEEE Transactions on Robotics.

Klein, G., Murray, D. (2007). "Parallel Tracking and Mapping for Small AR Workspaces." ISMAR.

Campos, C., Elvira, R., Gómez Rodríguez, J., Montiel, J., Tardós, J. (2021). "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multimap SLAM." IEEE Transactions on Robotics.

Engel, J., Koltun, V., Cremers, D. (2018). "Direct Sparse Odometry." IEEE Transactions on Pattern Analysis and Machine Intelligence.

Mourikis, A., Roumeliotis, S. (2007). "A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation." ICRA.

Forster, C., Carlone, L., Dellaert, F., Scaramuzza, D. (2017). "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry." IEEE Transactions on Robotics.

Kaess, M. et al. (2012). "iSAM2: Incremental Smoothing and Mapping Using the Bayes Tree." International Journal of Robotics Research.

Teed, Z., Deng, J. (2021). "DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras." NeurIPS.

Sturm, J., Engelhard, N., Endres, F., Burgard, W., Cremers, D. (2012). "A Benchmark for the Evaluation of RGB-D SLAM Systems." IROS.

Umeyama, S. (1991). "Least-Squares Estimation of Transformation Parameters Between Two Point Patterns." IEEE Transactions on Pattern Analysis and Machine Intelligence.

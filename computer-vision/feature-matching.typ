#import "../template.typ": xref

= Feature Matching and Robust Estimation <feature-matching>

Before geometry can be computed, images must be put into correspondence: this pixel in one image shows the same physical point as that pixel in another. Correspondence is the front end of panorama stitching, visual odometry, SLAM, structure from motion, visual localisation and augmented reality. The pipeline has three stages. Detect distinctive, repeatable points; describe the neighbourhood of each with a vector that survives changes in viewpoint and lighting; match descriptors and then reject the inevitable wrong matches with a robust estimator that fits a geometric model. This chapter covers classical and learned approaches to each stage, with most attention on RANSAC and its descendants, because robust estimation is where pipelines most often fail.

*See also:* #xref("computer-vision", "image-formation", label: "Image Formation") (classical descriptors), #xref("computer-vision", "multi-view-geometry", label: "Multi-View Geometry and Calibration"), #xref("computer-vision", "3d-vision", label: "3D Vision and Neural Fields"), #xref("search-and-ir", "vector-search", label: "Vector Search") (approximate nearest neighbours)

== Detecting Keypoints

A good keypoint is *repeatable* (found again in another view of the same scene) and *well localised* (its position is precise). Corners satisfy both; points on edges are localised only across the edge, and points in flat regions not at all.

The *Harris detector* formalises this with the *structure tensor* of image gradients $I_x, I_y$ summed over a window $w$:

$ M = sum_w mat(I_x^2, I_x I_y; I_x I_y, I_y^2), quad R = det M - k (op("tr") M)^2 $

Both eigenvalues of $M$ are large at a corner, one at an edge, and neither in a flat region. The response $R$ (with $k approx 0.04$ to $0.06$) avoids computing eigenvalues explicitly; Shi and Tomasi's variant uses the smaller eigenvalue directly and is what KLT tracking uses.

Corners are not scale invariant: a corner at one zoom level is a curve at another. *Scale-space* detectors find extrema of the difference of Gaussians across scales (SIFT) or of the determinant of the Hessian (SURF), returning a position, a scale and an orientation for each point. *FAST* takes the opposite trade-off, testing whether a contiguous arc of pixels on a 16-pixel circle is all brighter or all darker than the centre. It needs only comparisons, runs at hundreds of frames per second on a CPU, and is the detector inside ORB.

== Describing Keypoints

- *SIFT* (Lowe, 2004) divides a scale- and orientation-normalised patch into a $4 times 4$ grid and builds an 8-bin gradient-orientation histogram in each cell, giving a 128-dimensional vector. It is robust to viewpoint change up to about 50 degrees and to illumination change, and remains a strong baseline two decades later.
- *ORB* (Rublee et al., 2011) combines oriented FAST with *rotated BRIEF*: 256 pairwise intensity comparisons in a smoothed patch give a 256-bit binary string. Comparing two descriptors is a XOR and a population count, which a modern CPU does in a few nanoseconds. ORB is the descriptor in ORB-SLAM and many embedded pipelines.
- *Learned descriptors* train a network to produce embeddings in which matching patches are close. *SuperPoint* (DeTone et al., 2018) detects and describes in one fully convolutional pass, trained by self-supervision on synthetic shapes and then on homographically warped real images. DISK and ALIKED improve on it with reinforcement-style and deformable-convolution training respectively.

== Matching Descriptors

The baseline is *nearest-neighbour* search: for each descriptor in image A, find the closest in image B. Two filters remove most false matches cheaply:

- *Lowe's ratio test*: accept a match only if the nearest neighbour is clearly closer than the second nearest, $d_1 < tau d_2$ with $tau$ between 0.7 and 0.8. A repetitive pattern (windows on a façade) yields two similar candidates and fails the test, which is exactly the ambiguity that leads to wrong matches.
- *Mutual nearest neighbours*: keep a pair only if each descriptor is the other's nearest neighbour.

```cpp
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

using orb_descriptor = std::array<std::uint64_t, 4>;   // 256 bits

inline int hamming_distance(const orb_descriptor& a, const orb_descriptor& b) {
    int d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        d += std::popcount(a[i] ^ b[i]);   // one POPCNT instruction per word
    }
    return d;
}

struct feature_match {
    int query_index;
    int train_index;
    int distance;
};

// Brute-force matching with Lowe's ratio test. O(|query| * |train|),
// fast enough for a few thousand binary descriptors per image.
std::vector<feature_match> match_with_ratio_test(std::span<const orb_descriptor> query,
                                                 std::span<const orb_descriptor> train,
                                                 double ratio = 0.8) {
    std::vector<feature_match> matches;
    for (std::size_t q = 0; q < query.size(); ++q) {
        int best = std::numeric_limits<int>::max();
        int second = std::numeric_limits<int>::max();
        int best_index = -1;
        for (std::size_t t = 0; t < train.size(); ++t) {
            const int d = hamming_distance(query[q], train[t]);
            if (d < best) {
                second = best;
                best = d;
                best_index = static_cast<int>(t);
            } else if (d < second) {
                second = d;
            }
        }
        if (best_index >= 0 && best < ratio * second) {
            matches.push_back({static_cast<int>(q), best_index, best});
        }
    }
    return matches;
}
```

With 2000 descriptors per image, the brute-force loop computes 4 million distances of four `POPCNT`s each, a few milliseconds on one core and trivially vectorisable. For large databases (visual localisation against millions of map points), approximate nearest-neighbour indexes take over: k-d trees and hierarchical k-means for float descriptors, multi-index hashing for binary ones, and the same graph indexes used for text embeddings.

=== Learned Matchers

Nearest-neighbour matching treats every keypoint independently. *SuperGlue* (Sarlin et al., 2020) instead treats matching as a graph problem: a transformer with self-attention (within an image) and cross-attention (between images) refines all descriptors jointly, and a differentiable optimal-transport layer (Sinkhorn iterations) produces a soft assignment with an explicit "no match" bin. *LightGlue* (Lindenberger et al., 2023) makes the same idea adaptive, stopping early on easy image pairs and pruning keypoints that are clearly unmatchable, which makes it several times faster.

*Detector-free* methods skip keypoints altogether. *LoFTR* (Sun et al., 2021) matches dense transformer features on a coarse grid and refines matches to sub-pixel accuracy, which works on texture-poor surfaces where detectors find nothing. Dense matchers such as RoMa, and *3D-aware* models such as DUSt3R and MASt3R that regress a point map for each image pair directly, blur the boundary between matching and reconstruction.

== Robust Estimation: RANSAC

Even after filtering, a significant fraction of matches is wrong, often 30 to 70 percent on hard image pairs. Least squares is useless here: a single gross outlier can move the solution arbitrarily far. *RANSAC* (random sample consensus; Fischler and Bolles, 1981) inverts the logic:

1. Draw a *minimal sample*: the smallest number of matches $s$ that determines the model (4 for a homography, 5 for an essential matrix, 7 or 8 for a fundamental matrix, 3 for P3P).
2. Fit the model to the sample.
3. Count the *inliers*: matches whose residual (for example, reprojection or symmetric epipolar distance) is below a threshold $epsilon$.
4. Repeat, keeping the model with the most inliers, then refit it to all of its inliers by least squares.

If a fraction $w$ of matches are inliers, a sample is all-inlier with probability $w^s$. To have probability $p$ of drawing at least one all-inlier sample, RANSAC needs

$ N = (log(1 - p)) / (log(1 - w^s)) $

iterations. The dependence on $s$ is steep, which is why minimal solvers matter:

#table(
  columns: 4,
  [*Inlier ratio $w$*], [*$s = 4$ (homography)*], [*$s = 5$ (essential)*], [*$s = 8$ (eight-point)*],
  [0.8], [9], [12], [26],
  [0.5], [72], [146], [1 177],
  [0.3], [567], [1 893], [about 70 000],
)

(Values for $p = 0.99$.) The iteration count is not known in advance because $w$ is not, so practical implementations are *adaptive*: start with a pessimistic $w$, and whenever a model with more inliers is found, raise the estimate of $w$ and lower $N$.

```cpp
#include <cmath>
#include <limits>

// Iterations needed to draw at least one all-inlier minimal sample
// with probability `confidence`.
int ransac_iterations(double inlier_ratio, int sample_size, double confidence = 0.99) {
    const double p_good = std::pow(inlier_ratio, sample_size);
    if (p_good >= 1.0) return 1;
    if (p_good <= std::numeric_limits<double>::min()) return std::numeric_limits<int>::max();
    return static_cast<int>(std::ceil(std::log(1.0 - confidence) / std::log(1.0 - p_good)));
}
```

=== Better RANSACs

Plain RANSAC has known weaknesses, and each has a standard fix. Modern frameworks such as USAC (Raguram et al., 2013) and the implementations in OpenCV and PoseLib combine them:

- *Score, do not count.* MSAC scores inliers by their residuals (truncated quadratic) instead of counting them, preferring models that fit the inliers tightly. *MAGSAC++* (Barath et al., 2020) marginalises over the threshold $epsilon$, removing the most sensitive parameter.
- *Sample smart.* PROSAC samples from the best-ranked matches first (by ratio-test score), often finding a good model in a handful of iterations.
- *Refine locally.* LO-RANSAC runs an inner optimisation each time a new best model is found, because a model fitted to a minimal sample is noisy even when the sample is all inliers.
- *Check degeneracy.* A sample of points on one plane determines a homography but not a fundamental matrix; DEGENSAC detects this.
- *Verify early.* The sequential probability ratio test aborts the scoring of a clearly bad model after testing only a few matches.

Choosing the inlier threshold is still the most important practical decision. It should reflect the actual localisation noise of the keypoints (often 1 to 3 pixels at full resolution) and be expressed in a geometrically meaningful residual, not an algebraic one.

== Putting It Together

A typical calibrated two-view pipeline in 2026:

1. Detect and describe up to a few thousand keypoints per image (SuperPoint or ALIKED on a GPU, ORB on a CPU or $"MCU"$).
2. Match with LightGlue, or with mutual nearest neighbours plus the ratio test.
3. Estimate the essential matrix with a five-point minimal solver inside an adaptive, locally optimised RANSAC with MAGSAC++ scoring.
4. Decompose into rotation and translation direction, triangulate inliers, and pass the result to bundle adjustment.

Benchmarks for the whole pipeline (the Image Matching Challenge, MegaDepth and ScanNet pose estimation) report the *accuracy of the recovered pose*, typically as the area under the curve of angular error, rather than the number of matches. The number of matches is easy to inflate; a correct pose is not.

== Further Reading

Harris, C., Stephens, M. (1988). "A Combined Corner and Edge Detector." Alvey Vision Conference.

Lowe, D. (2004). "Distinctive Image Features from Scale-Invariant Keypoints." International Journal of Computer Vision.

Rublee, E., Rabaud, V., Konolige, K., Bradski, G. (2011). "ORB: An Efficient Alternative to SIFT or SURF." ICCV.

Fischler, M., Bolles, R. (1981). "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography." Communications of the ACM.

DeTone, D., Malisiewicz, T., Rabinovich, A. (2018). "SuperPoint: Self-Supervised Interest Point Detection and Description." CVPR Workshops.

Sarlin, P.-E., DeTone, D., Malisiewicz, T., Rabinovich, A. (2020). "SuperGlue: Learning Feature Matching with Graph Neural Networks." CVPR.

Lindenberger, P., Sarlin, P.-E., Pollefeys, M. (2023). "LightGlue: Local Feature Matching at Light Speed." ICCV.

Sun, J., Shen, Z., Wang, Y., Bao, H., Zhou, X. (2021). "LoFTR: Detector-Free Local Feature Matching with Transformers." CVPR.

Raguram, R., Chum, O., Pollefeys, M., Matas, J., Frahm, J.-M. (2013). "USAC: A Universal Framework for Random Sample Consensus." IEEE Transactions on Pattern Analysis and Machine Intelligence.

Barath, D., Noskova, J., Ivashechkin, M., Matas, J. (2020). "MAGSAC++, a Fast, Reliable and Accurate Robust Estimator." CVPR.

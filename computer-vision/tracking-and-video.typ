#import "../template.typ": xref

= Motion, Tracking, and Video Understanding <tracking-and-video>

A video is not a stack of independent images. Consecutive frames are strongly correlated, and the way pixels move between them carries information no single frame has: which objects are moving, how fast, which detections in frame 100 are the same person as in frame 90, and what action is unfolding. This chapter covers the three layers of temporal vision: *optical flow* (per-pixel motion), *object tracking* (following one or many objects through time, with most attention on the tracking-by-detection pipelines used in practice), and *video understanding* (recognising actions and events).

*See also:* #xref("computer-vision", "object-detection", label: "Object Detection"), #xref("computer-vision", "image-segmentation", label: "Image Segmentation") (video object segmentation), #xref("computer-vision", "vision-transformers", label: "Vision Transformers"), #xref("computer-vision", "visual-slam", label: "Visual Odometry and SLAM") (camera motion)

== Optical Flow

Optical flow assigns each pixel a 2D displacement $(u, v)$ to its position in the next frame. The classical starting point is *brightness constancy*: a point keeps its intensity as it moves, $I(x + u, y + v, t + 1) = I(x, y, t)$. A first-order Taylor expansion gives the *optical flow constraint*

$ I_x u + I_y v + I_t = 0 $

one equation in two unknowns per pixel. Only the component of motion along the image gradient is determined, the *aperture problem*: through a small window, a moving straight edge looks the same whatever its motion along the edge. Every flow method adds an assumption to resolve it.

*Lucas-Kanade* (1981) assumes the flow is constant in a small window and solves the resulting overdetermined system by least squares:

$ mat(sum I_x^2, sum I_x I_y; sum I_x I_y, sum I_y^2) mat(u; v) = - mat(sum I_x I_t; sum I_y I_t) $

The matrix is the structure tensor from corner detection, so Lucas-Kanade is well conditioned exactly at corners, which is why the KLT tracker pairs it with the Shi-Tomasi detector. The Taylor expansion holds only for small motions, so it runs coarse to fine on an image pyramid, using each level's estimate to warp the next. *Horn-Schunck* (1981) instead adds a global smoothness term, giving dense flow at the cost of blurring motion boundaries.

Learned methods now dominate. *RAFT* (Teed and Deng, 2020) computes features for both frames, builds a 4D *all-pairs correlation volume* (the similarity of every pixel in frame 1 with every pixel in frame 2, pooled at several scales), and refines a single high-resolution flow field with a recurrent unit that repeatedly looks up the correlation volume around its current estimate. That design handles large displacements and generalises well from synthetic training data. Later models add transformers, multi-frame context and better efficiency.

Flow is evaluated by *endpoint error* (EPE), the mean Euclidean distance between estimated and true flow vectors, on benchmarks such as MPI Sintel (rendered film with exact ground truth) and KITTI (driving, ground truth from lidar). Flow remains hard at occlusions (pixels that disappear have no correct match), in textureless regions, and under motion blur.

== Single-Object Tracking

*Single-object tracking* follows one target, specified by a box in the first frame, with no knowledge of its class. Three generations of methods:

- *Correlation filters* (MOSSE, 2010; KCF, 2015) learn a linear filter whose response peaks on the target, trained and evaluated in the Fourier domain, where correlation becomes element-wise multiplication. They run at hundreds of frames per second on a CPU and update online as the target changes appearance.
- *Siamese networks* (SiamFC, SiamRPN) embed the template and a search region with the same network and cross-correlate the embeddings. They are trained offline on large video datasets and need no online learning.
- *Transformer trackers* (OSTrack, MixFormer) process template and search tokens jointly with attention, and video segmentation models such as SAM 2 track masks rather than boxes with a memory of past frames.

The hard cases are occlusion, similar distractors, fast motion and large appearance change, and robust trackers combine a re-detection mechanism with confidence estimation so they can report "lost" instead of drifting onto the background.

== Multi-Object Tracking

*Multi-object tracking* (MOT) must find every object of given classes (people, vehicles) in every frame and keep a consistent identity for each through time. Almost all practical systems follow *tracking by detection*:

1. Run a detector on each frame (#xref("computer-vision", "object-detection", label: "Object Detection")).
2. Predict where each existing track should be now, usually with a Kalman filter.
3. *Associate* detections with predicted tracks by solving an assignment problem.
4. Update matched tracks, start tentative tracks for unmatched detections, and delete tracks unmatched for too long.

=== Motion Prediction with a Kalman Filter

SORT (Bewley et al., 2016) models each track's box with a constant-velocity Kalman filter. A minimal version over the box centre, with state $(x, y, dot(x), dot(y))$:

```cpp
#include <Eigen/Dense>

class constant_velocity_track {
public:
    using state_vector = Eigen::Vector4d;           // x, y, vx, vy (pixels, pixels/frame)
    using state_matrix = Eigen::Matrix4d;

    explicit constant_velocity_track(const Eigen::Vector2d& centre) {
        x_ << centre.x(), centre.y(), 0.0, 0.0;
        p_ = state_matrix::Identity();
        p_(2, 2) = p_(3, 3) = 100.0;                // velocity unknown at birth
    }

    void predict() {
        state_matrix f = state_matrix::Identity();
        f(0, 2) = f(1, 3) = 1.0;                    // x += vx, y += vy
        x_ = f * x_;
        p_ = f * p_ * f.transpose() + process_noise();
        ++frames_since_update_;
    }

    void update(const Eigen::Vector2d& measured_centre) {
        Eigen::Matrix<double, 2, 4> h = Eigen::Matrix<double, 2, 4>::Zero();
        h(0, 0) = h(1, 1) = 1.0;                    // we observe position only
        const Eigen::Matrix2d r = Eigen::Matrix2d::Identity() * measurement_variance;

        const Eigen::Vector2d innovation = measured_centre - h * x_;
        const Eigen::Matrix2d s = h * p_ * h.transpose() + r;
        const Eigen::Matrix<double, 4, 2> k = p_ * h.transpose() * s.inverse();
        x_ += k * innovation;
        p_ = (state_matrix::Identity() - k * h) * p_;
        frames_since_update_ = 0;
    }

    Eigen::Vector2d predicted_centre() const { return x_.head<2>(); }
    int frames_since_update() const { return frames_since_update_; }

private:
    static constexpr double measurement_variance = 4.0;    // (2 px)^2
    static state_matrix process_noise() {
        state_matrix q = state_matrix::Identity() * 0.01;
        q(2, 2) = q(3, 3) = 0.1;                          // velocity may drift
        return q;
    }

    state_vector x_;
    state_matrix p_;
    int frames_since_update_ = 0;
};
```

Real trackers also filter box width and height (or area and aspect ratio), and tune the noise matrices to the frame rate and the objects' dynamics.

=== Association

Association builds a cost matrix between predicted tracks and detections and solves it as a *linear assignment problem* with the Hungarian algorithm ($O(n^3)$, negligible for the tens of objects in a typical frame), rejecting pairs whose cost exceeds a gate. The choice of cost defines the tracker:

- *SORT*: $1 - "IoU"$ between predicted and detected boxes. Fast and simple, but it swaps identities whenever objects cross or are occluded, because nothing distinguishes them except position.
- *DeepSORT* (Wojke et al., 2017): adds an appearance embedding from a re-identification network, compared by cosine distance against a gallery of each track's recent embeddings, and gates by the Mahalanobis distance of the Kalman prediction. Tracks survive occlusions of a few seconds.
- *ByteTrack* (Zhang et al., 2022): the key observation is that low-confidence detections are often real objects that are partially occluded. It first associates high-confidence detections, then tries to match the remaining tracks to *low-confidence* detections instead of discarding them. This simple change recovered many fragmented tracks and set a new state of the art with motion cues alone.
- *OC-SORT* and its successors correct the Kalman filter's drift during occlusion using observations rather than its own predictions, which matters for non-linear motion such as dancers or players.

*End-to-end* trackers (MOTR and similar transformer models) carry track queries from frame to frame and avoid hand-tuned association, but tracking by detection remains dominant in deployed systems because each component can be tuned, replaced and debugged separately.

=== Measuring Tracking Quality

Tracking metrics weigh detection and association differently, and the choice changes rankings:

- *MOTA* combines false positives, misses and identity switches into one number. It is dominated by detection quality and barely penalises poor association.
- *IDF1* measures the fraction of detections assigned the correct identity over the whole sequence, emphasising association.
- *HOTA* (Luiten et al., 2021) is the geometric mean of a detection accuracy and an association accuracy term, averaged over localisation thresholds, and is now the primary metric on MOTChallenge and related benchmarks.

== Video Understanding

*Action recognition* classifies what happens in a clip ("opening a door", "serving a tennis ball"). Architectures have followed the image world with a temporal twist:

- *Two-stream networks* process RGB frames and stacked optical flow separately and fuse them, the flow stream supplying motion explicitly.
- *3D convolutions* (C3D, I3D) extend kernels over time. I3D (Carreira and Zisserman, 2017) inflates ImageNet-pretrained 2D filters into 3D, and the accompanying Kinetics dataset made large-scale video pretraining possible.
- *SlowFast* (Feichtenhofer et al., 2019) runs a slow pathway at low frame rate with many channels for appearance and a fast, thin pathway at high frame rate for motion.
- *Video transformers* (TimeSformer, ViViT, MViT) factorise attention into spatial and temporal parts to keep cost manageable. *VideoMAE* (Tong et al., 2022) pretrains them by masking 90 percent or more of the space-time patches and reconstructing them; the very high masking ratio works because video is so redundant.

Beyond classification, *temporal action localisation* finds the start and end of each action in untrimmed video, and *spatio-temporal detection* also locates the actor. Vision-language models now handle much of open-ended video understanding (question answering, captioning, retrieval) by encoding sampled frames and passing the tokens to a language model, with long videos the main open problem because the number of tokens grows with duration.

=== The Cost of Video

Video models are expensive because a ten-second clip at 30 frames per second is 300 images. Every practical system reduces the input:

- *Sample frames sparsely*: 8 to 32 frames per clip, uniformly or with a learned selector.
- *Reuse computation across frames*: process keyframes fully and propagate features to the frames between them with flow or cheap updates.
- *Use the codec's work*: compressed video already contains motion vectors and residuals, which some models consume directly instead of recomputing motion.
- *Run cascades*: a cheap model decides whether anything of interest happens before an expensive model looks closely.

== Further Reading

Lucas, B., Kanade, T. (1981). "An Iterative Image Registration Technique with an Application to Stereo Vision." IJCAI.

Horn, B., Schunck, B. (1981). "Determining Optical Flow." Artificial Intelligence.

Teed, Z., Deng, J. (2020). "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow." ECCV.

Bolme, D., Beveridge, J., Draper, B., Lui, Y. (2010). "Visual Object Tracking using Adaptive Correlation Filters." CVPR.

Bewley, A., Ge, Z., Ott, L., Ramos, F., Upcroft, B. (2016). "Simple Online and Realtime Tracking." ICIP.

Wojke, N., Bewley, A., Paulus, D. (2017). "Simple Online and Realtime Tracking with a Deep Association Metric." ICIP.

Zhang, Y. et al. (2022). "ByteTrack: Multi-Object Tracking by Associating Every Detection Box." ECCV.

Luiten, J. et al. (2021). "HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking." International Journal of Computer Vision.

Carreira, J., Zisserman, A. (2017). "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset." CVPR.

Feichtenhofer, C., Fan, H., Malik, J., He, K. (2019). "SlowFast Networks for Video Recognition." ICCV.

Tong, Z., Song, Y., Wang, J., Wang, L. (2022). "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training." NeurIPS.

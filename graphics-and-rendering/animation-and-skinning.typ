#import "../template.typ": xref

= Character Animation and Skinning <animation-and-skinning>

A rendered character is a mesh of tens of thousands of vertices, but it is animated through a skeleton of a few dozen to a few hundred joints. Animation decides how the joints move: from authored clips, motion capture, blends of both, procedural adjustments such as inverse kinematics, and increasingly from learned models. Skinning decides how the mesh follows the joints. This chapter covers skeletal hierarchies, linear blend skinning and its artefacts, dual-quaternion skinning, blend shapes for faces, animation clips and their compression, blending and state machines, motion matching, inverse kinematics, GPU skinning, and physics-driven secondary motion.

*See also:* #xref("graphics-and-rendering", "transforms-and-cameras", label: "Transforms, Projections, and Cameras") (matrices and quaternions), #xref("graphics-and-rendering", "realtime-engines", label: "Real-Time Engine Architecture"), #xref("graphics-and-rendering", "ray-tracing", label: "Ray Tracing") (refitting acceleration structures), #xref("numerical-computing", "ode-integration", label: "ODE Integration") (physics)

== Skeletons and Poses

A *skeleton* is a tree of *joints* (often called bones). Each joint stores its *local transform* relative to its parent, as translation, rotation (a quaternion) and scale. A joint's *model-space* transform is the product of the local transforms along the path from the root:

$ G_j = G_("parent"(j)) L_j $

Storing joints in an order where every parent precedes its children (a topological order) lets one forward loop compute all global transforms.

The mesh is modelled in a *bind pose* (often a T-pose or A-pose) in which each joint has model-space transform $B_j$. To deform a vertex by joint $j$, first move it into the joint's local frame in the bind pose with the *inverse bind matrix* $B_j^(-1)$, then out again with the joint's current transform $G_j$. The product $S_j = G_j B_j^(-1)$ is the *skinning matrix*: identity when the character is in its bind pose.

== Linear Blend Skinning

Most vertices near a joint should move partly with each of several joints (a vertex at the elbow follows both the upper arm and the forearm). *Linear blend skinning* (LBS), also called smooth skinning, gives each vertex up to four (sometimes eight) joint influences with weights $w_i$ summing to one, and blends the transformed positions:

$ bold(v)' = sum_i w_i S_(j_i) bold(v) $

```cpp
#include <array>
#include <cstdint>
#include <span>
#include <vector>

struct vec3 {
    float x, y, z;
};

// Affine transform stored as the top 3 rows of a 4x4 matrix (row-major).
struct mat3x4 {
    std::array<float, 12> m;

    vec3 transform_point(const vec3& p) const {
        return {m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3],
                m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
                m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]};
    }
};

struct skinned_vertex {
    vec3 position;                       // bind-pose position
    std::array<std::uint16_t, 4> joints;
    std::array<float, 4> weights;        // sum to 1
};

// skinning_matrices[j] = global_pose[j] * inverse_bind[j], computed once per frame.
void skin_lbs(std::span<const skinned_vertex> vertices,
              std::span<const mat3x4> skinning_matrices,
              std::vector<vec3>& out) {
    out.resize(vertices.size());
    for (std::size_t v = 0; v < vertices.size(); ++v) {
        const skinned_vertex& sv = vertices[v];
        // Blend the matrices, then transform once: equivalent to blending
        // the transformed points, and cheaper with four influences.
        mat3x4 blended{};
        for (int i = 0; i < 4; ++i) {
            const mat3x4& s = skinning_matrices[sv.joints[i]];
            for (int k = 0; k < 12; ++k) blended.m[k] += sv.weights[i] * s.m[k];
        }
        out[v] = blended.transform_point(sv.position);
    }
}
```

LBS is fast and works everywhere, but blending matrices linearly does not produce a rotation. When two influences rotate far apart, the blended matrix shrinks the geometry between them:

- *Candy-wrapper collapse*: twisting a forearm by 180 degrees pinches the wrist to a point.
- *Volume loss* at elbows and knees bent sharply, where the inside of the joint deflates.

Riggers work around these with extra *helper joints* (twist bones that share the twist along the forearm, corrective joints at the elbow), and with corrective blend shapes triggered by joint angles.

=== Normals and Tangents

Skinned normals and tangents must be transformed too. Using the blended matrix's upper $3 times 3$ part is correct only without non-uniform scale; with it, the inverse transpose of the blended matrix is needed (#xref("graphics-and-rendering", "transforms-and-cameras", label: "Transforms, Projections, and Cameras")). Most engines forbid non-uniform scale on skinned joints, partly for this reason.

== Dual-Quaternion Skinning

A *dual quaternion* encodes a rigid transform (rotation plus translation) in eight numbers. *Dual-quaternion skinning* (DQS; Kavan et al., 2008) blends the joints' dual quaternions linearly and then normalises the result, which always yields a rigid transform. Twisting no longer collapses the wrist, and bent joints keep their volume.

DQS has its own artefact: a slight *bulge* on the outside of bent joints. It also cannot represent scale, which matters for cartoon-style squash and stretch. Production systems therefore use LBS for some joints and DQS for others, or more sophisticated methods: *optimised centres of rotation* (Le and Hodgins, 2016) precompute a per-vertex rotation centre and avoid both artefacts at a cost close to LBS. *Delta Mush* smooths the skinned mesh and re-adds detail stored in local frames, cleaning up skinning without manual correctives.

== Blend Shapes

Faces deform in ways a skeleton expresses poorly: lips roll, cheeks puff, brows furrow. *Blend shapes* (morph targets) store a set of sculpted target meshes as per-vertex offsets $bold(Delta)_k$ from the neutral face, and animate weights $alpha_k$:

$ bold(v) = bold(v)_0 + sum_k alpha_k bold(Delta)_k $

Facial rigs commonly follow the *Facial Action Coding System* (FACS), with targets corresponding to individual muscle actions, plus corrective shapes that activate when combinations of targets are used together. A film-quality face has hundreds of targets. Because most targets move only a small region of the face, deltas are stored sparsely, and on the GPU the blend runs as a compute pass before skinning. Blend shapes are also the standard output of facial capture: video or depth sensors are solved into blend-shape weights each frame.

== Animation Clips

An *animation clip* stores, for each joint, curves of translation, rotation and scale over time. Sampling a clip at time $t$ interpolates between keys (linearly for translation and scale, nlerp or slerp for rotation). Raw motion capture at 60 to 120 frames per second for 100 joints is large, and a game may ship tens of thousands of clips, so clips are compressed:

- *Key reduction*: remove keys that interpolation can reproduce within an error tolerance.
- *Quantisation*: store values in 16 bits or fewer per component relative to each track's range; rotations are stored as three components of a normalised quaternion with the fourth reconstructed.
- *Error measured in the right space*: a small rotation error at the hip moves the hand far more than the same error at the wrist. Good compressors (the Animation Compression Library, for example) measure the error on virtual vertices at a distance from each joint, in model space, which propagates errors through the hierarchy as the viewer would see them.

=== Blending and State Machines

Characters rarely play one clip at a time:

- *Cross-fades* blend from one clip to another over a fraction of a second to hide the transition.
- *Blend spaces* interpolate between clips by parameters: walk and run clips blended by speed, with left and right strafes blended by direction. Clips must be *synchronised* (aligned by foot contacts) so that blending a walk and a run does not produce a stumble.
- *Additive animations* store a difference from a reference pose (a breathing cycle, a recoil, a lean) and add it on top of whatever else is playing.
- *Layers and masks* play different clips on different parts of the body: running legs with a reloading upper body.

An *animation state machine* (or blend tree graph) organises these: states such as idle, walk, run and jump, with conditions and blend times for transitions. Large games accumulate state machines with thousands of states and transitions, which are hard to author and maintain.

=== Root Motion

Locomotion clips are captured with the character actually moving. Either the game moves the character's capsule and the clip is played in place (feet may slide when the speeds disagree), or the clip's *root motion* drives the capsule (movement matches the animation exactly, but gameplay responsiveness depends on the animation). Many games mix the two, using root motion for actions such as vaults and attacks and procedural movement for steering.

== Motion Matching

*Motion matching* (popularised by Ubisoft; Clavet, 2016) replaces much of the state machine with search. A large database of unstructured motion capture is indexed by a feature vector per frame: positions and velocities of the feet and hips, and the future trajectory of the root over the next second. Several times a second the game builds a query from the current pose and the desired trajectory (from player input) and jumps to the database frame whose features are nearest, blending briefly to hide the jump. The result is responsive, natural motion without hand-built transitions, at the cost of large memory and careful feature weighting. *Learned motion matching* (Holden et al., 2020) replaces the database with small neural networks that compress it, cutting memory by an order of magnitude.

== Inverse Kinematics

*Forward kinematics* computes where the hand is from the joint angles. *Inverse kinematics* (IK) solves for joint angles that put the hand (or foot, or head) at a target, for foot placement on uneven ground, hands on weapons and ledges, and looking at points of interest.

For a two-bone chain (upper arm and forearm, or thigh and shin) the solution is analytic. With bone lengths $a$ and $b$ and distance $d$ from the root to the target, clamped to the reachable range $[|a - b|, a + b]$, the law of cosines gives the elbow's interior angle and the angle between the upper bone and the root-to-target direction:

```cpp
#include <algorithm>
#include <cmath>

struct two_bone_angles {
    float root_offset;     // angle between upper bone and root-to-target line
    float middle_interior; // interior angle at the middle joint (pi = straight)
};

two_bone_angles solve_two_bone(float upper_length, float lower_length, float target_distance) {
    const float a = upper_length;
    const float b = lower_length;
    const float d = std::clamp(target_distance, std::abs(a - b) + 1e-5f, a + b - 1e-5f);
    const float cos_mid = (a * a + b * b - d * d) / (2.0f * a * b);
    const float cos_root = (a * a + d * d - b * b) / (2.0f * a * d);
    return {std::acos(std::clamp(cos_root, -1.0f, 1.0f)),
            std::acos(std::clamp(cos_mid, -1.0f, 1.0f))};
}
```

A *pole vector* (where the knee or elbow should point) fixes the remaining rotation of the chain's plane about the root-to-target axis. Longer chains (spines, tails, tentacles) use iterative solvers: *cyclic coordinate descent* (CCD) rotates one joint at a time to point the end at the target, and *FABRIK* (Aristidou and Lasenby, 2011) alternately drags the chain from the target and from the root while preserving bone lengths. Both converge in a few iterations and accept joint limits. Full-body IK solvers balance many goals at once (both feet planted, both hands on an object, centre of mass over the feet) for interaction with the environment.

== GPU Skinning

Skinning thousands of vertices per character for dozens of characters belongs on the GPU. Two approaches:

- *Vertex-shader skinning*: the vertex shader reads the skinning matrices from a buffer and transforms each vertex as it is drawn. No extra memory, but the work repeats for every pass that draws the mesh (depth prepass, shadow cascades, main pass, motion vectors).
- *Compute pre-skinning*: a compute pass skins every vertex once per frame into a buffer, which all passes then read as a static mesh. This saves repeated work and is required for ray tracing, where the acceleration structure must be refitted or rebuilt from the deformed positions each frame. Previous-frame positions are kept for motion vectors.

Upload only the skinning matrices (or dual quaternions) per frame, store them as $3 times 4$ to save bandwidth, and use level of detail: distant characters use fewer bones and influences, update at lower rates, or switch to baked vertex-animation textures for crowds.

== Physics and Secondary Motion

Animation clips cannot react to collisions or produce every secondary motion. Physics fills the gap:

- *Ragdolls*: on death or impact, the skeleton is handed to a rigid-body simulation with joint limits, often blended with animation (*powered ragdolls* driven towards an animated pose by motors) so that characters react physically while staying controlled.
- *Secondary motion*: hair, ears, tails, jiggle and accessories driven by simple spring-damper chains attached to the skeleton.
- *Cloth*: capes and clothing simulated with *position-based dynamics* (Müller et al., 2007) or its extension *XPBD* (Macklin, Müller and Chentanez, 2016), which handle stiff constraints robustly at game time steps and run well on GPUs.

The general pattern in modern engines is a pipeline per character: sample and blend animation, apply procedural adjustments and IK, run physics for secondary parts, then skin.

== Further Reading

Magnenat-Thalmann, N., Laperrière, R., Thalmann, D. (1988). "Joint-Dependent Local Deformations for Hand Animation and Object Grasping." Graphics Interface.

Lewis, J., Cordner, M., Fong, N. (2000). "Pose Space Deformation: A Unified Approach to Shape Interpolation and Skeleton-Driven Deformation." SIGGRAPH.

Kavan, L., Collins, S., Žára, J., O'Sullivan, C. (2008). "Geometric Skinning with Approximate Dual Quaternion Blending." ACM Transactions on Graphics.

Le, B., Hodgins, J. (2016). "Real-time Skeletal Skinning with Optimized Centers of Rotation." ACM Transactions on Graphics.

Aristidou, A., Lasenby, J. (2011). "FABRIK: A Fast, Iterative Solver for the Inverse Kinematics Problem." Graphical Models.

Clavet, S. (2016). "Motion Matching and The Road to Next-Gen Animation." Game Developers Conference.

Holden, D., Kanoun, O., Perepichka, M., Popa, T. (2020). "Learned Motion Matching." ACM Transactions on Graphics.

Müller, M., Heidelberger, B., Hennix, M., Ratcliff, J. (2007). "Position Based Dynamics." Journal of Visual Communication and Image Representation.

Macklin, M., Müller, M., Chentanez, N. (2016). "XPBD: Position-Based Simulation of Compliant Constrained Dynamics." Motion in Games.

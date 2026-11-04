#import "../template.typ": xref

= Transforms, Projections, and Cameras <transforms-and-cameras>

Every vertex a GPU draws passes through a chain of coordinate transforms before it lands on a pixel, and every one of them hides a convention that differs between APIs, engines and textbooks. Getting them wrong produces mirrored scenes, inside-out lighting, z-fighting at a distance, and objects that jitter when the player walks far from the origin. This chapter builds the chain from object space to the screen, explains why normals need a different matrix, compares the ways to represent rotation, derives the perspective projection and its depth precision (and why reverse-Z fixes it), and covers the precision problems of large worlds.

*See also:* #xref("graphics-and-rendering", "rasterization-pipeline", label: "The Rasterization Pipeline"), #xref("graphics-and-rendering", "shaders", label: "Shaders and Shading Languages"), #xref("numerical-computing", "floating-point", label: "Floating Point"), #xref("computer-vision", "multi-view-geometry", label: "Multi-View Geometry and Calibration") (the same camera, seen from vision)

== The Coordinate Space Chain

A vertex position goes through these spaces:

1. *Object (model) space*: coordinates as authored, relative to the mesh's own origin.
2. *World space*: after the *model matrix* $M$ places the object in the scene.
3. *View (camera) space*: after the *view matrix* $V$, the inverse of the camera's world transform, puts the camera at the origin looking along a fixed axis.
4. *Clip space*: after the *projection matrix* $P$, a 4D homogeneous position. Clipping against the view volume happens here.
5. *Normalised device coordinates* (NDC): after the *perspective divide* by $w$, a cube or box with fixed bounds.
6. *Window (screen) space*: after the *viewport transform* maps NDC to pixels and to the depth range.

The vertex shader typically computes $bold(p)_"clip" = P V M bold(p)_"object"$, often with $P V$ or $P V M$ premultiplied on the CPU. The fixed-function hardware does the divide and viewport transform.

=== Conventions That Differ

#table(
  columns: 4,
  [*Convention*], [*OpenGL*], [*Direct3D*], [*Vulkan*],
  [NDC depth range], [$-1$ to $1$ (configurable to 0..1 since 4.5)], [$0$ to $1$], [$0$ to $1$],
  [NDC $y$ axis], [Up], [Up], [Down],
  [Framebuffer origin], [Bottom left], [Top left], [Top left],
  [Traditional view-space handedness], [Right-handed, looking down $-z$], [Left-handed, looking down $+z$], [Either, by matrix choice],
)

Handedness is a property of the matrices you write, not of the API, but samples and math libraries pick one and the projection matrix must match it. Vulkan's downward NDC $y$ is handled either by negating the $y$ scale in the projection matrix or by a negative viewport height. Mixing conventions mirrors the image and flips triangle winding, so back-face culling suddenly removes the wrong faces.

== Homogeneous Coordinates and Affine Transforms

A 3D point is written $(x, y, z, 1)$ and a direction $(x, y, z, 0)$. With $4 times 4$ matrices, translation becomes a matrix multiply, so translation, rotation and scale compose into one matrix, and directions are automatically unaffected by translation. An affine transform has the block form

$ M = mat(A, bold(t); bold(0)^top, 1) $

with $A$ a $3 times 3$ linear part. Composition order matters: $T R S$ (scale, then rotate, then translate, reading right to left for column vectors) is the usual order for placing an object. Engines store separate translation, rotation and scale components and build the matrix when needed, because decomposing a general matrix back into those components is ambiguous when shear or negative scale is present.

=== Why Normals Need the Inverse Transpose

A surface normal is not a direction along the surface but perpendicular to it. If a tangent $bold(t)$ transforms by $A$, the transformed normal $bold(n)'$ must stay perpendicular to it: $bold(n)'^top A bold(t) = 0$ for every $bold(t)$ with $bold(n)^top bold(t) = 0$. Choosing $bold(n)' = (A^(-1))^top bold(n)$ satisfies this, since $bold(n)^top A^(-1) A bold(t) = bold(n)^top bold(t) = 0$. For pure rotations $(A^(-1))^top = A$, which is why the bug of transforming normals with the model matrix only appears once non-uniform scale is introduced: a sphere scaled into an ellipsoid then has normals that point the wrong way and lights the wrong side. Renormalise afterwards, because the inverse transpose does not preserve length. When $A$ is a rotation times a uniform scale, the cofactor matrix (the inverse transpose times the determinant) gives the same direction without an inverse.

== Representing Rotation

#table(
  columns: 4,
  [*Representation*], [*Size*], [*Strengths*], [*Weaknesses*],
  [Euler angles], [3], [Intuitive for editing; compact], [Gimbal lock; interpolation is not smooth; order conventions vary],
  [Axis-angle / rotation vector], [3 or 4], [Natural for angular velocity and optimisation], [Composition requires conversion],
  [Rotation matrix], [9], [Direct to apply; composes by multiplication], [Drifts from orthonormality under accumulation; poor interpolation],
  [Unit quaternion], [4], [Smooth interpolation; cheap composition; easy renormalisation], [Double cover ($bold(q)$ and $-bold(q)$ are the same rotation); less intuitive],
)

*Gimbal lock* is the loss of a degree of freedom in Euler angles when the middle rotation aligns the first and third axes (pitch at $plus.minus 90$ degrees in yaw-pitch-roll), after which two of the angles rotate about the same axis. It is a property of the parameterisation, not of rotations, and it is why animation and physics systems store quaternions.

A unit quaternion $bold(q) = (cos(theta \/ 2), sin(theta \/ 2) hat(bold(u)))$ represents rotation by $theta$ about axis $hat(bold(u))$, and rotates a vector by $bold(v)' = bold(q) bold(v) bold(q)^*$. *Spherical linear interpolation* (slerp) moves at constant angular velocity along the shortest arc:

```cpp
#include <cmath>

struct quat {
    float w, x, y, z;
};

struct vec3 {
    float x, y, z;
};

inline quat operator*(const quat& a, const quat& b) {
    return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}

// v' = q v q*, expanded to avoid two full quaternion products.
inline vec3 rotate(const quat& q, const vec3& v) {
    const vec3 u{q.x, q.y, q.z};
    const vec3 t{2.0f * (u.y * v.z - u.z * v.y),
                 2.0f * (u.z * v.x - u.x * v.z),
                 2.0f * (u.x * v.y - u.y * v.x)};          // t = 2 (u x v)
    return {v.x + q.w * t.x + (u.y * t.z - u.z * t.y),
            v.y + q.w * t.y + (u.z * t.x - u.x * t.z),
            v.z + q.w * t.z + (u.x * t.y - u.y * t.x)};    // v + w t + u x t
}

inline quat slerp(quat a, quat b, float t) {
    float d = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    if (d < 0.0f) {                     // take the short way round
        b = {-b.w, -b.x, -b.y, -b.z};
        d = -d;
    }
    float wa, wb;
    if (d > 0.9995f) {                  // nearly parallel: lerp, then normalise
        wa = 1.0f - t;
        wb = t;
    } else {
        const float theta = std::acos(d);
        const float s = std::sin(theta);
        wa = std::sin((1.0f - t) * theta) / s;
        wb = std::sin(t * theta) / s;
    }
    quat r{wa * a.w + wb * b.w, wa * a.x + wb * b.x, wa * a.y + wb * b.y, wa * a.z + wb * b.z};
    const float n = std::sqrt(r.w * r.w + r.x * r.x + r.y * r.y + r.z * r.z);
    return {r.w / n, r.x / n, r.y / n, r.z / n};
}
```

The sign flip handles the double cover; without it, interpolating between $bold(q)$ and a nearby $-bold(q)'$ takes the long way round, a visible 360-degree spin. When blending many rotations at once (skeletal animation blend trees), normalised linear interpolation (nlerp) is cheaper, commutative and good enough for small angles.

== Perspective Projection and Depth Precision

A perspective projection maps the view frustum to the NDC box. For a right-handed view space looking down $-z$, vertical field of view $phi$, aspect ratio $a$, and a $[0, 1]$ depth range with near plane $n$ and far plane $f$:

$ P = mat(s \/ a, 0, 0, 0; 0, s, 0, 0; 0, 0, f \/ (n - f), n f \/ (n - f); 0, 0, -1, 0), quad s = 1 / tan(phi \/ 2) $

The last row sets $w_"clip" = -z_"view"$, the distance in front of the camera, and the divide by $w$ produces the perspective foreshortening. The stored depth is

$ d(z) = f / (f - n) (1 - n / z) $

for distance $z$ (positive). It is a hyperbolic function of distance: half of the depth range is used between $n$ and $2n$. With a 24-bit fixed-point depth buffer, $n = 0.1$ m and $f = 10$ km, depth values distinguish surfaces millimetres apart near the camera but only tens of metres apart at the far end, producing *z-fighting*, flickering where coplanar or nearly coplanar surfaces alternate.

=== Reverse-Z

*Reverse-Z* maps the near plane to depth 1 and the far plane to 0, and stores depth as 32-bit floating point. A floating-point number has roughly constant relative precision, with most representable values near zero. The hyperbolic depth function concentrates precision near the camera; reversing it puts the distant region, where depth values crowd together, next to zero, where floats are densest. The two effects nearly cancel, giving close to uniform relative precision over the whole range (Reed, 2015). The far plane can even be moved to infinity:

```cpp
#include <array>
#include <cmath>

using mat4 = std::array<std::array<float, 4>, 4>;   // row-major, column vectors

// Right-handed view space (camera looks down -z), depth range [0, 1],
// reverse-Z with an infinite far plane: depth = near / distance.
// Use a GREATER depth test, clear depth to 0, and a 32-bit float depth buffer.
mat4 perspective_reverse_z_infinite(float fov_y_radians, float aspect, float near_plane) {
    const float s = 1.0f / std::tan(fov_y_radians * 0.5f);
    return {{
        {s / aspect, 0.0f, 0.0f, 0.0f},
        {0.0f, s, 0.0f, 0.0f},          // negate for Vulkan's downward y, or flip the viewport
        {0.0f, 0.0f, 0.0f, near_plane},
        {0.0f, 0.0f, -1.0f, 0.0f},
    }};
}
```

With this matrix, $z_"clip" = n$ and $w_"clip" = -z_"view"$, so the stored depth is $n \/ d$ for distance $d$: exactly 1 at the near plane, approaching 0 at infinity. Three details must change together: the depth test becomes "greater", the depth buffer clears to 0, and the format must be floating point (reverse-Z with a 24-bit integer buffer gains nothing). Every major engine now uses reverse-Z.

=== Other Projection Variations

- *Orthographic projection* has no divide by depth, so depth is linear. It is used for shadow maps from directional lights, 2D and UI rendering, and CAD views.
- *Sub-pixel jitter*: temporal anti-aliasing and upscalers add a different sub-pixel offset to the projection each frame (typically following a Halton sequence) by modifying the third column of $P$, then accumulate the samples over time.
- *Oblique near planes* replace the near plane with an arbitrary clipping plane, which is how planar reflections clip geometry behind the mirror without an extra clip-plane cost.
- *Asymmetric frusta* are needed for VR (each eye's frustum is off-centre) and for tiled or multi-monitor rendering.

== Cameras

The view matrix is the inverse of the camera's world transform. Building it with a "look-at" function from an eye position, a target and an up vector is convenient but degenerates when the view direction is parallel to the up vector (looking straight down). First-person cameras accumulate yaw and pitch (clamping pitch short of 90 degrees) and rebuild the orientation each frame. Orbit and free-flight cameras are best stored as a position plus a quaternion, which avoids gimbal lock and interpolates smoothly for camera animation.

== Large Worlds and Floating-Point Precision

A 32-bit float has a 24-bit significand, so the spacing between representable values grows with magnitude:

#table(
  columns: 2,
  [*Distance from origin*], [*Spacing between adjacent floats*],
  [1 m], [about 0.1 µm],
  [1 km], [about 0.06 mm],
  [10 km], [about 1 mm],
  [100 km], [about 8 mm],
  [10 000 km (planet scale)], [about 1 m],
)

At 10 km, vertex positions snap to a millimetre grid, and the error is amplified when large world-space positions are multiplied by matrices and subtracted: a character's hands visibly jitter, shadows swim, and physics contacts misbehave. The fixes, in increasing robustness:

- *Camera-relative rendering*: subtract the camera position from object positions (in double precision on the CPU) before building model matrices, so that everything near the camera has small coordinates. The view matrix then has no translation.
- *Origin rebasing*: periodically shift the whole world so that the player stays near the origin. Physics engines and networking must agree on the shift.
- *Double precision on the CPU* for world positions (Unreal Engine 5's Large World Coordinates), with conversion to camera-relative single precision only for the GPU.

== Further Reading

Akenine-Möller, T. et al. (2018). "Real-Time Rendering." CRC Press, 4th ed.

Shoemake, K. (1985). "Animating Rotation with Quaternion Curves." SIGGRAPH.

Lengyel, E. (2016). "Foundations of Game Engine Development, Volume 1: Mathematics." Terathon Software.

Upchurch, P., Desbrun, M. (2012). "Tightening the Precision of Perspective Rendering." Journal of Graphics Tools.

Reed, N. (2015). "Depth Precision Visualized." NVIDIA Developer Blog.

Pharr, M., Jakob, W., Humphreys, G. (2023). "Physically Based Rendering: From Theory to Implementation." MIT Press, 4th ed.

The Khronos Group (2024). "Vulkan 1.3 Specification." Coordinate systems chapter.

#import "../template.typ": xref

= Deploying Vision Models <vision-deployment>

A model that scores well in a notebook is perhaps a third of a working vision product. The rest is a pipeline that decodes frames from a camera or a file, prepares them exactly as the training data was prepared, runs inference within a latency and power budget on the target hardware, turns raw outputs into decisions, and keeps working as the world drifts away from the training set. This chapter follows that pipeline: preprocessing and its silent correctness bugs, model compression (quantisation, pruning, distillation and efficient architectures), runtimes and accelerators, latency engineering, and monitoring in production.

*See also:* #xref("computer-vision", "cnn-architectures", label: "CNN Architectures"), #xref("computer-vision", "object-detection", label: "Object Detection") (post-processing and NMS), #xref("llm", "quantization", label: "Quantization") (LLM counterpart), #xref("gpu-architecture", "tensor-cores-and-mma", label: "Tensor Cores and MMA"), #xref("performance-engineering", "methodology", label: "Methodology")

== The Inference Pipeline

A typical camera-to-decision path has five stages:

1. *Capture and decode*: the camera delivers raw Bayer data processed by an image signal processor, or a compressed stream (H.264, H.265) that must be decoded.
2. *Preprocess*: colour conversion (usually YUV to RGB), resize, crop or pad, normalise, and lay out in the tensor format the model expects.
3. *Inference*.
4. *Post-process*: decode boxes or masks from raw outputs, apply confidence thresholds and non-maximum suppression, map coordinates back to the original image.
5. *Act*: track, count, alert, steer.

On GPUs and NPUs the model itself is often not the bottleneck. A 4K H.265 stream decoded on the CPU, then resized and converted with scalar code, can take longer than the network. Production pipelines keep frames on the accelerator from decode to inference: hardware decoders (NVDEC, VA-API, mobile media codecs) write into GPU memory, resizing and colour conversion run as GPU kernels or in the ISP, and only the final detections cross back to the CPU. NVIDIA DeepStream and GStreamer-based frameworks exist mainly to make that zero-copy path routine.

== Preprocessing: Where Silent Bugs Live

A preprocessing mismatch between training and serving rarely crashes anything. It just costs accuracy, often several points, and is easy to miss because predictions still look plausible. The usual suspects:

- *Channel order*: OpenCV loads BGR; most training pipelines use RGB.
- *Resize method*: training with an anti-aliased resize (PIL, torchvision's default for tensors since version 0.17) and serving with OpenCV's non-anti-aliased bilinear resize produces different pixels after strong downscaling.
- *Aspect ratio*: a model trained on letterboxed (padded) inputs sees distorted objects if serving stretches the image, and vice versa.
- *Normalisation constants*: ImageNet mean and standard deviation, scaling to $[0, 1]$, or $[-1, 1]$, depending on the backbone's original training.
- *Image orientation*: phone photos store rotation in EXIF metadata. Ignoring it feeds the model sideways images.
- *Colour range*: video YUV is usually *limited range* (luma 16 to 235). Converting it as full range washes out contrast.
- *Compression*: a model trained on clean PNGs and served heavily compressed JPEG frames from a security camera has a domain gap of its own.

The robust fix is to make preprocessing a single shared component, tested by feeding the same image through the training and serving paths and comparing the tensors numerically.

```cpp
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>

struct letterbox_transform {
    float scale;   // model pixels per source pixel
    float pad_x;   // left padding in model pixels
    float pad_y;   // top padding in model pixels

    // Map a point from model input coordinates back to the source image.
    std::array<float, 2> to_source(float x, float y) const {
        return {(x - pad_x) / scale, (y - pad_y) / scale};
    }
};

// Letterbox an interleaved RGB8 image into a planar (CHW) float tensor of
// size 3 x target x target, normalised as (v / 255 - mean) / stddev.
// Bilinear sampling with pixel-centre alignment; padding uses the value
// that normalises to zero. Production code runs this on the GPU or with SIMD.
letterbox_transform letterbox_rgb_to_chw(std::span<const std::uint8_t> rgb,
                                         int width, int height, int row_stride,
                                         int target, std::span<float> out,
                                         const std::array<float, 3>& mean,
                                         const std::array<float, 3>& stddev) {
    const float scale = std::min(static_cast<float>(target) / static_cast<float>(width),
                                 static_cast<float>(target) / static_cast<float>(height));
    const float new_w = static_cast<float>(width) * scale;
    const float new_h = static_cast<float>(height) * scale;
    const letterbox_transform t{scale, (target - new_w) / 2.0f, (target - new_h) / 2.0f};
    const std::size_t plane = static_cast<std::size_t>(target) * target;

    for (int y = 0; y < target; ++y) {
        for (int x = 0; x < target; ++x) {
            const std::size_t o = static_cast<std::size_t>(y) * target + x;
            const float sx = (x + 0.5f - t.pad_x) / scale - 0.5f;
            const float sy = (y + 0.5f - t.pad_y) / scale - 0.5f;
            if (sx < -0.5f || sy < -0.5f || sx > width - 0.5f || sy > height - 0.5f) {
                for (int c = 0; c < 3; ++c) out[c * plane + o] = 0.0f;   // padding
                continue;
            }
            const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, width - 1);
            const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, height - 1);
            const int x1 = std::min(x0 + 1, width - 1);
            const int y1 = std::min(y0 + 1, height - 1);
            const float fx = std::clamp(sx - static_cast<float>(x0), 0.0f, 1.0f);
            const float fy = std::clamp(sy - static_cast<float>(y0), 0.0f, 1.0f);
            for (int c = 0; c < 3; ++c) {
                auto px = [&](int xx, int yy) {
                    return static_cast<float>(rgb[static_cast<std::size_t>(yy) * row_stride + xx * 3 + c]);
                };
                const float top = px(x0, y0) * (1 - fx) + px(x1, y0) * fx;
                const float bottom = px(x0, y1) * (1 - fx) + px(x1, y1) * fx;
                const float v = (top * (1 - fy) + bottom * fy) / 255.0f;
                out[c * plane + o] = (v - mean[c]) / stddev[c];
            }
        }
    }
    return t;
}
```

The returned transform is what post-processing needs: detection boxes come out in model coordinates and must be mapped back through the same padding and scale. Getting the half-pixel offsets wrong shifts every box by up to a pixel, which matters for small objects and for any downstream geometry.

== Making Models Smaller and Faster

=== Quantisation

Most vision inference on edge hardware runs in 8-bit integers. Weights and activations are mapped to INT8 with a scale (and possibly a zero point), matrix multiplications accumulate in INT32, and results are rescaled. Compared with FP32, memory traffic falls by 4x and integer units deliver several times the throughput per watt.

- *Post-training quantisation* (PTQ) needs only a few hundred representative images to *calibrate* activation ranges. Per-channel scales for convolution weights are essential; per-tensor weight scales lose accuracy on depthwise convolutions in particular. Calibration by minimising quantisation error (entropy or MSE) beats taking the raw min and max, which outliers inflate.
- *Quantisation-aware training* (QAT) simulates quantisation in the forward pass during fine-tuning (with a straight-through gradient estimator), recovering most of the gap when PTQ is not good enough.

Classification backbones typically lose under one percentage point of accuracy with INT8 PTQ. Detectors and segmenters are more sensitive: box regression heads and the final layers often need higher precision, and mixed precision (INT8 backbone, FP16 head) is a common compromise. FP16 or BF16 is nearly lossless and is the default on GPUs; FP8 is becoming standard on recent data-centre GPUs. Below 8 bits, vision models degrade faster than language models, and 4-bit vision deployment remains niche.

=== Pruning, Distillation, and Efficient Architectures

- *Structured pruning* removes whole channels or blocks, producing a smaller dense network that runs faster on any hardware. Unstructured (individual-weight) sparsity rarely speeds up inference except on hardware with explicit support, such as NVIDIA's 2:4 structured sparsity.
- *Knowledge distillation* trains a small student to match a large teacher's outputs or features, usually recovering a significant part of the accuracy lost by shrinking.
- *Efficient architectures* are designed for the cost model of real hardware: depthwise separable convolutions (MobileNet), compound scaling (EfficientNet), and *structural re-parameterisation* (RepVGG), which trains with multi-branch blocks and folds them into single $3 times 3$ convolutions for inference.

FLOPs are a poor predictor of latency. Depthwise convolutions have few FLOPs but low arithmetic intensity, so they are memory-bound on GPUs, while a plain $3 times 3$ convolution with more FLOPs may run faster. Always measure on the target device; hardware-aware neural architecture search exists precisely because of this gap.

== Runtimes and Accelerators

Training frameworks are not deployment runtimes. The usual path exports the model to a portable graph (ONNX, or a framework-native format such as ExecuTorch's), then compiles it for the target:

#table(
  columns: 3,
  [*Target*], [*Typical runtime*], [*Notes*],
  [NVIDIA GPU, Jetson], [TensorRT], [Layer fusion, kernel auto-tuning, INT8 and FP8; engines are specific to GPU model and version],
  [Intel CPU, GPU, NPU], [OpenVINO], [Graph optimisation and INT8 for x86 and Intel accelerators],
  [Android, embedded Linux], [LiteRT (TensorFlow Lite), ExecuTorch, ONNX Runtime], [Delegates to GPU and vendor NPUs],
  [Apple devices], [Core ML], [Schedules across CPU, GPU and the Neural Engine],
  [Dedicated edge NPUs], [Vendor SDKs (Hailo, Qualcomm, Rockchip, NXP)], [Usually INT8 only; restricted operator sets],
  [Microcontrollers], [TFLite Micro, vendor tools], [Kilobytes of RAM; tiny CNNs for wake-word or presence detection],
)

The recurring practical problem is *operator coverage*. An NPU may support convolutions and common activations but not a particular attention variant, resize mode or post-processing op. Unsupported operators fall back to the CPU, and each fallback copies tensors across the boundary, which can make the "accelerated" model slower than running it entirely on the CPU. Check the compiled graph's partitioning, and design or modify models so that the whole network maps to the accelerator, moving awkward operations into post-processing code.

== Latency Engineering

The right metric depends on the application. A robot or vehicle needs bounded *latency* per frame. A video-analytics server processing hundreds of streams needs *throughput* per dollar. The techniques differ accordingly:

- *Batching* raises throughput on GPUs but adds waiting time; dynamic batching with a maximum delay balances the two.
- *Pipelining*: decode frame $n + 2$, preprocess $n + 1$ and infer $n$ concurrently on separate engines or streams, so that throughput approaches the slowest stage.
- *Resolution*: inference cost scales roughly with pixel count. Running a detector at 640 pixels instead of 1280 is about 4x cheaper; tiling high-resolution frames or a cheap first pass that selects regions recovers small objects where needed.
- *Frame skipping with tracking*: detect every $k$-th frame and track in between (#xref("computer-vision", "tracking-and-video", label: "Motion, Tracking, and Video Understanding")).
- *Tail latency*: report the 99th percentile, not the mean, measured with realistic thermal conditions, because passively cooled edge devices throttle after minutes of sustained load.

== Production: Drift and the Data Flywheel

Deployed models meet data the training set did not cover: new camera models, seasons, lighting, product packaging, a cleaning crew's uniforms. Accuracy degrades silently. Production vision systems therefore need:

- *Monitoring* of input statistics (brightness, blur, embedding distributions) and output statistics (detection counts, confidence histograms), with alerts on shifts.
- *Sampling for review*: send low-confidence and disagreement cases, plus a random sample, for human labelling.
- *A retraining loop* in which reviewed samples enter the training set, a new model is evaluated against a fixed regression suite and the current model, and rollout happens gradually with the ability to roll back.
- *Versioning* of models, preprocessing code and calibration data together, because a model is only correct in combination with the exact preprocessing it was trained with.

Sculley et al. (2015) observed that the model is a small box in a large diagram of data collection, verification, serving and monitoring infrastructure. Vision systems are a vivid case: the camera, the codec, and the world in front of the lens are all part of the system.

== Further Reading

Jacob, B. et al. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." CVPR.

Nagel, M. et al. (2021). "A White Paper on Neural Network Quantization." arXiv:2106.08295.

Howard, A. et al. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." arXiv:1704.04861.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.-C. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR.

Tan, M., Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML.

Ding, X. et al. (2021). "RepVGG: Making VGG-style ConvNets Great Again." CVPR.

Hinton, G., Vinyals, O., Dean, J. (2015). "Distilling the Knowledge in a Neural Network." NeurIPS Deep Learning Workshop.

Sculley, D. et al. (2015). "Hidden Technical Debt in Machine Learning Systems." NeurIPS.

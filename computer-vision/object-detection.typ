= Object Detection

Object detection requires simultaneously localising and classifying every object instance in an image. It is one of the core computer vision tasks, underpinning autonomous driving, robotics, medical imaging, and video understanding. This chapter covers the two-stage and one-stage detector families, anchor-free methods, and the transformer-based DETR family.

*See also:* _CNN Architectures_ (backbones and FPN), _Image Segmentation_ (Mask R-CNN), _3D Vision_ (3D object detection).

== Problem Formulation

The detector predicts a set of tuples $(c_i, b_i, s_i)$: class $c_i in {1,...,C}$, bounding box $b_i = (x, y, w, h)$ (centre, width, height), and confidence score $s_i in [0,1]$.

=== Intersection over Union

Localisation quality is measured by *Intersection over Union (IoU)*:

$ "IoU"(b_"pred", b_"gt") = "Area"(b_"pred" inter b_"gt") / "Area"(b_"pred" union b_"gt") in [0, 1]. $

A detection is a true positive if $"IoU" >= theta$ (typically 0.5).

=== Mean Average Precision

*mAP* is the primary detection metric. For each class, compute the area under the precision-recall curve (AP). mAP averages AP over classes. COCO mAP averages over 10 IoU thresholds from 0.5 to 0.95, denoted $"AP"^{"COCO"}$; VOC mAP uses threshold 0.5.

== Two-Stage Detectors

Two-stage detectors first propose candidate regions, then classify and refine them.

=== R-CNN Family

*R-CNN* (Girshick et al., 2014): (1) Selective search proposes $~2000$ regions per image; (2) warp each to fixed size; (3) extract CNN features; (4) classify with SVMs. Too slow for practical use (47s/image).

*Fast R-CNN* (Girshick, 2015): run CNN once on the full image; extract features for each proposal with *RoI pooling* (warps variable-size RoI to fixed grid by max-pooling). End-to-end training. $~9  times $ speedup.

*Faster R-CNN* (Ren et al., 2015): replace selective search with a *Region Proposal Network (RPN)* sharing the CNN backbone. RPN slides a small network over the feature map and proposes objectness scores and box deltas for $k=9$ anchors (3 scales, 3 aspect ratios) at each location. Total time: 5 fps on VGG-16.

=== Feature Pyramid Network

*FPN* (Lin et al., 2017) builds a multi-scale feature pyramid with lateral connections:
- Bottom-up: standard CNN forward pass, extracting $P_2, ..., P_5$ at strides $4, 8, 16, 32$.
- Top-down: upsample coarser levels, add to finer levels via $1 times 1$ lateral convolutions.

Each pyramid level detects objects at an appropriate scale. FPN is the standard backbone enhancement in Faster R-CNN, Mask R-CNN, and RetinaNet.

=== Cascade R-CNN

Cascade R-CNN (Cai & Vasconcelos, 2019) trains a sequence of detectors with increasing IoU thresholds $(0.5, 0.6, 0.7)$. Each stage's output is the next stage's input. Avoids quality mismatch between training and inference IoU thresholds. Strong baseline; +2–3 AP over Faster R-CNN.

== One-Stage Detectors

One-stage detectors predict boxes and classes directly from the feature map in a single forward pass, trading some accuracy for speed.

=== YOLO Family

*YOLOv1* (Redmon et al., 2016): divide image into $S times S$ grid; each cell predicts $B$ boxes and class probabilities jointly. 45 fps on Titan X.

*YOLOv3*: adds multi-scale predictions at 3 scales using FPN-like feature pyramid. Darknet-53 backbone.

*YOLOv5 / YOLOv8 / YOLO11*: community-maintained architectures with extensive tuning, anchor-free heads, and efficient deployment. YOLOv8 adds decoupled heads (separate classification and regression branches). YOLO11 achieves state-of-the-art speed/accuracy trade-offs on COCO.

=== SSD

SSD (Liu et al., 2016): predict boxes at multiple feature map scales without RPN; default boxes at each location cover multiple aspect ratios. Faster than Faster R-CNN but less accurate on small objects.

=== RetinaNet and Focal Loss

*RetinaNet* (Lin et al., 2017): one-stage detector with FPN backbone and the *focal loss* to address class imbalance. The focal loss down-weights easy negatives:

$ "FL"(p_t) = -alpha_t (1 - p_t)^gamma log(p_t). $

With $gamma = 2$, the modulating factor $(1 - p_t)^2$ reduces loss from easy examples (high $p_t$) by $1000 times $ relative to hard examples. RetinaNet matches two-stage accuracy at faster speed.

== Anchor-Free Detectors

Anchors are a design burden: they require manual tuning and create many hyperparameters. Anchor-free methods predict object centres or keypoints directly.

=== CornerNet and CenterNet

*CornerNet* (Law & Deng, 2018): detect top-left and bottom-right corners as heatmaps; group corners by embeddings. No anchors.

*CenterNet* (Zhou et al., 2019): detect object centres as heatmaps on a stride-4 feature map; regress width, height, and optional attributes. Simple, fast, anchor-free. 3D object detection adds depth and orientation as extra regression targets.

=== FCOS

*FCOS* (Tian et al., 2019): fully convolutional; each pixel predicts a box relative to itself if it falls inside a ground-truth box. Centerness score suppresses low-quality predictions far from object centres. Multi-level FPN with scale assignment by object size.

== Transformer-Based Detection

=== DETR

*DETR* (Carion et al., 2020): frames detection as a set prediction problem. A CNN extracts features; a transformer encoder-decoder processes them; $N=100$ learned *object queries* decode boxes and classes. Trained with *Hungarian matching* loss to assign predictions to ground-truth bijectively:

$ cal(L)_"match" = -1_({hat(c)_sigma(i) = c_i}) + 1_({c_i != emptyset}) cal(L)_"box" (b_i, hat(b)_sigma(i)). $

No NMS, no anchors. Slow convergence (500 epochs on COCO) and poor performance on small objects.

=== Deformable DETR

Deformable DETR (Zhu et al., 2020) replaces full attention with *deformable attention*: each query attends to a small set of learned sampling points around a reference location, dramatically reducing computational cost ($O(H W)$ to $O(H W dot K)$ where $K$ is the number of sampling points). Converges in 50 epochs; handles multi-scale features naturally.

=== DINO and Co-DETR

DINO (Zhang et al., 2022) adds contrastive denoising training and mixed query selection. Co-DETR (Zong et al., 2023) uses collaborative training with one-to-many auxiliary heads. State-of-the-art on COCO: Co-DINO with ViT-H backbone achieves 66 $"AP"^"COCO"$.

== Non-Maximum Suppression

After detection, overlapping predictions for the same object must be suppressed. *Hard NMS*: greedily select the highest-scoring box; suppress all other boxes with $"IoU" > theta$. *Soft NMS* (Bodla et al., 2017): decay scores of overlapping boxes rather than removing them: $s_i <- s_i f("IoU"(M, b_i))$ where $f$ is Gaussian or linear. Better for crowded scenes.

== Open-Vocabulary Detection

Modern detectors extend to arbitrary classes via language grounding:
- *CLIP* (Radford et al., 2021): align image and text embeddings; zero-shot classification at test time.
- *GLIP / GroundingDINO*: align phrase grounding with detection; train on image-text pairs to detect arbitrary text descriptions.
- *SAM* (Kirillov et al., 2023): segment anything model; not strictly a detector but enables class-agnostic instance proposal at scale.

== Benchmarks

#table(
  columns: 4,
  [*Model*], [*Backbone*], [*COCO AP*], [*FPS*],
  [Faster R-CNN], [ResNet-50-FPN], [37.4], [~15],
  [RetinaNet], [ResNet-50-FPN], [36.5], [~20],
  [YOLOv8-L], [CSPDarknet], [52.9], [~60],
  [DETR], [ResNet-50], [42.0], [~28],
  [DINO], [ResNet-50], [49.0], [~20],
  [Co-DINO], [ViT-H], [66.0], [~5],
)

== Further Reading

- Girshick, R. et al. (2014). Rich feature hierarchies for accurate object detection (R-CNN). _CVPR_.
- Ren, S. et al. (2015). Faster R-CNN: towards real-time object detection with region proposal networks. _NeurIPS_.
- Lin, T.-Y. et al. (2017). Feature pyramid networks for object detection. _CVPR_.
- Lin, T.-Y. et al. (2017). Focal loss for dense object detection. _ICCV_ (RetinaNet).
- Carion, N. et al. (2020). End-to-end object detection with transformers (DETR). _ECCV_.
- Zhu, X. et al. (2020). Deformable DETR. _ICLR 2021_.

#import "../template.typ": xref

= CNN Architectures

Convolutional neural networks are the backbone of classical deep vision. Although vision transformers now dominate large-scale benchmarks, CNNs remain indispensable in resource-constrained deployment, medical imaging, and as encoders in hybrid architectures. This chapter covers the convolution operation, pooling, normalization, and the evolution of architectures from LeNet to ConvNeXt.

*See also:* #xref("computer-vision", "image-formation", label: "Image Formation") (input representations), #xref("computer-vision", "object-detection", label: "Object Detection") (CNN as backbone), #xref("computer-vision", "image-segmentation", label: "Image Segmentation") (fully convolutional networks), #xref("llm", "transformer-architecture", label: "Transformer Architecture") (ViT alternative).

== The Convolution Operation

A 2D convolution with kernel $W in RR^(K times K times C_"in" times C_"out")$ maps an input feature map $X in RR^(H times W times C_"in")$ to $Y in RR^(H' times W' times C_"out")$:

$ Y[i, j, c] = sum_(m=0)^(K-1) sum_(n=0)^(K-1) sum_(c'=0)^(C_"in"-1) W[m, n, c', c] dot X[i dot s + m, j dot s + n, c'] + b[c] $

where $s$ is the stride and $H' = (H - K + 2p)/s + 1$ after padding $p$.

Key properties:
- *Weight sharing*: the same kernel is applied at every spatial location, giving $K^2 C_"in" C_"out"$ parameters regardless of $H, W$.
- *Local connectivity*: each output depends only on a $K times K$ receptive field.
- *Equivariance to translation*: shifting the input shifts the output.

=== Receptive Field

The *effective receptive field* of a neuron in layer $l$ is the region of the input that influences it. For stride-1 convolutions with kernel size $K$, the receptive field grows as $r_l = r_(l-1) + (K-1) product_(i<l) s_i$. Larger receptive fields require either deeper networks, larger kernels, or dilated convolutions.

=== Dilated (Atrous) Convolution

Dilated convolution inserts zeros between kernel elements, expanding the receptive field without increasing parameters or losing resolution:

$ Y[i, j] = sum_m sum_n W[m, n] dot X[i + d dot m, j + d dot n] $

where $d$ is the dilation rate. Used in DeepLab and WaveNet.

=== Depthwise Separable Convolution

Separates a $K times K$ convolution into:
1. *Depthwise*: $K times K$ per-channel convolution — $K^2 C$ parameters.
2. *Pointwise*: $1 times 1$ cross-channel mixing — $C C'$ parameters.

Total: $K^2 C + C C'$ vs. $K^2 C C'$ for a full convolution — roughly $1/C'$ the cost. Used in MobileNet, Xception, EfficientNet.

== Pooling and Downsampling

*Max pooling* selects the maximum in a $K times K$ window; *average pooling* averages. Both reduce spatial dimensions and introduce local shift invariance. *Global average pooling* (GAP) collapses a $H times W times C$ map to $1 times 1 times C$, replacing large fully connected layers (Lin et al., 2013) and reducing overfitting.

*Strided convolution* is a learnable alternative to pooling: stride 2 halves spatial dimensions while learning the downsampling filter.

== Normalization Layers

=== Batch Normalization

BN (Ioffe & Szegedy, 2015) normalises each feature channel across the batch dimension, then scales and shifts:

$ hat(x)^((k)) = (x^((k)) - mu_B^((k))) / sqrt(sigma_B^((k)2) + epsilon), quad y^((k)) = gamma^((k)) hat(x)^((k)) + beta^((k)). $

At inference, running statistics (computed during training) replace batch statistics. BN enables higher learning rates, reduces sensitivity to initialisation, and acts as a regulariser. It is less effective with small batch sizes.

=== Layer Normalization, Group Normalization

*Layer Norm* normalises across channels (and spatial dimensions) per sample — batch-size independent; standard in transformers. *Group Norm* (Wu & He, 2018) normalises within groups of channels — effective for small batch sizes in detection and segmentation.

== Architecture Evolution

=== LeNet-5 (LeCun et al., 1998)

The first successful deep CNN for digit recognition: two conv layers, two pooling layers, three fully connected layers. Established the conv-pool-FC template.

=== AlexNet (Krizhevsky et al., 2012)

Won ImageNet 2012 (15.3% top-5 error with ensemble, 16.4% single-model, vs. 26.2% for the next-best). Key advances: ReLU activations, dropout regularisation, GPU training, data augmentation. Established deep learning as the dominant paradigm.

=== VGG (Simonyan & Zisserman, 2014)

Showed that depth is the key factor: a homogeneous architecture of $3 times 3$ convolutions stacked to 16–19 layers. VGG-16 achieves 7.3% top-5 on ImageNet. Simple, strong baseline; widely used as a feature extractor.

=== Inception / GoogLeNet (Szegedy et al., 2014)

*Inception module*: parallel $1 times 1$, $3 times 3$, $5 times 5$ convolutions and $3 times 3$ max pooling, concatenated. Captures multi-scale features efficiently. InceptionV3/V4 refactored with factorised convolutions and batch norm.

=== ResNet (He et al., 2016)

Introduced the *residual connection*: $y = F(x, {W_i}) + x$. Enables training of very deep networks (50–1000+ layers) by providing gradient shortcuts. Key insight: it is easier to learn a residual mapping $F(x)$ than a direct mapping $H(x)$ when the optimal function is close to identity. ResNet-152 (6-model ensemble) achieves 3.57% top-5 on ImageNet; ResNet-50 single-model reaches 22.9% top-1. The skip connection is the single most impactful architectural innovation in deep learning.

=== DenseNet (Huang et al., 2017)

Connects each layer to all subsequent layers: $x_l = H_l ([x_0, x_1, ..., x_(l-1)])$. Maximises feature reuse, requires fewer parameters, mitigates vanishing gradients.

=== EfficientNet (Tan & Le, 2019)

Compound scaling: jointly scale width $w$, depth $d$, and resolution $r$ under a compute constraint. Uses NAS-derived MBConv blocks (depthwise separable + squeeze-and-excitation). EfficientNet-B7 achieves 84.3% top-1 on ImageNet at $37 times$ fewer parameters than GPipe. EfficientNetV2 adds fused-MBConv and progressive training.

=== ConvNeXt (Liu et al., 2022)

A pure CNN that matches ViT performance by systematically modernising ResNet: depthwise $7 times 7$ convolutions, inverted bottleneck, GELU activation, LayerNorm. Trained with the same recipe as DeiT (224 epochs, large augmentation). ConvNeXt-XL matches Swin-L on ADE20K segmentation. Demonstrates that transformer innovations largely transfer to CNNs.

== Squeeze-and-Excitation (SE) Networks

SE blocks (Hu et al., 2018) add a channel-wise attention mechanism:
1. *Squeeze*: global average pooling to get $1 times 1 times C$ descriptor.
2. *Excitation*: two FC layers with ReLU/Sigmoid: $s = sigma(W_2 "ReLU"(W_1 z))$.
3. *Scale*: multiply feature maps channel-wise by $s$.

SE adds $2 C^2 / r$ parameters ($r = 16$ typical) for a meaningful accuracy improvement. SE is integrated into EfficientNet, MobileNetV3, ResNeXt-WSL.

== Neural Architecture Search

NAS automates architecture design. Key methods:
- *DARTS* (Liu et al., 2019): differentiable relaxation of the architecture search space; optimise jointly with weights.
- *EfficientNet, MnasNet*: reinforcement learning controller or evolutionary search with hardware-aware latency objectives.
- *Once-for-All*: train one supernet; sub-networks inherit weights for zero-shot deployment at different compute budgets.

NAS-discovered architectures (EfficientNet, EfficientDet, MobileNetV3) dominate mobile deployment.

== Training CNNs

*Data augmentation*: random crop, horizontal flip, colour jitter, mixup, cutmix, rand augment. *Label smoothing*: replace hard targets with $(1-epsilon) y + epsilon / K$; $epsilon = 0.1$ is standard. *Cosine LR schedule* with warm-up. *Weight decay* $1 times 10^(-4)$ to $5 times 10^(-2)$ depending on architecture.

Modern recipe (from Wightman et al.): 300 epochs, AdamW, cosine decay, timm library. ConvNeXt/ViT train 300 epochs on ImageNet; weak baselines often underperform not because of architecture but because of training recipe.

== Benchmarks

#table(
  columns: 4,
  [*Model*], [*Year*], [*Top-1 (IN-1K)*], [*Params*],
  [ResNet-50], [2015], [76.1%], [25M],
  [EfficientNet-B4], [2019], [83.0%], [19M],
  [ConvNeXt-B], [2022], [83.8%], [89M],
  [ViT-B/16], [2020], [81.8%], [86M],
  [Swin-B], [2021], [85.2%], [88M],
  [ConvNeXt-XL], [2022], [87.0%], [350M],
)

== Further Reading

- LeCun, Y. et al. (1998). Gradient-based learning applied to document recognition. _Proc. IEEE_.
- He, K. et al. (2016). Deep residual learning for image recognition. _CVPR_.
- Tan, M., & Le, Q. V. (2019). EfficientNet: rethinking model scaling for CNNs. _ICML_.
- Liu, Z. et al. (2022). A ConvNet for the 2020s. _CVPR_.
- Hu, J. et al. (2018). Squeeze-and-excitation networks. _CVPR_.
- Wightman, R. et al. (2021). ResNet strikes back. _NeurIPS Workshops_.

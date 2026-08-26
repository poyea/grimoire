#import "../template.typ": xref

= Vision Transformers

The Vision Transformer (ViT) transplanted the transformer architecture to images, showing that a pure attention-based model pre-trained at sufficient scale outperforms CNNs without inductive biases for translation equivariance. This chapter covers ViT, its training recipe, hierarchical variants, and multimodal vision-language models.

*See also:* #xref("llm", "transformer-architecture", label: "Transformer Architecture") (attention mechanism, positional encoding), #xref("computer-vision", "cnn-architectures", label: "CNN Architectures") (pre-ViT baselines), #xref("computer-vision", "object-detection", label: "Object Detection") and #xref("computer-vision", "image-segmentation", label: "Image Segmentation") (ViT as backbone).

== Vision Transformer (ViT)

=== Architecture

ViT (Dosovitskiy et al., 2020) splits an $H times W$ image into $N = H W / P^2$ non-overlapping patches of size $P times P$ (typically $P = 16$ or $32$). Each patch is flattened and linearly projected to a $D$-dimensional embedding. A learnable [CLS] token is prepended; 1D learnable positional embeddings are added:

$ z_0 = [x_"cls"; E x_"p"^1; ...; E x_"p"^N] + E_"pos". $

A standard transformer encoder (alternating multi-head self-attention and MLP blocks with LayerNorm and residual connections) processes $z_0$. The CLS token representation $z_L^0$ is passed to a classification head.

For classification, ViT-B/16 uses $L=12$ layers, $D=768$, 12 heads, MLP width $3072$; ViT-L/16 uses $L=24$, $D=1024$, 16 heads.

=== Scaling and Pre-training Data

ViT-B trained on ImageNet-1K from scratch achieves only 77.9% top-1, below comparably-sized ResNets such as ResNet-152 (78.3%) that benefit from inductive biases. But ViT-L trained on JFT-300M (Google's 300M image internal dataset) achieves 87.8%, outperforming CNN baselines.

*Key finding*: ViT requires large-scale pre-training. Without it, the lack of inductive biases (translation equivariance, local connectivity) is a disadvantage. With sufficient data, global attention is an advantage.

=== DeiT: Data-Efficient Image Transformers

DeiT (Touvron et al., 2021) trains ViT on ImageNet-1K alone using knowledge distillation and strong augmentation (RandAugment, MixUp, CutMix, repeated augmentation). A *distillation token* interacts with a CNN teacher via hard-label distillation. DeiT-B achieves 81.8% top-1 on ImageNet-1K without extra data, matching ResNet-152. Made ViT practically accessible without industrial-scale data.

=== Positional Encodings

- *Learnable 1D*: ViT default; order patches by raster scan; works well but cannot generalise to different resolutions.
- *Sinusoidal 2D*: fixed; no parameters.
- *Relative position bias* (Swin, DeiT-III): add learned bias to attention logits based on relative token positions; more resolution-general.
- *RoPE* applied to 2D (e.g., LLaVA-Next): 2D rotary embeddings; enables flexible resolution handling.
- *Register tokens* (Dino V2): extra CLS-like tokens that absorb artefact features, improving attention map quality.

== Hierarchical Vision Transformers

Standard ViT operates at a single resolution, which is memory-expensive and incompatible with FPN-based detection/segmentation pipelines.

=== Swin Transformer

Swin (Liu et al., 2021) introduces:
- *Patch merging*: reduce resolution $2 times $ at each stage (like pooling in CNNs), producing a 4-level hierarchy at strides $4, 8, 16, 32$.
- *Shifted Window Attention (SW-MSA)*: compute attention only within local $M  times  M$ windows (typically $M=7$). Alternating windows shift by $(M/2, M/2)$ between layers to allow cross-window connections.

Window attention complexity: $O(M^2 N)$ vs. $O(N^2)$ for global attention. Swin is a drop-in replacement for ResNet in detection/segmentation pipelines. Swin-L achieves 87.3% top-1 on ImageNet, 58.7 $"AP"^"box"$ on COCO with Cascade Mask R-CNN.

=== Swin V2 and Related

Swin V2 (Liu et al., 2022) scales to 3B parameters with scaled cosine attention and log-spaced continuous relative position bias; pre-trained at $192  times  192$, fine-tuned at $1024  times  1024$; achieves 90.17% top-1 on ImageNet.

*PVT* (Pyramid Vision Transformer): similar hierarchical design with spatial-reduction attention. *MViT* (Fan et al., 2021): multiscale ViT for video; pools queries at each stage.

== Self-Supervised Pre-training

=== DINO and DINOv2

*DINO* (Caron et al., 2021): self-distillation with no labels. A student network is trained to match the output of a momentum teacher (EMA of student). The teacher's CLS token distributions are sharpened with a centering-and-sharpening trick to avoid collapse. DINO features exhibit emergent semantic segmentation in attention maps.

*DINOv2* (Oquab et al., 2023): scales DINO to ViT-g/14 (1.1B parameters); trains on LVD-142M (142M curated images); adds register tokens; achieves state-of-the-art on dense prediction without fine-tuning.

=== MAE: Masked Autoencoders

*MAE* (He et al., 2022): mask a random 75% of image patches; reconstruct masked patches with a lightweight decoder. The encoder processes only visible patches (75% reduction in FLOPs). Pre-trains ViT efficiently on ImageNet. Fine-tuned ViT-L/16 (MAE) achieves 85.9% top-1; ViT-H/14 achieves 86.9%.

MAE is the standard self-supervised recipe for ViT pre-training; it scales better than contrastive methods to large models.

== Vision-Language Models

=== CLIP

*CLIP* (Radford et al., 2021): train a vision encoder and text encoder jointly with a contrastive objective on 400M image-text pairs. The InfoNCE loss maximises similarity of paired embeddings and minimises similarity of unpaired:

$ cal(L) = -1/N sum_i log (e^(z_i^v dot z_i^t / tau)) / (sum_j e^(z_i^v dot z_j^t / tau)). $

CLIP enables zero-shot classification: encode a class as "a photo of a {class}" and find the nearest text embedding. 76.2% zero-shot top-1 on ImageNet. CLIP embeddings are widely used as visual features in multimodal LLMs, text-to-image models (Stable Diffusion CLIP text encoder), and retrieval systems.

=== OpenCLIP and SigLIP

*OpenCLIP* (Ilharco et al., 2021): open reproduction of CLIP; trained on LAION-5B. *SigLIP* (Zhai et al., 2023): replaces softmax contrastive loss with sigmoid binary cross-entropy applied independently to each pair, giving better scaling with no global batch normalisation required. SigLIP-So400m is the vision encoder in Gemini and PaliGemma.

=== Multimodal LLMs

To build a multimodal LLM (see also #xref("llm", "transformer-architecture", label: "Transformer Architecture")):
1. *Visual encoder*: pre-trained ViT (CLIP, SigLIP, DINOv2).
2. *Projector*: linear layer, MLP, or Q-Former that maps visual tokens to the LLM's embedding space.
3. *LLM*: pre-trained language model (LLaMA, Mistral, Gemma).

*LLaVA* (Liu et al., 2023): CLIP ViT-L + MLP projector + LLaMA 2. Instruction-tuned on visual conversations. *InternVL* (Chen et al., 2024): 6B ViT trained from scratch + InternLM2; achieves strongest open-source multimodal results.

== Efficiency and Deployment

=== Token Reduction

ViT with $P=16$ on $224^2$ produces 196 tokens; on $1024^2$ produces 4096. Methods to reduce tokens:
- *DynamicViT*: prune uninformative tokens during forward pass.
- *EViT*: fuse inattentive tokens with a weighted sum.
- *Q-Former* (BLIP-2): a small transformer with $K$ learnable queries extracts $K$ visual tokens, fixed regardless of resolution.

=== Efficient Attention

For high-resolution inputs, full self-attention is $O(N^2)$. Alternatives:
- Swin local windows (covered above).
- *FlashAttention*: hardware-efficient exact attention; standard in all ViT training.
- *Hyper-resolution*: process at reduced resolution, then refine.

== Benchmarks

#table(
  columns: 4,
  [*Model*], [*Pre-training*], [*IN-1K top-1*], [*Params*],
  [ViT-B/16], [ImageNet-21K], [84.0%], [86M],
  [DeiT-B], [ImageNet-1K], [81.8%], [87M],
  [Swin-B], [ImageNet-1K], [83.5%], [88M],
  [DINOv2 ViT-L], [LVD-142M], [86.3%], [307M],
  [ViT-H/14 (MAE)], [ImageNet-1K], [86.9%], [632M],
  [Swin V2-G], [ImageNet-21K], [90.2%], [3B],
)

== Further Reading

- Dosovitskiy, A. et al. (2020). An image is worth 16x16 words. _ICLR 2021_.
- Liu, Z. et al. (2021). Swin Transformer. _ICCV_.
- Touvron, H. et al. (2021). Training data-efficient image transformers (DeiT). _ICML_.
- He, K. et al. (2022). Masked autoencoders are scalable vision learners (MAE). _CVPR_.
- Radford, A. et al. (2021). Learning transferable visual models from natural language supervision (CLIP). _ICML_.
- Oquab, M. et al. (2023). DINOv2: learning robust visual features without supervision. _arXiv:2304.07193_.

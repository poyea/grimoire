#import "../template.typ": xref

= Image Generation: GANs, Tokenizers, Control, and Evaluation <generative-vision>

Generating images has gone from blurry 32-pixel digits to photorealistic, controllable, text-driven synthesis in about a decade. The mathematics of diffusion and flow matching, which now drive most systems, is developed in #xref("machine-learning-foundations", "diffusion-models", label: "Diffusion Models"). This chapter covers the vision-specific machinery around it: generative adversarial networks, which still matter as losses and fast generators; the image *tokenizers* and autoencoders that every latent model depends on; autoregressive and masked image models; how pretrained generators are controlled, personalised and used for editing; how generated images are evaluated, which is harder than it looks; and how their provenance is marked.

*See also:* #xref("machine-learning-foundations", "diffusion-models", label: "Diffusion Models"), #xref("computer-vision", "vision-transformers", label: "Vision Transformers"), #xref("llm", "multimodal", label: "Multimodal Models") (native image generation in LLMs), #xref("machine-learning-foundations", "loss-functions", label: "Loss Functions")

== Generative Adversarial Networks

A GAN (Goodfellow et al., 2014) trains two networks against each other. A *generator* $G$ maps random noise $bold(z)$ to an image; a *discriminator* $D$ tries to tell generated images from real ones. The original objective is the minimax game

$ min_G max_D EE_(bold(x) tilde p_"data") [log D(bold(x))] + EE_(bold(z) tilde p_z) [log(1 - D(G(bold(z))))] $

At the optimum the generator's distribution matches the data, but reaching it is notoriously unstable. The generator can *mode-collapse*, producing a few convincing images and ignoring the rest of the distribution, and the two networks can oscillate instead of converging. A decade of fixes (the Wasserstein loss with gradient penalty, spectral normalisation, careful architectures and regularisers) made GANs reliable enough for the *StyleGAN* family (Karras et al., 2019 onward) to produce faces indistinguishable from photographs. StyleGAN maps $bold(z)$ through an MLP to an intermediate latent $bold(w)$ that modulates every layer, which disentangles attributes such as pose, identity and lighting and made semantic editing in latent space practical. StyleGAN3 removed the aliasing that made fine details "stick" to pixel coordinates.

Diffusion models overtook GANs on diversity and on text conditioning at scale, but adversarial training survives in three important places:

- *As a loss inside image tokenizers* (below), where a discriminator makes reconstructions sharp instead of blurry.
- *In super-resolution and restoration*, where models such as ESRGAN produce plausible high-frequency detail.
- *In distilling diffusion models* to one to four sampling steps, where an adversarial loss (as in adversarial diffusion distillation) restores sharpness that pure distillation loses.

GAN generators also remain unbeaten for latency in narrow domains: a single forward pass, no iterative sampling.

== Image Tokenizers and Autoencoders

Generating pixels directly is wasteful. A $1024 times 1024$ RGB image has three million values, most of which encode imperceptible detail. Nearly every modern generator therefore works in the compressed latent space of a separately trained autoencoder, and the autoencoder's quality bounds the quality of every image the generator can produce.

- A plain *variational autoencoder* trained with a pixel reconstruction loss produces blurry images, because the mean squared error rewards averaging over plausible details.
- *VQ-VAE* (van den Oord et al., 2017) quantises the encoder output against a learned codebook, turning an image into a grid of discrete tokens, a representation an autoregressive transformer can model like text.
- *VQGAN* (Esser, Rombach and Ommer, 2021) adds a perceptual loss (distances between deep features) and an adversarial patch discriminator, giving sharp reconstructions at high compression. This recipe, perceptual plus adversarial plus a light KL or quantisation term, is the template for practically every tokenizer since.

Stable Diffusion's original autoencoder downsamples by 8 in each dimension into 4 latent channels: a $512 times 512 times 3$ image becomes $64 times 64 times 4$, 48 times fewer values. Later models (SD3, FLUX) moved to 16 latent channels, which noticeably improved small text and fine texture. The design tension is fundamental: more compression makes generation cheaper and easier to learn, less compression preserves detail. Recent tokenizers address it by aligning latents with pretrained representation models (so that the latent space is semantically organised as well as compact) and by one-dimensional tokenizers that describe an image with a few hundred tokens regardless of its layout.

== Autoregressive and Masked Image Generation

Once an image is a sequence of discrete tokens, a transformer can generate it token by token, like text. *Parti* (2022) showed that this scales to high-quality text-to-image generation. Raster-order generation is slow (one forward pass per token) and imposes an arbitrary order on a two-dimensional signal, so two alternatives developed:

- *Masked generation* (MaskGIT, 2022; Muse): predict all masked tokens in parallel, keep the most confident, and iterate. A few dozen steps generate the whole image.
- *Next-scale prediction* (VAR; Tian et al., 2024): generate a $1 times 1$ token map, then $2 times 2$, then progressively finer maps, each conditioned on all coarser ones. The coarse-to-fine order suits images and showed clean scaling behaviour.

Native multimodal language models generate images through the same token interface, which is one reason tokenizer research has accelerated. Continuous-token autoregressive models, which replace the discrete codebook with a small diffusion head per token, blur the distinction between the two families.

== Controlling a Pretrained Generator

A text prompt is a weak control signal: it cannot specify an exact pose, layout, or the appearance of a particular person or product. A pretrained generator is therefore usually extended rather than retrained:

- *ControlNet* (Zhang, Rao and Agrawala, 2023) adds spatial conditions such as edge maps, depth maps, human pose skeletons or segmentation maps. It copies the encoder of the frozen denoising network into a trainable branch and connects it back through *zero-initialised convolutions*, so training starts exactly at the original model's behaviour and cannot damage it early on. T2I-Adapter offers a lighter alternative.
- *IP-Adapter* conditions on a reference *image* through separate cross-attention layers, transferring style or subject without text.
- *LoRA* (Hu et al., 2021) fine-tunes by learning low-rank updates $Delta W = B A$ to attention weights. A style or concept adapter is a few megabytes, can be trained on one GPU in minutes, and several can be combined at inference.
- *DreamBooth* (Ruiz et al., 2023) teaches the model a specific subject from 3 to 5 images, bound to a rare identifier token, with a *prior-preservation loss* that keeps the model's general notion of the subject's class from collapsing onto those few examples. *Textual inversion* learns only a new token embedding, cheaper but less faithful.

These methods compose: a LoRA for style, a DreamBooth subject, and a ControlNet pose can all apply to one generation, which is how most production pipelines are assembled.

== Editing Real Images

Editing needs a way to put a real image into the generator's representation and change only what the user asks for:

- *SDEdit* (Meng et al., 2022) adds noise to the image up to an intermediate time step and denoises it with a new prompt. The noise level trades faithfulness to the input against freedom to change it.
- *Inversion*: deterministic samplers such as DDIM can be run backwards to find the noise that regenerates a given image, after which the image can be regenerated with a modified prompt. *Prompt-to-Prompt* (Hertz et al., 2023) keeps the cross-attention maps from the original generation so that the layout is preserved while words are swapped.
- *Instruction editing* (InstructPix2Pix; Brooks et al., 2023) trains a model on synthetic triples of image, instruction and edited image, so that "make it winter" works directly.
- *Inpainting and outpainting* condition on a mask and fill or extend regions, using models trained with masked inputs.

== Evaluating Generated Images

Evaluating a generative model is harder than evaluating a classifier, because there is no single correct output. The measures in common use each capture part of the picture.

=== Fréchet Inception Distance

*FID* (Heusel et al., 2017) embeds real and generated images with an Inception-v3 network, fits a Gaussian to each set of 2048-dimensional features, and reports the Fréchet distance between the Gaussians:

$ "FID" = norm(bold(mu)_r - bold(mu)_g)^2 + op("tr")(Sigma_r + Sigma_g - 2 (Sigma_r Sigma_g)^(1\/2)) $

The matrix square root is the delicate part. $Sigma_r Sigma_g$ is not symmetric, but its trace square root equals that of the symmetric matrix $Sigma_r^(1\/2) Sigma_g Sigma_r^(1\/2)$, which can be computed stably with symmetric eigendecompositions:

```cpp
#include <Eigen/Dense>
#include <cmath>

// Symmetric positive semi-definite square root via eigendecomposition.
Eigen::MatrixXd psd_sqrt(const Eigen::MatrixXd& m) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(m);
    const Eigen::VectorXd roots = es.eigenvalues().cwiseMax(0.0).cwiseSqrt();
    return es.eigenvectors() * roots.asDiagonal() * es.eigenvectors().transpose();
}

// Frechet distance between N(mu_r, sigma_r) and N(mu_g, sigma_g).
double frechet_distance(const Eigen::VectorXd& mu_r, const Eigen::MatrixXd& sigma_r,
                        const Eigen::VectorXd& mu_g, const Eigen::MatrixXd& sigma_g) {
    const Eigen::MatrixXd sqrt_r = psd_sqrt(sigma_r);
    const Eigen::MatrixXd cross = psd_sqrt(sqrt_r * sigma_g * sqrt_r);
    return (mu_r - mu_g).squaredNorm()
         + sigma_r.trace() + sigma_g.trace() - 2.0 * cross.trace();
}
```

FID is the standard number in papers, and it is easy to misuse:

- It is *biased by sample size*: fewer samples give a higher FID. Compare only numbers computed with the same count (50 000 is conventional).
- It is sensitive to *preprocessing*: the resizing filter and JPEG compression of the reference set shift FID by more than many claimed improvements (Parmar et al., 2022). Use one fixed reference implementation.
- Inception-v3 features were trained on ImageNet classification and are blind to many qualities people notice. Using a self-supervised feature extractor such as DINOv2 (FD-DINOv2) correlates better with human judgement (Stein et al., 2023).
- A single number merges *fidelity* and *diversity*. *Precision and recall* for generative models (Kynkäänniemi et al., 2019) separate them: precision is the fraction of generated samples that fall inside the real data's feature manifold, recall the fraction of real samples covered by the generated manifold.

=== Text Alignment and Human Preference

For text-to-image models, FID says nothing about whether the image matches the prompt. *CLIP score* measures image-text embedding similarity but is insensitive to counting, spatial relations and attribute binding ("a red cube on a blue sphere"). Compositional benchmarks such as GenEval check those properties with object detectors. Learned *preference models* (ImageReward, PickScore, HPS) trained on human comparisons approximate aesthetic and alignment judgements, and public arenas collect pairwise human votes. Human evaluation, with careful prompt sets and blind side-by-side comparisons, remains the final arbiter.

== Provenance and Misuse

Photorealistic generation creates obvious risks: fabricated evidence, non-consensual imagery, fraud and disinformation. Technical measures are partial:

- *Content credentials* (the C2PA standard) attach a cryptographically signed manifest recording how an image was created and edited. They are reliable when present but easily stripped.
- *Invisible watermarks* embedded during generation (SynthID, Tree-Ring and similar) survive moderate edits, compression and cropping, but not determined attacks, and work only for cooperating generators.
- *Detectors* trained to spot generated images degrade quickly on new generators and post-processing, and should not be treated as proof either way.

Generation systems also apply filters on prompts and outputs, and training-data governance (consent, licensing, removal of abusive content) has become as important as model architecture.

== Further Reading

Goodfellow, I. et al. (2014). "Generative Adversarial Nets." NeurIPS.

Karras, T., Laine, S., Aila, T. (2019). "A Style-Based Generator Architecture for Generative Adversarial Networks." CVPR.

van den Oord, A., Vinyals, O., Kavukcuoglu, K. (2017). "Neural Discrete Representation Learning." NeurIPS.

Esser, P., Rombach, R., Ommer, B. (2021). "Taming Transformers for High-Resolution Image Synthesis." CVPR.

Chang, H., Zhang, H., Jiang, L., Liu, C., Freeman, W. (2022). "MaskGIT: Masked Generative Image Transformer." CVPR.

Tian, K., Jiang, Y., Yuan, Z., Peng, B., Wang, L. (2024). "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction." NeurIPS.

Zhang, L., Rao, A., Agrawala, M. (2023). "Adding Conditional Control to Text-to-Image Diffusion Models." ICCV.

Ruiz, N. et al. (2023). "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation." CVPR.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S. (2017). "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium." NeurIPS.

Parmar, G., Zhang, R., Zhu, J.-Y. (2022). "On Aliased Resizing and Surprising Subtleties in GAN Evaluation." CVPR.

Stein, G. et al. (2023). "Exposing Flaws of Generative Model Evaluation Metrics and Their Unfair Treatment of Diffusion Models." NeurIPS.

Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., Aila, T. (2019). "Improved Precision and Recall Metric for Assessing Generative Models." NeurIPS.

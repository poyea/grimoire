#import "../template.typ": overbar, xref

= Diffusion Models <diffusion-models>

Diffusion models are generative models that learn to reverse a noise-addition process. They have become the dominant approach for high-fidelity image, audio, and video synthesis, surpassing GANs on perceptual quality benchmarks while being more stable to train. This chapter develops the score-based and DDPM formulations, derives the evidence lower bound, covers accelerated samplers, and connects diffusion to flow matching and consistency models.

*See also:* #xref("machine-learning-foundations", "probability-and-information", label: "Probability and Information") (ELBO, KL divergence), #xref("machine-learning-foundations", "loss-functions", label: "Loss Functions") (denoising score matching), #xref("machine-learning-foundations", "optimization", label: "Optimization") (variance reduction, EMA), #xref("llm", "transformer-architecture", label: "Transformer Architecture") (DiT, UNet attention).

== The Forward Process

The *forward process* gradually destroys data by adding Gaussian noise over $T$ steps. Given data $x_0 tilde q(x_0)$, define

$ q(x_t | x_(t-1)) = cal(N)(x_t; sqrt(1 - beta_t) x_(t-1), beta_t I) $

where $beta_1, ..., beta_T$ is a noise schedule (typically $beta_t in [10^(-4), 0.02]$). The marginal has a convenient closed form via the reparameterisation $alpha_t = 1 - beta_t$, $overbar(alpha)_t = product_(s=1)^t alpha_s$:

$ q(x_t | x_0) = cal(N)(x_t; sqrt(overbar(alpha)_t) x_0, (1 - overbar(alpha)_t) I). $

This means we can sample $x_t$ directly from $x_0$ without simulating the chain:

$ x_t = sqrt(overbar(alpha)_t) x_0 + sqrt(1 - overbar(alpha)_t) epsilon, quad epsilon tilde cal(N)(0, I). $

As $t -> T$, $overbar(alpha)_T approx 0$ and $x_T approx cal(N)(0, I)$.

== The Reverse Process

To generate new data, we run the reverse process from $x_T tilde cal(N)(0,I)$. The true reverse posterior is

$ q(x_(t-1) | x_t, x_0) = cal(N)(x_(t-1); tilde(mu)_t (x_t, x_0), tilde(beta)_t I) $

with

$ tilde(mu)_t = (sqrt(overbar(alpha)_(t-1)) beta_t) / (1 - overbar(alpha)_t) x_0 + (sqrt(alpha_t)(1 - overbar(alpha)_(t-1))) / (1 - overbar(alpha)_t) x_t, quad tilde(beta)_t = (1 - overbar(alpha)_(t-1)) / (1 - overbar(alpha)_t) beta_t. $

The model $p_theta (x_(t-1) | x_t)$ approximates this posterior. DDPM (Ho et al., 2020) parameterises it as

$ p_theta (x_(t-1) | x_t) = cal(N)(x_(t-1); mu_theta (x_t, t), sigma_t^2 I) $

where $sigma_t^2 = tilde(beta)_t$ (fixed) and $mu_theta$ is a neural network.

== The DDPM Training Objective

The ELBO on $log p_theta (x_0)$ decomposes into reconstruction and diffusion terms. Ho et al. show that maximising the ELBO is equivalent to minimising:

$ cal(L)_"simple" = EE_(t, x_0, epsilon) [||epsilon - epsilon_theta (x_t, t)||^2] $

where $epsilon_theta$ is a U-Net that predicts the noise $epsilon$ added to produce $x_t$. At inference, the prediction is converted to $mu_theta$ via:

$ mu_theta (x_t, t) = 1/sqrt(alpha_t) (x_t - beta_t/sqrt(1 - overbar(alpha)_t) epsilon_theta (x_t, t)). $

The noise-prediction parameterisation is better conditioned than predicting $x_0$ or $x_(t-1)$ directly.

== Score-Based Generative Models

Score-based models (Song & Ermon, 2019) parameterise the *score function* $nabla_x log p(x)$ and sample via Langevin dynamics:

$ x_(i+1) = x_i + (epsilon/2) s_theta (x_i) + sqrt(epsilon) z, quad z tilde cal(N)(0, I). $

A score network $s_theta (x, sigma)$ is trained with *denoising score matching*:

$ cal(L) = EE_(x_0, epsilon, sigma) [||s_theta (x_0 + sigma epsilon, sigma) + epsilon / sigma||^2]. $

Song et al. (2020) unified DDPM and score-based models via stochastic differential equations (SDEs). The forward process is a diffusion SDE; the reverse is the *reverse-time SDE* (Anderson, 1982):

$ d x = [f(x,t) - g(t)^2 nabla_x log p_t(x)] d t + g(t) d overbar(W). $

This framework recovers DDPM (VP-SDE) and SMLD/NCSN (VE-SDE) as special cases.

== Noise Schedules

The choice of schedule $beta_t$ controls the signal-to-noise ratio trajectory.

- *Linear schedule* (DDPM): $beta_t$ increases linearly from $10^(-4)$ to $0.02$.
- *Cosine schedule* (Nichol & Dhariwal, 2021): $overbar(alpha)_t = cos^2((t/T + s)/(1+s) dot pi/2)$; avoids abrupt SNR changes near $t=0$.
- *EDM schedule* (Karras et al., 2022): frames diffusion as continuous-time Gaussian convolution; optimal preconditioning of the network inputs leads to better training stability.

== Accelerated Samplers

Vanilla DDPM requires $T = 1000$ denoising steps. Accelerated samplers reduce this dramatically.

=== DDIM

DDIM (Song et al., 2020) derives a non-Markovian forward process with the same marginals as DDPM, enabling deterministic sampling with $10$–$50$ steps:

$ x_(t-1) = sqrt(overbar(alpha)_(t-1)) underbrace(((x_t - sqrt(1-overbar(alpha)_t) epsilon_theta) / sqrt(overbar(alpha)_t)), "predicted " x_0) + sqrt(1 - overbar(alpha)_(t-1)) epsilon_theta. $

DDIM samples are deterministic given fixed noise; interpolation in noise space gives smooth semantic interpolation.

=== DPM-Solver

DPM-Solver (Lu et al., 2022) solves the probability-flow ODE with high-order methods. DPM-Solver++ achieves high quality in 10–20 steps and is the standard sampler in Stable Diffusion.

=== Consistency Models

Consistency models (Song et al., 2023) learn to map any noisy $x_t$ directly to $x_0$ in one step, by enforcing self-consistency along ODE trajectories. LCM (Latent Consistency Models) distills Stable Diffusion to $4$–$8$ steps.

== Classifier-Free Guidance

Conditional generation uses a classifier signal to steer sampling. *Classifier guidance* (Dhariwal & Nichol, 2021) modifies the score:

$ tilde(s)(x, c) = s(x) + w nabla_x log p_phi (c | x). $

*Classifier-free guidance* (Ho & Salimans, 2022) avoids training a separate classifier. The model is trained jointly with and without conditioning ($c = emptyset$ with probability 0.1–0.2):

$ tilde(epsilon)_theta (x_t, c) = epsilon_theta (x_t, emptyset) + w [epsilon_theta (x_t, c) - epsilon_theta (x_t, emptyset)]. $

The guidance scale $w$ trades diversity for fidelity; $w = 7.5$ is a common default for text-to-image. CFG is the standard conditioning mechanism in all major text-to-image systems.

== Latent Diffusion Models

Running diffusion in pixel space is expensive. Latent diffusion models (Rombach et al., 2022) first compress images into a lower-dimensional latent space with a VAE ($8 times$ spatial downsampling), then run diffusion on latents. This is the foundation of Stable Diffusion:

1. Encode: $z_0 = cal(E)(x_0)$ with a frozen encoder.
2. Diffuse and denoise $z_t$ with a U-Net conditioned on text embeddings via cross-attention.
3. Decode: $hat(x)_0 = cal(D)(hat(z)_0)$ with a frozen decoder.

The U-Net uses spatial self-attention and cross-attention to the CLIP text embedding. Stable Diffusion XL adds a second text encoder (OpenCLIP ViT-bigG) and a two-stage architecture.

== Diffusion Transformers

DiT (Peebles & Xie, 2023) replaces the U-Net with a Vision Transformer (ViT). Patches of the noisy latent are treated as tokens; timestep and class conditioning are injected via adaptive layer normalisation (AdaLN). DiT-XL/2 (patch size 2, XL width) outperforms the U-Net baseline on ImageNet at 256$times$256 and 512$times$512. DiT is the architecture backbone of Sora and Flux.

== Flow Matching

Flow matching (Lipman et al., 2022; Liu et al., 2022) generalises diffusion by learning a vector field $v_theta (x, t)$ whose ODE transports samples from $p_0 = cal(N)(0, I)$ to $p_1 = p_"data"$:

$ d x / (d t) = v_theta (x, t). $

Training minimises the conditional flow matching objective using analytical conditional vector fields, bypassing the need to simulate the ODE. Rectified Flow (Liu et al., 2022) uses straight-line paths, enabling very fast sampling. Stable Diffusion 3 and Flux use flow matching instead of DDPM.

== Video and Audio Diffusion

Diffusion extends naturally to other modalities:
- *Video*: Sora (OpenAI, 2024) uses a DiT operating on spacetime patches of video latents with full 3D attention.
- *Audio*: WaveGrad, DiffWave apply diffusion to raw waveforms; AudioLDM operates in mel-spectrogram latent space.
- *3D shapes*: Point-E, Shap-E diffuse over point clouds and neural fields.

== Training Stability and Best Practices

Key practical considerations:
- *EMA*: maintain an exponential moving average of weights for inference; $"decay" = 0.9999$.
- *Mixed precision*: train in bfloat16; keep EMA in float32.
- *$v$-prediction*: predict the velocity $v = sqrt(overbar(alpha)) epsilon - sqrt(1-overbar(alpha)) x_0$ instead of noise; better at low noise levels.
- *Min-SNR weighting* (Hang et al., 2023): reweight the loss by $min("SNR"(t), gamma) / "SNR"(t)$ to balance training across timesteps.
- *Gradient checkpointing*: required for long sequences or large models due to memory.

== Further Reading

- Anderson, B. D. O. (1982). Reverse-time diffusion equation models. _Stochastic Processes and Their Applications_, 12(3).
- Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. _NeurIPS_.
- Song, Y. et al. (2020). Score-based generative modeling through stochastic differential equations. _ICLR 2021_.
- Rombach, R. et al. (2022). High-resolution image synthesis with latent diffusion models. _CVPR_.
- Nichol, A., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. _ICML_.
- Karras, T. et al. (2022). Elucidating the design space of diffusion-based generative models (EDM). _NeurIPS_.
- Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers (DiT). _ICCV_.
- Lipman, Y. et al. (2022). Flow matching for generative modeling. _ICLR 2023_.

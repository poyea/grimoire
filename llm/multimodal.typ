= Multimodal Models

A multimodal model takes inputs from more than one modality — images, audio, video, sometimes interleaved with text — and emits text (or other modalities). The dominant architectural pattern is *bolt a visual / audio encoder onto a text LLM via a projection module*, which trades training cost for the ability to inherit the LLM's reasoning. This chapter covers the projection-based pattern (LLaVA, Qwen-VL, Idefics), native multimodal training (Gemini, GPT-4o, Chameleon), audio LLMs (Whisper conditioning, AudioLM, Qwen-Audio), and the tokenization and training-data engineering required for each.

*See also:* _Transformer Architecture_ (the LLM backbone), _Pretraining_ (multimodal mixes), _Tokenization_ (image / audio tokens vs. continuous features).

== Why Bolt-On Wins (Mostly)

Training a large model from scratch on mixed-modality data is expensive and unstable. Bolt-on training reuses a strong pre-trained vision encoder (CLIP-ViT, SigLIP, EVA-CLIP) and a strong pre-trained LLM (LLaMA, Qwen, Mistral), connecting them with a small projection module. Only the projection trains in stage 1; both encoder and LLM partially unfreeze in stage 2.

Advantages:

- 100–1000× cheaper than native multimodal pretraining.
- Components can be swapped (better vision encoder, larger LLM) modularly.
- Open-source community can train competitive 7B–70B VLMs on modest budgets.

Disadvantages:

- Vision encoder bottleneck: the projection forwards a small token sequence (32–576 image patches), limiting fine spatial reasoning.
- Modality interactions are shallow — the model "sees" pre-encoded features, not raw pixels.
- Audio and video integration is awkward; each modality needs its own encoder + projection.

Native models (Gemini 1.5, GPT-4o, Chameleon) train a single transformer on tokenized images, audio, and text from scratch and reach higher quality ceiling on cross-modal tasks at much higher cost.

== Vision-Language: The LLaVA Pattern

LLaVA (Liu et al. 2023) is the canonical bolt-on VLM.

```
Image  → ViT encoder (frozen)         → [P_1, P_2, ..., P_576] patch features (d_v)
                                        ↓ MLP projection W: d_v → d_text
Text  → tokenizer → embed                ↓
                       [emb(<image>), W(P_1), ..., W(P_576), emb(prompt tokens)]
                                        ↓
                                  LLM (LLaMA)
                                        ↓
                                  generated text
```

The image patch features become *image tokens* interleaved with text tokens. The LLM attends over both jointly. Variants:

=== Projection Choices

- *Linear* (LLaVA-1.0): a single dense layer. Fast, sometimes too low-capacity.
- *MLP* (LLaVA-1.5): GELU-activated two-layer. The de facto standard.
- *Q-Former* (BLIP-2 / Idefics): a small transformer with learnable query tokens that cross-attend to the vision features and emit a fixed number of "summary" tokens. Compresses 576 → 32 tokens.
- *Perceiver Resampler* (Flamingo): perceiver-style cross-attention from learned queries to image features.

Q-Former and Perceiver compress but lose detail; MLPs preserve detail but inflate context cost. LLaVA-NeXT uses MLP at higher resolution to preserve OCR ability.

=== Training Stages

1. *Pretraining (alignment):* projection only, frozen encoder & LLM. Caption data (~558k LAION/CC pairs in LLaVA-1.5). Loss: next-token prediction on the caption given image.
2. *Instruction tuning:* projection + LLM trainable, vision encoder frozen. Visual instruction data (LLaVA-Instruct-150K — GPT-4-generated dialogues over COCO images). Now the model learns to follow visual instructions.

LLaVA-1.5 13B trains end-to-end in ~1 day on 8×A100.

=== Resolution Handling

A 336×336 image at ViT/14 yields $24 times 24 = 576$ patch tokens. OCR and fine-grained reasoning require higher resolution. Approaches:

- *Native high-res ViT* (SigLIP-SO/14 at 384, EVA-CLIP at 448).
- *AnyRes / Tile-based*: split a high-res image into a grid of low-res tiles, encode each separately, concatenate tokens. LLaVA-NeXT processes 672×672 by tiling into a 2×2 grid of 336×336. Brings OCR within reach.
- *Dynamic resolution*: Qwen-VL crops to a learned grid based on aspect ratio.

== Specific Architectures

=== Qwen-VL / Qwen2-VL

Qwen2-VL (Wang et al. 2024) uses a custom ViT with *2D-RoPE* (position encoding extended to two spatial axes) and *Naive Dynamic Resolution* — variable token counts per image based on size. Cross-attention is full (not cross-modal queries). Strong document and chart understanding.

=== Idefics 2 / 3

Idefics 3 (HuggingFace 2024) uses SigLIP-SO-400m + Llama 3.1; Perceiver Resampler to 64 query tokens; native multi-image input. Open weights, fully documented training pipeline.

=== InternVL 2 / 3

InternVL trains a 6B-parameter custom ViT (InternViT) — far larger than CLIP-ViT — and pairs it with Qwen2/InternLM-2 via MLP. Strong on Chinese-language documents.

=== LLaVA-OneVision

Multi-image and video extension of LLaVA. Each video is sampled at 1–2 fps; frames are encoded and concatenated. Trained on ~2M visual instructions.

== Native Multimodal Training

=== Chameleon (Meta 2024)

Chameleon trains a single transformer on a *unified token stream* of text and image tokens. Images are tokenized to discrete codes by a VQ-VAE-like image tokenizer (VQGAN-style, 8192-codebook). Both modalities share the same vocabulary and the same next-token-prediction loss. The model can generate text or images interleaved.

Challenges:

- *Modality collapse:* one modality dominates loss; the other is ignored. Solved with per-modality loss reweighting.
- *Tokenizer quality:* a poor image tokenizer caps generation quality. Chameleon iterates the tokenizer.

=== Gemini and GPT-4o

Gemini (Google 2023+) and GPT-4o (OpenAI 2024) are reported to be natively multimodal: a single transformer trained on text + images + audio + video tokens. Architecture details not public; behaviorally they exhibit modality fusion (e.g., reasoning about audio prosody in speech).

== Audio LLMs

=== Whisper-Style Conditioning

Whisper (Radford et al. 2022) is an encoder-decoder transformer trained on 680k hours of weakly-labeled speech, used as a *frozen audio encoder*. Audio LLMs (Qwen-Audio, SALMONN) take Whisper encoder output, project to the LLM token space, and treat as a prefix.

```
audio (16kHz)  → log-Mel filterbank (80 channels)
               → Whisper encoder (frozen)
               → [a_1, ..., a_T] features (d_audio)
                                  ↓ projection
                                  ↓ LLM
                                  ↓ text
```

=== AudioLM and Discrete Audio Tokens

AudioLM (Borsos et al. 2023) tokenizes audio into discrete codes via SoundStream / EnCodec. The LLM then operates on these tokens just like text. Enables joint *understanding + generation*: the model can answer about audio or *generate* audio (speech, music). VALL-E, MusicLM follow this pattern.

=== Real-Time Audio (Moshi)

Moshi (Kyutai 2024) is a full-duplex speech-to-speech model trained on dual audio streams (user + assistant) with text as an intermediate scratch. Achieves ~160ms latency for natural conversation — far below the ~1.5s of pipeline (ASR → text LLM → TTS) systems.

== Video

Video is sequences of frames + audio + sometimes text overlays. Strategies:

- *Frame sampling:* 1 fps for hour-long clips, 8 fps for short clips. Each frame is encoded with the image encoder; positional embedding marks the frame index.
- *Token reduction:* a 1-minute clip at 1 fps with 576 image tokens per frame = 35k tokens just for video. Aggressive resampling (Q-Former, 32 tokens/frame) or temporal pooling is necessary.
- *Temporal modeling:* VideoLLaMA uses a temporal adapter that 3D-pools features across frames before projection. Gemini reportedly trains on full long-form video.

== Multimodal Training Data

Public data sources:

- *Image-caption pairs:* LAION-2B, CC3M, CC12M, COYO-700M, DataComp.
- *VQA / instruction:* LLaVA-Instruct, VQAv2, GQA, ChartQA, DocVQA, OK-VQA, ScienceQA.
- *OCR:* PDF-A, RVL-CDIP, OCR-VQA.
- *Video:* WebVid-10M, HowTo100M, InternVid.
- *Audio:* Common Voice, LibriSpeech, AudioSet.

Quality matters more than quantity at the instruction-tuning stage. LLaVA-Instruct-150k punches above its weight because each example is GPT-4-distilled with carefully crafted scenarios. Subsequent open VLMs (ShareGPT4V, Cambrian-1) scale instruction data with model-distilled examples.

== Evaluation

Benchmarks:

- *MMMU* (Yue et al. 2023): college-level multimodal reasoning across 30 subjects.
- *MathVista, ChartQA, DocVQA*: domain-focused.
- *MMBench, SEED-Bench*: broad VLM capability tiers.
- *Video-MME, MVBench*: video.
- *AIR-Bench, AudioBench*: audio.

Beware of contamination: many benchmarks have leaked into image-caption training data. Cross-check with held-out probes.

== Pitfalls

- *Object hallucination:* VLMs report objects that are not in the image. Mitigations: contrastive decoding (VCD), CHAIR-style eval during training, instruction data that explicitly says "if the object is not present, say so."
- *Position blindness:* asking "what is on the left of X" frequently fails because the projection compresses spatial layout. AnyRes and 2D-RoPE help.
- *Reading text:* low-resolution vision encoders cannot read small text. OCR-tuned models (Qwen2-VL, InternVL 2) explicitly train on document images.
- *Cross-image reasoning:* most VLMs are single-image; multi-image and video require dedicated training and positional handling.

== Further Reading

Liu, H. et al. (2023). "Visual Instruction Tuning." NeurIPS.

Liu, H. et al. (2024). "LLaVA-NeXT: Improved reasoning, OCR, and world knowledge."

Li, J. et al. (2023). "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models." ICML.

Wang, P. et al. (2024). "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution." arXiv:2409.12191.

Chameleon Team (2024). "Chameleon: Mixed-Modal Early-Fusion Foundation Models." arXiv:2405.09818.

Radford, A. et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision." (Whisper)

Borsos, Z. et al. (2023). "AudioLM: A Language Modeling Approach to Audio Generation." TASLP.

Défossez, A. et al. (2024). "Moshi: a Speech-Text Foundation Model for Real-Time Dialogue."

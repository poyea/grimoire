= Distillation

Distillation transfers capability from a large *teacher* model to a smaller *student*. The student trains not (only) on the original labels but on the teacher's outputs, distributions, or reasoning traces. Done well, a 7B student can recover 80–95% of a 70B teacher's performance at 10× lower inference cost. This chapter covers the original Hinton formulation, modern LLM-specific variants — sequence-level, white-box, on-policy, reasoning-trace distillation — and the failure modes that limit how far distillation can go.

*See also:* _Pretraining_ (distillation is increasingly used as part of pretraining), _Quantization_ (a complementary axis: same architecture, smaller bits), _Reasoning Models_ (CoT trace distillation).

== Classical Knowledge Distillation

Hinton, Vinyals, Dean (2015): train the student to match the teacher's *soft probabilities* (post-softmax with temperature $T$), not just the hard labels. Soft probabilities carry "dark knowledge" about which other classes the teacher considered plausible — far richer than the one-hot label.

$ cal(L)_"KD" = alpha times "CE"(y_"hard", "softmax"(z_s)) + (1-alpha) times T^2 times "KL"("softmax"(z_t / T), "softmax"(z_s / T)) $

with $T in [2, 8]$, $alpha in [0.1, 0.5]$. The $T^2$ factor preserves gradient magnitude across temperatures.

For classification this is a complete recipe. For generative LLMs, the loss must be adapted to sequence outputs.

== Sequence-Level Distillation (Black-Box)

The simplest LLM distillation: query the teacher for outputs on a set of prompts, store (prompt, response) pairs, *SFT* the student on these. No access to teacher logits required; works against API-only frontier models.

```python
def collect_distillation_data(teacher_api, prompts: list[str]):
    pairs = []
    for p in prompts:
        resp = teacher_api.generate(p, temperature=0.7, max_tokens=2048)
        pairs.append({"prompt": p, "response": resp})
    return pairs

# Then standard SFT on the student with `pairs`.
```

This is how Alpaca (Stanford 2023) bootstrapped from GPT-3.5, Vicuna from GPT-4, and most early open-source instruction-tuned models. It is also how DeepSeek-R1-Distill-\* models (Qwen/LLaMA students) are trained — by SFT on R1's reasoning traces.

Limits:

- Student learns the teacher's *style* and *common reasoning patterns* but not its full latent capability. On out-of-distribution problems the student degrades faster than the teacher.
- Quality of the training prompts dominates. Alpaca's "self-instruct" prompts are narrow; later datasets (UltraChat, OpenHermes, WizardLM) are more diverse.

== White-Box Distillation

When teacher logits are accessible, the student can match the *full distribution* over the vocabulary at every step:

$ cal(L) = sum_t D("softmax"(z_t^"teacher" / T), "softmax"(z_t^"student" / T)) $

with $D$ typically forward KL. Equivalent to the sequence-level Hinton loss applied per token.

=== Forward vs. Reverse KL

- *Forward KL* $"KL"(p_t || p_s)$ is *mode-covering*: the student tries to cover all modes the teacher considers plausible. Risk: probability mass on tokens the student cannot model causes overgeneralization (the student becomes vague).
- *Reverse KL* $"KL"(p_s || p_t)$ is *mode-seeking*: the student concentrates on a few high-probability tokens of the teacher. Risk: dropping legitimate alternative answers.
- *Skew KL / JS divergence:* interpolations that trade off.

GKD (Generalized KD; Agarwal et al. 2023) showed reverse KL outperforms forward KL for instruction-following LLMs because the student's smaller capacity benefits from mode-seeking. MiniLLM (Gu et al. 2023) uses reverse KL with sampling-based gradient estimation.

== On-Policy Distillation

Static (prompt, teacher-response) data has a *distribution mismatch*: the student is trained on prompts paired with the teacher's response, but at inference the student sees prompts paired with *its own* prior tokens. The mismatch grows with sequence length.

*On-policy distillation* generates the student's own tokens during training and matches the teacher's *next-token distribution* at the student-generated context.

```
For each prompt:
  sample partial response y_<t from the student
  teacher.forward(prompt || y_<t) → z_t^teacher
  student.forward(prompt || y_<t) → z_t^student
  loss += KL(p_t^teacher || p_t^student)
```

GKD, DistiLLM, and MiniLLM use variants of this. Cost is higher (the student must run inference during training) but quality is much better. The student-generated context puts the teacher's distribution where it actually matters — at the states the student will visit.

== Reasoning-Trace Distillation

For reasoning models, the teacher emits long CoT traces. Two distillation modes:

=== Imitation (DeepSeek-R1-Distill)

SFT the student on (prompt, full trace) pairs. The student learns to *imitate* the teacher's reasoning surface form: writing thoughts, self-correcting, backtracking. DeepSeek-R1-Distill-Qwen-7B reaches AIME 24 = 55%, far above the base Qwen-7B at ~15%.

But the student's capability is bounded by *imitating* search, not *performing* search. On novel hard problems the student's traces look plausible but reach wrong conclusions more often than the teacher.

=== Search Distillation

Distill the *search procedure*, not just the trace. Generate $N$ traces with the teacher, label them by outcome, train the student with RL using teacher-grade rewards (or by SFT only on the high-reward traces — rejection sampling). The student learns both the trace style and the implicit search policy.

DeepSeek-R1 itself was created by RL'ing DeepSeek-V3-Base on verifiable rewards (no teacher) — when a strong teacher exists, you can shortcut by SFT-then-RL with the teacher's traces as warm start.

== When Distillation Plateaus

The student's capacity floor caps performance. Empirical observations:

- *Linear-in-log:* student quality scales approximately linearly with $log("params")$ for fixed teacher.
- *Capability ceiling:* on problems the *teacher* itself solves with low probability, the student inherits or amplifies the failure rate.
- *Skill gap:* certain capabilities (long-context coherence, deep multi-step reasoning) distill poorly — they require the larger model's representational depth.

A 7B student can match a 70B teacher on most instruction-following tasks but loses 10–20 points on hard math, multi-step coding, and long-context retrieval.

== Distillation in Pretraining

Recent practice: large model first, then *distill into multiple smaller students* as part of the release pipeline. Examples:

- *Gemma 2* (Google 2024): the 9B and 27B models are pretrained from scratch *with* a distillation loss from a much larger Gemini teacher, mixed with standard CE loss. Gemma 3 continues this.
- *NVIDIA Minitron* (2024): structured pruning + width / depth distillation; an 8B Llama becomes a 4B with retention of ~95% of MMLU.

Distillation as a pretraining objective (not just a finetune) is more compute-efficient than training a small model from scratch on the same tokens.

== Engineering

=== Logit Storage

Storing teacher logits for the entire training corpus is prohibitive (vocab = 128k × sequence length × dataset). Workarounds:

- *On-the-fly:* run teacher inference during training; offload teacher to CPU or to a separate GPU pool. Bottleneck: teacher throughput.
- *Top-K logits:* store only the top 100–500 teacher logits per token; renormalize. Cuts storage 1000×; minor quality loss.
- *Sparse top-K with temperature mixing:* store top-K plus the residual mass as a single "other" bucket.

DeepSpeed, Megatron, and TRL all support `top_k_logit_distillation` modes.

=== Tokenizer Mismatch

Teacher and student rarely share a tokenizer. Approaches:

- *Re-tokenize:* run the student's tokenizer on (prompt, teacher response); SFT on those tokens. Loses logit information but works.
- *Token alignment:* use byte-level matching or longest-common-substring to align teacher's subword sequence with student's. Lossy and complex.
- *Force-share tokenizer:* fix the student to use the teacher's tokenizer (requires re-embed at training start).

For sequence-level black-box distillation, re-tokenization is the standard answer.

=== Capacity / Compute Trade

Allocating training compute between teacher-inference and student-update matters. Rule of thumb: at student capacity << teacher, spend most compute on student update; at student capacity ~ 0.5 × teacher, balance is closer.

== Pitfalls

- *Mode collapse:* on policy + reverse KL → student concentrates on a single response per prompt, killing diversity. Mix in standard SFT loss as regularizer.
- *Hallucination amplification:* if the teacher confabulates on the distillation prompts, the student learns confabulation as fluent style. Filter teacher outputs (with the teacher itself, or with verifiers) before SFT.
- *Position bias:* if teacher data has all-the-time-correct answers, student gets miscalibrated — confidence everywhere. Include teacher errors in the data with explicit "I don't know" responses.
- *Eval contamination via teacher:* if the teacher saw an eval set, the student inherits the contamination through distillation. Decontaminate distillation prompts against your eval set.

== Recipe Summary

A practical distillation flow:

1. *Curate prompts.* Diverse, representative of target task distribution. 100k–10M.
2. *Generate teacher data.* Sample at $T = 0.7$ with rejection (filter on quality / correctness signals).
3. *SFT the student* (Hinton loss if logits accessible; else next-token CE).
4. *Optional on-policy stage* with reverse KL on student samples.
5. *Eval against teacher* — measure win rate / capability gap.
6. *RL polish* (DPO from teacher-preference pairs, or GRPO with verifiable rewards) to push past the imitation ceiling.

== Further Reading

Hinton, G., Vinyals, O., Dean, J. (2015). "Distilling the Knowledge in a Neural Network." NeurIPS Deep Learning Workshop.

Kim, Y., Rush, A. (2016). "Sequence-Level Knowledge Distillation." EMNLP.

Gu, Y. et al. (2023). "Knowledge Distillation of Large Language Models." (MiniLLM) ICLR 2024.

Agarwal, R. et al. (2023). "GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models." ICLR 2024.

Ko, J. et al. (2024). "DistiLLM: Towards Streamlined Distillation for Large Language Models." ICML.

Gemma Team (2024). "Gemma 2: Improving Open Language Models at a Practical Size."

Muralidharan, S. et al. (2024). "Compact Language Models via Pruning and Knowledge Distillation." (NVIDIA Minitron) NeurIPS.

Taori, R. et al. (2023). "Stanford Alpaca: An Instruction-following LLaMA model."

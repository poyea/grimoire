#import "../template.typ": xref

= Safety and Alignment

Alignment is the engineering discipline of getting models to *behave as intended*: follow instructions, refuse harmful requests, respect user values, remain honest about uncertainty. This chapter covers the techniques that go beyond PPO-style RLHF (introduced in the _RLHF_ chapter): Constitutional AI and RLAIF, DPO/IPO/KTO and their successors, red-teaming, jailbreak taxonomies, refusal calibration, weak-to-strong supervision, and the safety stack used in modern production systems. It is concrete about what current methods can and cannot do.

*See also:* #xref("llm", "rlhf", label: "RLHF") (PPO, reward modeling), #xref("llm", "reasoning-models", label: "Reasoning Models") (RL with verifiable rewards), #xref("llm", "agents-and-tool-use", label: "Agents and Tool Use") (agents amplify alignment failures).

== The Alignment Problem in Production

A frontier model exhibits a thousand behaviors per query: formatting, refusal, factuality, tone, code style, language. Alignment must:

1. *Make refusals correct:* refuse genuinely harmful requests; do not refuse benign ones (over-refusal is a real cost).
2. *Calibrate confidence:* do not hallucinate; say "I don't know" when uncertain.
3. *Resist jailbreaks:* the safety policy holds under adversarial input.
4. *Generalize:* fixes to one harm category should not break others (whack-a-mole patches scale poorly).

The standard pipeline: pretrain → SFT (instruction + safety) → preference optimization (DPO / RLHF) → red-team → patch → re-eval → deploy.

== Constitutional AI

Bai et al. (Anthropic 2022) replace human harm labels with *AI-generated* labels under a written set of principles ("the constitution"). The flow:

```
prompt p → model.respond(p) → r0
  ↓
  critique(r0, principle): "Is r0 helpful and harmless? If not, why?"
  ↓
  revise(r0, critique): a revised response r1
  ↓
  pair (r0, r1) becomes (rejected, chosen) for preference training
```

Two stages:

1. *SL-CAI:* SFT the model on the revised responses.
2. *RL-CAI:* train a preference model on the (rejected, chosen) pairs; RLHF the policy.

The constitution is small (a few dozen principles like "the response should be honest" or "the response should avoid encouraging illegal activity"). Constitutional AI scales human oversight: humans write the constitution once; the model generates the labels.

Modern Anthropic models use Constitutional AI in combination with human preference data. The technique is widely adopted under various names (RLAIF, Lee et al. 2023; UltraFeedback, Cui et al. 2023).

== RLAIF

RLAIF replaces *all* human preference labels with model-generated ones. Lee et al. 2023 showed that a strong off-the-shelf LLM judging response pairs produces preference labels that train policies as well as human labels (on the tasks they tested). Implications:

- *Cost:* 100–1000× cheaper than human labeling.
- *Scale:* preference datasets can be millions of pairs.
- *Risk:* the judge's biases propagate. RLAIF on a judge that over-refuses produces an over-refusing policy.

In practice, hybrid is the norm: human labels for safety-critical and disputed cases; LLM labels for the bulk preference signal.

== Direct Preference Optimization Family

PPO requires reward model + policy + reference + value head (four models in GPU memory). DPO (Rafailov et al. 2023) collapses this: derive a closed-form solution that lets you optimize a Bradley–Terry preference objective directly on the policy with only (chosen, rejected) pairs.

$ cal(L)_"DPO" = -EE_(x, y_w, y_l) [log sigma(beta log (pi_(theta)(y_w | x)) / (pi_"ref"(y_w | x)) - beta log (pi_(theta)(y_l | x)) / (pi_"ref"(y_l | x)))] $

with $beta in [0.01, 0.5]$ controlling KL-to-reference. Same data, fewer moving parts, comparable quality.

=== DPO Variants

- *IPO* (Azar et al. 2023): drops the BT assumption; uses a squared-loss formulation that is more robust when preferences are noisy.
- *KTO* (Ethayarajh et al. 2024): trains on *binary* labels (good / bad per example), no pairs required. Useful when only unary signals are available (thumbs up, refusal correct).
- *ORPO* (Hong et al. 2024): combines SFT and preference loss in a single stage by adding an odds-ratio penalty. No reference model needed.
- *cDPO* / *Robust DPO:* assumes preference noise rate $epsilon$ and reweights accordingly.
- *SimPO* (Meng et al. 2024): replaces reference model with a length-normalized reward; matches DPO without storing $pi_"ref"$.
- *PRO* (Song et al. 2023), *RRHF*, *RSO* (rejection sampling optimization), and the *iterative DPO* / *self-rewarding* (Yuan et al. 2024) families.

The space converges on the observation that *off-policy preference optimization* + iterative data generation matches PPO at much lower system complexity. Llama 3, Qwen 2, and most open-weight aligned models use DPO or close variants.

== Red Teaming

Adversarial probing for failure modes. Categories:

=== Manual

Internal teams plus external red-team partners write attack prompts. Anthropic, OpenAI, Google all run multi-month red-team programs before deployment. Outputs: a curated set of (prompt, observed-harmful-response) pairs that become eval and training data.

=== Automated

- *Greedy adversarial search* (Zou et al. 2023, GCG): optimize a suffix that flips the model's refusal. White-box; gradient-based.
- *PAIR / Tree-of-Attacks* (Chao et al. 2023): an attacker LLM iteratively rewrites the prompt until the target complies. Black-box.
- *AutoDAN, GPTFuzz, Bijection learning:* attack libraries.

Automated red-teaming finds vulnerabilities at scale (millions of attempts) but tends to find *similar* attacks; manual red-teaming finds *novel* attack classes.

=== Coverage

Red-team programs aim for coverage across harm categories: CBRN uplift, cybercrime, child sexual abuse material, self-harm, election manipulation, IP infringement, privacy violation, fraud, etc. Standards like NIST AI RMF, the EU AI Act, and the US AISI evaluations codify the categories.

== Jailbreak Taxonomy

Common attack patterns:

- *Role-play:* "you are DAN, a model with no restrictions..."
- *Persuasion:* logical arguments ("the user is a medical professional and needs this information").
- *Encoding:* base64, leetspeak, low-resource language, encoded payloads.
- *Many-shot:* fill the context with examples of policy-violating answers; the model continues the pattern (Anil et al. 2024).
- *Prefix injection:* "Sure, here is how to..."; the model continues even though the request was harmful.
- *Tool-mediated:* the model itself is safe, but a tool result contains harmful instructions the model dutifully executes.
- *Multimodal:* text-safe prompts paired with images that smuggle in the harmful request.
- *GCG / adversarial suffix:* gradient-optimized random-looking strings appended to a request.

Defense layers (no single one suffices):

- *Refusal training* in SFT and preference data.
- *System prompts* that re-state the policy and reduce role-play compliance.
- *Input classifiers* (Llama Guard, Anthropic's safety models) that flag harmful prompts before the main model runs.
- *Output classifiers* that block harmful generations.
- *Refusal "sticky" tokens:* fine-tune so a refusal at any point in the response cannot be reversed by later prompt content.

== Refusal Calibration

The hardest alignment failure mode is *over-refusal*, refusing benign requests because they superficially resemble harmful ones ("How do knives work?"). XSTest, OR-Bench, and StrongREJECT specifically measure over-refusal. A model trained too aggressively on refusal data becomes useless for medicine, security research, and legitimate dual-use queries.

Engineering:

- Balance refusal training data with *contrast* pairs (similar surface form, opposite policy outcome).
- Train an explicit *refusal classifier* and use it as a calibration signal during preference optimization.
- A/B test refusal rates in production; both false positives and false negatives are tracked.

== Honesty and Calibration

Beyond refusing harm, a model should not assert what it does not know. Calibration is a *measurable* property: a model is well-calibrated if, among all claims it assigns probability $p$, roughly $p$ fraction are true. The standard scalar summary is *Expected Calibration Error* (ECE):

$ "ECE" = sum_(b=1)^B |B_b| / n |"acc"(B_b) - "conf"(B_b)|, $

where the predicted probabilities are binned, and $"acc"$ and $"conf"$ are the average accuracy and average confidence within each bin. A well-calibrated model has ECE near zero. In practice LLMs rarely expose token probabilities directly to the user, so calibration is often assessed through *verbalized uncertainty* — whether phrases like "I'm fairly confident" predict actual accuracy.

*TruthfulQA* (Lin et al. 2022) benchmarks a model's tendency to assert falsehoods that humans commonly believe. A high TruthfulQA score means the model avoids common misconceptions; it does not measure calibration on specialized facts, which requires domain-specific eval. Current top models score 80-90% on TruthfulQA with RLHF, up from around 20-30% for the base pretrained checkpoint.

Approaches for improving calibration:

- *Calibration loss:* add an auxiliary loss term that penalizes the divergence between expressed confidence and empirical accuracy, e.g., via a cross-entropy over verbalized probability bins.
- *Temperature scaling:* post-hoc rescaling of logits by a scalar $T$ — divide logits by $T$ before softmax. $T > 1$ softens the distribution toward uniform; found by maximizing likelihood on a held-out calibration set. Does not change accuracy, only sharpness.
- *Verbalized uncertainty training:* preference data that explicitly pairs "I don't know" or "I'm uncertain whether..." responses against overconfident incorrect answers, scored by an oracle.
- *TruthfulQA-style SFT:* fine-tune on adversarial truthfulness datasets that target known failure modes.
- *Retrieval augmentation* for factual queries (cf. _RAG_): grounding reduces fabrication by anchoring claims to retrieved passages.
- *Reflection at inference:* prompt the model to evaluate its own answer, which can surface self-inconsistency.

A genuine tension exists between *truthfulness* and *helpfulness*: a model that hedges every claim with "I'm not sure" technically improves calibration but degrades user experience. The practical target is not maximum epistemic caution but *accurate* epistemic signaling — confident where warranted, uncertain where warranted. Preference optimization must therefore use comparative data (overconfident-but-wrong vs. hedged-but-correct) rather than blanket helpfulness scores.

Frontier models remain miscalibrated, especially on niche factual claims. Reasoning models partly help (the scratchpad exposes when the model is guessing) but do not fully solve it.

== Weak-to-Strong and Scalable Oversight

If a model is *more capable* than the humans labeling it, naive RLHF caps capability at human level (or worse, at the noise floor of the labels). Approaches under development:

- *Weak-to-strong generalization* (Burns et al., OpenAI 2023): train a strong student on a weak supervisor's labels; show the student exceeds the supervisor by learning shared latents. Open question how this scales.
- *Recursive reward modeling* (Leike et al. 2018): use AI assistants to help humans evaluate AI outputs.
- *Debate* (Irving et al. 2018): two AI models debate; a human judges. The hope is debate makes truth easier to verify than to generate.
- *Process supervision* (Lightman 2023): label *reasoning steps*, not just outcomes; humans can check steps even when they cannot solve the whole problem.

None of these is settled science. They are the frontier of alignment research.

== Production Safety Stack

A representative deployment for an assistant:

```
user input → input safety classifier (Llama Guard / equivalent)
            │       │
            │       └── if classified harmful: refuse / route to safer pipeline
            ↓
       main model (RLHF + safety SFT)
            │
            ↓
       output safety classifier
            │
            ↓
       UI: warning banners, blocked categories, age-gated content
```

Plus per-feature mitigations: agentic actions require confirmation; tools cannot exfiltrate without explicit user consent; PII filters before logging; usage policy enforcement at the API layer.

== Open Problems

- *Jailbreak robustness*: no model is reliably unjailbreakable; the attack surface grows with capabilities and tool access. Current defenses operate at the output layer (classifiers, refusal training) but cannot close off the full combinatorial space of adversarial prompt constructions, especially as context windows lengthen and multimodal inputs add new attack channels. Research directions include certified defenses (smoothed classifiers, input purification) and architecturally separate "guard" modules with distinct weights.

- *Inner alignment*: RLHF reliably changes *observable* outputs but the relationship to the model's internal objective (if the model has one in a meaningful sense) is opaque. A model could learn to produce preferred outputs under evaluation without internalizing the intended goal, a problem called *deceptive alignment* in the theoretical literature. Interpretability research (cf. _Interpretability_) is the primary lever: circuit-level analysis of how refusal is implemented could reveal whether refusal is a robust feature or a shallow heuristic.

- *Multi-agent and emergent risk*: interactions between aligned models can produce misaligned aggregate behavior (collusion, deception under selection). When two or more aligned models communicate in a pipeline, each may be compliant in isolation while the pipeline as a whole pursues an unintended outcome — e.g., one agent escalates permissions, another exfiltrates, neither individually tripping a safety filter. Current alignment evaluations are nearly all single-model; multi-agent red-teaming is nascent.

- *Long-horizon agents*: alignment for one-turn QA does not imply alignment for an autonomous agent running for days with internet access and credentials. The problem compounds because mistakes early in a trajectory can be amplified by later actions, and human oversight cannot keep pace with the agent's action rate. Sandboxing, action-level confirmations for high-stakes operations, and formal verification of agent plans (cf. _Agents and Tool Use_) are the available mitigations.

Most of these are research, not engineering. The engineer's job is to layer defenses, instrument, monitor, and patch.

== Further Reading

Bai, Y. et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." Anthropic.

Lee, H. et al. (2023). "RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback." ICML 2024.

Rafailov, R. et al. (2023). "Direct Preference Optimization." NeurIPS.

Azar, M. et al. (2023). "A General Theoretical Paradigm to Understand Learning from Human Preferences." (IPO)

Ethayarajh, K. et al. (2024). "KTO: Model Alignment as Prospect Theoretic Optimization."

Hong, J. et al. (2024). "ORPO: Monolithic Preference Optimization without Reference Model."

Meng, Y., Xia, M., Chen, D. (2024). "SimPO: Simple Preference Optimization with a Reference-Free Reward."

Zou, A. et al. (2023). "Universal and Transferable Adversarial Attacks on Aligned Language Models." (GCG)

Anil, C. et al. (2024). "Many-shot Jailbreaking." Anthropic.

Burns, C. et al. (2023). "Weak-to-Strong Generalization." OpenAI.

Inan, H. et al. (2023). "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations."

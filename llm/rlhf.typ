= RLHF and Alignment

Pretraining optimizes a language model to predict the next token across a massive corpus of internet text. This produces a model that is fluent and knowledgeable, but not necessarily _helpful_, _harmless_, or _honest_. The distributional mismatch between "predict tokens from Common Crawl" and "assist a user safely and usefully" is the alignment problem in its practical form.

Reinforcement Learning from Human Feedback (RLHF) is the dominant technique for bridging this gap. It encodes human preferences into a reward signal and uses that signal to steer the policy (the LLM) toward behavior aligned with user intent.

*Why pretraining alone is insufficient:*

- *Task mismatch:* Pretraining data contains both good and bad text; the model learns to reproduce the full distribution, not just the desirable subset.
- *Format mismatch:* A completion model trained on web text will not naturally produce structured, helpful, conversational responses.
- *Safety mismatch:* The model may reproduce harmful content that appears in the training corpus.
- *Sycophancy and verbosity:* Without explicit feedback, models tend toward superficially plausible rather than correct answers.

_See also: llm/introduction.typ (pretraining objectives and data pipelines)._

== The RLHF Pipeline

The standard RLHF pipeline consists of three stages:

```
┌──────────────────────────────────────────────────────────────────┐
│                      RLHF PIPELINE                               │
│                                                                  │
│  Stage 1: Supervised Fine-Tuning (SFT)                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Pretrained LLM  ──►  SFT on (prompt, response) pairs  │    │
│  │                        curated by human annotators      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  Stage 2: Reward Model Training                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Human annotators rank pairs of SFT outputs             │    │
│  │  Reward model R_φ trained on pairwise preferences       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                              ▼                                   │
│  Stage 3: RL Fine-Tuning (PPO)                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Policy π_θ (SFT model)  generates response y           │    │
│  │  Reward:  r = R_φ(x, y) - β · KL(π_θ || π_ref)        │    │
│  │  PPO update on π_θ to maximize expected reward          │    │
│  └─────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

*Stage 1 — Supervised Fine-Tuning (SFT):* A small dataset of high-quality (prompt, ideal response) pairs — typically 10k–100k examples — is used to fine-tune the pretrained model with standard cross-entropy loss. This transforms a completion model into a rough instruction-follower and provides a stable initialization for RL.

*Stage 2 — Reward Model:* Human annotators compare pairs of responses to the same prompt and indicate which is preferred. A separate model (usually the SFT model with a scalar head) is trained on these preferences to predict a reward score.

*Stage 3 — RL Fine-Tuning:* The SFT model is used as a policy and updated to maximize expected reward under the reward model, while a KL divergence penalty prevents excessive deviation from the reference (SFT) model.

== Reward Model

=== Bradley-Terry Preference Model

Given a prompt $x$, a preferred response $y_w$ ("winner"), and a rejected response $y_l$ ("loser"), the Bradley-Terry model defines the probability that $y_w$ is preferred over $y_l$:

$ P(y_w succ y_l | x) = sigma(r_phi (x, y_w) - r_phi (x, y_l)) $

where $sigma$ is the sigmoid function and $r_phi (x, y)$ is the scalar reward predicted by the reward model with parameters $phi$.

=== Pairwise Ranking Loss

The training objective is to maximize the log-likelihood of observed human preferences:

$ cal(L)_"RM" (phi) = -bb(E)_((x, y_w, y_l) ~ cal(D)) [log sigma(r_phi (x, y_w) - r_phi (x, y_l))] $

This is a binary cross-entropy loss on the margin between chosen and rejected reward scores. The reward model is typically implemented as the SFT LLM with the unembedding head replaced by a single linear layer mapping the final hidden state to a scalar.

*PyTorch reward model training step:*

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

class RewardModel(nn.Module):
    def __init__(self, base_model_name: str):
        super().__init__()
        # Load backbone (SFT model), remove lm_head
        self.backbone = AutoModelForCausalLM.from_pretrained(base_model_name)
        d_model = self.backbone.config.hidden_size
        # Replace language model head with scalar head
        self.reward_head = nn.Linear(d_model, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        # Get last-token hidden state as the sequence representation
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Shape: (batch, seq_len, d_model)
        last_hidden = outputs.hidden_states[-1]
        # Use the last non-padding token
        seq_lens = attention_mask.sum(dim=1) - 1  # (batch,)
        last_token = last_hidden[
            torch.arange(last_hidden.size(0)), seq_lens
        ]  # (batch, d_model)
        reward = self.reward_head(last_token).squeeze(-1)  # (batch,)
        return reward


def reward_model_step(
    model: RewardModel,
    optimizer: torch.optim.Optimizer,
    chosen_ids: torch.Tensor,       # (batch, seq_len)
    chosen_mask: torch.Tensor,
    rejected_ids: torch.Tensor,     # (batch, seq_len)
    rejected_mask: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad()

    r_chosen   = model(chosen_ids, chosen_mask)    # (batch,)
    r_rejected = model(rejected_ids, rejected_mask) # (batch,)

    # Bradley-Terry loss: -log σ(r_w - r_l)
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()

    loss.backward()
    optimizer.step()

    # Accuracy: fraction where chosen reward > rejected reward
    acc = (r_chosen > r_rejected).float().mean().item()
    return loss.item(), acc
```

*Practical notes:*
- Reward models are sensitive to prompt/response formatting; use identical tokenization as SFT.
- Add a mean-zero regularization term $lambda dot bb(E)[r^2]$ to prevent reward scale drift.
- Typical accuracy on held-out preference pairs: 65–75\% for human-level annotators; 70–80\% for a well-trained reward model.

== PPO in the LLM Context

=== Objective

The RL fine-tuning objective is:

$ max_(pi_theta) bb(E)_(x ~ cal(D), y ~ pi_theta (dot | x)) [R_phi (x, y)] - beta dot "KL"(pi_theta (dot | x) || pi_"ref" (dot | x)) $

where:
- $pi_theta$ is the policy (LLM being trained)
- $pi_"ref"$ is the frozen reference policy (SFT model)
- $R_phi (x, y)$ is the reward model score
- $beta$ is the KL penalty coefficient (typically 0.01–0.1)
- $x$ is sampled from a prompt dataset $cal(D)$
- $y$ is sampled token-by-token from $pi_theta$

=== KL Penalty Derivation

The per-token KL divergence between the policy and reference is:

$ "KL"(pi_theta || pi_"ref") = sum_(t=1)^T sum_(v in cal(V)) pi_theta (v | x, y_(< t)) log (pi_theta (v | x, y_(< t))) / (pi_"ref" (v | x, y_(< t))) $

In practice, for the sampled sequence $y = (y_1, ..., y_T)$, the KL is approximated token-by-token along the sampled trajectory:

$ "KL"_"sample" = sum_(t=1)^T log (pi_theta (y_t | x, y_(< t))) / (pi_"ref" (y_t | x, y_(< t))) $

The per-token reward augmented with KL becomes:

$ r_t = cases(R_phi (x, y) - beta dot "KL"_"sample" & "if" t = T, -beta dot log (pi_theta (y_t | x, y_(< t))) / (pi_"ref" (y_t | x, y_(< t))) & "otherwise") $

This formulation adds KL cost at every token and terminal reward only at the end of the sequence.

=== PPO Clipping

PPO prevents destructively large policy updates using a clipped surrogate objective. For each token position $t$, define the importance sampling ratio:

$ rho_t = (pi_theta (y_t | x, y_(< t))) / (pi_"old" (y_t | x, y_(< t))) $

The PPO-clip objective is:

$ cal(L)_"PPO" = bb(E)_t [min(rho_t hat(A)_t, "clip"(rho_t, 1 - epsilon, 1 + epsilon) hat(A)_t)] $

where $hat(A)_t$ is the estimated advantage at token $t$ (from a value function critic), and $epsilon = 0.2$ is the clip ratio.

*Why clipping is necessary:* In the LLM setting, the policy is a distribution over a vocabulary of 50k–130k tokens. A single gradient step without clipping can move the policy so far that the old rollouts become off-policy and the importance weights $rho_t$ become arbitrarily large, destabilizing training. Clipping ensures that no single update step extracts too much signal from any rollout.

=== PPO Training Loop (PyTorch Pseudocode)

```python
import torch
import torch.nn.functional as F
from torch.optim import AdamW

# Hyperparameters
BETA        = 0.05    # KL penalty coefficient
CLIP_EPS    = 0.2     # PPO clip ratio
GAE_LAMBDA  = 0.95    # Generalized Advantage Estimation lambda
GAMMA       = 1.0     # Discount factor (typically 1.0 for LLMs)
N_PPO_EPOCHS = 4      # Inner PPO epochs per rollout batch
LR          = 1e-6

def compute_advantages(rewards, values, gamma=GAMMA, lam=GAE_LAMBDA):
    """Generalized Advantage Estimation (GAE)."""
    # rewards, values: (batch, seq_len)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    T = rewards.size(1)
    for t in reversed(range(T)):
        next_val = values[:, t + 1] if t < T - 1 else torch.zeros(rewards.size(0))
        delta = rewards[:, t] + gamma * next_val - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_step(
    policy,          # LLM being trained
    ref_policy,      # Frozen SFT model
    critic,          # Value function (separate head or model)
    reward_model,    # Frozen reward model
    optimizer,
    prompts,         # (batch, prompt_len)
    prompt_mask,
):
    # ── Phase 1: Rollout ──────────────────────────────────────────
    with torch.no_grad():
        # Sample responses from current policy
        responses = policy.generate(
            prompts, max_new_tokens=512, do_sample=True, temperature=1.0
        )  # (batch, prompt_len + response_len)

        # Compute log-probs under current policy (becomes "old policy")
        old_logprobs = policy.log_probs(prompts, responses)   # (batch, T)

        # Compute log-probs under reference policy
        ref_logprobs = ref_policy.log_probs(prompts, responses) # (batch, T)

        # Per-token KL penalty
        kl_per_token = old_logprobs - ref_logprobs  # (batch, T)

        # Terminal reward from reward model
        terminal_reward = reward_model(prompts, responses)  # (batch,)

        # Assemble per-token rewards
        rewards = -BETA * kl_per_token                          # (batch, T)
        rewards[:, -1] += terminal_reward                       # add at EOS

        # Compute baseline values
        values = critic(prompts, responses)  # (batch, T)

    advantages, returns = compute_advantages(rewards, values)

    # ── Phase 2: PPO Update ───────────────────────────────────────
    for _ in range(N_PPO_EPOCHS):
        optimizer.zero_grad()

        # Forward pass: new log-probs
        new_logprobs = policy.log_probs(prompts, responses)  # (batch, T)

        # Importance sampling ratio
        ratio = torch.exp(new_logprobs - old_logprobs)       # (batch, T)

        # Normalize advantages
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Clipped surrogate objective
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value function loss
        new_values = critic(prompts, responses)
        value_loss = F.mse_loss(new_values, returns.detach())

        loss = policy_loss + 0.5 * value_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
```

*Practical PPO challenges in the LLM setting:*

#table(
  columns: (auto, auto),
  [*Challenge*], [*Mitigation*],
  [Policy and critic share backbone — updates interfere], [Separate heads; freeze backbone layers; low LR],
  [Long sequences make per-token credit assignment noisy], [Reward at EOS only; GAE with $lambda$ close to 1],
  [Reward hacking / overoptimization], [KL penalty $beta$; stop training by KL budget],
  [High memory: need policy, ref policy, critic, reward model], [ZeRO-3 sharding; 8-bit optimizer states],
  [Rollout generation is slow], [Batch rollouts; use vLLM for generation],
)

== Direct Preference Optimization (DPO)

DPO (Rafailov et al., 2023) eliminates the reward model and RL training loop entirely by directly optimizing the policy on preference data.

=== Full Derivation

*Start from the RLHF objective.* The constrained optimization problem is:

$ max_(pi_theta) bb(E)_(x ~ cal(D)) bb(E)_(y ~ pi_theta (dot | x)) [R(x, y)] - beta dot "KL"(pi_theta (dot | x) || pi_"ref" (dot | x)) $

This is a KL-constrained reward maximization problem. Its unique optimal solution in closed form is:

$ pi^* (y | x) = (1 / Z(x)) pi_"ref" (y | x) exp((1 / beta) R(x, y)) $

where $Z(x) = sum_y pi_"ref" (y | x) exp((1/beta) R(x, y))$ is the partition function.

*Solve for the reward.* Taking logs and rearranging:

$ log pi^* (y | x) = log pi_"ref" (y | x) + (1/beta) R(x, y) - log Z(x) $

$ R(x, y) = beta log (pi^* (y | x)) / (pi_"ref" (y | x)) + beta log Z(x) $

The partition function $Z(x)$ depends only on $x$, not on $y$. It cancels when we compute the reward _difference_ between two responses.

*Substitute into Bradley-Terry.* The probability that $y_w$ is preferred over $y_l$ under the optimal policy is:

$ P^* (y_w succ y_l | x) = sigma(R(x, y_w) - R(x, y_l)) $

$ = sigma lr((beta log (pi^* (y_w | x)) / (pi_"ref" (y_w | x)) - beta log (pi^* (y_l | x)) / (pi_"ref" (y_l | x)))) $

*Replace $pi^*$ with $pi_theta$.* The DPO training objective is the negative log-likelihood of preferences under this parameterization:

$ cal(L)_"DPO" (theta) = -bb(E)_((x, y_w, y_l) ~ cal(D)) [log sigma lr((beta (log (pi_theta (y_w | x)) / (pi_"ref" (y_w | x)) - log (pi_theta (y_l | x)) / (pi_"ref" (y_l | x)))))] $

This is the DPO loss. The log-ratios $log pi_theta (y | x) / pi_"ref" (y | x)$ are computed as the sum of per-token log-probability differences over the response tokens.

*Interpretation of the gradient:* The DPO gradient increases the likelihood of chosen responses and decreases the likelihood of rejected responses, weighted by how much the current model disagrees with the preference label (via the $sigma$ term — the gradient is small when the model already assigns the correct ordering confidently).

=== PyTorch DPO Training Step

```python
import torch
import torch.nn.functional as F

def sequence_log_prob(model, input_ids, attention_mask, response_mask):
    """
    Compute sum of log-probs for response tokens only.
    response_mask: 1 for response tokens, 0 for prompt tokens.
    Returns: (batch,) tensor of sum log-probs.
    """
    with torch.no_grad() if not model.training else torch.enable_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # Shift: predict token t from tokens 0..t-1
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)
    target_ids = input_ids[:, 1:]                          # (B, T-1)
    # Gather log-prob of actual tokens
    token_log_probs = log_probs.gather(
        2, target_ids.unsqueeze(-1)
    ).squeeze(-1)  # (B, T-1)
    # Mask to response tokens only
    resp_mask = response_mask[:, 1:].float()
    return (token_log_probs * resp_mask).sum(dim=-1)  # (B,)


def dpo_loss(
    policy,            # LLM being trained
    ref_policy,        # Frozen reference model
    chosen_ids,        # (B, T) input_ids for prompt + chosen response
    chosen_attn,       # (B, T) attention mask
    chosen_resp_mask,  # (B, T) 1 for chosen response tokens
    rejected_ids,      # (B, T)
    rejected_attn,
    rejected_resp_mask,
    beta: float = 0.1,
):
    # Log-probs under policy
    pi_logprob_w  = sequence_log_prob(policy, chosen_ids,   chosen_attn,   chosen_resp_mask)
    pi_logprob_l  = sequence_log_prob(policy, rejected_ids, rejected_attn, rejected_resp_mask)

    # Log-probs under frozen reference
    with torch.no_grad():
        ref_logprob_w = sequence_log_prob(ref_policy, chosen_ids,   chosen_attn,   chosen_resp_mask)
        ref_logprob_l = sequence_log_prob(ref_policy, rejected_ids, rejected_attn, rejected_resp_mask)

    # Log-ratios: implicit reward signal
    log_ratio_w = pi_logprob_w - ref_logprob_w  # (B,)
    log_ratio_l = pi_logprob_l - ref_logprob_l  # (B,)

    # DPO loss
    loss = -F.logsigmoid(beta * (log_ratio_w - log_ratio_l)).mean()

    # Diagnostic: implicit reward margin
    reward_margin = beta * (log_ratio_w - log_ratio_l).mean().item()
    return loss, reward_margin
```

=== PPO vs DPO vs REINFORCE vs GRPO

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Property*], [*PPO*], [*DPO*], [*REINFORCE*], [*GRPO*],
  [Requires reward model], [Yes], [No], [Yes], [Yes],
  [Requires critic/value network], [Yes], [No], [No], [No],
  [Online sampling], [Yes], [No], [Yes], [Yes],
  [Stability], [High (clipping)], [High (supervised)], [Low (high variance)], [Medium],
  [Memory overhead], [Very high], [Low (2× ref model)], [High], [Medium],
  [Reward hacking risk], [Medium], [Low], [High], [Medium],
  [Supports sparse rewards], [Yes], [No], [Yes], [Yes],
  [Introduced by], [Schulman 2017], [Rafailov 2023], [Williams 1992], [Shao 2024],
  [Used in], [InstructGPT, Claude], [Zephyr, Llama-2-chat], [Early RLHF], [DeepSeek-R1],
)

*DPO limitations:*
- Requires static offline preference data; cannot improve from online rollouts.
- Assumes the reference model is a reasonable prior; degrades if SFT quality is poor.
- Sensitive to data quality: noisy preference labels cause instability.
- Length bias: longer responses tend to accumulate more log-prob mass, inflating $log pi(y|x)$.

== GRPO: Group Relative Policy Optimization

GRPO (Shao et al., 2024), used in DeepSeek-R1, eliminates the critic network by normalizing rewards within a group of sampled responses. This avoids training a separate value function while still reducing gradient variance.

=== Algorithm

For each prompt $x_i$ in a batch, sample a group of $G$ responses:

$ {y_i^1, y_i^2, ..., y_i^G} ~ pi_theta_"old" (dot | x_i) $

Compute a scalar reward $r_i^j = R(x_i, y_i^j)$ for each response (from a reward model or a rule-based verifier).

*Normalize rewards within the group* to form advantages:

$ hat(A)_i^j = (r_i^j - "mean"_j (r_i^j)) / ("std"_j (r_i^j) + epsilon) $

The GRPO objective (with PPO-style clipping) is:

$ cal(L)_"GRPO" (theta) = -bb(E)_i (1/G) sum_(j=1)^G [min(rho_i^j hat(A)_i^j, "clip"(rho_i^j, 1-epsilon, 1+epsilon) hat(A)_i^j) - beta dot "KL"(pi_theta || pi_"ref")] $

where $rho_i^j = pi_theta (y_i^j | x_i) / pi_theta_"old" (y_i^j | x_i)$ is the per-response importance ratio (not per-token, to simplify implementation).

*Why this works:* The group mean serves as a baseline (analogous to the value function in PPO). Responses better than average in the group get positive advantage; worse-than-average responses get negative advantage. Within-group normalization ensures stable gradient magnitudes regardless of absolute reward scale.

=== PyTorch GRPO Training Loop

```python
import torch
import torch.nn.functional as F

def compute_sequence_logprob(model, input_ids, attention_mask, response_mask):
    """Sum log-probs of response tokens (same as DPO helper above)."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    token_lp = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    return (token_lp * response_mask[:, 1:].float()).sum(dim=-1)


def grpo_step(
    policy,
    ref_policy,
    reward_fn,          # callable: (prompts, responses) -> (batch,) rewards
    optimizer,
    prompts,            # (B, P) tokenized prompts
    prompt_mask,
    G: int = 8,         # group size
    beta: float = 0.04,
    clip_eps: float = 0.2,
):
    B = prompts.size(0)
    device = prompts.device

    # ── Sample G responses per prompt ────────────────────────────
    # Expand prompts: (B*G, P)
    prompts_exp  = prompts.repeat_interleave(G, dim=0)
    pmask_exp    = prompt_mask.repeat_interleave(G, dim=0)

    with torch.no_grad():
        responses = policy.generate(
            prompts_exp, max_new_tokens=512, do_sample=True, temperature=1.0
        )  # (B*G, P+T)
        response_mask = build_response_mask(prompts_exp, responses)

        # Old policy log-probs (for importance ratio denominator)
        old_logprobs = compute_sequence_logprob(
            policy, responses, pmask_exp, response_mask
        )  # (B*G,)

        # Reference log-probs (for KL penalty)
        ref_logprobs = compute_sequence_logprob(
            ref_policy, responses, pmask_exp, response_mask
        )  # (B*G,)

        # Rewards: (B*G,)
        rewards = reward_fn(prompts_exp, responses)

    # ── Normalize within group ────────────────────────────────────
    rewards = rewards.view(B, G)                      # (B, G)
    mean_r  = rewards.mean(dim=1, keepdim=True)       # (B, 1)
    std_r   = rewards.std(dim=1, keepdim=True) + 1e-8
    advantages = ((rewards - mean_r) / std_r).view(B * G)  # (B*G,)

    # ── PPO-clip update ───────────────────────────────────────────
    optimizer.zero_grad()

    new_logprobs = compute_sequence_logprob(
        policy, responses, pmask_exp, response_mask
    )  # (B*G,)

    ratio = torch.exp(new_logprobs - old_logprobs)    # (B*G,)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Per-sequence KL penalty
    kl = (old_logprobs - ref_logprobs).mean()
    loss = policy_loss + beta * kl

    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return loss.item(), kl.item()
```

*GRPO in DeepSeek-R1:* Uses verifiable rewards (math/code correctness), which are binary (0 or 1). The group normalization is particularly effective here: within a group of 8 rollouts, correct solutions get advantage $approx +1$ and incorrect solutions get $approx -1$, naturally calibrating the update magnitude.

== Constitutional AI and RLAIF

Constitutional AI (Anthropic, 2022) addresses a bottleneck in RLHF: the cost of collecting human preference labels at scale. Two key ideas:

=== RLAIF: AI-Generated Preference Labels

Instead of human annotators ranking response pairs, a capable AI model (the "feedback model") is prompted to evaluate and rank responses according to a set of principles. The generated preferences are then used to train a reward model identically to standard RLHF.

*Prompt template for AI feedback:*

```
Which of the following responses is more helpful, harmless, and honest?
Principle: [principle from constitution]

Response A: [response_a]
Response B: [response_b]

Choose A or B and briefly explain.
```

This approach scales cheaply and allows rapid iteration on the preference criteria (the "constitution"). Research findings show RLAIF-trained models are competitive with RLHF on helpfulness metrics while being more consistent on safety criteria.

=== Critique-Revision (CAI)

In the supervised phase of Constitutional AI, the model is asked to:

1. *Critique* its own response against a constitutional principle.
2. *Revise* the response to better satisfy the principle.

This produces (original prompt, revised response) pairs for SFT, replacing the need for human-written demonstrations.

```python
# Simplified CAI SFT data generation
def generate_cai_revision(model, prompt, response, principle):
    critique_prompt = f"""
    Human: {prompt}
    Assistant: {response}

    Critique the above response with respect to the following principle:
    "{principle}"

    What would be a better response?
    """
    revision = model.generate(critique_prompt)
    return revision  # Use as SFT target
```

*Advantages of RLAIF/CAI:*
- Scales to millions of preference pairs at low cost.
- Principles are explicit and auditable (unlike implicit human preferences).
- Iterative revision creates a curriculum of improving responses.

*Limitations:*
- AI feedback can inherit biases from the feedback model.
- Circular: feedback model quality bounds the trained model quality.
- Constitutional principles may conflict; resolution requires heuristics.

== Rejection Sampling Fine-Tuning (RST)

Rejection Sampling Fine-Tuning (also called Best-of-N supervised fine-tuning or iterative SFT) is a simpler alternative to RL that often achieves competitive results.

=== Algorithm

For each prompt $x_i$:

1. Sample $N$ responses from the current policy: $y_i^1, ..., y_i^N ~ pi_theta (dot | x_i)$.
2. Score each with the reward model: $r_i^j = R(x_i, y_i^j)$.
3. Keep the top-$k$ responses by reward (or filter by threshold $r > tau$).
4. Fine-tune the policy on the filtered (prompt, response) pairs with supervised cross-entropy loss.

Repeat for multiple rounds, updating the policy between rounds (iterative RST).

```python
def rejection_sampling_finetuning(
    policy,
    reward_model,
    sft_optimizer,
    prompts,
    N: int = 16,       # candidate responses per prompt
    k: int = 1,        # keep top-k per prompt
    n_rounds: int = 3,
):
    for round_idx in range(n_rounds):
        accepted_prompts   = []
        accepted_responses = []

        for prompt in prompts:
            # Sample N candidate responses
            candidates = policy.generate(
                prompt.unsqueeze(0).repeat(N, 1),
                do_sample=True, temperature=1.0
            )  # (N, T)

            # Score all candidates
            rewards = reward_model(
                prompt.unsqueeze(0).repeat(N, 1), candidates
            )  # (N,)

            # Select top-k
            top_k_idx = rewards.topk(k).indices
            for idx in top_k_idx:
                accepted_prompts.append(prompt)
                accepted_responses.append(candidates[idx])

        # Supervised fine-tuning on accepted pairs
        for prompt, response in zip(accepted_prompts, accepted_responses):
            sft_loss = cross_entropy_loss(policy, prompt, response)
            sft_optimizer.zero_grad()
            sft_loss.backward()
            sft_optimizer.step()

        print(f"Round {round_idx + 1}: kept {len(accepted_responses)} responses")
```

*Comparison: RST vs PPO vs DPO:*

#table(
  columns: (auto, auto, auto, auto),
  [*Property*], [*RST*], [*PPO*], [*DPO*],
  [Reward model needed], [Yes], [Yes], [No],
  [Complexity], [Low], [High], [Low],
  [Online samples], [Yes], [Yes], [No],
  [Handles sparse reward], [Yes], [Yes], [No],
  [Training instability], [Low], [Medium-High], [Low],
  [Sample efficiency], [Low ($N$ samples per update)], [High], [Depends on data],
)

RST is particularly effective when a verifiable reward signal is available (e.g., code execution, math verification), as the "reward model" is exact. This is the approach used in AlphaCode and early Codex fine-tuning.

== Safety Considerations

=== Reward Hacking

Reward hacking occurs when the policy finds a strategy that achieves high reward model scores without actually producing the behavior the reward was intended to capture. Because the reward model is an imperfect proxy for human preferences, it has exploitable blind spots.

*Examples:*
- *Length hacking:* Models learn that longer, verbose responses tend to receive higher human ratings. The policy exploits this by generating excessive hedging, repetition, or irrelevant padding.
- *Sycophancy:* Models agree with users' stated opinions, even when incorrect, because agreement is rewarded in preference comparisons.
- *Format gaming:* Models produce well-formatted but shallow responses (bulleted lists, bold headers) that look polished but lack substance.
- *Adversarial reward exploitation:* For neural reward models, the policy may find input patterns that cause the reward model to output high scores independently of content quality.

=== Goodhart's Law

Goodhart's Law states: _"When a measure becomes a target, it ceases to be a good measure."_ In the RLHF context: as the policy is optimized against the reward model, the correlation between reward model score and true quality degrades.

*Overoptimization:* Empirically, RLHF performance as a function of KL divergence from the reference policy follows an inverted-U curve. Early in training, reward increases with KL (the model is improving). After some KL budget $delta_"opt"$, the reward model is exploited and gold-standard quality (measured by a held-out oracle) begins to decrease even as the proxy reward continues to rise.

$ "True quality" approx a sqrt(delta) - b delta $

where $delta = "KL"(pi_theta || pi_"ref")$ and $a, b > 0$ are empirically fitted constants (Gao et al., 2022).

*Mitigations:*
- *KL budget:* Set a maximum allowed KL divergence and stop training when exceeded.
- *Ensemble reward models:* Train multiple reward models on different data splits; use the minimum score or a conservative aggregate to reduce exploitability.
- *Online reward model updates:* Periodically retrain the reward model on new policy rollouts, maintaining a moving target that tracks the policy distribution.
- *Constitutional constraints:* Hard-code inviolable rules (no CSAM, no synthesis instructions) that override the reward signal.
- *Evaluation diversity:* Track multiple independent evaluation metrics (human raters, automated evals, adversarial probes) to detect overoptimization early.

=== Alignment Tax

Fine-tuning for alignment can reduce capability on certain benchmarks. This is called the _alignment tax_. For example:

- RLHF reduces willingness to produce harmful content, but can also make models overly cautious and less willing to answer legitimate edge-case questions.
- The KL penalty that prevents reward hacking also prevents the model from moving far from the SFT distribution, limiting how much alignment improvement is possible.

Balancing the alignment tax against safety improvements is an active area of research, with techniques like representation engineering, activation steering, and circuit-level interpretability being explored as lower-cost alternatives.

== References

1. Christiano, P., et al. (2017). _Deep Reinforcement Learning from Human Preferences._ NeurIPS 2017.

2. Stiennon, N., et al. (2020). _Learning to Summarize with Human Feedback._ NeurIPS 2020.

3. Ouyang, L., et al. (2022). _Training Language Models to Follow Instructions with Human Feedback (InstructGPT)._ NeurIPS 2022.

4. Bai, Y., et al. (2022). _Constitutional AI: Harmlessness from AI Feedback._ Anthropic Technical Report.

5. Rafailov, R., et al. (2023). _Direct Preference Optimization: Your Language Model is Secretly a Reward Model._ NeurIPS 2023.

6. Schulman, J., et al. (2017). _Proximal Policy Optimization Algorithms._ arXiv:1707.06347.

7. Gao, L., et al. (2022). _Scaling Laws for Reward Model Overoptimization._ arXiv:2210.10760.

8. Shao, Z., et al. (2024). _DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)._ arXiv:2402.03300.

9. DeepSeek-AI. (2025). _DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning._ arXiv:2501.12948.

10. Williams, R. J. (1992). _Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning._ Machine Learning, 8(3-4):229–256.

_See also: llm/transformer-architecture.typ (LLM internals), gpu-architecture/ml-workloads.typ (training infrastructure), gpu-architecture/multi-gpu.typ (distributed training for RL)._

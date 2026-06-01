= Reasoning Models

A "reasoning model" allocates substantial *inference-time compute* to a chain of intermediate steps before emitting a final answer. The transformation from a one-shot generator to a deliberate reasoner — through prompting, training, or both — defines the o1 / R1 generation of models. This chapter covers chain-of-thought prompting and its variants, the algorithms (self-consistency, ToT, GoT, MCTS-style search), the RL training that produces "native" reasoners (DeepSeek-R1, GRPO, DAPO), reward design over verifiable tasks, and the engineering implications (long reasoning traces, KV cache for thinking tokens, hidden vs. visible thoughts).

*See also:* _RLHF_ (PPO-family algorithms reasoning RL extends), _Inference Optimization_ (long reasoning traces are decode-bound), _Evaluation_ (verifiable rewards for math/code benchmarks).

== Chain of Thought

Wei et al. (2022) showed that simply prompting an LLM with worked examples that include intermediate steps ("Let's think step by step") dramatically improves performance on math, logic, and symbolic reasoning. The model emits intermediate tokens — "thoughts" — before its answer, and accuracy on GSM8K jumps from ~17% to ~57% for PaLM 540B.

*Mechanism (current understanding):* CoT lets the model use *output tokens as scratch* — each token's computation depends on all prior tokens via attention, so a chain of tokens implements a deeper effective computation than a single forward pass. The model's expressiveness goes from $"TC"^0$ (constant-depth) to something like log-depth in the number of CoT tokens (Feng et al. 2023).

*Zero-shot CoT* (Kojima et al. 2022): append "Let's think step by step." with no examples. Works less reliably than few-shot but scales better as instruction-tuned models internalize the pattern.

```python
PROMPT = """
Q: There are 3 apples in a basket. Jane adds 5 more, then eats 2. How many remain?
A: Let's think step by step.
   Start with 3 apples. After Jane adds 5, there are 3 + 5 = 8.
   She eats 2, leaving 8 - 2 = 6.
   Answer: 6.

Q: {question}
A: Let's think step by step.
"""
```

== Self-Consistency

A single CoT trace can be wrong; multiple traces often have a "modal" correct answer. *Self-consistency* (Wang et al. 2022) samples $N$ CoT traces at $T > 0$ and majority-votes the final answer:

```python
from collections import Counter

def self_consistent_answer(prompt, model, N=10, T=0.7):
    answers = [model.generate(prompt, temperature=T).extract_answer()
               for _ in range(N)]
    return Counter(answers).most_common(1)[0][0]
```

At $N = 40$ on GSM8K, self-consistency adds 5–10 points over greedy CoT. Cost scales linearly with $N$.

*Weighted self-consistency* weights each trace by the model's log-prob of its own answer, slightly outperforming majority voting at smaller $N$.

== Tree and Graph of Thoughts

Self-consistency searches independent traces. ToT (Yao et al. 2023) interleaves search and evaluation:

1. Generate $k$ candidate next-steps from the current node.
2. Have the model *score* each candidate (or evaluate against a heuristic).
3. Expand the highest-scoring branches.
4. Repeat until terminal; backtrack on dead ends.

GoT (Besta et al. 2024) generalizes to a DAG: branches can be *merged* by combining partial reasoning. Useful when sub-problems recur (algorithmic problems, theorem proving).

Both are *test-time* algorithms — no model weights change. Compute scales with branching factor and depth; gains are large on combinatorial tasks (24-game, crossword) and small on tasks where a single coherent trace suffices.

== Process vs. Outcome Reward

Two distinct objectives for training reasoners:

- *Outcome Reward Model (ORM):* reward signals come from the final answer only. Simple, requires a verifier (ground truth, executor). Used by DeepSeek-R1-Zero, o1.
- *Process Reward Model (PRM):* a separate model scores each *step* in the reasoning trace. Trained on human-labeled (or LLM-labeled) per-step correctness. PRM800K (Lightman et al. 2023) provides 800k step labels for GSM8K-style problems.

PRMs give denser reward signals and can catch reasoning that is wrong *for the right answer*. They are expensive to train and prone to reward hacking (the trace looks reasonable but the model has memorized the surface form).

The DeepSeek-R1 result (below) showed that ORM with verifiable tasks + scale is competitive with PRM and avoids the labeling burden.

== RL on Verifiable Rewards

The o1 / R1 generation discovered that *RL on tasks with cheap automatic verification* (math problems with checkable answers, code with unit tests) drives the model to invent its own long, self-correcting CoT traces — without explicit step-by-step supervision.

=== Reward Design

For math:
$ r = cases(+1 & "if final answer matches ground truth", -0.1 & "if format invalid (no boxed answer)", 0 & "otherwise") $

For code:
$ r = "(unit tests passed)" / "(unit tests total)" $

The format reward is small but essential — it teaches the model to emit a parseable answer (`\\boxed{...}`).

=== PPO Recap

Standard PPO for RLHF (covered in _RLHF_) requires a critic network of comparable size to the policy, with significant memory cost. Reasoning RL uses two simplified variants.

=== GRPO (Group Relative Policy Optimization)

DeepSeekMath (Shao et al. 2024) and DeepSeek-R1 introduced GRPO: for each prompt, sample $G$ trajectories from the current policy; use the *group mean and std* of rewards as the baseline:

$ A_i = (r_i - mu) / sigma, quad mu = 1/G sum_i r_i $

Update with PPO-style clipped objective on $A_i$. No critic — the group itself plays the baseline role. KL divergence to a reference model regularizes (unbiased estimator).

$ cal(L)_"GRPO" = -EE_("group") [min(r_(theta) A_i, "clip"(r_(theta), 1-epsilon, 1+epsilon) A_i)] + beta D_"KL"(pi_(theta) || pi_"ref") $

Practical $G = 16$ to $64$, $epsilon = 0.2$, $beta = 0.001$. Memory roughly halves vs. PPO.

=== DAPO (Decoupled Clip and Dynamic Sampling)

DAPO (Yu et al. 2025) refines GRPO with:

1. *Clip-higher:* asymmetric PPO clip ($epsilon_"low" = 0.2, epsilon_"high" = 0.28$) preserves exploration on rare high-reward tokens.
2. *Dynamic sampling:* discard prompts where every trajectory in the group has identical reward (zero variance → zero gradient).
3. *Token-level loss:* sum (rather than mean) the loss over response tokens, so longer responses do not dilute their per-token gradient.
4. *Overlong reward shaping:* small negative reward proportional to length overflow, instead of zero, to soften the truncation cliff.

DAPO outperforms GRPO at the same compute on AIME24 (math) and HumanEval+.

=== Reasoning RL Recipe

```
SFT cold-start on a small CoT dataset (~10k worked examples)
  ↓
GRPO/DAPO rollouts: G=32 trajectories per problem, temperature 0.9
  ↓
Reward = (answer correct ? 1 : 0) + 0.05 × format_ok
  ↓
PPO update with KL to SFT init
  ↓
Iterate 1k–10k steps; ~1B reward queries total
```

DeepSeek-R1 reports the emergent appearance of:

- *Self-correction* ("Wait, that step is wrong, let me redo it...")
- *Verification* ("Let me check by substituting back...")
- *Backtracking* ("This path leads to a contradiction; try another.")

These behaviors are not in the SFT data. They emerge from the RL objective + diversity of sampled trajectories.

== Reasoning Trace Engineering

=== Thought Tokens

o1 separates *hidden* reasoning (not shown to user, billed but not returned) from the visible answer. Hidden tokens dominate the cost — a single hard problem can use 10–100k thought tokens.

Open-weight reasoners (DeepSeek-R1, QwQ) emit `<think> ... </think>` blocks that callers may strip before display. The model is *trained* to use this delimiter consistently; stripping at inference does not break behavior.

=== Inference Costs

A reasoning model decoding 32k thought tokens at 50 tok/s takes ~10 minutes per query. Implications:

- *Latency budgets* for production must be re-thought; chat-style "instant" responses are gone.
- *Batching* — long traces from different users can interleave (continuous batching).
- *Cost* — at 32k tokens × \$60/M output, a single complex query is \$2.

Serving systems benefit from *budget-aware* decoding: cap thought tokens at $T_"max"$, force a final answer attempt if not produced.

=== Self-Termination

Trained reasoners terminate their thinking when they "feel confident" (the policy learns this through RL). Failure mode: infinite or near-infinite traces on truly hard problems. Mitigation: hard stop-token budget at inference, retry with prompt nudges ("you have 1000 tokens left, produce your final answer").

== Verification and Search at Inference

Test-time *search* over reasoning is increasingly important:

- *Best-of-N + verifier:* generate $N$ traces, score each with a learned verifier or executor, return the highest-scoring. Used by AlphaCode, o1-style "reasoning effort" knob.
- *MCTS over CoT:* treat each reasoning step as a node, use UCB to explore. AlphaProof, rStar (Qi et al. 2024), Marco-o1 use variants.
- *Reflection:* after a candidate answer, prompt the model to *critique* its own trace; rewrite if a flaw is found (Madaan et al. 2023). Cheap improvement; degrades on overconfident models.

== Pitfalls

=== Reward Hacking

If the verifier accepts any string containing the correct number, the model learns to output `"3 7 12 42 ... answer: 42"`. Strict formatting (`\\boxed{42}`) and post-hoc parsing prevent this. For code, sandboxed execution with proper test isolation matters.

=== Length Hacking

Without length-aware rewards, models learn that *longer* traces win more rewards. Models talk to themselves indefinitely. DAPO's overlong reward shaping addresses this.

=== Mode Collapse

Group-relative methods (GRPO) zero out when all $G$ trajectories agree. After early training, models converge to one solution path per prompt — diversity collapses. Remedy: KL term to reference, temperature on rollouts, dynamic sampling (DAPO).

=== Distillation From Reasoners

You can SFT a smaller model on a frontier reasoner's traces (DeepSeek-R1-Distill-Qwen-7B). Gains are real but the student rarely matches the teacher on novel hard problems — it imitates the surface form of reasoning without the latent search capability. Use distillation for cost reduction, not as a substitute for RL training the small model directly.

== Further Reading

Wei, J. et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS.

Kojima, T. et al. (2022). "Large Language Models are Zero-Shot Reasoners." NeurIPS.

Wang, X. et al. (2022). "Self-Consistency Improves Chain of Thought Reasoning in Language Models." ICLR 2023.

Yao, S. et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models." NeurIPS.

Besta, M. et al. (2024). "Graph of Thoughts: Solving Elaborate Problems with Large Language Models." AAAI.

Lightman, H. et al. (2023). "Let's Verify Step by Step." ICLR 2024.

Shao, Z. et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300.

DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948.

Yu, Q. et al. (2025). "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." arXiv:2503.14476.

Madaan, A. et al. (2023). "Self-Refine: Iterative Refinement with Self-Feedback." NeurIPS.

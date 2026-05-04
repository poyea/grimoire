= Evaluation

Evaluating a large language model is harder than evaluating most software systems. A model with billions of parameters can excel at summarization while failing at arithmetic, ace multiple-choice science while hallucinating biographical facts, and refuse harmful requests while also refusing benign ones. No single number captures capability. This chapter covers the principal evaluation methodologies — from perplexity to human preference rankings — and the practical setup needed to reproduce published results.

*See also:* _llm/pretraining.typ_ (training loss, data quality), _llm/rlhf.typ_ (reward model evaluation), _llm/safety.typ_ (red-teaming).

== Perplexity

=== Definition

Perplexity measures how well a language model predicts a held-out corpus. Given a sequence of $N$ tokens $t_1, t_2, ..., t_N$, perplexity is defined as the exponentiated average negative log-likelihood under the model $p_theta$:

$
"PPL" = exp(-1/N sum_(i=1)^N log p_theta (t_i | t_1, ..., t_(i-1)))
$

A lower perplexity means the model assigns higher average probability to each token — it is less "surprised" by the text. A perplexity of 1 would be a perfect predictor; a uniform distribution over a vocabulary of size $V$ gives perplexity $V$ (e.g., 128 256 for LLaMA 3's tokenizer).

*Relationship to cross-entropy:* perplexity is simply $exp("CE")$ where CE is the mean cross-entropy in nats. Training loss in bits per byte (BPB) is a related unit common in character-level and byte-level models.

=== Computing Perplexity in PyTorch

For short sequences (shorter than the model's context window), perplexity is a straightforward forward pass:

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()

text = open("wikitext_sample.txt").read()
tokens = tokenizer(text, return_tensors="pt").input_ids.cuda()  # [1, N]

with torch.no_grad():
    logits = model(tokens).logits  # [1, N, V]

# shift: predict t_{i+1} from t_i
shift_logits = logits[:, :-1, :].contiguous()   # [1, N-1, V]
shift_labels = tokens[:, 1:].contiguous()        # [1, N-1]

loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
)
ppl = torch.exp(loss)
print(f"Perplexity: {ppl.item():.2f}")
```

=== Sliding Window for Long Texts

When the text is longer than the model's context window $L_"max"$, naive chunking is biased: the first token of each chunk has no context. The *sliding window* method avoids this by evaluating each token with a stride $s$ less than $L_"max"$, keeping a running context:

```python
def sliding_window_ppl(model, tokens, max_len=4096, stride=2048):
    """
    tokens: LongTensor [1, N]
    Returns perplexity over all tokens except the first max_len - stride.
    """
    N = tokens.size(1)
    nll_sum = 0.0
    n_tokens = 0

    for begin in range(0, N, stride):
        end   = min(begin + max_len, N)
        chunk = tokens[:, begin:end].cuda()

        # only score tokens in the new window (beyond the overlap)
        target_start = max(begin, begin + max_len - stride) - begin

        with torch.no_grad():
            logits = model(chunk).logits  # [1, T, V]

        shift_logits = logits[:, target_start - 1 : -1, :].contiguous()
        shift_labels = chunk[:, target_start:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
        )
        nll_sum += loss.item()
        n_tokens += shift_labels.numel()

        if end == N:
            break

    return torch.exp(torch.tensor(nll_sum / n_tokens))
```

A stride of $s = L_"max" / 2$ is the most common choice, balancing accuracy against compute.

=== Limitations

*Dataset contamination.* If the evaluation text appeared in the training corpus, perplexity is inflated optimistically. Deduplication and n-gram overlap checks (see Section 10) partially mitigate this.

*Tokenizer sensitivity.* Different tokenizers assign the same text to different numbers of tokens. Perplexities from models with different tokenizers are _not_ directly comparable. Bits per byte (BPB) normalizes for tokenizer granularity:

$
"BPB" = "CE (nats)" / (ln 2 dot "bytes per token")
$

*No semantic signal.* A model can achieve low perplexity on Wikipedia by memorizing common phrases while still being unable to reason. Perplexity measures fluency, not correctness.

*Benchmark perplexity on WikiText-103:*

#table(
  columns: (auto, auto, auto),
  [*Model*], [*\#Params*], [*PPL (WikiText-103)*],
  [GPT-2 Large],   [774M],  [17.5],
  [LLaMA 2 7B],    [7B],    [5.47],
  [LLaMA 3 8B],    [8B],    [4.81],
  [Mistral 7B v0.1],[7B],   [5.25],
  [LLaMA 3 70B],   [70B],   [3.58],
)

== Knowledge Benchmarks

=== MMLU

*Massive Multitask Language Understanding* (Hendrycks et al., 2021) covers 57 subjects from high school biology to professional law, presented as 4-choice multiple-choice questions. It tests breadth of world knowledge and reasoning.

Example question (abstract algebra):
```
Q: What is the order of the cyclic group Z_12?
A) 6   B) 12   C) 144   D) 24
Correct: B
```

Models are evaluated by selecting the answer token (A/B/C/D) with the highest probability after the prompt. Accuracy is averaged across all 57 tasks.

*Effect of few-shot prompting:* MMLU scores rise substantially from 0-shot to 5-shot, especially for smaller models. The few-shot examples establish the answer format and reduce formatting errors:

#table(
  columns: (auto, auto, auto, auto),
  [*Model*], [*0-shot*], [*5-shot*], [*Delta*],
  [LLaMA 2 7B],      [45.3], [48.6], [+3.3],
  [LLaMA 3 8B],      [65.3], [68.4], [+3.1],
  [Mistral 7B v0.3], [59.7], [62.5], [+2.8],
  [LLaMA 3 70B],     [79.5], [82.0], [+2.5],
  [GPT-4 (2024-04)], [86.1], [86.4], [+0.3],
)

=== GPQA

*Graduate-Level Google-Proof Q&A* (Rein et al., 2023) contains 448 PhD-level questions in biology, chemistry, and physics, verified to be non-googleable. Random baseline is 25%; expert human accuracy is around 65%. Most models below 70B parameters score near random.

=== ARC-Challenge

*AI2 Reasoning Challenge* (Clark et al., 2018) is a 4-choice science exam filtered to questions that retrieval-based systems and word-co-occurrence models fail. It tests genuine reasoning over factual knowledge.

=== Running MMLU with lm-evaluation-harness

EleutherAI's `lm-evaluation-harness` is the standard framework for reproducing published knowledge benchmark scores:

```python
# Command-line usage (runs MMLU 5-shot on LLaMA 3 8B)
# lm_eval --model hf \
#         --model_args pretrained=meta-llama/Meta-Llama-3-8B \
#         --tasks mmlu \
#         --num_fewshot 5 \
#         --batch_size 8 \
#         --output_path results/llama3_8b_mmlu.json

# Python API equivalent:
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

lm = HFLM(
    pretrained="meta-llama/Meta-Llama-3-8B",
    dtype="bfloat16",
    batch_size=8,
)

results = simple_evaluate(
    model=lm,
    tasks=["mmlu"],
    num_fewshot=5,
    limit=None,       # None = full test set
)

print(results["results"]["mmlu"]["acc,none"])
```

The harness handles prompt construction, batching, and aggregation. Always pin the harness version (`pip install lm-eval==0.4.3`) when comparing to published numbers — prompt templates change between versions.

== Reasoning Benchmarks

=== GSM8K

*Grade School Math 8K* (Cobbe et al., 2021): 8 500 grade-school math word problems requiring multi-step arithmetic reasoning. Correct answer requires an exact integer match. Chain-of-thought prompting (few-shot with reasoning steps) raises scores by 15–25 points on most models.

=== MATH

*Competition Mathematics* (Hendrycks et al., 2021): 12 500 problems from AMC, AIME, and similar competitions, across 7 subjects (algebra, geometry, number theory, etc.). Difficulty levels 1–5. State-of-the-art models achieve around 80% with tool use; without tools, 50–60%.

=== HumanEval

*HumanEval* (Chen et al., 2021): 164 handwritten Python programming problems. Each problem provides a function signature, docstring, and a few example inputs/outputs. Correctness is determined by running hidden unit tests. The canonical metric is Pass\@k.

=== MBPP

*Mostly Basic Python Problems* (Austin et al., 2021): 974 short Python programming tasks from a crowdsourced dataset. Easier than HumanEval on average; commonly evaluated at Pass\@1 and Pass\@10.

=== Pass\@k Metric

Running a single generation and checking correctness (Pass\@1) underestimates a model's true capability. Pass\@k measures whether _any_ of $k$ independent samples solves the problem.

*Definition.* Let $n gt.eq k$ total samples be drawn for a problem, of which $c$ are correct. The unbiased estimator of Pass\@k is:

$
"Pass@"k = 1 - binom(n-c, k) / binom(n, k)
= 1 - product_(i=0)^(k-1) (n - c - i) / (n - i)
$

This avoids combinatorial overflow via the product form. The Monte Carlo estimate (sample $n$, count correct $c$, apply formula) is standard.

```python
import numpy as np
from typing import List

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased Pass@k estimator.
    n: total samples drawn per problem
    c: number of correct samples
    k: k in Pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(
        [(n - c - i) / (n - i) for i in range(k)]
    ))

def evaluate_humaneval(
    problems: List[dict],
    generate_fn,   # callable(prompt, n) -> List[str]
    k_vals: List[int] = [1, 10, 100],
    n: int = 200,
) -> dict:
    """
    problems: list of {"prompt": str, "test": str, "entry_point": str}
    generate_fn: model sampler
    """
    results = {k: [] for k in k_vals}

    for problem in problems:
        samples = generate_fn(problem["prompt"], n=n)
        c = sum(run_tests(s, problem) for s in samples)

        for k in k_vals:
            results[k].append(pass_at_k(n, c, k))

    return {f"pass@{k}": np.mean(v) for k, v in results.items()}
```

*HumanEval Pass\@1 scores (temperature=0.2, n=200):*

#table(
  columns: (auto, auto, auto),
  [*Model*], [*Pass\@1*], [*Pass\@10*],
  [LLaMA 2 7B],      [12.8], [25.4],
  [LLaMA 3 8B],      [62.2], [78.5],
  [Mistral 7B v0.3], [37.4], [56.1],
  [LLaMA 3 70B],     [81.7], [92.1],
  [GPT-4 (2024-04)], [90.2], [96.3],
)

== Instruction Following

=== MT-Bench

*MT-Bench* (Zheng et al., 2023) consists of 80 challenging multi-turn conversations across 8 categories: writing, roleplay, reasoning, math, coding, extraction, STEM, and humanities. Each conversation has two turns; the model must maintain context and refine its answer.

Scoring uses GPT-4 as a judge, assigning each response a score from 1 to 10 based on a system prompt with detailed criteria. The final score is the average over all 160 turns (80 turn-1 + 80 turn-2).

=== AlpacaEval

*AlpacaEval* (Li et al., 2023) evaluates instruction-following quality by pairwise comparison against a reference model (originally text-davinci-003, later GPT-4 Turbo). An LLM judge picks the winner of each pair. The metric is *win rate*: fraction of comparisons where the evaluated model is preferred.

AlpacaEval 2.0 adds length-controlled win rate to penalize verbosity (models that win by being longer, not better).

=== IFEval

*Instruction Following Evaluation* (Zhou et al., 2023) uses 541 verifiable instructions with programmatic ground truth: "respond in exactly 3 bullet points", "include the word 'sustainability' at least twice", "do not use commas". No LLM judge is needed — correctness is checked with Python rules.

Two metrics:
- *Prompt-level accuracy:* fraction of prompts where all instructions are followed.
- *Instruction-level accuracy:* fraction of individual instruction constraints satisfied.

IFEval is preferred when reproducibility matters because it eliminates judge variance.

== LLM-as-Judge

=== MT-Bench Protocol

The key insight of MT-Bench is that a strong LLM (GPT-4) can judge response quality more consistently and cheaply than human raters for most capability dimensions. The judge prompt provides:

1. The original question.
2. The model's response.
3. A rubric (what a good answer includes).
4. An explicit instruction to rate 1–10 with a brief justification.

GPT-4 judge scores correlate at around 0.8 with human expert scores on MT-Bench, which is comparable to inter-human agreement.

=== Pairwise Comparison vs. Absolute Scoring

*Absolute scoring* (1–10) is fast but sensitive to anchoring and prompt framing. Small wording changes can shift scores by 1–2 points.

*Pairwise comparison* asks the judge "which response is better, A or B?" and is more robust. It is the basis for Chatbot Arena (see Section 6). The tradeoff is quadratic complexity in the number of models compared.

=== Biases in LLM Judges

*Position bias:* LLM judges prefer whichever response appears first (or second, depending on the judge model). Mitigation: swap A/B and average, or use only cases where both orderings agree.

*Verbosity bias:* judges tend to prefer longer responses even when the shorter one is more accurate. Length-controlled metrics (AlpacaEval 2.0) partially correct for this by regressing win rate on response length.

*Self-enhancement bias:* a model used as a judge tends to prefer outputs from models stylistically similar to itself.

=== Python Example: Calling an LLM Judge

```python
import json
from openai import OpenAI

client = OpenAI()

JUDGE_SYSTEM = """You are a strict, impartial judge evaluating AI assistant responses.
Score the response from 1 (terrible) to 10 (perfect). Be concise.
Output JSON: {"score": <int>, "reason": "<one sentence>"}"""

def llm_judge_score(question: str, response: str) -> dict:
    prompt = (
        f"[Question]\n{question}\n\n"
        f"[Response]\n{response}\n\n"
        "Rate this response."
    )
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(completion.choices[0].message.content)

def llm_judge_pairwise(question: str, resp_a: str, resp_b: str) -> str:
    """Returns 'A', 'B', or 'tie'."""
    prompt = (
        f"[Question]\n{question}\n\n"
        f"[Response A]\n{resp_a}\n\n"
        f"[Response B]\n{resp_b}\n\n"
        "Which response is better? Output JSON: "
        '{"winner": "A" | "B" | "tie", "reason": "<one sentence>"}'
    )
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an impartial judge."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(completion.choices[0].message.content)["winner"]
```

== Chatbot Arena and Elo Ratings

=== Overview

*Chatbot Arena* (Chiang et al., 2024) is a crowdsourced human evaluation platform where volunteers submit a question, receive responses from two anonymous models, and vote for the better response (or declare a tie). As of mid-2024, the Arena had accumulated over one million human votes, making it the largest open evaluation of LLMs.

Because Arena rankings arise from pairwise human preferences rather than automated benchmarks, they are considered the gold standard for measuring real-world conversational quality.

=== Bradley-Terry Model

Arena rankings are derived from the *Bradley-Terry* model, which assumes each model $i$ has a latent strength $s_i$ and the probability that model $i$ beats model $j$ is:

$
P(i > j) = s_i / (s_i + s_j)
$

Strength parameters are estimated by maximum likelihood over all pairwise outcomes. Elo scores are a log-scale reparametrization: $"Elo"_i = 400 log_10(s_i) + "base"$.

=== Elo Update Rule

When model $i$ beats model $j$ in a single comparison, their Elo scores are updated by:

$
"Elo"_i <- "Elo"_i + K (1 - E_(i j))
$
$
"Elo"_j <- "Elo"_j + K (0 - E_(j i))
$

where $K$ is the update factor (typically 32) and the expected score is:

$
E_(i j) = 1 / (1 + 10^(("Elo"_j - "Elo"_i) / 400))
$

```python
def elo_update(
    elo_i: float,
    elo_j: float,
    outcome: float,   # 1.0 = i wins, 0.5 = tie, 0.0 = j wins
    K: float = 32.0,
) -> tuple[float, float]:
    """Returns updated (elo_i, elo_j)."""
    E_ij = 1.0 / (1.0 + 10.0 ** ((elo_j - elo_i) / 400.0))
    E_ji = 1.0 - E_ij
    new_i = elo_i + K * (outcome - E_ij)
    new_j = elo_j + K * ((1.0 - outcome) - E_ji)
    return new_i, new_j
```

In practice, Arena uses maximum-likelihood estimation over the entire battle history rather than incremental updates, which gives more stable rankings. Bootstrap resampling provides confidence intervals.

=== Chatbot Arena Leaderboard (snapshot as of 2025-03; live leaderboard at lmarena.ai)

#table(
  columns: (auto, auto, auto),
  [*Model*], [*Arena Elo*], [*\# Battles*],
  [GPT-4o (2024-11)],     [1360], [50 000+],
  [Gemini 1.5 Pro],       [1298], [40 000+],
  [Claude 3.5 Sonnet],    [1295], [45 000+],
  [LLaMA 3.1 405B],       [1266], [30 000+],
  [Mistral Large 2],      [1251], [20 000+],
  [LLaMA 3.1 70B],        [1228], [25 000+],
  [LLaMA 3 8B],           [1152], [20 000+],
  [Mistral 7B v0.3],      [1072], [15 000+],
)

== Calibration

=== What Calibration Means

A model is *well-calibrated* if its stated confidence matches empirical accuracy. If a model says "I am 80% confident" across 100 questions, it should be correct on about 80 of them. Overconfidence is the most common failure: a model says 95% but is only correct 70% of the time.

=== Expected Calibration Error

*Expected Calibration Error (ECE)* partitions predictions into $M$ equal-width confidence bins and measures the weighted average gap between confidence and accuracy:

$
"ECE" = sum_(m=1)^M |B_m| / N |"acc"(B_m) - "conf"(B_m)|
$

where $B_m$ is the set of samples in bin $m$, $"acc"(B_m)$ is the fraction correct, and $"conf"(B_m)$ is the mean predicted probability.

```python
import torch
import numpy as np

def compute_ece(
    probs: torch.Tensor,    # [N] predicted probabilities for the correct class
    labels: torch.Tensor,   # [N] 1 if correct, 0 if wrong
    n_bins: int = 15,
) -> float:
    """
    Computes Expected Calibration Error.
    probs:  predicted confidence in [0, 1]
    labels: binary correctness
    """
    probs  = probs.cpu().numpy()
    labels = labels.cpu().numpy().astype(float)
    N      = len(probs)

    bins        = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / N) * abs(acc - conf)

    return float(ece)
```

A *reliability diagram* plots $"acc"(B_m)$ vs. $"conf"(B_m)$ for each bin. A perfectly calibrated model falls on the diagonal. Points above the diagonal indicate underconfidence; points below indicate overconfidence.

=== Temperature Scaling

*Temperature scaling* (Guo et al., 2017) is the simplest post-hoc calibration method. A single scalar $T$ is learned on a validation set to minimize negative log-likelihood of calibrated probabilities:

$
hat(p)_i = "softmax"(z_i / T)
$

where $z_i$ is the logit vector. $T > 1$ softens the distribution (reduces overconfidence); $T < 1$ sharpens it.

```python
import torch
import torch.optim as optim
import torch.nn.functional as F

def calibrate_temperature(
    logits: torch.Tensor,   # [N, C] raw logits on validation set
    labels: torch.Tensor,   # [N]    ground truth class indices
    max_iter: int = 1000,
) -> float:
    """Returns optimal temperature T."""
    T = torch.nn.Parameter(torch.ones(1, device=logits.device))
    optimizer = optim.LBFGS([T], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return T.item()
```

Temperature scaling does not change the model's rankings (argmax is preserved); it only adjusts the probability values. For multiple-choice benchmarks, this means accuracy is unaffected.

== Hallucination Metrics

=== TruthfulQA

*TruthfulQA* (Lin et al., 2022) contains 817 questions designed to elicit false answers that humans tend to believe (common misconceptions, urban legends, folk wisdom). Examples: "Is it legal to marry your first cousin in the US?" or "What happens if you swallow a seed?"

Two evaluation modes:
- *MC* (multiple choice): standard 4-choice accuracy, automated.
- *Generative*: free-form answers judged by a fine-tuned GPT-3 classifier that predicts truthfulness and informativeness.

Larger models often score _worse_ on TruthfulQA because they more confidently reproduce internet myths. RLHF-tuned models score substantially better.

=== FactScore

*FactScore* (Min et al., 2023) measures entity-level factuality in long-form generation. The pipeline:

1. *Generate* a biography or factual passage about a subject.
2. *Decompose* the response into atomic facts using an LLM (e.g., "Barack Obama was born in Hawaii").
3. *Verify* each fact against a knowledge source (Wikipedia retrieval + LLM judgment).
4. *Score* = fraction of atomic facts that are supported.

```python
# Conceptual FactScore pipeline
def factscore_pipeline(
    subject: str,
    model_response: str,
    retriever,       # search engine over Wikipedia
    judge_llm,       # LLM that verifies facts given evidence
) -> float:

    # Step 1: decompose into atomic claims
    atomic_facts = decompose_to_atoms(model_response, judge_llm)

    verified = []
    for fact in atomic_facts:
        # Step 2: retrieve relevant passages
        passages = retriever.search(subject + " " + fact, top_k=5)
        context  = "\n".join(p["text"] for p in passages)

        # Step 3: verify with judge
        verdict = judge_llm.verify(
            claim=fact,
            evidence=context,
        )  # returns True/False
        verified.append(verdict)

    return sum(verified) / len(verified) if verified else 0.0
```

FactScore for LLaMA 2 65B is around 55%; GPT-4 reaches around 73%. The metric is most useful for biography and knowledge-intensive generation tasks.

== Safety Evaluations

=== Red-Teaming

Red-teaming involves human adversaries or automated systems crafting prompts designed to elicit harmful, biased, or policy-violating outputs. Goals include finding:
- *Jailbreaks:* prompts that bypass safety training.
- *Bias:* systematic differences in outputs across demographic groups.
- *Harmful information:* instructions for dangerous activities.

Anthropic, OpenAI, and Google conduct pre-deployment red-team exercises. Results are typically not published in detail.

=== StrongREJECT

*StrongREJECT* (Souly et al., 2024) is a benchmark of 313 harmful prompts across 6 categories (cybercrime, CBRN, adult content, etc.) with automated scoring. The evaluator checks both whether the model refused _and_ whether a refusal was genuine (not a partial compliance disguised as a refusal).

=== WildGuard

*WildGuard* (Han et al., 2024) is a safety classifier trained on 92 000 human-annotated examples covering both harmful prompts and model responses. It predicts:
- Whether the prompt is harmful.
- Whether the response complies with a harmful request.
- Whether the response is unnecessarily refusing a benign request.

WildGuard is used as an automated judge in safety benchmarks, replacing expensive human annotation.

=== Refusal Rate vs. Helpfulness Tradeoff

Safety training creates a fundamental tension: a model that refuses more requests is safer but less helpful. The *over-refusal rate* measures how often the model refuses benign prompts.

#table(
  columns: (auto, auto, auto, auto),
  [*Model*], [*Harmful refusal rate*], [*Benign refusal rate*], [*Balance*],
  [LLaMA 2 Chat 7B],  [94%], [28%], [poor],
  [LLaMA 3 Instruct 8B], [91%], [8%],  [good],
  [Mistral 7B Instruct], [82%], [4%],  [good],
  [GPT-4o],              [97%], [3%],  [excellent],
)

A benign refusal rate above 10% is considered a significant helpfulness regression. WildGuard and IFEval together provide a joint measure of safety and helpfulness.

== Contamination Detection

=== The Problem

If benchmark questions appeared in the model's training data, reported scores are inflated. Contamination is widespread: Common Crawl snapshots contain Stack Overflow (HumanEval answers), Wikipedia (TruthfulQA context), and exam prep websites (MMLU questions).

=== N-gram Overlap

The simplest approach: compute the fraction of test questions where a long n-gram (13-gram or longer) appears in the training corpus. If greater than 20% of a benchmark's test set has 13-gram overlap, scores should be treated with caution.

```python
from collections import Counter

def ngram_overlap(text: str, corpus_ngrams: set, n: int = 13) -> float:
    """
    Returns fraction of n-grams in text that appear in corpus_ngrams.
    corpus_ngrams: pre-built set of all n-grams from training data.
    """
    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0
    text_grams = set(
        tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)
    )
    overlap = text_grams & corpus_ngrams
    return len(overlap) / len(text_grams)
```

=== Min-k% Probability Method

*Min-k% Prob* (Shi et al., 2024 — _Detecting Pretraining Data from Large Language Models_) does not require access to training data. Instead, it exploits the observation that memorized text has uniformly high token probabilities, while novel text has some low-probability tokens. The score is the mean log-probability of the bottom $k$% of tokens in the sequence:

```python
def min_k_prob(
    model,
    tokenizer,
    text: str,
    k_frac: float = 0.2,
) -> float:
    """
    Returns mean log-prob of the bottom k% tokens.
    Higher (less negative) -> likely memorized.
    """
    tokens = tokenizer(text, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        logits = model(tokens).logits  # [1, N, V]

    log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
    token_log_probs = log_probs.gather(
        1, tokens[0, 1:].unsqueeze(1)
    ).squeeze(1)  # [N-1]

    k = max(1, int(k_frac * len(token_log_probs)))
    bottom_k, _ = token_log_probs.topk(k, largest=False)
    return bottom_k.mean().item()
```

A threshold is fit on a held-out set of known-memorized vs. non-memorized examples. Min-k% achieves around 60–70% AUC for detecting benchmark contamination without needing training data access.

== Practical Evaluation Setup

=== lm-evaluation-harness CLI

The `lm-evaluation-harness` is the standard tool for reproducing knowledge and reasoning benchmark scores. Key flags:

```
# Install
pip install lm-eval==0.4.3

# MMLU 5-shot, full test set
lm_eval \
  --model hf \
  --model_args pretrained=meta-llama/Meta-Llama-3-8B,dtype=bfloat16 \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path results/

# Multiple tasks in one run
lm_eval \
  --model hf \
  --model_args pretrained=meta-llama/Meta-Llama-3-8B,dtype=bfloat16 \
  --tasks mmlu,arc_challenge,hellaswag,truthfulqa_mc2 \
  --num_fewshot 5 \
  --batch_size auto

# OpenAI API models
lm_eval \
  --model openai-chat-completions \
  --model_args model=gpt-4o \
  --tasks mmlu \
  --num_fewshot 5
```

=== Reproducing Published Numbers

Common reasons published numbers cannot be reproduced:

#table(
  columns: (auto, auto),
  [*Cause*], [*Fix*],
  [Different harness version],       [Pin `lm-eval==` to paper's version],
  [Different prompt template],       [Check task YAML in harness source],
  [Chat vs. base model],             [Use base model for knowledge benchmarks],
  [System prompt differences],       [Match exactly; avoid default chat templates],
  [Greedy vs. sampling],             [Knowledge benchmarks: greedy (temp=0)],
  [Tokenizer mismatch],              [Use model's own tokenizer, not a generic one],
  [Batch size rounding effects],     [Use batch\_size=1 for exact reproduction],
)

=== Common Mistakes

*Using chat models on base benchmarks.* MMLU and ARC are designed for base (pretrained) models. Instruction-tuned models inject system prompts that shift score distributions. Always specify which model variant was evaluated.

*Reporting only best shots.* MMLU scores at 0-shot, 5-shot, and 25-shot differ by up to 5 points. Always state the shot count.

*Not reporting confidence intervals.* For small test sets (HumanEval: 164 examples), a 1% accuracy difference is not statistically significant. Report 95% confidence intervals using bootstrap resampling.

*Mixing evaluation frameworks.* Do not compare numbers from different frameworks. `lm-evaluation-harness`, `OpenAI evals`, and `BIG-bench` use different prompt formats, normalization, and tie-breaking — scores are not directly comparable.

*Ignoring generation hyperparameters.* Pass\@1 on HumanEval at temperature 0.0 and temperature 0.8 can differ by 10 points. Always report temperature, top-p, and max tokens.

=== Complete Model Score Comparison

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  [*Model*], [*MMLU 5-shot*], [*GSM8K*], [*HumanEval P\@1*], [*GPQA*], [*TruthfulQA*],
  [LLaMA 2 7B],         [45.3], [14.6], [12.8], [28.1], [38.8],
  [LLaMA 3 8B],         [68.4], [79.6], [62.2], [34.2], [43.9],
  [LLaMA 3 70B],        [82.0], [93.0], [81.7], [46.7], [52.8],
  [Mistral 7B v0.3],    [62.5], [52.1], [37.4], [29.9], [42.2],
  [Mistral Large 2],    [84.0], [91.2], [92.0], [59.2], [72.3],
  [GPT-4 (2024-04)],    [86.4], [95.3], [90.2], [53.6], [59.0],
)

_All scores are approximate figures from published reports and leaderboards. Exact numbers vary by evaluation setup._

== References

- Hendrycks, D. et al. (2021). _Measuring Massive Multitask Language Understanding_. ICLR 2021.
- Chen, M. et al. (2021). _Evaluating Large Language Models Trained on Code_. arXiv:2107.03374.
- Cobbe, K. et al. (2021). _Training Verifiers to Solve Math Word Problems_. arXiv:2110.14168.
- Lin, S. et al. (2022). _TruthfulQA: Measuring How Models Mimic Human Falsehoods_. ACL 2022.
- Guo, C. et al. (2017). _On Calibration of Modern Neural Networks_. ICML 2017.
- Zheng, L. et al. (2023). _Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena_. NeurIPS 2023.
- Chiang, W. et al. (2024). _Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference_. arXiv:2403.04132.
- Min, S. et al. (2023). _FActScoring: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation_. EMNLP 2023.
- Rein, D. et al. (2023). _GPQA: A Graduate-Level Google-Proof Q&A Benchmark_. arXiv:2311.12022.
- Shi, W. et al. (2024). _Detecting Pretraining Data from Large Language Models_. ICLR 2024.
- Souly, A. et al. (2024). _A StrongREJECT for Empty Jailbreaks_. arXiv:2402.10260.
- Zhou, J. et al. (2023). _Instruction-Following Evaluation for Large Language Models_. arXiv:2311.07911.
- EleutherAI. _lm-evaluation-harness_. https://github.com/EleutherAI/lm-evaluation-harness

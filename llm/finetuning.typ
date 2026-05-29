= Fine-tuning

Pre-trained LLMs encode vast world knowledge but respond poorly to instructions and specific task formats out of the box. Fine-tuning adapts the model's behavior — teaching it to follow instructions, hold multi-turn conversations, or specialize in a domain — without relearning language from scratch. This chapter covers the full spectrum from expensive full fine-tuning to the parameter-efficient methods (LoRA, QLoRA) that dominate production work, plus the data formats, alignment algorithms, and practical recipes needed to run a real training job.

*See also:* _Pretraining_ (how base weights are obtained), _ML Workload Optimization on GPUs (GPU Architecture volume)_ (GEMM kernels, mixed-precision training), _GPU Memory Hierarchy (GPU Architecture volume)_ (HBM bandwidth and capacity constraints).

*Code note:* PyTorch is used for all runnable examples. C++ (libtorch) is shown for inference-time weight merging. bitsandbytes and HuggingFace `transformers` / `trl` are referenced as the standard production stack.

== Full Fine-tuning vs PEFT

Full fine-tuning updates every parameter. For a 7B model at BF16 that is 14 GB of weights; the Adam optimizer adds two more momentum buffers (another 28 GB), and gradients add 14 GB — over 56 GB just for optimizer state before activations. This pushes even 7B models off a single 80 GB A100 without gradient checkpointing. Parameter-Efficient Fine-Tuning (PEFT) methods freeze the base model and train only a small set of additional parameters.

=== Memory Comparison Table

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Method*], [*Trainable params*], [*7B GPU RAM*], [*70B GPU RAM*], [*Notes*],
  [Full FT (BF16)],    [100%],        [$>=$ 56 GB],  [$>=$ 560 GB],  [Weights + grad + Adam states],
  [LoRA (r=16)],       [~0.1–0.5%],   [~18 GB],         [~160 GB],         [Base frozen in BF16; LoRA in BF16],
  [QLoRA (r=16, NF4)], [~0.1–0.5%],   [~6–8 GB],        [~48 GB (2×24 GB)],[Base in 4-bit NF4; LoRA in BF16],
)

_Trainable params_ for LoRA at rank 16 targeting `q_proj` and `v_proj` in all layers of LLaMA 3 8B: approximately 20 M out of 8 000 M total (0.25%).

The GPU RAM column is the minimum for training with per-token batch size 1, gradient checkpointing enabled, and no Flash Attention activation recomputation overhead beyond checkpointing. Real jobs need headroom; multiply by ~1.3.

== LoRA: Low-Rank Adaptation

Hu et al. (2022) observed that the weight updates $Delta W$ during fine-tuning have low intrinsic rank. Instead of learning $Delta W in RR^(d times k)$ directly, LoRA parameterizes it as

$ W' = W + (alpha / r) dot B A $

where $A in RR^(r times k)$, $B in RR^(d times r)$, rank $r lt.double min(d, k)$, and $alpha$ is a scaling hyperparameter. $A$ is initialized with random Gaussian, $B$ with zeros, so $Delta W = 0$ at initialization and training starts from the original model output.

During forward pass the computation is:

$ h = W x + (alpha / r) dot B (A x) $

$A x$ costs $r dot k$ multiplications, $B(A x)$ costs $d dot r$ — a total of $r(d + k)$ versus $d dot k$ for the full update. For $d = k = 4096$ and $r = 16$: full is 16.8 M, LoRA is 131 K — a 128x reduction in the update path.

=== Rank Selection

#table(
  columns: (auto, auto, auto),
  [*r*], [*Typical use*], [*Trainable params (LLaMA 3 8B, all linear)*],
  [4],   [Light instruction tuning],      [~10 M],
  [16],  [Standard chat / task adapt],    [~40 M],
  [64],  [Domain-heavy adaptation],       [~160 M],
  [128], [Near full-FT quality],          [~320 M],
)

Higher rank recovers more expressive capacity at the cost of memory and compute. $r = 16$ with $alpha = 32$ (scaling factor 2) is the most common starting point.

=== Which Modules to Target

Targeting only `q_proj` and `v_proj` (attention query and value projections) was the original LoRA paper's recommendation. Later work showed that including all linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, and both MLP layers `gate_proj`, `up_proj`, `down_proj`) gives better results at the same rank.

#table(
  columns: (auto, auto, auto),
  [*Target set*], [*Params (r=16, 7B)*], [*Recommendation*],
  [`q_proj`, `v_proj`],              [~20 M],  [Baseline; smallest footprint],
  [All attention projections],        [~40 M],  [Better for instruction following],
  [All linear (attn + MLP)],          [~80 M],  [Best quality; use when GPU allows],
  [Embedding + lm\_head (+ linear)],  [~100 M], [Needed for new vocabulary tokens],
)

=== PyTorch LoRALinear Implementation

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a LoRA side path.

    The original weight is frozen. Only lora_A and lora_B are trained.

    Args:
        linear:  existing nn.Linear to wrap (weight is borrowed, not copied)
        r:       LoRA rank
        alpha:   LoRA scaling numerator; effective scale = alpha / r
        dropout: dropout applied to the input of the LoRA path
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.r = r
        self.scale = alpha / r

        # Borrow (do not copy) the frozen base weight.
        self.weight = linear.weight          # shape [out, in]
        self.bias   = linear.bias            # shape [out] or None
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        in_features  = linear.in_features
        out_features = linear.out_features

        # LoRA matrices: A ~ N(0, 1/r), B = 0 at init.
        self.lora_A = nn.Parameter(
            torch.empty(r, in_features).normal_(std=1.0 / math.sqrt(r))
        )
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path (no gradient through frozen weight).
        base_out = F.linear(x, self.weight, self.bias)

        # LoRA side path: x → dropout → A → B, then scale.
        lora_out = F.linear(
            F.linear(self.lora_dropout(x), self.lora_A),  # [B, L, r]
            self.lora_B,                                    # [B, L, out]
        )
        return base_out + self.scale * lora_out

    def merge_weights(self) -> nn.Linear:
        """Return a plain nn.Linear with LoRA merged into W for inference."""
        merged = nn.Linear(
            self.weight.shape[1], self.weight.shape[0],
            bias=self.bias is not None,
        )
        delta = self.scale * (self.lora_B @ self.lora_A)  # [out, in]
        merged.weight = nn.Parameter((self.weight + delta).contiguous())
        if self.bias is not None:
            merged.bias = nn.Parameter(self.bias.clone())
        return merged


def inject_lora(model: nn.Module, r: int = 16, alpha: float = 32.0,
                target_modules: tuple = ("q_proj", "v_proj")) -> nn.Module:
    """Replace named Linear layers in-place with LoRALinear."""
    for name, module in list(model.named_modules()):
        parts = name.rsplit(".", 1)
        parent_name, child_name = (parts[0], parts[1]) if len(parts) == 2 else ("", parts[0])
        if child_name in target_modules and isinstance(module, nn.Linear):
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha))
    return model
```

Usage with any HuggingFace model:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                              torch_dtype=torch.bfloat16)
model = inject_lora(model, r=16, alpha=32,
                    target_modules=("q_proj", "k_proj", "v_proj",
                                    "o_proj", "gate_proj", "up_proj", "down_proj"))

# Confirm only LoRA params have gradients.
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.0f}M")
# Trainable: 83.9M / 8030.3M
```

=== C++ Weight Merge for Inference

After training, the LoRA adapters are merged back into the base weight so inference sees a plain linear layer with zero overhead. The libtorch equivalent of `merge_weights()`:

```cpp
#include <torch/torch.h>

// Merge LoRA into a base weight tensor in-place.
// base_weight: [out_features, in_features]  (mutable)
// lora_B:      [out_features, r]
// lora_A:      [r, in_features]
// scale:       alpha / r
void merge_lora_weight(
    torch::Tensor& base_weight,
    const torch::Tensor& lora_B,
    const torch::Tensor& lora_A,
    float scale)
{
    // delta = scale * B @ A  →  [out, in]
    torch::Tensor delta = scale * torch::mm(lora_B, lora_A);
    base_weight.add_(delta);   // in-place: no extra allocation
}

// Example: load adapter safetensors and merge into llama.cpp gguf weights.
int main() {
    auto base = torch::load("llama3_q_proj.pt");          // [4096, 4096] BF16
    auto A    = torch::load("lora_A_q_proj.pt");          // [16, 4096]
    auto B    = torch::load("lora_B_q_proj.pt");          // [4096, 16]

    float scale = 32.0f / 16.0f;   // alpha=32, r=16
    merge_lora_weight(base, B, A, scale);
    torch::save(base, "llama3_q_proj_merged.pt");
    return 0;
}
```

The merged weight is identical in shape to the original; the serving engine (llama.cpp, vLLM, TRT-LLM) loads it without modification.

== QLoRA: 4-bit Base + BF16 LoRA

Dettmers et al. (2023) combined three innovations to make fine-tuning 65B (and larger) models accessible on consumer hardware:

1. *NF4 quantization* — the base model is stored in 4-bit NormalFloat4, a data type that is information-theoretically optimal for weights drawn from a normal distribution. Each NF4 value maps to a quantile of the standard normal.
2. *Double quantization* — the per-block quantization constants are themselves quantized (from FP32 to FP8), saving ~0.4 bits per parameter on top of NF4.
3. *Paged optimizers* — NVIDIA unified memory is used to page optimizer states to CPU RAM when GPU memory is full, preventing OOM during long sequences.

=== Memory Calculation: 65B on 2 × 48 GB

$
"Base weights (NF4):"       & quad 65 times 10^9 times 0.5 "B" = 32.5 "GB" \
"LoRA params (BF16, r=64):" & quad ~300 times 10^6 times 2 "B" = 0.6 "GB" \
"LoRA grads:"               & quad 0.6 "GB" \
"Adam states (LoRA only):"  & quad 0.6 times 2 = 1.2 "GB" \
"Activations (2K ctx, ckpt):",&quad ~3 "GB per GPU" \
"Total:"                    & quad ~38 "GB per GPU" quad ("fits" 2 times 48 "GB")
$

_Headroom note:_ peak memory can exceed the static estimate when paged optimizer states spill to CPU under fragmentation, when activation checkpointing is partially disabled, or when sequence length exceeds 2K. Budget +15–20% headroom per GPU to avoid OOM on long-context batches.

Without QLoRA the same job at BF16 would require $65 times 2 + 65 times 2 times 3 approx 520$ GB of GPU memory — eight 80 GB A100s minimum.

=== bitsandbytes QLoRA Setup

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # double quantization
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-70B",
    quantization_config=bnb_config,
    device_map="auto",                   # spread across available GPUs
    torch_dtype=torch.bfloat16,
)

# PEFT wraps the 4-bit frozen base with BF16 LoRA adapters.
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 327,155,712 || all params: 70,553,706,496 || trainable%: 0.46%
```

The paged Adam optimizer is enabled by passing `optim="paged_adamw_32bit"` to `transformers.TrainingArguments`.

=== Forward Pass Through a Quantized Layer

During the forward pass, bitsandbytes *dequantizes* each NF4 block to BF16 on-the-fly in a custom CUDA kernel, performs the GEMM in BF16, then discards the dequantized copy. The LoRA path runs entirely in BF16 and its result is added before returning. No gradient flows through the base weight (frozen), so only the LoRA grad tensors must be stored.

== Instruction Tuning

Instruction tuning teaches the model to follow a structured prompt format. The key insight is _loss masking_: we only compute cross-entropy loss on the _assistant_ tokens, not on the system prompt or user turn. This prevents the model from optimizing its own input.

=== Data Format

All mainstream instruction datasets use a three-role structure:

```
system:    "You are a helpful assistant."
user:      "Explain gradient descent in one paragraph."
assistant: "Gradient descent is an iterative optimization algorithm ..."
```

*FLAN* (Wei et al., 2022): 1 800+ tasks reformulated as natural-language instructions. Templates are hand-written per task. No multi-turn conversation.

*Alpaca* (Taori et al., 2023): 52 K instruction–response pairs generated from `text-davinci-003` using self-instruct. Single-turn only; widely used despite quality concerns.

*ShareGPT*: real ChatGPT conversations scraped from sharegpt.com. Multi-turn, diverse, higher quality than self-instruct datasets. Filtered versions (ShareGPT4 etc.) use GPT-4 responses.

=== Loss Masking

Given a tokenized conversation of length $L$, let $M_i = 1$ if token $i$ is part of an assistant turn, else $M_i = 0$. The masked loss is:

$ cal(L) = - (1 / (sum_i M_i)) sum_(i=1)^(L) M_i log p_theta (t_i | t_{< i}) $

Only assistant tokens contribute to the gradient. This is critical: without masking the model memorizes the user prompts and degrades on unseen instruction styles.

=== PyTorch DataCollator with Loss Masking

```python
from dataclasses import dataclass
from typing import List, Dict
import torch
from transformers import PreTrainedTokenizer


@dataclass
class InstructionCollator:
    """Collates tokenized conversations with assistant-only loss mask.

    Each example in the batch is expected to be a dict with keys:
        input_ids:  List[int]  — full conversation token ids
        labels:     List[int]  — same length; -100 for non-assistant tokens
    """
    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    pad_to_multiple_of: int = 8

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids_list = [ex["input_ids"][:self.max_length] for ex in examples]
        labels_list    = [ex["labels"][:self.max_length]    for ex in examples]

        # Pad to the longest sequence in the batch (right-pad).
        max_len = max(len(x) for x in input_ids_list)
        # Round up to multiple for efficient tensor cores.
        pad_to = ((max_len + self.pad_to_multiple_of - 1)
                  // self.pad_to_multiple_of * self.pad_to_multiple_of)

        pad_id = self.tokenizer.pad_token_id or 0

        input_ids = torch.full((len(examples), pad_to), pad_id, dtype=torch.long)
        labels    = torch.full((len(examples), pad_to), -100,   dtype=torch.long)
        attn_mask = torch.zeros((len(examples), pad_to),        dtype=torch.long)

        for i, (ids, labs) in enumerate(zip(input_ids_list, labels_list)):
            L = len(ids)
            input_ids[i, :L] = torch.tensor(ids,  dtype=torch.long)
            labels[i,    :L] = torch.tensor(labs, dtype=torch.long)
            attn_mask[i, :L] = 1

        return {
            "input_ids":      input_ids,
            "labels":         labels,
            "attention_mask": attn_mask,
        }


def build_labels(input_ids: List[int],
                 assistant_ranges: List[tuple]) -> List[int]:
    """Set labels to -100 everywhere except assistant token ranges.

    assistant_ranges: list of (start, end) index pairs (exclusive end).
    """
    labels = [-100] * len(input_ids)
    for start, end in assistant_ranges:
        labels[start:end] = input_ids[start:end]
    return labels
```

The `labels` tensor is passed directly to HuggingFace `CausalLM.forward()`; HuggingFace shifts labels by one internally and masks `-100` positions in the cross-entropy loss.

== Chat Templates

Chat templates encode conversation structure into the raw token stream that the model sees. Every model family uses different special tokens and delimiters; using the wrong template at inference time silently degrades quality.

=== LLaMA 3 Chat Template

LLaMA 3 uses the following structure (shown here with literal special tokens):

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

What is the capital of France?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Paris.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Why is it called that?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

```

The model generates until it produces `<|eot_id|>` (end-of-turn id). The special tokens `<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, and `<|eot_id|>` are reserved vocabulary entries with fixed ids in the 128 K LLaMA 3 tokenizer.

=== Applying Templates with the Tokenizer

HuggingFace tokenizers store the template as a Jinja2 string in `tokenizer.chat_template`. The `apply_chat_template` method handles all escaping and special-token insertion:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is LoRA?"},
    {"role": "assistant", "content": "LoRA is a parameter-efficient fine-tuning method..."},
    {"role": "user",      "content": "How does it save memory?"},
]

# tokenize=False to inspect the raw string first
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)

# For training, get token ids and compute the assistant mask.
encoding = tok.apply_chat_template(messages, tokenize=True,
                                    return_dict=True,
                                    return_assistant_tokens_mask=True)
input_ids = encoding["input_ids"]
# assistant_masks[i] == 1 where token i is part of an assistant turn
assistant_mask = encoding["assistant_tokens_mask"]
```

`add_generation_prompt=True` appends the opening `<|start_header_id|>assistant<|end_header_id|>\n\n` without closing it, prompting the model to generate the next assistant turn.

=== Common Template Special Tokens

#table(
  columns: (auto, auto, auto),
  [*Model family*], [*Turn delimiter*], [*End-of-turn token*],
  [LLaMA 3],   [`<|start_header_id|>` ... `<|end_header_id|>`], [`<|eot_id|>`],
  [LLaMA 2],   [`[INST]` ... `[/INST]`],                         [`</s>`],
  [Mistral v1], [`[INST]` ... `[/INST]`],                        [`</s>`],
  [Gemma],     [`<start_of_turn>user` / `model`],                [`<end_of_turn>`],
  [Phi-3],     [`<|user|>` / `<|assistant|>`],                   [`<|end|>`],
  [Qwen2],     [`<|im_start|>system/user/assistant`],            [`<|im_end|>`],
)

== DPO: Direct Preference Optimization

Reinforcement Learning from Human Feedback (RLHF) requires training a separate reward model and running PPO, which is complex, unstable, and memory-intensive. DPO (Rafailov et al., 2023) shows that the RLHF objective has a closed-form optimal policy that can be expressed as a supervised loss over preference pairs — no explicit reward model needed.

=== Derivation from the RLHF Objective

The standard RLHF objective maximizes expected reward subject to a KL constraint against a reference policy $pi_"ref"$ (the SFT model):

$ max_(pi_theta) EE_(x, y ~ pi_theta) [r(x, y)] - beta dot "KL"(pi_theta(dot|x) || pi_"ref"(dot|x)) $

The closed-form optimal policy under this objective is:

$ pi^*(y|x) = (1/Z(x)) pi_"ref"(y|x) exp(r(x,y) / beta) $

where $Z(x) = sum_y pi_"ref"(y|x) exp(r(x,y)/beta)$ is the partition function. Inverting to express the reward in terms of the optimal policy:

$ r(x,y) = beta log (pi^*(y|x) / pi_"ref"(y|x)) + beta log Z(x) $

The Bradley-Terry preference model gives the probability that response $y_w$ (the _winner_) is preferred over $y_l$ (the _loser_):

$ p(y_w succ y_l | x) = sigma (r(x, y_w) - r(x, y_l)) $

Substituting the reward expression and noting that $Z(x)$ cancels:

$ p(y_w succ y_l | x) = sigma lr(( beta log (pi_theta(y_w|x) / pi_"ref"(y_w|x)) - beta log (pi_theta(y_l|x) / pi_"ref"(y_l|x)) |)) $

The DPO loss is the negative log-likelihood of this preference:

$ cal(L)_"DPO"(pi_theta; pi_"ref") = -EE_((x, y_w, y_l) ~ cal(D)) lr([ log sigma lr(( beta log (pi_theta(y_w|x)) / (pi_"ref"(y_w|x)) - beta log (pi_theta(y_l|x)) / (pi_"ref"(y_l|x)) |)) ]) $

This is the entire loss: two forward passes (on winner and loser), no reward model, no sampling loop.

=== Why DPO Avoids an Explicit Reward Model

The key insight is that the optimal reward $r^*(x,y)$ under the RLHF objective is a deterministic function of the optimal policy $pi^*$ and the reference policy $pi_"ref"$. DPO directly optimizes $pi_theta$ to be this optimal policy, bypassing the intermediate reward model entirely. The reference policy $pi_"ref"$ (kept frozen) serves as a regularizer that prevents the trained policy from degenerating to out-of-distribution text.

$beta$ controls the strength of the KL constraint: small $beta$ allows large deviation from $pi_"ref"$, large $beta$ keeps the model close to SFT. Practical range: $beta in [0.1, 0.5]$. Values $beta < 0.05$ risk mode collapse (the policy drifts far from the reference and produces degenerate, overconfident outputs); values $beta > 1.0$ underfit the preference signal because the KL term dominates the gradient.

*Common failure mode.* If the reference model $pi_"ref"$ is weak or undertrained (e.g., SFT did not converge, or the reference is a base model rather than an SFT model), DPO can train an _overconfident_ policy whose probability mass diverges sharply from the reference distribution: the implicit reward $beta log(pi_theta / pi_"ref")$ grows without a useful regularization anchor. Always validate with held-out preference accuracy (the `accuracy` diagnostic returned in the training step below); if it climbs to 100% on the train set while held-out accuracy regresses, the policy is overfitting the preference noise.

=== PyTorch DPO Training Step

```python
import torch
import torch.nn.functional as F


def sequence_logprob(logits: torch.Tensor,
                     input_ids: torch.Tensor,
                     labels_mask: torch.Tensor) -> torch.Tensor:
    """Sum of log-probs over non-masked response tokens.

    Args:
        logits:      [B, L, V]  — raw model output (before softmax)
        input_ids:   [B, L]     — token ids (labels shifted by 1 from input)
        labels_mask: [B, L]     — 1.0 for response tokens, 0.0 for prompt
    Returns:
        [B] sum of log-probs over response tokens
    """
    # Shift: logits[i] predicts input_ids[i+1]
    shift_logits = logits[:, :-1, :].contiguous()   # [B, L-1, V]
    shift_ids    = input_ids[:, 1:].contiguous()     # [B, L-1]
    shift_mask   = labels_mask[:, 1:].contiguous()   # [B, L-1]

    log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, L-1, V]
    token_lp  = log_probs.gather(
        dim=-1, index=shift_ids.unsqueeze(-1)
    ).squeeze(-1)                                     # [B, L-1]

    return (token_lp * shift_mask).sum(dim=-1)        # [B]


def dpo_loss(policy_model: torch.nn.Module,
             ref_model:    torch.nn.Module,
             batch:        dict,
             beta:         float = 0.1) -> torch.Tensor:
    """One DPO training step.

    batch keys:
        winner_input_ids, winner_attention_mask, winner_labels_mask
        loser_input_ids,  loser_attention_mask,  loser_labels_mask
    """
    def get_logprob(model, ids, mask, labels_mask):
        with torch.no_grad() if model is ref_model else torch.enable_grad():
            out = model(input_ids=ids, attention_mask=mask)
        return sequence_logprob(out.logits, ids, labels_mask)

    pi_logprob_w = get_logprob(policy_model,
                                batch["winner_input_ids"],
                                batch["winner_attention_mask"],
                                batch["winner_labels_mask"])
    pi_logprob_l = get_logprob(policy_model,
                                batch["loser_input_ids"],
                                batch["loser_attention_mask"],
                                batch["loser_labels_mask"])

    with torch.no_grad():
        ref_logprob_w = get_logprob(ref_model,
                                     batch["winner_input_ids"],
                                     batch["winner_attention_mask"],
                                     batch["winner_labels_mask"])
        ref_logprob_l = get_logprob(ref_model,
                                     batch["loser_input_ids"],
                                     batch["loser_attention_mask"],
                                     batch["loser_labels_mask"])

    # Implicit reward difference
    reward_diff = beta * (
        (pi_logprob_w - ref_logprob_w) -
        (pi_logprob_l - ref_logprob_l)
    )                                             # [B]

    loss = -F.logsigmoid(reward_diff).mean()      # scalar

    # Diagnostic: fraction of pairs where policy prefers winner
    accuracy = (reward_diff > 0).float().mean()
    return loss, accuracy
```

In practice `trl.DPOTrainer` wraps this logic and handles reference model management, but the core computation above is the complete algorithm.

=== DPO vs PPO Trade-offs

#table(
  columns: (auto, auto, auto),
  [*Aspect*], [*PPO (RLHF)*], [*DPO*],
  [Reward model],        [Explicit, separately trained], [Implicit in policy ratio],
  [Training stability],  [Sensitive; reward hacking common], [Stable supervised loss],
  [GPU memory],          [4 models (actor, critic, ref, reward)], [2 models (policy, ref)],
  [Online sampling],     [Required (policy generates during training)], [Offline (static dataset)],
  [Best for],            [Complex reasoning, RLHF with verifiers], [Chat alignment, preference data],
)

== Catastrophic Forgetting

When a model is fine-tuned on a narrow dataset, gradient descent overwrites the weight configurations that encoded general knowledge. This is _catastrophic forgetting_. It is especially severe in full fine-tuning on small task-specific datasets.

=== Elastic Weight Consolidation (EWC)

EWC (Kirkpatrick et al., 2017) adds a regularization term that penalizes movement away from the pre-trained weights $theta^*$ in proportion to their importance for prior tasks. Importance is estimated by the diagonal of the Fisher information matrix $F$:

$ cal(L)_"EWC"(theta) = cal(L)_"task"(theta) + (lambda / 2) sum_i F_i (theta_i - theta^*_i)^2 $

$F_i$ is approximated as the expected squared gradient of the log-likelihood under the training data distribution:

$ F_i approx EE[(partial / (partial theta_i) log p(y|x; theta^*))^2] $

For large LLMs the Fisher diagonal is expensive to compute and store ($O(P)$ additional memory). In practice, EWC is rarely used for LLMs; LoRA is the preferred alternative.

=== Replay Buffers

A simpler approach: mix a small fraction (5–10%) of pre-training data or a diverse general dataset into each fine-tuning batch. The model continues to see general distribution data and its weights do not specialize entirely. This is sometimes called _continual pre-training mixing_.

```python
from torch.utils.data import ConcatDataset, WeightedRandomSampler

task_dataset    = load_task_data()      # fine-tuning data
general_dataset = load_general_data()  # replay buffer (e.g. C4 sample)

# 90% task, 10% general
weights = ([0.9 / len(task_dataset)]    * len(task_dataset)  +
           [0.1 / len(general_dataset)] * len(general_dataset))

combined = ConcatDataset([task_dataset, general_dataset])
sampler  = WeightedRandomSampler(weights, num_samples=len(task_dataset),
                                 replacement=True)
```

=== LoRA as Natural Regularizer

LoRA avoids catastrophic forgetting by construction: the base weights $W$ are never updated. All task-specific information is encoded in the small $B A$ matrices. The pre-trained representations in $W$ are fully preserved. After fine-tuning, merging ($W' = W + (alpha/r) dot B A$) applies the learned delta but preserves the weight directions that were important for the original task.

This makes LoRA especially effective in _multi-task_ settings: different LoRA adapters can be trained for different tasks and hot-swapped at inference time without modifying the base model.

== Practical Recipes

=== Learning Rate

#table(
  columns: (auto, auto, auto),
  [*Method*], [*Typical LR range*], [*Notes*],
  [Full FT],         [$1 times 10^(-5)$ to $5 times 10^(-5)$], [Cosine decay; warmup 3–5% of steps],
  [LoRA],            [$1 times 10^(-4)$ to $5 times 10^(-4)$], [Higher LR safe; only small adapter trained],
  [QLoRA],           [$1 times 10^(-4)$ to $2 times 10^(-4)$], [NF4 grad scale less stable; conservative upper end],
  [DPO],             [$5 times 10^(-7)$ to $5 times 10^(-6)$], [Very small; policy should move slowly from ref],
)

LoRA tolerates higher learning rates than full fine-tuning because the adapter is randomly initialized and the loss landscape for the low-dimensional subspace is better conditioned. The base model gradients are zero (frozen), so there is no risk of destroying pre-trained features.

=== Batch Size and Gradient Accumulation

Effective batch size = (per-device batch size) × (number of GPUs) × (gradient accumulation steps).

For instruction tuning, an effective batch size of 64–256 is typical. With a single 24 GB GPU and per-device batch size 2 (limited by sequence length and memory), use gradient accumulation steps = 32 to reach effective batch size 64.

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=32,     # effective batch = 64
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    optim="paged_adamw_32bit",          # QLoRA paged optimizer
    dataloader_num_workers=4,
    report_to="wandb",
)
```

=== Epochs and Overfitting

Most instruction datasets are small (10 K–100 K examples). Three epochs is a common default. Signs of overfitting: training loss continues to decrease while evaluation loss plateaus or rises. With LoRA, overfitting is less severe (fewer degrees of freedom), but it still occurs on very small datasets ($<$5 K examples). Options:

- Increase dropout in LoRA (`lora_dropout=0.1`).
- Use early stopping (`load_best_model_at_end=True`, `early_stopping_patience=3`).
- Reduce rank or number of targeted modules.
- Add more diverse data (replay buffer).

=== Evaluation Strategy

For chat/instruction models, perplexity on a held-out set is a weak proxy for quality. Better evaluation:

#table(
  columns: (auto, auto),
  [*Benchmark*], [*What it measures*],
  [MT-Bench],        [Multi-turn instruction following (GPT-4 as judge)],
  [AlpacaEval 2.0],  [Single-turn win rate vs GPT-4 Turbo],
  [IFEval],          [Strict instruction-following (verifiable constraints)],
  [MMLU],            [Knowledge retention (catastrophic forgetting signal)],
  [HumanEval],       [Code generation (task-specific)],
  [TruthfulQA],      [Hallucination resistance],
)

Run MMLU before and after fine-tuning to detect knowledge degradation. A $>$2% absolute drop is a signal to add replay data or reduce training epochs.

=== Full LoRA Fine-tuning Pipeline (End-to-End Sketch)

```python
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, TrainingArguments)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import torch

# 1. Tokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tok.pad_token = tok.eos_token
tok.padding_side = "right"   # causal LM: pad on the right

# 2. Model (QLoRA: 4-bit base)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16,
                          bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb, device_map="auto")
model.config.use_cache = False           # required for gradient checkpointing

# 3. LoRA config
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
    lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_cfg)

# 4. Dataset: apply chat template, mask prompt tokens
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

def format_example(example):
    text = tok.apply_chat_template(example["messages"],
                                    tokenize=False,
                                    add_generation_prompt=False)
    return {"text": text}

dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# 5. Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    args=TrainingArguments(
        output_dir="./llama3-8b-sft",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
    ),
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=True,   # SFTTrainer packs multiple short examples into one context
)
trainer.train()

# 6. Merge and save
model = model.merge_and_unload()   # fuses LoRA into base weights
model.save_pretrained("./llama3-8b-sft-merged")
tok.save_pretrained("./llama3-8b-sft-merged")
```

`packing=True` in `SFTTrainer` concatenates multiple training examples into a single sequence up to `max_seq_length`, separated by EOS tokens. This increases GPU utilization significantly when the dataset contains many short examples.

== References

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. _International Conference on Learning Representations (ICLR)_.

Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. _Advances in Neural Information Processing Systems (NeurIPS)_.

Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., & Le, Q. V. (2022). Finetuned language models are zero-shot learners. _International Conference on Learning Representations (ICLR)_.

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., & Hashimoto, T. B. (2023). Stanford Alpaca: An instruction-following LLaMA model. _GitHub_.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. _Advances in Neural Information Processing Systems (NeurIPS)_.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. _Proceedings of the National Academy of Sciences_.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022). Training language models to follow instructions with human feedback. _Advances in Neural Information Processing Systems (NeurIPS)_.

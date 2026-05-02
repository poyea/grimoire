= Pretraining

Pretraining is the process of learning a general-purpose language model from raw text by predicting the next token at scale. Every capability a model has — reasoning, code generation, factual recall — is acquired here, before any fine-tuning. This chapter covers the full stack: data pipelines, the training objective, scaling laws, numerical precision, memory management, optimizers, distributed training, and stability.

*See also:* _llm/transformer-architecture.typ_ (model internals), _gpu-architecture/multi-gpu.typ_ (hardware communication primitives used in distributed training), _gpu-architecture/ml-workloads.typ_ (GEMM kernels, Flash Attention).

== Data Pipeline

=== Tokenization and BPE

A tokenizer converts raw Unicode text into a sequence of integer token IDs drawn from a fixed vocabulary $V$. Byte-Pair Encoding (BPE) is the dominant algorithm. It starts from a base vocabulary of individual bytes (256 symbols) and iteratively merges the most frequent adjacent pair.

*BPE merge algorithm:*

+ Initialize vocabulary $cal(V) = {0, ..., 255}$ (raw bytes).
+ Represent the corpus as a list of byte-sequences separated by word boundaries.
+ Count all adjacent symbol pairs across the corpus.
+ Merge the most frequent pair $(a, b)$ into a new symbol $[a b]$, add it to $cal(V)$.
+ Repeat steps 3–4 until $|cal(V)| = V_"target"$ (e.g., $V = 128{,}256$ for LLaMA 3).

The merge priority list is the tokenizer: at inference time, you apply the merges in order of acquisition.

```python
# Minimal BPE training (illustrative, not production-speed)
from collections import Counter

def get_pairs(vocab: dict[tuple, int]) -> Counter:
    pairs: Counter = Counter()
    for symbols, freq in vocab.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs

def merge_vocab(pair: tuple, vocab: dict) -> dict:
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for symbols_str, freq in vocab.items():
        new_key = symbols_str.replace(bigram, replacement)
        new_vocab[new_key] = freq
    return new_vocab

# Build initial vocab from word frequencies
word_freqs = {"l o w </w>": 5, "l o w e r </w>": 2,
              "n e w e s t </w>": 6, "w i d e s t </w>": 3}

num_merges = 10
for i in range(num_merges):
    pairs = get_pairs({tuple(k.split()): v for k, v in word_freqs.items()})
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    word_freqs = merge_vocab(best, word_freqs)
    print(f"Merge {i+1}: {best}")
```

```python
# Production: tiktoken (OpenAI) — used by LLaMA 3, GPT-4
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")   # GPT-4 / LLaMA 3 tokenizer
ids = enc.encode("The transformer learns representations.")
print(ids)          # list of ints
print(enc.decode(ids))  # round-trip
```

*Vocabulary sizes and compression ratios:*

#table(
  columns: (auto, auto, auto, auto),
  [*Model*], [*Vocab V*], [*Algorithm*], [*Avg tokens/word (EN)*],
  [GPT-2],      [50 257],  [BPE],              [1.3],
  [LLaMA 1/2],  [32 000],  [SentencePiece BPE],[1.3],
  [LLaMA 3],    [128 256], [tiktoken BPE],     [1.2],
  [Gemma 2],    [256 000], [SentencePiece BPE],[1.15],
  [DeepSeek-V3],[129 280], [BPE],              [1.2],
)

A larger vocabulary reduces sequence length (lower compute) but increases the embedding matrix size and makes rare tokens harder to learn.

=== Deduplication with MinHash

Web-crawled data has massive duplication: the same news article, StackOverflow answer, or GitHub file appears hundreds of times. Training on duplicates wastes compute and causes memorization. MinHash LSH (Locality-Sensitive Hashing) deduplicates at scale.

*Algorithm sketch:*

+ Shingling: convert each document into a set of $k$-grams (typically $k=5$ word shingles or character $k$-grams).
+ MinHash signatures: apply $h$ independent hash functions to the shingle set. The _minhash_ of function $f_i$ over document $d$ is $m_i(d) = min_{s in d} f_i(s)$. Stack into a signature vector $bold(s)(d) in ZZ^h$.
+ Jaccard estimate: $hat(J)(d_1, d_2) = (1/h) sum_i bb(1)[m_i(d_1) = m_i(d_2)]$. This is an unbiased estimator of the true Jaccard similarity.
+ LSH banding: divide the $h$ hash values into $b$ bands of $r = h/b$ rows each. Two documents become a _candidate pair_ if they collide in at least one band. Tune $b, r$ to control the similarity threshold.
+ Deduplicate: remove one document from each near-duplicate pair (Jaccard $gt$ 0.8 is a common threshold).

```python
# MinHash deduplication skeleton (datasketch library)
from datasketch import MinHash, MinHashLSH

def doc_to_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    # 5-gram shingles over whitespace-split tokens
    tokens = text.lower().split()
    for i in range(len(tokens) - 4):
        shingle = " ".join(tokens[i:i+5])
        m.update(shingle.encode("utf-8"))
    return m

lsh = MinHashLSH(threshold=0.8, num_perm=128)
docs = ["The quick brown fox", "The quick brown fox jumps",
        "A completely different sentence about cats."]

minhashes = [doc_to_minhash(d) for d in docs]
for i, (doc, mh) in enumerate(zip(docs, minhashes)):
    lsh.insert(f"doc_{i}", mh)

# Query: find near-duplicates of doc 0
result = lsh.query(minhashes[0])
print("Near-duplicates of doc_0:", result)
```

At the scale of CommonCrawl (petabytes), this pipeline runs on Spark or Ray with billions of documents. The RedPajama and FineWeb datasets report removing 20–40% of documents as near-duplicates.

=== Dataset Mixture

Modern LLMs train on a weighted mixture of sources. The mixture ratios profoundly affect downstream capability — code-heavy mixtures improve reasoning, book-heavy mixtures improve long-form coherence.

*Typical mixture for a 1T–2T token pretraining run:*

#table(
  columns: (auto, auto, auto, auto),
  [*Source*], [*Raw size*], [*After filtering*], [*Weight*],
  [CommonCrawl (CC)],    [$tilde 70$ PB HTML],  [$tilde 3.8$ T tokens],  [67%],
  [Books (Gutenberg, Books3)], [$tilde 100$ GB], [$tilde 26$ B tokens],  [8%],
  [Wikipedia / Wikidata], [$tilde 20$ GB],      [$tilde 4$ B tokens],    [4%],
  [Code (GitHub, Stack)], [$tilde 1$ TB],       [$tilde 250$ B tokens],  [12%],
  [ArXiv / PubMed],       [$tilde 50$ GB],      [$tilde 30$ B tokens],   [4%],
  [StackExchange],        [$tilde 80$ GB],       [$tilde 20$ B tokens],  [5%],
)

LLaMA 3 uses 15T tokens, heavily oversampling high-quality sources. Repeat passes over high-quality data (books, Wikipedia) are common and beneficial up to ~4 epochs (Muennighoff et al., 2023).

=== Quality Filters

Raw crawl data is noisy. Standard filters applied in order:

+ *Language identification:* fastText lid.176 model; discard non-target-language documents (threshold $p_"lang" gt 0.65$).
+ *Perplexity filter:* train a small 5-gram KenLM on a clean seed corpus (Wikipedia); discard documents with perplexity above the 90th percentile.
+ *Heuristic filters:* discard documents with $lt 200$ words; word repetition ratio $gt 20%$; symbol-to-word ratio $gt 10%$; fraction of lines ending in ellipsis $gt 30%$; mean word length outside $[3, 10]$.
+ *Exact deduplication:* SHA-256 of normalized (lowercase, whitespace-collapsed) document content.
+ *Safety filters:* NSFW classifier, PII (phone numbers, emails, SSNs) redaction via regex + NER.

```python
import re, kenlm, fasttext

lm    = kenlm.Model("wiki_en_5gram.arpa")   # pre-built
lid   = fasttext.load_model("lid.176.bin")

def quality_score(doc: str) -> dict:
    words = doc.split()
    lang, prob = lid.predict(doc.replace("\n", " "), k=1)
    ppl = lm.perplexity(doc)
    rep_ratio = (len(words) - len(set(words))) / max(len(words), 1)
    symbol_ratio = len(re.findall(r"[^a-zA-Z0-9\s]", doc)) / max(len(words), 1)
    return {
        "lang": lang[0].replace("__label__", ""),
        "lang_prob": float(prob[0]),
        "perplexity": ppl,
        "rep_ratio": rep_ratio,
        "symbol_ratio": symbol_ratio,
        "word_count": len(words),
    }

def keep(doc: str) -> bool:
    s = quality_score(doc)
    return (s["lang"] == "en" and s["lang_prob"] > 0.65
            and s["perplexity"] < 500
            and s["rep_ratio"] < 0.20
            and s["symbol_ratio"] < 0.10
            and s["word_count"] >= 200)
```

== Causal Language Modeling Objective

=== Next-Token Prediction

A decoder-only transformer is trained with the _causal language modeling_ (CLM) objective. Given a sequence of tokens $(x_1, x_2, ..., x_T)$, the model learns to predict each token from all preceding tokens:

$ p_theta (x_1, ..., x_T) = product_(t=1)^T p_theta (x_t | x_1, ..., x_(t-1)) $

The parameters $theta$ are learned by minimizing the negative log-likelihood averaged over positions and training examples:

$ cal(L)(theta) = - 1/(N T) sum_(n=1)^N sum_(t=1)^T log p_theta (x_t^((n)) | x_1^((n)), ..., x_(t-1)^((n))) $

In practice the loss is computed as cross-entropy between the model's softmax output and the one-hot target:

$ "CE"(bold(p), bold(q)) = - sum_(v=1)^V q_v log p_v $

where $bold(q)$ is the one-hot target (all mass on $x_t$) and $bold(p) = "softmax"("logits"_t)$.

=== Perplexity

Perplexity is the standard intrinsic evaluation metric for language models. It measures how "surprised" the model is by held-out text:

$ "PPL"(theta, cal(D)) = exp(- 1/T sum_(t=1)^T log p_theta (x_t | x_(t-1))) $

A model assigning uniform probability over $V$ tokens has perplexity $V$. Good models achieve single-digit perplexity on in-distribution text (LLaMA 3 70B: PPL $approx$ 2.85 on Wikitext-103). Lower is better.

```python
import torch, torch.nn.functional as F

def compute_perplexity(model, input_ids: torch.Tensor) -> float:
    """
    input_ids: [1, T] — a single document, already tokenized.
    Returns scalar perplexity.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss is mean cross-entropy over non-padding tokens
        nll = outputs.loss.item()
    return float(torch.exp(torch.tensor(nll)))
```

```python
# Sliding-window perplexity for documents longer than context length
def sliding_ppl(model, enc, text: str, stride: int = 512,
                max_len: int = 4096) -> float:
    ids = enc.encode(text)
    T   = len(ids)
    nlls = []
    for begin in range(0, T, stride):
        end     = min(begin + max_len, T)
        chunk   = torch.tensor(ids[begin:end]).unsqueeze(0)
        ctx_len = end - begin
        target  = chunk.clone()
        # mask positions we've already scored in previous window
        target[:, :max(0, max_len - stride)] = -100
        with torch.no_grad():
            loss = model(chunk, labels=target).loss
        nlls.append(loss * (end - begin))
        if end == T:
            break
    return float(torch.exp(torch.stack(nlls).sum() / T))
```

== Chinchilla Scaling Laws

=== The Compute-Optimal Frontier

Kaplan et al. (2020) showed that loss scales as a power law in $N$ (parameters) and $D$ (training tokens). Hoffmann et al. (2022) — the _Chinchilla_ paper — refined these estimates and showed that prior models (GPT-3, Gopher) were significantly undertrained. Their key finding: for a fixed compute budget $C approx 6 N D$ FLOPs, the optimal allocation is:

$ N_"opt" = G_N dot C^(a) , quad D_"opt" = G_D dot C^(b) $

with $a approx b approx 0.5$, meaning *parameters and tokens should scale equally*. The Chinchilla rule of thumb: train on approximately $20 dot N$ tokens.

The predicted loss as a function of $N$ and $D$:

$ hat(L)(N, D) = E + A / N^alpha + B / D^beta $

with fitted constants $E = 1.61$, $A = 406.4$, $B = 410.7$, $alpha = 0.34$, $beta = 0.28$ (Hoffmann et al., 2022).

=== Concrete Budget Table

Compute $C$ is measured in FLOPs. For a dense transformer, $C approx 6 N D$ (forward + backward, ignoring attention and activations for simplicity).

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Compute budget C (FLOPs)*], [*Optimal N*], [*Optimal D (tokens)*], [*GPU-days (A100)*], [*Example*],
  [$10^(21)$], [$1.3$ B],   [$26$ B],   [$2$],       [Small experiment],
  [$10^(22)$], [$4$ B],    [$82$ B],   [$20$],      [—],
  [$10^(23)$], [$12$ B],   [$260$ B],  [$200$],     [LLaMA 1 13B (approx)],
  [$10^(24)$], [$37$ B],   [$820$ B],  [$2{,}000$], [Chinchilla 70B],
  [$3 times 10^(24)$], [$67$ B], [$1.5$ T],  [$6{,}000$], [LLaMA 2 70B (approx)],
  [$10^(25)$], [$120$ B],  [$2.6$ T],  [$20{,}000$],[LLaMA 3 70B (approx)],
)

_Note:_ LLaMA 3 8B trains on 15T tokens — deliberately far past Chinchilla-optimal for a small model, optimizing _inference_ cost at a fixed serving budget rather than training cost. This "overtrain small models" strategy is practical when models are deployed at scale.

```python
# Chinchilla optimal allocation given compute budget
def chinchilla_optimal(C_flops: float,
                       G_N: float = 8.8e9,
                       G_D: float = 2.2e10) -> tuple[float, float]:
    """
    Returns (N_opt, D_opt) following Hoffmann et al. 2022.
    C_flops: total compute in FLOPs (use 6*N*D approximation).
    Default G_N, G_D from the paper's IsoFLOP fits.
    """
    import math
    N_opt = G_N * (C_flops ** 0.5)
    D_opt = G_D * (C_flops ** 0.5)
    return N_opt, D_opt

for exp in [21, 22, 23, 24, 25]:
    N, D = chinchilla_optimal(10**exp)
    print(f"C=1e{exp}: N={N/1e9:.1f}B params, D={D/1e9:.1f}B tokens")
```

== Mixed Precision Training

=== FP32, BF16, and the Master Weight Pattern

Training in FP32 throughout uses 4 bytes per parameter. For a 7B model that is already 28 GB just for weights — before gradients (28 GB) and Adam states (56 GB). Mixed precision training (Micikevicius et al., 2018) dramatically reduces memory while preserving convergence:

+ *Master weights* are stored in FP32 (4 bytes/param). These are the source of truth updated by the optimizer.
+ *Forward and backward passes* use BF16 (2 bytes/param). BF16 has the same 8-bit exponent as FP32 and is thus more numerically stable than FP16 for large models.
+ *Loss scaling* (critical for FP16; less so for BF16): multiply the loss by a large scalar $S$ before backward, then divide gradients by $S$ before the optimizer step, to keep gradients in the representable range and avoid underflow.

*BF16 vs FP16:*

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Format*], [*Sign*], [*Exponent bits*], [*Mantissa bits*], [*Dynamic range*],
  [FP32],  [1], [8],  [23], [$approx 1.2 times 10^(-38)$ to $3.4 times 10^(38)$],
  [BF16],  [1], [8],  [7],  [same as FP32],
  [FP16],  [1], [5],  [10], [$approx 6 times 10^(-5)$ to $65504$],
)

BF16 is preferred for LLM training (H100, A100 both support it natively at high throughput). FP16 requires careful loss scaling and can still produce NaNs from gradient overflow.

=== PyTorch AMP Example

```python
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

model     = MyTransformer().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
scaler    = GradScaler()          # only needed for FP16; harmless for BF16

for batch in dataloader:
    input_ids = batch["input_ids"].cuda()
    labels    = batch["labels"].cuda()

    optimizer.zero_grad()

    # Forward pass in BF16 (or FP16 with dtype=torch.float16)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(input_ids, labels=labels)
        loss    = outputs.loss

    # Backward pass: scaler handles loss scaling (no-op for BF16)
    scaler.scale(loss).backward()

    # Unscale before clipping so clip threshold is in FP32 units
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
```

*Memory breakdown for a 7B model (BF16 compute, FP32 master weights):*

#table(
  columns: (auto, auto, auto),
  [*Component*], [*Dtype*], [*Memory*],
  [Model weights (inference copy)], [BF16], [14 GB],
  [Master weights (optimizer)],     [FP32], [28 GB],
  [Gradients],                      [FP32], [28 GB],
  [Adam m (first moment)],          [FP32], [28 GB],
  [Adam v (second moment)],         [FP32], [28 GB],
  [*Total*],                        [*—*],  [*126 GB*],
)

This is why a 7B model requires at least two 80 GB A100s for pretraining with standard mixed precision — or sharding via FSDP/ZeRO-3.

== Gradient Checkpointing

=== The Memory–Compute Tradeoff

During the forward pass, PyTorch stores all intermediate activations needed for backpropagation. For a transformer with $L$ layers, sequence length $S$, batch size $B$, and hidden dimension $d$, activation memory scales as $O(L B S d)$. At LLaMA 3 8B scale ($L=32$, $d=4096$, $S=8192$, $B=4$), this is roughly:

$ 32 times 4 times 8192 times 4096 times 2 " bytes" approx 8 " GB " $

just for the residual stream — before attention matrices. Full activation memory for a forward pass is $O(B S d_"ffn") times L approx 60$–$80$ GB.

*Gradient checkpointing* (Chen et al., 2016) reduces this to $O(sqrt(L))$ by storing only a subset of layer outputs (the _checkpoints_) and recomputing the others during the backward pass. The tradeoff: recomputation adds approximately 33% to total FLOPs.

=== PyTorch Example

```python
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class CheckpointedTransformerBlock(nn.Module):
    def __init__(self, attn, ffn, norm1, norm2):
        super().__init__()
        self.attn, self.ffn   = attn, ffn
        self.norm1, self.norm2 = norm1, norm2

    def forward(self, x, cos, sin):
        # checkpoint wraps a function; no graph stored for internals
        def attn_fn(x):
            return x + self.attn(self.norm1(x), cos, sin)
        def ffn_fn(x):
            return x + self.ffn(self.norm2(x))

        x = checkpoint(attn_fn, x, use_reentrant=False)
        x = checkpoint(ffn_fn,  x, use_reentrant=False)
        return x
```

```python
# Full model: checkpoint every layer
class LLaMA(nn.Module):
    def forward(self, x):
        for block in self.blocks:
            # use_reentrant=False avoids a subtle double-backward bug
            x = checkpoint(block, x, use_reentrant=False)
        return self.head(self.norm(x))
```

*Selective checkpointing:* gradient checkpoint only the attention layers (which have large $O(S^2)$ activation maps) and keep the FFN activations. This recovers ~50% of the memory savings with only ~15% recompute overhead.

```python
# Hugging Face transformers: enable gradient checkpointing
model.gradient_checkpointing_enable()

# Or per-module granularity:
from functools import partial
model.config.use_cache = False   # incompatible with checkpointing
for layer in model.model.layers:
    layer.forward = partial(checkpoint, layer.forward, use_reentrant=False)
```

== Optimizer

=== AdamW

Adam (Kingma & Ba, 2015) maintains per-parameter first and second moment estimates. AdamW (Loshchilov & Hutter, 2019) decouples weight decay from the gradient update, which is theoretically cleaner and empirically better than the L2-regularization-as-Adam variant.

*Update rule* for parameter $theta_t$ at step $t$:

$ m_t &= beta_1 m_(t-1) + (1 - beta_1) g_t \
v_t &= beta_2 v_(t-1) + (1 - beta_2) g_t^2 \
hat(m)_t &= m_t / (1 - beta_1^t) \
hat(v)_t &= v_t / (1 - beta_2^t) \
theta_(t+1) &= theta_t - eta (hat(m)_t / (sqrt(hat(v)_t) + epsilon)) - eta lambda theta_t $

The last term $eta lambda theta_t$ is the decoupled weight decay. Standard hyperparameters for LLM pretraining:

#table(
  columns: (auto, auto, auto),
  [*Hyperparameter*], [*Symbol*], [*Typical value*],
  [Learning rate],    [$eta$],          [$3 times 10^(-4)$ (7B), $1.5 times 10^(-4)$ (70B)],
  [Beta 1],           [$beta_1$],       [0.9],
  [Beta 2],           [$beta_2$],       [0.95],
  [Epsilon],          [$epsilon$],      [$10^(-8)$ (use $10^(-5)$ with BF16)],
  [Weight decay],     [$lambda$],       [0.1],
  [Gradient clip],    [$g_"max"$],      [1.0],
  [Warmup steps],     [—],              [2000 (7B–70B)],
  [Total steps],      [—],              [$tilde 10^6$ for 1T tokens, batch 2048],
)

=== Learning Rate Schedule: Cosine with Warmup

$ eta(t) = cases(
  eta_"max" dot t / t_"warm" & "if" t lt t_"warm",
  eta_"min" + 1/2 (eta_"max" - eta_"min")(1 + cos(pi (t - t_"warm") / (t_"total" - t_"warm"))) & "otherwise"
) $

Typical values: $t_"warm" = 2000$ steps, $eta_"min" = eta_"max" / 10$.

```python
import math

def cosine_with_warmup(step: int, max_lr: float, min_lr: float,
                       warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

# Integrate with PyTorch scheduler
from torch.optim.lr_scheduler import LambdaLR

def make_scheduler(optimizer, warmup_steps: int, total_steps: int):
    max_lr = optimizer.param_groups[0]["lr"]
    min_lr = max_lr / 10.0
    def lr_lambda(step: int) -> float:
        return cosine_with_warmup(step, max_lr, min_lr,
                                  warmup_steps, total_steps) / max_lr
    return LambdaLR(optimizer, lr_lambda)
```

=== AdamW Weight Update Kernel (C++)

In large-scale training, the optimizer step is memory-bandwidth bound. A fused CUDA/C++ kernel that updates all states in a single pass over memory reduces kernel launch overhead and improves cache utilization.

```cpp
// Fused AdamW weight update — operates on flattened parameter arrays.
// Compile with: nvcc -O3 -arch=sm_90 adamw_kernel.cu
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

__global__ void adamw_update_kernel(
    float*        __restrict__ master_w,   // FP32 master weights
    __nv_bfloat16* __restrict__ model_w,   // BF16 model weights (for forward)
    float*        __restrict__ grad,       // FP32 gradient
    float*        __restrict__ m,          // first moment (FP32)
    float*        __restrict__ v,          // second moment (FP32)
    const float   lr,
    const float   beta1,
    const float   beta2,
    const float   eps,
    const float   weight_decay,
    const float   bias_corr1,              // 1 - beta1^t
    const float   bias_corr2,              // 1 - beta2^t
    const int64_t n_elems)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elems) return;

    float g = grad[idx];
    float m_i = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_i = beta2 * v[idx] + (1.0f - beta2) * g * g;
    m[idx] = m_i;
    v[idx] = v_i;

    float m_hat = m_i / bias_corr1;
    float v_hat = v_i / bias_corr2;

    float w = master_w[idx];
    w = w - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w);
    master_w[idx] = w;
    model_w[idx]  = __float2bfloat16(w);   // downcast to BF16 for forward pass
}

// Host launcher
void adamw_step(/* ... pointers ...*/, int step, int64_t n_elems) {
    float bias_corr1 = 1.0f - powf(0.9f,  (float)step);
    float bias_corr2 = 1.0f - powf(0.95f, (float)step);
    int threads = 512;
    int blocks  = (int)((n_elems + threads - 1) / threads);
    adamw_update_kernel<<<blocks, threads>>>(
        /* ... */,
        bias_corr1, bias_corr2, n_elems);
}
```

```python
# PyTorch AdamW (standard usage)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
    fused=True,          # uses CUDA fused kernel when available
)

# Gradient clipping before optimizer step
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)   # free gradient memory immediately
```

== Distributed Training

*Cross-reference:* _gpu-architecture/multi-gpu.typ_ covers the underlying communication primitives (NCCL all-reduce, all-gather, reduce-scatter, NVLink/InfiniBand bandwidth).

Large language models require distributing computation across tens to thousands of GPUs. Three orthogonal parallelism strategies are combined in practice.

=== Data Parallel Training (DDP)

Each GPU holds a *full copy* of the model. The global batch is split: each GPU processes a _micro-batch_, computes gradients independently, then gradients are _all-reduced_ (summed and divided) across all GPUs before the optimizer step. After all-reduce, every GPU has identical gradients and performs an identical optimizer step.

$ g_"global" = (1/K) sum_(k=1)^K g^{(k)} $

All-reduce cost: $2(K-1)/K times P times "sizeof(float)"$ bytes transmitted per GPU for ring all-reduce over $K$ GPUs and $P$ parameters.

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# Initialize process group (called once per process)
dist.init_process_group(backend="nccl")    # NCCL for GPU-GPU communication
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = MyTransformer().cuda(local_rank)
model = DDP(model, device_ids=[local_rank],
            find_unused_parameters=False,   # True only if needed; has overhead
            gradient_as_bucket_view=True)   # reduce memory copies

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                               betas=(0.9, 0.95), weight_decay=0.1)

for batch in dataloader:   # dataloader uses DistributedSampler
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    loss.backward()        # DDP hooks trigger all-reduce here
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

dist.destroy_process_group()
```

Launch: `torchrun --nproc_per_node=8 train.py`

*Limitation:* DDP requires each GPU to hold a full model copy. For a 70B model (FP32 master weights + states $approx 560$ GB), a single DDP replica needs 7 $times$ 80 GB A100s — impractical to replicate across hundreds of GPUs.

=== ZeRO / FSDP: Sharding Parameters, Gradients, and Optimizer States

ZeRO (Zero Redundancy Optimizer, Rajbhandari et al., 2020) eliminates the memory redundancy in DDP by sharding model state across data-parallel ranks. PyTorch's _Fully Sharded Data Parallel_ (FSDP) implements ZeRO-3.

*ZeRO stages:*

#table(
  columns: (auto, auto, auto, auto),
  [*Stage*], [*What is sharded*], [*Memory per GPU (70B)*], [*Extra communication*],
  [DDP (stage 0)], [Nothing], [$tilde 560$ GB], [All-reduce gradients],
  [ZeRO-1],        [Optimizer states], [$tilde 280$ GB], [All-reduce gradients],
  [ZeRO-2],        [Optimizer states + gradients], [$tilde 140$ GB], [Reduce-scatter grads],
  [ZeRO-3 / FSDP], [Optimizer states + gradients + parameters], [$tilde 20$ GB], [All-gather params + reduce-scatter grads],
)

With ZeRO-3 / FSDP, each GPU holds $1/K$ of every tensor. Before a forward or backward pass through a given layer, the full layer weights are reconstructed via an _all-gather_; they are discarded immediately after use. Gradients are aggregated via _reduce-scatter_ (each rank keeps its shard of the reduced gradients).

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

# BF16 mixed precision policy
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,     # shard params in BF16
    reduce_dtype=torch.float32,     # reduce grads in FP32 for precision
    buffer_dtype=torch.bfloat16,
)

# Wrap each transformer block independently for fine-grained sharding
auto_wrap = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock}
)

model = FSDP(
    MyTransformer(),
    sharding_strategy=ShardingStrategy.FULL_SHARD,   # ZeRO-3
    mixed_precision=mp_policy,
    auto_wrap_policy=auto_wrap,
    device_id=local_rank,
    use_orig_params=True,    # required for parameter groups / weight decay masking
)

# Optimizer and training loop are identical to DDP
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4,
                               betas=(0.9, 0.95), weight_decay=0.1)
```

*Activation checkpointing with FSDP:*

```python
from torch.distributed.fsdp import checkpoint_wrapper, CheckpointImpl

for layer in model.model.layers:
    # Wrap each layer for both FSDP sharding and gradient checkpointing
    checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
```

=== Tensor Parallelism

Tensor parallelism (Shoeybi et al., 2019 — Megatron-LM) splits individual weight matrices across GPUs. It reduces per-GPU memory proportionally to the tensor-parallel degree $T$ and enables within-node, high-bandwidth parallelism over NVLink.

*Column-parallel linear (e.g., QKV projection, first FFN layer):*

$ Y = X W^T , quad W in RR^(d times k) $

Split $W$ column-wise: each GPU $i$ holds $W_i in RR^(d times k/T)$ and computes $Y_i = X W_i^T$. No communication is needed after this operation — $Y_i$ are independent and passed to the next layer.

*Row-parallel linear (e.g., output projection, second FFN layer):*

Split $W$ row-wise: each GPU $i$ holds $W_i in RR^(k/T times d)$ and receives its corresponding shard $X_i$ of the input. Computes partial output $Y_i = X_i W_i^T$. An _all-reduce_ combines the partial sums: $Y = sum_i Y_i$.

In an MLP block (column-parallel GELU row-parallel), only two all-reduces are needed per layer — one after the attention output projection, one after the FFN output projection.

```python
# Megatron-LM style column-parallel linear (conceptual)
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp_group):
        super().__init__()
        T = dist.get_world_size(tp_group)
        assert out_features % T == 0
        self.local_out = out_features // T
        self.weight = nn.Parameter(torch.empty(self.local_out, in_features))
        self.tp_group = tp_group
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each GPU computes its output shard independently
        return F.linear(x, self.weight)    # no communication needed here


class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, tp_group):
        super().__init__()
        T = dist.get_world_size(tp_group)
        assert in_features % T == 0
        self.local_in = in_features // T
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in))
        self.tp_group = tp_group
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partial = F.linear(x, self.weight)
        dist.all_reduce(partial, group=self.tp_group)   # sum partial outputs
        return partial
```

=== Combining Parallelism Strategies

Production LLM training typically combines all three:

#table(
  columns: (auto, auto, auto),
  [*Strategy*], [*Typical degree*], [*Communication domain*],
  [Tensor parallel (TP)],    [$T = 8$],           [Within node (NVLink)],
  [Pipeline parallel (PP)],  [$P = 4$–$8$],       [Between nodes (IB or NVLink)],
  [Data parallel (ZeRO-3)],  [$D = N_"GPU" / (T times P)$], [Between nodes],
)

For 8192 H100s (1024 nodes of 8), a typical 3D config is $T=8, P=8, D=128$. The _effective global batch size_ is $D times B_"micro"$. For LLaMA 3 70B: $D=512$, $B_"micro"=4$ sequences of length 8192 gives a global batch of 2048 sequences = $16.7$M tokens.

== Training Stability

=== Loss Spikes and Recovery

Training runs at scale routinely encounter loss spikes — sudden increases in loss by 0.1–1.0 nats. Common causes:

+ *Batch with anomalous data:* a single very-long or repetitive document dominates the batch gradient.
+ *Gradient explosion:* accumulated numerical errors compound, especially after long stable stretches.
+ *Learning rate too high:* insufficient warmup or an overly aggressive schedule.

Standard mitigations:
- *Gradient clipping* at max norm 1.0 is the first line of defense. Monitor the pre-clip gradient norm — a healthy run has $||g|| in [0.5, 2.0]$; norms above 10 indicate instability.
- *Loss spike detection:* if the loss at step $t$ exceeds 1.5$times$ the rolling mean over the past 100 steps, roll back to the last checkpoint and skip or down-weight the offending batch.
- *BF16 over FP16:* BF16's wider dynamic range (matching FP32) prevents the overflow/underflow cycles that cause FP16 loss spikes.

```python
# Monitoring gradient norm during training
def train_step(model, batch, optimizer, scaler, clip_norm=1.0):
    optimizer.zero_grad(set_to_none=True)
    with autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

    # Log and alert on anomalous gradient norms
    if grad_norm > 10.0:
        print(f"WARNING: large grad norm {grad_norm:.2f} at step {step}")

    scaler.step(optimizer)
    scaler.update()
    return loss.item(), grad_norm.item()
```

=== Gradient Norms as a Diagnostic

Plot $||g||_2$ per step. Typical patterns:

#table(
  columns: (auto, auto),
  [*Observation*], [*Interpretation*],
  [Norm steadily decreasing to near 0],   [Learning rate too low or model converged early],
  [Norm oscillates 0.5–2.0 with rare spikes], [Healthy training],
  [Norm consistently $gt 5$],            [Instability — reduce LR or increase warmup],
  [Norm suddenly jumps to $gt 50$ then NaN], [Gradient explosion — check for bad data or LR spike],
  [Norm drops sharply then loss stagnates], [Possible dead neurons or learning rate collapse],
)

=== µP: Maximal Update Parametrization

Standard initialization (e.g., Kaiming normal) causes feature scale to change with model width $d$, so optimal hyperparameters (learning rate, initialization scale) shift as model size changes — making hyperparameter transfer across scales unreliable.

µP (Yang et al., 2022) parametrizes weights so that the _feature update scale_ is $O(1)$ independent of width. Key changes relative to standard parametrization:

+ *Input weights:* $W_"in" tilde cal(N)(0, 1)$ (no $1/d$ factor) — input embeddings and first-layer weights.
+ *Hidden weights:* $W_"hidden" tilde cal(N)(0, 1/d)$ and *learning rate scaled by $1/d$*: $eta_"hidden" = eta_"base" / d$.
+ *Output weights:* $W_"out" tilde cal(N)(0, 1/d)$ with learning rate $eta_"out" = eta_"base"$.
+ *Attention logit scale:* use $1/d_k$ instead of $1/sqrt(d_k)$ (absorb into output projection scaling).

*Why it matters:* with µP, you can tune hyperparameters on a small proxy model (e.g., 40M params) and transfer them to the large model (7B, 70B) without re-tuning. Microsoft's Cerebras-GPT and phi-2 use µP.

```python
# mup library (Yang et al.) — drop-in replacement for nn.Linear
from mup import MuReadout, MuSharedReadout, set_base_shapes, make_base_shapes

# 1. Build a "base" (small) model and a "delta" model with different width
base_model  = MyTransformer(d_model=256)
delta_model = MyTransformer(d_model=512)
target      = MyTransformer(d_model=4096)   # the model you will train

# 2. Compute base shapes (defines the µP scaling rules)
base_shapes = make_base_shapes(base_model, delta_model, savefile="base_shapes.bsh")
set_base_shapes(target, base_shapes)

# 3. Use mup.MuAdamW instead of AdamW
from mup import MuAdamW
optimizer = MuAdamW(target.parameters(), lr=3e-4)
# lr will be automatically scaled per parameter group according to µP
```

=== Embedding and Output Layer Initialization

Embedding matrices are often initialized with $sigma = 0.01$ (smaller than Kaiming) to prevent large initial logits that saturate the softmax and produce uninformative gradients. The output (lm_head) weight is zero-initialized or tied to the embedding. Bias terms in attention projections are often omitted entirely in modern LLMs (LLaMA, Mistral, Gemma).

== References

- Radford, A. et al. "Language Models are Unsupervised Multitask Learners." OpenAI Blog, 2019. _(GPT-2)_
- Gage, P. "A New Algorithm for Data Compression." C Users Journal, 1994. _(BPE)_
- Broder, A. "On the resemblance and containment of documents." Compression and Complexity of Sequences, 1997. _(MinHash)_
- Micikevicius, P. et al. "Mixed Precision Training." ICLR, 2018.
- Chen, T. et al. "Training Deep Nets with Sublinear Memory Cost." arXiv:1604.06174, 2016. _(Gradient checkpointing)_
- Kingma, D. P. & Ba, J. "Adam: A Method for Stochastic Optimization." ICLR, 2015.
- Loshchilov, I. & Hutter, F. "Decoupled Weight Decay Regularization." ICLR, 2019. _(AdamW)_
- Kaplan, J. et al. "Scaling Laws for Neural Language Models." arXiv:2001.08361, 2020.
- Hoffmann, J. et al. "Training Compute-Optimal Large Language Models." NeurIPS, 2022. _(Chinchilla)_
- Muennighoff, N. et al. "Scaling Data-Constrained Language Models." NeurIPS, 2023.
- Rajbhandari, S. et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." SC20, 2020.
- Shoeybi, M. et al. "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." arXiv:1909.08053, 2019.
- Yang, G. et al. "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer." NeurIPS, 2022. _(µP)_
- Touvron, H. et al. "LLaMA 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288, 2023.
- Dubey, A. et al. "The LLaMA 3 Herd of Models." arXiv:2407.21783, 2024.
- Wenzek, G. et al. "CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data." LREC, 2020. _(perplexity-based filtering)_

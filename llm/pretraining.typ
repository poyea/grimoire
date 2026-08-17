#import "../template.typ": xref

= Pretraining

Pretraining is the process of learning a general-purpose language model from raw text by predicting the next token at scale. Every capability a model has (reasoning, code generation, factual recall) is acquired here, before any fine-tuning. This chapter covers the full stack: data pipelines, the training objective, scaling laws, numerical precision, memory management, optimizers, distributed training, and stability.

*See also:* #xref("llm", "transformer-architecture", label: "Transformer Architecture") (model internals), #xref("gpu-architecture", "multi-gpu", label: "Multi-GPU Communication and Scaling (GPU Architecture volume)") (hardware communication primitives used in distributed training), #xref("gpu-architecture", "ml-workloads", label: "ML Workload Optimization on GPUs (GPU Architecture volume)") (GEMM kernels, Flash Attention).

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
# Production: tiktoken (OpenAI), used by GPT-4 and (a related variant) by LLaMA 3
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")   # GPT-3.5 / GPT-4 (100 256 tokens)
# LLaMA 3 ships its own tiktoken merges with a 128 256-token vocabulary.
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

*SentencePiece and Unigram LM.* SentencePiece (Kudo & Richardson, 2018; used by LLaMA 1/2, Gemma, T5, mT5) is a tokenizer _framework_ that operates directly on raw Unicode strings (treating whitespace as a regular symbol, prefixed `▁`), so detokenization is fully reversible and language-agnostic. It supports both BPE and the _Unigram language model_ algorithm (Kudo, 2018) as alternatives. Unigram starts from a large seed vocabulary and iteratively prunes tokens that least increase the corpus likelihood under a unigram LM, yielding probabilistic segmentations (useful for subword regularization at training time). SentencePiece additionally enables _byte-fallback_: any token that fails to encode is decomposed into raw UTF-8 bytes (256 reserved ids), guaranteeing zero out-of-vocabulary tokens even on unseen scripts or emoji, the same property tiktoken achieves by starting from a byte-level base vocabulary.

=== Deduplication with MinHash

Web-crawled data has massive duplication: the same news article, StackOverflow answer, or GitHub file appears hundreds of times. Training on duplicates wastes compute and causes memorization. MinHash LSH (Locality-Sensitive Hashing) deduplicates at scale.

*Algorithm sketch:*

+ Shingling: convert each document into a set of $k$-grams (typically $k=5$ word shingles or character $k$-grams).
+ MinHash signatures: apply $h$ independent hash functions to the shingle set. The _minhash_ of function $f_i$ over document $d$ is $m_i(d) = min_(s in d) f_i(s)$. Stack into a signature vector $bold(s)(d) in ZZ^h$.
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

Modern LLMs train on a weighted mixture of sources. The mixture ratios profoundly affect downstream capability: code-heavy mixtures improve reasoning, book-heavy mixtures improve long-form coherence.

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

A model assigning uniform probability over $V$ tokens has perplexity $V$. Strong base LMs reach low single-digit perplexity on in-distribution corpora; figures depend heavily on tokenizer and the exact eval split, so any cross-model comparison must control for both. Lower is better.

```python
import torch, torch.nn.functional as F

def compute_perplexity(model, input_ids: torch.Tensor) -> float:
    """
    input_ids: [1, T] -- a single document, already tokenized.
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

Kaplan et al. (2020) showed that loss scales as a power law in $N$ (parameters) and $D$ (training tokens). Hoffmann et al. (2022, the _Chinchilla_ paper) refined these estimates and showed that prior models (GPT-3, Gopher) were significantly undertrained. Their key finding: for a fixed compute budget $C approx 6 N D$ FLOPs, the optimal allocation is:

$ N_"opt" = G_N dot C^(a) , quad D_"opt" = G_D dot C^(b) $

with $a approx b approx 0.5$, meaning *parameters and tokens should scale equally*. The Chinchilla rule of thumb: train on approximately $20 dot N$ tokens.

The predicted loss as a function of $N$ and $D$:

$ hat(L)(N, D) = E + A / N^alpha + B / D^beta $

with fitted constants $E = 1.69$, $A = 406.4$, $B = 410.7$, $alpha = 0.34$, $beta = 0.28$ (Hoffmann et al., 2022, Table 2, Approach 3).

=== Concrete Budget Table

Compute $C$ is measured in FLOPs. For a dense transformer, $C approx 6 N D$ (forward + backward, ignoring attention and activations for simplicity).

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  [*Compute budget C (FLOPs)*], [*Optimal N*], [*Optimal D (tokens)*], [*GPU-days (A100, 312 TF)*], [*GPU-days (H100, 989 TF)*], [*Example*],
  [$10^(21)$], [$1.3$ B],   [$26$ B],   [$2$],       [$0.6$],    [Small experiment],
  [$10^(22)$], [$4$ B],    [$82$ B],   [$20$],      [$6$],      [—],
  [$10^(23)$], [$12$ B],   [$260$ B],  [$200$],     [$63$],     [LLaMA 1 13B (approx)],
  [$10^(24)$], [$37$ B],   [$820$ B],  [$2{,}000$], [$630$],    [Chinchilla 70B],
  [$3 times 10^(24)$], [$67$ B], [$1.5$ T],  [$6{,}000$], [$1{,}900$], [LLaMA 2 70B (approx)],
  [$10^(25)$], [$120$ B],  [$2.6$ T],  [$20{,}000$],[$6{,}300$], [LLaMA 3 70B (approx)],
)

_Note:_ A100 numbers (312 TFLOPS BF16 dense) are kept for historical reference; H100 figures use 989 TFLOPS BF16 dense (1979 TFLOPS with 2:4 sparsity, roughly halving the H100 column when applicable). Real wall-clock days are 1.5–2$times$ these ideal-MFU numbers.

_Note:_ LLaMA 3 8B trains on 15T tokens, deliberately far past Chinchilla-optimal for a small model, optimizing _inference_ cost at a fixed serving budget rather than training cost. This "overtrain small models" strategy is practical when models are deployed at scale.

```python
# Chinchilla optimal allocation given compute budget
def chinchilla_optimal(C_flops: float,
                       G_N: float = 0.037,
                       G_D: float = 0.82) -> tuple[float, float]:
    """
    Returns (N_opt, D_opt) following Hoffmann et al. 2022.
    C_flops: total compute in FLOPs (use 6*N*D approximation).
    G_N, G_D calibrated so N_opt, D_opt reproduce the budget table
    (e.g. C=1e24 -> ~37B params, ~820B tokens).
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

Training in FP32 throughout uses 4 bytes per parameter. For a 7B model that is already 28 GB just for weights (before gradients (28 GB) and Adam states (56 GB)). Mixed precision training (Micikevicius et al., 2018) dramatically reduces memory while preserving convergence:

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

This is why a 7B model requires at least two 80 GB A100s for pretraining with standard mixed precision, or sharding via FSDP/ZeRO-3.

== Gradient Checkpointing

=== The Memory–Compute Tradeoff

During the forward pass, PyTorch stores all intermediate activations needed for backpropagation. For a transformer with $L$ layers, sequence length $S$, batch size $B$, and hidden dimension $d$, activation memory scales as $O(L B S d)$. At LLaMA 3 8B scale ($L=32$, $d=4096$, $S=8192$, $B=4$), this is roughly:

$ 32 times 4 times 8192 times 4096 times 2 " bytes" approx 8 " GB " $

just for the residual stream, before attention matrices. Full activation memory for a forward pass is $O(B S d_"ffn") times L approx 60$–$80$ GB.

*Gradient checkpointing* (Chen et al., 2016) reduces the per-activation layer-count factor from $O(L)$ to $O(sqrt(L))$ by storing only a subset of layer outputs (the _checkpoints_) and recomputing the others during the backward pass. The full activation memory therefore scales as $O(sqrt(L) dot B dot S dot d)$, where the $sqrt(L)$ is the layer-axis savings, while batch size $B$, sequence length $S$, and hidden width $d$ still enter linearly. The tradeoff: recomputation adds approximately 33% to total FLOPs.

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


== Further Reading

Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models." arXiv:2001.08361. (The original power-law relationships between compute, data, and loss.)

Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." NeurIPS. (Chinchilla; corrected the compute-optimal token-to-parameter ratio.)

Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS. (GPT-3 pretraining setup and data mixture.)

Touvron, H., et al. (2023). "LLaMA: Open and Efficient Foundation Language Models." arXiv:2302.13971. (Open recipe with detailed data and hyperparameter reporting.)

Shoeybi, M., et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." arXiv:1909.08053. (Tensor parallelism for large-scale pretraining.)

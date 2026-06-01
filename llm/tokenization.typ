= Tokenization

Tokenization sits between raw text and the model: it determines what a "token" *is*, and therefore what the model can natively represent. Choices made here propagate into vocabulary size (which controls the embedding and output projection matrices, often the largest single parameter blocks), context efficiency (tokens-per-byte), multilingual fairness, and downstream behaviors as concrete as arithmetic and code formatting. This chapter covers the major algorithms (BPE, WordPiece, Unigram/SentencePiece), the byte-level variant that dominates modern LLMs, the practical engineering of `tiktoken`-style fast tokenizers, and the failure modes that arise in multilingual and code settings.

*See also:* _Pretraining_ (vocab is fixed before training), _Data Curation_ (the tokenizer is fit on the curated corpus), _Inference Optimization_ (tokens-per-second is the throughput unit).

== Why Subwords

Two extremes — character-level and word-level — both fail for LLMs.

*Word-level* needs a vocabulary that grows with the corpus; out-of-vocabulary (OOV) words must be replaced with `<UNK>`, destroying information. Compound languages (German, Finnish) and agglutinative ones (Turkish, Hungarian) explode the vocab. Punctuation, casing, and morphology multiply variants of every stem.

*Character-level* has a tiny vocab but very long sequences (English is ~4 chars/word) and forces the model to learn morphology and word boundaries from scratch. Quadratic attention cost makes this impractical at scale.

*Subword* tokenization splits rare or unseen words into pieces that the model can recombine. A frequent word like `the` stays whole; a rare word like `Tokenization` becomes `Token` + `ization`. The vocab is bounded (typically 30k–256k); OOVs do not exist if every byte is reachable.

== Byte-Pair Encoding (BPE)

BPE was introduced as a compression algorithm (Gage 1994) and adapted to NMT by Sennrich, Haddow, Birch (2016). The training procedure is greedy and frequency-driven.

=== Training

```python
from collections import Counter

def get_pair_counts(splits: list[list[str]]) -> Counter:
    pairs = Counter()
    for sym_seq in splits:
        for a, b in zip(sym_seq, sym_seq[1:]):
            pairs[(a, b)] += 1
    return pairs

def merge_pair(pair: tuple[str, str], splits: list[list[str]]) -> list[list[str]]:
    a, b = pair
    out = []
    for seq in splits:
        new_seq, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                new_seq.append(a + b)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        out.append(new_seq)
    return out

def train_bpe(corpus: list[str], num_merges: int):
    # Pre-tokenize on whitespace; mark word boundaries with </w>.
    splits = [list(w) + ["</w>"] for w in corpus]
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pair_counts = get_pair_counts(splits)
        if not pair_counts:
            break
        best = max(pair_counts, key=pair_counts.get)
        splits = merge_pair(best, splits)
        merges.append(best)
    return merges
```

Each merge adds one token to the vocabulary. Training cost is $O(M times N)$ where $M$ is the merge count and $N$ is the corpus size; the modern `tokenizers` crate uses priority queues and incremental pair counting to bring this close to $O(N log V)$.

=== Encoding

Encoding applies merges in the order they were learned. For each pre-tokenized word, repeatedly find the highest-priority adjacent pair present in the merge table and merge it.

```python
def encode_word(word: str, merge_rank: dict[tuple[str, str], int]) -> list[str]:
    syms = list(word) + ["</w>"]
    while True:
        pairs = [(syms[i], syms[i+1]) for i in range(len(syms)-1)]
        ranked = [(merge_rank.get(p, float("inf")), i, p) for i, p in enumerate(pairs)]
        rank, idx, pair = min(ranked, key=lambda t: (t[0], t[1]))
        if rank == float("inf"):
            break
        syms = syms[:idx] + [pair[0] + pair[1]] + syms[idx+2:]
    return syms
```

*Determinism note:* encoding is order-sensitive. Two equivalent surface strings ("don't" vs "don 't") may tokenize differently, so pre-tokenization (regex split) is part of the contract.

== Byte-Level BPE

GPT-2 (Radford et al. 2019) introduced *byte-level* BPE: the base alphabet is the 256 bytes, not Unicode characters. Every UTF-8 string is representable; there is no `<UNK>`.

The trick is a reversible bijection between bytes and printable Unicode codepoints so the merge table can be stored as text. Bytes 0–32, 127, and 128–160 are mapped to private-use codepoints; the rest pass through. This keeps merges human-readable while preserving roundtrip safety.

```python
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte → printable codepoint map."""
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}
```

Byte-level BPE dominates modern LLMs (GPT-3/4, LLaMA, Mistral, Qwen) because it sidesteps Unicode normalization headaches and handles arbitrary binary content (emoji, code, base64).

== WordPiece

WordPiece (Schuster–Nakajima 2012, popularized by BERT) is structurally similar to BPE but scores candidate merges by likelihood under a unigram language model rather than raw frequency:

$ "score"(a, b) = "count"(a b) / ("count"(a) times "count"(b)) $

This penalizes merging two already-common tokens (which would not gain much information) and prefers merging tokens that genuinely co-occur. WordPiece also marks word-internal subwords with `##` (e.g., `play`, `##ing`), which makes detokenization unambiguous.

== Unigram and SentencePiece

Kudo (2018) proposed *Unigram* tokenization: model the probability of a tokenization as a product of token unigram probabilities, and *prune* an initially large vocabulary by EM.

=== Algorithm Sketch

1. Seed with a large vocabulary (e.g., all substrings up to length $k$).
2. For each sentence, find the Viterbi best segmentation under current unigram probabilities.
3. Re-estimate unigram probabilities from the segmentations (M-step).
4. Compute the *loss contribution* of each token (how much overall likelihood drops if it is removed).
5. Drop the bottom $p$% by contribution; repeat until the target vocab size.

=== SentencePiece

SentencePiece (Kudo–Richardson 2018) is the *implementation* — language-agnostic, treats input as a raw byte/char stream (no whitespace pre-tokenization), and represents whitespace as a normal token `▁` (U+2581). This means the tokenizer is reversible without external rules and works uniformly on Japanese, Chinese, and Khmer where whitespace is absent or different.

SentencePiece can run either BPE or Unigram under the hood; LLaMA 1/2 use SentencePiece-BPE, while T5, ALBERT, and XLNet use Unigram.

== tiktoken: Engineering a Fast Tokenizer

OpenAI's `tiktoken` is a representative production tokenizer. Three engineering choices matter:

1. *Pre-tokenization regex.* A Perl-compatible regex splits text into chunks before BPE applies. The `cl100k_base` (GPT-3.5/4) pattern is roughly:

```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

   This isolates contractions, numbers (≤3 digits), whitespace-leading punctuation, etc. The regex pre-tokenization prevents merges from crossing semantic boundaries (so `you're` does not merge with the next word).

2. *Rank-based BPE.* Instead of merge rules, store a `{token_bytes: rank}` map. Encoding is then "repeatedly merge the pair with the lowest rank present." Equivalent to ordered merges but faster because the loop is a single hashmap lookup per pair.

3. *Implementation in Rust* (via `pyo3`). Tokenizers are CPU-bound on the regex and the merge loop; Rust gives 10–50× speedups over pure Python.

== Multilingual and Code Tokenization

=== Multilingual Fairness

A tokenizer fit on English-heavy data assigns fewer bytes-per-token (BPT) to English than to, say, Burmese. Petrov et al. (2023) measured *premium* — the cost ratio relative to English — exceeding 15× for some low-resource languages on GPT-3.5. Consequences:

- Higher API cost and latency for non-English users.
- Lower effective context window in tokens-of-meaning.
- Degraded benchmark scores from longer sequences with fewer attention "slots".

Modern multilingual tokenizers (LLaMA 3 with 128k vocab, Aya-23, Qwen2 with 152k) substantially close the gap by training on more diverse corpora and growing the vocab.

=== Code

Code has long identifiers, deep indentation, and idiosyncratic punctuation (`->`, `::`, `=>`). Generic tokenizers waste tokens on whitespace runs and split common idents. Code-focused tokenizers (StarCoder, CodeLlama, DeepSeek-Coder) extend the vocab with whitespace runs (`▁▁▁▁` = four spaces) and frequent code n-grams.

== Failure Modes

=== Glitch Tokens

Tokens that appear in the vocab but rarely in training data become "glitches": the model has not learned a coherent representation. The infamous `SolidGoldMagikarp` (Rumbelow–Watkins 2023) was a Reddit username present in the GPT-2 BPE training corpus but absent from the actual training mix. Asking GPT-3 to repeat it produced bizarre, unrelated outputs. Modern tokenizers audit the vocab against the training distribution and remove tokens with frequency below a threshold.

=== Arithmetic

If `1234` tokenizes as `123` + `4` but `1235` tokenizes as `12` + `35`, the model sees different surface forms for adjacent numbers and struggles with carrying. LLaMA 3, Qwen2, and GPT-4 force per-digit tokenization (`1`, `2`, `3`, `4`) by adding the ten digits as standalone tokens and forbidding number merges in the pre-tokenization regex. The effect on arithmetic accuracy at fixed model size is large (Bostrom–Durrett 2020; Nogueira et al. 2021).

=== Trailing-Space Bias

`"hello"` and `" hello"` are different tokens. Sampling code that strips whitespace before computing logprobs misaligns probabilities with the model's tokenizer. The fix is *prompt boundary tokenization* — always feed the prompt through the tokenizer once and never modify token strings.

== Vocab Size Tradeoffs

Vocabulary size $V$ affects:

- *Embedding and output matrices:* size $V times d_"model"$ each. At $d=4096$, doubling $V$ from 32k to 64k adds 0.5 GiB (bf16) per matrix.
- *Tokens per byte (TPB):* larger $V$ → fewer tokens for the same text → more useful context per FLOP at inference.
- *Sample efficiency during training:* larger $V$ means each token sees fewer training examples; for a fixed dataset, very large vocabs degrade convergence.
- *Sampling temperature semantics:* with very large $V$, top-$k$ truncation behaves differently because the tail mass distribution shifts.

Empirical sweet spots: 32k–50k for early models (GPT-2, T5), 100k+ for multilingual (LLaMA 3 = 128k, Qwen2 = 152k). The Chinchilla scaling regime (compute-optimal) prefers vocab $V approx 0.27 times N^(0.5)$ where $N$ is parameter count (Tao et al. 2024).

== Practical Workflow

```python
# Training a SentencePiece BPE tokenizer (production-grade).
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="tok",
    vocab_size=32_000,
    model_type="bpe",
    character_coverage=0.9995,        # cover 99.95% of training chars
    byte_fallback=True,                # emit byte tokens for unseen chars
    split_digits=True,                 # per-digit numbers
    normalization_rule_name="nmt_nfkc",
    user_defined_symbols=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    train_extremely_large_corpus=True,
)
```

Audit the trained tokenizer:

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file="tok.model")

# Tokens-per-byte distribution by language
samples = {"en": "...", "zh": "...", "hi": "..."}
for lang, text in samples.items():
    tpb = len(sp.encode(text)) / len(text.encode("utf-8"))
    print(f"{lang}: {tpb:.3f} tokens/byte")

# Find low-frequency vocab entries
counts = Counter()
for line in open("corpus.txt"):
    counts.update(sp.encode(line))
rare = [(sp.id_to_piece(i), c) for i, c in counts.items() if c < 10]
print(f"{len(rare)} rare tokens (potential glitches)")
```

== Further Reading

Sennrich, R., Haddow, B., Birch, A. (2016). "Neural Machine Translation of Rare Words with Subword Units." ACL.

Kudo, T. (2018). "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates." ACL.

Kudo, T., Richardson, J. (2018). "SentencePiece: A Simple and Language Independent Subword Tokenizer." EMNLP.

Radford, A. et al. (2019). "Language Models are Unsupervised Multitask Learners." (GPT-2 technical report.)

Petrov, A. et al. (2023). "Language Model Tokenizers Introduce Unfairness Between Languages." NeurIPS.

Rumbelow, J., Watkins, M. (2023). "SolidGoldMagikarp (plus, prompt generation)." LessWrong post.

Bostrom, K., Durrett, G. (2020). "Byte Pair Encoding is Suboptimal for Language Model Pretraining." Findings of EMNLP.

Tao, T. et al. (2024). "Scaling Laws with Vocabulary." arXiv:2407.13623.

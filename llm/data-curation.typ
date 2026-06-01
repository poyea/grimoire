= Data Curation

A frontier LLM is trained on $10^(13)$ tokens. At that scale, the data pipeline determines model quality more than the architecture: small differences in deduplication, filtering, and decontamination shift downstream benchmarks by several points. This chapter walks the modern web-scale pipeline — extraction from Common Crawl, language identification, deduplication (exact, MinHash near-dup, semantic), quality filtering (heuristics, classifiers, perplexity), decontamination against eval sets, and the publicly documented pipelines that produced C4, RefinedWeb, RedPajama, Dolma, and FineWeb.

*See also:* _Pretraining_ (consumes this output), _Tokenization_ (fit on the curated corpus), _Evaluation_ (contamination invalidates benchmarks).

== The Pipeline Shape

A canonical pipeline:

```
Common Crawl (raw WARC, ~petabytes)
  → text extraction (trafilatura / resiliparse)
  → language ID (cld3 / fasttext-lid)
  → URL / domain filters (blocklists, adult, malware)
  → exact dedup (line, paragraph, document)
  → near-dup dedup (MinHash LSH)
  → quality filter (heuristics → classifier → perplexity)
  → PII redaction
  → decontamination vs. eval sets
  → tokenization + shuffling
  → training shards
```

Each stage removes 30–90% of input. The funnel collapses Common Crawl's ~100 trillion raw tokens into a 1–15 trillion-token training mix.

== Text Extraction

WARC files contain raw HTTP responses including HTML. Naive `<p>`-stripping yields menus, footers, cookie banners, and JavaScript. Modern extractors (`trafilatura`, `resiliparse`, `jusText`) score DOM nodes by text-to-tag ratio, link density, and visual position to find the *main article body*.

```python
import trafilatura

def extract(html: str) -> str | None:
    return trafilatura.extract(
        html,
        favor_precision=True,         # prefer dropping boilerplate
        include_comments=False,
        include_tables=False,
    )
```

*Failure mode:* extractors trained on news articles can miss reference-style content (Wikipedia tables, code snippets). Per-domain rules and HTML→Markdown converters (e.g., `readability-lxml` for academic mirrors) supplement the general extractor.

== Language Identification

After extraction, classify language per document. `cld3` (Google) and `fasttext-lid-176` are the workhorses. Below-threshold confidence (≤0.65) means *drop* — mixed-language and gibberish pages survive otherwise. For multilingual training, route by language so per-language quality filters can be tuned independently.

== Deduplication

Web data is heavily redundant: news syndication, mirrors, SEO spam, license templates. Lee et al. (2022) showed that exact deduplication alone reduces training loss and improves downstream performance. Below near-dup also matters: small wording changes do not create new information.

=== Exact

Hash documents (or 100-line shingles) with SHA-256. Two passes:

1. *Document-level:* drop any document whose full-text hash collides with one already seen.
2. *Sub-document:* drop lines, sentences, or paragraphs that appear in many documents (legal boilerplate, navigation).

```python
import hashlib
from collections import defaultdict

def line_dedup(docs: list[str], threshold: int = 100) -> list[str]:
    line_count = defaultdict(int)
    for d in docs:
        for line in d.splitlines():
            line_count[hashlib.sha256(line.strip().encode()).digest()] += 1
    return [
        "\n".join(
            line for line in d.splitlines()
            if line_count[hashlib.sha256(line.strip().encode()).digest()] < threshold
        )
        for d in docs
    ]
```

=== MinHash + LSH (Near-Duplicate)

For documents that differ by a few words, exact hashing fails. *MinHash* (Broder 1997) estimates Jaccard similarity over shingle sets in constant space per document.

Algorithm:

1. Shingle: split each document into overlapping $n$-grams (typically $n=5$ words or 9 chars).
2. For each of $k$ hash functions $h_i$, take $\min_(s in S) h_i(s)$. This produces a *signature* of $k$ integers.
3. Estimate $J(A, B) approx |"signature"(A) inter "signature"(B)| / k$.

For datasets of $10^9$ documents, comparing all pairs is impossible. *Locality-sensitive hashing (LSH)* bands the signature into $b$ bands of $r$ rows each ($k = b times r$). Two documents become candidates if any band matches. Probability of being a candidate at Jaccard $j$:

$ P(j) = 1 - (1 - j^r)^b $

Tune $b, r$ for a target *similarity threshold* $j^*$ (commonly 0.8 for near-dup). The S-curve transitions sharply, so $(b, r) = (20, 9)$ for $k = 180$ gives a sharp ~0.8 threshold.

```python
from datasketch import MinHash, MinHashLSH

def shingle(text: str, n: int = 5) -> set[str]:
    toks = text.split()
    return {" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)}

lsh = MinHashLSH(threshold=0.8, num_perm=128)
for doc_id, text in corpus:
    mh = MinHash(num_perm=128)
    for sh in shingle(text):
        mh.update(sh.encode())
    if lsh.query(mh):           # any near-duplicate already inserted
        continue
    lsh.insert(doc_id, mh)
```

At trillion-document scale, this is run in a distributed Spark/Dataflow job; the standard recipe is the *SlimPajama* deduplication pipeline (TogetherAI 2023), which dedupes RedPajama from 1.2T to 627B tokens.

=== Semantic Deduplication

Recent pipelines (D4 — Tirumala et al. 2023; SemDeDup — Abbas et al. 2023) cluster documents in *embedding space* and remove cluster-internal duplicates. This catches paraphrases and translations that MinHash misses. Cost is higher (one embedding pass over the entire corpus) and risk is also higher: aggressive semantic dedup can drop legitimate near-duplicates (multiple coverage of the same news event) and harm factual diversity.

== Quality Filtering

=== Heuristic Filters

Gopher (Rae et al. 2021) introduced a now-standard set of cheap signals:

- Mean line length 10–150 chars
- Fraction of lines starting with bullets ≤ 0.9
- Fraction of lines ending with ellipsis ≤ 0.3 (filters truncated SEO content)
- Symbol-to-word ratio ≤ 0.1
- Fraction of "stopwords" (the, of, to, ...) ≥ 0.06 (filters word-soup spam)
- Repeated-line ratio ≤ 0.3

```python
STOPWORDS = {"the", "be", "to", "of", "and", "a", "in", "that", "have", "it"}

def gopher_quality(text: str) -> bool:
    lines = text.splitlines()
    if not lines: return False
    words = text.split()
    if len(words) < 50 or len(words) > 100_000: return False
    if sum(1 for w in words if w.lower() in STOPWORDS) / len(words) < 0.06:
        return False
    if sum(1 for l in lines if l.startswith(("*", "-", "•"))) / len(lines) > 0.9:
        return False
    if sum(1 for l in lines if l.endswith("...")) / len(lines) > 0.3:
        return False
    return True
```

These filters run on every extracted document; they cut another 30–50% of survivors from extraction.

=== Classifier Filters

A binary classifier — typically a small fastText model — trained to distinguish "good" reference text (Wikipedia, books, curated web) from raw Common Crawl. Score every document; keep the top fraction. GPT-3 used this; Llama 1/2 used a CCNet-style classifier with Wikipedia as positive examples.

```python
import fasttext
clf = fasttext.train_supervised(
    input="quality_train.txt",       # __label__good / __label__bad lines
    epoch=5, lr=0.1, wordNgrams=2, dim=100,
)
# Inference
def keep(text: str, threshold: float = 0.5) -> bool:
    label, prob = clf.predict(text.replace("\n", " ")[:1000])
    return label[0] == "__label__good" and prob[0] >= threshold
```

*Pitfall:* the classifier inherits the bias of its positive set. Wikipedia-as-positive disfavors informal writing, dialogue, and code. Multi-classifier ensembles (one per domain) mitigate this.

=== Perplexity Filtering

CCNet (Wenzek et al. 2020) ranks documents by perplexity under a 5-gram language model trained on Wikipedia, partitioned into head/middle/tail buckets. The middle bucket is empirically best — head is too narrow (over-formal), tail is junk. Modern pipelines often use a small transformer LM (~1B params) for this scoring, which detects gibberish that n-gram models miss.

== PII and Safety

Personally identifiable information must be redacted *before* training. The widely-deployed approach combines:

- Regex for structured PII (emails, phone numbers, IP addresses, SSNs, credit cards).
- NER models for names and addresses where context-sensitivity matters.
- Hashing instead of removal for stable test-set construction.

```python
import re
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s-]?\d{3}[\s-]?\d{4}\b")

def redact(text: str) -> str:
    text = EMAIL.sub("[EMAIL]", text)
    text = PHONE.sub("[PHONE]", text)
    return text
```

Beyond PII, copyrighted-content opt-outs (`ai.robots.txt`, the EU AI Act's transparency requirements) increasingly drive *domain-level* exclusions before extraction. Toxic content filters (`detoxify`, `Perspective API`) are run on a sample and may filter or *reweight* documents.

== Decontamination

Evaluation benchmarks (MMLU, HumanEval, GSM8K, BBH) exist in their entirety on the public web. If a model has seen the test set, scores are meaningless. Decontamination removes training documents that overlap eval data.

=== n-gram Overlap

Standard recipe: build an index of all $n$-grams ($n = 13$ for natural-language benchmarks, $n = 50$ chars for code) appearing in eval prompts and answers. For each training document, drop it if it contains any eval $n$-gram. Llama 3 used $n = 8$; GPT-3 used $n = 13$.

```python
def build_ngram_set(eval_docs: list[str], n: int = 13) -> set[str]:
    ngrams = set()
    for d in eval_docs:
        toks = d.split()
        ngrams.update(" ".join(toks[i:i+n]) for i in range(len(toks)-n+1))
    return ngrams

EVAL_NGRAMS = build_ngram_set(load_all_eval_prompts())

def contaminated(text: str, n: int = 13) -> bool:
    toks = text.split()
    return any(" ".join(toks[i:i+n]) in EVAL_NGRAMS
               for i in range(len(toks)-n+1))
```

=== Limitations

$n$-gram matching misses *paraphrased* contamination (a forum post that solves the GSM8K problem in different words) and *translated* contamination (the eval question in another language). The current state of the art uses LLM-based contamination detection (Yang et al. 2023; Sainz et al. 2023) at the cost of expensive scoring.

== Mixing and Reweighting

Once filtered corpora exist per source (web, books, code, papers, math, multilingual), the final mix is determined by *upsampling rates*. Llama 3 documents these explicitly: web ~50%, code ~20%, math ~5%, multilingual ~15%, books/papers ~10%. The mix is set by experiments at smaller scale (DoReMi — Xie et al. 2023; the "data scaling laws" of Pouget et al. 2024) and held fixed during training, or annealed (more high-quality data near the end).

Per-source upsampling controls *exposure*: a 5B-token math corpus seen 5× contributes the same gradient signal as a 25B-token web corpus seen once. Upsampling rates beyond ~5× start to hurt (memorization without generalization).

== Public Pipelines (Snapshot)

| Pipeline | Tokens | Notes |
|---|---|---|
| C4 (Raffel 2020) | 156B | Common Crawl, single dump, English, language ID + Gopher-style heuristics. |
| The Pile (Gao 2020) | 825GB | 22 curated subsets; popular for early open models. |
| RedPajama-1T | 1.2T | LLaMA 1 reproduction; Common Crawl + GitHub + Wikipedia + Books + ArXiv + StackExchange. |
| SlimPajama | 627B | RedPajama after MinHash near-dup; ~50% reduction. |
| RefinedWeb (Penedo 2023) | 600B | Web-only, very aggressive heuristic + MinHash dedup; powered Falcon. |
| Dolma (Soldaini 2024) | 3T | Open, documented; OLMo training corpus. |
| FineWeb (Penedo 2024) | 15T | Common Crawl with classifier-based quality filter; matches Llama 3 quality on open data. |
| FineWeb-Edu (2024) | 1.3T | FineWeb scored by an educational-content classifier; punches above its weight at smaller scales. |

== Workflow Notes

- *Idempotency:* every stage writes to a content-addressed store. Re-running with new filters re-uses cached extraction.
- *Sampling first:* never run a new filter against the full pipeline before sampling 1B tokens and *reading* the survivors and the rejected. Heuristic filters are easy to write wrong and silently strip the highest-value content (math papers fail many "natural prose" checks).
- *Decontamination is part of *every* release:* including the eval mix in your decontamination scan. Models inherit contamination from the *base* corpus; SFT/DPO data adds more.

== Further Reading

Lee, K. et al. (2022). "Deduplicating Training Data Makes Language Models Better." ACL.

Wenzek, G. et al. (2020). "CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data." LREC.

Rae, J. et al. (2021). "Scaling Language Models: Methods, Analysis & Insights from Training Gopher." DeepMind technical report.

Penedo, G. et al. (2023). "The RefinedWeb Dataset for Falcon LLM." NeurIPS.

Soldaini, L. et al. (2024). "Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research."

Penedo, G. et al. (2024). "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale." arXiv:2406.17557.

Xie, S. M. et al. (2023). "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining." NeurIPS.

Tirumala, K. et al. (2023). "D4: Improving LLM Pretraining via Document De-Duplication and Diversification." NeurIPS.

Sainz, O. et al. (2023). "NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each Benchmark." Findings of EMNLP.

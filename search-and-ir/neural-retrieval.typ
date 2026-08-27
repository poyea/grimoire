#import "../template.typ": xref

= Neural Retrieval <neural-retrieval>

Neural retrieval applies pretrained language models to the matching problem itself, not just to reranking features. The field reorganized around BERT (2019–2021) into a small set of architectures distinguished by _where_ the query–document interaction happens: cross-encoders (full interaction, expensive, rerank-only), dense bi-encoders (no interaction until a single dot product, indexable), late interaction (per-token dot products, in between), and learned sparse models (neural term weighting on a classical inverted index). This chapter covers each, the training recipes that make or break them, and the hybrid systems production actually ships.

*See also:* #xref("search-and-ir", "ranking-classical", label: "Ranking: Classical Models") (BM25, the baseline and hybrid partner), #xref("search-and-ir", "vector-search", label: "Vector Search") (the ANN indexes dense retrieval needs), #xref("search-and-ir", "learning-to-rank", label: "Learning to Rank") (where rerankers slot into the cascade), #xref("search-and-ir", "rag-and-search-systems", label: "RAG and Search Systems") (retrieval as an LLM component), #xref("search-and-ir", "evaluation", label: "Evaluation") (BEIR and MS MARCO).

== Vocabulary Mismatch and the Case for Learning

Classical retrieval fails when query and document use different words for the same thing ("how to fix a flat" vs. "puncture repair"). Pre-neural fixes — synonym expansion, pseudo-relevance feedback (RM3), translation models — help at the margin. Dense representations attack the problem directly: embed text into a vector space where semantic similarity is geometric proximity. The earliest viable attempts (LSI via SVD, then DSSM at Microsoft, 2013) underperformed BM25; pretrained transformers changed the outcome.

== Cross-Encoders: Reranking with Full Interaction

Nogueira & Cho (2019) fine-tuned BERT to classify relevance on the concatenated pair `[CLS] query [SEP] document [SEP]`, scoring with the `[CLS]` output. Every query token attends to every document token, so the model sees exact matches, paraphrase, and context jointly. On MS MARCO passage ranking this lifted MRR\@10 from ~0.19 (BM25) to ~0.36 in one step — the largest single jump in the benchmark's history.

The cost: nothing can be precomputed, so scoring is a full transformer forward pass per pair. Cross-encoders therefore rerank a candidate set (top 100–1000 from a cheaper stage), at tens of milliseconds per batch on a GPU. They remain the quality ceiling among practical architectures and supply distillation targets and labels for everything below. MonoT5 (Nogueira et al., 2020) recasts the same idea generatively ("Query: ... Document: ... Relevant: true/false"), and modern LLM rerankers (RankGPT-style listwise prompting, fine-tuned rerankers from Cohere and Voyage) are its descendants.

== Dense Retrieval: Bi-Encoders

A bi-encoder embeds query and document _independently_ into $RR^d$ (typically $d in (384, 1024)$) and scores by dot product or cosine:

$ s(q, d) = E_Q (q) dot E_D (d) $

Document vectors are precomputed offline and indexed; at query time, one encoder pass plus an approximate nearest-neighbor search (see #xref("search-and-ir", "vector-search", label: "Vector Search")) retrieves from millions of documents in milliseconds. DPR (Karpukhin et al., 2020) established the recipe for open-domain QA — BERT-base dual encoders, in-batch negatives, one "hard" BM25 negative per query — and beat BM25 by 9–19 points top-20 accuracy on Natural Questions.

=== Training Is the Hard Part

The architecture is trivial; the negatives are everything. The contrastive loss for query $q$ with positive $d^+$ and negatives $d_1^-, ..., d_n^-$:

$ L = -log (e^(s(q, d^+) \/ tau)) / (e^(s(q, d^+) \/ tau) + sum_(j=1)^n e^(s(q, d_j^-) \/ tau)) $

- *In-batch negatives*: reuse other queries' positives as negatives — free, but mostly easy.
- *Hard negatives*: top BM25 or model-retrieved non-relevant documents. Mining negatives with the model being trained risks false negatives (unjudged relevant documents); ANCE (Xiong et al., 2021) refreshes negatives from the evolving index, RocketQA (Qu et al., 2021) denoises them with a cross-encoder, and large batches (thousands) plus cross-encoder distillation define the modern recipe (e.g., SentenceTransformers' MarginMSE, TAS-B, GTR, E5, BGE).
- *Distillation*: train the bi-encoder to match cross-encoder score margins; consistently worth 2–4 nDCG points.

=== The Out-of-Domain Problem

BEIR (Thakur et al., 2021) — 18 retrieval tasks, zero-shot — showed early dense models _losing_ to BM25 out of domain: a single vector blurs rare entities, identifiers, and numbers that lexical match nails. The gap has since narrowed via better pretraining (contrastive pretraining on web pairs, e.g., Contriever, E5) and instruction-tuned embedders, but exact-match brittleness is structural, which motivates hybrids.

== Late Interaction: ColBERT

ColBERT (Khattab & Zaharia, 2020) keeps per-token embeddings for the document and scores by *MaxSim*: each query token takes the maximum similarity over document tokens, summed:

$ s(q, d) = sum_(i in q) max_(j in d) E(q_i) dot E(d_j) $

This preserves token-level matching (so it generalizes far better on BEIR than single-vector models) while still precomputing document representations. Costs: index size — dozens of vectors per passage — which ColBERTv2 (Santhanam et al., 2022) attacks with residual compression against centroids, cutting storage 6–10$times$, and the PLAID engine accelerates with centroid-based candidate pruning.

== Learned Sparse Retrieval: SPLADE

Instead of dense vectors, predict a sparse weight for every vocabulary term — including terms _not in the text_ (expansion). SPLADE (Formal et al., 2021) uses BERT's masked-language-model head to produce per-term weights, with a log-saturation activation and FLOPS regularization to keep vectors sparse. The output is a weighted bag of words served by a *standard inverted index*, inheriting decades of efficiency work (WAND pruning, mature operations) while learning term importance and expansion. Earlier points on this line: DeepCT (learned tf replacement) and doc2query/docT5query (Nogueira & Lin, 2019), which expands documents with generated queries at index time — a trick that requires no new infrastructure at all. uniCOIL and TILDE are related contemporaries. Learned sparse models are roughly competitive with dense ones in domain and notably robust out of domain.

== Hybrid Retrieval

Lexical and dense errors are decorrelated, so combining them is the most reliable win in modern IR. Two standard fusions:

- *Score fusion*: $s = alpha dot "norm"("BM25") + (1 - alpha) dot "norm"("dense")$, with min–max or z-score normalization per query; $alpha$ tuned on a dev set.
- *Reciprocal rank fusion* (Cormack et al., 2009): ignore scores, sum $1 \/ (k + r_i)$ over each system's rank $r_i$ (typically $k = 60$). Calibration-free and hard to beat; the default in Elasticsearch's and OpenSearch's hybrid query support.

A typical production stack: BM25 and dense ANN retrieval in parallel, RRF fusion, cross-encoder rerank of the top 100. Each stage is independently swappable and measurable.

== Pitfalls

- *Benchmark overfitting*: MS MARCO's shallow, sparse labels (~1 judged positive per query) reward models trained on it far beyond their real-world advantage; always check BEIR-style out-of-domain numbers.
- *False negatives in training*: random or BM25 negatives often contain unlabeled positives; aggressive hard-negative mining without denoising _hurts_.
- *Stale embeddings*: re-encoding the corpus after every model update is an operational cost classical indexes never had; plan for versioned indexes and dual-serving during migration.
- *Long documents*: most encoders see 512 tokens; passage-level chunking with score aggregation (MaxP) is the workaround, and chunking policy materially affects quality (see #xref("search-and-ir", "rag-and-search-systems", label: "RAG and Search Systems")).
- *Tokenization blind spots*: product codes, version strings, and rare names fragment into subwords and lose identity; keep a lexical leg in the hybrid for these.

== Further Reading

- Karpukhin, V. et al. (2020). Dense passage retrieval for open-domain question answering. _EMNLP_.
- Nogueira, R., & Cho, K. (2019). Passage re-ranking with BERT. _arXiv:1901.04085_.
- Khattab, O., & Zaharia, M. (2020). ColBERT: efficient and effective passage search via contextualized late interaction over BERT. _SIGIR_.
- Formal, T., Piwowarski, B., & Clinchant, S. (2021). SPLADE: sparse lexical and expansion model for first stage ranking. _SIGIR_.
- Xiong, L. et al. (2021). Approximate nearest neighbor negative contrastive learning for dense text retrieval. _ICLR_. (ANCE)
- Thakur, N. et al. (2021). BEIR: a heterogeneous benchmark for zero-shot evaluation of information retrieval models. _NeurIPS Datasets and Benchmarks_.
- Lin, J., Nogueira, R., & Yates, A. (2021). _Pretrained Transformers for Text Ranking: BERT and Beyond_. Morgan & Claypool.

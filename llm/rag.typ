= Retrieval-Augmented Generation

RAG injects information retrieved from an external corpus into the LLM's prompt at query time. The model gains access to up-to-date, domain-specific, or proprietary knowledge without retraining, and citations become attributable. This chapter covers the full pipeline — chunking, embedding, indexing, hybrid retrieval, reranking, fusion strategies — plus the evaluation methodology and the design tradeoffs versus long context and fine-tuning.

*See also:* _Long Context_ (a complementary approach), _Vector and Similarity Search (Database volume)_ (HNSW, IVF, product quantization), _Agents and Tool Use_ (retrieval is a tool the agent can call).

== When RAG, When Long Context, When Fine-Tuning

| Concern | RAG | Long context | Fine-tuning |
|---|---|---|---|
| Update frequency | live | requires re-prefill | retrain |
| Citation / attribution | natural | hard | none |
| Cost per query | retrieval + small prompt | huge prefill | model cost |
| Knowledge size | unlimited | window-bounded | weights-bounded |
| Reasoning across whole corpus | weak | strong | strong (if memorized) |
| Privacy / per-tenant data | easy (filter at retrieval) | easy | hard (mix in training) |

Rule of thumb: RAG for factual lookup over volatile corpora; long context for integrative tasks over fixed documents; fine-tuning for *style*, format, or new capabilities — not for adding facts.

== Pipeline Shape

```
Documents → chunking → embedding → vector index
                                          ↑
                                          │  (build-time)
                                          │
Query → embed → vector ANN search ──┐     │
              ↘ BM25 keyword search ─┼─→ fusion → rerank → top-K context
                                    ↑
                                    │  (query-time)
                                    │
                       LLM prompt: [system, query, top-K passages]
                                    ↓
                                 generated answer + citations
```

== Chunking

Documents are split into passages of ~200–800 tokens. Chunking choices materially affect quality.

=== Strategies

- *Fixed-size:* hard split every $N$ tokens. Simple, fast; cuts mid-sentence.
- *Sentence-aware:* split on sentence boundaries; bin sentences into $N$-token chunks. Standard baseline.
- *Recursive structural:* split on `\n\n` first (paragraphs), then `\n`, then sentences, then characters. LangChain's `RecursiveCharacterTextSplitter`.
- *Semantic:* embed sliding windows; split where embedding similarity drops below a threshold (Greg Kamradt's "semantic chunker"). Better for prose; expensive.
- *Layout-aware:* for PDFs/HTML, respect headings, tables, code blocks. Tools: `unstructured`, `marker`, `LlamaParse`. Critical for technical documents.

=== Overlap

Adjacent chunks share ~10–20% overlap so a fact straddling the boundary appears in at least one chunk. Cost: storage and per-chunk embedding inflation. Skip overlap for very structured content (a row in a CSV is its own chunk).

=== Parent-Child

Index small chunks (~200 tokens) for retrieval precision; return their *parents* (~2000-token sections) for context. Provides recall without diluting embedding signal. Used by LlamaIndex's `AutoMergingRetriever`.

== Embedding

Each chunk and each query is mapped to a fixed-size vector. Choices:

=== Models

- *General-purpose:* OpenAI `text-embedding-3-large` (3072d), Voyage `voyage-3` (1024d), Cohere `embed-v3`.
- *Open-source:* `bge-large-en-v1.5`, `e5-mistral-7b-instruct`, `nomic-embed-text-v1.5`, `gte-Qwen2-7B-instruct`. The 7B-parameter embedders top MTEB leaderboards.
- *Multilingual:* `multilingual-e5-large`, `bge-m3`.
- *Domain-specific:* fine-tune on your corpus with contrastive loss if MTEB-tier models underperform on jargon.

=== Matryoshka Embeddings

Matryoshka Representation Learning (Kusupati et al. 2022) trains embeddings whose *prefixes* are also valid — useful prefixes of length 256, 512, 1024, 2048 retain most of the quality of the full vector. Lets you index at low dimension and re-score at high dimension. Adopted by OpenAI v3 (supports truncation to any dimension ≥ 256).

=== Sentence Encoding vs. Late Interaction

Standard *bi-encoders* (Sentence-BERT, MPNet) produce one vector per chunk; query similarity is cosine. Fast but coarse.

*ColBERT* (Khattab–Zaharia 2020) and ColBERTv2 use *late interaction*: each chunk stores per-token vectors; query similarity = sum of max-over-tokens. Much higher quality, ~10× storage. Useful when precision matters more than scale.

== Indexing

For $N approx 10^5$ or more chunks, brute-force cosine becomes slow. Vector indexes give approximate nearest neighbor (ANN) lookup:

- *HNSW* (Hierarchical Navigable Small World): graph-based, fast queries, slow build, large RAM. Default in pgvector, Qdrant, Weaviate, Milvus.
- *IVF* (Inverted File): clustered, smaller RAM, slightly slower queries. Faiss.
- *PQ* (Product Quantization): 8–32× compression with mild recall loss. Combined with IVF as IVF-PQ; standard for billion-scale indexes.
- *DiskANN / SPANN*: SSD-backed, billion-scale, low-RAM. Microsoft, Pinecone use derivatives.

Engine selection: pgvector for "I already have Postgres," Qdrant for production-grade open source, Pinecone for managed, Faiss for batch / research.

(See _Vector and Similarity Search_ in the Database volume for algorithmic depth.)

== Hybrid Retrieval

Pure-vector retrieval misses *lexical* hits (exact product codes, function names, rare proper nouns) because semantically similar embeddings can drift from the literal token. Pure-BM25 misses synonyms and paraphrases.

*Hybrid* runs both and fuses.

=== BM25 Refresher

Score document $D$ for query $Q$:

$ "BM25"(D, Q) = sum_(q in Q) "IDF"(q) times f(q, D)(k_1 + 1) / (f(q, D) + k_1 (1 - b + b times |D| / "avgdl")) $

with $k_1 in [1.2, 2.0], b approx 0.75$. Elasticsearch, OpenSearch, Tantivy, and `rank_bm25` (Python) all implement this.

=== Reciprocal Rank Fusion

The standard fusion is *RRF* (Cormack et al. 2009): for each candidate document $d$ across $R$ ranked lists $L_r$,

$ "RRF"(d) = sum_(r=1)^R 1 / (k + "rank"_(L_r)(d)), quad k = 60 "(typical)" $

Robust to score-scale differences between systems; no tuning required. Modern hybrid systems use RRF over BM25 + vector + (optionally) a structured query.

=== Score Fusion

Alternative: linear combination $alpha times s_"vec" + (1 - alpha) times s_"bm25"$ after z-scoring. Slightly outperforms RRF if $alpha$ is tuned on dev data; brittle when distributions shift.

== Reranking

Retrieval returns 50–200 candidates. A *reranker* — a cross-encoder that jointly encodes (query, candidate) — re-scores to find the top 5–10 most relevant. Cross-encoders are slow per pair but much more accurate than bi-encoders.

```python
from sentence_transformers import CrossEncoder
ce = CrossEncoder("BAAI/bge-reranker-v2-m3")

def rerank(query, candidates, top_k=8):
    pairs = [(query, c) for c in candidates]
    scores = ce.predict(pairs)
    ranked = sorted(zip(scores, candidates), reverse=True)
    return [c for _, c in ranked[:top_k]]
```

Production rerankers: Cohere Rerank 3, Voyage Rerank-2, bge-reranker-v2-m3, GPT-4-as-reranker (very expensive but very accurate). Reranking turns a "good enough" retriever into a precision-oriented system; it is the single largest quality lever after embedding choice.

== Generation

The retrieved passages are formatted into the prompt:

```
You are a helpful assistant. Answer the question using only the
sources below. Cite each claim with [n].

Sources:
[1] <passage 1>
[2] <passage 2>
...

Question: <user query>
Answer:
```

Engineering details:

- *Passage ordering:* place the most relevant passages near the top (some models attend more to recent context — "lost in the middle," Liu et al. 2023).
- *Source format:* numbered for easy citation; include metadata (URL, title) for the answer to cite.
- *Token budget:* leave room for the answer; OOM-style truncation should drop lowest-ranked passages, not earliest.

== Advanced Patterns

=== HyDE (Hypothetical Document Embeddings)

Gao et al. (2022): the query is short and embedding-dissimilar to long passages. Generate a *hypothetical answer* with the LLM first, embed *that*, and search. Improves retrieval on zero-shot tasks where the query and ideal passage have different surface forms.

=== Multi-query / Step-back

Generate $N$ paraphrases or *step-back* abstractions of the query, retrieve for each, fuse. Helps recall.

=== Self-RAG / Corrective RAG (CRAG)

Asai et al. (2024), Yan et al. (2024): the model emits a *retrieval decision* token — it can choose not to retrieve, retrieve once, or retrieve multiple times during generation. Reduces unnecessary retrieval and lets the model verify its own outputs.

=== Graph-RAG / Knowledge-graph RAG

Microsoft's GraphRAG (2024): instead of (or in addition to) chunk retrieval, build a knowledge graph from the corpus (entities, relations, communities). Retrieval traverses the graph; the LLM is prompted with the relevant subgraph. Better for global / multi-hop questions ("what are the main themes across these 500 documents?").

== Evaluation

Component metrics:

- *Retrieval:* Recall\@k, MRR, NDCG\@k. Need labeled query–passage pairs; LLM-as-judge can label at scale.
- *Reranker:* Pair accuracy on hard negatives.
- *Generation grounded-ness:* fraction of generated claims attributable to retrieved passages.

End-to-end metrics (LLM-as-judge or human):

- *Faithfulness:* every claim supported by a source.
- *Answer relevance:* the answer addresses the actual question.
- *Context precision and recall* (Ragas framework): retrieved passages match what the answer needs.

Reference frameworks: `ragas`, `trulens`, OpenAI Evals' RAG harness.

=== Hallucination Failure Modes

- *No relevant passage retrieved* → model confabulates from parametric knowledge.
- *Relevant passage retrieved but ignored* → model trusts its prior over the source.
- *Multiple contradictory passages retrieved* → model picks one without flagging the contradiction.

Mitigations: explicit "if not in sources, say so" in the system prompt; faithfulness-tuned models (Command-R, Cohere); per-claim citation requirements.

== Engineering Notes

- *Embedding caching:* embed once at ingest; recompute only when the embedding model changes.
- *Index sharding:* per-tenant indexes for SaaS; per-time-window for log/event corpora.
- *Recency boost:* multiply scores by an age-decay factor for news/incident data.
- *Filter-first:* pre-filter by metadata (`product = X`, `language = ja`) *before* ANN search to avoid wasting recall on irrelevant tenants/categories.
- *Cold-start:* with no labeled data, use the LLM to generate query–passage pairs from the corpus; train a domain-tuned embedder.

== Further Reading

Lewis, P. et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS.

Khattab, O., Zaharia, M. (2020). "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction." SIGIR.

Cormack, G. et al. (2009). "Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods." SIGIR.

Gao, L. et al. (2022). "Precise Zero-Shot Dense Retrieval without Relevance Labels." ACL 2023.

Liu, N. et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." TACL 2024.

Asai, A. et al. (2024). "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection." ICLR.

Edge, D. et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." (Microsoft GraphRAG)

Kusupati, A. et al. (2022). "Matryoshka Representation Learning." NeurIPS.

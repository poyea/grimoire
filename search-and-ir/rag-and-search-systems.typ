#import "../template.typ": xref

= RAG and Search Systems

Retrieval-augmented generation (RAG) puts a search engine in front of a language model: retrieve passages relevant to the user's request, place them in the prompt, and generate a grounded answer. It is the dominant pattern for giving LLMs access to private, fresh, or voluminous knowledge without retraining, and it makes retrieval quality directly visible — a RAG system is a search system whose final ranker is a generator. This chapter covers the pipeline end to end (chunking, indexing, retrieval, reranking, generation), query understanding for both classic search and RAG, advanced patterns (multi-hop, agentic, GraphRAG), RAG-specific evaluation, and the architecture of a complete production search system.

*See also:* #xref("search-and-ir", "neural-retrieval", label: "Neural Retrieval") (the embedding and reranking models), #xref("search-and-ir", "vector-search", label: "Vector Search") (the index underneath), _Evaluation_ (retrieval metrics RAG inherits), #xref("search-and-ir", "inverted-indexes", label: "Inverted Indexes") and #xref("search-and-ir", "ranking-classical", label: "Ranking: Classical Models") (the lexical leg of the hybrid).

== Why Retrieval for Generation

Three failure modes of a bare LLM motivate RAG: *hallucination* (fluent fabrication when the model lacks the fact), *staleness* (knowledge frozen at training time), and *access* (private corpora the model never saw). Retrieval addresses all three and adds *attributability* — answers can cite their sources. The lineage runs from open-domain QA (DrQA, 2017: TF-IDF retrieval plus a reading model; DPR, 2020: dense retrieval) through RAG proper (Lewis et al., 2020, which trained retriever and generator jointly with the retrieved passage as a latent variable) and REALM/FiD/Atlas, to today's decoupled pattern: an off-the-shelf retriever feeding an off-the-shelf instruction-tuned LLM through the prompt. Long-context models (hundreds of thousands of tokens) shrink some use cases but do not remove the need: corpora exceed any context window, attention degrades over long contexts ("lost in the middle", Liu et al., 2024), and retrieval is cheaper per query than stuffing megabytes into every prompt.

== The Ingestion Side

=== Chunking

Encoders and prompts both bound passage length, so documents are split into chunks — and chunking policy moves RAG quality as much as model choice:

- *Fixed-size* (200–512 tokens) with 10–20% overlap: the baseline; overlap guards against splitting an answer across a boundary.
- *Structure-aware*: split on headings, paragraphs, list items; keep tables and code blocks intact. Markdown/HTML structure is the cheapest quality win available.
- *Contextualized chunks*: prepend document title and section path to each chunk before embedding; or generate a short situating sentence per chunk with an LLM ("contextual retrieval"), which materially reduces retrieval failures on anaphoric chunks ("the company", "this method").
- *Parent-child*: retrieve over small chunks for precision, but hand the generator the enclosing larger section for context.

=== Indexing

Standard practice is a hybrid index per chunk: dense embedding (with the embedding model version recorded — re-embedding on model upgrade is a planned migration, see #xref("search-and-ir", "vector-search", label: "Vector Search")), BM25 over the raw text, and metadata fields (source, date, ACL, tenant) for filtering. Permissions deserve emphasis: RAG over corporate documents must enforce document-level ACLs *at retrieval time*; embedding leakage through an unfiltered index is a textbook incident.

== The Query Side

=== Query Understanding

Classic search pipelines normalize and interpret the query before retrieval: spelling correction, segmentation, synonym and entity expansion, intent classification (navigational / informational / transactional, per Broder's 2002 taxonomy), and facet extraction ("red running shoes size 10" to filters plus a relaxed text query). RAG adds LLM-powered variants:

- *Query rewriting*: condense a multi-turn conversation into a standalone query (resolving "what about its pricing?"); decompose complex questions into sub-queries.
- *HyDE* (Gao et al., 2023): generate a hypothetical answer document and embed _that_ for retrieval, bridging the query–document style gap zero-shot.
- *Multi-query expansion*: generate several paraphrases, retrieve for each, fuse with RRF.
- *Routing*: classify which index/tool a query needs (product catalog vs. documentation vs. web), or whether retrieval is needed at all (Self-RAG-style adaptive retrieval) — skipping retrieval for chit-chat saves latency and avoids distracting context.

=== Retrieval and Reranking

The retrieval stack inside RAG is exactly the hybrid cascade of #xref("search-and-ir", "neural-retrieval", label: "Neural Retrieval"): BM25 plus dense ANN, RRF fusion, top \~50–100 into a cross-encoder reranker, top 3–10 into the prompt. The reranker matters more in RAG than in ranked-list search because the generator sees only a handful of passages: precision\@5 is the binding constraint, and irrelevant context actively harms generation (models are distractible). Passage order in the prompt matters too; placing the strongest evidence first (or first and last) mitigates lost-in-the-middle effects.

== Generation and Grounding

The generator receives instructions, the retrieved passages (typically delimited and numbered), and the question, with directives to answer only from the context and cite passage numbers. Practical levers:

- *Citations*: require inline source markers; verify them post-hoc by checking the cited passage entails the claim (NLI models or LLM-as-judge).
- *Abstention*: an explicit "say you don't know if the context is insufficient" instruction plus retrieval-confidence thresholds; the worst RAG failure is a confident answer from irrelevant context.
- *Conflicts*: retrieved passages can disagree (stale vs. fresh docs); date-aware prompting and recency-boosted retrieval reduce this.

== Advanced Patterns

- *Multi-hop / iterative*: questions like "who advised the author of X?" need a retrieve–read–retrieve loop; frameworks interleave reasoning and retrieval (IRCoT-style), with each hop's answer parameterizing the next query.
- *Agentic RAG*: the LLM drives retrieval as a tool — choosing queries, judging sufficiency, and searching again — trading latency for robustness on hard questions.
- *GraphRAG* (Edge et al., 2024): build an entity–relation graph and hierarchical community summaries over the corpus at index time; global "summarize the themes" questions, which chunk retrieval fundamentally cannot answer, are served from community summaries.
- *Self-RAG / corrective RAG*: train or prompt the model to critique retrieved passages and its own draft, re-retrieving when evidence is weak.

== Evaluating RAG

Decompose, then measure end to end:

- *Retrieval*: recall\@k and nDCG against judged qrels (see _Evaluation_); for RAG specifically, "context contains the answer"\@k.
- *Generation, given context*: *faithfulness/groundedness* (is every claim supported by the retrieved passages?) and *answer relevance* — typically scored by an LLM judge (the RAGAS-style metric family; Es et al., 2024), with periodic human calibration of the judge.
- *End-to-end*: answer correctness on a gold QA set; citation precision/recall.

Most production failures localize to retrieval (the answer was never in the context) or chunking (the answer was split or stripped of context), not the generator — so instrument the pipeline to attribute failures per stage.

== Anatomy of a Production Search System

Pulling the volume together, a full system comprises:

1. *Ingestion*: crawlers/connectors, parsing and extraction, deduplication (SimHash/MinHash), chunking, enrichment (entities, embeddings), index writing — batch plus a streaming path for freshness.
2. *Indexes*: inverted (sharded by document, replicated), ANN, metadata/doc store; immutable segments with background merges.
3. *Query path*: understanding and rewriting, fan-out to shards, first-stage retrieval with pruning (WAND / HNSW beam), fusion, L1/L2 ranking (#xref("search-and-ir", "learning-to-rank", label: "Learning to Rank")), business-rule and diversity layers, snippet generation — all under a tail-latency budget (e.g., p99 under 200 ms) enforced with per-stage timeouts and graceful degradation (skip the reranker rather than miss the deadline).
4. *Feedback loop*: click and dwell logging, judgment collection, offline metric pipelines, interleaving and A/B infrastructure (_Evaluation_), and retraining schedules for rankers and embedders.

RAG adds the generation tier on top but changes nothing below it — which is the closing point of this volume: large language models did not replace the search stack; they became its most demanding customer.

== Pitfalls

- *Chunking blindness*: tables, code, and multi-paragraph arguments destroyed by naive splitting; inspect retrieved chunks by eye early and often.
- *Evaluating only end-to-end*: a wrong answer cannot be fixed without knowing which stage failed; keep stage-level metrics.
- *Context stuffing*: more passages is not better — precision beats volume once the answer is present, and distractors measurably increase hallucination.
- *Ignoring the lexical leg*: pure-vector RAG misses identifiers, SKUs, error codes; hybrid retrieval is the default, not an optimization.
- *Unversioned embeddings*: mixing vectors from different model versions in one index silently corrupts similarity.
- *ACL afterthoughts*: filter at retrieval, test with adversarial cross-tenant queries.

== Further Reading

- Lewis, P. et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. _NeurIPS_.
- Karpukhin, V. et al. (2020). Dense passage retrieval for open-domain question answering. _EMNLP_.
- Gao, L., Ma, X., Lin, J., & Callan, J. (2023). Precise zero-shot dense retrieval without relevance labels. _ACL_. (HyDE)
- Liu, N. et al. (2024). Lost in the middle: how language models use long contexts. _TACL_.
- Edge, D. et al. (2024). From local to global: a graph RAG approach to query-focused summarization. _arXiv:2404.16130_.
- Es, S. et al. (2024). RAGAS: automated evaluation of retrieval augmented generation. _EACL_.
- Broder, A. (2002). A taxonomy of web search. _SIGIR Forum_.

#import "../template.typ": xref

= Query Processing <query-processing>

Query processing turns a user query and an inverted index into a ranked list of documents, ideally in a few milliseconds over billions of documents. The art is in not scoring most of the collection: dynamic pruning algorithms such as WAND and Block-Max WAND skip the vast majority of postings while returning exactly the same top-$k$ results as exhaustive evaluation. This chapter covers boolean retrieval, traversal strategies, top-$k$ pruning, phrase and proximity evaluation, and the analysis chain that produces query terms in the first place.

*See also:* #xref("search-and-ir", "inverted-indexes", label: "Inverted Indexes") (the structures being traversed), #xref("search-and-ir", "ranking-classical", label: "Ranking: Classical Models") (the scores being maximized), #xref("search-and-ir", "neural-retrieval", label: "Neural Retrieval") (where pruning meets learned sparse models), #xref("search-and-ir", "rag-and-search-systems", label: "RAG and Search Systems") (query processing inside multi-stage pipelines); #xref("database", "query-processing", label: "Query Processing") (the relational analogue: operators, joins, and cardinality).

== Boolean Retrieval

The earliest IR model treats a query as a boolean expression over terms and returns the exact matching set.

- *AND*: intersect postings lists. With lists of length $L_1 <= L_2$, a merge costs $O(L_1 + L_2)$; with skip pointers or `nextGEQ`, intersection costs $O(L_1 log (L_2 / L_1))$ — drive the intersection from the rarest term.
- *OR*: a multi-way union, typically with a min-heap over list cursors.
- *NOT*: usually applied as a filter (`AND NOT`) rather than materializing a complement.

Query optimizers order conjuncts by ascending document frequency so that intermediate results stay small. Pure boolean retrieval survives in legal/patent search and as the *matching* phase beneath ranked retrieval: a ranked query is commonly evaluated as an OR (or a relaxed AND) whose matches are then scored.

== TAAT versus DAAT

Two traversal disciplines exist for scoring matching documents:

=== Term-at-a-Time (TAAT)

Process one query term's full postings list at a time, accumulating partial scores in a per-document *accumulator* array (or hash). After the last term, sort accumulators and return the top $k$. TAAT has sequential, cache-friendly access per list, but accumulator memory can reach the size of the union of postings. Accumulator-limiting heuristics (Moffat & Zobel, 1996) cap the number of accumulators by processing terms in order of decreasing rarity and freezing creation of new accumulators past a quota.

=== Document-at-a-Time (DAAT)

Advance a cursor in every term's list in parallel, always to the smallest current doc ID; compute each document's complete score the moment all cursors pass it; maintain the top $k$ in a min-heap. DAAT needs only $O(k)$ result memory, produces complete scores immediately (enabling safe pruning), and dominates in modern engines including Lucene. Its weakness is pointer-chasing across many lists, mitigated by block decoding.

== Top-k Pruning and Early Termination

Exhaustive DAAT scores every document containing any query term. Dynamic pruning skips documents that provably cannot enter the top $k$. Strategies are *safe* (identical results to exhaustive) or *unsafe* (approximate).

=== MaxScore

MaxScore (Turtle & Flood, 1995) precomputes each term's maximum possible contribution $U_t = max_d "score"(t, d)$. Let $theta$ be the current $k$-th best score in the heap. Sort terms by $U_t$; partition into *essential* terms (whose upper bounds can collectively beat $theta$) and *non-essential* terms. Only documents containing at least one essential term can qualify, so non-essential lists are never used to find candidates — only probed (via `nextGEQ`) to complete scores, and scoring of a candidate aborts as soon as its score plus remaining upper bounds falls below $theta$. As $theta$ rises, more terms become non-essential.

=== WAND

WAND — Weak/Weighted AND (Broder et al., 2003) — generalizes this with a pivot mechanism:

1. Sort term cursors by current doc ID.
2. Find the *pivot*: the first prefix of cursors whose summed upper bounds exceed $theta$; the pivot document is that cursor's doc ID.
3. If the preceding cursors are already on the pivot document, score it fully and update the heap; otherwise, advance them to the pivot with `nextGEQ` and repeat.

Every document before the pivot is skipped without decoding. WAND is safe and typically evaluates a small fraction of postings for disjunctive top-10 queries.

=== Block-Max WAND

Term-level upper bounds are loose: one outlier document inflates $U_t$ for the whole list. *Block-Max WAND* (BMW; Ding & Suel, 2011) stores a maximum impact score per compressed block (e.g., per 128 postings) alongside the skip data. After WAND selects a pivot, BMW checks the much tighter *block-level* upper bounds at the pivot; if they cannot beat $theta$, it jumps directly past the offending block boundary without decoding it. BMW gives 2–4$times$ speedups over WAND and is implemented in Lucene (as `WANDScorer` machinery with block-max metadata, since Lucene 8 made top-$k$ scoring the default via `impacts`). Variable-sized blocks (VBMW; Mallia et al., 2017) choose block boundaries to minimize upper-bound looseness.

=== Unsafe Early Termination

When exactness is negotiable:
- *Impact-ordered indexes* sort postings by quantized score contribution instead of doc ID; processing stops after a budget (score-at-a-time, e.g., JASS). Anytime behavior with graceful degradation.
- *Static index pruning* drops low-impact postings at build time.
- *Tiering*: a small high-quality tier (by static rank) is searched first; lower tiers only on insufficient results — standard in web search.

These interact with learned sparse retrieval: SPLADE-style term weights have flatter score distributions, which weakens WAND/BMW pruning and motivated dedicated work on pruning-friendly training (see #xref("search-and-ir", "neural-retrieval", label: "Neural Retrieval")).

== Phrase and Proximity Queries

A phrase query \"new york\" requires positional verification after doc-ID intersection: walk the two position lists checking $p^((2)) = p^((1)) + 1$ (generalizing to $k$-term phrases with offsets). Proximity operators relax equality to a window: score or filter on $min$-cover spans containing all terms. Sloppy phrase matching (Lucene's `slop`) permits up to $s$ position edits.

Proximity also serves ranking: adding a span-based bonus to BM25, e.g., the minimal window length containing all query terms, reliably improves web relevance. Because positional decoding is expensive, engines verify phrases only on documents that survive doc-level intersection, and may serve very frequent phrases from a dedicated phrase index.

== Query Parsing and Analysis Chains

Before any postings are read, raw query text passes through the same *analysis chain* used at indexing time — asymmetry between the two is a classic source of zero-result bugs.

A typical chain:
1. *Character filtering*: Unicode normalization (NFKC), HTML stripping, accent folding.
2. *Tokenization*: splitting on word boundaries; nontrivial for CJK languages (dictionary or model-based segmentation), URLs, hyphenations.
3. *Token filtering*: lowercasing, stopword removal (now often skipped — pruning makes stopwords cheap and they matter in phrases), synonym expansion, stemming (Porter, Snowball) or lemmatization. Stemming trades recall for precision errors; many engines index both stemmed and exact forms in parallel fields.
4. *Query construction*: parse operators (quotes, +/-, fields like `title:`), build a query tree of boolean/phrase/term nodes, and apply rewrites — multi-field expansion into a DisMax/combined-fields query, fuzzy matching via Levenshtein automata against the FST dictionary, and wildcard expansion bounded by a term budget.

Higher-level query understanding — spelling correction, segmentation into concepts, intent classification, vector query generation — belongs to the full pipeline picture in #xref("search-and-ir", "rag-and-search-systems", label: "RAG and Search Systems").

== Putting It Together

A representative ranked-query execution in a Lucene-style engine:

1. Analyze the query into terms; look up each in the FST dictionary.
2. Open per-segment postings cursors with block-max impact metadata.
3. Run DAAT with Block-Max WAND maintaining a global top-$k$ heap (sharing $theta$ across segments).
4. Verify positional constraints for phrase nodes on surviving candidates.
5. Return doc IDs and scores; fetch stored fields only for the final page of results.

Latency is dominated by the longest postings lists touched; p99 outliers come from queries of many common terms, handled by pruning, tiering, and timeout-based partial evaluation.

== Further Reading

- Turtle, H., & Flood, J. (1995). Query evaluation: strategies and optimizations. _Information Processing and Management_.
- Broder, A. et al. (2003). Efficient query evaluation using a two-level retrieval process. _CIKM_.
- Ding, S., & Suel, T. (2011). Faster top-k document retrieval using block-max indexes. _SIGIR_.
- Mallia, A. et al. (2017). Faster BlockMax WAND with variable-sized blocks. _SIGIR_.
- Tonellotto, N., Macdonald, C., & Ounis, I. (2018). Efficient query processing for scalable web search. _Foundations and Trends in IR_.
- Crane, M. et al. (2017). A comparison of document-at-a-time and score-at-a-time query evaluation. _WSDM_.

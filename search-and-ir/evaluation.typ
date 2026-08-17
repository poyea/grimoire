#import "../template.typ": xref

= Evaluation

Search quality cannot be eyeballed: rankings differ subtly, queries vary wildly, and intuition about "better" is unreliable. IR built one of the most disciplined evaluation traditions in computer science — the Cranfield paradigm of fixed collections, pooled judgments, and rank-based metrics, institutionalized by TREC since 1992 — and complemented it with online experimentation on live traffic. This chapter covers the offline metrics (precision/recall through nDCG), test-collection construction and its biases, statistical significance, the modern benchmarks (MS MARCO, BEIR), and online evaluation via interleaving and A/B tests.

*See also:* #xref("search-and-ir", "learning-to-rank", label: "Learning to Rank") (these metrics as training objectives and click bias), #xref("search-and-ir", "neural-retrieval", label: "Neural Retrieval") (the benchmark results cited there), #xref("search-and-ir", "ranking-classical", label: "Ranking: Classical Models") (the baselines every evaluation needs), #xref("search-and-ir", "rag-and-search-systems", label: "RAG and Search Systems") (evaluating retrieval inside a generation pipeline); #xref("llm", "evaluation", label: "Evaluation") (generation evaluation: benchmarks and LLM-as-judge).

== The Cranfield Paradigm

An offline test collection has three parts: a document corpus, a set of queries (topics), and *relevance judgments* (qrels) saying which documents are relevant to which queries. Systems are compared by running all queries and averaging a metric over them. The paradigm's core assumptions — relevance is topical, static, and independent of other retrieved documents and of the user — are all false in detail, yet the methodology has been validated repeatedly: system orderings on Cranfield collections broadly predict orderings in user studies. Fix the collection, vary the system; never compare numbers across collections.

== Set-Based Metrics

For a result set (ignoring order), with $R$ relevant documents in the collection and $r$ relevant among $n$ retrieved:

$ "precision" = r / n, quad "recall" = r / R, quad F_1 = (2 dot "precision" dot "recall") / ("precision" + "recall") $

Ranked retrieval evaluates these at cutoffs: $P@10$ (precision in the first page) and $"recall"@k$ — the latter the key metric for first-stage retrievers and ANN indexes, since a reranker cannot recover a document the candidate stage dropped.

== Rank-Based Metrics

=== Average Precision and MAP

Average precision (AP) for one query averages $P@k$ at each rank $k$ where a relevant document appears (relevant documents never retrieved contribute zero), dividing by $R$. AP rewards putting relevant documents early and integrates precision over recall levels (it approximates the area under the precision–recall curve). Mean average precision (MAP) averages AP over queries; it was TREC's headline metric for two decades and assumes binary relevance and recall-oriented users.

=== MRR

For tasks with one right answer (navigational queries, known-item search, QA passage retrieval), mean reciprocal rank scores each query by $1 \/ r_1$, the reciprocal rank of the first relevant result, averaged over queries. MRR\@10 is the official MS MARCO metric. It is coarse — a flip between ranks 1 and 2 costs 0.5; between 9 and 10, 0.011 — and statistically noisy.

=== nDCG

Discounted cumulative gain handles *graded* relevance: gain from a document with grade $"rel"_i$ at rank $i$ is discounted logarithmically,

$ "DCG"@k = sum_(i=1)^k (2^("rel"_i) - 1) / (log_2 (i + 1)) $

(the exponential gain emphasizes highly relevant documents; the original Järvelin–Kekäläinen formulation used linear gain). Normalizing by the *ideal* DCG — the DCG of the perfect ordering of judged documents — yields $"nDCG"@k in [0, 1]$, with 1 for the ideal ranking, comparable across queries with different numbers of relevant documents. nDCG\@10 is the de facto standard for web-style ranking and BEIR. ERR (Chapelle et al., 2009) is the cascade-model alternative: the expected reciprocal rank at which a user, scanning top-down and stopping with probability proportional to relevance, is satisfied — it penalizes redundancy after a perfect result more than nDCG does.

=== Which Metric When

- First-stage retrieval / ANN: recall\@100 or recall\@1000.
- Single-answer tasks: MRR\@10, success\@k.
- Graded web ranking: nDCG\@10.
- Recall-oriented tasks (legal, patent, systematic review): MAP, recall at depth.

== Building Judgments: Pooling and Its Discontents

Judging every document for every query is impossible, so TREC *pools*: take the top $k$ (e.g., 100) from every participating system, judge the union, and treat everything unjudged as non-relevant. This is fine for comparing the pooled systems but biased against later systems that retrieve relevant-but-unjudged documents — the *holes* problem, which bites hardest when evaluating dense retrievers on collections pooled from lexical-era runs. Mitigations: bpref and condensed-list variants (drop unjudged documents before scoring), judged\@k diagnostics, and re-judging campaigns. Inter-assessor agreement on relevance is modest (Cohen's kappa often 0.5–0.7 at TREC), but disagreement perturbs absolute scores far more than system *rankings*, which are stable (Voorhees, 2000). Statistical machinery: compare systems with a paired test across queries — the paired t-test is standard and robust in practice (Smucker et al., 2007) — report effect sizes, and beware multiple comparisons when sweeping configurations. Query-set variance dominates: 50 topics is the working TREC minimum, and differences under ~2 absolute nDCG points on 50 queries are rarely significant.

== Modern Benchmarks

- *MS MARCO* (Bajaj et al., 2016): \~8.8M web passages, \~530k training queries from Bing logs with *sparse* judgments (about one labeled positive per query). Its scale made neural training possible; its shallow labels make absolute scores misleading (many "non-relevant" retrieved passages are fine answers) and invite leaderboard overfitting.
- *TREC Deep Learning* (2019–): MS MARCO corpus with dense, graded NIST judgments on \~50 queries per year — the corrective lens for MARCO's sparse labels.
- *BEIR* (Thakur et al., 2021): 18 heterogeneous tasks (bio-medical, financial, argument, code-adjacent) for *zero-shot* evaluation, nDCG\@10 averaged across tasks; the standard out-of-domain check. MTEB extends the idea across embedding tasks beyond retrieval, with the same caveat that public leaderboards attract training-data contamination.

== Online Evaluation

Offline metrics rank systems; only users rank experiences. Online methods, in increasing cost:

=== Interleaving

Merge results from rankers A and B into one list shown to real users and credit the ranker whose documents get clicked. *Team-draft interleaving* (Radlinski et al., 2008) alternates draft picks (random first-picker per pair) so each result carries a team label; clicks score for the team. Interleaving is 10–100$times$ more sensitive than an A/B test on the same traffic because every impression is a paired within-user comparison — ideal as a cheap screen before A/B. Variants: balanced interleaving (earlier, subtle biases), probabilistic interleaving, and multileaving for comparing many rankers at once.

=== A/B Testing

Split traffic between control and treatment; compare metrics. Click-through rate alone is gameable (clickbait snippets raise CTR while quality falls), so mature search teams use guardrail and quality proxies: abandonment rate, time-to-first-click and click dwell time (clicks with dwell under \~10–30 s are "bad clicks"), pagination and query-reformulation rates, and long-term holdbacks for slow effects. Classic findings from this practice (Kohavi et al., at Bing/Microsoft): most ideas fail online, small latency regressions measurably reduce engagement and revenue, and offline gains routinely fail to replicate — which is the argument for keeping all three layers (offline, interleaving, A/B) in the loop.

== Pitfalls

- *Unjudged means non-relevant*: evaluating a new architecture on old pools undercounts its quality; report judged\@10 alongside nDCG\@10.
- *Cross-collection comparison*: an nDCG of 0.5 on one collection and 0.4 on another says nothing; only within-collection deltas are meaningful.
- *Sparse-label leaderboards*: MRR gains on MS MARCO past a point reflect label idiosyncrasies, not user value; confirm on TREC-DL's dense judgments.
- *Tuning on the test set*: repeated evaluation against a fixed qrel set is implicit training on it; keep a held-out query set.
- *Average hides variance*: a system that wins on average may catastrophically fail on a query slice (rare entities, non-English); report per-slice metrics and loss queries.
- *Offline–online gaps*: position-biased clicks make naive "click rank" comparisons favor the incumbent; use interleaving or propensity-corrected estimates.

== Further Reading

- Manning, C., Raghavan, P., & Schütze, H. (2008). _Introduction to Information Retrieval_, ch. 8. Cambridge University Press.
- Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. _ACM TOIS_. (nDCG)
- Voorhees, E. (2000). Variations in relevance judgments and the measurement of retrieval effectiveness. _Information Processing and Management_.
- Smucker, M., Allan, J., & Carterette, B. (2007). A comparison of statistical significance tests for information retrieval evaluation. _CIKM_.
- Radlinski, F., Kurup, M., & Joachims, T. (2008). How does clickthrough data reflect retrieval quality? _CIKM_.
- Kohavi, R., Tang, D., & Xu, Y. (2020). _Trustworthy Online Controlled Experiments_. Cambridge University Press.
- Thakur, N. et al. (2021). BEIR: a heterogeneous benchmark for zero-shot evaluation of information retrieval models. _NeurIPS Datasets and Benchmarks_.

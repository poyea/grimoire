#import "../template.typ": xref

= Learning to Rank

Learning to rank (LTR) replaces hand-tuned scoring formulas with models trained on relevance judgments or click data. Rather than guessing how to combine BM25, PageRank, freshness, and a hundred other signals, a learned ranker fits the combination to optimize a ranking metric. LTR powered the second generation of web search (Bing's RankNet lineage, Yandex's MatrixNet) and remains the standard final-stage ranker in production: gradient-boosted trees over rich features, with LambdaMART the perennial benchmark winner. This chapter covers the pointwise/pairwise/listwise taxonomy, RankNet through LambdaMART, features and training data, click models, and the cascade architecture LTR lives in.

*See also:* #xref("search-and-ir", "ranking-classical", label: "Ranking: Classical Models") (the features), _Evaluation_ (the metrics LTR optimizes), #xref("search-and-ir", "neural-retrieval", label: "Neural Retrieval") (learned first stages and cross-encoder rerankers), _Query Processing_ (why LTR runs only on a candidate set).

== The Problem Setup

Training data consists of queries, each with a set of candidate documents and relevance labels — graded judgments $y in {0, ..., 4}$ from human assessors (the TREC/Web standard: bad, fair, good, excellent, perfect) or implicit labels derived from clicks. Each query–document pair is represented by a feature vector $x_(q,d)$, and the model produces a score $f(x_(q,d))$; documents are ranked by score. Two facts shape everything:

- Ranking metrics (nDCG, MAP, MRR) depend on _sorted order_, so they are piecewise constant in the scores: zero gradient almost everywhere. Direct optimization is combinatorial; every practical method optimizes a surrogate.
- Only _relative_ order within a query matters. Scores need not be comparable across queries, and losses are computed per query.

== The Taxonomy

=== Pointwise

Treat each query–document pair independently: regress on the label (squared loss) or classify the grade (ordinal regression, e.g., McRank). Simple, but it optimizes the wrong thing — it penalizes absolute score errors equally everywhere, while ranking only cares about errors that swap document order, especially near the top.

=== Pairwise

Reduce ranking to binary classification on preference pairs: for documents $d_i, d_j$ with $y_i > y_j$ under the same query, the model should score $f(x_i) > f(x_j)$. RankNet, RankSVM (Joachims, 2002, trained on click-skip pairs), and RankBoost live here. Pairwise methods dominate in practice but, unweighted, still mismatch the metric: a swap at ranks 1–2 costs the same as one at ranks 100–101.

=== Listwise

Define the loss over the whole ranked list per query: ListNet (cross-entropy between permutation distributions via the Plackett–Luce model), SoftRank (smooth the rank distribution), AdaRank (boosting on the metric directly), and — by way of its gradients — LambdaRank. Listwise methods can weight errors by their metric impact, which is the key to optimizing nDCG.

== RankNet, LambdaRank, LambdaMART

The Burges line of work (Microsoft Research, 2005–2010) is the canonical LTR story.

=== RankNet

RankNet (Burges et al., 2005) models the probability that $d_i$ should rank above $d_j$ with a logistic function of the score difference, $s_i = f(x_i)$:

$ P_(i j) = 1 / (1 + e^(-sigma (s_i - s_j))) $

and minimizes cross-entropy against the true preference. The gradient with respect to $s_i$ from the pair $(i, j)$ with $y_i > y_j$ is

$ lambda_(i j) = -sigma / (1 + e^(sigma (s_i - s_j))) $

— a "force" pushing $i$ up and $j$ down, strongest when the pair is mis-ordered. Per-document lambdas sum over all pairs the document participates in.

=== LambdaRank

The insight of LambdaRank (Burges et al., 2006): you never need the loss, only its gradients. So scale each pair's lambda by the metric change from swapping the two documents:

$ lambda_(i j) = (-sigma) / (1 + e^(sigma (s_i - s_j))) dot |Delta "nDCG"_(i j)| $

Pairs whose swap would move a relevant document into the top ranks get large gradients; swaps deep in the list get tiny ones. Empirically (and with later theoretical support via LambdaLoss, Wang et al., 2018) this directly improves nDCG, and any metric with a computable $Delta$ can be plugged in (MAP, MRR, ERR).

=== LambdaMART

LambdaMART (Burges, 2010) drives gradient-boosted regression trees (MART) with lambda gradients instead of a neural network: each boosting round fits a small tree to the per-document lambdas and updates scores additively. It won the Yahoo! Learning to Rank Challenge (2010) and remains the strongest baseline on tabular ranking features. Production implementations: LightGBM (`lambdarank` objective), XGBoost (`rank:ndcg`), and the Elasticsearch/OpenSearch LTR plugins, which apply such models as a rescorer.

Why trees beat neural networks here: LTR features are heterogeneous tabular signals (scores, counts, ratios, booleans) with non-smooth interactions, exactly where GBDTs excel; neural rankers win only when they consume raw text (see #xref("search-and-ir", "neural-retrieval", label: "Neural Retrieval")).

== Features

A production feature vector mixes three classes:

- *Query–document* (dynamic): BM25 and language-model scores per field (title, body, anchors, URL), term proximity, exact/partial match flags, embedding cosine similarity.
- *Document-only* (static): PageRank, spam score, document length, freshness, quality classifier outputs, historical CTR.
- *Query-only*: query length, predicted intent/category, IDF statistics — useless alone for ranking within a query, but valuable in interactions ("for navigational queries, weight URL match heavily").

Feature hygiene matters: per-query normalization of unbounded features, log-transforms of heavy-tailed signals, and leakage checks (a "historical clicks on this result for this query" feature can make offline numbers spectacular and the model useless for new documents).

== Training Data: Judgments and Clicks

Editorial judgments are expensive (Yahoo!'s challenge set: ~36k queries, ~880k judged pairs) and pooled — only documents retrieved by participating systems are judged, biasing against novel rankers. Clicks are free and abundant but biased:

- *Position bias*: users click higher ranks regardless of relevance. Click models formalize this; the simplest, the position-based model (PBM), factorizes click probability as examination times relevance, $P("click") = P("exam" | "rank") dot P("rel" | q, d)$. The cascade model (Craswell et al., 2008) instead assumes top-down examination that stops at the first satisfying click.
- *Counterfactual LTR* (Joachims et al., 2017) corrects bias with inverse propensity scoring: weight each click by $1 \/ P("exam" | "rank")$, with propensities estimated from randomization (result swaps) or jointly with the ranker (Wang et al., 2018, regression EM). Unbiased LambdaMART variants apply the same correction inside boosting.
- Skips carry signal too: a click at rank 3 is implicit negative feedback on ranks 1–2 (Joachims, 2002).

== The Cascade Architecture

LTR is too expensive to score the whole corpus, so production search is a funnel:

1. *Retrieval / L0*: BM25 (with WAND-style pruning) and/or ANN over embeddings produce \~1,000–10,000 candidates per shard.
2. *First-stage ranking / L1*: a cheap model (small GBDT, linear) with index-resident features trims to a few hundred.
3. *Final ranking / L2*: the full LambdaMART or neural model with all features, often including cross-encoder scores, on the top \~100.
4. *Re-ranking layers*: diversity (MMR), freshness boosts, business rules, personalization.

Each stage trades recall for precision; the training subtlety is that downstream models only ever see what upstream stages surfaced (sample selection bias), so training data should include some randomized or exploration traffic.

== Pitfalls

- *Metric mismatch*: optimizing pairwise accuracy or pointwise RMSE and reporting nDCG; use lambda-weighted objectives.
- *Training/serving skew*: features computed differently offline (from logs) and online (from the live index) silently degrade the model. Log the served feature values.
- *Click feedback loops*: training on clicks produced by the current ranker entrenches it; without exploration or propensity correction the system cannot learn that an unshown document is good.
- *Per-query imbalance*: queries with hundreds of judged documents dominate pairwise losses unless pairs are normalized per query.
- *Stale negatives*: judgments pooled years ago miss newer relevant documents, so "unjudged" must not be treated as "irrelevant" when evaluating (see _Evaluation_, condensed lists and bpref).

== Further Reading

- Liu, T.-Y. (2009). Learning to rank for information retrieval. _Foundations and Trends in IR_.
- Burges, C. et al. (2005). Learning to rank using gradient descent. _ICML_. (RankNet)
- Burges, C. (2010). From RankNet to LambdaRank to LambdaMART: an overview. _Microsoft Research Technical Report MSR-TR-2010-82_.
- Joachims, T. (2002). Optimizing search engines using clickthrough data. _KDD_.
- Joachims, T., Swaminathan, A., & Schnabel, T. (2017). Unbiased learning-to-rank with biased feedback. _WSDM_.
- Chapelle, O., & Chang, Y. (2011). Yahoo! Learning to Rank Challenge overview. _JMLR Workshop and Conference Proceedings_.
- Wang, X. et al. (2018). The LambdaLoss framework for ranking metric optimization. _CIKM_.

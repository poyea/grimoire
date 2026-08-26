#import "../template.typ": xref

= Ranking: Classical Models

Classical ranking functions score documents against queries using only term statistics — no training data, no embeddings — yet remain remarkably hard to beat. BM25, published in 1994, is still the default first-stage ranker in nearly every production search system and a mandatory baseline in neural IR papers. This chapter derives TF-IDF and BM25, develops the language-modeling view of retrieval, extends to fielded documents with BM25F, and covers query-independent static signals, chiefly PageRank.

*See also:* #xref("search-and-ir", "inverted-indexes", label: "Inverted Indexes") (where tf and df live), _Query Processing_ (maximizing these scores under pruning), #xref("search-and-ir", "learning-to-rank", label: "Learning to Rank") (these scores as features), #xref("search-and-ir", "neural-retrieval", label: "Neural Retrieval") (learned alternatives and hybrids).

== The Vector Space Model and TF-IDF

The vector space model (Salton, 1970s) represents documents and queries as vectors over the vocabulary and ranks by similarity, typically the cosine. The weighting question — what value goes in each coordinate — is answered by two intuitions:

- *Term frequency*: a document mentioning a term often is more about it. Raw counts overstate this; common damped forms are $"tf" = 1 + log "tf"_(t,d)$ or $"tf"_(t,d) / ("tf"_(t,d) + c)$.
- *Inverse document frequency*: terms appearing in few documents are more discriminative:

$ "idf"(t) = log N / ("df"(t)) $

where $N$ is the collection size and $"df"(t)$ the number of documents containing $t$. The product gives the TF-IDF weight, and the score of $d$ for query $q$ is $sum_(t in q) "tf-idf"(t, d)$ (optionally length-normalized via cosine). IDF has a probabilistic justification: it approximates the Robertson–Sparck Jones relevance weight under the assumption that relevant documents are rare.

== BM25

BM25 (Robertson et al., 1994; "Best Match 25" from the Okapi system at City University London) is the culmination of the probabilistic relevance framework, combining saturating term frequency, document length normalization, and IDF:

$ "BM25"(q, d) = sum_(t in q) "idf"(t) dot ("tf"(t, d) dot (k_1 + 1)) / ("tf"(t, d) + k_1 dot (1 - b + b dot (|d|) / ("avgdl"))) $

with the Robertson–Sparck Jones IDF (note the +0.5 smoothing, which can go negative for terms in over half the collection; Lucene clamps with a +1 inside the log):

$ "idf"(t) = log (N - "df"(t) + 0.5) / ("df"(t) + 0.5) $

The two parameters have distinct, interpretable roles:

- $k_1$ controls *term-frequency saturation*. The tf component is a rational function rising from 0 toward an asymptote of $k_1 + 1$: the first occurrence of a term is worth far more than the tenth. With $k_1 = 0$ the model becomes binary (presence/absence); large $k_1$ approaches linear tf. Typical $k_1 in (1.2, 2.0)$; Lucene defaults to 1.2.
- $b in (0, 1)$ controls *length normalization*. With $b = 1$, tf is fully normalized by document length relative to the average ($|d| \/ "avgdl"$), embodying the "verbosity hypothesis" that long documents inflate tf without more relevance; with $b = 0$, length is ignored ("scope hypothesis": long documents simply cover more). Default $b = 0.75$. Tuning $b$ per corpus matters more than tuning $k_1$: titles and tweets want small $b$, verbose web pages larger.

BM25's virtues: per-term contributions are bounded (good for WAND upper bounds), it needs only tf, df, $|d|$, avgdl — all cheap index statistics — and its effectiveness is stable across collections. On BEIR's out-of-domain benchmark, plain BM25 outperformed many early dense retrievers (Thakur et al., 2021).

== Language Models for IR

The language-modeling approach (Ponte & Croft, 1998) reframes retrieval generatively: estimate a unigram language model $theta_d$ from each document, and rank by *query likelihood* — the probability the document's model generates the query:

$ P(q | theta_d) = product_(t in q) P(t | theta_d)^("tf"(t, q)) $

The maximum-likelihood estimate $P(t | theta_d) = "tf"(t, d) \/ |d|$ assigns zero probability to absent terms, zeroing the whole product, so smoothing with the collection model $P(t | C)$ is essential — and, remarkably, smoothing is what produces IDF-like and length-normalization effects (Zhai & Lafferty, 2001).

=== Jelinek-Mercer Smoothing

Linear interpolation with fixed weight $lambda$:

$ P(t | theta_d) = (1 - lambda) ("tf"(t, d)) / (|d|) + lambda P(t | C) $

Good for verbose queries; $lambda approx 0.7$ typical.

=== Dirichlet Smoothing

A Bayesian estimate with a Dirichlet prior of mass $mu$ over the collection distribution:

$ P(t | theta_d) = ("tf"(t, d) + mu P(t | C)) / (|d| + mu) $

Smoothing strength adapts to document length — short documents are smoothed more — which is usually the right behavior; $mu approx 1500$–2500 works across collections. Query likelihood with Dirichlet smoothing performs comparably to BM25 and is the standard formulation in academic systems (Indri, Galago, Anserini's QL baseline). The framework extends naturally to relevance feedback via relevance models (RM3), a strong classical query-expansion method still used as a baseline today.

== Field Weighting: BM25F

Real documents have structure — title, body, anchor text, URL — and a title match should outweigh a body match. Naively computing BM25 per field and summing is wrong: it saturates tf _per field_, so a term appearing once in each of five fields scores far more than five times in one field, and IDF gets counted repeatedly.

BM25F (Robertson, Zaragoza & Taylor, 2004) instead builds a weighted *pseudo-frequency* across fields first, then saturates once. For field $f$ with weight $w_f$, length $|d_f|$, and average length $"avgdl"_f$:

$ tilde("tf")(t, d) = sum_f w_f dot ("tf"(t, d_f)) / (1 - b_f + b_f dot (|d_f|) / ("avgdl"_f)) $

$ "BM25F"(q, d) = sum_(t in q) "idf"(t) dot (tilde("tf")(t, d)) / (k_1 + tilde("tf")(t, d)) $

Each field gets its own $b_f$ (titles typically need little length normalization) and weight $w_f$ (anchor text and title weighted several times the body in web search). Elasticsearch's `combined_fields` query implements BM25F-style scoring; the older `multi_match` best-fields mode takes a per-field max instead.

== Static Signals

Query-dependent scores are combined with *query-independent* (static) document priors: popularity, spam score, URL depth, freshness — and, most famously, link analysis.

=== PageRank

PageRank (Brin & Page, 1998) models a random surfer on the web graph: at each step, with probability $d$ (the damping factor, classically 0.85) follow a uniformly random outlink, otherwise teleport to a random page. A page's PageRank is its stationary visit probability:

$ "PR"(p) = (1 - d) / N + d sum_(q -> p) ("PR"(q)) / (L(q)) $

where $L(q)$ is the outdegree of $q$ and the sum runs over pages linking to $p$. Equivalently, PageRank is the principal eigenvector of the damped transition matrix $G = d M + (1 - d) (1\/N) bold(1) bold(1)^top$; the teleport term makes $G$ irreducible and aperiodic, guaranteeing a unique stationary distribution and geometric convergence of power iteration at rate $d$ (roughly 50–100 iterations to good precision).

The intuition: a link is an endorsement, weighted by the endorser's own importance and diluted by how many endorsements it hands out. Refinements include topic-sensitive PageRank (teleport to a topical seed set), Personalized PageRank (teleport to a single user's pages — also used for recommendations and graph similarity), and TrustRank for spam demotion. HITS (Kleinberg, 1998) is the contemporaneous alternative, computing mutually reinforcing hub and authority scores on a query-specific subgraph.

=== Combining Signals

Classically, static scores enter as an additive prior on the ranking score or a multiplicative boost (Elasticsearch `function_score`); in modern stacks they are simply features for learning to rank. A practical detail: heavy-tailed signals like PageRank are log-transformed or quantile-bucketed before use.

== What Classical Models Still Buy You

- *Zero training data* and full interpretability: every score decomposes into per-term, per-field contributions.
- *Efficiency*: bounded per-term contributions enable WAND-family pruning; one index serves all queries.
- *Robustness out of domain*: exact lexical match cannot hallucinate; rare identifiers, code tokens, and product SKUs match exactly where embeddings blur.

The standard production answer is not BM25 _or_ neural, but BM25 as one leg of a hybrid (see _Neural Retrieval_) and as the candidate generator beneath learned rankers (see #xref("search-and-ir", "learning-to-rank", label: "Learning to Rank")).

== Further Reading

- Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. _Foundations and Trends in IR_.
- Ponte, J., & Croft, W. B. (1998). A language modeling approach to information retrieval. _SIGIR_.
- Zhai, C., & Lafferty, J. (2001). A study of smoothing methods for language models applied to ad hoc information retrieval. _SIGIR_.
- Robertson, S., Zaragoza, H., & Taylor, M. (2004). Simple BM25 extension to multiple weighted fields. _CIKM_.
- Brin, S., & Page, L. (1998). The anatomy of a large-scale hypertextual web search engine. _WWW_.
- Lin, J., Crane, M. et al. (2016). Toward reproducible baselines: the open-source IR reproducibility challenge. _ECIR_.

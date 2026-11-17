#import "../template.typ": xref

= Distributed Search Architecture <distributed-search>

A single machine can hold an inverted index of a few hundred million documents and answer a few thousand queries per second. Web search, e-commerce catalogues, log analytics and enterprise search exceed that by orders of magnitude in index size, query rate, or both, while still answering in a few hundred milliseconds and staying available through machine failures. Distributing search means deciding how to split the index, how to fan queries out and merge results, how to keep scores consistent across pieces, how to make new documents searchable within seconds, and how to keep the tail latency of a query that touches hundreds of machines under control. This chapter covers document and term partitioning, scatter-gather query execution with a C++ calculation of how many results each shard must return, global statistics, replication and near-real-time indexing, tiering and selective search, caching, and shard sizing.

*See also:* #xref("search-and-ir", "inverted-indexes", label: "Inverted Indexes") (Lucene segments), #xref("search-and-ir", "query-processing", label: "Query Processing") (top-k pruning), #xref("search-and-ir", "vector-search", label: "Vector Search"), #xref("distributed-systems", "consensus-deep-dive", label: "Consensus Deep Dive"), #xref("performance-engineering", "concurrency-performance", label: "Concurrency Performance") (tail latency and fan-out).

== Partitioning the Index

There are two ways to split an inverted index across $S$ machines.

#table(
  columns: 3,
  [*Aspect*], [*Document partitioning*], [*Term partitioning*],
  [Each shard holds], [A complete index of a subset of documents], [The full posting lists of a subset of terms],
  [A query touches], [Every shard], [Only the shards holding its terms],
  [Work per query], [Every shard processes every query term over its documents], [Each term's posting list is processed once],
  [Load balance], [Even, if documents are spread randomly], [Skewed: popular terms create hot shards],
  [Multi-term scoring], [Local to each shard], [Posting lists must be shipped between shards to intersect],
  [Updates], [A new document goes to one shard], [A new document touches one shard per distinct term],
  [Used by], [Essentially all production engines], [Research, and some specialised systems],
)

Term partitioning looks attractive because each query touches few machines, but it loses in practice: query term frequencies follow a power law, so the shards holding common terms saturate, and conjunctive queries must move large posting lists across the network. Document partitioning wastes some work (every shard evaluates every term) but parallelises evenly and keeps each shard an ordinary, self-contained search engine. Hybrid schemes exist, but nearly all production systems, from Elasticsearch and Solr to web search engines, partition by document.

Documents are usually assigned to shards by hashing their identifier, which spreads them evenly. *Custom routing* (by tenant, by user, by date) puts related documents on one shard so that queries scoped to that key touch a single shard; it trades even load for fewer fan-outs, and it creates hot shards when one tenant is much larger than the rest.

== Scatter-Gather

A query arrives at a *broker* (coordinator), which sends it to one replica of every shard, waits for their local top results, and merges them. With document partitioning, the global top $k$ is contained in the union of the shards' local top $k$, so asking each shard for $k$ results is always correct. For deep pages ($k = 1000$ on page 50) with hundreds of shards, that transfers and merges $S k$ results to return $k$, which is why engines limit deep pagination and offer cursors (*search after* a given sort key) instead.

If documents are spread randomly, each shard holds only about $k \/ S$ of the global top $k$, so asking each for $k$ is wasteful. The number of global top-$k$ results on one shard is approximately binomial with parameters $k$ and $1 \/ S$, and a per-shard request size $k'$ can be chosen so that the probability that any shard held more than $k'$ of them is below a small $epsilon$:

```cpp
#include <cmath>

// P(Binomial(n, p) > x), summed in log space for stability.
double binomial_upper_tail(int n, double p, int x) {
    double total = 0.0;
    for (int i = x + 1; i <= n; ++i) {
        const double log_term = std::lgamma(n + 1.0) - std::lgamma(i + 1.0) -
                                std::lgamma(n - i + 1.0) + i * std::log(p) +
                                (n - i) * std::log1p(-p);
        total += std::exp(log_term);
    }
    return total;
}

// Smallest per-shard result count k' such that, with documents assigned to `shards`
// shards uniformly at random, the union bound on any shard holding more than k'
// of the global top k is at most eps.
int per_shard_k(int k, int shards, double eps) {
    const double p = 1.0 / shards;
    for (int kp = 0; kp <= k; ++kp)
        if (shards * binomial_upper_tail(k, p, kp) <= eps) return kp;
    return k;
}
```

For $k = 100$ over 10 shards with $epsilon = 10^(-3)$, each shard needs to return only 23 results instead of 100; for $k = 1000$ over 100 shards, 26 instead of 1000. The broker can detect the rare case where a shard's last returned score is still above the merged $k$-th score and fetch more from that shard, keeping results exact.

*Query-then-fetch* splits the work into two phases. In the query phase, shards return only document identifiers, scores and sort values; the broker merges them; in the fetch phase it retrieves stored fields, snippets and highlights only for the final $k$ documents from the shards that hold them. Snippet generation is expensive, so doing it for 10 documents rather than $10 S$ saves a great deal.

== Tail Latency at Fan-Out

A query that touches $S$ shards is as slow as its slowest shard. If each shard independently exceeds 100 ms one time in a hundred, a query over 100 shards exceeds it with probability $1 - 0.99^(100) approx 63$ percent. Large search systems therefore treat the tail as a design problem:

- *Replicas* for every shard let the broker send each sub-query to the least loaded or fastest replica, using adaptive replica selection based on observed response times and queue lengths.
- *Hedged requests*: if a shard has not answered within its typical p95 time, send the same sub-query to another replica and use whichever answers first.
- *Partial results*: under a deadline, return results from the shards that answered, flagged as partial. For web search, missing a small fraction of the index is often better than being slow.
- *Uniform shard sizes*: one oversized shard, or a shard sharing a machine with a heavy indexing workload, sets the latency for every query.

== Global Statistics

Scoring functions such as BM25 use collection statistics: inverse document frequency (IDF), average document length. Each shard computes them from its own documents. With random document assignment and large shards, local statistics are close to global ones. With custom routing, small shards, or time-based indexes (a shard per day of logs), they diverge, and the same document gets different scores on different shards, so the merged ranking depends on where documents happen to live.

Remedies: a preliminary round trip that collects term statistics from all shards before scoring (Elasticsearch's `dfs_query_then_fetch`), periodically distributed global statistics, or ranking features that do not depend on collection statistics. In learned ranking stacks (#xref("search-and-ir", "learning-to-rank", label: "Learning to Rank")), the first-stage score only selects candidates, so small inconsistencies matter less than in pure BM25 systems.

== Replication and Indexing

Each shard has a *primary* that accepts writes and one or more *replicas*. Two replication styles exist:

- *Document replication*: the primary forwards each indexing operation to the replicas, which index it themselves. Every copy pays the full indexing CPU cost, but replicas are immediately consistent at the operation level.
- *Segment replication*: only the primary indexes, and replicas copy the immutable segment files it produces. Indexing CPU is paid once, at the cost of replicas lagging by the copy interval.

Segment-based engines make new documents searchable through *near-real-time* (NRT) indexing. New documents are buffered in memory; a *refresh* (every second by default in Elasticsearch) writes the buffer as a new small segment and opens it for search, without the cost of a durable commit. Durability comes from a *transaction log* (translog) that records every operation and is replayed after a crash; periodic *flushes* perform full commits and truncate the log. Small segments are continuously merged into larger ones by a *merge policy* (tiered merging by size), the LSM-tree pattern that keeps the number of segments per shard, and so the per-query overhead, bounded. Deletions and updates mark documents as deleted in existing segments and are reclaimed by merges.

The refresh interval trades freshness against throughput: each refresh creates a segment that must later be merged. Bulk indexing typically disables refresh until the load completes. Systems that need in-place updates of numeric fields at high rates (prices, stock, counters) use designs that keep such fields in mutable, memory-resident structures outside the immutable segments, as Vespa does.

== Tiering and Selective Search

Searching every shard for every query is expensive. Two ideas reduce the work:

- *Tiered indexes* place the documents most likely to be relevant (high quality, popular, recent) in a small first tier that every query searches. Only if the first tier returns too few good results does the query proceed to the larger lower tiers. Web search engines have long used tiers to serve most queries from a fraction of the index.
- *Selective search* partitions documents by *topic* rather than randomly, by clustering the collection into topical shards. A *resource selection* step estimates which shards are likely to contain relevant documents for a query (from a small sample index of each shard or from shard-level term statistics) and searches only those. Kulkarni and Callan showed that searching a few topical shards matches the effectiveness of exhaustive search at a fraction of the cost, at the price of uneven load across shards, since popular topics concentrate traffic.

== Caching

Query streams are highly skewed: a small fraction of distinct queries accounts for a large fraction of traffic. Caching works at several levels:

- *Result caches* store final result pages for frequent queries at the broker. Hit rates of 30 to 50 percent are typical for web query streams; the rest is a long tail of unique queries that no cache can serve. Results must expire or be invalidated as the index changes, and freshness-sensitive queries need short lifetimes.
- *Posting list and filter caches* at the shards keep frequently used posting lists, filter bit sets (for example `status:published`) and decoded blocks in memory. They help even unique queries, because unique queries share terms and filters.
- *Operating system page cache*: Lucene relies on memory-mapped index files, so the most important cache is free RAM for the page cache. A shard whose index does not fit in memory falls off a performance cliff when its working set exceeds RAM.

== Shard Sizing and Rebalancing

Shard count is a capacity decision that is hard to change later, because document-to-shard assignment depends on it. Too few shards limit parallelism and make each shard slow to recover and move; too many add per-shard overhead (memory for segment metadata, scatter-gather fan-out, cluster state size). Common guidance for Elasticsearch is shards of roughly 10 to 50 GB. Strategies for growth include *oversharding* (more shards than machines, so shards can move as the cluster grows), split and shrink operations that rewrite an index with a different shard count, and *time-based indexes* for append-only data such as logs (one index per day, rolled over by size or age, with old indexes moved to cheaper hardware and eventually deleted).

When a node fails, its shards' replicas are promoted and new replicas are rebuilt elsewhere, which copies whole shards across the network. Large shards lengthen recovery and the window of reduced redundancy, another reason to bound shard size.

== Distributed Vector Search

Dense retrieval adds approximate nearest-neighbour indexes (#xref("search-and-ir", "vector-search", label: "Vector Search")) to each shard. The same scatter-gather pattern applies: each shard searches its graph or cluster index for its local top $k'$, and the broker merges. Two differences matter. Graph indexes such as HNSW are memory-hungry and slow to build, so vector shards are sized by RAM rather than disk. And approximate search per shard means recall errors compound: each shard's search misses some true neighbours, and the merged result inherits all those misses, so per-shard search parameters must be tuned for the recall of the merged result, not of one shard.

== Pitfalls

- Too many small shards, whose fixed per-shard costs dominate query latency and cluster overhead.
- Custom routing that creates one shard many times larger than the rest.
- Inconsistent scores across shards from local statistics, noticed only as unstable rankings.
- Deep pagination over many shards.
- Index sizes that exceed the page cache, turning every query into disk reads.
- Refresh intervals too short for the indexing rate, causing merge storms.

== Further Reading

Barroso, L. A., Dean, J. and Hölzle, U. (2003). "Web Search for a Planet: The Google Cluster Architecture." IEEE Micro.

Dean, J. (2009). "Challenges in Building Large-Scale Information Retrieval Systems." WSDM keynote, reference.

Dean, J. and Barroso, L. A. (2013). "The Tail at Scale." Communications of the ACM.

Moffat, A. et al. (2007). "A Pipelined Architecture for Distributed Text Query Evaluation." Information Retrieval.

Kulkarni, A. and Callan, J. (2015). "Selective Search: Efficient and Effective Search of Large Textual Collections." ACM Transactions on Information Systems.

Baeza-Yates, R. et al. (2007). "The Impact of Caching on Search Engines." SIGIR.

Risvik, K. M., Aasheim, Y. and Lidal, M. (2003). "Multi-Tier Architecture for Web Search Engines." LA-WEB.

Cambazoglu, B. B. and Baeza-Yates, R. (2015). _Scalability Challenges in Web Search Engines_. Morgan and Claypool.

Elastic (2024). "Size Your Shards." Elasticsearch documentation.

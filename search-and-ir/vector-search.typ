= Vector Search

Dense retrieval reduces matching to nearest-neighbor search in $RR^d$: given a query vector, find the $k$ closest of $N$ document vectors. Exact search is a linear scan — fine to a few hundred thousand vectors, hopeless at billions — so production systems use *approximate* nearest neighbor (ANN) indexes that trade a little recall for orders of magnitude in speed. This chapter covers the distance metrics, the two dominant index families (graph-based HNSW and cluster-based IVF), quantization for memory, and the operational concerns — filtering, updates, and the recall/latency/memory triangle — that decide real deployments.

*See also:* _Neural Retrieval_ (where the vectors come from), _Inverted Indexes_ (the lexical analogue; IVF is the closest structural cousin), _RAG and Search Systems_ (vector search as RAG's backbone), _Evaluation_ (recall\@k against exact search as the index metric).

== Distances and the Curse of Dimensionality

Common metrics: Euclidean distance, inner product (MIPS — maximum inner product search), and cosine similarity. For unit-normalized vectors all three induce the same ranking, since $|x - y|^2 = 2 - 2 x dot y$; most embedding models are trained for cosine and normalized at index time. Unnormalized inner product is genuinely different (it is not a metric — a vector may not be its own nearest neighbor) and is handled by reductions that append a dimension to convert MIPS to Euclidean NN.

In high dimensions, distances concentrate: the gap between the nearest and farthest point shrinks, and space-partitioning trees (k-d trees) degrade to near-linear scans beyond $d approx 20$. Real embeddings live on much lower-dimensional manifolds than their ambient $d$ of 384–3072, which is why ANN works at all — but no method offers good worst-case guarantees; everything is empirical, measured as recall\@k versus queries per second.

== Locality-Sensitive Hashing

LSH (Indyk & Motwani, 1998) hashes points so that near points collide with higher probability than far ones — for cosine, random hyperplane signs (Charikar, 2002): $h(x) = "sign"(r dot x)$, concatenated into multi-bit keys across multiple tables. LSH has clean sublinear theory and supports streaming inserts, but in practice needs many tables (memory) for high recall and is consistently outperformed by graph and IVF methods on the ANN-Benchmarks suite. Its lasting legacy is binary sketching: the related SimHash is used for near-duplicate detection at crawl time, and MinHash for set similarity.

== Graph Indexes: HNSW

Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2018) are the dominant high-recall index. The structure is a layered proximity graph:

- Layer 0 contains all points, each linked to about $2 M$ near neighbors ($M$ ~ 16–48); upper layers contain exponentially fewer points (each node's top layer is drawn geometrically), forming an expressway hierarchy reminiscent of a skip list.
- *Search*: start at the top layer's entry point, greedily descend — at each layer move to the neighbor closest to the query until no improvement, then drop a layer. At layer 0, run a best-first beam search keeping a candidate set of size `efSearch`; larger beams give higher recall at higher cost.
- *Insertion*: search for the new point's neighbors with beam `efConstruction`, connect, and prune each affected node's adjacency back to $M$ with a *diversity heuristic* (prefer neighbors not already close to each other), which keeps the graph navigable across clusters.

HNSW reaches 95–99% recall\@10 at sub-millisecond latency on million-scale sets and underlies Lucene/Elasticsearch dense vector search, pgvector, Qdrant, Weaviate, Milvus, and FAISS's `IndexHNSW`. Costs: memory — the full vectors plus \~$2 M$ 4-byte links per point must effectively stay in RAM, and construction is expensive ($O(N log N)$ with large constants). Deletes are awkward (tombstones degrade the graph until rebuild). DiskANN (Subramanya et al., 2019) adapts the idea to SSD: a flat Vamana graph on disk with compressed vectors in RAM to guide traversal, serving billion-scale sets on a single node.

== Cluster Indexes: IVF

The inverted file (IVF) approach mirrors the inverted index. Offline, cluster the corpus with k-means into `nlist` partitions (typically $approx sqrt(N)$ to $4 sqrt(N)$); each vector is assigned to its nearest centroid's posting list. At query time, compare the query to all centroids, visit only the `nprobe` nearest lists, and scan them. Recall is controlled by `nprobe` (1–5% of lists is common); the failure mode is a true neighbor sitting just across a Voronoi boundary of an unprobed cell.

IVF scans are memory-bandwidth-bound, which is why IVF is almost always paired with quantization — and why the combined IVF-PQ is the standard billion-scale configuration (FAISS `IVF...,PQ...`, ScaNN's tree-AH, Milvus IVF indexes). Compared to HNSW: cheaper to build, trivially shardable, lower memory, but typically lower recall at the same latency on smaller sets.

== Quantization

At $d = 768$ float32, one billion vectors is \~3 TB — compression is not optional.

=== Scalar and Binary Quantization

Per-dimension reduction to int8 (\~4$times$, usually under 1% recall loss) or to 1 bit per dimension (32$times$, with Hamming-distance scanning and exact re-scoring of survivors). Increasingly the default cheap option (Lucene int8/binary, pgvector halfvec, Matryoshka-style dimension truncation as a complementary trick).

=== Product Quantization

PQ (Jégou, Douze & Schmid, 2011) splits each vector into $m$ subvectors (e.g., $m = 96$ for $d = 768$) and quantizes each against its own 256-centroid codebook, giving $m$ bytes per vector — 32 bytes can represent a 768-dim vector, a 96$times$ compression. Distances are computed without decompression via lookup tables: precompute the query's distance to all 256 centroids per subspace ($256 m$ values), then each database vector's approximate distance is $m$ table lookups (*asymmetric distance computation*; SIMD-friendly 4-bit variants reach billions of comparisons per second). In IVF-PQ, vectors are quantized as *residuals* from their cell centroid, which tightens the codebooks considerably. OPQ learns a rotation that balances variance across subspaces before quantization; anisotropic quantization (ScaNN; Guo et al., 2020) weights errors that affect high-scoring inner products more. Quantization error caps achievable recall, so production systems *re-rank*: fetch the top few hundred by compressed distance, then re-score with full-precision vectors fetched from disk.

== Filtered and Hybrid Search

Real queries carry predicates: `category = "shoes" AND price < 100`, tenant isolation, ACLs. Strategies:

- *Post-filtering*: ANN first, filter after — breaks badly with selective filters (retrieve 100, keep 2).
- *Pre-filtering*: compute the allowed bitmap first, then search only matching vectors — exact for IVF (skip non-matching postings), but naive graph traversal restricted to a sparse allowed set disconnects; filtered-HNSW implementations (Qdrant, Weaviate, Vespa) traverse the full graph while only _scoring_ allowed nodes, or fall back to brute force below a cardinality threshold.
- *Filter-aware construction* (e.g., partitioning by tenant) when predicates are known and skewed.

Engines differ more on filtering quality than on raw ANN speed; it is the first thing to benchmark with production-shaped predicates.

== Operations

- *Recall measurement*: maintain a ground-truth set from exact search over a sample; track recall\@k continuously, since drift in the embedding distribution silently degrades a trained IVF/PQ index (centroids and codebooks go stale and need retraining).
- *Updates*: HNSW inserts are fine, deletes rot the graph; IVF lists append cheaply but cluster balance decays. Most systems adopt Lucene-style immutable segments with background merges, accepting per-segment search fan-out.
- *Sizing*: rough RAM per vector — HNSW float32: $4 d + 8 M$ bytes; IVF-PQ: $m$ bytes plus list overhead. The recall/latency/memory triangle is real: pick two.

== Further Reading

- Malkov, Y., & Yashunin, D. (2018). Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. _IEEE TPAMI_.
- Jégou, H., Douze, M., & Schmid, C. (2011). Product quantization for nearest neighbor search. _IEEE TPAMI_.
- Subramanya, S. J. et al. (2019). DiskANN: fast accurate billion-point nearest neighbor search on a single node. _NeurIPS_.
- Guo, R. et al. (2020). Accelerating large-scale inference with anisotropic vector quantization. _ICML_. (ScaNN)
- Indyk, P., & Motwani, R. (1998). Approximate nearest neighbors: towards removing the curse of dimensionality. _STOC_.
- Aumüller, M., Bernhardsson, E., & Faithfull, A. (2020). ANN-Benchmarks: a benchmarking tool for approximate nearest neighbor algorithms. _Information Systems_.
- Douze, M. et al. (2024). The Faiss library. _arXiv:2401.08281_.

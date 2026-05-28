= Vector and Similarity Search

Embedding models (BERT, CLIP, text-embedding-3) map text, images, and other data into dense vectors in $RR^d$ (typically $d$ = 768–3072). *Nearest-neighbor search* — finding the $k$ vectors closest to a query vector — is the performance-critical operation in retrieval-augmented generation, semantic search, and recommendation systems.

*See also:* _Query Optimization_, _Hardware-Aware Database Design_, _LLM volume_ (retrieval-augmented generation — ANN is the inner loop of RAG)

== Distance Metrics

#table(
  columns: (auto, auto, auto),
  [*Metric*], [*Formula*], [*When to use*],
  [L2 (Euclidean)],  [$||u - v||_2 = sqrt(sum_i (u_i - v_i)^2)$],   [Raw embeddings],
  [Inner product],   [$u · v = sum_i u_i v_i$],                       [Normalized vectors (= cosine)],
  [Cosine],          [$1 - (u · v) / (||u|| ||v||)$],                 [Direction matters, not magnitude],
  [Hamming],         [$sum_i |u_i - v_i|$ on bits],                   [Binary codes],
)

*Normalize at indexing time:* if vectors are L2-normalized, cosine similarity equals inner product — enabling faster SIMD-friendly inner product computation.

```python
import numpy as np

def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)

# After normalization: cosine(u, v) = dot(u, v)
u = l2_normalize(np.random.randn(1, 128).astype(np.float32))
v = l2_normalize(np.random.randn(100, 128).astype(np.float32))
scores = (u @ v.T)[0]   # inner products = cosine similarities
top_k  = np.argsort(scores)[-10:][::-1]
```

== Exact Search

Brute-force $k$-NN: compute distance to every vector.

```python
def exact_knn(query: np.ndarray, db: np.ndarray, k: int) -> np.ndarray:
    # db: [N, d], query: [d]
    dists = np.sum((db - query) ** 2, axis=1)   # L2^2
    return np.argpartition(dists, k)[:k]          # O(N) partial sort

# For d=768, N=1M: ~1.5 billion FLOPs per query → ~100ms on CPU, ~5ms on GPU
# Acceptable for N < 100K; too slow for production at scale
```

== HNSW (Hierarchical Navigable Small World)

HNSW (Malkov & Yashunin 2018) is the dominant ANN index for high-recall, low-latency search. It builds a multi-layer graph where:
- Layer 0: all vectors, high connectivity.
- Higher layers: subset of vectors, long-range "express lanes".

*Search:* start at the top layer's entry point, greedily descend toward the query, then explore layer 0 with beam search.

```
HNSW structure (M=5, ef_construction=200):
Layer 2:  ●                     ← 1-2 nodes, long-range connections
Layer 1:  ● ─ ● ─ ●           ← log(N) nodes
Layer 0:  all N vectors, M bidirectional edges each
```

```python
import hnswlib

d   = 128            # embedding dimension
N   = 1_000_000      # number of vectors

# Build index
index = hnswlib.Index(space='l2', dim=d)
index.init_index(max_elements=N,
                 ef_construction=200,  # beam width during build (quality ↑, speed ↓)
                 M=16)                 # max edges per node per layer
index.add_items(np.random.randn(N, d).astype(np.float32))

# Query
index.set_ef(50)     # beam width during search (quality ↑, latency ↑)
query = np.random.randn(1, d).astype(np.float32)
labels, distances = index.knn_query(query, k=10)
# Typical: >99% recall@10 at 1-2ms on 1M vectors, single thread
```

*Complexity:* $O(log N)$ per query (hops through layers), $O(N log N)$ build, $O(N dot M dot d)$ memory.

*HNSW hyperparameters:*

#table(
  columns: (auto, auto, auto),
  [*Parameter*], [*Effect*], [*Typical range*],
  [`M`],               [Edges per node — recall vs memory], [8–64],
  [`ef_construction`], [Build-time beam width — quality vs build speed], [100–500],
  [`ef` (search)],     [Search beam width — recall vs latency], [50–500],
)

== IVF (Inverted File Index)

Partition vectors into $k$ clusters using k-means. At query time, only search the nearest $n_"probe"$ clusters.

```python
import faiss

d      = 128
N      = 1_000_000
nlist  = 1000      # number of IVF cells (Voronoi partitions)
nprobe = 10        # cells to search at query time

quantizer = faiss.IndexFlatL2(d)        # coarse quantizer (exact)
index     = faiss.IndexIVFFlat(quantizer, d, nlist)
index.train(vectors)                    # k-means to build centroids
index.add(vectors)

index.nprobe = nprobe
distances, ids = index.search(query, k=10)
# nprobe/nlist tradeoff: nprobe=10/1000 → search 1% of data
# recall@10 ≈ 95% at nprobe=10; ≈ 99% at nprobe=50
```

*When IVF wins:* smaller memory footprint than HNSW (no graph edges); parallelizable; amenable to filtering. When HNSW wins: better recall at same latency; no training phase.

== Product Quantization (PQ)

Compress each $d$-dimensional vector into a compact code by splitting it into $m$ sub-vectors and quantizing each independently.

```
d=128, m=8 sub-vectors of dim 16:
  Each sub-space has k=256 centroids → 8-bit code per sub-vector
  Total: 8 bytes per vector (vs 512 bytes for float32) — 64× compression

Distance approximation:
  dist(q, v) ≈ Σ_j dist(q_j, centroids_j[code_j[v]])
  Precompute lookup table: dist(q_j, centroids_j[*]) for all j, k
  → Distance = sum of 8 table lookups (very fast)
```

```python
# FAISS IVFPQ: IVF partitioning + PQ compression
index = faiss.IndexIVFPQ(quantizer, d,
                          nlist=1000,
                          M=8,       # number of sub-vectors
                          nbits=8)   # bits per sub-vector code (256 centroids)
index.train(vectors)
index.add(vectors)
index.nprobe = 10
distances, ids = index.search(query, k=10)
# Memory: N * M bytes (vs N * d * 4 for exact)
# Recall: ~90% at nprobe=10; tune M and nbits for quality/memory tradeoff
```

== DiskANN

For billion-scale datasets that don't fit in RAM, DiskANN (Subramanya et al. 2019) stores the graph on SSD and uses a compressed in-memory cache for fast navigation.

*Key insight:* the first few hops of graph search (top HNSW layers) access a small fraction of nodes. Cache those in RAM with PQ compression; read full vectors from SSD only for final candidates.

```
DiskANN architecture:
  RAM: PQ-compressed full graph (for routing) + full vectors of hot nodes
  SSD: full-precision vectors + adjacency lists

Query:
  1. Navigate graph using PQ distances (RAM-resident)
  2. At final hop candidates: fetch full vectors from SSD (few random reads)
  3. Rerank with exact distances

Result: 5ms latency for 1B vectors at 95% recall on commodity SSD
```

== pgvector: Vector Search in PostgreSQL

```sql
-- Install pgvector and create a table with vector column
CREATE EXTENSION vector;

CREATE TABLE embeddings (
    id      BIGSERIAL PRIMARY KEY,
    content TEXT,
    vec     vector(1536)    -- OpenAI text-embedding-3-small dimension
);

-- Insert embeddings
INSERT INTO embeddings (content, vec)
VALUES ('Hello world', '[0.1, 0.2, ...]'::vector);

-- Exact k-NN (seq scan, no index)
SELECT id, content, vec <-> '[0.1, 0.2, ...]'::vector AS dist
FROM   embeddings
ORDER  BY dist
LIMIT  10;

-- Build HNSW index
CREATE INDEX ON embeddings USING hnsw (vec vector_l2_ops)
    WITH (m = 16, ef_construction = 64);

-- Approximate k-NN (uses index)
SET hnsw.ef_search = 100;
SELECT id, content, vec <-> query_vec AS dist
FROM   embeddings
ORDER  BY dist
LIMIT  10;

-- Filtered ANN: combine vector search with SQL predicates
SELECT id, content, vec <-> query_vec AS dist
FROM   embeddings
WHERE  content ILIKE '%database%'
ORDER  BY dist
LIMIT  10;
-- Note: HNSW index can't efficiently pre-filter; consider IVF for pre-filtered ANN
```

== Recall–Latency Tradeoff

```python
# Measure recall@k for different ef values
import numpy as np, hnswlib, time

true_knn  = exact_knn(queries, db, k=10)         # ground truth

for ef in [10, 20, 50, 100, 200]:
    index.set_ef(ef)
    t0 = time.perf_counter()
    results, _ = index.knn_query(queries, k=10)
    latency = (time.perf_counter() - t0) / len(queries) * 1000  # ms

    recalls = [len(set(r) & set(g)) / 10
               for r, g in zip(results, true_knn)]
    print(f"ef={ef:4d}  recall={np.mean(recalls):.3f}  latency={latency:.2f}ms")
# ef= 10  recall=0.921  latency=0.42ms
# ef= 50  recall=0.989  latency=1.21ms
# ef=200  recall=0.999  latency=4.80ms
```

== References

Malkov, Y., Yashunin, D. (2018). "Efficient and Robust Approximate Nearest Neighbor Search Using HNSW." TPAMI.

Subramanya, S., Devvrit, F., Simhadri, H., Krishnaswamy, R., Kadekodi, R. (2019). "DiskANN: Fast Accurate Billion-Point Nearest Neighbor Search on a Single Node." NeurIPS.

Jégou, H., Douze, M., Schmid, C. (2011). "Product Quantization for Nearest Neighbor Search." TPAMI.

Johnson, J., Douze, M., Jégou, H. (2021). "Billion-Scale Similarity Search with GPUs." IEEE Big Data.

Simhadri, H. et al. (2022). "Results of the NeurIPS'21 Challenge on Billion-Scale Approximate Nearest Neighbor Search." arXiv:2205.03763.

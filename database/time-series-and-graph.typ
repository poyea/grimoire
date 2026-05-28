= Time-Series and Graph Databases

Time-series and graph data have access patterns that don't fit relational models well. Specialized engines exploit domain structure for 10–100× better performance on their target workloads.

*See also:* _Storage Engines_, _Column Stores and Vectorized Execution_, _Streaming and Incremental Computation_

== Time-Series Data

A *time series* is a sequence of (timestamp, value) pairs. Characteristics:
- Writes are almost always appends (new measurements, never updates to old ones).
- Reads are range queries by time: "all values in the last 24 hours".
- High cardinality: millions of distinct metric names/tags (e.g., one per host+service+metric).
- Compression is critical: adjacent values are often similar (temperature drifts slowly).

=== Time-Series Compression

*Delta-of-delta encoding* (Gorilla/Facebook, Pelkonen et al. 2015): timestamps are nearly regular; values repeat or drift slowly.

```python
# Gorilla timestamp encoding (simplified)
def encode_timestamps(timestamps: list[int]) -> list[int]:
    """
    Timestamps are in seconds. Most deltas are constant (e.g., 60s).
    Store: first ts, then Δ, then Δ-of-Δ (usually 0).
    """
    if not timestamps:
        return []
    result   = [timestamps[0]]
    prev_t   = timestamps[0]
    prev_dod = 0
    prev_d   = 0
    for t in timestamps[1:]:
        d   = t - prev_t
        dod = d - prev_d
        result.append(dod)   # mostly 0 → 1 bit ("no change")
        prev_t   = t
        prev_d   = d
    return result

# 1-minute interval timestamps: [1700000000, 1700000060, 1700000120, ...]
# deltas all = 60, delta-of-delta all = 0 → compresses to ~1 bit per timestamp

# XOR encoding for float values (Gorilla)
def xor_encode_floats(values: list[float]) -> list[int]:
    import struct
    result = []
    prev_bits = 0
    for v in values:
        bits = struct.unpack('>Q', struct.pack('>d', v))[0]
        xor  = bits ^ prev_bits
        result.append(xor)   # leading/trailing zeros in XOR → short codes
        prev_bits = bits
    return result
```

Gorilla achieves ~1.37 bytes/sample vs ~16 bytes/sample uncompressed (10× compression).

=== InfluxDB / TimescaleDB Data Model

```sql
-- TimescaleDB (PostgreSQL extension): hypertable partitioned by time
CREATE TABLE cpu_metrics (
    time        TIMESTAMPTZ NOT NULL,
    host        TEXT,
    region      TEXT,
    usage_cpu   DOUBLE PRECISION,
    usage_mem   DOUBLE PRECISION
);

SELECT create_hypertable('cpu_metrics', 'time', chunk_time_interval => INTERVAL '1 day');
-- Each chunk is one day of data; old chunks can be compressed or tiered to S3

-- Enable native compression
ALTER TABLE cpu_metrics SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_segmentby = 'host'
);
SELECT add_compression_policy('cpu_metrics', INTERVAL '7 days');

-- Continuous aggregate: pre-compute 1-hour rollups
CREATE MATERIALIZED VIEW cpu_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS hour,
    host,
    AVG(usage_cpu)  AS avg_cpu,
    MAX(usage_cpu)  AS max_cpu
FROM cpu_metrics
GROUP BY 1, 2
WITH NO DATA;

SELECT add_continuous_aggregate_policy('cpu_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset   => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');
```

=== Prometheus Data Model

Prometheus stores time series as `metric_name{label=value,...}` → stream of (timestamp, float64) pairs.

```python
# PromQL: query language for time-series
# Rate of HTTP requests per second (5-minute window)
# rate(http_requests_total{job="api", status="200"}[5m])

# Equivalent in Python against Prometheus HTTP API:
import requests

def promql_query(query: str, timestamp=None) -> dict:
    params = {"query": query}
    if timestamp:
        params["time"] = timestamp
    return requests.get("http://prometheus:9090/api/v1/query",
                        params=params).json()

result = promql_query(
    'rate(http_requests_total{job="api", status="200"}[5m])'
)
for series in result["data"]["result"]:
    print(series["metric"], series["value"])
```

*Prometheus storage (TSDB):* 2-hour in-memory chunks; compacted to Parquet-like blocks on disk. Blocks have a min/max timestamp envelope — range queries skip irrelevant blocks entirely.

== Graph Databases

A *property graph* consists of:
- *Nodes:* entities with labels (type) and properties (key-value).
- *Edges:* directed, typed relationships between nodes with properties.

```
Graph: social network
  (Alice:Person {age:30}) -[:FRIENDS]-> (Bob:Person {age:28})
  (Alice:Person) -[:POSTED]-> (Post:Post {text:"Hello"})
  (Bob:Person) -[:LIKED]-> (Post:Post)
```

*Relational representation is painful:*

```sql
-- "Friends of friends of Alice who liked a post by Alice" in SQL
SELECT DISTINCT p3.name
FROM people p1
JOIN friendships f1 ON f1.user_id = p1.id
JOIN friendships f2 ON f2.user_id = f1.friend_id
JOIN people p3 ON p3.id = f2.friend_id
JOIN posts po ON po.author_id = p1.id
JOIN likes li ON li.post_id = po.id AND li.user_id = p3.id
WHERE p1.name = 'Alice';
-- 5 joins for a 2-hop query. For k-hop: O(k) joins.
```

*Cypher (Neo4j) for the same query:*

```cypher
MATCH (alice:Person {name: "Alice"})-[:FRIENDS*2]->(fof:Person),
      (alice)-[:POSTED]->(post:Post)<-[:LIKED]-(fof)
RETURN DISTINCT fof.name
```

=== Graph Storage: Adjacency List

Each node stores a pointer to its adjacency list. Traversal = follow pointers, not table joins.

```c
// Compressed Sparse Row (CSR) format: memory-efficient adjacency list
typedef struct {
    uint64_t *row_ptr;    // row_ptr[i] = start index of node i's neighbors in col[]
    uint64_t *col;        // col[row_ptr[i]..row_ptr[i+1]] = neighbors of node i
    uint64_t  n_nodes;
    uint64_t  n_edges;
} CSRGraph;

// Iterate over neighbors of node v:
void visit_neighbors(const CSRGraph *g, uint64_t v) {
    for (uint64_t j = g->row_ptr[v]; j < g->row_ptr[v + 1]; j++) {
        uint64_t neighbor = g->col[j];
        // process neighbor
    }
}
// No join! Pure pointer chasing.
```

=== BFS and Shortest Path

```python
from collections import deque

def bfs(graph: dict, start: str, end: str) -> list[str] | None:
    """Shortest path in an unweighted graph."""
    visited = {start: None}   # node → parent
    queue   = deque([start])
    while queue:
        node = queue.popleft()
        if node == end:
            # Reconstruct path
            path = []
            while node:
                path.append(node)
                node = visited[node]
            return path[::-1]
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited[neighbor] = node
                queue.append(neighbor)
    return None

# Cypher equivalent:
# MATCH p = shortestPath((a:Person {name: "Alice"})-[*]-(b:Person {name: "Carol"}))
# RETURN p
```

=== Weighted Shortest Path: Dijkstra and Bellman-Ford

BFS gives the fewest-hops path; weighted edges (latency, distance, cost) require Dijkstra (non-negative weights) or Bellman-Ford (allows negatives, detects negative cycles). Both are first-class operations in graph DBs: Neo4j's GDS library exposes `gds.shortestPath.dijkstra.stream`, and pgRouting in PostgreSQL ships both.

*Dijkstra* — $O((V + E) log V)$ with a binary heap:

```cpp
#include <limits>
#include <queue>
#include <utility>
#include <vector>

using NodeId = std::uint32_t;
using Weight = double;
struct Edge { NodeId to; Weight w; };

std::vector<Weight> dijkstra(const std::vector<std::vector<Edge>>& adj,
                             NodeId src) {
    std::vector<Weight> dist(adj.size(), std::numeric_limits<Weight>::infinity());
    using PQE = std::pair<Weight, NodeId>;
    std::priority_queue<PQE, std::vector<PQE>, std::greater<>> pq;
    dist[src] = 0.0;
    pq.emplace(0.0, src);
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;                  // stale entry
        for (const auto& e : adj[u]) {
            Weight nd = d + e.w;
            if (nd < dist[e.to]) {
                dist[e.to] = nd;
                pq.emplace(nd, e.to);
            }
        }
    }
    return dist;
}
```

*Bellman-Ford* — $O(V dot E)$, tolerates negative weights, detects negative cycles in one extra pass. Foundational for distance-vector routing protocols (RIP) and for arbitrage detection in financial graphs.

```cpp
struct WEdge { NodeId from, to; Weight w; };

// Returns true if no negative cycle reachable from src.
bool bellman_ford(const std::vector<WEdge>& edges, std::size_t n,
                  NodeId src, std::vector<Weight>& dist) {
    dist.assign(n, std::numeric_limits<Weight>::infinity());
    dist[src] = 0.0;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        bool relaxed = false;
        for (const auto& e : edges)
            if (dist[e.from] + e.w < dist[e.to]) {
                dist[e.to] = dist[e.from] + e.w;
                relaxed = true;
            }
        if (!relaxed) return true;                  // converged early
    }
    for (const auto& e : edges)                     // negative-cycle check
        if (dist[e.from] + e.w < dist[e.to]) return false;
    return true;
}
```

*A\** improves on Dijkstra when an admissible heuristic $h(v) <= d(v, "goal")$ exists (e.g., great-circle distance in road networks); the priority key becomes $g(v) + h(v)$.

=== PageRank (Iterative Graph Algorithm)

```python
def pagerank(graph: dict, damping: float = 0.85, iters: int = 20) -> dict:
    """
    graph: {node: [neighbors]}. Returns dict of node → rank.
    """
    N = len(graph)
    rank  = {v: 1.0 / N for v in graph}
    in_links = {v: [] for v in graph}
    for v, neighbors in graph.items():
        for u in neighbors:
            in_links[u].append(v)

    for _ in range(iters):
        new_rank = {}
        for v in graph:
            contrib = sum(rank[u] / len(graph[u]) for u in in_links[v])
            new_rank[v] = (1 - damping) / N + damping * contrib
        rank = new_rank
    return rank
```

*GraphX / Pregel model:* for distributed graph computation, the Pregel model sends messages along edges; each vertex updates its state based on received messages. Used in GraphX (Spark), Giraph (Facebook), graph algorithms in Neo4j.

=== Graph Databases in Practice

#table(
  columns: (auto, auto, auto),
  [*System*], [*Model*], [*Query language*],
  [Neo4j],        [Property graph, native graph storage], [Cypher],
  [Amazon Neptune],[Property graph + RDF],                [Cypher / SPARQL / Gremlin],
  [TigerGraph],    [Property graph, OLAP-optimized],      [GSQL],
  [DGraph],        [RDF-like],                            [DQL (GraphQL)],
  [PostgreSQL],    [Relational (with pgRouting for paths)],[SQL + pgRouting DSL],
)

== References

Pelkonen, T. et al. (2015). "Gorilla: A Fast, Scalable, In-Memory Time Series Database." VLDB.

Bornea, M. et al. (2013). "Building an Efficient RDF Store over a Relational Database." SIGMOD.

Malewicz, G. et al. (2010). "Pregel: A System for Large-Scale Graph Processing." SIGMOD.

Shao, B. et al. (2013). "Trinity: A Distributed Graph Engine on a Memory Cloud." SIGMOD.

Rodriguez, M., Neubauer, P. (2010). "The Graph Traversal Pattern." arXiv:1004.1001. (Gremlin)

Erdős, P., Rényi, A. (1960). "On the Evolution of Random Graphs." Pub. Math. Inst. Hung. Acad. Sci.

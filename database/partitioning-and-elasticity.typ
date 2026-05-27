= Partitioning and Elasticity

Partitioning (sharding) distributes data across multiple nodes to scale beyond a single machine's capacity. Elasticity means the cluster can grow and shrink while continuing to serve traffic. The key challenge: minimize data movement when rebalancing while maintaining even load.

*See also:* _database/consensus-and-replication.typ_, _database/transactions-distributed.typ_, _database/weakly-consistent-systems.typ_

== Partitioning Strategies

=== Range Partitioning

Assign contiguous key ranges to shards. Natural for range queries; prone to hot spots on monotonically increasing keys (e.g., timestamps, auto-increment IDs).

```sql
-- PostgreSQL declarative partitioning (range)
CREATE TABLE orders (
    order_id   BIGINT,
    created_at TIMESTAMPTZ NOT NULL,
    amount     NUMERIC
) PARTITION BY RANGE (created_at);

CREATE TABLE orders_2024_q1 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');
CREATE TABLE orders_2024_q2 PARTITION OF orders
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

-- Query pruning: planner eliminates irrelevant partitions
EXPLAIN SELECT * FROM orders WHERE created_at BETWEEN '2024-02-01' AND '2024-02-28';
-- Only scans orders_2024_q1 (partition pruning)
```

*Hot spot avoidance:* use a compound key `(shard_prefix, timestamp)` where `shard_prefix = hash(key) % N`. Distributes writes across N shards while preserving time-ordered reads within a shard.

=== Hash Partitioning

Assign rows to shards by `hash(key) mod N`. Even distribution; range queries require all-shard scatter-gather.

```python
def hash_shard(key: str, num_shards: int) -> int:
    import hashlib
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return h % num_shards

# Consistent hashing avoids full data reshuffle when N changes
```

```sql
-- PostgreSQL hash partitioning
CREATE TABLE users (
    user_id BIGINT,
    email   TEXT
) PARTITION BY HASH (user_id);

CREATE TABLE users_0 PARTITION OF users FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE users_1 PARTITION OF users FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE users_2 PARTITION OF users FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE users_3 PARTITION OF users FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

=== Directory / Lookup-Based Partitioning

A *routing table* maps each key (or key range) to a shard. Fully flexible; the routing table is the source of truth.

```python
class DirectoryRouter:
    def __init__(self):
        self.directory: dict[str, str] = {}   # key_prefix → shard_id

    def lookup(self, key: str) -> str:
        # Find the longest matching prefix
        for prefix in sorted(self.directory, key=len, reverse=True):
            if key.startswith(prefix):
                return self.directory[prefix]
        raise KeyError(f"No shard for key {key}")

    def reassign(self, key_prefix: str, new_shard: str):
        self.directory[key_prefix] = new_shard  # O(1) rebalancing metadata change
        # Background data migration: move rows with this prefix to new_shard
```

Used by: Vitess (YouTube's MySQL sharding), MongoDB's config server.

== Consistent Hashing

Solves the "when N changes, reshuffle N/N+1 of data" problem. Map both nodes and keys onto a ring (0..2^32). Each key is owned by the first node clockwise from its position.

```python
import hashlib, bisect

class ConsistentHashRing:
    def __init__(self, virtual_nodes: int = 150):
        self.ring    = []    # sorted list of (hash, node_id)
        self.vnodes  = virtual_nodes

    def add_node(self, node_id: str):
        for i in range(self.vnodes):
            h = self._hash(f"{node_id}:{i}")
            bisect.insort(self.ring, (h, node_id))

    def remove_node(self, node_id: str):
        self.ring = [(h, n) for h, n in self.ring if n != node_id]

    def get_node(self, key: str) -> str:
        if not self.ring:
            raise RuntimeError("Empty ring")
        h   = self._hash(key)
        idx = bisect.bisect_left(self.ring, (h, ""))
        return self.ring[idx % len(self.ring)][1]

    def _hash(self, s: str) -> int:
        return int(hashlib.md5(s.encode()).hexdigest(), 16)

ring = ConsistentHashRing(virtual_nodes=150)
for node in ["node-A", "node-B", "node-C"]:
    ring.add_node(node)

print(ring.get_node("user:42"))   # "node-B" (deterministic)

# Adding node-D: only ~25% of keys reassigned, not 75%
ring.add_node("node-D")
```

*Virtual nodes:* each physical node owns $V$ slots on the ring. Without virtual nodes, load imbalance is high for small $N$. With $V = 150$, standard deviation of load is $< 10%$.

Used by: Cassandra, Amazon DynamoDB, Redis Cluster.

== Rebalancing

When shards are added or removed, data must migrate. *Minimizing data movement* while ensuring even load is an optimization problem.

*Live migration pattern* (used by CockroachDB, Vitess):

```
1. Source shard: start streaming WAL to destination
2. Destination: apply WAL changes while bulk-copying existing rows
3. Once destination is caught up (lag < threshold):
   a. Pause writes to source for the migrating key range (brief)
   b. Apply final WAL delta to destination
   c. Update routing table atomically → new writes go to destination
   d. Resume writes
4. Source deletes migrated rows (background GC)
```

```python
# Live migration coordinator (pseudocode)
def migrate_range(key_range, source_shard, dest_shard, router):
    # Step 1: Start replication stream
    stream = source_shard.open_replication_stream(key_range)

    # Step 2: Bulk copy + apply stream
    dest_shard.bulk_copy(source_shard.snapshot(key_range))
    for event in stream:
        dest_shard.apply(event)
        if dest_shard.lag() < LAG_THRESHOLD:
            break

    # Step 3: Cutover
    with router.pause_writes(key_range):             # brief pause
        for event in stream.drain():                 # drain remaining
            dest_shard.apply(event)
        router.update(key_range, dest_shard)         # atomic update

    # Step 4: GC
    source_shard.delete_range(key_range)             # background
```

== Scatter-Gather Queries

Queries that don't include the shard key must fan out to all shards:

```python
# Scatter-gather: execute on all shards, merge results
import concurrent.futures

def scatter_gather(query: str, shards: list, merge_fn):
    with concurrent.futures.ThreadPoolExecutor() as ex:
        futures = {ex.submit(s.execute, query): s for s in shards}
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    return merge_fn(results)

# Example: COUNT(*) across all shards
total = scatter_gather(
    "SELECT COUNT(*) FROM orders",
    shards,
    merge_fn=lambda rs: sum(r[0]["count"] for r in rs)
)
```

*Cross-shard aggregation is expensive:* prefer shard keys that co-locate frequently joined data (tenant ID, user ID). Avoid sharding on a key that requires all aggregation to be cross-shard.

== Auto-Sharding Systems

*CockroachDB:* ranges (64 MB default). The cluster periodically rebalances ranges based on storage and QPS. Range splits happen automatically when a range exceeds size or QPS limits.

*Vitess:* VSchema defines the sharding key. The VTGate proxy routes queries; VShards are MySQL instances. Supports re-sharding (doubling shard count) with 0 downtime via traffic cutover.

*Cassandra:* consistent hashing with virtual nodes. No central coordinator — each node knows the ring state via gossip. Rebalancing is `nodetool repair` plus streaming.

== References

DeWitt, D., Gray, J. (1992). "Parallel Database Systems: The Future of High Performance Database Systems." CACM.

Karger, D. et al. (1997). "Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web." STOC.

Amazon DynamoDB Team. (2022). "Amazon DynamoDB: A Scalable, Predictably Performant, and Fully Managed NoSQL Database Service." USENIX ATC.

Corbett, J. et al. (2012). "Spanner: Google's Globally-Distributed Database." OSDI.

Shute, J. et al. (2013). "F1: A Distributed SQL Database That Scales." VLDB.

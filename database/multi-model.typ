#import "../template.typ": xref

= Multi-Model Databases <multi-model>

The "NoSQL" era convinced many that the relational model was inadequate for documents, key-value, graphs, and time series. A decade of operational experience reversed the conclusion partially: most NoSQL systems re-invented joins (poorly), transactions (eventually), and SQL (under another name). At the same time, relational systems absorbed JSON, vector, and graph types natively. The term *multi-model* now describes both directions — relational engines that handle documents, and document/graph/KV systems that grew SQL-like querying.

*See also:* #xref("database", "query-languages", label: "Query Languages"), #xref("database", "storage-engines", label: "Storage Engines"), #xref("database", "time-series-and-graph", label: "Time-Series and Graph"), #xref("database", "vector-and-similarity-search", label: "Vector and Similarity Search")

== JSON in Relational Systems

Adding semistructured data to SQL requires three pieces: a binary storage format, a path language, and indexing.

=== PostgreSQL `jsonb`

`jsonb` (binary JSON) was added in 9.4. The binary layout sorts keys, removes whitespace, and stores arrays/objects as length-prefixed entries enabling $O(log n)$ key lookup within an object. Operators `->`, `->>`, `#>`, `#>>` extract elements; `@>` is containment; `?`, `?|`, `?&` check key existence.

```sql
CREATE TABLE events(id BIGSERIAL, doc JSONB);

INSERT INTO events(doc) VALUES
  ('{"type":"click","user":{"id":42,"tags":["a","b"]},"ts":1700000000}');

-- Containment query: matches if doc has these fields
SELECT * FROM events WHERE doc @> '{"type":"click","user":{"id":42}}';

-- Path expression (SQL/JSON path)
SELECT jsonb_path_query(doc, '$.user.tags[*] ? (@ == "a")') FROM events;

-- GIN index for containment
CREATE INDEX events_doc_gin ON events USING GIN (doc jsonb_path_ops);

-- BTree index on an extracted scalar
CREATE INDEX events_user_id ON events ((doc->'user'->>'id'));
```

The `jsonb_path_ops` opclass produces one GIN entry per *value path* instead of one per key — smaller index, equality lookups only. PostgreSQL 12 added the SQL/JSON path language (`jsonb_path_query`, `@@`, `@?`).

=== MySQL JSON

MySQL 5.7's JSON type stores a binary `JsonBinary` with key dictionary, large object spill, and partial in-place update for documents that did not grow. JSON_TABLE (8.0) implements the SQL:2016 function. *Functional indexes* on JSON expressions emulate jsonb-style indexing.

=== SQL Server and Oracle

SQL Server stores JSON as `NVARCHAR` and reparses (until the 2025 binary JSON preview). Oracle's `OSON` (Optimized JSON) is binary and indexable via JSON Search Index.

=== When to use JSONB vs Normalized Columns

#table(
  columns: (auto, auto),
  [*Prefer JSONB*], [*Prefer columns*],
  [Schema varies per row, sparse fields], [Fixed schema, hot fields],
  [Read-mostly archive payloads], [High update churn (whole-doc rewrite cost)],
  [Need to ingest arbitrary nested data], [Need referential integrity, FKs],
  [Few cardinality-bounded queries], [Range queries, joins on the field],
)

== Document Stores

=== MongoDB

MongoDB stores BSON (Binary JSON) documents in collections within databases. The WiredTiger storage engine (since 3.2) is an LSM/B-tree hybrid with MVCC and document-level locking.

*Data model:*

```javascript
db.orders.insertOne({
  _id: ObjectId(),
  customer: { id: 42, name: "Ada" },
  items: [ { sku: "A", qty: 2, price: 9.99 } ],
  status: "paid",
  ts: ISODate("2026-05-01T00:00:00Z")
});
db.orders.createIndex({ "customer.id": 1, ts: -1 });
```

*Aggregation pipeline* is the workhorse query language:

```javascript
db.orders.aggregate([
  { $match: { status: "paid", ts: { $gte: ISODate("2026-01-01") } } },
  { $unwind: "$items" },
  { $group: { _id: "$items.sku",
              revenue: { $sum: { $multiply: ["$items.qty","$items.price"] } } } },
  { $sort: { revenue: -1 } },
  { $limit: 10 }
]);
```

Each stage is a stream operator; the optimizer pushes `$match` and `$project` before `$lookup` (joins). `$lookup` joins are nested-loop or hash; cross-collection joins are second-class compared to SQL.

*Transactions:* MongoDB added multi-document ACID transactions in 4.0 (replica sets) and 4.2 (sharded). They use an Oplog-based timestamped protocol over MongoDB's own consensus layer derived from Raft.

*Sharding* is hash- or range-based on a *shard key*. The query router (`mongos`) sends queries to the appropriate shards; queries without the shard key fan out to all shards.

=== Couchbase

Couchbase combines a memcached-compatible KV layer with N1QL (a SQL-for-JSON dialect) and full-text search. Documents live in *buckets* → *scopes* → *collections*. The Indexer process is separate from data nodes; *Global Secondary Indexes* (GSI) are maintained asynchronously, scanned with covering optimizations.

```sql
-- N1QL example
SELECT u.name, ARRAY_LENGTH(u.followers) AS n_followers
FROM   social._default.users u
WHERE  u.country = "US" AND u.score > 100
ORDER BY n_followers DESC LIMIT 50;
```

Couchbase's value proposition is sub-millisecond KV reads + SQL-style analytical queries on the same data via the index service.

== Key-Value Stores

KV stores trade rich querying for raw throughput and simple operational models.

=== Redis

Redis is single-threaded, in-memory, with optional AOF/RDB persistence. Data types are first-class: strings, lists, hashes, sets, sorted sets, streams, bitmaps, HyperLogLog, geospatial sets (sorted set + GeoHash). Commands operate atomically.

```
SET user:42 '{"name":"Ada"}'
ZADD leaderboard 9001 user:42
ZRANGEBYSCORE leaderboard 1000 +inf LIMIT 0 10
GEOADD places -122.4 37.7 "GoldenGate"
GEORADIUS places -122.41 37.75 5 km WITHCOORD WITHDIST
XADD events * type click uid 42
```

*Cluster mode* uses 16384 hash slots assigned to shards; `CRC16(key) mod 16384` picks the slot. *Pub/Sub*, *streams* (Kafka-like consumer groups), and *modules* (RediSearch, RedisJSON, RedisGraph) extend the core.

Redis 7.4 added *Multi-Threaded I/O* but the command executor is still single-threaded — a critical performance characteristic to budget around.

=== ScyllaDB

Scylla is a C++ rewrite of Cassandra with shard-per-core architecture (Seastar framework, run-to-completion futures, NUMA-aware allocation). It implements the Cassandra wire protocol and CQL, plus DynamoDB-compatible Alternator.

*Data model:* partition key + clustering key + columns (Cassandra Wide Column). LWT (lightweight transactions) provide linearizable single-row CAS via Paxos.

```cql
CREATE TABLE events(
  user_id   bigint,
  ts        timestamp,
  type      text,
  payload   text,
  PRIMARY KEY ((user_id), ts)        -- partition key, clustering key
) WITH CLUSTERING ORDER BY (ts DESC);

SELECT * FROM events WHERE user_id = 42 AND ts > '2026-01-01' LIMIT 100;
```

Tunable consistency (`ONE`, `QUORUM`, `LOCAL_QUORUM`, `ALL`) chooses how many replicas confirm a read/write. Scylla's shard-per-core eliminates lock contention and reaches > 1M ops/s/core on modern NVMe.

=== FoundationDB

FoundationDB is an ordered key-value store with *full ACID serializable transactions* across the entire cluster, on commodity hardware. The architecture separates *Resolvers* (transaction conflict detection), *Logs* (durable WAL), *Storage Servers* (data), and *Master/CC* (coordination).

*Transactions* are optimistic: client reads at a *read version* (timestamp from CC), buffers writes, and at commit the Resolver checks for conflicting writes between the read version and commit version. The simulation framework (FDB's killer feature) deterministically injects faults to find bugs.

```python
@fdb.transactional
def transfer(tr, src, dst, amount):
    s = int(tr[src] or 0)
    d = int(tr[dst] or 0)
    tr[src] = str(s - amount)
    tr[dst] = str(d + amount)
```

The *Layer* concept: SQL, document, queue semantics are libraries on top of the ordered KV interface. Snowflake metadata, iCloud back-end, and CockroachDB's range descriptors all use FDB.

== Wide-Column

The Cassandra/HBase/BigTable family stores tables as sparse, sorted maps:
`(row_key, column_family:column_qualifier, timestamp) -> value`.

This model fits append-heavy time-series, audit logs, and per-user activity feeds. It does *not* fit complex analytical queries; recent versions of all three have added secondary indexes and limited SQL but they remain best for known access paths.

== Graph Stores

=== Neo4j

Neo4j is the canonical *labeled property graph* database. Nodes and relationships each carry a label and arbitrary key-value properties. Storage is *index-free adjacency*: each node record points to its first relationship; relationships form doubly-linked lists per (start, type, direction). Traversal is pointer-chasing, $O(1)$ per hop.

```cypher
// Friend recommendation
MATCH (me:Person {id: $uid})-[:FRIEND]-(f)-[:FRIEND]-(fof)
WHERE  NOT (me)-[:FRIEND]-(fof) AND me <> fof
RETURN fof, count(*) AS strength
ORDER BY strength DESC LIMIT 10;
```

Neo4j 5 introduced *Composite Database* (federation) and made Cypher closer to GQL. The Bloom UI and Neo4j Graph Data Science library provide visualization and analytics (PageRank, community detection, embeddings).

*Limits:* Neo4j's single-write-master architecture caps cluster write throughput; recent Fabric/Composite features shard read workloads.

=== JanusGraph

JanusGraph (fork of Titan, 2017) is *not* a database — it is a graph layer over a pluggable storage backend (Cassandra, HBase, ScyllaDB, BerkeleyDB) and a pluggable indexing backend (Elasticsearch, Solr, Lucene). It implements Apache TinkerPop's Gremlin traversal language.

```groovy
g.V().has('Person','id', uid)
     .out('FRIEND').out('FRIEND').dedup()
     .where(without('me'))
     .groupCount().order(local).by(values, desc).limit(local, 10)
```

JanusGraph's strength is leveraging existing operational expertise (Cassandra, ES) for graphs; the weakness is that distributed Gremlin queries can be hard to optimize compared to Cypher/GQL.

=== Other Graph Systems

#table(
  columns: (auto, auto, auto),
  [*System*], [*Storage*], [*Language*],
  [Neo4j], [Native LPG, index-free adjacency], [Cypher / GQL],
  [JanusGraph], [Cassandra / HBase backend], [Gremlin],
  [TigerGraph], [Native MPP, partitioned], [GSQL],
  [Amazon Neptune], [Custom service], [Gremlin, openCypher, SPARQL],
  [DGraph], [Native sharded LPG], [DQL (GraphQL-ish)],
  [Memgraph], [In-memory], [Cypher],
  [SQL Server Graph Tables], [Tables with edge metadata], [SQL with `MATCH`],
  [AGE (PostgreSQL)], [PostgreSQL extension], [openCypher embedded in SQL],
)

== Multi-Model in One Engine

Several systems bundle multiple models:

- *ArangoDB* — JSON documents + graph (named graphs) + KV, single AQL query language.
- *OrientDB* — document + graph + KV, SQL with graph extensions.
- *Microsoft Azure Cosmos DB* — single core (atom-record-sequence) exposes APIs: SQL (document), MongoDB, Cassandra, Gremlin, Table.
- *Oracle Database 23ai* — relational + JSON-relational duality views + property graphs (SQL/PGQ) + vector.
- *SingleStore, TiDB, YugabyteDB* — relational core with JSON columns and (limited) graph patterns.

The *JSON-Relational Duality* idea (Oracle): the same data is queryable as rows or as documents; mutations on either projection update the other. Conceptually a generalization of updatable views.

== When Does Multi-Model Help?

- *Workloads dominated by one model* — pick the specialist (PostgreSQL for relational, Neo4j for traversal, Cassandra for write-heavy KV).
- *Polyglot persistence already in place* — multi-model can collapse operational surface area at moderate performance cost.
- *Greenfield project, uncertain shape* — multi-model lowers schema commitment; refactor later if a single model dominates.

The historical lesson: every successful "non-relational" system eventually grew transactions, joins, and a declarative query language. Choose for the dominant workload, not for the marketing model.

== Further Reading

Stonebraker, M., Hellerstein, J. (2005). "What Goes Around Comes Around." Readings in Database Systems.

Stonebraker, M. (2010). "SQL Databases v. NoSQL Databases." CACM.

Chasseur, C., Patel, J. (2013). "Design and Evaluation of Storage Organizations for Read-Optimized Main Memory Databases." VLDB.

MongoDB Inc. (2019). "Tunable Consistency in MongoDB." VLDB.

DataStax (2016). "Apache Cassandra: Internals of the Storage Engine." (Whitepaper).

Carbone, P., Liagouris, J. et al. (2020). "The FoundationDB Distributed Transactional Key-Value Store." SIGMOD.

Lakshman, A., Malik, P. (2010). "Cassandra — A Decentralized Structured Storage System." SIGOPS OSR.

Neo4j (2020). "Neo4j Internals: Native vs Non-Native Graph Storage." Whitepaper.

Francis, N. et al. (2018). "Cypher: An Evolving Query Language for Property Graphs." SIGMOD.

Rodriguez, M. (2015). "The Gremlin Graph Traversal Machine and Language." DBPL.

Liu, Z. et al. (2022). "JSON Relational Duality." Oracle CIDR.

Sadalage, P., Fowler, M. (2012). "NoSQL Distilled."

= Log-Based Systems

The *append-only log* is one of the most consequential data structures in distributed systems: every write is a new record at the tail, nothing is ever overwritten, and the entire history of a system can be reconstructed by replaying the sequence from the beginning. Lamport observed that any deterministic state machine fed the same ordered log of inputs reaches the same state — a principle that connects consensus, replication, stream processing, and storage under one roof. This chapter traces that principle from its theoretical roots through Apache Kafka's production architecture to unified batch-and-stream frameworks.

*See also:* _time-and-order.typ_, _transactions.typ_, `database/storage-engines.typ`, `distributed-systems/consensus-deep-dive.typ`

== The Log as Central Abstraction

Lamport's state machine replication insight (1978) is deceptively simple: if every replica starts from the same initial state and processes the same log of commands in the same order, all replicas converge. The log is the *source of truth*; the database is merely a materialized view of it.

Jay Kreps (2013) generalised this to data integration at scale: instead of point-to-point pipelines between $N$ data systems (requiring $O(N^2)$ connectors), every system publishes to and consumes from a shared, ordered, fault-tolerant log. The log becomes the integration plane. *Log-structured* thinking then permeates storage (LSM trees), change data capture, and stream processing.

Properties of an ideal log:
- *Durability:* a record acknowledged is persisted.
- *Ordering:* offsets within a partition define a total order; across partitions ordering is approximate (timestamps, Lamport clocks).
- *Replay:* consumers hold their own read cursor and may re-read any retained record.
- *Back-pressure isolation:* producers and consumers are decoupled in time.

== Apache Kafka Architecture

Kafka (Kreps, Narkhede, Rao — LinkedIn, 2011) is the de-facto standard log broker. Its architecture is built around four primitives.

=== Topics, Partitions, and Replicas

A *topic* is a named, append-only sequence of records. Topics are split into *partitions* — ordered, immutable, numbered sequences. Partitions are the unit of parallelism and ordering; messages within a partition are totally ordered by *offset*, a monotonically increasing 64-bit integer. Messages across partitions have no guaranteed order.

Each partition has a configurable *replication factor* $r$. The cluster maintains one *leader* replica and $r - 1$ *follower* replicas for each partition. All reads and writes flow through the leader.

=== In-Sync Replicas and Leader Election

The *in-sync replica set* (ISR) is the set of replicas that have caught up to within `replica.lag.time.max.ms` of the leader. A produce request with `acks=all` is acknowledged only after all ISR members have written the record to their local log.

When a leader fails, the controller (historically ZooKeeper-elected; now KRaft) selects the new leader from the ISR. If the ISR is empty and `unclean.leader.election.enable=true`, an out-of-sync replica may be elected at the cost of potential data loss.

```
ISR invariant:
  for each replica r in ISR:
    LEO(r) >= HW - max_lag
  HW = min(LEO(r) for r in ISR)
  A record at offset o is "committed" when HW > o
```

Here *LEO* is Log End Offset and *HW* is the High Watermark — the committed frontier.

=== Consumer Groups and Offset Management

A *consumer group* is a set of consumers that collectively read a topic. Each partition is assigned to exactly one consumer within the group, giving horizontal scale-out. A consumer commits its progress (the *offset*) to the `__consumer_offsets` internal topic, enabling restart without data loss.

Offset management semantics:
- *at-most-once:* commit before processing.
- *at-least-once:* commit after processing (default).
- *exactly-once:* requires transactions (see below).

== Log Compaction vs Retention

Kafka offers two retention policies per topic:

*Delete retention* keeps records for a configured time window or size limit, then purges the oldest segments. Suitable for event streams where old records have no value.

*Log compaction* retains only the *latest* record per key, effectively making the topic a changelog of a key-value store. The background *log cleaner* thread scans dirty segments and writes a compacted version. The result is a *compacted topic*: bounded storage, infinite replay of the latest value per key. Used for database changelogs (Kafka Streams' state store changelogs, Debezium source topics).

#table(
  columns: (1fr, 1fr, 1fr),
  table.header[*Policy*][*Storage*][*Use case*],
  [Delete], [Bounded by time/size], [Event streams, logs, metrics],
  [Compaction], [Bounded by key space], [Changelogs, config, CDC snapshots],
  [Compaction + Delete], [Both limits apply], [Hybrid: latest state + time window],
)

== Exactly-Once Semantics

Achieving exactly-once delivery in a distributed broker requires three cooperating mechanisms.

=== Idempotent Producers

Each producer is assigned a *producer ID* (PID). Every batch carries a monotone *sequence number* per (PID, partition). The broker deduplicates retried batches: if `sequence(incoming) == expected`, accept; if less, discard (duplicate); if greater, reject (gap error). This provides exactly-once at the producer-broker hop within a session.

=== Transactional API

A *transactional producer* (configured with `transactional.id`) coordinates atomic writes across multiple partitions. The protocol uses a *transaction coordinator* (one per `__transaction_state` partition):

```
producer.init_transactions()
producer.begin_transaction()
producer.send(topic_A, key, value)
producer.send(topic_B, key, value)
producer.send_offsets_to_transaction(offsets, consumer_group)
producer.commit_transaction()   // two-phase commit via coordinator
```

The coordinator writes a `COMMIT` or `ABORT` marker to each involved partition. This is a two-phase commit where the coordinator's log is the durable state, making coordinator crashes recoverable by replaying its own topic.

=== Read-Committed Isolation

Consumers using `isolation.level=read_committed` skip records belonging to open or aborted transactions. They buffer records up to the *Last Stable Offset* (LSO), which advances only when transactions complete. This closes the consume-produce loop for exactly-once stream processing.

== Kafka Streams and ksqlDB

*Kafka Streams* is a Java library (not a cluster) that runs inside application processes. Each instance reads partitions, applies transformations (map, filter, join, aggregate), and writes results back to Kafka topics. State is stored in embedded RocksDB, backed by a changelog topic for fault tolerance.

Key abstractions:
- `KStream` — unbounded stream of records.
- `KTable` — changelog stream interpreted as a key-value table (latest value per key).
- `GlobalKTable` — fully replicated table, no partitioning constraint.
- Windowed joins and aggregations using tumbling, hopping, or session windows.

*ksqlDB* is a SQL interface over Kafka Streams. Queries are compiled to Kafka Streams topologies and run as persistent server-side processes. Push queries stream results to clients; pull queries serve point-in-time lookups from materialized state.

== Apache Pulsar's Multi-Layer Storage

Pulsar (Yahoo, 2016) separates the serving layer (brokers) from the storage layer (*Apache BookKeeper*), enabling independent scaling and zero-copy topic migration.

*BookKeeper* stores data in *ledgers* — immutable, append-only segments. A Pulsar topic's *managed ledger* chains multiple BookKeeper ledgers. The *ensemble* is the set of BookKeeper nodes (bookies) for a ledger; the *write quorum* $Q_w$ and *ack quorum* $Q_a$ govern durability ($Q_a <= Q_w$). A write is acknowledged after $Q_a$ bookies confirm.

```
Pulsar topic = [ledger_1][ledger_2]...[ledger_N]  (active)
ledger_i     = [entry_0][entry_1]...[entry_M]      (immutable once closed)
entry        = (ledger_id, entry_id, data)
```

Because brokers are stateless, a new broker can take over a topic immediately — it simply opens the latest ledger. This contrasts with Kafka where partition leadership requires log sync among replicas.

Pulsar also offers *tiered storage*: offloading cold ledgers to object storage (S3, GCS) while keeping the topic interface unchanged.

== Broker Comparison

#table(
  columns: (auto, 1fr, 1fr, 1fr, 1fr),
  table.header[*Feature*][*Kafka*][*Pulsar*][*Kinesis*][*NATS JetStream*],
  [Storage model], [Log segments on disk], [BookKeeper ledgers], [Shard shards on SSD], [File-based streams],
  [Scaling storage], [Add brokers], [Add bookies independently], [Increase shards], [Add server nodes],
  [Multi-tenancy], [Via cluster/topic naming], [Native namespaces], [Via stream ARNs], [JetStream accounts],
  [Replay], [Configurable retention], [Tiered storage], [7 days default], [Configurable],
  [Exactly-once], [Transactions API], [In progress], [No (at-least-once)], [Publish dedup],
  [Ops complexity], [High (KRaft/ZK)], [High (ZK + BK)], [Low (managed)], [Low (single binary)],
)

== Change Data Capture

*Change data capture* (CDC) turns a database's internal change log into a stream consumable by downstream systems. Rather than polling tables, CDC reads the database's replication stream (*binlog* in MySQL, *WAL* in PostgreSQL, *redo log* in Oracle).

*Debezium* is the leading open-source CDC framework. It runs as a Kafka Connect source connector:

```
PostgreSQL WAL → Debezium PostgreSQL connector
             → Kafka topic (one per table)
             → downstream: Elasticsearch, data warehouse, cache
```

Each record in the CDC topic carries the before/after state of a row, the operation type (`INSERT`, `UPDATE`, `DELETE`), and a source position (LSN in PostgreSQL).

=== The Outbox Pattern

The *outbox pattern* solves the dual-write problem: writing to a database and publishing to a broker atomically. The application writes the event to an `outbox` table in the same local transaction as the business entity change. A CDC connector reads the `outbox` table and publishes to Kafka. The broker write is decoupled from the business transaction, eliminating distributed transaction overhead.

```sql
BEGIN;
  UPDATE orders SET status = 'confirmed' WHERE id = 42;
  INSERT INTO outbox (aggregate_type, aggregate_id, event_type, payload)
    VALUES ('order', 42, 'OrderConfirmed', '{"id":42,...}');
COMMIT;
-- CDC picks up the outbox row and publishes it
```

== Log-Structured Storage: LSM Trees

The log abstraction is not only a messaging pattern — it is a storage engine architecture. A *Log-Structured Merge-tree* (LSM tree) turns all writes into sequential append operations, yielding high write throughput on spinning and flash storage.

Write path:
+ Append to *Write-Ahead Log* (WAL) on disk.
+ Write to in-memory *MemTable* (a sorted structure, usually a skip list or red-black tree).
+ When MemTable reaches threshold, flush as an immutable *SSTable* (Sorted String Table) to $L_0$.
+ Background *compaction* merges and sorts SSTables across levels $L_0 dots L_k$, enforcing size amplification targets.

Read path: check MemTable, then each level's SSTables (aided by Bloom filters), from newest to oldest. The *read amplification* factor grows with level count; Bloom filters reduce it drastically for point lookups.

Used by: LevelDB, RocksDB, Cassandra, HBase, ScyllaDB, TiKV, and — indirectly — Kafka's log segments.

== Unified Batch and Stream Processing

Historically, analytics pipelines split into:
- *Batch layer:* reprocess all historical data nightly (Hadoop MapReduce).
- *Speed layer:* low-latency processing of recent events (Storm, Spark Streaming).

=== Lambda Architecture

The *Lambda architecture* (Marz 2012) runs both layers in parallel; a *serving layer* merges their outputs. Queries combine the batch view (accurate, slow) with the speed view (approximate, fresh). The cost is maintaining two codebases for the same business logic.

=== Kappa Architecture

The *Kappa architecture* (Kreps 2014) eliminates the batch layer: retain the full event history in the log broker, and reprocess by spinning up a new stream job from offset 0. Stream processing must be expressive enough to replace batch (joins, aggregations, windowing). Kafka's long retention and consumer group offset replay make this practical. Flink, Spark Structured Streaming, and Kafka Streams all support it.

#table(
  columns: (1fr, 1fr, 1fr),
  table.header[*Aspect*][*Lambda*][*Kappa*],
  [Correctness], [Batch layer is authoritative], [Single pipeline, reprocess to correct],
  [Latency], [Speed layer gives low latency], [Same pipeline, tunable latency],
  [Complexity], [Two codebases], [One codebase],
  [Replay cost], [Cheap (batch always exists)], [Depends on retention policy],
)

== Further Reading

Lamport, L. (1978). "Time, Clocks, and the Ordering of Events in a Distributed System." CACM.

Kreps, J., Narkhede, N., Rao, J. (2011). "Kafka: A Distributed Messaging System for Log Aggregation." NetDB workshop.

Kreps, J. (2013). "The Log: What every software engineer should know about real-time data's unifying abstraction." LinkedIn Engineering Blog.

O'Neil, P., Cheng, E., Gawlick, D., O'Neil, E. (1996). "The Log-Structured Merge-Tree (LSM-Tree)." Acta Informatica.

Wang, J., et al. (2019). "Apache Pulsar: Unified Queuing and Streaming." Whitepaper.

Debezium documentation. https://debezium.io/documentation/

Richardson, C. (2018). "Microservices Patterns." Manning. (Outbox pattern, Chapter 3.)

Marz, N., Warren, J. (2015). "Big Data: Principles and Best Practices of Scalable Real-Time Data Systems." Manning.

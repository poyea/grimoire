= Change Data Capture

Change data capture (CDC) turns a database's internal replication log into an event stream, so every insert, update, and delete in the OLTP system appears, in commit order, as a record consumers can react to. It replaces the two bad alternatives: periodic full dumps (expensive, stale, miss intermediate states) and dual writes from application code (race-prone, drift-prone). This chapter covers log-based CDC mechanics, the Debezium architecture, Postgres logical decoding and the MySQL binlog, the snapshot-to-streaming handoff, exactly-once concerns, the transactional outbox pattern, and landing CDC in a lakehouse.

*See also:* _Streaming_ (the processing layer downstream of CDC), _Lakehouse Engineering_ (merge-on-read for CDC sinks), _Schema Evolution_ (DDL flowing through CDC), _Data Quality_ (reconciling replicas against sources).

== Why Log-Based

Three CDC strategies exist, in ascending order of fidelity:

- *Query-based polling.* `select * where updated_at > :last_seen`. Misses deletes, misses intermediate updates between polls, requires a reliable `updated_at` column, and adds load to the primary.
- *Trigger-based.* Database triggers write changes to an audit table. Captures everything but adds latency to every transaction and couples capture to the write path.
- *Log-based.* Read the write-ahead log (WAL) / binlog that the database already produces for crash recovery and replication. Zero overhead on the transaction path, captures deletes and every intermediate state, and preserves commit order.

Log-based CDC is the only one that scales; the others survive as fallbacks for databases without accessible logs.

== Debezium Architecture

Debezium (Red Hat, 2016) is the de facto open-source CDC platform: a set of source connectors (Postgres, MySQL, MongoDB, SQL Server, Oracle, Cassandra) that decode the database log and emit change events, classically deployed on Kafka Connect.

```
 Postgres ──WAL──▶ logical decoding (pgoutput)
                        │
                  Debezium connector (Kafka Connect worker)
                        │  per-table topics: cdc.public.orders
                        ▼
                     Kafka ──▶ Flink / sink connectors / lakehouse
```

A change event has an *envelope*: `before` image, `after` image, `op` (`c`/`u`/`d`/`r` for create, update, delete, snapshot read), and a `source` block (LSN or binlog position, transaction ID, commit timestamp). The event *key* is the table's primary key, so a compacted Kafka topic holds the latest state per row.

Connector offsets (the log position) are stored in Kafka itself; on restart the connector resumes from the last committed position. Since Debezium 2.x, *Debezium Server* runs connectors without Kafka Connect, targeting Pulsar, Kinesis, Pub/Sub, or HTTP, and Flink CDC embeds the Debezium decoders directly inside Flink sources.

== Postgres: Logical Decoding

Postgres physical replication ships raw WAL blocks; *logical decoding* (9.4, 2014) reinterprets WAL into row-level changes via an output plugin. The moving parts:

- `wal_level = logical` makes the WAL carry enough information (full tuple data) to reconstruct rows.
- A *replication slot* is a named server-side cursor that pins WAL on disk until the consumer confirms (`confirmed_flush_lsn`). This is the durability mechanism and the classic foot-gun: an abandoned slot retains WAL until the disk fills. Monitor `pg_replication_slots` lag; Postgres 13 added `max_slot_wal_keep_size` as a safety valve.
- The *output plugin* formats changes: `pgoutput` (built-in since 10, what Debezium uses), `wal2json`, or `test_decoding`. A *publication* (`create publication dbz for table orders, customers`) scopes which tables are decoded.
- `REPLICA IDENTITY` controls the `before` image: `DEFAULT` gives only the old primary key; `FULL` gives the entire old row (needed for complete delete/update events, at WAL-volume cost).

Decoding happens at *commit* time: the decoder buffers a transaction's changes and emits them only when the commit record arrives, so consumers never see uncommitted data. Long-running transactions therefore delay everything behind them (Postgres 14 added streaming of large in-progress transactions). Failover is the operational weak point: replication slots were not preserved across physical failover until Postgres 17 (2024) added failover slots.

== MySQL: Binlog

MySQL's binary log is a separate log written for replication (unlike Postgres, where logical decoding rides the WAL). Requirements for CDC: `binlog_format = ROW` (statement-based replication is useless for CDC, since it logs SQL text, not rows) and `binlog_row_image = FULL` for complete before/after images. Position tracking uses *GTIDs* (global transaction identifiers, 5.6+), which survive failover better than file+offset coordinates.

Two MySQL-specific quirks shape connector design. First, the binlog carries no schema, only column ordinals — so the connector must maintain its own *schema history* (Debezium stores DDL events in a dedicated topic) to interpret old binlog entries after `ALTER TABLE`. Second, binlogs expire (`binlog_expire_logs_seconds`, default 30 days); if a connector is down longer than retention, it must re-snapshot.

== Snapshots and the Streaming Handoff

A new CDC pipeline must combine the *current state* (snapshot) with *subsequent changes* (stream) without gaps or unbounded locking. The naive approach — lock all tables, dump, then stream from the locked position — is unacceptable on large databases.

Modern connectors use lock-free chunked snapshotting, based on the watermark algorithm from *DBLog* (Netflix, 2019): read the table in primary-key chunks, and for each chunk write low/high watermarks into the log (or note log positions); changes that arrive within the window are deduplicated against the chunk by key, so each row's final state is correct regardless of interleaving. Debezium implements this as *incremental snapshots* (signal-triggered, resumable, can run while streaming is already live); Flink CDC parallelises chunk reads across task managers.

The handoff guarantee to design for: every row appears at least once (snapshot read or change event), and applying events in offset order converges to source state. Snapshot reads are emitted with `op = r` so downstream merges can treat them as upserts.

== Delivery Semantics

CDC pipelines are *at-least-once* end to end: a connector crash after publishing but before committing its offset replays a window of events. Three properties make this safe:

- *Idempotent application.* Events are keyed by primary key and carry full row state; applying "upsert by key" twice is harmless. This is the workhorse — most CDC consumers need idempotence, not transactions.
- *Ordering per key.* Kafka guarantees order per partition; partitioning by primary key guarantees per-row order. Cross-table transactional ordering is *not* preserved across topics — Debezium's transaction metadata topic (transaction ID + event counts) lets a careful consumer rebuild transaction boundaries, but few do.
- *Monotonic versions.* Use the source LSN/GTID (not the event timestamp) as the version for last-writer-wins merges; timestamps tie and skew.

True exactly-once is achievable for Kafka-to-Kafka segments (Kafka transactions in Connect since 3.3) and for Flink-to-Iceberg sinks (checkpoint-coordinated commits), but the simpler invariant "at-least-once + idempotent upsert by (key, version)" is what production systems actually rely on.

== The Outbox Pattern

The dual-write problem: a service that writes to its database *and* publishes to Kafka cannot make both atomic; any crash between the two leaves them inconsistent. The *transactional outbox* fixes this by routing the event through the database:

```sql
begin;
  update orders set status = 'PAID' where id = 42;
  insert into outbox (aggregate_id, type, payload)
  values (42, 'OrderPaid', '{"order_id":42,"amount":9900}');
commit;
```

The outbox insert commits atomically with the business write; CDC then relays the outbox row to Kafka. Debezium's `EventRouter` SMT extracts the payload and routes by aggregate type, and the outbox table can be kept empty (delete after insert in the same transaction — the WAL still carries the insert). The result is effectively exactly-once *intent capture*: the event exists iff the transaction committed. The outbox also decouples the *public contract* (curated payload) from the *private schema* (raw table CDC), which is the contract-friendly way to expose service data — raw table CDC leaks every internal migration to consumers.

== CDC to the Lakehouse

Landing CDC in Iceberg/Delta/Hudi is the standard way to keep an analytical replica minutes behind OLTP. Two-layer pattern:

- *Append-only changelog table* (bronze): every event as-is, partitioned by ingest time. Cheap to write, full audit history, replayable.
- *Mirror table* (silver): `MERGE` the changelog into a deduplicated current-state table, using merge-on-read so high-rate upserts do not rewrite files (see _Lakehouse Engineering_).

```sql
merge into silver.orders t
using (
  select * from (
    select *, row_number() over (
      partition by order_id order by source_lsn desc) rn
    from bronze.orders_changelog
    where ingest_dt = :batch) where rn = 1) s
on t.order_id = s.order_id
when matched and s.op = 'd' then delete
when matched then update set *
when not matched and s.op != 'd' then insert *;
```

The `row_number` dedup collapses multiple changes per key per batch; ordering by LSN (not timestamp) keeps last-writer-wins correct. Flink CDC and Hudi's DeltaStreamer fold these layers into one continuous job; Hudi's record-level index was designed precisely for this workload. Watch two costs: small-file pressure from frequent commits (compaction is mandatory) and schema changes arriving mid-stream (wire Debezium's schema-change topic into an alert, and prefer additive evolution).

== Pitfalls

- *Forgotten replication slots.* The single most common Postgres outage caused by CDC: a paused connector pins WAL until the primary's disk fills. Alert on slot lag bytes.
- *Snapshot without handoff discipline.* Dump-then-stream with a gap between dump time and stream start silently loses changes; use watermark-based incremental snapshots.
- *Timestamps as merge versions.* Same-millisecond updates and clock skew break last-writer-wins; use LSN/GTID.
- *Exposing raw tables as the public interface.* Every internal column rename becomes a consumer incident; put an outbox or a transformation layer in between.
- *Ignoring deletes.* Query-based CDC and naive append-only sinks never see deletes; GDPR erasure then fails to propagate. Log-based CDC plus tombstone handling end to end.
- *Cross-table consistency assumptions.* Per-topic CDC does not preserve transaction atomicity across tables; consumers that join `orders` and `order_items` streams must tolerate transient mismatch.

== Further Reading

Andreakis, A., Papapanagiotou, I. (2019). "DBLog: A Watermark Based Change-Data-Capture Framework." Netflix / arXiv:2010.12597. The lock-free chunked snapshot algorithm interleaving snapshot and log reads, adopted by Debezium incremental snapshots and Flink CDC.

Das, S. et al. (2012). "All Aboard the Databus! LinkedIn's Scalable Consistent Change Data Capture Platform." SoCC. The first large-scale published CDC platform; source of much of the field's vocabulary.

Kreps, J. (2014). "The Log: What Every Software Engineer Should Know About Real-Time Data's Unifying Abstraction." LinkedIn Engineering Blog. Frames CDC as one instance of log-centric architecture.

Richardson, C. (2018). _Microservices Patterns._ Manning. Chapter 3 defines the transactional outbox and polling-publisher patterns.

Debezium documentation, https://debezium.io/documentation/. Connector mechanics, envelope format, incremental snapshots, and the outbox event router.

PostgreSQL documentation, "Logical Decoding." https://www.postgresql.org/docs/current/logicaldecoding.html. Replication slots, output plugins, and publication semantics.

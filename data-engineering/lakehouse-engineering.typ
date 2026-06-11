= Lakehouse Engineering

The lakehouse pattern unifies the data lake's cheap object-storage substrate with the warehouse's $"ACID"$ semantics, by layering an *open table format* — Iceberg, Delta Lake, or Hudi — on top of Parquet files in $"S3"$ / $"GCS"$ / $"ADLS"$. This chapter is the engineering companion to `database/lakehouses-and-open-formats.typ`: that one focuses on the format internals; this one focuses on writing, maintaining, and operating lakehouse tables in production.

*See also:* _Lakehouses and Open Table Formats_ (database framing), _Batch Processing_, _Streaming_, _Change Data Capture_, _Cloud Cost Engineering_ (Cloud & Infrastructure volume).

== The Three Layers

A lakehouse table has three physical layers:

```
            ┌──────────────────────────────┐
catalog →   │ table pointer (Glue/Unity/HMS)│   versioned table_id → metadata
            └──────────────┬───────────────┘
                           │
            ┌──────────────▼───────────────┐
metadata →  │ snapshot (json/avro)         │   list of manifests, schema, stats
            └──────────────┬───────────────┘
                           │
            ┌──────────────▼───────────────┐
data →      │ data files (Parquet)         │   columnar row groups
            └──────────────────────────────┘
```

Every write produces a *new snapshot* — atomic at the catalog level. Reads pin a snapshot ID and proceed without locks.

== Writing to Iceberg from Spark

```python
spark = (SparkSession.builder
    .config("spark.sql.catalog.lh", "org.apache.iceberg.spark.SparkCatalog")
    .config("spark.sql.catalog.lh.type", "rest")
    .config("spark.sql.catalog.lh.uri", "https://catalog:8181")
    .getOrCreate())

spark.sql("""
  create table if not exists lh.gold.fct_revenue (
    dt date, category string, revenue decimal(18,2)
  )
  using iceberg
  partitioned by (dt)
  tblproperties (
    'write.format.default' = 'parquet',
    'write.parquet.compression-codec' = 'zstd',
    'write.target-file-size-bytes' = '536870912',  -- 512 MB
    'commit.retry.num-retries' = '10'
  )
""")

(revenue_df.writeTo("lh.gold.fct_revenue")
   .overwritePartitions())  # atomic per-partition overwrite
```

`overwritePartitions` replaces only the partitions present in `revenue_df` — the canonical idempotent upsert pattern.

== ACID via Optimistic Concurrency

All three formats use optimistic concurrency: writers build a new snapshot referencing some base snapshot, then $"CAS"$ the catalog pointer. Two concurrent writers race; the loser retries.

```
W1 reads snap S0 → writes data → commits S0 → S1 (success)
W2 reads snap S0 → writes data → commits S0 → S2 (FAILS, base is now S1)
W2 retries:        rebases on S1 → commits S1 → S2 (success)
```

For *append-only* tables, retries are always safe. For *merge* writes that overlap on rows, retries must re-read changed files (Iceberg's `merge-on-read` vs `copy-on-write` mode).

== Delta vs Iceberg vs Hudi

#table(
  columns: 4,
  [*Property*], [*Delta Lake*], [*Iceberg*], [*Hudi*],
  [Origin], [Databricks (2019)], [Netflix → ASF (2018)], [Uber → ASF (2017)],
  [Metadata], [JSON transaction log], [Avro manifest tree], [Timeline + indexes],
  [Catalog], [Unity / Hive], [REST / Glue / Hive / Polaris], [Hive / Glue],
  [Schema evolution], [Add, rename, drop], [Add, rename, drop, reorder, widen], [Add, drop],
  [Hidden partitioning], [No], [Yes], [Partial],
  [Time travel], [Yes (version, timestamp)], [Yes (snapshot id, timestamp)], [Yes (instant)],
  [Streaming sink], [Structured Streaming], [Flink, Spark Structured Streaming], [DeltaStreamer / Hudi Streamer (native)],
  [Index for upserts], [Z-order, deletion vectors], [Bloom, deletion vectors], [Bloom, $"HBase"$, record-level],
)

The 2024-2026 trend is convergence: Delta UniForm exposes a Delta table as Iceberg metadata; Iceberg added deletion vectors; Hudi 1.0 added a catalog-aware model. Choose by ecosystem (Databricks ↔ Delta, multi-engine ↔ Iceberg, heavy upsert ↔ Hudi).

== File-Level Maintenance

A lakehouse table is only fast if files are healthy. Three maintenance jobs are non-negotiable:

- *Compaction.* Coalesce small files into target-sized files. `OPTIMIZE` in Delta, `rewriteDataFiles` in Iceberg, clustering in Hudi.
- *Expiration.* Delete snapshots older than retention to release storage.
- *Orphan cleanup.* Delete data files no longer referenced by any live snapshot.

```sql
-- Iceberg via Spark procedures
call lh.system.rewrite_data_files(
  table => 'gold.fct_revenue',
  options => map('target-file-size-bytes', '536870912'));

call lh.system.expire_snapshots(
  table => 'gold.fct_revenue',
  older_than => TIMESTAMP '2026-04-01 00:00:00',
  retain_last => 5);

call lh.system.remove_orphan_files(
  table => 'gold.fct_revenue',
  older_than => TIMESTAMP '2026-04-01 00:00:00');
```

Schedule daily; budget 5–10% of total compute for maintenance.

== Z-Order and Liquid Clustering

For multi-dimensional pruning (filter by both `user_id` and `event_ts`), sort within partitions by a space-filling curve.

```sql
-- Delta
optimize gold.fct_revenue zorder by (user_id, dt);

-- Iceberg
alter table gold.fct_revenue write ordered by (user_id, dt);
```

Liquid Clustering (Delta) and partition transforms (Iceberg `bucket(N, user_id)`) eliminate the partition decision entirely — the format chooses a layout that adapts as data evolves.

== Streaming Upserts via Merge-on-Read

For high-rate $"CDC"$ ingestion, *copy-on-write* (rewrite the data file containing the changed row) is too expensive. *Merge-on-read* writes the change as a delete-vector + insert and resolves at read time. Periodic compaction collapses the deltas.

```python
# Iceberg
spark.sql("""
  alter table lh.silver.users set tblproperties (
    'write.delete.mode' = 'merge-on-read',
    'write.update.mode' = 'merge-on-read',
    'write.merge.mode'  = 'merge-on-read'
  )
""")
```

Trade-off: writes are fast but reads cost more until compaction. Pair with frequent `rewrite_position_delete_files`.

== Time Travel and Branching

```sql
-- Read previous snapshot
select * from gold.fct_revenue version as of 12345;
select * from gold.fct_revenue timestamp as of '2026-05-30 00:00:00';
```

Iceberg supports *branches* and *tags* (since 1.2). A branch is a named writable pointer; a tag is a frozen snapshot ID. Use cases: stage a backfill on a branch, validate, then fast-forward main.

== Operations Checklist

- *Catalog backed by Postgres / DynamoDB / RDS.* Glue is convenient but rate-limited; Unity / Polaris / Nessie scale better.
- *Multi-engine writes only via the catalog,* never by writing files behind its back.
- *Schema evolution policies.* Allow add; require explicit migration for rename or drop.
- *Monitoring.* Alert on `num_data_files / partition`, snapshot count, $"S3"$ list latencies.
- *Cost.* $"S3"$ requests (especially `LIST`) dominate at high snapshot churn. Use REST catalog or manifest caching.

== Pitfalls

- *Two writers, no catalog locking.* Some lock-free configurations (`hadoop` catalog on S3) allow lost writes under contention. Use a $"DB"$-backed catalog.
- *Forgotten compaction.* A streaming sink writes thousands of files / hour; without compaction, the table becomes unreadable in days.
- *Schema drift breaking downstream readers.* Pin schema version per consumer or use late-binding `variant` columns.
- *Cross-region $"S3"$ on hot paths.* Catalog metadata reads multiply per query — co-locate compute and storage.

== Further Reading

Armbrust, M. et al. (2020). "Delta Lake: High-Performance ACID Table Storage over Cloud Object Stores." VLDB.

Apache Iceberg Specification v2. https://iceberg.apache.org/spec/.

Hudi documentation, https://hudi.apache.org/docs/overview.

Databricks. "Lakehouse: A New Generation of Open Platforms." CIDR 2021.

Project Nessie documentation, https://projectnessie.org/.

= Lakehouses and Open Table Formats

A *data lakehouse* combines the cost-efficiency of object storage (S3, GCS, Azure Blob) with ACID semantics, schema enforcement, and SQL query support previously found only in data warehouses. Apache Iceberg, Delta Lake, and Apache Hudi are the three dominant open table formats.

*See also:* _Column Stores and Vectorized Execution_, _Storage Engines_, _Streaming and Incremental Computation_

== The Problem with Raw Data Lakes

Traditional data lakes (just files in S3) suffer from:

- *No ACID:* concurrent writers corrupt data; partial writes are visible.
- *No schema evolution:* adding a column breaks existing readers.
- *No deletes/updates:* "delete" means rewrite the entire partition (expensive).
- *Poor query performance:* no statistics, no partitioning metadata catalog.
- *No time travel:* no way to query "what did the data look like yesterday?".

Open table formats solve all of these.

== Apache Iceberg

Iceberg (Apache, Netflix 2017) uses a *snapshot-based* metadata layer on top of Parquet/ORC/Avro files.

```
Iceberg metadata hierarchy:
  catalog (e.g., Hive Metastore, AWS Glue, Nessie)
    └── table pointer → metadata.json (latest)
           └── snapshot-list
                  └── current snapshot → manifest-list.avro
                         └── manifest-1.avro  ← list of data files + stats
                         └── manifest-2.avro
                                └── data/part-00001.parquet
                                └── data/part-00002.parquet
```

=== ACID via Optimistic Concurrency

```python
# Iceberg write protocol (simplified)
def iceberg_write(table, new_files, spark_session):
    # 1. Read current snapshot ID
    current_snap = table.current_snapshot()

    # 2. Write Parquet files to object storage (not visible yet)
    written_files = [write_parquet(f) for f in new_files]

    # 3. Create new manifest referencing written files
    manifest = create_manifest(written_files)

    # 4. Create new snapshot (atomically via catalog CAS)
    new_snap = Snapshot(
        snapshot_id   = generate_id(),
        parent_id     = current_snap.id,
        manifest_list = [manifest] + current_snap.manifests,
        summary       = {"operation": "append", "added-files": len(written_files)}
    )

    # 5. Atomic swap: new_metadata.json replaces current pointer
    # If another writer committed first: retry from step 1 (optimistic)
    table.catalog.swap(table.name, expected=current_snap.id, new_snap=new_snap)
```

*ACID guarantees:*
- *Atomicity:* either the snapshot pointer is updated or it isn't.
- *Isolation:* readers always see a consistent snapshot; in-progress writes are invisible.
- *Consistency:* schema constraints enforced at commit.
- *Durability:* Parquet files in object storage are durable by design.

=== Time Travel

Every snapshot is retained (subject to retention policy). Query any historical state:

```sql
-- Iceberg time travel (SparkSQL)
SELECT * FROM orders VERSION AS OF 12345678;          -- specific snapshot ID
SELECT * FROM orders TIMESTAMP AS OF '2025-01-01';    -- point in time

-- List snapshots
SELECT * FROM orders.snapshots;
-- snapshot_id | parent_id | committed_at | operation | summary
--  123        | NULL      | 2025-01-01   | append    | ...
--  456        | 123       | 2025-01-02   | overwrite | ...

-- Rollback to a previous snapshot
CALL catalog.rollback_to_snapshot('db.orders', 123);
```

=== Partition Evolution

Iceberg allows changing the partition scheme without rewriting data:

```sql
-- Start with daily partitions
ALTER TABLE orders ADD PARTITION FIELD days(created_at);
-- After growth, switch to hourly:
ALTER TABLE orders ADD PARTITION FIELD hours(created_at);
ALTER TABLE orders DROP PARTITION FIELD days(created_at);
-- Old files stay in daily partition layout; new files use hourly
-- Iceberg query planner handles both layouts transparently
```

=== Hidden Partitioning

Unlike Hive partitioning (user must write `WHERE year=2025 AND month=1`), Iceberg transforms are hidden:

```sql
-- Table defined with: PARTITIONED BY (months(created_at))
-- User writes: (no partition filter needed)
SELECT * FROM orders WHERE created_at BETWEEN '2025-01-01' AND '2025-03-31';
-- Iceberg prunes to 3 monthly partitions automatically
```

== Delta Lake

Delta Lake (Databricks 2019) stores a *transaction log* (`_delta_log/`) as a series of JSON/Parquet checkpoint files.

```
Delta table structure:
  s3://my-bucket/orders/
    _delta_log/
      00000000000000000000.json   ← initial commit
      00000000000000000001.json   ← add files
      00000000000000000002.json   ← delete files
      00000000000000000010.parquet  ← checkpoint (every 10 commits)
    part-00000-xxx.parquet
    part-00001-xxx.parquet
```

```python
# Delta Lake: Python with delta-spark
from pyspark.sql import SparkSession
from delta.tables import DeltaTable

spark = SparkSession.builder \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .getOrCreate()

# Write
df.write.format("delta").mode("overwrite").save("s3://bucket/orders")

# MERGE (upsert) — key differentiator from Iceberg
delta_table = DeltaTable.forPath(spark, "s3://bucket/orders")
delta_table.alias("target").merge(
    source   = updates_df.alias("src"),
    condition= "target.order_id = src.order_id"
).whenMatchedUpdate(set={
    "status": "src.status",
    "updated_at": "src.updated_at"
}).whenNotMatchedInsert(values={
    "order_id":   "src.order_id",
    "status":     "src.status",
    "created_at": "src.created_at"
}).execute()

# Time travel
spark.read.format("delta") \
     .option("versionAsOf", 5) \
     .load("s3://bucket/orders")
```

== Apache Hudi

Hudi (Uber 2017) focuses on *streaming upserts* and *incremental processing*. Offers two table types:

- *Copy-on-Write (CoW):* on update, rewrite the entire Parquet file containing the updated row. Best for read-heavy workloads; updates are expensive.
- *Merge-on-Read (MoR):* updates go to a delta log file; reads merge base files with delta logs. Best for high write throughput; reads are slightly more expensive.

```python
# Hudi upsert (PySpark)
df.write.format("hudi") \
  .option("hoodie.table.name", "orders") \
  .option("hoodie.datasource.write.recordkey.field", "order_id") \
  .option("hoodie.datasource.write.precombine.field", "updated_at") \
  .option("hoodie.datasource.write.operation", "upsert") \
  .option("hoodie.datasource.write.table.type", "MERGE_ON_READ") \
  .mode("append") \
  .save("s3://bucket/orders")

# Incremental query: reads only records changed since a commit
spark.read.format("hudi") \
     .option("hoodie.datasource.query.type", "incremental") \
     .option("hoodie.datasource.read.begin.instanttime", "20250101000000") \
     .load("s3://bucket/orders")
```

== Parquet File Format

All three formats use Parquet as the default file format. Parquet is a columnar, binary, self-describing format.

```
Parquet file layout:
┌─────────────────────────────────────────────────────┐
│  4-byte magic: PAR1                                  │
├─────────────────────────────────────────────────────┤
│  Row Group 0 (e.g., 128MB of rows)                  │
│    Column Chunk 0: col "id"                          │
│      Page 0: [1,2,3,...] (dict or plain encoded)     │
│      Page 1: [...]                                   │
│    Column Chunk 1: col "amount"                      │
│      Page 0: [99.5, 12.3, ...]                       │
│    ...                                               │
├─────────────────────────────────────────────────────┤
│  Row Group 1 ...                                     │
├─────────────────────────────────────────────────────┤
│  File Footer (Thrift-encoded metadata):              │
│    schema, row group offsets, column statistics      │
│    (min/max/null count per column chunk)             │
│  4-byte footer length                                │
│  4-byte magic: PAR1                                  │
└─────────────────────────────────────────────────────┘
```

*Column statistics* enable predicate pushdown:

```python
import pyarrow.parquet as pq

# Read file metadata without reading data
meta = pq.read_metadata("orders.parquet")
for rg in range(meta.num_row_groups):
    col_stats = meta.row_group(rg).column(0).statistics
    if col_stats.max < 100 or col_stats.min > 200:
        # Skip this row group entirely — all values outside query range
        continue
```

== Format Comparison

#table(
  columns: (auto, auto, auto, auto),
  [*Feature*], [*Iceberg*], [*Delta Lake*], [*Hudi*],
  [ACID],             [✓], [✓], [✓],
  [Time travel],      [✓], [✓], [✓],
  [Streaming upserts],[✓], [✓ (MERGE)], [✓ (native)],
  [Partition evolution],[✓ (hidden)],[✓ (Delta 3.0+; older versions require rewrite)],[limited],
  [Incremental reads],[via snapshots],[via change data feed],[native],
  [Multi-engine],     [Best (open spec)], [Good (open spec since 2023)], [Good],
  [Used by],          [Netflix, Apple, AWS],     [Databricks, Meta], [Uber, ByteDance],
)

== References

Armbrust, M. et al. (2020). "Delta Lake: High-Performance ACID Table Storage over Cloud Object Stores." VLDB.

Apache Iceberg. "Iceberg Table Spec." https://iceberg.apache.org/spec/

Zaharia, M. et al. (2021). "Lakehouse: A New Generation of Open Platforms that Unify Data Warehousing and Advanced Analytics." CIDR.

Shute, J. et al. (2013). "F1: A Distributed SQL Database That Scales." VLDB.

Apache Parquet. "Parquet File Format." https://parquet.apache.org/docs/file-format/

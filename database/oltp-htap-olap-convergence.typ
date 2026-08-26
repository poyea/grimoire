#import "../template.typ": xref

= OLTP / HTAP / OLAP Convergence

For three decades the database world split cleanly: OLTP systems (Oracle, DB2, MySQL, Postgres) optimized short, point-access transactions on row stores; OLAP systems (Teradata, Vertica, Redshift, BigQuery) optimized long, scan-heavy aggregations on column stores. The split was so entrenched that the standard architecture was a *nightly ETL pipeline* shuffling rows from the OLTP master into a separate warehouse. The 2010s eroded the boundary. Memory got cheap enough to keep working sets resident, compilers got good enough to fuse transactional and analytical code paths, and customers grew impatient with the staleness of a warehouse that lagged production by hours. The result is *HTAP* — Hybrid Transactional / Analytical Processing — and a family of systems that, by varying internal design choices, sit anywhere on the spectrum from "OLTP with bolt-on columns" to "OLAP with bolt-on rows."

*See also:* #xref("database", "column-stores-and-vectorized-execution", label: "Column Stores and Vectorized Execution"), #xref("database", "query-compilation", label: "Query Compilation"), #xref("database", "cloud-native-databases", label: "Cloud-Native Databases"), #xref("database", "storage-engines", label: "Storage Engines"), _Distributed Transactions_

== Why Converge?

The classic ETL warehouse has structural problems:

- *Staleness.* The warehouse trails the source by the ETL window — minutes at best, days in practice.
- *Schema drift.* Source schema changes break extracts; reconciling them is human-time.
- *Duplicated cost.* Two storage systems, two compute clusters, two operational on-call rotations.
- *Lost context.* By the time data lands in the warehouse, the application semantics (which row caused which event) are flattened into rolled-up facts.

A converged engine collapses these costs but pays in design complexity. The fundamental tension is that the two workloads want opposite physical layouts:

#table(
  columns: 3,
  [*Dimension*], [*OLTP wants*], [*OLAP wants*],
  [Row layout], [Row-major (whole tuple together)], [Column-major (one column at a time)],
  [Indexes], [Many secondary B-trees], [Few; zone maps and min/max suffice],
  [Concurrency], [Fine-grained MVCC, short transactions], [Long read snapshots, batch loads],
  [Compression], [Light or none], [Heavy (RLE, dictionary, bit-packing)],
  [Execution], [Tuple-at-a-time, prepared statements], [Vectorized or compiled scans],
  [Durability], [Per-commit fsync], [Bulk-load checkpoints],
)

Three architectural strategies emerge to resolve the conflict.

== The Three HTAP Architectures

=== Dual-Store (Replicated)

Keep an OLTP row store and an OLAP column store, replicate from one to the other in near real time. The classic example is *TiDB + TiFlash*: TiKV holds the row store, TiFlash holds a columnar replica fed by Raft learners. The optimizer picks row or column per scan.

```
Client write ──► TiKV (row, Raft leader) ──► WAL ──► TiKV followers
                       │
                       └─► TiFlash (column replica, Raft learner)
                                      │
                                      ▼
                              vectorized OLAP scans
```

Pros: each engine is locally optimal; failure of one does not corrupt the other. Cons: storage doubled; consistency between replicas requires care (TiFlash exposes "learner read with safe timestamp").

=== Unified-Store (Single Format with Indexes)

One physical store handles both. SAP HANA pioneered this with a hot delta (row) and main (columnar) inside a single transactional engine. *SingleStore* (formerly MemSQL) commits to it too: every table is either a *rowstore* (hash + skiplist in memory) or a *columnstore* (segments on disk with a row-format in-memory tail). The 2022 *Universal Storage* update unified them — a columnstore table with a per-row in-memory hash index serves both transactional point reads and analytical scans without replication.

=== Compiled-Unified (No Format Split)

Generate code that materializes whichever access pattern the query needs. *Umbra* (TUM, Neumann et al.) is the canonical example: a single buffer-managed storage layer with variable-size pages, accessed by LLVM-compiled query pipelines. There is no separate column store; analytical queries simply compile into vectorized scans over the same pages that OLTP transactions read and write.

== Umbra: The Compiled HTAP Engine

Umbra (Neumann \& Freitag, CIDR 2020) is the in-memory-disk hybrid successor to HyPer. Its design starts from the observation that pure in-memory systems (HyPer, Hekaton, Silo) cannot grow past RAM, while pure disk systems pay a perpetual buffer-pool tax even when data fits in cache.

=== Variable-Size Pages and LeanStore

Umbra uses *variable-size pages* (powers of two from 64 KB to several MB) managed by a buffer manager descended from *LeanStore*. The page identifier is a *swip*: a 64-bit value that is either a memory pointer (resident) or a disk address (evicted). The first access pays a cache-miss-style check; subsequent accesses are essentially free pointer dereferences.

```cpp
// Conceptual swip access
template<class T>
T* resolve(Swip<T>& s) {
    if (s.isUnswizzled())              // bit pattern test
        bufferManager.loadAndSwizzle(s);
    return s.asPtr();                  // direct pointer afterwards
}
```

The variable-size pages let large objects (BLOBs, sorted runs, hash-table partitions) live on a single page instead of being chained. Combined with *pointer swizzling*, the steady-state cost of resident data is the same as a pure in-memory database.

=== Code Generation Without Long Compile Latency

Umbra generates LLVM IR per query, but adds a custom *flying-start interpreter* and a tiered compiler. Short queries skip optimization and run via a fast bytecode; long queries get full LLVM `-O3`. The result is that OLTP-class queries pay $approx 10$ µs of preparation while OLAP queries reach hand-tuned C performance.

=== Morsel-Driven Parallelism and MVCC

Like HyPer, Umbra uses *morsel-driven parallelism* (Leis et al., SIGMOD 2014): tables split into morsels (small chunks, typically 100 K tuples) dispatched to a thread pool. MVCC is implemented with *precision locking* and version chains kept short via aggressive garbage collection. Transactional and analytical queries share the same scan path; analytical reads see a consistent snapshot without blocking writes.

== SingleStore (MemSQL)

SingleStore began (2012) as MemSQL, an in-memory rowstore optimized for OLTP at scale, then added a disk-backed columnstore for analytics, then in 2022 unified them.

=== Rowstore Internals

Rowstore tables live entirely in memory, durable via a write-ahead log to disk. Indexes are *lock-free skiplists* (primary and secondary). The storage format is a tuple-at-a-time slot directory. Transactions are MVCC with per-partition coordinators.

=== Columnstore and Segments

Columnstore tables are organized into *segments* of $approx 1$ million rows. Each segment stores each column compressed (RLE, dictionary, LZ4). Segments are immutable; updates go to a per-segment *row-format delta* until a background flush rewrites the segment.

=== Universal Storage

Universal Storage adds a *hash index* — pointed at row offsets within columnstore segments — that turns the columnstore into a viable OLTP target. A `SELECT * WHERE pk = ?` does an index lookup, gets a (segment, row) pair, and reconstructs the row by gathering each column at that offset. Skiplists handle secondary indexes. The result: one storage format, OLTP latencies for indexed point queries, OLAP throughput for scans.

```sql
CREATE TABLE orders (
    order_id BIGINT NOT NULL,
    customer_id BIGINT,
    amount DECIMAL(12,2),
    placed_at DATETIME,
    SORT KEY (placed_at),         -- columnstore ordering
    UNIQUE HASH (order_id),       -- OLTP point lookup
    KEY (customer_id) USING HASH  -- secondary
);
```

The `SORT KEY` keeps segments clustered for analytical range scans; the `UNIQUE HASH` powers transactional access. SingleStore's planner picks between segment-scan-with-pushdown and index-then-gather based on selectivity.

== DuckDB on DuckLake

DuckDB is the embedded analytical engine (see #xref("database", "embedded-databases", label: "Embedded Databases") and _Column Stores_). For a long time it stayed strictly OLAP and strictly local. *DuckLake* (announced 2024) is an open table format that gives DuckDB a multi-writer, transactional, lakehouse-style storage layer — pushing it toward the HTAP edge from the OLAP side.

=== The DuckLake Format

DuckLake stores table metadata in a SQL catalog (Postgres, SQLite, or another DuckDB) rather than in JSON manifest files on object storage (the choice Iceberg and Delta made). Data files remain Parquet on S3/GCS. The insight: a real SQL catalog handles concurrent metadata operations (snapshot creation, schema evolution, compaction) using ordinary database transactions, eliminating the file-rename and atomic-pointer-swap gymnastics that Iceberg and Delta need.

```
┌────────────────────────────────────────┐
│ DuckLake catalog (Postgres)            │
│  - tables, schemas, snapshots          │
│  - file manifests as SQL rows          │
│  - ACID via Postgres transactions      │
└──────────────┬─────────────────────────┘
               │
               ▼
   Parquet data files on object storage
               ▲
               │
       ┌───────┴────────┐
   DuckDB process #1   DuckDB process #2   (any number, no leader)
```

=== Transactional Writes from an Analytical Engine

A `BEGIN; INSERT ...; COMMIT;` against a DuckLake table opens a transaction against the catalog database, writes new Parquet files to object storage, and atomically updates the snapshot pointer in the catalog. Multi-statement transactions are serializable because the catalog enforces them. DuckDB thereby acquires OLTP-style ACID without taking on the row-store complexity of a transactional storage engine — the heavy lifting is delegated to Postgres.

=== Why It Matters for HTAP

DuckLake makes the analytical engine *transactional enough* to absorb event-stream ingestion (small frequent commits) that previously needed a separate OLTP database. For workloads where the OLTP side is mostly append, DuckLake plus DuckDB collapses the dual-store HTAP stack into one columnar lake.

== Snowflake Hybrid Tables

Snowflake spent its first decade as a pure OLAP warehouse with micro-partition columnar storage on object storage. *Unistore* (GA 2024) introduces *Hybrid Tables*: row-oriented, in-place-updatable, secondary-indexed tables that live alongside columnar tables in the same warehouse and participate in the same transactions.

=== Architecture

Hybrid Tables use a separate storage tier — a distributed row store with primary and secondary B-tree indexes — replicated for durability and queryable from any virtual warehouse. The columnar side is unchanged. Cross-table joins between row and columnar tables run inside one query plan; cross-table transactions are serializable via Snowflake's transaction manager.

```sql
CREATE HYBRID TABLE customers (
    id NUMBER PRIMARY KEY,
    email STRING UNIQUE,
    region STRING,
    INDEX idx_region (region)
);

-- Mixed query: hybrid join columnar, both in one snapshot
SELECT c.email, SUM(o.amount)
FROM customers c                   -- hybrid (row, indexed)
JOIN orders_history o              -- columnar (Snowflake table)
  ON c.id = o.customer_id
WHERE c.region = 'EU'              -- index seek on hybrid
GROUP BY c.email;
```

=== Tradeoffs

Hybrid Tables are slower for pure scans than columnar tables and more expensive per byte than object storage. They are not a wholesale replacement for an OLTP database — Snowflake positions them for operational lookup tables, dimension tables, and small-to-medium workloads where keeping data inside the warehouse is worth the cost.

== Aurora DSQL: Active-Active OLTP at Cloud Scale

Aurora DSQL (AWS, GA 2024) sits on the OLTP side but inherits the disaggregated cloud-native architecture (see #xref("database", "cloud-native-databases", label: "Cloud-Native Databases")). Its convergence story is *geographic*: a single logical OLTP database that accepts writes in multiple regions with strong consistency, removing the historical OLTP constraint of "one writer, many read replicas."

=== Architectural Sketch

Per the AWS announcements (re\:Invent 2024), DSQL decouples three concerns:

1. *Query processors* (stateless Postgres-compatible front-ends) that run in every region.
2. *Adjudicator / journal* — a distributed log responsible for ordering transactions across regions. Uses a deterministic timestamp protocol (related to but distinct from Spanner's TrueTime; AWS uses synchronized clocks plus a transaction commit protocol that resolves conflicts at commit time).
3. *Storage* — a multi-tenant, regionally replicated KV layer holding indexed key-range data.

Transactions execute *optimistically* in any region: the query processor buffers reads and writes, then submits them to the journal. The journal serializes globally, detects write-write conflicts, and either commits or aborts.

=== Why "Active-Active"

Earlier multi-region OLTP options forced a choice: pick one region as primary (Aurora Global Database), accept eventual consistency (DynamoDB Global Tables), or accept Spanner-style commit-wait latency for every write. DSQL keeps strong consistency but pays the cost only on *conflicting* transactions; non-conflicting writes commit at single-region latency. The convergence angle: DSQL closes the gap between geo-distributed analytical platforms (BigQuery, Snowflake) and geo-distributed transactional ones.

== Comparison

#table(
  columns: 4,
  [*System*], [*Primary identity*], [*Convergence strategy*], [*HTAP position*],
  [Umbra], [Research OLTP+OLAP], [Compiled unified store], [Strong HTAP],
  [SingleStore], [Distributed SQL DB], [Universal Storage (one format + indexes)], [Strong HTAP],
  [TiDB + TiFlash], [Distributed OLTP], [Dual-store via Raft learners], [Replicated HTAP],
  [SAP HANA], [In-memory enterprise], [Delta/main column store], [Unified HTAP],
  [AlloyDB], [Managed Postgres], [Columnar engine on top of row store], [Replicated HTAP],
  [DuckDB + DuckLake], [Embedded OLAP], [SQL-catalog lakehouse for ACID], [OLAP leaning HTAP],
  [Snowflake Hybrid], [Cloud OLAP], [Bolt-on row tier (Unistore)], [OLAP leaning HTAP],
  [Aurora DSQL], [Cloud-native OLTP], [Active-active geo-distributed], [OLTP (geo-converged)],
)

== Open Problems

- *Workload isolation.* In any unified-store HTAP, a runaway analytical scan can starve OLTP latency SLOs. Resource governors, separate thread pools, and admission control (à la Hekaton's resource pools) are partial answers; the research frontier is *learned admission control* tied to query plans.
- *Snapshot cost.* Long analytical reads keep many MVCC versions live, inflating undo / version chains. Hyrise, Umbra, and SingleStore have all published custom GC strategies; none is fully solved.
- *Cost models that include layout choice.* The optimizer in a Universal-Storage system must decide *per scan* whether to use the row index or the columnar segments. Cardinality estimation errors that were merely annoying in OLAP now cost OLTP latency.
- *Open table formats and OLTP.* Iceberg, Delta, Hudi, and DuckLake demonstrate ACID over object storage for append-heavy workloads. Sub-millisecond point updates on object-storage-backed formats remain an open question; current designs (Hudi's MoR, Delta's deletion vectors, Iceberg v3 row-level updates) trade off freshness against compaction cost.
- *Geo-active-active with low-conflict workloads.* DSQL, CockroachDB, YugabyteDB, and Spanner each have a different point on the (latency, conflict-cost, strength) curve. A general theory of "best protocol given the conflict graph" is missing.

== Further Reading

Neumann, T., Freitag, M. (2020). "Umbra: A Disk-Based System with In-Memory Performance." CIDR.

Freitag, M., Bandle, M., Schmidt, T., Kemper, A., Neumann, T. (2020). "Adopting Worst-Case Optimal Joins in Relational Database Systems." VLDB.

Leis, V., Haubenschild, M., Kemper, A., Neumann, T. (2018). "LeanStore: In-Memory Data Management Beyond Main Memory." ICDE.

Kemper, A., Neumann, T. (2011). "HyPer: A Hybrid OLTP\&OLAP Main Memory Database System Based on Virtual Memory Snapshots." ICDE.

Neumann, T. (2011). "Efficiently Compiling Efficient Query Plans for Modern Hardware." VLDB.

Leis, V. et al. (2014). "Morsel-Driven Parallelism: A NUMA-Aware Query Evaluation Framework for the Many-Core Age." SIGMOD.

Färber, F. et al. (2012). "SAP HANA Database: Data Management for Modern Business Applications." SIGMOD Record.

SingleStore (2022). "Universal Storage: Fast Analytics, Fast Transactions, One Table." engineering.singlestore.com.

SingleStore Documentation. "Columnstore" and "Rowstore" reference, docs.singlestore.com.

Shamis, A. et al. (2023). "Building a Hybrid Transactional/Analytical Processing Engine on Top of a Distributed SQL Database." (SingleStore VLDB Industrial).

Müller, M., Raasveldt, M., Mühleisen, H. (2024). "DuckLake: SQL as a Lakehouse Format." duckdb.org/2024/06/26/ducklake announcement and design notes.

Raasveldt, M., Mühleisen, H. (2019). "DuckDB: An Embeddable Analytical Database." SIGMOD demo.

Dageville, B. et al. (2016). "The Snowflake Elastic Data Warehouse." SIGMOD.

Snowflake (2024). "Unistore and Hybrid Tables: Architecture Overview." docs.snowflake.com and Snowflake Summit talks.

AWS (2024). "Introducing Amazon Aurora DSQL." aws.amazon.com/blogs/database announcement, re\:Invent 2024 deep-dive sessions DAT405 and DAT406.

Vogels, W. (2024). "A Decade of Aurora: From Single-Writer to Active-Active." All Things Distributed.

Huang, D. et al. (2020). "TiDB: A Raft-Based HTAP Database." VLDB.

Bortnikov, V. et al. (2024). "AlloyDB: A Cloud-Native Database for HTAP." VLDB Industrial.

Özcan, F., Tian, Y., Tözün, P. (2017). "Hybrid Transactional/Analytical Processing: A Survey." SIGMOD tutorial.

Makreshanski, D. et al. (2017). "BatchDB: Efficient Isolated Execution of Hybrid OLTP+OLAP Workloads for Interactive Applications." SIGMOD.

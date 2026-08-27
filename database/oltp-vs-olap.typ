#import "../template.typ": xref

= OLTP vs OLAP <oltp-vs-olap>

Databases are commonly partitioned into two workload archetypes: *OLTP* (Online Transactional Processing), which serves high-concurrency short requests against individual records, and *OLAP* (Online Analytical Processing), which executes long-running aggregations over large portions of a dataset. Understanding this divide determines the appropriate storage layout, concurrency model, index strategy, and hardware provisioning for a system. Modern *HTAP* engines blur the boundary by serving both workloads from a single system, trading some specialisation for operational simplicity.

*See also:* #xref("database", "oltp-htap-olap-convergence", label: "OLTP / HTAP / OLAP Convergence"), #xref("database", "storage-engines", label: "Storage Engines"), #xref("database", "columnar-storage-and-vectorization", label: "Columnar Storage and Vectorization"), #xref("database", "transactions-distributed", label: "Distributed Transactions")

== Workload Characteristics

=== Access Patterns

*OLTP* workloads are characterised by narrow access: a transaction reads or modifies a small number of rows identified by primary key or a selective index. A typical e-commerce checkout touches fewer than 20 rows across 3–5 tables.

*OLAP* workloads are characterised by wide access: a query may aggregate hundreds of millions of rows but read only 2–5 of 200 columns. Access is sequential, not random; indexes are rarely beneficial.

#table(
  columns: (auto, auto, auto),
  [*Dimension*], [*OLTP*], [*OLAP*],
  [Request scope],     [Single rows / short ranges], [Full table scans, large ranges],
  [Columns accessed],  [Most or all columns],        [Few columns, many rows],
  [Operation mix],     [~50% read, ~50% write],      [>95% read],
  [Response target],   [< 10 ms (p99)],              [Seconds to minutes],
  [Concurrency],       [Thousands of sessions],      [Tens of concurrent queries],
  [Data volume],       [GB to low TB (working set)], [TB to PB (full history)],
  [Data freshness],    [Real-time],                  [Near-real-time to daily batch],
)

== OLTP: Transactional Systems

=== Point Reads and Writes

OLTP engines optimise for *point lookups* (fetch one row by key) and *point writes* (insert, update, or delete one row). These are served by:

- *B-Tree indexes* (PostgreSQL heap + B-Tree, InnoDB clustered B-Tree) providing $O(log N)$ lookup in 2–4 I/Os.
- *Buffer pool*: the working set of an OLTP database (often < 100 GB) fits largely in DRAM, so most reads are served from cache.

```sql
-- Typical OLTP transaction: read-modify-write on 3 rows.
BEGIN;
  SELECT balance FROM accounts WHERE id = 42 FOR UPDATE;
  UPDATE accounts SET balance = balance - 100 WHERE id = 42;
  UPDATE accounts SET balance = balance + 100 WHERE id = 99;
  INSERT INTO audit_log (from_id, to_id, amount, ts)
         VALUES (42, 99, 100, NOW());
COMMIT;
```

=== High Concurrency and ACID

OLTP databases expose *ACID* guarantees:

- *Atomicity*: the transaction commits or rolls back entirely (WAL + undo log).
- *Consistency*: integrity constraints (FK, UNIQUE, CHECK) are enforced at commit.
- *Isolation*: concurrent transactions observe a consistent view; implemented via MVCC or strict two-phase locking.
- *Durability*: committed data survives crashes (WAL flushed to durable storage before ACK).

*MVCC* (Multi-Version Concurrency Control) allows readers and writers to proceed concurrently without blocking each other by maintaining multiple row versions. PostgreSQL stores old row versions in the heap ("dead tuples") until `VACUUM` reclaims them; InnoDB keeps undo chains in a separate undo tablespace.

```
MVCC read (Snapshot Isolation):
  Transaction T2 starts at timestamp t=100.
  Concurrent transaction T1 (t=95) modifies row R → creates version R@t=95.
  T2 reads R → sees version valid at t=100, which may be original R@t=0
  (T1 not yet committed) or R@t=95 (T1 already committed before t=100).
  T2 never blocks on T1.
```

=== Row Storage and Index Structures

Row-oriented storage keeps all columns of a row together on disk, which is efficient for full-row fetches but wastes I/O for column-selective aggregations. A typical OLTP row layout (PostgreSQL heap tuple):

```
┌───────────────────────────────────────────────────────┐
│ t_xmin (4B) │ t_xmax (4B) │ t_ctid (6B) │ t_infomask │
│ NULL bitmap │ col1 │ col2 │ col3 │ ... │ colN          │
└───────────────────────────────────────────────────────┘
```

Each tuple carries transaction visibility metadata (`t_xmin`, `t_xmax`) enabling MVCC snapshot reads without a separate version store.

== OLAP: Analytical Systems

=== Full Scans and Aggregations

OLAP queries scan large fractions of a table, filtering on low-selectivity predicates and computing aggregations. The canonical *TPC-H* benchmark Query 1 illustrates this pattern:

```sql
-- TPC-H Q1: scan ~600M rows, group by 2 columns.
SELECT l_returnflag, l_linestatus,
       SUM(l_quantity), SUM(l_extendedprice),
       SUM(l_extendedprice * (1 - l_discount)),
       AVG(l_quantity), AVG(l_extendedprice), AVG(l_discount),
       COUNT(*)
FROM lineitem
WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;
```

Row-store engines on TPC-H Q1 are typically 10–100× slower than columnar engines because they read all ~16 columns per row when only 5 are needed.

=== Columnar Storage and MPP

OLAP engines use *columnar storage* (see `database/columnar-storage-and-vectorization.typ`) and *MPP* (Massively Parallel Processing) architectures:

- *Shared-nothing MPP*: each node holds a partition of the data and processes its slice independently; results are shuffled across the network and merged.
- *Query parallelism*: large scans are split across cores (intra-node) and nodes (inter-node) using exchange operators.

Systems: Snowflake (cloud-native, shared storage MPP), BigQuery (serverless columnar, Dremel engine), Redshift (shared-nothing columnar), ClickHouse (single-node and replicated columnar).

== HTAP: Hybrid Transactional/Analytical Processing

*HTAP* systems serve both OLTP and OLAP workloads from a unified engine, eliminating the latency of ETL pipelines. The key challenge is that OLTP favours row layout and OLAP favours columnar layout — HTAP systems resolve this with one of three strategies:

=== Dual Format (In-Memory Column Store Delta)

Maintain a *row store* for writes and an *in-memory columnar store* (delta store) that receives real-time updates. Analytical queries merge both stores. Example: SAP HANA (row + column delta), Oracle In-Memory Column Store.

```
HTAP dual format:
  Writes → Row store (B-Tree, MVCC, ACID)
                ↓ background migration (seconds to minutes lag)
  Reads  → Column store (SIMD vectorised, aggregations)
           + Row store delta (recent uncommitted / unmerged rows)
```

=== Distributed Row + Columnar Replica

Maintain row-store replicas for OLTP and asynchronously replicate to columnar learner replicas for OLAP. Example: *TiDB* (TiKV row store + TiFlash columnar replica, Raft-based replication), *CockroachDB* (row store; analytical queries possible but limited columnar optimisation).

=== Unified Columnar with MVCC

Use a single columnar layout but add full MVCC, serialisable isolation, and row-level locking. Example: *SingleStore* (MemSQL) stores data in both a row-based in-memory store and a disk-based columnar store, routing each query to the appropriate format automatically.

== Data Warehouse Architecture

A *data warehouse* is an OLAP system designed to support historical reporting and business intelligence. The canonical schema patterns are:

=== Star Schema

A *star schema* has one central *fact table* surrounded by *dimension tables*. The fact table holds measurable events (sales, clicks, shipments) with foreign keys to dimensions (time, product, customer, geography).

```
          ┌──────────────┐
          │ dim_product   │
          └──────┬───────┘
                 │
┌──────────┐     │     ┌──────────────┐
│ dim_time │─────●─────│  fact_sales  │─────┌──────────────┐
└──────────┘     │     │  (FK keys,   │     │ dim_customer  │
                 │     │   measures)  │     └──────────────┘
          ┌──────┴───────┐
          │ dim_geography │
          └──────────────┘
```

Star schema enables simple joins (fact → dim) and is well-supported by columnar optimisers that can push dimension filters before the join.

=== Snowflake Schema

A *snowflake schema* normalises dimension tables into sub-dimensions, reducing redundancy at the cost of additional joins.

```
dim_product → dim_category → dim_department
```

Most modern data warehouse optimisers handle snowflake schemas efficiently through join ordering and predicate pushdown, so the choice is primarily a data modelling decision.

=== Fact and Dimension Tables

*Fact tables* are wide (many FK columns + measure columns) and deep (billions of rows). *Dimension tables* are narrow and shallow (millions of rows at most). A columnar engine scans the fact table column by column, looking up dimension attributes only for rows that survive fact-side predicates — this is *late materialisation* applied to star joins.

== ETL vs ELT

*ETL* (Extract, Transform, Load) runs transformations in a dedicated pipeline before data reaches the warehouse. *ELT* (Extract, Load, Transform) loads raw data first and runs transformations inside the warehouse engine using SQL.

#table(
  columns: (auto, auto, auto),
  [*Aspect*], [*ETL*], [*ELT*],
  [Transform location],  [External pipeline (Spark, dbt)], [Inside warehouse (SQL)],
  [Latency],             [Minutes to hours],               [Near-real-time possible],
  [Raw data retention],  [Often discarded],                [Raw data preserved],
  [Tooling],             [Informatica, Spark, Glue],       [dbt, Dataform, Fivetran],
  [Typical target],      [Traditional warehouse],          [Cloud warehouse / lakehouse],
)

Modern *lakehouse* architectures (Delta Lake, Apache Iceberg) blur ETL/ELT by storing raw and transformed data in the same open-format table, with schema enforcement applied incrementally.

== Materialised Views

A *materialised view* pre-computes and stores the result of a query, turning expensive aggregations into cheap scans. They are central to both OLAP and HTAP:

```sql
-- PostgreSQL materialised view for a frequently run report.
CREATE MATERIALIZED VIEW monthly_revenue AS
  SELECT DATE_TRUNC('month', order_date) AS month,
         region,
         SUM(amount) AS revenue
  FROM orders
  GROUP BY 1, 2;

-- Refresh on demand or on a schedule:
REFRESH MATERIALIZED VIEW CONCURRENTLY monthly_revenue;
```

*Incremental view maintenance* (IVM) — supported by some systems (Materialize, Neon, ksqlDB) — updates the materialised view incrementally as base table changes arrive, avoiding full recomputation.

== Query Complexity Comparison

#table(
  columns: (auto, auto, auto),
  [*Complexity aspect*], [*OLTP*], [*OLAP*],
  [Typical SQL length],         [< 20 lines],         [20–200 lines],
  [Number of joins],            [2–5],                [5–20+],
  [Subqueries / CTEs],         [Rare],               [Common],
  [Window functions],           [Rare],               [Very common],
  [Execution time target],      [< 10 ms],            [Seconds–minutes],
  [Plan stability requirement], [High (OLTP SLA)],    [Lower (batch tolerance)],
)

== When to Use Which

Choose *OLTP* (PostgreSQL, MySQL, CockroachDB) when: transactions require ACID guarantees, access patterns are point-key-based, concurrency is high, and data freshness requirements are sub-second.

Choose *OLAP / data warehouse* (Snowflake, BigQuery, ClickHouse, Redshift) when: queries aggregate large historical datasets, read-to-write ratio is >10:1, and analytical latency of seconds is acceptable.

Choose *HTAP* (TiDB + TiFlash, SingleStore, Oracle In-Memory) when: operational analytics on fresh data is required (fraud detection, real-time dashboards) and the cost of maintaining separate OLTP + OLAP pipelines is prohibitive.

== Convergence Trends

The OLTP/OLAP divide is narrowing. Several forces drive convergence:

- *Cloud storage decoupling*: both OLTP and OLAP engines now use cloud object storage (S3, GCS) as the durability layer, enabling shared data without ETL.
- *In-memory columnar deltas*: OLTP engines (PostgreSQL with pg\_analytics / ParadeDB, Oracle In-Memory) add columnar acceleration without schema changes.
- *Open table formats*: Apache Iceberg and Delta Lake provide ACID transactions on columnar Parquet files, enabling SQL engines to serve both analytical and transactional queries on the same dataset.
- *Serverless elasticity*: cloud warehouses (Snowflake, BigQuery) auto-scale compute, closing the latency gap for short queries that previously required an always-on OLTP engine.

== Further Reading

Stonebraker, M. et al. (2007). "The End of an Architectural Era (It's Time for a Complete Rewrite)." VLDB.

Pavlo, A., Aslett, M. (2016). "What's Really New with NewSQL?" ACM SIGMOD Record.

Huang, D. et al. (2020). "TiDB: A Raft-Based HTAP Database." VLDB.

Armbrust, M. et al. (2021). "Lakehouse: A New Generation of Open Platforms that Unify Data Warehousing and Advanced Analytics." CIDR.

Kimball, R., Ross, M. (2013). *The Data Warehouse Toolkit*, 3rd ed. Wiley.

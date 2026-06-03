= Batch Processing

Batch processing is the workhorse of analytics: read a large bounded dataset, transform it through a $"DAG"$ of operators, write the result. The lineage runs MapReduce (2004) → Hive → Spark → Photon-class vectorized engines. This chapter covers Spark internals — Catalyst, Tungsten, Adaptive Query Execution ($"AQE"$), the shuffle, and the vectorized successors — because Spark remains the dominant engine for petabyte-scale jobs.

*See also:* _ETL vs ELT_, _Lakehouse Engineering_ (the storage layer Spark reads), _Orchestration_ (how batch jobs are scheduled), `database/query-processing.typ`, `database/columnar-storage-and-vectorization.typ`.

== The MapReduce Heritage

The 2004 MapReduce paper isolated three ideas: a functional `map` / `reduce` API, a shuffle that groups by key, and a fault model that restarts failed tasks from disk-checkpointed inputs. Hadoop $"MR"$ inherited two costs: every stage materialized to $"HDFS"$, and the $"JVM"$ map / reduce interface was slow per row.

```
input → map → shuffle (sort + partition by key) → reduce → output
                  ↓
              spill to disk
```

Spark's two contributions, on top of $"MR"$:

1. *In-memory pipelining* across narrow stages, with shuffle only at wide dependencies.
2. *Lineage-based recovery* ($"RDD"$) instead of checkpointing every stage.

== Spark Execution Model

A $"PySpark"$ query compiles to a logical plan, then a physical plan, then a $"DAG"$ of stages. A *stage* is a chain of narrow transformations bounded by shuffles.

```python
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("orders").getOrCreate()

orders   = spark.read.parquet("s3://bronze/orders/")
products = spark.read.parquet("s3://bronze/products/")

revenue_by_cat = (
    orders
      .filter(F.col("status") == "PAID")
      .join(F.broadcast(products), "product_id")  # broadcast: small side
      .groupBy("category")
      .agg(F.sum("amount").alias("revenue"))
)

revenue_by_cat.write.mode("overwrite").parquet("s3://gold/revenue_by_category/")
```

The filter and join (because `products` is broadcast) stay narrow; the `groupBy` triggers a shuffle. Inspect with `explain(mode="formatted")`.

== Catalyst (Logical Optimization)

Catalyst is a rule-based + cost-based query optimizer for tree-shaped logical plans. Standard rules: predicate pushdown into the scan, projection pruning, constant folding, join reordering. Cost-based join planning uses Parquet / Iceberg column statistics.

A scan over Parquet with a filter `dt = '2026-05-31' AND amount > 100`:

- *Predicate pushdown:* `dt` becomes a partition filter (no read), `amount > 100` becomes a Parquet row-group filter via min/max stats.
- *Projection pruning:* only the columns referenced downstream are read from disk.

== Tungsten (Physical Execution)

Tungsten is Spark's whole-stage code generation and off-heap binary memory layout. Two ideas:

- *Whole-stage codegen:* operators within a stage are fused into one $"JVM"$ method, eliminating virtual calls and per-row boxing.
- *UnsafeRow:* rows are stored as a fixed-offset byte buffer off-heap, with cache-friendly access and no $"GC"$ pressure.

The result is 5–10× speedup on $"CPU"$-bound aggregations over the iterator model.

== Adaptive Query Execution ($"AQE"$)

$"AQE"$ (Spark 3+) re-plans at shuffle boundaries using actual partition statistics. Three rewrites:

- *Coalesce shuffle partitions:* if the default 200 shuffle partitions produce many tiny outputs, fuse them.
- *Convert sort-merge join to broadcast join* when one side turns out small.
- *Skew join handling:* split outlier partitions and replicate the other side.

```
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
```

$"AQE"$ moved Spark closer to a true cost-based engine without requiring up-front stats.

== The Shuffle

Shuffle is the dominant cost of most jobs. The default $"SortShuffleManager"$:

1. Each map task partitions its output by key hash into `numReducers` buckets, spilling to local disk in sorted runs.
2. Reducers fetch their bucket from every map output over the network.
3. Reducer-side merge sorts the runs.

Optimizations: *push-based shuffle* (map outputs aggregated by external shuffle service to reduce fetch fan-out), *Cloud Shuffle Service* / disaggregated shuffle on $"S3"$.

== Photon and the Vectorized Successors

Photon (Databricks), Velox (Meta), Gluten + ClickHouse — the next generation replaces the row-at-a-time $"JVM"$ engine with a vectorized C++ engine that operates on Arrow-shaped columnar batches. Speedups of 2–5× on $"TPC-DS"$-class workloads come from $"SIMD"$, tighter cache locality, and elimination of $"JVM"$ overhead.

#table(
  columns: 4,
  [*Engine*], [*Language*], [*Execution*], [*Notes*],
  [Spark $"SQL"$ + Tungsten], [$"JVM"$ + codegen], [Whole-stage row], [Default],
  [Photon], [C++ vectorized], [Columnar batches], [Databricks only],
  [Velox], [C++ vectorized], [Columnar batches], [Open library, Presto / Spark backend],
  [Gluten], [Spark $"SQL"$ + Velox], [Columnar batches], [Drop-in for Spark],
)

== Partitioning and File Layout

Job runtime is dominated by I/O for analytical workloads. Three rules:

- *Partition by the filter you scan on most often* (usually date). Avoid high-cardinality partitions ($> 10^4$ files in a single directory creates listing pain).
- *Target 128 MB–1 GB Parquet files.* Smaller files explode metadata; larger files limit parallelism and skew.
- *Sort within partitions by the next-most-selective predicate* (Z-order / Hilbert in Delta / Iceberg) to maximize row-group pruning.

```python
(orders
   .repartition("dt")           # one file per partition
   .sortWithinPartitions("user_id")
   .write.partitionBy("dt").parquet("s3://silver/orders/"))
```

== Hive: What Remains

Hive (2008) gave $"SQL"$ on $"HDFS"$ and the Hive Metastore — a table catalog still used today as the namespace under Iceberg and Delta. Hive query execution (Tez, $"LLAP"$) is largely retired in new builds; the Hive Metastore $"API"$ (or its successor, Unity Catalog / Polaris / Glue) lives on.

== Failure Model

Spark restarts failed tasks based on $"RDD"$ lineage: the missing partition is recomputed from its parents. Stage retries are bounded (default 4). Speculative execution duplicates slow tasks; the first finisher wins. For long jobs, *checkpoint* to $"S3"$ to truncate lineage and avoid recomputing 12 hours of work after a single executor loss:

```python
sc.setCheckpointDir("s3://chk/")
big_df.checkpoint()
```

== Pitfalls

- *Wide narrow trap.* `groupBy().agg()` on a high-cardinality key with default 200 shuffle partitions creates either skew or too-small files. Set `spark.sql.shuffle.partitions` deliberately or rely on $"AQE"$.
- *Driver $"OOM"$ from `collect()`*. Always cap row counts.
- *$"UDF"$ tax.* Python $"UDF"$s serialize each row through Arrow; prefer $"SQL"$ or pandas $"UDF"$s.
- *Small-file plague* on streaming ingest. Run periodic `OPTIMIZE` / compaction.

== Further Reading

Dean, J., Ghemawat, S. (2004). "MapReduce: Simplified Data Processing on Large Clusters." OSDI.

Zaharia, M. et al. (2012). "Resilient Distributed Datasets." NSDI.

Armbrust, M. et al. (2015). "Spark $"SQL"$: Relational Data Processing in Spark." SIGMOD.

Behm, A. et al. (2022). "Photon: A Fast Query Engine for Lakehouse Systems." SIGMOD.

Pedreira, P. et al. (2022). "Velox: Meta's Unified Execution Engine." VLDB.

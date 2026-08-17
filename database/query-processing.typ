#import "../template.typ": xref

= Query Processing

A SQL query passes through a multi-stage pipeline before any data is touched: the engine first validates and resolves names, then builds a logical plan, optimises it with algebraic transformations and cost estimates, and finally executes it using a chosen evaluation strategy. Understanding each stage explains why the same SQL can run in 10 ms or 10 minutes depending on plan choices, and why modern engines invest heavily in compilation and adaptive re-optimisation.

*See also:* #xref("database", "query-optimization", label: "Query Optimization"), #xref("database", "sql-engines-internals", label: "SQL Engine Internals"), #xref("database", "joins-and-aggregation", label: "Joins and Aggregation"), #xref("database", "columnar-storage-and-vectorization", label: "Columnar Storage and Vectorization")

== Query Lifecycle

```
SQL text
  │
  ▼
┌──────────────┐
│  Parser       │  Tokenise, build AST.  Syntax errors caught here.
└──────┬───────┘
       │ AST
  ▼
┌──────────────┐
│  Binder       │  Resolve table/column names against the catalog.
│  (Analyzer)   │  Type-check expressions.  Semantic errors caught here.
└──────┬───────┘
       │ Bound logical plan
  ▼
┌──────────────┐
│  Planner      │  Convert AST → initial logical operator tree
│  (Logical)    │  (Scan, Filter, Join, Aggregate, Sort, Project).
└──────┬───────┘
       │ Logical plan
  ▼
┌──────────────┐
│  Optimizer    │  Apply rewrite rules (predicate pushdown, column pruning,
│               │  join reordering).  Cost-model search for physical plan.
└──────┬───────┘
       │ Physical plan (operators with chosen algorithms)
  ▼
┌──────────────┐
│  Executor     │  Drive operators to produce result tuples.
└──────────────┘
```

=== Parse

The *parser* converts SQL text into an Abstract Syntax Tree (AST). Most engines use a hand-written recursive-descent parser or a Bison/YACC grammar. Errors at this stage are purely syntactic ("unexpected token"). The parser does not consult the catalog — it knows nothing about whether `users` is a real table. PostgreSQL's gram.y is a ~16 000-line Bison grammar; DuckDB uses a hand-written recursive-descent parser for better error messages and incremental extension.

Modern systems track source locations for each AST node so that error messages can point to the exact character in the original query. Some engines (e.g., Calcite) produce a SQL-standard AST and normalise dialect differences at this layer.

=== Bind

The *binder* (or analyser) walks the AST and resolves every name against the *catalog* (schema metadata): table names, column names, function signatures, and type information. It produces a *bound logical plan* where every node carries resolved types. Semantic errors ("column `foo` does not exist") are raised here.

Binding involves: (1) resolving table references including CTEs, subqueries, and views by inlining their definitions; (2) column disambiguation — figuring out which table a bare `id` refers to when multiple tables are in scope; (3) function overload resolution matching argument types; (4) implicit cast insertion where the type system allows (e.g., promoting an integer literal to DECIMAL for comparison). The output is a fully-typed, unambiguous logical plan tree ready for optimisation.

=== Plan and Optimise

The *planner* generates an initial *logical plan* — a tree of relational algebra operators (Scan, Filter, Project, Join, Aggregate, Sort) — directly from the bound AST. This logical plan is correct but naive: it preserves the query's written structure including the order of joins and placement of predicates.

The *optimizer* then applies *rewrite rules* to the logical plan before selecting physical operators. Logical rewrites include predicate pushdown (moving Filter nodes toward leaf Scans), column pruning (eliminating Project columns unused by upstream operators), subquery unnesting (converting correlated subqueries to joins), and join reordering (permuting join inputs to minimize intermediate cardinality). After logical rewrites, the optimizer selects a *physical plan* in which each logical operator is replaced by a concrete algorithm — Hash Join vs Sort-Merge Join, sequential scan vs index scan, hash aggregate vs sorted aggregate — guided by cost estimates from the catalog's column statistics (histograms, NDV, null fraction). This cost-based search is the most complex stage; see `database/query-optimization.typ` for a dedicated treatment.

== Volcano / Iterator Model

The *volcano model* (Graefe 1994), also called the *iterator model* or *pull model*, is the dominant execution model in row-store engines (PostgreSQL, MySQL, SQLite). Each operator implements a `next()` method that returns one tuple at a time.

```
Plan tree (pull model):
      Sort
       │  next() ↑
      Filter
       │  next() ↑
      Hash Join
     /          \
  Scan(A)     Scan(B)
```

```c
// Simplified volcano operator interface.
typedef struct Operator Operator;
typedef struct Tuple    Tuple;

struct Operator {
    void   (*open)(Operator *op);          // initialise state
    Tuple *(*next)(Operator *op);          // return next tuple or NULL
    void   (*close)(Operator *op);         // release resources
    Operator *left;
    Operator *right;
};

// Filter operator: pull from child, apply predicate, discard non-matching.
Tuple *filter_next(Operator *op) {
    FilterState *s = (FilterState *)op;
    Tuple *t;
    while ((t = op->left->next(op->left)) != NULL) {
        if (eval_predicate(s->pred, t))
            return t;
    }
    return NULL; // exhausted
}
```

*Advantages of volcano*: simple implementation, easy to compose operators, works well with pipelined execution (no materialisation between operators).

*Disadvantages*: one virtual function call per tuple — at millions of tuples per second, the function call overhead and branch misprediction become dominant costs. Each tuple also carries type metadata, inflating per-tuple processing cost.

== Vectorised Model

The *vectorised model* amortises per-tuple overhead by processing a *batch* (vector) of 1 024–8 192 tuples per `next()` call. Each column within the batch is a typed array, enabling SIMD acceleration and better cache utilisation.

```python
class VectorisedScan:
    def __init__(self, table, batch_size=1024):
        self.pos = 0
        self.table = table
        self.batch_size = batch_size

    def next_batch(self):
        """Return dict of column arrays for the next batch, or None."""
        if self.pos >= len(self.table):
            return None
        end = min(self.pos + self.batch_size, len(self.table))
        batch = {col: self.table[col][self.pos:end]
                 for col in self.table}
        self.pos = end
        return batch

class VectorisedFilter:
    def __init__(self, child, predicate_col, threshold):
        self.child = child
        self.col   = predicate_col
        self.threshold = threshold

    def next_batch(self):
        batch = self.child.next_batch()
        if batch is None:
            return None
        mask = batch[self.col] > self.threshold  # numpy-style boolean mask
        return {col: arr[mask] for col, arr in batch.items()}
```

The vectorised model is used by DuckDB, MonetDB/X100, Snowflake, and ClickHouse. It achieves 10–100× better throughput than volcano for analytical workloads by:

- Reducing function call overhead from $O(N)$ to $O(N / "batch_size")$.
- Enabling auto-vectorisation and explicit SIMD (see `database/columnar-storage-and-vectorization.typ`).
- Improving branch prediction in tight inner loops over uniform-typed arrays.

== Query Compilation (LLVM JIT)

*Query compilation* takes the physical plan and generates native machine code specifically for that query, eliminating all interpreter overhead. The compiled code contains no virtual dispatch, no type checks, and can use CPU registers efficiently across operator boundaries.

=== HyPer / Umbra / DuckDB JIT

*HyPer* (TUM, now Tableau Hyper) pioneered the "data-centric" code generation model: instead of operators calling each other, the compiler generates one tight loop per pipeline-breaker boundary.

```
Plan:  Scan → Filter → HashJoin (build side materialised) → Aggregate

Pipelines:
  Pipeline 1: Scan → Filter → HashJoin_build   (materialise hash table)
  Pipeline 2: Scan → HashJoin_probe → Aggregate → output

Compiled code (pseudocode) for Pipeline 2:
  for each tuple t in outer_scan():
      if t.amount > 1000:               // filter inlined
          bucket = hash(t.key) % HT_SIZE
          for entry in ht[bucket]:
              if entry.key == t.key:    // probe inlined
                  agg_sum += t.amount   // aggregate inlined
```

*LLVM IR generation* is used by HyPer, Umbra (its successor), and DuckDB (for certain operators). The LLVM backend applies register allocation, instruction scheduling, and vectorisation passes before emitting native code.

```cpp
// Sketch: emit LLVM IR for a simple summation loop.
// (Production code uses IRBuilder<> from llvm/IR/IRBuilder.h)
//
// for (int64_t i = 0; i < n; ++i) sum += col[i];
//
// → LLVM IR:
//   entry:
//     br label %loop
//   loop:
//     %i   = phi i64 [ 0, %entry ], [ %i_next, %loop ]
//     %sum = phi i64 [ 0, %entry ], [ %sum_next, %loop ]
//     %ptr = getelementptr i64, i64* %col, i64 %i
//     %v   = load i64, i64* %ptr
//     %sum_next = add i64 %sum, %v
//     %i_next   = add i64 %i, 1
//     %cond = icmp slt i64 %i_next, %n
//     br i1 %cond, label %loop, label %exit
//   exit:
//     ret i64 %sum_next
```

*Compilation latency* (10–500 ms for LLVM) is amortised over long-running analytical queries, but is unacceptable for short OLTP queries. Engines like DuckDB use adaptive strategies: interpret for small inputs, JIT-compile for large ones.

== Operator Implementations

=== Hash Join

*Hash join* is the dominant join algorithm for large inputs in analytical engines.

```
Phase 1 — Build:
  for each tuple t in build_input (smaller relation):
      ht[hash(t.join_key)].append(t)

Phase 2 — Probe:
  for each tuple t in probe_input (larger relation):
      for match in ht[hash(t.join_key)]:
          if match.join_key == t.join_key:
              emit (match, t)
```

Complexity: $O(N + M)$ expected where $N$ = build size, $M$ = probe size. Memory requirement: the build-side hash table must fit in memory (or be partitioned to disk via *grace hash join*).

*Grace hash join* (for relations exceeding memory):

```
1. Partition both relations R and S into k buckets by hash(join_key).
2. For each partition i: load R_i into memory hash table, probe with S_i.
   (Each partition independently fits in memory.)
```

=== Sort-Merge Join

*Sort-merge join* sorts both inputs on the join key, then merges in a single linear pass. Preferred when inputs are already sorted (e.g., via an index scan) or when the result needs to be sorted for a downstream `ORDER BY`.

```
Sort R on R.key → R_sorted
Sort S on S.key → S_sorted
Merge:
  i = j = 0
  while i < |R_sorted| and j < |S_sorted|:
      if R_sorted[i].key == S_sorted[j].key: emit and advance both
      elif R_sorted[i].key < S_sorted[j].key: i++
      else: j++
```

Complexity: $O(N log N + M log M)$ for the sort phase, $O(N + M)$ for the merge.

=== Hash Aggregation

*Hash aggregation* builds an in-memory hash table keyed on the GROUP BY columns, accumulating aggregate state per group.

```python
def hash_aggregate(tuples, group_cols, agg_col):
    ht = {}
    for t in tuples:
        key = tuple(t[c] for c in group_cols)
        if key not in ht:
            ht[key] = {"sum": 0, "count": 0}
        ht[key]["sum"]   += t[agg_col]
        ht[key]["count"] += 1
    return [(key, s["sum"], s["count"]) for key, s in ht.items()]
```

For very large numbers of groups that exceed memory, *external hash aggregation* partitions into $k$ buckets to disk, then processes each bucket independently.

== Parallel Query Execution

Modern analytical engines exploit intra-query parallelism across CPU cores and across nodes.

=== Exchange Operators

An *exchange operator* (also called shuffle or repartition) is inserted into the plan to partition data across parallel workers. Three common variants:

```
Gather:       N workers → 1 consumer (merge sorted streams or collect)
Repartition:  N workers → N workers  (hash-partition by join key before join)
Broadcast:    1 producer → N workers (replicate small table for all workers)
```

```
Parallel hash join plan (4 workers):

  ┌─────────────────────────────────────────────────────────┐
  │ Gather (1 thread merges results)                        │
  ├─────────────────────────────────────────────────────────┤
  │ Hash Join × 4 workers                                   │
  │   Repartition (probe side, hash on join key) × 4        │
  │     Scan(orders)                                        │
  │   Repartition (build side, hash on join key) × 4        │
  │     Scan(customers)                                     │
  └─────────────────────────────────────────────────────────┘
```

=== Partition-Wise Joins

When both tables are already partitioned on the join key (e.g., co-located shards in a distributed system), a *partition-wise join* can run each partition pair independently without a repartition exchange, eliminating network shuffle.

```sql
-- PostgreSQL: partition-wise join on partitioned tables
SET enable_partitionwise_join = on;
EXPLAIN SELECT * FROM orders o JOIN customers c ON o.cust_id = c.id;
-- Plan shows Hash Join executed independently per partition pair.
```

== Adaptive Query Execution

Static cost models are imperfect: cardinality estimates can be wrong by orders of magnitude, especially for multi-predicate queries or correlated data. *Adaptive Query Execution* (AQE) re-optimises the plan at runtime as actual statistics become available.

=== Spark AQE

*Apache Spark 3.0+* introduced AQE with three main adaptations:

- *Coalescing shuffle partitions*: after a shuffle, if actual partition sizes are much smaller than expected, Spark merges small partitions to reduce overhead.
- *Switching join strategies*: if the build side of a planned sort-merge join turns out to be small enough after filtering, Spark switches to a broadcast hash join at runtime.
- *Skew join optimisation*: if some shuffle partitions are much larger than others (data skew), Spark splits skewed partitions and replicates the matching build-side partition.

```
AQE re-optimisation trigger points (Spark):
  Query start → static plan based on catalog statistics
        │
  Shuffle stage completes → actual partition sizes known
        │
  AQE re-evaluates:
    - Join strategy (broadcast if build < autoBroadcastJoinThreshold)
    - Partition count (coalesce if actual_size << target_size)
    - Skew detection (split if partition > 5× median AND > 256 MB)
        │
  Remaining stages execute with updated plan
```

=== Re-optimisation at Runtime

Beyond Spark, several engines implement runtime re-optimisation:

- *PostgreSQL generic/custom plans*: for prepared statements, PostgreSQL initially plans with parameter placeholders; after 5 executions, it compares average custom-plan cost vs generic-plan cost and switches permanently.
- *SQL Server AQPO*: Adaptive Query Processing with memory grant feedback (adjusts sort/hash operator memory on re-runs) and batch mode on row store.
- *CockroachDB*: re-plans queries with updated statistics when plan degradation is detected via execution metrics.

== Cost Model Basics

The optimizer selects among candidate physical plans by estimating cost. A cost model estimates:

$ "cost"(P) = "I/O cost" + "CPU cost" = "pages read" dot c_"IO" + "tuples processed" dot c_"CPU" $

=== Selectivity Estimation

*Selectivity* $sigma$ of a predicate is the fraction of rows satisfying it:

$ sigma("col = v") = 1 / "NDV"("col") $

where $"NDV"$ is the number of distinct values. For range predicates:

$ sigma("col" < v) = (v - "min"("col")) / ("max"("col") - "min"("col")) $

=== Histograms

*Equi-depth histograms* divide the value range into $B$ buckets each containing approximately $N/B$ values, storing min, max, and NDV per bucket. A predicate is evaluated against the overlapping buckets:

```
Column salary histogram (B=4):
  Bucket 0: [20k, 45k), count=250, NDV=200
  Bucket 1: [45k, 70k), count=250, NDV=180
  Bucket 2: [70k, 95k), count=250, NDV=120
  Bucket 3: [95k, 200k], count=250, NDV=90

Predicate: salary BETWEEN 60k AND 80k
  Bucket 1 overlap: (70k-60k)/(70k-45k) * 250 ≈ 100 rows
  Bucket 2 overlap: (80k-70k)/(95k-70k) * 250 ≈  100 rows
  Estimate: ~200 rows
```

=== Cardinality Estimation

*Join cardinality* estimation assumes independence between join columns:

$ |R join S| approx |R| dot |S| dot sigma_"join" $

This independence assumption breaks down for correlated predicates, leading to severe underestimates. Modern systems augment histograms with:

- *Multi-column statistics* (PostgreSQL `CREATE STATISTICS ... (dependencies)`)
- *Sampling-based estimation* (random sample of the table, execute predicates on sample)
- *Learned cardinality estimators* (neural network models trained on query workloads — research stage)

== Common Optimisations

=== Predicate Pushdown

Move filter predicates as close to their data source as possible, reducing rows processed by upstream operators.

```
Before pushdown:
  Filter (amount > 1000)
    Hash Join (orders ⋈ customers ON cust_id)
      Scan(orders)
      Scan(customers)

After pushdown:
  Hash Join (orders ⋈ customers ON cust_id)
    Filter (amount > 1000)           ← pushed below join
      Scan(orders)
    Scan(customers)
```

For columnar formats, predicate pushdown continues into the file reader (skipping Parquet row groups / ORC stripes).

=== Column Pruning

Remove columns from scans that are not referenced by the query. In row stores this has limited benefit (full row is read anyway), but in columnar stores it eliminates entire column files from I/O.

=== Join Reordering

The search space of join orderings for $n$ tables is $O(n!)$. Optimisers use dynamic programming (System R style) for $n <= 10$, and heuristics (greedy bushy tree, genetic algorithm) for larger queries.

```
Dynamic programming (System R):
  For each subset S of tables, compute cheapest plan to join S:
  optimal[{A}] = Scan(A)
  optimal[{A,B}] = min(Hash(A,B), NLJ(A,B), SMJ(A,B))
  optimal[{A,B,C}] = min over all splits:
      plan(join(optimal[S1], optimal[S2])) for S1 ∪ S2 = {A,B,C}
```

== Further Reading

Graefe, G. (1994). "Volcano — An Extensible and Parallel Query Evaluation System." IEEE TKDE.

Neumann, T. (2011). "Efficiently Compiling Efficient Query Plans for Modern Hardware." VLDB.

Boncz, P., Zukowski, M., Nes, N. (2005). "MonetDB/X100: Hyper-Pipelining Query Execution." CIDR.

Selinger, P.G. et al. (1979). "Access Path Selection in a Relational Database Management System." SIGMOD.

Zhu, S. et al. (2020). "Adaptive Query Processing in the Looking Glass." CIDR (Spark AQE).

Leis, V. et al. (2015). "How Good Are Query Optimizers, Really?" VLDB (cardinality estimation study).

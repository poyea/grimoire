#import "../template.typ": xref

= SQL Engine Internals

Optimizer textbooks describe System R dynamic programming as if every modern system implemented it directly. Real engines diverge sharply: PostgreSQL falls back to a genetic algorithm above a threshold; MySQL only embraced a cost-based optimizer in 5.6 and rewrote join planning in 8.0; DuckDB pushes vectors through a pipeline; ClickHouse pulls them in blocks. This chapter compares four open-source engines whose internals are well-documented.

*See also:* #xref("database", "query-optimization", label: "Query Optimization"), #xref("database", "query-compilation", label: "Query Compilation"), #xref("database", "column-stores-and-vectorized-execution", label: "Column Stores and Vectorized Execution"), #xref("database", "joins-and-aggregation", label: "Joins and Aggregation")

== PostgreSQL Planner

PostgreSQL's planner lives in `src/backend/optimizer/`. The pipeline is `parse` → `analyze` → `rewrite` → `plan` → `execute`. The planner operates on `Query` nodes and produces `Plan` trees.

=== Path Generation and DP

For each join level $k$ up to the threshold `geqo_threshold` (default 12), PostgreSQL enumerates left-deep, right-deep, and bushy join trees via dynamic programming (`make_rel_from_joinlist`). Each candidate is a *Path*; multiple paths per relation differ by sort order, index choice, and parallel degree.

```
For each base relation R_i:
    Generate SeqScan, IndexScan(idx_j), BitmapHeapScan, IndexOnlyScan paths
    Keep the Pareto-optimal frontier on (total_cost, startup_cost, pathkeys, parallel_aware)

For level = 2 .. N:
    For each pair (S, T) of disjoint subsets with |S|+|T| = level:
        For each pair of paths p_S, p_T:
            Generate NestLoop, HashJoin, MergeJoin paths
            Keep Pareto-optimal frontier
```

Paths are kept only when dominant on at least one dimension (this is *path pruning*). Sort orders matter because merge joins and `ORDER BY` can reuse them.

=== GEQO — Genetic Optimizer

When the join graph exceeds `geqo_threshold` relations, exhaustive DP becomes too expensive ($O(3^n)$). PostgreSQL switches to GEQO (Stillger, 1997), a genetic algorithm.

```
Population: ~ pool_size random join orders (permutations)
Each chromosome encodes a left-deep tree as a permutation of relation IDs.
Fitness = cost of the corresponding left-deep plan.

Repeat for `generations` rounds:
    Select parents by tournament
    Crossover: edge recombination (preserves adjacency from parents)
    Mutate: swap two genes with low probability
    Replace least-fit individuals
Return best-fitness plan
```

The seed defaults to 0, so plans are deterministic across runs for the same statistics, which is important for regression testing.

=== Cost Model

```
total_cost = startup_cost + run_cost
run_cost   = cpu_tuple_cost * N
           + cpu_operator_cost * N * predicate_complexity
           + (seq_page_cost | random_page_cost) * pages_fetched
```

Defaults: `seq_page_cost = 1.0`, `random_page_cost = 4.0`, `cpu_tuple_cost = 0.01`, `cpu_operator_cost = 0.0025`. The 4:1 random:sequential ratio reflects rotational disks and is routinely tuned to 1.1 on NVMe.

=== Statistics: `pg_statistic`

`ANALYZE` samples rows (default 300 × `default_statistics_target = 100` = 30 000) and stores per-column:

- `stanullfrac`: fraction of NULLs.
- `stawidth`: average width.
- `stadistinct`: distinct count ($n_"distinct"$); negative means fraction of rows.
- `most_common_vals` (MCV) + `most_common_freqs`: list of frequent values.
- `histogram_bounds`: equi-depth histogram for the non-MCV portion.
- `correlation`: physical-vs-logical correlation, used to discount index random reads.

*Extended statistics* (`CREATE STATISTICS`) capture multi-column dependencies, $n_"distinct"$ for column groups, and MCV lists for tuples; these are critical for correlated predicates that the independence assumption ruins.

=== Join Selectivity

```
Default selectivity (eq-join on a, b without MCV overlap):
  sel = 1 / max(n_distinct(a), n_distinct(b))

With MCV lists, the planner cross-joins MCV(a) × MCV(b), sums matching frequencies,
and applies the default selectivity to the residual mass.
```

=== Parallel Query

The planner emits `Gather` / `Gather Merge` nodes when a parallel-aware path is cheaper. `max_parallel_workers_per_gather` and `parallel_setup_cost` (1000) limit worker spawning. Parallel-safe functions are marked `PARALLEL SAFE` in `pg_proc`.

== MySQL Optimizer

MySQL's optimizer lives in `sql/sql_optimizer.cc` and surrounding files. Until 8.0 it used a greedy `simple_table_dependencies`-driven search; 8.0 added a *hypergraph optimizer* (`-hypergraph` switch, GA in 8.4) based on Moerkotte & Neumann's DPhyp algorithm.

=== Cost-Based Optimization Since 5.6

Pre-5.6, MySQL chose access paths heuristically: leftmost index wins, no histograms, no condition fanout. 5.6 introduced an Engine Independent Cost Model with `mysql.server_cost` and `mysql.engine_cost` system tables tracking per-operation costs (e.g. `disk_temptable_create_cost = 20`, `key_compare_cost = 0.05`).

8.0 added *histograms* (equi-height, JSON-stored in `information_schema.column_statistics`), invariant index dives, and condition filtering (`condition_fanout_filter`).

=== Hypergraph Optimizer (DPhyp)

DPhyp enumerates connected subgraphs of the join hypergraph, handling complex predicates (outer joins, semi-joins, theta joins) that classic Selinger DP cannot. The algorithm avoids cross products and reaches the optimal join tree for graphs of ~30 relations.

```
DPhyp(G):
  for each vertex v in reverse order:
    EmitCsg({v})
    EnumerateCsgRec({v}, BFS-neighbors(v) restricted to "smaller" vertices)
  // For each connected subgraph S found:
  //   EmitCmp(S) finds complement subgraphs T, costs join(S, T)
```

The hypergraph carries hyperedges for conflict-free join reordering (CD-A algorithm, Moerkotte 2013), which encodes semantic restrictions like "outer-join preserves left side."

=== Hash Join

MySQL 8.0.18 introduced hash join for inner and (8.0.20+) outer/semi joins, replacing the block-nested-loop fallback. The implementation is a *Grace* hash join: build side fits in `join_buffer_size`, else spills to chunked files indexed by hash prefix.

```cpp
// hash_join_iterator.cc, simplified
class HashJoinIterator {
  // Phase 1: build
  for (Row r : build_input) {
    if (mem_used + r.size > join_buffer_size) SpillToDisk();
    else hash_table.Insert(BuildKey(r), r);
  }
  // Phase 2: probe (with optional disk-resident chunks)
  for (Row r : probe_input) {
    for (Row b : hash_table.Lookup(ProbeKey(r))) {
      Emit(Combine(b, r));
    }
  }
};
```

== DuckDB

DuckDB is an in-process OLAP engine using *vectorized push-based* execution (Raasveldt & Mühleisen, CIDR 2020). Tuples flow as columnar *DataChunks* of $approx 2048$ rows.

=== Pipeline Compilation

The planner produces an operator tree; the *pipeline scheduler* breaks it into pipelines bounded by *sink* operators (hash-join build, aggregate finalize, sort). Each pipeline pushes chunks from a *source* through *operators* into a *sink*. Different pipelines run in parallel; within a pipeline, multiple *threads* execute on disjoint morsels.

```
Pipeline 1 (build):  Scan(R) ──► HashJoinBuild
Pipeline 2 (probe):  Scan(S) ──► HashJoinProbe ──► Aggregate (sink)
Pipeline 3 (final):  Aggregate ──► ResultCollector
```

The push model amortizes per-tuple overhead and matches morsel-driven parallelism (Leis 2014). Operators inherit from `PhysicalOperator` and implement `Execute(DataChunk &input, DataChunk &output)` for streaming, or `Sink/Combine/Finalize` for blocking operators.

=== Vectorized Expressions

Expression evaluation in `expression_executor.cpp` dispatches per chunk:

```cpp
// Simplified: add two int64 columns with NULL handling
void add_i64(Vector &a, Vector &b, Vector &out, idx_t count) {
  auto la = FlatVector::GetData<int64_t>(a);
  auto lb = FlatVector::GetData<int64_t>(b);
  auto lo = FlatVector::GetData<int64_t>(out);
  for (idx_t i = 0; i < count; i++) lo[i] = la[i] + lb[i];
  FlatVector::SetValidityMask(out, ValidityMask::And(a.validity, b.validity));
}
```

Vectors carry a *selection vector* for filtered chunks, avoiding compaction copies between operators.

=== Cost Estimation

DuckDB uses sampling and HyperLogLog sketches for cardinality estimation. The optimizer is rule-based for relational rewrites then performs DP join enumeration up to 64 relations using the Iterative Dynamic Programming heuristic from Kossmann & Stocker (2000).

== ClickHouse Pipeline Executor

ClickHouse is built around the *MergeTree* family of column-oriented tables and a *vectorized pull-based* executor (`Processors`, `PipelineExecutor`).

=== Processors

Each `IProcessor` is a node in a directed graph with *input ports* and *output ports*. The executor advances ports via state transitions: `NeedData`, `PortFull`, `Ready`, `Async`, `Finished`. A pull pulls a column block from upstream processors when the downstream has capacity.

```
                 ┌──────────────┐
   Source MergeTree ── Filter ── Aggregator(partial) ── Resize ──► Aggregator(final) ── Sink
   Source MergeTree ── Filter ── Aggregator(partial) ─┘
   Source MergeTree ── Filter ── Aggregator(partial) ─┘
```

A `Resize` processor fans out or fans in streams; `Aggregator` splits into partial-state per-thread then merges.

=== Blocks of Columns

Data flows as `Block`s: vectors of `IColumn` (PODs, strings, arrays, low-cardinality dictionaries) typically 65 536 rows each. SIMD kernels in `src/Common/`, `src/AggregateFunctions/`, and `src/Functions/` operate on raw column data. AggregateFunctions store partial state in arenas with `addBatch`, `mergeBatch`, `serializeBatch`.

=== Query Plan and Optimizations

`InterpreterSelectQuery` builds a logical `QueryPlan` of `IQueryPlanStep`s; `QueryPlanOptimizationSettings` controls pushdown, predicate reordering, projection rewrite, sorting elimination, and aggregator-to-direct-execution rewrite. The plan is then *built into a Pipeline* of processors.

ClickHouse omits a full cost-based join optimizer (until experimental `query_plan_join_swap_table` and new `allow_experimental_analyzer`). Joins default to *hash join with right table as build side*; users are expected to hint reorderings.

=== Pull vs Push

#table(
  columns: (auto, auto, auto),
  [*Aspect*], [*Push (DuckDB)*], [*Pull (ClickHouse)*],
  [Control flow], [Source drives], [Sink drives via state machine],
  [Backpressure], [Pipeline scheduler blocks source], [Output port saturation pauses processor],
  [Parallelism], [Morsel-driven within pipeline], [Independent processor threads],
  [Async I/O], [Coroutine-like callbacks], [Async processor state],
  [Complex DAGs], [Pipeline graph], [Native — Processor DAG generalises easily],
)

== Engine Comparison

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Engine*], [*Optimizer*], [*Execution*], [*Stats*], [*Parallelism*],
  [PostgreSQL], [DP + GEQO above 12 joins], [Iterator (Volcano), JIT via LLVM], [Histograms, MCV, extended stats], [Gather workers],
  [MySQL 8.4], [Hypergraph DPhyp], [Iterator, hash join], [Equi-height histograms], [Limited (parallel scan only)],
  [DuckDB], [Rule-based + DP join enum], [Vectorized push, morsel-driven], [HLL, samples], [Pipelines × morsels],
  [ClickHouse], [Rule-based, experimental analyzer], [Vectorized pull, processors], [Per-part sparse], [Per-stream processors],
)

== Common Themes

All four engines share core ideas: predicate pushdown, projection pruning, sort/hash join selection, partition-aware scans. They diverge on:

- *Estimation*: PostgreSQL invests in statistics; ClickHouse leaves this to the user.
- *Parallelism granularity*: DuckDB's morsel model offers finer load balancing than PostgreSQL's per-worker partitioning.
- *Execution model*: pull vs push is dual; the right choice depends on operator complexity and async-I/O patterns.

== Further Reading

Stillger, M., Spiliopoulou, M. (1996). "Genetic Programming in Database Query Optimization." GP Conference.

Stillger, M., Lohman, G. et al. (2001). "LEO — DB2's LEarning Optimizer." VLDB.

Moerkotte, G., Neumann, T. (2008). "Dynamic Programming Strikes Back." SIGMOD (DPhyp).

Moerkotte, G., Neumann, T. (2013). "Building Query Compilers." (draft monograph).

PostgreSQL Global Development Group. "Chapter 14: Performance Tips" and `src/backend/optimizer/README`.

Raasveldt, M., Mühleisen, H. (2020). "DuckDB: An Embeddable Analytical Database." SIGMOD demo / CIDR 2020 paper "Data Management for Data Science."

Leis, V. et al. (2014). "Morsel-Driven Parallelism." SIGMOD.

Neumann, T. (2011). "Efficiently Compiling Efficient Query Plans for Modern Hardware." VLDB.

Kossmann, D., Stocker, K. (2000). "Iterative Dynamic Programming: A New Class of Query Optimization Algorithms." TODS.

ClickHouse documentation: "Architecture Overview" and "Pipeline Executor" (clickhouse.com/docs).

Oracle MySQL documentation: "The Hypergraph Optimizer" (8.4 manual, Chapter 10).

Selinger, P. et al. (1979). "Access Path Selection in a Relational Database Management System." SIGMOD (System R).

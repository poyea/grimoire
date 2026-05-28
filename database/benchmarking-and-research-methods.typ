= Benchmarking and Research Methods

Benchmarking measures database performance under controlled conditions. Bad benchmarks produce misleading results — and the database field has a long history of systems being published with cherry-picked workloads. Understanding benchmark design, statistical analysis, and how to read research papers critically is as important as knowing the systems themselves.

*See also:* _database/observability-and-self-driving.typ_, _database/query-optimization.typ_

== Standard Benchmarks

=== OLTP Benchmarks

*TPC-C* (Transaction Processing Performance Council): the gold standard for OLTP. Models a wholesale supplier with 5 transaction types (NewOrder, Payment, OrderStatus, Delivery, StockLevel) across multiple warehouses.

```
TPC-C schema (simplified):
  Warehouse → District → Customer → Order → OrderLine
                                  → NewOrder (open orders)
            → Stock (item inventory per warehouse)

Workload: 45% NewOrder + 43% Payment + other transactions
Scale: "1 warehouse" ≈ 9 tables, ~100 MB
       "1000 warehouses" ≈ 100 GB

Metric: tpmC (transactions per minute, counting only NewOrder transactions)
```

*TPC-B* (banking benchmark, obsolete but simple):

```sql
-- TPC-B core transaction: debit/credit
BEGIN;
  UPDATE accounts SET abalance = abalance + delta WHERE aid = :aid;
  UPDATE tellers  SET tbalance = tbalance + delta WHERE tid = :tid;
  UPDATE branches SET bbalance = bbalance + delta WHERE bid = :bid;
  INSERT INTO history VALUES (:tid, :bid, :aid, :delta, now(), :filler);
COMMIT;
```

*pgbench* (PostgreSQL's built-in TPC-B-like benchmark):

```bash
# Initialize (scale=100: 100 × 100K accounts = 10M rows)
pgbench -i -s 100 mydb

# Run: 10 clients, 30 seconds
pgbench -c 10 -T 30 -P 5 mydb
# Output: tps = 12345.6 (± stddev)

# Custom transaction script
cat > custom.sql << 'EOF'
\set aid random(1, 10000000)
BEGIN;
SELECT abalance FROM accounts WHERE aid = :aid;
UPDATE accounts SET abalance = abalance + 100 WHERE aid = :aid;
COMMIT;
EOF
pgbench -c 10 -T 30 -f custom.sql mydb
```

=== OLAP Benchmarks

*TPC-H:* 8 tables, 22 analytical queries. Scale factor SF1 = 1 GB, SF1000 = 1 TB. Widely used but criticized for queries that don't represent real analytics (optimized away by simple heuristics).

```sql
-- TPC-H Query 1: aggregate revenue by return flag and line status
SELECT
    l_returnflag,
    l_linestatus,
    sum(l_quantity)                                          AS sum_qty,
    sum(l_extendedprice)                                     AS sum_base_price,
    sum(l_extendedprice * (1 - l_discount))                  AS sum_disc_price,
    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax))    AS sum_charge,
    avg(l_quantity)                                          AS avg_qty,
    avg(l_extendedprice)                                     AS avg_price,
    avg(l_discount)                                          AS avg_disc,
    count(*)                                                 AS count_order
FROM lineitem
WHERE l_shipdate <= date '1998-12-01' - interval '90 day'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;
```

*TPC-DS:* 99 queries, more complex schema, better reflects modern analytics (star schema + snowflake, skewed data distributions). The current standard for OLAP benchmarking.

*JOB (Join Order Benchmark):* 33 queries on the IMDB dataset (113 tables). Used to stress-test join ordering and cardinality estimation. All queries require 5+ way joins with real-world skew.

```python
# Reproducing JOB benchmark
import duckdb

con = duckdb.connect()
# Load IMDB dataset (CSV files)
con.execute("CREATE TABLE title AS SELECT * FROM 'title.csv'")
con.execute("CREATE TABLE cast_info AS SELECT * FROM 'cast_info.csv'")
# ... (113 tables total)

# JOB Query 1a: 5-way join (simplified)
result = con.execute("""
    SELECT MIN(mc.note) AS production_note,
           MIN(t.title) AS movie_title,
           MIN(t.production_year) AS movie_year
    FROM company_type ct, info_type it, movie_companies mc,
         movie_info_idx mi_idx, title t
    WHERE ct.kind = 'production companies'
      AND it.info = 'top 250 rank'
      AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
      AND t.production_year > 2010
      AND ct.id = mc.company_type_id
      AND t.id = mc.movie_id
      AND t.id = mi_idx.movie_id
      AND mc.movie_id = mi_idx.movie_id
      AND it.id = mi_idx.info_type_id
""").fetchall()
```

== Statistical Rigor in Benchmarking

Performance experiments are noisy. A single measurement proves nothing.

=== Confidence Intervals and Hypothesis Testing

```python
import numpy as np
from scipy import stats

def benchmark_compare(system_a_times: list[float],
                       system_b_times: list[float],
                       confidence: float = 0.95) -> dict:
    """
    Compare two systems. Returns whether the difference is statistically significant.
    """
    n_a, n_b    = len(system_a_times), len(system_b_times)
    mean_a      = np.mean(system_a_times)
    mean_b      = np.mean(system_b_times)

    # Welch's t-test (does not assume equal variance)
    t_stat, p_value = stats.ttest_ind(system_a_times, system_b_times,
                                       equal_var=False)

    # Confidence interval for the difference in means
    se_diff = np.sqrt(np.var(system_a_times)/n_a + np.var(system_b_times)/n_b)
    df      = (se_diff**2) / ((np.var(system_a_times)/n_a)**2/(n_a-1) +
                               (np.var(system_b_times)/n_b)**2/(n_b-1))
    t_crit  = stats.t.ppf((1 + confidence) / 2, df)
    ci      = (mean_a - mean_b - t_crit*se_diff,
               mean_a - mean_b + t_crit*se_diff)

    return {
        "mean_a":       mean_a,
        "mean_b":       mean_b,
        "speedup":      mean_b / mean_a,
        "p_value":      p_value,
        "significant":  p_value < (1 - confidence),
        "ci_95":        ci,
    }

# Example: A is 10ms (±1ms), B is 15ms (±5ms) — is B really slower?
a = np.random.normal(10, 1, 30)   # 30 runs of system A
b = np.random.normal(15, 5, 30)   # 30 runs of system B
print(benchmark_compare(a, b))
# p < 0.05 → statistically significant; CI doesn't cross 0 → A is faster
```

=== What to Report

```python
# Always report:
stats_report = {
    "median":     np.median(times),
    "p99":        np.percentile(times, 99),
    "mean":       np.mean(times),
    "std":        np.std(times),
    "n_runs":     len(times),
    "ci_95":      stats.t.interval(0.95, len(times)-1,
                                    loc=np.mean(times),
                                    scale=stats.sem(times)),
    "throughput": 1.0 / np.mean(times),   # if measuring latency
}
# Never report just "mean" without std/CI — it hides variance
# Never report a single run
```

*Latency vs throughput:* these are not independent. Under high load, latency degrades before throughput. Always report both and the load level (clients/concurrency).

== Microbenchmark Design

*Warm up before measuring:* JIT compilation, buffer pool cold start, OS page cache effects.

```bash
# Linux: isolate CPU for benchmarking
# Disable Turbo Boost (reduces variance)
echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo

# Pin process to a specific CPU core
taskset -c 2 ./my_benchmark

# Drop OS page cache (to simulate cold start)
sync && echo 3 > /proc/sys/vm/drop_caches

# Disable CPU frequency scaling
cpupower frequency-set -g performance
```

```python
import timeit, gc

def bench(fn, n_warmup=100, n_measure=1000):
    """Benchmark a function with warmup and multiple measurements."""
    # Warmup: allow JIT, caches to stabilize
    for _ in range(n_warmup):
        fn()
    gc.collect()

    times = []
    for _ in range(n_measure):
        t = timeit.default_timer()
        fn()
        times.append(timeit.default_timer() - t)
    return times
```

== Common Benchmarking Mistakes

#table(
  columns: (auto, auto),
  [*Mistake*], [*Why it's wrong*],
  [Single measurement],              [No variance information; noise dominates],
  [No warmup],                        [JIT, cache effects inflate first runs],
  [Reporting only mean],              [Hides long-tail latency (p99, p999)],
  [Not matching production config],   [Buffer pool, parallelism, fsync settings matter enormously],
  [Not reporting system config],      [Results not reproducible; different hardware = different results],
  [Comparing different workloads],    ["We're faster on Q1" but Q1 isn't in your workload],
  [Microbenchmark vs system bench],   [Tight loop microbenchmarks exclude I/O, locking, parse overhead],
)

== Reading Database Research Papers

*Key questions to ask:*

1. *What problem is being solved?* Is this problem real (measured in production) or hypothetical?
2. *What is the evaluation?* Micro or macro benchmark? What workload? Is it TPC-C/TPC-DS or a proprietary workload?
3. *What are the baselines?* Are competitors configured optimally? Are they the strongest available?
4. *What are the assumptions?* In-memory? No disk I/O? Single node? These often don't hold in practice.
5. *Is the improvement statistically significant?* Are error bars shown? Is n ≥ 30?

*Classic evaluation flaws:*

```
"Our system is 10× faster"
→ Did they configure the baseline with the same memory budget? Same hardware?
   Did they disable buffer pool for the competitor but enable it for themselves?

"We achieve 1M TPS"
→ At what latency? On what hardware? What's the workload?
   Under Snapshot Isolation or Serializable?

"Our new index reduces query time by 90%"
→ On which queries? What fraction of the workload? What's the index build cost?
```

== Flamegraphs for Database Profiling

_See also:_ _CPU Architecture volume_ for performance-counter background (IPC, branch mispredictions, cache-miss attribution) that gives flamegraph nodes their cost interpretation.


```bash
# Profile PostgreSQL backend for 30 seconds
sudo perf record -g -p $(pg_ctl -D /var/lib/pgsql/data status | grep PID | awk '{print $3}') -- sleep 30
sudo perf script | stackcollapse-perf.pl | flamegraph.pl > postgres_flamegraph.svg

# What to look for:
# - Large bar in ExecutorRun → query execution (expected)
# - Large bar in LockAcquire → lock contention problem
# - Large bar in hash_search → buffer pool lookup overhead
# - Large bar in CheckpointerMain → checkpoint I/O pressure
```

== References

Transaction Processing Performance Council. "TPC-C Standard." http://www.tpc.org/tpcc/

Leis, V. et al. (2015). "How Good Are Query Optimizers, Really?" VLDB. (JOB benchmark)

Raasveldt, M. et al. (2018). "Don't Hold My Data Hostage: A Case for Client Protocol Redesign." VLDB.

Brendan Gregg. "Systems Performance." 2nd ed. Addison-Wesley, 2020. (profiling methodology)

Mytkowicz, T. et al. (2009). "Producing Wrong Data Without Doing Anything Obviously Wrong!" ASPLOS. (measurement bias)

Curtsinger, C., Berger, E. (2015). "Stabilizer: Statistically Sound Performance Evaluation." ASPLOS.

= Observability and Self-Driving Databases

A database that cannot be observed cannot be tuned. Observability (metrics, logs, traces) is the prerequisite for self-driving capabilities — automated index selection, knob tuning, query plan selection, and anomaly detection.

*See also:* _database/query-optimization.typ_, _database/security-and-privacy.typ_, _database/benchmarking-and-research-methods.typ_

== Three Pillars of Database Observability

*Metrics:* numeric time-series (QPS, latency percentiles, cache hit ratio, lock wait time).
*Logs:* discrete events (slow queries, errors, autovacuum runs, deadlocks).
*Traces:* end-to-end request flows with timing at each stage (query parse, plan, execute, I/O).

== Key Metrics to Monitor

=== PostgreSQL

```sql
-- Connection and session state
SELECT state, count(*) FROM pg_stat_activity GROUP BY state;

-- Lock contention
SELECT pid, mode, relation::regclass, granted
FROM pg_locks
WHERE NOT granted;

-- Table-level I/O (cache hit ratio per table)
SELECT relname,
       heap_blks_hit,
       heap_blks_read,
       round(100.0 * heap_blks_hit / nullif(heap_blks_hit + heap_blks_read, 0), 1) AS hit_pct
FROM pg_statio_user_tables
ORDER BY heap_blks_read DESC
LIMIT 20;

-- Slow queries (requires pg_stat_statements)
CREATE EXTENSION pg_stat_statements;

SELECT
    left(query, 80)                                  AS query,
    calls,
    round(mean_exec_time::numeric, 2)                AS avg_ms,
    round(total_exec_time::numeric / 1000, 2)        AS total_sec,
    round(stddev_exec_time::numeric, 2)              AS stddev_ms,
    rows / calls                                      AS avg_rows
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;

-- Index usage (find unused indexes)
SELECT relname, indexrelname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan < 50
ORDER BY idx_scan;
```

=== InnoDB / MySQL

```sql
-- InnoDB buffer pool efficiency
SHOW ENGINE INNODB STATUS;
-- Look for: "Buffer pool hit rate 998 / 1000"  (99.8% — acceptable)
-- < 990: add memory or reduce working set

-- Slow query log
SHOW VARIABLES LIKE 'slow_query_log%';
SET GLOBAL slow_query_log = ON;
SET GLOBAL long_query_time = 1.0;  -- log queries > 1s

-- pt-query-digest (Percona Toolkit): aggregate slow query log
-- pt-query-digest /var/log/mysql/slow.log | head -100

-- Lock wait timeout events
SELECT * FROM performance_schema.events_waits_summary_global_by_event_name
WHERE event_name LIKE '%lock%'
ORDER BY sum_timer_wait DESC;
```

== Wait Event Analysis

PostgreSQL exposes *wait events* — what each backend is waiting for:

```sql
-- What are all active sessions waiting on?
SELECT
    pid,
    state,
    wait_event_type,
    wait_event,
    left(query, 60) AS query_snippet
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY wait_event_type NULLS LAST;

-- Aggregate: most common waits (run periodically, compare over time)
SELECT wait_event_type, wait_event, count(*)
FROM pg_stat_activity
WHERE state = 'active'
GROUP BY 1, 2
ORDER BY 3 DESC;
```

*Common waits and their causes:*

#table(
  columns: (auto, auto),
  [*Wait event*], [*Common cause*],
  [`Lock:relation`],         [Table-level lock contention (DDL + DML conflict)],
  [`Lock:tuple`],            [Row-level lock contention (hot rows)],
  [`IO:DataFileRead`],       [Cache miss — buffer pool too small],
  [`IO:WALWrite`],           [Slow WAL disk or synchronous_commit overhead],
  [`Client:ClientRead`],     [Application not consuming results fast enough],
  [`CPU`],                   [Expensive query (full scan, sort, hash join)],
)

== Query Plan Monitoring

*Detecting plan regressions:* the planner's choice of plan can change when statistics change (after large inserts), after `ANALYZE`, or after PostgreSQL version upgrades.

```sql
-- Track plan changes over time (requires pg_stat_statements)
-- Compare current plans to a baseline
SELECT
    queryid,
    query,
    calls,
    mean_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 100   -- queries averaging > 100ms
  AND calls > 10
ORDER BY mean_exec_time DESC;

-- Force a specific plan for a known-bad case (pg_hint_plan extension):
/*+ SeqScan(orders) HashJoin(orders customers) */
SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id;
```

== Automated Index Recommendation

*Problem:* database administrators can't manually enumerate all potentially useful indexes for a large schema. Workload-driven index selection automates this.

*Dexter* (PostgreSQL, open source): runs `EXPLAIN` with hypothetical indexes (using `hypopg` extension) to find which indexes would most reduce query cost.

```sql
-- Hypopg: test a hypothetical index without building it
CREATE EXTENSION hypopg;

-- Simulate: "what if there were an index on orders(customer_id, created_at)?"
SELECT * FROM hypopg_create_index('CREATE INDEX ON orders(customer_id, created_at)');

EXPLAIN SELECT * FROM orders WHERE customer_id = 42 AND created_at > '2025-01-01';
-- Planner uses hypothetical index → shows expected speedup WITHOUT building the index

SELECT hypopg_drop_index('myindex'::regclass);  -- clean up
```

*Workload-driven algorithm (simplified):*

```python
def recommend_indexes(workload: list[str], db) -> list[str]:
    """Given a list of queries, recommend beneficial indexes."""
    candidates = []
    for query in workload:
        plan = db.explain(query)
        # Find sequential scans with high cost as candidate index columns
        for node in plan.find_seq_scans():
            if node.filter_cols:
                candidates.append(node.filter_cols)

    # Baseline cost without any hypothetical index, then per-candidate diff.
    recommendations = []
    total_cost_before = sum(db.explain_cost(q) for q in workload)
    for cols in set(map(tuple, candidates)):
        hyp_idx = db.create_hypopg_index(cols)
        # EXPLAIN now considers the hypothetical index in plan selection.
        total_cost_after = sum(db.explain_cost(q) for q in workload)
        db.drop_hypopg_index(hyp_idx)
        if total_cost_before - total_cost_after > THRESHOLD:
            recommendations.append(f"CREATE INDEX ON table({', '.join(cols)})")
    return recommendations
```

== Knob Tuning (OtterTune, AutoDBA)

A database has hundreds of configuration parameters. Manual tuning is infeasible at scale. *Automated tuning* uses workload replay + ML to find optimal settings.

*OtterTune (Van Aken et al. 2017):* uses Gaussian Process regression to model the relationship between knob settings and workload performance. Iteratively samples and updates the model.

```python
# OtterTune simplified workflow
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np

class KnobTuner:
    def __init__(self, knob_bounds: dict):
        self.bounds = knob_bounds    # {"shared_buffers": (128, 16384), ...}
        self.gp     = GaussianProcessRegressor(kernel=Matern(nu=2.5))
        self.X      = []   # observed knob settings (normalized)
        self.y      = []   # observed throughput

    def suggest(self) -> dict:
        """Use Upper Confidence Bound to pick next config to try."""
        if len(self.X) < 5:
            # Random exploration initially
            return {k: np.random.randint(*v) for k, v in self.bounds.items()}
        # Fit model and find the setting maximizing UCB
        self.gp.fit(np.array(self.X), np.array(self.y))
        candidates = [self._random_config() for _ in range(200)]
        X_cand     = np.array([self._normalize(c) for c in candidates])
        mu, sigma  = self.gp.predict(X_cand, return_std=True)
        ucb        = mu + 2 * sigma
        return candidates[np.argmax(ucb)]

    def observe(self, config: dict, throughput: float):
        self.X.append(self._normalize(config))
        self.y.append(throughput)
```

*Bao (query-level tuning):* see _database/query-optimization.typ_ for Bao's learned query plan selection.

== Anomaly Detection

```python
# Simple z-score based anomaly detection for query latency
import numpy as np
from scipy import stats

def detect_latency_anomaly(latencies: list[float],
                            threshold_z: float = 3.0) -> list[int]:
    """Return indices of anomalous latency measurements."""
    arr = np.array(latencies)
    z   = np.abs(stats.zscore(arr))
    return list(np.where(z > threshold_z)[0])

# More robust: use IQR (less sensitive to extreme outliers)
def iqr_anomalies(latencies, multiplier=3.0) -> list[int]:
    arr = np.array(latencies)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return list(np.where((arr < lower) | (arr > upper))[0])
```

*Prometheus + Grafana alerting:*

```yaml
# Alert: p99 latency > 500ms for 5 minutes
groups:
  - name: database_alerts
    rules:
      - alert: HighQueryLatency
        expr: |
          histogram_quantile(0.99,
            rate(pg_query_duration_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p99 query latency {{ $value }}s exceeds 500ms"
```

== Self-Driving: NoisePage / Pilot

NoisePage (Carnegie Mellon, 2021) is a full self-driving DBMS prototype:

```
NoisePage self-driving loop:
  1. Workload forecast: predict future query mix (LSTM on past query logs)
  2. Action planning: enumerate actions (add index, change knob, reorg table)
  3. Cost modeling: estimate benefit of each action via learned cost models
  4. Action selection: choose action maximizing predicted performance
  5. Execute action (build index, set GUC, etc.)
  6. Observe actual performance, update models
```

*Key insight:* forecasting future workload allows *proactive* actions (build an index before a known report runs), not just reactive ones.

== References

Duan, S. et al. (2009). "Tuning Database Configuration Parameters with iTuned." VLDB.

Van Aken, D. et al. (2017). "Automatic Database Management System Tuning Through Large-Scale Machine Learning." SIGMOD. (OtterTune)

Marcus, R. et al. (2021). "Bao: Making Learned Query Optimization Practical." SIGMOD.

Pavlo, A. et al. (2019). "Self-Driving Database Management Systems." CIDR.

Dexter. "Automatically Select Missing PostgreSQL Indexes." https://github.com/ankane/dexter

PostgreSQL Documentation. "pg_stat_statements." https://www.postgresql.org/docs/current/pgstatstatements.html

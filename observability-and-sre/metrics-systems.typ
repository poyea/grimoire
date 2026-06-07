= Metrics Systems

A metrics system is a time-series database with a query language tuned for monitoring. The dominant lineage — Prometheus, VictoriaMetrics, Mimir, Thanos, Cortex, M3DB — descends from Borgmon, with shared assumptions: labeled multidimensional series, pull-based scraping, a functional query language, and chunked columnar storage on object storage. This chapter compares architectures, drills into PromQL, and covers the statistics (TDigest, HdrHistogram, DDSketch) without which percentile alerts are meaningless.

*See also:* _The Three Pillars and Beyond_, _SLO Engineering_, _Continuous Profiling_, _Time-Series and Graph Databases_ (database), _Network Observability_ (networking)

== Data Model

A Prometheus-family time series is identified by a metric name plus a set of labels:

```
http_requests_total{method="GET", route="/api/v1/users", status="200", instance="10.0.0.1:8080"}
```

The `(name, labels)` tuple is the *series identifier*. Each series is an append-only sequence of `(timestamp, float64)` pairs. The combinatorial space of label values dominates resource use; this is *cardinality*.

=== Four Metric Types

#table(
  columns: (auto, auto, auto),
  align: left,
  table.header[*Type*][*Semantics*][*Example*],
  [Counter], [Monotonically increasing], [`http_requests_total`],
  [Gauge], [Arbitrary up/down], [`memory_bytes`],
  [Histogram], [Bucketed observations], [`request_duration_seconds_bucket`],
  [Summary], [Pre-computed quantiles], [`request_duration_seconds{quantile="0.99"}`],
)

Counters survive process restarts via *reset detection*: the query engine notices when a value decreases and treats the drop as a reset rather than a negative rate. Summaries compute quantiles in the client, so they cannot be aggregated across instances; histograms can be.

== Architecture Comparison

#table(
  columns: (auto, auto, auto, auto),
  align: left,
  table.header[*System*][*Storage*][*HA Strategy*][*Notable Feature*],
  [Prometheus], [Local TSDB (mmap blocks)], [Run two, dedup at query], [Pull model, single binary],
  [VictoriaMetrics], [Custom inverted index + parts], [vmcluster sharding], [Lowest RAM/series; MetricsQL extensions],
  [Mimir], [Object storage + ingester WAL], [Replication factor 3], [Multi-tenant, horizontally scalable],
  [Thanos], [Object storage + sidecar], [Querier dedup], [Bolt-on to vanilla Prometheus],
  [Cortex], [Like Mimir (predecessor)], [Same], [Now superseded by Mimir],
  [M3DB], [Cassandra-style ring], [Quorum replication], [Uber-scale, complex ops],
  [InfluxDB IOx], [Parquet + Apache Arrow], [Per-shard], [SQL via DataFusion],
)

VictoriaMetrics typically uses 4–10$times$ less memory per active series than Prometheus by using a custom string interning scheme and delta-of-delta encoding similar to Gorilla. Mimir scales horizontally by sharding ingesters and pushing immutable blocks to S3 every two hours.

=== Pull vs Push

Prometheus pulls; VictoriaMetrics, InfluxDB, and OTel push by default. Pull's advantages: target discovery doubles as health-check, no need for client-side credentials, easier to firewall. Push's advantages: works behind NAT, batch jobs that exit too fast to be scraped, mobile/edge devices. The OTel Collector can convert: agents push to it, then it exposes a `/metrics` endpoint that Prometheus scrapes — best of both.

== PromQL Deep Dive

PromQL is a functional query language operating on *instant vectors* (a value per series at a single timestamp), *range vectors* (a slice of values per series over a window), scalars, and strings.

=== The Core Operators

```promql
# Per-instance request rate over 5 min, smoothed
rate(http_requests_total[5m])

# Aggregate rate across instances, grouped by route
sum by (route) (rate(http_requests_total[5m]))

# Error ratio (fraction)
sum(rate(http_requests_total{status=~"5.."}[5m]))
  / ignoring(status)
sum(rate(http_requests_total[5m]))

# P99 latency from histogram (Prometheus classic)
histogram_quantile(0.99,
  sum by (le, route) (rate(http_request_duration_seconds_bucket[5m])))

# Native histograms (Prometheus 2.40+) — exponential buckets
histogram_quantile(0.99, sum by (route) (rate(http_request_duration_seconds[5m])))
```

`rate` only works on counters; use `irate` for short windows (computes from last two points; volatile) and `increase` for human-readable totals. Always wrap raw counters in `rate` before aggregating; otherwise resets break the math.

=== Subqueries and `*_over_time`

```promql
# Max P99 over the last hour, evaluated every minute
max_over_time(
  histogram_quantile(0.99,
    sum by (le) (rate(http_request_duration_seconds_bucket[5m])))[1h:1m]
)
```

Subqueries are expensive — the range $[1h:1m]$ runs the inner query 60 times. Prefer recording rules to cache the inner result.

=== Joins: `on`, `ignoring`, `group_left`

```promql
# Annotate error rate with team label from a separate metric
sum by (service) (rate(errors_total[5m]))
  * on(service) group_left(team)
  service_team_info
```

`group_left` is the equivalent of a SQL many-to-one join. `on(service)` restricts the join key; `ignoring(...)` is the complement.

== Histograms: Classic vs Native

Classic Prometheus histograms expose per-bucket counters with explicit `le` (less-than-or-equal) labels:

```
http_request_duration_seconds_bucket{le="0.005"} 12
http_request_duration_seconds_bucket{le="0.01"}  45
http_request_duration_seconds_bucket{le="0.025"} 123
http_request_duration_seconds_bucket{le="+Inf"}  1000
http_request_duration_seconds_sum                42.3
http_request_duration_seconds_count              1000
```

`histogram_quantile` linearly interpolates inside the bucket containing the target quantile rank. Two failure modes:

- *Coarse buckets:* if $"P99"$ falls in a `le="1.0"` to `le="+Inf"` interval, the function returns $"+Inf"$.
- *Too many buckets:* each bucket is a separate series; 20 buckets $times$ 100 routes = 2000 series.

Native histograms (Prometheus 2.40+, "sparse histograms") encode exponentially spaced buckets in a single sample. Schema $s$ defines bucket width $2^(2^(-s))$; $s=8$ gives $approx 0.27%$ relative error. A single native histogram is one series regardless of bucket count, so cardinality drops by 10–50$times$.

== Sketches: TDigest, HdrHistogram, DDSketch

For aggregating quantiles across instances without exposing buckets, mergeable sketches are essential.

#table(
  columns: (auto, auto, auto, auto),
  align: left,
  table.header[*Sketch*][*Error*][*Mergeable?*][*Use*],
  [HdrHistogram], [Fixed relative, $approx 0.1%$], [Yes], [Latency in single process],
  [TDigest], [Variable, accurate at tails], [Yes (approx)], [General quantile sketch],
  [DDSketch], [Fixed relative, guaranteed], [Yes (exact)], [Distributed quantiles],
  [KLL], [$epsilon$-additive], [Yes], [Rank queries with proofs],
)

DDSketch (Datadog, 2019) has the unique property of *bounded relative error* that is preserved under merge — critical for distributed $"P99"$ computation. Its formula: bucket $i$ stores values in $[gamma^i, gamma^(i+1))$ with $gamma = (1+alpha)/(1-alpha)$; merging adds bucket counts.

```python
# DDSketch outline
class DDSketch:
    def __init__(self, alpha=0.01):
        self.gamma = (1 + alpha) / (1 - alpha)
        self.log_gamma = math.log(self.gamma)
        self.buckets = collections.Counter()
        self.count = 0

    def add(self, x):
        if x <= 0:
            return
        idx = math.ceil(math.log(x) / self.log_gamma)
        self.buckets[idx] += 1
        self.count += 1

    def quantile(self, q):
        rank = q * (self.count - 1)
        cum = 0
        for idx in sorted(self.buckets):
            cum += self.buckets[idx]
            if cum > rank:
                return 2 * self.gamma ** idx / (self.gamma + 1)

    def merge(self, other):
        self.buckets.update(other.buckets)
        self.count += other.count
```

== Storage Internals: Gorilla Encoding

Facebook's Gorilla paper (VLDB 2015) introduced the encoding now ubiquitous in TSDBs:

1. *Timestamp delta-of-delta:* store the difference between consecutive deltas. For a regular 15 s interval, the delta-of-delta is zero, encoded in 1 bit.
2. *Value XOR encoding:* `x_n XOR x_(n-1)` produces a value with many leading and trailing zeros for slowly changing series. Store leading-zero count + meaningful bits.

Typical compression: $1.37$ bytes per sample for Gorilla, vs $16$ bytes for raw `(timestamp_ns, float64)`. Prometheus's TSDB head block uses a variant; VictoriaMetrics adds Zstd on top of variable-bit-width encoding.

=== TSDB Block Layout

Prometheus persists data as immutable 2-hour blocks:

```
01HQVT3N5R/             # ULID block id
├── chunks/
│   └── 000001          # variable-length encoded chunks
├── index               # inverted index: label -> series -> chunk refs
├── tombstones          # deleted ranges
└── meta.json
```

Compaction merges adjacent blocks (2h → 8h → 24h → ... ) to bound the number of files. Retention is enforced by deleting old blocks.

== Recording Rules and Alerting Rules

Recording rules precompute expensive PromQL into new series; alerting rules fire when an expression returns nonempty results.

```yaml
groups:
- name: slo
  interval: 30s
  rules:
  - record: job:http_requests:rate5m
    expr: sum by (job, route) (rate(http_requests_total[5m]))

  - record: job:http_errors:rate5m
    expr: sum by (job, route) (rate(http_requests_total{status=~"5.."}[5m]))

  - alert: HighErrorRate
    expr: |
      job:http_errors:rate5m / job:http_requests:rate5m > 0.05
    for: 10m
    labels: { severity: page, team: checkout }
    annotations:
      summary: "Error rate {{ $value | humanizePercentage }} on {{ $labels.route }}"
      runbook: https://runbooks/checkout-errors
```

Recording rules also stabilize alerts: alert on `job:http_errors:rate5m`, not on the raw expression, so the alert and dashboard share definitions.

== Multi-Tenancy and Federation

At organization scale you need multi-tenancy (Mimir, Cortex) or hierarchical federation (Prometheus). Federation pulls aggregates from leaf Prometheus servers:

```yaml
# global Prometheus
scrape_configs:
- job_name: federate
  honor_labels: true
  metrics_path: /federate
  params:
    'match[]': ['{__name__=~"job:.*"}']  # only recording rule output
  static_configs:
    - targets: [prom-eu-1:9090, prom-us-1:9090]
```

Always restrict `match[]` to aggregated series; raw counters would overwhelm the global server.

== Exemplars in Practice

```python
from prometheus_client import Histogram

h = Histogram("rpc_duration_seconds", "RPC latency", ["method"])

with h.labels(method="GetUser").time() as t:
    span = tracer.start_span("GetUser")
    try:
        result = backend.get_user(uid)
    finally:
        span.end()
    # Attach exemplar
    t.exemplar(value=t._observed,
               labels={"trace_id": span.trace_id, "span_id": span.span_id})
```

== Anti-Patterns

- *Label explosion:* user IDs, request IDs, or unbounded paths in labels.
- *Time-of-day suffix:* `requests_2024_01_15` — defeats the point of labels.
- *Forgetting `rate`:* `sum(http_requests_total)` is meaningless and resets.
- *Quantile averaging:* averaging $"P99"$ across replicas is mathematically wrong; aggregate the buckets and recompute.
- *Aliasing alert and dashboard expressions:* keep both reading the same recording rule.

== Further Reading

Pelkonen, T. et al. (2015). "Gorilla: A Fast, Scalable, In-Memory Time Series Database." VLDB.

Masood, A., Volz, B. (2017). "Prometheus: Up & Running." O'Reilly.

Masson, C., Rim, J. E., Lee, H. K. (2019). "DDSketch: A Fast and Fully-Mergeable Quantile Sketch with Relative-Error Guarantees." VLDB.

Greenwald, M., Khanna, S. (2001). "Space-Efficient Online Computation of Quantile Summaries." SIGMOD.

Dunning, T., Ertl, O. (2019). "Computing Extremely Accurate Quantiles Using $t$-Digests."

Karger, D., HdrHistogram project. http://hdrhistogram.org

Grafana Labs. "Mimir Architecture." https://grafana.com/docs/mimir/

VictoriaMetrics. "Internals." https://docs.victoriametrics.com

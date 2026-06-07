= The Three Pillars and Beyond

Observability begins with three classical signal types — metrics, logs, traces — but production systems require two more first-class citizens: continuous profiles and discrete events. This chapter frames each pillar by its information density, query model, retention economics, and the questions it answers, then sketches how OpenTelemetry semantic conventions stitch them into a single graph keyed by resource and span context.

*See also:* _Metrics Systems_, _Distributed Tracing_, _Continuous Profiling_, _Logging Pipelines_, _Observability and Self-Driving Databases_ (database), _Network Observability_ (networking), _Kernel Tracing_ (linux-kernel)

== Why "Three Pillars" Underspecifies the Problem

Cindy Sridharan's 2017 framing — metrics, logs, traces — was a useful starting point but elides crucial distinctions. Metrics are pre-aggregated, low-cardinality, cheap to retain for years. Logs are post-hoc aggregated, often high-cardinality, expensive at scale. Traces capture causality but only at sampled instants. Add to these:

- *Profiles:* stack-sampled CPU, allocation, lock, and I/O attributions; the only signal that explains _where in code_ time was spent.
- *Events:* discrete state transitions (deploys, feature flag flips, config pushes, scaling events) that contextualize all other signals.

Charity Majors at Honeycomb argues for a unified *wide event* model where every signal is a structured event with arbitrary dimensions, and metrics/traces emerge as projections. OpenTelemetry's data model has converged on this: a `Resource` (process identity), `InstrumentationScope`, and per-signal records sharing `TraceId`/`SpanId` for correlation.

=== Information Density Compared

#table(
  columns: (auto, auto, auto, auto, auto),
  align: left,
  table.header[*Signal*][*Density*][*Cardinality*][*Cost / GB-month*][*Question Answered*],
  [Metrics], [Low (scalars)], [Bounded labels], [\$0.10–\$1], [Is something wrong?],
  [Logs], [Medium (text)], [Unbounded], [\$0.50–\$5], [What happened?],
  [Traces], [High (graphs)], [Per-request], [\$1–\$10], [Where did time go?],
  [Profiles], [Very high (stacks)], [Per-stack], [\$2–\$20], [Why is code slow?],
  [Events], [Variable], [Bounded], [\$0.10–\$2], [What changed?],
)

The "cost" column folds storage, ingestion, and query into one rough figure; real vendor pricing varies 10$times$ across these.

== The Resource Model

Every signal carries a `Resource`: the immutable identity of the producing process. Conventionally:

```yaml
resource:
  service.name: checkout
  service.namespace: payments
  service.instance.id: 7f2c-4a91
  service.version: 1.42.0
  deployment.environment: prod
  k8s.cluster.name: us-east-1-blue
  k8s.namespace.name: payments
  k8s.pod.name: checkout-58b9-q4m2c
  k8s.node.name: ip-10-0-1-42
  host.arch: arm64
  host.os.type: linux
  process.runtime.name: jvm
  process.runtime.version: 21.0.1
  telemetry.sdk.name: opentelemetry
  telemetry.sdk.language: java
  telemetry.sdk.version: 1.34.0
```

The resource is the join key across pillars. A metric exemplar with the same `(service.name, k8s.pod.name)` as a trace span lets you jump from a $"P99"$ spike to a concrete slow request.

== Telemetry as a Pipeline

Production telemetry is a multi-stage pipeline, not a single emit-and-store path:

#table(
  columns: (auto, auto, auto),
  align: left,
  table.header[*Stage*][*Purpose*][*Typical Tool*],
  [Instrumentation], [Emit raw signal], [OTel SDK, eBPF auto-instr],
  [Local agent], [Batch, retry, enrich], [OTel Collector (agent mode)],
  [Gateway], [Tail-sampling, fan-out, redact], [OTel Collector (gateway)],
  [Ingest], [Index, compress], [Prometheus remote_write, Loki, Tempo],
  [Storage], [Long-term object store], [S3 / GCS / object-store],
  [Query], [PromQL, LogQL, TraceQL], [Grafana, vendor UIs],
  [Action], [Alert, page, deploy], [Alertmanager, PagerDuty],
)

The collector pattern (an agent on every host plus a gateway tier) is now standard because it isolates instrumentation from backend choice, lets you enforce schema and PII rules in one place, and supports tail-based sampling without requiring SDK rewrites.

=== A Minimal OpenTelemetry Collector Config

```yaml
receivers:
  otlp:
    protocols:
      grpc: { endpoint: 0.0.0.0:4317 }
      http: { endpoint: 0.0.0.0:4318 }
  hostmetrics:
    collection_interval: 30s
    scrapers: { cpu: {}, memory: {}, disk: {}, network: {}, load: {} }

processors:
  batch:
    timeout: 5s
    send_batch_size: 8192
  memory_limiter:
    check_interval: 1s
    limit_percentage: 80
    spike_limit_percentage: 25
  resourcedetection:
    detectors: [env, system, gcp, ec2, eks]
  attributes/redact:
    actions:
      - key: http.request.header.authorization
        action: delete
      - key: user.email
        action: hash
  tail_sampling:
    decision_wait: 10s
    policies:
      - { name: errors, type: status_code, status_code: { status_codes: [ERROR] } }
      - { name: slow,   type: latency,     latency: { threshold_ms: 1000 } }
      - { name: prob,   type: probabilistic, probabilistic: { sampling_percentage: 1 } }

exporters:
  prometheusremotewrite: { endpoint: https://mimir:9009/api/v1/push }
  otlphttp/tempo:        { endpoint: https://tempo:4318 }
  loki:                  { endpoint: https://loki:3100/loki/api/v1/push }

service:
  pipelines:
    metrics: { receivers: [otlp, hostmetrics], processors: [memory_limiter, resourcedetection, batch], exporters: [prometheusremotewrite] }
    traces:  { receivers: [otlp], processors: [memory_limiter, resourcedetection, attributes/redact, tail_sampling, batch], exporters: [otlphttp/tempo] }
    logs:    { receivers: [otlp], processors: [memory_limiter, resourcedetection, attributes/redact, batch], exporters: [loki] }
```

The `memory_limiter` must come before `batch` so back-pressure propagates upstream rather than OOM-killing the collector. `tail_sampling` requires `decision_wait` long enough for the whole trace to arrive but short enough to bound memory.

== Cardinality Is the Master Variable

Every observability cost — disk, RAM, query time — scales with *active series* or *unique-event count*. A label like `user_id` on a metric, if mistakenly emitted, multiplies series count by users. Lessons:

- Move per-request identifiers (`request_id`, `trace_id`, `user_id`) out of metrics into traces/logs/events.
- Bound histogram bucket counts: 8–12 buckets, geometrically spaced.
- For high-cardinality dimensions you _must_ query, use a columnar event store (ClickHouse, Honeycomb, Axiom) rather than a TSDB.
- Adopt the *RED* method (Rate, Errors, Duration) per endpoint and the *USE* method (Utilization, Saturation, Errors) per resource as low-cardinality defaults.

A concrete heuristic: target $<= 10^6$ active series per Prometheus shard, $<= 10^7$ per Mimir/VictoriaMetrics shard. Above that, you are paying for cardinality you cannot consume.

== Exemplars: The Bridge Between Pillars

OpenMetrics exemplars attach a sampled `(trace_id, span_id, timestamp)` to a histogram bucket. Grafana renders them as dots on a latency panel; clicking one jumps to the trace.

```
http_request_duration_seconds_bucket{le="0.5",method="POST",route="/checkout"} 12453 # {trace_id="4bf92f3577b34da6a3ce929d0e0e4736",span_id="00f067aa0ba902b7"} 0.473 1701377812.123
```

This single line is metric + trace correlation in 200 bytes. Without exemplars you guess which trace caused the $"P99"$ spike; with them you click through.

== Wide Events and the "Single Pane"

Honeycomb's argument — repeated by Hightower, Sridharan, Majors — is that traces _are_ structured logs with parent pointers, metrics _are_ aggregations over those logs, and profiles _are_ logs of stack frames. If your storage can handle high-cardinality columnar event data (ClickHouse, Apache Parquet on object store, vendor backends), you can collapse the pillars.

The counter-argument: each pillar has different access patterns. Metrics queries scan years of low-cardinality data; trace queries fetch a handful of full requests; profile queries aggregate sampled stacks. Specialized stores still dominate at hyperscale because cost-per-query differs by orders of magnitude. The convergence happens at the SDK and the query language (PromQL / LogQL / TraceQL share grammar), not at the storage layer.

== Sampling, Not Aggregation, Is the Hard Problem

For traces and profiles you cannot retain everything. Sampling decisions interact:

- *Head sampling* (decide at root span): cheap, but biased — slow requests are statistically the same as fast ones.
- *Tail sampling* (decide after all spans arrive): unbiased but requires buffering the entire trace, expensive at scale.
- *Adaptive sampling:* keep all errors + slow tails, sample fast successes at $1$–$0.01$%; this is what most production deployments do.

Profiles use stratified time-based sampling: 100 Hz CPU profile = 1 stack/10 ms per core. At 1000 cores and 24 h, that is $8.64 dot 10^9$ stacks — only viable with stack-aggregating storage like Pyroscope's tree compression.

== Anti-Patterns to Avoid

- *Logging the trace_id only:* no parent/child = not a trace, just labeled logs.
- *Per-customer dashboards:* $10^4$ customers $times$ $10$ panels = $10^5$ time series the eye cannot scan.
- *Alerting on raw metrics:* alert on $"SLO"$ burn rate, not on $"P99" > X$ — see _SLO Engineering_.
- *Separate ingestion for each signal:* triples your collector failure modes.
- *Sampling at the SDK:* loses the ability to do tail-based decisions later.

== Mapping Questions to Pillars

#table(
  columns: (auto, auto),
  align: left,
  table.header[*Question*][*Primary Signal*],
  [Is the service healthy right now?], [Metrics ($"SLO"$ burn rate)],
  [What broke at 14:32?], [Events + logs],
  [Why is endpoint X slow?], [Traces ($"P99"$ spans)],
  [Why is CPU high?], [CPU profile],
  [Why is memory high?], [Allocation profile],
  [Which customer is hammering us?], [Wide events / high-cardinality store],
  [Is the regression in v1.42.0?], [Metrics filtered by `service.version`],
)

Match the question to the pillar; do not over-instrument the wrong one.

== Further Reading

Sridharan, C. (2018). _Distributed Systems Observability_. O'Reilly.

Majors, C., Fong-Jones, L., Miranda, G. (2022). _Observability Engineering_. O'Reilly.

Beyer, B. et al. (2016). _Site Reliability Engineering_. Google / O'Reilly. Chapters 6, 10.

OpenTelemetry Specification, v1.34. https://opentelemetry.io/docs/specs/otel/

Sigelman, B. et al. (2010). "Dapper, a Large-Scale Distributed Systems Tracing Infrastructure." Google Technical Report.

Gregg, B. (2020). _Systems Performance_, 2nd ed., Pearson. USE method, chapter 2.

Wilkie, T. (2018). "The RED Method." Weaveworks Blog.

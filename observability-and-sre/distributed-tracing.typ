= Distributed Tracing

A trace is a causally ordered tree (or DAG) of spans across processes that together represent one logical request. Tracing answers _where_ time went and _which_ downstream calls amplify latency or errors. This chapter starts from the Dapper paper, follows the evolution through Zipkin and Jaeger to OpenTelemetry, and covers the hard parts: context propagation, sampling, and tail-based decisions at scale.

*See also:* _The Three Pillars and Beyond_, _Metrics Systems_, _SLO Engineering_, #emph[networking/observability.typ], #emph[networking/grpc.typ], #emph[database/observability-and-self-driving.typ]

== The Span Model

A *span* is the unit of work: a service handling an RPC, a database query, a span emitted by an instrumented library. Spans carry:

- `trace_id` (128-bit, unique per request)
- `span_id` (64-bit, unique within a trace)
- `parent_span_id` (the caller; null for root)
- start/end timestamps (nanoseconds, ideally from a monotonic clock corrected to wall time)
- a `Status` (`OK`, `ERROR`, `UNSET`) with optional message
- attributes (key-value, bounded cardinality)
- *events* (timestamped annotations: `db.query_completed`, `exception`)
- *links* (references to spans in other traces, e.g., for fan-out or batching)
- a `SpanKind` (`SERVER`, `CLIENT`, `INTERNAL`, `PRODUCER`, `CONSUMER`)

The parent pointer plus the trace_id is enough to reassemble the tree at query time. Modern systems (OpenTelemetry, Tempo) store spans as flat records, indexing by `trace_id`, and reconstruct on read — this avoids cross-span coordination at write time.

=== A Concrete Span (OTLP/JSON)

```json
{
  "traceId":  "4bf92f3577b34da6a3ce929d0e0e4736",
  "spanId":   "00f067aa0ba902b7",
  "parentSpanId": "5fb397bf4c2a9a7b",
  "name": "POST /checkout",
  "kind": "SPAN_KIND_SERVER",
  "startTimeUnixNano": "1701377812123456789",
  "endTimeUnixNano":   "1701377812573456789",
  "attributes": [
    {"key":"http.method","value":{"stringValue":"POST"}},
    {"key":"http.route","value":{"stringValue":"/checkout"}},
    {"key":"http.status_code","value":{"intValue":200}},
    {"key":"net.peer.ip","value":{"stringValue":"10.0.0.42"}}
  ],
  "events": [
    {"name":"cache.miss","timeUnixNano":"1701377812210000000"},
    {"name":"db.query.complete","timeUnixNano":"1701377812370000000",
     "attributes":[{"key":"db.statement","value":{"stringValue":"SELECT ..."}}]}
  ],
  "status": {"code":"STATUS_CODE_OK"}
}
```

== Context Propagation

The hardest part of tracing is propagating context across process and language boundaries. W3C *Trace Context* standardizes the headers:

```
traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
                ^^                              ^^                ^^
                version                       parent-id          flags (sampled)
                       trace-id
tracestate: rojo=00f067aa0ba902b7,congo=t61rcWkgMzE
```

`traceparent` is required; `tracestate` carries vendor-specific data (a comma-separated list, each entry $<=$ 256 chars). The 8-bit `flags` field has only `sampled` (bit 0) defined today.

Baggage (W3C *Baggage* header) is a separate propagation channel for application data:

```
baggage: userId=alice,session=abcd1234,env=staging
```

Baggage propagates through the whole call tree; use it sparingly (each downstream span sees it, increases header size). It is not part of the trace itself.

=== Cross-Cutting Tools

- *gRPC:* W3C Trace Context lives in HTTP/2 headers; OTel's gRPC interceptors handle injection/extraction.
- *Message queues:* attach `traceparent` as a Kafka header / SQS message attribute. Add a `link` from the consumer span back to the producer.
- *Async / fan-out:* use `Span.AddLink` with semantic `messaging.operation = receive`.
- *Database drivers:* OTel JDBC, asyncpg, sqlx wrappers automatically add `db.system`, `db.statement` (sanitized), `db.connection_string` attributes.

== Sampling Strategies

For high-traffic services you cannot record every request. Sampling decides which traces survive.

=== Head-Based Sampling

The root span decides at trace creation; the decision is propagated via `traceparent` flags. Variants:

- *Always-on:* dev environments only.
- *Probabilistic:* keep $p$% uniformly. Cheap, biased toward "average" requests.
- *Rate-limiting:* up to N traces/sec, drop the rest.
- *Parent-based:* honor the upstream decision. Required for consistent trees.

Head sampling is easy and cheap but cannot keep "all errors" or "all slow requests" because the decision happens before outcomes are known.

=== Tail-Based Sampling

Buffer the entire trace until all spans arrive, then decide. Requires:

- A *gateway* tier (OTel Collector with `tail_sampling` processor) that shards by `trace_id` so all spans of one trace land on the same gateway.
- A *decision window* (typically 10–30 s) longer than max trace duration.
- *Policies* combined with OR/AND.

```yaml
processors:
  tail_sampling:
    decision_wait: 30s
    num_traces: 100000
    expected_new_traces_per_sec: 10000
    policies:
      - { name: errors,  type: status_code, status_code: { status_codes: [ERROR] } }
      - { name: slow,    type: latency, latency: { threshold_ms: 1000 } }
      - { name: key_op,  type: string_attribute,
          string_attribute: { key: http.route, values: [/checkout, /pay] } }
      - { name: rare_db, type: string_attribute,
          string_attribute: { key: db.statement, values: ["DELETE FROM .*"], enabled_regex_matching: true } }
      - { name: prob,    type: probabilistic, probabilistic: { sampling_percentage: 0.5 } }
```

Trade-off: tail sampling needs RAM proportional to (traces/sec) $times.o$ (decision_wait) $times.o$ (avg spans/trace). At 10k traces/sec, 30 s window, 50 spans/trace, that is 15 M spans buffered.

=== Adaptive and Throughput-Limited

Lightstep / Honeycomb implement *dynamic sampling*: for each unique combination of low-cardinality keys (route, status, customer tier), keep a target rate. Rare combinations are kept always; common ones are sampled aggressively. The result is statistically reweighted at query time so aggregates remain unbiased.

```python
# Sketch of dynamic sampling
class DynamicSampler:
    def __init__(self, target_per_key=10, window_s=30):
        self.counts = collections.Counter()
        self.target = target_per_key
        self.window = window_s

    def should_keep(self, key):
        self.counts[key] += 1
        rate = max(1, self.counts[key] / self.target)
        return random.random() < 1.0 / rate, rate  # weight = rate
```

The `weight` is stored on the span; queries multiply counts by weight to estimate true volume.

== Backend Comparison

#table(
  columns: (auto, auto, auto, auto, auto),
  align: left,
  table.header[*System*][*Storage*][*Query*][*Sampling*][*Strengths*],
  [Jaeger], [Cassandra / ES / Badger], [Trace-by-id, service+op search], [Head, basic tail], [OSS reference impl],
  [Zipkin], [MySQL / ES / Cassandra], [Limited search], [Head], [Lightweight],
  [Tempo], [Object storage (S3, GCS)], [TraceQL], [Defer to OTel Collector], [Cheap at scale],
  [OpenTelemetry], [N/A (spec + SDK)], [N/A], [Both], [Vendor-neutral],
  [Honeycomb], [Proprietary columnar], [Bubble-up, BubbleQL], [Dynamic], [High-cardinality],
  [Lightstep / ServiceNow], [Proprietary], [Service maps], [Adaptive], [Tail at $10^6$ traces/sec],
  [Datadog APM], [Proprietary], [App-level], [Adaptive], [Integrated suite],
)

Tempo's design point — object storage with a tiny ingester — is the cheapest per-span: about \$0.001 per million spans stored. The trade-off is no full-text search; you must already know the `trace_id` or filter by service+time, then read the trace.

=== TraceQL

Grafana's TraceQL is to traces what PromQL is to metrics — a declarative language for trace filtering:

```traceql
# All traces touching /checkout where total duration > 1s
{ resource.service.name = "checkout" && duration > 1s }

# Spans where the DB is slow
{ span.db.system = "postgres" && span.duration > 100ms }

# Structural: traces where a slow DB span is a descendant of /checkout
{ span.http.route = "/checkout" } >> { span.db.system = "postgres" && span.duration > 100ms }

# Aggregation
{ resource.service.name = "checkout" } | quantile_over_time(duration, 0.99) by (span.http.route)
```

`>>` is the descendant operator; `>` is direct child; `&&`/`||` combine.

== Lineage and Causal Tracing

Dapper (Sigelman et al., 2010) introduced the simple model used today. Earlier systems explored richer causality:

- *Magpie* (MSR, 2003): per-request event traces using PID + thread IDs; reconstructs paths via event schemas instead of context propagation. Catches lost-context bugs but harder to deploy.
- *X-Trace* (Berkeley, 2007): pushes propagation into IP options and HTTP headers; influenced W3C Trace Context.
- *Pivot Tracing* (SOSP 2015): dynamic instrumentation with happens-before joins, used for ad-hoc questions ("which slow disk caused the user-facing tail latency?").
- *Pythia* (HotOS 2019): hierarchical causal sketches with low overhead.

Modern OTel covers Dapper-style well; richer lineage (e.g., joins across asynchronous fan-out) is still active research.

== Instrumentation Patterns

=== Auto-Instrumentation

OTel offers zero-code instrumentation in JVM, Python, Node, .NET, Go, Ruby:

```bash
# Java: agent attaches at JVM start, hooks JDBC/HTTP/Kafka/gRPC
java -javaagent:opentelemetry-javaagent.jar \
     -Dotel.service.name=checkout \
     -Dotel.exporter.otlp.endpoint=http://otel-collector:4317 \
     -jar app.jar
```

Auto-instrumentation covers ~80% of the value with zero code change. The remaining 20% — business-level spans, custom attributes, key events — requires manual instrumentation:

```python
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("checkout.process")
def process(order):
    span = trace.get_current_span()
    span.set_attribute("order.id", order.id)
    span.set_attribute("order.total_cents", order.total_cents)
    span.set_attribute("customer.tier", order.customer.tier)

    if not in_stock(order):
        span.add_event("inventory.miss", {"sku": order.sku})
        span.set_status(trace.Status(trace.StatusCode.ERROR, "out of stock"))
        raise OutOfStock(order.sku)

    charge_card(order)  # creates child span via auto-instr
    return ship(order)
```

=== Naming and Attribute Conventions

Follow OTel semantic conventions: `http.request.method`, `http.response.status_code`, `db.system`, `messaging.system`, `rpc.system`. Custom attributes prefixed with your domain: `myapp.feature_flag`, `myapp.tenant_id`.

== Latency Math: Why Traces Disagree With Metrics

A trace shows one observation; a metric shows a quantile over many. They _will_ disagree:

- A single trace at $"P99"$ latency is by definition rare. Pulling "any" trace and assuming it represents the slow path is wrong.
- Exemplars solve this: the metric tells you the bucket and provides a span_id sampled from inside that bucket.
- $"P99"$ aggregated across services is _not_ the sum of per-service $"P99"$s. Use trace-level critical-path analysis instead.

=== Critical Path

Given a trace tree, the *critical path* is the longest chain of dependent spans. Parallel children do not extend it; sequential children do. Tooling: Lightstep "service map", Honeycomb "bubble-up", or post-process in Python:

```python
def critical_path(span, spans_by_parent):
    children = spans_by_parent.get(span.span_id, [])
    if not children:
        return [span]
    # Pick the child whose end time is closest to this span's end
    last_child = max(children, key=lambda s: s.end)
    return [span] + critical_path(last_child, spans_by_parent)
```

Optimizing anything off the critical path does not improve user-perceived latency.

== Anti-Patterns

- *Recording PII in attributes:* sanitize at the collector with `attributes/redact`.
- *Sampling at the SDK:* loses the option of tail-based decisions.
- *Forgetting to propagate context to async work:* breaks the tree; use `Context.attach`/`detach` properly.
- *Synchronous span export:* always batch and export off the request path.
- *Treating spans as logs:* attributes have schemas and cardinality limits; long stack traces belong in events with `exception.stacktrace`, not in 50 attributes.

== Further Reading

Sigelman, B. et al. (2010). "Dapper, a Large-Scale Distributed Systems Tracing Infrastructure." Google.

Fonseca, R. et al. (2007). "X-Trace: A Pervasive Network Tracing Framework." NSDI.

Barham, P. et al. (2003). "Magpie: Online Modelling and Performance-Aware Systems." HotOS.

Mace, J., Roelke, R., Fonseca, R. (2015). "Pivot Tracing: Dynamic Causal Monitoring for Distributed Systems." SOSP.

W3C. (2021). "Trace Context Level 1." https://www.w3.org/TR/trace-context/

OpenTelemetry Specification. https://opentelemetry.io/docs/specs/otel/

Las-Casas, P. et al. (2019). "Sifter: Scalable Sampling for Distributed Traces, Using Hierarchical Clustering." SoCC.

Sambasivan, R. R. et al. (2014). "So, You Want To Trace Your Distributed System? Key Design Insights from Years of Practical Experience." CMU-PDL-14-102.

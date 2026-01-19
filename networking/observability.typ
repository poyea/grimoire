= Network Observability

Observability encompasses metrics, logs, and traces to understand system behavior. In distributed systems, correlating these signals across network boundaries is essential for debugging and performance optimization.

*See also:* Debugging (for packet-level analysis), Service Mesh (for sidecar-based telemetry), Performance Reference (for baseline metrics)

== Observability Pillars

*Three pillars of observability:*

```
┌─────────────────────────────────────────────────────────────┐
│                    Observability                             │
├─────────────────┬─────────────────┬─────────────────────────┤
│     Metrics     │      Logs       │        Traces           │
├─────────────────┼─────────────────┼─────────────────────────┤
│ - Numeric       │ - Textual       │ - Causal               │
│ - Aggregated    │ - Individual    │ - Distributed          │
│ - Time-series   │ - Events        │ - Request-scoped       │
│                 │                 │                         │
│ "What happened" │ "Event details" │ "Request flow"         │
└─────────────────┴─────────────────┴─────────────────────────┘
```

*Cardinality considerations:*
- Metrics: Low cardinality (dimensions), high volume (samples)
- Logs: High cardinality (content), medium volume
- Traces: High cardinality (spans), low volume (sampled)

*Storage costs (typical):*
- Metrics: ~10 bytes/sample (compressed)
- Logs: ~200 bytes/event
- Traces: ~500 bytes/span

== Distributed Tracing Concepts

*Trace structure:*

```
Trace (single request across services)
├── Span A: API Gateway (root span)
│   ├── Span B: Auth Service
│   │   └── Span C: Redis lookup
│   └── Span D: Order Service
│       ├── Span E: Inventory check
│       └── Span F: Database query
```

*Span data model:*

```cpp
struct Span {
    // Identity
    TraceID trace_id;        // 128-bit, shared across trace
    SpanID span_id;          // 64-bit, unique per span
    SpanID parent_span_id;   // Links to parent (0 for root)

    // Timing
    uint64_t start_time_ns;
    uint64_t end_time_ns;

    // Metadata
    string operation_name;   // "HTTP GET /api/orders"
    string service_name;
    SpanKind kind;           // CLIENT, SERVER, PRODUCER, CONSUMER

    // Attributes (key-value pairs)
    map<string, Value> attributes;

    // Events (timestamped logs within span)
    vector<Event> events;

    // Links (related traces, e.g., async operations)
    vector<Link> links;

    // Status
    StatusCode status;       // UNSET, OK, ERROR
    string status_message;
};
```

*Context propagation:*

```
Service A                    Service B
┌───────────┐  HTTP Request  ┌───────────┐
│           │───────────────▶│           │
│  Span A   │  Headers:      │  Span B   │
│           │  traceparent:  │           │
│           │  00-{trace_id}-│           │
│           │  {span_id}-01  │           │
└───────────┘                └───────────┘
```

*W3C Trace Context [W3C Recommendation]:*

```
traceparent: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
             ▲  ▲                               ▲                  ▲
             │  │                               │                  │
          version  trace-id (32 hex)      parent-id (16 hex)   flags

tracestate: vendor1=value1,vendor2=value2
```

== OpenTelemetry

*OpenTelemetry (OTel) is the industry standard for observability instrumentation [CNCF]:*

*Architecture:*

```
┌─────────────────────────────────────────────────────────────┐
│                     Application                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              OpenTelemetry SDK                       │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │    │
│  │  │ Tracer   │ │  Meter   │ │  Logger  │            │    │
│  │  │ Provider │ │ Provider │ │ Provider │            │    │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘            │    │
│  │       │            │            │                   │    │
│  │       └────────────┼────────────┘                   │    │
│  │                    ▼                                │    │
│  │            ┌──────────────┐                        │    │
│  │            │   Exporter   │                        │    │
│  │            └──────┬───────┘                        │    │
│  └───────────────────┼─────────────────────────────────┘    │
│                      │ OTLP (gRPC/HTTP)                     │
└──────────────────────┼──────────────────────────────────────┘
                       ▼
              ┌──────────────────┐
              │  OTel Collector  │
              │  ┌────────────┐  │
              │  │ Receivers  │  │
              │  │ Processors │  │
              │  │ Exporters  │  │
              │  └────────────┘  │
              └────────┬─────────┘
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │  Jaeger │  │Prometheus│  │  Loki   │
    │(Traces) │  │(Metrics) │  │ (Logs)  │
    └─────────┘  └─────────┘  └─────────┘
```

*Instrumentation example (C++):*

```cpp
#include <opentelemetry/trace/provider.h>
#include <opentelemetry/exporters/otlp/otlp_grpc_exporter.h>
#include <opentelemetry/sdk/trace/tracer_provider.h>

namespace trace = opentelemetry::trace;
namespace otlp = opentelemetry::exporter::otlp;

// Initialize tracer
void init_tracer() {
    auto exporter = otlp::OtlpGrpcExporterFactory::Create();
    auto processor = trace::SimpleSpanProcessorFactory::Create(
        std::move(exporter));
    auto provider = trace::TracerProviderFactory::Create(
        std::move(processor));

    trace::Provider::SetTracerProvider(std::move(provider));
}

// Create spans
void handle_request(const Request& req) {
    auto tracer = trace::Provider::GetTracerProvider()
        ->GetTracer("my-service", "1.0.0");

    auto span = tracer->StartSpan("handle_request");
    auto scope = tracer->WithActiveSpan(span);

    // Add attributes
    span->SetAttribute("http.method", req.method);
    span->SetAttribute("http.url", req.url);

    // Create child span for database call
    {
        auto db_span = tracer->StartSpan("db_query");
        auto db_scope = tracer->WithActiveSpan(db_span);

        auto result = database.query(req.query);

        db_span->SetAttribute("db.statement", req.query);
        db_span->SetAttribute("db.rows_affected", result.rows);
    }

    span->SetStatus(trace::StatusCode::kOk);
}
```

*Auto-instrumentation (available libraries):*
- HTTP clients/servers (gRPC, curl, etc.)
- Database drivers (PostgreSQL, MySQL, Redis)
- Message queues (Kafka, RabbitMQ)
- Cloud SDKs (AWS, GCP, Azure)

== Jaeger Architecture

*Jaeger is a popular open-source distributed tracing system [Uber Engineering]:*

```
┌─────────────────────────────────────────────────────────────┐
│                     Jaeger Architecture                      │
│                                                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
│  │Service A│  │Service B│  │Service C│                     │
│  └────┬────┘  └────┬────┘  └────┬────┘                     │
│       │            │            │                           │
│       └────────────┼────────────┘                           │
│                    │ UDP/HTTP (spans)                       │
│                    ▼                                        │
│           ┌──────────────────┐                             │
│           │   Jaeger Agent   │  (per-host, batches spans)  │
│           │   (optional)     │                              │
│           └────────┬─────────┘                             │
│                    │ gRPC                                   │
│                    ▼                                        │
│           ┌──────────────────┐                             │
│           │ Jaeger Collector │  (validates, indexes)       │
│           └────────┬─────────┘                             │
│                    │                                        │
│                    ▼                                        │
│           ┌──────────────────┐                             │
│           │     Storage      │  (Cassandra, Elasticsearch, │
│           │                  │   Badger, ClickHouse)       │
│           └────────┬─────────┘                             │
│                    │                                        │
│                    ▼                                        │
│           ┌──────────────────┐                             │
│           │   Jaeger Query   │  (API + UI)                 │
│           └──────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

*Sampling strategies:*

```cpp
// 1. Constant sampling (all or nothing)
sampler:
  type: const
  param: 1  // 1 = sample all, 0 = sample none

// 2. Probabilistic sampling
sampler:
  type: probabilistic
  param: 0.1  // Sample 10% of traces

// 3. Rate limiting
sampler:
  type: ratelimiting
  param: 2.0  // 2 traces per second

// 4. Remote (adaptive)
sampler:
  type: remote
  param: http://jaeger-agent:5778/sampling
```

*Adaptive sampling:*
- Controller analyzes traffic patterns
- Increases sampling for rare operations
- Decreases for high-volume operations
- Ensures all operation types represented

*Performance impact:*
- Span creation: ~500ns
- Span export (batched): ~1-5μs amortized
- Memory overhead: ~1KB per in-flight span

== Metrics Pipelines

*Prometheus data model:*

```
# Metric name + labels = time series
http_requests_total{method="GET", path="/api", status="200"} 1523
                    └─────────── labels ────────────────┘   └─value─┘

# Types:
- Counter: Monotonically increasing (requests, bytes)
- Gauge: Can go up or down (temperature, queue size)
- Histogram: Distribution of values (latency percentiles)
- Summary: Client-side calculated percentiles
```

*Histogram vs Summary:*

```
Histogram (server-side aggregation):
http_request_duration_seconds_bucket{le="0.1"} 24054
http_request_duration_seconds_bucket{le="0.25"} 26789
http_request_duration_seconds_bucket{le="0.5"} 27892
http_request_duration_seconds_bucket{le="1"} 28554
http_request_duration_seconds_bucket{le="+Inf"} 28768
http_request_duration_seconds_sum 8734.5
http_request_duration_seconds_count 28768

Summary (client-side calculation):
http_request_duration_seconds{quantile="0.5"} 0.052
http_request_duration_seconds{quantile="0.9"} 0.287
http_request_duration_seconds{quantile="0.99"} 0.892
```

*When to use:*
- Histogram: Aggregatable, SLO calculations, flexible queries
- Summary: Accurate percentiles, but not aggregatable across instances

*Push vs Pull architecture:*

```
Pull (Prometheus):                Push (StatsD, OTLP):
┌──────────┐                      ┌──────────┐
│ Scraper  │──GET /metrics──▶     │  Agent   │◀── UDP packets
└──────────┘                      └──────────┘
     │                                 │
     ▼                                 ▼
┌──────────┐                      ┌──────────┐
│  TSDB    │                      │  Server  │
└──────────┘                      └──────────┘

Pros:                             Pros:
- No SDK in app                   - Works behind NAT
- Centralized control             - Short-lived jobs
- Service discovery               - Lower app coupling

Cons:                             Cons:
- Must expose endpoint            - Needs aggregation
- Network reachability            - Cardinality explosion risk
```

*Metrics instrumentation (C++):*

```cpp
#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/registry.h>

using namespace prometheus;

// Global registry and metrics
auto registry = std::make_shared<Registry>();

auto& request_counter = BuildCounter()
    .Name("http_requests_total")
    .Help("Total HTTP requests")
    .Register(*registry);

auto& request_latency = BuildHistogram()
    .Name("http_request_duration_seconds")
    .Help("Request latency in seconds")
    .Register(*registry);

// Per-handler metrics
auto& get_requests = request_counter.Add({{"method", "GET"}});
auto& get_latency = request_latency.Add(
    {{"method", "GET"}},
    Histogram::BucketBoundaries{0.001, 0.01, 0.1, 0.5, 1.0, 5.0}
);

void handle_get_request() {
    auto start = std::chrono::steady_clock::now();

    // ... handle request ...

    auto duration = std::chrono::steady_clock::now() - start;
    double seconds = std::chrono::duration<double>(duration).count();

    get_requests.Increment();
    get_latency.Observe(seconds);
}
```

== Log Correlation

*Structured logging with trace context:*

```cpp
#include <spdlog/spdlog.h>

void handle_request(const Request& req) {
    auto span = tracer->StartSpan("handle_request");

    // Extract trace context
    auto ctx = span->GetContext();
    auto trace_id = ctx.trace_id().ToHexString();
    auto span_id = ctx.span_id().ToHexString();

    // Log with correlation IDs
    spdlog::info(
        R"({{"trace_id":"{}","span_id":"{}","event":"request_received","method":"{}","path":"{}"}})",
        trace_id, span_id, req.method, req.path
    );

    // Process request...

    spdlog::info(
        R"({{"trace_id":"{}","span_id":"{}","event":"request_completed","status":{},"duration_ms":{}}})",
        trace_id, span_id, response.status, duration_ms
    );
}
```

*Log format (JSON for machine parsing):*

```json
{
  "timestamp": "2024-01-15T10:23:45.123Z",
  "level": "INFO",
  "service": "order-service",
  "trace_id": "0af7651916cd43dd8448eb211c80319c",
  "span_id": "b7ad6b7169203331",
  "event": "order_created",
  "order_id": "12345",
  "user_id": "user-789",
  "amount": 99.99,
  "duration_ms": 45
}
```

*Log aggregation pipeline:*

```
┌─────────────────────────────────────────────────────────────┐
│  Applications                                                │
│  ┌──────┐ ┌──────┐ ┌──────┐                                │
│  │ App1 │ │ App2 │ │ App3 │                                │
│  └──┬───┘ └──┬───┘ └──┬───┘                                │
│     │ stdout │        │                                     │
│     └────────┼────────┘                                     │
│              ▼                                              │
│  ┌────────────────────────┐                                │
│  │  Log Shipper           │  (Fluent Bit, Vector, Filebeat)│
│  │  - Parse structured    │                                 │
│  │  - Add metadata        │                                 │
│  │  - Buffer and batch    │                                 │
│  └───────────┬────────────┘                                │
│              │                                              │
│              ▼                                              │
│  ┌────────────────────────┐                                │
│  │  Log Aggregator        │  (Loki, Elasticsearch)         │
│  │  - Index by labels     │                                 │
│  │  - Compress storage    │                                 │
│  │  - Query interface     │                                 │
│  └───────────┬────────────┘                                │
│              │                                              │
│              ▼                                              │
│  ┌────────────────────────┐                                │
│  │  Query/Visualization   │  (Grafana)                     │
│  │  - Search by trace_id  │                                 │
│  │  - Correlation views   │                                 │
│  └────────────────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

*Loki query examples:*

```logql
# Find logs for specific trace
{service="order-service"} |= "trace_id=0af7651916cd43dd"

# Error rate by service
sum(rate({level="ERROR"}[5m])) by (service)

# Extract fields and filter
{service="api-gateway"}
  | json
  | duration_ms > 1000
  | line_format "{{.method}} {{.path}} took {{.duration_ms}}ms"
```

== Application Performance Monitoring (APM)

*APM combines traces, metrics, and service topology:*

```
┌─────────────────────────────────────────────────────────────┐
│                    APM Dashboard                             │
├─────────────────────────────────────────────────────────────┤
│  Service Map:                                                │
│                                                              │
│      ┌──────────┐     ┌──────────┐     ┌──────────┐        │
│      │ Frontend │────▶│   API    │────▶│ Database │        │
│      │  p99:45ms│     │ p99:120ms│     │ p99:30ms │        │
│      │  err:0.1%│     │ err:0.5% │     │ err:0.01%│        │
│      └──────────┘     └──────────┘     └──────────┘        │
│                            │                                 │
│                            ▼                                 │
│                       ┌──────────┐                          │
│                       │  Cache   │                          │
│                       │ p99:2ms  │                          │
│                       │ hit:95%  │                          │
│                       └──────────┘                          │
├─────────────────────────────────────────────────────────────┤
│  Key Metrics:                                                │
│  - Throughput: 1,234 req/s                                  │
│  - Latency P50: 23ms  P99: 187ms  P99.9: 892ms             │
│  - Error rate: 0.23%                                        │
│  - Apdex: 0.94                                              │
└─────────────────────────────────────────────────────────────┘
```

*RED method (for services):*
- *Rate:* Requests per second
- *Errors:* Failed requests per second
- *Duration:* Latency distribution

*USE method (for resources):*
- *Utilization:* Percentage of resource busy
- *Saturation:* Queue length / waiting work
- *Errors:* Error count

*SLI/SLO/SLA framework:*

```cpp
// Service Level Indicator (SLI): Measurable metric
double availability_sli = successful_requests / total_requests;
double latency_sli = requests_under_threshold / total_requests;

// Service Level Objective (SLO): Target for SLI
const double AVAILABILITY_SLO = 0.999;  // 99.9%
const double LATENCY_SLO = 0.95;        // 95% under 200ms

// Error budget
double error_budget = 1.0 - AVAILABILITY_SLO;  // 0.1%
double budget_consumed = (1.0 - availability_sli) / error_budget;
```

*Alerting strategy:*

```yaml
# Multi-window, multi-burn-rate alerts
groups:
  - name: slo-alerts
    rules:
      # Fast burn: 14.4x error rate over 1 hour
      - alert: HighErrorRateFast
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[1h]))
            /
            sum(rate(http_requests_total[1h]))
          ) > (14.4 * 0.001)  # 14.4x of 0.1% error budget
        for: 2m
        labels:
          severity: critical

      # Slow burn: 3x error rate over 3 days
      - alert: HighErrorRateSlow
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[3d]))
            /
            sum(rate(http_requests_total[3d]))
          ) > (3 * 0.001)
        for: 1h
        labels:
          severity: warning
```

== Network-Specific Observability

*TCP metrics to monitor:*

```cpp
// From /proc/net/tcp or getsockopt(TCP_INFO)
struct tcp_info {
    uint8_t  tcpi_state;
    uint8_t  tcpi_retransmits;      // # of retransmits on timeout
    uint32_t tcpi_rtt;              // Smoothed RTT (μs)
    uint32_t tcpi_rttvar;           // RTT variance (μs)
    uint32_t tcpi_snd_cwnd;         // Congestion window (segments)
    uint32_t tcpi_rcv_rtt;          // Receiver RTT estimate
    uint32_t tcpi_total_retrans;    // Total retransmits
    // ...
};

// Export as Prometheus metrics
void export_tcp_metrics(int fd) {
    struct tcp_info info;
    socklen_t len = sizeof(info);
    getsockopt(fd, IPPROTO_TCP, TCP_INFO, &info, &len);

    tcp_rtt_microseconds.Set(info.tcpi_rtt);
    tcp_retransmits_total.Set(info.tcpi_total_retrans);
    tcp_cwnd_segments.Set(info.tcpi_snd_cwnd);
}
```

*eBPF for network observability:*

```c
// Trace TCP retransmits
SEC("tracepoint/tcp/tcp_retransmit_skb")
int trace_retransmit(struct trace_event_raw_tcp_retransmit_skb *ctx) {
    struct event e = {};
    e.saddr = ctx->saddr;
    e.daddr = ctx->daddr;
    e.sport = ctx->sport;
    e.dport = ctx->dport;
    e.state = ctx->state;

    bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, &e, sizeof(e));
    return 0;
}
```

*Key network metrics:*
- Connection establishment rate / failures
- RTT distribution (P50, P99)
- Retransmission rate
- TCP state distribution
- Bytes in/out per service pair
- DNS lookup latency

== References

*Primary sources:*

W3C Trace Context. W3C Recommendation (2021). https://www.w3.org/TR/trace-context/

OpenTelemetry Specification. CNCF (2024). https://opentelemetry.io/docs/specs/

Sigelman, B.H. et al. (2010). "Dapper, a Large-Scale Distributed Systems Tracing Infrastructure." Google Technical Report.

Shkuro, Y. (2019). Mastering Distributed Tracing. Packt Publishing.

Sridharan, C. (2018). Distributed Systems Observability. O'Reilly Media.

Beyer, B. et al. (2016). Site Reliability Engineering. O'Reilly Media. Chapter 6: Monitoring Distributed Systems.

Gregg, B. (2020). Systems Performance: Enterprise and the Cloud, 2nd Edition. Addison-Wesley.

Prometheus Documentation. https://prometheus.io/docs/

Grafana Loki Documentation. https://grafana.com/docs/loki/

Jaeger Documentation. https://www.jaegertracing.io/docs/

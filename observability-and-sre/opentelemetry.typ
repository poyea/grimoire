= OpenTelemetry

OpenTelemetry (OTel) is the vendor-neutral standard for generating, processing, and exporting telemetry. Formed in 2019 from the merger of OpenTracing (2016, API-only) and OpenCensus (2018, Google's SDK-included approach), it became a CNCF incubating project and graduated in 2024; by repository activity it has been the second-most-active CNCF project after Kubernetes for years. The bet OTel makes is that instrumentation should be a write-once commodity decoupled from the backend that stores and queries the data. This chapter covers the signals model, the API/SDK split, OTLP, semantic conventions, context propagation, the Collector, sampling configuration, and instrumentation strategy.

*See also:* _Distributed Tracing_, _Metrics Systems_, _The Three Pillars and Beyond_, _Continuous Profiling_

== The Signals Model

OTel defines telemetry as *signals* sharing common infrastructure (context, resources, exporters):

- *Traces:* spans with trace/span IDs, attributes, events, links, and status (the Dapper model — see _Distributed Tracing_).
- *Metrics:* instruments (`Counter`, `UpDownCounter`, `Histogram`, `Gauge`, and observable/asynchronous variants) that produce sums, gauges, histograms, and *exponential histograms* (base-2 exponential bucket boundaries with a scale parameter, giving constant relative error and mergeability without pre-agreed bucket layouts).
- *Logs:* unlike traces and metrics, OTel did not define a new logging API for existing languages; it defines a log data model and *bridges* from established libraries (Logback, log4j, the Python `logging` module), enriching records with trace and span IDs for correlation.
- *Profiles:* the fourth signal, in development since the 2023 acquisition-by-donation of Elastic's profiling agent; OTLP profiles reached experimental status in 2024 (see _Continuous Profiling_).
- *Baggage:* not telemetry itself but a propagated key-value channel that travels with context.

Every signal hangs off two shared concepts. A *resource* is an immutable set of attributes describing the entity producing telemetry (`service.name`, `service.version`, `k8s.pod.name`, `cloud.region`) — attached once per SDK, not per event. *Context* is the in-process and cross-process carrier of the active span and baggage.

== API versus SDK

OTel's most consequential design decision is the strict separation of API and SDK:

- The *API* is a minimal, dependency-light facade: `get_tracer`, `start_span`, `counter.add`. Libraries (HTTP frameworks, database drivers) depend _only_ on the API. If no SDK is installed, every call is a no-op with near-zero cost.
- The *SDK* is the implementation the application owner wires up at startup: span processors, samplers, metric readers, exporters, resource detection.

The point is ecosystem economics: a library author can ship instrumentation without forcing an exporter, a vendor, or even any runtime cost on users — which is why instrumentation could spread through frameworks (Spring, ASP.NET Core, Express) rather than remaining an application afterthought. Stability guarantees differ accordingly: the tracing and metrics APIs are frozen at 1.0 with strong backward-compatibility promises, while SDK internals and the Collector evolve faster.

== OTLP

The OpenTelemetry Protocol is the native wire format: Protobuf-encoded payloads over *gRPC* (port 4317) or *HTTP/protobuf* (port 4318, with an HTTP/JSON variant). One protocol carries all signals, with request/response semantics that include partial-success reporting and retryable-versus-fatal status codes, plus backpressure via gRPC flow control. The payload hierarchy is `Resource` → `Scope` (the instrumentation library) → records (spans/metric points/log records), so resource attributes are encoded once rather than per span. OTLP reached 1.0 stability in 2023 and is now accepted natively by Prometheus (since 2.47/3.0), Grafana Tempo/Loki/Mimir, Jaeger (which deprecated its own ingestion formats in favor of OTLP), Elastic, Datadog, and every major vendor — making it the effective lingua franca that Zipkin's B3 and vendor agents never quite became.

== Semantic Conventions

Telemetry is only cross-tool queryable if everyone names things identically; *semantic conventions* are the agreed attribute vocabulary: `http.request.method`, `http.response.status_code`, `db.system`, `messaging.operation`, `rpc.system`, `network.peer.address`, plus resource conventions (`service.name`, `deployment.environment`). Conventions are stabilized domain by domain — HTTP reached stable in late 2023 (a breaking rename from `http.method`-era names, managed via the `OTEL_SEMCONV_STABILITY_OPT_IN` migration flag), databases in 2024, with messaging and GenAI following. The conventions are what allow a backend to compute RED metrics or build a service map from any compliant instrumentation without per-framework parsers; treat custom attributes as a namespaced extension (`myapp.tenant_id`), never as replacements for conventional names.

== Context Propagation

Cross-process correlation rides on *propagators* that inject and extract context into carrier headers. The default composite is:

- *W3C TraceContext* — the `traceparent` header (`version-traceid-parentid-flags`) and `tracestate` for vendor data, standardized as a W3C Recommendation in 2020 largely so that proxies, CDNs, and competing vendors would propagate each other's context (see _Distributed Tracing_ for the wire format).
- *W3C Baggage* — a separate `baggage` header carrying application key-values (`tenant=acme,deployment=canary`) through the entire downstream call tree. Baggage is propagated but not recorded; if you want it on spans, a processor must copy it explicitly. Use it sparingly: every entry rides on every outbound request, and untrusted edges should strip it (it is attacker-controllable input).

B3 (Zipkin) and Jaeger propagators remain available for brownfield interop; the in-process side is a `Context` abstraction (thread-local, async-local, or explicitly passed depending on language) whose correct flow across thread pools and async boundaries is historically the largest source of broken traces.

== The Collector

The Collector is a standalone Go service that receives, processes, and exports telemetry — the deployment keystone of most OTel architectures. Its pipeline model:

- *Receivers* ingest: OTLP, but also Prometheus scrape, Jaeger, Zipkin, Kafka, filelog, StatsD — about 90 receivers in the contrib distribution, which is how the Collector doubles as a migration vehicle from legacy agents.
- *Processors* transform in order: `batch` (always), `memory_limiter` (first, to shed load before OOM), `attributes`/`transform` (rename, redact PII via the OTTL transformation language), `filter`, `tail_sampling`, `k8sattributes` (enrich with pod metadata).
- *Exporters* emit: OTLP to a backend or another Collector tier, `prometheusremotewrite`, vendor exporters, with per-exporter retry and persistent queues.
- *Connectors* join pipelines, e.g., the `spanmetrics` connector that derives RED metrics from the span stream.

Two canonical topologies: an *agent* per host or sidecar (low-latency local collection, resource enrichment, immediate offload from the app) forwarding to a *gateway* tier (central policy: tail sampling, redaction, egress credentials, per-backend routing). Tail sampling forces the gateway tier to be sharded by `trace_id` (a load-balancing exporter exists for exactly this) so all spans of a trace reach the same instance. The Collector is also where vendor neutrality is realized operationally: switching backends is an exporter config change, with no application redeploy.

== Sampling Configuration

OTel supports sampling at three places, with different trade-offs (the strategy taxonomy is covered in _Distributed Tracing_):

1. *SDK (head) sampling* via `OTEL_TRACES_SAMPLER`: `parentbased_always_on` (default), `parentbased_traceidratio` with `OTEL_TRACES_SAMPLER_ARG=0.1`, or always-off. `parentbased_*` honors the upstream decision from `traceparent` flags, which is mandatory for consistent trees; the ratio sampler hashes the trace ID so independent services make the same decision for the same trace.
2. *Collector tail sampling* via the `tail_sampling` processor: buffer spans for a `decision_wait`, then apply policies (status-code, latency, attribute, probabilistic, rate-limiting, and composite combinations) — keep all errors and slow traces, 1 % of the rest.
3. *Jaeger-style remote sampling:* the SDK fetches per-operation strategies from a Collector endpoint, allowing central retuning without redeploys.

A subtlety the ecosystem is still resolving: sampled-out traces silently bias trace-derived metrics, so either derive metrics before sampling (spanmetrics at the agent tier) or record sampling probability for reweighting (the experimental consistent-probability sampling spec, which encodes the sampling threshold in `tracestate`).

== Instrumentation Strategy: Auto versus Manual

*Zero-code (auto) instrumentation* attaches without source changes: the Java agent (`-javaagent`) bytecode-weaves hooks into hundreds of libraries; Python/Node/.NET have equivalent agents or monkey-patching bootstrap commands; the OTel *Operator* for Kubernetes injects all of this via pod annotation. Go, lacking a runtime to hook, relies on wrapper libraries and an experimental eBPF-based agent. Auto-instrumentation yields the standard spans (HTTP server/client, DB calls, queue produce/consume) and is the right first step — it gets a service onto the trace graph in an afternoon.

*Manual instrumentation* adds what no agent can know: business-meaningful spans (`checkout.reserve_inventory`), domain attributes (`order.value`, `customer.tier`), events, and correct context propagation across hand-rolled async boundaries. The pragmatic strategy is layered: auto-instrument everything, then manually enrich the 20 % of code paths where the questions actually get asked, following semantic conventions for anything that has one. The main operational caveats of agents are startup cost (JVM agent weaving), version-skew sensitivity against instrumented libraries, and the temptation to never graduate to the manual layer that carries most diagnostic value.

== Ecosystem Status

As of the mid-2020s the stability picture is heterogeneous by design — each language SIG and signal stabilizes independently:

- *Stable:* tracing API/SDK and OTLP in all major languages; metrics API/SDK in Java, .NET, Python, Go, JS; logs bridges in Java and .NET, with others close behind; HTTP and database semantic conventions.
- *Maturing:* the Collector (1.0 of core components arriving piecemeal), logs SDKs in remaining languages, client-side/browser and mobile instrumentation, the file-based declarative configuration format.
- *Experimental:* profiles, the events API, entity model revisions to resources, consistent-probability sampling.

Vendor support is effectively universal — Datadog, New Relic, Splunk, Dynatrace, Honeycomb, Grafana, and the cloud providers accept OTLP, and several have replaced their proprietary agents with OTel distributions. The honest caveats: per-language maturity still varies (JS logs lagged Java by years), semantic-convention migrations have been genuinely painful for early adopters, and the project's breadth means the documentation often trails the code. None of these change the strategic picture: instrumenting with anything proprietary in a greenfield system is, at this point, the position that requires justification.

== Further Reading

OpenTelemetry Community. "OpenTelemetry Specification." https://opentelemetry.io/docs/specs/otel/

W3C. (2020). "Trace Context." W3C Recommendation. https://www.w3.org/TR/trace-context/

Sigelman, B. H. et al. (2010). "Dapper, a Large-Scale Distributed Systems Tracing Infrastructure." Google Technical Report. The conceptual ancestor of the OTel trace model.

Young, T., Parker, A. (2024). _Learning OpenTelemetry._ O'Reilly. Written by two OTel co-founders; the best single treatment of Collector topology and rollout strategy.

Blanco, D. G. (2023). _Practical OpenTelemetry._ Apress. SDK-level detail across languages.

CNCF. (2024). "OpenTelemetry Graduation Announcement." https://www.cncf.io/

Boten, A. (2022). _Cloud-Native Observability with OpenTelemetry._ Packt.

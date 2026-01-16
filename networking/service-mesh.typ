= Service Mesh

A service mesh is a dedicated infrastructure layer for handling service-to-service communication in microservices architectures, providing observability, traffic management, and security without application code changes.

*See also:* Application Protocols (for HTTP/gRPC), Concurrency Models (for proxy threading), Message Queues (for alternative communication patterns)

== Sidecar Proxy Pattern

*Core concept:* Deploy a proxy alongside each service instance, intercepting all network traffic.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Pod / Container                                                     │
│  ┌─────────────────┐    ┌─────────────────┐                         │
│  │   Application   │───▶│  Sidecar Proxy  │───▶ Network             │
│  │   (your code)   │◀───│  (Envoy/Linkerd)│◀─── Network             │
│  └─────────────────┘    └─────────────────┘                         │
│        :8080                  :15001                                 │
└─────────────────────────────────────────────────────────────────────┘
```

*Traffic interception (Kubernetes):*
- iptables rules redirect all inbound/outbound traffic to sidecar
- Application connects to localhost:8080 (thinks it's direct)
- Sidecar intercepts, applies policies, forwards to actual destination

```bash
# Istio's init container sets up iptables:
iptables -t nat -A PREROUTING -p tcp -j REDIRECT --to-port 15001
iptables -t nat -A OUTPUT -p tcp -j REDIRECT --to-port 15001
```

*Advantages:*
- Language-agnostic: Works with any application (Java, Go, Python, etc.)
- No code changes: mTLS, retries, tracing "for free"
- Consistent policies: Uniform security/traffic rules across fleet

*Disadvantages:*
- Latency overhead: +1-5ms per hop (two proxy traversals)
- Resource consumption: Each sidecar uses 50-200MB RAM, 0.1-0.5 CPU
- Complexity: Additional failure modes, debugging difficulty

== Data Plane vs Control Plane

```
┌─────────────────────────────────────────────────────────────────────┐
│  Control Plane (Istiod / Linkerd Control Plane)                     │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐           │
│  │  Pilot        │  │  Citadel      │  │  Galley       │           │
│  │  (config/     │  │  (certificate │  │  (config      │           │
│  │   discovery)  │  │   authority)  │  │   validation) │           │
│  └───────┬───────┘  └───────┬───────┘  └───────────────┘           │
│          │                  │                                       │
│          ▼                  ▼                                       │
│    xDS API (gRPC)     Certificate Push                              │
└─────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Data Plane (Envoy Proxies)                                         │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐                │
│  │ Envoy A │  │ Envoy B │  │ Envoy C │  │ Envoy D │                │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘                │
│       │            │            │            │                      │
│       ▼            ▼            ▼            ▼                      │
│  [Service A]  [Service B]  [Service C]  [Service D]                │
└─────────────────────────────────────────────────────────────────────┘
```

*Data Plane responsibilities:*
- Request routing, load balancing, health checking
- mTLS termination/origination
- Metrics collection, access logging
- Circuit breaking, retries, timeouts

*Control Plane responsibilities:*
- Service discovery (which pods implement which services)
- Configuration distribution (routing rules, policies)
- Certificate management (issue, rotate, revoke)
- Policy enforcement coordination

*xDS Protocol:* Envoy's discovery APIs for dynamic configuration [Envoy Project 2023]:
- LDS (Listener Discovery): Which ports to listen on
- RDS (Route Discovery): How to route requests
- CDS (Cluster Discovery): Backend service endpoints
- EDS (Endpoint Discovery): Individual pod IPs
- SDS (Secret Discovery): TLS certificates

== mTLS and Zero-Trust Networking

*Traditional perimeter security:* Trust everything inside firewall.

*Zero-trust model:* Verify every request, regardless of network location.

```
┌──────────────┐              ┌──────────────┐
│  Service A   │              │  Service B   │
│  ┌────────┐  │   mTLS       │  ┌────────┐  │
│  │ Envoy  │──┼──────────────┼──│ Envoy  │  │
│  └────────┘  │  encrypted   │  └────────┘  │
└──────────────┘  + verified  └──────────────┘
```

*mTLS (mutual TLS) provides:*
1. *Encryption:* All traffic encrypted (AES-256-GCM)
2. *Authentication:* Both sides present certificates (SPIFFE identity)
3. *Integrity:* Tampering detected via MAC

*Certificate lifecycle (Istio):*
```
1. Envoy requests certificate from Istiod (SDS API)
2. Istiod verifies workload identity (Kubernetes service account)
3. Istiod signs certificate with mesh CA (default: 24h validity)
4. Envoy uses certificate for mTLS connections
5. Auto-rotation before expiry
```

*SPIFFE identity format:*
```
spiffe://cluster.local/ns/default/sa/my-service-account
       └─trust domain─┘   └namespace┘  └service account─┘
```

*Authorization policies:*
```yaml
# Istio AuthorizationPolicy
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: allow-frontend
spec:
  selector:
    matchLabels:
      app: backend
  rules:
  - from:
    - source:
        principals: ["cluster.local/ns/default/sa/frontend"]
    to:
    - operation:
        methods: ["GET", "POST"]
        paths: ["/api/*"]
```

*Performance impact:* TLS handshake adds 1-2ms (amortized with connection pooling). Encryption overhead: ~5% CPU increase [Cloudflare 2019].

== Traffic Management

=== Load Balancing

*Algorithms available in Envoy:*
- Round Robin (default): Sequential distribution
- Least Connections: Route to least-loaded backend
- Random: Uniform random selection
- Ring Hash: Consistent hashing for session affinity
- Maglev: Google's consistent hashing algorithm [Eisenbud et al. 2016]

```yaml
# Istio DestinationRule
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: backend-lb
spec:
  host: backend.default.svc.cluster.local
  trafficPolicy:
    loadBalancer:
      simple: LEAST_CONN
```

=== Retries and Timeouts

```yaml
# Istio VirtualService
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: backend-retry
spec:
  hosts:
  - backend
  http:
  - route:
    - destination:
        host: backend
    retries:
      attempts: 3
      perTryTimeout: 2s
      retryOn: 5xx,reset,connect-failure
    timeout: 10s
```

*Retry conditions (Envoy):*
- `5xx`: HTTP 500-599 responses
- `reset`: Connection reset
- `connect-failure`: Connection failed
- `retriable-4xx`: HTTP 409 (conflict)
- `gateway-error`: HTTP 502, 503, 504

*Exponential backoff:* Base interval (25ms) with jitter, capped at max (250ms).

=== Circuit Breaking

*Prevent cascade failures by stopping requests to unhealthy services.*

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: backend-circuit-breaker
spec:
  host: backend
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
```

*Outlier detection:* Eject unhealthy endpoints from load balancing pool.

=== Traffic Splitting (Canary Deployments)

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: canary-release
spec:
  hosts:
  - myservice
  http:
  - route:
    - destination:
        host: myservice
        subset: v1
      weight: 90
    - destination:
        host: myservice
        subset: v2
      weight: 10
```

== Observability

=== Distributed Tracing

*Automatic span injection:* Envoy generates trace spans for each request.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Trace: 4bf92f3577b34da6                                            │
│  ├── Span: frontend (12ms)                                          │
│  │   └── Span: api-gateway (8ms)                                    │
│  │       ├── Span: user-service (3ms)                               │
│  │       └── Span: order-service (4ms)                              │
│  │           └── Span: inventory-db (2ms)                           │
└─────────────────────────────────────────────────────────────────────┘
```

*Required header propagation (application must forward):*
- `x-request-id`: Unique request identifier
- `x-b3-traceid`: Trace ID (128-bit)
- `x-b3-spanid`: Current span ID
- `x-b3-parentspanid`: Parent span ID
- `x-b3-sampled`: Sampling decision

*Trace backends:* Jaeger, Zipkin, Datadog, AWS X-Ray (via OpenTelemetry).

=== Metrics (Prometheus)

*Standard metrics exported by Envoy:*

```
# Request count
istio_requests_total{
  source_workload="frontend",
  destination_workload="backend",
  response_code="200"
} 12345

# Request duration histogram
istio_request_duration_milliseconds_bucket{
  source_workload="frontend",
  destination_workload="backend",
  le="100"
} 11000

# TCP bytes transferred
istio_tcp_sent_bytes_total{...} 1234567890
```

*Golden signals for SRE:*
1. *Latency:* Request duration (p50, p90, p99)
2. *Traffic:* Requests per second
3. *Errors:* Error rate (5xx / total)
4. *Saturation:* Connection pool utilization

=== Access Logging

```json
{
  "authority": "backend:8080",
  "bytes_received": 256,
  "bytes_sent": 1024,
  "downstream_local_address": "10.0.0.5:8080",
  "downstream_remote_address": "10.0.0.3:45678",
  "duration": 15,
  "method": "POST",
  "path": "/api/orders",
  "protocol": "HTTP/2",
  "response_code": 200,
  "response_flags": "-",
  "upstream_cluster": "outbound|8080||backend.default.svc.cluster.local",
  "x_request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

== Envoy Proxy Architecture

*Envoy = high-performance L4/L7 proxy:* C++, non-blocking, single-process multi-threaded [Lyft Engineering 2017].

```
┌─────────────────────────────────────────────────────────────────────┐
│  Envoy Process                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │  Main Thread                                                     ││
│  │  - xDS communication (gRPC to control plane)                     ││
│  │  - Configuration updates                                         ││
│  │  - Stats aggregation                                             ││
│  └─────────────────────────────────────────────────────────────────┘│
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │
│  │ Worker 0  │ │ Worker 1  │ │ Worker 2  │ │ Worker 3  │           │
│  │ (epoll)   │ │ (epoll)   │ │ (epoll)   │ │ (epoll)   │           │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │
│       │             │             │             │                   │
│       ▼             ▼             ▼             ▼                   │
│  Connection   Connection   Connection   Connection                  │
│  handling     handling     handling     handling                    │
└─────────────────────────────────────────────────────────────────────┘
```

*Key design choices:*
- *Worker threads:* One per CPU core, each with own epoll loop
- *Zero shared state:* Workers don't share connections (thread-local)
- *Non-blocking:* All I/O via libevent (epoll on Linux)
- *Hot restart:* New process drains old connections gracefully

*Filter chains:*
```
Listener → Network Filters → HTTP Connection Manager → HTTP Filters → Router
           (TCP proxy,        (codec, stats)           (auth, rate   (upstream
            TLS termination)                            limit, lua)   selection)
```

== Performance Overhead

*Latency impact (measured per hop):*

#table(
  columns: 4,
  align: (left, right, right, left),
  table.header([Scenario], [P50], [P99], [Notes]),
  [Direct (no mesh)], [baseline], [baseline], [Application-to-application],
  [Istio (HTTP)], [+2-3ms], [+5-8ms], [Two proxy hops + mTLS],
  [Istio (gRPC)], [+1-2ms], [+3-5ms], [HTTP/2 multiplexing helps],
  [Linkerd], [+1-1.5ms], [+2-4ms], [Rust-based, lighter weight],
  [Cilium (eBPF)], [+0.2-0.5ms], [+1-2ms], [Kernel-level, no sidecar],
)

*Benchmarks [Kinvolk 2021, Solo.io 2022]:*
- Istio 1.12+: ~3ms P50 latency overhead per hop
- Linkerd 2.11+: ~1.5ms P50 latency overhead per hop
- Throughput reduction: 10-30% compared to direct

*Resource consumption per sidecar:*
- Memory: 50-150MB (Envoy), 20-50MB (Linkerd proxy)
- CPU: 0.1-0.5 cores under load
- At 1000 pods: 50-150GB total memory for sidecars alone

*Optimization strategies:*
1. *Connection pooling:* Reuse connections (avoid TLS handshake per request)
2. *Protocol selection:* HTTP/2 or gRPC reduces overhead vs HTTP/1.1
3. *Sidecar tuning:* Adjust `concurrency`, disable unused features
4. *eBPF acceleration:* Cilium's socket-level proxying (kernel bypass)

== When to Use Service Mesh

*Use service mesh when:*
- Microservices count > 10-20 services
- Need consistent mTLS across all services
- Require traffic management (canary, A/B testing)
- Want unified observability without code changes
- Operating in zero-trust network environment

*Avoid service mesh when:*
- Monolithic application or few services
- Latency-critical path (sub-millisecond requirements)
- Resource-constrained environment
- Team lacks Kubernetes/networking expertise
- Simple traffic patterns (no canary/splitting needs)

*Alternatives:*
- *Library-based:* gRPC interceptors, Spring Cloud (code changes required)
- *Ingress-only:* API gateway at edge (Envoy, Kong, NGINX)
- *eBPF mesh:* Cilium (lower overhead, fewer features)

*Decision matrix:*

#table(
  columns: 4,
  align: (left, left, left, left),
  table.header([Requirement], [Service Mesh], [Library], [Gateway-only]),
  [mTLS everywhere], [Yes], [Manual], [Edge only],
  [Observability], [Automatic], [Code changes], [Edge only],
  [Traffic splitting], [Yes], [Code changes], [Edge only],
  [Latency overhead], [2-5ms/hop], [~0], [1-2ms edge],
  [Language support], [Any], [Per-language], [Any],
  [Complexity], [High], [Medium], [Low],
)

== References

Envoy Project (2023). Envoy Proxy Documentation. https://www.envoyproxy.io/docs/

Istio Authors (2023). Istio Documentation. https://istio.io/latest/docs/

Morgan, W. & Buoyant Inc. (2023). Linkerd Documentation. https://linkerd.io/docs/

Eisenbud, D.E., et al. (2016). "Maglev: A Fast and Reliable Software Network Load Balancer." NSDI '16.

Kinvolk (2021). "Service Mesh Benchmark 2021." https://kinvolk.io/blog/

Burns, B., Grant, B., Oppenheimer, D., Brewer, E., & Wilkes, J. (2016). "Borg, Omega, and Kubernetes." ACM Queue.

NIST (2020). "Zero Trust Architecture." Special Publication 800-207.

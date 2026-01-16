= Service Discovery

Service discovery enables services to locate and communicate with each other in distributed systems without hardcoded addresses.

*See also:* Application Protocols (for DNS fundamentals), Message Queues (for distributed communication patterns), Concurrency Models (for handling discovery events)

== Problem Statement

*Challenge:* In dynamic environments (cloud, containers, autoscaling), service instances:
- Start/stop unpredictably
- Have ephemeral IP addresses
- Scale horizontally
- Fail and recover

*Static configuration fails:* Hardcoded IPs or hostnames require manual updates and downtime.

*Solution:* Service discovery - automatic registration, health checking, and lookup.

== Discovery Patterns

*Two fundamental approaches:*

```
┌─────────────────────────────────────────────────────────────────┐
│  CLIENT-SIDE DISCOVERY                                          │
│                                                                 │
│  ┌────────┐    ┌──────────────┐    ┌─────────────────────────┐  │
│  │ Client │───▶│   Registry   │    │  Service A (10.0.0.1)   │  │
│  │        │◀───│              │    │  Service A (10.0.0.2)   │  │
│  │        │────────────────────────▶  Service A (10.0.0.3)   │  │
│  └────────┘    └──────────────┘    └─────────────────────────┘  │
│                                                                 │
│  Client queries registry, chooses instance, connects directly   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  SERVER-SIDE DISCOVERY                                          │
│                                                                 │
│  ┌────────┐    ┌──────────────┐    ┌─────────────────────────┐  │
│  │ Client │───▶│ Load Balancer│───▶│  Service A (10.0.0.1)   │  │
│  │        │    │   /Router    │    │  Service A (10.0.0.2)   │  │
│  │        │    │              │    │  Service A (10.0.0.3)   │  │
│  └────────┘    └──────────────┘    └─────────────────────────┘  │
│                      │                                          │
│                      ▼                                          │
│               ┌──────────────┐                                  │
│               │   Registry   │                                  │
│               └──────────────┘                                  │
│                                                                 │
│  Client uses fixed endpoint; router queries registry, forwards  │
└─────────────────────────────────────────────────────────────────┘
```

*Client-side discovery:*
- *Advantages:* Lower latency (direct connection), fewer hops, client can implement custom load balancing
- *Disadvantages:* Client complexity, language-specific libraries, harder to update

*Server-side discovery:*
- *Advantages:* Simple clients, centralized policy, language-agnostic
- *Disadvantages:* Additional hop latency, load balancer becomes bottleneck

== DNS-Based Discovery

*Traditional approach using DNS [RFC 1035, RFC 2782].*

*A/AAAA records:* Simple name to IP mapping.

```
api.example.com.    60    IN    A    10.0.0.1
api.example.com.    60    IN    A    10.0.0.2
api.example.com.    60    IN    A    10.0.0.3
```

*SRV records [RFC 2782]:* Service location with port and priority.

```
_http._tcp.api.example.com.  60  IN  SRV  10 5 8080 api1.example.com.
_http._tcp.api.example.com.  60  IN  SRV  10 5 8080 api2.example.com.
_http._tcp.api.example.com.  60  IN  SRV  20 0 8080 api3.example.com.
```

*SRV record format:* `priority weight port target`
- Lower priority = preferred
- Weight = load distribution within same priority

*DNS-SD (DNS Service Discovery) [RFC 6763]:*

```
; Service type enumeration
_services._dns-sd._udp.local.  PTR  _http._tcp.local.
_services._dns-sd._udp.local.  PTR  _grpc._tcp.local.

; Service instances
_http._tcp.local.  PTR  api-v1._http._tcp.local.
_http._tcp.local.  PTR  api-v2._http._tcp.local.

; Instance details
api-v1._http._tcp.local.  SRV  0 0 8080 10.0.0.1.
api-v1._http._tcp.local.  TXT  "version=1.0" "env=prod"
```

*DNS limitations:*
- *TTL caching:* Stale entries persist until TTL expires (30s-300s typical)
- *No health checking:* Dead instances remain in DNS until manually removed
- *UDP packet size:* Limited to ~512 bytes without EDNS0 (truncates large responses)

== Consul

*HashiCorp Consul: Service mesh with discovery, health checks, KV store.*

*Architecture:*

```
┌─────────────────────────────────────────────────────────────────┐
│  Datacenter                                                     │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │ Consul      │    │ Consul      │    │ Consul      │          │
│  │ Server      │◀──▶│ Server      │◀──▶│ Server      │          │
│  │ (Leader)    │    │ (Follower)  │    │ (Follower)  │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│        ▲                  ▲                  ▲                  │
│        │    Gossip (Serf) │                  │                  │
│        ▼                  ▼                  ▼                  │
│  ┌───────────┐      ┌───────────┐      ┌───────────┐            │
│  │ Consul    │      │ Consul    │      │ Consul    │            │
│  │ Agent     │      │ Agent     │      │ Agent     │            │
│  │ (Client)  │      │ (Client)  │      │ (Client)  │            │
│  ├───────────┤      ├───────────┤      ├───────────┤            │
│  │ Service A │      │ Service B │      │ Service C │            │
│  └───────────┘      └───────────┘      └───────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

*Service registration:*

```json
{
  "service": {
    "name": "api",
    "port": 8080,
    "tags": ["v1", "primary"],
    "check": {
      "http": "http://localhost:8080/health",
      "interval": "10s",
      "timeout": "2s"
    }
  }
}
```

*Health checks:*
- HTTP: GET endpoint, expect 2xx
- TCP: Connection succeeds
- Script: Exit code 0 = healthy, 1 = warning, other = critical
- TTL: Service must heartbeat within interval

*Service query (HTTP API):*

```bash
# Get healthy instances of "api" service
curl http://localhost:8500/v1/health/service/api?passing=true

# Response
[{
  "Service": {
    "ID": "api-1",
    "Address": "10.0.0.1",
    "Port": 8080,
    "Tags": ["v1"]
  }
}]
```

*DNS interface:*

```bash
# Consul provides DNS on port 8600
dig @127.0.0.1 -p 8600 api.service.consul SRV

# Returns only healthy instances
api.service.consul.  0  IN  SRV  1 1 8080 api-1.node.dc1.consul.
```

*Key-Value store:*

```bash
# Store configuration
consul kv put config/api/rate_limit 1000

# Watch for changes
consul watch -type=key -key=config/api/rate_limit handler.sh
```

*Performance:* Gossip protocol: O(log N) convergence. Typical: 100-500ms for membership changes.

== etcd

*CoreOS etcd: Distributed key-value store using Raft consensus [Ongaro & Ousterhout 2014].*

*Architecture:*

```
┌─────────────────────────────────────────────────────────────────┐
│                         etcd Cluster                            │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   etcd      │    │   etcd      │    │   etcd      │          │
│  │   Node 1    │◀──▶│   Node 2    │◀──▶│   Node 3    │          │
│  │  (Leader)   │    │ (Follower)  │    │ (Follower)  │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│        ▲                                                        │
│        │  gRPC                                                  │
│        ▼                                                        │
│  ┌─────────────────────────────────────────────────────┐        │
│  │                    Applications                      │        │
│  │  (Register services, watch for changes, read/write) │        │
│  └─────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

*Service registration pattern:*

```go
// Register service with lease (TTL)
lease, _ := client.Grant(ctx, 30)  // 30 second TTL

client.Put(ctx, "/services/api/10.0.0.1:8080",
    `{"host":"10.0.0.1","port":8080}`,
    clientv3.WithLease(lease.ID))

// Keep-alive: Refresh lease periodically
ch, _ := client.KeepAlive(ctx, lease.ID)
```

*Service discovery with watch:*

```go
// Get all instances
resp, _ := client.Get(ctx, "/services/api/", clientv3.WithPrefix())

for _, kv := range resp.Kvs {
    fmt.Printf("Instance: %s\n", kv.Value)
}

// Watch for changes (real-time updates)
watchCh := client.Watch(ctx, "/services/api/", clientv3.WithPrefix())

for watchResp := range watchCh {
    for _, event := range watchResp.Events {
        switch event.Type {
        case mvccpb.PUT:
            fmt.Printf("New/updated: %s\n", event.Kv.Value)
        case mvccpb.DELETE:
            fmt.Printf("Removed: %s\n", event.Kv.Key)
        }
    }
}
```

*Consistency:* Strong consistency (linearizable reads). Tradeoff: Higher latency than eventually consistent systems.

== ZooKeeper

*Apache ZooKeeper: Coordination service for distributed systems.*

*ZNode structure for services:*

```
/services
  /api
    /instance-0000000001  →  {"host":"10.0.0.1","port":8080}
    /instance-0000000002  →  {"host":"10.0.0.2","port":8080}
  /database
    /primary              →  {"host":"10.0.1.1","port":5432}
```

*Ephemeral nodes:* Automatically deleted when session ends (service crash = auto-deregister).

*Sequential nodes:* Append unique monotonic ID (useful for leader election).

*Watch mechanism:* One-time notification on change (must re-register after trigger).

== Kubernetes Service Discovery

*Built-in discovery via Services and DNS.*

```
┌─────────────────────────────────────────────────────────────────┐
│  Kubernetes Cluster                                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  kube-dns / CoreDNS                                 │        │
│  │  (Watches API server, serves DNS)                   │        │
│  └─────────────────────────────────────────────────────┘        │
│        ▲                          │                             │
│        │                          ▼                             │
│  ┌───────────┐    DNS query    ┌────────────────────────┐       │
│  │    Pod    │────────────────▶│  api.default.svc.cluster.local │
│  │ (Client)  │◀────────────────│  → 10.96.0.100 (ClusterIP)    │
│  └───────────┘                 └────────────────────────┘       │
│        │                                                        │
│        │ Connection to ClusterIP                                │
│        ▼                                                        │
│  ┌───────────────────────────────────────────────────┐          │
│  │  kube-proxy (iptables/IPVS)                       │          │
│  │  Load balances to Pod IPs                         │          │
│  └───────────────────────────────────────────────────┘          │
│        │                                                        │
│        ├──────────────┬──────────────┐                          │
│        ▼              ▼              ▼                          │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                    │
│  │  Pod      │  │  Pod      │  │  Pod      │                    │
│  │ (api)     │  │ (api)     │  │ (api)     │                    │
│  │ 10.0.0.5  │  │ 10.0.0.6  │  │ 10.0.0.7  │                    │
│  └───────────┘  └───────────┘  └───────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

*Service types:*
- *ClusterIP:* Internal virtual IP, load balanced to pods
- *NodePort:* Expose on each node's IP at static port
- *LoadBalancer:* Cloud provider load balancer (external)
- *Headless (ClusterIP: None):* Returns pod IPs directly (for client-side load balancing)

*DNS naming:*
```
<service>.<namespace>.svc.cluster.local
api.default.svc.cluster.local      →  10.96.0.100
api.production.svc.cluster.local   →  10.96.0.200
```

*Headless service query:*
```bash
# Returns all pod IPs (not ClusterIP)
dig api.default.svc.cluster.local A

api.default.svc.cluster.local.  30  IN  A  10.0.0.5
api.default.svc.cluster.local.  30  IN  A  10.0.0.6
api.default.svc.cluster.local.  30  IN  A  10.0.0.7
```

== Caching and Consistency

*Tradeoff:* Freshness vs performance.

*Caching strategies:*

#table(
  columns: 4,
  align: (left, left, left, left),
  table.header([Strategy], [Freshness], [Latency], [Use Case]),
  [No cache], [Real-time], [High (network RTT)], [Critical routing],
  [TTL-based], [Seconds-stale], [Low (local)], [General services],
  [Watch-based], [Near real-time], [Low (push updates)], [Dynamic environments],
  [Hybrid], [Configurable], [Low], [Production systems],
)

*TTL tuning:*
- Short TTL (5-30s): Fast failover, higher registry load
- Long TTL (60-300s): Lower load, slower failover

*Watch-based caching:* Subscribe to registry changes, update local cache on notification. Best of both worlds: low latency + near real-time updates.

== Failure Modes and Resilience

*Registry failure:*
- *Cached data:* Continue using last-known-good endpoints
- *Graceful degradation:* Route to fewer instances vs complete failure
- *Registry clustering:* 3-5 node quorum (Consul, etcd, ZooKeeper)

*Network partition:*
- Split-brain: Different clients see different service lists
- CAP theorem: Choose consistency (block) or availability (serve stale)
- Consul: CP system (prefers consistency)
- Eureka: AP system (prefers availability)

*Service instance failure:*
- Health check interval determines detection time
- Typical: 10-30s detection, 3 failed checks before removal
- Fast failover: Shorter intervals, more registry load

*Client resilience patterns:*
```
1. Retry with backoff: Exponential backoff on connection failure
2. Circuit breaker: Stop calling failing service after threshold
3. Fallback: Use cached response or default value
4. Load balancing: Round-robin, least-connections, weighted
```

*Thundering herd prevention:*
- Jittered retry: Add randomness to retry delays
- Rate limiting: Limit discovery queries per client
- Connection pooling: Reuse existing connections

== Comparison

#table(
  columns: 5,
  align: (left, left, left, left, left),
  table.header([System], [Consistency], [Health Checks], [DNS Interface], [Use Case]),
  [Consul], [CP (Raft)], [Built-in], [Yes], [Service mesh],
  [etcd], [CP (Raft)], [TTL leases], [No], [Kubernetes, config],
  [ZooKeeper], [CP (ZAB)], [Ephemeral nodes], [No], [Hadoop, Kafka],
  [Eureka], [AP], [Heartbeat], [No], [Spring Cloud],
  [Kubernetes], [Eventual], [Probes], [Yes (CoreDNS)], [Container orchestration],
)

*Decision guide:*
- *Kubernetes environment:* Use built-in Services (simplest)
- *Multi-datacenter:* Consul (WAN federation)
- *Already using etcd:* Leverage for service discovery
- *Spring ecosystem:* Eureka (native integration)

== References

Fielding, R. & Taylor, R. (2000). "Architectural Styles and the Design of Network-based Software Architectures." Doctoral dissertation, UC Irvine.

Ongaro, D. & Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm." USENIX ATC '14.

RFC 2782: A DNS RR for specifying the location of services (DNS SRV). Gulbrandsen, A., Vixie, P., & Esibov, L. (2000).

RFC 6763: DNS-Based Service Discovery. Cheshire, S. & Krochmal, M. (2013).

HashiCorp (2023). Consul Architecture. https://www.consul.io/docs/architecture

Burns, B. (2018). Designing Distributed Systems. O'Reilly Media.

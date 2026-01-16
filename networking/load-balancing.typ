= Load Balancing

Load balancers distribute traffic across multiple servers to improve availability, throughput, and response time. Critical infrastructure component for high-scale systems.

*See also:* Transport Layer (for TCP/UDP), Application Protocols (for HTTP/2, HTTP/3), Kernel Bypass (for high-performance packet processing)

== L4 vs L7 Load Balancing

*Layer 4 (Transport):* Operates on TCP/UDP. Routes based on IP addresses and ports. No payload inspection.

*Layer 7 (Application):* Operates on HTTP/gRPC/etc. Routes based on headers, URLs, cookies, content.

```
                L4 Load Balancer                      L7 Load Balancer

   Client                Server                Client                Server
     │                     │                    │                     │
     │   TCP SYN          │                    │   TCP + TLS         │
     ├────────────────────►│                   ├────────────────────►│
     │                     │  (Connection      │                     │
     │   Packets forwarded │   forwarded       │   HTTP parsed       │  (Full proxy,
     │   at IP/port level) │   directly)       │   Content-aware     │   terminates
     │                     │                    │   routing)          │   connection)
     │◄────────────────────┤                   │◄────────────────────┤
```

*L4 characteristics:*
- Latency: 10-50μs added (just IP/port rewrite)
- Throughput: 10-100M PPS (kernel bypass: 200M+)
- CPU cost: Minimal (no payload parsing)
- Connection: Forwarded or NAT'd, client sees server or LB IP
- Use cases: TCP services, databases, high-throughput streaming

*L7 characteristics:*
- Latency: 100μs-1ms added (TLS termination, HTTP parsing)
- Throughput: 100K-1M requests/sec per core
- CPU cost: Significant (TLS, compression, parsing)
- Connection: Terminated at LB, new connection to backend
- Use cases: HTTP APIs, microservices, content routing

*Decision matrix:*

#table(
  columns: 4,
  align: (left, center, center, center),
  table.header([Requirement], [L4], [L7], [Notes]),
  [URL-based routing], [-], [+], [L7 required],
  [SSL/TLS termination], [-], [+], [Offload crypto from backends],
  [HTTP header manipulation], [-], [+], [Add X-Forwarded-For, etc.],
  [WebSocket support], [+], [+], [L4 simpler for long-lived],
  [Maximum throughput], [+], [-], [L4 10-100x faster],
  [gRPC/HTTP2 multiplexing], [-], [+], [L7 understands streams],
  [TCP services (DB, Redis)], [+], [-], [L4 protocol-agnostic],
)

== Load Balancing Algorithms

=== Round Robin

*Simplest algorithm:* Rotate through backends sequentially.

```cpp
class RoundRobin {
    std::vector<Backend> backends;
    std::atomic<uint64_t> counter{0};

public:
    Backend* select() {
        uint64_t idx = counter.fetch_add(1, std::memory_order_relaxed);
        return &backends[idx % backends.size()];
    }
};
```

*Pros:* O(1), no state, fair distribution
*Cons:* Ignores server capacity, connection count, response time

*Weighted Round Robin:* Assign weights based on capacity.

```
Server A (weight 3): ███
Server B (weight 1): █
Sequence: A, A, A, B, A, A, A, B, ...
```

=== Least Connections

*Route to server with fewest active connections.*

```cpp
class LeastConnections {
    struct Backend {
        std::string addr;
        std::atomic<int> active_conns{0};
    };
    std::vector<Backend> backends;

public:
    Backend* select() {
        Backend* best = &backends[0];
        int min_conns = best->active_conns.load(std::memory_order_relaxed);

        for (auto& b : backends) {
            int conns = b.active_conns.load(std::memory_order_relaxed);
            if (conns < min_conns) {
                min_conns = conns;
                best = &b;
            }
        }
        best->active_conns.fetch_add(1, std::memory_order_relaxed);
        return best;
    }
};
```

*Pros:* Adapts to slow backends, handles heterogeneous load
*Cons:* O(n) selection, requires connection tracking

*Weighted Least Connections:* `score = active_conns / weight`

=== Consistent Hashing

*Problem:* Standard hash `server = hash(key) % N` redistributes all keys when N changes.

*Solution:* Hash ring with virtual nodes [Karger et al. 1997].

```
                Hash Ring (0 to 2^32-1)
                         0
                         │
                    ┌────┴────┐
                   A1        B1     ← Virtual nodes
                  /            \
                 /              \
            ┌───┘                └───┐
           A2                        B2
            │                        │
           key1 → A                 key2 → B
            │                        │
           A3                        C1
            └───┐                ┌───┘
                 \              /
                  \            /
                   C2        C3
                    └────┬────┘
                         │
                       2^31
```

```cpp
class ConsistentHash {
    std::map<uint32_t, Backend*> ring;  // Sorted by hash
    static constexpr int VNODES = 150;  // Virtual nodes per backend

public:
    void add_backend(Backend* b) {
        for (int i = 0; i < VNODES; i++) {
            std::string vnode = b->addr + "#" + std::to_string(i);
            uint32_t h = hash(vnode);  // e.g., xxHash, MurmurHash
            ring[h] = b;
        }
    }

    void remove_backend(Backend* b) {
        for (int i = 0; i < VNODES; i++) {
            std::string vnode = b->addr + "#" + std::to_string(i);
            ring.erase(hash(vnode));
        }
    }

    Backend* select(const std::string& key) {
        if (ring.empty()) return nullptr;
        uint32_t h = hash(key);
        auto it = ring.lower_bound(h);
        if (it == ring.end()) it = ring.begin();  // Wrap around
        return it->second;
    }
};
```

*Key properties:*
- Adding/removing server affects only K/N keys (K = keys, N = servers)
- Virtual nodes ensure uniform distribution (150-200 recommended)
- Lookup: O(log N) with balanced tree

*Variants:*
- *Ketama:* MD5-based, used by memcached
- *Jump Hash:* O(1) memory, O(ln N) time, but no weighted support
- *Maglev:* Google's LB, consistent + even distribution [Eisenbud et al. 2016]

== Health Checks and Failover

*Active health checks:* LB periodically probes backends.

```
                     Load Balancer
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │Server A │      │Server B │      │Server C │
    │  (OK)   │      │ (FAIL)  │      │  (OK)   │
    └─────────┘      └─────────┘      └─────────┘
         │                │                │
      HTTP 200        Timeout           HTTP 200
         │                │                │
         └────────┬───────┴───────┬────────┘
                  │               │
              In pool         Removed
```

*Health check types:*

#table(
  columns: 3,
  align: (left, left, left),
  table.header([Type], [Method], [Use Case]),
  [TCP], [SYN → SYN-ACK], [Basic connectivity],
  [HTTP], [GET /health → 200], [Application liveness],
  [gRPC], [grpc.health.v1], [gRPC services],
  [Script], [Custom command], [Complex validation],
)

*Timing parameters:*
- *Interval:* 5-30s typical (balance between detection speed and load)
- *Timeout:* 2-5s (must be < interval)
- *Threshold:* 2-3 failures before marking down (avoid flapping)
- *Rise:* 2-3 successes to mark healthy again

*Passive health checks:* Monitor real traffic for failures.

```cpp
struct BackendHealth {
    std::atomic<int> consecutive_failures{0};
    std::atomic<int> consecutive_successes{0};
    std::atomic<bool> healthy{true};

    void record_result(bool success) {
        if (success) {
            consecutive_failures.store(0, std::memory_order_relaxed);
            if (consecutive_successes.fetch_add(1) >= RISE_THRESHOLD) {
                healthy.store(true, std::memory_order_release);
            }
        } else {
            consecutive_successes.store(0, std::memory_order_relaxed);
            if (consecutive_failures.fetch_add(1) >= FALL_THRESHOLD) {
                healthy.store(false, std::memory_order_release);
            }
        }
    }
};
```

*Failover strategies:*
- *Immediate:* Route to backup on first failure (fast, may cause flapping)
- *Threshold:* N failures in M seconds (stable, slower detection)
- *Circuit breaker:* Open after failures, half-open to test recovery

== Session Persistence (Sticky Sessions)

*Problem:* Stateful backends require same client → same server.

*Solution:* Bind client to specific backend for session duration.

=== Persistence Methods

*1. Source IP hash:*
```cpp
Backend* select(const IPAddr& client_ip) {
    uint32_t h = hash(client_ip);
    return &backends[h % backends.size()];
}
```
*Limitation:* NAT/proxy causes many clients to share IP.

*2. Cookie-based (L7):*
```http
HTTP/1.1 200 OK
Set-Cookie: SERVERID=backend2; Path=/; HttpOnly

# Subsequent requests:
Cookie: SERVERID=backend2
```
*Best for HTTP:* Works through NAT, survives backend restarts.

*3. Session table:*
```cpp
std::unordered_map<SessionID, Backend*> session_table;

Backend* select(const SessionID& sid) {
    auto it = session_table.find(sid);
    if (it != session_table.end() && it->second->healthy)
        return it->second;
    Backend* b = load_balance_select();  // Fallback
    session_table[sid] = b;
    return b;
}
```
*Tradeoff:* Memory usage grows with sessions, needs cleanup.

*Persistence vs availability:* If sticky backend fails:
- *Strict:* Return error (data integrity)
- *Fallback:* Route to another (availability, may lose session)

== DSR (Direct Server Return)

*High-throughput optimization:* Response bypasses load balancer.

```
Normal Flow (Proxy Mode):
Client ──► LB ──► Server
Client ◄── LB ◄── Server      (LB in return path, bottleneck)

DSR Flow:
Client ──► LB ──► Server
Client ◄──────── Server       (Direct return, LB only sees requests)
```

*Traffic ratio:* Typical web: 1:10 request:response ratio. DSR removes 90% of LB traffic.

*Implementation:*
```
1. Client sends to VIP (Virtual IP) owned by LB
2. LB forwards packet to server (preserves original dest IP = VIP)
3. Server has VIP configured on loopback (accepts packet)
4. Server responds directly to client source IP
5. Client receives response from VIP (matches its request)
```

*Linux server configuration:*
```bash
# On backend servers - configure VIP on loopback
ip addr add 10.0.0.100/32 dev lo

# Disable ARP for VIP (LB should answer ARP, not backends)
echo 1 > /proc/sys/net/ipv4/conf/lo/arp_ignore
echo 2 > /proc/sys/net/ipv4/conf/lo/arp_announce
```

*DSR constraints:*
- L4 only (cannot modify HTTP headers)
- Same L2 network or IP tunneling required
- No connection tracking at LB
- Health checks must be separate mechanism

*Performance:* 10-50x throughput improvement for response-heavy workloads.

== HAProxy Configuration Patterns

*HAProxy:* Industry-standard L4/L7 load balancer [Tarreau, 2001-present].

*Basic L7 configuration:*
```
global
    maxconn 50000
    log /dev/log local0
    stats socket /var/run/haproxy.sock mode 660

defaults
    mode http
    timeout connect 5s
    timeout client  30s
    timeout server  30s
    option httplog
    option dontlognull

frontend http_front
    bind *:80
    bind *:443 ssl crt /etc/ssl/certs/server.pem

    # Route based on URL
    acl is_api path_beg /api/
    use_backend api_servers if is_api
    default_backend web_servers

backend web_servers
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200

    server web1 10.0.1.1:8080 check weight 100
    server web2 10.0.1.2:8080 check weight 100
    server web3 10.0.1.3:8080 check weight 50 backup

backend api_servers
    balance leastconn
    option httpchk GET /api/health

    cookie SERVERID insert indirect nocache
    server api1 10.0.2.1:8080 check cookie api1
    server api2 10.0.2.2:8080 check cookie api2
```

*L4 (TCP) mode:*
```
frontend tcp_front
    mode tcp
    bind *:3306
    default_backend mysql_servers

backend mysql_servers
    mode tcp
    balance source                    # IP hash for session persistence
    option tcp-check

    server mysql1 10.0.3.1:3306 check
    server mysql2 10.0.3.2:3306 check backup
```

== NGINX Configuration Patterns

*NGINX:* High-performance web server and reverse proxy [Sysoev, 2004-present].

```nginx
upstream backend_pool {
    # Algorithm selection
    least_conn;                      # Or: ip_hash, hash $request_uri consistent;

    # Health checks (NGINX Plus / OSS with module)
    zone backend_zone 64k;

    server 10.0.1.1:8080 weight=5 max_fails=3 fail_timeout=30s;
    server 10.0.1.2:8080 weight=3 max_fails=3 fail_timeout=30s;
    server 10.0.1.3:8080 backup;
}

server {
    listen 80;
    listen 443 ssl http2;

    ssl_certificate     /etc/ssl/server.crt;
    ssl_certificate_key /etc/ssl/server.key;

    location / {
        proxy_pass http://backend_pool;

        # Headers for backend
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout    30s;
        proxy_read_timeout    30s;

        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    # Sticky sessions via cookie
    location /app/ {
        proxy_pass http://backend_pool;

        sticky cookie srv_id expires=1h domain=.example.com path=/;
    }
}

# TCP/UDP load balancing (stream module)
stream {
    upstream mysql_backend {
        server 10.0.2.1:3306 weight=5;
        server 10.0.2.2:3306;
    }

    server {
        listen 3306;
        proxy_pass mysql_backend;
        proxy_connect_timeout 1s;
    }
}
```

== Performance Characteristics

*Benchmarks (typical modern hardware, 2023):*

#table(
  columns: 5,
  align: (left, right, right, right, left),
  table.header([Load Balancer], [L4 PPS], [L7 RPS], [Latency], [Notes]),
  [HAProxy], [1-2M], [200K-500K], [50-200μs], [Single-threaded, nbproc for scaling],
  [NGINX], [1-2M], [100K-300K], [100-300μs], [Worker processes],
  [Envoy], [500K-1M], [50K-200K], [200-500μs], [Full observability],
  [Linux IPVS], [5-10M], [N/A (L4)], [10-30μs], [Kernel-level, DSR],
  [Katran (XDP)], [10M+], [N/A (L4)], [\<10μs], [Facebook, kernel bypass],
  [Maglev], [10M+], [N/A (L4)], [\<10μs], [Google, consistent hash],
)

*Scaling patterns:*

```
Single LB (vertical scaling):
┌──────────────────────────────────────┐
│        Load Balancer (1M RPS)        │
└────────────────────┬─────────────────┘
         ┌───────────┼───────────┐
         ▼           ▼           ▼
      Server 1   Server 2   Server 3

Multiple LBs (horizontal scaling):
         ┌─────────────────────────────┐
         │    DNS / ECMP / Anycast     │
         └──────────────┬──────────────┘
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
      ┌─────┐        ┌─────┐        ┌─────┐
      │ LB1 │        │ LB2 │        │ LB3 │
      └──┬──┘        └──┬──┘        └──┬──┘
         └───────┬──────┴──────┬───────┘
                 ▼             ▼
              Servers       Servers
```

*Horizontal scaling methods:*
- *DNS round robin:* Simple, slow failover (TTL-bound)
- *ECMP (Equal-Cost Multi-Path):* Router distributes across LBs
- *Anycast:* Same IP advertised from multiple locations, BGP routes to nearest
- *GLB (Global Load Balancer):* L4 distribution to regional L7 LBs

*Capacity planning:*
- L7 LB: Budget 1 core per 50K-100K HTTP RPS
- L4 LB: Budget 1 core per 1-2M PPS
- Memory: ~1KB per active connection (L7), ~100B per flow (L4)
- Network: LB bandwidth >= sum of backend bandwidth (unless DSR)

== References

Karger, D., Lehman, E., Leighton, T., Panigrahy, R., Levine, M., & Lewin, D. (1997). "Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web." STOC '97.

Eisenbud, D.E., et al. (2016). "Maglev: A Fast and Reliable Software Network Load Balancer." NSDI '16.

Tarreau, W. (2023). HAProxy Configuration Manual. https://www.haproxy.org/

Sysoev, I. (2023). NGINX Documentation. https://nginx.org/en/docs/

Schroeder, B., et al. (2006). "Web Servers and Traffic: Impact of Workload on Performance." USENIX ATC '06.

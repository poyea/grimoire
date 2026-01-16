= Connection Pooling

Connection pooling amortizes the cost of establishing connections across multiple requests, eliminating per-request handshake overhead.

*See also:* Transport Layer (TCP handshake costs), Application Protocols (HTTP/2 multiplexing), Sockets API (connection lifecycle)

== Why Pooling Matters

*Connection establishment costs:*

#table(
  columns: 3,
  align: (left, right, left),
  table.header([Phase], [Latency], [Notes]),
  [TCP handshake], [1 RTT (20-100ms)], [SYN → SYN-ACK → ACK],
  [TLS 1.3 handshake], [1 RTT (20-100ms)], [ClientHello → ServerHello + encrypted],
  [TLS 1.2 handshake], [2 RTT (40-200ms)], [Additional round-trip for key exchange],
  [DNS resolution], [10-50ms], [If not cached],
  [Database auth], [5-20ms], [Password verification, session setup],
)

*Total new connection cost:* 50-350ms depending on distance and protocol.

*Per-request overhead without pooling:*
```
Request 1: [TCP+TLS: 150ms] + [Query: 5ms] = 155ms
Request 2: [TCP+TLS: 150ms] + [Query: 5ms] = 155ms
...
1000 requests: 1000 × 155ms = 155 seconds
```

*With connection pooling:*
```
Connection setup: 150ms (once)
Request 1-1000: 1000 × 5ms = 5 seconds
Total: 5.15 seconds (30× faster)
```

*CPU cost:* TLS handshake consumes 2-10ms CPU time per connection (RSA/ECDHE key exchange). At 1000 connections/second, handshakes alone consume 2-10 CPU cores.

== Pool Sizing Strategies

=== Little's Law

*Fundamental queueing theory [Little 1961]:*

$ L = lambda times W $

Where:
- $L$ = average number of items in system (pool size needed)
- $lambda$ = arrival rate (requests per second)
- $W$ = average time in system (request duration)

*Example:* 100 requests/second, 50ms average latency:

$ L = 100 times 0.05"s" = 5 "connections" $

*With safety margin (2×):*

$ "Pool Size" = 2 times lambda times W $

=== HikariCP Formula

*Database-specific sizing [Brettwooldridge 2014]:*

$ "Pool Size" = T_n times (C_m - 1) + 1 $

Where:
- $T_n$ = number of threads requiring connections
- $C_m$ = number of simultaneous connections per thread

*Simplified rule:* For CPU-bound workloads:

$ "Pool Size" = "CPU cores" times 2 + 1 $

*Rationale:* Beyond $2 times "cores"$, context switching overhead exceeds parallelism benefit.

*PostgreSQL recommendation [wiki.postgresql.org]:*
- Small app: 10-20 connections
- Medium app: 20-50 connections
- Large app: 50-100 connections (rarely more)

*Anti-pattern:* Setting pool size = max_connections. Exhausts database resources, causes connection refusal.

=== Dynamic Sizing

*Adaptive algorithms:*

```cpp
// Simple adaptive pool sizing
struct AdaptivePool {
    int min_size;           // Never shrink below
    int max_size;           // Never grow beyond
    int current_size;
    double target_utilization;  // e.g., 0.7 (70%)

    void adjust() {
        double utilization = (double)active / current_size;

        if (utilization > 0.9 && current_size < max_size) {
            grow(current_size * 1.5);  // Grow 50%
        } else if (utilization < 0.3 && current_size > min_size) {
            shrink(current_size * 0.75);  // Shrink 25%
        }
    }
};
```

*Hysteresis:* Use different thresholds for grow (90%) vs shrink (30%) to prevent oscillation.

== Idle Connection Management

=== Keep-Alive Configuration

*HTTP Keep-Alive [RFC 7230]:*
```http
Connection: keep-alive
Keep-Alive: timeout=60, max=1000
```

*TCP Keep-Alive [RFC 1122]:*
```cpp
int optval = 1;
setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &optval, sizeof(optval));

// Linux-specific tuning
int idle = 60;      // Start probes after 60s idle
int interval = 10;  // 10s between probes
int count = 5;      // Close after 5 failed probes

setsockopt(sock, IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
setsockopt(sock, IPPROTO_TCP, TCP_KEEPINTVL, &interval, sizeof(interval));
setsockopt(sock, IPPROTO_TCP, TCP_KEEPCNT, &count, sizeof(count));
```

*Total timeout:* $60 + 10 times 5 = 110$ seconds before dead connection detected.

=== Idle Timeout Strategies

*Pool-level timeouts:*

#table(
  columns: 3,
  align: (left, right, left),
  table.header([Setting], [Typical Value], [Purpose]),
  [idleTimeout], [10 min], [Close unused connections to free resources],
  [maxLifetime], [30 min], [Prevent stale connections, force refresh],
  [connectionTimeout], [30 sec], [Max wait for available connection],
  [validationTimeout], [5 sec], [Max time for health check query],
)

*Trade-off:* Short idle timeout saves resources but increases connection churn. Long timeout wastes resources but reduces latency spikes.

*Recommendation:* Set `idleTimeout` slightly less than server-side timeout to avoid half-open connections.

== Connection Validation and Health Checks

=== Validation Strategies

*1. Test-on-borrow (synchronous):*
```cpp
Connection* get_connection() {
    Connection* conn = pool.take();
    if (!conn->validate()) {
        conn->close();
        conn = create_new();
    }
    return conn;
}
```

*Overhead:* 1-5ms per request (validation query). Suitable for low-throughput applications.

*2. Test-on-return:*
```cpp
void return_connection(Connection* conn) {
    if (conn->validate()) {
        pool.return(conn);
    } else {
        conn->close();  // Discard bad connection
    }
}
```

*Advantage:* No latency on borrow path. *Disadvantage:* Validation happens after idle period.

*3. Background validation (asynchronous):*
```cpp
void background_validator() {
    while (running) {
        for (Connection* conn : pool.idle_connections()) {
            if (!conn->validate()) {
                pool.remove(conn);
                conn->close();
            }
        }
        sleep(30);  // Validate every 30 seconds
    }
}
```

*Best practice:* Combine background validation with maxLifetime to catch both dead and stale connections.

=== Validation Queries

*Database-specific:*
```sql
-- PostgreSQL
SELECT 1

-- MySQL
SELECT 1  -- or: /* ping */ (driver-level)

-- Oracle
SELECT 1 FROM DUAL

-- SQL Server
SELECT 1
```

*JDBC optimization:* Use `Connection.isValid(timeout)` instead of test query (uses protocol-level ping).

*Network-level:* TCP keepalive probes detect OS-level connection death but not application-level issues (e.g., database restart with same IP).

== HTTP/1.1 vs HTTP/2 Pooling Differences

=== HTTP/1.1 Connection Pooling

*Characteristics:*
- One request per connection at a time (head-of-line blocking)
- Pipelining rarely used (server must respond in order)
- Pool size = max concurrent requests

*Browser defaults [Chromium]:*
```
Max connections per host: 6
Max total connections: 256
```

*Server-side pooling (reverse proxy):*
```nginx
upstream backend {
    server 10.0.0.1:8080;
    keepalive 32;  # Pool size per worker
}

server {
    location / {
        proxy_http_version 1.1;
        proxy_set_header Connection "";  # Enable keepalive
        proxy_pass http://backend;
    }
}
```

=== HTTP/2 Multiplexing

*Key difference:* Single connection handles unlimited concurrent streams.

```
HTTP/1.1: 100 concurrent requests → 100 connections
HTTP/2:   100 concurrent requests → 1 connection, 100 streams
```

*Pooling implications:*
- Pool size typically 1 per origin (single multiplexed connection)
- Connection is reused until max streams exhausted or connection dies
- *Max concurrent streams:* 100-256 typical (negotiated via SETTINGS frame)

*When to use multiple HTTP/2 connections:*
1. Bandwidth exceeds single connection capacity (rare)
2. Connection affinity requirements (sticky sessions)
3. Fault tolerance (connection failure affects all streams)

*gRPC (over HTTP/2):*
```go
// Single connection with multiplexed streams
conn, _ := grpc.Dial("server:443",
    grpc.WithTransportCredentials(creds),
    grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(16*1024*1024)),
)
// All RPCs share this connection
```

== Database Connection Pooling Patterns

=== Connection Pool Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Application                                                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│  │ Thread  │ │ Thread  │ │ Thread  │  ...                   │
│  └────┬────┘ └────┬────┘ └────┬────┘                        │
│       │           │           │                              │
│       └───────────┼───────────┘                              │
│                   ▼                                          │
│           ┌──────────────┐                                   │
│           │ Connection   │ ← Pool manages lifecycle          │
│           │    Pool      │                                   │
│           └──────┬───────┘                                   │
│                  │                                           │
└──────────────────┼───────────────────────────────────────────┘
                   │
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│  Conn   │  │  Conn   │  │  Conn   │  → Physical DB connections
└─────────┘  └─────────┘  └─────────┘
```

=== Common Implementations

*HikariCP (Java) - high-performance:*
```java
HikariConfig config = new HikariConfig();
config.setJdbcUrl("jdbc:postgresql://localhost/db");
config.setMaximumPoolSize(10);
config.setMinimumIdle(5);
config.setIdleTimeout(600000);     // 10 min
config.setMaxLifetime(1800000);    // 30 min
config.setConnectionTimeout(30000); // 30 sec

HikariDataSource ds = new HikariDataSource(config);
```

*pgbouncer (PostgreSQL proxy):*
```ini
[databases]
mydb = host=localhost dbname=mydb

[pgbouncer]
pool_mode = transaction    # Release on transaction end
max_client_conn = 1000     # Clients can connect
default_pool_size = 20     # Connections to PostgreSQL
reserve_pool_size = 5      # Emergency overflow
```

*Pool modes:*
- *Session:* Connection held until client disconnects (like no pooling)
- *Transaction:* Released after each transaction (recommended)
- *Statement:* Released after each statement (breaks multi-statement transactions)

=== Connection Affinity

*Problem:* Some operations require same connection across requests.

*Use cases:*
1. Transactions spanning multiple requests
2. Temporary tables, session variables
3. Prepared statements (in some databases)

*Solution: Thread-local binding:*
```cpp
thread_local Connection* bound_connection = nullptr;

void begin_transaction() {
    bound_connection = pool.acquire();
    bound_connection->execute("BEGIN");
}

void commit() {
    bound_connection->execute("COMMIT");
    pool.release(bound_connection);
    bound_connection = nullptr;
}
```

== Common Pitfalls

=== Connection Leaks

*Symptom:* Pool exhaustion, "unable to acquire connection" errors.

*Cause:* Connections borrowed but never returned.

```cpp
// BUG: Exception causes leak
void process_request() {
    Connection* conn = pool.acquire();
    conn->execute(query);  // Throws exception!
    pool.release(conn);    // Never reached
}

// FIX: RAII pattern
void process_request() {
    auto conn = pool.acquire_scoped();  // Returns on destructor
    conn->execute(query);  // Exception-safe
}
```

*Detection:*
```cpp
struct PooledConnection {
    Connection* conn;
    std::chrono::time_point<std::chrono::steady_clock> acquired_at;
    std::string stack_trace;  // Debug builds
};

void detect_leaks() {
    auto now = std::chrono::steady_clock::now();
    for (auto& pc : active_connections) {
        auto held_time = now - pc.acquired_at;
        if (held_time > std::chrono::minutes(5)) {
            log_warning("Possible leak: connection held for %d min\n%s",
                        held_time.count(), pc.stack_trace.c_str());
        }
    }
}
```

=== Pool Exhaustion

*Symptom:* Request timeouts waiting for connections.

*Causes:*
1. Pool size too small for load
2. Connection leaks
3. Long-running queries blocking pool
4. Database too slow (all connections active)

*Mitigation:*
```cpp
// 1. Timeout on acquire
Connection* conn = pool.acquire(timeout_ms=5000);
if (!conn) {
    throw PoolExhaustedException();
}

// 2. Circuit breaker pattern
if (pool.available() < pool.size() * 0.1) {
    // Less than 10% available
    return fast_fail();  // Don't wait, fail immediately
}

// 3. Queue with priority
Connection* conn = pool.acquire(priority=HIGH);  // Cut the line
```

=== Half-Open Connections

*Symptom:* Connection appears valid but is dead (firewall timeout, server restart).

*Detection:*

```cpp
// Network-level: TCP keepalive (detects after 60-120 seconds)
setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &enabled, sizeof(enabled));

// Application-level: Periodic validation
bool is_connection_alive(Connection* conn) {
    try {
        conn->execute("SELECT 1", timeout_ms=1000);
        return true;
    } catch (...) {
        return false;
    }
}
```

*Firewall considerations:* Many firewalls drop idle connections after 5-30 minutes. Set pool `maxIdleTime` below firewall timeout.

=== Connection Storms

*Symptom:* All connections created simultaneously at startup or after pool drain.

*Problem:* Database overwhelmed with concurrent connection requests.

*Solution: Lazy initialization with rate limiting:*
```cpp
class ThrottledPool {
    std::atomic<int> pending_creates{0};
    static constexpr int MAX_PENDING = 5;

    Connection* acquire() {
        if (auto conn = try_get_idle()) {
            return conn;
        }

        // Rate-limit new connections
        while (pending_creates >= MAX_PENDING) {
            wait(10ms);
        }

        pending_creates++;
        auto conn = create_connection();
        pending_creates--;
        return conn;
    }
};
```

== Performance Metrics

*Key metrics to monitor:*

#table(
  columns: 3,
  align: (left, left, left),
  table.header([Metric], [Healthy Range], [Action if Exceeded]),
  [Pool utilization], [50-80%], [Scale pool or optimize queries],
  [Wait time for connection], [< 10ms], [Increase pool size],
  [Connection creation rate], [< 1/sec], [Check for leaks],
  [Active connections], [< 90% of pool], [Review long-running queries],
  [Connection age], [< maxLifetime], [Verify rotation working],
)

*Latency impact:*
```
Without pooling:  p50 = 180ms, p99 = 450ms (includes handshake)
With pooling:     p50 = 25ms,  p99 = 80ms  (query only)
Improvement:      ~7× reduction in median latency
```

== References

Little, J.D.C. (1961). "A Proof for the Queuing Formula: L = λW." Operations Research 9(3): 383-387.

Brettwooldridge, B. (2014). "HikariCP: About Pool Sizing." github.com/brettwooldridge/HikariCP/wiki/About-Pool-Sizing.

RFC 7230: Hypertext Transfer Protocol (HTTP/1.1): Message Syntax and Routing. Fielding, R. & Reschke, J. (2014).

RFC 7540: Hypertext Transfer Protocol Version 2 (HTTP/2). Belshe, M., Peon, R., & Thomson, M. (2015).

PostgreSQL Wiki. "Number of Database Connections." wiki.postgresql.org/wiki/Number_Of_Database_Connections.

Scherer, W.N., Lea, D., & Scott, M.L. (2006). "Scalable Synchronous Queues." PPoPP '06.

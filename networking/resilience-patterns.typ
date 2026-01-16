= Resilience Patterns

Distributed systems fail in complex ways. Resilience patterns prevent cascading failures and enable graceful degradation under load.

*See also:* Concurrency Models (for thread isolation), Application Protocols (for HTTP retry semantics), Message Queues (for async decoupling)

== Timeouts

*First line of defense:* Prevent indefinite waiting for unresponsive services.

*Timeout types:*
```
┌─────────────────────────────────────────────────────────────┐
│  Connect Timeout    │  Time to establish TCP connection     │
│  (typical: 1-5s)    │  Fails fast if host unreachable       │
├─────────────────────┼───────────────────────────────────────┤
│  Read Timeout       │  Time waiting for response data       │
│  (typical: 5-30s)   │  Guards against slow/hung services    │
├─────────────────────┼───────────────────────────────────────┤
│  Write Timeout      │  Time to send request data            │
│  (typical: 5-10s)   │  Guards against network congestion    │
├─────────────────────┼───────────────────────────────────────┤
│  Total/Request      │  End-to-end operation time            │
│  (typical: 30-60s)  │  Includes retries, redirects          │
└─────────────────────┴───────────────────────────────────────┘
```

*Implementation (POSIX sockets):*
```c
struct timeval timeout;
timeout.tv_sec = 5;
timeout.tv_usec = 0;

// Read timeout
setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

// Write timeout
setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

// Connect timeout (non-blocking approach)
fcntl(sock, F_SETFL, O_NONBLOCK);
connect(sock, addr, len);  // Returns immediately
// Use select/poll/epoll to wait with timeout
```

*Common mistake:* Using same timeout for all operations. Connect timeout should be short (fast failure detection), read timeout longer (allow processing time).

== Retries with Exponential Backoff

*Problem:* Transient failures (network blips, temporary overload) succeed on retry.

*Naive retry:* Immediate retries can overwhelm recovering service (retry storm).

*Solution:* Exponential backoff with jitter.

```
Attempt 1: Immediate
Attempt 2: Wait 100ms + random(0, 50ms)
Attempt 3: Wait 200ms + random(0, 100ms)
Attempt 4: Wait 400ms + random(0, 200ms)
...
Max delay: Cap at 30-60 seconds
```

*Implementation:*
```c
int retry_with_backoff(int (*operation)(void*), void* ctx, int max_retries) {
    int base_delay_ms = 100;
    int max_delay_ms = 30000;

    for (int attempt = 0; attempt < max_retries; attempt++) {
        int result = operation(ctx);
        if (result == SUCCESS) return SUCCESS;

        if (attempt < max_retries - 1) {
            // Exponential backoff: base * 2^attempt
            int delay = base_delay_ms * (1 << attempt);
            if (delay > max_delay_ms) delay = max_delay_ms;

            // Add jitter (0 to 50% of delay)
            int jitter = rand() % (delay / 2);
            delay += jitter;

            usleep(delay * 1000);
        }
    }
    return FAILURE;
}
```

*Jitter strategies:*
- *Full jitter:* `random(0, delay)` - spreads retries uniformly
- *Equal jitter:* `delay/2 + random(0, delay/2)` - bounded spread
- *Decorrelated jitter:* `random(base, prev_delay * 3)` - AWS recommendation

*Idempotency requirement:* Only retry idempotent operations (GET, PUT, DELETE) safely. Non-idempotent operations (POST) may require request IDs for deduplication.

== Circuit Breaker Pattern

*Problem:* Continuing to call failing service wastes resources, delays failure detection, prevents recovery.

*Solution:* Track failures and "trip" circuit after threshold, failing fast without calling downstream.

*State machine:*
```
                    success
         ┌─────────────────────────────┐
         │                             │
         v            failure          │
     ┌────────┐    threshold met    ┌──┴───────┐
     │ CLOSED │ ─────────────────> │   OPEN    │
     │        │                     │           │
     └────────┘                     └─────┬─────┘
         ^                                │
         │       success                  │ timeout
         │    ┌────────────┐              │ expires
         │    │            │              v
         └────┤ HALF-OPEN  │<─────────────┘
              │            │
              └────────────┘
                    │
                    │ failure
                    v
               Back to OPEN
```

*States:*
- *CLOSED:* Normal operation, requests pass through, failures counted
- *OPEN:* Circuit tripped, requests fail immediately without calling service
- *HALF-OPEN:* After timeout, allow single probe request to test recovery

*Implementation:*
```c
typedef enum { CLOSED, OPEN, HALF_OPEN } CircuitState;

typedef struct {
    CircuitState state;
    int failure_count;
    int failure_threshold;     // e.g., 5 failures
    int success_threshold;     // e.g., 3 successes in half-open
    time_t last_failure_time;
    int open_timeout_sec;      // e.g., 30 seconds
    pthread_mutex_t lock;
} CircuitBreaker;

int circuit_breaker_call(CircuitBreaker* cb, int (*fn)(void*), void* ctx) {
    pthread_mutex_lock(&cb->lock);

    // Check if should transition from OPEN to HALF_OPEN
    if (cb->state == OPEN) {
        if (time(NULL) - cb->last_failure_time > cb->open_timeout_sec) {
            cb->state = HALF_OPEN;
            cb->failure_count = 0;
        } else {
            pthread_mutex_unlock(&cb->lock);
            return CIRCUIT_OPEN;  // Fail fast
        }
    }

    pthread_mutex_unlock(&cb->lock);

    // Execute operation
    int result = fn(ctx);

    pthread_mutex_lock(&cb->lock);
    if (result == SUCCESS) {
        if (cb->state == HALF_OPEN) {
            cb->failure_count++;  // Count successes in half-open
            if (cb->failure_count >= cb->success_threshold) {
                cb->state = CLOSED;
                cb->failure_count = 0;
            }
        } else {
            cb->failure_count = 0;  // Reset on success
        }
    } else {
        cb->failure_count++;
        cb->last_failure_time = time(NULL);
        if (cb->failure_count >= cb->failure_threshold) {
            cb->state = OPEN;
        }
    }
    pthread_mutex_unlock(&cb->lock);

    return result;
}
```

*Tuning parameters:*
- *Failure threshold:* 5-10 failures (too low = false positives, too high = slow detection)
- *Open timeout:* 30-60 seconds (allows downstream to recover)
- *Success threshold:* 3-5 (prevents premature closure)

*Metrics to monitor:* Circuit state transitions, failure rate, open duration.

== Bulkhead Pattern

*Problem:* One slow/failing dependency consumes all threads/connections, causing total system failure.

*Solution:* Isolate resources per dependency (like ship bulkheads prevent total flooding).

```
┌─────────────────────────────────────────────────────┐
│                   Application                        │
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ Service A   │  │ Service B   │  │ Service C   │ │
│  │ Thread Pool │  │ Thread Pool │  │ Thread Pool │ │
│  │ (10 threads)│  │ (20 threads)│  │ (5 threads) │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘ │
│         │                │                │         │
└─────────┼────────────────┼────────────────┼─────────┘
          v                v                v
     Service A        Service B        Service C
     (critical)       (normal)         (non-critical)
```

*Implementation strategies:*

*1. Thread pool isolation:*
```c
// Separate thread pool per dependency
ThreadPool* payment_pool = threadpool_create(10);
ThreadPool* inventory_pool = threadpool_create(20);
ThreadPool* analytics_pool = threadpool_create(5);

// Payment service failure doesn't exhaust inventory threads
threadpool_submit(payment_pool, call_payment_service, ctx);
```

*2. Connection pool isolation:*
```c
// Separate connection pools
ConnectionPool* db_pool = pool_create(host_db, 50);
ConnectionPool* cache_pool = pool_create(host_cache, 100);
ConnectionPool* external_pool = pool_create(host_api, 20);
```

*3. Semaphore-based:*
```c
sem_t payment_semaphore;
sem_init(&payment_semaphore, 0, 10);  // Max 10 concurrent calls

int call_with_bulkhead(int (*fn)(void*), void* ctx) {
    if (sem_trywait(&payment_semaphore) != 0) {
        return BULKHEAD_FULL;  // Reject immediately
    }

    int result = fn(ctx);
    sem_post(&payment_semaphore);
    return result;
}
```

*Sizing:* Allocate based on criticality and expected load. Critical paths get more resources.

== Fallbacks and Graceful Degradation

*Principle:* When primary fails, provide degraded but functional response.

*Fallback strategies:*
```
┌────────────────────┬─────────────────────────────────────┐
│  Strategy          │  Example                            │
├────────────────────┼─────────────────────────────────────┤
│  Cached data       │  Return stale product catalog       │
│  Default value     │  Show "N/A" for unavailable price   │
│  Alternative svc   │  Use backup payment provider        │
│  Degraded feature  │  Disable recommendations, show list │
│  Static response   │  Return pre-computed fallback page  │
└────────────────────┴─────────────────────────────────────┘
```

*Implementation:*
```c
Response* get_recommendations(UserId user) {
    // Try primary recommendation service
    Response* resp = call_with_circuit_breaker(
        recommendation_cb, fetch_recommendations, user
    );

    if (resp != NULL) return resp;

    // Fallback 1: Try cache
    resp = cache_get(user_recommendations_key(user));
    if (resp != NULL) return resp;

    // Fallback 2: Return popular items (static)
    return get_popular_items();
}
```

*Key insight:* Fallbacks should be fast and reliable. Cascading fallbacks that also fail create complexity without benefit.

== Rate Limiting and Throttling

*Purpose:* Protect services from overload, ensure fair resource allocation.

*Algorithms:*

*1. Token bucket:*
```c
typedef struct {
    int tokens;
    int max_tokens;
    int refill_rate;      // tokens per second
    time_t last_refill;
    pthread_mutex_t lock;
} TokenBucket;

int token_bucket_acquire(TokenBucket* tb) {
    pthread_mutex_lock(&tb->lock);

    // Refill tokens based on elapsed time
    time_t now = time(NULL);
    int elapsed = now - tb->last_refill;
    tb->tokens += elapsed * tb->refill_rate;
    if (tb->tokens > tb->max_tokens) tb->tokens = tb->max_tokens;
    tb->last_refill = now;

    // Try to acquire
    if (tb->tokens > 0) {
        tb->tokens--;
        pthread_mutex_unlock(&tb->lock);
        return ALLOWED;
    }

    pthread_mutex_unlock(&tb->lock);
    return RATE_LIMITED;
}
```

*2. Sliding window:*
```c
// Track request timestamps, count requests in window
int sliding_window_check(Window* w, int limit, int window_sec) {
    time_t now = time(NULL);
    time_t cutoff = now - window_sec;

    // Remove expired entries
    while (w->head && w->head->timestamp < cutoff) {
        remove_head(w);
    }

    if (w->count >= limit) return RATE_LIMITED;

    append(w, now);
    return ALLOWED;
}
```

*Client-side throttling:* Back off when receiving 429 (Too Many Requests) or 503 (Service Unavailable).

== Implementation Libraries

*Production-ready implementations:*

#table(
  columns: 3,
  align: (left, left, left),
  table.header([Library], [Language], [Features]),
  [resilience4j], [Java], [Circuit breaker, retry, bulkhead, rate limiter],
  [Polly], [.NET], [Retry, circuit breaker, timeout, bulkhead],
  [go-resiliency], [Go], [Circuit breaker, deadline, retrier, semaphore],
  [Hystrix], [Java], [Deprecated, but influential design],
  [Sentinel], [Java], [Flow control, circuit breaking, adaptive],
)

*Netflix Hystrix (historical):* Pioneered many patterns, now deprecated in favor of resilience4j. Key contributions: command pattern, thread pool isolation, request collapsing.

== Combining Patterns

*Typical composition:*
```
Request
   │
   v
┌─────────────────┐
│  Rate Limiter   │──> 429 if exceeded
└────────┬────────┘
         v
┌─────────────────┐
│    Bulkhead     │──> 503 if pool full
└────────┬────────┘
         v
┌─────────────────┐
│ Circuit Breaker │──> Fallback if open
└────────┬────────┘
         v
┌─────────────────┐
│  Retry + Timeout│──> Fallback if exhausted
└────────┬────────┘
         v
    Downstream
      Service
```

*Order matters:* Rate limiter first (cheapest check), then bulkhead (resource protection), then circuit breaker (failure detection), finally retry (recovery attempt).

== Design Considerations

*1. Failure budgets:* Define acceptable failure rates (e.g., 0.1% errors). Configure thresholds accordingly.

*2. Testing:* Use chaos engineering (Netflix Chaos Monkey) to verify resilience under failure conditions.

*3. Observability:* Instrument all patterns with metrics:
- Circuit breaker state changes
- Retry attempt counts
- Bulkhead rejection rates
- Fallback invocation frequency

*4. Async considerations:* Patterns apply equally to async/reactive systems, but implementation differs (reactive streams, CompletableFuture, etc.).

*5. Distributed tracing:* Correlate requests across retries and fallbacks for debugging.

== References

Nygard, M.T. (2018). Release It! Design and Deploy Production-Ready Software. 2nd ed. Pragmatic Bookshelf.

Netflix Technology Blog (2012). "Fault Tolerance in a High Volume, Distributed System." https://netflixtechblog.com/

Brooker, M. (2015). "Exponential Backoff And Jitter." AWS Architecture Blog.

resilience4j Project (2023). resilience4j Documentation. https://resilience4j.readme.io/

Microsoft Azure (2023). "Retry pattern." Cloud Design Patterns.

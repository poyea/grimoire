#import "../template.typ": xref

= Resilience Patterns

Distribution makes partial failure the normal operating condition: dependencies time out, return errors, slow down, or, worst of all, slow down *intermittently*. Resilience is the system's ability to keep delivering acceptable service while parts of it fail, and to recover without human heroics. This chapter covers the core patterns, timeouts, retries, circuit breakers, bulkheads, load shedding, graceful degradation, the arithmetic that motivates them, and chaos engineering as the discipline that verifies they actually work.

*See also:* #xref("software-architecture", "monoliths-and-microservices", label: "Monoliths and Microservices") (why call chains multiply failure), #xref("software-architecture", "event-driven-architecture", label: "Event-Driven Architecture") (asynchronous decoupling as a resilience tool), #xref("software-architecture", "architecture-evaluation", label: "Architecture Evaluation") (availability as a quality attribute with testable scenarios).

== The Arithmetic of Partial Failure

Availability composes multiplicatively along synchronous call chains. If a request fans out through five services, each at 99.9% availability, the best-case composite is $0.999^5 approx 0.995$, half a percent of requests failing, roughly 3.6 hours of effective unavailability per month, from components that each meet a "three nines" $"SLO"$. Tail latency compounds similarly: if each hop's p99 is 100 ms, a request touching five hops sequentially has a far worse end-to-end p99, and a request that fans out to $N$ servers in parallel is as slow as the slowest of the $N$ (Dean & Barroso, "The Tail at Scale", _CACM_ 2013).

Two consequences drive everything in this chapter:
- Depth of synchronous chains is an architectural liability; flatten them or make hops asynchronous.
- Every remote call needs an explicit policy for the three outcomes: success, failure, and *no answer yet*, the third being the dangerous one.

== Timeouts

The timeout is the foundational pattern; every other pattern presumes it. A remote call without a timeout is a latent thread leak: when the dependency hangs, callers accumulate blocked threads until the *caller* dies, which is how a slow downstream becomes a cascading outage upstream.

Practice:
- Set timeouts everywhere: connect, request, and (often forgotten) DNS and connection-pool checkout. Library defaults are frequently infinite or absurd.
- Derive values from the dependency's observed latency distribution (e.g. a small multiple of p99), not from folklore. A timeout far above p99 protects nothing; far below it manufactures failures.
- Propagate *deadlines*, not per-hop timeouts: pass the remaining budget down the chain (gRPC deadline propagation does this natively) so a request with 200 ms left does not start a 2 s downstream call.
- Pair every timeout with a decision: retry, fail, or degrade. A timeout that nobody handles is just a slower error.

== Retries: Useful and Dangerous

Retries convert transient failures (a dropped packet, a deploying instance) into successes, and convert overload into catastrophe. A retry is deliberate load amplification: at three attempts, a struggling dependency receives up to triple traffic precisely when it can least afford it, the classic *retry storm*.

Safe retry policy:
- Retry only *idempotent* operations, or make operations idempotent with idempotency keys (as Stripe's API does: the client supplies a key; the server stores and replays the first outcome).
- Retry only errors that can plausibly succeed on retry (timeouts, 503, connection reset), never deterministic failures (400, 404, business rejections).
- Use *exponential backoff with jitter*. Without jitter, synchronised clients retry in waves; AWS's analysis (Brooker, 2015) shows "full jitter", sleeping a uniform random time up to the backoff cap, smooths load best.
- Cap total attempts and total elapsed time against the request deadline.
- Bound system-wide amplification: retry at *one* layer (avoid retries stacked in the client, the mesh, and the SDK simultaneously), and use a *retry budget* (e.g. retries may be at most 10% of requests, as in Finagle and Envoy) or adaptive/token-bucket retries (AWS SDKs) so retries shut off under broad failure.

== Circuit Breaker

The circuit breaker (Nygard, _Release It!_, 2007) stops calling a dependency that is evidently down, failing fast instead of queueing doomed work. It is a state machine wrapped around a call:

- *Closed*: calls flow; failures are counted (typically a failure-rate threshold over a rolling window, e.g. 50% of the last 20+ calls).
- *Open*: calls fail immediately without touching the dependency, returning an error or fallback. After a cooldown, transition to half-open.
- *Half-open*: a small number of trial calls pass through; success closes the circuit, failure reopens it.

The breaker protects both sides: callers stop burning threads and latency budget on a dead dependency, and the dependency gets breathing room to recover instead of being hammered while restarting. Subtleties: scope breakers per dependency *and* per endpoint where behaviour differs; emit state-change metrics and alerts (a stuck-open breaker is a silent feature outage); choose fallbacks deliberately (cached data, default value, queued-for-later, or honest error). Netflix's Hystrix (2012) popularised the pattern with bulkheaded thread pools; it is now in maintenance, succeeded by Resilience4j on the JVM and by service-mesh implementations (Envoy outlier detection) at the platform layer.

== Bulkheads

Bulkheads partition resources so one failing dependency or tenant cannot exhaust shared capacity, named after a ship's watertight compartments. Forms, from fine to coarse:

- *Per-dependency connection or thread pools*: if calls to the recommendations service hang, they exhaust only their own pool of 20 threads, and checkout's pool is untouched. Semaphore-based isolation is the lightweight variant.
- *Per-class workload pools*: separate interactive traffic from batch/cron traffic at the pool, queue, or instance level so a backfill cannot starve users.
- *Deployment-level cells*: shard the entire stack into independent *cells* (AWS cell-based architecture; Shopify's pods), each serving a subset of customers, so blast radius is one cell. Combine with *shuffle sharding* (AWS, 2014): assign each customer a random small subset of nodes, so two customers rarely share their full set and a poison workload takes down almost nobody else entirely.

Bulkheads cost utilisation, reserved capacity sits idle, which is the explicit price of bounded blast radius.

== Load Shedding and Backpressure

A system pushed past saturation does not degrade linearly; goodput collapses as queues grow and every request times out after consuming resources, *congestion collapse*. The resilient response is to reject excess work early and cheaply:

- *Load shedding*: when a saturation signal trips (queue depth, concurrency, CPU), reject requests at admission with a fast 503, ideally the cheapest possible code path. Prefer shedding by priority: drop prefetches and crawlers before user checkouts.
- *Backpressure*: bounded queues that push the "slow down" signal upstream rather than buffering unboundedly (TCP flow control, Reactive Streams demand signalling, Kafka consumer pull). Unbounded queues convert overload now into memory exhaustion and gigantic latency later, queues should be short.
- *Adaptive concurrency limits*: rather than a static max, infer the capacity from observed latency gradients (Netflix's concurrency-limits library, applying TCP congestion-control ideas like Vegas/Gradient to RPC).
- *Little's law* gives the sizing maths: $L = lambda W$, mean concurrency equals arrival rate times mean latency, so a service handling 1,000 rps at 50 ms holds about 50 in flight; a concurrency cap far above that only adds queueing.

Related failure mode: the *metastable failure* (Bronson et al., HotOS 2021), where a trigger (brief outage) shifts the system into a self-sustaining bad state (retry storms, cold caches) that persists after the trigger ends. Recovery often requires deliberately shedding to below normal load, which is why "turn retries off and shed hard" is a standard incident remediation.

== Graceful Degradation and Fallbacks

Resilient systems offer reduced service instead of no service. Patterns:

- *Static or cached fallbacks*: serve stale recommendations, last-known prices with a banner, or a default configuration. Netflix degrades to non-personalised rows when personalisation is down.
- *Feature shedding*: a kill switch per non-critical feature (reviews, related items) so operators can drop load deliberately during incidents.
- *Queue-and-retry-later*: accept the write (order, email) into a durable queue and process when the dependency recovers, converting a synchronous failure into eventual completion.
- *Fail static*: serve a fully cached/static version of critical pages when the dynamic stack is down.

The hard part is product, not code: deciding *which* failures map to which degraded experience requires explicit tiering of features by criticality, agreed with product owners before the incident.

== Health Checks, Watchdogs, and Recovery

- Distinguish *liveness* (process should be restarted) from *readiness* (process should receive traffic), Kubernetes encodes the distinction. A readiness check should verify the instance can do useful work, but beware deep health checks that fail when a *shared* dependency blips: they can mark every instance unready at once, turning a degradation into a total outage. Prefer shallow checks plus dependency-aware degradation.
- *Crash-only* thinking (Candea & Fox, 2003): design components so that crash-and-restart is a safe, fast, and routine recovery path, no graceful-shutdown-only invariants.
- Watch for *poison* inputs that crash every replica in turn (a malformed message redelivered after each crash); pair restarts with dead-lettering.

== Chaos Engineering

Resilience mechanisms rot silently: the fallback path nobody exercised is broken, the timeout was lost in a refactor. Chaos engineering verifies resilience empirically: form a hypothesis about steady state ("error rate stays below 0.1% if one cache node dies"), inject the failure in a controlled, blast-radius-limited experiment, and compare. Netflix's Chaos Monkey (2011) randomly terminates instances in production; the discipline was formalised in the Principles of Chaos Engineering (2015) and Basiri et al. (_IEEE Software_, 2016). Modern tooling (Gremlin, AWS Fault Injection Service, LitmusChaos) injects latency, packet loss, and dependency errors, not just instance death. Start in staging, automate experiments into pipelines (continuous verification), and treat every game day finding as a defect with an owner.

== Pitfalls

- Timeouts without deadline propagation: downstream work continues after the caller gave up, wasting capacity exactly under load.
- Retries at multiple layers multiplying into $3 times 3 times 3$ amplification.
- Circuit breakers with thresholds so high they never open, or fallbacks that call the same failing dependency.
- Health checks coupled to shared dependencies, amplifying a blip into full unreadiness.
- Unbounded queues "absorbing" load, deferring collapse and destroying latency.
- Resilience theatre: patterns added but never tested with fault injection; the first real test is the outage.
- Treating resilience as a library concern only: the biggest wins are architectural (asynchrony, cells, criticality tiering), not annotations.

== Further Reading

- Nygard, M. (2018). _Release It!_, 2nd ed. Pragmatic Bookshelf.
- Dean, J., & Barroso, L. A. (2013). The tail at scale. _Communications of the ACM_, 56(2).
- Brooker, M. (2015). Exponential backoff and jitter. AWS Architecture Blog.
- Bronson, N., et al. (2021). Metastable failures in distributed systems. _HotOS '21_.
- Basiri, A., et al. (2016). Chaos engineering. _IEEE Software_, 33(3).
- Beyer, B., et al. (2016). _Site Reliability Engineering_. O'Reilly.

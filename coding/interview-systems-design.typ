= Interview Systems Design Playbook

The systems-design round tests your ability to take a vague product brief ("design Twitter") and, in 45 minutes, produce a workable architecture: API surface, data model, scale estimate, storage choice, hot-path optimisations, and failure modes. This chapter is the playbook for the canonical question set: rate limiter, news feed, chat / messaging, search typeahead, URL shortener, and distributed counter. It is the *interview* counterpart to _Advanced Systems_: the same primitives (caches, log-structured storage, sharding, replication) but oriented toward 45-minute communication rather than internals.

*See also:* _Advanced Systems_, _Database_, _Networking_, _Hashing_, _Probabilistic Data Structures_.

== The Framework

A robust answer follows seven beats, in order. Spending one to two minutes each leaves time for deep-dives.

#table(
  columns: (auto, auto),
  [*Step*], [*What you say*],
  [1. Clarify requirements], [Functional / non-functional. Read vs write heavy? Latency SLO? Consistency?],
  [2. Back-of-envelope], [QPS, storage per year, bandwidth, working-set vs cache budget.],
  [3. API], [REST / gRPC endpoints with request / response schemas.],
  [4. Data model], [Tables / documents / key-value. Sharding key. Indexes.],
  [5. High-level architecture], [Box-and-line diagram: clients, LB, app, cache, DB, queue.],
  [6. Deep-dives], [The 1-2 hot subsystems the interviewer cares about.],
  [7. Failure / scale], [Replication, leader election, hot keys, multi-region, observability.],
)

*Numbers you should know cold:* 1 GHz CPU = 1 ns / op; main memory $approx 100$ ns; SSD random read $approx 100$ μs; disk seek $approx 10$ ms; cross-DC RTT $approx 50-150$ ms; 1 Gbps NIC = $approx 100$ MB/s; one machine $approx 10^4$ – $10^5$ QPS for simple HTTP. One day = $approx 86 400$ s; $10^9$ DAU / 86 400 s $approx 12 000$ QPS *average* (peak $5 -10 times$).

== Rate Limiter

*Goal:* permit up to $N$ requests per $T$ seconds per (user, route) — protect downstream services and enforce quotas.

*Algorithms.*

#table(
  columns: (auto, auto, auto),
  [*Algorithm*], [*Behaviour*], [*State / key*],
  [Fixed window counter], [Simple, allows $2N$ burst at boundaries], [1 counter + epoch],
  [Sliding window log], [Exact; $O(N)$ memory per key], [Sorted set of timestamps],
  [Sliding window counter], [Approximation; weighted blend of current + prev window], [2 counters],
  [Token bucket], [Allows bursts up to bucket size; smooth average], [(tokens, last\_refill)],
  [Leaky bucket], [Smooth output rate (queueing semantics)], [Queue + drain rate],
)

*Token bucket* is the default for public APIs (Stripe, AWS, GitHub). Pseudocode (atomic in Redis via a Lua script or Redis Stack `CL.THROTTLE`):

```java
// allowed = true iff the call may proceed; updates state atomically.
public boolean allow(String key, long capacity, double refillPerSec) {
    long now = System.currentTimeMillis();
    State s = store.get(key);                              // atomic read+CAS
    if (s == null) s = new State(capacity, now);
    double tokens = s.tokens + (now - s.last) * refillPerSec / 1000.0;
    tokens = Math.min(capacity, tokens);
    boolean ok = tokens >= 1.0;
    if (ok) tokens -= 1.0;
    store.put(key, new State(tokens, now));
    return ok;
}
```

*Distributed considerations.*
- *Centralised counter* in Redis is the default; latency $approx 0.5$ ms, becomes a hotspot at extreme QPS.
- *Sharded counter* by `hash(key) mod shards` removes hot-key contention.
- *Token-bucket replicas* per app instance with periodic sync trade exactness for $0$-RTT decisions; good for permissive quotas.
- *429 + Retry-After* header; reject before queueing to avoid amplifying back-pressure.

== URL Shortener (TinyURL / bit.ly)

*Requirements.* `POST /shorten {long_url}` $->$ short code (6-7 chars, base62). `GET /{code}` $->$ 301 redirect. Read-heavy ($approx 100:1$). Latency $< 50$ ms p99. Custom aliases optional.

*Estimate.* 500 M new URLs / month $approx 200$ writes / s. $100 times$ reads = 20 K QPS. 5 years storage $approx 30 B$ entries $times 500$ B $approx 15$ TB.

*Encoding.*
- *Counter + base62:* monotonic ID assigned by a sharded ID service (Snowflake, Sonyflake) $->$ encode in base62 (62 chars: $62^7 approx 3.5 dot 10^12$). Predictable but enables scraping; mitigate via per-tenant ID space or short HMAC tag.
- *Random 7-char base62:* $approx 3.5 dot 10^12$ codes; collision probability tiny but must `INSERT ... IF NOT EXISTS`.
- *Hash long URL:* MD5/SHA-1 truncated; allows idempotent dedup of the *same* URL but loses if multiple users want distinct codes.

*Storage.* Key-value: `code -> (long_url, owner_id, created_at, expires_at)`. Cassandra / DynamoDB / Bigtable; partition key = `code` for uniform load.

*Cache.* Edge cache (Fastly / Cloudflare) on `GET /{code}` with `Cache-Control: public, max-age=3600`; Redis LFU between app and DB. Read amplification falls $100times$ to ~200 DB QPS at steady state.

*Analytics.* Click events to Kafka $->$ ClickHouse / BigQuery for dashboards (decoupled from redirect hot path).

*Auth + abuse.* Validate URL with Google Safe Browsing on `POST`; rate-limit per IP; signed expiring short links for private content.

== Distributed Counter

*Problem.* "Likes" on a post viewed by 10 M users. A single row UPDATE is doomed.

*Sharded counter (Cassandra / Spanner / Megastore pattern).* Partition the counter into $k$ buckets per object; writes go to a random bucket; reads sum buckets.

```java
// Increment: writer picks shard at random
void incr(long postId) {
    int shard = ThreadLocalRandom.current().nextInt(SHARDS);
    db.update("UPDATE likes SET c = c + 1 WHERE post_id=? AND shard=?", postId, shard);
}
// Read: sum
long count(long postId) {
    return db.query("SELECT SUM(c) FROM likes WHERE post_id=?", postId).getLong(0);
}
```

*Tune $k$:* larger $k$ spreads writes but increases read fan-out. Typically $k = 16$ for medium, $256$ for celebrities (and *adaptive promotion* on write throughput).

*Eventually-consistent counters.* PN-counters (a CRDT) allow concurrent multi-region increments without coordination; convergent merges replace LWW. Used by Riak.

*Approximate counters.* HyperLogLog gives unique-counts at $2$ KB/object with $\<2$% error; CMS gives heavy hitters. Both shine in stream analytics — see _Probabilistic Data Structures_ and _Streaming Algorithms_.

== News Feed (Facebook / Twitter Home Timeline)

*Two extreme designs:*

#table(
  columns: (auto, auto, auto),
  [*Design*], [*Read path*], [*Write path*],
  [Pull / fan-out on read], [Query all followees, merge top-K], [Cheap insert],
  [Push / fan-out on write], [Read pre-materialised feed list], [Insert into each follower's feed],
  [Hybrid (Twitter)], [Push for normal users; pull on read for celebrities], [Conditional fan-out],
)

*Estimate.* 300 M DAU $times 100$ feed-loads/day = 30 B / day $approx 350$ K QPS. Average user follows 200 accounts, posts 1×/day: a pure push fan-out is $200 dot 300 "M" = 60$ B writes/day — feasible but expensive. Celebrities (50 M followers) make pure push catastrophic; hence the *hybrid*.

*Architecture (push side).*
1. `POST /tweet` $->$ Tweet service writes to *Tweet store* (LSM KV).
2. Emit `tweet_created` event to Kafka.
3. *Fan-out worker* reads followers from graph service and writes `(user_id, tweet_id, ts)` into each follower's *home-timeline cache* (Redis sorted set, capped at 800).
4. `GET /home` reads top-K from Redis; cache miss $->$ rebuild from tweet store + graph service.

*Ranking.* Recent ML reranker (Elasticsearch + lightweight LR/GBDT) over a candidate pool of ~1000.

*Hot path budget.* Feed read p99 must be \<300 ms across geos: Redis (\<2 ms) + ranker (\<50 ms) + hydrate (\<30 ms) + network. Cold path (rebuild) is async and acceptable for \<0.1% of reads.

== Chat / Messaging (WhatsApp / Slack)

*Requirements.* 1-1 + group; delivery $approx <100$ ms; *exactly-once* user-visible delivery; offline + multi-device sync; presence; typing indicators; end-to-end encryption (Signal protocol) optional.

*Transport.* Persistent connection per device: WebSocket or HTTP/2 long-lived; protocol on top: XMPP, MQTT, or a custom binary frame. Each connection is keyed by `(user_id, device_id)`; load-balanced via *consistent hashing* by user so all of a user's devices land on the same edge node (simplifies fan-out).

*Architecture.*
1. *Gateway* (stateful WS server) terminates the connection, authenticates, registers presence.
2. *Message service* receives sends, assigns monotonic `(channel_id, seq)`.
3. *Persisted log* per channel in Cassandra/Kafka — `channel_id` partition key, `seq` clustering. Retrieve via `WHERE channel_id=? AND seq > last_seen`.
4. *Fan-out service* publishes to recipient gateways via a *routing table* (`user_id -> gateway_id`) in Redis / etcd.
5. *Push service* (APNs / FCM) for offline recipients.

*Exactly-once at user layer.* The wire is at-least-once; clients dedup by `(channel_id, seq)`. Acks travel back the same path so the sender can mark *delivered* and *read*.

*Group chat scale.* For large channels (Slack \#general at 10 K members), fan-out at *post time* is wasteful; switch to *pull-on-read* via a per-channel cursor. Subscriptions are managed by *channels* the gateway already maintains.

== Search Typeahead (Autocomplete)

*Goal.* As the user types each character, suggest top-K completions ranked by popularity and personalisation. p99 \<100 ms.

*Index.* A trie keyed by query prefix mapping to top-K completions:

```python
class TrieNode:
    __slots__ = ("children", "top_k")
    def __init__(self):
        self.children = {}                       # char -> TrieNode
        self.top_k = []                          # list[(score, query)]

class Typeahead:
    def __init__(self, k=10): self.root = TrieNode(); self.k = k
    def add(self, query, score):
        node = self.root
        for ch in query:
            node = node.children.setdefault(ch, TrieNode())
            self._merge(node.top_k, (score, query))
    def suggest(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children: return []
            node = node.children[ch]
        return [q for _, q in node.top_k]
    def _merge(self, lst, item):
        # keep top-K by score; tie-break lexicographically
        lst.append(item); lst.sort(reverse=True); del lst[self.k:]
```

*Production reality.* The trie is built *offline* from query logs (last $T$ hours) and *sharded* by prefix to fit per-machine RAM. Live updates flow via a Kafka stream; nodes hot-reload an updated shard every few minutes. Personalisation is added as a *re-rank* step on the candidate list using user features and a learned model.

*Latency budget.* Trie lookup is $O(|p|)$ (3-7 chars), $\<200$ μs. CDN-edge cache on `(prefix, locale)` absorbs most reads.

== Cross-Cutting Concerns

*Consistent hashing.* Distribute keys to a moving set of servers with minimal reshuffling on add/remove. Use *virtual nodes* (~100-500 per physical) to smooth load.

*Quorum reads/writes (Dynamo-style).* $W + R > N$ gives read-your-writes per key; tune $(N, W, R)$ per workload. Cassandra defaults to $N = 3$, `QUORUM = 2`.

*Idempotency.* Every mutating API takes a client-generated `Idempotency-Key` header. Server stores `(key, response)` for 24 h; retries return the cached response. Saves the day during retries, timeouts, and exactly-once illusions.

*Backpressure.* Bound queues, propagate `429`/`503` early. Add *adaptive concurrency* (TCP-Vegas-style, Netflix `concurrency-limits`) to the caller.

*Observability.* Per-request: RED metrics (Rate, Errors, Duration) at every hop; structured logs with `trace_id`; sampled distributed traces (OpenTelemetry). On-call must answer "what is slow / failing right now" in $<1$ minute.

== Interview Anti-Patterns

- Jumping to a diagram before clarifying requirements.
- Picking a database before stating the access pattern.
- Forgetting to estimate QPS / storage before choosing tech.
- Hand-waving "use Kafka" without explaining the consumer semantics.
- Ignoring failure modes (single-region only, no replicas, no idempotency).
- Reaching for blockchain. Don't.

== Reusable Component Cheatsheet

#table(
  columns: (auto, auto),
  [*Need*], [*Default choice*],
  [Edge cache + DDoS], [CloudFront / Fastly / Cloudflare],
  [App-tier load balancer], [Envoy / NGINX / AWS ALB],
  [Hot cache], [Redis (sentinel or cluster mode)],
  [OLTP], [Postgres ($\<$10 TB) / Spanner / Aurora],
  [KV at scale], [DynamoDB / Cassandra / Bigtable],
  [Search], [Elasticsearch / OpenSearch / Vespa],
  [Queue / log], [Kafka / Kinesis / Pulsar],
  [Object storage], [S3 / GCS],
  [OLAP], [BigQuery / Snowflake / ClickHouse],
  [Coordination], [etcd / Zookeeper / Consul],
  [Service mesh], [Istio / Linkerd],
)

== Further Reading

*Kleppmann, M. (2017).* Designing Data-Intensive Applications. O'Reilly. The single best book for this round.

*Bondi, A.B. (2014).* Foundations of Software and System Performance Engineering. Addison-Wesley.

*Burns, B. (2018).* Designing Distributed Systems. O'Reilly.

*Sridharan, C. (2018).* Distributed Systems Observability. O'Reilly.

*Dean, J. & Barroso, L.A. (2013).* The Tail at Scale. CACM 56(2): 74-80.

*DeCandia, G. et al. (2007).* Dynamo: Amazon's Highly Available Key-value Store. SOSP 2007.

*Corbett, J.C. et al. (2013).* Spanner: Google's Globally Distributed Database. ACM TOCS 31(3).

*Kreps, J., Narkhede, N. & Rao, J. (2011).* Kafka: a Distributed Messaging System for Log Processing. NetDB.

*Tang, C. et al. (2015).* Holistic Configuration Management at Facebook. SOSP 2015.

*Xu, A. (2020).* System Design Interview - An Insider's Guide. ByteByteGo. Interview-focused; complements Kleppmann.

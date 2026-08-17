#import "../template.typ": xref

= Causal Consistency

Causal consistency is the strongest consistency model that remains available under partition: replicas must apply writes in an order consistent with happens-before, but concurrent writes may be observed in different orders at different replicas. _Time and Order_ introduced the clock machinery (Lamport, vector clocks, HLC); this chapter uses that machinery to build *systems*: causally consistent stores, the metadata they carry, the session guarantees they decompose into, and the stability protocols that let them prune state.

*See also:* #xref("distributed-systems", "time-and-order", label: "Time and Order") (clock algorithms and causal broadcast), #xref("distributed-systems", "crdts", label: "CRDTs") (op-based CRDTs require causal delivery), #xref("distributed-systems", "transactions", label: "Transactions") (stronger isolation levels), #xref("distributed-systems", "consensus-deep-dive", label: "Consensus Deep Dive") (linearizability, which causal consistency deliberately forgoes).

== Why Causal, Exactly

Two classic anomalies motivate the model:

+ *Lost context:* Alice removes her boss from a photo album's access list, then posts a photo. On a remote replica the photo arrives before the ACL change. The write "post photo" causally depends on "remove boss"; causal consistency forbids exposing the effect without its cause.
+ *Comment-before-post:* Bob replies to Alice's message; a third reader sees the reply but not the original.

Linearizability prevents both but requires coordination on every write and is unavailable under partition (CAP). Causal consistency prevents both *and* is available: Mahajan, Alvisi, and Dahlin (2011) proved that *real-time causal* consistency is the strongest model achievable by an always-available, one-way-convergent system, and Attiya, Ellen, and Morrison (2015) sharpened the result for the observable causal variant. Causal is, in a precise sense, the ceiling for AP systems.

== Happens-Before as the Contract

The store-level adaptation of Lamport's relation (Ahamad et al. 1995): $e_1 arrow.r.hook e_2$ if they are ordered in one session, if $e_1$ produced a value that $e_2$ read, or transitively. A causally consistent store guarantees:

- If write $w_1 arrow.r.hook w_2$, no replica applies $w_2$ before $w_1$, and no read returns $w_2$'s value while $w_1$ is invisible.
- Concurrent writes may be applied in any order, so the store needs a convergence rule for them: last-writer-wins, multi-value (return all siblings), or a CRDT merge. *Causal+* consistency (Lloyd et al. 2011) names the combination: causal consistency plus convergent conflict handling, so replicas do not merely respect causality but also agree, eventually, on conflict outcomes.

== The Metadata Question

Everything in causal systems reduces to: how do you represent "what this write depends on," and how much does that cost?

=== Vector Clocks and Their Cost

A full vector clock ($O(N)$ entries for $N$ writers, see _Time and Order_) characterizes causality exactly; Charron-Bost (1991) proved $O(N)$ is necessary for any timestamp scheme that captures concurrency precisely. For a datacenter with a handful of nodes this is fine; for a system where every client device is a writer, it is unbounded, and pruning entries reintroduces false concurrency or false ordering. Systems therefore choose where to spend: track fewer dependencies (coarser, more conservative ordering) or track them per-key (more metadata, more parallelism).

=== Version Vectors versus Vector Clocks

The two are often conflated. A *vector clock* timestamps every *event* and must tick on sends, receives, and local events to order arbitrary events. A *version vector* (Parker et al. 1983, from the LOCUS file system) tracks only the *update history of one object*: it ticks on writes to that object and is compared to detect whether two replicas of the object are ordered or in conflict. Version vectors answer "did these replicas diverge?"; vector clocks answer "did these events causally precede one another?". Version vectors are the right tool for replicated KV stores (Dynamo, Riak), and they are smaller: one entry per *replica of the key*, not per client.

=== Dotted Version Vectors

Plain version vectors fail in the client-server pattern: two clients write through the same server, the server ticks the same vector entry for both, and the second write appears to dominate the first even though the clients never saw each other's values, a *false causality*; the alternative (clients own entries) is unbounded. Dotted version vectors (Preguiça, Baquero, Almeida et al. 2010, refined 2014) fix this with one extra *dot*: a DVV is a pair $("vector", "dot")$ where the vector is the causal past the writer had seen, and the dot $(r, n)$ names this specific write as the $n$-th event at replica $r$, possibly discontiguous with the vector. Two server-side writes from independent clients now get distinct dots above overlapping vectors and are correctly recognized as siblings. Riak adopted DVVs in 1.4/2.0, eliminating the sibling explosion that plagued plain-VV configurations under heavy write concurrency.

=== Hybrid Logical Clocks as Causal Timestamps

Where the system can tolerate *potential* rather than exact causality, an HLC (see _Time and Order_ for the algorithm) collapses metadata to one 64-bit scalar: $e_1 arrow.r.hook e_2 ==> "HLC"(e_1) < "HLC"(e_2)$, like a Lamport clock, but staying within NTP skew of wall time. The price is one-directional inference only: HLC order does not imply causality, so HLCs cannot *detect* concurrency, only respect it. That trade is ideal for snapshots and sessions, which is exactly how MongoDB and CockroachDB use them.

== COPS and Eiger: Scalable Causal Stores

*COPS* (Clusters of Order-Preserving Servers, Lloyd, Freedman, Kaminsky, Andersen, SOSP 2011) was the first design to make causal+ consistency scale across sharded clusters, coining *scalable causal consistency*. Mechanics:

- Each client library tracks a *context*: the nearest dependencies of everything it has read or written (dependency tracking is per-key-version, and only *nearest* dependencies are kept since transitivity covers the rest).
- A `put` carries its dependency list. The local cluster applies it immediately (locality of writes); replication to remote clusters is asynchronous.
- A remote cluster *delays applying* a replicated write until dependency checks confirm every listed dependency is locally visible. Ordering is enforced at the destination, not by a serialization point at the source.

COPS-GT adds causally consistent read-only *transactions*: a two-round protocol that returns a consistent cut across keys, using the dependency metadata to detect and patch a torn first round. *Eiger* (Lloyd et al., NSDI 2013) generalizes to a column-family data model (it was evaluated against Cassandra), replaces version-based with *operation-based* dependencies (fewer to check), and adds write-only transactions via a variant of 2PC that still never blocks reads.

The lesson production systems took: explicit per-write dependency lists are expensive under heavy fan-out (the "metadata explosion"); later designs (ChainReaction, Orbe, GentleRain) compressed dependencies down to vectors or even a single stabilization timestamp, trading remote-visibility latency for metadata. GentleRain (Du et al. 2014) is the endpoint of that line: one physical timestamp per write, with visibility gated on a cluster-wide *Global Stable Time*, which is causal stability in disguise.

== Session Guarantees

Terry et al. (1994, the Bayou project) decomposed causal consistency into four per-session guarantees, each independently useful and independently cheap:

- *Read Your Writes:* a session's reads see all its prior writes. (No "I posted but my timeline is missing it.")
- *Monotonic Reads:* once a session has seen a value, it never reads an earlier state. (No time-travel between page refreshes hitting different replicas.)
- *Monotonic Writes:* a session's writes apply everywhere in session order.
- *Writes Follow Reads:* a write issued after reading a value is ordered after that value everywhere. (The comment-before-post fix.)

All four together, applied across sessions transitively, yield causal consistency; Brzeziński et al. (2004) formalized the equivalence. Implementation is a token: the session carries a vector (or HLC scalar) summarizing what it has seen and written; a replica serves the session only if it has caught up to the token, otherwise the request waits or is rerouted. This is exactly MongoDB's design below, and the model behind "session consistency" in Azure Cosmos DB.

== Causal Stability

A write is *causally stable* at a replica when no future delivery can be concurrent with it: every peer's known clock has passed the write's timestamp (the write's timestamp is $<=$ the pointwise minimum across the replica's view of all peers' vectors). Stability is the workhorse of garbage collection in causal systems:

- Stable writes can be applied to a compact materialized state and their metadata (dependency lists, dots, siblings) discarded.
- In op-based CRDTs, stability is when tombstones become prunable (see _CRDTs_).
- In GentleRain-style designs, the Global Stable Time *is* the read snapshot: reads at $"GST"$ see only stable writes, so no per-read dependency checks are needed at all.

The failure mode is the laggard: stability advances at the pace of the slowest (or most disconnected) replica, so one offline datacenter freezes garbage collection and, in GST designs, freezes snapshot freshness. Production systems bound this with replica eviction policies and by keeping the stability quorum within well-connected datacenters.

== Production Systems

=== MongoDB Causal Sessions

MongoDB 3.6 (2017) introduced *causally consistent sessions* built on its HLC-based `clusterTime`. Every node gossips the highest `clusterTime` it has seen; each session tracks `operationTime` of its last operation and sends it with the next request via `afterClusterTime`. A secondary serving the read simply waits until its replication has advanced past that point. With both `readConcern: "majority"` and `writeConcern: "majority"`, the session gets all four Bayou guarantees even across failovers; with weaker concerns, a rollback of unacknowledged writes can violate them. The signed `clusterTime` (HMAC) prevents a malicious client from poisoning the cluster by advancing the clock arbitrarily. Tunable consistency here is per-session and costs one scalar token, a direct payoff of choosing HLC over vectors.

=== Riak

Riak is causal at the *per-key* level: version vectors, then DVVs, order versions of each object and surface true concurrent siblings to the client (or auto-resolve via LWW or Riak DT CRDT merge). Cross-key causality is not tracked, a deliberate scope reduction that keeps metadata bounded. The DVV migration is the canonical production validation of the dotted approach: before it, busy keys behind load balancers accumulated thousands of spurious siblings.

=== Others in Brief

Azure Cosmos DB offers session and bounded-staleness levels with consistent-prefix semantics; AntidoteDB (from the SyncFree/LightKone EU projects) implements transactional causal+ consistency with CRDT objects, the closest thing to a production Cure (Akkoorath et al. 2016); Neo4j causal clusters use bookmark tokens, session guarantees by another name.

== Choosing Your Metadata: A Summary

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  table.header[*Scheme*][*Detects concurrency*][*Size*][*Typical use*],
  [Vector clock], [Yes, exactly], [$O(N)$ writers], [Causal broadcast, research systems],
  [Version vector], [Yes, per object], [$O(R)$ replicas], [Dynamo-style KV],
  [Dotted version vector], [Yes, incl. via-server writes], [$O(R)$ + 1 dot], [Riak],
  [Dependency lists], [Yes, per write], [Varies, can explode], [COPS, Eiger],
  [HLC scalar], [No, respects only], [$O(1)$], [MongoDB sessions, snapshots],
  [Global stable time], [No, hides it], [$O(1)$ per read], [GentleRain-style stores],
)

The pattern: exact causality costs linear metadata (provably), so production systems either scope it (per key, per session) or approximate it (scalar clocks, stability cuts) and accept conservative ordering. Causal consistency is less a single protocol than a budget allocation problem over this table.

== Exercises

1. Explain why the photo-album anomaly (post visible before the ACL change) violates causal consistency but not eventual consistency. Which of the two store-level happens-before clauses orders the two writes?
  _Hint: the photo post was issued in the same session after the ACL write._

2. Two clients write to the same key through one server. Show, with concrete vector states, how a plain version vector makes the second write falsely dominate the first, and how a dotted version vector recognizes the writes as siblings.
  _Hint: both writes tick the same server entry; a DVV gives each write its own dot above the shared causal past._

3. A COPS remote cluster receives a replicated `put` with a dependency list. Walk through what happens if one dependency is not yet locally visible, and explain why ordering is enforced at the destination rather than at the source. What goes wrong with this design under heavy fan-out?
  _Hint: the write is buffered until dependency checks pass; per-write dependency lists can explode in size._

4. For each of the four Bayou session guarantees, give a one-sentence user-visible anomaly that occurs when that guarantee alone is missing.
  _Hint: think of a posted-but-missing item, time-travel between refreshes, reordered edits, and a reply without its original._

5. A system tracks causality with HLC scalars only. Can it detect that two writes were concurrent? Can it guarantee that a causally later write is never applied first? Justify both answers from the direction of the HLC implication.
  _Hint: $e_1 arrow.r.hook e_2 ==> "HLC"(e_1) < "HLC"(e_2)$, but not the converse._

6. A GentleRain-style store gates reads on the Global Stable Time, and one of its five datacenters goes offline for an hour. Describe the effect on read snapshot freshness and on garbage collection, and name the mitigation production systems use.
  _Hint: stability advances at the pace of the slowest replica; consider eviction or a well-connected stability quorum._

== Further Reading

Ahamad, M., Neiger, G., Burns, J., Kohli, P., Hutto, P. (1995). "Causal Memory: Definitions, Implementation, and Programming." Distributed Computing 9(1).

Lloyd, W., Freedman, M., Kaminsky, M., Andersen, D. (2011). "Don't Settle for Eventual: Scalable Causal Consistency for Wide-Area Storage with COPS." SOSP.

Lloyd, W., Freedman, M., Kaminsky, M., Andersen, D. (2013). "Stronger Semantics for Low-Latency Geo-Replicated Storage." NSDI (Eiger).

Terry, D., Demers, A., Petersen, K., Spreitzer, M., Theimer, M., Welch, B. (1994). "Session Guarantees for Weakly Consistent Replicated Data." PDIS.

Mahajan, P., Alvisi, L., Dahlin, M. (2011). "Consistency, Availability, and Convergence." UT Austin TR-11-22.

Attiya, H., Ellen, F., Morrison, A. (2015). "Limitations of Highly-Available Eventually-Consistent Data Stores." PODC.

Parker, D.S., et al. (1983). "Detection of Mutual Inconsistency in Distributed Systems." IEEE TSE.

Preguiça, N., Baquero, C., Almeida, P.S., Fonte, V., Gonçalves, R. (2010). "Dotted Version Vectors: Logical Clocks for Optimistic Replication." arXiv:1011.5808.

Charron-Bost, B. (1991). "Concerning the Size of Logical Clocks in Distributed Systems." IPL 39(1).

Du, J., Iorgulescu, C., Roy, A., Zwaenepoel, W. (2014). "GentleRain: Cheap and Scalable Causal Consistency with Physical Clocks." SoCC.

Tyulenev, M., et al. (2019). "Implementation of Cluster-wide Logical Clock and Causal Consistency in MongoDB." SIGMOD.

Akkoorath, D., et al. (2016). "Cure: Strong Semantics Meets High Availability and Low Latency." ICDCS.

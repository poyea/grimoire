= Introduction: Models and Impossibilities

A distributed system is a collection of independent computers that appears to its users as a single coherent system. The discipline studies what is *possible* under partial failure, asynchrony, and adversarial scheduling — and what is *impossible*. Every engineering decision in a real system (timeout values, quorum sizes, lease durations) is a concrete answer to one of these theoretical questions.

*See also:* _Time and Order_, _Failure Detection_, _Consensus Deep Dive_, _Formal Methods_, and _Consensus and Replication_ (database-specific framing).

== The Eight Fallacies

Peter Deutsch and James Gosling (Sun, 1994/1997) enumerated the assumptions that quietly poison naive distributed designs:

+ The network is reliable.
+ Latency is zero.
+ Bandwidth is infinite.
+ The network is secure.
+ Topology doesn't change.
+ There is one administrator.
+ Transport cost is zero.
+ The network is homogeneous.

Every fallacy maps to a class of bugs. Believing #1 produces silent data loss when retries reorder. Believing #2 produces $N+1$ query patterns that work in tests and explode under WAN deployment. Believing #5 produces hard-coded IPs that break on autoscaling. The skill of the distributed engineer is to internalize the negation of each.

== System Models

A *system model* is a triple: (synchrony assumption, failure model, network model). Algorithms must be analyzed in the model they target — a Paxos correctness proof under partial synchrony does not transfer to asynchronous Byzantine settings.

=== Synchrony

#table(
  columns: (auto, 1fr, 1fr),
  table.header[*Model*][*Message delay*][*Process step time*],
  [Synchronous], [Bounded, known $Delta$], [Bounded, known],
  [Partially synchronous (DLS88)], [Bounded but unknown, or eventually bounded], [Same],
  [Asynchronous], [Arbitrary, unbounded], [Arbitrary],
)

Real networks are *partially synchronous*: they behave synchronously most of the time but exhibit unbounded delay during partitions, GC pauses, or VM live-migration. Algorithms designed for this model (Paxos, Raft, PBFT) are *safe* in asynchrony and *live* once synchrony is restored — they cannot violate consistency, but they may stall.

=== Failure Models

- *Crash-stop:* a process halts and never recovers. Simplest model.
- *Crash-recovery:* a process may halt and later restart, possibly losing volatile state but preserving stable storage.
- *Omission:* a process fails to send or receive some messages but is otherwise correct.
- *Byzantine:* a process behaves arbitrarily — sends conflicting messages to different peers, lies about its state, colludes with other faulty nodes. Required model for blockchains and adversarial environments.

Faults compose. A network partition can be modeled as simultaneous omission faults on all cross-partition links. A long GC pause looks like a crash followed by a recovery.

=== Network Models

- *Reliable point-to-point links:* messages are eventually delivered (no loss), no duplication, no creation. Implemented over TCP plus retries plus deduplication.
- *Fair-loss links:* infinite retransmissions eventually succeed. The minimal model needed by most algorithms.
- *Authenticated:* recipients can verify sender identity (PKI or shared secret) — required for Byzantine protocols.

== FLP Impossibility

Fischer, Lynch, and Paterson (1985): in a purely asynchronous system where even *one* process may crash, there is no deterministic protocol that solves consensus and always terminates.

*Proof sketch.* Define a *bivalent* configuration as one whose outcome (the eventually decided value) is not yet determined — both 0 and 1 reachable. Show that (a) some initial configuration is bivalent, and (b) from any bivalent configuration the adversary can schedule messages to reach another bivalent configuration. By induction, an infinite execution exists that never decides.

*Why it matters.* The result rules out *deterministic, always-terminating, asynchronous* consensus. Real systems escape via:

- *Partial synchrony* — assume eventual timing bounds (Paxos, Raft).
- *Randomization* — Ben-Or's protocol terminates with probability 1 (Nakamoto, Algorand).
- *Failure detectors* — abstract the missing timing assumption into an oracle ($diamond.stroked S$ is the weakest detector that solves consensus, Chandra–Toueg).

== CAP and PACELC

Brewer (2000) conjectured, and Gilbert and Lynch (2002) formalized: a replicated register cannot simultaneously provide *Consistency* (linearizability), *Availability* (every request to a non-failed node returns), and *Partition tolerance* (the system continues despite arbitrary message loss). Under partition, choose CP or AP.

PACELC (Abadi 2010) extends: even when there is no Partition, there is a Latency-Consistency tradeoff. A CP system that synchronously replicates pays cross-region RTT on every write; an AP system trades freshness for sub-millisecond local writes.

#table(
  columns: (auto, 1fr, 1fr),
  table.header[*System*][*Partition behavior*][*Else (no-partition)*],
  [Spanner], [CP], [PC — synchronous Paxos cost],
  [DynamoDB (eventual)], [AP], [EL — local reads],
  [Cassandra (QUORUM)], [Tunable], [Tunable],
  [HBase], [CP], [PC],
  [Cosmos DB], [Configurable], [Configurable],
)

Note CAP applies to a single register under specific definitions. Modern systems compose primitives with different CAP profiles per operation: e.g., DynamoDB strong reads (CP) coexist with eventually consistent reads (AP) on the same table.

== Linearizability Versus Serializability

Two often-confused correctness conditions:

- *Linearizability* (Herlihy–Wing 1990) is about *single objects*: operations appear to occur atomically at some point between their invocation and response. It composes (a collection of linearizable objects is linearizable as a whole).
- *Serializability* is about *transactions* over multiple objects: there exists some total order of transactions equivalent to the concurrent execution. It does not imply any real-time order between disjoint transactions.

*Strict serializability* = serializable + linearizable. Spanner provides strict serializability via TrueTime; CockroachDB provides serializable plus a weaker linearizable guarantee per row.

== A Worked Example: a Two-Node Counter

Consider two nodes, A and B, each holding a copy of an integer $x$. Clients issue `inc` and `read`. What can go wrong?

```
Client1 -> A: inc        Client2 -> B: inc
A: x = 1, replicate ->   B: x = 1, replicate ->
Network reorders both replications.
A applies B's update: x = 1 (idempotent on key, wrong!).
B applies A's update: x = 1.
Final state: x = 1, but two increments occurred.
```

Fixes correspond to entire subfields of this book:

- *State-based CRDT* (G-Counter): each node tracks its own count, merges take element-wise max. We will cover this in _Replication Protocols_.
- *Operation-based CRDT*: deliver `inc` exactly once in causal order — requires reliable causal broadcast (_Time and Order_).
- *Consensus*: serialize every `inc` through a leader (_Consensus Deep Dive_).
- *Compensating transaction*: detect divergence, run a reconciliation (_Transactions Across Systems_).

The choice depends on availability requirements, write rate, and the application's tolerance for staleness.

== Failure Frequencies in Practice

Empirical data from Google (Barroso 2009), Facebook (Maneas 2020), and Backblaze drive reports informs realistic models:

- Hard disk AFR: 0.5--5% depending on model and age.
- DRAM uncorrectable error rate: $approx 25,000$ FIT per Mbit (Schroeder et al. 2009).
- Rack-level power events: monthly at hyperscale.
- DC-level outages: 1--2 per year per site (cooling, fiber cut, BGP misconfiguration).
- Correlated failures dominate independent ones; Markov-chain availability calculations that assume independence wildly overestimate reliability.

Design implication: replication factor 3 across one rack is approximately equivalent to replication factor 1 across three racks because rack-level correlated failure swamps disk-level independent failure.

== Two Generals and the Coordinated Attack

The Two Generals problem (Akkoyunlu et al. 1975) proves: with a lossy channel, no finite protocol can give both parties certainty that the other knows the message was received. This applies recursively to acknowledgements.

*Consequence.* No exactly-once message delivery exists end-to-end with finite messages. Real systems implement "effectively-once" by combining at-least-once delivery with idempotent receivers, using deduplication keys (Kafka producer IDs, gRPC retry tokens).

== Asynchronous Versus Eventual Consistency Spectrum

```
strict serializable
   |
linearizable
   |
sequential
   |
causal+
   |
read-your-writes / monotonic reads
   |
eventual
   |
no guarantee
```

Each step weaker level admits more concurrency and tolerates more failures but pushes complexity to the application. Bailis et al. (2014) showed that many real workloads need only causal+ or session guarantees, which can be implemented with bounded staleness on AP systems.

== Reading Map for This Book

- *Foundations:* this chapter, _Time and Order_, _Failure Detection_.
- *Coordination:* _Leader Election_, _Consensus Deep Dive_, _Coordination Services_.
- *Data:* _Replication Protocols_, _State Machine Replication_, _Gossip_, _Transactions_.
- *Systems:* _Distributed Scheduling_, _Case Studies_.
- *Verification:* _Formal Methods_.

== Further Reading

Fischer, M., Lynch, N., Paterson, M. (1985). "Impossibility of Distributed Consensus with One Faulty Process." JACM 32(2).

Dwork, C., Lynch, N., Stockmeyer, L. (1988). "Consensus in the Presence of Partial Synchrony." JACM 35(2).

Gilbert, S., Lynch, N. (2002). "Brewer's Conjecture and the Feasibility of Consistent, Available, Partition-Tolerant Web Services." SIGACT News.

Abadi, D. (2012). "Consistency Tradeoffs in Modern Distributed Database System Design." IEEE Computer (PACELC).

Lynch, N. (1996). _Distributed Algorithms_. Morgan Kaufmann.

Cachin, C., Guerraoui, R., Rodrigues, L. (2011). _Introduction to Reliable and Secure Distributed Programming_. Springer.

Bailis, P., Davidson, A., Fekete, A., Ghodsi, A., Hellerstein, J., Stoica, I. (2014). "Highly Available Transactions: Virtues and Limitations." VLDB.

Herlihy, M., Wing, J. (1990). "Linearizability: A Correctness Condition for Concurrent Objects." TOPLAS.

Chandra, T., Toueg, S. (1996). "Unreliable Failure Detectors for Reliable Distributed Systems." JACM.

Akkoyunlu, E., Ekanadham, K., Huber, R. (1975). "Some Constraints and Tradeoffs in the Design of Network Communications." SOSP.

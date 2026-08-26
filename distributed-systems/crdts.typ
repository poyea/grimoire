#import "../template.typ": xref

= Conflict-Free Replicated Data Types

CRDTs let replicas accept writes locally, without coordination, and still converge to the same state once they have exchanged updates. The trick is to restrict the data type so that concurrent updates *commute by construction*: there is nothing to "resolve" at merge time because the type's semantics already define a deterministic outcome for every interleaving. This chapter develops the lattice theory behind that guarantee, walks through the canonical counter, register, set, and sequence CRDTs, and ends where CRDTs end: at global invariants that genuinely require consensus.

*See also:* #xref("distributed-systems", "gossip", label: "Gossip Protocols") (dissemination and a first look at G-Counters), #xref("distributed-systems", "causal-consistency", label: "Causal Consistency") (the delivery order op-based CRDTs require), #xref("distributed-systems", "time-and-order", label: "Time and Order") (version vectors, HLC timestamps for LWW), #xref("distributed-systems", "consensus-deep-dive", label: "Consensus Deep Dive") (what to use when CRDTs cannot).

== Strong Eventual Consistency

Eventual consistency promises only that replicas *eventually* agree, saying nothing about how conflicts are resolved or whether resolution is deterministic. Shapiro, Preguiça, Baquero, and Zawirski (2011) defined the stronger property that CRDTs actually provide, *strong eventual consistency* (SEC):

+ *Eventual delivery:* every update delivered at one correct replica is eventually delivered at all correct replicas.
+ *Convergence:* replicas that have delivered the same *set* of updates have equivalent state, regardless of delivery order.
+ *Termination:* all operations complete locally, without waiting on other replicas.

Convergence on the same set, not the same sequence, is the key strengthening: there is no reconciliation phase, no rollback, and no need for replicas to agree on an order. SEC sits at the availability extreme of CAP: a CRDT replica is always writable, even when partitioned, and the 2011 paper proves SEC is achievable wait-free in asynchronous networks where consensus is not.

== Two Formulations: State-Based and Op-Based

=== State-Based (CvRDT) and Join Semilattices

A state-based CRDT is a triple: a set of states $S$, a partial order $subset.sq.eq$ on $S$, and a *join* (merge) $union.sq$ that computes the least upper bound of two states. $(S, subset.sq.eq, union.sq)$ must form a *join semilattice*, which makes merge:

- *Commutative:* $a union.sq b = b union.sq a$
- *Associative:* $(a union.sq b) union.sq c = a union.sq (b union.sq c)$
- *Idempotent:* $a union.sq a = a$

Updates must be *inflations*: every update moves the state upward in the order ($s subset.sq.eq "update"(s)$). Given these properties, the convergence proof is short: any two replicas that have absorbed the same set of updates hold the join of those updates' effects, and the join of a set is unique in a semilattice; commutativity and associativity make the merge order irrelevant, and idempotence makes duplicate delivery harmless. Replication therefore needs only an unreliable, unordered transport. Gossip and anti-entropy (see #xref("distributed-systems", "gossip", label: "Gossip Protocols")) are a perfect fit.

The cost is payload size: naive CvRDTs ship the entire state on every exchange.

=== Op-Based (CmRDT)

An op-based CRDT ships operations instead of states. Each update splits into a *prepare* phase (executed once, at the origin, may read local state to produce the operation, e.g. generating a unique tag) and an *effect* phase (executed at every replica). Convergence requires that *concurrent* effects commute; effects related by happens-before may be ordered.

Because effects are not idempotent in general, op-based CRDTs push requirements onto the transport: *exactly-once, causal delivery*. Causal broadcast (Birman 1991; see #xref("distributed-systems", "time-and-order", label: "Time and Order")) provides exactly this, at the cost of vector-clock metadata and delivery buffering. The two formulations are formally equivalent: each can emulate the other (the 2011 paper gives both constructions), so the choice is an engineering trade between bandwidth (state-based) and delivery machinery (op-based).

== Canonical CRDTs

=== G-Counter and PN-Counter

The grow-only counter keeps one entry per replica; increment bumps your own entry, value sums all entries, merge is pointwise max (a product of max-semilattices, hence a semilattice). Decrement breaks the inflation rule, so the *PN-Counter* pairs two G-Counters, $P$ for increments and $N$ for decrements, with value $sum_i P[i] - sum_i N[i]$. Note what is lost: a PN-Counter cannot enforce "never below zero." That is a global invariant (see the limits section below).

=== LWW-Register

A register holding $("value", "timestamp")$; merge keeps the entry with the larger timestamp, tie-broken by replica ID. This is a semilattice on the lexicographic order of $("timestamp", "replica id")$. The semantics are exactly as good as the timestamps: with wall clocks, skew silently drops writes (the Cassandra failure mode catalogued in #xref("distributed-systems", "time-and-order", label: "Time and Order")); HLC timestamps restore causal monotonicity, so a write never loses to an update it causally followed. The alternative, the *MV-Register*, keeps all concurrent values (a version-vector-guarded set, as in Dynamo) and pushes the choice to the reader.

=== OR-Set: Add-Wins and Remove-Wins

A 2P-Set (a grow-only "added" set plus a grow-only tombstone set) forbids re-adding a removed element. The *Observed-Remove Set* fixes this with unique tags: `add(e)` creates a fresh tag $(("replica"), ("counter"))$ for $e$; `remove(e)` deletes exactly the tags *observed locally* at the time of removal. A concurrent `add(e)` carries a tag the remover never saw, so the element survives: *add-wins* semantics. The dual *remove-wins* set tombstones the element itself so that a concurrent remove beats any concurrent add; it is rarer in practice but offered by Riak alongside the add-wins variant. Which is "right" is an application decision (shopping carts famously want add-wins; access-control lists arguably want remove-wins).

The *optimized OR-Set* of Bieniusa et al. (2012) eliminates per-element tombstones by keeping a version vector of seen dots: a tag absent from the element set but covered by the vector is known-removed. This is the same causal-context idea as dotted version vectors (see #xref("distributed-systems", "causal-consistency", label: "Causal Consistency")) and bounds metadata to $O("elements" + "replicas")$.

=== Sequence CRDTs: RGA and Friends

Collaborative text needs a replicated list where concurrent inserts at the same position converge. The shared idea across designs is to give each element a *stable, totally ordered identifier* that never changes as neighbours come and go:

- *WOOT* (Oster et al. 2006): each character records its left and right neighbour identities; integration places it consistently between them. Tombstones are never removed.
- *Treedoc* (Preguiça et al. 2009) and *Logoot* (Weiss et al. 2009): identifiers are paths in a tree / dense position strings, so ordering is identifier comparison. Identifiers can grow unboundedly under adversarial insert patterns.
- *RGA*, the Replicated Growable Array (Roh et al. 2011): each element is identified by the timestamp of its insertion; `insert-after(ref, elem)` links the new element after `ref`, and concurrent inserts after the same reference are ordered by descending timestamp, so all replicas linearize the siblings identically. Deletes tombstone. RGA's insert-after model underlies most production implementations.

Interleaving is the subtle failure mode: two users concurrently typing words at the same position can converge to character-interleaved garbage under Logoot-style identifiers. Kleppmann et al. (2019) showed several published algorithms interleave and proposed fixes; sibling ordering by origin makes RGA-family designs largely immune.

== Deltas: Shrinking State-Based Payloads

Delta-state CRDTs (Almeida, Shoker, Baquero 2018) keep the semilattice but ship *delta-mutators*: an update returns a small state $delta$ such that merging $delta$ into the full state has the same effect as the update. Deltas are themselves lattice elements, so they can be batched (joined together), reordered, and duplicated safely; replicas buffer recent deltas per neighbour and fall back to full-state anti-entropy after a gap. This recovers op-based bandwidth while keeping state-based delivery guarantees (none needed), which is why deltas dominate modern implementations (Akka Distributed Data, Redis Enterprise CRDBs, automerge-style sync protocols).

== Garbage Collection and Tombstones

CRDT metadata only grows: OR-Set tags, sequence tombstones, per-replica counter entries. Three strategies bound it:

- *Causal-context compaction:* replace explicit tombstones with a version vector of seen dots (optimized OR-Set, dotted version vectors). Removal becomes "covered by the vector but not present," and the vector compacts contiguous dots into one integer per replica.
- *Causal stability:* an operation is *causally stable* at a replica once it is known to have been delivered everywhere (its timestamp is below the minimum of all replicas' vector entries). Stable tombstones can be discarded because no concurrent operation referencing them can still arrive (Baquero, Almeida, Shoker 2014). The catch: stability stalls if any replica is offline, which is exactly the regime (offline-first apps) where CRDTs are most attractive.
- *Consensus-assisted GC:* periodically run an agreement round to retire dead replica IDs and truncate history. Ironically, pruning a coordination-free data type is itself a coordination problem.

== Collaborative Text Editing in Production

*Yjs* (Jahns, building on the YATA algorithm of Nicolaescu et al. 2016) is an RGA-family list CRDT engineered for speed: contiguous runs of characters typed by one user are coalesced into single items, so a 100k-character document is a few thousand items, not 100k. Its sync protocol exchanges state vectors and computes minimal diffs, effectively a delta-CRDT.

*Automerge* (Kleppmann et al.) targets JSON documents: maps, lists (RGA-based), text, and counters compose into a tree, with full editing history retained for time-travel and merge audit. Its columnar binary encoding compresses per-operation metadata by storing operation fields in compressed columns, shrinking documents by orders of magnitude versus naive per-op JSON.

*Peritext* (Litt, Kleppmann et al. 2022) tackles *rich text*: bold/italic spans are not character properties but anchored ranges with explicit expansion rules, so concurrent formatting and editing merge with intent preserved (e.g. bolding a sentence while someone inserts a word inside it bolds the new word too). It is a reminder that CRDT design is mostly *semantics* design: the data structure is easy once you have decided what concurrent users should get.

== Limits: No Global Invariants Without Coordination

A CRDT guarantees convergence, not correctness of application invariants that span replicas. You cannot, coordination-free, maintain "balance $>= 0$", "at most one seat 14A", or "usernames are unique": two partitioned replicas can each locally satisfy the invariant while their join violates it. This is not an implementation gap but a theorem. The CALM result (Hellerstein and Alvaro 2020) states that exactly the *monotone* programs have coordination-free, consistent implementations, and invariants like non-negativity are non-monotone (adding information, a decrement, can falsify them). The *escrow* technique partitions a numeric budget across replicas so each can decrement its share locally, reintroducing coordination only on rebalance; bounded counters in Antidote take this approach.

When the invariant is real, use consensus for the invariant-bearing decisions and CRDTs for everything else (see #xref("distributed-systems", "consensus-deep-dive", label: "Consensus Deep Dive")). Mixed designs are common: Riak uses CRDTs per key but strong consistency buckets when needed; RedBlue consistency (Li et al. 2012) formalizes the split into coordination-free "blue" and serialized "red" operations.

== Where CRDTs Run Today

- *Riak DT* (Basho, 2.0 in 2014): counters, sets, maps, flags, registers as first-class bucket types; the map composes other CRDTs recursively.
- *Redis Enterprise CRDBs:* active-active geo-replication where each data type has CRDT semantics (counters merge additively, sets are OR-Sets).
- *Akka Distributed Data:* delta-CRDTs gossiped across an actor cluster for service discovery and shared configuration.
- *SoundCloud's Roshi:* an LWW-element-set over Redis for the fan-out timeline index.
- *Figma, Apple Notes, and most collaborative editors:* sequence/JSON CRDTs (or close cousins) for offline-tolerant multi-user editing.
- *Phoenix Presence (Elixir):* an OR-Set-like CRDT tracking who is online per topic, merged via gossip with no central registry.

== Exercises

1. State the three properties of strong eventual consistency and explain why "same set of updates implies equivalent state" is strictly stronger than plain eventual consistency. Which property rules out a reconciliation-and-rollback design?
  _Hint: SEC removes any need for replicas to agree on an order; convergence is determined by the set alone._

2. Prove that a G-Counter's merge (pointwise max over per-replica entries) forms a join semilattice, and show where the proof breaks if decrement of one's own entry were allowed.
  _Hint: a product of max-semilattices is a semilattice; decrement violates the inflation requirement._

3. Replicas $A$ and $B$ start with an OR-Set containing element $e$ with tag $t_1$. Concurrently, $A$ executes `remove(e)` while $B$ executes `add(e)` (fresh tag $t_2$). Trace the tag sets after both replicas merge, and explain why the result is add-wins. How would a remove-wins set decide instead?
  _Hint: $A$ deletes only the tags it observed; $t_2$ was never observed by the remover._

4. A team builds an LWW-Register keyed on wall-clock timestamps across servers with up to 2 seconds of clock skew. Describe the anomaly that can silently occur, and explain how HLC timestamps change the guarantee.
  _Hint: a causally later write can carry a smaller wall-clock timestamp; HLC ensures a write never loses to an update it causally followed._

5. Compare state-based, op-based, and delta-state CRDTs along two axes: payload size and required delivery guarantees. Why can deltas be duplicated and reordered safely while op-based effects cannot?
  _Hint: deltas are lattice elements merged with an idempotent join; effects are not idempotent in general._

6. Your product needs a replicated counter that must never go below zero, with replicas accepting decrements during partitions. Explain why no CRDT can provide this coordination-free, citing the relevant theorem, and sketch the escrow workaround.
  _Hint: non-negativity is non-monotone, so CALM says coordination is required; escrow pre-partitions the budget across replicas._

== Further Reading

Shapiro, M., Preguiça, N., Baquero, C., Zawirski, M. (2011). "Conflict-Free Replicated Data Types." SSS; and the companion technical report "A Comprehensive Study of Convergent and Commutative Replicated Data Types." INRIA RR-7506.

Roh, H., Jeon, M., Kim, J., Lee, J. (2011). "Replicated Abstract Data Types: Building Blocks for Collaborative Applications." JPDC.

Almeida, P.S., Shoker, A., Baquero, C. (2018). "Delta State Replicated Data Types." JPDC.

Bieniusa, A., et al. (2012). "An Optimized Conflict-Free Replicated Set." INRIA RR-8083.

Baquero, C., Almeida, P.S., Shoker, A. (2014). "Making Operation-Based CRDTs Operation-Based." DAIS.

Kleppmann, M., Gomes, V., Mulligan, D., Beresford, A. (2019). "Interleaving Anomalies in Collaborative Text Editors." PaPoC.

Litt, G., Lim, S., Kleppmann, M., van Hardenberg, P. (2022). "Peritext: A CRDT for Rich-Text Collaboration." CSCW.

Hellerstein, J., Alvaro, P. (2020). "Keeping CALM: When Distributed Consistency Is Easy." CACM 63(9).

Li, C., et al. (2012). "Making Geo-Replicated Systems Fast as Possible, Consistent when Necessary." OSDI.

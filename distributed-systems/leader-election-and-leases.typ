#import "../template.typ": xref

= Leader Election and Leases <leader-election-and-leases>

Many distributed protocols are simpler with a distinguished coordinator: a sequencer for ordering, an exclusive writer to a shard, a master scheduling tasks. The challenge is electing a leader despite failures, and *bounding* leadership so a partitioned former leader cannot corrupt state. Leases and fencing tokens are how this bound is enforced.

*See also:* #xref("distributed-systems", "failure-detection", label: "Failure Detection"), #xref("distributed-systems", "consensus-deep-dive", label: "Consensus Deep Dive") (Raft leader election, reconfiguration), #xref("distributed-systems", "coordination-services", label: "Coordination Services") (Chubby, etcd leases).

== Why Elect a Leader?

Leaderless protocols (EPaxos, gossip CRDTs) exist, but a single leader simplifies:

- *Ordering:* the leader assigns log positions, eliminating cross-replica conflicts.
- *Throughput:* one Paxos roundtrip per command instead of two (Multi-Paxos).
- *Client routing:* clients send writes to a known endpoint.
- *Garbage collection:* the leader can safely truncate old log entries.

Cost: a leader is a single point of contention and a tail-latency target. Multi-leader and leaderless schemes trade simplicity for scalability.

== The Bully Algorithm

Garcia-Molina (1982). When a node notices the leader is gone:

```
on detect_leader_failure(self):
    higher = [n for n in nodes if n.id > self.id]
    if not higher:
        announce(COORDINATOR, self.id)
        return
    send ELECTION to all higher
    wait timeout
    if no ANSWER received:
        announce(COORDINATOR, self.id)
    # else: wait for a COORDINATOR message

on receive_ELECTION(from j):
    send ANSWER to j
    start own election if not already
```

Properties: $O(N^2)$ messages worst case; tolerates crash failures only; assumes synchronous network for the timeout. Rarely used in modern systems but pedagogically clean.

== Ring-Based Election

Chang–Roberts (1979): processes arranged in a logical ring, election token circulates carrying the maximum ID seen. The node whose ID matches the returning token becomes leader. $O(N log N)$ amortized with smart bookkeeping; the LCR algorithm.

== Raft-Style Election

Modern systems use *terms* (monotonic election epochs):

```
Follower state:
    timer = randomize(electionTimeoutMin, electionTimeoutMax)
    if timer expires without heartbeat:
        currentTerm += 1
        votedFor = self
        state = Candidate
        send RequestVote(term=currentTerm, lastLogIdx, lastLogTerm) to all

Receiver of RequestVote(t, lastIdx, lastTerm):
    if t < currentTerm:    reply false
    if t > currentTerm:    currentTerm = t; votedFor = nil; state = Follower
    if votedFor in {nil, candidate} and candidate_log_at_least_as_up_to_date:
        votedFor = candidate
        reply true

Candidate:
    if received majority of yes votes for currentTerm:
        state = Leader
        send heartbeats
```

Randomized timeouts ($150$--$300$ ms typical) reduce split votes to near-zero. Up-to-date log requirement: $("lastTerm", "lastIdx")$ of candidate $>=$ that of voter — ensures committed entries survive.

Split-brain prevention: a term has at most one leader because winning requires a majority and any two majorities intersect in some voter, who can vote for only one candidate per term.

== Leases

A *lease* is a time-limited grant: "you are the leader from $t_0$ to $t_0 + L$." If the holder cannot reach the granting authority before expiry, it must voluntarily relinquish (stop serving as leader). Implementations:

=== Centralized Lease (Chubby, ZooKeeper)

A coordination service tracks the current lease holder via an ephemeral lock/lease znode. The holder must renew before expiry; if its session dies, the service deletes the lease and notifies watchers.

```
acquire():
    try create("/leader", value=self.id, ephemeral=True)
    on success: I am leader for session lifetime
    on EEXISTS: watch the node, retry on deletion

renewal: background heartbeat keeps the session alive (every L/3)
```

Chubby's KeepAlive carries piggybacked events. A lease delay parameter (default 12 s) gives the client time to detect master changes.

=== Distributed Lease via Consensus

A consensus group (Raft, Paxos) commits a record `(leader=X, lease_until=T)`. The leader becomes valid once committed, and remains so until $T$ on the wall clock of any replica — assuming bounded clock drift $rho$:

$ "real lease end" >= T - rho dot ("max drift period") $

CockroachDB uses *epoch-based leases*: the leader writes its epoch into a meta range; followers serve based on epoch validity rather than wall-clock time. Avoids clock skew entirely at the cost of an extra range write per liveness heartbeat.

== Fencing Tokens

The fundamental safety issue: a former leader, partitioned away during a GC pause, may *believe* it still holds the lease and continue writing to backing storage. The classic example (Kleppmann 2017): a client holds a Redis-backed distributed lock, pauses 15 s for GC, wakes, and writes to S3 — but a new lock holder has been writing for 14 s.

The fix is a monotonic *fencing token* embedded in every protected operation:

```
lock_acquire() -> token   // strictly increasing, e.g., from etcd revision
storage_write(key, value, fence=token):
    if token < storage.highest_fence_seen:
        abort  // stale leader
    storage.highest_fence_seen = max(storage.highest_fence_seen, token)
    apply write
```

The *storage* enforces monotonicity. This requires the storage layer to participate; opaque blob stores cannot. Use cases:

- S3 conditional writes ($"If-Match"$ on ETag, or recent conditional `If-None-Match` / object lock).
- Cosmos DB and DynamoDB conditional puts.
- HDFS file open generation stamp.
- Spanner: timestamps act as implicit fences via strict serializability.

Without storage cooperation, leases are unsafe. This is the dirty secret behind "highly available locks" built on Redis or Memcached.

== Clock-Bounded Versus Clock-Free Leases

Two regimes:

#table(
  columns: (auto, 1fr, 1fr),
  table.header[*Approach*][*Clock-bounded*][*Clock-free*],
  [Source of truth], [Wall clock + bounded drift], [Quorum heartbeats / epoch],
  [Risk], [Skew or VM pause leaks beyond lease], [None from clocks],
  [Cost], [Pure local check, fast], [Quorum traffic for renewal],
  [Example], [Spanner read leases, Chubby], [CockroachDB epoch leases, ZooKeeper sessions],
)

A subtle point: even "clock-free" leases on a consensus group rely on the *consensus group's* progress to expire stale lease holders, which itself relies on a partially synchronous model.

== Co-located Leader Optimization

For locality-aware systems (CockroachDB, Spanner), leader leases are assigned to replicas geographically near the dominant traffic source. CockroachDB's "leaseholder" can differ from the Raft leader to colocate reads/writes; load-based rebalancing migrates leases between replicas.

== Avoiding Herd Effects on Election

If the leader dies, all followers race to become candidate. Mitigations:

- *Randomized timeouts* (Raft).
- *Pre-vote phase* (Raft thesis): a candidate first asks if peers would vote, without bumping its term. Avoids disruptive term inflation during partitions.
- *Priority-based candidacy:* prefer the replica with the most up-to-date log (Raft's log-completeness check already implements this).

== Operational Failure Modes

- *Clock skew / lease expiry thrash:* When NTP steps cause wall-clock jumps >5 ms, a leader may believe its lease has expired while followers still consider it valid. This creates a window where two nodes each believe they can act as leader. Mitigation: use monotonic clocks for lease timers (not wall clock), and validate the lease with a heartbeat round-trip before issuing writes.

- *Network partition with old leader:* A partitioned leader holding a lease continues issuing writes; the new leader elected in the majority partition also writes. On partition heal, conflict resolution must handle diverged log tails. Solution: the old leader must fence (fail-stop or fence-token checks) before accepting further writes.

- *GC pause expiry:* JVM stop-the-world GC pauses of 500 ms--2 s (pre-ZGC) can expire a 1-second lease while the holder is frozen. The node resumes believing it holds the lease, but the cluster has already elected a new leader. Mitigation: heartbeat threads must run on separate OS threads, not user threads; use ZGC or Shenandoah on JVM.

- *Cascading election storms:* Under load, election timeouts fire simultaneously across multiple nodes; each candidate votes for itself and no quorum forms. Randomized election timeouts (Raft: 150--300 ms random range) break symmetry; PBFT and Paxos use proposer backoff to prevent repeated collisions.

== Combined Pattern: Lease + Fence + Replicated Log

The robust recipe:

+ Acquire lease via consensus group (etcd, Raft cluster).
+ Receive a monotonic *fencing token* (e.g., the Raft log index where lease was committed).
+ Embed the token in every write to backing storage.
+ Renew the lease at $L/3$ intervals; on renewal failure, immediately stop and exit.
+ Storage layer rejects writes with stale fencing tokens.

This is the architecture used by Kafka controller (KRaft), HDFS NameNode HA with ZK, Kubernetes controller-manager (leader-election library).

== Further Reading

Burrows, M. (2006). "The Chubby Lock Service for Loosely-Coupled Distributed Systems." OSDI.

Ongaro, D., Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm." USENIX ATC (Raft).

Kleppmann, M. (2017). "How to Do Distributed Locking." Blog post (the Redlock critique).

Garcia-Molina, H. (1982). "Elections in a Distributed Computing System." IEEE TC.

Chang, E., Roberts, R. (1979). "An Improved Algorithm for Decentralized Extrema-Finding in Circular Configurations of Processes." CACM.

Taft, R. et al. (2020). "CockroachDB: The Resilient Geo-Distributed SQL Database." SIGMOD (epoch leases).

Gray, C., Cheriton, D. (1989). "Leases: An Efficient Fault-Tolerant Mechanism for Distributed File Cache Consistency." SOSP.

Ousterhout, J. (2018). "Always Measure One Level Deeper." CACM.

Hunt, P., Konar, M., Junqueira, F., Reed, B. (2010). "ZooKeeper: Wait-Free Coordination for Internet-Scale Systems." USENIX ATC.

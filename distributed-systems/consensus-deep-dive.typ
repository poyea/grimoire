= Consensus Deep Dive

Consensus is the abstraction every fault-tolerant distributed system reduces to: $N$ processes propose values, a single value is decided, and all correct processes eventually learn it. Beneath Paxos, Raft, EPaxos, and the Byzantine family lies a shared structure — *quorum intersection* plus *value carry-forward* across configuration epochs — that this chapter unfolds.

*See also:* _Introduction_ (FLP), _Log-Based Systems_, _Leader Election and Leases_, and _Consensus and Replication_ (Databases volume — database-side framing).

== The Consensus Problem

Properties:

- *Agreement:* no two correct processes decide differently.
- *Validity:* a decided value was proposed by some process.
- *Termination:* every correct process eventually decides.

FLP rules out deterministic asynchronous solutions. We assume partial synchrony.

== Single-Decree Paxos

Lamport (1989/1998). Roles: Proposer, Acceptor, Learner. A *ballot number* $n$ is a (round, proposer-id) pair, totally ordered.

Two phases:

```
Phase 1a (PREPARE):
    proposer chooses n higher than any seen
    send PREPARE(n) to acceptors

Phase 1b (PROMISE):
    acceptor: if n > highest_promised:
        highest_promised = n
        reply PROMISE(n, accepted_n, accepted_v)
    else: reply NACK(highest_promised)

Phase 2a (ACCEPT):
    proposer: upon majority of PROMISEs:
        v = accepted_v with highest accepted_n among PROMISEs
            (else proposer's own initial value)
        send ACCEPT(n, v) to acceptors

Phase 2b (ACCEPTED):
    acceptor: if n >= highest_promised:
        accepted_n = n; accepted_v = v
        reply ACCEPTED(n) to learners

Decision:
    when a majority of ACCEPTED(n) for some v exist, v is decided
```

*Why it is safe.* Any two majorities intersect; so any value committed in ballot $n$ is reported back during Phase 1 of any later ballot $n' > n$, and the new proposer must re-propose it.

*Why it can stall.* Two proposers with rising ballot numbers can preempt each other indefinitely (dueling proposers). Solution: elect a *distinguished proposer* (leader); the result is Multi-Paxos.

== Multi-Paxos

Reuses Phase 1 across many slots: a leader runs Phase 1 once to "own" all future slots, then commits each command with a single Phase 2 roundtrip. Steady-state throughput: 1 RTT per command. On leader change, Phase 1 must be repeated to discover any partially-committed values in unfilled slots.

Practical issues handled in real implementations (Chubby, Spanner, Google's MultiPaxos lib):

- *Log compaction* via snapshots; truncate prefix of agreed log.
- *Reconfiguration:* use an $alpha$-slot lookahead, or joint consensus (see the Raft paper and _Coordination Services_).
- *Read leases:* the leader holds a read lease to serve linearizable reads locally.
- *Batching and pipelining:* group commands per RTT; pipeline accepts without waiting.

== Vertical Paxos

Lamport, Malkhi, Zhou (2009). Generalizes leader handoff: a new leader can be elected *and* the configuration changed in one step. Requires an external "master" service to certify the configuration. Spanner's Paxos groups use a variant.

== Raft

Ongaro and Ousterhout (2014). Designed for understandability. State machine of each peer: Follower, Candidate, Leader. Three subproblems decoupled:

+ *Leader election* — randomized timeouts, terms (see _Leader Election_).
+ *Log replication* — leader appends entries, replicates via `AppendEntries(prevLogIdx, prevLogTerm, entries[], leaderCommit)`. Followers reject if `prevLog` mismatches, leader backs off and retries.
+ *Safety* — `Election Restriction`: candidate's log must be at least as up-to-date as voters'. `Commitment rule`: a leader may only commit entries from its own term; older-term entries are committed indirectly by being followed by a current-term entry.

The last point is subtle. Figure 8 of the Raft paper shows a scenario where a leader replicates an old-term entry to a majority, crashes, and a different new leader overwrites it. The rule prevents the original leader from declaring it committed before its own term's entry follows.

```
AppendEntries on follower:
    if term < currentTerm:        return (currentTerm, false)
    if term > currentTerm:        currentTerm = term; voted_for = nil
    state = Follower; reset_election_timer()
    if log[prevLogIdx].term != prevLogTerm:    return (currentTerm, false)
    # truncate conflicting suffix, append new
    for i, e in enumerate(entries):
        idx = prevLogIdx + 1 + i
        if idx <= log.last_idx and log[idx].term != e.term:
            log.truncate_from(idx)
        if idx > log.last_idx:
            log.append(e)
    if leaderCommit > commitIndex:
        commitIndex = min(leaderCommit, log.last_idx)
    return (currentTerm, true)
```

Used by etcd, Consul, CockroachDB, TiKV, RethinkDB, Kafka KRaft, Vitess.

== EPaxos

Moraru, Andersen, Kaminsky (2013). *Egalitarian* Paxos: no leader, every replica may propose, conflicts detected and ordered on the fly.

Two cases per command $c$:

- *Fast path* (1 RTT): if a fast quorum agrees on the dependencies of $c$, commit immediately. The basic protocol uses a fast quorum of $2f$ replicas (out of $N = 2f+1$); the optimized variant in the paper reduces it to $f + floor((f+1)\/2)$ — for $N=3$ ($f=1$) that is 2 of 3, for $N=5$ ($f=2$) 3 of 5. The size is chosen so any two fast quorums overlap enough that conflicting dependency sets are always detected during recovery.
- *Slow path* (2 RTT): if dependencies conflict between replicas, run a Paxos-style accept.

Benefits: 1-RTT commits even cross-region when commands don't conflict; load balanced across replicas. Cost: complex execution algorithm (linearize the dependency graph at apply time). Few production systems use it; SiloR and recent academic systems do.

== Flexible Paxos

Howard, Malkhi, Spiegelman (2017). Generalizes quorum intersection: Paxos requires Phase 1 quorum $Q_1$ and Phase 2 quorum $Q_2$ to intersect. The *only* constraint is $Q_1 inter Q_2 != emptyset$ — not that both be majorities.

Implications: smaller fast-path $Q_2$ (e.g., 2 of 5) at the cost of larger leader-change $Q_1$ (e.g., 4 of 5). Trade common-case latency for recovery cost. Used by FoundationDB's transaction log layout and explicitly by FPaxos.

== Generalized Paxos

Lamport (2005). Decides on a *partial order* of commands rather than a total log. Independent (commutative) commands need not be ordered relative to each other; conflicting ones do. Predecessor to EPaxos. Powerful but requires commutativity analysis at the application level — rarely deployed pure, but its ideas live on in CRDT-based stores and lazy replication.

== Byzantine Fault Tolerance

When nodes may lie. Classic result: tolerating $f$ Byzantine faults requires $n >= 3f + 1$ (Lamport, Shostak, Pease 1982). Intuition: a quorum of $2f+1$ excludes $f$ faulty nodes; two such quorums intersect in $f+1$, of which at least one is correct.

=== PBFT

Castro and Liskov (1999). Three phases per request:

```
client -> primary: REQUEST(o, t, c)
primary -> all:    PRE-PREPARE(v, n, REQUEST)
all -> all:        PREPARE(v, n, digest)         // 2f matching => "prepared"
all -> all:        COMMIT(v, n, digest)          // 2f+1 matching => "committed-local"
all -> client:     REPLY(v, t, c, i, r)          // f+1 matching => committed
```

View changes handle primary failures (similar in spirit to Paxos prepare). Authenticated channels via MACs; modern variants use threshold signatures to reduce $O(N^2)$ to $O(N)$ messages per phase.

=== HotStuff

Yin et al. (2019). Chained, three-phase BFT with $O(N)$ communication using threshold signatures. The leader collects $2f+1$ signed votes into a quorum certificate (QC). A block is committed once 3 consecutive QCs extend it. View change is identical in structure to normal operation — the *responsiveness* property: a correct leader can commit at network speed, not at $Delta$ timeout.

Used by Facebook's Diem / Aptos, the Sui blockchain, Cypherium.

=== Tendermint

Kwon, J., Buchman, E. (2014). BFT consensus with immediate finality (no rollback). Steps: propose, prevote, precommit; advance on $2f+1$ votes. Liveness requires a partially synchronous network. Used by Cosmos SDK chains.

=== Nakamoto Consensus

Bitcoin's "longest chain wins" (Nakamoto 2008). Probabilistic finality: a block is "safe" once $k$ confirmations follow ($k=6$ by convention, $approx 60$ min). Tolerates Byzantine nodes if honest miners control $>50%$ hash rate. Asynchronous safety only probabilistically; eventual liveness assuming partial synchrony of the gossip network. Foundation of PoW chains; PoS variants (Algorand, Ouroboros, Ethereum Casper FFG) trade hash rate for staked capital.

== Comparison

#table(
  columns: (auto, 1fr, 1fr, 1fr, 1fr),
  table.header[*Protocol*][*Faults*][*Common-case RTT*][*Throughput*][*Notes*],
  [Paxos / Multi-Paxos], [Crash, $f < n/2$], [1 (after Phase 1)], [Leader-bound], [Workhorse],
  [Raft], [Crash, $f < n/2$], [1], [Leader-bound], [Easier to teach],
  [EPaxos], [Crash, $f < n/2$], [1 (no conflict)], [Balanced], [Complex execution],
  [Flexible Paxos], [Crash, tunable], [1, smaller quorum], [Higher], [Slower recovery],
  [PBFT], [Byzantine, $f < n/3$], [3], [$O(N^2)$ msg], [View change costly],
  [HotStuff], [Byzantine, $f < n/3$], [3 (pipelined)], [$O(N)$ msg], [Used in blockchains],
  [Nakamoto], [Byzantine, $f < n/2$ hash], [Probabilistic], [Low], [PoW chains],
)

== Multi-Group Sharding

Most real systems scale by partitioning state across many consensus groups (Spanner ranges, CockroachDB ranges, TiKV regions, FoundationDB shards). Cross-shard transactions need an additional protocol layer — usually 2PC over the per-shard Paxos groups, giving *Paxos Commit* (Gray and Lamport 2006), which fault-tolerantly replaces 2PC's vulnerable coordinator.

== Practical Engineering

- *Quorum read* vs *leader read* vs *read lease:* leader read with lease is the standard; quorum read costs an RTT but tolerates leader liveness gaps.
- *Witness replicas:* a non-voting replica holds metadata only, providing crash-tolerance without storage cost. CockroachDB and Spanner support them.
- *Pipelining and out-of-order Accepts:* a leader does not wait for slot $i$ to commit before sending slot $i+1$, increasing throughput.
- *Quorum leases* (Moraru et al. 2014): cache leases on quorum nodes so reads at any of them are linearizable without contacting the leader.

== Failure Scenarios to Reason About

+ *Leader isolated after committing entry $E$:* a new majority elects a new leader. The new leader's Phase 1 (or Raft commitment rule) ensures $E$ is preserved.
+ *Leader receives quorum acks but crashes before responding to client:* client retries see $E$ committed (idempotency required) or absent (new leader chose differently). Use unique request IDs.
+ *Network partition with no majority side:* both sides stall — safety holds.
+ *Slow follower lags by GBs of log:* leader installs snapshot via `InstallSnapshot` RPC.

== Further Reading

Lamport, L. (1998). "The Part-Time Parliament." TOCS.

Lamport, L. (2001). "Paxos Made Simple." SIGACT News.

Ongaro, D., Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm." USENIX ATC.

Moraru, I., Andersen, D., Kaminsky, M. (2013). "There Is More Consensus in Egalitarian Parliaments." SOSP.

Howard, H., Malkhi, D., Spiegelman, A. (2017). "Flexible Paxos: Quorum Intersection Revisited." OPODIS.

Castro, M., Liskov, B. (1999). "Practical Byzantine Fault Tolerance." OSDI.

Yin, M., Malkhi, D., Reiter, M., Gueta, G., Abraham, I. (2019). "HotStuff: BFT Consensus with Linearity and Responsiveness." PODC.

Lamport, L., Shostak, R., Pease, M. (1982). "The Byzantine Generals Problem." TOPLAS.

Lamport, L., Malkhi, D., Zhou, L. (2009). "Vertical Paxos and Primary-Backup Replication." PODC.

Gray, J., Lamport, L. (2006). "Consensus on Transaction Commit." TODS.

Nakamoto, S. (2008). "Bitcoin: A Peer-to-Peer Electronic Cash System."

Buchman, E. (2016). "Tendermint: Byzantine Fault Tolerance in the Age of Blockchains." MS thesis.

#import "../template.typ": xref

= Gossip Protocols <gossip>

Epidemic algorithms spread information through a cluster the way a rumour spreads through a crowd: each node periodically contacts a small random set of peers, and the information reaches every node in $O(log N)$ rounds with high probability. This chapter covers the three fundamental gossip variants, convergence analysis, anti-entropy repair, the SWIM failure detector, and how CRDTs pair naturally with gossip dissemination.

*See also:* #xref("distributed-systems", "failure-detection", label: "Failure Detection"), #xref("distributed-systems", "consensus-deep-dive", label: "Consensus Deep Dive"), #xref("distributed-systems", "leader-election-and-leases", label: "Leader Election and Leases"), #xref("database", "consensus-and-replication", label: "Consensus and Replication") (database-side framing).

== Epidemic Spreading Models

The epidemiology analogy is precise. Map *susceptible* to "node has not seen value $v$", *infected* to "node holds $v$ and will forward it", and *removed* to "node holds $v$ but has stopped forwarding." The three gossip strategies correspond to different rules for transitioning between states.

=== Push Gossip

In each gossip round, every infected node picks $k$ random peers (*fanout*) and sends them its state. Susceptible peers become infected immediately.

```
loop every T_gossip:
    peers = random_sample(membership_list, fanout=k)
    for p in peers:
        send(p, my_state)
```

*Round analysis.* Let $I(t)$ be infected count after round $t$, $S = N - I(t)$ susceptible. Each infected node infects each susceptible independently with probability $k/N$. In expectation:

$ I(t+1) = I(t) + I(t) dot (N - I(t)) dot k / N $

For small $I(t)$, growth is exponential; once $I(t) approx N/2$, remaining susceptible nodes are hit with probability $approx 1 - (1 - k/N)^(N/2) approx 1 - e^(-k/2)$. Full convergence requires $O(log N)$ rounds with fanout $k = O(log N)$.

=== Pull Gossip

Each node periodically pulls from $k$ random peers: "do you have anything newer than my digest $D$?" The *pull* variant is more efficient in the late stages when few susceptible nodes remain, because a susceptible node finds an infected peer in $O(N / I(t))$ expected probes rather than relying on a lucky push.

=== Push-Pull Gossip

Combine both: the initiator pushes its digest *and* receives a push back. Each exchange synchronises both parties, halving divergence per round. Push-pull converges at the same $O(log N)$ asymptotic rate as pure push but with a smaller constant; it is the dominant choice in practice (Cassandra, Consul).

== Convergence Guarantees

*Theorem (Karp et al. 2000).* With fanout $k = c dot log N$ for any constant $c > 1$, push gossip infects all $N$ nodes within $(1 + epsilon) log N$ rounds with probability $1 - N^(-(c-1))$.

The *residue* — number of nodes that never received the message — is $O(1)$ for $k = log N$, and falls to zero with overwhelming probability for $k = 2 log N$. Network bandwidth consumed: $O(N log N)$ messages total, each of size $O(|"payload"|)$, versus $O(N)$ for a single broadcast that risks a single point of failure.

== Anti-Entropy

Pure gossip *propagates new writes* quickly but does not repair *existing divergence*. *Anti-entropy* sessions compare full state and reconcile differences.

=== Merkle Tree Reconciliation

Partition the keyspace into a binary tree of hash buckets. Two nodes compare root hashes; if they differ, they descend the tree level by level until they identify the differing leaf ranges. Only those ranges are exchanged. Cost: $O(log M)$ round-trips for $M$ keys if divergence is small.

```
def anti_entropy_session(local, remote):
    if local.root_hash() == remote.root_hash():
        return  # in sync
    for level in range(tree_depth):
        local_hashes = local.hashes_at_level(level)
        remote_hashes = remote.hashes_at_level(level)
        diff_buckets = [b for b in all_buckets(level)
                        if local_hashes[b] != remote_hashes[b]]
        if not diff_buckets:
            break
    exchange(local, remote, diff_buckets)
```

Cassandra runs anti-entropy repairs as scheduled `nodetool repair` jobs. DynamoDB runs a background anti-entropy process continuously. Riak uses an active anti-entropy (AAE) subsystem with per-vnode Merkle trees persisted to disk.

=== Read Repair

As a low-cost complement, coordinators performing reads can compare replicas' digests and issue *read repair* writes to stale replicas asynchronously (Cassandra) or synchronously (configurable). Read repair costs zero extra network round-trips in the common case.

== SWIM Failure Detector

Scalable Weakly-consistent Infection-style Membership (*SWIM*, Das et al. 2002) replaces heartbeat-to-all ($O(N)$ load per node) with gossip-based membership updates, achieving $O(1)$ load per node while bounding false-positive failure rates.

=== Probe and Indirect Probe

```
loop every T_probe:
    m = random_member()
    send PING to m
    wait T_ping_timeout
    if no ACK received:
        K = random_subset(members, k=3)
        send PING-REQ(m) to each node in K
        wait T_indirect_timeout
        if no ACK forwarded:
            mark m as SUSPECTED
            gossip SUSPECT(m)
```

- *Direct probe* catches crash failures in one round.
- *Indirect probes* via $k$ intermediaries distinguish crashes from transient network partitions. A legitimate crash requires $k+1$ paths to all be broken.

=== Suspicion and Confirmation

A *SUSPECTED* member has a configurable suspicion timeout. If it sends a counter-gossip (*ALIVE* message with a higher incarnation number), it clears itself. After timeout expiry with no rebuttal, the node is declared *DEAD* and its entry is gossip-propagated with a *CONFIRM(dead)* tag.

*Incarnation numbers* prevent stale ALIVE messages from reviving confirmed-dead nodes. Each node increments its incarnation on startup and when rebutting a false suspicion.

=== Gossip Dissemination Layer

Membership changes (SUSPECT, ALIVE, DEAD, JOIN) piggyback on existing probe messages, limiting bandwidth while ensuring $O(log N)$ dissemination. Each piggyback entry carries a *transmit count*; entries are dropped after $lambda log N$ retransmissions.

Real implementations: Consul's memberlist library, Serf (HashiCorp), Akka Cluster (Phi Accrual detector layered on gossip), ScyllaDB.

== CRDTs as Gossip-Friendly State

*Conflict-free Replicated Data Types* ($"CRDT"$s) are data types whose merge operation is commutative, associative, and idempotent — exactly the properties gossip dissemination can guarantee. A replica need not receive updates in order, and duplicates are harmless.

=== G-Counter

A grow-only counter over $N$ nodes. Each node $i$ maintains a vector $V$ of length $N$.

$ "increment"(i): V[i] <- V[i] + 1 $
$ "value"(): sum_i V[i] $
$ "merge"(V_a, V_b): V[i] <- max(V_a [i], V_b [i]) $

Gossip propagates the full vector; merge is component-wise max. Convergence is guaranteed when all vectors have propagated.

=== OR-Set (Observed-Remove Set)

Supports add and remove without the "remove wins" vs "add wins" ambiguity. Each element is tagged with a unique token on add; remove carries the token set. An element is in the set iff any of its tokens survive.

```
add(e):
    t = unique_token()
    elements[e].add(t)
    gossip ADD(e, t)

remove(e):
    tokens = elements[e].copy()  # observed tokens
    for t in tokens: elements[e].discard(t)
    gossip REMOVE(e, tokens)

merge(A, B):
    for e in A.elements | B.elements:
        result[e] = A.elements[e] | B.elements[e]
        # tokens removed locally are already absent from elements[e];
        # union of surviving tokens is the correct state-based merge
```

=== LWW-Register

*Last-Write-Wins Register*: each write carries a timestamp; merge picks the higher timestamp.

$ "merge"(r_a, r_b) = cases(r_a "if" r_a."ts" > r_b."ts", r_b "otherwise") $

Simple but *timestamp accuracy matters*: NTP jitter can cause lost writes. Hybrid Logical Clocks (HLCs) combine physical time with a logical counter, bounding drift while preserving causal order (see `distributed-systems/time-and-order.typ`).

== Real Systems

=== Cassandra Gossip

Each node runs a gossip loop every 1 second, exchanging *endpoint state* (generation, version, application states like `LOAD`, `STATUS`, `TOKENS`) with 1–3 random peers. State is versioned per key; only higher-version values overwrite. Cassandra gossip drives token ring membership, schema propagation, and rack/datacenter topology.

=== Consul Memberlist

HashiCorp's `memberlist` library (Go) is a standalone SWIM implementation used by Consul, Nomad, and Serf. It supports both UDP probes and TCP fall-back for large payloads. The gossip port (default 8301 LAN, 8302 WAN) carries membership and user-event piggybacks. `serf` adds a higher-level event bus over memberlist.

=== Serf

Serf extends memberlist with *events* (broadcast arbitrary key-value payloads) and *queries* (request-response over gossip, with a fan-in response collector). Use cases: deployment orchestration, health check propagation, dynamic load-balancer membership.

== Performance Tuning

- *Fanout $k$:* increase for faster convergence at the cost of bandwidth. Cassandra defaults to 3.
- *Gossip interval:* 1 s is typical; lower for faster detection, higher for WAN cost reduction.
- *Message size:* apply compression (snappy, zstd) on payloads $>$ 1 KB. Merkle digests are cheap.
- *Indirect probe count:* 3–5 is typical. Too few increases false-positive rate; too many wastes bandwidth.
- *Suspicion multiplier:* scale suspicion timeout with cluster size to avoid cascading false suspicions under load spikes.

== Further Reading

Demers, A., et al. (1987). "Epidemic Algorithms for Replicated Database Maintenance." PODC.

Das, A., Gupta, I., Motivala, A. (2002). "SWIM: Scalable Weakly-consistent Infection-style Process Group Membership Protocol." DSN.

Karp, R., Schindelhauer, C., Shenker, S., Vocking, B. (2000). "Randomized Rumor Spreading." FOCS.

Shapiro, M., Preguica, N., Baquero, C., Zawirski, M. (2011). "Conflict-Free Replicated Data Types." SSS.

van Renesse, R., Minsky, Y., Hayden, M. (1998). "A Gossip-Style Failure Detection Service." Middleware.

Lakshman, A., Malik, P. (2010). "Cassandra: A Decentralized Structured Storage System." SIGOPS OSR.

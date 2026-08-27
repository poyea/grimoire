#import "../template.typ": xref

= Failure Detection <failure-detection>

A failure detector is an oracle that tells a process which peers it believes to have crashed. It is the abstraction that lets asynchronous systems make progress despite FLP. The art is balancing *completeness* (every crashed process is eventually suspected) against *accuracy* (no correct process is suspected forever).

*See also:* _Introduction_ (FLP), #xref("distributed-systems", "leader-election-and-leases", label: "Leader Election and Leases"), #xref("distributed-systems", "gossip", label: "Gossip Protocols"), #xref("distributed-systems", "coordination-services", label: "Coordination Services") (ZooKeeper session expiry).

== Chandra–Toueg Classes

Chandra and Toueg (1996) classify failure detectors by completeness and accuracy:

#table(
  columns: (auto, 1fr, 1fr),
  table.header[*Class*][*Completeness*][*Accuracy*],
  [$P$ — Perfect], [Strong (every crash detected)], [Strong (no false positives)],
  [$S$ — Strong], [Strong], [Weak (some correct process never suspected)],
  [$diamond.stroked P$ — Eventually Perfect], [Strong], [Eventually strong],
  [$diamond.stroked S$ — Eventually Strong], [Strong], [Eventually weak],
  [$Omega$ — Eventual Leader], [N/A], [Eventually all agree on one correct leader],
)

Key theorem: $diamond.stroked S$ is the weakest failure detector that solves consensus with a majority of correct processes. $Omega$ is equivalent. This grounds the engineering reality: Raft's leader election is an $Omega$ implementation.

== Heartbeats

The simplest detector: each peer sends a heartbeat every $T_h$; a peer is suspected if no heartbeat arrives for $T_d > T_h$.

Two failure modes:

- *False positive:* GC pause, network blip. Cost: spurious failover, leader thrashing.
- *False negative:* asymmetric partition where heartbeats flow but data does not.

The fixed-threshold approach has no principled way to trade these. In practice $T_d$ is chosen as a multiple ($3$--$10 times$) of $T_h$, padded by an empirical safety factor.

```python
class FixedHeartbeat:
    def __init__(self, interval, threshold):
        self.last = {}
        self.t_h = interval
        self.t_d = threshold

    def on_heartbeat(self, peer, now):
        self.last[peer] = now

    def suspect(self, peer, now):
        return now - self.last.get(peer, 0) > self.t_d
```

== Phi-Accrual

Hayashibara et al. (2004) replaced the binary suspect/trust with a continuous *suspicion level* $phi(t)$. Given a sliding window of inter-arrival samples $X_1, ..., X_n$ modeled as normal $N(mu, sigma^2)$:

$ phi(t) = -log_10 P("next interval" > t - t_("last")) $

The application picks a threshold (Akka default $phi >= 8$, Cassandra $phi >= 8$ also). Properties:

- Self-tuning: adapts $mu$ and $sigma$ to network conditions.
- Smooth: a momentary delay raises $phi$ gradually rather than flipping.
- Tunable per application: latency-critical paths use low threshold, batch jobs use higher.

```python
import math, statistics
class PhiAccrual:
    def __init__(self, window=100):
        self.intervals = []
        self.window = window
        self.last = None

    def heartbeat(self, now):
        if self.last is not None:
            self.intervals.append(now - self.last)
            self.intervals = self.intervals[-self.window:]
        self.last = now

    def phi(self, now):
        if not self.intervals or self.last is None:
            return 0.0
        mu = statistics.mean(self.intervals)
        sigma = max(statistics.pstdev(self.intervals), 1e-3)
        t = now - self.last
        # P(X > t) under N(mu, sigma) using complementary error function
        z = (t - mu) / sigma
        p = 0.5 * math.erfc(z / math.sqrt(2))
        return -math.log10(max(p, 1e-300))
```

Cassandra uses $phi$-accrual for gossip-based detection; Akka Cluster uses it for cluster membership.

== SWIM

Das, Gupta, Motivala (2002) — Scalable Weakly-consistent Infection-style Membership. Each round, every node:

+ Picks a random target $T$, sends `PING`.
+ If no `ACK` within timeout, picks $k$ random witnesses, asks them to `PING-REQ($T$)`.
+ If still no `ACK`, marks $T$ as suspect; gossips the suspicion.
+ Suspect transitions to dead after timeout if no `ALIVE` refutation.

Key properties:

- *Constant load per node* regardless of cluster size — gossip rounds carry membership deltas piggyback on PING/ACK.
- *Indirect probing* eliminates most false positives from one-way network blackouts.
- *Suspicion + dissemination* gives a process time to defend itself if mistakenly accused.

```
on_round(self):
    target = random.choice(self.members - {self})
    if not self.ping(target, timeout=T1):
        witnesses = random.sample(self.members - {self, target}, k)
        if not any(self.ping_req(w, target, timeout=T2) for w in witnesses):
            self.gossip(SUSPECT(target, incarnation=target.inc))
```

Failure detection time bound: $O(T_("round"))$, dissemination time: $O(log N)$ rounds. Used by Hashicorp's memberlist (Consul, Serf, Nomad), Cassandra (variant), AWS Auto Scaling.

== Lifeguard

Dadgar et al. (Hashicorp 2018) extend SWIM with three refinements:

+ *Self-awareness* (Local Health Multiplier): a node noticing it has been missed frequently increases its own probe rate and reduces its suspicion tolerance — recognizes it might be the slow one.
+ *Dogpile avoidance:* refute suspicions with monotonic *incarnation numbers* that survive restarts.
+ *Buddy system:* prefer probing nodes that recently suspected this node, accelerating refutation.

Empirically reduces false positives by 98%+ under partial network degradation. Default in modern Consul / Nomad.

== Asymmetric and Gray Failures

A class of bugs invisible to simple detectors:

- *One-way partition:* A can send to B, B cannot send to A. Heartbeats fail in one direction but the affected node may still be serving clients.
- *Slow node ("limp mode"):* disk SMART errors, NIC FEC retransmits, microcode bugs cause 100--1000$times$ latency without outright failure.
- *Gray failure* (Huang et al. 2017): the node is alive from the membership service's view but unable to make application progress.

Mitigations:

- *Bidirectional health checks:* every link tested in both directions.
- *Application-level health* (`/healthz` and `/readyz` semantics): query an actual codepath, not just TCP.
- *Outlier ejection:* eject nodes whose p99 latency exceeds cluster median by $k sigma$, as Envoy's outlier detector does.
- *Hedged requests* (Dean and Barroso 2013): send to a backup after $p_95$, cancel the slower.

== Quorum Failure Detectors

For consensus systems, the relevant question is not "is X alive?" but "do enough peers form a quorum?" Quorum-based detectors expose: `is_quorum_reachable(group)`. They are robust to one-way partitions because reachability is computed over message-exchange evidence within a recent window.

== Practical Pitfalls

- *Coordinated omission:* Load generators that pause under backpressure underreport timeout rates -- if the sender is also slow, missed timeouts are never measured. Gil Tene's HdrHistogram analysis (2015) shows this can hide 10--100x the true tail latency. Fix: use open-loop load generation where probe intervals are wall-clock-fixed, not arrival-rate-adjusted.

- *NTP step jumps:* A monotonic-clock-based failure detector is immune to NTP steps, but wall-clock-based heartbeat timestamps are not. A 100 ms backward NTP step makes heartbeats appear to arrive in the future, triggering false positives. Fix: always use `CLOCK_MONOTONIC` for interval measurement; record wall time only for human-readable logs.

- *GC pauses:* JVM STW collection pauses of 500 ms--4 s (G1GC in worst case) routinely exceed 200--500 ms FD thresholds. G1GC improved this significantly; ZGC and Shenandoah target under 10 ms even for large heaps. Go's GC achieves under 1 ms pauses at 100 GB heaps; Rust avoids GC pauses entirely. Set the FD threshold above 2x your worst observed GC pause at p99.

- *Container CPU throttling:* CFS bandwidth control (`cpu.cfs_period_us` / `cpu.cfs_quota_us`) can suspend a container for a full 100 ms period if it exhausts its quota. A heartbeat thread in a throttled container appears dead to the cluster even though the process is healthy. Fix: give FD/heartbeat processes dedicated CPU shares or set `cpu.cfs_quota_us = -1` (unlimited) for lease-holding processes.

== Tuning Example

Suppose RTT $approx 1$ ms with $sigma = 0.5$ ms on a LAN. To keep MTTR $<5$ s at $99.9%$ accuracy under occasional 200 ms network blips:

- Heartbeat interval $T_h = 200$ ms.
- $phi$ threshold $= 8$, window 100 samples.
- Expected detection time after crash: $approx 3 times T_h = 600$ ms (3 missed heartbeats push $phi > 8$).
- False positive rate under 200 ms blips: tail beyond $mu + (8 / log_10 e) sigma$ from a heavy-tailed empirical distribution; window auto-adjusts after first occurrences.

== Comparison

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  table.header[*Detector*][*Strength*][*Weakness*][*Best fit*],
  [Fixed heartbeat], [Trivial], [Brittle to network jitter], [Stable LAN],
  [$phi$-accrual], [Self-tuning], [Sensitive to window size], [WAN, mixed jitter],
  [SWIM], [$O(1)$ per-node load], [Slower detection], [Large clusters],
  [Lifeguard], [Robust to gray failures], [More complex state], [Production at scale],
  [Quorum-based], [Tied to safety property], [Coupled to consensus], [Replicated state machines],
)

== Further Reading

Chandra, T., Toueg, S. (1996). "Unreliable Failure Detectors for Reliable Distributed Systems." JACM.

Chandra, T., Hadzilacos, V., Toueg, S. (1996). "The Weakest Failure Detector for Solving Consensus." JACM.

Hayashibara, N., Défago, X., Yared, R., Katayama, T. (2004). "The $phi$-Accrual Failure Detector." SRDS.

Das, A., Gupta, I., Motivala, A. (2002). "SWIM: Scalable Weakly-consistent Infection-style Process Group Membership Protocol." DSN.

Dadgar, A., Phillips, J., Currey, J. (2018). "Lifeguard: SWIM-ing with Situational Awareness." HashiCorp.

Huang, P. et al. (2017). "Gray Failure: The Achilles' Heel of Cloud-Scale Systems." HotOS.

Dean, J., Barroso, L. (2013). "The Tail at Scale." CACM.

Gupta, I., Chandra, T., Goldszmidt, G. (2001). "On Scalable and Efficient Distributed Failure Detectors." PODC.

= Time and Order

Distributed systems have no global "now". What they have are *clocks* (each imperfect) and *messages* (which establish happens-before relationships). The art of this chapter is converting one into the other: using clocks to approximate causality and using causality to compensate for clock skew.

*See also:* _Introduction_, _Consensus Deep Dive_, _Replication Protocols_, and _Consensus and Replication_ (TrueTime in Spanner).

== Physical Clocks

A physical clock is a hardware oscillator plus a counter. Three properties matter:

- *Resolution:* the smallest measurable increment (1 ns on TSC, 1 ms on `gettimeofday`).
- *Accuracy:* offset from "true" UTC at a moment.
- *Stability (drift):* rate of change of offset. Crystal oscillators drift by 1--100 ppm; OCXO and atomic clocks by $<10^(-9)$.

A drift of 100 ppm equals $approx 8.6$ s/day. Without synchronization, free-running clocks diverge by minutes per week.

=== NTP

Network Time Protocol (Mills 1985; standardised in RFC 5905, 2010) corrects offset using four timestamps in a request-response exchange:

```
t1 = client send time      (local clock)
t2 = server receive time   (server clock)
t3 = server send time      (server clock)
t4 = client receive time   (local clock)

offset = ((t2 - t1) + (t3 - t4)) / 2
delay  = (t4 - t1) - (t3 - t2)
```

Assumes symmetric network delay; asymmetric paths bias offset. NTP achieves $approx 1$--10 ms accuracy over WAN, $approx 0.1$--1 ms over LAN. The control loop (PLL/FLL) smooths step changes to avoid time jumps, but a poorly tuned NTP daemon can leave the clock unsynchronized for hours after a step.

=== PTP

Precision Time Protocol (IEEE 1588) uses hardware timestamps on NIC ports and a master-slave hierarchy with boundary clocks. Achieves sub-microsecond accuracy on switched LANs that support transparent or boundary clocks. Widely deployed in finance (MiFID II requires 100 us traceability), telecom (5G fronthaul), and Spanner-style databases.

=== TSC and Monotonic Clocks

`CLOCK_MONOTONIC` on Linux is derived from the CPU TSC plus kernel offset. It never goes backwards but is not synchronized to UTC; it may jump on VM migration. Always use monotonic clocks for measuring intervals (deadlines, RTTs) and wall clocks only for human-facing timestamps.

== Lamport Clocks

Lamport (1978) defined the *happens-before* relation $arrow.r.hook$ on events:

+ Same process, $e_1$ before $e_2$ $==> e_1 arrow.r.hook e_2$.
+ $e_1$ is `send(m)` and $e_2$ is `recv(m)` $==> e_1 arrow.r.hook e_2$.
+ Transitive closure.

Events not related by $arrow.r.hook$ are *concurrent*. A Lamport timestamp $L(e)$ assigns integers such that $e_1 arrow.r.hook e_2 ==> L(e_1) < L(e_2)$. Algorithm:

```
on local event:    L += 1
on send(m):        L += 1; piggyback L on m
on recv(m, L_m):   L = max(L, L_m) + 1
```

Lamport timestamps give a total order consistent with causality (tie-break by process ID), but $L(e_1) < L(e_2)$ does *not* imply causality. Useful for mutual exclusion, ordering events in a log, but cannot detect concurrent updates.

== Vector Clocks

Vector clocks (Fidge 1988, Mattern 1989) detect concurrency. Each process $i$ maintains $V_i$, an $N$-vector:

```
on local event at i:     V_i[i] += 1
on send(m) at i:         V_i[i] += 1; send V_i with m
on recv(m, V_m) at j:    V_j[k] = max(V_j[k], V_m[k]) for all k
                          V_j[j] += 1
```

Comparison: $V <= V'$ iff $forall k: V[k] <= V'[k]$. $V < V'$ iff $V <= V'$ and $V != V'$. Concurrent: neither $V < V'$ nor $V' < V$.

*Space cost:* $O(N)$ per timestamp where $N$ is the number of writers. For client-facing systems where every device is a writer, this is unbounded. Dotted version vectors and interval tree clocks (ITCs) address this.

```python
class VectorClock:
    def __init__(self, node_id, n):
        self.id = node_id
        self.v = [0] * n
    def tick(self):
        self.v[self.id] += 1
    def update(self, other):
        self.v = [max(a, b) for a, b in zip(self.v, other.v)]
        self.tick()
    def compare(self, other):
        le = all(a <= b for a, b in zip(self.v, other.v))
        ge = all(a >= b for a, b in zip(self.v, other.v))
        if le and ge: return "equal"
        if le: return "before"
        if ge: return "after"
        return "concurrent"
```

== Version Vectors and Dotted Version Vectors

Dynamo's version vectors track *per-key* writers; pruning is needed to bound size. Dotted Version Vectors (Preguiça et al. 2010) decouple causal context from current version: a *dot* $(i, n)$ uniquely identifies a single write event, while a vector summarizes all dominated writes. Solves the false-conflict problem that plain VVs suffer in client-server settings (Riak uses DVVs).

== Hybrid Logical Clocks

HLC (Kulkarni et al. 2014) combines physical and logical components:

```
hlc = (pt, l)   // pt = physical time, l = logical counter

on local event:
    pt_now = now()
    if pt_now > pt:    pt = pt_now; l = 0
    else:              l += 1

on send/recv with received (pt_m, l_m):
    pt_new = max(pt, pt_m, now())
    if pt_new == pt == pt_m:    l = max(l, l_m) + 1
    elif pt_new == pt:          l += 1
    elif pt_new == pt_m:        l = l_m + 1
    else:                       l = 0
    pt = pt_new
```

Properties:

- Captures causality like a Lamport clock.
- HLC stays close to physical time (bounded drift = max NTP skew).
- 64-bit representation (e.g., 48 bits ms + 16 bits logical) fits in a timestamp column.

Used by CockroachDB, MongoDB (cluster time), YugabyteDB, FaunaDB. The bounded drift property lets HLC drive snapshot reads: a transaction at HLC $T$ can read from any replica whose HLC has advanced past $T$.

== TrueTime

Spanner's TrueTime (Corbett et al. 2012) exposes `TT.now()` returning an interval $[t_("earliest"), t_("latest")]$ guaranteed to contain absolute time. Achieved via GPS receivers and atomic clocks in every datacenter, with Marzullo-style outlier rejection at the TrueTime daemon.

Spanner uses TrueTime for:

- *Commit wait:* after assigning commit timestamp $s$, wait until $"TT.now().earliest" > s$ before releasing locks. Guarantees external consistency.
- *Snapshot reads:* a read at timestamp $s$ can be served by any replica whose safe time $>= s$.

Reported $epsilon$ (half-width) is typically $<10$ ms; commit wait of $2 epsilon = 14$ ms is amortized against WAN-RTT Paxos commits.

== Closed Timestamps

CockroachDB's closed timestamp protocol (Taft et al. 2020) inverts TrueTime: instead of waiting for the clock, the leader periodically announces "no transaction with timestamp $<= T$ will commit on this range from now on." Followers can serve reads at $<= T$ without contacting the leader.

Tradeoff: stale reads up to the closed-timestamp interval (default 3 s); no clock-bounded write amplification.

== Causal Broadcast

A broadcast primitive that preserves happens-before. Algorithm (Birman 1991):

```
on broadcast(m) at i:
    V_i[i] += 1; piggyback V_i
    send (m, V_i) to all

on receipt of (m, V_m) at j:
    wait until V_m[i] = V_j[i] + 1
            and V_m[k] <= V_j[k] for all k != i
    deliver m to application
    V_j[i] = V_m[i]
```

The wait enforces FIFO from sender $i$ and "no missed dependencies" from other senders.

== Snapshot Algorithms

Chandy–Lamport (1985) computes a consistent global snapshot in a strongly-connected network with FIFO channels:

```
Initiator:
    record own state
    send MARKER on every outgoing channel

On receipt of MARKER on channel c:
    if first marker:
        record state
        state(c) = empty
        send MARKER on every outgoing channel
        start recording incoming on all other channels
    else:
        state(c) = messages recorded on c since this process recorded its state
```

The snapshot may not correspond to any global instant, but it is *causally consistent*: every recorded message was sent before the corresponding receive was recorded. Used to compute distributed deadlock, termination, and garbage collection (Flink's checkpointing extends it for streaming).

== Comparison Table

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  table.header[*Scheme*][*Captures causality*][*Cost*][*Physical-time correlation*],
  [Lamport], [Partial (total order, not causal)], [$O(1)$], [None],
  [Vector clock], [Yes], [$O(N)$], [None],
  [Dotted VV], [Yes], [$O("active writers")$], [None],
  [HLC], [Yes], [$O(1)$], [Bounded by skew],
  [TrueTime], [Implicit via wait], [$O(1)$ + hardware], [Tight bound $epsilon$],
  [Closed timestamp], [Via causal wait], [$O(1)$ msg overhead], [Bounded interval],
)

== Why Clock Skew Causes Real Bugs

1. *Spanner without commit wait:* a later transaction could observe a timestamp that has not yet passed on another replica, breaking external consistency.
2. *Cassandra last-write-wins:* a write with a future timestamp from a clock-skewed client suppresses subsequent correct writes for the skew duration.
3. *Kafka log retention by time:* a clock jump deletes recent data.
4. *JWT expiry:* skew between issuer and validator causes spurious auth failures (RFC 7519 recommends a small leeway).

Mitigations: NTP monitoring, capped time deltas, monotonic checks on incoming timestamps, idempotency keys decoupled from time.

== Further Reading

Lamport, L. (1978). "Time, Clocks, and the Ordering of Events in a Distributed System." CACM 21(7).

Fidge, C. (1988). "Timestamps in Message-Passing Systems That Preserve the Partial Ordering." Australian Computer Science Communications.

Mattern, F. (1989). "Virtual Time and Global States of Distributed Systems."

Kulkarni, S., Demirbas, M., Madappa, D., Avva, B., Leone, M. (2014). "Logical Physical Clocks." OPODIS.

Corbett, J. et al. (2012). "Spanner: Google's Globally-Distributed Database." OSDI.

Taft, R. et al. (2020). "CockroachDB: The Resilient Geo-Distributed SQL Database." SIGMOD.

Mills, D. (2006). _Computer Network Time Synchronization: The Network Time Protocol_. CRC Press.

Chandy, K.M., Lamport, L. (1985). "Distributed Snapshots: Determining Global States of Distributed Systems." TOCS.

Preguiça, N., Baquero, C., Almeida, P., Fonte, V., Gonçalves, R. (2010). "Dotted Version Vectors: Logical Clocks for Optimistic Replication." arXiv.

Schwarz, R., Mattern, F. (1994). "Detecting Causal Relationships in Distributed Computations: In Search of the Holy Grail." Distributed Computing.

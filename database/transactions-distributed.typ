= Distributed Transactions

A distributed transaction spans multiple nodes (shards, replicas, or data centers). Achieving ACID properties in this setting requires *atomic commit protocols* (2PC, 3PC) and *distributed concurrency control*. The fundamental tension: coordination overhead vs consistency guarantees.

*See also:* _database/concurrency-control.typ_, _database/consensus-and-replication.typ_, _database/isolation-and-consistency-models.typ_

== Two-Phase Commit (2PC)

2PC achieves atomic commit across multiple participants. One node acts as the *coordinator*; others are *participants* (each holding a shard of the data).

```
Phase 1: Prepare (Voting)
  Coordinator → all: PREPARE
  Each participant:
    - Execute txn locally, acquire all locks
    - Force WAL to stable storage (can now recover)
    - Reply YES (will commit) or NO (abort)

Phase 2: Commit/Abort
  If all said YES:
    Coordinator writes COMMIT to its own log (point of no return)
    Coordinator → all: COMMIT
    Participants: apply changes, release locks, reply ACK
  Else:
    Coordinator → all: ABORT
    Participants: undo changes, release locks
```

```python
# 2PC coordinator (simplified)
class Coordinator:
    def __init__(self, participants):
        self.participants = participants

    def run(self, txn) -> bool:
        # Phase 1: gather votes
        votes = []
        for p in self.participants:
            try:
                vote = p.prepare(txn)   # blocks until participant is ready
                votes.append(vote)
            except Exception:
                votes.append(False)

        decision = all(votes)

        # Log decision before sending (durability guarantee)
        self._log_decision(txn.id, decision)

        # Phase 2: broadcast decision
        for p in self.participants:
            if decision:
                p.commit(txn.id)
            else:
                p.abort(txn.id)

        return decision
```

*2PC failure modes:*

#table(
  columns: (auto, auto),
  [*Failure scenario*], [*Behavior*],
  [Participant crashes before PREPARE], [Coordinator times out → abort],
  [Participant crashes after YES vote],  [Participant asks coordinator on recovery (uncertain txn)],
  [Coordinator crashes after PREPARE, before COMMIT], [Participants block indefinitely holding locks],
  [Coordinator crashes after logging COMMIT], [Participants ask each other for decision (cooperative recovery)],
)

*The blocking problem:* if the coordinator crashes after participants have voted YES but before sending COMMIT, participants are stuck holding locks until the coordinator recovers. This is the fundamental limitation of 2PC — it is a *blocking* protocol.

== Three-Phase Commit (3PC)

Adds a `PRE-COMMIT` phase that allows participants to infer the commit decision even if the coordinator fails. Not used in practice because it assumes a synchronous network (no arbitrary message delays), which distributed systems don't guarantee.

== Paxos Commit

Lamport (2006): replace each participant's log write with a Paxos instance. Survives coordinator failure without blocking, at the cost of more messages (2F+1 acceptors per shard for F fault tolerance).

== Calvin: Deterministic Concurrency Control

Calvin (Thomson et al. 2012): avoid 2PC entirely by pre-ordering all transactions globally before execution.

```
Calvin architecture:
  1. Sequencer: batches incoming txns, assigns global order in log
  2. Scheduler: each shard reads its portion of the log, executes in order
  3. No locks needed across shards — deterministic execution ensures no conflicts

Key insight: if all shards agree on the input order, they will arrive at
the same output state independently. Coordination happens at sequencing, not execution.
```

*Throughput:* Calvin achieves ~670K txns/sec on a 3-node cluster (CRDT-free) — no distributed lock contention.

*Limitation:* transactions must declare their read/write sets upfront (deterministic). Cannot handle "read-then-decide-what-to-write" patterns without a round-trip to the sequencer.

== Spanner: External Consistency with TrueTime

Spanner (Corbett et al. 2012) achieves *external consistency* (linearizability at the transaction level) globally across data centers using GPS/atomic clock uncertainty bounds.

*TrueTime API:*

```
TT.now()   → [earliest, latest]   (time interval, typically ε ≤ 7ms wide)
TT.after(t) → bool                (is t definitely in the past?)
TT.before(t) → bool               (is t definitely in the future?)
```

*Commit wait:* before a read-write transaction commits with timestamp $s$, the coordinator waits until `TT.after(s)` — ensuring the commit timestamp is unambiguously in the past before any client can observe it.

```
Commit protocol (read-write txn):
  1. Client reads at snapshot ts = TT.now().latest (global snapshot)
  2. Prepare phase (2PC across shards)
  3. Choose commit_ts = max(TT.now().latest, prepare_ts_max + 1)
  4. Apply Paxos log entry at each shard (durable)
  5. Wait until TT.after(commit_ts)  ← commit wait (≤ 2ε ≈ 14ms)
  6. Release locks, reply to client

Result: all future transactions see commit_ts in the past → linearizable
```

*Read-only transactions:* no locks, no 2PC. Pick `ts = TT.now().latest`; read from any replica whose applied log covers ts. Often served locally within a datacenter.

```python
# Simplified TrueTime simulation
import time, random

class TrueTime:
    def __init__(self, epsilon_ms=7):
        self.epsilon = epsilon_ms / 1000.0

    def now(self):
        t = time.time()
        return t - self.epsilon, t + self.epsilon  # (earliest, latest)

    def after(self, t) -> bool:
        return time.time() - self.epsilon > t

tt = TrueTime()
earliest, latest = tt.now()
# commit_ts = latest; wait until TT.after(latest)
commit_ts = latest
while not tt.after(commit_ts):
    time.sleep(0.001)
# Now commit_ts is unambiguously in the past
```

== CockroachDB: HLC-Based Distributed Transactions

CockroachDB uses *Hybrid Logical Clocks* (HLC) instead of GPS. HLC combines physical time with a logical counter:

$"hlc" = (t_"physical", t_"logical")$

- Physical time from local wall clock (NTP-synchronized, ε ≈ 250ms).
- Logical counter increments when physical times collide.
- *Uncertainty window:* if a txn reads a value with hlc in the uncertainty window (now.physical ± ε), it may retry with a later timestamp to guarantee external consistency.

```python
# HLC implementation
import time

class HLC:
    def __init__(self):
        self.l = 0    # logical component
        self.c = 0    # counter (tie-break within same l)

    def now(self) -> tuple:
        pt = int(time.time() * 1e9)   # nanoseconds
        self.l = max(self.l, pt)
        if self.l == pt:
            self.c = 0
        else:
            self.c += 1
        return (self.l, self.c)

    def recv(self, msg_ts: tuple) -> tuple:
        pt = int(time.time() * 1e9)
        l_prev = self.l
        self.l = max(self.l, msg_ts[0], pt)
        if self.l == l_prev == msg_ts[0]:
            self.c = max(self.c, msg_ts[1]) + 1
        elif self.l == l_prev:
            self.c += 1
        elif self.l == msg_ts[0]:
            self.c = msg_ts[1] + 1
        else:
            self.c = 0
        return (self.l, self.c)
```

== Saga Pattern

Long-running distributed transactions (spanning minutes/hours) can't hold locks. The *Saga pattern* decomposes the transaction into a sequence of local transactions with compensating transactions for rollback.

```python
# Saga: book flight + hotel + car (each a separate microservice)
class BookingOrchestrator:
    def execute(self, booking_id: str):
        try:
            flight  = flight_svc.book(booking_id)    # T1
            hotel   = hotel_svc.book(booking_id)     # T2
            car     = car_svc.book(booking_id)       # T3
        except CarUnavailable:
            hotel_svc.cancel(booking_id)  # C2 — compensate T2
            flight_svc.cancel(booking_id) # C1 — compensate T1
            raise
        return (flight, hotel, car)

# Compensating transactions must be idempotent and always succeed
# Saga provides ACD (no isolation) — intermediate states are visible
```

*Saga vs 2PC:* Saga is eventually consistent (intermediate states visible); 2PC is ACID-isolated. Saga is appropriate for business processes where compensation (refund, cancel) is acceptable.

== References

Gray, J. (1978). "Notes on Database Operating Systems." Lecture Notes in Computer Science. (2PC origin)

Lamport, L. (2006). "Fast Paxos." Distributed Computing 19(2).

Thomson, A., Diamond, T., Weng, S., Ren, K., Shao, P., Abadi, D. (2012). "Calvin: Fast Distributed Transactions for Partitioned Database Systems." SIGMOD.

Corbett, J. et al. (2012). "Spanner: Google's Globally-Distributed Database." OSDI.

Kulkarni, S. et al. (2014). "Logical Physical Clocks and Consistent Snapshots in Globally Distributed Databases." OPODIS. (HLC)

Garcia-Molina, H. (1987). "Sagas." SIGMOD.

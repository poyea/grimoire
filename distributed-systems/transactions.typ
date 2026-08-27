#import "../template.typ": xref

= Distributed Transactions <transactions>

A transaction is the unit of *all-or-nothing* change: either every operation in a group commits durably, or none does. Distributing that guarantee across multiple nodes, disks, and data centres requires careful coordination of atomicity, concurrency control, and failure handling — the subject of this chapter.

*See also:* #xref("distributed-systems", "consensus-deep-dive", label: "Consensus Deep Dive"), #xref("distributed-systems", "gossip", label: "Gossip Protocols"), #xref("distributed-systems", "coordination-services", label: "Coordination Services"), #xref("database", "consensus-and-replication", label: "Consensus and Replication") (database-side framing), #xref("database", "storage-engines", label: "Storage Engines") (database-side framing), #xref("database", "transactions-distributed", label: "Distributed Transactions") (the database framing of the same protocols).

== ACID

*Atomicity* — a transaction's writes either all apply or all roll back, even across crashes.

*Consistency* — the database moves from one valid state to another; application-defined invariants hold at commit boundaries.

*Isolation* — concurrent transactions appear to execute serially (under the strictest definition).

*Durability* — committed writes survive crashes; achieved via write-ahead logging ($"WAL"$) and fsync.

ACID does not prescribe *how* isolation is achieved; that is left to *isolation levels* (§ Isolation Levels).

== Distributed Atomicity: Two-Phase Commit

*Two-Phase Commit* (2PC, Gray 1978) is the standard protocol for atomic commit across multiple resource managers (*participants*). A *coordinator* drives two rounds:

```
Phase 1 — Prepare:
    coordinator -> all participants: PREPARE(txn_id)
    participant: flush WAL, acquire locks, reply VOTE-YES or VOTE-NO

Phase 2 — Commit or Abort:
    if all VOTE-YES:
        coordinator writes COMMIT record to its own WAL
        coordinator -> all: COMMIT(txn_id)
        participants: flush commit record, release locks, ack
    else:
        coordinator -> all: ABORT(txn_id)
        participants: undo WAL, release locks
```

*Failure modes of 2PC:*

- *Coordinator crash after PREPARE, before COMMIT record:* participants are *in-doubt* — they have voted YES and hold locks indefinitely until the coordinator recovers. This is the *blocking* problem: 2PC is not fault-tolerant against coordinator crashes.
- *Participant crash after VOTE-YES:* coordinator waits; must retry COMMIT after recovery.
- *Network partition after COMMIT record written:* unreachable participants remain in-doubt. Manually resolving in-doubt transactions is a common DBA nightmare.

=== Three-Phase Commit

3PC (Skeen 1981) adds a *PRE-COMMIT* phase between PREPARE and COMMIT, allowing participants to time out and abort (rather than block) if the coordinator fails before PRE-COMMIT. It avoids the indefinite block of 2PC *only under synchronous networks* — in a real asynchronous network, a partition during PRE-COMMIT can still cause split-brain commits. 3PC is therefore rarely deployed in production.

*Paxos Commit* (Gray and Lamport 2006) replaces the coordinator's single WAL write with a consensus round across $2f+1$ replicas of the coordinator, eliminating the coordinator single point of failure without the synchrony assumptions of 3PC.

== Saga Pattern

A *Saga* (Garcia-Molina and Salem 1987) decomposes a long-running transaction into a sequence of shorter local transactions $T_1, T_2, ..., T_n$, each with a *compensating transaction* $C_i$ that semantically undoes $T_i$. If $T_k$ fails, the saga executes $C_(k-1), ..., C_1$ in reverse order.

Sagas provide ACD without full isolation: intermediate states are visible. They are appropriate when full locking is impractical (long-lived, cross-service) and compensation is possible.

=== Choreography

Each service listens on an event bus (Kafka, SNS) and reacts by executing its local transaction and publishing a result event. No central orchestrator exists.

```
OrderService publishes ORDER_CREATED
  -> PaymentService: PAYMENT_RESERVED or PAYMENT_FAILED
  -> InventoryService: STOCK_RESERVED or STOCK_FAILED
  -> ShippingService: SHIPMENT_CREATED
  -> OrderService updates status
```

*Pros:* no central SPOF, loose coupling. *Cons:* hard to trace, compensations are implicit in event handlers, difficult to add new steps.

=== Orchestration

A central *orchestrator* service issues commands to participants and tracks state in a durable state machine (e.g., AWS Step Functions, Temporal, Conductor).

```
orchestrator:
    state = PENDING
    call PaymentService.reserve()  -> state = PAYMENT_RESERVED
    call InventoryService.reserve() -> state = STOCK_RESERVED
    call ShippingService.create()  -> state = COMPLETE
    on failure at any step:
        trigger compensations in reverse
```

*Pros:* single place to observe saga state, explicit flow, easier to modify. *Cons:* orchestrator is a coordination bottleneck, requires durable execution platform.

== Isolation Levels

$"SQL"$-92 defines four levels in ascending strength:

- *Read Uncommitted:* dirty reads allowed. Almost never used.
- *Read Committed:* reads see only committed data; non-repeatable reads and phantom reads possible. Default in PostgreSQL, Oracle.
- *Repeatable Read:* re-reading a row returns the same value; phantom reads possible (new rows appear). Default in MySQL InnoDB.
- *Serialisable:* full isolation; equivalent to some serial schedule. Most expensive.

Between Repeatable Read and Serialisable sits *Snapshot Isolation* ($"SI"$): a transaction sees a consistent snapshot taken at its start time. SI prevents most anomalies but allows *write skew* (two transactions each read a shared condition, write to disjoint rows, collectively violating an invariant). *Serialisable Snapshot Isolation* ($"SSI"$, Cahill et al. 2008) detects dangerous write-skew cycles via read/write conflict tracking and aborts one transaction.

#table(
  columns: (auto, 1fr, 1fr, 1fr, 1fr),
  table.header[*Level*][*Dirty Read*][*Non-Repeatable Read*][*Phantom Read*][*Write Skew*],
  [Read Uncommitted], [Yes], [Yes], [Yes], [Yes],
  [Read Committed],   [No],  [Yes], [Yes], [Yes],
  [Repeatable Read],  [No],  [No],  [Yes], [Yes],
  [Snapshot Isolation],[No], [No],  [No],  [Yes],
  [Serialisable],     [No],  [No],  [No],  [No],
)

== Multi-Version Concurrency Control

*MVCC* avoids reader-writer conflicts by keeping multiple *versions* of each row, tagged with transaction timestamps or IDs. Readers see the latest version committed before their snapshot; writers append new versions.

*Postgres MVCC:* each row tuple carries `xmin` (creating transaction ID) and `xmax` (deleting transaction ID). A reader with snapshot $S$ sees a tuple if $"xmin"(t) <= S$ and $"xmax"(t) > S$. Old versions are cleaned by *VACUUM*.

*Version chain depth* determines read latency: a long-running transaction pins old versions, bloating the table. Postgres `autovacuum` aggressiveness must be tuned in write-heavy workloads.

=== Distributed MVCC: Percolator

Google Percolator (Peng and Dabek 2010) implements cross-row ACID transactions over Bigtable using MVCC. A *timestamp oracle* (Timestamp Server) issues globally monotone timestamps. Transactions use a two-phase locking protocol over Bigtable cells, with the *primary lock* record acting as the 2PC coordinator record.

```
Begin:          get start_ts from oracle
Write intent:   write LOCK(primary, ts=start_ts) to primary cell
                write LOCK(secondary, ptr=primary) to secondary cells
Prewrite:       if no conflicting locks/writes: write data cells
Commit:         get commit_ts from oracle
                replace primary LOCK with WRITE(commit_ts)
                async: clean secondary locks
Read(key, ts):  check if locked; if so, wait or roll forward
                return latest WRITE record with commit_ts <= ts
```

=== Distributed MVCC: CockroachDB HLC

CockroachDB uses *Hybrid Logical Clocks* ($"HLC"$, Kulkarni et al. 2014) to assign timestamps without a centralised oracle. Each node's HLC is $"HLC" = (l, c)$ where $l = max("physical time", "received l")$ and $c$ is a logical counter. HLCs advance monotonically, bound physical clock drift to $epsilon$ (configured max offset, default 500 ms), and are used as MVCC version timestamps. Uncertainty windows handle reads during clock skew: CockroachDB's *uncertainty restart* aborts a transaction and bumps its read timestamp if it encounters a write with timestamp inside the uncertainty interval.

== Deterministic Databases

Traditional databases make scheduling decisions at runtime based on lock conflicts. *Deterministic databases* pre-order all operations before execution, eliminating non-determinism and simplifying replication.

=== Calvin

Thomson et al. (2012). A *sequencer* layer orders transactions in 10 ms batches into a global log replicated via Paxos. A *scheduler* layer on each shard reads the log and executes transactions in that order using a lock table; because the order is globally known, locks are acquired deterministically without deadlock. No 2PC coordinator is needed: each shard knows what to commit or abort from the pre-ordained order.

$ "throughput" approx N_("shards") dot "shard throughput" / "cross-shard fraction" $

=== ARIA

Lu et al. (2020). Deterministic execution without a centralised sequencer: a batch of transactions is first executed speculatively (each transaction reads from a "reservation" table), then conflicts are detected, and conflicting transactions retry in deterministic order. Better utilises all cores than Calvin's sequencer bottleneck.

== NewSQL Comparison

#table(
  columns: (auto, 1fr, 1fr, 1fr, 1fr),
  table.header[*System*][*Consensus*][*Isolation*][*Timestamp*][*Approach*],
  [Spanner],      [Paxos per shard],   [Serialisable],   [TrueTime (GPS)],    [2PC + Paxos Commit],
  [CockroachDB],  [Raft per range],    [Serialisable],   [HLC],               [2PC + HLC uncertainty],
  [TiDB/TiKV],    [Raft per region],   [SI / Serialisable], [TSO oracle],     [Percolator model],
  [YugabyteDB],   [Raft per tablet],   [SI / Serialisable], [HLC],            [Percolator model],
  [Calvin/FaunaDB],[Raft sequencer],   [Serialisable],   [Log position],      [Deterministic],
  [Citus],        [PG replication],    [Read Committed], [PG clock],          [Sharded Postgres],
)

== Practical Failure Handling

*Idempotency keys:* clients include a unique request ID so that retried commits are safe. The server stores the result against the key; duplicate requests return the cached result.

*Saga compensation design:* compensating transactions must be *idempotent* and *eventually successful* — they cannot fail permanently. Externally-visible actions (email sent, charge processed) require careful compensation semantics (refund, apology email) rather than true undo.

*Distributed deadlock:* 2PC-based systems can deadlock across shards. Detectors use cycle detection in a *global wait-for graph* or rely on timeouts. Spanner avoids distributed deadlock with wound-wait: older transactions wound younger ones.

*In-doubt transaction resolution:* when a 2PC coordinator log is lost, the DBA must query participants and force a decision. Document the procedure; automate where possible with Paxos Commit.

== Further Reading

Gray, J. (1978). "Notes on Database Operating Systems." Lecture Notes in Computer Science.

Garcia-Molina, H., Salem, K. (1987). "Sagas." SIGMOD.

Peng, D., Dabek, F. (2010). "Large-Scale Incremental Processing Using Distributed Transactions and Notifications." OSDI.

Thomson, A., et al. (2012). "Calvin: Fast Distributed Transactions for Partitioned Database Systems." SIGMOD.

Cahill, M., et al. (2008). "Serializable Isolation for Snapshot Databases." SIGMOD.

Corbett, J., et al. (2013). "Spanner: Google's Globally Distributed Database." TOCS.

Taft, R., et al. (2020). "CockroachDB: The Resilient Geo-Distributed SQL Database." SIGMOD.

Lu, Y., et al. (2020). "ARIA: Reordering Execution for Scalable, Predictable Serialisability." VLDB.

Kulkarni, S., et al. (2014). "Logical Physical Clocks and Consistent Snapshots in Globally Distributed Databases." OPODIS.

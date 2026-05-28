= Isolation and Consistency Models

The ANSI SQL standard defines four isolation levels by the anomalies they permit. Adya (1999) formalized this with a graph-based framework that precisely characterizes each level and exposes gaps in the ANSI specification. Understanding this framework is essential for reasoning about correctness in real systems.

*See also:* _Concurrency Control_, _Distributed Transactions_, _Weakly Consistent Systems_

== ANSI SQL Isolation Levels

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Level*], [*Dirty Read*], [*Non-Repeatable Read*], [*Phantom*], [*Write Skew*],
  [Read Uncommitted], [✓ (allowed)], [✓], [✓], [✓],
  [Read Committed],   [✗],           [✓], [✓], [✓],
  [Repeatable Read],  [✗],           [✗], [✓], [✓],
  [Serializable],     [✗],           [✗], [✗], [✗],
)

*Anomaly definitions:*

- *Dirty Read:* T1 reads a value written by T2 before T2 commits. If T2 aborts, T1 read data that never existed.
- *Non-Repeatable Read:* T1 reads row X; T2 updates and commits X; T1 re-reads X and sees a different value.
- *Phantom:* T1 queries a range; T2 inserts/deletes rows in that range and commits; T1 re-queries and sees different rows.
- *Write Skew:* T1 and T2 each read a set of rows, each decides to write based on what it read, and the combined writes violate an invariant that neither write alone would violate.

```sql
-- Write skew example: "at least one doctor must be on call"
-- Two doctors (Alice, Bob) both want to go off-call simultaneously
-- Under Snapshot Isolation, both transactions can succeed — bug!

-- T1 (Alice):
BEGIN;
  SELECT count(*) FROM on_call;  -- sees 2 (Alice + Bob)
  UPDATE on_call SET active = false WHERE name = 'Alice';
COMMIT;                           -- succeeds

-- T2 (Bob, concurrent with T1):
BEGIN;
  SELECT count(*) FROM on_call;  -- also sees 2 (snapshot taken before T1 commit)
  UPDATE on_call SET active = false WHERE name = 'Bob';
COMMIT;                           -- also succeeds → now 0 doctors on call! Bug.

-- Fix: use SELECT FOR UPDATE (acquires locks) or Serializable isolation
BEGIN;
  SELECT count(*) FROM on_call FOR UPDATE;  -- blocks T2 until T1 commits
  ...
```

== Adya's Formalization

Adya models isolation as constraints on the *dependency graph* between transactions. Three types of dependencies:

- *WR (write-reads):* T2 reads a version written by T1 (T1 → T2).
- *WW (write-writes):* T2 overwrites a version written by T1 (T1 → T2).
- *RW (read-writes, anti-dependency):* T2 overwrites a version that T1 had read (T1 --anti--> T2).

*G0 (dirty write):* cycle of WW edges only. Preventing G0 = no dirty writes.

*G1a (dirty read):* T2 reads a version of X written by an aborted T1.

*G1b (intermediate read):* T2 reads an intermediate (not final) version of X written by T1 — i.e., T1 later overwrites X again before commit.

*G1c (circular information flow):* a cycle consisting of WR and WW edges only.

*G2 (anti-dependency cycle):* a cycle that involves at least one RW anti-dependency edge. This is write skew.

*Isolation levels by forbidden cycles:*

#table(
  columns: (auto, auto),
  [*Level*], [*Forbidden phenomena*],
  [Read Uncommitted], [G0],
  [Read Committed],   [G0, G1a, G1b, G1c],
  [Snapshot Isolation], [G0, G1\*, G2-item],
  [Serializable],     [G0, G1\*, G2],
)

*Snapshot Isolation (SI)* forbids G2-item (single-object write skew) but permits G2 (multi-object write skew like the doctor example above). This is why SI ≠ Serializable.

== Serializable Snapshot Isolation (SSI)

SSI (Cahill et al. 2008) detects and aborts dangerous structures in the dependency graph at runtime to make SI serializable, with minimal overhead.

*Dangerous structure (pivot):* any RW anti-dependency that is part of a cycle. SSI tracks *concurrent read/write conflicts* and aborts transactions when a cycle would form.

```
Theory: in any non-serializable SI execution, there exists a "pivot" transaction T_pivot
such that the dependency graph contains:
  T1 --RW--> T_pivot --RW--> T2
  where T1 has a WR/WW dependency from T2 (closing the cycle).
SSI tracks this pattern and aborts one of {T1, T2, T_pivot} preemptively.
```

*PostgreSQL SSI implementation:* adds a `SIREAD` lock (tracking read sets) and detects *dangerous structure* by examining concurrent `SIREAD` vs write conflicts.

```sql
-- Enable SSI in PostgreSQL
SET default_transaction_isolation = 'serializable';

-- The doctor example is now safe:
BEGIN ISOLATION LEVEL SERIALIZABLE;
  SELECT count(*) FROM on_call;   -- 2 doctors, SIREAD lock acquired
  UPDATE on_call SET active = false WHERE name = 'Alice';
COMMIT;   -- PostgreSQL detects the dangerous structure and aborts one of the txns
```

*Performance:* SSI adds ~10–15% overhead vs SI (Postgres benchmarks). Much less than full locking (2PL) under read-heavy workloads.

== Snapshot Isolation in Practice

Most "Repeatable Read" implementations in databases are actually SI:

#table(
  columns: (auto, auto, auto),
  [*Database*], [*"Repeatable Read" is actually*], [*Serializable mode*],
  [PostgreSQL],   [SI],                [SSI (true serializable)],
  [MySQL InnoDB], [SI],                [2PL with gap locks (≈ serializable)],
  [Oracle],       [SI (version-based)],[Serializable = SI (not true)],
  [SQL Server],   [SI (with RCSI)],    [True serializable (2PL)],
  [CockroachDB],  [SI by default],     [SSI],
)

*Oracle's "Serializable" is not true serializable* — it's SI, so write skew is still possible. Applications relying on Oracle's serializable for write-skew safety are buggy.

== Read Committed vs Snapshot Isolation

```sql
-- Read Committed: each statement gets a fresh snapshot
-- Non-repeatable read is possible

-- Session 1:
BEGIN;
  SELECT balance FROM accounts WHERE id=1;  -- 500
  -- Session 2 commits: UPDATE accounts SET balance=400 WHERE id=1;
  SELECT balance FROM accounts WHERE id=1;  -- 400 (different snapshot per stmt)
COMMIT;

-- Snapshot Isolation: snapshot taken at transaction start
BEGIN;
  SELECT balance FROM accounts WHERE id=1;  -- 500
  -- Session 2 commits: UPDATE accounts SET balance=400 WHERE id=1;
  SELECT balance FROM accounts WHERE id=1;  -- still 500 (same snapshot)
COMMIT;
```

*PostgreSQL defaults to Read Committed* (not SI!) because:
1. Most OLTP workloads are fine with it.
2. Lower overhead (no need to track a snapshot across multiple statements).
3. Applications often don't expect txn-level consistency.

== Consistency Models in Distributed Systems

Single-node isolation levels extend to distributed settings with additional concepts:

*Linearizability (external consistency):* every operation appears to take effect instantaneously at some point between its invocation and response. Implies serializability + real-time order.

```
Timeline:
  T1: write(X=1) ──────────────────────────────────── complete
  T2:    read(X) ───── returns 1  ✓ (linearizable)
  T3:    read(X) ─── returns 0   ✗ (not linearizable if T2 read 1)
```

*Causal consistency:* if write W1 causally precedes W2, all processes see W1 before W2. Does not require global ordering of concurrent writes.

*Eventual consistency:* all replicas converge to the same value if updates stop. Says nothing about intermediate states.

*Strong session guarantees:*

#table(
  columns: (auto, auto),
  [*Guarantee*], [*Meaning*],
  [Read-your-writes],       [After writing X, you always read your own write],
  [Monotonic reads],         [Once you read version V of X, you never see older versions],
  [Writes follow reads],     [After reading X=V and writing Y, Y's write is visible with the context of X=V],
  [Monotonic writes],        [Writes from a session are applied in order],
)

== Preventing Anomalies in Practice

```sql
-- 1. Explicit locking for write skew prevention
BEGIN;
  -- Lock all rows relevant to the invariant check
  SELECT * FROM on_call WHERE active = true FOR UPDATE;
  IF (SELECT count(*) FROM on_call WHERE active = true) > 1 THEN
    UPDATE on_call SET active = false WHERE name = 'Alice';
  END IF;
COMMIT;

-- 2. DEFERRABLE constraints for within-txn consistency
ALTER TABLE on_call ADD CONSTRAINT at_least_one_active
    CHECK (EXISTS (SELECT 1 FROM on_call WHERE active = true))
    DEFERRABLE INITIALLY DEFERRED;
-- PostgreSQL checks the constraint at COMMIT, not per-statement

-- 3. Predicate locks (SSI approach)
-- Automatically acquired under ISOLATION LEVEL SERIALIZABLE
-- No manual intervention needed
```

== References

Adya, A. (1999). "Weak Consistency: A Generalized Theory and Optimistic Implementations for Distributed Transactions." PhD thesis, MIT.

Cahill, M., Röhm, U., Fekete, A. (2008). "Serializable Isolation for Snapshot Databases." SIGMOD.

Bailis, P. et al. (2013). "Highly Available Transactions: Virtues and Limitations." VLDB.

Kleppmann, M. (2017). "Designing Data-Intensive Applications." O'Reilly. Ch. 7.

Fekete, A. et al. (2005). "Making Snapshot Isolation Serializable." ACM TODS 30(2).

Ports, D., Grittner, K. (2012). "Serializable Snapshot Isolation in PostgreSQL." VLDB.

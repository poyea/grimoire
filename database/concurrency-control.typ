= Concurrency Control

Concurrency control ensures that concurrent transactions produce results consistent with some serial execution. The two dominant paradigms are *pessimistic* (two-phase locking) and *optimistic* (OCC / MVCC). Modern OLTP systems have converged on MVCC with serial validation.

*See also:* _database/isolation-and-consistency-models.typ_, _database/recovery-and-logging.typ_, _database/transactions-distributed.typ_

== Two-Phase Locking (2PL)

*2PL theorem:* A schedule is conflict-serializable if every transaction follows the two-phase locking protocol:
1. *Growing phase:* acquire locks, never release.
2. *Shrinking phase:* release locks, never acquire.

The point at which a transaction releases its first lock is the *lock point*. Ordering transactions by their lock points gives a valid serial order.

```
T1: lock(A) → lock(B) → unlock(A) → unlock(B)   ✓ two-phase
T2: lock(A) → unlock(A) → lock(B) → unlock(B)   ✗ not two-phase
                          ↑ new lock acquired after releasing A
```

*Strict 2PL (S2PL):* release all locks only at commit/abort. Prevents *cascading aborts* — if T1 writes data read by T2, T1's abort doesn't force T2 to abort. Used by PostgreSQL (row-level locks) and InnoDB.

```
Strict 2PL:
  BEGIN
    lock(row_A, shared)
    lock(row_B, exclusive)
    ... work ...
  COMMIT → release ALL locks atomically
```

=== Lock Modes and Compatibility

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Requested →*], [*IS*], [*IX*], [*S*],  [*X*],
  [IS], [✓], [✓], [✓], [✗],
  [IX], [✓], [✓], [✗], [✗],
  [S],  [✓], [✗], [✓], [✗],
  [X],  [✗], [✗], [✗], [✗],
)

- *IS (Intention Shared):* table-level intent to acquire row-level S locks.
- *IX (Intention Exclusive):* table-level intent to acquire row-level X locks.
- *S (Shared):* read lock; multiple transactions can hold S simultaneously.
- *X (Exclusive):* write lock; incompatible with all other locks.

```sql
-- Explicit locking in PostgreSQL
BEGIN;
  -- Acquire S lock on row (held until COMMIT with S2PL)
  SELECT * FROM accounts WHERE id = 1 FOR SHARE;

  -- Acquire X lock
  SELECT * FROM accounts WHERE id = 2 FOR UPDATE;

  UPDATE accounts SET balance = balance - 100 WHERE id = 2;
COMMIT;
```

=== Deadlock Detection

Deadlocks occur when transactions form a cycle in the waits-for graph.

```
T1 holds lock(A), waiting for lock(B)
T2 holds lock(B), waiting for lock(A)
→ cycle: T1 → T2 → T1  (deadlock)
```

```python
# Waits-for graph cycle detection (Kahn's algorithm)
from collections import defaultdict, deque

def detect_deadlock(waits_for: dict[int, list[int]]) -> list[int] | None:
    """Returns a transaction in a cycle, or None if no deadlock."""
    in_degree = defaultdict(int)
    for txn, waitees in waits_for.items():
        for w in waitees:
            in_degree[w] += 1
    queue = deque(t for t in waits_for if in_degree[t] == 0)
    visited = 0
    while queue:
        t = queue.popleft()
        visited += 1
        for w in waits_for.get(t, []):
            in_degree[w] -= 1
            if in_degree[w] == 0:
                queue.append(w)
    all_txns = set(waits_for) | {w for ws in waits_for.values() for w in ws}
    if visited < len(all_txns):
        # Find a node in a cycle
        return [t for t in all_txns if in_degree[t] > 0]
    return None

# PostgreSQL runs deadlock detection every deadlock_timeout (default 1s)
# Victim selection: abort the transaction with the lowest cost (fewest locks)
```

== Optimistic Concurrency Control (OCC)

OCC (Kung & Robinson 1981): execute without locking, validate at commit time.

Three phases:
1. *Read phase:* execute reads/writes in private workspace; record read set and write set.
2. *Validation phase:* check that no conflict occurred with concurrently committed transactions.
3. *Write phase:* if validation passes, install writes atomically.

```python
# OCC validation (forward validation)
class OCCTransaction:
    def __init__(self, start_ts: int):
        self.start_ts  = start_ts
        self.read_set  = set()   # keys read
        self.write_set = {}      # key → new value

    def validate(self, committed: list["OCCTransaction"]) -> bool:
        # Check: for every committed txn T_j that committed after I started,
        # T_j's write set must not intersect my read set.
        for tj in committed:
            if tj.commit_ts > self.start_ts:
                if tj.write_set.keys() & self.read_set:
                    return False   # conflict: must abort and retry
        return True
```

*OCC is ideal when conflicts are rare* (read-heavy, low contention). Under high write contention, validation failures cause many retries, degrading throughput — this is the "OCC cliff".

== MVCC (Multi-Version Concurrency Control)

*Key idea:* instead of blocking readers with writer locks, maintain multiple versions of each row. Readers see a consistent snapshot; writers create new versions.

Every MVCC system answers two questions:
1. *Which version does a reader see?* (visibility rule)
2. *How are old versions reclaimed?* (garbage collection / VACUUM)

```
Versions of row id=42 over time:
  txn_id=100  created row: (id=42, name="Alice", balance=500)
  txn_id=105  updated row: (id=42, name="Alice", balance=400)   ← new version
  txn_id=110  updated row: (id=42, name="Alice", balance=350)   ← newer version

  A reader with snapshot ts=103 sees balance=500 (version from txn 100).
  A reader with snapshot ts=107 sees balance=400 (version from txn 105).
```

=== PostgreSQL MVCC Internals

PostgreSQL uses *tuple versioning in the heap*: old versions stay in the heap file; a chain of `ctid` pointers links them. `xmin`/`xmax` fields on each tuple control visibility.

```sql
-- Inspect tuple versions (xmin/xmax visibility)
SELECT xmin, xmax, ctid, id, balance
FROM   accounts
WHERE  id = 42;
-- xmin: txn that inserted this tuple version
-- xmax: txn that deleted/updated this tuple version (0 = live)
-- ctid: (page, slot) — physical location of the current version
```

*Visibility rule:* tuple version is visible to transaction T if:
- `xmin` committed AND `xmin < T.snapshot_xmin`
- `xmax = 0` OR (`xmax` not committed) OR (`xmax >= T.snapshot_xmin`)

```c
// Simplified PostgreSQL visibility check
bool heap_tuple_visible(HeapTuple tuple, Snapshot snap) {
    TransactionId xmin = tuple->t_xmin;
    TransactionId xmax = tuple->t_xmax;

    if (!TransactionIdDidCommit(xmin)) return false;
    if (TransactionIdPrecedes(snap->xmin, xmin)) return false;
    if (xmax == InvalidTransactionId) return true;   // not deleted
    if (!TransactionIdDidCommit(xmax)) return true;  // deleter aborted
    if (!TransactionIdPrecedes(xmax, snap->xmin)) return true; // deleter visible
    return false;
}
```

=== VACUUM and Dead Tuple Reclamation

Old tuple versions (dead tuples) accumulate and must be reclaimed:

```sql
-- Manual VACUUM (can run concurrently with queries)
VACUUM VERBOSE accounts;

-- VACUUM FULL: rewrites table, reclaims space — requires exclusive lock
VACUUM FULL accounts;

-- Autovacuum triggers (per table):
-- when dead tuples > autovacuum_vacuum_threshold + autovacuum_vacuum_scale_factor * n_live_tup

-- Monitor vacuum needs
SELECT relname,
       n_live_tup,
       n_dead_tup,
       round(100.0 * n_dead_tup / nullif(n_live_tup + n_dead_tup, 0), 1) AS dead_pct,
       last_autovacuum
FROM   pg_stat_user_tables
ORDER  BY n_dead_tup DESC
LIMIT  10;
```

*Transaction ID wraparound:* PostgreSQL uses 32-bit txn IDs. After ~2 billion transactions, IDs wrap around, causing all rows to appear "in the future". `VACUUM FREEZE` prevents this by freezing old tuples with a special marker.

=== InnoDB MVCC (Undo Segments)

InnoDB stores old versions in *undo segments* (rollback segments), not in the heap. The heap always has the latest version; older versions are reconstructed by applying undo records backward.

```
InnoDB row version chain:
  heap: (id=42, balance=350, roll_ptr→undo_log_1)
        undo_log_1: (balance=400, roll_ptr→undo_log_2)  ← prev version
        undo_log_2: (balance=500, roll_ptr=NULL)          ← oldest version

Read at snapshot ts=103:
  1. Read heap row (balance=350, committed by txn 110 > 103 → not visible)
  2. Follow roll_ptr to undo_log_1 (balance=400, committed by txn 105 > 103 → not visible)
  3. Follow roll_ptr to undo_log_2 (balance=500, committed by txn 100 ≤ 103 → visible ✓)
```

*Purge thread:* cleans undo logs when no active transaction needs older versions. Long-running transactions prevent purge, causing undo segment bloat.

== Timestamp Ordering (T/O)

Assign each transaction a timestamp at start. For each data item:
- `R-TS(X)`: largest timestamp of any transaction that read X.
- `W-TS(X)`: largest timestamp of any transaction that wrote X.

*Basic T/O rules:*
- T reads X: if `ts(T) < W-TS(X)` → abort (too late to read this version). Else update `R-TS(X)`.
- T writes X: if `ts(T) < R-TS(X)` → abort. If `ts(T) < W-TS(X)` → skip write (Thomas Write Rule). Else update `W-TS(X)`.

T/O is starvation-free but causes high abort rates under contention. Used as the base for MVCC with snapshot timestamps.

== Silo: MVCC + OCC for In-Memory Databases

Silo (Tu et al. 2013) achieves scalable OLTP on multi-core machines by combining per-core epochs, OCC validation, and a tree of version numbers.

*Key ideas:*
- Global epoch counter advances every ~40 ms.
- Each record has a `tid` = (epoch, sequence, thread_id).
- Read phase: snapshot record tids. Write phase: lock records (compare-and-swap), validate read set tids unchanged, commit.
- No global lock manager — record locking via CAS on the tid field.

```cpp
// Silo-style commit protocol (simplified)
bool silo_commit(Transaction& txn) {
    // Phase 1: lock write set (CAS on tid field)
    for (auto& [key, rec] : txn.write_set) {
        while (!rec->tid.compare_exchange_weak(rec->observed_tid,
                                               rec->observed_tid | LOCK_BIT))
            ; // spin (brief — no deadlock possible in OCC)
    }
    // Phase 2: validate read set (tids unchanged?)
    for (auto& [key, rec] : txn.read_set) {
        if (rec->tid != txn.observed_tids[key]) {
            // abort: release write-set locks
            for (auto& [k, r] : txn.write_set)
                r->tid.fetch_and(~LOCK_BIT);
            return false;
        }
    }
    // Phase 3: install writes, release locks
    uint64_t commit_tid = generate_tid(txn.write_set);
    for (auto& [key, rec] : txn.write_set) {
        rec->value = txn.new_values[key];
        rec->tid.store(commit_tid);       // releases lock, sets new version
    }
    return true;
}
```

== References

Kung, H.T., Robinson, J.T. (1981). "On Optimistic Methods for Concurrency Control." ACM TODS 6(2).

Bernstein, P., Hadzilacos, V., Goodman, N. (1987). "Concurrency Control and Recovery in Database Systems." Addison-Wesley.

Tu, S., Zheng, W., Kohler, E., Liskov, B., Madden, S. (2013). "Speedy Transactions in Multicore In-Memory Databases." SOSP.

Neumann, T., Mühlbauer, T., Kemper, A. (2015). "Fast Serializable Multi-Version Concurrency Control for Main-Memory Database Systems." SIGMOD.

Larson, P.-Å. et al. (2011). "High-Performance Concurrency Control Mechanisms for Main-Memory Databases." VLDB (Hekaton).

= Recovery and Logging

A database must survive crashes: power loss, kernel panics, storage errors. Recovery ensures that committed transactions are durable (Durability) and uncommitted transactions are rolled back (Atomicity). ARIES (1992) is the definitive algorithm; every major RDBMS implements a variant.

*See also:* _database/buffer-pool-and-io.typ_, _database/storage-engines.typ_, _database/concurrency-control.typ_

== Write-Ahead Logging (WAL)

*WAL rule:* Before a dirty page is flushed to disk, every log record describing a change to that page must first be written to the log file (on stable storage). This forces a specific ordering: log before data.

*Two sub-rules:*
- *Undo rule (steal policy):* A dirty page can be written to disk before the txn commits (steal), so the log must contain enough information to undo those changes.
- *Redo rule (no-force policy):* A txn can commit without flushing all its dirty pages (no-force), so the log must have enough information to redo those changes after a crash.

```
WAL log stream (sequential, append-only):
┌──────┬──────┬────────────────────────────────────────────┐
│ LSN  │ TxID │ Type + Payload                              │
├──────┼──────┼────────────────────────────────────────────┤
│ 1001 │  T1  │ BEGIN                                       │
│ 1002 │  T1  │ UPDATE page=42 off=0 before=[old] after=[new]│
│ 1003 │  T2  │ BEGIN                                       │
│ 1004 │  T1  │ UPDATE page=17 off=8 before=[...] after=[..]│
│ 1005 │  T1  │ COMMIT  ← T1 durable once this LSN fsynced  │
│ 1006 │  T2  │ UPDATE page=42 off=16 ...                   │
│ 1007 │  T2  │ ROLLBACK                                    │
└──────┴──────┴────────────────────────────────────────────┘
```

*Log Sequence Number (LSN):* monotonically increasing identifier for each log record. Every page header stores the *pageLSN* — the LSN of the last log record that modified that page. A page can only be flushed if `pageLSN ≤ flushedLSN`.

== ARIES Algorithm

ARIES (Algorithms for Recovery and Isolation Exploiting Semantics, Mohan et al. 1992) has three phases after a crash:

```
Phase 1: Analysis
  Scan log forward from last checkpoint.
  Rebuild:
    - dirty_page_table (DPT): pages modified but not flushed
    - transaction_table (TT): active txns at crash time
  Determine redoLSN = min(recLSN) across DPT

Phase 2: Redo
  Scan forward from redoLSN.
  For each UPDATE record:
    if page not in DPT → skip (already flushed)
    if recLSN > LSN of this record → skip
    if pageLSN on disk ≥ LSN → skip (idempotent)
    else: reapply the change (redo)

Phase 3: Undo
  Process loser transactions (in TT, not committed) in reverse LSN order.
  For each undo: write CLR (compensation log record), undo the change.
  CLR is redoable but not undoable — handles crash during undo.
```

*Compensation Log Records (CLRs):* when undoing log record with LSN $L$, write a CLR with `undoNextLSN` pointing to the predecessor of $L$ in that txn's chain. If we crash again during undo, we restart at the CLR and skip already-undone records.

```python
# Simplified ARIES redo phase
def redo_phase(log: list[LogRecord], dpt: dict, disk_pages: dict):
    redo_lsn = min(r["recLSN"] for r in dpt.values())
    for rec in log:
        if rec["lsn"] < redo_lsn:
            continue
        if rec["type"] != "UPDATE":
            continue
        pid = rec["page_id"]
        if pid not in dpt:
            continue                 # page already stable on disk
        if dpt[pid]["recLSN"] > rec["lsn"]:
            continue
        page_lsn = disk_pages[pid]["pageLSN"]
        if page_lsn >= rec["lsn"]:
            continue                 # idempotent check
        # apply redo
        disk_pages[pid]["data"]    = rec["after"]
        disk_pages[pid]["pageLSN"] = rec["lsn"]
```

== Checkpointing

Without checkpoints, crash recovery must replay the entire log — unbounded recovery time.

*Fuzzy checkpoint* (ARIES):

```
1. Write BEGIN_CHECKPOINT record to log
2. Write END_CHECKPOINT record containing:
   - current transaction_table (active txns, lastLSN)
   - current dirty_page_table (recLSN per dirty page)
   (No need to flush dirty pages!)
3. Update master record on disk: pointer to BEGIN_CHECKPOINT LSN
```

Recovery starts from `master_record → BEGIN_CHECKPOINT`, so log scan is bounded by checkpoint interval (typically seconds to minutes).

== PostgreSQL WAL Implementation

PostgreSQL WAL is organized in 16 MB segments. Each record has a generic header plus payload.

```c
// Simplified WAL record header (pg_internal.h)
typedef struct XLogRecord {
    uint32_t  xl_tot_len;   // total record length
    TransactionId xl_xid;   // transaction ID
    XLogRecPtr    xl_prev;  // LSN of previous record
    uint8_t   xl_info;      // flag bits
    RmgrId    xl_rmid;      // resource manager (heap, btree, ...)
    pg_crc32c xl_crc;       // CRC of record
} XLogRecord;
```

*Logical replication* in PostgreSQL decodes WAL records into row-level change events (INSERT/UPDATE/DELETE), making WAL a replication log as well.

```sql
-- Monitor WAL activity
SELECT pg_current_wal_lsn(),
       pg_walfile_name(pg_current_wal_lsn()),
       pg_wal_lsn_diff(pg_current_wal_lsn(), '0/0') / 1e9 AS wal_gb;

-- WAL write rate
SELECT pg_stat_get_bgwriter_stat_reset_time(),
       buffers_checkpoint,
       buffers_clean,
       buffers_backend
FROM   pg_stat_bgwriter;
```

== Group Commit

*Problem:* every COMMIT requires an fsync (costly: ~0.1–10 ms depending on storage). Under high write load, fsyncing each commit serially limits throughput.

*Group commit:* batch multiple commits into a single fsync.

```
T1 commits → enqueue in WAL buffer
T2 commits → enqueue in WAL buffer
T3 commits → enqueue in WAL buffer
                  ↓
           flush WAL buffer to disk (one fsync)
                  ↓
      T1, T2, T3 all acknowledged as durable
```

*PostgreSQL `commit_delay` + `commit_siblings`:* waits up to N µs if ≥ M concurrent transactions are active, allowing group commit to form naturally.

```sql
-- Configure group commit behavior
SET commit_delay      = 1000;  -- wait 1ms if siblings present
SET commit_siblings   = 5;     -- threshold: 5 concurrent active txns

-- Or use synchronous_commit = off for async commit (no durability guarantee)
-- Useful for bulk loads where losing last few rows is acceptable
SET synchronous_commit = off;
```

== InnoDB Redo Log and Doublewrite Buffer

InnoDB's redo log is a circular ring buffer (default 50 MB, can be up to 512 GB in MySQL 8.0+). Log records describe *physical changes* (byte offsets into pages).

*Doublewrite buffer:* before flushing dirty pages, InnoDB writes them to a contiguous doublewrite area on disk first. If a crash happens mid-page-write (torn write), recovery can restore from the doublewrite area.

```
InnoDB flush sequence:
  1. Write 16 KB pages to doublewrite buffer (sequential)
  2. fsync doublewrite buffer
  3. Write pages to their real locations (potentially random)
  4. fsync data file
  (If crash after step 1 but before step 4: restore from doublewrite)
```

Modern NVMe drives guarantee atomic 4 KB writes (sector size), so torn writes are less common — MySQL 8.0.20+ allows disabling doublewrite on NVMe with `innodb_doublewrite = 0`.

== Point-in-Time Recovery (PITR)

With a base backup + archived WAL segments, you can restore to any past point in time:

```bash
# PostgreSQL PITR setup
# 1. Take base backup
pg_basebackup -D /backup/base -Ft -z -P

# 2. Archive WAL continuously (in postgresql.conf)
# archive_mode = on
# archive_command = 'cp %p /wal_archive/%f'

# 3. Restore to specific time
cat > /backup/restore/recovery.conf << 'EOF'
restore_command = 'cp /wal_archive/%f %p'
recovery_target_time = '2025-12-01 14:30:00'
recovery_target_action = 'promote'
EOF
```

== Log-Structured Storage and WAL Elision

LSM-Tree systems (RocksDB, Cassandra) already write sequentially. Writes go to the WAL for durability, then to the MemTable. On flush, the SSTable is inherently consistent (no partial writes possible — immutable files). Recovery replays the WAL from the last flush point.

```
RocksDB recovery:
  1. Find the last flushed SSTable set (MANIFEST file)
  2. Replay WAL from that sequence number forward
  3. Rebuild MemTable; SSTs are already consistent
```

RocksDB's MANIFEST file (a log of file-level changes) plays a role analogous to ARIES's dirty page table, but at the file level rather than the page level.

== References

Mohan, C., Haderle, D., Lindsay, B., Pirahesh, H., Schwarz, P. (1992). "ARIES: A Transaction Recovery Method Supporting Fine-Granularity Locking and Partial Rollbacks Using Write-Ahead Logging." TODS 17(1).

Gray, J., Reuter, A. (1992). "Transaction Processing: Concepts and Techniques." Morgan Kaufmann.

Ramakrishnan, R., Gehrke, J. (2002). "Database Management Systems." Ch. 18–19. McGraw-Hill.

PostgreSQL Documentation. "WAL Reliability." https://www.postgresql.org/docs/current/wal-reliability.html

Dong, S. et al. (2021). "RocksDB: Evolution of Development Priorities in a Key-Value Store." ACM TOS.

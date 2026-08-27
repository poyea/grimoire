#import "../template.typ": xref

= Embedded Databases <embedded-databases>

An embedded database runs in the same process as the application; there is no server, no IPC, no protocol layer. The tradeoffs invert the usual ones: schema flexibility yields to startup latency and binary footprint; concurrency control degenerates to file locking or single-writer MVCC; durability relies on the host application not crashing in the middle of fsync. SQLite, DuckDB, RocksDB, and LMDB cover the design space — row OLTP, column OLAP, LSM key-value, and mmap-COW key-value respectively.

*See also:* #xref("database", "storage-engines", label: "Storage Engines"), #xref("database", "buffer-pool-and-io", label: "Buffer Pool and I/O"), #xref("database", "recovery-and-logging", label: "Recovery and Logging"), #xref("database", "concurrency-control", label: "Concurrency Control")

== SQLite

SQLite is the most-deployed database in history — every Android phone, every iOS app, every Firefox profile, most aircraft avionics. The codebase is a single `sqlite3.c` amalgamation, ~250k lines of C, with public-domain license and famously thorough test coverage (~640 test lines per source line via TH3).

=== Storage: Single-File B-Tree

A SQLite database is one file divided into fixed-size *pages* (default 4096 bytes). Page 1 holds the *file header* (magic `"SQLite format 3\0"`, page size, freelist root, schema cookie). Each table is a B+Tree rooted at a page recorded in `sqlite_schema`. Each index is a separate B-Tree.

```
File layout:
  Page 1: header + sqlite_schema root (B-Tree of CREATE TABLE/INDEX statements)
  Page 2..N: B-Tree pages, freelist trunk/leaf pages, overflow chains for big rows
```

Row storage uses a *cell* per record. Inner cells hold `(rowid_or_key, child_page)`; leaf cells hold `(rowid, payload)`. Payloads exceeding a threshold spill into overflow chains.

=== Write-Ahead Log

The default journaling mode since 3.7 is *WAL*. Writes append frames to `database-wal`; readers see the database file plus a *wal-index* (`-shm` shared-memory file) mapping page numbers to the latest WAL frame.

```
Transaction commit:
  1. Append frames to -wal (one per dirty page); each frame has a checksum.
  2. Append a commit frame (mxFrame set; this is the atomic commit point).
  3. fsync(-wal).
  4. Other connections see the new mxFrame via -shm and read from WAL.

Checkpoint (PRAGMA wal_checkpoint or auto every ~1000 frames):
  1. fsync(-wal).
  2. Copy frames into the main DB file in page order.
  3. fsync(database).
  4. Reset -wal header (restart) or truncate (truncate mode).
```

WAL gives concurrent readers + one writer without blocking; pre-WAL rollback journal mode blocked readers during writes.

=== Concurrency

SQLite uses *single-writer* concurrency. Locks (`SHARED`, `RESERVED`, `PENDING`, `EXCLUSIVE`) live on the database file via `fcntl(F_SETLK)` (or VFS-specific equivalents on Windows / WAL-shm). Transaction begin escalates from `SHARED` to `RESERVED` on first write; commit upgrades to `EXCLUSIVE` long enough to flush.

=== Virtual Tables

A *virtual table* is a C extension implementing the `sqlite3_module` API. SQLite calls `xBestIndex`, `xOpen`, `xFilter`, `xNext`, `xColumn`, `xClose` — letting external data appear as a SQL table.

```c
static sqlite3_module my_module = {
    .iVersion   = 3,
    .xCreate    = csv_create,    // CREATE VIRTUAL TABLE
    .xConnect   = csv_connect,
    .xBestIndex = csv_best_index, // returns estimated cost
    .xOpen      = csv_open,
    .xFilter    = csv_filter,     // start scan with bound args
    .xNext      = csv_next,
    .xEof       = csv_eof,
    .xColumn    = csv_column,
    .xRowid     = csv_rowid,
    .xDestroy   = csv_destroy,
};
sqlite3_create_module(db, "csv", &my_module, NULL);
// SQL: CREATE VIRTUAL TABLE logs USING csv('access.log');
```

CSV, JSON1, FTS5 (full-text), R*Tree (spatial), and dbstat are all built-in virtual tables. The mechanism powers PostgreSQL FDWs conceptually and is how Litestream and rqlite interpose.

=== Query Planner

SQLite's planner is the "Next-Generation Query Planner" (NGQP), a polynomial-time N-Nearest-Neighbor algorithm chosen for predictability over optimality. It computes per-table costs from `sqlite_stat1` (and optional `sqlite_stat4` distribution samples).

== DuckDB

DuckDB (Raasveldt & Mühleisen, 2019) is the "SQLite for analytics" — a single-process columnar OLAP engine. The library is a few MB, embedded into Python (`duckdb` package), R, Java, and command-line tools.

=== Storage Format

DuckDB stores data as *row groups* (default 122 880 rows) split into *vectors* (2 048 rows). Each vector is compressed independently using bitpacking, dictionary, RLE, frame-of-reference, or delta encoding depending on the column profile. The on-disk format (`.duckdb`) is a single file with versioning, MVCC undo information, and ART indexes for primary keys.

=== Vectorized Push Engine

(Discussed in detail in #xref("database", "column-stores-and-vectorized-execution", label: "Column Stores and Vectorized Execution") and #xref("database", "sql-engines-internals", label: "SQL Engine Internals").) The engine streams `DataChunk` columnar batches through pipelines. Aggregation uses thread-local hash tables that combine via a final merge.

=== Interop

DuckDB reads Parquet, CSV, JSON, Iceberg, Delta directly without ingest. The `httpfs` extension reads S3/GCS/Azure ranges via `Range:` requests. The `arrow` extension consumes Arrow IPC streams zero-copy.

```python
import duckdb
con = duckdb.connect("analytics.duckdb")
con.execute("""
  CREATE TABLE summary AS
    SELECT date_trunc('hour', ts) AS hour, count(*) AS n
    FROM read_parquet('s3://bucket/events/*/*.parquet')
    GROUP BY 1
""")
df = con.execute("SELECT * FROM summary").df()    # zero-copy to pandas via Arrow
```

== RocksDB

RocksDB (Facebook, 2012) is a fork of LevelDB tuned for server SSDs. It powers MyRocks, CockroachDB (pre-Pebble), TiKV, Kafka Streams state stores, and countless internal systems.

=== Architecture

LSM-tree with memtable (skiplist or hashskiplist) → SSTs in $L_0 ... L_n$. See #xref("database", "storage-engines", label: "Storage Engines") for level vs tiered details. RocksDB exposes column families, snapshots, prefix bloom filters, and per-CF compaction tuning.

=== Column Families

A column family is an independently-tuned LSM-tree sharing the same WAL. Different CFs may use different comparators, compaction styles, and compression.

```cpp
DB* db;
std::vector<ColumnFamilyDescriptor> cfs = {
  {"default", ColumnFamilyOptions()},
  {"index",   ColumnFamilyOptions().OptimizeUniversalStyleCompaction()},
  {"events",  ColumnFamilyOptions().OptimizeLevelStyleCompaction()},
};
std::vector<ColumnFamilyHandle*> handles;
DB::Open(DBOptions(), "/data/rocks", cfs, &handles, &db);

WriteBatch batch;
batch.Put(handles[1], "user:42", "...");
batch.Put(handles[2], "evt:00001", "...");
db->Write(WriteOptions(), &batch);     // atomic across CFs via shared WAL
```

=== Compaction Knobs

RocksDB exposes ~150 options. Critical ones: `write_buffer_size`, `max_write_buffer_number`, `level0_file_num_compaction_trigger`, `target_file_size_base`, `max_bytes_for_level_base`, `compaction_style` (level vs universal vs FIFO), `bloom_locality`, and `index_block_restart_interval`. Misconfigured RocksDB produces 30× write amplification; the *Universal* style trades read amp for write amp and matches time-series workloads.

=== MyRocks and TiKV

*MyRocks*: a MySQL storage engine substituting InnoDB with RocksDB. Better space (compression, no fragmentation) but worse for range-heavy OLTP. *TiKV*: Raft on top of column families; each region is a Raft group with its own log column family.

=== Pebble

CockroachDB replaced RocksDB with *Pebble* (Go) in 2020 to avoid CGO overhead and gain tighter integration. Pebble is API-compatible with RocksDB's basic surface but is leaner and tuned for Cockroach's MVCC layout.

== LMDB

LMDB (Symas Lightning Memory-Mapped Database) is an mmap-based B+Tree by Howard Chu (OpenLDAP). It is the storage engine behind OpenLDAP back-mdb and many embedded systems requiring zero-copy reads.

=== mmap + Copy-on-Write B+Tree

The entire database file is `mmap`'d. Readers traverse pointers directly in the mapped region — no buffer pool, no decompression, no allocation. Writers append new pages in a *copy-on-write* manner: any modified page is rewritten in a fresh free page, propagating upward to a new root.

```
Initial:    root → A → B (leaf)
Insert k:   allocate B' = B + {k}, A' = A with pointer to B', new root R' → A'
Commit:     write meta page selecting R' as current root (atomic 8-byte update)
```

This *single-level meta page* gives crash safety without a WAL: either the meta page update lands (new root visible) or it does not (old root). Read transactions hold a snapshot view by pinning the meta page at their start; they cannot be aborted by writers.

=== MVCC by Design

LMDB supports *one* writer + many readers concurrently. Writers serialize via a process-wide mutex. Readers do not block writers, but a long-running reader pins free pages and prevents reuse, causing the file to grow.

=== Tradeoffs

- *Pros:* zero-copy reads, no startup cost, crash-safe by construction, $approx$ 30 KLOC of C.
- *Cons:* writes serialized; sparse writes inflate file (mmap + COW); requires 64-bit virtual address space ($gt.eq 2$× DB size for safety).

== Use-Case Matrix

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Engine*], [*Data Model*], [*Workload Sweet Spot*], [*Footprint*], [*Concurrency*],
  [SQLite], [Row B-Tree, full SQL], [Mobile/desktop apps, config, edge sync], [< 1 MB], [Many readers + 1 writer (WAL)],
  [DuckDB], [Column store, full SQL], [Analyst notebooks, ETL, embedded BI], [$tilde$ 30 MB], [MVCC, single-process],
  [RocksDB], [Sorted KV (bytes)], [State stores, write-heavy KV, custom DBs], [$tilde$ 10 MB], [Multi-thread, snapshots],
  [LMDB], [Sorted KV (bytes)], [Read-mostly KV, configuration, LDAP, Tor consensus], [$tilde$ 50 KB], [Single writer + many readers],
  [BadgerDB], [Sorted KV (LSM, Go)], [Native Go services], [$tilde$ 10 MB], [Multi-thread MVCC],
  [Tkrzw], [Hash/B+/skiplist], [Custom indexes in C++ services], [$tilde$ 2 MB], [Configurable],
)

== Operational Tradeoffs

- *Backup:* SQLite has `.backup` online API; DuckDB lacks streaming backup (snapshot the file under transaction); RocksDB has `Checkpoint::CreateCheckpoint` (hardlinks SSTs); LMDB uses `mdb_env_copy` (consistent because of COW).
- *Replication:* none built in. Litestream tails the SQLite WAL to object storage. rqlite and dqlite wrap SQLite with Raft. CockroachDB wraps Pebble with Raft. LMDB has `mdb_env_copy` for offline snapshotting only.
- *Schema migration:* SQLite has limited `ALTER TABLE`; idiomatic migrations create a new table and `INSERT ... SELECT`. DuckDB rewrites partitions on type change.
- *Crash semantics:* SQLite WAL is atomic on commit; LMDB COW meta-page is atomic; RocksDB requires `WriteOptions.sync = true` for durability — otherwise a host crash loses recent writes.
- *Multi-process access:* SQLite via file locks (fine but slow on networked filesystems); LMDB via `mmap` + shared-mutex (single host only); RocksDB *forbids* multi-process opens; DuckDB allows multiple read-only attachments to the same file.

== Embedded Pitfalls

- Forgetting to `PRAGMA journal_mode = WAL` in SQLite leaves you on rollback-journal mode (reader-blocks-writer).
- DuckDB connections share buffer-manager memory; spawning per-request connections in a web service wastes memory — pool them.
- RocksDB's default options are tuned for tests, not production; always start from `OptimizeLevelStyleCompaction()` or `OptimizeForPointLookup()`.
- LMDB `MDB_NOSYNC` doubles throughput but loses ACID-D on crash. Use only for caches.
- Embedding any of these in a multi-tenant container without resource isolation lets one tenant fill page cache to evict another.

== Further Reading

Hipp, D.R. et al. (2010). "SQLite: Past, Present, and Future." VLDB Industrial Track.

Hipp, D.R. (2018). "SQLite: The Database at the Edge of the Network." (USENIX ATC keynote notes).

SQLite documentation: "Write-Ahead Logging", "Virtual Table Mechanism Of SQLite", "The Next-Generation Query Planner" (sqlite.org).

Raasveldt, M., Mühleisen, H. (2019). "DuckDB: An Embeddable Analytical Database." SIGMOD demo.

Raasveldt, M., Mühleisen, H. (2020). "Data Management for Data Science." CIDR.

Dong, S. et al. (2017). "Optimizing Space Amplification in RocksDB." CIDR.

Dong, S. et al. (2021). "Evolution of Development Priorities in Key-Value Stores Serving Large-Scale Applications: The RocksDB Experience." FAST.

Chu, H. (2011). "MDB: A Memory-Mapped Database and Backend for OpenLDAP." LDAPCon.

Cockroach Labs (2020). "Introducing Pebble: A RocksDB Inspired Key-Value Store Written in Go." Blog & GitHub `cockroachdb/pebble`.

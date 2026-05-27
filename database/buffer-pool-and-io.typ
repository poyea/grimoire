= Buffer Pool and I/O

The buffer pool is the database's in-process page cache. It decouples the query engine (which works on in-memory page pointers) from the storage device (which is orders of magnitude slower). Understanding the buffer pool is prerequisite to reasoning about I/O costs, dirty page management, and performance tuning.

*See also:* _database/storage-engines.typ_, _database/recovery-and-logging.typ_, _database/hardware-aware-design.typ_

== Buffer Pool Structure

The buffer pool is a fixed-size array of *frames* (in-memory buffers, each frame = one page = typically 8 KB or 16 KB). A hash table maps (file_id, page_id) → frame_id for O(1) lookup.

```
Buffer Pool (N frames):
┌──────┬────────────┬──────────┬───────────┬─────────────────┐
│Frame │  PageID    │  Dirty   │  PinCount │  Replacement    │
│  0   │ (f=1,p=42) │  true    │     2     │  LRU chain ptr  │
│  1   │ (f=1,p=17) │  false   │     0     │  LRU chain ptr  │
│  2   │ empty      │  —       │     0     │  —              │
│ ...  │    ...     │   ...    │    ...    │      ...        │
└──────┴────────────┴──────────┴───────────┴─────────────────┘
         ↑ hash table: (f,p) → frame index
```

*Pin count:* number of active users (threads) holding the page. A frame with `pinCount > 0` cannot be evicted — it is "pinned". The caller must call `unpin()` after use. This interface guarantees the buffer pool won't reclaim a frame while a thread is reading/writing it.

```c
// Buffer pool interface (simplified)
typedef struct BufferPool BufferPool;

// Pin a page (read it from disk if not in pool). Returns frame pointer.
Page* buf_fix(BufferPool *bp, FileID fid, PageID pid, bool exclusive);

// Unpin a page. dirty=true marks it for eventual flush.
void  buf_unfix(BufferPool *bp, Page *page, bool dirty);

// Typical usage pattern:
Page *p = buf_fix(bp, 1, 42, /*exclusive=*/false);
int   v = read_int(p, offset);
buf_unfix(bp, p, /*dirty=*/false);
```

== Page Replacement Policies

When the buffer pool is full and a new page is needed, a *victim* frame must be chosen and (if dirty) written back to disk.

=== LRU (Least Recently Used)

The classic policy: evict the frame that was last accessed longest ago.

```python
from collections import OrderedDict

class LRUBufferPool:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache    = OrderedDict()   # page_id → (data, pin_count, dirty)

    def fix(self, page_id: int) -> dict:
        if page_id in self.cache:
            self.cache.move_to_end(page_id)   # most recently used
            entry = self.cache[page_id]
            entry["pin"] += 1
            return entry
        data = self._read_from_disk(page_id)
        entry = {"data": data, "pin": 1, "dirty": False}
        if len(self.cache) >= self.capacity:
            self._evict()
        self.cache[page_id] = entry
        return entry

    def _evict(self):
        for pid, entry in self.cache.items():
            if entry["pin"] == 0:
                if entry["dirty"]:
                    self._write_to_disk(pid, entry["data"])
                del self.cache[pid]
                return
        raise RuntimeError("All frames pinned — buffer pool exhausted")
```

*LRU failure case — sequential scan:* scanning a 10 GB table with a 1 GB buffer pool will evict pages that will never be reused, thrashing the cache.

=== LRU-K

Track the K-th most recent access time. Evict the page whose K-th access is furthest in the past. With K=2, a page must be accessed twice before being considered "hot", preventing a single scan from polluting the pool.

=== Clock (Second-Chance FIFO)

A hand sweeps through frames in a circular fashion. Each frame has a *reference bit*. When a frame is accessed, set its bit. When the clock hand reaches a frame:
- If bit = 1: clear it (second chance), advance.
- If bit = 0: evict (not recently used).

O(1) amortized, low overhead. Used in many OS page caches.

=== CLOCK-Pro / ARC

PostgreSQL uses a *clock sweep* with a usage count (0–5). Pages with count > 0 are decremented; frames at 0 are candidates for eviction. This approximates LRU-K without a full LRU chain.

```sql
-- PostgreSQL buffer pool statistics
SELECT
    name,
    setting,
    unit
FROM pg_settings
WHERE name IN ('shared_buffers', 'effective_cache_size', 'wal_buffers');

-- See what's in the buffer pool (pg_buffercache extension)
SELECT relname, count(*) AS buffers,
       round(count(*) * 8 / 1024.0, 1) AS mb
FROM   pg_buffercache
JOIN   pg_class ON pg_class.relfilenode = pg_buffercache.relfilenode
WHERE  isdirty
GROUP  BY relname
ORDER  BY buffers DESC
LIMIT  10;
```

== I/O Modes and fsync

*Buffered I/O (default):* writes go to the OS page cache; the OS decides when to flush to disk. Fast, but a crash can lose data not yet flushed.

*O_DIRECT:* bypasses the OS page cache; the database manages its own buffering. Avoids double-buffering. Requires aligned I/O (512 B or 4 KB sector-aligned buffers).

*O_SYNC / fsync:* guarantees data is on stable storage before the call returns. Necessary for WAL durability.

```c
// Linux I/O modes comparison
int fd_buffered = open("data.db", O_RDWR);               // OS cache
int fd_direct   = open("data.db", O_RDWR | O_DIRECT);    // bypass cache

// fsync: flush dirty pages in OS cache to disk
fsync(fd_buffered);

// fdatasync: sync data only, skip metadata update (slightly faster)
fdatasync(fd_buffered);

// O_DSYNC: each write() call blocks until data hits disk
int fd_dsync    = open("wal.log", O_RDWR | O_DSYNC);
```

*PostgreSQL `wal_sync_method` options:*

#table(
  columns: (auto, auto, auto),
  [*Method*], [*How it works*], [*Notes*],
  [`fsync`],          [fsync() after each WAL write],   [Default; reliable],
  [`fdatasync`],      [fdatasync() — skip metadata],    [Slightly faster; Linux],
  [`open_sync`],      [O_SYNC on WAL file open],         [Some systems only],
  [`open_datasync`],  [O_DSYNC on WAL file open],        [Linux; good for NVMe],
)

== Prefetching and Read-Ahead

Sequential scans benefit from *read-ahead*: predict which pages will be needed and issue I/O before they're requested. Linux automatically does this for sequential file access (readahead syscall or madvise(MADV_SEQUENTIAL)).

For databases doing explicit random-access patterns, *async I/O* (io_uring on Linux) allows issuing multiple I/O requests without blocking:

```c
// io_uring example: read 4 pages concurrently
#include <liburing.h>

struct io_uring ring;
io_uring_queue_init(64, &ring, 0);

// Submit 4 async reads
for (int i = 0; i < 4; i++) {
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    io_uring_prep_read(sqe, fd, buffers[i], PAGE_SIZE, page_offsets[i]);
    sqe->user_data = i;
}
io_uring_submit(&ring);

// Harvest completions
for (int i = 0; i < 4; i++) {
    struct io_uring_cqe *cqe;
    io_uring_wait_cqe(&ring, &cqe);
    int idx = cqe->user_data;
    // buffer[idx] is now populated
    io_uring_cqe_seen(&ring, cqe);
}
io_uring_queue_exit(&ring);
```

== Buffer Pool Sizing

*Rule of thumb:* for OLTP, the working set (hot rows + indexes) should fit in the buffer pool. For OLAP, buffer pool is less critical — sequential scans don't benefit from caching.

```
PostgreSQL sizing:
  shared_buffers         = 25% of RAM        (buffer pool)
  effective_cache_size   = 75% of RAM        (hint to planner about OS cache)
  work_mem               = 4MB to 256MB      (per-sort/hash-join allocation)

InnoDB sizing:
  innodb_buffer_pool_size = 70–80% of RAM
  innodb_buffer_pool_instances = 8           (reduces lock contention)
```

*Buffer pool hit ratio:* fraction of page requests served from memory.

```sql
-- PostgreSQL buffer pool hit ratio
SELECT
    sum(heap_blks_hit)  AS heap_hits,
    sum(heap_blks_read) AS heap_reads,
    round(
        100.0 * sum(heap_blks_hit)
              / nullif(sum(heap_blks_hit) + sum(heap_blks_read), 0),
        2
    ) AS hit_ratio_pct
FROM pg_statio_user_tables;

-- Target: > 99% for OLTP workloads
```

== Dirty Page Flushing and Checkpointing

The *background writer* (bgwriter in PostgreSQL, page cleaner in InnoDB) continuously writes dirty pages to disk, keeping clean frames available for page misses without stalling user queries.

The *checkpointer* periodically triggers a full dirty-page flush to limit recovery time (bounded by `checkpoint_completion_target` in PostgreSQL).

```
InnoDB flush mechanisms:
  - LRU flushing:  page_cleaner flushes LRU tail to keep free_list populated
  - Flush list:    flush dirty pages in LSN order (ensures WAL consistency)
  - Adaptive flushing: accelerates flushing when redo log fill rate is high
```

```sql
-- InnoDB buffer pool health
SHOW ENGINE INNODB STATUS;  -- see "BUFFER POOL AND MEMORY" section

-- Key metrics from performance_schema
SELECT * FROM performance_schema.memory_summary_global_by_event_name
WHERE event_name LIKE '%innodb%buffer%';
```

== References

Effelsberg, W., Haerder, T. (1984). "Principles of Database Buffer Management." ACM TODS 9(4).

Graefe, G. (1993). "Query Evaluation Techniques for Large Databases." ACM CSUR 25(2).

PostgreSQL Documentation. "Resource Consumption." https://www.postgresql.org/docs/current/runtime-config-resource.html

Axboe, J. (2019). "Efficient IO with io_uring." https://kernel.dk/io_uring.pdf

Sadoghi, M., Canim, M., Bhattacharjee, B., Nagel, F., Ross, K. (2014). "Reducing Database Locking Contention Through Multi-version Concurrency." VLDB.

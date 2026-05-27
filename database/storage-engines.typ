= Storage Engines

A storage engine maps the logical data model (tables, rows, columns) onto persistent bytes on disk or flash. The two dominant families are *B-Tree* (update-in-place) and *LSM-Tree* (log-structured merge). The right choice depends on the read/write ratio, key distribution, and hardware.

*See also:* _database/buffer-pool-and-io.typ_, _database/recovery-and-logging.typ_, _database/hardware-aware-design.typ_

== B-Tree

A B-Tree of order $m$ is a balanced tree where each node holds $ceil(m/2) - 1$ to $m - 1$ keys and has $ceil(m/2)$ to $m$ children. All values live in leaf nodes (B+ variant — universal in databases). Inner nodes hold only keys and child pointers.

*Why B-Tree?* Disk/NVMe I/O is page-granular. A B-Tree node fills one page (4–16 KB), so a height-3 tree with branching factor 400 holds $400^3 approx 64M$ keys with at most 3 I/Os per lookup.

```
Height-3 B+Tree, branching factor 4 (simplified):
              [30 | 60]
             /    |    \
       [10|20]  [35|45|55]  [65|80]
       /   |  \    ...           \
    leafs leafs leafs           leafs
    (each leaf → next leaf via sibling pointer)
```

*Leaf sibling pointers* enable efficient range scans without revisiting inner nodes.

```c
// Simplified B+Tree node structure (InnoDB style)
typedef struct BTreeNode {
    uint16_t   level;          // 0 = leaf
    uint16_t   num_records;
    page_id_t  left_sibling;   // leaf level only
    page_id_t  right_sibling;  // leaf level only
    // variable-length slot directory follows
} BTreeNode;

// Point lookup: 3 page reads for height-3 tree
PageID btree_search(BTree *tree, Key key) {
    PageID cur = tree->root;
    while (!is_leaf(cur)) {
        Page *p = buf_fix(cur);     // pin page from buffer pool
        cur = inner_search(p, key); // binary search on inner keys
        buf_unfix(p);
    }
    return cur; // leaf page containing key
}
```

=== B-Tree Write Path (Update-in-Place)

```
INSERT key=42:
  1. Descend to leaf  (read path, O(log N) I/Os)
  2. Insert into leaf — if page has space: modify in-place, mark dirty
  3. If page overflows (> max records):
       a. Allocate new page
       b. Split records: half to new page
       c. Push middle key up to parent
       d. Recursively split parent if needed (rare: O(log N) amortized)
```

*Write amplification:* each inserted key causes 1 page write on average (amortized), plus WAL write. Much less write amplification than LSM for random small writes... but *random writes* to large B-Trees exceed SSD erase granularity, causing internal fragmentation.

=== Structural Modification Operations (SMOs)

B-Tree splits and merges are *SMOs*. Concurrent access during SMOs requires careful locking:

```
Crabbing/coupling protocol (optimistic):
  1. Acquire read latch on root
  2. Acquire read latch on child
  3. Release parent latch
  4. If child is "safe" (not full for insert, not half-empty for delete)
     continue crabbing. Else restart with write latches top-down.
```

InnoDB uses a *page modification log* (btr_mtr) to make SMOs atomic with respect to WAL.

== LSM-Tree (Log-Structured Merge Tree)

*Key insight:* convert random writes into sequential writes by batching mutations in memory, then periodically flushing and merging sorted runs on disk.

```
Write path:
  Write → WAL (sequential, for durability)
        → MemTable (in-memory sorted structure, e.g. skip list)
        → When MemTable full: flush to L0 SSTable (sorted, immutable)
        → Background compaction: merge L0 → L1 → L2 → ... SSTables
```

*SSTable (Sorted String Table):* immutable file of key-value pairs, sorted by key, with a block index and Bloom filter.

```
SSTable layout:
┌──────────────────────────────────────┐
│  Data blocks  (compressed 4KB each)  │
├──────────────────────────────────────┤
│  Index block  (first key per block)  │
├──────────────────────────────────────┤
│  Bloom filter (check key existence)  │
├──────────────────────────────────────┤
│  Footer (offsets, magic, checksum)   │
└──────────────────────────────────────┘
```

=== Compaction Strategies

#table(
  columns: (auto, auto, auto, auto),
  [*Strategy*], [*Write amp*], [*Read amp*], [*Space amp*],
  [Leveled (LevelDB/RocksDB)],   [10–30×], [low], [~1.1×],
  [Tiered (Cassandra STCS)],     [4–8×],   [high],[~10×],
  [FIFO],                         [~1×],    [high],[unbounded],
  [Hybrid (RocksDB Universal)],  [4–10×],  [med], [~2×],
)

*Leveled compaction* (RocksDB default): each level has a size budget; when L_i exceeds its budget, one SSTable is compacted into L\_{i+1}. Keeps read amplification bounded at $O(L)$ levels.

```python
# RocksDB Python (rocksdb3) — LSM in action
import rocksdb

opts = rocksdb.Options()
opts.create_if_missing = True
opts.compression = rocksdb.CompressionType.lz4_compression
opts.max_write_buffer_number = 3        # MemTable count before stall
opts.level0_file_num_compaction_trigger = 4

db = rocksdb.DB("/tmp/mydb", opts)

# Write batch (atomic, bypasses individual WAL syncs)
batch = rocksdb.WriteBatch()
for i in range(1000):
    batch.put(f"key:{i:06d}".encode(), f"value:{i}".encode())
db.write(batch)

# Point lookup: check MemTable → L0 Bloom → L0 data → L1...
val = db.get(b"key:000042")

# Range scan: merge iterators across all levels
it = db.iterkeys()
it.seek(b"key:000100")
for key in it:
    if key > b"key:000200":
        break
    print(key)
```

=== Read Path and Bloom Filters

A Bloom filter is a probabilistic set membership structure. For $n$ elements and $k$ hash functions into a bit array of size $m$:

$ P("false positive") approx (1 - e^(-k n / m))^k $

At $k = 10$, $m/n = 14.4$ bits/element, FPR $approx 0.1%$ — so 99.9% of non-existent key lookups skip the SSTable entirely.

```python
# Simple Bloom filter implementation
import hashlib
import math

class BloomFilter:
    def __init__(self, n: int, fpr: float = 0.001):
        m = -n * math.log(fpr) / (math.log(2) ** 2)
        self.m = int(m)
        self.k = int(self.m / n * math.log(2))
        self.bits = bytearray(self.m // 8 + 1)

    def _hashes(self, key: bytes):
        h1 = int(hashlib.md5(key).hexdigest(),  16)
        h2 = int(hashlib.sha1(key).hexdigest(), 16)
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, key: bytes):
        for idx in self._hashes(key):
            self.bits[idx // 8] |= 1 << (idx % 8)

    def __contains__(self, key: bytes) -> bool:
        return all((self.bits[i // 8] >> (i % 8)) & 1
                   for i in self._hashes(key))

bf = BloomFilter(n=1_000_000, fpr=0.001)
bf.add(b"user:42")
assert b"user:42" in bf
assert b"user:99" not in bf  # (with high probability)
```

== B-Tree vs LSM-Tree Comparison

#table(
  columns: (auto, auto, auto),
  [*Dimension*], [*B-Tree*], [*LSM-Tree*],
  [Write throughput],     [Moderate (random I/O)],    [High (sequential I/O)],
  [Write amplification],  [~2–4× (WAL + page write)], [10–30× (leveled compaction)],
  [Read latency],         [O(log N), predictable],     [O(L) levels, Bloom helps],
  [Space amplification],  [~1.1–1.3× (fragmentation)],[~1.1× leveled, ~10× tiered],
  [Range scans],          [Excellent (leaf chain)],    [Good (merge iterators)],
  [Compaction pauses],    [None (in-place updates)],   [Yes (background CPU + I/O)],
  [Example DBs],          [InnoDB, PostgreSQL, SQLite],[RocksDB, Cassandra, ClickHouse],
)

== Heap Files and Slotted Pages

*Heap file:* an unordered collection of pages. Tuples are appended; deletions leave gaps (reclaimed by VACUUM in PostgreSQL or compaction in InnoDB).

*Slotted page layout:*

```
┌──────────────────────────────────────────────────┐
│ Page header (LSN, checksum, free_space_ptr, ...)  │
├──────────────────────────────────────────────────┤
│ Slot array → [off1, len1] [off2, len2] ...        │  grows →
├──────────────────────────────────────────────────┤
│                  free space                       │
├──────────────────────────────────────────────────┤
│ ...tuple2... ...tuple1...                         │  ← grows
└──────────────────────────────────────────────────┘
```

The *slot array* grows forward, tuple data grows backward. A tuple is addressed by (page_id, slot_number) — the physical offset can change during compaction without invalidating external references.

== Learned Index Structures

*Learned indexes* (Kraska et al. 2018) replace B-Tree nodes with ML models that predict the position of a key in a sorted array.

```
Key → f(key) → predicted position ± error bound
```

A recursive model index (RMI) uses two-stage linear regression: a top model picks a sub-model; the sub-model predicts the position. For 200M integer keys on SSDs, RMI achieves 1.5–3× faster lookups than a cache-optimized B-Tree at 2× smaller footprint.

*Limitation:* inserts require re-training; current production use is mostly read-heavy immutable datasets (e.g., ClickHouse SortedIndexes, CockroachDB experiments).

```python
# Toy linear learned index (1-level, sorted keys)
import numpy as np

keys   = np.sort(np.random.randint(0, 10**9, size=10**6))
N      = len(keys)

# "Model": linear fit key → position
m      = N / (keys[-1] - keys[0] + 1)
b      = -m * keys[0]

def lookup(key: int, max_err: int = 1000) -> int:
    pred = int(m * key + b)
    pred = max(0, min(N - 1, pred))
    # binary search within error window
    lo, hi = max(0, pred - max_err), min(N - 1, pred + max_err)
    idx = np.searchsorted(keys[lo:hi+1], key)
    return lo + idx

print(keys[lookup(keys[42])])  # == keys[42]
```

== References

O'Neil, P. et al. (1996). "The Log-Structured Merge-Tree (LSM-Tree)." Acta Informatica.

Bayer, R., McCreight, E. (1972). "Organization and Maintenance of Large Ordered Indexes." Acta Informatica.

Dong, S. et al. (2021). "RocksDB: Evolution of Development Priorities in a Key-Value Store." ACM TOS.

Kraska, T. et al. (2018). "The Case for Learned Index Structures." SIGMOD.

Graefe, G. (2010). "A Survey of B-Tree Locking Techniques." ACM TODS.

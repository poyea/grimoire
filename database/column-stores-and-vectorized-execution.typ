= Column Stores and Vectorized Execution

Column-oriented storage (C-Store, MonetDB, Vertica, ClickHouse, DuckDB) achieves 10–100× better OLAP performance than row stores by exploiting column locality, compression, and SIMD vectorization. The key insight: analytical queries typically read a few columns from millions of rows — column storage eliminates reading irrelevant columns entirely.

*See also:* _database/query-compilation.typ_, _database/hardware-aware-design.typ_, _database/lakehouses-and-open-formats.typ_

== Row Store vs Column Store

```
Row store (PostgreSQL, InnoDB heap):
Page: [id|name|age|email] [id|name|age|email] ...
  → Reading AGG(age): must read (and discard) id, name, email for every row
  → Cache line loads all columns; only one needed

Column store (DuckDB, ClickHouse):
  id column:    [1, 2, 3, 4, 5, ...]     (contiguous int32 array)
  name column:  ["Alice", "Bob", ...]     (dict-encoded string array)
  age column:   [28, 34, 25, ...]         (contiguous int32 array ← what we need)
  email column: [...]
  → Reading AGG(age): read ONLY the age column; 4 bytes per row instead of ~100
  → SIMD: process 8 ages per AVX2 instruction
```

*Example speedup:* `SELECT AVG(price) FROM orders` on 100M rows:
- Row store: ~5 seconds (read 100M × 80 bytes = 8 GB)
- Column store: ~0.1 seconds (read 100M × 4 bytes = 400 MB, SIMD-accelerated)

== Compression in Column Stores

Since a column contains values of a single type and often low cardinality or sorted order, compression ratios of 5–20× are common. This multiplies the effective memory bandwidth.

=== Run-Length Encoding (RLE)

For sorted or low-cardinality columns:

```python
def rle_encode(values: list) -> list[tuple]:
    """[(value, run_length), ...]"""
    if not values:
        return []
    result = [(values[0], 1)]
    for v in values[1:]:
        if v == result[-1][0]:
            result[-1] = (result[-1][0], result[-1][1] + 1)
        else:
            result.append((v, 1))
    return result

data = [1,1,1,2,2,3,3,3,3,3]
print(rle_encode(data))   # [(1,3),(2,2),(3,5)]
# 10 ints → 3 pairs = 70% compression
# Aggregate SUM: no need to decompress — multiply value × run_length
```

=== Dictionary Encoding

Replace string values with small integer codes:

```python
def dict_encode(column: list[str]) -> tuple[list[int], list[str]]:
    dictionary = list(dict.fromkeys(column))   # preserve order, unique
    code_map   = {v: i for i, v in enumerate(dictionary)}
    codes      = [code_map[v] for v in column]
    return codes, dictionary

column     = ["US", "CA", "US", "UK", "US", "CA"]
codes, dic = dict_encode(column)
# codes: [0, 1, 0, 2, 0, 1]  (1 byte each vs 2 bytes per string)
# dictionary: ["US", "CA", "UK"]

# Predicate on codes (no string comparison!):
us_code = dic.index("US")
matching_rows = [i for i, c in enumerate(codes) if c == us_code]
```

=== Delta Encoding + Bitpacking

For sorted integer columns (timestamps, IDs):

```python
import struct

def delta_bitpack(values: list[int]) -> bytes:
    """Delta encode then pack with minimum bits needed."""
    deltas = [values[0]] + [values[i] - values[i-1] for i in range(1, len(values))]
    max_delta = max(deltas)
    bits_needed = max_delta.bit_length()  # e.g., 12 bits for deltas up to 4095

    # Bitpack: store each delta in bits_needed bits
    # (simplified: just struct pack as bytes for clarity)
    return struct.pack(f"<{len(deltas)}H", *deltas)

timestamps = sorted([1700000000 + i * 60 for i in range(1000)])  # 1-minute intervals
packed = delta_bitpack(timestamps)
# Uncompressed: 1000 × 8 bytes = 8000 bytes
# Delta = 60 for all → 6 bits each → ~750 bytes (10× compression)
```

=== Compression Comparison

#table(
  columns: (auto, auto, auto, auto),
  [*Scheme*], [*Best for*], [*Compress ratio*], [*Supports late materialization*],
  [RLE],             [Sorted/repeated values],  [10–100×], [Yes],
  [Dictionary],      [Low-cardinality strings], [5–20×],   [Yes (operate on codes)],
  [Delta + Bitpack], [Sorted timestamps/IDs],   [4–16×],   [Yes],
  [LZ4/Zstd],        [Arbitrary columns],        [2–5×],    [No (must decompress)],
)

== Late Materialization

Avoid reconstructing full tuples until necessary. Operate on column arrays; only join columns at the final output stage.

```
Query: SELECT name FROM employees WHERE age > 30 AND dept = 'Eng'

Early materialization (row store style):
  1. Scan all rows: reconstruct (id, name, age, dept) tuples
  2. Filter age > 30: keep matching tuples
  3. Filter dept = 'Eng': keep matching tuples
  4. Project name

Late materialization (column store):
  1. Scan age column → bool array [F,T,T,F,T,...]
  2. Scan dept column → bool array [T,T,F,T,T,...]
  3. AND the two bool arrays → position list [1,4,...]
  4. ONLY NOW: fetch name column at those positions → ["Bob","Eve",...]
  5. Output

Why faster: steps 1-3 work on compressed columns; avoid reading name until needed
```

```python
import numpy as np

def late_materialization_query(ages, depts, names):
    # Step 1-2: generate boolean masks (SIMD-friendly)
    mask_age  = ages > 30
    mask_dept = np.array([d == "Eng" for d in depts])

    # Step 3: combine
    mask = mask_age & mask_dept
    positions = np.where(mask)[0]

    # Step 4: materialize only what we need
    return names[positions]
```

== Vectorized Aggregation with SIMD

Modern CPUs can compute 8 int32 additions in one AVX2 instruction. Column stores align data to exploit this.

```c
#include <immintrin.h>
#include <stdint.h>

// Sum an array of int32 using AVX2 (8 values per instruction)
int64_t simd_sum_i32(const int32_t *arr, int n) {
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(arr + i));
        acc = _mm256_add_epi32(acc, v);
    }
    // Horizontal reduction: add 8 lanes
    int32_t tmp[8];
    _mm256_storeu_si256((__m256i *)tmp, acc);
    int64_t sum = 0;
    for (int j = 0; j < 8; j++) sum += tmp[j];
    // Handle tail
    for (; i < n; i++) sum += arr[i];
    return sum;
}
// Throughput: 8 int32 additions per cycle ≈ 30 GFLOPS on 4GHz CPU
```

*DuckDB*'s execution: processes 2048-element *vectors* at a time. Each operator receives a vector, computes over it (SIMD), passes to next operator. Operators are inlined via C++ templates — no virtual dispatch in the hot path.

```python
# DuckDB: high-performance OLAP in Python
import duckdb

# In-process analytics on 100M rows
con = duckdb.connect()
con.execute("CREATE TABLE orders AS SELECT * FROM 'orders_100m.parquet'")

result = con.execute("""
    SELECT
        date_trunc('month', created_at) AS month,
        SUM(amount)                     AS revenue,
        COUNT(DISTINCT customer_id)     AS unique_customers
    FROM orders
    WHERE status = 'completed'
    GROUP BY 1
    ORDER BY 1
""").fetchdf()
# ~0.5s on 100M rows (vs ~60s in pandas)
```

== Column Groups (Hybrid)

Pure column stores are slow for queries that need many columns per row (OLTP-style point lookups). *Column groups* partition columns into groups that are frequently accessed together.

```
orders table:
  Group A (frequently accessed together): (order_id, customer_id, status, created_at)
  Group B (analytics): (amount, discount, tax, shipping_cost)
  Group C (rare):      (notes, metadata, tags)

→ Store each group as a mini-row-store within the column store framework
```

Vertica, SAP HANA, and Microsoft SQL Server (columnstore indexes) use hybrid approaches.

== ClickHouse

ClickHouse (Yandex, 2016) is the dominant open-source OLAP column store. Key design:

```sql
-- MergeTree engine: primary index + sorted storage
CREATE TABLE events (
    event_date   Date,
    user_id      UInt64,
    event_type   LowCardinality(String),   -- dictionary encoding
    duration_ms  UInt32,
    payload      String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(event_date)           -- partition by month
ORDER BY (user_id, event_date)              -- sparse primary index on this key
SETTINGS index_granularity = 8192;         -- one index entry per 8192 rows

-- Query: exploits sort order → skips granules via sparse index
SELECT user_id, SUM(duration_ms)
FROM events
WHERE user_id = 12345 AND event_date >= '2025-01-01'
GROUP BY user_id;
-- Reads only the granules containing user_id=12345 → fast
```

*Sparse index:* stores the first key per granule (8192 rows). Point lookup: binary search in index → read one granule (8192 rows). For a table with 1B rows: 1B/8192 ≈ 122K index entries — fits in RAM.

== References

Stonebraker, M. et al. (2005). "C-Store: A Column-Oriented DBMS." VLDB. (foundational column store)

Boncz, P., Zukowski, M., Nes, N. (2005). "MonetDB/X100: Hyper-Pipelining Query Execution." CIDR. (vectorized execution)

Abadi, D. et al. (2013). "The Design and Implementation of Modern Column-Oriented Database Systems." Foundations and Trends in Databases.

Raasveldt, M., Mühleisen, H. (2019). "DuckDB: an Embeddable Analytical Database." SIGMOD.

ClickHouse Documentation. "MergeTree." https://clickhouse.com/docs/en/engines/table-engines/mergetree-family/mergetree

Willhalm, T. et al. (2009). "SIMD-Scan: Ultra Fast in-Memory Table Scan Using On-Chip Vector Processing Units." VLDB.

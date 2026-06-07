= Columnar Storage and Vectorization

Analytical workloads read a small subset of columns across millions of rows, making the traditional row-oriented layout wasteful — every cache line pulled from disk or DRAM carries irrelevant field bytes. *Columnar storage* reorders the physical layout so that all values of a single column are contiguous, enabling aggressive compression and SIMD-friendly batch processing. Combined with *vectorised execution*, columnar engines routinely outperform row-store engines by 10–100× on OLAP queries.

*See also:* _Storage Engines_, _Query Compilation_, _OLTP vs OLAP_, _Lakehouses and Open Table Formats_

== Row vs Columnar Layout

In a *row store*, each page holds complete tuples sequentially. A scan of a single column (e.g., `price`) must read every column in every row, wasting I/O bandwidth and polluting CPU caches with unused bytes.

```
Row store — 3 rows × 4 columns (id INT, name VARCHAR, price FLOAT, qty INT):
Page: [1|Alice|9.99|5] [2|Bob|4.50|12] [3|Carol|7.75|3]
      ←———— 48 bytes ————→ ←———— 48 bytes ————→ ←———— 48 bytes ————→
Reading `price` requires loading all 144 bytes; useful data: 12 bytes.
```

A *column store* places all values of each column together:

```
Column store — same data:
id:    [1] [2] [3]
name:  [Alice] [Bob] [Carol]
price: [9.99] [4.50] [7.75]   ← one cache line for all three prices
qty:   [5] [12] [3]
```

*Cache line utilisation* for a 64-byte cache line carrying 8-byte floats: row store delivers 1/6 useful bytes (~17%); column store delivers 8/8 (~100%) when only `price` is needed.

*Compression ratio* is far better in column stores because values within a column share the same type, domain, and often statistical distribution — a prerequisite for entropy- and run-based codecs.

=== Cache Line and Bandwidth Analysis

For a table with $w$ bytes per row, $c$ columns, and a query selecting $s$ of them:

$ "selectivity" = s / c $

Row-store I/O: $N dot w$ bytes. Column-store I/O: $N dot (w dot s / c)$ bytes.

At 100 columns and $s = 5$ selected, the column store transfers 20× less data — the dominant factor for memory-bandwidth-bound queries.

== Compression Techniques

=== Dictionary Encoding

*Dictionary encoding* replaces high-cardinality string or numeric values with compact integer codes. A separate dictionary maps code → original value.

```
Original column (status):  ["active", "inactive", "active", "pending", "active"]
Dictionary:                 {0: "active", 1: "inactive", 2: "pending"}
Encoded column (2-bit):     [0, 1, 0, 2, 0]
```

Space saving: 5 × 8 bytes (average) → 5 × 2 bits + ~30 bytes dict = ~31 bytes vs 40 bytes. Saving grows with column cardinality ratio. Predicates like `status = 'active'` become integer comparisons on the encoded column — no string comparison needed.

=== Run-Length Encoding (RLE)

*Run-length encoding* replaces consecutive identical values with (value, count) pairs. Ideal for sorted or low-cardinality columns.

```
Original:  [A, A, A, B, B, A, A, A, A, C]
RLE:       [(A,3), (B,2), (A,4), (C,1)]
```

Aggregations like `COUNT(*) WHERE col = 'A'` can be answered by summing run counts without decompressing individual values.

=== Delta Encoding

*Delta encoding* stores the difference between consecutive values rather than the values themselves. Particularly effective for monotonically increasing sequences (timestamps, auto-increment IDs).

```
Original timestamps (seconds): [1700000000, 1700000001, 1700000003, 1700000010]
Deltas:                        [1700000000, 1, 2, 7]
```

With delta encoding, a base value plus small deltas often fit in fewer bits, enabling subsequent bit-packing.

=== Bit-Packing

*Bit-packing* stores integers using only as many bits as required by the maximum value in a block, rather than a fixed 32 or 64 bits.

```
Values 0–15 fit in 4 bits; 8 values pack into 4 bytes vs 32 bytes (8× saving).
```

*Frame-of-reference (FOR)* encoding subtracts a block minimum before bit-packing, shrinking the effective range. Combined with delta encoding this is called *delta-of-delta + bit-packing* and is used in Gorilla (Facebook's time-series TSDB) and Parquet.

```python
import struct

def bit_pack(values: list[int], bits: int) -> bytes:
    """Pack integers into a byte array using `bits` bits each."""
    out, buf, pos = bytearray(), 0, 0
    for v in values:
        buf |= (v & ((1 << bits) - 1)) << pos
        pos += bits
        while pos >= 8:
            out.append(buf & 0xFF)
            buf >>= 8
            pos -= 8
    if pos:
        out.append(buf & 0xFF)
    return bytes(out)
```

== Parquet and ORC Format Internals

=== Apache Parquet

*Apache Parquet* is the de-facto standard columnar file format for analytical workloads. A Parquet file is organised as:

```
Parquet file layout:
┌────────────────────────────────────────────┐
│  Magic bytes: PAR1                          │
├────────────────────────────────────────────┤
│  Row Group 0                                │
│    Column Chunk (col 0): metadata + pages   │
│      Data Page v2 (dictionary-encoded)      │
│      Data Page v2 (bit-packed)              │
│    Column Chunk (col 1): ...                │
├────────────────────────────────────────────┤
│  Row Group 1 ... Row Group N                │
├────────────────────────────────────────────┤
│  File Footer (Thrift-encoded metadata)      │
│    Row group offsets, schema, statistics,   │
│    column chunk locations, key-value meta   │
├────────────────────────────────────────────┤
│  Footer length (4 bytes, little-endian)     │
│  Magic bytes: PAR1                          │
└────────────────────────────────────────────┘
```

A *row group* is the unit of parallel I/O (default 128 MB). Each *column chunk* holds all values for one column within that row group. Within a column chunk, *pages* (~1 MB) are the unit of compression and encoding. Statistics stored per column chunk (min, max, null count) and per page enable *predicate pushdown*: a reader can skip entire row groups or pages without decompressing.

*Bloom filters* in Parquet are stored per column chunk and answer "does value X appear in this column chunk?" in one probabilistic lookup, avoiding decompression for point-lookup style predicates on high-cardinality columns.

=== Apache ORC

*ORC* (Optimized Row Columnar) uses a similar structure — stripes (≈ row groups), indexes (lightweight statistics every 10 000 rows), and column data — but integrates tighter with Hive types and uses its own lightweight compression (ZLIB, Snappy, LZO, Zstd). ORC's *stream* abstraction separates presence, length, and data into individually seekable sub-streams per column, allowing efficient skipping.

#table(
  columns: (auto, auto, auto),
  [*Feature*], [*Parquet*], [*ORC*],
  [Ecosystem],        [Spark, Flink, DuckDB, Trino], [Hive, Spark, Presto],
  [Compression],      [Snappy, Zstd, Gzip, LZ4],     [Zlib, Snappy, Zstd, LZO],
  [Encoding],         [RLE, dict, bit-pack, delta],   [RLE, dict, direct, delta],
  [Bloom filters],    [Yes],                          [Yes],
  [Nested types],     [Dremel encoding],               [ORC struct/list/map],
  [Row group size],   [128 MB (default)],              [64 MB (default stripe)],
)

== Vectorised Execution

*Vectorised execution* processes a *batch* (typically 1 024–8 192 rows) of a single column per operator call, rather than one row at a time (volcano model) or generating machine code per query (compilation). It exploits:

- *CPU caches*: a 1 024-element int32 column fits in 4 KB — well within L1 cache.
- *SIMD*: AVX2 processes 8 × 32-bit values per instruction; AVX-512 processes 16.
- *Branch elimination*: tight loops over uniform-typed arrays avoid per-row type dispatch.

=== SIMD: AVX2 and AVX-512

*AVX2* (x86, 256-bit registers, 2013+) provides `_mm256_*` intrinsics for packed integer and float arithmetic. *AVX-512* (512-bit, 2017+) doubles the width and adds masked operations.

```c
#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

// AVX2: add two int32 arrays, 8 elements per iteration.
void add_int32_avx2(const int32_t *a, const int32_t *b,
                    int32_t *out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i vc = _mm256_add_epi32(va, vb);
        _mm256_storeu_si256((__m256i *)(out + i), vc);
    }
    // scalar tail
    for (; i < n; ++i) out[i] = a[i] + b[i];
}

// Vectorised predicate: count elements where price > threshold.
int32_t count_gt_avx2(const float *prices, size_t n, float threshold) {
    __m256 vt = _mm256_set1_ps(threshold);
    __m256i acc = _mm256_setzero_si256();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vp = _mm256_loadu_ps(prices + i);
        __m256  cmp = _mm256_cmp_ps(vp, vt, _CMP_GT_OQ);
        // -1 (0xFFFFFFFF) where true, 0 where false
        acc = _mm256_sub_epi32(acc,
                _mm256_castps_si256(cmp)); // subtract -1 = add 1
    }
    // horizontal sum of acc
    __m128i lo = _mm256_castsi256_si128(acc);
    __m128i hi = _mm256_extracti128_si256(acc, 1);
    __m128i sum128 = _mm_add_epi32(lo, hi);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    int32_t result = _mm_cvtsi128_si32(sum128);
    // scalar tail
    for (; i < n; ++i) result += (prices[i] > threshold) ? 1 : 0;
    return result;
}
```

=== Column Batch Processing

A vectorised engine passes *selection vectors* (arrays of active row indices) between operators instead of copying data:

```
Scan operator outputs:
  price_col[0..1023]  — raw column batch
  sel_vec[0..M-1]     — indices of rows passing predicates (M ≤ 1024)

Filter (price > 5.0) on batch:
  Input:  price_col, sel_vec (all 1024)
  Output: new sel_vec with indices where price_col[i] > 5.0

Aggregation (SUM):
  Iterate over sel_vec, accumulate price_col[sel_vec[j]]
```

This avoids materialising intermediate filtered rows while still using tight SIMD loops.

== Late Materialisation

*Late materialisation* (also called *late projection*) delays fetching non-predicate columns until after filtering has reduced the row count. Only then are additional columns fetched using the surviving row positions.

```
Query: SELECT name, revenue FROM orders WHERE status = 'shipped' AND amount > 1000

Early materialisation (row store style):
  Fetch id, name, status, amount, revenue for all rows → apply predicates → project

Late materialisation (columnar):
  1. Scan status column → position list P1
  2. Scan amount column at positions P1 → position list P2  (much smaller)
  3. Fetch name, revenue columns only at positions P2 → return result
```

For low-selectivity predicates (< 5% of rows pass), late materialisation reduces data read by the selectivity ratio.

== Predicate Pushdown to Storage

Columnar formats embed per-column statistics (min, max, distinct count, null count, bloom filter) at multiple granularities. A query engine exploits these during the *planning* phase to eliminate I/O before decompression:

```
Query: SELECT SUM(amount) FROM sales WHERE region = 'EU' AND year = 2023

Parquet reader:
  For each row group:
    if max(year) < 2023 or min(year) > 2023: SKIP entire row group
    if bloom_filter(region).might_contain('EU') == false: SKIP row group
  For each page within surviving row groups:
    if page-level stats exclude match: SKIP page
  Decompress and evaluate predicate only on surviving pages.
```

This is *predicate pushdown to storage*, and for highly selective queries on well-sorted data it can eliminate 99%+ of I/O.

== DuckDB's Vectorised Engine

*DuckDB* is an in-process OLAP engine that exemplifies the modern vectorised columnar design. Key architectural choices:

- *Push-based pipeline model*: operators push batches downstream rather than pulling (avoids function call overhead of volcano pull model).
- *Morsel-driven parallelism*: the table is divided into morsels (chunks of ~1 000 rows); threads steal morsels from a shared queue for load balancing.
- *Adaptive string storage*: strings ≤ 12 bytes stored inline in a fixed-width slot; longer strings stored in a separate buffer referenced by pointer + offset.
- *Compressed execution*: DuckDB operates directly on dictionary-encoded or RLE-encoded columns without decompression, using selection vectors.

```sql
-- DuckDB reading Parquet with predicate pushdown
EXPLAIN SELECT region, SUM(amount)
FROM read_parquet('sales_*.parquet', hive_partitioning = true)
WHERE year = 2023
GROUP BY region;
-- Output plan shows: PARQUET_SCAN with filters pushed into file reader,
-- followed by HASH_AGGREGATE with vectorised hash table.
```

== Row vs Columnar Summary

#table(
  columns: (auto, auto, auto, auto),
  [*Dimension*], [*Row Store*], [*Columnar Store*], [*Winner*],
  [Point lookup (single row)],    [1 page read], [N column seeks],       [Row],
  [Full column scan],             [Read all cols],[Read 1 col],           [Columnar],
  [Compression ratio],            [2–5×],        [5–20×],                [Columnar],
  [SIMD utilisation],             [Low (mixed types)],[High (homogeneous)],[Columnar],
  [Write (insert/update)],        [Append page], [Append each col file], [Row],
  [OLTP workload],                [Excellent],   [Poor],                 [Row],
  [OLAP workload],                [Poor],        [Excellent],            [Columnar],
  [Hybrid (HTAP)],                [Possible],    [Possible],             [Both],
)

== Further Reading

Abadi, D. et al. (2008). "Column-Stores vs. Row-Stores: How Different Are They Really?" SIGMOD.

Boncz, P., Zukowski, M., Nes, N. (2005). "MonetDB/X100: Hyper-Pipelining Query Execution." CIDR.

Apache Parquet format specification. https://parquet.apache.org/docs/file-format/

Raasveldt, M., Mühleisen, H. (2019). "DuckDB: an Embeddable Analytical Database." SIGMOD.

Lemire, D., Boytsov, L. (2015). "Decoding Billions of Integers per Second Through Vectorization." Software: Practice and Experience.

Willhalm, T. et al. (2009). "SIMD-Scan: Ultra Fast in-Memory Table Scan using On-Chip Vector Processing Units." VLDB.

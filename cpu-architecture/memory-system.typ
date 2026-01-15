= Memory System

DRAM (Dynamic Random-Access Memory) provides main memory storage. Understanding DRAM organization, timing, and access patterns is critical for performance optimization.

*See also:* Cache Hierarchy (for DRAM interface), Virtual Memory (for address translation), Multicore (for memory bandwidth sharing)

== DRAM Organization

A DRAM cell consists of one transistor and one capacitor to store a single bit. The capacitor charge represents the bit value: a charged capacitor stores 1, while a discharged capacitor stores 0. Reading is destructive because it discharges the capacitor, requiring a refresh after each read. Periodic recharging every 64ms prevents data loss, incurring a 10-15% performance overhead.

== DRAM Hierarchy

```
DIMM (Dual Inline Memory Module)
  ├─ Channel 0
  │   ├─ Rank 0 (8 chips)
  │   └─ Rank 1 (8 chips)
  ├─ Channel 1
  │   ├─ Rank 0
  │   └─ Rank 1
  ...

Chip internals:
  ├─ Banks (4-16 per chip, DDR4)
  │   ├─ Rows (32K-128K rows per bank)
  │   └─ Columns (1K-4K columns per bank)
```

Each bank is an independent array that can operate in parallel with other banks. A row contains 1-8 KB of data and represents the page size. The column is the minimal addressable unit, typically 8 bytes.

== DRAM Access

The row buffer acts as a cache for one row per bank, typically holding 8 KB. The DRAM access pattern involves three steps: Row Activate (RAS) loads a row into the row buffer with latency of approximately 15-20 ns (tRCD = RAS-to-CAS delay), Column Access (CAS) reads a column from the row buffer with latency of 10-15 ns, and Precharge closes the row to prepare for the next activate with latency of 15-20 ns (tRP = row precharge time). The total random access latency sums to tRP + tRCD + CAS = 40-55 ns.

```
1. Row Activate (RAS): Load row into row buffer
   - Latency: ~15-20 ns (tRCD = RAS-to-CAS delay)

2. Column Access (CAS): Read column from row buffer
   - Latency: ~10-15 ns (CAS latency)

3. Precharge: Close row, prepare for next activate
   - Latency: ~15-20 ns (tRP = row precharge time)

Total random access latency: tRP + tRCD + CAS = 40-55 ns
```

A row buffer hit occurs when accessing the same row during sequential access. The first access requires Activate + CAS = 25-35 ns, while subsequent accesses to the same row require only CAS = 10-15 ns, providing a 2-3x speedup.

```
First access: Activate + CAS = 25-35 ns
Subsequent accesses (same row): CAS only = 10-15 ns (2-3x faster!)
```

A row buffer miss occurs when accessing a different row in the same bank, requiring Precharge (close old row) + Activate (open new row) + CAS = 40-55 ns.

```
Precharge (close old row) + Activate (open new row) + CAS = 40-55 ns
```

== Memory Bandwidth

*DDR4-2400 bandwidth calculation:*

```
Clock: 1200 MHz (DDR = Double Data Rate → 2400 MT/s)
Bus width: 64 bits = 8 bytes
Theoretical bandwidth: 2400 × 8 = 19.2 GB/s per channel

Dual-channel: 2 × 19.2 = 38.4 GB/s
Quad-channel: 4 × 19.2 = 76.8 GB/s (server CPUs)
```

*Effective bandwidth:* 80-90% of theoretical (overhead: refresh, row buffer misses, command timing).

*Bandwidth saturation:*

```c
// Sequential copy (row buffer hits)
for (int i = 0; i < N; i++)
    dst[i] = src[i];
// Bandwidth: ~34 GB/s (90% of dual-channel DDR4-2400)

// Random access (row buffer misses)
for (int i = 0; i < N; i++)
    dst[random()] = src[random()];
// Bandwidth: ~5-10 GB/s (latency-bound, not bandwidth-bound)
```

== Memory Interleaving

*Problem:* Sequential addresses in same bank → serialize accesses.

*Solution:* Interleave addresses across banks/channels.

```
Sequential addresses:
Addr 0 → Channel 0, Bank 0
Addr 64 → Channel 0, Bank 1
Addr 128 → Channel 0, Bank 2
Addr 192 → Channel 0, Bank 3
Addr 256 → Channel 1, Bank 0
...

Effect: 8 consecutive cache lines spread across 8 banks
→ Parallel access → higher bandwidth
```

*Address mapping (typical):*

```
Physical Address Bits:
├─────┬────────┬──────┬──────────┬────────┤
│ Row │ Channel│ Rank │ Bank     │ Column │
└─────┴────────┴──────┴──────────┴────────┘

Low bits (column, bank, channel) used for interleaving
→ Sequential addresses access different banks/channels
```

== Memory Controller

*Integrated Memory Controller (IMC):* On-die (CPU), low latency.

```
CPU Core → L1 → L2 → L3 → Memory Controller → DRAM

Memory controller functions:
1. Translate physical address → DIMM/rank/bank/row/column
2. Schedule DRAM commands (optimize for bandwidth)
3. Track open rows per bank (row buffer management)
4. Handle refresh
5. ECC (Error-Correcting Code) if enabled
```

*Out-of-order memory access:* Reorder requests to maximize row buffer hits.

```
Request queue:
1. Read addr 0x1000 (Bank 0, Row 5)
2. Read addr 0x1040 (Bank 0, Row 5) ← Same row!
3. Read addr 0x2000 (Bank 0, Row 10)

Optimized order: 1, 2 (row buffer hit), 3
→ Save one activate + precharge
```

== DRAM Timings

*Key timing parameters (DDR4-2400 CL16):*

```
CAS Latency (CL): 16 cycles × 0.833 ns = 13.3 ns
tRCD (RAS to CAS): 16 cycles = 13.3 ns
tRP (Row Precharge): 16 cycles = 13.3 ns
tRAS (Row Active time): 36 cycles = 30 ns

Notation: DDR4-2400 CL16-18-18-36
          (CL-tRCD-tRP-tRAS)

Lower timings = faster (but more expensive, less stable)
```

*Latency calculation:*

```
Random access latency:
= tRP + tRCD + CL + (controller overhead)
= 16 + 16 + 16 + (~8) = 56 cycles = 46 ns @ 1200 MHz
= ~200 CPU cycles @ 4 GHz
```

== Memory Latency Optimization

*1. Sequential access:* Maximize row buffer hits.

```c
// GOOD: Sequential (row buffer hits)
for (int i = 0; i < N; i++)
    sum += arr[i];
// ~15 ns per access (CAS only)

// BAD: Large strides (row buffer misses)
for (int i = 0; i < N; i += 1024)
    sum += arr[i];
// ~50 ns per access (activate + CAS + precharge)
```

*2. Prefetching:* Hardware prefetcher hides latency.

```c
// Detected pattern → prefetch ahead
for (int i = 0; i < N; i++)
    process(arr[i]);  // Prefetcher loads arr[i+8..i+16] ahead

// Effective latency: ~0 (prefetch hides 50 ns DRAM latency)
```

*3. Memory-level parallelism:* Issue multiple requests.

```c
// CPU can track 10-12 outstanding misses
// Parallel requests to different banks → overlap latency

sum1 = arr1[i];  // Bank 0
sum2 = arr2[i];  // Bank 1
sum3 = arr3[i];  // Bank 2
// All 3 accesses in parallel → 50 ns total (not 150 ns)
```

== ECC Memory

*Error correction:* Detect/correct single-bit errors, detect double-bit errors.

*Overhead:*
- Extra DRAM chip: 8 data chips + 1 ECC chip (12.5% capacity overhead)
- Latency: +1-2 cycles for ECC calculation
- Bandwidth: ~5% reduction

*Use case:* Servers, mission-critical systems (prevent silent corruption).

== Bandwidth Optimization Strategies

*1. Stream efficiently:*

```c
// BAD: Read-modify-write (3× bandwidth)
for (int i = 0; i < n; i++)
    array[i] = 0;  // Read old value, write new value

// GOOD: Write-only stream (1× bandwidth)
memset(array, 0, n * sizeof(int));  // Or non-temporal stores
// Or explicit non-temporal:
for (int i = 0; i < n; i += 16)
    _mm_stream_si128((__m128i*)&array[i], _mm_setzero_si128());
```

*2. Prefetch for irregular access:*

```c
// BAD: Pointer chasing (serialized, ~200 cycles per access)
struct Node {
    int data;
    struct Node* next;
};

int sum = 0;
for (struct Node* p = head; p != NULL; p = p->next) {
    sum += p->data;  // Wait for p→next load before next iteration
}

// BETTER: Software prefetch (overlaps latency)
for (struct Node* p = head; p != NULL; p = p->next) {
    if (p->next) __builtin_prefetch(p->next, 0, 3);  // Prefetch next node
    if (p->next && p->next->next)
        __builtin_prefetch(p->next->next, 0, 3);  // Prefetch 2 ahead
    sum += p->data;
}
// Can hide ~50% of latency with good prefetch distance
```

*3. Maximize row buffer hits:*

```c
// BAD: Large stride (row buffer misses)
int matrix[1024][1024];
for (int col = 0; col < 1024; col++)
    for (int row = 0; row < 1024; row++)
        sum += matrix[row][col];  // Stride = 4096 bytes, new row every access

// GOOD: Sequential access (row buffer hits)
for (int row = 0; row < 1024; row++)
    for (int col = 0; col < 1024; col++)
        sum += matrix[row][col];  // Sequential, same row for 4KB
// Speedup: 3-5× from better DRAM access pattern
```

*Measuring memory bandwidth:*

```bash
# Monitor bandwidth usage
perf stat -e uncore_imc/event=0x04,umask=0x03/,\
uncore_imc/event=0x04,umask=0x0c/ ./program

# Or use Intel MLC tool
mlc --loaded_latency  # Shows latency under bandwidth load
mlc --bandwidth_matrix  # Bandwidth between NUMA nodes

# Simple bandwidth test
dd if=/dev/zero of=/dev/null bs=1M count=10000 iflag=fullblock
```

*Modern DDR5 characteristics (2024):*

```
DDR5-4800:
- Clock: 2400 MHz (DDR = 4800 MT/s)
- Bandwidth: 38.4 GB/s per channel
- Dual-channel: 76.8 GB/s
- On-die ECC (reliability improvement)
- Lower voltage: 1.1V (vs 1.2V for DDR4)

Latency comparison:
DDR4-3200 CL16: ~50ns
DDR5-4800 CL40: ~50ns (similar despite higher absolute latency)
```

== References

Jacob, B., Ng, S., & Wang, D. (2007). Memory Systems: Cache, DRAM, Disk. Morgan Kaufmann.

Hennessy, J.L. & Patterson, D.A. (2017). Computer Architecture: A Quantitative Approach (6th ed.). Morgan Kaufmann. Chapter 2 Appendix (Memory Technology).

JEDEC Standard (2020). JESD79-4B: DDR4 SDRAM Specification.

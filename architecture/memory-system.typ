= Memory System

DRAM (Dynamic Random-Access Memory) provides main memory storage. Understanding DRAM organization, timing, and access patterns is critical for performance optimization.

*See also:* Cache Hierarchy (for DRAM interface), Virtual Memory (for address translation), Multicore (for memory bandwidth sharing)

== DRAM Organization

*DRAM cell:* 1 transistor + 1 capacitor = 1 bit.

*Capacitor charge:* Stores bit (charged = 1, discharged = 0).

*Destructive read:* Reading discharges capacitor → must refresh after read.

*Refresh:* Periodic recharge (every 64ms) to prevent data loss → 10-15% overhead.

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

*Bank:* Independent array, can operate in parallel.

*Row:* 1-8 KB data (page size).

*Column:* Minimal addressable unit (typically 8 bytes).

== DRAM Access

*Row Buffer:* Cache one row per bank (8 KB typical).

*Access pattern:*

```
1. Row Activate (RAS): Load row into row buffer
   - Latency: ~15-20 ns (tRCD = RAS-to-CAS delay)

2. Column Access (CAS): Read column from row buffer
   - Latency: ~10-15 ns (CAS latency)

3. Precharge: Close row, prepare for next activate
   - Latency: ~15-20 ns (tRP = row precharge time)

Total random access latency: tRP + tRCD + CAS = 40-55 ns
```

*Row buffer hit:* Accessing same row (sequential access).

```
First access: Activate + CAS = 25-35 ns
Subsequent accesses (same row): CAS only = 10-15 ns (2-3x faster!)
```

*Row buffer miss:* Accessing different row in same bank.

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

== References

Jacob, B., Ng, S., & Wang, D. (2007). Memory Systems: Cache, DRAM, Disk. Morgan Kaufmann.

Hennessy, J.L. & Patterson, D.A. (2017). Computer Architecture: A Quantitative Approach (6th ed.). Morgan Kaufmann. Chapter 2 Appendix (Memory Technology).

JEDEC Standard (2020). JESD79-4B: DDR4 SDRAM Specification.

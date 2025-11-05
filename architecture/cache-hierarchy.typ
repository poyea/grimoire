= Cache Hierarchy

Memory hierarchy bridges the speed gap between fast CPU registers and slow main memory. Modern processors have 3-4 levels of cache between registers and DRAM.

*See also:* CPU Fundamentals (for register access), Memory System (for DRAM organization), Virtual Memory (for address translation), Multicore (for cache coherence)

== The Memory Wall

*Problem:* CPU speed increased 50%/year (1980-2005), DRAM latency improved only 7%/year [Hennessy & Patterson 2017].

*Result:* Memory latency gap.

```
Year  CPU Cycle Time  DRAM Latency  Latency (cycles)
1980      250 ns         250 ns            1
1990       50 ns         200 ns            4
2000       10 ns         100 ns           10
2010        3 ns          80 ns           27
2020        0.3 ns        60 ns          200
```

*Without caches:* Every memory access = 200 cycle stall → CPU spends 99% time waiting.

*With caches:* 95% hit rate, 4 cycle L1 → average = 0.95×4 + 0.05×200 = 13.8 cycles (14x speedup).

== Cache Hierarchy Overview

*Typical modern CPU (2020+):*

```
┌─────────────────┐
│   CPU Core      │
│  ┌───────────┐  │
│  │ Registers │  │  32-64 registers × 64-bit = 256-512 bytes
│  └─────┬─────┘  │  Latency: 0 cycles
│        │        │
│  ┌─────┴─────┐  │
│  │ L1 Cache  │  │  32-64 KB per core, split I/D
│  └─────┬─────┘  │  Latency: 4-5 cycles
│        │        │  Bandwidth: ~100 GB/s
│  ┌─────┴─────┐  │
│  │ L2 Cache  │  │  256-512 KB per core
│  └─────┬─────┘  │  Latency: 12-15 cycles
└────────┼────────┘  Bandwidth: ~50 GB/s
         │
    ┌────┴────┐
    │L3 Cache │       8-64 MB shared
    └────┬────┘       Latency: 40-80 cycles
         │            Bandwidth: ~20-30 GB/s
    ┌────┴────┐
    │  DRAM   │       8-128 GB
    └─────────┘       Latency: ~200 cycles (60-100 ns)
                      Bandwidth: ~20-50 GB/s (per channel)
```

*Cache line:* 64 bytes (all modern CPUs). Unit of transfer between cache levels.

*Why 64 bytes?*
- Spatial locality: Adjacent data often accessed together
- Bus efficiency: Amortize address transfer over 64 bytes
- Tradeoff: Too large → waste bandwidth, too small → more transfers

== Cache Organization

*Direct-mapped cache:*

```
Address bits: [Tag | Index | Offset]

Example: 32 KB cache, 64-byte lines
- Offset: 6 bits (64 = 2^6)
- Index: 9 bits (32K / 64 = 512 lines = 2^9)
- Tag: remaining bits

Lookup: Index selects cache line, compare tag, if match → hit
```

*Direct-mapped problem:* Two addresses with same index → conflict misses.

```cpp
// Pathological case: stride = cache_size
int arr[8192];  // 32 KB
for (int i = 0; i < 8192; i += 512) {
    arr[i] = 0;  // Every access maps to same cache line → 0% hit rate!
}
```

*Set-associative cache:*

```
N-way set associative: N cache lines per set

Address: [Tag | Set Index | Offset]

Example: 32 KB, 64-byte lines, 8-way associative
- Offset: 6 bits
- Index: 6 bits (32K / 64 / 8 = 64 sets = 2^6)
- Tag: remaining bits

Lookup: Index selects set of 8 lines, check all 8 tags in parallel
```

*Associativity tradeoffs:*
- Higher associativity: Fewer conflict misses, more complex/slower lookup
- L1: 8-way typical (fast lookup required)
- L2: 8-16 way
- L3: 16-20 way (larger capacity, can afford complexity)

*Fully associative:* Any line can go anywhere. Used for TLB (small, needs fast lookup).

== Cache Replacement Policies

*LRU (Least Recently Used):* Evict line unused for longest time.

*Implementation (8-way):*
- Track access order with 3-bit counters per line (log2(8) = 3)
- On access: Set accessed line to 0, increment others
- On eviction: Evict line with highest counter

*Pseudo-LRU:* Binary tree approximation, less hardware.

```
        Root
       /    \
      0      1
     / \    / \
    0   1  2   3
   /|\ /|\ ...
 L0 L1...L7

Each node: 1 bit (left=0, right=1)
Access L3: Set root=1, node1=1
Evict: Follow 0 bits → L0
```

*LRU approximation:* Modern CPUs use variants (not true LRU - too expensive for 20-way L3).

== Cache Performance Metrics

*Miss rate:* Fraction of accesses that miss.

```
Miss Rate = Misses / Total Accesses

L1: 2-5% typical
L2: 10-30% of L1 misses
L3: 30-50% of L2 misses
```

*AMAT (Average Memory Access Time):*

```
AMAT = Hit Time + Miss Rate × Miss Penalty

Example: L1 with 95% hit rate
AMAT = 4 + 0.05 × 200 = 14 cycles average
```

*Inclusive vs exclusive caches:*

*Inclusive (Intel):* L3 contains everything in L1/L2.
- Pro: Simple coherence (check L3 only)
- Con: Wasted capacity (duplication)

*Exclusive (AMD Zen):* L3 contains only evicted data from L1/L2.
- Pro: Effective capacity = L1 + L2 + L3
- Con: Complex coherence

== Three C's of Cache Misses

*Compulsory (cold) misses:* First access to data - unavoidable.

*Capacity misses:* Working set larger than cache.

```cpp
// Working set = 8 MB, L3 = 4 MB → capacity misses
int arr[2000000];  // 8 MB
for (int i = 0; i < 2000000; i++) {
    arr[i] = 0;  // Evict previous data, miss on every access
}
```

*Conflict misses:* Addresses map to same cache set.

```cpp
// Two arrays, same alignment mod cache size
int *a = malloc(1048576);  // 1 MB at address X
int *b = malloc(1048576);  // 1 MB at address X + 8MB (conflicts with a)

for (int i = 0; i < N; i++) {
    result += a[i] + b[i];  // a[i] and b[i] conflict → thrashing
}
```

*Solution:* Pad arrays, align differently, or use cache-oblivious algorithms.

== Cache Line States (MESI Protocol)

*For multicore coherence (see Multicore section):*

```
M (Modified):   Dirty, exclusive to this core
E (Exclusive):  Clean, exclusive to this core
S (Shared):     Clean, may be in other caches
I (Invalid):    Not present

State transitions on read/write/snoop
```

*False sharing:* Different variables on same cache line → performance disaster.

```cpp
struct Counter {
    int count1;  // Offset 0
    int count2;  // Offset 4 (same 64-byte cache line!)
};

// Thread 1 writes count1 → invalidates cache line in Thread 2
// Thread 2 writes count2 → invalidates cache line in Thread 1
// Ping-pong effect: 10-100x slower than separate cache lines
```

*Solution:* Align to cache line boundaries.

```cpp
struct Counter {
    alignas(64) int count1;  // Own cache line
    alignas(64) int count2;  // Own cache line
};
```

== Prefetching

*Hardware prefetcher:* Detects patterns, fetches data before use.

*Stream prefetcher:* Sequential access detection.

```cpp
// Detected pattern: arr[i], arr[i+1], arr[i+2], ...
for (int i = 0; i < N; i++) {
    sum += arr[i];  // Prefetcher loads arr[i+8], arr[i+16] ahead
}
```

*Stride prefetcher:* Constant-stride access.

```cpp
// Detected pattern: arr[0], arr[10], arr[20], ... (stride = 10)
for (int i = 0; i < N; i += 10) {
    sum += arr[i];  // Prefetcher learns stride, prefetches arr[i+10k]
}
```

*Prefetch distance:* How far ahead to fetch (tunable, ~10-20 cache lines typical).

*Prefetch effectiveness:*
- Sequential: 90-95% of misses eliminated
- Stride: 80-90% eliminated
- Random: 0% (no pattern)

*Software prefetch:*

```cpp
// Explicit prefetch hint
for (int i = 0; i < N; i++) {
    __builtin_prefetch(&arr[i + 16], 0, 3);  // Prefetch 16 ahead
    process(arr[i]);
}

// Parameters: address, write(1)/read(0), temporal locality (0-3)
```

*When to use software prefetch:*
- Linked lists (unpredictable next pointers)
- Complex data structures
- Known future access not detectable by hardware

*Cost:* Prefetch uses cache bandwidth. Over-prefetching can evict useful data.

== Cache-Conscious Programming

*1. Spatial locality:* Access nearby memory.

```cpp
// GOOD: Row-major (sequential)
for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
        sum += matrix[i][j];

// BAD: Column-major (stride = cols)
for (int j = 0; j < cols; j++)
    for (int i = 0; i < rows; i++)
        sum += matrix[i][j];  // Each access misses cache line
```

*2. Temporal locality:* Reuse recently-accessed data.

```cpp
// GOOD: Reuse matrix[i] before eviction
for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
        result[i] += a[i][k] * b[k];  // Reuse a[i][k] for all b[k]
    }
}
```

*3. Blocking (tiling):* Fit working set in cache.

```cpp
// Matrix multiply: O(N^3) operations, O(N^2) data
// Naive: Evicts data before reuse if N^2 > cache

// Blocked: Process B×B tiles that fit in cache
#define B 64  // Chosen to fit 3 B×B tiles in L1
for (int ii = 0; ii < N; ii += B)
  for (int jj = 0; jj < N; jj += B)
    for (int kk = 0; kk < N; kk += B)
      for (int i = ii; i < min(ii+B, N); i++)
        for (int j = jj; j < min(jj+B, N); j++)
          for (int k = kk; k < min(kk+B, N); k++)
            C[i][j] += A[i][k] * B[k][j];

// Speedup: 5-10x for large N (cache misses reduced dramatically)
```

== Cache Performance Measurement

*Using performance counters:*

```bash
# Linux perf
perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./program

# Example output:
#   100,000,000  L1-dcache-loads
#     5,000,000  L1-dcache-load-misses    # 5% miss rate
#    10,000,000  LLC-loads                # L3 accesses
#     3,000,000  LLC-load-misses          # 30% L3 miss rate
```

*Intel VTune / AMD uProf:* GUI-based profiling with cache miss attribution.

*Cachegrind (Valgrind):* Simulates cache behavior, reports misses per source line.

```bash
valgrind --tool=cachegrind --cachegrind-out-file=out ./program
cg_annotate out --auto=yes
```

== Advanced Topics

*Non-Temporal Stores:* Bypass cache for write-only data.

```cpp
// Streaming store (AVX)
_mm256_stream_si256((__m256i*)&dest[i], data);
// Writes directly to memory, doesn't pollute cache
// Use for large memcpy, video processing
```

*Cache Partitioning (CAT - Cache Allocation Technology):*

Intel feature: Partition L3 cache across VMs/processes to prevent interference [Intel SDM].

```bash
# Allocate 50% of L3 to process
pqos -e "llc:0=0xf;llc:1=0xf0"
```

*Victim Cache:* Small fully-associative cache for recently evicted lines [Jouppi 1990].

Stores lines evicted from L1 → reduces conflict misses. Transparent to software.

== References

*Primary sources:*

Hennessy, J.L. & Patterson, D.A. (2017). Computer Architecture: A Quantitative Approach (6th ed.). Morgan Kaufmann. Chapter 2 (Memory Hierarchy Design).

Intel Corporation (2023). Intel 64 and IA-32 Architectures Optimization Reference Manual. Section 2.2 (Caching).

Jouppi, N.P. (1990). "Improving Direct-Mapped Cache Performance by the Addition of a Small Fully-Associative Cache and Prefetch Buffers." ISCA '90.

*Performance analysis:*

Drepper, U. (2007). "What Every Programmer Should Know About Memory." Red Hat, Inc. https://people.freebsd.org/~lstewart/articles/cpumemory.pdf

Fog, A. (2023). Optimizing Software in C++. Technical University of Denmark. Chapter 7 (Cache and Memory).

*Research:*

Smith, A.J. (1982). "Cache Memories." ACM Computing Surveys 14(3): 473-530.

Baer, J-L. & Wang, W-H. (1988). "On the Inclusion Properties for Multi-Level Cache Hierarchies." ISCA '88.

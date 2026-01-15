= Cache Hierarchy

Memory hierarchy bridges the speed gap between fast CPU registers and slow main memory. Modern processors have 3-4 levels of cache between registers and DRAM.

*See also:* CPU Fundamentals (for register access), Memory System (for DRAM organization), Virtual Memory (for address translation), Multicore (for cache coherence)

== The Memory Wall

The memory wall problem arose because CPU speeds increased at approximately 50% per year from 1980 to 2005, while DRAM latency improved by only 7% per year [Hennessy & Patterson 2017]. This disparity created an ever-widening memory latency gap. In 1980, a memory access cost just one CPU cycle, but by 2020, the same access required 200 cycles due to the dramatic difference in improvement rates.

```
Year  CPU Cycle Time  DRAM Latency  Latency (cycles)
1980      250 ns         250 ns            1
1990       50 ns         200 ns            4
2000       10 ns         100 ns           10
2010        3 ns          80 ns           27
2020        0.3 ns        60 ns          200
```

Without caches, every memory access would incur a 200-cycle stall, causing the CPU to spend 99% of its time waiting. With caches, assuming a 95% hit rate and 4-cycle L1 latency, the average access time becomes 0.95×4 + 0.05×200 = 13.8 cycles, providing a 14x speedup.

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

The cache line is the fundamental unit of transfer between cache levels, with a standard size of 64 bytes across all modern CPUs. This size was chosen for several reasons: spatial locality means adjacent data is often accessed together, bus efficiency benefits from amortizing the address transfer overhead across 64 bytes of data, and it represents an optimal tradeoff where larger sizes would waste bandwidth on unused data while smaller sizes would require more frequent transfers.

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

Direct-mapped caches suffer from a critical problem: when two addresses map to the same index, conflict misses occur. A pathological case arises when the stride equals the cache size, resulting in zero hit rate:

```cpp
// Pathological case: stride = cache_size
int arr[8192];  // 32 KB
for (int i = 0; i < 8192; i += 512) {
    arr[i] = 0;  // Every access maps to same cache line → 0% hit rate!
}
```

Set-associative caches address this problem by providing N cache lines per set. For example, a 32 KB cache with 64-byte lines and 8-way associativity uses 6 bits for the offset, 6 bits for the set index (giving 64 sets), and the remaining bits for the tag. The lookup process uses the index to select a set of 8 lines, then checks all 8 tags in parallel.

```
N-way set associative: N cache lines per set

Address: [Tag | Set Index | Offset]

Example: 32 KB, 64-byte lines, 8-way associative
- Offset: 6 bits
- Index: 6 bits (32K / 64 / 8 = 64 sets = 2^6)
- Tag: remaining bits

Lookup: Index selects set of 8 lines, check all 8 tags in parallel
```

The choice of associativity involves tradeoffs: higher associativity reduces conflict misses but increases lookup complexity and latency. L1 caches typically use 8-way associativity to maintain fast lookups, L2 caches use 8-16 way, while L3 caches can afford 16-20 way associativity given their larger capacity and less stringent latency requirements. Fully associative caches, where any line can be placed anywhere, are used for small structures like TLBs that require fast lookups.

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

Cache hierarchies can be organized as either inclusive or exclusive. In Intel's inclusive design, the L3 cache contains everything present in L1 and L2 caches. This approach simplifies coherence protocols since checking only the L3 is sufficient, but wastes capacity due to duplication. AMD's Zen architecture uses an exclusive design where L3 contains only data evicted from L1 and L2, providing an effective capacity equal to the sum of all cache levels, though at the cost of more complex coherence management.

== Three C's of Cache Misses

Compulsory or cold misses occur on the first access to data and are unavoidable by definition.

Capacity misses happen when the working set exceeds the cache size. For example, when processing an 8 MB array with only a 4 MB L3 cache, each access evicts previously loaded data, resulting in misses on every access:

```cpp
// Working set = 8 MB, L3 = 4 MB → capacity misses
int arr[2000000];  // 8 MB
for (int i = 0; i < 2000000; i++) {
    arr[i] = 0;  // Evict previous data, miss on every access
}
```

Conflict misses occur when multiple addresses map to the same cache set despite sufficient total capacity. This thrashing behavior can dramatically degrade performance:

```cpp
// Two arrays, same alignment mod cache size
int *a = malloc(1048576);  // 1 MB at address X
int *b = malloc(1048576);  // 1 MB at address X + 8MB (conflicts with a)

for (int i = 0; i < N; i++) {
    result += a[i] + b[i];  // a[i] and b[i] conflict → thrashing
}
```

Solutions include padding arrays to change alignment, aligning them differently in memory, or using cache-oblivious algorithms that perform well regardless of cache parameters.

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

Prefetch distance, which determines how far ahead data is fetched, is tunable with typical values of 10-20 cache lines. The effectiveness of prefetching varies dramatically by access pattern: sequential access eliminates 90-95% of misses, constant-stride access eliminates 80-90%, while random access patterns see no benefit due to the absence of detectable patterns.

Software prefetching allows explicit control through compiler intrinsics. The parameters specify the address to prefetch, whether the access will be a write (1) or read (0), and the temporal locality level (0-3):

```cpp
// Explicit prefetch hint
for (int i = 0; i < N; i++) {
    __builtin_prefetch(&arr[i + 16], 0, 3);  // Prefetch 16 ahead
    process(arr[i]);
}

// Parameters: address, write(1)/read(0), temporal locality (0-3)
```

Software prefetch is most beneficial for linked lists where next pointers are unpredictable, complex data structures, and cases where future accesses are known but not detectable by hardware prefetchers. However, prefetching consumes cache bandwidth, and over-prefetching can evict useful data, potentially harming performance.

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

*Modern CPU cache sizes (2023-2024):*

```
Intel Raptor Lake (13th gen):
- L1: 48 KB I-cache + 32 KB D-cache per P-core
- L2: 2 MB per P-core, 4 MB shared per E-core cluster
- L3: 36 MB shared (inclusive)

AMD Zen 4 (Ryzen 7000):
- L1: 32 KB I-cache + 32 KB D-cache per core
- L2: 1 MB per core (exclusive)
- L3: 32 MB shared per CCD (exclusive)
- Effective capacity: L2 + L3 = 40 MB (due to exclusivity)

Apple M3:
- L1: 192 KB I-cache + 128 KB D-cache per P-core
- L2: 16 MB shared per core cluster
- L3 (System Level Cache): 24-32 MB
```

== Real-World Optimization Examples

*Example 1: Matrix transpose optimization*

```c
// BAD: Column-major access (cache miss on every access)
void transpose_naive(float* A, float* B, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B[j*n + i] = A[i*n + j];  // B[j][i] = A[i][j]
}
// Performance: ~50 cycles per element (L1 miss rate: 90%)

// GOOD: Blocked transpose (tile fits in cache)
void transpose_blocked(float* A, float* B, int n) {
    const int BLOCK = 32;  // 32×32 floats = 4 KB (fits in L1)
    for (int i = 0; i < n; i += BLOCK)
        for (int j = 0; j < n; j += BLOCK)
            for (int ii = i; ii < min(i+BLOCK, n); ii++)
                for (int jj = j; jj < min(j+BLOCK, n); jj++)
                    B[jj*n + ii] = A[ii*n + jj];
}
// Performance: ~4 cycles per element (L1 miss rate: 10%)
// Speedup: 12x
```

*Example 2: Structure of Arrays vs Array of Structures*

```c
// BAD: Array of Structures (poor spatial locality)
struct Particle {
    float x, y, z;     // Position
    float vx, vy, vz;  // Velocity
    float mass;        // 28 bytes per particle
    float padding;     // 32 bytes with padding
};

struct Particle particles[10000];

void update_positions(struct Particle* p, int n, float dt) {
    for (int i = 0; i < n; i++) {
        p[i].x += p[i].vx * dt;  // Loads entire 32-byte struct
        p[i].y += p[i].vy * dt;  // Only use 12 bytes, waste 20 bytes
        p[i].z += p[i].vz * dt;  // bandwidth per cache line!
    }
}
// Cache line utilization: 37.5% (12/32 bytes used)

// GOOD: Structure of Arrays (perfect spatial locality)
struct ParticlesSoA {
    float x[10000];
    float y[10000];
    float z[10000];
    float vx[10000];
    float vy[10000];
    float vz[10000];
};

void update_positions_soa(struct ParticlesSoA* p, int n, float dt) {
    for (int i = 0; i < n; i++) {
        p->x[i] += p->vx[i] * dt;  // Sequential access
        p->y[i] += p->vy[i] * dt;  // Perfect prefetching
        p->z[i] += p->vz[i] * dt;  // 100% cache line utilization
    }
}
// Cache line utilization: 100% (16 floats per 64-byte line)
// Speedup: 2-3x
```

*Example 3: Loop fusion for cache reuse*

```c
// BAD: Multiple passes (evict data between loops)
for (int i = 0; i < n; i++)
    a[i] = b[i] + c[i];

for (int i = 0; i < n; i++)
    d[i] = a[i] * 2.0f;

for (int i = 0; i < n; i++)
    e[i] = d[i] + 1.0f;
// Problem: 'a' and 'd' evicted from cache between loops

// GOOD: Fused loop (keep data in cache)
for (int i = 0; i < n; i++) {
    float temp1 = b[i] + c[i];
    float temp2 = temp1 * 2.0f;
    e[i] = temp2 + 1.0f;
}
// Benefit: Data stays in registers/L1, no intermediate array storage
// Speedup: 3-4x for large n
```

== Cache-Oblivious Algorithms

Cache-oblivious algorithms perform well across all cache levels without knowing cache parameters [Frigo et al. 1999].

*Cache-oblivious matrix multiply:*

```c
// Recursive divide-and-conquer automatically adapts to cache size
void matmul_recursive(float* A, float* B, float* C,
                      int n, int i0, int j0, int k0) {
    if (n <= 32) {  // Base case: small enough for L1
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++)
                for (int j = 0; j < n; j++)
                    C[(i0+i)*N + (j0+j)] +=
                        A[(i0+i)*N + (k0+k)] * B[(k0+k)*N + (j0+j)];
    } else {
        int m = n / 2;  // Divide into quadrants
        // Recursively multiply 8 subproblems
        matmul_recursive(A, B, C, m, i0,   j0,   k0);
        matmul_recursive(A, B, C, m, i0,   j0,   k0+m);
        // ... 6 more recursive calls
    }
}
// Automatically optimal for L1, L2, L3 without tuning!
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

== Debugging Cache Issues

*Detecting cache problems:*

```bash
# Quick cache health check
perf stat -e cycles,instructions,L1-dcache-load-misses,LLC-load-misses ./program

# Rule of thumb:
# L1 miss rate > 10%: Cache thrashing, check data layout
# LLC miss rate > 30%: Working set too large, consider blocking

# Find hot cache-missing code
perf record -e mem_load_retired.l3_miss ./program
perf report --sort=symbol,dso --stdio

# Identify false sharing (multicore)
perf c2c record -a -- ./program
perf c2c report --stdio
# Look for high HITM (cache line modified by another core)
```

*Common cache-related bugs:*

```c
// 1. Unintentional cache line sharing
struct SharedData {
    int counter1;  // Thread 1 writes
    int counter2;  // Thread 2 writes (same cache line → false sharing!)
} __attribute__((packed));

// Fix: Pad to separate cache lines
struct SharedData {
    int counter1;
    char pad1[60];  // Force new cache line
    int counter2;
    char pad2[60];
} __attribute__((aligned(64)));

// 2. Cache pollution from streaming data
void process_stream(char* huge_buffer, size_t size) {
    for (size_t i = 0; i < size; i++) {
        process(huge_buffer[i]);  // Evicts useful cache data!
    }
}

// Fix: Use non-temporal stores (bypass cache)
void process_stream_nt(char* huge_buffer, size_t size) {
    for (size_t i = 0; i < size; i += 64) {
        _mm_stream_si64((long long*)&huge_buffer[i], 0);
    }
}

// 3. Power-of-2 stride conflict
int matrix[1024][1024];  // 1024 * 4 bytes = 4096 bytes = cache size!
for (int i = 0; i < 1024; i++)
    sum += matrix[i][0];  // Every access conflicts in same cache set!

// Fix: Add padding to break power-of-2 alignment
int matrix[1024][1025];  // Extra column breaks alignment
```

*Profiling specific cache issues:*

```bash
# Cache line splits (unaligned access crosses cache line)
perf stat -e mem_inst_retired.split_loads,mem_inst_retired.split_stores ./program

# Hardware prefetcher effectiveness
perf stat -e l1d_pend_miss.pending,l1d_pend_miss.fb_full ./program

# Memory-level parallelism
perf stat -e cycle_activity.stalls_l1d_miss,\
cycle_activity.stalls_l2_miss,\
cycle_activity.stalls_l3_miss ./program
```

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

Frigo, M., Leiserson, C.E., Prokop, H., & Ramachandran, S. (1999). "Cache-Oblivious Algorithms." FOCS '99.

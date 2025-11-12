= Performance Analysis and Measurement

Understanding where time is spent requires measurement tools and techniques. Hardware performance counters provide low-overhead visibility into microarchitectural events.

*See also:* CPU Fundamentals (for IPC metrics), Cache Hierarchy (for miss rates), Branch Prediction (for misprediction rates)

== Performance Counters

Hardware counters are dedicated CPU registers that track execution events. Common events include cycles (clock cycles elapsed), instructions (instructions retired), branches (branch instructions executed), branch-misses (mispredicted branches), L1-dcache-loads (L1 data cache loads), L1-dcache-load-misses (L1 data cache misses), LLC-loads (last-level cache loads), and LLC-load-misses (last-level cache misses).

```
Common events:
- cycles: Clock cycles elapsed
- instructions: Instructions retired
- branches: Branch instructions
- branch-misses: Mispredicted branches
- L1-dcache-loads: L1 data cache loads
- L1-dcache-load-misses: L1 data cache misses
- LLC-loads: Last-level cache loads
- LLC-load-misses: Last-level cache misses
```

*Linux perf:*

```bash
# Basic stats
perf stat ./program

# Output:
#  1,234,567,890 cycles
#  1,000,000,000 instructions    # IPC = 1000M / 1234M = 0.81
#     50,000,000 branches
#      2,500,000 branch-misses   # 5% mispredict rate
#    100,000,000 L1-dcache-loads
#      5,000,000 L1-dcache-load-misses  # 5% miss rate

# Multiple events
perf stat -e cycles,instructions,cache-misses,branch-misses ./program

# Per-thread
perf stat --per-thread ./program
```

== Sampling and Profiling

Statistical sampling periodically records the program counter (PC) of the executing instruction to identify performance hotspots. The `perf record -g` command records a profile with callgraph information, `perf report` displays hot functions and call chains, and `perf annotate` shows the assembly code annotated with sample counts for a specific function.

```bash
# Record profile
perf record -g ./program  # -g = callgraph

# View report
perf report
# Shows hot functions and call chains

# Annotate source
perf annotate function_name
# Shows assembly with sample counts
```

*Example output:*

```
Overhead  Command  Shared Object  Symbol
  45.23%  program  program        [.] process_data
  23.45%  program  libc.so.6      [.] memcpy
  12.34%  program  program        [.] compute_result
```

== Top-Down Microarchitecture Analysis (TMAM)

*Intel Top-Down method [Yasin 2014]:* Categorize performance bottlenecks.

```
Level 1:
├─ Frontend Bound: Instruction fetch/decode bottleneck
├─ Backend Bound: Execution bottleneck
├─ Bad Speculation: Branch mispredicts, pipeline flushes
└─ Retiring: Useful work (higher is better!)

Level 2 (Frontend Bound):
├─ I-cache misses
├─ ITLB misses
└─ Decode stalls

Level 2 (Backend Bound):
├─ Memory Bound: Cache/TLB misses, memory latency
└─ Core Bound: Execution unit saturation, dependencies
```

*Using TMAM:*

```bash
# Intel VTune
vtune -collect hotspots ./program

# Linux perf (limited TMAM support)
perf stat --topdown -a -- ./program

# Output:
# retiring: 40%  (useful work)
# bad speculation: 10%  (branch mispredicts)
# frontend bound: 20%  (instruction fetch bottleneck)
# backend bound: 30%  (execution bottleneck)
```

== Cache Miss Analysis

*Identifying cache bottlenecks:*

```bash
# Cache hierarchy stats
perf stat -e L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses ./program

# Detailed cache events (Intel)
perf stat -e mem_load_retired.l1_hit,\
mem_load_retired.l2_hit,\
mem_load_retired.l3_hit,\
mem_load_retired.l3_miss ./program

# Cache-to-cache transfer (false sharing detection)
perf c2c record ./program
perf c2c report
```

*Interpreting results:*

```
L1 miss rate < 5%: Good
L1 miss rate > 10%: Cache optimization needed

LLC miss rate < 10%: Good
LLC miss rate > 30%: Memory-bound, consider:
- Improve locality (blocking, tiling)
- Reduce working set
- Prefetching hints
```

== Branch Prediction Analysis

```bash
# Branch statistics
perf stat -e branches,branch-misses ./program

# Mispredict rate
# Good: < 5%
# OK: 5-10%
# Poor: > 10%

# Detailed branch events
perf record -e branch-misses ./program
perf report  # Shows functions with most mispredicts

# Branch trace
perf record -e branches:pp ./program
perf script  # Detailed branch history
```

== Memory Bandwidth

```bash
# Intel Memory Latency Checker
mlc --bandwidth_matrix

# Output: Bandwidth (GB/s) between nodes
#        Node 0  Node 1
# Node 0  35.2    18.4
# Node 1  18.3    35.1

# Stream benchmark
./stream.exe
# Measures copy, scale, add, triad bandwidth
```

*Typical results (DDR4-2400, dual-channel):*

```
Copy:  ~34 GB/s (read + write)
Scale: ~33 GB/s (read + write)
Add:   ~32 GB/s (2 reads + write)
Triad: ~31 GB/s (2 reads + write + FMA)

Theoretical: 38.4 GB/s
Efficiency: ~85%
```

== TLB Analysis

```bash
# TLB miss rates
perf stat -e dTLB-loads,dTLB-load-misses,iTLB-loads,iTLB-load-misses ./program

# If TLB miss rate > 1%: Consider huge pages
```

*Huge page benefit:*

```bash
# Measure with 4 KB pages
perf stat ./program
# dTLB-load-misses: 10,000,000

# Enable transparent huge pages
echo always > /sys/kernel/mm/transparent_hugepage/enabled

# Measure again
perf stat ./program
# dTLB-load-misses: 500,000 (20× reduction!)
```

== Cycle Accounting

*Stall analysis (Intel):*

```bash
perf stat -e cycles,stalled-cycles-frontend,stalled-cycles-backend ./program

# Frontend stalls: Instruction fetch/decode bottleneck
#   - I-cache misses
#   - Branch mispredicts
#   - Decode bandwidth

# Backend stalls: Execution bottleneck
#   - Data cache misses
#   - Long-latency operations (divide, sqrt)
#   - Execution port contention
```

== Bottleneck Decision Tree

```
Step 1: Quick health check
perf stat -e cycles,instructions,cache-misses,branch-misses ./program

IPC = instructions / cycles

IPC < 0.5: Severe bottleneck
├─ Cache misses > 5%?
│  ├─ Yes: Memory-bound (see Step 2a)
│  └─ No: Check branch-misses (see Step 2b)

IPC 0.5-1.5: Moderate bottleneck
├─ Frontend stalls > 20%?
│  ├─ Yes: Instruction fetch bottleneck (see Step 3a)
│  └─ No: Backend bottleneck (see Step 3b)

IPC 1.5-2.5: Good performance
└─ Optimization opportunities:
   ├─ Vectorization (see Step 4a)
   ├─ Multicore parallelism (see Step 4b)
   └─ Algorithm improvements

IPC > 2.5: Excellent performance
└─ Check system-level bottlenecks:
   ├─ Memory bandwidth saturation
   ├─ I/O wait time
   └─ Multicore scaling

---

Step 2a: Memory-bound diagnosis
perf stat -e L1-dcache-load-misses,LLC-load-misses,dTLB-load-misses ./program

├─ L1 miss rate > 10%?
│  ├─ Yes: Poor data locality
│  │   Solutions:
│  │   - Use cache blocking/tiling
│  │   - Improve data layout (AoS → SoA)
│  │   - Reduce working set size
│  └─ No: Check L3
├─ LLC miss rate > 30%?
│  ├─ Yes: Working set exceeds LLC
│  │   Solutions:
│  │   - Algorithmic changes
│  │   - Streaming optimizations
│  │   - Non-temporal stores for write-only data
│  └─ No: Check TLB
└─ TLB miss rate > 1%?
    └─ Yes: Enable huge pages
        echo always > /sys/kernel/mm/transparent_hugepage/enabled

Step 2b: Branch misprediction diagnosis
perf record -e branch-misses:pp ./program
perf report

├─ Mispredict rate > 10%?
│  Solutions:
│  - Profile-Guided Optimization (PGO)
│  - Eliminate branches (use CMOV, masks)
│  - Sort data for predictability
│  - Reduce branch count

Step 3a: Frontend bottleneck diagnosis
perf stat -e idq_uops_not_delivered.core,\
icache_64b.iftag_miss ./program

├─ I-cache misses high?
│  Solutions:
│  - PGO for better code layout
│  - Reduce code size
│  - Link-time optimization (LTO)
└─ Branch mispredicts?
    See Step 2b

Step 3b: Backend bottleneck diagnosis
perf stat -e cycle_activity.stalls_mem_any,\
cycle_activity.stalls_total ./program

├─ Memory stalls > 50%?
│  └─ See Step 2a (memory-bound)
└─ Execution stalls?
    ├─ Long-latency ops (div, sqrt)?
    │   Solutions:
    │   - Replace division with multiplication
    │   - Use faster approximations
    └─ Data dependencies?
        Solutions:
        - Loop unrolling
        - Break dependency chains
        - Increase ILP

Step 4a: Vectorization check
gcc -O3 -march=native -fopt-info-vec-missed code.c

├─ Loops not vectorized?
│  Solutions:
│  - Add __restrict to pointers
│  - Remove loop-carried dependencies
│  - Simplify control flow
│  - Use #pragma GCC ivdep
└─ Verify SIMD usage:
    perf stat -e fp_arith_inst_retired.256b_packed_single ./program

Step 4b: Multicore scalability
perf stat -e cache-misses,cycles,instructions -a ./program

├─ Cache misses scale linearly with threads?
│  └─ False sharing problem
│      Solutions:
│      - Pad data to cache line boundaries
│      - Thread-local storage
│      - perf c2c to identify hot lines
└─ Speedup < 0.8 * num_threads?
    ├─ Load imbalance
    ├─ Synchronization overhead
    └─ Memory bandwidth saturation
```

== Diagnostic Workflow

*Complete performance investigation:*

```bash
#!/bin/bash
# Step 1: Baseline measurement
echo "=== Baseline ==="
perf stat -e cycles,instructions,cache-references,cache-misses,\
branches,branch-misses,L1-dcache-load-misses,LLC-load-misses ./program

# Step 2: Identify hot functions
echo "=== Hot Functions ==="
perf record -g ./program
perf report --stdio | head -30

# Step 3: Cache analysis
echo "=== Cache Analysis ==="
perf stat -e L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,dTLB-load-misses ./program

# Step 4: Branch analysis
echo "=== Branch Analysis ==="
perf stat -e branches,branch-misses,\
br_misp_retired.all_branches ./program

# Step 5: Pipeline stalls
echo "=== Pipeline Stalls ==="
perf stat -e cycles,stalled-cycles-frontend,\
stalled-cycles-backend ./program

# Step 6: Memory bandwidth
echo "=== Memory Bandwidth ==="
perf stat -e cpu/event=0xd1,umask=0x01/,\
cpu/event=0xd1,umask=0x02/ ./program

# Step 7: Top-down analysis (Intel only)
echo "=== Top-Down Microarchitecture Analysis ==="
perf stat --topdown ./program
```

*Interpreting results:*

```bash
# Example output interpretation:
# 1,234,567,890 cycles
# 1,000,000,000 instructions
# IPC = 1000M / 1234M = 0.81  ← Below ideal (1.5-2.0)

# 50,000,000 branches
# 2,500,000 branch-misses
# Miss rate = 2.5M / 50M = 5%  ← Acceptable (<5% is good)

# 100,000,000 cache-references
# 10,000,000 cache-misses
# Miss rate = 10%  ← High! Memory-bound workload

# Action: Investigate cache misses
# perf record -e mem_load_retired.l3_miss ./program
# perf report → identify functions with most L3 misses
```

== Microbenchmarking

*Accurate timing:*

```c
#include <time.h>

struct timespec start, end;
clock_gettime(CLOCK_MONOTONIC, &start);

// Code to measure
for (int i = 0; i < N; i++) {
    operation();
}

clock_gettime(CLOCK_MONOTONIC, &end);

double elapsed = (end.tv_sec - start.tv_sec) +
                 (end.tv_nsec - start.tv_nsec) * 1e-9;
printf("Time per iteration: %.2f ns\n", elapsed * 1e9 / N);
```

*Pitfalls:*

```c
// BAD: Compiler optimizes away
for (int i = 0; i < N; i++) {
    result = operation();  // Dead code elimination if result unused
}

// GOOD: Prevent optimization
volatile int sink;
for (int i = 0; i < N; i++) {
    sink = operation();  // Volatile prevents elimination
}

// BETTER: Use Google Benchmark library
static void BM_Operation(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(operation());
    }
}
BENCHMARK(BM_Operation);
```

== Tools Summary

| Tool | Purpose | Platform |
|:-----|:--------|:---------|
| perf | General profiling, counters | Linux |
| Intel VTune | Advanced profiling, TMAM | Linux, Windows |
| AMD uProf | AMD-specific profiling | Linux, Windows |
| valgrind/cachegrind | Cache simulation | Linux, macOS |
| gprof | Function-level profiling | Linux, macOS |
| Instruments | macOS profiling | macOS |
| ETW/WPA | Windows profiling | Windows |

== References

Yasin, A. (2014). "A Top-Down Method for Performance Analysis and Counters Architecture." ISPASS '14.

Gregg, B. (2013). Systems Performance: Enterprise and the Cloud. Prentice Hall. Chapter 6 (CPUs).

Intel Corporation (2023). Intel 64 and IA-32 Architectures Optimization Reference Manual. Appendix B (Performance Monitoring Events).

Levinthal, D. (2009). "Performance Analysis Guide for Intel Core i7 Processor and Intel Xeon 5500 processors." Intel Corporation.

= Performance Analysis and Measurement

Understanding where time is spent requires measurement tools and techniques. Hardware performance counters provide low-overhead visibility into microarchitectural events.

*See also:* CPU Fundamentals (for IPC metrics), Cache Hierarchy (for miss rates), Branch Prediction (for misprediction rates)

== Performance Counters

*Hardware counters:* CPU tracks events with dedicated registers.

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

*Statistical sampling:* Periodically record PC (program counter) of executing instruction.

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
IPC < 1.0:
├─ Frontend stalls > 20%?
│  ├─ Yes: I-cache miss or branch mispredict
│  └─ No: Check backend
├─ Backend stalls > 50%?
│  ├─ L3 miss rate > 10%?
│  │  ├─ Yes: Memory-bound → improve locality
│  │  └─ No: Core-bound → reduce dependencies
│  └─ Branch mispredict > 5%?
│     ├─ Yes: Improve predictability
│     └─ No: Check long-latency ops (div, sqrt)

IPC > 2.5:
└─ Instruction-level parallelism good, check:
   ├─ Memory bandwidth saturated?
   ├─ SIMD vectorization opportunities?
   └─ Multicore scaling?
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

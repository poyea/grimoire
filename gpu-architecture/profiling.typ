= GPU Profiling and Debugging

Effective GPU optimization requires measurement. This section covers profiling tools, metrics interpretation, and debugging techniques for CUDA applications.

*See also:* Performance Optimization (for optimization techniques), Execution Model (for understanding metrics), Memory Hierarchy (for memory-related metrics)

== Profiling Tools Overview

```
Tool               Vendor   Use Case
─────────────────────────────────────────────────────────────────────
Nsight Compute     NVIDIA   Kernel-level profiling, metrics
Nsight Systems     NVIDIA   System-wide timeline, API traces
nvprof (legacy)    NVIDIA   Command-line profiling
rocprof            AMD      AMD GPU profiling
omniperf           AMD      Advanced AMD profiling
Intel VTune        Intel    Intel GPU profiling
```

== NVIDIA Nsight Compute

Nsight Compute (ncu) provides detailed kernel-level analysis.

*Basic usage:*

```bash
# Profile all kernels
ncu ./program

# Profile specific kernel
ncu --kernel-name "regex:matmul.*" ./program

# Full metric collection
ncu --set full ./program

# Save to file
ncu -o profile ./program
ncu-ui profile.ncu-rep  # Open in GUI
```

*Common profiling commands:*

```bash
# Memory throughput analysis
ncu --section MemoryWorkloadAnalysis ./program

# Compute throughput analysis
ncu --section ComputeWorkloadAnalysis ./program

# Occupancy analysis
ncu --section Occupancy ./program

# Warp state analysis
ncu --section WarpStateStatistics ./program

# Source-level analysis
ncu --section SourceCounters ./program

# All sections
ncu --set full ./program
```

*Key metrics interpretation:*

```
Metric                          Good Value    Meaning
──────────────────────────────────────────────────────────────────────
SM Throughput (%)               > 80%         Compute utilization
Memory Throughput (%)           > 60%         Memory bandwidth used
Achieved Occupancy (%)          > 50%         Active warps / max warps
Warp Execution Efficiency (%)   > 80%         Non-divergent execution
Memory L1/L2 Hit Rate (%)       > 50%         Cache effectiveness

Roofline Analysis:
- Point below roofline: Not hitting limits
- Point on memory roof: Memory-bound
- Point on compute roof: Compute-bound
```

*Example output analysis:*

```
Section: GPU Speed Of Light Throughput
───────────────────────────────────────────────────────────────────────
DRAM Frequency             cycle/nsecond       9.75
SM Frequency               cycle/nsecond       1.98
Elapsed Cycles             cycle               5,231
Memory [%]                 %                   72.31    ← Memory-bound!
DRAM Throughput            %                   72.31
Duration                   usecond             2.64
L1/TEX Cache Throughput    %                   45.12
L2 Cache Throughput        %                   48.56
SM Active Cycles           cycle               4,892
Compute (SM) [%]           %                   38.42    ← Low compute

Analysis: Kernel is memory-bound (72% memory, 38% compute)
Action: Improve cache hit rates, reduce memory traffic
```

== NVIDIA Nsight Systems

Nsight Systems provides system-wide timeline view for understanding application behavior.

*Basic usage:*

```bash
# Trace application
nsys profile ./program

# With CUDA API tracing
nsys profile --trace=cuda,nvtx ./program

# With GPU metrics
nsys profile --gpu-metrics-device=0 ./program

# Save to file
nsys profile -o report ./program
nsys-ui report.nsys-rep  # Open in GUI
```

*Analyzing timeline:*

```
Common issues visible in timeline:
1. Kernel launch gaps (CPU idle time)
2. Synchronization points (cudaDeviceSynchronize)
3. Memory transfer bottlenecks
4. Low GPU utilization
```

*NVTX annotations:*

```cpp
#include <nvtx3/nvToolsExt.h>

void my_function() {
    nvtxRangePush("Phase 1: Data loading");
    load_data();
    nvtxRangePop();

    nvtxRangePush("Phase 2: Computation");
    compute();
    nvtxRangePop();

    nvtxRangePush("Phase 3: Output");
    save_results();
    nvtxRangePop();
}

// Or use RAII
nvtxRangePushA("My Region");
// ... code ...
nvtxRangePop();

// Link with: -lnvToolsExt
```

== Performance Counters and Metrics

*Hardware counters:*

```bash
# List available metrics
ncu --query-metrics

# Collect specific metrics
ncu --metrics \
    sm__cycles_elapsed.avg,\
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    dram__bytes.sum,\
    l1tex__t_bytes.sum \
    ./program
```

*Key metric categories:*

```
Memory Metrics:
─────────────────────────────────────────────────────────────────────
dram__bytes.sum                    Total DRAM bytes transferred
l1tex__t_bytes.sum                 L1 bytes transferred
lts__t_bytes.sum                   L2 bytes transferred
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum   Global loads
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum   Global stores

Compute Metrics:
─────────────────────────────────────────────────────────────────────
sm__sass_thread_inst_executed.sum  Instructions executed
sm__inst_executed.sum              Instructions per SM
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum   FP32 adds
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum   FP32 muls
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum   FP32 FMAs

Occupancy Metrics:
─────────────────────────────────────────────────────────────────────
sm__warps_active.avg.pct_of_peak_sustained_active   Active occupancy
sm__maximum_warps_per_active_cycle                   Theoretical max
launch__registers_per_thread                         Registers used
launch__shared_mem_per_block_allocated              Shared mem used
```

*Calculating efficiency:*

```bash
# Calculate memory bandwidth efficiency
Achieved BW = dram__bytes.sum / (kernel_time_ns / 1e9)
Efficiency = Achieved BW / Peak BW × 100%

# Example:
# dram__bytes.sum = 1,000,000,000 bytes
# kernel_time = 10 ms
# Peak BW = 1008 GB/s (RTX 4090)
# Achieved BW = 1GB / 0.01s = 100 GB/s
# Efficiency = 100 / 1008 = 9.9%  ← Poor!
```

== Memory Debugging

*Detecting memory issues:*

```bash
# Check for uncoalesced access
ncu --metrics \
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
    l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio \
    ./program

# Ideal: 4 (128 bytes / 32-byte sector)
# High values (>8): Poor coalescing

# Check for bank conflicts
ncu --metrics \
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    ./program

# Ideal: 0
```

*CUDA-memcheck / Compute Sanitizer:*

```bash
# Memory error detection
compute-sanitizer --tool memcheck ./program

# Race condition detection
compute-sanitizer --tool racecheck ./program

# Memory initialization check
compute-sanitizer --tool initcheck ./program

# Synchronization checking
compute-sanitizer --tool synccheck ./program

# Example output:
# ========= Invalid __global__ write of size 4
# =========     at 0x00000070 in kernel(float*, int)
# =========     by thread (256,0,0) in block (0,0,0)
# =========     Address 0x7f5c30000400 is out of bounds
```

== Warp Divergence Analysis

*Detecting divergence:*

```bash
# Warp execution efficiency
ncu --metrics \
    smsp__thread_inst_executed_per_inst_executed.ratio \
    ./program

# Ideal: 32 (all 32 threads active)
# Lower values indicate divergence

# Branch efficiency
ncu --metrics \
    smsp__sass_average_branch_targets_threads_uniform.pct \
    ./program

# High percentage = uniform branches (good)
```

*Source-level analysis:*

```bash
# Source correlation
ncu --section SourceCounters \
    --source-folders /path/to/source \
    ./program

# Shows per-line metrics including:
# - Instructions executed
# - Memory traffic
# - Stall reasons
```

== Occupancy Analysis

*Theoretical vs achieved occupancy:*

```bash
# Occupancy metrics
ncu --section Occupancy ./program

# Output includes:
# - Theoretical Occupancy: Max possible based on resources
# - Achieved Occupancy: Actual runtime occupancy
# - Limiting factors: Registers, shared memory, block size
```

*Occupancy calculator:*

```cpp
// Runtime occupancy query
int blockSize = 256;
int minGridSize;
int maxBlockSize;

cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize,
                                    kernel, 0, 0);

int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
                                               kernel, blockSize, 0);

int device;
cudaGetDevice(&device);
cudaDeviceProp props;
cudaGetDeviceProperties(&props, device);

float occupancy = (numBlocks * blockSize) /
                  (float)(props.maxThreadsPerMultiProcessor);

printf("Occupancy: %.2f%%\n", occupancy * 100);
```

== Latency Analysis

*Stall reasons:*

```bash
ncu --metrics \
    smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
    smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
    smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
    smsp__warp_issue_stalled_wait_per_warp_active.pct,\
    smsp__warp_issue_stalled_membar_per_warp_active.pct \
    ./program

# Stall reasons:
# - barrier: Waiting at __syncthreads()
# - long_scoreboard: Waiting for global/local memory
# - short_scoreboard: Waiting for shared memory
# - wait: Waiting for dependency
# - membar: Waiting for memory fence
```

*Interpreting stalls:*

```
Stall Type              Cause                    Fix
─────────────────────────────────────────────────────────────────────
Barrier                 __syncthreads()          Reduce sync points
Long Scoreboard         Global memory latency    More ILP, prefetching
Short Scoreboard        Shared memory latency    Avoid bank conflicts
Wait                    Instruction dependency   Break dependency chains
Math Pipe Throttle      Compute unit busy        Not necessarily bad
Membar                  Memory fences            Reduce fence usage
Not Selected            Low occupancy            Increase parallelism
```

== Debugging Techniques

*Printf debugging:*

```cpp
__global__ void debug_kernel(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Limit output (only first few threads)
    if (tid < 5) {
        printf("Thread %d: data[%d] = %f\n", tid, tid, data[tid]);
    }

    // Or only first block
    if (blockIdx.x == 0 && threadIdx.x < 5) {
        printf("Block 0, Thread %d: value = %f\n", threadIdx.x, data[tid]);
    }
}

// Compile with: nvcc -arch=sm_80 -G debug.cu  (debug mode)
// Note: printf limited to ~1MB buffer, may truncate
```

*Assertions:*

```cpp
__global__ void kernel_with_asserts(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    assert(tid < n);  // Stops kernel if false
    assert(data[tid] >= 0.0f);  // Check values

    // Process...
}

// Enable with: -DNDEBUG not set
// Compile: nvcc -G kernel.cu  (assertions enabled in debug)
```

*CUDA-GDB:*

```bash
# Compile with debug info
nvcc -g -G -O0 kernel.cu -o program

# Debug with cuda-gdb
cuda-gdb ./program

# Common commands:
(cuda-gdb) break kernel_name         # Set breakpoint at kernel
(cuda-gdb) cuda thread (0,0,0)       # Select thread
(cuda-gdb) cuda block (0,0,0)        # Select block
(cuda-gdb) cuda kernel               # Show kernel info
(cuda-gdb) cuda grid                 # Show grid info
(cuda-gdb) print threadIdx.x         # Print thread index
(cuda-gdb) info cuda threads         # List threads
```

*Error checking:*

```cpp
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

kernel<<<grid, block>>>(d_data, n);
CUDA_CHECK(cudaGetLastError());  // Check for launch errors
CUDA_CHECK(cudaDeviceSynchronize());  // Check for execution errors
```

== AMD ROCm Profiling

*rocprof basics:*

```bash
# Profile application
rocprof ./program

# With metrics
rocprof --stats ./program

# Specific counters
rocprof -i metrics.txt ./program

# metrics.txt:
# pmc: SQ_WAVES,SQ_INSTS_VALU,TA_TOTAL_CYCLES
```

*omniperf (advanced):*

```bash
# Profile
omniperf profile -n my_profile ./program

# Analyze
omniperf analyze -p my_profile

# Web interface
omniperf analyze -p my_profile --gui
```

*rocgdb:*

```bash
# Debug AMD GPU code
rocgdb ./program

# Similar commands to cuda-gdb
(rocgdb) break kernel_name
(rocgdb) run
(rocgdb) info threads
```

== Performance Debugging Workflow

```
1. Establish baseline
   └─ Record initial performance (time, throughput)

2. Profile with Nsight Systems
   └─ Identify GPU utilization, gaps, bottlenecks
   └─ Check for CPU-GPU synchronization issues

3. Profile kernels with Nsight Compute
   └─ Memory-bound or compute-bound?
   └─ Occupancy analysis
   └─ Warp efficiency

4. Identify limiting factor
   ├─ Memory bandwidth → Improve access patterns
   ├─ Compute → Increase ILP, use faster instructions
   ├─ Occupancy → Reduce register/shared memory usage
   └─ Latency → More parallelism

5. Implement optimization

6. Verify improvement
   └─ Re-profile, compare to baseline
   └─ Check for regressions

7. Repeat until target performance
```

*Diagnostic decision tree:*

```
Low performance
│
├─ GPU Utilization < 80%?
│  ├─ Yes: Check timeline for gaps
│  │  ├─ CPU bottleneck → Overlap, async
│  │  ├─ Kernel launch overhead → Fuse kernels
│  │  └─ Synchronization → Remove unnecessary syncs
│  └─ No: Profile kernel details
│
├─ Memory throughput > 60% peak?
│  ├─ Yes: Memory-bound
│  │  ├─ Poor coalescing → Fix access patterns
│  │  ├─ Low cache hit rate → Improve locality
│  │  └─ High memory traffic → Reduce data movement
│  └─ No: Compute or latency bound
│
├─ Compute throughput > 60% peak?
│  ├─ Yes: Compute-bound (good!)
│  │  └─ Optimize instructions if needed
│  └─ No: Latency-bound
│
└─ Latency-bound
   ├─ Low occupancy → Increase parallelism
   ├─ High stalls → More ILP, prefetch
   └─ Divergence → Restructure control flow
```

== Common Profiling Pitfalls

*1. Profiling overhead:*

```
Profiling adds overhead (10-100%+).
Use for relative comparisons, not absolute timing.

For accurate timing:
- Use CUDA events (cudaEventRecord)
- Use NVTX with Nsight Systems (lower overhead)
- Disable profiling for final measurements
```

*2. Warmup effects:*

```cpp
// First kernel launch includes initialization overhead
kernel<<<grid, block>>>(data);  // Slow (JIT, caching, etc.)
cudaDeviceSynchronize();

// Subsequent launches are representative
for (int i = 0; i < 10; i++) {
    kernel<<<grid, block>>>(data);  // Measure these
}

// Profile with warmup
ncu --launch-skip 1 --launch-count 5 ./program
```

*3. Small kernel artifacts:*

```
Very short kernels dominated by launch overhead.
Profile with:
- Multiple iterations
- Larger problem sizes
- Grid-stride loops to increase work per launch
```

*4. Frequency scaling:*

```
GPU frequency varies with:
- Power limits
- Thermal throttling
- AVX-512 usage

Lock frequency for consistent measurements:
nvidia-smi -pm 1                    # Enable persistence mode
nvidia-smi -lgc <min>,<max>        # Lock GPU clocks
nvidia-smi --lock-memory-clocks=<freq>  # Lock memory clocks
```

== References

NVIDIA Corporation (2024). Nsight Compute Documentation. https://docs.nvidia.com/nsight-compute/

NVIDIA Corporation (2024). Nsight Systems Documentation. https://docs.nvidia.com/nsight-systems/

NVIDIA Corporation (2024). CUDA-GDB Documentation. https://docs.nvidia.com/cuda/cuda-gdb/

NVIDIA Corporation (2024). Compute Sanitizer Documentation. https://docs.nvidia.com/compute-sanitizer/

AMD (2024). ROCm Profiling Tools. https://rocm.docs.amd.com/projects/rocprofiler/

AMD (2024). Omniperf Documentation. https://rocm.docs.amd.com/projects/omniperf/

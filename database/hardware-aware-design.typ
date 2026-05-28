= Hardware-Aware Database Design

Database performance is ultimately constrained by hardware: DRAM bandwidth, NVMe latency, cache hierarchy, RDMA, and GPU compute. Modern systems are redesigned for each generation of hardware — NVMe SSDs, CXL memory pooling, and GPU OLAP represent the current frontier.

*See also:* _database/buffer-pool-and-io.typ_, _database/column-stores-and-vectorized-execution.typ_, _database/storage-engines.typ_, _database/query-compilation.typ_ (codegen + SIMD), _CPU Architecture volume_ (pipelines, NUMA, cache hierarchy), _GPU Architecture volume_ (SIMT, memory hierarchy for GPU OLAP)

== CPU and Cache Hierarchy

*Cache miss is the dominant cost* in in-memory databases. Random pointer chasing (B-Tree traversal, hash probing) causes L3 cache misses (~40 ns), while sequential array access stays in L1/L2 (\<1 ns).

#table(
  columns: (auto, auto, auto),
  [*Level*], [*Latency*], [*Size*],
  [L1 cache], [~0.5 ns],   [32–64 KB],
  [L2 cache], [~4 ns],     [256 KB – 1 MB],
  [L3 cache], [~30–40 ns], [8–64 MB],
  [DRAM],     [~80 ns],    [GB–TB],
  [NVMe SSD], [~80–200 µs],[TB–PB],
  [HDD],      [~5–10 ms],  [TB–PB],
)

*Cache-oblivious data structures* (e.g., cache-oblivious B-Tree, van Emde Boas layout) achieve optimal cache behavior at all levels without knowing the cache size.

```c
// Cache-friendly vs cache-hostile access pattern
#define N 1024
int matrix[N][N];

// Cache-hostile (column-major traversal of row-major array):
long sum_cols(int m[N][N]) {
    long s = 0;
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            s += m[i][j];   // stride = N*4 bytes → cache miss per access
    return s;
}

// Cache-friendly (row-major traversal):
long sum_rows(int m[N][N]) {
    long s = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            s += m[i][j];   // sequential → prefetched into L1
    return s;
}
// sum_rows is ~10× faster for large N due to cache efficiency
```

=== NUMA (Non-Uniform Memory Access)

Multi-socket servers have separate memory controllers. Accessing memory on a remote socket costs ~2× DRAM latency.

```c
// NUMA-aware memory allocation (Linux)
#include <numa.h>

// Allocate on the local NUMA node of the calling thread
void *buf = numa_alloc_local(size);

// Pin thread to socket 0
struct bitmask *mask = numa_allocate_cpumask();
numa_bitmask_setbit(mask, 0);   // socket 0, CPU 0
numa_sched_setaffinity(0, mask);
```

*Morsel-driven parallelism* (see _database/query-compilation.typ_) assigns morsels to threads on the socket that owns the memory — avoiding NUMA remote accesses.

== NVMe SSDs

NVMe exposes parallelism that HDDs don't have: 8–64 I/O queues, each with 64K deep slots. Sequential bandwidth: 5–14 GB/s. Random 4KB read: 100–800 µs, 500K–1M IOPS.

*Implications for database design:*
- O_DIRECT bypass: avoid double buffering (OS cache + DB buffer pool).
- io_uring: submit 64 I/O requests without a syscall per request.
- Direct-attach NVMe is 2–5× faster than network-attached storage for random IOPS.

```c
// io_uring: submit multiple async reads in one syscall batch
#include <liburing.h>

#define BATCH 16
#define PAGE_SIZE 4096

void async_read_pages(int fd, uint64_t *offsets, void **bufs, int n) {
    struct io_uring ring;
    io_uring_queue_init(64, &ring, 0);

    // Submit batch
    for (int i = 0; i < n; i++) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, fd, bufs[i], PAGE_SIZE, offsets[i]);
        sqe->user_data = i;
    }
    io_uring_submit(&ring);

    // Harvest completions
    for (int i = 0; i < n; i++) {
        struct io_uring_cqe *cqe;
        io_uring_wait_cqe(&ring, &cqe);
        // bufs[cqe->user_data] is now ready
        io_uring_cqe_seen(&ring, cqe);
    }
    io_uring_queue_exit(&ring);
}
```

*WiscKey (FAST 2016):* separates keys from values in LSM-Trees. Keys stay in the LSM log (small, sorted); values go to a separate value log (large, random). Reduces write amplification for large values at the cost of more random reads on value lookups.

== RDMA (Remote Direct Memory Access)

RDMA allows a server to read/write another server's memory without involving the remote CPU. Latency: 1–5 µs (vs 50–200 µs for TCP). Bandwidth: 100–400 Gbit/s.

*Use in databases:* distributed key-value stores, RPC replacement in OLTP, DSM (Distributed Shared Memory) for graph databases.

```c
// RDMA one-sided READ (pseudocode using libibverbs)
#include <infiniband/verbs.h>

// After connection setup, read from remote host's memory:
struct ibv_sge sge = {
    .addr   = (uint64_t)local_buf,  // local destination
    .length = READ_SIZE,
    .lkey   = mr->lkey,
};

struct ibv_send_wr wr = {
    .opcode      = IBV_WR_RDMA_READ,
    .sg_list     = &sge,
    .num_sge     = 1,
    .wr.rdma     = {
        .remote_addr = remote_addr,   // remote source (no remote CPU involvement)
        .rkey        = remote_rkey,
    },
    .send_flags  = IBV_SEND_SIGNALED,
};

struct ibv_send_wr *bad_wr;
ibv_post_send(qp, &wr, &bad_wr);  // submit RDMA read
// Poll completion queue for result
```

*FaRM (SOSP 2014, Microsoft):* distributed in-memory transactions using RDMA one-sided operations to bypass remote CPUs for reads; uses RDMA writes for replication. Achieves 4.5M TPS at 58 µs median latency on 90 machines.

== GPU Databases

GPUs have 900 GB/s HBM bandwidth (vs ~100 GB/s DRAM) and 10,000+ compute cores — ideal for data-parallel aggregation, sort, and join on large datasets.

*RAPIDS (NVIDIA):* GPU-accelerated DataFrame and SQL engine (cuDF, cuSQL).

```python
import cudf

# cuDF: pandas-like API but runs on GPU
df = cudf.read_parquet("orders_100m.parquet")

# Filter + groupby: runs on GPU via CUDA kernels
result = (
    df[df["amount"] > 100]
    .groupby("customer_id")["amount"]
    .sum()
    .reset_index()
    .sort_values("amount", ascending=False)
)
# 100M rows in ~0.2s (vs ~8s with pandas on CPU)
```

*CUDA sorting (Thrust):*

```cpp
#include <thrust/sort.h>
#include <thrust/device_vector.h>

// Sort 100M integers on GPU
thrust::device_vector<int> keys(100'000'000);
// ... fill keys ...
thrust::sort(keys.begin(), keys.end());   // radix sort on GPU: ~0.5s
// vs std::sort on CPU: ~12s
```

*Challenges for GPU databases:*
- PCIe bandwidth bottleneck: moving data CPU↔GPU costs 16–32 GB/s.
- GPU memory capacity (80 GB max for H100) limits working set size.
- GPU-resident databases (e.g., BlazingSQL, OmniSci/HeavyAI) keep data in GPU VRAM — only viable for datasets fitting in GPU memory.

== CXL (Compute Express Link)

CXL (2022+) is a PCIe-based interconnect for cache-coherent memory sharing between CPUs and accelerators. CXL 3.0 enables *memory pooling*: a shared pool of DRAM accessible by multiple hosts at NUMA-like latency (~300 ns).

*Implications:* database buffer pools can span multiple servers' CXL-attached memory. Working sets larger than a single server's RAM become feasible without network round trips.

```
CXL memory architecture (future):
  Host A (CPU + local DRAM) ─── CXL ──┐
                                        ├── Shared Memory Pool (1 TB CXL DRAM)
  Host B (CPU + local DRAM) ─── CXL ──┘

  Buffer pool can span the shared pool — no explicit message passing
  Coherence is hardware-managed (CXL.cache protocol)
```

== Persistent Memory (Intel Optane)

Optane DCPMM (discontinued 2022 but architecturally important): byte-addressable, persistent, 3× slower than DRAM, 10× faster than NVMe.

*Use in databases:* WAL on persistent memory eliminates fsync cost (write becomes clflush + mfence — ~300 ns vs 100 µs fsync on NVMe).

```c
// Persistent memory WAL write (pmdk library)
#include <libpmem.h>

uint8_t *pmem;
size_t   mapped_len;
int      is_pmem;

pmem = pmem_map_file("/mnt/pmem/wal.log", WAL_SIZE,
                     PMEM_FILE_CREATE, 0666,
                     &mapped_len, &is_pmem);

void write_log_record(const LogRecord *rec, size_t len) {
    memcpy(pmem + log_offset, rec, len);
    pmem_persist(pmem + log_offset, len);   // clflush + sfence, not fsync
    log_offset += len;
    // ~300 ns latency vs ~100 µs for NVMe fsync
}
```

== Hardware Comparison for Database Workloads

#table(
  columns: (auto, auto, auto, auto),
  [*Hardware*], [*Bandwidth*], [*Latency*], [*DB use case*],
  [DRAM],           [50–100 GB/s],  [80 ns],       [Buffer pool, in-memory DB],
  [HBM (GPU)],      [900 GB/s],     [100 ns],      [GPU analytics],
  [CXL memory],     [50 GB/s],      [200–400 ns],  [Expanded buffer pool],
  [Optane PMEM],    [50 GB/s],      [300 ns],      [WAL, persistent indexes],
  [NVMe SSD],       [7 GB/s],       [80–200 µs],   [Cold storage, large datasets],
  [RDMA (IB/RoCE)], [25–400 Gbit/s],[1–5 µs],      [Distributed in-memory DB],
  [100G Ethernet],  [12.5 GB/s],    [5–50 µs],     [Distributed DB replication],
)

== References

Graefe, G., Kimura, H. (2010). "Data Passing in Database Systems." ACM SIGMOD Record.

Leis, V. et al. (2018). "LeanStore: In-Memory Data Management Beyond Main Memory." ICDE.

Zhang, H. et al. (2015). "In-Memory Big Data Management and Processing: A Survey." TKDE.

Dragojevic, A. et al. (2014). "FaRM: Fast Remote Memory." NSDI.

Kim, S. et al. (2019). "Managing Non-Volatile Memory in Database Systems." SIGMOD.

He, B. et al. (2008). "Relational Query Co-Processing on Graphics Processors." TODS.

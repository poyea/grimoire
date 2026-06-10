= Memory Performance

On modern hardware, most programs are not compute-bound but memory-bound: a load from DRAM costs 60-100 ns, time enough for several hundred instructions. This chapter covers data layout for cache friendliness, false sharing, allocator selection and strategies, NUMA, the bandwidth/latency distinction, and the tools that make memory behavior visible.

*See also:* _CPU Profiling_ (top-down analysis identifies memory-bound code), _Concurrency Performance_ (contention on shared cache lines), and the CPU Architecture volume's _Caches_ and _Virtual Memory_ chapters (the hardware this chapter programs against).

== The Cost Hierarchy

Approximate latencies on a current x86 server (load-to-use):

#table(
  columns: 3,
  [*Level*], [*Latency*], [*Typical size*],
  [L1d hit], [4-5 cycles (about 1 ns)], [32-48 KiB per core],
  [L2 hit], [12-16 cycles], [512 KiB - 2 MiB per core],
  [L3 hit], [40-60 cycles], [tens of MiB, shared],
  [Local DRAM], [70-100 ns], [hundreds of GiB],
  [Remote-socket DRAM], [120-200 ns], [via UPI/Infinity Fabric],
)

Caches move *lines* of 64 bytes; the unit of memory performance is the line, not the byte. Two consequences dominate everything below: spatial locality (use all of a fetched line) and the hardware prefetcher (sequential and strided access is nearly free; pointer chasing pays full latency per hop).

== Data Layout: SoA vs. AoS

*Array-of-Structures (AoS)* stores records contiguously; *Structure-of-Arrays (SoA)* stores each field in its own array. If a loop touches only one or two fields of a 64-byte struct, AoS wastes most of every cache line and defeats vectorization; SoA delivers dense, unit-stride streams that prefetch perfectly and map directly onto SIMD lanes. This is why columnar formats (Arrow, Parquet) dominate analytics and why game engines (Unity DOTS, ECS architectures) restructure entities into component arrays. The hybrid *AoSoA* (tiles of SoA sized to a SIMD width) keeps locality across fields while preserving vectorizability.

Other layout levers:

- *Struct packing and field ordering*: order fields by descending alignment to eliminate padding (`pahole` shows the holes); split hot and cold fields into separate structs so hot loops touch fewer lines.
- *Pointer-heavy structures*: each pointer dereference is a potential cache miss with no prefetch help. B-trees beat binary trees in memory for the same reason they win on disk: fanout amortizes the miss. Indices into arrays beat pointers (smaller, relocatable, prefetchable).
- *Hugepages*: 2 MiB pages cut TLB misses for large heaps; transparent hugepages or explicit `madvise(MADV_HUGEPAGE)`. TLB coverage with 4 KiB pages is only a few MiB, far smaller than L3.

== False Sharing

Coherence operates on whole lines. If two threads write to *different* variables that happen to share a 64-byte line, the line ping-pongs between cores in Modified state, costing a coherence round-trip (tens of ns) per write, with neither thread logically sharing data. Classic victims: arrays of per-thread counters, adjacent mutexes, a hot atomic next to a frequently-written field.

Detection: `perf c2c record` reports lines with HITM (hit-modified) accesses and the offsets and call sites involved. Fix by padding and aligning hot per-thread data to line boundaries: `alignas(64)` in C++, `#[repr(align(64))]` or `crossbeam_utils::CachePadded` in Rust, `@Contended` on the JVM. Note that adjacent-line prefetchers can make the effective conflict granularity 128 bytes on Intel, which is why `std::hardware_destructive_interference_size` may exceed 64.

== Allocation Strategies

=== General-purpose allocators

`malloc` performance varies enormously under multithreading. *jemalloc* (per-thread caches, size-class arenas; default in FreeBSD and Redis, formerly Rust) and *tcmalloc* (Google; per-CPU caches using restartable sequences, very fast small-object paths) both avoid the global-lock behavior that makes naive allocators a scalability bottleneck. *mimalloc* (Microsoft, 2019) uses sharded free lists per page and achieves strong results with a small codebase. Switching is often a one-line `LD_PRELOAD` and can be worth 5-20% on allocation-heavy services; jemalloc additionally exposes detailed introspection (`malloc_stats_print`) and heap profiling.

=== Arenas, pools, and regions

The fastest allocation is a pointer bump, and the fastest free is freeing nothing individually:

- *Arena (region) allocation*: bump-allocate from a large block; free the whole arena at once at the end of a scope (request, frame, compilation unit). Per-request arenas are standard in servers (Apache pools, protobuf arenas) and compilers.
- *Object pools / free lists*: recycle fixed-size objects, avoiding both allocator overhead and cache-cold fresh memory.
- *Slab allocation*: the kernel's approach for fixed-size kernel objects; the same idea underlies size-class allocators.

Beyond speed, arenas improve *locality* (objects allocated together sit together) and eliminate fragmentation within the arena. The trade-off is lifetime discipline: nothing in the arena may outlive it.

=== Garbage-collected runtimes

In GC languages the levers differ: allocation rate (bytes/sec) drives GC frequency, so reducing garbage (object reuse, value types, escape analysis) is the optimization; large heaps trade pause frequency against duration; and modern collectors (G1, ZGC, Shenandoah) target pause times in the sub-millisecond range at some throughput cost.

== NUMA Effects

Multi-socket machines (and increasingly chiplet-based single sockets) have non-uniform memory access: each socket owns local DRAM, and remote access pays 1.5-2 times the latency plus interconnect bandwidth limits. Linux allocates pages on *first touch*: the thread that first writes a page determines its home node. The classic bug is a master thread initializing all memory (placing it on node 0) and worker threads on node 1 paying remote latency forever.

Remedies: initialize data in the threads that will use it (parallel first touch); pin threads (`numactl --cpunodebind`, `pthread_setaffinity_np`) and memory (`numactl --membind`, `mbind`); interleave (`numactl --interleave=all`) for bandwidth-bound, uniformly-shared workloads; or enable automatic NUMA balancing and verify it helps. Diagnose with `numastat` (remote vs. local allocation counters) and `perf stat -e node-loads,node-load-misses`. Databases and JVMs have explicit NUMA modes for this reason.

== Bandwidth-Bound vs. Latency-Bound

Two distinct ways to be memory-bound, with different fixes:

- *Latency-bound*: dependent chains of misses (pointer chasing); the core idles waiting for one line at a time. Bandwidth meters show low utilization. Fixes: better layout, fewer indirections, software prefetching (`__builtin_prefetch` 100-300 cycles ahead), more memory-level parallelism (restructure so multiple misses are in flight; out-of-order cores can sustain 10-20 outstanding misses).
- *Bandwidth-bound*: streaming through data faster than DRAM can supply, around 100-400 GB/s per socket depending on channels and generation (measure with the STREAM benchmark or `mlc`). Adding threads past saturation does nothing; per-access latency *rises* under load. Fixes: touch less data (compression, smaller types, SoA so only needed fields stream), fuse passes over the data, non-temporal stores for write-once buffers, cache blocking/tiling so each block is reused from cache.

The *roofline model* (Williams et al., 2009) makes the diagnosis quantitative: plot achieved FLOP/s against *arithmetic intensity* (FLOPs per byte); kernels left of the ridge point are bandwidth-bound and no amount of compute optimization will help them.

== Memory Profiling Tools

- *heaptrack* (Linux): traces every allocation with stacks at modest overhead; reports peak heap, allocation hotspots, temporary-allocation churn, and leaks; GUI with flame graphs of allocation sites.
- *Valgrind massif*: snapshot-based heap profiler; precise but $10-30 times$ slowdown, suited to offline analysis.
- *jemalloc/tcmalloc heap profiling*: statistical sampling of live allocations in production at negligible cost; output in pprof format.
- *Bytehound, dhat*: allocation lifetime and access-pattern analysis.
- *`perf mem` / VTune memory-access analysis*: PEBS-sampled loads with latency and data source (which cache level, which NUMA node), connecting misses to source lines and to the *data structures* responsible.
- *AddressSanitizer / LeakSanitizer*: correctness rather than performance, but the usual first stop when "memory" misbehaves.

A useful discipline: track *allocation rate* (bytes and calls per request) as a first-class metric in benchmarks; it is cheap to measure and is a leading indicator of both allocator pressure and cache pollution.

== Further Reading

- Drepper, U. (2007). What every programmer should know about memory. Red Hat technical report.
- Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: an insightful visual performance model. _CACM_, 52(4).
- Evans, J. (2006). A scalable concurrent malloc(3) implementation for FreeBSD. _BSDCan_.
- Leijen, D. et al. (2019). Mimalloc: free list sharding in action. _APLAS_.
- Lameter, C. (2013). NUMA (Non-Uniform Memory Access): an overview. _ACM Queue_, 11(7).

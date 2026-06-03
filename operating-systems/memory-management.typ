= Memory Management

Memory management spans the boundary between hardware (the MMU, TLB, cache hierarchy) and software (the kernel allocator, the runtime, the application). Done well, processes see a uniform, contiguous, large address space; done poorly, the same processes see TLB storms, NUMA penalties, and unpredictable swap stalls. This chapter is the conceptual treatment; Linux-specific reclaim and `mmap` mechanics live in `linux-kernel/memory-reclaim.typ` and `linux-kernel/mmap-memory.typ`, and the hardware path is in `cpu-architecture/virtual-memory.typ`.

*See also:* `cpu-architecture/virtual-memory.typ`, `linux-kernel/memory-reclaim.typ`, `linux-kernel/mmap-memory.typ`.

== Address Spaces and Virtual Memory

Virtual memory provides four orthogonal benefits, often conflated:

1. *Isolation* — processes cannot read each other's pages without explicit sharing.
2. *Relocation* — physical addresses are decoupled from program-visible addresses, allowing the kernel to place pages anywhere.
3. *Overcommit* — virtual address space can exceed physical RAM; unmapped or paged-out pages cost nothing until touched.
4. *Sharing* — multiple virtual mappings can refer to the same physical frame (libraries, COW, `shm`).

The hardware contract is a page table that maps virtual page numbers to physical frame numbers plus permission bits (read/write/execute/user). The MMU consults the table on every memory reference; the *Translation Lookaside Buffer* ($"TLB"$) caches recent translations to avoid the multi-level walk. Modern x86-64 has 4-level (48-bit VA) or 5-level (57-bit VA) tables, AArch64 supports 4 KB / 16 KB / 64 KB granules with up to 52-bit VAs.

== Paging Mechanics

A *page fault* fires when the MMU finds no valid translation (or a permission mismatch). The kernel fault handler examines the faulting address, the VMA covering it, and the fault type:

#table(columns: (auto, 1fr),
  [*Fault*], [*Resolution*],
  [Demand page-in], [Read page from backing store],
  [Anonymous zero-fill], [Allocate page, zero, install],
  [COW], [Allocate page, copy, install writable],
  [Stack grow], [Extend VMA, install zero page],
  [SIGSEGV], [Address not in any VMA — kill],
)

Demand paging keeps RAM working set tight at the cost of fault latency. *Pre-fault* (`MAP_POPULATE`) and *read-ahead* (clustered I/O around the faulted page) trade memory for latency.

*Page replacement* answers "which page to evict when RAM is full." Theoretical bounds: Belady's MIN (evict the page used furthest in future) is optimal but clairvoyant. Practical algorithms approximate by recency:

- *LRU* — evict least-recently-used. Exact LRU requires updating a timestamp on every access — too expensive.
- *Clock* / *Second-chance* — circular scan of pages with reference bits; the standard Unix approximation.
- *LRU-K* — track $k$ most recent accesses; less susceptible to scan-once workloads polluting the cache.
- *ARC* / *2Q* — split into recently-touched and frequently-touched lists; resistant to scan pollution.
- *MGLRU* (multi-generational LRU, Linux 6.1+) — generation-based aging with bloom-filter access detection; reduces full-scan cost on terabyte-RAM systems.

== Huge Pages and TLB Pressure

A 4 KB page touches 1024 cache lines of page-table state to map 4 MB; the $"TLB"$ holds ~64-2048 entries depending on level. A workload with > $"TLB"$ × 4 KB hot data thrashes translation. *Huge pages* (2 MB and 1 GB on x86-64; 64 KB / 2 MB / 32 MB / 1 GB on AArch64) raise the per-entry coverage and dramatically reduce $"TLB"$ misses.

Two delivery models:
- *Explicit / HugeTLB*: reserved pool, applications opt in via `MAP_HUGETLB` or `hugetlbfs`. Strict but predictable.
- *Transparent / THP*: kernel opportunistically promotes contiguous 4 KB pages into 2 MB; `khugepaged` collapses fragmented regions. Easier but introduces tail-latency surprises (allocator stalls during defragmentation).

Databases and JVMs typically prefer explicit huge pages because THP's compaction stalls show up as p99 spikes.

== NUMA

Non-Uniform Memory Access ($"NUMA"$) systems have memory partitioned across sockets; access to remote-socket memory is 1.5-3× slower than local. The OS must:

1. *Allocate locally* — first-touch policy: physical frame backing a virtual page is allocated on the node of the touching CPU.
2. *Schedule locally* — keep tasks on CPUs near their memory (see `linux-kernel/scheduler.typ`).
3. *Migrate when necessary* — Linux's *AutoNUMA* periodically samples page residency vs access patterns and migrates pages or tasks.

A subtle pitfall: a parent thread allocating buffers then handing them to worker threads on other nodes pins the memory to the parent's node. The fix is to allocate from the consumer or use `mbind` / `numactl --membind`.

== Allocators

Kernel and user-space allocators face different constraints. Kernel allocators must handle physical fragmentation, IRQ-context calls, and DMA constraints (contiguity, address ceilings).

*Buddy allocator* (kernel page allocator): physical memory split into power-of-two blocks; splits and coalesces on allocation. $O(log "max-order")$ ops; fragmentation bounded by 50% worst-case for the same order.

*Slab / SLUB / SLOB* (kernel object allocator): caches typed objects (`task_struct`, `dentry`) to amortize page-allocator cost and improve cache locality. SLUB is Linux's current default.

User-space allocators (ptmalloc, jemalloc, tcmalloc, mimalloc) optimize differently:

#table(columns: (auto, 1fr),
  [*Allocator*], [*Design*],
  [ptmalloc], [glibc default, per-thread arenas, slow on fragmentation],
  [jemalloc], [size classes, thread caches, decay-based purging],
  [tcmalloc], [Google, central freelist plus thread caches, pageheap],
  [mimalloc], [Microsoft, segments + free-deferred lists, low overhead],
  [snmalloc], [message-passing across threads, no shared locks on free],
)

Common pitfall: a producer thread `malloc`s; a consumer thread `free`s. ptmalloc punishes this (cross-arena), mimalloc and snmalloc don't.

== Swap, Compression, and Memory Tiers

When RAM is exhausted the kernel either evicts clean pages (file-backed — re-fetch from disk) or pages out dirty anonymous pages (to swap). Swap-to-disk is slow enough that modern systems prefer *compressed RAM* (`zswap`, `zram`) for the first tier of pressure and only spill to disk under sustained shortage.

CXL-attached memory introduces a new tier: 100-300 ns away, attached over PCIe lanes. Page-promotion / demotion daemons (`damon`, NUMA-balancing extensions) now schedule pages across local DRAM, CXL DRAM, and NVMe swap.

== Pitfalls

- *Overcommit* without OOM kill policy: `vm.overcommit_memory=1` lets allocations succeed beyond RAM+swap, gambling on never being touched. Useful for `fork` heavy workloads, dangerous otherwise.
- *Memory leaks vs growth*: a process that keeps allocating without freeing is a leak; one that holds an unbounded cache is *growth* — the OS cannot distinguish.
- *False sharing* in NUMA: two threads on different sockets writing distinct fields of the same cache line cause line bouncing through the interconnect.
- *Page-cache trashing*: `cp` of a large file evicts working set; `posix_fadvise(POSIX_FADV_DONTNEED)` lets bulk copies opt out.
- *MADV_FREE vs MADV_DONTNEED*: the former defers reclaim; a subsequent read returns old data until eviction. A common allocator bug source.

== Further Reading

Denning, P. (1968). "The Working Set Model for Program Behavior." CACM.

Belady, L. (1966). "A Study of Replacement Algorithms for a Virtual-Storage Computer." IBM Systems Journal.

Megiddo, N., Modha, D. (2003). "ARC: A Self-Tuning, Low Overhead Replacement Cache." FAST.

Bonwick, J. (1994). "The Slab Allocator: An Object-Caching Kernel Memory Allocator." USENIX.

Evans, J. (2006). "A Scalable Concurrent malloc(3) Implementation for FreeBSD." BSDCan (jemalloc).

Leijen, D., Zorn, B., de Moura, L. (2019). "Mimalloc: Free List Sharding in Action." APLAS.

Lameter, C., Kim, M. (2008). "The SLUB Allocator." LWN.

Tanenbaum, A., Bos, H. "Modern Operating Systems," Chapter 3.

Bovet, D., Cesati, M. "Understanding the Linux Kernel," Chapters 8-9 (cross-reference for Linux mechanics).

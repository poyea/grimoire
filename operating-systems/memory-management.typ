= Memory Management

Memory management spans the boundary between hardware (the MMU, TLB, cache hierarchy) and software (the kernel allocator, the runtime, the application). Done well, processes see a uniform, contiguous, large address space; done poorly, the same processes see TLB storms, NUMA penalties, and unpredictable swap stalls. This chapter is the conceptual treatment; Linux-specific reclaim and `mmap` mechanics live in `linux-kernel/memory-reclaim.typ` and `linux-kernel/mmap-memory.typ`, and the hardware path is in `cpu-architecture/virtual-memory.typ`.

*See also:* _Virtual Memory_ (cpu-architecture), _Memory Reclaim_ (linux-kernel), _mmap and Memory Management_ (linux-kernel).

== Address Spaces and Virtual Memory

Virtual memory provides four orthogonal benefits, often conflated:

1. *Isolation* — processes cannot read each other's pages without explicit sharing.
2. *Relocation* — physical addresses are decoupled from program-visible addresses, allowing the kernel to place pages anywhere.
3. *Overcommit* — virtual address space can exceed physical RAM; unmapped or paged-out pages cost nothing until touched.
4. *Sharing* — multiple virtual mappings can refer to the same physical frame (libraries, COW, `shm`).

The hardware contract is a page table that maps virtual page numbers to physical frame numbers plus permission bits (read/write/execute/user). The MMU consults the table on every memory reference; the *Translation Lookaside Buffer* ($"TLB"$) caches recent translations to avoid the multi-level walk. Modern x86-64 has 4-level (48-bit VA) or 5-level (57-bit VA) tables, AArch64 supports 4 KB / 16 KB / 64 KB granules with up to 52-bit VAs.

== mmap Internals

`mmap(2)` is the kernel's unified interface for mapping files, anonymous memory, shared memory, and device memory into a process's address space. The kernel represents each mapping as a *Virtual Memory Area* (`vm_area_struct` / VMA): a contiguous range of virtual addresses with uniform permissions and a pointer to the `vm_ops` function table that handles faults, msync, and unmapping. All VMAs for a process are organized in an interval tree (a red-black tree keyed on `[vm_start, vm_end)`) plus a linked list for sequential scanning; the kernel binary-searches the tree on every page fault to find the covering VMA.

Key mapping modes:

#table(columns: (auto, 1fr),
  [*Flag*], [*Behavior*],
  [`MAP_PRIVATE` + file], [COW-backed by file; writes produce anonymous private copies],
  [`MAP_SHARED` + file], [Changes visible to all mappers and written back to file on msync / eviction],
  [`MAP_ANONYMOUS`], [Zero-initialized; backed by swap or compressed RAM, no file],
  [`MAP_POPULATE`], [Pre-fault all pages at `mmap` time; eliminates later fault latency at the cost of upfront I/O],
  [`MAP_FIXED`], [Force mapping at exact address; dangerous if address is already mapped],
  [`MAP_HUGETLB`], [Use huge pages; must be pre-reserved in the HugeTLB pool],
)

*Demand paging vs MAP_POPULATE:* by default `mmap` is lazy — it creates the VMA but installs no PTEs. Each first access faults in one page. `MAP_POPULATE` asks the kernel to fault in all pages immediately (equivalent to `mlock` minus the memory reservation guarantee). The trade-off: a server that memory-maps a 10 GB file for random access benefits from demand paging (only hot pages touch RAM); a sequential log-reader benefits from `MAP_POPULATE | MAP_SEQUENTIAL` to front-load I/O.

*Memory-mapped files vs read/write:* `read`/`write` copy data through a kernel buffer and a user buffer — two copies. `mmap` + direct access avoids the user-buffer copy; the page cache is the buffer. The win is largest for random access to large files and for producer-consumer patterns where the file itself is the shared state. The loss is that `mmap` page faults can block on I/O at an arbitrary instruction, making worst-case latency harder to reason about; `O_DIRECT` `read/write` trades bandwidth for predictable latency.

=== Copy-on-Write Mechanics

COW is used in three places: `fork`, `MAP_PRIVATE` file mappings, and kernel object sharing (e.g., `vmsplice`). After `fork`, parent and child share all physical pages with read-only PTEs. On the first write to a shared page, the MMU raises a *protection fault*. The kernel fault handler (`do_wp_page` in Linux) checks the page's reference count: if it is 1 (no other mapper), the handler simply marks the PTE writable and returns; if > 1, it allocates a new page, copies the contents, installs the new writable PTE in the faulting task, and decrements the old page's reference count. The copy is invisible to the other task.

COW makes `fork` cheap for read-mostly workloads. The risk is *copy explosion*: a child that writes every page triggers as many copy operations as there are dirty pages, with each copy consuming a burst of CPU and memory bus bandwidth. Large Redis instances that fork for `BGSAVE` can double their RSS within seconds if the foreground is write-intensive.

=== userfaultfd

`userfaultfd(2)` (Linux 4.3+) lets a *user-space handler* serve page faults instead of the kernel. The process registers a range with `UFFDIO_REGISTER`; when any thread faults inside that range, the fault is forwarded to the handler process via a read on the userfaultfd descriptor. The handler can then install a page with `UFFDIO_COPY` or `UFFDIO_ZEROPAGE`. Use cases:

- *Live migration* (QEMU): the guest's memory is lazily copied to the destination; faults on un-copied pages are served from the source over the network.
- *Checkpoint/restore* (CRIU): restore memory on demand rather than all at once.
- *Custom allocators*: implement address-space-aware demand paging for specialized workloads.

The faulting thread blocks until the handler responds, so handler latency directly affects the application. Linux 5.7+ adds `UFFD_FEATURE_MINOR_FAULTS` for `MAP_SHARED` regions and `UFFD_FEATURE_WP_ASYNC` for write-protect tracking without blocking.

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

A 4 KB granularity needs 1024 PTEs — 128 cache lines of page-table state — to map 4 MB; the $"TLB"$ holds ~64-2048 entries depending on level. A workload with > $"TLB"$ × 4 KB hot data thrashes translation. *Huge pages* (2 MB and 1 GB on x86-64; 64 KB / 2 MB / 32 MB / 1 GB on AArch64) raise the per-entry coverage and dramatically reduce $"TLB"$ misses.

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

=== jemalloc and tcmalloc Slab Design

Both jemalloc (used by Firefox, Facebook) and tcmalloc (Google) use a *slab / size-class* design that avoids the fragmentation of traditional boundary-tag allocators:

*jemalloc* divides allocations into ~200 size classes (e.g., 8, 16, 32, 48, …, 14 KB). Each class is served from *runs* — contiguous spans of pages sub-divided into fixed-size slots. Runs belong to *arenas* (typically one per CPU); each arena manages its own runs independently to reduce contention. A *thread cache* (tcache) holds a per-thread magazine of recently freed objects per size class; the common `malloc`/`free` path never leaves the thread cache. On periodic *decay*, idle pages in a run are returned to the OS via `madvise(MADV_FREE)` or `madvise(MADV_DONTNEED)`, trading RSS for TLB residency.

*tcmalloc* similarly uses size classes for small objects (≤ 256 KB), serving them from per-thread free-lists. Large objects go to a *pageheap* that manages spans of contiguous pages with a radix-tree index. The central freelist (one per size class, shared) is the point of cross-thread coordination; threads batch-transfer objects to/from it to amortize lock cost.

Common pitfall: a producer thread `malloc`s; a consumer thread `free`s. ptmalloc punishes this (cross-arena), mimalloc and snmalloc don't. jemalloc and tcmalloc handle it via background threads that periodically consolidate cross-thread frees into the central freelist.

== Swap, Compression, and Memory Tiers

When RAM is exhausted the kernel either evicts clean pages (file-backed — re-fetch from disk) or pages out dirty anonymous pages (to swap). Swap-to-disk is slow enough that modern systems prefer *compressed RAM* (`zswap`, `zram`) for the first tier of pressure and only spill to disk under sustained shortage.

CXL-attached memory introduces a new tier between DRAM and NVMe: 100-300 ns latency, attached over PCIe 5.0 lanes with coherent load/store semantics. The kernel models CXL memory as a high-bandwidth NUMA node with higher latency; `numactl --membind` or `mbind()` pins allocations to it. Page-promotion / demotion daemons (`damon`, NUMA-balancing extensions) now schedule pages across local DRAM, CXL DRAM, and NVMe swap — hot pages promoted to local DRAM, warm pages demoted to CXL, cold pages swapped. Tiering policy is tunable via `damon_reclaim` and the kernel's `memory.tiering` cgroup v2 interface, with hardware vendors exposing bandwidth and latency statistics via the CXL telemetry protocol.

== OOM Killer and Memory Cgroups

When the system cannot reclaim enough memory to satisfy an allocation, the kernel invokes the *Out-of-Memory killer*. The OOM killer scores each process and kills the highest-scored one.

=== OOM Score Calculation

The kernel computes `oom_score` for each process as roughly:

$ "oom\_score" = ("RSS" + "swap") / "RAM\_total" times 1000 $

adjusted by `oom_score_adj` (set in `/proc/<pid>/oom_score_adj`, range −1000 to +1000). Setting `oom_score_adj = -1000` makes a process immune; setting it to `+1000` makes it the preferred victim. Container runtimes routinely set `oom_score_adj = 1000` on sandbox processes and `-1000` on critical daemons. After choosing a victim the kernel sends `SIGKILL` and accounts the freed pages. If the kill does not free enough memory (e.g., the victim shared most pages), the cycle repeats.

Tunable: `vm.panic_on_oom=1` converts OOM into a kernel panic — appropriate for latency-critical systems where a partial kill is worse than a reboot.

=== Memory Cgroups

Control group v2 (`cgroup2`) provides fine-grained memory accounting and enforcement per group:

#table(columns: (auto, 1fr),
  [*Knob*], [*Effect*],
  [`memory.max`], [Hard limit; processes exceeding it are OOM-killed within the cgroup],
  [`memory.high`], [Soft limit; kernel throttles allocations and reclaims aggressively above this threshold],
  [`memory.swap.max`], [Limit swap usage; `0` disables swap for the cgroup],
  [`memory.low`], [Protection against global reclaim; pages below this are evicted last],
  [`memory.current`], [Current RSS + file cache (read-only)],
  [`memory.events`], [Counts OOM kills, `memory.high` throttles, etc. (read-only)],
)

The interaction between `memory.high` and `memory.max` implements a two-tier response: throttle first (slowing allocations to encourage the application to release memory), hard-kill only when throttling fails. Kubernetes uses `memory.max` = container limit and `memory.high` = slightly below limit to trigger GC or memory pressure callbacks before an OOM kill. The `memory.oom.group` flag (Linux 5.17+) kills all processes in the cgroup atomically when any member would be OOM-killed — useful for containers where a partial kill leaves the app in an inconsistent state.

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

Gorman, M. (2004). "Understanding the Linux Virtual Memory Manager." Prentice Hall.

Harizopoulos, S. et al. (2013). "OLTP Through the Looking Glass." SIGMOD.

Corbet, J. (2015). "userfaultfd." LWN.net.

Bovet, D., Cesati, M. "Understanding the Linux Kernel," Chapters 8-9 (cross-reference for Linux mechanics).

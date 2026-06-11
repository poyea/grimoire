= Memory Reclaim

Linux ships memory aggressively: free RAM is "wasted RAM", so the kernel fills it with page cache and anonymous allocations until pressure forces eviction. The machinery that decides *what* to evict, *when*, and *how aggressively* is the memory reclaim subsystem in `mm/vmscan.c`, `mm/page_alloc.c`, and (for the modern path) `mm/vmscan.c`'s MGLRU code. Misconfigured reclaim is the single most common cause of "the box has 256 GB and is somehow OOM-killing my workload".

This chapter follows the lifecycle: how pages get on LRU lists, how kswapd and direct reclaim choose victims, how PSI quantifies pressure, how the OOM killer chooses targets, what MGLRU changes, and how cgroup v2's `memory.*` knobs constrain everything per-container.

== Page Classifications

Every reclaim-eligible page lives on one of the per-node LRU lists. There are five:

#table(columns: (auto, 1fr),
  [`LRU_INACTIVE_ANON`], [Anonymous pages not recently touched; reclaim by swap.],
  [`LRU_ACTIVE_ANON`], [Hot anonymous pages.],
  [`LRU_INACTIVE_FILE`], [File-backed pages not recently touched; reclaim by writeback (if dirty) then free.],
  [`LRU_ACTIVE_FILE`], [Hot file-backed pages.],
  [`LRU_UNEVICTABLE`], [`mlock`ed, `ramfs`, or otherwise pinned; never reclaimed.],
)

Pages move between active and inactive based on access bits, using the *two-list LRU* approximation of LRU/2. A page on inactive that's referenced gets promoted to active; an active page whose reference bit clears decays to inactive.

The split between *anon* and *file* lets the kernel control swap aggressiveness independently of cache reclaim via `vm.swappiness` (0-200): 100 means anon and file are balanced; 0 means "reclaim file first, never swap unless desperate"; 200 means "prefer swap over cache eviction".

== Watermarks and Zones

Each NUMA node has zones (`ZONE_DMA`, `ZONE_DMA32`, `ZONE_NORMAL`, `ZONE_MOVABLE`), each with three watermarks: `min`, `low`, `high`. Allocations consult the zone's free count:

- *above `high`*: no reclaim needed.
- *between `low` and `high`*: wake kswapd asynchronously.
- *between `min` and `low`*: do *direct reclaim* in the allocating context.
- *below `min`*: PF_MEMALLOC reserves only; everything else stalls or fails.

The watermark sizes scale with zone size and `vm.min_free_kbytes`. Boosting `min_free_kbytes` is the standard fix for workloads that allocate in bursts and need headroom for atomic allocations.

== kswapd: Background Reclaim

`kswapd` is the per-node kernel thread (`mm/vmscan.c`, `kswapd()` → `balance_pgdat`) woken when a zone drops below `low`. It scans LRUs, writes back dirty file pages, evicts clean pages, and keeps going until the zone is back above `high` (with hysteresis to avoid thrashing).

```
kswapd hot path:
  for each zone in node:
    scan_control sc = {.gfp_mask, .order, .nr_to_reclaim, ...};
    shrink_node(pgdat, &sc);
      → get_scan_count()  # how many anon vs file to scan
      → shrink_lruvec()
        → shrink_inactive_list()  # produces freeable pages
        → shrink_active_list()    # demotes active to inactive
      → shrink_slab()              # invokes registered slab shrinkers
```

`get_scan_count` computes the anon:file ratio from `swappiness`, the relative sizes of the lists, and recent refault rates (WORKINGSET tracking). The clever bit: pages evicted from the file LRU that get refaulted later are tracked via *shadow entries* in the page-cache xarray; persistent refaults shift the balance back toward keeping more file.

== Direct Reclaim

When allocations outrun kswapd, the caller does the work itself in *direct reclaim*. This stalls the allocating thread inside `__alloc_pages_slowpath` → `try_to_free_pages`. PSI memory pressure spikes; latency-sensitive workloads suffer.

Mitigations:

- *Raise `min_free_kbytes`* so kswapd has more time to catch up.
- *`vm.watermark_scale_factor`*: gap between `low` and `high`; bigger gap = more kswapd work per wake = less direct reclaim.
- *Per-cgroup `memory.low` reservations* shield critical groups from being targets.
- *Use `MADV_DONTNEED` / `madvise(MADV_COLD)`* to demote pages voluntarily before pressure hits.

== Writeback Coupling

Reclaiming a dirty file page requires writing it out. The writeback threads (`mm/page-writeback.c`) cap dirty pages globally (`vm.dirty_ratio`, `vm.dirty_background_ratio`) and per-bdi. Under reclaim pressure, the scanner may *wait* on writeback (`wait_on_page_writeback`), the "stalled in direct reclaim writing back dirty pages" pattern that kills tail latencies on slow disks.

Modern advice: prefer time-based limits (`vm.dirty_expire_centisecs`, `vm.dirty_writeback_centisecs`) over ratio-based on big-RAM hosts where 20% of memory is many GB of pending writes.

== Shrinkers: Slab Reclaim Plugin Points

Many subsystems own caches the page reclaimer cannot directly see (dcache, icache, slab caches, GPU drivers, btrfs caches). They register *shrinkers*:

```c
struct shrinker {
    unsigned long (*count_objects)(struct shrinker *, struct shrink_control *);
    unsigned long (*scan_objects)(struct shrinker *, struct shrink_control *);
    int seeks;
    long batch;
    unsigned flags;        // NUMA_AWARE, MEMCG_AWARE
};
register_shrinker(&my_shrinker);
```

`count_objects` reports how many freeable items exist; `scan_objects` is asked to free up to `sc->nr_to_scan`. The reclaim loop calls all registered shrinkers proportionally each scan round. dentry/inode shrinkers (`fs/dcache.c`, `fs/inode.c`) are the dominant ones.

A common pathology: a shrinker that allocates memory inside `scan_objects` → deadlock under pressure. Shrinkers must use `GFP_NOFS` or pre-allocated state.

== MGLRU: Multi-Generational LRU

The two-list LRU is a 20-year-old approximation. *Multi-Gen LRU* (`mm/vmscan.c`, configured via `CONFIG_LRU_GEN`, available as of Linux 6.1 (2022)) replaces it with per-cgroup, per-NUMA-node *generations* and *tiers*.

- *Generations* (default 4) approximate access recency. New pages enter the youngest; aging promotes pages whose page-table A bits indicate recent use; the oldest generation is the eviction candidate set.
- *Tiers* (per-generation) approximate access *frequency* based on refault distance.

The aging walk reads page-table A bits directly (`lru_gen_walk_mm`), giving far more accurate recency than the rotate-when-scanned approximation of classic LRU. The result is dramatically better behaviour under memory pressure on workloads with mixed cold/warm/hot access patterns; Google reported 7-12% throughput gains on web-serving workloads and large reductions in tail latency (measured 2023).

Knobs:

```bash
# Enable / disable
echo y > /sys/kernel/mm/lru_gen/enabled

# Show generation populations
cat /sys/kernel/debug/lru_gen
```

MGLRU integrates with memcg: each cgroup has its own generation lists, so reclaim under cgroup pressure doesn't perturb the global LRU.

== PSI: Pressure Stall Information

PSI (`kernel/sched/psi.c`, mainlined 4.20) measures, for each resource (CPU, memory, IO), how long tasks were *stalled* waiting for it: what fraction of wall-clock time would have made forward progress with more resource.

`/proc/pressure/memory`:

```
some avg10=2.45 avg60=1.10 avg300=0.30 total=12345678
full avg10=0.50 avg60=0.20 avg300=0.05 total=2345678
```

- *some*: at least one task was stalled.
- *full*: all non-idle tasks were stalled (the box made no useful progress).

The averages are exponential moving averages of the stall fraction. `full > 0` for memory means the whole system spent some fraction of the last interval thrashing.

Per-cgroup PSI lives at `/sys/fs/cgroup/<group>/memory.pressure`. Watchdogs use `poll(2)` on these files with thresholds (`some 150000 1000000` = "wake me when 'some' exceeds 150 ms per 1 s window") to detect overload before OOM. Facebook's *oomd* is the canonical user.

== The OOM Killer

When reclaim cannot free enough memory to satisfy an allocation and no further progress is possible, the OOM killer (`mm/oom_kill.c`) selects a victim. The selection function `oom_badness()` scores each candidate roughly as:

```
score = (RSS + swap + pgtables) / total_memory * 1000
score += oom_score_adj
```

`oom_score_adj` (`/proc/<pid>/oom_score_adj`, -1000..+1000) is the operator's lever. `-1000` is "immune"; sshd typically gets `-1000`, and runtime-critical services get negative biases.

The victim's tasks are SIGKILLed; if it has children sharing memory, they may go too. On systems with memcg, the OOM is scoped to the cgroup that hit its limit (`memory.max`); only members of that group are candidates.

Tuning posture:

- *Set `vm.overcommit_memory=2`* (strict accounting) for workloads where overcommit-then-kill is unacceptable. Allocations fail with `ENOMEM` instead.
- *Use `memory.high`* (cgroup v2) to throttle a group via reclaim before it hits `memory.max` and triggers OOM.
- *Run `oomd` / `systemd-oomd`*: userspace agents that monitor PSI and pre-emptively kill chosen targets *before* the kernel's blunt selector fires.

== memcg: Cgroup v2 Memory Controller

`memory.*` files under each cgroup v2 directory control and observe memory use:

#table(columns: (auto, 1fr),
  [`memory.current`], [Bytes currently charged.],
  [`memory.low`], [Soft floor: reclaim avoids this group while siblings have more than their `low`.],
  [`memory.min`], [Hard floor: never reclaim below this (counts against parent's available memory).],
  [`memory.high`], [Throttle threshold: exceeding it forces direct reclaim in the offending task.],
  [`memory.max`], [Hard limit: exceeding triggers cgroup OOM.],
  [`memory.swap.max`], [Swap usage cap.],
  [`memory.pressure`], [PSI for this cgroup.],
  [`memory.events`], [Counters: `low`, `high`, `max`, `oom`, `oom_kill`.],
  [`memory.stat`], [Detailed breakdown (anon, file, kernel, slab, etc.).],
)

The recommended pattern: set `memory.high` (soft) and `memory.max` (hard) separately. `high` produces graceful throttling visible via `memory.events:high` counter; `max` is the safety net.

Memcg tracks slab allocations (`kmem` accounting, now unified in cgroup v2), kernel stacks, page-table pages, and everything else attributable to a task. The kernel's accounting overhead is non-trivial; for ultra-high-rate workloads `cgroup.memory=nokmem` opts out (but lets containers escape kernel-memory accounting).

== Zswap and zRAM

Swap doesn't have to mean disk. *zswap* is a compressed write-back cache between the page reclaim path and the actual swap device: pages destined for swap are compressed (LZO/LZ4/zstd) and stored in a memory pool first; only when the pool fills are they written to backing storage.

```bash
echo 1   > /sys/module/zswap/parameters/enabled
echo lz4 > /sys/module/zswap/parameters/compressor
echo 20  > /sys/module/zswap/parameters/max_pool_percent
```

*zRAM* is a synthetic compressed block device (`drivers/block/zram/`) used *as* a swap device; common on Android and resource-constrained Linux. 2-4× effective memory expansion is typical.

== Transparent Huge Pages and Reclaim Interaction

THP (`mm/khugepaged.c`) tries to collapse 4K pages into 2M ones in the background. Under reclaim pressure, splitting THPs back to 4K is expensive (TLB invalidations across all CPUs that mapped them). Common pitfall: `transparent_hugepage=always` on database hosts causes latency spikes during reclaim. Many database deployments set it to `madvise` or `never`. See _MMap and Memory Mapped Files_.

== NUMA and Memory Reclaim

Each NUMA node has its own LRU lists and kswapd. `vm.zone_reclaim_mode` controls whether a node prefers reclaiming its own pages or allocating across to a remote node. The historical default (1) caused surprising latency on database workloads; modern kernels default to 0 (allocate remotely under pressure).

`numactl --membind` or per-task `set_mempolicy` constrains allocations to specific nodes; reclaim then targets only those nodes' LRUs.

== Damon: Data Access Monitor

DAMON (`mm/damon/`, since 5.15) samples page-access patterns at low cost (regions sampled, not individual pages) and exposes them to userspace and to in-kernel schemes. `DAMOS` (DAMON-based Operation Schemes) lets you say declaratively: "any 4 MiB region untouched for 60 s, `MADV_COLD` it", enabling pro-active cold-page demotion to swap or to tier-2 memory (CXL).

== Observability Cheatsheet

```bash
# Pressure right now (system and per-cgroup)
cat /proc/pressure/memory
cat /sys/fs/cgroup/myservice/memory.pressure

# Where is reclaim happening?
bpftrace -e 'kprobe:shrink_node { @[kstack] = count(); }'

# How often are pages refaulting (workingset miss)?
cat /proc/vmstat | grep -E 'workingset|pgsteal|pgscan'

# OOM events
journalctl -k | grep -i 'killed process'
cat /sys/fs/cgroup/myservice/memory.events

# Per-cgroup memory breakdown
cat /sys/fs/cgroup/myservice/memory.stat
```

`pgsteal_*` counters are reclaim victories; `pgscan_*` are work done; their ratio (`scan/steal`) is a thrashing indicator: values much above 1 mean the scanner is iterating without freeing.

== Patterns and Anti-Patterns

- *Pattern*: Set `memory.high` slightly below `memory.max` so PSI rises early; have a control plane react.
- *Pattern*: Pin critical files into the cache with `vmtouch` or `mmap` + `MAP_POPULATE` + `mlock`.
- *Anti-pattern*: `echo 3 > /proc/sys/vm/drop_caches` in production — invalidates the entire cache and steals throughput from every workload sharing the host.
- *Anti-pattern*: `swappiness=0` thinking it disables swap. It does not; under sufficient pressure the kernel still swaps. Use `memory.swap.max=0` per cgroup to actually forbid.
- *Anti-pattern*: Raising `vm.overcommit_ratio` to allocate "more than physical". The kernel cheerfully accepts; the OOM killer cheerfully delivers.

== Further Reading

Kernel docs: `Documentation/admin-guide/mm/`, especially `concepts.rst`, `multigen_lru.rst`, `damon/`, and `Documentation/accounting/psi.rst`.

Gorman, M. (2004). _Understanding the Linux Virtual Memory Manager_. Prentice Hall.

Corbet, J. (2018-2024). LWN articles on PSI, MGLRU, DAMON, memcg v2, oomd.

Yu, Y. (2022). _Multi-Gen LRU Framework_ (LSF/MM talks; LWN summary).

Brown, N. (2024). _zswap and friends_ — LWN.

`mm/vmscan.c`, `mm/page_alloc.c`, `mm/oom_kill.c`, `mm/memcontrol.c`, `mm/damon/core.c`, `kernel/sched/psi.c`.

*See also:* _mmap and Memory Management_ (where the page cache and anonymous pages originate), _Cgroups and Namespaces_ (memcg controllers), _Scheduler_ (PSI integrates with CFS load tracking), _VFS and Filesystems_ (dentry/inode shrinkers, writeback path).

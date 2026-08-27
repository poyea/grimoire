#import "../template.typ": xref

= Page Reclaim and OOM <page-reclaim-and-oom>

Every page of physical memory is eventually contested: the page cache wants it for file data, applications want it for heap, the kernel wants it for slabs. *Reclaim* is the machinery that resolves the contest by evicting the least valuable pages, and the *OOM killer* is the admission of defeat when reclaim cannot keep up. This chapter covers when reclaim runs, how victims are chosen, where evicted data goes, how the system measures its own distress, and what happens when nothing else works. The conceptual replacement-algorithm background is in #xref("operating-systems", "memory-management", label: "Memory Management"); Linux source-level mechanics are in `linux-kernel/memory-reclaim.typ`.

*See also:* #xref("operating-systems", "memory-management", label: "Memory Management"), _Virtual Memory_, #xref("operating-systems", "storage-stack", label: "Storage Stack"), #xref("linux-kernel", "memory-reclaim", label: "Memory Reclaim") (linux-kernel), #xref("linux-kernel", "cgroups-namespaces", label: "Cgroups and Namespaces") (linux-kernel).

== Triggers and Watermarks

Reclaim runs in two modes distinguished by *who pays*:

- *Background (asynchronous)*: a kernel thread (`kswapd` per NUMA node in Linux) wakes when free memory drops below a *low watermark* and reclaims until a *high watermark* is restored. Nobody's allocation blocks; the cost is hidden in background CPU and I/O.
- *Direct (synchronous)*: an allocating task that finds free memory below the *min watermark* reclaims on its own, inside the allocation path. Latency lands on the allocator — direct reclaim is the canonical source of multi-millisecond allocation stalls.

The watermark gap is the system's shock absorber: a wide gap lets background reclaim absorb bursts; a narrow gap pushes bursty allocators into direct reclaim. Linux exposes the gap via `vm.watermark_scale_factor`. A small reserve below min is held for `GFP_ATOMIC` allocations (interrupt context cannot sleep to reclaim).

Reclaim also triggers without global shortage: cgroup limits (`memory.high`, `memory.max`), proactive reclaim (`memory.reclaim`, DAMON-driven policies), and contiguity demands (compaction needs movable pages out of the way).

== Choosing Victims: LRU Approximations

True LRU needs an ordered list updated on every reference, which no MMU provides. What hardware does provide is the per-PTE *accessed bit*, set on reference and clearable by software. All practical algorithms sample it.

=== Clock

The classic approximation: pages on a circular list, a hand sweeping. If the accessed bit is set, clear it and advance (second chance); if clear, the page has not been touched since the last sweep — evict. Clock degrades under sequential scans: a one-pass file read sets bits on pages that will never be touched again, granting them an undeserved second chance and evicting the real working set.

=== Two-List (Active/Inactive)

Linux's long-standing design splits pages into an *active* list (the protected working set) and an *inactive* list (eviction candidates), each approximately FIFO. New pages enter inactive; a second reference while on the inactive list promotes to active; reclaim takes from the inactive tail; the active list is trimmed to refill inactive when the ratio skews. The two-reference promotion rule is the scan resistance: a streamed file's pages get exactly one reference and die on the inactive list without ever displacing the active set. File and anonymous pages keep separate list pairs, with `vm.swappiness` biasing the scan ratio between them.

=== MGLRU

The multi-generational LRU (Linux 6.1+) replaces the two-list binary with $n$ *generations* (typically 4) per type. Pages are aged by walking page tables — checking accessed bits where translations actually live — rather than by physically moving pages between lists, which is far cheaper on terabyte-RAM machines and captures recency more faithfully (the two-list design only learns about a reference when reclaim happens to scan the page). Eviction takes from the oldest generation; a feedback loop (PID controller) balances file vs anonymous eviction by comparing their *refault rates*. Bloom filters remember which page tables were recently young to skip cold ones during walks.

== Working Sets and Refault Distance

Denning's *working set* — the pages referenced in the last $tau$ time units — is the theoretical target: keep each process's working set resident and it makes progress; evict into it and the process thrashes. Direct measurement is impractical, so modern kernels measure the *consequences* of misestimation instead.

When a page is evicted, the kernel leaves a *shadow entry* recording the eviction timestamp (in units of reclaim activity). If the page is faulted back in — a *refault* — the gap between eviction and refault is the *refault distance*: how much more memory would have been needed to keep the page resident. A refault distance smaller than the active list means the active list is hoarding pages colder than the ones being evicted, and the kernel responds by deactivating active pages (Linux's workingset detection). The same shadow-entry data feeds MGLRU's eviction balancing and PSI's notion of wasted work.

*Thrashing* is the failure mode: aggregate working sets exceed RAM, every eviction is a future refault, and the system spends its cycles on I/O that reconstructs what it just destroyed. Throughput collapses superlinearly. The classical remedy is load shedding — run fewer things — which is precisely what cgroup limits and OOM killing implement in modern form.

== Dirty Pages and Writeback

A clean file page is free to reclaim: drop it, re-read later. A *dirty* page must be written back first, which couples reclaim to the storage stack. Kernels therefore write back dirty data continuously and ahead of need: flusher threads wake periodically (`dirty_expire_centisecs`) and when the dirty fraction crosses a background threshold (`dirty_background_ratio`); above a hard threshold (`dirty_ratio`), writers themselves are throttled, turning memory pressure into write-side backpressure rather than reclaim-side stalls.

The balance matters: too much allowed dirty data means a burst of writes followed by reclaim finding nothing but dirty pages (and `sync`/unmount taking tens of seconds); too little wastes the page cache's write-coalescing. Writeback from direct reclaim is avoided in modern Linux — stack depth and lock context make it dangerous — so reclaim instead waits on or skips dirty pages, another reason dirty buildup translates into allocation latency.

== Swap, zswap, and zram

Anonymous pages have no backing file; evicting them requires *swap*. Swap's reputation comes from rotating disks, where a refault cost ~10 ms; on NVMe it is tens of microseconds, and the calculus changes. Swap is not a RAM extension but a way to evict *cold anonymous* pages so the memory can hold something warmer — a system with no swap is forced to keep every leaked, idle anonymous page resident forever while evicting possibly-hot file cache.

Compressed memory inserts a tier above the device:

- *zswap*: a compressed in-RAM cache in front of a real swap device. Pages swap out into the pool (~3-5 µs, typical 2-3:1 compression); when the pool fills, the LRU-oldest entries are decompressed and written to the backing device. Cold data eventually leaves RAM entirely.
- *zram*: a compressed RAM block device used *as* the swap device. No backing store: the compressed copy stays in RAM forever, so incompressible or truly cold data still occupies memory. Standard on Android, ChromeOS, and Fedora desktops.

The trade: a compressed-memory refault costs microseconds of CPU instead of an I/O, shrinking the thrashing penalty by orders of magnitude, at the price of dedicating CPU cycles and a fraction of RAM to the pool.

== Pressure Stall Information

Free-memory counters are a poor distress signal: a healthy system keeps free memory near the watermarks by design (idle RAM is wasted RAM). *PSI* (Pressure Stall Information, Linux 4.20+) measures the thing that actually hurts: the fraction of wall-clock time tasks spend *stalled* on memory — in direct reclaim, waiting for refault I/O, throttled by `memory.high`. `/proc/pressure/memory` reports `some` (at least one task stalled) and `full` (all runnable tasks stalled simultaneously) as 10/60/300-second averages, per system and per cgroup.

PSI converts "is there memory pressure?" from guesswork into a number with units (lost productivity), and it is the sensor for the modern userspace response stack: `oomd` and `systemd-oomd` trigger on sustained PSI, and PSI poll thresholds let latency-sensitive services react within milliseconds.

== Out of Memory

When reclaim cannot satisfy an allocation — every candidate page is hot, dirty-and-unwritable, pinned, or already gone — the kernel must destroy something. Killing is the correct last resort: the alternative under overcommit (failing allocations) punishes whoever happens to allocate next, often an innocent and unprepared victim, and deadlock looms if the allocator is itself needed for reclaim.

=== The Kernel OOM Killer

The Linux OOM killer selects the process whose death frees the most memory: `oom_score` is essentially the task's RSS + swap + page-table footprint as a fraction of available memory, shifted by the per-process `oom_score_adj` (−1000 disables, +1000 volunteers). Notably absent from the modern heuristic: niceness, runtime, or "badness" guesses that older kernels attempted — the metric is purely *recovery per kill*. The victim receives `SIGKILL` and a time-limited *OOM reaper* thread tears down its address space even if the task is stuck uninterruptible, closing the historical deadlock where the victim could not die because it was blocked on the very memory being reclaimed.

The kernel killer's weakness is timing, not victim choice: it fires only when allocation is truly impossible, which on a swapping or refault-storming machine can be minutes of near-total unresponsiveness ("livelock before deadlock"). The machine is thrashing — technically making progress — so the killer never engages.

=== Userspace OOM Killing

Userspace daemons trade the kernel's perfect knowledge for earlier engagement and richer policy:

- *oomd / systemd-oomd* (Meta, now in systemd): watches PSI and cgroup memory statistics; kills entire cgroups when memory pressure exceeds a threshold for a sustained window (e.g., `full` > 20% for 20 s) or when a cgroup's swap usage crosses a limit. Policy is expressed over services, not processes.
- *earlyoom*: simpler — polls available memory + swap and SIGTERMs (then SIGKILLs) the largest process when both fall below thresholds. Popular on desktops where the goal is "never let the UI freeze."

The shared philosophy: a fast kill of the right service beats minutes of thrash followed by the kernel killing something arbitrary.

=== Cgroup Memory Control

The memory controller scopes the whole stack — accounting, reclaim, and OOM — to a service:

#table(columns: (auto, 1fr),
  [*Knob*], [*Effect*],
  [`memory.min` / `memory.low`], [Protection: pages below are exempt from (min) or deprioritized in (low) external reclaim],
  [`memory.high`], [Throttle threshold: allocations above it force reclaim on the allocating task, slowing it without killing],
  [`memory.max`], [Hard limit: reclaim, then cgroup-local OOM kill],
  [`memory.oom.group`], [Kill the cgroup as a unit, not one process from it],
  [`memory.reclaim`], [Proactive reclaim: userspace asks the kernel to evict N bytes now, during idle periods],
  [`memory.pressure`], [Per-cgroup PSI],
)

Together these implement working-set management as policy: protect the database's cache with `memory.low`, cap the batch job with `memory.max` plus `oom.group`, let a balloon-style daemon drive `memory.reclaim` against `memory.pressure` feedback to discover each service's true working set (the point where further reclaim starts producing refaults).

== Pitfalls

- *Disabling swap to "avoid swapping"*: removes the kernel's only lever against cold anonymous memory; pressure then falls entirely on the file cache, often evicting hot text and causing the very thrash the change meant to prevent.
- *Reading "low free memory" as a problem*: check PSI and refault rates instead; free memory near the watermark is the steady state.
- *`oom_score_adj = -1000` on large daemons*: an unkillable process that is also the biggest consumer forces the killer to slaughter everything else first.
- *`memory.max` without `oom.group`*: the kernel kills one worker of a multi-process service, leaving a zombie deployment that is alive but broken.
- *Ignoring direct reclaim in latency budgets*: p99 spikes that correlate with allocation bursts and `pgsteal_direct` are reclaim, not the application; widening watermarks or adding proactive reclaim fixes what profiling the application will not find.

== Further Reading

Denning, P. (1968). "The Working Set Model for Program Behavior." CACM.

Corbato, F. (1968). "A Paging Experiment with the Multics System." MIT Project MAC (the clock algorithm).

Johnson, T., Shasha, D. (1994). "2Q: A Low Overhead High Performance Buffer Management Replacement Algorithm." VLDB.

Weiner, J. (2018). "PSI: Pressure Stall Information." Linux kernel documentation and LPC talk.

Weiner, J. (2014). "workingset: per-zone refault distance accounting." Linux kernel commit series.

Zhao, Y. et al. (2022). "Multi-Generational LRU." LWN.net series and kernel documentation.

Corbet, J. (2017). "The OOM reaper." LWN.net.

Facebook Engineering (2018). "oomd: a userspace OOM killer." Meta engineering blog.

Gorman, M. (2004). "Understanding the Linux Virtual Memory Manager." Prentice Hall, Chapters 10-13.

Tanenbaum, A., Bos, H. "Modern Operating Systems," Chapter 3 (page replacement).

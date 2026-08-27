#import "../template.typ": xref

= Virtual Memory <virtual-memory>

Virtual memory is the abstraction that lets every process believe it owns a private, contiguous, enormous address space while the kernel quietly multiplexes a finite set of physical frames behind it. The machinery is a collaboration: the MMU translates on every reference, the TLB caches translations, and the kernel fills in the gaps via page faults. This chapter covers the architectural mechanics: translation structures, TLB behavior, fault handling, copy-on-write, huge pages, and NUMA placement. Linux's `mmap` interface and VMA bookkeeping live in `linux-kernel/mmap-memory.typ`; allocator and reclaim policy are in #xref("operating-systems", "memory-management", label: "Memory Management") and #xref("operating-systems", "page-reclaim-and-oom", label: "Page Reclaim and OOM").

*See also:* #xref("operating-systems", "memory-management", label: "Memory Management"), #xref("operating-systems", "page-reclaim-and-oom", label: "Page Reclaim and OOM"), #xref("operating-systems", "processes-and-threads", label: "Processes and Threads"), #xref("linux-kernel", "mmap-memory", label: "mmap and Memory Management") (linux-kernel), #xref("linux-kernel", "memory-reclaim", label: "Memory Reclaim") (linux-kernel); #xref("cpu-architecture", "virtual-memory", label: "Virtual Memory") (the hardware side: page-table walks, TLB structure, huge pages).

== Address Translation

The hardware contract is simple: a virtual address splits into a *virtual page number* (VPN) and an *offset*; the page table maps VPN to *physical frame number* (PFN); the offset passes through unchanged. With 4 KB pages the offset is 12 bits, so a 48-bit virtual address has a 36-bit VPN — $2^36$ potential pages, far too many for a flat table (a flat table for a 48-bit space would need 512 GB of PTEs per process).

The solution is *sparseness*: most address spaces are mostly empty, so the table is itself paged.

=== Multi-Level Page Tables

x86-64 uses a radix tree of 4 levels (PML4 → PDPT → PD → PT), each level a 4 KB page of 512 eight-byte entries indexed by 9 bits of the VPN. A 48-bit VA decomposes as $9 + 9 + 9 + 9 + 12$. Five-level paging (Linux's `la57`) prepends a PML5 level for 57-bit VAs. AArch64 is structurally similar but parameterized by granule size (4 KB / 16 KB / 64 KB) and supports up to 52-bit VAs.

Properties of the radix design:

- *Space proportional to mapped regions*: an unmapped gigabyte costs one absent entry at an upper level, not a million PTEs.
- *Walk cost*: a TLB miss costs up to 4-5 dependent memory reads. Hardware *page-walk caches* (x86 calls them paging-structure caches) cache upper-level entries so most walks resolve in 1-2 reads.
- *Permission bits at every level*: a non-writable PD entry makes the whole 2 MB region read-only regardless of leaf PTEs; the effective permission is the intersection.

Each leaf PTE carries the PFN plus control bits: present, writable, user/supervisor, *accessed* (set by hardware on any reference), *dirty* (set on write), no-execute, and cache-type bits. The accessed and dirty bits are the raw material for reclaim algorithms (see #xref("operating-systems", "page-reclaim-and-oom", label: "Page Reclaim and OOM")).

=== Inverted and Hashed Page Tables

The radix tree scales with virtual space; an *inverted page table* scales with physical memory instead: one entry per physical frame, recording which (process, VPN) maps it. Lookup requires a search, so practical designs (PowerPC, classic PA-RISC, Itanium's long-format VHPT) use a *hashed page table*: hash the (ASID, VPN) pair into a bucket of candidate entries and search the chain. The win is bounded table size on machines with huge sparse address spaces; the losses are hash collisions (variable-latency walks) and awkward support for sharing, since one frame maps to one entry — shared mappings need auxiliary structures. Modern designs have largely converged on radix trees plus large TLBs, with hashing surviving in POWER's HPT mode and in software-filled TLB architectures (MIPS, early SPARC) where the kernel can use any structure it likes and the hardware only sees the TLB.

== The TLB

The *Translation Lookaside Buffer* caches recent VPN→PFN translations. Typical sizing: 64-100 L1 dTLB entries, 1024-3072 unified L2 TLB entries. A TLB hit adds nothing to the access; a miss costs a page walk (tens of cycles with warm paging-structure caches, hundreds when the walk itself misses cache).

=== TLB Reach

*Reach* is entries × page size: a 1536-entry TLB of 4 KB pages covers 6 MB. Any workload whose hot data exceeds reach pays a translation miss tax on a fraction of all references; pointer-chasing workloads over tens of gigabytes can spend 20-40% of cycles in page walks. Reach is the core argument for huge pages: the same 1536 entries at 2 MB cover 3 GB.

=== ASIDs and Context Switching

A naive TLB must be flushed on every address-space switch, since VPN 0x1000 means different frames in different processes. *Address Space Identifiers* (ASIDs on AArch64 and RISC-V, PCIDs on x86) tag each TLB entry with the owning address space, so entries from multiple processes coexist and a context switch only changes the active tag. The kernel manages a small ASID namespace (8-16 bits) with generation-based recycling: when ASIDs run out, bump a generation counter and flush. PCID became practically important after Meltdown: kernel page-table isolation (KPTI) switches page tables on every syscall, and without PCID each switch would flush the entire TLB.

=== TLB Shootdowns

The TLB is *not coherent*: hardware that updates a PTE does not invalidate stale copies in other cores' TLBs. When the kernel unmaps or downgrades a mapping that other CPUs may have cached, it must perform a *shootdown*: invalidate locally, then interrupt every CPU that might hold the translation (tracked via the mm's CPU mask) and have each run an invalidation. On x86 this is an IPI-based protocol costing microseconds and scaling with core count; AArch64's broadcast `tlbi` instructions and x86's newer `INVLPGB` push the invalidation into hardware. Shootdown cost is why `munmap`-heavy workloads (allocators returning memory aggressively, `MADV_DONTNEED` churn) can bottleneck on IPIs, and why batching deferred invalidations is a standard kernel optimization.

== Page Faults and Demand Paging

A *page fault* is the MMU's upcall: no valid translation, or a permission mismatch. Faults are not errors; they are the kernel's primary lazy-evaluation mechanism.

- A *minor fault* is resolved without I/O: the data is already in memory (page cache, zero page, COW source) and the handler only installs a PTE. Cost: microseconds.
- A *major fault* requires reading from backing store (file or swap). Cost: tens of microseconds on NVMe to milliseconds on disk — four to six orders of magnitude above a TLB hit.

*Demand paging* exploits this: `mmap`, `fork`, and `exec` create mappings without populating them, and pages materialize on first touch. A process that maps a 2 GB binary but executes 5% of it never pays for the rest. The costs are fault latency at arbitrary instructions and the loss of batching; *fault-around* (installing a cluster of neighboring PTEs per fault) and *read-ahead* recover most of the batching for sequential access.

The fault handler's decision tree: find the region covering the address (no region → segmentation fault); check permissions against the fault type (write to read-only shared page → protection fault, possibly COW); then resolve by zero-fill, page-cache lookup, swap-in, or file read.

== Copy-on-Write

COW shares physical frames between logically distinct copies until one side writes. The mechanics: mark the shared PTEs read-only, keep a reference count on the frame, and resolve the write-protection fault by copying. If the refcount is 1 at fault time the copy is skipped and the PTE is simply made writable.

The canonical client is `fork`: parent and child share everything, so forking a 10 GB process costs only a page-table copy. Others: `MAP_PRIVATE` file mappings (writes produce anonymous copies, the file is untouched), the shared *zero page* backing untouched anonymous memory, and kernel same-page merging (KSM) which deduplicates identical frames across VMs and re-COWs them on write.

Two systemic costs deserve respect. First, *latent copy storms*: a forked snapshot (Redis `BGSAVE`) converts the parent's write rate into a copy rate, potentially doubling RSS in seconds. Second, COW interacts subtly with concurrency — the historical Dirty COW bug raced the COW fault path against `madvise(MADV_DONTNEED)`, and later GUP-vs-COW bugs raced the refcount check against page pinning.

== Huge Pages

Radix page tables natively support *block mappings*: a leaf at the PD level maps 2 MB on x86-64, at the PDPT level 1 GB. Benefits compound: one TLB entry covers 512× the memory, walks terminate a level early, and 512 PTE updates collapse into one.

Delivery models:

#table(columns: (auto, 1fr),
  [*Model*], [*Character*],
  [Explicit (HugeTLB)], [Pre-reserved pool; opt-in via `MAP_HUGETLB` or `hugetlbfs`; predictable, never broken up],
  [Transparent (THP)], [Kernel promotes/collapses 4 KB runs opportunistically; no application change; compaction stalls and memory bloat as failure modes],
)

The bloat problem: a 2 MB page allocated for a region where the application touches 4 KB wastes 511 frames; sparse-heap workloads can inflate RSS 2-3×. The fragmentation problem: after months of uptime, free physical memory is shredded into non-contiguous 4 KB pieces and 2 MB allocations require *compaction* (migrating pages to create contiguity), whose latency leaks into allocation paths. Databases mostly choose explicit huge pages for the predictability.

== Memory-Mapped I/O

Not all physical addresses are RAM. Device registers and frame buffers occupy physical address ranges, and the same translation machinery maps them into kernel or user virtual space — this is *memory-mapped I/O* (MMIO), distinct from x86's legacy port I/O. Two attributes matter:

1. *Cacheability*: device registers must be mapped uncached (or *device memory* on ARM, with mandated access size and no speculation); a cached read of a status register would return stale data. Frame buffers use *write-combining* to batch stores without read caching.
2. *Ordering*: writes to device memory must not be reordered past the point where they trigger side effects; architectures provide memory types and barriers (`mmiowb`, ARM's device-nGnRE semantics) to enforce this.

PTE cache-type bits (x86 PAT, ARM MAIR indices) carry this per mapping, which is why `mmap` of `/dev/mem` or a PCI BAR through a driver produces different PTE attributes than a file mapping.

== NUMA Placement

On multi-socket (and increasingly multi-chiplet) machines, physical frames live on specific nodes, and remote access costs 1.5-3× local latency plus contended interconnect bandwidth. Virtual memory is the natural control point, since placement is decided when a fault picks a frame:

- *First-touch*: allocate the frame on the node of the faulting CPU. Correct by default for thread-local data, wrong when an initializer thread touches buffers later consumed elsewhere.
- *Interleave*: round-robin frames across nodes; trades best-case latency for predictable bandwidth, good for large shared structures.
- *Explicit binding*: `mbind`/`numactl` pin a range to a node set.
- *Dynamic migration*: the kernel samples access patterns (Linux's AutoNUMA periodically unmaps pages and observes which node faults them back) and migrates pages toward their consumers, or tasks toward their memory.

Tiered memory (CXL expanders, persistent memory) generalizes the picture: nodes differ not just in distance but in kind, and the placement problem becomes promotion/demotion across tiers — covered from the reclaim side in #xref("operating-systems", "page-reclaim-and-oom", label: "Page Reclaim and OOM").

== Pitfalls

- *Treating all faults alike*: minor and major faults differ by orders of magnitude; `perf stat -e minor-faults,major-faults` distinguishes them, RSS growth alone does not.
- *Ignoring shootdown cost*: per-request `mmap`/`munmap` or aggressive `MADV_DONTNEED` in a multi-threaded process serializes on IPIs; allocator tuning (decay times) is often the fix.
- *THP as a free win*: it frequently is, but verify RSS and tail latency; `madvise`-only mode confines THP to opted-in regions.
- *First-touch from the wrong thread*: parallel initialization, or initialization by the consumer, is the standard remedy.
- *Assuming TLB coherence*: any code that modifies PTEs (including userfaultfd-style tricks) must reason about which CPUs hold stale entries.

== Further Reading

Denning, P. (1970). "Virtual Memory." ACM Computing Surveys.

Kandiraju, G., Sivasubramaniam, A. (2002). "Going the Distance for TLB Prefetching: An Application-Driven Study." ISCA (TLB behavior studies).

Basu, A. et al. (2013). "Efficient Virtual Memory for Big Memory Servers." ISCA (direct segments, TLB reach).

Navarro, J. et al. (2002). "Practical, Transparent Operating System Support for Superpages." OSDI.

Kwon, Y. et al. (2016). "Coordinated and Efficient Huge Page Management with Ingens." OSDI.

Amit, N. (2017). "Optimizing the TLB Shootdown Algorithm with Page Access Tracking." USENIX ATC.

Jacob, B., Mudge, T. (1998). "Virtual Memory in Contemporary Microprocessors." IEEE Micro (inverted/hashed table survey).

Gorman, M. (2004). "Understanding the Linux Virtual Memory Manager." Prentice Hall.

Lameter, C. (2013). "NUMA (Non-Uniform Memory Access): An Overview." ACM Queue.

Tanenbaum, A., Bos, H. "Modern Operating Systems," Chapter 3.

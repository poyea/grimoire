= The Storage Stack

Between `write(2)` and a flash cell sit half a dozen layers, each with its own queueing, ordering, and failure semantics. Understanding the stack as a whole — VFS, page cache, block layer, I/O scheduler, driver, controller, media — is what separates a system that scales from one that mysteriously stalls under load. This chapter is the conceptual treatment; Linux-specific `blk-mq` and io_uring are in `linux-kernel/block-layer.typ` and `linux-kernel/io-uring.typ`.

*See also:* _The Block Layer_ (linux-kernel), _io_uring_ (linux-kernel), _Buffer Pool and I/O_ (database).

== Layers

```
  application       read/write/io_uring_submit
       |
  VFS / page cache  cache hit returns here
       |
  file system       translate file offset to block address
       |
  block layer       merge, reorder, queue per device
       |
  driver            translate to device-native command (NVMe NVMHCI, SCSI)
       |
  controller        firmware FTL, ECC, GC, wear leveling
       |
  media             NAND program/erase, HDD seek
```

Each downward step is a translation; each upward step a completion notification (interrupt, polled completion queue, or callback). Latencies vary by orders of magnitude across layers:

#table(columns: (auto, auto, auto),
  [*Operation*], [*Typical Latency*], [*Notes*],
  [Page-cache hit], [~50 ns], [memcpy from cache],
  [NVMe Gen4 read], [~10-100 $mu$s], [device-limited],
  [SATA SSD read], [~100 $mu$s], [protocol overhead],
  [HDD random read], [~5-10 ms], [seek + rotation],
  [Context switch on syscall], [~1-3 $mu$s], [Meltdown/KPTI raised this],
  [Spectre mitigations], [+~100 ns], [per indirect call],
)

== Block Layer

The block layer presents a uniform "linear array of 512 B (or 4 KB) sectors" abstraction over heterogeneous devices. Its job:

- *Request building*: package an upper-layer I/O into a `bio` (Linux) or analog (`buf` on FreeBSD).
- *Merging*: adjacent or overlapping I/Os to the same device coalesce into a single command — vital for HDDs, useful on SSDs to reduce CPU per I/O.
- *Queueing*: hold pending I/Os until the device can accept more.
- *Completion delivery*: interrupt-driven, polled, or hybrid (NAPI-style).

Historically (Linux pre-3.13) the queue was a single per-device structure with a global lock — a scaling disaster at million-IOPS NVMe. *Multi-queue block layer* (blk-mq) replaced it with per-CPU submission queues mapped to device-side hardware queues. NVMe supports up to 64K queues × 64K depth; blk-mq lets you actually use them.

*I/O scheduling* once meant elevators on HDDs (`cfq`, `deadline`, `noop`). On modern flash it largely doesn't — the device's own FTL has better information. The current crop:

- `none` / `noop` — pass straight through; appropriate for fast SSDs.
- `mq-deadline` — light reordering with a starvation guarantee; safe default.
- `bfq` — fairness-aware, useful for desktop interactivity.
- `kyber` — token-based, p99 latency oriented; good for mixed reader/writer workloads.

== Interrupts vs Polling

A traditional driver fires an interrupt on each completion: CPU is freed during the wait, but interrupt servicing costs 1-3 $mu$s. At MIOPS rates, interrupt overhead dominates.

*Polling* — the driver spins waiting for completion — eliminates interrupt cost but burns a CPU. NVMe's *hybrid polling* combines: poll for the expected device latency, fall back to interrupt if it takes longer. Linux's `io_uring` `IORING_SETUP_IOPOLL` and `IORING_SETUP_SQPOLL` push this further (kernel-side polling threads, no syscall per submission).

The crossover where polling wins is when the device latency is shorter than the interrupt overhead — i.e., on Optane and Gen5 NVMe.

== NVMe

NVMe (Non-Volatile Memory Express) replaced SCSI/AHCI for flash. Key innovations:

- *Submission/completion queue pairs* in shared host memory; the device DMAs directly, no command FIFO bottleneck.
- *No PCIe interrupts per I/O*: MSI-X interrupts can be coalesced or polled.
- *Up to 65,536 queues* of 65,536 depth — designed for million-core futures.
- *Streamlined command set*: ~13 commands vs SCSI's hundreds.

NVMe extensions worth knowing:

#table(columns: (auto, 1fr),
  [*Feature*], [*Purpose*],
  [Namespaces], [Logical partitioning of one physical device],
  [Zoned namespaces (ZNS)], [Append-only zones — exposes flash structure to the host],
  [Streams], [Hint which data has correlated lifetime — improves GC],
  [TP4053 simple copy], [Device-side block copy without host data path],
  [NVMe-oF (over Fabrics)], [Same protocol over RDMA, TCP, or FC],
  [Computational storage (TP4091)], [Push compute to the drive],
)

ZNS deserves special mention: it discards the random-write illusion and exposes append-only zones (the underlying NAND erase-block structure). The host FS or DB takes over what was the FTL's job (segment cleaning) but with full-stack visibility. F2FS, btrfs, and RocksDB all have ZNS variants.

== Caching and Write-Back

The page cache buffers reads (read-ahead) and absorbs writes (write-back). The write-back daemon (`pdflush` historically, per-bdi `flusher` threads now) flushes dirty pages when:

- Dirty page count exceeds `dirty_background_ratio` (background flush starts).
- Dirty page count exceeds `dirty_ratio` (writers throttle).
- A page exceeds `dirty_expire_centisecs` (age-based flush).

Pitfall: a process doing `fsync` may stall for tens of seconds while a *different* process's accumulated dirty pages drain — the "fsync stall" problem. Cgroup writeback throttling (`io.latency`, `io.cost`) addresses this by accounting dirty pages per cgroup.

== Stable Writes and the Ordering Question

A `write` returns when bytes are in the page cache, not on disk. To be durable, the application must `fsync` and the FS must issue a barrier (FLUSH + FUA) to force the controller cache. *Many* deployed systems get this wrong:

- The disk controller has a volatile DRAM cache; without FUA the controller's "completion" doesn't mean "on NAND."
- The FS journal needs ordering between journal entry, journal commit, and target write — barriers enforce this.
- The drive may lie. Some consumer SSDs ignore FUA. Power-loss testing is the only honest verification.

The PostgreSQL `fsync` bug of 2018 (Linux silently discarded dirty pages after an EIO without persisting an error to subsequent `fsync`s) demonstrated that even widely-deployed systems can be wrong about layer interactions. The fix took years.

== Distributed Block Storage

Cloud block devices (EBS, Persistent Disk, Azure Disk) and on-prem SAN expose the block-device API over a network. Latency goes from $mu$s to ms; the API model is unchanged. Underlying storage is typically a replicated log (chain replication, Paxos-on-each-write) — see `database/consensus-and-replication.typ` for the protocols.

== Pitfalls

- *Mixed I/O sizes* defeat read-ahead: a `read(4K)` followed by `read(1M)` causes the kernel to prefetch the wrong window.
- *O_DIRECT* bypasses the page cache but requires alignment (LBA + memory). Misalignment may silently fall back to buffered I/O or fail with EINVAL depending on FS.
- *TRIM/discard* is "advice not a command"; many SSDs queue or coalesce discards. A "TRIMmed" drive may still leak old data.
- *4K vs 512 B emulation*: a drive may advertise 512 B sectors while internally using 4 KB ("512e"); unaligned writes do read-modify-write at the controller, halving throughput.

== Further Reading

Axboe, J. (2014). "Linux Block IO: Introducing Multi-queue SSD Access on Multi-core Systems." SYSTOR.

Yang, J. et al. (2014). "When Poll is Better than Interrupt." FAST.

Bjørling, M. et al. (2021). "ZNS: Avoiding the Block Interface Tax for Flash-Based SSDs." USENIX ATC.

Caulfield, A. et al. (2010). "Moneta: A High-Performance Storage Array Architecture for Next-Generation, Non-Volatile Memories." MICRO.

Picoli, I., Hedam, N., Bonnet, P., Bjørling, M. (2020). "uFLIP-OC: Understanding Flash I/O Patterns on Open-Channel Solid-State Drives." APSys.

Corbet, J. (2018). "PostgreSQL's fsync surprise." LWN.

Hellwig, C. (2021). "The Block Layer in Linux 5.x." Vault.

Bovet, D., Cesati, M. "Understanding the Linux Kernel," Chapter 14 (Linux block layer concretized).

#pagebreak()

=== File Systems

A file system is a translation from a flat block device into a named, hierarchical, durable namespace. The translation must survive crashes, scale to billions of files, and present a useful concurrency model. Different file systems answer those constraints differently — and the design choices are remarkably persistent: ext4's roots reach back to ffs (1984), ZFS's snapshot algebra to WAFL (1994), and modern F2FS to log-structured ideas from Sprite LFS (1992).

*See also:* _VFS and Filesystems_ (linux-kernel), _Storage Engines_ (database).

== Anatomy

Almost every general-purpose FS has the same conceptual layout:

#table(columns: (auto, 1fr),
  [*Structure*], [*Role*],
  [Superblock], [Magic, size, root-inode pointer, feature flags],
  [Inode table], [Metadata records — owner, mode, timestamps, extent/block pointers],
  [Directory], [Maps names to inode numbers; itself a file],
  [Allocation map], [Free blocks / inodes — bitmap, B-tree, or extent tree],
  [Journal / log], [Crash-consistency record (if any)],
  [Data blocks], [User payload],
)

A file is its inode, not its name; hard links are multiple directory entries pointing to one inode (reference counted). A symlink is a tiny inode whose payload is a target path resolved at open time.

== Crash Consistency

Without precautions, a multi-block update (e.g., "extend file: allocate block, link block to inode, update inode size") can be torn by a crash: any subset of the writes may have reached disk. The resulting FS may be inconsistent — orphan inode, dangling extent, double-allocated block.

Four established techniques:

*fsck offline repair* (original Unix ffs): assume the FS is consistent on boot; if not, walk every inode and reconcile. Tractable when disks were 100 MB; ruinous on 100 TB. Survived in `e2fsck`.

*Soft updates* (McKusick & Ganger 1999, FreeBSD UFS2): order writes such that any subset visible after crash is *consistent* (possibly with leaked resources, reclaimed by a background scrubber). Elegant, fragile to extend with new operations.

*Journaling* (ext3/4, XFS, JFS): write intentions to a serial log, then apply to the main FS. On recovery, replay the log. Sub-modes:
- *Metadata-only* (default ext4 `data=ordered`): journals metadata, orders data writes before metadata commit. Cheap, correct enough for most.
- *Full data journaling* (`data=journal`): journals everything, doubles write amplification.
- *Writeback* (`data=writeback`): journals metadata, data unordered. Fastest, can expose stale data.

*Copy-on-write* (ZFS, btrfs, APFS, bcachefs): never overwrite. Modified data is written to new blocks; the superblock atomically swings to a new root pointer. Snapshots are free (retain the old root); the cost is per-write *amplification* from updating the entire B-tree spine.

#table(columns: (auto, auto, auto, auto),
  [*FS*], [*Strategy*], [*Snapshot*], [*Checksum*],
  [ext4], [Journal], [via LVM], [metadata only],
  [XFS], [Journal], [`xfs_io`], [v5 metadata],
  [ZFS], [COW], [native], [full data + meta],
  [btrfs], [COW], [native], [full data + meta],
  [F2FS], [LFS], [native], [meta + opt. data],
  [bcachefs], [COW + LSM], [native], [full data + meta],
)

== Allocation and Layout

*Block group* design (ext family, XFS allocation groups): partition the disk into ~1 GB regions, each with its own inode table and bitmap. Allocation tries to keep an inode and its data in the same group to bound seek distance. Effective when "seek" was the dominant cost.

*Extents* (XFS, ext4): represent a contiguous run of blocks as `(start, length)` rather than per-block pointers, drastically reducing metadata for large files. The classic Berkeley FS used direct + indirect block pointers — fine for 1980s files but pathological for video files.

*Log-structured FS* (LFS, F2FS): treat the entire disk as an append-only log. Beautifully fast for writes — no in-place updates — but requires *segment cleaning* (compaction) to recover space, with the LFS-vs-update-in-place tradeoff that has played out repeatedly in storage research. F2FS is the production design optimized for flash (which itself behaves as a log internally).

== Caching: Buffer vs Page

Early Unix kernels had a *buffer cache* (block-indexed) for FS metadata and a separate *page cache* (file-offset-indexed) for `mmap`/`read`. Modern systems unify them — the page cache is authoritative, FS metadata blocks live in it too. This makes `mmap` and `read` see the same data and removes the dual-write hazard.

`fsync(fd)` forces dirty pages of a file plus the FS metadata needed to find them onto stable storage. `fdatasync(fd)` skips metadata that doesn't affect retrieval (e.g., mtime). Crash-safe writes require `fsync` of *both* the file and its containing directory after a `rename` — a subtlety many applications miss (the "fsync gate" controversy of 2009).

== POSIX and Its Discontents

POSIX semantics impose constraints that hurt scaling:

- *Atomic rename* — required for crash-safe write-then-rename idioms.
- *Last-close unlink visibility* — a file unlinked while open persists until the last fd closes.
- *Strong write ordering* — a `read` after a `write` in another thread must see the write (per the byte-range lock if held, otherwise undefined-but-usually-ordered).
- *Hard links across directories* — defeats simple tree mental models.

Distributed FS designs routinely relax these: GFS dropped atomic appends; HDFS made files write-once; object stores (S3, GCS) dropped the directory abstraction entirely (prefixes are a *convention*). The lesson is recurring: full POSIX is hard to scale; relaxed semantics buy throughput.

== Modern Features

*Reflink* (`FICLONE` ioctl): a COW-shared copy. `cp --reflink=auto` copies a 1 TB file in microseconds; subsequent writes diverge. Supported on XFS, btrfs, bcachefs, APFS.

*Snapshots*: a frozen view of the FS at a point in time. COW filesystems get them nearly free; journaling FSes piggyback on LVM (less efficient).

*Send/receive*: serialize the delta between two snapshots for backup/replication. ZFS `send` and btrfs `send` are widely used; bandwidth proportional to changed extents, not total size.

*Checksums*: end-to-end protection against silent data corruption (a real concern at PB scale — see CERN 2007 study finding 1 corruption per ~$10^7$ reads). ZFS pioneered always-on checksums; btrfs followed. ext4 added metadata-only checksums in 2012.

*Encryption*: per-file (fscrypt) or per-volume (LUKS below the FS, ZFS native). Per-file enables differential per-user keys; per-volume is simpler.

== Pitfalls

- *Write barriers and `nobarrier` mount*: disables FUA / cache flushes for speed; survives kernel crashes, *not* power loss. Battery-backed RAID changes the math.
- *Sparse files* are a leaky abstraction: `du` reports allocated, `ls -l` reports logical, `cp` may or may not preserve sparseness.
- *Directory hash collisions* (htree, ext4): pathologically named files can degrade lookups; rare but observable.
- *Cross-FS rename* fails with `EXDEV`; applications must fall back to copy-then-unlink, losing atomicity.

== Further Reading

McKusick, M. et al. (1984). "A Fast File System for UNIX." TOCS.

Rosenblum, M., Ousterhout, J. (1992). "The Design and Implementation of a Log-Structured File System." TOCS.

Hitz, D., Lau, J., Malcolm, M. (1994). "File System Design for an NFS File Server Appliance." USENIX (WAFL).

McKusick, M., Ganger, G. (1999). "Soft Updates: A Technique for Eliminating Most Synchronous Writes in the Fast Filesystem." USENIX ATC.

Bonwick, J., Moore, B. (2003). "ZFS: The Last Word in Filesystems." Sun Microsystems.

Rodeh, O. (2008). "B-trees, Shadowing, and Clones." TOS.

Lee, C. et al. (2015). "F2FS: A New File System for Flash Storage." FAST.

Pillai, T. et al. (2014). "All File Systems Are Not Created Equal: On the Complexity of Crafting Crash-Consistent Applications." OSDI.

Bairavasundaram, L. et al. (2007). "An Analysis of Data Corruption in the Storage Stack." FAST.

Bovet, D., Cesati, M. "Understanding the Linux Kernel," Chapter 12 (Linux-side VFS).

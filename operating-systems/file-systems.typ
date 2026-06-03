= File Systems

A file system is a translation from a flat block device into a named, hierarchical, durable namespace. The translation must survive crashes, scale to billions of files, and present a useful concurrency model. Different file systems answer those constraints differently — and the design choices are remarkably persistent: ext4's roots reach back to ffs (1984), ZFS's snapshot algebra to WAFL (1994), and modern F2FS to log-structured ideas from Sprite LFS (1992).

*See also:* `operating-systems/storage-stack.typ`, `linux-kernel/vfs-and-fs.typ`, `database/storage-engines.typ`.

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

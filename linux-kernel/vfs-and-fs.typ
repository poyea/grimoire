= VFS and Filesystems

The Virtual File System (VFS) is the kernel's polymorphism layer for filesystems. Every `open`, `read`, `write`, `stat`, `unlink` enters a small set of generic functions in `fs/` that dispatch through function-pointer tables (`*_operations` structs) implemented by the concrete filesystem (ext4, XFS, btrfs, F2FS, tmpfs, FUSE, procfs, overlayfs). Without the VFS, every syscall would need a `switch` on filesystem type; with it, a `cat` of `/proc/meminfo` and a `cat` of `/data/foo.txt` on XFS share exactly the same syscall fast path.

== The Four Core Objects

VFS state is built from four reference-counted objects, each with an `_operations` vtable:

#table(columns: (auto, 1fr, 1fr),
  [*Object*], [*Represents*], [*Ops table*],
  [`struct super_block`], [a mounted filesystem instance], [`super_operations`],
  [`struct inode`], [a file (data + metadata, no name)], [`inode_operations`],
  [`struct dentry`], [a name in the directory tree (cached)], [`dentry_operations`],
  [`struct file`], [an open file description (fd target)], [`file_operations`],
)

An inode is the file; a dentry is a path component pointing at it. Hardlinks are two dentries sharing one inode. The `file` is what `fd_array[fd]` ultimately resolves to and carries the current `f_pos`, `f_flags`, and a pointer to the inode.

```c
// include/linux/fs.h (simplified)
struct file {
    struct path             f_path;       // dentry + mount
    struct inode           *f_inode;
    const struct file_operations *f_op;
    atomic_long_t           f_count;
    unsigned int            f_flags;      // O_NONBLOCK, O_APPEND, ...
    fmode_t                 f_mode;       // FMODE_READ, FMODE_WRITE
    loff_t                  f_pos;
    void                   *private_data; // filesystem-private
};
```

The per-process file-descriptor table (`task->files->fdt->fd[]`) holds `struct file *` pointers; `dup`, `dup2`, `fork` all manipulate references on the same `struct file`.

== The Path Walk

`open("/etc/passwd", ...)` is a path walk through dentry cache (dcache) lookups, implemented in `fs/namei.c`. The walker (`link_path_walk` and `walk_component`) consumes path components one at a time, calling `lookup_fast` (RCU walk on the dcache hash) and falling back to `lookup_slow` (which acquires `i_rwsem` on the parent inode and invokes `inode_operations->lookup`).

```c
// Skeleton of the RCU-walk fast path
static int lookup_fast(struct nameidata *nd)
{
    struct dentry *parent = nd->path.dentry;
    struct dentry *dentry = __d_lookup_rcu(parent, &nd->last, &seq);
    if (!dentry || read_seqcount_retry(&parent->d_seq, nd->seq))
        return -ECHILD;            // fall back to ref-walk
    if (d_is_negative(dentry)) return -ENOENT;
    nd->path.dentry = dentry;
    return 0;
}
```

Two walks coexist: an RCU-walk that takes *no* locks and *no* refcounts (relies on seqlocks), and a ref-walk that takes per-dentry references. Most lookups never leave RCU walk; only on permission denial, mountpoint crossing, or symlink resolution does the walker drop back. This is what makes `stat /usr/bin/ls` cost ~200 ns on a hot dcache and not 2 us.

Symlinks are followed by `inode_operations->get_link`; the walker maintains a stack with a depth limit (`MAXSYMLINKS = 40`) to break loops.

== Dcache and Inode Cache

The dcache lives in `fs/dcache.c`. Each `struct dentry` carries:

- `d_name`: the component string (inline for short names).
- `d_parent`: back-pointer for upward walks.
- `d_inode`: the inode it resolves to (NULL for *negative dentries*, which cache non-existence).
- `d_lru`: link into an LRU list for reclaim.
- `d_seq`: seqcount for RCU-walk validation.

Negative dentries are an under-appreciated optimization: a `stat` on a missing file leaves a negative dentry behind so the next miss is a memory access, not an inode-ops lookup. Configure servers do thousands of `stat` calls on optional config files at startup; the dcache absorbs them.

The inode cache (`fs/inode.c`) is hashed by superblock + ino; `iget_locked` is the canonical lookup-or-allocate. Inodes are evicted under memory pressure via the shrinker (`prune_icache_sb`), which respects the per-mount `nr_unused` count surfaced in `/proc/sys/fs/inode-state`.

== File Operations

`read`/`write` dispatch through `file->f_op->read_iter` / `write_iter`. The iter form takes a `struct iov_iter` that abstracts user buffers, kernel buffers, bvec arrays, and pipe pages uniformly; the same `read_iter` services `read`, `pread`, `readv`, `preadv`, and io_uring `IORING_OP_READV`.

```c
// Minimal file_operations for a char device or pseudo-fs
static const struct file_operations my_fops = {
    .owner          = THIS_MODULE,
    .open           = my_open,
    .release        = my_release,
    .read_iter      = my_read_iter,
    .write_iter     = my_write_iter,
    .llseek         = noop_llseek,
    .unlocked_ioctl = my_ioctl,
    .mmap           = my_mmap,
    .fsync          = my_fsync,
};
```

Buffered reads on a regular file end up in `generic_file_read_iter` → `filemap_read`, which copies from the page cache. Direct I/O bypasses the cache via `iomap_dio_rw` (modern path) or filesystem-specific `direct_IO` callbacks.

== Page Cache Integration

The page cache is the unifying performance abstraction of Linux file I/O. Every file-backed inode owns an `address_space` (`inode->i_mapping`): an XArray (formerly radix tree) keyed by page index, storing folios. `read` copies out of folios; `write` dirties them; writeback (`fs/fs-writeback.c`) flushes them under memory pressure or `fsync`.

The `address_space_operations` table (`a_ops`) lets each filesystem plug into the cache:

```c
struct address_space_operations {
    int  (*read_folio)(struct file *, struct folio *);     // fault-in
    int  (*writepages)(struct address_space *, struct writeback_control *);
    bool (*dirty_folio)(struct address_space *, struct folio *);
    int  (*write_begin)(struct file *, struct address_space *,
                        loff_t pos, unsigned len,
                        struct folio **foliop, void **fsdata);
    int  (*write_end)(struct file *, struct address_space *,
                      loff_t pos, unsigned len, unsigned copied,
                      struct folio *folio, void *fsdata);
    sector_t (*bmap)(struct address_space *, sector_t);
    /* ... */
};
```

The `iomap` framework (`fs/iomap/`) is the modern replacement for the `write_begin`/`write_end` dance: filesystems describe an extent map and the generic code handles cache population, dirty tracking, and bio submission. XFS, ext4 (since 6.3 for buffered writes), btrfs (partially), and GFS2 use it. See _MMap and Memory Mapped Files_ for the mmap path that shares this same cache.

Folios — multi-page units introduced in 5.16 and dominant by 6.x — let the page cache track 64 KB or 2 MB extents as one accounting unit, reducing radix-tree pressure and matching modern hardware (NVMe atomic writes, large sector sizes).

== Filesystem Stacking and Mounts

A *mount* binds a superblock at a path. `struct vfsmount` (and its container `struct mount` in `fs/mount.h`) hangs off a *mount namespace*. The same superblock can be mounted at multiple points. The path walker crosses mounts by detecting `DCACHE_MOUNTED` on the dentry and following `mnt_hashtable` to the child mount.

Stacking filesystems layer one inode-ops set over another:

- *overlayfs* (`fs/overlayfs/`): the workhorse behind container images. An `upperdir` and one or more `lowerdir`s; reads fall through to the first layer holding the file; writes to lower files trigger *copy-up* into upper.
- *eCryptfs* and *fscrypt* provide per-file encryption (fscrypt is now in-tree at `fs/crypto/`, used by ext4, F2FS, UBIFS).
- *FUSE* (`fs/fuse/`) exposes userspace filesystems; requests are marshalled to a daemon over `/dev/fuse`.

Bind mounts (`mount --bind src dst`) are a special case: a new `struct mount` referencing the same dentry; no new superblock. Combined with namespaces this is how container runtimes assemble rootfs trees without copying.

== ext4

ext4 (`fs/ext4/`) is the conservative workhorse, the default on most distros and the filesystem with the longest production history.

- *Extents*: replaces ext3's indirect-block tree with extent trees, dramatically improving large-file performance.
- *Journaling* (`fs/jbd2/`): metadata-only by default (`data=ordered`); optional `data=journal` for full data journaling at a steep write-amplification cost.
- *Delayed allocation*: dirty pages aren't assigned blocks until writeback, enabling extent merging and locality.
- *Inline data*: tiny files (less than 60 bytes by default) live in the inode itself.
- *fast commits* (5.10+): a lighter-weight journal path for `fsync` of small metadata changes.

Limits: 1 EiB volume, 16 TiB file, 4 G inodes (allocated at mkfs time; running out is a famously painful recovery scenario).

== XFS

XFS (`fs/xfs/`) is SGI's allocation-group filesystem, dominant for large volumes and parallel workloads.

- *Allocation groups*: the volume is divided into independent AGs (typically 1 GiB each in modern mkfs defaults); allocators in different AGs do not contend.
- *B+tree everything*: free-space (by offset and by size), inode allocation, extent maps — all B+trees, all crc-protected on v5 (the default since 2013).
- *Delayed logging*: the log is an in-memory accumulator flushed in batches; massively reduces journal traffic on metadata-heavy workloads (millions of `creat`/`unlink`/s).
- *Reverse-mapping (rmap)* + *reflink*: per-AG rmap btree lets `xfs_scrub` validate ownership; reflink (4.9+) provides O(1) `cp --reflink` copies via shared extents.
- *Online repair* (`xfs_scrub`, plus `online repair` infrastructure landing across 6.x): the most ambitious filesystem-repair effort in the kernel.

XFS is the canonical choice for >50 TiB volumes; it is the default in RHEL 7+.

== btrfs

btrfs (`fs/btrfs/`) provides copy-on-write, snapshots, checksums, and integrated volume management.

- *CoW everywhere*: every write goes to a new location, then the metadata tree is updated atomically. Snapshots are O(1) — just a new tree root.
- *Subvolumes*: cheap independent filesystems within one storage pool; the unit of snapshots and quotas.
- *Built-in RAID*: 0/1/10/5/6 (RAID 5/6 still flagged unstable for metadata as of 6.x due to the "write hole" problem).
- *Checksums*: every data block has a CRC32C (or BLAKE2b/xxhash/SHA-256) checksum; mismatches surface as `EIO` rather than silently corrupting reads.
- *send/receive*: incremental snapshot replication, used heavily by SUSE and by backup tools.

The CoW model is a tradeoff: superb for snapshots and integrity, painful for random-write-heavy databases (where `chattr +C` to disable CoW is the standard workaround).

== F2FS

F2FS (`fs/f2fs/`) is the Flash-Friendly File System, designed by Samsung for NAND.

- *Log-structured*: writes are sequential into "segments" (2 MiB by default); cleaner reclaims fragmented segments in the background, aligning with how flash translation layers prefer sequential writes.
- *Multi-head logging*: separate hot/warm/cold logs for nodes, data, and metadata reduce garbage-collection cost.
- *NAT/SIT*: node address table and segment-info table form the index, keeping it small and in-memory friendly.
- *Atomic writes* and *volatile writes* for databases that want to manage their own crash consistency.

F2FS dominates Android internal storage; less common on servers but increasingly used on ZNS (Zoned Namespace) SSDs where its log structure is a natural fit.

== procfs, sysfs, debugfs, tmpfs

Pseudo-filesystems are VFS plumbing without a backing store:

- *procfs* (`fs/proc/`): per-process state under `/proc/<pid>/` and global state (`/proc/meminfo`, `/proc/cpuinfo`). Each entry's `read` runs a callback that formats kernel state on demand.
- *sysfs* (`fs/sysfs/`): the device-model export, with `/sys/class/`, `/sys/block/`, `/sys/devices/`. Backed by *kernfs*.
- *debugfs* (`fs/debugfs/`): convention-free dump for developer use; mounted at `/sys/kernel/debug/`.
- *tmpfs* (`mm/shmem.c`): RAM-backed, swap-aware; the backend for `/tmp`, `/dev/shm`, and POSIX shared memory.

Their `file_operations` typically use `seq_file` (`fs/seq_file.c`), an iterator helper that handles partial reads and rendering of variable-length output (the source of `/proc/meminfo`'s formatting).

== FUSE

FUSE (`fs/fuse/`) bridges the VFS to userspace daemons. The kernel module exposes `/dev/fuse`; a daemon reads request structures, performs them, and writes back responses. Each VFS operation that hits a FUSE inode marshalled into a request, queued, and waits.

The cost is two context switches per operation. Recent work (FUSE-BPF, fuse passthrough, io_uring-based FUSE) targets this overhead. `virtiofs` (`fs/fuse/virtio_fs.c`) uses FUSE protocol over virtio rings for VM-to-host filesystem sharing in Firecracker/QEMU, which is far faster than 9P.

== Locking Summary

VFS locking is famously intricate. Highlights:

- *`i_rwsem`*: per-inode read-write semaphore protecting directory operations and metadata. Held shared for lookups, exclusive for `rename`/`unlink`/`create`.
- *`d_lock`*: per-dentry spinlock for cache manipulations.
- *`s_umount`*: per-superblock rwsem held during mount/umount; remount takes it exclusive.
- *Lock ordering for rename*: locks both parent inodes in inode-address order, then both target dentries. `lock_rename` and `lock_rename_child` encapsulate this.

See `Documentation/filesystems/locking.rst` for the authoritative table.

== Observability

bpftrace one-liners for common questions:

```bash
# Files opened, with full path
bpftrace -e 'tracepoint:syscalls:sys_enter_openat {
  printf("%-16s %s\n", comm, str(args->filename));
}'

# Page-cache misses per filesystem
bpftrace -e 'kprobe:filemap_read { @[comm] = count(); }'

# Slow VFS reads (>10 ms)
bpftrace -e '
kprobe:vfs_read { @s[tid] = nsecs; }
kretprobe:vfs_read /@s[tid]/ {
  $d = nsecs - @s[tid];
  if ($d > 10000000) { printf("%s %d ms\n", comm, $d / 1000000); }
  delete(@s[tid]);
}'
```

`/proc/sys/vm/drop_caches=3` flushes dentries, inodes, and page cache — a blunt instrument useful for benchmarking cold-cache numbers, never appropriate in production.

== Further Reading

Bovet, D. and Cesati, M. (2005). _Understanding the Linux Kernel_, ch. 12, 17, 18.

Corbet, J. et al. (2005). _Linux Device Drivers_, 3rd ed., ch. 18 (the chardev `file_operations` skeleton is still current).

Kernel docs: `Documentation/filesystems/vfs.rst`, `path-walking.rst`, `porting.rst`, `locking.rst`.

XFS algorithms: Sweeney, A. et al. (1996). _Scalability in the XFS File System_, USENIX ATC.

ext4: Mathur, A. et al. (2007). _The new ext4 filesystem: current status and future plans_, OLS.

btrfs: Rodeh, O. et al. (2013). _BTRFS: The Linux B-Tree Filesystem_, ACM TOS.

F2FS: Lee, C. et al. (2015). _F2FS: A New File System for Flash Storage_, FAST.

LWN: Corbet's "folio" series (2021-2023); McKenney's coverage of RCU-walk path lookup.

*See also:* _MMap and Memory Mapped Files_ (page cache shares the same folios as file-backed mappings), _IO uring_ (modern async path through the same `read_iter`/`write_iter` entry points), _Cgroups and Namespaces_ (mount namespaces, the container-rootfs primitive), _Block Layer_ (what sits beneath the page cache writeback path).

= The Block Layer

Beneath every filesystem read and every `O_DIRECT` write sits the block layer: the kernel subsystem that batches, reorders, and dispatches I/O to storage devices. It is the home of `struct bio`, request queues, the multi-queue scheduling framework (blk-mq), and the I/O schedulers (`none`, `mq-deadline`, `bfq`, `kyber`). Its job is to bridge the gap between filesystem-level "write this folio" intent and device-level "issue this NVMe command on submission queue 7" mechanics.

The block layer's design has been re-thought twice. The original single-queue `request_queue` (with `elevator_*` schedulers) was built around rotational disks and a single dispatch lock. blk-mq, merged in 3.13 (2014) and made the only path by 5.0 (2019), replaced it with per-CPU software queues and per-device hardware queues, the only viable shape for NVMe devices that expect millions of IOPS.

== struct bio: The Atom of I/O

A `bio` (block I/O) describes one contiguous I/O against a block device. It is the universal currency between filesystems and the block layer.

```c
// include/linux/blk_types.h (simplified)
struct bio {
    struct bio          *bi_next;        // chain (for splits/merges)
    struct block_device *bi_bdev;
    blk_opf_t            bi_opf;         // REQ_OP_READ | REQ_SYNC | ...
    unsigned short       bi_vcnt;        // bvec count
    unsigned short       bi_max_vecs;
    atomic_t             __bi_cnt;       // refcount
    struct bvec_iter     bi_iter;        // sector, size, bvec idx, bvec done
    bio_end_io_t        *bi_end_io;      // completion callback
    void                *bi_private;
    struct bio_vec      *bi_io_vec;      // page/offset/len tuples
};

struct bio_vec {
    struct page         *bv_page;
    unsigned int         bv_len;
    unsigned int         bv_offset;
};
```

A `bio` is a scatter-gather list: each `bio_vec` is a `(page, offset, length)` tuple. A 1 MiB write hitting 256 non-contiguous 4 KiB pages becomes one `bio` with 256 bvecs. Submission is asynchronous: `submit_bio(bio)` returns immediately, and `bi_end_io` is invoked from interrupt context when the device signals completion.

`bi_opf` packs the operation type (`REQ_OP_READ`, `REQ_OP_WRITE`, `REQ_OP_DISCARD`, `REQ_OP_FLUSH`, `REQ_OP_ZONE_APPEND`, ...) with flag bits (`REQ_SYNC`, `REQ_META`, `REQ_PRIO`, `REQ_FUA`, `REQ_PREFLUSH`, `REQ_NOWAIT`, `REQ_POLLED`). The flags drive scheduler decisions and barrier semantics.

== submit_bio and the bio_set Pool

A filesystem allocates a bio from a `mempool` (so allocations don't fail under memory pressure), fills it, and calls `submit_bio`. The kernel transforms bios into requests inside the multi-queue infrastructure.

```c
struct bio *bio = bio_alloc(bdev, nr_pages, REQ_OP_READ, GFP_NOIO);
bio->bi_iter.bi_sector = lba;
bio_add_page(bio, page, len, offset);
bio->bi_end_io = my_complete;
bio->bi_private = ctx;
submit_bio(bio);
```

The `GFP_NOIO` allocation flag forbids recursion back into I/O; this is essential because we are *in* the I/O path; a regular `GFP_KERNEL` allocation could trigger reclaim that itself wants to issue writes, deadlocking. The same logic motivates `PF_MEMALLOC` and the bio mempool.

== Request Queues and blk-mq

A `request_queue` is the per-block-device dispatch structure. Under blk-mq it consists of two layers of queues:

- *Software queues* (`blk_mq_ctx`): one per CPU. Submissions go here first, so the per-CPU lock is uncontended on the hot path.
- *Hardware queues* (`blk_mq_hw_ctx`): one per device hardware submission queue (an NVMe drive may expose 16, 64, or more). A mapping table assigns CPUs to hctx based on NUMA locality and IRQ affinity.

A request (`struct request`) is the merged unit dispatched to the driver; it wraps one or more contiguous bios. Adjacent bios can be merged before dispatch (same device, adjacent LBAs, compatible flags) to amortize per-request overhead. The merge logic lives in `block/blk-merge.c`.

```c
// Driver-side: register the per-device ops
static const struct blk_mq_ops my_mq_ops = {
    .queue_rq    = my_queue_rq,       // dispatch one request
    .complete    = my_complete,        // invoked from softirq after IRQ
    .init_hctx   = my_init_hctx,
    .init_request = my_init_request,
    .poll        = my_poll,            // optional, for IRQ-free poll mode
    .map_queues  = blk_mq_map_queues,
};
```

`queue_rq` is the entry point; it returns `BLK_STS_OK` on success or `BLK_STS_RESOURCE` to push back (the kernel will retry). Completions feed `blk_mq_complete_request`, which may run the `complete` callback on a different CPU than the IRQ via IPI to keep per-CPU caches warm for the originating thread (`rq_affinity` sysfs knob).

== I/O Schedulers

Per hardware queue, a scheduler decides which request goes next. Configured via `/sys/block/<dev>/queue/scheduler`:

#table(columns: (auto, 1fr),
  [`none`], [No reordering. The default for NVMe; the device's own queues and internal scheduling already do the work, so an extra layer just adds latency.],
  [`mq-deadline`], [Two FIFOs per direction (read/write) plus a sorted dispatch tree. Each request has a deadline; if a deadline expires, dispatch that request next regardless of position. The pragmatic choice for SATA SSDs and HDDs where preventing starvation matters.],
  [`bfq`], [Budget Fair Queueing. Per-process I/O queues with weights and budgets; designed for interactive desktop workloads where a `dd` shouldn't make the browser unresponsive. Higher CPU cost; rarely used on servers.],
  [`kyber`], [Token-bucket style: aims for fixed latency targets (`read_lat_nsec`, `write_lat_nsec`) by throttling submissions when measured latency exceeds them. Lighter than BFQ; sometimes used on NVMe when fairness *and* low latency matter.],
)

Rule of thumb: rotational disk → `mq-deadline` (or `bfq`); SATA/SAS SSD → `mq-deadline`; NVMe SSD → `none`; NVMe + multi-tenant fairness needs → `kyber`. Always benchmark; defaults change between kernels.

== The NVMe Path

NVMe is the modern fast path. The driver (`drivers/nvme/host/`) maps each request to a *Submission Queue Entry* (SQE) in a per-CPU NVMe submission queue, rings the doorbell, and lets the device DMA the data. Completion entries land in the matching Completion Queue, raise an MSI-X interrupt, and trigger `blk_mq_complete_request`.

Key NVMe features the block layer plumbs:

- *Multiple queues*: typically `min(nr_cpus, device_max_queues)`. Each queue gets its own MSI-X vector and per-CPU IRQ affinity.
- *Polling* (`hipri` / `REQ_POLLED`) — for ultra-low-latency reads, skip interrupts entirely. io_uring's `IORING_SETUP_IOPOLL` rides this. The driver's `.poll` op checks the CQ; `blk_poll` is called by the consumer in a tight loop.
- *Atomic writes* (NVMe 1.4+) — sector-aligned writes that the device guarantees are torn-free; databases use this to skip double-write buffers.
- *Streams*: hints for write classification, surfaced via `WRITE_HINT_*` and `RWH_WRITE_LIFE_*` (since 4.13).
- *Zoned namespaces* (ZNS) — sequential-write-required zones; the block layer's zoned support (`blk_zoned.c`) tracks zone state and exposes `report_zones`/`reset_zone`/`finish_zone`/`open_zone` operations.

The kernel can issue ~1.5 M IOPS per core to a Gen4 NVMe with io_uring + `none` scheduler + registered files/buffers. The bottleneck moves to PCIe and the device's own controller, not the kernel.

== ublk: Userspace Block Devices Done Right

Userspace block devices used to mean NBD (slow, single-threaded over a socket) or FUSE-like hacks. `ublk` (`drivers/block/ublk_drv.c`, since 6.0) finally does it well: a userspace daemon implements a block device, communicating with the kernel over an io_uring shared-memory ring. Each I/O is one SQE/CQE round-trip; on the userspace side a multi-threaded daemon services them.

```c
// Daemon-side sketch
int ctrl = open("/dev/ublk-control", O_RDWR);
ioctl(ctrl, UBLK_CMD_ADD_DEV, &dev_info);

struct io_uring ring;
io_uring_queue_init(QD, &ring, 0);
ioctl(ctrl, UBLK_CMD_START_DEV, &dev_info);

while (1) {
    struct io_uring_cqe *cqe;
    io_uring_wait_cqe(&ring, &cqe);
    struct ublksrv_io_desc *iod = decode(cqe);
    // do the I/O against the backing store (file, network, RBD, ...)
    commit_completion(&ring, iod, result);
}
```

Use cases: cloud-block-device clients (talk to ceph/RBD/EBS from userspace), encrypted/compressed virtual disks without a kernel module, network-backed disks for VMs. Performance is competitive with kernel drivers when the backing path is itself fast.

== Discard, Trim, and Write Zeroes

The block layer exposes lifecycle hints to flash:

- *`REQ_OP_DISCARD`* / `BLKDISCARD` ioctl — tells the device "I no longer need these LBAs". The FTL can free internal blocks, reducing write amplification. Filesystems issue discard either synchronously on unlink (mount option `discard`) or in batches via `fstrim`.
- *`REQ_OP_WRITE_ZEROES`* — "make this range read as zeros". Modern SSDs do this metadata-only, replacing what used to be a multi-GB sequential zero write.
- *`REQ_OP_SECURE_ERASE`* — cryptographic erase (for self-encrypting drives).

Discard semantics vary by device: some return the prior contents, some return zeros, some are nondeterministic. `/sys/block/<dev>/queue/discard_zeroes_data` reports the device's promise.

== Flushes and Barriers

`fsync` ultimately translates to a *flush* (`REQ_PREFLUSH`) and/or *FUA* (`REQ_FUA`, force unit access) on supporting devices. Without these, writes that have been DMA-accepted by the device may still live in volatile cache; a power loss loses them.

The block layer's flush state machine (`block/blk-flush.c`) ensures correctness even with reorderable queues: when a request carries `REQ_PREFLUSH`, the layer issues a flush, waits for it, then issues the write; `REQ_FUA` ensures the device persists this specific write before signaling completion. This is why databases that rely on `fsync` semantics still get correctness even when the queue depth is hundreds.

If `nobarrier` or `nv_cache` is in play, you are trusting that the device's cache is battery-backed or non-volatile. Most NVMe enterprise drives advertise *Power-Loss Protection* (PLP) and effectively make `REQ_PREFLUSH` a no-op without sacrificing durability.

== Block Throttling: cgroup blkio / io.max

The block layer integrates with cgroup v2's `io` controller (`block/blk-throttle.c`, `block/blk-iolatency.c`, `block/blk-iocost.c`):

- *`io.max`* — hard bandwidth and IOPS caps per device per cgroup.
- *`io.latency`* — soft latency targets; the controller throttles cgroups that push others past their target.
- *`io.cost`* — model-driven proportional sharing; each operation has a cost in "vtime" units, and groups consume vtime according to weights. Used by Facebook's resource control at scale.

```bash
echo "8:0 rbps=10485760 wbps=10485760 riops=1000 wiops=1000" \
     > /sys/fs/cgroup/mygroup/io.max
```

These hooks live on the submission path; throttled bios sleep on a per-cgroup wait list. See _Cgroups and Namespaces_.

== Polling vs Interrupt-Driven I/O

For sub-10 us latency targets, interrupt latency dominates. The block layer supports two polling modes:

- *Classic polling* (`/sys/block/<dev>/queue/io_poll = 1`): the submitter spins in the kernel waiting for completion.
- *Hybrid polling*: estimate the expected device latency, sleep for most of it, then poll. Latency wins without burning a core.
- *io_uring `IOPOLL`*: combines submission and completion polling end-to-end; no interrupts at all on the I/O path.

Polling trades CPU for latency, and on a NUMA system the polling core must share L2/LLC with the submitter to avoid cross-socket coherence traffic.

== Observability

```bash
# Per-disk IOPS/throughput
iostat -xz 1

# Block-layer histograms (microseconds)
bpftrace -e '
kprobe:blk_account_io_start { @start[arg0] = nsecs; }
kprobe:blk_account_io_done /@start[arg0]/ {
  @us = hist((nsecs - @start[arg0]) / 1000);
  delete(@start[arg0]);
}'

# What's blocking on disk?
bpftrace -e '
tracepoint:block:block_rq_insert {
  @[args->comm] = count();
}'
```

`blktrace`/`blkparse` and `btt` give per-IO traces with queue/dispatch/completion timestamps; `iolatency` from `bcc` gives histograms. Recent kernels expose `/sys/block/<dev>/queue/iostats` and PSI's `/proc/pressure/io` for cgroup-level stall accounting.

== Common Tuning Knobs

```bash
# Queue depth (drives request_queue size and merge window)
echo 1024 > /sys/block/nvme0n1/queue/nr_requests

# Disable I/O accounting (saves cycles in hot loops)
echo 0 > /sys/block/nvme0n1/queue/iostats

# Add-randomness (entropy harvesting from completions; usually disable on SSD)
echo 0 > /sys/block/nvme0n1/queue/add_random

# Optimal IO size hint (used by mkfs to align)
cat /sys/block/nvme0n1/queue/optimal_io_size

# Read-ahead (KB)
blockdev --setra 256 /dev/nvme0n1
```

Read-ahead is double-edged: too low hurts sequential reads; too high pollutes the page cache with speculatively read pages that get evicted unused.

== Further Reading

Kernel docs: `Documentation/block/`, especially `blk-mq.rst`, `queue-sysfs.rst`, `writeback_cache_control.rst`, `blktrace.rst`.

Axboe, J. (2013). _Multi-Queue Block I/O_. The original blk-mq design discussion.

Bjørling, M. et al. (2013). _Linux Block IO: Introducing Multi-queue SSD Access on Multi-core Systems_, SYSTOR.

Bjørling, M. et al. (2021). _ZNS: Avoiding the Block Interface Tax for Flash-based SSDs_, USENIX ATC.

NVMe Base Spec: #link("https://nvmexpress.org/specifications/")[nvmexpress.org].

LWN: "An introduction to blk-mq" (Corbet 2014); "ublk" series (2022-2023); "Zoned namespaces" (2020).

`drivers/nvme/host/pci.c`, `block/blk-mq.c`, `block/blk-flush.c`, `block/mq-deadline.c`, `block/kyber-iosched.c`, `block/bfq-iosched.c`.

*See also:* _VFS and Filesystems_ (filesystems submit the bios this layer dispatches), _IO uring_ (the IOPOLL mode rides the polling path), _Cgroups and Namespaces_ (io controller), _Interrupts and Bottom Halves_ (MSI-X delivery and IRQ affinity for completion path).

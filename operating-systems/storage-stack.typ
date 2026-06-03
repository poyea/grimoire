= The Storage Stack

Between `write(2)` and a flash cell sit half a dozen layers, each with its own queueing, ordering, and failure semantics. Understanding the stack as a whole — VFS, page cache, block layer, I/O scheduler, driver, controller, media — is what separates a system that scales from one that mysteriously stalls under load. This chapter is the conceptual treatment; Linux-specific `blk-mq` and io_uring are in `linux-kernel/block-layer.typ` and `linux-kernel/io-uring.typ`.

*See also:* `operating-systems/file-systems.typ`, `linux-kernel/block-layer.typ`, `linux-kernel/io-uring.typ`, `database/buffer-pool-and-io.typ`.

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

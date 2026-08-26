#import "../template.typ": xref

= I/O Performance

Storage and network I/O are where nanosecond-scale CPUs meet microsecond- and millisecond-scale devices, and where the operating system inserts its thickest layer of indirection: page caches, schedulers, socket buffers, protocol stacks. This chapter covers the storage stack from syscall to flash, the asynchronous I/O models and `io_uring`, network performance from socket tuning to kernel bypass, and the measurement tools that distinguish device limits from software overhead.

*See also:* #xref("performance-engineering", "queueing-theory", label: "Queueing Theory") (disks and NICs are queueing systems; `iostat`'s columns are its vocabulary), #xref("performance-engineering", "concurrency-performance", label: "Concurrency Performance") (thread-pool sizing for blocking I/O), #xref("performance-engineering", "benchmarking", label: "Benchmarking") (`fio` discipline), and #xref("performance-engineering", "capacity-planning", label: "Capacity Planning") (provisioning for I/O headroom).

== The Numbers

Approximate per-operation costs, current hardware:

#table(
  columns: 3,
  [*Operation*], [*Latency*], [*Notes*],
  [Syscall (round trip)], [100-300 ns], [More with mitigations (KPTI) enabled],
  [Page-cache read hit], [1-5 microseconds], [Dominated by copy and syscall],
  [NVMe flash read], [10-100 microseconds], [QD1, 4 KiB],
  [SATA SSD read], [50-150 microseconds], [Interface-limited at about 550 MB/s],
  [HDD seek + rotation], [4-12 ms], [About 100-200 IOPS per spindle],
  [Same-DC network RTT], [50-500 microseconds], [Plus serialization],
  [Cross-region RTT], [tens of ms], [Speed of light is the floor],
)

Two consequences. First, an HDD is about $10^5$ times slower per random operation than DRAM, and an NVMe drive about $10^2$-$10^3$ times; the page cache exists to hide this. Second, modern NVMe devices reach hundreds of thousands to millions of IOPS, but only at high queue depth: software that issues one request at a time (queue depth 1) sees the per-op latency, not the device's parallelism, and a million-IOPS device delivers 20K IOPS to a synchronous single-threaded caller.

== The Storage Stack

A `read()` traverses: VFS, the filesystem, the page cache, the block layer (request merging, the I/O scheduler), the NVMe/SCSI driver, and the device. Each layer has levers:

- *Page cache*: reads hit cache after first touch; writes are buffered (dirty pages) and written back by `kworker` threads per `vm.dirty_ratio` / `dirty_background_ratio`. The classic burst pathology: a large write fills dirty memory at RAM speed, then `fsync` or the dirty threshold stalls everything at disk speed. `O_DIRECT` bypasses the cache for applications with their own buffer pools (databases); `fadvise(DONTNEED)` and `madvise` give the kernel lifetime hints.
- *Filesystem*: extent allocation, journaling mode, and metadata behavior differ (ext4, XFS, ZFS, Btrfs); `fallocate` preallocation avoids fragmentation and allocation on the write path; `fsync` cost varies by filesystem and journal placement and is the latency floor of every durable commit.
- *Block layer*: requests are merged and reordered. Schedulers: `none` (NVMe default; the device reorders better than software), `mq-deadline` (bounded starvation), `bfq` (fairness for desktops). For NVMe, `none` is almost always right.
- *The device*: SSDs are log-structured internally; the flash translation layer (FTL) remaps writes, garbage-collects, and slows down when full or unTRIMmed. Sustained random-write performance can be a small fraction of the fresh-drive number; steady-state benchmarks require preconditioning (see _Benchmarking_). Mixed read/write workloads interfere: a write burst inflates read tail latency.

=== Sequential vs. random, sync vs. async

The four-quadrant characterization of any storage workload: block size, read/write mix, sequential/random, and queue depth. HDDs care overwhelmingly about sequentiality (200 MB/s sequential vs. about 1 MB/s at 4 KiB random); SSDs care about queue depth and write history. Log-structured designs (LSM trees, write-ahead logs, Kafka) exist to convert random writes into sequential ones, a transformation worth orders of magnitude on HDDs and still material on flash (FTL-friendly, lower write amplification).

== Asynchronous I/O Models

- *Blocking + threads*: simple; costs a thread (stack memory, switch latency) per in-flight operation. Fine into the thousands of concurrent operations, with the pool sized per the wait/service ratio.
- *Readiness-based (`epoll`, `kqueue`)*: the kernel reports readiness, the application then performs nonblocking reads/writes. The standard model for network servers (nginx, Node.js, event loops), but it does not work for regular files, which are always "ready".
- *Completion-based*: submit an operation, get notified on completion. POSIX AIO is largely a glibc thread-pool emulation; Linux-native `libaio` works only with `O_DIRECT` and silently blocks in edge cases. Windows IOCP got this model right decades earlier.
- *`io_uring`* (Jens Axboe, Linux 5.1, 2019): two shared-memory rings (submission and completion) between user space and kernel. Batches many operations per syscall, or with `SQPOLL` eliminates syscalls from the steady state entirely; supports files, sockets, `fsync`, `accept`, timeouts, and chained ops; registered buffers and zero-copy variants cut copy costs. It is the first Linux interface delivering full NVMe queue-depth parallelism from one thread, and the foundation for high-performance runtimes (Tokio's `io-uring` backend, libuv experiments, RocksDB's `MultiRead`).

The rule of thumb: storage devices deliver their rated IOPS only if the software keeps their queues full; the async model is how one thread keeps 64+ operations in flight.

== Network Performance

=== The software path

A packet's receive path: NIC DMA to a ring buffer, interrupt (moderated/coalesced), driver NAPI poll, protocol stack, socket buffer, application `read`. Costs that matter:

- *Per-packet, not per-byte, overhead dominates at small sizes*: the stack costs microseconds per packet, so 64-byte packets at line rate are far harder than 1500-byte ones. Offloads (TSO/GSO on send, GRO/LRO on receive) batch segmentation so the stack sees fewer, larger units; jumbo frames help within a controlled network.
- *Copies and syscalls*: `sendfile` and `splice` avoid the user-space copy for file-to-socket paths; `MSG_ZEROCOPY` helps for large sends; `io_uring` batches the syscalls.
- *Scaling across cores*: RSS hashes flows to multiple NIC queues with per-queue interrupts; RPS/RFS steer in software; SO_REUSEPORT shards an accept queue across worker processes. Without these, one core saturates handling all interrupts while the rest idle: visible as a single hot `ksoftirqd`.

=== Latency vs. throughput tuning

- *Throughput*: ensure window sizes cover the bandwidth-delay product ($"BDP" = "bandwidth" times "RTT"$; 10 Gbit/s at 10 ms needs about 12.5 MB of window); enable autotuning (`tcp_rmem`/`tcp_wmem` maxima); pick a congestion control suited to the path (CUBIC default; BBR for lossy or long-fat paths, where loss-based control collapses).
- *Latency*: `TCP_NODELAY` for request/response protocols (Nagle's algorithm interacting with delayed ACKs is the classic hidden 40 ms); interrupt coalescing trades per-packet latency for CPU; busy polling (`SO_BUSY_POLL`) burns CPU to shave the interrupt path.
- *Connection costs*: TCP handshake (1 RTT) plus TLS (1-2 RTTs) make connection reuse and pooling mandatory for short requests; TLS session resumption and QUIC 0-RTT attack the same overhead protocol-side.

=== Kernel bypass

When per-packet software cost is the bottleneck (millions of packets per second, microsecond budgets): *DPDK* (poll-mode drivers in user space, dedicated spinning cores, the standard in NFV and trading systems), *AF_XDP* (zero-copy sockets fed from an XDP program, a middle ground retaining the kernel), and *RDMA* (the NIC reads/writes remote memory directly, single-digit microsecond one-sided ops, standard in HPC and increasingly in storage backends via NVMe-oF). The cost is operational: no kernel tooling, custom stacks, burned cores.

== Measurement

- `iostat -x`: per-device IOPS (`r/s`, `w/s`), throughput, queue depth (`aqu-sz`), per-I/O latency (`r_await`, `w_await`), and `%util` (which, on parallel devices, does not mean "full": see #xref("performance-engineering", "methodology", label: "Performance Methodology")).
- *BPF tools* (bcc/bpftrace): `biolatency` (block-I/O latency histograms, the honest view of the tail), `biosnoop` (per-I/O trace with process attribution), `ext4slower`/`xfsslower` (filesystem-level operations exceeding a threshold, which include cache and locking effects the block layer never sees), `tcpretrans`, `tcplife`.
- `fio`: the standard storage microbenchmark; specify engine (`io_uring`), block size, queue depth, randomness, and runtime, and report latency percentiles, not just bandwidth. Sweep queue depth to map the device's throughput-latency curve.
- Network: `ss -ti` (per-connection RTT, cwnd, retransmits), `nstat` (stack counters), `iperf3` (path capacity), `ethtool -S` (NIC drops and per-queue stats).

The recurring diagnostic question is *where the time goes*: device service time (`fio` against the raw device), filesystem and cache (BPF filesystem tools), or the application's own serialization on the I/O path (off-CPU profiling).

== Pitfalls

- *Benchmarking the page cache*: a "disk" benchmark whose working set fits in RAM measures memcpy. Use `O_DIRECT`, working sets larger than RAM, or explicit cache drops, and say which you did.
- *Fresh-drive SSD numbers*: performance before the FTL reaches steady state overstates sustained random writes severalfold. Precondition.
- *QD1 numbers quoted as device capability* (or the reverse: high-QD throughput quoted where the application runs at QD1).
- *Ignoring `fsync` semantics*: a database benchmark with `fsync` disabled, or on a device that lies about flush (volatile write cache without power-loss protection), measures a different durability contract.
- *Tail-blindness*: mean disk latency hides GC pauses inside the SSD; only latency histograms (`biolatency`, `fio` percentiles) show the multimodal truth.

== Further Reading

- Gregg, B. (2020). _Systems Performance_, 2nd ed., chs. 8-10 (File Systems, Disks, Network). Addison-Wesley.
- Axboe, J. (2019). Efficient IO with io_uring. Kernel documentation/paper, kernel.dk.
- Corbet, J., Rubini, A., & Kroah-Hartman, G. (2005). _Linux Device Drivers_, 3rd ed. O'Reilly.
- Cardwell, N. et al. (2016). BBR: congestion-based congestion control. _ACM Queue_, 14(5).
- Didona, D. et al. (2022). Understanding modern storage APIs: a systematic study. _SYSTOR_ (and the SPDK literature, Yang et al. 2017); see also Barroso, L. et al. (2017). Attack of the killer microseconds. _CACM_, 60(4).

= io_uring

The traditional Linux I/O syscalls (`read`, `write`, `recv`, `send`, `epoll_wait`) are *synchronous from the caller's point of view*: each operation crosses the ring boundary, the kernel does the work, and control returns. At ~120-250 ns per crossing (see _ABI and Syscalls_), an IO-heavy workload doing millions of small operations per second spends a substantial fraction of its CPU on the transition itself, not the work. POSIX AIO never lived up to its name — the glibc implementation is a userspace thread pool — and the kernel `aio_*` interface is restricted to `O_DIRECT` file I/O. `io_uring`, introduced in Linux 5.1 (May 2019) by Jens Axboe, is the first truly general-purpose async I/O interface on Linux: arbitrary syscalls, batched, with zero per-operation ring transition in the fast path.

The cost model changes from "N operations = N syscalls" to "N operations = 1 syscall (or 0 with `SQPOLL`)". For Nginx-class workloads, this can be the difference between 1 M and 3 M req/s on the same hardware.

== Two Rings in Shared Memory

The userland program and the kernel communicate through *two* lock-free single-producer / single-consumer ring buffers, both backed by pages the kernel `mmap`s into the process address space:

#table(columns: (auto, 1fr, 1fr),
  [*Ring*], [*Producer*], [*Consumer*],
  [Submission Queue (SQ)], [userspace], [kernel],
  [Completion Queue (CQ)], [kernel], [userspace],
)

Each entry in the SQ is a `struct io_uring_sqe` (64 bytes — one cache line on x86-64, deliberate); each entry in the CQ is a `struct io_uring_cqe` (16 bytes by default, 32 with `IORING_SETUP_CQE32`). The SQE describes a syscall to perform — opcode, fd, buffer pointer, length, plus a 64-bit `user_data` cookie that the kernel echoes back in the matching CQE.

Because both rings are in shared memory, the only thing the kernel cares about is the head/tail indices. Userspace updates the SQ tail with a release store; the kernel observes the new tail with an acquire load. There is no syscall on the submission path until the kernel needs to be woken up.

```c
// Layout (simplified) of what io_uring_setup() maps into the process:
struct sq_ring {
    uint32_t *head;        // kernel-owned (kernel advances after consuming)
    uint32_t *tail;        // user-owned   (user advances after publishing)
    uint32_t *ring_mask;   // mask = ring_entries - 1
    uint32_t *array;       // indirection: array[idx & mask] -> sqe index
    struct io_uring_sqe *sqes;
};
struct cq_ring {
    uint32_t *head;        // user-owned   (user advances after consuming)
    uint32_t *tail;        // kernel-owned (kernel advances after publishing)
    struct io_uring_cqe *cqes;
};
```

The `array` indirection on the SQ side lets userspace fill SQEs in any order and then submit a permutation — useful for prioritizing or for keeping a pool of SQEs hot in cache.

== Setup and the Three Syscalls

There are only three syscalls in the whole interface:

#table(columns: (auto, 1fr),
  [`io_uring_setup(entries, params)`], [create a ring, return an fd that owns the SQ/CQ memory],
  [`io_uring_enter(fd, to_submit, min_complete, flags, sig)`], [kick the kernel to process SQEs and/or wait for CQEs],
  [`io_uring_register(fd, opcode, args, nr_args)`], [pre-register resources (buffers, files, eventfd, ...)],
)

Everything else is library code on top of those three. The blessed userspace library is *liburing*; below is a hand-rolled raw-syscall setup to make the kernel interface visible:

```c
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

static int io_uring_setup(unsigned entries, struct io_uring_params *p) {
    return syscall(__NR_io_uring_setup, entries, p);
}

int ring_init(struct app_ring *r, unsigned entries) {
    struct io_uring_params p = {0};
    int fd = io_uring_setup(entries, &p);
    if (fd < 0) return -1;

    size_t sq_sz = p.sq_off.array + p.sq_entries * sizeof(uint32_t);
    size_t cq_sz = p.cq_off.cqes  + p.cq_entries * sizeof(struct io_uring_cqe);

    void *sq = mmap(NULL, sq_sz, PROT_READ|PROT_WRITE,
                    MAP_SHARED|MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    void *cq = mmap(NULL, cq_sz, PROT_READ|PROT_WRITE,
                    MAP_SHARED|MAP_POPULATE, fd, IORING_OFF_CQ_RING);
    void *sqes = mmap(NULL, p.sq_entries * sizeof(struct io_uring_sqe),
                      PROT_READ|PROT_WRITE, MAP_SHARED|MAP_POPULATE,
                      fd, IORING_OFF_SQES);

    r->fd = fd;
    r->sq_head = (uint32_t*)((char*)sq + p.sq_off.head);
    r->sq_tail = (uint32_t*)((char*)sq + p.sq_off.tail);
    r->sq_mask = *(uint32_t*)((char*)sq + p.sq_off.ring_mask);
    r->sq_array = (uint32_t*)((char*)sq + p.sq_off.array);
    r->sqes = (struct io_uring_sqe*)sqes;
    r->cq_head = (uint32_t*)((char*)cq + p.cq_off.head);
    r->cq_tail = (uint32_t*)((char*)cq + p.cq_off.tail);
    r->cqes = (struct io_uring_cqe*)((char*)cq + p.cq_off.cqes);
    return 0;
}
```

`MAP_POPULATE` is worth it here: these pages will be touched on every submission, page faulting them on the hot path is a regression.

== Submission and Completion

The fast-path loop is a four-step dance entirely in userland — read the head/tail, write an SQE, publish, eventually reap:

```c
static struct io_uring_sqe *get_sqe(struct app_ring *r) {
    uint32_t tail = *r->sq_tail;
    uint32_t head = __atomic_load_n(r->sq_head, __ATOMIC_ACQUIRE);
    if (tail - head >= (r->sq_mask + 1)) return NULL;  // ring full
    struct io_uring_sqe *sqe = &r->sqes[tail & r->sq_mask];
    r->sq_array[tail & r->sq_mask] = tail & r->sq_mask;
    *r->sq_tail = tail + 1;  // not yet visible; release happens at submit
    return sqe;
}

static void prep_read(struct io_uring_sqe *sqe, int fd, void *buf,
                      size_t len, off_t off, uint64_t cookie) {
    sqe->opcode = IORING_OP_READ;
    sqe->fd = fd;
    sqe->addr = (uint64_t)buf;
    sqe->len = len;
    sqe->off = off;
    sqe->user_data = cookie;
    sqe->flags = 0;
}

static int submit(struct app_ring *r, unsigned to_submit, unsigned min_complete) {
    __atomic_thread_fence(__ATOMIC_RELEASE);  // publish tail
    return syscall(__NR_io_uring_enter, r->fd, to_submit, min_complete,
                   min_complete ? IORING_ENTER_GETEVENTS : 0, NULL, 0);
}
```

Completions are reaped without any syscall at all when results are already there:

```c
static int reap(struct app_ring *r, void (*handle)(struct io_uring_cqe*)) {
    uint32_t head = *r->cq_head;
    uint32_t tail = __atomic_load_n(r->cq_tail, __ATOMIC_ACQUIRE);
    int n = 0;
    while (head != tail) {
        handle(&r->cqes[head & r->sq_mask]);
        head++; n++;
    }
    __atomic_store_n(r->cq_head, head, __ATOMIC_RELEASE);
    return n;
}
```

The `user_data` field is the only thing tying a CQE back to its SQE. A common idiom is to stuff a tagged pointer in there: low bits identify the operation type, high bits a pointer to a request struct.

== SQPOLL: Zero Syscalls in the Steady State

Even with batching, `io_uring_enter` is still a syscall. `IORING_SETUP_SQPOLL` removes it: the kernel spawns a dedicated kthread (per ring) that busy-polls the SQ tail. The producer just publishes SQEs and the kernel picks them up; no `enter` needed.

```c
struct io_uring_params p = {
    .flags = IORING_SETUP_SQPOLL,
    .sq_thread_idle = 1000,   // ms before the kthread parks
};
io_uring_setup(entries, &p);
```

After `sq_thread_idle` ms of inactivity the kthread sleeps and sets `IORING_SQ_NEED_WAKEUP` in the ring flags; the user must re-issue `io_uring_enter(.., IORING_ENTER_SQ_WAKEUP)` to revive it. Real-world code checks that flag on every submit.

Trade-offs of SQPOLL:

- *Latency:* near-zero on the submission side; a CPU core is dedicated to polling.
- *Throughput:* eliminates ~80-150 ns per submission (the syscall transition) — at 5 M ops/s that's 25-50% of a core saved on the producer.
- *Power / packing:* the dedicated poller is always-on while busy; pin it to an isolated core (`IORING_SETUP_SQ_AFF` + `sq_thread_cpu`) so it doesn't steal user time. See _CPU Affinity and Isolation_.
- *Privilege:* unprivileged SQPOLL was restricted after CVE-2022-29582; modern kernels require either CAP_SYS_NICE or a matching ring fd to attach to an existing poller via `IORING_SETUP_ATTACH_WQ`.

== liburing Equivalents

Hand-rolling SQEs is rarely worth it. liburing wraps the same logic and is what real applications use:

```c
#include <liburing.h>

struct io_uring ring;
io_uring_queue_init(256, &ring, 0);

struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
io_uring_prep_read(sqe, fd, buf, 4096, 0);
io_uring_sqe_set_data64(sqe, REQ_READ);
io_uring_submit(&ring);

struct io_uring_cqe *cqe;
io_uring_wait_cqe(&ring, &cqe);
if (cqe->res < 0) fprintf(stderr, "read: %s\n", strerror(-cqe->res));
io_uring_cqe_seen(&ring, cqe);
```

`io_uring_submit_and_wait(&ring, n)` is the canonical batched-event-loop call — one syscall handles both submission and completion-waiting.

== Linked, Drained, Async, and Hardlink SQEs

`sqe->flags` can chain operations without a userspace round trip:

#table(columns: (auto, 1fr),
  [`IOSQE_IO_LINK`], [next SQE waits for this one to succeed; on failure all linked are short-circuited with `-ECANCELED`],
  [`IOSQE_IO_HARDLINK`], [like `LINK` but next runs regardless of this one's result],
  [`IOSQE_IO_DRAIN`], [defer this SQE until all *previously* submitted ones complete],
  [`IOSQE_ASYNC`], [force kernel to push into the io-wq thread pool instead of trying inline],
  [`IOSQE_FIXED_FILE`], [`sqe->fd` is an index into the registered file table, not a raw fd],
  [`IOSQE_BUFFER_SELECT`], [pick a buffer from a registered buffer group at completion time],
)

Linked SQEs are how you do `openat` → `read` → `close` as one logical request without any user-side state machine, or `connect` → `send` → `recv` in a single submit:

```c
sqe = io_uring_get_sqe(&ring);
io_uring_prep_openat(sqe, AT_FDCWD, path, O_RDONLY, 0);
sqe->flags |= IOSQE_IO_LINK;

sqe = io_uring_get_sqe(&ring);
io_uring_prep_read(sqe, /*fd hint=*/ -1, buf, 4096, 0);
sqe->flags |= IOSQE_FIXED_FILE | IOSQE_IO_LINK;
sqe->fd = /* slot index, populated by previous open via fixed-files */;

sqe = io_uring_get_sqe(&ring);
io_uring_prep_close_direct(sqe, slot);

io_uring_submit(&ring);  // 1 syscall, 3 operations
```

== Registered Resources

`io_uring_register` pre-pins resources so the kernel doesn't have to re-validate them on every operation:

- *Buffers* (`IORING_REGISTER_BUFFERS`): pin user pages once. `IORING_OP_READ_FIXED` / `WRITE_FIXED` skip `get_user_pages` on each call — a ~10-20% speedup for high-rate small I/O.
- *Files* (`IORING_REGISTER_FILES`): a slot table indexed by `sqe->fd` when `IOSQE_FIXED_FILE` is set. The kernel takes a reference once; per-op it's an array index instead of a hashtable lookup.
- *Eventfd* (`IORING_REGISTER_EVENTFD`): wake an eventfd whenever a CQE arrives — bridges io_uring into an existing epoll-based event loop.
- *Ring restrictions* (`IORING_REGISTER_RESTRICTIONS`): allowlist of opcodes/flags. Combined with a seccomp filter, used by hardened sandboxes to whitelist exactly which io_uring operations a sandboxed process may issue.

Provided buffers (`IORING_OP_PROVIDE_BUFFERS` / `IORING_REGISTER_PBUF_RING`) are the dual of fixed buffers — you give the kernel a pool of receive buffers up front, and `recv` ops with `IOSQE_BUFFER_SELECT` pull one at completion time (the buffer ID is reported in `cqe->flags >> 16`). This is the foundation of zero-copy receive paths: the kernel never blocks waiting for userspace to post a buffer, and userspace never speculatively posts one per connection.

== Multishot Operations

A normal `recv` SQE produces one CQE and is done. A *multishot* `recv` keeps firing CQEs every time data arrives until it's cancelled or the buffer pool runs dry:

```c
sqe = io_uring_get_sqe(&ring);
io_uring_prep_recv_multishot(sqe, fd, NULL, 0, 0);
sqe->flags |= IOSQE_BUFFER_SELECT;
sqe->buf_group = GROUP_ID;
```

Each completion has `IORING_CQE_F_MORE` set; the absence of that flag means the operation has terminated and you must re-arm. Multishot also exists for `accept`, `poll`, and `timeout`. For an HTTP server with 100 K connections, multishot accept removes the "wake up to call accept then go back to sleep" loop entirely.

== Zero-Copy Send

`IORING_OP_SEND_ZC` (Linux 6.0+) is the network equivalent of `splice` for outbound traffic: the kernel pins the user buffer, hands it directly to the NIC's tx queue, and emits *two* CQEs — one when the send is "issued" (`IORING_CQE_F_MORE` set, kernel has the data) and a second `IORING_CQE_F_NOTIF` when the NIC has actually transmitted the bytes and the buffer is safe to free or overwrite. The userspace contract is: don't touch the buffer between the two CQEs.

Microbenchmarks (loopback, 64 KB writes, modern kernel): non-zc `send` ~5.2 GB/s/core; `send_zc` ~12 GB/s/core. The win narrows on real NICs because the bottleneck shifts to PCIe and DMA, but the CPU-cycles-per-byte still drops by 30-50%.

== Cancellation

`IORING_OP_ASYNC_CANCEL` cancels a previously submitted operation by its `user_data` cookie or by fd. Returns `0` on cancel, `-ENOENT` if already completed, `-EALREADY` if cancellation is in flight. The cancelled op completes with `-ECANCELED`. This is how you implement request timeouts cleanly:

```c
sqe = io_uring_get_sqe(&ring);
io_uring_prep_recv(sqe, fd, buf, sz, 0);
io_uring_sqe_set_data64(sqe, REQ_ID);
sqe->flags |= IOSQE_IO_LINK;

sqe = io_uring_get_sqe(&ring);
struct __kernel_timespec ts = { .tv_sec = 5 };
io_uring_prep_link_timeout(sqe, &ts, 0);
// if recv hasn't completed in 5s, the linked timeout fires and recv is cancelled
io_uring_submit(&ring);
```

== Comparison with epoll and POSIX AIO

#table(columns: (auto, 1fr, 1fr, 1fr),
  [], [*epoll*], [*POSIX/libaio*], [*io_uring*],
  [Model], [readiness], [completion], [completion],
  [Op coverage], [pollable fds only], [`O_DIRECT` files only], [almost every syscall],
  [Syscalls/op (steady)], [1-2], [1], [0 (SQPOLL) or 1/batch],
  [Buffered file I/O], [no (blocks)], [no], [yes],
  [Sockets], [yes], [no], [yes],
  [Batching], [submit one-at-a-time], [batched], [batched],
  [Linked sequences], [no], [no], [yes],
  [Zero-copy], [via `splice`/`sendfile`], [no], [`SEND_ZC`, fixed buffers],
  [Cancellation], [close fd], [`io_cancel` (best-effort)], [first-class],
)

The "completion" vs "readiness" distinction matters: epoll tells you "fd is readable now", and you still have to call `read`. io_uring tells you "the read you asked for is done, here are the bytes". With epoll your read can still return `EAGAIN` after a spurious wakeup; with io_uring the data is in the buffer when the CQE appears.

== Performance Numbers

Rough orders of magnitude (5.15+ kernel, modern x86-64):

- *Null op* (`IORING_OP_NOP`, batched 32 at a time, no SQPOLL): ~25 ns/op amortized.
- *4 KB cached file read*, registered buffer + registered fd: ~350 ns/op vs ~550 ns for `pread` (~35% lower).
- *Single-socket TCP echo* (1 KB messages, 1 connection): ~1.1 M req/s with io_uring + multishot recv vs ~700 K with epoll + recv (~55% higher).
- *Disk I/O* (NVMe, 4 KB random reads, `O_DIRECT`, queue depth 128): saturates the device at the same IOPS as libaio but with ~40% lower CPU. This is why Ceph, ScyllaDB, and PostgreSQL added io_uring backends.
- *SQPOLL overhead:* one CPU core pinned at 100% per ring while the kthread is awake. Worth it above ~100 K ops/s; lossy below that.

== Pitfalls

- *Buffer lifetime.* SQEs reference user buffers by raw pointer. Don't free or stack-pop the buffer until the matching CQE arrives. Easy bug class.
- *SQE field reuse after submit.* The SQE memory is owned by the kernel as soon as the tail is published. Treat `io_uring_get_sqe` as "allocate" and stop touching the SQE the moment you call `submit`.
- *CQE backpressure.* If the CQ fills, the kernel sets `IORING_SQ_CQ_OVERFLOW` and starts dropping completions (or stores them in an overflow list, depending on kernel). Always size CQ entries $>=$ in-flight SQEs (`IORING_SETUP_CQSIZE`).
- *fork(2).* The ring is not inherited safely across fork; child must not touch it. Use `IORING_SETUP_REGISTERED_FD_ONLY` and explicit teardown.
- *Signal handling.* `io_uring_enter`'s `sig` argument plays the same role as `pselect`'s mask — use it instead of self-pipe tricks.

== Security

io_uring is powerful enough that it has had a steady CVE drumbeat. The bug pattern is typically a kernel-side use-after-free between an SQE in-flight and a resource (file, socket, buffer) being torn down. Hardened deployments default `kernel.io_uring_disabled = 2` (disable for unprivileged users) and rely on seccomp `SECCOMP_FILTER_ARCH_*` to block the three syscalls outright in untrusted processes. Google's ChromeOS and Android disable io_uring in sandboxed renderers for this reason.

For server-side workloads where you control the binary, the right move is the opposite: enable io_uring, use `IORING_REGISTER_RESTRICTIONS` to allowlist only the opcodes you need, register all files and buffers up front, and run the SQPOLL kthread on an isolated core.

== Further Reading

Axboe, J. (2019). _Efficient IO with io_uring_. Available at #link("https://kernel.dk/io_uring.pdf")[kernel.dk/io_uring.pdf].

liburing source: #link("https://github.com/axboe/liburing")[github.com/axboe/liburing] — examples/ directory is the best tutorial.

Linux kernel `io_uring/` subtree at #link("https://elixir.bootlin.com/linux/latest/source/io_uring")[elixir.bootlin.com] — start with `io_uring.c` and `io_uring/rsrc.c`.

Corbet, J. (2020-2024). _io_uring_ article series on LWN.net — incremental coverage of every new opcode and security event.

_See also: ABI and Syscalls (syscall transition cost model), CPU Affinity and Isolation (SQPOLL placement), Interrupts and NAPI (the kernel-side path that feeds io_uring network completions)._

#import "../template.typ": xref

= Inter-Process Communication <ipc-mechanisms>

Once the operating system isolates processes into separate address spaces, it must hand back a controlled way for them to cooperate. Inter-process communication (IPC) is the set of primitives that punch holes (carefully) through that isolation: to move bytes, to share pages, to wake a sleeper, or to pass a capability. Every IPC mechanism is a negotiated trade across three axes: cost per message, payload framing, and channel lifetime ownership. This chapter treats those primitives conceptually; their Linux realizations live in `linux-kernel/networking-stack.typ` and the kernel's signal and futex code.

*See also:* #xref("operating-systems", "processes-and-threads", label: "Processes and Threads"), #xref("operating-systems", "memory-management", label: "Memory Management"), #xref("networking", "sockets-api", label: "POSIX Sockets API") (networking), #xref("cpu-architecture", "synchronization", label: "Synchronization Primitives") (architecture).

== A Taxonomy

It helps to sort the zoo by what the mechanism fundamentally does, not by which syscall family it belongs to:

#table(columns: (auto, 1fr),
  [*Class*], [*Members*],
  [Data transfer], [Pipes, FIFOs, message queues, sockets (stream + datagram)],
  [Shared memory], [`mmap` MAP_SHARED, `memfd`, System V / POSIX `shm`],
  [Synchronization], [Semaphores, futexes, process-shared mutexes / condvars],
  [Notification], [Signals, `eventfd`, `signalfd`, `pidfd`],
)

The classes are not airtight: a socket carries data but also synchronizes (a blocking `read` is a wait); shared memory carries no events and so must be paired with a synchronization primitive. The art of systems design is choosing the smallest combination that meets the workload's latency and ordering needs.

== Pipes and FIFOs

A pipe is the oldest Unix IPC: a unidirectional kernel-resident byte stream with a fixed-size buffer (64 KB default on Linux, tunable via `F_SETPIPE_SZ`). `pipe(2)` returns two file descriptors — read end and write end — that only related processes can share, because the descriptors propagate across `fork`. A FIFO (named pipe) is the same buffer given a filesystem name via `mkfifo`, so unrelated processes can rendezvous by path.

```c
int fd[2];
pipe(fd);
if (fork() == 0) {            // child reads
    close(fd[1]);
    char buf[256];
    read(fd[0], buf, sizeof buf);
} else {                      // parent writes
    close(fd[0]);
    write(fd[1], "hello", 5);
}
```

Two guarantees matter. First, *atomicity*: a `write` of at most `PIPE_BUF` bytes (POSIX minimum 512, Linux 4096) is delivered without interleaving against other writers — useful when multiple producers share one pipe. Writes larger than `PIPE_BUF` may be split and interleaved. Second, *backpressure*: a writer blocks when the buffer is full and a reader blocks when it is empty, giving flow control for free.

The notorious edge case is *SIGPIPE*. Writing to a pipe whose read end is closed raises `SIGPIPE`, whose default disposition kills the process. Network servers that forget to ignore it (`signal(SIGPIPE, SIG_IGN)` or `MSG_NOSIGNAL`) die when a peer disconnects mid-write.

== Unix Domain Sockets

Unix domain sockets (UDS, address family `AF_UNIX`) are the local sibling of the Berkeley sockets API (see `networking/sockets-api.typ`). They come in `SOCK_STREAM` (reliable, ordered byte stream, connection-oriented) and `SOCK_DGRAM` (reliable, message-framed, no connection) flavours — and locally, unlike UDP, datagrams are reliable and ordered. Because the kernel never serializes for the network, a UDS is roughly twice as fast as TCP over the loopback interface: no checksums, no protocol headers, no Nagle.

Two capabilities make UDS more than a fast pipe:

- *File-descriptor passing.* Using `sendmsg` with an `SCM_RIGHTS` control message, a process can hand a live file descriptor to another process. The kernel installs a new descriptor in the receiver pointing at the same open file object, which is the basis of privilege separation (a sandboxed worker receives an already-opened socket it could never have opened itself).
- *Credential passing.* `SO_PASSCRED` / `SCM_CREDENTIALS` let the kernel attest the sender's PID, UID, and GID, unforgeable by userspace; this is the foundation of `polkit` and D-Bus authorization.

The *abstract namespace* (Linux) binds a UDS to a name beginning with a NUL byte instead of a filesystem path, so the socket has no inode, needs no `unlink` cleanup, and vanishes automatically when the last reference closes.

== System V vs POSIX IPC

Unix carries two parallel families of message queues, semaphores, and shared memory. They differ less in capability than in *naming and lifetime*.

#table(columns: (auto, auto, auto),
  [*Aspect*], [*System V*], [*POSIX*],
  [Identifier], [`key_t` via `ftok`, integer ID], [`/name` string],
  [Create], [`msgget`/`semget`/`shmget`], [`mq_open`/`sem_open`/`shm_open`],
  [Lifetime], [Kernel-persistent until `IPC_RMID`], [Kernel-persistent until `unlink`],
  [Inspection], [`ipcs` / `ipcrm`], [Under `/dev/shm`, `/dev/mqueue`],
  [Descriptor I/O], [No — opaque IDs], [Yes — usable with `poll`],
)

Both families are *kernel-persistent*: an object outlives the process that created it and survives until explicitly removed (`IPC_RMID`, `shm_unlink`, `mq_unlink`) or until reboot. This is the source of the classic leak: a crashed process leaves a System V segment occupying memory with no owning process, visible only via `ipcs`. POSIX IPC at least exposes objects as paths under `/dev/shm` and `/dev/mqueue`, and its descriptors integrate with `poll`/`epoll`, which System V's opaque integer IDs cannot. New code should prefer POSIX (or skip both for `memfd` + sockets); System V survives mainly for compatibility.

== Shared Memory

The lowest-latency IPC is no copy at all: two processes map the same physical pages and read and write them directly. `mmap` with `MAP_SHARED` over a file, a POSIX `shm_open` object, or an anonymous `memfd_create` region all achieve this; `memfd` is attractive because it needs no filesystem name and pairs naturally with `SCM_RIGHTS` to ship the region to a peer.

```c
int fd = memfd_create("ring", MFD_CLOEXEC);
ftruncate(fd, SIZE);
void *p = mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
// send fd to peer over a UDS via SCM_RIGHTS; both mmap the same pages
```

Shared memory carries no synchronization and no notification. The kernel is out of the loop after the mapping is established, so correctness is entirely the programmer's burden:

- *Memory ordering.* On a weakly-ordered architecture (AArch64, POWER), a writer's store to a payload and its store to a "ready" flag can be observed out of order by the reader. Correct code needs explicit acquire/release barriers or C11 atomics; see `cpu-architecture/synchronization.typ`.
- *False sharing.* Two unrelated variables that share a 64-byte cache line ping-pong between cores' caches under write contention, silently destroying throughput. Pad hot fields to a cache line. See `operating-systems/memory-management.typ`.
- *Waking the peer.* Spinning wastes a core; to sleep, pair the region with a futex or an `eventfd`.

== Signals as IPC

A signal is the thinnest possible message: a single integer asynchronously interrupting the target. As an IPC mechanism it is severely limited, with only ~30 standard numbers, no queuing of duplicates, and delivery to an arbitrary thread at an arbitrary instruction.

The deepest hazard is *async-signal-safety*. A handler may interrupt the main flow mid-`malloc`; calling any non-reentrant function (most of libc) from a handler risks deadlock or corruption. The safe set is tiny (`man 7 signal-safety`). The idiomatic dodge is the *self-pipe trick* or its modern form, `signalfd`, which converts signals into readable bytes on a descriptor so they can be handled synchronously in an `epoll` loop instead of an async context.

*Realtime signals* (`SIGRTMIN`..`SIGRTMAX`) improve on the standard ones: they queue (multiple instances are not coalesced), deliver in order, and carry a small payload via `sigqueue`'s `sigval`. They remain a poor general transport but suffice for low-rate notification.

== Cross-Process Synchronization

Two processes sharing memory still need to coordinate access. The mechanism that made this cheap is the *futex* — "fast userspace mutex."

The futex insight: the common case (an uncontended lock) needs no kernel at all. A lock is just an integer in shared memory; acquiring it is an atomic compare-and-swap in userspace. Only when a thread must *block* (the lock was held) or *wake* a waiter does it enter the kernel via the `futex(2)` syscall (`FUTEX_WAIT` / `FUTEX_WAKE`). For the uncontended path the cost is a single atomic instruction, tens of nanoseconds versus a syscall's microsecond.

```c
// uncontended acquire never enters the kernel
if (atomic_compare_exchange_strong(&lock, &expected, LOCKED))
    return;                                  // got it in userspace
syscall(SYS_futex, &lock, FUTEX_WAIT, LOCKED, NULL, NULL, 0);
```

Built atop futexes:

- *Process-shared mutexes / condvars.* A `pthread_mutex_t` placed in `MAP_SHARED` memory and initialized with `PTHREAD_PROCESS_SHARED` synchronizes across processes, not just threads.
- *Robust futexes.* If a process dies holding a normal lock, that lock is wedged forever. A *robust* mutex registers held locks with the kernel; on the holder's death the kernel marks them `OWNERDEAD` so the next acquirer can recover (or declare the protected state inconsistent).
- *Priority inheritance* (`FUTEX_LOCK_PI`) addresses priority inversion for realtime workloads — see `operating-systems/scheduling-theory.typ`.

== Event Notification: eventfd and pidfd

`eventfd` is a kernel-maintained 64-bit counter exposed as a single descriptor. A `write` adds to the counter; a `read` drains it; the descriptor is readable in `epoll` whenever the counter is nonzero. It is the canonical lightweight wakeup to pair with shared memory: producers bump the eventfd, the consumer's event loop wakes without any data copy. In semaphore mode (`EFD_SEMAPHORE`) each `read` decrements by one, turning it into a counting semaphore over a pollable fd.

`pidfd` solves the PID-reuse race (see `operating-systems/processes-and-threads.typ`). A PID is just an integer that the kernel may recycle the instant a process is reaped; code that stores a PID and later signals it can hit an unrelated victim. A `pidfd` is a stable handle to a specific process: `pidfd_send_signal` delivers a signal to *that* process or fails cleanly if it is already gone — no race, no wrong target. A `pidfd` is also `poll`-able for exit, replacing the `SIGCHLD` + `wait` dance with an event-loop-friendly readiness notification.

== A Comparison of Mechanisms

Numbers below are order-of-magnitude on a modern x86-64 machine, round-trip where applicable; treat them as ratios, not promises — they swing with message size, core topology, and mitigation state (KPTI/Spectre raise every syscall-bound row).

#table(columns: (auto, auto, auto, auto, auto),
  [*Mechanism*], [*Latency*], [*Throughput*], [*Payload*], [*Persistence*],
  [Pipe / FIFO], [~3-10 $mu$s], [moderate], [byte stream], [process],
  [Unix domain socket], [~5-15 $mu$s], [high], [stream or datagram], [process / path],
  [Message queue], [~5-15 $mu$s], [moderate], [framed message], [kernel],
  [Shared memory + futex], [~0.1-1 $mu$s], [very high], [raw bytes], [process],
  [Signal], [~2-5 $mu$s], [very low], [int (+`sigval`)], [transient],
  [eventfd], [~2-5 $mu$s], [n/a], [64-bit counter], [process],
)

The pattern is consistent: any mechanism whose fast path stays in userspace (shared memory + futex) beats any that crosses the syscall boundary on every operation by an order of magnitude. The price is that shared memory makes the programmer responsible for framing, ordering, and notification that the kernel-mediated channels provide for free.

== IPC as the Central Primitive: Microkernels

In a monolithic kernel, IPC is one service among many. In a *microkernel*, almost everything (drivers, filesystems, the network stack) is a userspace server reached by IPC, so IPC latency is the system's defining performance metric. This is the lesson of the L4 lineage: Liedtke's 1993 insight was that the original Mach's slow IPC (hundreds of cycles) doomed the microkernel idea. A ruthlessly optimized fast path (registers-only transfer, no scheduler invocation, direct address-space switch) could cut a round trip to a few hundred cycles, making the decomposition viable.

These systems are built on *message passing over capabilities* — an endpoint is an unforgeable handle, and possessing it is the right to send to it:

#table(columns: (auto, 1fr),
  [*System*], [*Model*],
  [Mach ports], [Capability-protected message queues; the basis of macOS/XNU IPC],
  [seL4 endpoints], [Synchronous, capability-gated; the fast path is formally verified],
  [Android Binder], [Object-oriented RPC with reference counting and fd passing],
  [Windows ALPC], [Advanced Local Procedure Call; the RPC substrate beneath Win32],
  [D-Bus], [Userspace message bus over UDS; desktop and systemd service IPC],
)

The synchronous rendezvous favored by L4 and seL4 (sender blocks until receiver takes the message) avoids buffering and lets the kernel hand the CPU *directly* to the receiver — a "direct process switch" that skips the scheduler entirely. Binder and D-Bus instead optimize for the object/RPC ergonomics that application platforms want, trading raw latency for a richer programming model.

== Pitfalls

- *SIGPIPE kills writers.* A peer closing its read end turns your next `write` into a fatal signal. Ignore `SIGPIPE` or use `MSG_NOSIGNAL` / `send` on every socket write.
- *System V IPC leaks survive process exit.* A crashed process leaves segments and semaphore sets allocated; without an `IPC_RMID` they accumulate until reboot. Audit with `ipcs`; prefer POSIX `shm` or `memfd`.
- *Partial reads on stream sockets.* `SOCK_STREAM` and pipes have no message boundaries; one `write` may arrive as several `read`s and vice versa. You must frame the protocol yourself (length prefix or delimiter). Datagram sockets preserve boundaries.
- *Forgetting memory barriers with shared memory.* Publishing a payload then a ready flag without a release barrier lets a reader on a weakly-ordered CPU see the flag before the data. Use C11 atomics, never plain stores, for cross-process flags.
- *Unbounded message queues.* A fast producer against a slow consumer on a queue with no backpressure exhausts kernel memory; size limits exist (`/proc/sys/kernel/msgmax`) precisely to bound this.

== Further Reading

Liedtke, J. (1993). "Improving IPC by Kernel Design." SOSP.

Liedtke, J. (1995). "On Micro-Kernel Construction." SOSP.

Klein, G. et al. (2009). "seL4: Formal Verification of an OS Kernel." SOSP.

Drepper, U. (2011). "Futexes Are Tricky." Red Hat technical report.

Franke, H., Russell, R., Kirkwood, M. (2002). "Fuss, Futexes and Furwocks: Fast Userlevel Locking in Linux." Ottawa Linux Symposium.

Stevens, W. R., Rago, S. (2013). "Advanced Programming in the UNIX Environment" (3rd ed.), Chapters 15-17.

Stevens, W. R., Fenner, B., Rudoff, A. (2003). "UNIX Network Programming, Volume 1: The Sockets Networking API" (3rd ed.).

Kerrisk, M. (2010). "The Linux Programming Interface," Chapters 43-54.

Tanenbaum, A., Bos, H. (2014). "Modern Operating Systems" (4th ed.), Chapter 2.


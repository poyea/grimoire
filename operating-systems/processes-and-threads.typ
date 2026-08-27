#import "../template.typ": xref

= Processes and Threads <processes-and-threads>

The process is the operating system's unit of resource ownership; the thread is its unit of scheduling. Every modern kernel, from monolithic Unix derivatives to microkernels like seL4, draws this distinction in some form, even when the surface vocabulary differs. This chapter treats the abstractions conceptually; Linux's specific implementation lives in `linux-kernel/scheduler.typ` and `linux-kernel/cgroups-namespaces.typ`.

*See also:* #xref("operating-systems", "scheduling-theory", label: "Scheduling Theory"), #xref("operating-systems", "ipc-mechanisms", label: "Inter-Process Communication"), #xref("linux-kernel", "scheduler", label: "The Scheduler") (implementation), #xref("operating-systems", "virtual-memory", label: "Virtual Memory") (architecture).

== The Process Abstraction

A process bundles an address space, a set of open resource handles (file descriptors, sockets, devices), credentials (user/group, capabilities), and one or more threads of control. The minimum kernel-side state for a process is roughly:

#table(columns: (auto, 1fr),
  [*Field*], [*Purpose*],
  [PID / parent PID], [Identity in the process tree],
  [Address space descriptor], [Page tables, VMA list, brk/mmap regions],
  [File descriptor table], [Open file objects, close-on-exec flags],
  [Credentials], [UID/GID, capability sets, security labels],
  [Signal state], [Pending mask, handlers, alternate stack],
  [Resource limits], [`rlimit` style caps on CPU, memory, fds],
  [Accounting], [CPU time, page faults, I/O bytes],
  [Exit status / wait queue], [Reaped by parent via `wait`-family calls],
)

The classical Unix model creates new processes by `fork` (clone the caller) then `exec` (replace the image). `fork` is conceptually expensive (duplicating an address space) but copy-on-write page tables make it cheap in practice; only when a child writes a shared page does the kernel allocate a private copy. Plan 9 and Windows reject `fork` for `spawn`-style primitives (`rfork`, `CreateProcess`) that combine creation with selective inheritance and avoid the COW dance entirely.

```c
pid_t pid = fork();
if (pid == 0) {                    // child
    execve("/bin/ls", argv, envp); // never returns on success
    _exit(127);                    // exec failed
} else if (pid > 0) {              // parent
    int status;
    waitpid(pid, &status, 0);
}
```

The `vfork` / `posix_spawn` variants exist precisely because the COW cost (though sublinear) still touches every page table entry, and for short-lived `exec` callees that work is wasted.

== Threads

A thread is an independently schedulable register context (program counter, stack pointer, general registers, FP state) sharing an address space with its siblings. The choice of where threads live is one of OS design's oldest debates.

=== Kernel Threads (1:1 Model)

*Kernel threads* (Linux NPTL, Windows, modern Solaris) place one kernel scheduling entity per user thread. Blocking syscalls block only the calling thread; the kernel scheduler has full visibility over all threads for load-balancing and priority decisions. The cost is a kernel stack and a `task_struct` per thread, roughly 16 KB on Linux. Under the hood, Linux's `clone(2)` syscall with `CLONE_VM | CLONE_SIGHAND | CLONE_THREAD` creates a thread that shares the address space, signal handlers, and thread group with its parent. The `task_struct` that results differs from a process only in which resources it shares.

=== User Threads (N:1 Model)

*User threads* (early Java green threads, GNU Pth, Lua coroutines) multiplex many user-level execution contexts on a single kernel thread. Context switches are cheap (a `setjmp`/`longjmp` analog, ~50-100 ns) because no kernel transition is needed. However, any blocking syscall stalls all siblings, and SMP parallelism is impossible since only one kernel thread is runnable at a time. This model has largely been abandoned for general concurrency; it survives in cooperative coroutine-style libraries where blocking calls are explicitly avoided.

=== Hybrid / Green Threads (M:N Model)

*M:N threading* (Solaris LWPs, FreeBSD KSE, Go goroutines, Java virtual threads since JDK 21) places a pool of $M$ kernel threads under a user-space scheduler that multiplexes $N >> M$ lightweight execution units across them. This is the most flexible model and the most complex to implement correctly.

The critical challenge is the *scheduler activation problem*: when a goroutine (or virtual thread) makes a blocking syscall, the runtime must detect the block and park the kernel thread so another goroutine can run on a different OS thread. Go's runtime solves this by wrapping every syscall: if a goroutine enters a slow syscall (`read`, `write`, network I/O), the runtime detaches it from its OS thread (called an *M* in Go's $M:P:G$ model) and spins up a fresh thread from a pool to keep the other goroutines running. Java's Project Loom uses OS-level continuations to unmount a virtual thread's stack when it blocks and remount it on a different carrier thread when the operation completes, entirely invisible to the application.

#table(columns: (auto, auto, auto, auto, auto),
  [*Model*], [*Example*], [*Switch cost*], [*Blocking syscall*], [*SMP*],
  [Process], [context switch between two processes], [~5-10 $mu$s], [process only], [native],
  [1:1 kernel thread], [Linux NPTL], [~1-3 $mu$s], [thread only], [native],
  [N:1 user thread], [GNU Pth], [~100 ns], [whole process], [no],
  [M:N green thread], [Go goroutine], [~100-300 ns], [needs handoff], [via M threads],
  [Virtual thread], [Java JDK 21+], [~200 ns], [auto-unmount], [via carrier pool],
)

== Linux task_struct and clone() Flags

In Linux every thread and process is represented by a `task_struct`. The kernel makes no categorical distinction between processes and threads; both are "tasks." What differs is how they share resources, controlled by the flags passed to `clone(2)`:

#table(columns: (auto, 1fr),
  [*Flag*], [*Effect*],
  [`CLONE_VM`], [Share virtual address space (page tables)],
  [`CLONE_FS`], [Share filesystem root, cwd, and umask],
  [`CLONE_FILES`], [Share the file descriptor table],
  [`CLONE_SIGHAND`], [Share signal handler table],
  [`CLONE_THREAD`], [Join the same thread group (same TGID); `getpid()` returns TGID],
  [`CLONE_NEWPID`], [New PID namespace; child appears as PID 1 inside it],
  [`CLONE_NEWNET`], [New network namespace; isolated interfaces and routing tables],
  [`CLONE_SYSVSEM`], [Share System V semaphore undo list],
)

`fork()` calls `clone` with no sharing flags. `pthread_create` calls `clone` with `CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD` (plus a few more). Containers are just tasks with a full set of `CLONE_NEW*` namespace flags. A single mechanism thus covers the full spectrum from heavyweight fork to lightweight thread to isolated container.

=== fork() vs posix_spawn() Performance

`fork` is conceptually a full address-space copy. Copy-on-write page tables make it cheap for small processes (the kernel duplicates page-table entries but not physical pages), but even a COW fork touches every PTE, which for a large process may mean millions of TLB shootdown IPIs across CPUs and substantial time in `dup_mm`. Measurements on a 1 GB heap show `fork` costing 5-15 ms on a NUMA server with many CPUs, while `posix_spawn` (which internally calls `clone` with `CLONE_VFORK` or a similar optimization and immediately `exec`s) costs tens of microseconds because it never duplicates the parent's page tables at all. The practical rule: use `posix_spawn` (or `vfork + exec`) when the goal is to start a child that immediately calls `exec`; reserve `fork` for cases where the child genuinely needs to inspect the parent's address space before exec.

=== PID Namespaces and PID 1

A PID namespace virtualizes the process ID space. Inside a new PID namespace the first process has PID 1, fulfilling the *init* role. If PID 1 exits, the kernel sends `SIGKILL` to every other process in the namespace, making namespace lifetime tied to its init. Container runtimes exploit this: each container's init (`tini`, `s6`, or a language runtime) is PID 1 inside its namespace; killing it cleanly tears down the container.

From outside the namespace, processes have their real (host-namespace) PIDs, and the kernel maintains a mapping. PID namespaces are hierarchical; a process can see PIDs in its namespace and all ancestor namespaces but not in descendants. `pidfd` handles cross namespace boundaries stably; they reference the kernel's `task_struct` directly, making them immune to PID reuse even across namespace boundaries.

== Stacks, TLS, and Context Switching

Each thread owns a stack. Sizing is awkward: too small risks overflow; too large wastes virtual address space when threads are numerous. The standard trick is a *guard page*: one unmapped page below the stack, so overflow traps deterministically as a segfault rather than corrupting a neighbor. Go sidesteps the problem entirely with *segmented stacks* (now replaced by *contiguous stacks* that double in size): goroutine stacks start at 4 KB and grow dynamically, freeing the programmer from stack-size estimation.

=== Thread-Local Storage Layout

Thread-local storage (TLS) provides per-thread globals. The ELF TLS ABI uses a segment register (`%fs` on x86-64, `tpidr_el0` on AArch64) pointing at a per-thread *Thread Control Block* (TCB); static TLS variables live at fixed negative offsets from the TCB pointer, computed at link time. Dynamic TLS (`dlopen`-loaded modules) uses an indirection through a *DTV* (Dynamic Thread Vector), an array of pointers with one entry per module. The first time a dynamic TLS variable is accessed in a thread, the runtime allocates a per-module block and stores its address in the DTV. The cost of a static TLS access is a single segment-relative load; dynamic TLS adds a DTV dereference and may call into the allocator on first access.

A context switch saves the outgoing thread's volatile state to its kernel stack, optionally switches CR3 / TTBR (if the address space changes, i.e., it's a cross-process switch), then restores the incoming thread. Costs:

- Same-process thread switch: ~1 $mu$s (register save/restore, scheduler pick).
- Cross-process switch: add ~200 ns for TLB shootdown amortization, plus the cost of cold instruction/data caches in the new process.

Mitigations like KPTI (after Meltdown) double the TLB cost by maintaining separate user/kernel page tables; this is one reason microkernel IPC was historically slow and why L4-family kernels obsess over the fast path.

== Process Lifecycle and Zombies

A process exits via `_exit` (or a fatal signal); the kernel tears down its address space, closes file descriptors, and transitions the task to *zombie* state, retaining only enough metadata for the parent to call `wait`. A parent that never reaps leaks zombies until PID space is exhausted. The `SIGCHLD` signal notifies the parent; `prctl(PR_SET_CHILD_SUBREAPER)` lets init-like processes adopt orphaned descendants.

Daemonization classically uses a double-fork to detach from the controlling terminal and reparent to init; modern Linux prefers `systemd`-style service supervision (see `operating-systems/boot-and-init.typ`) where the supervisor stays in the foreground and the init system handles backgrounding.

== Cancellation

Killing a thread cleanly is hard. POSIX defines deferred cancellation (cancellation points are syscalls and library functions where the runtime checks a flag), but this leaves resources (mutexes held, fds open) in an undefined state unless every cancellation point is wrapped in a cleanup handler (`pthread_cleanup_push`). Most modern languages reject this model entirely: Go has no goroutine cancellation, only cooperative `context.Context`; Rust's async futures cancel by dropping; structured concurrency (Trio, Kotlin coroutines) makes cancellation a structural property of nested scopes.

The systems lesson: *abrupt* termination of a unit holding shared state is a design error; the OS gives you `SIGKILL` as a last resort, but using it routinely indicates a missing supervision protocol.

== Capability Models and Process Identity

Classical Unix identifies the process by UID/GID (coarse and ambient). Capability-based systems (Mach ports, seL4 endpoints, Capsicum, Fuchsia handles) instead identify it by the set of unforgeable references it possesses; revoking access means revoking the handle. The two models can coexist (Linux capabilities split root into ~40 flags; Linux 4.3+ ambient capabilities propagate across `execve`), but the philosophical gulf is wide. See `operating-systems/security-models.typ`.

== Async-Signal-Safety Constraints

Signal handlers interrupt execution at an arbitrary instruction, including the middle of a `malloc`, a `printf`, or a `lock` operation. A signal handler that calls any non-reentrant function risks deadlock (the handler tries to lock a mutex the interrupted code already holds) or heap corruption (the handler calls `malloc` while the allocator's free-list is half-updated).

The POSIX list of *async-signal-safe* functions is deliberately short (`man 7 signal-safety`). Notably absent: `malloc`/`free`, `printf`/`fprintf`, most `<stdio.h>`, `exit` (use `_exit`), C++ STL, and all threading primitives. Safe alternatives in a handler: write to a pre-allocated pipe or `eventfd`, set a `volatile sig_atomic_t` flag, or use `signalfd` to convert signals to readable fd events handled outside an async context.

```c
static volatile sig_atomic_t g_shutdown = 0;

void handle_sigterm(int sig) {
    g_shutdown = 1;   // safe: sig_atomic_t write
    // do NOT call printf, malloc, pthread_mutex_lock here
}
```

The `signalfd` approach is cleaner: block all signals with `sigprocmask`, then poll the `signalfd` descriptor in the event loop; signals arrive as structured `signalfd_siginfo` structs, handled in a normal synchronous context.

== Pitfalls

- *fork in a multithreaded process* duplicates only the calling thread; mutexes held by other threads remain locked forever in the child. `pthread_atfork` exists to paper over this; the cleaner answer is `posix_spawn`.
- *Signal-safety:* signal handlers may interrupt arbitrary code. The list of async-signal-safe functions is short (`man 7 signal-safety`); `malloc`, `printf`, and most of libc are not on it.
- *PID reuse:* PIDs wrap. Code that stores a PID and later acts on it races with reuse; Linux 5.3+ provides `pidfd` handles that don't.
- *Errno is thread-local* in modern libc but was once global; old code reading `errno` after thread-aware library calls is broken.
- *clone() flag mismatches:* calling `clone` without `CLONE_SIGHAND` but with `CLONE_VM` creates a thread-like task that shares memory but has independent signal handlers; valid for some sandboxing designs but a footgun if accidental.
- *Goroutine leaks:* Go's M:N scheduler never garbage-collects goroutines that are blocked on a channel or waiting for I/O that never arrives; a missing `context.Context` cancellation is the usual cause.

== Further Reading

Ritchie, D., Thompson, K. (1974). "The UNIX Time-Sharing System." CACM.

Anderson, T. et al. (1992). "Scheduler Activations: Effective Kernel Support for the User-Level Management of Parallelism." TOCS.

Pike, R. et al. (1995). "The Use of Name Spaces in Plan 9." Operating Systems Review.

Baumann, A., Appavoo, J., Krieger, O., Roscoe, T. (2019). "A Fork() in the Road." HotOS.

Drepper, U. (2013). "ELF Handling for Thread-Local Storage." Red Hat technical report.

Cox-Buday, K. (2016). "Concurrency in Go." O'Reilly. (covers goroutine scheduler internals)

Goetz, B. et al. (2006). "Java Concurrency in Practice." Addison-Wesley. (Project Loom landed in JDK 21, 2023 — JEP 444)

Kerrisk, M. (2013). "Namespaces in operation." LWN.net series (`https://lwn.net/Articles/531114/`).

Tanenbaum, A., Bos, H. (2014). "Modern Operating Systems" (4th ed.), Chapters 2-3.

Silberschatz, A., Galvin, P., Gagne, G. (2018). "Operating System Concepts" (10th ed.), Chapters 3-4.


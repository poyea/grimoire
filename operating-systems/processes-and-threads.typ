= Processes and Threads

The process is the operating system's unit of resource ownership; the thread is its unit of scheduling. Every modern kernel — from monolithic Unix derivatives to microkernels like seL4 — draws this distinction in some form, even when the surface vocabulary differs. This chapter treats the abstractions conceptually; Linux's specific implementation lives in `linux-kernel/scheduler.typ` and `linux-kernel/cgroups-namespaces.typ`.

*See also:* `operating-systems/scheduling-theory.typ`, `operating-systems/ipc-mechanisms.typ`, `linux-kernel/scheduler.typ`, `cpu-architecture/virtual-memory.typ`.

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

The classical Unix model creates new processes by `fork` (clone the caller) then `exec` (replace the image). `fork` is conceptually expensive — duplicating an address space — but copy-on-write page tables make it cheap in practice; only when a child writes a shared page does the kernel allocate a private copy. Plan 9 and Windows reject `fork` for `spawn`-style primitives (`rfork`, `CreateProcess`) that combine creation with selective inheritance and avoid the COW dance entirely.

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

The `vfork` / `posix_spawn` variants exist precisely because the COW cost — though sublinear — still touches every page table entry, and for short-lived `exec` callees that work is wasted.

== Threads

A thread is an independently schedulable register context (program counter, stack pointer, general registers, FP state) sharing an address space with its siblings. The choice of where threads live is one of OS design's oldest debates.

*Kernel threads* (1:1 model — Linux NPTL, Windows, modern Solaris) put one kernel scheduling entity per user thread. Blocking syscalls block only the calling thread; the kernel scheduler sees everything. The cost is a kernel stack and task struct per thread (~16 KB on Linux).

*User threads* (N:1 — early Java green threads, GNU Pth) multiplex many user threads on a single kernel thread. Context switches are cheap (just a `setjmp`/`longjmp` analog), but any blocking syscall stalls all siblings, and SMP parallelism is impossible.

*Hybrid* (M:N — Solaris LWP, FreeBSD KSE, Go goroutines) place a pool of kernel threads under a user-space scheduler that multiplexes user-level fibers across them. This is the most flexible model and the most complex to implement; getting blocking syscall handoff right (the "schedule activation" problem) is subtle. Go's runtime is the most successful contemporary M:N system; the user-space scheduler hands off to a fresh OS thread when a goroutine enters a blocking syscall.

#table(columns: (auto, auto, auto, auto),
  [*Model*], [*Switch cost*], [*Blocking syscall*], [*SMP*],
  [1:1], [~1-3 $mu$s], [thread only], [native],
  [N:1], [~100 ns], [whole process], [no],
  [M:N], [~100 ns], [needs handoff], [via M threads],
)

== Stacks, TLS, and Context Switching

Each thread owns a stack. Sizing is awkward: too small risks overflow; too large wastes virtual address space when threads are numerous. The standard trick is a *guard page* — one unmapped page below the stack — so overflow traps deterministically as a segfault rather than corrupting a neighbor.

Thread-local storage (TLS) provides per-thread globals. The ELF TLS ABI uses a segment register (`%fs` on x86-64, `tpidr_el0` on AArch64) pointing at a per-thread TCB; static TLS lives at fixed negative offsets from the TCB pointer.

A context switch saves the outgoing thread's volatile state to its kernel stack, optionally switches CR3 / TTBR (if the address space changes — i.e., it's a cross-process switch), then restores the incoming thread. Costs:

- Same-process thread switch: ~1 $mu$s (register save/restore, scheduler pick).
- Cross-process switch: add ~200 ns for TLB shootdown amortization, plus the cost of cold instruction/data caches in the new process.

Mitigations like KPTI (after Meltdown) double the TLB cost by maintaining separate user/kernel page tables; this is one reason microkernel IPC was historically slow and why L4-family kernels obsess over the fast path.

== Process Lifecycle and Zombies

A process exits via `_exit` (or a fatal signal); the kernel tears down its address space, closes file descriptors, and transitions the task to *zombie* state, retaining only enough metadata for the parent to call `wait`. A parent that never reaps leaks zombies until PID exhaustion. The `SIGCHLD` signal notifies the parent; `prctl(PR_SET_CHILD_SUBREAPER)` lets init-like processes adopt orphaned descendants.

Daemonization classically uses a double-fork to detach from the controlling terminal and reparent to init; modern Linux prefers `systemd`-style service supervision (see `operating-systems/boot-and-init.typ`) where the supervisor stays in the foreground and the init system handles backgrounding.

== Cancellation

Killing a thread cleanly is hard. POSIX defines deferred cancellation — cancellation points are syscalls and library functions where the runtime checks a flag — but this leaves resources (mutexes held, fds open) in an undefined state unless every cancellation point is wrapped in a cleanup handler (`pthread_cleanup_push`). Most modern languages reject this model entirely: Go has no goroutine cancellation, only cooperative `context.Context`; Rust's async futures cancel by dropping; structured concurrency (Trio, Kotlin coroutines) makes cancellation a structural property of nested scopes.

The systems lesson: *abrupt* termination of a unit holding shared state is a design error; the OS gives you `SIGKILL` as a last resort, but using it routinely indicates a missing supervision protocol.

== Capability Models and Process Identity

Classical Unix identifies the process by UID/GID — coarse and ambient. Capability-based systems (Mach ports, seL4 endpoints, Capsicum, Fuchsia handles) instead identify it by the set of *unforgeable references* it possesses; revoking access means revoking the handle. The two models can coexist (Linux capabilities split root into ~40 flags; Linux 5.10+ ambient capabilities propagate across `execve`), but the philosophical gulf is wide. See `operating-systems/security-models.typ`.

== Pitfalls

- *fork in a multithreaded process* duplicates only the calling thread; mutexes held by other threads remain locked forever in the child. `pthread_atfork` exists to paper over this; the cleaner answer is `posix_spawn`.
- *Signal-safety:* signal handlers may interrupt arbitrary code. The list of async-signal-safe functions is short (`man 7 signal-safety`); `malloc`, `printf`, and most of libc are not on it.
- *PID reuse:* PIDs wrap. Code that stores a PID and later acts on it races with reuse; Linux 5.3+ provides `pidfd` handles that don't.
- *Errno is thread-local* in modern libc but was once global; old code reading `errno` after thread-aware library calls is broken.

== Further Reading

Ritchie, D., Thompson, K. (1974). "The UNIX Time-Sharing System." CACM.

Anderson, T. et al. (1992). "Scheduler Activations: Effective Kernel Support for the User-Level Management of Parallelism." TOCS.

Pike, R. et al. (1995). "The Use of Name Spaces in Plan 9." Operating Systems Review.

Baumann, A., Appavoo, J., Krieger, O., Roscoe, T. (2019). "A Fork() in the Road." HotOS.

Drepper, U. (2013). "ELF Handling for Thread-Local Storage." Red Hat technical report.

Tanenbaum, A., Bos, H. (2014). "Modern Operating Systems" (4th ed.), Chapters 2-3.

Silberschatz, A., Galvin, P., Gagne, G. (2018). "Operating System Concepts" (10th ed.), Chapters 3-4.

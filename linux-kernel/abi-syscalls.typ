= ABI and Syscalls

Every meaningful interaction between a user-space program and the kernel goes through a syscall. The mechanics — how arguments are passed, how the CPU transitions to ring 0, how the kernel validates pointers, how it returns — are surprisingly subtle, and the cost is non-negligible: a syscall round trip on modern x86-64 is ~100-300 cycles for the privilege transition itself, plus whatever work the handler does. Understanding the path lets you predict performance and exploit shortcuts like the vDSO.

== x86-64 System V ABI

The System V AMD64 ABI defines how user-space C functions pass arguments and return values. It also (with one exception, see syscalls below) defines the convention syscalls use.

*User-space function call:*

#table(columns: (auto, 1fr),
  [*Register*], [*Use*],
  [`rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`], [first 6 integer/pointer arguments],
  [`xmm0`-`xmm7`], [first 8 float/double arguments],
  [`rax`], [integer return value],
  [`xmm0`], [float/double return value],
  [`rbx`, `rbp`, `r12`-`r15`], [callee-saved],
  [`rax`, `rcx`, `rdx`, `rsi`, `rdi`, `r8`-`r11`], [caller-saved],
)

*Stack:* 16-byte aligned at the call site (before `call` is issued, so 16-byte aligned just after `call` would push the return address making `rsp % 16 == 8` on entry). Red zone of 128 bytes below `rsp` is usable by leaf functions without adjusting the stack pointer.

*Variadic functions:* `rax` holds the number of `xmm` registers used. `printf("%d", x)` sets `rax = 0`.

== Syscall Calling Convention

x86-64 Linux uses the `syscall` instruction (introduced with AMD64) rather than the legacy `int 0x80` (32-bit) or `sysenter` (Intel 32-bit). The convention is *almost* the same as the user-space ABI, with one critical difference:

#table(columns: (auto, 1fr),
  [*Register*], [*Use*],
  [`rax`], [syscall number (input), return value (output)],
  [`rdi`, `rsi`, `rdx`, `r10`, `r8`, `r9`], [arguments 1-6],
  [`rcx`], [clobbered (CPU stores return address here)],
  [`r11`], [clobbered (CPU stores RFLAGS here)],
)

Note `r10` instead of `rcx` for the 4th argument — `rcx` is unavailable because the `syscall` instruction itself uses it to save the return RIP. This is why the libc syscall wrappers contain a `mov %rcx, %r10` shuffle.

*Return values:* On success, `rax` holds the result. On error, `rax` is in the range `[-4095, -1]` and represents `-errno`. glibc wraps this:

```c
long ret = syscall(SYS_read, fd, buf, count);
if (ret < 0) {
    errno = -ret;
    ret = -1;
}
```

== A `read()` from User to Kernel and Back

Tracing one call concretely:

```c
ssize_t n = read(fd, buf, 4096);
```

Step by step (modern Linux 6.x on x86-64):

1. *libc wrapper.* `glibc`'s `read` is a thin asm wrapper that loads `__NR_read` (= 0) into `rax`, moves arguments into the syscall registers, and issues `syscall`.
2. *CPU transition.* `syscall` reads `MSR_LSTAR` (kernel entry point), saves RIP to `rcx` and RFLAGS to `r11`, masks RFLAGS via `MSR_SFMASK`, switches to ring 0, and jumps. CPU does *not* automatically switch stacks; the kernel does that explicitly.
3. *Kernel entry stub* (`entry_SYSCALL_64` in `arch/x86/entry/entry_64.S`). Swaps `gs` to per-CPU kernel data via `swapgs`, switches to the kernel stack, saves the user register frame on the kernel stack.
4. *Mitigations.* On vulnerable CPUs the entry path runs IBRS/STIBP/SSBD setup, KPTI page-table swap (Meltdown mitigation), and `RET` retpoline / IBPB barriers. These can add 50-200 cycles depending on which mitigations are active.
5. *Dispatch.* Kernel reads `rax`, range-checks it against `NR_syscalls`, and indirect-calls `sys_call_table[rax]` — which on a syscall-table-protected kernel is itself a thunked call (`__x64_sys_read`).
6. *Handler.* `sys_read` validates `fd`, locates the `struct file*`, copies up to 4096 bytes into `buf` via `copy_to_user` (this is the actual work), and returns the byte count.
7. *Exit stub.* Restores user registers, runs more mitigations (e.g. `verw` for L1TF), executes `sysretq` which flips back to ring 3 with RIP from `rcx` and RFLAGS from `r11`.

A null syscall (`getpid` with vDSO disabled) is a useful microbenchmark: ~120 ns on a Skylake without mitigations, ~250 ns with the full Meltdown/Spectre stack on the same hardware. That overhead is per syscall — not per byte transferred — which is why batching syscalls (`io_uring`, `sendmmsg`, `readv`) is high-leverage for IO-heavy workloads.

== The Syscall Table

Each architecture maintains its own table mapping numbers to handlers:

```
arch/x86/entry/syscalls/syscall_64.tbl   # x86-64
arch/arm64/include/asm/unistd32.h        # arm64 (compat)
include/uapi/asm-generic/unistd.h        # generic baseline
```

A typical row:

```
0   common  read    sys_read
1   common  write   sys_write
2   common  open    sys_open
...
257 common  openat  sys_openat
```

The numbers are stable ABI: once allocated, a syscall number is never reused. This is why `open` (5) and `openat` (257) coexist — `open` cannot be removed without breaking every binary that ever called it.

*Adding a new syscall* (rare, requires LKML acceptance):

1. Reserve a number in `syscall_64.tbl` (and the equivalent table for every architecture).
2. Implement `SYSCALL_DEFINE<n>(name, ...)` in the relevant subsystem.
3. Add the libc wrapper (in glibc, musl, etc.) — but most tools just call `syscall(__NR_foo, ...)` directly until the wrapper lands.

The `SYSCALL_DEFINE` macro auto-generates argument-marshaling stubs (`__x64_sys_foo`) and adds Spectre-v1 hardening on argument loads.

== vDSO: Syscalls Without the Ring Transition

For a small set of read-only operations that the kernel can answer without any privileged work, the kernel maps a small shared object (the *vDSO*, "virtual dynamic shared object") into every process's address space. User code calls into the vDSO like any normal library, no ring transition occurs, and the result is computed entirely in user space against kernel-maintained shared memory.

On x86-64 Linux, the vDSO exposes:

- `clock_gettime` (the big one — used by every monotonic-clock-reading hot path)
- `gettimeofday`
- `time`
- `getcpu`
- `clock_getres`

```
$ ldd /bin/true
        linux-vdso.so.1 (0x00007fff...)
        ...
```

The vDSO maps a kernel-updated page (`vvar`) containing the current time, CLOCK_TAI offset, the TSC's last-read value, and a per-CPU getcpu cache. `clock_gettime(CLOCK_MONOTONIC)` reads the TSC, applies the kernel-supplied scale and offset, and returns. *No syscall is issued.*

Microbenchmark on Skylake at 3.4 GHz:

```
clock_gettime via vDSO:        ~20 ns
clock_gettime via syscall:     ~250 ns (with mitigations)
```

For latency-sensitive tracing or per-request timestamping, vDSO is the difference between "free" and "actively painful." If you ever see `clock_gettime` consuming significant CPU in a profile, the vDSO is broken or disabled — check `vdso=0` is not on the kernel command line.

*Failure modes:* If the kernel cannot guarantee a steady TSC (e.g. unsynchronized across sockets, frequency drift, virtualization without invariant TSC), it falls back to a syscall internally. `cat /sys/devices/system/clocksource/clocksource0/current_clocksource` should be `tsc` on healthy modern hardware.

== seccomp: Filtering Syscalls in BPF

`seccomp` (secure computing mode) lets a process install a BPF program that runs on every syscall and decides whether to allow, kill, log, or trap it. This is one of the most important sandboxing primitives in modern Linux — used by Chrome's renderer, Docker's default profile, systemd's `SystemCallFilter=`, and every modern container runtime.

There are two modes:

- *Strict mode* (`SECCOMP_SET_MODE_STRICT`): only `read`, `write`, `_exit`, and `sigreturn` are allowed. Anything else kills the process. Mostly historical.
- *Filter mode* (`SECCOMP_SET_MODE_FILTER`): install a classic BPF program, evaluated on every syscall. Returns a verdict.

*BPF return verdicts* (high 16 bits):

```
SECCOMP_RET_KILL_PROCESS     SIGSYS the whole process
SECCOMP_RET_KILL_THREAD      SIGSYS just the thread
SECCOMP_RET_TRAP             deliver SIGSYS, debugger can inspect
SECCOMP_RET_ERRNO            return -errno (low 16 bits = errno)
SECCOMP_RET_USER_NOTIF       hand off to a userspace supervisor (5.0+)
SECCOMP_RET_TRACE            ptrace can intercept
SECCOMP_RET_LOG              allow + audit log
SECCOMP_RET_ALLOW            allow
```

*Minimal example: deny `openat` of any path containing forbidden pattern, allow everything else.*

```c
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <sys/prctl.h>

struct sock_filter filter[] = {
    // Load syscall nr from seccomp_data
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
    // If nr == __NR_openat, jump to deny
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_openat, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (EACCES & 0xffff)),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
};

struct sock_fprog prog = {
    .len = sizeof(filter) / sizeof(filter[0]),
    .filter = filter,
};

prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);          // required without CAP_SYS_ADMIN
syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0, &prog);
```

*libseccomp* wraps this in a vastly more pleasant API:

```c
scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_ALLOW);
seccomp_rule_add(ctx, SCMP_ACT_ERRNO(EACCES), SCMP_SYS(openat), 0);
seccomp_load(ctx);
```

*Performance:* seccomp adds 5-50 ns per syscall depending on filter complexity (it's a BPF program executed on the syscall hot path). For tight syscall loops, this matters; for normal applications, it is invisible.

*Caveats and pitfalls:*

- The filter sees the *raw syscall number*, not the libc function name. `open` and `openat` are different syscall numbers; filter both or neither.
- BPF cannot dereference user pointers (no `copy_from_user`). It can compare argument *values* but not the strings they point to. To filter by path, use `SECCOMP_RET_USER_NOTIF` and consult userspace.
- 32-bit compat syscalls (`__X32_SYSCALL_BIT`) have different numbers. Match `arch` from `seccomp_data` first.
- Once installed, a filter cannot be removed. `PR_SET_NO_NEW_PRIVS` is sticky too.

== Practical Notes

- *Counting syscalls:* `perf stat -e 'raw_syscalls:sys_enter' ./prog` gives a system-wide total. `strace -c` gives a per-syscall breakdown for one process.
- *Minimizing syscalls:* `io_uring` (5.1+) batches submission and completion across many operations. `sendmmsg`/`recvmmsg` batch network IO. `readv`/`writev` batch buffer-list IO. `MAP_POPULATE` mmap avoids page-fault syscalls later.
- *vDSO troubleshooting:* If `clock_gettime` is suddenly slow, check the clocksource (`/sys/devices/system/clocksource/clocksource0/current_clocksource`) and `dmesg | grep -i tsc`. A clocksource fallback to `hpet` or `acpi_pm` will tank vDSO performance.

*See also:* _Kernel Bypass (Networking volume)_ — DPDK and AF_XDP avoid the syscall path entirely for packet IO. _kernel-tracing.typ_ — eBPF tracepoints on `raw_syscalls:sys_enter` give per-syscall observability.

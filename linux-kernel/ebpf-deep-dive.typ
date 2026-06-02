= eBPF Deep Dive

eBPF (extended Berkeley Packet Filter) is a sandboxed in-kernel virtual machine that runs verified, JITed userspace-supplied programs at attach points throughout the kernel. It turned the kernel from a recompile-and-reboot artefact into a *programmable platform*: tracing, networking, security, observability, and even schedulers (`sched_ext`) are now things you load at runtime. The cost model — single-digit nanoseconds per instruction after JIT, with verifier-proven safety — is what makes it production-grade where kprobes alone were too dangerous and userspace tracing too slow.

This chapter assumes familiarity with the use-case-specific material in _Kernel Tracing_ (bpftrace one-liners, tracepoints) and _Networking Stack_ (XDP, TC). Here we cover the machinery beneath: the verifier, the JIT, the map menagerie, CO-RE & BTF, the loader libraries, and the modern frontier — kfuncs, sleepable programs, BPF LSM, and `struct_ops`.

== The eBPF VM

The eBPF instruction set has 11 64-bit general registers (`r0`-`r10`, with `r10` the read-only frame pointer), a 512-byte stack, and ~100 instructions covering ALU64/32, jumps, loads/stores, atomics, and function calls.

Programs are loaded via the `bpf(BPF_PROG_LOAD)` syscall, which runs the verifier and (on success) JITs the bytecode to native machine code. There is no interpreter on most production kernels (`bpf_jit_enable=1` is the default).

```c
union bpf_attr attr = {
    .prog_type = BPF_PROG_TYPE_KPROBE,
    .insns     = (__u64)(unsigned long)insns,
    .insn_cnt  = n,
    .license   = (__u64)(unsigned long)"GPL",
    .log_buf   = (__u64)(unsigned long)log,
    .log_size  = sizeof(log),
    .log_level = 1,
};
int prog_fd = syscall(__NR_bpf, BPF_PROG_LOAD, &attr, sizeof(attr));
```

Program *types* (`BPF_PROG_TYPE_*`) constrain what helpers and context the program can access. A `KPROBE` program sees `struct pt_regs *`; an `XDP` program sees `struct xdp_md *`; a `TRACING` (fentry/fexit) program gets typed function arguments via BTF.

== The Verifier

The verifier (`kernel/bpf/verifier.c`, ~25 kLoC) is what makes eBPF safe. It performs symbolic execution of every reachable path through the program, tracking for each register:

- *Type*: `SCALAR_VALUE`, `PTR_TO_CTX`, `PTR_TO_STACK`, `PTR_TO_MAP_VALUE`, `PTR_TO_PACKET`, ...
- *Value range*: `[umin, umax]` and `[smin, smax]` derived from constant propagation and bounded arithmetic.
- *Alignment* and *bounds* for pointers.

Key invariants enforced:

- *No unbounded loops.* Originally none at all; since 5.3 `BPF_FUNC_loop` and bounded for-loops (with verifier-proved bounds) are allowed.
- *Every memory access is bounds-checked.* `pkt->data + offset` is only allowed if the verifier proved `offset + size <= pkt->data_end` on this path.
- *No use-after-free.* References to map values, packet data, and socket pointers have explicit lifetime bounds. `bpf_sk_release` must be called on every path that took a reference.
- *Stack zero-initialization* of slots before read.
- *Helper-function ABI*: each `bpf_func_proto` declares argument types (e.g. `ARG_PTR_TO_MAP_KEY`); arguments are checked at every call site.

The verifier rejects programs whose state graph grows beyond `BPF_COMPLEXITY_LIMIT_STATES` (1M). The classic "verifier rejected" failure mode is a loop the kernel cannot prune.

Practical tip: when verifier output is impenetrable, dump it with `log_level=2`. The trace shows, instruction by instruction, the inferred register state. `bpftool prog tracelog` and `veristat` make this less painful at scale.

== JIT Compilation

After verification, the JIT (`arch/<arch>/net/bpf_jit_comp.c`) emits native code. On x86-64 the typical mapping is:

- eBPF r0-r5 → callee-clobbered host regs (`rax`, `rdi`, `rsi`, `rdx`, `rcx`, `r8`).
- eBPF r6-r9 → callee-saved (`rbx`, `r13`-`r15`).
- eBPF r10 → `rbp`.

ALU operations map 1-1 to host instructions; bounds-checked loads compile to single MOVs. The result is a function pointer the kernel calls like any other — overhead per instruction is essentially that of native code.

The constant blinding pass (`bpf_jit_blinding_enabled`) XORs immediates with a per-load secret to defeat JIT-spraying. Speculative-execution mitigations (`bpf_spec_v1`, `bpf_spec_v4`) insert speculation barriers around array bounds checks.

== Maps

Maps are the only way eBPF programs share state — with each other, with userspace, and across invocations. The map zoo as of 6.x:

#table(columns: (auto, 1fr),
  [`BPF_MAP_TYPE_HASH`], [General-purpose hash table. Up to `max_entries`; per-CPU spinlock on update.],
  [`BPF_MAP_TYPE_ARRAY`], [Fixed-size array indexed by `u32`. Lock-free reads; updates are atomic at element size.],
  [`BPF_MAP_TYPE_PERCPU_HASH` / `PERCPU_ARRAY`], [Per-CPU slot; aggregation is userspace's job. The high-throughput counter pattern.],
  [`BPF_MAP_TYPE_LRU_HASH`], [Bounded-size hash with LRU eviction. Connection-tracking and flow-cache uses.],
  [`BPF_MAP_TYPE_RINGBUF`], [Per-map MPSC ring buffer; replaces `perf_event_array` for event streaming. Reserves space up front, allowing zero-copy `bpf_ringbuf_reserve`/`bpf_ringbuf_submit`.],
  [`BPF_MAP_TYPE_PERF_EVENT_ARRAY`], [Per-CPU `perf` ring buffers; the classic events channel, now mostly superseded by ringbuf.],
  [`BPF_MAP_TYPE_PROG_ARRAY`], [Indirection for `bpf_tail_call` — bounded program-to-program jumps without recursion.],
  [`BPF_MAP_TYPE_HASH_OF_MAPS` / `ARRAY_OF_MAPS`], [Maps of maps; cgroup-scoped or per-namespace policy.],
  [`BPF_MAP_TYPE_SOCKMAP` / `SOCKHASH`], [Tables of `struct sock *` for stream splicing and L7 routing.],
  [`BPF_MAP_TYPE_LPM_TRIE`], [Longest-prefix-match trie — CIDR routing in eBPF.],
  [`BPF_MAP_TYPE_STACK_TRACE`], [Per-id stack traces; the backbone of profiling.],
  [`BPF_MAP_TYPE_BLOOM_FILTER`], [Set membership.],
  [`BPF_MAP_TYPE_CPUMAP` / `DEVMAP` / `XSKMAP`], [Redirect targets for `XDP_REDIRECT`.],
  [`BPF_MAP_TYPE_TASK_STORAGE` / `SK_STORAGE` / `INODE_STORAGE` / `CGRP_STORAGE`], [Object-local storage: a per-task / per-socket / per-inode / per-cgroup slot, freed automatically with the object.],
  [`BPF_MAP_TYPE_QUEUE` / `STACK`], [FIFO / LIFO.],
)

The mental model: pick `PERCPU_*` for hot counters, `RINGBUF` for events, `LRU_HASH` for caches, object-local storage for per-object state.

== Helpers and kfuncs

Programs call kernel functionality through *helpers* (`bpf_helper_defs.h` — a fixed numeric ABI) or, increasingly, *kfuncs* — kernel functions exposed by BTF type names and resolved at load time.

```c
// helpers vs kfuncs
__u64 ts = bpf_ktime_get_ns();              // helper, numbered ID
struct sock *sk = bpf_skc_lookup_tcp(...);  // helper returning ref
bpf_sk_release(sk);                          // must release

// kfunc (looks up by name; requires BTF on both sides)
extern struct task_struct *bpf_task_acquire(struct task_struct *) __ksym;
extern void bpf_task_release(struct task_struct *) __ksym;
```

kfuncs are how new kernel APIs reach eBPF without growing the frozen helper-ID space, and how typed object lifetimes (with verifier-tracked references) become first-class. The "trusted pointers" graph in the verifier rests on kfunc annotations.

== BTF and CO-RE

eBPF programs traditionally needed to be compiled against the exact kernel headers of the host they ran on — a packaging nightmare. *CO-RE* (Compile Once, Run Everywhere) and its enabling technology *BTF* (BPF Type Format) fix this.

*BTF* is a compact debug-format dialect (essentially trimmed DWARF) embedded in the kernel image (`/sys/kernel/btf/vmlinux`) describing every type the kernel exports. The eBPF compiler (clang with `-g`) emits BTF relocations for every field access, recording "I read `task->mm->pgd` — patch this offset for whatever kernel runs me".

```c
// CO-RE field access — note BPF_CORE_READ
#include <bpf/bpf_core_read.h>

struct task_struct *task = (void *)bpf_get_current_task();
pid_t pid = BPF_CORE_READ(task, tgid);              // bounded chain
char comm[16];
BPF_CORE_READ_STR_INTO(&comm, task, comm);
```

libbpf rewrites the offsets at load time using the local BTF. The result: one binary runs on 4.18 RHEL kernels and 6.x mainline alike, without recompilation.

CO-RE also enables *field existence* checks (`bpf_core_field_exists`) and *enum/type variant* handling, letting one program adapt to structural changes between kernel versions.

== libbpf, bpftrace, bcc, Aya

The userspace loader landscape:

- *libbpf* — the canonical C library shipped in-tree. Object lifecycle, CO-RE relocations, skeleton generation (`bpftool gen skeleton`). The right answer for production tools.
- *BCC* — older Python/C++ frontend, ships clang at runtime and compiles per host. Heavy but feature-rich; many existing tools live here.
- *bpftrace* — high-level DSL, awk for the kernel. Built on libbpf. Best for ad-hoc tracing and one-liners; see _Kernel Tracing_.
- *Aya* (Rust), *cilium/ebpf* (Go), *libxdp*, *libbpf-go* — language bindings of varying maturity. The Go and Rust ecosystems are now first-class.

libbpf skeleton workflow:

```sh
clang -O2 -g -target bpf -c prog.bpf.c -o prog.bpf.o
bpftool gen skeleton prog.bpf.o > prog.skel.h
```

```c
// userspace loader
#include "prog.skel.h"
struct prog *skel = prog__open_and_load();
prog__attach(skel);
// poll ringbuf, read maps, ...
prog__destroy(skel);
```

== Tail Calls and BPF-to-BPF Calls

A program can call another loaded program via `bpf_tail_call(ctx, &prog_array_map, index)` — a no-return jump into the program at `prog_array[index]`. Bounded depth (33), no return — useful for state-machine-style demuxing in XDP pipelines.

Separately, *BPF-to-BPF function calls* (since 4.16) let a program contain multiple static functions, JITed as regular call/ret — recursion forbidden, depth bounded. This is what makes large programs (Cilium's are 10,000+ instructions) practical to verify: each function is verified independently.

== Sleepable BPF

Some attach points run in contexts that *can* sleep — LSM hooks, syscall fentry, iter programs. `BPF_F_SLEEPABLE` programs may call helpers that block (`bpf_copy_from_user`, `bpf_d_path`) and use `srcu_*`-protected map ops. The verifier enforces additional rules: no preempt-disabled or rcu_read_lock sections.

This unblocks important patterns: walking user pages (path strings, syscall arguments), interacting with userspace via uprobe-style hooks without copying-and-praying, and integrating with `BPF_PROG_TYPE_LSM` for synchronous policy decisions.

== BPF LSM

`BPF_PROG_TYPE_LSM` programs attach to Linux Security Module hooks (`security_*` functions). They run synchronously with the security decision, get a typed view of the operation, and return a verdict (`0` for allow, `-EPERM` to deny).

```c
SEC("lsm/file_open")
int BPF_PROG(deny_open_etc_shadow, struct file *file)
{
    char path[64];
    bpf_d_path(&file->f_path, path, sizeof(path));
    if (bpf_strncmp(path, sizeof("/etc/shadow") - 1, "/etc/shadow") == 0)
        return -EPERM;
    return 0;
}
```

This is policy-as-code without a kernel module, complementing SELinux/AppArmor rather than replacing them. See _Security Modules_.

== sched_ext: BPF Schedulers

`sched_ext` (mainlined 6.12) lets eBPF programs *implement* a scheduling class — `enqueue`, `dispatch`, `runnable`, `quiescent` callbacks via `struct_ops`. The kernel falls back to CFS if the BPF scheduler is detached or errors. Projects like `scx_rusty`, `scx_lavd`, and `scx_layered` ship policies optimized for gaming, build farms, and tiered workloads.

== Observability of eBPF Itself

```sh
# List programs and maps
bpftool prog show
bpftool map show

# Dump a program's JITed code
bpftool prog dump jited id 42

# Pretty-print map contents
bpftool map dump id 7

# Profile what an XDP program spends time in
perf record -e bpf-output,cycles -ag -- sleep 5
perf report
```

`bpftop` (released 2024) is `htop` for eBPF programs — per-program cycles, run count, average runtime.

== Performance Numbers

- *Helper call overhead*: ~10-25 ns (similar to a regular function call after JIT).
- *Per-CPU counter increment*: ~3 ns (one MOV after JIT).
- *Map lookup* (hash, 10k entries): ~50-80 ns.
- *Ringbuf submit* (uncontended): ~30-50 ns.
- *XDP program dispatch*: ~30 ns including the indirect call.
- *fentry vs kprobe*: fentry is ~2× faster (~50 ns vs ~120 ns) because it uses the ftrace direct-call trampoline instead of an INT3.

== Security Posture

eBPF has had its share of CVEs — almost all in the verifier. Mitigations now in place:

- *Unprivileged eBPF* disabled by default (`kernel.unprivileged_bpf_disabled = 1`).
- *Spectre v1/v4* speculative-bounds checks in the verifier.
- *Constant blinding* of immediates.
- *JIT randomization* (`bpf_jit_harden`).
- *Capability split*: `CAP_BPF` separates from `CAP_SYS_ADMIN`; combined with `CAP_PERFMON`, `CAP_NET_ADMIN` it lets you grant exactly the eBPF surface a tool needs.

In hardened deployments: keep unprivileged eBPF off, gate access via cgroup `bpf_program` controls (`BPF_PROG_TYPE_CGROUP_*`), and audit loaded programs via `bpftool prog`.

== Further Reading

Kernel docs: `Documentation/bpf/` — especially `verifier.rst`, `bpf_design_QA.rst`, `btf.rst`, `kfuncs.rst`, `prog_sk_lookup.rst`.

Gregg, B. (2019). _BPF Performance Tools_. Addison-Wesley.

Calavera, D. and Fontana, L. (2023). _Learning eBPF_. O'Reilly.

Starovoitov, A. (2018). _BPF as a fundamentally better dataplane_. NetDevConf.

Fleming, M. (2020). _Bringing BPF to your CI_. KernelCon.

LWN: Corbet's BPF series (verifier evolution, BTF, CO-RE, sleepable, kfuncs).

`kernel/bpf/`, especially `verifier.c`, `core.c`, `helpers.c`, `ringbuf.c`, `btf.c`, `arraymap.c`, `hashtab.c`.

ebpf.io — the community portal, with curated tool index.

*See also:* _Kernel Tracing_ (tracepoints, kprobes, perf — the trace-side attach points), _Networking Stack_ (XDP, TC, sockmap — the network-side attach points), _Security Modules_ (BPF LSM in context), _Scheduler_ (`sched_ext` BPF schedulers).

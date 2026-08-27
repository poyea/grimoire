#import "../template.typ": xref

= Speculative Execution Security <speculative-execution-security>

Speculative execution, the engine of modern $"IPC"$, leaks data through microarchitectural side channels. Since 2018 a steady stream of disclosures has forced a complete rethink of the boundary between architectural and microarchitectural state.

*See also:* #xref("cpu-architecture", "branch-prediction", label: "Branch Prediction"), #xref("cpu-architecture", "out-of-order-deep-dive", label: "Out-of-Order Execution Deep Dive"), #xref("cpu-architecture", "cache-hierarchy", label: "Cache Hierarchy"), #xref("cpu-architecture", "virtual-memory", label: "Virtual Memory")

== Threat Model

Speculative execution attacks combine three ingredients:

1. *A speculation window:* mispredicted branch, faulting load, or transient bypass of an architectural check.
2. *A secret-dependent operation* executed transiently — typically a load whose address depends on a secret byte.
3. *A microarchitectural side channel* (commonly Flush+Reload on a cache line) that survives the rollback and reveals which transient path was taken.

Architecturally nothing changed: the transient instructions are squashed and never retire. Microarchitecturally, the cache footprint persists, and the attacker reads the secret one bit at a time.

== Spectre Family

=== Spectre v1 — Bounds Check Bypass (CVE-2017-5753)

The classic gadget [Kocher et al. 2019]:

```c
if (x < array1_size) {
    y = array2[array1[x] * 4096];
}
```

After training the conditional with many in-bounds $x$, the attacker sends an out-of-bounds $x$. The branch predictor predicts taken, the transient load reads `array1[x]` (out of bounds), and the dependent load brings a cache line of `array2` into L1 whose index encodes the secret byte. After squash, the attacker times accesses to `array2` to recover the byte.

*Mitigations:*
- `LFENCE` after the bounds check serializes speculation. Cost: 1-10 cycles per fence.
- Speculative load hardening ($"SLH"$): mask the index with a predicate derived from the branch condition. Clang `-mspeculative-load-hardening`. 10-30% perf loss.
- The Linux kernel uses `array_index_nospec()` macro that compiles to a mask + `cmov`.

=== Spectre v2 — Branch Target Injection (CVE-2017-5715)

The attacker poisons the indirect branch predictor ($"IBP"$) so that a victim's indirect branch transiently jumps to an attacker-chosen gadget in the victim's address space. Cross-process, cross-privilege.

*Mitigations:*

#table(
  columns: 3,
  [*Mitigation*], [*Mechanism*], [*Cost*],
  [Retpoline], [Replace indirect branch with `ret` trampoline], [5-30% on indirect-heavy code],
  [$"IBRS"$ (legacy)], [$"MSR"$ write on every privilege transition flushes $"IBP"$], [20-50%],
  [eIBRS (enhanced)], [Hardware tags $"IBP"$ entries by privilege; one-time enable], [2-5%],
  [$"STIBP"$], [Prevents sibling-thread cross-pollution on $"SMT"$], [5-15% with $"SMT"$ on],
  [$"IBPB"$], [Full $"IBP"$ flush on context switch], [Adds ~1-5$mu$s per switch],
)

Retpoline became obsolete with Retbleed (below) on pre-eIBRS Intel and Zen 1/2.

=== Spectre v4 — Speculative Store Bypass (CVE-2018-3639)

Memory disambiguation predicts that a load does not alias an older unretired store. If the prediction is wrong and the store would have written a secret, the load transiently reads stale data and leaks it.

*Mitigation:* $"SSBD"$ ($"MSR"$ bit) disables the bypass predictor. 2-8% perf hit. Linux exposes via `prctl(PR_SET_SPECULATION_CTRL)` for per-process opt-in.

== Meltdown (CVE-2017-5754)

Meltdown [Lipp et al. 2018] is a *transient execution* attack on user/kernel isolation that uniquely affects Intel (and a few ARM cores). A user-mode load from a kernel address takes a permission fault — but Intel's pipeline forwards the loaded data to dependent transient instructions *before* the fault is raised. The dependent transient load encodes the kernel byte into the cache.

*Mitigation:* Kernel page-table isolation ($"KPTI"$, originally $"KAISER"$). Unmaps kernel from the user page tables; syscalls and interrupts swap CR3. Cost: $"TLB"$ flush per privilege transition, 5-30% on syscall-heavy workloads (database, web servers). Mostly avoided with $"PCID"$ tagging.

AMD CPUs check permissions before forwarding and are not affected.

== Microarchitectural Data Sampling (MDS)

A family disclosed May 2019. Instead of crossing privilege architecturally, $"MDS"$ leaks data sitting in microarchitectural buffers (line fill buffer, store buffer, load port) belonging to any context that recently used them.

#table(
  columns: 3,
  [*Variant*], [*Buffer*], [*CVE*],
  [$"RIDL"$ ($"MLPDS"$)], [Load ports], [CVE-2018-12127],
  [Fallout ($"MSBDS"$)], [Store buffer], [CVE-2018-12126],
  [ZombieLoad ($"MFBDS"$)], [Line fill buffer], [CVE-2018-12130],
  [$"MDSUM"$], [Uncacheable memory variant], [CVE-2019-11091],
  [$"TAA"$], [TSX async abort variant], [CVE-2019-11135],
  [L1DES (CacheOut)], [Eviction into fill buffer], [CVE-2020-0549],
  [SRBDS], [Special register bus], [CVE-2020-0543],
)

*Mitigation:* `VERW` instruction (re-purposed) flushes affected buffers on every privilege transition. Combined with $"SMT"$ being disabled or co-scheduled in security domains. Cost: 5-15% typical, up to 40% on $"SMT"$-disabled DB workloads.

== L1TF — Foreshadow (CVE-2018-3615/3620/3646)

L1 Terminal Fault: a faulting page-table entry's *physical address* is still used to issue the load against L1; if attacker-controlled, an attacker can read any L1 line, including from other VMs or $"SGX"$ enclaves.

*Mitigation:* invert unused $"PTE"$ bits so cleared $"PTE"$s point to non-cached physical addresses; flush L1 on $"VM"$ entry. Disable $"SMT"$ for $"SGX"$/hypervisor isolation.

== Retbleed (CVE-2022-29900/29901)

`ret` instructions, the foundation of retpoline, are themselves predicted — by the return stack buffer ($"RSB"$) and, when that underflows, by the same indirect predictor retpoline was supposed to avoid. On Intel Skylake-era and AMD Zen 1/2, this defeats retpoline.

*Mitigation:* on Intel, eIBRS where available (Cascade Lake / Ice Lake and later); Skylake-era parts lack it and fall back to full $"IBRS"$. On AMD pre-Zen 3, "untrained $"RET"$" software sequence ($"jmp2ret"$) plus $"IBPB"$. Linux kernel `retbleed=auto` selects per CPU.

== Downfall (CVE-2022-40982) and Inception/SLS

*Downfall* (Intel, 2023): `GATHER` instruction transiently leaks data from the vector register file across $"SMT"$ siblings on Skylake through Tiger Lake. Microcode mitigation disables gather speculation; cost up to 50% on $"AVX"$-heavy code that uses gather.

*Inception / Speculative Return Stack Overflow* ($"SRSO"$, CVE-2023-20569, AMD Zen 1-4, 2023): trains the $"RSB"$ across context switches; fixed via microcode + new `lfence;jmp` sequences and an "$"SBPB"$" branch predictor barrier.

*Straight-Line Speculation* ($"SLS"$): after an unconditional branch the CPU speculatively fetches and executes following bytes. ARM mitigation: insert `dsb sy; isb` or compiler `-mharden-sls=all`.

== Side Channels Beyond Cache

- *Port contention* ($"PortSmash"$): two $"SMT"$ threads contend for execution ports; secret-dependent timing leaks.
- *$"TLB"$ side channels:* fine-grained $"TLB"$ eviction reveals address access patterns.
- *Power side channels* ($"PLATYPUS"$): $"RAPL"$ energy readings correlate with secret data; mitigation requires root for $"RAPL"$.
- *Frequency side channels* ($"Hertzbleed"$, 2022): $"DVFS"$ throttling depends on workload; remote timing leaks crypto keys.

== Performance Cost Summary

#table(
  columns: 4,
  [*Mitigation*], [*Workload class*], [*Typical cost*], [*Worst case*],
  [$"KPTI"$ + $"PCID"$], [Syscall-heavy], [2-5%], [30% (Redis)],
  [Retpoline], [Indirect-call-heavy], [5-10%], [25%],
  [eIBRS], [General], [2-5%], [10%],
  [$"SSBD"$], [General], [2-8%], [15%],
  [$"MDS"$ + $"VERW"$], [Privilege transitions], [5-10%], [40% ($"SMT"$ off)],
  [$"SMT"$ disabled], [Throughput], [15-30%], [50%],
  [All-of-the-above], [Cloud guest], [10-25%], [50%+],
)

Cloud providers default to "all mitigations on"; HPC sites often disable selectively after threat-modeling.

== Defensive Coding

```c
// Linux kernel pattern
static inline unsigned long array_index_nospec(unsigned long idx,
                                               unsigned long sz)
{
    unsigned long mask = ~(idx < sz ? 0UL : -1UL);
    OPTIMIZER_HIDE_VAR(mask);
    return idx & mask;
}

// Clang: speculative load hardening
// $ clang -mspeculative-load-hardening foo.c

// GCC: -mindirect-branch=thunk  (retpoline)
//      -mfunction-return=thunk
```

For crypto: constant-time code, no secret-dependent branches *or* memory addresses. Libraries like libsodium and BearSSL document their constant-time guarantees.

== Detection

```
# Linux: which mitigations are active?
ls /sys/devices/system/cpu/vulnerabilities/
cat /sys/devices/system/cpu/vulnerabilities/spectre_v2
# -> "Mitigation: Enhanced IBRS, IBPB conditional, RSB filling"
```

== Further Reading

Kocher, P. et al. (2019). "Spectre Attacks: Exploiting Speculative Execution." _IEEE S&P '19_.

Lipp, M. et al. (2018). "Meltdown: Reading Kernel Memory from User Space." _USENIX Security '18_.

Van Schaik, S. et al. (2019). "RIDL: Rogue In-Flight Data Load." _IEEE S&P '19_.

Schwarz, M. et al. (2019). "ZombieLoad: Cross-Privilege-Boundary Data Sampling." _ACM CCS '19_.

Wikner, J. & Razavi, K. (2022). "Retbleed: Arbitrary Speculative Code Execution with Return Instructions." _USENIX Security '22_.

Moghimi, D. (2023). "Downfall: Exploiting Speculative Data Gathering." _USENIX Security '23_.

Wang, Y. et al. (2022). "Hertzbleed: Turning Power Side-Channel Attacks Into Remote Timing Attacks on x86." _USENIX Security '22_.

Canella, C. et al. (2019). "A Systematic Evaluation of Transient Execution Attacks and Defenses." _USENIX Security '19_.

Intel Corporation (2024). _Affected Processors: Transient Execution Attacks & Related Security Issues by CPU_.

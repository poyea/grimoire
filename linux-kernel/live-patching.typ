#import "../template.typ": xref

= Live Patching

Live patching applies critical kernel fixes (typically security CVEs) to a *running* kernel without reboot. The unit is a kernel module that redirects calls to a buggy function to a replacement implementation. Done right, the patched kernel is indistinguishable in behaviour from one booted with the fix; done wrong, you have racy in-flight callers running half-old half-new code. The Linux *livepatch* infrastructure (`kernel/livepatch/`, mainlined 4.0) and the userspace toolchains around it (kpatch from Red Hat, kGraft from SUSE now merged into livepatch, and Ksplice from Oracle out-of-tree) provide the engineering rigour.

This is fleet-scale plumbing. A cloud operator with 100k hosts cannot reboot all of them for every CVE; livepatch lets them roll out a fix in minutes. The trade-off is severe constraints on what can be patched: bugs in interrupt handlers, scheduler internals, semantic changes to data layouts: all generally off-limits.

== The Problem in One Picture

You discover `foo(x)` has a bug. You want every future call to invoke `foo_v2(x)` instead. Two things must be true at switchover:

1. *Atomically*, future callers of `foo` invoke `foo_v2`.
2. *No in-flight call to the old `foo`* remains on any CPU's stack (otherwise you'd half-execute the patch).

(1) is solved by ftrace's function-trampoline mechanism. (2) is solved by the *consistency model*: a per-task migration discipline that waits for every thread to leave any patched function before declaring the patch complete.

== ftrace Hooks: The Substrate

Modern kernels are compiled with `-pg` (or `-fpatchable-function-entry=N,M` on newer GCC/Clang), which leaves a 5-byte NOP at the entry of every non-inline function. ftrace can swap this NOP for a CALL to a trampoline, the same mechanism that powers function tracing (`/sys/kernel/debug/tracing/set_ftrace_filter`), kprobes-on-ftrace, and now livepatch.

```
Original:           foo:
  nop5                ...

After ftrace hook:  foo:
  call ftrace_call    ; jumps to trampoline
                      ; trampoline saves regs, calls handlers
                      ; for livepatch, redirects RIP to foo_v2
```

The kernel uses *text_poke* (`arch/x86/kernel/alternative.c`) to swap the bytes atomically across CPUs while threads execute the function, relying on x86's `int3` short-circuit dance to make the patch visible without stopping the world.

== klp_func and klp_object

A livepatch module declares the functions it patches via `struct klp_patch` containing arrays of `struct klp_object` (per ELF module patched) and `struct klp_func` (per function within an object):

```c
#include <linux/livepatch.h>

static int livepatch_cmdline_proc_show(struct seq_file *m, void *v)
{
    seq_printf(m, "%s\n", "this is the patched cmdline");
    return 0;
}

static struct klp_func funcs[] = {
    {
        .old_name = "cmdline_proc_show",
        .new_func = livepatch_cmdline_proc_show,
    }, { }
};

static struct klp_object objs[] = {
    {
        /* .name = NULL ⇒ vmlinux */
        .funcs = funcs,
    }, { }
};

static struct klp_patch patch = {
    .mod = THIS_MODULE,
    .objs = objs,
};

static int livepatch_init(void) { return klp_enable_patch(&patch); }
static void livepatch_exit(void) { /* atomic-replace handles teardown */ }

module_init(livepatch_init);
module_exit(livepatch_exit);
MODULE_LICENSE("GPL");
MODULE_INFO(livepatch, "Y");
```

`klp_enable_patch` registers the replacements with ftrace and starts the consistency-model transition. Unloading the module is gated by per-patch refcounts.

== The Consistency Model

Naive function replacement is unsafe: an in-flight invocation of the old function might run code that assumes invariants the patch changes. Consider a patch that adds locking to a previously lockless function: the half-old, half-new race is exactly the bug you were trying to fix, reborn.

Livepatch's *hybrid consistency model* (Josh Poimboeuf, merged 4.12) draws from kpatch (per-function switch on every call) and kGraft (per-task switch when the task is safe):

- Each task is in *universe 0* (old code) or *universe 1* (patched code).
- Newly created tasks join universe 1.
- A task migrates from 0 → 1 when:
  - It is sleeping and a stack walk shows *no patched function on its stack* (i.e. it cannot resume into old code).
  - Or it crosses a userspace boundary (syscall entry/exit); by definition it is not inside any kernel function.
  - Or, for idle tasks, when they enter idle.

Until every task has migrated, both universes coexist. ftrace trampolines consult the current task's universe at call time and dispatch accordingly. When all tasks have moved, the patch is "complete" and the trampoline shortens to call only the new function.

```c
// kernel/livepatch/transition.c (the heart of it)
static void klp_check_stack(struct task_struct *task, ...)
{
    save_stack_trace_tsk_reliable(task, &trace);
    for each frame in trace:
        if function_is_being_patched(frame.ip):
            return -EAGAIN;   // can't migrate yet
    set_tsk_thread_flag(task, TIF_PATCH_PENDING_CLEAR);
}
```

The stack walk must be *reliable*: it must either return a complete trace or admit failure. Architectures need `HAVE_RELIABLE_STACKTRACE` (x86_64, arm64, s390, powerpc); most others can't livepatch with the consistency model.

Stuck tasks (a long-sleeping syscall holding a patched function) block completion. `/sys/kernel/livepatch/<patch>/transition` shows status. The operator can `kill -SIGSTOP/-SIGCONT` to nudge problem tasks, or simply wait.

== Atomic Replace

`KLP_REPLACE` (`patch.replace = true`) is the modern default: enabling a new patch atomically replaces every prior livepatch. This makes cumulative patches the unit of deployment. Fix 5 CVEs by loading one module that subsumes all four prior ones; disable-and-reload races vanish.

== kpatch-build: From Source Diff to Module

Hand-authoring `klp_func` arrays is error-prone. `kpatch-build` (Red Hat's tool) takes a patch as a regular kernel source diff:

```
--- a/fs/proc/cmdline.c
+++ b/fs/proc/cmdline.c
@@ -7,7 +7,7 @@
 static int cmdline_proc_show(struct seq_file *m, void *v)
 {
-       seq_puts(m, saved_command_line);
+       seq_puts(m, "patched");
        seq_putc(m, '\n');
        return 0;
 }
```

It builds the kernel twice (vanilla and patched), diffs the resulting `.o` files at the symbol level, generates a `.c` file with the changed functions plus the livepatch metadata, and compiles a livepatch module. The build harness also detects:

- New symbol references that need to be resolved against vmlinux/module symbols via `klp_symbols`.
- Static data changes (which generally cannot be live-patched; different `.o` for the same `.c` is a red flag).
- Init/exit code, inline functions affecting multiple call sites.

```bash
kpatch-build --sourcedir /lib/modules/$(uname -r)/build cve-2024-XXXX.patch
sudo kpatch load kpatch-cve-2024-XXXX.ko
sudo kpatch list
```

SUSE's `klp-build` tool is the analogue for kGraft/livepatch in the SLES ecosystem.

== What Livepatch *Cannot* Do

Hard limits, born from the consistency model and ftrace mechanics:

- *Data structure layout changes.* You can't add a field to `struct task_struct` and expect old code to use the new layout. The workaround is to side-table the new field in a separately allocated map keyed by task pointer.

- *Semantics of inline functions.* Inlined into N callers; livepatch can only redirect N call sites. `kpatch-build` either declines or generates per-caller patches.

- *Code in ftrace itself, or in NMI handlers.* The trampoline mechanism cannot recursively patch its own machinery; NMI cannot be safely intercepted by call indirection.

- *Init code* (`__init`): already freed.

- *Assembly entry/exit code* without an ftrace hook (syscall entry, IRQ stubs).

- *Changing module initialization*.

For these cases the answer is to schedule a reboot. Live patching is a tool to buy time, not a substitute for reboots.

== Shadow Variables

For the "I need to add a field" pattern, livepatch provides *shadow variables*: a hashtable keyed by `(object_pointer, id)` that maps to a small extension blob.

```c
int *shadow_data = klp_shadow_get_or_alloc(task, SHADOW_DATA_ID,
                                            sizeof(int), GFP_KERNEL,
                                            shadow_data_ctor, NULL);
```

Old code is unaware of the shadow; patched functions check for it on entry and supply default behaviour if absent (necessary during the transition window when both universes coexist). On patch removal, the framework frees each shadow entry via a destructor callback.

== Pre/Post Callbacks

A patch can register `pre_patch`/`post_patch` callbacks invoked when an *object* (module) transitions. Used to initialize new state (e.g., register a new sysctl), perform cleanup of pre-existing state, or run sanity checks. Symmetric `pre_unpatch`/`post_unpatch` run on disable.

== Performance Impact

A patched function pays the ftrace trampoline cost on every call: ~30-50 ns on x86-64 plus the call to the new function. After all tasks have migrated and the kernel switches the trampoline to a direct call into the new function (the "fast path"), overhead drops to a single indirect call (~5-10 ns).

A complete livepatch with hundreds of replaced functions can shave a percent or two off throughput while transition is in progress and converge to within noise after completion. Hot-path functions (scheduler internals, packet receive) are usually *not* patched live for this reason.

== Atomic Replace and Cumulative Patches

The recommended deployment model: every published livepatch is *cumulative*: it carries every prior fix plus the new one, with `replace=true`. Operations become:

- Apply latest livepatch module → previous one auto-disabled.
- Disable everything → load the empty (`replace=true`, no funcs) revert module.

No state machine of "which sequence of patches did this host receive" to maintain; the host either has the latest cumulative module or it doesn't.

== Distributions and Vendors

- *kpatch* — Red Hat / Fedora; `dnf install kpatch-patch-<kernel-version>`.
- *kGraft / live patching* — SUSE; `zypper patch`.
- *Ubuntu Livepatch (Canonical)* — Ubuntu Pro subscription, separate kpatch-format modules.
- *Ksplice* — Oracle; the original and still out-of-tree; the only one supporting *data* patches via clever code generation.
- *Kpatch-cloud* — Azure, GCP, AWS-hosted services that ship custom livepatches for managed kernels.

All ship security CVEs as livepatches within hours-to-days of CVE disclosure for supported kernels.

== Observability

```bash
# Loaded patches
ls /sys/kernel/livepatch/
cat /sys/kernel/livepatch/<patch>/enabled
cat /sys/kernel/livepatch/<patch>/transition

# Per-function status
ls /sys/kernel/livepatch/<patch>/<object>/

# Which tasks are stuck on transition?
grep -L 0 /proc/*/patch_state

# ftrace's view
cat /sys/kernel/debug/tracing/enabled_functions | head
```

When a transition is stuck, `/proc/PID/patch_state` shows `0` (universe 0, not migrated) for the laggards. A value of `0` for kernel threads usually means an idle worker; sending it any signal (or its workqueue any work) lets it complete its current iteration and pass through the migration point.

== Building One End to End

A reproducible recipe for fixing a hypothetical bug in `fs/proc/cmdline.c`:

```bash
# 1. Clone matching kernel source
git clone --depth 1 --branch v6.6 \
    https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git
cd linux

# 2. Apply the source patch
git apply ../fix.patch

# 3. Build a livepatch module
kpatch-build -t vmlinux -s . ../fix.patch
# → produces livepatch-fix.ko

# 4. Load
sudo insmod livepatch-fix.ko

# 5. Verify
cat /sys/kernel/livepatch/livepatch_fix/enabled    # 1
cat /sys/kernel/livepatch/livepatch_fix/transition # 0 after completion
cat /proc/cmdline                                   # observes patched output
```

== Pitfalls

- *Forgetting `MODULE_INFO(livepatch, "Y")`* — the kernel refuses to enable.
- *Patching across modules* — the patched module must be loaded; `klp_object.name` must match the module name; transition stalls until the module is loaded.
- *Symbols only present in some configs* — `klp_resolve_symbols` is per-build.
- *Patching `__init` functions* — they are freed; cannot be intercepted. Audit the patch source for `__init` annotations.
- *Reboot still required for kernel-version changes* — livepatches are kernel-version-specific; the next kernel update reboots anyway.

== Further Reading

Kernel docs: `Documentation/livepatch/` (especially `livepatch.rst`, `cumulative-patches.rst`, `module-elf-format.rst`, `shadow-vars.rst`, `system-state.rst`).

Poimboeuf, J. (2017). _Living with the kernel live-patching_, LinuxCon NA.

Pavlík, V. (2014). _kGraft: live patching of the Linux kernel_, SUSE.

Arnautov, J. and Sosnowski, P. (2014). _kpatch: have your security and eat it too_, LinuxCon.

Arnold, J. and Kaashoek, F. (2009). _Ksplice: Automatic Rebootless Kernel Updates_, EuroSys.

LWN: Corbet's livepatch coverage (2014-2024) and Edge's "Anatomy of a livepatch" series.

kpatch source: #link("https://github.com/dynup/kpatch")[github.com/dynup/kpatch].

`kernel/livepatch/`, `arch/x86/kernel/ftrace.c`, `arch/x86/kernel/alternative.c`.

*See also:* #xref("linux-kernel", "kernel-modules", label: "Kernel Modules") (livepatches are a special module flavour), #xref("linux-kernel", "kernel-tracing", label: "Kernel Tracing") (ftrace, the trampoline substrate), #xref("linux-kernel", "rcu-and-locking", label: "RCU and Locking") (consistency-model invariants resemble RCU grace periods), #xref("linux-kernel", "security-modules", label: "Security Modules") (livepatching is how production fleets respond to LSM/syscall CVEs without reboots).

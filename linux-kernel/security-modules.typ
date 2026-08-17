#import "../template.typ": xref

= Security Modules

Linux's DAC (discretionary access control: uid/gid + mode bits) is necessary but insufficient: a compromised process running as `root` ignores it, a vulnerable service running as a user can still touch everything that user owns, and there is no fine-grained mediation of network, IPC, or capability operations. The *Linux Security Module* (LSM) framework, added in 2.6 to host SELinux, is the kernel's hook architecture for stacking *mandatory* access control on top of DAC. Today it hosts SELinux, AppArmor, Landlock, SMACK, TOMOYO, Yama, the integrity modules (IMA/EVM), and the modern programmable plug-in *BPF LSM*.

This chapter walks the LSM hook architecture, then surveys the major modules and their use models, and ends with seccomp-bpf, the syscall-level filter that is technically not an LSM but lives in the same security mental model.

== The LSM Hook Architecture

LSM hooks (`include/linux/lsm_hook_defs.h`, ~250 hooks as of 6.x) are call-out points scattered through every security-relevant kernel path: `file_open`, `inode_permission`, `bprm_check_security`, `socket_connect`, `task_kill`, `ptrace_access_check`, ...

Each enabled LSM registers callbacks for the hooks it cares about. The framework calls them in registration order; if any returns a non-zero (negative errno) verdict, the operation is denied. Hooks are designed to *short-circuit on deny*; a module that doesn't care simply returns 0.

```c
// kernel/cred.c (excerpt)
int task_kill(struct task_struct *p, struct kernel_siginfo *info,
              int sig, const struct cred *cred)
{
    return call_int_hook(task_kill, 0, p, info, sig, cred);
}
```

LSM stacking (multiple modules active simultaneously) was a long-running project; today the major modules (SELinux + AppArmor + BPF LSM + Landlock + Yama) can all be loaded together via `CONFIG_LSM=` and the `lsm=` boot parameter.

== SELinux

SELinux (NSA-origin, mainlined 2.6) is the most comprehensive MAC implementation in Linux. It labels every subject (process) and object (file, socket, IPC) with a *security context* (`user:role:type:level`) and decides access based on a policy that allows specific `type` pairs to perform specific permissions.

```
unconfined_u:unconfined_r:unconfined_t:s0
system_u:object_r:httpd_sys_content_t:s0
```

Policy fragments look like:

```
allow httpd_t httpd_sys_content_t : file { read getattr };
allow httpd_t http_port_t        : tcp_socket name_bind;
type_transition httpd_t tmp_t : file httpd_tmp_t;
```

The reference policy (used by RHEL/Fedora) ships ~10k types and ~100k rules, providing exhaustive coverage of every system daemon. Per-distribution tooling (`audit2allow`, `semanage`, `seinfo`) makes editing manageable. Modes:

- *Enforcing*: deny on violation; write AVC denial to audit.
- *Permissive*: log but allow (the development/debug mode).
- *Disabled*: hooks not consulted.

SELinux's strengths: complete coverage, mature policy ecosystem, MLS (multi-level security) support. Its weakness: complexity. Many sysadmins still `setenforce 0`, which is regrettable: AVC denials are almost always either a real misconfiguration to fix or a missing policy rule a one-liner can add.

`/etc/selinux/config` for boot mode; `getenforce`/`setenforce` at runtime; `journalctl _TRANSPORT=audit` or `ausearch -m AVC` for denials.

== AppArmor

AppArmor (Canonical-maintained, default on Ubuntu and SUSE) uses *path-based* rather than label-based confinement. A profile names a binary by path and lists what it may do:

```
/usr/sbin/nginx {
  capability net_bind_service,
  network inet stream,
  /etc/nginx/** r,
  /var/www/** r,
  /var/log/nginx/*.log w,
  /run/nginx.pid wk,
}
```

The pros: dramatically simpler authoring; profiles are readable; the unit of confinement maps to "this binary". The cons: rename-tricks and mount-tricks can defeat path-based MAC; namespaces complicate path resolution; less expressive than SELinux for multi-role systems.

AppArmor is the right choice for "lock down this one service" workflows; SELinux for "lock down the whole system".

== Landlock

Landlock (mainlined 5.13) is the *unprivileged* MAC: a process can voluntarily install a Landlock ruleset on itself, restricting what *it and its descendants* can do. No privilege needed; no admin policy file; the application sandboxes itself.

```c
struct landlock_ruleset_attr attr = {
    .handled_access_fs = LANDLOCK_ACCESS_FS_READ_FILE |
                         LANDLOCK_ACCESS_FS_WRITE_FILE,
};
int ruleset = landlock_create_ruleset(&attr, sizeof(attr), 0);

struct landlock_path_beneath_attr beneath = {
    .allowed_access = LANDLOCK_ACCESS_FS_READ_FILE,
    .parent_fd = open("/etc", O_PATH | O_CLOEXEC),
};
landlock_add_rule(ruleset, LANDLOCK_RULE_PATH_BENEATH, &beneath, 0);

prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
landlock_restrict_self(ruleset, 0);
```

Landlock complements seccomp (which filters syscalls) by filtering filesystem and network access at the LSM hook layer. As of 6.7 it covers filesystem accesses, network bind/connect, signals, and an expanding surface.

Use case: a parser library that needs to read inputs and nothing else can confine itself to its working directory in 10 lines of code. Tools like `firejail`, `bubblewrap`, and Chromium's sandbox use it where available.

== SMACK and TOMOYO

*SMACK* (Simplified Mandatory Access Control Kernel) is a label-based MAC simpler than SELinux. Used in Tizen, on some embedded Linux distributions, and in safety-critical contexts where policy auditability matters more than expressiveness.

*TOMOYO* uses *learning mode* path-based policies: run the application in learning mode, it records every access made, then switch to enforcing. Niche but interesting for one-off appliances.

== Yama

Yama is a tiny LSM with one job: restrict `ptrace`. `kernel.yama.ptrace_scope` settings:

```
0 = classic ptrace permissions
1 = only descendants (default on Ubuntu); gdb-attach requires sudo or yama.ptrace_scope=0 in unprivileged container
2 = admin-only
3 = no ptrace at all (irrevocable until reboot)
```

This stops "any process running as you can read memory of any other process running as you", which was the historical pre-Yama posture and made browser sandbox escapes much easier.

== IMA and EVM

The *Integrity Measurement Architecture* (IMA) measures (hashes) every file as it is opened and either appends to an audit log (TPM-extended PCR), enforces an *appraisal* against signed reference values, or both.

```bash
# IMA policy: measure every binary executed
echo 'measure func=BPRM_CHECK mask=MAY_EXEC' >> /sys/kernel/security/ima/policy

# Appraisal: refuse to execute binaries without a valid signature
echo 'appraise func=BPRM_CHECK appraise_type=imasig' >> /sys/kernel/security/ima/policy
```

*EVM* (Extended Verification Module) protects file metadata (xattrs, mode, owner) with an HMAC so an attacker can't simply rewrite an IMA xattr to whitelist a malicious binary.

Together IMA+EVM provide *boot-time and runtime* file integrity, with optional TPM-rooted remote attestation. This is the foundation of measured-boot pipelines (Fedora Silverblue, Confidential Computing).

== BPF LSM

BPF LSM (`BPF_PROG_TYPE_LSM`, since 5.7) lets eBPF programs implement LSM hooks. Each program is attached to a named hook and returns a verdict.

```c
SEC("lsm/file_open")
int BPF_PROG(audit_open, struct file *file)
{
    if (is_sensitive_path(file)) {
        log_event(file);
        return audit_only_mode ? 0 : -EPERM;
    }
    return 0;
}
```

BPF LSM is the modern programmable extension: ship runtime-loaded policy without a kernel module, iterate quickly, integrate with the BPF observability ecosystem (ringbuf events, maps for policy state). Falco, Tetragon (Cilium), and Tracee all use it. Sleepable BPF (see _eBPF Deep Dive_) unlocks helpers like `bpf_d_path` and `bpf_copy_from_user` inside LSM hooks.

BPF LSM does not replace SELinux/AppArmor; those provide the comprehensive baseline, while BPF LSM adds targeted runtime detection and policy.

== seccomp-bpf

`seccomp` filters syscalls. Mode 1 (legacy strict) allows only `read`, `write`, `_exit`, `sigreturn`. Mode 2 (*seccomp-bpf*, since 3.5) lets a process install a cBPF filter that inspects the syscall number and arguments and returns a verdict per call:

- `SECCOMP_RET_ALLOW` — execute normally.
- `SECCOMP_RET_ERRNO` — return the embedded errno without executing.
- `SECCOMP_RET_KILL` (process) / `KILL_THREAD` — terminate.
- `SECCOMP_RET_TRAP` — deliver SIGSYS.
- `SECCOMP_RET_TRACE` — wake an attached ptracer.
- `SECCOMP_RET_USER_NOTIF` — forward to a userspace supervisor via a notification fd; the supervisor decides and returns a verdict (5.0+). This is the *user-mode driver* pattern container runtimes use to virtualize specific syscalls.
- `SECCOMP_RET_LOG` — allow but log.

```c
struct sock_filter filter[] = {
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
    BPF_JUMP(BPF_JMP | BPF_JEQ, __NR_execve, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
};
struct sock_fprog prog = { .len = ARRAY_SIZE(filter), .filter = filter };

prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0, &prog);
```

`PR_SET_NO_NEW_PRIVS` is mandatory; without it a setuid binary could escape the filter via execve. Filters compose: a process can install multiple; all are consulted; the most restrictive verdict wins.

Container runtimes (Docker, containerd, podman) ship default seccomp profiles allowlisting ~330 syscalls; Kubernetes pod security uses `RuntimeDefault` profile or operator-defined ones in `/var/lib/kubelet/seccomp/`.

seccomp is *not* an LSM (it sits before LSM hooks), but it's the most-deployed sandboxing mechanism in Linux. Combine with Landlock and BPF LSM for layered defense.

== Capabilities

Linux split root's omnipotence into ~40 *capabilities* (`man capabilities(7)`): `CAP_NET_BIND_SERVICE`, `CAP_SYS_ADMIN`, `CAP_DAC_OVERRIDE`, `CAP_CHOWN`, ...

A process has four capability sets: *Permitted*, *Effective*, *Inheritable*, *Ambient*, plus a *Bounding* mask. `setcap cap_net_bind_service+ep /usr/sbin/nginx` lets nginx bind to port 80 without running as root.

The pathological capability is `CAP_SYS_ADMIN`: about 30% of all capability checks reference it. Stripping it is a strong sandbox; granting it to a container effectively grants root.

`CAP_BPF` (split from CAP_SYS_ADMIN in 5.8) lets eBPF tools run without full admin. `CAP_PERFMON` is the parallel split for `perf`.

== Common Stacks in Practice

- *RHEL/Fedora server*: SELinux (enforcing) + IMA (audit) + seccomp via container runtime.
- *Ubuntu desktop/server*: AppArmor (enforce) + Yama + seccomp.
- *Container runtime defaults*: seccomp profile + capability drop + AppArmor or SELinux profile per container + read-only rootfs + Landlock (recent runtimes).
- *Hardened embedded*: SMACK or AppArmor + IMA-appraise + signed kernel.

The defense-in-depth principle: no single layer is sufficient. seccomp limits syscalls; LSM mediates objects; capabilities scope privilege; namespaces virtualize the view; cgroups limit consumption. A real exploit chain has to bypass *all* of them.

== Observability

```bash
# AVC denials (SELinux)
ausearch -m AVC -ts recent

# AppArmor denials
journalctl -k | grep apparmor

# Recent seccomp kills
journalctl -k | grep seccomp

# Live LSM event stream via BPF
bpftrace -e 'lsm:file_open { @[comm] = count(); }'   # requires CONFIG_BPF_LSM

# What LSMs are active?
cat /sys/kernel/security/lsm
```

== Further Reading

Kernel docs: `Documentation/admin-guide/LSM/`, `Documentation/userspace-api/landlock.rst`, `Documentation/userspace-api/seccomp_filter.rst`, `Documentation/security/IMA-templates.rst`.

Smalley, S., Vance, C. and Salamon, W. (2001). _Implementing SELinux as a Linux Security Module_, NAI Labs.

Cowan, C. et al. (2000). _SubDomain: Parsimonious Server Security_, LISA. (AppArmor's ancestor.)

Salaün, M. (2017). _Landlock: programmatic access control_. LWN series.

Edge, J. (2018-2024). LWN articles on BPF LSM, Landlock, seccomp user-notif.

`security/selinux/`, `security/apparmor/`, `security/landlock/`, `security/integrity/ima/`, `security/bpf/`, `kernel/seccomp.c`.

*See also:* #xref("linux-kernel", "ebpf-deep-dive", label: "eBPF Deep Dive") (BPF LSM internals), #xref("linux-kernel", "containers-in-the-kernel", label: "Containers in the Kernel") (namespaces, capabilities, and per-container LSM profiles), #xref("linux-kernel", "abi-syscalls", label: "ABI and Syscalls") (seccomp filters syscalls before they execute), #xref("linux-kernel", "cgroups-namespaces", label: "Cgroups and Namespaces") (the other half of container confinement).

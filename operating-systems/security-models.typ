= Security Models

Every protection system answers one question: *which subject may perform which operation on which object?* The answers differ wildly (Unix mode bits, SELinux type enforcement, an seL4 capability table) but all are concrete projections of the same abstract structure, and all aspire to the same ideal of a small, trustworthy decision point that nothing can bypass. This chapter treats the models conceptually; the Linux mechanisms that implement them live in `linux-kernel/security-modules.typ` and `linux-kernel/cgroups-namespaces.typ`.

*See also:* _Processes and Threads_, _Memory Management_, _Security Modules_ (implementation), _cgroups and Namespaces_ (implementation), _Asymmetric Cryptography_ (cryptography).

== The Reference Monitor

Anderson's 1972 study gave the field its organizing abstraction: a *reference monitor* mediates every access by a subject to an object, consulting an authorization database to permit or deny. To be trustworthy it must satisfy three properties:

- *Complete mediation*: no access path bypasses it. A check that one syscall enforces and another forgets is not a reference monitor.
- *Tamperproof*: the monitor and its policy database cannot be modified by the subjects it governs. This is why policy lives in the kernel, not in a user-writable file the subject controls.
- *Verifiable*: small and simple enough that its correctness can be established, ideally by proof. The implementation of a verified monitor is the *security kernel*; its trusted surface is the *trusted computing base* (TCB).

These three are in tension with everything a real OS wants to do. Performance pushes toward caching decisions (violating mediation if the cache is stale); features push toward complexity (violating verifiability). The art is keeping the TCB small while the system grows.

=== The Access Matrix

Lampson (1971) framed authorization as a matrix: rows are subjects, columns are objects, and cell $A[s, o]$ holds the rights subject $s$ has on object $o$ (read, write, execute, own, ...). The matrix is the ground truth; no real system stores it densely because it is enormous and sparse. There are two ways to slice it:

- *By column*: store with each object the list of subjects and their rights. This is an *access control list* (ACL): "Who may touch this file?"
- *By row*: store with each subject the list of objects it may reach and how. This is a *capability list*: "What may this process touch?"

The choice is not cosmetic. ACLs make it easy to ask "who can access $o$?" and to revoke everyone at once; capabilities make it easy to ask "what can $s$ reach?", to delegate a single right, and to confine a process by simply not handing it a capability. Revocation is hard for capabilities; enumeration is hard for ACLs. Almost every real model is one of these two with patches for the other's weakness.

== Discretionary Access Control

DAC means the *owner* of an object sets its policy at their discretion. Classic Unix is the canonical example: each inode carries an owner UID, a group GID, and nine permission bits (read/write/execute for owner, group, other) plus three special bits.

#table(columns: (auto, auto, 1fr),
  [*Bit*], [*Octal*], [*Effect*],
  [setuid], [4000], [Run with the file owner's UID, not the caller's],
  [setgid], [2000], [Run with the file group; on dirs, inherit group],
  [sticky], [1000], [On dirs (`/tmp`), only the owner may unlink a file],
)

DAC is simple and familiar, but it is also the source of the deepest structural problem in Unix security: *ambient authority*. A process's rights are an implicit property of its identity (its UID), not of what it was asked to do. Every operation a process performs is checked against all the authority it happens to hold, whether or not that authority is relevant to the task.

This produces the *confused deputy*. A privileged program (the deputy) acts on behalf of a less-privileged client; if the client can name an object the deputy can reach but the client cannot, the deputy will dutifully use its own authority on the client's behalf. The original 1988 example was a compiler with billing-file write access tricked into overwriting that file via an attacker-chosen output path. `setuid-root` binaries are deputies by construction, which is why they are a perennial source of privilege-escalation bugs.

== POSIX Capabilities

Linux splits the monolithic power of root into roughly forty distinct *capabilities*, each a single bit of privilege (the term here is unrelated to object-capabilities below, an unfortunate naming collision). `CAP_NET_BIND_SERVICE` lets a process bind ports below 1024; `CAP_SYS_ADMIN` is the notorious catch-all; `CAP_DAC_OVERRIDE` bypasses file permission checks. The goal is to grant a daemon only the slice of root it needs, retiring `setuid-root`.

Each thread carries several capability sets:

```text
Permitted   (P) : the upper bound on what the thread may make effective
Effective   (E) : the bits actually checked at the moment of a privileged op
Inheritable (I) : bits preserved across execve, ANDed with the file's I set
Bounding    (B) : a ceiling no exec can raise; bits dropped here are gone for good
Ambient     (A) : bits that survive execve of a non-privileged binary (Linux 4.3+)
```

Files carry capability sets too (stored in the `security.capability` xattr): a permitted set, an inheritable set, and an effective bit. The transformation on `execve` is, schematically:

```c
P_new = (P_inheritable_file & I_thread) | (P_permitted_file) | A_thread;
E_new = effective_bit_file ? P_new : A_thread;
I_new = I_thread;
A_new = (file_is_privileged) ? 0 : A_thread;
```

Ambient capabilities were added because inheritable capabilities were nearly useless in practice: they required the target binary to carry a matching file-inheritable set, so a plain script could never receive them. Ambient bits (kernel 4.3, 2015) propagate across `execve` of an unmarked binary, finally letting a launcher hand a daemon exactly `CAP_NET_BIND_SERVICE` and nothing else. The catch is that ambient bits are cleared the instant a `setuid` or file-capability binary is executed, to avoid laundering privilege.

== Mandatory Access Control

Under MAC the policy is set by the system (the security administrator), and no subject, not even the object's owner or root, may relax it. DAC asks "does the owner allow this?"; MAC asks "does the system policy allow this?", and a request must pass both.

=== Bell-LaPadula and Biba

Bell-LaPadula (1976) is a *confidentiality* model. Subjects and objects carry a level from a totally ordered set (Unclassified $<$ Confidential $<$ Secret $<$ Top Secret), generalized to a lattice with compartments. Two rules:

- *Simple security property* — no read up: a subject may read an object only if the subject's level dominates the object's.
- *\*-property* (star) — no write down: a subject may write an object only if the object's level dominates the subject's.

Together these stop information flowing downward in classification. The counterintuitive "no write down" prevents a Secret process from leaking into an Unclassified file.

Biba (1977) is the exact dual for *integrity*: no read down, no write up. A high-integrity process must not read low-integrity data (it might be corrupt) nor allow low-integrity subjects to write high-integrity objects. Confidentiality and integrity pull in opposite directions; a system enforcing both lives in the intersection.

=== Lattices and the Chinese Wall

Generalizing to *multi-level security* (MLS), levels form a lattice $(L, prec.eq)$ under a dominates relation, with a least upper bound for any pair of labels. Information may flow from label $a$ to label $b$ only when $a prec.eq b$. A category set turns the chain into a lattice: $("Secret", {"NUCLEAR"})$ and $("Secret", {"CRYPTO"})$ are incomparable.

The *Chinese Wall* model (Brewer-Nash, 1989) is history-sensitive: an analyst may access any dataset initially, but once they read data from one company in a conflict-of-interest class, all competing datasets in that class become forbidden. The accessible set shrinks based on prior accesses: policy is a function of history, not just labels.

== MAC Implementations

=== SELinux

SELinux implements *type enforcement*. Every subject and object gets a *security context* `user:role:type:level`, e.g. `system_u:system_r:httpd_t:s0`. The interesting field is the *type* (a *domain* when it labels a process). Policy is a vast table of `allow` rules:

```text
allow httpd_t httpd_sys_content_t : file { read getattr open };
```

The web server domain `httpd_t` may read files of type `httpd_sys_content_t` and nothing else, because SELinux is default-deny. Domain transitions (e.g. `init_t` exec'ing the httpd binary transitions to `httpd_t`) are themselves policy-governed. The result is fine-grained but the policy is enormous (the Fedora reference policy is tens of thousands of rules). The central trade-off is precise confinement at the cost of a steep authoring and debugging burden.

=== AppArmor

AppArmor confines by *path* rather than label. A profile names a binary and lists the file paths and permissions it may use:

```text
/usr/sbin/nginx {
  capability net_bind_service,
  /var/www/** r,
  /var/log/nginx/*.log w,
}
```

This is far easier to write and reason about, and needs no filesystem labelling, but it inherits the weaknesses of paths: hard links, bind mounts, and renames can make the same inode reachable under a name the profile never mentioned. The label-vs-path choice is the defining axis: SELinux binds policy to the object's identity, AppArmor to the name used to reach it.

#table(columns: (auto, auto, auto),
  [*Aspect*], [*SELinux*], [*AppArmor*],
  [Policy anchor], [inode label (type)], [filesystem path],
  [Survives rename/hardlink], [yes], [no — name-based],
  [Authoring difficulty], [high], [moderate],
  [Needs labelled FS], [yes (xattrs)], [no],
)

== The LSM Framework

Linux does not bake any one MAC model into the kernel. The *Linux Security Modules* framework places hooks at every security-relevant decision point: `inode_permission`, `bprm_check_security`, `socket_connect`, and hundreds more. A hook is a callback the core kernel invokes after its own DAC checks pass but before it acts; the module returns allow or `-EACCES`. This keeps policy logic out of the core and lets SELinux, AppArmor, Smack, or TOMOYO be the decision-maker.

Originally exactly one "major" module could be active. Modern kernels support *stacking*: capability and the small "minor" modules (Yama, LoadPin, SafeSetID, Landlock) always compose, and stacking of the larger modules has been progressively enabled. A request must satisfy *every* stacked module; the hooks are conjunctive, consistent with the reference-monitor ideal that any module may deny.

== Capability-Based Security

This is *object*-capabilities, distinct from POSIX capability bits. A capability is an unforgeable reference that both designates an object and confers the authority to use it. Possession is permission; there is no separate ACL lookup. Because you cannot name what you do not hold, the confused-deputy problem largely dissolves: a process has authority only over objects whose capabilities it was explicitly given, so it cannot be tricked into using authority it never received on an attacker's object.

The Unix file descriptor is already a near-capability: an opaque, unforgeable, per-process handle that grants the access negotiated at `open`. Capability designs generalize this to *everything*.

- *Capsicum* (FreeBSD) adds a *capability mode*: after `cap_enter()` a process loses all global namespaces (no open-by-path, no PIDs, no sysctls) and may act only through file descriptors, each narrowed by a *rights* mask via `cap_rights_limit()`. A process can sandbox itself incrementally.
- *seL4* is a microkernel whose entire API is capability-invocation, with a machine-checked proof that the implementation matches its specification and enforces its access-control model, the strongest realization of "verifiable."
- *Fuchsia* builds its userspace on *handles* to kernel objects; a component receives a bundle of handles at startup and can reach nothing else.

The unifying discipline is the *principle of least authority* (POLA): a component should hold exactly the authority its task requires, no more. Ambient-authority DAC violates POLA by construction, since a process wields its whole UID for every act. Capability systems make least authority the default, since authority must be deliberately conferred.

#table(columns: (auto, auto, auto, auto),
  [*Property*], [*DAC*], [*MAC*], [*Object-capability*],
  [Who sets policy], [object owner], [system admin], [whoever delegates a cap],
  [Granularity], [user/group], [label/type], [per-object reference],
  [Authority model], [ambient (identity)], [ambient (label)], [explicit (possession)],
  [Revocation], [edit ACL], [relabel/policy], [hard (needs indirection)],
  [Confused-deputy resistance], [poor], [partial], [strong],
)

== Sandboxing and Privilege Separation

=== seccomp

`seccomp` restricts which syscalls a process may issue. The original mode allowed only `read`, `write`, `exit`, `sigreturn`. *seccomp-bpf* generalizes this to a BPF program run on every syscall, deciding from the syscall number and argument registers whether to allow, kill, trap, or return an errno:

```c
// Allow read/write, kill on anything else (sketch)
BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
BPF_JUMP(BPF_JMP | BPF_JEQ, __NR_read,  0, 1),
BPF_STMT(BPF_RET, SECCOMP_RET_ALLOW),
BPF_JUMP(BPF_JMP | BPF_JEQ, __NR_write, 0, 1),
BPF_STMT(BPF_RET, SECCOMP_RET_ALLOW),
BPF_STMT(BPF_RET, SECCOMP_RET_KILL_PROCESS),
```

A crucial limitation: seccomp filters argument registers, not memory the pointers reference, so it cannot safely inspect path strings (TOCTOU). It is a syscall-surface reducer, not a full policy engine.

=== Namespaces and cgroups

Isolation also comes from *not sharing*. Namespaces virtualize a global resource (PID, mount, network, user, ...) so a process sees its own instance; cgroups bound resource consumption. Together they are the substrate of containers, providing confinement by restricting the *namespace* a process can even name, complementary to restricting *operations*. The details are in `linux-kernel/cgroups-namespaces.typ`.

=== pledge and unveil

OpenBSD's `pledge()` declares the set of operation classes a program promises to stay within ("stdio", "rpath", "inet"); a violation kills the process. `unveil()` makes most of the filesystem invisible, revealing only named paths with named permissions. The design wins on ergonomics (a few lines, no policy file) at the cost of OpenBSD's coarser, hand-curated classes.

=== Case study: OpenSSH privilege separation

`sshd` is the canonical *privilege separation* design. A small, trusted *monitor* runs as root and performs only the handful of operations that genuinely need privilege (authentication, PTY allocation). All untrusted work, such as parsing network input and running the protocol state machine before login, happens in an unprivileged, chrooted child that talks to the monitor over a socket. A bug in the parser (the large, exposed attack surface) compromises only the deprivileged child; the attacker still cannot do anything the monitor refuses. This is the reference monitor recursively applied: shrink the privileged TCB to the smallest piece that must be trusted.

== Integrity and Verified Execution

Confidentiality and access control assume the code itself is genuine. Several mechanisms anchor that assumption:

- *IMA/EVM*: the Integrity Measurement Architecture measures (hashes) files as they are opened and can appraise them against signed reference values; EVM protects the security xattrs (including those measurements and SELinux labels) with an HMAC or signature so they cannot be tampered with offline.
- *dm-verity*: a read-only block device backed by a Merkle tree of block hashes rooted in a signed value; any modified block fails verification on read. It underpins verified boot on Android and ChromeOS.
- *Signed kernel modules*: the kernel refuses to load modules whose signature does not chain to a trusted key, closing an obvious path to kernel-level code injection.

These extend the tamperproof property from the running monitor down to the bytes on disk it was loaded from.

== Pitfalls

- *setuid is a loaded gun.* A `setuid-root` binary inherits the caller's environment, file descriptors, `umask`, and resource limits while running as root; forgetting to sanitize any of them (`LD_PRELOAD`, an inherited fd 2) is a classic escalation. Prefer fine-grained file capabilities or a privsep monitor.
- *The confused deputy is not a Unix bug, it is an ambient-authority bug.* Adding more checks to the deputy rarely fixes it; removing the ambient authority (capabilities) does.
- *"Just set it permissive."* Disabling SELinux enforcement to make an app work converts a deny into a logged warning; the access still happens. The fix is an `audit2allow`-derived policy delta, not abandonment.
- *seccomp allowlists miss syscall variants.* Filtering `open` but not `openat`/`openat2`, or `select` but not `pselect6`, leaves an escape. libc may transparently choose the variant you forgot; allowlist by behavior, deny by default.
- *Capability inheritance surprises.* Inheritable POSIX capabilities do nothing without a matching file-inheritable set; people expect them to propagate like ambient bits and silently get an unprivileged process. Conversely, `CAP_SYS_ADMIN` is so broad that granting it "to bind a mount" effectively re-grants root.
- *Path-based confinement and hard links.* An AppArmor profile keyed on `/var/www/**` does not constrain the same inode reached via a hard link elsewhere; label-based MAC does.

== Further Reading

Anderson, J. P. (1972). "Computer Security Technology Planning Study." ESD-TR-73-51 (the reference-monitor report).

Lampson, B. (1974). "Protection." ACM Operating Systems Review (the access matrix).

Bell, D., LaPadula, L. (1976). "Secure Computer System: Unified Exposition and Multics Interpretation." MITRE MTR-2997.

Biba, K. (1977). "Integrity Considerations for Secure Computer Systems." MITRE TR-3153.

Brewer, D., Nash, M. (1989). "The Chinese Wall Security Policy." IEEE S&P.

Hardy, N. (1988). "The Confused Deputy: (or why capabilities might have been invented)." ACM Operating Systems Review.

Saltzer, J., Schroeder, M. (1975). "The Protection of Information in Computer Systems." Proceedings of the IEEE.

Loscocco, P., Smalley, S. (2001). "Integrating Flexible Support for Security Policies into the Linux Operating System." USENIX ATC (SELinux/LSM).

Watson, R. et al. (2010). "Capsicum: Practical Capabilities for UNIX." USENIX Security.

Klein, G. et al. (2009). "seL4: Formal Verification of an OS Kernel." SOSP.

Provos, N., Friedl, M., Honeyman, P. (2003). "Preventing Privilege Escalation." USENIX Security (OpenSSH privsep).

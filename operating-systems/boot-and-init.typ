#import "../template.typ": xref

= Boot and Init

Booting is a chain of trust and handoffs: each stage knows just enough to find, validate, and transfer control to the next, shedding privilege and adding abstraction as it goes. From the first instruction the CPU fetches out of firmware ROM to the moment a login prompt appears, half a dozen distinct programs run, each in a different environment: no MMU, then flat memory, then a kernel, then a tiny root, then the real root. This chapter treats the path conceptually; Linux's concrete early-boot code and module machinery live in `linux-kernel/introduction.typ` and `linux-kernel/kernel-modules.typ`.

*See also:* #xref("operating-systems", "processes-and-threads", label: "Processes and Threads"), #xref("operating-systems", "storage-stack", label: "The Storage Stack"), #xref("linux-kernel", "kernel-modules", label: "Kernel Modules") (implementation), #xref("cpu-architecture", "cpu-fundamentals", label: "CPU Fundamentals") (architecture).

== The Boot Chain

The whole sequence is a relay of progressively richer environments. Each handoff is also a trust boundary: ideally each stage verifies the next before jumping to it.

```text
  firmware     power-on; bring up RAM, find a boot device
     |
  bootloader   load kernel + initramfs, build a command line
     |
  kernel       decompress, set up MMU, discover devices
     |
  initramfs    temporary root with just-enough drivers
     |
  init (PID 1) mount real root, start the service graph
     |
  services     network, login, daemons (the running system)
```

Three properties recur at every step. First, *minimalism*: a stage carries only the code needed to reach the next one (the bootloader cannot mount ext4; the initramfs cannot run a desktop). Second, *handoff convention*: a documented register/memory ABI for passing control plus parameters (the kernel command line, a device tree or ACPI tables, the initramfs address). Third, *trust*: a signature, hash, or measurement that lets a stage decide whether the next one is allowed to run.

== Firmware

Firmware is the code in non-volatile memory the CPU executes first. It initializes DRAM (memory training), enumerates buses, and locates a boot device. Two families dominate the x86 world.

#table(columns: (auto, 1fr),
  [*Legacy BIOS*], [Real-mode (16-bit) at reset; reads a 512 B Master Boot Record from the first sector; chain-loads whatever code lives there. No standard signature scheme, no partitioning awareness beyond the MBR table.],
  [*UEFI*], [Runs in 32/64-bit mode; understands the GPT partition table and a FAT-formatted EFI System Partition (ESP); loads PE executables (`.efi` applications) by file path; exposes runtime services and a variable store.],
)

=== UEFI Boot Manager and the ESP

UEFI ships a *boot manager* that reads ordered NVRAM variables (`BootOrder` and a `Boot0001`, `Boot0002`, ... list), each pointing at an `.efi` binary on the ESP (conventionally `/EFI/<vendor>/grubx64.efi` or `\EFI\BOOT\BOOTX64.EFI` as the removable fallback). There is no MBR bottleneck: the loader is an ordinary file in a FAT filesystem. Tools edit these variables at runtime:

```bash
efibootmgr -c -d /dev/nvme0n1 -p 1 \
  -L "linux" -l '\EFI\linux\grubx64.efi'
```

=== Secure Boot, shim, and MOK

Secure Boot makes the firmware verify each `.efi` image's signature against keys in its database (`db`) before executing it, with a revocation list (`dbx`). Because distributions cannot all hold Microsoft's signing key, the common arrangement inserts a small Microsoft-signed first-stage loader called *shim*: shim carries the distribution's own certificate, verifies the real bootloader (GRUB) against it, and consults a Machine Owner Key list (*MOK*) that the local administrator can enroll for self-signed kernels. The trust chain becomes:

```text
firmware (db) -> shim (MS-signed) -> GRUB (distro/MOK-signed)
             -> kernel -> signed modules
```

=== Measured vs Verified Boot, TPM PCRs

Two distinct ideas are easily conflated. *Verified boot* refuses to continue if a signature check fails — it enforces. *Measured boot* does not block anything; instead each stage hashes the next and *extends* that hash into a Trusted Platform Module Platform Configuration Register (PCR) before handing off. A PCR is append-only: $"PCR"_(n) <- H("PCR"_(n) || "measurement")$, so the final register values are a tamper-evident fingerprint of exactly what ran. Software can later ask the TPM to *seal* a secret (e.g. a disk-encryption key) to a set of PCR values, so the secret only unseals if the boot chain matches. Verified boot says "stop if wrong"; measured boot says "record what happened and let a later policy decide."

== Bootloaders

The bootloader's job is narrow: locate the kernel and initramfs on some filesystem, load them into RAM, assemble a command line, and jump to the kernel's documented entry point.

#table(columns: (auto, 1fr),
  [*GRUB2*], [The Unix default. Two-stage: a tiny first stage loads a larger core image that understands many filesystems, RAID, LVM, and crypto. Supports the Multiboot protocol (boot non-Linux kernels), a scripting config (`grub.cfg`), and a menu.],
  [*systemd-boot*], [Minimal UEFI-only manager; no filesystem drivers of its own; kernels live on the ESP as files (or as Unified Kernel Images). Simple drop-in `.conf` entries; nothing to "install" into a boot sector.],
  [*U-Boot*], [The embedded/ARM standard. Rich device support, scripting, network boot (TFTP), and a hardware-bringup role; commonly loads a kernel + device tree blob + initramfs on boards without UEFI.],
  [*coreboot*], [A firmware *replacement* (not strictly a bootloader): minimal hardware init, then hands to a "payload" (SeaBIOS, GRUB, a Linux kernel via LinuxBoot, or EDK2). Shrinks the proprietary firmware surface.],
)

A typical GRUB entry shows the three deliverables (kernel, command line, initramfs):

```ini
menuentry "Linux" {
    linux  /vmlinuz-6.6 root=/dev/mapper/vg-root ro quiet
    initrd /initramfs-6.6.img
}
```

== Kernel Handoff

When control reaches the kernel image, the visible binary is usually a small *self-decompressing* stub plus a compressed payload. The stub relocates and inflates the real kernel, then early architecture code runs: set up a stack, enable the MMU and paging, parse the command line and the hardware description (ACPI tables on x86, a Device Tree blob on most ARM/RISC-V), and bring up the boot CPU. Driver and device discovery follow; only built-in drivers exist at this point, which is the entire reason the next stage is needed.

The kernel then mounts the *initramfs* as a temporary root filesystem (an in-memory tmpfs unpacked from the cpio image the bootloader loaded), and executes `/init` inside it as PID 1. Once that early userspace has located and mounted the *real* root, it does not reboot or re-exec the kernel; it performs `switch_root` (modern) or the older `pivot_root`: the real root is moved to `/`, the initramfs memory is freed, and the real `/sbin/init` is exec'd, inheriting PID 1.

```bash
# inside initramfs /init, after mounting the real root at /newroot
exec switch_root /newroot /sbin/init
```

== initramfs / initrd

The initramfs exists to solve a bootstrapping paradox: to mount the root filesystem you may need drivers or tooling that themselves live on the root filesystem. The initramfs is a self-contained early userspace that ships exactly those pieces:

- Storage and bus drivers as modules (NVMe, SATA, USB, RAID controllers).
- Logic to assemble the root: LVM activation, `mdadm` RAID, `cryptsetup` for LUKS-encrypted roots, or a DHCP client for network/NFS root.
- The `switch_root` handoff once the real root is ready.

It is built as a *cpio archive* (optionally compressed) that the kernel unpacks into a tmpfs, distinct from the older `initrd`, which was a fixed-size block image mounted as a real device. Generators (`dracut`, `mkinitcpio`, `initramfs-tools`) scan the running system, copy the needed modules and binaries, and pack them:

```bash
find . | cpio -o -H newc | zstd > /boot/initramfs.img
```

== PID 1 and Init Systems

When the real `/sbin/init` starts, it is PID 1 — the ancestor of every other process and the system's policy engine for "what should be running." The history of init is a migration from *imperative shell scripts run in order* to a *declarative dependency graph with built-in supervision*.

=== SysV init and BSD init

Classic *SysV init* defines numbered *runlevels* (0 halt, 1 single-user, 3 multi-user, 5 graphical, 6 reboot). Entering a runlevel runs a directory of `rc` scripts in lexical order (`S01...`, `S99...`), sequentially. It is simple and transparent but slow (no parallelism), fragile (an error mid-script can wedge boot), and supervision-free (once a daemon forks into the background, init forgets it). *BSD init* is even simpler: a hand-written `/etc/rc` shell script plus `/etc/ttys` for getty supervision; readable, but with no dependency model at all.

=== systemd

*systemd* recasts the problem as a graph of typed *units* (`.service`, `.socket`, `.mount`, `.target`, `.timer`). Targets replace runlevels as synchronization points (`multi-user.target` ≈ runlevel 3). Three ideas distinguish it:

- *Parallelism by dependency*: units declare `Requires`/`After`, and systemd starts everything whose prerequisites are met, concurrently.
- *Activation*: services can be started lazily on demand by *socket* or *D-Bus* activation rather than at boot.
- *cgroup-based tracking*: every service runs in its own control group, so systemd reliably knows the full process tree of a service; no daemon can "escape" by double-forking. This solves the supervision gap SysV had.

```ini
[Unit]
Description=Web server
After=network.target

[Service]
ExecStart=/usr/bin/httpd -f
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

=== runit, s6, OpenRC, launchd

A parallel tradition keeps init *small* and pushes supervision into a tree of tiny long-lived supervisors:

- *runit* and *s6* build a *supervision tree*: each service is a child of a supervisor process that restarts it if it dies and forwards signals. PID 1 is a few hundred lines; complexity lives in composable daemons. The s6 family is prized for correctness and for running well inside containers.
- *OpenRC* (Gentoo, Alpine) adds dependency-ordered parallel startup on top of a traditional init, without replacing PID 1 wholesale.
- *launchd* (macOS) anticipated much of systemd: a single daemon managing jobs described by plists, with on-demand socket and path activation.

#table(columns: (auto, auto, auto, auto, auto),
  [*System*], [*Model*], [*Parallel start*], [*Supervision*], [*Activation*],
  [SysV init], [ordered scripts], [no], [none], [none],
  [BSD init], [single rc script], [no], [getty only], [none],
  [systemd], [dependency graph], [yes], [cgroup-tracked], [socket / D-Bus / path],
  [runit / s6], [supervision tree], [yes (deps via bundles)], [per-service supervisor], [none (s6: yes)],
  [OpenRC], [dependency scripts], [yes], [optional], [none],
  [launchd], [job plists], [yes], [yes], [socket / path],
)

== Supervision and Why PID 1 Is Special

PID 1 carries two non-negotiable kernel obligations. First, *it must never exit*: if PID 1 dies the kernel panics, because there is no ancestor left to anchor the process tree. Second, *it must reap zombies*: when any process's parent exits, its orphaned children are reparented to PID 1 (or to a registered subreaper), and PID 1 must `wait` on them or they accumulate as zombies until PID exhaustion (see the zombie and `PR_SET_CHILD_SUBREAPER` discussion in `operating-systems/processes-and-threads.typ`).

The *supervision-tree* philosophy generalizes this: rather than daemonizing and detaching, a service stays in the foreground as a child of a supervisor that knows its exact state, restarts it on failure with backoff, captures its output, and forwards signals. "Run it and forget it" becomes "own it for its whole lifetime." This is why modern services are written *not* to double-fork — the old daemonization dance actively fights the supervisor.

== Socket Activation and Ordering vs Readiness

*Socket activation* inverts startup: the init system creates and listens on a service's socket *before* the service starts, then launches the service (passing it the already-open fd) only when the first connection arrives. Benefits: services start lazily, boot parallelizes freely (clients can connect to a socket whose server is still starting, as the kernel buffers), and dependency ordering between two socket-activated services largely disappears.

This exposes a subtle trap: *ordering is not readiness*. `After=postgresql.service` guarantees only that the unit was *started* first, not that the database is accepting queries. A correctly written service signals readiness explicitly (systemd `Type=notify` with `sd_notify`, or a health check) so dependents wait for *ready*, not merely *launched*. Socket activation sidesteps part of this by making the socket itself the readiness boundary.

== Containers and Minimal Init

Inside a container, the entrypoint process becomes PID 1 of that PID namespace, inheriting the kernel's PID 1 contract whether it is prepared for it or not. Most application binaries are not: they do not install default handlers for `SIGTERM` (so `docker stop` waits out its grace period and then `SIGKILL`s), and they do not reap reparented children (so zombies pile up inside the container). The fix is a tiny purpose-built init:

#table(columns: (auto, 1fr),
  [*tini* / *dumb-init*], [~Hundreds of lines; install as PID 1, exec the real app, then forward all signals to it and reap any orphaned children. `docker run --init` injects exactly such a process.],
)

This is the same logic as a full init system, distilled to the two kernel obligations: forward signals, reap zombies.

== Pitfalls

- *PID 1 that does not forward signals*: an app run directly as container PID 1 ignores `SIGTERM`, so orchestrators escalate to `SIGKILL` after the grace period (abrupt termination, lost flushes). Use `--init`, `tini`, or `exec` the app so it *is* PID 1 with proper handlers.
- *initramfs missing the root driver*: regenerate the initramfs after a hardware or storage-layout change, or the kernel boots, finds no driver for the NVMe/RAID/LUKS root, and drops to an emergency shell ("cannot find root device"). The image must contain every module on the path to `/`.
- *Secure Boot bricking custom kernels*: a self-built or out-of-tree-module kernel is unsigned; with Secure Boot on, firmware or shim refuses it. Either enroll a MOK and sign the kernel/modules, or you will face a non-booting machine after the next kernel build.
- *Ordering mistaken for readiness*: `After=` orders *start*, not *availability*; a service that connects immediately to a just-"started" dependency races against it. Use readiness signaling (`Type=notify`, health checks) or socket activation.
- *Daemonizing under a supervisor*: a service that double-forks into the background hides its real PID from a cgroup-unaware supervisor and confuses restart logic. Run in the foreground.

== Further Reading

Intel et al. (2024). "Unified Extensible Firmware Interface (UEFI) Specification," v2.10.

Trusted Computing Group (2019). "TCG PC Client Platform Firmware Profile Specification."

Garrett, M. (2012). "Secure Boot and Restricted Boot." (mjg59 writings on shim and MOK).

Free Software Foundation. "GNU GRUB Manual" (Multiboot specification and two-stage design).

Poettering, L., Sievers, K. (2010). "systemd System and Service Manager" (project documentation and the "Rethinking PID 1" essay).

Pape, G. "runit — a UNIX init scheme with service supervision." (runit documentation).

Bernstein, D. J. "daemontools" and Skarnet's "s6" supervision-suite documentation.

Apple Inc. "Daemons and Services Programming Guide" (launchd design).

Denx Software. "Das U-Boot Reference Manual."

Tanenbaum, A., Bos, H. (2014). "Modern Operating Systems" (4th ed.), Chapter 1 (system boot).

#import "template.typ": project

#project("Linux Kernel")[
  #align(center)[
  #block(
    fill: luma(245),
    inset: 10pt,
    width: 80%,
    text(size: 9pt)[
      #align(center)[
        #text(size: 24pt, weight: "bold")[Linux Kernel: Internals for Systems Engineers]
      ]
    ]
  )
]

#outline(
  title: "Table of Contents",
  depth: 2,
)

#emph[Enjoy.]

#pagebreak()

#include "linux-kernel/introduction.typ"
#pagebreak()

#include "linux-kernel/abi-syscalls.typ"
#pagebreak()

#include "linux-kernel/mmap-memory.typ"
#pagebreak()

#include "linux-kernel/cpu-affinity.typ"
#pagebreak()

#include "linux-kernel/scheduler.typ"
#pagebreak()

#include "linux-kernel/cgroups-namespaces.typ"
#pagebreak()

#include "linux-kernel/interrupts.typ"
#pagebreak()

#include "linux-kernel/kernel-tracing.typ"
#pagebreak()

#include "linux-kernel/kernel-modules.typ"
#pagebreak()

= Conclusion

The Linux kernel is the universal substrate of modern server and cloud computing. Mastery of its primitives — syscall ABI, virtual memory, scheduling classes, cgroups, interrupts, and tracing — is what separates systems engineers who can diagnose tail-latency outliers from those who cannot.

*Key synthesis:*

- *Syscalls* cost ~100-300 cycles on modern x86; vDSO eliminates the ring transition entirely for read-only fast paths.
- *mmap* is the single most important memory primitive: file-backed mappings, COW, huge pages, and userfaultfd all build on it.
- *CPU affinity + isolation* (`isolcpus`, `nohz_full`, IRQ pinning) is the foundation of low-latency / real-time deployments.
- *CFS* schedules by virtual runtime on a red-black tree; `SCHED_DEADLINE` provides EDF guarantees for hard real-time.
- *cgroups v2 + namespaces* are the kernel-level container primitives — Docker is just userspace glue around `clone(CLONE_NEW*)`.
- *NAPI* and *threaded IRQs* let high-rate networking scale beyond per-packet interrupt cost.
- *eBPF* turned kernel observability from a recompile-and-reboot exercise into safe, programmable, production-grade tracing.

== Further Reading

Bovet, D. and Cesati, M. (2005). _Understanding the Linux Kernel_, 3rd ed. O'Reilly.

Love, R. (2010). _Linux Kernel Development_, 3rd ed. Addison-Wesley.

Mauerer, W. (2008). _Professional Linux Kernel Architecture_. Wiley.

Gregg, B. (2020). _Systems Performance: Enterprise and the Cloud_, 2nd ed. Addison-Wesley.

Gregg, B. (2019). _BPF Performance Tools_. Addison-Wesley.

Linux kernel source: #link("https://elixir.bootlin.com/linux/latest/source")[elixir.bootlin.com] (browsable cross-reference).

Kernel documentation: #link("https://www.kernel.org/doc/html/latest/")[kernel.org/doc].
]

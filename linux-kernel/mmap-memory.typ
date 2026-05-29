= mmap and Memory Management

`mmap` is the most important memory primitive in Linux. Almost everything in the kernel's memory subsystem — file I/O, shared libraries, `fork()`'s copy-on-write, anonymous heaps, hugepages — is either implemented as `mmap` or trivially layered on top of it. This chapter is a deep dive into what `mmap` actually does, and the family of features built around it.

== mmap Fundamentals

```c
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
```

A successful `mmap` returns a pointer to a *virtual memory area* (VMA) — a contiguous range of virtual addresses backed by something the kernel will fill in on demand. Crucially, *the kernel does not (in general) allocate physical pages immediately.* It records the mapping in the process's `mm_struct` and returns. Physical pages are allocated lazily on first access via the page-fault handler.

You can confirm this with `cat /proc/<pid>/maps`:

```
7f3a4c000000-7f3a4c800000 rw-p 00000000 00:00 0
7f3a4c800000-7f3a4ca00000 r--p 00000000 fd:01 12345  /usr/bin/ls
7f3a4ca00000-7f3a4cc00000 r-xp 00200000 fd:01 12345  /usr/bin/ls
```

Each line is a VMA. Columns: `addr-end perm offset major:minor inode path`.

The *resident set size* (`/proc/<pid>/status: VmRSS`) is what is actually backed by physical memory; the *virtual size* (`VmSize`) is the sum of all VMAs and can be vastly larger.

== Anonymous vs File-Backed

The two big modes:

#table(columns: (auto, 1fr, 1fr),
  [*Mode*], [*Backed by*], [*Typical use*],
  [`MAP_ANONYMOUS`], [zero-filled pages from the page allocator (and swap if reclaimed)], [`malloc`'s big allocations, JIT code, anonymous shared memory],
  [File-backed (`fd >= 0`)], [the page cache; eventually the underlying file via the block layer], [shared libraries, `mmap`-based file I/O, on-disk databases],
)

*MAP_PRIVATE vs MAP_SHARED:*

- `MAP_PRIVATE`: writes are *copy-on-write*. The original page is shared until you write, at which point the kernel allocates a fresh page, copies, and remaps it. Used for executables (each process gets its own COW copy of `.data`).
- `MAP_SHARED`: writes propagate to the underlying object — for file mappings, eventually back to disk; for anonymous shared mappings, visible to other processes mapping the same region. Used for IPC (`shm_open` + `mmap`), and for direct file modification.

The `MAP_ANONYMOUS | MAP_SHARED` combination is the basis of `posix_shm` and POSIX semaphores' shared state.

== Copy-on-Write and fork()

`fork()` does not copy the parent's memory. It copies the *page tables*, marks every writable page in both parent and child read-only, and lets the page-fault handler do the actual copying lazily. This makes `fork()` cheap on a 100 GB process, as long as nobody writes much before `exec`.

Concretely:

1. Parent calls `fork()`.
2. Kernel walks parent's page tables, duplicating PTEs into a new `mm_struct` for the child.
3. Every writable PTE in *both* tables is flipped to read-only and marked COW.
4. Both processes resume.
5. On a write to a COW page, the CPU traps. The handler:
   - Allocates a fresh physical page.
   - Copies the contents.
   - Remaps the writer's PTE to point at the new page, writable.
   - Decrements the refcount on the original page; if it reaches 1, the *other* process's PTE can be flipped back to writable on its next fault (handled lazily).

Memory-overcommit considerations: if both processes scribble all over their address space post-`fork()`, you need ~2× the original RAM. This is why `vfork()` and `posix_spawn` exist — they avoid the duplication when you only need to `exec` immediately.

== Huge Pages

Standard pages are 4 KiB on x86-64, ARM64, etc. The TLB has a fixed number of entries (~64 in L1, ~1500-3000 in L2 on modern x86), so a working set larger than `entries × 4 KiB` will start TLB-thrashing. Huge pages — 2 MiB and 1 GiB on x86-64, 64 KiB / 2 MiB / 1 GiB on ARM64 — let one TLB entry cover much more memory.

There are *two* huge-page systems in Linux, and they are not interchangeable.

=== Transparent Huge Pages (THP)

The kernel opportunistically promotes naturally-aligned 2 MiB ranges of anonymous memory to huge pages. Configured via:

```
/sys/kernel/mm/transparent_hugepage/enabled       # always | madvise | never
/sys/kernel/mm/transparent_hugepage/defrag         # always | defer | madvise | never
/sys/kernel/mm/transparent_hugepage/khugepaged/    # background promotion knobs
```

*`always`* — kernel promotes whenever it can. Simple, but the synchronous compaction on allocation can introduce tail-latency spikes (the famous "THP stall").

*`madvise`* — promotion only happens for ranges marked with `madvise(addr, len, MADV_HUGEPAGE)`. This is what most production systems should use.

*`never`* — disabled; useful when THP hurts (some databases, some real-time workloads).

```c
posix_memalign(&p, 2 * 1024 * 1024, size);
madvise(p, size, MADV_HUGEPAGE);
// Now this range is a candidate for THP promotion
```

Why THP can hurt:

- *Compaction stalls.* Allocating a 2 MiB contiguous physical range may require moving pages around. Synchronous compaction on the page-fault path introduces multi-millisecond stalls.
- *Memory bloat.* THP rounds up. A `mmap` of slightly more than 2 MiB might end up backed by 4 MiB of physical RAM.
- *Splitting cost.* Operations that need 4 KiB granularity (e.g. NUMA migration of a few pages) split THPs back to baseline pages, undoing the work.

*Diagnostic:* `cat /proc/<pid>/smaps | grep AnonHugePages` shows per-VMA THP residency. `cat /proc/meminfo | grep -i huge` shows system-wide.

=== hugetlbfs and Explicit Huge Pages

For workloads that *must* have huge pages (databases, low-latency trading, DPDK), use the explicit `hugetlbfs` interface. You pre-reserve huge pages at boot or runtime, and applications request them via `MAP_HUGETLB`.

*Reserving at runtime:*

```
echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
echo 4    > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
```

*Boot-time (more reliable for 1 GiB pages — fragmentation makes runtime allocation flaky):*

```
default_hugepagesz=1G hugepagesz=1G hugepages=8
```

*Use:*

```c
void *p = mmap(NULL, 2 * 1024 * 1024,
               PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
               -1, 0);
```

Or for 1 GiB pages:

```c
mmap(NULL, 1UL << 30,
     PROT_READ | PROT_WRITE,
     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
     -1, 0);
```

These pages are *never swapped, never split, never migrated* (without explicit migration calls). They are a fixed pool, dedicated.

For shared-memory huge-page allocations across processes, mount `hugetlbfs` and back a file:

```
mount -t hugetlbfs none /mnt/huge
```

```c
int fd = open("/mnt/huge/region", O_CREAT | O_RDWR, 0600);
ftruncate(fd, size);
void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
```

This is the pattern DPDK and SPDK use for their hugepage memory pools.

== Useful mmap Flags

- *`MAP_POPULATE`* — pre-fault all pages now, instead of on first access. Eliminates the page-fault cost from your critical path. Historically only honored for `MAP_PRIVATE`; honored for `MAP_SHARED | MAP_ANONYMOUS` from 6.0 onwards. On older kernels you must touch the pages manually to force allocation.
- *`MAP_LOCKED`* — equivalent to `mmap` followed by `mlock`. Pages cannot be swapped or reclaimed. Subject to `RLIMIT_MEMLOCK`.
- *`MAP_NORESERVE`* — don't reserve swap space. Lets you mmap a sparse range much larger than RAM+swap; access beyond what's actually allocatable will SIGBUS.
- *`MAP_FIXED_NOREPLACE`* — like `MAP_FIXED` but fails (instead of silently unmapping) if the address range is already in use. Strongly preferred over `MAP_FIXED` for anything user-controlled.
- *`MAP_STACK`* — hint that this region will be a thread stack. On most architectures it's a no-op; the flag exists so the kernel can pick stack-friendly placement on architectures where that matters.

== mlock and mlockall

`mlock` pins pages in physical memory: they cannot be swapped, cannot be reclaimed by the page cache eviction path, and must be present at the time of the call (the kernel will fault them in if needed).

```c
mlock(addr, len);                          // pin a range
mlockall(MCL_CURRENT | MCL_FUTURE);        // pin everything, including future allocs
```

*When to use it:* Real-time threads where any page fault is unacceptable. Trading systems, audio pipelines, RT-kernel tasks. Combine with `mlockall(MCL_CURRENT | MCL_FUTURE)` plus `SCHED_FIFO`/`SCHED_DEADLINE` for hard latency guarantees.

*Limits:* `RLIMIT_MEMLOCK` caps how much an unprivileged process can lock. `CAP_IPC_LOCK` removes the cap.

*Common bug:* `mlockall(MCL_CURRENT)` only locks what is currently mapped. `MCL_FUTURE` is required if any later `mmap`/`brk` would otherwise be unlocked.

== userfaultfd

`userfaultfd` lets a userspace process handle page faults for a region of memory it owns. Created in 4.3, it is the foundation of:

- *Live VM migration* (QEMU post-copy): the destination starts running, faults stream pages on demand from the source.
- *CRIU* checkpoint/restore.
- *Distributed shared memory libraries.*
- *Userspace garbage collectors* (e.g. read barriers).

Skeleton:

```c
int uffd = syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK);

struct uffdio_api api = { .api = UFFD_API, .features = 0 };
ioctl(uffd, UFFDIO_API, &api);

struct uffdio_register reg = {
    .range = { .start = (uintptr_t)addr, .len = len },
    .mode  = UFFDIO_REGISTER_MODE_MISSING,
};
ioctl(uffd, UFFDIO_REGISTER, &reg);

// Spawn a fault-handler thread:
struct uffd_msg msg;
read(uffd, &msg, sizeof msg);
// msg.arg.pagefault.address tells us where.
// Provide a page:
struct uffdio_copy copy = {
    .dst = msg.arg.pagefault.address & ~(PAGE_SIZE - 1),
    .src = (uintptr_t)source_page,
    .len = PAGE_SIZE,
};
ioctl(uffd, UFFDIO_COPY, &copy);
```

The faulting thread blocks until the userspace handler resolves the fault.

*Caveat:* `userfaultfd` has been a recurring source of CVEs (because it lets unprivileged code influence kernel page-fault handling). Modern kernels gate it behind `vm.unprivileged_userfaultfd=0` by default; root or `CAP_SYS_PTRACE` is required.

== madvise: Hints and Imperatives

`madvise` tells the kernel what you plan to do with a memory range. Some advice is purely a hint; some is binding.

#table(columns: (auto, 1fr),
  [*Advice*], [*Effect*],
  [`MADV_WILLNEED`], [Prefetch from disk now (page cache populates asynchronously).],
  [`MADV_DONTNEED`], [*Binding.* Discard the pages immediately. Next access re-faults from zero (anon) or disk (file).],
  [`MADV_FREE`], [Pages may be discarded later if memory pressure rises; otherwise retained. Faster than `DONTNEED` for malloc-frees.],
  [`MADV_HUGEPAGE` / `MADV_NOHUGEPAGE`], [Toggle THP eligibility for the range.],
  [`MADV_COLD` / `MADV_PAGEOUT`], [Mark pages as cold; `PAGEOUT` actively writes them to swap.],
  [`MADV_RANDOM` / `MADV_SEQUENTIAL`], [Adjust readahead behavior on file mappings.],
  [`MADV_DONTFORK` / `MADV_WIPEONFORK`], [Control fork inheritance; useful for secrets and DMA buffers.],
)

*Performance idiom:* `glibc` and `tcmalloc` use `MADV_FREE` to return memory to the kernel without paying the fault cost on re-use. `MADV_DONTNEED` is correct but ~10× slower in microbenchmarks; `MADV_FREE` returns instantly and only pays the cost if the kernel actually reclaims.

== Practical Recipes

*Database-style file mapping:*

```c
int fd = open(path, O_RDONLY);
struct stat st; fstat(fd, &st);
void *p = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
madvise(p, st.st_size, MADV_RANDOM);   // disable readahead for random access
```

*Pre-fault a hot region:*

```c
void *p = mmap(NULL, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
mlock(p, size);                         // also pin in RAM
madvise(p, size, MADV_HUGEPAGE);
```

*DPDK-style 1 GiB hugepage pool:* see hugetlbfs section above.

== Diagnostics

```
cat /proc/<pid>/maps             # VMA list
cat /proc/<pid>/smaps            # detailed per-VMA: RSS, PSS, dirty, hugepage residency
cat /proc/<pid>/numa_maps        # which NUMA node each VMA's pages live on
cat /proc/meminfo                # system-wide: HugePages_Total, AnonHugePages, etc.
cat /proc/buddyinfo              # free-page fragmentation per order
```

`pmap -X <pid>` is a friendlier `smaps`. `/proc/<pid>/pagemap` lets you decode physical addresses if you have root.

*See also:* _Virtual Memory (CPU Architecture volume)_ (TLB, page tables, address translation hardware), _Zero-Copy Networking (Networking volume)_ (mmap-based zero-copy I/O patterns), _Advanced Algorithms in Modern Systems (Coding volume)_ (custom allocators on top of mmap).

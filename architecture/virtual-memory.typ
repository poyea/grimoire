= Virtual Memory

Virtual memory provides memory abstraction: each process sees isolated, contiguous address space despite physical memory being fragmented and shared. The MMU (Memory Management Unit) translates virtual addresses to physical addresses using page tables.

*See also:* Cache Hierarchy (for physical address caching), Memory System (for DRAM organization), CPU Fundamentals (for TLB as specialized cache)

== Virtual vs Physical Addresses

Modern architectures use 48-bit virtual address spaces on both x86-64 and ARM64, providing 256 TB of addressable memory per process. Physical address spaces typically range from 40 to 48 bits, supporting 1 to 256 TB of actual RAM. The Memory Management Unit (MMU) translates Virtual Addresses (VA) to Physical Addresses (PA) through a page table walk.

```
Virtual Address (48-bit):
├─────────────┼───────────┼───────┤
│  VPN (36)   │  Offset (12)      │  4 KB pages
└─────────────┴───────────────────┘

Physical Address (40-bit):
├─────────────┼───────────────────┤
│  PFN (28)   │  Offset (12)      │
└─────────────┴───────────────────┘

VPN = Virtual Page Number
PFN = Physical Frame Number
Offset = Position within page (copied directly)
```

== Page Table Structure

*4-level page table (x86-64):*

```
Virtual Address (48-bit):
├────┬────┬────┬────┬──────────┤
│ L4 │ L3 │ L2 │ L1 │ Offset   │
│ 9b │ 9b │ 9b │ 9b │   12b    │
└────┴────┴────┴────┴──────────┘

Level 4 (PML4): 512 entries, covers 256 TB
Level 3 (PDP):  512 entries, covers 512 GB
Level 2 (PD):   512 entries, covers 1 GB
Level 1 (PT):   512 entries, covers 2 MB

Each entry: 64-bit (8 bytes)
Page table size per level: 512 × 8 = 4 KB (fits in one page!)
```

*Page table entry (PTE):*

```
63         52 51        12 11    9 8 7 6 5 4 3 2 1 0
├───────────┼─────────────┼───────┬─┬─┬─┬─┬─┬─┬─┬─┬─┤
│  Reserved │     PFN     │  Avail│G│P│D│A│P│P│U│R│P│
│           │             │       │ │A│ │ │C│W│/│/│ │
│           │             │       │ │T│ │ │D│T│S│W│ │
└───────────┴─────────────┴───────┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

P (Present): 1 = page in memory, 0 = page fault
R/W (Read/Write): 0 = read-only, 1 = writable
U/S (User/Supervisor): 0 = kernel only, 1 = user accessible
PWT (Page Write-Through): Cache write policy
PCD (Page Cache Disable): 1 = uncached (MMIO)
A (Accessed): Set by CPU on read/write (for LRU)
D (Dirty): Set by CPU on write (needs writeback)
PAT (Page Attribute Table): Memory type
G (Global): Don't flush from TLB on CR3 change
PFN (Physical Frame Number): Bits 51-12 of physical address
```

== Page Table Walk

*Hardware walk (4 DRAM accesses for 4-level table):*

```
1. Read CR3 register → PML4 base address
2. Load PML4[L4_index] → PDP address      (DRAM access 1)
3. Load PDP[L3_index] → PD address        (DRAM access 2)
4. Load PD[L2_index] → PT address         (DRAM access 3)
5. Load PT[L1_index] → PTE (contains PFN) (DRAM access 4)
6. Combine PFN + offset → Physical address

Latency: 4 × 200 cycles = 800 cycles (without TLB!)
```

*Example:* Virtual address 0x0000_7f8a_b000_1234

```
VA: 0000 0000 0000 0000 0111 1111 1000 1010 1011 0000 0000 0001 0010 0011 0100
    └──────────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └──────────────┘
       Reserved       L4=255    L3=138     L2=176     L1=1        Offset=0x234

1. PML4[255] → PDP at 0x10000000
2. PDP[138]  → PD at 0x20000000
3. PD[176]   → PT at 0x30000000
4. PT[1]     → PFN = 0x40000
5. PA = 0x40000 << 12 | 0x234 = 0x40000234
```

== TLB (Translation Lookaside Buffer)

Page table walks are prohibitively expensive at 800 cycles, making caching essential. The Translation Lookaside Buffer (TLB) caches virtual to physical address translations, dramatically reducing translation overhead.

The TLB structure on x86-64 processors like Intel Skylake includes specialized components. The L1 Data TLB (DTLB) holds 64 entries in a 4-way associative organization for 4 KB pages, with an additional 32 entries for larger 2 MB/4 MB/1 GB pages. The L1 Instruction TLB (ITLB) provides 128 entries in an 8-way associative structure for 4 KB instruction pages. The shared L2 TLB contains 1536 entries in a 12-way associative organization, supporting all page sizes.

```
L1 DTLB (Data):     64 entries, 4-way associative, 4 KB pages
L1 DTLB:            32 entries, 4-way associative, 2 MB/4 MB/1 GB pages
L1 ITLB (Instruction): 128 entries, 8-way associative, 4 KB pages
L2 TLB (Shared):    1536 entries, 12-way associative, all page sizes

[Intel Skylake microarchitecture]
```

*TLB lookup (parallel with cache):*

```
1. Virtual address → split into VPN + offset
2. TLB lookup: Compare VPN against all TLB entries in parallel
   - Hit: Return PFN immediately (0 extra cycles)
   - Miss: Trigger page table walk (hardware or software)
3. Physical address = PFN | offset
4. Cache lookup using physical address
```

*TLB coverage:*

```
4 KB pages:   64 entries × 4 KB = 256 KB coverage (L1 DTLB)
2 MB pages:   32 entries × 2 MB = 64 MB coverage
1 GB pages:   32 entries × 1 GB = 32 GB coverage

For 8 GB working set:
- 4 KB pages: Need 2M entries → TLB miss rate ~99.9%
- 2 MB pages: Need 4K entries → TLB miss rate ~75%
- 1 GB pages: Need 8 entries → TLB miss rate ~0% (fits in TLB)
```

*TLB miss penalty:*

```
TLB hit:  0 cycles (translation cached)
TLB miss: 20-100 cycles (page walk latency)
          - 4-level walk: 4 DRAM accesses
          - If page tables cached in L3: ~40-80 cycles
          - If page tables in DRAM: ~200 cycles per level = 800 cycles
```

== Huge Pages

TLB coverage becomes insufficient for large working sets, where the limited number of TLB entries cannot cover all actively used pages. Larger pages provide a solution by increasing the memory range covered by each TLB entry.

Two-megabyte pages on x86-64 skip the L1 page table level, reducing the page table walk from 4 levels to 3. With 32 TLB entries, 2 MB pages provide 64 MB of coverage, which is 512× better than 4 KB pages. One-gigabyte pages skip both the L1 and L2 page table levels, reducing the walk to just 2 levels. With 32 TLB entries, 1 GB pages cover 32 GB of memory, providing 262,144× better coverage than 4 KB pages.

*Enabling huge pages (Linux):*

```bash
# Transparent Huge Pages (THP) - automatic
echo always > /sys/kernel/mm/transparent_hugepage/enabled

# Explicit huge pages
echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

# Mount hugetlbfs
mount -t hugetlbfs none /mnt/hugepages
```

*Using huge pages in code:*

```cpp
#include <sys/mman.h>

// Anonymous 2 MB pages
void* ptr = mmap(NULL, 2*1024*1024, PROT_READ|PROT_WRITE,
                 MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB, -1, 0);

// Or allocate normally and let THP promote
void* ptr = mmap(NULL, 100*1024*1024, PROT_READ|PROT_WRITE,
                 MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
madvise(ptr, 100*1024*1024, MADV_HUGEPAGE);  // Hint to use huge pages
```

*Performance impact:*

```
Workload: Random access to 8 GB array
4 KB pages:  TLB miss rate ~99%, latency ~50% higher
2 MB pages:  TLB miss rate ~5%, latency baseline
1 GB pages:  TLB miss rate <1%, latency 10% better

[Measured on Intel Xeon with 1536-entry L2 TLB]
```

*Tradeoffs:*
- *Pro:* Fewer TLB misses, fewer page table levels
- *Con:* Internal fragmentation (waste memory), slower allocation

== Page Faults

Page faults occur when the CPU attempts to access memory that is not currently mapped. There are three types, each with different causes and costs.

A minor fault (soft fault) occurs when a page exists in RAM but is not yet mapped in the page tables, typically due to lazy allocation. When `mmap` allocates memory with `MAP_ANONYMOUS`, the page tables are created but physical pages are not allocated until first access. The first write triggers a minor fault, where the kernel allocates a physical page, updates the page table entry, and resumes execution. This costs 5000-20000 cycles (approximately 2-5 μs).

```cpp
char* ptr = mmap(NULL, 1GB, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);
// Page tables allocated, but physical pages NOT allocated

ptr[0] = 1;  // First write: minor fault
// Kernel allocates physical page, updates PTE, resumes
// Cost: 5000-20000 cycles (~2-5 μs)
```

A major fault (hard fault) occurs when the page resides on disk, either swapped out or file-backed. When memory-mapping a file, the first access triggers a major fault where the kernel must read the page from disk. SSD latency is approximately 100 μs, while HDD latency ranges from 5-10 ms, costing over 1,000,000 cycles.

```cpp
// mmap file
int fd = open("data.bin", O_RDONLY);
char* data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);

char c = data[0];  // First access: major fault
// Kernel reads page from disk (SSD: ~100 μs, HDD: ~5-10 ms)
// Cost: 1,000,000+ cycles
```

An invalid access occurs when the page has no valid mapping, resulting in a segmentation fault (SIGSEGV). Dereferencing a null pointer causes a page fault with the Present bit cleared, and since no valid mapping exists, the kernel sends SIGSEGV to the process.

```cpp
char* ptr = NULL;
*ptr = 1;  // Page fault, P bit = 0, no valid mapping → SIGSEGV
```

*Page fault handler (simplified):*

```
1. CPU: Page fault exception → save registers, switch to kernel
2. Kernel: Determine fault type (look up VMA - Virtual Memory Area)
   - No VMA: Invalid access → SIGSEGV
   - VMA exists, page not present:
     - Anonymous: Allocate page (minor fault)
     - File-backed: Load from disk (major fault)
     - Swapped: Load from swap (major fault)
3. Update PTE (set P bit, set PFN)
4. Return to user mode, retry instruction
```

*Copy-on-write (COW):*

```cpp
// fork() creates child process
pid_t pid = fork();

// Parent and child share physical pages (read-only)
// First write triggers COW:
data[0] = 1;
// 1. Page fault (PTE marked read-only)
// 2. Kernel allocates new physical page
// 3. Copy old page to new page
// 4. Update PTE to point to new page, set writable
// 5. Resume
```

*Performance:* COW makes fork() cheap (copy PTEs, not pages). Write pays cost lazily.

== Page Table Management

*CR3 register:* Points to PML4 base (top-level page table).

```asm
; Switch address space (context switch)
mov rax, [new_cr3]
mov cr3, rax         ; Flush TLB (except global pages)

; TLB flush overhead: ~100 cycles + subsequent TLB misses
```

*TLB shootdown:* Multicore problem.

```
Core 0: Unmaps page
1. Invalidate local TLB
2. Send IPI (Inter-Processor Interrupt) to all cores
3. Wait for ACK

Core 1..N: Receive IPI
1. Invalidate TLB entries for unmapped page
2. Send ACK

Cost: 1000-5000 cycles depending on core count
```

*PCID (Process Context Identifier):* Tag TLB entries with address space ID.

```
Without PCID: Context switch flushes entire TLB
With PCID: TLB entries tagged with PCID → no flush needed

// Set PCID in CR3
mov rax, [page_table_base]
or  rax, [pcid_value]  ; PCID in bits 11:0
mov cr3, rax           ; Switch address space, keep TLB entries
```

*Performance:* PCID reduces context switch overhead 20-50% [Ahn et al. 2012].

== INVLPG and TLB Invalidation

*INVLPG instruction:* Invalidate single TLB entry.

```asm
invlpg [rax]  ; Invalidate TLB entry for virtual address in rax
; Cost: ~50 cycles
```

*Bulk invalidation:*

```asm
mov cr3, rax  ; Reload CR3 → flush entire TLB
; Cost: ~100 cycles + cache misses on subsequent accesses
```

*When to invalidate:*
- After changing PTE (e.g., mprotect, munmap)
- After freeing physical page

*Lazy TLB shootdown:* Defer remote TLB invalidation to reduce IPI overhead [Black et al. 1989].

== Memory Types and Caching

*PAT (Page Attribute Table):* Control caching per page.

*Memory types [Intel SDM]:*

```
UC (Uncacheable):        No caching, used for MMIO
WC (Write-Combining):    Buffered writes, no read caching (framebuffers)
WT (Write-Through):      Write immediately to memory, read cached
WP (Write-Protected):    Cached, writes cause fault
WB (Write-Back):         Fully cached (default for normal memory)
```

*Setting memory type:*

```cpp
// Via PTE bits (PCD, PWT, PAT)
pte |= (1 << 3);  // PWT = 1
pte |= (1 << 4);  // PCD = 1
// Combination determines memory type via PAT MSR

// Or via mmap flags
void* ptr = mmap(NULL, size, PROT_READ|PROT_WRITE,
                 MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
madvise(ptr, size, MADV_DONTFORK);  // Don't inherit on fork
```

== NUMA (Non-Uniform Memory Access)

*Multi-socket systems:* Each CPU has local memory.

```
┌───────────┐          ┌───────────┐
│  CPU 0    │          │  CPU 1    │
│  ├─L1/L2  │          │  ├─L1/L2  │
│  └─L3     │          │  └─L3     │
└─────┬─────┘          └─────┬─────┘
      │                      │
  ┌───┴────┐            ┌───┴────┐
  │ Memory │◀──────────▶│ Memory │
  │ (Node 0)│  QPI/IF   │ (Node 1)│
  └────────┘            └────────┘

Local access (Node 0 → Memory 0): ~200 cycles
Remote access (Node 0 → Memory 1): ~300-400 cycles (1.5-2x slower)
```

*NUMA policy:*

```cpp
#include <numaif.h>

// Bind memory to node
void* ptr = mmap(...);
mbind(ptr, size, MPOL_BIND, &nodemask, maxnode, 0);

// Allocate on local node
set_mempolicy(MPOL_PREFERRED, &nodemask, maxnode);
```

*Check NUMA topology:*

```bash
numactl --hardware

# Output:
# available: 2 nodes (0-1)
# node 0 cpus: 0 1 2 3
# node 1 cpus: 4 5 6 7
# node distances:
# node   0   1
#   0:  10  21
#   1:  21  10
```

*Performance impact:* Remote access 1.5-2x slower. Pin threads to same NUMA node as data.

== Debugging Virtual Memory Issues

*Detecting excessive page faults:*

```bash
# Monitor page faults
perf stat -e page-faults,minor-faults,major-faults ./program

# Minor faults > 100k/sec: Check memory allocation patterns
# Major faults > 100/sec: Likely swapping (very bad!)

# Real-time monitoring
watch -n 1 'cat /proc/$(pgrep program)/status | grep -E "VmSize|VmRSS|VmSwap"'

# Detailed fault analysis
perf record -e page-faults ./program
perf report  # Shows where faults occur
```

*Common issues and fixes:*

```c
// ISSUE 1: Lazy allocation causing faults on first access
char* buffer = mmap(NULL, 1GB, PROT_READ|PROT_WRITE,
                    MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);
// First access: minor fault for every page (256k faults for 1GB!)

// FIX: Pre-fault pages
madvise(buffer, 1GB, MADV_WILLNEED);  // Hint to pre-populate
// Or manually touch all pages:
for (size_t i = 0; i < 1GB; i += 4096)
    buffer[i] = 0;

// ISSUE 2: TLB thrashing
int* huge_array = malloc(4GB);  // 1M pages = TLB thrashing
for (int i = 0; i < 1000000000; i++)
    sum += huge_array[random() % 1000000000];  // TLB miss on every access

// FIX: Enable huge pages
echo always > /sys/kernel/mm/transparent_hugepage/enabled
// Or explicit huge pages
int* huge_array = mmap(NULL, 4GB, PROT_READ|PROT_WRITE,
                       MAP_ANONYMOUS|MAP_PRIVATE|MAP_HUGETLB, -1, 0);

// ISSUE 3: COW overhead in fork()
pid_t pid = fork();
if (pid == 0) {
    // Child process
    large_buffer[0] = 1;  // COW fault on first write to any page!
}

// FIX: Use threads instead of processes, or use vfork() for exec-only child
```

*Debugging TLB performance:*

```bash
# Measure TLB misses
perf stat -e dTLB-load-misses,dTLB-store-misses,iTLB-load-misses ./program

# Check page table walk cost
perf stat -e page_walker.walks,page_walker.cycles ./program
# High cycles/walk ratio indicates expensive walks (DRAM latency)

# Verify huge page usage
grep Huge /proc/meminfo
# HugePages_Total should be > 0
# HugePages_Free should be < HugePages_Total (some in use)

# Per-process huge page usage
cat /proc/$(pgrep program)/smaps | grep -E 'AnonHugePages|ShmemHugePages'
```

*Memory mapping debugging:*

```bash
# View process memory map
cat /proc/$(pgrep program)/maps

# Example output:
# 00400000-00500000 r-xp ... /path/to/program   # Code segment
# 00700000-00800000 rw-p ... [heap]             # Heap
# 7f0000000000-7f0010000000 rw-p ... [anon]     # mmap region
#
# Flags: r=read, w=write, x=execute, p=private, s=shared

# Check for memory leaks via mapping growth
watch -n 1 'cat /proc/$(pgrep program)/maps | wc -l'
# Growing count → memory leaks (unmapped regions)
```

== References

*Primary sources:*

Intel Corporation (2023). Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 3A: System Programming Guide. Chapter 4 (Paging).

AMD (2023). AMD64 Architecture Programmer's Manual, Volume 2: System Programming. Chapter 5 (Page Translation and Protection).

*Textbooks:*

Hennessy, J.L. & Patterson, D.A. (2017). Computer Architecture: A Quantitative Approach (6th ed.). Morgan Kaufmann. Appendix B.4 (Virtual Memory).

Tanenbaum, A.S. & Bos, H. (2014). Modern Operating Systems (4th ed.). Pearson. Chapter 3 (Memory Management).

*Research:*

Black, D.L., Rashid, R.F., Golub, D.B., Hill, C.R., & Baron, R.V. (1989). "Translation Lookaside Buffer Consistency: A Software Approach." ASPLOS III.

Ahn, J., Kwon, Y., Kim, J., & Kim, C. (2012). "A Study on the Effects of PCID on TLB Performance." International Journal of Computer Applications.

*Performance analysis:*

Drepper, U. (2007). "What Every Programmer Should Know About Memory." Red Hat, Inc. Section 4 (Virtual Memory).

Lameter, C. (2013). "NUMA (Non-Uniform Memory Access): An Overview." Linux Symposium.

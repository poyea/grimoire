#import "../template.typ": xref

= Low-Latency Systems <low-latency-systems>

Some systems are judged not by throughput or average latency but by how fast and how consistently they respond to a single event: a trading engine reacting to a market data packet, a game server, an audio pipeline, a robot's control loop, a real-time bidding service. Their budgets are measured in microseconds, and the enemy is less the average cost of the code than *jitter*: the occasional 50 microsecond pause from an interrupt, a page fault, a frequency change or a garbage collection. This chapter covers the latency costs of common operations, the sources of jitter in a general-purpose operating system, how to isolate cores and keep the hot path free of system calls, faults and allocation, lock-free handoff between threads with a C++ single-producer single-consumer queue, kernel-bypass networking, and how to measure latency distributions honestly with a C++ histogram.

*See also:* #xref("performance-engineering", "methodology", label: "Performance Methodology") (percentiles and coordinated omission), #xref("performance-engineering", "concurrency-performance", label: "Concurrency Performance") (tail latency and fan-out), #xref("networking", "kernel-bypass", label: "Kernel Bypass"), #xref("operating-systems", "scheduling-theory", label: "Scheduling Theory"), #xref("embedded-and-realtime", "timing-analysis", label: "Timing Analysis and WCET").

== The Budget

Approximate costs on a current server, which set what a microsecond-scale budget can afford:

#table(
  columns: 2,
  [*Operation*], [*Typical cost*],
  [L1 hit / L2 hit / L3 hit], [1 ns / 4 ns / 10 to 20 ns],
  [DRAM access (cache miss)], [80 to 120 ns],
  [Cross-core cache line transfer], [50 to 100 ns on one socket, more across sockets],
  [Uncontended atomic read-modify-write], [5 to 20 ns],
  [System call (simple)], [100 ns to 1 µs, more with mitigations for speculative-execution attacks],
  [Minor page fault], [About 1 µs],
  [Context switch including cache disturbance], [2 to 10 µs],
  [Thread wake-up from a blocking wait], [5 to 50 µs],
  [Exit from a deep CPU idle state], [Tens to over a hundred µs],
  [Kernel TCP or UDP round trip on one host], [10 to 30 µs],
  [Kernel-bypass network round trip between adjacent hosts], [2 to 5 µs],
  [Major page fault (disk)], [Milliseconds],
)

The table makes the design rules obvious. A hot path with a 5 µs budget cannot block and wake up, cannot take a page fault, cannot let its core sleep, and can afford only a handful of cache misses.

== Sources of Jitter

A general-purpose OS is optimised for throughput and fairness, and many of its mechanisms interrupt a running thread:

- *Scheduler preemption*: other threads, kernel threads and per-CPU work (RCU callbacks, timers, workqueues) run on the same core.
- *Interrupts*: network, disk and inter-processor interrupts (including TLB shootdowns when another thread changes memory mappings) stop the current thread for microseconds.
- *Timer ticks*: the periodic scheduler tick fires hundreds or thousands of times per second on every busy core.
- *Power management*: cores in deep idle states take tens of microseconds to wake; frequency changes stall execution briefly; a core that runs at lower frequency because its neighbours are busy (or because of heavy vector instructions on some processors) executes everything more slowly.
- *Memory management*: page faults on first touch, transparent huge page compaction, NUMA balancing migrating pages, and swapping.
- *System management interrupts* (SMIs): firmware code that the OS cannot see or prevent, sometimes lasting hundreds of microseconds.
- *Runtime effects*: garbage collection, JIT compilation, lazy class loading, and allocator housekeeping.
- *The application itself*: logging, a rarely taken code path that is cold in the caches, a lock held by a slow thread.

Each is rare, so averages hide them. They define the p99.9 and the maximum, which are the numbers that matter for these systems.

== Isolating the Hot Path

The standard Linux recipe dedicates cores to latency-critical threads and moves everything else away:

- *Isolate cores* from the general scheduler (`isolcpus`, or cpusets / cgroup partitions), and *pin* each critical thread to its own isolated core.
- *Stop the tick* on those cores with `nohz_full`, and move RCU callback processing elsewhere with `rcu_nocbs`.
- *Steer interrupts* away from isolated cores through `/proc/irq/*/smp_affinity` (and stop `irqbalance` from undoing it), except for the network queue the critical thread polls, which goes to that thread's core or a neighbour.
- *Pin the frequency*: disable deep C-states (`intel_idle.max_cstate`, `processor.max_cstate`, or `/dev/cpu_dma_latency`), use the performance governor, and consider disabling turbo for consistency rather than peak speed.
- *Disable SMT siblings* of critical cores, or leave them idle, so no other thread shares the core's execution units and L1 cache.
- *Respect NUMA*: keep the thread, its memory and the NIC on the same socket.

Isolated cores spend most of their time *busy polling*: spinning on a queue or a NIC ring rather than blocking. Polling burns a core at 100 percent, which is the price of never paying a wake-up.

== Keeping the Hot Path Clean

Inside the critical thread, the rule is: do nothing on the hot path that might touch the kernel or a slow path.

- *No system calls*: no blocking I/O, no `sleep`, no `gettimeofday` (on Linux it is usually served from the vDSO without a system call, but verify), no mutex that could block.
- *No page faults*: allocate and touch all memory at start-up, lock it with `mlockall(MCL_CURRENT | MCL_FUTURE)`, and use huge pages for large tables to reduce TLB misses. Disable transparent huge page compaction if it causes stalls.
- *No allocation*: pre-allocate object pools and ring buffers. General-purpose allocators have rare slow paths (refilling thread caches, returning memory to the OS) that show up in the tail.
- *No synchronous logging*: write fixed-size binary records to a lock-free ring buffer and let another thread format and write them.
- *Warm the caches*: a code path that runs once per minute is cold when it runs, missing in the instruction cache, data cache and branch predictor. Trading systems often run the hot path continuously on dummy inputs (discarding the results) so that the real event finds everything warm.
- *Keep data compact*: the state touched per event should fit in L1 and L2. Pointer-rich structures and hash tables with large entries spread it across many cache lines.

== Handing Work Between Threads

Threads still need to communicate, for example a network thread passing decoded messages to a strategy thread. Locks are ruled out: a thread holding a lock can be descheduled, and the other must wait for it. A *single-producer single-consumer* (SPSC) ring buffer needs no locks and no read-modify-write atomics, only loads and stores with acquire and release ordering.

```cpp
#include <array>
#include <atomic>
#include <cstddef>
#include <new>

template <class T, std::size_t Capacity>
class spsc_queue {
    static_assert((Capacity & (Capacity - 1)) == 0, "capacity must be a power of two");
    static constexpr std::size_t line = 64;

    // Producer-owned line: its index plus a cached copy of the consumer's index.
    struct alignas(line) producer_side {
        std::atomic<std::size_t> tail{0};
        std::size_t cached_head = 0;
    } prod_;
    // Consumer-owned line: its index plus a cached copy of the producer's index.
    struct alignas(line) consumer_side {
        std::atomic<std::size_t> head{0};
        std::size_t cached_tail = 0;
    } cons_;
    alignas(line) std::array<T, Capacity> slots_;

public:
    bool try_push(const T& value) {
        const std::size_t t = prod_.tail.load(std::memory_order_relaxed);
        if (t - prod_.cached_head == Capacity) {
            prod_.cached_head = cons_.head.load(std::memory_order_acquire);
            if (t - prod_.cached_head == Capacity) return false;   // full
        }
        slots_[t & (Capacity - 1)] = value;
        prod_.tail.store(t + 1, std::memory_order_release);       // publish the slot
        return true;
    }

    bool try_pop(T& out) {
        const std::size_t h = cons_.head.load(std::memory_order_relaxed);
        if (h == cons_.cached_tail) {
            cons_.cached_tail = prod_.tail.load(std::memory_order_acquire);
            if (h == cons_.cached_tail) return false;                // empty
        }
        out = slots_[h & (Capacity - 1)];
        cons_.head.store(h + 1, std::memory_order_release);        // free the slot
        return true;
    }
};
```

The design has three performance details. Each index is written by only one thread, so there is no contention, only cache line transfers when the other side reads it. Each side keeps a *cached* copy of the other side's index and rereads the shared atomic only when the cache says the queue is full or empty, so in steady state most operations touch no shared line at all. And producer and consumer state live on separate cache lines, avoiding false sharing. The release store on the index makes the slot's contents visible to the other thread before the index update, and the acquire load on the other side completes the synchronisation. Indices are free-running counters, and the subtraction `t - cached_head` remains correct after they wrap around, because unsigned arithmetic is modular.

A handoff through such a queue between two cores on one socket costs on the order of 50 to 100 ns, dominated by moving two cache lines. Multi-producer or multi-consumer queues need compare-and-swap and are slower; where possible, give each producer its own SPSC queue.

== Kernel-Bypass Networking

The kernel network stack is general and safe, and it costs several microseconds per packet in each direction: interrupts, system calls, copies, socket buffers and protocol processing. Kernel bypass moves the NIC's receive and transmit rings into the application's address space:

- *DPDK* gives user-space poll-mode drivers for many NICs and leaves the protocol stack to the application.
- *Vendor stacks* (Solarflare/AMD Onload and ef_vi, NVIDIA VMA) accelerate the standard socket API by intercepting it, or offer a lower-level interface for the fastest path.
- *AF_XDP* provides a fast path to user space through the kernel's XDP hook while keeping the kernel driver.
- *io_uring* with polling modes reduces system calls for kernel-stack I/O, a middle ground.

With bypass and busy polling, an application can receive a UDP packet, process it, and transmit a response in a few microseconds wire to wire. FPGA-based systems go further, handling the critical path in hardware in hundreds of nanoseconds.

== Measuring Latency Honestly

Latency distributions in these systems are heavy-tailed, so the summary must show the tail.

- *Record every event*, not samples, into a histogram with bounded relative error across many orders of magnitude, and report percentiles up to p99.99 and the maximum.
- *Timestamp at the right points*: hardware timestamps on the NIC measure what the outside world sees; timestamps inside the application miss queueing in the NIC and the kernel. On x86, the invariant TSC read with `rdtsc` costs about 20 cycles, is synchronised across cores on modern systems, and can be converted to nanoseconds with a calibrated factor.
- *Avoid coordinated omission*: a load generator that waits for each response before sending the next hides exactly the stalls being measured (#xref("performance-engineering", "methodology", label: "Performance Methodology")). Send at a fixed schedule and measure from the intended send time.
- *Run long enough* to capture rare events: a stall that happens once a minute needs hours of data to characterise.

A log-linear histogram, the structure behind HdrHistogram, records values with fixed relative precision in constant time and memory:

```cpp
#include <array>
#include <bit>
#include <cstdint>

// Values (for example nanoseconds) with about 1.6 percent relative precision:
// 64 linear sub-buckets per power of two.
class latency_histogram {
    static constexpr int sub_bits = 6;
    static constexpr std::uint64_t sub_count = 1u << sub_bits;
    std::array<std::uint64_t, 64 * sub_count> counts_{};
    std::uint64_t total_ = 0;

    static std::size_t index_of(std::uint64_t v) {
        if (v < sub_count) return static_cast<std::size_t>(v);
        const int msb = 63 - std::countl_zero(v);
        const int shift = msb - sub_bits;
        return static_cast<std::size_t>((shift + 1) * sub_count + ((v >> shift) & (sub_count - 1)));
    }

    static std::uint64_t lower_bound_of(std::size_t i) {
        if (i < sub_count) return i;
        const std::size_t shift = i / sub_count - 1;
        return (sub_count + i % sub_count) << shift;
    }

public:
    void record(std::uint64_t v) { ++counts_[index_of(v)]; ++total_; }

    // Smallest recorded bucket value at or above quantile q in [0, 1].
    std::uint64_t quantile(double q) const {
        const auto target = static_cast<std::uint64_t>(q * static_cast<double>(total_));
        std::uint64_t seen = 0;
        for (std::size_t i = 0; i < counts_.size(); ++i) {
            seen += counts_[i];
            if (seen > target) return lower_bound_of(i);
        }
        return 0;
    }
};
```

Recording is a count-leading-zeros instruction, a shift and an increment, cheap enough for every event on the hot path. Histograms from different threads or time intervals merge by adding counts, which makes them suitable for aggregation; averaging percentiles across hosts or intervals, by contrast, is meaningless.

== Pitfalls

- Optimising average latency while the p99.9 is set by interrupts, page faults or GC.
- Pinning threads without isolating the cores, so other work still runs there.
- Leaving deep C-states enabled on cores that wait for events.
- Logging, allocating or taking a mutex on the hot path "just once".
- Measuring with a closed-loop load generator and reporting percentiles that omit the stalls.
- Benchmarking on a warm, idle machine and deploying next to noisy neighbours on shared hardware.

== Further Reading

Dean, J. and Barroso, L. A. (2013). "The Tail at Scale." Communications of the ACM.

Li, J. et al. (2014). "Tales of the Tail: Hardware, OS, and Application-level Sources of Tail Latency." SoCC.

Tene, G. (2015). "How NOT to Measure Latency." Strange Loop talk, reference.

Belay, A. et al. (2014). "IX: A Protected Dataplane Operating System for High Throughput and Low Latency." OSDI.

Kaffes, K. et al. (2019). "Shinjuku: Preemptive Scheduling for µsecond-scale Tail Latency." NSDI.

Ousterhout, A. et al. (2019). "Shenango: Achieving High CPU Efficiency for Latency-sensitive Datacenter Workloads." NSDI.

Barroso, L. A. et al. (2017). "Attack of the Killer Microseconds." Communications of the ACM.

Linux Kernel Documentation (2024). "NO_HZ: Reducing Scheduling-Clock Ticks." kernel.org documentation.

DPDK Project (2024). "Data Plane Development Kit Programmer's Guide." DPDK documentation.

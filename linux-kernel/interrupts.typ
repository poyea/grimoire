= Interrupts and Bottom Halves

When a hardware device needs attention — a packet arrived, a disk completed a transfer, a timer expired — it raises an interrupt. The CPU stops whatever it was doing, transfers control to the kernel's interrupt handler, runs that handler with interrupts disabled, and resumes. This is fast for the device (microsecond response) but expensive in aggregate: every interrupt is a pipeline flush, a register save, and a forced preemption. At 10 GbE line rate (~14.8 Mpps for 64-byte frames), one interrupt per packet would saturate a CPU on overhead alone.

Linux's solution is a layered architecture: do the *minimum* work in the synchronous interrupt handler, and defer the rest to lower-priority contexts (softirqs, tasklets, workqueues, threaded IRQs). For high-rate devices, switch entirely to *polling* under load (NAPI) and let interrupts only kick the polling loop into motion.

== The Two-Half Model

Every interrupt-driven driver in Linux splits its work between:

- *Top half* (the hard IRQ handler): runs synchronously when the interrupt fires, with that IRQ line masked. Must be short — ideally a few hundred nanoseconds at most. Acknowledges the device, stashes the work to do, schedules the bottom half.
- *Bottom half*: runs later (usually within microseconds, on the same CPU), with interrupts enabled. Does the actual processing.

The top half is the only context allowed in real-IRQ context. It cannot sleep, cannot acquire mutex/semaphore (only spinlocks with `_irqsave`), and cannot perform anything that might block.

== Registering an Interrupt Handler

```c
#include <linux/interrupt.h>

static irqreturn_t my_isr(int irq, void *dev) {
    struct my_dev *d = dev;
    u32 status = readl(d->base + REG_STATUS);
    if (!(status & INT_PENDING))
        return IRQ_NONE;        // not ours; line was shared
    writel(status, d->base + REG_STATUS);   // ack
    tasklet_schedule(&d->rx_tasklet);
    return IRQ_HANDLED;
}

request_irq(d->irq, my_isr, IRQF_SHARED, "my_dev", d);
```

`IRQF_SHARED` allows multiple drivers on the same IRQ line (PCI INTx); each must check whether the interrupt is theirs. MSI/MSI-X interrupts are non-shared by construction and get their own vectors.

== Bottom Half Mechanisms

Four flavors, each with different trade-offs.

#table(columns: (auto, 1fr, 1fr),
  [*Mechanism*], [*Context*], [*Use case*],
  [Softirq], [Atomic, fixed set defined at compile time], [Network, block IO, RCU, timers — kernel-internal only],
  [Tasklet], [Atomic, dynamic, runs on the CPU that scheduled it; built on softirqs.], [Driver-side deferred work. Deprecated for new code.],
  [Workqueue], [Process context (kthread). Can sleep, take mutexes, do IO.], [Anything that needs to block.],
  [Threaded IRQ], [Process context, dedicated kthread per IRQ.], [RT-friendly; predictable preemption.],
)

=== Softirqs

A fixed array of softirq vectors, defined statically:

```
HI_SOFTIRQ          high-priority tasklets
TIMER_SOFTIRQ       timer wheel
NET_TX_SOFTIRQ      network transmit
NET_RX_SOFTIRQ      network receive (NAPI poll)
BLOCK_SOFTIRQ       block-IO completion
IRQ_POLL_SOFTIRQ    irq_poll (storage)
TASKLET_SOFTIRQ     normal tasklets
SCHED_SOFTIRQ       scheduler load balancing
HRTIMER_SOFTIRQ     hrtimers (4.16+ moved expiry from hardirq to softirq)
RCU_SOFTIRQ         RCU callbacks
```

A driver can't add a new softirq — the set is fixed. Softirqs run after the top half exits, on the *same* CPU, with interrupts enabled but preemption disabled. If a softirq runs too long, it is handed off to the per-CPU `ksoftirqd/N` kthread to avoid starving user tasks.

=== Tasklets

Built on top of `TASKLET_SOFTIRQ` and `HI_SOFTIRQ`. A driver can dynamically register a tasklet handler and schedule it:

```c
DECLARE_TASKLET(my_tasklet, my_tasklet_fn);
tasklet_schedule(&my_tasklet);
```

Tasklet guarantee: a given tasklet runs on at most one CPU at a time, on the CPU that scheduled it. Convenient — no locking against yourself — but a serialization bottleneck on multi-queue hardware.

Tasklets are *deprecated for new drivers* (kernel docs explicitly say so since ~5.0). Replacements are workqueues for sleepable work and `BH_WORK` / `irq_poll` for atomic deferred processing.

=== Workqueues

A workqueue queues `struct work_struct` items into a kthread pool that runs them in process context. Can sleep, can call any kernel API, can take any lock.

```c
static void my_work_fn(struct work_struct *w) {
    struct my_dev *d = container_of(w, struct my_dev, work);
    // sleeping operations are fine here
    mutex_lock(&d->lock);
    do_long_thing(d);
    mutex_unlock(&d->lock);
}

INIT_WORK(&d->work, my_work_fn);
schedule_work(&d->work);                    // system_wq
```

The default `system_wq` is shared across the system. For your own pool with controlled concurrency:

```c
struct workqueue_struct *wq = alloc_workqueue("mydev",
    WQ_UNBOUND | WQ_HIGHPRI, 0);
queue_work(wq, &d->work);
```

`WQ_UNBOUND` decouples worker threads from any specific CPU. `WQ_HIGHPRI` gives them higher nice. `WQ_MEM_RECLAIM` reserves a rescuer thread so the queue makes forward progress even under memory pressure (essential for storage drivers).

=== Threaded IRQs

Instead of running the top half in IRQ context, the kernel can run a thin top-half stub and run the *real* handler in a dedicated per-IRQ kthread. The kthread is `SCHED_FIFO` priority 50 by default and is fully preemptable, schedulable, and bound by affinity rules — unlike a real IRQ.

```c
request_threaded_irq(irq,
    primary_handler,        // hard-IRQ context, can be NULL → kernel uses irq_default_primary_handler
    threaded_handler,       // process context, can sleep
    IRQF_ONESHOT,
    "my_dev", dev);
```

`IRQF_ONESHOT` keeps the IRQ masked until the threaded handler completes — required when there is no primary handler.

This is the foundation of the *PREEMPT_RT* patch set's interrupt model: nearly all IRQs become threaded, so the only non-preemptible code is a fast acknowledgment in the primary handler. Without RT, drivers that want predictable latency under load (audio, industrial control) can opt into threaded IRQs selectively.

== NAPI: Polling for High-Rate Networking

At 1 Gbps, one interrupt per packet is feasible (~80 kpps for 1500-byte frames). At 10/25/100 Gbps, it isn't — the per-IRQ overhead would saturate a core. *NAPI* (New API, named in 2002) is the kernel's polling model.

The flow:

1. Packets arrive → NIC fires an interrupt.
2. The driver's hard-IRQ handler *masks* further interrupts on this queue and schedules a `NET_RX` softirq.
3. The softirq runs `napi_poll`: drains up to `weight` packets (typically 64) from the ring.
4. If the ring drained fully (no more packets ready), re-enable interrupts. Done.
5. If `weight` packets were processed but more remain, do *not* re-enable interrupts. The softirq will reschedule itself; or if it's been running too long, hand off to `ksoftirqd/N`.

Result: under load, the system spends most of its time polling, not entering interrupt context. Under low load, the latency of a single packet remains the cost of one interrupt. The transition is automatic.

Driver skeleton (post-2014 NAPI API):

```c
netif_napi_add(netdev, &q->napi, my_poll);

static int my_poll(struct napi_struct *n, int budget) {
    int work = 0;
    while (work < budget && (skb = drain_ring(...))) {
        netif_receive_skb(skb);
        work++;
    }
    if (work < budget) {
        napi_complete_done(n, work);
        enable_irq(...);
    }
    return work;
}
```

*busy_read / busy_poll* (since 3.13): a userspace socket can spin in the kernel polling NAPI directly, avoiding the softirq round trip for ultra-low-latency reads. `setsockopt(SO_BUSY_POLL, &usec, ...)`. Later kernels added `SO_PREFER_BUSY_POLL` and integration with `epoll`/`io_uring`.

== Interrupt Coalescing and Moderation

Even with NAPI, *coalescing* helps: the NIC waits a short time or for several packets before firing the interrupt, amortizing the IRQ cost.

```
ethtool -c eth0                    # current settings
ethtool -C eth0 rx-usecs 50 rx-frames 32
ethtool -C eth0 adaptive-rx on     # NIC auto-tunes based on load
```

Tradeoff: higher `rx-usecs` = fewer interrupts = higher throughput but higher latency at the tail. For latency-bound workloads (RDMA, financial), set `rx-usecs 0` and rely on NAPI. For pure throughput, let adaptive moderation do its thing.

== MSI / MSI-X

Modern PCIe devices use *Message Signaled Interrupts*. Instead of toggling a physical INTx pin, the device DMAs a small message (an interrupt vector) to a special address. The CPU's local APIC delivers it as the corresponding IRQ.

- *MSI:* up to 32 vectors per device.
- *MSI-X:* up to 2048 vectors per device, each independently maskable and routable to a specific CPU.

A 100 GbE NIC typically uses 32-128 MSI-X vectors — one per queue. Each queue pair (RX/TX) gets its own vector and IRQ, which lets you steer queue 0 to CPU 0, queue 1 to CPU 1, etc., for a perfectly parallel receive path. See *cpu-affinity.typ* for the IRQ-pinning recipe.

== Per-CPU IRQ Cost in the Real World

Approximate costs on modern x86-64 (Skylake-era and later):

#table(columns: (auto, 1fr),
  [*Operation*], [*Cycles*],
  [Bare interrupt entry (no mitigations)], [~150-300],
  [With KPTI + Spectre mitigations], [~400-700],
  [+ saving full register state to kernel stack], [~50-100],
  [+ APIC EOI write], [~20-100],
  [Total: hard IRQ overhead per event], [~600-1000 cycles ≈ 200-300 ns],
)

That's *before* the handler does anything useful. At 14 Mpps (10G line rate, 64B frames), one IRQ per packet would consume the entire CPU just on overhead. Hence NAPI.

== /proc/interrupts

The single most useful interrupt diagnostic.

```
$ cat /proc/interrupts
            CPU0       CPU1       CPU2       CPU3
   0:        125          0          0          0   IO-APIC    2-edge      timer
   8:          1          0          0          0   IO-APIC    8-edge      rtc0
   9:          0          0          0          0   IO-APIC    9-fasteoi   acpi
  24:    8127345          0          0          0   PCI-MSI 1048576-edge   ahci[0000:00:17.0]
  25:          0  211451623          0          0   PCI-MSI 524288-edge    eth0-rx-0
  26:          0          0  208314129          0   PCI-MSI 524289-edge    eth0-rx-1
  27:          0          0          0  202118732   PCI-MSI 524290-edge    eth0-rx-2
```

Per-CPU columns. `eth0-rx-N` going to `CPUN` is the goal — RSS is steering correctly. If all NIC IRQs land on CPU 0, your `irqbalance` is misbehaving or you have RFS/RPS misconfigured.

`grep -E '^(\s*[0-9]+:)' /proc/interrupts | awk '{ print $1, $NF }'` is a quick "what hardware is interrupting?" listing.

== Real-Time Considerations

The `PREEMPT_RT` patch set (mainlined progressively over 5.x and 6.x) changes the interrupt model in two key ways:

- *Threaded by default.* Most IRQ handlers run in their own kthread, schedulable at user-controlled priority.
- *Spinlocks are sleepable.* Most kernel `spinlock_t` becomes `rt_mutex`-backed, with priority inheritance. The few remaining truly atomic spinlocks are `raw_spinlock_t`.

Combined, the kernel becomes preemptible nearly everywhere, with bounded preemption-disabled regions (raw spinlocks only). With proper isolation (`isolcpus`, IRQ pinning) and `SCHED_FIFO`/`SCHED_DEADLINE`, achievable worst-case scheduling latency on RT-Linux is in the tens of microseconds — sufficient for hard real-time control loops.

For non-RT kernels, threaded IRQs are still selectively useful: opt in with `IRQF_NO_THREAD` cleared and `request_threaded_irq`, then bump the kthread's priority via `chrt -f -p 80 $(pgrep irq/24)`.

== Diagnostics Checklist

```
cat /proc/interrupts                                # IRQ counts per CPU
cat /proc/softirqs                                  # softirq counts per CPU
mpstat -P ALL 1                                     # %irq, %soft per CPU
perf top -e irq:irq_handler_entry                   # which IRQs are hottest
bpftrace -e 'tracepoint:irq:irq_handler_entry { @[args->name] = count(); }'
bpftrace -e 'tracepoint:irq:softirq_entry { @[args->vec] = hist(nsecs); }'
ethtool -S eth0 | grep -E 'rx_dropped|tx_dropped|rx_no_buffer'
```

A high `%soft` on a CPU that wasn't expected to be doing network work means RFS/RPS is steering work there; check `/sys/class/net/eth0/queues/rx-N/rps_cpus`.

*See also:* _cpu-affinity.typ_ (steering IRQs away from isolated cores), _scheduler.typ_ (threaded IRQ priority, RT throttling), _Kernel Bypass (Networking volume)_ (DPDK and AF_XDP bypass the IRQ path entirely).

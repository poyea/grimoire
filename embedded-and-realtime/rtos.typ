#import "../template.typ": xref

= Real-Time Operating Systems <rtos>

A real-time operating system ($"RTOS"$) supplies a preemptive priority-based scheduler, deterministic IPC primitives, and a memory-frugal kernel — typically a few kilobytes of code, no MMU dependency, and no dynamic allocation in the critical path. This chapter dissects the three dominant designs: *FreeRTOS* (minimal, source-available, AWS-stewarded), *Zephyr* (Apache-licensed, Linux-style infrastructure, broad driver model), and *PREEMPT_RT Linux* (full POSIX, real-time only after invasive patches).

*See also:* #xref("operating-systems", "real-time-scheduling", label: "Real-Time Scheduling"), #xref("embedded-and-realtime", "peripherals-and-drivers", label: "Peripherals and Drivers"), #xref("embedded-and-realtime", "mcus-and-soc", label: "Microcontrollers and Systems-on-Chip"), #xref("linux-kernel", "scheduler", label: "The Scheduler") (implementation), #xref("operating-systems", "processes-and-threads", label: "Processes and Threads") (operating systems)

== What Makes an OS "Real-Time"

A real-time system must guarantee a *worst-case* response, not an average. The OS contribution to that bound is the sum of:

1. *Interrupt latency* — time from IRQ assertion to first instruction of the handler.
2. *Interrupt nesting / disabling* — longest path with IRQs masked.
3. *Scheduling latency* — time from a task being made runnable to it executing.
4. *Context-switch cost* — register save + restore + cache effects.
5. *IPC latency* — semaphore give, queue send, event flag wakeup.

A general-purpose OS optimizes throughput and average latency, freely disabling preemption in long syscalls. An $"RTOS"$ trades throughput for *bounded* and *short* critical sections, and exposes those bounds in datasheets so $"WCET"$ analysis is possible.

== FreeRTOS

FreeRTOS started in 2003 as a single-file scheduler for 8-bit AVRs; it now ports to 40+ architectures and ships as the kernel of Amazon FreeRTOS. The core is ~9 000 lines of C, MIT-licensed.

=== Tasks and the Scheduler

Each task is a function with an infinite loop and a stack of user-specified size. Priorities are integers 0..`configMAX_PRIORITIES-1`; the scheduler picks the highest-priority ready task, round-robining peers if `configUSE_TIME_SLICING` is set.

```c
void blinky(void *arg) {
    TickType_t last = xTaskGetTickCount();
    for (;;) {
        gpio_toggle(LED_PIN);
        vTaskDelayUntil(&last, pdMS_TO_TICKS(500));
    }
}

int main(void) {
    xTaskCreate(blinky, "blink", 128, NULL, tskIDLE_PRIORITY + 1, NULL);
    vTaskStartScheduler();
    for (;;);                       // never reached
}
```

`vTaskDelayUntil` gives jitter-free periodic execution: the wake time is computed from the previous wake time, not from "now after the work".

=== Context Switch on Cortex-M

FreeRTOS implements the scheduler in `port.c` per architecture. On Cortex-M the trick is *PendSV*: the lowest-priority exception, raised by a write to `ICSR.PENDSVSET`, which performs the context switch tail-chained after the triggering ISR.

```c
// PendSV handler — performs the context switch (Cortex-M4, no FPU lazy stack)
__attribute__((naked)) void PendSV_Handler(void) {
    __asm volatile (
        "mrs   r0, psp                 \n" // current PSP
        "isb                           \n"
        "ldr   r3, =pxCurrentTCB       \n"
        "ldr   r2, [r3]                \n"
        "stmdb r0!, {r4-r11}           \n" // save R4-R11
        "str   r0, [r2]                \n" // TCB->pxTopOfStack = new PSP
        "cpsid i                       \n"
        "bl    vTaskSwitchContext      \n" // pick next task
        "cpsie i                       \n"
        "ldr   r3, =pxCurrentTCB       \n"
        "ldr   r2, [r3]                \n"
        "ldr   r0, [r2]                \n" // new top of stack
        "ldmia r0!, {r4-r11}           \n"
        "msr   psp, r0                 \n"
        "isb                           \n"
        "bx    lr                      \n"
    );
}
```

The hardware has already stacked `R0`–`R3`, `R12`, `LR`, `PC`, `xPSR` on entry; `PendSV` only needs to save the non-volatile registers. Total switch cost on a Cortex-M4 at 168 MHz: \~80 cycles, \~500 ns.

=== Synchronization Primitives

FreeRTOS exposes binary semaphores, counting semaphores, mutexes (with priority inheritance), recursive mutexes, queues (the all-purpose IPC), stream/message buffers (single-reader, single-writer, lock-free), event groups, and direct-to-task notifications (the fastest path: \~45 % faster than a semaphore by skipping the queue object).

```c
SemaphoreHandle_t lock = xSemaphoreCreateMutex();   // priority inheritance

void worker(void *arg) {
    for (;;) {
        if (xSemaphoreTake(lock, pdMS_TO_TICKS(10)) == pdTRUE) {
            shared_state_update();
            xSemaphoreGive(lock);
        }
    }
}
```

=== Memory Models

Five `heap_n.c` implementations ship: `heap_1` (no free), `heap_2` (free, no coalesce), `heap_3` (thin wrapper over `malloc`), `heap_4` (free + coalesce, best general-purpose), `heap_5` (multi-region for non-contiguous SRAM). Hard real-time systems often pick `heap_1` and create everything at boot — no fragmentation is possible if nothing is ever freed.

== Zephyr

Zephyr originated at Wind River as the Rocket kernel, was donated to the Linux Foundation in 2017, and now spans 450+ supported boards. It targets the same $"MCU"$ class as FreeRTOS but with a Linux-style philosophy: device tree, Kconfig, west build tool, full driver model, networking, Bluetooth, USB.

=== Threads, Fibers, and Work Queues

Zephyr unifies tasks and fibers into *threads* with priorities split into *cooperative* (negative, never preempted — they run until they block or yield) and *preemptible* (zero or positive). Below preemptible threads sit *system work queues* — single-thread queues running deferred work; ISRs offload to them via `k_work_submit`.

```c
K_THREAD_STACK_DEFINE(sensor_stack, 1024);
static struct k_thread sensor_data;

void sensor_loop(void *a, void *b, void *c) {
    const struct device *dev = DEVICE_DT_GET(DT_NODELABEL(bme280));
    while (1) {
        struct sensor_value t;
        sensor_sample_fetch(dev);
        sensor_channel_get(dev, SENSOR_CHAN_AMBIENT_TEMP, &t);
        printk("T=%d.%06d\n", t.val1, t.val2);
        k_sleep(K_SECONDS(1));
    }
}

int main(void) {
    k_thread_create(&sensor_data, sensor_stack, K_THREAD_STACK_SIZEOF(sensor_stack),
                    sensor_loop, NULL, NULL, NULL,
                    5, 0, K_NO_WAIT);
    return 0;
}
```

=== Device Tree Driven

Every Zephyr driver instance is selected at compile time from device tree nodes. `DEVICE_DT_INST_DEFINE` macros emit a `struct device` per `status = "okay"` node. The build never enumerates a bus at runtime — boot time stays below 100 ms even on the slowest cores.

=== Logging, Shell, Tracing

Zephyr ships a deferred logger (ISRs enqueue, a dedicated thread formats and emits), a feature-rich shell over $"UART"$ or USB, and an SEGGER SystemView / Tracealyzer trace backend. None of these come "free" with FreeRTOS; they are major reasons Zephyr is winning new designs.

=== Comparison

#table(
  columns: 4,
  [*Aspect*], [*FreeRTOS*], [*Zephyr*], [*PREEMPT_RT Linux*],
  [License], [MIT], [Apache 2.0], [GPL v2],
  [Footprint], [\~5–10 KB], [\~50–200 KB], [\~20+ MB],
  [Scheduler], [Fixed prio + RR], [Fixed prio + EDF (optional)], [SCHED_FIFO/RR + CFS + EEVDF],
  [MMU], [Optional MPU], [Optional MPU], [Required],
  [Drivers], [Vendor HALs], [Unified DT driver model], [Linux kernel],
  [Networking], [LWIP add-on], [Native IP, BLE, Thread], [Full Linux stack],
  [Latency floor], [Sub-microsecond], [Sub-microsecond], [20--80 µs worst-case],
  [Cert kit available], [SafeRTOS (TÜV)], [Lynx, Auterion (partial)], [ELISA, Wind River Linux],
)

== PREEMPT_RT Linux

The PREEMPT_RT patchset, merged in stages between 2.6.x and 6.12 (when the last out-of-tree pieces landed), converts mainline Linux into a true real-time kernel. The core changes:

1. *Sleeping spinlocks*: most `spinlock_t` become `rt_mutex` underneath, allowing preemption while held.
2. *Threaded IRQs*: each IRQ becomes a kernel thread schedulable at `SCHED_FIFO`, defeating runaway hardware that monopolizes CPU.
3. *Priority inheritance*: built into `rt_mutex` and futexes (`FUTEX_LOCK_PI`).
4. *High-resolution timers*: per-CPU `hrtimer` infrastructure with nanosecond resolution.
5. *Forced preemption points* in long codepaths (e.g. RCU, ksoftirqd).

```bash
# Configure a thread for real-time on PREEMPT_RT
chrt -f 80 ./my_realtime_app          # SCHED_FIFO priority 80

# Lock memory to avoid page faults
mlockall(MCL_CURRENT | MCL_FUTURE);   // in C

# Reserve a CPU
isolcpus=3 nohz_full=3 rcu_nocbs=3    // kernel cmdline
taskset -c 3 ./my_realtime_app
```

A typical PREEMPT_RT cyclic test on a modern x86 server shows worst-case latency in the 20–80 microsecond range — orders of magnitude worse than a Cortex-M $"NVIC"$ but acceptable for industrial robotics, audio, and software-defined radio.

== Tickless and Power-Aware Schedulers

Both FreeRTOS (`configUSE_TICKLESS_IDLE`) and Zephyr support *tickless* operation: instead of a periodic SysTick interrupt, the scheduler programs the next wake based on the earliest timer in any task. Idle drops the core to `WFI` (or deeper sleep), reducing average power by an order of magnitude in event-driven workloads. The wake-up latency rises (re-locking the PLL after stop mode can take 100 microseconds) — a real-time budget must include this.

== Lock-Free Patterns

Many $"RTOS"$ codebases avoid locks in the producer-consumer path entirely:

```c
// Single-producer (ISR), single-consumer (task) ring buffer — lock-free
typedef struct {
    uint8_t buf[256];
    volatile uint8_t head;        // written by ISR
    volatile uint8_t tail;        // written by task
} ring_t;

static inline bool ring_push(ring_t *r, uint8_t b) {
    uint8_t h = r->head;
    uint8_t n = (uint8_t)(h + 1);
    if (n == r->tail) return false;  // full
    r->buf[h] = b;
    __DMB();                          // ensure data visible before head update
    r->head = n;
    return true;
}
```

The hardware ordering barrier (`DMB` on Armv7-M) is essential when head/tail and the buffer payload live in normal cacheable memory and are read across the ISR/task boundary.

== Picking an RTOS

#table(
  columns: 2,
  [*If you need...*], [*Pick...*],
  [Smallest possible footprint], [FreeRTOS heap_1, no networking],
  [Modern driver model + BLE/Thread + shell], [Zephyr],
  [POSIX, full Linux userland, hard RT under \~100 microseconds], [PREEMPT_RT],
  [Aerospace certification (DO-178C DAL A)], [VxWorks Cert, INTEGRITY-178, SafeRTOS],
  [Automotive ASIL D], [QNX Hypervisor, PikeOS, AUTOSAR Classic on OSEK],
  [Hard determinism + open source], [NuttX, RTEMS, Zephyr (with care)],
)

== Further Reading

Barry, R. (2024). "Mastering the FreeRTOS Real Time Kernel."

Zephyr Project (2024). "Zephyr Documentation: Kernel Services."

Rostedt, S., Hart, D. (2007). "Internals of the RT Patch." Linux Symposium.

Wind River (2024). "VxWorks 7 Kernel Programmer's Guide."

Real-Time Linux Foundation (2024). "PREEMPT_RT Patchset Documentation."

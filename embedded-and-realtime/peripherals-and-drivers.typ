= Peripherals and Drivers

Every microcontroller or application processor exposes its peripherals — UARTs, SPI controllers, DMA engines, GPIO banks — as memory-mapped registers or, on legacy x86, as port I/O addresses. This chapter covers how hardware exposes peripheral state to software, how interrupts and DMA transfer data efficiently, the electrical and framing details of UART/SPI/I2C, and how the Linux driver model translates all of it into a portable, maintainable kernel interface.

*See also:* _Real-Time Operating Systems_, _Microcontrollers and Systems-on-Chip_, `linux-kernel/scheduler.typ`, `operating-systems/processes-and-threads.typ`

== Memory-Mapped I/O vs Port I/O

=== Memory-Mapped I/O

*Memory-Mapped I/O* ($"MMIO"$) places peripheral control registers in the same address space as RAM. The CPU uses ordinary load/store instructions; the memory bus routes accesses to the peripheral rather than DRAM when the address falls in a peripheral region.

On ARM Cortex-M, the entire 4 GB address space is partitioned by the *ARM memory map*: 0x4000_0000–0x5FFF_FFFF is Peripheral, 0xE000_0000–0xE00F_FFFF is Private Peripheral Bus ($"PPB"$, NVIC, SysTick). Writing `*(volatile uint32_t*)0x40020018 = 1 << 5` asserts pin PA5 on an STM32.

*Volatile is mandatory.* The compiler must not cache, reorder, or eliminate MMIO accesses. Every register pointer must be `volatile`-qualified, or wrapped in a barrier macro.

=== Port I/O

*Port I/O* (x86 `in`/`out` instructions) uses a separate 16-bit I/O address space independent of RAM. Peripheral registers are accessed via I/O port numbers, not memory addresses. The x86 uses I/O ports for legacy hardware (PIC, PIT, legacy UART at 0x3F8). Modern x86 peripherals (PCIe, USB, SATA) use MMIO. Linux's `inb(port)` / `outb(val, port)` macros wrap the `in`/`out` instructions.

== Interrupt Handling

=== Vectored Interrupt Architecture

Each interrupt source has an entry in the *vector table* — an array of function pointers stored at a fixed address (ARM Cortex-M: address 0, or relocated via the Vector Table Offset Register $"VTOR"$). On reset, the CPU loads the stack pointer from vector 0 and the reset handler address from vector 1. Peripheral IRQs occupy vector 16 onward.

```c
// Minimal Cortex-M vector table (GCC)
__attribute__((section(".vectors")))
const void *vector_table[] = {
    (void*)&_stack_top,       // 0: initial SP
    reset_handler,            // 1: Reset
    nmi_handler,              // 2: NMI
    hardfault_handler,        // 3: HardFault
    // ...
    usart1_irq_handler,       // 53: USART1 (STM32F4)
};
```

=== NVIC on ARM Cortex-M

The *Nested Vectored Interrupt Controller* ($"NVIC"$) on Cortex-M manages up to 240 external IRQs. Key features:

- *Priority levels:* 8 bits wide (implementation uses 3–8 bits). Lower numeric value = higher priority.
- *Nesting:* a higher-priority IRQ preempts a lower-priority $"ISR"$ mid-execution. The CPU saves xPSR, PC, LR, R0–R3, R12 automatically on the *exception stack frame* (hardware push), enabling a C ISR without any prologue.
- *Tail-chaining:* if another IRQ is pending when an ISR returns, the CPU skips the pop/push cycle and directly vectors to the next handler, saving 12 cycles.
- *Late arrival:* if a higher-priority IRQ arrives during the stack-push of a lower-priority one, the CPU switches vectors before the first ISR instruction.

$"IRQ latency" = "vector fetch" + "stack push" + "pipeline refill" approx 12 "cycles at 0 wait-states"$

=== ISR Latency Budget

In a hard real-time system, the worst-case interrupt latency ($"WCIL"$) must be bounded:

$ "WCIL" = "IRQ assertion delay" + "longest IRQ-disabled critical section" + "NVIC response" + "ISR prologue" $

Minimise critical sections: prefer `__disable_irq()`/`__enable_irq()` only around register reads that must be atomic, never around lengthy computations. Use *interrupt priority grouping* to ensure time-critical ISRs (e.g., motor commutation) preempt slower ones (e.g., UART TX).

=== ISR Best Practices

ISRs must be short. Post work to a task queue or semaphore; do not call blocking functions.

```c
// ISR: flag only
void USART1_IRQHandler(void) {
    uint32_t sr = USART1->SR;
    if (sr & USART_SR_RXNE) {
        rx_buf[rx_head++ & (RX_BUF_SIZE-1)] = USART1->DR;
        BaseType_t woken;
        xSemaphoreGiveFromISR(rx_sem, &woken);
        portYIELD_FROM_ISR(woken);
    }
}
```

== DMA

*Direct Memory Access* ($"DMA"$) controllers transfer data between peripherals and memory without CPU involvement. The CPU programs a DMA channel with source address, destination address, transfer count, and data width; the DMA engine then bursts data over the AHB/AXI bus.

=== Transfer Modes

- *Single transfer:* one beat per DMA request from the peripheral.
- *Burst transfer:* the DMA controller reads/writes $2^n$ beats per bus arbitration (e.g., 4-word burst for cache-line alignment). Reduces bus turnaround overhead.
- *Circular mode:* the DMA wraps the buffer pointer automatically; used for continuous ADC sampling or UART ring buffers. The CPU is notified at *half-complete* and *full-complete* interrupts to process each half while the other is being filled.

=== Cache Coherence on ARM

On Cortex-A processors with data caches, DMA and the CPU share physical memory but the CPU may hold stale cached copies.

*DMA read from peripheral to memory (receive):*
1. Invalidate (not clean) the destination buffer in the D-cache before starting DMA: `SCB_InvalidateDCache_by_Addr(buf, len)`.
2. Start DMA.
3. After DMA complete interrupt: CPU reads from invalidated cache lines, which causes a cache miss and a fresh load from DRAM (which DMA wrote).

*DMA write from memory to peripheral (transmit):*
1. Clean (flush) the source buffer: `SCB_CleanDCache_by_Addr(buf, len)` — write dirty lines to DRAM.
2. Start DMA.
3. DMA reads from DRAM (which now has the correct data).

Cortex-M cores (M4, M7) with data caches require the same discipline. Cortex-M0/M3 are cache-less; coherence is automatic.

== Serial Protocols

=== UART

*Universal Asynchronous Receiver-Transmitter* ($"UART"$) is asynchronous, full-duplex, point-to-point. Frame format:

```
[IDLE high] [START=0] [D0 D1 D2 D3 D4 D5 D6 D7] [PARITY?] [STOP=1] [IDLE]
```

The receiver samples the line at the centre of each bit using an internal clock derived from the baud rate. Both ends must agree on baud rate (e.g., 115200 baud = 115200 bits/s). Clock error tolerance is typically ±2–3% across the full frame.

*RS-232* (±12 V logic) adds voltage level shifting for long cables. *RS-485* (differential pair) allows multi-drop buses up to 1200 m.

=== SPI

*Serial Peripheral Interface* ($"SPI"$) is synchronous, full-duplex, single-master multi-slave. Four wires: SCLK (clock), MOSI (master out), MISO (master in), $overline("CS")$ (chip select, active low per slave).

Data is shifted on SCLK edges; polarity (CPOL) and phase (CPHA) configure which edge is active:

#table(
  columns: (auto, 1fr, 1fr),
  table.header[*Mode*][*CPOL*][*CPHA*],
  [0], [0 (idle low)],  [0 (sample on rising)],
  [1], [0 (idle low)],  [1 (sample on falling)],
  [2], [1 (idle high)], [0 (sample on falling)],
  [3], [1 (idle high)], [1 (sample on rising)],
)

SPI has no built-in addressing or flow control. The master controls all timing; the slave must respond within one clock cycle. Common speeds: 1–50 MHz. Used for flash memory, ADCs, displays.

=== I2C

*Inter-Integrated Circuit* ($"I"^2"C"$) is synchronous, half-duplex, multi-master multi-slave over two wires: SDA (data, open-drain) and SCL (clock, open-drain). Pull-up resistors (1–10 kΩ) are required.

Frame structure:
```
[START] [7-bit ADDR] [R/W] [ACK] [DATA byte] [ACK] ... [STOP]
```

*START* = SDA falls while SCL is high. *STOP* = SDA rises while SCL is high. These are illegal mid-frame, enabling framing.

*Clock stretching:* a slave may hold SCL low to pause the master. *Arbitration:* if two masters drive SDA simultaneously, the one that loses (drives high while another drives low) detects the collision and backs off.

Standard speed: 100 kHz. Fast: 400 kHz. Fast-plus: 1 MHz. High-speed: 3.4 MHz (requires special START condition).

== GPIO

*General-Purpose Input/Output* ($"GPIO"$) pins are software-configurable as digital input, output, or alternate function (routed to a peripheral like UART or SPI).

Key register groups (STM32 example):
- `MODER`: mode (input/output/alternate/analog) per pin, 2 bits each.
- `ODR`/`IDR`: output data register / input data register.
- `BSRR`: atomic bit set/reset — upper 16 bits reset, lower 16 bits set. Avoids read-modify-write races.
- `OSPEEDR`: slew rate (low/medium/high/very-high).
- `PUPDR`: pull-up/pull-down enable.

```c
// Configure PA5 as push-pull output, no pull, medium speed
GPIOA->MODER  = (GPIOA->MODER & ~(3u << 10)) | (1u << 10);
GPIOA->OSPEEDR = (GPIOA->OSPEEDR & ~(3u << 10)) | (1u << 10);
// Toggle
GPIOA->BSRR = (1u << 5);   // set
GPIOA->BSRR = (1u << 21);  // reset (bit 16+5)
```

== Device Tree

The *Device Tree* ($"DT"$) is a data structure describing hardware topology passed from firmware to the Linux kernel at boot. It encodes: CPU core count, memory ranges, peripheral base addresses, IRQ numbers, clock parents, and pin mux assignments. Board-specific `.dts` files override or extend SoC-specific `.dtsi` files.

```dts
// Fragment: UART node in a .dts
&uart0 {
    compatible = "arm,pl011";
    reg = <0x9000000 0x1000>;    // base address, size
    interrupts = <GIC_SPI 33 IRQ_TYPE_LEVEL_HIGH>;
    clocks = <&clk_uart>;
    status = "okay";
    pinctrl-names = "default";
    pinctrl-0 = <&uart0_pins>;
};
```

The kernel's *Device Tree binding specification* (in `Documentation/devicetree/bindings/`) defines the required and optional properties per `compatible` string. A binding document is YAML-schema validated by `dt-schema`. The DT compiler (`dtc`) produces a flattened device tree blob (`.dtb`) which the bootloader (U-Boot, UEFI) loads at the kernel's registered DT address.

== Linux Driver Model

=== Platform Driver

A *platform driver* manages SoC peripherals that are not auto-discovered (unlike PCIe). The driver registers against a table of `compatible` strings; the kernel's DT probe matches DT nodes to drivers.

```c
static const struct of_device_id myuart_of_match[] = {
    { .compatible = "myco,myuart-1.0" },
    {}
};
MODULE_DEVICE_TABLE(of, myuart_of_match);

static struct platform_driver myuart_driver = {
    .probe  = myuart_probe,
    .remove = myuart_remove,
    .driver = {
        .name           = "myuart",
        .of_match_table = myuart_of_match,
    },
};
module_platform_driver(myuart_driver);
```

=== Probe and Remove

`probe()` is called when a matching device is found. It should:
1. Obtain resources: `platform_get_resource()` + `devm_ioremap_resource()`.
2. Map registers: `devm_ioremap()`.
3. Request IRQ: `devm_request_irq()`.
4. Register with a subsystem: `misc_register()`, `tty_register_driver()`, etc.

`devm_*` functions tie lifetime to the device; they are freed automatically on `remove()` or probe failure.

```c
static int myuart_probe(struct platform_device *pdev)
{
    struct myuart_priv *priv;
    struct resource *res;
    int irq, ret;

    priv = devm_kzalloc(&pdev->dev, sizeof(*priv), GFP_KERNEL);
    if (!priv)
        return -ENOMEM;

    res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
    priv->base = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(priv->base))
        return PTR_ERR(priv->base);

    irq = platform_get_irq(pdev, 0);
    ret = devm_request_irq(&pdev->dev, irq, myuart_isr,
                           IRQF_SHARED, "myuart", priv);
    if (ret)
        return ret;

    platform_set_drvdata(pdev, priv);
    return misc_register(&priv->miscdev);
}
```

=== Minimal Character Device Driver

A *character device* exposes `open`, `read`, `write`, `ioctl` file operations. The kernel assigns a major/minor number pair; `udev` creates the `/dev` node.

```c
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/uaccess.h>

#define BUF_SIZE 256
static char kbuf[BUF_SIZE];
static size_t kbuf_len;

static ssize_t mydev_read(struct file *f, char __user *buf,
                          size_t count, loff_t *off)
{
    size_t n = min(count, kbuf_len - (size_t)*off);
    if (n == 0)
        return 0;
    if (copy_to_user(buf, kbuf + *off, n))
        return -EFAULT;
    *off += n;
    return n;
}

static ssize_t mydev_write(struct file *f, const char __user *buf,
                           size_t count, loff_t *off)
{
    size_t n = min(count, (size_t)(BUF_SIZE - 1));
    if (copy_from_user(kbuf, buf, n))
        return -EFAULT;
    kbuf[n] = '\0';
    kbuf_len = n;
    return n;
}

static const struct file_operations mydev_fops = {
    .owner = THIS_MODULE,
    .read  = mydev_read,
    .write = mydev_write,
};

static struct miscdevice mydev = {
    .minor = MISC_DYNAMIC_MINOR,
    .name  = "mydev",
    .fops  = &mydev_fops,
};

static int __init mydev_init(void) { return misc_register(&mydev); }
static void __exit mydev_exit(void) { misc_deregister(&mydev); }
module_init(mydev_init);
module_exit(mydev_exit);
MODULE_LICENSE("GPL");
```

`copy_to_user` / `copy_from_user` are mandatory: kernel and user pointers live in different address spaces on systems with MMUs, and these functions safely fault on invalid user pointers.

== Further Reading

ARM Limited. "Cortex-M4 Devices Generic User Guide." ARM DUI 0553B.

Corbet, J., Rubini, A., Kroah-Hartman, G. (2005). "Linux Device Drivers, 3rd Edition." O'Reilly. (Free online.)

Yaghmour, K., et al. (2008). "Building Embedded Linux Systems." O'Reilly.

Linux Kernel Documentation. "Driver Model." `Documentation/driver-api/driver-model/`.

Linux Kernel Documentation. "Device Tree Usage." `Documentation/devicetree/usage-model.rst`.

Free Electrons (Bootlin). "Embedded Linux Kernel and Driver Development." Training materials (freely available PDF).

Massa, A. (2003). "Embedded Software Development with eCos." Prentice Hall. (interrupt/DMA chapters remain accurate for bare-metal patterns.)

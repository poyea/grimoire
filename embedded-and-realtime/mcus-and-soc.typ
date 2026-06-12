= Microcontrollers and Systems-on-Chip

A microcontroller ($"MCU"$) packages a CPU core, on-chip SRAM and flash, and a peripheral fabric onto one die — typically tens of kilobytes of RAM, megahertz to a few hundred MHz clock, and a power budget measured in milliwatts. A system-on-chip ($"SoC"$) scales this up: application-class cores (Cortex-A, RISC-V U-mode), DRAM controllers, GPUs, and accelerators sit alongside one or more $"MCU"$-class real-time cores. This chapter walks the silicon, the memory map, the interrupt fabric, and the boot path that every embedded engineer must internalize.

*See also:* _Real-Time Operating Systems_, _Peripherals and Drivers_, _ARM Deep Dive_ (architecture), _RISC-V_ (architecture)

== The Cortex-M Family

Arm's Cortex-M cores dominate the 32-bit $"MCU"$ market. The lineup trades area, power, and determinism:

#table(
  columns: 5,
  [*Core*], [*ISA*], [*Pipeline*], [*Features*], [*Use*],
  [M0/M0+], [Thumb-1 + subset Thumb-2], [2-3 stage], [No cache, no FPU, MPU optional], [Sensor nodes, BLE radios],
  [M3], [Thumb-2], [3-stage], [Bit-banding, MPU optional], [Legacy general-purpose],
  [M4], [Thumb-2 + DSP], [3-stage], [Single-precision FPU, SIMD], [Motor control, audio],
  [M7], [Thumb-2 + DSP], [6-stage dual-issue], [I/D-cache, double FPU, TCM], [High-end control],
  [M33], [Armv8-M Mainline], [3-stage], [TrustZone-M, FPU], [Secure $"IoT"$],
  [M55/M85], [Armv8.1-M], [4-7 stage], [Helium MVE vector, MPU], [ML at the edge],
)

The Armv7-M / Armv8-M programmer's model exposes 16 general-purpose registers (`R0`–`R15`), with `R13`=SP (banked into `MSP` and `PSP`), `R14`=LR, `R15`=PC. The `xPSR` register combines `APSR` (flags), `IPSR` (active exception number), and `EPSR` (Thumb bit, IT state).

=== Reset and Boot

On reset the Cortex-M does not jump to a fixed address; it reads two words from the *vector table* at the address given by `VTOR` (defaulting to 0x0000_0000):

```c
// Linker-generated vector table fragment (ARM Cortex-M)
extern uint32_t _estack;            // top of SRAM
void Reset_Handler(void);
void NMI_Handler(void);
void HardFault_Handler(void);
void SysTick_Handler(void);

__attribute__((section(".isr_vector"), used))
const void *const g_vectors[] = {
    (void *)&_estack,               // 0x00: initial MSP
    Reset_Handler,                  // 0x04: initial PC
    NMI_Handler,                    // 0x08
    HardFault_Handler,              // 0x0C
    /* ... MemManage, BusFault, UsageFault, reserved ... */
    SysTick_Handler,                // 0x3C
    /* IRQ0..IRQn from the SoC datasheet follow */
};
```

`Reset_Handler` initialises `.data` from flash, zeros `.bss`, configures the clock tree (PLL, prescalers), then calls `main`. Many vendors run a small ROM bootloader first that consults boot-mode pins (`BOOT0`, `BOOT1`) to decide whether to map flash, system memory (DFU), or SRAM at address 0.

=== Memory Map

The Armv7-M memory map is architecturally partitioned:

```
0x0000_0000 - 0x1FFF_FFFF  Code        (flash, often aliased to 0x0800_0000)
0x2000_0000 - 0x3FFF_FFFF  SRAM        (bit-band region on M3/M4: 1 MB)
0x4000_0000 - 0x5FFF_FFFF  Peripheral  (bit-band region: 1 MB)
0x6000_0000 - 0x9FFF_FFFF  External RAM
0xA000_0000 - 0xDFFF_FFFF  External device (strongly-ordered)
0xE000_0000 - 0xE00F_FFFF  Private Peripheral Bus (NVIC, SysTick, SCB, MPU)
0xE010_0000 - 0xFFFF_FFFF  Vendor-specific
```

Bit-banding (M3/M4 only) maps each bit in a 1 MB region to a 32-bit word in a 32 MB alias region — a load-modify-store on a single GPIO bit becomes an atomic single-word write.

=== Tightly-Coupled Memory and Caches

The Cortex-M7 introduces I-cache and D-cache (typically 16 KB each) plus *Instruction TCM* and *Data TCM* — SRAMs hung directly off the core with single-cycle access and no cache nondeterminism. $"WCET"$-critical handlers and DMA descriptors usually live in TCM precisely to avoid cache modelling.

== The Nested Vectored Interrupt Controller

The $"NVIC"$ is an Arm-defined, tightly integrated controller in every Cortex-M. Key properties:

- Up to 240 external interrupt lines (`IRQ0`..`IRQ239`), plus 16 internal exceptions.
- Each interrupt has an 8-bit priority field; the architecture mandates the top 3-8 bits be implemented (vendor choice). Lower numeric value = higher priority.
- *Preemption* is controlled by a split of the priority field into *group* (preempt) and *sub* priority via `AIRCR.PRIGROUP`.
- Tail-chaining: if a second IRQ is pending when an ISR returns, the $"NVIC"$ skips the stack restore/save and dispatches the next handler in ~6 cycles.
- Late arrival: a higher-priority IRQ arriving during stacking pre-empts before the lower handler runs.

```c
// Configure SysTick at 1 kHz, then enable USART1 IRQ at priority 5
SysTick->LOAD = (SystemCoreClock / 1000U) - 1U;
SysTick->VAL  = 0;
SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk
              | SysTick_CTRL_TICKINT_Msk
              | SysTick_CTRL_ENABLE_Msk;

NVIC_SetPriorityGrouping(3U);              // split 8-bit field 4/4 preempt/sub
                                           // (STM32 implements 4 bits, so 4 preempt / 0 sub)
NVIC_SetPriority(USART1_IRQn, 5U);
NVIC_EnableIRQ(USART1_IRQn);

__enable_irq();                            // PRIMASK = 0
```

=== Exception Entry and Exit

On exception entry the hardware pushes eight registers onto the active stack (`R0`–`R3`, `R12`, `LR`, `PC`, `xPSR`) — the *stack frame*. `LR` is loaded with an *EXC_RETURN* magic value (e.g. `0xFFFF_FFFD` = return to thread mode, use $"PSP"$) that the BX instruction interprets to unwind correctly. With the FPU enabled, lazy stacking reserves 18 more words but defers the actual FP register save until the handler touches `S0`–`S15`.

Worst-case interrupt latency on Armv7-M is *12 cycles* (M3/M4) from IRQ assertion to first handler instruction, assuming a single-cycle memory; flash wait states and DMA bus contention increase this. The architecture guarantees deterministic latency — no microcoded entry sequence, no shadow register file save.

== The System Control Block and MPU

The $"SCB"$ exposes `VTOR` (vector table relocation), `AIRCR` (priority grouping, system reset), `SCR` (sleep behaviour), `CCR` (alignment fault enable, unaligned access trap, divide-by-zero trap), `CFSR/HFSR/MMFSR/BFSR/UFSR` (fault status), `SHCSR` (system handler enable).

The *Memory Protection Unit* on Cortex-M defines 8 or 16 regions with base address, size, access permissions (privileged/unprivileged R/W/X), shareability, and cacheability. Unlike a full MMU there is no virtual-to-physical translation — the $"MPU"$ enforces protection only. An $"RTOS"$ typically reprograms the $"MPU"$ on each context switch to give each task an isolated region of SRAM and read-only access to code.

== Clock Trees and Power Domains

A modern $"MCU"$ contains a forest of oscillators (internal RC, crystal HSE, low-power LSI, LSE 32.768 kHz) feeding PLLs, then dividers, then per-peripheral clock gates. STM32 H7, for example, has three PLLs, four AHB clocks, and per-bus dividers — the peripheral clock you compute determines $"UART"$ baud, timer resolution, and ADC sampling frequency. Misconfiguring the clock is the single most common bring-up bug.

Power domains are usually structured as VBAT (backup, RTC), VDD (core), VDDA (analog). Sleep modes (`WFI`, `WFE`, stop, standby) gate clocks, retain SRAM, or power down everything except an RTC alarm. Wake-up latency rises from a few cycles ($"WFI"$) to milliseconds (standby) — a real-time design must budget this.

== Application-Class SoCs

The line blurs above Cortex-M7. NXP i.MX RT runs a Cortex-M7 at 600 MHz with no flash on-die (XIP from QSPI), eight DMA channels, and a memory controller — it is an "$"MCU"$ in form, $"SoC"$ in capability". Above that:

- *Heterogeneous $"SoC"$s*: i.MX 8, STM32 MP1, NXP S32G mix Cortex-A (Linux) and Cortex-M/R (real-time). The cores share DDR via an interconnect and communicate via shared memory + mailboxes (e.g. OpenAMP RPMsg).
- *Cortex-R*: lockstep cores ($"DCLS"$) for automotive — two cores run in lockstep one cycle apart, output is compared, mismatch raises a safety fault.
- *RISC-V $"MCU"$s*: SiFive E-series, GD32V, ESP32-C6, CH32V — typically RV32IMC with a CLINT (Core Local Interruptor) + PLIC (Platform-Level Interrupt Controller).

=== Comparing Interrupt Architectures

#table(
  columns: 4,
  [*Architecture*], [*Controller*], [*Latency*], [*Vectoring*],
  [Cortex-M], [Integrated $"NVIC"$], [12 cycles], [Hardware vector table],
  [Cortex-R], [GIC + VIC mode], [\~20 cycles], [Vectored or branch],
  [Cortex-A], [GIC-400/500/600], [\~50+ cycles], [Software dispatch],
  [RISC-V $"MCU"$], [CLINT+PLIC], [\~10 cycles], [`mtvec` table or single entry],
  [Xtensa LX], [INTC], [\~6 cycles], [Vectored],
)

== A Minimal Bring-Up

```c
// STM32F4 minimal blink — no HAL, no CMSIS startup
#define RCC_AHB1ENR  (*(volatile uint32_t *)0x40023830)
#define GPIOD_MODER  (*(volatile uint32_t *)0x40020C00)
#define GPIOD_ODR    (*(volatile uint32_t *)0x40020C14)

void Reset_Handler(void) {
    RCC_AHB1ENR |= (1U << 3);          // enable GPIOD clock
    GPIOD_MODER &= ~(0x3U << (12 * 2));
    GPIOD_MODER |=  (0x1U << (12 * 2)); // PD12 output
    for (;;) {
        GPIOD_ODR ^= (1U << 12);
        for (volatile int i = 0; i < 200000; i++) { }
    }
}
```

Linker script fragment:

```
MEMORY {
  FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
  SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}
SECTIONS {
  .isr_vector : { KEEP(*(.isr_vector)) } > FLASH
  .text       : { *(.text*) }            > FLASH
  .data       : { *(.data*) }            > SRAM AT > FLASH
  .bss        : { *(.bss*) *(COMMON) }   > SRAM
  _estack     = ORIGIN(SRAM) + LENGTH(SRAM);
}
```

== Device Tree on Embedded Linux and Zephyr

Higher-end $"SoC"$s use a *device tree* to describe the immutable hardware (memory ranges, IRQ numbers, clocks). A Zephyr fragment for an STM32 $"UART"$:

```dts
&usart2 {
    pinctrl-0 = <&usart2_tx_pa2 &usart2_rx_pa3>;
    pinctrl-names = "default";
    current-speed = <115200>;
    status = "okay";

    dmas = <&dma1 6 4 0x440 0x3>,
           <&dma1 5 4 0x480 0x3>;
    dma-names = "tx", "rx";
};
```

The build system parses this into compile-time constants and instantiates driver objects — no runtime enumeration on a typical $"MCU"$.

== Further Reading

ARM (2024). "Cortex-M7 Devices Generic User Guide." DUI 0646.

Yiu, J. (2013). "The Definitive Guide to ARM Cortex-M3 and Cortex-M4 Processors."

STMicroelectronics (2024). "RM0090: STM32F4xx Reference Manual."

NXP (2024). "i.MX RT1170 Processor Reference Manual."

RISC-V International (2024). "RISC-V Privileged Architecture Specification."

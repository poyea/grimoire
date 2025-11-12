= CPU Fundamentals

The Central Processing Unit executes instructions stored in memory. Understanding the instruction set architecture (ISA) and execution model is foundational to all performance analysis.

*See also:* Pipelining (for instruction execution stages), Superscalar (for parallel execution), Cache Hierarchy (for data access)

== Instruction Set Architecture (ISA)

The Instruction Set Architecture defines the contract between hardware and software, specifying the instructions, registers, addressing modes, and memory model that software can rely upon.

The major ISAs in use today include x86-64 (AMD64), which is a Complex Instruction Set Computer (CISC) with variable-length encoding and 16 general-purpose registers; ARM (AArch64), a Reduced Instruction Set Computer (RISC) with fixed-length encoding and 31 general-purpose registers; and RISC-V, an open RISC ISA with modular extensions.

The CISC versus RISC distinction involves several key differences. CISC architectures like x86-64 use variable instruction lengths from 1 to 15 bytes, support complex addressing modes with memory operands, have fewer registers (16 general-purpose), and employ microcode for complex instructions. An example is `mov rax, [rbx + rcx*8 + 0x100]`, which performs a memory load with scaling. RISC architectures like ARM and RISC-V use fixed 32-bit instruction lengths, follow a load/store architecture where only load and store instructions access memory, provide many registers (31-32 general-purpose), and use simple, regular instruction formats. An example is `ldr x0, [x1, x2, lsl #3]`, which loads with a shifted offset.

```
CISC (x86-64):
- Variable instruction length (1-15 bytes)
- Complex addressing modes (memory operands)
- Few registers (16 GP registers)
- Microcode for complex instructions
- Example: mov rax, [rbx + rcx*8 + 0x100]  # Memory load with scaling

RISC (ARM, RISC-V):
- Fixed instruction length (32-bit)
- Load/store architecture (only load/store access memory)
- Many registers (31-32 GP registers)
- Simple, regular instruction format
- Example: ldr x0, [x1, x2, lsl #3]  # Load with shifted offset
```

Historically, the CISC versus RISC debate dominated the 1980s and 1990s, but has since been largely resolved. Modern CISC implementations like x86 internally translate instructions to RISC-like micro-ops [Intel SDM].

*Modern implementations (2023-2024):*

```
Intel Raptor Lake (13th/14th gen):
- 8 P-cores (Performance) + 16 E-cores (Efficiency)
- P-cores: 6-wide decode, 512-entry ROB
- E-cores: 4-wide decode, smaller structures
- Hybrid architecture optimized by OS scheduler

AMD Zen 4 (Ryzen 7000):
- 4-way decode, 256-entry ROB
- Unified scheduler with 6 ALU ports
- Improved branch predictor (12K BTB entries)
- DDR5 support, up to 5.7 GHz boost

Apple M3:
- ARM-based, 16-wide decode (widest in industry!)
- Massive reorder buffer (600+ entries estimated)
- 3nm process, excellent performance-per-watt
```

== x86-64 Registers

*General-purpose registers (64-bit):*

```
rax (accumulator)        r8  (general)
rbx (base)               r9  (general)
rcx (counter)            r10 (general)
rdx (data)               r11 (general)
rsi (source index)       r12 (general)
rdi (destination index)  r13 (general)
rbp (base pointer)       r14 (general)
rsp (stack pointer)      r15 (general)

Access smaller portions:
rax (64-bit), eax (32-bit), ax (16-bit), al/ah (8-bit)
```

*Special-purpose registers:*
- *rip (instruction pointer):* Address of next instruction
- *rflags:* Status flags (ZF, CF, SF, OF, etc.)
- *Segment registers:* cs, ds, ss, es, fs, gs (mostly legacy, fs/gs used for TLS)

*SIMD registers:*
- *XMM (128-bit):* xmm0-xmm15 (SSE)
- *YMM (256-bit):* ymm0-ymm15 (AVX)
- *ZMM (512-bit):* zmm0-zmm31 (AVX-512)

*Floating-point:*
- *x87 FPU stack:* st(0)-st(7) (legacy)
- Modern code uses XMM/YMM for FP via SSE/AVX

== Instruction Encoding

*x86-64 instruction format (variable-length):*

```
[Prefixes] [REX] [Opcode] [ModR/M] [SIB] [Displacement] [Immediate]
 0-4 bytes   1B    1-3B      1B       1B     0-4B         0-4B

Total: 1-15 bytes
```

*Example encoding:*
```asm
mov rax, [rbx + rcx*8 + 0x100]

Encoding: 48 8B 84 CB 00 01 00 00
48       = REX.W prefix (64-bit operand)
8B       = MOV opcode (register <- memory)
84       = ModR/M byte (register rax, SIB follows)
CB       = SIB byte (base=rbx, index=rcx, scale=8)
00010000 = Displacement (0x100 = 256)
```

*Decoding complexity:*
- Variable length → cannot determine instruction boundaries without decoding
- Complex addressing → need multiple pipeline stages for address calculation
- Alignment: Instructions can span cache line boundaries → potential extra fetch

*ARM instruction format (fixed 32-bit):*

```
31      28 27    24 23    20 19    16 15    12 11     0
├─────────┼────────┼────────┼────────┼────────┼────────┤
│  Cond   │ OpCode │   Rn   │   Rd   │  Operand2       │
└─────────┴────────┴────────┴────────┴─────────────────┘

Simpler decode: All instructions 32-bit aligned, regular format
```

== Instruction Types

*Data movement:*
```asm
mov rax, rbx          ; Register to register (1 cycle, 0.25 CPI)
mov rax, [rbx]        ; Load from memory (4-5 cycles L1 hit, 1 per cycle throughput)
mov [rax], rbx        ; Store to memory (1 cycle throughput, async)
lea rax, [rbx+rcx*8]  ; Load effective address (1 cycle, addr calculation)
```

*Arithmetic:*
```asm
add rax, rbx          ; Addition (1 cycle latency, 0.25 CPI)
sub rax, rbx          ; Subtraction (1 cycle)
imul rax, rbx         ; Multiply (3 cycles latency, 1 CPI on modern CPUs)
idiv rbx              ; Divide (rax / rbx → rax, 20-40 cycles, not pipelined!)
```

*Logical:*
```asm
and rax, rbx          ; Bitwise AND (1 cycle)
or  rax, rbx          ; Bitwise OR (1 cycle)
xor rax, rax          ; Bitwise XOR (0 cycle idiom - recognized as zero)
not rax               ; Bitwise NOT (1 cycle)
```

*Control flow:*
```asm
jmp label             ; Unconditional jump
je  label             ; Jump if equal (ZF=1)
call func             ; Push return address, jump
ret                   ; Pop return address, jump
```

*Comparison:*
```asm
cmp rax, rbx          ; Compare (sets flags, 1 cycle)
test rax, rax         ; Bitwise AND, sets flags (common idiom for zero check)
```

== Instruction Latency vs Throughput

Latency measures the cycles from when operands are ready until the result becomes available, which matters for dependency chains. Throughput, measured in cycles per instruction (CPI), indicates how many cycles elapse before another instruction of the same type can be issued, with its reciprocal representing instructions per cycle.

Modern CPUs like Zen 3 and Skylake exhibit varying latency and throughput characteristics across different instruction types:

| Instruction | Latency (cycles) | Throughput (CPI) | Execution Units |
|:------------|:----------------:|:----------------:|:----------------|
| mov r, r    | 0-1 | 0.25 | Renamed away or ALU |
| add r, r    | 1 | 0.25 | 4 ALU ports |
| imul r, r   | 3 | 1 | 1 multiplier |
| load        | 4-5 | 0.5 | 2 load ports |
| store       | 1 (ST) | 1 | 1 store port + 1 data |
| idiv        | 20-40 | 20-40 | 1 divider, not pipelined |

The key insight is the distinction between throughput-bound and latency-bound code. Throughput-bound code consists of independent operations that can be issued in parallel, with all four additions potentially executing in the same cycle using the available ALU ports:

```asm
add rax, rbx     ; Cycle 0
add rcx, rdx     ; Cycle 0 (parallel)
add r8,  r9      ; Cycle 0 (parallel)
add r10, r11     ; Cycle 0 (parallel)
; All 4 execute in same cycle (4 ALU ports)
```

Latency-bound code involves dependent operations that are serialized by data dependencies. Each instruction must wait for the previous one to complete, resulting in four cycles total despite each instruction having a 0.25 CPI throughput:

```asm
add rax, rbx     ; Cycle 0, result ready cycle 1
add rax, rcx     ; Cycle 1, waits for rax from cycle 0
add rax, rdx     ; Cycle 2, waits for rax from cycle 1
add rax, r8      ; Cycle 3, waits for rax from cycle 2
; Total: 4 cycles despite 0.25 CPI per instruction
```

== Zero-Latency Idioms

Modern CPUs recognize special instruction patterns that can be executed with zero latency and without using an execution unit. The patterns `xor rax, rax` and `sub rax, rax` both zero the register in 0 cycles, `mov rax, rax` is eliminated as a no-op during register renaming, and `test rax, rax` checks for zero in 0 cycles when the previous writer is known:

```asm
xor rax, rax          ; Zero register (0 cycles, no execution unit)
sub rax, rax          ; Zero register (0 cycles)
mov rax, rax          ; No-op, eliminated during rename
test rax, rax         ; Check if zero (0 cycles if previous writer known)
```

This works because register renaming tracks that the result is a constant value and eliminates the actual execution. These idioms provide dependency breaking, allowing out-of-order execution to proceed without waiting for previous operations. A false dependency where the old register value is irrelevant but the CPU waits anyway can be avoided by using the proper form that the CPU recognizes:

```asm
; BAD: False dependency
xor eax, eax      ; Old rax value irrelevant, but CPU may wait
add eax, 5

; GOOD: Dependency broken
xor rax, rax      ; CPU recognizes zero, no dependency
add eax, 5        ; Can execute immediately
```

== Addressing Modes

*x86-64 memory addressing:*

```
[base + index*scale + displacement]

base: any GP register
index: any GP register except rsp
scale: 1, 2, 4, or 8
displacement: signed 32-bit constant

Examples:
mov rax, [rbx]                  ; Base only
mov rax, [rbx + 0x100]          ; Base + displacement
mov rax, [rbx + rcx]            ; Base + index
mov rax, [rbx + rcx*8]          ; Base + scaled index
mov rax, [rbx + rcx*8 + 0x100]  ; Full form
```

*Address calculation (AGU - Address Generation Unit):*
- Simple modes (base + disp): 1 cycle
- Complex modes (base + index*scale + disp): 1 cycle (dedicated AGU units)
- Modern CPUs: 2-3 AGU units → 2-3 addresses/cycle

*LEA (Load Effective Address) for arithmetic:*
```asm
lea rax, [rbx + rcx*8 + 5]  ; rax = rbx + rcx*8 + 5 (1 cycle)
; More efficient than: mov rax, rcx; shl rax, 3; add rax, rbx; add rax, 5
```

== Micro-ops (μops)

*x86 instructions decode into simpler micro-ops:*

```asm
; Simple instruction (1 μop)
add rax, rbx       → add rax, rbx

; Complex instruction (2 μops)
add rax, [rbx]     → load tmp, [rbx]
                   → add rax, tmp

; Very complex (microcode, many μops)
cpuid              → 30+ μops (serialize pipeline, flush buffers)
```

*Fusion:* Combine multiple μops into single entity.

*Macro-fusion:* cmp + branch → single μop
```asm
cmp rax, rbx
je  label
; Fused into single branch μop
```

*Micro-fusion:* Load + ALU op → single μop
```asm
add rax, [rbx]
; Fused into load-ALU μop (if both fit in single μop cache entry)
```

*ROB (Reorder Buffer) limit:* 224-512 μops in flight (modern CPUs). Limits instruction window for out-of-order execution.

== Calling Convention (x86-64 System V ABI)

*Function arguments:*
```
Integer/pointer args: rdi, rsi, rdx, rcx, r8, r9
Float args: xmm0-xmm7
Additional args: stack

Return value:
Integer: rax (+ rdx for 128-bit)
Float: xmm0
```

*Callee-saved registers:* rbx, rbp, r12-r15 (must preserve).

*Caller-saved registers:* rax, rcx, rdx, rsi, rdi, r8-r11 (can be clobbered).

*Example:*
```asm
; Caller
mov rdi, 10        ; First arg
mov rsi, 20        ; Second arg
call func
; rax contains return value

; Callee
func:
    push rbx       ; Save callee-saved register
    mov rbx, rdi   ; Use first arg
    add rax, rbx   ; Compute result
    pop rbx        ; Restore
    ret
```

*Function call overhead:*
- call: ~2-3 cycles (push return address, predict target)
- ret: ~2-3 cycles (pop address, predict return)
- Total: ~5-10 cycles depending on prediction accuracy

*Inlining eliminates this overhead.* Compiler auto-inlines small functions with -O2/-O3.

== Performance Counters

*Hardware counters measure execution:*

```bash
# Linux perf tool
perf stat -e cycles,instructions,branches,branch-misses ./program

# Example output:
#   1,234,567,890  cycles
#   1,000,000,000  instructions    # IPC = 1000M / 1234M = 0.81
#      50,000,000  branches
#       2,500,000  branch-misses   # 5% mispredict rate
```

*Key metrics:*
- *IPC (Instructions Per Cycle):* > 2 = good, < 1 = bottleneck
- *Branch miss rate:* < 5% = good, > 10% = problematic
- *Cache miss rate:* Depends on level, see Cache Hierarchy section

*Intel-specific counters:*
- Micro-ops issued/retired
- Pipeline stalls (frontend, backend)
- TLB misses
- Memory bandwidth usage

*See Performance Analysis section for detailed profiling techniques.*

== Common Performance Pitfalls

*1. False dependencies from partial register writes:*

```asm
; BAD: Partial register stall
mov al, 1         ; Write 8-bit subregister
add rax, rbx      ; Read full 64-bit register → stall waiting for merge

; GOOD: Write full register
xor eax, eax      ; Zero idiom clears full register
mov al, 1
add rax, rbx      ; No stall
```

*2. Unnecessary memory operations:*

```asm
; BAD: Load-store redundancy
mov rax, [mem]
mov [mem], rax    ; Unnecessary store

; GOOD: Keep in register
mov rax, [mem]
; Use rax...
```

*3. Division performance:*

```c
// BAD: Division in hot loop (20-40 cycles!)
for (int i = 0; i < n; i++) {
    result[i] = data[i] / constant;
}

// GOOD: Convert to multiplication
float inv = 1.0f / constant;  // Once, outside loop
for (int i = 0; i < n; i++) {
    result[i] = data[i] * inv;  // 4-5 cycles
}
```

*4. Branch misprediction in tight loops:*

```c
// BAD: Unpredictable branch
for (int i = 0; i < n; i++) {
    if (data[i] > threshold) {  // Random data → 50% mispredict!
        sum += data[i];
    }
}

// GOOD: Branchless with CMOV
for (int i = 0; i < n; i++) {
    int mask = -(data[i] > threshold);  // 0 or -1
    sum += data[i] & mask;
}
```

*5. Cache line splits:*

```c
// BAD: Structure spans cache lines
struct Data {
    char padding[60];
    int value;  // Crosses 64-byte boundary!
};

// GOOD: Align to cache line
struct Data {
    alignas(64) int value;
    char padding[60];
};
```

== Compiler Optimization Flags

*GCC/Clang optimization levels:*

```bash
-O0  # No optimization (debugging)
-O1  # Basic optimization
-O2  # Recommended for production (balance speed/size)
-O3  # Aggressive optimization (may increase code size)
     # Enables: vectorization, loop unrolling, inline functions
-Os  # Optimize for size
-Ofast # -O3 + fast-math (non-standard floating point)

# Architecture-specific
-march=native      # Use all CPU features (AVX2, BMI2, etc.)
-march=skylake     # Target specific microarchitecture
-mtune=native      # Tune for CPU without requiring features

# Useful flags
-ffast-math        # Aggressive FP optimization (breaks IEEE754)
-funroll-loops     # Unroll loops for better ILP
-finline-functions # Aggressive inlining
-fomit-frame-pointer  # Free up rbp register

# Profiling
-pg                # Generate profiling info for gprof
-fprofile-generate # First pass for PGO
-fprofile-use      # Second pass uses profile data
```

*Performance analysis during compilation:*

```bash
# See what got vectorized
gcc -O3 -march=native -fopt-info-vec-all code.c

# Generate assembly
gcc -S -O3 -march=native code.c -o code.s

# See optimization remarks
clang -O3 -Rpass=inline -Rpass=vectorize code.c
```

== Debugging Performance Issues

*Quick diagnosis with perf:*

```bash
# Is it CPU-bound or memory-bound?
perf stat -e cycles,instructions,cache-misses,L1-dcache-load-misses ./program

# High IPC (>2): CPU-bound, check for vectorization
# Low IPC (<1): Likely memory-bound or branch mispredicts

# Check specific bottleneck
perf stat -e cycles,stalled-cycles-frontend,stalled-cycles-backend ./program

# Frontend stalls: Instruction fetch issues (I-cache, branch prediction)
# Backend stalls: Execution issues (D-cache, dependencies)
```

*Assembly inspection:*

```bash
# Disassemble with Intel syntax (easier to read)
objdump -d -M intel program | less

# Look for hot spots
perf record ./program
perf report
perf annotate function_name  # See assembly with profiling data
```

== References

*Primary sources:*

Intel Corporation (2023). Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 1: Basic Architecture. Order Number 253665.

Intel Corporation (2023). Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 2: Instruction Set Reference. Order Number 325383.

AMD (2023). AMD64 Architecture Programmer's Manual, Volume 1: Application Programming. Publication #24592.

ARM Holdings (2023). ARM Architecture Reference Manual ARMv8, for A-profile architecture. ARM DDI 0487.

*Textbooks:*

Hennessy, J.L. & Patterson, D.A. (2017). Computer Architecture: A Quantitative Approach (6th ed.). Morgan Kaufmann. Chapters 1-2.

Patterson, D.A. & Hennessy, J.L. (2020). Computer Organization and Design (6th ed.). Morgan Kaufmann. Chapter 2.

*Optimization guides:*

Fog, A. (2023). Instruction Tables: Lists of Instruction Latencies, Throughputs and Micro-operation Breakdowns. Technical University of Denmark. https://www.agner.org/optimize/instruction_tables.pdf

Fog, A. (2023). The Microarchitecture of Intel, AMD and VIA CPUs. Technical University of Denmark. https://www.agner.org/optimize/microarchitecture.pdf

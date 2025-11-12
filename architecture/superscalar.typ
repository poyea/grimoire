= Superscalar and Out-of-Order Execution

Superscalar processors execute multiple instructions per cycle using multiple execution units. Out-of-order execution reorders instructions dynamically to hide latencies and maximize throughput.

*See also:* Pipelining (for in-order execution baseline), Branch Prediction (for speculative execution), CPU Fundamentals (for instruction dependencies)

== Superscalar Execution

A scalar pipeline can execute at most one instruction per cycle, while a superscalar processor can execute N instructions per cycle, making it an N-way superscalar architecture. Modern CPUs typically implement 4-way superscalar execution: Cycle 1 fetches 4 instructions, Cycle 2 decodes them, Cycle 3 executes all 4 if they are independent and execution units are available, and Cycle 4 retires them. The theoretical maximum is 4 instructions per cycle (IPC), but practical performance typically achieves 2-3 IPC due to data dependencies and resource constraints.

```
4-way superscalar (modern CPUs):

Cycle 1: Fetch 4 instructions
Cycle 2: Decode 4 instructions
Cycle 3: Execute 4 instructions (if independent, 4 execution units)
Cycle 4: Retire 4 instructions

Theoretical max: 4 IPC (instructions per cycle)
Practical: 2-3 IPC typical due to dependencies
```

*Execution units (Intel Skylake):*

```
Port 0: ALU, FP_ADD, FP_MUL, Branch
Port 1: ALU, FP_ADD, FP_MUL
Port 2: Load (AGU)
Port 3: Load (AGU)
Port 4: Store Data
Port 5: ALU, Vector Shuffle
Port 6: ALU, Branch
Port 7: Store Address (AGU)

Total: 8 execution ports → theoretical 8 μops/cycle
Practical: 4-5 μops/cycle sustained
```

*Independent instructions can execute in parallel:*

```asm
; Cycle 1: All 4 execute in parallel (4 IPC)
add rax, rbx    ; Port 0 or 1 or 5 or 6
add rcx, rdx    ; Port 0 or 1 or 5 or 6
add r8,  r9     ; Port 0 or 1 or 5 or 6
add r10, r11    ; Port 0 or 1 or 5 or 6
```

*Dependent instructions serialize:*

```asm
add rax, rbx    ; Cycle 1
add rax, rcx    ; Cycle 2 (depends on rax from cycle 1)
add rax, rdx    ; Cycle 3 (depends on rax from cycle 2)
add rax, r8     ; Cycle 4 (depends on rax from cycle 3)
; Total: 4 cycles, IPC = 1
```

== Out-of-Order Execution (OoO)

In-order execution suffers from stalls during long-latency operations. When a load takes 5 cycles, subsequent dependent instructions must wait, even if other independent instructions could execute. The load completes in cycle 5, the add executes in cycle 6, the multiply (which is independent) must wait until cycle 7, and the subtract completes in cycle 10, for a total of 10 cycles.

```asm
; In-order execution:
ld   rax, [rbx]    ; 5 cycles (cache hit)
add  rcx, rax      ; Stalled 5 cycles waiting for rax
mul  rdx, r8       ; Stalled (waits for add), independent!
sub  r9,  r10      ; Stalled (waits for mul), independent!

Total: 5 + 1 + 3 + 1 = 10 cycles
```

Out-of-order execution identifies that the multiply and subtract are independent of the load, allowing them to execute in parallel with it. The load occupies cycles 1-5, the multiply executes during cycles 1-3 in parallel, the subtract completes in cycle 1, and only the dependent add must wait until cycle 6. This reduces total execution time to 6 cycles, compared to 10 cycles for in-order execution.

```asm
ld   rax, [rbx]    ; Cycle 1-5
mul  rdx, r8       ; Cycle 1-3 (parallel with ld!)
sub  r9,  r10      ; Cycle 1 (parallel with ld and mul!)
add  rcx, rax      ; Cycle 6 (after ld completes)

Total: 6 cycles (vs 10 in-order)
```

== Tomasulo's Algorithm

*Key idea [Tomasulo 1967]:* Track dependencies in hardware, execute when operands ready.

*Components:*

```
1. Reservation Stations (RS): Queue for instructions waiting for operands
2. Register Renaming: Eliminate false dependencies (WAR, WAW)
3. Common Data Bus (CDB): Broadcast results to waiting instructions
4. Reorder Buffer (ROB): Maintain program order for retirement
```

*Execution flow:*

```
1. Issue: Dispatch instruction to reservation station
   - Allocate ROB entry
   - Rename registers (allocate physical registers)
   - Check operands: Ready → value, Not ready → tag

2. Execute: When all operands ready
   - Send to execution unit
   - Compute result

3. Write Result: Broadcast on CDB
   - Update ROB
   - Wake up waiting instructions in RS

4. Commit (Retire): When instruction reaches head of ROB
   - Update architectural register file
   - Free physical register from previous mapping
   - Handle exceptions (precise exceptions)
```

== Register Renaming

False dependencies limit parallelism even when no true data flow exists. Write After Read (WAR) hazards occur when a write must wait for a previous read to complete, and Write After Write (WAW) hazards occur when multiple writes to the same register must preserve program order, even though the intermediate values are never used.

```asm
; WAR hazard (Write After Read)
add rax, rbx    ; Read rax
...
mov rax, rcx    ; Write rax - must wait for add to read old rax

; WAW hazard (Write After Write)
mov rax, rbx    ; Write rax
...
mov rax, rcx    ; Write rax - must preserve program order
```

Register renaming solves this by mapping architectural registers to a larger pool of physical registers. Modern CPUs provide 128 or more physical registers (P0, P1, P2, ..., P127) to back the 16 architectural registers (rax, rbx, ..., r15). The Register Alias Table (RAT) maintains the mapping from architectural to physical registers.

When `mov rax, rbx` executes, rax is mapped to physical register P10 and rbx to P20. The subsequent `mov rax, rcx` remaps rax to a different physical register P11, eliminating the WAW hazard. The following `add rdx, rax` uses the new mapping (P11), creating no dependency on the first instruction. This eliminates false dependencies and increases the instruction window for out-of-order execution.

```
Architectural registers: rax, rbx, ..., r15 (16 registers)
Physical registers: P0, P1, P2, ..., P127 (128+ registers in modern CPUs)

RAT (Register Alias Table): Maps architectural → physical

Example:
mov rax, rbx    ; P10 ← P20 (rax mapped to P10, rbx to P20)
mov rax, rcx    ; P11 ← P30 (rax remapped to P11, eliminates WAW)
add rdx, rax    ; P40 ← P40 + P11 (uses new rax, no dependency)
```

== Reorder Buffer (ROB)

*Purpose:* Maintain program order for retirement (precise exceptions).

```
ROB: Circular buffer, 224-512 entries (modern CPUs)

ROB entry:
- Instruction PC
- Destination register (architectural)
- Result value (or ready bit)
- Exception status

Operations:
- Issue: Allocate ROB entry (tail)
- Execute: Fill result when complete
- Commit: Retire from head (in program order)
```

*Precise exceptions:* Only committed instructions are visible to exception handler.

```asm
Inst 1: ld  rax, [rbx]     ; Executes, ROB[0]
Inst 2: div rcx, rdx       ; Executes, ROB[1], EXCEPTION (divide by zero)
Inst 3: add r8,  r9        ; Executes out-of-order, ROB[2]

Commit:
1. ROB[0] commits (no exception)
2. ROB[1] commits → EXCEPTION → flush ROB[2], report exception at Inst 2
3. ROB[2] never commits (speculative, discarded)

Result: Exception appears to occur at Inst 2 (program order preserved)
```

== Memory Ordering

*Problem:* Loads/stores can execute out-of-order → memory consistency issues.

*Load/Store Queue:*
- *Load Queue (LQ):* Track outstanding loads
- *Store Queue (SQ):* Buffer stores until retirement

*Memory disambiguation:*

```asm
st [rax], rbx     ; Store to unknown address
ld rcx, [rdx]     ; Load from unknown address

Problem: If rax == rdx, load must wait for store (dependency)
         If rax != rdx, load can execute ahead (no dependency)

Solution: Predict independence, execute speculatively
- If prediction correct: Performance win
- If prediction wrong: Flush pipeline, replay load
```

*Store-to-load forwarding:*

```asm
st [rax], rbx     ; Store 5 to address 0x1000
ld rcx, [rax]     ; Load from same address

Optimization: Forward store data directly to load (no memory access)
Latency: ~5 cycles (vs ~200 if load misses cache)
```

== Instruction Window and ILP

*Instruction window:* Number of instructions CPU can examine for out-of-order execution.

```
Window size = ROB size = 224-512 entries (modern CPUs)

Larger window:
- More opportunities to find independent instructions
- Hide longer latencies (cache miss, divide)
- Diminishing returns beyond ~200-300 instructions
```

*ILP (Instruction-Level Parallelism):* Inherent parallelism in code.

```
Perfectly parallel code (no dependencies):
ILP = ∞, limited only by execution units

Real code:
ILP = 2-4 typical (measured by OoO window)
IPC = 2-3 (actual, limited by ILP and execution bandwidth)
```

*Amdahl's Law for ILP:*

```
Speedup = 1 / (Serial_fraction + Parallel_fraction / Width)

Example: 80% parallel code, 4-wide superscalar
Speedup = 1 / (0.2 + 0.8/4) = 1 / 0.4 = 2.5x
```

== Speculative Execution

*Speculate beyond branches:* Execute instructions before knowing if branch taken.

```asm
    cmp rax, rbx
    je  taken
    add rcx, rdx     ; Speculatively execute (predict not taken)
    mul r8,  r9      ; Speculatively execute
    ...
taken:
    sub r10, r11

If prediction correct: Instructions commit normally
If prediction wrong:   Flush speculative instructions, start over
```

*ROB tracks speculation:* Speculative instructions marked, not committed until branch resolves.

*Spectre vulnerability [Kocher et al. 2019]:* Speculative execution leaks data via cache side channels.

== Performance Example

```c
// Latency-bound (serialized):
int sum = 0;
for (int i = 0; i < N; i++) {
    sum += arr[i];  // Dependency chain: sum[i] = sum[i-1] + arr[i]
}
// IPC ~1.0 (latency of add = 1 cycle, limits throughput)

// Parallelism via unrolling:
int sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
for (int i = 0; i < N; i += 4) {
    sum1 += arr[i];
    sum2 += arr[i+1];
    sum3 += arr[i+2];
    sum4 += arr[i+3];
}
int sum = sum1 + sum2 + sum3 + sum4;
// IPC ~3.5 (4 independent chains, limited by load throughput)
```

== Practical Optimization for OoO

*Guideline 1: Expose instruction-level parallelism*

```c
// BAD: Single accumulator (dependency chain)
float sum = 0.0f;
for (int i = 0; i < n; i++) {
    sum += arr[i];  // IPC ~1.0, serialized by dependency
}

// GOOD: Multiple accumulators (parallel chains)
float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
for (int i = 0; i < n; i += 4) {
    sum0 += arr[i];
    sum1 += arr[i+1];
    sum2 += arr[i+2];
    sum3 += arr[i+3];
}
float sum = sum0 + sum1 + sum2 + sum3;
// IPC ~3.0, 3x faster!
```

*Guideline 2: Keep pipeline full*

```c
// BAD: Complex conditions in hot loop
for (int i = 0; i < n; i++) {
    if (complex_check(arr[i])) {  // Stalls pipeline
        result[i] = expensive_compute(arr[i]);
    }
}

// GOOD: Separate filtering from computation
int count = 0;
for (int i = 0; i < n; i++) {
    if (complex_check(arr[i])) {
        indices[count++] = i;
    }
}
for (int j = 0; j < count; j++) {
    result[indices[j]] = expensive_compute(arr[indices[j]]);
}
// Better pipelining: fewer branch mispredicts, better prefetching
```

*Guideline 3: Avoid memory aliasing*

```c
// BAD: Compiler can't prove independence
void process(int* a, int* b, int* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];  // May alias, can't reorder
    }
}

// GOOD: Use restrict to guarantee no aliasing
void process(int* __restrict a, int* __restrict b,
             int* __restrict c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];  // Proven independent, aggressive optimization
    }
}
```

*Measuring ILP effectiveness:*

```bash
# Check instruction throughput
perf stat -e uops_issued.any,uops_executed.thread ./program

# High IPC but low utilization → frontend bottleneck
# Low IPC with high backend stalls → dependencies or memory

# Port utilization (Intel)
perf stat -e uops_dispatched_port.port_0,\
uops_dispatched_port.port_1,\
uops_dispatched_port.port_5 ./program
# Balanced usage across ports = good ILP
```

== References

Tomasulo, R.M. (1967). "An Efficient Algorithm for Exploiting Multiple Arithmetic Units." IBM Journal of Research and Development 11(1): 25-33.

Smith, J.E. & Sohi, G.S. (1995). "The Microarchitecture of Superscalar Processors." Proceedings of the IEEE 83(12): 1609-1624.

Hennessy, J.L. & Patterson, D.A. (2017). Computer Architecture: A Quantitative Approach (6th ed.). Morgan Kaufmann. Chapter 3 (Instruction-Level Parallelism).

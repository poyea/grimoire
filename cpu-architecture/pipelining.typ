= Pipelining

Pipelining overlaps instruction execution: while one instruction executes, the next instruction decodes, and the one after that fetches. Analogous to assembly line in manufacturing.

*See also:* CPU Fundamentals (for instruction encoding), Superscalar (for multiple pipelines), Branch Prediction (for control hazards)

== Classic 5-Stage Pipeline

The classic RISC pipeline consists of five stages [Hennessy & Patterson 2017]: Instruction Fetch (IF) reads the instruction from the I-cache, Instruction Decode (ID) decodes the opcode and reads registers, Execute (EX) performs ALU operations or address calculations, Memory Access (MEM) handles loads and stores to the D-cache, and Write Back (WB) writes the result to the register file. In steady state, one instruction completes per cycle (CPI = 1) with a latency of 5 cycles per instruction and throughput of 1 instruction/cycle, providing 5x faster execution than a non-pipelined design.

```
IF  (Instruction Fetch): Read instruction from I-cache
ID  (Instruction Decode): Decode opcode, read registers
EX  (Execute): ALU operation or address calculation
MEM (Memory Access): Load/store to D-cache
WB  (Write Back): Write result to register file

Timeline:
Cycle:  1    2    3    4    5    6    7
Inst 1: IF   ID   EX  MEM   WB
Inst 2:      IF   ID   EX  MEM   WB
Inst 3:           IF   ID   EX  MEM   WB
Inst 4:                IF   ID   EX  MEM

Steady state: 1 instruction completes per cycle (CPI = 1)
Latency: 5 cycles per instruction
Throughput: 1 instruction/cycle (5x faster than non-pipelined)
```

=== Stage Details

*IF (Instruction Fetch):*
- PC → Instruction Memory → IR
- PC incremented by 4 (or branch target)
- I-cache access: 1-4 cycles on miss (L1 hit ~1 cycle)
- Modern CPUs fetch 4-6 instructions per cycle (superscalar)

*ID (Instruction Decode):*
- Opcode decoded, register file read
- Immediate values sign-extended
- Control signals generated (ALUOp, MemRead, MemWrite, RegWrite, Branch)
- Register renaming (Tomasulo's) maps architectural → physical registers

*EX (Execute):*
- ALU performs operation (add, sub, AND, OR, shift, compare)
- Branch target address computed: PC + (sign-extended offset × 4)
- For loads/stores: effective address = base + offset

*MEM (Memory Access):*
- Load: Data Memory[address] → data register
- Store: register → Data Memory[address]
- D-cache access: L1 hit ~4-5 cycles, L2 ~12 cycles, L3 ~40 cycles, DRAM ~200 cycles
- Memory ordering: loads may bypass earlier stores if no address conflict

*WB (Write Back):*
- Result written to destination register
- For loads: memory data → register file
- For ALU ops: ALU result → register file

Modern x86 processors face additional decode complexity due to variable-length instructions, requiring multi-stage decode. The simplified x86 pipeline flows from Fetch through Predecode, Decode, Micro-op Queue, Execute, and finally Retire. The decode stages include length decode to determine instruction boundaries, instruction decode to identify the opcode, and micro-op translation to convert CISC instructions to RISC-like μops. Intel Skylake features a 16+ stage pipeline, while AMD Zen 3 uses a 19-stage pipeline.

```
Modern x86 pipeline (simplified):
Fetch → Predecode → Decode → Micro-op Queue → Execute → Retire

Decode stages:
1. Length decode (determine instruction boundaries)
2. Instruction decode (identify opcode)
3. Micro-op translation (CISC → RISC-like μops)

Intel Skylake: 16+ stage pipeline
AMD Zen 3: 19 stage pipeline
Intel Raptor Lake: 14-18 stages (reduced vs Skylake)
AMD Zen 4: 19 stages (similar to Zen 3)
Apple M3: ~16 stages (estimated)
```

*Modern pipeline characteristics:*

```
Trend: Moderate depth (14-20 stages) for balance
- Too shallow: Lower frequency
- Too deep: Higher misprediction penalty
- Modern CPUs: 3-4 GHz base, 5+ GHz boost

Pipeline width (decode):
- Intel P-cores: 6-wide
- AMD Zen 4: 4-wide
- Apple M3: 16-wide (!!)

Branch misprediction recovery:
- ~15-20 cycles typical
- Determines acceptable misprediction rate (<5%)
```

== Pipeline Hazards

Three types of hazards prevent the pipeline from achieving ideal CPI of 1: structural, data, and control hazards.

=== Structural Hazards

*Definition:* Two instructions need the same hardware resource in the same cycle.

```
add r1, r2, r3    ; Uses ALU in EX stage
ld  r4, [r5]      ; Uses memory in MEM stage
                  ; No conflict - different resources

; But if memory has single port:
ld  r1, [r2]      ; MEM stage cycle 4
st  r3, [r4]      ; MEM stage cycle 4 - conflict! Stall 1 cycle
```

*Examples and fixes:*
- Single-ported memory: IF and MEM both need memory → stall
  - _Fix:_ separate I-cache and D-cache (Harvard architecture)
- Single ALU: two ALU instructions in superscalar → stall
  - _Fix:_ multiple functional units
- Register file ports: simultaneous read and write
  - _Fix:_ dual-ported register file (read in first half-cycle, write in second)

=== Data Hazards

*RAW (Read After Write) — true dependency:*

```asm
ADD R1, R2, R3    ; R1 ← R2 + R3
SUB R4, R1, R5    ; R4 ← R1 - R5  (needs R1 from ADD)
```

_Resolution:_ forwarding from EX/MEM → EX, or 1-cycle stall if load-use.

*WAR (Write After Read) — anti-dependency:*

```asm
ADD R1, R2, R3    ; reads R2
SUB R2, R4, R5    ; writes R2
```

_Resolution:_ cannot occur in simple 5-stage (reads in ID before writes in WB). Relevant in out-of-order CPUs; resolved by register renaming.

*WAW (Write After Write) — output dependency:*

```asm
ADD R1, R2, R3    ; writes R1
SUB R1, R4, R5    ; writes R1
```

_Resolution:_ cannot occur in simple 5-stage (in-order, one write per cycle). In superscalar/OoO: register renaming eliminates it.

_IPC impact:_ On a 4-wide superscalar, a RAW chain like `ld rax,[mem]; add rbx,rax; sub rcx,rbx` serializes three dependent ops, dropping IPC from ~4.0 to ~1.3 for this sequence. A pipeline flush from a control hazard costs 12--20 cycles on modern x86 (Skylake: ~16 cycles, Zen 4: ~13 cycles).

=== Control Hazards

*Control hazard:* Branch changes control flow.

```asm
cmp rax, rbx
je  target        ; Branch decision at EX stage (cycle 3)
add rcx, rdx      ; Fetched speculatively, may be wrong path
; If branch taken, flush add instruction (wasted work)
```

*Branch penalty analysis:*

#table(
  columns: (auto, auto, auto),
  [*Strategy*], [*Penalty (cycles)*], [*Effective CPI impact*],
  [Stall until resolved], [1-3], [+0.15 to +0.45 (15% branch rate)],
  [Predict not taken], [1 on taken], [~+0.07 (50% taken)],
  [Static predict backward taken], [1 on misprediction], [~+0.03 (loops)],
  [Dynamic 2-bit predictor], [1 on misprediction], [~+0.015 (93% accuracy)],
  [Tournament predictor], [~15 on misprediction], [~+0.006 (97% accuracy)],
  [TAGE predictor], [~15 on misprediction], [~+0.003 (99%+ accuracy)],
)

== Forwarding (Bypassing)

*Solution to data hazards:* Forward result before writeback.

```
add rax, rbx      ; Result ready after EX stage (cycle 3)
sub rcx, rax      ; Needs rax in EX stage (cycle 5)

Forwarding path: EX → EX (bypass MEM, WB stages)

Timeline with forwarding:
Cycle:  1    2    3    4    5
add:    IF   ID   EX  MEM   WB  (rax = rbx + value)
sub:         IF   ID ─EX→ MEM   (uses rax via forwarding)
                      ↑
                   Forward result from add
```

=== Forwarding Paths

*EX/MEM → EX forwarding:*

```asm
ADD R1, R2, R3    ; result available end of EX
SUB R4, R1, R5    ; needs R1 at start of EX → forward from EX/MEM latch
```

No stall needed. Forwarding MUX selects pipeline register over register file.

*MEM/WB → EX forwarding:*

```asm
ADD R1, R2, R3    ; result in MEM/WB latch
AND R6, R1, R7    ; 2 cycles later, can still forward
```

*Load-use hazard (requires 1 stall):*

```asm
ld  rax, [rbx]    ; Result ready after MEM (cycle 4)
add rcx, rax      ; Needs rax in EX (cycle 3) → STALL 1 cycle

Timeline:
Cycle:  1    2    3    4    5    6
ld:     IF   ID   EX  MEM   WB
add:         IF   ID [stall] EX  MEM  (forwarding from ld MEM stage)
```

_Cannot forward backward in time._ Hardware inserts a bubble (NOP).

*Stall avoidance — instruction scheduling:*

```asm
; Before scheduling (1 stall):
ld  rax, [rbx]
add rcx, rax      ; Stall!

; After scheduling (0 stalls):
ld  rax, [rbx]
sub rdx, rdi      ; Independent instruction fills the slot
add rcx, rax      ; rax ready now, available via MEM/WB forward
```

Compilers (`gcc -O2`, `clang`) perform this reordering automatically.

== Branch Prediction

*Problem:* Branch outcome unknown until EX stage → stall pipeline.

*Solution:* Speculate (guess) branch direction, execute speculatively.

```asm
    cmp  rax, rbx
    je   taken_path      ; Predict: TAKEN (90% accurate)
    ; Speculatively fetch taken_path
    ; If prediction correct: No penalty
    ; If prediction wrong: Flush pipeline (10-20 cycle penalty)
taken_path:
    add  rcx, rdx
```

*Branch misprediction cost:* Flush all speculatively executed instructions.

```
Pipeline depth = 16 stages (Intel Skylake)
Misprediction penalty = 16-20 cycles (flush + refill)

10% mispredict rate, 1 branch per 5 instructions:
Overhead = 0.2 branches/inst × 0.1 mispredict × 20 cycles = 0.4 CPI penalty
```

*See Branch Prediction section for detailed prediction algorithms.*

=== Speculative Execution and Recovery

When the processor predicts a branch, it continues fetching and executing instructions along the predicted path. If the prediction is wrong, all speculative work must be discarded.

```
Recovery mechanism:
1. Checkpoint register state at each branch
2. On misprediction, restore checkpoint
3. Flush all instructions after the branch
4. Redirect fetch to correct target

Cost breakdown (Skylake, ~16 stage pipeline):
- Detection:     1-2 cycles (compare prediction vs actual)
- Flush:         1 cycle (mark speculative instructions invalid)
- Refill:        14-18 cycles (re-fetch from correct path)
- Total penalty: ~16-20 cycles
```

*Branch Target Buffer (BTB):*
- Caches branch target addresses for taken branches
- Indexed by PC, returns predicted target
- Miss in BTB → predict not-taken (or stall)
- Modern BTBs: 4K-8K entries (L1), 16K+ entries (L2)

*Return Address Stack (RAS):*
- Hardware stack for CALL/RET prediction
- CALL pushes return address, RET pops
- Typically 16-32 entries deep
- Accuracy: >99% for well-structured code

== Deep Pipelines and Pipeline Depth Comparison

*Pentium 4 (Netburst):* 31-stage pipeline, 3.8 GHz clock.

*Advantages:*
- Higher clock frequency (shorter stages)
- More instructions in flight

*Disadvantages:*
- Branch misprediction costs 31 cycles! (vs 5 for classic RISC)
- Deeper pipeline = more hazards, more forwarding paths
- Performance often worse than shorter pipelines

=== Pipeline Depth Across CPU Generations

#table(
  columns: (auto, auto, auto, auto),
  [*Processor*], [*Year*], [*Pipeline Stages*], [*Notes*],
  [MIPS R2000], [1985], [5], [Classic textbook pipeline],
  [Intel Pentium], [1993], [5], [Dual-issue superscalar],
  [Intel Pentium Pro], [1995], [14], [First x86 OoO execution],
  [Intel Pentium 4 (Willamette)], [2000], [20], [High clock speed strategy],
  [Intel Pentium 4 (Prescott)], [2004], [31], [Deepest x86 pipeline ever],
  [Intel Core (Yonah)], [2006], [14], [Return to shorter pipeline],
  [Intel Skylake], [2015], [14-19], [Variable depth, $mu$op cache shortcut],
  [Apple M1 (Firestorm)], [2020], [~16], [Wide 8-issue OoO],
  [AMD Zen 4], [2022], [~19], [Simultaneous multithreading],
)

*Trend:* Ultra-deep pipelines (Pentium 4 era, 20-31 stages) increased frequency but suffered severe branch misprediction penalties. Modern designs favor moderate depth (14-19) with wider issue and better predictors.

== Pipeline Stalls and Performance

=== Performance Equations

*Basic pipeline speedup:*

$ "Speedup" = "Pipeline depth" / (1 + "Stall cycles per instruction") $

*CPI with stalls:*

$ "CPI"_"pipeline" = 1 + p_"branch" dot m_"branch" dot P_"branch" + p_"load" dot p_"load-use" dot 1 $

Where:
- $p_"branch"$: fraction of instructions that are branches (~15-20%)
- $m_"branch"$: branch misprediction rate
- $P_"branch"$: misprediction penalty (cycles)
- $p_"load"$: fraction of loads (~25%)
- $p_"load-use"$: fraction of loads with dependent next instruction (~40%)

*Example:* 20-stage pipeline, 5% misprediction rate, 15% branch rate:

$ "CPI" = 1 + 0.15 dot 0.05 dot 20 + 0.25 dot 0.4 dot 1 = 1 + 0.15 + 0.1 = 1.25 $

*Expanded CPI breakdown:*

```
CPI = Ideal_CPI + Stalls_per_inst

Stalls_per_inst = Data_hazards + Control_hazards + Structural_hazards

Example:
Ideal: 1 instruction/cycle
Data hazards: 0.2 stalls/inst (load-use, cache miss)
Branch mispredicts: 0.1 branch/inst × 0.05 mispredict × 20 cycles = 0.1 stalls/inst
Total CPI = 1 + 0.2 + 0.1 = 1.3
IPC = 1/CPI = 0.77
```

*Measuring stalls:*

```bash
perf stat -e cycles,instructions,stalled-cycles-frontend,stalled-cycles-backend ./program

# Frontend stalls: Instruction fetch/decode bottleneck
# Backend stalls: Execution bottleneck (cache miss, long-latency op)
```

=== Impact of Cache Misses on Pipeline Performance

Cache misses are the dominant source of pipeline stalls in modern processors. When a load misses the L1 D-cache, the pipeline must stall (or continue with other independent instructions in an OoO core) until the data arrives.

```
Effective memory access time:
T_eff = hit_rate × T_L1 + miss_rate_L1 × (hit_rate_L2 × T_L2
        + miss_rate_L2 × (hit_rate_L3 × T_L3 + miss_rate_L3 × T_DRAM))

Example (typical desktop workload):
L1 hit rate: 95%, T_L1 = 4 cycles
L2 hit rate: 80% (of L1 misses), T_L2 = 12 cycles
L3 hit rate: 90% (of L2 misses), T_L3 = 40 cycles
DRAM: T_DRAM = 200 cycles

T_eff = 0.95×4 + 0.05×(0.80×12 + 0.20×(0.90×40 + 0.10×200))
      = 3.8 + 0.05×(9.6 + 0.20×(36 + 20))
      = 3.8 + 0.05×(9.6 + 11.2)
      = 3.8 + 1.04 = 4.84 cycles
```

In-order pipelines stall entirely on a cache miss. Out-of-order (OoO) processors can continue executing independent instructions during a miss, partially hiding the latency. This is one of the primary motivations for OoO execution.

_See also:_ Memory Hierarchy chapter for cache design details; Superscalar chapter for OoO execution and instruction-level parallelism.

== Exceptions and Interrupts in Pipelines

Handling exceptions in a pipelined processor is challenging because multiple instructions are in-flight simultaneously.

*Precise exceptions:* The processor must ensure that all instructions before the faulting instruction have completed, and no instruction after it has committed results. This is the standard requirement for correct exception handling.

```
Exception types by pipeline stage:
IF:  Page fault (instruction fetch), misaligned PC
ID:  Undefined opcode, illegal instruction
EX:  Arithmetic overflow, divide by zero
MEM: Page fault (data access), misaligned address
WB:  (rare) None in simple pipelines

Challenge: An earlier instruction (in program order) may cause an
exception AFTER a later instruction already in the pipeline.
Example:
  Inst 1 (in MEM): page fault!
  Inst 2 (in EX):  already executing — must be squashed
  Inst 3 (in ID):  already decoded — must be squashed
```

*Solution — reorder buffer (ROB):*
- Instructions commit (write results) in program order
- If an exception occurs, all later instructions are discarded
- Results are held in the ROB until commit
- Modern CPUs: 224 entries (Skylake), 320 entries (Zen 4), 600+ entries (Apple M1)

== References

Hennessy, J.L. & Patterson, D.A. (2017). Computer Architecture: A Quantitative Approach (6th ed.). Morgan Kaufmann. Appendix C (Pipelining).

Patterson, D.A. & Hennessy, J.L. (2020). Computer Organization and Design (6th ed.). Morgan Kaufmann. Chapter 4 (The Processor).

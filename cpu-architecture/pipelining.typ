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

*Structural hazard:* Resource conflict (e.g., single memory port).

```
add r1, r2, r3    ; Uses ALU in EX stage
ld  r4, [r5]      ; Uses memory in MEM stage
                  ; No conflict - different resources

; But if memory has single port:
ld  r1, [r2]      ; MEM stage cycle 4
st  r3, [r4]      ; MEM stage cycle 4 - conflict! Stall 1 cycle
```

*Solution:* Multiple execution units, separate I-cache/D-cache (Harvard architecture).

*Data hazard:* Instruction needs result of previous instruction.

```asm
add rax, rbx      ; Cycle 1: IF, Cycle 2: ID, Cycle 3: EX, Cycle 4: MEM, Cycle 5: WB
sub rcx, rax      ; Needs rax from add (available cycle 5)
                  ; sub starts cycle 2: IF, Cycle 3: ID (needs rax!) → stall
```

*RAW (Read After Write):* True dependency.
*WAR (Write After Read):* Anti-dependency (solved by register renaming).
*WAW (Write After Write):* Output dependency (solved by register renaming).

_IPC impact:_ On a 4-wide superscalar, a RAW chain like `ld rax,[mem]; add rbx,rax; sub rcx,rbx` serializes three dependent ops, dropping IPC from ~4.0 to ~1.3 for this sequence. A pipeline flush from a control hazard costs 12--20 cycles on modern x86 (Skylake: ~16 cycles, Zen 4: ~13 cycles).

*Control hazard:* Branch changes control flow.

```asm
cmp rax, rbx
je  target        ; Branch decision at EX stage (cycle 3)
add rcx, rdx      ; Fetched speculatively, may be wrong path
; If branch taken, flush add instruction (wasted work)
```

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

*Load-use hazard:* Load result not ready until MEM stage.

```asm
ld  rax, [rbx]    ; Result ready after MEM (cycle 4)
add rcx, rax      ; Needs rax in EX (cycle 3) → STALL 1 cycle

Timeline:
Cycle:  1    2    3    4    5    6
ld:     IF   ID   EX  MEM   WB
add:         IF   ID [stall] EX  MEM  (forwarding from ld MEM stage)
```

*Compiler scheduling:* Rearrange instructions to avoid stalls.

```asm
; Before optimization (1 stall):
ld  rax, [rbx]
add rcx, rax      ; Stall!

; After optimization (0 stalls):
ld  rax, [rbx]
sub rdx, rdi      ; Independent instruction fills bubble
add rcx, rax      ; rax ready now
```

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

== Deep Pipelines (Pentium 4)

*Pentium 4 (Netburst):* 31-stage pipeline, 3.8 GHz clock.

*Advantages:*
- Higher clock frequency (shorter stages)
- More instructions in flight

*Disadvantages:*
- Branch misprediction costs 31 cycles! (vs 5 for classic RISC)
- Deeper pipeline = more hazards, more forwarding paths
- Performance often worse than shorter pipelines

*Modern trend:* 12-20 stages (balance frequency vs misprediction cost).

== Pipeline Stalls and Performance

*CPI (Cycles Per Instruction):*

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

== References

Hennessy, J.L. & Patterson, D.A. (2017). Computer Architecture: A Quantitative Approach (6th ed.). Morgan Kaufmann. Appendix C (Pipelining).

Patterson, D.A. & Hennessy, J.L. (2020). Computer Organization and Design (6th ed.). Morgan Kaufmann. Chapter 4 (The Processor).

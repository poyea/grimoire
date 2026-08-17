#import "../template.typ": xref

= Instruction Selection and Code Generation

The backend translates the optimised IR into target machine code. The three classical phases are instruction selection (matching IR patterns to machine instructions), instruction scheduling (reordering instructions to fill pipeline slots), and register allocation (covered in the previous chapter). This chapter focuses on selection and scheduling, plus the LLVM SelectionDAG and GlobalISel pipelines, and closes with object file emission and link-time optimisation.

*See also:* #xref("compilers", "ir-design", label: "IR Design and SSA Form"), #xref("compilers", "register-allocation", label: "Register Allocation"), #xref("compilers", "dataflow-analysis", label: "Dataflow Analysis")

== Instruction Selection

Instruction selection chooses one or more machine instructions to implement each IR operation. The goal is to cover the IR tree with instruction patterns at minimum cost.

*Tree pattern matching:* if the IR is a tree of operations (one result per expression), selection reduces to a covering problem. Each machine instruction is described as a _pattern_ --- a subtree of IR nodes --- with an associated cost. Dynamic programming over the tree finds the minimum-cost cover.

*BURG* (Bottom-Up Rewriting Grammar) formalises this as a string-rewriting system. Each rule rewrites an IR tree node (with children rewritten to non-terminals) to a non-terminal at a given cost. The BURS generator produces a table-driven matcher from a grammar.

*LLVM SelectionDAG:*

+ Lower LLVM IR to a target-independent _SelectionDAG_ --- a DAG of `SDNode` objects. Nodes carry types (`i32`, `f64`, `v4i32`, ...) and a target-independent opcode.
+ *Legalise types:* replace unsupported types with supported ones (see below).
+ *Legalise operations:* replace unsupported opcodes with sequences of supported ones.
+ *Instruction selection:* pattern-match DAG nodes to `MachineSDNode` objects using TableGen-generated tables.
+ *Scheduling:* linearise the DAG into a `MachineInstr` sequence.

```
// Simplified SelectionDAG for:  return a + b * 4
//
//   CopyToReg(ret_reg)
//       |
//      ADD
//     /   \
//   CopyFromReg(a)   SHL
//                   /   \
//            CopyFromReg(b)  Const(2)
```

*LLVM GlobalISel* (introduced in LLVM 6, production in LLVM 12+) operates on _Generic MIR_ --- an $"SSA"$ representation using generic opcodes (`G_ADD`, `G_LOAD`, ...) at the `MachineFunction` level. It avoids SelectionDAG's issues:

- No quadratic DAG rebuilding.
- $"SSA"$ is preserved throughout instruction selection and into register allocation.
- Target-specific lowering is done incrementally by a _legaliser_ and _instruction selector_ that rewrite generic opcodes to target opcodes.

GlobalISel is now the default on AArch64 at `-O0` and increasingly at higher optimisation levels.

== Legalisation

Not every IR type or operation maps directly to a machine instruction. Legalisation transforms IR until all types and operations are natively supported.

- *Promote:* widen a narrow type to a supported width. `i1 add` on a 32-bit machine becomes `i32 add` with a mask. `f16` arithmetic becomes `f32`.
- *Expand:* decompose a large type into smaller ones. A 64-bit integer on a 32-bit target splits into a pair of 32-bit values; `i64 add` expands to two `i32 adds` with carry propagation.
- *Scalarise:* a vector operation on an unsupported width is split into scalar operations on each element.
- *Custom lowering:* the target provides a hook to emit an arbitrary instruction sequence for a specific opcode/type combination.

```cpp
// Target legalisation hook (LLVM TableGen style)
setOperationAction(ISD::UDIV, MVT::i32, Expand);
// tells the legaliser to expand i32 udiv into a library call
// or a sequence of shifts/subtracts
setOperationAction(ISD::ROTL, MVT::i32, Custom);
// target provides LowerROTL()
```

Legalisation is iterative: expanding one operation may introduce another illegal one.

== Peephole Optimisation

After instruction selection, a _peephole optimiser_ scans a small window over the machine instruction stream and replaces inefficient patterns with better ones.

*Classic examples:*

- `mov r1, r2; mov r2, r1` --- the second move is redundant if `r1` and `r2` are not used between the pair.
- `imul rax, 8` --- on targets where shift is cheaper, rewrite to `shl rax, 3`.
- `add rax, 0` --- remove no-op arithmetic.
- `cmp rax, 0; jne label` --- if the previous instruction already set flags for `rax`, the `cmp` is redundant.

LLVM implements peepholes in several places: `MachineCombiner` applies target-provided patterns on `MachineInstr` trees; `PeepholeOptimizer` handles copy elimination and redundant extends; target-specific `MachineInstr` combiners run late in the pipeline.

TableGen's `Pat<>` patterns can also express peephole rules declaratively, letting the backend author enumerate rewrites once and have them applied during instruction selection.

== Instruction Scheduling

Machine instructions must be ordered in a linear sequence. The scheduler's job is to find an order that keeps the pipeline busy, hides memory latency, and avoids structural or data hazards.

*Hazard types:*

- *Structural:* two instructions need the same functional unit in the same cycle (e.g., two floating-point multiplies on a single-issue FPU).
- *Data RAW (read-after-write):* instruction $B$ reads a value produced by $A$; $B$ must start at least $"latency"(A)$ cycles after $A$.
- *Data WAW / WAR:* write-after-write and write-after-read hazards, relevant on out-of-order hardware where the compiler must maintain ordering for in-order simulators or VLIW targets.
- *Control:* branches affect which instructions follow; the branch delay slot on MIPS/SPARC must be filled with a useful instruction or a NOP.

*List scheduling:*

+ Build a _data-dependence graph_ (DDG): directed edges from producer to consumer, weighted by latency.
+ Maintain a _ready set_ of instructions with all predecessors issued.
+ At each cycle, pick the highest-priority ready instruction and issue it.
+ Priority heuristics: longest path from node to exit (critical path), number of successors, register pressure.

```
// List scheduling priority: critical path length
fn schedule(ddg: &DDG) -> Vec<Instr> {
    compute_heights(ddg);  // height = longest path to exit
    let mut ready: BinaryHeap<(i32, Instr)> = collect_roots(ddg);
    let mut result = vec![];
    while let Some((_, instr)) = ready.pop() {
        result.push(instr);
        for succ in ddg.successors(instr) {
            decrement_in_degree(succ);
            if in_degree(succ) == 0 {
                ready.push((height(succ), succ));
            }
        }
    }
    result
}
```

*Register-pressure-aware scheduling:* the heuristic penalises instructions that extend live ranges. Scheduling too aggressively for latency can increase register pressure and cause more spills; scheduling conservatively reduces pressure at the cost of pipeline slots.

LLVM implements pre-register-allocation scheduling (`MachineScheduler`) and post-RA scheduling (`PostRAScheduler`), each with pluggable strategies per target.

== Software Pipelining and Modulo Scheduling

For innermost loops, the scheduler can overlap multiple iterations to fully utilise functional units. This is _software pipelining_.

The *initiation interval* (II) is the number of cycles between starting successive iterations. Two lower bounds apply:

- $"ResMII"$: resource constraint --- II must be large enough that no functional unit is oversubscribed within a single iteration's resource budget.
- $"RecMII"$: recurrence constraint --- II must be at least as long as the longest cycle in the DDG (a loop-carried dependence).

$II_min = max("ResMII", "RecMII")$.

*Modulo scheduling (Rau 1994):* schedule one iteration in $II$ slots such that the schedule is periodic --- each successive iteration starts one $II$ offset later. The kernel repeats; a prologue fills the pipeline and an epilogue drains it.

```
// Conceptual kernel schedule (II = 3 cycles, 2 iterations overlapped)
Cycle 0: load A[i]
Cycle 1: load A[i+1],  compute f(A[i-1])
Cycle 2: store B[i-1], compute f(A[i])
// repeats every 3 cycles
```

Modulo scheduling is critical on VLIW targets (Intel Itanium, TI C6000 DSP) where the compiler is solely responsible for filling issue slots. LLVM's `MachinePipeliner` implements the SMS (Swing Modulo Scheduling) variant.

== Prologue and Epilogue Insertion

Every non-leaf function needs a prologue (function entry) and epilogue (function exit) to manage the stack frame and callee-saved registers.

*Prologue:*

+ Adjust the stack pointer: `sub rsp, frame_size`.
+ Save callee-saved registers that the function uses.
+ Optionally establish a frame pointer: `mov rbp, rsp`.

*Epilogue:*

+ Restore callee-saved registers.
+ Restore the stack pointer (via frame pointer or explicit `add rsp`).
+ Return.

*Frame pointer omission* (`-fomit-frame-pointer`): freeing `rbp` (x86-64) or `r7` (ARM) adds one extra general-purpose register. Stack unwinding must then rely on `.eh_frame` (DWARF CFI) or Windows unwind tables rather than chasing the frame pointer chain.

LLVM's `PrologEpilogInserter` pass runs after register allocation (because it needs to know which callee-saved registers were actually used) and inserts the appropriate machine instructions. It also computes the final stack frame layout, assigning concrete offsets to each spill slot and local variable.

== Debug Information

The debugger needs to map every machine instruction back to a source location, and to know where each variable lives (register, stack slot, or expression) at each point.

*DWARF* is the standard debug format on Linux/macOS/ELF targets (version 5 is current). Key components:

- *.debug_info:* typed descriptions of compilation units, functions, variables, and types in a tree of DIEs (Debug Information Entries).
- *.debug_line:* compact line-number tables mapping instruction addresses to source file/line/column.
- *.debug_loc / .debug_loclists:* location lists tracking where a variable lives as its assignment migrates across registers and stack slots during optimisation.
- *.eh_frame:* call-frame information for stack unwinding (used by both exception handling and debuggers).

```dwarf
// Simplified DWARF location expression for a variable
// "lives in register rbx from offset 0x10 to 0x40,
//  then at [rsp+8] from 0x40 to 0x80"
DW_AT_location: location list
  [0x10, 0x40): DW_OP_reg3   (rbx)
  [0x40, 0x80): DW_OP_breg7 8 (rsp + 8)
```

LLVM uses `DIBuilder` to attach `!dbg` metadata to IR instructions; the backend propagates this metadata through instruction selection and scheduling to produce accurate location lists. Register allocation decisions (splits, spills) directly affect debug quality: a value spilled to the stack generates additional location-list entries.

*Split DWARF* (`.dwo` files) separates the bulk of debug information into a side file, keeping link-time overhead low. *Compressed debug sections* (`.zdebug_*`) reduce binary size. On Windows, the *PDB* format replaces DWARF.

== Object File and Linking

The assembler and object-file writer translate `MachineInstr` streams into binary.

*ELF sections:*

- `.text`: executable machine code.
- `.rodata`: read-only constants (string literals, jump tables).
- `.data` / `.bss`: initialised and zero-initialised globals.
- `.eh_frame`: DWARF call-frame information for unwinding.

*Relocations:* references to external symbols (functions, globals) that the linker must patch. Position-independent code uses the *GOT* (Global Offset Table) for data and the *PLT* (Procedure Linkage Table) for function calls, allowing shared libraries to load at arbitrary addresses.

*Link-Time Optimisation (LTO):* with LTO, object files contain LLVM bitcode in addition to (or instead of) machine code. The linker passes all bitcode to a final LLVM compilation, enabling cross-file inlining, dead code elimination, and whole-program devirtualisation.

- *Full LTO:* all bitcode is merged into one module; optimal but slow and memory-intensive.
- *Thin LTO:* each module is compiled independently using a lightweight summary of other modules. Cross-module inlining uses function summaries; full function bodies are fetched lazily. Compilation is parallel; link time is comparable to traditional linking.

*Profile-Guided Optimisation (PGO):*

+ Compile with instrumentation: each branch and call site records a counter.
+ Run representative workloads.
+ Recompile with profile data: branch probabilities feed the scheduler and inliner; hot functions are placed in `.text.hot`; cold functions in `.text.unlikely`, improving instruction cache locality.

```
// Typical PGO workflow
clang -fprofile-generate -O2 -o app app.c
./app < representative_input
llvm-profdata merge -output=app.profdata default_*.profraw
clang -fprofile-use=app.profdata -O2 -o app_pgo app.c
```

== Further Reading

Aho, A. V., Lam, M. S., Sethi, R., Ullman, J. D. (2006). _Compilers: Principles, Techniques, and Tools_, 2nd ed. (Dragon Book.) Addison-Wesley.

Cooper, K. D., Torczon, L. (2022). _Engineering a Compiler_, 3rd ed. Morgan Kaufmann.

Rau, B. R. (1994). "Iterative Modulo Scheduling." _MICRO._

Lattner, C., Adve, V. (2004). "LLVM: A Compilation Framework for Lifelong Program Analysis and Transformation." _CGO._

Levine, J. R. (1999). _Linkers and Loaders._ Morgan Kaufmann.

DWARF Debugging Standard Committee. (2017). _DWARF Debugging Information Format_, Version 5.

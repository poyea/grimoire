= Register Allocation

Register allocation maps an unbounded set of virtual registers (or $"SSA"$ values) to a finite set of machine registers. It is one of the hardest problems in compiler backends --- optimal allocation is NP-complete in general --- and the quality of allocation directly determines instruction count, spill code, and ultimately runtime performance.

*See also:* _IR Design and SSA Form_, _Dataflow Analysis_, _Instruction Selection and Code Generation_

== The Problem

A compiler's middle-end works with an unlimited supply of virtual registers. The target machine has perhaps 16 or 32 general-purpose registers. Register allocation bridges this gap.

- *Virtual registers vs physical registers:* after $"SSA"$ deconstruction, each virtual register needs a home --- either a physical register or a stack slot.
- *Live ranges and interference:* two values _interfere_ if they are simultaneously live at any program point. They cannot share a register.
- *Spilling:* when no register is free at a definition or use, the value is written to a stack slot and reloaded on demand. Spill code is expensive: it adds memory traffic and increases code size.
- *Coalescing:* a move instruction `v1 = v2` is unnecessary if `v1` and `v2` occupy the same register. Coalescing attempts to merge the live ranges of move-connected values, eliminating the copy.

The allocator must assign registers while minimising spill code and eliminating as many copies as possible, subject to the target ABI's register constraints.

== Graph Colouring Formulation

The classical formulation due to Chaitin (1982) reduces register allocation to $k$-graph-colouring.

- *Interference graph:* nodes represent live ranges; an edge connects two nodes if they interfere. Two nodes may share a colour (register) only if they are not adjacent.
- *$k$-colouring:* assigning $k$ colours to the graph corresponds to allocating $k$ registers. Graph $k$-colouring for arbitrary $k$ is NP-complete. However, in register allocation $k$ is fixed at compile time (typically 8–32 physical registers), so the decision problem is polynomial for fixed $k$. The practical hardness arises from simultaneously optimising spilling, coalescing, and live-range splitting decisions; Chaitin's heuristic addresses this greedily rather than optimally.

*Chaitin's algorithm:*

+ *Build* the interference graph by computing live ranges for each virtual register.
+ *Coalesce* move-related pairs where it is safe to do so.
+ *Simplify:* repeatedly remove any node of degree $< k$ and push it on a stack. Such nodes can always be coloured regardless of how their neighbours are coloured.
+ *Spill selection:* if no node of degree $< k$ exists, choose a node to spill (push it marked as a potential spill) and continue.
+ *Select:* pop nodes from the stack; assign the lowest available colour. If a spill candidate cannot be coloured at select time, it becomes an actual spill.
+ *Spill code insertion:* insert stores at defs and loads before uses for each actual spill, then restart.

```
// Pseudocode: simplification phase
while !worklist.is_empty() {
    if let Some(n) = worklist.find(|n| degree(n) < K) {
        stack.push(n);
        remove_from_graph(n);
    } else {
        // No low-degree node: select a spill candidate
        let n = worklist.pick_spill_candidate();
        stack.push(Spill(n));
        remove_from_graph(n);
    }
}
```

*Spill heuristics:* a common metric is $"cost" / "degree"$ where cost counts the dynamic execution frequency of the value's defs and uses (weighted by loop depth). Spilling a high-degree but rarely used value reduces interference more cheaply than spilling a hot value.

== Linear Scan Allocation

Graph colouring is $O(n^2)$ in the size of the interference graph. For JIT compilers, compilation latency matters as much as code quality. Linear scan allocation (Poletto & Sarkar 1999) runs in $O(n log n)$.

*Algorithm:*

+ Compute a live interval $["start", "end"]$ for each virtual register --- the earliest definition to the latest use --- by a single pass over the linearised instruction sequence.
+ Sort intervals by start point.
+ Maintain an _active_ set of intervals currently occupying registers.
+ Scan left to right:
  - When an interval starts: expire active intervals whose end $<$ current position, freeing their registers. If a register is free, assign it. Otherwise, spill: evict the active interval with the largest end point if it ends after the current interval (the current interval gets the register; the evicted one is spilled), or spill the current interval.

```python
def linear_scan(intervals, num_regs):
    active = []   # sorted by end point
    free_regs = list(range(num_regs))
    allocation = {}

    for iv in sorted(intervals, key=lambda i: i.start):
        # Expire finished intervals
        for a in list(active):
            if a.end < iv.start:
                active.remove(a)
                free_regs.append(allocation[a])

        if not free_regs:
            # Spill: evict the interval ending latest
            spill = max(active, key=lambda a: a.end)
            if spill.end > iv.end:
                allocation[iv] = allocation[spill]
                del allocation[spill]
                active.remove(spill)
                spill.spilled = True
            else:
                iv.spilled = True
        else:
            allocation[iv] = free_regs.pop()
            active.append(iv)
            active.sort(key=lambda a: a.end)

    return allocation
```

*Limitations:* live intervals are conservative --- a value may not be live throughout its interval if there is a gap. Extended linear scan (Wimmer & Franz 2010) handles $"SSA"$ form by computing precise per-use intervals and splitting them.

Linear scan is used in the JVM client compiler (C1), early V8, and LLVM's fast register allocator (`-O0`).

== SSA-Based Allocation (LLVM's Greedy RA)

LLVM's default register allocator (`RegAllocGreedy`) operates on $"SSA"$ form after phi-node lowering and takes advantage of the structural properties of $"SSA"$.

- *Phi-web coalescing:* values connected by phi-nodes are candidates for the same physical register. If two values in a phi-web are simultaneously live, they must be in different registers; otherwise they can share one.
- *Live range splitting:* rather than spilling an entire live range, the allocator splits it at a cheap point (e.g., a call boundary or an infrequently executed edge), generating a shorter range that may be colourable.
- *Priority queue:* unallocated intervals are ordered by *spill weight* (def/use frequency × loop depth / interval length). High-weight intervals are allocated first.
- *Eviction:* if a high-priority interval needs a register occupied by a lower-priority one, the allocator evicts the lower-priority interval and requeues it for reallocation.
- *Rematerialisation:* before emitting a spill, the allocator checks whether the value is cheaper to recompute (see below).

The loop iterates: allocate, split or evict as needed, repeat until all intervals are assigned or confirmed spills. In practice, LLVM's greedy allocator produces code quality close to graph colouring at a fraction of the compile time.

== Spill Code and Rematerialisation

*Spilling* a live range $v$ inserts:

- A `store [sp + offset], v` immediately after each definition.
- A `v' = load [sp + offset]` immediately before each use, using a fresh virtual register `v'` with a short live range.

Stack slot assignment runs after the allocator: non-interfering spilled values can share a stack slot (they are never simultaneously live), reducing frame size.

*Rematerialisation* avoids the load entirely when the value is cheap to recompute:

- Integer constants: `mov rax, 42` is at least as fast as a load.
- Address calculations: `lea rax, [rip + global]` is typically a single cycle and has no memory latency.
- Simple arithmetic on loop-invariant inputs.

LLVM marks virtual registers with `isRematerializable` during lowering; the allocator substitutes recomputation for reload wherever the operands are still available.

*Partial spilling* spills a live range only across cold regions (e.g., exception handlers, cold branches) rather than throughout its entire extent. Combined with profile data, partial spilling concentrates spill traffic where it matters least.

== Coalescing

A move instruction `copy dst, src` is a no-op if `dst` and `src` are assigned the same physical register. The allocator tries to merge their live ranges.

*Aggressive coalescing:* merge unconditionally. Risk: the merged node may have degree $≥ k$, turning a colourable graph into one that requires spills.

*Conservative coalescing (Briggs 1994):* merge `a` and `b` only if the resulting node has fewer than $k$ high-degree neighbours ($"degree" ≥ k$). This guarantees the merged node is no harder to colour than the originals.

*George & Appel's Iterated Register Coalescing (IRC, 1996):* interleaves simplification and coalescing. Moves are classified as:

- *Coalesced:* safely merged; move eliminated.
- *Constrained:* the two nodes interfere; cannot merge.
- *Frozen:* cannot yet determine safety; mark as non-move-related and simplify.
- *Spilled:* spill candidate after coalescing fails.

IRC is the standard algorithm in production compilers. Copy propagation and $"SSA"$ coalescing before register allocation reduce the number of moves the allocator must handle.

== Calling Conventions and Register Windows

Every function call interacts with the allocator through the ABI.

*Caller-saved (volatile) registers* are not preserved across calls; the caller must save them if their values are needed after the call.

*Callee-saved (non-volatile) registers* must be restored before the function returns; the callee saves them on entry.

The allocator must account for call sites: values live across a call in caller-saved registers incur a save/restore, whereas callee-saved registers incur a prologue/epilogue cost amortised over the function. An allocator that prefers callee-saved registers for long-lived values reduces spill traffic across calls.

#table(
  columns: 3,
  [*ABI*], [*Argument registers*], [*Callee-saved*],
  [x86-64 SysV], [rdi, rsi, rdx, rcx, r8, r9], [rbx, rbp, r12--r15],
  [ARM64 (AAPCS64)], [x0--x7], [x19--x28, x29 (fp)],
  [RISC-V (LP64)], [a0--a7], [s0--s11],
)

*SPARC register windows* present a different model: each function gets a fresh window of registers; the `save`/`restore` instructions slide the window, making the caller's out-registers become the callee's in-registers. The allocator does not need to save registers explicitly, but the hardware must manage window overflow.

== Further Reading

Chaitin, G. J. et al. (1981). "Register Allocation via Coloring." _Computer Languages._

George, L., Appel, A. W. (1996). "Iterated Register Coalescing." _TOPLAS._

Briggs, P., Cooper, K. D., Torczon, L. (1994). "Improvements to Graph Coloring Register Allocation." _TOPLAS._

Poletto, M., Sarkar, V. (1999). "Linear Scan Register Allocation." _TOPLAS._

Wimmer, C., Franz, M. (2010). "Linear Scan Register Allocation on SSA Form." _CGO._

Braun, M. et al. (2013). "Register Allocation via Partitioned Boolean Quadratic Programming." (See also LLVM RegAllocGreedy source.) _CC._

Cooper, K. D., Torczon, L. (2022). _Engineering a Compiler_, 3rd ed. Morgan Kaufmann.

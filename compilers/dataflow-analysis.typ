#import "../template.typ": xref

= Dataflow Analysis

Dataflow analysis is the engineering core of every optimizer: it answers questions of the form "at this program point, what could be true about the program state?" by propagating facts through the control-flow graph until they stabilize. This chapter focuses on the algorithms and data structures used in production — bitvectors, sparse iteration, $"SSA"$-aware variants.

*See also:* #xref("compilers", "ir-design", label: "IR Design and SSA Form"), #xref("compilers", "optimisation-passes", label: "Optimisation Passes"), and the #xref("programming-languages", "operational-semantics", label: "Operational Semantics") chapter in the Languages & Compilers volume

== Lattices and Monotone Frameworks

A dataflow analysis is determined by:

- A *lattice* $(L, inter)$ of facts, with bottom $bot$ and top $top$.
- A *transfer function* $f_n : L -> L$ for each node $n$, monotone in $L$.
- A *direction* (forward or backward).

The solution at node $n$ is the least fixed point of

$ "IN"(n) = inter_(p in "preds"(n)) "OUT"(p), quad "OUT"(n) = f_n("IN"(n)) $

(forward) or symmetrically for backward analyses. Kildall's algorithm iterates until no $"IN"/"OUT"$ changes; convergence is guaranteed for finite lattices of bounded height with monotone transfer functions.

== Reaching Definitions

*Question:* at point $p$, which definitions $d : x = ...$ might supply the value of $x$ used at $p$?

*Lattice:* $L = 2^D$ (subsets of all definitions), join is union, $bot = emptyset$.

*Transfer:* $f_n(I) = "Gen"(n) union (I \\ "Kill"(n))$ where `Gen` is the set of definitions in $n$ and `Kill` is the set of definitions of any variable that $n$ redefines.

```cpp
struct BBInfo { BitSet gen, kill, in, out; };

void reaching_defs(CFG& cfg, std::vector<BBInfo>& info) {
    std::deque<int> worklist;
    for (int i = 0; i < cfg.size(); ++i) worklist.push_back(i);
    while (!worklist.empty()) {
        int b = worklist.front(); worklist.pop_front();
        BitSet in;
        for (int p : cfg[b].preds) in |= info[p].out;
        info[b].in = in;
        BitSet new_out = info[b].gen | (in - info[b].kill);
        if (new_out != info[b].out) {
            info[b].out = new_out;
            for (int s : cfg[b].succs) worklist.push_back(s);
        }
    }
}
```

*Cost.* The worklist visits each block at most $O(d)$ times where $d$ is the loop-nesting depth in reverse-post-order traversal. With bitvector operations the inner loop is $O(|D|/64)$ word operations.

== Live Variables

*Backward* analysis. A variable is _live_ at $p$ if its current value may be used along some path from $p$ before being overwritten.

$ "OUT"(n) = union_(s in "succs"(n)) "IN"(s), quad "IN"(n) = "Use"(n) union ("OUT"(n) \\ "Def"(n)) $

Liveness drives:

- Register allocation (interference graphs).
- Dead-store elimination.
- Spilling: a value not live across a call is cheap to recompute.

```cpp
void liveness(CFG& cfg, std::vector<BBInfo>& info) {
    std::deque<int> worklist;
    for (int i = cfg.size()-1; i >= 0; --i) worklist.push_back(i);
    while (!worklist.empty()) {
        int b = worklist.front(); worklist.pop_front();
        BitSet out;
        for (int s : cfg[b].succs) out |= info[s].in;
        info[b].out = out;
        BitSet new_in = info[b].use | (out - info[b].def);
        if (new_in != info[b].in) {
            info[b].in = new_in;
            for (int p : cfg[b].preds) worklist.push_back(p);
        }
    }
}
```

== Available Expressions and Very Busy Expressions

*Available* (forward, $inter$): expressions computed on every path. Drives common-subexpression elimination.

*Very busy* (backward, $inter$): expressions that will be used on every path from $p$. Drives code hoisting / partial redundancy elimination.

#table(
  columns: 5,
  [*Analysis*], [*Direction*], [*$inter$*], [*Initial $top$*], [*Initial entry/exit*],
  [Reaching defs], [forward], [$union$], [$emptyset$], [$emptyset$],
  [Liveness], [backward], [$union$], [$emptyset$], [$emptyset$],
  [Available exprs], [forward], [$inter$], [$U$ (all exprs)], [$emptyset$ at entry],
  [Very busy exprs], [backward], [$inter$], [$U$], [$emptyset$ at exit],
)

== Live Intervals

For register allocation, we collapse a variable's liveness across blocks into a single interval $[s, e]$ in a linearization of the program. Wimmer-Mössenböck (2005) showed how to compute live intervals directly from $"SSA"$ form without first running classical liveness:

```cpp
struct Interval { int start, end; std::vector<UsePoint> uses; };

void build_intervals(SSA& ssa, std::vector<Interval>& intervals) {
    auto& blocks = ssa.blocks_in_reverse_layout();
    std::unordered_map<int, BitSet> live_in;
    for (Block* b : blocks) {
        BitSet live;
        for (Block* s : b->succs) {
            live |= live_in[s->id];
            // values passed to s as phi/block args
            for (auto [arg, val] : s->phi_args_from(b)) live.set(val);
        }
        // every value live at exit covers the whole block
        for (int v : live) intervals[v].extend(b->begin, b->end);
        // walk instructions backward
        for (Instr* i = b->last; i; i = i->prev) {
            if (int d = i->def()) {
                intervals[d].start = i->pos;
                live.clear(d);
            }
            for (int u : i->uses()) {
                if (!live.test(u)) intervals[u].extend(b->begin, i->pos);
                live.set(u);
                intervals[u].uses.push_back({i->pos, i->kind});
            }
        }
        // loop fix-up: extend operands live throughout
        if (b->is_loop_header()) {
            for (int v : live) intervals[v].extend(b->begin, b->loop_end);
        }
        live_in[b->id] = live;
    }
}
```

Intervals are then sorted by start position for linear-scan allocation.

== Sparse Conditional Constant Propagation

$"SCCP"$ (Wegman-Zadeck 1991) propagates constants while simultaneously discovering unreachable branches. It operates on the $"SSA"$ value graph with three lattice points per variable: $bot$ (unknown), constant $c$, $top$ (overdefined).

```cpp
enum LatticeKind { Bot, Const, Top };
struct Lat { LatticeKind k; uint64_t c; };

Lat meet(Lat a, Lat b) {
    if (a.k == Bot) return b;
    if (b.k == Bot) return a;
    if (a.k == Top || b.k == Top) return {Top, 0};
    return a.c == b.c ? a : Lat{Top, 0};
}

void sccp(SSA& ssa) {
    std::deque<SSAEdge> ssa_wl;
    std::deque<CFGEdge> cfg_wl;
    std::vector<Lat> lat(ssa.num_values(), {Bot, 0});
    std::vector<bool> reachable(ssa.num_blocks(), false);

    cfg_wl.push_back({nullptr, ssa.entry});
    while (!cfg_wl.empty() || !ssa_wl.empty()) {
        if (!cfg_wl.empty()) {
            auto [from, to] = cfg_wl.front(); cfg_wl.pop_front();
            if (reachable[to->id]) continue;
            reachable[to->id] = true;
            for (Instr* i : to->instrs) visit_instr(i, lat, ssa_wl, cfg_wl);
        } else {
            SSAEdge e = ssa_wl.front(); ssa_wl.pop_front();
            if (reachable[e.user->block->id]) visit_instr(e.user, lat, ssa_wl, cfg_wl);
        }
    }
}
```

*Why "sparse"?* Classical constant propagation visits every basic block; $"SCCP"$ visits only $"SSA"$ defs whose operands change, and only $"CFG"$ blocks reachable from entry. Discovering a branch condition is constant disables the unreached successor, which may transitively disable many definitions.

*Conditional* in the name: by tracking reachability, $"SCCP"$ folds branches whose conditions become constant — equivalent to running constant folding and dead-code elimination simultaneously to a fixed point.

== Interprocedural and Context-Sensitive

The single-procedure dataflow framework extends with the *call-string* approach (Sharir-Pnueli 1981) or *functional* approach (encode the procedure's summary as a function on the input lattice). Both have exponential blow-up; practical IPA limits context to one or two calls (1-CFA, 2-CFA) or uses Andersen-style inclusion for points-to.

== Range Analysis and Bit-Tracking

Modern optimizers carry *known-bits* and *value-range* lattices alongside the constant lattice. LLVM's `ValueTracking` answers questions like "is this value non-negative?" or "are bits 31:16 zero?" cheaply.

```llvm
; Known bits propagation
%a = and i32 %x, 255          ; a: 0x000000ff known
%b = or  i32 %a, 256          ; b: 0x000001ff known (bit 8 set)
%c = shl i32 %b, 16           ; c: bits 31:24 zero, bit 24 set
```

InstCombine uses known-bits to fold `(x & 0xFF) >> 16 == 0` to `true`.

== Algorithmic Tricks

*Reverse post-order* visiting reduces forward analyses to $O((d+1)|"CFG"|)$ where $d$ is the depth of the loop nest. Most analyses converge in one or two RPO passes.

*Bitvector packing*. Live sets, available expression sets, and reaching definition sets are typically dense; pack them into `uint64_t` arrays and use SIMD for union/intersection on cold paths.

*$"SSA"$-based liveness* (Hack 2006) computes liveness in $O(|V| + |E| + |"uses"|)$ time without a fixed-point iteration: for each use, walk up the dominator tree from the use to the def, marking blocks live-through.

== Pointer/Alias Analysis

A dataflow analysis whose lattice is the powerset of abstract objects. Two ends of the spectrum:

#table(
  columns: 4,
  [*Algorithm*], [*Cost*], [*Precision*], [*Used by*],
  [Andersen (inclusion)], [$O(n^3)$], [flow-/context-insensitive], [LLVM (CFL-AA)],
  [Steensgaard (unification)], [near-linear], [coarse], [historically GCC],
  [DSA (data-structure)], [polynomial], [field-/heap-sensitive], [LLVM optional],
  [TBAA (type-based)], [$O(1)$ per query], [language-level], [LLVM/Clang default],
)

Production C/C++ compilers rely mostly on $"TBAA"$ derived from the source language's strict-aliasing rules, with a basic flow-insensitive pointer analysis as fallback.

== Worked Example: Sparse Constant Propagation on SSA

```llvm
; Before SCCP                       ; After SCCP
define i32 @f(i32 %x) {             define i32 @f(i32 %x) {
entry:                              entry:
  %c = icmp eq i32 1, 1               br label %T
  br i1 %c, label %T, label %F      T:
T:                                    ret i32 42
  %a = add i32 1, 41                F:                       ; unreachable removed
  br label %M
F:
  %b = add i32 %x, 0
  br label %M
M:
  %r = phi i32 [%a, %T], [%b, %F]
  ret i32 %r
}
```

After $"SCCP"$: `%c` becomes constant `true`, block `F` becomes unreachable, the $phi$ collapses to `%a = 42`, and the function returns the constant. Classical separate constant propagation + dead code elimination would need two iterations.

== Further Reading

Kildall, G. (1973). "A Unified Approach to Global Program Optimization." POPL.

Wegman, M., Zadeck, K. (1991). "Constant Propagation with Conditional Branches." TOPLAS (SCCP).

Sharir, M., Pnueli, A. (1981). "Two Approaches to Interprocedural Data Flow Analysis." In _Program Flow Analysis: Theory and Applications._

Andersen, L. (1994). "Program Analysis and Specialization for the C Programming Language." PhD thesis, Copenhagen.

Steensgaard, B. (1996). "Points-to Analysis in Almost Linear Time." POPL.

Wimmer, C., Mössenböck, H. (2005). "Optimized Interval Splitting in a Linear Scan Register Allocator." VEE.

Hack, S. (2006). "Register Allocation for Programs in SSA Form." PhD thesis, Karlsruhe.

Cooper, K., Torczon, L. _Engineering a Compiler_, ch. 9.

Khedker, U., Sanyal, A., Karkare, B. (2009). _Data Flow Analysis: Theory and Practice._ CRC Press.

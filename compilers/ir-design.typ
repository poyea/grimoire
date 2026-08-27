#import "../template.typ": xref

= IR Design and SSA Form <ir-design>

The choice of intermediate representation determines what optimizations are tractable. $"SSA"$ made dataflow sparse; $"CPS"$ made control explicit; sea-of-nodes erased the program order; $"MLIR"$ made the IR itself extensible. This chapter compares the practical tradeoffs of each.

*See also:* #xref("compilers", "dataflow-analysis", label: "Dataflow Analysis"), #xref("compilers", "optimisation-passes", label: "Optimisation Passes"), and the #xref("programming-languages", "operational-semantics", label: "Operational Semantics") chapter in the Languages & Compilers volume

== Why an IR?

A compiler frontend produces an $"AST"$ that mirrors the source language; a backend wants something close to the target. The IR sits between, satisfying:

- *Source-independent:* same IR from C, Rust, Swift after frontend lowering.
- *Target-independent:* same IR feeds x86, ARM, RISC-V.
- *Analyzable:* easy to compute defs/uses, reaching definitions, loop structure.
- *Transformable:* local rewrites preserve well-formedness.

LLVM IR is the textbook compromise; it is typed, $"SSA"$, with explicit control flow, but verbose. GCC's Gimple is similar but untyped. HotSpot's $"C2"$ uses sea-of-nodes; Cranelift uses a hybrid block-with-SSA form.

== Static Single Assignment

In $"SSA"$ every variable is assigned exactly once; control-flow merges are reconciled by $phi$-nodes.

```llvm
; Before SSA:                  ; After SSA:
;   x = 1                      ;   x1 = 1
;   if c:                      ;   br i1 %c, label %T, label %F
;     x = 2                    ; T: x2 = 2; br %M
;   else:                      ; F: x3 = 3; br %M
;     x = 3                    ; M: x4 = phi [x2,T], [x3,F]
;   use(x)                     ;   call use(x4)
```

*Why $"SSA"$ wins:*

- *Def-use chains are explicit and unique.* No iteration to find a variable's definition; the $"SSA"$ name _is_ the definition pointer.
- *Sparse analyses.* Constant propagation visits only the $"SSA"$ value graph, not every basic block.
- *Better register allocation.* SSA-form coloring exploits live-range chordality (Hack, Goos 2006).
- *Easier verifier.* Each name has one type, one definition site.

== Computing SSA: Cytron's Algorithm

The classical algorithm uses _dominance frontiers_ to place $phi$-nodes. A block $Y$ is in the dominance frontier of $X$ if $X$ dominates a predecessor of $Y$ but not $Y$ itself.

```cpp
// Compute dominance frontier
std::vector<BitSet> dom_frontier(const CFG& cfg, const DomTree& dt) {
    std::vector<BitSet> DF(cfg.num_blocks());
    for (Block* b : cfg.blocks()) {
        if (b->preds.size() < 2) continue;
        for (Block* p : b->preds) {
            Block* runner = p;
            while (runner != dt.idom(b)) {
                DF[runner->id].set(b->id);
                runner = dt.idom(runner);
            }
        }
    }
    return DF;
}

// Place phi-functions for variable v defined in blocks Defs[v]
void place_phis(SSAState& ssa, Variable v) {
    std::queue<Block*> worklist;
    BitSet has_phi, ever_on_worklist;
    for (Block* b : ssa.defs[v]) {
        worklist.push(b); ever_on_worklist.set(b->id);
    }
    while (!worklist.empty()) {
        Block* x = worklist.front(); worklist.pop();
        for (BlockId y : ssa.df[x->id]) {
            if (has_phi.test(y)) continue;
            ssa.blocks[y]->phis.emplace_back(v);
            has_phi.set(y);
            if (!ever_on_worklist.test(y)) {
                worklist.push(ssa.blocks[y]);
                ever_on_worklist.set(y);
            }
        }
    }
}
```

After placement, a second pass renames variables by depth-first traversal of the dominator tree, maintaining a stack per variable.

*Pruned $"SSA"$* skips $phi$-placement for dead variables; *semi-pruned* skips variables local to a block. Modern compilers use the linear-time algorithm of Sreedhar and Gao (1995) based on DJ-graphs, or Cooper-Harvey-Kennedy's iterative dataflow approach for simplicity.

== Out of SSA

Before register allocation, $phi$-nodes must be replaced by parallel copies on incoming edges — the "$phi$-to-copy" transformation. The naive scheme inserts copies at the end of each predecessor:

```
; phi  x4 = phi [x2, T], [x3, F]
; ⇒
; in T: x4 = x2
; in F: x4 = x3
```

Subtleties: *parallel copy semantics* (all $phi$s of a block read old values), *critical edges* must be split, and *swap problems* ($"x4" = phi["x5"], "x5" = phi["x4"]$) require a temporary. Sreedhar-Gao Method III is the standard algorithm.

== Continuation-Passing Style

$"CPS"$ makes control flow explicit by passing every continuation as an extra argument. It's the dual representation favored by ML compilers (SML/NJ, OCaml's now-defunct CPS frontend, Scheme).

```scheme
; direct style:
(define (fact n)
  (if (= n 0) 1 (* n (fact (- n 1)))))

; CPS:
(define (fact n k)
  (if (= n 0)
      (k 1)
      (fact (- n 1) (lambda (r) (k (* n r))))))
```

Every call is a tail call; every binding is a $lambda$. This dual perfectly captures $"SSA"$ — Appel showed that $"SSA"$ and $"CPS"$ are equivalent (1998): $phi$-functions in $"SSA"$ correspond to continuation parameters in $"CPS"$.

*ANF* (Administrative Normal Form, Flanagan et al. 1993) is the readable cousin: every non-trivial subexpression is named with a `let`.

```ocaml
(* direct *)            (* ANF *)
(f x) + (g y)           let a = f x in
                        let b = g y in
                        a + b
```

ANF and $"CPS"$ are interconvertible; modern functional compilers (MLton, GHC's STG, OCaml's Flambda 2) use ANF-like representations to keep let-bindings explicit while staying readable.

== Sea of Nodes

Click's $"C2"$ IR represents everything — data, control, and memory — as nodes in a single graph. There's no basic block list; "where" a computation lives is decided by a separate Global Code Motion pass.

```
;  Node kinds: Region, If, Phi, Add, Load, Store, Return, ...
;
;        Start
;          |
;        Region──┐
;         │      │
;       If(c)    Phi(x)
;       /  \      │
;     True False  │
;       \  /     Add
;       Region────┘
;          │
;        Return
```

*Properties:*

- *Single SSA value graph.* `Add` depends on its two `Phi` operands; that's it. No "what block am I in?"
- *Control and data unified.* `If` is a data node producing two control outputs; `Phi` consumes control + values.
- *Memory $"SSA"$.* `Load`/`Store` chained through a memory token, enabling alias-precise scheduling.

Click's Global Code Motion places each node at the latest legal block that still dominates its uses, then hoists loop-invariants out. The result is a precise schedule with no explicit block structure to maintain.

HotSpot's $"C2"$, Graal, and the V8 turbofan IR all descend from this design.

== MLIR and Multi-Level IRs

The lesson of LLVM IR: a single fixed IR is great for a C-like middle-end but terrible for everything else — tensor frameworks, GPU kernels, hardware synthesis, dataflow systems. MLIR (Lattner 2021) makes the IR itself extensible via *dialects*.

```mlir
// affine dialect: polyhedral loops
func.func @matmul(%A: memref<MxKxf32>, %B: memref<KxNxf32>,
                  %C: memref<MxNxf32>) {
  affine.for %i = 0 to M {
    affine.for %j = 0 to N {
      affine.for %k = 0 to K {
        %a = affine.load %A[%i, %k] : memref<MxKxf32>
        %b = affine.load %B[%k, %j] : memref<KxNxf32>
        %c = affine.load %C[%i, %j] : memref<MxNxf32>
        %p = arith.mulf %a, %b : f32
        %s = arith.addf %c, %p : f32
        affine.store %s, %C[%i, %j] : memref<MxNxf32>
      }
    }
  }
  return
}
```

Each *operation* belongs to a dialect (`affine`, `arith`, `gpu`, `llvm`, `tosa`, ...). *Lowering* progressively replaces high-level dialect ops with lower-level ones, ending at `llvm` for CPU or `nvvm`/`spirv` for accelerators.

Key data structures: *operations* (op-code + operands + results + regions + attributes), *regions* (lists of blocks), and *blocks* (lists of ops + block arguments — MLIR uses block arguments instead of $phi$-nodes, which is cleaner for non-CFG dialects).

== Block Arguments vs Phi-Nodes

Cranelift, Swift's SIL, and MLIR replace $phi$-nodes with block arguments: each block declares parameter names, and branches pass values.

```
;; LLVM (phi)              ;; Cranelift CLIF
;; B3:                     ;; B3(x4: i32):
;;   x4 = phi [x2,B1],     ;;
;;            [x3,B2]      ;;
;;   ...                   ;;   ...
;;                         ;; B1:
;;                         ;;   jump B3(x2)
;;                         ;; B2:
;;                         ;;   jump B3(x3)
```

Semantics identical; the syntactic shift makes parallel-copy semantics explicit at the branch and avoids the "all $phi$s read simultaneously" rule. Most new compilers adopt block arguments.

== Types in the IR

LLVM IR is _typed_: every value carries an `i32`, `i64`, `float`, `<4 x i32>`, pointer, or aggregate type. Recent LLVM moved to *opaque pointers* (`ptr` instead of `i32*`) — element types live on the load/store, not the pointer, matching how the backend always treated memory.

Untyped IRs (Gimple, classic three-address code) are simpler but lose type-based-alias-analysis cheaply. MLIR keeps types per dialect and uses *interfaces* (TypeID-keyed) for generic transformations.

== Comparison

#table(
  columns: 5,
  [*IR*], [*Form*], [*Used by*], [*Strengths*], [*Weaknesses*],
  [LLVM IR], [SSA + blocks, typed], [Clang, rustc, Swift], [ecosystem, RIRs], [verbose, monolithic],
  [Gimple], [SSA, untyped 3-addr], [GCC], [simple, fast], [no rich types],
  [Sea of Nodes], [graph, no blocks], [HotSpot C2, Graal], [great GVN/scheduling], [hard to debug],
  [CPS/ANF], [$lambda$-calculus], [SML/NJ, OCaml], [closures first-class], [poor for low-level],
  [MLIR], [multi-dialect SSA], [TF, IREE, CIRCT], [extensible], [steep learning curve],
  [Cranelift CLIF], [SSA + block args], [Wasmtime, rustc-cg-cranelift], [fast compile], [smaller opt suite],
)

== A Worked Example: LLVM IR

```llvm
define i32 @sum(i32* %a, i32 %n) {
entry:
  %cmp0 = icmp sgt i32 %n, 0
  br i1 %cmp0, label %loop, label %done

loop:
  %i   = phi i32 [0, %entry], [%i.next, %loop]
  %acc = phi i32 [0, %entry], [%acc.next, %loop]
  %p   = getelementptr i32, i32* %a, i32 %i
  %v   = load i32, i32* %p, align 4
  %acc.next = add i32 %acc, %v
  %i.next   = add nsw i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %done

done:
  %r = phi i32 [0, %entry], [%acc.next, %loop]
  ret i32 %r
}
```

Notice: explicit $"GEP"$ for pointer arithmetic, `nsw` flag for undefined-on-overflow semantics, alignment on `load`. The IR is verbose but every operation is unambiguous.

== Further Reading

Cytron, R. et al. (1991). "Efficiently Computing Static Single Assignment Form." TOPLAS.

Sreedhar, V., Gao, G. (1995). "A Linear Time Algorithm for Placing $phi$-Nodes." POPL.

Appel, A. (1998). "SSA is Functional Programming." SIGPLAN Notices.

Click, C., Paleczny, M. (1995). "A Simple Graph-Based Intermediate Representation." IR Workshop.

Flanagan, C. et al. (1993). "The Essence of Compiling with Continuations." PLDI.

Lattner, C. et al. (2021). "MLIR: Scaling Compiler Infrastructure for Domain-Specific Computation." CGO.

Hack, S., Grund, D., Goos, G. (2006). "Register Allocation for Programs in SSA-Form." CC.

LLVM Project. _LLVM Language Reference Manual._

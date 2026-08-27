#import "../template.typ": xref

= JIT Compilation and Runtime Systems <jit-and-runtimes>

Just-in-time compilation defers code generation to runtime, enabling optimisations based on observed execution profiles. JITs underpin JavaScript engines (V8, SpiderMonkey), the JVM (HotSpot, GraalVM), Python (PyPy), and increasingly ML frameworks (XLA, Triton, torch.compile). This chapter covers tiered compilation, deoptimisation, garbage collection interactions, and the LLVM ORC JIT API.

*See also:* #xref("compilers", "optimisation-passes", label: "Optimisation Passes"), #xref("compilers", "register-allocation", label: "Register Allocation"), #xref("compilers", "ir-design", label: "IR Design and SSA Form")

== Why JIT?

A compiler must choose when to compile: ahead of time (AOT) before deployment, or just in time (JIT) during execution.

*AOT* has unlimited compile time, can perform whole-program analysis, and produces statically-linked binaries with no startup cost. Its blind spot is dynamic information: type tags, loop trip counts, and hot paths that only emerge at runtime.

*JIT* trades startup latency for two classes of opportunity unavailable to AOT:

- *Runtime type information.* A JavaScript value tagged as `int32` in a particular call can be assumed `int32` for the duration of that hot path, enabling unboxed arithmetic.
- *Profile data.* A loop that actually executes $10^7$ times deserves aggressive vectorisation; a loop that runs once does not.

*Warmup cost* is the fundamental JIT liability. Every cycle spent compiling is a cycle not executing. Short-lived programs or programs with very short hot phases pay full compilation cost without recovering it. This motivates *tiered compilation*: get code running immediately (cheaply), then upgrade the hottest parts.

*Deoptimisation* is the safety valve. When a speculative assumption — "this argument is always int32" — is falsified at runtime, the JIT must roll back to a safe interpreted state. Without deoptimisation, speculative optimisations would be unsound.

*Interpreter baseline:* A pure interpreter has zero compile time and full deoptimisation support but runs 10--100x slower than optimised native code at steady state. The interpreter is tier 0 in every modern system.

== Tiered Compilation

Tiered systems run multiple compilers of increasing quality, moving functions to higher tiers as their invocation count crosses thresholds.

*V8 tiers (2024):*

#table(
  columns: 3,
  [*Tier*], [*Component*], [*Notes*],
  [0], [Ignition bytecode interpreter], [Counts invocations and loop back-edges],
  [1], [Sparkplug baseline JIT], [Compiles bytecode to native without optimisation; fast],
  [2], [Maglev mid-tier JIT], [Type-specialised; added in V8 v11],
  [3], [Turbofan optimising JIT], [Full speculation, inlining, GVN, range analysis],
)

Each tier collects *type feedback* via inline caches (ICs), recording observed operand types. When a function is promoted to Turbofan, it reads the IC state and generates type-specialised code.

*JVM HotSpot tiers:*

#table(
  columns: 3,
  [*Tier*], [*Component*], [*Notes*],
  [0], [Interpreter], [Profiles invocation and back-edge counts],
  [1], [C1 client compiler], [3-pass IR; fast; no speculation],
  [2], [C2 server compiler], [Sea-of-nodes IR; full global opts; threshold ~10 000 invocations],
)

The C1/C2 handoff uses *on-stack replacement* (OSR) to transition a running loop from interpreted to compiled code mid-execution.

== Speculative Optimisation and Guards

Speculative optimisation assumes a likely dynamic property and inserts a *guard* (runtime check) to validate it. If the guard passes, execution continues on the fast optimised path; if it fails, a *deopt* is triggered.

*Type specialisation:* the JIT observes that function `add(a, b)` always receives `int32` arguments. It generates:

```
guard: typeof(a) == int32 && typeof(b) == int32  →  deopt if fails
fast path: %r = iadd a, b                        ; unboxed integer add
```

Without the guard, the JIT would need a generic add that handles int, float, string, BigInt, and object coercion.

*Inline caching (IC):* a polymorphic call site first checks a small cache of previously-seen (type, target) pairs. A monomorphic IC has one entry; megamorphic sites with many types fall back to a hash table or the interpreter.

```
; inline cache for  obj.method()
if (obj.shape == cached_shape)
    call cached_method          ; fast path: direct call
else
    ic_miss_handler(obj)        ; slow path: look up and update cache
```

*On-stack replacement (OSR):* enables switching from interpreter to JIT (or JIT to deopt) while the function is on the call stack. The compiler emits a mapping from each OSR entry point (typically a loop header) to the interpreter frame layout, allowing mid-execution tier transitions.

== Deoptimisation

When a guard fails, the JIT must reconstruct the interpreter state at the point of failure and resume interpretation.

*Frame reconstruction* is the core problem. The optimising compiler may have:

- Scalar-replaced objects (individual fields in registers, no heap object).
- Reordered or eliminated instructions.
- Dead code not executed since the last safepoint.

The JIT emits *deopt metadata* at every guarded point: a table mapping each interpreter-level variable to its current location (register, stack slot, or rematerialisable expression). The deopt handler reads this table and rebuilds the interpreter frame.

*Deopt materialisation:* if a scalar-replaced object must be deoptimised, the handler allocates it on the heap and fills in the fields from the deopt metadata.

*Side-exit counters* track how often each deopt fires. If a guard fires repeatedly, the JIT can:

- *Patch* the code to stop speculating on that branch (binary patching, nop the fast path).
- *Recompile* with a broader type assumption or no speculation.

*Deopt storms* occur when a cascading series of deoptimisations causes thrashing between tiers. V8 mitigates this with *bailout budgets*: a function is banned from optimisation for an exponentially-increasing time after repeated deopt cycles.

== Garbage Collectors and JIT Interaction

A precise GC must know exactly which pointers are live at any point of execution. For JIT-compiled code this requires the compiler to emit *stack maps*.

*Safepoints* are program points at which the GC is allowed to run. For concurrent GCs, safepoints must be reached frequently (within a few milliseconds). The JIT inserts safepoint polls at loop back-edges and method entries:

```
; safepoint poll
load %flag, [safepoint_page]   ; if GC wants to stop the world,
                                ; this page is made inaccessible,
                                ; causing a signal/fault
```

At each safepoint, the compiler emits a *stack map* entry that records, for every stack slot and live register, whether it holds a GC-managed pointer.

*Write barriers* notify the GC when the mutator stores a pointer. Generational collectors need write barriers to track old-to-young pointers (remembered sets). Concurrent collectors (G1, ZGC, Shenandoah) need barriers to maintain snapshot invariants:

```
; generational write barrier (simplified)
store %new_val, [%obj + offset]
if (old_gen(%obj) && young_gen(%new_val))
    remembered_set_add(%obj)
```

*Read barriers* are needed by concurrent moving collectors (ZGC, Shenandoah): every pointer load must check whether the referent has been moved and, if so, return the forwarding pointer. This imposes overhead on every load but enables concurrent object relocation.

*Concurrent JIT compilation* means the JIT background threads compete with GC threads for CPU. The runtime must ensure: JIT-compiled code is not installed while a safepoint is in progress; newly-compiled code's stack maps are visible before the code executes.

== LLVM ORC JIT API

LLVM's *ORC* (On-Request Compilation) v2 framework provides a composable, thread-safe JIT construction kit. It replaced the older MCJIT (which did not support concurrent compilation) starting with LLVM 11.

Core abstractions:

- *JITDylib*: a logical dynamic library. Symbols are looked up in a defined search order across JITDylibs. The `MainJD` usually contains user-compiled code; a `ProcessSymbols` JITDylib exposes the host process's exported symbols.
- *MaterializationUnit*: a deferred compilation unit that will materialise symbols on demand.
- *IRLayer / ObjectLayer / CompileLayer*: pluggable pipeline stages. An IRLayer receives LLVM IR modules; a CompileLayer compiles them to object files; an ObjectLayer links them.

Lazy compilation example:

```cpp
auto JIT = LLJITBuilder().create();
// add IR module — functions not yet compiled
ExitOnErr(JIT->addIRModule(std::move(TSM)));
// lookup triggers materialisation (compilation + linking) of just that symbol
auto Sym = ExitOnErr(JIT->lookup("my_function"));
auto Fn = Sym.toPtr<int(int)>();
int result = Fn(42);   // first call: compile; subsequent calls: native speed
```

Concurrent compilation is supported via `ConcurrentIRCompiler`, which uses a thread pool. Functions not yet called are never compiled, keeping cold-start memory low.

ORC is used by: Julia (every function is JIT-compiled via ORC), Swift (LLDB expression evaluator), and Clang JIT (`clang -jit`).

== ML Compiler JITs

Machine learning workloads motivate a new generation of domain-specific JITs. The key difference from general-purpose JITs: shapes (tensor dimensions) and layouts (row-major, column-major, tiled) are often not known until runtime, and kernel fusion decisions depend on memory bandwidth, not instruction count.

*XLA (Accelerated Linear Algebra):* Google's compiler for TensorFlow and JAX. XLA represents computations as HLO (High-Level Optimizer) graphs, applies algebraic simplification and fusion at the HLO level, then lowers to LLVM IR (CPU) or NVPTX/cuBLAS (GPU). XLA is a JIT: shapes are not known at graph definition time; the first execution with a new shape triggers recompilation.

*Triton:* OpenAI's Python-level GPU kernel JIT. A Triton kernel is written in a Python-embedded DSL that describes blocked tiled operations over tensors. Triton compiles to MLIR's `triton` dialect, lowers through `triton_gpu` and `llvm` dialects, and emits PTX. The JIT infers the tile sizes and shared-memory layout that maximise occupancy for the target GPU.

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n)
    y = tl.load(y_ptr + offs, mask=offs < n)
    tl.store(out_ptr + offs, x + y, mask=offs < n)
```

*torch.compile (TorchDynamo + TorchInductor):* PyTorch's JIT pipeline (introduced in PyTorch 2.0). TorchDynamo intercepts Python bytecode at runtime, traces through it to build an FX graph of tensor operations, and hands the graph to a backend. TorchInductor lowers the graph to Triton (GPU) or C++/OpenMP (CPU) with loop fusion and tiling. Unlike XLA, TorchDynamo handles arbitrary Python control flow by falling back to eager execution when tracing fails.

*Why JIT for ML:* kernel fusion eliminates intermediate tensors and reduces memory bandwidth; tiling decisions depend on tensor shape, which changes per batch; hardware targets (CUDA compute capability, TPU generation) vary across deployments. Static compilation with AOT shapes is possible (TensorRT) but requires re-export for each shape and hardware configuration.

== Further Reading

Aycock, J. (2003). "A Brief History of Just-In-Time." ACM CSUR.

Wimmer, C., Franz, M. (2010). "Linear Scan Register Allocation on SSA Form." CGO.

Würthinger, T. et al. (2013). "One VM to Rule Them All." Onward!

LLVM Project. _ORC Design and Implementation._ llvm.org/docs/ORCv2.html

Lattner, C. et al. (2021). "MLIR: Scaling Compiler Infrastructure for Domain-Specific Computation." CGO.

Chen, T. et al. (2018). "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." OSDI.

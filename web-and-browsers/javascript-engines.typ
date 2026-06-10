= JavaScript Engines

A JavaScript engine must run a dynamically typed, garbage-collected, eval-capable language at near-native speed, starting from source text it has never seen, often with a budget of milliseconds before the user notices. The result is some of the most sophisticated systems software in existence: multi-tier JIT compilers, speculative optimization with deoptimization, hidden classes, inline caches, and concurrent garbage collectors. This chapter uses V8 (Chrome, Node.js) as the spine, with JavaScriptCore (Safari) and SpiderMonkey (Firefox) for contrast.

*See also:* _Browser Architecture_ (where V8 lives in the renderer), _Event Loop and Scheduling_ (how script execution is driven), _WebAssembly_ (the engines' second language), _Web Performance_ (script cost in the field).

== Parsing and Startup

Parsing is on the critical path of page load, so engines avoid doing it. V8 *pre-parses* (lazy-parses) functions on first encounter: a fast scanner verifies syntax and records scope information, but builds no AST; the full parse happens only when the function is first called. This is why immediately-invoked function expressions historically carried a `(`-prefix heuristic, and why huge bundles pay a parse tax even for code never executed.

Startup is further amortized with *snapshots* (V8 serializes a pre-initialized heap with built-ins, deserialized in under a millisecond) and *bytecode/code caching*: Chrome caches Ignition bytecode on disk keyed by script URL after a script is seen on two loads, cutting repeat-visit compile time by 40% or more.

== The Tiering Pipeline

All three major engines converged on multi-tier execution: cheap tiers start fast, expensive tiers make hot code fast, and profiling in lower tiers feeds speculation in higher ones.

#table(
  columns: 4,
  [*Engine*], [*Tier 1*], [*Middle tiers*], [*Top tier*],
  [V8], [Ignition (bytecode interpreter)], [Sparkplug (2021, baseline), Maglev (2023, mid-opt)], [TurboFan],
  [JavaScriptCore], [LLInt (interpreter)], [Baseline JIT, DFG JIT], [FTL (B3 backend)],
  [SpiderMonkey], [interpreter], [Baseline Interpreter + Baseline JIT], [WarpMonkey/Ion],
)

Ignition compiles each function to a compact register-machine bytecode (designed to minimize memory: bytecode is roughly 4–8× smaller than baseline machine code). Sparkplug compiles that bytecode to machine code in a single linear pass — no IR, no register allocation — purely to eliminate dispatch overhead. Maglev builds an SSA graph with feedback-driven types but skips TurboFan's heaviest analyses, filling the gap for code that is warm but not scorching. TurboFan builds a sea-of-nodes graph (replaced by the simpler *Turboshaft* CFG-based IR for its back half, 2023–2024), performs typed lowering, escape analysis, redundancy elimination, and emits optimized code.

Functions move up tiers when invocation and loop-iteration counters cross thresholds, and can be *on-stack replaced* (OSR) mid-loop so a hot loop doesn't have to return before benefiting.

== Hidden Classes and Inline Caches

The foundational trick, inherited from Self (Ungar and Smith's work at Stanford/Sun, 1980s–90s): although JavaScript objects are spec'd as dictionaries, real programs create objects in consistent shapes. Engines give each object a *hidden class* (V8: "map"; JSC: "structure"; SpiderMonkey: "shape") describing its layout. Adding a property transitions the object to a successor map; objects built by the same constructor in the same order share maps.

*Inline caches* (ICs) exploit this: each property-access site remembers the maps it has seen and the corresponding offset. States escalate from uninitialized → *monomorphic* (one map: a compare-and-load, a few instructions) → *polymorphic* (2–4 maps: a short dispatch chain) → *megamorphic* (fallback to a global hash-table cache, an order of magnitude slower). The optimizing tiers consume IC feedback to *speculate*: TurboFan compiles a monomorphic load as an unconditional offset load guarded by a map check.

Practical corollaries: initialize all properties in the constructor in a fixed order; don't `delete` properties (it can force dictionary mode); don't mix shapes at hot call sites.

== Speculation and Deoptimization

Optimized code is built on assumptions — this variable is a small integer, this map never changes, `Array.prototype` hasn't been patched. Each assumption is protected by a *guard* (an explicit check) or a *dependency* (the engine registers that this code must be discarded if, say, a prototype is mutated). When a guard fails, the engine *deoptimizes*: it reconstructs interpreter state (locals, stack, position) from the optimized frame's side data and resumes in Ignition, an OSR-exit in reverse. Repeated deopts at the same site mark the feedback as polluted and may permanently disable optimization for the function.

Number representation interacts heavily with speculation: V8 stores small integers as *Smis* (31-bit tagged integers on 64-bit platforms with pointer compression), boxing other numbers as heap doubles; arrays carry *elements kinds* (packed Smi → packed double → packed elements → holey variants) that only transition in one direction. Writing `1.5` into an integer array, or creating holes with `arr[1000] = x`, permanently degrades every subsequent access to that array.

== Garbage Collection

V8's *Orinoco* collector is generational, mostly parallel, and mostly concurrent:

- *Young generation* (scavenger): a semi-space copying collector, parallelized across threads; survivors of two scavenges are promoted. Most objects die young, so most GC work is proportional to survivors, not garbage.
- *Old generation*: concurrent marking (mutator keeps running, with write barriers tracking new edges), parallel sweeping, and incremental compaction. Typical marking pauses are under a millisecond; full pauses target single-digit milliseconds even on large heaps.
- *Unified heap*: Oilpan (Blink's C++ GC) and V8 trace each other's references in one marking phase, collecting DOM/JS cycles.

JavaScriptCore uses the *Riptide* concurrent collector with a "retreating wavefront" barrier; SpiderMonkey uses generational GC with incremental marking and compacting. All engines expose pressure through `FinalizationRegistry` and `WeakRef` (ES2021), deliberately under-specified to keep GC behavior unobservable.

== JavaScriptCore and SpiderMonkey Notes

JSC's *DFG* tier speculates from value profiles, and its top tier *FTL* originally lowered to LLVM (2014) before replacing it with the bespoke *B3* backend (2016) for 5× faster compile times. JSC pioneered *concurrent JIT* compilation off the main thread. SpiderMonkey's 2020 *WarpMonkey* rewrite replaced Ion's complex global type-inference system (TI) with transpiling the same *CacheIR* (a shared IC intermediate representation) that the baseline tiers use — simpler, more predictable, and faster on real sites, a notable case of a major engine deleting cleverness for the win.

== Pitfalls

- *Microbenchmarking lies*: a 10-line loop will be monomorphic, fully inlined, and possibly dead-code-eliminated; production code is polymorphic and cold. Measure real workloads (and beware the JIT warming up mid-measurement).
- *Hidden class divergence*: conditionally-added properties (`if (x) this.flag = true`) fork object shapes and poison ICs downstream.
- *`arguments`, `with`, `eval`, and getters on hot paths* inhibit or complicate optimization; `try/catch` no longer does (a 2017-era myth).
- *Megamorphic call sites in frameworks*: a single dispatch helper touching every component's objects becomes megamorphic by design; engines added megamorphic caches, but shape-stable code is still markedly faster.
- *Holding the main thread*: even a fast engine cannot help a 200 ms synchronous task; chunk work or move it to a worker (see _Event Loop and Scheduling_).
- *Premature `delete`*: prefer setting properties to `undefined` or using `Map` for genuinely dynamic key sets.

== Further Reading

- Hölzle, U., Chambers, C., & Ungar, D. (1991). Optimizing dynamically-typed object-oriented languages with polymorphic inline caches. _ECOOP_.
- Hölzle, U., Chambers, C., & Ungar, D. (1992). Debugging optimized code with dynamic deoptimization. _PLDI_.
- V8 blog (v8.dev): "Ignition" (2016), "Sparkplug" (2021), "Maglev" (2023), "Trash talk: the Orinoco garbage collector" (2019), "Pointer compression in V8" (2020).
- Pizlo, F. (2016). "Introducing the B3 JIT compiler"; (2017) "Introducing Riptide: WebKit's retreating wavefront concurrent garbage collector" (webkit.org).
- SpiderMonkey team (2020). "Warp: improved JS performance in Firefox 83" (hacks.mozilla.org).

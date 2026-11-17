#import "../template.typ": xref

= Whole-Program and Post-Link Optimisation <whole-program-optimisation>

Separate compilation is a scalability device: each translation unit compiles alone, in parallel, and only changed files recompile. It is also an optimisation barrier. A compiler working on one file cannot inline a function defined in another, cannot tell that a virtual call has only one possible target in the whole program, cannot drop a function nobody calls, and cannot lay out code by how often it runs together with code in other files. Large server applications, whose instruction working sets are megabytes and whose hot paths cross hundreds of files, lose a lot to this barrier: front-end stalls from instruction cache and TLB misses often account for a large share of their cycles. This chapter covers what linkers do beyond concatenation, full and Thin link-time optimisation, interprocedural optimisations with devirtualisation as the main example, profile collection, code layout algorithms with a C++ implementation of greedy function ordering, and post-link optimisers such as BOLT and Propeller.

*See also:* #xref("compilers", "backend-and-codegen", label: "Backend and Code Generation") (object files, LTO and PGO basics), #xref("compilers", "optimisation-passes", label: "Optimisation Passes") (inlining), #xref("cpu-architecture", "branch-prediction", label: "Branch Prediction"), #xref("performance-engineering", "cpu-profiling", label: "CPU Profiling").

== What the Linker Can Already Do

Even without LTO, a linker performs several program-wide optimisations on machine code:

- *Section garbage collection.* Compiling with `-ffunction-sections -fdata-sections` puts every function and object in its own section. With `--gc-sections`, the linker starts from the entry point and exported symbols, follows relocations, and discards every unreachable section.
- *COMDAT deduplication.* Inline functions and template instantiations are emitted in every translation unit that uses them, in COMDAT groups. The linker keeps one copy of each group.
- *Identical code folding (ICF).* Functions whose machine code and relocations are byte-identical (common for template instantiations over types with the same layout) are merged. *Safe* ICF skips functions whose address is taken, because C and C++ promise that distinct functions have distinct addresses.
- *Relaxation.* On architectures with several branch or address forms (RISC-V, x86 GOT loads), the linker rewrites instruction sequences to shorter ones once final addresses are known.

Linker speed matters for these builds as much as output quality. lld and mold parallelise symbol resolution, relocation processing, and output writing across cores and link large binaries several times faster than the original GNU ld, which shortens the edit-build cycle for every developer on a large code base.

== Link-Time Optimisation

With LTO, object files carry compiler IR (LLVM bitcode or GCC GIMPLE) instead of, or in addition to, machine code. At link time, the linker resolves symbols and hands the IR back to the compiler, which can now optimise across file boundaries.

*Full LTO* merges every module into one and runs the optimisation pipeline on it. It sees everything, but the merged module is a single unit of work: memory use grows with program size, the back end runs serially or with coarse partitioning, and any change to any file redoes the whole link-time compilation. For programs with millions of lines this means tens of gigabytes and tens of minutes per link.

*ThinLTO* splits the work into three phases:

+ *Compile.* Each translation unit is optimised as usual and written as bitcode together with a compact *summary*: for each function, its linkage, size, call edges (with profile hotness when available), referenced globals, and type-test information for devirtualisation.
+ *Thin link.* A serial but fast step reads only the summaries. It builds the global call graph and decides, for each module, which functions to *import* from other modules (small and hot callees, so they can be inlined), which symbols can be internalised (not referenced outside the program), which are dead, and which virtual calls can be devirtualised.
+ *Back ends.* Each module is optimised again in parallel, now with its imported function bodies and the global decisions applied, and compiled to machine code. Results are cached by a hash of the module and its imports, so after a small change only the affected modules recompile.

ThinLTO achieves most of the performance of full LTO at memory and time costs close to a normal build, which is why it is the default LTO mode for large C++ projects.

== Interprocedural Optimisations

Seeing the whole program enables a family of optimisations:

- *Internalisation.* A function not referenced outside the program becomes internal, so the optimiser may change its signature, specialise it, inline it everywhere and delete it, or drop it if unused.
- *Cross-module inlining.* The largest single win, because inlining exposes everything else: constant propagation into the callee, dead branch elimination, and better register allocation across the former call.
- *Interprocedural constant propagation and function specialisation.* If a function is always called with a constant argument, or with a few distinct constants, the compiler can fold it in or clone a specialised version per value.
- *Attribute inference.* Proving that a function does not write memory, does not throw, does not free its arguments, or always returns, and recording it as attributes that improve optimisation at every call site.
- *Global variable optimisation.* A global that is written once with a constant becomes a constant; a global used only in one function becomes a local.

== Devirtualisation

A virtual call is an indirect call through a vtable. It costs a load or two and an indirect branch, which modern predictors handle well, but its real cost is that it blocks inlining. Devirtualisation replaces it with a direct call when the target can be determined.

#table(
  columns: 3,
  [*Technique*], [*Reasoning*], [*Requires*],
  [Final classes and methods], [`final` rules out overriding], [Programmer annotation],
  [Class hierarchy analysis (CHA)], [If no class derived from the static type overrides the method, the call has one target], [Whole class hierarchy],
  [Rapid type analysis (RTA)], [Only classes that are actually instantiated can be receivers], [Whole program, including allocation sites],
  [Whole-program devirtualisation], [Type metadata on vtables plus the full link: if only one vtable matches, call directly; if a few, branch on vtable address], [LTO with `-fwhole-program-vtables`, visibility guarantees],
  [Speculative devirtualisation (indirect call promotion)], [Profile says one target dominates; emit a guarded direct call with the indirect call as fallback], [Profile data],
)

Speculative devirtualisation turns a profiled indirect call into a comparison and a direct call. In source terms, with `vptr_is` standing for the vtable-pointer comparison the compiler emits:

```cpp
// before: shape->area() with profile showing 95 percent of receivers are circle
double area_of(const shape* s) {
    if (s->vptr_is(&circle::vtable))                // emitted as a vtable address compare
        return static_cast<const circle*>(s)->circle::area();   // direct, inlinable
    return s->area();                               // original indirect call
}
```

The guard costs a load and a predictable branch, and the direct path can be inlined. The visibility requirement for whole-program devirtualisation is real: if a shared library loaded at run time can define new derived classes, the "only one target" conclusion is false. Compilers therefore apply it only to classes with hidden visibility or when told the program is closed.

== Collecting Profiles

Profile-guided optimisation needs a profile that reflects production. Three ways to get one:

#table(
  columns: 3,
  [*Method*], [*How*], [*Trade-off*],
  [Instrumentation (`-fprofile-generate`)], [Counters on edges or blocks in a special build], [Exact counts; the instrumented binary is slower and must be run on representative input],
  [Sampling (AutoFDO, SPGO)], [Sample production hardware with `perf` and last-branch records, map addresses back to source with debug information], [Free in production; less precise, degraded by optimisation changing the code],
  [Context-sensitive sampling (CSSPGO)], [Sampling plus pseudo-probes inserted by the compiler, with calling context], [Close to instrumentation quality at sampling cost],
)

Warehouse-scale operators favour sampling because the profile comes from real traffic at no extra cost, and a continuous profiling fleet can feed it into every release build automatically. The risk is *staleness*: when the code changes, profile records that no longer match are lost, so profiles must be refreshed regularly.

== Code Layout

For large binaries, where code sits in memory matters as much as what it does. Hot code spread across many pages causes instruction cache misses, iTLB misses, and poor use of the decoded-instruction cache; hot and cold code interleaved in one function wastes cache lines on cold bytes. Layout optimisation works at three levels:

- *Basic-block ordering* within a function: put the likely successor of each branch immediately after it (falling through rather than taken), and move cold blocks such as error handling to the end or out of the function. The ext-TSP model scores a layout by how many profiled edges become fall-throughs or short jumps, and greedy chain merging maximises it.
- *Hot/cold splitting*: move cold blocks into a separate function body in a cold section, shrinking the hot function.
- *Function ordering* across the binary: place functions that call each other frequently next to each other, and all hot functions together, so the hot working set occupies few pages (which can then be backed by huge pages).

Pettis and Hansen's 1990 algorithm for function ordering is still the reference point. Treat the call graph as an undirected graph weighted by call counts. Each function starts as its own chain. Repeatedly take the heaviest edge whose endpoints lie in different chains and merge the two chains, choosing the orientation that puts the two endpoints closest together.

```cpp
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

struct call_edge {
    int a, b;
    std::uint64_t weight;
};

std::vector<int> pettis_hansen_order(int functions, std::vector<call_edge> edges) {
    std::vector<std::deque<int>> chains(functions);
    std::vector<int> chain_of(functions);
    std::vector<std::uint64_t> heat(functions, 0);
    for (int f = 0; f < functions; ++f) { chains[f] = {f}; chain_of[f] = f; }
    for (const auto& e : edges) { heat[e.a] += e.weight; heat[e.b] += e.weight; }

    std::sort(edges.begin(), edges.end(),
              [](const call_edge& x, const call_edge& y) { return x.weight > y.weight; });

    for (const auto& e : edges) {
        int ca = chain_of[e.a], cb = chain_of[e.b];
        if (ca == cb) continue;
        auto& x = chains[ca];
        auto& y = chains[cb];
        // Orient the chains so e.a sits in the back half of x and e.b in the front half of y.
        const auto pos = [](const std::deque<int>& c, int f) {
            return static_cast<std::ptrdiff_t>(std::find(c.begin(), c.end(), f) - c.begin());
        };
        const auto half = [](const std::deque<int>& c) { return static_cast<std::ptrdiff_t>(c.size()); };
        if (pos(x, e.a) < half(x) / 2) std::reverse(x.begin(), x.end());
        if (pos(y, e.b) >= (half(y) + 1) / 2) std::reverse(y.begin(), y.end());
        for (int f : y) { x.push_back(f); chain_of[f] = ca; }
        y.clear();
    }

    std::vector<int> order_of_chains;
    for (int c = 0; c < functions; ++c)
        if (!chains[c].empty()) order_of_chains.push_back(c);
    auto chain_heat = [&](int c) {
        std::uint64_t h = 0;
        for (int f : chains[c]) h += heat[f];
        return h;
    };
    std::sort(order_of_chains.begin(), order_of_chains.end(),
              [&](int p, int q) { return chain_heat(p) > chain_heat(q); });

    std::vector<int> layout;
    for (int c : order_of_chains) layout.insert(layout.end(), chains[c].begin(), chains[c].end());
    return layout;
}
```

The heaviest call pairs end up adjacent, and chains are emitted hottest first, so the cold remainder of the program follows the hot region. Later algorithms refine the objective. C3 (call-chain clustering) uses directed edges and places each callee *after* its hottest caller, because a call jumps forward into the callee and returns; it also caps cluster sizes at a page so that clusters align with the iTLB. On large data-center applications it improved on Pettis-Hansen measurably.

== Post-Link Optimisation

Some layout decisions are best made after the final binary exists. The compiler's view of a basic block is an approximation of the machine code that will be emitted, inlining changes everything, and sampled profiles map most precisely to the final instructions.

*BOLT* (Meta) disassembles an already-linked binary, reconstructs control-flow graphs for its functions, attaches `perf` samples with last-branch records directly to machine-code edges, and then re-lays out basic blocks and functions (using ext-TSP and C3), splits hot and cold code, and rewrites the binary. Because it works on the binary, it needs no rebuild and sees exactly the code that runs. Applied to large data-center services already built with PGO and LTO it reported speedups of up to about 7 percent, and up to about 20 percent on compilers such as Clang and GCC, whose huge code footprints suffer most from front-end stalls.

*Propeller* (Google) reaches the same goal by relinking instead of rewriting. A profile of the binary drives the compiler to emit each basic block in its own section with layout directives, and the linker then places sections in the optimised order. Splitting the work this way keeps memory use bounded and allows distributed and incremental builds, which matters for binaries of hundreds of megabytes.

== Costs and Practice

Whole-program optimisation costs build time, build memory, and debuggability: stack traces from heavily inlined and reordered code are harder to read, and reproducible builds require fixing the profile as an input. A common production recipe for large C++ services is ThinLTO plus sampled PGO for every release build, with BOLT or Propeller added for the largest and hottest binaries, where a few percent of throughput is worth thousands of machines.

== Further Reading

Pettis, K. and Hansen, R. C. (1990). "Profile Guided Code Positioning." PLDI.

Dean, J., Grove, D. and Chambers, C. (1995). "Optimization of Object-Oriented Programs Using Static Class Hierarchy Analysis." ECOOP.

Bacon, D. F. and Sweeney, P. F. (1996). "Fast Static Analysis of C++ Virtual Function Calls." OOPSLA.

Johnson, T., Amini, M. and Li, X. D. (2017). "ThinLTO: Scalable and Incremental LTO." CGO.

Chen, D., Li, D. X. and Moseley, T. (2016). "AutoFDO: Automatic Feedback-Directed Optimization for Warehouse-Scale Applications." CGO.

Ottoni, G. and Maher, B. (2017). "Optimizing Function Placement for Large-Scale Data-Center Applications." CGO.

Newell, A. and Pupyrev, S. (2020). "Improved Basic Block Reordering." IEEE Transactions on Computers.

Panchenko, M. et al. (2019). "BOLT: A Practical Binary Optimizer for Data Centers and Beyond." CGO.

Shen, H. et al. (2023). "Propeller: A Profile Guided, Relinking Optimizer for Warehouse-Scale Applications." ASPLOS.

Ayers, G. et al. (2019). "AsmDB: Understanding and Mitigating Front-End Stalls in Warehouse-Scale Computers." ISCA.

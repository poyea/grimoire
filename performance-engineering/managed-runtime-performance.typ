#import "../template.typ": xref

= Managed Runtime Performance <managed-runtime-performance>

A large share of production software runs on managed runtimes: the JVM, .NET, Go, JavaScript engines, and the Python and Ruby interpreters. These runtimes allocate and free memory automatically, compile code while it runs, and insert safety checks, and each of those services has a cost profile that native-code intuition does not predict. Allocation is nearly free while collection may not be; the first thousand calls of a method run many times slower than the millionth; an innocent interface call may block all inlining. This chapter covers warm-up and tiered compilation, allocation and the generational hypothesis, the main garbage collector designs and their trade-offs, tuning heap size and pause targets, escape analysis, boxing and object layout, call-site polymorphism, and how to measure managed code without fooling yourself.

*See also:* #xref("compilers", "jit-and-runtimes", label: "JIT Compilation and Runtime Systems"), #xref("compilers", "garbage-collection", label: "Garbage Collection"), #xref("web-and-browsers", "javascript-engines", label: "JavaScript Engines"), #xref("performance-engineering", "benchmarking", label: "Benchmarking"), #xref("performance-engineering", "memory-performance", label: "Memory Performance").

== Warm-Up

A JIT-compiled runtime starts by interpreting bytecode (or running a quick baseline compiler), collects profiles of types and branches, and recompiles hot methods with an optimising compiler that speculates on those profiles. HotSpot's tiered compilation goes through the interpreter, the C1 compiler with profiling, and the C2 (or Graal) optimising compiler; .NET has tier-0 and tier-1 with on-stack replacement; V8 has Ignition, Sparkplug, Maglev and TurboFan.

The consequences for performance work:

- *Start-up and warm-up costs.* A service may need tens of seconds to minutes of traffic before reaching steady-state speed, and its first requests are slow. Rolling deployments and autoscaling that send full traffic to a cold instance produce latency spikes.
- *Steady state is not guaranteed.* A study of VM benchmarks found that many never reach a stable steady state at all, oscillating or slowing down after deoptimisations, so "run until warm" is an assumption to check, not a fact.
- *Deoptimisation.* When a speculation fails (a new type appears at a call site, an uncommon branch is taken), the runtime discards compiled code and falls back to the interpreter until it recompiles, causing sudden slowdowns long after start-up.

Remedies trade flexibility for start-up speed: class data sharing and ahead-of-time caches (the JVM's CDS and Project Leyden's AOT cache), checkpoint and restore of a warmed process (CRaC), ahead-of-time compilation (GraalVM native image, .NET ReadyToRun and Native AOT), and synthetic warm-up traffic before an instance joins the load balancer.

== Allocation Is Cheap, Collection Is Not

Most managed runtimes allocate by *bump pointer* in a thread-local allocation buffer (TLAB): a pointer increment and a bounds check, about as cheap as stack allocation. The cost appears later, in garbage collection, and it is proportional to different things for different collectors:

- *Tracing* collectors do work proportional to the *live* data they trace, plus, for non-moving collectors, sweeping proportional to the heap.
- *Copying and compacting* collectors also move live objects, which costs bandwidth but leaves free memory contiguous for bump allocation.

The *generational hypothesis* says most objects die young. Generational collectors exploit it by allocating into a small young generation and collecting it often: since few objects survive, a young collection copies little and reclaims a lot. Objects that survive several collections are promoted to the old generation, which is collected rarely. Two rates then drive cost: the *allocation rate* determines how often young collections run, and the *promotion rate* how fast the old generation fills. Allocating a temporary per request is cheap; allocating long-lived caches that churn (objects that live just long enough to be promoted and then die) is expensive, because it defeats the hypothesis.

== Collector Designs

#table(
  columns: 4,
  [*Collector*], [*Design*], [*Pauses*], [*Trade-off*],
  [Parallel (JVM)], [Stop-the-world generational, multi-threaded copying and compaction], [Proportional to live data; can be seconds on large heaps], [Best throughput for batch work],
  [G1 (JVM default)], [Region-based, concurrent marking, incremental evacuation of regions chosen to meet a pause target (200 ms by default)], [Tens to hundreds of ms], [Balanced; pause target is a goal, not a guarantee],
  [ZGC, Shenandoah (JVM)], [Concurrent marking *and* concurrent relocation using load barriers on reference reads], [Under a millisecond, independent of heap size], [Barrier cost on every reference load and more CPU overall],
  [Go], [Concurrent tri-colour mark-sweep, non-moving, with a pacer], [Sub-millisecond stop-the-world phases], [No compaction (fragmentation), mark assists slow allocating goroutines],
  [.NET], [Generational with workstation and server modes, background collection of the old generation], [Short gen-0/1 pauses; background gen-2], [Server mode uses more memory for throughput],
  [V8 (Orinoco)], [Generational, parallel scavenger, concurrent marking, incremental compaction], [A few ms, scheduled in idle time], [Tuned for interactive pages],
)

The *concurrent compacting* collectors changed the landscape for latency-sensitive JVM services: pauses no longer grow with heap size, so multi-hundred-gigabyte heaps are practical with sub-millisecond pauses. Their costs are a *barrier* on every reference load (a check that the reference is up to date, taking a slow path while relocation is in progress), extra CPU spent on concurrent work, and the need for headroom: if the application allocates faster than the collector can reclaim memory concurrently, allocation stalls. Generational modes of ZGC (default since JDK 23) reduce that risk by collecting young objects more often.

Go chose a different point: a non-moving collector keeps pointers stable for C interoperability and simplifies the runtime, and value types and stack allocation keep allocation rates low. Its *pacer* starts each cycle early enough to finish before the heap reaches its target; when the application allocates faster than marking proceeds, allocating goroutines are drafted into *mark assist*, which appears as latency in exactly the goroutines that allocate most.

== Tuning Heap Size and Pause Goals

The fundamental trade-off is between memory and CPU. A larger heap means fewer collections, so less total GC work, at the cost of footprint and, for stop-the-world collectors, longer individual pauses.

- *Go*: `GOGC` (default 100) sets the heap target as live heap times $1 + "GOGC" \/ 100$, so with 1 GB live the heap grows to 2 GB before the next cycle. Doubling `GOGC` roughly halves GC CPU and adds memory. `GOMEMLIMIT` sets a soft total-memory ceiling, letting the collector run more often only when memory is tight, which is the right way to run in containers.
- *JVM*: size the heap explicitly (`-Xms` equal to `-Xmx` avoids resizing pauses), give it two to three times the live set, choose the collector by goal (Parallel for throughput, G1 for balance, ZGC for latency), and use container-aware sizing (`-XX:MaxRAMPercentage`) rather than defaults computed for the host.
- *Read the GC logs.* Unified logging (`-Xlog:gc*`) or `GODEBUG=gctrace=1` shows pause times, frequency, and heap occupancy after collection, which is the live set. Tuning without them is guessing.

The most effective GC tuning is usually reducing allocation: a profiler's allocation view (async-profiler, JFR, Go's `pprof` allocation profile) points at the few call sites that allocate most of the garbage.

== Escape Analysis, Boxing, and Layout

Optimising compilers try to avoid heap allocation altogether. *Escape analysis* determines whether an object can be reached outside the method (or the inlined region) that creates it; if not, the JIT can *scalar-replace* it, keeping its fields in registers. This works only after inlining, so it is fragile: a call that fails to inline, or an object stored into a field or passed to an unknown method, makes it escape. Go performs escape analysis at compile time and reports decisions with `go build -gcflags=-m`, which shows exactly which values move to the heap and why.

*Boxing* is a frequent hidden cost. In Java, generic collections hold references, so a `List<Integer>` of a million numbers is a million separate objects, each with a header, plus an array of pointers to them: roughly four to five times the memory of an `int[]` and a cache miss per element during iteration. Primitive collections, arrays of primitives, and value types (.NET `struct`, Go values, the JVM's upcoming value classes) avoid it.

Object layout matters for cache behaviour. A HotSpot object carries a 12-byte header with compressed class pointers (8-byte headers are arriving with compact object headers), fields are aligned and reordered by the JVM, and every object reference is a potential cache miss. Data-heavy code (analytics engines, caches) often moves its data *off-heap* into large byte buffers or memory segments, managing layout explicitly and making the data invisible to the collector.

== Call Sites and Polymorphism

JIT compilers inline aggressively, guided by profiles of which types actually occur at each call site:

- *Monomorphic* (one observed type): the call is inlined behind a cheap type guard.
- *Bimorphic* (two types): both targets inlined behind guards (HotSpot's default limit).
- *Megamorphic* (more): a virtual or interface dispatch with no inlining, which also blocks escape analysis and other optimisations downstream.

A call site becomes megamorphic through *profile pollution*: a shared utility method called with many types makes its internal call sites megamorphic for every caller. Benchmarks that exercise one type look fast, while production with many types does not. JavaScript engines use the same idea with inline caches keyed on hidden classes, and code that creates objects with differing property orders turns monomorphic sites polymorphic.

== Measuring Managed Code

Measurement in managed runtimes has specific traps:

- *Dead-code elimination* removes benchmark work whose result is unused, and *constant folding* precomputes it. Benchmark harnesses (JMH for the JVM, BenchmarkDotNet, Go's `testing.B`) provide sinks and warm-up handling.
- *Warm-up and run-to-run variance*: a JIT makes different decisions in different runs, so steady-state performance varies between JVM invocations, not just between iterations. Run multiple forks.
- *Safepoint bias*: many JVM profilers sample only at safepoints (points where the runtime can stop threads), so they attribute time to the wrong code. Profilers built on `AsyncGetCallTrace` and `perf_events` (async-profiler) or JDK Flight Recorder avoid it.

In Java, a JMH benchmark that measures an operation correctly looks like this:

```java
@State(Scope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(3)
public class ParseBenchmark {
    private String input;

    @Setup
    public void setup() { input = "1234567890"; }

    @Benchmark
    public void parse(Blackhole bh) {
        bh.consume(Long.parseLong(input));   // the sink defeats dead-code elimination
    }
}
```

Reading the input from a field rather than a literal stops the JIT from constant-folding the parse, `Blackhole` stops it from discarding the result, and three forks expose run-to-run variation from JIT decisions.

== Pitfalls

- Sending production traffic to cold instances, or measuring before warm-up and drawing steady-state conclusions.
- Tuning collector flags before reducing the allocation rate.
- Default heap sizing inside containers, leading to out-of-memory kills or wasted memory.
- Boxed collections of primitives in hot data structures.
- Shared generic helpers that make hot call sites megamorphic.
- Profilers with safepoint bias.

== Further Reading

Jones, R., Hosking, A. and Moss, E. (2023). _The Garbage Collection Handbook_ (2nd edition). CRC Press.

Georges, A., Buytaert, D. and Eeckhout, L. (2007). "Statistically Rigorous Java Performance Evaluation." OOPSLA.

Barrett, E. et al. (2017). "Virtual Machine Warmup Blows Hot and Cold." OOPSLA.

Mytkowicz, T. et al. (2010). "Evaluating the Accuracy of Java Profilers." PLDI.

Detlefs, D. et al. (2004). "Garbage-First Garbage Collection." ISMM.

Flood, C. H. et al. (2016). "Shenandoah: An Open-Source Concurrent Compacting Garbage Collector for OpenJDK." PPPJ.

Yang, A. M. and Wrigstad, T. (2022). "Deep Dive into ZGC: A Modern Garbage Collector in OpenJDK." ACM Transactions on Programming Languages and Systems.

Hudson, R. L. (2018). "Getting to Go: The Journey of Go's Garbage Collector." Go blog, reference.

Hölzle, U., Chambers, C. and Ungar, D. (1991). "Optimizing Dynamically-Typed Object-Oriented Languages with Polymorphic Inline Caches." ECOOP.

Shipilëv, A. (2024). "Java Microbenchmark Harness (JMH)." OpenJDK documentation.

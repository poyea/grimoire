= CPU Profiling

Profiling answers the question a benchmark cannot: _where_ does the time go. This chapter covers sampling and instrumentation profilers, the Linux `perf` and eBPF toolchains, flame graphs, off-CPU analysis, hardware performance counters and Intel's top-down methodology, and how profiles feed back into the compiler.

*See also:* _Benchmarking_ (establishing that there is a regression worth profiling), _Memory Performance_ (when the profile says "stalled on memory"), and the CPU Architecture volume's _Performance Analysis and Measurement_ and _Out-of-Order Execution_ chapters (what the PMU events actually count).

== Sampling vs. Instrumentation

An *instrumenting* profiler inserts code at function entry and exit (or relies on the runtime to do so) and records exact call counts and durations. It is precise about counts but distorts what it measures: the probe overhead is per-call, so cheap, hot functions are penalized most, which skews the profile exactly where it matters. `gprof`, `callgrind` (which simulates rather than instruments, at roughly $50 times$ slowdown), and many APM tracers fall in this family.

A *sampling* profiler interrupts the program at a fixed rate (commonly 99 Hz or 997 Hz, prime to avoid lockstep with periodic activity) and records the stack. Overhead is per-sample, typically under 1-2%, independent of call frequency, and the result converges to the true time distribution. The cost is statistical: a function consuming 0.1% of time needs around $10^4$ samples to be measured to $plus.minus 30%$. Sampling is the default for production work; instrumentation when exact counts (calls, allocations) are the question.

=== Stack unwinding

Samples are only as good as their stacks. Three mechanisms: *frame pointers* (cheap and reliable, but compilers omit them by default; recompiling with `-fno-omit-frame-pointer` costs about 1-2% and major fleets, including Meta and Netflix, accept it fleet-wide), *DWARF* unwinding (no recompilation, but slow and memory-hungry in the kernel, so `perf` defers it to userspace by copying stack snapshots), and *LBR* (last branch record, the CPU's own 32-entry branch history, limited depth but zero software cost). JIT runtimes need extra help: `perf-map-agent` for the JVM, `--perf-basic-prof` for Node.js.

== The Linux Toolchain

=== perf

`perf` is the kernel's standard profiler. The core workflow:

```
perf record -F 99 -g -p <pid> -- sleep 30   # sample at 99 Hz with stacks
perf report                                  # interactive TUI
perf stat -d ./program                       # counter summary
perf top                                     # live system-wide view
```

`perf stat` prints instructions, cycles, "IPC", cache misses, and branch mispredictions; an "IPC" well below the machine's width (4-6 for modern big cores) signals stalls worth decomposing with top-down analysis. `perf c2c` finds cache-line contention; `perf sched` analyzes scheduler latency.

=== eBPF and bpftrace

eBPF attaches sandboxed programs to kernel and user events, aggregating *in the kernel* so only summaries cross to userspace, which makes always-on production tracing affordable. The `bcc` collection and `bpftrace` one-liners cover most needs: `profile` (timed stack sampling), `offcputime`, `biolatency` (block I/O latency histograms), `runqlat` (scheduler queue latency), `funclatency`. Example:

```
bpftrace -e 'profile:hz:99 { @[ustack] = count(); }'
```

=== VTune and vendor tools

Intel VTune Profiler adds microarchitectural analyses on top of the PMU: a guided top-down view, memory-access analysis with per-load latencies (via PEBS precise sampling), and threading analysis. AMD uProf and Arm Streamline are the counterparts. PEBS/IBS precise events matter because ordinary PMU interrupts suffer *skid*, attributing the sample several instructions after the true culprit.

== Flame Graphs

Brendan Gregg's *flame graph* (2011) visualizes a collection of stack samples: each box is a frame, width is proportional to the number of samples containing that frame, the y-axis is stack depth, and the x-axis is alphabetical (not time). Wide plateaus at the top are where CPU time is actually spent; wide frames lower down show expensive subsystems. Variants:

- *On-CPU* flame graphs from `perf record` samples, the default.
- *Off-CPU* flame graphs, where width is blocked time instead of samples.
- *Differential* flame graphs, coloring frames red/blue by regression vs. a baseline, the fastest way to localize "what changed between these two builds".

Generated with the original `flamegraph.pl` scripts, `perf script report flamegraph`, or natively by tools like `cargo flamegraph`, py-spy, and async-profiler.

== Off-CPU Analysis

A CPU profiler is blind to time spent *not* running: blocked on locks, disk, network, or the scheduler queue. For latency-sensitive services this is often the majority of wall-clock time. *Off-CPU analysis* instruments the scheduler (eBPF `offcputime` hooks context-switch events) and records, for each blocked period, the stack and the duration. The result is the complement of the CPU profile; combined views (Gregg's "hot/cold" flame graphs, or wall-clock profilers like async-profiler's wall mode and py-spy's `--idle`) show both. Caution: off-CPU events are far more frequent than profile samples; eBPF aggregation and a minimum-duration filter keep overhead tolerable.

== Hardware Counters and Top-Down Analysis

The PMU (performance monitoring unit) counts microarchitectural events: cycles, instructions, cache hits and misses per level, branch mispredictions, TLB misses, stall cycles. Raw event lists are enormous and model-specific; the *Top-down Microarchitecture Analysis Method* (TMAM, Yasin 2014) organizes them into a decision tree over pipeline *slots* (issue opportunities, $4-6$ per cycle):

$ "slots" = "Retiring" + "Bad Speculation" + "Frontend Bound" + "Backend Bound" $

- *Retiring*: useful work. High retiring with poor performance suggests an algorithmic or vectorization problem, not a stall problem.
- *Bad speculation*: slots wasted on mispredicted paths; pursue branch structure.
- *Frontend bound*: instruction fetch/decode starvation; large code footprint, i-cache and i-TLB misses; consider PGO, BOLT, hugepages for text.
- *Backend bound*: execution stalls, subdivided into *memory bound* (cache misses, bandwidth) and *core bound* (execution-port pressure, long dependency chains).

`perf stat --topdown` and VTune compute the level-1 breakdown directly; a healthy compute kernel retires more than 50% of slots, while a typical pointer-chasing server application can sit below 30% with memory-bound dominating. The method's value is pruning: it tells you which detailed analysis _not_ to do.

== Profile-Guided Optimization

Profiles can feed the compiler. *PGO* (`-fprofile-generate` / `-fprofile-use` in GCC/Clang) uses execution counts to drive inlining, branch layout (hot path fall-through), register allocation, and basic-block ordering; typical gains are 5-15% on large branchy binaries. *AutoFDO* removes the instrumented-build step by deriving counts from production `perf` LBR samples. *BOLT* (Panchenko et al., 2019, now in LLVM) reorders basic blocks and functions in the final binary, attacking frontend-bound stalls directly; Meta reports 2-8% on top of PGO for large services. Google's fleet-wide profiler feeds AutoFDO continuously, making the optimization loop fully automatic.

== Continuous Profiling

Profiling only during incidents loses the baseline. *Continuous profiling* samples the whole fleet at low frequency (for example 99 Hz, around 0.5% overhead) and stores profiles over time, enabling "compare this week against last week" diffs and attributing cost per service and per line. Google's "Google-Wide Profiling" (Ren et al., 2010) pioneered this; current implementations include Parca, Grafana Pyroscope, Datadog and Elastic continuous profilers, most built on eBPF with frame-pointer or DWARF-less unwinding. The pprof format and flame-graph diff are the lingua franca. As a side effect, continuous profiling has become a FinOps tool: the profile *is* the cost breakdown.

== Further Reading

- Gregg, B. (2016). The flame graph. _CACM_, 59(6).
- Yasin, A. (2014). A top-down method for performance analysis and counters architecture. _ISPASS_.
- Ren, G. et al. (2010). Google-Wide Profiling: a continuous profiling infrastructure for data centers. _IEEE Micro_, 30(4).
- Panchenko, M. et al. (2019). BOLT: a practical binary optimizer for data centers and beyond. _CGO_.
- Gregg, B. (2019). _BPF Performance Tools_. Addison-Wesley.

= Benchmarking

A benchmark is an experiment, and most benchmarks are bad experiments: they measure the wrong thing, on the wrong workload, with no error bars, and then generalize. This chapter covers the failure modes of microbenchmarks, the tools that mitigate them, the statistics needed to claim "A is faster than B", and how to run experiments safely in production.

*See also:* _Performance Methodology_ (what to measure and why), _CPU Profiling_ (finding where time goes once a benchmark shows a regression), and the CPU Architecture volume's _Branch Prediction_ and _Caches_ chapters (the microarchitectural state that makes small benchmarks lie).

== Why Microbenchmarks Lie

A microbenchmark isolates a small piece of code, which means it also isolates it from the conditions under which it normally runs. The classic failure modes:

=== Dead code elimination

Optimizing compilers and JITs remove computation whose result is unused. A loop that computes a value and discards it benchmarks an empty loop. The fix is a *black hole*: JMH's `Blackhole.consume()`, criterion/Google Benchmark's `black_box()` / `DoNotOptimize()`, which tells the compiler the value escapes. Symmetrically, *constant folding* precomputes results from inputs the compiler can see; inputs must be read from non-constant state.

=== JIT warmup and tiered compilation

On the JVM, code starts interpreted, gets compiled by C1, then recompiled by C2 with profile-guided speculation; the first thousands of iterations measure the compiler, not the code. Steady state may take seconds to reach, and *deoptimization* can knock the code back. JMH runs explicit warmup iterations and forks fresh JVMs per trial because JIT decisions are path-dependent: the order in which call sites warm up changes which methods get inlined (monomorphic vs. megamorphic dispatch).

=== Alignment and layout

Code and data placement is a hidden variable. Mytkowicz et al. (2009), in "Producing wrong data without doing anything obviously wrong!", showed that changing the UNIX environment size (which shifts the stack and thus alignment) changes SPEC benchmark performance by as much as 33%, more than many published optimizations. Link order, branch placement relative to 32-byte fetch boundaries, and heap layout all contribute. Mitigations: randomize layout across runs (the Stabilizer approach by Curtsinger & Berger) or at least vary it and report the spread.

=== Other classics

- *Cache state*: the benchmark's working set fits in L2; production's does not.
- *Branch predictor training*: a loop over sorted data trains the predictor perfectly; real input is not sorted.
- *Frequency scaling*: turbo boost gives the first seconds a higher clock; thermal throttling takes it away. Pin the frequency (`cpupower frequency-set`) or report it.
- *Interference*: cron jobs, other tenants, SMT siblings. Use `cset shield` / `isolcpus` or at least `taskset`.

== Tooling

#table(
  columns: 3,
  [*Tool*], [*Domain*], [*Key features*],
  [JMH], [JVM], [Forking, warmup control, blackholes, `perfasm` profiler],
  [criterion], [Rust], [Bootstrap CIs, outlier classification, regression detection],
  [Google Benchmark], [C++], [Auto iteration count, `DoNotOptimize`, counters],
  [hyperfine], [CLI commands], [Warmup runs, shell-noise calibration, outlier warning],
  [pytest-benchmark], [Python], [Calibration, stats, regression comparison],
  [wrk2 / vegeta], [HTTP load], [Constant-throughput open-loop, HDR histograms],
)

`hyperfine --warmup 3 'cmd-a' 'cmd-b'` is the minimum bar for comparing command-line programs; it runs each enough times to estimate variance and flags statistical outliers. JMH's `-prof perfasm` mode annotates the generated assembly with hardware-counter samples, which is the only reliable way to confirm a JVM microbenchmark measures what you think it does.

== Statistical Rigor

=== Variance comes first

A single run is an anecdote. Run $n >= 10$ trials, report the median and a dispersion measure (IQR or MAD are more robust than standard deviation for skewed timing data). Timing distributions are typically right-skewed: the minimum is the least-noisy estimate of "the code's cost", while the upper quantiles measure the system.

=== Confidence intervals

For a difference between two systems, the question is whether the *confidence interval of the difference* excludes zero, not whether the means differ. With unknown, possibly unequal variances, Welch's $t$-test applies; for non-normal timing data, the Mann-Whitney U test or *bootstrap* resampling (used internally by criterion) is safer. A practical report: "B is faster than A by $7.2% plus.minus 1.8%$ (95% CI, $n = 30$)".

=== Multiple comparisons

Testing 20 configurations at $alpha = 0.05$ yields an expected one false positive. Correct with Bonferroni ($alpha \/ m$) or, less conservatively, Benjamini-Hochberg false-discovery-rate control. The same trap appears when re-running a flaky benchmark "until it shows the improvement".

=== Effect size

Statistical significance is not engineering significance. A 0.3% speedup can be significant with enough samples and still be smaller than the run-to-run effect of code layout. Define a minimum effect of interest (say 2%) up front, and treat anything below the layout-noise floor as unproven.

== Benchmarking Crimes

Gernot Heiser's *systems benchmarking crimes* catalogue, the standard checklist for reviewing papers and blog posts, includes:

- Selective benchmarking: reporting only the subset of workloads that flatter the system.
- Improper baselines: comparing against an untuned or outdated competitor.
- Arithmetic-mean of ratios: ratios must be aggregated with the geometric mean; the arithmetic mean of normalized results depends on the normalization baseline (Fleming & Wallace, 1986).
- No indication of variance: a bar chart without error bars.
- Benchmarking a simulated or scaled-down system and projecting to full scale.
- Throughput without latency (or the reverse): degraded operating modes hide in the unreported metric.

== A/B Experiments in Production

Synthetic benchmarks cannot reproduce production's workload mix, data distribution, or interference, so mature organizations measure performance changes in production:

- *Canarying*: route a small fraction (1-5%) of traffic to the new version; compare RED metrics against the control group. Watch percentiles, not means; a canary can improve p50 while regressing p99.9.
- *A/B with proper randomization*: assign by user or request hash, not by host or time, to avoid confounding with hardware and diurnal load. Pre-register the metrics and the decision rule.
- *Shadow traffic*: duplicate requests to the candidate and discard responses; measures latency and errors with zero user risk but doubles backend load and cannot exercise writes safely.
- *Interleaved trials*: alternate versions on the same hosts in short time slices to cancel host effects, a paired design with far higher statistical power than host-split designs.

Netflix's automated canary analysis (Kayenta) and statistical approaches like sequential testing (always-valid p-values) address the peeking problem: checking the dashboard repeatedly and stopping when it looks good inflates false positives exactly like multiple comparisons.

== Reproducibility

A benchmark result that cannot be reproduced is a rumor. Record, alongside the numbers: exact source revision and compiler version with flags; hardware model, core count, memory, and microcode version; OS and kernel version; frequency-scaling governor and SMT state; dataset and seed; and the full benchmark harness configuration. Containers pin the userland but not the kernel or hardware; bare-metal CI runners with pinned configuration (as used by Rust's `perf.rust-lang.org` and the Linux kernel's automated regression testing) are the gold standard. Expect cloud VMs to show 5-15% run-to-run variance from noisy neighbors and varying host hardware; either control for it with paired designs or use enough samples to average it out.

== Further Reading

- Heiser, G. _Systems Benchmarking Crimes_. https://gernot-heiser.org/benchmarking-crimes.html
- Mytkowicz, T. et al. (2009). Producing wrong data without doing anything obviously wrong! _ASPLOS_.
- Curtsinger, C., & Berger, E. (2013). STABILIZER: statistically sound performance evaluation. _ASPLOS_.
- Fleming, P., & Wallace, J. (1986). How not to lie with statistics: the correct way to summarize benchmark results. _CACM_, 29(3).
- Georges, A. et al. (2007). Statistically rigorous Java performance evaluation. _OOPSLA_.

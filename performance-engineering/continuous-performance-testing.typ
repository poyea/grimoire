#import "../template.typ": xref

= Continuous Performance Testing <continuous-performance-testing>

Performance rarely collapses in one commit. It erodes: a 0.5 percent slowdown here, an extra allocation there, a dependency upgrade that changes a default. Each change passes review because its effect is invisible, and a year later the service needs a third more machines. Catching regressions as they happen requires running benchmarks on every change, which runs into two problems: performance measurements are noisy, often by more than the regressions being hunted, and a suite of hundreds of benchmarks produces false alarms by chance. This chapter covers sources of noise and how to reduce them, choosing metrics, comparing two builds with sound statistics including a C++ bootstrap, controlling false positives across many benchmarks, detecting change points in performance history with a C++ implementation, bisecting to the culprit, and gating and production-side checks.

*See also:* #xref("performance-engineering", "benchmarking", label: "Benchmarking"), #xref("performance-engineering", "methodology", label: "Performance Methodology"), #xref("performance-engineering", "compiler-aware-optimization", label: "Writing Code for the Optimiser") (layout bias), #xref("observability-and-sre", "continuous-profiling", label: "Continuous Profiling").

== Why Regressions Need Machines to Catch Them

A regression of 2 percent in a request handler is invisible to its author and to reviewers, below the noise of any single run, and often below day-to-day variation in production metrics. Yet twenty such regressions compound to 50 percent. The only defence is systematic: run a fixed benchmark suite on every change (or every merge to the main branch), keep the history, and have software decide when a change is real.

Two error types frame everything that follows. A *false positive* blames a commit for noise, wastes an engineer's time, and, if frequent, teaches the team to ignore the system. A *false negative* lets a regression through. The noise level of the measurements determines the smallest regression that can be detected at an acceptable false-positive rate.

== Sources of Noise

#table(
  columns: 2,
  [*Source*], [*Mitigation*],
  [Shared or virtualised CI runners, noisy neighbours], [Dedicated bare-metal machines for benchmarks; at least the same instance type every time],
  [CPU frequency scaling, turbo, thermal throttling], [Fix the frequency; disable turbo; let machines cool between runs],
  [Other processes, interrupts], [Isolate cores, pin benchmark threads, stop unrelated services],
  [Code and data layout (ASLR, link order)], [Several builds or layout randomisation; distrust changes of 1 to 3 percent with no counter-level explanation],
  [JIT and GC nondeterminism], [Multiple process forks; warm-up; record GC activity],
  [Caches, file system state, network], [Warm or flush consistently; local inputs only],
  [Time-of-day effects on shared infrastructure], [Interleave baseline and candidate runs rather than running one after the other],
)

Cloud machines are particularly noisy. Studies of benchmarking on public clouds found variation of several percent between instances of the same type and occasionally far more, enough to hide small regressions completely. The most effective single technique on noisy hardware is *interleaving*: run the baseline and candidate builds alternately on the same machine in the same time window (A, B, A, B, ...), so that slow periods affect both equally, and compare them pairwise.

== Choosing the Metric

Wall-clock time is what users feel but the noisiest thing to measure. Alternatives with less noise:

- *Instructions retired* (from `perf stat` or hardware counters) varies by well under 1 percent between runs, so it detects tiny changes. The Rust compiler's performance tracking uses it as its primary metric for this reason. It misses regressions caused by cache misses, branch mispredictions or contention, which change time but not instruction count, so pair it with a time-based check.
- *Simulated cycles* from Cachegrind are fully deterministic and include a cache model. SQLite tracks performance this way.
- *Allocation counts and bytes* catch a frequent cause of slowdowns in managed languages deterministically.
- *Resource budgets* (binary size, JavaScript bundle size, memory high-water mark, number of database queries per request) are deterministic and worth gating on directly.

Use deterministic proxies to detect changes and time to confirm that they matter.

== Comparing Two Builds

Given samples of a metric from the baseline and candidate builds, the questions are whether they differ and by how much. Performance samples are rarely normal: they are skewed, often multi-modal (a run either hit a slow path or did not), and contain outliers. Methods that respect this:

- Report the *ratio of medians* (or of trimmed means) with a confidence interval, which answers "how much slower" directly.
- Compute the interval by *bootstrap*: resample each set with replacement many times, compute the ratio for each resample, and take percentiles of the resulting distribution.
- Test for a difference with a rank-based test such as *Mann-Whitney U*, which needs no normality assumption.
- Decide with both statistical and practical significance: flag a change only if the confidence interval excludes no-change *and* the effect exceeds a threshold that matters (say 2 percent).

```cpp
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

double median(std::vector<double> v) {
    const std::size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    if (v.size() % 2 == 1) return v[mid];
    const double upper = v[mid];
    return 0.5 * (upper + *std::max_element(v.begin(), v.begin() + mid));
}

struct interval {
    double low, point, high;
};

// Percentile bootstrap confidence interval for median(candidate) / median(baseline).
interval median_ratio_ci(const std::vector<double>& baseline,
                         const std::vector<double>& candidate,
                         double confidence = 0.95, int resamples = 10'000,
                         std::uint64_t seed = 1) {
    std::mt19937_64 rng(seed);
    auto resample = [&](const std::vector<double>& v) {
        std::uniform_int_distribution<std::size_t> pick(0, v.size() - 1);
        std::vector<double> r(v.size());
        for (auto& x : r) x = v[pick(rng)];
        return r;
    };
    std::vector<double> ratios(resamples);
    for (auto& r : ratios) r = median(resample(candidate)) / median(resample(baseline));
    std::sort(ratios.begin(), ratios.end());
    const double tail = (1.0 - confidence) / 2.0;
    const auto at = [&](double q) {
        return ratios[static_cast<std::size_t>(q * (ratios.size() - 1))];
    };
    return {at(tail), median(candidate) / median(baseline), at(1.0 - tail)};
}
```

A result of $[1.012, 1.021, 1.031]$ says the candidate is about 2 percent slower, and the interval excludes 1. With interleaved runs, pairing each baseline run with its neighbouring candidate run and bootstrapping the paired ratios gives a tighter interval, because shared slow periods cancel.

How many runs? The width of the interval shrinks with the square root of the sample count, so detecting a regression half as large needs four times the runs. Kalibera and Jones showed that, for systems with variation at several levels (between builds, between process launches, between iterations), the optimal allocation of runs across levels depends on where the variance is, and measuring that once saves a great deal of machine time.

== Many Benchmarks, Many Alarms

A suite of 500 benchmarks tested at a 5 percent significance level produces about 25 false alarms per comparison even when nothing changed. Approaches:

- Control the *false discovery rate* with the Benjamini-Hochberg procedure: sort the $m$ p-values, find the largest $k$ with $p_((k)) <= k alpha \/ m$, and flag the first $k$. The expected fraction of false alarms among the flagged results is then at most $alpha$.
- Require practical significance (a minimum effect) in addition to statistical significance.
- Require confirmation: rerun flagged benchmarks before alerting, or require the change to persist over several subsequent commits.
- Track each benchmark's historical noise and set its threshold relative to that noise rather than one global threshold.

== Change-Point Detection in History

Testing each commit against its parent misses gradual drifts and is sensitive to one noisy run. Looking at the *history* of a benchmark as a time series asks a better question: where does the distribution of results shift? *Change-point detection* finds those shifts. MongoDB replaced threshold-based alerting with the E-Divisive Means algorithm on their performance CI and reported far fewer false alerts and better detection of real regressions.

Binary segmentation is the simplest effective method: find the split of the series that best separates it into two segments with different means, accept it if the shift is large relative to the noise, and recurse on both halves.

```cpp
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// Indices where the mean of `series` shifts, by binary segmentation.
// A split is accepted if the mean shift exceeds `min_shift` (relative to the left mean)
// and its t-like statistic exceeds `min_score`.
void find_change_points(const std::vector<double>& series, std::size_t begin, std::size_t end,
                        double min_shift, double min_score, std::size_t min_segment,
                        std::vector<std::size_t>& out) {
    if (end - begin < 2 * min_segment) return;

    std::vector<double> prefix(end - begin + 1, 0.0), prefix_sq(end - begin + 1, 0.0);
    for (std::size_t i = begin; i < end; ++i) {
        prefix[i - begin + 1] = prefix[i - begin] + series[i];
        prefix_sq[i - begin + 1] = prefix_sq[i - begin] + series[i] * series[i];
    }
    const auto n = static_cast<double>(end - begin);
    const double total = prefix.back(), total_sq = prefix_sq.back();

    double best_score = 0.0;
    std::size_t best = 0;
    for (std::size_t k = min_segment; k + min_segment <= end - begin; ++k) {
        const auto nl = static_cast<double>(k), nr = n - nl;
        const double ml = prefix[k] / nl, mr = (total - prefix[k]) / nr;
        const double ssl = prefix_sq[k] - nl * ml * ml;
        const double ssr = (total_sq - prefix_sq[k]) - nr * mr * mr;
        const double pooled_sd = std::sqrt(std::max(ssl + ssr, 1e-300) / (n - 2.0));
        const double score = std::abs(mr - ml) / (pooled_sd * std::sqrt(1.0 / nl + 1.0 / nr));
        if (score > best_score && std::abs(mr - ml) >= min_shift * std::abs(ml)) {
            best_score = score;
            best = begin + k;
        }
    }
    if (best_score < min_score) return;
    find_change_points(series, begin, best, min_shift, min_score, min_segment, out);
    out.push_back(best);
    find_change_points(series, best, end, min_shift, min_score, min_segment, out);
}
```

On a series of 200 noisy measurements with a 3 percent step at position 120 and 1 percent noise, a call with `min_shift = 0.01`, `min_score = 6` and `min_segment = 10` finds the step within a commit or two (in a test over 50 random series, at position 120 in 44 and within two positions in the rest), while ignoring individual outliers that a threshold on each commit would report. E-Divisive uses a distribution-free energy statistic instead of a mean shift and a permutation test for significance, which makes it robust to non-normal, multi-modal data at a higher computational cost.

== Finding the Culprit

A change point narrows a regression to a range of commits, especially when benchmarks run only on every few merges. *Bisection* finds the exact commit: `git bisect run` with a script that builds, runs the benchmark several times, and exits with status 0 or 1 based on a threshold, finds the culprit in $log_2$ of the range length steps. The script must be robust to noise (use enough repetitions and a threshold between the before and after levels), or bisection will follow noise down the wrong half. Commits that do not build are skipped with exit code 125.

== Gating and Alerting

Where to put the check is a policy choice:

- *Blocking gates* on pull requests work for deterministic metrics (binary size, instruction counts with a tolerance, allocation counts, query counts) because they do not flake.
- *Alerting* on main-branch history, with change-point detection and an owner rotation that triages each alert, suits noisy time-based metrics. An alert that is closed as "noise" should feed back into that benchmark's threshold.
- *Release gates* run long, realistic load tests on candidate builds and compare them with the previous release on identical hardware.

== Production as the Final Benchmark

Benchmarks model production imperfectly, so the last line of defence is production itself. *Canary analysis* deploys a new version to a small fraction of servers alongside an equal number of baseline servers receiving the same traffic, and compares their latency, error and resource metrics statistically before continuing the rollout; Netflix and Google's Kayenta automates this. *Continuous profiling* in production makes regressions explainable: diffing the fleet-wide CPU profile before and after a release shows which functions got more expensive.

== Pitfalls

- Benchmarking on shared CI runners and treating each result as exact.
- Comparing single runs, or means of heavily skewed samples.
- A suite whose false alarms outnumber its true detections, which trains people to ignore it.
- Blocking merges on noisy time-based benchmarks, leading to retries until they pass.
- Benchmarks that no longer reflect production workloads.
- No owner for alerts.

== Further Reading

Kalibera, T. and Jones, R. (2013). "Rigorous Benchmarking in Reasonable Time." ISMM.

Leitner, P. and Cito, J. (2016). "Patterns in the Chaos: A Study of Performance Variation and Predictability in Public IaaS Clouds." ACM Transactions on Internet Technology.

Laaber, C., Scheuner, J. and Leitner, P. (2019). "Software Microbenchmarking in the Cloud. How Bad Is It Really?" Empirical Software Engineering.

Daly, D. et al. (2020). "The Use of Change Point Detection to Identify Software Performance Regressions in a Continuous Integration System." ICPE.

Matteson, D. S. and James, N. A. (2014). "A Nonparametric Approach for Multiple Change Point Analysis of Multivariate Data." Journal of the American Statistical Association.

Benjamini, Y. and Hochberg, Y. (1995). "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." Journal of the Royal Statistical Society, Series B.

Efron, B. and Tibshirani, R. J. (1993). _An Introduction to the Bootstrap_. Chapman and Hall.

Mann, H. B. and Whitney, D. R. (1947). "On a Test of Whether One of Two Random Variables Is Stochastically Larger than the Other." Annals of Mathematical Statistics.

Mytkowicz, T. et al. (2009). "Producing Wrong Data Without Doing Anything Obviously Wrong!" ASPLOS.

#import "../template.typ": xref

= Performance Methodology

Performance work without methodology degenerates into guessing: tweaking flags, restarting services, and blaming the network. This chapter covers the structured methods that turn performance analysis into engineering: the USE and RED checklists, workload characterization, the scaling laws (Amdahl, Gustafson, and the Universal Scalability Law), and the statistical discipline needed to talk about latency honestly.

*See also:* #xref("performance-engineering", "benchmarking", label: "Benchmarking") (measuring correctly), #xref("performance-engineering", "queueing-theory", label: "Queueing Theory") (the mathematics behind utilization-latency curves), #xref("performance-engineering", "capacity-planning", label: "Capacity Planning") (applying these models forward in time), and the CPU Architecture volume's #xref("cpu-architecture", "pipelining", label: "Pipelining") chapter (where hardware-level bottlenecks originate).

== The USE Method

Brendan Gregg's *USE method* is a checklist for resource-oriented analysis. For every resource (CPU, memory, disks, NICs, locks, file descriptors), check three things:

- *Utilization*: the fraction of time the resource was busy, or the fraction of capacity used.
- *Saturation*: the degree of queued, waiting work (run-queue length, swap activity, NIC drops).
- *Errors*: error event counts, which are often silently retried and hidden.

The power of USE is completeness: it enumerates resources first, then metrics, so problems are found by elimination rather than by hunch. On Linux, utilization comes from `mpstat`, `iostat -x` (the `%util` column), and `sar`; saturation from run-queue length (`vmstat`'s `r` column), pressure stall information (PSI, `/proc/pressure/`), and disk queue depth (`aqu-sz`).

A subtlety: 100% utilization of a disk reported by `iostat` does not mean the device cannot accept more work, because modern SSDs and RAID arrays service requests in parallel. Saturation (queue length and wait time) is the more reliable signal.

== The RED Method

Where USE is resource-oriented, Tom Wilkie's *RED method* is request-oriented, designed for microservices:

- *Rate*: requests per second.
- *Errors*: failed requests per second.
- *Duration*: distribution of request latency (not just the mean).

RED maps directly onto service-level objectives and is the standard dashboard layout in Prometheus and Grafana deployments. The two methods are complementary: RED tells you a service is slow; USE tells you which resource is making it slow.

== Workload Characterization

Before optimizing, characterize what the system is actually being asked to do. The standard questions: who is sending the load (clients, services, batch jobs), why (code paths, triggers), what (request types, read/write mix, object sizes), and how it changes over time (diurnal cycles, bursts). Many "performance problems" dissolve at this stage: a retry storm, an unintended full-table scan, a misconfigured client polling at 100 Hz.

Useful characterization dimensions:

#table(
  columns: 3,
  [*Dimension*], [*Example metrics*], [*Tools*],
  [Request mix], [GET/PUT ratio, query types], [Access logs, `pg_stat_statements`],
  [Object sizes], [p50/p99 payload bytes], [Histograms in middleware],
  [Concurrency], [Active connections, in-flight requests], [Connection pool metrics],
  [Temporal pattern], [Diurnal peak-to-trough ratio], [Time-series dashboards],
  [Locality], [Cache hit ratio, working set size], [`perf`, cache stats],
)

== Scaling Laws

=== Amdahl's Law

If a fraction $p$ of a program's work is parallelizable and the rest is serial, the speedup on $n$ processors is bounded:

$ S(n) = 1 / ((1 - p) + p / n) $

As $n -> infinity$, $S -> 1 \/ (1 - p)$. With $p = 0.95$, the ceiling is $20 times$ regardless of core count. Amdahl's law is the formal statement of "optimize the bottleneck": reducing the serial fraction matters more than adding parallelism.

=== Gustafson's Law

Gustafson (1988) reframed the question: in practice we grow the problem with the machine. If the serial fraction stays fixed in absolute time while parallel work scales with $n$, the *scaled speedup* is

$ S(n) = (1 - p) + p n $

which is linear in $n$. Amdahl answers "how much faster is a fixed workload"; Gustafson answers "how much more work can we do in fixed time". Both are correct for their respective questions.

=== The Universal Scalability Law

Neil Gunther's *Universal Scalability Law (USL)* extends Amdahl with a second penalty term for coherency, the cost of keeping shared state consistent (cache-line ping-pong, lock convoys, consensus rounds):

$ C(n) = n / (1 + sigma (n - 1) + kappa n (n - 1)) $

where $C(n)$ is throughput relative to one node, $sigma$ is the *contention* coefficient (serialization, Amdahl-like), and $kappa$ is the *coherency* coefficient (crosstalk). The crucial property: when $kappa > 0$, throughput has a maximum at $n^* = sqrt((1 - sigma) / kappa)$ and then _decreases_, retrograde scaling. Real systems exhibit this: adding nodes past the peak makes them slower. Fitting $sigma$ and $kappa$ to a handful of load-test points (nonlinear regression, as in the `usl` R package) yields a predictive capacity model from sparse data.

== Latency, Throughput, Utilization

The three core quantities are related but not interchangeable:

- *Latency* is per-request time; users feel it directly.
- *Throughput* is requests completed per second; capacity and cost are denominated in it.
- *Utilization* is the busy fraction of a resource.

They trade off: batching raises throughput at the cost of latency; running at high utilization raises both throughput and latency (sharply, near saturation, as the queueing chapter quantifies). A system tuned for maximum throughput (deep queues, large batches) is usually a poor latency system, and vice versa. State which one you are optimizing before starting.

== Percentiles and Coordinated Omission

=== Why percentiles

Latency distributions are heavy-tailed; means are dominated by the tail and hide it at the same time. Report percentiles: p50 (median), p95, p99, p99.9, and max. Note that percentiles do not compose: the p99 of a service is not the average of per-host p99s, and the p99 of a fan-out request is governed by the per-leg p99 raised to the number of legs (see #xref("performance-engineering", "concurrency-performance", label: "Concurrency Performance")). Aggregate with histograms (HDR histograms, Prometheus native histograms), never by averaging precomputed percentiles.

=== Coordinated omission

Gil Tene's term for a pervasive measurement bug. A closed-loop load generator that issues a request, waits for the response, then issues the next one *coordinates* with the system under test: when the system stalls for 2 seconds, the generator silently stops sampling during exactly the period when latency was worst. The recorded distribution omits the bad samples.

The fix: measure from the *intended* send time, not the actual send time. If the schedule says a request should have been sent at $t$ and it completed at $t'$, the latency is $t' - t$, including the time it spent waiting to be sent. Tools that correct for this: `wrk2` (constant-throughput mode), HdrHistogram's recorder, and YCSB in its coordinated-omission-aware mode. Uncorrected benchmarks routinely understate p99.9 by one to two orders of magnitude.

== A Working Process

A defensible performance engagement looks like:

1. *Define the goal*: a target percentile latency at a target throughput, or a cost target.
2. *Characterize the workload* before touching anything.
3. *Measure*: USE across resources, RED across services; establish a baseline with variance.
4. *Form a hypothesis* about the dominant bottleneck; predict the effect of the fix quantitatively.
5. *Change one thing*, re-measure, compare against the baseline with statistics (next chapter).
6. *Stop* when the goal is met. Optimization past the requirement is spent engineering budget.

The anti-patterns are familiar: the *streetlight method* (looking where the tools are convenient), *random-change tuning*, and *blame-someone-else*. Methodology is what distinguishes performance engineering from performance folklore.

== Further Reading

- Gregg, B. (2020). _Systems Performance: Enterprise and the Cloud_, 2nd ed. Addison-Wesley.
- Gunther, N. (2007). _Guerrilla Capacity Planning_. Springer.
- Amdahl, G. (1967). Validity of the single processor approach to achieving large scale computing capabilities. _AFIPS_.
- Gustafson, J. (1988). Reevaluating Amdahl's law. _CACM_, 31(5).
- Tene, G. (2013). How NOT to measure latency. _Strange Loop_ talk.

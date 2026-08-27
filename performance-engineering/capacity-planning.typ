#import "../template.typ": xref

= Capacity Planning <capacity-planning>

Capacity planning answers a forward-looking question: how much resource will the workload need, when, and at what cost. Done well, it is measurement plus a model plus a margin; done badly, it is last year's spend times a guess. This chapter covers demand forecasting, headroom policy, load testing for capacity (not just performance), scalability modeling with the USL, autoscaling, overload protection, and cost-aware sizing.

*See also:* #xref("performance-engineering", "queueing-theory", label: "Queueing Theory") (why headroom is non-negotiable), #xref("performance-engineering", "methodology", label: "Performance Methodology") (the USL, workload characterization), #xref("performance-engineering", "benchmarking", label: "Benchmarking") (trustworthy load tests), and #xref("performance-engineering", "concurrency-performance", label: "Concurrency Performance") (the scaling limits the plan must respect).

== The Core Loop

Capacity planning is a control loop, not an annual document:

1. *Measure demand* in workload units (requests/s, messages/s, GB ingested/day), not resource units. Resource usage is demand times cost-per-unit; conflating them hides efficiency regressions and efficiency wins alike.
2. *Measure capacity* empirically: the throughput at which the service still meets its latency SLO, per replica, on production-like hardware. This number, not CPU count, is what you provision against.
3. *Forecast demand* over the lead time of adding capacity (minutes for autoscaled containers, weeks for reserved instances, quarters for datacenters and GPUs).
4. *Provision* forecasted demand plus headroom, and *validate* with load tests and the next cycle's measurements.

The key derived metric is *service demand*: resource-seconds consumed per workload unit (CPU-ms per request, IOPS per transaction, bytes of RAM per session). It links business growth to infrastructure: if the product forecast says $2 times$ requests and the demand per request is flat, the plan is $2 times$ capacity. Track demand-per-unit over releases; a 20% regression is invisible in absolute dashboards while traffic is shrinking and devastating when it grows back.

== Forecasting Demand

- *Decompose the series*: trend, seasonality (diurnal, weekly, annual), and events. Peak-to-trough ratios of 2-5 times are common in consumer traffic; you provision for the peak, so forecasting the *peak* (e.g., p99 of daily maxima) matters more than the mean.
- *Methods*: linear or log-linear regression on the trend is often sufficient at quarterly horizons; Holt-Winters and seasonal ARIMA capture seasonality; Prophet-style decomposition handles holidays. Sophistication beyond this rarely beats getting the *driver* right: forecast in business units (users, orders) where product teams have plans, then convert via measured demand-per-unit.
- *Plan for the planned*: launches, marketing pushes, sales events (Black Friday is not an anomaly, it is a calendar entry), and TV moments dominate organic growth at short horizons. The planning interface to product/marketing is more valuable than any model.
- *State the uncertainty*: a forecast is a distribution; provision to a chosen quantile of it and write the choice down.

== Headroom and Utilization Targets

Queueing theory dictates that latency-sensitive services cannot run hot: at 90% utilization, response time is about $10 times$ the service time for M/M/1-like behavior, and bursts at sub-minute timescales sit far above the dashboard average. Practical targets:

#table(
  columns: 3,
  [*Workload*], [*Typical peak utilization target*], [*Why*],
  [Latency-sensitive serving], [40-60% CPU at peak], [Queueing knee, burst absorption],
  [Throughput batch], [80-95%], [Latency irrelevant; cost dominates],
  [Storage capacity (bytes)], [70-80%], [Rebuild/rebalance space, growth lead time],
  [Network links], [50-70% at peak], [Failover doubles load on survivors],
)

Headroom must also cover *failure*: an N+1 (or N+2) policy means surviving the loss of a replica, an availability zone, or a region at peak while still meeting SLOs. A service spread across 3 zones that must survive one zone's loss can run each zone at most at $2 \/ 3$ of its capacity, before any queueing margin. Headroom for failure, bursts, and growth lead time stack; this is why well-run fleets "look underutilized."

== Finding Capacity: Load Testing and the USL

Per-replica capacity is measured, not computed. The procedure: drive open-loop load (see #xref("performance-engineering", "queueing-theory", label: "Queueing Theory")) at increasing rates against a production-like replica with production-like data and request mix, recording latency percentiles at each plateau. Capacity is the highest rate at which the SLO percentile holds, not the rate at which throughput peaks; the two can differ by 2 times, and the region between them is where the service is technically "up" and practically unusable.

For horizontal scaling, fit the *Universal Scalability Law* (see #xref("performance-engineering", "methodology", label: "Performance Methodology")) to throughput measured at several replica/core counts:

$ C(n) = n / (1 + sigma (n - 1) + kappa n (n - 1)) $

A handful of points (4-6 well-spaced load tests) yields $sigma$ and $kappa$ by nonlinear regression, and with them: the predicted throughput at fleet sizes you have not tested, the peak at $n^* = sqrt((1 - sigma) / kappa)$, and an early warning that the architecture has a scaling ceiling below next year's forecast. A fitted $kappa$ significantly above zero is a finding: some shared component (a coordination service, a hot row, a global lock) will need redesign, and the fit says when. Validate the model against one held-out measurement; an USL fit that misses badly usually means the bottleneck *shifted* between test points (CPU-bound at small $n$, then a shared database), which is itself the important discovery.

Capacity numbers decay: re-measure after major releases, hardware generation changes, and data growth (a database's capacity is a function of its working set vs. RAM).

== Autoscaling

Autoscaling converts the provisioning lead time from weeks to minutes, but it is a feedback controller and inherits control problems:

- *Signal choice*: scale on a leading, demand-proportional signal (requests per replica, queue depth, concurrency via Little's law) rather than a lagging one (CPU works for CPU-bound services; latency is too noisy and too late). Target-tracking on "requests per replica = measured capacity times target utilization" is the cleanest formulation.
- *Lag and instability*: instance boot plus warmup (JIT, cache fill, connection establishment) can take minutes; during a sharp spike the fleet is effectively static, so *burst headroom must exist before the spike*. Short cooldowns plus aggressive scale-down produce oscillation; asymmetric policy (fast up, slow down) is standard.
- *Floors and limits*: a minimum replica count covers cold-start and correlated failure; quota and budget caps prevent a retry storm or a bug from autoscaling into a bill.
- *Cluster vs. application scaling*: pod autoscaling (HPA) is bounded by node provisioning; capacity planning at the cluster/quota level is still required underneath the "serverless" abstraction.

Autoscaling does not remove capacity planning; it changes the planned quantity from "replicas" to "burst headroom, scaling limits, and quota."

== Overload Protection

The plan will be wrong someday, so overload behavior is part of capacity design. An open system pushed past capacity does not plateau, it collapses: queues grow, latency exceeds client timeouts, clients retry, effective load multiplies, and goodput falls toward zero. This *metastable failure* state can persist after the trigger is gone because retry load alone exceeds capacity. Defenses, in order of preference:

- *Admission control / load shedding*: reject excess early and cheaply (a 429 costs microseconds; a timed-out request costs a full service time plus a retry). Shed by criticality class, keep goodput at capacity.
- *Bounded queues and deadlines*: queue lengths sized so queueing delay cannot exceed the request deadline; propagate deadlines and cancel work whose caller has given up.
- *Client behavior*: exponential backoff with jitter; retry budgets (e.g., retries at most 10% of requests) so retries cannot multiply load; circuit breakers.
- *Graceful degradation*: cheaper fallback results (stale cache, smaller candidate sets) under pressure.

Test overload deliberately: drive 1.5-2 times capacity and verify goodput stays near capacity rather than collapsing. This single test catches more real outages than most latency benchmarks.

== Cost-Aware Planning

Capacity is denominated in money. The levers, roughly in order of leverage:

- *Efficiency before quantity*: a 30% reduction in CPU per request (profiling, see #xref("performance-engineering", "cpu-profiling", label: "CPU Profiling")) is a 30% fleet reduction at every future scale, compounding with growth.
- *Purchase mix*: reserved/committed capacity for the base load (1-3 year commitments at 30-60% discounts), on-demand for the forecast band, spot/preemptible for interruption-tolerant batch (60-90% discounts, with eviction handling).
- *Shape fitting*: match instance shape to the binding resource (memory-bound services on memory-optimized shapes); the non-binding resources are pure waste. Bin-packing efficiency (requested vs. used, used vs. allocatable) is the cluster-level version.
- *Time shifting*: batch and training workloads moved to the diurnal trough consume capacity already paid for at peak.
- *Unit economics*: report cost per workload unit (dollars per million requests) alongside the latency SLO; it is the metric that makes efficiency work legible to the business and detects regressions the same way demand-per-unit does.

== Pitfalls

- *Provisioning to averages*: the peak, the burst within the peak, and the failure-mode load are the constraints; the average is an accounting fiction.
- *Resource metrics without workload metrics*: "CPU is at 70%" cannot distinguish growth from regression, or say what happens at 2 times traffic.
- *Linear extrapolation of nonlinear systems*: doubling replicas does not double capacity when $kappa > 0$, and doubling a database's data does not leave its per-query cost flat once the working set spills RAM.
- *Testing scale-up but not overload or failure*: the plan must survive a zone loss at peak and 1.5 times capacity with retries; neither appears in a standard load test.
- *Stale capacity numbers*: every release changes demand-per-unit; a quarterly capacity number with weekly deploys is a guess wearing a spreadsheet.
- *Headroom shame*: leadership reads 50% utilization as waste; the queueing math says it is the latency SLO's purchase price. Write the policy down so the argument happens once.

== Worked Example

A service is load-tested (open-loop, SLO-bounded, per the procedure above) at three fleet sizes:

#table(
  columns: 3,
  [*Replicas $n$*], [*Throughput (req/s)*], [*Speedup $C(n)$*],
  [1], [2,000], [1.000],
  [8], [11,380], [5.690],
  [32], [18,069], [9.034],
)

Fit the USL. Rearranging $C(n) = n \/ (1 + sigma (n - 1) + kappa n (n - 1))$ gives one linear equation per measurement in $sigma$ and $kappa$:

$ n = 8: quad 1 + 7 sigma + 56 kappa = 8 / 5.690 = 1.406 $
$ n = 32: quad 1 + 31 sigma + 992 kappa = 32 / 9.034 = 3.542 $

Subtracting, $24 sigma + 936 kappa = 2.136$, i.e. $sigma + 39 kappa = 0.0890$; the first equation gives $sigma + 8 kappa = 0.0580$. Subtracting again: $31 kappa = 0.0310$, so

$ kappa = 0.0010, quad sigma = 0.050 $

(With more than three points, nonlinear least squares replaces this elimination, but the arithmetic is the same idea.) Now forecast. The fitted peak is at

$ n^* = sqrt((1 - sigma) / kappa) = sqrt(0.95 / 0.001) approx 31 $

and the predicted ceiling is $C(31) = 31 \/ (1 + 0.05 times 30 + 0.001 times 930) = 31 \/ 3.43 = 9.04$, i.e. about $9.04 times 2000 approx 18,100$ req/s, after which adding replicas reduces throughput. The demand forecast says next year's peak is 25,000 req/s. The plan cannot be "buy more replicas": the architecture saturates at 18,100 regardless of fleet size, and at a 60% peak-utilization headroom target the usable ceiling is only about 10,900 req/s, which this quarter's peak of 9,500 req/s already crowds.

The fit has converted a load test into a finding with a deadline: the $kappa = 0.001$ coherence term (some shared component, a hot row, a coordination service) must be engineered away before demand crosses the ceiling. And the fit quantifies the redesign target: halving $kappa$ to 0.0005 moves $n^*$ to $sqrt(0.95 \/ 0.0005) approx 44$ but the ceiling only to $C(44) = 44 \/ (1 + 0.05 times 43 + 0.0005 times 44 times 43) = 10.7$, about 21,500 req/s, still short of the forecast. Reaching 25,000 req/s plus headroom requires cutting the serial fraction $sigma$ as well, and the next round of load tests must validate whichever fix ships against a held-out measurement.

== Further Reading

- Gunther, N. (2007). _Guerrilla Capacity Planning_. Springer.
- Allspaw, J. (2008). _The Art of Capacity Planning_. O'Reilly (2nd ed. 2017 with Arun Kejariwal).
- Beyer, B., Jones, C., Petoff, J., & Murphy, N. R. (2016). _Site Reliability Engineering_, chs. on load balancing, overload, and capacity. O'Reilly.
- Bronson, N., Aghayev, A., Charapko, A., & Zhu, T. (2021). Metastable failures in distributed systems. _HotOS_.
- Barroso, L., Hölzle, U., & Ranganathan, P. (2018). _The Datacenter as a Computer_, 3rd ed. Morgan & Claypool.

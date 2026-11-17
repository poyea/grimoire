#import "../template.typ": xref

= Autoscaling and Capacity Management <autoscaling>

Elastic capacity is the defining economic promise of the cloud: pay for what the load needs now, not for the peak forecast a year ago. Autoscaling delivers that promise only partly. It is a feedback controller with measurement lag, actuation delay, and a plant (the application) whose response to more capacity is nonlinear and sometimes negative. Capacity also runs out: quotas, zonal stock-outs, and dependencies that do not scale set hard ceilings. This chapter treats autoscaling as control and queueing, covers the scaling layers in VM groups and Kubernetes, and then turns to the capacity planning that autoscaling cannot replace.

*See also:* #xref("cloud-and-infrastructure", "kubernetes-internals", label: "Kubernetes Internals"), #xref("cloud-and-infrastructure", "serverless", label: "Serverless") (scale to zero and cold starts), #xref("cloud-and-infrastructure", "cost-engineering", label: "Cloud Cost Engineering") (spot and commitments), #xref("observability-and-sre", "slo-engineering", label: "SLO Engineering"), #xref("software-architecture", "resilience-patterns", label: "Resilience Patterns").

== Autoscaling as a Control Loop

Every autoscaler runs the same loop:

```text
  ┌──────────── measure ◄──── metrics pipeline (10-60 s lag)
  │                                   ▲
  ▼                                   │
decide (target, policy) ──► actuate ──► provision (30 s - 5 min) ──► warm up ──► serve
```

The total delay from a load change to useful new capacity is the sum of metric collection and aggregation, the decision interval, provisioning (scheduling a pod, launching a VM, pulling an image), and warm-up (JIT compilation, cache filling, connection pools, readiness checks). For a VM-based service this is commonly two to five minutes. Any load that rises faster than capacity can follow within that window must be absorbed by *headroom* already running. Autoscaling therefore decides how much headroom to keep, not whether to keep it.

Delay is also what makes control loops oscillate. A controller that reacts strongly to a metric that reflects the state of a few minutes ago overshoots: it adds capacity for a spike that has passed, sees utilisation fall, removes capacity, and repeats. The standard defences are asymmetric reaction (scale out fast, scale in slowly), a tolerance band where no action is taken, and stabilisation windows that act on the most conservative recent recommendation.

== Choosing the Signal

#table(
  columns: 3,
  [*Signal*], [*Good for*], [*Problems*],
  [CPU utilisation], [CPU-bound stateless services], [Misses I/O-bound and memory-bound work; GC and background work add noise],
  [Requests per target], [Web services with uniform request cost], [Breaks when request cost varies a lot],
  [Queue depth or consumer lag], [Workers draining a queue or stream], [Needs a known per-worker processing rate],
  [Concurrency (in-flight requests)], [Serverless and request-per-container platforms], [Needs a per-instance concurrency target],
  [Latency], [Detecting saturation], [Lagging and highly nonlinear; a poor primary control signal],
  [Custom business metric], [Workloads where none of the above tracks load], [Requires a reliable metrics pipeline],
)

The best signal is a *leading* measure of work arriving (requests, messages, in-flight work) rather than a *lagging* measure of strain (latency, error rate). Latency stays flat until utilisation nears saturation and then rises steeply, so by the time it moves, queues have already formed.

== Target Tracking

Target tracking keeps a metric near a set point by scaling proportionally. The Kubernetes Horizontal Pod Autoscaler uses:

$ "desired" = ceil("current" times "metric" / "target") $

If 10 pods run at 90 percent CPU against a 60 percent target, the HPA wants $ceil(10 times 90 \/ 60) = 15$ pods. It skips changes when the ratio is within a tolerance (10 percent by default), takes the maximum over several metrics, and applies a *stabilisation window* for scale-down (300 seconds by default): it uses the highest recommendation seen in the window, so a brief dip does not remove pods that will be needed again a minute later. Scale-up has no window by default but is rate-limited by policies (by default, at most doubling or adding four pods, whichever is more, every 15 seconds).

```cpp
#include <algorithm>
#include <cmath>
#include <deque>
#include <utility>

class target_tracker {
    double target_;
    double tolerance_;
    double down_window_s_;
    int min_replicas_, max_replicas_;
    std::deque<std::pair<double, int>> history_;   // (time, recommendation)

public:
    target_tracker(double target, int lo, int hi,
                   double tolerance = 0.1, double down_window_s = 300.0)
        : target_(target), tolerance_(tolerance), down_window_s_(down_window_s),
          min_replicas_(lo), max_replicas_(hi) {}

    int decide(double now, int current, double metric) {
        const double ratio = metric / target_;
        int raw = current;
        if (std::abs(ratio - 1.0) > tolerance_)
            raw = static_cast<int>(std::ceil(current * ratio));
        raw = std::clamp(raw, min_replicas_, max_replicas_);

        history_.emplace_back(now, raw);
        while (!history_.empty() && history_.front().first < now - down_window_s_)
            history_.pop_front();

        if (raw >= current) return raw;
        int stabilised = raw;
        for (const auto& [t, r] : history_) stabilised = std::max(stabilised, r);
        return std::min(stabilised, current);
    }
};
```

Two properties of the proportional rule matter in practice. First, it assumes the metric scales inversely with replica count. That holds for CPU in a well-balanced stateless service, not for a metric dominated by a shared bottleneck such as a database. Second, the target sets the headroom. A 60 percent CPU target leaves 40 percent of capacity to absorb a spike during the scaling delay; raising it to 80 percent saves cost and shrinks that buffer.

== Other Policies

- *Step scaling* maps alarm bands to fixed adjustments ("above 70 percent add 2, above 90 percent add 5"). It predates target tracking and is still useful for metrics that do not scale proportionally.
- *Scheduled scaling* sets minimums ahead of known events: business hours, batch windows, a product launch. It is the simplest way to beat provisioning delay.
- *Predictive scaling* forecasts load from recent history (AWS uses days to weeks of data and favours daily and weekly cycles) and provisions ahead of the forecast, with reactive scaling still handling surprises.
- *Queue-based scaling* sizes workers from backlog. KEDA in Kubernetes and step policies on queue metrics in VM groups both implement it.

For a queue, Little's law gives the target directly. If messages arrive at $lambda$ per second, each takes $S$ seconds to process, and each worker handles $k$ messages concurrently, steady state needs $lambda S \/ k$ workers. To also drain a backlog of $B$ messages within $T$ seconds, add $B S \/ (k T)$.

== Queueing and the Utilisation Knee

Why not run at 95 percent utilisation and scale only at the edge? Because waiting time grows without bound as utilisation approaches one. For $c$ identical servers with Poisson arrivals at rate $lambda$ and exponential service at rate $mu$ (the M/M/$c$ model), let $a = lambda \/ mu$ be the offered load and $rho = a \/ c$ the utilisation. The probability that an arrival must wait is given by the Erlang C formula, and the probability of waiting longer than $t$ is

$ P(W > t) = C(c, a) e^(-(c mu - lambda) t) $

```cpp
#include <cmath>

// Erlang C via the numerically stable Erlang B recurrence.
double erlang_c(int servers, double offered_load) {
    if (offered_load >= servers) return 1.0;
    double b = 1.0;
    for (int k = 1; k <= servers; ++k)
        b = offered_load * b / (k + offered_load * b);
    const double rho = offered_load / servers;
    return b / (1.0 - rho * (1.0 - b));
}

// Smallest server count such that P(wait > t) <= target.
int servers_for(double arrival_rate, double service_rate, double t, double target) {
    const double a = arrival_rate / service_rate;
    for (int c = static_cast<int>(std::floor(a)) + 1;; ++c) {
        const double p = erlang_c(c, a) * std::exp(-(c * service_rate - arrival_rate) * t);
        if (p <= target) return c;
    }
}
```

The model is optimistic for real services (service times have heavy tails, arrivals are burstier than Poisson), but it explains two robust effects. Large pools can run hotter than small ones for the same waiting-time target, because statistical multiplexing smooths the load: with 10 ms requests and a target of at most 1 percent waiting longer than 5 ms, 500 requests per second need 10 servers (50 percent utilisation), 1,000 need 15 (67 percent), and 5,000 need 57 (88 percent). And for any pool, the last few percent of utilisation cost a disproportionate amount of tail latency. A sound target utilisation comes from measuring the latency-versus-load curve of the real service, not from a rule of thumb.

== Scaling Layers

=== VM Groups

AWS Auto Scaling groups, GCP managed instance groups, and Azure VM scale sets keep a set of identical VMs at a desired count, spread them across zones, replace unhealthy ones, and register them with load balancers. Launch templates pin the image and configuration. Instance warm-up settings stop a newly launched VM's low early metrics from triggering scale-in, and *warm pools* keep pre-initialised stopped instances that start in seconds rather than minutes.

=== Pods

The HPA changes replica counts. The Vertical Pod Autoscaler changes resource requests, which historically required restarting the pod; in-place resize is now available in recent Kubernetes versions. Running HPA and VPA on the same metric (CPU) makes them fight, since each reacts to the other's change. Pair HPA on CPU or request rate with VPA in recommendation-only mode, or VPA on memory only.

=== Nodes

Pods that cannot be scheduled need nodes. The Cluster Autoscaler watches for unschedulable pods, simulates which node group would fit them, and grows that group. It removes nodes whose requested resources stay below a threshold (50 percent by default) for a period (10 minutes by default) if their pods fit elsewhere. Karpenter skips node groups: it reads the pending pods' constraints, chooses instance types to fit them directly through the provider's fleet API, and continuously *consolidates* by replacing under-used nodes with fewer or cheaper ones.

Node autoscaling has the longest delay of any layer (VM boot, node registration, image pulls). The common workaround is *overprovisioning*: run low-priority placeholder pods that reserve spare capacity. Real pods preempt them immediately, and the evicted placeholders become unschedulable, which triggers a new node in the background.

== Startup Time Is the Hidden Lever

Everything in the control loop gets easier when new capacity serves sooner. Typical contributions and their fixes:

#table(
  columns: 3,
  [*Phase*], [*Typical time*], [*Reduction*],
  [VM launch and boot], [30-90 s], [Smaller images, warm pools, faster instance families],
  [Bootstrap scripts and configuration], [10 s to minutes], [Bake configuration into images instead of installing at boot],
  [Container image pull], [5-60 s], [Smaller images, pre-pulled base layers, lazy-loading snapshotters],
  [Application start and JIT warm-up], [1-60 s], [Ahead-of-time compilation, class data sharing, snapshot and restore],
  [Readiness and load balancer registration], [10-30 s], [Tight health-check intervals, slow start ramps],
)

Halving startup time can halve the headroom needed for the same burst, which usually saves more money than tuning the scaling policy.

== Capacity Is Not Infinite

Autoscaling assumes the provider has spare capacity. Sometimes it does not.

- *Quotas.* Accounts have per-region limits on vCPUs, IPs, and load balancers. A scale-out that hits a quota fails as an API error, usually during the incident it was meant to handle. Track quota headroom as a monitored metric.
- *Insufficient capacity.* A zone can run out of a particular instance type, especially large, GPU, or newly launched types. Request flexibility across several instance types and all zones, and use allocation strategies that weigh available capacity rather than lowest price alone.
- *Spot reclamation.* Spot capacity can be withdrawn with two minutes' notice on AWS (about 30 seconds on GCP and Azure). Mixed groups with an on-demand base and a spot top layer keep a floor that cannot be reclaimed.
- *Reservations.* For capacity that must exist during a regional event, on-demand capacity reservations (or GCP and Azure equivalents) guarantee it, at full price whether used or not.

== Static Stability

Scaling actions depend on the control plane: launching instances, updating load balancers, writing DNS. Control planes are more complex and more likely to fail than the data plane, and they are exactly what is stressed during a large incident. A design is *statically stable* if it keeps working when the control plane is unavailable, with no new resources launched.

For zone failure this is concrete. With $z$ zones and peak load $L$, the surviving $z - 1$ zones must carry $L$ without launching anything. Each zone must therefore hold capacity $L \/ (z - 1)$, and total provisioned capacity is $z L \/ (z - 1)$: 50 percent overprovisioning with three zones, 100 percent with two. This is why three zones is the default for critical services and why autoscaling minimums are set from the zone-loss case, not from average load.

== Scaling Does Not Fix Everything

- *Shared bottlenecks.* Adding application servers in front of a saturated database adds connections and contention and makes latency worse. Scale signals should include dependency health, and connection pools should cap total load on the database regardless of replica count.
- *Retry storms.* An overloaded service sheds requests; clients retry; load rises further. Autoscaling cannot outrun a retry multiplier. Retry budgets, exponential backoff with jitter, and load shedding at the edge keep demand finite while capacity catches up.
- *Cold caches.* New instances with empty caches send more traffic to the backing store. A large scale-out can turn a cache-friendly workload into a database overload.
- *Scale-in damage.* Removing instances drops in-flight work unless instances drain. Use connection draining, lifecycle hooks, and pod termination grace periods long enough for the slowest request.

== Long-Term Capacity Planning

Autoscaling handles minutes to hours. Capacity planning handles months: forecasting peak demand from growth trends and planned events, checking that quotas, reservations, commitments, and dependencies (databases, third-party rate limits) cover the forecast plus zone-loss headroom, and running load tests to the planned peak. The output is a set of commitments (reservations, savings plans, quota increases) and a list of components that will not scale linearly and need redesign before the peak arrives.

== Pitfalls

- Scaling on latency or error rate alone, reacting after users are already hurt.
- Symmetric, aggressive scale-in that removes capacity just before the next spike.
- HPA and VPA both acting on CPU.
- Minimum counts set from average load rather than from losing a zone.
- One instance type in one zone, so a capacity shortage blocks every scale-out.
- Scale-out that saturates a shared database instead of serving more traffic.
- Ignoring startup time, then compensating with ever-larger headroom.

== Further Reading

Åström, K. J. and Murray, R. M. (2008). _Feedback Systems: An Introduction for Scientists and Engineers_. Princeton University Press.

Harchol-Balter, M. (2013). _Performance Modeling and Design of Computer Systems: Queueing Theory in Action_. Cambridge University Press.

Little, J. D. C. (1961). "A Proof for the Queuing Formula: L = λW." Operations Research.

Rzadca, K. et al. (2020). "Autopilot: Workload Autoscaling at Google." EuroSys.

Verma, A. et al. (2015). "Large-scale Cluster Management at Google with Borg." EuroSys.

Gias, A. U., Casale, G. and Woodside, M. (2019). "ATOM: Model-Driven Autoscaling for Microservices." ICDCS.

Dean, J. and Barroso, L. A. (2013). "The Tail at Scale." Communications of the ACM.

Brooker, M. (2019). "Timeouts, Retries, and Backoff with Jitter." Amazon Builders' Library.

Weiss, B. and Furr, M. (2020). "Static Stability Using Availability Zones." Amazon Builders' Library.

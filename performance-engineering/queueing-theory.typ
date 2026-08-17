#import "../template.typ": xref

= Queueing Theory

Every shared resource with variable demand is a queue: a CPU run queue, a disk, a connection pool, a load balancer, a barista. Queueing theory supplies the small set of formulas that explain why latency explodes near saturation, why variability is as expensive as load, and why "the server is only at 80%" is not reassurance. This chapter covers Little's law, the M/M/1 family, the utilization-latency curve, the effects of variability and multiple servers, and the practical translation to systems work.

*See also:* #xref("performance-engineering", "methodology", label: "Performance Methodology") (USE saturation metrics are queue lengths), #xref("performance-engineering", "capacity-planning", label: "Capacity Planning") (these models, applied forward), #xref("performance-engineering", "concurrency-performance", label: "Concurrency Performance") (thread pools as queueing systems), and #xref("performance-engineering", "io-performance", label: "I/O Performance") (`iostat` is a queueing report).

== Notation and Little's Law

A queueing station is described by arrival rate $lambda$ (work arriving per second), service rate $mu$ (work one server completes per second; service time $S = 1 \/ mu$), the number of servers $c$, and utilization

$ rho = lambda / (c mu) $

Kendall's notation A/S/c names the arrival process, the service-time distribution, and the server count: M/M/1 is Poisson arrivals (M for memoryless), exponential service, one server; M/G/1 generalizes the service distribution; M/M/c models a pool.

*Little's law* (Little, 1961) is the most useful identity in performance work:

$ L = lambda W $

The average number of items in the system equals throughput times average time in the system. It assumes only stationarity, no distributional assumptions, and applies to any boundary you draw: a queue, a server, a whole datacenter. Practical uses:

- *Compute the hidden quantity from the two you can measure.* A service at 5,000 req/s with 200 ms mean latency holds $5000 times 0.2 = 1000$ requests in flight; that is the concurrency the system must support (threads, connections, memory per request).
- *Sanity-check metrics.* If reported concurrency, throughput, and latency violate $L = lambda W$, one of the meters is wrong.
- *Closed-loop benchmark math.* With $N$ load-generator threads in a closed loop with think time $Z$: $X = N \/ (R + Z)$, the interactive response time law. A benchmark with 10 threads and zero think time _cannot_ measure throughput above $10 \/ R$; throughput and latency are not independently settable in a closed system (compare coordinated omission in _Performance Methodology_).

== M/M/1: The Shape of Saturation

For M/M/1, the mean number in system and the mean response time are

$ L = rho / (1 - rho) , quad W = S / (1 - rho) $

The $1 \/ (1 - rho)$ factor is the single most important curve in performance engineering:

#table(
  columns: 3,
  [*Utilization $rho$*], [*Response time (multiple of $S$)*], [*Mean queue (in system)*],
  [50%], [$2 times$], [1],
  [80%], [$5 times$], [4],
  [90%], [$10 times$], [9],
  [95%], [$20 times$], [19],
  [99%], [$100 times$], [99],
)

Latency is hyperbolic in utilization: going from 80% to 90% busy doubles response time; from 90% to 95% doubles it again. The knee is not a defect to be tuned away; it is the mathematics of random arrivals meeting finite capacity. Headroom is not waste, it is the price of latency. The tail is worse than the mean: for M/M/1, response times are exponentially distributed, so the p99 is $ln(100) approx 4.6$ times the mean, which itself is already inflated by $1 \/ (1 - rho)$.

== Variability: M/G/1 and Pollaczek-Khinchine

Random arrivals queue even below saturation, and *variability multiplies the damage*. For M/G/1 (general service times), the Pollaczek-Khinchine formula gives the mean wait in queue:

$ W_q = (rho S) / (1 - rho) dot (1 + C_s^2) / 2 $

where $C_s^2$ is the squared coefficient of variation of service time (variance over mean squared). Exponential service has $C_s^2 = 1$; constant service ($C_s^2 = 0$) halves the queueing delay; a heavy-tailed service distribution with $C_s^2 = 10$ multiplies it by 5.5 at the same utilization. The systems translation: *one slow request type poisons the latency of every request behind it*. This is why mixing 10 ms queries and 10 s analytics scans in one pool is ruinous, why HTTP/2 head-of-line blocking at the TCP layer mattered, and why isolating workload classes (separate pools, size-based scheduling) is usually worth more than adding capacity. Kingman's approximation extends this to general arrivals (G/G/1): wait scales with $(C_a^2 + C_s^2) \/ 2$, so bursty arrivals (retry storms, thundering herds, synchronized cron jobs) hurt exactly like variable service.

Scheduling interacts with variability: FIFO is fair but lets elephants block mice; *shortest-remaining-processing-time* is mean-optimal; processor sharing (what an OS scheduler approximates) insulates small jobs from large ones without knowing sizes. Size-aware dispatch and "fast lane" queues are the practical forms.

== Multiple Servers: M/M/c and Pooling

With $c$ servers fed by one shared queue (M/M/c), the probability an arrival must wait is given by the *Erlang C* formula, and waiting time falls dramatically with pool size at equal $rho$: one big fast server beats many slow ones for mean latency, and one shared queue beats per-server queues. A single queue feeding 16 workers at 90% utilization waits far less than 16 separate M/M/1 queues at 90%, because idle servers and waiting work can never coexist. This is the argument for shared run queues with stealing, for load balancers using join-shortest-queue rather than random, and against statically partitioned capacity.

Two related results worth knowing:

- *The power of two choices* (Mitzenmacher, 2001): dispatching each arrival to the shorter of two _randomly sampled_ queues achieves nearly the benefit of join-shortest-queue at a fraction of the coordination cost: maximum queue length drops from $Theta(log n \/ log log n)$ (random) to $Theta(log log n)$. This is implemented in Envoy, NGINX (`least_conn` over a sample), and most modern balancers.
- *Erlang B* governs loss systems (no queue, blocked calls dropped): the right model for connection-limited resources that reject rather than wait.

The trade-off pooling ignores: a shared queue requires shared dispatch (a contention point, see _Concurrency Performance_) and destroys cache locality; thread-per-core designs deliberately accept worse queueing behavior for better per-request cost.

== Open vs. Closed Systems

In an *open* system arrivals are independent of completions (the internet does not slow down because your server did). In a *closed* system, a fixed population waits for responses before re-issuing (a fixed thread pool of clients, a batch pipeline). The difference matters enormously under overload: a closed system self-throttles and degrades gracefully; an open system at $lambda > c mu$ has *unbounded* queue growth: latency climbs until timeouts, retries add load, and the system spirals. Schroeder, Wierman, and Harchol-Balter (2006) showed that closed-loop benchmarks systematically understate the latency an open-arrival production workload will see at the same throughput. Most production services face open (or partly open) arrivals; most load tests are closed. Use open-loop generators (`wrk2`, Vegeta at fixed rate) when the production process is open.

Networks of queues extend the single station: Jackson networks justify analyzing stations of a pipeline independently, each with the $lambda$ implied by flow balance, and *operational analysis* (Denning & Buzen, 1978) derives bottleneck bounds from measured visit counts and service demands: system throughput is capped by $1 \/ D_max$, the reciprocal of the largest total service demand at any station, which identifies the bottleneck device from utilization measurements alone ($U_i = lambda D_i$).

== Using the Theory Honestly

The assumptions behind the closed forms (Poisson arrivals, stationarity, exponential service) rarely hold exactly; arrivals are bursty and correlated, service is heavy-tailed, and load shifts. The models still earn their keep as:

- *Bounds and shapes*: the $1 \/ (1 - rho)$ blow-up, the variability multiplier, and Little's law are robust even when the exact constants are not.
- *Back-of-envelope filters*: a design whose Little's-law concurrency exceeds its connection pool, or that plans to run a latency-sensitive disk at 95%, is wrong before any benchmark runs.
- *Interpretation of measurements*: `iostat`'s `aqu-sz` and `await` are $L$ and $W$; PSI is saturation; a hockey-stick latency-vs-throughput plot from a load test is the curve these formulas predict, and fitting it locates effective capacity.

When the formulas are insufficient (priorities, retries, timeouts, correlated failures), discrete-event simulation is cheap and decisive.

== Pitfalls

- *Provisioning to average utilization*: averages over 5-minute windows hide sub-second bursts that sit deep in the hyperbolic region; queueing happens at the timescale of arrivals, not of dashboards.
- *Treating utilization as linear in cost*: the step from 60% to 90% "saves" a third of the fleet and multiplies queueing delay several-fold.
- *Closed-loop tests for open-loop systems*: see above; this and coordinated omission are the two standard ways load tests lie.
- *Ignoring retries*: retries multiply $lambda$ exactly when $rho$ is highest; an unjittered retry policy converts a brief saturation into a metastable outage.
- *Pooling heterogeneous work*: P-K says the variance, not just the mean, of service time sets the wait. Segregate the elephants.

== Worked Example

Size a service: requests arrive at $lambda = 160$ req/s, mean service time is $S = 20$ ms, so each server completes $mu = 1 \/ S = 50$ req/s. The offered load is $a = lambda \/ mu = 3.2$ servers' worth of work; the proposed pool is $c = 4$, giving

$ rho = lambda / (c mu) = 160 / 200 = 0.80 $

For M/M/c, the probability an arrival waits is Erlang C. With $a = 3.2$, $c = 4$: the terms $a^k \/ k!$ for $k = 0..3$ are $1$, $3.2$, $5.12$, $5.461$ (sum $14.781$), and $a^4 \/ 4! = 4.369$, which divided by $1 - rho = 0.2$ gives $21.845$. So

$ P_"wait" = 21.845 / (14.781 + 21.845) = 0.596 $

The mean wait in queue is

$ W_q = P_"wait" / (c mu - lambda) = 0.596 / 40 = 14.9 "ms" $

so mean response time is $W = S + W_q = 34.9$ ms, and Little's law says the system holds $L = lambda W = 160 times 0.0349 approx 5.6$ requests on average, so a connection pool of, say, 8 has margin.

Two checks the formulas make cheap:

- *Pooling dividend.* The same hardware as four separate M/M/1 queues, each at $lambda = 40$, $mu = 50$, $rho = 0.8$, waits $W_q = rho \/ (mu - lambda) = 0.8 \/ 10 = 80$ ms, more than five times the shared-queue wait of 14.9 ms. Idle servers next to waiting work is the entire difference.
- *Headroom sensitivity.* Let traffic grow 12.5% to $lambda = 180$ ($rho = 0.9$). Redoing Erlang C with $a = 3.6$: terms $1, 3.6, 6.48, 7.776$ (sum $18.856$), $a^4 \/ 4! = 6.998$, over $1 - rho = 0.1$ gives $69.98$, so $P_"wait" = 69.98 \/ 88.84 = 0.788$ and $W_q = 0.788 \/ 20 = 39.4$ ms. A 12.5% load increase multiplied the queueing delay by $2.6 times$ and pushed $W$ from 34.9 ms to 59.4 ms. If the SLO is 50 ms mean, capacity is 4 servers at today's load and 5 servers at next quarter's, and the hyperbola says there is no way to "tune" around that.

== Further Reading

- Harchol-Balter, M. (2013). _Performance Modeling and Design of Computer Systems: Queueing Theory in Action_. Cambridge University Press.
- Little, J. D. C. (1961). A proof for the queuing formula $L = lambda W$. _Operations Research_, 9(3).
- Schroeder, B., Wierman, A., & Harchol-Balter, M. (2006). Open versus closed: a cautionary tale. _NSDI_.
- Mitzenmacher, M. (2001). The power of two choices in randomized load balancing. _IEEE TPDS_, 12(10).
- Denning, P., & Buzen, J. (1978). The operational analysis of queueing network models. _ACM Computing Surveys_, 10(3).
- Lazowska, E. et al. (1984). _Quantitative System Performance_. Prentice-Hall (free online).

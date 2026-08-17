#import "../template.typ": xref

= SLO Engineering

Service Level Objectives translate reliability ambitions into operational contracts: a number the product team owns, the engineering team defends, and the business uses to make prioritization decisions. Unlike SLAs, which are legal instruments with financial penalties, SLOs are internal agreements that drive on-call behavior and feature-vs-reliability tradeoffs. This chapter builds the full stack — SLI measurement, budget arithmetic, multi-window burn rate alerting, and the organizational practices that make SLOs durable.

*See also:* #xref("observability-and-sre", "metrics-systems", label: "Metrics Systems"), #xref("observability-and-sre", "distributed-tracing", label: "Distributed Tracing"), #xref("observability-and-sre", "the-three-pillars-and-beyond", label: "The Three Pillars and Beyond")

== SLI, SLO, SLA: Precise Definitions

An *SLI* (Service Level Indicator) is a quantitative measurement of service behavior, expressed as a ratio: good events divided by total events over a rolling or calendar window. An *SLO* (Service Level Objective) is a target threshold on an SLI, owned by the team. An *SLA* (Service Level Agreement) is a contractual commitment to a customer, typically looser than the internal SLO, with defined remedies for breach.

#table(
  columns: (auto, auto, auto, auto),
  align: left,
  table.header[*Term*][*Owner*][*Audience*][*Consequence of breach*],
  [SLI], [Engineering], [Internal], [Data point],
  [SLO], [Engineering/Product], [Internal], [Error budget consumed],
  [SLA], [Business/Legal], [External], [Refund, penalty, churn],
)

The critical practice: set the SLA several points looser than the SLO. If the SLO is 99.9 % availability, the SLA might be 99.5 %. This gives engineering headroom to detect and fix problems before contractual breach.

=== Choosing the Right SLI

Not every metric is an SLI. Good SLIs are directly correlated with user experience, ratio-based (so they are bounded 0–1), and stable enough to trend over weeks. Common SLI families:

- *Availability:* fraction of requests that return a non-error response.
- *Latency:* fraction of requests served within a threshold (e.g., under 300 ms). Preferred over P99 as an SLI because it is a ratio, not a distribution statistic.
- *Freshness:* fraction of reads that return data updated within a threshold.
- *Correctness:* fraction of operations whose output matches a known-good reference (canary comparison).
- *Throughput:* fraction of time a pipeline processes data fast enough to keep up with ingest.

== Error Budget

The *error budget* is $1 - "SLO"$ expressed as a fraction of total events (or time). It quantifies how much unreliability is acceptable before the SLO is breached.

For a 30-day calendar window:

$ "budget"_"minutes" = 30 times 24 times 60 times (1 - "SLO") $

Worked example for 99.9 % availability over 30 days:

$ "budget" = 43200 times 0.001 = 43.2 "min/month" $

For 99.95 %: $43200 times 0.0005 = 21.6$ min. For 99.99 %: $43200 times 0.0001 = 4.32$ min — about four and a half minutes per month, leaving almost no headroom for planned maintenance.

#table(
  columns: (auto, auto, auto),
  align: left,
  table.header[*SLO*][*Monthly budget (min)*][*Yearly budget (hr)*],
  [99 %], [432], [87.6],
  [99.5 %], [216], [43.8],
  [99.9 %], [43.2], [8.76],
  [99.95 %], [21.6], [4.38],
  [99.99 %], [4.32], [0.876],
)

The budget is a shared resource. Deploys, infrastructure migrations, and incident recovery all draw from it. When the budget is depleted, the team's priority shifts from new features to reliability improvements — a forcing function that makes the tradeoff explicit.

=== Burn Rate

*Burn rate* is how fast the error budget is being consumed relative to the rate that would exactly exhaust it by window end. A burn rate of 1.0 means the budget is being consumed at the rate that would exhaust it in exactly one window. A burn rate of 2.0 means it will be exhausted in half the window.

$ "burn rate" = ("error rate") / (1 - "SLO") $

For a 99.9 % SLO, the allowed error rate is 0.1 %. If the current error rate is 1 %, the burn rate is:

$ "burn rate" = 0.01 / 0.001 = 10 $

At burn rate 10, the error budget is exhausted in $30 / 10 = 3$ days.

== Multi-Window Burn Rate Alerting

Simple burn rate threshold alerts have a fatal flaw: a 1-hour window has high recall (catches fast outages) but low precision (noisy for slow burns). A long window catches slow burns but pages too late during fast outages. The Google SRE Book's recommended solution is *multi-window burn rate alerts*, which require both a short window and a long window to simultaneously exceed thresholds.

=== Alert Windows and Thresholds

The standard configuration from the Google SRE Workbook (Chapter 5):

#table(
  columns: (auto, auto, auto, auto),
  align: left,
  table.header[*Severity*][*Burn rate*][*Long window*][*Short window*],
  [Page], [14×], [1 h], [5 min],
  [Page], [6×], [6 h], [30 min],
  [Ticket], [3×], [3 d (72 h)], [6 h],
  [Ticket], [1×], [30 d], [—],
)

At 14× burn rate, the budget is exhausted in $30/14 approx 2.1$ days. The 1-hour window catches this within an hour of onset. The 5-minute short window resets the alert quickly after recovery, preventing flap-based noise.

=== PromQL Implementation

```promql
# Availability SLI recording rules (evaluate every 30s)
- record: job:http_requests_total:rate1h
  expr: sum by (job) (rate(http_requests_total[1h]))

- record: job:http_errors_total:rate1h
  expr: sum by (job) (rate(http_requests_total{code=~"5.."}[1h]))

- record: job:slo_error_ratio:rate1h
  expr: job:http_errors_total:rate1h / job:http_requests_total:rate1h

# Multi-window burn rate alert (page — fast burn)
- alert: SLOBurnRateFast
  expr: |
    (
      job:slo_error_ratio:rate1h > (14 * 0.001)
    ) and (
      job:slo_error_ratio:rate5m > (14 * 0.001)
    )
  for: 2m
  labels:
    severity: page
  annotations:
    summary: "Fast burn: {{ $value | humanizePercentage }} error rate"

# Multi-window burn rate alert (ticket — slow burn)
- alert: SLOBurnRateSlow
  expr: |
    (
      job:slo_error_ratio:rate6h > (3 * 0.001)
    ) and (
      job:slo_error_ratio:rate72h > (3 * 0.001)
    )
  for: 15m
  labels:
    severity: ticket
```

The `for: 2m` clause on the fast-burn alert prevents spurious pages from a single bad scrape interval. The long-window recording rules should be precomputed; subquery evaluation of 6-hour windows on every alert evaluation is expensive.

== Latency SLOs

Availability SLOs measure binary success; latency SLOs measure whether responses are *fast enough*. A latency SLI is a ratio — the fraction of requests served within a threshold — not a raw percentile. This makes it aggregable across replicas and compatible with burn rate math.

$ "latency SLI" = ("requests completed in" < T) / ("total requests") $

Choose the threshold $T$ from user research, not from histograms. If users notice latency above 300 ms, that is your threshold. A 99th-percentile latency number is useful for debugging but is a poor SLI because it is not a ratio, it cannot be summed across instances, and it is sensitive to outliers.

=== Multi-Threshold Latency SLOs

For request classes with different latency profiles, define multiple SLIs:

#table(
  columns: (auto, auto, auto),
  align: left,
  table.header[*SLI*][*Threshold*][*SLO*],
  [Interactive reads $< 50$ ms], [50 ms], [99 %],
  [Interactive reads $< 300$ ms], [300 ms], [99.9 %],
  [Batch writes $< 5$ s], [5 s], [95 %],
)

In PromQL with a classic histogram:

```promql
# Fraction of requests under 300 ms
sum(rate(http_request_duration_seconds_bucket{le="0.3"}[5m]))
  / sum(rate(http_request_duration_seconds_count[5m]))
```

== SLOs as Product-Engineering Contracts

An SLO without organizational backing is a metric. With backing, it is a contract between product and engineering that makes tradeoffs legible. The contract has four clauses:

1. *Definition:* what the SLI measures, how it is computed, what counts as a good event.
2. *Target:* the SLO percentage and the measurement window.
3. *Budget policy:* what happens when budget is exhausted (freeze features, mandatory reliability sprint).
4. *Review cadence:* when the SLO is reconsidered (quarterly is common).

The budget policy is the most important clause. Without it, teams treat the SLO as advisory. A freeze policy — no non-critical deploys when budget is $< 10$% remaining — creates immediate incentive alignment.

=== Risks of Over-Tight SLOs

Setting an SLO too high has compounding costs:

- *Alert fatigue:* a 99.99 % SLO on a service with 0.05 % organic error rate leaves only 8 minutes of annual budget. Any dependency failure pages immediately.
- *Innovation tax:* teams with exhausted budgets cannot ship features, even unrelated ones.
- *SLA inversion:* internal SLO tighter than the customer-facing SLA means any internal incident triggers an SLA review.
- *Measurement noise:* at very high SLOs, measurement error (probe failures, clock skew) can appear as budget consumption.

A practical heuristic: start with a target 10–20 % looser than current performance. Tighten only when the team has observability and runbooks to defend it.

== SLO Dashboards

An SLO dashboard has four panels at minimum:

1. *Error budget remaining* (time series, as percentage): how much budget is left in the current window.
2. *Burn rate* (time series, 1 h and 6 h): current consumption rate.
3. *SLI ratio* (time series): the raw availability or latency ratio.
4. *Budget drawdown* (cumulative): total budget consumed since window start.

```promql
# Budget remaining (%)
(1 - (
  sum_over_time(job:slo_error_ratio:rate5m[30d]) * 5 / (60 * 24 * 30)
)) / (1 - 0.999) * 100
```

Dashboard annotations showing deploys, incidents, and maintenance windows let teams correlate budget drawdown with root causes. Without annotations, burn rate spikes are uninterpretable.

== Further Reading

Beyer, B. et al. (2016). _Site Reliability Engineering._ O'Reilly. Chapters 4–5 (SLOs and Error Budgets).

Beyer, B. et al. (2018). _The Site Reliability Workbook._ O'Reilly. Chapter 5 (Alerting on SLOs).

Nygard, M. (2018). _Release It! 2nd ed._ Pragmatic Programmers. Chapter 5 (Stability Patterns).

Treynor Sloss, B. et al. (2017). "The Calculus of Service Availability." ACM Queue 15(2).

Hausenblas, M. (2022). _Cloud Observability in Action._ Manning. Chapter 8.

Google. "SRE Books." https://sre.google/books/

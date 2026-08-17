#import "../template.typ": xref

= Alerting

An alert is a claim that something deserves human attention; a page is a claim that it deserves attention _now_. Alerting design is therefore an exercise in classifier engineering: the alert pipeline labels system states as page-worthy or not, and it can be wrong in both directions. Too many false positives and responders stop trusting the pager; too many false negatives and customers become your monitoring. This chapter covers the symptom-versus-cause principle, alert quality as precision and recall, the multi-window multi-burn-rate method from the SRE Workbook with its arithmetic, the routing machinery (Alertmanager), and the human side — on-call health.

*See also:* #xref("observability-and-sre", "slo-engineering", label: "SLO Engineering"), #xref("observability-and-sre", "incident-management", label: "Incident Management"), #xref("observability-and-sre", "metrics-systems", label: "Metrics Systems"), #xref("observability-and-sre", "the-three-pillars-and-beyond", label: "The Three Pillars and Beyond")

== Symptoms versus Causes

Rob Ewaschuk's "My Philosophy on Alerting" (2013, later folded into the SRE book's monitoring chapter) draws the central distinction:

- A *symptom* alert fires on user-visible badness: error ratio, latency SLI, freshness. It answers "are users hurting?"
- A *cause* alert fires on an internal condition believed to lead to badness: disk 90 % full, replica lag, CPU saturation, a process restart.

Page on symptoms; keep causes for dashboards, tickets, and diagnosis. The reasoning: cause-based alerts encode a model of how the system fails, and that model is always incomplete in both directions. Causes fire when the system has degraded gracefully (a dead replica behind a healthy load balancer — nobody needs to wake up), and they stay silent for novel failure modes the model never anticipated. Symptom alerts are robust to both: they fire if and only if users are affected, regardless of mechanism.

Two legitimate exceptions: imminent, hard deadlines where waiting for symptoms forfeits the response window (disk will fill in 4 hours, certificate expires in 24), and symptoms-as-seen-by-an-intermediate-layer — your database's latency is a "cause" to the web tier but a symptom of the database service itself, and the database team may reasonably page on it.

== Alert Quality: Precision, Recall, and Pager Fatigue

Treat the pager as a binary classifier over time windows and the standard vocabulary applies:

$ "precision" = ("true pages") / ("true pages" + "false pages"), quad "recall" = ("true pages") / ("true pages" + "missed incidents") $

A "true page" is one that required urgent human action. Add two more properties: *detection time* (how long after onset the page fires) and *reset time* (how long after recovery it stops firing). Every alerting design trades among these four; the multi-window method below exists precisely because a single threshold cannot optimize all of them.

Precision matters more than intuition suggests because of *pager fatigue*, the operational cousin of alarm fatigue documented in clinical settings (where studies have found 72–99 % of clinical alarms to be non-actionable, with deadly desensitization effects). The same dynamics hold on-call: responders who learn that most pages are noise begin to acknowledge-and-ignore, and the latency of response to _real_ pages rises. Noise also hides signal directly — a real incident's page arrives interleaved with four flapping known-noisy alerts.

Practical hygiene that keeps the classifier honest:

- *Every page gets a disposition:* actionable, not-actionable, or duplicate. Review weekly; an alert whose pages are repeatedly non-actionable gets retuned, demoted to ticket, or deleted.
- *Every page has a runbook link* (see _Incident Management_). If no action is conceivable, it is not a page.
- *Budget pages like errors:* Google's guideline of at most about two incidents per 12-hour shift is a recall-side bound too — beyond it, follow-up quality collapses.
- *Delete fearlessly:* an alert that has never fired truly, or whose condition is covered by a symptom alert, is pure risk.

== Multi-Window, Multi-Burn-Rate SLO Alerts

The SRE Workbook (chapter 5) develops SLO alerting through six iterations; the destination is the multi-window, multi-burn-rate alert. The setup: an SLO of, say, 99.9 % over a 30-day window gives an error budget fraction $1 - "SLO" = 0.001$. The *burn rate* normalizes the observed error rate by the budget:

$ "burn rate" = ("observed error ratio") / (1 - "SLO") $

Burn rate 1 consumes exactly the budget over the full window; burn rate $B$ exhausts it in $30 \/ B$ days.

=== The Budget-Consumption Arithmetic

Choose alerts by the *fraction of budget consumed* before a human is notified. If an alert fires when burn rate $B$ has been sustained over a long window of length $T_"long"$, the budget consumed at firing time is:

$ "budget consumed" = (B times T_"long") / T_"window" $

The Workbook's recommended three-tier configuration for a 30-day window:

#table(
  columns: (auto, auto, auto, auto, auto),
  align: left,
  table.header[*Response*][*Burn rate*][*Long window*][*Short window*][*Budget at fire*],
  [Page], [14.4], [1 h], [5 min], [2 %],
  [Page], [6], [6 h], [30 min], [5 %],
  [Ticket], [1], [3 d], [6 h], [10 %],
)

The numbers come from the formula: $14.4 times 1 \/ 720 = 2 %$ of the 30-day budget (720 hours), $6 times 6 \/ 720 = 5 %$, and $1 times 72 \/ 720 = 10 %$. The tiers cover the spectrum: a fast burn (total outage is burn rate $1\/0.001 = 1000$) trips the 1-hour window within minutes; a slow leak at just above budget rate is caught by the 3-day ticket alert before it silently eats the month.

=== Why Two Windows

A single long window has slow *reset*: after a 10-minute outage trips the 1-hour alert, the 1-hour average stays elevated for the remaining 50 minutes, re-paging or blocking resolution. The fix is to require a *short window* (conventionally $1\/12$ of the long window) to be elevated _simultaneously_:

$ ("rate"[1 h] > 14.4 times 0.001) and ("rate"[5 "min"] > 14.4 times 0.001) $

The short window confirms the burn is still happening _now_, cutting reset time from the long-window length to roughly the short-window length, with negligible loss of detection capability. Detection time for a total outage with the 14.4×/1 h alert is about $(0.001 times 14.4) \/ 1.0 times 60 approx 0.86$ minutes of sustained 100 % errors — under a minute — while a mere 0.2 % error ratio (burn rate 2) never pages and correctly lands as a ticket days later.

Low-traffic services break the math: at 10 requests/hour, one failure is a 10 % error ratio over the short window. Mitigations include lengthening windows, aggregating related small services into one SLO, or generating synthetic probe traffic to raise the denominator.

== Routing, Deduplication, and Silences

Between the rule engine and the pager sits an alert router; Prometheus *Alertmanager* is the reference design. Its pipeline:

- *Grouping:* alerts sharing labels (e.g., `cluster`, `alertname`) are batched into one notification — a switch failure producing 200 instance-down alerts becomes one page listing 200 instances. Timing knobs: `group_wait` (initial batching delay, ~30 s), `group_interval` (delay before notifying about new members, ~5 min), `repeat_interval` (re-notification period, hours).
- *Routing tree:* a hierarchy matching on labels sends `severity: page, team: db` to PagerDuty and `severity: ticket` to a queue. Routing on labels rather than alert names keeps the policy in one place.
- *Inhibition:* suppress alerts implied by another firing alert — if `DatacenterOnFire` fires, inhibit every per-service alert carrying the same `datacenter` label. This is the mechanism that keeps cause alerts from amplifying a known symptom.
- *Silences:* time-bounded, label-matched mutes with an author and a comment, used for planned maintenance. Silences must expire; an unbounded silence is a deleted alert without the honesty.
- *Deduplication and high availability:* Alertmanager instances gossip notification state, so running three replicas does not triple the pages; the paging service deduplicates further on an incident key.

The router is also where flap damping lives (`for:` clauses in rules, group intervals in routing) and where the "page once per incident, not once per evaluation" contract is enforced (see _Incident Management_ for what happens downstream).

== On-Call Health Metrics

The alerting system's output is consumed by humans whose capacity is finite and measurable. Metrics worth tracking per rotation:

- *Pages per shift,* split business-hours versus out-of-hours. Out-of-hours pages carry the real cost; one 3 a.m. page degrades the next day. Sustained rates above about two per shift indicate either real reliability debt or classifier noise — and the disposition data tells you which.
- *Actionability ratio:* the precision of the pager, from the per-page dispositions above. Below roughly 50 %, expect desensitization.
- *Time to acknowledge:* rising ack times are the leading indicator of fatigue, visible before anyone says they are burned out.
- *Sleep-hours interruptions and weekend pages:* track distributional fairness across the rotation, not just team totals.
- *Escalation rate:* how often the primary fails to ack — a paging-pipeline health signal, not a personal one.

The SRE book pairs these with structural limits: minimum rotation sizes (eight people for a 24/7 single-site rotation, or six per site dual-site), compensation or time off in lieu for on-call, and the error-budget feedback loop — if the pager is hot, budget policy should be redirecting engineering time toward reliability rather than adding alerts. An alert review meeting that examines every page from the past week, with authority to delete alerts, is the single highest-leverage recurring practice in this chapter.

== Further Reading

Beyer, B. et al. (2018). _The Site Reliability Workbook._ O'Reilly. Chapter 5 (Alerting on SLOs) — the multi-window multi-burn-rate derivation.

Beyer, B. et al. (2016). _Site Reliability Engineering._ O'Reilly. Chapters 6 (Monitoring Distributed Systems) and 11 (Being On-Call).

Ewaschuk, R. (2013). "My Philosophy on Alerting." Public Google doc, later adapted into the SRE book.

Prometheus Authors. "Alertmanager Documentation." https://prometheus.io/docs/alerting/latest/alertmanager/

Sendelbach, S., Funk, M. (2013). "Alarm Fatigue: A Patient Safety Concern." AACN Advanced Critical Care 24(4). The clinical literature on alarm desensitization.

Hausenblas, M. (2022). _Cloud Observability in Action._ Manning. Chapter 9.

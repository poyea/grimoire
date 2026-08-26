#import "../template.typ": xref

= Incident Management

An incident is an unplanned degradation of service that demands coordinated human response. Incident management is the discipline that turns that response from ad-hoc heroics into a repeatable process: detection, mobilization, mitigation, resolution, and learning. The structures here descend from the Incident Command System (developed by California fire agencies in the 1970s, later adopted by FEMA) and were adapted by Google, PagerDuty, and Atlassian into the patterns most engineering organizations now run. This chapter covers the lifecycle, the roles, the paging machinery, and the postmortem practice — including why "MTTR" is a more fragile number than dashboards suggest.

*See also:* #xref("observability-and-sre", "alerting", label: "Alerting"), #xref("observability-and-sre", "slo-engineering", label: "SLO Engineering"), #xref("observability-and-sre", "chaos-engineering", label: "Chaos Engineering"), #xref("observability-and-sre", "the-three-pillars-and-beyond", label: "The Three Pillars and Beyond")

== The Incident Lifecycle

A useful decomposition of an incident's timeline:

1. *Onset:* the fault begins affecting users (often before anyone notices).
2. *Detection:* monitoring (or a customer) surfaces the problem. Time from onset to detection is *TTD*.
3. *Triage and mobilization:* severity is assigned, responders are paged, an incident channel is opened.
4. *Mitigation:* user impact is stopped or reduced — rollback, failover, feature-flag disable, traffic drain. Time from mobilization to this point is *TTM* (time to mitigate).
5. *Resolution:* the underlying fault is fixed and the system returns to a fully healthy state.
6. *Learning:* postmortem, action items, and follow-through.

The most important distinction in the lifecycle is *mitigation versus resolution*. Mature organizations optimize relentlessly for mitigation — generic levers like rollback and drain that work without understanding the root cause — and accept that diagnosis can happen afterward, off the clock. Google's SRE book frames this as: during an incident, make the system work; understand why later.

=== Severity Levels

Severity classification exists to calibrate response, not to assign blame to the number. A typical scheme:

#table(
  columns: (auto, auto, auto, auto),
  align: left,
  table.header[*Sev*][*Impact*][*Response*][*Example*],
  [SEV-1], [Critical, widespread user impact or data loss], [Page IC + execs, all-hands, status page], [Full outage, payment failure],
  [SEV-2], [Major degradation, partial outage], [Page on-call team, open incident channel], [One region down, P99 latency 10×],
  [SEV-3], [Minor impact, workaround exists], [Business-hours response, ticket], [Degraded non-critical feature],
  [SEV-4], [No user impact, risk identified], [Backlog], [Redundancy lost, budget burn anomaly],
)

Two rules make severity schemes work in practice. First, *when in doubt, escalate*: it is cheap to downgrade a SEV-1 to a SEV-2 and expensive to discover an hour in that you under-called it. Second, severity is assigned by *impact*, not by suspected cause — a one-line config error and a datacenter fire are both SEV-1 if users cannot check out.

== Roles: The Incident Command System

The ICS insight is that coordination is a job, and the person doing it must not also be debugging. The core roles:

- *Incident Commander (IC):* owns the response. Assigns work, makes mitigation decisions, declares severity changes and resolution. Explicitly does _not_ touch keyboards to fix things. PagerDuty's training materials phrase the IC's authority bluntly: during the incident, the IC outranks the CEO on response decisions.
- *Operations / Subject Matter Experts (ops lead):* hands-on investigation and mitigation, reporting findings to the IC.
- *Communications lead:* updates stakeholders, the status page, and customer support on a fixed cadence (every 30 minutes for SEV-1 is common), shielding responders from "any update?" pings.
- *Scribe:* maintains a timestamped log of observations, hypotheses, and actions — the raw material for the postmortem.

For small incidents one person wears all hats; the discipline is knowing when to split them. A common trigger: if the responder cannot simultaneously debug and answer questions in the channel, hand off either the IC or ops role. Handoffs (including across time zones for long incidents) must be explicit: "I am now IC" in the channel, with a state summary.

== Paging and Escalation

The paging pipeline runs from alert source (Alertmanager, monitoring SaaS) through a paging service (PagerDuty, Opsgenie, Grafana OnCall) to a human. Key mechanics:

- *Escalation policies:* if the primary on-call does not acknowledge within $N$ minutes (5–15 typical), page the secondary, then the team lead. Every page must terminate at a human who will acknowledge.
- *Schedules and overrides:* rotations of a week or less, with follow-the-sun rotations for global teams to avoid 3 a.m. pages.
- *Acknowledgment semantics:* ack stops escalation but starts the clock on the responder; an acked-then-ignored page is the worst failure mode.
- *Deduplication keys:* repeated firings of the same alert update one incident rather than re-paging.

A page is a contract: it asserts that a human must act _now_. Anything that does not meet that bar belongs in a ticket queue (see _Alerting_). Google's SRE book suggests a ceiling of roughly two paging incidents per 12-hour shift; beyond that, responders cannot follow up properly on any of them.

== Runbooks

A runbook (playbook) is the bridge between an alert and a mitigation. The minimum viable runbook for an alert answers four questions:

1. What does this alert mean, in terms of user impact?
2. How do I confirm it is real? (dashboard links, queries to run)
3. What are the known mitigations, in order of preference? (exact commands, rollback procedure)
4. When and how do I escalate?

Good runbooks are imperative and copy-pasteable; "investigate the database" is not a step. They decay quickly — a stale runbook that issues a destructive command against a renamed cluster is worse than none — so teams attach runbook review to alert review, and some validate runbook commands in game days (see #xref("observability-and-sre", "chaos-engineering", label: "Chaos Engineering")). Transposit, Netflix, and others have pushed toward executable runbooks: scripts with confirmation prompts rather than prose, which both speeds mitigation and is a step toward automating the response away entirely. The end state of a perfect runbook is automation that makes the page unnecessary.

== Postmortems

The postmortem (incident retrospective) converts an incident's cost into organizational learning. The canonical structure: summary, impact (duration, users affected, SLO budget consumed, revenue if known), timeline with timestamps, contributing factors, what went well / what went poorly / where we got lucky, and action items with owners and deadlines.

=== Blamelessness and Contributing Factors

The *blameless* framing, popularized by John Allspaw's 2012 Etsy post "Blameless PostMortems and a Just Culture," rests on a practical argument: if engineers fear punishment, they hide information, and the organization learns less than it paid for. Blameless does not mean consequence-free; it means the analysis assumes people acted reasonably given what they knew, and asks why the system made the error easy and the detection slow.

Modern incident analysis, drawing on safety science (Sidney Dekker's _The Field Guide to Understanding 'Human Error'_, 2002; Richard Cook's "How Complex Systems Fail," 1998), rejects the phrase "root cause" in favor of *contributing factors*. Complex-system failures require multiple conditions to align: the bug, plus the gap in review, plus the missing alert, plus the stale runbook. Stopping at the first "root cause" (usually the last human action before the outage) systematically produces shallow fixes. The "5 whys" technique shares this flaw — each "why" picks one branch of a tree and discards the others.

Action items deserve skepticism too: postmortems that produce twenty action items typically see few completed. Better practice is two or three high-leverage items tracked like any other engineering work, plus explicit acceptance of risks not worth fixing.

=== MTTR and Its Critiques

Mean time to recovery decomposes additively:

$ "MTTR" = "TTD" + "TTT" + "TTM" $

where TTD is time to detect, TTT time to triage and mobilize, and TTM time from mobilization to mitigation. The decomposition is useful because the levers differ: TTD improves with better alerting, TTT with paging hygiene and clear ownership, TTM with rollback speed and runbooks.

But MTTR as a tracked metric has serious problems, articulated forcefully by Štěpán Davidovič's "Incident Metrics in SRE" (Google, 2021) and the VOID Report (Courtney Nash, 2021–2023, analyzing thousands of public incident reports):

- *Skewed distributions:* incident durations are roughly log-normal with heavy tails. The mean is dominated by outliers; with typical incident counts (tens per quarter), the variance of the sample mean is so large that a real 10 % improvement is statistically invisible. Davidovič's Monte Carlo simulations show teams would mostly be reading noise.
- *Shallow proxy:* duration does not measure impact (a 5-minute payment outage beats a 5-hour outage of an internal tool) or learning.
- *Gaming pressure:* timestamps for "start" and "resolved" are judgment calls; tracking the metric pressures the judgment.

The practical conclusion is not to ignore time — the TTD/TTM decomposition per incident is genuinely diagnostic — but to avoid treating fleet-wide MTTR trends as evidence of anything. Count of SEV-1s, SLO budget consumption, and action-item completion rates are sturdier signals.

== Incident Tooling

Tooling reduces coordination overhead during the worst possible time to have overhead:

- *Paging:* PagerDuty (2009), Opsgenie, Splunk On-Call, Grafana OnCall (open source) — schedules, escalation, dedup.
- *Incident coordination:* incident.io, FireHydrant, Rootly, Blameless, and Netflix's open-source Dispatch (2020). These bots create the Slack channel, assign roles, track the timeline, and generate the postmortem skeleton from channel history.
- *Status pages:* Atlassian Statuspage, instatus — external comms decoupled from the production stack (host the status page outside your own infrastructure).
- *ChatOps:* the incident channel as system of record; the scribe role increasingly automated via channel export and LLM-assisted timeline drafts.

A subtle tooling requirement: the incident stack must not share fate with production. If the SSO provider is down, can responders reach the runbooks? Game days should test the response system itself, not just the serving stack.

== Further Reading

Beyer, B. et al. (2016). _Site Reliability Engineering._ O'Reilly. Chapters 13–15 (Emergency Response, Managing Incidents, Postmortem Culture).

Allspaw, J. (2012). "Blameless PostMortems and a Just Culture." Code as Craft (Etsy engineering blog).

Cook, R. I. (1998). "How Complex Systems Fail." Cognitive Technologies Laboratory, University of Chicago.

Dekker, S. (2002). _The Field Guide to Understanding 'Human Error'._ Ashgate.

Davidovič, Š. (2021). "Incident Metrics in SRE: Critically Evaluating MTTR and Friends." O'Reilly / Google.

Nash, C. (2021). "The VOID Report." Verica Open Incident Database. https://www.thevoid.community/

PagerDuty. "Incident Response Documentation." https://response.pagerduty.com/

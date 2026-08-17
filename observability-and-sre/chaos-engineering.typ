#import "../template.typ": xref

= Chaos Engineering

Chaos engineering is the discipline of experimenting on a distributed system in order to build confidence in its ability to withstand turbulent conditions in production. The phrase comes from Netflix, whose Chaos Monkey (2011) randomly terminated production instances to force engineers to build termination-tolerant services. The practice has since matured from "break things randomly" into controlled, hypothesis-driven experimentation with explicit blast-radius limits — closer to clinical trials than to vandalism. This chapter covers the principles, the fault-injection toolbox, the major platforms, and game-day practice, then situates chaos experiments relative to testing and formal methods.

*See also:* #xref("observability-and-sre", "incident-management", label: "Incident Management"), #xref("observability-and-sre", "slo-engineering", label: "SLO Engineering"), #xref("observability-and-sre", "alerting", label: "Alerting"), #xref("observability-and-sre", "the-three-pillars-and-beyond", label: "The Three Pillars and Beyond")

== Principles

The canonical formulation is the "Principles of Chaos Engineering" manifesto (principlesofchaos.org, 2015), written by the Netflix chaos team and elaborated in Basiri et al., "Chaos Engineering" (IEEE Software, 2016). Four steps:

1. *Define steady state* as a measurable output of the system that indicates normal behavior. Netflix used SPS (stream starts per second), a business metric, rather than CPU or queue depth — internal metrics can look fine while users suffer, and vice versa. SLO-grade SLIs (see _SLO Engineering_) are natural steady-state signals.
2. *Hypothesize* that the steady state continues in both the control group and the experimental group. The hypothesis must be falsifiable and written down before the experiment: "if we terminate one Kafka broker, consumer lag stays below 10 s and checkout SLI stays above 99.9 %."
3. *Inject real-world events:* instance death, network partition, latency, dependency failure, resource exhaustion, region loss.
4. *Disprove the hypothesis* by observing a statistically meaningful divergence between control and experiment. A disproved hypothesis is a found weakness; a confirmed one is earned confidence.

=== Blast Radius and Abort Conditions

The operational core of safe chaos is *minimizing blast radius*: start with one instance in staging, then one instance in production, then a small percentage of traffic, expanding only after each level passes. Every experiment carries:

- *Abort conditions:* automated halt when steady-state metrics breach thresholds. Netflix's ChAP (Chaos Automation Platform, 2017) routes a small slice of traffic (around 1 %) to a control cluster and an identically sized experiment cluster, comparing SLIs in real time and aborting automatically on divergence.
- *A kill switch:* a single, well-known, fast way to stop all injected faults.
- *Scheduling discipline:* business hours, with the owning team present — never during an active incident or a change freeze.

Running in production is the stated ideal (only production has real traffic patterns, data shapes, and emergent interactions), but the principle is conditional: you earn production chaos by surviving staging chaos, and some experiments (e.g., data-corruption injection) may never be appropriate in production.

== The Fault Injection Toolbox

Faults are injected at several layers, with different fidelity and risk:

#table(
  columns: (auto, auto, auto),
  align: left,
  table.header[*Fault class*][*Examples*][*Typical mechanism*],
  [Compute], [Instance/pod kill, process crash, VM reboot], [Cloud API, `kill -9`, pod delete],
  [Network], [Latency, packet loss, partition, DNS failure], [`tc netem`, iptables, eBPF, service mesh],
  [Resource], [CPU burn, memory pressure, disk fill, file-descriptor exhaustion], [stress-ng, cgroup limits],
  [Dependency], [Error/latency injection on RPC to a downstream], [Sidecar/mesh fault filters (Envoy), client-library hooks],
  [State and time], [Clock skew, certificate expiry, corrupted cache], [Time namespaces, targeted writes],
  [Regional], [AZ evacuation, region failover], [Traffic steering, DNS, cloud FIS actions],
)

Two mechanisms deserve emphasis. Application-level injection (Netflix's FIT, "Failure Injection Testing," 2014) tags requests with failure instructions in headers, so faults follow a specific request through the call graph — enabling per-customer or per-request blast radius far tighter than killing machines. And service-mesh injection (Envoy/Istio fault filters) lets you return errors or add latency for a percentage of calls between two named services with a configuration change, no code modification.

Latency injection is generally more revealing than outright failure: many systems handle a dead dependency (fast failure, circuit breaker opens) better than a slow one (threads pile up, pools exhaust, the failure cascades). The classic finding of dependency chaos is a "soft" dependency that was assumed optional but whose slowness takes down the critical path.

== Platforms

- *Chaos Monkey (Netflix, 2011; open-sourced 2012)* terminates instances during business hours; part of the broader Simian Army (Latency Monkey, Conformity Monkey, and Chaos Kong, which simulated whole-region failure). Its enduring contribution is cultural: failure became routine, so termination-tolerance became table stakes.
- *Gremlin (2016)* — commercial "failure as a service" founded by Netflix/Amazon chaos alumni, with a library of attacks (resource, network, state), scenario orchestration, automatic rollback, and agent-based targeting.
- *LitmusChaos (2018; CNCF incubating)* — Kubernetes-native: experiments are CRDs, a ChaosHub provides reusable experiment definitions, and probes evaluate steady-state hypotheses. Chaos Mesh (PingCAP, 2020; also CNCF) is the other major Kubernetes option, with strong network and I/O fault support and time-skew injection used to test TiDB.
- *Cloud-provider FIS:* AWS Fault Injection Simulator (2021) and Azure Chaos Studio inject faults at the cloud-control-plane level (instance, AZ, API throttling) with IAM-scoped guardrails and stop conditions wired to CloudWatch alarms.
- *Toxiproxy (Shopify, 2014)* — a TCP proxy for deterministic network-fault simulation in integration tests, bridging chaos and conventional testing.

== Game Days, DiRT, and the Wheel of Misfortune

A *game day* is a scheduled exercise in which a team injects a planned failure and practices the human response end to end: detection, paging, diagnosis, mitigation, communication. The fault is real (or realistically simulated); the schedule is known; the learning targets are both the system and the responders.

Google's *DiRT* (Disaster Recovery Testing, annual since 2006; Krishnan, "Weathering the Unexpected," ACM Queue, 2012) is the large-scale ancestor: company-wide exercises that have simulated earthquakes disconnecting headquarters, datacenter evacuations, and the loss of key personnel. DiRT's signature findings are rarely about servers — they are about process: the emergency bridge number nobody could find, the runbook behind the SSO that was down, the single person who knew the failover procedure being "unavailable" by exercise rule. Amazon runs comparable region-failover exercises; many organizations now require annual DR tests for compliance (and chaos tooling makes those tests honest rather than paper exercises).

The *Wheel of Misfortune* (SRE book, chapter 28) is the tabletop variant: a role-playing exercise in which a game master replays a past incident and an on-call engineer (often a new team member) talks through diagnosis and response, with the team as audience. It is dirt-cheap, zero-risk training for the rarest skill — calm, structured reasoning under pacing pressure — and doubles as a test of whether documentation suffices for someone who was not there.

== Verifying Recovery Procedures

A recovery procedure that has never been executed is a hypothesis, not a capability. Chaos practice treats backups, failovers, and runbooks as code to be tested:

- *Backups:* the test is restore, not backup. Periodically restore to a scratch environment and verify integrity and restore _time_ — a 14-hour restore against a 1-hour RTO is a failed test. The GitLab database incident of January 2017 is the canonical cautionary tale: five backup/replication mechanisms, none of which had been verified, none of which worked when needed.
- *Failover:* exercise database and region failover on schedule, measuring actual RTO/RPO against targets. Untested failover paths accumulate drift: stale credentials, security-group changes, capacity that no longer fits.
- *Runbooks:* execute them during game days exactly as written, by someone who did not write them. Every ambiguity found in an exercise is an ambiguity removed from a 3 a.m. incident (see _Incident Management_).
- *Alerting and paging:* fault injection is also a test of detection — if the experiment degrades the steady state and nothing pages, you have found an alerting gap as surely as a resilience gap (see _Alerting_).

== Relationship to Testing and Formal Methods

Chaos engineering complements, rather than replaces, other verification techniques. The distinction is what each explores:

- *Unit/integration tests* verify known requirements against known inputs: "given X, the system does Y." Chaos experiments probe unknown behavior under conditions too complex to enumerate — emergent interactions among services, retries, timeouts, and real traffic. Basiri et al. frame the difference as testing makes assertions; experimentation generates new knowledge.
- *Fault-injection testing in CI* (Toxiproxy, Jepsen-style harnesses) is deterministic chaos: the same partitions and crash schedules replayed on every commit. Kyle Kingsbury's *Jepsen* project (2013–present) has used systematic partition and clock-skew injection to find consistency violations in dozens of production databases (etcd, MongoDB, Cassandra, PostgreSQL commit protocols), demonstrating how much falsifiable fault injection can find that vendors' own tests did not.
- *Formal methods* verify the design, not the implementation. Amazon's use of TLA+ (Newcombe et al., "How Amazon Web Services Uses Formal Methods," CACM 2015) found subtle bugs in DynamoDB and S3 protocols that testing could not plausibly reach (one required a 35-step interleaving). But a verified specification says nothing about whether the deployed code, configuration, and infrastructure match it. FoundationDB's deterministic simulation testing (Zhou et al., SIGMOD 2021) sits between: the real implementation runs in a simulated world where the scheduler, network, and disks are controlled and faults are injected exhaustively across millions of seeds.

A reasonable synthesis: formal methods for protocol design, deterministic simulation and CI fault injection for implementation logic, and chaos experiments in production for the full sociotechnical system — including the humans, dashboards, and pagers that the other techniques cannot model.

== Further Reading

Basiri, A., Behnam, N., de Rooij, R., Hochstein, L., Kosewski, L., Reynolds, J., Rosenthal, C. (2016). "Chaos Engineering." IEEE Software 33(3).

Rosenthal, C., Jones, N. (2020). _Chaos Engineering: System Resiliency in Practice._ O'Reilly.

Krishnan, K. (2012). "Weathering the Unexpected." ACM Queue 10(9). The DiRT program at Google.

Basiri, A. et al. (2019). "Automating Chaos Experiments in Production." ICSE-SEIP. The ChAP platform.

Newcombe, C. et al. (2015). "How Amazon Web Services Uses Formal Methods." Communications of the ACM 58(4).

Zhou, J. et al. (2021). "FoundationDB: A Distributed Unbundled Transactional Key Value Store." SIGMOD. Deterministic simulation testing.

Beyer, B. et al. (2016). _Site Reliability Engineering._ O'Reilly. Chapter 28 (Wheel of Misfortune).

Principles of Chaos Engineering. https://principlesofchaos.org/

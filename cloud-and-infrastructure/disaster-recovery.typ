#import "../template.typ": xref

= Multi-Region Architecture and Disaster Recovery <disaster-recovery>

Zones protect against the loss of a building. Regions protect against the loss of a metropolitan area, a regional control plane, or a provider's bad deployment. Separate accounts and immutable backups protect against the most common disaster of all: someone, or something, deleting or corrupting the data with valid credentials. Each layer costs more than the last, and the right design follows from two numbers the business must state, how much data it can lose and how long it can be down, rather than from a default of "multi-region active-active". This chapter covers recovery objectives, the classic DR strategies, replication and its consistency trade-offs, failover mechanics, hidden dependencies, and testing.

*See also:* #xref("cloud-and-infrastructure", "iaas-fundamentals", label: "IaaS Fundamentals") (regions, zones, and latency), #xref("cloud-and-infrastructure", "autoscaling", label: "Autoscaling and Capacity Management") (static stability), #xref("cloud-and-infrastructure", "cloud-storage", label: "Cloud Storage Services") (backups and object lock), #xref("distributed-systems", "consensus-deep-dive", label: "Consensus Deep Dive"), #xref("database", "consensus-and-replication", label: "Consensus and Replication"), #xref("observability-and-sre", "incident-management", label: "Incident Management").

== Failure Scopes

#table(
  columns: 3,
  [*Scope*], [*Examples*], [*Primary defence*],
  [Host or instance], [Hardware fault, kernel panic], [Redundant instances, autoscaling replacement],
  [Zone], [Power or cooling loss, network partition of one facility], [Multi-zone deployment with static stability],
  [Region], [Regional control-plane or service outage, bad regional deployment, natural disaster], [Second region],
  [Provider-global], [Global service outage (identity, DNS, CDN), bad global configuration push], [Avoid global control-plane dependencies; second provider in rare cases],
  [Account or credential], [Leaked administrator credential, ransomware, insider], [Separate backup account, immutable storage],
  [Logical], [Bad migration, application bug writing wrong data, accidental delete], [Point-in-time recovery, versioning, delayed replicas],
)

Replication defends against the first four rows and does nothing for the last two: it copies the corruption or the deletion to every replica within seconds. A DR plan that is only replication is a plan for hardware failure.

== Recovery Objectives

- *Recovery point objective (RPO):* the maximum acceptable amount of data loss, measured as time. RPO 5 minutes means that after recovery, at most the last 5 minutes of writes may be missing.
- *Recovery time objective (RTO):* the maximum acceptable time from the disaster until service is restored.

Both are business decisions, set per workload, and they drive cost steeply. A payments ledger may need RPO near zero and RTO of minutes; an analytics pipeline may tolerate a day of each. Classifying workloads into a few tiers with stated objectives, then designing each tier once, is far more effective than debating each service.

Availability and DR objectives are linked but different. A service can have 99.99 percent availability from good zonal redundancy and still have no regional DR at all; the rare regional event then consumes years of error budget in one go.

== DR Strategies

#table(
  columns: 4,
  [*Strategy*], [*What runs in the recovery region*], [*Typical RPO / RTO*], [*Relative cost*],
  [Backup and restore], [Nothing; backups are copied there], [Hours / hours to a day], [Lowest],
  [Pilot light], [Replicated data stores; compute defined in code but off or minimal], [Minutes / tens of minutes to hours], [Low],
  [Warm standby], [Full stack at reduced capacity, receiving replicated data], [Seconds to minutes / minutes], [Medium],
  [Multi-site active-active], [Full stack serving live traffic in every region], [Near zero / near zero for reads, depends on write design], [Highest],
)

The strategies differ mostly in what must happen at recovery time. Backup and restore must provision everything and load data, which is slow and dependent on control-plane capacity during a large event (when every other customer is doing the same). Warm standby only scales up and redirects traffic. Active-active redirects traffic that is already partly there.

Infrastructure as code is a precondition for every strategy above backup and restore, and in practice for that too. A recovery region built by hand drifts from production within weeks; a recovery region built from the same templates, with region as a parameter, can be verified continuously.

== Data Replication Across Regions

Compute is easy to duplicate; data is the hard part. The speed of light sets the trade-off: a round trip between distant regions costs 60 to 200 ms, so every option chooses between write latency, RPO, and conflict handling.

#table(
  columns: 3,
  [*Approach*], [*Behaviour*], [*Examples*],
  [Asynchronous primary-replica], [Writes go to one region and replicate with lag; failover loses unreplicated writes (RPO is the lag)], [Aurora Global Database, cross-region read replicas, object replication],
  [Synchronous quorum across regions], [Commit waits for a majority of regions; RPO zero, write latency at least one inter-region round trip], [Spanner and CockroachDB multi-region configurations],
  [Multi-writer with conflict resolution], [Every region accepts writes; concurrent updates resolved by last-writer-wins, CRDTs, or application logic], [DynamoDB global tables, Cosmos DB multi-region writes, Cassandra],
  [Event log replication], [Replicate an ordered log; consumers rebuild state in each region], [Kafka MirrorMaker 2, Cluster Linking],
)

Asynchronous replication is the common choice, and its RPO is the replication lag *at the moment of failure*, not the average. Lag spikes under exactly the conditions that precede many regional failures (network congestion, overload), so measure and alert on its high percentiles. When the old primary returns, the writes it accepted but never replicated must be either discarded or reconciled; having a defined procedure for that is part of the design.

Multi-writer systems move the problem from failover to every write. Last-writer-wins silently drops one of two concurrent updates, which is acceptable for a user's display name and unacceptable for an account balance. A common compromise is *home-region* or *single-writer per key*: each customer's data has one writing region at a time, other regions serve reads, and a failover moves the home. This keeps strong per-key semantics with most of active-active's latency benefits for reads. DynamoDB has since added a multi-region strongly consistent mode for global tables, at the cost of cross-region write latency.

== Availability Arithmetic

With independent components, availabilities multiply in series and failure probabilities multiply in parallel:

$ A_"series" = product_i A_i, quad A_"parallel" = 1 - product_i (1 - A_i) $

```cpp
#include <cmath>
#include <vector>

double series(const std::vector<double>& a) {
    double r = 1.0;
    for (double x : a) r *= x;
    return r;
}

double parallel(const std::vector<double>& a) {
    double fail = 1.0;
    for (double x : a) fail *= 1.0 - x;
    return 1.0 - fail;
}

// Parallel copies that also share a common-mode failure with availability a_common.
double parallel_with_common_mode(const std::vector<double>& a, double a_common) {
    return a_common * parallel(a);
}

double downtime_minutes_per_year(double availability) {
    return (1.0 - availability) * 365.25 * 24 * 60;
}
```

Two regions at 99.9 percent each give 99.9999 percent in theory. Reality is worse for three reasons. Failures are correlated: both regions run the same software, receive the same configuration pushes, and depend on the same global services, so the common-mode term dominates. Failover is not instantaneous, so every regional event costs at least the detection and switch time. And the failover mechanism is itself a component in series. A service with a 99.99 percent common-mode dependency cannot exceed 99.99 percent however many regions it runs in.

== Failover Mechanics

=== Steering Traffic

#table(
  columns: 3,
  [*Mechanism*], [*Switch time*], [*Caveats*],
  [DNS with health checks (Route 53, Cloud DNS, Traffic Manager)], [TTL plus resolver behaviour: typically 1-5 minutes], [Some resolvers and clients ignore TTLs; long-lived connections do not re-resolve],
  [Anycast global load balancer (GCP global LB, Front Door, Global Accelerator)], [Seconds], [Ties the edge to one provider's global infrastructure],
  [Client-side region lists], [Immediate, per request], [Requires control of the clients (mobile apps, SDKs, internal services)],
)

=== Do Not Fail Over Through the Control Plane

A failover that must create resources, change DNS records through a management API, or update load balancer configuration depends on control planes that are degraded during large events. Recovery should be a *data-plane* action against things that already exist: flipping a health-check-driven routing control, changing a weight that the data plane reads, or simply letting health checks fail. AWS Route 53 Application Recovery Controller is built around this idea: its routing controls are served by a highly available data plane spread across five regions, so changing them does not depend on any single region's control plane.

=== Who Decides

Automatic failover reacts in seconds but can be triggered by a partial or false signal: a monitoring outage looks like a service outage, and a flapping link causes repeated failovers that do more damage than staying put. For stateful systems, the danger is *split brain*, both regions believing they are primary and accepting writes. Practical designs automate detection and preparation, then require a human decision (or a quorum of independent health signals from outside both regions) for the final promotion of a database, and fence the old primary so it cannot accept writes even if it comes back.

== Hidden Global Dependencies

A second region only helps if the recovery path does not depend on the failed one. Dependencies that commonly hide in a single region:

- Global services with a home region. On AWS, the control planes for IAM, Route 53, and some other global services run in us-east-1. Their data planes (authenticating requests, answering DNS queries) are distributed and keep working, but changing roles or DNS records during a us-east-1 event may not.
- Build, deploy, and artifact systems (CI, container registries, package mirrors) in one region, so new code cannot ship to the healthy region.
- Secrets and keys held only in a regional secret store or KMS. Multi-region keys and replicated secrets fix this, but must be set up in advance.
- Identity providers, observability backends, and paging systems in the same region as the workload, so operators lose visibility exactly when they need it.
- Third-party SaaS dependencies that themselves run in one region.

Large outages have repeatedly shown these effects. The February 2017 S3 outage in us-east-1, triggered by a maintenance command that removed more servers than intended, also broke the AWS status dashboard, which depended on S3. The December 2021 us-east-1 event, caused by congestion on AWS's internal network, impaired control-plane APIs and monitoring while many running workloads kept serving. In October 2025, a race in the automation that manages DynamoDB's DNS records in us-east-1 left the regional endpoint without addresses, and services across AWS that depended on DynamoDB there failed with it.

== Cells and Blast Radius

A region is a large failure domain to depend on. *Cell-based architecture* partitions a service into many independent, identical copies (cells), each serving a subset of customers, with a thin routing layer in front. A bad deployment, poison request, or overload then affects one cell's customers, and deployments roll out cell by cell.

*Shuffle sharding* goes further: each customer is assigned a random subset of $k$ workers out of $n$. A misbehaving customer can exhaust only its own $k$ workers, and another customer is fully affected only if its subset is identical, which has probability $1 \/ binom(n, k)$. With $n = 8$ and $k = 2$ there are 28 possible shards, so a noisy customer fully overlaps with about 4 percent of others instead of all of them.

== Backups as the Last Line

Backups must survive every failure above them in the scope table, including the loss of the production account:

- Copy backups to a separate account, in another region, owned by a small security team, with write-only access from production.
- Use immutable storage (object lock in compliance mode, vault lock) so that even administrators cannot delete backups before their retention ends.
- Keep point-in-time recovery for databases, with a retention long enough to cover the time it takes to notice logical corruption. Weeks, not days.
- Restore regularly and automatically. An untested backup is a hypothesis.

== Testing

DR that has not been exercised does not work, for mundane reasons: an expired certificate in the standby region, a quota never raised, a hard-coded endpoint, a runbook that refers to a dashboard that was renamed. Testing escalates in realism:

+ Tabletop exercises that walk through the runbook.
+ Restore tests: rebuild databases from backups in an isolated environment and verify the data.
+ Partial failover: shift a small share of traffic to the recovery region continuously, so the path is always warm.
+ Full region evacuation drills during business hours, with the service owners present.
+ Failback: returning to the original region is a second failover and must be tested too.

Measure the achieved RPO and RTO in each test and compare them with the objectives. The gap is the real DR status.

== Cost

Multi-region designs pay for duplicated capacity (from a small pilot light to a full second copy), cross-region data transfer for replication (every write crosses at least once), and operational complexity, which is the largest and least visible cost. A rule that holds well: choose the cheapest strategy that meets the stated objectives, then spend on testing it rather than on a more elaborate strategy that is never tested.

== Pitfalls

- Treating replication as backup.
- RPO computed from average replication lag instead of lag under stress.
- Failover runbooks that call control-plane APIs of the failed region.
- A standby region that has never served production traffic.
- Global dependencies (CI, secrets, identity, monitoring) left in the primary region.
- Automatic database promotion without fencing, causing split brain.
- Backups in the same account as production, deletable by the same credentials.
- No plan for the writes stranded in the old primary after failover.

== Further Reading

Kleppmann, M. (2017). _Designing Data-Intensive Applications_. O'Reilly.

Corbett, J. C. et al. (2012). "Spanner: Google's Globally-Distributed Database." OSDI.

DeCandia, G. et al. (2007). "Dynamo: Amazon's Highly Available Key-value Store." SOSP.

Shapiro, M. et al. (2011). "Conflict-Free Replicated Data Types." SSS.

Brooker, M., Chen, T. and Ping, F. (2020). "Millions of Tiny Databases." NSDI.

Ford, D. et al. (2010). "Availability in Globally Distributed Storage Systems." OSDI.

Beyer, B. et al. (2016). _Site Reliability Engineering: How Google Runs Production Systems_. O'Reilly.

Huang, P. et al. (2017). "Gray Failure: The Achilles' Heel of Cloud-Scale Systems." HotOS.

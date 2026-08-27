#import "../template.typ": xref

= Monoliths and Microservices <monoliths-and-microservices>

"Monolith versus microservices" is the most over-litigated debate in software architecture, and most of the heat comes from treating it as binary. The real spectrum runs from a tangled monolith, through a modular monolith, to coarse-grained services, to fine-grained microservices, and systems move along it in both directions. This chapter covers modularity within a single deployable, decomposition strategies, the antipatterns that make distribution fail, and the case studies, including the celebrated reversals, that anchor the debate in evidence.

*See also:* #xref("software-architecture", "domain-driven-design", label: "Domain-Driven Design") (bounded contexts as the unit of decomposition), #xref("software-architecture", "distributed-data-patterns", label: "Distributed Data Patterns") (what happens to the database when you split), #xref("software-architecture", "resilience-patterns", label: "Resilience Patterns") (the failure modes distribution buys you).

== Definitions, Precisely

A *monolith* is a single deployment unit: one process (or homogeneous replica set) containing all application code, typically backed by one database. A *microservice* (Lewis & Fowler, 2014) is an independently deployable service, organised around a business capability, owning its own data, communicating over the network, and operated by a small team ("two-pizza team" in Amazon's phrasing, from Bezos's early-2000s API mandate).

The key word is *independently deployable*. If deploying service A requires coordinating with services B and C, you do not have microservices; you have a distributed monolith.

== The Modular Monolith

A *modular monolith* is one deployable composed of well-bounded modules, ideally one per bounded context, with enforced internal boundaries: modules expose a small public API and communicate through it (in-process calls or in-process events), never by reaching into each other's tables or internals.

Enforcement is what separates a modular monolith from an aspirational package diagram:
- Compile-time visibility: Java modules/ArchUnit rules, Go internal packages, .NET internal types with architecture tests.
- Schema separation: one database server, but separate schemas per module with no cross-schema foreign keys or joins.
- Frameworks: Spring Modulith (2022) verifies module dependency rules and tests modules in isolation.

Shopify is the flagship example: a Rails monolith on the order of millions of lines of code, restructured from 2017 onward into "components" with enforced boundaries (their `packwerk` tool, open-sourced 2020, checks dependency and privacy violations in CI). Shopify has repeatedly and publicly defended the choice: the monolith handles flash-sale peaks in the tens of millions of requests per minute, scaled by *pod sharding*, running many full copies of the monolith, each serving a subset of shops, rather than by service decomposition.

The pragmatic guidance, echoed by Fowler ("MonolithFirst", 2015) and Newman: a modular monolith gives you most of microservices' design benefits (clear ownership, bounded contexts) with none of the distributed-systems tax, and clean module boundaries are the best possible starting point if you later extract services.

== Why Microservices: The Honest List

Benefits that genuinely require independent deployability:
- *Independent deploy cadence*: 50 teams releasing without a shared release train. This is the dominant driver at scale; DORA research correlates loosely-coupled architecture with elite delivery performance.
- *Independent scaling*: scale the image-processing service to 200 instances while checkout runs 10; also heterogeneous hardware (GPU vs. memory-optimised).
- *Fault isolation*: a memory leak in recommendations cannot OOM checkout (only if the integration is asynchronous or properly bulkheaded, see #xref("software-architecture", "resilience-patterns", label: "Resilience Patterns")).
- *Technology heterogeneity*: a JVM service next to a Python ML service, within reason.

Costs, paid immediately and forever: network latency and partial failure on every internal call, distributed transactions become sagas, debugging requires distributed tracing, every service needs its own CI/CD, on-call, and security surface, and local development needs orchestration. Newman's rule: do not adopt microservices without a problem on the benefits list that you actually have.

== Conway's Law

"Any organization that designs a system... will produce a design whose structure is a copy of the organization's communication structure" (Melvin Conway, 1968). Empirically robust, MacCormack et al. (Harvard, 2008/2012) found product structure mirrors organisational structure across codebases, and central to modern architecture practice:

- The *Inverse Conway Maneuver* (Thoughtworks, popularised ~2015): restructure teams to match the architecture you want, and the architecture will follow. Team boundaries are architectural decisions.
- _Team Topologies_ (Skelton & Pais, 2019) operationalises this: *stream-aligned* teams own slices of the domain end to end; platform, enabling, and complicated-subsystem teams reduce their cognitive load. Service boundaries should track team cognitive-load limits, not just domain seams.
- Corollary: a microservice owned by three teams, or one team owning fifteen chatty services, both fight Conway's law and lose.

== Decomposition Strategies

When extracting services from a monolith, the unit of decomposition should be a *bounded context* or business capability, never a technical layer (an "entity service" or shared "data service" recreates coupling at network cost).

Practical strategies:
- *Strangler fig* (Fowler, 2004): place a routing facade in front of the monolith; build or extract capabilities behind it; route traffic over incrementally; retire monolith code as it is starved (detailed in #xref("software-architecture", "evolutionary-architecture", label: "Evolutionary Architecture")).
- *Extract by value and risk*: start with a module that is frequently changed (deploy-independence payoff) but not the most critical (limit blast radius). Newman suggests "easy and valuable" first to build operational muscle.
- *Branch by abstraction* for in-place seams; *change data capture* or an anticorruption layer where the monolith's data must be mirrored during transition.
- *Database decomposition last and hardest*: split tables before splitting the schema; break joins and foreign keys deliberately (Newman, _Monolith to Microservices_, 2019, devotes half the book to this).

== Service Granularity

Granularity is a trade-off, not a target size. "Micro" is a misnomer; nothing useful is measured in lines of code. Richards & Ford frame it as opposing forces:

- *Granularity disintegrators* (reasons to split): divergent scalability or fault-tolerance needs, different security requirements (PCI scope isolation), different rates of change, team ownership conflicts.
- *Granularity integrators* (reasons to merge): the two candidate services share a transaction (workflows needing ACID across them), chatty synchronous communication between them, heavy shared data.

A reliable smell: if two services are always deployed together, always change together, or call each other on most requests, they are one service wearing two pods. The right starting grain for most teams is closer to "service per bounded context" (a handful to a few dozen services) than "service per entity" (hundreds).

== The Distributed Monolith Antipattern

A *distributed monolith* has the operational costs of microservices and the coupling of a monolith. Diagnostic symptoms:

- Lockstep deployment: releases require coordinated multi-service deploys or a shared release calendar.
- Shared database: multiple services reading and writing the same tables, so schema changes ripple everywhere.
- Synchronous call chains: request fan-out depth of 4–5+ services, so availability multiplies down ($0.999^5 approx 0.995$) and tail latency adds up.
- Shared internal libraries containing domain logic, forcing simultaneous upgrades.
- Entity services and anaemic "data access services" that every other service must call.

The usual cause is decomposing by technical layer or by org-chart noun without giving services their own data, i.e., skipping the strategic-design step.

== Case Studies

=== Amazon Prime Video (2023)

In March 2023, Prime Video's audio/video quality-monitoring team published a post explaining how they reduced infrastructure cost by about 90% by moving the service *from* a distributed, serverless design (AWS Step Functions orchestrating Lambda functions, with intermediate video frames passed through S3) *to* a monolithic process: all components compiled into one task running on ECS/EC2, scaled by running copies.

The internet declared microservices dead; the sober reading is narrower and more useful. The workload was a high-volume media pipeline where the dominant costs were Step Functions state transitions and S3 round-trips between steps, orchestration and data transfer overhead, not compute. Merging steps into one process eliminated the per-step tax. The lesson is about *granularity and workload shape*: serverless step-per-function was the wrong grain for a tight data-flow pipeline. Notably, even DHH and Kelsey Hightower converged on "right-size the architecture", and the service remained one team's bounded component within Amazon's wider service architecture.

=== Shopify

As above: a deliberately monolithic core, made modular with `packwerk`-enforced component boundaries, scaled horizontally by pod sharding (each pod a full stack serving a subset of merchants), with a small number of satellite services where isolation genuinely pays (e.g. storefront rendering). Demonstrates that modularity and extreme scale do not require distribution.

=== Segment

Segment (customer-data platform) published "Goodbye Microservices" (Alexandra Noonan, 2018). To isolate failures among ~140 destination integrations, they had split a monolithic worker into one service and one queue per destination, ~140 services sharing logic through dozens of versioned shared libraries. Operational load exploded: library updates required touching every service, autoscaling 140 differently-shaped services was unmanageable, and on-call drowned. They consolidated back into a single monolithic service ("Centrifuge") with per-destination in-process isolation. Diagnosis in our vocabulary: service-per-integration was the wrong granularity, and shared libraries made it a distributed monolith.

=== Counterpoint: Netflix, Amazon retail, Uber

The pattern also works at scale in its favour: Amazon's 2002 API mandate and Netflix's 2008–2016 migration (hundreds of services after their 2008 database-corruption outage) created the playbook; Uber, after experiencing fine-grained sprawl (thousands of services), consolidated toward larger "domain-oriented" services (DOMA, 2020). Across all the case studies the constant is *re-grained when the pain signal changed*, which is the actual lesson.

== Decision Summary

#table(
  columns: 3,
  [*Signal*], [*Points toward*], [*Why*],
  [Small org, < ~3 teams], [Modular monolith], [Distribution tax exceeds autonomy benefit],
  [Deploy contention across many teams], [Microservices], [Independent deployability is the core benefit],
  [Divergent scaling/fault/security needs], [Extract those services], [Targeted disintegrators],
  [Shared transactions, chatty calls], [Merge/keep together], [Granularity integrators],
  [Tight data-flow pipeline], [Coarse process], [Prime Video lesson: orchestration overhead],
  [Unclear domain boundaries], [Monolith first], [Boundaries are cheaper to fix in-process],
)

== Further Reading

- Newman, S. (2021). _Building Microservices_, 2nd ed. O'Reilly.
- Newman, S. (2019). _Monolith to Microservices_. O'Reilly.
- Skelton, M., & Pais, M. (2019). _Team Topologies_. IT Revolution.
- Fowler, M. (2015). MonolithFirst. martinfowler.com.
- Kolny, M. (2023). Scaling up the Prime Video audio/video monitoring service and reducing costs by 90%. Prime Video Tech Blog.
- Noonan, A. (2018). Goodbye microservices: from 100s of problem children to 1 superstar. Segment Engineering Blog.
- Conway, M. (1968). How do committees invent? _Datamation_, 14(4).

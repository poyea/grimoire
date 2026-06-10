= Evolutionary Architecture

Architecture is traditionally framed as "the decisions that are hard to change", which quietly assumes the goal is to get them right up front. Evolutionary architecture inverts the assumption: since requirements, technology, and the organisation *will* change in ways no one can predict, the architecture's primary job is to support *guided, incremental change across multiple dimensions* (Ford, Parsons & Kua, _Building Evolutionary Architectures_, 2017). The guidance comes from *fitness functions*, executable checks that the qualities you care about still hold, and the increments come from techniques like strangler fig, branch by abstraction, and expand–contract. This chapter covers the mechanics.

*See also:* _Architecture Evaluation_ (assessing qualities at a point in time; fitness functions make that assessment continuous), _Monoliths and Microservices_ (decomposition as a common evolution), _Domain-Driven Design_ (bounded contexts as the natural quantum of change).

== Fitness Functions

Borrowed from evolutionary computing, an *architectural fitness function* is any mechanism that provides an objective assessment of some architectural characteristic. The shift it represents: quality attributes stop being adjectives in a slide deck and become tests that fail builds.

Useful classification axes (Ford et al.):
- *Atomic vs. holistic*: an atomic function checks one characteristic in isolation (a dependency rule); a holistic one exercises several in combination (a chaos experiment validating resilience plus data integrity).
- *Triggered vs. continual*: triggered functions run in CI/CD; continual ones run in production permanently (a synthetic transaction asserting end-to-end latency, an availability monitor against an $"SLO"$).
- *Static vs. dynamic*: static functions have fixed pass/fail thresholds; dynamic ones adjust against a baseline (performance must not regress more than 5% from the rolling mean).

Concrete examples in production use:
- *Dependency and layering rules*: ArchUnit (Java), NetArchTest (.NET), eslint import rules and dependency-cruiser (JS/TS) assert "domain packages must not import infrastructure", failing the build on violation. This makes the architecture diagram enforceable.
- *Cycle detection*: forbid cyclic dependencies between components, the structural property that most reliably predicts unmaintainability.
- *Performance budgets*: Lighthouse CI bundles-size and page-weight budgets; load-test gates on p99 latency.
- *Operational fitness*: Netflix's Chaos Monkey is a fitness function for instance-failure tolerance; their "conformity monkey" checked instances against architectural standards.
- *Security and compliance*: dependency-vulnerability scanning, policy-as-code (OPA) checks that all buckets are encrypted.

Governance follows: instead of an architecture review board approving designs annually, the architects encode constraints as fitness functions and let the pipeline govern every commit, faster feedback, no drift between the reviewed design and the shipped system.

== Incremental Change: The Deployment Pipeline as Enabler

Evolution presupposes that small changes are cheap and safe to ship. The supporting machinery is the continuous-delivery toolkit (Humble & Farley, 2010): trunk-based development, a deployment pipeline running the fitness functions, and *progressive delivery*, canary releases, feature flags, and automated rollback on metric regression. Without this substrate, "evolutionary architecture" is aspiration; with it, the architecture can change in production weekly without ceremony. DORA's research (Forsgren, Humble & Kim, _Accelerate_, 2018) supplies the evidence: loosely coupled architectures plus deployment automation correlate with both speed and stability, against the folk belief that they trade off.

== Strangler Fig

The strangler fig (Fowler, 2004; named for the fig that grows around a host tree until the host dies and the fig stands alone) is the canonical pattern for incrementally replacing a legacy system:

+ Put an interception layer in front of the legacy system, an HTTP proxy or gateway, a message router, or DNS/edge routing, so you control where each request goes.
+ Pick a capability (a vertical slice, ideally a bounded context), build or extract it as a new component, and route that capability's traffic to it. Run *in parallel* with shadowing or dark reads where the risk warrants comparing outputs.
+ Repeat, capability by capability. The legacy system shrinks ("is strangled") until it can be decommissioned.

Why it beats big-bang rewrites: value ships continuously, risk is bounded per slice, the legacy system remains the safety net (routing can fall back), and the effort can be paused at any point with everything still working, none of which is true of a rewrite, whose track record is poor enough that Fowler offers the pattern explicitly as the alternative. The hard parts are data (the new component needs the legacy data: use replication, CDC, or an anticorruption layer during transition) and the temptation to stop at 80%, leaving two systems forever.

== Branch by Abstraction and Expand–Contract

For changes *inside* a codebase that are too large for one commit but must not block trunk-based development:

*Branch by abstraction* (Hammant, 2007; Fowler, 2014):
+ Introduce an abstraction over the subsystem to be replaced (an interface in front of the old persistence layer).
+ Migrate callers to the abstraction, incrementally, on trunk.
+ Build the new implementation behind the abstraction; toggle between implementations with a feature flag; verify (possibly running both and comparing, "scientist"-style, after GitHub's Scientist library).
+ Remove the old implementation and, if it has no further use, the abstraction.
The codebase compiles and ships throughout, the point, in contrast to a long-lived VCS branch that rots and merges catastrophically.

*Expand–contract* (parallel change) applies the same idea to interfaces and schemas:
+ *Expand*: add the new column/field/endpoint alongside the old; write to both (or backfill).
+ *Migrate*: move readers to the new shape, at their own pace.
+ *Contract*: when telemetry shows no remaining readers of the old shape, remove it.
This is the only safe way to change a database schema or a published API under zero-downtime deployment, where old and new code versions run simultaneously during every rollout. The corollary discipline: every schema migration must be compatible with the *previous* application version (N−1 compatibility), and destructive steps ship in a later release than the code that stops depending on them.

== Coupling, Quanta, and Reuse

Evolvability is mostly a function of coupling. Ford et al. define the *architectural quantum* as the smallest independently deployable unit with high functional cohesion, the unit at which evolution happens. A monolith is one quantum (everything evolves together); well-factored services are many small quanta (each evolves alone). Two cautions:

- *Inappropriate reuse couples*. A shared domain library used by twenty services means a change to it forces twenty redeployments; shared infrastructure code is usually fine, shared *domain* code usually is not ("prefer duplication over coupling" at service boundaries, echoing DDD's bounded-context independence).
- *Contracts are the coupling surface*. Consumer-driven contract testing (Pact) is a fitness function for evolvability across team boundaries: a provider learns at build time whether a change breaks any consumer, enabling independent deployment without integration-environment lockstep.

== Architecture Decision Records

Evolution erases context: two years later nobody remembers *why* the system uses asynchronous replication, and the constraint gets "fixed" into an outage. *Architecture Decision Records* (Nygard, 2011) counter this: a short, immutable document per significant decision, stored in the repository, with a stable shape, *context* (forces in play), *decision*, *status* (proposed/accepted/superseded), and *consequences* (including the negative ones). Superseded ADRs are never deleted, only linked forward, so the decision history is itself queryable. ADRs are cheap, demand no tooling, and are the single highest-leverage documentation practice for evolving systems; they also feed evaluation methods (see _Architecture Evaluation_) with honest rationale.

== Managing Technical Debt and Drift

- Distinguish *deliberate* debt (recorded shortcut with a repayment trigger, ideally in an ADR) from *bit rot* (unnoticed erosion). Fitness functions exist to make erosion loud: the day someone adds a domain-to-infrastructure import, the build fails, not the architecture review three quarters later.
- Schedule evolution: dependency upgrades automated (Dependabot/Renovate) so the system never drifts years behind; "keep-the-lights-on" capacity explicitly budgeted per team.
- Watch *last responsible moment*: defer decisions that are cheap to defer (which message broker) and decide early the ones that are genuinely expensive to reverse (multi-tenancy model, consistency model, programming-language ecosystem). Reversibility, not importance, determines timing.

== Pitfalls

- *Fitness-function theatre*: a wall of checks that are weak proxies (line coverage as "quality") or perpetually suppressed. Few, sharp, build-failing checks beat dashboards nobody reads.
- *Sandcastle abstraction*: branch-by-abstraction layers that never get removed after the migration, permanent indirection as the fossil record of every refactor.
- *Strangler stall*: the routing layer goes in, two services come out, and the migration is declared "ongoing" for five years, paying for both systems. Set decommissioning milestones with the same rigour as feature milestones.
- *Speculative generality*: building "flexibility" for futures that never arrive is the opposite of evolutionary design, YAGNI applies to architecture too; the cheap option is the system that is easy to *change*, not the system that anticipated everything.
- *Evolving without telemetry*: contract phases of expand–contract executed on faith rather than usage data remove things that were still in use.

== Further Reading

- Ford, N., Parsons, R., & Kua, P. (2017). _Building Evolutionary Architectures_. O'Reilly. (2nd ed. with Sadalage, 2022.)
- Fowler, M. (2004). StranglerFigApplication. martinfowler.com.
- Humble, J., & Farley, D. (2010). _Continuous Delivery_. Addison-Wesley.
- Forsgren, N., Humble, J., & Kim, G. (2018). _Accelerate_. IT Revolution.
- Nygard, M. (2011). Documenting architecture decisions. cognitect.com blog.
- Feathers, M. (2004). _Working Effectively with Legacy Code_. Prentice Hall.

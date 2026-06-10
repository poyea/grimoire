= Architectural Styles

An architectural style is a named collection of design decisions: how a system is partitioned into components, how those components communicate, and which constraints govern their evolution. Styles are not mutually exclusive; real systems mix them. This chapter surveys the major styles, layered, hexagonal, clean, pipes-and-filters, event-driven, microkernel, and space-based, and develops the trade-off vocabulary needed to choose between them.

*See also:* _Monoliths and Microservices_ (deployment-level decomposition), _Event-Driven Architecture_ (event-driven style in depth), _Architecture Evaluation_ (how to assess a chosen style against quality attributes).

== Layered Architecture

The layered (n-tier) style organises code into horizontal layers, classically presentation, business logic, persistence, and database, where each layer may call only the layer directly beneath it. It is the default style of most enterprise frameworks: Java EE, classic .NET, Rails MVC all encourage it.

Strengths:
- Familiar to nearly every developer; low onboarding cost.
- Separation of technical concerns: UI changes do not touch SQL.
- Layers of isolation: a closed layer can be replaced (swap JPA for jOOQ) without rippling upward.

Weaknesses:
- *Architecture sinkhole antipattern*: requests pass through layers that add no value, just delegation. Richards suggests that if more than roughly 80% of requests are sinkholes, the layering is ceremony, not architecture.
- Layers encourage organising by technical concern rather than by domain, so a single feature change touches every layer (low deployability, poor team autonomy).
- Tends toward a single deployment unit and a single shared database, which constrains scalability to whole-system replication.

A useful refinement distinguishes *open* layers (may be bypassed) from *closed* layers (must be traversed), documented explicitly so the constraint is testable.

== Hexagonal Architecture (Ports and Adapters)

Alistair Cockburn proposed hexagonal architecture in 2005 to fix a recurring failure: business logic leaking into UI and database code, making the application impossible to test or drive programmatically. The rule is simple:

- The *application core* contains domain logic and defines *ports*: interfaces describing what the core needs (a `PaymentGateway`, an `OrderRepository`) and what it offers (use-case interfaces).
- *Adapters* implement ports for specific technologies: a REST controller is a driving adapter; a PostgreSQL repository is a driven adapter.
- Dependencies point inward only. The core never imports an adapter.

The payoff is substitutability: in tests, swap the real database adapter for an in-memory fake; in production, swap Stripe for Adyen behind the same port. The hexagon shape itself is incidental, Cockburn chose it only to suggest multiple symmetric ports rather than a privileged "top" and "bottom".

== Clean Architecture

Robert C. Martin's clean architecture (2012, book 2017) generalises hexagonal, onion (Palermo, 2008), and DCI into concentric rings: *entities* (enterprise-wide business rules) at the centre, then *use cases* (application-specific rules), then *interface adapters* (controllers, presenters, gateways), then *frameworks and drivers* at the rim. The single governing constraint is the *dependency rule*: source-code dependencies may point only inward, enforced by dependency inversion at each boundary.

Criticisms worth knowing:
- The ring count is often taken too literally; Martin himself notes the number of circles is schematic.
- Mapping data between rings (entity to use-case DTO to view model) adds boilerplate; teams routinely report 3–4 representations of the same concept.
- For CRUD-heavy services with little domain logic, the indirection costs more than it protects. Style choice should follow the *volatility* of the parts being isolated: isolate the database behind a port only if you plausibly need to change or fake it.

== Pipes and Filters

Pipes-and-filters decomposes processing into a chain of *filters* (transformations) connected by *pipes* (data channels). Unix shells are the canonical example: `grep | sort | uniq -c`. Each filter is independently testable, composable, and parallelisable.

The style dominates data engineering: ETL pipelines, compiler phases (lex, parse, optimise, emit), and stream processors (Kafka Streams topologies, Apache Beam) are all pipes-and-filters. Variants include *producer-consumer* with bounded queues for backpressure, and *tee-and-merge* topologies for fan-out/fan-in.

Trade-offs: excellent modifiability and reuse; weak fit for interactive request/response, because end-to-end latency is the sum of stage latencies and error handling across stages is awkward (a failure in stage 5 of 7 must be compensated or the partial output discarded).

== Event-Driven Architecture

In the event-driven style, components communicate by emitting and reacting to *events*, immutable facts about something that happened. Two canonical topologies (Richards & Ford, _Fundamentals of Software Architecture_, 2020):

- *Broker topology*: events flow through a message broker (Kafka, RabbitMQ); each processor reacts and emits further events. Highly decoupled and scalable; no central coordinator, so error handling and end-to-end visibility are hard.
- *Mediator topology*: a mediator (workflow engine, orchestrator) receives an initiating event and dispatches commands to processors. Better control and error recovery; the mediator becomes a coupling and scaling bottleneck.

Event-driven systems score highest of all styles on scalability, elasticity, and fault tolerance in Richards' star ratings, and lowest on simplicity and testability. The style is covered in depth in its own chapter.

== Microkernel (Plug-in) Architecture

The microkernel style splits a system into a minimal *core* providing lifecycle and contract management, plus independent *plug-in* modules. Eclipse (OSGi bundles), web browsers (extensions), Jenkins (1,800+ plugins), VS Code, and payment platforms with per-country tax plug-ins all use it.

Design decisions that matter:
- *Registry*: how the core discovers plug-ins (manifest files, classpath scanning, a database table).
- *Contract*: versioned plug-in API; the core must tolerate plug-ins built against older contract versions.
- *Isolation*: in-process plug-ins are fast but a misbehaving plug-in crashes the host; out-of-process plug-ins (Chrome's site isolation, VS Code extension host) trade latency for fault containment.

Microkernel is the only style Richards rates well for both simplicity and adaptability, but the core is a single deployment unit, so it inherits monolithic scaling limits.

== Space-Based Architecture

Space-based architecture (named after tuple spaces, e.g. JavaSpaces) targets extreme, spiky concurrency, ticketing sales, flash sales, auction systems, by removing the database from the synchronous request path:

- *Processing units* hold application data in replicated in-memory data grids (Hazelcast, Apache Ignite, Coherence).
- A *virtualised middleware* layer (messaging grid, data grid, processing grid, deployment manager) routes requests and manages elastic scale-out.
- *Data pumps* asynchronously write changes to the database of record; *data readers* hydrate new processing units on startup.

Throughput scales nearly linearly with processing units because no request waits on a disk-backed database. Costs: eventual consistency with the system of record, complex cache-collision behaviour as replicated grids grow (collision rate grows with update rate and replication latency), and high operational complexity. It is a niche style, but the only one purpose-built for elasticity under load measured in hundreds of thousands of concurrent users.

== Choosing: Trade-off Analysis

There are no best styles, only trade-offs against the quality attributes that matter for the system at hand. A condensed comparison, following Richards & Ford's ratings:

#table(
  columns: 6,
  [*Style*], [*Simplicity*], [*Scalability*], [*Deployability*], [*Fault tolerance*], [*Cost*],
  [Layered], [High], [Low], [Low], [Low], [Low],
  [Hexagonal/Clean], [Medium], [Low], [Medium], [Low], [Low],
  [Pipes & filters], [High], [Medium], [Medium], [Low], [Low],
  [Microkernel], [High], [Low], [Medium], [Low], [Low],
  [Event-driven], [Low], [High], [Medium], [High], [Medium],
  [Space-based], [Low], [Very high], [Medium], [High], [High],
)

Heuristics that survive contact with practice:
- Start from the two or three *driving* quality attributes (see _Architecture Evaluation_), not from a style you admire.
- Monolithic styles (layered, microkernel, clean as a single deployable) minimise cost and operational burden; distributed styles buy scalability and fault isolation with the *fallacies of distributed computing* (Deutsch, 1994): the network is not reliable, latency is not zero, bandwidth is not infinite.
- Styles compose: a microservice is often internally hexagonal; an event-driven system's processors may be layered; a space-based processing unit may host plug-ins.
- The architecture should make the *most frequent change* cheap. If most changes are domain features, partition by domain (vertical slices, bounded contexts) rather than by technical layer.

== Further Reading

- Richards, M., & Ford, N. (2020). _Fundamentals of Software Architecture_. O'Reilly.
- Cockburn, A. (2005). Hexagonal architecture (ports and adapters). alistair.cockburn.us.
- Martin, R. C. (2017). _Clean Architecture_. Prentice Hall.
- Shaw, M., & Garlan, D. (1996). _Software Architecture: Perspectives on an Emerging Discipline_. Prentice Hall.
- Bass, L., Clements, P., & Kazman, R. (2021). _Software Architecture in Practice_, 4th ed. Addison-Wesley.

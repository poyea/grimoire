= Domain-Driven Design

Domain-driven design (DDD), introduced by Eric Evans in his 2003 book, is a school of design that puts the business domain, not the database schema, not the framework, at the centre of the architecture. Its core claim: the hardest part of most software is not technical but conceptual, and software that does not embody a rigorous model of its domain rots into a *big ball of mud*. DDD splits into *strategic* design (where are the boundaries?) and *tactical* design (how is the model expressed in code?).

*See also:* _Monoliths and Microservices_ (bounded contexts as service boundaries), _Event-Driven Architecture_ (domain events as integration mechanism), _Distributed Data Patterns_ (aggregates and consistency boundaries across services).

== Ubiquitous Language

The ubiquitous language is a shared, rigorous vocabulary developed jointly by developers and domain experts, and used *everywhere*: conversations, documentation, class names, method names, database tables, test descriptions. If the business says "policy lapses", the code must contain `policy.lapse()`, not `policyRecord.setStatusFlag(4)`.

Two disciplines make it work:
- The language is *scoped to a bounded context*: "Customer" legitimately means different things in Sales and in Shipping; pretending it is one concept produces the bloated god-class.
- The language evolves: when a conversation reveals that "cancel" and "withdraw" are different business operations, the model and code are refactored to match, immediately. Evans calls this *knowledge crunching*.

The practical test: read a use-case implementation aloud to a domain expert. If they cannot follow it, the model has diverged from the domain.

== Bounded Contexts

A *bounded context* is the boundary within which a particular model and language are valid and internally consistent. It is DDD's most influential idea, the conceptual ancestor of microservice boundaries.

Heuristics for finding context boundaries:
- *Linguistic*: the same word means different things ("Product" in catalogue vs. fulfilment), or different words mean the same thing.
- *Organisational*: different teams, departments, or domain experts own different processes (Conway alignment).
- *Data ownership and lifecycle*: an Order in checkout is mutable; an Order in accounting is an immutable financial fact.
- *Rate of change*: pricing rules change weekly; the warehouse model changes yearly.

Within a context, the model is kept pure; *between* contexts, translation is explicit. A typical e-commerce system might have contexts for Catalogue, Ordering, Payments, Fulfilment, and Identity, each with its own Customer-shaped concept holding only the attributes that context needs.

== Context Mapping

A *context map* documents the relationships between bounded contexts, including the power dynamics between the teams that own them. Evans and Vernon catalogue the recurring patterns:

#table(
  columns: 2,
  [*Pattern*], [*Meaning*],
  [Partnership], [Two teams succeed or fail together; coordinated planning],
  [Shared kernel], [Small shared model/code owned jointly; cheap but couples release cycles],
  [Customer–supplier], [Downstream needs feed upstream's backlog; negotiated interface],
  [Conformist], [Downstream adopts upstream's model wholesale; no translation, no leverage],
  [Anticorruption layer (ACL)], [Downstream translates upstream's model into its own at the boundary],
  [Open host service], [Upstream publishes a stable protocol for many consumers],
  [Published language], [Shared, well-documented interchange format (e.g. an industry schema)],
  [Separate ways], [No integration; duplicate the small overlap],
)

The *anticorruption layer* deserves emphasis: when integrating with a legacy system or a vendor API, a thin adapter and translator layer keeps the foreign model from leaking into your context. ACLs are also the standard mechanism in strangler-fig migrations (see _Evolutionary Architecture_).

== Tactical Patterns: Aggregates, Entities, Value Objects

Tactical DDD provides building blocks for expressing the model in code.

=== Entities and Value Objects

- An *entity* has identity that persists through change: Order \#10423 remains the same order as lines are added. Equality is by identifier.
- A *value object* is defined entirely by its attributes and is immutable: `Money(amount: 100, currency: "EUR")`, `DateRange`, `Address`. Equality is structural. Value objects are the workhorse of expressive models; they eliminate primitive obsession (passing `BigDecimal` plus `String` everywhere) and give invariants a home (a `Money` constructor rejects mixing currencies).

=== Aggregates

An *aggregate* is a cluster of entities and value objects with a single *aggregate root* through which all access flows, and a boundary that defines the unit of *transactional consistency*. Vernon's rules of thumb (2013):

+ Protect true invariants inside the boundary: a business rule like "an order's total must equal the sum of its lines" lives inside the Order aggregate and is enforced in one transaction.
+ Keep aggregates small. Large aggregates serialise concurrent writers (optimistic-lock contention) and load slowly.
+ Reference other aggregates *by identity only* (store `customerId`, not a `Customer` object graph).
+ Update one aggregate per transaction; coordinate across aggregates with domain events and eventual consistency.

Rule 4 is why aggregates map cleanly to distributed systems: the aggregate boundary is exactly the boundary within which strong consistency is cheap.

=== Repositories, Factories, Domain Services

- A *repository* provides collection-like access to aggregate roots (`orders.findById(id)`), hiding persistence. One repository per aggregate, not per table.
- A *factory* encapsulates complex creation logic that does not belong on any one object.
- A *domain service* holds domain logic that spans aggregates (e.g. a funds-transfer policy between two Account aggregates) and is stateless.

== Domain Events

A *domain event* is an immutable record of something that happened in the domain, named in the past tense in the ubiquitous language: `OrderPlaced`, `PaymentCaptured`, `PolicyLapsed`. Introduced informally by Evans and elaborated by Vernon, domain events serve three roles:

- Within a context: decouple side effects (when `OrderPlaced`, the loyalty module awards points) from the command that caused them.
- Between contexts: the published event becomes the integration contract, often delivered through a broker with an outbox (see _Event-Driven Architecture_).
- As a modelling tool: events are the things domain experts naturally narrate, which is what event storming exploits.

A common discipline: an aggregate method validates a command, mutates state, and records one or more events; infrastructure dispatches them after the transaction commits, avoiding ghost events from rolled-back transactions.

== Strategic vs. Tactical: Where Teams Go Wrong

The most common DDD failure mode is adopting tactical patterns, entities, repositories, layered "DDD-style" projects, without strategic design. The result is a well-decorated big ball of mud: one giant shared model with repository classes. The community consensus (Vernon, Millett, Tune) is blunt: *strategic design is the part that matters*. Bounded contexts and context maps deliver value even in a plain CRUD codebase; tactical patterns pay off only where domain logic is genuinely complex.

The complement: not every context deserves DDD. A *core domain* (where the business differentiates) merits deep modelling; *supporting subdomains* can be simpler; *generic subdomains* (auth, invoicing) should be bought or borrowed. Investing modelling effort uniformly is waste.

== Event Storming

Event storming, invented by Alberto Brandolini around 2013, is a workshop format for rapidly exploring a domain with the people who know it. On a long wall of butcher paper:

+ Domain experts and developers write *domain events* on orange stickies and arrange them on a timeline.
+ Add *commands* (blue) that trigger events and *actors* (yellow) who issue them.
+ Add *read models* (green), *policies/reactions* (lilac: "whenever X, then Y"), and *external systems* (pink).
+ Mark *hot spots* (red) where people disagree or knowledge is missing, the disagreements are the most valuable output.
+ Pivotal events and clusters of related stickies reveal candidate *bounded contexts* and *aggregates*.

A big-picture session takes a day and routinely surfaces process misunderstandings that years of requirements documents missed. Variants: *process-level* storming (one workflow in detail) and *design-level* storming (deriving aggregates and commands ready for implementation).

== Further Reading

- Evans, E. (2003). _Domain-Driven Design: Tackling Complexity in the Heart of Software_. Addison-Wesley.
- Vernon, V. (2013). _Implementing Domain-Driven Design_. Addison-Wesley.
- Vernon, V. (2016). _Domain-Driven Design Distilled_. Addison-Wesley.
- Brandolini, A. (2021). _Introducing EventStorming_. Leanpub.
- Millett, S., & Tune, N. (2015). _Patterns, Principles, and Practices of Domain-Driven Design_. Wrox.
- Khononov, V. (2021). _Learning Domain-Driven Design_. O'Reilly.

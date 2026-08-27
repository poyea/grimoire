#import "../template.typ": xref

= Distributed Data Patterns <distributed-data-patterns>

Splitting a system into services is, in the end, a decision about data: who owns it, where the truth lives, and what happens to transactions and joins that used to be a single SQL statement. This chapter covers the theory that bounds the design space, CAP, PACELC, consistency models, and the patterns that fill it: database-per-service, sagas, the transactional outbox, change data capture, replication and partitioning, and the reporting problem that database-per-service creates.

*See also:* #xref("software-architecture", "event-driven-architecture", label: "Event-Driven Architecture") (the messaging substrate most of these patterns ride on), #xref("software-architecture", "monoliths-and-microservices", label: "Monoliths and Microservices") (when to split at all), #xref("software-architecture", "resilience-patterns", label: "Resilience Patterns") (what failure does to data flows in flight).

== The Theoretical Bounds

*CAP* (Brewer's conjecture 2000, proved by Gilbert & Lynch 2002): a distributed data system cannot simultaneously provide consistency (linearisability), availability (every request to a non-failed node gets a response), and partition tolerance. Since network partitions are not optional in a distributed system, the real choice is what to sacrifice *during* a partition: refuse some requests (CP) or serve possibly-stale data (AP). CAP is widely abused; Brewer's own 2012 retrospective ("CAP twelve years later") stresses that partitions are rare and the interesting design work is in the recovery and in the nuances between the extremes.

*PACELC* (Abadi, 2012) is the more useful formulation: if Partitioned, trade Availability vs. Consistency; Else, trade Latency vs. Consistency. The "else" clause matters daily, not just during partitions: synchronous replication for stronger consistency costs write latency on every single request. DynamoDB and Cassandra are PA/EL; Spanner-style systems and HBase sit toward PC/EC.

Consistency models form a spectrum, not a binary:
- *Linearisability*: operations appear instantaneous at a single point between invocation and response; the strongest practical guarantee, what etcd provides for coordination (ZooKeeper linearises writes; reads need `sync`).
- *Sequential and causal consistency*: weaker orderings; causal (with mechanisms like version vectors) is the strongest model achievable with availability during partitions.
- *Read-your-writes, monotonic reads*: session guarantees that fix the most user-visible anomalies of eventual consistency cheaply.
- *Eventual consistency*: replicas converge given quiescence, a liveness promise with no safety content on its own; everything depends on the conflict-resolution rule (last-writer-wins drops data; CRDTs, Shapiro et al. 2011, merge mathematically; or surface conflicts to the application, as Dynamo's shopping cart did).

The architectural use of all this: per *operation*, decide the weakest model the business tolerates, and pay for nothing stronger. Account balance display tolerates staleness; withdrawal authorisation does not.

== Database per Service

The defining microservice data rule (Lewis & Fowler, 2014): each service owns its data store exclusively; other services get to it only through the service's API or its published events. Rationale:

- *Encapsulation*: schemas can change without coordinating consumers; the database stops being a shared, frozen integration point, the *shared database* integration style is precisely what makes systems unsplittable.
- *Fit*: each service picks the store its access pattern needs (relational, document, graph, search), polyglot persistence (Fowler & Sadalage, 2012).
- *Isolation*: one service's runaway query cannot take down another's database.

The bill arrives immediately: no cross-service ACID transactions, no cross-service joins, and consistency between services becomes eventual by default. The rest of this chapter is the bill's itemisation. Note the rule is about *ownership*, not hardware: separate schemas with no cross-schema access on one database server satisfy it for a modular monolith; shared tables never do.

== Sagas

A *saga* (Garcia-Molina & Salem, SIGMOD 1987, rediscovered for microservices) replaces a distributed transaction with a sequence of local transactions, each committed in one service, where every step has a *compensating transaction* that semantically undoes it. If step 4 of 5 fails, the saga runs compensations for steps 3, 2, 1 in reverse order. Two-phase commit (XA) is the road not taken: it provides atomicity but couples availability of all participants and holds locks across the slowest one, which is why it is essentially absent from large-scale service architectures.

Coordination styles:
- *Choreography*: each service listens for the previous step's event and acts ("OrderCreated" triggers payment, "PaymentCaptured" triggers inventory). No central coordinator, minimal coupling; but the workflow exists only implicitly across codebases, cyclic dependencies creep in, and answering "where is order 42's saga?" requires tracing. Sensible up to roughly three or four steps.
- *Orchestration*: a saga orchestrator (a state machine in the initiating service, or an engine like Temporal or Camunda) commands each step and handles failures explicitly. The workflow is in one place, testable and observable; the cost is the orchestrator's coupling to every participant. Preferred for long or frequently changing workflows.

Sagas are *ACD*, not ACID: atomicity (eventually, via compensation) and durability survive, but *isolation is lost*, other transactions can observe the saga's intermediate states. Countermeasures (catalogued by Richardson, _Microservices Patterns_, 2018): *semantic locks* (mark the order "PENDING" so others keep hands off), *commutative updates*, *pessimistic ordering* (put the riskiest, hardest-to-compensate step last, e.g. capture payment after inventory is reserved), and *reread-and-verify*. Also: some steps are not compensatable (a sent email); classify steps as compensatable, *pivot* (the go/no-go point), and *retriable* (after the pivot, steps must be retried to completion, never compensated).

== The Dual-Write Problem and the Transactional Outbox

A service that writes to its database *and* publishes an event has a problem: the two operations cannot be made atomic across a database and a broker. Crash between them and you get a state change nobody heard about, or an announcement of a change that rolled back. Every architecture that "updates the DB then publishes to Kafka" inline has this bug.

The *transactional outbox* pattern fixes it:
+ In the same local ACID transaction as the state change, insert the event into an `outbox` table.
+ A separate *message relay* reads the outbox and publishes to the broker, marking rows sent, either by *polling* the table or, better, by tailing the database's transaction log.

The guarantee is at-least-once publication in commit order, so consumers need idempotency (see #xref("software-architecture", "event-driven-architecture", label: "Event-Driven Architecture")), but no event is ever lost or phantom. The log-tailing variant is *change data capture* (CDC): tools like Debezium read MySQL's binlog or PostgreSQL's logical replication stream and emit row changes (or just outbox rows) into Kafka. CDC also serves data integration beyond the outbox, feeding caches, search indexes, and warehouses from the source of record without dual writes, and the *event-carried* style of keeping local replicas of other services' reference data current. Caution: raw CDC on business tables leaks the service's internal schema as a public contract; the outbox table (an explicit, versioned event shape) is the contract-safe variant.

== Replication and Partitioning Inside a Service

Within each service's store, the classic mechanics still apply (Kleppmann, _Designing Data-Intensive Applications_, 2017, is the standard treatment):

- *Replication*: single-leader (reads scale out, writes do not; replication lag causes read-your-writes anomalies on replicas, route the writing user's reads to the leader briefly), multi-leader (write availability across regions, but conflicts), leaderless with quorums (Dynamo-style, $R + W > N$ for read-your-writes under favourable conditions, with sloppy quorums weakening even that).
- *Partitioning (sharding)*: by key range (range scans work; hot ranges skew) or by hash (even spread; range queries die). Secondary indexes are either local (scatter-gather reads) or global (expensive, possibly async, writes). Resharding is the operation everyone defers and regrets; consistent hashing and pre-split virtual shards (e.g. Vitess) make it survivable.
- *Hot keys*: a single celebrity row defeats hashing; mitigate with key salting, request coalescing, and caching.

These choices surface in service APIs: a partition key choice constrains which queries are cheap, and replication topology fixes which consistency guarantees you can offer per endpoint.

== Queries Across Services

Database-per-service kills the cross-entity join. Replacements, in increasing weight:

- *API composition*: a composing service (or BFF/gateway) calls each owner and joins in memory. Fine for small fan-out and small result sets; degrades into the distributed N+1 problem for list views ("orders with customer names, paginated") and inherits the availability product of every participant.
- *CQRS read models / materialised views*: a dedicated view service subscribes to the owning services' events and maintains a pre-joined, query-shaped store (e.g. an "order history" document per customer in Elasticsearch). Reads are fast, local, and survivable when source services are down; the price is eventual consistency, projection-rebuild machinery, and another stateful component.
- *Data warehouse / lake for analytics*: operational services stream changes (CDC or events) into an analytical store; analysts never query production services. A *data mesh* (Dehghani, 2019) generalises this with domain-aligned ownership of analytical "data products", organisationally, database-per-service extended to analytics.

The anti-pattern is reaching into another service's database "just for reporting": it silently reinstates the shared database, freezing the owner's schema forever.

== Choosing: A Field Guide

#table(
  columns: 3,
  [*Problem*], [*Reach for*], [*Accepting*],
  [Multi-service business transaction], [Saga (orchestrated if > 3–4 steps)], [Lost isolation; compensation logic],
  [Reliable state-change events], [Transactional outbox + CDC relay], [At-least-once; idempotent consumers],
  [Cross-service list/report queries], [CQRS read model fed by events], [Eventual consistency; rebuild tooling],
  [Other services' reference data], [Event-carried local replica], [Staleness window],
  [Cross-region writes], [Multi-leader or CRDTs], [Conflict resolution as a domain problem],
  [Strict invariant (unique username, no double spend)], [Single owner, one aggregate, one local transaction], [Design boundaries so the invariant fits inside],
)

The last row is the deepest lesson: the cheapest distributed-data pattern is the one you avoid needing. Draw aggregate and service boundaries so that the invariants requiring atomicity live *inside* one boundary (see _Domain-Driven_ _Design_), and let everything that crosses boundaries be eventually consistent by explicit, documented choice.

== Pitfalls

- Dual writes without an outbox, the most common correctness bug in event-driven services, invisible until a crash lands between the two writes.
- Sagas without designed compensations: the failure path is sketched in a wiki, untested, and wrong; test compensations with fault injection like any other code path.
- Treating eventual consistency as an implementation detail instead of a product decision: the UI must be designed for "pending" states, or support discovers them for you.
- Last-writer-wins conflict resolution silently discarding writes under clock skew.
- "Just one little cross-service join" in SQL, the shared-database antipattern on the instalment plan.
- Ignoring replication lag in read-your-writes flows: the user updates their profile, the next page reads a replica, and the edit "disappears".
- Resharding deferred until the hot partition is on fire; partition-key choices deserve capacity maths at design time.

== Further Reading

- Kleppmann, M. (2017). _Designing Data-Intensive Applications_. O'Reilly.
- Richardson, C. (2018). _Microservices Patterns_. Manning.
- Garcia-Molina, H., & Salem, K. (1987). Sagas. _SIGMOD '87_.
- Abadi, D. (2012). Consistency tradeoffs in modern distributed database system design. _IEEE Computer_, 45(2).
- Brewer, E. (2012). CAP twelve years later: how the "rules" have changed. _IEEE Computer_, 45(2).
- Shapiro, M., et al. (2011). Conflict-free replicated data types. _SSS 2011_.

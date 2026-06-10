= Event-Driven Architecture

In an event-driven architecture (EDA), components communicate by producing and consuming *events*: immutable records that something happened ("OrderPlaced", "PaymentCaptured"). Producers do not know who consumes; consumers do not know who produced. This inversion, from "call the service that does X" to "announce that X happened", is what buys EDA its decoupling, scalability, and fault tolerance, and also what makes it the hardest mainstream style to reason about end to end. This chapter covers the messaging vocabulary, broker and mediator topologies, delivery semantics, event sourcing and CQRS, and the failure modes that distinguish production systems from conference demos.

*See also:* _Architectural Styles_ (where EDA sits among the styles), _Distributed Data Patterns_ (sagas, outbox, CQRS data flow), _Resilience Patterns_ (handling consumer and broker failure).

== Events, Commands, and Messages

Terminology discipline pays off, because the words encode intent and coupling:

- A *command* is a request directed at one logical receiver: "ReserveInventory". The sender expects it to be acted on (and often expects a reply). Commands couple the sender to the existence of a handler.
- An *event* is a statement of fact, past tense, with no designated receiver: "InventoryReserved". Zero, one, or many consumers may react; the producer must not care.
- A *message* is the transport envelope for either. Queues carry messages; whether a message is a command or an event is a semantic, not a technical, distinction.

A useful litmus test from Hohpe & Woolf's _Enterprise Integration Patterns_ (2003): if removing a consumer breaks the producer's business process, you have a command in event's clothing, and the implicit coupling will surface as a production incident.

Event payload design runs a spectrum:
- *Notification events* carry only an identifier ("Order 42 changed"); consumers call back for details. Small, but reintroduces synchronous coupling and load on the producer.
- *Event-carried state transfer* (Fowler, 2017) carries the full relevant state; consumers keep local replicas and never call back. Decoupled and read-scalable, at the cost of payload size and eventual consistency of the replicas.
- *Delta events* carry just the change. Compact, but consumers must apply deltas in order, which raises the ordering stakes.

== Broker Topology

In the *broker topology* (Richards & Ford, 2020), there is no central coordinator. An initiating event enters the broker; each processor subscribes to the event types it cares about, does its work, and emits further events announcing what it did, even if nothing currently listens, since advertising state change is what enables future extension.

Properties:
- Highest decoupling, scalability, and elasticity of any EDA variant; processors scale independently per event type.
- *No end-to-end visibility or control.* There is no single place that knows whether "place order" completed. Determining workflow state requires correlating events across processors (correlation IDs and distributed tracing are mandatory, not optional).
- *Error handling is hard.* If the payment processor fails mid-chain, the upstream processors have already committed and moved on; recovery requires compensating events, not rollback.

== Mediator Topology

In the *mediator topology*, an event mediator (a workflow engine such as Camunda or Temporal, a stream processor, or custom orchestration code) receives the initiating event and dispatches *commands* to processors in a defined order, tracking workflow state.

Properties:
- The mediator knows the workflow, so it can detect, retry, and compensate failed steps; end-to-end state is queryable.
- Processors are more coupled (they receive commands, implying a known commander), and the mediator is a scaling and change bottleneck: every workflow change touches it.
- Complex workflows often use tiers of mediators (simple ones in a lightweight engine, long-running human-in-the-loop ones in a BPM tool).

The choice mirrors *orchestration versus choreography* in saga design (see _Distributed Data Patterns_): choreography (broker) for simple, stable flows with few steps; orchestration (mediator) when the workflow itself is complex, audited, or frequently changed.

== Brokers and Logs: Queues versus Streams

Two broker families with different contracts:

- *Message queues* (RabbitMQ, ActiveMQ, Amazon SQS, JMS brokers): a message is delivered to one consumer per queue and deleted on acknowledgement. Competing consumers give easy work distribution; routing (topic exchanges, headers) is rich. History is gone once consumed.
- *Event logs* (Apache Kafka, Amazon Kinesis, Apache Pulsar in stream mode, Redpanda): events are appended to a partitioned, durable, replicated log and retained for a configured period (or forever, with compaction). Consumers track their own offsets, so multiple independent consumer groups read the same stream at their own pace, and new consumers can *replay history*.

Kafka's partition model is the load-bearing detail: ordering is guaranteed *only within a partition*, and a partition is consumed by at most one consumer per group. Therefore the partition key (typically an aggregate ID, e.g. order ID) determines both ordering scope and maximum parallelism. Choosing a hot key (one celebrity user, one big tenant) creates a hot partition no amount of consumers can absorb.

== Delivery Semantics

Every messaging system offers one of three honest guarantees:

- *At-most-once*: fire and forget; messages may be lost. Acceptable for metrics and telemetry, rarely for business events.
- *At-least-once*: the broker redelivers until acknowledged, so consumers *will* see duplicates (after a crash between processing and acking). This is the practical default.
- *Exactly-once*: achievable only within a closed system. Kafka's "exactly-once semantics" (idempotent producers plus transactions, KIP-98, 2017) covers Kafka-to-Kafka read-process-write topologies; the moment a consumer touches an external system (a database, an email API), the guarantee ends at the boundary.

The architectural consequence: *consumers must be idempotent*. Standard techniques: a unique event ID with a processed-IDs table checked transactionally; natural idempotency (setting a value rather than incrementing); or version/sequence checks on the target aggregate. Designing for at-least-once plus idempotency is cheaper and more robust than chasing exactly-once across heterogeneous systems.

Ordering deserves the same scepticism: global ordering across a distributed broker does not exist at scale. Design consumers to tolerate reordering outside the partition key's scope, or carry sequence numbers and buffer.

== Event Sourcing

*Event sourcing* makes the event log the system of record: instead of storing current state and emitting events as a side effect, the application stores the full sequence of domain events per aggregate and derives state by replaying them (current state = "fold" of events, optionally accelerated with periodic snapshots).

Benefits:
- A complete, audit-grade history by construction; temporal queries ("what was this account's balance on March 3?") are first-class.
- New read models can be built retroactively by replaying history.
- Natural fit with DDD aggregates: an aggregate's events are its transactional unit, appended with an optimistic-concurrency check on the expected version.

Costs, which are substantial:
- *Schema evolution*: events are forever. Upcasting old event versions on read, or copy-transform migrations, must be designed in from day one.
- Querying across aggregates requires projections (hence the routine pairing with CQRS); there is no ad-hoc SQL over current state.
- Deleting data (GDPR right to erasure) conflicts with an immutable log; mitigations include crypto-shredding (encrypt personal data per subject, destroy the key).
- It is unfamiliar; teams underestimate the tooling (event store, projections, replay infrastructure). Greg Young's advice stands: apply it per bounded context where the history *is* the domain (ledgers, trading, claims), not system-wide.

== CQRS

*Command Query Responsibility Segregation* (Young, 2010; rooted in Meyer's command-query separation) splits the model used to change state from the model(s) used to read it. Commands go to a write model (often DDD aggregates, possibly event-sourced); events from the write side update one or more denormalised *read models* (a SQL view table, Elasticsearch index, Redis cache) shaped exactly for each query.

CQRS is independent of event sourcing, a CQRS read model can be fed by database triggers or CDC, but the two compose naturally: projections subscribe to the event stream. The price is *eventual consistency between write and read sides*: a user may not immediately see their own write. Mitigations: read-your-own-writes via the write model for the acting user, version-stamped reads, or UI optimism. Use CQRS where read and write shapes genuinely diverge or read load dwarfs write load; for plain CRUD it is pure overhead, a point Young himself makes emphatically.

== Operational Realities and Pitfalls

- *Dead letter queues*: a poison message that always fails must, after bounded retries, be parked in a DLQ with alerting, otherwise it blocks the queue (or, in Kafka, stalls the partition). A DLQ nobody monitors is a black hole; pair it with redrive tooling.
- *Backpressure*: producers can outrun consumers. Bounded queues, consumer lag monitoring (Kafka consumer lag is the single most important EDA health metric), and load shedding beat unbounded buffering, which converts overload into an out-of-memory crash later.
- *Event storms and cycles*: processor A emits an event that (transitively) triggers A again. Cycles in the event graph must be detected by design review; correlation IDs plus hop counts catch them at runtime.
- *Schema management*: events are public API. Use a schema registry (Avro/Protobuf/JSON Schema with compatibility rules); only make backward-compatible changes (add optional fields); never repurpose a field. CloudEvents (CNCF, 1.0 in 2019) standardises the envelope.
- *The "event-driven" distributed monolith*: if consumers immediately call the producer back for data, or every event change requires coordinated multi-team deploys, the broker is just an expensive RPC bus.
- *Testing*: unit-test processors against event fixtures; contract-test schemas in CI; accept that end-to-end determinism is gone and invest in tracing (OpenTelemetry context propagation through message headers) instead.

== When to Use It

EDA earns its complexity when: consumers genuinely vary independently of producers (extensibility), load is spiky or asymmetric per step (elasticity), workflows are naturally reactive (fraud detection, notifications, integrations), or audit/replay is a domain requirement. It is a poor first choice when the domain is mostly synchronous request/response with users waiting, when strong consistency is pervasive, or when the team lacks operational maturity in monitoring and on-call, because EDA moves complexity from code into the runtime, where only operations can see it.

== Further Reading

- Hohpe, G., & Woolf, B. (2003). _Enterprise Integration Patterns_. Addison-Wesley.
- Richards, M., & Ford, N. (2020). _Fundamentals of Software Architecture_, ch. 14. O'Reilly.
- Kleppmann, M. (2017). _Designing Data-Intensive Applications_, chs. 4, 11. O'Reilly.
- Fowler, M. (2017). What do you mean by "Event-Driven"? martinfowler.com.
- Young, G. (2010). CQRS Documents. goodenoughsoftware.net.
- Stopford, B. (2018). _Designing Event-Driven Systems_. O'Reilly.

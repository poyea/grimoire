= Streaming

Streaming systems process unbounded data with bounded latency. Three concerns dominate: ordering (event time vs processing time), correctness under failure (exactly-once semantics), and state management (how billions of keyed aggregates fit in memory). This chapter covers Kafka as the durable log, Flink as the canonical processing engine, and the streaming-$"SQL"$ wave (RisingWave, Materialize, ksqlDB).

*See also:* _Change Data Capture_ (the most common stream source), _Lakehouse Engineering_ (streaming writes to Iceberg / Delta / Hudi), _Streaming and Incremental Computation_ (database framing), _Log-Based Systems_ (distributed-systems framing).

== The Log as Substrate

Kafka is a partitioned, replicated, append-only log. A *topic* is split into *partitions*; each partition is an ordered sequence of immutable records on disk, replicated to $f + 1$ brokers. Consumers track *offsets* per partition.

```
topic: orders, 12 partitions
  ┌──────────────────────────────────────────┐
  │ p0  [r0 r1 r2 r3 r4 ...]                 │  ← leader (broker 3)
  │ p1  [r0 r1 r2 r3 ...]                    │  ← leader (broker 1)
  │ ...                                       │
  └──────────────────────────────────────────┘
producer  ──▶ partition by key (hash) ──▶ leader
consumer  ──▶ assigned partitions via group coordinator
```

Key properties:

- *Per-partition order.* Records with the same key are read in write order. No global order.
- *Retention.* Time-based or compaction-based (keep only the latest record per key).
- *Replay.* Consumers reset offsets and replay history; this is what makes Kafka the substrate for reprocessing.

== Event Time vs Processing Time

A clickstream event has both a *creation time* (`event_ts`) and a *processing time* (when the engine sees it). Mobile clients buffer offline and dump events hours later. Any system that aggregates by hour must aggregate by event time, not by arrival.

A *watermark* $W(t)$ is the engine's estimate "we have seen all events with `event_ts <= W(t)`." Watermarks advance as events arrive; a window closes when its end time falls below the current watermark. Late events past the watermark either land in a side output or update an already-emitted result.

== Flink Job

Flink models computation as a $"DAG"$ of operators with managed state and checkpointed barriers.

```java
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
env.enableCheckpointing(60_000);  // every 60s

DataStream<Order> orders = env
    .fromSource(KafkaSource.<Order>builder()
        .setBootstrapServers("kafka:9092")
        .setTopics("orders")
        .setValueOnlyDeserializer(new OrderDeser())
        .build(), WatermarkStrategy
            .<Order>forBoundedOutOfOrderness(Duration.ofSeconds(30))
            .withTimestampAssigner((o, ts) -> o.eventTs),
        "orders");

DataStream<Revenue> hourly = orders
    .filter(o -> "PAID".equals(o.status))
    .keyBy(o -> o.category)
    .window(TumblingEventTimeWindows.of(Time.hours(1)))
    .allowedLateness(Time.minutes(10))
    .aggregate(new SumAmount());

// catalog is a loaded FlinkCatalog; table must be resolved before sink construction
Table dest = catalog.loadTable(TableIdentifier.of("gold", "revenue_hourly"));
FlinkSink.forRow(hourly, RevenueSchema.INSTANCE)
    .table(dest)
    .build();
```

The `BoundedOutOfOrderness(30s)` watermark says "events may be up to 30s late." `allowedLateness(10m)` keeps window state for 10 more minutes so very-late events can still update.

== Exactly-Once Semantics

"Exactly-once" in streaming means: under any failure, each input event affects the output exactly once. It does *not* mean each event is processed exactly once internally (that would require a global lock).

Flink achieves it via *asynchronous distributed snapshots* (Chandy–Lamport variant). Barriers are injected into the stream at the source; each operator snapshots its state when all input barriers arrive, then forwards the barrier downstream. On failure, the entire job rewinds to the last completed checkpoint.

For external sinks, exactly-once requires a *transactional* sink:

- Kafka producer transactions: write + offset commit are atomic.
- $"JDBC"$: two-phase commit ($"XA"$) or idempotent upserts.
- Iceberg / Delta: atomic snapshot commit; pending files referenced only after commit.

== Engine Comparison

#table(
  columns: 4,
  [*System*], [*Model*], [*State*], [*Niche*],
  [Kafka Streams], [Library, partition $=$ task], [$"RocksDB"$ in-process], [Microservices on $"JVM"$],
  [Flink], [Dataflow, distributed], [$"RocksDB"$ + checkpoints], [Heavy stateful jobs],
  [Spark Structured Streaming], [Micro-batch, event-time], [$"RocksDB"$ / memory], [Unified with batch],
  [ksqlDB], [$"SQL"$ over Kafka Streams], [$"RocksDB"$], [Kafka-native $"SQL"$],
  [Materialize], [Differential dataflow], [In-memory arrangements], [Incremental views, low latency],
  [RisingWave], [Distributed $"SQL"$], [Tiered (S3-backed)], [Cloud-native streaming $"SQL"$],
)

== Streaming SQL

Differential dataflow (Materialize, RisingWave) maintains the answer to a $"SQL"$ query incrementally as inputs change. The engine tracks the partial derivative of the query w.r.t. each input.

```sql
-- Materialize / RisingWave
create source orders
  from kafka broker 'kafka:9092' topic 'orders'
  format avro using schema registry 'http://sr:8081';

create materialized view revenue_by_category as
select category, sum(amount) as revenue
from orders
where status = 'PAID'
group by category;
```

The view is kept current within milliseconds. Cost: every join input must fit in $"RAM"$ (Materialize) or tiered storage (RisingWave). Compare with `database/streaming-and-incremental-computation.typ` for the broader incremental-view-maintenance theory.

== Watermark Pitfalls

- *Idle partition stalling.* If one Kafka partition has no events, its watermark never advances and the global watermark (min across partitions) stalls. Configure *idle source timeout*.
- *Skewed event-time per partition.* Some sources emit hours behind others. Either tolerate it (`allowedLateness`) or buffer per-source up to the slowest.
- *Wall-clock windows.* `TumblingProcessingTimeWindows` is fine for $"SLO"$ dashboards but wrong for analytics that must agree with batch.

== Backpressure and Scaling

Flink's network stack uses credit-based flow control: each upstream task tracks free buffers downstream and only sends when credit is available. Backpressure propagates from the slow sink up to the source, naturally throttling reads from Kafka. Symptoms in $"UI"$: input rate falls, checkpoint duration grows, $"GC"$ time grows. Remedies: scale parallelism, increase $"RocksDB"$ block cache, partition the slow key.

== Pitfalls

- *Reading exactly-once as "at-most-once retries."* Exactly-once requires transactional sink + replayable source + deterministic operators.
- *Unbounded state.* Stateful operators ($"GROUP BY"$ without windows, deduplication) grow without bound; always add $"TTL"$.
- *Schema evolution.* Avro / Protobuf with a schema registry; do not parse $"JSON"$ ad-hoc.
- *Tight Kafka coupling.* If the only consumer is one Flink job, you may not need Kafka — write directly to Iceberg / Delta and let micro-batch Spark drive.

== Further Reading

Akidau, T. et al. (2015). "The Dataflow Model: A Practical Approach to Balancing Correctness, Latency, and Cost in Massive-Scale, Unbounded, Out-of-Order Data Processing." VLDB 8(12). Introduces the unified model of event time, watermarks, windowing, and triggers that underlies Apache Beam and Flink's streaming semantics.

Zaharia, M. et al. (2013). "Discretized Streams: Fault-Tolerant Streaming Computation at Scale." SOSP. Presents Spark Streaming's micro-batch model, showing how deterministic re-execution of batch intervals achieves exactly-once semantics and fault recovery.

Carbone, P. et al. (2015). "Apache Flink: Stream and Batch Processing in a Single Engine." IEEE Data Engineering Bulletin 38(4). Describes Flink's dataflow DAG model, asynchronous distributed snapshots (Chandy–Lamport variant), and unified batch/streaming execution.

Kreps, J. (2014). "The Log: What Every Software Engineer Should Know About Real-Time Data's Unifying Abstraction." LinkedIn Engineering Blog. Argues that the append-only log is the fundamental data structure underlying databases, replication, and stream processing systems.

McSherry, F., Murray, D., Isaacs, R., Isard, M. (2013). "Differential Dataflow." CIDR. Introduces a framework for incremental, iterative computation over changing data collections, the theoretical basis for Materialize and Neon's streaming SQL engines.

Budiu, M. et al. (2023). "DBSP: Automatic Incremental View Maintenance for Rich Query Languages." VLDB 16(7). Formalises the algebraic theory of incremental stream processing with a circuit model, providing a compositional foundation for streaming SQL.

Akidau, T., Chernyak, S., Lax, R. (2018). _Streaming Systems._ O'Reilly Media. Expands the Dataflow model paper into a full treatment of the what/where/when/how framework; the standard practitioner reference for stream-processing design.

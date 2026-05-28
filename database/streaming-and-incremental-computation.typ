= Streaming and Incremental Computation

Stream processing maintains continuously updated query results as new data arrives, without re-running the full query. *Differential dataflow* and *DBSP* provide formal foundations; Kafka Streams, Flink, and Spark Structured Streaming are industrial implementations.

*See also:* _Lakehouses and Open Table Formats_, _Weakly Consistent Systems_, _Time-Series and Graph Databases_

== The Streaming Model

*Stream:* an unbounded, ordered sequence of events with timestamps.
*Window:* a bounded view of a stream, enabling finite aggregations.

```
Event stream:
  t=1: {user=A, action="click", page="/home"}
  t=2: {user=B, action="buy",   item="X",  amount=50}
  t=3: {user=A, action="buy",   item="Y",  amount=30}
  t=4: {user=B, action="click", page="/cart"}
  ...

Query: "Revenue per user in the last 10 minutes" → continuously updated result
```

=== Window Types

#table(
  columns: (auto, auto, auto),
  [*Window*], [*Definition*], [*Example*],
  [Tumbling],   [Fixed-size, non-overlapping],  [Revenue per 5-minute bucket],
  [Sliding],    [Fixed-size, overlapping],       [Rolling 1-hour average, every minute],
  [Session],    [Gap-bounded by inactivity],     [User sessions (30-min gap = new session)],
  [Global],     [All events since start],        [Total revenue all time],
)

```python
# Tumbling window aggregation (manual implementation)
from collections import defaultdict
import time

def tumbling_window(events, window_size_sec: int):
    """Emit aggregates at the end of each window."""
    buckets = defaultdict(list)
    for event in events:
        bucket = int(event["ts"] // window_size_sec)
        buckets[bucket].append(event)

    for bucket_start, evts in sorted(buckets.items()):
        total = sum(e["amount"] for e in evts if "amount" in e)
        yield {
            "window_start": bucket_start * window_size_sec,
            "window_end":   (bucket_start + 1) * window_size_sec,
            "revenue":      total
        }
```

== Event Time vs Processing Time

*Event time:* the time the event actually occurred (in the data).
*Processing time:* the time the event arrived at the processor.

The difference = *latency* + *out-of-order delivery*. A GPS event from a tunnel may arrive seconds after events that logically followed it.

*Why it matters:*

```
Events (event time): [t=1, t=2, t=5, t=3, t=4]   ← t=3,4 arrived late
Processing order:    [t=1, t=2, t=5, t=3, t=4]

Tumbling window [0,5]:
  Processing-time window closes at wall-clock-T=5: sees events {1,2,5} → revenue=X
  → Misses t=3 and t=4 that were in the window but arrived late

Event-time window [0,5]:
  Window closes when watermark reaches t=5 (accepting bounded lateness)
  → Sees events {1,2,3,4,5} → revenue=X+Y (correct)
```

*Watermark:* a lower bound on event time. The system declares "I will not see events with event_time < W anymore" — triggering window close at event time W.

```python
class WatermarkTracker:
    def __init__(self, max_lateness_sec: float = 5.0):
        self.watermark    = 0
        self.max_lateness = max_lateness_sec

    def advance(self, event_time: float) -> float:
        self.watermark = max(self.watermark,
                             event_time - self.max_lateness)
        return self.watermark
```

== Apache Kafka Streams

Kafka Streams is a JVM library for building stateful stream processors. State is stored in RocksDB (local) + changelog topic (replicated). The canonical API is JVM-only, so the example below is in Java — non-JVM clients use the lower-level `librdkafka` consumer/producer in C++.

```java
// Kafka Streams: count events per user per 1-minute tumbling window
StreamsBuilder builder = new StreamsBuilder();

KStream<String, Event> events = builder.stream("events");

KTable<Windowed<String>, Long> counts = events
    .groupBy((key, event) -> event.getUserId())
    .windowedBy(TimeWindows.ofSizeWithNoGrace(Duration.ofMinutes(1)))
    .count(Materialized.as("event-counts-store"));

counts.toStream()
      .foreach((windowedKey, count) ->
          System.out.printf("user=%s window=%s count=%d%n",
              windowedKey.key(),
              windowedKey.window().startTime(),
              count));
```

*Exactly-once semantics:* Kafka Streams uses Kafka transactions to atomically commit state store updates + output topic offsets + consumer offsets.

== Apache Flink

Flink is the most capable open-source stream processor: event time, stateful computation, exactly-once, and SQL support.

```python
# Flink Table API (PyFlink)
from pyflink.table import TableEnvironment, EnvironmentSettings

env = TableEnvironment.create(EnvironmentSettings.in_streaming_mode())

# Define source
env.execute_sql("""
    CREATE TABLE orders (
        order_id    BIGINT,
        customer_id BIGINT,
        amount      DECIMAL(10,2),
        event_time  TIMESTAMP(3),
        WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic'     = 'orders',
        'format'    = 'json'
    )
""")

# Continuous aggregation with event-time tumbling window
result = env.sql_query("""
    SELECT
        customer_id,
        TUMBLE_START(event_time, INTERVAL '1' MINUTE) AS window_start,
        SUM(amount) AS revenue
    FROM orders
    GROUP BY
        customer_id,
        TUMBLE(event_time, INTERVAL '1' MINUTE)
""")
```

*Flink checkpointing:* periodically save the state of all operators to a distributed snapshot (Chandy-Lamport algorithm adapted for dataflows). On failure, restart from the last checkpoint — exactly-once guaranteed.

```
Flink checkpoint (every 10s):
  Barrier injected into source partition
  → Propagates through DAG like a watermark
  → Each operator saves its state (e.g., window aggregates) when barrier arrives
  → Coordinator confirms: snapshot at barrier timestamp is consistent and complete
  → On failure: reload state from snapshot, replay Kafka from the checkpoint offset
```

== Differential Dataflow

McSherry et al. (2013): a framework where computations are expressed as dataflow programs over *changes* (deltas) rather than full datasets. Each output changes only when its inputs change — true incremental computation.

*Collections:* multisets of (value, time, multiplicity) triples.
- Multiplicity +1: element added.
- Multiplicity -1: element removed.

```rust
// Differential dataflow (Rust) — social network triangle count
use differential_dataflow::input::Input;
use differential_dataflow::operators::*;

fn main() {
    timely::execute_from_args(std::env::args(), |worker| {
        let (mut input, probe) = worker.dataflow::<u64, _, _>(|scope| {
            let (handle, edges) = scope.new_collection::<(u32, u32), i64>();

            // Compute triangles: (a,b), (b,c), (a,c) all present
            let triangles = edges.join_map(
                &edges,
                |&b, &a, &c| (a, *c),   // (a,b) ⋈ (b,c) → (a,c)
            ).semijoin(&edges);           // filter: (a,c) must exist

            let count = triangles.count();
            count.inspect(|x| println!("triangles: {:?}", x));
            (handle, count.probe())
        });

        // Insert edges one at a time — only affected triangles update
        input.insert((1, 2)); worker.step();
        input.insert((2, 3)); worker.step();
        input.insert((1, 3)); worker.step();  // triangle (1,2,3) formed!
        // Output: "+1 triangle" — not a full recount
    }).unwrap();
}
```

*Why it's powerful:* for a graph with 1B edges and 1 new edge inserted, differential dataflow computes the new triangle count in time proportional to the number of affected triangles, not 1B.

== DBSP (Database Stream Processing)

DBSP (Budiu et al. 2023) generalizes differential dataflow to support arbitrary SQL queries with GROUP BY, JOINs, and subqueries over streams, with a formal proof of correctness for incremental maintenance.

*Core idea:* any relational algebra operator $Q$ has an *incremental version* $Delta Q$ that takes changes as input and produces changes as output:

$Delta Q(a, Delta a) = Q(a + Delta a) - Q(a)$

For SQL aggregations: if `SUM(x)` was 100 and new rows with `x = [10, -5]` arrive, `Δ SUM = 5` (no full re-aggregation).

```python
# DBSP-style incremental GROUP BY SUM
class IncrementalGroupBySum:
    def __init__(self):
        self.state = {}   # group_key → current sum

    def apply_delta(self, delta: list[tuple]) -> list[tuple]:
        """
        delta: list of (key, value_delta, multiplicity)
          multiplicity=+1: insert, -1: delete
        Returns: changes to the output collection
        """
        output_changes = {}
        for key, val, mult in delta:
            old = self.state.get(key, 0)
            self.state[key] = old + val * mult
            new = self.state[key]
            if old != new:
                output_changes[key] = (old, new)   # (before, after)
        return [(k, new - old) for k, (old, new) in output_changes.items()]

igs = IncrementalGroupBySum()
changes = igs.apply_delta([("US", 50, +1), ("CA", 30, +1)])
# changes: [("US", 50), ("CA", 30)]  — initial deltas
changes = igs.apply_delta([("US", 20, +1), ("US", 10, -1)])
# changes: [("US", 10)]  — net delta is +10 for US
```

== Spark Structured Streaming

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import window, sum as _sum

spark = SparkSession.builder.getOrCreate()

# Read from Kafka
orders = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "broker:9092") \
    .option("subscribe", "orders") \
    .load() \
    .selectExpr("CAST(value AS STRING) as json") \
    .select(from_json("json", schema).alias("data")) \
    .select("data.*")

# Windowed aggregation
agg = orders.groupBy(
    orders.customer_id,
    window(orders.event_time, "10 minutes", "5 minutes")  # sliding window
).agg(_sum("amount").alias("revenue"))

# Write to sink (Delta Lake for ACID)
agg.writeStream \
   .format("delta") \
   .outputMode("update") \
   .option("checkpointLocation", "s3://bucket/checkpoints/revenue") \
   .start("s3://bucket/streaming_revenue")
```

*Output modes:*
- `append`: emit only new rows (for append-only operations).
- `update`: emit changed rows (for windowed aggregations).
- `complete`: re-emit entire result table (for global aggregations).

== References

McSherry, F., Murray, D., Isaacs, R., Isard, M. (2013). "Differential Dataflow." CIDR.

Budiu, M. et al. (2023). "DBSP: Automatic Incremental View Maintenance for Rich Query Languages." VLDB.

Carbone, P. et al. (2015). "Apache Flink: Stream and Batch Processing in a Single Engine." IEEE Data Eng. Bull.

Zaharia, M. et al. (2016). "Apache Spark: A Unified Engine for Big Data Processing." CACM 59(11).

Akidau, T. et al. (2015). "The Dataflow Model: A Practical Approach to Balancing Correctness, Latency, and Cost." VLDB. (Beam model)

Kreps, J. (2014). "Questioning the Lambda Architecture." O'Reilly Radar.

= Watermarks and Windows

Unbounded streams force a question batch never faces: when is a result *complete enough* to emit? The Dataflow model (Akidau et al., 2015) decomposed the answer into four orthogonal choices — *what* is computed (the transformation), *where* in event time it is grouped (windowing), *when* results are emitted (watermarks and triggers), and *how* refinements relate (accumulation modes). This chapter works through that machinery: event vs processing time, watermark generation and propagation in Flink and the Beam model, window types, triggers, allowed lateness, and accumulation modes.

*See also:* _Streaming_ (engines and exactly-once mechanics), _Change Data Capture_ (a major source of timestamped streams), _Batch Processing_ (the bounded special case where the watermark jumps from $-infinity$ to $+infinity$).

== Two Clocks

Every event has an *event time* (when it happened, stamped at the source) and a *processing time* (the wall clock when an operator observes it). In an ideal system they coincide; in reality, network buffering, mobile clients syncing after a flight, partitioned brokers, and backpressure introduce *skew* that is unbounded and varies over time. Plotting processing time against event time gives the skew diagram: the ideal is the diagonal; the actual progress line wanders above it, and the horizontal gap at any point is the current event-time lag.

The consequence: any computation grouped by event time (revenue per hour, sessions per user) over data ordered by processing time must *reorder* and must decide how long to wait for stragglers. Waiting forever gives correctness with infinite latency; not waiting gives bounded latency with missing data. Watermarks make this trade-off explicit and tunable.

== Watermarks

A watermark is a monotonically non-decreasing assertion flowing through the dataflow: $W(t)$ at an operator means "expect no further events with event time $<= t$." Watermarks let an engine declare event-time windows *complete*: a window $[s, e)$ can be finalised once the watermark passes $e$.

Two ideal forms exist only in theory: a *perfect watermark* (never wrong, requires omniscient knowledge of the source) and purely *heuristic watermarks* used in practice. A heuristic watermark that advances too slowly adds latency; too fast, and correct events become *late data*. Common generation strategies:

- *Bounded out-of-orderness.* $W = max("event_ts seen") - delta$ for a fixed slack $delta$ (Flink's `forBoundedOutOfOrderness(Duration.ofSeconds(30))`). Simple, by far the most used; $delta$ is a guess that should be informed by measuring the actual lateness distribution.
- *Source-aware watermarks.* Kafka sources track the maximum timestamp *per partition* and emit the minimum, since each partition is roughly ordered. Pub/Sub in Dataflow computes watermarks from broker-side backlog statistics.
- *Punctuated watermarks.* The producer embeds explicit "end of period" markers (e.g., a device emits "log closed for hour H"); the watermark advances on markers rather than heuristics.

== Propagation

Watermarks travel *with the data* through the operator graph. The rules, identical in essence in Flink and Beam:

- An operator with multiple inputs takes the *minimum* of its input watermarks: it cannot promise more than its slowest input.
- An operator's output watermark is its input watermark minus whatever it still holds in state (buffered windows, pending timers). Beam formalises this as input watermark vs output watermark per stage, with the gap being the stage's event-time latency.
- Within Flink, watermarks are broadcast to all output channels and merged with `min` at each receiving task, including across shuffles.

The `min` rule creates the classic failure mode: *one stalled input stalls everything*. An idle Kafka partition, a quiet region, or one slow CDC table holds the global watermark at its last value, and no window downstream ever closes. Flink's answer is idleness detection (`withIdleness(Duration.ofMinutes(1))`): a source split that produces nothing for the timeout is marked idle and excluded from the `min` until it resumes. The dual hazard: a marked-idle partition that wakes up with old timestamps produces instantly-late data. Flink 1.15+ also added *watermark alignment*, which throttles sources that run too far ahead of the group, bounding the in-flight state that fast partitions would otherwise pile up while waiting for slow ones.

== Window Types

A window assigner maps each element to one or more windows:

- *Tumbling (fixed).* Aligned, non-overlapping intervals of size $s$: element with timestamp $t$ goes to window $floor(t / s) dot s$. One window per element. The default for periodic reports.
- *Sliding (hopping).* Size $s$, slide $p < s$: each element lands in $s / p$ overlapping windows ("5-minute average, updated every minute" gives 5 windows per element — state and output multiply accordingly).
- *Session.* Data-driven: each element opens a window of `[t, t + gap)`; overlapping per-key windows *merge*. Window boundaries are unknowable in advance, which is why session windows require merging window support and are the canonical hard case for watermark reasoning (a window is complete only when the watermark passes last-event + gap).
- *Global.* One window per key spanning all time; meaningful only with a custom trigger, used for count-based and custom-policy windowing.

Beam adds custom `WindowFn`s (calendar windows, keyed session gaps); Flink additionally exposes the lower-level primitive underneath all of them: *keyed state plus event-time timers* in a `KeyedProcessFunction`, which is what you drop to when no assigner fits.

== Triggers

The watermark answers "when is the window complete?"; the *trigger* answers "when do we materialise output?" — and the two are deliberately decoupled, because for a 24-hour window nobody wants to wait 24 hours for the first number. The Beam trigger algebra composes:

- `AfterWatermark.pastEndOfWindow()` — the completeness trigger; fires once when the watermark passes the window end (the *on-time* pane).
- `.withEarlyFirings(AfterProcessingTime.pastFirstElementInPane().plusDelayOf(60s))` — speculative panes every minute before completion.
- `.withLateFirings(AfterPane.elementCountAtLeast(1))` — a corrective pane per late element after completion.

Each emitted result is a *pane* tagged with its timing (`EARLY`, `ON_TIME`, `LATE`) and index, so downstream consumers can distinguish a speculative number from a final one. Flink's DataStream API has a simpler built-in set (`EventTimeTrigger` is the default; `ContinuousEventTimeTrigger`, `CountTrigger`, custom `Trigger` subclasses), and Flink SQL exposes early/late firing only via configuration options — the full algebra is a Beam-model concept.

== Allowed Lateness

When the watermark passes a window's end, the engine could drop the window's state immediately — but then every late element is lost. *Allowed lateness* keeps window state alive for an extra grace period $l$: state for window $[s, e)$ is garbage-collected at watermark $e + l$. Within the grace period, a late element re-fires the trigger and emits a corrected pane; after it, late elements are dropped — Flink can route them to a *side output* for offline reconciliation:

```java
OutputTag<Event> lateTag = new OutputTag<>("late") {};
SingleOutputStreamOperator<Agg> result = events
    .keyBy(e -> e.key)
    .window(TumblingEventTimeWindows.of(Time.hours(1)))
    .allowedLateness(Time.minutes(10))
    .sideOutputLateData(lateTag)
    .aggregate(new SumAgg());
DataStream<Event> dropped = result.getSideOutput(lateTag);
```

The cost model: state retention is proportional to (window size + allowed lateness) $times$ key cardinality. Ten minutes of lateness on hourly windows is cheap; 7 days of lateness on minute windows over a billion keys is not. The honest framing: allowed lateness moves the completeness/latency trade-off from "drop late data" to "pay state for corrections," and the side output is the audit trail for what was still dropped.

== Accumulation Modes

When a trigger fires multiple panes for one window, what does each pane contain? The Beam model defines three modes:

- *Discarding.* Each pane contains only elements since the previous firing. Panes are deltas; correct iff the downstream consumer sums them. Cheapest in state.
- *Accumulating.* Each pane contains the full refined result so far. Correct for sinks that overwrite by key (upsert into a database keyed by window); the common default.
- *Accumulating and retracting.* Each pane carries the new result *plus a retraction* of the previous one, so downstream aggregations that already consumed the old value can subtract it. Essential when panes feed a second grouping (re-windowed or re-keyed aggregates); this is exactly the changelog semantics Flink SQL emits for updating results (`+I`/`-U`/`+U`/`-D` rows) and the foundation of incremental view maintenance.

Choosing wrong is a correctness bug, not a tuning issue: accumulating panes summed downstream double-count; discarding panes upserted downstream under-count.

== Putting It Together

A worked configuration for hourly revenue with speculative updates, in Beam terms: tumbling 1-hour windows; watermark from Kafka per-partition timestamps with 30 s bound and 1-minute idleness; trigger = on-watermark with early firings every minute and late firings per element; allowed lateness 15 minutes; accumulating mode into an upsert sink keyed by `(category, window_end)`. The dashboard sees a number within a minute of the hour starting, the number converges through the hour, finalises shortly after the hour ends, and self-corrects for 15 minutes; events later than that land in the side output and are reconciled by the nightly batch job — the streaming/batch agreement check from _Data Quality_ closes the loop.

== Pitfalls

- *Stalled watermark from one idle partition.* The `min` rule means a single quiet source freezes every downstream window. Configure idleness handling and alert on watermark lag per source.
- *Watermark slack as folklore.* `delta = 30s` copied from a tutorial; measure the real lateness CDF (e.g., 99.9th percentile lateness) and set $delta$ from it, then watch the side-output rate.
- *Processing-time windows that must agree with batch.* They never will; reprocessing the same Kafka data tomorrow yields different processing-time windows. Event time for anything reconciled or replayed.
- *Sliding windows with tiny slide.* Size 1 h, slide 1 s means 3600 windows per element; use an incremental aggregate over tumbling panes or a custom process function instead.
- *Session windows plus high lateness.* Merging windows can resurrect and re-merge sessions on late data, churning state; cap the gap and lateness deliberately.
- *Timestamps assigned downstream of a shuffle.* Watermarks generated after reordering no longer bound anything; assign timestamps and watermarks as close to the source as possible.

== Further Reading

Akidau, T. et al. (2015). "The Dataflow Model: A Practical Approach to Balancing Correctness, Latency, and Cost in Massive-Scale, Unbounded, Out-of-Order Data Processing." VLDB 8(12). The what/where/when/how decomposition; windows, triggers, and accumulation modes as orthogonal axes.

Akidau, T., Chernyak, S., Lax, R. (2018). _Streaming Systems._ O'Reilly. The book-length treatment, including the watermark skew diagrams and the perfect-vs-heuristic watermark distinction.

Akidau, T. et al. (2021). "Watermarks in Stream Processing Systems: Semantics and Comparative Analysis of Apache Flink and Google Cloud Dataflow." VLDB 14(12). Formal semantics of watermark generation and propagation, comparing the two production implementations.

Carbone, P. et al. (2015). "Apache Flink: Stream and Batch Processing in a Single Engine." IEEE Data Engineering Bulletin 38(4). Flink's dataflow model in which watermarks travel as stream elements.

Li, J. et al. (2008). "Out-of-Order Processing: A New Architecture for High-Performance Stream Systems." VLDB. The pre-Dataflow formulation of progress indicators ("low watermarks") for out-of-order streams.

Tucker, P., Maier, D., Sheard, T., Fegaras, L. (2003). "Exploiting Punctuation Semantics in Continuous Data Streams." IEEE TKDE 15(3). Origin of punctuations, the ancestor of the watermark concept.

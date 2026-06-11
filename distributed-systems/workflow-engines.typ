= Workflow Engines

Modern distributed applications frequently need to coordinate multi-step processes that span hours, days, or weeks — processes that must survive server restarts, handle partial failures gracefully, and retry individual steps without re-executing completed work. *Workflow engines* solve this by externalising durable state and encoding execution semantics in a framework rather than in application code. This chapter covers the durable execution model, the leading engines (Temporal, Airflow, Prefect, Dagster), the relationship to sagas, and failure handling patterns.

*See also:* _Distributed Transactions_, _Failure Detection_, _Consensus Deep Dive_

== The Durable Execution Model

In a traditional microservice, in-flight state lives in process memory. A crash loses it. The application must reconstruct state by polling databases or relying on external retry logic — logic that is usually ad-hoc, under-tested, and subtly broken.

*Durable execution* externalises the execution state: every step's input, output, and control flow decision is persisted to a durable log before the step result is used. On restart, the engine replays the log to restore the in-memory state without re-executing side effects. The application code is written as if it runs in a single long-lived process; the engine provides the illusion.

Three invariants of the durable execution model:

- *Completeness:* every activity (side-effectful step) executes at least once and its result is recorded.
- *Idempotent replay:* replaying the history into the workflow function must produce the same sequence of decisions as the original execution.
- *Isolation of side effects:* all external I/O (API calls, DB writes, sends) occurs inside explicitly-declared activities, never in the workflow function body.

== Temporal

*Temporal* (Maxim Fateev, Samar Abbas — forked from Cadence, 2019) is a durable execution platform. A Temporal cluster stores workflow state; application code runs in *workers* that the cluster orchestrates.

=== Core Concepts

A *workflow* is a function written in an ordinary programming language (Go, Java, Python, TypeScript, .NET). It coordinates activities and maintains durable local variables. A workflow must be *deterministic* — given the same history, it must make the same decisions on replay.

An *activity* is a function that performs I/O or other side effects. Activities are retried automatically on failure (with configurable retry policies). Their results are stored in the workflow history before the workflow function sees them.

A *worker* is an application process that polls one or more *task queues* for workflow and activity tasks, executes them, and returns results to the Temporal cluster. Workers are stateless and horizontally scalable; the cluster assigns tasks to available workers.

```
Temporal cluster:
  - History service: stores events, drives workflow state machines
  - Matching service: routes tasks to task queues
  - Frontend service: gRPC API for clients and workers

Worker process:
  for task in poll(task_queue):
    if task.type == WORKFLOW:
      replay_and_execute(workflow_fn, task.history)
    elif task.type == ACTIVITY:
      execute(activity_fn, task.input)
      report_completion(task.token, result)
```

=== History Replay

When a worker picks up a workflow task, it replays the full *event history* through the workflow function. During replay, calls to activity stubs, timers, and signals are intercepted: if the event already exists in history, the recorded result is returned immediately without re-executing. If the event does not yet exist, a command is scheduled and the workflow suspends.

This is the *event sourcing* pattern applied to control flow. The history is the single source of truth; the in-memory workflow state is a derivative.

=== Signals, Queries, and Schedules

*Signals* are asynchronous messages sent to a running workflow. The workflow can block on a signal (via a coroutine yield or blocking call on the signal channel) and resume when it arrives. Signals are durably recorded in the history.

*Queries* are synchronous reads of workflow state. They do not mutate history; the worker replays the history to compute the current state and returns it.

*Schedules* (Temporal 1.18+) replace external cron jobs: the cluster itself manages recurring workflow launches with configurable overlap policies, catch-up limits, and jitter.

=== Versioning

Because history replay must be deterministic, changing workflow code that is still executing is dangerous — a replayed history may encounter a different branch than the original. Temporal provides a `get_version` (Go/Java) / `patched` (Python) API:

```python
v = workflow.patched("add-validation-step")
if v:
    result = await workflow.execute_activity(validate, args)
```

Old histories take the `False` branch; new histories take the `True` branch. This allows in-place code changes without draining all open executions.

== Apache Airflow

*Apache Airflow* (Airbnb, 2014; Apache top-level 2019) is a platform for authoring, scheduling, and monitoring *data pipelines* as code. Pipelines are defined as *Directed Acyclic Graphs* (DAGs) of tasks.

=== DAG Model

A *DAG* in Airflow is a Python file that instantiates `DAG` and `Operator` objects and defines dependencies:

```python
with DAG("order_pipeline", schedule="@daily", start_date=datetime(2025, 1, 1)) as dag:
    extract = PythonOperator(task_id="extract", python_callable=extract_fn)
    transform = PythonOperator(task_id="transform", python_callable=transform_fn)
    load = PythonOperator(task_id="load", python_callable=load_fn)
    extract >> transform >> load
```

Each *task instance* corresponds to one (DAG, task, execution date) tuple. Airflow tracks state in a relational database (PostgreSQL or MySQL).

=== Scheduler and Executor Types

The *scheduler* is responsible for parsing DAG files, triggering DAG runs at the correct time, and placing task instances on the executor queue.

The *executor* determines how tasks are actually run:

#table(
  columns: (auto, 1fr, 1fr),
  table.header[*Executor*][*Mechanism*][*Use case*],
  [LocalExecutor], [Subprocess on scheduler node], [Development, small scale],
  [CeleryExecutor], [Celery workers via message broker], [Multi-node, widely deployed],
  [KubernetesExecutor], [One pod per task instance], [Isolation, bursty workloads],
  [CeleryKubernetesExecutor], [Hybrid: Celery default, K8s for tagged tasks], [Mixed workloads],
)

With `KubernetesExecutor`, each task gets a fresh pod with configurable resources and Docker image, providing strong isolation but higher startup latency (typically 5–30 s per task).

=== XCom and Sensors

*XCom* (cross-communication) allows tasks to push and pull small values (serialised Python objects) via the metadata database. It is not designed for large data; large results should be stored externally (S3, GCS) and only the path pushed via XCom.

*Sensors* are special operators that poke an external condition (file existence, HTTP endpoint, Kafka message) on a schedule and block the DAG until the condition is met. Long-running sensors hold a worker slot; `mode="reschedule"` releases the slot between polls.

== Prefect and Dagster

=== Prefect

*Prefect* (2018, rewritten as Prefect 2 / "Orion" in 2022) emphasises developer experience. Workflows are plain Python functions decorated with `@flow`; tasks are `@task`. The Prefect server (or Prefect Cloud) stores run metadata but does not execute code — execution happens wherever the flow runs (local process, Docker container, Kubernetes job, serverless function).

Prefect 2 uses *deployment artifacts* that package a flow with its infrastructure and parameter schema. Schedules and manual triggers launch runs via *work pools* that pull from a queue.

=== Dagster

*Dagster* (2018) introduces an *asset-centric model*: the primary abstraction is a *software-defined asset* (SDA) — a named, versioned artifact (a database table, a file, an ML model) produced by a computation. Pipelines are implicitly defined by the dependency graph of assets.

Key advantages of the asset model:
- *Observability:* the lineage graph is the data graph, not just the computation graph.
- *Materialisation policies:* assets can be re-materialised lazily (only when stale), eagerly, or on schedule.
- *Data quality:* asset checks express expectations (e.g., row count, schema) alongside the asset.

Dagster uses *Ops* for imperative computation and *Resources* for injectable infrastructure (database connections, API clients), promoting testability.

== Sagas vs Workflows

Both sagas and workflow engines address long-running business processes. Their scopes differ:

A *saga* (Garcia-Molina and Salem 1987; Richarson 2018 popularisation) is a sequence of local transactions, each with a corresponding *compensating transaction* that semantically undoes it. Sagas provide *ACI* (no isolation) semantics: intermediate states are visible. They are implemented either *choreographically* (each service reacts to events) or *orchestrationally* (a coordinator issues commands).

A *workflow engine* is an orchestration framework that may implement sagas as one pattern among many. Workflow engines provide richer primitives: signals, timers, versioning, queries, and nested sub-workflows. The engine handles retry, replay, and state persistence; the saga logic is just application code.

#table(
  columns: (1fr, 1fr, 1fr),
  table.header[*Aspect*][*Saga (choreography)*][*Workflow engine*],
  [State location], [Distributed across services], [Centralised in engine history],
  [Observability], [Requires event tracing], [Built-in timeline view],
  [Compensation], [Domain-coded per service], [Domain-coded, engine retries],
  [Coupling], [Low (event-driven)], [Higher (centralised orchestrator)],
  [Complexity at scale], [Hard to reason about ordering], [History is auditable],
)

Use sagas (choreography) when teams are fully autonomous and event-driven coupling is acceptable. Use a workflow engine when you need observability, complex branching, long human-approval steps, or when saga debugging becomes intractable.

== Failure Handling Patterns

=== Retry Policies

Every activity in a workflow engine should declare a retry policy specifying maximum attempts, initial interval, backoff coefficient, and maximum interval:

```python
retry = RetryPolicy(
    maximum_attempts=10,
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    non_retryable_error_types=["InvalidInputError"],
)
```

Exponential backoff with jitter prevents thundering-herd retries against a recovering dependency.

=== Compensation

When a workflow cannot proceed, it may need to undo already-completed steps. Compensation activities are invoked explicitly:

```python
compensations = []
try:
    await reserve_inventory(order)
    compensations.append(release_inventory)
    await charge_payment(order)
    compensations.append(refund_payment)
    await ship_order(order)
except Exception:
    for comp in reversed(compensations):
        await comp(order)   # each comp is itself retried on failure
    raise
```

Compensation is not rollback — it is a new forward action that semantically reverses the effect.

=== Heartbeating

Long-running activities (video encoding, ML training, bulk data export) must periodically call `heartbeat()` with optional progress state. If the worker crashes, the cluster detects the missed heartbeat and reschedules the activity on another worker. The new worker can retrieve the last heartbeat's state and resume from a checkpoint rather than restarting from scratch.

```python
async def export_activity(ctx, table_name):
    last_offset = ctx.info().heartbeat_details or 0
    for batch in read_table(table_name, start=last_offset):
        process(batch)
        last_offset += len(batch)
        activity.heartbeat(last_offset)   # survives worker crash
```

== Determinism Constraints

Workflow function code runs in a deterministic sandbox. Violations cause history replay divergence and corrupt workflow state. Forbidden in workflow code:

- *External I/O:* no network calls, database reads, or file access — use activities.
- *Random values:* `random.random()`, `uuid.uuid4()` — use `workflow.now()` and workflow-seeded random APIs.
- *Wall-clock time:* `datetime.now()` — use `workflow.now()` which returns the event-history timestamp.
- *Non-deterministic iteration:* iterating over unordered sets or dicts (Python 3.7+ dicts are ordered; sets are not).
- *Threading / async tasks spawned outside framework:* only use framework-provided coroutine/child-workflow mechanisms.

Temporal's SDK provides lint rules and workflow sandboxing (Python) that detect many violations at development time.

== Worked Example: Order-Processing Workflow in Temporal

The following pseudocode (Python-style) shows a complete order workflow with compensation, a human-approval signal, and heartbeated fulfilment.

```python
@workflow.defn
class OrderWorkflow:
    def __init__(self):
        self.approved = False

    @workflow.signal
    def approve(self):
        self.approved = True

    @workflow.run
    async def run(self, order: Order) -> str:
        compensations = []
        try:
            # Step 1: reserve inventory
            await workflow.execute_activity(
                reserve_inventory, order,
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(maximum_attempts=5),
            )
            compensations.append((release_inventory, order))

            # Step 2: charge payment
            await workflow.execute_activity(
                charge_payment, order,
                start_to_close_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(maximum_attempts=3,
                    non_retryable_error_types=["CardDeclinedError"]),
            )
            compensations.append((refund_payment, order))

            # Step 3: wait for human approval (high-value orders)
            if order.total > 10_000:
                await workflow.wait_condition(
                    lambda: self.approved,
                    timeout=timedelta(days=2),
                )

            # Step 4: fulfil with heartbeating
            await workflow.execute_activity(
                fulfil_order, order,
                schedule_to_close_timeout=timedelta(hours=4),
                heartbeat_timeout=timedelta(minutes=5),
            )
            return "completed"

        except Exception as e:
            for fn, args in reversed(compensations):
                await workflow.execute_activity(fn, args,
                    retry_policy=RetryPolicy(maximum_attempts=10))
            return f"failed: {e}"
```

This 40-line function encodes retry, compensation, a durable timer-backed human-approval gate, and crash recovery — capabilities that would require hundreds of lines of ad-hoc state-machine code in a traditional service.

== Further Reading

Fateev, M., Abbas, S. (2020). "Temporal: Open Source Durable Execution." Temporal Technologies whitepaper.

Garcia-Molina, H., Salem, K. (1987). "Sagas." ACM SIGMOD.

Richardson, C. (2018). "Microservices Patterns." Manning. (Sagas, Chapter 4.)

Airflow documentation. https://airflow.apache.org/docs/

Dagster documentation. https://docs.dagster.io/

Temporal documentation. https://docs.temporal.io/

Stopford, B. (2018). "Designing Event-Driven Systems." O'Reilly. (Workflow patterns in event-driven architectures.)

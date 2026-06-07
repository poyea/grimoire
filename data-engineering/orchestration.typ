= Orchestration

An orchestrator schedules and supervises the $"DAG"$ of jobs that move and transform data. The classical concern is "run this Spark job at 02:00, then this dbt job, then this load." Modern systems extend that with backfill management, asset awareness, durable execution, and event-driven triggers. The four major schools are Airflow (task-centric), Dagster (asset-centric), Prefect (Python-first dynamic), and Temporal (durable workflows).

*See also:* _ETL vs ELT_ (what the orchestrator runs), _Batch Processing_, _Streaming_ (for hybrid event $+$ batch flows), _Workflow Engines_ (distributed-systems framing).

== What an Orchestrator Provides

- *Scheduling.* Cron, sensors, event triggers.
- *Dependency execution.* Task B starts when task A succeeds; partition-aware so 30 days backfill in parallel.
- *Retries and timeouts.* Per-task policy.
- *State and history.* Audit which run produced which output.
- *Backfill and replay.* Re-run subset of $"DAG"$ for past dates.
- *Observability.* Logs, $"UI"$, $"SLA"$ alerts.

== Airflow: Task-Centric DAGs

Airflow models pipelines as $"DAG"$s of *operators*. The scheduler polls the metadata $"DB"$, decides which task instances are runnable, and dispatches them to workers (Celery, Kubernetes, Local).

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.emr import EmrAddStepsOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "data-platform",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "sla": timedelta(hours=2),
}

with DAG(
    "daily_revenue",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    max_active_runs=1,
) as dag:

    ingest = EmrAddStepsOperator(
        task_id="spark_ingest_orders",
        job_flow_id="{{ var.value.emr_cluster }}",
        steps=[{
            "Name": "ingest",
            "ActionOnFailure": "CONTINUE",
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": ["spark-submit", "--deploy-mode", "cluster",
                         "s3://jobs/ingest.py",
                         "--dt", "{{ ds }}"],
            },
        }],
    )

    def run_dbt(**ctx):
        import subprocess
        subprocess.check_call(
            ["dbt", "run", "--select", "tag:revenue",
             "--vars", f"{{run_date: {ctx['ds']}}}"],
            cwd="/opt/dbt")

    transform = PythonOperator(task_id="dbt_run", python_callable=run_dbt)

    ingest >> transform
```

The `{{ ds }}` Jinja macro is the *logical date* of the run. Airflow's gold idea is that a $"DAG"$ run is keyed by `(dag_id, logical_date)` — re-running the same date is idempotent at the orchestrator layer.

Pain points: $"DAG"$s are imperative Python (hard to test), dynamic $"DAG"$s require generating tasks at parse time, and task boundaries cut across logical asset boundaries.

== Dagster: Asset-Centric

Dagster inverts the model: declare *assets* (the tables, files, models) and Dagster derives the $"DAG"$ from dependencies. The scheduler knows _what_ exists, when it was materialized, and from which upstream version.

```python
from dagster import asset, AssetIn, Definitions, ScheduleDefinition, define_asset_job

@asset(partitions_def=daily_partitions, group_name="bronze")
def raw_orders(context):
    dt = context.partition_key
    extract_orders_to_s3(dt)
    return f"s3://bronze/orders/dt={dt}/"

@asset(ins={"raw_orders": AssetIn()}, partitions_def=daily_partitions)
def stg_orders(context, raw_orders):
    spark_clean(raw_orders, f"s3://silver/orders/dt={context.partition_key}/")

@asset(ins={"stg_orders": AssetIn()}, partitions_def=daily_partitions)
def fct_revenue(context, stg_orders):
    dbt_run("fct_revenue", dt=context.partition_key)

defs = Definitions(
    assets=[raw_orders, stg_orders, fct_revenue],
    schedules=[ScheduleDefinition(
        job=define_asset_job("daily", selection="*fct_revenue"),
        cron_schedule="0 2 * * *",
    )],
)
```

A change to `stg_orders` automatically marks `fct_revenue` as *stale*. Lineage is first-class. Backfill is just "materialize asset X for partition range Y."

== Prefect: Dynamic Python

Prefect treats flows as ordinary Python functions decorated with `@flow` / `@task`. Dynamism (loops, conditionals creating tasks at runtime) is natural.

```python
from prefect import flow, task

@task(retries=3, retry_delay_seconds=60)
def load_partition(dt: str):
    spark_submit(f"s3://jobs/ingest.py --dt {dt}")

@flow
def backfill(start: str, end: str):
    for dt in pd.date_range(start, end).strftime("%Y-%m-%d"):
        load_partition.submit(dt)

if __name__ == "__main__":
    backfill("2026-01-01", "2026-01-31")
```

Prefect's deployment model decouples flow code from infrastructure (work pools, queues). Best when pipelines are dynamic and Python-native.

== Temporal: Durable Workflows

Temporal is not a data orchestrator per se but a *durable execution* runtime. Workflows are deterministic Python / Go / Java functions whose state is persisted at every `await`; if a worker dies, another resumes from the same line.

```python
from temporalio import workflow, activity

@activity.defn
async def spark_step(dt: str) -> str:
    return submit_emr_step(dt)

@workflow.defn
class IngestWorkflow:
    @workflow.run
    async def run(self, dt: str) -> None:
        await workflow.execute_activity(
            spark_step, dt,
            start_to_close_timeout=timedelta(hours=2),
            retry_policy=RetryPolicy(maximum_attempts=5))
```

Temporal shines for long-running per-entity workflows (order fulfillment, user onboarding); Airflow / Dagster handle scheduled bulk transformations more naturally.

== Comparison

#table(
  columns: 5,
  [*System*], [*Unit*], [*Model*], [*State store*], [*Best for*],
  [Airflow], [Task], [Imperative $"DAG"$, scheduler-driven], [Postgres metadata], [Scheduled $"ETL"$ on legacy stacks],
  [Dagster], [Asset], [Declarative, lineage-first], [Postgres + storage], [Modern lakehouse $"ELT"$],
  [Prefect], [Task / Flow], [Dynamic Python], [Cloud / self-host], [Python-heavy pipelines],
  [Temporal], [Workflow], [Durable execution], [Cassandra / Postgres], [Per-entity, long-running],
)

== Backfill Semantics

The bar for "good" backfill: re-run any window without manual partitioning by hand, in parallel, without re-running unrelated downstreams.

Airflow: `airflow dags backfill -s 2026-01-01 -e 2026-01-31 daily_revenue` (set `max_active_runs` and `pool` to throttle).

Dagster: from the $"UI"$ or $"CLI"$, select asset $+$ partition range; Dagster fans out automatically and only re-materializes the selected slice.

The deeper question — *what* to backfill — is a contract question. A downstream marketing dashboard may want full historical restatement, while a feature store may only want last 30 days. Encode that in asset-group policies.

== Sensors and Event Triggers

Polling cron is wasteful. Modern $"DAG"$s start on events:

- $"S3"$ object created → ingest partition.
- Kafka topic offset advanced → trigger downstream batch.
- Upstream asset materialized → run dependent asset.

Dagster auto-materialization policies (rules like "materialize when any parent updates") subsume cron for most assets.

== Pitfalls

- *Treating the orchestrator as a job queue.* A $"DAG"$ with 10k tasks per run will overwhelm Airflow's scheduler. Push fan-out into the data engine (Spark dynamic partition discovery) and keep $"DAG"$s coarse.
- *Side effects in $"DAG"$ files.* Airflow parses $"DAG"$ files frequently; do not call $"DB"$ / $"S3"$ at module import.
- *Retry on non-idempotent tasks.* Always design tasks to be safely re-runnable for the same logical key.
- *Hidden coupling via shared state.* If two $"DAG"$s read each other's outputs without declared dependencies, you have a lineage gap. Move to asset-centric.

== Further Reading

Beauchemin, M. (2017). "The Rise of the Data Engineer." Medium.

Dagster documentation, https://docs.dagster.io.

Apache Airflow documentation, https://airflow.apache.org/docs/.

Temporal documentation, https://docs.temporal.io.

Prefect documentation, https://docs.prefect.io.

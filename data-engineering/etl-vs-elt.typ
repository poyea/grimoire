#import "../template.typ": xref

= ETL vs ELT

$"ETL"$ (extract, transform, load) and $"ELT"$ (extract, load, transform) describe where transformation runs. In the warehouse era of the 1990s and 2000s, compute on the target system was scarce and transformation happened in a separate engine (Informatica, Talend) before loading curated rows. The cloud warehouse and lakehouse era inverted that economics: storage is cheap, compute is elastic, and $"SQL"$ engines (BigQuery, Snowflake, Databricks, Trino) can run transformations on raw data at scale.

*See also:* #xref("data-engineering", "batch-processing", label: "Batch Processing") (the engines that run the transforms), #xref("data-engineering", "lakehouse-engineering", label: "Lakehouse Engineering") (the storage layer that made $"ELT"$ cheap), #xref("data-engineering", "orchestration", label: "Orchestration") (how the steps are scheduled), #xref("database", "oltp-vs-olap", label: "OLTP vs OLAP") (database framing).

== A Concrete Distinction

Consider ingesting raw clickstream into an analytical model `fct_session`.

#table(
  columns: 3,
  [*Stage*], [*ETL*], [*ELT*],
  [Extract], [API / log scrape], [API / log scrape],
  [Transform], [Spark or Informatica job builds curated rows _before_ load], [Raw lands as-is in bronze table],
  [Load], [Only curated rows written to warehouse], [Raw written first, transformation runs _inside_ warehouse / lake],
  [Reprocessing], [Re-run external job; reload], [Re-run $"SQL"$ on retained raw],
  [Schema], [Decided upfront], [Late-bound, evolves with `MERGE` / `ALTER`],
)

The $"ELT"$ pattern preserves the raw record, which is what makes backfill, lineage, and contract debugging tractable.

== A Minimal ELT in dbt

dbt has become the de-facto $"ELT"$ transformation runtime: it compiles Jinja-templated $"SQL"$ into a $"DAG"$ of materializations.

```sql
-- models/staging/stg_events.sql
{{ config(materialized='view') }}

select
  cast(event_id   as string)     as event_id,
  cast(user_id    as bigint)     as user_id,
  cast(ts         as timestamp)  as event_ts,
  cast(properties as json)       as properties
from {{ source('raw', 'events') }}
where ts >= dateadd(day, -30, current_date)
```

```sql
-- models/marts/fct_session.sql
{{ config(
    materialized = 'incremental',
    unique_key   = 'session_id',
    on_schema_change = 'append_new_columns'
) }}

with sessionized as (
  select
    user_id,
    event_ts,
    sum(case when gap > interval '30 minutes' then 1 else 0 end)
      over (partition by user_id order by event_ts) as session_idx
  from (
    select
      user_id,
      event_ts,
      event_ts - lag(event_ts) over (
        partition by user_id order by event_ts
      ) as gap
    from {{ ref('stg_events') }}
    {% if is_incremental() %}
      where event_ts >= (select max(session_start) from {{ this }}) - interval '1 day'
    {% endif %}
  )
)
select
  md5(cast(user_id as varchar) || '|' || cast(session_idx as varchar)) as session_id,
  user_id,
  min(event_ts) as session_start,
  max(event_ts) as session_end,
  count(*)      as event_count
from sessionized
group by user_id, session_idx
```

`is_incremental()` toggles the `where` clause so a full refresh and an incremental run share one file. The watermark (`max(session_start) - 1 day`) is intentionally loose so out-of-order events still join.

== Decision Matrix

#table(
  columns: 3,
  [*Driver*], [*Prefer ETL*], [*Prefer ELT*],
  [Target compute cost], [Expensive / on-prem warehouse], [Cheap elastic compute],
  [$"PII"$ at source], [Mask before load], [Land raw in restricted bronze zone],
  [Schema stability], [Stable, known upfront], [Evolving, third-party],
  [Reprocessing frequency], [Rare], [Common (model changes weekly)],
  [Tooling], [Informatica, Talend, custom Spark], [dbt, Dataform, SQLMesh],
)

A common production answer is *EtLT*: a small pre-load transform (decryption, masking, format normalization) is cheaper than rewriting petabytes inside the warehouse, while the bulk of business logic stays in $"SQL"$.

== Idempotency and the Merge Pattern

Every pipeline must tolerate replays. The canonical $"ELT"$ idempotent write is `MERGE`:

```sql
merge into analytics.fct_orders t
using staging.orders_delta s
  on t.order_id = s.order_id
when matched and s.updated_at > t.updated_at then update set *
when not matched then insert *;
```

For append-only fact tables, partition overwrite is simpler and avoids the `MERGE` cost:

```sql
insert overwrite table analytics.fct_pageview
  partition (dt = '2026-05-31')
select * from staging.pageview where dt = '2026-05-31';
```

The orchestrator (next chapter) guarantees the partition $"DAG"$ runs once per logical date even if a worker dies mid-task.

== Schema Evolution

$"ELT"$ tolerates schema drift because raw data is retained. Two practical patterns:

- *Semi-structured columns:* land payloads as $"JSON"$ / `variant` and project columns lazily. BigQuery `JSON`, Snowflake `VARIANT`, Databricks `variantType`.
- *Open table formats:* Iceberg / Delta / Hudi allow `ADD COLUMN`, `RENAME COLUMN`, and type widening without rewriting files. See #xref("data-engineering", "lakehouse-engineering", label: "Lakehouse Engineering").

*Anti-pattern:* nullable-everything wide tables that hide schema breakage. Pair evolution with contracts (#xref("data-engineering", "data-quality", label: "Data Quality")).

== Cost Comparison Sketch

A 1 TB clickstream ingest, transformed daily for 30 days:

#table(
  columns: 3,
  [*Step*], [*ETL on Spark cluster*], [*ELT in Snowflake*],
  [Compute], [$"EMR"$ ~\$120/day = \$3.6k], [Warehouse credits ~\$60/day = \$1.8k],
  [Storage], [$"S3"$ raw only ~\$23], [$"S3"$ + Snowflake stages ~\$70],
  [Engineer time], [Higher (PySpark, $"YAML"$, $"DAG"$)], [Lower (pure $"SQL"$ + dbt)],
)

The engineer-time delta usually dominates the cloud bill in the first two years; the compute delta dominates after. #xref("cloud-and-infrastructure", "cost-engineering", label: "Cloud Cost Engineering") (Cloud & Infrastructure volume) formalizes this.

== Streaming ELT

The same pattern applies to streaming: raw $"CDC"$ lands in a bronze table, silver applies dedup and typing, gold serves downstream. The medallion architecture is the streaming-aware restatement of $"ELT"$. See _Streaming_ and `database/streaming-and-incremental-computation.typ`.

== Pitfalls

- *Raw-as-source-of-truth fallacy.* If upstream rotates schemas without notice, your raw zone is only as good as your contract enforcement. Land + validate.
- *Warehouse compute as the only hammer.* For per-row Python (e.g., $"NLP"$, image features), pull data out to Spark / Ray; do not write $"UDF"$-heavy $"SQL"$.
- *Forgetting the "T" in $"ELT"$.* Without dbt-style modeling, $"ELT"$ becomes "load and forget"; analysts duplicate cleaning code in every dashboard.
- *Ignoring partition design.* A poorly partitioned bronze table makes every backfill a full table scan. Default to date partitioning and re-evaluate as data grows.

== Further Reading

Kimball, R., Ross, M. (2013). _The Data Warehouse Toolkit_, 3rd ed. Wiley.

Reis, J., Housley, M. (2022). _Fundamentals of Data Engineering._ O'Reilly.

dbt Labs. "How we structure our dbt projects." https://docs.getdbt.com/best-practices.

Databricks. "The Medallion Architecture." https://docs.databricks.com/lakehouse/medallion.html.

Snowflake. "Continuous Data Pipelines." https://docs.snowflake.com.

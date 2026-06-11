= Data Quality

Data quality is the discipline of detecting, preventing, and communicating defects in data before they corrupt downstream decisions. Unlike software bugs, data bugs are silent: a pipeline can run green for months while loading nulls into a revenue column, and the failure surfaces only when a finance report disagrees with the bank statement. This chapter covers the dimensions used to define quality, declarative expectations frameworks (Great Expectations, dbt tests, Soda), statistical anomaly detection on metrics, data contracts, the write-audit-publish pattern, circuit breakers, and data SLAs.

*See also:* _Schema Evolution_ (contracts at the schema layer), _Orchestration_ (where checks run in the DAG), _Change Data Capture_ (quality of replicated data), _Batch Processing_.

== Dimensions of Quality

The classical taxonomy (Wang & Strong, 1996) names dozens of dimensions; in practice six dominate engineering work:

- *Completeness.* Are all expected rows and values present? Measured as null rate, row count vs source, partition presence.
- *Accuracy.* Do values reflect reality? Hardest to measure automatically; typically validated against a trusted reference (reconciliation against the billing system).
- *Consistency.* Do the same facts agree across systems? `sum(orders.amount)` in the warehouse must match the OLTP source within tolerance.
- *Timeliness / freshness.* How stale is the data? Measured as `now() - max(event_ts)` or `now() - last_load_ts`.
- *Validity.* Do values conform to the schema and domain rules? `country_code in ISO-3166`, `amount >= 0`, foreign keys resolve.
- *Uniqueness.* No duplicate primary keys. Streaming and at-least-once delivery make this the most commonly violated dimension.

A useful operational split: *schema checks* (cheap, run on every batch), *row-level checks* (moderately cheap, run on every batch), and *distributional checks* (require history, run as monitoring).

== Expectations Frameworks

The 2018-2022 wave of tooling made quality checks declarative: you state what should be true, the framework compiles it to SQL or DataFrame operations and reports violations.

*Great Expectations* (2018) introduced the "expectation" vocabulary. An expectation suite is a JSON/YAML document of assertions evaluated against a batch:

```python
validator.expect_column_values_to_not_be_null("order_id")
validator.expect_column_values_to_be_unique("order_id")
validator.expect_column_values_to_be_between(
    "amount", min_value=0, max_value=1_000_000)
validator.expect_table_row_count_to_be_between(
    min_value=10_000, max_value=500_000)
```

GX produces *Data Docs* — HTML reports of pass/fail per expectation — and supports *profiling* (generating a candidate suite from observed data). Its weakness is operational weight; the lighter `GX Core` (2024) and checkpoint-as-code style address this.

*dbt tests* embed checks in the transformation layer. Generic tests (`not_null`, `unique`, `accepted_values`, `relationships`) are declared in YAML next to the model; singular tests are SQL queries that must return zero rows:

```yaml
models:
  - name: fct_orders
    columns:
      - name: order_id
        tests: [not_null, unique]
      - name: status
        tests:
          - accepted_values:
              values: ['PLACED', 'PAID', 'SHIPPED', 'CANCELLED']
```

Because dbt knows the DAG, a failing test can block downstream models (`dbt build` stops the lineage subtree). Packages like `dbt-expectations` and `elementary` extend the vocabulary with distributional and freshness tests.

*Soda* (Soda Core / Soda Cloud) uses a checks language, SodaCL, designed to be readable by analysts:

```yaml
checks for fct_orders:
  - row_count > 10000
  - missing_count(order_id) = 0
  - duplicate_count(order_id) = 0
  - freshness(loaded_at) < 2h
  - avg(amount) between 40 and 90
```

Soda pushes computation into the warehouse (one scan compiles to a handful of aggregate queries) and ships anomaly detection on metric history in the cloud product.

== Anomaly Detection on Metrics

Rule-based checks catch known failure modes; *metric monitoring* catches unknown ones. The pattern: compute a time series of table-level metrics per load (row count, null rates, distinct counts, column means, freshness), then flag points that deviate from the learned seasonal pattern. Monte Carlo, Anomalo, Bigeye, Metaplane, and Elementary all operate this way, differing mainly in how much they auto-instrument.

Detection methods, roughly in order of sophistication:

- *Static thresholds.* `row_count > 0`. Cheap, brittle.
- *Relative deltas.* Today within $plus.minus 20%$ of yesterday or the same weekday last week. Handles weekly seasonality crudely.
- *Seasonal decomposition.* Fit trend + weekly/daily seasonality (STL, Prophet-style) and alert when the residual exceeds $k sigma$. Handles Black Friday only if it was in training data.
- *Change-point detection.* Detect distribution shifts (CUSUM, Bayesian online change-point) rather than single outliers; better for "the upstream team silently changed the enum encoding."

The hard problem is *alert fatigue*: a monitor that fires daily gets muted. Practical mitigations: alert on persistent anomalies (two consecutive violations), route by table tier (page only for tier-1 tables), and require an owner for every monitor.

== Data Contracts

A *data contract* is an explicit, versioned agreement between a data producer and its consumers covering schema, semantics, quality guarantees, and SLAs — enforced in CI and at runtime rather than discovered after breakage. The term was popularised around 2022 (Chad Sanderson, Andrew Jones) as a reaction to "the analytics team finds out about the column rename in production."

A contract typically specifies:

- *Schema.* Field names, types, nullability — registered in a schema registry or as Protobuf/Avro definitions in the producer's repo.
- *Semantics.* `amount` is gross, in minor currency units, post-discount.
- *Quality guarantees.* `order_id` unique; `<= 0.1%` null `user_id`.
- *SLA.* Data for day $d$ available by 06:00 UTC on $d+1$; 99.5% monthly.
- *Change policy.* Backward-compatible changes allowed anytime; breaking changes require a major version and a deprecation window.

Enforcement points: CI on the producer's repo (schema diff against the registered contract fails the build for breaking changes), the schema registry's compatibility check at publish time, and runtime validation at the consumption boundary. The Open Data Contract Standard (ODCS, Linux Foundation/Bitol) and `datacontract-cli` provide a YAML format plus tooling that compiles contracts into Soda or Great Expectations checks.

The organisational point matters more than the format: contracts shift quality ownership *left*, to the producing service team, instead of leaving the data team to reverse-engineer intent from broken dashboards.

== Write-Audit-Publish

Write-audit-publish (WAP) is the staging pattern that makes quality checks *blocking*: never let unvalidated data become visible.

```
write   →  load into staging (branch / staging table / hidden partition)
audit   →  run checks against staging
publish →  atomic swap into the consumer-visible location (or drop + alert)
```

Implementations:

- *Iceberg branches.* Write to a branch (`spark.wap.branch = 'audit'`), validate the branch, then `fast_forward` main to it. The publish step is an atomic catalog pointer swap.
- *Delta / warehouse.* Load into `staging.fct_orders`, audit, then `alter table .. swap` (Snowflake) or partition exchange.
- *Airflow / Dagster.* Model audit as a task between load and publish; failure blocks publish and pages the owner.

WAP converts data incidents from "consumers saw bad data for 6 hours" into "the 06:00 load is late pending investigation" — almost always the better failure mode.

== Circuit Breakers

A *data circuit breaker* halts a pipeline (or downstream consumption) when quality checks fail, by analogy with the microservice pattern (Nygard, _Release It!_, 2007). Intuit and Netflix described production implementations around 2018-2020. Design decisions:

- *Hard vs soft failures.* Hard checks (primary key duplicated, schema mismatch) trip the breaker and block publish. Soft checks (mean shifted 15%) warn but pass. Most frameworks encode this as severity: `error` vs `warn` in dbt, `fail` vs `warn` in Soda.
- *Blast radius.* Trip only the affected subtree of the DAG, not the whole platform. dbt's `build` semantics and Dagster asset checks give this for free.
- *Manual reset with override.* On-call must be able to acknowledge a known-benign anomaly (a real marketing spike) and force publish — with an audit trail.
- *Stale-vs-wrong policy per table.* For a fraud-feature table, stale data may be worse than slightly anomalous data; the breaker policy is a per-asset decision, not a global one.

== Data SLAs

A data SLA applies SRE vocabulary to datasets. Define per tier-1 asset:

- *SLI.* The measured indicator: freshness (`now() - max(loaded_at)`), completeness vs source, check pass rate.
- *SLO.* The target: fresh within 2h, 99.5% of days.
- *Error budget.* $0.5% approx 3.6$ hours/month of violation; exhausting it triggers a reliability review instead of new feature work.

Severity tiers keep this tractable: tier-1 (executive reporting, ML features in production, regulatory) gets paging alerts, WAP, and contracts; tier-2 gets monitoring; tier-3 gets best-effort. Without tiering, every table implicitly demands tier-1 treatment and the team drowns.

Two metrics summarise program health: *time-to-detection* (anomaly occurred → alert) and *time-to-resolution*. A mature platform detects most incidents from its own monitors rather than from a consumer complaint; the ratio of monitor-detected to consumer-reported incidents is the single best maturity indicator.

== Pitfalls

- *Checks without owners.* A failing test that nobody is paged for is documentation, not protection.
- *Auditing after publish.* Detection-only monitoring tells you consumers already saw the bad data. Move tier-1 checks into the WAP path.
- *Over-asserting.* Pinning `avg(amount) between 40 and 90` breaks on the first legitimate pricing change. Prefer learned baselines for distributional properties; reserve static rules for invariants.
- *Validating only the final table.* A null introduced in bronze surfaces in gold with the lineage obscured. Check at ingestion boundaries too.
- *Sampling silently.* Frameworks that sample large tables can miss rare-key duplicates; know which checks run on full data.

== Further Reading

Wang, R., Strong, D. (1996). "Beyond Accuracy: What Data Quality Means to Data Consumers." Journal of Management Information Systems 12(4). The foundational taxonomy of data-quality dimensions from the consumer's perspective.

Schelter, S. et al. (2018). "Automating Large-Scale Data Quality Verification." VLDB 11(12). Describes Deequ, Amazon's declarative quality library on Spark, including incremental metric computation and anomaly detection on metric history.

Breck, E., Polyzotis, N., Roy, S., Whang, S., Zinkevich, M. (2019). "Data Validation for Machine Learning." MLSys. Google's TFX data-validation system: schema inference, drift and skew detection between training and serving data.

Jones, A. (2023). _Driving Data Quality with Data Contracts._ Packt. A book-length treatment of contracts as the producer-side enforcement mechanism.

Nygard, M. (2007). _Release It!_ Pragmatic Bookshelf. Origin of the circuit-breaker stability pattern that data platforms adapted.

Great Expectations documentation, https://docs.greatexpectations.io/. Soda documentation, https://docs.soda.io/. dbt tests, https://docs.getdbt.com/docs/build/data-tests.

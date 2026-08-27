#import "../template.typ": xref

= Schema Evolution <schema-evolution>

Schemas change: products add fields, teams rename columns, types widen. The engineering problem is letting producers and consumers evolve *independently* without coordinated deploys or broken pipelines. This chapter covers compatibility modes (backward, forward, full), the evolution semantics of Avro, Protobuf, and JSON Schema, schema registries, table-format evolution in Iceberg and Delta (column mapping, field IDs), migration mechanics, and versioning strategy.

*See also:* #xref("data-engineering", "data-quality", label: "Data Quality") (contracts enforce evolution policy), #xref("data-engineering", "change-data-capture", label: "Change Data Capture") (schema changes flowing through CDC), #xref("data-engineering", "lakehouse-engineering", label: "Lakehouse Engineering") (table-level evolution in practice), #xref("data-engineering", "streaming", label: "Streaming").

== Readers, Writers, and Compatibility

A schema change is safe or unsafe only relative to *who deploys first*. Define: the *writer schema* is what data was serialised with; the *reader schema* is what the consumer expects. Compatibility modes (the Confluent Schema Registry vocabulary):

- *Backward compatible.* New readers can read data written with the *old* schema. Safe when consumers upgrade first (or read historical data). Allowed: delete fields, add fields with defaults.
- *Forward compatible.* Old readers can read data written with the *new* schema. Safe when producers upgrade first — the normal case for event streams, where you cannot redeploy every consumer. Allowed: add fields (old readers ignore them), delete fields that had defaults.
- *Full.* Both directions; the intersection: add or remove only fields with defaults.
- *Transitive variants* (`BACKWARD_TRANSITIVE`, etc.) check against *all* registered versions, not just the latest — important because a consumer may replay a Kafka topic containing every historical version.

Rule of thumb: topics with replay or many consumers want `FULL_TRANSITIVE`; internal request/response APIs often get away with `BACKWARD`.

== Avro Semantics

Avro (2009, out of Hadoop) resolves writer schema against reader schema at decode time — *both schemas must be present*. The binary encoding carries no field names or tags; it is a positional encoding driven entirely by the writer schema. This makes Avro compact and makes the registry (or file-embedded schema, as in Avro container files and Parquet footers) mandatory.

Resolution rules: fields are matched *by name*; reader fields missing from the writer get their *default*; writer fields missing from the reader are skipped; numeric *promotions* are allowed (`int → long → float → double`). Renames are breaking unless declared via `aliases`.

```json
{"type": "record", "name": "Order", "fields": [
  {"name": "order_id", "type": "string"},
  {"name": "amount",   "type": "long"},
  {"name": "channel",  "type": ["null", "string"], "default": null}
]}
```

The `["null","string"]` union with `default: null` is the canonical "optional field" idiom; adding such a field is full-compatible. Adding a field *without* a default breaks backward compatibility (new reader cannot fill it when reading old data).

== Protobuf Semantics

Protobuf encodes fields by *tag number*, not name. Unknown tags are skipped (and since proto3 release 3.5, retained and re-serialised). Consequences:

- Renaming a field is wire-compatible (the tag is unchanged) but breaks JSON encoding and source code.
- *Never reuse a tag number.* A reused tag decodes old data as the new type — silent corruption. Use `reserved 4, 9;` for removed fields.
- proto3 has no required fields and no custom defaults: missing scalar fields decode as zero values, which makes "absent" and "zero" indistinguishable unless you use `optional` (experimental in 3.12, 2020; generally available in 3.15, 2021) or wrapper types.
- Changing types within a wire-format class (`int32 ↔ int64 ↔ bool`) "works" on the wire with truncation hazards; treat it as breaking.

Protobuf's discipline is enforced socially and by `buf breaking` (the Buf CLI compares `.proto` files across git revisions in CI) rather than by a runtime registry, though Confluent's registry also supports Protobuf subjects.

== JSON Schema Semantics

JSON Schema is a *validation* language, not a serialisation format, so "compatibility" means: does every document valid under one schema remain valid under the other? This is subtler than Avro/Protobuf because constraints are open-ended. Two traps dominate:

- `additionalProperties: true` (the default) means an old document may contain a field the new schema now declares with a stricter type — so *adding* a constrained field is not forward-safe. Registries that check JSON Schema compatibility (Confluent does, since 5.5) reason about `additionalProperties` explicitly; closed content models (`additionalProperties: false`) make evolution checkable but forbid unknown-field tolerance.
- Tightening any constraint (`maxLength`, `enum`, `required`) is backward-breaking by construction.

JSON's self-describing nature is why it survives in low-governance pipelines, and exactly why those pipelines drift.

== Schema Registries

A schema registry is a versioned database of schemas with a compatibility gate at registration time. Confluent Schema Registry (2015) is the reference design: schemas are registered under a *subject* (typically `<topic>-value`), each gets a global ID, and producers prepend a 5-byte header (magic byte + 4-byte schema ID) to every Kafka message. Consumers fetch the writer schema by ID and resolve against their reader schema. Alternatives: Apicurio (Red Hat), AWS Glue Schema Registry, Buf Schema Registry (Protobuf-native), Karapace (Aiven, Apache-2.0 reimplementation after Confluent's licence change).

Operational guidance:

- Set subject compatibility explicitly; the default (`BACKWARD`) is often the wrong direction for event streams where producers deploy first.
- Disable client auto-registration in production (`auto.register.schemas=false`); register via CI so compatibility failures break the producer's build, not a 3 a.m. publish.
- The registry is on the hot path only at cold start (clients cache by ID), but it is a single point of failure for *new* schema IDs; run it highly available.

== Table-Format Evolution

Warehouses and lakehouses face the same problem at rest. The classical Hive table failed here: columns were resolved *by position or by name* against Parquet files, so renaming a column orphaned old files and dropping a column shifted positions.

*Iceberg* solved this with *field IDs*: every column has an immutable integer ID stored in both table metadata and Parquet file metadata. Name and position are display details. Hence Iceberg supports add, drop, rename, reorder, and type widening (`int → long`, `float → double`, `decimal` precision increase) as pure metadata operations — zero data files rewritten, and old files remain readable because resolution is by ID.

*Delta Lake* originally resolved by name; *column mapping* (`delta.columnMapping.mode = 'name'` or `'id'`, Delta 1.2+, 2022) adds the same indirection, enabling rename and drop without rewrite. Dropped columns leave data physically present until files are rewritten (`REORG TABLE ... PURGE` forces it, relevant for GDPR deletion). Delta also supports `mergeSchema` on write (auto-add new columns) and *type widening* as a table feature (Delta 3.2+).

Iceberg additionally evolves *partition specs*: because partitioning is by hidden transform, you can change `days(ts)` to `hours(ts)` and old data keeps its old layout while new writes use the new spec; queries plan across both.

```sql
-- Iceberg: all metadata-only
alter table lh.silver.orders rename column amt to amount;
alter table lh.silver.orders alter column amount type bigint;
alter table lh.silver.orders write ordered by (user_id);
alter table lh.silver.orders add partition field hours(event_ts);
```

== Migrations and the Expand-Contract Pattern

Truly breaking changes (split a column, change semantics, fix a wrong type) cannot be metadata-only. The standard choreography is *expand-contract* (parallel change):

+ *Expand.* Add the new field/table alongside the old; producers write *both* (dual-write within one record is safe; dual-write across systems is not — see #xref("data-engineering", "change-data-capture", label: "Change Data Capture") on the outbox pattern).
+ *Migrate.* Backfill historical data into the new shape; move consumers one by one, validating new against old (shadow reads, reconciliation queries).
+ *Contract.* After a deprecation window with usage telemetry showing zero readers, drop the old field.

For event streams the equivalent is a *versioned topic* (`orders.v2`) with a translator job downcasting v2 to v1 during the window, or an upcaster on the consumer side. For warehouse models, dbt's convention is versioned models (`fct_orders_v2`) with a contract block and a deprecation date.

== Versioning Strategy

A coherent policy looks like:

- *Semantic versioning of schemas.* Major = breaking (new subject/topic/model version), minor = compatible addition, patch = docs/metadata.
- *Compatibility mode per data class.* `FULL_TRANSITIVE` for tier-1 event streams; `BACKWARD` for internal tables; explicit contract review for anything crossing a team boundary.
- *CI as the gate.* `buf breaking`, registry compatibility check, or `datacontract-cli` diff on every producer PR — evolution policy that is not machine-enforced will be violated.
- *Schema-on-write at the boundary, tolerant readers inside.* Consumers should ignore unknown fields (Postel's law) but validate the fields they use.
- *Deprecation windows with telemetry.* You cannot contract what you cannot observe; track reads per field/version (query logs, registry usage metrics) before dropping anything.

== Pitfalls

- *Confusing wire compatibility with semantic compatibility.* Repurposing `status = 3` from `SHIPPED` to `DELIVERED` passes every registry check and corrupts every consumer.
- *Reusing Protobuf tags or Avro field names with new meanings.* Reserve them forever.
- *Auto-register in production.* One misconfigured producer registers a junk schema and, under `NONE` compatibility, poisons the subject.
- *Hive-style positional Parquet reads.* Any engine reading lakehouse files without field-ID resolution silently mis-maps columns after a reorder.
- *Defaults that lie.* `default: 0` for a new `discount` field makes old records claim "no discount" when the truth is "unknown"; prefer nullable with `null` default.

== Further Reading

Kleppmann, M. (2017). _Designing Data-Intensive Applications_, Chapter 4 ("Encoding and Evolution"). O'Reilly. The clearest treatment of backward/forward compatibility across Avro, Protobuf, and Thrift.

Apache Avro Specification, "Schema Resolution." https://avro.apache.org/docs/current/specification/. The normative writer/reader resolution rules.

Protocol Buffers Language Guide, "Updating a Message Type." https://protobuf.dev/programming-guides/proto3/. The canonical list of safe and unsafe Protobuf changes.

Confluent. "Schema Evolution and Compatibility." https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html. Defines the seven compatibility modes and their transitive variants.

Apache Iceberg Specification, "Schema Evolution" and "Partition Evolution." https://iceberg.apache.org/spec/. Field-ID-based resolution for table formats.

Ambler, S., Sadalage, P. (2006). _Refactoring Databases: Evolutionary Database Design._ Addison-Wesley. Origin of the expand-contract (parallel change) migration discipline.

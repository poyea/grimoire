#import "../template.typ": xref

= Cloud-Native Databases <cloud-native-databases>

A cloud-native database is one designed *for* the cloud, not merely *deployed in* the cloud. The distinguishing architectural moves are: separating compute from storage, treating the log as the durable substrate, exploiting object storage for tiered cold data, and scaling read/write tiers independently. Aurora pioneered log-as-storage; Neon and AlloyDB pushed further into disaggregation; PlanetScale shipped Vitess as a managed service; TiDB and YugabyteDB rebuilt MySQL/PostgreSQL surface on top of distributed transactional KV stores.

*See also:* #xref("database", "transactions-distributed", label: "Distributed Transactions"), #xref("database", "consensus-and-replication", label: "Consensus and Replication"), #xref("database", "partitioning-and-elasticity", label: "Partitioning and Elasticity"), #xref("database", "lakehouses-and-open-formats", label: "Lakehouses and Open Table Formats")

== Why Disaggregate?

Traditional databases couple storage and compute on the same node. Cloud workloads make this painful:

- *Independent scaling.* Read-heavy workloads need more CPU, not more disk. Disaggregation lets you add stateless replicas.
- *Fast failover.* If the storage layer is independently durable, replacing a crashed compute node is seconds, not a recover-from-backup operation.
- *Cheap branching / point-in-time recovery.* Storage that retains every page version supports zero-copy clones.
- *Elastic billing.* Customers pay for compute when used, storage always; both stretch independently.

The cost is *network in the critical path*: every page miss is now a remote read, every commit involves remote acknowledgements.

== Amazon Aurora

Aurora (Verbitski et al., SIGMOD 2017 / 2018) re-architected MySQL and PostgreSQL with one core idea: *the log is the database*.

=== Log-as-Storage

A traditional database writes both the data pages and the WAL to storage. Aurora ships *only the log records* to a distributed storage tier composed of 6 storage nodes across 3 AZs. The storage tier applies the log records to materialize pages on demand.

```
Compute (MySQL/Postgres front-end):
    Buffer pool + query engine + recovery manager removed
    Produces redo log records → sends to storage tier
       │
       ▼
6-of-6 storage nodes across 3 AZs (2 per AZ):
    Persist log records; gossip with peers; materialize pages lazily
    Quorum: write requires 4-of-6 acknowledgement (4w)
             read requires 3-of-6 (3r)  → 4 + 3 > 6 ensures intersection
```

This *4-of-6 / 3-of-6 quorum* tolerates one AZ failure plus one additional node. The bandwidth savings are dramatic: a transaction writing one 16 KB page traditionally generates $approx 32$ KB of network (page + log); Aurora ships only the log delta ($approx$ tens of bytes).

=== Read Replicas

Up to 15 read replicas share the same storage. Each replica subscribes to the redo log stream and updates its own buffer pool. No physical replication of data pages; only log apply.

=== Backtrack and Clones

Storage retains historical log records for a configurable window; *Backtrack* rewinds a cluster to a past LSN without restoring from backup. *Database cloning* creates a new compute cluster pointing at the same storage tier with copy-on-write semantics.

=== Aurora Serverless and Aurora DSQL

*Aurora Serverless v2* scales compute (ACUs) in seconds based on load. *Aurora DSQL* (2024 GA) takes disaggregation further: a fully serverless, multi-region, strongly consistent SQL surface with optimistic concurrency, designed for the "active-active across continents" workload. DSQL uses a custom transaction coordinator with deterministic resolution and zero RTT for read-only transactions in each region.

== Google AlloyDB

AlloyDB (2022) is Google's "PostgreSQL with cloud-native plumbing." It inherits Aurora's log-shipping idea via the *intelligent storage* tier and adds a *columnar engine* layered on top of row storage for HTAP.

=== Columnar Engine

A background process selects hot columns (heuristically: those appearing in analytical queries) and maintains an in-memory columnar representation kept in sync with the row store. The planner chooses row vs columnar at runtime per scan.

```
Row store (PostgreSQL heap) ── change stream ──► Columnar cache (per-column compressed)
                                                       ▲
                                                       │
                                          Optimizer reads columnar
                                          when query is scan-heavy
```

AlloyDB also offers index advisor, vector search (ScaNN), and AI integration (Vertex AI embeddings inside SQL).

== Neon

Neon (2021) is a from-scratch Postgres rewrite of the storage layer with maximum disaggregation. The compute is unmodified PostgreSQL; the storage tier consists of *pageservers* and *safekeepers*.

=== Architecture

```
Compute (vanilla Postgres) ──► WAL ──► Safekeepers (Paxos-replicated WAL)
                                          │
                                          ▼
                                       Pageserver
                                  (reconstructs pages from WAL)
                                          │
                                          ▼
                                    S3 (cold storage)
```

*Safekeepers* form a Paxos group, durably accepting WAL writes (typically 3 nodes). *Pageservers* consume the WAL, build LSM-like layered files, and serve `get_page_at_lsn(rel, blkno, lsn)` to compute nodes. Cold data spills to S3.

=== Branching

Because the pageserver indexes pages by LSN, a *branch* is just a fork at a particular LSN: a new compute instance reads `(lsn, page)` pairs from the shared pageserver, with COW for divergence. This gives Git-style branching of databases used heavily in CI/preview environments.

=== Scale-to-Zero

The compute tier scales to zero when idle (the pageserver is shared infrastructure). Cold starts return in a few seconds because the buffer pool starts empty and the pageserver streams pages on demand.

== PlanetScale (Vitess)

Vitess began at YouTube (2010) to shard MySQL. PlanetScale (2018) productizes it as a managed multi-region MySQL.

=== Sharding via VTGate

```
Client ──► VTGate (stateless SQL router)
                │  parses SQL, rewrites with shard key, fans out
                ▼
            VTTablet (per MySQL replica)
                │
                ▼
            MySQL (InnoDB) — one shard
```

*Keyspaces* are logical databases; each is divided into *shards* by a *Vindex* (vindex = "Vitess Index", a function from row to keyspace ID). Common Vindex types: hash, lookup, numeric, unicode. Cross-shard transactions use a 2PC-like *atomic distributed transaction* opt-in mode; most workloads stick to single-shard transactions.

=== Online Schema Changes

PlanetScale's *non-blocking schema changes* use Vitess's `gh-ost`-derived workflow: a copy table is built in background, deltas streamed via binlog, then a fast cutover. Combined with *branching* (a Git-like dev/staging/main schema model) and *deploy requests*, this gives a deployment workflow inspired by software CI.

=== Vitess Boost and Read Caches

VTGate caches query results with TTL invalidation; Vitess Boost (2022) added a streaming materialized-view layer (built on Noria) for sub-millisecond reads of pre-computed aggregates.

== TiDB

TiDB (PingCAP, 2016) is a MySQL-compatible SQL layer on top of *TiKV*, a Raft-replicated, range-partitioned KV store inspired by Spanner. It adds *TiFlash*, a columnar replica for analytics (HTAP).

```
TiDB SQL nodes (stateless, MySQL wire protocol)
        │ (regions located via PD)
        ▼
TiKV nodes ── Raft groups ──► Replicas
       │
       └── columnar copy via Raft Learner ──► TiFlash nodes
        ▲
        │
    PD (Placement Driver: cluster metadata + TSO timestamp oracle, Raft-replicated)
```

=== Transactions

TiDB uses *Percolator* (Google, 2010): optimistic, snapshot-isolated, 2PC over single-key Paxos (here: Raft). Each transaction has a `start_ts` and `commit_ts` from PD's TSO; locks are stored as KV pairs in TiKV. Pessimistic mode added later (MySQL compatibility).

=== HTAP via TiFlash

TiFlash is a *learner* in the Raft group: it receives the log but does not vote. It transcodes row format to ClickHouse-like columnar storage. The TiDB optimizer chooses row (TiKV) or columnar (TiFlash) per query, or even per operator (push partial aggregates to TiFlash, join with TiKV-served lookups).

== YugabyteDB

YugabyteDB (2018) re-implements PostgreSQL on top of *DocDB*, a Spanner/HBase-inspired sharded transactional store. Goals: PostgreSQL surface, geo-distributed, ACID.

```
PostgreSQL query layer (forked PG 11/15)
        ▼
DocDB: per-tablet RocksDB + Raft
        │
        └─ Hybrid logical clocks (HLC) for cross-tablet ordering
```

=== Hybrid Logical Clocks

Spanner uses TrueTime (GPS + atomic clocks). YugabyteDB uses HLCs: a timestamp combines wall clock with a Lamport counter, propagated on RPCs to enforce causality. The protocol is *Distributed Snapshot Isolation* by default with optional Serializable.

=== Multi-Region Topologies

YugabyteDB supports three deployment models:

- *Synchronous replication across regions*: Spanner-like; high write latency, strongly consistent.
- *Read replicas*: single-region writes, eventually-consistent reads elsewhere.
- *xCluster* (asynchronous replication): independent clusters per region, conflict resolution by application.

== CockroachDB

(Briefly, for comparison.) CockroachDB is the closest open-source analogue to Spanner: range-sharded MVCC KV, Raft per range, distributed SQL planner, serializable isolation by default. Uses HLCs and assumes bounded clock skew (`max_offset`) rather than TrueTime hardware.

== Side-by-Side

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  [*System*], [*Wire*], [*Storage*], [*Compute/Storage Split*], [*Consensus*], [*Notable Feature*],
  [Aurora], [MySQL/Postgres], [Log-structured pages, 6 replicas], [Yes — log only to storage], [Custom 4-of-6 quorum], [No checkpoints; instant clones],
  [Aurora DSQL], [Postgres], [Disaggregated, multi-region], [Yes], [Custom OCC], [Active-active multi-region SQL],
  [AlloyDB], [Postgres], [Intelligent storage + columnar cache], [Yes], [Quorum WAL], [HTAP via dual-format],
  [Neon], [Postgres], [Pageserver + safekeepers + S3], [Yes — fully disaggregated], [Paxos (safekeepers)], [Branching, scale-to-zero],
  [PlanetScale], [MySQL], [Sharded MySQL via Vitess], [No (shared-nothing shards)], [MySQL semisync], [Schema branching, gh-ost cutover],
  [TiDB], [MySQL], [TiKV (rocksdb + Raft) + TiFlash], [Yes — stateless SQL nodes], [Raft], [HTAP via Raft learner],
  [YugabyteDB], [Postgres], [DocDB (rocksdb + Raft)], [Yes], [Raft + HLC], [Geo-partitioned PG],
  [CockroachDB], [Postgres], [Pebble + Raft per range], [Yes], [Raft + HLC], [Range rebalancing, locality leases],
  [Spanner], [SQL/gRPC], [Colossus + TrueTime], [Yes], [Paxos], [Bounded clock skew via GPS/atomic],
)

== Common Architectural Patterns

1. *Stateless compute / stateful storage*: every system above shares this.
2. *Replicated WAL is the source of truth*: pages derive from log, not the other way.
3. *Consensus per shard / range*: Raft or Paxos at the *partition* level, not the cluster level.
4. *Timestamp oracle or HLC*: needed for cross-shard snapshot consistency.
5. *Object storage as the tier-3 home*: Aurora to S3, Neon to S3, Snowflake to S3, all converge.
6. *Branching as a first-class operation*: Aurora clones, Neon branches, PlanetScale dev branches.

== Tradeoffs and Caveats

- *Latency floor.* Every commit travels to a quorum across AZs; even Aurora adds $approx 1.5$ ms over local NVMe MySQL for single-row writes.
- *Cold buffer pools.* Disaggregated compute restarts with an empty cache; the first queries can be $10 times$ slower.
- *Network attached storage cost.* Cross-AZ traffic is metered and dominates cost for some workloads.
- *Vendor lock-in via control plane.* The open-source bits (Postgres, MySQL, Vitess, TiDB, Yugabyte) are portable; the disaggregated storage is not.
- *Operator complexity moved, not eliminated.* TiDB and YugabyteDB clusters require operating Raft groups, PD/coordinator services, and rebalancers.

== Further Reading

Verbitski, A. et al. (2017). "Amazon Aurora: Design Considerations for High Throughput Cloud-Native Relational Databases." SIGMOD.

Verbitski, A. et al. (2018). "Amazon Aurora: On Avoiding Distributed Consensus for I/Os, Commits, and Membership Changes." SIGMOD.

Vuppalapati, M. et al. (2020). "Building An Elastic Query Engine on Disaggregated Storage." NSDI (Snowflake).

Google Cloud (2022). "AlloyDB for PostgreSQL Under the Hood: Intelligent Storage." Blog series.

Bortnikov, V. et al. (2024). "AlloyDB: A Cloud-Native Database for HTAP." VLDB Industrial.

Neon (2022). "Get me out of the cloud: Neon's separation of storage and compute." neon.tech engineering blog.

Slootman, F. et al. (2016). "The Snowflake Elastic Data Warehouse." SIGMOD.

Vitess docs and Slack archives (PlanetScale/CNCF).

Huang, D. et al. (2020). "TiDB: A Raft-Based HTAP Database." VLDB.

Lai, J. et al. (2023). "Distributed PostgreSQL: A Look Inside YugabyteDB." VLDB Industrial.

Taft, R. et al. (2020). "CockroachDB: The Resilient Geo-Distributed SQL Database." SIGMOD.

Corbett, J. et al. (2012). "Spanner: Google's Globally-Distributed Database." OSDI.

Peng, D., Dabek, F. (2010). "Large-scale Incremental Processing Using Distributed Transactions and Notifications." OSDI (Percolator).

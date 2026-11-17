#import "../template.typ": xref

= Cloud Storage Services <cloud-storage>

Cloud storage comes in four shapes: object stores, network block volumes, managed file systems, and local instance disks. Each one makes a different promise about durability, consistency, latency, and failure scope, and each is priced on different meters. Most production incidents and most surprise bills in storage come from treating one shape as if it were another: using an object store as a file system, a zonal block volume as if it were regional, or local NVMe as if it survived a stop. This chapter covers the semantics and performance models of each, how durability is engineered and calculated, storage classes and lifecycle, encryption, and data protection.

*See also:* #xref("cloud-and-infrastructure", "iaas-fundamentals", label: "IaaS Fundamentals") (block and object basics), #xref("database", "cloud-native-databases", label: "Cloud-Native Databases"), #xref("database", "lakehouses-and-open-formats", label: "Lakehouses and Open Table Formats"), #xref("cloud-and-infrastructure", "cost-engineering", label: "Cloud Cost Engineering").

== The Four Shapes

#table(
  columns: 5,
  [*Shape*], [*Examples*], [*Failure scope*], [*Typical latency*], [*Interface*],
  [Object store], [S3, Cloud Storage, Blob Storage], [Region (or multi-region)], [10-100 ms first byte], [HTTP API on whole objects],
  [Network block], [EBS, Persistent Disk, Managed Disk], [Zone (regional variants exist)], [0.2-1 ms per I/O], [Block device, one writer],
  [Managed file], [EFS, Filestore, Azure Files, FSx], [Zone or region], [1-10 ms per metadata op], [NFS or SMB, many writers],
  [Local instance disk], [Instance store NVMe, Local SSD], [One host], [Under 0.1 ms], [Block device, lost with the host],
)

The latency column hides the real distinction. Block and local disks serve many small random I/Os; object stores serve few large sequential transfers and scale by parallelism. A workload that issues one 4 KiB object `GET` per user request pays a network round trip, TLS, HTTP parsing, and a metadata lookup every time.

== Object Storage Semantics

An object store is a flat namespace of keys mapped to immutable byte sequences plus metadata. There are no directories, no partial overwrites, no appends (outside special classes), and no rename. A "folder" is a key prefix and a delimiter convention in the `LIST` API. "Renaming" a prefix of a million objects is a million copies and a million deletes, which is why lakehouse table formats never rename data files.

=== Consistency

S3, Cloud Storage, and Blob Storage all give strong read-after-write consistency for single objects: a successful `PUT` is visible to every later `GET` and `LIST`. Before December 2020 S3 was eventually consistent for overwrites and listings, and much older tooling (S3Guard for Hadoop, EMRFS consistent view) existed only to paper over that. There are still no multi-object transactions: writing a manifest and its data files is several independent operations, and a reader can see any interleaving of them.

=== Conditional Writes

Without conditions, two concurrent writers to the same key race and the last one wins silently. Conditional requests turn the store into a compare-and-swap register:

#table(
  columns: 3,
  [*Store*], [*Create only if absent*], [*Replace only if unchanged*],
  [S3], [`If-None-Match: *` (since 2024)], [`If-Match: <etag>` (since 2024)],
  [Cloud Storage], [`ifGenerationMatch=0`], [`ifGenerationMatch=<generation>`],
  [Blob Storage], [`If-None-Match: *`], [`If-Match: <etag>`],
)

This is enough to build a commit protocol: a writer creates `_log/000042.json` with "create only if absent", and whoever succeeds owns version 42. Delta Lake commits this way on stores that support it; on S3 it needed an external lock table (DynamoDB) until conditional writes arrived.

=== Request Rates and Key Design

S3 partitions its index by key prefix and scales a prefix to at least 3,500 writes and 5,500 reads per second, splitting hot prefixes automatically. The split takes minutes, so a sudden burst on a new prefix sees `503 Slow Down` errors until the index catches up. Spreading load across prefixes (for example `shard=07/2026/11/17/...` rather than `2026/11/17/...`) gives the partitioner more room. Cloud Storage behaves similarly and recommends ramping up request rates gradually and avoiding sequential key names.

=== Multipart Upload

Large objects are uploaded in parts that are sent in parallel, retried independently, and stitched together by a final `Complete` call. S3's limits shape the part size: at most 10,000 parts, each between 5 MiB and 5 GiB (the last part may be smaller), for objects up to 5 TiB.

```cpp
#include <algorithm>
#include <cstdint>
#include <optional>

constexpr std::uint64_t MiB = 1ull << 20;
constexpr std::uint64_t GiB = 1ull << 30;
constexpr std::uint64_t TiB = 1ull << 40;

constexpr std::uint64_t min_part = 5 * MiB;
constexpr std::uint64_t max_part = 5 * GiB;
constexpr std::uint64_t max_parts = 10'000;
constexpr std::uint64_t max_object = 5 * TiB;

struct upload_plan {
    std::uint64_t part_size;
    std::uint64_t part_count;
};

std::optional<upload_plan> plan_upload(std::uint64_t object_size,
                                       std::uint64_t preferred_part = 64 * MiB) {
    if (object_size > max_object) return std::nullopt;
    const std::uint64_t needed = (object_size + max_parts - 1) / max_parts;
    std::uint64_t part = std::max({preferred_part, needed, min_part});
    part = (part + MiB - 1) / MiB * MiB;
    if (part > max_part) return std::nullopt;
    const std::uint64_t count = object_size == 0 ? 1 : (object_size + part - 1) / part;
    return upload_plan{part, count};
}
```

Parallelism is the throughput knob for both uploads and downloads. A single HTTP stream to an object store typically reaches 50 to 100 MB/s; saturating a 25 Gbps instance takes dozens of concurrent ranged requests. Incomplete multipart uploads are invisible in normal listings but are billed as storage until aborted, so every bucket should have a lifecycle rule that aborts them after a few days.

== Durability Engineering

The "eleven nines" figure (99.999999999 percent annual durability) is a design target derived from a model, not a measurement. The model asks: given a coding scheme, a per-device failure rate, and a repair time, how likely is it that more fragments fail before repair finishes than the code can tolerate?

For a Reed-Solomon code with $k$ data and $m$ parity fragments, $n = k + m$, data is lost if more than $m$ of the $n$ fragments fail within one repair window. With independent failures of probability $p$ per fragment per window:

$ P_"loss" = sum_(i = m + 1)^n binom(n, i) p^i (1 - p)^(n - i) $

```cpp
#include <cmath>

double log_choose(int n, int k) {
    return std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
}

double loss_probability(int k, int m, double p) {
    const int n = k + m;
    double total = 0.0;
    for (int i = m + 1; i <= n; ++i)
        total += std::exp(log_choose(n, i) + i * std::log(p) + (n - i) * std::log1p(-p));
    return total;
}

// Annual failure rate afr, repair time in hours -> per-window probability,
// then scale by the number of windows per year.
double annual_loss(int k, int m, double afr, double repair_hours) {
    const double windows = 8766.0 / repair_hours;
    const double p = afr / windows;
    return -std::expm1(windows * std::log1p(-loss_probability(k, m, p)));
}
```

With a 2 percent annual drive failure rate and a 6-hour repair window, three-way replication ($k = 1, m = 2$) loses a given stripe with probability about $4 times 10^(-12)$ per year, and RS(10, 4) gives about $10^(-18)$ at 1.4 times overhead instead of 3. Note the `expm1` and `log1p` calls: computing $1 - (1 - x)^w$ directly rounds to zero in double precision for $x$ this small. Two facts make the real answer worse than the formula: failures are correlated (a rack loses power, a firmware bug hits one drive model, an operator deletes the wrong thing), and repair time grows with the amount of data on the failed device. Real systems place fragments across failure domains (racks, zones) so a single correlated event cannot remove more than $m$ fragments, and they prioritise repair of stripes that have lost the most fragments.

The model also does not cover the dominant real cause of data loss: a correct request to delete or overwrite the data. Durability protects against hardware; only versioning, object lock, and backups in a separate account protect against people and software.

== Storage Classes and Lifecycle

Object stores price capacity against access. Colder classes cost less per GB-month but charge per GB retrieved and bill a minimum storage duration and minimum object size.

#table(
  columns: 5,
  [*S3 class*], [*Relative storage price*], [*Retrieval*], [*Minimum duration*], [*Use*],
  [Standard], [1.0], [Free, milliseconds], [None], [Hot data],
  [Standard-IA], [About 0.55], [Per-GB fee, milliseconds], [30 days, 128 KiB], [Monthly access],
  [Glacier Instant Retrieval], [About 0.17], [Higher per-GB fee, milliseconds], [90 days, 128 KiB], [Quarterly access],
  [Glacier Flexible Retrieval], [About 0.16], [Minutes to hours], [90 days], [Archives with planned restores],
  [Glacier Deep Archive], [About 0.04], [Up to 12 hours (48 for bulk)], [180 days], [Compliance archives],
)

Cloud Storage (Standard, Nearline, Coldline, Archive) and Blob Storage (Hot, Cool, Cold, Archive) follow the same pattern. Cloud Storage's cold classes keep millisecond access; Azure Archive, like Glacier Flexible Retrieval and Deep Archive, requires a rehydration step.

A lifecycle policy moves or deletes objects by age, prefix, tag, or version state. When access patterns are unknown, S3 Intelligent-Tiering and Cloud Storage Autoclass move objects between tiers automatically for a small per-object monitoring fee. The fee makes them a poor fit for billions of tiny objects; aggregate small objects into larger ones first.

The break-even for moving an object colder is a simple inequality. Moving from Standard to Standard-IA saves $Delta s$ per GB-month but costs $r$ per GB per retrieval. If the object is read $a$ times per month, the move pays off only when $a r < Delta s$. Retrieval prices are usually set so that one read per month roughly cancels the saving.

== Block Storage Performance

Network block volumes have three meters: capacity, IOPS, and throughput. Volume types differ in how the last two are granted.

#table(
  columns: 4,
  [*AWS type*], [*Media*], [*IOPS model*], [*Use*],
  [gp3], [SSD], [3,000 IOPS and 125 MiB/s baseline, more provisioned independently of size], [Default for most workloads],
  [gp2], [SSD], [3 IOPS per GiB, burst to 3,000 from a credit bucket], [Legacy default],
  [io2 Block Express], [SSD], [Provisioned, up to 256,000 IOPS, sub-millisecond], [Large databases],
  [st1 / sc1], [HDD], [Throughput-based with burst credits], [Sequential scans, logs],
)

gp2's burst bucket is a token bucket, and it explains a classic failure: a database on a small gp2 volume runs well for hours, exhausts its credits during a batch job, and drops to its baseline (100 IOPS for a 33 GiB volume) until the bucket refills.

```cpp
#include <algorithm>

struct gp2_volume {
    double size_gib;
    double credits = 5.4e6;
    static constexpr double max_credits = 5.4e6;
    static constexpr double burst_iops = 3000.0;

    double baseline() const { return std::clamp(3.0 * size_gib, 100.0, 16000.0); }

    double serve(double demand_iops, double seconds) {
        const double base = baseline();
        if (base >= burst_iops) return std::min(demand_iops, base);
        const double want = std::min(demand_iops, burst_iops);
        const double served = std::min(want, base + credits / seconds);
        credits = std::clamp(credits + (base - served) * seconds, 0.0, max_credits);
        return served;
    }
};
```

At full burst, the bucket drains at $3000 - b$ I/Os per second where $b$ is the baseline, so a full bucket lasts $5.4 times 10^6 \/ (3000 - b)$ seconds: about 30 minutes for a 100 GiB volume. gp3 removed the credit model, which is the main reason to migrate.

Other behaviours that matter in practice:

- *Instance limits.* The instance has its own EBS bandwidth and IOPS ceiling. A 64,000-IOPS volume attached to a small instance delivers what the instance allows, not what the volume is provisioned for.
- *Queue depth.* Network latency means one outstanding I/O at 0.5 ms gives at most 2,000 IOPS. Reaching provisioned IOPS needs enough concurrent I/O (Little's law: concurrency equals throughput times latency).
- *Lazy restore.* A volume created from a snapshot is usable immediately, but each block is fetched from object storage on first touch, with latency in the tens of milliseconds. Databases restored from snapshot run slowly until every block has been read once. Pre-warm by reading the whole device, or pay for fast snapshot restore.
- *Zonal scope.* A volume lives in one zone and cannot be attached elsewhere. Regional variants (GCP regional Persistent Disk, Azure zone-redundant disks) synchronously replicate across two or three zones at a write latency cost.

== Managed File Systems

Managed NFS and SMB services (EFS, Filestore, Azure Files, FSx for Lustre, NetApp ONTAP, OpenZFS) give POSIX-like shared access to many clients. They fill the gap for applications that need a real file system with locking, rename, and partial writes across hosts.

The performance model is dominated by metadata. Every `open`, `stat`, and `close` is a network round trip to a metadata service that must be consistent across clients. Workloads with many small files (package installs, source checkouts, PHP applications) can be ten to a hundred times slower on network file systems than on local disk, while large sequential reads scale well. EFS offers throughput that scales with stored data (bursting), provisioned, or elastic modes; Lustre separates metadata and object servers and is built for parallel HPC and ML training reads.

NFS close-to-open consistency is weaker than local POSIX: a client is guaranteed to see another client's writes only after that client closes the file and this one opens it. Applications that coordinate through file contents, rather than through locks or a separate service, often break on network file systems.

== Local Instance Storage

Instance store volumes and Local SSDs are physical NVMe devices on the host. They give hundreds of thousands to millions of IOPS at tens of microseconds, with no per-I/O charge. They vanish when the instance stops, is terminated, or its host fails (some clouds preserve them across live migration and reboot, never across stop).

They are the right choice when the data is a cache or is already replicated by the application: Cassandra, ScyllaDB, Kafka with replication across zones, Elasticsearch replicas, and build caches. They are the wrong choice for a single-node database whose only copy would sit on them. Treat losing a node's local disk as a routine event that triggers rebuild from replicas.

== Encryption

All three major clouds encrypt object storage at rest by default, and GCP and Azure encrypt disks by default. On AWS, EBS encryption by default is a per-account, per-region setting that should be turned on. The real choice is who controls the key:

#table(
  columns: 3,
  [*Mode*], [*Key control*], [*Cost and effect*],
  [Provider-managed keys], [Provider], [Free; protects against stolen disks only],
  [Customer-managed keys in KMS], [Customer, through IAM and key policy], [Per-request KMS fees; key policy is a second access gate; revoking the key makes data unreadable],
  [Customer-supplied or external keys], [Customer, outside the provider], [Provider never stores the key; losing it loses the data],
)

KMS uses *envelope encryption*. The data is encrypted locally with a data key; the data key is encrypted under a KMS key and stored next to the data. Decrypting requires a KMS call to unwrap the data key, which is where IAM and the audit log apply. Per-object data keys can make KMS request charges exceed storage charges on buckets with many small reads; S3 Bucket Keys reduce this by deriving object keys from a bucket-level data key that is cached for a short time.

== Data Protection

#table(
  columns: 3,
  [*Mechanism*], [*Protects against*], [*Does not protect against*],
  [Versioning], [Overwrites and deletes by mistake], [Deleting the versions; account compromise],
  [Object lock (governance or compliance mode)], [Deletion before a retention date, even by administrators in compliance mode], [Data written wrong in the first place],
  [Cross-region replication], [Region loss], [Logical corruption (it replicates the damage)],
  [Snapshots], [Volume corruption and deletion, point-in-time restore], [Account compromise if kept in the same account],
  [Backups in a separate account with restricted access], [Ransomware, insider error, account takeover], [Nothing, if restores are tested],
)

Replication is not backup. Replication copies every change, including the bad ones, within seconds. Backups preserve the past and must be held where the credentials that can damage production cannot reach them. Ransomware attacks on cloud accounts now target snapshots and replicas first.

== Choosing a Shape

- Data shared by many services, large, written once and read many times: object storage.
- One database or file system per instance, needing low-latency random I/O: network block storage, with snapshots for protection.
- Legacy or shared POSIX workloads across hosts: managed file systems, after checking the metadata rate.
- Replicated data stores and caches that want maximum IOPS: local NVMe.
- Analytics tables: object storage with an open table format on top, which supplies the transactions the store lacks.

== Pitfalls

- Treating prefixes as directories: "renaming" or listing huge prefixes with sequential calls.
- Missing lifecycle rules for incomplete multipart uploads and noncurrent versions, which bill forever.
- Small gp2 volumes that run out of burst credits under load.
- Benchmarking a restored volume before pre-warming, or blaming the database for lazy-restore latency.
- Moving small objects to IA or Glacier classes and paying the 128 KiB minimum and per-object fees on each.
- KMS request throttling and cost on hot buckets without Bucket Keys or caching.
- Calling cross-region replication a backup.

== Further Reading

Calder, B. et al. (2011). "Windows Azure Storage: A Highly Available Cloud Storage Service with Strong Consistency." SOSP.

Huang, C. et al. (2012). "Erasure Coding in Windows Azure Storage." USENIX ATC.

Muralidhar, S. et al. (2014). "f4: Facebook's Warm BLOB Storage System." OSDI.

Ford, D. et al. (2010). "Availability in Globally Distributed Storage Systems." OSDI.

Bornholt, J. et al. (2021). "Using Lightweight Formal Methods to Validate a Key-Value Storage Node in Amazon S3." SOSP.

Brooker, M., Chen, T. and Ping, F. (2020). "Millions of Tiny Databases." NSDI.

Armbrust, M. et al. (2020). "Delta Lake: High-Performance ACID Table Storage over Cloud Object Stores." VLDB.

Pinheiro, E., Weber, W.-D. and Barroso, L. A. (2007). "Failure Trends in a Large Disk Drive Population." FAST.

= Cloud Cost Engineering

Cloud bills are engineering artifacts: every line item maps to an architectural decision made by a developer. Treating cost as a first-class engineering metric — with the same observability discipline applied to latency or error rates — routinely yields 40–60 % reductions without sacrificing reliability. This chapter covers the full stack from raw bill visibility through unit economics, purchase commitment strategy, storage tiering, egress optimisation, and the FinOps culture that sustains the practice.

*See also:* _iaas-fundamentals.typ_, _kubernetes-internals.typ_, _serverless.typ_, `networking/load-balancing.typ`.

== Cost Visibility

The first step is knowing what you are paying for. All three hyperscalers expose a *Cost Explorer* or equivalent:

#table(
  columns: 3,
  [*Cloud*], [*Tool*], [*Export path*],
  [AWS], [Cost Explorer + Cost and Usage Report ($"CUR"$)], [S3 → Athena or Redshift],
  [GCP], [Cloud Billing export], [BigQuery],
  [Azure], [Cost Management + Advisor], [Storage account → Power BI or Log Analytics],
)

The raw CUR contains one row per hour per resource, with dimensions for service, usage type, region, and — crucially — resource tags. Without tags, the export is a noise floor.

=== Tagging Strategy

A *tagging taxonomy* should be established before the first workload launches, not retrofitted later:

```text
Required tags (enforced by SCP/Organization Policy):
  env          = prod | staging | dev
  team         = payments | auth | data-platform | ...
  service      = checkout | user-service | analytics-etl | ...
  cost-centre  = CC-1042 | CC-2017 | ...
  managed-by   = terraform | helm | manual

Optional but recommended:
  customer     = internal | <tenant-id>   (for SaaS showback)
  commit       = <git-sha>
```

Enforce mandatory tags via AWS Service Control Policies (`aws:RequestTag` conditions), GCP Organization Policy constraint `constraints/compute.requireShieldedVm` analogue for labels, or Azure Policy `deny` on resources missing required tags. Untagged resources should generate a Jira ticket automatically via a Cost Anomaly Monitor webhook.

=== Showback vs Chargeback

*Showback* allocates costs to teams in a reporting dashboard without affecting actual invoices — easy to start, low friction, but creates no spending incentive. *Chargeback* uses internal transfer pricing to make teams feel real cost: a \$10 000 over-spend appears in their quarterly budget. Most organisations start with showback and migrate to chargeback once allocation accuracy exceeds ~85 %.

A hybrid model: showback by default, but shared infrastructure (NAT gateways, transit gateways, monitoring agents) allocated via a *blended rate* based on EC2 instance-hours consumed per team, avoiding debates about exact byte-counts.

== Unit Economics

Aggregate spend metrics ("we spent \$400 000 on EC2 this month") are opaque. *Unit economics* normalises cost to a business metric:

$ "cost per request" = "total infra spend" / "total requests" $

$ "cost per active user" = "total infra spend" / "monthly active users" $

$ "gross margin impact" = 1 - "COGS" / "revenue" $

Track these as time-series metrics alongside $"p50"$ / $"p99"$ latency. A cost-per-request that rises during a quiet period with flat traffic indicates a regression — likely an unindexed query, a polling loop, or an accidental resource leak.

```python
# Emit cost-per-request to CloudWatch/Datadog from a daily Lambda
import boto3, datetime

ce = boto3.client("ce")
cw = boto3.client("cloudwatch")

def handler(event, context):
    end   = datetime.date.today().isoformat()
    start = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    resp  = ce.get_cost_and_usage(
        TimePeriod={"Start": start, "End": end},
        Granularity="DAILY",
        Filter={"Tags": {"Key": "service", "Values": ["checkout"]}},
        Metrics=["UnblendedCost"],
    )
    cost_usd = float(resp["ResultsByTime"][0]["Total"]["UnblendedCost"]["Amount"])
    # Pull request count from your metrics store
    requests = get_request_count_yesterday("checkout")
    cw.put_metric_data(
        Namespace="FinOps",
        MetricData=[{"MetricName": "CostPerRequest",
                     "Value": cost_usd / requests if requests else 0,
                     "Unit": "None"}],
    )
```

== Rightsizing

*Rightsizing* matches the instance family and size to actual workload resource consumption. The standard workflow:

+ Collect $"CPU"$ and memory utilisation at P95 over a 2-week window (CloudWatch, GCP Recommender, Azure Advisor all provide this natively).
+ Identify instances where P95 $"CPU"$ < 20 % and P95 memory < 40 %.
+ Propose a move one size-step down within the same family, or a family change (e.g., `m6i` → `t4g`) for burstable workloads.
+ Run a shadow deployment for 48 hours with load-replay before promotion.

$ "savings" approx 0.5 times "on-demand price" times "over-provisioned vCPUs" $

Family-level decisions matter as much as size:

#table(
  columns: 4,
  [*Family*], [*Profile*], [*Use case*], [*Approximate saving vs general*],
  [`t4g`], [Burstable ARM], [Dev, low-traffic APIs], [up to 40 %],
  [`c7g`], [Compute-optimised ARM], [CPU-bound services], [20-35 %],
  [`r7i`], [Memory-optimised], [In-memory caches, analytics], [0 % (right tool)],
  [`inf2`], [ML inference], [Neural-net serving], [60-80 % vs GPU],
  [`g5`], [GPU training], [LLM fine-tuning], [Baseline],
)

=== Graviton / Ampere / Arm

AWS Graviton3, GCP Tau T2A (Ampere Altra), and Azure Dpsv5 (Ampere) all offer ~20 % price-performance improvement over equivalent x86. The switch requires a container rebuild (`GOARCH=arm64`, `docker buildx`) but no application code changes for most Go, Java, Python, and Node workloads.

== Purchase Commitment Strategy

Commitment instruments trade flexibility for discount:

#table(
  columns: 4,
  [*Instrument*], [*Discount*], [*Commitment*], [*Flexibility*],
  [On-demand], [0 %], [None], [Full],
  [Savings Plan (Compute)], [up to 66 %], [1 or 3 year \$/hr], [Any family/region],
  [Reserved Instance (Standard)], [up to 72 %], [1 or 3 year specific type], [Fixed AZ + family],
  [Reserved Instance (Convertible)], [up to 54 %], [1 or 3 year], [Can exchange],
  [Spot / Preemptible], [50–90 %], [None], [Can be reclaimed],
)

The recommended portfolio for a mature workload:

- 60–70 % covered by *Compute Savings Plans* (most flexible; apply to EC2, Lambda, Fargate automatically).
- 10–15 % *Standard RIs* for predictable, instance-family-stable databases and cache nodes.
- 10–20 % *Spot* for batch jobs, ML training, CI runners — anything with a checkpoint-and-resume capability.
- 5–10 % *on-demand* headroom for unplanned spikes.

=== Spot Interruption Model

A Spot instance is reclaimed when the hyperscaler needs the capacity back. AWS gives a 2-minute warning via the *instance metadata service* and EventBridge:

```python
# Poll the IMDS interruption notice endpoint
import urllib.request, time

def check_spot_interruption():
    url = "http://169.254.169.254/latest/meta-data/spot/interruption-action"
    try:
        with urllib.request.urlopen(url, timeout=1) as r:
            return r.read().decode()   # "terminate" if notice is active
    except Exception:
        return None                    # No notice

while True:
    action = check_spot_interruption()
    if action:
        checkpoint_state()
        drain_connections()
        sys.exit(0)
    time.sleep(5)
```

GCP Preemptible VMs are reclaimed after at most 24 hours and give a 30-second ACPI G2 soft-off signal. Azure Spot VMs follow the same 30-second eviction notice model.

== Storage Tiering

Object storage is priced per GB-month, but retrieval costs differ dramatically across tiers:

#table(
  columns: 4,
  [*S3 Class*], [*Storage \$/GB-mo*], [*Retrieval*], [*Minimum duration*],
  [Standard], [\$0.023], [Free], [None],
  [Intelligent-Tiering], [\$0.023 / \$0.0125], [Free], [None],
  [Standard-IA], [\$0.0125], [\$0.01/GB], [30 days],
  [One Zone-IA], [\$0.01], [\$0.01/GB], [30 days],
  [Glacier Instant Retrieval], [\$0.004], [\$0.03/GB], [90 days],
  [Glacier Flexible Retrieval], [\$0.0036], [\$0.01/GB bulk], [90 days],
  [Glacier Deep Archive], [\$0.00099], [\$0.02/GB], [180 days],
)

An *S3 Lifecycle policy* automates the transitions:

```xml
<LifecycleConfiguration>
  <Rule>
    <Id>log-tiering</Id>
    <Status>Enabled</Status>
    <Filter><Prefix>logs/</Prefix></Filter>
    <Transition>
      <Days>30</Days>
      <StorageClass>STANDARD_IA</StorageClass>
    </Transition>
    <Transition>
      <Days>90</Days>
      <StorageClass>GLACIER_IR</StorageClass>
    </Transition>
    <Expiration><Days>365</Days></Expiration>
  </Rule>
</LifecycleConfiguration>
```

For datasets where access patterns are unknown, *S3 Intelligent-Tiering* autonomously moves objects between frequent and infrequent tiers with no retrieval fee — the monitoring charge (\$0.0025 per 1 000 objects) pays back within days on large buckets.

== Data Transfer and Egress Costs

Egress is the most dangerous hidden cost in cloud architecture. The general rule:

- *Intra-AZ:* free (AWS, GCP, Azure all waive same-AZ traffic).
- *Inter-AZ within a region:* \$0.01–0.02/GB.
- *Inter-region:* \$0.02–0.08/GB.
- *Internet egress:* \$0.05–0.09/GB for the first 10 TB/month, less with volume tiers.

A service doing 100 TB/month of Internet egress pays $100 "TB" times \$0.05 approx \$5 000 slash"month"$ on egress alone, not counting compute. Common mitigations:

+ *CloudFront / Cloud CDN / Azure CDN:* CDN egress is roughly half the origin egress rate, and popular objects are served from cache at zero origin cost.
+ *S3 Transfer Acceleration* / *Direct Connect* / *Cloud Interconnect:* dedicated paths with lower per-GB pricing for high-volume or latency-sensitive transfers.
+ *Compress before transfer:* gzip or zstd at the application layer; a 3:1 compression ratio on log data directly multiplies by three any egress saving.
+ *Co-locate consumers:* move analytics jobs into the same region as the data source; query results (small) egress instead of raw datasets (large).

== FinOps Culture

*FinOps* is the practice of embedding cost ownership into engineering teams rather than treating cloud spend as a finance problem. Practical mechanisms:

- *Weekly cost review* in team stand-up: each team owns a CloudWatch or Datadog cost dashboard filtered to their tag.
- *Anomaly alerting:* AWS Cost Anomaly Detection, GCP Budget alerts, or Azure Cost Management alerts configured at 10 % over 7-day forecast. Alert goes to the team Slack channel, not just finance.
- *Cost as a CI gate:* Infracost or OpenCost runs on every Terraform PR and posts a delta to the pull request. A > 20 % cost increase requires a comment justifying the change.
- *Tagging compliance metric:* percentage of monthly spend with all required tags, tracked as a KPI alongside uptime.
- *Gamification:* public leaderboard of cost-per-request by service; teams that improve their metric get recognition in the engineering all-hands.

== Worked Optimisation Example

A real-world checkout service running on AWS showed the following monthly spend before optimisation:

#table(
  columns: 3,
  [*Line item*], [*Before*], [*After*],
  [EC2 on-demand (m5.2xlarge ×8)], [\$4 320], [\$1 036 (Graviton3 + Savings Plan)],
  [RDS Multi-AZ (db.m5.4xlarge)], [\$2 880], [\$1 620 (db.r7g.2xlarge RI 1yr)],
  [NAT Gateway (egress)], [\$1 200], [\$180 (VPC endpoints for S3/DynamoDB)],
  [S3 storage (raw logs, Standard)], [\$920], [\$115 (Lifecycle → Glacier)],
  [Data Transfer (inter-AZ)], [\$640], [\$320 (co-locate app + DB in same AZ)],
  [*Total*], [*\$9 960*], [*\$3 271*],
)

The 67 % reduction came from four changes: instance family migration, one reserved instance commitment, VPC endpoints eliminating NAT Gateway fees on AWS-internal traffic, and a storage lifecycle policy. None required any application code change.

$ "saving %" = (9960 - 3271) / 9960 times 100 approx 67 % $

== Further Reading

Storment, J. R. and Fuller, M. (2019). _Cloud FinOps._ O'Reilly.

Greenberg, A. et al. (2009). "The Cost of a Cloud: Research Problems in Data Center Networks." ACM SIGCOMM Computer Communication Review.

AWS (2024). "AWS Cost Optimization Pillar — Well-Architected Framework." docs.aws.amazon.com.

Google Cloud (2024). "Cost Optimization on Google Cloud." cloud.google.com.

Infracost (2024). "Cloud Cost Estimation for Terraform." infracost.io.

Levy, A. et al. (2020). "Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider." ATC.

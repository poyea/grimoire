= IaaS Fundamentals

Infrastructure-as-a-Service abstracts compute, storage, and networking behind APIs so that capacity becomes an operational expense rather than a procurement cycle. The three hyperscalers (AWS, GCP, Azure) expose conceptually parallel primitives — virtual machines, virtual networks, object stores, managed identities — but the API shapes, isolation boundaries, and pricing models differ enough that portable design requires understanding each abstraction in detail. This chapter maps the primitives side-by-side and digs into the substrate (Nitro, Titan, hypervisors) that makes them possible.

*See also:* `containers.typ`, `iac.typ`, `networking/data-center-networking.typ`, `networking/load-balancing.typ`, `linux-kernel/cgroups-namespaces.typ`.

== The IaaS Stack

At the bottom sits physical hardware in a region/zone topology. Above it, a hypervisor (KVM, Hyper-V, custom) partitions the machine, a control plane orchestrates placement, and a metadata service exposes per-instance identity. A typical "boot a VM" call traverses:

+ Public API (REST/gRPC) authenticated via signed request.
+ Quota and policy check (RBAC, service control policies).
+ Placement engine selecting host, rack, AZ.
+ Image fetch from object storage; root volume cloned via copy-on-write.
+ Hypervisor instantiation; vNIC attached to the tenant overlay.
+ Metadata service populated; user-data executed on first boot.

This sequence is essentially identical on AWS, GCP, and Azure; the differences are in latency (seconds to tens of seconds), placement constraints, and what the hypervisor offloads to dedicated hardware.

== Cross-Cloud Abstraction Map

#table(
  columns: 4,
  [*Concept*], [*AWS*], [*GCP*], [*Azure*],
  [VM], [EC2 Instance], [Compute Engine VM], [Virtual Machine],
  [VM image], [AMI], [Image / Machine Image], [Managed Image / Gallery],
  [Block volume], [EBS], [Persistent Disk], [Managed Disk],
  [Object store], [S3], [Cloud Storage], [Blob Storage],
  [Virtual network], [VPC], [VPC Network], [VNet],
  [Subnet], [Subnet (AZ-scoped)], [Subnet (regional)], [Subnet],
  [Firewall], [Security Group + NACL], [Firewall Rule], [NSG],
  [Load balancer], [ELB/ALB/NLB], [Cloud Load Balancing], [Load Balancer / App Gateway],
  [DNS], [Route 53], [Cloud DNS], [Azure DNS],
  [Identity], [IAM], [IAM + Service Account], [Entra ID + Managed Identity],
  [KMS], [KMS / CloudHSM], [Cloud KMS], [Key Vault],
  [Region], [Region (e.g. us-east-1)], [Region (us-central1)], [Region (eastus)],
  [Failure domain], [Availability Zone], [Zone], [Availability Zone],
  [Object-store consistency], [Strong read-after-write], [Strong], [Strong],
)

A few subtleties are not visible from the table. GCP subnets are *regional* (span zones), while AWS subnets are *zonal*; this changes how multi-AZ services are wired. AWS security groups are stateful (return traffic auto-allowed) whereas NACLs are stateless. Azure's identity model fuses cloud RBAC with directory identities in a way that has no direct AWS analogue.

== Virtual Machines and Hypervisors

Modern public-cloud VMs run on a thin hypervisor with extensive hardware offload. AWS Nitro moves the VPC data plane, EBS attachment, and instance security onto the Nitro Card (a custom $"PCIe"$ device), leaving the Nitro Hypervisor as a minimal KVM derivative that performs essentially only $"vCPU"$ scheduling and memory management. The result is that EC2 metal instances are simply VMs with no hypervisor overhead between guest and bare hardware — a property leveraged by nested virtualization workloads such as macOS instances and Firecracker hosts.

```text
   ┌──────────────────────────────────────────────────────┐
   │ Guest OS (Linux/Windows)                             │
   ├──────────────────────────────────────────────────────┤
   │ Nitro Hypervisor (minimal KVM)                       │
   ├──────────────────────────────────────────────────────┤
   │ Nitro Cards: VPC, EBS, Security Chip, Controller     │
   ├──────────────────────────────────────────────────────┤
   │ Bare-metal server                                    │
   └──────────────────────────────────────────────────────┘
```

GCP's Titan-rooted infrastructure and the Andromeda virtual network achieve similar separation: the host runs a hardened KVM, and Titan provides the root of trust for both the host and any attached accelerators. Azure's Boost SmartNIC has converged on the same architecture.

*Performance implications:*
- Network and storage bandwidth scale with instance size because the offload card has dedicated queues per $"vCPU"$.
- $"SR-IOV"$ allows the guest to talk directly to virtual functions, avoiding the host network stack.
- Instances expose Enhanced Networking / gVNIC / Accelerated Networking — all variants of $"SR-IOV"$ + DPDK-style polling.

== Block Storage and Replication

EBS, Persistent Disk, and Managed Disk are network-attached block stores. Each write is synchronously replicated to multiple copies (typically 2-3) within a zone before acknowledgment. Snapshots are incremental, stored in the regional object store, and form a chain pointing back to a base image.

```python
# AWS SDK: create snapshot and chain
import boto3
ec2 = boto3.client("ec2")
snap = ec2.create_snapshot(VolumeId="vol-0abc", Description="nightly")
# Snapshot becomes "completed" after async background copy.
ec2.create_volume(SnapshotId=snap["SnapshotId"], AvailabilityZone="us-east-1a",
                  VolumeType="gp3", Iops=6000, Throughput=250)
```

*Durability vs availability:* Three-way zonal replication gives ~5 nines durability but the whole zone is a single failure domain. Cross-zone durability requires snapshots (async) or filesystem-level replication (DRBD, ZFS send/recv). For regional-strength durability use object storage (11 nines on S3) backing the dataset.

== Object Storage as the Universal Substrate

Object stores (S3, GCS, Blob) are the only IaaS primitive that combines durability, capacity elasticity, and global accessibility. Internally they shard objects across thousands of nodes via consistent hashing + erasure coding. Reed-Solomon $(k, m)$ codes — e.g., $(10, 4)$ — recover any 4 lost shards out of 14 with $1.4 times$ overhead vs $3 times$ for triple replication.

```text
Object "logs/2026-06-03.gz" (1 GB)
   │
   ▼  split into 64 MiB stripes
[s0][s1][s2]...[s15]
   │
   ▼  RS(10,4) per stripe
[d0..d9][p0..p3]   → 14 fragments per stripe
   │
   ▼ distributed across racks/zones
```

S3 moved to strong read-after-write consistency in 2020 by introducing a strongly consistent metadata store (a Paxos-replicated cache layer in front of the eventually-consistent index). GCS and Azure Blob have always been strongly consistent at the object level.

== Virtual Networks

A cloud $"VPC"$ is a per-tenant overlay implemented with $"VXLAN"$ or a proprietary encapsulation (Andromeda, AWS Hyperplane, Azure VFP). The control plane programs flow tables on every host so that a packet from instance $A$ to instance $B$ is encapsulated with the tenant ID, routed across the underlay, decapsulated, and delivered without the underlying network ever seeing tenant IPs.

```text
Instance A (10.0.1.5)
  → host vSwitch (look up flow)
  → encap [outer: host A → host B | inner: 10.0.1.5 → 10.0.2.7 | tenant=42]
  → underlay routes by outer header
  → host B vSwitch decap, deliver to Instance B (10.0.2.7)
```

The flow table is the bottleneck for connection rates; Nitro and Andromeda push flow lookup into hardware to sustain millions of connections per second per host.

== Identity and Authorization

Every API call carries a signed identity. AWS uses Signature V4 (HMAC-SHA256 over canonical request); GCP uses OAuth 2 bearer tokens minted from service-account keys or workload identity; Azure uses Entra-issued JWTs. The authorization decision is then made by IAM, which evaluates policies against the requested resource and action.

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:PutObject"],
    "Resource": "arn:aws:s3:::reports-2026/*",
    "Condition": {"StringEquals": {"aws:RequestedRegion": "us-east-1"}}
  }]
}
```

The least-privilege principle requires scoping by resource and condition; permission-boundary patterns are essential at scale because individual roles routinely accumulate dozens of statements.

== Regions, Zones, and Failure Domains

A *region* is a metropolitan area; an *availability zone* is an independently powered, cooled, and networked facility. Round-trip latency between AZs in the same region is ~1-2 ms; between regions, 30-200 ms. Workload designers pick a tier:

- *Single-AZ:* cheapest, lowest latency, ~99.9% effective availability.
- *Multi-AZ:* synchronous replication within a region, ~99.99%.
- *Multi-region:* asynchronous replication, ~99.999% but bounded by speed of light.

Synchronous cross-region replication is fundamentally incompatible with low write latency; this is the same observation behind Spanner's TrueTime and Aurora's quorum reads.

== Pricing Primitives

IaaS pricing decomposes into:

- *On-demand* compute, billed per second.
- *Reserved/Committed-use* discounts (1-3 years, 30-60% off).
- *Spot/Preemptible* (50-90% off, can be reclaimed in seconds).
- *Egress* — outbound bytes to the public Internet, the single largest hidden cost in most architectures.
- *Storage* — per GB-month, plus per-request fees on object stores.

A useful mental model: compute is roughly \$0.05 per vCPU-hour on-demand, storage is \$0.02 per GB-month, egress is \$0.05--0.09 per GB. Egress dominates as soon as a service serves more than a few TB/month externally — covered in detail in `cost-engineering.typ`.

== Quotas and Control-Plane Limits

Every cloud enforces both *soft quotas* (raisable per account) and *hard limits* (API-level rate caps). The control plane itself is rate-limited — a runaway CI job that calls `DescribeInstances` thousands of times per minute will be throttled and may starve the operator's recovery scripts. Production deployments cache responses and use change-feed APIs (EventBridge, Pub/Sub, Event Grid) instead of polling.

== Bare Metal and Dedicated Hosts

For workloads with hypervisor-incompatible licenses (Oracle, some HPC), regulatory isolation requirements, or hardware feature needs (custom MSRs, nested virt for Firecracker), the clouds offer bare-metal instances and dedicated hosts. Nitro's offload model means bare-metal instances retain managed networking and storage even without a hypervisor — a distinguishing capability versus traditional colo.

== Worked Example: Multi-AZ Web Service

```hcl
# Minimal multi-AZ on AWS via Terraform (full IaC treatment in iac.typ)
resource "aws_vpc" "main" { cidr_block = "10.0.0.0/16" }
resource "aws_subnet" "a" { vpc_id = aws_vpc.main.id; cidr_block = "10.0.1.0/24"; availability_zone = "us-east-1a" }
resource "aws_subnet" "b" { vpc_id = aws_vpc.main.id; cidr_block = "10.0.2.0/24"; availability_zone = "us-east-1b" }
resource "aws_lb" "web" { load_balancer_type = "application"; subnets = [aws_subnet.a.id, aws_subnet.b.id] }
resource "aws_autoscaling_group" "asg" {
  min_size = 2; max_size = 10
  vpc_zone_identifier = [aws_subnet.a.id, aws_subnet.b.id]
  target_group_arns = [aws_lb_target_group.web.arn]
}
```

The LB performs health checks; the ASG replaces failed instances; instances pull configuration from a metadata service and secrets from KMS-encrypted Parameter Store. This 30-line skeleton is what almost all production services are built on, regardless of cloud.

== Further Reading

Brooker, M. et al. (2017). "AWS Nitro System." AWS re:Invent.

Dalton, M. et al. (2018). "Andromeda: Performance, Isolation, and Velocity at Scale in Cloud Network Virtualization." NSDI.

Calder, B. et al. (2011). "Windows Azure Storage: A Highly Available Cloud Storage Service with Strong Consistency." SOSP.

DeCandia, G. et al. (2007). "Dynamo: Amazon's Highly Available Key-value Store." SOSP.

Verbitski, A. et al. (2017). "Amazon Aurora: Design Considerations for High Throughput Cloud-Native Relational Databases." SIGMOD.

Greenberg, A. et al. (2009). "VL2: A Scalable and Flexible Data Center Network." SIGCOMM.

Firestone, D. et al. (2018). "Azure Accelerated Networking: SmartNICs in the Public Cloud." NSDI.

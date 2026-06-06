= Serverless Computing

*Serverless* is a misnomer — servers exist, but the operator no longer provisions, patches, or capacity-plans them. The defining contract is *pay-per-invocation*: the provider bills for compute time consumed, not for idle capacity. This shifts the economic model from always-on reserved capacity toward a step function where cost scales exactly with load, making serverless ideal for spiky, event-driven, or rarely-triggered workloads. This chapter covers the execution model, cold start physics, Lambda internals, event sources, serverless containers, and the stateless constraints that govern system design.

*See also:* _iaas-fundamentals.typ_, _containers.typ_, _kubernetes-internals.typ_, _iac.typ_, `observability-and-sre/sre-fundamentals.typ`.

== The FaaS Execution Model

*Function-as-a-Service* ($"FaaS"$) presents the following contract:

+ The developer packages a function (code + dependencies) as a deployment artifact.
+ The platform receives an event, finds or creates an execution environment, and invokes the function handler.
+ The function runs to completion; the platform captures the return value and logs.
+ The environment may be reused for subsequent invocations (*warm*) or discarded.

The lifecycle of a single invocation is:

```text
Event source → trigger → cold start (if needed) → handler() → response
                                │
                     ┌──────────┴───────────────────────┐
                     │  Download image / extract code   │  ~200-500 ms
                     │  Start sandbox (microVM/cgroup)  │  ~50-200 ms
                     │  Init runtime (JVM, Node, etc.)  │  ~10-2000 ms
                     │  Run user init code (handler pkg)│  ~10-5000 ms
                     └──────────────────────────────────┘
                                Cold Start Overhead
```

After the handler returns, the environment is frozen (process paused, CPU deallocated) for a platform-defined window — typically 5-15 minutes of inactivity before the environment is destroyed. During the frozen window, a new invocation thaws the environment in microseconds: this is the *warm path*.

=== Concurrency Model

Each concurrent invocation requires its own execution environment; $"FaaS"$ platforms do not multiplex invocations within one environment. If 1000 requests arrive simultaneously, the platform launches up to 1000 parallel environments. This property, called *scale-to-zero-and-back*, is both the strength (automatic horizontal scaling) and the weakness (cold starts under burst traffic).

The concurrency limit per function is configurable and subject to account quotas. For AWS Lambda the default is 1000 concurrent executions per region; bursting above the baseline is rate-limited at 500 new environments per minute.

== Lambda Internals: Firecracker and MicroVMs

AWS Lambda uses *Firecracker*, an open-source Virtual Machine Monitor ($"VMM"$) built on $"KVM"$. Each Lambda execution environment is an independent microVM with:

- Dedicated kernel (Amazon Linux 2 or custom runtime).
- Fixed memory allocation (from 128 MiB to 10 GiB, user-selected).
- Ephemeral `/tmp` storage (512 MiB default, up to 10 GiB).
- No persistent disk; no inter-environment network.

Firecracker's boot time is under 125 ms because it implements only the minimal device model required: virtio-net, virtio-block, and a serial console. There is no $"BIOS"$, no $"ACPI"$ enumeration, no legacy PCI — components that account for seconds of conventional VM boot time.

```text
┌──────────────────────────────────────────────┐
│ Lambda Execution Environment (Firecracker VM)│
│  ┌────────────────────────────────────────┐  │
│  │ Amazon Linux 2 kernel                  │  │
│  │ Lambda runtime (node20, python3.12, …) │  │
│  │ Function code + layers                 │  │
│  │ /tmp (ephemeral)                       │  │
│  └────────────────────────────────────────┘  │
│  vCPU (burstable), fixed RAM                 │
└──────────────────────────────────────────────┘
     │                    │
  virtio-net           virtio-blk
     │                    │
  Lambda data plane    Lambda layer cache (S3-backed)
```

*Execution environment reuse:* When Lambda reuses a warm environment, global variables, database connections, and in-memory caches persist across invocations. This is both useful (connection pooling) and dangerous (shared mutable state, stale credentials).

```python
import boto3

# Runs ONCE per environment lifecycle — connection is reused across invocations.
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("orders")

def handler(event, context):
    # Runs on every invocation.
    item = table.get_item(Key={"id": event["order_id"]})
    return item["Item"]
```

== Cold Start Mitigation

Cold starts are the primary latency complaint in serverless architectures. The contributing factors, in order of magnitude:

#table(
  columns: 3,
  [*Factor*], [*Typical Cost*], [*Controllable?*],
  [MicroVM creation (Firecracker boot)], [50–125 ms], [No (platform)],
  [Runtime initialisation (Node, Python)], [10–50 ms], [Partially (layer caching)],
  [JVM / CLR initialisation], [500–3000 ms], [Yes (SnapStart, native image)],
  [User init code (imports, SDK clients)], [10–5000 ms], [Yes],
  [Layer / image download], [100–500 ms (cold host)], [No (platform cache)],
)

=== Provisioned Concurrency

AWS Lambda *Provisioned Concurrency* pre-warms a fixed number of environments, eliminating cold starts for that many parallel invocations. The trade-off is cost: provisioned environments are billed continuously at a lower rate than on-demand, approximating the economics of a reserved instance.

```bash
aws lambda put-provisioned-concurrency-config \
  --function-name my-api \
  --qualifier prod \
  --provisioned-concurrent-executions 50
```

Provisioned concurrency is commonly combined with Application Auto Scaling to track scheduled traffic patterns (e.g., ramp up before market open, ramp down overnight).

=== SnapStart

*Lambda SnapStart* (Java only, as of 2024) snapshots the initialised execution environment after the init phase and restores it for subsequent cold starts. The snapshot is a memory image (similar to a VM snapshot); restore time is typically under 200 ms regardless of JVM initialisation cost. The constraint: functions must be idempotent across restores and must not cache time-sensitive data (timestamps, random seeds) during init.

=== Language and Runtime Choice

Cold start latency correlates strongly with runtime initialisation cost:

#table(
  columns: 3,
  [*Runtime*], [*Typical Cold Start*], [*Notes*],
  [Rust (custom runtime)], [< 10 ms], [Minimal stdlib, no GC pause],
  [Python 3.x], [20–80 ms], [Fast import if deps are minimal],
  [Node.js 20], [40–100 ms], [V8 snapshot helps; avoid large bundles],
  [Go], [50–150 ms], [Compiled binary; no JVM overhead],
  [Java 21 (no SnapStart)], [800–3000 ms], [JVM class loading dominates],
  [Java 21 (SnapStart)], [100–200 ms], [Restore from memory snapshot],
  [.NET 8 (NativeAOT)], [30–80 ms], [AOT eliminates JIT warm-up],
)

== Event Sources and Triggers

Lambda and equivalent $"FaaS"$ functions are invoked by *event sources*. The invocation model differs by source type:

=== Synchronous Invocation

The caller blocks waiting for the response. Used for $"API"$ Gateway / $"ALB"$ (HTTP requests), $"SDK"$ direct invocation, and Step Functions tasks. Errors propagate back to the caller; retries are the caller's responsibility.

=== Asynchronous Invocation

Lambda queues the event internally and returns immediately. Used for S3 event notifications, $"SNS"$ topics, and `InvocationType=Event` $"SDK"$ calls. Lambda retries twice on failure and routes to a *Dead Letter Queue* ($"DLQ"$) or an *On-Failure* destination after exhausting retries.

=== Stream and Queue Polling

For Kinesis, DynamoDB Streams, $"SQS"$, and Kafka, Lambda acts as a *poll-based consumer*: the Lambda service reads records in batches and invokes the function. The *event source mapping* is a managed consumer with configurable batch size, parallelisation factor (Kinesis: up to 10 concurrent batches per shard), and bisect-on-error (binary search on a failing batch to isolate the bad record).

```python
def handler(event, context):
    for record in event["Records"]:
        body = record["body"]  # SQS message body
        process(body)
    # Returning normally → batch deleted from queue.
    # Raising exception → batch returns to queue (or DLQ after maxReceiveCount).
```

== Serverless Containers

Pure $"FaaS"$ imposes packaging constraints (zip or container ≤ 10 GiB uncompressed for Lambda) and execution time limits (15 minutes for Lambda). *Serverless containers* relax these by running full container workloads without managing nodes:

=== AWS Fargate

*Fargate* runs $"ECS"$ tasks or $"EKS"$ pods in ephemeral microVMs (Firecracker or Kata Containers) allocated on-demand. There are no $"EC2"$ nodes to manage; the unit of billing is vCPU-seconds and GB-seconds per running task. Cold start is higher than Lambda (~10-30 seconds for image pull + container start) but there is no 15-minute execution limit and the container is a standard OCI image with no Lambda-specific packaging.

=== Google Cloud Run

*Cloud Run* is Google's serverless container platform: it accepts any container image that listens on `PORT`, scales from 0 to 1000 instances on $"HTTP"$ traffic, and bills only while handling requests. Cloud Run's concurrency model differs from Lambda: a single container instance can handle multiple simultaneous requests (configurable; default 80), reducing cold starts dramatically because one instance absorbs many requests.

== Cost Model

The economic comparison between serverless and always-on depends on utilisation:

$ "cost"_"serverless" = N_"invocations" times t_"avg" times "price"_"GB-s" $

$ "cost"_"always-on" = N_"instances" times t_"running" times "price"_"instance" $

Serverless wins when $"utilisation"$ is low (below ~15–20% of a comparably-sized instance). Above that threshold, reserved instances are cheaper. The crossover depends on invocation duration: a 100 ms Lambda at \$0.0000166 per GB-s (128 MiB) costs roughly \$0.0000000021 per invocation — negligible until millions of invocations per second.

*Hidden costs* include:
- API Gateway: \$3.50 per million $"HTTP"$ API requests.
- Provisioned concurrency: billed continuously at \$0.000004646 per GB-s.
- Data transfer: same egress costs as $"EC2"$ — often the dominant term at scale.
- DynamoDB on-demand: \$1.25 per million write request units.

== Stateless Constraints and Workarounds

$"FaaS"$ environments are *ephemeral*: state written to the function's memory or to `/tmp` is lost when the environment is recycled. Production serverless systems externalise all state:

#table(
  columns: 3,
  [*State Type*], [*Serverless Solution*], [*Notes*],
  [Session / cache], [ElastiCache (Redis), DAX], [Sub-millisecond reads; connection pooling via RDS Proxy],
  [Durable key-value], [DynamoDB], [Single-digit ms; scales to any throughput with on-demand mode],
  [Relational], [Aurora Serverless v2, RDS Proxy], [RDS Proxy pools connections to avoid exhaustion],
  [Workflow state], [AWS Step Functions], [Durable execution graph with retry, timeout, and branching],
  [Message queues], [SQS, SNS, EventBridge], [Decouple producers from consumers; SQS provides at-least-once],
  [Object storage], [S3], [Artifacts, large results, inter-function payloads > 256 KB],
  [Secrets], [Secrets Manager, Parameter Store], [Cached per environment; TTL-refreshed],
)

=== Step Functions for Orchestration

Long-running processes that exceed the 15-minute Lambda limit are decomposed into state machine steps. Each state is a Lambda invocation; the Step Functions service durably persists the execution state between steps:

```json
{
  "Comment": "Order processing pipeline",
  "StartAt": "ValidateOrder",
  "States": {
    "ValidateOrder": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123:function:ValidateOrder",
      "Next": "ChargePayment",
      "Retry": [{"ErrorEquals": ["Lambda.ServiceException"], "MaxAttempts": 3}]
    },
    "ChargePayment": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:us-east-1:123:function:ChargePayment",
      "Next": "FulfillOrder",
      "Catch": [{"ErrorEquals": ["PaymentFailed"], "Next": "NotifyFailure"}]
    },
    "FulfillOrder":  { "Type": "Task", "Resource": "arn:aws:lambda:us-east-1:123:function:FulfillOrder", "End": true },
    "NotifyFailure": { "Type": "Task", "Resource": "arn:aws:lambda:us-east-1:123:function:Notify",        "End": true }
  }
}
```

== Vendor Comparison

#table(
  columns: 5,
  [*Feature*], [*AWS Lambda*], [*GCP Cloud Functions*], [*Azure Functions*], [*Cloudflare Workers*],
  [Max duration], [15 min], [60 min (gen 2)], [Unlimited (Premium)], [30 s (CPU time)],
  [Max memory], [10 GiB], [32 GiB (gen 2)], [14 GiB (Premium)], [128 MiB],
  [Cold start (Python)], [50–150 ms], [80–200 ms], [100–300 ms], [< 5 ms (V8 isolate)],
  [Concurrency model], [1 invocation/env], [1 invocation/env], [1 invocation/env], [Many req/isolate],
  [Isolation], [Firecracker microVM], [gVisor / Firecracker], [Hyper-V (Premium)], [V8 isolate],
  [VPC access], [Yes (ENI injection)], [Yes (Serverless VPC)], [Yes (VNet integration)], [Limited (Hyperdrive)],
  [Provisioned concurrency], [Yes], [Yes (min instances)], [Yes (Premium)], [Not needed],
  [Max deployment size], [50 MB zip / 10 GiB image], [100 MB], [500 MB], [1 MB (script)],
  [Free tier], [1M req/month], [2M req/month], [1M req/month], [100K req/day],
)

*Cloudflare Workers* use V8 isolates rather than microVMs: startup overhead is under 5 ms because there is no OS or kernel boundary — each worker is an isolate within a shared V8 heap. The trade-off is a severely constrained execution model (no arbitrary filesystem, no native modules, limited CPU time) and a smaller memory limit.

== Observability in Serverless

Serverless complicates observability because the unit of execution is the invocation, not the process. Key practices:

- *Structured logging* to CloudWatch / Cloud Logging with `aws_request_id` as the correlation key.
- *X-Ray / Cloud Trace* distributed tracing: Lambda auto-instruments the init and invocation phases; add $"SDK"$ segments for downstream calls.
- *Lambda Insights*: enhanced CloudWatch metrics (memory utilisation, init duration, CPU steal) published as a managed Lambda Extension.
- *Cold start tracking*: the `Init Duration` field in the $"REPORT"$ log line is only present on cold invocations — filter for it in log queries.

```text
REPORT RequestId: abc-123  Duration: 45.23 ms  Billed Duration: 46 ms
       Memory Size: 512 MB  Max Memory Used: 87 MB  Init Duration: 312.44 ms
```

The presence of `Init Duration` is the definitive indicator of a cold start.

== Further Reading

Agache, A. et al. (2020). "Firecracker: Lightweight Virtualization for Serverless Applications." NSDI.

Jonas, E. et al. (2019). "Cloud Programming Simplified: A Berkeley View on Serverless Computing." arXiv:1902.03383.

Eismann, S. et al. (2021). "A Review of Serverless Use Cases and Their Characteristics." arXiv:2008.11110.

Manner, J. et al. (2018). "Cold Start Influencing Factors in Function as a Service." CLOUD.

AWS. (2024). "Lambda Operator Guide." docs.aws.amazon.com/lambda/latest/operatorguide.

Brooker, M. (2023). "Lambda SnapStart: Fast Startup for Java Functions." AWS Blog.

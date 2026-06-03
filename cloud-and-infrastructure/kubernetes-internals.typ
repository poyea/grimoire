= Kubernetes Internals

Kubernetes is best understood as a *control loop platform built around a strongly-consistent key-value store*. The apiserver is the only client of etcd; everything else — controllers, schedulers, kubelets — watches the apiserver and writes back desired state through it. This level-triggered, declarative model is the deep structural decision that makes the system extensible (CRDs and custom controllers are first-class) and self-healing (every reconciliation step recomputes from current state, never from deltas).

*See also:* `containers.typ`, `service-mesh-deep-dive.typ`, `iac.typ`, `multi-tenancy.typ`, `networking/container-networking.typ`, `linux-kernel/cgroups-namespaces.typ`.

== Architectural Overview

```text
   ┌──────────────────────────────────────────────────────────┐
   │ Control Plane                                            │
   │ ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
   │ │ kube-       │  │ kube-        │  │ kube-controller-│  │
   │ │ apiserver   │  │ scheduler    │  │ manager         │  │
   │ └──────┬──────┘  └──────┬───────┘  └────────┬────────┘  │
   │        │ watch/write    │ watch              │ watch    │
   │        ▼                                     │          │
   │   ┌─────────┐                                │          │
   │   │  etcd   │ ◀──────────────────────────────┘          │
   │   └─────────┘                                            │
   └──────────────────────────────────────────────────────────┘
                          ▲
                          │ watch/list
   ┌──────────────────────┴──────────────────────────────────┐
   │ Node 1 ... Node N                                       │
   │  kubelet → CRI → containerd → runc                      │
   │  kube-proxy / CNI / CSI                                 │
   └─────────────────────────────────────────────────────────┘
```

== etcd as the System of Record

etcd is a Raft-replicated KV store with a watch primitive that streams ordered revisions. Every Kubernetes object is one etcd key:

```text
/registry/pods/default/web-7d8f                → Pod object (protobuf)
/registry/services/default/web                 → Service object
/registry/deployments/default/web              → Deployment object
```

Each write increments a monotonic `revision`; clients can `watch` from a specific revision to receive all subsequent changes. This is the foundation of the informer/cache pattern in client-go.

*Sizing/perf:* etcd's recommended limit is 8 GB; clusters exceeding this paginate poorly. Large clusters typically split unrelated CRDs to a separate apiserver+etcd via API aggregation.

== The apiserver: REST + Admission

The apiserver enforces a strict request pipeline:

+ Authentication (X.509, OIDC, ServiceAccount JWT, webhook).
+ Authorization (RBAC, ABAC, webhook).
+ Mutating admission (webhooks, e.g. inject sidecars).
+ Schema validation (OpenAPI, structural schema for CRDs).
+ Validating admission (webhooks, OPA/Gatekeeper, Kyverno).
+ Storage to etcd.

```yaml
# Example ValidatingAdmissionPolicy (built-in CEL admission, GA in 1.30)
apiVersion: admissionregistration.k8s.io/v1
kind: ValidatingAdmissionPolicy
metadata: {name: "require-resource-limits"}
spec:
  failurePolicy: Fail
  matchConstraints:
    resourceRules:
    - apiGroups: [""]; apiVersions: ["v1"]; operations: ["CREATE","UPDATE"]; resources: ["pods"]
  validations:
  - expression: "object.spec.containers.all(c, has(c.resources.limits.memory))"
    message: "every container must declare memory limits"
```

CEL-based admission replaces many webhook deployments and removes the latency hop on hot paths.

== Controllers and the Reconciliation Pattern

A controller is a loop:

```go
for {
  obj := <-workqueue
  current := lister.Get(obj.Key())   // local cache (informer)
  desired := derive(current)          // pure function of spec + cluster state
  if !equal(currentStatus, desired) {
    apiserver.Update(desired)
  }
}
```

Informers maintain a local cache by long-polling the apiserver watch endpoint, providing O(1) reads and "list once, watch forever" semantics. The workqueue rate-limits retries (exponential backoff with jitter).

*Built-in controllers* (in kube-controller-manager): ReplicaSet, Deployment, StatefulSet, DaemonSet, Job, CronJob, Node, Endpoint, ServiceAccount, GC, Namespace, PV/PVC binders.

== The Scheduler

The scheduler watches unscheduled pods, scores nodes, and writes the binding:

```text
   Pending Pod  ─┐
                ▼
   ┌─────────────────────────────────────────────────┐
   │ Filter   (predicates): drop infeasible nodes    │
   │   NodeAffinity, Resources, Taints/Tolerations,  │
   │   VolumeBinding, PodAffinity, TopologySpread    │
   ├─────────────────────────────────────────────────┤
   │ Score    (priorities): rank feasible nodes      │
   │   LeastAllocated, BalancedAllocation, ImageLoc  │
   ├─────────────────────────────────────────────────┤
   │ Reserve / Permit / Bind                         │
   └─────────────────────────────────────────────────┘
```

Plugins extend each stage via the scheduling framework (Go interface; no webhook hop). For large clusters, the scheduler subsamples nodes (e.g. 5%) when feasible nodes are abundant — covered in the Borg paper as well.

== kubelet and the CRI

```text
kubelet
  watch pods on this node
  ├── PodSyncLoop (every 10s + on event)
  ├── ProbeWorker (liveness/readiness/startup)
  ├── ImageGC, ContainerGC
  └── PLEG (pod lifecycle event generator)
       │ CRI
       ▼
  containerd  → runc  → container
```

The CRI is a gRPC API with two services: `RuntimeService` (sandboxes, containers, exec/attach/portforward) and `ImageService` (pull, list, remove).

== Networking: CNI

CNI plugins implement pod networking. The model: a single flat L3 fabric where every pod has a routable IP; no NAT between pods.

#table(
  columns: 5,
  [*CNI*], [*Encapsulation*], [*Data Plane*], [*Policy*], [*Notes*],
  [Calico], [native routing / IPIP / VXLAN], [iptables / eBPF], [yes (NetworkPolicy)], [BGP between nodes],
  [Cilium], [VXLAN or native], [eBPF (XDP/tc)], [yes + L7], [identity-based, replaces kube-proxy],
  [Flannel], [VXLAN / host-gw], [iptables], [no], [simple, demo-grade],
  [AWS VPC CNI], [none (ENI per pod)], [native VPC], [via SG], [pod IP = VPC IP],
  [Azure CNI], [none], [native VNet], [via NSG], [pod IP from subnet],
  [Cilium + Cluster Mesh], [WireGuard / native], [eBPF], [global identity], [multi-cluster],
)

*eBPF cut-through (Cilium):* an XDP program at the NIC drop point can route a packet for a known pod IP directly to the pod's veth without traversing iptables, conntrack, or the bridge — saving 5-15 us per packet.

```c
// Simplified eBPF cut: lookup destination pod and redirect at XDP.
SEC("xdp")
int pod_redirect(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *end  = (void *)(long)ctx->data_end;
    struct ethhdr *eth = data;
    if ((void*)(eth + 1) > end) return XDP_PASS;
    struct iphdr *ip = (void*)(eth + 1);
    if ((void*)(ip + 1) > end) return XDP_PASS;
    __u32 dst = ip->daddr;
    struct pod_ep *ep = bpf_map_lookup_elem(&pod_map, &dst);
    if (!ep) return XDP_PASS;
    return bpf_redirect(ep->ifindex, 0);
}
```

== Services and kube-proxy

A `Service` is a stable virtual IP fronting a set of pods (`Endpoints` / `EndpointSlice`). kube-proxy programs the data plane:

- *iptables mode:* O(N) chains; convergence quadratic at thousands of endpoints.
- *ipvs mode:* hash-based; constant time at scale.
- *eBPF mode (Cilium):* socket-level redirect, bypasses iptables entirely.

```yaml
apiVersion: v1
kind: Service
metadata: {name: web}
spec:
  selector: {app: web}
  ports: [{port: 80, targetPort: 8080}]
  type: ClusterIP
```

== Storage: CSI

CSI standardises external storage drivers. The driver runs as a sidecar pair: a *controller plugin* (CreateVolume, ControllerPublishVolume) and a *node plugin* (NodeStageVolume, NodePublishVolume). The kubelet calls the node plugin via gRPC over a Unix socket.

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata: {name: gp3}
provisioner: ebs.csi.aws.com
parameters: {type: gp3, iops: "6000", throughput: "250"}
volumeBindingMode: WaitForFirstConsumer    # zone match scheduler
reclaimPolicy: Delete
```

== Custom Resources and Operators

CRDs add new API types backed by etcd; an operator is a controller that reconciles them. This is how databases (Vitess, CockroachDB), pipelines (Argo, Tekton), and ML platforms (Kubeflow) ship on Kubernetes.

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata: {name: rediscluster.cache.example.com}
spec:
  group: cache.example.com
  scope: Namespaced
  names: {kind: RedisCluster, plural: rediscluster, singular: rediscluster}
  versions:
  - name: v1
    served: true; storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            required: [replicas]
            properties:
              replicas: {type: integer, minimum: 3, maximum: 99}
              version:  {type: string}
```

== Workload Identity

Pods authenticate to cloud APIs via projected ServiceAccount tokens that are exchanged for cloud credentials at a federated IAM endpoint (IRSA on EKS, Workload Identity on GKE, AKS Workload Identity). No long-lived secrets on disk.

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: report-writer
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123:role/ReportWriter
```

== Multi-Tenancy Considerations (preview)

- *Namespaces* are not security boundaries by themselves; RBAC + NetworkPolicy + ResourceQuota + PSA are needed.
- *Cluster-per-tenant* gives strong isolation but costs control plane resources.
- *Virtual clusters* (vcluster) run a tenant apiserver as a pod, sharing host nodes.

See `multi-tenancy.typ` for the full treatment.

== Scaling Limits

Kubernetes' tested limits (community SIG-Scalability): 5000 nodes, 150 000 pods, 300 000 containers, 100 pods/sec start rate. Beyond this, the apiserver memory, etcd write throughput, and watch fan-out become bottlenecks. Hyperscalers run thousands of clusters federated by GitOps tools rather than scaling individual clusters past these thresholds.

== Worked Example: A Deployment's Lifecycle

+ `kubectl apply -f deploy.yaml` → POST to apiserver.
+ Admission, validation; Deployment stored in etcd.
+ Deployment controller observes; creates ReplicaSet.
+ ReplicaSet controller creates N Pods.
+ Pods are unscheduled; scheduler binds them.
+ kubelet on each node sees its pod (watch); calls CRI to pull image, set up cgroups, network (CNI), volumes (CSI).
+ Container running; kubelet reports status; controllers reconcile; aggregated status appears in `kubectl get deploy`.

Every step is level-triggered: re-running the loop is safe and the only correct behaviour under loss.

== Further Reading

Burns, B., Grant, B., Oppenheimer, D., Brewer, E., Wilkes, J. (2016). "Borg, Omega, and Kubernetes." CACM.

Verma, A. et al. (2015). "Large-scale cluster management at Google with Borg." EuroSys.

Schwarzkopf, M. et al. (2013). "Omega: flexible, scalable schedulers for large compute clusters." EuroSys.

Hindman, B. et al. (2011). "Mesos: A Platform for Fine-Grained Resource Sharing in the Data Center." NSDI.

Kubernetes SIG-Scalability: "Kubernetes Scalability Thresholds." k8s.io/docs.

Cilium docs and "eBPF Datapath" whitepaper, Isovalent 2023.

CRI, CNI, CSI specifications (kubernetes/community).

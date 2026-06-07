= Multi-Tenancy

Multi-tenancy is the practice of serving multiple customers or teams from a shared infrastructure pool while providing isolation strong enough that each tenant experiences the system as if it were dedicated to them. Kubernetes has become the dominant substrate for cloud-native multi-tenancy, offering isolation at progressively stronger levels — namespace, virtual cluster, and physical cluster — each with a distinct cost, operational overhead, and threat-model profile. This chapter works through the full stack: isolation models, threat model, RBAC design, network policy, resource governance, security admission, audit logging, and fleet management.

*See also:* _kubernetes-internals.typ_, _containers.typ_, _iac.typ_, `cloud-and-infrastructure/iaas-fundamentals.typ`.

== Isolation Models

The choice of isolation model is primarily a function of the required blast radius and compliance posture:

#table(
  columns: 4,
  [*Model*], [*Shared components*], [*Blast radius*], [*Relative cost*],
  [Namespace-per-tenant], [Control plane, nodes, CNI, etcd], [Cluster-wide], [1×],
  [Virtual cluster (vcluster)], [Host nodes, CNI underlay], [Node pool], [1.05–1.2×],
  [Node-pool-per-tenant], [Control plane, CNI], [Node pool], [1.1–1.3×],
  [Cluster-per-tenant (hard multi-tenancy)], [Nothing (dedicated)], [Single cluster], [3–5×],
)

*Namespace-per-tenant* is the lowest-cost option and sufficient when all tenants are internal engineering teams under a single trust boundary. The control plane and etcd are shared; a bug in one tenant's workload cannot corrupt another tenant's etcd keys, but a control-plane outage affects all tenants simultaneously.

*Cluster-per-tenant* provides the strongest isolation — separate etcd, separate control-plane processes, separate node pools — and is the correct choice when tenants are external customers with contractual SLA obligations, when different regulatory regimes apply (e.g., one tenant is PCI-DSS scoped, another is not), or when tenants need to bring their own Kubernetes version or custom admission webhooks.

*Virtual clusters* occupy the middle ground and are discussed in detail in a later section.

== Threat Model

=== Blast Radius

The *blast radius* of a tenant compromise is the set of resources an attacker can reach after escaping the tenant's intended isolation boundary. In a namespace model:

- A compromised pod with `cluster-admin` via a misconfigured RBAC binding can read secrets across all namespaces.
- A container escape (CVE-class vulnerability in the container runtime or kernel) reaches the host node and potentially all pods co-scheduled on it.
- A malicious admission webhook registered by one tenant can intercept admission requests for other namespaces if the `webhookConfiguration` is not namespace-scoped.

Blast radius analysis should be a named artifact in the architecture review, expressed as a directed graph from each tenant boundary to each class of sensitive resource (secrets, PVCs, other tenants' pods).

=== Noisy Neighbour

Without *ResourceQuota* and *LimitRange*, a single tenant can monopolise CPU, memory, API request rate, and etcd storage. CPU throttling is invisible to the tenant application (processes simply run slower), making it one of the hardest noisy-neighbour effects to diagnose. The mitigation is covered in the Resource Governance section.

=== Data Exfiltration

The most common data exfiltration path in shared clusters is misconfigured *RBAC* allowing a pod in namespace A to `get` or `list` secrets in namespace B. The second most common is pod-to-pod network reachability in the absence of NetworkPolicy — covered below.

== RBAC Design Patterns

=== Least Privilege

The foundational rule: no service account should have a `ClusterRole`. Everything should be a `Role` scoped to the tenant's namespace. The one exception is the cluster's own infrastructure controllers (e.g., the cert-manager controller must read `ClusterIssuer` resources).

=== Namespace-Scoped Role Pattern

```yaml
# Grant a CI pipeline read-only access to its own namespace only
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ci-reader
  namespace: tenant-acme
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log", "services", "configmaps"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["apps"]
    resources: ["deployments", "replicasets"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: ci-reader-binding
  namespace: tenant-acme
subjects:
  - kind: ServiceAccount
    name: ci-runner
    namespace: tenant-acme
roleRef:
  kind: Role
  name: ci-reader
  apiGroup: rbac.authorization.k8s.io
```

=== Aggregated ClusterRoles

For platform teams that manage multiple tenants, *aggregated ClusterRoles* allow composing fine-grained roles without touching tenant namespaces:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: platform-readonly
  labels:
    rbac.platform.io/aggregate-to-platform-ops: "true"
aggregationRule:
  clusterRoleSelectors:
    - matchLabels:
        rbac.platform.io/aggregate-to-platform-ops: "true"
rules: []   # filled by aggregation controller
```

Any `ClusterRole` with the matching label is merged in automatically — new CRDs and resources are picked up without editing the operator's binding.

== NetworkPolicy Recipes

By default, all pods in a Kubernetes cluster can communicate with all other pods. A *default-deny* posture must be explicitly configured.

=== Default Deny All

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: tenant-acme
spec:
  podSelector: {}       # matches all pods in namespace
  policyTypes:
    - Ingress
    - Egress
```

Apply this to every new tenant namespace as part of the namespace provisioning pipeline.

=== Allow Same-Namespace

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-same-namespace
  namespace: tenant-acme
spec:
  podSelector: {}
  policyTypes: [Ingress, Egress]
  ingress:
    - from:
        - podSelector: {}   # any pod in the same namespace
  egress:
    - to:
        - podSelector: {}
```

=== Ingress from Ingress Controller Only

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-ingress-controller
  namespace: tenant-acme
spec:
  podSelector:
    matchLabels:
      app: frontend
  policyTypes: [Ingress]
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ingress-nginx
          podSelector:
            matchLabels:
              app.kubernetes.io/name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8080
```

Note that NetworkPolicy is enforced by the CNI plugin; Cilium and Calico provide richer L7 policy (HTTP path, DNS name) beyond the standard L3/L4 API.

== Resource Governance

=== ResourceQuota

```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: tenant-acme-quota
  namespace: tenant-acme
spec:
  hard:
    requests.cpu: "8"
    requests.memory: 16Gi
    limits.cpu: "16"
    limits.memory: 32Gi
    persistentvolumeclaims: "10"
    services.loadbalancers: "2"
    count/secrets: "50"
```

Quota enforcement is exact: if a Pod creation would exceed any dimension, the API server rejects the request with a `403 Forbidden`. Setting `limits.cpu` higher than `requests.cpu` by a fixed ratio (here 2:1) encodes the cluster's overcommit policy for CPU while keeping memory non-overcommitted (1:1 here due to swap-less nodes).

=== LimitRange

`ResourceQuota` governs namespace totals; `LimitRange` governs per-object defaults and maxima, preventing pods from being submitted without resource declarations (which would count zero against the quota but still consume real resources):

```yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: tenant-acme-limits
  namespace: tenant-acme
spec:
  limits:
    - type: Container
      default:
        cpu: "500m"
        memory: 256Mi
      defaultRequest:
        cpu: "100m"
        memory: 128Mi
      max:
        cpu: "4"
        memory: 8Gi
    - type: PersistentVolumeClaim
      max:
        storage: 50Gi
```

Together, `ResourceQuota` + `LimitRange` implement the noisy-neighbour guarantee: $"CPU"_"tenant" <= "quota"."limits"."cpu"$ at all times.

== Pod Security Admission

*Pod Security Admission* ($"PSA"$) replaced PodSecurityPolicy in Kubernetes 1.25. It defines three levels:

#table(
  columns: 3,
  [*Level*], [*What it blocks*], [*Typical use*],
  [`privileged`], [Nothing], [System namespaces (`kube-system`)],
  [`baseline`], [Host namespaces, privileged containers, hostPath except explicit allow-list], [Internal teams],
  [`restricted`], [Everything in baseline + requires non-root UID, drops all capabilities, disallows seccomp: Unconfined], [Tenant workloads, PCI scope],
)

Enforce `restricted` by labelling each tenant namespace:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: tenant-acme
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/audit: restricted
```

The `warn` and `audit` labels emit warnings and audit log events without blocking — use these in `staging` namespaces before enforcing in `prod`.

== Audit Logging for Compliance

The Kubernetes API server emits an *audit log* for every request. For PCI-DSS and SOC 2 compliance, the audit policy must capture reads and writes to sensitive resources:

```yaml
# /etc/kubernetes/audit-policy.yaml (excerpt)
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
  # Log all secret access at RequestResponse level (captures value!)
  - level: RequestResponse
    resources:
      - group: ""
        resources: ["secrets"]
  # Log pod exec and port-forward
  - level: RequestResponse
    resources:
      - group: ""
        resources: ["pods/exec", "pods/portforward", "pods/proxy"]
  # Log RBAC changes
  - level: RequestResponse
    resources:
      - group: "rbac.authorization.k8s.io"
        resources: ["roles", "rolebindings", "clusterroles", "clusterrolebindings"]
  # Metadata only for read-heavy resources
  - level: Metadata
    resources:
      - group: ""
        resources: ["pods", "services", "configmaps"]
  # Don't log health checks
  - level: None
    users: ["system:kube-proxy"]
    verbs: ["watch"]
    resources:
      - group: ""
        resources: ["endpoints", "services"]
```

Ship audit logs to an immutable sink (CloudTrail S3 + Object Lock, GCS with retention lock, Azure Blob immutable storage) and retain for 90 days (PCI-DSS) or one year (SOC 2 Type II). Alert on `verb=delete` against `secrets` or RBAC resources outside of approved CI pipelines.

== Virtual Cluster Architecture

*vcluster* (by Loft Labs) creates a fully functional Kubernetes control plane running inside a namespace of a *host cluster*. Each virtual cluster has its own API server, scheduler, and controller manager running as pods. Workloads scheduled by the virtual cluster's scheduler are translated to host-cluster pods via a sync controller.

```text
Host Cluster
└── Namespace: vc-tenant-acme
    ├── Pod: vcluster-api-server    (k3s or k8s API server)
    ├── Pod: vcluster-controller    (sync controller)
    └── Synced workload pods        (appear in host namespace, vc-tenant-acme-*)
```

From the tenant's perspective, they have a standard `kubeconfig` pointing at their API server with full `cluster-admin` rights inside the vcluster. From the host cluster's perspective, the tenant's pods are ordinary pods subject to host ResourceQuota and NetworkPolicy.

Trade-offs versus a dedicated cluster:

- *Pro:* tenant gets `cluster-admin`; can install CRDs and admission webhooks without affecting other tenants.
- *Pro:* control-plane cost is ~0.5 CPU + 1 GiB RAM per vcluster, versus ~3 nodes for a managed cluster.
- *Con:* node-level isolation is still shared; a container escape reaches the host node.
- *Con:* API server latency adds one hop (tenant API → sync → host API).

== Fleet Management

At scale, managing dozens or hundreds of clusters requires a *fleet management* layer.

=== Cluster API

*Cluster API* ($"CAPI"$) is a Kubernetes-native API for provisioning and lifecycle-managing clusters. A `Cluster` custom resource describes the desired control plane and node pool; CAPI controllers reconcile against AWS ($"CAPA"$), GCP ($"CAPG"$), Azure ($"CAPZ"$), or vSphere infrastructure providers.

```yaml
apiVersion: cluster.x-k8s.io/v1beta1
kind: Cluster
metadata:
  name: tenant-acme-prod
  namespace: fleet-management
spec:
  clusterNetwork:
    pods:
      cidrBlocks: ["10.128.0.0/16"]
  controlPlaneRef:
    apiVersion: controlplane.cluster.x-k8s.io/v1beta1
    kind: KubeadmControlPlane
    name: tenant-acme-prod-cp
  infrastructureRef:
    apiVersion: infrastructure.cluster.x-k8s.io/v1beta2
    kind: AWSCluster
    name: tenant-acme-prod
```

Day-2 operations — node pool scaling, Kubernetes version upgrades, certificate rotation — are expressed as updates to these CRDs and reconciled automatically.

=== ACM and Rancher

*Red Hat Advanced Cluster Management* ($"ACM"$) and *Rancher* provide a management-plane UI and policy engine across heterogeneous clusters. Both use a hub-spoke model: a management cluster hosts the ACM/Rancher controllers; spoke clusters run an agent that receives `Policy` or `ManagedCluster` objects and enforces them locally. This enables fleet-wide enforcement of:

- Namespace existence and labelling.
- Required ResourceQuota and LimitRange.
- Gatekeeper / OPA policies (e.g., "no images from unapproved registries").
- GitOps sync via ArgoCD or Flux, with per-cluster overrides.

== Further Reading

Rice, L. (2020). _Container Security._ O'Reilly.

Kubernetes SIG Auth (2024). "Pod Security Admission." kubernetes.io/docs.

Loft Labs (2024). "vcluster Documentation." vcluster.com/docs.

Cluster API Authors (2024). "The Cluster API Book." cluster-api.sigs.k8s.io.

PCI Security Standards Council (2022). _PCI DSS v4.0._ pcisecuritystandards.org.

Gilman, E. and Barth, D. (2017). _Zero Trust Networks._ O'Reilly.

Burns, B. et al. (2016). "Borg, Omega, and Kubernetes." ACM Queue.

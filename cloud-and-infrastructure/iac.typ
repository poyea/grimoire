#import "../template.typ": xref

= Infrastructure as Code

Managing cloud resources through version-controlled, declarative configuration files — rather than through consoles or ad-hoc scripts — is now the baseline expectation for production systems. *Infrastructure as Code* ($"IaC"$) makes topology reproducible, auditable, and testable: the diff of a pull request is the diff of the infrastructure. This chapter covers the dominant toolchains, their execution models, and the GitOps and policy-as-code practices that tie them together.

*See also:* #xref("cloud-and-infrastructure", "iaas-fundamentals", label: "IaaS Fundamentals"), #xref("cloud-and-infrastructure", "kubernetes-internals", label: "Kubernetes Internals"), #xref("cloud-and-infrastructure", "containers", label: "Containers: OCI, runc, containerd, Image Layers"), #xref("observability-and-sre", "the-three-pillars-and-beyond", label: "The Three Pillars and Beyond") (observability-and-sre), #xref("distributed-systems", "consensus-deep-dive", label: "Consensus Deep Dive") (distributed-systems).

== Declarative vs Imperative IaC

*Declarative* tools (Terraform, CloudFormation, Pulumi) express desired state; the engine computes the delta and applies it. *Imperative* tools (Ansible, shell scripts) express steps to run; idempotency is the author's problem. Declarative IaC dominates for long-lived infrastructure because the tool owns the reconciliation loop and can detect *drift* — resources that differ from the desired state due to manual edits, cloud-side events, or external automation.

The core invariant of declarative IaC is:

$ "apply"("desired state", "current state") -> "actions" $

The actions minimise the edit distance between current and desired, subject to provider-specific ordering constraints (e.g., a security group must exist before an instance that references it).

== Terraform

*Terraform* (HashiCorp) is the lingua franca of multi-cloud IaC. Resources are described in *HCL* (HashiCorp Configuration Language), a JSON superset with interpolation and expressions. Terraform computes a dependency graph from references between resources, then executes creation, update, and deletion in dependency order with parallelism where the graph permits.

=== Providers and the Plugin Architecture

A *provider* is a Go plugin that maps HCL resources to API calls. The AWS provider alone wraps several thousand resource types. Providers are fetched from the Terraform Registry and pinned by hash in `.terraform.lock.hcl`:

```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.50"
    }
  }
  required_version = ">= 1.8"
}

provider "aws" {
  region = var.aws_region
}
```

=== State

Terraform maintains a *state file* (`terraform.tfstate`) that records the last-known mapping from logical resource names to provider-assigned IDs. Without state, Terraform cannot determine what already exists and would attempt to re-create everything on every apply.

In team environments, state must be stored remotely with *locking* to prevent concurrent mutations. The standard pattern is S3 + DynamoDB:

```hcl
terraform {
  backend "s3" {
    bucket         = "my-org-tf-state"
    key            = "prod/networking/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    kms_key_id     = "arn:aws:kms:us-east-1:123456789012:key/mrk-abc"
    dynamodb_table = "terraform-lock"
  }
}
```

DynamoDB uses a conditional write on the lock item; the first writer wins and all others receive a `ConditionalCheckFailedException`. State files should be encrypted at rest (KMS) and access-controlled to prevent credential leakage via state outputs.

=== Plan / Apply Lifecycle

```text
terraform init      # fetch providers, initialise backend
terraform validate  # parse and type-check HCL
terraform plan      # diff desired vs state → execution plan
terraform apply     # execute plan, update state
terraform destroy   # special apply that removes all resources
```

The *plan* is a serialisable JSON document (`-out=plan.bin`) that can be reviewed in CI before a human approves `apply`. This separation is the key gate in GitOps pipelines.

=== Workspaces

*Workspaces* provide isolated state files within one backend path, enabling environment separation (dev/staging/prod) without duplicating configuration:

```bash
terraform workspace new staging
terraform workspace select staging
terraform apply -var-file=staging.tfvars
```

Workspaces are best for environments that share the same code but differ by variable values. For fundamentally different configurations, separate root modules are cleaner.

=== Modules

A *module* is a reusable directory of `.tf` files with declared inputs (`variable`) and outputs (`output`). Modules are the primary abstraction mechanism:

```hcl
module "vpc" {
  source  = "./modules/vpc"
  name    = "prod-vpc"
  cidr    = "10.0.0.0/16"
  azs     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}
```

=== Worked Example: VPC Module

The following module creates a VPC, one public subnet per AZ, and a security group allowing $"HTTPS"$ ingress:

```hcl
# modules/vpc/variables.tf
variable "name"  { type = string }
variable "cidr"  { type = string }
variable "azs"   { type = list(string) }

# modules/vpc/main.tf
resource "aws_vpc" "this" {
  cidr_block           = var.cidr
  enable_dns_hostnames = true
  tags = { Name = var.name }
}

resource "aws_subnet" "public" {
  count                   = length(var.azs)
  vpc_id                  = aws_vpc.this.id
  cidr_block              = cidrsubnet(var.cidr, 4, count.index)
  availability_zone       = var.azs[count.index]
  map_public_ip_on_launch = true
  tags = { Name = "${var.name}-public-${var.azs[count.index]}" }
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id
  tags   = { Name = "${var.name}-igw" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }
  tags = { Name = "${var.name}-public-rt" }
}

resource "aws_route_table_association" "public" {
  count          = length(var.azs)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "web" {
  name        = "${var.name}-web-sg"
  description = "Allow HTTPS inbound"
  vpc_id      = aws_vpc.this.id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = { Name = "${var.name}-web-sg" }
}

# modules/vpc/outputs.tf
output "vpc_id"            { value = aws_vpc.this.id }
output "public_subnet_ids" { value = aws_subnet.public[*].id }
output "web_sg_id"         { value = aws_security_group.web.id }
```

The `cidrsubnet(var.cidr, 4, count.index)` call divides the /16 into /20s: index 0 → `10.0.0.0/20`, index 1 → `10.0.16.0/20`, etc.

== Pulumi

*Pulumi* uses general-purpose languages (TypeScript, Python, Go, Java, C\#) to express the same resource graph that Terraform expresses in HCL. The trade-off: real loops, conditionals, and abstractions without HCL's workarounds, at the cost of a heavier runtime and slower iteration on schema changes.

```python
import pulumi_aws as aws

vpc = aws.ec2.Vpc("main", cidr_block="10.0.0.0/16",
                  enable_dns_hostnames=True)

subnets = [
    aws.ec2.Subnet(f"public-{i}",
                   vpc_id=vpc.id,
                   cidr_block=f"10.0.{i}.0/24",
                   availability_zone=f"us-east-1{'abc'[i]}")
    for i in range(3)
]
```

Pulumi's state model is identical to Terraform's: a backend (Pulumi Cloud, S3, Azure Blob) stores the resource graph. The execution model is also identical: a language runtime evaluates the program, registers resource declarations, and Pulumi's engine diffs against state.

== AWS CDK

The *AWS Cloud Development Kit* ($"CDK"$) synthesises CloudFormation templates from TypeScript/Python/Java code. Unlike Pulumi, CDK is AWS-only and the output is CloudFormation JSON; the cloud-side execution engine is CloudFormation changesets. CDK's *constructs* are layered: L1 constructs map 1:1 to CloudFormation resources; L2 constructs add opinionated defaults; L3 (patterns) compose multiple L2s into common architectures (e.g., `ApplicationLoadBalancedFargateService`).

== Ansible vs Declarative IaC

*Ansible* is a push-based configuration management tool: tasks run on remote hosts via $"SSH"$, mutating state step by step. It is well-suited for software installation, file templating, and OS-level configuration — tasks where "desired state" is hard to describe declaratively. The friction appears when Ansible is used for cloud resource management: idempotency relies on each module individually checking whether the resource exists, which is brittle and slow compared to Terraform's unified state model.

The practical division: Terraform owns cloud resources (VPCs, instances, databases), Ansible owns what runs on those instances (packages, services, configuration files). The two are often combined — Terraform provisions the instance, outputs its IP, Ansible picks it up from the Terraform state or inventory.

== Drift Detection

*Drift* is the divergence of real infrastructure from the IaC state. Causes include: manual console edits, cloud-side auto-remediation, third-party tooling, and provider bugs. Terraform detects drift on `plan` via the *refresh* step (provider `Read` calls); CloudFormation has a dedicated drift detection API. Continuous drift detection runs `terraform plan` on a schedule and alerts on non-empty plans.

Drift is a leading indicator of operational risk: a drifted resource means the IaC apply will produce a surprise, ranging from a no-op to a destructive replacement.

== GitOps

*GitOps* extends the IaC model: the Git repository is the single source of truth, and a controller continuously reconciles the cluster or cloud to match HEAD. Two schools of thought:

- *Push-based GitOps (CI/CD):* A pipeline triggers on merge, runs `terraform apply`, and updates state. Simple but couples apply to pipeline availability.
- *Pull-based GitOps:* A controller running inside the target environment polls Git and applies changes. More resilient — the cluster heals itself even if the CI system is down.

*Flux* and *ArgoCD* implement pull-based GitOps for Kubernetes manifests. For Terraform, *Atlantis* implements push-based GitOps: it comments a plan on a PR and applies on merge. *Terraform Cloud* / *Spacelift* / *env0* offer managed runners with plan/apply approval workflows.

=== ArgoCD Application Spec

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: nginx-prod
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/my-org/k8s-manifests
    targetRevision: main
    path: apps/nginx/overlays/prod
  destination:
    server: https://kubernetes.default.svc
    namespace: nginx
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

`selfHeal: true` causes ArgoCD to revert manual `kubectl` edits, enforcing Git as the authority.

== Policy as Code

*Policy as code* applies the same version-control and review discipline to compliance rules that IaC applies to resources. Two major systems:

=== OPA and Conftest

*Open Policy Agent* ($"OPA"$) evaluates Rego policies against arbitrary JSON. *Conftest* wraps $"OPA"$ for CI: it reads Terraform plan JSON, Kubernetes manifests, Dockerfiles, etc. and evaluates policies:

```rego
# policy/no_public_s3.rego
package main

deny[msg] {
  r := input.resource_changes[_]
  r.type == "aws_s3_bucket_public_access_block"
  r.change.after.block_public_acls == false
  msg := sprintf("S3 bucket %v must block public ACLs", [r.address])
}
```

```bash
terraform show -json plan.bin | conftest test -
```

=== Sentinel

*Sentinel* is HashiCorp's policy framework, embedded in Terraform Cloud and Vault. Policies are evaluated in the plan phase before `apply` is permitted:

```python
import "tfplan/v2" as tfplan

main = rule {
  all tfplan.resource_changes as _, changes {
    changes.type is not "aws_iam_access_key" or
    changes.change.actions contains "delete"
  }
}
```

Sentinel enforces that no `aws_iam_access_key` is ever created (keys should use IAM roles + IRSA instead).

== Further Reading

Brikman, Y. (2022). _Terraform: Up & Running_, 3rd ed. O'Reilly.

HashiCorp. (2024). "Terraform Language Documentation." developer.hashicorp.com/terraform/language.

Pulumi. (2024). "Pulumi Architecture and Concepts." pulumi.com/docs/concepts.

Weaveworks. (2021). "GitOps: Operating Model for Cloud Native." gitops.tech.

Open Policy Agent. (2024). "OPA Documentation." openpolicyagent.org/docs.

Rahman, A. et al. (2019). "Seven Reasons Why Infrastructure-as-Code is so Difficult." ICSME.

Weiss, M. (2023). "Drift Detection at Scale." HashiConf 2023.

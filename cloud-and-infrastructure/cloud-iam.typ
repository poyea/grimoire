#import "../template.typ": xref

= Cloud Identity and Access Management <cloud-iam>

In a data center, the network perimeter was the main security boundary. In the cloud, every resource is reachable through a public control-plane API, so identity is the perimeter. Every API call is authenticated as some principal and authorised by evaluating policies attached to that principal, to the resource, and to the organisation around both. Most serious cloud breaches are not exploits of the provider; they are a valid credential used with permissions it should never have had. This chapter covers the resource hierarchy, how each cloud models principals and policies, AWS policy evaluation in detail, temporary credentials and workload identity federation, the confused deputy problem, privilege escalation paths, and how least privilege is approached and verified at scale.

*See also:* #xref("cloud-and-infrastructure", "multi-tenancy", label: "Multi-Tenancy") (Kubernetes RBAC), #xref("cloud-and-infrastructure", "iaas-fundamentals", label: "IaaS Fundamentals"), #xref("cryptography-and-security", "key-exchange-and-pki", label: "Key Exchange and PKI"), #xref("database", "security-and-privacy", label: "Database Security and Privacy").

== The Resource Hierarchy

Each cloud organises resources in a tree, and the tree is where guardrails and inheritance live.

#table(
  columns: 4,
  [*Level*], [*AWS*], [*GCP*], [*Azure*],
  [Root], [Organization], [Organization], [Entra tenant / root management group],
  [Grouping], [Organizational units (OUs)], [Folders], [Management groups],
  [Billing and isolation unit], [Account], [Project], [Subscription],
  [Inner grouping], [Tags only], [Labels only], [Resource groups],
  [Guardrails], [SCPs and RCPs on OUs and accounts], [Organization policies, deny policies], [Azure Policy, deny assignments],
)

The AWS account is a hard boundary: by default nothing in one account can touch another, and access across accounts must be granted explicitly on both sides. This makes "one account per workload per environment" the standard landing-zone design. It caps the blast radius of a stolen credential, separates quotas and bills, and turns organisational boundaries into enforcement boundaries. GCP projects and Azure subscriptions play the same role, but permissions granted at a folder or management group are *inherited* downwards, so a grant high in the tree reaches everything beneath it.

== Principals

A principal is anything that can make an authenticated request.

- *Human users.* In mature setups, humans have no long-lived cloud credentials at all. They sign in to a corporate identity provider and receive short-lived sessions through SAML or OIDC federation (AWS IAM Identity Center, Google Workspace or Cloud Identity, Entra ID).
- *Roles.* An AWS role has no credentials of its own. It has a *trust policy* (who may assume it) and permission policies (what a session may do). Assuming it returns temporary credentials.
- *Service accounts.* In GCP, a service account is an identity that workloads run as. In Azure, managed identities serve the same purpose and are tied to a resource's lifecycle.
- *Workloads from elsewhere.* CI pipelines, Kubernetes pods, and other clouds authenticate with an OIDC token from their own issuer and exchange it for cloud credentials (workload identity federation, covered below).

The root user of an AWS account (and the equivalents elsewhere) bypasses most policy. It should have hardware MFA, no access keys, and be used only for the few tasks that require it.

== Policy Models

The three clouds express permissions differently, and the difference shows in how they are audited.

*AWS* attaches JSON documents. An identity-based policy on a role says what that role may do. A resource-based policy on, say, an S3 bucket or KMS key says who may act on it. Each statement has an `Effect` (`Allow` or `Deny`), `Action`, `Resource`, optional `Principal` (resource policies only), and optional `Condition`.

*GCP* uses allow policies bound to a resource: a list of bindings from a *role* (a named set of permissions such as `roles/storage.objectViewer`) to *members*. The effective policy at a resource is the union of its own bindings and those of every ancestor. Deny policies, evaluated before allow policies, and IAM Conditions (CEL expressions on request attributes) refine this.

*Azure* uses role assignments: a principal, a role definition, and a scope (management group, subscription, resource group, or resource). Assignments inherit to child scopes. Deny assignments exist but are created by the platform (for example by managed applications and deployment stacks), not written directly. Directory roles in Entra ID are a separate system that controls the identity plane itself.

Whatever the syntax, the authorisation question is the same: given a request $(p, a, r, c)$ of principal, action, resource, and context, does the combination of all applicable policies allow it?

== AWS Policy Evaluation

AWS evaluates a request against every policy type that applies and combines them as follows:

+ If any applicable policy has a matching explicit `Deny`, the request is denied.
+ Organisation guardrails act as ceilings: the request must be allowed by every SCP from the root down to the caller's account, and by any RCP on the resource's account. They grant nothing themselves.
+ If a permissions boundary is set on the principal, it must also allow the action (another ceiling).
+ If the session was created with a session policy, it must also allow the action (another ceiling).
+ Finally there must be a grant: an `Allow` in an identity-based policy or in the resource-based policy. Within one account, either is enough. Across accounts, *both* the caller's identity policy and the resource policy in the other account must allow it.
+ Otherwise, the request is implicitly denied.

The rule "deny wins, ceilings intersect, grants union" is compact enough to model directly. Action patterns use `*` and `?` wildcards; action names compare case-insensitively and resource ARNs case-sensitively.

```cpp
#include <cctype>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

bool glob_match(std::string_view pat, std::string_view s, bool fold_case) {
    auto eq = [fold_case](char a, char b) {
        return fold_case ? std::tolower(static_cast<unsigned char>(a)) ==
                               std::tolower(static_cast<unsigned char>(b))
                         : a == b;
    };
    std::size_t p = 0, i = 0, star = std::string_view::npos, mark = 0;
    while (i < s.size()) {
        if (p < pat.size() && (pat[p] == '?' || eq(pat[p], s[i]))) {
            ++p; ++i;
        } else if (p < pat.size() && pat[p] == '*') {
            star = p++; mark = i;
        } else if (star != std::string_view::npos) {
            p = star + 1; i = ++mark;
        } else {
            return false;
        }
    }
    while (p < pat.size() && pat[p] == '*') ++p;
    return p == pat.size();
}

enum class effect { allow, deny };

struct statement {
    effect eff;
    std::vector<std::string> actions;
    std::vector<std::string> resources;
};

struct policy {
    std::vector<statement> statements;
};

struct request {
    std::string action;
    std::string resource;
    bool cross_account;
};

std::optional<effect> evaluate(const policy& p, const request& r) {
    bool allowed = false;
    for (const auto& st : p.statements) {
        bool action_hit = false, resource_hit = false;
        for (const auto& a : st.actions) action_hit |= glob_match(a, r.action, true);
        for (const auto& res : st.resources) resource_hit |= glob_match(res, r.resource, false);
        if (!action_hit || !resource_hit) continue;
        if (st.eff == effect::deny) return effect::deny;
        allowed = true;
    }
    return allowed ? std::optional{effect::allow} : std::nullopt;
}

struct policy_set {
    std::vector<policy> scps;               // one per level from the root to the account
    std::vector<policy> rcps;
    std::vector<policy> identity;
    std::optional<policy> resource;
    std::optional<policy> boundary;
    std::optional<policy> session;
};

bool authorise(const policy_set& ps, const request& r) {
    auto any_deny = [&](const std::vector<policy>& v) {
        for (const auto& p : v)
            if (evaluate(p, r) == effect::deny) return true;
        return false;
    };
    auto denies = [&](const std::optional<policy>& p) {
        return p && evaluate(*p, r) == effect::deny;
    };
    if (any_deny(ps.scps) || any_deny(ps.rcps) || any_deny(ps.identity) ||
        denies(ps.resource) || denies(ps.boundary) || denies(ps.session))
        return false;

    auto every_allows = [&](const std::vector<policy>& v) {
        for (const auto& p : v)
            if (evaluate(p, r) != effect::allow) return false;
        return true;
    };
    auto ceiling_ok = [&](const std::optional<policy>& p) {
        return !p || evaluate(*p, r) == effect::allow;
    };
    if (!every_allows(ps.scps) || !every_allows(ps.rcps) ||
        !ceiling_ok(ps.boundary) || !ceiling_ok(ps.session))
        return false;

    bool identity_allows = false;
    for (const auto& p : ps.identity) identity_allows |= evaluate(p, r) == effect::allow;
    const bool resource_allows = ps.resource && evaluate(*ps.resource, r) == effect::allow;

    return r.cross_account ? identity_allows && resource_allows
                           : identity_allows || resource_allows;
}
```

The model leaves out conditions, `NotAction` and `NotResource`, principal matching in resource policies, and a few service-specific exceptions (KMS key policies, for example, must explicitly delegate to IAM before identity policies count, and role trust policies behave differently from other resource policies). It does capture the structure that most mistakes come from: forgetting that a permissions boundary or SCP silently removes an `Allow`, or that a bucket policy alone can grant access inside its own account.

== Temporary Credentials

Long-lived access keys are the most common source of cloud compromise: they leak into Git history, CI logs, container images, and laptops, and they stay valid until someone notices. Temporary credentials replace them.

A call to STS `AssumeRole` returns an access key, a secret key, and a session token valid for 15 minutes to 12 hours (one hour by default). The role's trust policy must allow the caller. Unless that trust policy names the caller directly within the same account, the caller's own policies must also allow `sts:AssumeRole` on the role, and across accounts they always must. Inside the cloud, instances, functions, and containers obtain such credentials automatically from the metadata service or an equivalent endpoint, and the SDKs refresh them before expiry.

Short lifetimes do not make a leaked credential harmless; they bound the window. Pairing them with conditions such as `aws:SourceVpce`, `aws:SourceIp`, or a resource-perimeter policy ("only principals from my organisation, from my networks") means a stolen session token is useless outside the environment it was issued to.

== Workload Identity Federation

Workloads outside the cloud used to need a stored access key. Federation removes it. The workload obtains a signed OIDC token from an issuer it already trusts and exchanges it for cloud credentials:

```text
GitHub Actions job                       AWS
  │  1. request OIDC token (aud=sts.amazonaws.com)
  │◄─ JWT signed by token.actions.githubusercontent.com
  │     sub = "repo:acme/payments:ref:refs/heads/main"
  │  2. AssumeRoleWithWebIdentity(role, JWT) ───────────►  STS
  │                                                        │ verify signature via issuer JWKS
  │                                                        │ check trust policy conditions on aud, sub
  │◄──────────────────────── temporary credentials ────────┘
```

The same pattern underlies Kubernetes workload identity: EKS IRSA and Pod Identity, GKE Workload Identity Federation, and AKS Workload Identity all let a pod's service account token stand in for a cloud identity.

The security of the whole scheme rests on the trust policy conditions. The issuer proves *who signed the token*; only the `sub` and `aud` conditions say *which workload* may assume the role. A trust policy that checks the issuer but leaves `sub` as `repo:*` lets any repository on GitHub, including an attacker's, assume the role. Always pin `sub` to the exact repository and branch or environment, and pin `aud`.

```json
{
  "Effect": "Allow",
  "Principal": {"Federated": "arn:aws:iam::111122223333:oidc-provider/token.actions.githubusercontent.com"},
  "Action": "sts:AssumeRoleWithWebIdentity",
  "Condition": {
    "StringEquals": {
      "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
      "token.actions.githubusercontent.com:sub": "repo:acme/payments:environment:production"
    }
  }
}
```

== The Confused Deputy

A *confused deputy* is a privileged program tricked into using its authority on behalf of a party that should not have it. In the cloud it appears when a third-party service (a monitoring vendor, say) assumes a role in each customer's account. Customer A tells the vendor "my role is `arn:...:222:role/Vendor`". If the vendor assumes whatever role ARN a customer supplies, customer B can supply A's role ARN and read A's data through the vendor.

The fix is to bind the delegation to the requesting party:

- *External ID.* The vendor generates a unique ID per customer and always sends it on `AssumeRole`. Customer A's trust policy requires A's external ID. B cannot cause the vendor to send it, because the vendor sends B's own ID for B's requests.
- *Source conditions for service principals.* When an AWS service (such as SNS or S3 event notifications) acts for a resource, policies should check `aws:SourceArn` or `aws:SourceAccount` so that another account's resource cannot direct that service at yours.

== Privilege Escalation Paths

Permissions compose, and many innocuous-looking ones are equivalent to administrator. A principal can escalate if it can:

#table(
  columns: 2,
  [*Permission*], [*Why it escalates*],
  [`iam:PassRole` with broad resource, plus create rights on a compute service], [Launch an instance, function, or job that runs as a more powerful role, then use it],
  [`iam:CreatePolicyVersion` or `iam:PutRolePolicy` on its own role], [Rewrite its own permissions],
  [`iam:UpdateAssumeRolePolicy`], [Add itself to the trust policy of a powerful role],
  [`lambda:UpdateFunctionCode` on a function with a powerful role], [Replace the code that runs with that role],
  [`iam.serviceAccounts.actAs` or `getAccessToken` in GCP], [Act as, or mint tokens for, a more privileged service account],
  [`Microsoft.Authorization/roleAssignments/write` in Azure], [Grant itself any role at that scope],
)

Treat these as administrator permissions in reviews. `iam:PassRole` deserves special care: restrict its `Resource` to the specific roles a deployer needs, and add `iam:PassedToService` conditions so a role can only be passed to the intended service.

== Approaching Least Privilege

Least privilege is easy to state and hard to reach, because nobody knows in advance the exact set of actions an application calls. Practical programs combine several techniques:

- *Start from evidence.* Generate policies from observed activity (CloudTrail-based policy generation, GCP role recommendations, Azure access reviews), then prune with last-accessed data.
- *Layer guardrails.* SCPs and organisation policies deny whole categories that no workload should use: disabling audit logs, leaving the organisation, regions the company does not operate in, public S3 access.
- *Use resource perimeters.* Resource and resource-control policies that allow access only from principals in the organisation (`aws:PrincipalOrgID`) and from expected networks stop data exfiltration through a stolen credential to an outside account.
- *Attribute-based access control.* Grant by matching tags ("a principal tagged `team=payments` may act on resources tagged `team=payments`") rather than enumerating ARNs. Policies stay small as resources multiply, but tag-write permissions then become security-critical.
- *Break-glass access.* Keep a small number of emergency roles, protected with strong MFA, alerting on every use, so that day-to-day roles need not be broad "just in case".

== Verifying Policies

Policies are programs, and their semantics can be reasoned about mechanically. AWS's Zelkova translates policies into SMT formulas and answers questions such as "does this bucket policy allow access to anyone outside account X?" for all possible requests, not just test cases. IAM Access Analyzer and S3 "Block Public Access" checks are built on it. Tiros applies similar reasoning to network reachability.

Newer authorisation languages are designed for analysis from the start. Cedar (used by Amazon Verified Permissions) has a formally specified semantics, a validator that type-checks policies against a schema, and a decidable fragment that can be encoded to SMT for questions such as "is policy set $A$ more permissive than $B$?". That turns a policy change review from reading JSON into checking a proof obligation.

== Auditing

Every control-plane call is logged: AWS CloudTrail, GCP Cloud Audit Logs, Azure Activity Log and Entra sign-in logs. Data-plane calls (object reads, KMS decrypts) are logged only if data events are enabled, and they are billed per event. An organisation-wide trail written to a separate, locked logging account, with alerts on root use, policy changes, trust-policy changes, and guardrail changes, is the minimum. Without it, an investigation cannot tell what a leaked credential did.

== Pitfalls

- Long-lived access keys for CI or applications where federation or instance roles would work.
- Federation trust policies that check the issuer but wildcard the `sub` claim.
- `iam:PassRole` on `*`, which makes any compute-create permission an administrator grant.
- Third-party roles without external IDs.
- Grants at a GCP folder or Azure management group that reach far more projects than intended.
- Assuming a deny in an SCP is the only way to block access, and forgetting that bucket or key policies can grant access within an account on their own.
- Audit logging enabled per account rather than organisation-wide, or stored where the attacker's credential can delete it.

== Further Reading

Backes, J. et al. (2018). "Semantic-based Automated Reasoning for AWS Access Policies using SMT." FMCAD.

Cutler, J. W. et al. (2024). "Cedar: A New Language for Expressive, Fast, Safe, and Analyzable Authorization." OOPSLA.

Hardy, N. (1988). "The Confused Deputy (or why capabilities might have been invented)." ACM SIGOPS Operating Systems Review.

Saltzer, J. H. and Schroeder, M. D. (1975). "The Protection of Information in Computer Systems." Proceedings of the IEEE.

Backes, J. et al. (2019). "Reachability Analysis for AWS-Based Networks." CAV.

Sakimura, N. et al. (2014). "OpenID Connect Core 1.0." OpenID Foundation specification.

Jones, M., Bradley, J. and Sakimura, N. (2015). "JSON Web Token (JWT)." RFC 7519.

Hu, V. C. et al. (2014). "Guide to Attribute Based Access Control (ABAC) Definition and Considerations." NIST SP 800-162.

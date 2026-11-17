#import "../template.typ": xref

= Cloud Networking <cloud-networking>

A virtual network in the cloud looks like a classic enterprise network: address ranges, subnets, route tables, firewalls, gateways. Underneath, nothing is a box with cables. Every route, rule, and gateway is state that the control plane pushes into host virtual switches and into fleets of shared middleboxes. This changes the cost model, the failure modes, and which designs scale. This chapter covers address planning, routing, stateful and stateless filtering, connecting many networks together, private access to services, hybrid links to on-premises sites, and the traffic paths that drive both latency and the bill.

*See also:* #xref("cloud-and-infrastructure", "iaas-fundamentals", label: "IaaS Fundamentals") (VPC overlay basics), #xref("networking", "data-center-networking", label: "Data Center Networking"), #xref("networking", "load-balancing", label: "Load Balancing"), #xref("networking", "bgp-routing", label: "BGP Routing"), #xref("networking", "nat", label: "Network Address Translation (NAT)"), #xref("networking", "sdn-programmable-networks", label: "SDN and Programmable Networks").

== Address Planning

A VPC owns one or more private CIDR blocks (RFC 1918 space: `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, plus `100.64.0.0/10` shared space on some clouds). Subnets carve that range into pieces. Each cloud keeps a few addresses in every subnet for itself:

#table(
  columns: 4,
  [*Property*], [*AWS*], [*GCP*], [*Azure*],
  [VPC scope], [Region], [Global], [Region],
  [Subnet scope], [One AZ], [Region (all zones)], [Region (all zones)],
  [Reserved per subnet], [5 (first 4, last 1)], [4 (first 2, last 2)], [5 (first 4, last 1)],
  [Smallest subnet], [/28], [/29], [/29],
  [Add ranges later], [Secondary CIDRs], [Expand primary, add secondary], [Add address spaces],
)

The address plan is the hardest decision to undo. Three rules hold on every cloud:

+ *Never overlap.* Peering, transit routing, and VPN links all refuse or misroute overlapping prefixes. Give every VPC in the organisation, and every on-premises site, a distinct block from a central registry (an IPAM service or a spreadsheet that is treated as code).
+ *Size for the pods, not the VMs.* Kubernetes CNIs that give each pod a VPC address (AWS VPC CNI, GKE VPC-native, Azure CNI) consume addresses 10 to 50 times faster than the node count suggests. A /24 subnet holds 251 usable AWS addresses, which is a handful of nodes at 30 pods each.
+ *Leave room to grow.* A /16 per VPC with /20 subnets per tier and zone is a common default. It wastes addresses, but private space is free and renumbering a live network is not.

Overlap detection is an interval problem on 32-bit integers. Two prefixes overlap exactly when one contains the other, because CIDR blocks are aligned:

```cpp
#include <cstdint>
#include <optional>
#include <string_view>
#include <charconv>

struct cidr {
    std::uint32_t base;
    int len;
};

std::optional<cidr> parse_cidr(std::string_view s) {
    std::uint32_t addr = 0;
    const char* p = s.data();
    const char* end = s.data() + s.size();
    for (int octet = 0; octet < 4; ++octet) {
        unsigned v = 0;
        auto [next, ec] = std::from_chars(p, end, v);
        if (ec != std::errc{} || v > 255) return std::nullopt;
        addr = (addr << 8) | v;
        p = next;
        if (octet < 3) {
            if (p == end || *p != '.') return std::nullopt;
            ++p;
        }
    }
    if (p == end || *p != '/') return std::nullopt;
    int len = 0;
    auto [next, ec] = std::from_chars(p + 1, end, len);
    if (ec != std::errc{} || next != end || len < 0 || len > 32) return std::nullopt;
    return cidr{addr, len};
}

constexpr std::uint32_t mask_of(int len) {
    return len == 0 ? 0u : ~0u << (32 - len);
}

bool contains(cidr outer, std::uint32_t addr) {
    return (addr & mask_of(outer.len)) == (outer.base & mask_of(outer.len));
}

bool overlaps(cidr a, cidr b) {
    const int len = a.len < b.len ? a.len : b.len;
    return (a.base & mask_of(len)) == (b.base & mask_of(len));
}
```

Comparing under the shorter mask is enough: if the shorter prefix covers the longer one's base, the longer block lies wholly inside it.

== Route Tables

Each subnet is associated with a route table. A route maps a destination prefix to a target: the local VPC, an internet gateway, a NAT gateway, a peering connection, a transit gateway, a VPN, a firewall endpoint, or a network interface. Lookup is *longest prefix match*: the most specific route wins, and the implicit `local` route for the VPC CIDR cannot be overridden by a less specific one.

A binary trie gives $O(W)$ lookup for $W = 32$ bits. Real host switches compile the table into flow caches, but the semantics are the same:

```cpp
#include <array>
#include <memory>
#include <string>

struct route_target {
    std::string id;   // "local", "igw-1", "nat-3", "tgw-7", ...
};

class route_table {
    struct node {
        std::array<std::unique_ptr<node>, 2> child;
        std::optional<route_target> target;
    };
    node root_;

public:
    void add(cidr prefix, route_target t) {
        node* n = &root_;
        for (int i = 0; i < prefix.len; ++i) {
            const int bit = (prefix.base >> (31 - i)) & 1;
            if (!n->child[bit]) n->child[bit] = std::make_unique<node>();
            n = n->child[bit].get();
        }
        n->target = std::move(t);
    }

    std::optional<route_target> lookup(std::uint32_t addr) const {
        const node* n = &root_;
        std::optional<route_target> best = n->target;
        for (int i = 0; i < 32 && n; ++i) {
            n = n->child[(addr >> (31 - i)) & 1].get();
            if (n && n->target) best = n->target;
        }
        return best;
    }
};
```

A typical three-tier layout uses one table per tier:

#table(
  columns: 3,
  [*Subnet tier*], [*Default route (`0.0.0.0/0`)*], [*Reachable from the internet*],
  [Public (load balancers, NAT gateways)], [Internet gateway], [Yes, if the resource has a public IP],
  [Private (application servers)], [NAT gateway in the same zone], [No; outbound only],
  [Isolated (databases)], [None], [No; no internet path at all],
)

A subnet is "public" only because of its route table. An instance with a public IP in a subnet whose default route points at a NAT gateway cannot receive inbound internet traffic, which surprises people who think of public IPs as a property of the machine.

== Gateways and Address Translation

An *internet gateway* is not a device in the path. It is a logical 1:1 NAT between an instance's private address and its public address, applied by the host and the edge. It has no bandwidth limit of its own and no hourly charge.

A *NAT gateway* is a managed many-to-one source NAT for private subnets. It is a real fleet (AWS builds NAT gateways, Network Load Balancers, and PrivateLink on the same Hyperplane system), so it has per-zone scope, port limits, and a price. Each NAT gateway public IP supports about 55,000 simultaneous connections to one destination IP and port, because the source port is the only free field in the 5-tuple. Workloads that fan out to one API endpoint exhaust ports long before bandwidth.

Two NAT gateway cost traps are common:

- *Processing charges.* AWS bills per GB processed through a NAT gateway, on top of the hourly fee and any egress. Pulling container images or S3 data through NAT can cost more than the compute. Gateway endpoints for S3 and DynamoDB route that traffic around NAT for free.
- *Cross-zone hairpins.* Private subnets in zone B that use the NAT gateway in zone A pay cross-zone transfer on every byte and lose that path when zone A fails. Use one NAT gateway per zone and one route table per zone.

GCP Cloud NAT is different in kind: it has no gateway instances. The Andromeda control plane allocates NAT IPs and port ranges to each VM, and each host performs the translation. There is no choke point, but port allocation per VM becomes the tuning knob.

== Filtering: Stateful and Stateless

Clouds provide two filter layers with different semantics.

#table(
  columns: 3,
  [*Property*], [*Security group (AWS) / NSG (Azure) / firewall rule (GCP)*], [*Network ACL (AWS)*],
  [Attached to], [Network interface (or VM tag / service account in GCP)], [Subnet],
  [State], [Stateful: return traffic is allowed], [Stateless: both directions need rules],
  [Rule types], [Allow only (AWS); allow and deny with priority (Azure, GCP)], [Allow and deny, ordered by number],
  [Evaluation], [All rules, any match allows (AWS)], [First match wins],
  [Can reference], [Other security groups, prefix lists], [CIDR blocks only],
)

*Stateful* means the host keeps a connection tracking entry. When an outbound connection is allowed, the reverse 5-tuple is recorded and inbound packets that match it pass without consulting the inbound rules. A stateless filter has no such memory, so a NACL that allows outbound HTTPS must also allow inbound TCP from port 443 to the client's ephemeral range (1024 to 65535 to cover every OS).

```cpp
#include <cstdint>
#include <unordered_set>
#include <vector>

struct five_tuple {
    std::uint32_t src, dst;
    std::uint16_t sport, dport;
    std::uint8_t proto;
    bool operator==(const five_tuple&) const = default;
};

struct tuple_hash {
    std::size_t operator()(const five_tuple& t) const {
        std::uint64_t h = (std::uint64_t{t.src} << 32) ^ t.dst;
        h ^= (std::uint64_t{t.sport} << 24) ^ (std::uint64_t{t.dport} << 8) ^ t.proto;
        h *= 0x9e3779b97f4a7c15ull;
        return static_cast<std::size_t>(h ^ (h >> 29));
    }
};

five_tuple reversed(const five_tuple& t) {
    return {t.dst, t.src, t.dport, t.sport, t.proto};
}

struct sg_rule {
    cidr peer;
    std::uint16_t port_lo, port_hi;
    std::uint8_t proto;
};

class security_group {
    std::vector<sg_rule> inbound_, outbound_;
    std::unordered_set<five_tuple, tuple_hash> conntrack_;

    static bool any_allows(const std::vector<sg_rule>& rules,
                           std::uint32_t peer, std::uint16_t port, std::uint8_t proto) {
        for (const auto& r : rules)
            if (r.proto == proto && contains(r.peer, peer) &&
                port >= r.port_lo && port <= r.port_hi)
                return true;
        return false;
    }

public:
    void allow_in(sg_rule r) { inbound_.push_back(r); }
    void allow_out(sg_rule r) { outbound_.push_back(r); }

    bool egress(const five_tuple& t) {
        if (conntrack_.contains(t)) return true;
        if (!any_allows(outbound_, t.dst, t.dport, t.proto)) return false;
        conntrack_.insert(t);
        return true;
    }

    bool ingress(const five_tuple& t) {
        if (conntrack_.contains(reversed(t))) return true;
        if (!any_allows(inbound_, t.src, t.dport, t.proto)) return false;
        conntrack_.insert(reversed(t));
        return true;
    }
};
```

The model leaves out entry expiry, but expiry is where stateful filters fail in production. Idle entries time out (AWS lets the established-TCP timeout be tuned per interface, from 60 seconds up to its 5-day default, and NAT gateways and Network Load Balancers drop idle flows after 350 seconds), and a long-lived idle connection whose entry has gone sees its next packet dropped. Use TCP keepalives shorter than the timeout. Connection tracking tables also have a per-instance size limit; a host that accepts millions of short connections can exhaust it. AWS skips tracking for flows that are allowed in both directions by rules of `0.0.0.0/0` on all ports, which is one reason high-rate front ends use wide-open groups behind a stateless filter.

Prefer security-group-to-security-group references ("allow 5432 from the `app` group") over CIDR rules. The rule then follows instances as they scale, and the intent is visible in the rule.

== Connecting Many Networks

=== Peering

VPC peering joins two VPCs with a direct route on each side. It has no bandwidth limit and no single point of failure, because it is just more entries in the host flow tables. It is also *non-transitive*: if A peers with B and B peers with C, A cannot reach C through B. Full connectivity among $n$ VPCs needs $n(n-1)/2$ peerings and route entries in every table, which stops scaling at a few dozen networks.

=== Hub and Spoke

A *transit gateway* (AWS Transit Gateway, Azure Virtual WAN hub, GCP Network Connectivity Center) is a regional router that spokes attach to. It has its own route tables, so it can express segmentation: production spokes share one table, development spokes another, and both reach a shared-services table, but production and development never reach each other.

```text
               ┌───────────────────────────┐
  prod-a ──────┤                           ├────── shared-services
  prod-b ──────┤     Transit Gateway       ├────── inspection VPC (firewalls)
  dev-a  ──────┤  rt-prod | rt-dev | rt-shr├────── Direct Connect / VPN
               └───────────────────────────┘
```

The hub costs a per-attachment hourly fee and a per-GB processing fee, and it adds a hop. Large organisations run both: a transit gateway for general reachability and direct peering for the few high-volume pairs where the per-GB charge matters.

=== Centralised Inspection

Regulated environments often force all north-south and east-west traffic through firewalls. The pattern is an inspection VPC attached to the hub, with hub routes that send traffic to it. The hard part is *symmetry*: a stateful firewall must see both directions of a flow. AWS Gateway Load Balancer solves this with GENEVE encapsulation and flow-hash stickiness, so both directions of a 5-tuple reach the same appliance. Without it, asymmetric routing across zones silently drops return traffic.

== Private Access to Services

Managed services (object storage, queues, databases, third-party SaaS) live outside the customer's VPC. Three mechanisms keep traffic to them off the internet:

#table(
  columns: 3,
  [*Mechanism*], [*How it works*], [*Trade-offs*],
  [Gateway endpoint (AWS, S3 and DynamoDB only)], [Route-table entry pointing a prefix list at the service], [Free; same region only; cannot be reached from on-premises],
  [Interface endpoint (AWS PrivateLink, GCP Private Service Connect, Azure Private Link)], [A network interface with a private IP in the consumer subnet, mapped to the provider's load balancer], [Hourly and per-GB fees; one-way (consumer starts connections); works across overlapping CIDRs],
  [Private Google Access / service endpoints], [Private-IP VMs reach provider APIs over provider routes], [Coarse; traffic stays on the backbone but targets public service IPs],
)

Interface endpoints matter beyond privacy. Because the consumer only sees an address in its own subnet, the provider can sit in a VPC with an overlapping CIDR. This is the standard way for a SaaS vendor to offer private connectivity to thousands of customers whose address plans it does not control.

Endpoints interact with DNS. An interface endpoint for a service usually overrides the public name (for example `sqs.us-east-1.amazonaws.com`) inside the VPC with a private hosted zone, so clients pick up the private path without code changes. When the override is missing, traffic quietly falls back to NAT and the internet, and both the bill and the data-exfiltration analysis change.

== DNS Inside the VPC

Every VPC has a resolver at a fixed address (on AWS, the VPC base plus two, and `169.254.169.253`). It answers for instance names, private hosted zones attached to the VPC, and endpoint overrides, and it forwards everything else. Split-horizon DNS (the same name resolving to a private address inside and a public one outside) is built from a private zone that shadows the public one.

Hybrid DNS needs forwarding in both directions: inbound resolver endpoints let on-premises servers resolve cloud private zones, and outbound endpoints with forwarding rules let VMs resolve on-premises names. DNS is a frequent source of hybrid outages because it is configured in a third place, separate from routes and firewalls, and it fails as a timeout rather than an error.

== Hybrid Connectivity

#table(
  columns: 4,
  [*Option*], [*Medium*], [*Typical bandwidth*], [*Notes*],
  [Site-to-site VPN], [IPsec over the internet], [About 1.25 Gbps per tunnel (AWS)], [Minutes to set up; latency and loss follow the internet; use two tunnels with BGP],
  [Dedicated circuit (Direct Connect, Dedicated Interconnect, ExpressRoute)], [Physical cross-connect at a colocation facility], [1, 10, 100 Gbps (400 Gbps on some)], [Weeks to provision; private, predictable; much cheaper egress per GB],
  [Partner circuit], [Shared port via a carrier], [50 Mbps to 10 Gbps], [Faster to get; capacity shared at the partner],
)

All three exchange routes with BGP. The cloud side advertises VPC prefixes, the customer router advertises on-premises prefixes, and AS path prepending or MED steers which link is primary. A circuit is one physical path; production designs use two circuits at two facilities plus a VPN as a last resort, and test the failover, because a BGP session that has never flapped in anger often carries a misconfigured filter.

IPsec has overhead that matters at scale: ESP in tunnel mode adds about 50 to 80 bytes per packet, so the inner MTU drops (AWS VPN advertises 1446 bytes for the tunnel interface) and TCP MSS clamping is needed to avoid fragmentation.

== MTU and Path MTU

Inside a VPC, most clouds support jumbo frames (AWS: 9001 bytes; GCP: configurable up to 8896). Traffic that leaves through an internet gateway, a VPN, or some peering paths is limited to 1500 or less. Path MTU discovery relies on ICMP "fragmentation needed" (IPv4) and "packet too big" (IPv6) messages, so a security group or NACL that drops all ICMP creates the classic *black hole*: TCP handshakes succeed, small requests work, and large responses hang. Allow the ICMP types needed for PMTUD, or clamp MSS at the edge.

== Load Balancers in the Cloud

Cloud load balancers are distributed systems, not appliances. Two families exist:

- *Layer 4* (AWS NLB, GCP passthrough Network LB, Azure Load Balancer): per-flow hashing in the host or a Hyperplane or Maglev fleet, client IP preserved, very high connection rates, no TLS termination by default. GCP's passthrough balancer uses direct server return: replies go straight from the backend to the client.
- *Layer 7* (AWS ALB, GCP Application LB, Azure Application Gateway / Front Door): a managed proxy fleet that terminates TLS and HTTP, routes on host and path, and opens new connections to backends. Client IP arrives in `X-Forwarded-For`.

Global anycast front ends (GCP global Application LB, AWS Global Accelerator, Azure Front Door) announce one IP from many edge locations. The client's TCP or QUIC handshake ends at the nearest edge, and the long haul to the backend region runs on the provider backbone over warm connections. This cuts tail latency for distant users more than any backend tuning.

Two operational points recur. First, cloud L7 balancers scale out behind DNS, and their IPs change; clients and firewalls that pin IPs break. Second, cross-zone load balancing (spreading each zone's front-end traffic across all zones' backends) improves balance but incurs cross-zone transfer charges on some clouds.

== IPv6

IPv6 removes NAT and address exhaustion from the design: AWS assigns a /56 to the VPC and a /64 to each subnet, all globally routable. Inbound reachability is then controlled only by filters, and an *egress-only internet gateway* gives private subnets outbound IPv6 with stateful return, the IPv6 analogue of NAT without translation. Dual-stack is the realistic state for years: many managed services, peering features, and on-premises networks still assume IPv4. Since 2024 AWS charges for every public IPv4 address, which has turned IPv6 adoption into a cost question as well as a capacity one.

== Instance Metadata Service

Every VM reaches a link-local metadata service at `169.254.169.254` for its identity, user data, and temporary credentials. This makes the metadata endpoint the prize in server-side request forgery attacks: an application tricked into fetching an attacker-chosen URL can return the instance's role credentials. The 2019 Capital One breach followed this path. IMDSv2 on AWS requires a session token obtained with a `PUT` request and sets the response IP TTL (hop limit) to 1 by default, which defeats most SSRF primitives and blocks access from containers behind an extra network hop. GCP requires a `Metadata-Flavor: Google` header for the same reason. Enforce IMDSv2 for every instance.

== Traffic Paths and Their Cost

Network design in the cloud is cost design. Approximate list prices in a major region:

#table(
  columns: 2,
  [*Path*], [*Typical charge*],
  [Within one AZ, private IP], [Free],
  [Across AZs in one region], [About \$0.01 per GB in each direction (AWS)],
  [Across regions], [\$0.02 per GB and up],
  [To the internet], [\$0.05 to \$0.09 per GB, falling with volume],
  [Through a NAT gateway], [About \$0.045 per GB processed, plus egress],
  [Through a transit gateway or interface endpoint], [About \$0.01 to \$0.02 per GB processed],
  [Over a dedicated circuit], [About \$0.02 per GB out, plus port-hours],
)

A chatty microservice deployment spread evenly across three zones sends about two thirds of its east-west calls across zone boundaries. Topology-aware routing (Kubernetes `trafficDistribution: PreferClose`, zone-aware service mesh routing) keeps calls in-zone when capacity allows and often removes the largest line item on the network bill. #xref("cloud-and-infrastructure", "cost-engineering", label: "Cloud Cost Engineering") covers the accounting side.

== Pitfalls

- Overlapping CIDRs between VPCs or with on-premises, discovered when the first peering request fails.
- A single NAT gateway shared by all zones: one zone failure takes out egress everywhere, and every other zone pays cross-zone charges.
- NACLs added "for defence in depth" without ephemeral port rules, breaking return traffic in one direction only.
- Long-lived idle connections cut by middlebox idle timeouts (350 s for AWS NAT gateways and NLB, 60 s ALB default).
- ICMP fully blocked, causing PMTUD black holes on VPN and internet paths.
- Interface endpoints without the DNS override, so traffic silently takes the NAT path.
- IMDSv1 left enabled on instances that run code reachable from the internet.

== Further Reading

Dalton, M. et al. (2018). "Andromeda: Performance, Isolation, and Velocity at Scale in Cloud Network Virtualization." NSDI.

Eisenbud, D. E. et al. (2016). "Maglev: A Fast and Reliable Software Network Load Balancer." NSDI.

Firestone, D. et al. (2018). "Azure Accelerated Networking: SmartNICs in the Public Cloud." NSDI.

Firestone, D. (2017). "VFP: A Virtual Switch Platform for Host SDN in the Public Cloud." NSDI.

Koponen, T. et al. (2014). "Network Virtualization in Multi-tenant Datacenters." NSDI.

Patel, P. et al. (2013). "Ananta: Cloud Scale Load Balancing." SIGCOMM.

Hinden, R. and Deering, S. (2006). "IP Version 6 Addressing Architecture." RFC 4291.

Rekhter, Y. et al. (1996). "Address Allocation for Private Internets." RFC 1918.

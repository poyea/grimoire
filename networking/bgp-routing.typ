= BGP Routing

Border Gateway Protocol (BGP) is the de facto standard for inter-domain routing, forming the backbone of internet connectivity between autonomous systems.

*See also:* Internet Layer (for IP addressing), Load Balancing (for traffic engineering), Network Security (for route filtering)

== BGP Fundamentals

*BGP provides [RFC 4271]:*
- Path vector routing (full AS path visibility)
- Policy-based routing (operator-controlled decisions)
- Incremental updates (only changes propagated)
- TCP-based transport (port 179, reliable delivery)

*Two BGP flavors:*
- *eBGP (external BGP):* Between different autonomous systems
- *iBGP (internal BGP):* Within same autonomous system

*Key distinction:*

```
                  AS 64500                      AS 64501
       ┌──────────────────────────┐   ┌──────────────────────────┐
       │                          │   │                          │
       │   R1 ──iBGP── R2       │   │   R3 ──iBGP── R4        │
       │        \      /         │   │        \      /          │
       │         \    /          │   │         \    /           │
       │          R5 ───────────────eBGP─────── R6             │
       │                          │   │                          │
       └──────────────────────────┘   └──────────────────────────┘
```

*eBGP vs iBGP differences:*
- eBGP: TTL=1 by default (directly connected), AS_PATH modified
- iBGP: TTL=255, requires full mesh or route reflectors, AS_PATH unchanged

== Autonomous System (AS) Architecture

*Autonomous System:* A collection of IP networks under single administrative control with unified routing policy.

*AS number allocation [RFC 6996]:*
- 16-bit ASN: 1-65535 (original, still common)
- 32-bit ASN: 1-4294967295 (extended, RFC 6793)
- Private ASN: 64512-65534 (16-bit), 4200000000-4294967294 (32-bit)

*AS types:*
1. *Stub AS:* Single connection to internet (most enterprises)
2. *Multihomed AS:* Multiple providers, no transit
3. *Transit AS:* Carries traffic between other ASes (ISPs)

*Global routing table size (2024):*
- Full IPv4 table: ~950,000 prefixes
- Full IPv6 table: ~200,000 prefixes
- Memory requirement: ~500MB-2GB depending on implementation

== BGP Message Types

*Four message types [RFC 4271]:*

```
┌─────────────────────────────────────────────────────────────┐
│  BGP Message Header (19 bytes minimum)                       │
├─────────────────────────────────────────────────────────────┤
│  Marker (16 bytes, all 1s for unauthenticated)              │
│  Length (2 bytes, total message length)                      │
│  Type (1 byte): OPEN=1, UPDATE=2, NOTIFICATION=3, KEEPALIVE=4│
└─────────────────────────────────────────────────────────────┘
```

*1. OPEN (connection establishment):*
- BGP version (4)
- Local AS number
- Hold time (default 90s, 0 = no keepalives)
- BGP identifier (router ID, typically highest IP)
- Optional parameters (capabilities negotiation)

*2. UPDATE (route advertisement/withdrawal):*
- Withdrawn routes (prefixes to remove)
- Path attributes (AS_PATH, NEXT_HOP, etc.)
- NLRI (Network Layer Reachability Information - prefixes)

*3. NOTIFICATION (error handling):*
- Error code and subcode
- Closes BGP session
- Common errors: HOLD_TIMER_EXPIRED, UPDATE_MESSAGE_ERROR

*4. KEEPALIVE (liveness check):*
- Sent every hold_time/3 (default 30s)
- 19-byte header only
- Missing 3 consecutive = session down

== BGP Path Attributes

*Path attributes determine route selection [RFC 4271]:*

*Well-known mandatory:*
- *ORIGIN:* Route source (IGP=0, EGP=1, INCOMPLETE=2)
- *AS_PATH:* Sequence of ASes traversed
- *NEXT_HOP:* IP address of next-hop router

*Well-known discretionary:*
- *LOCAL_PREF:* iBGP preference (higher = preferred, default 100)
- *ATOMIC_AGGREGATE:* Route has been aggregated

*Optional transitive:*
- *AGGREGATOR:* AS and router that aggregated
- *COMMUNITY:* 32-bit tag for policy (RFC 1997)

*Optional non-transitive:*
- *MED (Multi-Exit Discriminator):* Hint to external AS (lower = preferred)
- *ORIGINATOR_ID:* Route reflector loop prevention
- *CLUSTER_LIST:* Route reflector cluster path

== BGP Route Selection Algorithm

*Decision process (in order of priority):*

```cpp
// Pseudocode for BGP best path selection
Route select_best_path(vector<Route> candidates) {
    // 1. Highest weight (Cisco proprietary, local to router)
    candidates = filter_by_max(candidates, &Route::weight);
    if (candidates.size() == 1) return candidates[0];

    // 2. Highest LOCAL_PREF
    candidates = filter_by_max(candidates, &Route::local_pref);
    if (candidates.size() == 1) return candidates[0];

    // 3. Locally originated route preferred
    candidates = prefer_local_origin(candidates);
    if (candidates.size() == 1) return candidates[0];

    // 4. Shortest AS_PATH
    candidates = filter_by_min(candidates, &Route::as_path_length);
    if (candidates.size() == 1) return candidates[0];

    // 5. Lowest ORIGIN (IGP < EGP < INCOMPLETE)
    candidates = filter_by_min(candidates, &Route::origin);
    if (candidates.size() == 1) return candidates[0];

    // 6. Lowest MED (within same AS)
    candidates = filter_by_min_med(candidates);
    if (candidates.size() == 1) return candidates[0];

    // 7. eBGP over iBGP
    candidates = prefer_ebgp(candidates);
    if (candidates.size() == 1) return candidates[0];

    // 8. Lowest IGP metric to NEXT_HOP
    candidates = filter_by_min(candidates, &Route::igp_metric);
    if (candidates.size() == 1) return candidates[0];

    // 9. Oldest route (stability)
    candidates = filter_by_max(candidates, &Route::age);
    if (candidates.size() == 1) return candidates[0];

    // 10. Lowest router ID
    candidates = filter_by_min(candidates, &Route::router_id);
    return candidates[0];
}
```

*Practical tiebreakers:*
- Steps 1-4 handle 99% of decisions
- LOCAL_PREF is primary iBGP control
- AS_PATH is primary eBGP control

== BGP Communities

*Communities enable policy tagging [RFC 1997, RFC 8092]:*

*Well-known communities:*
- `NO_EXPORT` (0xFFFFFF01): Don't advertise to eBGP peers
- `NO_ADVERTISE` (0xFFFFFF02): Don't advertise to any peer
- `NO_EXPORT_SUBCONFED` (0xFFFFFF03): Don't advertise outside confederation

*Standard community format:* ASN:VALUE (e.g., 64500:100)

*Large communities [RFC 8092]:* 4-byte ASN : 4-byte value : 4-byte value

*Common community uses:*

```
# Traffic engineering
64500:1000   → Set LOCAL_PREF to 100 (low preference)
64500:2000   → Set LOCAL_PREF to 200 (high preference)
64500:3000   → Prepend AS once to all peers
64500:4000   → Blackhole this prefix

# Customer tagging
64500:100    → Customer routes
64500:200    → Peer routes
64500:300    → Transit routes
```

*Route filtering with communities:*

```
# Cisco IOS example
route-map CUSTOMER-IN permit 10
 match community CUSTOMER-TAG
 set local-preference 150

ip community-list standard CUSTOMER-TAG permit 64500:100
```

== iBGP Scaling: Route Reflectors

*Problem:* iBGP requires full mesh ($n times (n-1)/2$ sessions). 100 routers = 4,950 sessions.

*Solution:* Route reflectors [RFC 4456].

```
                Without RR              With RR
              (full mesh)           (hub-and-spoke)

           R1──R2──R3              R1   R2   R3
            │\  │  /│                \   │   /
            │ \ │ / │                 \  │  /
            │  \│/  │                  \ │ /
           R4──R5──R6                   RR  (Route Reflector)
                                       / │ \
                                      /  │  \
                                    R4   R5   R6

         15 sessions               6 sessions
```

*RR rules:*
1. Routes from clients → reflect to all clients and non-clients
2. Routes from non-clients → reflect to clients only
3. Add ORIGINATOR_ID and CLUSTER_LIST for loop prevention

*Redundancy:* Deploy multiple RRs per cluster. Clients peer with 2+ RRs.

== eBGP Peering and Multihoming

*Peering relationships:*

1. *Customer-Provider:* Customer pays provider for transit
   - Provider advertises customer routes to everyone
   - Provider advertises all routes to customer

2. *Peer-to-Peer:* Free exchange of customer routes
   - Only exchange customer routes (no transit)
   - Usually settlement-free

3. *Paid Peering:* Peer-to-peer with payment (asymmetric traffic)

*Multihoming configurations:*

```
                    ┌─────────────────┐
                    │   Your Network   │
                    │   AS 64500       │
                    └────────┬─────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
       ┌──────┴──────┐             ┌──────┴──────┐
       │  Provider A  │             │  Provider B  │
       │   AS 1234    │             │   AS 5678    │
       └─────────────┘             └──────────────┘
```

*Traffic engineering for multihoming:*

*Inbound (how others reach you):*
- AS_PATH prepending: Add your ASN multiple times
- Selective advertising: Advertise subsets to each provider
- Communities: Signal provider to deprioritize

*Outbound (how you reach others):*
- LOCAL_PREF: Higher = preferred exit
- Weight: Cisco-specific, highest wins

== BGP Convergence and Performance

*Convergence time factors:*
- MRAI (Minimum Route Advertisement Interval): 30s eBGP, 0s iBGP
- Hold timer: 90s default (3 missed keepalives)
- Route processing: 1-100ms per route

*Convergence timeline (typical failure):*

```
t=0:     Link failure detected
t=0-3s:  IGP reconverges (OSPF/IS-IS)
t=3s:    BGP session times out or BFD triggers
t=3-30s: BGP updates propagate (MRAI delays)
t=30s+:  Downstream ASes receive updates
```

*Fast convergence techniques:*

1. *BFD (Bidirectional Forwarding Detection):*
   - Sub-second failure detection (50-300ms typical)
   - Hardware-accelerated on modern routers

```
router bgp 64500
 neighbor 10.0.0.1 fall-over bfd
```

2. *Reduce timers:*
```
router bgp 64500
 timers bgp 10 30  # Keepalive 10s, Hold 30s
```

3. *BGP PIC (Prefix Independent Convergence):*
   - Pre-compute backup paths
   - Sub-second failover for covered prefixes

*Performance benchmarks:*
- Route processing: 10,000-100,000 routes/second (modern routers)
- Memory per route: ~500 bytes (varies by implementation)
- Full table convergence: 5-60 seconds depending on topology

== BGP Security

*Common attacks and mitigations:*

*1. Route hijacking:*
- Attacker announces victim's prefixes
- Historic examples: YouTube hijack by Pakistan Telecom (2008)

*Mitigations:*
- RPKI (Resource Public Key Infrastructure) [RFC 6480]
- ROA (Route Origin Authorization): Signed prefix-to-AS mapping
- ROV (Route Origin Validation): Reject invalid origins

```
# RPKI validation states
VALID:    Prefix matches ROA
INVALID:  Prefix covered by ROA but wrong origin AS
UNKNOWN:  No ROA exists for prefix
```

*2. BGP session hijacking:*
- TCP session injection
- Mitigations: TCP-AO (RFC 5925), TTL security (GTSM)

```
router bgp 64500
 neighbor 10.0.0.1 ttl-security hops 1
```

*3. Route leaks:*
- Misconfigured AS advertises routes it shouldn't
- Mitigation: BGP roles [RFC 9234], strict filtering

*Best practices:*
- Filter bogons (RFC 1918, default route)
- Max-prefix limits
- AS_PATH filtering (reject too-long paths)
- IRR (Internet Routing Registry) validation

== BGP in the Data Center

*Modern DC designs use BGP for underlay:*

*Clos/Leaf-Spine topology:*

```
       ┌────────┬────────┬────────┐
       │ Spine1 │ Spine2 │ Spine3 │
       └───┬────┴───┬────┴───┬────┘
           │        │        │
    ┌──────┼────────┼────────┼──────┐
    │      │        │        │      │
┌───┴─┐ ┌──┴──┐ ┌───┴──┐ ┌───┴─┐ ┌──┴──┐
│Leaf1│ │Leaf2│ │Leaf3 │ │Leaf4│ │Leaf5│
└──┬──┘ └──┬──┘ └──┬───┘ └──┬──┘ └──┬──┘
   │       │       │        │       │
  [Servers connected to leaves]
```

*Why BGP for DC?*
- Equal-cost multipath (ECMP) for load balancing
- Fast convergence with BFD
- Unified protocol (no OSPF+BGP complexity)
- Mature tooling and debugging

*BGP unnumbered [RFC 5765]:*
- Use link-local IPv6 for peering
- No IP address planning required
- Simplifies automation

*Performance in DC:*
- Convergence: $<1$s with BFD
- ECMP paths: 16-128 (hardware dependent)
- Scale: Thousands of leaves, millions of routes

== References

*Primary sources:*

RFC 4271: A Border Gateway Protocol 4 (BGP-4). Rekhter, Y., Li, T., & Hares, S. (2006).

RFC 4456: BGP Route Reflection. Bates, T., Chen, E., & Chandra, R. (2006).

RFC 1997: BGP Communities Attribute. Chandra, R., Traina, P., & Li, T. (1996).

RFC 8092: BGP Large Communities Attribute. Heitz, J. et al. (2017).

RFC 6480: An Infrastructure to Support Secure Internet Routing. Lepinski, M. & Kent, S. (2012).

RFC 5765: Security Extensions for BGP. Lepinski, M. (2010).

RFC 9234: Route Leak Prevention and Detection Using Roles. Azimov, A. et al. (2022).

Caesar, M. & Rexford, J. (2005). "BGP Routing Policies in ISP Networks." IEEE Network 19(6): 5-11.

Gill, P., Schapira, M., & Goldberg, S. (2011). "A Survey of Interdomain Routing Policies." ACM SIGCOMM CCR 44(1): 28-34.

Chung, T. et al. (2019). "RPKI is Coming of Age: A Longitudinal Study of RPKI Deployment and Invalid Route Origins." IMC '19.

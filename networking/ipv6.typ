#import "../template.typ": xref

= IPv6

IPv6 [RFC 8200] replaces IPv4's 32-bit address space with 128 bits, eliminates header checksums and in-network fragmentation, and bakes autoconfiguration into the protocol itself via Neighbor Discovery [RFC 4861] and SLAAC [RFC 4862]. Deployment is now majority traffic at major eyeball networks (Google measures roughly 45-50% IPv6 client traffic as of 2026), but coexistence with the IPv4 internet still drives most operational complexity: dual stack, NAT64/DNS64, 6rd, and 464XLAT remain everyday concerns.

*See also:* #xref("networking", "nat", label: "NAT") (for NAT64 stateful translation), #xref("networking", "dns", label: "DNS") (for AAAA, DNS64 synthesis), #xref("networking", "network-security", label: "Network Security") (for RA guard, SeND).

== Addressing Architecture

IPv6 addresses are 128 bits, written as eight 16-bit hex groups separated by colons, with `::` collapsing one run of zero groups (RFC 5952 canonical form):

```
2001:0db8:0000:0000:0000:ff00:0042:8329
2001:db8::ff00:42:8329          (canonical)
```

*Address scopes [RFC 4291]:*

#table(
  columns: (auto, auto, auto),
  [*Scope*], [*Prefix*], [*Use*],
  [Loopback], [`::1/128`], [Equivalent to `127.0.0.1`],
  [Unspecified], [`::/128`], [Source before assignment (DAD, DHCPv6)],
  [Link-local (LL)], [`fe80::/10`], [Single link, never routed; mandatory on every interface],
  [Unique local (ULA)], [`fc00::/7` (in practice `fd00::/8`)], [Private, like RFC 1918; not globally routed],
  [Global unicast (GUA)], [`2000::/3`], [Public internet],
  [Multicast], [`ff00::/8`], [Replaces IPv4 broadcast + multicast],
  [IPv4-mapped], [`::ffff:0:0/96`], [Dual-stack sockets (`::ffff:192.0.2.1`)],
  [Documentation], [`2001:db8::/32`], [Examples only],
)

Multicast scope is encoded in the second nibble: `ff02::1` (all-nodes link-local), `ff02::2` (all-routers), `ff02::1:ffXX:XXXX` (solicited-node, used by ND).

*Interface identifiers (IID):* The low 64 bits of a GUA/ULA. Originally derived from the MAC via modified EUI-64 (flip the U/L bit, insert `fffe`), now usually a stable opaque IID [RFC 7217] or a temporary privacy address [RFC 8981] to thwart tracking.

```bash
# Linux: show all addresses with scope
ip -6 addr show dev eth0

# Disable EUI-64, use stable privacy addresses
sysctl -w net.ipv6.conf.eth0.addr_gen_mode=2   # 2 = RFC 7217 stable privacy
sysctl -w net.ipv6.conf.eth0.use_tempaddr=2    # prefer temporary addresses
```

*Required addresses per interface:* link-local, solicited-node multicast for every unicast, all-nodes multicast. A host typically also holds one GUA plus one or more privacy addresses.

== Header Structure

The IPv6 header is fixed at 40 bytes and intentionally simpler than IPv4 — no checksum (transport layers must checksum), no options (moved to extension headers), no fragmentation fields (fragmentation is host-only via the Fragment extension header).

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|Version| Traffic Class |           Flow Label                  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|         Payload Length        |  Next Header  |   Hop Limit   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Source Address (128b)                   |
+                                                               +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Destination Address (128b)                 |
+                                                               +
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

*Fields:*
- *Version:* 6.
- *Traffic Class:* 8 bits (DSCP + ECN), same semantics as IPv4 ToS.
- *Flow Label:* 20 bits [RFC 6437]. Hash input for ECMP without parsing transport headers — critical for IPSec/encrypted traffic.
- *Payload Length:* 16-bit length of payload (extensions + transport). Jumbograms use the Hop-by-Hop Jumbo option for payloads larger than 65535 bytes.
- *Next Header:* Type of the following header (extension or transport: 6=TCP, 17=UDP, 58=ICMPv6, 44=Fragment, 50=ESP).
- *Hop Limit:* Replaces IPv4 TTL.

*Extension headers* form a linked list via Next Header. Defined order: Hop-by-Hop, Destination (1), Routing, Fragment, AH, ESP, Destination (2). Middleboxes that drop unknown extension headers are a real operational hazard [RFC 7872 measured 30-55% drop rates for some types].

*Minimum MTU:* 1280 bytes (vs. IPv4's 68). Path MTU Discovery [RFC 8201] is mandatory because routers never fragment — they emit `ICMPv6 Packet Too Big` (type 2) instead. Blocking ICMPv6 silently breaks IPv6.

== ICMPv6 and Neighbor Discovery

ICMPv6 [RFC 4443] is far more central than IPv4's ICMP: it carries ND, MLD, PMTUD, and Router Advertisements. Filtering it aggressively breaks the protocol.

*ND messages [RFC 4861]:* all use ICMPv6 with Hop Limit 255 (receivers MUST drop if Hop Limit `!=` 255 — prevents off-link forgery):

#table(
  columns: (auto, auto, auto),
  [*Type*], [*Name*], [*Purpose*],
  [133], [Router Solicitation (RS)], [Host asks for an RA at boot],
  [134], [Router Advertisement (RA)], [Router announces prefix, MTU, flags],
  [135], [Neighbor Solicitation (NS)], [Address resolution (replaces ARP); DAD],
  [136], [Neighbor Advertisement (NA)], [Reply to NS; unsolicited on address change],
  [137], [Redirect], [Better first-hop hint],
)

*Address resolution* uses the solicited-node multicast `ff02::1:ffXX:XXXX` (low 24 bits of the target). Only nodes whose addresses share those 24 bits process the NS, drastically reducing broadcast-style noise compared with ARP.

```
A (fe80::a) wants MAC for fe80::b
  A -> ff02::1:ffYY:YYYY  NS    target=fe80::b, SLLA=mac-a
  B -> fe80::a            NA    target=fe80::b, TLLA=mac-b, flags=S,O
```

*Duplicate Address Detection (DAD)* sends an NS for the candidate address with `::` as source before claiming it. If anyone responds with an NA, the address is abandoned. Optimistic DAD [RFC 4429] lets hosts use the address provisionally to cut startup latency.

*Neighbor cache states* (mirrors of ARP cache but with FSM):

```
  INCOMPLETE → REACHABLE → STALE → DELAY → PROBE → REACHABLE
                  ↑                                    │
                  └────────────────────────────────────┘
```

- `REACHABLE` is upper-layer confirmed (e.g., TCP ACK arrived). Default 30s.
- `STALE` entries are not refreshed until traffic flows; the first packet triggers `DELAY` then `PROBE` (unicast NS).

```bash
ip -6 neigh show
# fe80::1 dev eth0 lladdr 00:1a:2b:3c:4d:5e router REACHABLE
# 2001:db8::5 dev eth0 lladdr aa:bb:... STALE

ip -6 neigh flush dev eth0
```

*Neighbor cache pitfalls:*
- *GC thrash:* `net.ipv6.neigh.default.gc_thresh{1,2,3}` default to 128/512/1024. Top-of-rack switches with thousands of VMs need 16384+ or NS storms ensue.
- *Slow path on STALE:* an entry idle for ~30 min goes STALE; the very next packet incurs an extra RTT for the unicast NS probe. For low-latency RPC, send periodic keepalives or pin entries with `ip neigh replace ... nud permanent`.
- *RA-induced default route flapping:* if `accept_ra=1` and the router stops sending RAs, the default route expires and traffic blackholes. Use `accept_ra=2` on servers with static config.
- *Bogus NA hijack:* unsolicited NAs with the `Override` flag overwrite cache entries. Defend with RA Guard + ND inspection on the switch (akin to DHCP snooping).

== SLAAC, RA, and DHCPv6

Stateless Address Autoconfiguration [RFC 4862] lets a host derive a GUA from an RA-supplied prefix without any server. The RA carries flags that steer host behavior:

#table(
  columns: (auto, auto, auto),
  [*Flag*], [*Meaning*], [*Common combination*],
  [M (Managed)], [Get addresses via DHCPv6], [Enterprise: M=1, O=1, A=0],
  [O (Other)], [Get DNS/NTP via DHCPv6], [Home: M=0, O=0, A=1 (SLAAC + RDNSS option)],
  [A (Autonomous, per-prefix)], [Use SLAAC on this prefix], [],
  [L (On-link, per-prefix)], [Prefix is on-link], [],
)

*RA option highlights:* Prefix Information (the `/64`s to autoconfigure), MTU, Route Information [RFC 4191] for non-default routes, RDNSS / DNSSL [RFC 8106] for DNS servers, PREF64 [RFC 8781] advertising NAT64 prefix.

```bash
# Linux: read RA-derived state
ip -6 route show                              # default via fe80::1 expires N
sysctl net.ipv6.conf.eth0.accept_ra
ip -6 ntable show                             # ND table parameters

# Inspect RA on the wire
tcpdump -i eth0 -vv 'icmp6 and ip6[40] == 134'
```

*DHCPv6 [RFC 8415]* runs over UDP/546(client)/547(server) and addresses link-scoped multicast `ff02::1:2`. Unlike DHCPv4 it does *not* convey a default gateway — the router is always learned from the RA. Two flavors:
- *Stateful (IA_NA / IA_TA):* server assigns addresses; required for audit-heavy networks.
- *Stateless:* only DNS, domain search, NTP options; addresses come from SLAAC.

*Prefix Delegation (DHCPv6-PD) [RFC 8415 §6.3]* hands a CPE router a `/56` or `/48` to subnet downstream. This is how residential ISPs deploy IPv6: the customer router requests `IA_PD`, sub-allocates `/64`s to LANs, and re-advertises them in its own RAs.

*Android caveat:* Android still does not implement stateful DHCPv6. Networks that require DHCPv6 for address assignment do not work for Android clients; SLAAC + RDNSS is the lowest-common-denominator.

== MLD (Multicast Listener Discovery)

MLDv2 [RFC 3810] is the IPv6 counterpart of IGMPv3. Hosts announce multicast group membership so switches can avoid flooding. ND itself depends on multicast (solicited-node groups), so any switch doing MLD snooping must whitelist `ff02::1:ff00:0/104` or break IPv6 entirely — a classic deployment bug.

```bash
ip -6 maddr show dev eth0
# 1: eth0
#     inet6 ff02::1                          users 1
#     inet6 ff02::1:ff42:8329                users 1   (solicited-node)
```

== Transition and Coexistence

Few networks are pure IPv6; the long tail of IPv4-only services drives a zoo of transition mechanisms.

#table(
  columns: (auto, auto, auto, auto),
  [*Mechanism*], [*RFC*], [*Model*], [*Where used*],
  [Dual stack], [RFC 4213], [Run v4 + v6 in parallel], [Enterprises, servers],
  [6in4], [RFC 4213], [Static IPv6-in-IPv4 tunnel (proto 41)], [Hurricane Electric tunnels],
  [6to4], [RFC 3056], [`2002::/16` derived from IPv4; anycast relays], [Deprecated (RFC 7526)],
  [Teredo], [RFC 4380], [UDP-encapsulated, NAT-traversing], [Legacy Windows; mostly dead],
  [6rd], [RFC 5969], [ISP-controlled 6to4 with ISP prefix], [Free.fr, Swisscom early v6],
  [NAT64 / DNS64], [RFC 6146 / 6147], [Stateful v6→v4 translation + synthesized AAAA], [Mobile carriers, IPv6-only clouds],
  [464XLAT], [RFC 6877], [Customer-side stateless XLAT + NAT64], [T-Mobile, most LTE/5G carriers],
  [MAP-T / MAP-E], [RFC 7597 / 7599], [Algorithmic stateless v4-in-v6], [Comcast, JP IPv6 IPoE],
  [DS-Lite], [RFC 6333], [IPv4-in-IPv6 to carrier CGN], [European cable ISPs],
)

*Dual stack* is the simplest and the recommended default for servers: bind a socket with `AF_INET6` and (unless `IPV6_V6ONLY` is set) it accepts both v4 (via `::ffff:0:0/96`) and v6. Happy Eyeballs v2 [RFC 8305] on the client races A and AAAA lookups and connection attempts, preferring v6 by ~50 ms to avoid penalizing broken v6 paths.

*NAT64 + DNS64* enables IPv6-only client networks (now standard in mobile cores and AWS VPC IPv6-only subnets):

```
Client v6-only ── AAAA google.com ──> DNS64
                       │
                       │   A 142.250.80.46
                       │   synthesize AAAA 64:ff9b::142.250.80.46  (well-known prefix RFC 6052)
                       ▼
Client ── packet to 64:ff9b::8efa:502e ──> NAT64 box ── translates to v4 ──> 142.250.80.46
```

Limitations: literal IPv4 addresses in application payloads break (no DNS path), as do DNSSEC-validating stub resolvers (synthesized AAAA fails validation). 464XLAT papers over this by adding a stateless `CLAT` on the client that translates v4 packets to v6 with the NAT64 prefix before they hit the network.

*6rd* lets an ISP roll out IPv6 over its existing IPv4 infrastructure quickly: each customer gets `ISP_prefix:customer_v4_bits::/60`, encapsulated in IPv4 to a 6rd Border Relay. Operationally simpler than native v6 but adds 20 bytes of overhead and a fragmentation hazard.

== Linux and Operational Snippets

```bash
# Enable forwarding + accept RAs as a router (uncommon)
sysctl -w net.ipv6.conf.all.forwarding=1
sysctl -w net.ipv6.conf.eth0.accept_ra=2     # 2 = accept even when forwarding

# Static config
ip -6 addr add 2001:db8:1::1/64 dev eth0
ip -6 route add default via fe80::1 dev eth0

# Tune neighbor table for a busy ToR
sysctl -w net.ipv6.neigh.default.gc_thresh1=4096
sysctl -w net.ipv6.neigh.default.gc_thresh2=8192
sysctl -w net.ipv6.neigh.default.gc_thresh3=16384

# Path MTU
ip -6 route get 2001:db8::1
# expires 596sec mtu 1450  → PMTUD found a bottleneck link

# DAD failures (ipv6 disabled until resolved)
journalctl -k | grep -i "duplicate address"
```

*Socket programming:* prefer `AF_INET6` with `getaddrinfo(AI_ADDRCONFIG | AI_V4MAPPED)`. Set `IPV6_V6ONLY=0` for a dual-stack listener (default on BSD/Windows is `1`; Linux defaults to `0` but distros vary). For server selection, use `IPV6_PKTINFO` to learn which address received the datagram (essential for multi-homed UDP).

```c
int s = socket(AF_INET6, SOCK_STREAM, 0);
int off = 0;
setsockopt(s, IPPROTO_IPV6, IPV6_V6ONLY, &off, sizeof(off));
struct sockaddr_in6 sa = { .sin6_family = AF_INET6,
                           .sin6_port = htons(443),
                           .sin6_addr = in6addr_any };
bind(s, (struct sockaddr*)&sa, sizeof(sa));
```

== Performance Notes

- *Header overhead:* 40 vs 20 bytes. On 64-byte minimum Ethernet frames the payload share is smaller; in DC fabrics this is rarely a bottleneck.
- *Hardware offloads:* modern NICs do v6 checksum/TSO/LRO/RSS over flow label. Some older NICs only RSS on v4 5-tuple — verify with `ethtool -n eth0 rx-flow-hash tcp6`.
- *Extension headers* are slow path on most ASICs (Tofino, Trident) — avoid Hop-by-Hop options for production traffic; they often punt to CPU.
- *Routing table size:* IPv6 BGP table is ~200k routes (2026), 7x smaller than v4 (~970k). FIB memory pressure is much lower.

== Further Reading

*Core RFCs:*

RFC 8200: Internet Protocol, Version 6 (IPv6) Specification. Deering, S. & Hinden, R. (2017).

RFC 4291: IP Version 6 Addressing Architecture. Hinden, R. & Deering, S. (2006).

RFC 4861: Neighbor Discovery for IP version 6 (IPv6). Narten, T., Nordmark, E., Simpson, W. & Soliman, H. (2007).

RFC 4862: IPv6 Stateless Address Autoconfiguration. Thomson, S., Narten, T. & Jinmei, T. (2007).

RFC 4443: Internet Control Message Protocol (ICMPv6) for the Internet Protocol Version 6. Conta, A., Deering, S. & Gupta, M. (2006).

RFC 8201: Path MTU Discovery for IP version 6. McCann, J., Deering, S., Mogul, J. & Hinden, R. (2017).

RFC 8415: Dynamic Host Configuration Protocol for IPv6 (DHCPv6). Mrugalski, T., et al. (2018).

RFC 3810: Multicast Listener Discovery Version 2 (MLDv2) for IPv6. Vida, R. & Costa, L. (2004).

*Transition:*

RFC 6146: Stateful NAT64. Bagnulo, M., Matthews, P. & van Beijnum, I. (2011).

RFC 6147: DNS64 — DNS Extensions for Network Address Translation from IPv6 Clients to IPv4 Servers. Bagnulo, M., Sullivan, A., Matthews, P. & van Beijnum, I. (2011).

RFC 6877: 464XLAT — Combination of Stateful and Stateless Translation. Mawatari, M., Kawashima, M. & Byrne, C. (2013).

RFC 5969: IPv6 Rapid Deployment on IPv4 Infrastructures (6rd). Townsley, W. & Troan, O. (2010).

RFC 7526: Deprecating 6to4. Troan, O. & Carpenter, B. (2015).

RFC 8305: Happy Eyeballs Version 2. Schinazi, D. & Pauly, T. (2017).

*Operational and security:*

RFC 7217: A Method for Generating Semantically Opaque Interface Identifiers with IPv6 SLAAC. Gont, F. (2014).

RFC 8981: Temporary Address Extensions for SLAAC in IPv6. Gont, F., Krishnan, S., Narten, T. & Draves, R. (2021).

RFC 7872: Observations on the Dropping of Packets with IPv6 Extension Headers in the Real World. Gont, F., Linkova, J., Chown, T. & Liu, W. (2016).

RFC 6105: IPv6 Router Advertisement Guard. Levy-Abegnoli, E., Van de Velde, G., Popoviciu, C. & Mohacsi, J. (2011).

*Books and measurement:*

Hagen, S. (2014). _IPv6 Essentials_ (3rd ed.), O'Reilly.

Huston, G. _IPv6 BGP table reports_, APNIC (ongoing).

Czyz, J., Allman, M., Zhang, J., Iekel-Johnson, S., Osterweil, E. & Bailey, M. (2014). "Measuring IPv6 Adoption." SIGCOMM '14.

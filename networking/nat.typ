#import "../template.typ": xref

= Network Address Translation (NAT)

Network Address Translation rewrites IP and port headers as packets cross an administrative boundary. Originally introduced to mitigate IPv4 address exhaustion, NAT has become a de-facto security and topology hiding mechanism, and a perennial source of pain for peer-to-peer protocols.

*See also:* #xref("networking", "transport-layer", label: "Transport Layer") (for TCP/UDP), #xref("networking", "stateful-firewalls", label: "Stateful Firewalls") (for conntrack internals), #xref("networking", "ipv6", label: "IPv6") (for the long-term NAT-free alternative).

== NAT Overview

*Problem:* IPv4 address exhaustion (4.3 billion addresses for 8+ billion devices).

*Solution:* NAT allows multiple devices to share a single public IP address by translating private addresses at the network boundary.

```
                    NAT Gateway
Private Network     (Public IP)        Internet
┌──────────────┐   ┌───────────┐   ┌──────────────┐
│ 192.168.1.10 │──▶│           │──▶│              │
│ 192.168.1.11 │──▶│ 203.0.113.1 │──▶│  Server      │
│ 192.168.1.12 │──▶│           │──▶│  93.184.216.34│
└──────────────┘   └───────────┘   └──────────────┘
      src:192.168.1.10:54321       src:203.0.113.1:40001
      dst:93.184.216.34:80         dst:93.184.216.34:80
```

*Key insight:* NAT operates at layer 3 (IP) and layer 4 (ports), rewriting headers as packets traverse the boundary. The 5-tuple `(src_ip, src_port, dst_ip, dst_port, proto)` is mapped to a new external 5-tuple and stored in a state table.

== NAT Types

*1. SNAT (Source NAT):* Modifies source address of outgoing packets.

```bash
# iptables: SNAT to specific address
iptables -t nat -A POSTROUTING -o eth0 -j SNAT --to-source 203.0.113.1

# nftables equivalent
nft add rule nat postrouting oifname "eth0" snat to 203.0.113.1
```

*Use case:* Internal hosts accessing internet through gateway with static public IP.

*2. Masquerading:* Dynamic SNAT using outgoing interface's current IP.

```bash
# iptables: Masquerade (auto-detect interface IP)
iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE

# nftables equivalent
nft add rule nat postrouting oifname "eth0" masquerade
```

*Use case:* Home routers with dynamic (DHCP-assigned) public IPs. Slight overhead vs SNAT (lookup per packet).

*3. DNAT (Destination NAT):* Modifies destination address of incoming packets.

```bash
# iptables: Forward port 80 to internal server
iptables -t nat -A PREROUTING -p tcp --dport 80 -j DNAT --to-destination 192.168.1.10:8080

# nftables equivalent
nft add rule nat prerouting tcp dport 80 dnat to 192.168.1.10:8080
```

*Use case:* Port forwarding, load balancing, hosting services behind NAT.

*4. Full Cone NAT (1:1):* Once internal host sends to any external address, any external host can send to mapped external address.

*5. Restricted Cone NAT:* External host can reply only if internal host previously contacted it.

*6. Symmetric NAT:* Different mapping for each external destination. Most restrictive; breaks many P2P protocols.

== Carrier-Grade NAT (CGNAT)

CGNAT (also called large-scale NAT, LSN, or NAT444) is operated by ISPs to share a small pool of public IPv4 addresses across thousands of subscribers, typically the only way mobile carriers and broadband ISPs can still deliver IPv4 service after RIR pool exhaustion (2011-2019 across regions).

*Address space:* RFC 6598 reserves `100.64.0.0/10` (4 million addresses) specifically for CGNAT; distinct from RFC 1918 to avoid collision with subscriber LANs.

```
Subscriber CPE      ISP CGNAT          Internet
192.168.1.0/24  →  100.64.x.y/32  →  203.0.113.0/24 (pool)
   (RFC 1918)       (RFC 6598)         (Public)
```

*Port allocation strategies:*
- *Static port block:* Each subscriber gets, e.g., 1024 ports on one public IP. Predictable, easy to log, but wasteful.
- *Dynamic on-demand:* Ports allocated as flows arrive. Better utilization, but logging becomes a nightmare (timestamped flow records required for law-enforcement requests).
- *Deterministic NAT (RFC 7422):* Algorithmic mapping `subscriber → (public_ip, port_range)`. Eliminates per-flow logging.

*Scale numbers (typical Tier-1 mobile operator):*

#table(
  columns: (auto, auto),
  [*Metric*], [*Value*],
  [Subscribers per public IPv4], [64-1024],
  [Concurrent flows per subscriber], [100-500],
  [Total flows on a single CGNAT box], [10-100 million],
  [Connection setup rate], [1M+ per second],
  [State table memory], [tens of GB],
)

*Problems caused by CGNAT:*
- Server-side rate limiting (e.g., "too many requests from this IP") penalizes innocent subscribers sharing the IP.
- Geo-IP databases lose accuracy.
- Inbound connections (gaming, P2P, self-hosted services) impossible without port forwarding cooperation from the ISP.
- Port exhaustion under load (especially with web pages opening many parallel TLS connections).

*Mitigations:* PCP (Port Control Protocol, RFC 6887) lets clients request inbound port mappings. IPv6 deployment removes the need entirely for native traffic.

== NAT Traversal

*Problem:* NAT breaks end-to-end connectivity. Hosts behind NAT cannot receive unsolicited incoming connections.

*Challenge for P2P, VoIP, gaming:*
```
Host A (NAT)                      Host B (NAT)
192.168.1.10 ──▶ NAT-A ──┐   ┌── NAT-B ◀── 192.168.1.20
                          │   │
                          ▼   ▼
                       Internet

Problem: Neither host knows the other's public IP\:port
```

=== STUN — Session Traversal Utilities for NAT

*STUN [RFC 5389/8489]:*
```
1. Client sends request to STUN server
2. Server responds with client's public IP:port (as seen by server)
3. Client now knows its external address for P2P signaling

Client              STUN Server           Peer
  │  Binding Req        │                   │
  ├────────────────────▶│                   │
  │  Binding Resp       │                   │
  │  (your IP:port)     │                   │
  │◀────────────────────┤                   │
  │                     │                   │
  │  Share via signaling│                   │
  ├─────────────────────────────────────────▶
```

```bash
# Probe public mapping with stun-client
stun stun.l.google.com:19302

# Or with the `pystun3` library
python3 -c "import stun; print(stun.get_ip_info())"
```

=== UDP and TCP Hole Punching

*Hole punching:* Both peers send UDP packets to each other's external address simultaneously. First outbound packet from each side creates a NAT mapping; the subsequent inbound packet from the peer matches the mapping and flows through.

```
1. Both A and B register with rendezvous server
2. Server returns each peer's public (IP, port)
3. A → B (creates A's NAT mapping; dropped at B's NAT)
4. B → A (creates B's NAT mapping; matches A's; flows through)
5. A → B now matches B's mapping; bidirectional flow established
```

TCP hole punching is harder: the SYN/SYN-ACK timing window is tight, and many NATs drop unsolicited SYNs aggressively. Success rate hovers around 50%.

=== TURN — Traversal Using Relays around NAT

*TURN [RFC 5766/8656]:* Fallback when hole punching fails (symmetric NAT). All traffic relayed through TURN server.

```
A ──UDP──▶ TURN ──UDP──▶ B
       (relays both directions; allocates a public port for B)
```

TURN servers consume bandwidth proportional to all relayed flows; operators (Twilio, Cloudflare, Google Meet) run global fleets and price by GB.

=== ICE — Interactive Connectivity Establishment

*ICE [RFC 8445]:* Framework combining STUN, TURN, and direct connectivity checks. Used by WebRTC.

```
ICE candidate gathering:
1. Host candidates (local IP:port)
2. Server reflexive candidates (STUN-discovered public IP:port)
3. Relay candidates (TURN server allocation)

ICE connectivity checks:
- Try all candidate pairs in priority order
- Select best working pair (prefer direct > STUN > TURN)
- Trickle ICE: emit candidates as they become available
  (cuts WebRTC setup from ~5s to <1s)
```

=== UPnP IGD and NAT-PMP / PCP

For consumer routers, the device itself can request a port mapping from the gateway:

```bash
# UPnP: query and request port mapping
upnpc -l                    # list current mappings
upnpc -a 192.168.1.10 8080 8080 TCP   # forward external 8080 → LAN 8080

# NAT-PMP / PCP (Apple / RFC 6887): lower overhead, no XML
pcp map tcp 8080 -d 1h
```

Enterprise and carrier NATs disable these protocols (security policy). Success rate on home routers is ~60%, falling fast as households adopt CGNAT.

== Common NAT Configurations

*Basic NAT gateway (home router):*
```bash
# Enable IP forwarding
echo 1 > /proc/sys/net/ipv4/ip_forward

# NAT for outbound traffic
iptables -t nat -A POSTROUTING -o wan0 -j MASQUERADE

# Allow forwarding for established connections
iptables -A FORWARD -i lan0 -o wan0 -j ACCEPT
iptables -A FORWARD -i wan0 -o lan0 -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
```

*Port forwarding (expose internal service):*
```bash
# Forward port 443 to internal web server
iptables -t nat -A PREROUTING -i wan0 -p tcp --dport 443 -j DNAT --to 192.168.1.10:443
iptables -A FORWARD -i wan0 -p tcp -d 192.168.1.10 --dport 443 -j ACCEPT
```

*1:1 NAT (bidirectional, e.g., DMZ host):*
```bash
iptables -t nat -A PREROUTING  -d 203.0.113.50 -j DNAT --to 192.168.1.50
iptables -t nat -A POSTROUTING -s 192.168.1.50 -j SNAT --to 203.0.113.50
```

*Hairpin NAT* (internal hosts accessing internal server via public IP):
```bash
iptables -t nat -A POSTROUTING -s 192.168.1.0/24 -d 192.168.1.10 \
         -p tcp --dport 80 -j MASQUERADE
```

== Common Pitfalls

*1. Port exhaustion:* Each NAT mapping consumes a source port (16 bits = 65535 ports per public IP). A single subscriber opening 500 parallel HTTP/2 connections to Google can fill 5-10% of a CGNAT public IP's port pool. Mitigation: more public IPs, deterministic NAT, lower TCP TIME_WAIT.

*2. Asymmetric routing:* Traffic enters via one path, exits via another. NAT sees only half the flow; conntrack drops it as INVALID.

*3. MTU/PMTUD black-holing:* NAT devices that filter ICMP "Fragmentation Needed" break Path MTU Discovery. Workaround:
```bash
iptables -A FORWARD -p tcp --tcp-flags SYN,RST SYN \
         -j TCPMSS --clamp-mss-to-pmtu
```

*4. FTP/SIP ALG:* Active FTP and SIP embed IP addresses in payload. Requires application-layer gateway (ALG) modules, and these ALGs are themselves a frequent source of bugs.
```bash
modprobe nf_conntrack_ftp
modprobe nf_nat_ftp
modprobe nf_conntrack_sip
modprobe nf_nat_sip
```

*5. Symmetric NAT breaking WebRTC:* Detected only at runtime; clients fall back to TURN with ~2-5ms added latency and bandwidth cost on the relay.

== NAT Strategy Comparison

#table(
  columns: (auto, auto, auto, auto),
  [*NAT Type*], [*Mapping*], [*Filtering*], [*P2P Compatibility*],
  [Full Cone], [Endpoint-independent], [Endpoint-independent], [Excellent: any external host can reach mapped port],
  [Restricted Cone], [Endpoint-independent], [Address-restricted], [Good: must send outbound first],
  [Port Restricted Cone], [Endpoint-independent], [Address + port restricted], [Moderate: STUN usually works],
  [Symmetric], [Endpoint-dependent], [Endpoint-dependent], [Poor: requires TURN relay],
)

=== NAT Traversal Technique Comparison

#table(
  columns: (auto, auto, auto, auto),
  [*Technique*], [*Success Rate*], [*Latency Overhead*], [*Notes*],
  [STUN], [~85% (fails symmetric)], [1 RTT discovery], [Lightweight, UDP only],
  [TURN], [~100%], [+2-5ms relay hop], [Fallback relay, bandwidth cost],
  [ICE], [~95%+], [~100-500ms negotiation], [Combines STUN + TURN, used by WebRTC],
  [UPnP / NAT-PMP / PCP], [~60% (home routers)], [None after setup], [Disabled on enterprise / CGNAT],
  [UDP hole punching], [~80%], [1-2 RTT coordination], [Requires rendezvous server],
  [TCP hole punching], [~50%], [2-3 RTT], [Harder due to SYN filtering],
)

== Further Reading

RFC 3022: Traditional IP Network Address Translator (Srisuresh & Egevang, 2001).

RFC 4787: NAT Behavioral Requirements for Unicast UDP (Audet & Jennings, 2007).

RFC 5389 / RFC 8489: Session Traversal Utilities for NAT (STUN) (Rosenberg et al.).

RFC 5766 / RFC 8656: Traversal Using Relays around NAT (TURN).

RFC 6598: IANA-Reserved IPv4 Prefix for Shared Address Space (CGNAT) (Weil et al., 2012).

RFC 6887: Port Control Protocol (PCP) (Wing, Cheshire et al., 2013).

RFC 7422: Deterministic Address Mapping to Reduce Logging in CGN Deployments.

RFC 8445: Interactive Connectivity Establishment (ICE) (Keranen et al., 2018).

Ford, B., Srisuresh, P. & Kegel, D. (2005). "Peer-to-Peer Communication Across Network Address Translators." USENIX ATC.

Donley, C. et al. (2011). "Assessing the Impact of Carrier-Grade NAT on Network Applications." IETF draft / measurements.

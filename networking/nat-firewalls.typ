= NAT & Firewalls

Network Address Translation (NAT) and firewalls are fundamental to modern networking, enabling address conservation, security boundaries, and traffic control.

*See also:* Internet Layer (for IP addressing), Transport Layer (for TCP/UDP), Sockets API (for NAT traversal in applications)

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

*Key insight:* NAT operates at layer 3 (IP) and layer 4 (ports), rewriting headers as packets traverse the boundary.

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

== Connection Tracking (conntrack)

*Problem:* NAT requires tracking active connections to correctly translate return packets.

*Solution:* conntrack subsystem in Linux kernel maintains state table of all connections.

```
Connection tracking flow:

  Incoming ─▶ PREROUTING ─▶ conntrack ─▶ routing ─▶ FORWARD ─▶ POSTROUTING ─▶ Outgoing
              (DNAT)         lookup      decision              (SNAT)
                               │
                               ▼
                        ┌─────────────┐
                        │ State Table │
                        │ ─────────── │
                        │ tuple → NAT │
                        │ mapping     │
                        └─────────────┘
```

*Connection states:*
- *NEW:* First packet of connection (SYN for TCP, any for UDP)
- *ESTABLISHED:* Subsequent packets in tracked connection
- *RELATED:* New connection related to existing (FTP data, ICMP errors)
- *INVALID:* Packet doesn't match known connection (potential attack)

*Conntrack tuple (5-tuple):* `<src_ip, src_port, dst_ip, dst_port, protocol>`

```bash
# View active connections
conntrack -L

# Example output:
# tcp  6 431999 ESTABLISHED src=192.168.1.10 dst=93.184.216.34 sport=54321 dport=80
#      src=93.184.216.34 dst=203.0.113.1 sport=80 dport=40001 [ASSURED] mark=0 use=1

# Monitor connections in real-time
conntrack -E

# Clear all tracked connections
conntrack -F
```

*Conntrack table sizing:*
```bash
# View current limit
cat /proc/sys/net/netfilter/nf_conntrack_max  # Default: 65536

# View current count
cat /proc/sys/net/netfilter/nf_conntrack_count

# Increase limit (high-traffic NAT gateway)
echo 262144 > /proc/sys/net/netfilter/nf_conntrack_max
# Or permanently in /etc/sysctl.conf:
# net.netfilter.nf_conntrack_max = 262144
```

*Memory usage:* ~300 bytes per entry. 262144 entries = ~80MB.

== iptables Architecture

*iptables* uses tables containing chains of rules for packet processing.

```
Tables and their chains:

┌─────────────────────────────────────────────────────────────────────────┐
│                           PACKET FLOW                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                         ┌───────────────┐
                         │   PREROUTING  │ ◀── raw, mangle, nat (DNAT)
                         └───────┬───────┘
                                 │
                    ┌────────────┴────────────┐
                    │     Routing Decision    │
                    └────────────┬────────────┘
                    ▼                          ▼
           ┌───────────────┐          ┌───────────────┐
           │     INPUT     │          │    FORWARD    │ ◀── mangle, filter
           │ (local dest)  │          │  (transit)    │
           └───────┬───────┘          └───────┬───────┘
                   │                          │
                   ▼                          │
           ┌───────────────┐                  │
           │  Local Process │                  │
           └───────┬───────┘                  │
                   │                          │
                   ▼                          ▼
           ┌───────────────┐          ┌───────────────┐
           │    OUTPUT     │──────────│  POSTROUTING  │ ◀── mangle, nat (SNAT)
           └───────────────┘          └───────────────┘
```

*Tables:*
- *filter:* Default table, packet filtering (ACCEPT, DROP, REJECT)
- *nat:* Network address translation (SNAT, DNAT, MASQUERADE)
- *mangle:* Packet header modification (TTL, TOS, MARK)
- *raw:* Bypasses conntrack (high-performance, stateless filtering)

*Rule matching:*
```bash
# Basic rule structure
iptables -t <table> -A <chain> <match> -j <target>

# Examples:
# Drop incoming SSH from specific IP
iptables -A INPUT -p tcp --dport 22 -s 10.0.0.5 -j DROP

# Allow established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Log and drop invalid packets
iptables -A INPUT -m conntrack --ctstate INVALID -j LOG --log-prefix "INVALID: "
iptables -A INPUT -m conntrack --ctstate INVALID -j DROP
```

== nftables Architecture

*nftables* is the modern replacement for iptables (Linux 3.13+, default in many distros since 2019).

*Advantages over iptables:*
- Single framework (replaces iptables, ip6tables, arptables, ebtables)
- Atomic rule updates (no packet loss during reload)
- Built-in sets and maps for efficient matching
- Simpler syntax, better performance

```bash
# nftables configuration structure
table inet filter {          # inet = IPv4 + IPv6
    chain input {
        type filter hook input priority 0; policy drop;

        ct state established,related accept
        ct state invalid drop

        tcp dport 22 accept
        tcp dport { 80, 443 } accept
    }

    chain forward {
        type filter hook forward priority 0; policy drop;
    }

    chain output {
        type filter hook output priority 0; policy accept;
    }
}

table inet nat {
    chain prerouting {
        type nat hook prerouting priority -100;
        tcp dport 80 dnat to 192.168.1.10:8080
    }

    chain postrouting {
        type nat hook postrouting priority 100;
        oifname "eth0" masquerade
    }
}
```

*Sets for efficient matching:*
```bash
# Define set of allowed IPs
nft add set inet filter allowed_hosts { type ipv4_addr \; }
nft add element inet filter allowed_hosts { 10.0.0.1, 10.0.0.2, 10.0.0.3 }
nft add rule inet filter input ip saddr @allowed_hosts accept
```

*Performance:* O(1) set lookup vs O(n) linear rule matching in iptables.

== Performance Implications

*Stateful inspection overhead:*

```
Per-packet processing cost:

Operation                    | Cycles   | Time (3GHz CPU)
─────────────────────────────┼──────────┼─────────────────
Conntrack lookup (hash)      | ~200-500 | ~70-170ns
NAT translation              | ~100-200 | ~35-70ns
Filter rule matching (10)    | ~500-1000| ~170-330ns
Filter rule matching (100)   | ~2000-5000| ~700-1700ns
nftables set lookup          | ~100-200 | ~35-70ns
```

*Conntrack table contention:*
- Single global lock in older kernels (pre-4.7)
- Per-bucket locking in modern kernels (4.7+)
- At 1M+ connections, consider conntrack bypass for known-safe traffic

*Bypass strategies:*
```bash
# Skip conntrack for high-volume, stateless traffic
iptables -t raw -A PREROUTING -p udp --dport 53 -j NOTRACK
iptables -t raw -A OUTPUT -p udp --sport 53 -j NOTRACK

# nftables equivalent
nft add rule inet raw prerouting udp dport 53 notrack
```

*Benchmarks (10Gbps NIC, small packets):*
- Without conntrack: ~14.8 Mpps (line rate for 64B packets)
- With conntrack: ~8-10 Mpps (~30% overhead)
- With 100 iptables rules: ~3-5 Mpps
- With nftables sets: ~10-12 Mpps

== NAT Traversal

*Problem:* NAT breaks end-to-end connectivity. Hosts behind NAT cannot receive unsolicited incoming connections.

*Challenge for P2P, VoIP, gaming:*
```
Host A (NAT)                      Host B (NAT)
192.168.1.10 ──▶ NAT-A ──┐   ┌── NAT-B ◀── 192.168.1.20
                          │   │
                          ▼   ▼
                       Internet

Problem: Neither host knows the other's public IP:port
```

*STUN (Session Traversal Utilities for NAT) [RFC 5389]:*
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

*Hole punching:* Both peers send UDP packets to each other's external address simultaneously. First packet creates NAT mapping; subsequent packets flow through.

*TURN (Traversal Using Relays around NAT) [RFC 5766]:*
Fallback when hole punching fails (symmetric NAT). All traffic relayed through TURN server.

*ICE (Interactive Connectivity Establishment) [RFC 8445]:*
Framework combining STUN, TURN, and direct connectivity checks. Used by WebRTC.

```
ICE candidate gathering:
1. Host candidates (local IP:port)
2. Server reflexive candidates (STUN-discovered public IP:port)
3. Relay candidates (TURN server allocation)

ICE connectivity checks:
- Try all candidate pairs
- Select best working pair (prefer direct > STUN > TURN)
```

== Common Configurations

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

*Stateful firewall (server):*
```bash
# Default deny
iptables -P INPUT DROP
iptables -P FORWARD DROP
iptables -P OUTPUT ACCEPT

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT

# Allow established
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Allow specific services
iptables -A INPUT -p tcp --dport 22 -j ACCEPT   # SSH
iptables -A INPUT -p tcp --dport 80 -j ACCEPT   # HTTP
iptables -A INPUT -p tcp --dport 443 -j ACCEPT  # HTTPS
```

== Common Pitfalls

*1. Conntrack table exhaustion:*
```bash
# Symptom: "nf_conntrack: table full, dropping packet" in dmesg
# Solution: Increase table size or reduce timeouts
net.netfilter.nf_conntrack_max = 524288
net.netfilter.nf_conntrack_tcp_timeout_established = 86400  # 1 day vs 5 days
```

*2. Asymmetric routing:* Traffic enters via one path, exits via another. Conntrack sees only half the connection.

*3. MTU issues with NAT:* Ensure PMTU discovery works or clamp MSS.
```bash
iptables -A FORWARD -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu
```

*4. FTP/SIP ALG:* Active FTP and SIP embed IP addresses in payload. Requires application-layer gateway (ALG) modules.
```bash
modprobe nf_conntrack_ftp
modprobe nf_nat_ftp
```

*5. Hairpin NAT:* Internal hosts accessing internal server via public IP. Requires special rules.
```bash
# Allow internal traffic to loop back through NAT
iptables -t nat -A POSTROUTING -s 192.168.1.0/24 -d 192.168.1.10 -p tcp --dport 80 -j MASQUERADE
```

== How a Stateful Firewall Works

A stateful firewall tracks full connection state rather than evaluating each packet independently. Every accepted new flow is recorded in a state table; return packets match the existing entry and bypass rule evaluation.

=== Stateless vs Stateful vs DPI

#table(
  columns: (auto, auto, auto, auto),
  [*Type*], [*Inspection Level*], [*Per-packet Cost*], [*Typical Use*],
  [Stateless ACL], [IP/port/proto only], [$O(n)$ rule list], [Core routers, backbone ACLs],
  [Stateful], [State table + headers], [$O(1)$ hash on ESTABLISHED], [Perimeter firewalls, NAT GW],
  [NGFW / DPI], [Payload, TLS SNI, AppID], [3-10x stateful], [Enterprise, IDS/IPS, DLP],
  [eBPF/XDP], [Custom, kernel bypass], [$<$ 100 ns, line-rate], [Cloud-native, hyperscale CDN],
)

=== TCP State Machine in conntrack

```
NEW         → SYN received; entry created; waiting for SYN-ACK
ESTABLISHED → SYN-ACK + ACK seen; bidirectional flow allowed
FIN_WAIT    → FIN seen; connection closing
TIME_WAIT   → 2×MSL timer (~60s); prevent stale segment reuse
INVALID     → No matching state; DROP immediately
```

*SYN flood defence* — attacker sends many SYNs to fill the NEW table:

```bash
# SYN cookies: issue cryptographic SYN-ACK without allocating state
echo 1 > /proc/sys/net/ipv4/tcp_syncookies

# nftables: rate-limit new TCP connections per source IP
table inet filter {
    chain input {
        type filter hook input priority 0; policy drop;

        ct state established,related accept
        ct state invalid drop

        # Allow loopback
        iifname "lo" accept

        # Rate-limit new connections: 50/sec, burst 100
        tcp flags & (fin|syn|rst|ack) == syn \
            limit rate 50/second burst 100 packets accept
        tcp flags & (fin|syn|rst|ack) == syn drop
    }
}
```

=== Packet Walk Through the Firewall

```
Incoming packet
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ 1. XDP hook (optional eBPF fast-path)                    │
│    → IP blocklist / DDoS scrubbing at line rate          │
│    → Returns XDP_DROP before kernel stack is touched     │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│ 2. conntrack lookup on 5-tuple                           │
│    HIT  (ESTABLISHED) → fast accept, skip rules         │
│    MISS (NEW)         → create entry, run rule chains   │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
                     PREROUTING chain
                     (DNAT, mark)
                            │
                   ┌────────┴────────┐
              local?              forward?
                   │                 │
              INPUT chain       FORWARD chain
                   │                 │
                   └────────┬────────┘
                            │
                            ▼
                       Rule match:
                    ACCEPT / DROP /
                    REJECT / LOG
                            │
                            ▼
                    POSTROUTING (SNAT)
```

*ESTABLISHED fast path:* once a flow is ESTABLISHED, conntrack returns immediately — no rule chains traversed. This is why stateful firewalls stay fast at millions of concurrent connections.

=== eBPF/XDP Firewall (Modern, Kernel-Bypass)

XDP hooks into the NIC driver at interrupt time — packet never enters the kernel networking stack.

```cpp
// xdp_firewall.c — drop packets matching blocklist BPF map
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <arpa/inet.h>

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key,   __u32);   // src IPv4
    __type(value, __u8);    // 1 = blocked
    __uint(max_entries, 1 << 20);
} blocklist SEC(".maps");

SEC("xdp")
int xdp_fw(struct xdp_md *ctx) {
    void *data     = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end) return XDP_PASS;
    if (eth->h_proto != bpf_htons(ETH_P_IP)) return XDP_PASS;

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end) return XDP_PASS;

    __u32 src = ip->saddr;
    if (bpf_map_lookup_elem(&blocklist, &src))
        return XDP_DROP;    // dropped before any kernel processing

    return XDP_PASS;
}

char LICENSE[] SEC("license") = "GPL";
```

```bash
# Compile and attach
clang -O2 -target bpf -c xdp_firewall.c -o xdp_firewall.o
ip link set eth0 xdp obj xdp_firewall.o sec xdp

# Add IP to blocklist (atomic, zero-downtime, no iptables reload)
bpftool map update pinned /sys/fs/bpf/blocklist \
    key hex c0 a8 01 05 \   # 192.168.1.5 in little-endian
    value hex 01

# Dump all blocked IPs
bpftool map dump pinned /sys/fs/bpf/blocklist

# Show XDP stats
bpftool prog show
ip link show eth0 | grep xdp
```

*Performance (Intel i40e 25 GbE, 64B packets):*

#table(
  columns: (auto, auto, auto),
  [*Approach*], [*Throughput*], [*Latency per packet*],
  [iptables DROP], [3-5 Mpps], [~600 ns],
  [nftables DROP], [10-12 Mpps], [~170 ns],
  [XDP DROP (driver mode)], [24+ Mpps (line rate)], [$<$ 60 ns],
  [XDP DROP (offload, SmartNIC)], [100+ Mpps], [$<$ 10 ns],
)

=== Next-Generation Firewall (NGFW) Internals

*Application identification without decryption:*
- Port 443 may be HTTPS, Zoom, Netflix, SSH-over-HTTPS, or C2 malware
- TLS SNI (Server Name Indication): hostname visible in ClientHello plaintext
- JA3/JA4 fingerprint: hash of TLS parameters — identifies client library
- Flow byte pattern ML classifier: trained on first 10 packets

```bash
# Extract TLS SNI with tshark (Wireshark CLI)
tshark -i eth0 -Y "tls.handshake.type == 1" \
       -T fields -e ip.src -e tls.handshake.extensions_server_name

# JA3 fingerprint (client TLS fingerprint)
tshark -i eth0 -Y "tls.handshake.type == 1" \
       -T fields -e ja3.hash
```

*TLS inspection (SSL bump):*
```
Client ──TLS──▶ NGFW ──TLS──▶ Server
        (MITM: decrypt, inspect, re-encrypt)
```
NGFW acts as CA: issues a certificate signed by a trusted enterprise CA.
Required: enterprise CA root cert installed on all managed endpoints.

*IPS rule example (Suricata — CVE-2021-44228 Log4Shell):*
```
alert http any any -> $HTTP_SERVERS any (
    msg:"ET EXPLOIT Apache log4j RCE (Log4Shell)";
    flow:to_server,established;
    content:"${jndi:"; fast_pattern; nocase;
    reference:cve,2021-44228;
    classtype:attempted-admin; sid:2034647; rev:5;
)
```

=== Cloud-Native Firewalling

*Linux nftables — production server baseline:*
```bash
#!/usr/sbin/nft -f
flush ruleset

table inet filter {
    # IP allowlist set (O(1) hash lookup)
    set mgmt_hosts {
        type ipv4_addr
        elements = { 10.0.0.1, 10.0.0.2 }
    }

    chain input {
        type filter hook input priority 0; policy drop;

        iifname "lo" accept
        ct state { established, related } accept
        ct state invalid drop

        # ICMP/ICMPv6
        ip protocol icmp accept
        ip6 nexthdr icmpv6 accept

        # SSH only from management hosts
        tcp dport 22 ip saddr @mgmt_hosts accept

        # Public services
        tcp dport { 80, 443 } accept

        # Log and drop everything else
        limit rate 5/minute log prefix "FIREWALL DROP: " flags all
        drop
    }

    chain forward { type filter hook forward priority 0; policy drop; }
    chain output  { type filter hook output  priority 0; policy accept; }
}
```

*Cilium (eBPF-based, Kubernetes):*
```yaml
# Deny all → allow only frontend→backend:8080
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: backend-ingress
spec:
  endpointSelector:
    matchLabels: { app: backend }
  ingress:
  - fromEndpoints:
    - matchLabels: { app: frontend }
    toPorts:
    - ports:
      - port: "8080"
        protocol: TCP
```

```bash
# Live drop monitoring
cilium monitor --type drop

# Policy enforcement status
cilium endpoint list
cilium policy get
```

*AWS Security Groups* — stateful, evaluated at hypervisor ENI:
```bash
# Allow HTTPS from anywhere; SSH from bastion only
aws ec2 authorize-security-group-ingress \
    --group-id sg-0abc123 --protocol tcp --port 443 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress \
    --group-id sg-0abc123 --protocol tcp --port 22 \
    --source-group sg-0bastion
```
Return traffic is automatically allowed (stateful) — no explicit egress rule needed.

== NAT Strategy Comparison

#table(
  columns: (auto, auto, auto, auto),
  [*NAT Type*], [*Mapping*], [*Filtering*], [*P2P Compatibility*],
  [Full Cone], [Endpoint-independent], [Endpoint-independent], [Excellent — any external host can reach mapped port],
  [Restricted Cone], [Endpoint-independent], [Address-restricted], [Good — must send outbound first],
  [Port Restricted Cone], [Endpoint-independent], [Address + port restricted], [Moderate — STUN usually works],
  [Symmetric], [Endpoint-dependent], [Endpoint-dependent], [Poor — requires TURN relay],
)

=== NAT Traversal Technique Comparison

#table(
  columns: (auto, auto, auto, auto),
  [*Technique*], [*Success Rate*], [*Latency Overhead*], [*Notes*],
  [STUN], [~85% (fails symmetric)], [1 RTT discovery], [Lightweight, UDP only],
  [TURN], [~100%], [+2-5ms relay hop], [Fallback relay, bandwidth cost],
  [ICE], [~95%+], [~100-500ms negotiation], [Combines STUN + TURN, used by WebRTC],
  [UPnP/NAT-PMP], [~60% (home routers)], [None after setup], [Not available on enterprise/carrier NAT],
  [UDP hole punching], [~80%], [1-2 RTT coordination], [Requires rendezvous server],
  [TCP hole punching], [~50%], [2-3 RTT], [Harder due to SYN filtering],
)

=== Firewall Rule Processing Performance

#table(
  columns: (auto, auto, auto),
  [*Framework*], [*Rule Lookup*], [*Performance at Scale*],
  [iptables (linear)], [$O(n)$ per packet], [Degrades >5000 rules; ~10% throughput loss per 1000 rules],
  [iptables + ipset], [$O(1)$ hash / $O(log n)$ tree], [Handles 100K+ entries efficiently],
  [nftables (sets)], [$O(1)$ hash lookup], [Native set support, better than iptables at scale],
  [nftables (maps)], [$O(1)$ verdict maps], [Single rule replaces many; 2-5x fewer rules needed],
  [eBPF/XDP], [$O(1)$ hash maps], [Line-rate processing, bypass kernel stack],
)

_Modern deployments: prefer nftables over iptables. For >10 Gbps, consider XDP for stateless filtering._

== References

RFC 3022: Traditional IP Network Address Translator. Srisuresh, P. & Egevang, K. (2001).

RFC 5389: Session Traversal Utilities for NAT (STUN). Rosenberg, J. et al. (2008).

RFC 5766: Traversal Using Relays around NAT (TURN). Mahy, R. et al. (2010).

RFC 8445: Interactive Connectivity Establishment (ICE). Keranen, A. et al. (2018).

Welte, H. & Ayuso, P.N. "Netfilter/iptables Project." netfilter.org.

nftables Wiki. "nftables documentation." wiki.nftables.org.

Rosen, R. (2014). "Linux Kernel Networking: Implementation and Theory." Apress.

= Stateful Firewalls

A stateful firewall tracks full connection state rather than evaluating each packet independently. Every accepted new flow is recorded in a state table; return packets match the existing entry and bypass rule evaluation. This chapter covers Linux netfilter / nftables, connection tracking (conntrack), eBPF/XDP firewalling, and next-generation deep-packet-inspection (NGFW) systems.

*See also:* _NAT_ (which depends on conntrack), _Network Security_ (for higher-level policy), _Kernel Bypass_ (for XDP context), _Container Networking_ (for Cilium / per-pod policy).

== Connection Tracking (conntrack)

*Problem:* Stateful filtering and NAT both require tracking active connections to correctly handle return packets.

*Solution:* the `conntrack` subsystem in the Linux kernel maintains a state table of all flows.

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

== iptables Architecture

`iptables` uses tables containing chains of rules for packet processing.

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

# Drop incoming SSH from specific IP
iptables -A INPUT -p tcp --dport 22 -s 10.0.0.5 -j DROP

# Allow established connections
iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# Log and drop invalid packets
iptables -A INPUT -m conntrack --ctstate INVALID -j LOG --log-prefix "INVALID: "
iptables -A INPUT -m conntrack --ctstate INVALID -j DROP
```

== nftables Architecture

`nftables` is the modern replacement for iptables (Linux 3.13+, default in many distros since 2019).

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

== Zone-Based Firewalls

A *zone* groups interfaces with a shared trust level (e.g., `trusted`, `internal`, `public`, `dmz`). Rules are written between zone pairs rather than between individual interfaces — far easier to reason about when many NICs / VLANs are present.

```bash
# firewalld (Red Hat / Fedora — zone-based wrapper over nftables)
firewall-cmd --get-zones
firewall-cmd --zone=public  --add-service=https --permanent
firewall-cmd --zone=public  --add-port=8080/tcp --permanent
firewall-cmd --zone=trusted --change-interface=eth1 --permanent
firewall-cmd --reload
```

```bash
# ufw (Debian / Ubuntu uncomplicated frontend)
ufw default deny incoming
ufw default allow outgoing
ufw allow from 10.0.0.0/8 to any port 22
ufw allow 443/tcp
ufw enable
```

Cisco ASA, Palo Alto, and Fortinet all expose policy in the same zone-pair form: `from-zone trust to-zone untrust { ... }`.

== eBPF / XDP Firewalls

XDP hooks into the NIC driver at interrupt time — packets are filtered before they enter the kernel networking stack, achieving line-rate drop performance.

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
    key hex c0 a8 01 05 \
    value hex 01

# Dump all blocked IPs
bpftool map dump pinned /sys/fs/bpf/blocklist

# XDP stats
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

== Performance Implications

*Per-packet processing cost:*

#table(
  columns: (auto, auto, auto),
  [*Operation*], [*Cycles*], [*Time \@ 3 GHz*],
  [Conntrack lookup (hash)], [200-500], [70-170 ns],
  [NAT translation], [100-200], [35-70 ns],
  [Filter rule matching (10 rules)], [500-1000], [170-330 ns],
  [Filter rule matching (100 rules)], [2000-5000], [700-1700 ns],
  [nftables set lookup], [100-200], [35-70 ns],
)

*Conntrack table contention:*
- Single global lock in older kernels (pre-4.7)
- Per-bucket locking in modern kernels (4.7+)
- At 1M+ connections, consider conntrack bypass for known-safe traffic

*Bypass strategies:*
```bash
# Skip conntrack for high-volume, stateless traffic (e.g., DNS resolver)
iptables -t raw -A PREROUTING -p udp --dport 53 -j NOTRACK
iptables -t raw -A OUTPUT     -p udp --sport 53 -j NOTRACK

# nftables equivalent
nft add rule inet raw prerouting udp dport 53 notrack
```

*Throughput benchmarks (10 Gbps NIC, 64B packets):*
- Without conntrack: ~14.8 Mpps (line rate)
- With conntrack: ~8-10 Mpps (~30% overhead)
- With 100 iptables rules: ~3-5 Mpps
- With nftables sets: ~10-12 Mpps

== Next-Generation Firewall (NGFW) Internals

*Application identification without decryption:*
- Port 443 may be HTTPS, Zoom, Netflix, SSH-over-HTTPS, or C2 malware
- TLS SNI (Server Name Indication): hostname visible in ClientHello plaintext (until ECH adoption)
- JA3 / JA4 fingerprint: hash of TLS parameters — identifies client library
- Flow byte-pattern ML classifier: trained on first 10 packets

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
NGFW acts as CA: issues a certificate signed by a trusted enterprise CA. Required: enterprise CA root cert installed on all managed endpoints. Incompatible with HPKP, certificate transparency monitors, and (notably) QUIC, which encrypts most of its handshake.

*IPS rule example (Suricata — CVE-2021-44228 Log4Shell):*
```
alert http any any -> $HTTP_SERVERS any (
    msg:"ET EXPLOIT Apache log4j RCE (Log4Shell)";
    flow:to_server,established;
    content:"$\{jndi:"; fast_pattern; nocase;
    reference:cve,2021-44228;
    classtype:attempted-admin; sid:2034647; rev:5;
)
```

== Cloud-Native Firewalling

*Linux nftables — production server baseline:*
```bash
#!/usr/sbin/nft -f
flush ruleset

table inet filter {
    set mgmt_hosts {
        type ipv4_addr
        elements = { 10.0.0.1, 10.0.0.2 }
    }

    chain input {
        type filter hook input priority 0; policy drop;

        iifname "lo" accept
        ct state { established, related } accept
        ct state invalid drop

        ip protocol icmp accept
        ip6 nexthdr icmpv6 accept

        tcp dport 22 ip saddr @mgmt_hosts accept
        tcp dport { 80, 443 } accept

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
cilium monitor --type drop
cilium endpoint list
cilium policy get
```

*AWS Security Groups* — stateful, evaluated at the hypervisor ENI:
```bash
aws ec2 authorize-security-group-ingress \
    --group-id sg-0abc123 --protocol tcp --port 443 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress \
    --group-id sg-0abc123 --protocol tcp --port 22 \
    --source-group sg-0bastion
```
Return traffic is automatically allowed (stateful) — no explicit egress rule needed.

=== Framework Comparison

#table(
  columns: (auto, auto, auto),
  [*Framework*], [*Rule Lookup*], [*Performance at Scale*],
  [iptables (linear)], [$O(n)$ per packet], [Degrades $>$5000 rules; ~10% throughput loss per 1000 rules],
  [iptables + ipset], [$O(1)$ hash / $O(log n)$ tree], [Handles 100K+ entries efficiently],
  [nftables (sets)], [$O(1)$ hash lookup], [Native set support, better than iptables at scale],
  [nftables (maps)], [$O(1)$ verdict maps], [Single rule replaces many; 2-5x fewer rules needed],
  [eBPF/XDP], [$O(1)$ hash maps], [Line-rate processing, bypass kernel stack],
)

_Modern deployments: prefer nftables over iptables. For $>$ 10 Gbps, consider XDP for stateless filtering._

== Further Reading

RFC 5382: NAT Behavioral Requirements for TCP. Guha et al., 2008.

RFC 6146: Stateful NAT64. Bagnulo et al., 2011.

Ayuso, P. (2006). "Netfilter's Connection Tracking System." `;login:` USENIX.

Welte, H. & Ayuso, P. "Netfilter / iptables Project." `netfilter.org`.

nftables Wiki. "nftables documentation." `wiki.nftables.org`.

Høiland-Jørgensen, T. et al. (2018). "The eXpress Data Path." CoNEXT.

Rosen, R. (2014). "Linux Kernel Networking: Implementation and Theory." Apress.

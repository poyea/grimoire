= SDN and Programmable Networks

Software-Defined Networking (SDN) decouples the control plane from the data plane, enabling centralized network management and programmable packet processing.

*See also:* Container Networking (for CNI implementations), Service Mesh (for application-layer routing), Kernel Bypass (for high-performance data planes)

== SDN Architecture

*Traditional networking:* Distributed control, each device makes independent forwarding decisions.

*SDN model [ONF TR-502]:*
- *Data plane:* Packet forwarding (switches, routers)
- *Control plane:* Routing decisions (centralized controller)
- *Management plane:* Configuration, monitoring (applications)

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│    (Network apps, orchestration, analytics)                  │
├─────────────────────────────────────────────────────────────┤
│                  Northbound API (REST, gRPC)                 │
├─────────────────────────────────────────────────────────────┤
│                     Control Layer                            │
│              (SDN Controller: ONOS, OpenDaylight)            │
├─────────────────────────────────────────────────────────────┤
│              Southbound API (OpenFlow, P4Runtime)            │
├─────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                       │
│        (OpenFlow switches, P4 switches, smart NICs)          │
└─────────────────────────────────────────────────────────────┘
```

*SDN benefits:*
- Centralized visibility: Global network view
- Programmability: Software-defined policies
- Vendor independence: Standard interfaces
- Rapid innovation: Decouple hardware from software

*SDN challenges:*
- Controller scalability: Single point of bottleneck
- Latency: Control plane RTT for first packet
- Reliability: Controller failure = network failure

== OpenFlow Protocol

*OpenFlow is the foundational SDN southbound protocol [ONF OpenFlow 1.5.1]:*

*Flow table structure:*

```
┌──────────────────────────────────────────────────────────────┐
│                        Flow Table                             │
├─────────────┬──────────┬───────────┬──────────┬─────────────┤
│ Match Fields│ Priority │  Counters │  Actions │   Timeouts  │
├─────────────┼──────────┼───────────┼──────────┼─────────────┤
│ src_ip=     │          │ packets:  │ forward  │ idle: 60s   │
│ 10.0.0.1    │   100    │   1523    │ port 2   │ hard: 300s  │
├─────────────┼──────────┼───────────┼──────────┼─────────────┤
│ dst_port=80 │    90    │   45231   │ forward  │ idle: 120s  │
│             │          │           │ port 3   │             │
├─────────────┼──────────┼───────────┼──────────┼─────────────┤
│ *           │     0    │   892     │ to_ctrl  │ none        │
│ (table-miss)│          │           │          │             │
└─────────────┴──────────┴───────────┴──────────┴─────────────┘
```

*Match fields (OpenFlow 1.5):*
- Layer 2: MAC src/dst, VLAN ID, Ethernet type
- Layer 3: IP src/dst, protocol, DSCP, TTL
- Layer 4: TCP/UDP src/dst ports, flags
- Tunnel: VXLAN VNI, GRE key, MPLS label

*Actions:*
- `OUTPUT`: Forward to port
- `DROP`: Discard packet
- `SET_FIELD`: Modify header fields
- `PUSH/POP`: Add/remove VLAN, MPLS tags
- `GROUP`: Apply group actions (multicast, ECMP)

*OpenFlow messages:*

```cpp
// Controller to switch
enum ofp_type {
    OFPT_HELLO           = 0,   // Version negotiation
    OFPT_FEATURES_REQUEST = 5,  // Query switch capabilities
    OFPT_FLOW_MOD        = 14,  // Add/modify/delete flows
    OFPT_PACKET_OUT      = 13,  // Send packet from controller
    OFPT_BARRIER_REQUEST = 20,  // Synchronization
};

// Switch to controller
enum ofp_type {
    OFPT_PACKET_IN       = 10,  // Unknown packet to controller
    OFPT_FLOW_REMOVED    = 11,  // Flow entry expired
    OFPT_PORT_STATUS     = 12,  // Port state change
    OFPT_STATS_REPLY     = 19,  // Statistics response
};
```

*Performance characteristics:*
- Flow setup latency: 1-10ms (controller RTT + processing)
- Table size: 1K-1M entries (TCAM limited)
- Throughput: Line rate when flows cached

== SDN Controllers

*Major open-source controllers:*

*1. ONOS (Open Network Operating System):*
- Distributed, clustered architecture
- Focus on carrier-grade deployments
- Java-based, OSGi modules

*2. OpenDaylight:*
- Model-driven (YANG data models)
- Extensive southbound protocol support
- Enterprise focus

*3. Ryu:*
- Python-based, lightweight
- Good for prototyping and research
- Single-threaded (limited scale)

*Controller scalability [Tootoonchian et al. 2012]:*

| Controller | Throughput (flows/s) | Latency (ms) |
|------------|---------------------|--------------|
| NOX        | 30,000              | 10-100       |
| Beacon     | 1,000,000+          | 2-5          |
| ONOS       | 500,000+ (cluster)  | 5-10         |

*High availability patterns:*

```
                 ┌─────────┐
                 │  App    │
                 └────┬────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
   │  Ctrl1  │───│  Ctrl2  │───│  Ctrl3  │  (Raft consensus)
   └────┬────┘   └────┬────┘   └────┬────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
              ┌───────┴───────┐
              │    Switches   │
              └───────────────┘
```

== P4: Protocol-Independent Packet Processors

*P4 enables programmable data planes [Bosshart et al. 2014]:*

*Key innovation:* Define packet parsing and processing in software, compile to hardware.

```p4
// P4_16 example: Simple L2 switch

// Header definitions
header ethernet_t {
    bit<48> dst_addr;
    bit<48> src_addr;
    bit<16> ether_type;
}

struct headers {
    ethernet_t ethernet;
}

// Parser definition
parser MyParser(packet_in packet, out headers hdr) {
    state start {
        packet.extract(hdr.ethernet);
        transition accept;
    }
}

// Match-Action pipeline
control MyIngress(inout headers hdr,
                  inout metadata meta,
                  inout standard_metadata_t std_meta) {

    action forward(bit<9> port) {
        std_meta.egress_spec = port;
    }

    action broadcast() {
        std_meta.mcast_grp = 1;
    }

    action drop() {
        mark_to_drop(std_meta);
    }

    table l2_forward {
        key = {
            hdr.ethernet.dst_addr: exact;
        }
        actions = {
            forward;
            broadcast;
            drop;
        }
        size = 1024;
        default_action = broadcast();
    }

    apply {
        l2_forward.apply();
    }
}
```

*P4 architecture:*

```
┌─────────────────────────────────────────────────────────────┐
│  P4 Program                                                  │
│  ┌─────────┐    ┌──────────────────┐    ┌─────────────────┐ │
│  │ Parser  │───▶│ Match-Action     │───▶│   Deparser      │ │
│  │         │    │ Pipeline         │    │                 │ │
│  └─────────┘    └──────────────────┘    └─────────────────┘ │
└────────────────────────┬────────────────────────────────────┘
                         │ Compile
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Target-specific code (ASIC, FPGA, software switch)         │
└─────────────────────────────────────────────────────────────┘
```

*P4 targets:*
- *Barefoot Tofino:* Programmable ASIC, 6.5 Tbps
- *Memory abstraction:* TCAM, SRAM, registers
- *bmv2:* Reference software switch
- *NetFPGA:* FPGA-based research platform
- *DPDK:* Software data plane

*P4Runtime API:*
- gRPC-based control plane interface
- Populate tables, read counters
- Stream digests to controller

```protobuf
// P4Runtime table entry
message TableEntry {
    uint32 table_id = 1;
    repeated FieldMatch match = 2;
    Action action = 3;
    int32 priority = 4;
    uint64 idle_timeout_ns = 5;
}
```

*Performance:*
- Tofino: 12.8 Tbps, 100ns latency
- bmv2: ~1 Mpps (software)
- Typical P4 program: 10-20 match-action stages

== eBPF for Networking

*eBPF (extended Berkeley Packet Filter) enables kernel-space programmability [Gregg 2019]:*

*Networking use cases:*
- XDP: Fast packet processing (see Kernel Bypass chapter)
- TC (Traffic Control): Packet classification and shaping
- Socket filtering: Application-level packet inspection
- Load balancing: Cilium, Katran

*TC eBPF example:*

```c
#include <linux/bpf.h>
#include <linux/pkt_cls.h>
#include <bpf/bpf_helpers.h>

// Rate limiting map
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10000);
    __type(key, __u32);   // Source IP
    __type(value, __u64); // Packet count
} rate_limit_map SEC(".maps");

SEC("tc")
int rate_limiter(struct __sk_buff *skb) {
    void *data_end = (void *)(long)skb->data_end;
    void *data = (void *)(long)skb->data;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return TC_ACT_SHOT;

    if (eth->h_proto != bpf_htons(ETH_P_IP))
        return TC_ACT_OK;

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return TC_ACT_SHOT;

    __u32 src_ip = ip->saddr;
    __u64 *count = bpf_map_lookup_elem(&rate_limit_map, &src_ip);

    if (count) {
        if (*count > 1000) {  // Rate limit threshold
            return TC_ACT_SHOT;
        }
        __sync_fetch_and_add(count, 1);
    } else {
        __u64 initial = 1;
        bpf_map_update_elem(&rate_limit_map, &src_ip, &initial, BPF_ANY);
    }

    return TC_ACT_OK;
}

char _license[] SEC("license") = "GPL";
```

*Attach to interface:*

```bash
# Compile BPF program
clang -O2 -target bpf -c rate_limiter.c -o rate_limiter.o

# Attach to TC ingress
tc qdisc add dev eth0 clsact
tc filter add dev eth0 ingress bpf da obj rate_limiter.o sec tc
```

*eBPF maps for networking:*

| Map Type | Use Case | Performance |
|----------|----------|-------------|
| HASH | Connection tracking | O(1) lookup |
| LRU_HASH | Flow cache | Auto-eviction |
| ARRAY | Per-CPU counters | Fastest |
| LPM_TRIE | Routing tables | Longest prefix match |
| DEVMAP | XDP redirect | Port forwarding |
| CPUMAP | XDP CPU steering | Load balancing |

*Cilium architecture (eBPF-based CNI):*

```
┌─────────────────────────────────────────────────────────────┐
│  Kubernetes                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │   Pod A  │  │   Pod B  │  │   Pod C  │                   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│       │             │             │                          │
│       └─────────────┼─────────────┘                          │
│                     │                                        │
│              ┌──────┴──────┐                                │
│              │   Cilium    │  eBPF programs:                │
│              │   Agent     │  - L3/L4 policy enforcement    │
│              │             │  - Load balancing (kube-proxy) │
│              │             │  - Network visibility          │
│              └─────────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

*Performance comparison:*

| Solution | Latency | Throughput | CPU |
|----------|---------|------------|-----|
| iptables | 5-10μs | 1M pps | High |
| Cilium (eBPF) | 1-2μs | 10M+ pps | Low |
| kube-proxy | 10-20μs | 500K pps | High |

== Network Function Virtualization (NFV)

*NFV moves network functions from hardware to software:*

*Traditional:* Dedicated hardware appliances (firewall, load balancer, IDS)
*NFV:* Software running on commodity servers

*ETSI NFV architecture:*

```
┌─────────────────────────────────────────────────────────────┐
│                          OSS/BSS                             │
├─────────────────────────────────────────────────────────────┤
│  NFV Orchestrator (NFVO)                                     │
│  - Service lifecycle management                              │
│  - Resource orchestration                                    │
├──────────────────────┬──────────────────────────────────────┤
│  VNF Manager (VNFM)  │  Virtualized Infrastructure Manager  │
│  - VNF lifecycle     │  (VIM)                               │
│  - Scaling           │  - Compute/storage/network resources │
├──────────────────────┴──────────────────────────────────────┤
│                 NFV Infrastructure (NFVI)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ Compute  │  │ Storage  │  │ Network  │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

*VNF examples:*
- Virtual firewall (pfSense, OPNsense)
- Virtual load balancer (HAProxy, nginx)
- Virtual router (VyOS, FRRouting)
- DPI (Suricata, Snort)

*Service chaining:*

```
Packet → vFirewall → vIDS → vLoadBalancer → Server
         (VNF1)     (VNF2)    (VNF3)
```

*Performance considerations:*
- DPDK for VNF data plane: 10-40 Gbps
- SR-IOV for VM networking: Near line rate
- Container-based VNFs: Faster startup (seconds vs minutes)

== Segment Routing and SRv6

*Segment Routing enables source routing in modern networks [RFC 8402]:*

*Key concepts:*
- *Segment:* Instruction (node, link, service)
- *Segment List:* Ordered list of segments (path)
- *SID (Segment ID):* Label (MPLS) or IPv6 address

*SRv6 (Segment Routing over IPv6):*

```
┌────────────────────────────────────────────────────────────────┐
│  IPv6 Header                                                    │
│  src: 2001:db8::1  dst: 2001:db8:1:: (first SID)              │
├────────────────────────────────────────────────────────────────┤
│  Segment Routing Header (SRH)                                   │
│  Segments Left: 2                                               │
│  Segment List: [2001:db8:3::, 2001:db8:2::, 2001:db8:1::]     │
├────────────────────────────────────────────────────────────────┤
│  Payload (TCP/UDP)                                              │
└────────────────────────────────────────────────────────────────┘
```

*SRv6 Network Programming [RFC 8986]:*

```
SID format: LOC:FUNCT:ARG
  - LOC: Locator (routing prefix)
  - FUNCT: Function (behavior)
  - ARG: Arguments

Functions:
  - End: Endpoint, update destination to next SID
  - End.X: Endpoint with cross-connect to neighbor
  - End.DT4: Decapsulate and route via IPv4 table
  - End.DT6: Decapsulate and route via IPv6 table
```

*Use cases:*
- Traffic engineering: Explicit path selection
- VPN: Service chaining through SIDs
- Fast reroute: Pre-computed backup paths

== References

*Primary sources:*

ONF TR-502: SDN Architecture. Open Networking Foundation (2014).

ONF OpenFlow Switch Specification 1.5.1. Open Networking Foundation (2015).

Bosshart, P. et al. (2014). "P4: Programming Protocol-Independent Packet Processors." ACM SIGCOMM CCR 44(3): 87-95.

RFC 8402: Segment Routing Architecture. Filsfils, C. et al. (2018).

RFC 8986: Segment Routing over IPv6 (SRv6) Network Programming. Filsfils, C. et al. (2021).

Tootoonchian, A. et al. (2012). "On Controller Performance in Software-Defined Networks." USENIX HotICE '12.

Gregg, B. (2019). BPF Performance Tools. Addison-Wesley.

Høiland-Jørgensen, T. et al. (2018). "The eXpress Data Path: Fast Programmable Packet Processing in the Operating System Kernel." CoNEXT '18.

McKeown, N. et al. (2008). "OpenFlow: Enabling Innovation in Campus Networks." ACM SIGCOMM CCR 38(2): 69-74.

Kim, H. & Feamster, N. (2013). "Improving Network Management with Software Defined Networking." IEEE Communications Magazine 51(2): 114-119.

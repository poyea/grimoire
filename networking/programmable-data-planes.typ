#import "../template.typ": xref

= Programmable Data Planes and P4

Programmable data planes extend SDN beyond OpenFlow's fixed match-action pipeline: the parser, match-action tables, and packet layout itself become software, compiled to a target ASIC, FPGA, NIC, or software switch. P4 [Bosshart et al. 2014] is the lingua franca; PISA (Protocol-Independent Switch Architecture) is the dominant abstract machine; P4Runtime [P4.org 2018] is the gRPC control API. The space is in flux after Intel exited the Tofino merchant ASIC business in early 2023, leaving Cisco Silicon One P200, Marvell Teralynx 10, AMD Pensando Elba, and Nvidia BlueField as the heirs.

*See also:* #xref("networking", "sdn-programmable-networks", label: "SDN and Programmable Networks") (for OpenFlow, controllers), #xref("networking", "kernel-bypass", label: "Kernel Bypass") (for XDP / AF-XDP software targets), #xref("networking", "data-center-networking", label: "Data Center Networking") (for in-network compute placement).

== Why Programmable Data Planes

Fixed-function ASICs ship a hard-coded pipeline (VLAN, MPLS, IPv4/IPv6, a few tunnels). Adding a new header or table layout means a multi-year silicon respin. The PISA insight is that the *match-action* substrate is general; only the headers and table widths differ across protocols. By exposing the pipeline as a programmable resource, operators can:

- Add new encapsulations (e.g., Geneve TLVs, SRv6 micro-SIDs) without new silicon.
- Implement custom telemetry (INT — In-band Network Telemetry).
- Run *in-network compute*: caching (NetCache), aggregation (SwitchML), consensus acceleration (NetPaxos), coordination (NetChain).
- Strip features they will never use, reclaiming TCAM and stages for what matters.

The cost is loss of vendor abstraction: a P4 program written for Tofino does not run unmodified on Silicon One; even register widths and stage counts differ.

== PISA: The Abstract Machine

PISA decomposes a switch into a programmable parser, an ingress match-action pipeline, a traffic manager (queues + replication), an egress pipeline, and a deparser:

```
   ┌──────────┐   ┌─────────────────────────┐   ┌───────┐   ┌─────────────────────────┐   ┌──────────┐
   │  Parser  │ → │  Ingress Match-Action   │ → │  TM   │ → │   Egress Match-Action   │ → │ Deparser │
   │ (state   │   │ stage 0 │ ... │ stage N │   │ queues│   │ stage 0 │ ... │ stage M │   │ (emit hdr)│
   │ machine) │   │ + ALUs, TCAM, SRAM, reg │   │ + repl│   │ + ALUs, TCAM, SRAM, reg │   │           │
   └──────────┘   └─────────────────────────┘   └───────┘   └─────────────────────────┘   └──────────┘
```

*Constraints that shape P4 programs:*
- *Per-stage memory:* each stage owns a fixed slice of TCAM (ternary, used for LPM and ACLs) and SRAM (exact-match, registers). You cannot read the *same* table twice in different stages without duplication.
- *Single-pass:* no loops, no recursion. Recirculation/resubmit are explicit and cost throughput.
- *ALU width:* typically 32 bits per ALU per stage. Wide ops (128-bit IPv6 compare) burn multiple ALUs.
- *Register access:* one read-modify-write per register array per packet (subject to target rules). This is what makes per-flow stateful logic hard.

If a program doesn't fit (stages, TCAM, action width), the compiler fails — there is no graceful degradation.

== P4 Language Fundamentals

P4 (current dialect: P4_16, ISO-style spec at p4.org) is a small, statically typed, target-parameterized language. A program consists of:

```
header_type  ───────► what bits look like on the wire
parser       ───────► a state machine that extracts headers
control      ───────► imperative code: tables, actions, externs
table        ───────► (key spec, action list, default action, size)
action       ───────► straight-line code that mutates headers / metadata
extern       ───────► target-supplied object: Register, Counter, Meter, Checksum, Hash
package      ───────► the top-level "main" that wires controls together
```

*Header and metadata declaration:*

```p4
header ethernet_h {
    bit<48> dst_mac;
    bit<48> src_mac;
    bit<16> ether_type;
}

header ipv4_h {
    bit<4>  version;
    bit<4>  ihl;
    bit<8>  diffserv;
    bit<16> total_len;
    bit<16> identification;
    bit<3>  flags;
    bit<13> frag_offset;
    bit<8>  ttl;
    bit<8>  protocol;
    bit<16> hdr_checksum;
    bit<32> src_addr;
    bit<32> dst_addr;
}

struct headers_t   { ethernet_h eth; ipv4_h ipv4; }
struct metadata_t  { bit<16> l4_lookup; bool  is_multicast; }
```

*Parser:* a state machine starting at `start`, ending at `accept` or `reject`:

```p4
parser MyParser(packet_in pkt, out headers_t hdr, inout metadata_t md,
                inout standard_metadata_t std) {
    state start {
        pkt.extract(hdr.eth);
        transition select(hdr.eth.ether_type) {
            0x0800: parse_ipv4;
            0x86DD: parse_ipv6;
            default: accept;
        }
    }
    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition accept;
    }
}
```

*Match-action table:*

```p4
action forward(bit<9> port) { std_meta.egress_spec = port; }
action drop()               { mark_to_drop(std_meta); }

table ipv4_lpm {
    key     = { hdr.ipv4.dst_addr : lpm; }
    actions = { forward; drop; NoAction; }
    size    = 16384;
    default_action = drop();
}

control Ingress(inout headers_t hdr, inout metadata_t md,
                inout standard_metadata_t std) {
    apply {
        if (hdr.ipv4.isValid()) ipv4_lpm.apply();
    }
}
```

*Match kinds:* `exact`, `lpm` (longest prefix), `ternary` (mask), `range` (specific targets), `selector` (action profile / ECMP). Targets restrict which kinds are usable per stage (e.g., LPM consumes TCAM).

*Externs* are target-specific objects with a P4-visible interface:

```p4
Register<bit<32>, bit<16>>(1024) flow_counter;     // 1024 x 32-bit cells
action bump(bit<16> idx) {
    bit<32> v; flow_counter.read(v, idx);
    flow_counter.write(idx, v + 1);
}
```

Hashes (`Hash<bit<16>>(HashAlgorithm.crc16)`), meters, counters, checksum units, and the digest mechanism (send small records to the control plane) are all externs. *Hash collisions* on flow keys are a constant practical concern for stateful designs.

== Compilation and Architectures

A P4 program targets a *PISA architecture*: a package signature that names the controls and externs the target supplies.

#table(
  columns: (auto, auto, auto),
  [*Architecture*], [*Target(s)*], [*Notes*],
  [`v1model`], [BMv2 simple_switch, many demos], [Original P4_14 model carried into P4_16],
  [`PSA` (Portable Switch Architecture)], [BMv2 psa_switch], [Vendor-neutral spec, limited adoption],
  [`TNA` (Tofino Native Arch)], [Intel/Barefoot Tofino 1/2/3], [Most production P4 deployed],
  [`T2NA`], [Tofino 2], [Adds new externs, more stages],
  [`PNA` (Portable NIC Architecture)], [SmartNICs / DPUs], [DASH on Pensando],
  [`ebpf_model`], [Linux eBPF/XDP via p4c-ebpf], [Subset; control plane via bpf maps],
  [`ubpf`], [Userspace eBPF VM], [DPDK softswitch],
)

```bash
# Reference compiler (open source)
p4c-bm2-ss -o myprog.json --p4runtime-files myprog.p4info.txt myprog.p4

# Run on BMv2 software switch
simple_switch -i 0@veth0 -i 1@veth2 --log-console myprog.json

# eBPF backend (XDP)
p4c-ebpf --target xdp -o myprog.c myprog.p4
clang -O2 -g -target bpf -c myprog.c -o myprog.o
ip link set dev eth0 xdp obj myprog.o sec xdp
```

The compiler emits two artifacts: the binary the dataplane loads, and a *P4Info* file describing the tables/actions/externs as named entities the control plane can address.

== P4Runtime: The Control API

P4Runtime [P4.org spec, current 1.4] is a gRPC service that lets a controller install/modify table entries, read counters, push digests, and re-load the program without per-vendor protobuf dialects.

```
service P4Runtime {
  rpc Write(WriteRequest) returns (WriteResponse);
  rpc Read (ReadRequest)  returns (stream ReadResponse);
  rpc SetForwardingPipelineConfig(...) returns (...);
  rpc GetForwardingPipelineConfig(...) returns (...);
  rpc StreamChannel(stream StreamMessageRequest)
    returns (stream StreamMessageResponse);   // master arbitration, digests, PacketIO
}
```

*Master arbitration:* multiple controllers connect; the highest `election_id` becomes master. Loss of mastership is signalled via the stream.

```python
# python p4runtime-shell snippet
import p4runtime_sh.shell as sh
sh.setup(device_id=0, grpc_addr='127.0.0.1:9559',
         election_id=(0, 1),
         config=sh.FwdPipeConfig('myprog.p4info.txt', 'myprog.json'))

te = sh.TableEntry('Ingress.ipv4_lpm')(action='Ingress.forward')
te.match['hdr.ipv4.dst_addr'] = '10.0.1.0/24'
te.action['port'] = '3'
te.insert()
```

*PacketIO* (`packet_in` / `packet_out`) tunnels exceptional packets through the gRPC stream, replacing OpenFlow's PACKET_IN. Latency is poor (ms range) — anything performance-critical stays in the dataplane.

P4Runtime is unopinionated about *topology, routing, or policy* — those live in the controller (ONOS Stratum, SD-Fabric, custom). The P4Info gives the controller table names and IDs; the controller is responsible for keeping its in-memory state consistent with the device.

== Hardware Targets after Tofino

Intel announced end-of-life for Tofino in January 2023. The programmable-ASIC market re-shuffled:

#table(
  columns: (auto, auto, auto, auto),
  [*Target*], [*Vendor*], [*Programmability*], [*Notes (2025-2026)*],
  [Tofino 1/2/3], [Intel/Barefoot (EoL)], [Native P4 (TNA)], [Large installed base at hyperscalers; community fork efforts],
  [Silicon One P200/G200], [Cisco], [P4-like SDK (private)], [51.2 Tbps; deep buffers; "P4-style" but not the open compiler],
  [Teralynx 10], [Marvell (ex-Innovium)], [P4 via Marvell SDK], [12.8/25.6 Tbps merchant; Microsoft/Meta deployments],
  [Pensando Elba (DPU)], [AMD], [P4 with PNA], [200 Gbps DPU; runs DASH SONiC stack],
  [BlueField-3 DPU], [Nvidia], [DOCA, eBPF, limited P4 via DOCA Flow], [Not a strict P4 device; flow programming via DOCA],
  [Trident 5/Tomahawk 6], [Broadcom], [NPL (Networking Programming Language)], [Broadcom's P4 alternative; closed],
  [Reconfigurable line cards], [Cisco/Arista/Juniper], [Microcode or NPU SDK], [Often expose a P4 subset],
)

The practical effect: *P4-the-language* survives, but *P4-the-portable-ecosystem* has fractured. Production deployments increasingly accept vendor lock-in or fall back to NPL or vendor SDKs. Open targets (BMv2, eBPF, XDP, DPDK softswitches) dominate research and CI.

*Software targets* worth knowing:
- *BMv2 (`simple_switch`, `simple_switch_grpc`):* the reference. Slow (~Gb/s), but the contract for correctness.
- *p4c-ebpf / XDP:* compiles a P4 subset (no recirculation, simple externs) to BPF. Real line-rate on commodity NICs.
- *T4P4S / DPDK:* P4 to DPDK userspace. 10-40 Gbps per core.
- *PISCES / Open vSwitch:* OVS with a P4 frontend; production-deployable for virtual switching.

== In-Network Compute Applications

Programmable data planes opened a research wave of "compute on the switch." Three load-bearing exemplars:

*NetCache* [Jin et al., SOSP 2017]: a Tofino switch caches the hottest items of a key-value store. The switch parses application headers, looks up cached keys in stages, and serves reads in the dataplane. Result: a single ToR fronting 128 storage servers reaches 2 billion QPS for skewed workloads. Key trick: use the dataplane to identify hot keys (count-min sketch in registers) and a small control loop to swap cache contents.

*SwitchML* [Sapio et al., NSDI 2021]: in-network aggregation for distributed ML. Workers send gradient chunks tagged with an index; the switch sums them in registers and multicasts the result, halving cross-rack bandwidth versus a parameter server. Constraint-driven design — fixed-point quantization (no FP in the dataplane), per-chunk slot reservation, retransmit on packet loss.

*NetChain* [Jin et al., NSDI 2018]: chain-replicated coordination service running entirely in switches. Sub-RTT operations (10s of μs) replacing Paxos/ZooKeeper for lock service workloads.

*INT (In-band Network Telemetry):* every switch on the path appends per-hop metadata (queue depth, hop latency, ingress timestamp) into the packet. End-host or collector reconstructs precise per-flow path telemetry. INT-MD (metadata in packet), INT-MX (export only) variants. Deployed at Alibaba, Google.

*Limits of in-network compute:* no floating point, tiny per-flow state (kilobits, not megabits), no general-purpose loops, recirculation is expensive. Designs that work tend to be *streaming aggregations over fixed-size windows* or *cached approximations of an external service*.

== Putting It Together: A Tiny INT Example

```p4
header int_md_h {
    bit<8>  hop_count;
    bit<8>  switch_id;
    bit<24> queue_depth;
    bit<32> ingress_ts;
}

control IngressINT(inout headers_t hdr, inout metadata_t md,
                   inout standard_metadata_t std) {
    apply {
        if (hdr.int_md.isValid()) {
            hdr.int_md.hop_count    = hdr.int_md.hop_count + 1;
            hdr.int_md.switch_id    = MY_SWITCH_ID;
            hdr.int_md.queue_depth  = (bit<24>) std.enq_qdepth;
            hdr.int_md.ingress_ts   = (bit<32>) std.ingress_global_timestamp;
        }
    }
}
```

This 6-line action would have required new silicon in the pre-P4 era. With P4 it's a recompile.

== Operational Pitfalls

- *Table resource exhaustion:* TCAM is precious. A naive 5-tuple ACL with `/24` masks blows out the table fast. Use stage-shared SRAM (exact match) and hash-based prefix expansion where possible.
- *Hash collisions in stateful tables:* register arrays indexed by hash will collide; designs need cuckoo-hash, d-left, or a "reject and punt to CPU" path.
- *Pipeline reload disruption:* `SetForwardingPipelineConfig` on Tofino traditionally drops traffic for seconds. Hot-swap support varies by target.
- *Control-plane drift:* P4Runtime writes are *not transactional across multiple devices*. Treat the controller-to-switch link as an eventually-consistent system.
- *Debuggability:* once a program is running on hardware, there is no `gdb`. Use BMv2 first; on hardware rely on counter externs, INT, and mirror-to-CPU for the slow path.

== Further Reading

*Foundational:*

Bosshart, P., Daly, D., Gibb, G., Izzard, M., McKeown, N., Rexford, J., Schlesinger, C., Talayco, D., Vahdat, A., Varghese, G. & Walker, D. (2014). "P4: Programming Protocol-Independent Packet Processors." _ACM SIGCOMM Computer Communication Review_, 44(3).

Bosshart, P., Gibb, G., Kim, H.-S., Varghese, G., McKeown, N., Izzard, M., Mujica, F. & Horowitz, M. (2013). "Forwarding Metamorphosis: Fast Programmable Match-Action Processing in Hardware for SDN." _SIGCOMM '13_.

P4 Language Consortium (2024). _P4_16 Language Specification, version 1.2.5_. p4.org.

P4 Language Consortium (2024). _P4Runtime Specification, version 1.4.0_. p4.org.

*In-network compute:*

Jin, X., Li, X., Zhang, H., Soulé, R., Lee, J., Foster, N., Kim, C. & Stoica, I. (2017). "NetCache: Balancing Key-Value Stores with Fast In-Network Caching." _SOSP '17_.

Sapio, A., Canini, M., Ho, C.-Y., Nelson, J., Kalnis, P., Kim, C., Krishnamurthy, A., Moshref, M., Ports, D. & Richtárik, P. (2021). "Scaling Distributed Machine Learning with In-Network Aggregation." _NSDI '21_.

Jin, X., Li, X., Zhang, H., Foster, N., Lee, J., Soulé, R., Kim, C. & Stoica, I. (2018). "NetChain: Scale-Free Sub-RTT Coordination." _NSDI '18_.

Dang, H. T., Sciascia, D., Canini, M., Pedone, F. & Soulé, R. (2015). "NetPaxos: Consensus at Network Speed." _SOSR '15_.

*Telemetry and verification:*

Kim, C., Sivaraman, A., Katta, N., Bas, A., Dixit, A. & Wobker, L. (2015). "In-band Network Telemetry via Programmable Dataplanes." _SIGCOMM Industrial Demo_.

Liu, J., Hallahan, W., Schlesinger, C., Sharif, M., Lee, J., Soulé, R., Wang, H., Cascaval, C., McKeown, N. & Foster, N. (2018). "p4v: Practical Verification for Programmable Data Planes." _SIGCOMM '18_.

*Targets and industry context:*

Intel Corporation (2023). "End-of-Life Notification for Intel Tofino Switch Silicon." Customer notification, January 2023.

Cisco Systems (2024). _Silicon One P200 Product Brief_.

Marvell Technology (2024). _Teralynx 10 Programmable Switch_.

AMD (2023). _AMD Pensando Elba DPU_.

Open Networking Foundation (2020). _Stratum: Enabling the Era of Next-Generation SDN_.

Hauser, F., Häberle, M., Merling, D., Lindner, S., Gurevich, V., Zeiger, F., Frank, R. & Menth, M. (2023). "A Survey on Data Plane Programming with P4: Fundamentals, Advances, and Applied Research." _Journal of Network and Computer Applications_, 212.

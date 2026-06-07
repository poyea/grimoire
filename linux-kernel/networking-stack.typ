= The Networking Stack

The Linux networking stack handles ~30 M pps per core on commodity hardware in 2025, runs on every device with an IP address from a Raspberry Pi to a 400 GbE switch ASIC, and is the substrate beneath every container runtime's networking model. Its architecture is layered but porous: drivers, the NAPI poll loop, the IP/TCP/UDP code in `net/ipv4` and `net/ipv6`, traffic-control queueing disciplines, netfilter hooks, and the eBPF programmable data plane (XDP, TC, sk_lookup) all interlock through a single data structure: the `struct sk_buff`.

== sk_buff: The Packet Object

A `sk_buff` (or *skb*) describes one packet plus metadata. It is comparatively large (~232 bytes head, plus a shared `skb_shared_info` trailing the data area) and famously costly to allocate — every packet allocates and frees an skb (or recycles one from a per-CPU cache). Cutting skb allocation cost (`skb_recycle`, page-pool, page-fragment allocators) has been a recurring optimization theme.

```c
// include/linux/skbuff.h (heavily simplified)
struct sk_buff {
    struct sk_buff       *next, *prev;     // queue links
    struct net_device    *dev;
    unsigned char        *head, *data;     // buffer start, payload start
    unsigned char        *tail, *end;      // payload end, buffer end
    unsigned int          len;             // total length incl. frags
    unsigned int          data_len;        // bytes in frags (not in linear area)
    __be16                protocol;
    __u16                 transport_header; // offset from head
    __u16                 network_header;
    __u16                 mac_header;
    struct sock          *sk;              // owning socket (for tx)
    ktime_t               tstamp;
    union { __u32 mark; __u32 reserved_tailroom; };
    __u32                 hash;            // RSS hash from NIC
    __u16                 queue_mapping;   // tx queue
    refcount_t            users;
};
```

`head/data/tail/end` form a four-pointer window: `head..end` is the allocated buffer; `data..tail` is the current payload. As a packet travels up the stack, `skb_pull` advances `data` past each header; on egress `skb_push` retreats it as headers are prepended. *Headroom* (`data - head`) lets layers prepend without copying; drivers allocate enough headroom (`NET_SKB_PAD`) to fit the IP and L2 headers.

The `skb_shared_info` at `end` holds page fragments — the *non-linear* part. A 64 KiB GSO super-segment lives mostly here, with one `skb_frag_t` per page. `skb_clone` produces a new skb sharing the data area (reference-counted); `skb_copy` deep-copies.

== Rx Path: From NIC to Socket

The receive path runs in two contexts: hard IRQ (very short) and softirq / NAPI (the workhorse). See _Interrupts and NAPI_ for the IRQ side. The condensed flow:

1. NIC DMAs frame into a pre-posted rx ring buffer, raises MSI-X.
2. Driver IRQ handler schedules NAPI (`napi_schedule`) and returns.
3. `NET_RX_SOFTIRQ` runs, calls driver `poll()`, which builds skbs from the rx ring and hands them to `napi_gro_receive`.
4. *GRO* (Generic Receive Offload) coalesces adjacent TCP segments into a super-skb (up to 64 KiB), which is the stack's single biggest CPU win on receive.
5. `__netif_receive_skb_core` dispatches by `protocol` to `ip_rcv` / `ipv6_rcv` / `arp_rcv` ... and runs netfilter `PRE_ROUTING` hooks.
6. Routing decision: deliver locally or forward.
7. Local delivery walks transport handlers (`tcp_v4_rcv`, `udp_rcv`); netfilter `LOCAL_IN`. Socket lookup (`__inet_lookup_*`) finds the destination `struct sock`; payload is queued on `sk->sk_receive_queue` (or `sk_backlog` if the socket is locked).
8. User process wakes up from `epoll_wait`/`recvmsg`/io_uring completion.

GRO is implemented per-protocol via `gro_receive` and `gro_complete` callbacks; TCP, UDP-L4 (with GRO-on-UDP since 5.0, essential for QUIC), GENEVE, and VXLAN all participate.

== Tx Path

Send is the symmetric dance:

1. `sendmsg` / io_uring `IORING_OP_SEND` enters `tcp_sendmsg` or `udp_sendmsg`.
2. Data is copied into skb pages (or zero-copied via `MSG_ZEROCOPY` since 4.14, holding the user pages with a notification on completion).
3. TCP segmentation: if *TSO/GSO* is on, build one large super-segment and let the NIC (TSO) or `tcp_gso_segment` (software GSO) split it into MTU-sized packets just before transmission.
4. Routing → `ip_output` → netfilter `LOCAL_OUT`/`POST_ROUTING` → neighbor (ARP/ND) → `dev_queue_xmit`.
5. `dev_queue_xmit` selects a tx queue (`netdev_pick_tx`, using `skb->queue_mapping` or RSS/XPS), enqueues into the qdisc.
6. Qdisc dequeues; driver's `ndo_start_xmit` posts to the tx ring; NIC DMAs and transmits; completion IRQ frees the skb (or transfers it back to a pool).

*GSO* (Generic Segmentation Offload) plus *TSO* (TCP Segmentation Offload) push segmentation to the latest possible point. A `sendmsg(64K)` traverses TCP and IP exactly once instead of 44 times, which is the difference between 10 Gbps and 40 Gbps on the same core.

== Queueing Disciplines (qdiscs)

Between the protocol stack and the driver lives the *qdisc*, the kernel's traffic shaper, scheduler, and prioritizer. Per-device, configured via `tc` (`iproute2`):

#table(columns: (auto, 1fr),
  [`pfifo_fast`], [Three-band FIFO based on ToS; the legacy default.],
  [`mq` / `mq-prio`], [Multi-queue wrapper: per-tx-queue child qdiscs. Default on modern multi-queue NICs.],
  [`fq` / `fq_codel`], [Per-flow fair queueing; `fq_codel` adds CoDel AQM. Required by TCP BBR's pacing; default on many distros.],
  [`cake`], [Modernized `fq_codel`+`htb`+DRR with traffic-class awareness. Used by router projects.],
  [`htb`], [Hierarchical Token Bucket — classic class-based shaping with borrowing.],
  [`tbf`], [Simple token-bucket rate cap.],
  [`netem`], [Network emulator: latency, jitter, loss, reorder. The test-fixture qdisc.],
)

```bash
# Replace default qdisc with fq, enable BBR
sysctl -w net.core.default_qdisc=fq
sysctl -w net.ipv4.tcp_congestion_control=bbr

# Per-flow fair queueing with pacing on eth0
tc qdisc replace dev eth0 root fq pacing
```

A qdisc must be lock-safe (one CPU dequeues, multiple may enqueue). Most modern qdiscs use per-CPU lock-free enqueue paths.

== Traffic Control: Filters and Actions

`tc filter` + `tc action` on the egress (`root`) or ingress (`clsact ingress`) qdisc lets the kernel classify, mangle, redirect, or police packets. Classifier types include `u32` (byte-pattern), `flower` (5-tuple + more), and `bpf` (eBPF program). Actions include `mirred` (mirror/redirect to another interface), `nat`, `pedit` (packet edit), and `gact` (drop/pass/reclassify).

```bash
# Add a clsact qdisc on eth0 for both ingress and egress eBPF
tc qdisc add dev eth0 clsact

# Attach an eBPF program at ingress
tc filter add dev eth0 ingress bpf da obj prog.o sec ingress
```

This is the second-most-common attach point for eBPF networking (after XDP). The TC layer sees packets with full skb metadata, can modify them, and integrates with netfilter, making it the sweet spot for container networking (Cilium attaches both at TC and XDP).

== XDP: Express Data Path

*XDP* runs eBPF programs at the earliest possible point: in the NIC driver's poll routine, before any skb has been allocated. The verdict is one of:

- `XDP_PASS` — proceed up the stack (skb is built normally).
- `XDP_DROP` — free the rx descriptor (~30 M pps drop rate per core, the basis of DDoS protection).
- `XDP_TX` — bounce out the same NIC.
- `XDP_REDIRECT` — forward to another interface, CPU, or `AF_XDP` socket via a `bpf_redirect_map`.
- `XDP_ABORTED` — error path; emits a tracepoint.

```c
SEC("xdp")
int drop_udp_53(struct xdp_md *ctx)
{
    void *data     = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end) return XDP_PASS;
    if (eth->h_proto != bpf_htons(ETH_P_IP)) return XDP_PASS;
    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end) return XDP_PASS;
    if (ip->protocol != IPPROTO_UDP) return XDP_PASS;
    struct udphdr *udp = (void *)ip + ip->ihl * 4;
    if ((void *)(udp + 1) > data_end) return XDP_PASS;
    if (udp->dest == bpf_htons(53)) return XDP_DROP;
    return XDP_PASS;
}
```

Three modes:

- *Native XDP*: driver builds an `xdp_buff` from the rx descriptor and invokes the program. Lowest overhead; requires driver support.
- *Generic XDP*: runs after skb allocation (`netif_receive_skb`). Works on any device at ~half the throughput of native.
- *Offloaded XDP*: program JITs into NIC firmware (Netronome). Niche.

`AF_XDP` is the userspace zero-copy sibling: an XDP program redirects a frame into a userspace ring (`UMEM`), and a user process processes it without any kernel TCP/IP involvement at all. This is the polite alternative to DPDK. See _Kernel Bypass_ in the networking volume.

== Netfilter and nftables

Netfilter hooks live at five points: `PRE_ROUTING`, `LOCAL_IN`, `FORWARD`, `LOCAL_OUT`, `POST_ROUTING`. iptables (legacy) and nftables (modern, since 3.13) register at these hooks. nftables uses a single virtual machine (`net/netfilter/nf_tables_core.c`) with set-based matches — far faster than iptables' linear chains for large rulesets.

Connection tracking (`nf_conntrack`) records 5-tuples and state for stateful firewalling and NAT. It is a CPU and memory hog at high pps; `nf_conntrack_max`, `nf_conntrack_buckets`, and the `notrack` rule on hot flows are the standard tuning knobs.

== Socket Layer and BPF Hooks

The socket layer (`net/socket.c`, `net/core/sock.c`) sits on top of the transport protocols. `struct sock` is the protocol-agnostic base; `tcp_sock`, `udp_sock`, `inet_sock` extend it.

eBPF program types that hook the socket layer:

#table(columns: (auto, 1fr),
  [`BPF_PROG_TYPE_XDP`], [Earliest rx hook, no skb.],
  [`BPF_PROG_TYPE_SCHED_CLS`], [TC ingress/egress with full skb.],
  [`BPF_PROG_TYPE_SCHED_ACT`], [TC action, complements `SCHED_CLS`.],
  [`BPF_PROG_TYPE_SOCK_OPS`], [Per-connection TCP state-machine callbacks; used to tune RTO, ECN, and congestion-control selection per flow.],
  [`BPF_PROG_TYPE_SK_SKB`], [Stream parser/verdict for sockmap — splice packets between sockets in-kernel.],
  [`BPF_PROG_TYPE_SK_MSG`], [`sendmsg`-time program; peek at user payload and redirect to another socket via sockmap.],
  [`BPF_PROG_TYPE_SOCK_FILTER`], [Classic seccomp/SO_ATTACH_BPF socket filter.],
  [`BPF_PROG_TYPE_CGROUP_SOCK*`], [Per-cgroup `connect`/`bind`/`sendmsg`/`recvmsg` interception for service-mesh transparent proxy without iptables.],
  [`BPF_PROG_TYPE_SK_LOOKUP`], [Override socket-lookup decisions to build custom listener pools (e.g., SO_REUSEPORT++).],
)

Cilium, Katran, Calico-eBPF, and modern service meshes are built almost entirely from this menu — XDP for L4 load balancing, TC for ingress policy, `SOCK_OPS` for TCP tuning, `CGROUP_SOCK_ADDR` for transparent service redirection.

== TCP: Congestion Control and Pacing

`net/ipv4/tcp_*.c` is among the kernel's most heavily optimized code. Pluggable congestion control is registered via `tcp_register_congestion_control`:

#table(columns: (auto, 1fr),
  [`reno`], [Textbook AIMD.],
  [`cubic`], [Long-time default; cubic window growth, RTT-fair.],
  [`bbr` v1-v3], [Bandwidth × RTT model, paces sends, ignores loss. Requires `fq` qdisc.],
  [`dctcp`], [Datacenter TCP — ECN-driven; tiny queues.],
  [`bbr-prague`], [L4S-aware variant.],
)

Selection: `sysctl net.ipv4.tcp_congestion_control=bbr` (or per-route via `ip route ... congctl bbr`, or per-socket via `setsockopt TCP_CONGESTION`).

TSQ (TCP Small Queues) caps the bytes in transit per socket inside the kernel's tx path, preventing bufferbloat in the qdisc layer. Pacing (with `fq`) replaces the burst behaviour of ACK-clocked sends; it is essential for BBR and beneficial in general.

== UDP, QUIC, and GRO-on-UDP

UDP was historically just "send-recv with checksums", but the rise of QUIC turned UDP into a hot path. The kernel added:

- *GRO over UDP* — coalesces consecutive UDP packets with the same 5-tuple, dramatically reducing per-packet cost on the QUIC receive path.
- *GSO over UDP* (`UDP_SEGMENT`) — `sendmsg` of a 64 KiB buffer with `cmsg(UDP_SEGMENT, gso_size)` produces many MTU-sized packets in one call.
- *`SO_REUSEPORT` BPF dispatcher*: for a QUIC server, route incoming packets to the worker that owns the connection ID.

This brings userspace QUIC implementations within ~80% of TCP-in-kernel throughput on the same hardware.

== Performance Numbers

Rough, modern Linux (6.x), single core, 100 GbE NIC with multi-queue + RSS:

- *XDP_DROP*: ~30 Mpps/core (line-rate of 25-40 GbE small packets).
- *XDP_TX (L2 reflect)*: ~25 Mpps/core.
- *Forwarding via kernel route table*: ~3-5 Mpps/core.
- *netfilter conntrack on*: drops to ~1-2 Mpps/core.
- *TCP throughput (64 KiB MSS, single flow)*: 40-90 Gbps with GRO/TSO on (single core).
- *MSG_ZEROCOPY*: ~30% CPU saving on send for >16 KiB messages.

The takeaway: drop the right packets in XDP, do real work in TC, terminate connections with TCP, and reach for AF_XDP / DPDK only when you must.

== Observability

```bash
# Per-queue rx/tx stats
ethtool -S eth0 | grep -E 'rx_|tx_'

# qdisc stats
tc -s qdisc show dev eth0

# Live socket table (replaces netstat)
ss -tninp

# TCP retransmits per second
bpftrace -e 'tracepoint:tcp:tcp_retransmit_skb { @ = count(); }'

# Where are packets being dropped?
bpftrace -e 'tracepoint:skb:kfree_skb { @[kstack] = count(); }' | head -40
```

The `drop_monitor` netlink interface and `perf record -e skb:kfree_skb` are the canonical drop-localization tools.

== Container Networking

Container networking is a story told entirely in this chapter's primitives:

- *veth pair* in a network namespace, peer in the host.
- *bridge* (Linux bridge or OVS) for L2 fan-out, or a *routed* model with per-pod /32 routes.
- *iptables* / *nftables* / *eBPF TC* for policy.
- *VXLAN* / *Geneve* / *WireGuard* for overlays.
- *XDP* or *IPVS* for service load balancing.

Cilium replaces iptables-based kube-proxy with XDP+TC eBPF programs and a per-CPU map for service backends — sub-microsecond service-IP rewriting.

== Further Reading

Kernel docs: `Documentation/networking/` (especially `napi.rst`, `xdp.rst`, `gen-stats.rst`, `bpf-sockmap.rst`).

Cardwell, N. et al. (2017). _BBR: Congestion-Based Congestion Control_, CACM.

Høiland-Jørgensen, T. et al. (2018). _The eXpress Data Path: Fast Programmable Packet Processing in the Operating System Kernel_, CoNEXT.

Brunella, M. et al. (2020). _hXDP: Efficient Software Packet Processing on FPGA NICs_, OSDI.

LWN: Corbet's GRO, GSO, BBR, and XDP series; "A look at sockmap" (2018).

`net/core/dev.c`, `net/ipv4/tcp_*.c`, `net/sched/`, `net/xdp/`, `drivers/net/ethernet/*/`.

Hsieh, J. et al. (2024). _Cilium: Cloud-Native Networking and Security with eBPF_, O'Reilly.

*See also:* _Interrupts and NAPI_ (rx softirq path, IRQ affinity), _eBPF Deep Dive_ (the VM these XDP/TC programs target), _Cgroups and Namespaces_ (network namespaces and per-cgroup socket hooks), _IO uring_ (modern sendmsg/recvmsg path).

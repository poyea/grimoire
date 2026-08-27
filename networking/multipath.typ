#import "../template.typ": rfc, xref

= Multipath Transport <multipath>

Multipath transports (MPTCP, MPQUIC, and SCTP-CMT) use several network paths simultaneously between a single pair of endpoints, improving throughput, resilience, and seamless handover between heterogeneous links (Wi-Fi + LTE, dual-uplink data centres). This chapter focuses on MPTCP, the most widely deployed multipath transport, with a survey of MPQUIC.

*See also:* #xref("networking", "transport-layer", label: "Transport Layer") (single-path TCP), #xref("networking", "congestion-control", label: "Congestion Control") (coupled CC algorithms), #xref("networking", "quic-and-http3", label: "QUIC and HTTP/3") (MPQUIC builds on QUIC), #xref("networking", "wireless-protocols", label: "Wireless Protocols") (heterogeneous radio handover).

== Motivation

A modern smartphone has at least two network interfaces (Wi-Fi and cellular); a server in a Clos data centre typically has two or four uplinks; a sensor on a vehicle may roam between 4G, 5G, and satellite. Single-path TCP cannot exploit these in parallel and cannot survive a primary interface going down without breaking the connection.

Multipath transports add a layer above per-path congestion control:

```
   Application
        │   (single byte-stream API)
   ┌────┴────┐
   │  MPTCP  │  ← schedules data across subflows, reorders on receive
   └─┬─────┬─┘
     │     │
  TCP-A  TCP-B   ← each subflow is a regular TCP connection
     │     │
  Wi-Fi  LTE
```

Goals (#rfc(6182), the MPTCP architecture document):
+ *Throughput:* aggregate $≥$ any single path.
+ *Robustness:* survive failure of $n-1$ paths.
+ *Fairness:* do not steal bandwidth from single-path TCP sharing any bottleneck.

The third goal makes coupled congestion control mandatory and is where most algorithmic complexity lives.

== MPTCP Architecture

*Sub-flow.* Each MPTCP connection is composed of one or more *subflows*, each a real TCP connection with its own 4-tuple, sequence space, and ACKs. Subflows are added or removed dynamically.

*Connection-level vs subflow-level sequencing.* MPTCP introduces a 64-bit *Data Sequence Number* (DSN) on top of the per-subflow TCP sequence numbers. The receiver reassembles the byte stream by DSN; the sender retransmits at the DSN level if a subflow disappears.

*Components:*

+ *Path Manager*: discovers, advertises, and tears down subflows. Linux `mptcpd` user-space daemon plus kernel API; Apple has its own per-app policies.
+ *Scheduler*: decides which subflow carries each segment.
+ *Congestion Coupling*: links per-subflow congestion windows so total throughput is fair to single-path TCP.

== Handshake and Wire Format

MPTCP signalling is carried in TCP options. The kernel adds the `MP_CAPABLE` option on the initial SYN; if the peer supports it, both endpoints exchange 64-bit tokens used to authenticate later subflow joins.

```
Client                                Server
  │ SYN  + MP_CAPABLE(key_c)            │
  ├────────────────────────────────────▶│
  │ SYN/ACK + MP_CAPABLE(key_s)         │
  │◀────────────────────────────────────┤
  │ ACK  + MP_CAPABLE(key_c, key_s)     │
  ├────────────────────────────────────▶│
  │                                     │
  │ ──── primary subflow established ──│
  │                                     │
  │ on second interface coming up:      │
  │ SYN + MP_JOIN(token, HMAC)          │
  ├────────────────────────────────────▶│
  │ SYN/ACK + MP_JOIN(HMAC)             │
  │◀────────────────────────────────────┤
  │ ACK                                 │
  ├────────────────────────────────────▶│
  │ ──── second subflow established ────│
```

*Fallback:* if any middlebox strips the option, the connection silently degrades to single-path TCP without aborting.

== Schedulers

The scheduler picks a subflow for every outgoing segment. The Linux kernel ships several:

#table(
  columns: (auto, auto),
  [*Scheduler*], [*Behaviour*],
  [`default` (lowest-RTT)], [Send on the subflow with the smallest smoothed $"RTT"$ that has available cwnd. Good general-purpose choice.],
  [`roundrobin`], [Strict alternation. Simple but ignores heterogeneous latency.],
  [`redundant`], [Send the *same* segment on every subflow. Trades bandwidth for tail-latency reduction. Used by some financial services.],
  [`BLEST`], [Blocking Estimation: avoids using slow paths if they would block fast-path reordering buffer.],
  [`ECF`], [Earliest Completion First: picks the path expected to deliver soonest, accounting for cwnd remaining and RTT.],
)

```bash
# Inspect available schedulers (Linux 5.6+ kernel MPTCP)
ls /proc/sys/net/mptcp/scheduler

# Select a scheduler
sysctl -w net.mptcp.scheduler=blest
```

== Coupled Congestion Control

If each subflow ran independent NewReno, an MPTCP flow with two paths sharing a single bottleneck would receive twice the bandwidth of a single-path TCP flow, which is unfair. Coupled CC sums (or partially sums) the windows so the aggregate behaves fairly.

=== LIA — Linked Increases Algorithm (#rfc(6356))

For each subflow $r$, on ACK:
$ w_r ← w_r + min(alpha / w_"total", 1 / w_r) $
where $w_"total" = sum_(s in "subflows") w_s$ and $alpha$ is computed so the total throughput equals what a single-path TCP would get on the best path. LIA was the original deployed coupling.

*Weakness:* not Pareto-optimal; may underutilise paths that are not bottlenecked together.

=== OLIA — Opportunistic Linked Increases (Khalili et al., CoNEXT 2012)

OLIA fixes LIA's non-optimality. It partitions subflows into "best" and "max-window" sets, distributing window growth so that idle capacity on lightly loaded paths is used:

$ w_r ← w_r + alpha_r / w_r $
with $alpha_r$ chosen via a linear program over per-path losses and RTTs. Default in Linux MPTCP prior to upstream merge; still selectable.

=== BALIA — Balanced Linked Adaptation (Walid et al., IEEE/ACM ToN 2016)

BALIA aims to balance responsiveness (LIA) and Pareto-optimality (OLIA) with a unified rule. It is the current default in many implementations:

$ w_r ← w_r + (x_r / "RTT"_r) (1 + alpha_r) (4 + alpha_r) / 5 $
where $x_r$ is the per-path throughput estimate. Empirically smoother under sudden path change than OLIA.

```bash
# Linux: choose congestion coupling
sysctl -w net.mptcp.congestion_control=balia
# Other options: lia, olia, cubic (uncoupled, for testing)
```

== Linux Upstream MPTCP

Linux gained upstream MPTCP support in kernel 5.6 (March 2020), replacing the long-lived out-of-tree MultiPath TCP Linux project. Status check and a minimal client:

```bash
# Is MPTCP enabled?
sysctl net.mptcp.enabled
# net.mptcp.enabled = 1

# Per-route endpoint configuration via iproute2
ip mptcp endpoint add 192.0.2.5 dev wlan0 subflow signal
ip mptcp endpoint add 198.51.100.7 dev wwan0 subflow backup

ip mptcp endpoint show
# 192.0.2.5 id 1 subflow signal dev wlan0
# 198.51.100.7 id 2 subflow backup dev wwan0
```

```c
// MPTCP socket — single new IPPROTO value
int fd = socket(AF_INET, SOCK_STREAM, IPPROTO_MPTCP);
// All other socket APIs (connect, send, recv) unchanged.
```

```bash
# Observe subflows with `ss`
ss -tmM
# State  Recv-Q Send-Q Local Address:Port   Peer Address:Port
# ESTAB  0      0      192.0.2.5:54321      203.0.113.1:443
#        skmem(...) mptcp flags:Mmec_ token:abcd1234 ...
#        subflows: 198.51.100.7:60000 - 203.0.113.1:443
```

== Apple's MPTCP Deployment

Apple was the first hyperscaler to deploy MPTCP at scale: Siri on iOS 7 (2013) used MPTCP between iPhones and Apple's voice-recognition servers, surviving Wi-Fi → LTE handover transparently. Subsequent expansions:
- Apple Maps tile fetching (iOS 11+).
- Apple Music streaming.
- iOS 14+: the public `URLSession.multipathServiceType` API exposes three modes:
  - `none`: single path
  - `handover`: secondary used only when primary fails
  - `interactive`: favour low latency, may use both paths concurrently
  - `aggregate`: maximise throughput (developer mode only)

Apple's path manager is more conservative than Linux's: typically one cellular subflow is held in *backup* state and activated only when Wi-Fi degrades.

== MPQUIC

Multipath QUIC (IETF draft `draft-ietf-quic-multipath`) brings the same ideas to QUIC:
- Multiple network paths share one QUIC connection ID space.
- Each path has its own packet number space, RTT estimate, and congestion controller (instance of $"BBR"$ or $"CUBIC"$).
- Stream data is multiplexed across paths via a scheduler; reordering is handled at the connection-level by QUIC's existing offset/stream machinery.
- No new transport-layer header is required; MPQUIC reuses the existing connection migration primitives plus `PATH_ABANDON` / `PATH_AVAILABLE` frames.

```
QUIC connection (CID space)
├── Path A: client@Wi-Fi  ←→ server  (BBR instance A)
└── Path B: client@LTE     ←→ server  (BBR instance B)
        scheduler: round-robin / lowest-RTT / redundant
```

Implementations:
- Multipath QUIC in `picoquic` and `quic-go` (reference, experimental).
- Cloudflare and Apple have signalled production interest.
- The IETF QUIC WG adopted multipath as an extension in 2023.

== Performance Lessons

- *Heterogeneous paths hurt.* A subflow with 4$times$ the RTT of another can drag overall throughput below single-path if the scheduler is naive; schedulers like BLEST and ECF exist for this reason.
- *Reordering buffer.* The receiver may hold seconds of data waiting for a slow subflow. Tune `net.mptcp.checksum_enabled` and consider `redundant` scheduler for latency-sensitive traffic.
- *Battery cost.* Holding a cellular radio active for MPTCP backup subflow drains the phone; Apple's `handover` mode is the default for this reason.
- *Middlebox traversal.* MPTCP option-stripping middleboxes are common ($>$ 15% of paths historically). The graceful fallback is essential.

== Further Reading

#rfc(6182): Architectural Guidelines for Multipath TCP Development. Ford et al., 2011.

#rfc(6824) (obsoleted by #rfc(8684)): TCP Extensions for Multipath Operation with Multiple Addresses. Ford et al., 2013.

#rfc(6356): Coupled Congestion Control for Multipath Transport Protocols (LIA). Raiciu et al., 2011.

#rfc(8684): TCP Extensions for Multipath Operation (MPTCP v1, supersedes 6824). 2020.

Khalili, R. et al. (2012). "MPTCP is not Pareto-Optimal: Performance Issues and a Possible Solution." CoNEXT.

Peng, Q., Walid, A., Hwang, J. & Low, S. (2016). "Multipath TCP: Analysis, Design, and Implementation (BALIA)." IEEE/ACM ToN.

Paasch, C., Ferlin, S., Alay, O. & Bonaventure, O. (2014). "Experimental Evaluation of Multipath TCP Schedulers." SIGCOMM CSWS.

Bonaventure, O. & Seo, S. (2016). "Multipath TCP Deployments." IETF Journal.

De Coninck, Q. & Bonaventure, O. (2017). "Multipath QUIC: Design and Evaluation." CoNEXT.

draft-ietf-quic-multipath (latest revision): Multipath Extension for QUIC.

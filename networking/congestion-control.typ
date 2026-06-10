= Congestion Control

Congestion control governs how fast a sender may inject data into the network. The choice of algorithm directly determines tail latency, link utilisation, and fairness, and is one of the few transport-layer knobs that has measurable end-user impact. This chapter covers the loss-based classics ($"Reno"$, $"NewReno"$, $"CUBIC"$), the rate-model family ($"BBR"$ v1/v2/v3), and modern marking-based approaches (ECN, L4S with DualPI2 and $"Prague"$).

*See also:* _Transport Layer_ (TCP basics), _QUIC and HTTP/3_ (pluggable CC), _Multipath_ (coupled CC), _Data Center Networking_ (DCTCP and ECN at high speed).

== The Problem

A sender ramps up its sending rate; somewhere along the path a bottleneck queue fills, packets drop, RTT rises, and goodput collapses. The job of congestion control is to find the *bandwidth-delay product* (BDP) of the path and keep the flow near it, without persistent queueing.

$ "BDP" = "bottleneck bandwidth" times "min RTT" $

Two distinct signals are available:
- *Loss:* packet drops indicate the bottleneck buffer overflowed (classic).
- *Delay / rate:* increases in measured $"RTT"$ or stalls in delivery rate indicate queue build-up *before* loss (modern).

Marking (ECN, L4S) lets routers communicate congestion explicitly without drop or delay.

== Loss-Based Algorithms

=== Reno (RFC 5681)

The original AIMD (additive-increase, multiplicative-decrease) controller. On each ACK in *congestion avoidance*:
$ "cwnd" ← "cwnd" + "MSS"^2 / "cwnd" $
On any loss event:
$ "cwnd" ← "cwnd" / 2,\  "ssthresh" ← "cwnd" $

Reno triggers fast retransmit on three duplicate ACKs and fast recovery to skip slow-start after a single loss. Suffers badly when multiple segments are lost in one window, as it exits fast recovery prematurely.

=== NewReno (RFC 6582)

Refines fast recovery: stays in recovery until *all* outstanding data at the start of the recovery has been acknowledged. Each "partial ACK" triggers an immediate retransmission of the next unacknowledged segment. Recovers from $k$ losses in $O(k)$ RTTs.

=== CUBIC (RFC 8312)

Default in Linux since 2007 (kernel 2.6.23), in Windows since 10 (2018). Replaces Reno's linear cwnd growth with a *cubic* function of time since the last loss:
$ W(t) = C dot (t - K)^3 + W_max $
where $W_max$ is the cwnd value at the last loss, $K$ is the time to return to it, and $C ≈ 0.4$. This gives a long, flat region near $W_max$ (slow probing) followed by aggressive ramp once it is exceeded.

*Key property:* growth is independent of RTT, making it much fairer than Reno across heterogeneous-RTT flows and far faster on high-BDP paths (transatlantic 1 Gb/s with $"RTT"=100"ms"$ needs to reach cwnd $approx 12500$ MSS; Reno would take $\sim 1000$ RTTs to recover from one loss, $"CUBIC"$ does it in seconds).

```bash
# Inspect and choose CC on Linux
sysctl net.ipv4.tcp_available_congestion_control
sysctl net.ipv4.tcp_congestion_control
sysctl -w net.ipv4.tcp_congestion_control=cubic

# Per-route override
ip route change default via 192.0.2.1 congctl bbr
```

== BBR — Bottleneck Bandwidth and Round-trip propagation time

$"BBR"$ (Cardwell et al., ACM Queue 2016) abandons loss as a signal. It maintains running estimates of:
- $"btlBw"$: bottleneck bandwidth (max delivery rate in recent windows)
- $"RTprop"$: minimum RTT seen recently

and operates at $"cwnd" = "btlBw" times "RTprop"$ (the BDP itself), keeping the queue near empty.

=== BBR v1 (2016)

Four-state machine:
+ *STARTUP* — exponential ramp until delivery rate plateaus (slow start, but rate-based).
+ *DRAIN* — sends below BDP for one RTT to flush the queue built in startup.
+ *PROBE_BW* — cycles pacing gain through $[1.25, 0.75, 1, 1, 1, 1, 1, 1]$ to probe for bandwidth changes.
+ *PROBE_RTT* — every 10s, drops cwnd to 4 packets for 200ms to re-measure $"RTprop"$.

*Wins:* dramatic throughput improvement on lossy paths (Wi-Fi, mobile), low tail latency.

*Problems found in deployment:*
- *Unfair to CUBIC* on shared bottlenecks with shallow buffers: BBR v1 takes more than its fair share.
- *Retransmission rate* can be high on truly loss-y paths because it ignores loss.
- *Multi-flow oscillation:* multiple v1 flows can synchronise their probe cycles.

=== BBR v2 (2019, draft)

Adds explicit loss and ECN responses on top of the v1 model:
- *Inflight cap:* limits in-flight bytes to $approx 1.25 times "BDP" + 4$ MSS even during PROBE_BW.
- *Loss-rate threshold:* when packet loss exceeds 2%, reduce inflight cap.
- *ECN response:* react to ECE marks similarly to DCTCP for $"DCTCP"$/L4S coexistence.

Deployed widely on Google services and (selectively) by Dropbox, Spotify, and Akamai.

=== BBR v3 (2023+, in development)

Refinements focused on shallow-buffer fairness and aggregate behaviour with many concurrent flows; relaxes the v1/v2 STARTUP overshoot; better convergence with CUBIC. Available as a Linux out-of-tree patch and in Google's QUICHE.

```bash
# Enable BBR (Linux 4.9+)
modprobe tcp_bbr
sysctl -w net.ipv4.tcp_congestion_control=bbr
# Pace packets — required for BBR's rate model
sysctl -w net.core.default_qdisc=fq
```

== ECN — Explicit Congestion Notification (RFC 3168)

Routers can *mark* packets (set CE = Congestion Experienced in the IP header) instead of dropping them when the queue grows. The receiver echoes the mark via the ECE TCP flag; the sender reacts as if a loss had occurred, but without retransmission. This eliminates the latency penalty of drop-based signalling.

```
IP header bits 14-15 (ECN field):
  00  Not-ECT      packet not ECN-capable
  10  ECT(0)       sender is ECN-capable, classic semantics
  01  ECT(1)       sender is ECN-capable, scalable / L4S semantics
  11  CE           Congestion Experienced (set by router)
```

*Deployment.* ECN negotiation is in the TCP SYN/SYN-ACK. Despite being RFC since 2001, classic ECN was held back for years by middlebox brokenness; today $>$ 80% of paths support it, and DCTCP-style ECN is standard inside data centres.

=== DCTCP (RFC 8257)

DCTCP weighs reaction to the *fraction* of marked packets rather than reacting once per RTT:
$ alpha ← (1 - g) alpha + g dot F $
$ "cwnd" ← "cwnd" (1 - alpha / 2) $
where $F$ is the fraction of bytes ECE-marked in the last RTT and $g approx 1/16$ is an EWMA gain. Result: fine-grained, low-variance reaction; achieves near-line-rate utilisation at near-zero queueing in tightly engineered data-centre fabrics. Requires shallow-threshold AQM at switches (typically RED with a step-function K threshold).

== L4S — Low Latency, Low Loss, Scalable Throughput

L4S (RFCs 9330-9332, 2023) takes DCTCP's idea to the public internet. Two classes of traffic share a single bottleneck queue but are AQM-managed separately:

- *Classic* (ECT(0) / Not-ECT): $"CUBIC"$, $"Reno"$ ... reacts coarsely to loss/marking.
- *Scalable* (ECT(1)): $"Prague"$, DCTCP, $"BBR"$v2/v3 with L4S extension — reacts finely to per-packet marking.

=== DualPI2 AQM

DualPI2 is the reference AQM for L4S: a single physical queue with two virtual queues sharing it. Classic traffic sees a deeper drop-based PI2 controller; scalable traffic sees a shallow, step-marked queue at $approx 1$ms target. A *coupling* term ensures Classic and Scalable flows get the same throughput share when they compete.

```
                    │
   IP packet ───────┼──── ECT(1)? ── scalable virtual queue ─── shallow mark
                    │                                            (~1ms target)
                    │── else ──── classic virtual queue ───────── PI2 drop / mark
                    │                                            (~15ms target)
                    ▼
             single physical queue
```

=== TCP Prague

Prague is the scalable congestion controller designed for L4S: DCTCP-style but with internet-grade fixes (RTT independence, slow-start safety, classic-flow coexistence, and burst tolerance). Available as a Linux module (`tcp_prague`). Achieves single-digit-millisecond queueing latency at full link utilisation when paired with a DualPI2 AQM.

```bash
# L4S-capable kernel (5.13+ with tcp_prague patches)
sysctl -w net.ipv4.tcp_congestion_control=prague
sysctl -w net.ipv4.tcp_ecn=1

# Inspect ECN behaviour on a flow
ss -ti dst :443
# ... ecn ecnseen ts sack cubic wscale:7,7 rto:204 rtt:1.3/0.5 ...
```

== Algorithm Comparison

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Algorithm*], [*Signal*], [*Default In*], [*Strength*], [*Weakness*],
  [Reno], [Loss (3 dup ACK)], [historical], [simple], [multi-loss recovery],
  [NewReno], [Loss], [legacy], [handles multi-loss], [slow on high-BDP],
  [CUBIC], [Loss], [Linux, Windows], [high BDP, RTT-fair], [bufferbloat-prone],
  [BBR v1], [BW / RTT model], [YouTube initial], [lossy paths, low latency], [unfair to CUBIC],
  [BBR v2], [BW + loss + ECN], [Google services], [fairness vs CUBIC], [complex tuning],
  [BBR v3], [BW + loss + ECN], [Chromium QUIC], [shallow-buffer fairness], [still maturing],
  [DCTCP], [ECN fraction], [data centres], [near-zero queue], [requires AQM],
  [Prague (L4S)], [ECT(1) marking], [L4S testbeds], [$\sim$1ms latency at full utilisation], [needs DualPI2 path],
)

=== Classic vs Scalable

#table(
  columns: (auto, auto),
  [*Classic CC*], [*Scalable CC*],
  [Reduces $"cwnd"$ by $approx 50%$ per signal], [Reduces in proportion to marking fraction],
  [Signals once per RTT (or less)], [Signals on most packets at congestion],
  [Throughput $prop 1 / sqrt(p)$, where $p$ is loss/mark probability], [Throughput $prop 1 / p$],
  [Tolerates $~ 100$ms queueing], [Targets $<$ 1ms queueing],
  [Examples: $"Reno"$, $"CUBIC"$, $"BBR"$v1], [Examples: DCTCP, $"Prague"$, $"BBR"$v2-L4S],
)

== Measurement and Diagnostics

```bash
# Per-socket congestion stats
ss -tin
# rcvmss:1448 advmss:1448 cwnd:43 ssthresh:32 bytes_acked:64512
# segs_out:78 segs_in:75 send 4.0Mbps lastsnd:8 lastrcv:8 lastack:8 pacing_rate 4.7Mbps
# delivery_rate 3.8Mbps app_limited busy:248ms unacked:1 rcv_rtt:14 rcv_space:14600

# Live tracing
bpftrace -e 'kprobe:tcp_cong_avoid_ai { @cwnd = lhist(arg1, 0, 1000, 10); }'

# Capture and visualise with tcptrace + xplot
tcpdump -w /tmp/cap.pcap -i eth0 host 198.51.100.10
tcptrace -G /tmp/cap.pcap
xplot.org a2b_tput.xpl
```

== Pitfalls

- *Bufferbloat.* Deep dumb buffers (cable modems, home routers) destroy loss-based CC by hiding congestion behind seconds of queueing latency. CoDel and fq_codel AQM are the consumer-side fix; CUBIC alone cannot help.
- *Pacing required for BBR.* Without `fq` qdisc, BBR's rate model degrades. Always pair `tcp_congestion_control=bbr` with `net.core.default_qdisc=fq`.
- *ECN black-holing.* Some legacy paths drop or rewrite the ECN bits. Linux probes for ECN failure and degrades gracefully (`tcp_ecn_fallback`).
- *Coexistence.* Mixing $"BBR"$v1 and $"CUBIC"$ on the same bottleneck favours BBR; choose v2 or v3 for fairness when interoperating.

== Exercises

1. A path has a bottleneck bandwidth of 100 Mb/s and a minimum RTT of 80 ms. Compute the BDP in bytes and in 1500-byte packets. What happens to queueing delay if a sender keeps twice the BDP in flight?
  _Hint: multiply rate by delay, then convert bits to bytes; the excess inflight sits in the bottleneck queue._

2. Explain why Reno's recovery from a single loss takes roughly $W/2$ RTTs of linear growth, and why CUBIC's $W(t) = C dot (t - K)^3 + W_max$ recovers a high-BDP path so much faster. What property makes CUBIC fairer across flows with different RTTs?
  _Hint: Reno grows about one MSS per RTT; CUBIC's growth is a function of wall-clock time since the last loss, not of RTT._

3. Describe the purpose of each of BBR v1's four states (STARTUP, DRAIN, PROBE_BW, PROBE_RTT). Why must PROBE_RTT periodically drop cwnd to a few packets rather than just observing passively?
  _Hint: a standing queue of the flow's own making inflates every RTT sample; the queue must be drained to see the propagation delay._

4. A BBR v1 flow shares a shallow-buffered bottleneck with a CUBIC flow. Predict the outcome and explain which v2 additions (inflight cap, loss-rate threshold, ECN response) address it.
  _Hint: v1 ignores loss, so it keeps its model-derived rate while CUBIC backs off on every drop._

5. With DCTCP's update rules $alpha arrow.l (1 - g) alpha + g dot F$ and $"cwnd" arrow.l "cwnd" (1 - alpha\/2)$, compute the cwnd reduction when the steady-state marked fraction is $F = 0.1$ versus when every packet is marked ($F = 1$). How does this differ from classic ECN's reaction?
  _Hint: at steady state $alpha approx F$; classic ECN halves cwnd once per RTT regardless of the marked fraction._

6. L4S separates traffic using the ECT(1) codepoint into a shallow-marked scalable queue and a deeper classic queue. Explain why scalable and classic flows cannot safely share a single classic AQM, using the throughput relations $prop 1\/p$ and $prop 1\/sqrt(p)$.
  _Hint: at the same marking probability the two laws give wildly different rates; the DualPI2 coupling term equalizes them._

== Further Reading

RFC 5681: TCP Congestion Control. Allman et al., 2009.

RFC 6582: NewReno Modification to TCP's Fast Recovery Algorithm.

RFC 8312: CUBIC for Fast Long-Distance Networks. Rhee et al., 2018.

RFC 8257: DCTCP — Data Center TCP for High-Performance Networks.

RFC 3168: The Addition of Explicit Congestion Notification (ECN) to IP.

RFC 9330-9332: L4S Architecture, DualPI2, and Identifier.

Cardwell, N. et al. (2016). "BBR: Congestion-Based Congestion Control." ACM Queue.

Cardwell, N. et al. (2019). "BBRv2: A Model-Based Congestion Control." IETF 105.

Alizadeh, M. et al. (2010). "Data Center TCP (DCTCP)." SIGCOMM.

Briscoe, B. et al. (2016). "Reducing Internet Latency: A Survey of Techniques and Their Merits." IEEE Commun. Surveys.

De Schepper, K. et al. (2017). "PI2: A Linearized AQM for both Classic and Scalable TCP." CoNEXT.

Ha, S., Rhee, I. & Xu, L. (2008). "CUBIC: A New TCP-Friendly High-Speed TCP Variant." SIGOPS.

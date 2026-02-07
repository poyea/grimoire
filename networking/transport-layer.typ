= Transport Layer

The transport layer provides end-to-end communication between applications. Two primary protocols: TCP (reliable, ordered, connection-oriented) and UDP (unreliable, unordered, connectionless).

*See also:* Sockets API (for programming interface), Application Protocols (for protocol design), Concurrency Models (for handling multiple connections)

== TCP (Transmission Control Protocol)

*TCP provides [RFC 793]:*
- Reliable delivery (retransmission of lost packets)
- Ordered delivery (sequence numbers)
- Flow control (receiver window)
- Congestion control (sender rate limiting)

*TCP segment structure:*

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
├───────────────────────────────────────────────────────────────┤
│          Source Port          │       Destination Port        │
├───────────────────────────────────────────────────────────────┤
│                        Sequence Number                        │
├───────────────────────────────────────────────────────────────┤
│                     Acknowledgment Number                     │
├───────────┬───────┬─┬─┬─┬─┬─┬─┬─┬─┬─────────────────────────┤
│Data Offset│ Res   │C│E│U│A│P│R│S│F│        Window Size        │
│           │       │W│C│R│C│S│S│Y│I│                           │
│           │       │R│E│G│K│H│T│N│N│                           │
├───────────┴───────┴─┴─┴─┴─┴─┴─┴─┴─┴───────────────────────────┤
│           Checksum            │       Urgent Pointer          │
├───────────────────────────────┴───────────────────────────────┤
│                    Options (0-40 bytes)                       │
├───────────────────────────────────────────────────────────────┤
│                             Data                              │
└───────────────────────────────────────────────────────────────┘
```

*Minimum header:* 20 bytes (without options)
*Maximum header:* 60 bytes (with 40 bytes of options)

*Key fields:*
- *Sequence number:* Byte offset of first data byte (32-bit, wraps)
- *ACK number:* Next expected byte from sender
- *Window size:* Receiver buffer space (16-bit, scaled with window scale option)
- *Flags:* SYN (establish), ACK (acknowledge), FIN (close), RST (reset), PSH (push data)

== TCP Connection Establishment (3-Way Handshake)

```
Client                          Server
  │                               │
  │  SYN, seq=x                  │
  ├────────────────────────────▶ │ (1 RTT)
  │                               │
  │  SYN+ACK, seq=y, ack=x+1    │
  │ ◀────────────────────────────┤ (2 RTT)
  │                               │
  │  ACK, seq=x+1, ack=y+1      │
  ├────────────────────────────▶ │ (3 RTT)
  │                               │
  │  Data                        │
  ├────────────────────────────▶ │
```

*Latency:* 1 RTT before client can send data, 1.5 RTT before server can send data.

*Modern optimization - TCP Fast Open (TFO) [RFC 7413]:*

```
Client                          Server
  │  SYN + TFO cookie + data    │
  ├────────────────────────────▶ │ (0 RTT!)
  │  SYN+ACK + response data    │
  │ ◀────────────────────────────┤
```

*Benefit:* 0-RTT data transmission for repeat connections. Saves ~50-100ms on cross-country connections.

*Enable TFO:*
```bash
# Linux
echo 3 > /proc/sys/net/ipv4/tcp_fastopen  # 1=client, 2=server, 3=both
```

== TCP Congestion Control

*Problem:* Sender faster than network capacity → packet loss → retransmissions → congestion collapse [Jacobson 1988].

*TCP congestion control algorithms:*

*1. Reno (classic, 1990):*
- Slow start: cwnd \*= 2 every RTT until ssthresh
- Congestion avoidance: cwnd += 1/cwnd every RTT
- Fast retransmit/recovery: 3 duplicate ACKs → retransmit, cwnd /= 2

*2. CUBIC (Linux default since 2.6.19):*
```cpp
// Simplified CUBIC increase function
double cwnd_cubic(double time_since_loss, double cwnd_max) {
    double K = pow(cwnd_max * 0.7 / 0.4, 1.0/3.0);  // Inflection point
    double t = time_since_loss - K;
    return 0.4 * t*t*t + cwnd_max;
}
```

*Characteristics:*
- Fast ramp-up after loss (cubic function)
- RTT-fair: same throughput regardless of RTT
- Scales to high-bandwidth networks (10Gbps+)

*Performance [Ha et al. 2008]:*
- 10Gbps link, 100ms RTT: CUBIC achieves 95% utilization
- Reno: only 60% utilization (slow cwnd growth)

*3. BBR (Bottleneck Bandwidth and RTT, Google 2016):*

```cpp
// BBR operates in 4 phases cyclically:
enum BBRMode {
    BBR_STARTUP,       // Exponential search for bandwidth
    BBR_DRAIN,         // Drain queue built during startup
    BBR_PROBE_BW,      // Oscillate around optimal point
    BBR_PROBE_RTT      // Reduce queue delay periodically
};

// Key insight: Don't use loss as congestion signal
// Instead: Measure bottleneck_bw and min_rtt
// Send at pacing_rate = bottleneck_bw * gain
```

*Advantages:*
- Works well on lossy links (WiFi, cellular) - doesn't misinterpret loss as congestion
- Lower latency: doesn't fill buffers
- ~2-10x throughput on high-loss networks [Cardwell et al. 2016]

*Disadvantages:*
- Can be unfair to loss-based congestion control
- Requires careful tuning in shared environments

*BBR vs CUBIC numerical walkthrough:*

Scenario: 100 Mbps bottleneck link, 50ms RTT, 1% random packet loss.

```
                    CUBIC                           BBR
                    ─────                           ───
Initial cwnd:       10 segments (14.6 KB)           10 segments (14.6 KB)
MSS:                1460 bytes                      1460 bytes

Slow start:
  RTT 1:            cwnd = 20 (29.2 KB)             cwnd = 20 (29.2 KB)
  RTT 2:            cwnd = 40 (58.4 KB)             cwnd = 40 (58.4 KB)
  RTT 3:            cwnd = 80 (116.8 KB)            cwnd = 80 (116.8 KB)
  ...               doubles until loss               doubles until BtlBw estimated

BDP = 100 Mbps × 50ms = 625 KB ≈ 428 segments

After reaching BDP:
  CUBIC:            cwnd grows until loss detected
                    1% loss → cwnd cut to 0.7 × cwnd_max
                    cwnd_max = 428, after loss: cwnd ≈ 300
                    Cubic regrowth: W(t) = 0.4(t - K)³ + cwnd_max
                    K = ∛(cwnd_max × 0.3 / 0.4) = ∛(321) ≈ 6.9 RTTs
                    Recovery to cwnd_max takes ~14 RTTs = 700ms

  BBR:              Measures BtlBw = 100 Mbps, RTprop = 50ms
                    pacing_rate = BtlBw × gain (1.25 in PROBE_BW)
                    cwnd = 2 × BDP = 1250 KB (inflight cap)
                    Ignores 1% loss entirely (not a congestion signal)
                    Steady throughput ≈ 98 Mbps

Steady-state throughput:
  CUBIC:            Sawtooth pattern between 70-100 Mbps
                    Average ≈ 85 Mbps (loses ~15% to loss recovery)
                    Fills router buffers → adds 10-50ms latency

  BBR:              Stable at ~98 Mbps
                    Minimal buffering → RTT stays near 50ms
                    PROBE_RTT phase: brief dip every ~10s (cwnd = 4)

Summary (100 Mbps, 50ms RTT, 1% loss):
  CUBIC avg throughput:  ~85 Mbps    avg RTT: 60-100ms
  BBR avg throughput:    ~98 Mbps    avg RTT: 50-55ms
```

*High-bandwidth scenario (10 Gbps, 100ms RTT, 0.01% loss):*
```
BDP = 10 Gbps × 100ms = 125 MB ≈ 85,616 segments

CUBIC: cwnd recovery after loss takes ~40 RTTs = 4 seconds
       Average utilization: ~60-70% (slow cubic regrowth)

BBR:   Maintains ~95% utilization
       pacing_rate adapts within 6-8 RTTs
```

*Enable BBR:*
```bash
echo "net.core.default_qdisc=fq" >> /etc/sysctl.conf
echo "net.ipv4.tcp_congestion_control=bbr" >> /etc/sysctl.conf
sysctl -p
```

== TCP Flow Control (Receiver Window)

*Problem:* Sender faster than receiver → buffer overflow.

*Solution:* Receiver advertises available buffer space in TCP Window field (16-bit, max 65535 bytes).

*Window scaling [RFC 1323]:*
```cpp
// Option in SYN: window scale factor (0-14)
// Actual window = TCP_window << scale_factor
// Max window = 65535 << 14 = 1GB
```

*Optimal window size:* $W = "BDP" ("Bandwidth-Delay Product")$

Example: 10Gbps link, 50ms RTT:
```
BDP = 10 Gbps × 0.05s = 0.5 Gbit = 62.5 MB
```

Requires window scale factor ~= 20 (65KB × 2^20 = 68GB, scaled down to fit).

*Modern tuning (Linux):*
```bash
# Auto-tuning (enabled by default)
net.ipv4.tcp_window_scaling = 1
net.ipv4.tcp_moderate_rcvbuf = 1  # Auto-adjust buffer

# Buffer sizes (min, default, max)
net.ipv4.tcp_rmem = 4096 87380 67108864  # 4KB, 85KB, 64MB
net.ipv4.tcp_wmem = 4096 16384 67108864  # 4KB, 16KB, 64MB
```

== TCP Retransmission and Timeouts

*RTO (Retransmission Timeout) calculation [RFC 6298]:*

```cpp
// Karn's algorithm with Jacobson's variance
SRTT = (1 - α) × SRTT + α × RTT_sample  // α = 1/8
RTTVAR = (1 - β) × RTTVAR + β × |SRTT - RTT_sample|  // β = 1/4

RTO = SRTT + 4 × RTTVAR
RTO = max(RTO, 1 second)  // Floor
```

*Exponential backoff:* If retransmission times out again, RTO \*= 2 (up to 120s typical).

*Fast retransmit:* 3 duplicate ACKs → assume packet lost → retransmit immediately (don't wait for RTO).

*SACK (Selective Acknowledgment) [RFC 2018]:*

Without SACK:
```
Lost packet 100
ACK repeats: 99, 99, 99, 99...
Sender must retransmit 100, wait for ACK, then retransmit 101, 102...
```

With SACK:
```
ACK: 99, SACK blocks: 101-200
Sender knows: 100 lost, 101-200 received → retransmit only 100
```

*Performance:* SACK improves throughput 10-30% on lossy networks.

== TCP Nagle's Algorithm and Delay

*Nagle's algorithm [RFC 896]:* Coalesce small writes to reduce packet count.

```cpp
// Simplified logic
if (outstanding_data == 0 || write_size >= MSS) {
    send_immediately();
} else {
    buffer_until_ack_or_timeout();
}
```

*Problem for interactive applications:*
```cpp
write(sock, "GET ", 4);       // Buffered
write(sock, "/index.html", 11);  // Buffered
write(sock, "\r\n", 2);       // Sent when ACK arrives (40ms delay!)
```

*Solution: Disable Nagle*
```cpp
int flag = 1;
setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
```

*Tradeoff:*
- Nagle ON: Fewer packets, higher latency (good for bulk transfer)
- Nagle OFF: More packets, lower latency (good for interactive apps)

== TCP_CORK (Linux-Specific)

*Opposite of TCP_NODELAY:* Buffer aggressively until cork removed.

```cpp
int on = 1;
setsockopt(sock, IPPROTO_TCP, TCP_CORK, &on, sizeof(on));

write(sock, headers, 100);
write(sock, body, 10000);

int off = 0;
setsockopt(sock, IPPROTO_TCP, TCP_CORK, &off, sizeof(off));  // Flush
```

*Use case:* HTTP response with headers + body. Cork prevents sending headers alone.

== UDP (User Datagram Protocol)

*UDP characteristics [RFC 768]:*
- Connectionless (no handshake)
- Unreliable (no retransmission)
- Unordered (no sequence numbers)
- Minimal overhead (8-byte header vs 20+ for TCP)

*UDP header:*
```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
├───────────────────────────────────────────────────────────────┤
│          Source Port          │       Destination Port        │
├───────────────────────────────────────────────────────────────┤
│            Length             │           Checksum            │
├───────────────────────────────────────────────────────────────┤
│                             Data                              │
```

*When to use UDP:*
1. *Low latency required:* Gaming, VoIP, live streaming (loss tolerable)
2. *Broadcast/multicast:* DNS, DHCP, service discovery
3. *Custom reliability:* QUIC, RTP (application-layer retransmission)
4. *High packet rate:* Sensor data, metrics (occasional loss acceptable)

*Performance advantage:*
- No connection setup: 0-RTT communication
- No flow/congestion control: Full link speed immediately
- No head-of-line blocking: Independent packets

*Example latency (localhost):*
- TCP: 10-30μs (includes connection overhead)
- UDP: 5-10μs (direct send)

== QUIC (Quick UDP Internet Connections)

*Modern transport protocol (HTTP/3 foundation) [RFC 9000]:*

*Key innovations:*
1. *0-RTT connection resumption:* TLS + transport in single handshake
2. *No head-of-line blocking:* Multiple streams multiplexed, independent loss recovery
3. *Connection migration:* Survives IP address changes (mobile networks)
4. *Built-in encryption:* TLS 1.3 integrated

*QUIC vs TCP+TLS:*

```
TCP + TLS 1.3:
Client → Server: TCP SYN                (1 RTT)
Client → Server: TLS ClientHello        (2 RTT)
Client → Server: HTTP request           (3 RTT)

QUIC (0-RTT):
Client → Server: QUIC handshake + HTTP request  (1 RTT)
```

*Latency savings:* 2 RTT = 100-200ms on cross-country connections.

*Congestion control:* QUIC implements CUBIC or BBR in userspace (no kernel dependency).

*Adoption:* Google services, Facebook, Cloudflare, ~50% of internet traffic [2023].

== TCP Performance Pathologies

*1. Silly Window Syndrome:* Receiver advertises small windows → sender sends tiny segments.

*Solution:* Don't advertise window < MSS unless buffer empty [RFC 1122].

*2. Delayed ACKs:* Receiver waits up to 500ms to ACK (hoping for data to piggyback).

*Problem:* Increases latency. *Solution:* TCP_QUICKACK (Linux).

```cpp
int flag = 1;
setsockopt(sock, IPPROTO_TCP, TCP_QUICKACK, &flag, sizeof(flag));
```

*3. Incast:* Many senders → single receiver → synchronized loss → timeout.

*Common in datacenters:* Partition-aggregate workloads (MapReduce).

*Solutions:*
- Reduce RTO minimum (default 200ms → 1ms in datacenter)
- Use DCTCP (Data Center TCP) - ECN-based congestion control

*4. Bufferbloat:* Large router queues → high latency under load.

*Example:* Home router with 1s buffer (DSL era legacy) → 1000ms latency spike.

*Solutions:*
- FQ_CoDel (Fair Queuing + Controlled Delay) at routers
- BBR at endpoints (detects queue buildup)

== References

*Primary sources:*

RFC 793: Transmission Control Protocol. Postel, J. (1981).

RFC 768: User Datagram Protocol. Postel, J. (1980).

RFC 7413: TCP Fast Open. Cheng, Y., Chu, J., Radhakrishnan, S., & Jain, A. (2014).

RFC 9000: QUIC: A UDP-Based Multiplexed and Secure Transport. Iyengar, J. & Thomson, M. (2021).

Jacobson, V. (1988). "Congestion Avoidance and Control." SIGCOMM '88.

Ha, S., Rhee, I., & Xu, L. (2008). "CUBIC: A New TCP-Friendly High-Speed TCP Variant." ACM SIGOPS Operating Systems Review 42(5): 64-74.

Cardwell, N., Cheng, Y., Gunn, C.S., Yeganeh, S.H., & Jacobson, V. (2016). "BBR: Congestion-Based Congestion Control." Communications of the ACM 60(2): 58-66.

Mathis, M., Mahdavi, J., Floyd, S., & Romanow, A. (1996). "TCP Selective Acknowledgment Options." RFC 2018.

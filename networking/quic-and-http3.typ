#import "../template.typ": rfc, xref

= QUIC and HTTP/3 <quic-and-http3>

QUIC is a UDP-based, encrypted, stream-multiplexed transport: effectively a redesign of TCP+TLS+HTTP/2 into a single user-space protocol. Standardised as #rfc(9000) (2021), QUIC underpins HTTP/3 (#rfc(9114)) and now carries the majority of traffic for Google, Cloudflare, and Meta. This chapter covers stream multiplexing, 0-RTT, connection migration, pluggable congestion control, the HTTP/3 mapping, and real-world deployment lessons.

*See also:* #xref("networking", "transport-layer", label: "Transport Layer") (for TCP background), #xref("networking", "congestion-control", label: "Congestion Control") (for $"BBR"$ and $"CUBIC"$), _TLS_ (QUIC integrates TLS 1.3), #xref("networking", "kernel-bypass", label: "Kernel Bypass") (most QUIC stacks live in user space).

== Why QUIC?

TCP has three deeply baked-in problems that no incremental fix can address:

+ *Head-of-line blocking at the transport layer.* A single lost segment stalls every higher-level stream. HTTP/2 multiplexes streams over one TCP connection, but one lost packet blocks all of them.
+ *Handshake latency.* TCP needs 1 RTT for the three-way handshake; TLS 1.2 adds 2 more; TLS 1.3 reduces to 1; first byte arrives at $2$-$3$ RTTs.
+ *Ossification.* Middleboxes inspect and rewrite TCP headers. Deploying TCP Fast Open or new options to the public internet has taken a decade and is still unreliable.

QUIC's response:
- Runs over UDP; middleboxes cannot inspect inside the encrypted payload.
- Built-in TLS 1.3; handshake and key exchange merged with transport setup (1 RTT for new, 0 RTT for resumed connections).
- Streams are first-class; loss on stream 1 does not stall stream 2.
- Connection ID decoupled from `(src_ip, src_port, dst_ip, dst_port)`; survives NAT rebinding and Wi-Fi → 5G handoff (connection migration).

== Packet and Frame Structure

A QUIC packet contains a header (long or short form) plus one or more *frames*. Frames carry control or data information; the entire payload after the header is AEAD-encrypted with TLS 1.3 keys.

```
Long header (handshake / Initial):
+----+--------+------+--------+--------+----------+-----------+
| 1  | type   | ver  | DCID   | SCID   | token    | payload   |
+----+--------+------+--------+--------+----------+-----------+

Short header (1-RTT data):
+---+----------+----------+-----------+
| 0 | flags    | DCID     | payload   |
+---+----------+----------+-----------+
```

Common frame types:

#table(
  columns: (auto, auto),
  [*Frame*], [*Purpose*],
  [`STREAM`], [Application stream data with offset and FIN bit],
  [`ACK`], [Acknowledge received packets; includes ACK delay for accurate $"RTT"$],
  [`CRYPTO`], [TLS handshake messages],
  [`MAX_DATA` / `MAX_STREAM_DATA`], [Flow control credit],
  [`NEW_CONNECTION_ID`], [Issue additional CIDs for migration],
  [`PATH_CHALLENGE` / `PATH_RESPONSE`], [Validate new path during migration],
  [`CONNECTION_CLOSE`], [Terminate with error code],
)

*Key invariant:* packet numbers are monotonically increasing and never re-used (unlike TCP sequence numbers which retransmit with same SEQ). This eliminates the retransmission ambiguity that complicates TCP $"RTT"$ measurement.

== Streams and Multiplexing

A QUIC stream is a lightweight ordered byte stream identified by a 62-bit integer. Lowest two bits encode:
- bit 0: 0 = client-initiated, 1 = server-initiated
- bit 1: 0 = bidirectional, 1 = unidirectional

```
client bidi:  0, 4, 8, ...
server bidi:  1, 5, 9, ...
client uni:   2, 6, 10, ...
server uni:   3, 7, 11, ...
```

Each stream maintains its own offset; a loss in stream 4 does not block delivery of stream 8, eliminating HTTP/2's transport-level head-of-line blocking. Flow control is two-tier: per-stream and per-connection.

== 0-RTT Resumption

If the client has cached a session ticket from a prior connection, it can send application data *in the very first flight*, alongside the ClientHello:

```
Client → Server (Initial + 0-RTT data):
   - CRYPTO frame: ClientHello (with PSK extension)
   - STREAM frame:  GET /index.html  (encrypted with 0-RTT key)

Server → Client:
   - CRYPTO frame: ServerHello, Finished
   - STREAM frame: HTTP/3 response (encrypted with 1-RTT key)
```

*Security caveat:* 0-RTT data is replayable. Servers MUST treat 0-RTT-carried requests as idempotent or reject them. HTTP/3 servers commonly accept only GET / HEAD over 0-RTT, downgrading POST and friends to 1-RTT.

== Connection Migration

The QUIC *Connection ID* (CID) identifies a connection independently of the 4-tuple. When a client's IP changes (laptop suspends, phone switches Wi-Fi to LTE), it continues sending packets with the same CID; the server validates the new path with `PATH_CHALLENGE` / `PATH_RESPONSE` and resumes the connection with no TCP-style reconnection or TLS rehandshake.

```
Phase 1 — Wi-Fi:    client@192.0.2.5:54321 ──CID=0xABC──▶ server
Phase 2 — LTE:      client@198.51.100.7:60000 ──CID=0xABC──▶ server
                    server: PATH_CHALLENGE → client: PATH_RESPONSE
                    server now uses 198.51.100.7:60000 for this CID
```

Each endpoint issues multiple CIDs (`NEW_CONNECTION_ID` frame) so observers cannot correlate flows across migration. Apple's iOS uses QUIC migration extensively for backgrounding-tolerant uploads.

== Pluggable Congestion Control

The QUIC specification mandates congestion control "equivalent to TCP NewReno" by default but allows any algorithm. Real deployments overwhelmingly use $"BBR"$ or $"CUBIC"$:

#table(
  columns: (auto, auto, auto),
  [*Stack*], [*Default CC*], [*Notes*],
  [Google QUICHE], [$"BBR"$v2], [Same code drives Chrome and YouTube],
  [Cloudflare quiche (Rust)], [$"CUBIC"$], [$"BBR"$v2 selectable],
  [Meta mvfst (C++)], [$"BBR"$ + Copa], [Used for Facebook / Instagram],
  [Microsoft msquic], [$"CUBIC"$], [Ships in Windows kernel and IIS],
  [LiteSpeed lsquic], [$"BBR"$ / $"CUBIC"$], [Powerful for CDN edges],
)

Because CC lives in user space, a server can A/B-test new algorithms (e.g. $"BBR"$v3, Copa, $"PCC"$) without kernel cooperation, a major operational advantage over TCP.

== HTTP/3 Mapping

HTTP/3 (#rfc(9114)) is HTTP semantics layered onto QUIC streams. The mapping is straightforward:

- One HTTP request/response pair = one bidirectional QUIC stream.
- Headers compressed with *QPACK* (#rfc(9204)), like HPACK but tolerant of out-of-order stream delivery (uses a separate encoder / decoder stream).
- Two unidirectional control streams (one per direction) carry `SETTINGS`, `GOAWAY`, etc.
- Server push uses `PUSH_PROMISE` (rarely deployed; Chrome removed support).

```
Stream 0 (bidi, client): GET /index.html  → 200 OK + body
Stream 4 (bidi, client): GET /style.css   → 200 OK + body
Stream 2 (uni,  client): QPACK encoder stream
Stream 3 (uni,  server): control stream (SETTINGS frame)
```

*Discovery:* HTTP/3 endpoints advertise themselves via the `Alt-Svc` header in an HTTP/1.1 or HTTP/2 response, or via the `HTTPS` DNS record (#rfc(9460)):

```
Alt-Svc: h3=":443"; ma=86400
```

Or DNS:
```
example.com. IN HTTPS 1 . alpn="h3,h2" port=443
```

== Tooling and Observability

```bash
# curl with HTTP/3 (requires recent curl built with --with-nghttp3)
curl --http3 -v https://cloudflare-quic.com/

# Force HTTP/3 only (no fallback)
curl --http3-only https://www.google.com/

# Capture and decrypt QUIC in Wireshark
# 1. Export TLS keys from client:
SSLKEYLOGFILE=/tmp/keys.log curl --http3 https://example.com
# 2. In Wireshark → Preferences → Protocols → TLS → (Pre)-Master-Secret log file
# 3. QUIC payloads now decrypted; per-stream view available
```

```bash
# qlog (structured QUIC trace, JSON); supported by quiche, mvfst, msquic, picoquic
# Open in qvis (https://qvis.quictools.info) for sequence diagrams,
# congestion window plots, RTT estimates, and packet loss visualisation.
```

```bash
# Server-side metrics from Cloudflare quiche
bpftrace -e 'usdt:/usr/local/bin/h3-server:quiche:pkt_lost { @[probe] = count(); }'
```

== Real-World Deployments

*Google.* Originally deployed gQUIC in 2013 over Chrome ↔ Google front-ends; migrated to IETF QUIC in 2020. As of 2024, $>$ 50% of Google traffic is QUIC (Chrome → google.com, YouTube). Internal CC: $"BBR"$v2, evolving toward $"BBR"$v3.

*Cloudflare.* Open-source `quiche` library (Rust) powers their edge. Public stats show ~30% of HTTPS requests use HTTP/3, with notable wins on mobile networks (median page-load improvement 5-12%).

*Meta.* `mvfst` (C++) drives Facebook, Instagram, WhatsApp. Migration support is critical for the mobile use case. Internal experiments with Copa and learning-based CC.

*Apple.* iOS 14+ uses QUIC for iCloud and the App Store. Network.framework exposes QUIC to third-party apps (iOS 15+).

*Microsoft.* `msquic` ships in Windows 11 / Server 2022 kernel, used by SMB over QUIC (file shares without VPN).

=== Performance Observations from Production

- HTTP/3 wins biggest on lossy, high-RTT mobile links: 10-20% page-load improvement common. On clean fibre, HTTP/2 and HTTP/3 are within noise.
- CPU cost of QUIC is 2-3$times$ that of TCP+TLS due to user-space crypto and per-packet syscalls. UDP `GSO` / `GRO` and `sendmmsg` partially mitigate. SmartNIC offload (Nvidia BlueField) helps further.
- UDP is rate-limited or blocked on $<$ 5% of paths (corporate firewalls, some cellular). Clients fall back to HTTP/2 over TCP using `Alt-Svc` failure caching.
- QUIC's per-packet AEAD overhead (~16 B tag) reduces effective MTU; combined with the 8-byte UDP header, payload per packet is ~24 B less than TCP+TLS.

== Pitfalls

- *MTU and amplification.* During the handshake, servers may reply with at most 3$times$ the bytes the client sent, limiting the initial server flight and preventing reflection attacks. Initial packets are padded to ≥ 1200 B to validate path MTU.
- *UDP receive buffer tuning.* High-throughput QUIC servers easily overflow the default `net.core.rmem_max`. Raise to 16 MiB or more.
- *Stateless reset oracle.* A misconfigured load balancer routing CIDs to the wrong server can leak existence of valid CIDs via stateless reset tokens. Use connection-ID-aware LB hashing (e.g., Maglev keyed on first 8 bytes of CID).
- *0-RTT replay.* Always restrict to idempotent requests; consider replay caching.

== Further Reading

#rfc(9000): QUIC: A UDP-Based Multiplexed and Secure Transport. Iyengar & Thomson, 2021.

#rfc(9001): Using TLS to Secure QUIC.

#rfc(9002): QUIC Loss Detection and Congestion Control.

#rfc(9114): HTTP/3.

#rfc(9204): QPACK Field Compression for HTTP/3.

#rfc(9460): Service Binding and Parameter Specification via the DNS (SVCB and HTTPS RRs).

Langley, A. et al. (2017). "The QUIC Transport Protocol: Design and Internet-Scale Deployment." SIGCOMM.

Kosek, M., Shreedhar, T. & Bajpai, V. (2022). "Beyond QUIC v1 — A First Look at Recent Transport Layer Extensions." IEEE Communications.

Yang, X. et al. (2020). "Making QUIC Quicker With NIC Offload." ACM SIGCOMM Workshop EPIQ.

Marx, R. et al. (2020). "Same Standards, Different Decisions: A Study of QUIC and HTTP/3 Implementation Diversity." EPIQ.

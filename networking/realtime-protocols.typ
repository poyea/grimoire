= Real-Time Protocols

Real-time communication requires specialized protocols that prioritize low latency over reliability. This chapter covers WebRTC, RTP/RTCP, SRT, and media streaming fundamentals.

*See also:* Transport Layer (for UDP fundamentals), Application Protocols (for HTTP-based streaming), Wireless Protocols (for mobile considerations)

== Real-Time Communication Fundamentals

*Latency requirements by application:*

| Application | Max Latency | Jitter Tolerance |
|-------------|-------------|------------------|
| Live conversation | < 150ms | < 30ms |
| Gaming (FPS) | < 50ms | < 10ms |
| Live sports | < 5s | High |
| Video conferencing | < 200ms | < 50ms |
| Music collaboration | < 10ms | < 1ms |

*Why UDP for real-time:*

```
TCP vs UDP for real-time media:

TCP (100ms RTT, 1% loss):
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ P1 │ P2 │ P3 │LOST│WAIT│WAIT│ P4 │ P5 │
└────┴────┴────┴────┴────┴────┴────┴────┘
                   ↑
            Packet 3 lost → retransmit
            Head-of-line blocking: +200ms delay

UDP (application handles loss):
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ P1 │ P2 │ P3 │LOST│ P4 │ P5 │ P6 │ P7 │
└────┴────┴────┴────┴────┴────┴────┴────┘
                   ↑
            Packet 3 lost → concealment
            No blocking: continue with P4
```

*End-to-end latency breakdown:*

```
┌─────────────────────────────────────────────────────────────┐
│  Capture → Encode → Network → Decode → Render               │
│   10ms     20ms     50ms      20ms     10ms    = 110ms     │
│                                                             │
│  Breakdown:                                                 │
│  - Capture: Camera/mic sampling                            │
│  - Encode: Compression (H.264, Opus)                       │
│  - Network: RTT/2 + jitter buffer                          │
│  - Decode: Decompression                                   │
│  - Render: Display/speaker output                          │
└─────────────────────────────────────────────────────────────┘
```

== WebRTC Architecture

*WebRTC (Web Real-Time Communication) enables browser-based real-time media [W3C/IETF]:*

*Protocol stack:*

```
┌─────────────────────────────────────────────────────────────┐
│                    Application (JavaScript)                  │
├─────────────────────────────────────────────────────────────┤
│  RTCPeerConnection API                                       │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ Media: getUserMedia(), addTrack()                       ││
│  │ Data:  RTCDataChannel (SCTP over DTLS)                 ││
│  └─────────────────────────────────────────────────────────┘│
├──────────────────┬──────────────────────────────────────────┤
│      SRTP        │              SCTP                        │
│  (Secure RTP)    │  (Stream Control Transmission Protocol)  │
├──────────────────┴──────────────────────────────────────────┤
│                    DTLS (Datagram TLS)                       │
├─────────────────────────────────────────────────────────────┤
│                    ICE (Interactive Connectivity)            │
├─────────────────────────────────────────────────────────────┤
│                    UDP (rarely TCP/TLS)                      │
└─────────────────────────────────────────────────────────────┘
```

*Connection establishment:*

```
┌─────────────┐    Signaling Server    ┌─────────────┐
│   Peer A    │ ◀────────────────────▶ │   Peer B    │
└──────┬──────┘    (WebSocket/HTTP)    └──────┬──────┘
       │                                       │
       │  1. Create offer (SDP)               │
       ├────────────▶ relay ────────────────▶│
       │                                       │
       │  2. Create answer (SDP)              │
       │◀──────────── relay ◀────────────────┤
       │                                       │
       │  3. ICE candidates                    │
       ├◀───────────▶ relay ◀───────────────▶│
       │                                       │
       │  4. DTLS handshake (direct)          │
       │◀─────────────────────────────────────▶
       │                                       │
       │  5. SRTP media flow (direct)         │
       │◀═════════════════════════════════════▶
```

*SDP (Session Description Protocol) example:*

```
v=0
o=- 4611731400430051336 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0 1
a=msid-semantic: WMS

m=audio 9 UDP/TLS/RTP/SAVPF 111 103 104
c=IN IP4 0.0.0.0
a=rtcp:9 IN IP4 0.0.0.0
a=ice-ufrag:abcd
a=ice-pwd:aabbccdd...
a=fingerprint:sha-256 AA:BB:CC:...
a=setup:actpass
a=mid:0
a=sendrecv
a=rtcp-mux
a=rtpmap:111 opus/48000/2
a=fmtp:111 minptime=10;useinbandfec=1

m=video 9 UDP/TLS/RTP/SAVPF 96 97 98
a=rtpmap:96 VP8/90000
a=rtpmap:97 VP9/90000
a=rtpmap:98 H264/90000
```

*ICE (Interactive Connectivity Establishment) [RFC 8445]:*

```cpp
// Candidate types (in preference order)
enum CandidateType {
    HOST,        // Local interface address
    SRFLX,       // Server reflexive (STUN)
    PRFLX,       // Peer reflexive (discovered during check)
    RELAY        // TURN server relay
};

// ICE connectivity checks
struct CandidatePair {
    Candidate local;
    Candidate remote;
    uint64_t priority;      // Computed from candidate priorities
    State state;            // WAITING, IN_PROGRESS, SUCCEEDED, FAILED
};

// Priority calculation
uint32_t compute_priority(CandidateType type, uint32_t local_pref,
                          uint32_t component_id) {
    uint32_t type_pref;
    switch (type) {
        case HOST:  type_pref = 126; break;
        case SRFLX: type_pref = 100; break;
        case PRFLX: type_pref = 110; break;
        case RELAY: type_pref = 0;   break;
    }
    return (type_pref << 24) + (local_pref << 8) + (256 - component_id);
}
```

*TURN (Traversal Using Relays around NAT) [RFC 5766]:*

```
When direct connection fails (symmetric NAT):

┌──────────┐                      ┌──────────┐
│  Peer A  │──────────────────────│  Peer B  │
│  (NAT)   │         ✗            │  (NAT)   │
└──────────┘    Direct fails      └──────────┘
      │                                  │
      │     ┌─────────────────┐         │
      └────▶│   TURN Server   │◀────────┘
            │  (Public IP)    │
            │  Relays media   │
            └─────────────────┘

Latency cost: +20-50ms (extra hop)
Bandwidth cost: Server pays for relay
```

== RTP/RTCP Protocols

*RTP (Real-time Transport Protocol) [RFC 3550]:*

*RTP header:*

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│V=2│P│X│  CC   │M│     PT      │       Sequence Number         │
├───┴─┴─┴───────┴─┴─────────────┴───────────────────────────────┤
│                           Timestamp                           │
├───────────────────────────────────────────────────────────────┤
│                             SSRC                              │
├───────────────────────────────────────────────────────────────┤
│                      CSRC (0-15 items)                        │
├───────────────────────────────────────────────────────────────┤
│                      Payload (audio/video)                    │
└───────────────────────────────────────────────────────────────┘
```

*Key fields:*
- *V (Version):* Always 2
- *PT (Payload Type):* Codec identifier (e.g., 111 = Opus, 96 = VP8)
- *Sequence Number:* Detects loss, reordering (16-bit, wraps)
- *Timestamp:* Media timing (90kHz for video, 48kHz for audio)
- *SSRC:* Synchronization source identifier (random 32-bit)

*RTCP (RTP Control Protocol) [RFC 3550]:*

*RTCP packet types:*

| Type | Name | Purpose |
|------|------|---------|
| 200 | SR (Sender Report) | Sender statistics, NTP/RTP timestamp mapping |
| 201 | RR (Receiver Report) | Receiver statistics per SSRC |
| 202 | SDES | Source description (CNAME) |
| 203 | BYE | Leave notification |
| 205 | RTPFB | Transport-layer feedback (NACK, TMMBR) |
| 206 | PSFB | Payload-specific feedback (PLI, FIR, REMB) |

*Receiver Report structure:*

```cpp
struct ReceiverReport {
    uint8_t fraction_lost;      // Loss fraction since last RR (0-255)
    int32_t cumulative_lost;    // Total packets lost (24-bit signed)
    uint32_t ext_highest_seq;   // Extended highest sequence received
    uint32_t jitter;            // Interarrival jitter (timestamp units)
    uint32_t lsr;               // Last SR timestamp (middle 32 bits of NTP)
    uint32_t dlsr;              // Delay since last SR (1/65536 seconds)
};

// RTT calculation at sender
double calculate_rtt(uint32_t lsr, uint32_t dlsr, uint32_t now_ntp) {
    if (lsr == 0) return -1;  // No SR received yet
    return (now_ntp - lsr - dlsr) / 65536.0;  // Seconds
}
```

*Jitter calculation:*

```cpp
// RFC 3550 jitter estimation
void update_jitter(uint32_t rtp_timestamp, uint32_t arrival_time) {
    int32_t transit = arrival_time - rtp_timestamp;
    int32_t delta = abs(transit - last_transit);

    // Exponential moving average
    jitter = jitter + (delta - jitter) / 16;

    last_transit = transit;
}
```

*Feedback mechanisms:*

```cpp
// NACK: Request retransmission of lost packets
struct RTCP_NACK {
    uint16_t pid;   // Packet ID (seq number of first lost)
    uint16_t blp;   // Bitmask of following 16 packets (1 = lost)
};

// PLI (Picture Loss Indication): Request keyframe
// FIR (Full Intra Request): Force keyframe with seq number

// REMB (Receiver Estimated Maximum Bitrate)
struct RTCP_REMB {
    uint32_t ssrc;           // Media sender SSRC
    uint8_t num_ssrc;
    uint32_t bitrate_bps;    // Estimated available bandwidth
};
```

== Jitter Buffers

*Jitter buffer absorbs network timing variations:*

```
Network arrival:
  │ │   │││     │  │    ││
──┴─┴───┴┴┴─────┴──┴────┴┴──────▶ time
  Variable inter-packet gaps

After jitter buffer:
  │  │  │  │  │  │  │  │  │  │
──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──▶ time
  Regular playout interval
```

*Adaptive jitter buffer:*

```cpp
class AdaptiveJitterBuffer {
    deque<Packet> buffer;
    uint32_t target_delay_ms = 50;
    double jitter_estimate = 0;

public:
    void receive_packet(Packet pkt, uint64_t arrival_time) {
        // Update jitter estimate
        if (!buffer.empty()) {
            double expected_interval = (pkt.timestamp - last_timestamp)
                                       / clock_rate * 1000.0;
            double actual_interval = arrival_time - last_arrival_time;
            double jitter_sample = abs(actual_interval - expected_interval);

            jitter_estimate = 0.9 * jitter_estimate + 0.1 * jitter_sample;
        }

        // Adapt target delay (typically 2-4x jitter)
        target_delay_ms = max(20.0, 3 * jitter_estimate);
        target_delay_ms = min(target_delay_ms, 500.0);  // Cap at 500ms

        // Insert packet in sequence order
        insert_sorted(buffer, pkt);

        last_timestamp = pkt.timestamp;
        last_arrival_time = arrival_time;
    }

    optional<Packet> get_next_packet(uint64_t playout_time) {
        if (buffer.empty()) return nullopt;

        Packet& front = buffer.front();
        uint64_t packet_playout = front.arrival_time + target_delay_ms;

        if (playout_time >= packet_playout) {
            Packet pkt = std::move(buffer.front());
            buffer.pop_front();
            return pkt;
        }

        return nullopt;  // Not ready yet
    }
};
```

*Trade-offs:*
- Larger buffer: Less underruns, higher latency
- Smaller buffer: Lower latency, more concealment needed
- Typical ranges: 20-200ms (voice), 50-500ms (video)

== Congestion Control for Real-Time

*GCC (Google Congestion Control) [draft-ietf-rmcat-gcc]:*

```cpp
// Delay-based controller
class GCC_DelayController {
    double estimated_rate_bps;
    double threshold = 12.5;  // ms (adaptive)
    OveruseDetector detector;

public:
    void on_frame_group(vector<Packet> frames) {
        // Calculate inter-arrival delta
        double delta_time = frames.back().arrival - frames.front().arrival;
        double delta_send = frames.back().send_time - frames.front().send_time;
        double delay_delta = delta_time - delta_send;

        // Kalman filter for noise estimation
        auto [signal, state] = detector.detect(delay_delta);

        switch (state) {
            case OVERUSE:
                estimated_rate_bps *= 0.85;  // Decrease
                threshold = max(6.0, threshold - 1.0);
                break;
            case UNDERUSE:
                estimated_rate_bps *= 1.05;  // Increase
                break;
            case NORMAL:
                estimated_rate_bps *= 1.05;  // Additive increase
                threshold = min(600.0, threshold + 0.5);
                break;
        }
    }
};

// Loss-based controller (REMB feedback)
class GCC_LossController {
public:
    double update(double current_rate, double loss_fraction) {
        if (loss_fraction < 0.02) {
            // Less than 2% loss: increase
            return current_rate * 1.08;
        } else if (loss_fraction > 0.10) {
            // More than 10% loss: decrease
            return current_rate * (1.0 - 0.5 * loss_fraction);
        }
        return current_rate;
    }
};
```

*SCReAM (Self-Clocked Rate Adaptation for Multimedia) [RFC 8298]:*

- Self-clocked: Uses receive timestamps, no explicit feedback needed
- CWND-based: Similar to TCP, but for RTP
- ECN-aware: Uses ECN marks for congestion signal

*BBR for WebRTC:*
- Experimental support in Chrome
- Better performance on lossy networks
- Lower latency than loss-based algorithms

== SRT (Secure Reliable Transport)

*SRT provides reliable, encrypted media transport [Haivision]:*

*Features:*
- ARQ (Automatic Repeat reQuest) for packet recovery
- AES-128/256 encryption
- Firewall traversal (similar to ICE)
- Low latency: 120ms-8000ms configurable

*SRT handshake:*

```
Caller                           Listener
  │                                  │
  │  Induction Request              │
  ├─────────────────────────────────▶
  │                                  │
  │  Induction Response (cookie)    │
  │◀─────────────────────────────────│
  │                                  │
  │  Conclusion Request (cookie)    │
  ├─────────────────────────────────▶
  │                                  │
  │  Conclusion Response            │
  │◀─────────────────────────────────│
  │                                  │
  │  Data exchange                  │
  │◀════════════════════════════════▶│
```

*ARQ mechanism:*

```cpp
// SRT uses selective repeat ARQ
struct SRT_LossReport {
    uint32_t first_lost_seq;
    uint32_t last_lost_seq;
};

// Sender maintains buffer of packets for retransmission
// Latency setting determines buffer size
// Too Low Latency Drop: If latency exceeded, drop packet

void srt_process_loss_report(SRT_LossReport report) {
    for (uint32_t seq = report.first_lost_seq;
         seq <= report.last_lost_seq; seq++) {

        Packet* pkt = send_buffer.find(seq);
        if (pkt && !pkt->too_late()) {
            retransmit(pkt);
            stats.retransmitted++;
        } else {
            stats.dropped++;  // Too late to recover
        }
    }
}
```

*SRT vs other protocols:*

| Protocol | Latency | Reliability | Encryption | Use Case |
|----------|---------|-------------|------------|----------|
| RTP/UDP | 50-150ms | None | Optional (SRTP) | Interactive |
| SRT | 120ms-8s | ARQ | AES | Contribution |
| RIST | 100ms-5s | ARQ | DTLS | Broadcast |
| WebRTC | 100-300ms | NACK/FEC | DTLS/SRTP | Browser |

== Media Streaming Protocols

*Adaptive Bitrate Streaming (ABR):*

```
┌─────────────────────────────────────────────────────────────┐
│  Origin Server                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Source Video → Encoder → Multiple bitrates          │   │
│  │                                                       │   │
│  │  video_240p.m4s  (500 kbps)                         │   │
│  │  video_480p.m4s  (1.5 Mbps)                         │   │
│  │  video_720p.m4s  (3 Mbps)                           │   │
│  │  video_1080p.m4s (6 Mbps)                           │   │
│  │  video_4k.m4s    (15 Mbps)                          │   │
│  │                                                       │   │
│  │  manifest.mpd (DASH) or playlist.m3u8 (HLS)         │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
               ┌─────────────────────────┐
               │         CDN             │
               └─────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
    ┌─────────┐        ┌─────────┐        ┌─────────┐
    │ Player  │        │ Player  │        │ Player  │
    │ (ABR)   │        │ (ABR)   │        │ (ABR)   │
    └─────────┘        └─────────┘        └─────────┘
```

*HLS (HTTP Live Streaming) [Apple]:*

```m3u8
#EXTM3U
#EXT-X-VERSION:4
#EXT-X-STREAM-INF:BANDWIDTH=500000,RESOLUTION=426x240
video_240p.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=1500000,RESOLUTION=854x480
video_480p.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=3000000,RESOLUTION=1280x720
video_720p.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=6000000,RESOLUTION=1920x1080
video_1080p.m3u8

# Media playlist (video_720p.m3u8)
#EXTM3U
#EXT-X-VERSION:4
#EXT-X-TARGETDURATION:6
#EXT-X-MEDIA-SEQUENCE:0
#EXTINF:6.000,
segment_0.ts
#EXTINF:6.000,
segment_1.ts
#EXTINF:6.000,
segment_2.ts
```

*DASH (Dynamic Adaptive Streaming over HTTP) [MPEG]:*

```xml
<?xml version="1.0" encoding="UTF-8"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011"
     type="dynamic"
     minimumUpdatePeriod="PT2S"
     minBufferTime="PT4S">
  <Period>
    <AdaptationSet mimeType="video/mp4" segmentAlignment="true">
      <Representation id="720p" bandwidth="3000000" width="1280" height="720">
        <SegmentTemplate media="video_720p_$Number$.m4s"
                        initialization="video_720p_init.mp4"
                        duration="6000" timescale="1000"/>
      </Representation>
      <Representation id="1080p" bandwidth="6000000" width="1920" height="1080">
        <SegmentTemplate media="video_1080p_$Number$.m4s"
                        initialization="video_1080p_init.mp4"
                        duration="6000" timescale="1000"/>
      </Representation>
    </AdaptationSet>
  </Period>
</MPD>
```

*Low-Latency HLS (LL-HLS):*
- Partial segments (0.33s instead of 6s)
- Blocking playlist reloads
- Latency: 2-5 seconds (vs 15-30s traditional)

*Low-Latency DASH (LL-DASH):*
- CMAF (Common Media Application Format) chunks
- Chunked transfer encoding
- Latency: 2-5 seconds

*ABR algorithm (buffer-based):*

```cpp
class BufferBasedABR {
    double buffer_level_s;
    double reservoir = 5.0;   // Minimum buffer
    double cushion = 10.0;    // Target buffer
    vector<int> bitrates = {500, 1500, 3000, 6000};  // kbps

public:
    int select_bitrate(double throughput_kbps) {
        // BBA (Buffer-Based Adaptation)
        if (buffer_level_s < reservoir) {
            // Emergency: pick lowest
            return bitrates[0];
        }

        if (buffer_level_s > cushion) {
            // Comfortable: can increase
            double rate_upper = throughput_kbps * 0.9;
            return select_max_below(rate_upper);
        }

        // Linear interpolation between reservoir and cushion
        double ratio = (buffer_level_s - reservoir) / (cushion - reservoir);
        int max_idx = static_cast<int>(ratio * (bitrates.size() - 1));
        return bitrates[max_idx];
    }
};
```

== Performance Metrics

*Key metrics for real-time media:*

```cpp
struct MediaQoSMetrics {
    // Latency
    double glass_to_glass_ms;     // Capture to display
    double network_rtt_ms;
    double jitter_buffer_ms;

    // Quality
    double packet_loss_percent;
    double frame_rate_fps;
    double resolution_height;
    double bitrate_kbps;

    // Stalls/interruptions
    uint32_t freeze_count;
    double freeze_duration_s;

    // Audio specific
    double mos_score;             // Mean Opinion Score (1-5)
    double audio_level_dbfs;

    // Video specific
    double psnr;                  // Peak Signal-to-Noise Ratio
    double vmaf;                  // Video Multimethod Assessment Fusion
};
```

*WebRTC getStats() API:*

```javascript
const stats = await pc.getStats();
for (const report of stats.values()) {
    if (report.type === 'inbound-rtp' && report.kind === 'video') {
        console.log('Packets received:', report.packetsReceived);
        console.log('Packets lost:', report.packetsLost);
        console.log('Jitter:', report.jitter);
        console.log('Frames decoded:', report.framesDecoded);
        console.log('Frames dropped:', report.framesDropped);
    }

    if (report.type === 'candidate-pair' && report.nominated) {
        console.log('RTT:', report.currentRoundTripTime);
        console.log('Available bandwidth:',
                    report.availableOutgoingBitrate);
    }
}
```

*Typical benchmarks:*

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Packet loss | < 0.5% | < 2% | > 5% |
| Jitter | < 30ms | < 75ms | > 100ms |
| E2E latency | < 150ms | < 300ms | > 500ms |
| Video freeze | < 1% time | < 3% | > 5% |

== References

*Primary sources:*

RFC 3550: RTP: A Transport Protocol for Real-Time Applications. Schulzrinne, H. et al. (2003).

RFC 8445: Interactive Connectivity Establishment (ICE). Keranen, A., Holmberg, C., & Rosenberg, J. (2018).

RFC 5766: Traversal Using Relays around NAT (TURN). Mahy, R., Matthews, P., & Rosenberg, J. (2010).

RFC 8298: Self-Clocked Rate Adaptation for Multimedia (SCReAM). Johansson, I. & Sarker, Z. (2017).

W3C WebRTC 1.0: Real-Time Communication Between Browsers. W3C Recommendation (2021).

Holmer, S., Lundin, H., Carlucci, G., De Cicco, L., & Mascolo, S. (2015). "A Google Congestion Control Algorithm for Real-Time Communication." IETF draft-ietf-rmcat-gcc.

Haivision (2023). SRT Protocol Technical Overview. https://www.haivision.com/resources/

Apple (2023). HTTP Live Streaming. https://developer.apple.com/streaming/

ISO/IEC 23009-1: Dynamic Adaptive Streaming over HTTP (DASH). MPEG (2022).

Yin, X., Jindal, A., Sekar, V., & Sinopoli, B. (2015). "A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP." SIGCOMM '15.

Cardwell, N. et al. (2017). "BBR: Congestion-Based Congestion Control." ACM Queue 14(5): 20-53.

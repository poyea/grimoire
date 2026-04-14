= Wireless Protocols

Modern wireless networking encompasses WiFi (802.11), cellular (4G/5G), and emerging technologies like mmWave. Understanding PHY/MAC layer behavior is essential for optimizing wireless application performance.

*See also:* Transport Layer (for TCP over wireless), Application Protocols (for mobile optimization), Observability (for wireless monitoring)

== WiFi 6 (802.11ax) Architecture

*802.11ax objectives [IEEE 802.11ax-2021]:*
- 4x throughput improvement in dense environments
- Better power efficiency for IoT devices
- Improved outdoor performance
- Backward compatibility with 802.11a/b/g/n/ac

*Key technologies:*

*1. OFDMA (Orthogonal Frequency Division Multiple Access):*

```
Traditional (802.11ac):          OFDMA (802.11ax):
┌────────────────────────┐       ┌────────────────────────┐
│     User A (full BW)   │       │ User A │ B │ C │ D │   │
├────────────────────────┤       │        │   │   │   │   │
│     User B (full BW)   │       │   (Resource Units)     │
├────────────────────────┤       │                        │
│     User C (full BW)   │       │  Simultaneous TX!      │
└────────────────────────┘       └────────────────────────┘
Time-division (sequential)       Frequency-division (parallel)
```

*Resource Unit (RU) sizes:*
- 26-tone: 2 MHz (IoT, low throughput)
- 52-tone: 4 MHz
- 106-tone: 8 MHz
- 242-tone: 20 MHz
- 484-tone: 40 MHz
- 996-tone: 80 MHz
- 2x996-tone: 160 MHz

*2. MU-MIMO improvements:*
- Uplink MU-MIMO (new in 802.11ax)
- Up to 8 spatial streams
- Trigger-based transmission for coordinated UL

*3. 1024-QAM modulation:*
- 802.11ac: 256-QAM (8 bits/symbol)
- 802.11ax: 1024-QAM (10 bits/symbol)
- 25% throughput increase (ideal conditions)
- Requires SNR > 35 dB

*4. BSS Coloring:*
- Identifies overlapping BSSs
- Allows spatial reuse (transmit during neighbor's transmission)
- 6-bit color in PHY header

*Performance characteristics:*

| Metric | 802.11ac | 802.11ax |
|--------|----------|----------|
| Max data rate | 6.9 Gbps | 9.6 Gbps |
| Subcarrier spacing | 312.5 kHz | 78.125 kHz |
| Symbol duration | 3.2 μs | 12.8 μs |
| Guard interval | 0.4/0.8 μs | 0.8/1.6/3.2 μs |
| Dense deployment | Poor | 4x better |

== WiFi 7 (802.11be) - Extremely High Throughput

*802.11be (WiFi 7) targets [IEEE P802.11be]:*
- 46 Gbps peak throughput
- \< 5ms latency for AR/VR
- 320 MHz channels
- Multi-Link Operation (MLO)

*Key innovations:*

*1. 320 MHz channels:*

```
6 GHz band (WiFi 6E/7):
┌────────────────────────────────────────────────────────────┐
│  5.925 GHz                                    7.125 GHz    │
│  ├──320 MHz──┼──320 MHz──┼──320 MHz──┼──320 MHz──┤       │
│                                                            │
│  59 x 20 MHz channels available                           │
└────────────────────────────────────────────────────────────┘
```

*2. Multi-Link Operation (MLO):*

```
       ┌─────────────────────────────────────────┐
       │              WiFi 7 AP                   │
       │   ┌────────┐ ┌────────┐ ┌────────┐     │
       │   │ 2.4GHz │ │  5GHz  │ │  6GHz  │     │
       │   └───┬────┘ └───┬────┘ └───┬────┘     │
       └───────┼──────────┼──────────┼──────────┘
               │          │          │
               └──────────┼──────────┘
                          │  MLO aggregation
                          ▼
              ┌───────────────────────┐
              │     WiFi 7 Client     │
              │   (single MAC addr)   │
              └───────────────────────┘
```

*MLO modes:*
- *Simultaneous TX/RX (STR):* Different links, same time
- *Enhanced Multi-Link Single Radio (eMLSR):* Fast link switching
- *Non-STR:* Alternating links

*3. 4096-QAM:*
- 12 bits per symbol (vs 10 for 1024-QAM)
- 20% throughput increase
- Requires SNR > 42 dB (very short range)

*4. Preamble puncturing:*
- Skip interfered subcarriers
- Use rest of wide channel
- Improves spectrum efficiency

*Latency improvements:*
- Target Wake Time (TWT) refinements
- Restricted TWT for time-sensitive apps
- Multi-link low latency: \< 2ms (gaming, AR/VR)

== 3G UMTS / HSPA Architecture

*3G* (UMTS — Universal Mobile Telecommunications System, 3GPP Release 99, 2000) was the first fully packet-switched mobile data standard.

*Air interface: W-CDMA (Wideband CDMA)*
- 5 MHz channel; all users share the same frequency, separated by orthogonal spreading codes (factor 4–512)
- Rake receiver: coherently combines up to N multipath copies
- Closed-loop power control at 1500 Hz — essential to counter the near-far problem

*Architecture:*

```
UE ──── UTRAN ───────────────────────── CN (Core Network)
        ┌──────────┐                    ┌──────────────────────────┐
        │  Node B  │                    │ MSC  (circuit-switched)  │──▶ PSTN
        │(base stn)│──── RNC ──────────│ SGSN (packet-switched)   │
        └──────────┘  (radio network   │ GGSN (internet gateway)  │──▶ IP
                       controller)     └──────────────────────────┘
```

*HSPA evolution (3.5G → 3.75G):*

#table(
  columns: (auto, auto, auto, auto),
  [*Standard*], [*3GPP Rel.*], [*Peak DL*], [*Key Innovation*],
  [UMTS R99], [Rel. 99], [384 kbps], [W-CDMA, dedicated channels],
  [HSDPA], [Rel. 5 (2002)], [14.4 Mbps], [AMC + HARQ + 2ms TTI scheduling at Node B],
  [HSUPA], [Rel. 6 (2004)], [5.76 Mbps UL], [Enhanced uplink dedicated channel],
  [HSPA+], [Rel. 7/8 (2008)], [42 Mbps / 84 Mbps], [64-QAM, 2×2 MIMO, dual-carrier],
)

*HSDPA key innovations (why it was fast for its era):*
- *AMC:* Node B selects modulation/coding scheme per-UE every 2ms based on CQI feedback
- *HARQ (chase combining):* failed transmissions are stored and soft-combined, not discarded — effective SNR accumulates across retransmits
- *Moved scheduler to Node B:* 3G RNC was too far (80-100ms RTT) to react to fast channel changes; 2ms TTI at Node B enabled channel-aware scheduling

== 4G LTE / LTE-Advanced Architecture

*LTE* (Long Term Evolution, 3GPP Rel. 8, 2008) is a clean-break architecture from 3G:

- All-IP: voice over LTE (VoLTE) via SIP/RTP, no circuit-switched fallback
- Flat RAN: eNodeB handles RRC, PDCP, RLC, MAC, PHY — RNC eliminated
- Air interface: OFDMA (DL) / SC-FDMA (UL) — replaced CDMA entirely
- Round-trip latency: ~10ms (vs 80-100ms UMTS)

*LTE network architecture (EPC + E-UTRAN):*

```
┌───────────────────────────────────────────────────────────────┐
│                   EPC (Evolved Packet Core)                    │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────────────┐   │
│  │ MME  │  │  SGW │  │  PGW │  │ HSS  │  │ PCRF (QoS)   │   │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──────┘  └──────────────┘   │
│     │S1-MME   │S1-U      │SGi (internet)                      │
└─────┼─────────┼──────────┼────────────────────────────────────┘
      │         │          │
      └────┬────┘          │
           │ S1 interface  │
┌──────────┴───────────────────────────────────────────────────┐
│                       E-UTRAN                                  │
│  ┌──────────────────┐  X2  ┌──────────────────┐              │
│  │    eNodeB        │──────│    eNodeB        │              │
│  │  RRC, PDCP, RLC  │      │  RRC, PDCP, RLC  │              │
│  │  MAC, PHY        │      │  MAC, PHY        │              │
│  └────────┬─────────┘      └──────────────────┘              │
└───────────┼──────────────────────────────────────────────────┘
            │ LTE-Uu air interface
       ┌────┴────┐
       │   UE   │
       └─────────┘
```

*Core functions:*
- *MME:* Authentication (AKA protocol), attach/detach, handover signaling, paging
- *SGW:* User-plane mobility anchor during intra-LTE handover; per-UE bearer tunnels (GTP-U)
- *PGW:* IP address assignment, internet access, DPI, policy enforcement
- *HSS:* Subscriber database (IMSI, encryption keys, service profiles — equivalent to HLR+AuC)

*LTE air interface — OFDMA resource grid:*

```
Frequency
  ▲
  │  RB  RB  RB  RB  ...
  │ ┌──┬──┬──┬──┬──┐
  │ │  │  │  │  │  │  ← 12 subcarriers × 15 kHz = 180 kHz per RB
  │ │  │  │  │  │  │
  │ └──┴──┴──┴──┴──┘
  │   ←── 0.5 ms slot ──→     2 slots = 1 TTI (1 ms subframe)
  └──────────────────────────▶ Time

Channel BW:  1.4   3    5   10   15   20  MHz
# of RBs:     6   15   25   50   75  100
Peak DL:   ~10  ~21  ~37  ~75 ~112 ~150 Mbps  (SISO, 64-QAM, R=1)
```

*Downlink scheduler (runs every 1ms TTI at eNodeB):*
```cpp
// Proportional Fair scheduler — balance throughput and fairness
// CQI = channel quality indicator reported by UE every 5ms (1–15)
void schedule_tti(vector<UE>& active_ues) {
    for (UE& ue : active_ues) {
        // Metric = instantaneous rate / average throughput
        // → favors UEs with good channel AND starved UEs
        ue.pf_metric = (double)cqi_to_mbps(ue.cqi)
                       / ue.avg_throughput_mbps;
    }
    sort(active_ues, [](auto& a, auto& b) {
        return a.pf_metric > b.pf_metric;
    });
    assign_resource_blocks(active_ues);
}
```

*HARQ (Hybrid ARQ) in LTE:*
- 8 parallel HARQ processes (hide 8ms HARQ RTT with pipelining)
- *Chase combining:* retransmit same bits, combine soft values → +3 dB per retransmit
- *Incremental Redundancy (IR):* retransmit different parity bits → each retransmit adds coding gain

*LTE-Advanced (Rel. 10, 2011) enhancements:*

#table(
  columns: (auto, auto, auto),
  [*Feature*], [*Peak Gain*], [*Mechanism*],
  [Carrier Aggregation (CA)], [Up to 5×20 MHz = 1 Gbps DL], [Combine up to 5 component carriers (CC)],
  [DL MIMO 8×8], [8x spatial streams], [8 TX antennas at eNodeB, codebook precoding],
  [Heterogeneous Networks (HetNet)], [10x area capacity], [Macro + small cells + femtocells],
  [eICIC], [30-40% cell-edge gain], [Almost Blank Subframes reduce macro interference on small cells],
  [Relay nodes], [Coverage extension], [Wireless backhaul relay, no fibre needed],
)

*VoLTE (Voice over LTE, IMS-based):*
- Dedicated GBR bearer (QCI 1): guaranteed bit rate, max 100ms one-way delay
- Codec: AMR-WB (50–7000 Hz, 23.85 kbps) — wideband "HD Voice"
- Setup: SIP INVITE over LTE → ~2s (vs 7s circuit-switched 3G)

*Hands-on: LTE signal analysis:*

```bash
# ModemManager — check signal quality
mmcli -m 0 --signal-get
# Output:
#   rsrp: -85.00 dBm    (> -90 good,  < -110 poor)
#   rsrq: -12.00 dB     (> -10 good,  < -20 poor)
#   sinr:  22.00 dB     (> 20 excellent)
#   rssi: -65.00 dBm    (total received power)

# Current access tech and band
mmcli -m 0 | grep "access tech"

# Force 4G-only (useful for testing)
mmcli -m 0 --set-allowed-modes=4G

# AT commands (raw modem, e.g. Qualcomm/Sierra)
picocom -b 115200 /dev/ttyUSB2
AT+CEREG?       # registration state (0,1 = registered home)
AT+QNWINFO      # tech, band, ARFCN (e.g. LTE BAND 3, 1600)
AT+QRSRP        # RSRP per carrier
AT+QCAINFO      # carrier aggregation: PCC + SCCs
```

*LTE signal parameters guide:*

#table(
  columns: (auto, auto, auto, auto),
  [*Metric*], [*Excellent*], [*Good*], [*Poor (edge)*],
  [RSRP], [$>$ -80 dBm], [-80 to -100 dBm], [$<$ -110 dBm],
  [RSRQ], [$>$ -10 dB], [-10 to -15 dB], [$<$ -20 dB],
  [SINR], [$>$ 20 dB], [10 to 20 dB], [$<$ 0 dB],
  [CQI], [12–15], [7–11], [1–6],
)

*LTE → 5G comparison:*

#table(
  columns: (auto, auto, auto),
  [*Parameter*], [*LTE-A (Rel. 13)*], [*5G NR (Rel. 16)*],
  [Peak DL], [1 Gbps], [20 Gbps],
  [User-plane latency], [10-30 ms], [1-4 ms (URLLC: $<$ 1 ms)],
  [Spectrum], [Sub-6 GHz], [Sub-6 GHz + mmWave (24-52 GHz)],
  [Channel BW], [Up to 100 MHz (5-CC CA)], [Up to 400 MHz (mmWave)],
  [MIMO], [8×8 DL], [Massive MIMO 32×32 to 64×64+],
  [Subcarrier spacing], [Fixed 15 kHz], [Flexible: 15/30/60/120/240 kHz],
  [Core], [EPC (monolithic, per-function VMs)], [5GC (service-based, cloud-native microservices)],
  [Network slicing], [No], [Yes — E2E QoS slices per tenant/service],
  [Device density], [~10K/km²], [1M/km² (mMTC)],
)

== 5G Architecture

*5G (NR - New Radio) provides [3GPP Release 15+]:*
- eMBB: Enhanced Mobile Broadband (up to 20 Gbps)
- URLLC: Ultra-Reliable Low-Latency Communications (1ms, 99.999%)
- mMTC: Massive Machine Type Communications (1M devices/km²)

*5G network architecture:*

```
┌─────────────────────────────────────────────────────────────┐
│                        5G Core (5GC)                         │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐          │
│  │ AMF │ │ SMF │ │ UPF │ │ PCF │ │ UDM │ │ NRF │          │
│  └──┬──┘ └──┬──┘ └──┬──┘ └─────┘ └─────┘ └─────┘          │
│     │       │       │                                       │
│     └───────┴───────┴─── Service-Based Architecture (SBA)  │
├─────────────────────────────────────────────────────────────┤
│                     NG Interface                             │
├─────────────────────────────────────────────────────────────┤
│                    RAN (gNB - gNodeB)                       │
│  ┌────────────────────┐    ┌────────────────────┐          │
│  │    gNB-CU          │────│    gNB-DU          │          │
│  │ (Central Unit)     │    │ (Distributed Unit) │          │
│  │ - RRC, PDCP        │    │ - RLC, MAC, PHY    │          │
│  └────────────────────┘    └────────────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                         UE (User Equipment)                  │
└─────────────────────────────────────────────────────────────┘
```

*Core network functions:*
- *AMF (Access and Mobility Management):* Authentication, mobility
- *SMF (Session Management):* PDU session establishment
- *UPF (User Plane Function):* Packet routing, QoS enforcement
- *PCF (Policy Control):* Policy decisions
- *UDM (Unified Data Management):* Subscriber data

*5G NR frequency bands:*

| Range | Frequency | Bandwidth | Use Case |
|-------|-----------|-----------|----------|
| FR1 (Sub-6 GHz) | 410 MHz - 7.125 GHz | 5-100 MHz | Coverage, capacity |
| FR2 (mmWave) | 24.25 - 52.6 GHz | 50-400 MHz | Ultra-high capacity |
| FR2-2 | 52.6 - 71 GHz | Up to 2 GHz | Future expansion |

*5G NR physical layer:*
- Flexible numerology: 15/30/60/120/240 kHz subcarrier spacing
- Slot duration: 1ms (15 kHz) to 62.5 μs (240 kHz)
- Mini-slot: 2-7 OFDM symbols (URLLC low latency)

== mmWave Technology

*Millimeter wave (24-100 GHz) characteristics:*

*Advantages:*
- Massive bandwidth: 400 MHz - 2 GHz channels
- High capacity: Multi-Gbps per user
- Spectrum availability: Less congested

*Challenges:*
- High path loss: 20 dB worse than 3.5 GHz at same distance
- Blockage: Human body attenuates 20-35 dB
- Rain fade: 10 dB/km at 60 GHz (heavy rain)
- Limited range: 100-300m typical

*Path loss formula:*

```
Free space path loss (dB) = 20×log₁₀(d) + 20×log₁₀(f) + 20×log₁₀(4π/c)

At 28 GHz vs 3.5 GHz (same distance):
ΔPL = 20×log₁₀(28/3.5) = 18 dB additional loss
```

*Beamforming (essential for mmWave):*

```
     Antenna Array
    ┌─┬─┬─┬─┬─┬─┬─┬─┐
    │ │ │ │ │ │ │ │ │  64-256 elements typical
    └─┴─┴─┴─┴─┴─┴─┴─┘
          │
          │  Phase-shifted signals
          │  create directional beam
          ▼
        /    \
       /      \
      /   UE   \
     /    ●     \
    ────────────────
    Beam gain: 20-30 dB
```

*Beam management:*
- *Beam sweeping:* AP transmits reference signals in multiple directions
- *Beam tracking:* UE reports best beam index
- *Beam recovery:* Fast fallback when beam blocked

*Practical mmWave deployment:*
- Dense small cells: 100-200m inter-site distance
- Indoor-outdoor coordination
- Mesh backhaul between cells
- Use case: Stadiums, urban cores, enterprise

== Spectrum Management

*Spectrum allocation fundamentals:*

*Licensed spectrum:*
- Exclusive use (purchased at auction)
- Guaranteed interference-free
- Examples: Cellular bands, some 5G bands

*Unlicensed spectrum:*
- Shared access (regulatory rules)
- No interference protection
- Examples: 2.4 GHz, 5 GHz, 6 GHz (WiFi)

*Shared spectrum (CBRS - 3.5 GHz):*

```
Priority tiers:
┌─────────────────────────────────────┐
│ Incumbent (Navy radar, satellites)  │ ← Highest priority
├─────────────────────────────────────┤
│ PAL (Priority Access License)       │ ← Purchased at auction
├─────────────────────────────────────┤
│ GAA (General Authorized Access)     │ ← Open access
└─────────────────────────────────────┘

SAS (Spectrum Access System) coordinates access
```

*Dynamic Spectrum Sharing (DSS):*

```
Time/Frequency grid:
┌────┬────┬────┬────┬────┬────┬────┐
│ 4G │ 5G │ 4G │ 5G │ 5G │ 4G │ 5G │  ← Subframe allocation
├────┼────┼────┼────┼────┼────┼────┤
│    Shared carrier (e.g., 20 MHz)   │
└────┴────┴────┴────┴────┴────┴────┘

Dynamic allocation based on traffic demand
```

*Coexistence mechanisms:*

*1. Listen Before Talk (LBT):*
```cpp
// 802.11 CSMA/CA
while (channel_busy()) {
    wait(random_backoff());
}
transmit();
```

*2. LAA (Licensed Assisted Access - LTE in unlicensed):*
- LTE uses LBT in 5 GHz
- Coexists with WiFi
- Carrier aggregation with licensed anchor

*3. NR-U (5G NR Unlicensed):*
- 5 GHz and 6 GHz operation
- Enhanced LBT mechanisms
- Standalone or anchored mode

== Wireless Performance Optimization

*Latency components:*

```
┌──────────────────────────────────────────────────────────────┐
│ Air interface latency (WiFi)                                  │
│ ┌─────────┬──────────┬────────────┬───────────┬───────────┐ │
│ │ DIFS    │ Backoff  │ TX time    │ SIFS      │ ACK       │ │
│ │ 34 μs   │ 0-1ms    │ 50-500 μs  │ 16 μs     │ 40 μs     │ │
│ └─────────┴──────────┴────────────┴───────────┴───────────┘ │
│ Total: 150 μs - 2 ms (load dependent)                        │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ 5G NR latency (user plane)                                   │
│ ┌───────────────────────────────────────────────────────────┐│
│ │ Processing │  TX  │ Propagation │  RX  │ Processing      ││
│ │  0.2 ms   │ TTI  │   < 0.1 ms  │ TTI  │   0.2 ms        ││
│ └───────────────────────────────────────────────────────────┘│
│ Sub-6 GHz: 4-10 ms    mmWave: 1-4 ms    URLLC: < 1 ms       │
└──────────────────────────────────────────────────────────────┘
```

*Application-level optimizations:*

*1. Reduce round trips:*
```cpp
// Bad: Multiple small requests
GET /config HTTP/1.1
GET /user HTTP/1.1
GET /data HTTP/1.1

// Good: Batch or use HTTP/2 multiplexing
// Or use QUIC for 0-RTT
```

*2. Handle variable latency:*
```cpp
// Adaptive timeout based on network conditions
class AdaptiveTimeout {
    double rtt_estimate = 100;  // Initial estimate (ms)
    double rtt_variance = 50;

public:
    void update(double measured_rtt) {
        rtt_estimate = 0.875 * rtt_estimate + 0.125 * measured_rtt;
        rtt_variance = 0.75 * rtt_variance +
                       0.25 * abs(measured_rtt - rtt_estimate);
    }

    double get_timeout() {
        return rtt_estimate + 4 * rtt_variance;
    }
};
```

*3. Prefetch aggressively:*
- Predict next resources
- Use idle time for background sync
- Cache responses locally

*WiFi roaming optimization:*

*802.11r (Fast BSS Transition):*
- Pre-authentication with target AP
- Roam time: 50ms → \< 10ms
- Critical for VoIP, real-time apps

*802.11k (Radio Resource Management):*
- AP provides neighbor reports
- Client knows where to roam

*802.11v (BSS Transition Management):*
- AP can suggest/steer clients
- Load balancing across APs

== Power Management

*WiFi power states:*

*Legacy Power Save Mode (PSM):*
```
Active ──▶ Doze (beacon interval, typically 100ms)
         ◀── Wake on TIM (Traffic Indication Map)
```

*Target Wake Time (TWT) - 802.11ax:*
```
┌─────────────────────────────────────────────────────────────┐
│  Client negotiates specific wake times with AP               │
│                                                              │
│  Time ─────────────────────────────────────────────────────▶│
│        │      │      │      │      │                        │
│  Sleep ████████████████████████████                         │
│             │      │      │      │                          │
│  Active     █      █      █      █                          │
│             ▲      ▲      ▲      ▲                          │
│            TWT    TWT    TWT    TWT                         │
│         sessions                                             │
└─────────────────────────────────────────────────────────────┘
```

*Benefits:*
- Predictable wake times (better sleep scheduling)
- Reduces contention (coordinated access)
- Battery life: 4-7x improvement for IoT devices

*5G power optimization:*
- *DRX (Discontinuous Reception):* Sleep between PDCCH monitoring
- *RRC Inactive state:* Maintain connection context, deep sleep
- *BWP (Bandwidth Part) adaptation:* Reduce active bandwidth

== References

*Primary sources:*

IEEE 802.11ax-2021: Wireless LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications Amendment 1: Enhancements for High-Efficiency WLAN. IEEE (2021).

IEEE P802.11be/D4.0: Wireless LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications Amendment: Enhancements for Extremely High Throughput (EHT). IEEE (2023).

3GPP TS 38.300: NR; NR and NG-RAN Overall Description. 3GPP Release 17 (2023).

3GPP TS 23.501: System Architecture for the 5G System. 3GPP Release 17 (2023).

Khorov, E., Kiryanov, A., Lyakhov, A., & Bianchi, G. (2019). "A Tutorial on IEEE 802.11ax High Efficiency WLANs." IEEE Communications Surveys & Tutorials 21(1): 197-216.

Narayanan, A. et al. (2020). "A First Look at Commercial 5G Performance on Smartphones." WWW '20.

Rangan, S., Rappaport, T.S., & Erkip, E. (2014). "Millimeter-Wave Cellular Wireless Networks: Potentials and Challenges." Proceedings of the IEEE 102(3): 366-385.

Qualcomm (2023). "WiFi 7: The Next Generation of Wi-Fi." Technical White Paper.

Samsung (2023). "5G NR Physical Layer Overview." Technical White Paper.

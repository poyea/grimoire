#import "../template.typ": xref

= Fieldbuses: CAN, LIN, and Deterministic Ethernet <fieldbuses>

The serial links in #xref("embedded-and-realtime", "peripherals-and-drivers", label: "Peripherals and Drivers") ($"UART"$, SPI, I2C) connect chips on one board. A *fieldbus* connects controllers across a machine, a vehicle or a factory, over metres of cable, through electrical noise, with bounded latency and defined behaviour when nodes fail. A modern car carries 70 to 150 electronic control units on a mix of CAN, CAN FD, LIN and automotive Ethernet, and a production line runs EtherCAT, PROFINET or Modbus. This chapter covers how these buses arbitrate, how they detect and contain faults, and how to prove that a message always arrives by its deadline.

*See also:* #xref("embedded-and-realtime", "peripherals-and-drivers", label: "Peripherals and Drivers"), #xref("embedded-and-realtime", "timing-analysis", label: "Timing Analysis and WCET"), #xref("networking", "link-layer", label: "Link Layer") (networking), #xref("networking", "realtime-protocols", label: "Real-Time Protocols") (networking)

== Controller Area Network

CAN was developed by Bosch for in-vehicle networking and standardised as ISO 11898. It is a multi-master broadcast bus on a twisted pair with 120 ohm termination at both ends. It has no addresses: every frame carries an *identifier* that names the *content* ("engine speed", "brake pedal position"), every node receives every frame, and hardware acceptance filters decide which frames a node cares about.

=== Dominant, Recessive, and Arbitration

The two wires (CAN_H and CAN_L) are driven differentially. A *dominant* bit (logical 0) is actively driven, and a *recessive* bit (logical 1) is the idle state. If any node drives dominant, the bus is dominant: the bus is a wired-AND.

That property gives CAN its central feature, *bitwise non-destructive arbitration*. Nodes that want to transmit start together after the bus goes idle and send their identifier most significant bit first, reading the bus back as they go. A node that sends recessive but reads dominant has lost: some other node has a lower identifier. It stops transmitting at once and becomes a receiver. The winner never notices the contest, so no bandwidth is wasted on collisions. The consequences:

- *The identifier is the priority.* Numerically lower identifiers win. The bus behaves as a fixed-priority, non-preemptive scheduler with the identifier as the priority.
- *Identifiers must be unique.* Two nodes sending the same identifier with different data both pass arbitration and then collide in the data field, causing bit errors.
- *Bus length is limited by bit time.* Every node must see each bit settle within one bit time, round trip included, so 1 Mbit/s allows about 40 m of cable and 125 kbit/s about 500 m.

=== Frame Format and Bit Stuffing

A classic CAN data frame with an 11-bit identifier (CAN 2.0A) consists of start-of-frame, identifier, RTR, control field with a 4-bit data length code, 0 to 8 data bytes, a 15-bit CRC, acknowledge slot, and end-of-frame, followed by a 3-bit interframe space. CAN 2.0B adds an extended 29-bit identifier. Receivers acknowledge by driving the ACK slot dominant. A transmitter that sees no acknowledgement treats the frame as failed, which is why a lone node on a bench retransmits forever.

CAN has no separate clock line. Receivers resynchronise on recessive-to-dominant edges, so the transmitter guarantees edges by *bit stuffing*: after five consecutive bits of the same value it inserts one bit of the opposite value, which receivers remove. Stuffing makes frame length data-dependent. The worst-case length of a frame with $s$ data bytes is

$ C = (g + 8 s + 13 + floor((g + 8 s - 1) / 4)) tau_"bit" $

where $g = 34$ for 11-bit identifiers and $g = 54$ for 29-bit identifiers (Davis et al., 2007). An 8-byte standard frame therefore takes at most 135 bit times, 270 µs at 500 kbit/s. That gives a ceiling of about 3 700 frames per second at full load, and a practical design budget near half that.

=== Bit Timing

Each bit time is divided into *time quanta* derived from the controller clock, grouped into a synchronisation segment, propagation segment, and two phase segments. The receiver samples at the *sample point* between the phase segments. Automotive practice (CiA 601 and SAE J2284) places it at 75 to 87.5 percent of the bit, late enough for propagation delay to settle. The *synchronisation jump width* bounds how far a node may stretch or shrink a bit to stay in step with the transmitter. Every node on a bus must use compatible sample points. A single node configured differently causes intermittent errors that come and go with temperature and cable length.

=== Error Detection and Fault Confinement

CAN detects five classes of error: bit errors (a transmitter reads back a value it did not send, outside arbitration), stuff errors, CRC errors, form errors (a fixed-format bit has the wrong value), and acknowledgement errors. Any node that detects an error destroys the frame for everyone by sending an *error flag*, six dominant bits that deliberately violate the stuffing rule, and the transmitter retries automatically. The residual probability of an undetected error is very low (on the order of $10^(-11)$ per message under the usual error models).

A faulty node must not be able to take the whole bus down by sending error flags forever. Each controller keeps a *transmit error counter* (TEC) and a *receive error counter* (REC), incremented by 8 on errors it probably caused and by 1 on others, and decremented on success:

#table(
  columns: 3,
  [*State*], [*Condition*], [*Behaviour*],
  [Error active], [TEC and REC at most 127], [Normal; sends active (dominant) error flags],
  [Error passive], [TEC or REC above 127], [Sends passive (recessive) error flags that cannot destroy other nodes' frames; waits extra time before retransmitting],
  [Bus off], [TEC above 255], [Disconnects from the bus; may rejoin after observing 128 sequences of 11 recessive bits],
)

A node with a broken transceiver or wrong bit rate thus quickly takes itself off the bus. Application software must decide what to do in bus-off: recover automatically (common, but it can mask a real fault), recover after a delay, or report a diagnostic trouble code and stay off.

=== CAN FD and CAN XL

Classic CAN's 8-byte payload and 1 Mbit/s ceiling became a bottleneck for firmware updates, sensor fusion and security (an authentication tag alone can take 8 bytes). *CAN FD* (ISO 11898-1:2015) keeps arbitration at the nominal rate, where the round-trip constraint applies, and then switches to a faster *data phase*, typically 2 to 5 Mbit/s and up to 8 Mbit/s with good transceivers, because only one node is transmitting. Payloads grow to 64 bytes, and stronger CRCs (17 or 21 bits) plus a stuff-bit counter fix a weakness in classic CAN's CRC coverage. *CAN XL* (ISO 11898-1:2024) goes further, with payloads up to 2048 bytes, data phases above 10 Mbit/s, and a field that can carry Ethernet frames, placing it between CAN FD and 10BASE-T1S.

=== A Transmit Path in C++

Controllers expose a small number of transmit mailboxes. If several mailboxes hold frames, the controller must send the lowest identifier first, or a high-priority message waits behind a low-priority one inside the node, an internal priority inversion that breaks the bus-level analysis. Configure the controller for identifier-priority transmit order (on bxCAN, `TXFP = 0`), and keep the software queue in priority order too:

```cpp
#include <array>
#include <cstdint>
#include <span>

struct can_frame {
    std::uint32_t id;          // 11-bit or 29-bit identifier
    bool extended;
    std::uint8_t length;       // 0..8 for classic CAN
    std::array<std::uint8_t, 8> data;
};

// Fixed-capacity queue ordered by identifier: lowest id leaves first.
template <std::size_t N>
class can_tx_queue {
public:
    bool push(const can_frame& f) {
        if (size_ == N) return false;              // overflow is a design error
        std::size_t i = size_;
        while (i > 0 && frames_[i - 1].id > f.id) {
            frames_[i] = frames_[i - 1];
            --i;
        }
        frames_[i] = f;
        ++size_;
        return true;
    }
    bool pop(can_frame& out) {
        if (size_ == 0) return false;
        out = frames_[0];
        for (std::size_t i = 1; i < size_; ++i) frames_[i - 1] = frames_[i];
        --size_;
        return true;
    }

private:
    std::array<can_frame, N> frames_{};
    std::size_t size_ = 0;
};
```

Insertion is linear in queue length, which is fine for the handful of entries a node queues, and its worst case is easy to bound for timing analysis. Both `push` and `pop` run under the driver's critical section, because the transmit-complete ISR pops while tasks push.

== Response-Time Analysis for CAN

Because arbitration is fixed-priority and frames are not preemptible once started, CAN message latency can be analysed like non-preemptive fixed-priority scheduling (Tindell, Burns and Wellings, 1995). For message $m$ with worst-case transmission time $C_m$, period $T_m$ and queuing jitter $J_m$:

$ R_m = J_m + w_m + C_m $

$ w_m = B_m + sum_(k in "hp"(m)) ceil((w_m + J_k + tau_"bit") / T_k) C_k $

$B_m$ is the longest lower-priority frame, which may have just started when $m$ was queued. The queuing delay $w_m$ is solved by fixed-point iteration as in task analysis. Davis, Burns, Bril and Lukkien (2007) showed that the original analysis was optimistic in one case: when the busy period at priority $m$ extends beyond $T_m$, a later instance of $m$ can have a longer response than the first, so the analysis must check every instance in the busy period. Commercial tools implement the corrected version.

The practical lessons are those of #xref("embedded-and-realtime", "timing-analysis", label: "Timing Analysis and WCET"): assign identifiers by deadline (deadline-monotonic), keep bus load well below 100 percent (many OEMs cap it at 50 to 70 percent), account for stuff bits, and remember that one node sending a high-priority identifier too often delays every message below it.

== LIN

LIN (Local Interconnect Network, ISO 17987) is a cheap single-wire bus for body electronics such as window motors, mirrors and seat switches, where a CAN controller and transceiver would cost too much. It runs at up to 20 kbit/s on a $"UART"$-compatible physical layer. One *commander* (master) node polls up to 16 *responder* (slave) nodes according to a static *schedule table*. The commander sends a header (break, sync byte 0x55, protected identifier), and exactly one node answers with up to 8 data bytes and a checksum. Because the schedule is static and only the commander starts transfers, LIN timing is fully deterministic and needs no arbitration analysis. Responders can even synchronise their cheap RC oscillators to the sync byte, avoiding a crystal. A LIN cluster usually hangs off a CAN or Ethernet gateway ECU.

== Automotive and Industrial Ethernet

Bandwidth for cameras, radar, displays and zonal architectures has moved the vehicle backbone to Ethernet over a single twisted pair:

#table(
  columns: 4,
  [*Standard*], [*Rate*], [*Topology*], [*Typical use*],
  [10BASE-T1S (IEEE 802.3cg-2019)], [10 Mbit/s], [Multidrop bus with PLCA], [Replacing CAN on sensor and actuator edges],
  [100BASE-T1 (802.3bw-2015)], [100 Mbit/s], [Point to point], [ECU links, diagnostics],
  [1000BASE-T1 (802.3bp-2016)], [1 Gbit/s], [Point to point], [Zonal backbone, cameras],
  [MultiGBASE-T1 (802.3ch-2020)], [2.5 to 10 Gbit/s], [Point to point], [High-resolution sensors, backbone],
)

10BASE-T1S is notable because it brings back a shared bus. *Physical layer collision avoidance* (PLCA) gives each node a transmit opportunity in turn, a round-robin scheme that bounds access latency without CAN's bit-level arbitration.

Ethernet on its own is not deterministic: switches queue frames, and a large best-effort frame can delay a control frame. *Time-Sensitive Networking* (TSN, the IEEE 802.1 family) adds the missing pieces:

- *802.1AS* (gPTP): sub-microsecond time synchronisation across the network.
- *802.1Qbv* (time-aware shaper): gate control lists open and close each egress queue on a global schedule, reserving time windows for scheduled traffic.
- *802.1Qbu* and *802.3br* (frame preemption): an express frame can interrupt a preemptable frame mid-transmission, bounding blocking to one fragment rather than one maximum-size frame.
- *802.1Qav* (credit-based shaper) for audio and video streams, and *802.1CB* for frame replication over redundant paths.

=== Industrial Fieldbuses

Factory automation evolved its own Ethernet-based buses before TSN existed:

- *EtherCAT* (IEC 61158): the controller sends one frame that passes through every node in a logical ring. Each node reads and writes its slice of the frame *on the fly* in hardware as it passes, so a cycle of hundreds of I/O points completes in tens of microseconds, with distributed clocks synchronised to well under a microsecond.
- *PROFINET IRT* and *EtherNet/IP* with CIP Sync: reserved time slots and IEEE 1588 clock synchronisation on switched Ethernet.
- *Modbus RTU*: the long-lived simple option. A single master polls slaves over RS-485, frames are delimited by 3.5 character times of silence and protected by a CRC-16. It is easy to implement and still everywhere in building automation and energy metering.

== Security on the Bus

Classic CAN has no authentication: any node can send any identifier, and the 2015 remote compromise of a Jeep Cherokee through its infotainment unit showed what that means once one networked ECU is compromised. The countermeasures:

- *AUTOSAR SecOC* appends a truncated message authentication code and a freshness value (counter or timestamp) to frames, using keys provisioned per ECU. CAN FD's larger payload makes room for the extra bytes.
- *Gateways and firewalls* separate domains (infotainment, powertrain, chassis), forwarding only whitelisted identifiers.
- *Intrusion detection* watches for frames arriving faster than their period or identifiers appearing on the wrong segment.
- *CANsec*, defined with CAN XL, offers link-layer authenticated encryption in the spirit of MACsec, and automotive Ethernet uses MACsec (IEEE 802.1AE) directly.

Message authentication is covered in #xref("cryptography-and-security", "hashing-and-macs", label: "Hashing and MACs"). The embedded-specific issues are key storage in a hardware security module and keeping verification within the frame's latency budget.

== Choosing a Bus

#table(
  columns: 2,
  [*Requirement*], [*Typical choice*],
  [Cheap, slow body electronics, one master], [LIN],
  [Robust multi-master control traffic, analysable latency], [CAN or CAN FD],
  [Larger payloads on existing CAN wiring], [CAN FD, later CAN XL],
  [Edge sensors in an Ethernet-based architecture], [10BASE-T1S],
  [Cameras, radar, backbone], [100BASE-T1 to MultiGBASE-T1 with TSN],
  [Motion control with sub-millisecond cycles], [EtherCAT or PROFINET IRT],
  [Simple polling over long RS-485 runs], [Modbus RTU],
)

== Further Reading

ISO (2015). "ISO 11898-1:2015 Road vehicles: Controller area network (CAN), Part 1: Data link layer and physical signalling." Standard.

Bosch (1991). "CAN Specification Version 2.0." Robert Bosch GmbH.

Tindell, K., Burns, A., Wellings, A. (1995). "Calculating Controller Area Network (CAN) Message Response Times." Control Engineering Practice.

Davis, R., Burns, A., Bril, R., Lukkien, J. (2007). "Controller Area Network (CAN) Schedulability Analysis: Refuted, Revisited and Revised." Real-Time Systems.

ISO (2016). "ISO 17987: Road vehicles: Local Interconnect Network (LIN)." Standard.

IEEE (2019). "IEEE 802.3cg-2019: Physical Layers and Management Parameters for 10 Mb/s Operation over Single Balanced Twisted-pair Cabling." Standard.

Miller, C., Valasek, C. (2015). "Remote Exploitation of an Unaltered Passenger Vehicle." Black Hat USA.

Jansen, D., Buttner, H. (2004). "Real-time Ethernet: the EtherCAT Solution." Computing and Control Engineering.

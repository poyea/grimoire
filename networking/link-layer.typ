= Link Layer & Hardware

The link layer handles physical transmission of bits over a medium. For Ethernet (dominant LAN technology), this includes frame structure, MAC addressing, and hardware interrupt handling.

*See also:* Internet Layer (for IP over Ethernet), Zero-Copy (for DMA mechanics), Kernel Bypass (for avoiding kernel packet processing)

== Ethernet Frame Structure

*Ethernet II frame format (RFC 894, DIX Ethernet):*

```
┌──────────┬──────────┬──────┬─────────┬─────┬─────┐
│ Preamble │   Dest   │ Src  │  Type   │ Pay │ FCS │
│  8 bytes │ MAC (6B) │ MAC  │ (2B)    │load │ 4B  │
└──────────┴──────────┴──────┴─────────┴─────┴─────┘
            \_____ Ethernet Header _____/
```

*Fields:*
- *Preamble:* 7 bytes of `0x55` + 1 byte SFD (`0xD5`) for synchronization
- *Destination MAC:* 48-bit hardware address (e.g., `00:1A:2B:3C:4D:5E`)
- *Source MAC:* 48-bit sender address
- *EtherType:* Protocol identifier (`0x0800` = IPv4, `0x86DD` = IPv6, `0x0806` = ARP)
- *Payload:* 46-1500 bytes (MTU = 1500 bytes for standard Ethernet)
- *FCS (Frame Check Sequence):* CRC-32 for error detection

*Frame size:*
- Minimum: 64 bytes (including 18-byte header + FCS)
- Maximum: 1518 bytes (excluding preamble)
- Jumbo frames: up to 9000 bytes (non-standard, datacenter use)

*Performance implications:*
- Small packets: header overhead = 18/64 = 28% for minimum frame
- Large packets: header overhead = 18/1518 = 1.2%
- *PPS (Packets Per Second) vs throughput:* 10 Gbps link theoretical max = 14.88 million packets/sec (64-byte frames) or 812K packets/sec (1518-byte frames)

*Calculation:* $"PPS" = "Bandwidth" / (("Frame size" + "IFG") × 8)$ where IFG (Inter-Frame Gap) = 12 bytes

== Network Interface Card (NIC)

*Hardware components:*

```
┌────────────────────────────────────────┐
│              NIC                        │
│  ┌──────────┐    ┌─────────────┐      │
│  │   MAC    │───▶│   DMA       │      │
│  │ Controller│    │   Engine    │      │
│  └──────────┘    └──────┬──────┘      │
│       │                  │              │
│   PHY chip          Ring Buffers       │
│       │              in DRAM            │
└───────┼──────────────────┼──────────────┘
        │                  │
     Ethernet           PCIe bus
      cable         to main memory
```

*NIC functions:*
1. *Frame reception:* PHY decodes physical signal → MAC validates FCS → DMA writes to RAM
2. *Frame transmission:* DMA reads from RAM → MAC adds FCS → PHY encodes to signal
3. *Hardware offload:* Checksum calculation (TCP/UDP/IP), TSO (TCP Segmentation Offload), LRO (Large Receive Offload)

*PCIe bandwidth:*
- PCIe 3.0 x8: ~8 GB/s per direction (64 Gbps)
- PCIe 4.0 x16: ~32 GB/s per direction (256 Gbps)
- Modern NICs (100G, 200G) require PCIe 4.0 to avoid bottleneck

== Hardware Interrupts

*Legacy interrupt processing:*

```
1. Packet arrives at NIC
2. NIC DMA writes packet to ring buffer in RAM (~200ns PCIe latency)
3. NIC triggers hardware interrupt (IRQ) (~100ns)
4. CPU context switches to interrupt handler (~1000ns)
5. Kernel processes packet, copies to socket buffer
6. Application awakened from blocking recv() (~1000ns context switch)

Total: ~2500ns = 2.5μs minimum latency
```

*Problem:* Each packet = one interrupt. At high packet rates (1M+ PPS), CPU spends 100% time in interrupt handlers [Salim & Olsson 2001].

*Interrupt coalescing (NIC feature):*

```cpp
// Delay interrupt until:
// - N packets received (e.g., 32), OR
// - T microseconds elapsed (e.g., 100μs)
// Configured via ethtool

// Example: Intel 82599 NIC registers
#define EITR_INTERVAL 0x00000014  // Interrupt throttle: 20μs
```

*Tradeoff:*
- No coalescing: Low latency (2-3μs), high CPU overhead (10-20% per 1M PPS)
- Aggressive coalescing: High latency (50-100μs), low CPU overhead (1-2%)
- Typical setting: 50-100μs for throughput, 5-10μs for latency-sensitive

*Measurement [Mogul & Ramakrishnan 1997]:*
- Interrupt overhead: 3-10μs per interrupt (context switch + cache pollution)
- Cache impact: L2 cache miss rate increases 20-30% during interrupt storms

== DMA and Ring Buffers

*DMA (Direct Memory Access):* NIC transfers packets directly to/from RAM without CPU involvement [Intel 82599 Datasheet §7.1].

*Ring buffer structure:*

```cpp
// Simplified Linux driver (e1000e, igb)
struct rx_ring {
    struct rx_desc* desc;    // Descriptor array
    struct sk_buff** buffers; // Packet buffer pointers
    uint16_t head;           // Next descriptor to process
    uint16_t tail;           // Next descriptor NIC will write
    uint16_t count;          // Ring size (typically 512-4096)
};

struct rx_desc {
    uint64_t buffer_addr;    // Physical address of packet buffer
    uint16_t length;         // Packet length
    uint16_t status;         // DD (Descriptor Done) bit
};
```

*Receive path:*

```
1. Driver allocates ring of descriptors (512-4096 entries)
2. Each descriptor points to a pre-allocated packet buffer (2KB-4KB)
3. Driver writes descriptor physical addresses to NIC registers
4. NIC DMA writes incoming packets to buffers, sets DD bit
5. Interrupt fires (coalesced)
6. Driver processes all packets with DD=1, refills ring
```

*Performance characteristics:*

*Ring size impact:*
- Small ring (128-256): Low memory, risk of drops under burst traffic
- Large ring (2048-4096): High memory (8MB-16MB), better burst absorption
- Typical: 512-1024 for latency, 2048-4096 for throughput

*Cache effects:* Ring descriptor array = sequential access = prefetcher friendly. Packet buffers = random access (poor locality).

*NUMA awareness [Corbet 2010]:*
```cpp
// Allocate ring buffers on same NUMA node as NIC
int node = dev_to_node(&pdev->dev);
desc = kmalloc_node(size, GFP_KERNEL, node);
```
Cross-NUMA access penalty: 1.5-2x latency (~300ns vs ~150ns local)

== NAPI (New API)

*Problem with legacy interrupts:* Interrupt-per-packet doesn't scale [Salim & Olsson 2001, "Improving Linux Networking Performance"].

*NAPI (New API) - polling mode:*

```cpp
// Simplified NAPI structure
struct napi_struct {
    struct list_head poll_list;
    int (*poll)(struct napi_struct*, int budget);
    int weight;  // Max packets per poll (typically 64)
};

// Driver implementation
static int driver_poll(struct napi_struct* napi, int budget) {
    int work_done = 0;

    while (work_done < budget) {
        struct rx_desc* desc = &ring->desc[ring->head];

        if (!(desc->status & RXD_STATUS_DD))
            break;  // No more packets

        struct sk_buff* skb = ring->buffers[ring->head];
        skb_put(skb, desc->length);
        netif_receive_skb(skb);  // Pass to network stack

        refill_rx_desc(ring, ring->head);
        ring->head = (ring->head + 1) % ring->count;
        work_done++;
    }

    if (work_done < budget) {
        napi_complete(napi);
        enable_interrupts(adapter);  // Re-enable interrupts
    }

    return work_done;
}
```

*NAPI operation:*
1. First packet arrives → interrupt fires
2. Driver disables interrupts, schedules NAPI poll
3. Poll processes up to `budget` packets (typically 64) via polling
4. If more packets remain, poll reschedules itself (no interrupt)
5. When ring empty, re-enable interrupts

*Performance:* Under load, NAPI processes millions of packets with ~100-200 interrupts/sec (vs 1M+ interrupts/sec legacy).

*Budget parameter:* Balances latency vs fairness. Too high = one NIC monopolizes CPU. Too low = increased scheduling overhead.

== Hardware Offloads

*TSO (TCP Segmentation Offload):*

```cpp
// Application writes 64KB to TCP socket
write(sockfd, data, 65536);

// Without TSO: kernel generates ~45 1448-byte TCP segments
// With TSO: kernel generates 1 large descriptor, NIC segments

// ethtool -k eth0 | grep tcp-segmentation-offload
tcp-segmentation-offload: on
```

*Benefit:* CPU generates 1/45th of headers, NIC handles segmentation in hardware. Throughput increases 20-40% [Mogul & Ramakrishnan 1997].

*Cost:* Increased latency (microseconds) for segmentation, requires NIC support.

*LRO/GRO (Large Receive Offload / Generic Receive Offload):*

Inverse of TSO. NIC/kernel coalesces multiple incoming packets into single large buffer.

```cpp
// 45 incoming 1448-byte packets → 1 64KB buffer
// Reduces:
// - Per-packet overhead (context switches, cache pollution)
// - TCP ACK processing (1 ACK instead of 45)
```

*Danger:* Breaks forwarding, routing, packet capture. Use GRO (kernel-based, safe) instead of LRO.

*Checksum offload:*

```cpp
// IP/TCP/UDP checksum calculation offloaded to NIC
// CPU cost: ~50-100 cycles per checksum [Intel optimization manual]
// NIC calculates during DMA (zero CPU cost)

// Verification via ethtool
ethtool -k eth0 | grep checksum
rx-checksumming: on
tx-checksumming: on
```

== RSS (Receive Side Scaling)

*Problem:* Single CPU core processes all packets = bottleneck at ~1-2M PPS.

*RSS:* NIC distributes packets across multiple RX queues based on hash [Intel 82599 Datasheet §7.1.2].

```cpp
// RSS hash calculation (Toeplitz hash)
hash = toeplitz_hash(src_ip, dst_ip, src_port, dst_port, secret_key);
queue = hash % num_queues;

// Each queue has:
// - Dedicated ring buffer
// - Dedicated IRQ (CPU affinity)
// - Dedicated NAPI poll

// Configure via ethtool
ethtool -L eth0 combined 8  // 8 RX/TX queues
```

*Scaling:*
- 1 queue: 1-2M PPS (single core bottleneck)
- 4 queues: 4-8M PPS (linear scaling if flows balanced)
- 8+ queues: Diminishing returns (memory bandwidth, cross-core communication)

*Flow affinity:* Same 5-tuple (src IP/port, dst IP/port, protocol) always hashed to same queue = maintains TCP ordering.

*CPU affinity:*
```bash
# Pin IRQ to specific CPU
echo 1 > /proc/irq/<irq_num>/smp_affinity

# Modern approach: irqbalance daemon distributes automatically
```

== XPS (Transmit Packet Steering)

*Problem:* Multiple CPUs transmitting = contention on TX queue lock.

*XPS:* Map each CPU to dedicated TX queue (lock-free) [Corbet 2010].

```cpp
// Kernel selects TX queue based on CPU
int cpu = smp_processor_id();
int queue = map[cpu];

// Configure
echo <cpu_mask> > /sys/class/net/eth0/queues/tx-0/xps_cpus
```

== Latency Analysis

*Packet receive path latency breakdown (modern hardware):*

#table(
  columns: 3,
  align: (left, right, left),
  table.header([Stage], [Latency], [Notes]),
  [Wire propagation], [\~5ns/m], [Speed of light in copper/fiber],
  [NIC PHY processing], [\~200-500ns], [Signal decoding, clock recovery],
  [DMA to RAM], [\~150-300ns], [PCIe transfer, DRAM write],
  [Interrupt delivery], [\~100-200ns], [APIC routing],
  [Context switch], [\~1000-2000ns], [Save/restore registers, TLB flush],
  [Kernel processing], [\~500-1000ns], [IP/TCP validation, routing lookup],
  [Copy to socket buffer], [\~100-500ns], [Depends on packet size],
  [Application wake], [\~1000-2000ns], [Scheduler, context switch],
)

*Total minimum: ~3-7μs for localhost, ~10-30μs for network hop*

*Reference:* Rizzo (2012) "netmap: A Novel Framework for Fast Packet I/O"; Intel (2023) "Ethernet Controller Datasheet".

== Performance Tuning

*NIC ring buffer sizing:*
```bash
# Check current
ethtool -g eth0

# Increase for throughput (reduce drops under burst)
ethtool -G eth0 rx 4096 tx 4096

# Tradeoff: Memory usage vs burst absorption
# 4096 descriptors × 2KB buffer = 8MB per queue
```

*Interrupt coalescing:*
```bash
# Low latency (trading, gaming)
ethtool -C eth0 rx-usecs 5 rx-frames 16

# High throughput (storage, bulk transfer)
ethtool -C eth0 rx-usecs 100 rx-frames 64
```

*CPU isolation:*
```bash
# Isolate CPUs from scheduler (kernel boot param)
isolcpus=2-7 nohz_full=2-7

# Pin network IRQs to isolated CPUs
# Pin application threads to isolated CPUs
# Result: Deterministic latency, no scheduler jitter
```

*NUMA tuning:*
```bash
# Check NIC NUMA node
cat /sys/class/net/eth0/device/numa_node

# Run application on same node
numactl --cpunodebind=0 --membind=0 ./server
```

== Limitations of Kernel Network Stack

*Overhead per packet [Rizzo 2012]:*
- Interrupt/context switch: ~1000ns
- Memory allocation (sk_buff): ~200-500ns
- Protocol processing: ~500-1000ns
- System call overhead: ~100-300ns
- Memory copies: ~5ns/byte

*Total CPU cost: ~2000-3000ns + copy time*

At 10Gbps (1.25GB/s), processing 1M packets/sec:
- CPU time: 2-3 seconds of CPU time per second = unsustainable
- Solution: Batching (NAPI), offload (TSO/LRO), or bypass (DPDK/XDP)

*Next:* See Kernel Bypass section for DPDK, XDP alternatives that eliminate this overhead.

== References

*Primary sources:*

Salim, J.H. & Olsson, R. (2001). "Beyond Softnet." Proceedings of the 5th Annual Linux Showcase & Conference.

Mogul, J.C. & Ramakrishnan, K.K. (1997). "Eliminating Receive Livelock in an Interrupt-driven Kernel." ACM Transactions on Computer Systems 15(3): 217-252.

Corbet, J. (2010). "Network transmit packet steering." LWN.net. https://lwn.net/Articles/412062/

Rizzo, L. (2012). "netmap: A Novel Framework for Fast Packet I/O." USENIX ATC '12.

Intel Corporation (2023). Intel 82599 10 GbE Controller Datasheet. Document 321483-010.

*Textbooks:*

Stevens, W.R., Fenner, B., & Rudoff, A.M. (2003). Unix Network Programming, Volume 1: The Sockets Networking API (3rd ed.). Addison-Wesley.

Marek, C. & Corbet, J. (2005). The Linux Networking Architecture: Design and Implementation of Network Protocols in the Linux Kernel. Prentice Hall.

*RFCs:*

RFC 894: A Standard for the Transmission of IP Datagrams over Ethernet Networks. Hornig, C. (1984).

RFC 1122: Requirements for Internet Hosts – Communication Layers. Braden, R. (1989).

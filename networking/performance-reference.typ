= Performance Reference

Quick reference for network latency, throughput, and CPU costs.

== Latency Numbers (2025)

*CPU & Memory:*
- L1 cache: 1 ns
- L2 cache: 3 ns
- L3 cache: 15 ns
- Main memory (DRAM): 100 ns
- Mutex lock/unlock: 50 ns
- System call (getpid): 100-300 ns
- Context switch: 1-2 μs
- Page fault (memory): 5 μs
- Page fault (SSD): 100 μs

*Network (localhost):*
- Shared memory (SPSC queue): 20-50 ns
- Unix domain socket: 1-5 μs
- TCP localhost (127.0.0.1): 10-30 μs

*Network (datacenter, < 1ms RTT):*
- Same rack: 100-200 μs
- Same datacenter: 200-500 μs
- Cross-datacenter: 1-5 ms

*Network (WAN):*
- Cross-country (US): 50-100 ms
- Intercontinental: 200-300 ms

*Disk:*
- NVMe SSD read: 10-50 μs
- SATA SSD read: 50-150 μs
- HDD read: 5-10 ms
- Network storage (NFS, CIFS): 1-10 ms

== Throughput Limits

*Single TCP connection theoretical max:*
```
Throughput = WindowSize / RTT

Example: 10MB window, 50ms RTT
= 10 × 8 Mbit / 0.05s = 1.6 Gbps
```

*Packet processing rates:*
- Kernel (single core): 1-3M PPS
- XDP (single core): 24M PPS
- DPDK (single core): 40-60M PPS
- Theoretical max (10G, 64B packets): 14.88M PPS

*Bandwidth vs packet size (10 Gbps link):*
- 64-byte packets: 14.88M PPS
- 128-byte packets: 8.13M PPS
- 512-byte packets: 2.35M PPS
- 1518-byte packets: 812K PPS

== CPU Costs

*Per-packet overhead (modern CPU, 3GHz):*
- Interrupt: 1000 ns (~3000 cycles)
- System call (send/recv): 500 ns (~1500 cycles)
- TCP/IP processing: 500-1000 ns (~1500-3000 cycles)
- Memory copy (1KB): 200 ns (~600 cycles, 5ns/byte)
- Checksum (1KB, hardware): 0 ns (NIC offload)
- Checksum (1KB, software): 50 ns (~150 cycles)
- Crypto (AES-128, 1KB, AES-NI): 100 ns (~300 cycles)

*Context switch breakdown:*
- Direct cost (save/restore registers): 200-400 ns
- Indirect cost (TLB flush, cache pollution): 1-3 μs
- Total: 1-4 μs depending on workload

== Memory Bandwidth

*DDR4-2400:*
- Theoretical: 19.2 GB/s per channel
- Practical: 15-17 GB/s (80-90% efficiency)
- Dual-channel: 30-34 GB/s

*Impact on networking:*
- 10 Gbps = 1.25 GB/s
- With 2 copies (kernel ↔ user): 2.5 GB/s (8% of bandwidth)
- With 4 copies (socket buffer + app buffer): 5 GB/s (15% of bandwidth)

== TCP Window Scaling

*Required window sizes for full utilization:*

#table(
  columns: 3,
  align: (left, right, right),
  table.header([Bandwidth], [RTT], [Window Size]),
  [1 Gbps], [10ms], [1.25 MB],
  [1 Gbps], [100ms], [12.5 MB],
  [10 Gbps], [10ms], [12.5 MB],
  [10 Gbps], [100ms], [125 MB],
  [100 Gbps], [10ms], [125 MB],
)

*Formula:* BDP (Bandwidth-Delay Product) = Bandwidth × RTT

== Interrupt Coalescing Impact

#table(
  columns: 4,
  align: (left, right, right, right),
  table.header([Coalesce Time], [Latency], [PPS], [CPU % (10G)]),
  [0 (no coalesce)], [2-3 μs], [14.88M], [40-50%],
  [10 μs], [10-15 μs], [100K], [5-10%],
  [50 μs], [50-70 μs], [20K], [1-3%],
  [100 μs], [100-150 μs], [10K], [\<1%],
)

*Tradeoff:* Low latency requires high CPU. Tune based on workload.

== Queue Depths

*Typical ring buffer sizes:*
- NIC RX ring: 512-4096 descriptors
- NIC TX ring: 512-4096 descriptors
- Socket receive buffer: 128KB-4MB
- Socket send buffer: 16KB-4MB
- TCP window: 64KB-1GB (with scaling)

*Rule of thumb:* Ring size = bandwidth × latency / packet size

Example: 10Gbps, 100μs burst, 1518B packets
= 10 Gbps × 100 μs / (1518 × 8 bits) = 823 packets

== References

Dean, J. & Barroso, L.A. (2013). "The Tail at Scale." Communications of the ACM 56(2): 74-80.

Gregg, B. (2013). Systems Performance: Enterprise and the Cloud. Prentice Hall.

Intel Corporation (2023). Intel 64 and IA-32 Architectures Optimization Reference Manual.

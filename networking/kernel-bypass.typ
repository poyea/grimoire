= Kernel Bypass

Kernel networking stack adds 2-5μs latency per packet. Kernel bypass eliminates this overhead by giving applications direct access to NIC hardware.

*See also:* Link Layer (for NIC ring buffers), Zero-Copy (for optimizing kernel path), Lock-Free (for userspace queues)

== Why Bypass the Kernel?

*Kernel overhead per packet:*
- Interrupt handling: ~1μs
- Context switch: ~1μs
- Protocol stack (TCP/IP): ~500ns-1μs
- Memory copies: ~500ns
- *Total: 2-5μs minimum*

*At 10Gbps (1.25GB/s, 64-byte packets = 14.8M PPS):*
- Kernel processing: 14.8M × 3μs = 44 seconds of CPU per second (impossible!)
- Solution: Batch processing (NAPI) or bypass kernel entirely

*Bypass benefits:*
1. Latency: 200-500ns vs 2-5μs (5-10x improvement)
2. Throughput: 40-200M PPS vs 1-3M PPS (kernel)
3. Determinism: No scheduler jitter, interrupts
4. CPU efficiency: Poll mode uses 100% CPU but processes 10x packets

== DPDK (Data Plane Development Kit)

*Architecture:* Userspace PMD (Poll Mode Driver) + huge pages + CPU isolation.

```cpp
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>

int main(int argc, char** argv) {
    // 1. Initialize EAL (Environment Abstraction Layer)
    rte_eal_init(argc, argv);

    // 2. Create mempool for packet buffers (huge pages)
    struct rte_mempool* mbuf_pool = rte_pktmbuf_pool_create(
        "MBUF_POOL", 8192,  // 8K buffers
        256,                 // Cache size
        0,                   // Private data size
        RTE_MBUF_DEFAULT_BUF_SIZE,
        rte_socket_id()
    );

    // 3. Configure NIC port
    uint16_t port_id = 0;
    struct rte_eth_conf port_conf = {};
    port_conf.rxmode.max_rx_pkt_len = RTE_ETHER_MAX_LEN;

    rte_eth_dev_configure(port_id, 1, 1, &port_conf);  // 1 RX queue, 1 TX queue

    // 4. Setup RX queue
    rte_eth_rx_queue_setup(port_id, 0, 1024, rte_socket_id(), NULL, mbuf_pool);

    // 5. Setup TX queue
    rte_eth_tx_queue_setup(port_id, 0, 1024, rte_socket_id(), NULL);

    // 6. Start device
    rte_eth_dev_start(port_id);

    // 7. Main loop - poll for packets
    struct rte_mbuf* pkts[32];
    while (1) {
        uint16_t nb_rx = rte_eth_rx_burst(port_id, 0, pkts, 32);

        for (uint16_t i = 0; i < nb_rx; i++) {
            // Process packet (parse headers, forward, etc.)
            process_packet(pkts[i]);
        }

        // Send packets
        rte_eth_tx_burst(port_id, 0, pkts, nb_rx);
    }
}
```

*Key components:*

*1. Poll Mode Driver (PMD):*
- No interrupts - CPU continuously polls NIC ring buffer
- Tight loop checks descriptor "done" bit: ~50-100 cycles per check
- 100% CPU usage, but 10-100x throughput vs interrupts

*2. Huge pages (2MB/1GB):*
```bash
echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
```
- Reduces TLB misses: 512x fewer TLB entries (2MB vs 4KB pages)
- DMA requires physically contiguous memory - huge pages simplify allocation

*3. CPU isolation:*
```bash
isolcpus=4-7 nohz_full=4-7  # Kernel boot parameters
```
- Reserve CPUs for DPDK (no scheduler, no interrupts)
- Pin DPDK threads to isolated CPUs
- Result: Deterministic latency (\<1μs jitter)

*Performance [Intel DPDK benchmarks]:*
- Single core: 40-60M PPS (14.88M theoretical max for 64B packets at 10G)
- Multi-core (RSS): Scales linearly to 200M+ PPS (with multiple NICs)

== XDP (eXpress Data Path)

*In-kernel fast path:* BPF programs execute in NIC driver, before skb allocation [Høiland-Jørgensen et al. 2018].

```c
// XDP program (compiled to BPF bytecode)
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>

SEC("xdp")
int xdp_filter(struct xdp_md* ctx) {
    void* data_end = (void*)(long)ctx->data_end;
    void* data = (void*)(long)ctx->data;

    // Parse Ethernet header
    struct ethhdr* eth = data;
    if ((void*)(eth + 1) > data_end)
        return XDP_DROP;

    // Parse IP header
    if (eth->h_proto != htons(ETH_P_IP))
        return XDP_PASS;  // Not IPv4, pass to kernel

    struct iphdr* ip = (void*)(eth + 1);
    if ((void*)(ip + 1) > data_end)
        return XDP_DROP;

    // Drop packets from specific IP
    if (ip->saddr == htonl(0xC0A80001))  // 192.168.0.1
        return XDP_DROP;

    // Forward to userspace via AF_XDP socket
    return XDP_REDIRECT;
}

char _license[] SEC("license") = "GPL";
```

*XDP verdicts:*
- `XDP_DROP`: Discard packet (DDoS mitigation)
- `XDP_PASS`: Send to kernel stack
- `XDP_TX`: Bounce back to same NIC (load balancer)
- `XDP_REDIRECT`: Send to different NIC or AF_XDP socket

*Performance:* 24M PPS single core (close to DPDK, but kernel-integrated).

*Advantages over DPDK:*
- Integrated with kernel (can use iptables, routing table)
- Can fall back to kernel stack for complex packets
- No huge page/CPU isolation requirements

*Disadvantages:*
- BPF program size limited (4096 instructions)
- No floating point, limited loops
- Requires Linux 4.18+ and compatible NIC driver

== AF_XDP (XDP Sockets)

*Zero-copy userspace access:* XDP redirects packets to userspace via shared ring buffers.

```cpp
#include <linux/if_xdp.h>
#include <bpf/xsk.h>

// Create AF_XDP socket
struct xsk_socket_info {
    struct xsk_ring_cons rx;
    struct xsk_ring_prod tx;
    struct xsk_umem_info* umem;
    struct xsk_socket* xsk;
};

// 1. Allocate UMEM (packet buffers)
void* buffer = mmap(NULL, NUM_FRAMES * FRAME_SIZE,
                    PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

struct xsk_umem* umem;
xsk_umem__create(&umem, buffer, NUM_FRAMES * FRAME_SIZE, &rx_ring, &tx_ring, NULL);

// 2. Create XDP socket
struct xsk_socket_config cfg = {};
cfg.rx_size = XSK_RING_CONS__DEFAULT_NUM_DESCS;
cfg.tx_size = XSK_RING_PROD__DEFAULT_NUM_DESCS;

xsk_socket__create(&xsk, ifname, queue_id, umem, &rx_ring, &tx_ring, &cfg);

// 3. Receive packets
while (1) {
    uint32_t idx_rx = 0;
    unsigned int rcvd = xsk_ring_cons__peek(&rx_ring, BATCH_SIZE, &idx_rx);

    for (unsigned int i = 0; i < rcvd; i++) {
        const struct xdp_desc* desc = xsk_ring_cons__rx_desc(&rx_ring, idx_rx++);
        void* pkt = xsk_umem__get_data(buffer, desc->addr);
        // Process packet at pkt
    }

    xsk_ring_cons__release(&rx_ring, rcvd);
}
```

*Performance:* 10-20M PPS (between kernel and DPDK), with kernel integration benefits.

== RDMA (Remote Direct Memory Access)

*Zero-copy network:* NIC directly reads/writes remote memory, bypassing both CPUs [Infiniband Trade Association].

```cpp
#include <infiniband/verbs.h>

// RDMA Write: Copy local memory to remote memory (no remote CPU involvement!)
struct ibv_send_wr wr = {};
struct ibv_sge sge = {};

sge.addr = (uintptr_t)local_buf;
sge.length = size;
sge.lkey = mr->lkey;

wr.wr_id = 0;
wr.opcode = IBV_WR_RDMA_WRITE;
wr.send_flags = IBV_SEND_SIGNALED;
wr.sg_list = &sge;
wr.num_sge = 1;
wr.wr.rdma.remote_addr = remote_addr;
wr.wr.rdma.rkey = remote_key;

ibv_post_send(qp, &wr, &bad_wr);  // Async, returns immediately
```

*Latency:* 1-2μs (vs 10-30μs TCP), measured RTT within datacenter [Kalia et al. 2016].

*Use cases:* Distributed storage (Ceph), HPC, high-frequency trading.

*Limitations:* Requires special NICs (ConnectX, Chelsio), complex setup.

== Comparison

| Technique | Latency | Throughput | CPU | Kernel Integration | Complexity |
|:----------|--------:|-----------:|----:|:-------------------|:-----------|
| Kernel (epoll) | 10-30μs | 1-3M PPS | 20% | Full | Low |
| XDP | 1-3μs | 24M PPS | 100% | Partial | Medium |
| AF_XDP | 2-5μs | 10-20M PPS | 100% | Partial | Medium |
| DPDK | 0.5-1μs | 40-200M PPS | 100% | None | High |
| RDMA | 1-2μs | 100M msgs/s | 5% | None | Very High |

*Decision tree:*
1. Need kernel stack (iptables, routing)? → XDP
2. Need maximum throughput (packet forwarding, DPI)? → DPDK
3. Need lowest latency (trading, telemetry)? → RDMA
4. Standard application (web, database)? → Kernel (io_uring)

== References

Intel Corporation (2023). DPDK Programmer's Guide. https://doc.dpdk.org/

Høiland-Jørgensen, T., et al. (2018). "The eXpress Data Path: Fast Programmable Packet Processing in the Operating System Kernel." CoNEXT '18.

Kalia, A., Kaminsky, M., & Andersen, D.G. (2016). "Design Guidelines for High Performance RDMA Systems." USENIX ATC '16.

Infiniband Trade Association. InfiniBand Architecture Specification, Volume 1, Release 1.4.

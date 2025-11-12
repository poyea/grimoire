= Zero-Copy Networking

Traditional network I/O involves multiple data copies between kernel and userspace. Zero-copy techniques eliminate unnecessary copying to improve throughput and reduce CPU usage.

*See also:* Link Layer (for DMA mechanics), Kernel Bypass (for eliminating kernel entirely), Sockets API (for traditional copy-based I/O)

== Traditional Copy Path

*Standard `send()` operation:*

```
User buffer               Kernel buffer           NIC buffer
   (app)                     (socket)             (DMA ring)
    │                           │                      │
    │  1. write(sock, buf, n)  │                      │
    ├──────── copy ───────────▶ │                      │
    │       (~5ns/byte)         │                      │
    │                           │  2. TCP/IP stack     │
    │                           │     (headers)        │
    │                           ├──────  copy  ───────▶│
    │                           │     (~5ns/byte)      │
    │                           │                      │ 3. DMA to NIC
    │                           │                      ├─────▶ Wire
```

*Cost breakdown (1KB packet):*
- User → kernel copy: 1024 × 5ns = 5μs
- Kernel → NIC copy: 1024 × 5ns = 5μs
- TCP/IP processing: ~1-2μs
- *Total: ~11-12μs per packet*

*Memory bandwidth:* Copying 1GB = 2× memory bandwidth (read + write) = 10GB/s on DDR4-2400.

At 10Gbps line rate (1.25GB/s), copying alone consumes 25-30% of memory bandwidth.

== sendfile() - Kernel-to-Kernel Copy

*Use case:* Serve static files (web server, file transfer).

```cpp
#include <sys/sendfile.h>

// Traditional (2 copies):
char buf[8192];
while ((n = read(file_fd, buf, sizeof(buf))) > 0) {
    send(socket_fd, buf, n, 0);
}
// read: disk → kernel → user (1 copy)
// send: user → kernel → NIC (2 copies)
// Total: 3 copies

// Zero-copy sendfile (1 copy):
off_t offset = 0;
sendfile(socket_fd, file_fd, &offset, file_size);
// disk → kernel → NIC (1 copy, kernel-only)
// Eliminates user → kernel → user round-trip
```

*Performance [Pai et al. 2000]:*
- Traditional: 450 MB/s, 60% CPU
- sendfile(): 900 MB/s, 30% CPU
- *2x throughput, 50% CPU reduction*

*Limitation:* Only works for file → socket. Cannot modify data in transit (no encryption, compression).

== splice() - Zero-Copy Pipe

*Linux-specific [Corbet 2006]:* Move data between file descriptors via kernel pipe buffers.

```cpp
#include <fcntl.h>

// Create pipe
int pipefd[2];
pipe(pipefd);

// Splice file → pipe → socket (zero copies to userspace)
while (remaining > 0) {
    // File → pipe (zero-copy if filesystem supports)
    ssize_t n = splice(file_fd, NULL, pipefd[1], NULL, SPLICE_SIZE, SPLICE_F_MOVE);

    // Pipe → socket (zero-copy)
    splice(pipefd[0], NULL, socket_fd, NULL, n, SPLICE_F_MOVE);

    remaining -= n;
}
```

*SPLICE_F_MOVE flag:* Move pages instead of copying (if possible).

*Performance:* Similar to `sendfile()`, but more flexible (works with pipes, supports chaining).

== mmap() - Shared Memory Mapping

*Map file directly into address space:*

```cpp
#include <sys/mman.h>

// Map file into memory
void* addr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, file_fd, 0);

// Send mapped memory (kernel copies, but no user copy)
send(socket_fd, addr, file_size, 0);

munmap(addr, file_size);
```

*Tradeoff:*
- *Advantage:* No explicit user copy, lazy loading (page faults on access)
- *Disadvantage:* Page faults = TLB misses = 100-200 cycles each, potential page-in from disk

*When mmap wins:* Large files accessed partially/randomly (database, memory-mapped I/O).

*When mmap loses:* Small files, sequential access (overhead > benefit).

== MSG_ZEROCOPY (Linux 4.14+)

*Zero-copy `send()`:* Asynchronous transmission without copying.

```cpp
#include <linux/errqueue.h>

// Enable zero-copy on socket
int optval = 1;
setsockopt(sock, SOL_SOCKET, SO_ZEROCOPY, &optval, sizeof(optval));

// Send with MSG_ZEROCOPY flag
char buf[1024*1024];  // Must remain valid until transmission completes!
ssize_t n = send(sock, buf, sizeof(buf), MSG_ZEROCOPY);

// Kernel notifies completion via error queue
struct sock_extended_err* serr;
struct msghdr msg = {};
struct cmsghdr* cm;
char control[128];

msg.msg_control = control;
msg.msg_controllen = sizeof(control);

// Wait for completion notification
recvmsg(sock, &msg, MSG_ERRQUEUE);

for (cm = CMSG_FIRSTHDR(&msg); cm; cm = CMSG_NXTHDR(&msg, cm)) {
    if (cm->cmsg_level == SOL_IP && cm->cmsg_type == IP_RECVERR) {
        serr = (struct sock_extended_err*)CMSG_DATA(cm);
        if (serr->ee_errno == 0 && serr->ee_origin == SO_EE_ORIGIN_ZEROCOPY) {
            // Transmission complete, safe to reuse buf
        }
    }
}
```

*Constraints:*
1. Minimum size: 10KB (smaller packets not worth overhead)
2. Buffer must remain unchanged until notification
3. Async notification = complex programming model

*Performance [Netdev 0x12, 2018]:*
- Small packets (< 10KB): Slower than copy (notification overhead)
- Large packets (> 100KB): 10-30% CPU reduction
- Bulk transfer (1MB+): 40-60% CPU reduction

*Use case:* High-throughput applications (video streaming, backup) with large messages.

== recv() Zero-Copy Challenges

*Problem:* Receiving requires knowing buffer size in advance. Kernel must copy to provide data to userspace.

*Partial solution - MSG_TRUNC:*
```cpp
// Peek message size without copying
ssize_t size = recv(sock, NULL, 0, MSG_PEEK | MSG_TRUNC);

// Allocate exact size
char* buf = malloc(size);
recv(sock, buf, size, 0);  // Copy (unavoidable with sockets API)
```

*True zero-copy recv:* Requires kernel bypass (see DPDK, XDP) or specialized APIs (io_uring with registered buffers).

== io_uring Zero-Copy (Linux 5.1+)

*Modern asynchronous I/O:* Shared ring buffers between kernel and userspace [Axboe 2019].

```cpp
#include <liburing.h>

struct io_uring ring;
io_uring_queue_init(256, &ring, 0);

// Submit send operation (zero-copy if large enough)
struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
io_uring_prep_send(sqe, sockfd, buf, len, MSG_ZEROCOPY);
io_uring_submit(&ring);

// Wait for completion
struct io_uring_cqe* cqe;
io_uring_wait_cqe(&ring, &cqe);
// Check cqe->res for result
io_uring_cqe_seen(&ring, cqe);
```

*Advantages:*
- Batched syscalls: Submit N operations with 1 syscall
- Zero-copy for qualifying operations
- Unified API for network, disk, timers

*Performance:* 2-3x faster than epoll for high-throughput workloads [Axboe 2020].

== Hardware Support - DMA

*Direct Memory Access:* NIC transfers data directly to/from RAM, bypassing CPU.

```cpp
// Kernel allocates DMA-capable memory (physically contiguous)
void* dma_buf = dma_alloc_coherent(dev, size, &dma_handle, GFP_KERNEL);

// Program NIC descriptor with physical address
rx_desc->buffer_addr = dma_handle;

// NIC writes directly to dma_buf via DMA (CPU-free transfer)
```

*DMA vs CPU copy:*
- DMA: 0% CPU, PCIe bandwidth-limited (~8GB/s for PCIe 3.0 x8)
- CPU copy: 20-30% CPU for 10Gbps, DRAM bandwidth-limited

*Scatter-Gather DMA:* Transfer from multiple discontiguous buffers in single operation.

```cpp
// Descriptor chain for fragmented packet
struct sg_desc {
    uint64_t addr;
    uint32_t len;
} descs[N];

// NIC reads descs and DMAs each fragment
// Eliminates kernel reassembly
```

== Zero-Copy Limitations

*1. Small packets:* Setup overhead > copy cost.
- Threshold: ~4-10KB depending on hardware

*2. Data modification:* Cannot encrypt/compress if passing pointers.
- Workaround: Offload to NIC (TLS offload, IPsec)

*3. Buffer ownership:* Async zero-copy = complex lifetime management.
- Solution: Reference counting, completion notifications

*4. Memory alignment:* DMA requires page-aligned buffers.
- Unaligned data must be copied to aligned buffer

== Practical Recommendations

| Scenario | Technique | Rationale |
|:---------|:----------|:----------|
| Static file serving | sendfile() | Simple, kernel-optimized |
| Large bulk transfer | MSG_ZEROCOPY | CPU savings > complexity cost |
| High-throughput server | io_uring | Batching + zero-copy |
| Low-latency trading | Kernel bypass (DPDK) | Eliminate kernel entirely |
| Standard CRUD app | Traditional send/recv | Simplicity > micro-optimization |

== References

Pai, V.S., Druschel, P., & Zwaenepoel, W. (2000). "IO-Lite: A Unified I/O Buffering and Caching System." ACM Transactions on Computer Systems 18(1): 37-66.

Corbet, J. (2006). "Splice and Tee." LWN.net. https://lwn.net/Articles/178199/

Axboe, J. (2019). "Efficient IO with io_uring." Linux Plumbers Conference.

Netdev 0x12 (2018). "MSG_ZEROCOPY." Linux Kernel Networking Developers Conference.

Axboe, J. (2020). "What's new with io_uring." Kernel Recipes.

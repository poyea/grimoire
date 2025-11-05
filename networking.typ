#set document(title: "Networking", author: "John Law")
#set page(
  paper: "us-letter",
  margin: (x: 1cm, y: 1cm),
  header: [
    #smallcaps[_Networking Notes by #link("https://github.com/poyea")[\@poyea]_]
    #h(0.5fr)
    #emph(text[#datetime.today().display()])
    #h(0.5fr)
    #emph(link("https://github.com/poyea/grimoire")[poyea/grimoire::networking])
  ],
  footer: context [
    #align(right)[#counter(page).display("1")]
  ]
)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")

#align(center)[
  #block(
    fill: luma(245),
    inset: 10pt,
    width: 80%,
    text(size: 9pt)[
      #align(center)[
        #text(size: 24pt, weight: "bold")[Networking]
      ]
    ]
  )
]

#outline(
  title: "Table of Contents",
  depth: 2,
)

#emph[Enjoy.]

#pagebreak()

= Introduction

This reference covers computer networking from the physical layer through application protocols, with focus on:

- *Performance*: Latency, throughput, CPU costs at each layer
- *Implementation*: POSIX APIs, zero-copy techniques, kernel bypass
- *Concurrency*: Thread models, event-driven I/O, lock-free data structures
- *Modern protocols*: HTTP/2, HTTP/3, QUIC, ZeroMQ patterns
- *Academic rigor*: Referenced with RFCs, research papers, textbooks

*Intended audience:* Systems programmers, network engineers, performance engineers building high-throughput low-latency systems.

*Notation:*
- Latency numbers are typical for modern hardware (2020+)
- Code examples use C/C++ and Linux unless specified
- Performance measurements assume x86-64 architecture

== Network Stack Overview

The OSI 7-layer model is pedagogical; practical implementations use TCP/IP 4-layer model:

```
┌─────────────────────────────────────┐
│  Application Layer                   │  HTTP, DNS, SSH, ZMQ
├─────────────────────────────────────┤
│  Transport Layer                     │  TCP, UDP, QUIC
├─────────────────────────────────────┤
│  Internet Layer                      │  IP (IPv4/IPv6), ICMP
├─────────────────────────────────────┤
│  Link Layer                          │  Ethernet, WiFi, ARP
└─────────────────────────────────────┘
```

*Data flow:* Application → Socket → TCP/UDP → IP → Link → Physical wire → ... → Physical → Link → IP → TCP/UDP → Socket → Application

*Key insight:* Each layer adds overhead. High-performance systems minimize copying and context switches between layers.

#pagebreak()

#include "networking/link-layer.typ"
#pagebreak()

#include "networking/internet-layer.typ"
#pagebreak()

#include "networking/transport-layer.typ"
#pagebreak()

#include "networking/sockets-api.typ"
#pagebreak()

#include "networking/io-multiplexing.typ"
#pagebreak()

#include "networking/application-protocols.typ"
#pagebreak()

#include "networking/zero-copy.typ"
#pagebreak()

#include "networking/kernel-bypass.typ"
#pagebreak()

#include "networking/concurrency-models.typ"
#pagebreak()

#include "networking/lock-free.typ"
#pagebreak()

#include "networking/message-queues.typ"
#pagebreak()

#include "networking/performance-reference.typ"
#pagebreak()

= Conclusion

Modern networking requires understanding multiple layers:

*Hardware level:* Interrupt coalescing, DMA, ring buffers determine baseline performance. Kernel bypass (DPDK, XDP) eliminates syscall overhead for specialized applications.

*Protocol level:* TCP provides reliability at cost of latency. UDP sacrifices guarantees for speed. QUIC modernizes transport layer with 0-RTT and built-in encryption.

*Application level:* Event-driven I/O (epoll) scales to 100K+ connections. Lock-free queues enable multi-core parallelism without mutex contention.

*Message queues:* ZeroMQ abstracts socket complexity, provides patterns (pub-sub, req-rep, push-pull) with minimal overhead.

*Key tradeoffs:*
- Latency vs throughput: batching improves throughput but increases latency
- Reliability vs speed: TCP vs UDP
- Flexibility vs performance: kernel networking vs bypass
- Simplicity vs scalability: blocking I/O vs event-driven

*Performance hierarchy (ascending latency):*
1. Shared memory (lock-free SPSC): 20-50ns
2. Unix domain sockets: 1-5μs
3. TCP localhost (127.0.0.1): 10-30μs
4. TCP same datacenter (< 1ms RTT): 100μs - 1ms
5. TCP cross-country (50-100ms RTT): 50-100ms
6. TCP intercontinental (200-300ms RTT): 200-300ms

*When to use what:*
- *Blocking sockets:* Simple clients, low concurrency (< 100 connections)
- *epoll:* High concurrency servers (10K+ connections)
- *io_uring:* Maximum performance, kernel 5.1+ (replacing epoll)
- *DPDK/XDP:* Packet processing, ultra-low latency trading, DDoS mitigation
- *RDMA:* HPC, distributed databases, high-throughput low-latency
- *ZeroMQ:* Distributed systems, microservices, avoiding raw socket complexity

Further reading: Stevens et al. (2003) "Unix Network Programming"; Kerrisk (2010) "The Linux Programming Interface"; Marek & Corbet (2005) "The Linux Networking Architecture".

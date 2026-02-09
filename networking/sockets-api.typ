= POSIX Sockets API

The Berkeley Sockets API provides the standard interface for network programming across Unix systems [POSIX.1-2017].

*See also:* I/O Multiplexing (for handling multiple connections), Transport Layer (for protocol details), Concurrency Models (for server architectures)

== Socket System Calls

*Core API:*

```cpp
// Create socket
int socket(int domain, int type, int protocol);
// domain: AF_INET (IPv4), AF_INET6 (IPv6), AF_UNIX (local)
// type: SOCK_STREAM (TCP), SOCK_DGRAM (UDP)
// Returns: file descriptor

// Bind to address
int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);

// Listen for connections (TCP only)
int listen(int sockfd, int backlog);
// backlog: queue size for pending connections

// Accept connection (TCP server)
int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
// Blocks until client connects, returns new socket for communication

// Connect to server (TCP client)
int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen);

// Data transfer
ssize_t send(int sockfd, const void *buf, size_t len, int flags);
ssize_t recv(int sockfd, void *buf, size_t len, int flags);
ssize_t sendto(int sockfd, const void *buf, size_t len, int flags,
               const struct sockaddr *dest_addr, socklen_t addrlen);  // UDP
ssize_t recvfrom(int sockfd, void *buf, size_t len, int flags,
                 struct sockaddr *src_addr, socklen_t *addrlen);  // UDP

// Close socket
int close(int sockfd);
```

== TCP Server Example

```cpp
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string.h>

int create_tcp_server(uint16_t port) {
    // 1. Create socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return -1;

    // 2. Allow port reuse (avoid "Address already in use")
    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // 3. Bind to port
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;  // Listen on all interfaces
    addr.sin_port = htons(port);        // Convert to network byte order

    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(sock);
        return -1;
    }

    // 4. Listen (backlog = 128)
    if (listen(sock, 128) < 0) {
        close(sock);
        return -1;
    }

    return sock;
}

void handle_client(int client_sock) {
    char buffer[4096];
    ssize_t n = recv(client_sock, buffer, sizeof(buffer), 0);
    if (n > 0) {
        send(client_sock, "HTTP/1.1 200 OK\r\n\r\nHello\n", 26, 0);
    }
    close(client_sock);
}

int main() {
    int server = create_tcp_server(8080);

    while (1) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);

        int client = accept(server, (struct sockaddr*)&client_addr, &addr_len);
        if (client < 0) continue;

        handle_client(client);  // Blocking - only 1 client at a time!
    }
}
```

*Performance:* Blocking model serves 1 request at a time. ~50-100 req/sec max (limited by RTT).

== Non-Blocking I/O

*Problem:* `accept()`, `recv()`, `send()` block → CPU idle → poor throughput.

*Solution:* Non-blocking sockets return immediately with EAGAIN/EWOULDBLOCK.

```cpp
#include <fcntl.h>

// Make socket non-blocking
int flags = fcntl(sockfd, F_GETFL, 0);
fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);

// Non-blocking accept
int client = accept(server, NULL, NULL);
if (client < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
        // No connection available, try later
    } else {
        // Actual error
    }
}

// Non-blocking recv
ssize_t n = recv(sockfd, buf, len, 0);
if (n < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
        // No data available, try later
    }
} else if (n == 0) {
    // Connection closed
}
```

*Busy-wait problem:* Polling in tight loop wastes CPU. *Solution:* I/O multiplexing (next section).

== Socket Options

*Common options:*

```cpp
// 1. SO_REUSEADDR: Allow binding to port in TIME_WAIT state
int opt = 1;
setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

// 2. SO_KEEPALIVE: Detect dead connections
setsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, &opt, sizeof(opt));

// 3. SO_RCVBUF / SO_SNDBUF: Buffer sizes
int bufsize = 1024 * 1024;  // 1MB
setsockopt(sock, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));

// 4. TCP_NODELAY: Disable Nagle's algorithm (low latency)
setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

// 5. SO_LINGER: Control close() behavior
struct linger ling = {1, 0};  // Enable, 0 sec timeout = RST on close
setsockopt(sock, SOL_SOCKET, SO_LINGER, &ling, sizeof(ling));
```

== Socket States and TIME_WAIT

*TCP state machine [RFC 793]:*

```
CLOSED → SYN_SENT → ESTABLISHED → FIN_WAIT_1 → FIN_WAIT_2 → TIME_WAIT → CLOSED
```

*TIME_WAIT problem:*
- After close(), socket remains in TIME_WAIT for 2×MSL (typically 60-120s)
- Purpose: Ensure delayed packets don't corrupt new connection on same port
- *Symptom:* "Address already in use" when restarting server quickly

*Solutions:*
1. `SO_REUSEADDR`: Allow binding to TIME_WAIT port (standard practice)
2. `SO_REUSEPORT`: Multiple sockets bind to same port (load balancing)
3. Reduce `net.ipv4.tcp_fin_timeout` (dangerous, breaks RFC)

```bash
# Check TIME_WAIT sockets
netstat -an | grep TIME_WAIT | wc -l

# If thousands of TIME_WAIT: possible issue (e.g., client not reusing connections)
```

== Latency of Socket Operations

*System call overhead [Linux 5.x, x86-64]:*

#table(
  columns: 3,
  align: (left, right, left),
  table.header([Operation], [Latency], [Notes]),
  [socket()], [~1-2μs], [Allocate fd, socket structure],
  [bind()], [~500ns], [Update kernel tables],
  [connect() (localhost)], [~30-50μs], [Full 3-way handshake],
  [accept() (ready)], [~2-4μs], [Pop from accept queue],
  [accept() (blocked)], [~10-20μs], [Wake from sleep + context switch],
  [send() (buffered)], [~500ns-2μs], [Copy to kernel buffer],
  [recv() (ready)], [~500ns-2μs], [Copy from kernel buffer],
  [close()], [~1-3μs], [Update state, free resources],
)

*Key insight:* System call overhead ~= L3 cache miss. Amortize by batching I/O.

== Zero-Length Recv Optimization

*Problem:* `recv()` always allocates kernel buffer, copies data (even if 0 bytes available).

*Optimization:* Use `MSG_TRUNC` flag to peek size without copying:

```cpp
// Check bytes available without copying
ssize_t n = recv(sock, NULL, 0, MSG_TRUNC | MSG_PEEK);
if (n > 0) {
    char* buf = allocate_exact_size(n);
    recv(sock, buf, n, 0);  // Exact size, one syscall
}
```

*Benefit:* Avoid over-allocation, reduce memory waste.

#pagebreak()

== I/O Multiplexing with epoll

*Problem:* One-thread-per-client does not scale beyond ~1K connections (stack memory + context switch overhead).

*Solution:* `epoll` provides $O(1)$ event notification for ready file descriptors. Handles 10K+ concurrent connections on a single thread (C10K problem).

*API overview:*

```cpp
int epoll_create1(int flags);           // Create epoll instance
int epoll_ctl(int epfd, int op,         // Add/modify/remove watched fds
              int fd, struct epoll_event *event);
int epoll_wait(int epfd,                // Wait for events
               struct epoll_event *events,
               int maxevents, int timeout);
```

*epoll-based TCP server:*

```cpp
#include <sys/epoll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

constexpr int MAX_EVENTS = 1024;

void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

int main() {
    int server = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(8080);
    bind(server, (struct sockaddr*)&addr, sizeof(addr));
    listen(server, 128);
    set_nonblocking(server);

    // 1. Create epoll instance
    int epfd = epoll_create1(0);

    // 2. Register server socket for read events
    struct epoll_event ev = {};
    ev.events = EPOLLIN;
    ev.data.fd = server;
    epoll_ctl(epfd, EPOLL_CTL_ADD, server, &ev);

    struct epoll_event events[MAX_EVENTS];
    char buf[4096];

    // 3. Event loop
    while (true) {
        int n_ready = epoll_wait(epfd, events, MAX_EVENTS, -1);

        for (int i = 0; i < n_ready; ++i) {
            if (events[i].data.fd == server) {
                // Accept all pending connections
                while (true) {
                    int client = accept(server, nullptr, nullptr);
                    if (client < 0) break;  // EAGAIN = no more
                    set_nonblocking(client);
                    ev.events = EPOLLIN | EPOLLET;  // Edge-triggered
                    ev.data.fd = client;
                    epoll_ctl(epfd, EPOLL_CTL_ADD, client, &ev);
                }
            } else {
                // Handle client data
                int fd = events[i].data.fd;
                ssize_t n = recv(fd, buf, sizeof(buf), 0);
                if (n <= 0) {
                    close(fd);  // Removes from epoll automatically
                } else {
                    send(fd, buf, n, 0);  // Echo back
                }
            }
        }
    }

    close(epfd);
    close(server);
}
```

*Performance comparison:*

#table(
  columns: 3,
  align: (left, right, left),
  table.header([Model], [10K Connections], [Notes]),
  [Thread-per-client], [~80 MB RAM], [8 KB stack each + context switches],
  [`select()`], [$O(n)$ per call], [Scans entire fd set; limit 1024 fds],
  [`poll()`], [$O(n)$ per call], [No fd limit but still linear scan],
  [`epoll`], [$O(1)$ per event], [Kernel maintains ready list; edge-triggered mode],
)

*Edge-triggered vs level-triggered:*
- *Level-triggered (default):* `epoll_wait` returns fd whenever data is available. Simpler but more syscalls.
- *Edge-triggered (`EPOLLET`):* Returns fd only on state change. Must drain all data per event (`recv` until `EAGAIN`). Fewer syscalls, higher throughput.

== UDP Server Example

UDP sockets require no `listen()` or `accept()`. Each `recvfrom()` returns the sender's address for reply via `sendto()`. No connection state is maintained.

```cpp
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>

int main() {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(9000);
    bind(sock, (struct sockaddr*)&addr, sizeof(addr));

    char buf[65535];  // Max UDP payload

    while (true) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);

        // Receive datagram + sender address
        ssize_t n = recvfrom(sock, buf, sizeof(buf), 0,
                             (struct sockaddr*)&client_addr, &addr_len);
        if (n < 0) continue;

        // Echo back to sender (no connection needed)
        sendto(sock, buf, n, 0,
               (struct sockaddr*)&client_addr, addr_len);
    }

    close(sock);
}
```

*TCP vs UDP socket differences:*

#table(
  columns: 3,
  align: (left, left, left),
  table.header([Aspect], [TCP (`SOCK_STREAM`)], [UDP (`SOCK_DGRAM`)]),
  [Setup], [`listen` + `accept`], [`bind` only],
  [Transfer], [`send` / `recv`], [`sendto` / `recvfrom`],
  [State], [Per-connection fd], [Single fd for all clients],
  [Ordering], [Guaranteed in-order], [No ordering guarantee],
  [Max payload], [Stream (no boundary)], [65,507 bytes per datagram],
)

*Performance:* UDP avoids 3-way handshake (~1 RTT saved). Single-socket model uses $O(1)$ file descriptors regardless of client count.

== Error Handling Patterns

*Common socket error codes:*

#table(
  columns: 3,
  align: (left, left, left),
  table.header([Error], [Cause], [Strategy]),
  [`ECONNRESET`], [Peer sent RST (crashed or aborted)], [Close fd, clean up session],
  [`EPIPE`], [Write to closed connection], [Close fd; suppress `SIGPIPE` with `MSG_NOSIGNAL`],
  [`ETIMEDOUT`], [Connection timed out (peer unreachable)], [Retry with backoff or fail],
  [`EADDRINUSE`], [Port already bound], [Set `SO_REUSEADDR` before `bind()`],
  [`ECONNREFUSED`], [No server listening on target port], [Retry or report to caller],
  [`EAGAIN`], [Non-blocking op would block], [Retry later (epoll/poll)],
  [`EINTR`], [Syscall interrupted by signal], [Retry the syscall immediately],
)

*Robust recv loop:* Handles partial reads, connection resets, and signals.

```cpp
#include <sys/socket.h>
#include <cerrno>
#include <cstdint>

// Read exactly `len` bytes. Returns bytes read, or -1 on error.
ssize_t recv_exact(int sockfd, uint8_t* buf, size_t len) {
    size_t total = 0;

    while (total < len) {
        ssize_t n = recv(sockfd, buf + total, len - total, 0);

        if (n > 0) {
            total += n;             // Partial read, continue
        } else if (n == 0) {
            break;                  // Peer closed connection
        } else {
            if (errno == EINTR) {
                continue;           // Signal interrupted, retry
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;              // Non-blocking: no more data now
            }
            // ECONNRESET, ETIMEDOUT, etc.
            return -1;             // Unrecoverable error
        }
    }

    return static_cast<ssize_t>(total);
}
```

*Robust send loop:* Handles partial writes and suppresses `SIGPIPE`.

```cpp
ssize_t send_all(int sockfd, const uint8_t* buf, size_t len) {
    size_t total = 0;

    while (total < len) {
        ssize_t n = send(sockfd, buf + total, len - total, MSG_NOSIGNAL);

        if (n > 0) {
            total += n;
        } else if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
            return -1;  // EPIPE, ECONNRESET, etc.
        }
    }

    return static_cast<ssize_t>(total);
}
```

*Key practices:*
- Always check return values of `send()` and `recv()` --- partial transfers are normal.
- Use `MSG_NOSIGNAL` on `send()` to avoid `SIGPIPE` killing the process (alternative: `signal(SIGPIPE, SIG_IGN)`).
- Distinguish retriable errors (`EINTR`, `EAGAIN`) from fatal errors (`ECONNRESET`, `EPIPE`).
- Set `SO_RCVTIMEO` / `SO_SNDTIMEO` to bound blocking calls and prevent indefinite hangs.

== References

POSIX.1-2017: The Open Group Base Specifications Issue 7. IEEE Std 1003.1-2017.

Stevens, W.R., Fenner, B., & Rudoff, A.M. (2003). Unix Network Programming, Volume 1: The Sockets Networking API (3rd ed.). Addison-Wesley. Chapters 4-6.

Kerrisk, M. (2010). The Linux Programming Interface. No Starch Press. Chapter 56-61.

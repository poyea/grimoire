= I/O Multiplexing

I/O multiplexing allows monitoring multiple file descriptors for readiness, enabling single-threaded servers to handle many concurrent connections.

*See also:* Sockets API (for blocking I/O), Concurrency Models (for threading alternatives), Kernel Bypass (for eliminating epoll overhead)

== The C10K Problem

*Challenge:* Handle 10,000 concurrent connections on single server [Kegel 1999].

*Naive approach fails:*
- Thread-per-connection: 10K threads × 8MB stack = 80GB memory (infeasible)
- Blocking I/O in loop: Services only 1 connection at a time

*Solution:* I/O multiplexing - monitor many sockets, process only ready ones.

== select()

*POSIX.1:* Monitor up to FD_SETSIZE (typically 1024) file descriptors.

```cpp
#include <sys/select.h>

fd_set readfds;
FD_ZERO(&readfds);
FD_SET(sockfd1, &readfds);
FD_SET(sockfd2, &readfds);

struct timeval timeout = {.tv_sec = 5, .tv_usec = 0};

int ready = select(max_fd + 1, &readfds, NULL, NULL, &timeout);

if (ready > 0) {
    if (FD_ISSET(sockfd1, &readfds)) {
        // sockfd1 is ready to read
    }
    if (FD_ISSET(sockfd2, &readfds)) {
        // sockfd2 is ready to read
    }
}
```

*Limitations:*
1. $O(n)$ scan of all FDs on each call
2. FD_SETSIZE limit (1024 typical)
3. Kernel copies FD set on each call
4. Must reconstruct FD set after each call

*Performance:* Poor for > 100 connections.

== poll()

*Improvement over select:* No FD_SETSIZE limit, cleaner API.

```cpp
#include <poll.h>

struct pollfd fds[2];
fds[0].fd = sockfd1;
fds[0].events = POLLIN;  // Monitor for read
fds[1].fd = sockfd2;
fds[1].events = POLLIN;

int ready = poll(fds, 2, 5000);  // 5 second timeout

if (ready > 0) {
    if (fds[0].revents & POLLIN) {
        // sockfd1 is ready
    }
    if (fds[1].revents & POLLIN) {
        // sockfd2 is ready
    }
}
```

*Still $O(n)$ complexity:* Kernel scans all FDs on each call.

== epoll (Linux)

*Scalable I/O multiplexing:* $O(1)$ per ready FD [Gammo et al. 2004].

```cpp
#include <sys/epoll.h>

// 1. Create epoll instance
int epollfd = epoll_create1(0);

// 2. Add socket to epoll
struct epoll_event ev;
ev.events = EPOLLIN;  // Monitor for read
ev.data.fd = sockfd;
epoll_ctl(epollfd, EPOLL_CTL_ADD, sockfd, &ev);

// 3. Wait for events
struct epoll_event events[MAX_EVENTS];
while (1) {
    int nfds = epoll_wait(epollfd, events, MAX_EVENTS, -1);  // Block until ready

    for (int i = 0; i < nfds; i++) {
        if (events[i].events & EPOLLIN) {
            int fd = events[i].data.fd;
            // Read from fd
        }
    }
}
```

*Key advantages:*
1. $O(1)$ complexity: Kernel maintains ready list, returns only ready FDs
2. Edge-triggered mode: Event fires only on state change (vs level-triggered)
3. No FD limit (scales to 100K+ connections)

*Edge-triggered vs level-triggered:*

```cpp
// Level-triggered (default): Event fires while condition true
ev.events = EPOLLIN;

// Edge-triggered: Event fires once when FD becomes ready
ev.events = EPOLLIN | EPOLLET;

// Edge-triggered requires draining FD:
while (1) {
    ssize_t n = read(fd, buf, sizeof(buf));
    if (n <= 0) break;  // EAGAIN or EOF
}
```

*Performance [Banga et al. 1999]:*
- select/poll: $O(n)$ → 100-1000 connections max
- epoll: $O(1)$ → 100K+ connections

== io_uring (Linux 5.1+)

*Modern async I/O:* Shared ring buffers, batched syscalls [Axboe 2019].

```cpp
#include <liburing.h>

struct io_uring ring;
io_uring_queue_init(256, &ring, 0);

// Submit multiple operations with single syscall
struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
io_uring_prep_accept(sqe, listenfd, NULL, NULL, 0);

io_uring_submit(&ring);  // Batch submit

// Reap completions
struct io_uring_cqe* cqe;
io_uring_wait_cqe(&ring, &cqe);
// Process cqe->res (result)
io_uring_cqe_seen(&ring, cqe);
```

*Advantages over epoll:*
1. Batching: Submit N operations with 1 syscall (vs N syscalls)
2. Zero-copy: Kernel can access userspace buffers directly
3. Unified API: Network, disk, timers (vs separate APIs)

*Performance:* 2-3x faster than epoll for high-throughput workloads.

== Comparison

#table(
  columns: 5,
  align: (left, left, right, left, left),
  table.header([Mechanism], [Complexity], [Max Connections], [Batch Support], [Zero-Copy]),
  [select], [$O(n)$], [1024], [No], [No],
  [poll], [$O(n)$], [Unlimited], [No], [No],
  [epoll], [$O(1)$], [100K+], [No], [No],
  [io_uring], [$O(1)$], [100K+], [Yes], [Yes],
)

*Recommendation:*
- Legacy systems: poll()
- Modern Linux: epoll (mature, widely supported)
- Cutting-edge: io_uring (best performance, requires Linux 5.1+)

== References

Kegel, D. (1999). "The C10K Problem." http://www.kegel.com/c10k.html

Banga, G., Druschel, P., & Mogul, J.C. (1999). "Resource Containers: A New Facility for Resource Management in Server Systems." OSDI '99.

Gammo, L., Brecht, T., Shukla, A., & Pariag, D. (2004). "Comparing and Evaluating epoll, select, and poll Event Mechanisms." Ottawa Linux Symposium.

Axboe, J. (2019). "Efficient IO with io_uring." Linux Plumbers Conference.

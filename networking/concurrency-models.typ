= Concurrency Models

Network servers must handle multiple simultaneous connections. Choice of concurrency model affects performance, scalability, and code complexity.

*See also:* I/O Multiplexing (for event-driven I/O), Lock-Free (for inter-thread communication), Sockets API (for blocking vs non-blocking)

== Thread-Per-Connection

*Simple model:* Spawn thread for each connection.

```cpp
void* handle_client(void* arg) {
    int client_sock = *(int*)arg;
    char buf[4096];

    while (1) {
        ssize_t n = recv(client_sock, buf, sizeof(buf), 0);  // Blocking
        if (n <= 0) break;
        send(client_sock, buf, n, 0);  // Echo
    }

    close(client_sock);
    return NULL;
}

int main() {
    int server = create_server(8080);

    while (1) {
        int client = accept(server, NULL, NULL);

        pthread_t thread;
        pthread_create(&thread, NULL, handle_client, &client);
        pthread_detach(thread);
    }
}
```

*Advantages:*
- Simple programming model (synchronous code)
- Natural parallelism (CPU cores utilized)

*Disadvantages:*
- Memory: Thread stack = 8MB default × 1000 threads = 8GB
- Context switches: 10K threads = scheduler overhead
- Scalability: Limited to ~10K connections

*Tuning:*
```bash
# Reduce stack size
ulimit -s 512  # 512KB per thread
```

== Thread Pool

*Improvement:* Fixed number of threads, work queue.

```cpp
ThreadPool pool(8);  // 8 worker threads

while (1) {
    int client = accept(server, NULL, NULL);
    pool.submit([client]() {
        handle_client(client);
    });
}
```

*Advantages:*
- Bounded memory (fixed threads)
- No thread creation overhead

*Disadvantages:*
- Still blocks on I/O (thread idle during network wait)
- Scalability: Limited by thread count × blocking time

== Event-Driven (Reactor Pattern)

*Single-threaded, non-blocking I/O with epoll/io_uring.*

```cpp
int epollfd = epoll_create1(0);

// Add listening socket
epoll_ctl(epollfd, EPOLL_CTL_ADD, server_sock, &ev);

while (1) {
    struct epoll_event events[MAX_EVENTS];
    int nfds = epoll_wait(epollfd, events, MAX_EVENTS, -1);

    for (int i = 0; i < nfds; i++) {
        int fd = events[i].data.fd;

        if (fd == server_sock) {
            // Accept new connection
            int client = accept(server_sock, NULL, NULL);
            set_nonblocking(client);
            epoll_ctl(epollfd, EPOLL_CTL_ADD, client, &ev);
        } else {
            // Handle client I/O
            char buf[4096];
            ssize_t n = recv(fd, buf, sizeof(buf), 0);
            if (n > 0) {
                send(fd, buf, n, 0);
            } else {
                close(fd);
                epoll_ctl(epollfd, EPOLL_CTL_DEL, fd, NULL);
            }
        }
    }
}
```

*Advantages:*
- Scalable: Single thread handles 100K+ connections
- No context switch overhead
- Low memory: No per-connection thread stack

*Disadvantages:*
- Complex programming (callback hell, state machines)
- Cannot utilize multiple CPU cores (single-threaded)
- Blocking operations (disk I/O, heavy computation) stall event loop

*Solution for multi-core:* Multiple event loops, one per core.

== Proactor Pattern

*Asynchronous I/O:* Kernel performs I/O, notifies completion.

```cpp
struct io_uring ring;
io_uring_queue_init(256, &ring, 0);

// Submit async recv
struct io_uring_sqe* sqe = io_uring_get_sqe(&ring);
io_uring_prep_recv(sqe, client_sock, buf, sizeof(buf), 0);
io_uring_sqe_set_data(sqe, ctx);  // Attach context
io_uring_submit(&ring);

// Later: Completion notification
struct io_uring_cqe* cqe;
io_uring_wait_cqe(&ring, &cqe);

Context* ctx = io_uring_cqe_get_data(cqe);
// ctx->buf contains received data
```

*Advantages over Reactor:*
- True async: Kernel performs I/O (vs userspace polling readiness)
- Batching: Submit multiple ops with single syscall

== Comparison

#table(
  columns: 5,
  align: (left, right, right, left, right),
  table.header([Model], [Connections], [CPU Cores], [Complexity], [Memory]),
  [Thread-per-connection], [10K], [Full], [Low], [High (8GB)],
  [Thread pool], [10K], [Full], [Low], [Medium (64MB)],
  [Reactor (epoll)], [100K+], [Single], [High], [Low (8MB)],
  [Reactor × cores], [100K+], [Full], [High], [Low (64MB)],
  [Proactor (io_uring)], [100K+], [Full], [Medium], [Low (64MB)],
)

*Recommendation:*
- Low concurrency (< 1K): Thread pool (simplicity)
- High concurrency (10K+): Reactor (epoll) or Proactor (io_uring)
- CPU-bound tasks: Thread pool with work-stealing

== References

Schmidt, D.C. et al. (2000). Pattern-Oriented Software Architecture Volume 2: Patterns for Concurrent and Networked Objects. Wiley.

Kegel, D. (1999). "The C10K Problem." http://www.kegel.com/c10k.html

Welsh, M., Culler, D., & Brewer, E. (2001). "SEDA: An Architecture for Well-Conditioned, Scalable Internet Services." SOSP '01.

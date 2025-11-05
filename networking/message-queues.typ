= Message Queues and ZeroMQ

Message queues abstract socket complexity, providing high-level patterns for distributed systems.

*See also:* Sockets API (for raw socket programming), Lock-Free (for local inter-thread queues), Concurrency Models (for architectural patterns)

== ZeroMQ Architecture

*ZeroMQ = sockets on steroids:* Async I/O, automatic reconnection, load balancing, built-in patterns [Hint & Sustrik 2013].

*Core concepts:*
- *No broker:* Peer-to-peer messaging (unlike RabbitMQ, Kafka)
- *Socket types:* REQ/REP, PUB/SUB, PUSH/PULL, DEALER/ROUTER
- *Transports:* TCP, IPC (Unix domain), inproc (in-process), PGM (multicast)
- *Async by default:* Non-blocking send/recv, background I/O threads

== ZeroMQ Socket Types

*1. REQ/REP (Request-Reply):*

```cpp
#include <zmq.h>

// Server
void* context = zmq_ctx_new();
void* responder = zmq_socket(context, ZMQ_REP);
zmq_bind(responder, "tcp://*:5555");

char buffer[256];
while (1) {
    zmq_recv(responder, buffer, 256, 0);  // Receive request
    // Process...
    zmq_send(responder, "World", 5, 0);   // Send reply
}
```

```cpp
// Client
void* requester = zmq_socket(context, ZMQ_REQ);
zmq_connect(requester, "tcp://localhost:5555");

zmq_send(requester, "Hello", 5, 0);      // Send request
zmq_recv(requester, buffer, 256, 0);      // Receive reply
```

*Pattern:* Lockstep - must alternate send/recv.

*2. PUB/SUB (Publish-Subscribe):*

```cpp
// Publisher
void* publisher = zmq_socket(context, ZMQ_PUB);
zmq_bind(publisher, "tcp://*:5556");

while (1) {
    zmq_send(publisher, "TOPIC data", 10, 0);  // Broadcast to all subscribers
    sleep(1);
}
```

```cpp
// Subscriber
void* subscriber = zmq_socket(context, ZMQ_SUB);
zmq_connect(subscriber, "tcp://localhost:5556");
zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "TOPIC", 5);  // Topic filter

while (1) {
    char buffer[256];
    zmq_recv(subscriber, buffer, 256, 0);  // Receive matching messages
}
```

*Characteristics:*
- One-way (pub doesn't know about subs)
- No delivery guarantee (fire-and-forget)
- Topic-based filtering

*3. PUSH/PULL (Pipeline):*

```cpp
// Producer (PUSH)
void* pusher = zmq_socket(context, ZMQ_PUSH);
zmq_bind(pusher, "tcp://*:5557");

for (int i = 0; i < 100; i++) {
    zmq_send(pusher, &i, sizeof(i), 0);  // Distribute to workers
}
```

```cpp
// Worker (PULL)
void* puller = zmq_socket(context, ZMQ_PULL);
zmq_connect(puller, "tcp://localhost:5557");

int task;
while (1) {
    zmq_recv(puller, &task, sizeof(task), 0);  // Receive task
    process(task);
}
```

*Load balancing:* PUSH distributes round-robin to connected PULLs.

== ZeroMQ Performance

*Latency (localhost, single message):*
- TCP sockets: 10-30μs
- ZeroMQ (inproc): 1-3μs (shared memory)
- ZeroMQ (IPC): 5-10μs (Unix domain socket)
- ZeroMQ (TCP): 15-40μs (includes ZeroMQ framing overhead)

*Throughput (bulk transfer):*
- ZeroMQ (inproc): 10-20M msgs/sec (lock-free queue)
- ZeroMQ (TCP): 1-5M msgs/sec (depends on message size)

*Overhead:* ZeroMQ adds ~5-10μs latency vs raw sockets, but simplifies code 10x.

== Multipart Messages

*Atomic message groups:*

```cpp
// Send multipart (header + body)
zmq_send(socket, "Header", 6, ZMQ_SNDMORE);  // More parts follow
zmq_send(socket, "Body", 4, 0);               // Last part

// Receive multipart
char header[64], body[1024];
zmq_recv(socket, header, 64, 0);

int more;
size_t more_size = sizeof(more);
zmq_getsockopt(socket, ZMQ_RCVMORE, &more, &more_size);

if (more) {
    zmq_recv(socket, body, 1024, 0);
}
```

*Use case:* Message envelope (routing info + payload).

== High-Water Mark (HWM)

*Buffer limit:* Max queued messages before blocking or dropping.

```cpp
int hwm = 1000;  // Queue up to 1000 messages
zmq_setsockopt(socket, ZMQ_SNDHWM, &hwm, sizeof(hwm));
zmq_setsockopt(socket, ZMQ_RCVHWM, &hwm, sizeof(hwm));

// When HWM reached:
// - REQ/DEALER: Send blocks
// - PUB: Messages dropped (no backpressure)
// - PUSH: Send blocks until receiver drains
```

*Tuning:* Large HWM = more memory, better burst tolerance. Small HWM = less memory, risk blocking/drops.

== ZeroMQ vs Raw Sockets

*ZeroMQ advantages:*
1. *Automatic reconnection:* Handles network failures transparently
2. *Load balancing:* Built-in round-robin for PUSH/PULL
3. *Patterns:* High-level abstractions (pub-sub, pipeline)
4. *Multi-transport:* Switch between TCP/IPC/inproc without code changes

*Raw socket advantages:*
1. *Lower latency:* No ZeroMQ framing overhead
2. *Kernel integration:* Works with iptables, traffic control
3. *Standard:* Portable across all Unix systems

*Decision:* Use ZeroMQ for distributed systems (microservices, clusters). Use raw sockets for kernel-integrated or ultra-low-latency applications.

== Practical Example: Worker Pool

```cpp
// Ventilator (produces tasks)
void* ventilator = zmq_socket(context, ZMQ_PUSH);
zmq_bind(ventilator, "tcp://*:5557");

for (int i = 0; i < 1000; i++) {
    int workload = rand() % 100;
    zmq_send(ventilator, &workload, sizeof(workload), 0);
}

// Worker (processes tasks)
void* receiver = zmq_socket(context, ZMQ_PULL);
zmq_connect(receiver, "tcp://localhost:5557");

void* sender = zmq_socket(context, ZMQ_PUSH);
zmq_connect(sender, "tcp://localhost:5558");

int task;
while (1) {
    zmq_recv(receiver, &task, sizeof(task), 0);
    usleep(task * 1000);  // Simulate work
    int result = task * 2;
    zmq_send(sender, &result, sizeof(result), 0);
}

// Sink (collects results)
void* sink = zmq_socket(context, ZMQ_PULL);
zmq_bind(sink, "tcp://*:5558");

for (int i = 0; i < 1000; i++) {
    int result;
    zmq_recv(sink, &result, sizeof(result), 0);
    printf("Result: %d\n", result);
}
```

*Scalability:* Add workers by running more worker processes - no code changes needed.

== References

Hintjens, P. & Sustrik, M. (2013). ZeroMQ: Messaging for Many Applications. O'Reilly Media.

ZeroMQ Project (2023). ZeroMQ Guide. https://zguide.zeromq.org/

Sustrik, M. (2012). "How to make messaging queues perform." Blog post. https://www.250bpm.com/

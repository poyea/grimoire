= WebSockets

WebSocket is a full-duplex communication protocol that provides persistent, bidirectional communication between client and server over a single TCP connection [RFC 6455].

*See also:* Application Protocols (for HTTP foundations), Transport Layer (for TCP internals), I/O Multiplexing (for handling many WebSocket connections)

== Motivation and Use Cases

*Problem with HTTP:* Request-response model requires client to initiate all communication. Server cannot push data proactively.

*Workarounds before WebSocket:*
1. *Polling:* Client repeatedly requests updates (wasteful: many empty responses)
2. *Long polling:* Server holds request open until data available (connection churn)
3. *HTTP streaming:* Server keeps response open (half-duplex, proxy issues)

*WebSocket advantages:*
- Full-duplex: Both sides send independently
- Low overhead: 2-14 byte frame header vs 500-800 byte HTTP headers
- Persistent: Single TCP connection for lifetime of communication
- Real-time: Sub-millisecond message delivery after connection established

*Ideal use cases:*
- Real-time applications: Chat, collaborative editing, live dashboards
- Gaming: Multiplayer state synchronization
- Financial: Stock tickers, trading platforms
- IoT: Device telemetry, command-and-control

== HTTP Upgrade Handshake

WebSocket connections begin as HTTP/1.1 requests with an Upgrade header [RFC 6455 Section 4].

*Client request:*
```http
GET /chat HTTP/1.1
Host: server.example.com
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13
Origin: http://example.com
Sec-WebSocket-Protocol: chat, superchat
```

*Server response (101 Switching Protocols):*
```http
HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
Sec-WebSocket-Protocol: chat
```

*Key generation:* Server computes `Sec-WebSocket-Accept` to prove it understands WebSocket:

```
Accept = Base64(SHA-1(Key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"))
```

The magic GUID prevents caching proxies from replaying old responses.

*Handshake latency:* 1 RTT (same as HTTP request). After handshake, protocol switches to binary WebSocket frames.

== WebSocket Frame Format

All WebSocket communication uses a binary frame format [RFC 6455 Section 5]:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-------+-+-------------+-------------------------------+
|F|R|R|R| opcode|M| Payload len |    Extended payload length    |
|I|S|S|S|  (4)  |A|     (7)     |           (16/64)             |
|N|V|V|V|       |S|             |   (if payload len==126/127)   |
| |1|2|3|       |K|             |                               |
+-+-+-+-+-------+-+-------------+ - - - - - - - - - - - - - - - +
|     Extended payload length continued, if payload len == 127  |
+ - - - - - - - - - - - - - - - +-------------------------------+
|                               |Masking-key, if MASK set       |
+-------------------------------+-------------------------------+
|          Masking-key (continued)                              |
+-------------------------------- - - - - - - - - - - - - - - - +
|                     Payload Data                              |
+---------------------------------------------------------------+
```

*Frame header fields:*

- *FIN (1 bit):* Final fragment of message (1 = complete message or last fragment)
- *RSV1-3 (3 bits):* Reserved for extensions (must be 0 unless negotiated)
- *Opcode (4 bits):* Frame type (see below)
- *MASK (1 bit):* Payload is masked (client-to-server frames MUST be masked)
- *Payload length:* 7-bit length, or 126 (16-bit follows), or 127 (64-bit follows)
- *Masking key (4 bytes):* XOR key for payload (present if MASK=1)

*Opcodes:*
```
0x0  Continuation frame (fragmented message continues)
0x1  Text frame (UTF-8 encoded)
0x2  Binary frame
0x8  Connection close
0x9  Ping
0xA  Pong
```

*Header sizes:*
- Minimum: 2 bytes (unmasked, payload <= 125 bytes)
- Client frames: +4 bytes for masking key
- Extended length: +2 bytes (126) or +8 bytes (127)

== Masking

*Why masking?* Prevents cache poisoning attacks on intermediary proxies [RFC 6455 Section 10.3].

*Attack scenario without masking:*
1. Attacker crafts WebSocket payload that looks like valid HTTP response
2. Intermediary proxy caches the "response"
3. Other clients receive poisoned cached content

*Masking algorithm:*
```cpp
// Client generates random 4-byte masking key
uint32_t masking_key = random();

// XOR each payload byte with corresponding key byte
for (size_t i = 0; i < payload_len; i++) {
    masked[i] = payload[i] ^ ((uint8_t*)&masking_key)[i % 4];
}
```

*Performance note:* Masking is cheap (XOR), but prevents zero-copy send on clients. Server-to-client frames are unmasked.

== Message Fragmentation

Large messages can be split across multiple frames [RFC 6455 Section 5.4]:

```
Frame 1: FIN=0, opcode=0x1 (text), payload="Hello "
Frame 2: FIN=0, opcode=0x0 (continuation), payload="World"
Frame 3: FIN=1, opcode=0x0 (continuation), payload="!"
```

*Reassembled message:* "Hello World!"

*Use cases:*
- Streaming large payloads without buffering entire message
- Interleaving control frames (ping/pong) with data frames
- Memory-constrained environments

*Interleaving rules:*
- Control frames (ping, pong, close) CAN be injected between data fragments
- Data frames from different messages CANNOT interleave

== Ping/Pong Keepalive

*Purpose:* Detect dead connections, prevent NAT/firewall timeout [RFC 6455 Section 5.5.2].

*Ping frame:*
```
FIN=1, opcode=0x9, payload=<optional application data>
```

*Pong response:*
```
FIN=1, opcode=0xA, payload=<echo ping payload>
```

*Protocol requirements:*
- Endpoint receiving Ping MUST respond with Pong "as soon as practical"
- Pong payload MUST match Ping payload exactly
- Unsolicited Pong frames are allowed (serves as unidirectional heartbeat)

*Typical intervals:*
- Client-side ping: 30-60 seconds
- Server-side ping: 30-60 seconds
- Timeout after missed pongs: 2-3 intervals

*Implementation pattern:*
```cpp
// Server-side keepalive
void websocket_keepalive(ws_connection* conn) {
    uint64_t now = time_ms();
    if (now - conn->last_pong > PONG_TIMEOUT_MS) {
        ws_close(conn, 1001, "Pong timeout");
        return;
    }
    if (now - conn->last_ping > PING_INTERVAL_MS) {
        ws_send_ping(conn, &now, sizeof(now));  // Timestamp as payload
        conn->last_ping = now;
    }
}
```

== Connection Closure

*Clean close handshake [RFC 6455 Section 7]:*

```
Initiator                        Responder
    |                               |
    |  Close frame (status, reason) |
    |------------------------------>|
    |                               |
    |  Close frame (status, reason) |
    |<------------------------------|
    |                               |
    |  TCP FIN                      |
    |------------------------------>|
```

*Common status codes:*
- 1000: Normal closure
- 1001: Going away (server shutdown, browser navigating away)
- 1002: Protocol error
- 1003: Unsupported data type
- 1006: Abnormal closure (no close frame received - reserved, never sent)
- 1008: Policy violation
- 1009: Message too big
- 1011: Server error

== Comparison with Alternatives

#table(
  columns: 5,
  align: (left, left, left, left, left),
  table.header([Feature], [WebSocket], [SSE], [Long Polling], [HTTP/2 Push]),
  [Direction], [Bidirectional], [Server-to-client], [Bidirectional\*], [Server-to-client],
  [Connection], [Persistent], [Persistent], [Reconnects], [Persistent],
  [Overhead], [2-14 bytes], [~50 bytes], [~800 bytes], [9 bytes],
  [Browser support], [Universal], [Universal], [Universal], [Universal],
  [Binary data], [Yes], [No (text only)], [Base64], [Yes],
  [Proxy friendly], [Moderate], [Good], [Excellent], [Good],
)

\*Long polling is half-duplex per connection (client sends separate request).

*When to use WebSocket:*
- Bidirectional real-time communication required
- High message frequency (> 1 msg/sec)
- Binary data transfer
- Gaming, trading, collaborative apps

*When to use Server-Sent Events (SSE):*
- Server-to-client only (notifications, feeds)
- Text-based data sufficient
- Simpler implementation needed
- Auto-reconnection desirable (built into EventSource API)

*When to use HTTP/2 streams:*
- Already using HTTP/2 infrastructure
- Request-response pattern with server push hints
- CDN/proxy optimization important

== Performance Characteristics

*Latency comparison (message delivery after connection established):*

```
WebSocket:        < 1ms (frame overhead only)
SSE:              < 1ms (similar)
Long polling:     50-500ms (new HTTP request per message)
Short polling:    polling_interval / 2 average
```

*Throughput (messages per second, single connection):*

```
WebSocket:        100K+ msg/s (limited by TCP)
SSE:              50K+ msg/s (text parsing overhead)
Long polling:     ~100 msg/s (connection setup overhead)
```

*Connection establishment overhead:*

```
WebSocket:        1 RTT (HTTP upgrade) + TCP handshake
SSE:              1 RTT + TCP handshake
Long polling:     1 RTT + TCP handshake PER MESSAGE
HTTP/2:           0 RTT (stream on existing connection)
```

*Memory per connection:*
- WebSocket: 4-16 KB (socket buffers + application state)
- SSE: Similar
- Long polling: Same, but connection churn increases GC pressure

== Scalability Considerations

*Connection limits:*
- Linux default: ~1M connections per server (with tuning)
- File descriptor limit: `ulimit -n` (increase to 1M+)
- Memory: Primary constraint at scale

*Scaling patterns:*

*1. Horizontal scaling with sticky sessions:*
```
Client -> Load Balancer (IP hash) -> Server N
```
WebSocket requires session affinity - same client must reach same server.

*2. Pub/sub backbone:*
```
Clients <-> WS Servers <-> Redis/Kafka <-> WS Servers <-> Clients
```
Enables cross-server message delivery.

*3. Connection pooling (for backend services):*
Reuse WebSocket connections between services; don't create per-request.

*Benchmark reference [TechEmpower 2023]:*
- Top WebSocket frameworks: 500K+ concurrent connections per server
- Message throughput: 1M+ messages/second (aggregate)

== Security Considerations

*1. Origin validation [RFC 6455 Section 10.2]:*
```cpp
// Server MUST check Origin header
const char* origin = get_header(req, "Origin");
if (!is_allowed_origin(origin)) {
    return http_error(403, "Forbidden");
}
```
Unlike HTTP cookies, WebSocket has no same-origin restriction by default.

*2. Use WSS (WebSocket Secure):*
```
wss://server.example.com/socket
```
- Encrypts all traffic (TLS)
- Prevents MITM attacks
- Required for production deployments
- Works through corporate proxies (port 443)

*3. Authentication:*
- Pass token in query string: `wss://server/socket?token=xxx`
- Or in first message after connection
- Or via cookie (sent during HTTP upgrade)

*4. Rate limiting:*
```cpp
if (conn->messages_this_second > MAX_MESSAGES_PER_SECOND) {
    ws_close(conn, 1008, "Rate limit exceeded");
}
```

*5. Message size limits:*
```cpp
if (frame->payload_length > MAX_MESSAGE_SIZE) {
    ws_close(conn, 1009, "Message too big");
}
```

*6. Input validation:*
- Validate UTF-8 for text frames
- Validate JSON/protocol structure
- Sanitize before storage or broadcast

== Implementation Notes

*Client (JavaScript):*
```javascript
const ws = new WebSocket('wss://server.example.com/chat');

ws.onopen = () => {
    ws.send(JSON.stringify({type: 'subscribe', channel: 'updates'}));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleMessage(data);
};

ws.onclose = (event) => {
    console.log(`Closed: ${event.code} ${event.reason}`);
    // Implement reconnection logic
};
```

*Server (C/Linux pseudocode):*
```cpp
// After HTTP upgrade handshake
while (1) {
    ws_frame frame;
    int ret = ws_read_frame(conn, &frame);

    if (frame.opcode == WS_OPCODE_PING) {
        ws_send_pong(conn, frame.payload, frame.payload_len);
    } else if (frame.opcode == WS_OPCODE_CLOSE) {
        ws_send_close(conn, 1000, "Normal closure");
        break;
    } else if (frame.opcode == WS_OPCODE_TEXT) {
        handle_message(conn, frame.payload, frame.payload_len);
    }
}
```

*Common libraries:*
- C: libwebsockets, uWebSockets
- Go: gorilla/websocket, nhooyr/websocket
- Rust: tokio-tungstenite, actix-web
- Node.js: ws, Socket.IO (with fallbacks)
- Python: websockets, aiohttp

== References

RFC 6455: The WebSocket Protocol. Fette, I. & Melnikov, A. (2011).

RFC 7692: Compression Extensions for WebSocket. Yoshino, T. (2015).

RFC 8441: Bootstrapping WebSockets with HTTP/2. McManus, P. (2018).

Lubbers, P. & Greco, F. (2010). "HTML5 WebSocket: A Quantum Leap in Scalability for the Web." SOA World Magazine.

Wang, V., Salim, F., & Moskovits, P. (2013). The Definitive Guide to HTML5 WebSocket. Apress.

#import "../template.typ": xref

= Application Protocols

Application-layer protocols define message formats and communication patterns for specific services.

*See also:* #xref("networking", "transport-layer", label: "Transport Layer") (for TCP/UDP foundations), #xref("networking", "sockets-api", label: "Sockets API") (for implementation), _TLS_ (for encryption layer)

== HTTP (Hypertext Transfer Protocol)

*Request/response protocol over TCP [RFC 7230-7235].*

*HTTP/1.1 request:*
```http
GET /index.html HTTP/1.1\r\n
Host: www.example.com\r\n
User-Agent: Mozilla/5.0\r\n
Accept: text/html\r\n
Connection: keep-alive\r\n
\r\n
```

*HTTP/1.1 response:*
```http
HTTP/1.1 200 OK\r\n
Content-Type: text/html\r\n
Content-Length: 1234\r\n
Connection: keep-alive\r\n
\r\n
<html>...</html>
```

*HTTP/1.1 features:*
- *Persistent connections:* Reuse TCP connection for multiple requests (vs new connection per request in HTTP/1.0)
- *Pipelining:* Send multiple requests without waiting for responses (rarely used - head-of-line blocking)
- *Chunked transfer encoding:* Stream response without knowing length upfront

*Performance limitations:*
- Head-of-line blocking: One slow response blocks subsequent requests
- Text-based parsing: CPU overhead for parsing headers
- No compression: Headers repeated for every request

== HTTP/2

*Binary, multiplexed protocol [RFC 7540].*

*Key improvements:*
1. *Binary framing:* Efficient parsing (vs text-based HTTP/1.1)
2. *Multiplexing:* Multiple requests/responses interleaved on single connection
3. *Server push:* Server initiates sending resources (before client requests)
4. *Header compression:* HPACK algorithm reduces overhead by 80-90%

*Stream prioritization:*
```
Client sends:
- Stream 1: GET /style.css (priority 10)
- Stream 2: GET /image.jpg (priority 5)

Server sends Stream 1 first (higher priority)
```

*Performance:* 20-40% faster page load vs HTTP/1.1 [Grigorik 2013].

== HTTP/3 (QUIC)

*HTTP over QUIC (UDP-based) [RFC 9114].*

*Advantages over HTTP/2:*
1. *0-RTT connection resumption:* Faster than TCP+TLS handshake
2. *No head-of-line blocking:* Loss in one stream doesn't block others
3. *Connection migration:* Survives IP address changes (mobile networks)

*Adoption:* Google, Facebook, Cloudflare (~30% of internet traffic, 2023).

== DNS (Domain Name System)

*Hierarchical name resolution [RFC 1035].*

*Query flow:*
```
1. Client → Resolver: "www.example.com A?"
2. Resolver → Root: ".com NS?"
3. Resolver → .com: "example.com NS?"
4. Resolver → example.com: "www A?"
5. Resolver → Client: "93.184.216.34"
```

*Record types:*
- A: IPv4 address
- AAAA: IPv6 address
- CNAME: Canonical name (alias)
- MX: Mail exchanger
- NS: Name server
- TXT: Text (SPF, DKIM, etc.)

*Performance:* Typical query 10-50ms. *Optimization:* Caching (TTL-based), local resolver.

== TLS (Transport Layer Security)

*Encryption layer above TCP [RFC 8446].*

*TLS 1.3 handshake (1-RTT):*
```
Client → Server: ClientHello + key_share
Server → Client: ServerHello + key_share + {EncryptedExtensions, Certificate, CertificateVerify, Finished}
Client → Server: {Finished}
[Application data can flow]
```

*0-RTT resumption:* Client sends encrypted data in first flight (reusing previous session).

*Performance cost:*
- Handshake: 1 RTT = 20-100ms depending on distance
- Encryption overhead: 5-15% CPU (AES-GCM), 1-3% with AES-NI hardware

== WebSocket

*Full-duplex framing over a single TCP connection [RFC 6455].*

WebSocket starts as an HTTP/1.1 `Upgrade` handshake and then switches to a framed binary protocol on the same connection. After the upgrade, either side can send frames at any time — eliminating the request/response constraint of HTTP. This makes it the standard transport for real-time applications (chat, collaborative editing, live dashboards, game state).

*Upgrade handshake:*
```http
GET /ws HTTP/1.1
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13

HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
```

*Frame format* (2-byte minimum header): FIN bit, opcode (text/binary/ping/pong/close), mask bit, payload length (7, 16, or 64 bits). Client-to-server frames must be masked with a random key to prevent cache-poisoning attacks on HTTP proxies.

*Trade-offs:* WebSocket adds no multiplexing (head-of-line blocking per connection), no built-in flow control, and no typed schemas. For server-to-client-only push, *Server-Sent Events* (SSE, `text/event-stream`) is simpler. For bidirectional structured RPC, gRPC or WebTransport (QUIC-based) supersede it.

== MQTT

*Publish-subscribe protocol for constrained devices [OASIS Standard, v5.0].*

MQTT is a lightweight pub/sub protocol designed for IoT sensors, embedded systems, and mobile clients on lossy links. A central *broker* (Mosquitto, EMQX, HiveMQ) routes messages from publishers to subscribers by *topic* — a `/`-delimited string (`home/bedroom/temperature`). Clients subscribe with optional wildcards (`+` single-level, `#` multi-level).

*Three QoS levels:*
#table(columns: (auto, auto, 1fr),
  [*Level*], [*Name*], [*Guarantee*],
  [0], [At most once], [Fire and forget — no ACK, possible loss],
  [1], [At least once], [Publisher retries until PUBACK — possible duplicates],
  [2], [Exactly once], [4-way handshake (PUBLISH → PUBREC → PUBREL → PUBCOMP)],
)

*Retained messages:* the broker stores the last message per topic and delivers it immediately to new subscribers — useful for "last known state" semantics. *Will messages:* a client registers a message to be published if it disconnects unexpectedly (dead man's switch for device monitoring).

MQTT v5 added topic aliases (reduce wire size), message expiry, shared subscriptions (load-balanced consumer groups analogous to Kafka consumer groups), and reason codes on CONNACK/PUBACK.

*Performance:* 2-byte minimum header; a $"QoS"=0$ publish is a single TCP write. Brokers handle millions of concurrent connections by keeping per-session state in memory.

== gRPC

*High-performance RPC framework using HTTP/2 and Protocol Buffers.*

gRPC (Google, 2015) defines services and message types in `.proto` files and generates typed client/server stubs in any supported language. The wire encoding is Protocol Buffers (binary, schema-driven, ~5–10× smaller than equivalent JSON). Transport is HTTP/2, providing multiplexing, header compression, and built-in flow control.

*Service definition:*
```protobuf
service UserService {
  rpc GetUser (GetUserRequest) returns (User);
  rpc ListUsers (ListRequest) returns (stream User);
  rpc BatchCreate (stream CreateRequest) returns (BatchResult);
  rpc Chat (stream Message) returns (stream Message);
}
```

*Four call types:*
- *Unary:* one request, one response (replaces REST GET/POST).
- *Server streaming:* one request, N responses (result pagination, live feeds).
- *Client streaming:* N requests, one response (bulk upload, aggregation).
- *Bidirectional streaming:* N requests, M responses interleaved (real-time chat, game state, collaborative editing).

*Key features:* deadline propagation (every RPC carries a deadline that is honoured through call chains), cancellation (an HTTP/2 RST_STREAM tears down all in-flight work), interceptors (middleware for auth, logging, metrics), and load balancing via per-RPC header-based routing. *grpc-gateway* translates REST+JSON ↔ gRPC for clients that cannot speak HTTP/2.

*Trade-offs:* gRPC requires HTTP/2; browser support uses grpc-web (a proxy translates between HTTP/1.1 and HTTP/2). Schema evolution must preserve field numbers (adding fields is safe; renaming is not). Debugging binary frames requires tooling (`grpcurl`, Wireshark gRPC dissector).

== References

RFC 7230: Hypertext Transfer Protocol (HTTP/1.1): Message Syntax and Routing. Fielding, R. & Reschke, J. (2014).

RFC 7540: Hypertext Transfer Protocol Version 2 (HTTP/2). Belshe, M., Peon, R., & Thomson, M. (2015).

RFC 9114: HTTP/3. Bishop, M. (2022).

RFC 6455: The WebSocket Protocol. Fette, I. & Melnikov, A. (2011).

OASIS MQTT Version 5.0 Standard. (2019).

Google. gRPC Documentation. https://grpc.io/docs/

Grigorik, I. (2013). High Performance Browser Networking. O'Reilly Media.

== Further Reading

IETF. (2015). "Hypertext Transfer Protocol Version 2 (HTTP/2)." RFC 7540. Specifies binary framing, multiplexing, header compression (HPACK), and server push — the foundational changes that address HTTP/1.1's head-of-line blocking.

IETF. (2022). "HTTP/3." RFC 9114. Defines HTTP semantics over QUIC, eliminating TCP-level head-of-line blocking and enabling connection migration for mobile clients.

Fette, I., Melnikov, A. (2011). "The WebSocket Protocol." RFC 6455. Standardises the HTTP upgrade handshake and framing layer for full-duplex communication over a single TCP connection.

OASIS. (2019). "MQTT Version 5.0." OASIS Standard. Specifies the publish-subscribe protocol for constrained IoT devices, including QoS levels, retained messages, and the v5 extensions for shared subscriptions and message expiry.

Google. (2015). "gRPC: A High Performance, Open Source Universal RPC Framework." https://grpc.io/. Documents the HTTP/2-based RPC framework with Protocol Buffers encoding, deadline propagation, and bidirectional streaming.

Grigorik, I. (2013). _High Performance Browser Networking._ O'Reilly Media. Covers the full network stack from TCP through HTTP/2 and WebSocket with practical performance guidance; an accessible bridge between protocol specs and real-world engineering.

Iyengar, J., Thomson, M. (2021). "QUIC: A UDP-Based Multiplexed and Secure Transport." RFC 9000. Defines the QUIC transport protocol underlying HTTP/3, with integrated TLS 1.3, stream multiplexing, and loss recovery without TCP's head-of-line blocking.

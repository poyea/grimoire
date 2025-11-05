= Application Protocols

Application-layer protocols define message formats and communication patterns for specific services.

*See also:* Transport Layer (for TCP/UDP foundations), Sockets API (for implementation)

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

== References

RFC 7230: Hypertext Transfer Protocol (HTTP/1.1): Message Syntax and Routing. Fielding, R. & Reschke, J. (2014).

RFC 7540: Hypertext Transfer Protocol Version 2 (HTTP/2). Belshe, M., Peon, R., & Thomson, M. (2015).

RFC 9114: HTTP/3. Bishop, M. (2022).

Grigorik, I. (2013). High Performance Browser Networking. O'Reilly Media.

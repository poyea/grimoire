= gRPC (Remote Procedure Calls)

gRPC is a high-performance, cross-platform RPC framework built on HTTP/2 and Protocol Buffers.

*See also:* Application Protocols (for HTTP/2 foundations), Transport Layer (for TCP/QUIC), Message Queues (for alternative messaging patterns)

== Overview

*gRPC characteristics [grpc.io]:*
- Binary serialization (Protocol Buffers)
- HTTP/2 transport (multiplexing, flow control, header compression)
- Bidirectional streaming
- Built-in code generation
- Language-agnostic (C++, Java, Go, Python, etc.)

*Use cases:*
- Microservices communication (low latency, strong typing)
- Mobile clients (efficient bandwidth, battery usage)
- Real-time streaming (live data feeds, chat)
- Polyglot systems (cross-language service mesh)

== Protocol Buffers (Protobuf) Encoding

*Binary wire format:* Each field encoded as (field_number << 3 | wire_type) + value.

*Wire types:*
```
Type  Name            Used For
────────────────────────────────────────────────
0     Varint          int32, int64, uint32, uint64, sint32, sint64, bool, enum
1     64-bit          fixed64, sfixed64, double
2     Length-delim    string, bytes, embedded messages, packed repeated
5     32-bit          fixed32, sfixed32, float
```

*Varint encoding:* Variable-length integers (MSB continuation bit).

```
Value: 300 = 0x012C = 100101100 binary

Varint encoding:
  10101100 00000010
  ^        ^
  |        +-- High bits (0x02)
  +-- Low 7 bits + continuation (0xAC)

Wire bytes: AC 02
```

*Message example:*
```protobuf
message Person {
  string name = 1;    // field 1, wire type 2 (length-delimited)
  int32 age = 2;      // field 2, wire type 0 (varint)
}
```

*Encoded as:*
```
 0                   1                   2
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3
├─────────────────────────────────────────────────┤
│ 0A │ 05 │ 'A' 'l' 'i' 'c' 'e' │ 10 │ 1E │
├────┴────┴──────────────────────┴────┴────┤
  0A = (1 << 3) | 2 = field 1, wire type 2 (string)
  05 = length 5
  10 = (2 << 3) | 0 = field 2, wire type 0 (varint)
  1E = 30 (age)
```

*Size comparison (encoding "Alice", age 30):*
- JSON: `{"name":"Alice","age":30}` = 26 bytes
- Protobuf: 9 bytes (65% smaller)

== gRPC over HTTP/2

*HTTP/2 provides:*
1. *Binary framing:* Efficient parsing (vs text-based HTTP/1.1)
2. *Multiplexing:* Multiple RPCs over single TCP connection
3. *Flow control:* Per-stream and connection-level
4. *Header compression:* HPACK reduces metadata overhead

*gRPC frame structure:*

```
┌─────────────────────────────────────────────────────────────────┐
│                      HTTP/2 HEADERS frame                       │
│  :method = POST                                                 │
│  :path = /package.Service/Method                                │
│  :scheme = https                                                │
│  content-type = application/grpc                                │
│  te = trailers                                                  │
│  grpc-timeout = 10S                                             │
├─────────────────────────────────────────────────────────────────┤
│                      HTTP/2 DATA frame                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Compressed │   Message     │      Protobuf Message      │    │
│  │   Flag     │   Length      │        (N bytes)           │    │
│  │  (1 byte)  │  (4 bytes)    │                            │    │
│  └─────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                     HTTP/2 HEADERS frame (trailers)             │
│  grpc-status = 0                                                │
│  grpc-message = (optional error message)                        │
└─────────────────────────────────────────────────────────────────┘
```

*Length-Prefixed Message (LPM) format:*
```
 0                   1                   2                   3     4+
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1   ...
├───────────────┬───────────────────────────────────────────────┬─────────┐
│  Compressed   │              Message Length                   │ Message │
│    (1 bit)    │              (4 bytes, BE)                    │  Data   │
└───────────────┴───────────────────────────────────────────────┴─────────┘
```

*Multiplexing benefit:*
```
Single TCP Connection:
  ┌──────────────────────────────────────────────────────────────┐
  │ Stream 1: GetUser()     ──▶ ◀── Response                    │
  │ Stream 3: ListOrders()  ──▶ ◀── Response ◀── Response ◀──   │
  │ Stream 5: UpdateCart()  ──▶ ◀── Response                    │
  └──────────────────────────────────────────────────────────────┘
  No head-of-line blocking at HTTP layer (still exists at TCP layer)
```

== Streaming Modes

*1. Unary RPC:* Single request, single response (traditional RPC).

```protobuf
service UserService {
  rpc GetUser(GetUserRequest) returns (User);
}
```

```
Client                          Server
  │  HEADERS + DATA (request)     │
  ├────────────────────────────▶ │
  │                               │
  │  HEADERS + DATA (response)    │
  │  + TRAILERS                   │
  │ ◀────────────────────────────┤
```

*2. Server streaming:* Single request, stream of responses.

```protobuf
service StockService {
  rpc StreamPrices(StockRequest) returns (stream PriceUpdate);
}
```

```
Client                          Server
  │  HEADERS + DATA (request)     │
  ├────────────────────────────▶ │
  │                               │
  │ ◀── DATA (price 1)           │
  │ ◀── DATA (price 2)           │
  │ ◀── DATA (price 3)           │
  │ ◀── TRAILERS                 │
```

*Use case:* Real-time feeds, pagination, large result sets.

*3. Client streaming:* Stream of requests, single response.

```protobuf
service UploadService {
  rpc UploadFile(stream FileChunk) returns (UploadStatus);
}
```

```
Client                          Server
  │  HEADERS                      │
  ├────────────────────────────▶ │
  │  DATA (chunk 1) ──▶          │
  │  DATA (chunk 2) ──▶          │
  │  DATA (chunk 3) ──▶          │
  │  END_STREAM                  │
  │                               │
  │ ◀── DATA (status) + TRAILERS │
```

*Use case:* File uploads, batch writes, aggregation.

*4. Bidirectional streaming:* Independent request and response streams.

```protobuf
service ChatService {
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}
```

```
Client                          Server
  │  HEADERS                      │
  ├────────────────────────────▶ │
  │  DATA ──▶     ◀── DATA       │
  │  DATA ──▶     ◀── DATA       │
  │       ◀── DATA               │
  │  DATA ──▶                    │
  │  END_STREAM   ◀── DATA       │
  │              ◀── TRAILERS    │
```

*Use case:* Chat, gaming, collaborative editing.

== Deadlines and Cancellation

*Deadlines:* Time limit for RPC completion (propagates through call chain).

```cpp
// C++ client setting deadline
grpc::ClientContext context;
context.set_deadline(std::chrono::system_clock::now() +
                     std::chrono::seconds(10));

Status status = stub->GetUser(&context, request, &response);
if (status.error_code() == grpc::DEADLINE_EXCEEDED) {
    // Handle timeout
}
```

*Wire format:* `grpc-timeout` header (e.g., "10S" = 10 seconds, "500m" = 500 milliseconds).

*Deadline propagation:*
```
Client (deadline: 10s)
    │
    ├── Service A (remaining: 9.5s)
    │       │
    │       └── Service B (remaining: 9.0s)
    │               │
    │               └── Service C (remaining: 8.5s)
```

*Cancellation:* Client or server can abort RPC mid-stream.

```cpp
// Client cancellation
context.TryCancel();

// Server-side check
if (context->IsCancelled()) {
    return Status(grpc::CANCELLED, "Client cancelled");
}
```

*Benefits:*
- Prevents wasted work on timed-out requests
- Releases resources (connections, memory)
- Enables graceful degradation under load

== Metadata

*Key-value pairs sent with requests/responses (like HTTP headers).*

```cpp
// Client sending metadata
grpc::ClientContext context;
context.AddMetadata("x-request-id", "abc123");
context.AddMetadata("authorization", "Bearer token");

// Server reading metadata
const auto& metadata = context->client_metadata();
auto it = metadata.find("x-request-id");
if (it != metadata.end()) {
    std::string request_id(it->second.data(), it->second.size());
}
```

*Binary metadata:* Keys ending in `-bin` are base64-encoded.

```cpp
context.AddMetadata("x-trace-bin", binary_trace_context);
```

*Use cases:*
- Authentication tokens
- Request tracing (OpenTelemetry)
- Custom routing hints
- Compression preferences

== Performance Characteristics

*gRPC vs REST/JSON latency (typical microservices):*

```
Metric              gRPC/Protobuf     REST/JSON
─────────────────────────────────────────────────
Serialization       1-5 us            10-50 us
Deserialization     1-5 us            20-100 us
Message size        1x (baseline)     2-10x larger
Connection setup    Shared (HTTP/2)   Per-request (HTTP/1.1)
Throughput          100K+ RPS         10-50K RPS
```

*Benchmark results [various sources]:*
- Latency: gRPC 30-50% lower than REST for small messages
- Throughput: gRPC 2-10x higher (binary encoding + connection reuse)
- CPU usage: Protobuf parsing ~10x faster than JSON
- Bandwidth: Protobuf 50-90% smaller than JSON

*When REST may be preferable:*
- Browser clients (gRPC-Web adds complexity)
- Public APIs (human-readable, tooling)
- Simple CRUD operations (HTTP semantics)
- Caching (HTTP caching infrastructure)

*When gRPC excels:*
- Internal microservices (low latency, strong typing)
- Streaming workloads (server push, bidirectional)
- Polyglot environments (code generation)
- High-throughput systems (10K+ RPS)

== Load Balancing Considerations

*L4 (TCP) load balancing:* Distributes connections, not requests.

```
Problem with HTTP/2 + L4 LB:
  Client ──▶ L4 LB ──▶ Server A (receives ALL requests)
                  ╳──▶ Server B (idle)
                  ╳──▶ Server C (idle)

  HTTP/2 multiplexes all RPCs over single connection!
```

*L7 (gRPC-aware) load balancing:* Distributes individual RPCs.

```
Client ──▶ L7 LB ──▶ Server A (receives RPC 1, 4, 7...)
                ├──▶ Server B (receives RPC 2, 5, 8...)
                └──▶ Server C (receives RPC 3, 6, 9...)
```

*Client-side load balancing (recommended for internal):*
```cpp
// gRPC channel with round-robin
grpc::ChannelArguments args;
args.SetLoadBalancingPolicyName("round_robin");

auto channel = grpc::CreateCustomChannel(
    "dns:///my-service.internal:50051",
    grpc::InsecureChannelCredentials(),
    args
);
```

*Service mesh integration:* Envoy, Istio, Linkerd provide L7 load balancing + observability.

*Health checking:*
```protobuf
// Standard health check protocol
service Health {
  rpc Check(HealthCheckRequest) returns (HealthCheckResponse);
  rpc Watch(HealthCheckRequest) returns (stream HealthCheckResponse);
}
```

== Connection Management

*Channel:* Virtual connection to endpoint (manages underlying HTTP/2 connections).

```cpp
// Create channel (expensive - reuse!)
auto channel = grpc::CreateChannel(
    "localhost:50051",
    grpc::InsecureChannelCredentials()
);

// Create stub (cheap - can create many)
auto stub = MyService::NewStub(channel);
```

*Best practices:*
- One channel per target service (not per RPC)
- Channels handle reconnection automatically
- Configure keepalive for long-lived connections

```cpp
grpc::ChannelArguments args;
args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 10000);     // Ping every 10s
args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 5000);   // Wait 5s for pong
args.SetInt(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA, 0);  // Allow pings
```

== Error Handling

*Status codes:* gRPC defines standard error codes.

```
Code  Name                Description
──────────────────────────────────────────────────────────────
0     OK                  Success
1     CANCELLED           Operation cancelled
2     UNKNOWN             Unknown error
3     INVALID_ARGUMENT    Client error - bad request
4     DEADLINE_EXCEEDED   Timeout
5     NOT_FOUND           Resource not found
6     ALREADY_EXISTS      Resource already exists
7     PERMISSION_DENIED   Authorization failure
8     RESOURCE_EXHAUSTED  Rate limiting, quota
13    INTERNAL            Server bug
14    UNAVAILABLE         Service temporarily unavailable (retry)
```

*Rich error details (google.rpc.Status):*
```protobuf
import "google/rpc/status.proto";
import "google/rpc/error_details.proto";

// Server can attach structured error info
google.rpc.BadRequest bad_request;
bad_request.add_field_violations()->set_field("email");
bad_request.mutable_field_violations(0)->set_description("Invalid format");
```

== References

*Primary sources:*

gRPC Project (2024). gRPC Documentation. https://grpc.io/docs/

Google (2024). Protocol Buffers Encoding. https://protobuf.dev/programming-guides/encoding/

IETF RFC 7540: Hypertext Transfer Protocol Version 2 (HTTP/2). Belshe, M., Peon, R., & Thomson, M. (2015).

Nally, M. (2020). "gRPC vs REST: Understanding gRPC, OpenAPI and REST." Google Cloud Blog.

*Performance references:*

Indrasiri, K. & Kuruppu, D. (2021). gRPC: Up and Running. O'Reilly Media.

Nygard, M. (2018). Release It! Design and Deploy Production-Ready Software. Pragmatic Bookshelf.

gRPC Performance Best Practices (2024). https://grpc.io/docs/guides/performance/

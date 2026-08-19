#import "../template.typ": rfc, xref

= API Design

An API is a published promise. Unlike internal code, it cannot be refactored at will: every consumer compounds the cost of every design mistake. This chapter covers REST maturity and resource modelling, versioning and pagination, error design under #rfc(9457), the gRPC and GraphQL trade space, idempotency, backwards compatibility, and contract tooling with OpenAPI.

*See also:* #xref("software-architecture", "event-driven-architecture", label: "Event-Driven Architecture") (asynchronous contracts and event schema evolution), #xref("software-architecture", "distributed-data-patterns", label: "Distributed Data Patterns") (API composition over service-owned data), #xref("software-architecture", "evolutionary-architecture", label: "Evolutionary Architecture") (deprecation policy as a governance practice).

== REST and the Richardson Maturity Model

REST, defined in Roy Fielding's 2000 dissertation, is an architectural style whose constraints (client–server, statelessness, cacheability, uniform interface, layered system) explain why the web scales. Leonard Richardson's maturity model (2008) grades how much of it an HTTP API actually uses:

- *Level 0*: one URI, one verb, RPC-over-POST (classic SOAP-ish endpoints like `/api`).
- *Level 1*: resources, distinct URIs per concept (`/orders/123`), but still mostly POST.
- *Level 2*: HTTP verbs and status codes used with their semantics, GET is safe and cacheable, PUT/DELETE are idempotent, 201 vs. 200 vs. 409 carry meaning. The plateau where nearly all production "REST" APIs live.
- *Level 3*: hypermedia (HATEOAS), responses carry links naming the legal next actions, so clients are decoupled from URI structure and workflow.

Level 3 remains rare outside niches (HAL, JSON:API, GitHub's API partially) because mainstream client tooling does not exploit it; pragmatically, target a disciplined level 2 and document it with OpenAPI. Fielding's own complaint, "if the engine of application state is not hypertext, it is not REST", is technically correct and commercially ignored.

== Resource Modelling

Resource design is domain modelling under HTTP constraints:

- Model *nouns*, not verbs: `POST /orders/123/cancellation` (or a state transition via PATCH) beats `POST /cancelOrder`. When an operation refuses to be a noun, reify the process: a `/transfers` resource with a status field models a long-running operation and gives you polling, history, and idempotency for free.
- Choose the consumer's granularity: one round-trip per screen beats elegant normalisation. Chatty fine-grained resources push integration logic onto every client (the under-fetching problem GraphQL was built to fix).
- Sub-resources express ownership (`/customers/42/addresses`); avoid nesting beyond two levels, deep paths encode relationships that may change.
- Use stable, opaque identifiers. Exposing database sequence IDs leaks information (the *German tank problem*: sequential order IDs let competitors estimate volume) and invites enumeration; random IDs (UUIDv7 for index locality) avoid both.

== Versioning

Every strategy is a different distribution of pain between provider and consumer:

#table(
  columns: 3,
  [*Strategy*], [*Example*], [*Trade-off*],
  [URI version], [`/v2/orders`], [Visible, cache-friendly; "different resource" purists object; most common (Stripe-adjacent, Twitter, Google)],
  [Header / media type], [`Accept: application/vnd.api.v2+json`], [Clean URIs; harder to test in a browser, easy to omit],
  [Query parameter], [`?api-version=2024-06-01`], [Azure's approach; explicit, slightly noisy],
  [Date-based pinning], [`Stripe-Version: 2024-06-20`], [Account pinned to a version; provider maintains transforms between adjacent versions],
)

Stripe's model (described by Brandur Leach, 2017) is the gold standard for long-lived public APIs: each breaking change is a dated version with a pair of request/response transformation functions; the core codebase targets only the latest version, and middleware chains transforms to serve any historical version. Some accounts run versions many years old. The deeper lesson: versioning is a *compatibility machinery* problem, not a URI-naming debate, and the cheapest version is the one you never have to mint, design additively (see below) so most changes are non-breaking.

== Pagination

- *Offset/limit* (`?offset=100&limit=20`): simple, supports jump-to-page; degrades on deep pages (the database still scans and discards `offset` rows) and *drifts*, inserts and deletes between requests shift items across page boundaries, so items are skipped or duplicated.
- *Cursor/keyset* (`?after=eyJpZCI6MTAyM30&limit=20`): the cursor encodes the last-seen sort key (`WHERE (created, id) > (:c, :i) ORDER BY created, id LIMIT 20`); stable under concurrent writes and $O(log n)$ regardless of depth. The standard for feeds and large collections (Stripe `starting_after`, Slack, GitHub GraphQL connections). Costs: no random access, cursor must be opaque (base64 the key, do not promise its structure), and the sort key must be unique and immutable, hence the `(created, id)` tiebreak.
- Return pagination metadata uniformly (`next` cursor or #rfc(8288) `Link` headers) and document an explicit maximum page size; an unbounded `limit` parameter is a self-service denial-of-service endpoint.

== Error Design and #rfc(9457)

Status codes are a coarse taxonomy, machine triage, not diagnosis: 400 vs. 401 vs. 403 vs. 404 vs. 409 vs. 422 vs. 429 each trigger different client behaviour (fix the request, re-authenticate, give up, retry later). The body carries the rest. *#rfc(9457), Problem Details for HTTP APIs* (2023, obsoleting #rfc(7807)) standardises it:

```json
{
  "type": "https://api.example.com/problems/insufficient-funds",
  "title": "Insufficient funds",
  "status": 422,
  "detail": "Balance is 30.00 EUR; transfer requires 45.00 EUR.",
  "instance": "/transfers/abc-123",
  "balance": 30.00
}
```

`type` is a stable URI identifying the error *kind*, the field clients should switch on, never the human-readable `detail` string, which you must remain free to reword. Extension members (like `balance` above) carry structured context. Media type: `application/problem+json`. Practices that matter: include a correlation/request ID for support; for validation, return *all* field errors in one response (an `errors` array), not one per round-trip; never leak stack traces or internal class names (an information-disclosure finding in any security review).

== gRPC and GraphQL Trade-offs

=== gRPC

gRPC (Google, open-sourced 2015, from the internal Stubby system) is RPC over HTTP/2 with Protocol Buffers: binary encoding, contract-first `.proto` files, generated clients in 10+ languages, deadlines propagated across call chains, and four streaming modes (unary, server-, client-, bidirectional-streaming). Typical wins over JSON/REST: 3–10$times$ smaller payloads and substantially lower serialisation CPU; deadline propagation and built-in load-balancing hooks suit deep service-to-service call graphs. Costs: not browser-native (grpc-web needs a proxy), payloads are not human-readable, and field-number discipline is required for evolution. Default choice for internal east–west traffic in polyglot service fleets.

=== GraphQL

GraphQL (Facebook 2012, public 2015) lets the client specify the exact shape of data it needs against a typed schema, solving mobile over-/under-fetching: one round trip replaces a waterfall of REST calls. Costs are equally structural: every query is a potential ad-hoc join, so servers need query-depth and complexity limits, dataloader batching to kill N+1 resolver storms, and persisted queries to cap the attack surface; HTTP caching is largely lost (everything is a POST to `/graphql`); and the flexibility moves performance unpredictability to the server team. Best fit: a *backend-for-frontend* aggregation tier over many sources with diverse UI clients (GitHub's public API v4, Shopify's Storefront API); poor fit: simple resource CRUD or service-to-service calls.

Rule of thumb: REST for public resource-oriented APIs, gRPC for internal RPC, GraphQL where independent UI teams consume many backends, and any combination thereof in one system.

== Idempotency

An operation is idempotent if performing it $n >= 1$ times has the effect of performing it once. HTTP declares GET, PUT, DELETE idempotent and POST not. This matters because *retries are mandatory* in distributed systems (see _Resilience Patterns_), and a retried non-idempotent request is a duplicate payment.

The standard fix is the *idempotency key* (Stripe's design, now an IETF draft `Idempotency-Key` header): the client generates a unique key per logical operation (a UUID); the server atomically records the key with the response of the first execution and replays that stored response for any retry with the same key. Implementation subtleties: the key store needs a TTL (Stripe: 24 hours); concurrent duplicates must either block or get 409; the key must be checked *and reserved* in the same transaction as the side effect, or a crash between them recreates the duplicate (the same atomicity problem the outbox pattern solves).

== Backwards Compatibility

The contract rule: *be conservative in what you send, be liberal in what you accept* needs a precise version for APIs. Non-breaking (additive) changes: adding an optional request field, adding a response field, adding an enum value *you emit only after consumers tolerate unknowns*, adding a new endpoint. Breaking: removing or renaming anything, changing types or semantics, tightening validation, making an optional field required, changing default sort order, and, the one everyone forgets, changing error `type`s that clients switch on.

Defensive practices:
- Consumers must ignore unknown fields (Postel's law applied client-side); providers should test this with consumer-driven contract tests (Pact) that verify each consumer's actual expectations rather than the whole schema.
- Protobuf encodes the discipline structurally: fields are identified by number, never reuse or renumber; `reserved` retired numbers; everything is optional in proto3.
- Treat the API description as the reviewed artefact: a CI diff of the OpenAPI document (tools: `oasdiff`, Optic) flags breaking changes before they ship.

== OpenAPI and Contract-First Development

OpenAPI (formerly Swagger; 3.0 in 2017, 3.1 in 2021 aligning with JSON Schema 2020-12) is the lingua franca for describing HTTP APIs: paths, operations, parameters, schemas, security schemes, examples. Two workflows:

- *Design-first*: write the OpenAPI document, review it like code (style guides linted with Spectral), then generate server stubs, client SDKs, and mock servers (Prism) so consumer teams build in parallel.
- *Code-first*: generate the document from annotations (FastAPI, springdoc). Lower friction, but the contract becomes an implementation by-product and drift is discovered by consumers.

Either way, the document earns its keep in CI: request/response validation against the schema at test time, breaking-change diffing on every PR, and generated reference documentation that cannot go stale. Large API programmes (Microsoft, Google AIP, Zalando's REST guidelines) layer organisation-wide style rules on top, naming, pagination shape, error format, so hundreds of APIs feel like one.

== Further Reading

- Fielding, R. (2000). _Architectural Styles and the Design of Network-based Software Architectures_. PhD dissertation, UC Irvine.
- Nottingham, M., Wilde, E., & Dalal, S. (2023). #rfc(9457): Problem Details for HTTP APIs. IETF.
- Leach, B. (2017). APIs as infrastructure: future-proofing Stripe with versioning. Stripe Engineering Blog.
- Fowler, M. (2010). Richardson Maturity Model. martinfowler.com.
- Geewax, J. J. (2021). _API Design Patterns_. Manning.
- Zalando RESTful API and Event Guidelines. opensource.zalando.com/restful-api-guidelines.

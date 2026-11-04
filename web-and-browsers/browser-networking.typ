#import "../template.typ": xref

= Networking and Resource Loading <browser-networking>

A page load is a negotiation between the browser, the network and the server over what to fetch, in what order, over which connections, and what can be reused from earlier visits. For most sites, the network dominates loading time far more than parsing, layout or script. This chapter follows a resource from discovery to use: the navigation and connection setup, the HTTP cache and its freshness rules, connection management across HTTP/1.1, HTTP/2 and HTTP/3, how browsers prioritise requests, the preload scanner and resource hints, speculative loading and the back/forward cache, and compression.

*See also:* #xref("networking", "quic-and-http3", label: "QUIC and HTTP/3"), #xref("networking", "application-protocols", label: "Application Protocols"), #xref("web-and-browsers", "html-parsing-and-dom", label: "HTML Parsing and the DOM") (speculative parsing), #xref("web-and-browsers", "web-performance", label: "Web Performance")

== Anatomy of a Navigation

When a user follows a link, the browser process (not the renderer) starts the *navigation*:

1. Check whether the target is already *prerendered* or in the *back/forward cache*. If so, the page can be shown almost instantly.
2. Resolve the host name (from the DNS cache, or through the system resolver or DNS over HTTPS). HTTPS DNS records (type 65) can also advertise HTTP/3 support and encrypted client hello keys.
3. Reuse an existing connection, or open a new one: TCP plus TLS 1.3 costs two round trips before the request can be sent (one with TLS 0-RTT resumption), and QUIC costs one (zero with resumption).
4. Send the request, follow any redirects (each one usually costs a further round trip and often a new connection), and stream the response to a renderer chosen by site isolation rules.
5. The renderer starts parsing as soon as the first bytes arrive, discovering subresources.

On a mobile network with a 100 ms round-trip time, the connection setup and one redirect alone can cost 400 ms before the first byte of HTML. That is why the fastest loading techniques avoid the network entirely (caching, prerendering) or overlap it with something else (preconnect, early hints).

== The HTTP Cache

The browser cache (RFC 9111) decides whether a stored response can be reused without asking the server (*fresh*), must be revalidated (*stale*), or must not be stored at all.

=== Freshness

A response's *freshness lifetime* comes from, in order of precedence: `Cache-Control: s-maxage` (shared caches only), `Cache-Control: max-age`, the difference between `Expires` and `Date`, or, if none is present, a *heuristic* (commonly 10 percent of the time since `Last-Modified`). The response is fresh while its *current age* is below its lifetime. Age accounts for time spent in upstream caches and in transit:

```cpp
#include <algorithm>
#include <chrono>
#include <optional>

using seconds = std::chrono::seconds;
using time_point = std::chrono::sys_seconds;

struct stored_response {
    time_point request_time;            // when we sent the request
    time_point response_time;           // when the response arrived
    time_point date_header;             // Date
    seconds age_header{0};              // Age, if any upstream cache added one
    std::optional<seconds> max_age;     // Cache-Control: max-age
    std::optional<time_point> expires;  // Expires
    std::optional<time_point> last_modified;
};

// RFC 9111, section 4.2.3.
seconds current_age(const stored_response& r, time_point now) {
    const seconds apparent_age = std::max(seconds{0}, r.response_time - r.date_header);
    const seconds response_delay = r.response_time - r.request_time;
    const seconds corrected_age_value = r.age_header + response_delay;
    const seconds corrected_initial_age = std::max(apparent_age, corrected_age_value);
    const seconds resident_time = now - r.response_time;
    return corrected_initial_age + resident_time;
}

// RFC 9111, sections 4.2.1 and 4.2.2 (private cache: s-maxage ignored).
seconds freshness_lifetime(const stored_response& r) {
    if (r.max_age) return *r.max_age;
    if (r.expires) return std::max(seconds{0}, *r.expires - r.date_header);
    if (r.last_modified) return (r.date_header - *r.last_modified) / 10;   // heuristic
    return seconds{0};
}

bool is_fresh(const stored_response& r, time_point now) {
    return current_age(r, now) < freshness_lifetime(r);
}
```

=== Validation

A stale response is not discarded: the browser sends a *conditional request* with `If-None-Match` (the stored `ETag`) or `If-Modified-Since` (the stored `Last-Modified`). If nothing changed, the server answers `304 Not Modified` with no body, saving the transfer but not the round trip.

=== Directives That Matter

- `no-store`: never write the response to any cache. Use it for sensitive personal data.
- `no-cache`: store it, but revalidate before every use. Despite the name, this *does* cache.
- `immutable`: the response will never change during its freshness lifetime, so the browser should not revalidate it even when the user reloads.
- `stale-while-revalidate=N`: serve a stale response for up to $N$ seconds while revalidating in the background.
- `private` and `public`: whether shared caches (CDNs, proxies) may store the response.

The standard pattern for static assets is *fingerprinted URLs*: build tools put a content hash in the file name (`app.3f9a1c.js`) and serve it with `Cache-Control: max-age=31536000, immutable`. The HTML that references the assets is served with `no-cache`, so a new deployment changes the HTML, which points to new asset URLs, and old assets are never served stale.

Since 2020, browsers *partition* the HTTP cache by top-level site (and, in Chromium, by frame site), to stop sites from probing whether a user has visited another site through cache timing. A side effect is that a library loaded from a public CDN is no longer shared between different sites, which removed most of the historical argument for using public CDNs for common libraries.

== Connections

- *HTTP/1.1* sends one request at a time per connection, so browsers open up to six connections per host. Pages split assets across domains ("domain sharding") to get more parallelism, a practice that HTTP/2 made counterproductive.
- *HTTP/2* multiplexes many concurrent streams over one TCP connection with header compression (HPACK). It removed application-level head-of-line blocking, but a lost TCP packet still stalls all streams until it is retransmitted.
- *HTTP/3* runs over QUIC (#xref("networking", "quic-and-http3", label: "QUIC and HTTP/3")), which provides independent streams over UDP, so loss on one stream does not block the others, combines the transport and TLS handshakes, and allows connection migration when a phone switches networks. Browsers discover HTTP/3 through the `Alt-Svc` response header or HTTPS DNS records, and race it against TCP in case UDP is blocked.

*Connection coalescing* lets HTTP/2 and HTTP/3 reuse one connection for several host names when they resolve to the same address and the server's certificate covers them all, so `cdn.example.com` and `www.example.com` may share a connection. Cross-origin requests made *without credentials* (such as fonts, which are always fetched in CORS anonymous mode) use a separate connection pool from credentialed ones, which can cause an unexpected second connection to the same host.

== Prioritisation

A page may request a hundred resources at once, but they are not equally urgent. Render-blocking stylesheets and the largest image should not wait behind analytics scripts. Browsers assign each request a priority from its type and context:

#table(
  columns: 2,
  [*Resource*], [*Typical Chromium priority*],
  [Main document, CSS in the head, fonts], [Highest],
  [Synchronous and early scripts, images in the initial viewport (after layout)], [High],
  [Async and deferred scripts, images outside the viewport], [Low],
  [Prefetches for future navigations], [Lowest],
)

Over HTTP/2 and HTTP/3, priorities are sent to the server with the *Extensible Prioritization Scheme* (RFC 9218): an urgency from 0 to 7 and an `incremental` flag that says whether the resource is useful when partially received (progressive images) or only when complete (scripts). Servers and CDNs differ widely in how well they honour priorities, and a server that ignores them interleaves everything, delaying critical resources.

Authors can adjust priorities with `fetchpriority="high"` or `"low"` on images, scripts and links, most usefully to raise the priority of the hero image that determines Largest Contentful Paint. Script attributes also matter: classic `<script>` in the head blocks parsing; `defer` downloads in parallel and runs after parsing in document order; `async` runs as soon as it arrives, in any order; `type="module"` behaves like `defer` by default.

== Discovering Resources Early

=== The Preload Scanner

While the main HTML parser is blocked on a synchronous script, a lightweight *preload scanner* (#xref("web-and-browsers", "html-parsing-and-dom", label: "HTML Parsing and the DOM")) reads ahead in the raw HTML and starts fetching the scripts, stylesheets and images it finds. It is one of the most effective optimisations in browsers, and it only works for resources visible in the HTML markup. Resources discovered only by running script (a framework that injects image tags), by applying CSS (background images, web fonts), or by importing modules one level at a time are discovered late. Much of loading performance work consists of making critical resources visible to the preload scanner.

=== Resource Hints

```html
<!-- Open a connection early to an origin we will need. -->
<link rel="preconnect" href="https://images.example-cdn.com">

<!-- Fetch a late-discovered critical resource now, at the right priority. -->
<link rel="preload" href="/fonts/inter-var.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="/hero.avif" as="image" fetchpriority="high">

<!-- Fetch and parse a module and its imports ahead of use. -->
<link rel="modulepreload" href="/app/main.js">

<!-- Low-priority fetch of something the next navigation probably needs. -->
<link rel="prefetch" href="/checkout/bundle.js">
```

Preload is powerful and easy to overuse: every preloaded resource competes for bandwidth with the ones the page discovers naturally, and preloading something not used within a few seconds triggers a console warning. The `crossorigin` attribute must match how the resource is later requested, or the preload is wasted and the resource is fetched twice.

=== Early Hints

A server often knows which critical resources a page needs before it has finished generating the HTML (for example while querying a database). The `103 Early Hints` informational response (RFC 8297) lets it send `Link: rel=preload` and `rel=preconnect` headers immediately, so the browser starts fetching during the server's "think time". CDNs can send early hints from cached hints while the origin is still working.

== Speculative Loading and the Back/Forward Cache

The fastest navigation is one whose work already happened.

- *Speculation rules* (a JSON script block) tell the browser which links to *prefetch* (fetch the document) or *prerender* (fully load and render it in a hidden tab) and when: immediately, or "moderately" on hover or pointer-down. Activating a prerendered page is nearly instantaneous. Prerendering runs the page's script, so analytics and other side effects must check `document.prerendering` and wait for activation.
- The *back/forward cache* (bfcache) keeps a whole page, including its JavaScript heap, in memory when the user navigates away, and restores it on Back or Forward with no loading at all. Back and forward are a large share of navigations, so eligibility matters. Common blockers: `unload` event handlers (use `pagehide` instead), open connections such as WebSockets that are not closed on `pagehide`, and in some browsers `Cache-Control: no-store` on the document. Chrome's DevTools list the reasons a page was not eligible.

== Compression

Text resources are compressed in transit:

- *gzip* (DEFLATE) is universal.
- *Brotli* compresses 15 to 25 percent better on typical web text, partly thanks to a built-in dictionary of common web strings. Static assets are best compressed ahead of time at the highest level; dynamic responses use a lower level to save CPU.
- *Zstandard* (zstd) is supported by recent Chromium and Firefox versions and offers Brotli-like ratios with faster compression, attractive for dynamic content.
- *Compression dictionary transport* lets a site use a previous version of a resource (or a custom dictionary) as the compression dictionary for the next version. For a JavaScript bundle that changed slightly between deployments, the transfer shrinks to little more than the difference.

Images and fonts are already compressed, so modern formats matter more than transport compression for them: AVIF and WebP images are typically much smaller than JPEG and PNG at the same quality, and WOFF2 is the standard font format.

== Further Reading

Fielding, R., Nottingham, M., Reschke, J. (2022). "HTTP Semantics." RFC 9110.

Fielding, R., Nottingham, M., Reschke, J. (2022). "HTTP Caching." RFC 9111.

Bishop, M. (2022). "HTTP/3." RFC 9114.

Oku, K., Pardue, L. (2022). "Extensible Prioritization Scheme for HTTP." RFC 9218.

Oku, K. (2017). "An HTTP Status Code for Indicating Hints." RFC 8297.

Grigorik, I. (2013). "High Performance Browser Networking." O'Reilly Media.

WHATWG (2024). "HTML Living Standard." Living standard (link types, speculation rules integration, bfcache).

WHATWG (2024). "Fetch Standard." Living standard (request priorities, CORS modes, cache partitioning keys).

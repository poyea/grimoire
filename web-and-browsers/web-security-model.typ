#import "../template.typ": xref

= The Web Security Model <web-security-model>

A browser runs code from millions of mutually distrusting parties, often in the same process and sometimes in the same page, while holding the user's credentials for all of them. The security model that makes this survivable grew by accretion: an origin-based isolation rule from 1995, cookies designed before it, and three decades of patches for the attacks that the combination allowed. This chapter covers the origin and the same-origin policy, cross-origin resource sharing, cookies and their attributes, the classic attacks (cross-site scripting, cross-site request forgery, clickjacking) and their defences, transport security, and the post-Spectre model of cross-origin isolation and site isolation.

*See also:* #xref("web-and-browsers", "browser-architecture", label: "Browser Architecture") (site isolation and sandboxing), #xref("cryptography-and-security", "tls", label: "TLS"), #xref("cryptography-and-security", "side-channel-attacks", label: "Side-Channel Attacks") (Spectre), #xref("networking", "network-security", label: "Network Security")

== Origins and the Same-Origin Policy

An *origin* is the triple (scheme, host, port): `https://example.com:443` and `https://example.com:8443` are different origins, as are `http://` and `https://` on the same host, and `a.example.com` and `b.example.com`. The *same-origin policy* (SOP) says that script from one origin may not *read* data belonging to another. It does not stop one origin from *sending* requests to another or *embedding* its resources:

#table(
  columns: 2,
  [*Allowed across origins*], [*Blocked across origins*],
  [Navigating to, linking to, submitting forms to another origin], [Reading the response of a `fetch` or `XMLHttpRequest`, unless CORS allows it],
  [Embedding images, scripts, stylesheets, media, iframes], [Reading the pixels of a cross-origin image drawn on a canvas (the canvas becomes "tainted")],
  [Sending `postMessage` to another window], [Accessing the DOM of a cross-origin iframe or window],
)

This asymmetry (writes and embeds allowed, reads blocked) is the root of several attack classes. It exists because the web was built on linking and embedding before anyone designed a security model.

A related but different concept is the *site*: the scheme plus the *registrable domain* (one label below a public suffix such as `.com` or `.co.uk`, from the Public Suffix List). `a.example.com` and `b.example.com` are different origins but the same site. Cookies, `SameSite`, site isolation and storage partitioning are defined in terms of sites, not origins, which is a frequent source of confusion.

== Cross-Origin Resource Sharing

CORS (part of the Fetch standard) lets a server opt in to cross-origin reads. The browser adds an `Origin` header to cross-origin requests and exposes the response to script only if the server answers with a matching `Access-Control-Allow-Origin`.

Requests that a plain HTML form could already send (GET, HEAD, POST with form-like content types and no custom headers) are sent directly; the CORS check only gates reading the response. Anything else (a `PUT`, a JSON body, an `Authorization` header) first triggers a *preflight*: an `OPTIONS` request asking whether the real request is allowed.

```
OPTIONS /api/orders HTTP/1.1
Origin: https://shop.example
Access-Control-Request-Method: PUT
Access-Control-Request-Headers: content-type

HTTP/1.1 204 No Content
Access-Control-Allow-Origin: https://shop.example
Access-Control-Allow-Methods: GET, PUT
Access-Control-Allow-Headers: content-type
Access-Control-Allow-Credentials: true
Access-Control-Max-Age: 600
Vary: Origin
```

Points that cause real vulnerabilities:

- *CORS is not access control.* It protects the user's browser from leaking data to other origins; it does nothing against `curl`. Authorisation must be enforced on the server.
- *Credentials*: cookies are sent on cross-origin `fetch` only with `credentials: "include"`, and the response is readable only if the server returns `Access-Control-Allow-Credentials: true` together with a specific origin (never `*`).
- *Reflecting the `Origin` header* back into `Access-Control-Allow-Origin` for every request, combined with credentials, lets any site read authenticated responses. Allow-list origins explicitly, and send `Vary: Origin` so that caches do not serve one origin's response to another.
- The *preflight* exists to protect servers that assumed browsers could never send such requests cross-origin. It is not a security boundary for new APIs, which should check authentication themselves.

== Cookies

Cookies predate the same-origin policy and follow different rules: they are scoped by domain and path, not origin, and historically were sent on every request to the matching domain, regardless of which site initiated it. That *ambient authority* is what CSRF exploits. Modern attributes constrain them:

#table(
  columns: 2,
  [*Attribute*], [*Effect*],
  [`Secure`], [Sent only over HTTPS],
  [`HttpOnly`], [Invisible to JavaScript (`document.cookie`), limiting theft by XSS],
  [`SameSite=Strict`], [Never sent on requests initiated by another site, even top-level navigations],
  [`SameSite=Lax`], [Sent on top-level GET navigations from other sites, not on cross-site subresource requests or POSTs; the default in Chromium-based browsers when no attribute is set],
  [`SameSite=None`], [Sent on all requests; requires `Secure`],
  [`__Host-` prefix], [Must be `Secure`, have `Path=/` and no `Domain`, so it cannot be set or overwritten by a sibling subdomain],
  [`Partitioned`], [Stored separately per top-level site (CHIPS), for embedded third-party use without cross-site tracking],
)

A good default for session cookies is `__Host-session=...; Secure; HttpOnly; SameSite=Lax; Path=/`. Third-party cookies, the basis of cross-site tracking, are blocked by default in Safari and Firefox, and storage of all kinds is increasingly *partitioned* by top-level site, so that an iframe from `widget.example` embedded in two different sites sees two separate storage jars.

== Cross-Site Scripting

*Cross-site scripting* (XSS) is the injection of attacker-controlled script into a page of the target origin, where it runs with that origin's full authority: it can read the DOM, call APIs with the user's cookies, and exfiltrate data. Three forms:

- *Reflected*: the payload comes from the request (a query parameter echoed into the page).
- *Stored*: the payload is saved on the server (a comment, a profile field) and served to other users.
- *DOM-based*: client-side script copies untrusted data into a dangerous *sink* such as `innerHTML`, `document.write`, `eval` or a `javascript:` URL.

The primary defence is *context-aware output encoding*: escaping data differently for HTML text, attribute values, URLs and script contexts, which modern template systems and frameworks (React, Angular, Vue) do by default. XSS today mostly arises where developers bypass them (`dangerouslySetInnerHTML`, `v-html`, string-built HTML) or in DOM sinks.

Two browser mechanisms add defence in depth:

- *Content Security Policy* (CSP) restricts what script may run. A *strict* CSP based on nonces or hashes is effective; allow-list policies naming domains are routinely bypassed through script gadgets on allowed CDNs (Weichselbaum et al., 2016).
- *Trusted Types* make DOM sinks reject plain strings, accepting only typed values produced by reviewed policy functions, which turns DOM XSS from a code-review problem into a type error.

```
Content-Security-Policy:
  script-src 'nonce-r4nd0m' 'strict-dynamic';
  object-src 'none';
  base-uri 'none';
  require-trusted-types-for 'script';
  frame-ancestors 'self'
```

```js
// With require-trusted-types-for 'script', this throws:
//   element.innerHTML = userComment;
// Sinks accept only values created by a named, reviewed policy:
const sanitizer = trustedTypes.createPolicy("comment-html", {
  createHTML: (input) => DOMPurify.sanitize(input, { RETURN_TRUSTED_TYPE: false }),
});
element.innerHTML = sanitizer.createHTML(userComment);
```

== Cross-Site Request Forgery

*CSRF* tricks the user's browser into sending a state-changing request to a site where the user is logged in: a hidden form on `evil.example` that POSTs to `bank.example/transfer`, carrying the bank's cookies automatically. The SOP does not prevent this because sending is allowed; only reading the response is blocked, and the attacker does not need the response.

Defences, in order of preference:

- *`SameSite` cookies* (`Lax` or `Strict`) stop the session cookie from being attached to cross-site POSTs, defeating classic CSRF. Remember that "site" is coarser than origin: a compromised or user-controlled subdomain is same-site.
- *Fetch metadata* headers (`Sec-Fetch-Site`, `Sec-Fetch-Mode`, `Sec-Fetch-Dest`), sent by all modern browsers, let the server reject cross-site requests to endpoints that should never receive them, as a simple *resource isolation policy*.
- *Anti-CSRF tokens*: an unpredictable per-session value included in forms and checked on the server, which another site cannot read.
- Requiring a custom header (such as `X-Requested-With`) forces a preflight for cross-origin requests and thus blocks simple form-based CSRF, though it relies on CORS being configured correctly.

== Clickjacking and Framing

A malicious page can load a target site in a transparent iframe and trick the user into clicking its buttons ("Delete account") while they think they are clicking something else. Sites prevent being framed with the CSP `frame-ancestors` directive (or the older `X-Frame-Options: DENY`). Pages that legitimately embed third-party content can restrict it with the iframe `sandbox` attribute, which by default denies scripts, forms, popups and same-origin access, enabling each capability explicitly.

== Transport Security

An active network attacker can modify any HTTP response, so everything above assumes HTTPS.

- *HSTS* (`Strict-Transport-Security`) tells the browser to use HTTPS for the domain for a period, preventing downgrade attacks on later visits. The *preload list* built into browsers protects even the first visit.
- *Mixed content*: an HTTPS page loading scripts over HTTP would hand control to a network attacker, so browsers block such loads, and upgrade or block mixed images and media.
- *Subresource Integrity* (`integrity="sha384-..."` on script and link tags) makes the browser verify the hash of a resource from a CDN, so a compromised CDN cannot inject code.

== Spectre and Cross-Origin Isolation

The 2018 Spectre attacks showed that code running in a process can read *any* memory in that process through speculative execution side channels (#xref("cryptography-and-security", "side-channel-attacks", label: "Side-Channel Attacks")). Since a page's process contained cross-origin data (images, scripts, iframes from other sites), the same-origin policy as enforced by the renderer could no longer be trusted alone. The response came in layers:

1. *Site isolation* (#xref("web-and-browsers", "browser-architecture", label: "Browser Architecture")): each site gets its own renderer process, so a compromised or Spectre-exploiting renderer holds only its own site's data.
2. *Cross-Origin Read Blocking* (CORB, later the opaque-response blocking algorithm ORB): the network service refuses to deliver cross-origin responses that look like HTML, JSON or XML into a renderer that requested them as an image or script, since those can never legitimately be used that way.
3. *Reduced timer precision* and the temporary removal of `SharedArrayBuffer`, which allows building high-resolution timers.
4. *Cross-origin isolation*: a page can re-enable `SharedArrayBuffer` and precise timers only if it guarantees that no cross-origin data enters its process without consent. It does so with two headers:
   - `Cross-Origin-Opener-Policy: same-origin` (COOP) severs the window relationship with cross-origin popups and openers, so they can live in other processes.
   - `Cross-Origin-Embedder-Policy: require-corp` (COEP) loads cross-origin subresources only if they opt in with `Cross-Origin-Resource-Policy` or CORS. The `credentialless` variant loads them without cookies instead.

When both are set, `self.crossOriginIsolated` is true, and features such as `SharedArrayBuffer` (needed for multithreaded WebAssembly) become available. Resources that should never be embedded cross-origin can protect themselves with `Cross-Origin-Resource-Policy: same-origin`.

== Permissions and Powerful Features

Features that expose sensitive capabilities (camera, microphone, geolocation, clipboard, USB, Bluetooth) require a *secure context* (HTTPS), usually a user gesture, and an explicit user permission grant per origin. *Permissions Policy* headers and the iframe `allow` attribute let a page control which of these its embedded frames may even request, preventing an ad iframe from asking for the camera. Browsers increasingly expire unused permissions and reject prompts that are not tied to user intent.

== A Header Baseline

#table(
  columns: 2,
  [*Header*], [*Purpose*],
  [`Strict-Transport-Security: max-age=63072000; includeSubDomains; preload`], [Enforce HTTPS],
  [`Content-Security-Policy` (strict, nonce-based, with `frame-ancestors`)], [Limit script execution; prevent framing],
  [`Cross-Origin-Opener-Policy: same-origin`], [Process isolation from popups],
  [`Cross-Origin-Resource-Policy: same-origin` on private resources], [Prevent cross-origin embedding],
  [`Referrer-Policy: strict-origin-when-cross-origin`], [Do not leak URLs to other sites],
  [`X-Content-Type-Options: nosniff`], [Stop the browser guessing content types],
  [Session cookies with `__Host-`, `Secure`, `HttpOnly`, `SameSite=Lax`], [Limit cookie theft and CSRF],
)

== Further Reading

Barth, A. (2011). "The Web Origin Concept." RFC 6454.

WHATWG (2024). "Fetch Standard." Living standard (CORS, fetch metadata, CORP).

Bingler, S., West, M., Wilander, J. (2024). "Cookies: HTTP State Management Mechanism." IETF draft-ietf-httpbis-rfc6265bis.

Weichselbaum, L., Spagnuolo, M., Lekies, S., Janc, A. (2016). "CSP Is Dead, Long Live CSP! On the Insecurity of Whitelists and the Future of Content Security Policy." ACM CCS.

Kocher, P. et al. (2019). "Spectre Attacks: Exploiting Speculative Execution." IEEE Symposium on Security and Privacy.

Reis, C., Moshchuk, A., Oskov, N. (2019). "Site Isolation: Process Separation for Web Sites within the Browser." USENIX Security.

W3C (2024). "Content Security Policy Level 3." W3C Working Draft specification.

Zalewski, M. (2011). "The Tangled Web: A Guide to Securing Modern Web Applications." No Starch Press.

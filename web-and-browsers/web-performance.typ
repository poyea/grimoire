= Web Performance

Web performance is the discipline of making pages load fast, respond instantly, and stay visually stable — and of *proving* it with measurement, because intuition about latency is reliably wrong. This chapter covers the modern metrics (Core Web Vitals), the measurement stack (field vs. lab), the loading waterfall and how to shorten it, runtime responsiveness, and the optimization techniques with the best evidence behind them. It is the applied capstone of everything earlier in this volume: every metric maps to a pipeline stage from the preceding chapters.

*See also:* _Rendering Pipeline_ (what LCP and jank measure), _Event Loop and Scheduling_ (the machinery behind INP), _CSS and Layout_ (CLS and layout thrashing), _Browser Architecture_ (the navigation steps being timed), and the Networking volume (TCP/QUIC, HTTP/2 and /3, CDNs).

== Core Web Vitals

Google's Web Vitals initiative (2020) distilled performance into three field metrics, made consequential by their inclusion in search ranking (the "page experience" signal, 2021). Thresholds are assessed at the *75th percentile* of real-user page loads, segmented by mobile and desktop:

#table(
  columns: 4,
  [*Metric*], [*Measures*], [*Good*], [*Poor*],
  [LCP (Largest Contentful Paint)], [Loading: render time of the largest image or text block in the viewport], [≤ 2.5 s], [> 4.0 s],
  [INP (Interaction to Next Paint)], [Responsiveness: worst-case latency from user input to the next painted frame], [≤ 200 ms], [> 500 ms],
  [CLS (Cumulative Layout Shift)], [Visual stability: sum of layout-shift scores in the worst 5-second window], [≤ 0.1], [> 0.25],
)

*INP* replaced *FID* (First Input Delay) in March 2024; FID measured only input *delay* of the *first* interaction and was nearly always "good", while INP covers delay + processing + presentation for the worst interaction (98th percentile for high-interaction pages), which moved the industry's attention from load-time script to long tasks throughout the session. *CLS* scores each unexpected shift as impact fraction × distance fraction; shifts within 500 ms of user input are excused.

Supporting diagnostics: *TTFB* (time to first byte: redirects + DNS + connection + server time; good ≤ 0.8 s), *FCP* (first contentful paint), *TBT* (Total Blocking Time — the lab proxy for INP: sum of long-task time beyond 50 ms between FCP and interactive), and the Long Animation Frames API (LoAF, Chrome 123, 2024) for attributing INP to specific scripts.

== Field vs. Lab

The two measurement regimes answer different questions and disagree constantly:

- *Field (RUM)*: real users, real devices, real networks. Sources: the `web-vitals` JS library feeding your analytics, and *CrUX* (Chrome User Experience Report) — Chrome's opt-in telemetry, public via BigQuery and the CrUX API, and the dataset search ranking actually uses.
- *Lab*: reproducible synthetic runs. Lighthouse (throttled, default "moderated 4× CPU slowdown + slow 4G"), WebPageTest (real devices, filmstrips, waterfall analysis), and DevTools tracing.

Classic divergences: lab catches regressions pre-ship but cannot see INP across a session; field LCP often beats lab because returning users hit caches; a fast median hides a terrible p75 on low-end Android. The discipline is: set budgets in lab metrics enforced in CI, but optimize against field p75.

The browser exposes the raw substrate through *Performance APIs*: `PerformanceObserver` with entry types `navigation`, `resource` (per-request timing waterfall), `largest-contentful-paint`, `layout-shift`, `event`, `longtask`, `long-animation-frame`, plus `performance.mark`/`measure` and *server-timing* headers for backend attribution.

== The Loading Waterfall

LCP decomposes into four roughly sequential parts — TTFB, resource load delay, resource load time, render delay — and the optimization playbook attacks each:

*Get bytes sooner.* Cut redirects (each costs a round trip). CDN the HTML, not just assets. HTTP/2+ multiplexing removed head-of-line blocking at the HTTP layer; HTTP/3/QUIC removes it at the transport layer and cuts connection setup to 1 RTT (0 for resumption). Early Hints (HTTP 103, 2022) lets the server say "preload these" while still rendering the page. Compress with Brotli (or Zstandard, Chrome 123); compression saves 60–80% on text resources.

*Discover resources earlier.* The preload scanner finds resources in raw HTML — so resources referenced only from CSS (background images, fonts) or constructed by JavaScript are discovered late. Fixes: `<link rel="preload">` for late-discovered critical resources, `fetchpriority="high"` on the LCP image (Priority Hints, 2022), `rel="preconnect"` for critical third-party origins. Anti-pattern: lazy-loading the LCP image (`loading="lazy"` adds a deliberate delay — a top CrUX-observed mistake).

*Ship less, defer the rest.* `defer`/`module` scripts don't block parsing; `async` doesn't block but executes on arrival. Code-splitting and tree-shaking in bundlers; responsive images (`srcset`, `sizes`) and modern formats — AVIF and WebP are typically 30–50% smaller than JPEG; `font-display: swap` plus `size-adjust` metrics overrides avoid invisible text and font-swap CLS. Critical CSS inlining removes a render-blocking round trip.

*Reuse prior work.* `Cache-Control: max-age=31536000, immutable` with hashed filenames; service-worker precaching for repeat visits; *bfcache* (back/forward cache) makes history navigations instant — kept alive by avoiding `unload` handlers and `Cache-Control: no-store`; and the Speculation Rules API (2023–2024) prerenders likely next pages, making intra-site navigation effectively zero-LCP.

== Runtime Responsiveness

INP work is event-loop work (see _Event Loop and Scheduling_): an interaction's latency is input delay (a long task already running) + handler processing + presentation (the rendering pipeline). The playbook:

- Break long tasks with `scheduler.yield()` / `postTask`; defer non-visual side effects (analytics, state sync) until after the next paint (`requestAnimationFrame` then `setTimeout`).
- Avoid layout thrashing in handlers; batch reads and writes (see _CSS and Layout_).
- Shrink DOM size: style, layout, and memory all scale with node count; `content-visibility: auto` and list virtualization fence what's offscreen.
- Move computation to workers; hydrate lazily (or partially — islands architectures, React Server Components, and resumability in Qwik exist substantially to cut main-thread script).
- Watch third parties: tag managers, A/B testing, and ad scripts are the dominant cause of field long tasks; load them after interactive, sandbox them in iframes or Partytown-style workers, and measure them separately.

== The Business Case and Its Caveats

The canonical numbers: Amazon's "100 ms costs 1% of revenue" (2006, internal A/B), Google's 400 ms search delay reducing searches per user 0.6% (Brutlag, 2009), Pinterest's 40% perceived-latency cut yielding 15% more sign-ups (2017), and the Deloitte/Google "Milliseconds Make Millions" study (2020): a 0.1 s mobile speed improvement lifted retail conversions ~8%. Caveats worth stating: most published figures are correlational, survivorship-filtered, and a decade old; the robust generalizations are that latency effects are real, nonlinear (cliffs around perceptual thresholds: ~100 ms "instant", ~1 s flow break, ~10 s abandonment — Nielsen's thresholds, from Miller 1968 and Card et al.), and largest for the slowest cohort, which is precisely what p75-on-mobile targets.

== Pitfalls

- *Optimizing the median on your MacBook*: your users' p75 device is a mid-range Android on flaky 4G; throttle, or better, read CrUX.
- *Score chasing*: Lighthouse's score is a weighted lab composite; shipping `<lighthouse score>`-driven hacks (e.g. lazy-loading everything, deferring the LCP image) can worsen field vitals.
- *Measuring without attribution*: knowing INP is 600 ms is useless; LoAF and `event` timing entries tell you *which* script.
- *Preloading everything*: `rel="preload"` steals bandwidth from naturally-discovered critical resources; more than a handful is self-defeating.
- *Ignoring soft navigations*: SPAs report one navigation per session to CrUX; route changes need custom instrumentation (the soft navigations API is still experimental).
- *CLS from late personalization*: cookie banners, A/B-injected hero content, and ads without reserved slots are the field's top shift sources; reserve space (`aspect-ratio`, `min-height`).
- *Unload handlers*: a single `unload` listener (often from old analytics) disables bfcache, silently doubling effective back-navigation LCP.

== Further Reading

- Walton, P., et al. "Web Vitals", "Optimize LCP / INP / CLS" guides (web.dev, continuously updated; INP threshold rationale in "Defining the INP metric").
- Brutlag, J. (2009). Speed matters for Google web search. Google research memo; and Schurman, E. & Brutlag, J., "The user and business impact of server delays" (Velocity 2009).
- HTTP Archive (Viscomi, R., ed.). _Web Almanac_, Performance chapter (httparchive.org, annual) — field-data state of the union.
- Wagner, J., & Hempenius, K. "Optimize Largest Contentful Paint" and the LCP subparts model (web.dev, 2023).
- Grigorik, I. (2013). _High Performance Browser Networking_. O'Reilly. (Free at hpbn.co; the networking half of this chapter.)

= Rendering Pipeline

After style and layout produce a geometry tree, the browser must turn it into pixels — repeatedly, sixty or more times per second, while the page scrolls, animates, and mutates underneath. This chapter covers the stages downstream of layout: paint, layerization, rasterization, and compositing, with Chromium's pipeline (Blink + cc + Viz) as the running example, plus Firefox's WebRender as the contrasting design.

*See also:* _CSS and Layout_ (the geometry that paint consumes), _Browser Architecture_ (the GPU process and Viz), _Event Loop and Scheduling_ (frame timing and `requestAnimationFrame`), _Web Performance_ (LCP, INP, and jank).

== The Stages of a Frame

The canonical per-frame pipeline on the renderer main thread, in order:

+ *Input handling*: dispatch queued input events (pointer, keyboard) whose handlers may mutate the DOM.
+ *`requestAnimationFrame` callbacks*: the sanctioned hook for script-driven animation, run once per frame before style.
+ *Style*: recalculate computed styles for invalidated elements.
+ *Layout*: recompute geometry for dirty subtrees (the fragment tree in LayoutNG).
+ *Pre-paint*: build property trees and compute paint invalidation.
+ *Paint*: record display lists — not pixels — for each element.
+ *Commit*: hand the display lists and property trees to the compositor thread.

After commit, work leaves the main thread entirely: tiling, rasterization, and draw happen on the compositor thread, raster worker threads, and the GPU process. A frame that skips stages is cheaper: a `transform` change re-runs only composite; a `color` change re-runs paint and composite; a `width` change re-runs everything. This is the *rendering waterfall* that "CSS triggers" tables describe.

== Paint: Display Lists, Not Pixels

Paint does not produce bitmaps. It walks the layout tree in *stacking order* (the painter's algorithm order defined by stacking contexts and `z-index`) and records drawing commands — "fill this rect with this color", "draw this glyph run" — into a *display list*. In Blink these are `PaintRecord`s built on Skia's recording infrastructure; the actual rasterization is deferred.

Paint order within a stacking context is specified precisely: background and borders, negative z-index children, block backgrounds, floats, inline content, then positive z-index children. Properties like `opacity` below 1, `transform`, `filter`, and `isolation: isolate` create new stacking contexts, which is why a `z-index: 9999` element can still be trapped under a sibling with `opacity: 0.99`.

*Paint invalidation* is tracked per display item: Blink's *paint artifact* diffing means a small visual change re-records only affected items rather than whole layers.

== Property Trees and Layerization

Until around 2016, compositing reasoning was per-layer: each composited layer carried its own transform, clip, and effect state, duplicated and frequently wrong. Blink's *Slimming Paint* project replaced this with four global *property trees*:

- *Transform tree*: every node is a transform (including scroll offsets, which are just transforms).
- *Clip tree*: rectangular and rounded clips.
- *Effect tree*: opacity, filters, blend modes, masks.
- *Scroll tree*: scrollable regions and their constraints.

Every display item points at a node in each tree. Layerization — deciding which content gets its own GPU texture — becomes a late, separable optimization (*composite after paint*, shipped in Chrome 94, 2021) instead of a structural commitment made during style.

A layer is warranted when content moves independently: scrolling content, `transform`/`opacity` animations, `will-change: transform`, video, canvas, and out-of-process iframes. Layers cost memory (width × height × 4 bytes per texture, before mipmaps), so *layer explosions* — hundreds of accidentally-promoted layers — can exhaust GPU memory on mobile. Engines fight back with layer squashing and by ignoring `will-change` hints under memory pressure.

== Tiling and Rasterization

The compositor thread (cc, "the Chrome compositor") divides each layer into *tiles*, typically 256×256 or 512×512 pixels. Tiles are prioritized by distance from the viewport and rasterized by a pool of worker threads, which replay the recorded display lists through Skia. Since around 2020 Chromium rasterizes on the GPU by default (*GPU rasterization*, then *OOP-R*: raster in the GPU process over a command buffer), with Skia's Ganesh backend giving way to *Graphite* (Dawn/WebGPU-based, rolling out from 2024).

Key consequences of tiling:

- Scrolling pre-rasterizes tiles beyond the viewport, so most scrolls hit ready textures (checkerboarding is the visible failure when raster can't keep up).
- Low-resolution placeholder tiles can be shown during fast scrolls and pinch-zoom, re-rasterized at the right scale afterward.
- Invalidating a small rect re-rasters only the tiles it touches.

== Compositing and Viz

The final draw is performed by *Viz*, Chromium's display compositor in the GPU process. Each frame source (renderer compositors, the browser UI, video) submits a *compositor frame* containing quads referencing textures; Viz aggregates frames from all surfaces (crucial for out-of-process iframes, each rendered by a different process), performs occlusion culling, and issues the final draw, presenting via OpenGL, Vulkan, Metal, or D3D through ANGLE/Dawn abstractions.

Because the compositor thread owns scrolling and a copy of the property trees, *threaded scrolling* keeps pages responsive even when the main thread is blocked by JavaScript: the compositor applies the scroll offset and draws existing tiles without the main thread's involvement. The exception is content with non-passive event listeners: a `touchstart`/`wheel` listener that might call `preventDefault()` forces the compositor to wait for the main thread — the reason passive listeners (`{ passive: true }`, 2016) were introduced. Similarly, *compositor-driven animations* of `transform` and `opacity` (and `background-position`-free `@keyframes` generally) run entirely off the main thread, which is the mechanistic basis of the "animate only transform and opacity" rule.

== Frame Scheduling and VSync

Displays refresh at fixed intervals (16.7 ms at 60 Hz, 8.3 ms at 120 Hz). The GPU process distributes *BeginFrame* signals aligned to vsync; the renderer's scheduler decides per-frame whether to produce a main-thread frame, a compositor-only frame, or nothing. A main-thread frame that misses its deadline causes a dropped or delayed frame — *jank*. The compositor can also operate in low-latency mode for canvas and input-heavy content, trading pipelining depth for responsiveness. The full pipeline (input → main → commit → raster → draw → display) is normally pipelined across 2–3 vsync intervals, which is why even a perfectly smooth page has roughly 1–2 frames of input latency.

== WebRender: The Rasterize-Everything Alternative

Firefox's *WebRender* (from Servo, shipped progressively since Firefox 67, 2019) takes a different stance: instead of caching rasterized tiles and compositing them, it re-renders the entire visible scene on the GPU every frame, like a game engine. The display list is translated into GPU batches; text, gradients, and box shadows are drawn by specialized shaders; picture caching was later added to skip genuinely static regions. The bet is that GPUs are fast enough that redrawing is cheaper than the bookkeeping of invalidation — true for many scenes, with the trade-off of higher steady-state GPU load.

== Pitfalls

- *Forcing layerization with `will-change` everywhere*: each layer costs texture memory and compositing time; promote only elements that actually animate, and remove the hint after the animation.
- *Animating layout properties*: `top`, `left`, `width`, `height`, and `margin` re-enter layout every frame; use `transform: translate(...)` and `scale(...)` instead.
- *Large paint areas behind small changes*: a tiny blinking caret inside a layer with an expensive `box-shadow` can re-raster the shadow; isolate expensive static decoration from animated content.
- *Reading "CSS triggers" tables as gospel*: which property triggers what differs per engine and version (e.g. `opacity` is composite-only solely when the element is already promoted).
- *Canvas and `getImageData`*: reading back GPU-rasterized canvas pixels stalls the pipeline; prefer `willReadFrequently: true` (software canvas) when readbacks are frequent.
- *Assuming 60 Hz*: 120 Hz mobile displays and variable-refresh desktops halve your frame budget; measure against `requestAnimationFrame` timestamps, not wall-clock assumptions.

== Further Reading

- Chromium design docs: "Life of a Pixel" (Steiner, S., continuously updated), "Compositor Thread Architecture", and "Slimming Paint / CompositeAfterPaint" (chromium.org).
- Chromium Graphics team. "Viz: the Chromium display compositor" and "RenderingNG" series (developer.chrome.com, 2021).
- Mozilla GFX team (2017–2019). "WebRender newsletter" series and Glenn Watson's "WebRender capture" posts (mozillagfx.wordpress.com).
- Lewis, P. "The Anatomy of a Frame" (aerotwist.com, 2016).

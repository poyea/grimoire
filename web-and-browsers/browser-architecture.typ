= Browser Architecture

A modern browser is one of the largest pieces of software most people run: Chromium contains roughly 35 million lines of code, comparable to an operating system kernel plus much of its userland. This chapter covers the multi-process architecture that browsers converged on, the security boundaries between processes, inter-process communication, and an end-to-end overview of how a URL typed in the address bar becomes pixels on screen.

*See also:* _Rendering Pipeline_ (compositing and rasterization in depth), _JavaScript Engines_ (V8 inside the renderer), _Event Loop and Scheduling_ (task scheduling within a renderer), and the Networking volume (HTTP, TLS, DNS for the network service).

== The Multi-Process Model

Early browsers (Netscape, IE6, Firefox until 2016) were single-process: one crash in any tab, plugin, or extension took down the whole browser. Chrome shipped in 2008 with a multi-process design, and every major engine has since followed (Firefox's Electrolysis/Fission, WebKit2).

The principal process types in Chromium:

#table(
  columns: 3,
  [*Process*], [*Count*], [*Responsibility*],
  [Browser], [1], [UI, omnibox, tab strip, profile, coordination, privileged I/O],
  [Renderer], [many], [Blink (DOM, CSS, layout), V8; one per site instance],
  [GPU], [1], [Rasterization, compositing, WebGL/WebGPU command submission],
  [Network], [1], [Sockets, HTTP cache, cookies, TLS (a "utility" service)],
  [Utility], [varies], [Audio, storage, data decoding, on-demand services],
  [Plugin/PPAPI], [legacy], [Historically Flash; removed in 2021],
)

The trade-off is memory: each renderer carries a copy of V8's heap, Blink's structures, and shared library overhead. Chrome mitigates this with *process reuse* heuristics under memory pressure (process-per-site rather than process-per-site-instance on low-memory Android devices) and with shared read-only memory for V8 startup snapshots.

== Site Isolation

A renderer process is treated as *potentially compromised*: a malicious page that exploits a Blink or V8 bug gains arbitrary code execution inside the renderer. The defense is two-layered.

*Sandboxing* restricts what a compromised renderer can do. On Linux, renderers run under a seccomp-BPF filter plus namespace isolation; on Windows, restricted tokens, job objects, and win32k lockdown; on macOS, the Seatbelt sandbox. A renderer cannot open files, create sockets, or spawn processes — every privileged operation is a request to the browser process, which validates it.

*Site isolation* (Chrome 67, 2018) restricts what a compromised renderer can *know*. Each renderer hosts documents from only one *site* (scheme plus eTLD+1, so `https://mail.example.com` and `https://www.example.com` share a site). Cross-site iframes are placed in separate processes — *out-of-process iframes (OOPIF)* — with the frame tree split across renderers and composited together. The browser process enforces that a renderer for `evil.com` cannot receive cookies, stored data, or *cross-origin read blocking (CORB/ORB)*-protected responses belonging to `bank.com`.

Site isolation became urgent after Spectre (2018): speculative-execution side channels let JavaScript read any memory in its own address space, so the only robust defense is to never map cross-site data into the renderer at all. The cost was about 10–13% more memory; Firefox shipped the equivalent (Fission) in 2021.

== Inter-Process Communication: Mojo

Chromium's IPC layer is *Mojo*. Interfaces are declared in `.mojom` IDL files and compiled into C++, Java, and JavaScript bindings. Key concepts:

- *Message pipes*: bidirectional, asynchronous channels carrying serialized messages and handles. Pipes can themselves be sent over pipes, so the object graph is dynamic.
- *Remotes and receivers*: a `Remote<Foo>` is the calling end of an interface; a `Receiver<Foo>` binds an implementation. Calls are asynchronous by default; replies arrive as callbacks.
- *Brokering*: only the browser process can create new processes and distribute initial pipe endpoints; this keeps the privilege hierarchy explicit.
- *Shared memory and data pipes*: bulk data (decoded images, network response bodies) flows through shared buffers rather than serialized messages.

Mojo replaced "legacy IPC" (a single ordered channel per renderer with hand-numbered message IDs) and enabled the *servicification* of Chrome: network, audio, and storage moved from the browser process into separate sandboxed services with no change to callers.

== Threads Inside a Renderer

Each renderer is itself heavily multithreaded:

- *Main thread*: DOM, style, layout, JavaScript execution, event dispatch. The contended resource in web performance.
- *Compositor thread*: handles input scrolling and layer animation without the main thread (threaded scrolling).
- *Raster/worker threads*: tile rasterization, background parsing, GC tasks.
- *IO thread*: receives Mojo messages and routes them to the right thread.
- *Worker threads*: one per dedicated worker, each with its own V8 isolate.

== Servo and Parallelism

Mozilla's *Servo* project (2012–2020, now Linux Foundation hosted) explored how much of the pipeline can be parallelized when an engine is designed for it in Rust. Its signature results: parallel CSS *selector matching and style resolution* using a work-stealing thread pool over the DOM tree, and parallel layout for independent subtrees. Stylo, Servo's style system, shipped inside Firefox 57 "Quantum" (2017) and routinely delivers near-linear speedups on style recalculation across 4–8 cores. Full parallel layout proved harder: floats, inline formatting, and percentage resolution create sequential dependencies, which is one reason mainstream engines still run layout on a single thread per document.

== Renderer Memory Model

Blink and V8 each manage garbage-collected heaps. Blink's C++ objects use *Oilpan*, a tracing GC for DOM objects, so a JavaScript wrapper and its C++ DOM node can be collected together without the reference-counting cycles that plagued earlier engines. V8 and Oilpan run *unified heap* garbage collection: a single marking phase traverses the cross-language object graph, eliminating an entire class of DOM/JS leak. PartitionAlloc, Chromium's hardened allocator, separates allocations by type and partition to blunt use-after-free exploitation, and *MiraclePtr* (`raw_ptr<T>`, 2022) quarantines freed memory still referenced by dangling pointers.

== From Omnibox to Pixels

The life of a navigation, end to end (each stage is expanded in later chapters):

+ *Input*. The omnibox classifies the string: search query or URL? The browser process initiates the navigation.
+ *Fetch*. The network service performs DNS resolution, establishes a TCP+TLS (or QUIC) connection, sends the request, and streams the response. Safe Browsing checks run on the URL.
+ *Commit*. Based on the response's origin, the browser picks or creates a renderer process (site isolation) and commits the navigation; the response body streams to that renderer.
+ *Parse*. The HTML parser tokenizes the byte stream and builds the DOM; the preload scanner discovers subresources early; scripts may block parsing.
+ *Style and layout*. CSS is parsed into the CSSOM; style recalculation assigns computed styles; layout produces a geometry tree (fragment tree in LayoutNG).
+ *Paint and composite*. Paint records display lists per layer; the compositor thread tiles and rasterizes them on the GPU process; layers are drawn to the screen by the display compositor (Viz).
+ *Interact*. The event loop dispatches input, runs script, and re-enters style/layout/paint for whatever changed — the steady state of a living page.

Steps 4–6 are the *critical rendering path*; minimizing the work between commit and first paint is the core of web performance engineering.

== Engine Landscape

#table(
  columns: 4,
  [*Engine*], [*Browser*], [*JS engine*], [*Notes*],
  [Blink], [Chrome, Edge, Opera, Brave], [V8], [Forked from WebKit, 2013],
  [WebKit], [Safari, all iOS browsers until 2024], [JavaScriptCore], [Forked from KHTML, 2001],
  [Gecko], [Firefox], [SpiderMonkey], [Stylo and WebRender from Servo],
  [Servo], [embeddings, experiments], [SpiderMonkey], [Rust, parallel style],
)

The EU Digital Markets Act forced Apple to permit non-WebKit engines on iOS in 2024, the first crack in a decade of engine monoculture on that platform.

== Further Reading

- Reis, C., Moshchuk, A., & Oskov, N. (2019). Site isolation: process separation for web sites within the browser. _USENIX Security_.
- Barth, A., Jackson, C., Reis, C. (2008). The security architecture of the Chromium browser. Technical report.
- Chromium design docs: "Multi-process architecture", "Mojo docs", and "Life of a Navigation" (chromium.org).
- Anderson, B. et al. (2016). Engineering the Servo web browser engine using Rust. _ICSE SEIP_.

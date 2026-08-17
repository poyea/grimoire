#import "../template.typ": xref

= WebAssembly

WebAssembly (Wasm) is the web's second language: a compact, statically typed binary instruction format designed to be a portable compilation target with near-native performance, validated and sandboxed by construction. Announced in 2015 by a four-browser coalition, an MVP shipped in all engines in 2017, and Wasm 2.0/3.0-era features (GC, threads, SIMD, tail calls, exceptions) have steadily landed since. This chapter covers the design, how engines execute it, the memory and security model, and the ecosystem beyond the browser.

*See also:* #xref("web-and-browsers", "javascript-engines", label: "JavaScript Engines") (shared compiler infrastructure and tiering), #xref("web-and-browsers", "browser-architecture", label: "Browser Architecture") (the sandbox Wasm lives inside), #xref("web-and-browsers", "web-performance", label: "Web Performance") (when Wasm actually wins).

== Design: A Sandboxed Stack Machine

A Wasm *module* contains typed functions over four value types in the MVP — `i32`, `i64`, `f32`, `f64` (plus `v128` with SIMD, and reference types later) — operating on a structured *stack machine*. Control flow is structured by construction: `block`, `loop`, `if`, and branches that may only target enclosing constructs. There is no `goto`, so validation is a single linear pass, the type of every stack slot is known statically, and irreducible control flow cannot be expressed (compilers from languages that have it, like C with `goto`, relax it via the *relooper* or, in modern Binaryen/LLVM, stackifier algorithms).

The binary format is dense (typically 30–50% smaller than equivalent minified asm.js) and designed for *streaming*: a module can be validated and compiled while still downloading (`WebAssembly.instantiateStreaming`, 2017), with code sections laid out so function bodies can be compiled in parallel as they arrive.

Wasm's lineage runs through *asm.js* (Mozilla, 2013) — a statically typeable JavaScript subset with `x|0` integer coercions that engines recognized and ahead-of-time compiled — which proved big C++ codebases (Unreal Engine, ported in a week) could run on the web, and whose limitations (parse cost, doubles-only semantics edge cases) directly motivated a real binary format.

== Linear Memory and the Security Model

A module's memory is one (or more, with multi-memory) resizable *linear memory*: a contiguous, byte-addressable buffer, exposed to JavaScript as an `ArrayBuffer`. All loads and stores are bounds-checked against the current memory size. On 64-bit platforms engines avoid per-access checks with the *guard page* trick: reserve 4 GiB plus guard region of virtual address space so any 32-bit address plus offset lands either in memory or in protected pages, turning bounds violations into trapped signals at zero per-access cost. (Memory64, shipped 2024–2025, brings explicit bounds checks back — one reason 64-bit Wasm is measurably slower.)

Crucially, *code is not in linear memory*. Function pointers are indices into typed *tables*; an indirect call checks the function's signature at runtime. The call stack is engine-managed and unaddressable. Consequences: classic stack smashing and code injection are impossible by construction — but memory-unsafe source languages still suffer heap corruption *within* linear memory (a buffer overflow can overwrite the module's own data, as Lehmann, Kinder, and Pradel demonstrated in "Everything Old is New Again", USENIX Security 2020). Wasm sandboxing protects the host from the module, not the module from itself.

== How Engines Execute Wasm

Engines reuse their JIT infrastructure with Wasm-specific tiers:

#table(
  columns: 3,
  [*Engine*], [*Baseline tier*], [*Optimizing tier*],
  [V8], [Liftoff (single-pass, ~10× faster compile)], [TurboFan],
  [SpiderMonkey], [RabaldrMonkey (baseline)], [Ion],
  [JavaScriptCore], [LLInt-in-place + BBQ], [OMG (B3)],
)

Liftoff (2018) compiles roughly tens of MB/s per thread so large modules start almost immediately; TurboFan recompiles hot functions in the background (*dynamic tiering*, 2022, replaced compiling everything eagerly). Because Wasm is statically typed, the optimizing tier needs no speculation, inline caches, or deoptimization for core code — compiled code is predictable, which is much of Wasm's performance story: not faster peak arithmetic than well-JITted JavaScript, but no warmup cliffs, no hidden-class hazards, no GC pauses for linear-memory data. Honest benchmarks put typical Wasm at roughly 1.2–1.5× native time (Jangda et al., USENIX ATC 2019), and commonly 1.5–3× faster than equivalent JavaScript on compute-heavy kernels.

Engines cache compiled machine code (keyed by module bytes) in HTTP cache metadata, so repeat visits skip compilation entirely.

== Post-MVP Features

- *Threads* (2019, behind cross-origin isolation): shared linear memory via `SharedArrayBuffer`, atomics, and wait/notify; pthreads compile via a worker pool.
- *SIMD* (fixed-width 128-bit, 2021): `v128` with portable lanewise ops; *relaxed SIMD* (2023) adds FMA and platform-variant semantics for more speed.
- *Reference types and typed function references*: opaque host references (`externref`) flow through Wasm without copying.
- *Wasm GC* (Chrome 119 and Firefox 120, late 2023; Safari 18.2, late 2024): structs and arrays allocated on the engine's GC heap, enabling Java (J2CL/Kotlin), Dart/Flutter, and OCaml to target Wasm without shipping a garbage collector in linear memory.
- *Tail calls* (2023), *exception handling* (2024, redesigned `exnref` form), *multi-value returns*, *bulk memory*, *multi-memory* (2024), *Memory64* (2025), and *JS Promise Integration (JSPI)* (2024–2025) for suspending Wasm on async host calls.

== Toolchains and the Boundary

Emscripten (C/C++, predating Wasm itself), Rust's `wasm32-unknown-unknown` target with `wasm-bindgen`, Go (both the large-runtime official target and TinyGo), AssemblyScript (TypeScript-like), and Blazor (.NET runtime in Wasm) are the main producers; Binaryen's `wasm-opt` is the standard post-link optimizer.

The JS↔Wasm boundary is where performance dies: only numbers (and references) cross directly, so strings and structures are copied through linear memory with encoder/decoder glue. Call overhead itself is now small (near-zero after V8's 2018 and Firefox's "calls between JS and WebAssembly are finally fast" work), but per-call *data marshalling* is not. Design rule: cross the boundary rarely, with big batches of plain numbers — a Wasm function called per-pixel will lose to JavaScript; one called per-frame with a million-pixel buffer wins. Real deployments follow this shape: Figma's C++ multiplayer engine and renderer, Google Sheets' calculation engine (Java via Wasm GC), Photoshop on the web (Emscripten, 2021–2023), FFmpeg, SQLite, and ML runtimes with SIMD+threads.

== Beyond the Browser: WASI and the Component Model

Wasm's sandboxing and portability made it a server-side and embedded target. *WASI* (WebAssembly System Interface, 2019) defines capability-based system APIs — file descriptors and sockets are unforgeable handles passed in, not ambient authority; WASI Preview 2 (2024) rebuilt it on the *component model*, which gives modules typed, language-neutral interfaces (WIT IDL) and shared-nothing composition. Standalone runtimes (Wasmtime, Wasmer, WasmEdge) and edge platforms (Fastly Compute, Cloudflare Workers' Wasm support) exploit microsecond-scale instantiation for per-request isolation — far cheaper than containers, stronger than threads. Solomon Hykes' line, "If WASM+WASI existed in 2008, we wouldn't have needed to create Docker", overstates politely but captures the trajectory.

== Pitfalls

- *Porting everything to Wasm for speed*: DOM-heavy or string-heavy workloads gain nothing and pay marshalling tax; profile first, port kernels only.
- *Ignoring download size*: a Rust module with a large dependency tree or a .NET runtime can dwarf the JavaScript it replaces; `wasm-opt -Oz`, code-splitting, and streaming compilation matter.
- *Chatty boundaries*: per-element calls across JS↔Wasm dominate runtime; batch.
- *Assuming memory safety*: Wasm protects the host; C bugs still corrupt the module's own linear memory and lack ASLR/canary mitigations native platforms have.
- *Forgetting cross-origin isolation*: threads and `SharedArrayBuffer` silently require COOP/COEP headers; the failure mode is a missing API, not an error message.
- *Leaking linear memory*: Emscripten/Rust heaps grow but never shrink back to the OS; long-lived pages should pool and reuse allocations.

== Further Reading

- Haas, A., Rossberg, A., Schuff, D., Titzer, B., et al. (2017). Bringing the web up to speed with WebAssembly. _PLDI_. (Best paper; the canonical formal design.)
- Jangda, A., Powers, B., Berger, E., & Guha, A. (2019). Not so fast: analyzing the performance of WebAssembly vs. native code. _USENIX ATC_.
- Lehmann, D., Kinder, J., & Pradel, M. (2020). Everything old is new again: binary security of WebAssembly. _USENIX Security_.
- Zakai, A. (2011). Emscripten: an LLVM-to-JavaScript compiler. _OOPSLA companion_.
- V8 blog: "Liftoff" (2018), "Dynamic tiering" (2022), "WebAssembly Garbage Collection (WasmGC) now enabled by default" (2023, developer.chrome.com).

= Event Loop and Scheduling

JavaScript's concurrency model is famously "single-threaded with an event loop", but the real machinery is richer: multiple task queues with different priorities, a microtask queue with run-to-completion semantics that surprises almost everyone, rendering interleaved at vsync boundaries, and — in modern browsers — a cooperative scheduler with explicit priorities. This chapter covers the HTML event loop as specified, how Chromium actually schedules it, and the APIs for yielding, prioritizing, and offloading work.

*See also:* _JavaScript Engines_ (what runs inside each task), _Rendering Pipeline_ (the rendering steps the loop interleaves), _Web Performance_ (INP, long tasks, and TBT).

== Tasks and Task Sources

The HTML specification defines the event loop as: pick one runnable *task* from one of the *task queues*, run it to completion, drain microtasks, then possibly *update the rendering*, repeat. Tasks come from *task sources* — timers, user interaction, networking, DOM manipulation, `postMessage`, history traversal — and the spec deliberately lets the browser choose *which* queue to service next. That freedom is the basis of real scheduling: Chromium's Blink scheduler maintains dozens of queues per frame and prioritizes input over timers, throttles timers in background tabs (to 1 per second, then to 1 per minute under "intensive throttling", 2021), and can pause queues entirely during page lifecycle freezing.

Run-to-completion is the core guarantee: a task is never preempted, so no other script observes intermediate state — and a 300 ms task blocks input for 300 ms. Anything over 50 ms is a *long task* by definition (the basis of Total Blocking Time and a major input to INP).

== Microtasks

The *microtask queue* is drained to exhaustion after every task and, more precisely, every time the JavaScript execution stack empties (a "microtask checkpoint" — this includes after each event listener during dispatch). Microtasks are promise reactions, `queueMicrotask`, and `MutationObserver` callbacks.

Two consequences trip up working engineers:

- A microtask that enqueues microtasks can starve the loop forever: `function f() { Promise.resolve().then(f) }` never yields to rendering or input, unlike the equivalent `setTimeout` chain.
- Ordering: in the classic quiz, `Promise.resolve().then(...)` runs before `setTimeout(..., 0)` because microtasks drain before the next task; and `await` is sugar for `.then`, so everything after an `await` runs in a microtask.

```js
setTimeout(() => console.log("task"));
Promise.resolve().then(() => console.log("microtask"));
console.log("sync");
// → sync, microtask, task
```

`MutationObserver` (which replaced synchronous mutation events; the deprecated events were finally removed from Chrome in 2024) batches DOM mutations into one microtask delivery — the mechanism Vue and other frameworks historically used for `nextTick`.

== Rendering Opportunities

"Update the rendering" is not after every task. The loop takes a *rendering opportunity* aligned with the display's refresh: run `requestAnimationFrame` callbacks, then style, layout, paint (the pipeline of the _Rendering Pipeline_ chapter). Between vsyncs, the loop may process many tasks — or one long one. Corollaries:

- Two synchronous style changes in one task never both render; only the final state paints.
- `requestAnimationFrame` runs *before* paint, so it reads pre-frame state and its writes appear in the same frame. rAF scheduled from within rAF runs next frame — the standard animation loop.
- `requestIdleCallback` (Chrome 47, 2015) runs when the loop is idle, with a deadline argument bounding work before the next frame; pair with a 50 ms-chunk strategy for background work. It is unsupported in Safari, so feature-detect.
- `setTimeout(fn, 0)` actually means roughly 1 ms in modern Chrome (the historical 4 ms clamp now applies only after 5 levels of nesting), and timers are aggressively throttled in background tabs and cross-origin iframes.

== Timers, Clamping, and Resolution

Timer guarantees are minimums, never exact: the callback becomes runnable after the delay and then waits its turn. `performance.now()` provides a monotonic clock, but its resolution is deliberately coarsened (100 microseconds in Chrome, 1 ms in Safari and in non-isolated contexts) as a Spectre mitigation; full-precision timers and `SharedArrayBuffer` require *cross-origin isolation* (COOP+COEP headers, 2020).

== The Scheduler API

`scheduler.postTask` (Chrome 94, 2021; the Prioritized Task Scheduling spec) finally exposes prioritized scheduling directly:

- Priorities: `user-blocking`, `user-visible` (default), `background`, with dynamic re-prioritization via `TaskController`.
- `scheduler.yield()` (Chrome 129, 2024) yields and resumes with *continuation priority* — the resumed work goes to the front of its queue rather than the back, fixing the classic problem that `await new Promise(r => setTimeout(r))` lets arbitrary other work cut in line.
- `isInputPending()` (Chrome 87, from Facebook's collaboration) lets a long loop poll whether input is waiting and yield only then, avoiding unnecessary yields.

The break-up-long-tasks idiom, 2024 edition:

```js
async function processAll(items) {
  for (const item of items) {
    process(item);
    if (navigator.scheduling?.isInputPending?.() ?? true) {
      await scheduler.yield();   // resume ahead of other queued work
    }
  }
}
```

== Workers and Off-Main-Thread Work

The only true parallelism for page script: *dedicated workers* (own event loop, own V8 isolate, no DOM), *shared workers* (one instance per origin shared across tabs), and *service workers* (event-driven network proxies, terminated when idle). Communication is `postMessage` with structured cloning; `ArrayBuffer`s can be *transferred* (zero-copy ownership move), and with cross-origin isolation, `SharedArrayBuffer` plus `Atomics` gives genuine shared memory — `Atomics.wait` may only block in workers, never the main thread. `OffscreenCanvas` (Chrome 69; Safari 17, 2023) moves canvas rasterization to a worker, and Houdini's worklets run tiny pieces (paint, audio) on engine-controlled threads.

The practical rule: workers excel at coarse-grained, message-passing parallelism (parsing, compression, image processing — comlink makes the ergonomics tolerable); they cannot touch the DOM, so "move my framework to a worker" founders on synchronous DOM access.

== Node.js Contrast

Node's loop (libuv) is phase-based: timers → pending callbacks → poll (I/O) → check (`setImmediate`) → close callbacks, with microtasks drained between callbacks and `process.nextTick` forming a separate, even-higher-priority queue ahead of promises. Browser intuitions mostly transfer; `setImmediate` vs `setTimeout(0)` ordering and `nextTick` starvation are the Node-specific traps.

== Pitfalls

- *Awaiting in a loop without yielding*: `await fetch(...)` yields, but `await` on an already-resolved promise only defers to the microtask queue — rendering and input still starve.
- *Microtask starvation*: recursive promise chains block rendering indefinitely; recursive `setTimeout` does not.
- *Assuming timer fidelity*: background-tab throttling breaks `setInterval`-driven clocks and games; drive animation from rAF timestamps and recompute from wall-clock deltas.
- *Listening without `passive: true`*: non-passive `wheel`/`touchstart` listeners force the compositor to wait on the main thread before scrolling (see _Rendering Pipeline_).
- *Reading layout in rAF after writes*: rAF runs before style/layout; write-then-read inside one callback still forces synchronous layout.
- *`setInterval` re-entrancy*: if the handler exceeds the interval, callbacks queue up or coalesce per engine; prefer self-rescheduling `setTimeout`.

== Further Reading

- WHATWG. _HTML Living Standard_, §8.1.7 "Event loops" (html.spec.whatwg.org).
- Archibald, J. (2015). "Tasks, microtasks, queues and schedules" (jakearchibald.com); and his JSConf.Asia 2018 talk "In the Loop".
- W3C/WICG. _Prioritized Task Scheduling_ spec; "Optimize long tasks" (developer.chrome.com, 2022, updated for `scheduler.yield`).
- Chromium scheduling docs: "Blink Scheduler" and Sasha Kondrashov's "Threading and tasks in Chrome" (chromium.org).

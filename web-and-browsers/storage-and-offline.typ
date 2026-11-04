#import "../template.typ": xref

= Storage, Service Workers, and Offline <storage-and-offline>

Web applications that work offline, sync in the background, and start instantly from a home-screen icon depend on two browser capabilities: client-side storage and *service workers*, programmable proxies that sit between a page and the network. Both are full of sharp edges: quotas that differ by browser, storage that can be evicted without warning, transactions that close unexpectedly, and caches that can pin users to an old version of an app indefinitely. This chapter covers the storage APIs and what each is for, quotas and eviction, IndexedDB's transaction model, the origin private file system, service-worker lifecycle and caching strategies, background capabilities, and coordination between tabs.

*See also:* #xref("web-and-browsers", "browser-networking", label: "Networking and Resource Loading") (the HTTP cache), #xref("web-and-browsers", "web-security-model", label: "The Web Security Model") (storage partitioning), #xref("web-and-browsers", "event-loop-and-scheduling", label: "Event Loop and Scheduling") (workers), #xref("database", "embedded-databases", label: "Embedded Databases") (SQLite)

== The Storage APIs

#table(
  columns: 4,
  [*API*], [*Model*], [*Availability*], [*Use for*],
  [Cookies], [Small strings sent with every matching request], [Main thread (and Cookie Store API in service workers)], [Session identifiers only],
  [`localStorage`, `sessionStorage`], [Synchronous string key-value, about 5 MiB per origin], [Main thread only], [Tiny preferences; never on hot paths],
  [IndexedDB], [Asynchronous, transactional object store with indexes; stores structured-cloneable values and blobs], [Window and workers], [Application data, offline records],
  [Cache API], [Map from `Request` to `Response`], [Window and workers], [Offline copies of HTTP resources],
  [Origin private file system (OPFS)], [Private file system; synchronous access handles in workers], [Window (async), workers (sync)], [Large binary data, databases such as SQLite compiled to WebAssembly],
)

`localStorage` deserves a warning. Its API is synchronous, so a read may block the main thread on disk I/O, and browsers must load an origin's entire store into memory before the first access. It is fine for a theme preference and wrong for anything large or frequently written.

All of these APIs are scoped to the *origin*, and in modern browsers they are also partitioned by *top-level site* when used in a third-party context (#xref("web-and-browsers", "web-security-model", label: "The Web Security Model")). An embedded widget therefore cannot use storage to recognise the same user across different sites.

== Quotas and Eviction

Browsers give each origin a share of the disk, managed by a *storage manager*:

- `navigator.storage.estimate()` reports usage and quota. Chromium allows an origin up to about 60 percent of total disk space; Firefox and Safari apply different and generally tighter limits. Writes beyond the quota fail with `QuotaExceededError`, which applications must handle.
- By default storage is *best-effort*: under disk pressure the browser evicts whole origins, least recently used first, without asking. `navigator.storage.persist()` requests *persistent* storage, which is evicted only by the user. Browsers grant it based on engagement (an installed app, granted notifications, frequent visits) rather than always asking.
- Safari's tracking prevention deletes all script-writable storage (IndexedDB, `localStorage`, service-worker registrations, Cache API) for sites the user has not interacted with as a first party in the last seven days of browser use. Web apps added to the home screen are exempt.

The consequence is that browser storage is a *cache*, not a source of truth, unless persistence is granted, and even then users can clear it. Applications must be able to rebuild local state from the server.

Opaque responses deserve special mention: cross-origin responses fetched without CORS cannot be inspected, so to avoid leaking their size, Chromium charges each one in the Cache API as a large padded size (several megabytes). A service worker that caches a few hundred opaque images can exhaust its quota.

== IndexedDB

IndexedDB is a transactional store of JavaScript values. A database contains *object stores* (like tables), each keyed by a key path or an auto-incrementing key, with optional *indexes* on value properties. Every read and write happens inside a *transaction* scoped to a set of stores, in `readonly` or `readwrite` mode. Under the hood, Firefox and Safari use SQLite, and Chromium historically used LevelDB.

Its transaction lifetime rule is the most common source of bugs: a transaction stays active only while it has pending requests or while their callbacks run. It *auto-commits* as soon as control returns to the event loop with no request outstanding. Awaiting anything that is not an IndexedDB request (a `fetch`, a timer, a promise from another library) lets the transaction commit, and the next request throws `TransactionInactiveError`:

```js
// Wrong: the fetch lets the transaction auto-commit.
const tx = db.transaction("orders", "readwrite");
const store = tx.objectStore("orders");
const latest = await (await fetch("/api/orders/latest")).json();
store.put(latest);            // throws TransactionInactiveError

// Right: do the network work first, then open a short transaction.
const fresh = await (await fetch("/api/orders/latest")).json();
const tx2 = db.transaction("orders", "readwrite");
tx2.objectStore("orders").put(fresh);
await new Promise((resolve, reject) => {
  tx2.oncomplete = resolve;
  tx2.onerror = () => reject(tx2.error);
});
```

Schema changes happen in `upgradeneeded` when a page opens the database with a higher version number. If another tab still has the old version open, the upgrade is *blocked* until that tab closes its connection, so every page should listen for `versionchange` and close its connection (and usually reload) when a newer version of the app needs to upgrade.

The raw API is callback-based and verbose. Thin promise wrappers (such as `idb`) are the usual way to use it, and they respect the transaction rules above.

== The Origin Private File System

OPFS gives each origin a private, sandboxed file system that the user does not see in their file manager. Its distinctive feature is `createSyncAccessHandle()`, available only in dedicated workers, which provides synchronous, in-place reads and writes at offsets. That is exactly the interface a database engine expects, and it is how SQLite compiled to WebAssembly runs in browsers with real durability and performance, giving web apps a full relational database locally. OPFS is also a good place for large media and model files, which are awkward in IndexedDB.

== Service Workers

A service worker is a script registered for a *scope* (a URL path prefix on an origin). Once installed, it receives a `fetch` event for every request made by pages in its scope, including navigations, and can answer from a cache, from the network, or from code. It runs on its own thread, has no DOM, and is started on demand and stopped when idle, so it must not keep state in global variables.

=== Lifecycle

1. *Register*: a page calls `navigator.serviceWorker.register("/sw.js")`. Service workers require HTTPS (or localhost).
2. *Install*: the new worker runs its `install` handler, typically *precaching* the app shell. If any precache request fails, installation fails and the old worker stays.
3. *Wait*: if an older worker controls open pages, the new one waits until all those pages close, so that one page never sees two versions of the app. `self.skipWaiting()` skips the wait, at the risk of a page loaded by the old version now talking to the new one.
4. *Activate*: the new worker's `activate` handler deletes caches from old versions. `clients.claim()` makes it control already-open pages immediately.
5. *Update*: on navigations (and at most every 24 hours) the browser re-fetches the worker script, bypassing the HTTP cache by default, and installs a new version if it differs by even one byte.

The waiting phase is the root of the classic complaint "users don't get the new version": the update installed, but it will not activate until every tab of the app closes. Applications usually show an "update available, reload" prompt that calls `skipWaiting` on request.

=== Caching Strategies

```js
const VERSION = "v42";
const SHELL_CACHE = `shell-${VERSION}`;
const RUNTIME_CACHE = "runtime";
const SHELL = ["/", "/app.3f9a1c.js", "/app.8b21d0.css", "/offline.html"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(SHELL_CACHE).then((c) => c.addAll(SHELL)));
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    for (const name of await caches.keys()) {
      if (name.startsWith("shell-") && name !== SHELL_CACHE) await caches.delete(name);
    }
    if (self.registration.navigationPreload) {
      await self.registration.navigationPreload.enable();
    }
  })());
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.mode === "navigate") {
    // Network first for HTML so deployments are seen; fall back offline.
    event.respondWith((async () => {
      try {
        return (await event.preloadResponse) || (await fetch(request));
      } catch {
        return (await caches.match(request)) || (await caches.match("/offline.html"));
      }
    })());
    return;
  }
  if (request.destination === "image") {
    // Stale-while-revalidate: answer from cache, refresh in the background.
    event.respondWith((async () => {
      const cache = await caches.open(RUNTIME_CACHE);
      const cached = await cache.match(request);
      const network = fetch(request).then((response) => {
        if (response.ok) cache.put(request, response.clone());
        return response;
      });
      event.waitUntil(network.catch(() => {}));
      return cached || network;
    })());
  }
  // Everything else: default browser handling (HTTP cache, network).
});
```

The common strategies, and when each fits:

- *Cache first*: versioned static assets whose URLs change when their content does.
- *Network first, cache fallback*: HTML and API data that should be fresh when online but available offline.
- *Stale-while-revalidate*: avatars, images and other content where slightly old is acceptable and speed matters.
- *Network only*: requests with side effects, analytics, anything that must not be served from a cache.

Two performance notes. Starting a stopped service worker takes tens of milliseconds or more on mobile devices, and a navigation must wait for it before the request can even be sent; *navigation preload* (used above) sends the navigation request in parallel with worker startup. And a service worker that intercepts every request and just calls `fetch` adds overhead for no benefit; leaving requests unhandled lets the browser use its normal fast path.

== Background Capabilities

Service workers also receive events when no page is open:

- *Push*: a server sends an encrypted message through the browser vendor's push service (authenticated with VAPID keys), the browser wakes the service worker, and the worker must show a notification. Web push works on all major browsers, on iOS only for web apps added to the home screen.
- *Background sync*: a page registers a sync tag when an action fails offline, and the browser fires a `sync` event when connectivity returns, letting the worker retry. Periodic background sync allows occasional refreshes for installed apps with high engagement. Support for both is mostly limited to Chromium.
- *Background fetch*: large downloads that continue after the page closes, with browser-provided progress UI.

The *web app manifest* (name, icons, start URL, display mode) plus a service worker make an app *installable*, so it launches from the home screen or desktop in its own window like a native app.

== Coordinating Tabs

A user may have the same app open in several tabs, all sharing one origin's storage and possibly one service worker:

- `BroadcastChannel` sends messages to all same-origin contexts, for example "the user logged out" or "new data synced".
- The *Web Locks API* (`navigator.locks.request(name, callback)`) provides cross-tab mutual exclusion, useful for electing one tab to own a WebSocket connection or to run a sync job.
- The `storage` event notifies other tabs of `localStorage` changes, and a `SharedWorker` gives tabs a common thread with shared in-memory state (where supported).

Designing an offline-capable app means treating every tab, the service worker, and the server as replicas that can diverge: queues of pending writes, conflict resolution on sync, and idempotent server APIs. The same techniques that keep distributed systems consistent apply in the browser.

== Further Reading

W3C (2024). "Service Workers Nightly." W3C Editor's Draft specification.

W3C (2024). "Indexed Database API 3.0." W3C Working Draft specification.

WHATWG (2024). "Storage Standard." Living standard (quotas, buckets, persistence).

WHATWG (2024). "File System Standard." Living standard (origin private file system).

W3C (2024). "Push API." W3C Working Draft specification.

W3C (2024). "Web App Manifest." W3C Working Draft specification.

Archibald, J. (2014). "The Offline Cookbook." web.dev documentation.

WebKit (2020). "Full Third-Party Cookie Blocking and More." WebKit blog.

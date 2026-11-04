#import "../template.typ": xref

= Modules, Bundlers, and the Build Toolchain <modules-and-tooling>

The JavaScript a browser runs is rarely the JavaScript a developer wrote. Between them sits a toolchain that resolves hundreds or thousands of modules from a package registry, strips TypeScript types and compiles JSX, removes unused code, splits the result into chunks, minifies and fingerprints it, and emits source maps that let the debugger show the original code again. This chapter covers the module systems that toolchain works with, the ES module loading model in browsers, why bundling survives even though browsers load modules natively, how bundlers work inside, source maps, hot module replacement, and the supply-chain risks that come with depending on a registry of millions of packages.

*See also:* #xref("web-and-browsers", "javascript-engines", label: "JavaScript Engines") (parsing and code caching), #xref("web-and-browsers", "browser-networking", label: "Networking and Resource Loading") (modulepreload, caching fingerprinted assets), #xref("web-and-browsers", "web-performance", label: "Web Performance"), #xref("compilers", "frontends", label: "Frontends") (parsing and transformation)

== From Globals to Modules

JavaScript had no module system for its first twenty years. Scripts shared one global scope, and libraries avoided collisions with immediately invoked function expressions and namespace objects. Two community systems filled the gap:

- *CommonJS* (Node.js): `require()` loads a module synchronously and returns its `module.exports` object. Synchronous loading suits a file system, not a network.
- *AMD* (RequireJS): asynchronous `define()` calls for browsers, now largely historical.

*ES modules* (ESM), standardised in ES2015 and supported by all browsers since about 2018, are part of the language:

```js
// geometry.js
export const TAU = 2 * Math.PI;
export function circle_area(r) { return (TAU / 2) * r * r; }
export default class Point { constructor(x, y) { this.x = x; this.y = y; } }

// app.js
import Point, { circle_area } from "./geometry.js";
const { render } = await import("./charts.js");   // dynamic, loaded on demand
```

The design differs from CommonJS in ways that matter to tools:

- *Static structure*: `import` and `export` declarations appear only at the top level with literal specifiers, so the dependency graph and the set of exported names are known without running code. That is what makes tree shaking possible.
- *Live bindings*: an importer sees the exporter's variable, not a copy, so updates are visible, and circular imports work as long as values are not used before they are initialised.
- *Asynchronous, three-phase loading*: the whole graph is fetched and parsed, then *linked* (imports bound to exports), then *evaluated* in post-order. Top-level `await` pauses evaluation of a module and its dependents.
- Modules always run in strict mode, are deferred by default in browsers, and execute once per URL.

Interoperation between the two systems remains a source of friction in Node.js, where both coexist; recent Node versions can `require()` synchronous ES modules, which removes much of it.

== Modules in the Browser

A `<script type="module">` loads its graph natively. Bare specifiers such as `import { html } from "lit"` are not URLs, so the browser needs an *import map* to resolve them:

```html
<script type="importmap">
{
  "imports": {
    "lit": "/vendor/lit@3.2.1/index.js",
    "lit/": "/vendor/lit@3.2.1/"
  }
}
</script>
<link rel="modulepreload" href="/app/main.js">
<script type="module" src="/app/main.js"></script>
```

Native loading has a structural performance problem: the browser discovers a module's imports only after fetching and parsing it, so a dependency graph $d$ levels deep costs at least $d$ sequential round trips. With HTTP/2 or HTTP/3 the per-request cost is small, but the *waterfall* remains. `modulepreload` hints let the server or HTML declare the graph up front, which helps for the first levels. *Import attributes* (`import data from "./data.json" with { type: "json" }`) extend module loading to JSON and CSS, with the type stated explicitly so that a server cannot trick the page into executing a file it expected to be data.

== Why Bundle Anyway

Shipping thousands of small modules individually is still slower than bundling them, for several reasons:

- *Round trips*: deep import graphs create waterfalls that preloading only partly fixes.
- *Per-module overhead*: each request and each module record has fixed costs in the network stack, the cache and the engine.
- *Compression*: Brotli and gzip compress one large file better than many small ones, because repeated patterns across modules share one dictionary window.
- *Transformations*: TypeScript, JSX, and syntax that older target browsers lack must be compiled anyway.
- *Dead code*: libraries export far more than an application uses, and only whole-program analysis can remove the rest.

Bundling everything into one file has the opposite problem: users download code for every route on the first visit, and any change invalidates the whole cached bundle. Production builds therefore *split* code: one chunk per route or lazily loaded feature (from dynamic `import()` points), shared chunks for code used by several routes, and long-lived vendor chunks for libraries that change rarely, each with a content hash in its file name so that it can be cached for a year.

== Inside a Bundler

All major bundlers (webpack, Rollup, esbuild, Parcel, and the Rust-based Rspack, Rolldown and Turbopack) perform similar phases:

1. *Resolution*: turn each specifier into a file, following Node's resolution rules: relative paths, `node_modules` lookup, and the `exports` map in `package.json`, whose *conditions* (`"import"`, `"require"`, `"browser"`, `"development"`) select different files for different environments.
2. *Transformation*: run each file through loaders or plugins: TypeScript type stripping, JSX compilation, CSS processing, syntax lowering for older browsers. The slow JavaScript-based tools (Babel, the TypeScript compiler) have largely been replaced in this role by native-code transformers (esbuild in Go, SWC and Oxc in Rust) that are tens of times faster.
3. *Graph construction*: parse every module, record its imports and exports, and build the module graph.
4. *Tree shaking*: starting from the entry points, mark the exports actually used and drop the rest. Because JavaScript has side effects at module load (a module might patch a global), a bundler may only drop an unused module if it can prove it has none, or if the package declares `"sideEffects": false` in `package.json`. Annotations such as `/*#__PURE__*/` mark individual calls as removable.
5. *Chunking and scope hoisting*: assign modules to chunks, then concatenate the modules in each chunk into one scope where possible (renaming to avoid collisions) instead of wrapping each in a function, which makes the output smaller and faster to start.
6. *Minification*: remove whitespace, shorten local names, fold constants and drop unreachable code (Terser, esbuild, SWC).
7. *Emission*: write chunks with content hashes, a manifest mapping logical names to hashed files, and source maps.

Development servers take a different approach for speed. *Vite* serves source files to the browser as native ES modules, transforming each file on request, so starting the server does not require bundling the application; dependencies are pre-bundled once with esbuild. Production builds still bundle, for the reasons above.

== Source Maps

Debugging minified, bundled code is impossible without a map back to the sources. A *source map* (format version 3, now standardised by Ecma TC39 as ECMA-426) is a JSON file listing the original sources and a `mappings` string. The mappings encode, for each generated line, segments of one to five integers: generated column, source file index, original line, original column, and optionally a name index. Each value is stored as a *delta* from the previous segment and encoded as a base64 variable-length quantity (VLQ):

```cpp
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

int base64_value(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

// Decode one segment of a source map "mappings" string (no ',' or ';').
// Each base64 digit carries 5 value bits plus a continuation bit (0x20);
// the lowest bit of each decoded number is its sign.
std::optional<std::vector<std::int64_t>> decode_vlq_segment(std::string_view segment) {
    std::vector<std::int64_t> fields;
    std::int64_t value = 0;
    int shift = 0;
    for (char c : segment) {
        const int digit = base64_value(c);
        if (digit < 0 || shift > 60) return std::nullopt;
        value += static_cast<std::int64_t>(digit & 0x1F) << shift;
        if (digit & 0x20) {            // more digits follow
            shift += 5;
            continue;
        }
        const bool negative = (value & 1) != 0;
        value >>= 1;
        fields.push_back(negative ? -value : value);
        value = 0;
        shift = 0;
    }
    if (shift != 0) return std::nullopt;   // truncated number
    return fields;
}

// decode_vlq_segment("AACA") -> {0, 0, 1, 0}: same column and source,
// one line further down in the original file, same original column.
```

Source maps can be served publicly, kept private and uploaded to error-tracking services (which symbolicate stack traces from production), or omitted. Publishing them exposes the original source, which may or may not matter. Mappings are frequently imprecise after aggressive minification, which is why breakpoints sometimes land on the wrong line.

== Hot Module Replacement

During development, *hot module replacement* (HMR) swaps edited modules into the running page without a reload, preserving application state. The dev server watches files, recompiles the changed module, and notifies the browser over a WebSocket. The client runtime then walks *up* the module graph from the changed module looking for an *accept boundary*: a module that declared it can handle updates to itself or its dependencies (frameworks generate these for components). Every module between the changed one and the boundary is re-executed; if no boundary is found, the page reloads. React Fast Refresh and its equivalents in other frameworks keep component state across edits by re-rendering components with their existing state.

== The Supply Chain

A typical web application depends, directly and transitively, on hundreds to thousands of npm packages, any of which runs code during installation (`postinstall` scripts) and in the final product. Incidents have exploited every link:

- A maintainer handed over a popular package to a stranger who added code to steal cryptocurrency wallets (event-stream, 2018).
- Typosquatting packages with names similar to popular ones, and dependency confusion between private and public package names.
- In 2024 the domain behind the widely embedded polyfill.io CDN was acquired by a new owner and served malicious code to the many sites still loading scripts from it.

Mitigations, roughly in order of payoff:

- *Lockfiles*, committed and enforced in CI (`npm ci`), so that builds use exactly the reviewed versions.
- *Fewer dependencies*: many small utility packages can be replaced by a few lines of code or by modern built-ins.
- *Self-hosting* third-party scripts, or at least *Subresource Integrity* hashes on CDN scripts (#xref("web-and-browsers", "web-security-model", label: "The Web Security Model")).
- *Disabling install scripts* by default and allowing them per package.
- *Provenance*: npm packages built in CI can carry signed provenance statements (based on Sigstore) linking each tarball to the source commit and build that produced it, so consumers can verify where code came from.
- *Automated scanning* for known vulnerabilities and malicious packages, with dependency update tooling that keeps versions current without accepting changes blindly.

== Further Reading

Ecma International (2024). "ECMAScript 2024 Language Specification." ECMA-262 standard (modules, top-level await).

Ecma International (2024). "Source Map Format Specification." ECMA-426 standard.

WHATWG (2024). "HTML Living Standard." Living standard (module scripts, import maps).

Node.js Project (2024). "Modules: Packages." Node.js documentation (exports and conditions).

Clark, L. (2018). "ES Modules: A Cartoon Deep-Dive." Mozilla Hacks.

Harris, R. (2015). "Tree-shaking versus Dead Code Elimination." Rollup author's essay, reference.

Zimmermann, M., Staicu, C.-A., Tenny, C., Pradel, M. (2019). "Small World with High Risks: A Study of Security Threats in the npm Ecosystem." USENIX Security.

Newman, Z., Meyers, J., Torres-Arias, S. (2022). "Sigstore: Software Signing for Everybody." ACM CCS.

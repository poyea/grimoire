= HTML Parsing and the DOM

HTML is the only mainstream language whose parser is fully specified *including error recovery*: every byte sequence produces a well-defined tree in every conforming browser. This chapter covers the HTML5 parsing algorithm, the speculative optimizations engines layer on top of it, and the internals of the DOM — including shadow DOM, custom elements, and mutation observers.

*See also:* _Browser Architecture_ (where parsing sits in a navigation), _CSS and Layout_ (how the DOM and CSSOM combine), _Event Loop and Scheduling_ (when parser-yielded tasks run).

== Why HTML Parsing Is Special

Before HTML5 (spec work 2004–2008, finalized 2014), error handling was unspecified, and real pages — over 90% of which were malformed by the strict SGML rules — rendered differently across browsers. The WHATWG reverse-engineered dominant behavior into a deterministic state machine. Two properties make HTML parsing unusual:

- *It is not context-free.* `<script>` content, raw text elements, and foster parenting depend on parser state, so you cannot parse HTML with a grammar-based generated parser.
- *It is reentrant.* `document.write()` inside a script inserts bytes into the very stream being parsed. The parser must pause at `</script>`, execute the script, and resume on a possibly modified input stream.

== The Two-Stage Pipeline

=== Tokenization

The tokenizer is a state machine with about 80 states (`Data`, `TagOpen`, `TagName`, `AttributeName`, `CharacterReference`, ...). It consumes the decoded character stream and emits tokens: start tags (with attributes), end tags, character runs, comments, DOCTYPE, end-of-file. Character encoding itself is sniffed: BOM, then `Content-Type` header, then a `<meta charset>` prescan of the first 1024 bytes, then heuristics — a wrong late guess forces a full reparse.

The tree construction stage can switch the tokenizer's state: seeing `<script>` puts the tokenizer into `ScriptData` mode where `<` is mostly literal text. This feedback loop is why the two stages are coupled.

=== Tree Construction

Tree construction consumes tokens under an *insertion mode* (`in head`, `in body`, `in table`, `in cell`, ... — 23 modes) and maintains:

- The *stack of open elements*: ancestors of the current insertion point. Seeing `</p>` pops to the nearest `p`, implicitly closing children.
- The *list of active formatting elements*: enables the *adoption agency algorithm*, which repairs misnested formatting like `<b>one<i>two</b>three</i>` by cloning the `i` element so both trees are well-formed.
- *Foster parenting*: character data or unexpected elements inside `<table>` (but outside cells) are relocated *before* the table, a Netscape-era behavior now mandated.

Error recovery is thus not an afterthought: it is most of the algorithm. The spec's `in body` insertion mode alone covers dozens of "unexpected token" cases, each with a precise rule.

=== Scripts Block Parsing

A classic `<script src>` halts tree construction until the script downloads and executes, because the script may `document.write()` or query the partially-built DOM. Modern attributes relax this:

#table(
  columns: 3,
  [*Attribute*], [*Fetch*], [*Execute*],
  [(none)], [blocking], [immediately, in order],
  [`defer`], [parallel], [after parsing, in order, before `DOMContentLoaded`],
  [`async`], [parallel], [as soon as ready, any order],
  [`type="module"`], [parallel (deferred)], [after parsing, with dependency graph],
)

== Speculative Parsing

Engines refuse to let a blocking script stall resource discovery. While the main parser waits, the *preload scanner* (WebKit/Blink's `HTMLPreloadScanner`, Firefox's speculative parser) tokenizes ahead on the raw input, ignoring tree construction, and issues *speculative fetches* for `src`, `href`, `srcset`, and preloadable resources it sees. Measurements at Google attributed roughly a 20% average load-time improvement to the preload scanner; it is the single reason "put scripts at the bottom" stopped being critical advice.

Blink goes further with *background HTML parsing*: tokenization runs on a separate thread, shipping token batches to the main thread for tree construction. Speculation can fail — a `document.write()` that injects markup invalidates the lookahead, forcing a re-tokenize from the write point — which is one of several reasons `document.write` is effectively deprecated (Chrome ignores writes that inject blocking scripts on slow 2G connections).

== DOM Internals

The DOM is a tree of C++ objects (in Blink: `Node`, `Element`, `Text`, with `Document` as the root) exposed to JavaScript through generated *bindings*. Internals worth knowing:

- *Wrappers are lazy.* A DOM node gets a JavaScript wrapper object only when script first touches it; a million-node document with no script costs no JS heap. The unified V8/Oilpan heap keeps wrapper and node alive or dead together.
- *Live collections.* `getElementsByTagName` returns a live `HTMLCollection` backed by a cached, invalidation-tracked node list; `querySelectorAll` returns a static `NodeList`. Iterating a live collection while mutating the tree is a classic accidental quadratic.
- *Attributes vs. properties.* Attributes are string data in the element; properties live on the wrapper and *reflect* attributes through spec-defined parsing (e.g. `input.maxLength` parses, clamps, and defaults).
- *Tree mutation cost.* `appendChild` is cheap; what costs is the invalidation it triggers — style, layout, and live-collection caches — paid later at the next style/layout flush.

== Shadow DOM

Shadow DOM gives an element a private subtree with *encapsulation*: outside CSS selectors do not match inside, and inside styles do not leak out. Core API (v1, shipped everywhere by 2018, including Firefox 63):

```js
const root = host.attachShadow({ mode: "open" });
root.innerHTML = "<style>p { color: red }</style><slot></slot>";
```

- *Slots* perform composition: light-DOM children of the host are *distributed* into `<slot>` elements, producing a *flat tree* that style and layout operate on — the rendered tree is neither the light tree nor the shadow tree alone.
- *Event retargeting*: events crossing a shadow boundary have their `target` rewritten to the host, preserving encapsulation; `composedPath()` reveals the full path for open roots.
- *Styling hooks*: `:host`, `:host()`, `::slotted()`, CSS custom properties (which inherit through boundaries), and `::part()` for explicitly exported internals.
- *Declarative shadow DOM* (`<template shadowrootmode="open">`, all engines by 2024) allows server-rendered shadow trees without JavaScript, fixing the SSR story for web components.

Browsers themselves use shadow DOM ("user-agent shadow roots") to implement `<input type=range>`, `<video>` controls, and `<details>`.

== Custom Elements

Custom elements let authors define new tags with lifecycle hooks:

```js
class XCounter extends HTMLElement {
  static observedAttributes = ["value"];
  connectedCallback() { /* inserted into a document */ }
  disconnectedCallback() { /* removed */ }
  attributeChangedCallback(name, oldV, newV) { /* observed attr changed */ }
}
customElements.define("x-counter", XCounter);
```

The parser may meet `<x-counter>` before its definition loads; the element is created as an "undefined" element and *upgraded* in place when `define()` runs — the `:defined` pseudo-class styles the gap. Lifecycle callbacks are not dispatched synchronously mid-parse; they queue in a *custom element reaction queue* drained at well-defined points, keeping tree construction reentrancy-safe. *Customized built-ins* (`is="x-button"` extending `HTMLButtonElement`) shipped in Blink and Gecko but WebKit never implemented them, leaving autonomous elements as the portable form.

== Mutation Observers

`MutationObserver` (2012) replaced the synchronous, pathologically slow DOM Mutation Events. Observers record mutations (childList, attributes, characterData, with optional subtree scope and old values) and deliver them *asynchronously as a microtask*: all mutations from the current script run are batched into one callback invocation, so a thousand `appendChild` calls trigger one delivery rather than a thousand events.

```js
new MutationObserver(records => render(records))
  .observe(target, { childList: true, subtree: true });
```

This microtask timing means observers see the final state of a synchronous batch — ideal for frameworks and sanitizers — but cannot veto or interleave with mutations. For element visibility and size, the analogous batched observers are `IntersectionObserver` and `ResizeObserver` (the latter delivered at rendering time, between layout and paint).

== Further Reading

- WHATWG. _HTML Living Standard_, §13 "Parsing HTML documents" (html.spec.whatwg.org).
- Grosskurth, A., & Godfrey, M. (2005). A reference architecture for web browsers. _ICSM_.
- Garsiel, T., & Irish, P. (2011). How browsers work: behind the scenes of modern web browsers. (web.dev)
- Bidelman, E. et al. Shadow DOM v1 and Custom Elements v1 guides (web.dev/articles).

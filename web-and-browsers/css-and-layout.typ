#import "../template.typ": xref

= CSS and Layout <css-and-layout>

Style and layout turn a DOM tree and a pile of stylesheets into a geometry tree: every box's position and size. This is the most algorithmically dense stage of the rendering pipeline, and the one with the sharpest performance cliffs. This chapter covers the CSSOM, selector matching, the cascade, invalidation, the architecture of layout engines, and the cost model authors must internalize.

*See also:* #xref("web-and-browsers", "html-parsing-and-dom", label: "HTML Parsing and the DOM") (the input tree, shadow DOM flat tree), #xref("web-and-browsers", "rendering-pipeline", label: "Rendering Pipeline") (paint and compositing downstream of layout), #xref("web-and-browsers", "web-performance", label: "Web Performance") (CLS and layout-driven metrics).

== The CSSOM

Stylesheets are parsed into the *CSS Object Model*: a tree of `CSSStyleSheet`, `CSSRule`, and declaration blocks. CSS parsing is forgiving by design — an unrecognized declaration or malformed selector invalidates only that declaration or rule, which is how CSS has remained forward-compatible for 30 years. Notable internals:

- Parsing is fast and parallelizable; engines parse stylesheets off the main thread as they stream in.
- Declarations are stored pre-tokenized; computed values are produced later, per element, during style resolution.
- CSS *blocks rendering but not parsing*: the parser continues building DOM while a stylesheet loads, but script that queries style (`getComputedStyle`) and the first paint must wait, since rendering with incomplete style would flash unstyled content.
- *Constructable stylesheets* (`new CSSStyleSheet()`, `adoptedStyleSheets`) let many shadow roots share one parsed sheet instead of duplicating `<style>` text per component.

== Selector Matching

Style resolution asks, for each element: which rules match? Naively this is (elements × selectors), so engines optimize aggressively.

=== Right-to-Left Matching

Selectors are matched *right to left*. For `.sidebar article p`, the engine starts from a candidate element and checks the rightmost *key selector* (`p`) first; only if it matches does it walk ancestors looking for `article` then `.sidebar`. Most elements fail the key selector immediately, so the expensive ancestor walk rarely runs. This is why a long descendant chain ending in `div` (matching almost everything) is far worse than one ending in a rare class.

=== Rule Hashes and Bloom Filters

Engines bucket rules by key selector into hash maps — by id, class, tag name, and attribute — so an element only tests rules whose key could match it. For the ancestor walk, WebKit introduced (2011) and all engines adopted an *ancestor bloom filter*: as style recalculation descends the DOM, it pushes hashes of each ancestor's tag, id, and classes into a counting bloom filter. Before walking ancestors for a descendant selector, the engine probes the filter; a miss proves no ancestor can match and skips the walk entirely. False positives merely cost a redundant walk, never a wrong result.

Servo's Stylo (in Firefox since 2017) parallelizes the whole traversal: style resolution is embarrassingly parallel across siblings, so a work-stealing pool resolves style for independent subtrees concurrently, with a *rule tree* sharing the matched-rule lists and a style sharing cache reusing computed styles between similar siblings (cousin sharing). Blink similarly caches via the *MatchedPropertiesCache*.

== Specificity and the Cascade

For each property of each element, the *cascade* picks one winning declaration, ordered by:

+ Origin and importance: transition > user-agent `!important` > user `!important` > author `!important` > animation > author normal > user normal > user-agent normal.
+ *Cascade layers* (`@layer`, 2022): later layers beat earlier ones; unlayered styles beat all layers.
+ *Specificity*: a triple $(a, b, c)$ — ids, then (classes, attributes, pseudo-classes), then (type selectors, pseudo-elements) — compared lexicographically, never carrying over (one id beats any number of classes). `:where()` contributes zero; `:is()` and `:not()` take the specificity of their most specific argument.
+ Source order: last declaration wins.

Then *defaulting* fills unset properties via inheritance (for inherited properties like `color`, `font-*`) or initial values, producing the *computed style*; layout later resolves it to *used* values (e.g. `width: auto` becomes a pixel count).

== Style Invalidation

Recomputing style for the whole tree on every mutation would be ruinous, so engines invalidate narrowly. Blink compiles selectors into *invalidation sets*: for the rule `.theme-dark .card`, adding class `theme-dark` to an element schedules invalidation only of descendants matching `.card`, not the whole subtree. Sibling combinators produce sibling invalidation sets; `:has()` (shipped 2022–2023) required new upward invalidation machinery and careful bloom-filter-style fast rejects, which is why it took a decade to ship.

`:hover` on a deep ancestor, attribute selectors on frequently-mutated attributes, and `*` rules defeat these optimizations — the classic causes of long "Recalculate Style" entries in traces.

== Layout Engines

Layout assigns each box its geometry by the rules of the active *formatting context*: block, inline, table, flex, grid. Two architectural generations:

=== Tree-Mutating Layout (legacy)

Older engines (pre-2019 Blink, current WebKit) store geometry as mutable fields on the layout tree (`RenderObject`/`LayoutObject`). Layout walks the tree and writes positions in place. Problems: dirty-bit complexity, under- and over-invalidation bugs, no memoization of subtree results, and infamous edge cases where layout depended on the previous layout's state.

=== LayoutNG and Fragment Trees

Blink's *LayoutNG* (rolled out 2019–2021) makes layout (mostly) functional: layout takes immutable inputs — a node plus *constraint space* (available size, percentage bases) — and produces an immutable *fragment tree* as output. The same node can produce multiple fragments (pagination, multicolumn). Benefits: results are cacheable keyed by constraints, re-layout of unchanged subtrees is a cache hit, and fragmentation, which the old engine never handled correctly, falls out of the model. *Taffy*, the Rust layout library used by Dioxus, Bevy UI, and Zed (descended from Stretch/Yoga-style trees), takes the same pure-function approach for flexbox and grid outside browsers, as does Servo's layout.

=== Flexbox Sketch

Flex layout in one paragraph: lay out each item to find its *flex base size* and hypothetical main size; sum them; distribute free space proportionally to `flex-grow` (or shrinkage weighted by `flex-shrink` × base size), iterating because min/max constraints can freeze items and re-distribute the remainder; then resolve cross sizes, align items (`align-items`, stretching as needed), and align lines. The iteration-until-fixed-point step is why flex items sometimes need two layout passes and why `flex-basis: 0` vs `auto` changes distribution semantics.

=== Grid Sketch

Grid layout places items into an explicit/implicit track matrix (auto-placement is a deterministic cursor algorithm), then sizes tracks by the spec's *track sizing algorithm*: resolve fixed and content-based minimums (min-content/max-content contributions, spanning items distributed across their tracks), then expand flexible `fr` tracks to fill remaining space proportionally. Track sizing can require measuring items, and item layout depends on track sizes, so grid — like flex — interleaves measure and layout passes.

== The Reflow Cost Model

Layout is synchronous-on-demand: mutations only mark the tree dirty; geometry is recomputed at the next *layout flush*, normally once per frame before paint. But reading a layout-dependent value from script — `offsetWidth`, `getBoundingClientRect()`, `scrollTop` — forces an immediate flush if the tree is dirty. Alternating writes and reads in a loop forces a full layout per iteration: *layout thrashing*, the classic accidental $O(n^2)$.

```js
// Thrashes: each iteration writes (dirty) then reads (forced layout).
for (const el of items) {
  el.style.width = container.offsetWidth / 2 + "px";
}
// Fix: read once, then write; or batch reads and writes in separate phases.
```

Mitigations and cost rules of thumb:

- Layout cost scales with the number of boxes whose geometry can change; `contain: layout` / `contain: strict` and `content-visibility: auto` (Chrome 85, 2020) fence subtrees so the engine can skip or defer them — `content-visibility` can cut initial rendering work dramatically on long pages.
- Animate `transform` and `opacity`, which skip layout and paint, never `top/left/width/height`.
- Prefer `requestAnimationFrame` write batching; `ResizeObserver` delivers after layout, giving a sanctioned read point.
- CLS (Cumulative Layout Shift) is the user-facing symptom of late layout: reserve space with `aspect-ratio` and explicit dimensions.

== Further Reading

- CSS Working Group. _CSS Cascading and Inheritance Level 5_; _CSS Flexible Box Layout Module Level 1_; _CSS Grid Layout Module Level 2_ (w3.org/TR).
- Ateş, F., & Stockwell, I. (2021). "Inside Blink's LayoutNG" and Blink LayoutNG design docs (chromium.org).
- Meyerovich, L., & Bodík, R. (2010). Fast and parallel webpage layout. _WWW_.
- WebKit blog (2011). "CSS selector performance: the style sharing and bloom filter optimizations."

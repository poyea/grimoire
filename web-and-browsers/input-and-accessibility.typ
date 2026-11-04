#import "../template.typ": xref

= Input, Scrolling, and Accessibility <input-and-accessibility>

A page is not only drawn; it is touched, scrolled, typed into, and read aloud. Input handling decides whether the page *feels* fast: a scroll that stutters or a button that responds 400 ms late makes an application seem broken even if it loads quickly. Accessibility decides whether people using screen readers, keyboards, switch devices, magnification or voice control can use it at all. Both depend on how the browser turns the DOM into something other than pixels: hit-test regions for the compositor and an accessibility tree for assistive technology. This chapter covers the input pipeline and threaded scrolling, event dispatch, interaction latency and how to keep it low, focus and text input, and the accessibility tree, ARIA and how to test with real assistive technology.

*See also:* #xref("web-and-browsers", "rendering-pipeline", label: "Rendering Pipeline") (compositor thread), #xref("web-and-browsers", "event-loop-and-scheduling", label: "Event Loop and Scheduling") (tasks and yielding), #xref("web-and-browsers", "web-performance", label: "Web Performance") (Interaction to Next Paint), #xref("web-and-browsers", "html-parsing-and-dom", label: "HTML Parsing and the DOM")

== The Input Pipeline

An input event travels from the operating system to the browser process, which routes it to the renderer that owns the target frame. Inside the renderer it arrives first at the *compositor thread*, not the main thread, and the compositor decides whether the main thread needs to see it at all.

This matters most for scrolling. The compositor holds the layer tree and the scroll offsets (#xref("web-and-browsers", "rendering-pipeline", label: "Rendering Pipeline")), so it can scroll by itself, moving layers and drawing new frames even while the main thread is busy running JavaScript. This *threaded scrolling* is why a modern browser scrolls smoothly on pages with heavy script. It works only if the compositor knows that no script will cancel the scroll. A `touchstart`, `touchmove` or `wheel` listener could call `preventDefault()` to stop scrolling, so the compositor marks the region under such listeners as *non-fast-scrollable* and must wait for the main thread to run them before scrolling, putting script back on the critical path.

*Passive listeners* remove the wait by promising not to cancel:

```js
// The compositor can scroll immediately; the listener runs later on the main thread.
window.addEventListener("touchmove", trackGesture, { passive: true });
```

Browsers make document-level touch and wheel listeners passive by default for this reason. Where a component really must prevent scrolling (a map or a drawing canvas), the CSS `touch-action` property declares it statically, which the compositor can read without asking the main thread.

== Event Dispatch

DOM events propagate in three phases: *capture* from the window down to the target, the *target* itself, and *bubble* back up. Delegating many handlers to one listener on a common ancestor relies on bubbling.

*Pointer events* unify mouse, touch and pen into one model with `pointerdown`, `pointermove`, `pointerup` and `pointercancel`, plus pressure, tilt and pointer type. They replace separate mouse and touch handling, and `setPointerCapture` keeps delivering a drag's events to the element where it started.

Browsers treat *continuous* and *discrete* events differently. High-frequency events such as `pointermove` can arrive at hundreds of hertz on modern hardware, faster than the display refreshes. Browsers *align* them to animation frames, delivering at most one per frame just before `requestAnimationFrame` callbacks, with the intermediate samples available through `getCoalescedEvents()` for drawing applications that need every point, and `getPredictedEvents()` for latency-hiding prediction. Discrete events (`pointerdown`, `click`, `keydown`) are delivered immediately.

== Interaction Latency

*Interaction to Next Paint* (INP), a Core Web Vital since 2024, measures responsiveness: for each click, tap or key press, the time from the input until the next frame is presented, reporting roughly the worst interaction of the visit. Good is at most 200 ms. Each interaction has three parts:

1. *Input delay*: the event waits because the main thread is busy with another task (often a long script task, a timer, or hydration of a server-rendered page).
2. *Processing time*: the event handlers run.
3. *Presentation delay*: style, layout, paint and compositing of the resulting update.

The cure for input delay and long processing is the same: break long tasks so that the event loop can respond between pieces (#xref("web-and-browsers", "event-loop-and-scheduling", label: "Event Loop and Scheduling")). A handler should do the minimum needed for the visual response, then yield before the rest:

```js
button.addEventListener("click", async () => {
  // 1. Immediate visual feedback, painted in the next frame.
  button.setAttribute("aria-busy", "true");
  spinner.hidden = false;

  // 2. Yield so the browser can render that feedback before heavy work.
  await scheduler.yield();

  // 3. The expensive part, itself split into chunks that yield.
  for (const chunk of partition(items, 200)) {
    processChunk(chunk);
    await scheduler.yield();
  }
  spinner.hidden = true;
  button.removeAttribute("aria-busy");
});
```

`scheduler.yield()` (with a fallback to `setTimeout(resolve, 0)` where unsupported) puts the continuation at the front of the task queue, so yielding does not mean losing one's place behind unrelated work. Presentation delay is reduced by keeping DOM updates small: avoiding layout thrashing, using `content-visibility: auto` to skip rendering off-screen content, and not re-rendering large component trees for small state changes.

== Focus and Keyboard Interaction

Every interactive control must be reachable and operable with the keyboard, both for accessibility and for power users:

- Native elements (`<button>`, `<a href>`, `<input>`, `<select>`, `<details>`) are focusable and keyboard-operable by default. A `<div>` with a click handler is neither.
- `tabindex="0"` adds an element to the tab order in document order; `tabindex="-1"` makes it focusable by script only. Positive `tabindex` values override document order and almost always produce a confusing order.
- The `:focus-visible` pseudo-class shows a focus ring for keyboard users without showing it on mouse clicks. Removing focus outlines without a replacement makes a site unusable by keyboard.
- Modal dialogs must trap focus inside while open and restore it afterwards. The native `<dialog>` element with `showModal()` does both, and makes the rest of the page *inert*. The `inert` attribute makes any subtree non-focusable and hidden from assistive technology.

Text input is subtler than it looks. Languages such as Chinese, Japanese and Korean use an *input method editor* (IME), which composes characters over several keystrokes; `compositionstart`, `compositionupdate` and `compositionend` events bracket the composition, and handlers that act on every `keydown` or `input` event during composition (for example submitting on Enter) break those users. Custom text editors are among the hardest components on the web for this reason, and most build on `contenteditable` or, increasingly, on the `EditContext` API.

== The Accessibility Tree

Assistive technologies (screen readers such as NVDA, JAWS, VoiceOver and TalkBack, as well as voice control and switch access) do not read the DOM. The browser builds an *accessibility tree* from the DOM and computed styles and exposes it through each operating system's accessibility API: UI Automation and IAccessible2 on Windows, NSAccessibility on macOS, AT-SPI on Linux, and the platform APIs on Android and iOS. Each node has:

- a *role* (button, link, heading, checkbox, dialog, list),
- an *accessible name* (computed from content, `<label>`, `alt`, `aria-label` or `aria-labelledby`, by the accessible name computation algorithm),
- *states and properties* (checked, expanded, disabled, level, value),
- relationships (labelled by, controls, owns).

Styles affect the tree: `display: none` and `visibility: hidden` remove content from it, but content hidden only visually (positioned off screen, or clipped with a "visually hidden" class) remains, which is how text intended only for screen-reader users is provided. Building and updating the accessibility tree costs CPU time, which browsers pay only when assistive technology is active; very large DOMs therefore hurt assistive technology users disproportionately.

=== Semantic HTML First, ARIA Second

*ARIA* (Accessible Rich Internet Applications) attributes override and extend roles, names and states for custom widgets. The first rule of ARIA is not to use it when a native element exists: a `<button>` has the right role, keyboard behaviour, focusability and form participation built in, while `<div role="button">` gets only the role, and every other behaviour must be re-implemented by hand. Incorrect ARIA is worse than none, because it tells assistive technology something false. Surveys of popular home pages consistently find more detectable accessibility errors on pages that use ARIA than on pages that do not.

When a custom widget is necessary, the ARIA Authoring Practices Guide gives the expected roles, states and keyboard interactions. A disclosure (show/hide) control is the simplest example:

```html
<button type="button" aria-expanded="false" aria-controls="shipping-details">
  Shipping details
</button>
<div id="shipping-details" hidden>
  Orders ship within two business days.
</div>
<script>
  const toggle = document.querySelector("[aria-controls=shipping-details]");
  const panel = document.getElementById("shipping-details");
  toggle.addEventListener("click", () => {
    const open = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", String(!open));
    panel.hidden = open;
  });
</script>
```

Because it is a real `<button>`, Enter and Space already work, and the screen reader announces "Shipping details, button, collapsed", then "expanded" after activation. (The native `<details>` and `<summary>` elements provide the same behaviour with no script.)

*Live regions* (`aria-live="polite"` or `"assertive"`, or roles such as `status` and `alert`) announce dynamic changes such as "Saved" or form errors without moving focus. The region must exist in the DOM before its content changes, or many screen readers miss the update.

== Beyond Screen Readers

Accessibility covers far more than screen readers:

- *Zoom and reflow*: content must work at 400 percent zoom, equivalent to a 320 CSS pixel wide viewport, without horizontal scrolling for ordinary text (WCAG 2.2 success criterion 1.4.10).
- *Colour and contrast*: text needs a contrast ratio of at least 4.5:1 against its background (3:1 for large text), and information must not be conveyed by colour alone.
- *User preferences* exposed as media queries: `prefers-reduced-motion` (disable parallax and large animations, which can trigger vestibular disorders), `prefers-contrast`, and `forced-colors` (Windows high-contrast themes, where the browser replaces author colours and custom widgets drawn with backgrounds can disappear).
- *Target size*: interactive targets of at least 24 by 24 CSS pixels (WCAG 2.2) help users with tremors and everyone on a phone.

== Standards and Testing

The *Web Content Accessibility Guidelines* (WCAG 2.2, 2023) define testable success criteria at levels A, AA and AAA; level AA is the usual legal and contractual target. Accessibility is increasingly a legal requirement: the European Accessibility Act has applied to many consumer products and services since June 2025, and accessibility lawsuits under the Americans with Disabilities Act are common in the United States.

Testing works in layers:

- *Automated checkers* (axe-core, Lighthouse, browser DevTools) run in CI and catch missing names, contrast failures and invalid ARIA, but only an estimated third or so of WCAG issues.
- *Keyboard-only testing*: unplug the mouse and complete every task. It finds focus traps, invisible focus and unreachable controls in minutes.
- *Screen reader testing* with the common pairings (NVDA with Firefox or Chrome, JAWS with Chrome, VoiceOver with Safari, TalkBack with Chrome), because browsers and screen readers differ in what they support.
- *Testing with disabled users*, which finds problems that no checklist anticipates.

== Further Reading

W3C (2023). "Web Content Accessibility Guidelines (WCAG) 2.2." W3C Recommendation specification.

W3C (2023). "Accessible Rich Internet Applications (WAI-ARIA) 1.2." W3C Recommendation specification.

W3C (2024). "ARIA Authoring Practices Guide." W3C WAI documentation.

W3C (2024). "Accessible Name and Description Computation 1.2." W3C Working Draft specification.

W3C (2024). "Pointer Events Level 3." W3C Working Draft specification.

W3C (2024). "UI Events." W3C Working Draft specification (composition events).

WebAIM (2024). "The WebAIM Million: An Accessibility Analysis of the Top 1,000,000 Home Pages." WebAIM report.

Wagner, J., Pollard, B. (2024). "Interaction to Next Paint (INP)." web.dev documentation.

// template.typ

// Styling for the HTML export only. Kept deliberately small: readable
// measure, a legible mono stack for the 900-odd code blocks, and table
// rules, since the raw export ships with no styling beyond the MathML
// alignment rules Typst inlines itself.
#let _html-css = "
:root { color-scheme: light dark; }
body {
  font: 16px/1.65 Georgia, 'Times New Roman', serif;
  max-width: 46rem; margin: 0 auto; padding: 0 1.1rem 4rem;
}
/* Sticky, because a volume is one long page: coding.html alone is 2.3 MB,
   so the way back to the index should never scroll away. */
header { position: sticky; top: 0; z-index: 10;
  background: Canvas; margin-bottom: 2rem; }
header nav { font-family: system-ui, sans-serif; font-size: .85rem;
  padding: .9rem 0; border-bottom: 1px solid #8883; }
/* Keep anchor targets clear of the sticky header when the TOC jumps. */
h1, h2, h3, h4, h5, h6 { scroll-margin-top: 3.6rem; }
header nav .sep { opacity: .45; margin: 0 .45em; }
header nav .vol { font-weight: 600; }
h1, h2, h3, h4 { font-family: system-ui, sans-serif; line-height: 1.25;
  margin: 2.2em 0 .6em; }
h3 { font-size: 1.2rem; } h4 { font-size: 1.03rem; }
/* Chapters are h2. In the PDF each starts on a fresh page; HTML export
   drops those 57 pagebreaks, so without a rule the previous chapter's
   Further Reading runs straight into the next chapter's title. */
h2 { font-size: 1.6rem; margin-top: 3.5rem; padding-top: 2rem;
  border-top: 2px solid #8884; }
h2:first-of-type { margin-top: 1rem; padding-top: 0; border-top: 0; }
pre { font: 13px/1.5 ui-monospace, 'DejaVu Sans Mono', monospace;
  background: #8881; padding: .8rem 1rem; border-radius: 4px;
  overflow-x: auto; }
code { font-family: ui-monospace, 'DejaVu Sans Mono', monospace;
  font-size: .92em; }
table { border-collapse: collapse; width: 100%; display: block;
  overflow-x: auto; font-size: .93rem; margin: 1.2em 0; }
td, th { border: 1px solid #8884; padding: .35rem .6rem; text-align: left; }
a { color: inherit; text-underline-offset: 2px; }
footer { margin-top: 4rem; padding-top: 1rem; border-top: 1px solid #8883;
  font-family: system-ui, sans-serif; font-size: .82rem; opacity: .75; }
math { font-size: 1.02em; }
"

#let project(title, body) = {
  set document(title: title, author: "John Law")
  set page(
    paper: "us-letter",
    margin: (x: 1.5cm, y: 1.5cm),
    header: [
      #smallcaps[_#title Notes by #link("https://github.com/poyea")[\@poyea]_]
      #h(1fr)
      #emph(text[#datetime.today().display()])
      #h(1fr)
      #emph(link("https://github.com/poyea/grimoire")[poyea/grimoire])
    ],
    footer: context align(right)[#counter(page).display("1")]
  )
  set text(font: "New Computer Modern", size: 11pt)
  set heading(numbering: "1.")

  // HTML export gets a page chrome of its own: the `set page` rule above
  // is paged-only and Typst warns that it ignored it. A `<style>` emitted
  // here lands in the body, which browsers honour -- wrapping it in
  // `html.head` would nest a second <head> inside <body>, since Typst
  // generates the real one itself.
  context if target() == "html" {
    html.style(_html-css)
    html.header[
      #html.nav[
        #html.a(href: "./index.html")[Grimoire]
        #html.span(class: "sep")[/]
        #html.span(class: "vol")[#title]
      ]
    ]
  }

  // Document Body
  body

  context if target() == "html" {
    html.footer[
      #html.p[
        #title -- notes by
        #html.a(href: "https://github.com/poyea")[\@poyea].
        Rendered from the Typst sources; the
        #html.a(href: "https://github.com/poyea/grimoire/releases")[released PDFs]
        remain the reference typesetting.
        #html.a(href: "https://github.com/poyea/grimoire")[poyea/grimoire]
      ]
    ]
  }
}

// -----------------------------------------------------------------------------
// Theorem-like environments
//
// All counters are scoped per kind so that, e.g., theorems and lemmas number
// independently. Each helper takes an optional `name` (the parenthetical
// label) and a body. Usage:
//
//   #theorem(name: "Kleene")[Every regular language ...]
//   #proof[By induction on ...]
//
// These are opt-in: existing chapters that do not call them are unaffected.
// -----------------------------------------------------------------------------

#let _thm-counter = counter("grimoire-theorem")
#let _lem-counter = counter("grimoire-lemma")
#let _def-counter = counter("grimoire-definition")
#let _prop-counter = counter("grimoire-proposition")
#let _cor-counter = counter("grimoire-corollary")
#let _ex-counter = counter("grimoire-example")

#let _boxed(kind, counter, body, name: none, italic: true) = {
  counter.step()
  let header = context {
    let n = counter.display()
    strong[#kind #n#if name != none [ (#name)].]
  }
  block(
    breakable: true,
    above: 0.8em,
    below: 0.8em,
    [#header #h(0.4em) #if italic { emph(body) } else { body }],
  )
}

#let theorem(body, name: none) = _boxed("Theorem", _thm-counter, body, name: name)
#let lemma(body, name: none) = _boxed("Lemma", _lem-counter, body, name: name)
#let proposition(body, name: none) = _boxed("Proposition", _prop-counter, body, name: name)
#let corollary(body, name: none) = _boxed("Corollary", _cor-counter, body, name: name)
#let definition(body, name: none) = _boxed("Definition", _def-counter, body, name: name, italic: false)
#let example(body, name: none) = _boxed("Example", _ex-counter, body, name: name, italic: false)

#let proof(body) = block(
  breakable: true,
  above: 0.6em,
  below: 0.8em,
  [#emph[Proof.] #h(0.4em) #body #h(1fr) $square.stroked$],
)

// -----------------------------------------------------------------------------
// Terminal / shell block
// -----------------------------------------------------------------------------

#let terminal(body) = block(
  fill: luma(240),
  inset: 8pt,
  radius: 3pt,
  width: 100%,
  breakable: true,
  text(font: "DejaVu Sans Mono", size: 9pt, body),
)

// -----------------------------------------------------------------------------
// Overbar / underbar
//
// MathML Core has no primitive for a rule drawn over or under an
// expression, so Typst's HTML export drops it and emits only the base:
// see typst/typst, crates/typst-html/src/mathml.rs, where MathKind::Line
// is routed to ignored_math_item(). That silently turns $overline(L)$
// (language complement) into a plain L, and the active-low $overline("CS")$
// into CS -- a different claim, not just plainer typography.
//
// These render through accents, which do export (as <mover>/<munder>),
// when the target is HTML, and keep the true full-width rule in the PDF.
// -----------------------------------------------------------------------------

#let overbar(body) = context {
  if target() == "html" { math.macron(body) } else { math.overline(body) }
}

#let underbar(body) = context {
  if target() == "html" {
    math.attach(math.limits(body), b: sym.macron)
  } else {
    math.underline(body)
  }
}

// -----------------------------------------------------------------------------
// RFC reference
//
// #rfc(9293) renders "RFC 9293" as a link to the canonical RFC Editor
// page, so a reader can follow the citation instead of retyping the
// number. Ranges and slash pairs (RFC 7230-7235, RFC 5389/8489) stay as
// plain text, since they name more than one document.
// -----------------------------------------------------------------------------

#let rfc(number) = link(
  "https://www.rfc-editor.org/rfc/rfc" + str(number) + ".html",
)[RFC #number]

// -----------------------------------------------------------------------------
// Cross-reference helper
//
// #xref("database", "partitioning-and-elasticity") renders as a styled
// in-text reference to another chapter. Resolves to a GitHub link so the
// PDF stays clickable.
//
// Pass `label` to keep the prose reading naturally while still linking:
//
//   #xref("web-and-browsers", "css-and-layout", label: "CSS and Layout")
//
// renders as _CSS and Layout_ rather than the raw path.
// -----------------------------------------------------------------------------

#let xref(subject, slug, label: none) = {
  let url = "https://github.com/poyea/grimoire/blob/main/" + subject + "/" + slug + ".typ"
  if label == none {
    emph(link(url)[#subject\/#slug])
  } else {
    emph(link(url)[#label])
  }
}

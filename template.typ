// template.typ
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

  // Document Body
  body
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
// Cross-reference helper
//
// #xref("database", "partitioning-and-elasticity") renders as a styled
// in-text reference to another chapter. Resolves to a GitHub link so the
// PDF stays clickable.
// -----------------------------------------------------------------------------

#let xref(subject, slug) = {
  let url = "https://github.com/poyea/grimoire/blob/main/" + subject + "/" + slug + ".typ"
  emph(link(url)[#subject\/#slug])
}

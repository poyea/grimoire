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

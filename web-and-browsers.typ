#import "template.typ": project

#project("Web and Browser Internals")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Web and Browser Internals]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "web-and-browsers/browser-architecture.typ"
  #pagebreak()

  #include "web-and-browsers/html-parsing-and-dom.typ"
  #pagebreak()

  #include "web-and-browsers/css-and-layout.typ"
  #pagebreak()

  #include "web-and-browsers/rendering-pipeline.typ"
  #pagebreak()

  #include "web-and-browsers/javascript-engines.typ"
  #pagebreak()

  #include "web-and-browsers/event-loop-and-scheduling.typ"
  #pagebreak()

  #include "web-and-browsers/webassembly.typ"
  #pagebreak()

  #include "web-and-browsers/web-performance.typ"
  #pagebreak()

]

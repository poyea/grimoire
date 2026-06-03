#import "template.typ": project

#project("Embedded And Realtime")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Embedded And Realtime]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "embedded-and-realtime/mcus-and-soc.typ"
  #pagebreak()

  #include "embedded-and-realtime/rtos.typ"
  #pagebreak()

]

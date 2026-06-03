#import "template.typ": project

#project("Data Engineering")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Data Engineering]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "data-engineering/batch-processing.typ"
  #pagebreak()

  #include "data-engineering/etl-vs-elt.typ"
  #pagebreak()

  #include "data-engineering/lakehouse-engineering.typ"
  #pagebreak()

  #include "data-engineering/orchestration.typ"
  #pagebreak()

  #include "data-engineering/streaming.typ"
  #pagebreak()

]

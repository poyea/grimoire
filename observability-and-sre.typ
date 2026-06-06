#import "template.typ": project

#project("Observability And Sre")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Observability And Sre]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "observability-and-sre/continuous-profiling.typ"
  #pagebreak()

  #include "observability-and-sre/distributed-tracing.typ"
  #pagebreak()

  #include "observability-and-sre/metrics-systems.typ"
  #pagebreak()

  #include "observability-and-sre/slo-engineering.typ"
  #pagebreak()

  #include "observability-and-sre/the-three-pillars-and-beyond.typ"
  #pagebreak()

]

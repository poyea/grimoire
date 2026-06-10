#import "template.typ": project

#project("Performance Engineering")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Performance Engineering]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "performance-engineering/methodology.typ"
  #pagebreak()

  #include "performance-engineering/benchmarking.typ"
  #pagebreak()

  #include "performance-engineering/cpu-profiling.typ"
  #pagebreak()

  #include "performance-engineering/memory-performance.typ"
  #pagebreak()

  #include "performance-engineering/concurrency-performance.typ"
  #pagebreak()

  #include "performance-engineering/io-performance.typ"
  #pagebreak()

  #include "performance-engineering/queueing-theory.typ"
  #pagebreak()

  #include "performance-engineering/capacity-planning.typ"
  #pagebreak()

]

#import "template.typ": project

#project("Software Architecture")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Software Architecture]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "software-architecture/architectural-styles.typ"
  #pagebreak()

  #include "software-architecture/domain-driven-design.typ"
  #pagebreak()

  #include "software-architecture/monoliths-and-microservices.typ"
  #pagebreak()

  #include "software-architecture/api-design.typ"
  #pagebreak()

  #include "software-architecture/event-driven-architecture.typ"
  #pagebreak()

  #include "software-architecture/resilience-patterns.typ"
  #pagebreak()

  #include "software-architecture/evolutionary-architecture.typ"
  #pagebreak()

  #include "software-architecture/architecture-evaluation.typ"
  #pagebreak()

  #include "software-architecture/distributed-data-patterns.typ"
  #pagebreak()

]

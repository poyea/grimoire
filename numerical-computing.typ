#import "template.typ": project

#project("Numerical Computing")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Numerical Computing]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "numerical-computing/floating-point.typ"
  #pagebreak()

  #include "numerical-computing/error-analysis.typ"
  #pagebreak()

  #include "numerical-computing/linear-systems.typ"
  #pagebreak()

  #include "numerical-computing/iterative-methods.typ"
  #pagebreak()

  #include "numerical-computing/eigenvalue-problems.typ"
  #pagebreak()

  #include "numerical-computing/fft.typ"
  #pagebreak()

  #include "numerical-computing/ode-integration.typ"
  #pagebreak()

  #include "numerical-computing/optimization-algorithms.typ"
  #pagebreak()

]

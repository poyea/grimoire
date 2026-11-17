#import "template.typ": project

#project("Formal Methods")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Formal Methods]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "formal-methods/notation.typ"
  #pagebreak()

  #include "formal-methods/abstract-interpretation.typ"
  #pagebreak()

  #include "formal-methods/alloy.typ"
  #pagebreak()

  #include "formal-methods/deductive-verification.typ"
  #pagebreak()

  #include "formal-methods/model-checking.typ"
  #pagebreak()

  #include "formal-methods/propositional-and-fol.typ"
  #pagebreak()

  #include "formal-methods/sat-and-smt.typ"
  #pagebreak()

  #include "formal-methods/separation-logic.typ"
  #pagebreak()

  #include "formal-methods/symbolic-execution-and-fuzzing.typ"
  #pagebreak()

  #include "formal-methods/theorem-proving.typ"
  #pagebreak()

  #include "formal-methods/tla-plus.typ"
  #pagebreak()

  #include "formal-methods/verified-distributed-systems.typ"
  #pagebreak()

]

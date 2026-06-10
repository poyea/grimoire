#import "template.typ": project

#project("Quantum Computing")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Quantum Computing]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "quantum-computing/notation.typ"
  #pagebreak()

  #include "quantum-computing/error-correction.typ"
  #pagebreak()

  #include "quantum-computing/hardware-architectures.typ"
  #pagebreak()

  #include "quantum-computing/quantum-algorithms.typ"
  #pagebreak()

  #include "quantum-computing/nisq-and-benchmarking.typ"
  #pagebreak()

  #include "quantum-computing/qubits-and-gates.typ"
  #pagebreak()

]

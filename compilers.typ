#import "template.typ": project

#project("Compilers")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Compilers]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "compilers/dataflow-analysis.typ"
  #pagebreak()

  #include "compilers/frontends.typ"
  #pagebreak()

  #include "compilers/ir-design.typ"
  #pagebreak()

  #include "compilers/register-allocation.typ"
  #pagebreak()

  #include "compilers/backend-and-codegen.typ"
  #pagebreak()

  #include "compilers/optimisation-passes.typ"
  #pagebreak()

  #include "compilers/jit-and-runtimes.typ"
  #pagebreak()

]

#import "template.typ": project

#project("Cloud And Infrastructure")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Cloud And Infrastructure]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "cloud-and-infrastructure/containers.typ"
  #pagebreak()

  #include "cloud-and-infrastructure/iaas-fundamentals.typ"
  #pagebreak()

  #include "cloud-and-infrastructure/kubernetes-internals.typ"
  #pagebreak()

]

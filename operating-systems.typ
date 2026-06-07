#import "template.typ": project

#project("Operating Systems")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Operating Systems]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "operating-systems/ipc-mechanisms.typ"
  #pagebreak()

  #include "operating-systems/memory-management.typ"
  #pagebreak()

  #include "operating-systems/processes-and-threads.typ"
  #pagebreak()

  #include "operating-systems/scheduling-theory.typ"
  #pagebreak()

  #include "operating-systems/storage-stack.typ"
  #pagebreak()

  #include "operating-systems/boot-and-init.typ"
  #pagebreak()

  #include "operating-systems/security-models.typ"
  #pagebreak()

]

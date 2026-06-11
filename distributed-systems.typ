#import "template.typ": project

#project("Distributed Systems")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Distributed Systems]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "distributed-systems/introduction.typ"
  #pagebreak()

  #include "distributed-systems/time-and-order.typ"
  #pagebreak()

  #include "distributed-systems/failure-detection.typ"
  #pagebreak()

  #include "distributed-systems/leader-election-and-leases.typ"
  #pagebreak()

  #include "distributed-systems/consensus-deep-dive.typ"
  #pagebreak()

  #include "distributed-systems/coordination-services.typ"
  #pagebreak()

  #include "distributed-systems/gossip.typ"
  #pagebreak()

  #include "distributed-systems/causal-consistency.typ"
  #pagebreak()

  #include "distributed-systems/crdts.typ"
  #pagebreak()

  #include "distributed-systems/transactions.typ"
  #pagebreak()

  #include "distributed-systems/log-based-systems.typ"
  #pagebreak()

  #include "distributed-systems/workflow-engines.typ"
  #pagebreak()

]

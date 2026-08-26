#import "template.typ": project

#project("Machine Learning Foundations")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Machine Learning Foundations]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "machine-learning-foundations/notation.typ"
  #pagebreak()

  #include "machine-learning-foundations/mathematics-for-younger-self.typ"
  #pagebreak()

  #include "machine-learning-foundations/generalization-theory.typ"
  #pagebreak()

  #include "machine-learning-foundations/linear-algebra-for-ml.typ"
  #pagebreak()

  #include "machine-learning-foundations/optimization.typ"
  #pagebreak()

  #include "machine-learning-foundations/probability-and-information.typ"
  #pagebreak()

  #include "machine-learning-foundations/loss-functions.typ"
  #pagebreak()

  #include "machine-learning-foundations/information-theory.typ"
  #pagebreak()

  #include "machine-learning-foundations/network-information-theory.typ"
  #pagebreak()

  #include "machine-learning-foundations/reinforcement-learning.typ"
  #pagebreak()

  #include "machine-learning-foundations/diffusion-models.typ"
  #pagebreak()

]

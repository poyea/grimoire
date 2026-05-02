#import "template.typ": project

#project("LLM")[
  #align(center)[
  #block(
    fill: luma(245),
    inset: 10pt,
    width: 80%,
    text(size: 9pt)[
      #align(center)[
        #text(size: 24pt, weight: "bold")[Large Language Models: Internals, Training & Serving]
      ]
    ]
  )
]

#outline(
  title: "Table of Contents",
  depth: 2,
)

#emph[Enjoy.]

#pagebreak()

#include "llm/introduction.typ"
#pagebreak()

#include "llm/transformer-architecture.typ"
#pagebreak()

#include "llm/inference-optimization.typ"
#pagebreak()

= Conclusion

More chapters coming soon.
]

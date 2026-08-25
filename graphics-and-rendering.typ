#import "template.typ": project

#project("Graphics and Rendering")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Graphics And Rendering]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "graphics-and-rendering/gi-techniques.typ"
  #pagebreak()

  #include "graphics-and-rendering/physically-based-rendering.typ"
  #pagebreak()

  #include "graphics-and-rendering/rasterization-pipeline.typ"
  #pagebreak()

  #include "graphics-and-rendering/ray-tracing.typ"
  #pagebreak()

  #include "graphics-and-rendering/realtime-engines.typ"
  #pagebreak()

  #include "graphics-and-rendering/shaders.typ"
  #pagebreak()

]

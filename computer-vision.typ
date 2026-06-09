#import "template.typ": project

#project("Computer Vision")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Computer Vision]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "computer-vision/image-formation.typ"
  #pagebreak()

  #include "computer-vision/cnn-architectures.typ"
  #pagebreak()

  #include "computer-vision/object-detection.typ"
  #pagebreak()

  #include "computer-vision/image-segmentation.typ"
  #pagebreak()

  #include "computer-vision/vision-transformers.typ"
  #pagebreak()

  #include "computer-vision/3d-vision.typ"
  #pagebreak()

]

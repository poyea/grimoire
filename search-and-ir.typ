#import "template.typ": project

#project("Search and Information Retrieval")[
  #align(center)[#block(fill: luma(245), inset: 10pt, width: 80%, text(size: 9pt)[
    #align(center)[#text(size: 24pt, weight: "bold")[Search and Information Retrieval]]
  ])]

  #outline(title: "Table of Contents", depth: 2)

  #emph[Enjoy.]

  #pagebreak()

  #include "search-and-ir/inverted-indexes.typ"
  #pagebreak()

  #include "search-and-ir/query-processing.typ"
  #pagebreak()

  #include "search-and-ir/ranking-classical.typ"
  #pagebreak()

  #include "search-and-ir/learning-to-rank.typ"
  #pagebreak()

  #include "search-and-ir/neural-retrieval.typ"
  #pagebreak()

  #include "search-and-ir/vector-search.typ"
  #pagebreak()

  #include "search-and-ir/evaluation.typ"
  #pagebreak()

  #include "search-and-ir/rag-and-search-systems.typ"
  #pagebreak()

]

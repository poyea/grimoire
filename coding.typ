#set document(title: "Coding", author: "John Law")
#set page(
  paper: "us-letter",
  margin: (x: 1cm, y: 1cm),
  header: [
    #smallcaps[_Coding Notes by #link("https://github.com/poyea")[\@poyea]_]
    #h(0.5fr)
    #emph(text[#datetime.today().display()])
    #h(0.5fr)
    #emph(link("https://github.com/poyea/grimoire")[poyea/grimoire])
  ],
)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")

#align(center)[
  #block(
    fill: luma(245),
    inset: 10pt,
    width: 80%,
    text(size: 9pt)[
      #align(center)[
        #text(size: 24pt, weight: "bold")[Coding]
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

#include "coding/arrays.typ"
#pagebreak()

#include "coding/hashing.typ"
#pagebreak()

#include "coding/two-pointers.typ"
#pagebreak()

#include "coding/sliding-window.typ"
#pagebreak()

#include "coding/stack.typ"
#pagebreak()

#include "coding/binary-search.typ"
#pagebreak()

#include "coding/linked-list.typ"
#pagebreak()

#include "coding/trees.typ"
#pagebreak()

#include "coding/heap-priority-queue.typ"
#pagebreak()

#include "coding/backtracking.typ"
#pagebreak()

#include "coding/tries.typ"
#pagebreak()

#include "coding/graphs.typ"
#pagebreak()

#include "coding/dynamic-programming-1d.typ"
#pagebreak()

#include "coding/dynamic-programming-2d.typ"
#pagebreak()

#include "coding/greedy.typ"
#pagebreak()

#include "coding/reference.typ"

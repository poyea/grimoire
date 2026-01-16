#import "template.typ": project

#project("Coding")[
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

// ============================================================================
// CHAPTER 0: Reference & Meta-Knowledge
// ============================================================================

#include "coding/reference.typ"
#pagebreak()

#include "coding/problem-solving.typ"
#pagebreak()

// ============================================================================
// PART I: Foundations - Basic Data Structures & Patterns
// ============================================================================

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

#include "coding/linked-list.typ"
#pagebreak()

// ============================================================================
// PART II: Search & Trees - Algorithmic Techniques
// ============================================================================

#include "coding/binary-search.typ"
#pagebreak()

#include "coding/trees.typ"
#pagebreak()

#include "coding/heap-priority-queue.typ"
#pagebreak()

#include "coding/tries.typ"
#pagebreak()

// ============================================================================
// PART III: Advanced Paradigms - Algorithmic Thinking
// ============================================================================

#include "coding/backtracking.typ"
#pagebreak()

#include "coding/greedy.typ"
#pagebreak()

#include "coding/dynamic-programming.typ"
#pagebreak()

// ============================================================================
// PART IV: Graph Theory
// ============================================================================

#include "coding/graphs.typ"
#pagebreak()

#include "coding/advanced-graphs.typ"
#pagebreak()

#include "coding/union-find.typ"
#pagebreak()

// ============================================================================
// PART V: Specialized Topics
// ============================================================================

#include "coding/bit-manipulation.typ"
#pagebreak()

#include "coding/string-algorithms.typ"
#pagebreak()

#include "coding/math-number-theory.typ"
#pagebreak()

#include "coding/advanced-systems.typ"
#pagebreak()

#include "coding/advanced-java-sections.typ"
]

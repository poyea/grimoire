#set document(title: "Coding", author: "John Law")
#set page(
  paper: "us-letter",
  margin: (x: 1cm, y: 1cm),
  header: [
    #smallcaps[_Coding Notes by #link("https://github.com/poyea")[\@poyea]_]
    #h(0.5fr)
    #emph(text[#datetime.today().display()])
    #h(0.5fr)
    #emph(link("https://github.com/poyea/grimoire")[poyea/grimoire::coding])
  ],
  footer: context [
    #align(right)[#counter(page).display("1")]
  ]
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

// ============================================================================
// CHAPTER 0: Reference & Meta-Knowledge
// ============================================================================

#include "coding/reference.typ"
#pagebreak()

= Problem-Solving Framework

== Pattern Recognition

*Identify the pattern before coding:*

*Two Pointers:*
- Sorted array + $O(n)$ required
- Finding pairs/triplets with specific sum
- Removing duplicates in-place
- Merging sorted sequences

*Sliding Window:*
- Contiguous subarray/substring optimization
- "All subarrays of length k"
- "Longest/shortest substring with constraint"
- Running aggregates (sum, product, min/max)

*Hash Table:*
- $O(1)$ lookup requirement
- Counting frequencies/duplicates
- Anagrams, grouping by property
- Complement search (two sum pattern)

*Stack:*
- Matching pairs (parentheses, tags)
- Monotonic sequences (next greater/smaller element)
- Expression evaluation/parsing
- Depth-first exploration with backtracking

*Heap/Priority Queue:*
- K-th largest/smallest element
- Continuous median
- Merging K sorted streams
- Top K frequent elements

*Dynamic Programming:*
- Optimal substructure: solution uses optimal subsolutions
- Overlapping subproblems: same subproblems recomputed
- Keywords: "maximize", "minimize", "count ways", "longest/shortest"
- Check: can brute force with recursion → memoize → tabulate

*Graphs:*
- Connected components → DFS/BFS/Union-Find
- Shortest path → BFS (unweighted), Dijkstra (weighted)
- Cycle detection → DFS with coloring
- Topological sort → DFS or Kahn's algorithm
- Matrix as implicit graph (4/8-directional neighbors)

*Greedy:*
- Local optimum → global optimum (must prove correctness!)
- Sorting often helps reveal greedy structure
- "Earliest deadline first", "most profitable first"
- Exchange argument or stay-ahead proof required

*Backtracking:*
- Generate all combinations/permutations
- Constraint satisfaction
- Pruning with feasibility checks
- "Find all valid..." → backtracking

== Complexity Analysis Quick Reference

*Time complexity:*
- Single loop over n → $O(n)$
- Nested loops over n → $O(n^2)$
- Dividing problem in half each step → $O(log n)$
- Divide and conquer (balanced) → $O(n log n)$
- Exploring all subsets → $O(2^n)$
- Exploring all permutations → $O(n!)$
- Sort then process → $O(n log n)$

*Space complexity:*
- Fixed variables → $O(1)$
- Array/hash of size n → $O(n)$
- Recursion depth d → $O(d)$ stack space
- 2D matrix → $O(n m)$

== Common Optimization Strategies

*From brute force to optimal:*

1. *Eliminate redundant work:*
   - Recomputing same values → memoization/DP
   - Repeated searches → preprocess + hash/sort
   - Scanning entire array → prefix sums, segment tree

2. *Choose better data structure:*
   - Array search → hash table ($O(n)$ → $O(1)$)
   - Repeated min/max → heap ($O(n)$ → $O(log n)$)
   - Range queries → prefix sum, sparse table

3. *Reduce problem size:*
   - Binary search instead of linear
   - Two pointers instead of nested loops
   - Sliding window instead of checking all subarrays

4. *Algorithmic paradigm shift:*
   - Greedy vs DP (prove greedy works if possible - it's faster)
   - Iterative vs recursive (avoid stack overflow)
   - Bottom-up vs top-down (better cache locality)

== Performance Red Flags

*Avoid these patterns in hot paths:*
- `unordered_set<pair<int,int>>` → use `set<pair>` or manual hash
- Frequent `substr()` → use string_view or indices
- `priority_queue` with custom comparator → ensure inlining
- Recursion depth > 10K → risk stack overflow, use iteration
- Copying large containers → use move semantics or references
- Division/modulo in tight loop → strength reduction (compiler usually handles)
- Linked lists for sequential access → use vector
- `map`/`set` when order not needed → use `unordered_*`

== Edge Cases Checklist

*Always verify:*
- Empty input: `n = 0`, empty string, null pointer
- Single element: `n = 1`
- All elements same: `[5,5,5,5,5]`
- Sorted/reverse sorted input
- Integer overflow: sum of large numbers, product
- Negative numbers: if problem assumes positive
- Duplicate elements: if problem assumes unique
- Out of bounds: array access, string indexing

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

#include "coding/advanced_java.typ"
#pagebreak()

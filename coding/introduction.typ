= Algorithms and Data Structures: An Orientation

This subject is a practitioner's reference for algorithms and data structures. Each chapter covers one core data structure or algorithm family, presenting the essential invariants, worked problems that illustrate canonical patterns, complexity analysis, and implementation sketches. The goal is not to replace a textbook. CLRS gives you proofs; Skiena gives you design intuition. This reference gives you density: the material most likely to surface in an interview loop, a competitive programming contest, or a production engineering decision, distilled into a navigable form.

The chapters are ordered from foundational to advanced, but each is self-contained. A reader preparing for interviews can work linearly through Part I and Part II, then dip into whatever the job description demands. A competitive programmer will find the later parts — segment trees, flows, geometry, randomized algorithms — more immediately relevant. A software engineer looking for the right tool for a specific problem should start with the complexity landscape table below, then jump to the corresponding chapter.

*See also:* _Problem-Solving Framework_ (pattern recognition and optimization strategies), _Reference_ (complexity cheat sheet, common recurrences, master theorem summary).

== How to Use This Subject

- Each chapter covers one topic: a core data structure or algorithm family.
- Worked problems show canonical patterns, not exhaustive solutions. Understanding the pattern matters more than memorizing the solution.
- Complexity annotations follow standard Big-O notation. Space complexity is noted where non-trivial or where it drives the choice between alternatives.
- Code is Python unless otherwise noted. Python is chosen for readability, not performance. In production or contest contexts, re-implement in the target language; the algorithmic structure is identical.
- Pseudocode is used only where a language-neutral presentation is cleaner; it is always accompanied by a Python or C++ translation.

== The Complexity Landscape

The table below maps a problem characteristic to the canonical tool. Use it to quickly narrow the chapter you need.

#table(
  columns: (auto, auto, auto),
  inset: 7pt,
  align: (left, left, left),
  table.header(
    [*Goal*], [*Canonical structure*], [*Typical complexity*],
  ),
  [O(1) lookup by key],
  [Hash map],
  [$O(1)$ avg],

  [Ordered traversal / range queries],
  [Balanced BST / segment tree],
  [$O(log n)$],

  [Shortest path (unweighted)],
  [BFS],
  [$O(V + E)$],

  [Shortest path (weighted, non-negative)],
  [Dijkstra],
  [$O((V+E) log V)$],

  [Shortest path (negative weights)],
  [Bellman-Ford],
  [$O(V E)$],

  [All-pairs shortest path],
  [Floyd-Warshall],
  [$O(V^3)$],

  [Minimum spanning tree],
  [Kruskal / Prim],
  [$O(E log V)$],

  [Subarray / subwindow problems],
  [Sliding window / two pointers],
  [$O(n)$],

  [Overlapping subproblems],
  [Dynamic programming],
  [varies],

  [Partition into disjoint sets],
  [Union-Find],
  [$O(alpha(n))$ amortised],

  [Top-k / priority scheduling],
  [Heap],
  [$O(n log k)$],

  [Prefix queries],
  [Prefix sum / Fenwick tree],
  [$O(log n)$],
)

== Complexity Classes in Practice

The six complexity classes that matter for most practical work, from fastest to slowest:

*$O(1)$* — Constant time. Array indexing, hash-map lookup, stack push/pop. These operations do not scale with input size.

*$O(log n)$* — Logarithmic. Binary search, balanced BST operations, heap push/pop. Doubling $n$ adds one step. At $n = 10^9$, this is about 30 operations.

*$O(n)$* — Linear. A single pass through the input. The minimum cost for problems that must inspect every element.

*$O(n log n)$* — Linearithmic. Comparison-based sorting (merge sort, heap sort), many divide-and-conquer algorithms. The practical ceiling for problems that must sort or process all data.

*$O(n^2)$* — Quadratic. Nested loops over the input. Acceptable for $n <= 10^4$; too slow for $n = 10^6$.

*$O(2^n)$* — Exponential. Exhaustive enumeration of subsets. Only practical for $n <= 25$ or so.

The practical rule of thumb: modern hardware executes roughly $10^8$ simple operations per second. This gives the following rough feasibility table at $n = 10^6$:

#table(
  columns: (auto, auto, auto),
  inset: 7pt,
  align: (left, right, left),
  table.header(
    [*Complexity*], [*Ops at $n = 10^6$*], [*Verdict*],
  ),
  [$O(n)$],         [$10^6$],   [fast],
  [$O(n log n)$],   [$≈ 2 times 10^7$], [fast],
  [$O(n sqrt(n))$], [$≈ 10^9$], [marginal],
  [$O(n^2)$],       [$10^{12}$], [too slow],
  [$O(2^n)$],       [astronomical], [infeasible],
)

At $n = 10^3$, $O(n^2)$ is fine ($10^6$ ops). At $n = 10^8$, only $O(n)$ or $O(n log n)$ algorithms are safe. Always estimate $n$ before choosing an approach.

== Recurrence Relations

Many recursive algorithms produce a recurrence relation for their running time. The three most useful tools for solving recurrences are:

*Master theorem.* For $T(n) = a T(n\/b) + f(n)$ where $a >= 1$ and $b > 1$:

- Case 1: $f(n) = O(n^(log_b a - epsilon))$ for some $epsilon > 0$ → $T(n) = Theta(n^(log_b a))$.
- Case 2: $f(n) = Theta(n^(log_b a))$ → $T(n) = Theta(n^(log_b a) log n)$.
- Case 3: $f(n) = Omega(n^(log_b a + epsilon))$ and regularity holds → $T(n) = Theta(f(n))$.

*Common master theorem results:*

#table(
  columns: (auto, auto, auto),
  inset: 7pt,
  align: (left, left, left),
  table.header(
    [*Algorithm*], [*Recurrence*], [*Result*],
  ),
  [Merge sort],    [$T(n) = 2T(n\/2) + O(n)$],    [$O(n log n)$],
  [Binary search], [$T(n) = T(n\/2) + O(1)$],      [$O(log n)$],
  [Strassen],      [$T(n) = 7T(n\/2) + O(n^2)$],  [$O(n^(2.81))$],
  [Karatsuba],     [$T(n) = 3T(n\/2) + O(n)$],     [$O(n^(1.585))$],
)

*Substitution / expansion.* For recurrences outside the master theorem's scope, expand the first few levels and identify the pattern. For example, $T(n) = T(n-1) + O(1)$ expands to $T(n) = T(1) + (n-1) dot O(1) = O(n)$.

When the master theorem does not apply directly (non-polynomial differences between $f(n)$ and $n^(log_b a)$, variable branching factors, floors and ceilings), use the Akra-Bazzi method or expand by substitution.

== Space-Time Trade-offs

Most algorithmic improvements trade space for time. This exchange is so common that recognising it quickly is itself a core skill:

- *Hash maps* trade $O(n)$ space for $O(1)$ lookup, replacing an $O(n)$ linear scan or $O(log n)$ binary search.
- *Memoization / DP tables* trade $O(n)$ or $O(n^2)$ space to reduce exponential recursion to polynomial time.
- *Prefix sums* trade $O(n)$ space for $O(1)$ range-sum queries, replacing an $O(n)$ scan per query.
- *Fenwick and segment trees* trade $O(n)$ space for $O(log n)$ range queries and point updates, replacing $O(n)$ per update in a plain prefix-sum array.
- *Trie* trades $O(n dot k)$ space (where $k$ is key length) for $O(k)$ prefix lookup, replacing $O(n dot k)$ linear scan.

The inverse trade (time for space) appears in streaming algorithms and online algorithms, where the constraint is memory rather than time. Recognise both directions.

== Interview and Contest Strategy

*Pattern recognition first.* Before writing any code, classify the problem. The key distinctions:

- *Sliding window vs two pointers:* sliding window for contiguous subarrays with a maintained aggregate; two pointers for sorted arrays with pair/triplet constraints or in-place merging.
- *Binary search:* whenever the answer space is monotone and you can check a candidate in $O(n)$ or better, binary search on the answer is likely $O(n log n)$ overall.
- *Heap for top-k:* if the problem asks for the $k$ largest, smallest, or most frequent elements without requiring all elements sorted, a heap of size $k$ is the right tool.
- *Union-Find for connectivity:* any problem asking "are these two nodes in the same component?" with dynamic edge additions is a union-find problem.
- *Trie for prefix:* autocomplete, prefix counting, longest common prefix — trie first.
- *Segment tree for range updates with queries:* if both point updates and range queries (sum, min, max) appear, a segment tree (or Fenwick tree for sums) is correct.

*Complexity verification before coding.* Estimate $n$ from the constraints. Pick the target complexity class. Verify that your chosen algorithm achieves it before writing a single line.

*Edge cases that always matter:*
- Empty input: zero-length array, empty string, null root.
- Single element: the algorithm must not assume at least two elements.
- Duplicates: many algorithms break silently on duplicate keys.
- Integer overflow: sums of $10^9$-scale values require 64-bit integers.
- Negative values: algorithms that assume non-negative input (Dijkstra, certain DP formulations) fail silently with negative inputs.
- Off-by-one: boundary conditions in binary search, sliding window bounds, and DP base cases are the most common source of wrong answers.

== Further Reading

Cormen, T., Leiserson, C., Rivest, R., Stein, C. (2022). "Introduction to Algorithms" (4th ed.). MIT Press. — The canonical theoretical reference for proofs, recurrences, amortised analysis, and NP-completeness. Chapter 4 alone is a complete treatment of the master theorem and substitution method.

Skiena, S. (2020). "The Algorithm Design Manual" (3rd ed.). Springer. — The best single-volume reference for algorithm selection by problem type. The "war stories" and hitchhiker's guide chapters are uniquely practical.

Knuth, D. (1968-1973). "The Art of Computer Programming, Volumes 1-3." Addison-Wesley. — Exhaustive treatment of sorting, searching, and combinatorial algorithms. Indispensable for anyone who needs the ground truth on a classical algorithm's behaviour.

Sedgewick, R., Wayne, K. (2011). "Algorithms" (4th ed.). Addison-Wesley. — Reference-quality coverage with Java implementations and exceptional visual diagrams. Strongest on graph algorithms and string processing.

Laaksonen, A. (2018). "Competitive Programmer's Handbook." — Free online. Covers the full competitive programming toolkit from basic data structures through advanced graph theory, geometry, and game theory. Concise and example-driven.

Roughgarden, T. (2017-2020). "Algorithms Illuminated" (Parts 1-4). Soundlikeyourself Publishing. — Based on Stanford's algorithms course. The clearest modern treatment of divide-and-conquer, greedy algorithms, dynamic programming, and NP-completeness for a general engineering audience.

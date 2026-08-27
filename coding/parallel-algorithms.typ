#import "../template.typ": xref

= Parallel Algorithms <parallel-algorithms>

*Parallel algorithms* exploit multiple processors or cores to reduce wall-clock time below the sequential lower bound. This chapter introduces the *PRAM* model and its work-span analysis, then walks through the parallel patterns that recur in every modern data-parallel library (Cilk, TBB, OpenMP, CUDA, JAX `pmap`): parallel scan / prefix sum, sample sort, parallel BFS, segmented operations, and parallel reductions.

*See also:* _GPU Architecture_, _CPU Architecture_, #xref("coding", "dynamic-programming", label: "Dynamic Programming"), #xref("coding", "graphs", label: "Graphs"), #xref("coding", "flows-and-matching", label: "Network Flows and Matching").

== The PRAM Model

In the *Parallel Random Access Machine* (PRAM), $p$ processors share a common memory and execute synchronously. Conflict policies:

#table(
  columns: (auto, auto, auto),
  [*Variant*], [*Read*], [*Write*],
  [EREW], [Exclusive], [Exclusive],
  [CREW], [Concurrent], [Exclusive],
  [CRCW (common)], [Concurrent], [Concurrent (all must agree)],
  [CRCW (arbitrary)], [Concurrent], [Concurrent (one arbitrary wins)],
  [CRCW (priority)], [Concurrent], [Concurrent (lowest-id wins)],
)

PRAM is unrealistic (uniform memory, perfect sync) but pedagogically clean. Modern alternatives are *BSP* (Bulk Synchronous Parallel, Valiant 1990) and *LogP*.

== Work, Span, and Brent's Theorem

For a parallel algorithm:

- *Work* $W$ = total operations executed (sequential cost).
- *Span* (depth, critical path) $S$ = longest dependency chain.
- *Parallelism* $W / S$ = max useful processors.

*Brent's theorem.* On $p$ processors,

$ T_p <= W/p + S. $

A *work-efficient* algorithm has $W = O(T_("seq"))$ (no asymptotic blow-up) and ideally polylog span. The slogan: *minimize span without inflating work*.

#table(
  columns: (auto, auto, auto, auto),
  [*Problem*], [*Work*], [*Span*], [*Parallelism*],
  [Sum / Reduce], [$O(n)$], [$O(log n)$], [$O(n / log n)$],
  [Prefix Sum (Scan)], [$O(n)$], [$O(log n)$], [$O(n / log n)$],
  [Merge sort (parallel)], [$O(n log n)$], [$O(log^2 n)$], [—],
  [Sample sort], [$O(n log n)$], [$O(log^2 n)$ exp.], [—],
  [Matrix multiply ($n times n$)], [$O(n^3)$], [$O(log n)$], [—],
  [List ranking (pointer jump)], [$O(n log n)$], [$O(log n)$], [—],
)

== Parallel Reduction

Pairwise tree reduction halves the active set per step:

```python
# log n span; assumes n is a power of 2; in-place
def reduce_inplace(a, op):
    n = len(a); step = 1
    while step < n:
        for i in range(0, n, 2*step):       # parfor
            a[i] = op(a[i], a[i + step])
        step *= 2
    return a[0]
```

In CUDA this is the *warp shuffle* pattern; in OpenMP it is `#pragma omp parallel for reduction(+:s)`.

== Prefix Sum / Scan

The most important parallel primitive. Given $a_0, ..., a_(n-1)$ and an associative operator $plus.o$, compute $b_i = a_0 plus.o ... plus.o a_i$ (*inclusive*) or $b_i = a_0 plus.o ... plus.o a_(i-1)$ (*exclusive*).

=== Hillis-Steele (Span $O(log n)$, Work $O(n log n)$)

```python
def scan_hillis_steele(a):
    n = len(a); b = a[:]
    d = 1
    while d < n:
        new = b[:]
        for i in range(d, n):                # parfor
            new[i] = b[i - d] + b[i]
        b = new; d *= 2
    return b
```

Optimal span, but *not* work-efficient. Good when $n <= 32$ within a CUDA warp where work blow-up is small.

=== Blelloch (Work $O(n)$, Span $O(log n)$)

Two phases, both as balanced binary trees over $n$ leaves:

```text
# Up-sweep (reduce): a[2*i + 1] += a[2*i] at each level
# Then set a[n - 1] = 0  (identity for exclusive scan)
# Down-sweep: for each internal node, left  <- current; right <- current + old-left
```

```python
def scan_blelloch(a, identity=0, op=lambda x, y: x + y):
    n = len(a)
    assert (n & (n - 1)) == 0
    # Up-sweep
    d = 1
    while d < n:
        for i in range(0, n, 2*d):           # parfor
            a[i + 2*d - 1] = op(a[i + d - 1], a[i + 2*d - 1])
        d *= 2
    a[n - 1] = identity
    # Down-sweep
    d = n // 2
    while d >= 1:
        for i in range(0, n, 2*d):           # parfor
            t = a[i + d - 1]
            a[i + d - 1] = a[i + 2*d - 1]
            a[i + 2*d - 1] = op(t, a[i + 2*d - 1])
        d //= 2
    return a  # exclusive scan
```

*Applications of scan:* stream compaction (filter), radix sort (per-bin offsets), polynomial evaluation, sparse-matrix-vector multiply (segment offsets), parallel BFS frontier expansion.

== Sample Sort

Sample sort is a parallel generalisation of quicksort with *many* pivots, the de facto sort on distributed systems.

```text
1. Each of p processors locally sorts its block.
2. Each picks (s = O(p log n)) sample keys, all-gathered.
3. Sort and pick (p - 1) splitters from the global sample.
4. Each processor partitions its sorted block into p buckets by splitter.
5. All-to-all: bucket k goes to processor k.
6. Each processor merges (or sorts) the received pieces.
```

*Analysis.* With high probability the sample is "balanced": each processor receives at most $(1 + epsilon) n/p$ items. Work $O(n log n)$, communication $O(n/p)$ per processor — optimal up to constants. Used by Spark / Flink `sortBy`, MapReduce TeraSort.

== Parallel Merge

Merging two sorted arrays $A[1..n]$, $B[1..m]$ in span $O(log(n + m))$:

```text
def parallel_merge(A, B, C):
    if n + m below threshold: sequential merge
    mid = n / 2
    j = binary_search(B, A[mid])           # cross-rank
    C[mid + j] = A[mid]
    parallel: parallel_merge(A[..mid],   B[..j],   C[..mid+j])
              parallel_merge(A[mid+1..], B[j..],   C[mid+j+1..])
```

This is the kernel for *parallel merge sort* with span $O(log^2 n)$.

== Parallel BFS

A sequential BFS has work $O(V + E)$ and inherently sequential frontier expansion. The parallel BFS operates *frontier-at-a-time*:

```text
frontier <- {source}
visited <- {source}; level[source] <- 0
while frontier non-empty:
    # Build next frontier in parallel
    parfor v in frontier:
        for w in neighbors(v):
            if not visited[w]:
                CAS visited[w] = true
                if won the CAS:
                    level[w] <- level[v] + 1
                    append w to next_frontier
    swap frontier <- next_frontier
```

*Work* $O(V + E)$; *span* $O(D log V)$ where $D$ is the diameter. The atomic CAS or *bitmap* makes duplicate detection cheap. On directed graphs with high frontier expansion ratios, the *direction-optimizing BFS* (Beamer-Asanovic-Patterson 2012) flips to a "pull" traversal — every neighbour checks if any predecessor is in the frontier. This is the engine behind Galois, Ligra, GraphX, and most Graph500 winners.

=== Parallel SSSP

The $Delta$-stepping algorithm (Meyer-Sanders 2003) buckets unsettled vertices by *tentative distance* into intervals $[0, Delta), [Delta, 2Delta), ...$ and relaxes a whole bucket in parallel.

== Pointer Jumping and List Ranking

Given a linked list of $n$ nodes, computing the rank (distance from end) of every node:

```python
# At each step every node "jumps over" its current next, halving its list-suffix length
def list_rank(next, rank):
    n = len(next)
    # rank[i] = 1 except sentinel(last) = 0
    for _ in range(int.bit_length(n)):     # log n iterations
        for i in range(n):                 # parfor
            if next[i] != i:
                rank[i] += rank[next[i]]
                next[i] = next[next[i]]
```

Span $O(log n)$, work $O(n log n)$ — *not* work-optimal; the Cole-Vishkin technique brings it to $O(n)$ work.

== Concurrency Primitives

#table(
  columns: (auto, auto),
  [*Primitive*], [*Use*],
  [Atomic add / CAS], [Counters, hash table inserts, lock-free queues],
  [Barrier], [Phase synchronization (BSP supersteps)],
  [Read-write lock], [Read-heavy shared state],
  [Hazard pointers / RCU], [Lock-free reclamation],
  [Work-stealing deque], [Cilk, TBB, Tokio, Go runtime],
)

*Cilk's work-stealing scheduler* (Blumofe-Leiserson 1999) achieves $T_p <= W/p + O(S)$ in expectation — Brent's theorem with optimal scheduling overhead. The cost is $O(1)$ amortized per `spawn`/`sync`.

== Amdahl's and Gustafson's Laws

*Amdahl* (1967): if a fraction $f$ of work is sequential, max speedup is $1/f$. A 5% sequential portion caps speedup at 20× no matter the processor count.

*Gustafson* (1988): in practice we scale problem size with $p$; the sequential portion shrinks relative to the (growing) parallel one, so usable speedup grows linearly with $p$. The two laws don't contradict; they describe different *workloads*.

== Memory and Communication Costs

Real machines are not PRAMs. Two extensions matter:

- *Cache-oblivious* algorithms (Frigo et al. 1999): recursively divide-and-conquer until problems fit in cache, achieving optimal $O(n/B + 1)$ cache transfers without knowing block size $B$. Matrix multiply, FFT, sort.
- *External / streaming* models: count *block I/Os* of size $B$ to disk; sorting $n$ elements requires $Theta((n / B) log_(M/B)(n/B))$ transfers (Aggarwal-Vitter 1988), with $M$ = RAM size.

On GPUs, *coalesced memory access* and *warp divergence* dominate constants; on CPUs, *false sharing* (two threads modifying different bytes of the same cache line) is the silent killer of scalability.

== Where the Patterns Live in Practice

#table(
  columns: (auto, auto),
  [*Pattern*], [*Real system using it*],
  [Tree reduction], [`AllReduce` in NCCL / MPI],
  [Scan (Blelloch)], [CUB / Thrust `exclusive_scan`, JAX `scan`],
  [Sample sort], [Spark `sortBy`, MapReduce TeraSort],
  [Direction-optimizing BFS], [Graph500 reference, Ligra],
  [$Delta$-stepping], [GAPBS, Ligra, Galois],
  [Work-stealing], [Cilk, TBB, Rayon, Go scheduler],
  [Pipelined scan], [GPU FFT, prefix-sum kernels],
)

== Further Reading

*JáJá, J. (1992).* An Introduction to Parallel Algorithms. Addison-Wesley. Classic PRAM textbook.

*Blelloch, G.E. (1990).* Vector Models for Data-Parallel Computing. MIT Press. Origin of the scan-centric mental model.

*Blumofe, R.D. & Leiserson, C.E. (1999).* Scheduling Multithreaded Computations by Work Stealing. JACM 46(5): 720-748.

*Frigo, M., Leiserson, C.E., Prokop, H. & Ramachandran, S. (1999).* Cache-Oblivious Algorithms. FOCS 1999.

*Beamer, S., Asanovic, K. & Patterson, D. (2012).* Direction-Optimizing Breadth-First Search. SC 2012.

*Meyer, U. & Sanders, P. (2003).* $Delta$-Stepping: A Parallelisable Shortest Path Algorithm. J. Algorithms 49(1): 114-152.

*Valiant, L.G. (1990).* A Bridging Model for Parallel Computation. CACM 33(8): 103-111. BSP.

*Hwu, W.W., Kirk, D.B. & Hajj, I.E. (2022).* Programming Massively Parallel Processors, 4th ed. Morgan Kaufmann.

*Mattson, T.G., Sanders, B.A. & Massingill, B.L. (2004).* Patterns for Parallel Programming. Addison-Wesley.

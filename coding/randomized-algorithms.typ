#import "../template.typ": xref

= Randomized Algorithms

*Randomization* often yields algorithms that are simpler, faster, or more practical than their deterministic counterparts: it breaks symmetry, smooths adversarial inputs, and enables sublinear computation. This chapter covers the two flavours of randomization (Las Vegas and Monte Carlo), Karger's surprising min-cut algorithm, MinHash for set similarity, reservoir sampling for streams, and treaps as a self-balancing BST with no rotations to memorise.

*See also:* #xref("coding", "probabilistic-data-structures", label: "Probabilistic Data Structures"), #xref("coding", "streaming-algorithms", label: "Streaming Algorithms"), #xref("coding", "hashing", label: "Hashing"), #xref("coding", "trees", label: "Trees").

== Two Flavours of Randomization

#table(
  columns: (auto, auto, auto),
  [*Property*], [*Las Vegas*], [*Monte Carlo*],
  [Output], [Always correct], [Possibly wrong with bounded prob.],
  [Runtime], [Random variable, usually bounded in expectation], [Deterministic or bounded],
  [Examples], [Randomized quicksort, treaps], [Karger min-cut, Miller-Rabin, MinHash],
  [Conversion], [LV $->$ MC: cap runtime, output garbage], [MC $->$ LV: verify and retry],
)

*One-sided error* MC algorithms can be amplified to arbitrarily low error probability by independent repetition. Two-sided error can be amplified by *majority vote* and Chernoff bounds.

== Randomized Quicksort

The simplest non-trivial Las Vegas algorithm. Picking the pivot uniformly at random gives expected runtime $O(n log n)$ on *every* input, defeating the adversarial $O(n^2)$ worst case of deterministic median-of-three on sorted-and-then-rotated inputs.

*Proof sketch.* Let $X_(i j)$ indicate that the $i$-th and $j$-th smallest elements are ever compared. They are compared iff the first pivot chosen from ${z_i, ..., z_j}$ is $z_i$ or $z_j$ — probability $2/(j - i + 1)$. Total expected comparisons $sum_(i < j) 2/(j-i+1) = O(n log n)$.

```python
import random
def qsort(a):
    if len(a) <= 1: return a
    p = a[random.randrange(len(a))]
    return qsort([x for x in a if x < p]) + [x for x in a if x == p] + qsort([x for x in a if x > p])
```

== Karger's Min-Cut

For an undirected multigraph $G$, find a *global* minimum cut. Surprisingly, contraction of a uniformly random edge succeeds with non-trivial probability.

*Algorithm.* Repeatedly pick a uniform edge and contract its endpoints (merging the vertex sets, removing self-loops, keeping parallel edges). After $n - 2$ contractions, two super-vertices remain; the edges between them form a candidate cut.

*Correctness.* Let $C$ be a fixed min cut of size $k$. The probability that contraction never selects an edge of $C$ is

$ product_(i=0)^(n-3) (1 - k / (m_i)) >= product_(i=0)^(n-3) (1 - 2/(n-i)) = 2/(n(n-1)). $

Repeat $T = O(n^2 log n)$ times and return the smallest cut found: success probability $1 - 1/n$. Total work $O(n^4 log n)$.

*Karger-Stein (1996)* improves this to $tilde O(n^2)$ by recursing only after contracting down to $n / sqrt(2)$ vertices (where the survival probability is still $>= 1/2$), and recursing twice on independent contractions. The recurrence $T(n) = 2 T(n / sqrt(2)) + O(n^2)$ solves to $O(n^2 log n)$.

```python
import random, copy
def karger(adj):
    # adj: dict[node, list[node]] with multi-edges
    nodes = list(adj.keys())
    while len(nodes) > 2:
        u = random.choice(nodes)
        v = random.choice(adj[u])
        # merge v into u
        adj[u] = [w for w in adj[u] if w != v] + [w for w in adj[v] if w != u]
        for w in adj[v]:
            if w != u:
                adj[w] = [u if x == v else x for x in adj[w]]
        del adj[v]; nodes.remove(v)
    a, b = nodes
    return len(adj[a])  # cut size
```

== MinHash

*Problem:* estimate Jaccard similarity $J(A, B) = |A inter B| / |A union B|$ for two large sets, in space sub-linear in $|A|, |B|$, with one pass per set.

*Idea.* Fix a random permutation $pi$ of the universe. Let $h_pi(S) = min_(x in S) pi(x)$. Then

$ Pr[h_pi(A) = h_pi(B)] = |A inter B| / |A union B| = J(A, B). $

Use $k$ independent hash functions $h_1, ..., h_k$; the fraction of indices on which the signatures agree estimates $J$ with standard error $approx 1/sqrt(k)$.

```python
import hashlib, random
def minhash(S, seeds):
    sigs = [min(int(hashlib.md5(f"{s}-{x}".encode()).hexdigest(), 16) for x in S) for s in seeds]
    return sigs

def jaccard_est(sig_a, sig_b):
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)
```

*Locality-Sensitive Hashing (LSH).* Split the $k$ MinHash values into $b$ bands of $r$ rows; documents sharing any band hash to the same bucket. Tuning $b r = k$ gives the classical S-curve probability $1 - (1 - J^r)^b$ of being a candidate pair. Used by web-scale near-duplicate detection (Broder 1997 at AltaVista).

== Reservoir Sampling

*Problem:* sample $k$ items uniformly at random from a stream of unknown length $n$, using $O(k)$ memory and one pass.

*Algorithm R (Vitter 1985):* keep the first $k$ items; for $i = k+1, ..., n$, replace a random element of the reservoir with probability $k/i$.

```python
import random
def reservoir(stream, k):
    res = []
    for i, x in enumerate(stream):
        if i < k: res.append(x)
        else:
            j = random.randrange(i + 1)
            if j < k: res[j] = x
    return res
```

*Correctness.* By induction, after seeing $i$ items every prefix element is in the reservoir with probability $k / i$. When item $i + 1$ arrives, it enters with probability $k / (i+1)$; an existing element is kept with probability $1 - 1/(i+1) dot.op 1 = i/(i+1)$, so its new probability is $k / i dot i/(i+1) = k/(i+1)$.

*Algorithm L (Li 1994):* skip the next-replacement index by a geometric distribution. Reduces RNG calls from $O(n)$ to $O(k log(n/k))$ — huge win for very long streams.

*Weighted reservoir (Efraimidis-Spirakis 2006):* assign each item the key $u_i^(1/w_i)$ where $u_i tilde "Uniform"(0,1)$; keep the top-$k$ keys in a min-heap.

== Treaps

A *treap* is a binary search tree on keys whose nodes additionally carry random *priorities*; the tree obeys BST order on keys and heap order on priorities. With priorities drawn uniformly the expected height is $O(log n)$.

*Operations* via split / merge (no rotations to memorise):

```cpp
struct Treap {
    int key, prio, sz;
    Treap *l = nullptr, *r = nullptr;
    Treap(int k) : key(k), prio(rand()), sz(1) {}
};
int size(Treap* t) { return t ? t->sz : 0; }
void upd(Treap* t) { if (t) t->sz = 1 + size(t->l) + size(t->r); }

// split by key: left has keys < k, right has keys >= k
void split(Treap* t, int k, Treap*& a, Treap*& b) {
    if (!t) { a = b = nullptr; return; }
    if (t->key < k) { split(t->r, k, t->r, b); a = t; }
    else            { split(t->l, k, a, t->l); b = t; }
    upd(t);
}
Treap* merge(Treap* a, Treap* b) {
    if (!a || !b) return a ? a : b;
    if (a->prio > b->prio) { a->r = merge(a->r, b); upd(a); return a; }
    else                   { b->l = merge(a,    b->l); upd(b); return b; }
}
Treap* insert(Treap* t, int k) {
    Treap *a, *b; split(t, k, a, b);
    return merge(merge(a, new Treap(k)), b);
}
Treap* erase(Treap* t, int k) {
    Treap *a, *b, *c; split(t, k, a, b); split(b, k + 1, b, c);
    delete b; return merge(a, c);
}
```

*Implicit treaps* use subtree size as the "key" — they implement an array supporting $O(log n)$ insert / erase / reverse / range-sum, a Swiss-army knife in competitive programming.

== Skip Lists (Honourable Mention)

A *skip list* (Pugh 1990) augments a sorted linked list with multiple "express lanes" whose membership is decided by independent coin flips. Search / insert / delete are $O(log n)$ expected; the structure is far easier to implement concurrently than a balanced BST (Java's `ConcurrentSkipListMap`, RocksDB's MemTable).

== Bloom Filters and Count-Min (See Cross-References)

These standard randomized sketches are covered in detail in #xref("coding", "probabilistic-data-structures", label: "Probabilistic Data Structures") and #xref("coding", "streaming-algorithms", label: "Streaming Algorithms"); both are Monte Carlo with one-sided error.

== Tail Bounds Cheat-Sheet

#table(
  columns: (auto, auto, auto),
  [*Bound*], [*Statement*], [*When*],
  [Markov], [$Pr[X >= a] <= EE[X]/a$], [$X >= 0$],
  [Chebyshev], [$Pr[|X - mu| >= k sigma] <= 1/k^2$], [Finite variance],
  [Chernoff (mult.)], [$Pr[X >= (1+delta) mu] <= e^(-mu delta^2/3)$], [Sum of indep. $[0,1]$ vars],
  [Hoeffding], [$Pr[|bar X - mu| >= t] <= 2 e^(-2 n t^2 / R^2)$], [Bounded $[a, b]$, $R = b - a$],
  [Union bound], [$Pr[union E_i] <= sum Pr[E_i]$], [Always],
)

These tools convert "expected to work" into "works with high probability", essential for proving correctness of MC algorithms and tuning the number of repetitions.

== Derandomization Sketch

Two important techniques to remove randomness:

- *Method of conditional expectations:* greedily fix variables to whichever value preserves the average. Yields a deterministic $1/2$-approximation for MAX-CUT from the random-assignment analysis.
- *Pairwise independence and $k$-wise hashing:* many algorithms only need limited independence (e.g., $2$-universal hashing for Count-Min); the random source can be replaced by polynomials over $"GF"(p)$ with seed length $O(log n)$.

== Comparison Summary

#table(
  columns: (auto, auto, auto),
  [*Algorithm*], [*Type*], [*Bound*],
  [Randomized Quicksort], [Las Vegas], [$O(n log n)$ expected],
  [Karger Min-Cut], [Monte Carlo], [$O(n^4 log n)$ for $1 - 1/n$],
  [Karger-Stein], [Monte Carlo], [$tilde O(n^2)$],
  [MinHash], [Monte Carlo], [$O(k)$ per set, error $O(1/sqrt(k))$],
  [Reservoir (R)], [Las Vegas], [$O(n)$ time, $O(k)$ space],
  [Treap ops], [Las Vegas], [$O(log n)$ expected],
  [Miller-Rabin], [Monte Carlo], [$O(k log^3 n)$, error $4^(-k)$],
)

== Further Reading

*Motwani, R. & Raghavan, P. (1995).* Randomized Algorithms. Cambridge University Press. The canonical textbook.

*Mitzenmacher, M. & Upfal, E. (2017).* Probability and Computing, 2nd ed. Cambridge University Press.

*Karger, D.R. (1993).* Global Min-cuts in $cal(R)cal(N)cal(C)$, and Other Ramifications of a Simple Min-cut Algorithm. SODA 1993.

*Karger, D.R. & Stein, C. (1996).* A New Approach to the Minimum Cut Problem. JACM 43(4): 601-640.

*Broder, A.Z. (1997).* On the Resemblance and Containment of Documents. SEQUENCES 1997: 21-29. MinHash.

*Vitter, J.S. (1985).* Random Sampling with a Reservoir. ACM Trans. Math. Software 11(1): 37-57.

*Seidel, R. & Aragon, C.R. (1996).* Randomized Search Trees. Algorithmica 16: 464-497. Treaps.

*Pugh, W. (1990).* Skip Lists: A Probabilistic Alternative to Balanced Trees. CACM 33(6): 668-676.

#import "../template.typ": proof, theorem, xref

= Online Algorithms

*Online* algorithms make irrevocable decisions on a sequence of inputs without seeing the future. They are evaluated by the *competitive ratio* — the worst-case ratio of the online cost to that of the optimal *offline* algorithm that sees the whole sequence in advance. This chapter develops the standard playbook: ski rental, paging (LRU, FIFO, Marking), the $k$-server problem, and how randomization beats deterministic lower bounds.

*See also:* #xref("coding", "greedy", label: "Greedy"), #xref("coding", "dynamic-programming", label: "Dynamic Programming"), #xref("coding", "streaming-algorithms", label: "Streaming Algorithms"), #xref("coding", "randomized-algorithms", label: "Randomized Algorithms").

== Competitive Ratio

For an online algorithm ALG on input sequence $sigma$,

$ "ALG"(sigma) <= c dot.op "OPT"(sigma) + alpha quad forall sigma $

means ALG is *$c$-competitive* (with additive $alpha$ usually zero for normalized problems). For randomized algorithms we compare $EE["ALG"(sigma)]$ against OPT under various adversary models:

#table(
  columns: (auto, auto),
  [*Adversary*], [*Knows*],
  [Oblivious], [The algorithm but not its random bits],
  [Adaptive online], [Past random outcomes; not future],
  [Adaptive offline], [All random bits; compared offline],
)

The oblivious adversary is the standard, weakest model — and the one against which randomization usually helps the most.

== Ski Rental: The Canonical Online Problem

You can rent skis for \$1/day or buy them for \$$B$. Each morning you learn whether your knee is OK to ski. Decide rent vs buy without knowing how many days you will ski.

=== Deterministic: Break-Even

Rent for $B - 1$ days, then buy on day $B$. If you ski $d$ days, you pay $min(d, 2B - 1)$. Optimal offline pays $min(d, B)$. Ratio = $2 - 1/B -> 2$. *No deterministic algorithm beats $2$* (adversary stops you the day after you buy).

=== Randomized: $e/(e-1) approx 1.58$

*Algorithm RAND:* sample buy day $j$ from distribution $p_j = (1 - 1/B)^(B - j) / (B (1 - (1 - 1/B)^B))$ for $j = 1, ..., B$. Expected ratio: $e/(e-1) approx 1.582$. This is tight against the oblivious adversary.

```python
import random, math
def ski_rental_random(B):
    # Returns the day on which to buy. Until then, rent.
    weights = [(1 - 1/B)**(B - j) for j in range(1, B+1)]
    Z = sum(weights); probs = [w/Z for w in weights]
    r = random.random(); s = 0
    for j, p in enumerate(probs, 1):
        s += p
        if r <= s: return j
    return B
```

*General pattern.* Many "decide once, irreversible" problems (TCP acknowledgement timing, snoopy caching, lease scheduling) reduce to ski rental.

== Paging

Memory holds $k$ pages out of universe of $N$. A request to a page not in cache is a *fault*; on a fault, you must evict some page. Minimise faults.

*Offline optimum (Belady 1966):* on a fault, evict the page whose next use is furthest in the future. Optimal, but requires future knowledge.

=== Deterministic Online

#table(
  columns: (auto, auto, auto),
  [*Algorithm*], [*Eviction policy*], [*Competitive*],
  [LRU], [Least recently used], [$k$-competitive],
  [FIFO], [First in first out], [$k$-competitive],
  [LIFO], [Last in first out], [Not competitive],
  [LFU], [Least frequently used], [Not competitive],
)

#theorem(name: "Sleator-Tarjan 1985")[No deterministic online paging algorithm is better than $k$-competitive.]

#proof[Adversary uses $k + 1$ pages. Whatever ALG holds, request the missing page each round; ALG faults every step. Over $k$ phases of $k$ requests each, OPT (Belady) faults at most once per phase. Ratio $-> k$.]

=== Marking Algorithm (Randomized)

*Marking (Fiat-Karp-Luby-Naor-Rabani 1991):* divide requests into phases; mark a page when used; on a fault, evict a uniformly random *unmarked* page; clear all marks when all $k$ pages are marked.

#theorem[Marking is $2 H_k$-competitive against the oblivious adversary; this is tight up to constants (lower bound $H_k = ln k + O(1)$).]

```python
import random
class Marking:
    def __init__(self, k): self.k = k; self.cache = {}; self.marked = set()
    def access(self, p):
        if p in self.cache:
            self.marked.add(p); return False  # hit
        if len(self.cache) >= self.k:
            unmarked = [q for q in self.cache if q not in self.marked]
            if not unmarked: self.marked.clear(); unmarked = list(self.cache)
            victim = random.choice(unmarked); del self.cache[victim]
        self.cache[p] = True; self.marked.add(p); return True  # fault
```

=== Practice: LRU Still Wins

Despite worse worst-case ratio, *LRU* dominates real workloads because of *locality of reference*. *LRU-K*, *2Q*, and *ARC* (IBM 2003) refine LRU; *CLOCK-Pro* and *LIRS* are common in databases (PostgreSQL, MySQL). Linux uses an *active/inactive two-list* LRU with usage bits.

== $k$-Server Problem

$k$ servers live in a metric space. A sequence of requests arrives at points; one server must move to each requested point. Minimise total distance moved. *Generalises paging* (uniform metric on $N$ pages, $k$ server slots).

*Conjecture (Manasse-McGeoch-Sleator 1988):* there is a $k$-competitive deterministic algorithm for *every* metric. Proven for the line and trees (DC algorithm), then via the *Work Function Algorithm* (WFA) for all metrics with ratio $2k - 1$ (Koutsoupias-Papadimitriou 1995) — the long-standing best, *believed* to be tight at $k$ but proof open.

*Lower bound.* No deterministic algorithm is better than $k$-competitive on any metric with $>= k + 1$ points.

=== Greedy Is Bad

Always moving the *closest* server fails: on the line with two servers at $0$ and $2$, requests $1, 0, 1, 0, 1, ...$ alternate, the close server ping-pongs while the other sits idle; ratio unbounded.

=== Double-Coverage (Line / Tree, $k$-Competitive)

When a request $r$ arrives, the two servers adjacent to $r$ on the line both move toward $r$ at the same speed; whichever reaches it first serves. Costs at most $k dot "OPT"$. Generalises to trees.

== Bin Packing (Online)

Items of size $a_i in (0, 1]$ arrive one by one; pack into unit-capacity bins, minimise count.

#table(
  columns: (auto, auto),
  [*Algorithm*], [*Competitive*],
  [Next-Fit], [$2$],
  [First-Fit], [$1.7$],
  [Best-Fit], [$1.7$],
  [First-Fit Decreasing (offline!)], [$11/9$],
  [Best online (Seiden 2002)], [$1.5878$],
)

Online bin packing has a lower bound $approx 1.5403$ (Balogh-Békési-Galambos 2012). The gap is open.

== List Update (Sleator-Tarjan)

Maintain a singly linked list; accesses cost the index of the item; you may move accessed item *for free* toward the head and swap adjacent items for cost 1.

*Move-To-Front (MTF):* on every access, move the item to the head.

#theorem[MTF is $2$-competitive against the optimal offline algorithm — analyzed via the *potential* $Phi$ = number of inversions with the offline list. This was historically the first amortized analysis of an online algorithm.]

== Online Convex Optimization (Sketch)

In each round $t$, an algorithm plays $x_t in cal(K) subset.eq RR^d$; the adversary reveals convex $f_t$; the algorithm pays $f_t(x_t)$. *Regret* is

$ R_T = sum_(t=1)^T f_t(x_t) - min_(x in cal(K)) sum_t f_t(x). $

*Online Gradient Descent* (Zinkevich 2003): $x_(t+1) = Pi_(cal(K))(x_t - eta_t nabla f_t(x_t))$ with $eta_t = O(1/sqrt(t))$ achieves $R_T = O(sqrt(T))$. *Online Newton Step* and *Follow-the-Regularized-Leader* are the foundations of modern online learning, contextual bandits, and adversarial robustness.

== Yao's Principle: Lower Bounds via Distributions

To prove a *randomized* lower bound, give a distribution $D$ over inputs and show every *deterministic* algorithm has expected cost $>= L$ on $D$ (against a fixed offline optimum). By Yao's minimax principle, this is a lower bound on the competitive ratio of any randomized algorithm against the oblivious adversary.

Used to prove the $H_k$ paging lower bound (Fiat et al.), $e/(e-1)$ ski rental, and many bin-packing bounds.

== Recipe for Designing an Online Algorithm

1. *Identify offline OPT.* Often greedy is optimal offline (paging = Belady).
2. *Build a potential function $Phi$* tracking the "gap" between online state and OPT's state.
3. *Amortized inequality:* online cost $+ Delta Phi <= c dot "OPT cost"$.
4. *Telescope* to get $sum "online" <= c sum "OPT" + Phi_("init")$.
5. *Randomize* to defeat adversarial worst cases when deterministic ratios hit lower bounds.

== Summary Table

#table(
  columns: (auto, auto, auto),
  [*Problem*], [*Det. best*], [*Rand. best*],
  [Ski Rental], [$2 - 1/B$], [$e/(e-1) approx 1.58$],
  [Paging (size $k$)], [$k$ (LRU/FIFO)], [$H_k$ (Marking)],
  [$k$-Server (general metric)], [$2k - 1$ (WFA)], [conj. $O(log k)$ (Bansal et al.)],
  [List Update], [$2$ (MTF)], [$1.6$ (COMB)],
  [Online Bin Packing], [$1.58$], [open],
  [OCO], [$sqrt T$ (OGD)], [—],
)

== Further Reading

*Borodin, A. & El-Yaniv, R. (1998).* Online Computation and Competitive Analysis. Cambridge University Press. The definitive monograph.

*Sleator, D.D. & Tarjan, R.E. (1985).* Amortized Efficiency of List Update and Paging Rules. CACM 28(2): 202-208.

*Manasse, M., McGeoch, L. & Sleator, D. (1988).* Competitive Algorithms for Server Problems. J. Algorithms 11(2): 208-230.

*Koutsoupias, E. & Papadimitriou, C.H. (1995).* On the $k$-Server Conjecture. JACM 42(5): 971-983.

*Fiat, A., Karp, R.M., Luby, M., McGeoch, L.A., Sleator, D.D. & Young, N.E. (1991).* Competitive Paging Algorithms. J. Algorithms 12(4): 685-699.

*Belady, L.A. (1966).* A Study of Replacement Algorithms for a Virtual-Storage Computer. IBM Systems Journal 5(2): 78-101.

*Megiddo, N. & Modha, D.S. (2003).* ARC: A Self-Tuning, Low Overhead Replacement Cache. FAST 2003.

*Hazan, E. (2016).* Introduction to Online Convex Optimization. Foundations and Trends in Optimization. Free PDF.

*Zinkevich, M. (2003).* Online Convex Programming and Generalized Infinitesimal Gradient Ascent. ICML 2003.

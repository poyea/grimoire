= Joins and Aggregation

Join algorithms are the performance-critical core of relational query processing. The right algorithm depends on input sizes, available memory, sort order, and key distribution. Aggregation is deeply intertwined with join order and physical layout.

*See also:* _Query Optimization_, _Query Compilation_, _Column Stores and Vectorized Execution_

== Nested Loop Join (NLJ)

The simplest join: for each tuple in the outer relation, scan the inner relation.

```python
def nested_loop_join(outer, inner, predicate):
    for r in outer:
        for s in inner:
            if predicate(r, s):
                yield {**r, **s}

# Cost: O(|R| × |S|) I/Os — catastrophic for large tables
# Only useful when inner is tiny or there's an index on the inner join key
```

*Index nested loop join:* when the inner table has an index on the join key, the inner scan becomes a point lookup.

```
Cost: |R| × (index probe cost on S)
    = |R| × O(log |S|)    (B-Tree index)
```

```sql
-- PostgreSQL chooses index NLJ when outer is small and inner has an index
SET enable_hashjoin = off;
SET enable_mergejoin = off;

EXPLAIN SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id
WHERE o.order_date = '2025-01-01';
-- Index Scan on customers using customers_pkey (for each order row)
```

== Hash Join

Build a hash table on the smaller relation (build side), then probe with tuples from the larger relation (probe side).

```python
def hash_join(build_rel, probe_rel, key):
    # Build phase: O(|build|) time and space
    ht = {}
    for r in build_rel:
        k = r[key]
        ht.setdefault(k, []).append(r)

    # Probe phase: O(|probe|) time
    for s in probe_rel:
        for r in ht.get(s[key], []):
            yield {**r, **s}

# Total I/O cost: O(|R| + |S|)  — optimal for joins without sorted output needed
```

*Grace Hash Join* (when build side doesn't fit in memory):

```
1. Partition both relations into k buckets using h1(key):
   R → R_0, R_1, ..., R_{k-1}
   S → S_0, S_1, ..., S_{k-1}
   (Matching tuples always land in the same bucket)

2. For each pair (R_i, S_i):
   Load R_i into memory, build hash table using h2(key) ≠ h1
   Probe with S_i
   (Each bucket pair must fit in memory: |R_i| ≤ buffer pool / 2)

Cost: 3 × (|R| + |S|) I/Os  (read+write each partition, then re-read)
```

```c
// Vectorized hash join probe (simplified, 32-bit keys)
#include <stdint.h>
#include <string.h>

typedef struct { uint32_t key; uint32_t val; } Entry;

typedef struct {
    Entry   *table;
    uint32_t mask;       // size - 1 (power of two)
} HashTable;

// Open-addressing linear probe
void ht_insert(HashTable *ht, uint32_t key, uint32_t val) {
    uint32_t h = key * 2654435761u;          // Fibonacci hashing
    for (uint32_t i = h & ht->mask; ; i = (i+1) & ht->mask) {
        if (ht->table[i].key == 0) { ht->table[i] = (Entry){key, val}; return; }
    }
}

// Vectorized probe: process 8 probe keys at once
void ht_probe_batch(const HashTable *ht,
                    const uint32_t *probe_keys, int n,
                    uint32_t *out_vals, uint16_t *sel, int *cnt) {
    *cnt = 0;
    for (int i = 0; i < n; i++) {
        uint32_t k = probe_keys[i];
        uint32_t h = k * 2654435761u & ht->mask;
        while (ht->table[h].key && ht->table[h].key != k)
            h = (h + 1) & ht->mask;
        if (ht->table[h].key == k) {
            out_vals[*cnt] = ht->table[h].val;
            sel[(*cnt)++]  = i;
        }
    }
}
```

== Merge Join (Sort-Merge Join)

If both inputs are sorted on the join key, a single linear scan suffices.

```python
def merge_join(R_sorted, S_sorted, key):
    i, j = 0, 0
    while i < len(R_sorted) and j < len(S_sorted):
        r, s = R_sorted[i], S_sorted[j]
        if r[key] == s[key]:
            # Must handle duplicates: gather all matching S tuples
            j_start = j
            while j < len(S_sorted) and S_sorted[j][key] == r[key]:
                yield {**r, **S_sorted[j]}
                j += 1
            i += 1
            # If next R has same key, reset j
            if i < len(R_sorted) and R_sorted[i][key] == r[key]:
                j = j_start
        elif r[key] < s[key]:
            i += 1
        else:
            j += 1
# Cost: O(|R|+|S|) after sorting; total O(|R| log|R| + |S| log|S|)
# Best when inputs are already sorted (e.g., from an index scan)
```

== Worst-Case Optimal Joins (Leapfrog Triejoin, Generic Join)

For *cyclic joins* (e.g., triangle query $R(A,B) ⋈ S(B,C) ⋈ T(A,C)$), pairwise binary joins are provably suboptimal. The AGM bound (Atserias, Grohe, Marx 2008) gives the worst-case output size:

$ |R join S join T| <= sqrt(|R| dot |S| dot |T|) $

For a triangle on 3 relations each of size $N$: output $<= N^(3\/2)$, but binary joins produce intermediate results of size up to $N^2$.

*Generic Join / Leapfrog Triejoin* (Veldhuizen 2014) achieves $O(N^(rho^*))$ time where $rho^*$ is the *fractional edge cover number* of the query hypergraph — the optimum of the LP that assigns weights $x_e in [0,1]$ to each relation such that every attribute is covered ($sum_(e in.rev v) x_e >= 1$), minimizing $sum_e x_e$. For a 3-cycle (triangle), $rho^* = 3/2$, recovering the $N^(3/2)$ bound above.

```python
# Generic Join for triangle query R(a,b) ⋈ S(b,c) ⋈ T(a,c)
# Iterators: each is a sorted list of values

def generic_join_triangle(R_ab, S_bc, T_ac):
    """Enumerate triangles. Each input: sorted list of (k1, k2) tuples."""
    # Build indices
    R_by_a = group_by_first(R_ab)
    S_by_b = group_by_first(S_bc)
    S_by_bc = {(b,c) for b,c in S_bc}
    T_by_a = group_by_first(T_ac)

    for a in sorted(set(R_by_a) & set(T_by_a)):   # intersect on 'a'
        b_candidates = R_by_a[a]                    # R gives b values for this a
        c_candidates = T_by_a[a]                    # T gives c values for this a
        for b in b_candidates:
            s_c = {c for (bb, c) in S_bc if bb == b}  # S gives c values for this b
            for c in s_c & set(c_candidates):          # intersect
                yield (a, b, c)
```

Leapfrog Triejoin replaces the linear set intersection with a merge of sorted iterators that "leapfrog" past non-matching values — achieving the AGM bound in practice.

*Used in:* EmptyHeaded (Aberger et al. 2016), Umbra, LogicBlox.

== Yannakakis Algorithm for Acyclic Joins

For *acyclic* join queries (those whose hypergraph is acyclic), Yannakakis (1981) gives a linear-time algorithm:

```
1. Pick a leaf relation R_i (appears in only one join).
2. Semi-join reduce: R_i ← R_i ⋉ parent(R_i)   (keep only rows in R_i that match parent)
3. Remove R_i from the hypergraph.
4. Repeat until one relation remains.
5. Enumerate from the remaining relation upward (reverse order).
```

*Cost:* O(|input| + |output|) — output-sensitive. For queries where the join output is small, Yannakakis avoids constructing large intermediate results entirely.

== Aggregation

*Hash aggregation* (most common): build a hash table on group-by keys, accumulate aggregates.

```python
def hash_aggregate(rows, group_keys, agg_fn):
    groups = {}
    for row in rows:
        k = tuple(row[g] for g in group_keys)
        if k not in groups:
            groups[k] = agg_fn.init()
        agg_fn.accumulate(groups[k], row)
    for k, state in groups.items():
        yield dict(zip(group_keys, k)) | agg_fn.finalize(state)

# Example: SUM(amount) GROUP BY customer_id
class SumAgg:
    def init(self): return 0
    def accumulate(self, state, row): return state + row["amount"]
    def finalize(self, state): return {"total": state}
```

*Two-phase aggregation* for parallel execution:

```sql
-- PostgreSQL parallel hash aggregation
EXPLAIN SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id;
-- Finalize HashAggregate        ← merges partial results
--   Gather (4 workers)
--     Partial HashAggregate     ← each worker computes partial SUM
--       Parallel Seq Scan on orders
```

*Sort aggregation:* sort on group-by key, then stream through groups. Produces sorted output (useful if next operator needs it); avoids hash table memory.

== Window Functions

Window functions compute across a "window" of rows without collapsing them into groups.

```sql
-- Running total, rank, lag/lead — all examples of window functions
SELECT
    order_id,
    customer_id,
    amount,
    -- Partition: restart accumulation per customer
    SUM(amount) OVER (
        PARTITION BY customer_id
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total,
    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY amount DESC) AS rank_by_amt,
    amount - LAG(amount, 1, 0) OVER (PARTITION BY customer_id ORDER BY order_date) AS delta
FROM orders;
```

*Implementation:* sort by (PARTITION key, ORDER key), then stream through with a sliding aggregate. Spills to disk if the partition data exceeds `work_mem`.

== References

Shapiro, L. (1986). "Join Processing in Database Systems with Large Main Memories." ACM TODS 11(3).

DeWitt, D., Gray, J. (1992). "Parallel Database Systems: The Future of High Performance Database Systems." CACM 35(6).

Atserias, A., Grohe, M., Marx, D. (2008). "Size Bounds and Query Plans for Relational Joins." FOCS. (AGM bound)

Ngo, H., Porat, E., Ré, C., Rudra, A. (2018). "Worst-Case Optimal Join Algorithms." JACM 65(3).

Veldhuizen, T. (2014). "Leapfrog Triejoin: A Simple, Worst-Case Optimal Join Algorithm." ICDT.

Yannakakis, M. (1981). "Algorithms for Acyclic Database Schemes." VLDB.

Aberger, C. et al. (2016). "EmptyHeaded: A Relational Engine for Graph Processing." SIGMOD.

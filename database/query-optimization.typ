= Query Optimization

The query optimizer transforms a declarative SQL query into an efficient physical execution plan. This is the hardest engineering problem in databases: the search space of plans grows super-exponentially with the number of joins, cardinality estimation is error-prone, and cost models are approximations. Modern optimizers combine dynamic programming, heuristics, and increasingly, learned components.

*See also:* _Query Compilation_, _Joins and Aggregation_, _Storage Engines_

== Optimization Pipeline

```
SQL text
  ↓  Parser         → AST (abstract syntax tree)
  ↓  Binder         → resolve table/column names, type-check
  ↓  Logical rewriter → view expansion, constant folding, subquery unnesting
  ↓  Logical optimizer → predicate pushdown, join reordering (logical plan)
  ↓  Physical optimizer → choose scan type, join algorithm, index use (physical plan)
  ↓  Execution engine
```

Each stage has well-defined transformation rules. The optimizer applies rules in a cost-guided search.

== Relational Algebra Equivalences

The optimizer is free to reorder operations using algebraic equivalences. Key ones:

```
Selection pushdown (always):
  σ_p(R ⋈ S)  →  σ_p(R) ⋈ S          if p only references R
  σ_{p1 ∧ p2}(R)  →  σ_p1(σ_p2(R))   (split conjuncts)

Join commutativity and associativity:
  R ⋈ S  =  S ⋈ R
  (R ⋈ S) ⋈ T  =  R ⋈ (S ⋈ T)        → enables join reordering

Projection pushdown:
  π_A(R ⋈ S)  →  π_A(π_{A∪join_cols}(R) ⋈ π_{A∪join_cols}(S))
```

```sql
-- Example: the optimizer pushes predicates before joins
-- User writes:
SELECT o.id, c.name FROM orders o JOIN customers c ON o.cid = c.id
WHERE c.country = 'US' AND o.amount > 1000;

-- Optimizer rewrites to (conceptually):
-- σ_{amount>1000}(orders) ⋈ σ_{country='US'}(customers)
-- instead of joining first, then filtering
```

== Join Ordering

For $n$ tables, the number of possible left-deep trees is $n!$, the number of bushy trees is larger. Dynamic programming (System R style) computes the optimal plan bottom-up:

```python
# System R style dynamic programming join ordering
import itertools

def dp_join_order(tables: list[str], cost: callable, card: callable) -> dict:
    """Returns best_plan[frozenset(tables)] = (plan, estimated_cost)."""
    best = {}

    # Base case: single table access
    for t in tables:
        best[frozenset([t])] = (t, cost(t))

    # Build up sets of increasing size
    for size in range(2, len(tables) + 1):
        for subset in itertools.combinations(tables, size):
            s = frozenset(subset)
            best[s] = (None, float('inf'))
            for t in subset:
                left_set = s - {t}
                if left_set not in best:
                    continue
                left_plan, left_cost = best[left_set]
                join_cost = left_cost + card(left_set) + cost(t)  # simplified
                if join_cost < best[s][1]:
                    best[s] = ((left_plan, t), join_cost)
    return best

# For 5 tables: 5! = 120 left-deep plans considered
# PostgreSQL uses geqo (genetic algorithm) for > geqo_threshold tables (default 12)
```

*Bushy trees* can be asymptotically better than left-deep trees for certain schemas (e.g., star schemas where fact table is best joined last). PostgreSQL explores some bushy trees; commercial systems (Oracle, SQL Server) explore more.

== Cardinality Estimation

The biggest source of optimizer error. Cardinality = number of rows output by an operator.

*Selectivity* of predicate $p$: fraction of tuples satisfying $p$.

*Classic assumptions (often wrong):*
- Attribute value independence: $"sel"(p_1 ∧ p_2) = "sel"(p_1) × "sel"(p_2)$
- Uniform value distribution: estimate from column min/max.
- Independent join predicates.

```sql
-- PostgreSQL statistics: pg_statistic
SELECT attname,
       n_distinct,         -- estimated distinct values (-1 if fraction of rows)
       correlation,        -- physical ordering correlation
       most_common_vals,   -- MCVs
       most_common_freqs,  -- frequency of MCVs
       histogram_bounds    -- bucket boundaries for non-MCVs
FROM   pg_stats
WHERE  tablename = 'orders'
  AND  attname   = 'amount';

-- Increase statistics target for better estimates on skewed columns:
ALTER TABLE orders ALTER COLUMN amount SET STATISTICS 500;  -- default 100
ANALYZE orders;
```

*Extended statistics* (multi-column) correct independence errors:

```sql
-- Capture correlation between city and state (highly correlated)
CREATE STATISTICS order_city_state (dependencies)
    ON city, state FROM orders;
ANALYZE orders;
-- Now the planner knows P(city='NYC' AND state='NY') ≠ P(city='NYC') × P(state='NY')
```

=== Cardinality Estimation Errors Propagate

Errors multiply through joins:

```
Table R: 1M rows, estimate 100K (10× underestimate)
Table S: 1M rows, estimate 100K (10× underestimate)
Join R ⋈ S: true 10K rows, estimate 100K × 100K / 1M = 10K ← "correct" by cancellation
Add Table T: estimate compounds, error can reach 10^n for n-table joins
```

Experiments (Leis et al. 2015) show PostgreSQL's estimates off by >1000× for 5-table joins — enough to choose the wrong join algorithm.

== Physical Plan Selection

Given a logical plan, the optimizer assigns a *physical operator* to each logical operator:

#table(
  columns: (auto, auto, auto),
  [*Logical*], [*Physical options*], [*When best*],
  [Scan R],       [SeqScan, IndexScan, IndexOnlyScan, BitmapScan], [depends on selectivity],
  [R ⋈ S],        [NestedLoop, HashJoin, MergeJoin],              [depends on size, sort, key],
  [Agg(R)],       [HashAgg, SortAgg, StreamAgg],                  [depends on input sort],
  [Sort(R)],      [QuickSort, ExternalSort (disk)],                [fits in work_mem or not],
)

```sql
-- Force a specific plan for debugging (PostgreSQL)
SET enable_hashjoin    = off;
SET enable_nestloop    = off;   -- force merge join
SET enable_seqscan     = off;   -- force index use

EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM orders o JOIN customers c ON o.cid = c.id WHERE c.country = 'US';
```

== EXPLAIN Output Interpretation

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT customer_id, SUM(amount)
FROM   orders
WHERE  created_at > now() - interval '30 days'
GROUP  BY customer_id;
```

```
HashAggregate  (cost=52183..52384 rows=20100 width=12)
               (actual time=1203..1241 rows=18432 loops=1)
  Group Key: customer_id
  Buffers: shared hit=12030 read=2110
  ->  Bitmap Heap Scan on orders  (cost=1234..47890 ...)
        (actual time=23..987 rows=420000 loops=1)
        Recheck Cond: (created_at > ...)
        Buffers: shared hit=12030 read=2110
        ->  Bitmap Index Scan on orders_created_at_idx
              (actual time=18..18 rows=420000)
```

Key numbers:
- `cost=startup..total`: estimated cost in arbitrary units (page I/Os × 1.0 + CPU × 0.01...).
- `actual time=first..total ms`: real elapsed time.
- `rows`: if estimate >> actual, planner made an error → add statistics.
- `Buffers shared hit/read`: cache hit = no I/O, read = disk I/O.

*Huge discrepancy between estimated and actual rows signals cardinality errors.*

== Learned Query Optimization

*Bao (Marcus et al. 2021):* treats join ordering as a bandit problem. For each query, Bao generates a small set of plan candidates (by disabling planner hints), runs the query with the predicted-best plan, observes actual runtime, and updates a learned model. Over time, Bao learns which plan features predict fast execution for this workload/schema pair.

```python
# Bao's high-level loop (simplified)
def bao_select_plan(query, db, model):
    # Generate candidate plans by toggling PostgreSQL hints
    hints  = [None, "no_hash", "no_nestloop", "no_merge"]
    plans  = [explain_with_hint(query, db, h) for h in hints]
    # Model predicts cost (tree-structured neural net on plan)
    preds  = [model.predict(p) for p in plans]
    best   = plans[preds.index(min(preds))]
    actual = execute_and_time(query, db, best)
    model.update(best, actual)   # online learning
    return best
```

*NEO (Marcus et al. 2019):* learns a value function over partial plans using deep RL, replacing the DP search entirely with neural guidance.

*Practical status:* Bao is production-ready (integrated into some cloud DBs). Full RL-based optimizers (NEO) are still research.

== Subquery Unnesting

Correlated subqueries are expensive (re-execute per outer row). Optimizers unnest them into joins where possible.

```sql
-- Correlated subquery (O(n×m) execution naively)
SELECT * FROM orders o
WHERE o.amount > (
    SELECT AVG(amount) FROM orders WHERE customer_id = o.customer_id
);

-- Optimizer unnests to a join (O(n + m)):
SELECT o.*
FROM   orders o
JOIN   (SELECT customer_id, AVG(amount) AS avg_amt
        FROM   orders GROUP BY customer_id) avg_o
    ON avg_o.customer_id = o.customer_id
WHERE  o.amount > avg_o.avg_amt;
```

PostgreSQL performs lateral unnesting, semi-join flattening, and EXISTS-to-join transformation automatically for most patterns.

== References

Selinger, P. et al. (1979). "Access Path Selection in a Relational Database Management System." SIGMOD (System R).

Ioannidis, Y. (1996). "Query Optimization." ACM CSUR 28(1).

Leis, V. et al. (2015). "How Good Are Query Optimizers, Really?" VLDB.

Marcus, R. et al. (2019). "Neo: A Learned Query Optimizer." VLDB.

Marcus, R. et al. (2021). "Bao: Making Learned Query Optimization Practical." SIGMOD.

Graefe, G. (1993). "Query Evaluation Techniques for Large Databases." ACM CSUR 25(2).

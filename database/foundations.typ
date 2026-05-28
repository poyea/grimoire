= Database Foundations

Databases manage persistent, shared, and reliable data. The field rests on three pillars: the relational model (Codd 1970), transaction theory (ACID), and the architecture of a DBMS. Every modern system — from SQLite to Spanner — can be analyzed through these lenses.

*See also:* _Concurrency Control_, _Storage Engines_, _Isolation and Consistency Models_, _Hardware-Aware Database Design_

== The Relational Model

A *relation* $R$ is a set of tuples over a fixed schema $R(A_1: D_1, ..., A_n: D_n)$. No ordering, no duplicates (sets, not bags, in the pure model). SQL databases use bags (multisets) for performance — `DISTINCT` restores set semantics.

*Relational algebra operators:*

#table(
  columns: (auto, auto, auto),
  [*Operator*], [*Symbol*], [*SQL equivalent*],
  [Selection],       [$sigma_(p)(R)$],              [`WHERE p`],
  [Projection],      [$pi_(A)(R)$],                 [`SELECT A`],
  [Join],            [$R join S$],                   [`R JOIN S ON ...`],
  [Union],           [$R union S$],                  [`UNION ALL`],
  [Difference],      [$R \\ S$],                     [`EXCEPT`],
  [Rename],          [$rho_(B\/A)(R)$],              [`AS`],
  [Grouping/Agg],    [$gamma_(A, f)(R)$],            [`GROUP BY A`],
)

*Relational algebra is closed:* every operator takes relations and returns a relation, enabling arbitrary composition.

```sql
-- Find orders placed by customers in 'NY' with total > 100
SELECT c.name, SUM(o.amount) AS total
FROM   customers c
JOIN   orders    o ON o.customer_id = c.id
WHERE  c.state = 'NY'
GROUP  BY c.id, c.name
HAVING SUM(o.amount) > 100;
-- σ_{state='NY'} ⋈ γ_{name, Σ(amount)} ⋈ σ_{Σ>100}
```

== Functional Dependencies and Normal Forms

A *functional dependency* (FD) $X -> Y$ holds on $R$ when any two tuples that agree on $X$ must agree on $Y$.

*Armstrong's axioms* (sound and complete):
- *Reflexivity:* If $Y subset.eq X$, then $X -> Y$
- *Augmentation:* If $X -> Y$, then $X Z -> Y Z$
- *Transitivity:* If $X -> Y$ and $Y -> Z$, then $X -> Z$

```cpp
// Check if FD set implies X -> Y (closure algorithm)
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using Attr   = std::string;
using AttrSet = std::unordered_set<Attr>;
using FD     = std::pair<AttrSet, AttrSet>;

bool is_subset(const AttrSet& a, const AttrSet& b) {
    for (const auto& x : a)
        if (!b.count(x)) return false;
    return true;
}

AttrSet closure(const std::vector<FD>& fds, const AttrSet& x) {
    AttrSet cl = x;
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& [lhs, rhs] : fds) {
            if (is_subset(lhs, cl) && !is_subset(rhs, cl)) {
                cl.insert(rhs.begin(), rhs.end());
                changed = true;
            }
        }
    }
    return cl;
}

// closure(fds, {"A"})      -> {"A","B","C"}
// closure(fds, {"A","D"})  -> {"A","B","C","D","E"}
```

*Normal forms:*

- *1NF:* All attributes atomic (no repeating groups).
- *2NF:* 1NF + every non-key attribute is fully functionally dependent on the primary key (no partial deps).
- *3NF:* 2NF + no transitive dependencies (non-key → non-key).
- *BCNF:* For every non-trivial FD $X -> Y$, $X$ is a superkey. Eliminates all FD-based anomalies but may lose lossless join decomposition.

```sql
-- Bad: Employees(emp_id, emp_name, dept_id, dept_name, dept_location)
-- dept_id -> dept_name, dept_location  (transitive dep on emp_id)
-- BCNF decomposition:
CREATE TABLE departments (
    dept_id     INT PRIMARY KEY,
    dept_name   TEXT NOT NULL,
    location    TEXT NOT NULL
);
CREATE TABLE employees (
    emp_id   INT PRIMARY KEY,
    emp_name TEXT NOT NULL,
    dept_id  INT REFERENCES departments(dept_id)
);
```

== Keys and Constraints

- *Superkey:* A set of attributes whose closure contains all attributes.
- *Candidate key:* A minimal superkey.
- *Primary key:* The chosen candidate key (non-null).
- *Foreign key:* A set of attributes referencing a primary key in another relation — enforces referential integrity.

```sql
-- Declarative integrity constraints
CREATE TABLE orders (
    order_id    BIGSERIAL PRIMARY KEY,
    customer_id INT       NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    status      TEXT      NOT NULL CHECK (status IN ('pending','shipped','closed')),
    amount      NUMERIC   NOT NULL CHECK (amount > 0),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

== ACID Properties

Every database transaction must satisfy ACID:

#table(
  columns: (auto, auto, auto),
  [*Property*], [*Guarantee*], [*Mechanism*],
  [Atomicity],   [All-or-nothing — partial writes never visible],          [Undo log / rollback],
  [Consistency], [DB moves from one valid state to another],               [Constraints + application logic],
  [Isolation],   [Concurrent txns don't interfere],                        [Concurrency control (locks, MVCC)],
  [Durability],  [Committed data survives crashes],                        [WAL / fsync],
)

```sql
BEGIN;
  UPDATE accounts SET balance = balance - 100 WHERE id = 1;
  UPDATE accounts SET balance = balance + 100 WHERE id = 2;
  -- If either UPDATE fails, ROLLBACK unwinds both (Atomicity)
COMMIT;  -- fsync'd WAL record ensures Durability
```

*Why isolation is hard:* Serializability (the gold standard — equivalent to running txns one at a time) requires either expensive locking (2PL) or optimistic validation. Real systems often expose weaker isolation levels (Snapshot Isolation, Read Committed) for throughput.

== The SQL Standard

SQL is declarative: you describe *what*, not *how*. The planner decides the physical execution plan.

```sql
-- Window functions: running total without GROUP BY
SELECT
    order_id,
    amount,
    SUM(amount) OVER (PARTITION BY customer_id
                      ORDER BY created_at
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        AS running_total
FROM orders;

-- CTEs for readability and reuse
WITH monthly AS (
    SELECT date_trunc('month', created_at) AS month,
           SUM(amount) AS revenue
    FROM   orders
    GROUP  BY 1
)
SELECT month, revenue,
       revenue - LAG(revenue) OVER (ORDER BY month) AS delta
FROM   monthly;
```

== DBMS Architecture Layers

```
┌─────────────────────────────────────────────┐
│  Client / Application                        │
├─────────────────────────────────────────────┤
│  Query Parser   → AST                        │
├─────────────────────────────────────────────┤
│  Logical Rewriter / View Expansion           │
├─────────────────────────────────────────────┤
│  Query Optimizer  → Physical Plan            │
├─────────────────────────────────────────────┤
│  Execution Engine (volcano / vectorized)     │
├─────────────────────────────────────────────┤
│  Transaction Manager  (lock mgr, MVCC)       │
├─────────────────────────────────────────────┤
│  Buffer Pool Manager  (page cache)           │
├─────────────────────────────────────────────┤
│  Storage Engine  (B-Tree / LSM / heap)       │
├─────────────────────────────────────────────┤
│  OS / Filesystem / NVMe                      │
└─────────────────────────────────────────────┘
```

Each layer has a clean interface. The buffer pool hands *page pointers* to the storage engine; the executor never issues I/O directly — it pins pages from the buffer pool.

== Tuple Representation

*Fixed-length record* (row-store, heap file):

```
| null_bitmap (4 B) | col1 (INT 4B) | col2 (BIGINT 8B) | col3 (FLOAT 8B) |
```

*Variable-length record* (PostgreSQL):

```
| header (23 B) | null_bitmap | col1 | padding | col2 ptr (off, len) | ... | col2 data |
```

PostgreSQL stores variable-length attributes inline up to ~2 KB; larger values go to TOAST (The Oversized-Attribute Storage Technique), stored out-of-line in a separate table, possibly compressed.

```sql
-- Inspect tuple structure
SELECT attname, atttypid::regtype, attlen, attalign
FROM   pg_attribute
WHERE  attrelid = 'orders'::regclass AND attnum > 0;
```

== References

Codd, E. F. (1970). "A Relational Model of Data for Large Shared Data Banks." CACM 13(6).

Armstrong, W. W. (1974). "Dependency Structures of Data Base Relationships." IFIP.

Date, C. J. (2003). "An Introduction to Database Systems." 8th ed. Addison-Wesley.

Hellerstein, J., Stonebraker, M., Hamilton, J. (2007). "Architecture of a Database System." Foundations and Trends in Databases.

Ramakrishnan, R., Gehrke, J. (2002). "Database Management Systems." 3rd ed. McGraw-Hill.

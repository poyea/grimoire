= Query Languages

A query language is the surface through which users describe what they want; the optimizer and engine decide how. SQL has dominated for fifty years because relational algebra composes cleanly, but it now sits beside SQL/JSON for semistructured data, GQL (ISO/IEC 39075:2024) for property graphs, and Datalog descendants for recursive analytics. This chapter surveys the modern standards landscape and the algebraic kernels behind each.

*See also:* _Foundations_, _Query Optimization_, _Time-Series and Graph_, _Streaming and Incremental Computation_

== The SQL Standard

SQL is governed by ISO/IEC 9075. The standard is split into numbered parts: Framework (Part 1), Foundation (Part 2 — the bulk of the language), Call-Level Interface (Part 3), Persistent Stored Modules (Part 4), External Routines (Part 13), XML (Part 14), Multidimensional Arrays (Part 15, MDA), Property Graphs (Part 16, SQL/PGQ), and JSON (Part 6 in SQL:2023, formerly Part 6 of the Foundation).

*Major revisions:*

#table(
  columns: (auto, auto, auto),
  [*Edition*], [*Year*], [*Notable Additions*],
  [SQL-86 / SQL-89], [1986/89], [Original ANSI baseline.],
  [SQL-92], [1992], [Outer joins, CASE, set ops, schema info, three-part names.],
  [SQL:1999], [1999], [Recursive WITH, triggers, user-defined types, OLAP grouping.],
  [SQL:2003], [2003], [Window functions, MERGE, XML (Part 14), sequences, identity cols.],
  [SQL:2008], [2008], [TRUNCATE, INSTEAD OF triggers, fetch first N rows.],
  [SQL:2011], [2011], [System- and application-time tables (temporal).],
  [SQL:2016], [2016], [Row pattern recognition ($"MATCH_RECOGNIZE"$), JSON (Part 6), polymorphic table functions.],
  [SQL:2019], [2019], [Multidimensional arrays (Part 15).],
  [SQL:2023], [2023], [Property graph queries via SQL/PGQ, expanded SQL/JSON, $"ANY_VALUE"$, $"UNIQUE NULL TREATMENT"$, $"UNDERSCORE"$ numeric literals.],
)

Vendors implement subsets. PostgreSQL is closest to the standard among open systems; SQL Server and Oracle have extensive proprietary surface. The standard's full feature taxonomy uses *feature IDs* like `F311` (schema definition) and `T611` (window functions).

=== Recursive Common Table Expressions

Recursive CTEs (SQL:1999) lift SQL from first-order to fixpoint logic. The recursive query is a least fixpoint over a monotone operator.

```sql
-- Bill of materials: explode an assembly into all components
WITH RECURSIVE parts(part_id, qty) AS (
    SELECT part_id, 1
    FROM   assembly
    WHERE  part_id = :root
  UNION ALL
    SELECT s.child_id, p.qty * s.qty
    FROM   parts p
    JOIN   subparts s ON s.parent_id = p.part_id
)
SELECT part_id, SUM(qty) AS total_qty
FROM   parts
GROUP BY part_id;
```

Semantics: evaluate the anchor; repeatedly apply the recursive term; stop when no new rows arrive. `UNION` (vs `UNION ALL`) gives set semantics — the engine must dedupe each iteration. Cycles in the input graph produce non-termination with `UNION ALL`; engines either require user-managed cycle detection or, since SQL:2023, support `CYCLE ... SET ... USING` to materialize a path-cycle marker.

```sql
WITH RECURSIVE edges(src, dst, path) AS (
    SELECT src, dst, ARRAY[src, dst] FROM graph
  UNION ALL
    SELECT e.src, g.dst, path || g.dst
    FROM   edges e JOIN graph g ON e.dst = g.src
    WHERE  g.dst <> ALL(path)              -- cycle prevention
)
SELECT * FROM edges;
```

PostgreSQL evaluates recursive CTEs with a working table and an intermediate table; it does not currently inline mutually recursive references. DB2 and SQL Server support `MAXRECURSION` hints.

=== Window Functions

Window functions (SQL:2003) compute over a partitioned, ordered frame without collapsing rows. The frame clause (`ROWS`, `RANGE`, `GROUPS`) chooses how the window slides.

```sql
SELECT
    customer_id,
    order_date,
    amount,
    SUM(amount) OVER (PARTITION BY customer_id
                      ORDER BY order_date
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total,
    LAG(amount, 1) OVER (PARTITION BY customer_id ORDER BY order_date) AS prev_amount,
    NTILE(4) OVER (ORDER BY amount DESC) AS amount_quartile
FROM orders;
```

Internally a window operator sorts (or hash-partitions) once and streams a sliding aggregator. PostgreSQL implements `WindowAgg` over a sort node; DuckDB uses streaming segment trees for `RANGE` frames; ClickHouse uses block-level partial states.

=== MATCH_RECOGNIZE (Row Pattern Recognition)

SQL:2016 added regex-style row pattern matching over time-series data.

```sql
SELECT *
FROM   ticker
MATCH_RECOGNIZE (
    PARTITION BY symbol
    ORDER BY ts
    MEASURES  A.ts AS start_ts, LAST(C.ts) AS end_ts, MATCH_NUMBER() AS m
    PATTERN   (A B+ C+)
    DEFINE    B AS B.price < PREV(B.price),
              C AS C.price > PREV(C.price)
);
```

This detects V-shaped reversals: a starting row $A$, one-or-more declining rows $B$, then one-or-more ascending rows $C$. Oracle, Snowflake, Trino, and Flink SQL implement it; PostgreSQL does not yet.

== SQL/JSON

JSON support entered the standard in SQL:2016 and was substantially expanded in SQL:2023. There are three layers: storage, the JSON path language, and SQL/JSON functions.

```sql
-- Constructors
SELECT JSON_OBJECT('id' VALUE 7, 'tags' VALUE JSON_ARRAY('a','b'));

-- Path queries (SQL/JSON path is a separate sublanguage)
SELECT JSON_VALUE(doc, '$.address.city' RETURNING TEXT) AS city
FROM   customers;

SELECT JSON_QUERY(doc, '$.orders[*] ? (@.amount > 100)' WITH WRAPPER) AS big_orders
FROM   customers;

-- Tabularize: JSON_TABLE turns documents into relations
SELECT t.*
FROM   customers c,
       JSON_TABLE(c.doc, '$.orders[*]'
           COLUMNS (
             oid   INT  PATH '$.id',
             amt   NUMERIC PATH '$.amount',
             items NESTED PATH '$.items[*]'
                   COLUMNS (sku TEXT PATH '$.sku', qty INT PATH '$.qty')
           )) t;
```

The SQL/JSON path language (BNF in Part 2 §9.40) supports filters (`?`), wildcard (`*`), recursive descent in some dialects, and arithmetic. Oracle's original implementation predates the standard and uses slightly different syntax (`JSON_VALUE`'s `RETURNING` clause is standard, but Oracle defaults differ).

*Storage representations.* PostgreSQL's `jsonb` is a binary tree with sorted key entries enabling $O(log n)$ lookup; MySQL's `JSON` type stores a binary header table; SQL Server stores `NVARCHAR` and reparses on access (now changing in 2025+). MongoDB BSON shares many ideas with the binary path.

== GQL and Cypher

GQL (ISO/IEC 39075:2024) is the first new ISO query language standard in 37 years (SQL being the previous one in 1987). It standardizes property-graph querying. Cypher (Neo4j) was the primary inspiration; openCypher, PGQL (Oracle), and GSQL (TigerGraph) contributed.

```cypher
// Cypher: friends-of-friends not already friends
MATCH (me:Person {id: $uid})-[:FRIEND*2..2]-(fof:Person)
WHERE NOT (me)-[:FRIEND]-(fof) AND me <> fof
RETURN fof.name, count(*) AS mutual_friends
ORDER BY mutual_friends DESC
LIMIT 20;
```

GQL syntax is close to Cypher with adjustments:

```gql
MATCH (me:Person WHERE me.id = $uid)
      -[:FRIEND]-> {2} (fof:Person)
WHERE NOT EXISTS { (me)-[:FRIEND]->(fof) }
RETURN fof.name, COUNT(*) AS mutual
ORDER BY mutual DESC
LIMIT 20;
```

*Core constructs:*

- *Node patterns* `(v:Label {prop:val})` and *edge patterns* `-[r:TYPE]->`.
- *Variable-length paths* `-[*1..5]->`, *quantified path patterns* (GQL).
- *Path modes*: `WALK` (default), `TRAIL` (no repeated edge), `ACYCLIC`, `SIMPLE` (no repeated node), `SHORTEST k`.
- *Composable subqueries* via `CALL { ... }` returning rows.

*SQL/PGQ* (SQL:2023 Part 16) embeds the same pattern language inside SQL via `GRAPH_TABLE`:

```sql
SELECT * FROM GRAPH_TABLE (social_graph
    MATCH (p1:Person)-[:KNOWS]->(p2:Person)-[:KNOWS]->(p3:Person)
    WHERE p1.id = 42 AND p1 <> p3
    COLUMNS (p3.name AS fof_name)
) AS t;
```

This means an SQL engine can expose graph workloads without abandoning the relational model — the optimizer compiles `GRAPH_TABLE` into joins under a fixed schema mapping.

== Datalog and Its Descendants

Datalog is a syntactic subset of Prolog without function symbols, restricted to definite Horn clauses. Every program has a unique least fixpoint, computable bottom-up.

```
% Edges and reachability
edge(a, b). edge(b, c). edge(c, d).

reach(X, Y) :- edge(X, Y).
reach(X, Y) :- edge(X, Z), reach(Z, Y).
```

Naive evaluation re-derives all facts each round; *semi-naive* evaluation only re-evaluates rules using deltas from the previous iteration. *Magic sets* rewrites queries to push selections into recursion.

Stratified negation extends Datalog with `not p(X)` only when $p$ does not depend on the negated atom recursively. *Datalog#sub[$plus.minus$]* adds existential rules (tuple-generating dependencies) used in knowledge-base reasoning.

*Modern Datalog systems:*

- *Soufflé*: parallel Datalog compiler used for static program analysis (Doop, points-to).
- *LogicBlox / Vadalog*: commercial enterprise Datalog with aggregates.
- *DDlog (Differential Datalog)*: compiles Datalog to Differential Dataflow for incremental maintenance.
- *3DF, Datafrog*: research and embedded variants used in Rust borrow-checker prototypes and Materialize internals.

```
// DDlog: types are explicit, output relations stream incrementally
input relation Edge(src: u32, dst: u32)
output relation Path(src: u32, dst: u32)

Path(x, y) :- Edge(x, y).
Path(x, z) :- Edge(x, y), Path(y, z).
```

When `Edge` is updated, DDlog emits exactly the inserts and deletes to `Path` rather than recomputing — the incremental view maintenance problem solved by Differential Dataflow's timestamped multisets.

=== Differential Datalog and DBSP

Differential Dataflow (McSherry et al., 2013) extends dataflow with multisets, timestamps, and partial orders that admit consistent updates. DBSP (Budiu et al., 2023) reformulates the same idea as a stream-circuit calculus where every relational operator has a well-defined "derivative" satisfying $f(s + Delta s) = f(s) + (partial f)(Delta s)$ for small enough $Delta s$. Materialize, Feldera, and parts of RisingWave build on these foundations.

== Stream and Continuous Query Languages

Flink SQL, ksqlDB, RisingWave, and Materialize each extend SQL with temporal operators: `MATCH_RECOGNIZE`, windowed aggregations (`TUMBLE`, `HOP`, `SESSION`, `CUMULATE`), and temporal joins.

```sql
-- Flink SQL: tumbling window count
SELECT
    user_id,
    TUMBLE_START(event_time, INTERVAL '5' MINUTE) AS window_start,
    COUNT(*) AS events
FROM   click_stream
GROUP BY user_id, TUMBLE(event_time, INTERVAL '5' MINUTE);
```

The SQL:2016 standard added time-windowed table functions (`TUMBLE(...)`) and the streaming community has converged on these constructions.

== GraphQL Versus Database Query Languages

A common confusion: *GraphQL* is an API protocol, not a database query language. It defines a typed schema and an over-the-wire field-selection syntax. There is no algebra, no joins, no recursion (beyond hand-written resolvers), and no standardized cost model. Federating GraphQL requests across a database still requires real query languages underneath.

== Compilation Targets and IRs

Modern systems compile high-level query languages to *intermediate representations*:

- *Substrait* — a cross-engine relational IR (protobuf) used by Spark, Velox, DuckDB, DataFusion to share plans.
- *Calcite RelNode* — Java relational tree underlying Flink, Beam, Hive.
- *MLIR LinAlg / Comet* — research IR for relational + linear-algebra fusion.

This decoupling lets a Cypher front-end target the same execution engine as SQL.

== Comparison

#table(
  columns: (auto, auto, auto, auto),
  [*Language*], [*Algebra*], [*Recursion*], [*Standard*],
  [SQL:2023], [Bag relational + window + grouping sets], [Stratified recursive CTE], [ISO/IEC 9075],
  [SQL/JSON path], [Hierarchical], [Recursive descent (vendor)], [ISO/IEC 9075-2 §9.40],
  [Cypher / openCypher], [Multiset graph], [Quantified paths], [openCypher community],
  [GQL], [Multiset graph], [Quantified, path modes], [ISO/IEC 39075:2024],
  [Datalog], [Set, monotone Horn], [Least fixpoint], [No ISO standard],
  [Differential Datalog], [Multiset with timestamps], [Fixpoint over deltas], [Research],
  [SPARQL 1.1], [Graph patterns over RDF], [Property paths], [W3C Recommendation],
  [Substrait], [Relational IR], [N/A], [Open spec],
)

== Further Reading

ISO/IEC 9075:2023. "Information technology — Database languages — SQL." (multi-part).

ISO/IEC 39075:2024. "Information technology — Database languages — GQL."

Date, C.J., Darwen, H. (1996). "A Guide to the SQL Standard." Addison-Wesley.

Eisenberg, A., Melton, J. et al. (2004). "SQL:2003 Has Been Published." SIGMOD Record.

Melton, J., Michels, J.-E. et al. (2017). "SQL:2016 Row Pattern Recognition." SIGMOD Record.

Deutsch, A. et al. (2022). "Graph Pattern Matching in GQL and SQL/PGQ." SIGMOD.

Francis, N. et al. (2018). "Cypher: An Evolving Query Language for Property Graphs." SIGMOD.

Abiteboul, S., Hull, R., Vianu, V. (1995). "Foundations of Databases." Addison-Wesley (Datalog chapters).

Ceri, S., Gottlob, G., Tanca, L. (1989). "What You Always Wanted to Know About Datalog (And Never Dared to Ask)." TKDE.

Scholz, B. et al. (2016). "On Fast Large-Scale Program Analysis in Datalog." CC (Soufflé).

McSherry, F., Murray, D., Isaacs, R., Isard, M. (2013). "Differential Dataflow." CIDR.

Budiu, M. et al. (2023). "DBSP: Automatic Incremental View Maintenance for Rich Query Languages." VLDB.

Ryzhyk, L., Budiu, M. (2019). "Differential Datalog." Datalog 2.0 Workshop.

Begoli, E. et al. (2018). "Apache Calcite: A Foundational Framework for Optimized Query Processing." SIGMOD.

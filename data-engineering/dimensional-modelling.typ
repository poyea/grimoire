#import "../template.typ": xref

= Dimensional Modelling <dimensional-modelling>

Operational databases are designed for writing: normalised tables, each fact stored once, transactions touching a few rows. Analytics needs the opposite: questions such as "revenue by region, product category and month, compared with last year" scan millions of rows and group them by descriptive attributes. *Dimensional modelling*, codified by Ralph Kimball, organises analytical data into *facts* (measurements of business events) and *dimensions* (the context used to filter and group them). Two decades after it was designed for row-store data warehouses, it remains the dominant way to structure data for business intelligence, now on columnar warehouses and lakehouses. This chapter covers grain and fact table types, dimensions and surrogate keys, slowly changing dimensions with a C++ type 2 implementation and its SQL `MERGE` equivalent, conformed dimensions and the bus matrix, common patterns, alternatives (Data Vault, wide denormalised tables), and the semantic layer.

*See also:* #xref("data-engineering", "etl-vs-elt", label: "ETL vs ELT"), #xref("data-engineering", "lakehouse-engineering", label: "Lakehouse Engineering"), #xref("data-engineering", "schema-evolution", label: "Schema Evolution"), #xref("database", "oltp-vs-olap", label: "OLTP vs OLAP"), #xref("database", "query-optimization", label: "Query Optimization") (star joins).

== Facts, Dimensions, and the Star Schema

A *fact table* records measurements of a business process: one row per sales line, per payment, per page view. Its columns are *measures* (amount, quantity, duration) and *foreign keys* to dimensions. A *dimension table* describes the who, what, where and when: customer, product, store, date. Joining the fact table to its dimensions forms a *star schema*:

```text
                 dim_date
                    │
   dim_customer ── fact_sales ── dim_product
                    │
                 dim_store
```

Queries follow one pattern: filter and group by dimension attributes, aggregate fact measures.

```sql
SELECT d.year, d.month, p.category, s.region, SUM(f.net_amount) AS revenue
FROM fact_sales f
JOIN dim_date d     ON f.date_key = d.date_key
JOIN dim_product p  ON f.product_key = p.product_key
JOIN dim_store s    ON f.store_key = s.store_key
WHERE d.year IN (2025, 2026) AND s.country = 'DE'
GROUP BY d.year, d.month, p.category, s.region;
```

Query optimisers recognise this *star join* shape. They filter the small dimensions first, then use the qualifying keys to prune the large fact table, through semi-join reduction, Bloom filters pushed into the fact scan, or dynamic partition pruning. Dimensions are wide and short (thousands to millions of rows, dozens of descriptive columns); fact tables are narrow and long (billions of rows, a few keys and measures).

Normalising dimensions further, splitting `dim_product` into product, subcategory and category tables, gives a *snowflake schema*. It saves little storage, adds joins, and confuses users, so Kimball's advice is to keep dimensions denormalised unless a sub-dimension is very large or shared.

== Grain First

The most important design decision is the *grain*: exactly what one fact row represents. "One row per order line, as sold" is a grain; "sales data" is not. Declaring the grain first decides which dimensions apply (an order-line grain can have a product dimension; an order-header grain cannot) and which measures are valid. Mixing grains in one table, such as storing order totals and line amounts together, causes double counting when users sum across rows.

The atomic grain, the most detailed level the source provides, is almost always right for the base fact table. Aggregates can be derived from it; details cannot be recovered from aggregates.

== Types of Fact Tables

#table(
  columns: 3,
  [*Type*], [*One row per*], [*Example*],
  [Transaction], [Business event at a point in time], [Each sales line, each payment],
  [Periodic snapshot], [Entity per period, regardless of activity], [Account balance at end of each day, inventory per warehouse per week],
  [Accumulating snapshot], [Instance of a process with defined milestones, updated as it progresses], [An order with dates for placed, paid, shipped, delivered and lag measures between them],
  [Factless], [Event or coverage with no numeric measure], [Student attended class; product was on promotion in a store],
)

Measures have *additivity*: *additive* measures (sales amount) can be summed across all dimensions; *semi-additive* measures (account balance) can be summed across some dimensions but not time (summing daily balances over a month is meaningless; take the last or the average); *non-additive* measures (ratios, unit prices, percentages) cannot be summed at all and should be stored as their additive components (numerator and denominator) and computed at query time.

== Dimensions and Surrogate Keys

Dimension rows are identified by *surrogate keys*: meaningless integers assigned by the warehouse, separate from the *natural* (business) key such as a customer number from the source system. Surrogate keys:

- allow several versions of the same customer to coexist (essential for history, below);
- insulate the warehouse from source systems that reuse, change or merge natural keys;
- integrate several sources whose natural keys collide;
- represent "unknown" and "not applicable" with reserved rows (key 0 or $-1$) instead of nulls in fact tables, so inner joins never silently drop facts.

The *date dimension* is the canonical example of a rich dimension: one row per day, with year, quarter, month, week, day of week, fiscal periods, holiday flags and business-day indicators. Precomputing these attributes makes calendar logic consistent across every report.

== Slowly Changing Dimensions

Dimension attributes change: a customer moves, a product is recategorised. How a dimension handles change determines whether historical facts are reported against the old or new values. The standard techniques are numbered:

#table(
  columns: 3,
  [*Type*], [*Handling*], [*Effect on history*],
  [0], [Never change the attribute (original value retained)], [Facts always show the original value],
  [1], [Overwrite in place], [All history restated to the current value],
  [2], [Add a new row with a new surrogate key; close the old row with validity dates and a current flag], [Each fact keeps the version current when it occurred],
  [3], [Add a "previous value" column], [Limited history: current and one prior value],
  [4], [Move rapidly changing attributes to a separate mini-dimension], [Avoids row explosion in the main dimension],
  [6 (1 + 2 + 3)], [Type 2 rows plus a type 1 "current value" column on every row], [Report by historical or current value at will],
)

*Type 2* is the workhorse. When a tracked attribute of customer C-42 changes from `Berlin` to `Munich` on 2026-03-01, the existing row is closed (`valid_to = 2026-03-01`, `is_current = false`) and a new row with a new surrogate key opens (`valid_from = 2026-03-01`, `valid_to = 9999-12-31`, `is_current = true`). Facts loaded before the change point to the old key; facts after point to the new one. Reports by city then attribute each sale to where the customer lived at the time.

Applying a daily snapshot of the source to a type 2 dimension:

```cpp
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct customer_version {
    std::int64_t surrogate_key;
    std::string natural_key;
    std::string city;        // tracked as type 2
    std::string email;       // tracked as type 1 (overwrite all versions)
    std::string valid_from;  // ISO dates compare correctly as strings
    std::string valid_to;    // "9999-12-31" while current
    bool is_current;
};

struct source_customer {
    std::string natural_key, city, email;
};

// Apply one source snapshot taken on `as_of` to the dimension.
void apply_scd2(std::vector<customer_version>& dim,
                const std::vector<source_customer>& snapshot,
                const std::string& as_of, std::int64_t& next_key) {
    std::map<std::string, std::size_t> current;              // natural key -> row index
    for (std::size_t i = 0; i < dim.size(); ++i)
        if (dim[i].is_current) current[dim[i].natural_key] = i;

    for (const auto& src : snapshot) {
        const auto it = current.find(src.natural_key);
        if (it == current.end()) {                            // new member
            dim.push_back({next_key++, src.natural_key, src.city, src.email,
                           as_of, "9999-12-31", true});
            continue;
        }
        // Type 1 attribute: overwrite every version of this member.
        for (auto& row : dim)
            if (row.natural_key == src.natural_key) row.email = src.email;

        customer_version& cur = dim[it->second];
        if (cur.city != src.city) {                           // type 2 change
            cur.valid_to = as_of;
            cur.is_current = false;
            dim.push_back({next_key++, src.natural_key, src.city, src.email,
                           as_of, "9999-12-31", true});
        }
    }
}
```

In SQL warehouses the same logic is usually two statements, or one `MERGE` that closes changed rows plus an insert of their new versions (dbt's *snapshots* generate this automatically):

```sql
-- Close current rows whose tracked attributes changed.
MERGE INTO dim_customer d
USING stg_customer s
  ON d.natural_key = s.natural_key AND d.is_current
WHEN MATCHED AND d.city <> s.city THEN
  UPDATE SET valid_to = CURRENT_DATE, is_current = FALSE;

-- Insert new members and new versions of changed members.
INSERT INTO dim_customer (natural_key, city, email, valid_from, valid_to, is_current)
SELECT s.natural_key, s.city, s.email, CURRENT_DATE, DATE '9999-12-31', TRUE
FROM stg_customer s
LEFT JOIN dim_customer d ON d.natural_key = s.natural_key AND d.is_current
WHERE d.natural_key IS NULL;
```

Loading facts then requires a *lookup by effective date*: for each fact, find the dimension version whose `[valid_from, valid_to)` interval contains the event time. Late-arriving facts (events that arrive after a dimension change was applied) and late-arriving dimensions (facts that reference a customer not yet loaded, handled with an inferred placeholder row that is filled in later) are the common complications.

== Conformed Dimensions and the Bus Matrix

An enterprise has many business processes (orders, shipments, returns, web sessions, support tickets), each with its own fact table. If each uses its own customer and product tables with different keys and definitions, their numbers cannot be combined: "revenue per active customer" needs the same notion of customer in both places. *Conformed dimensions* are shared, identical dimensions (or strict subsets of them) used by every fact table that needs them. With conformed dimensions, *drill-across* queries aggregate each fact table separately by the shared attributes and then join the results.

The *bus matrix* is the planning tool: business processes as rows, dimensions as columns, a mark where a process uses a dimension. It shows which dimensions must be conformed first and lets teams build the warehouse incrementally, one process at a time, without creating silos.

== Common Patterns

- *Degenerate dimensions*: identifiers with no attributes of their own (an order number) stored directly in the fact table.
- *Junk dimensions*: a small table holding all combinations of low-cardinality flags (payment type, gift wrap, channel), referenced by one key instead of many flag columns.
- *Role-playing dimensions*: one physical dimension used in several roles, such as `dim_date` as order date, ship date and delivery date, through views or aliases.
- *Bridge tables* for many-to-many relationships, such as accounts with several holders or patients with several diagnoses, with allocation weights so that measures are not double counted.
- *Hierarchies*: fixed-depth hierarchies flattened into dimension columns; ragged or variable-depth ones (organisation charts) with a bridge table of ancestor-descendant pairs.

== Alternatives and Variations

*Data Vault* (Dan Linstedt) models the integration layer as *hubs* (business keys), *links* (relationships between hubs) and *satellites* (descriptive attributes with full history, insert-only). It is designed for auditability, parallel loading and absorbing source changes without restructuring, at the cost of many joins. Organisations that use it typically build dimensional *marts* on top for consumption.

*One Big Table* (wide denormalised tables) pre-joins facts with their dimension attributes into a single table. Columnar storage makes the repeated attribute values cheap (dictionary and run-length encoding compress them well) and joins disappear from queries. It is fast and simple for a known set of questions, but restating history (type 1 changes) means rewriting the whole table, and every new attribute is a schema change on the largest table. Many teams keep a dimensional model as the source of truth and generate wide tables from it for specific dashboards.

Columnar warehouses changed some of the old trade-offs (storage is cheap, scans are fast, joins on small dimensions are cheap) but not the core reasons for dimensional modelling: a shared, understandable vocabulary for the business, correct handling of history, and consistent definitions across teams.

== The Semantic Layer

A model is only useful if everyone computes metrics the same way. Without coordination, "active users" and "net revenue" get defined differently in every dashboard. A *semantic layer* (dbt's semantic layer and MetricFlow, Cube, LookML in Looker, metric views in lakehouse platforms) defines metrics once, in terms of the dimensional model, with their aggregation rules, filters and allowed dimensions, and generates the SQL for each query. Semi-additive and non-additive measures, time comparisons and fiscal calendars are encoded once and reused, and BI tools, notebooks and LLM-based query interfaces all draw on the same definitions.

== Pitfalls

- Undeclared or mixed grain, leading to double counting.
- Natural keys as fact foreign keys, which makes history impossible and breaks when sources change.
- Type 1 overwrites where the business needed history, discovered only when last year's reports change.
- Summing semi-additive or non-additive measures.
- Separate, unconformed customer or product dimensions per team.
- Nulls in fact foreign keys instead of explicit unknown members.

== Further Reading

Kimball, R. and Ross, M. (2013). _The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling_ (3rd edition). Wiley.

Kimball, R. et al. (2008). _The Data Warehouse Lifecycle Toolkit_ (2nd edition). Wiley.

Inmon, W. H. (2005). _Building the Data Warehouse_ (4th edition). Wiley.

Linstedt, D. and Olschimke, M. (2015). _Building a Scalable Data Warehouse with Data Vault 2.0_. Morgan Kaufmann.

Adamson, C. (2010). _Star Schema: The Complete Reference_. McGraw-Hill.

Chaudhuri, S. and Dayal, U. (1997). "An Overview of Data Warehousing and OLAP Technology." ACM SIGMOD Record.

Gray, J. et al. (1997). "Data Cube: A Relational Aggregation Operator Generalizing Group-By, Cross-Tab, and Sub-Totals." Data Mining and Knowledge Discovery.

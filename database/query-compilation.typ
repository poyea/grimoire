= Query Compilation

The execution engine translates a physical query plan into CPU instructions. The traditional *Volcano* (iterator) model is simple but slow — one virtual function call per row per operator. Modern systems use *vectorized execution* or *code generation* (compiling plans to machine code) to close the gap to hand-written C.

*See also:* _database/query-optimization.typ_, _database/joins-and-aggregation.typ_, _database/column-stores-and-vectorized-execution.typ_

== Volcano Iterator Model

The classic model (Graefe 1990): each operator implements `next()` returning one tuple at a time. The root operator pulls tuples from the tree.

```python
# Volcano model: each operator is a generator
class SeqScan:
    def __init__(self, table):
        self.rows = iter(table)
    def next(self):
        return next(self.rows, None)   # one row per call

class Filter:
    def __init__(self, child, predicate):
        self.child = child
        self.pred  = predicate
    def next(self):
        while True:
            row = self.child.next()
            if row is None: return None
            if self.pred(row): return row   # virtual dispatch per row!

class Project:
    def __init__(self, child, cols):
        self.child = child
        self.cols  = cols
    def next(self):
        row = self.child.next()
        if row is None: return None
        return {c: row[c] for c in self.cols}

# Usage: SELECT name FROM orders WHERE amount > 100
plan = Project(Filter(SeqScan(orders), lambda r: r["amount"] > 100), ["name"])
while (row := plan.next()) is not None:
    print(row)
```

*Why it's slow:*
- Virtual dispatch (`next()` call) per tuple per operator: ~100M ops/s limit.
- No vectorization: one value at a time — CPU can't use SIMD.
- Poor cache locality: L1 cache thrashes with operator state.
- Branching in tight loops inhibits branch prediction.

== Vectorized Execution (MonetDB/X100 Model)

Process a *batch* (vector) of 1024–8192 values at a time per operator. Tight inner loops on fixed-size typed arrays let the CPU's SIMD units engage.

```c
// Vectorized filter: process a batch of 1024 ints
// Returns number of passing rows and fills sel[] with their indices
int filter_gt(const int32_t *col, int n,
              int32_t threshold,
              uint16_t *sel_out) {
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        sel_out[cnt] = i;          // branch-free select
        cnt += (col[i] > threshold);
    }
    return cnt;
}

// Vectorized project: apply column expression to vector
void project_mul(const int32_t *a, const int32_t *b,
                 int64_t *out, int n) {
    for (int i = 0; i < n; i++)
        out[i] = (int64_t)a[i] * b[i];  // compiler auto-vectorizes to SIMD
}
```

*SIMD parallelism:* AVX2 processes 8 × 32-bit integers per instruction; AVX-512 processes 16. For a filter on 1024 ints, this reduces ~1024 comparisons to ~128 AVX2 instructions — 8× throughput.

```c
#include <immintrin.h>

// AVX2 filter: find indices where col[i] > threshold
int avx2_filter_gt(const int32_t *col, int n,
                   int32_t threshold, uint16_t *sel) {
    __m256i vthresh = _mm256_set1_epi32(threshold);
    int cnt = 0, i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i v    = _mm256_loadu_si256((const __m256i*)(col + i));
        __m256i mask = _mm256_cmpgt_epi32(v, vthresh);
        int     bits = _mm256_movemask_ps((__m256)mask);
        while (bits) {
            sel[cnt++] = i + __builtin_ctz(bits);
            bits &= bits - 1;
        }
    }
    for (; i < n; i++)
        if (col[i] > threshold) sel[cnt++] = i;
    return cnt;
}
```

*DuckDB* (column-oriented OLAP) uses vectorized execution with vectors of 2048 values and achieves near-optimal single-core throughput for aggregation and join workloads.

== Query Compilation (Code Generation)

*Neumann 2011 (HyPer):* compile the query plan directly to LLVM IR, which is JIT-compiled to native machine code. No virtual dispatch; the entire pipeline collapses into a tight loop with inlined operators.

*Push model (pipeline / produce-consume):* instead of pulling tuples up, operators *push* tuples down through a pipeline. When a pipeline is compiled, all operators in it become inlined code in a single loop.

```
Plan:  Scan → Filter → HashBuild
       Scan → ProbeHash → Aggregate

Compiled pipeline 1 (build side):
  for each row in table:                // inlined Scan
    if row.amount > 100:               // inlined Filter (no branch mispred)
      ht.insert(row.customer_id, row)  // inlined HashBuild

Compiled pipeline 2 (probe side):
  for each row in orders:
    entry = ht.lookup(row.customer_id) // inlined HashProbe
    if entry: agg[entry.region] += row.amount  // inlined Aggregate
```

The compiled code has no virtual calls, no tuple-at-a-time overhead, and fits working data in CPU registers.

```python
# Code generation (simplified LLVM IR generation for a filter)
import llvmlite.ir as ir
import llvmlite.binding as llvm

# Generate IR for: int32* filter_gt(int32* col, int n, int32 thresh, int16* sel)
mod    = ir.Module(name="query")
fn_ty  = ir.FunctionType(ir.IntType(32),
                          [ir.PointerType(ir.IntType(32)),
                           ir.IntType(32),
                           ir.IntType(32),
                           ir.PointerType(ir.IntType(16))])
fn     = ir.Function(mod, fn_ty, name="filter_gt_jit")
col, n, thresh, sel = fn.args

entry  = fn.append_basic_block("entry")
loop   = fn.append_basic_block("loop")
body   = fn.append_basic_block("body")
end    = fn.append_basic_block("end")

builder = ir.IRBuilder(entry)
cnt_ptr = builder.alloca(ir.IntType(32))
i_ptr   = builder.alloca(ir.IntType(32))
builder.store(ir.Constant(ir.IntType(32), 0), cnt_ptr)
builder.store(ir.Constant(ir.IntType(32), 0), i_ptr)
builder.branch(loop)
# ... (loop body: load, compare, conditionally store index, increment cnt)
```

=== HyPer vs DuckDB Strategy

#table(
  columns: (auto, auto, auto),
  [*System*], [*Approach*], [*When best*],
  [HyPer / Umbra], [LLVM codegen per query], [Complex queries; amortizes compile time],
  [DuckDB],        [Vectorized execution],    [Ad-hoc analytics; low compile overhead],
  [Velox],         [Vectorized (Facebook)],   [Shared execution library],
  [Spark Tungsten],[Whole-stage codegen],     [JVM overhead elimination],
)

*Compilation latency:* LLVM JIT for a complex query takes 50–200 ms. For queries running seconds to minutes, this is fine. For sub-millisecond OLTP, vectorized execution wins.

== Expression Compilation

Even in interpreted systems, *expression compilation* (compiling WHERE clause and projection expressions) pays off:

```python
# Python: compile filter expression to lambda at plan time
import ast, types

def compile_predicate(expr_str: str):
    """Compile 'amount > 100 and status = "paid"' to a Python callable."""
    code = compile(f"lambda row: {expr_str}", "<predicate>", "eval")
    return eval(code)

pred = compile_predicate("row['amount'] > 100 and row['status'] == 'paid'")
filtered = [r for r in rows if pred(r)]
```

PostgreSQL compiles expression trees to a JIT-compiled form using LLVM (enabled with `jit = on`) when a query has significant expression evaluation cost.

```sql
-- PostgreSQL JIT compilation
SET jit = on;
SET jit_above_cost = 100000;  -- enable JIT for expensive queries

EXPLAIN (ANALYZE, BUFFERS)
SELECT SUM(amount), COUNT(*)
FROM   orders
WHERE  amount > 100 AND status = 'paid' AND created_at > '2024-01-01';
-- Look for: "JIT: Functions: N  Options: Inlining true  Optimization true"
```

== Morsel-Driven Parallelism

*HyPer (Leis et al. 2014):* parallelize query execution by splitting the input into *morsels* (chunks of ~10,000 rows) and scheduling them across worker threads dynamically.

```
Table: 10M rows → 1000 morsels of 10K rows each
Workers: 8 threads
  Thread 0: morsel 0, 8, 16, ...  (steal-based scheduling)
  Thread 1: morsel 1, 9, 17, ...
  ...

Each morsel is compiled/vectorized independently.
No global synchronization until aggregation merge.
```

*Advantages:* NUMA-aware (morsels processed by the socket that owns the memory), adaptable to skew (short morsels → even load), integrates with OS scheduler via thread pinning.

```sql
-- PostgreSQL parallel query (closest equivalent)
SET max_parallel_workers_per_gather = 4;

EXPLAIN SELECT customer_id, SUM(amount)
FROM   orders
GROUP  BY customer_id;
-- Look for: Gather → Partial HashAggregate → Parallel Seq Scan
--           (4 workers each scanning 1/4 of the table)
```

== References

Graefe, G. (1990). "Encapsulation of Parallelism in the Volcano Query Processing System." SIGMOD.

Boncz, P., Zukowski, M., Nes, N. (2005). "MonetDB/X100: Hyper-Pipelining Query Execution." CIDR.

Neumann, T. (2011). "Efficiently Compiling Efficient Query Plans for Modern Hardware." VLDB.

Leis, V., Boncz, P., Kemper, A., Neumann, T. (2014). "Morsel-Driven Parallelism: A NUMA-Aware Query Evaluation Framework for the Many-Core Age." SIGMOD.

Kersten, T. et al. (2018). "Everything You Always Wanted to Know About Compiled and Vectorized Queries But Were Afraid to Ask." VLDB.

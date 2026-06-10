= Tree Decomposition Techniques

*Flattening trees into arrays unlocks range-query machinery on paths and subtrees. Euler tours, binary lifting, heavy-light decomposition, and centroid decomposition are the four core tools — each trading different constants to answer different families of queries on trees in $O(log n)$ or $O(log^2 n)$.*

*See also:* Segment Trees and Range Queries (sparse table, iterative segment tree), Trees (DFS, pointer-based tree basics), Graphs (DFS on general graphs), Advanced Graph Algorithms (offline LCA variants)

== Euler Tour and DFS Ordering

*Core idea:* A DFS over $n$ nodes visits exactly $2n - 1$ edges. Recording entry time `tin` and exit time `tout` for each node maps every subtree to a contiguous range in DFS order.

*DFS-order array:* Node $v$'s subtree occupies indices `[tin[v], tout[v]]` in the Euler-tour array. Any subtree aggregate becomes a range query.

```cpp
#include <vector>
#include <algorithm>
using namespace std;

struct EulerTour {
    int n;
    vector<vector<int>> adj;
    vector<int> tin, tout, order;
    int timer = 0;

    EulerTour(int n) : n(n), adj(n), tin(n), tout(n) {}

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Iterative DFS to avoid stack overflow on deep trees
    void build(int root) {
        order.reserve(n);
        vector<pair<int,int>> stk;   // (node, parent)
        stk.push_back({root, -1});

        while (!stk.empty()) {
            auto [v, p] = stk.back();
            stk.pop_back();

            if (v < 0) {
                // Returning from node ~v
                tout[~v] = timer - 1;
                continue;
            }

            tin[v] = timer++;
            order.push_back(v);
            stk.push_back({~v, p});      // Push sentinel for tout

            for (int u : adj[v]) {
                if (u != p) stk.push_back({u, v});
            }
        }
    }

    // Is u an ancestor of v?
    bool is_ancestor(int u, int v) const {
        return tin[u] <= tin[v] && tout[v] <= tout[u];
    }
};
```

*Subtree range:* To sum node values in the subtree of $v$, query the range `[tin[v], tout[v]]` on a Fenwick or segment tree indexed by DFS entry time.

*Time:* Build $O(n)$. Subtree query reduces to $O(log n)$ with a Fenwick tree.

*Cache behavior:* DFS order is contiguous in memory. A range query touches $O(log n)$ segment tree nodes with sequential index arithmetic -- far better than pointer-chasing the original tree.

== Binary Lifting for LCA

*Problem:* Given a rooted tree, answer Lowest Common Ancestor (LCA) queries: $"lca"(u, v)$ = deepest node that is an ancestor of both $u$ and $v$.

*Naive approach:* Walk both nodes up to the root. $O(n)$ per query.

*Binary lifting:* Precompute $"up"[v][k]$ = the $2^k$-th ancestor of node $v$. Then any ancestor at distance $d$ is reachable in $O(log n)$ jumps.

```cpp
struct BinaryLifting {
    static constexpr int LOG = 18;   // Handles n up to 2^18 = 262144

    int n;
    vector<int> depth;
    vector<array<int, LOG>> up;
    vector<vector<int>> adj;

    BinaryLifting(int n) : n(n), depth(n, 0), up(n), adj(n) {
        for (auto& row : up) row.fill(-1);
    }

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // O(n log n) build via iterative BFS
    void build(int root) {
        vector<int> order;
        order.reserve(n);
        vector<int> parent(n, -1);
        vector<bool> visited(n, false);

        // BFS to establish depths and direct parents
        vector<int> queue = {root};
        visited[root] = true;
        for (int qi = 0; qi < (int)queue.size(); qi++) {
            int v = queue[qi];
            order.push_back(v);
            up[v][0] = (parent[v] == -1) ? root : parent[v];

            for (int u : adj[v]) {
                if (!visited[u]) {
                    visited[u] = true;
                    parent[u] = v;
                    depth[u] = depth[v] + 1;
                    queue.push_back(u);
                }
            }
        }

        // Fill table in BFS order so up[v][k-1] is already computed
        for (int k = 1; k < LOG; k++) {
            for (int v : order) {
                int mid = up[v][k - 1];
                up[v][k] = (mid == -1) ? -1 : up[mid][k - 1];
            }
        }
    }

    // Lift v by exactly d levels; returns -1 if d > depth[v]
    int kth_ancestor(int v, int d) const {
        for (int k = 0; k < LOG; k++) {
            if ((d >> k) & 1) {
                v = up[v][k];
                if (v == -1) return -1;
            }
        }
        return v;
    }

    // O(log n) LCA query
    int lca(int u, int v) const {
        if (depth[u] < depth[v]) swap(u, v);

        // Bring u up to same depth as v
        u = kth_ancestor(u, depth[u] - depth[v]);

        if (u == v) return u;

        // Binary-lift both until just below the LCA
        for (int k = LOG - 1; k >= 0; k--) {
            if (up[u][k] != up[v][k]) {
                u = up[u][k];
                v = up[v][k];
            }
        }
        return up[u][0];
    }
};
```

*Complexity:*
- Space: $O(n log n)$ for the lifting table
- Build: $O(n log n)$
- Query: $O(log n)$

*Cache note:* `up[v][k]` accesses the $v$-th row then column $k$. With `LOG = 18` and 4-byte ints, each row is 72 bytes -- fits two cache lines. Sequential BFS processing keeps recently-written rows warm.

== Sparse Table for Static RMQ

*Problem:* Range Minimum Query on a static array: find $min(a_l, ..., a_r)$ in $O(1)$.

*Key insight:* Minimum is idempotent -- overlapping intervals are fine. Precompute minimums over all intervals of length $2^k$.

```cpp
struct SparseTableRMQ {
    vector<vector<int>> table;
    vector<int> log2_floor;
    int n;

    SparseTableRMQ(const vector<int>& arr) {
        n = arr.size();
        int levels = __lg(n) + 1;
        table.assign(levels, vector<int>(n));
        log2_floor.resize(n + 1);

        log2_floor[1] = 0;
        for (int i = 2; i <= n; i++)
            log2_floor[i] = log2_floor[i / 2] + 1;

        table[0] = arr;
        for (int k = 1; k < levels; k++) {
            for (int i = 0; i + (1 << k) <= n; i++) {
                table[k][i] = min(table[k-1][i],
                                  table[k-1][i + (1 << (k-1))]);
            }
        }
    }

    // O(1) query: minimum of arr[l..r] (inclusive)
    int query(int l, int r) const {
        int k = log2_floor[r - l + 1];
        return min(table[k][l], table[k][r - (1 << k) + 1]);
    }
};
```

*Complexity:*
- Build: $O(n log n)$ time and space
- Query: $O(1)$

=== Euler-Tour LCA via RMQ

*Alternative LCA:* Record the node visited on each step of the Euler walk (entering and leaving each node = $2n - 1$ entries). The LCA of $u$ and $v$ is the minimum-depth node between the first occurrences of $u$ and $v$ in this walk.

```cpp
struct LCA_RMQ {
    int n;
    vector<vector<int>> adj;
    vector<int> euler;     // 2n-1 entries
    vector<int> depth_arr; // depth at each euler position
    vector<int> first;     // first occurrence of node v in euler
    vector<int> node_depth;
    SparseTableRMQ* rmq = nullptr;

    LCA_RMQ(int n) : n(n), adj(n), first(n, -1), node_depth(n, 0) {}

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void dfs(int v, int p, int d) {
        first[v] = euler.size();
        euler.push_back(v);
        depth_arr.push_back(d);
        node_depth[v] = d;

        for (int u : adj[v]) {
            if (u == p) continue;
            dfs(u, v, d + 1);
            euler.push_back(v);
            depth_arr.push_back(d);
        }
    }

    void build(int root) {
        euler.reserve(2 * n - 1);
        depth_arr.reserve(2 * n - 1);
        dfs(root, -1, 0);
        rmq = new SparseTableRMQ(depth_arr);
    }

    // Returns LCA of u and v
    int lca(int u, int v) const {
        int l = first[u], r = first[v];
        if (l > r) swap(l, r);
        // Minimum depth in euler[l..r]
        int best_depth = rmq->query(l, r);
        // Walk from l to find node with that depth (simple scan for clarity)
        for (int i = l; i <= r; i++) {
            if (depth_arr[i] == best_depth) return euler[i];
        }
        return -1;  // Unreachable
    }
};
```

*Note:* In production, store the index-of-minimum (not the value) in the sparse table to retrieve the euler node in $O(1)$ without the linear scan above. The pattern is shown conceptually here.

*Comparison:*
- Binary lifting LCA: $O(n log n)$ build, $O(log n)$ query -- simpler to implement correctly
- Euler-tour+RMQ LCA: $O(n log n)$ build, $O(1)$ query -- preferred when LCA dominates runtime

== Heavy-Light Decomposition

*Goal:* Decompose a tree into $O(log n)$ contiguous chains so any root-to-leaf path intersects few chains. This reduces path queries to $O(log n)$ range queries.

=== Chain Theory

*Heavy edge:* For each non-leaf node $v$, the edge to the child with the largest subtree is the heavy edge. All other edges are light.

*Key claim:* Any root-to-leaf path crosses at most $O(log n)$ light edges.

_Proof:_ Crossing a light edge means moving to a subtree of size $< n(v)/2$. The subtree size at least halves each time, so the path can cross at most $log_2 n$ light edges.

*Heavy chain:* A maximal path of consecutive heavy edges. Each chain maps to a contiguous range in a DFS linearization, enabling segment tree queries.

```cpp
struct HLD {
    int n, timer = 0;
    vector<vector<int>> adj;
    vector<int> parent, depth, subtree_sz;
    vector<int> heavy;     // heavy child of each node (-1 if leaf)
    vector<int> head;      // top of the chain containing each node
    vector<int> pos;       // position of node in HLD linearization
    vector<int> node_at;   // inverse of pos

    HLD(int n) : n(n), adj(n), parent(n,-1), depth(n,0),
                 subtree_sz(n,1), heavy(n,-1), head(n), pos(n), node_at(n) {}

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Pass 1: compute subtree sizes and heavy children
    void dfs_size(int v, int p, int d) {
        parent[v] = p;
        depth[v]  = d;
        subtree_sz[v] = 1;
        int max_sz = 0;

        for (int u : adj[v]) {
            if (u == p) continue;
            dfs_size(u, v, d + 1);
            subtree_sz[v] += subtree_sz[u];
            if (subtree_sz[u] > max_sz) {
                max_sz   = subtree_sz[u];
                heavy[v] = u;
            }
        }
    }

    // Pass 2: assign chain heads and linearization positions
    void dfs_hld(int v, int h) {
        head[v] = h;
        pos[v]  = timer;
        node_at[timer] = v;
        timer++;

        if (heavy[v] != -1)
            dfs_hld(heavy[v], h);         // Continue same chain

        for (int u : adj[v]) {
            if (u == parent[v] || u == heavy[v]) continue;
            dfs_hld(u, u);                // New chain starts at u
        }
    }

    void build(int root) {
        dfs_size(root, -1, 0);
        dfs_hld(root, root);
    }

    // Path query: sum of node values on path u->v
    // seg: iterative segment tree over node_at[] values
    // Returns aggregate over path using a provided segment tree
    template<typename SegTree>
    int64_t path_query(int u, int v, SegTree& seg) const {
        int64_t result = 0;

        while (head[u] != head[v]) {
            if (depth[head[u]] < depth[head[v]]) swap(u, v);
            // u's chain head is deeper: query from head[u] to u
            result += seg.query(pos[head[u]], pos[u] + 1);
            u = parent[head[u]];          // Jump to parent of chain top
        }

        // Same chain: query the segment between u and v
        if (depth[u] > depth[v]) swap(u, v);
        result += seg.query(pos[u], pos[v] + 1);
        return result;
    }
};
```

*Complexity:*
- Build: $O(n)$ (two DFS passes)
- Path query: $O(log^2 n)$ (at most $O(log n)$ chain jumps, each $O(log n)$ segment tree query)
- Path query with sparse table on chains: $O(log n)$ -- replaces segment tree queries with $O(1)$ RMQ

*Why two DFS passes:* Subtree sizes must be fully computed before heavy children can be identified, so the linearization pass must follow.

=== HLD with Iterative Segment Tree

Full working example combining HLD and an iterative segment tree for sum queries:

```cpp
#include <vector>
#include <numeric>
using namespace std;

struct IterSegTree {
    vector<int64_t> t;
    int n;

    IterSegTree(int n, const vector<int>& init_vals) : n(n), t(2 * n, 0) {
        for (int i = 0; i < n; i++) t[n + i] = init_vals[i];
        for (int i = n - 1; i > 0; i--) t[i] = t[2*i] + t[2*i+1];
    }

    void update(int i, int val) {
        for (t[i += n] = val; i > 1; i >>= 1)
            t[i >> 1] = t[i] + t[i ^ 1];
    }

    // Sum over [l, r)
    int64_t query(int l, int r) const {
        int64_t res = 0;
        for (l += n, r += n; l < r; l >>= 1, r >>= 1) {
            if (l & 1) res += t[l++];
            if (r & 1) res += t[--r];
        }
        return res;
    }
};

// Thin wrapper connecting HLD positions to segment tree values
struct HLDTree {
    HLD hld;
    IterSegTree seg;

    HLDTree(int n, const vector<int>& node_vals, int root)
        : hld(n), seg(n, [&]{
            // Build initial array in HLD position order after hld.build
            return vector<int>(n, 0);  // placeholder; fill after build
        }()) {
        // In practice: build HLD first, then construct seg from permuted vals
        hld.build(root);
        // Permute node_vals by hld.pos
        vector<int> perm(n);
        for (int v = 0; v < n; v++) perm[hld.pos[v]] = node_vals[v];
        seg = IterSegTree(n, perm);
    }

    void update(int v, int val) {
        seg.update(hld.pos[v], val);
    }

    int64_t path_query(int u, int v) {
        return hld.path_query(u, v, seg);
    }
};
```

== Centroid Decomposition

*Problem:* Answer distance queries or path-aggregate queries efficiently by decomposing the tree into a hierarchy where every path passes through $O(log n)$ centroids.

*Centroid of a tree:* A node whose removal leaves no connected component with more than $n/2$ nodes. Every tree has at least one centroid, computable in $O(n)$.

*Centroid decomposition:* Recursively find the centroid, remove it, and decompose each remaining subtree. The resulting centroid tree has depth $O(log n)$ because each subtree has at most half the original nodes.

```cpp
struct CentroidDecomp {
    int n;
    vector<vector<int>> adj;
    vector<int> subtree_sz;
    vector<bool> removed;
    vector<int> centroid_parent;

    CentroidDecomp(int n)
        : n(n), adj(n), subtree_sz(n), removed(n, false), centroid_parent(n, -1) {}

    void add_edge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    int compute_size(int v, int p) {
        subtree_sz[v] = 1;
        for (int u : adj[v]) {
            if (u != p && !removed[u])
                subtree_sz[v] += compute_size(u, v);
        }
        return subtree_sz[v];
    }

    int find_centroid(int v, int p, int tree_sz) {
        for (int u : adj[v]) {
            if (u == p || removed[u]) continue;
            if (subtree_sz[u] > tree_sz / 2)
                return find_centroid(u, v, tree_sz);
        }
        return v;
    }

    // Build centroid tree; call build(root) to start
    void build(int v, int cp) {
        int sz = compute_size(v, -1);
        int c  = find_centroid(v, -1, sz);

        centroid_parent[c] = cp;
        removed[c] = true;                // Logically remove centroid

        for (int u : adj[c]) {
            if (!removed[u])
                build(u, c);             // Recurse on remaining subtrees
        }
    }

    void build(int root) { build(root, -1); }
};
```

*Distance-query pattern:* For each node $v$, precompute the distance from $v$ to all its ancestors in the centroid tree. To answer "minimum distance from node $u$ to any node in a set $S$", walk up $u$'s centroid ancestors ($O(log n)$ of them) and query a per-centroid auxiliary structure.

```cpp
// Typical usage skeleton for counting paths of length <= k:
// For each centroid c, store sorted distances from c to all
// nodes in c's centroid subtree. Answer queries with binary search.
// Total storage: O(n log n). Query: O(log^2 n).
```

*Complexity:*
- Build: $O(n log n)$ (each node appears in $O(log n)$ centroid subtrees)
- Centroid tree depth: $O(log n)$
- Path queries (depends on application): $O(log^2 n)$ typical

*Use cases:*
- Count paths with sum exactly $k$: store hashed distances per centroid
- All-pairs distances on trees: $O(n log n)$ storage, $O(log n)$ per pair
- Tree DP on paths without re-rooting

== Complexity and Cache-Behavior Comparison

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Technique*], [*Build*], [*Query*], [*Space*], [*Memory access pattern*],
  [Pointer-tree traversal], [$O(n)$], [$O(n)$ path], [$O(n)$], [Pointer chasing, random],
  [Euler tour + Fenwick], [$O(n)$], [$O(log n)$ subtree], [$O(n)$], [Sequential, cache-friendly],
  [Binary lifting LCA], [$O(n log n)$], [$O(log n)$], [$O(n log n)$], [Strided row access, warm],
  [Euler tour + RMQ LCA], [$O(n log n)$], [$O(1)$], [$O(n log n)$], [Two random reads per query],
  [HLD + segment tree], [$O(n)$], [$O(log^2 n)$ path], [$O(n)$], [Sequential chains, good locality],
  [HLD + sparse table], [$O(n log n)$], [$O(log n)$ path], [$O(n log n)$], [Sequential chains + $O(1)$ RMQ],
  [Centroid decomp.], [$O(n log n)$], [$O(log^2 n)$], [$O(n log n)$], [Ancestor walks, moderate],
)

*Practical guidance:*
- Subtree aggregates: Euler tour + Fenwick tree. Minimal overhead, sequential memory.
- LCA, few queries: Binary lifting. Simpler correctness, fits in cache for $n <= 10^5$.
- LCA, query-intensive: Euler-tour RMQ (Farach-Colton--Bender for true $O(1)$ with linear build).
- Path aggregates with updates: HLD + iterative segment tree. Industry standard.
- Distance queries across arbitrary node subsets: Centroid decomposition.

== Complexity Reference

#table(
  columns: (auto, auto, auto, auto),
  [*Operation*], [*Build*], [*Query*], [*Notes*],
  [Subtree sum (Euler+Fenwick)], [$O(n)$], [$O(log n)$], [Point updates supported],
  [LCA (binary lifting)], [$O(n log n)$], [$O(log n)$], [Also answers $k$-th ancestor],
  [LCA (Euler+RMQ)], [$O(n log n)$], [$O(1)$], [Static tree only],
  [Path query, no updates (HLD+RMQ)], [$O(n log n)$], [$O(log n)$], [Chains use sparse table],
  [Path query, with updates (HLD+seg)], [$O(n)$], [$O(log^2 n)$], [Each chain jump $O(log n)$],
  [Centroid dist. query], [$O(n log n)$], [$O(log^2 n)$], [Per centroid binary search],
  [Centroid path count], [$O(n log n)$], [$O(log^2 n)$], [With sorted distance arrays],
)

== References

*Primary Sources:*

*Harel, D. & Tarjan, R.E. (1984)*. Fast Algorithms for Finding Nearest Common Ancestors. SIAM Journal on Computing 13(2): 338--355.

*Bender, M.A. & Farach-Colton, M. (2000)*. The LCA Problem Revisited. LATIN 2000, Lecture Notes in Computer Science 1776: 88--94.

*Sleator, D.D. & Tarjan, R.E. (1983)*. A Data Structure for Dynamic Trees. Journal of Computer and System Sciences 26(3): 362--391. (Link-cut trees; HLD is a static simplification.)

*Algorithms & Theory:*

*Tarjan, R.E. (1979)*. Applications of Path Compression on Balanced Trees. Journal of the ACM 26(4): 690--715.

== Further Reading

Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). _Introduction to Algorithms_, 4th ed. MIT Press. (Chapter 21 on disjoint-set forests; Chapter 22 on depth-first search -- foundations for Euler tours.)

Sedgewick, R., & Wayne, K. (2011). _Algorithms_, 4th ed. Addison-Wesley. (Chapter 4 on graph processing; tree path problems as motivation for decomposition.)

Skiena, S. S. (2020). _The Algorithm Design Manual_, 3rd ed. Springer. (Chapter 15 on dynamic programming on trees; centroid decomposition as a divide-and-conquer technique.)

Tarjan, R. E., & Vishkin, U. (1985). "An Efficient Parallel Biconnectivity Algorithm." _SIAM Journal on Computing_ 14(4): 862--874. (Parallel Euler-tour techniques; single-source ancestor computations.)

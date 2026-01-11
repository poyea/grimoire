= Advanced Graph Algorithms

*Graph representation performance:* Adjacency list (sparse) = pointer chasing. Adjacency matrix (dense) = contiguous but $O(V^2)$ space. Edge list = sort-friendly for MST algorithms.

== Dijkstra's Algorithm (Single-Source Shortest Path)

*Problem:* Find shortest paths from source to all vertices in weighted graph (non-negative weights).

*Priority Queue Implementation:* $O((V + E) log V)$

```cpp
vector<int> dijkstra(int n, const vector<vector<pair<int,int>>>& adj, int src) {
    vector<int> dist(n, INT_MAX);
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;

    dist[src] = 0;
    pq.push({0, src});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();

        if (d > dist[u]) continue;  // Already processed with better distance

        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }

    return dist;
}
```

*Heap operations:*
- `push()`: $O(log V)$, ~10-15 cycles per comparison in heap [binary heap on vector]
- `pop()`: $O(log V)$, swap + sift-down
- Total heap operations: $O(E)$ pushes (one per edge relaxation)

*Cache behavior:*
- `dist` array: sequential access when V < 10K = fits in L2
- Priority queue: heap in vector = contiguous = good locality
- Adjacency list traversal: scattered memory access = cache-unfriendly for large sparse graphs

*Optimization - Lazy Deletion:*
Don't mark as visited, just check `d > dist[u]` and skip. Avoids separate visited array.

*Comparison - Heap Types:*

#table(
  columns: 5,
  align: (left, left, left, left, left),
  table.header([Heap Type], [Push], [Pop], [Decrease-Key], [Notes]),
  [Binary Heap (std)], [O(log V)], [O(log V)], [O(V)], [Best for sparse graphs],
  [D-ary Heap (d=4)], [O(log_d V)], [O(d log_d V)], [O(log_d V)], [Better for dense graphs],
  [Fibonacci Heap], [O(1) amort], [O(log V)], [O(1) amort], [Theory only, high const],
  [Pairing Heap], [O(1) amort], [O(log V)], [O(log V)], [Simpler than Fibonacci],
)

*Recommendation:* Binary heap for most cases. D=4 heap for dense graphs (more cache-friendly children).

*D-ary Heap (d=4):*
```cpp
// Children at indices: 4*i+1, 4*i+2, 4*i+3, 4*i+4
// Parent at: (i-1)/4
// Cache-friendly: 4 children in single cache line (32 bytes for int pairs)
```

== Bellman-Ford (Negative Weights)

*Problem:* Shortest path with negative edge weights. Detects negative cycles.

*Edge List Approach:* $O(V E)$

```cpp
struct Edge { int u, v, w; };

pair<vector<int>, bool> bellmanFord(int n, const vector<Edge>& edges, int src) {
    vector<int> dist(n, INT_MAX);
    dist[src] = 0;

    // Relax edges V-1 times
    for (int i = 0; i < n - 1; i++) {
        for (const auto& [u, v, w] : edges) {
            if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
            }
        }
    }

    // Check for negative cycle
    for (const auto& [u, v, w] : edges) {
        if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
            return {dist, false};  // Negative cycle exists
        }
    }

    return {dist, true};  // No negative cycle
}
```

*Performance:*
- Sequential edge iteration = excellent cache locality
- No heap operations = simpler, more predictable
- Branch on relaxation: depends on graph structure

*SIMD Optimization:* Relax multiple edges in parallel using AVX2.

```cpp
// Process 8 edges simultaneously
__m256i dist_u = _mm256_i32gather_epi32(dist.data(), u_indices, 4);
__m256i dist_v = _mm256_loadu_si256((__m256i*)&dist[v_base]);
__m256i weights = _mm256_loadu_si256((__m256i*)&edge_weights[i]);

__m256i new_dist = _mm256_add_epi32(dist_u, weights);
__m256i cmp = _mm256_cmpgt_epi32(dist_v, new_dist);

// Masked store (AVX-512) or manual iteration to update
```

Gather operation ($#sym.tilde.op$10 cycles) limits speedup to ~2-3x.

*Optimization - SPFA (Shortest Path Faster Algorithm):*

```cpp
vector<int> spfa(int n, const vector<vector<pair<int,int>>>& adj, int src) {
    vector<int> dist(n, INT_MAX);
    vector<bool> in_queue(n, false);

    deque<int> queue;
    dist[src] = 0;
    queue.push_back(src);
    in_queue[src] = true;

    while (!queue.empty()) {
        int u = queue.front();
        queue.pop_front();
        in_queue[u] = false;

        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                if (!in_queue[v]) {
                    queue.push_back(v);
                    in_queue[v] = true;
                }
            }
        }
    }

    return dist;
}
```

Average $O(E)$ but worst-case still $O(V E)$. Faster in practice for random graphs.

== Floyd-Warshall (All-Pairs Shortest Path)

*Problem:* Find shortest paths between all pairs of vertices.

*DP Approach:* $O(V^3)$

```cpp
vector<vector<int>> floydWarshall(int n, vector<vector<int>> dist) {
    // dist[i][j] = initial edge weights (INF if no edge)

    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dist[i][k] != INT_MAX && dist[k][j] != INT_MAX) {
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }

    return dist;
}
```

*Cache optimization - Blocked Floyd-Warshall:*

```cpp
const int B = 64;  // Block size to fit in cache

for (int kk = 0; kk < n; kk += B) {
    for (int ii = 0; ii < n; ii += B) {
        for (int jj = 0; jj < n; jj += B) {
            // Process B×B block
            for (int k = kk; k < min(n, kk+B); k++) {
                for (int i = ii; i < min(n, ii+B); i++) {
                    for (int j = jj; j < min(n, jj+B); j++) {
                        dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                    }
                }
            }
        }
    }
}
```

*Performance:*
- Standard: column-major access = cache miss per iteration for large n
- Blocked: keeps B×B submatrix in cache = 2-4x speedup for n > 512

*Memory layout:* Row-major storage critical. Column-major innermost loop = poor locality.

== Minimum Spanning Tree (MST)

=== Kruskal's Algorithm

*Greedy + Union-Find:* $O(E log E)$

```cpp
struct Edge {
    int u, v, weight;
    bool operator<(const Edge& other) const {
        return weight < other.weight;
    }
};

class UnionFind {
    vector<int> parent, rank;

public:
    UnionFind(int n) : parent(n), rank(n, 0) {
        iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // Path compression
        }
        return parent[x];
    }

    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;

        if (rank[px] < rank[py]) swap(px, py);
        parent[py] = px;
        if (rank[px] == rank[py]) rank[px]++;

        return true;
    }
};

int kruskal(int n, vector<Edge>& edges) {
    sort(edges.begin(), edges.end());  // O(E log E)

    UnionFind uf(n);
    int mst_weight = 0;
    int edges_added = 0;

    for (const auto& [u, v, w] : edges) {
        if (uf.unite(u, v)) {
            mst_weight += w;
            if (++edges_added == n - 1) break;  // MST complete
        }
    }

    return mst_weight;
}
```

*Performance bottleneck:* Sorting edges. For dense graphs, $E = O(V^2)$ → $O(V^2 log V)$ sort time.

*Cache:* Edge array sorted = sequential access. Union-Find parent array = random access but small (fits in cache for V < 100K).

=== Prim's Algorithm

*Greedy + Priority Queue:* $O((V + E) log V)$

```cpp
int prim(int n, const vector<vector<pair<int,int>>>& adj) {
    vector<bool> visited(n, false);
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;

    pq.push({0, 0});  // {weight, vertex}
    int mst_weight = 0;

    while (!pq.empty()) {
        auto [w, u] = pq.top();
        pq.pop();

        if (visited[u]) continue;
        visited[u] = true;
        mst_weight += w;

        for (auto [v, edge_w] : adj[u]) {
            if (!visited[v]) {
                pq.push({edge_w, v});
            }
        }
    }

    return mst_weight;
}
```

*Kruskal vs Prim:*
- Kruskal: better for sparse graphs, simpler code, edge-centric
- Prim: better for dense graphs (avoids sorting all edges), vertex-centric
- Both produce same MST weight (not necessarily same edges)

== Topological Sort

*Problem:* Linear ordering of vertices in DAG (Directed Acyclic Graph).

*DFS Approach:* $O(V + E)$

```cpp
bool dfs(int node, const vector<vector<int>>& adj,
         vector<int>& state, vector<int>& topo) {
    // state: 0=unvisited, 1=visiting, 2=visited
    state[node] = 1;

    for (int neighbor : adj[node]) {
        if (state[neighbor] == 1) return false;  // Cycle detected
        if (state[neighbor] == 0 && !dfs(neighbor, adj, state, topo)) {
            return false;
        }
    }

    state[node] = 2;
    topo.push_back(node);  // Add in reverse finish order
    return true;
}

vector<int> topologicalSort(int n, const vector<vector<int>>& adj) {
    vector<int> state(n, 0), topo;

    for (int i = 0; i < n; i++) {
        if (state[i] == 0 && !dfs(i, adj, state, topo)) {
            return {};  // Cycle exists, not a DAG
        }
    }

    reverse(topo.begin(), topo.end());
    return topo;
}
```

*Kahn's Algorithm (BFS):* $O(V + E)$

```cpp
vector<int> kahnTopologicalSort(int n, const vector<vector<int>>& adj) {
    vector<int> indegree(n, 0);

    for (int u = 0; u < n; u++) {
        for (int v : adj[u]) {
            indegree[v]++;
        }
    }

    deque<int> queue;
    for (int i = 0; i < n; i++) {
        if (indegree[i] == 0) {
            queue.push_back(i);
        }
    }

    vector<int> topo;

    while (!queue.empty()) {
        int u = queue.front();
        queue.pop_front();
        topo.push_back(u);

        for (int v : adj[u]) {
            if (--indegree[v] == 0) {
                queue.push_back(v);
            }
        }
    }

    return topo.size() == n ? topo : vector<int>{};  // Empty if cycle
}
```

*DFS vs Kahn:*
- DFS: recursive, natural for finding cycles, reverse finish order
- Kahn: iterative, explicit indegree tracking, forward order
- Cache: Kahn has better locality (BFS queue vs recursion stack)

== Strongly Connected Components (SCC)

=== Kosaraju's Algorithm

*Two-pass DFS:* $O(V + E)$

```cpp
void dfs1(int node, const vector<vector<int>>& adj,
          vector<bool>& visited, vector<int>& finish_order) {
    visited[node] = true;
    for (int neighbor : adj[node]) {
        if (!visited[neighbor]) {
            dfs1(neighbor, adj, visited, finish_order);
        }
    }
    finish_order.push_back(node);
}

void dfs2(int node, const vector<vector<int>>& rev_adj,
          vector<bool>& visited, vector<int>& component, int comp_id) {
    visited[node] = true;
    component[node] = comp_id;

    for (int neighbor : rev_adj[node]) {
        if (!visited[neighbor]) {
            dfs2(neighbor, rev_adj, visited, component, comp_id);
        }
    }
}

vector<int> kosaraju(int n, const vector<vector<int>>& adj) {
    // Build reverse graph
    vector<vector<int>> rev_adj(n);
    for (int u = 0; u < n; u++) {
        for (int v : adj[u]) {
            rev_adj[v].push_back(u);
        }
    }

    // First DFS: compute finish times
    vector<bool> visited(n, false);
    vector<int> finish_order;

    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            dfs1(i, adj, visited, finish_order);
        }
    }

    // Second DFS: find SCCs in reverse finish order
    fill(visited.begin(), visited.end(), false);
    vector<int> component(n);
    int comp_id = 0;

    for (int i = n - 1; i >= 0; i--) {
        int node = finish_order[i];
        if (!visited[node]) {
            dfs2(node, rev_adj, visited, component, comp_id++);
        }
    }

    return component;
}
```

=== Tarjan's Algorithm

*Single-pass DFS with Stack:* $O(V + E)$, more space-efficient

```cpp
class TarjanSCC {
    vector<vector<int>> adj;
    vector<int> ids, low, on_stack, component;
    vector<bool> visited;
    stack<int> stk;
    int id_counter = 0, comp_id = 0;

    void dfs(int node) {
        ids[node] = low[node] = id_counter++;
        stk.push(node);
        on_stack[node] = true;
        visited[node] = true;

        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                dfs(neighbor);
            }
            if (on_stack[neighbor]) {
                low[node] = min(low[node], low[neighbor]);
            }
        }

        // Root of SCC
        if (ids[node] == low[node]) {
            while (true) {
                int v = stk.top();
                stk.pop();
                on_stack[v] = false;
                component[v] = comp_id;
                if (v == node) break;
            }
            comp_id++;
        }
    }

public:
    TarjanSCC(int n, const vector<vector<int>>& graph)
        : adj(graph), ids(n, -1), low(n), on_stack(n, false),
          component(n), visited(n, false) {}

    vector<int> findSCCs() {
        for (int i = 0; i < adj.size(); i++) {
            if (!visited[i]) {
                dfs(i);
            }
        }
        return component;
    }
};
```

*Kosaraju vs Tarjan:*
- Kosaraju: simpler, two passes, requires reverse graph
- Tarjan: one pass, more complex bookkeeping, slightly faster
- Cache: Tarjan better (single traversal vs two)

== Cache-Aware Graph Traversal

*Problem:* Large graphs don't fit in cache. Adjacency list = scattered memory.

*Solution - Graph Reordering:*

```cpp
// Breadth-First Search order relabeling
vector<int> bfs_relabel(int n, const vector<vector<int>>& adj) {
    vector<int> new_id(n, -1);
    deque<int> queue;
    int next_id = 0;

    queue.push_back(0);
    new_id[0] = next_id++;

    while (!queue.empty()) {
        int u = queue.front();
        queue.pop_front();

        for (int v : adj[u]) {
            if (new_id[v] == -1) {
                new_id[v] = next_id++;
                queue.push_back(v);
            }
        }
    }

    return new_id;
}
```

Vertices accessed together (neighbors in BFS) get consecutive IDs = better cache locality.

*Hilbert curve ordering:* For 2D grids as graphs, use space-filling curve to map 2D → 1D preserving locality.

== References

*Algorithms:*

*Dijkstra, E.W. (1959)*. A Note on Two Problems in Connexion with Graphs. Numerische Mathematik 1: 269-271.

*Bellman, R. (1958)*. On a Routing Problem. Quarterly of Applied Mathematics 16: 87-90.

*Floyd, R.W. (1962)*. Algorithm 97: Shortest Path. Communications of the ACM 5(6): 345.

*Kruskal, J.B. (1956)*. On the Shortest Spanning Subtree of a Graph. Proceedings of the AMS 7(1): 48-50.

*Prim, R.C. (1957)*. Shortest Connection Networks And Some Generalizations. Bell System Technical Journal 36(6): 1389-1401.

*Tarjan, R. (1972)*. Depth-First Search and Linear Graph Algorithms. SIAM Journal on Computing 1(2): 146-160.

*Cormen, T.H., Leiserson, C.E., Rivest, R.L., & Stein, C. (2009)*. Introduction to Algorithms (3rd ed.). MIT Press. ISBN 978-0262033848.

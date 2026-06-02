= Network Flows and Matching

*Network flow* is the universal solvent of combinatorial optimization: shortest paths, bipartite matching, project selection, image segmentation, and many scheduling problems all reduce to max-flow or min-cost flow. This chapter presents the modern toolkit: Dinic's algorithm, ISAP, Push-Relabel, the Successive Shortest Paths (SSP) min-cost flow, and the two pillars of bipartite assignment (Hopcroft-Karp and Kuhn-Munkres / Hungarian).

*See also:* _Graphs_, _Advanced Graphs_, _Linear Programming and Simplex_ (LP duality is the source of max-flow min-cut), _Approximation Algorithms_ (LP-rounding uses flows).

== Maximum Flow: Problem Statement

Given a directed graph $G = (V, E)$ with non-negative capacities $c : E -> RR_(>=0)$, a source $s$, and a sink $t$, find a flow $f : E -> RR_(>=0)$ that

- obeys capacity: $0 <= f(e) <= c(e)$,
- conserves flow at every $v != s, t$,
- maximizes $sum_(e "into" t) f(e) - sum_(e "out of" t) f(e)$.

*Max-flow min-cut* (Ford-Fulkerson 1956): the maximum flow value equals the minimum capacity of an $s$-$t$ cut. The two problems are LP duals.

== Edmonds-Karp Baseline

Edmonds-Karp is Ford-Fulkerson with BFS for augmenting paths. Each augmentation pushes at least one unit, and one can show the number of augmentations is $O(V E)$, giving overall $O(V E^2)$.

```cpp
struct EdmondsKarp {
    int n;
    vector<vector<int>> cap;       // capacity matrix
    vector<vector<int>> adj;
    EdmondsKarp(int n) : n(n), cap(n, vector<int>(n, 0)), adj(n) {}
    void add(int u, int v, int c) {
        cap[u][v] += c;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    int bfs(int s, int t, vector<int>& parent) {
        fill(parent.begin(), parent.end(), -1);
        parent[s] = s;
        queue<pair<int,int>> q; q.push({s, INT_MAX});
        while (!q.empty()) {
            auto [u, flow] = q.front(); q.pop();
            for (int v : adj[u]) if (parent[v] == -1 && cap[u][v] > 0) {
                parent[v] = u;
                int nf = min(flow, cap[u][v]);
                if (v == t) return nf;
                q.push({v, nf});
            }
        }
        return 0;
    }
    int maxflow(int s, int t) {
        int flow = 0, df;
        vector<int> parent(n);
        while ((df = bfs(s, t, parent)) > 0) {
            flow += df;
            int v = t;
            while (v != s) { int u = parent[v]; cap[u][v] -= df; cap[v][u] += df; v = u; }
        }
        return flow;
    }
};
```

Use only for tiny graphs or as a correctness oracle when testing faster solvers.

== Dinic's Algorithm

Dinic builds a *level graph* with BFS from $s$, then finds blocking flows with DFS. The level graph forbids backward or same-level edges, so each blocking-flow phase strictly increases the shortest $s$-$t$ distance. There are at most $V$ phases, so the total cost is $O(V^2 E)$ in general and $O(E sqrt(V))$ on unit-capacity networks (and bipartite matching). For graphs with integer capacities, Gabow's capacity-scaling variant achieves $O(E^2 log U)$ where $U$ is the maximum capacity, which is preferable when $U$ is small.

```cpp
struct Dinic {
    struct Edge { int to, rev; long long cap; };
    int n;
    vector<vector<Edge>> g;
    vector<int> level, iter;
    Dinic(int n) : n(n), g(n), level(n), iter(n) {}
    void add(int u, int v, long long c) {
        g[u].push_back({v, (int)g[v].size(), c});
        g[v].push_back({u, (int)g[u].size() - 1, 0});
    }
    bool bfs(int s, int t) {
        fill(level.begin(), level.end(), -1);
        queue<int> q; level[s] = 0; q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (auto& e : g[u]) if (e.cap > 0 && level[e.to] < 0) {
                level[e.to] = level[u] + 1; q.push(e.to);
            }
        }
        return level[t] >= 0;
    }
    long long dfs(int u, int t, long long pushed) {
        if (u == t || pushed == 0) return pushed;
        for (int& i = iter[u]; i < (int)g[u].size(); ++i) {
            auto& e = g[u][i];
            if (e.cap > 0 && level[e.to] == level[u] + 1) {
                long long d = dfs(e.to, t, min(pushed, e.cap));
                if (d > 0) { e.cap -= d; g[e.to][e.rev].cap += d; return d; }
            }
        }
        return 0;
    }
    long long maxflow(int s, int t) {
        long long flow = 0;
        while (bfs(s, t)) {
            fill(iter.begin(), iter.end(), 0);
            while (long long f = dfs(s, t, LLONG_MAX)) flow += f;
        }
        return flow;
    }
};
```

*Implementation tips:*
- Store edges in a flat `vector<Edge>` with `rev` index pointers; pushing on `e` decreases `e.cap` and increases `g[e.to][e.rev].cap`.
- The `iter[u]` "current arc" pointer is critical: it is what makes the DFS amortised, not quadratic.

=== ISAP (Improved Shortest Augmenting Path)

ISAP avoids the full BFS between phases of Dinic by maintaining distance labels and a *gap heuristic*: when no vertex has a particular label $d$, all vertices with label $>= d$ are disconnected from the sink and can be skipped. ISAP runs in $O(V^2 E)$ but is typically 2-5× faster than Dinic on dense graphs and is the algorithm of choice in many competitive programming flow templates.

*Sketch:* Initialise heights with a reverse BFS from $t$. Augment along admissible edges $(u, v)$ where $h[u] = h[v] + 1$. When stuck at $u$, *retreat*: set $h[u] = 1 + min{h[v] : (u,v) "has capacity"}$. Detect gaps; terminate when $h[s] >= n$.

== Push-Relabel

Goldberg and Tarjan's push-relabel maintains a *preflow* (excess allowed at internal nodes) rather than a feasible flow.

*Operations:*
- *Push* along $(u, v)$ admissible: $h[u] = h[v] + 1$ and residual $> 0$. Send $min("excess"[u], r(u,v))$.
- *Relabel* $u$ when it has excess but no admissible outgoing edge: $h[u] <- 1 + min{h[v] : r(u,v) > 0}$.

*Complexity:* $O(V^2 sqrt(E))$ with the highest-label rule, $O(V^3)$ in general. The *gap* and *global relabel* heuristics are essential in practice; without them push-relabel is often slower than Dinic.

```cpp
struct PushRelabel {
    struct Edge { int to, rev; long long cap; };
    int n; vector<vector<Edge>> g;
    vector<long long> excess; vector<int> height, cnt; vector<bool> active;
    vector<queue<int>> B; int b;
    PushRelabel(int n) : n(n), g(n), excess(n), height(n), cnt(2*n+1), active(n), B(2*n+1), b(0) {}
    void add(int u, int v, long long c) {
        g[u].push_back({v, (int)g[v].size(), c});
        g[v].push_back({u, (int)g[u].size() - 1, 0});
    }
    void enqueue(int v) { if (!active[v] && excess[v] > 0 && height[v] < n) { active[v] = true; B[height[v]].push(v); b = max(b, height[v]); } }
    void push(int u, Edge& e) {
        long long d = min(excess[u], e.cap);
        if (d && height[u] == height[e.to] + 1) {
            e.cap -= d; g[e.to][e.rev].cap += d;
            excess[u] -= d; excess[e.to] += d; enqueue(e.to);
        }
    }
    void gap(int k) {
        for (int v = 0; v < n; ++v) if (height[v] >= k) {
            --cnt[height[v]]; height[v] = max(height[v], n + 1);
            ++cnt[height[v]]; enqueue(v);
        }
    }
    void relabel(int u) {
        --cnt[height[u]]; height[u] = 2 * n;
        for (auto& e : g[u]) if (e.cap) height[u] = min(height[u], height[e.to] + 1);
        ++cnt[height[u]]; enqueue(u);
    }
    void discharge(int u) {
        for (auto& e : g[u]) { if (excess[u] <= 0) break; push(u, e); }
        if (excess[u] > 0) {
            if (cnt[height[u]] == 1) gap(height[u]); else relabel(u);
        }
    }
    long long maxflow(int s, int t) {
        height[s] = n; active[s] = active[t] = true; cnt[0] = n - 1; cnt[n] = 1;
        for (auto& e : g[s]) { excess[s] += e.cap; push(s, e); }
        while (b >= 0) {
            if (B[b].empty()) { --b; continue; }
            int u = B[b].front(); B[b].pop(); active[u] = false;
            discharge(u);
        }
        return excess[t];
    }
};
```

=== When to use which

#table(
  columns: (auto, auto, auto),
  [*Algorithm*], [*Worst case*], [*Best for*],
  [Edmonds-Karp], [$O(V E^2)$], [Tiny graphs, oracle],
  [Dinic], [$O(V^2 E)$, $O(E sqrt(V))$ unit], [General purpose, bipartite],
  [ISAP], [$O(V^2 E)$], [Dense graphs, contests],
  [Push-Relabel], [$O(V^2 sqrt(E))$], [Very dense, image segmentation],
)

== Min-Cost Max-Flow

Add a cost $w(e) >= 0$ per unit on each edge. Find max flow of minimum cost. The classical method is *Successive Shortest Paths* (SSP): repeatedly augment along the cheapest $s$-$t$ path in the residual graph (Bellman-Ford or Johnson-reweighted Dijkstra). With Johnson potentials each iteration is $O((V + E) log V)$; total work is $O(F (V + E) log V)$ for max-flow value $F$.

```cpp
struct MCMF {
    struct Edge { int to, rev; long long cap, cost; };
    int n; vector<vector<Edge>> g; vector<long long> h, dist; vector<int> pv, pe; vector<bool> in_q;
    MCMF(int n) : n(n), g(n), h(n), dist(n), pv(n), pe(n), in_q(n) {}
    void add(int u, int v, long long cap, long long cost) {
        g[u].push_back({v, (int)g[v].size(), cap, cost});
        g[v].push_back({u, (int)g[u].size() - 1, 0, -cost});
    }
    pair<long long,long long> run(int s, int t) {
        long long flow = 0, cost = 0;
        fill(h.begin(), h.end(), 0);
        // (Optional Bellman-Ford for negative edges; omitted: assume non-negative initial costs.)
        while (true) {
            priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
            fill(dist.begin(), dist.end(), LLONG_MAX);
            dist[s] = 0; pq.push({0, s});
            while (!pq.empty()) {
                auto [d, u] = pq.top(); pq.pop();
                if (d > dist[u]) continue;
                for (int i = 0; i < (int)g[u].size(); ++i) {
                    auto& e = g[u][i];
                    if (e.cap > 0) {
                        long long nd = d + e.cost + h[u] - h[e.to];
                        if (nd < dist[e.to]) { dist[e.to] = nd; pv[e.to] = u; pe[e.to] = i; pq.push({nd, e.to}); }
                    }
                }
            }
            if (dist[t] == LLONG_MAX) break;
            for (int i = 0; i < n; ++i) if (dist[i] < LLONG_MAX) h[i] += dist[i];
            long long aug = LLONG_MAX;
            for (int v = t; v != s; v = pv[v]) aug = min(aug, g[pv[v]][pe[v]].cap);
            for (int v = t; v != s; v = pv[v]) {
                auto& e = g[pv[v]][pe[v]];
                e.cap -= aug; g[v][e.rev].cap += aug;
            }
            flow += aug; cost += aug * h[t];
        }
        return {flow, cost};
    }
};
```

*Alternatives:* *Cycle-cancelling* (Klein) repeatedly cancels negative-cost cycles in the residual graph; *Network Simplex* is strongly polynomial and outperforms SSP on transportation-style instances (the OR-tools and LEMON libraries use it).

== Bipartite Matching

A *matching* in a bipartite graph $G = (L union.sq R, E)$ is a set of edges with no shared endpoint. Maximum matching reduces to max-flow: add $s -> L$ and $R -> t$ unit-capacity edges and run Dinic.

=== Hopcroft-Karp

For *unweighted* bipartite matching, Hopcroft-Karp runs in $O(E sqrt(V))$ by augmenting along multiple shortest augmenting paths per phase (the BFS-then-DFS structure is exactly Dinic on unit-capacity graphs).

```cpp
struct HopcroftKarp {
    int nL, nR;
    vector<vector<int>> adj;     // adj[u] for u in L
    vector<int> pairL, pairR, dist;
    static const int NIL = 0, INF_D = INT_MAX;
    HopcroftKarp(int nL, int nR) : nL(nL), nR(nR), adj(nL + 1), pairL(nL + 1, NIL), pairR(nR + 1, NIL), dist(nL + 1) {}
    void add(int u, int v) { adj[u].push_back(v); }  // u in [1..nL], v in [1..nR]
    bool bfs() {
        queue<int> q;
        for (int u = 1; u <= nL; ++u) {
            if (pairL[u] == NIL) { dist[u] = 0; q.push(u); } else dist[u] = INF_D;
        }
        bool found = false;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : adj[u]) {
                int pu = pairR[v];
                if (pu == NIL) found = true;
                else if (dist[pu] == INF_D) { dist[pu] = dist[u] + 1; q.push(pu); }
            }
        }
        return found;
    }
    bool dfs(int u) {
        for (int v : adj[u]) {
            int pu = pairR[v];
            if (pu == NIL || (dist[pu] == dist[u] + 1 && dfs(pu))) {
                pairL[u] = v; pairR[v] = u; return true;
            }
        }
        dist[u] = INF_D; return false;
    }
    int matching() {
        int m = 0;
        while (bfs()) for (int u = 1; u <= nL; ++u) if (pairL[u] == NIL && dfs(u)) ++m;
        return m;
    }
};
```

=== Kuhn-Munkres (Hungarian Algorithm)

For *weighted* bipartite matching (assignment problem), Kuhn-Munkres finds a perfect matching of maximum (or minimum) total weight in $O(n^3)$. Maintain dual potentials $u_i$ (rows) and $v_j$ (cols) with $u_i + v_j >= w_(i j)$. Equality subgraph contains tight edges; augment in it; when stuck, update potentials by the slack $delta = min(u_i + v_j - w_(i j))$ over $i$ in tree, $j$ outside.

```cpp
// O(n^3) Hungarian for square cost matrix, returns assignment (a[j] = i)
// minimises sum of a[j]->j costs. Use INT_MAX/2 padding for rectangular cases.
vector<int> hungarian(const vector<vector<int>>& a) {
    int n = a.size(), m = a[0].size();
    vector<int> u(n+1), v(m+1), p(m+1), way(m+1);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        vector<int> minv(m+1, INT_MAX);
        vector<bool> used(m+1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], delta = INT_MAX, j1 = -1;
            for (int j = 1; j <= m; ++j) if (!used[j]) {
                int cur = a[i0-1][j-1] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else         { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    vector<int> assign(m+1, -1);
    for (int j = 1; j <= m; ++j) assign[j] = p[j];
    return assign;
}
```

*Applications:* job scheduling, image registration, sensor-to-target assignment, MOT data association (the Hungarian step in trackers like SORT).

== Reductions Worth Knowing

- *Vertex capacities:* split $v$ into $v_("in") -> v_("out")$ with the capacity on the internal edge.
- *Multiple sources/sinks:* add super-source / super-sink.
- *Edge-disjoint paths:* unit capacities; Menger's theorem.
- *Minimum vertex cover (bipartite) = maximum matching* (König's theorem).
- *Maximum independent set (bipartite)* $= n -$ max matching.
- *Project selection / closure:* min-cut on a bipartite-like construction (Hochbaum).
- *Image segmentation:* model pixels as a graph; min-cut gives MAP estimate for binary MRFs.

== Complexity Summary

#table(
  columns: (auto, auto, auto),
  [*Problem*], [*Best algorithm*], [*Complexity*],
  [Max flow (general)], [Dinic / ISAP / Push-Relabel], [$O(V^2 E)$ – $O(V E log V)$ in practice],
  [Max flow (unit cap.)], [Dinic], [$O(E sqrt(V))$],
  [Min-cost max flow], [SSP + potentials], [$O(F (V+E) log V)$],
  [Min-cost flow (strongly poly.)], [Network simplex / Orlin], [$O(E^2 log V)$ (Orlin)],
  [Bipartite matching], [Hopcroft-Karp], [$O(E sqrt(V))$],
  [Weighted bipartite matching], [Hungarian], [$O(n^3)$],
  [Global min cut (undirected)], [Stoer-Wagner / Karger-Stein], [$O(V E + V^2 log V)$ / $O(n^2 log^3 n)$],
)

== Further Reading

*Ahuja, R.K., Magnanti, T.L. & Orlin, J.B. (1993).* Network Flows: Theory, Algorithms, and Applications. Prentice Hall. ISBN 0-13-617549-X. The definitive treatise.

*Goldberg, A.V. & Tarjan, R.E. (1988).* A New Approach to the Maximum-Flow Problem. JACM 35(4): 921-940. Push-relabel.

*Dinic, E.A. (1970).* Algorithm for Solution of a Problem of Maximum Flow in a Network with Power Estimation. Soviet Math. Doklady 11: 1277-1280.

*Hopcroft, J.E. & Karp, R.M. (1973).* An $n^(5/2)$ Algorithm for Maximum Matchings in Bipartite Graphs. SIAM J. Computing 2(4): 225-231.

*Kuhn, H.W. (1955).* The Hungarian Method for the Assignment Problem. Naval Research Logistics Quarterly 2: 83-97.

*Orlin, J.B. (2013).* Max Flows in $O(n m)$ Time, or Better. STOC 2013.

*Cormen, T.H., Leiserson, C.E., Rivest, R.L. & Stein, C. (2022).* Introduction to Algorithms, 4th ed. MIT Press. Chapters on flow networks.

*Schrijver, A. (2003).* Combinatorial Optimization: Polyhedra and Efficiency. Springer. Three-volume reference for the theory.

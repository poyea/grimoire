= Graphs

*Representation tradeoffs:*
- *Adjacency matrix:* $O(V^2)$ space. O(1) edge lookup. Cache-friendly for dense graphs. Row-major iteration optimal.
- *Adjacency list:* $O(V + E)$ space. O(degree) edge lookup. Better for sparse graphs. Pointer chasing = cache-unfriendly.
- *Edge list:* $O(E)$ space. Used for Kruskal's, union-find. Must sort for many algorithms.

```cpp
// Adjacency list: vector of vectors (cache-friendly)
vector<vector<int>> adj(n);

// Alternative: vector of unordered_set (fast removal, more overhead)
vector<unordered_set<int>> adj(n);
```

== Number of Islands

*Problem:* Count number of islands in 2D grid ('1' = land, '0' = water).

*Approach - BFS:* $O(n m)$ time, $O(n m)$ space

```cpp
int numIslands(vector<vector<char>>& grid) {
    int rows = grid.size(), cols = grid[0].size();
    int islands = 0;

    auto bfs = [&](int r, int c) {
        deque<pair<int, int>> queue = {{r, c}};
        grid[r][c] = '0';  // Mark visited in-place

        constexpr int dirs[4][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}};

        while (!queue.empty()) {
            auto [row, col] = queue.front();
            queue.pop_front();

            for (auto [dr, dc] : dirs) {
                int nr = row + dr, nc = col + dc;
                if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] == '1') {
                    grid[nr][nc] = '0';
                    queue.push_back({nr, nc});
                }
            }
        }
    };

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (grid[r][c] == '1') {
                islands++;
                bfs(r, c);
            }
        }
    }
    return islands;
}
```

*2D grid access patterns:*
- Row-major iteration (`grid[r][c]`): cache-friendly, prefetcher works
- Random neighbor access: depends on island shape, generally poor locality
- In-place marking eliminates `unordered_set<pair<int,int>>`: saves $#sym.tilde.op$40 bytes per cell + hash overhead

*DFS vs BFS:*
- DFS: O(nm) stack depth worst case (snake island) = stack overflow risk
- BFS: O(nm) queue size worst case (diagonal island)
- Cache: both have poor locality due to random neighbor access

== Clone Graph

*Problem:* Deep copy an undirected graph.

*Approach - DFS with HashMap:* $O(V + E)$ time, $O(V)$ space

```cpp
class Node {
public:
    int val;
    vector<Node*> neighbors;
    Node(int _val) : val(_val) {}
};

Node* cloneGraph(Node* node) {
    if (!node) return nullptr;

    unordered_map<Node*, Node*> cloned;

    function<Node*(Node*)> dfs = [&](Node* curr) -> Node* {
        if (cloned.count(curr)) return cloned[curr];

        Node* copy = new Node(curr->val);
        cloned[curr] = copy;  // Must add before recursing to handle cycles

        for (Node* neighbor : curr->neighbors) {
            copy->neighbors.push_back(dfs(neighbor));
        }
        return copy;
    };

    return dfs(node);
}
```

*Hash map performance:*
- `unordered_map<Node*, Node*>`: pointer keys = fast hash (identity)
- Load factor: default 1.0. Rehashing at 0.75-0.9 = better cache hit rate
- `reserve(V)` if vertex count known: avoids rehash overhead

== Course Schedule (Cycle Detection)

*Problem:* Determine if you can finish all courses given prerequisites (detect cycle in directed graph).

*Approach - DFS with Coloring:* $O(V + E)$ time, $O(V)$ space

```cpp
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>> adj(numCourses);
    for (auto& edge : prerequisites) {
        adj[edge[0]].push_back(edge[1]);
    }

    enum State { UNVISITED, VISITING, VISITED };
    vector<State> state(numCourses, UNVISITED);

    function<bool(int)> hasCycle = [&](int node) -> bool {
        if (state[node] == VISITING) return true;  // Back edge = cycle
        if (state[node] == VISITED) return false;  // Already processed

        state[node] = VISITING;
        for (int neighbor : adj[node]) {
            if (hasCycle(neighbor)) return true;
        }
        state[node] = VISITED;
        return false;
    };

    for (int i = 0; i < numCourses; i++) {
        if (state[i] == UNVISITED && hasCycle(i)) {
            return false;
        }
    }
    return true;
}
```

*Three-color algorithm:*
- UNVISITED (white): not seen
- VISITING (gray): in current DFS path
- VISITED (black): finished processing
- Cycle exists iff we reach VISITING node (back edge)

*State representation:*
- `vector<State>` vs `vector<int>`: enum is self-documenting, compiles to same code
- Could use 2 bits per node: `vector<uint8_t> state((n+3)/4)` for memory efficiency

== Pacific Atlantic Water Flow

*Problem:* Find cells where water can flow to both Pacific and Atlantic oceans.

*Approach - Reverse DFS from Oceans:* $O(n m)$ time, $O(n m)$ space

```cpp
vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
    int rows = heights.size(), cols = heights[0].size();

    vector<vector<bool>> pacific(rows, vector<bool>(cols));
    vector<vector<bool>> atlantic(rows, vector<bool>(cols));

    function<void(int, int, vector<vector<bool>>&)> dfs =
        [&](int r, int c, vector<vector<bool>>& ocean) {
        ocean[r][c] = true;

        constexpr int dirs[4][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}};
        for (auto [dr, dc] : dirs) {
            int nr = r + dr, nc = c + dc;
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols &&
                !ocean[nr][nc] && heights[nr][nc] >= heights[r][c]) {
                dfs(nr, nc, ocean);
            }
        }
    };

    // Start from borders
    for (int i = 0; i < rows; i++) {
        dfs(i, 0, pacific);
        dfs(i, cols - 1, atlantic);
    }
    for (int j = 0; j < cols; j++) {
        dfs(0, j, pacific);
        dfs(rows - 1, j, atlantic);
    }

    // Find intersection
    vector<vector<int>> result;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (pacific[r][c] && atlantic[r][c]) {
                result.push_back({r, c});
            }
        }
    }
    return result;
}
```

*Memory optimization:*
`vector<vector<bool>>` contains bit-packed `vector<bool>` objects. The inner `vector<bool>` uses 1 bit per element but has slower access due to bit manipulation (masking and shifting). Use `vector<vector<uint8_t>>` for 8x memory but $#sym.tilde.op$2-3x faster access.

*Bitset alternative:*
```cpp
bitset<10000> pacific, atlantic;  // For max grid size 100x100
int idx = r * cols + c;
pacific[idx] = true;
```
Faster bitwise AND for intersection.

*Cache:* Row-major iteration in result collection = good spatial locality. DFS traversal = poor locality (random access pattern).

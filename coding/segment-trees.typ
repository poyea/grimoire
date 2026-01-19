= Segment Trees and Range Queries

*Efficient range operations:* Segment trees and Fenwick trees enable $O(log n)$ range queries and point updates on arrays. Essential for competitive programming and database indexing [Bentley 1977].

*See also:* Trees (for basic tree traversal), Bit Manipulation (for Fenwick tree bit tricks), Dynamic Programming (for range DP)

== Segment Tree Fundamentals

*Problem:* Given array, answer range queries (sum, min, max) and handle point updates efficiently.

*Naive approach:* Query $O(n)$, Update $O(1)$
*Prefix sum:* Query $O(1)$, Update $O(n)$
*Segment tree:* Query $O(log n)$, Update $O(log n)$

*Structure:* Complete binary tree where each node represents an interval.

```cpp
class SegmentTree {
    vector<int64_t> tree;
    int n;

    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            build(arr, 2 * node, start, mid);
            build(arr, 2 * node + 1, mid + 1, end);
            tree[node] = tree[2 * node] + tree[2 * node + 1];
        }
    }

public:
    SegmentTree(const vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n);  // Safe upper bound
        build(arr, 1, 0, n - 1);
    }

    // Point update: set arr[idx] = val
    void update(int node, int start, int end, int idx, int val) {
        if (start == end) {
            tree[node] = val;
        } else {
            int mid = (start + end) / 2;
            if (idx <= mid) {
                update(2 * node, start, mid, idx, val);
            } else {
                update(2 * node + 1, mid + 1, end, idx, val);
            }
            tree[node] = tree[2 * node] + tree[2 * node + 1];
        }
    }

    void update(int idx, int val) {
        update(1, 0, n - 1, idx, val);
    }

    // Range sum query [l, r]
    int64_t query(int node, int start, int end, int l, int r) {
        if (r < start || end < l) {
            return 0;  // Identity for sum
        }
        if (l <= start && end <= r) {
            return tree[node];  // Fully contained
        }
        int mid = (start + end) / 2;
        return query(2 * node, start, mid, l, r) +
               query(2 * node + 1, mid + 1, end, l, r);
    }

    int64_t query(int l, int r) {
        return query(1, 0, n - 1, l, r);
    }
};
```

*Space:* $O(4n)$ worst case (can be $O(2n)$ with careful indexing)

*Time:*
- Build: $O(n)$
- Query: $O(log n)$
- Update: $O(log n)$

*Cache behavior:* Tree stored in array = reasonable locality. Parent-child access = index arithmetic, no pointer chasing.

== Iterative Segment Tree

*Faster in practice:* Eliminates recursion overhead, better branch prediction.

```cpp
class IterativeSegmentTree {
    vector<int64_t> tree;
    int n;

public:
    IterativeSegmentTree(const vector<int>& arr) {
        n = arr.size();
        tree.resize(2 * n);

        // Leaves
        for (int i = 0; i < n; i++) {
            tree[n + i] = arr[i];
        }

        // Internal nodes (build bottom-up)
        for (int i = n - 1; i > 0; i--) {
            tree[i] = tree[2 * i] + tree[2 * i + 1];
        }
    }

    void update(int idx, int val) {
        idx += n;  // Leaf position
        tree[idx] = val;

        // Propagate up
        while (idx > 1) {
            idx /= 2;
            tree[idx] = tree[2 * idx] + tree[2 * idx + 1];
        }
    }

    int64_t query(int l, int r) {  // [l, r)
        int64_t res = 0;
        l += n;
        r += n;

        while (l < r) {
            if (l & 1) res += tree[l++];  // l is right child
            if (r & 1) res += tree[--r];  // r is right child
            l /= 2;
            r /= 2;
        }
        return res;
    }
};
```

*Key insight:* Bottom-up traversal. Query processes left and right boundaries moving toward root.

*Performance:* 2-3x faster than recursive version (no function call overhead).

== Lazy Propagation

*Problem:* Range updates (add val to all elements in [l, r]).

*Naive segment tree:* Range update = $O(n)$ (update each element)

*Lazy propagation:* Defer updates to children until needed. Range update = $O(log n)$.

```cpp
class LazySegmentTree {
    vector<int64_t> tree, lazy;
    int n;

    void push(int node, int start, int end) {
        if (lazy[node] != 0) {
            tree[node] += lazy[node] * (end - start + 1);

            if (start != end) {  // Not a leaf
                lazy[2 * node] += lazy[node];
                lazy[2 * node + 1] += lazy[node];
            }
            lazy[node] = 0;
        }
    }

    void updateRange(int node, int start, int end, int l, int r, int val) {
        push(node, start, end);

        if (r < start || end < l) return;

        if (l <= start && end <= r) {
            lazy[node] += val;
            push(node, start, end);
            return;
        }

        int mid = (start + end) / 2;
        updateRange(2 * node, start, mid, l, r, val);
        updateRange(2 * node + 1, mid + 1, end, l, r, val);
        tree[node] = tree[2 * node] + tree[2 * node + 1];
    }

    int64_t queryRange(int node, int start, int end, int l, int r) {
        push(node, start, end);

        if (r < start || end < l) return 0;
        if (l <= start && end <= r) return tree[node];

        int mid = (start + end) / 2;
        return queryRange(2 * node, start, mid, l, r) +
               queryRange(2 * node + 1, mid + 1, end, l, r);
    }

public:
    LazySegmentTree(const vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n);
        lazy.resize(4 * n, 0);
        build(arr, 1, 0, n - 1);
    }

    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            build(arr, 2 * node, start, mid);
            build(arr, 2 * node + 1, mid + 1, end);
            tree[node] = tree[2 * node] + tree[2 * node + 1];
        }
    }

    void update(int l, int r, int val) {
        updateRange(1, 0, n - 1, l, r, val);
    }

    int64_t query(int l, int r) {
        return queryRange(1, 0, n - 1, l, r);
    }
};
```

*Complexity with lazy propagation:*
- Range update: $O(log n)$
- Range query: $O(log n)$

*Use cases:*
- Range add, range sum
- Range set, range min/max
- Any associative operation with composable updates

== Fenwick Tree (Binary Indexed Tree)

*More space-efficient:* Uses bit manipulation for implicit tree structure.

*Key property:* `i & -i` isolates lowest set bit.

```cpp
class FenwickTree {
    vector<int64_t> tree;
    int n;

public:
    FenwickTree(int size) : n(size), tree(size + 1, 0) {}

    FenwickTree(const vector<int>& arr) : FenwickTree(arr.size()) {
        for (int i = 0; i < arr.size(); i++) {
            add(i + 1, arr[i]);
        }
    }

    // Add val to index i (1-indexed)
    void add(int i, int64_t val) {
        while (i <= n) {
            tree[i] += val;
            i += i & -i;  // Move to parent
        }
    }

    // Prefix sum [1, i]
    int64_t sum(int i) {
        int64_t result = 0;
        while (i > 0) {
            result += tree[i];
            i -= i & -i;  // Move to predecessor
        }
        return result;
    }

    // Range sum [l, r] (1-indexed)
    int64_t range(int l, int r) {
        return sum(r) - sum(l - 1);
    }

    // Point query (value at index i)
    int64_t get(int i) {
        return range(i, i);
    }
};
```

*Complexity:*
- Space: $O(n)$ (half of segment tree)
- Update: $O(log n)$
- Query: $O(log n)$

*Why `i & -i` works:*
- `-i` in two's complement = flip bits and add 1
- `i & -i` gives lowest set bit
- This bit pattern encodes the tree structure

Example: i = 6 = 0b110
- `-6` = 0b...11111010
- `6 & -6` = 0b010 = 2

== Fenwick Tree: Range Update, Point Query

*Trick:* Store differences instead of values.

```cpp
class FenwickTreeRangeUpdate {
    vector<int64_t> tree;
    int n;

    void add(int i, int64_t val) {
        while (i <= n) {
            tree[i] += val;
            i += i & -i;
        }
    }

    int64_t sum(int i) {
        int64_t result = 0;
        while (i > 0) {
            result += tree[i];
            i -= i & -i;
        }
        return result;
    }

public:
    FenwickTreeRangeUpdate(int size) : n(size), tree(size + 1, 0) {}

    // Add val to range [l, r]
    void rangeAdd(int l, int r, int64_t val) {
        add(l, val);
        add(r + 1, -val);
    }

    // Point query: value at index i
    int64_t get(int i) {
        return sum(i);
    }
};
```

*Insight:* Adding val at l and -val at r+1 creates a "pulse" that affects exactly [l, r] in prefix sums.

== Fenwick Tree: Range Update, Range Query

*Two Fenwick trees:* Combine techniques for full range operations.

```cpp
class FenwickTreeFull {
    vector<int64_t> tree1, tree2;
    int n;

    void add(vector<int64_t>& t, int i, int64_t val) {
        while (i <= n) {
            t[i] += val;
            i += i & -i;
        }
    }

    int64_t sum(vector<int64_t>& t, int i) {
        int64_t result = 0;
        while (i > 0) {
            result += t[i];
            i -= i & -i;
        }
        return result;
    }

    int64_t prefixSum(int i) {
        return sum(tree1, i) * i - sum(tree2, i);
    }

public:
    FenwickTreeFull(int size) : n(size), tree1(size + 1, 0), tree2(size + 1, 0) {}

    void rangeAdd(int l, int r, int64_t val) {
        add(tree1, l, val);
        add(tree1, r + 1, -val);
        add(tree2, l, val * (l - 1));
        add(tree2, r + 1, -val * r);
    }

    int64_t rangeSum(int l, int r) {
        return prefixSum(r) - prefixSum(l - 1);
    }
};
```

*Mathematical derivation:*
After range add of val to [l, r]:
$
"sum"(1, i) = sum_(j=1)^i a_j + "contributions from updates"
$

The two-tree approach tracks: $sum "val"$ and $sum "val" times j$ to reconstruct prefix sums.

== 2D Fenwick Tree

*Problem:* 2D range sum queries and point updates.

```cpp
class FenwickTree2D {
    vector<vector<int64_t>> tree;
    int n, m;

public:
    FenwickTree2D(int rows, int cols) : n(rows), m(cols) {
        tree.assign(n + 1, vector<int64_t>(m + 1, 0));
    }

    void add(int x, int y, int64_t val) {
        for (int i = x; i <= n; i += i & -i) {
            for (int j = y; j <= m; j += j & -j) {
                tree[i][j] += val;
            }
        }
    }

    int64_t sum(int x, int y) {
        int64_t result = 0;
        for (int i = x; i > 0; i -= i & -i) {
            for (int j = y; j > 0; j -= j & -j) {
                result += tree[i][j];
            }
        }
        return result;
    }

    // Rectangle sum [(x1,y1), (x2,y2)]
    int64_t rectSum(int x1, int y1, int x2, int y2) {
        return sum(x2, y2) - sum(x1 - 1, y2) -
               sum(x2, y1 - 1) + sum(x1 - 1, y1 - 1);
    }
};
```

*Complexity:*
- Space: $O(n m)$
- Update: $O(log n times log m)$
- Query: $O(log n times log m)$

== Segment Tree with Multiple Operations

*Min/Max Segment Tree:*

```cpp
class MinSegmentTree {
    vector<int> tree;
    int n;
    static const int INF = INT_MAX;

    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            build(arr, 2 * node, start, mid);
            build(arr, 2 * node + 1, mid + 1, end);
            tree[node] = min(tree[2 * node], tree[2 * node + 1]);
        }
    }

public:
    MinSegmentTree(const vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n, INF);
        build(arr, 1, 0, n - 1);
    }

    void update(int node, int start, int end, int idx, int val) {
        if (start == end) {
            tree[node] = val;
        } else {
            int mid = (start + end) / 2;
            if (idx <= mid) {
                update(2 * node, start, mid, idx, val);
            } else {
                update(2 * node + 1, mid + 1, end, idx, val);
            }
            tree[node] = min(tree[2 * node], tree[2 * node + 1]);
        }
    }

    int query(int node, int start, int end, int l, int r) {
        if (r < start || end < l) return INF;
        if (l <= start && end <= r) return tree[node];

        int mid = (start + end) / 2;
        return min(query(2 * node, start, mid, l, r),
                   query(2 * node + 1, mid + 1, end, l, r));
    }
};
```

== Persistent Segment Tree

*Problem:* Maintain all historical versions of segment tree.

*Use case:* Range query on subarray [versions l to r].

```cpp
class PersistentSegmentTree {
    struct Node {
        int64_t val;
        int left, right;  // Indices to children (not pointers)
    };

    vector<Node> nodes;
    vector<int> roots;  // Root of each version
    int n;

    int build(const vector<int>& arr, int start, int end) {
        int node = nodes.size();
        nodes.push_back({0, -1, -1});

        if (start == end) {
            nodes[node].val = arr[start];
        } else {
            int mid = (start + end) / 2;
            nodes[node].left = build(arr, start, mid);
            nodes[node].right = build(arr, mid + 1, end);
            nodes[node].val = nodes[nodes[node].left].val +
                              nodes[nodes[node].right].val;
        }
        return node;
    }

    int update(int prev, int start, int end, int idx, int val) {
        int node = nodes.size();
        nodes.push_back(nodes[prev]);  // Copy previous node

        if (start == end) {
            nodes[node].val = val;
        } else {
            int mid = (start + end) / 2;
            if (idx <= mid) {
                nodes[node].left = update(nodes[prev].left, start, mid, idx, val);
            } else {
                nodes[node].right = update(nodes[prev].right, mid + 1, end, idx, val);
            }
            nodes[node].val = nodes[nodes[node].left].val +
                              nodes[nodes[node].right].val;
        }
        return node;
    }

    int64_t query(int node, int start, int end, int l, int r) {
        if (r < start || end < l || node == -1) return 0;
        if (l <= start && end <= r) return nodes[node].val;

        int mid = (start + end) / 2;
        return query(nodes[node].left, start, mid, l, r) +
               query(nodes[node].right, mid + 1, end, l, r);
    }

public:
    PersistentSegmentTree(const vector<int>& arr) {
        n = arr.size();
        nodes.reserve(n * 40);  // ~40 nodes per version for log n updates
        roots.push_back(build(arr, 0, n - 1));
    }

    void update(int idx, int val) {
        int newRoot = update(roots.back(), 0, n - 1, idx, val);
        roots.push_back(newRoot);
    }

    int64_t query(int version, int l, int r) {
        return query(roots[version], 0, n - 1, l, r);
    }

    int numVersions() const { return roots.size(); }
};
```

*Space:* $O(n + q log n)$ where q = number of updates

*Key insight:* Only $O(log n)$ nodes change per update. Share unchanged subtrees.

== Merge Sort Tree

*Problem:* Count elements in range [l, r] that are $<= k$.

```cpp
class MergeSortTree {
    vector<vector<int>> tree;
    int n;

    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = {arr[start]};
        } else {
            int mid = (start + end) / 2;
            build(arr, 2 * node, start, mid);
            build(arr, 2 * node + 1, mid + 1, end);

            // Merge sorted arrays
            merge(tree[2 * node].begin(), tree[2 * node].end(),
                  tree[2 * node + 1].begin(), tree[2 * node + 1].end(),
                  back_inserter(tree[node]));
        }
    }

public:
    MergeSortTree(const vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n);
        build(arr, 1, 0, n - 1);
    }

    // Count elements <= k in range [l, r]
    int countLessEqual(int node, int start, int end, int l, int r, int k) {
        if (r < start || end < l) return 0;
        if (l <= start && end <= r) {
            return upper_bound(tree[node].begin(), tree[node].end(), k) -
                   tree[node].begin();
        }

        int mid = (start + end) / 2;
        return countLessEqual(2 * node, start, mid, l, r, k) +
               countLessEqual(2 * node + 1, mid + 1, end, l, r, k);
    }

    int query(int l, int r, int k) {
        return countLessEqual(1, 0, n - 1, l, r, k);
    }
};
```

*Complexity:*
- Build: $O(n log n)$
- Space: $O(n log n)$
- Query: $O(log^2 n)$

== Comparison of Range Query Structures

#table(
  columns: 5,
  align: (left, center, center, center, left),
  table.header([Structure], [Build], [Point Update], [Range Query], [Features]),
  [Prefix Sum], [$O(n)$], [$O(n)$], [$O(1)$], [Static only],
  [Segment Tree], [$O(n)$], [$O(log n)$], [$O(log n)$], [General purpose],
  [Lazy Seg Tree], [$O(n)$], [$O(log n)$], [$O(log n)$], [Range updates],
  [Fenwick Tree], [$O(n log n)$], [$O(log n)$], [$O(log n)$], [Lower constant],
  [Sparse Table], [$O(n log n)$], [N/A], [$O(1)$], [Static, min/max/gcd],
)

== Sparse Table (Static RMQ)

*Problem:* Range minimum query on static array in $O(1)$.

```cpp
class SparseTable {
    vector<vector<int>> table;
    vector<int> log2_table;
    int n;

public:
    SparseTable(const vector<int>& arr) {
        n = arr.size();
        int k = __lg(n) + 1;

        table.assign(k, vector<int>(n));
        log2_table.resize(n + 1);

        // Precompute logs
        log2_table[1] = 0;
        for (int i = 2; i <= n; i++) {
            log2_table[i] = log2_table[i / 2] + 1;
        }

        // Base case: intervals of length 1
        table[0] = arr;

        // Build table
        for (int j = 1; j < k; j++) {
            for (int i = 0; i + (1 << j) <= n; i++) {
                table[j][i] = min(table[j - 1][i],
                                  table[j - 1][i + (1 << (j - 1))]);
            }
        }
    }

    int query(int l, int r) {
        int j = log2_table[r - l + 1];
        return min(table[j][l], table[j][r - (1 << j) + 1]);
    }
};
```

*Complexity:*
- Build: $O(n log n)$ time and space
- Query: $O(1)$

*Key insight:* Overlapping intervals OK for min/max (idempotent operations).

== References

*Primary Sources:*

*Bentley, J.L. (1977)*. Solutions to Klee's Rectangle Problems. Unpublished manuscript, Carnegie-Mellon University.

*Fenwick, P.M. (1994)*. A New Data Structure for Cumulative Frequency Tables. Software: Practice and Experience 24(3): 327-336.

*Algorithms & Theory:*

*de Berg, M. et al. (2008)*. Computational Geometry: Algorithms and Applications (3rd ed.). Springer. ISBN 978-3-540-77973-5.

*Bender, M.A. & Farach-Colton, M. (2000)*. The LCA Problem Revisited. LATIN 2000, pp. 88-94.

*Sleator, D.D. & Tarjan, R.E. (1985)*. Self-Adjusting Binary Search Trees. Journal of the ACM 32(3): 652-686.

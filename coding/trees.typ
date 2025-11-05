= Trees

*Trees organize data hierarchically with parent-child relationships. Binary trees: each node has ≤2 children. Traversals: inorder, preorder, postorder (recursive/iterative).*

*See also:* Graphs (trees are acyclic connected graphs), Binary Search (for binary search trees), Dynamic Programming (tree DP), Heap & Priority Queue (for array-based trees)

*Cache locality note:* Tree traversal = pointer chasing = poor cache behavior. Each node access can be cache miss ($#sym.tilde.op$200 cycles). Array-based heaps/segment trees have better locality.

```cpp
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x = 0) : val(x), left(nullptr), right(nullptr) {}
};
```

*Memory layout:* TreeNode = 24 bytes on 64-bit systems (4-byte int + 4-byte padding + 2×8-byte pointers). Nodes scattered in heap = no spatial locality.

== Invert Binary Tree

*Problem:* Invert a binary tree (swap left and right children at every node).

*Approach - Recursive:* $O(n)$ time, $O(h)$ space where h is height

```cpp
TreeNode* invertTree(TreeNode* root) {
    if (!root) return nullptr;

    swap(root->left, root->right);
    invertTree(root->left);
    invertTree(root->right);

    return root;
}
```

*Stack depth:* Balanced tree: $O(log n)$ frames. Skewed tree: $O(n)$ frames = stack overflow risk. Each recursive call adds a stack frame ($#sym.tilde.op$16-48 bytes including return address and saved registers).

== Maximum Depth

*Problem:* Find the maximum depth (height) of a binary tree.

*Approach - Recursive:* $O(n)$ time, $O(h)$ space

```cpp
int maxDepth(TreeNode* root) {
    if (!root) return 0;
    return 1 + max(maxDepth(root->left), maxDepth(root->right));
}
```

*Tail recursion:* Not tail-recursive (need both subtrees before combining). Compiler cannot optimize to iteration.

== Same Tree

*Problem:* Check if two binary trees are structurally identical with same values.

*Approach - Recursive:* $O(n)$ time, $O(h)$ space

```cpp
bool isSameTree(TreeNode* p, TreeNode* q) {
    if (!p && !q) return true;
    if (!p || !q) return false;
    if (p->val != q->val) return false;

    return isSameTree(p->left, q->left) &&
           isSameTree(p->right, q->right);
}
```

*Short-circuit evaluation:* && operator stops on first false. If left subtrees differ, right subtrees never evaluated = saves traversal.

== Binary Tree Level Order Traversal

*Problem:* Return level order traversal of binary tree (BFS).

*Approach - BFS with Queue:* $O(n)$ time, $O(n)$ space

```cpp
vector<vector<int>> levelOrder(TreeNode* root) {
    if (!root) return {};

    vector<vector<int>> result;
    deque<TreeNode*> queue = {root};

    while (!queue.empty()) {
        int levelSize = queue.size();
        vector<int> level;
        level.reserve(levelSize);

        for (int i = 0; i < levelSize; i++) {
            TreeNode* node = queue.front();
            queue.pop_front();

            level.push_back(node->val);
            if (node->left) queue.push_back(node->left);
            if (node->right) queue.push_back(node->right);
        }
        result.push_back(move(level));
    }
    return result;
}
```

*Queue choice:* `deque` = O(1) both ends. `vector` requires pop_front() = O(n). `queue<TreeNode*, vector<TreeNode*>>` adds wrapper overhead.

*Memory:* Worst case (complete tree): bottom level has n/2 nodes in queue. Total space = O(n).

== Validate Binary Search Tree

*Problem:* Determine if a binary tree is a valid BST.

*Approach - Recursive with Bounds:* $O(n)$ time, $O(h)$ space

```cpp
bool isValidBST(TreeNode* root, long minVal = LONG_MIN, long maxVal = LONG_MAX) {
    if (!root) return true;

    if (root->val <= minVal || root->val >= maxVal) {
        return false;
    }

    return isValidBST(root->left, minVal, root->val) &&
           isValidBST(root->right, root->val, maxVal);
}
```

*Integer overflow:* Use `long` for bounds to handle INT_MIN/INT_MAX node values. Alternative: use `optional<int>` or nullptr for infinity.

== Kth Smallest in BST

*Problem:* Find kth smallest element in a binary search tree.

*Approach - Inorder Traversal (Iterative):* $O(n)$ time, $O(h)$ space

```cpp
int kthSmallest(TreeNode* root, int k) {
    vector<TreeNode*> stack;
    TreeNode* curr = root;
    int count = 0;

    while (curr || !stack.empty()) {
        // Go left as far as possible
        while (curr) {
            stack.push_back(curr);
            curr = curr->left;
        }

        curr = stack.back();
        stack.pop_back();

        if (++count == k) return curr->val;

        curr = curr->right;
    }
    return -1;
}
```

*Early termination:* Returns immediately upon finding kth element. Average case: $O(h + k)$. Best for small k.

*Optimization for frequent queries:* Augment each node with subtree size. Allows $O(h)$ search.

== Binary Tree Maximum Path Sum

*Problem:* Find maximum path sum in binary tree (path can start/end at any node).

*Approach - DFS with Global Max:* $O(n)$ time, $O(h)$ space

```cpp
class Solution {
    int maxSum = INT_MIN;

    int dfs(TreeNode* root) {
        if (!root) return 0;

        // Ignore negative paths
        int leftMax = max(0, dfs(root->left));
        int rightMax = max(0, dfs(root->right));

        // Update global max with split path
        maxSum = max(maxSum, root->val + leftMax + rightMax);

        // Return single path for parent
        return root->val + max(leftMax, rightMax);
    }

public:
    int maxPathSum(TreeNode* root) {
        dfs(root);
        return maxSum;
    }
};
```

*Key insight:* At each node, consider split path (left + node + right) for global max, but return single path to parent.

*Member variable:* Using class member avoids passing reference through recursion (cleaner but non-reentrant).

== Serialize/Deserialize Binary Tree

*Problem:* Design algorithm to serialize/deserialize binary tree to/from string.

*Approach - Preorder with Null Markers:* $O(n)$ time, $O(n)$ space

```cpp
class Codec {
public:
    string serialize(TreeNode* root) {
        if (!root) return "N";

        return to_string(root->val) + "," +
               serialize(root->left) + "," +
               serialize(root->right);
    }

    TreeNode* deserialize(string data) {
        istringstream iss(data);
        return deserializeHelper(iss);
    }

private:
    TreeNode* deserializeHelper(istringstream& iss) {
        string val;
        if (!getline(iss, val, ',')) return nullptr;
        if (val == "N") return nullptr;

        TreeNode* node = new TreeNode(stoi(val));
        node->left = deserializeHelper(iss);
        node->right = deserializeHelper(iss);
        return node;
    }
};
```

*String building:* Repeated `+` creates many temporary strings. For large trees: use `ostringstream` or `reserve()` capacity.

*Alternative format:* Binary serialization (`memcpy` of values + structure) is 5-10x faster but not human-readable.

== Array-Based Tree Representations

*Pointer-based trees:* Poor cache locality, scattered memory, pointer chasing = slow.

*Array-based trees:* Contiguous memory, better cache behavior, implicit parent/child relationships.

=== Binary Heap Layout (Complete Tree)

*BFS/Level-order storage:*
```cpp
// Array: [root, L, R, LL, LR, RL, RR, ...]
// Indices: 0,   1, 2, 3,  4,  5,  6,  ...
vector<int> tree;

int parent(int i) { return (i - 1) / 2; }
int left(int i) { return 2 * i + 1; }
int right(int i) { return 2 * i + 2; }

// Example tree:
//       1
//      / \
//     2   3
//    / \
//   4   5
// Stored as: [1, 2, 3, 4, 5]
```

*Advantages:*
- No pointers needed = save 16 bytes per node (two 8-byte pointers)
- Sequential memory = excellent prefetcher efficiency
- Parent/child access = arithmetic, no pointer dereference
- Cache-friendly: level-order traversal = sequential scan

*Disadvantages:*
- Must be complete tree (all levels filled except possibly last)
- Wastes space for unbalanced trees
- Random access still $O(log n)$ levels = $O(log n)$ cache misses

=== Van Emde Boas (vEB) Layout

*Problem with BFS layout:* Deep trees → large jumps between parent and child.

*vEB layout:* Recursive subdivision for better cache locality.

*Idea:*
1. Split tree into top $sqrt(h)$ levels and bottom $sqrt(h)$ levels
2. Store top recursively, then each subtree recursively
3. Result: parent/child closer in memory

*Why vEB layout improves cache performance:*

Standard BFS layout visits nodes in this order: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11...
When traversing from node 1 to its grandchild 4, the memory distance is 3 elements.
For a deep tree, visiting node 1 then its great-great-grandchild 16 spans 15 elements (960 bytes for integers), likely spanning multiple cache lines.

vEB layout recursively groups related subtrees together in memory. The top $sqrt(h)$ levels are stored first, followed by each subtree stored recursively. This ensures that a traversal of depth d typically touches at most $O(log_B d)$ cache lines rather than $O(d)$ cache lines, where B is the cache line size. The recursive subdivision naturally aligns with cache hierarchy: frequently-accessed top levels stay in L1/L2, while deeper subtrees may reside in L3 or RAM but are accessed less frequently.

*Example (height 4 tree):*
```
Standard BFS: [1, 2,3, 4,5,6,7, 8,9,10,11,12,13,14,15]
vEB layout:   [1, 2,3, 8,9,10,11, 4,5, 12,13, 6,7, 14,15]
              [top 2 levels][left subtree][mid subtrees][right subtree]
```

*Implementation (recursive index mapping):*
```cpp
// Helper: size of vEB tree of height h
int veb_size(int h) {
    return (1 << h) - 1;
}

// Recursive vEB index mapping from BFS index to vEB index
int veb_index(int i, int h) {
    if (h <= 2) return i;  // Base case: BFS

    int h_top = h / 2;
    int h_bot = h - h_top;
    int top_size = veb_size(h_top);

    if (i < top_size) {
        return veb_index(i, h_top);  // In top tree
    } else {
        int bot_size = veb_size(h_bot);
        int subtree = (i - top_size) / bot_size;
        int offset = (i - top_size) % bot_size;
        return top_size + subtree * veb_size(h_bot) + veb_index(offset, h_bot);
    }
}
```

*Cache performance:*
- BFS layout: $O(log n)$ cache misses per root-to-leaf path
- vEB layout: $O(log_B n)$ cache misses (B = cache line size in elements)
- Speedup: 2-4x for large trees that don't fit in cache

*When to use:* Static trees, range queries, computational geometry.

=== Segment Tree (Array-Based)

*Purpose:* Range queries (sum, min, max) in $O(log n)$ time, updates in $O(log n)$.

*Structure:*
```cpp
class SegmentTree {
    vector<int> tree;
    int n;

public:
    SegmentTree(const vector<int>& arr) {
        n = arr.size();
        tree.resize(4 * n);  // Needs 4n space (conservative)
        build(arr, 1, 0, n - 1);
    }

    void build(const vector<int>& arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            build(arr, 2 * node, start, mid);
            build(arr, 2 * node + 1, mid + 1, end);
            tree[node] = tree[2 * node] + tree[2 * node + 1];  // Merge
        }
    }

    int query(int node, int start, int end, int l, int r) {
        if (r < start || end < l) return 0;  // Outside range
        if (l <= start && end <= r) return tree[node];  // Inside range

        int mid = (start + end) / 2;
        int left_sum = query(2 * node, start, mid, l, r);
        int right_sum = query(2 * node + 1, mid + 1, end, l, r);
        return left_sum + right_sum;
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
            tree[node] = tree[2 * node] + tree[2 * node + 1];
        }
    }
};
```

*Space:* 4n worst case, 2n for perfect binary tree.

*Complexity:*
- Build: $O(n)$
- Query: $O(log n)$
- Update: $O(log n)$

*Lazy propagation:* Defer updates to children until needed. Enables range updates in $O(log n)$.

=== Fenwick Tree (Binary Indexed Tree)

*More space-efficient than segment tree:* Uses bit manipulation for parent/child.

```cpp
class FenwickTree {
    vector<int> tree;
    int n;

public:
    FenwickTree(int size) : n(size), tree(size + 1, 0) {}

    // Add val to index i (1-indexed)
    void update(int i, int val) {
        while (i <= n) {
            tree[i] += val;
            i += i & -i;  // Add LSB (least significant bit)
        }
    }

    // Sum of [1, i]
    int query(int i) {
        int sum = 0;
        while (i > 0) {
            sum += tree[i];
            i -= i & -i;  // Remove LSB
        }
        return sum;
    }

    // Range sum [l, r]
    int range_query(int l, int r) {
        return query(r) - query(l - 1);
    }
};
```

*Key trick:* `i & -i` isolates lowest set bit.
- `i = 6 = 0b110` → `i & -i = 0b010 = 2`
- Parent of i: `i - (i & -i)`
- Next i: `i + (i & -i)`

*Space:* $O(n)$ - half of segment tree

*Complexity:* Same as segment tree but simpler, lower constants.

*Limitation:* Only associative operations (sum, XOR). Can't do min/max directly (need workarounds).

== Cache-Oblivious Algorithms for Trees

*B-tree (cache-aware):* Tuned for specific block size $B$. Each node holds $B-1$ keys.

*Cache-oblivious B-tree:* Achieves optimal cache complexity without knowing $B$.

*Van Emde Boas tree (implicit):*
- Recursively split by $sqrt(h)$
- Automatically adapts to cache hierarchy
- $O(log_B n)$ transfers per search for any cache size $B$

*Memory layout optimization:*
```cpp
// BAD: Many small allocations
struct Node {
    int val;
    Node* left;
    Node* right;
};
// Each node = separate allocation = scattered = cache-hostile

// GOOD: Arena allocation (bump allocator)
class TreeArena {
    vector<char> memory;
    size_t offset = 0;

public:
    TreeArena(size_t size) : memory(size) {}

    template<typename T>
    T* allocate() {
        T* ptr = reinterpret_cast<T*>(&memory[offset]);
        offset += sizeof(T);
        return ptr;
    }
};

// Usage: allocate all nodes from arena
TreeArena arena(1000000);  // 1MB arena
Node* node = arena.allocate<Node>();
// All nodes clustered in memory = better locality
```

*Packed structures:*
```cpp
// Before: 24 bytes per node (padding)
struct Node {
    int val;         // 4 bytes
    // 4 bytes padding
    Node* left;      // 8 bytes
    Node* right;     // 8 bytes
};

// After: 20 bytes per node (tight packing)
struct __attribute__((packed)) Node {
    int val;
    Node* left;
    Node* right;
};
// Or use uint32_t indices instead of pointers (4 bytes each)
// 12 bytes total if using 32-bit indices
```

== Tree Traversal Optimizations

*Morris traversal (threading):* $O(1)$ space inorder traversal without stack/recursion.

```cpp
void morrisInorder(TreeNode* root) {
    TreeNode* curr = root;

    while (curr) {
        if (!curr->left) {
            // No left subtree, visit and go right
            cout << curr->val << " ";
            curr = curr->right;
        } else {
            // Find inorder predecessor
            TreeNode* pred = curr->left;
            while (pred->right && pred->right != curr) {
                pred = pred->right;
            }

            if (!pred->right) {
                // Create thread
                pred->right = curr;
                curr = curr->left;
            } else {
                // Thread exists, remove it
                pred->right = nullptr;
                cout << curr->val << " ";
                curr = curr->right;
            }
        }
    }
}
```

*Advantages:* No recursion, no stack, $O(1)$ space.

*Disadvantages:* Modifies tree temporarily (not thread-safe), complex.

*Iterative with explicit stack (better in practice):*
```cpp
void iterativeInorder(TreeNode* root) {
    vector<TreeNode*> stack;
    TreeNode* curr = root;

    while (curr || !stack.empty()) {
        // Go left as far as possible
        while (curr) {
            stack.push_back(curr);
            curr = curr->left;
        }

        curr = stack.back();
        stack.pop_back();

        cout << curr->val << " ";
        curr = curr->right;
    }
}
```

*Stack size:* $O(h)$ where $h$ is height. Balanced tree: $O(log n)$, skewed: $O(n)$.

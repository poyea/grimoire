= Trees

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

= Trees

== Invert Binary Tree

*Problem:* Invert a binary tree (swap left and right children at every node).

*Approach - Recursive:* $O(n)$ time, $O(h)$ space where h is height
- Base case: if `root is None`: return None
- Swap children:
  + `temp = root.left`
  + `root.left = root.right`
  + `root.right = temp`
- Recursively invert subtrees:
  + `invertTree(root.left)`
  + `invertTree(root.right)`
- Return `root`

== Maximum Depth of Binary Tree

*Problem:* Find the maximum depth (height) of a binary tree.

*Approach - Recursive:* $O(n)$ time, $O(h)$ space
- Base case: if `root is None`: return 0
- Return `1 + max(maxDepth(root.left), maxDepth(root.right))`

== Same Tree

*Problem:* Check if two binary trees are structurally identical with same values.

*Approach - Recursive:* $O(n)$ time, $O(h)$ space
- Base cases:
  + If both None: return True
  + If one is None: return False
  + If `p.val != q.val`: return False
- Return `isSameTree(p.left, q.left) and isSameTree(p.right, q.right)`

*Key Python concepts:*
- Use `.val` to compare node values, not direct node comparison

== Subtree of Another Tree

*Problem:* Check if one tree is a subtree of another tree.

*Approach - Recursive with Helper:* $O(n #sym.times m)$ time, $O(h)$ space
- Create helper function `isSameTree(p, q)` (see problem 3)
- Base cases:
  + If both None: return True
  + If one is None: return False
  + If `isSameTree(root, subRoot)`: return True
- Check subtrees: return `isSubtree(root.left, subRoot) or isSubtree(root.right, subRoot)`

== Lowest Common Ancestor of a BST

*Problem:* Find lowest common ancestor of two nodes in a binary search tree.

*Approach - Recursive:* $O(h)$ time, $O(h)$ space where h is height
- If `root is None`: return None
- If both p and q are smaller than root:
  + LCA is in left subtree: return `LCA(root.left, p, q)`
- If both p and q are larger than root:
  + LCA is in right subtree: return `LCA(root.right, p, q)`
- Else: return `root` (split point is LCA)

*Key insight:* In BST, LCA is the split point where p and q diverge to different subtrees.

== Binary Tree Level Order Traversal

*Problem:* Return level order traversal of binary tree (BFS).

*Approach - BFS with Queue:* $O(n)$ time, $O(n)$ space
- Initialize `result = []`, `queue = [root]`
- While `queue`:
  + Get `levelSize = len(queue)` (process one level at a time)
  + Create `levelValues = []`
  + For i in range(levelSize):
    - `curr = queue.pop(0)`
    - If curr: append `curr.val` to levelValues
    - Add children to queue: `queue.append(curr.left)`, `queue.append(curr.right)`
  + If levelValues not empty: append to result
- Return `result`

*Key insight:* Use queue length to process one level at a time.

== Validate Binary Search Tree

*Problem:* Determine if a binary tree is a valid BST.

*Approach - Recursive with Bounds:* $O(n)$ time, $O(h)$ space
- Create helper function `valid(root, minVal, maxVal)`:
  + If `root is None`: return True
  + If `root.val <= minVal or root.val >= maxVal`: return False
  + Return `valid(root.left, minVal, root.val) and valid(root.right, root.val, maxVal)`
- Return `valid(root, float("-inf"), float("inf"))`

*Key insight:* Each node must be within bounds defined by ancestors, not just parent.

== Kth Smallest Element in BST

*Problem:* Find kth smallest element in a binary search tree.

*Approach - Inorder Traversal (Iterative):* $O(n)$ time, $O(h)$ space
- Initialize `stack = []`, `curr = root`, `values = []`
- While `stack` or `curr`:
  + Go left as far as possible:
    - While `curr`: append to stack, move `curr = curr.left`
  + Process node: `curr = stack.pop()`, append `curr.val` to values
  + Move right: `curr = curr.right`
- Return `values[k-1]`

*Key insight:* Inorder traversal of BST gives sorted order.

== Construct Binary Tree from Preorder and Inorder Traversal

*Problem:* Build binary tree from preorder and inorder traversal arrays.

*Approach - Recursive:* $O(n^2)$ time, $O(n)$ space
- Base case: if not preorder or not inorder: return None
- Root is first element: `root = TreeNode(preorder[0])`
- Find root in inorder: `mid = inorder.index(preorder[0])`
- Recursively build left: `root.left = buildTree(preorder[1:mid+1], inorder[:mid])`
- Recursively build right: `root.right = buildTree(preorder[mid+1:], inorder[mid+1:])`
- Return `root`

*Key insight:* Preorder gives root, inorder splits left/right subtrees.

*Key Python concepts:*
- Preorder: root, left, right
- Inorder: left, root, right

==  Binary Tree Maximum Path Sum

*Problem:* Find maximum path sum in binary tree (path can start/end at any node).

*Approach - DFS with Global Max:* $O(n)$ time, $O(h)$ space
- Initialize global variable: `self.maxSum = float("-inf")`
- Define recursive function `dfs(root)`:
  + If `root is None`: return 0
  + Get max from children (ignore negative): `leftMax = max(0, dfs(root.left))`
  + `rightMax = max(0, dfs(root.right))`
  + Update global max with split path: `self.maxSum = max(self.maxSum, root.val + leftMax + rightMax)`
  + Return max single path: `root.val + max(leftMax, rightMax)`
- Call `dfs(root)`, return `self.maxSum`

*Key insight:* At each node, consider split path for global max but return single path to parent.

==  Serialize and Deserialize Binary Tree

*Problem:* Design algorithm to serialize/deserialize binary tree to/from string.

*Approach - Preorder with Markers:* $O(n)$ time, $O(n)$ space

*Serialize:*
- Use preorder DFS with stack
- Append node values or 'N' for null to result list
- Join with commas: `",".join(res)`

*Deserialize:*
- Split string: `data = data.split(",")`
- Use index pointer: `self.i = 0`
- Define recursive `dfs()`:
  + If `data[self.i] == 'N'`: increment i, return None
  + Create node: `curr = TreeNode(int(data[self.i]))`
  + Increment `self.i`
  + Recursively build: `curr.left = dfs()`, `curr.right = dfs()`
  + Return `curr`
- Return `dfs()`

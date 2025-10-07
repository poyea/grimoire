= Reference Notes

== Python Data Structures & Operations

*Collections:*
- `collections.defaultdict(set)` - dictionary with default set values
  + Used in Sudoku to track rows/columns/grids (grid position = row // 3, col // 3)
- Copy list: `arr[:]` (deep copy)
- Concatenate lists: `arr1 + arr2`
- Insert at position: `list.insert(index, val)`

*Built-in Functions:*
- Infinity: `float("inf")` or `float("-inf")`
- XOR operator: `^`
- Min element by length: `min(array, key=len)`
- Permutations/combinations: `itertools.permutations(list)`

== Common Algorithms

*Merge Sort - $O(n log n)$:*
- Continuously split array into halves until size 1
- Merge sorted halves using two pointers
- Functions: `merge(arr, l, m, r)` and `mergeSort(arr, l, r)`

*String Sorting:*
```python
from functools import cmp_to_key
def compare(a, b):
    return -1 if a + b > b + a else 1
arr = sorted(arr, key=cmp_to_key(compare))
```

== Search & Traversal Patterns

*BFS (Breadth-First Search):*
- Use queue, process level by level
- Pattern: `queue = [root]`, then `for i in range(len(queue)): ...`

*DFS (Depth-First Search):*
- Use stack or recursion
- Go deep, backtrack to unexplored branches

*Binary Search:*
- Pattern: `mid = (left + right) // 2`
- Used in rotated arrays, finding min/max

== Heap Operations

*Heaps:*
- Binary tree: parent ≤ children (min heap)
- Operations: add/pop in $O(log n)$, get min in $O(1)$
- Access min/max: `heap[0]`
- Max heap in Python: negate values (`heappush(heap, -val)`)

== Additional Problem Patterns

*Fibonacci:*
- DP: `dp[i] = dp[i-1] + dp[i-2]`
- Recursive: $O(2^n)$, base cases n=0,1

*Symmetric Tree:*
- Recursive: compare `(left.left, right.right)` and `(left.right, right.left)`
- Iterative: BFS with queue

*Move Zeroes:*
- Two pointers: `left` tracks position, `right` iterates
- Swap non-zero elements forward
- Alternative: `nums.sort(key=lambda x: x == 0)`

*Trapping Rain Water:*
- Approach 1: $O(n)$ space - precompute leftMax and rightMax arrays
- Approach 2: $O(1)$ space - two pointers with running max values

*Missing Number:*
- Optimal $O(n)$: sum(0 to n) - sum(nums)

*LRU Cache:*
- Use OrderedDict or HashMap + Doubly Linked List
- $O(1)$ get and put operations

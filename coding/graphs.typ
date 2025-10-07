= Graphs

== Number of Islands

*Problem:* Count number of islands in 2D grid ('1' = land, '0' = water).

*Approach - BFS:* $O(n #sym.times m)$ time, $O(n #sym.times m)$ space
- Initialize `result = 0`, `visited = set()`, get rows and cols
- Define `bfs(r, c)`:
  + Create queue: `queue = [(r, c)]`
  + Add to visited: `visited.add((r, c))`
  + While queue:
    * `row, col = queue.pop(0)`
    * For all 4 directions (up, down, left, right):
      - If in bounds, equals '1', and not visited:
        + Add to queue and visited
- For each cell in grid:
  + If `grid[r][c] == '1' and (r, c) not in visited`:
    * Increment result
    * Call `bfs(r, c)`
- Return result

*Key insight:* Each BFS explores one complete island.

== Clone Graph

*Problem:* Deep copy an undirected graph.

*Approach - DFS with HashMap:* $O(n + e)$ time, $O(n)$ space
- Create hashmap: `oldToNew = {}`
- Define `dfs(node)`:
  + If `node in oldToNew`: return `oldToNew[node]` (already cloned)
  + Create copy: `copy = Node(node.val)`
  + Store mapping: `oldToNew[node] = copy`
  + For each neighbor:
    - `copy.neighbors.append(dfs(neighbor))`
  + Return copy
- Return `dfs(node) if node else None`

*Key insight:* HashMap prevents infinite loops and tracks cloned nodes.

== Pacific Atlantic Water Flow

*Problem:* Find cells where water can flow to both Pacific and Atlantic oceans.

*Approach - Reverse DFS from Oceans:* $O(n #sym.times m)$ time, $O(n #sym.times m)$ space
- Initialize `pacific = set()`, `atlantic = set()`
- Define `dfs(r, c, visited, prevHeight)`:
  + If out of bounds, visited, or `heights[r][c] < prevHeight`: return
  + Add to visited: `visited.add((r, c))`
  + DFS all 4 directions with `heights[r][c]` as prevHeight
- Start DFS from ocean borders:
  + Pacific: top row and left column
  + Atlantic: bottom row and right column
- Find intersection: cells in both pacific and atlantic sets
- Return result list

*Key insight:* Start from oceans and flow upward (reverse flow) to find reachable cells.

== Course Schedule

*Problem:* Determine if you can finish all courses given prerequisites (detect cycle in directed graph).

*Approach - DFS Cycle Detection:* $O(n + p)$ time where p is prerequisites
- Build adjacency list: `graph = {i: [] for i in range(numCourses)}`
- For each `[course, prereq]`: `graph[course].append(prereq)`
- Initialize `visited = set()` (tracks current DFS path)
- Define `dfs(course)`:
  + If `course in visited`: return False (cycle detected)
  + If `graph[course] == []`: return True (no prerequisites)
  + Add to visited: `visited.add(course)`
  + For each prereq: if `not dfs(prereq)`: return False
  + Backtrack: `visited.remove(course)`
  + Mark as processed: `graph[course] = []`
  + Return True
- For each course: if `not dfs(course)`: return False
- Return True

*Key insight:* Visited set tracks current path to detect cycles; clear prerequisites after processing.

== Course Schedule II

*Problem:* Return ordering of courses to finish all (topological sort).

*Approach - DFS Topological Sort:* $O(n + p)$ time
- Similar to Course Schedule but collect courses in post-order
- Add courses to result after exploring all prerequisites
- Reverse result at end to get correct order

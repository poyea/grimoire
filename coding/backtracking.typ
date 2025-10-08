= Backtracking

== Combination Sum

*Problem:* Find all combinations that sum to target (can reuse elements).

*Decision Tree:* $O(2^t)$ time where t = target value

```cpp
class Solution {
    void dfs(vector<int>& cand, int i, vector<int>& curr, int total,
             int target, vector<vector<int>>& result) {
        if (total == target) {
            result.push_back(curr);
            return;
        }
        if (i >= cand.size() || total > target) return;

        // Include cand[i] (can reuse, so stay at index i)
        curr.push_back(cand[i]);
        dfs(cand, i, curr, total + cand[i], target, result);
        curr.pop_back();

        // Exclude cand[i]
        dfs(cand, i + 1, curr, total, target, result);
    }

public:
    vector<vector<int>> combinationSum(vector<int>& cand, int target) {
        vector<vector<int>> result;
        vector<int> curr;
        dfs(cand, 0, curr, 0, target, result);
        return result;
    }
};
```

*Pattern:* At each position, either include (stay at i) or exclude (move to i+1).

== Word Search

*Problem:* Check if word exists in board (4-directional, no cell reuse).

*Backtracking DFS:* $O(n m 4^L)$ time where L = word length

```cpp
class Solution {
    bool dfs(vector<vector<char>>& board, string& word, int r, int c,
             int i, set<pair<int,int>>& visited) {
        if (i == word.length()) return true;

        int rows = board.size(), cols = board[0].size();
        if (r < 0 || r >= rows || c < 0 || c >= cols ||
            visited.count({r, c}) || board[r][c] != word[i]) {
            return false;
        }

        visited.insert({r, c});

        bool found = dfs(board, word, r+1, c, i+1, visited) ||
                     dfs(board, word, r-1, c, i+1, visited) ||
                     dfs(board, word, r, c+1, i+1, visited) ||
                     dfs(board, word, r, c-1, i+1, visited);

        visited.erase({r, c});  // Backtrack
        return found;
    }

public:
    bool exist(vector<vector<char>>& board, string word) {
        int rows = board.size(), cols = board[0].size();
        set<pair<int,int>> visited;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (dfs(board, word, r, c, 0, visited)) return true;
            }
        }
        return false;
    }
};
```

*Optimization:* Use `board[r][c] = '#'` to mark visited instead of set. Restore on backtrack.

```cpp
bool dfs(vector<vector<char>>& board, string& word, int r, int c, int i) {
    if (i == word.length()) return true;
    if (r < 0 || r >= board.size() || c < 0 || c >= board[0].size() ||
        board[r][c] != word[i]) return false;

    char temp = board[r][c];
    board[r][c] = '#';

    bool found = dfs(board, word, r+1, c, i+1) ||
                 dfs(board, word, r-1, c, i+1) ||
                 dfs(board, word, r, c+1, i+1) ||
                 dfs(board, word, r, c-1, i+1);

    board[r][c] = temp;  // Backtrack
    return found;
}
```

== Backtracking Performance

*Stack frame overhead:*
- Each recursive call = $#sym.tilde.op$16-48 bytes stack frame (varies with local variables and saved registers)
- Deep recursion (depth 1000) = 16-48KB stack usage
- Default stack size: 1-8MB. Deep recursion may risk overflow on some platforms.

*Tail call optimization:*
Backtracking is NOT tail-recursive (need to restore state after call). Compiler cannot optimize to iteration.

*Iterative backtracking (manual stack):*
```cpp
// Avoids recursion overhead, explicit control
struct State { int r, c, i; };
stack<State> stk;
stk.push({start_r, start_c, 0});

while (!stk.empty()) {
    auto [r, c, i] = stk.top();
    stk.pop();

    // Process state, push next states...
}
```

Pros: No stack overflow, easier to add pruning.
Cons: More code, harder to read.

*Branch ordering:*
```cpp
// Try most promising direction first (early termination)
if (heuristic_best(r+1, c)) {
    if (dfs(board, word, r+1, c, i+1)) return true;
}
// Then try others...
```

*Cache behavior:*
- Backtracking = random memory access = cache unfriendly
- In-place marking eliminates `unordered_set`: saves $#sym.tilde.op$40 bytes per cell + hash overhead
- Small boards (< 20×20): entire board fits in L1 cache

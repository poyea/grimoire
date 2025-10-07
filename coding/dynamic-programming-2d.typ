= 2-D Dynamic Programming

== Unique Paths

*Problem:* Count unique paths from top-left to bottom-right in m×n grid (can only move right or down).

*Approach - Space-Optimized DP:* $O(m #sym.times n)$ time, $O(n)$ space
- Initialize `dp = [1] * n` (top row has 1 path per cell)
- For each row i in range(1, m):
  + Create `newRow = [1] * n` (leftmost column has 1 path)
  + For j in range(1, n):
    - `newRow[j] = newRow[j-1] + dp[j]` (sum of paths from left and above)
  + Update `dp = newRow`
- Return `dp[-1]`

*Key insight:* Number of paths to cell = paths from left + paths from above.

== Longest Common Subsequence

*Problem:* Find length of longest common subsequence between two strings.

*Approach - 2D DP (Bottom-up):* $O(n #sym.times m)$ time, $O(n #sym.times m)$ space
- Initialize `dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]`
- For i in range(len(text1) - 1, -1, -1):
  + For j in range(len(text2) - 1, -1, -1):
    * If `text1[i] == text2[j]`:
      - `dp[i][j] = 1 + dp[i+1][j+1]` (match - add 1)
    * Else:
      - `dp[i][j] = max(dp[i+1][j], dp[i][j+1])` (skip one char)
- Return `dp[0][0]`

*Key insight:* If chars match, add 1 and move diagonal; else take max of skipping either char.

*Key Python concepts:*
- 2D DP initialization: `dp = [[0 for j in range(cols)] for i in range(rows)]`

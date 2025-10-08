= 2-D Dynamic Programming

*Row-major order critical:* C++ stores 2D arrays row-major: `a[i][j]` and `a[i][j+1]` are adjacent in memory. Iterating columns first = cache miss per access.

```cpp
// GOOD: row-major (cache-friendly)
for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
        process(dp[i][j]);

// BAD: column-major (cache-hostile)
for (int j = 0; j < cols; j++)
    for (int i = 0; i < rows; i++)
        process(dp[i][j]);  // Each access = cache miss
```

== Unique Paths

*Problem:* Count unique paths from top-left to bottom-right in m×n grid (only move right/down).

*Approach - Space-Optimized DP:* $O(m n)$ time, $O(n)$ space

```cpp
int uniquePaths(int m, int n) {
    vector<int> dp(n, 1);

    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[j] += dp[j-1];  // dp[j] = paths from above, dp[j-1] = from left
        }
    }

    return dp[n-1];
}
```

*Key insight:* Only need previous row. Rolling array reduces O(mn) → O(n) space.

*In-place update:* `dp[j] += dp[j-1]` updates left-to-right. Previous `dp[j]` (from above) still valid when accessed. Eliminates need for two arrays.

*Cache performance:* `dp` array = n integers. For typical grids (n < 1000), entire array fits in L1 cache (32-48KB). Each iteration = sequential access = perfect prefetching.

== Longest Common Subsequence

*Problem:* Find length of longest common subsequence between two strings.

*Approach - 2D DP:* $O(n m)$ time, $O(n m)$ space

```cpp
int longestCommonSubsequence(string text1, string text2) {
    int m = text1.length(), n = text2.length();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                dp[i][j] = 1 + dp[i-1][j-1];
            } else {
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
            }
        }
    }

    return dp[m][n];
}
```

*Memory layout optimization:*
```cpp
// Allocate as single contiguous block (better cache locality)
vector<int> dp((m+1) * (n+1), 0);
#define DP(i,j) dp[(i)*(n+1) + (j)]

for (int i = 1; i <= m; i++) {
    for (int j = 1; j <= n; j++) {
        if (text1[i-1] == text2[j-1]) {
            DP(i,j) = 1 + DP(i-1, j-1);
        } else {
            DP(i,j) = max(DP(i-1,j), DP(i,j-1));
        }
    }
}
```

*Space-optimized version:* $O(min(m, n))$ space using two rows:

```cpp
int longestCommonSubsequence(string text1, string text2) {
    int m = text1.length(), n = text2.length();

    // Ensure text2 is shorter for better cache locality
    if (m < n) {
        swap(text1, text2);
        swap(m, n);
    }

    vector<int> prev(n + 1, 0), curr(n + 1, 0);

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (text1[i-1] == text2[j-1]) {
                curr[j] = 1 + prev[j-1];
            } else {
                curr[j] = max(prev[j], curr[j-1]);
            }
        }
        swap(prev, curr);
    }

    return prev[n];
}
```

*Array swap:* `swap(prev, curr)` swaps pointers, not data. O(1) operation. `curr = prev` would copy entire array.

*Cache analysis:*
- Small strings (n < 1000): both `prev` and `curr` fit in L1 = ~4 cycles per access
- Large strings: sequential access still cache-friendly due to prefetching

*SIMD potential:* `max(prev[j], curr[j-1])` can be vectorized across 8 elements using `_mm256_max_epi32()`. Requires careful handling of dependencies.

*Alignment:* For SIMD, ensure `dp` is 32-byte aligned:
```cpp
alignas(32) vector<int> dp(n + 1);
// Or use aligned_alloc / _mm_malloc
```

*Branch prediction:* `if (text1[i-1] == text2[j-1])` depends on string similarity. Random strings = 1/26 matches = predictable. Similar strings = unpredictable.

*Branchless alternative:*
```cpp
bool match = (text1[i-1] == text2[j-1]);
curr[j] = match * (1 + prev[j-1]) + (!match) * max(prev[j], curr[j-1]);
```
Usually slower unless branch mispredicts are severe (>30%).

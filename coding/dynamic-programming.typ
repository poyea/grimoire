= Dynamic Programming

*Dynamic Programming (DP)* solves problems by breaking them into overlapping subproblems, storing results to avoid recomputation.

*Two key properties:*
1. *Optimal substructure:* Optimal solution built from optimal subsolutions
2. *Overlapping subproblems:* Same subproblems computed multiple times

*Development approach:*
1. Recursive solution (brute force)
2. Memoization (top-down DP)
3. Tabulation (bottom-up DP)
4. Space optimization (rolling arrays, state reduction)

*See also:* Greedy algorithms (when local optimum = global optimum), Backtracking (for exhaustive search), Graphs (for shortest path DP)

== 1-D Dynamic Programming

*Cache performance:* DP on arrays = sequential access = excellent prefetching. CPU loads entire cache line (64 bytes = 16 ints) per access, amortizing latency.

=== Climbing Stairs

*Problem:* Count distinct ways to climb n stairs (can take 1 or 2 steps at a time).

*Approach - Space Optimized:* $O(n)$ time, $O(1)$ space

```cpp
int climbStairs(int n) {
    if (n <= 2) return n;

    int prev1 = 1, prev2 = 2;
    for (int i = 3; i <= n; i++) {
        int curr = prev1 + prev2;
        prev1 = prev2;
        prev2 = curr;
    }
    return prev2;
}
```

*Key insight:* Fibonacci recurrence. Only need last 2 values = O(1) space.

*Register allocation:* `prev1`, `prev2`, `curr` typically kept in registers (no memory loads). Loop body achieves $#sym.tilde.op$3-5 cycles per iteration with good branch prediction on modern CPUs.

=== House Robber

*Problem:* Maximum money you can rob from houses without robbing adjacent houses.

*Approach - Space Optimized:* $O(n)$ time, $O(1)$ space

```cpp
int rob(vector<int>& nums) {
    int rob1 = 0, rob2 = 0;  // rob1 = max at i-2, rob2 = max at i-1

    for (int num : nums) {
        int temp = max(rob2, rob1 + num);
        rob1 = rob2;
        rob2 = temp;
    }
    return rob2;
}
```

*Branchless max:* Modern compilers convert `max()` to conditional move (CMOV). No branch mispredicts.

```cpp
// Assembly (x86-64):
// cmp  rob2, rob1_plus_num
// cmovl rob2, rob1_plus_num  // Conditional move if less
```

*See also:* House Robber II (circular array, consider first/last house separately)

=== Longest Palindromic Substring

*Problem:* Find longest palindromic substring.

*Approach - Expand Around Center:* $O(n^2)$ time, $O(1)$ space

```cpp
string longestPalindrome(string s) {
    int start = 0, maxLen = 0;

    auto expandAroundCenter = [&](int left, int right) {
        while (left >= 0 && right < s.length() && s[left] == s[right]) {
            left--;
            right++;
        }
        int len = right - left - 1;
        if (len > maxLen) {
            maxLen = len;
            start = left + 1;
        }
    };

    for (int i = 0; i < s.length(); i++) {
        expandAroundCenter(i, i);      // Odd length
        expandAroundCenter(i, i + 1);  // Even length
    }

    return s.substr(start, maxLen);
}
```

*String access pattern:* Expanding from center = bidirectional access. Cache line may contain both `s[left]` and `s[right]` for small palindromes.

*Manacher's algorithm:* O(n) time but complex. Only worth for very large strings or repeated queries.

*See also:* String Algorithms (for advanced pattern matching techniques)

=== Coin Change

*Problem:* Find minimum coins needed to make amount (infinite supply of each coin).

*Approach - DP:* $O(op("amount") × op("coins"))$ time, $O(op("amount"))$ space

```cpp
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, INT_MAX);
    dp[0] = 0;

    for (int a = 1; a <= amount; a++) {
        for (int coin : coins) {
            if (a >= coin && dp[a - coin] != INT_MAX) {
                dp[a] = min(dp[a], dp[a - coin] + 1);
            }
        }
    }

    return dp[amount] == INT_MAX ? -1 : dp[amount];
}
```

*Memory access:* Sequential scan of `dp` array = cache-friendly. `coins` array small enough to fit in L1 cache.

*SIMD opportunity:* Inner loop can be vectorized for multiple amounts simultaneously using AVX2. Process 8 amounts in parallel.

```cpp
// Conceptual SIMD (actual implementation more complex):
__m256i amounts = _mm256_setr_epi32(a, a+1, a+2, ..., a+7);
__m256i results = _mm256_min_epi32(dp_vec, candidate_vec);
```

*Related:* Knapsack problems (similar unbounded/bounded resource allocation), Greedy (if coins have special property like powers of 2)

=== Maximum Product Subarray

*Problem:* Find contiguous subarray with largest product.

*Approach - Track Min/Max:* $O(n)$ time, $O(1)$ space

```cpp
int maxProduct(vector<int>& nums) {
    int result = nums[0];
    int currMin = 1, currMax = 1;

    for (int num : nums) {
        if (num == 0) {
            currMin = currMax = 1;
            result = max(result, 0);
            continue;
        }

        int temp = currMax * num;
        currMax = max({num, currMax * num, currMin * num});
        currMin = min({num, temp, currMin * num});

        result = max(result, currMax);
    }
    return result;
}
```

*Key insight:* Track both min and max because negative × negative = positive. Negative currMin can become currMax.

*Integer overflow:* Products can exceed INT_MAX. Use `long long` or detect overflow with `__builtin_mul_overflow()`.

*See also:* Maximum Subarray (Kadane's algorithm for sum instead of product), Sliding Window (for contiguous subarray patterns)

=== Longest Increasing Subsequence

*Problem:* Find length of longest strictly increasing subsequence.

*Approach 1 - DP:* $O(n^2)$ time, $O(n)$ space

```cpp
int lengthOfLIS(vector<int>& nums) {
    vector<int> dp(nums.size(), 1);

    for (int i = 1; i < nums.size(); i++) {
        for (int j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }

    return *max_element(dp.begin(), dp.end());
}
```

*Approach 2 - Binary Search + Patience Sort:* $O(n log n)$ time, $O(n)$ space

```cpp
int lengthOfLIS(vector<int>& nums) {
    vector<int> tails;  // tails[i] = smallest tail of LIS of length i+1

    for (int num : nums) {
        auto it = lower_bound(tails.begin(), tails.end(), num);
        if (it == tails.end()) {
            tails.push_back(num);
        } else {
            *it = num;
        }
    }

    return tails.size();
}
```

*Binary search optimization:* `lower_bound()` on small vectors (< 64 elements): linear search can be faster due to prefetching. Use `if (tails.size() < 64)` heuristic.

*Cache:* `tails` array typically small = stays in L1 cache. Binary search has poor spatial locality but good temporal locality.

*See also:* Binary Search (for optimization techniques), Greedy (patience sort is a greedy approach)

== 2-D Dynamic Programming

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

*See also:* Arrays (for memory layout and cache optimization details)

=== Unique Paths

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

*Related:* Graphs (grid problems can be modeled as implicit graphs), Backtracking (for generating all paths)

=== Longest Common Subsequence

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
- Small strings (n < 1000): both `prev` and `curr` fit in L1 = $#sym.tilde.op$4 cycles per access
- Large strings: sequential access still cache-friendly due to prefetching

*SIMD potential:* `max(prev[j], curr[j-1])` can be vectorized across 8 elements using `_mm256_max_epi32()`. Requires careful handling of dependencies.

*Alignment:* For SIMD, ensure `dp` is 32-byte aligned:
```cpp
alignas(32) vector<int> dp(n + 1);
// Or use aligned_alloc / _mm_malloc
```

*Branch prediction:* `if (text1[i-1] == text2[j-1])` depends on string similarity. Random strings = $#sym.tilde.op$1/26 match rate = predictable pattern for modern branch predictors. Similar strings = higher match rate but still predictable. Branch misprediction penalty is low ($#sym.tilde.op$15-20 cycles) on modern CPUs.

*Note:* Arithmetic with booleans rarely improves performance over well-predicted branches on modern CPUs. The compiler already optimizes simple branches to conditional moves when beneficial.

*See also:* String Algorithms (for edit distance and other string matching problems), Greedy (for special cases with optimal greedy solutions)

== Multi-Dimensional Dynamic Programming

*Higher dimensions:* 3D+ DP used for problems with multiple independent state variables.

*Memory considerations:*
- 3D array: $O(n × m × k)$ space can quickly exceed cache
- Flatten to 1D: `dp[i][j][k]` → `dp[i*m*k + j*k + k]` for better locality
- Space optimization: often can reduce one dimension using rolling arrays

=== Example: 3D DP - Longest Common Subsequence of 3 Strings

*Problem:* Find LCS of three strings.

*Approach:* $O(n × m × p)$ time and space

```cpp
int longestCommonSubsequence3(string a, string b, string c) {
    int n = a.length(), m = b.length(), p = c.length();

    // 3D DP table
    vector<vector<vector<int>>> dp(n+1,
        vector<vector<int>>(m+1,
            vector<int>(p+1, 0)));

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            for (int k = 1; k <= p; k++) {
                if (a[i-1] == b[j-1] && b[j-1] == c[k-1]) {
                    dp[i][j][k] = 1 + dp[i-1][j-1][k-1];
                } else {
                    dp[i][j][k] = max({
                        dp[i-1][j][k],
                        dp[i][j-1][k],
                        dp[i][j][k-1]
                    });
                }
            }
        }
    }

    return dp[n][m][p];
}
```

*Space optimization:* Reduce to $O(m × p)$ using rolling arrays:

```cpp
int longestCommonSubsequence3(string a, string b, string c) {
    int n = a.length(), m = b.length(), p = c.length();

    // Only keep current and previous "layer" (i dimension)
    vector<vector<int>> prev(m+1, vector<int>(p+1, 0));
    vector<vector<int>> curr(m+1, vector<int>(p+1, 0));

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= m; j++) {
            for (int k = 1; k <= p; k++) {
                if (a[i-1] == b[j-1] && b[j-1] == c[k-1]) {
                    curr[j][k] = 1 + prev[j-1][k-1];
                } else {
                    curr[j][k] = max({
                        prev[j][k],
                        curr[j-1][k],
                        curr[j][k-1]
                    });
                }
            }
        }
        swap(prev, curr);
        // Clear curr for next iteration
        for (auto& row : curr)
            fill(row.begin(), row.end(), 0);
    }

    return prev[m][p];
}
```

*Cache optimization:* For large dimensions, blocked iteration improves cache reuse. Process the DP table in cache-sized blocks rather than full dimensions.

=== State Compression Techniques

*Bitmask DP:* When state involves subsets, use bitmasks to represent state.

```cpp
// Example: Traveling Salesman Problem (TSP)
// dp[mask][i] = min cost to visit cities in mask, ending at i
vector<vector<int>> dp(1 << n, vector<int>(n, INT_MAX));
dp[1][0] = 0;  // Start at city 0

for (int mask = 1; mask < (1 << n); mask++) {
    for (int i = 0; i < n; i++) {
        if (!(mask & (1 << i))) continue;
        for (int j = 0; j < n; j++) {
            if (mask & (1 << j)) continue;
            int newMask = mask | (1 << j);
            dp[newMask][j] = min(dp[newMask][j],
                                 dp[mask][i] + dist[i][j]);
        }
    }
}
```

*Coordinate compression:* When ranges are large but sparse, map values to dense indices.

```cpp
// Example: Large coordinate space but few distinct values
vector<int> coords = {1, 1000000, 5, 999999};
sort(coords.begin(), coords.end());
coords.erase(unique(coords.begin(), coords.end()), coords.end());

// Map original values to compressed indices [0, 1, 2, 3]
unordered_map<int, int> compressed;
for (int i = 0; i < coords.size(); i++) {
    compressed[coords[i]] = i;
}

// Now use dp[compressed[value]] instead of dp[value]
```

*Digit DP:* For counting numbers with specific digit properties.

```cpp
// Example: Count numbers ≤ N with sum of digits = target
int digitDP(string num, int pos, int sum, int tight,
            vector<vector<vector<int>>>& memo) {
    if (pos == num.length()) {
        return sum == target ? 1 : 0;
    }
    if (memo[pos][sum][tight] != -1) {
        return memo[pos][sum][tight];
    }

    int limit = tight ? (num[pos] - '0') : 9;
    int result = 0;

    for (int digit = 0; digit <= limit; digit++) {
        result += digitDP(num, pos + 1, sum + digit,
                          tight && (digit == limit), memo);
    }

    return memo[pos][sum][tight] = result;
}
```

*See also:* Bit Manipulation (for bitmask techniques), Graphs (for state-space DP like shortest path)

=== DP on Trees

*Tree DP:* DP where states are defined on tree nodes. Process bottom-up using DFS.

```cpp
// Example: Maximum independent set in tree
// dp[node][0] = max value when node not selected
// dp[node][1] = max value when node selected
unordered_map<TreeNode*, pair<int,int>> dp;

pair<int,int> dfs(TreeNode* node) {
    if (!node) return {0, 0};

    auto [leftExclude, leftInclude] = dfs(node->left);
    auto [rightExclude, rightInclude] = dfs(node->right);

    int exclude = max(leftExclude, leftInclude) +
                  max(rightExclude, rightInclude);
    int include = node->val + leftExclude + rightExclude;

    return dp[node] = {exclude, include};
}

int maxValue = [&]() {
    auto [exclude, include] = dfs(root);
    return max(exclude, include);
}();
```

*See also:* Trees (for tree traversal techniques), Graphs (for general graph DP)

== Common DP Patterns Summary

*Sequence DP:* Linear sequence of decisions
- Climbing Stairs, House Robber, Maximum Subarray
- State: `dp[i]` = optimal solution for first i elements

*Subsequence DP:* Non-contiguous selections
- LIS, LCS, Edit Distance
- State: `dp[i]` or `dp[i][j]` for comparing positions

*Knapsack DP:* Resource allocation
- 0/1 Knapsack, Coin Change, Partition Equal Subset Sum
- State: `dp[i][capacity]` = optimal for first i items with given capacity

*Grid DP:* 2D space navigation
- Unique Paths, Minimum Path Sum, Dungeon Game
- State: `dp[i][j]` = optimal solution reaching position (i,j)

*Interval DP:* Optimal solution over ranges
- Palindrome Partitioning, Burst Balloons
- State: `dp[i][j]` = optimal for interval [i,j]

*Tree DP:* Decisions on tree nodes
- House Robber III, Binary Tree Cameras
- State: `dp[node][state]` = optimal for subtree rooted at node

*Bitmask DP:* Subset-based states
- TSP, Assignment Problem, Hamiltonian Path
- State: `dp[mask]` where mask represents subset

*See also:* Backtracking (for exhaustive search baseline), Greedy (when DP unnecessary), Graphs (for shortest path DP variants)

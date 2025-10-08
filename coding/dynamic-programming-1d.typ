= 1-D Dynamic Programming

*Cache performance:* DP on arrays = sequential access = excellent prefetching. CPU loads entire cache line (64 bytes = 16 ints) per access, amortizing latency.

== Climbing Stairs

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

== House Robber

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

== Longest Palindromic Substring

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

== Coin Change

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

== Maximum Product Subarray

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

== Longest Increasing Subsequence

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

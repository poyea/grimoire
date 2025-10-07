= 1-D Dynamic Programming

== Climbing Stairs

*Problem:* Count distinct ways to climb n stairs (can take 1 or 2 steps at a time).

*Approach 1 - DP Array:* $O(n)$ time, $O(n)$ space
- Initialize `dp = [0] * (n+1)`
- Base cases: `dp[1] = 1`, `dp[2] = 2`
- For i in range(3, n+1): `dp[i] = dp[i-1] + dp[i-2]`
- Return `dp[n]`

*Approach 2 - Space Optimized:* $O(n)$ time, $O(1)$ space
- If `n <= 2`: return n
- Initialize `prev1 = 1`, `prev2 = 2`
- For i in range(3, n+1):
  + `curr = prev1 + prev2`
  + `prev1 = prev2`, `prev2 = curr`
- Return `prev2`

*Key insight:* Same as Fibonacci sequence.

== House Robber

*Problem:* Maximum money you can rob from houses without robbing adjacent houses.

*Approach - DP:* $O(n)$ time, $O(n)$ space
- Initialize `dp = [0] * len(nums)`
- Base cases: `dp[0] = nums[0]`, `dp[1] = max(nums[0], nums[1])`
- For i in range(2, len(nums)):
  + `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`
- Return `dp[-1]`

*Key insight:* Either rob current house + i-2, or skip and take i-1.

== House Robber II

*Problem:* Houses are in a circle (first and last are adjacent).

*Approach - Two DP Runs:* $O(n)$ time, $O(n)$ space
- If `len(nums) == 1`: return `nums[0]`
- Define helper function using House Robber I logic
- Run twice: once excluding last house, once excluding first house
- Return `max(helper(nums[:-1]), helper(nums[1:]))`

*Key insight:* Can't rob both first and last house - try excluding each.

== Longest Palindromic Substring

*Problem:* Find longest palindromic substring.

*Approach - Expand Around Center:* $O(n^2)$ time, $O(1)$ space
- Initialize `result = ""`, `maxLen = 0`
- For each index i:
  + Check odd-length palindromes: expand from `(i, i)`
  + Check even-length palindromes: expand from `(i, i+1)`
  + For each expansion:
    - While `l >= 0 and r < len(s) and s[l] == s[r]`:
      + If `(r - l + 1) > maxLen`: update result
      + Expand: `l -= 1`, `r += 1`
- Return `result`

*Key insight:* Palindrome can be expanded from center - check all possible centers.

== Palindromic Substrings

*Problem:* Count all palindromic substrings.

*Approach - Expand Around Center:* $O(n^2)$ time, $O(1)$ space
- Same as Longest Palindromic Substring
- Instead of tracking longest, increment counter for each palindrome found

== Decode Ways

*Problem:* Count ways to decode string where 'A'=1, 'B'=2, ..., 'Z'=26.

*Approach - DP:* $O(n)$ time, $O(n)$ space
- If `s[0] == '0'`: return 0
- Initialize `dp = [0] * len(s)`, `dp[0] = 1`
- Handle `dp[1]` based on valid single/double digit combinations
- For i in range(2, len(s)):
  + If `s[i] == '0'`: only valid if previous is '1' or '2'
  + Check if two-digit combo is valid (10-26)
  + Update `dp[i]` accordingly
- Return `dp[-1]`

*Key insight:* At each position, can decode as single digit or two-digit number if valid.

== Coin Change

*Problem:* Find minimum coins needed to make amount (infinite supply of each coin).

*Approach - DP:* $O(op("amount") × op("coins"))$ time, $O(op("amount"))$ space
- Initialize `dp = [float('inf')] * (amount + 1)`
- Base case: `dp[0] = 0`
- For i in range(1, amount + 1):
  + For each coin:
    - If `i >= coin`: `dp[i] = min(dp[i], dp[i - coin] + 1)`
- Return `dp[amount]` if not infinity, else -1

*Key insight:* Build up from 0 to amount, trying each coin at each step.

== Maximum Product Subarray

*Problem:* Find contiguous subarray with largest product.

*Approach - Track Min/Max:* $O(n)$ time, $O(1)$ space
- Initialize `result = max(nums)`, `currMin = 1`, `currMax = 1`
- For each num:
  + If `num == 0`: reset `currMin = 1`, `currMax = 1`, continue
  + Store temp: `temp = currMax`
  + Update `currMax = max(num * currMax, num * currMin, num)`
  + Update `currMin = min(num * temp, num * currMin, num)`
  + Update `result = max(result, currMax)`
- Return `result`

*Key insight:* Track both min and max because negative numbers can flip min to max.

== Word Break

*Problem:* Determine if string can be segmented into words from dictionary.

*Approach - DP (Bottom-up):* $O(n #sym.times m #sym.times k)$ time where m is dict size, k is avg word length
- Initialize `dp = [False] * (len(s) + 1)`, `dp[-1] = True`
- For i in range(len(s) - 1, -1, -1):
  + For each word in wordDict:
    * If `s[i:i+len(word)] == word`:
      - `dp[i] = dp[i + len(word)]`
    * If `dp[i]`: break
- Return `dp[0]`

*Key insight:* Work backwards - position i is valid if any word matches and rest is valid.

==  Longest Increasing Subsequence

*Problem:* Find length of longest strictly increasing subsequence.

*Approach - DP:* $O(n^2)$ time, $O(n)$ space
- Initialize `dp = [1] * len(nums)` (each element is subsequence of length 1)
- For i in range(len(nums) - 1, -1, -1):
  + For j in range(i + 1, len(nums)):
    - If `nums[j] > nums[i]`:
      + `dp[i] = max(dp[i], dp[j] + 1)`
- Return `max(dp)`

*Key insight:* For each position, check all larger elements after it.

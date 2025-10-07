= Greedy

== Maximum Subarray

*Problem:* Find contiguous subarray with maximum sum.

*Approach - Kadane's Algorithm:* $O(n)$ time, $O(1)$ space
- Initialize `maxSum = float('-inf')`, `currSum = 0`
- For each num in nums:
  + Add to current sum: `currSum += num`
  + Update max: `maxSum = max(maxSum, currSum)`
  + If `currSum < 0`: reset `currSum = 0` (discard negative prefix)
- Return `maxSum`

*Key insight:* Negative sum prefix can never help future subarrays - discard it.

== Jump Game

*Problem:* Determine if you can reach last index (each element is max jump length).

*Approach - Greedy (Backwards):* $O(n)$ time, $O(1)$ space
- Initialize `goal = len(nums) - 1` (target to reach)
- For i in range(len(nums) - 2, -1, -1):
  + If `i + nums[i] >= goal`:
    - Update `goal = i` (can reach from here)
- Return `goal == 0`

*Key insight:* Work backwards - if we can reach position i, we just need to reach i (not the end).

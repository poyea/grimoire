= Greedy

== Maximum Subarray (Kadane's Algorithm)

*Problem:* Find contiguous subarray with maximum sum.

*Approach - Kadane's Algorithm:* $O(n)$ time, $O(1)$ space

```cpp
int maxSubArray(vector<int>& nums) {
    int maxSum = nums[0];
    int currSum = 0;

    for (int num : nums) {
        currSum = max(0, currSum + num);  // Reset if negative
        maxSum = max(maxSum, currSum);
    }

    return maxSum;
}
```

*Key insight:* Negative prefix sum can never help future subarrays - discard it.

*Branch prediction:* `max(0, currSum + num)` becomes `CMOV` (conditional move). No branch, no mispredicts. Perfect for modern CPUs.

*Assembly pattern (x86-64):*
```asm
add  ecx, edx        ; currSum += num
xor  eax, eax        ; eax = 0
cmp  ecx, eax
cmovl ecx, eax       ; if currSum < 0: currSum = 0
```

*SIMD vectorization challenges:* Kadane's algorithm has loop-carried dependencies (`currSum` depends on previous iteration's `currSum`), making effective SIMD parallelization difficult. Advanced techniques exist but add significant complexity (e.g., processing multiple independent streams, prefix sum approaches).

*For high-performance needs:* Consider data-level parallelism by processing multiple arrays simultaneously rather than trying to vectorize a single array's Kadane's algorithm.

== Jump Game

*Problem:* Determine if you can reach last index (each element is max jump length).

*Approach - Greedy (Backwards):* $O(n)$ time, $O(1)$ space

```cpp
bool canJump(vector<int>& nums) {
    int goal = nums.size() - 1;

    for (int i = nums.size() - 2; i >= 0; i--) {
        if (i + nums[i] >= goal) {
            goal = i;
        }
    }

    return goal == 0;
}
```

*Key insight:* Work backwards - if we can reach position i, we just need to reach i (not the end).

*Branch prediction:* `if (i + nums[i] >= goal)` predictability depends on input. Sorted array = predictable. Random = unpredictable. $#sym.tilde.op$10-20% of iterations update goal typically.

*Branchless alternative:*
```cpp
bool canJump(vector<int>& nums) {
    int goal = nums.size() - 1;

    for (int i = nums.size() - 2; i >= 0; i--) {
        bool canReach = (i + nums[i] >= goal);
        goal = canReach ? i : goal;  // CMOV in assembly
    }

    return goal == 0;
}
```

Eliminates branches but computes `i + nums[i]` every iteration (vs lazy evaluation with if). Profile to decide.

*Forward greedy (alternative):*
```cpp
bool canJump(vector<int>& nums) {
    int maxReach = 0;

    for (int i = 0; i <= maxReach && i < nums.size(); i++) {
        maxReach = max(maxReach, i + nums[i]);
    }

    return maxReach >= nums.size() - 1;
}
```

Better cache behavior (forward iteration) but more complex loop condition.

*Cache:* Backward iteration = poor spatial locality (going against prefetcher). Forward iteration = excellent locality. For large arrays: forward version can be 2-3x faster.

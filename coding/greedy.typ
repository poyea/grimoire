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

*SIMD vectorization:* Process 8 elements in parallel with AVX2. Requires horizontal max reduction at end.

```cpp
// Conceptual AVX2 version (simplified):
__m256i curr = _mm256_setzero_si256();
__m256i maxVec = _mm256_set1_epi32(INT_MIN);

for (int i = 0; i < n; i += 8) {
    __m256i nums_vec = _mm256_loadu_si256((__m256i*)&nums[i]);
    curr = _mm256_add_epi32(curr, nums_vec);
    curr = _mm256_max_epi32(curr, _mm256_setzero_si256());
    maxVec = _mm256_max_epi32(maxVec, curr);
}
// Horizontal max of maxVec...
```

*When to vectorize:* Arrays > 10K elements. Overhead of setup/reduction not worth it for small arrays.

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

*Branch prediction:* `if (i + nums[i] >= goal)` predictability depends on input. Sorted array = predictable. Random = unpredictable. ~10-20% of iterations update goal typically.

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

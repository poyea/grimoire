= Binary Search

== Find Minimum in Rotated Sorted Array

*Problem:* Find the minimum element in a rotated sorted array (no duplicates).

*Approach - Binary Search:* $O(log n)$ time, $O(1)$ space

```cpp
int findMin(vector<int>& nums) {
    int result = nums[0];
    int left = 0, right = nums.size() - 1;

    while (left <= right) {
        // Early exit: sorted subarray
        if (nums[left] < nums[right]) {
            result = min(result, nums[left]);
            break;
        }

        int mid = left + (right - left) / 2;  // Avoid overflow
        result = min(result, nums[mid]);

        // Minimum is in right half
        if (nums[mid] >= nums[left]) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    return result;
}
```

*Cache & Branch Optimization:*
- Array scan: sequential access = prefetcher friendly when falling back to linear scan
- Branch predictor trains on sorted/unsorted pattern after few iterations
- Avoid `(left + right) / 2` due to integer overflow. Use `left + (right - left) / 2`
- Modern compilers automatically optimize division by power-of-2 to right shift

*Hardware insight:* Binary search has poor cache locality (random access pattern). For small arrays (n < 32-128 elements), linear scan can outperform due to prefetching and branch prediction. The exact crossover depends on data type and access patterns.

== Search in Rotated Sorted Array

*Problem:* Search for target in rotated sorted array, return index or -1.

*Approach - Modified Binary Search:* $O(log n)$ time, $O(1)$ space

```cpp
int search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (nums[mid] == target) return mid;

        // Determine which half is sorted
        if (nums[mid] >= nums[left]) {
            // Left half is sorted
            if (nums[left] <= target && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            // Right half is sorted
            if (nums[mid] < target && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    return -1;
}
```

*Key insight:* At least one half is always sorted - exploit this to determine search direction.

*Branch prediction:* Pattern depends on rotation point. CPU branch predictor achieves $#sym.tilde.op$ 85-90% accuracy after warmup. Use `__builtin_expect()` if rotation point is known to be left/right biased.

*SIMD alternative:* For small arrays (n < 32) with high query rate, consider linear SIMD scan with `_mm256_cmpeq_epi32()` - avoids branch mispredicts entirely.

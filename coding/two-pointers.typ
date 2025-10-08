= Two Pointers

== Valid Palindrome

*Problem:* Check if string is palindrome (alphanumeric only, case-insensitive).

*Two Pointers:* $O(n)$ time, $O(1)$ space

```cpp
bool isPalindrome(string s) {
    int l = 0, r = s.length() - 1;

    while (l < r) {
        while (l < r && !isalnum(s[l])) l++;
        while (l < r && !isalnum(s[r])) r--;

        if (tolower(s[l++]) != tolower(s[r--])) return false;
    }
    return true;
}
```

== Two Sum II - Sorted Array

*Problem:* Find two indices where `nums[i] + nums[j] == target`. Return 1-indexed.

*Two Pointers:* $O(n)$ time, $O(1)$ space

```cpp
vector<int> twoSum(vector<int>& nums, int target) {
    int l = 0, r = nums.size() - 1;

    while (l < r) {
        int sum = nums[l] + nums[r];
        if (sum == target) return {l + 1, r + 1};
        else if (sum < target) l++;
        else r--;
    }
    return {};
}
```

*Why it works:* Array is sorted. If sum too small, increase left. If sum too large, decrease right.

== 3Sum

*Problem:* Find all unique triplets that sum to zero.

*Sort + Two Pointers:* $O(n^2)$ time, $O(1)$ space (excluding output)

```cpp
vector<vector<int>> threeSum(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> result;

    for (int i = 0; i < nums.size(); i++) {
        if (i > 0 && nums[i] == nums[i-1]) continue; // Skip duplicates

        int l = i + 1, r = nums.size() - 1;
        while (l < r) {
            int sum = nums[i] + nums[l] + nums[r];
            if (sum == 0) {
                result.push_back({nums[i], nums[l], nums[r]});
                l++;
                while (l < r && nums[l] == nums[l-1]) l++; // Skip duplicates
            } else if (sum < 0) {
                l++;
            } else {
                r--;
            }
        }
    }
    return result;
}
```

*Critical:* Skip duplicates for first element and after finding triplet to avoid duplicate results.

== Container With Most Water

*Problem:* Find max area formed by two vertical lines.

*Two Pointers:* $O(n)$ time, $O(1)$ space

```cpp
int maxArea(vector<int>& height) {
    int l = 0, r = height.size() - 1;
    int maxArea = 0;

    while (l < r) {
        int area = min(height[l], height[r]) * (r - l);
        maxArea = max(maxArea, area);

        if (height[l] < height[r]) l++;
        else r--;
    }
    return maxArea;
}
```

*Greedy choice:* Always move pointer with smaller height. Moving larger height can never improve area (width decreases, height stays ≤ min).

== Two-Pointer Performance

*Prefetching:*
```cpp
// Manual prefetch hints for large arrays
int twoSum(vector<int>& nums, int target) {
    int l = 0, r = nums.size() - 1;

    while (l < r) {
        __builtin_prefetch(&nums[l + 1]);  // Hint: will access soon
        __builtin_prefetch(&nums[r - 1]);

        int sum = nums[l] + nums[r];
        if (sum == target) return {l + 1, r + 1};
        else if (sum < target) l++;
        else r--;
    }
    return {};
}
```

*Memory access pattern:*
- Left pointer: forward, sequential = prefetcher friendly
- Right pointer: backward, sequential = prefetcher can adapt
- Both converge = good spatial locality as pointers get closer

*Cache line reuse:*
When `l` and `r` point to same cache line (64 bytes apart = ~16 ints), single cache line loaded for both accesses. Final iterations = very cache-efficient.

*Branch prediction:*
`if-else` chain: pattern depends on data distribution. Sorted + target near median = branches alternate = poor prediction. Use profile-guided optimization (PGO) for critical paths.

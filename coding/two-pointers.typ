= Two Pointers

*Two pointers technique processes arrays/sequences with two indices moving toward each other or in same direction. Requires: sorted data or specific pattern. Benefits: $O(1)$ space vs $O(n)$ for hash-based solutions.*

*See also:* Sliding Window (for variable-size windows), Hashing (for unsorted data alternatives), Binary Search (for search-based two pointer variants)

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

*See also:* Hashing (for Two Sum on unsorted array with $O(n)$ time but $O(n)$ space)

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

*See also:* Greedy (for proving correctness of greedy strategies), Dynamic Programming (for problems where greedy doesn't work)

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
When `l` and `r` point to same cache line (64 bytes apart = $#sym.tilde.op$16 ints), single cache line loaded for both accesses. This benefit is minor since it only applies to the final $#sym.tilde.op$16 elements of iteration.

*Branch prediction:*
`if-else` chain: pattern depends on data distribution. Sorted + target near median = branches alternate = poor prediction. Use profile-guided optimization (PGO) for critical paths.

== SIMD Palindrome Check

*Vectorized string comparison (AVX2):*
```cpp
#include <immintrin.h>

bool isPalindromeSIMD(const string& s) {
    // Filter to alphanumeric and lowercase
    string filtered;
    filtered.reserve(s.length());
    for (char c : s) {
        if (isalnum(c)) filtered += tolower(c);
    }

    int n = filtered.length();
    int l = 0, r = n - 1;

    // Process 32 chars at once (AVX2 = 256 bits = 32 bytes)
    while (r - l >= 31) {
        __m256i left_chunk = _mm256_loadu_si256((__m256i*)&filtered[l]);

        // Reverse right chunk manually (no native reverse in AVX2)
        char reversed[32];
        for (int i = 0; i < 32; i++) {
            reversed[i] = filtered[r - i];
        }
        __m256i right_chunk = _mm256_loadu_si256((__m256i*)reversed);

        // Compare
        __m256i cmp = _mm256_cmpeq_epi8(left_chunk, right_chunk);
        int mask = _mm256_movemask_epi8(cmp);

        if (mask != -1) return false;  // Not all equal

        l += 32;
        r -= 32;
    }

    // Scalar cleanup
    while (l < r) {
        if (filtered[l++] != filtered[r--]) return false;
    }

    return true;
}
```

*Limitation:* No native reverse instruction. Better approach: SIMD for preprocessing (lowercase, filter), then scalar comparison.

*Practical SIMD use case - batch palindrome checking:*
```cpp
// Check 8 strings in parallel (data-level parallelism)
// Better than vectorizing single string
```

== Branchless Two Pointers

*CMOV-based comparison:*
```cpp
vector<int> twoSumBranchless(vector<int>& nums, int target) {
    int l = 0, r = nums.size() - 1;

    while (l < r) {
        int sum = nums[l] + nums[r];
        bool found = (sum == target);
        bool less = (sum < target);

        // Branchless update using arithmetic
        l += less;              // Increment if sum < target
        r -= !less && !found;   // Decrement if sum >= target and not found

        if (found) return {l + 1, r + 1};  // Still need branch for early exit
    }
    return {};
}

// Assembly uses CMOV:
// cmp    eax, edx
// setl   cl        ; Set cl = 1 if less
// movzx  ecx, cl
// add    esi, ecx  ; l += less
```

*Tradeoff:* Eliminates branch mispredictions (which cost approximately 15-20 cycles each when they occur) but introduces data dependencies through CMOV instructions. Data dependencies force the CPU to wait for previous operations to complete before executing dependent instructions, reducing instruction-level parallelism.#footnote[Instruction-level parallelism (ILP) is the CPU's ability to execute multiple instructions simultaneously using techniques like out-of-order execution and multiple execution units. Data dependencies limit ILP because dependent instructions must wait for their operands.] This approach is only beneficial when branches are unpredictable (such as with random data), where the cost of frequent mispredictions outweighs the serialization penalty.

== 3Sum Optimization - Hash Set Alternative

*Two pointers (original):* $O(n^2)$ with $O(1)$ space.

*Hash set approach:* $O(n^2)$ with $O(n)$ space, better for unsorted input.

```cpp
vector<vector<int>> threeSumHash(vector<int>& nums) {
    set<vector<int>> result_set;

    for (int i = 0; i < nums.size(); i++) {
        unordered_set<int> seen;
        int target = -nums[i];

        for (int j = i + 1; j < nums.size(); j++) {
            int complement = target - nums[j];
            if (seen.count(complement)) {
                vector<int> triplet = {nums[i], nums[j], complement};
                sort(triplet.begin(), triplet.end());
                result_set.insert(triplet);
            }
            seen.insert(nums[j]);
        }
    }

    return vector<vector<int>>(result_set.begin(), result_set.end());
}
```

*When to use:*
- Input already sorted: two pointers
- Input not sorted + many duplicates: hash set (avoid sort cost)
- Input not sorted + few duplicates: sort + two pointers

== Container With Most Water - SIMD Scan

*SIMD brute force approach:* O(n²) complexity but parallelizes inner loop.

```cpp
int maxAreaSIMD(vector<int>& height) {
    int n = height.size();
    int maxArea = 0;

    // For small arrays: try all pairs with SIMD
    for (int l = 0; l < n; l++) {
        int hl = height[l];

        // Process 8 right pointers at once
        int r = l + 8;
        for (; r < n; r += 8) {
            __m256i widths = _mm256_setr_epi32(
                r-7-l, r-6-l, r-5-l, r-4-l,
                r-3-l, r-2-l, r-1-l, r-l
            );

            __m256i heights_r = _mm256_loadu_si256((__m256i*)&height[r-7]);
            __m256i heights_l = _mm256_set1_epi32(hl);

            // min(hl, hr) * width
            __m256i min_h = _mm256_min_epi32(heights_l, heights_r);
            __m256i areas = _mm256_mullo_epi32(min_h, widths);

            // Horizontal max
            int temp[8];
            _mm256_storeu_si256((__m256i*)temp, areas);
            for (int i = 0; i < 8; i++) {
                maxArea = max(maxArea, temp[i]);
            }
        }

        // Scalar cleanup for remaining elements
        for (r = max(l + 1, (r - 8)); r < n; r++) {
            maxArea = max(maxArea, min(height[l], height[r]) * (r - l));
        }
    }

    return maxArea;
}
```

*Complexity:* Still $O(n^2)$ but with 6-8x speedup from SIMD. Two-pointer $O(n)$ algorithm is still better for large n.

*When SIMD brute force wins:* Small n (< 1000) where constant factor matters more than big-O.

== Memory Access Patterns

*Two pointers convergence:*
```cpp
// Left pointer: forward sequential (good)
// Right pointer: backward sequential (good)
// Converge: both in cache as gap closes

// Cache behavior analysis:
// Early iterations: l and r far apart = 2 cache misses per iteration
// Late iterations: l and r close = same cache line = 0-1 miss per iteration
// Average: ~1.5 cache misses per iteration for large arrays
```

*Prefetching for large arrays:*
```cpp
int maxArea(vector<int>& height) {
    int l = 0, r = height.size() - 1;
    int maxArea = 0;

    while (l < r) {
        // Prefetch next iterations
        if (l + 4 < r) {
            __builtin_prefetch(&height[l + 4]);
            __builtin_prefetch(&height[r - 4]);
        }

        int area = min(height[l], height[r]) * (r - l);
        maxArea = max(maxArea, area);

        if (height[l] < height[r]) l++;
        else r--;
    }

    return maxArea;
}
// Speedup: 5-10% for arrays > 10MB
```

== Parallel Two Pointers (Multithreading)

*3Sum with thread-level parallelism:*
```cpp
#include <thread>
#include <mutex>

vector<vector<int>> threeSumParallel(vector<int>& nums, int num_threads = 4) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> result;
    mutex result_mutex;

    auto worker = [&](int start, int end) {
        vector<vector<int>> local_result;

        for (int i = start; i < end; i++) {
            if (i > start && nums[i] == nums[i-1]) continue;

            int l = i + 1, r = nums.size() - 1;
            while (l < r) {
                int sum = nums[i] + nums[l] + nums[r];
                if (sum == 0) {
                    local_result.push_back({nums[i], nums[l], nums[r]});
                    l++;
                    while (l < r && nums[l] == nums[l-1]) l++;
                } else if (sum < 0) {
                    l++;
                } else {
                    r--;
                }
            }
        }

        lock_guard<mutex> lock(result_mutex);
        result.insert(result.end(), local_result.begin(), local_result.end());
    };

    // Partition work
    int chunk_size = nums.size() / num_threads;
    vector<thread> threads;

    for (int t = 0; t < num_threads; t++) {
        int start = t * chunk_size;
        int end = (t == num_threads - 1) ? nums.size() : (t + 1) * chunk_size;
        threads.emplace_back(worker, start, end);
    }

    for (auto& th : threads) th.join();

    return result;
}
```

*Speedup:* 3-4x on 4 cores (not perfect due to synchronization overhead and uneven work distribution).

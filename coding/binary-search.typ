= Binary Search

*Binary search finds target in sorted array in $O(log n)$ time. Key invariant: search space halves each iteration. Template applies to any monotonic function.*

*See also:* Two Pointers (for linear search on sorted arrays), Dynamic Programming (for optimization problems with search space), Trees (binary search trees)

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

*Branch prediction:* Pattern depends on rotation point. CPU branch predictor achieves $#sym.tilde.op$85-90% accuracy after warmup. Use `__builtin_expect()` if rotation point is known to be left/right biased.

*SIMD alternative:* For small arrays (n < 32) with high query rate, consider linear SIMD scan with `_mm256_cmpeq_epi32()` - avoids branch mispredicts entirely.

== Cache-Aware Binary Search

*Problem with standard binary search:*
- Random jumps across array = poor cache locality
- Each comparison can cause cache miss ($#sym.tilde.op$200 cycles)
- Small arrays: entire array fits in cache = not an issue
- Large arrays (> 1MB): most accesses miss L3 = slow

*Linear search crossover point:*
```cpp
// For small arrays, linear search outperforms binary search
template<typename T>
T* search(T* arr, int n, T target) {
    if (n < 64) {
        // Linear: sequential = prefetcher works
        for (int i = 0; i < n; i++) {
            if (arr[i] == target) return &arr[i];
        }
        return nullptr;
    } else {
        // Binary search for large n
        return lower_bound(arr, arr + n, target);
    }
}
```

Crossover typically n = 32-128 depending on element size and access pattern.

*Why the crossover exists:*
- Binary search: O(log n) comparisons, but each comparison may cause a cache miss due to random memory access, costing approximately 200 cycles per miss
- Linear search: O(n) comparisons, but sequential memory access allows the CPU's hardware prefetcher to preload cache lines (64 bytes = 16 integers for int32), reducing the effective cost to approximately 4-5 cycles per element

For small n (typically 32-128 depending on element size), the O(n) × 4 cycles of linear search is less than O(log n) × 200 cycles of binary search with cache misses.

== Eytzinger (BFS) Layout

*Idea:* Store array in breadth-first order like a binary heap. Improves cache locality - children are close to parent in memory.

*Standard layout (sorted):*
```
Index: 0  1  2  3  4  5  6  7  8  9  10
Value: 1  2  3  4  5  6  7  8  9  10 11
```

*Eytzinger layout (BFS order):*
```
Index: 0  1  2  3  4  5  6  7  8  9  10
Value: 6  3  9  2  5  8  10 1  4  7  11
      (root)(L)(R)(LL)(LR)(RL)(RR)...
```

*Conversion:*
```cpp
void eytzinger(const vector<int>& sorted, vector<int>& bfs, int& i, int k = 1) {
    if (k <= sorted.size()) {
        eytzinger(sorted, bfs, i, 2 * k);      // Left child
        bfs[k] = sorted[i++];                  // Current node
        eytzinger(sorted, bfs, i, 2 * k + 1);  // Right child
    }
}

// Usage:
vector<int> sorted = {1,2,3,4,5,6,7,8,9,10,11};
vector<int> bfs(sorted.size() + 1);  // 1-indexed
int idx = 0;
eytzinger(sorted, bfs, idx);
```

*Binary search on Eytzinger layout:*
```cpp
int eytzinger_search(const vector<int>& bfs, int target) {
    if (bfs.empty()) return -1;
    int k = 1;  // Start at root (1-indexed)

    while (k < bfs.size()) {
        // Prefetch both children if they exist
        if (2 * k < bfs.size()) {
            __builtin_prefetch(&bfs[2 * k]);
        }
        if (2 * k + 1 < bfs.size()) {
            __builtin_prefetch(&bfs[2 * k + 1]);
        }

        if (bfs[k] == target) return k;
        else if (target < bfs[k]) k = 2 * k;      // Go left
        else k = 2 * k + 1;                       // Go right
    }
    return -1;  // Not found
}
```

*Performance:*
- Standard binary search: random jumps, $#sym.tilde.op$log n cache misses
- Eytzinger layout: children close to parent, 2-3x fewer cache misses
- Cost: conversion overhead, need to rebuild on updates
- Best for: static data, high query rate

*When to use:* Read-heavy workloads, array > 1MB, infrequent updates.

== Branch-Free Binary Search

*Idea:* Eliminate branches using conditional moves (CMOV). Trade predictable control flow for data dependencies.

```cpp
int branchfree_search(const vector<int>& arr, int target) {
    int lo = 0;
    int hi = arr.size();

    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        // Branchless: update lo or hi based on comparison
        int cmp = arr[mid] < target;
        lo = cmp ? mid + 1 : lo;
        hi = cmp ? hi : mid;
    }

    return (lo < arr.size() && arr[lo] == target) ? lo : -1;
}

// Compiler generates CMOV instructions:
// cmp    eax, edx
// cmovl  ebx, ecx   ; Conditional move if less
```

*Alternative using bit manipulation:*
```cpp
int branchfree_search(const vector<int>& arr, int target) {
    int lo = 0;
    int n = arr.size();

    for (int step = 1 << (31 - __builtin_clz(n)); step > 0; step >>= 1) {
        int mid = lo + step - 1;
        if (mid < n && arr[mid] < target) {
            lo += step;
        }
    }

    return (lo < n && arr[lo] == target) ? lo : -1;
}
```

*Performance tradeoffs:*
- Eliminates branch mispredicts ($#sym.tilde.op$15-20 cycles saved per mispredict)
- Introduces data dependency. The CMOV instruction#footnote[CMOV (Conditional Move) is an x86 instruction that performs a register-to-register move based on a condition flag, eliminating the need for branch instructions. Unlike branches, CMOV executes unconditionally but only commits the result if the condition is true.] has a 1-2 cycle latency, which means subsequent instructions that depend on its result must wait for it to complete. This serializes execution, preventing the CPU from executing other independent instructions in parallel during those cycles.
- Best for: unpredictable data, CPU with slow branch predictor
- Worse for: sorted/predictable data, modern CPUs with good prediction

*Benchmark results (1M elements):*
- Sorted data, sequential queries: standard = 30ms, branchfree = 45ms (worse)
- Random data, random queries: standard = 50ms, branchfree = 40ms (better)

== Interpolation Search

*When binary search isn't optimal:* Uniformly distributed data.

*Idea:* Estimate position based on value, like looking up a phone book.

```cpp
int interpolation_search(const vector<int>& arr, int target) {
    int lo = 0, hi = arr.size() - 1;

    while (lo <= hi && target >= arr[lo] && target <= arr[hi]) {
        if (arr[lo] == arr[hi]) {
            return arr[lo] == target ? lo : -1;
        }

        // Interpolate position
        int pos = lo + ((target - arr[lo]) * (hi - lo)) / (arr[hi] - arr[lo]);

        if (arr[pos] == target) return pos;
        else if (arr[pos] < target) lo = pos + 1;
        else hi = pos - 1;
    }
    return -1;
}
```

*Complexity:*
- Best case (uniform distribution): $O(log log n)$
- Worst case (non-uniform): $O(n)$ - degrades to linear
- Average case (uniform): much better than binary search

*Integer overflow risk:*
```cpp
// BAD: can overflow
int pos = lo + ((target - arr[lo]) * (hi - lo)) / (arr[hi] - arr[lo]);

// GOOD: use double or check overflow
int pos = lo + (int)(((long long)(target - arr[lo]) * (hi - lo)) / (arr[hi] - arr[lo]));
```

*When to use:*
- Data is uniformly distributed (or close to it)
- Large datasets (> 1M elements)
- Cost of comparison is high

*When NOT to use:*
- Data has clusters or skew
- Small datasets (overhead not worth it)

== Exponential Search

*Combine linear + binary:* Good for unbounded/infinite arrays or when target is near beginning.

```cpp
int exponential_search(const vector<int>& arr, int target) {
    if (arr[0] == target) return 0;

    // Find range where target exists
    int bound = 1;
    while (bound < arr.size() && arr[bound] < target) {
        bound *= 2;
    }

    // Binary search in range [bound/2, min(bound, size-1)]
    return binary_search(arr.begin() + bound / 2,
                        arr.begin() + min(bound + 1, (int)arr.size()),
                        target);
}
```

*Complexity:* $O(log i)$ where $i$ is position of target.

*Best for:*
- Unbounded or very large arrays
- Target likely near start
- Unknown array size (streaming data)

== Ternary Search (for Unimodal Functions)

*Finding maximum/minimum of unimodal function:*

```cpp
double ternary_search(function<double(double)> f, double lo, double hi) {
    const double eps = 1e-9;

    while (hi - lo > eps) {
        double m1 = lo + (hi - lo) / 3;
        double m2 = hi - (hi - lo) / 3;

        if (f(m1) < f(m2)) {
            lo = m1;  // Maximum in [m1, hi]
        } else {
            hi = m2;  // Maximum in [lo, m2]
        }
    }

    return (lo + hi) / 2;
}
```

*Complexity:* $O(log_(3/2) n) approx O(1.58 log n)$ - slower than binary search but works for unimodal functions where binary search doesn't apply.

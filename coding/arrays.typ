= Arrays

== Product of Array Except Self

*Problem:* Return array where `result[i] = product of all elements except nums[i]`. No division.

*Optimal - Prefix/Suffix:* $O(n)$ time, $O(1)$ space

```cpp
vector<int> productExceptSelf(vector<int>& nums) {
    int n = nums.size();
    vector<int> res(n, 1);
    // Build prefix products
    for (int i = 1; i < n; i++) {
        res[i] = res[i-1] * nums[i-1];
    }
    // Build suffix products and multiply
    int suffix = 1;
    for (int i = n-1; i >= 0; i--) {
        res[i] *= suffix;
        suffix *= nums[i];
    }
    return res;
}
```

*Key insight:* `res[i] = prefix[i] * suffix[i]`. Reuse output array for prefix, track suffix with scalar.

*SIMD vectorization - why it's limited:*

Product except self has loop-carried dependency (suffix depends on previous iteration). SIMD speedup limited to 1.2-1.5x vs 8x theoretical. The suffix computation is inherently sequential:
```cpp
// Sequential dependency chain (cannot parallelize):
suffix[i] = nums[i] * nums[i+1] * ... * nums[n-1]
```

Better SIMD candidates: element-wise operations without dependencies.

*Better SIMD example - element-wise add:*
```cpp
void addArraysSIMD(const vector<int>& a, const vector<int>& b, vector<int>& c) {
    int n = a.size();
    int i = 0;

    // Process 8 ints at once
    for (; i + 7 < n; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i*)&a[i]);
        __m256i vb = _mm256_loadu_si256((__m256i*)&b[i]);
        __m256i vc = _mm256_add_epi32(va, vb);
        _mm256_storeu_si256((__m256i*)&c[i], vc);
    }

    // Scalar cleanup
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
// Speedup: ~6-7x on modern CPUs (near theoretical 8x)
```

== Vector Growth Strategies

*std::vector reallocation:*
```cpp
vector<int> v;
// Typical growth: capacity doubles when full
v.push_back(1);  // capacity = 1
v.push_back(2);  // capacity = 2 (reallocate + copy)
v.push_back(3);  // capacity = 4 (reallocate + copy)
v.push_back(4);  // capacity = 4 (no realloc)
v.push_back(5);  // capacity = 8 (reallocate + copy)
```

*Amortized analysis:*
- Total copies for n insertions: n/2 + n/4 + n/8 + ... = $O(n)$
- Amortized O(1) per insertion
- But: worst-case single insert = O(n) copy

*Growth factor tradeoffs:*
- Factor 2.0 (typical): fast growth, wastes ~50% memory on average
- Factor 1.5 (MSVC): slower growth, wastes ~33% memory
- Factor φ ≈ 1.618 (golden ratio): theoretical optimal for memory reuse

*Pre-allocation eliminates copies:*
```cpp
vector<int> v;
v.reserve(1000);  // Allocate once
for (int i = 0; i < 1000; i++) {
    v.push_back(i);  // No reallocation
}
```

*reserve() vs resize():*
```cpp
vector<int> v;
v.reserve(100);   // Allocates space, size() = 0
v.resize(100);    // Allocates space, size() = 100, initializes to 0

// reserve(): for incremental push_back
// resize(): when final size known, want default-initialized elements
```

*shrink_to_fit():*
```cpp
vector<int> v(1000);
v.resize(10);  // size = 10, capacity still 1000
v.shrink_to_fit();  // capacity = 10 (may reallocate)
// Use to reclaim memory, costs reallocation
```

== Memory Layout & Alignment

*Cache line alignment (64 bytes):*
```cpp
// BAD: False sharing (multithreading)
struct Counters {
    atomic<int> count1;  // Offset 0
    atomic<int> count2;  // Offset 4 (same cache line!)
};
// Two threads updating count1/count2 = cache ping-pong

// GOOD: Separate cache lines
struct Counters {
    alignas(64) atomic<int> count1;  // Own 64-byte line
    alignas(64) atomic<int> count2;  // Own 64-byte line
};
```

*SIMD alignment (32 bytes for AVX2):*
```cpp
// Aligned allocation for SIMD
alignas(32) int data[8];  // Stack allocation, aligned

// Heap allocation (aligned_alloc C++17)
int* data = (int*)aligned_alloc(32, 1000 * sizeof(int));
// Or use _mm_malloc (compiler intrinsic)
int* data = (int*)_mm_malloc(1000 * sizeof(int), 32);
_mm_free(data);

// Aligned load (faster than unaligned)
__m256i v = _mm256_load_si256((__m256i*)data);  // Requires 32-byte alignment
// vs
__m256i v = _mm256_loadu_si256((__m256i*)data);  // Unaligned (1-2 cycle penalty)
```

*Struct padding:*
```cpp
// Before: 12 bytes with padding
struct Item {
    int id;      // 4 bytes
    // 4 bytes padding
    double val;  // 8 bytes
};  // Total: 16 bytes

// After: Reorder for tight packing
struct Item {
    double val;  // 8 bytes
    int id;      // 4 bytes
    // 4 bytes padding
};  // Still 16 bytes (alignment requirement)

// Packed (avoid unless necessary - slower access)
struct __attribute__((packed)) Item {
    int id;
    double val;
};  // Total: 12 bytes, but unaligned access penalty
```

*Array of Structures (AoS) vs Structure of Arrays (SoA):*
```cpp
// AoS: Traditional, poor cache for partial access
struct Point { int x, y, z; };
vector<Point> points(1000);

for (auto& p : points) {
    p.x += 1;  // Access x, but y,z loaded into cache too (wasted)
}

// SoA: Better cache locality for partial access
struct Points {
    vector<int> x, y, z;
};
Points points;
points.x.resize(1000);
points.y.resize(1000);
points.z.resize(1000);

for (auto& x : points.x) {
    x += 1;  // Only x loaded, perfect cache usage
}
// Enables SIMD: process 8 x values at once
```

*When to use SoA:*
- Frequently access only some fields
- SIMD vectorization needed
- Large datasets (> L3 cache)

*When to use AoS:*
- Usually access all fields together
- Small objects (< 64 bytes)
- Better for random access patterns

== String Encode and Decode

*Problem:* Serialize/deserialize `vector<string>` to single string. Handle delimiters in strings.

*Length Prefix:* $O(n)$ time, $O(1)$ extra space

```cpp
class Codec {
public:
    string encode(vector<string>& strs) {
        string result;
        for (const auto& s : strs) {
            result += to_string(s.length()) + "#" + s;
        }
        return result;
    }
    vector<string> decode(string s) {
        vector<string> result;
        size_t i = 0;

        while (i < s.length()) {
            size_t hash_pos = s.find('#', i);
            int len = stoi(s.substr(i, hash_pos - i));
            result.push_back(s.substr(hash_pos + 1, len));
            i = hash_pos + 1 + len;
        }
        return result;
    }
};
```

*Why it works:* Length prefix avoids delimiter conflicts. Format: `"4#word5#hello"`

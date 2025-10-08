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

== Array Performance Optimization

*Alignment for SIMD:*
```cpp
// Ensure 32-byte alignment for AVX2
alignas(32) int data[1024];

// Dynamic allocation with alignment
int* aligned_data = static_cast<int*>(aligned_alloc(32, n * sizeof(int)));
// Or use _mm_malloc(n * sizeof(int), 32);
```

*SIMD vectorization example:*
```cpp
// Sum array using AVX2 (8 ints at once)
int sum_simd(const int* arr, size_t n) {
    __m256i sum_vec = _mm256_setzero_si256();

    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256i v = _mm256_loadu_si256((__m256i*)&arr[i]);
        sum_vec = _mm256_add_epi32(sum_vec, v);
    }

    // Horizontal sum
    alignas(32) int temp[8];
    _mm256_store_si256((__m256i*)temp, sum_vec);
    int sum = 0;
    for (int j = 0; j < 8; j++) sum += temp[j];

    // Handle remaining elements
    for (; i < n; i++) sum += arr[i];
    return sum;
}
```

*When SIMD helps:*
- Large arrays (> 1000 elements): setup cost amortized
- Simple operations (add, multiply, min, max): single instruction
- Aligned data: `_mm256_load_si256()` faster than `_mm256_loadu_si256()`

*When SIMD doesn't help:*
- Small arrays (< 100 elements): overhead > benefit
- Complex logic with dependencies
- Frequent branches inside loop

*Prefetch hints:*
```cpp
for (int i = 0; i < n; i++) {
    __builtin_prefetch(&arr[i + 16]);  // Prefetch ahead
    process(arr[i]);
}
```

*Cache blocking for 2D arrays:*
```cpp
// BAD: column-major (cache misses)
for (int j = 0; j < cols; j++)
    for (int i = 0; i < rows; i++)
        sum += matrix[i][j];

// GOOD: row-major (cache hits)
for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++)
        sum += matrix[i][j];
```

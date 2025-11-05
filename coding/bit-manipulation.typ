= Bit Manipulation

*Hardware foundation:* Bitwise operations compile to single CPU instructions with 1-cycle latency, 0.25-0.5 cycle throughput on modern CPUs [Intel Opt. Manual 2023, Appx. C]. Branch-free by nature = perfect for pipelined execution.

== Single Number

*Problem:* Find the single number in array where every other number appears twice.

*XOR Properties:* $O(n)$ time, $O(1)$ space

```cpp
int singleNumber(vector<int>& nums) {
    int result = 0;
    for (int num : nums) {
        result ^= num;
    }
    return result;
}
```

*Key properties of XOR:*
- $a xor a = 0$ (self-canceling)
- $a xor 0 = a$ (identity)
- Commutative and associative

*Assembly (x86-64):*
```asm
xor  eax, eax        ; result = 0
loop:
  xor  eax, [rdi]    ; result ^= *ptr (1 cycle latency)
  add  rdi, 4        ; ptr++ (1 cycle)
  loop loop          ; (6 cycles branch mispredict on exit)
```

*SIMD vectorization:*
```cpp
int singleNumberSIMD(vector<int>& nums) {
    __m256i result = _mm256_setzero_si256();

    int i = 0;
    for (; i + 7 < nums.size(); i += 8) {
        __m256i chunk = _mm256_loadu_si256((__m256i*)&nums[i]);
        result = _mm256_xor_si256(result, chunk);  // 8 XORs in parallel
    }

    // Horizontal reduction
    __m128i lo = _mm256_castsi256_si128(result);
    __m128i hi = _mm256_extracti128_si256(result, 1);
    __m128i xor128 = _mm_xor_si128(lo, hi);

    int final_result = _mm_extract_epi32(xor128, 0) ^
                       _mm_extract_epi32(xor128, 1) ^
                       _mm_extract_epi32(xor128, 2) ^
                       _mm_extract_epi32(xor128, 3);

    // Scalar cleanup
    for (; i < nums.size(); i++) {
        final_result ^= nums[i];
    }

    return final_result;
}
```

Speedup: ~4-5x (memory bandwidth limited, not compute limited).

== Number of 1 Bits (Hamming Weight)

*Problem:* Count set bits in integer.

*Approach 1 - Loop:* $O(k)$ where k = number of set bits

```cpp
int hammingWeight(uint32_t n) {
    int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}
```

*Approach 2 - Brian Kernighan's Algorithm:* $O(k)$ but faster

```cpp
int hammingWeight(uint32_t n) {
    int count = 0;
    while (n) {
        n &= n - 1;  // Clear lowest set bit
        count++;
    }
    return count;
}
```

*Why it works:* `n - 1` flips all trailing zeros and the lowest set bit. ANDing clears that bit.

Example: `n = 0b1011000`
- `n - 1 = 0b1010111`
- `n & (n-1) = 0b1010000` (cleared lowest 1)

*Approach 3 - Hardware Intrinsic:* $O(1)$

```cpp
int hammingWeight(uint32_t n) {
    return __builtin_popcount(n);
}
```

*Performance:* Compiles to POPCNT instruction (single cycle on modern CPUs, 3 cycles latency typical) [Intel Opt. Manual 2023, Appx. C]. Requires `-mpopcnt` or `-march=native`.

Without hardware support, compiler generates lookup table or parallel bit manipulation (slower).

*SWAR (SIMD Within A Register):* Software fallback when POPCNT unavailable

```cpp
int popcount_swar(uint32_t n) {
    n = n - ((n >> 1) & 0x55555555);                    // Pair reduction
    n = (n & 0x33333333) + ((n >> 2) & 0x33333333);     // Nibble reduction
    n = (n + (n >> 4)) & 0x0F0F0F0F;                    // Byte reduction
    n = n + (n >> 8);                                    // 16-bit reduction
    n = n + (n >> 16);                                   // 32-bit reduction
    return n & 0x3F;
}
```

Takes ~10-15 cycles vs 3 cycles for POPCNT.

== Counting Bits

*Problem:* Count set bits for all numbers 0 to n.

*Approach 1 - DP with Bit Trick:* $O(n)$ time, $O(n)$ space

```cpp
vector<int> countBits(int n) {
    vector<int> dp(n + 1);
    for (int i = 1; i <= n; i++) {
        dp[i] = dp[i >> 1] + (i & 1);
        // Or: dp[i] = dp[i & (i-1)] + 1;
    }
    return dp;
}
```

*Key insight:*
- `dp[i >> 1]` = count for number with lowest bit removed
- `(i & 1)` = contribution of lowest bit (0 or 1)

*Cache behavior:* Sequential write pattern = excellent. Entire `dp` array for small n (< 256K) fits in L2 cache.

*Approach 2 - Vectorized POPCNT:*

```cpp
vector<int> countBitsSIMD(int n) {
    vector<int> result(n + 1);

    int i = 0;
    for (; i + 7 < n; i += 8) {
        // Generate sequence: i, i+1, i+2, ..., i+7
        __m256i nums = _mm256_setr_epi32(i, i+1, i+2, i+3, i+4, i+5, i+6, i+7);

        // No native POPCNT for AVX2, must do scalar or use AVX-512 VPOPCNTD
        // Scalar is actually faster due to POPCNT throughput
        for (int j = 0; j < 8; j++) {
            result[i + j] = __builtin_popcount(i + j);
        }
    }

    for (; i <= n; i++) {
        result[i] = __builtin_popcount(i);
    }

    return result;
}
```

*AVX-512 alternative:* `_mm512_popcnt_epi32` (16 popcounts in parallel, ~3-4 cycles throughput).

== Reverse Bits

*Problem:* Reverse bits of 32-bit unsigned integer.

*Approach 1 - Bit-by-bit:* $O(32) = O(1)$

```cpp
uint32_t reverseBits(uint32_t n) {
    uint32_t result = 0;
    for (int i = 0; i < 32; i++) {
        result = (result << 1) | (n & 1);
        n >>= 1;
    }
    return result;
}
```

*Approach 2 - Divide and Conquer:* $O(log w)$ where w = word size

```cpp
uint32_t reverseBits(uint32_t n) {
    n = ((n >> 1) & 0x55555555) | ((n & 0x55555555) << 1);   // Swap adjacent bits
    n = ((n >> 2) & 0x33333333) | ((n & 0x33333333) << 2);   // Swap adjacent pairs
    n = ((n >> 4) & 0x0F0F0F0F) | ((n & 0x0F0F0F0F) << 4);   // Swap adjacent nibbles
    n = ((n >> 8) & 0x00FF00FF) | ((n & 0x00FF00FF) << 8);   // Swap adjacent bytes
    n = (n >> 16) | (n << 16);                                // Swap halves
    return n;
}
```

*Performance:* 5 operations, fully pipelined, ~2-3 cycles total (limited by dependency chain).

*x86-64 BSWAP + bit reversal:* Byte swap is single instruction (BSWAP, 1 cycle), then reverse bits within bytes using lookup table.

```cpp
uint32_t reverseBits(uint32_t n) {
    static const uint8_t reverse_table[256] = {
        // Precomputed bit reversal for 0-255
        0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, /* ... */
    };

    return (reverse_table[n & 0xFF] << 24) |
           (reverse_table[(n >> 8) & 0xFF] << 16) |
           (reverse_table[(n >> 16) & 0xFF] << 8) |
           (reverse_table[(n >> 24) & 0xFF]);
}
```

Lookup table (256 bytes) fits in L1 cache. Total: ~4-6 cycles including loads.

== Missing Number

*Problem:* Find missing number in array containing [0, n].

*Approach 1 - XOR:* $O(n)$ time, $O(1)$ space

```cpp
int missingNumber(vector<int>& nums) {
    int result = nums.size();
    for (int i = 0; i < nums.size(); i++) {
        result ^= i ^ nums[i];
    }
    return result;
}
```

*Why it works:* XOR of [0..n] with array elements cancels all present numbers.

*Approach 2 - Sum Formula:* $O(n)$ time, $O(1)$ space

```cpp
int missingNumber(vector<int>& nums) {
    int n = nums.size();
    long long expected = (long long)n * (n + 1) / 2;  // Gauss formula
    long long actual = 0;
    for (int num : nums) {
        actual += num;
    }
    return expected - actual;
}
```

*Overflow consideration:* Use `long long` for intermediate computation. For n = 100,000: expected sum ≈ 5 billion (fits in int64).

*Performance comparison:*
- XOR: branch-free, no overflow risk, ~1.2x slower than addition on modern CPUs
- Sum: faster (ADD has 0.25 cycle throughput vs XOR 0.33), but overflow risk

== Sum of Two Integers (No +/- Operators)

*Problem:* Add two integers using only bitwise operations.

*Full Adder Logic:* $O(log n)$ iterations worst case, $O(1)$ average for random inputs

```cpp
int getSum(int a, int b) {
    while (b != 0) {
        int carry = (unsigned)(a & b) << 1;  // Carry bits
        a = a ^ b;                            // Sum without carry
        b = carry;
    }
    return a;
}
```

*Why it works:*
- `a ^ b` computes sum ignoring carry (0+0=0, 0+1=1, 1+0=1, 1+1=0)
- `(a & b) << 1` computes carry (carry when both bits set)
- Repeat until no carry

*Unsigned cast:* Avoids undefined behavior of left-shifting negative numbers.

*Loop iterations:* Equals longest carry propagation chain. Average ~2-3 iterations for random data.

*Hardware parallel adder:* Real CPUs use carry-lookahead or carry-select adders that compute in $O(log w)$ gate delays, achieving 1-cycle latency [Hennessy & Patterson 2017, §3.3].

== Power of Two Check

*Problem:* Check if number is power of 2.

*Bit Trick:* $O(1)$

```cpp
bool isPowerOfTwo(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}
```

*Why it works:* Power of 2 has exactly one bit set. `n & (n-1)` clears lowest bit, resulting in 0.

*Assembly:*
```asm
test  edi, edi         ; Check n > 0
jle   .false
lea   eax, [rdi-1]     ; eax = n - 1
and   eax, edi         ; eax = n & (n-1)
sete  al               ; al = (eax == 0)
ret
.false:
  xor   eax, eax
  ret
```

~4-5 cycles including branch.

*Branchless version:*
```cpp
bool isPowerOfTwo(int n) {
    return (n > 0) & ((n & (n - 1)) == 0);
}
```

Uses bitwise AND instead of logical AND to avoid branching. ~3 cycles.

== Subset Generation (All Subsets)

*Problem:* Generate all subsets of set.

*Bit Masking:* $O(n #sym.times 2^n)$ time

```cpp
vector<vector<int>> subsets(vector<int>& nums) {
    int n = nums.size();
    int total = 1 << n;  // 2^n subsets
    vector<vector<int>> result;
    result.reserve(total);

    for (int mask = 0; mask < total; mask++) {
        vector<int> subset;
        for (int i = 0; i < n; i++) {
            if (mask & (1 << i)) {
                subset.push_back(nums[i]);
            }
        }
        result.push_back(move(subset));
    }
    return result;
}
```

*Cache behavior:* Inner loop has branch on `mask & (1 << i)`. For random-like masks, ~50% branch prediction accuracy = poor performance.

*Optimization - Iterate set bits:*
```cpp
vector<vector<int>> subsets(vector<int>& nums) {
    int n = nums.size();
    int total = 1 << n;
    vector<vector<int>> result;
    result.reserve(total);

    for (int mask = 0; mask < total; mask++) {
        vector<int> subset;
        int m = mask;
        while (m) {
            int bit = __builtin_ctz(m);  // Count trailing zeros (find first set bit)
            subset.push_back(nums[bit]);
            m &= m - 1;  // Clear lowest bit
        }
        result.push_back(move(subset));
    }
    return result;
}
```

*CTZ intrinsic:* Compiles to BSF/TZCNT instruction, 3 cycles latency [Intel Opt. Manual 2023].

*Iterating in Gray code order:* Consecutive subsets differ by exactly one element = better cache locality for certain applications.

```cpp
// Generate subsets in Gray code order
for (int gray = 0; gray < (1 << n); gray++) {
    int mask = gray ^ (gray >> 1);  // Binary to Gray code
    // Process subset for 'mask'
}
```

== Hamming Distance

*Problem:* Count positions where bits differ between two integers.

*XOR + Popcount:* $O(1)$

```cpp
int hammingDistance(int x, int y) {
    return __builtin_popcount(x ^ y);
}
```

*XOR highlights differences:* Bits are 1 where x and y differ.

*Batch Hamming distances:* Compute distances for array of pairs.

```cpp
// AVX2: Process 8 pairs in parallel
void hammingDistancesBatch(const vector<pair<int,int>>& pairs, vector<int>& result) {
    result.resize(pairs.size());

    for (int i = 0; i + 7 < pairs.size(); i += 8) {
        // Load 8 x values and 8 y values, XOR them, popcount
        // Requires AVX-512 for vectorized popcount
        for (int j = 0; j < 8; j++) {
            result[i+j] = __builtin_popcount(pairs[i+j].first ^ pairs[i+j].second);
        }
    }
}
```

== Bitwise AND of Numbers Range

*Problem:* Bitwise AND of all numbers in range [left, right].

*Key insight:* Result is common prefix of left and right.

```cpp
int rangeBitwiseAnd(int left, int right) {
    int shift = 0;
    while (left < right) {
        left >>= 1;
        right >>= 1;
        shift++;
    }
    return left << shift;
}
```

*Why it works:* In range [left, right], any bit position that differs will have both 0 and 1 in the range, so AND is 0. Only the common prefix survives.

*Alternative - Clear rightmost different bit:*
```cpp
int rangeBitwiseAnd(int left, int right) {
    while (left < right) {
        right &= right - 1;  // Clear rightmost set bit
    }
    return right;
}
```

Converges faster for small ranges.

*CLZ optimization:*
```cpp
int rangeBitwiseAnd(int left, int right) {
    if (left == 0) return 0;
    int shift = __builtin_clz(left) - __builtin_clz(right);
    if (shift > 0) return 0;  // Different MSB positions

    int diff = __builtin_clz(left ^ right);
    int mask = ~0u << (32 - diff);
    return left & mask;
}
```

Uses CLZ (count leading zeros) intrinsic, compiles to LZCNT (3 cycles) [Intel Opt. Manual 2023, Appx. C].

== Bit Packing for Cache Efficiency

*Problem:* Store boolean flags efficiently.

*Naive approach:* `vector<bool>` (bit-packed but slow access)

```cpp
vector<bool> flags(1000000);  // 1M bits = 125KB
flags[i] = true;               // Slow: read-modify-write, no atomic
```

*Manual bit packing:* Better control, faster access

```cpp
class BitArray {
    vector<uint64_t> data;
    size_t size_;

public:
    BitArray(size_t n) : size_(n), data((n + 63) / 64) {}

    void set(size_t i) {
        data[i >> 6] |= 1ULL << (i & 63);
    }

    void clear(size_t i) {
        data[i >> 6] &= ~(1ULL << (i & 63));
    }

    bool get(size_t i) const {
        return (data[i >> 6] >> (i & 63)) & 1;
    }
};
```

*Performance:*
- `i >> 6` = divide by 64 (shift, 1 cycle)
- `i & 63` = modulo 64 (AND, 1 cycle)
- Bit operations: 1-2 cycles each
- Total: ~4-6 cycles per operation vs 10-15 for `vector<bool>`

*Cache benefit:* 64 bools packed into 8 bytes (one cache line = 512 bools). 8x reduction vs `vector<char>`.

*SIMD bit operations:* Process 256 bits at once with AVX2.

```cpp
// Set all bits in range [start, end)
void setRange(size_t start, size_t end) {
    size_t start_word = start >> 6;
    size_t end_word = end >> 6;

    if (start_word == end_word) {
        uint64_t mask = ((1ULL << (end - start)) - 1) << (start & 63);
        data[start_word] |= mask;
    } else {
        // First partial word
        data[start_word] |= ~0ULL << (start & 63);

        // Full words in between (vectorize this loop)
        for (size_t i = start_word + 1; i < end_word; i++) {
            data[i] = ~0ULL;
        }

        // Last partial word
        if (end & 63) {
            data[end_word] |= (1ULL << (end & 63)) - 1;
        }
    }
}
```

== Advanced Bit Tricks

*Isolate lowest set bit:*
```cpp
int lowestBit = x & -x;
// Example: x = 0b1011000 → lowestBit = 0b0001000
```

*Clear lowest set bit:*
```cpp
x &= x - 1;  // Brian Kernighan's trick
```

*Round up to next power of 2:*
```cpp
int nextPowerOf2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}
```

Or using CLZ:
```cpp
int nextPowerOf2(int n) {
    if (n <= 1) return 1;
    return 1 << (32 - __builtin_clz(n - 1));
}
```

*Swap without temporary:*
```cpp
a ^= b;
b ^= a;
a ^= b;
// Now a and b are swapped
```

Not recommended - modern compilers optimize normal swap to register moves. XOR swap has dependency chain (slower).

*Check if two numbers have opposite signs:*
```cpp
bool oppositeSigns(int x, int y) {
    return (x ^ y) < 0;  // Sign bit differs
}
```

*Compute absolute value without branching:*
```cpp
int abs_branchless(int x) {
    int mask = x >> 31;  // All 1s if negative, all 0s if positive
    return (x ^ mask) - mask;
}
```

Equivalent to `(x + mask) ^ mask` form. ~3 cycles vs ~5 cycles for branched `abs()` on unpredictable input.

== Bit Manipulation in Hash Functions

*FNV-1a hash (common in hash tables):*
```cpp
uint32_t fnv1a_hash(const char* str, size_t len) {
    uint32_t hash = 2166136261u;  // FNV offset basis
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint8_t)str[i];
        hash *= 16777619u;  // FNV prime
    }
    return hash;
}
```

*Multiply-shift hash:*
```cpp
uint32_t mult_shift_hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x85ebca6b;  // Random-like constant
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
}
```

Avalanche property: flipping any input bit changes ~50% of output bits. Used in hash table mixing.

*MurmurHash3 finalizer:*
```cpp
uint32_t murmur3_finalizer(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}
```

Total: ~7-9 cycles, excellent avalanche, non-cryptographic.

== References

*Primary Sources:*

*Intel Corporation (2023)*. Intel 64 and IA-32 Architectures Optimization Reference Manual. Order Number 248966-046.

*Agner Fog (2023)*. Instruction Tables: Lists of Instruction Latencies, Throughputs and Micro-operation Breakdowns for Intel, AMD and VIA CPUs.

*Warren, H.S. (2012)*. Hacker's Delight (2nd ed.). Addison-Wesley. ISBN 978-0321842688.

*Algorithms & Theory:*

*Knuth, D.E. (1997)*. The Art of Computer Programming, Volume 1: Fundamental Algorithms (3rd ed.). Addison-Wesley.

*Anderson, S.E. (2005)*. Bit Twiddling Hacks. Stanford University Graphics Lab.

*Hennessy, J.L. & Patterson, D.A. (2017)*. Computer Architecture: A Quantitative Approach (6th ed.). Morgan Kaufmann. ISBN 978-0128119051.

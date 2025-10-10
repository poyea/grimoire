= Hashing

== Contains Duplicate

*Problem:* Check if array has duplicates.

*Hash Set:* $O(n)$ time, $O(n)$ space

```cpp
bool containsDuplicate(vector<int>& nums) {
    unordered_set<int> seen;
    for (int num : nums) {
        if (seen.count(num)) return true;
        seen.insert(num);
    }
    return false;
}
```

*Alternative:* Sort and adjacent check for $O(1)$ space but $O(n log n)$ time.

== Valid Anagram

*Problem:* Check if two strings are anagrams.

*Frequency Map:* $O(n)$ time, $O(1)$ space (26 chars)

```cpp
bool isAnagram(string s, string t) {
    if (s.length() != t.length()) return false;

    unordered_map<char, int> count;
    for (char c : s) count[c]++;
    for (char c : t) {
        if (--count[c] < 0) return false;
    }
    return true;
}
```

*Optimization:* Use `array<int, 26>` for O(1) space if only lowercase letters.

== Two Sum

*Problem:* Find indices `i, j` where `nums[i] + nums[j] == target`.

*Hash Map:* $O(n)$ time, $O(n)$ space

```cpp
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> seen;

    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];
        if (seen.count(complement)) {
            return {seen[complement], i};
        }
        seen[nums[i]] = i;
    }
    return {};
}
```

*Critical:* Check complement before inserting to avoid using same element twice.

#pagebreak()

== Group Anagrams

*Problem:* Group anagrams together.

*Sorted Key:* $O(n #sym.times k log k)$ time where $k = "max string length"$

```cpp
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> groups;
    for (string& s : strs) {
        string key = s;
        sort(key.begin(), key.end());
        groups[key].push_back(s);
    }
    vector<vector<string>> result;
    for (auto& [_, group] : groups) {
        result.push_back(move(group));
    }
    return result;
}
```

*Optimization:* Use char count array as key for $O(n #sym.times k)$ time:

```cpp
string getKey(const string& s) {
    array<int, 26> count = {};
    for (char c : s) count[c - 'a']++;
    string key;
    for (int i = 0; i < 26; i++) {
        if (count[i]) {
            key += string(count[i], 'a' + i);
        }
    }
    return key;
}
```

== Top K Frequent Elements

*Problem:* Return k most frequent elements.

*Bucket Sort:* $O(n)$ time, $O(n)$ space

```cpp
vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> freq;
    for (int num : nums) freq[num]++;
    int n = nums.size();
    vector<vector<int>> buckets(n + 1);
    for (auto& [num, count] : freq) {
        buckets[count].push_back(num);
    }
    vector<int> result;
    for (int i = n; i >= 0 && result.size() < k; i--) {
        for (int num : buckets[i]) {
            result.push_back(num);
            if (result.size() == k) return result;
        }
    }
    return result;
}
```

*Alternative - Min Heap:* $O(n log k)$ time, better for streaming data:

```cpp
// Use priority_queue with custom comparator
priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> minHeap;
// Keep heap size ≤ k, top k elements have highest frequency
```

== Longest Consecutive Sequence

*Problem:* Find length of longest consecutive sequence in unsorted array.

*Hash Set:* $O(n)$ time, $O(n)$ space

```cpp
int longestConsecutive(vector<int>& nums) {
    unordered_set<int> numSet(nums.begin(), nums.end());
    int maxLen = 0;
    for (int num : numSet) {
        // Only start counting from sequence starts
        if (!numSet.count(num - 1)) {
            int length = 1;
            while (numSet.count(num + length)) {
                length++;
            }
            maxLen = max(maxLen, length);
        }
    }
    return maxLen;
}
```

*Key optimization:* Check `num - 1` to identify sequence starts. Prevents redundant work, ensures O(n).

== Hash Table Deep Dive

*Load factor & performance:*
```cpp
unordered_set<int> s;
s.max_load_factor(0.75);  // Default 1.0
s.reserve(10000);  // Pre-allocate buckets
```

- Load factor = elements / buckets
- Higher load = more collisions = slower lookups (more linked list traversal)
- Lower load = less collisions but more memory
- Optimal: 0.75-0.9 for cache efficiency

*Collision resolution:*
- *Chaining (STL default):* Each bucket = linked list. Cache-hostile for long chains.
- *Open addressing (robin hood, linear probing):* Store in array with probing. Better cache locality but not in STL.

*Hash function analysis:*
```cpp
// Common implementation uses identity hash for integers (not guaranteed by standard)
hash<int>{}(42) == 42;  // Note: identity hash can cause clustering with power-of-2 bucket counts
// Bad pattern: many collisions
for (int i = 0; i < 1000; i += 8) {
    set.insert(i);  // All hash to same bucket % 8
}
// Custom hash to avoid patterns:
struct CustomHash {
    size_t operator()(int x) const {
        x ^= x >> 16;
        x *= 0x85ebca6b;
        x ^= x >> 13;
        return x;
    }
};
unordered_set<int, CustomHash> s;
```

*Memory layout:*
`unordered_set` = array of buckets + linked lists. Each node = $#sym.tilde.op$16-24 bytes depending on implementation (8-byte next ptr, optionally 8-byte cached hash, value + padding). 1000 elements ≈ 16-24KB minimum.

*Cache behavior:*
- Small sets (< 100 elements): entire hash table fits in L1 = fast
- Large sets: bucket array may fit in cache, but nodes scattered = pointer chasing
- Alternative: `flat_hash_set` (absl::) uses open addressing = 2-3x faster

== Robin Hood Hashing (Open Addressing)

*Core idea:* Open addressing with "rich help poor" - minimize variance in probe distances.

*Algorithm:*
- Hash collision → linear probing
- Track PSL (Probe Sequence Length) = distance from ideal position
- On insert: if new element's PSL > existing element's PSL, swap and continue inserting the evicted element
- Result: tight clustering, bounded worst-case search

```cpp
template<typename K, typename V>
class RobinHoodMap {
    struct Entry {
        K key;
        V val;
        uint8_t psl;  // Probe sequence length
        bool occupied;
    };

    vector<Entry> table;
    size_t count = 0;
    float max_load = 0.9;

    size_t hash(const K& key) const {
        return std::hash<K>{}(key) & (table.size() - 1);  // Assumes power-of-2 size
    }

public:
    RobinHoodMap(size_t capacity = 16) : table(capacity) {}

    void insert(K key, V val) {
        if (count + 1 > table.size() * max_load) resize();

        size_t idx = hash(key);
        uint8_t psl = 0;

        while (true) {
            if (!table[idx].occupied) {
                table[idx] = {key, val, psl, true};
                count++;
                return;
            }

            if (table[idx].key == key) {
                table[idx].val = val;  // Update
                return;
            }

            // Robin Hood: swap if new element is "poorer"
            if (psl > table[idx].psl) {
                swap(key, table[idx].key);
                swap(val, table[idx].val);
                swap(psl, table[idx].psl);
            }

            idx = (idx + 1) & (table.size() - 1);
            psl++;
        }
    }

    V* find(const K& key) {
        size_t idx = hash(key);
        uint8_t psl = 0;

        while (table[idx].occupied) {
            if (table[idx].key == key) return &table[idx].val;

            // Early termination: if current PSL < search PSL, element not present
            if (psl > table[idx].psl) return nullptr;

            idx = (idx + 1) & (table.size() - 1);
            psl++;
        }
        return nullptr;
    }

    void resize() {
        vector<Entry> old = move(table);
        table.clear();
        table.resize(old.size() * 2);
        count = 0;

        for (auto& e : old) {
            if (e.occupied) insert(e.key, e.val);
        }
    }
};
```

*Performance characteristics:*
- Average probe length: 1.5-2.0 (vs 2-5 for standard linear probing)
- Cache-friendly: linear probing = sequential access = prefetcher works
- Deletion complexity: tombstones or backward shift (not shown above)
- 2-3x faster than `std::unordered_map` for integer keys

*When to use:*
- Integer or small struct keys (avoid large key copies during swaps)
- High load factor tolerable (0.8-0.9)
- Performance critical hash tables

== Hash Function Quality

*std::hash properties:*
```cpp
// Integer hash (many implementations use identity)
hash<int>{}(42) == 42;  // Common but not guaranteed

// Problem: sequential keys with power-of-2 table size
for (int i = 0; i < 1000; i += 8) {
    set.insert(i);  // All collide at bucket i % 8
}
```

*Better integer hash (mixing function):*
```cpp
struct BetterIntHash {
    size_t operator()(uint64_t x) const {
        // MurmurHash3 finalizer
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccd;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53;
        x ^= x >> 33;
        return x;
    }
};

unordered_set<int, BetterIntHash> s;
```

*String hash (FNV-1a):*
```cpp
struct FNV1aHash {
    size_t operator()(const string& s) const {
        size_t hash = 14695981039346656037ULL;
        for (char c : s) {
            hash ^= static_cast<size_t>(c);
            hash *= 1099511628211ULL;
        }
        return hash;
    }
};
// std::hash<string> already uses good algorithm (implementation-defined)
```

*Pair/tuple hash:*
```cpp
// DON'T: default pair hash doesn't exist
// unordered_set<pair<int,int>> s;  // Compilation error

// Custom pair hash:
struct PairHash {
    size_t operator()(const pair<int,int>& p) const {
        size_t h1 = hash<int>{}(p.first);
        size_t h2 = hash<int>{}(p.second);
        return h1 ^ (h2 << 1);  // Simple combine
    }
};

unordered_set<pair<int,int>, PairHash> s;

// Better: boost::hash_combine equivalent
struct BetterPairHash {
    size_t operator()(const pair<int,int>& p) const {
        size_t seed = hash<int>{}(p.first);
        seed ^= hash<int>{}(p.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};
```

== Hash DOS Attack Prevention

*Adversarial input problem:*
- Attacker crafts keys that all hash to same bucket
- Chaining degrades to O(n) per operation
- Example: many web frameworks vulnerable to POST parameter flooding

*Solution 1: Random seed (std::hash may use this):*
```cpp
struct RandomizedHash {
    static size_t seed;

    size_t operator()(int x) const {
        return hash<int>{}(x) ^ seed;
    }
};
size_t RandomizedHash::seed = random_device{}();
```

*Solution 2: Cryptographic hash (overkill, slow):*
```cpp
#include <openssl/sha.h>
struct SHA256Hash {
    size_t operator()(const string& s) const {
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256((unsigned char*)s.c_str(), s.size(), hash);
        return *(size_t*)hash;  // Use first 8 bytes
    }
};
// 10-50x slower than FNV-1a, only for untrusted input
```

*Solution 3: Limit bucket size:*
```cpp
// If bucket size > threshold, switch to balanced tree (not in STL)
// Java's HashMap does this since Java 8
```

== Advanced Hash Table Variants

*Cuckoo hashing:*
- 2+ hash functions, 2+ tables
- Worst-case O(1) lookup (check both tables)
- Insert: collision → "kick out" existing element → rehash to other table
- Pros: guaranteed O(1) read, no tombstones
- Cons: complex, higher load factor limit (~0.5)

*Flat hash map (Google Abseil / Swiss Tables):*

Modern flat hash map design combining best of multiple techniques.

*Core design (Swiss Tables):*
```cpp
// Metadata byte per slot (SIMD-friendly)
struct Group {
    uint8_t ctrl[16];  // SSE register width
    // ctrl[i] values:
    // - 0x80-0xFF (MSB=1): empty slot
    // - 0x00-0x7F (MSB=0): H2 hash (7 bits of hash)
    // Special values:
    // - 0xFF: never used (empty)
    // - 0xFE: deleted (tombstone)
};

// Layout: [ctrl bytes...] [slots...]
// ctrl and slots are separate for SIMD matching
```

*Collision resolution - quadratic probing with SIMD:*
```cpp
// Probe sequence: triangular numbers (0, 1, 3, 6, 10, 15...)
size_t probe_index(size_t hash, size_t i) {
    return (hash + i * (i + 1) / 2) & mask;
}

// SIMD lookup in group of 16 slots
__m128i ctrl_vec = _mm_loadu_si128((__m128i*)ctrl);
__m128i target = _mm_set1_epi8(h2);  // Broadcast H2 hash
__m128i cmp = _mm_cmpeq_epi8(ctrl_vec, target);
int mask = _mm_movemask_epi8(cmp);

// mask = bitmap of matching slots (check up to 16 at once)
while (mask != 0) {
    int index = __builtin_ctz(mask);  // Find first set bit
    // Check actual key at slots[index]
    mask &= mask - 1;  // Clear lowest bit
}
```

*Key innovations:*
1. *Metadata separation:* Control bytes separate from data = better cache usage
2. *SIMD probing:* Check 16 slots simultaneously with SSE2
3. *H2 hash filtering:* 7-bit secondary hash reduces false positives
4. *Quadratic probing:* Better distribution than linear, avoids clustering

*Memory layout example:*
```
// Group of 16 slots:
ctrl:  [h2][h2][empty][h2][deleted][h2]...[h2]  (16 bytes)
slots: [K,V][K,V][---][K,V][---][K,V]...[K,V]  (16 × sizeof(pair<K,V>))
```

*Why it's fast:*
- SIMD lookup: check 16 ctrl bytes in ~3-5 cycles
- No pointer chasing (vs chaining)
- Good cache locality (vs std::unordered_map)
- Low load factor (typically 87.5% = 14/16 slots used)

*Comparison to other schemes:*
- Robin Hood: simpler, but no SIMD acceleration
- Cuckoo: O(1) worst-case lookup, but higher failure rate on insert
- Chaining (std::unordered_map): poor cache, pointer overhead

*Hopscotch hashing:*
- Hybrid: open addressing + local chaining
- "Neighborhood" = H consecutive slots (H = 32 typical)
- Bitmap tracks which slots in neighborhood are occupied
- Pros: cache-friendly, good for parallel access
- Cons: more complex than Robin Hood

*Performance comparison (1M inserts, random int keys):*
| Hash Table Type          | Insert (ms) | Lookup (ms) | Memory (MB) |
|:-------------------------|------------:|------------:|------------:|
| std::unordered_map       |         450 |         380 |          48 |
| Robin Hood (custom)      |         180 |         120 |          32 |
| absl::flat_hash_map      |         150 |         100 |          28 |
| tsl::robin_map           |         160 |         110 |          30 |

*Recommendation:* Use `absl::flat_hash_map` or `tsl::robin_map` for performance-critical code. Fallback to `std::unordered_map` for simplicity.

*Implementation references:*
- `absl::flat_hash_map`: Swiss Tables design [Kulukundis 2017, CppCon talk]
- `tsl::robin_map`: Robin Hood hashing [Tessil, GitHub]
- `ska::flat_hash_map`: Another Swiss Tables variant

*Further reading:*
- Kulukundis, M. (2017). "Designing a Fast, Efficient, Cache-friendly Hash Table, Step by Step." CppCon 2017.
- Alcantara, D. et al. (2011). "Building an Efficient Hash Table on the GPU." IEEE IPDPS.
- Richter, S. et al. (2015). "A Seven-Dimensional Analysis of Hashing Methods." PVLDB 9(3).

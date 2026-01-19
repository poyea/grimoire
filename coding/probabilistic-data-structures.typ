= Probabilistic Data Structures

*Trade exactness for efficiency:* Probabilistic data structures provide approximate answers with bounded error probability, enabling sub-linear space complexity impossible with exact structures. Widely used in databases, networking, and big data systems [Cormode & Muthukrishnan 2005].

*See also:* Hashing (for hash function design), Bit Manipulation (for bit-level operations), Streaming Algorithms (for related online algorithms)

== Bloom Filters

*Problem:* Test set membership with space efficiency. False positives allowed, false negatives not.

*Core idea:* Use k hash functions to set k bits in a bit array. Query: check if all k bits are set.

*Implementation:*

```cpp
class BloomFilter {
    vector<uint64_t> bits;
    size_t num_bits;
    size_t num_hashes;

    // Double hashing: h_i(x) = h1(x) + i * h2(x)
    pair<uint64_t, uint64_t> hash(const string& item) const {
        uint64_t h1 = 0, h2 = 0;
        for (char c : item) {
            h1 = h1 * 31 + c;
            h2 = h2 * 37 + c;
        }
        return {h1, h2};
    }

    void setBit(size_t pos) {
        bits[pos >> 6] |= (1ULL << (pos & 63));
    }

    bool getBit(size_t pos) const {
        return (bits[pos >> 6] >> (pos & 63)) & 1;
    }

public:
    // n = expected elements, p = target false positive rate
    BloomFilter(size_t n, double p) {
        // Optimal: m = -n * ln(p) / (ln(2))^2
        num_bits = static_cast<size_t>(-n * log(p) / (log(2) * log(2)));
        num_bits = max(num_bits, 64UL);

        // Optimal: k = (m/n) * ln(2)
        num_hashes = static_cast<size_t>(num_bits * log(2) / n);
        num_hashes = max(num_hashes, 1UL);

        bits.resize((num_bits + 63) / 64, 0);
    }

    void insert(const string& item) {
        auto [h1, h2] = hash(item);
        for (size_t i = 0; i < num_hashes; i++) {
            size_t pos = (h1 + i * h2) % num_bits;
            setBit(pos);
        }
    }

    bool mayContain(const string& item) const {
        auto [h1, h2] = hash(item);
        for (size_t i = 0; i < num_hashes; i++) {
            size_t pos = (h1 + i * h2) % num_bits;
            if (!getBit(pos)) return false;
        }
        return true;  // May be false positive
    }
};
```

*Complexity:*
- Space: $O(m)$ bits where $m = -n ln(p) / (ln 2)^2$
- Insert: $O(k)$ hash computations
- Query: $O(k)$ hash computations

*False positive probability:*
$
P("false positive") approx (1 - e^(-k n \/ m))^k
$

For optimal k: $P approx (0.6185)^(m\/n)$

*Example sizing:*
- 1M elements, 1% FP rate: m = 9.6M bits (1.2 MB), k = 7 hashes
- 1M elements, 0.1% FP rate: m = 14.4M bits (1.8 MB), k = 10 hashes

*Cache behavior:* Random bit access = poor locality. Mitigation: blocked Bloom filter (partition into cache-line-sized blocks).

== Counting Bloom Filters

*Problem:* Support deletions in Bloom filter.

*Solution:* Replace bits with counters.

```cpp
class CountingBloomFilter {
    vector<uint8_t> counters;  // 4-bit counters (can pack 2 per byte)
    size_t num_counters;
    size_t num_hashes;

    size_t hash_i(const string& item, size_t i) const {
        uint64_t h = 0;
        for (char c : item) {
            h = h * (31 + i) + c;
        }
        return h % num_counters;
    }

public:
    CountingBloomFilter(size_t n, double p) {
        num_counters = static_cast<size_t>(-n * log(p) / (log(2) * log(2)));
        num_hashes = static_cast<size_t>(num_counters * log(2) / n);
        counters.resize(num_counters, 0);
    }

    void insert(const string& item) {
        for (size_t i = 0; i < num_hashes; i++) {
            size_t pos = hash_i(item, i);
            if (counters[pos] < 255) counters[pos]++;  // Saturating increment
        }
    }

    void remove(const string& item) {
        for (size_t i = 0; i < num_hashes; i++) {
            size_t pos = hash_i(item, i);
            if (counters[pos] > 0) counters[pos]--;
        }
    }

    bool mayContain(const string& item) const {
        for (size_t i = 0; i < num_hashes; i++) {
            if (counters[hash_i(item, i)] == 0) return false;
        }
        return true;
    }
};
```

*Space tradeoff:* 4x more memory than standard Bloom filter (4-bit counters vs 1-bit).

*Counter overflow:* 4 bits sufficient for most cases. Probability of overflow to 16 is negligible for properly sized filters [Fan et al. 2000].

== Cuckoo Filters

*Improvement over Bloom filters:* Support deletion, better locality, often smaller.

*Core idea:* Store fingerprints in buckets using cuckoo hashing.

```cpp
class CuckooFilter {
    static const size_t BUCKET_SIZE = 4;  // Entries per bucket
    static const size_t FINGERPRINT_BITS = 8;
    static const size_t MAX_KICKS = 500;

    vector<array<uint8_t, BUCKET_SIZE>> buckets;
    size_t num_buckets;

    uint8_t fingerprint(const string& item) const {
        uint64_t h = 0;
        for (char c : item) h = h * 31 + c;
        // Ensure non-zero fingerprint
        return (h % 255) + 1;
    }

    size_t hash1(const string& item) const {
        uint64_t h = 0;
        for (char c : item) h = h * 37 + c;
        return h % num_buckets;
    }

    size_t altIndex(size_t i, uint8_t fp) const {
        // i XOR hash(fingerprint)
        return (i ^ (fp * 0x5bd1e995)) % num_buckets;
    }

public:
    CuckooFilter(size_t capacity) {
        num_buckets = (capacity + BUCKET_SIZE - 1) / BUCKET_SIZE;
        num_buckets = max(num_buckets, 1UL);
        buckets.resize(num_buckets);
        for (auto& b : buckets) b.fill(0);
    }

    bool insert(const string& item) {
        uint8_t fp = fingerprint(item);
        size_t i1 = hash1(item);
        size_t i2 = altIndex(i1, fp);

        // Try both buckets
        for (size_t i : {i1, i2}) {
            for (size_t j = 0; j < BUCKET_SIZE; j++) {
                if (buckets[i][j] == 0) {
                    buckets[i][j] = fp;
                    return true;
                }
            }
        }

        // Kick existing entries
        size_t i = (rand() % 2) ? i1 : i2;
        for (size_t n = 0; n < MAX_KICKS; n++) {
            size_t j = rand() % BUCKET_SIZE;
            swap(fp, buckets[i][j]);
            i = altIndex(i, fp);

            for (size_t k = 0; k < BUCKET_SIZE; k++) {
                if (buckets[i][k] == 0) {
                    buckets[i][k] = fp;
                    return true;
                }
            }
        }
        return false;  // Filter is full
    }

    bool contains(const string& item) const {
        uint8_t fp = fingerprint(item);
        size_t i1 = hash1(item);
        size_t i2 = altIndex(i1, fp);

        for (size_t i : {i1, i2}) {
            for (size_t j = 0; j < BUCKET_SIZE; j++) {
                if (buckets[i][j] == fp) return true;
            }
        }
        return false;
    }

    bool remove(const string& item) {
        uint8_t fp = fingerprint(item);
        size_t i1 = hash1(item);
        size_t i2 = altIndex(i1, fp);

        for (size_t i : {i1, i2}) {
            for (size_t j = 0; j < BUCKET_SIZE; j++) {
                if (buckets[i][j] == fp) {
                    buckets[i][j] = 0;
                    return true;
                }
            }
        }
        return false;
    }
};
```

*Complexity:*
- Space: $(2 + epsilon) times f$ bits per element (f = fingerprint bits)
- Insert: $O(1)$ amortized
- Query: $O(1)$ worst case (check 2 buckets)
- Delete: $O(1)$

*Comparison with Bloom filter:*
#table(
  columns: 4,
  align: (left, center, center, center),
  table.header([Property], [Bloom Filter], [Counting Bloom], [Cuckoo Filter]),
  [Deletion], [No], [Yes], [Yes],
  [Space (1% FP)], [9.6 bits/elem], [38.4 bits/elem], [12 bits/elem],
  [Lookup time], [$O(k)$], [$O(k)$], [$O(1)$],
  [Cache locality], [Poor], [Poor], [Good],
)

*When to use Cuckoo:* When deletions needed, or when space efficiency critical with moderate FP rate.

== HyperLogLog

*Problem:* Count distinct elements (cardinality estimation) in stream.

*Exact solution:* Hash set = $O(n)$ space. For 1B elements = ~8GB.

*HyperLogLog:* $O(log log n)$ space with ~2% error.

*Core idea:* Hash each element, count trailing zeros. More trailing zeros = more distinct elements seen.

```cpp
class HyperLogLog {
    static const int P = 14;  // Precision: 2^P registers
    static const int M = 1 << P;  // 16384 registers
    vector<uint8_t> registers;

    uint64_t hash(const string& item) const {
        uint64_t h = 0xcbf29ce484222325ULL;  // FNV offset
        for (char c : item) {
            h ^= c;
            h *= 0x100000001b3ULL;  // FNV prime
        }
        return h;
    }

    int countLeadingZeros(uint64_t x) const {
        if (x == 0) return 64;
        return __builtin_clzll(x);
    }

public:
    HyperLogLog() : registers(M, 0) {}

    void add(const string& item) {
        uint64_t h = hash(item);
        size_t idx = h >> (64 - P);  // First P bits = register index
        uint64_t w = h << P;          // Remaining bits
        int rho = countLeadingZeros(w) + 1;  // Position of first 1
        registers[idx] = max(registers[idx], static_cast<uint8_t>(rho));
    }

    double estimate() const {
        // Harmonic mean of 2^(-register[i])
        double sum = 0;
        int zeros = 0;
        for (int i = 0; i < M; i++) {
            sum += pow(2.0, -registers[i]);
            if (registers[i] == 0) zeros++;
        }

        double raw = alpha() * M * M / sum;

        // Bias correction for small cardinalities
        if (raw <= 2.5 * M && zeros > 0) {
            // Linear counting
            return M * log(static_cast<double>(M) / zeros);
        }

        // Large range correction
        if (raw > (1ULL << 32) / 30.0) {
            return -pow(2, 32) * log(1 - raw / pow(2, 32));
        }

        return raw;
    }

    // Merge two HLL sketches
    void merge(const HyperLogLog& other) {
        for (int i = 0; i < M; i++) {
            registers[i] = max(registers[i], other.registers[i]);
        }
    }

private:
    double alpha() const {
        // Bias correction constant
        switch (P) {
            case 4: return 0.673;
            case 5: return 0.697;
            case 6: return 0.709;
            default: return 0.7213 / (1 + 1.079 / M);
        }
    }
};
```

*Complexity:*
- Space: $O(2^P)$ bytes = 16KB for P=14
- Add: $O(1)$
- Estimate: $O(2^P)$
- Merge: $O(2^P)$

*Accuracy:*
$
"Standard Error" = 1.04 / sqrt(m)
$

For m = 16384 (P=14): SE = 0.81% (typical error within 2-3%)

*Use cases:*
- Unique visitor counting (Redis HLL)
- Database query optimization (cardinality estimation)
- Network traffic analysis

== Count-Min Sketch

*Problem:* Estimate frequency of items in stream.

*Exact solution:* Hash map = $O(n)$ space for n distinct items.

*Count-Min Sketch:* $O(1/epsilon times log(1/delta))$ space with multiplicative error.

```cpp
class CountMinSketch {
    vector<vector<int>> table;
    size_t width;   // Number of counters per row
    size_t depth;   // Number of hash functions

    size_t hash_i(const string& item, size_t i) const {
        uint64_t h = i * 0x9e3779b97f4a7c15ULL;
        for (char c : item) {
            h = h * 31 + c;
        }
        return h % width;
    }

public:
    // epsilon = error factor, delta = failure probability
    CountMinSketch(double epsilon, double delta) {
        width = static_cast<size_t>(ceil(exp(1) / epsilon));
        depth = static_cast<size_t>(ceil(log(1 / delta)));
        table.assign(depth, vector<int>(width, 0));
    }

    void add(const string& item, int count = 1) {
        for (size_t i = 0; i < depth; i++) {
            table[i][hash_i(item, i)] += count;
        }
    }

    int estimate(const string& item) const {
        int min_count = INT_MAX;
        for (size_t i = 0; i < depth; i++) {
            min_count = min(min_count, table[i][hash_i(item, i)]);
        }
        return min_count;
    }

    // Merge two sketches (same dimensions required)
    void merge(const CountMinSketch& other) {
        for (size_t i = 0; i < depth; i++) {
            for (size_t j = 0; j < width; j++) {
                table[i][j] += other.table[i][j];
            }
        }
    }
};
```

*Guarantee:*
With probability $1 - delta$:
$
hat(f)(x) <= f(x) + epsilon times N
$

where $f(x)$ = true frequency, $hat(f)(x)$ = estimated frequency, $N$ = total count.

*Complexity:*
- Space: $O(w times d)$ counters
- Update: $O(d)$
- Query: $O(d)$

*Example sizing:*
- epsilon = 0.01, delta = 0.01: width = 272, depth = 5, total = 5.4KB
- epsilon = 0.001, delta = 0.001: width = 2718, depth = 7, total = 76KB

*Use cases:*
- Heavy hitters detection
- Network flow monitoring
- Database approximate aggregates

== Count-Min Sketch with Conservative Update

*Improvement:* Reduce over-counting by only incrementing the minimum counter(s).

```cpp
void addConservative(const string& item, int count = 1) {
    int min_val = INT_MAX;

    // Find minimum
    for (size_t i = 0; i < depth; i++) {
        min_val = min(min_val, table[i][hash_i(item, i)]);
    }

    // Only increment counters at or below minimum
    for (size_t i = 0; i < depth; i++) {
        size_t idx = hash_i(item, i);
        if (table[i][idx] <= min_val) {
            table[i][idx] = min_val + count;
        }
    }
}
```

*Benefit:* Reduces error for low-frequency items. Same worst-case guarantee, better average case.

== MinHash (Locality-Sensitive Hashing)

*Problem:* Estimate Jaccard similarity between sets.

*Jaccard similarity:* $J(A, B) = |A inter B| / |A union B|$

```cpp
class MinHash {
    size_t num_hashes;
    vector<uint64_t> a, b;  // Hash function parameters

public:
    MinHash(size_t k = 100) : num_hashes(k), a(k), b(k) {
        random_device rd;
        mt19937_64 gen(rd());
        uniform_int_distribution<uint64_t> dist;
        for (size_t i = 0; i < k; i++) {
            a[i] = dist(gen);
            b[i] = dist(gen);
        }
    }

    vector<uint64_t> signature(const unordered_set<int>& s) const {
        vector<uint64_t> sig(num_hashes, UINT64_MAX);

        for (int elem : s) {
            for (size_t i = 0; i < num_hashes; i++) {
                uint64_t h = a[i] * elem + b[i];
                sig[i] = min(sig[i], h);
            }
        }
        return sig;
    }

    double similarity(const vector<uint64_t>& sig1,
                      const vector<uint64_t>& sig2) const {
        size_t matches = 0;
        for (size_t i = 0; i < num_hashes; i++) {
            if (sig1[i] == sig2[i]) matches++;
        }
        return static_cast<double>(matches) / num_hashes;
    }
};
```

*Property:* $P(min hash(A) = min hash(B)) = J(A, B)$

*Accuracy:* With k hash functions, standard error = $1/sqrt(k)$

*Use cases:*
- Near-duplicate detection (web pages, documents)
- Clustering similar items
- Recommendation systems

== Quotient Filters

*Alternative to Bloom/Cuckoo:* Better cache locality, supports merging.

*Core idea:* Store quotient and remainder of hash separately.

```cpp
class QuotientFilter {
    static const int Q = 16;  // Quotient bits
    static const int R = 8;   // Remainder bits
    static const size_t SIZE = 1 << Q;

    struct Slot {
        uint8_t remainder : R;
        bool is_occupied : 1;
        bool is_continuation : 1;
        bool is_shifted : 1;
    };

    vector<Slot> table;

    pair<size_t, uint8_t> split(uint64_t h) const {
        return {h >> R, h & ((1 << R) - 1)};  // quotient, remainder
    }

public:
    QuotientFilter() : table(SIZE) {}

    void insert(uint64_t hash_value) {
        auto [q, r] = split(hash_value);

        // Find run start (series of elements with same quotient)
        size_t run_start = q;
        while (run_start > 0 && table[run_start].is_shifted) {
            run_start--;
        }

        // Find insert position within run
        size_t pos = run_start;
        while (pos < SIZE && (table[pos].is_continuation ||
               (table[pos].is_occupied && table[pos].remainder < r))) {
            pos++;
        }

        // Shift elements and insert
        Slot to_insert = {static_cast<uint8_t>(r), true, pos != q, pos != q};

        while (table[pos].is_occupied) {
            swap(to_insert, table[pos]);
            to_insert.is_shifted = true;
            pos = (pos + 1) % SIZE;
        }
        table[pos] = to_insert;
        table[q].is_occupied = true;
    }

    bool mayContain(uint64_t hash_value) const {
        auto [q, r] = split(hash_value);
        if (!table[q].is_occupied) return false;

        size_t pos = q;
        while (pos > 0 && table[pos].is_shifted) pos--;

        // Scan runs up to quotient q
        for (size_t i = 0; i <= q; i++) {
            if (!table[i].is_occupied) continue;

            // Skip to this run
            while (pos < SIZE && table[pos].is_shifted &&
                   !table[pos].is_continuation) {
                pos++;
            }

            if (i == q) {
                // Search in this run
                while (pos < SIZE) {
                    if (table[pos].remainder == r) return true;
                    if (!table[(pos + 1) % SIZE].is_continuation) break;
                    pos++;
                }
                return false;
            }

            // Skip this run
            while (pos < SIZE && table[(pos + 1) % SIZE].is_continuation) {
                pos++;
            }
            pos++;
        }
        return false;
    }
};
```

*Advantages over Bloom:*
- Better cache locality (consecutive memory access)
- Supports resizing (merge two filters)
- Similar space efficiency

== Performance Comparison

#table(
  columns: 5,
  align: (left, center, center, center, center),
  table.header([Structure], [Space], [Insert], [Query], [Use Case]),
  [Bloom Filter], [$O(n)$ bits], [$O(k)$], [$O(k)$], [Membership test],
  [Cuckoo Filter], [$O(n)$ bits], [$O(1)$ amort], [$O(1)$], [Membership + delete],
  [HyperLogLog], [$O(log log n)$], [$O(1)$], [$O(m)$], [Cardinality],
  [Count-Min], [$O(1/epsilon)$], [$O(d)$], [$O(d)$], [Frequency],
  [MinHash], [$O(k)$], [$O(|S| dot k)$], [$O(k)$], [Set similarity],
)

== References

*Primary Sources:*

*Bloom, B.H. (1970)*. Space/Time Trade-offs in Hash Coding with Allowable Errors. Communications of the ACM 13(7): 422-426.

*Flajolet, P. et al. (2007)*. HyperLogLog: The Analysis of a Near-Optimal Cardinality Estimation Algorithm. DMTCS Proceedings.

*Cormode, G. & Muthukrishnan, S. (2005)*. An Improved Data Stream Summary: The Count-Min Sketch and its Applications. Journal of Algorithms 55(1): 58-75.

*Fan, B. et al. (2014)*. Cuckoo Filter: Practically Better Than Bloom. ACM CoNEXT 2014.

*Broder, A.Z. (1997)*. On the Resemblance and Containment of Documents. SEQUENCES 1997.

*Implementation References:*

*Fan, L. et al. (2000)*. Summary Cache: A Scalable Wide-Area Web Cache Sharing Protocol. IEEE/ACM Transactions on Networking 8(3): 281-293.

*Bender, M.A. et al. (2012)*. Don't Thrash: How to Cache Your Hash on Flash. PVLDB 5(11): 1627-1637.

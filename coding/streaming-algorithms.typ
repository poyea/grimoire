= Streaming Algorithms

*Process data in one pass:* Streaming algorithms handle data too large to store, using sublinear space and limited passes. Essential for real-time analytics, network monitoring, and big data processing [Muthukrishnan 2005].

*See also:* Probabilistic Data Structures (for Bloom filters, HyperLogLog), Hashing (for hash-based techniques), Sliding Window (for window-based processing)

== Reservoir Sampling

*Problem:* Select k items uniformly at random from a stream of unknown length.

*Key property:* Each item has equal probability $k/n$ of being in the sample.

```cpp
class ReservoirSampler {
    vector<int> reservoir;
    int k;
    int count;
    mt19937 rng;

public:
    ReservoirSampler(int sampleSize, int seed = 42)
        : k(sampleSize), count(0), rng(seed) {
        reservoir.reserve(k);
    }

    void add(int item) {
        count++;

        if (count <= k) {
            reservoir.push_back(item);
        } else {
            // Replace with probability k/count
            uniform_int_distribution<int> dist(0, count - 1);
            int j = dist(rng);
            if (j < k) {
                reservoir[j] = item;
            }
        }
    }

    const vector<int>& getSample() const {
        return reservoir;
    }
};
```

*Proof of correctness:* By induction, after n items:
- Each of first k items replaced with probability $(1 - k/n)$ at each step
- P(item i in reservoir) = $k/i times product_(j=i+1)^n (1 - 1/j) = k/n$

*Weighted Reservoir Sampling (A-Res Algorithm):*

```cpp
class WeightedReservoirSampler {
    struct Item {
        int value;
        double key;  // Random key for comparison
    };

    priority_queue<Item, vector<Item>, function<bool(Item, Item)>> heap;
    int k;
    mt19937 rng;
    uniform_real_distribution<double> dist;

public:
    WeightedReservoirSampler(int sampleSize, int seed = 42)
        : k(sampleSize), rng(seed), dist(0.0, 1.0) {
        auto cmp = [](const Item& a, const Item& b) {
            return a.key > b.key;  // Min-heap by key
        };
        heap = priority_queue<Item, vector<Item>, decltype(cmp)>(cmp);
    }

    void add(int item, double weight) {
        // Key = random^(1/weight) - higher weight = higher key on average
        double key = pow(dist(rng), 1.0 / weight);

        if (heap.size() < k) {
            heap.push({item, key});
        } else if (key > heap.top().key) {
            heap.pop();
            heap.push({item, key});
        }
    }

    vector<int> getSample() {
        vector<int> result;
        auto heapCopy = heap;
        while (!heapCopy.empty()) {
            result.push_back(heapCopy.top().value);
            heapCopy.pop();
        }
        return result;
    }
};
```

*Complexity:*
- Standard: $O(1)$ per item, $O(k)$ space
- Weighted: $O(log k)$ per item, $O(k)$ space

== Morris Counting (Approximate Counting)

*Problem:* Count to n using $O(log log n)$ bits.

*Idea:* Store $log_2(n)$ approximately. Increment with decreasing probability.

```cpp
class MorrisCounter {
    int value;  // Stores log2(count) approximately
    mt19937 rng;

public:
    MorrisCounter(int seed = 42) : value(0), rng(seed) {}

    void increment() {
        // Increment with probability 2^(-value)
        uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(rng) < pow(2.0, -value)) {
            value++;
        }
    }

    int64_t estimate() const {
        return (1LL << value) - 1;  // 2^value - 1
    }
};
```

*Analysis:*
- E[estimate] = n (unbiased)
- Variance = $n(n-1)/2$
- Standard error = $n / sqrt(2)$

*Morris+ (reduced variance):*

```cpp
class MorrisPlusCounter {
    vector<int> counters;
    int numCounters;
    mt19937 rng;

public:
    MorrisPlusCounter(int k = 64, int seed = 42)
        : numCounters(k), counters(k, 0), rng(seed) {}

    void increment() {
        uniform_real_distribution<double> dist(0.0, 1.0);
        for (int& c : counters) {
            if (dist(rng) < pow(2.0, -c)) {
                c++;
            }
        }
    }

    double estimate() const {
        double sum = 0;
        for (int c : counters) {
            sum += (1LL << c) - 1;
        }
        return sum / numCounters;
    }
};
```

*Variance reduction:* Standard error = $n / sqrt(2k)$

== Misra-Gries (Frequent Items)

*Problem:* Find items appearing more than $n/k$ times (heavy hitters).

*Space:* $O(k)$ counters

```cpp
class MisraGries {
    unordered_map<int, int> counters;
    int k;

public:
    MisraGries(int numCounters) : k(numCounters) {}

    void add(int item) {
        if (counters.count(item)) {
            counters[item]++;
        } else if (counters.size() < k - 1) {
            counters[item] = 1;
        } else {
            // Decrement all counters, remove zeros
            vector<int> toRemove;
            for (auto& [key, count] : counters) {
                if (--count == 0) {
                    toRemove.push_back(key);
                }
            }
            for (int key : toRemove) {
                counters.erase(key);
            }
        }
    }

    // Returns candidates (may include false positives)
    vector<int> getFrequent() const {
        vector<int> result;
        for (auto& [key, count] : counters) {
            result.push_back(key);
        }
        return result;
    }

    int getCount(int item) const {
        auto it = counters.find(item);
        return (it != counters.end()) ? it->second : 0;
    }
};
```

*Guarantee:*
- No false negatives: items with frequency > $n/k$ will be reported
- Estimated count is at most $n/k$ less than true count

*Complexity:* $O(k)$ space, $O(k)$ worst-case per item (amortized $O(1)$)

== Space-Saving Algorithm

*Improvement over Misra-Gries:* Better error bounds, simpler decrements.

```cpp
class SpaceSaving {
    struct Entry {
        int item;
        int count;
        int error;  // Maximum overcount
    };

    list<Entry> entries;
    unordered_map<int, list<Entry>::iterator> itemMap;
    int k;

public:
    SpaceSaving(int numCounters) : k(numCounters) {}

    void add(int item) {
        if (itemMap.count(item)) {
            // Increment existing
            auto it = itemMap[item];
            it->count++;
            // Maintain sorted order (optional optimization)
        } else if (entries.size() < k) {
            // Add new entry
            entries.push_back({item, 1, 0});
            itemMap[item] = prev(entries.end());
        } else {
            // Replace minimum
            auto minIt = min_element(entries.begin(), entries.end(),
                [](const Entry& a, const Entry& b) {
                    return a.count < b.count;
                });

            itemMap.erase(minIt->item);
            int minCount = minIt->count;

            minIt->item = item;
            minIt->count = minCount + 1;
            minIt->error = minCount;

            itemMap[item] = minIt;
        }
    }

    vector<pair<int, int>> getTopK(int topK) {
        vector<Entry> sorted(entries.begin(), entries.end());
        sort(sorted.begin(), sorted.end(),
             [](const Entry& a, const Entry& b) {
                 return a.count > b.count;
             });

        vector<pair<int, int>> result;
        for (int i = 0; i < min(topK, (int)sorted.size()); i++) {
            result.push_back({sorted[i].item, sorted[i].count});
        }
        return result;
    }
};
```

*Guarantee:*
- True count $in$ [estimated - error, estimated]
- For item with frequency f > $n/k$: guaranteed in top-k

== Exponentially Decaying Window

*Problem:* Track statistics where recent items matter more.

```cpp
class ExponentialDecay {
    double sum;
    double count;
    double decay;      // Decay factor per item
    int64_t timestamp;

public:
    ExponentialDecay(double halfLife)
        : sum(0), count(0), timestamp(0) {
        decay = pow(0.5, 1.0 / halfLife);
    }

    void add(double value) {
        timestamp++;
        sum = sum * decay + value;
        count = count * decay + 1;
    }

    double getWeightedAverage() const {
        return (count > 0) ? sum / count : 0;
    }

    double getWeightedSum() const {
        return sum;
    }
};
```

*Property:* Item added t steps ago has weight $"decay"^t$.

== Sliding Window Algorithms

=== Sliding Window Maximum

*Problem:* Find maximum in sliding window of size k.

```cpp
class SlidingWindowMax {
    deque<pair<int, int>> dq;  // (value, index)
    int k;

public:
    SlidingWindowMax(int windowSize) : k(windowSize) {}

    void add(int value, int index) {
        // Remove elements outside window
        while (!dq.empty() && dq.front().second <= index - k) {
            dq.pop_front();
        }

        // Remove smaller elements (they can never be max)
        while (!dq.empty() && dq.back().first <= value) {
            dq.pop_back();
        }

        dq.push_back({value, index});
    }

    int getMax() const {
        return dq.front().first;
    }
};

vector<int> maxSlidingWindow(const vector<int>& nums, int k) {
    SlidingWindowMax swm(k);
    vector<int> result;

    for (int i = 0; i < nums.size(); i++) {
        swm.add(nums[i], i);
        if (i >= k - 1) {
            result.push_back(swm.getMax());
        }
    }

    return result;
}
```

*Complexity:* $O(n)$ total, amortized $O(1)$ per element

=== DABA (Decayed Bloom filter for Approximate membership)

*Problem:* Approximate set membership in sliding window.

```cpp
class SlidingBloomFilter {
    vector<int> counters;  // Counting Bloom filter
    queue<vector<size_t>> history;  // Hash positions for each item
    size_t windowSize;
    size_t numHashes;

    vector<size_t> getHashes(int item) {
        vector<size_t> hashes(numHashes);
        uint64_t h = item;
        for (size_t i = 0; i < numHashes; i++) {
            h = h * 31 + i;
            hashes[i] = h % counters.size();
        }
        return hashes;
    }

public:
    SlidingBloomFilter(size_t window, size_t filterSize, size_t hashes)
        : windowSize(window), numHashes(hashes), counters(filterSize, 0) {}

    void add(int item) {
        auto hashes = getHashes(item);
        history.push(hashes);

        for (size_t pos : hashes) {
            counters[pos]++;
        }

        // Remove old item if window full
        if (history.size() > windowSize) {
            auto oldHashes = history.front();
            history.pop();
            for (size_t pos : oldHashes) {
                counters[pos]--;
            }
        }
    }

    bool mayContain(int item) {
        auto hashes = getHashes(item);
        for (size_t pos : hashes) {
            if (counters[pos] == 0) return false;
        }
        return true;
    }
};
```

== Distinct Elements in Window

*Problem:* Approximate count of distinct elements in sliding window.

```cpp
class SlidingHyperLogLog {
    struct TimestampedRegister {
        uint8_t value;
        int64_t timestamp;
    };

    vector<TimestampedRegister> registers;
    int64_t windowSize;
    int64_t currentTime;
    int p;  // Precision

    size_t hash(int item) {
        // Simple hash for demonstration
        return item * 0x9e3779b97f4a7c15ULL;
    }

public:
    SlidingHyperLogLog(int64_t window, int precision = 14)
        : windowSize(window), currentTime(0), p(precision) {
        registers.resize(1 << p, {0, -windowSize - 1});
    }

    void add(int item) {
        currentTime++;
        size_t h = hash(item);
        size_t idx = h >> (64 - p);
        int rho = __builtin_clzll(h << p | (1ULL << (p - 1))) + 1;

        if (rho >= registers[idx].value ||
            currentTime - registers[idx].timestamp > windowSize) {
            registers[idx] = {static_cast<uint8_t>(rho), currentTime};
        }
    }

    double estimate() {
        int m = 1 << p;
        double sum = 0;
        int validRegisters = 0;

        for (int i = 0; i < m; i++) {
            if (currentTime - registers[i].timestamp <= windowSize) {
                sum += pow(2.0, -registers[i].value);
                validRegisters++;
            } else {
                sum += 1.0;  // Treat expired as empty
            }
        }

        double alpha = 0.7213 / (1 + 1.079 / m);
        return alpha * m * m / sum;
    }
};
```

== Stream Statistics

=== Running Mean and Variance (Welford's Algorithm)

```cpp
class RunningStats {
    int64_t n;
    double mean;
    double M2;  // Sum of squared differences

public:
    RunningStats() : n(0), mean(0), M2(0) {}

    void add(double x) {
        n++;
        double delta = x - mean;
        mean += delta / n;
        double delta2 = x - mean;
        M2 += delta * delta2;
    }

    double getMean() const { return mean; }

    double getVariance() const {
        return (n > 1) ? M2 / (n - 1) : 0;
    }

    double getStdDev() const {
        return sqrt(getVariance());
    }

    int64_t getCount() const { return n; }
};
```

*Property:* Numerically stable, avoids catastrophic cancellation.

=== Quantile Estimation (t-Digest)

*Problem:* Estimate quantiles (median, percentiles) from stream.

```cpp
class TDigest {
    struct Centroid {
        double mean;
        double weight;
    };

    vector<Centroid> centroids;
    double compression;
    double totalWeight;

    double maxSize(double q) {
        // Scale function: smaller centroids at extremes
        return 4 * totalWeight * compression * q * (1 - q);
    }

public:
    TDigest(double comp = 100) : compression(comp), totalWeight(0) {}

    void add(double value, double weight = 1.0) {
        centroids.push_back({value, weight});
        totalWeight += weight;

        // Merge if too many centroids
        if (centroids.size() > 3 * compression) {
            compress();
        }
    }

    void compress() {
        sort(centroids.begin(), centroids.end(),
             [](const Centroid& a, const Centroid& b) {
                 return a.mean < b.mean;
             });

        vector<Centroid> merged;
        double cumWeight = 0;

        for (auto& c : centroids) {
            if (merged.empty()) {
                merged.push_back(c);
            } else {
                double q = (cumWeight + c.weight / 2) / totalWeight;
                double limit = maxSize(q);

                if (merged.back().weight + c.weight <= limit) {
                    // Merge with last centroid
                    double newWeight = merged.back().weight + c.weight;
                    merged.back().mean = (merged.back().mean * merged.back().weight +
                                          c.mean * c.weight) / newWeight;
                    merged.back().weight = newWeight;
                } else {
                    merged.push_back(c);
                }
            }
            cumWeight += c.weight;
        }

        centroids = merged;
    }

    double quantile(double q) {
        compress();
        if (centroids.empty()) return 0;

        double cumWeight = 0;
        double target = q * totalWeight;

        for (int i = 0; i < centroids.size(); i++) {
            double nextCumWeight = cumWeight + centroids[i].weight;

            if (nextCumWeight >= target) {
                // Interpolate within centroid
                double fraction = (target - cumWeight) / centroids[i].weight;

                if (i == 0) {
                    return centroids[i].mean;
                } else {
                    return centroids[i - 1].mean +
                           (centroids[i].mean - centroids[i - 1].mean) * fraction;
                }
            }
            cumWeight = nextCumWeight;
        }

        return centroids.back().mean;
    }

    double median() { return quantile(0.5); }
    double percentile(double p) { return quantile(p / 100.0); }
};
```

*Complexity:* $O(delta)$ space where $delta$ = compression parameter

== Sketches for Matrix Operations

=== Frequent Directions (Low-Rank Approximation)

*Problem:* Maintain low-rank approximation of streaming matrix.

```cpp
class FrequentDirections {
    int d;      // Dimension
    int ell;    // Sketch size
    vector<vector<double>> B;  // Sketch matrix

    void reduceRank() {
        // Compute SVD and shrink singular values
        // Simplified: just zero out smallest row
        double minNorm = DBL_MAX;
        int minIdx = 0;

        for (int i = 0; i < ell; i++) {
            double norm = 0;
            for (int j = 0; j < d; j++) {
                norm += B[i][j] * B[i][j];
            }
            if (norm < minNorm) {
                minNorm = norm;
                minIdx = i;
            }
        }

        fill(B[minIdx].begin(), B[minIdx].end(), 0.0);
    }

public:
    FrequentDirections(int dim, int sketchSize)
        : d(dim), ell(sketchSize), B(sketchSize, vector<double>(dim, 0)) {}

    void addRow(const vector<double>& row) {
        // Find zero row
        int zeroRow = -1;
        for (int i = 0; i < ell; i++) {
            bool isZero = true;
            for (int j = 0; j < d; j++) {
                if (B[i][j] != 0) {
                    isZero = false;
                    break;
                }
            }
            if (isZero) {
                zeroRow = i;
                break;
            }
        }

        if (zeroRow == -1) {
            reduceRank();
            zeroRow = 0;
            for (int i = 0; i < ell; i++) {
                double norm = 0;
                for (int j = 0; j < d; j++) norm += B[i][j] * B[i][j];
                if (norm == 0) { zeroRow = i; break; }
            }
        }

        for (int j = 0; j < d; j++) {
            B[zeroRow][j] = row[j];
        }
    }

    const vector<vector<double>>& getSketch() const { return B; }
};
```

*Guarantee:* For any unit vector x: $||A x||^2 - ||B x||^2 <= ||A||_F^2 / ell$

== Performance Comparison

#table(
  columns: 4,
  align: (left, center, center, left),
  table.header([Algorithm], [Space], [Update], [Problem]),
  [Reservoir Sampling], [$O(k)$], [$O(1)$], [Uniform sampling],
  [Morris Counting], [$O(log log n)$], [$O(1)$], [Approximate count],
  [Misra-Gries], [$O(k)$], [$O(k)$], [Heavy hitters],
  [Space-Saving], [$O(k)$], [$O(log k)$], [Top-k frequent],
  [Sliding Window Max], [$O(k)$], [$O(1)$ amort], [Window maximum],
  [t-Digest], [$O(delta)$], [$O(delta)$], [Quantiles],
)

== References

*Primary Sources:*

*Muthukrishnan, S. (2005)*. Data Streams: Algorithms and Applications. Foundations and Trends in Theoretical Computer Science 1(2): 117-236.

*Vitter, J.S. (1985)*. Random Sampling with a Reservoir. ACM Transactions on Mathematical Software 11(1): 37-57.

*Algorithms & Theory:*

*Morris, R. (1978)*. Counting Large Numbers of Events in Small Registers. Communications of the ACM 21(10): 840-842.

*Misra, J. & Gries, D. (1982)*. Finding Repeated Elements. Science of Computer Programming 2(2): 143-152.

*Metwally, A., Agrawal, D., & El Abbadi, A. (2005)*. Efficient Computation of Frequent and Top-k Elements in Data Streams. ICDT 2005.

*Dunning, T. & Ertl, O. (2019)*. Computing Extremely Accurate Quantiles Using t-Digests. arXiv:1902.04023.

*Ghashami, M. et al. (2016)*. Frequent Directions: Simple and Deterministic Matrix Sketching. SIAM Journal on Computing 45(5): 1762-1792.

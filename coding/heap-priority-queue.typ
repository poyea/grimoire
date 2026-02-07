= Heap/Priority Queue

*A binary heap is a complete binary tree stored as an array, providing $O(log n)$ insert and extract-min/max with excellent cache locality. Foundation for priority queues, heap-sort, and greedy algorithms.*

*See also:* Trees (heap is a complete binary tree), Graphs (Dijkstra's, Prim's use priority queues), Greedy (scheduling, Huffman coding)

== Heap Fundamentals

*Heap property:*
- *Min-heap:* every parent $<=$ both children. Root = minimum element.
- *Max-heap:* every parent $>=$ both children. Root = maximum element.
- *Complete binary tree:* all levels filled except possibly last, which fills left-to-right.

*Array representation:* Parent at index `i`, left child at `2i+1`, right child at `2i+2`. Parent of node at `j` is at `(j-1)/2`. No pointer overhead -- sequential memory = cache-friendly.

```
Heap array: [root, L1, R1, L1-L, L1-R, R1-L, R1-R, ...]
Indices:     [0,    1,  2,  3,    4,    5,    6,    ...]
```

*Sift-up (bubble up):* After inserting at end, compare with parent and swap upward until heap property restored. $O(log n)$ worst case, $O(1)$ average (most insertions terminate near bottom).

*Sift-down (heapify down):* After removing root (swap root with last, remove last), compare with children and swap downward. $O(log n)$ always.

*Build heap (heapify):* Bottom-up sift-down from last non-leaf to root. $O(n)$ time -- not $O(n log n)$ -- because most nodes are near the bottom and sift very short distances.

```cpp
// Min-heap implementation core operations
void sift_up(vector<int>& heap, int i) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (heap[i] < heap[parent]) {
            swap(heap[i], heap[parent]);
            i = parent;
        } else break;
    }
}

void sift_down(vector<int>& heap, int i, int size) {
    while (2 * i + 1 < size) {
        int left = 2 * i + 1, right = 2 * i + 2;
        int smallest = i;
        if (left < size && heap[left] < heap[smallest]) smallest = left;
        if (right < size && heap[right] < heap[smallest]) smallest = right;
        if (smallest == i) break;
        swap(heap[i], heap[smallest]);
        i = smallest;
    }
}

// Build heap in O(n) -- start from last non-leaf
void build_heap(vector<int>& arr) {
    for (int i = arr.size() / 2 - 1; i >= 0; i--) {
        sift_down(arr, i, arr.size());
    }
}
```

*Why build-heap is $O(n)$:* At depth $d$ from bottom, there are $approx n / 2^(d+1)$ nodes each sifting at most $d$ levels. Summation: $sum_(d=0)^(log n) n / 2^(d+1) dot d = O(n)$.

*C++ STL priority_queue:*
```cpp
// Max-heap (default)
priority_queue<int> max_heap;

// Min-heap
priority_queue<int, vector<int>, greater<int>> min_heap;

// Custom comparator -- min-heap by second element of pair
auto cmp = [](const pair<int,int>& a, const pair<int,int>& b) {
    return a.second > b.second;  // greater = min-heap
};
priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> pq(cmp);
```

*Cache behavior:*
- Small heaps (< 8K elements $approx$ 32KB): fits in L1 data cache $approx$ 4-5 cycles per access
- Medium heaps (< 64K elements $approx$ 256KB): fits in L2 cache $approx$ 12-15 cycles per access
- Large heaps: only root path ($log n$ nodes) accessed per operation = $log n$ cache misses worst case

== Heap-Sort

*Algorithm:* Build a max-heap from the array, then repeatedly extract the maximum (swap root with last unsorted element) and sift-down to restore heap property. In-place, $O(n log n)$ worst case guaranteed.

```cpp
void sift_down_max(vector<int>& arr, int i, int size) {
    while (2 * i + 1 < size) {
        int left = 2 * i + 1, right = 2 * i + 2;
        int largest = i;
        if (left < size && arr[left] > arr[largest]) largest = left;
        if (right < size && arr[right] > arr[largest]) largest = right;
        if (largest == i) break;
        swap(arr[i], arr[largest]);
        i = largest;
    }
}

void heap_sort(vector<int>& arr) {
    int n = arr.size();

    // Phase 1: Build max-heap in O(n)
    for (int i = n / 2 - 1; i >= 0; i--) {
        sift_down_max(arr, i, n);
    }

    // Phase 2: Extract max repeatedly in O(n log n)
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);        // Move current max to sorted position
        sift_down_max(arr, 0, i);      // Restore heap on reduced range
    }
}
```

*Heap-sort vs quicksort vs mergesort:*
- Heap-sort: $O(n log n)$ worst case, in-place, *not* stable. Poor cache locality (long-range swaps).
- Quicksort: $O(n log n)$ average, $O(n^2)$ worst, in-place, not stable. Best cache behavior.
- Mergesort: $O(n log n)$ always, $O(n)$ extra space, stable. Good for linked lists and external sort.

*When to use heap-sort:* Need guaranteed $O(n log n)$ with $O(1)$ extra space. Rare in practice -- quicksort faster due to cache effects. Used in `std::sort` implementations as introsort fallback (switches from quicksort to heap-sort when recursion depth exceeds $2 log n$).

== Top-K Problems

*Pattern:* Use a heap of size $K$ to track the $K$ best elements seen so far. Avoids sorting the entire input.

*Kth Largest Element:*

```cpp
// Approach 1: Min-heap of size K -- O(n log k) time, O(k) space
int find_kth_largest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> min_heap;
    for (int num : nums) {
        min_heap.push(num);
        if ((int)min_heap.size() > k) {
            min_heap.pop();  // Evict smallest -- heap always holds K largest
        }
    }
    return min_heap.top();  // Smallest of the K largest = Kth largest
}
```

*Why min-heap for Kth largest?* The root is the smallest among the $K$ largest elements seen. Any new element larger than root replaces it, maintaining the top-$K$ invariant. After processing all elements, root = Kth largest.

*Top K Frequent Elements:*

```cpp
vector<int> top_k_frequent(vector<int>& nums, int k) {
    // Step 1: Count frequencies -- O(n)
    unordered_map<int, int> freq;
    for (int n : nums) freq[n]++;

    // Step 2: Min-heap of size k by frequency -- O(m log k) where m = distinct elements
    auto cmp = [](const pair<int,int>& a, const pair<int,int>& b) {
        return a.second > b.second;  // min-heap by frequency
    };
    priority_queue<pair<int,int>, vector<pair<int,int>>, decltype(cmp)> min_heap(cmp);

    for (auto& [val, cnt] : freq) {
        min_heap.push({val, cnt});
        if ((int)min_heap.size() > k) min_heap.pop();
    }

    // Step 3: Extract results
    vector<int> result;
    while (!min_heap.empty()) {
        result.push_back(min_heap.top().first);
        min_heap.pop();
    }
    return result;
}
```

*Alternative -- bucket sort:* If max frequency $<=$ $n$, use array of buckets indexed by frequency. $O(n)$ time but $O(n)$ space. Preferred when $K$ is large relative to distinct element count.

== Priority Queue Applications

*Dijkstra's Shortest Path:* Min-heap keyed by distance. Greedily extract nearest unvisited node and relax neighbors. $O((V + E) log V)$ with binary heap.

```cpp
vector<int> dijkstra(vector<vector<pair<int,int>>>& adj, int src, int n) {
    vector<int> dist(n, INT_MAX);
    dist[src] = 0;

    // {distance, node}
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<pair<int,int>>> pq;
    pq.push({0, src});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > dist[u]) continue;  // Stale entry -- skip

        for (auto [v, w] : adj[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}
```

*Stale entry handling:* C++ `priority_queue` has no decrease-key. Instead, insert duplicates and skip stale entries where `d > dist[u]`. Adds at most $O(E)$ entries -- still $O((V + E) log V)$ overall.

*Prim's Minimum Spanning Tree:* Same pattern as Dijkstra but keyed by edge weight to the growing tree rather than cumulative distance. $O((V + E) log V)$ with binary heap.

*Huffman Coding:* Build optimal prefix code tree by repeatedly extracting two minimum-frequency nodes and merging. $O(n log n)$ where $n$ = alphabet size. Priority queue makes the greedy selection efficient.

== Merge K Sorted Lists

*Problem:* Merge $K$ sorted linked lists into one sorted list.

*Approach -- Min-heap of size K:* $O(N log K)$ time where $N$ = total elements, $O(K)$ space.

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x = 0, ListNode* n = nullptr) : val(x), next(n) {}
};

ListNode* merge_k_lists(vector<ListNode*>& lists) {
    auto cmp = [](ListNode* a, ListNode* b) { return a->val > b->val; };
    priority_queue<ListNode*, vector<ListNode*>, decltype(cmp)> pq(cmp);

    // Initialize heap with head of each non-empty list
    for (ListNode* head : lists) {
        if (head) pq.push(head);
    }

    ListNode dummy;
    ListNode* tail = &dummy;
    while (!pq.empty()) {
        ListNode* node = pq.top();
        pq.pop();
        tail->next = node;
        tail = tail->next;
        if (node->next) {
            pq.push(node->next);  // Push next from same list
        }
    }
    return dummy.next;
}
```

*Why $O(N log K)$:* Each of the $N$ total elements is inserted and extracted from the heap exactly once. Each heap operation costs $O(log K)$ since the heap never exceeds size $K$.

*Alternatives:*
- Divide and conquer (merge pairs repeatedly): same $O(N log K)$ time, $O(log K)$ stack space. Slightly better cache behavior.
- Merge all into one list then sort: $O(N log N)$ -- worse when $K << N$.

== Median Maintenance (Two-Heap Pattern)

*Problem:* Design data structure to support adding numbers and finding median efficiently.

*Approach -- Two Heaps:* $O(log n)$ add_num, $O(1)$ find_median

- *max_heap:* stores the smaller half (max element on top)
- *min_heap:* stores the larger half (min element on top)
- Invariant: `max_heap.size() == min_heap.size()` or `max_heap.size() == min_heap.size() + 1`

```cpp
class MedianFinder {
    priority_queue<int> max_heap;  // smaller half
    priority_queue<int, vector<int>, greater<int>> min_heap;  // larger half

public:
    void add_num(int num) {
        max_heap.push(num);

        // Ensure max(smaller half) <= min(larger half)
        if (!min_heap.empty() && max_heap.top() > min_heap.top()) {
            min_heap.push(max_heap.top());
            max_heap.pop();
        }

        // Balance sizes: max_heap can have at most 1 more element
        if (max_heap.size() > min_heap.size() + 1) {
            min_heap.push(max_heap.top());
            max_heap.pop();
        } else if (min_heap.size() > max_heap.size()) {
            max_heap.push(min_heap.top());
            min_heap.pop();
        }
    }

    double find_median() {
        if (max_heap.size() > min_heap.size()) {
            return max_heap.top();
        }
        return (max_heap.top() + min_heap.top()) / 2.0;
    }
};
```

*Key insight:* max_heap top = largest of smaller half. min_heap top = smallest of larger half. Median is either max_heap top (odd count) or average of both tops (even count).

*Sliding window median:* Same two-heap idea but need lazy deletion. Track elements to remove in a hash map, only actually remove when they appear at heap top. $O(n log n)$ for $n$ operations.

*Alternative -- Bucket approach:* For bounded range $[0, M]$, use counting array. $O(1)$ insert, $O(M)$ median. Faster if $M < log n$.

== Heap Variations

*Binary heap* (standard): Simple, cache-friendly array. Insert and extract in $O(log n)$. Used in `std::priority_queue`.

*D-ary heap:* Each node has $d$ children instead of 2. Shallower tree = faster insert ($O(log_d n)$) but slower extract (must compare $d$ children). Optimal $d$ depends on workload: $d = 4$ often best in practice due to cache line alignment.

*Binomial heap:* Collection of binomial trees. Supports merge in $O(log n)$. Insert amortized $O(1)$. Useful when frequent merging of heaps is needed.

*Fibonacci heap:* Decrease-key in amortized $O(1)$ instead of $O(log n)$. Makes Dijkstra $O(V log V + E)$ instead of $O((V + E) log V)$. Theoretically optimal but poor constant factors and complex implementation. Rarely used in practice outside theoretical analysis.

*Indexed priority queue:* Supports decrease-key by maintaining a position map from element ID to heap index. Practical alternative to Fibonacci heap for Dijkstra/Prim. $O(log n)$ decrease-key.

```cpp
// Indexed min-priority queue (simplified)
class IndexedPQ {
    vector<int> heap;    // heap[i] = element ID at position i
    vector<int> pos;     // pos[id] = position in heap (-1 if absent)
    vector<int> keys;    // keys[id] = priority value

    void swim(int i) {
        while (i > 0 && keys[heap[i]] < keys[heap[(i-1)/2]]) {
            swap(heap[i], heap[(i-1)/2]);
            pos[heap[i]] = i;
            pos[heap[(i-1)/2]] = (i-1)/2;
            i = (i-1)/2;
        }
    }

    void sink(int i, int sz) {
        while (2*i+1 < sz) {
            int j = 2*i+1;
            if (j+1 < sz && keys[heap[j+1]] < keys[heap[j]]) j++;
            if (keys[heap[i]] <= keys[heap[j]]) break;
            swap(heap[i], heap[j]);
            pos[heap[i]] = i;
            pos[heap[j]] = j;
            i = j;
        }
    }

public:
    IndexedPQ(int max_n) : pos(max_n, -1), keys(max_n) {}

    void insert(int id, int key) {
        keys[id] = key;
        pos[id] = heap.size();
        heap.push_back(id);
        swim(heap.size() - 1);
    }

    void decrease_key(int id, int new_key) {
        keys[id] = new_key;
        swim(pos[id]);
    }

    int extract_min() {
        int id = heap[0];
        swap(heap[0], heap.back());
        pos[heap[0]] = 0;
        pos[id] = -1;
        heap.pop_back();
        if (!heap.empty()) sink(0, heap.size());
        return id;
    }

    bool contains(int id) { return pos[id] != -1; }
    bool empty() { return heap.empty(); }
};
```

== Complexity Reference

#table(
  columns: (auto, auto, auto),
  inset: 5pt,
  align: left,
  table.header(
    [*Operation*], [*Binary Heap*], [*Notes*],
  ),
  [Insert], [$O(log n)$, avg $O(1)$], [Sift-up; most elements near bottom],
  [Extract-min/max], [$O(log n)$], [Swap root with last, sift-down],
  [Peek (min/max)], [$O(1)$], [Root element],
  [Build heap], [$O(n)$], [Bottom-up heapify],
  [Heap-sort], [$O(n log n)$], [Build + $n$ extractions, in-place],
  [Merge two heaps], [$O(n)$], [Rebuild; binomial/Fibonacci: $O(log n)$],
  [Decrease-key], [$O(log n)$], [Fibonacci heap: amortized $O(1)$],
  [Delete arbitrary], [$O(log n)$], [Decrease to $-infinity$, then extract],
  [Search], [$O(n)$], [Heap is not ordered for search],
)

*Space:* $O(n)$ for all heap types. Array-based binary heap has zero pointer overhead.

*C++ `std::priority_queue` specifics:*
- `push()`: $O(log n)$ -- sift-up
- `pop()`: $O(log n)$ -- sift-down
- `top()`: $O(1)$
- No `remove(value)` or `contains()` -- use lazy deletion or indexed PQ
- No decrease-key -- insert duplicate and skip stale entries
- Default is max-heap; use `greater<T>` comparator for min-heap
- Backed by `vector<T>` (contiguous memory = cache-friendly)

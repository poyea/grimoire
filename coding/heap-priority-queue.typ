= Heap/Priority Queue

*Array-based heap performance:* Binary heap stored in array = excellent cache locality. Parent at `i`, children at `2i+1`, `2i+2`. Sequential memory access during sift operations.

== Find Median from Data Stream

*Problem:* Design data structure to support adding numbers and finding median efficiently.

*Approach - Two Heaps:* $O(log n)$ addNum, $O(1)$ findMedian

```cpp
class MedianFinder {
    priority_queue<int> maxHeap;  // Smaller half (max on top)
    priority_queue<int, vector<int>, greater<int>> minHeap;  // Larger half (min on top)

public:
    void addNum(int num) {
        // Add to max heap first
        maxHeap.push(num);

        // Balance values: ensure max(small) <= min(large)
        if (!minHeap.empty() && maxHeap.top() > minHeap.top()) {
            minHeap.push(maxHeap.top());
            maxHeap.pop();
        }

        // Balance sizes: |size_diff| <= 1
        if (maxHeap.size() > minHeap.size() + 1) {
            minHeap.push(maxHeap.top());
            maxHeap.pop();
        } else if (minHeap.size() > maxHeap.size()) {
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }
    }

    double findMedian() {
        if (maxHeap.size() > minHeap.size()) {
            return maxHeap.top();
        }
        return (maxHeap.top() + minHeap.top()) / 2.0;
    }
};
```

*Heap implementation details:*
- `priority_queue` uses `vector` internally = contiguous array = cache-friendly
- `push()`: append + sift-up. Average O(1), worst O(log n)
- `pop()`: swap with last + sift-down. O(log n) always

*Memory layout:*
```
Heap array: [root, L1, R1, L1-L, L1-R, R1-L, R1-R, ...]
Indices:     [0,    1,  2,  3,    4,    5,    6,    ...]
```
Parent-child jumps = predictable stride pattern. CPU prefetcher can partially predict.

*Cache behavior:*
- Small heaps (< 8K elements = $#sym.tilde.op$32KB): entire heap fits in L1 data cache = $#sym.tilde.op$4-5 cycles per access
- Medium heaps (< 64K elements = $#sym.tilde.op$256KB): fits in L2 cache = $#sym.tilde.op$12-15 cycles per access
- Large heaps: only root path (log n nodes) accessed during operations = log n cache misses worst case

*Alternative - Bucket approach:* For bounded range [0, M], use counting array. O(1) insert, O(M) median. Faster if M < log n.

*SIMD optimization:* Heapify operation on small heaps (n < 32) can use SIMD compare-and-swap for parallel sift-down.

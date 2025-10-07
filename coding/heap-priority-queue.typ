= Heap/Priority Queue

== Find Median from Data Stream

*Problem:* Design data structure to support adding numbers and finding median efficiently.

*Approach - Two Heaps:* $O(log n)$ add, $O(1)$ findMedian
- Use two heaps to maintain median:
  + `small` - max heap (negate values) for smaller half
  + `large` - min heap for larger half
- Keep heaps balanced: `|len(small) - len(large)| <= 1`

*addNum(num):*
- Add to small heap: `heappush(self.small, -1 * num)`
- Balance values: if `max(small) > min(large)`:
  + Move max from small to large
- Rebalance sizes: if one heap has 2+ more elements than other, move element

*findMedian():*
- If `len(small) > len(large)`: return `-small[0]`
- If `len(large) > len(small)`: return `large[0]`
- Else: return `(-small[0] + large[0]) / 2.0`

*Key insight:* Two heaps maintain sorted middle elements for $O(1)$ median access.

*Key Python concepts:*
- `heapq` module for min heap
- Negate values for max heap: `heappush(heap, -val)`

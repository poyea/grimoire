= Linked List

*Linked lists store elements with explicit pointers between nodes. Enables $O(1)$ insertion/deletion at known positions, but sacrifices random access and cache locality.*

*See also:* Arrays (for cache-friendly alternatives), Trees (similar pointer-based structure), Two Pointers (for linked list traversal patterns), Hashing (for fast cycle detection alternatives)

*Critical performance note:* Linked lists have poor cache behavior. Each `next` pointer dereference = potential cache miss ($#sym.tilde.op$200 cycles). Array-based solutions are typically 10-100x faster. Use linked lists only when required (LRU cache, memory pooling).

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x = 0, ListNode* n = nullptr) : val(x), next(n) {}
};
```

== Reverse Linked List

*Problem:* Reverse a singly linked list.

*Approach 1 - Iterative:* $O(n)$ time, $O(1)$ space

```cpp
ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;

    while (head) {
        ListNode* next = head->next;  // Prefetch hint opportunity
        head->next = prev;
        prev = head;
        head = next;
    }
    return prev;
}
```

*Pointer chasing cost:* Each iteration has dependent load: `head->next`. CPU stalls $#sym.tilde.op$40-75 cycles on L3 miss, up to $#sym.tilde.op$200 cycles on main memory access [Drepper 2007, §3.3; Intel Memory Latency Checker measurements]. Cannot parallelize - loads are serialized. This is because each load depends on the result of the previous load: we must load `head->next` before we can load `head->next->next`. Modern out-of-order execution CPUs can overlap independent loads, but pointer chasing creates a dependency chain where each load becomes a bottleneck. The CPU's memory-level parallelism#footnote[Memory-level parallelism (MLP) is the ability to have multiple outstanding memory requests in flight simultaneously. Modern CPUs can track 10-12 concurrent cache misses, but pointer chasing limits this to 1-2 because each miss depends on the previous one completing.] is effectively reduced to 1, eliminating one of the primary techniques CPUs use to hide memory latency.

*Prefetch optimization:*
```cpp
ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;

    while (head) {
        if (head->next) {
            __builtin_prefetch(head->next->next);  // Prefetch 2 ahead
        }
        ListNode* next = head->next;
        head->next = prev;
        prev = head;
        head = next;
    }
    return prev;
}
```

*Approach 2 - Recursive:* $O(n)$ time, $O(n)$ space (stack frames)

```cpp
ListNode* reverseList(ListNode* head) {
    if (!head || !head->next) return head;

    ListNode* newHead = reverseList(head->next);
    head->next->next = head;
    head->next = nullptr;
    return newHead;
}
```

*Stack frame cost:* Each call = $#sym.tilde.op$16-48 bytes (return addr, saved registers, local vars). 10K nodes = 160-480KB stack. Risk stack overflow. Iterative version has no call overhead.

== Merge Two Sorted Lists

*Problem:* Merge two sorted linked lists into one sorted list.

*Approach - Dummy Node:* $O(n + m)$ time, $O(1)$ space

```cpp
ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
    ListNode dummy;
    ListNode* curr = &dummy;

    while (list1 && list2) {
        if (list1->val < list2->val) {
            curr->next = list1;
            list1 = list1->next;
        } else {
            curr->next = list2;
            list2 = list2->next;
        }
        curr = curr->next;
    }
    curr->next = list1 ? list1 : list2;
    return dummy.next;
}
```

*Branch prediction:* If lists have similar value distributions, branches are $#sym.tilde.op$50% predictable = many mispredicts ($#sym.tilde.op$15-20 cycle penalty each on misprediction). Modern CPUs have $#sym.tilde.op$95% prediction accuracy for simple patterns.

== Remove Nth Node From End

*Problem:* Remove nth node from end of linked list.

*Approach - Two Pointers:* $O(n)$ time, $O(1)$ space

```cpp
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode dummy(0, head);
    ListNode* fast = head;
    ListNode* slow = &dummy;

    // Move fast n steps ahead
    for (int i = 0; i < n; i++) {
        fast = fast->next;
    }

    // Move both until fast reaches end
    while (fast) {
        __builtin_prefetch(fast->next);  // Prefetch next iteration
        __builtin_prefetch(slow->next);
        fast = fast->next;
        slow = slow->next;
    }

    ListNode* toDelete = slow->next;
    slow->next = slow->next->next;
    delete toDelete;  // Avoid memory leak

    return dummy.next;
}
```

*Two-pointer pattern:* Maintains n-gap between pointers. Still suffers pointer chasing but processes list in single pass.

== Linked List Cycle Detection

*Problem:* Detect if linked list has a cycle.

*Approach - Floyd's Tortoise & Hare:* $O(n)$ time, $O(1)$ space

```cpp
bool hasCycle(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;

    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;

        if (slow == fast) return true;
    }
    return false;
}
```

*Cache behavior:* If cycle exists, eventually both pointers traverse same nodes repeatedly = high cache hit rate. Initial traversal before cycle entry = all cache misses.

*Memory-level parallelism:* Fast pointer issues 2 dependent loads per iteration. Modern out-of-order CPUs can overlap these loads if memory subsystem supports multiple outstanding misses.

== Merge K Sorted Lists

*Problem:* Merge k sorted linked lists into one sorted list.

*Approach - Divide & Conquer:* $O(n log k)$ time, $n$ = total nodes, $k$ = number of lists

```cpp
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode dummy;
    ListNode* curr = &dummy;

    while (l1 && l2) {
        if (l1->val < l2->val) {
            curr->next = l1;
            l1 = l1->next;
        } else {
            curr->next = l2;
            l2 = l2->next;
        }
        curr = curr->next;
    }
    curr->next = l1 ? l1 : l2;
    return dummy.next;
}

ListNode* mergeKLists(vector<ListNode*>& lists) {
    if (lists.empty()) return nullptr;

    while (lists.size() > 1) {
        vector<ListNode*> merged;
        merged.reserve((lists.size() + 1) / 2);

        for (size_t i = 0; i < lists.size(); i += 2) {
            ListNode* l1 = lists[i];
            ListNode* l2 = (i + 1 < lists.size()) ? lists[i + 1] : nullptr;
            merged.push_back(mergeTwoLists(l1, l2));
        }
        lists = move(merged);
    }
    return lists[0];
}
```

*Alternative - Min Heap:* $O(n log k)$ time but worse constants. Priority queue operations = 3-5x overhead vs direct merge. Use only if streaming input.

*Memory insight:* Divide-and-conquer reuses pointers (no new nodes). Heap creates temporary nodes. Zero allocation = zero malloc/free overhead (typically $#sym.tilde.op$20-100 cycles each for modern allocators like tcmalloc, jemalloc).

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
ListNode* reverse_list(ListNode* head) {
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
ListNode* reverse_list_prefetch(ListNode* head) {
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
ListNode* reverse_list_recursive(ListNode* head) {
    if (!head || !head->next) return head;

    ListNode* new_head = reverse_list_recursive(head->next);
    head->next->next = head;
    head->next = nullptr;
    return new_head;
}
```

*Stack frame cost:* Each call = $#sym.tilde.op$16-48 bytes (return addr, saved registers, local vars). 10K nodes = 160-480KB stack. Risk stack overflow. Iterative version has no call overhead.

== Merge Two Sorted Lists

*Problem:* Merge two sorted linked lists into one sorted list.

*Approach - Dummy Node:* $O(n + m)$ time, $O(1)$ space

```cpp
ListNode* merge_two_lists(ListNode* list1, ListNode* list2) {
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
ListNode* remove_nth_from_end(ListNode* head, int n) {
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

    ListNode* to_delete = slow->next;
    slow->next = slow->next->next;
    delete to_delete;  // Avoid memory leak

    return dummy.next;
}
```

*Two-pointer pattern:* Maintains n-gap between pointers. Still suffers pointer chasing but processes list in single pass.

== Linked List Cycle Detection

*Problem:* Detect if linked list has a cycle.

*Approach - Floyd's Tortoise & Hare:* $O(n)$ time, $O(1)$ space

```cpp
bool has_cycle(ListNode* head) {
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

*Find cycle start:* After detection, reset one pointer to head. Move both one step at a time. They meet at the cycle entry.

```cpp
ListNode* detect_cycle(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;

    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {
            slow = head;
            while (slow != fast) {
                slow = slow->next;
                fast = fast->next;
            }
            return slow;  // Cycle entry point
        }
    }
    return nullptr;
}
```

*Why it works:* If cycle length is $C$ and distance from head to cycle start is $F$, when slow enters cycle, fast is $F mod C$ steps into the cycle. They meet after slow travels $C - (F mod C)$ more steps. Resetting slow to head and stepping both by 1: they meet at cycle start after $F$ steps.

*Cache behavior:* If cycle exists, eventually both pointers traverse same nodes repeatedly = high cache hit rate. Initial traversal before cycle entry = all cache misses.

== Copy List with Random Pointer

*Problem:* Deep copy a linked list where each node has both `next` and a `random` pointer to any node in the list.

*Approach - Interleave then split:* $O(n)$ time, $O(1)$ extra space (excluding output)

```cpp
struct RandomNode {
    int val;
    RandomNode *next, *random;
    RandomNode(int x) : val(x), next(nullptr), random(nullptr) {}
};

RandomNode* copy_random_list(RandomNode* head) {
    if (!head) return nullptr;

    // Pass 1: Interleave copies -- A->A'->B->B'->C->C'
    for (RandomNode* curr = head; curr; curr = curr->next->next) {
        RandomNode* copy = new RandomNode(curr->val);
        copy->next = curr->next;
        curr->next = copy;
    }

    // Pass 2: Set random pointers
    for (RandomNode* curr = head; curr; curr = curr->next->next) {
        if (curr->random) {
            curr->next->random = curr->random->next;
        }
    }

    // Pass 3: Split into original and copy
    RandomNode* new_head = head->next;
    for (RandomNode* curr = head; curr; curr = curr->next) {
        RandomNode* copy = curr->next;
        curr->next = copy->next;
        copy->next = copy->next ? copy->next->next : nullptr;
    }

    return new_head;
}
```

*Alternative - HashMap:* $O(n)$ time, $O(n)$ space. Map `original -> copy`. Simpler but uses extra memory.

== LRU Cache

*Problem:* Design a cache with $O(1)$ get and put, evicting least-recently-used item when capacity exceeded.

*Approach - HashMap + Doubly Linked List:* $O(1)$ for both operations

```cpp
class LRUCache {
    struct Node {
        int key, val;
        Node *prev, *next;
        Node(int k = 0, int v = 0) : key(k), val(v), prev(nullptr), next(nullptr) {}
    };

    int capacity;
    unordered_map<int, Node*> cache;
    Node head, tail;  // Sentinel nodes

    void remove(Node* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }

    void push_front(Node* node) {
        node->next = head.next;
        node->prev = &head;
        head.next->prev = node;
        head.next = node;
    }

public:
    LRUCache(int cap) : capacity(cap) {
        head.next = &tail;
        tail.prev = &head;
    }

    int get(int key) {
        if (!cache.count(key)) return -1;
        Node* node = cache[key];
        remove(node);
        push_front(node);  // Move to front (most recently used)
        return node->val;
    }

    void put(int key, int value) {
        if (cache.count(key)) {
            Node* node = cache[key];
            node->val = value;
            remove(node);
            push_front(node);
            return;
        }

        if ((int)cache.size() == capacity) {
            Node* lru = tail.prev;  // Least recently used = back
            remove(lru);
            cache.erase(lru->key);
            delete lru;
        }

        Node* node = new Node(key, value);
        push_front(node);
        cache[key] = node;
    }

    ~LRUCache() {
        Node* curr = head.next;
        while (curr != &tail) {
            Node* next = curr->next;
            delete curr;
            curr = next;
        }
    }
};
```

*Why doubly linked list?* Need $O(1)$ removal from arbitrary position (on cache hit, move to front). Singly linked list requires $O(n)$ to find predecessor.

*Why not `std::list`?* `std::list` works but `splice()` + iterator invalidation rules add complexity. Hand-rolled doubly linked list with sentinel nodes is simpler for this specific pattern.

*Memory layout:* Each `Node` = 32 bytes (key + val + 2 pointers). HashMap entry = $#sym.tilde.op$48 bytes (key + pointer + bucket overhead). Total per entry $approx$ 80 bytes. For 1000 entries: $#sym.tilde.op$80KB (fits in L2 cache).

== Merge K Sorted Lists

*Problem:* Merge k sorted linked lists into one sorted list.

*Approach - Divide & Conquer:* $O(n log k)$ time, $n$ = total nodes, $k$ = number of lists

```cpp
ListNode* merge_two_lists(ListNode* l1, ListNode* l2) {
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

ListNode* merge_k_lists(vector<ListNode*>& lists) {
    if (lists.empty()) return nullptr;

    while (lists.size() > 1) {
        vector<ListNode*> merged;
        merged.reserve((lists.size() + 1) / 2);

        for (size_t i = 0; i < lists.size(); i += 2) {
            ListNode* l1 = lists[i];
            ListNode* l2 = (i + 1 < lists.size()) ? lists[i + 1] : nullptr;
            merged.push_back(merge_two_lists(l1, l2));
        }
        lists = move(merged);
    }
    return lists[0];
}
```

*Alternative - Min Heap:* $O(n log k)$ time but worse constants. Priority queue operations = 3-5x overhead vs direct merge. Use only if streaming input.

*Memory insight:* Divide-and-conquer reuses pointers (no new nodes). Heap creates temporary nodes. Zero allocation = zero malloc/free overhead (typically $#sym.tilde.op$20-100 cycles each for modern allocators like tcmalloc, jemalloc).

== Bottom-Up Merge Sort for Linked Lists

*Problem:* Sort a linked list in $O(n log n)$ time with $O(1)$ extra space.

*Approach:* Iteratively merge sublists of size 1, 2, 4, 8, ... (bottom-up, avoids recursion overhead).

```cpp
ListNode* sort_list(ListNode* head) {
    if (!head || !head->next) return head;

    // Count length
    int len = 0;
    for (ListNode* curr = head; curr; curr = curr->next) len++;

    ListNode dummy(0, head);

    for (int size = 1; size < len; size *= 2) {
        ListNode* prev = &dummy;
        ListNode* curr = dummy.next;

        while (curr) {
            // Split off first sublist of 'size'
            ListNode* left = curr;
            for (int i = 1; i < size && curr->next; i++) curr = curr->next;
            ListNode* right = curr->next;
            curr->next = nullptr;

            // Split off second sublist of 'size'
            curr = right;
            for (int i = 1; i < size && curr && curr->next; i++) curr = curr->next;
            ListNode* next = nullptr;
            if (curr) {
                next = curr->next;
                curr->next = nullptr;
            }

            // Merge and attach
            prev->next = merge_two_lists(left, right);
            while (prev->next) prev = prev->next;
            curr = next;
        }
    }

    return dummy.next;
}
```

*Why bottom-up?* Top-down merge sort uses $O(log n)$ stack space for recursion. Bottom-up is truly $O(1)$ extra space. Both are $O(n log n)$ time.

*Why linked lists for merge sort?* Merging two sorted linked lists is $O(1)$ extra space (re-link pointers). For arrays, merge requires $O(n)$ temp buffer. This makes merge sort the preferred $O(n log n)$ sort for linked lists.

== Complexity Reference

#table(
  columns: (auto, auto, auto),
  [*Operation*], [*Time*], [*Space*],
  [Access by index], [$O(n)$], [$O(1)$],
  [Insert at head], [$O(1)$], [$O(1)$],
  [Insert at tail (with tail ptr)], [$O(1)$], [$O(1)$],
  [Insert at position], [$O(n)$], [$O(1)$],
  [Delete at position], [$O(n)$], [$O(1)$],
  [Search], [$O(n)$], [$O(1)$],
  [Reverse], [$O(n)$], [$O(1)$],
  [Cycle detection (Floyd's)], [$O(n)$], [$O(1)$],
  [Merge two sorted], [$O(n + m)$], [$O(1)$],
  [Merge K sorted], [$O(N log K)$], [$O(log K)$ or $O(K)$],
  [Sort (merge sort)], [$O(n log n)$], [$O(1)$ bottom-up, $O(log n)$ top-down],
  [LRU Cache (get/put)], [$O(1)$], [$O(n)$ total],
)

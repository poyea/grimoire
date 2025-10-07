= Linked List

== Reverse Linked List

*Problem:* Reverse a singly linked list.

*Approach 1 - Iterative:* $O(n)$ time, $O(1)$ space
- Initialize `prev = None`
- While `head`:
  + Store next: `temp = head.next`
  + Reverse pointer: `head.next = prev`
  + Move prev and head forward: `prev = head`, `head = temp`
- Return `prev`

*Approach 2 - Recursive:* $O(n)$ time, $O(n)$ space (call stack)
- Base case: if `head is None`: return None
- Recursively reverse rest: `newHead = reverseList(head.next)`
- Reverse current: `head.next.next = head`
- Set current next to None: `head.next = None`
- Return `newHead`

== Merge Two Sorted Lists

*Problem:* Merge two sorted linked lists into one sorted list.

*Approach - Dummy Node:* $O(n + m)$ time, $O(1)$ space
- Create dummy node: `result = ListNode()`
- Initialize pointer: `curr = result`
- While both `list1` and `list2`:
  + If `list1.val < list2.val`:
    - `curr.next = list1`, `list1 = list1.next`
  + Else:
    - `curr.next = list2`, `list2 = list2.next`
  + Move curr: `curr = curr.next`
- Attach remaining nodes: `curr.next = list1 or list2`
- Return `result.next` (skip dummy)

== Reorder List

*Problem:* Reorder list to L0→Ln→L1→Ln-1→L2→Ln-2→...

*Approach - Stack:* $O(n)$ time, $O(n)$ space
- Push all node values to stack
- Initialize `curr = head`, pop first element from stack
- While stack not empty:
  + Alternate between popping from front and back
  + Create new node and attach: `curr.next = ListNode(val)`
  + Move curr: `curr = curr.next`
- Set `head.next = None` before loop to avoid cycle

*Alternative $O(1)$ space:* Find middle, reverse second half, merge two halves

== Remove Nth Node From End of List

*Problem:* Remove nth node from end of linked list.

*Approach - Two Pointers:* $O(n)$ time, $O(1)$ space
- Create dummy node: `dummy = ListNode(0, head)`
- Initialize `right = head`
- Move right n steps forward:
  + For i in range(n): `right = right.next`
- Initialize `left = dummy`
- Move both pointers until right reaches end:
  + While `right`: `left = left.next`, `right = right.next`
- Remove node: `left.next = left.next.next`
- Return `dummy.next`

*Key insight:* n-step gap between pointers ensures left stops at node before target.

== Linked List Cycle

*Problem:* Detect if linked list has a cycle.

*Approach - Floyd's Cycle Detection (Fast & Slow):* $O(n)$ time, $O(1)$ space
- Initialize `slow = head`, `fast = head`
- While `fast` and `fast.next`:
  + Move pointers: `slow = slow.next`, `fast = fast.next.next`
  + If `slow == fast`: return True (cycle detected)
- Return False

*Key insight:* Fast pointer catches up to slow pointer if there's a cycle.

== Merge K Sorted Lists

*Problem:* Merge k sorted linked lists into one sorted list.

*Approach - Divide and Conquer:* $O(n log k)$ time where n is total nodes, k is number of lists
- Define helper function `mergeTwoLists(l1, l2)` (same as problem 2)
- While `len(lists) > 1`:
  + Create `mergedLists = []`
  + For i in range(0, len(lists), 2):
    - Get `list1 = lists[i]`
    - Get `list2 = lists[i+1]` if exists, else None
    - Append `mergeTwoLists(list1, list2)` to mergedLists
  + Update `lists = mergedLists`
- Return `lists[0]`

*Key insight:* Pair-wise merging reduces k lists in $O(log k)$ rounds.

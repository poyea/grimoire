= Greedy

*Greedy algorithms* build a solution incrementally, choosing the locally optimal option at each step, hoping it leads to a globally optimal solution.

*Two key properties required:*
1. *Greedy choice property:* A locally optimal choice can always be part of some globally optimal solution
2. *Optimal substructure:* An optimal solution to the problem contains optimal solutions to subproblems

*Proof strategies:*
- *Exchange argument:* Assume an optimal solution $O$ differs from greedy solution $G$. Show you can swap a choice in $O$ for the greedy choice without worsening the result, contradicting $O$'s supposed superiority.
- *Greedy stays ahead:* Show by induction that after each step, the greedy solution is at least as good as any other solution at that point.

*See also:* Dynamic Programming (when greedy fails), Graphs (Dijkstra, Prim, Kruskal are greedy), Heaps (priority queues used in many greedy algorithms)

== Maximum Subarray (Kadane's Algorithm)

*Problem:* Find contiguous subarray with maximum sum.

*Approach - Kadane's Algorithm:* $O(n)$ time, $O(1)$ space

```cpp
int max_sub_array(vector<int>& nums) {
    int max_sum = nums[0];
    int curr_sum = 0;

    for (int num : nums) {
        if (curr_sum < 0) curr_sum = 0;  // Reset if negative
        curr_sum += num;
        max_sum = max(max_sum, curr_sum);
    }

    return max_sum;
}
```

*Key insight:* Negative prefix sum can never help future subarrays -- discard it.

== Interval Scheduling (Activity Selection)

*Problem:* Given $n$ intervals $[s_i, f_i)$, select maximum number of non-overlapping intervals.

*Approach - Sort by finish time:* $O(n log n)$ time, $O(1)$ extra space

```cpp
int max_non_overlapping(vector<vector<int>>& intervals) {
    // Sort by finish time
    sort(intervals.begin(), intervals.end(),
         [](const vector<int>& a, const vector<int>& b) {
             return a[1] < b[1];
         });

    int count = 1;
    int last_finish = intervals[0][1];

    for (int i = 1; i < (int)intervals.size(); i++) {
        if (intervals[i][0] >= last_finish) {  // No overlap
            count++;
            last_finish = intervals[i][1];
        }
    }

    return count;
}
```

*Key insight:* Always pick the interval that finishes earliest. This leaves the most room for future intervals.

*Proof (greedy stays ahead):* After selecting $k$ intervals, the greedy finish time $f_k^G <= f_k^A$ for any other valid selection $A$. By induction, greedy can always fit at least as many intervals as any other strategy.

*Why sort by start time fails:* An interval $[0, 100)$ starting earliest blocks all others. Finish time avoids this.

== Interval Partitioning

*Problem:* Given $n$ intervals (e.g., lectures), find the minimum number of rooms so no two overlapping intervals share a room.

*Approach - Sweep line with min-heap:* $O(n log n)$ time, $O(n)$ space

```cpp
int min_rooms(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end());  // Sort by start
    // Min-heap of end times (earliest ending room on top)
    priority_queue<int, vector<int>, greater<int>> pq;

    for (auto& interval : intervals) {
        // If earliest-ending room is free, reuse it
        if (!pq.empty() && pq.top() <= interval[0]) {
            pq.pop();
        }
        pq.push(interval[1]);
    }

    return pq.size();
}
```

*Key insight:* The answer equals the maximum number of intervals overlapping at any point (the depth). The min-heap tracks room end times; we reuse the earliest-ending room when possible.

*Alternative -- event sweep:* Count +1 at each start, -1 at each end. Max running count = answer. Same $O(n log n)$ due to sort.

== Fractional Knapsack

*Problem:* Given items with weights and values, and capacity $W$, maximize total value. Items can be split (take fractions).

*Approach - Sort by value/weight ratio:* $O(n log n)$ time

```cpp
double fractional_knapsack(vector<int>& weights, vector<int>& values, int capacity) {
    int n = weights.size();
    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);

    // Sort by value-to-weight ratio descending
    sort(idx.begin(), idx.end(), [&](int a, int b) {
        return (double)values[a] / weights[a] > (double)values[b] / weights[b];
    });

    double total_value = 0;
    int remaining = capacity;

    for (int i : idx) {
        if (remaining == 0) break;
        int take = min(weights[i], remaining);
        total_value += (double)take / weights[i] * values[i];
        remaining -= take;
    }

    return total_value;
}
```

*Key insight:* Greedily take items with highest value-per-unit-weight first. If the last item doesn't fully fit, take a fraction.

*Greedy vs 0/1 knapsack:* Fractional knapsack works greedily because you can take partial items. The 0/1 variant requires DP because a locally optimal choice (highest ratio) might leave awkward remaining capacity. Example: items $(w=3, v=4)$ and $(w=2, v=3), (w=2, v=3)$ with $W=4$. Greedy picks the first item (ratio 1.33), value = 4. Optimal picks the two smaller items, value = 6.

== Huffman Coding

*Problem:* Build a prefix-free binary code that minimizes total encoding length, given character frequencies.

*Approach - Priority queue (min-heap):* $O(n log n)$ time

```cpp
struct HuffmanNode {
    int freq;
    char ch;
    HuffmanNode *left, *right;
    HuffmanNode(char c, int f) : freq(f), ch(c), left(nullptr), right(nullptr) {}
};

HuffmanNode* build_huffman_tree(const string& chars, const vector<int>& freqs) {
    auto cmp = [](HuffmanNode* a, HuffmanNode* b) { return a->freq > b->freq; };
    priority_queue<HuffmanNode*, vector<HuffmanNode*>, decltype(cmp)> pq(cmp);

    for (int i = 0; i < (int)chars.size(); i++) {
        pq.push(new HuffmanNode(chars[i], freqs[i]));
    }

    while (pq.size() > 1) {
        HuffmanNode* left = pq.top(); pq.pop();
        HuffmanNode* right = pq.top(); pq.pop();
        HuffmanNode* parent = new HuffmanNode('\0', left->freq + right->freq);
        parent->left = left;
        parent->right = right;
        pq.push(parent);
    }

    return pq.top();
}

// Generate codes via DFS
void build_codes(HuffmanNode* node, string prefix,
                unordered_map<char, string>& codes) {
    if (!node) return;
    if (!node->left && !node->right) {
        codes[node->ch] = prefix.empty() ? "0" : prefix;
        return;
    }
    build_codes(node->left, prefix + "0", codes);
    build_codes(node->right, prefix + "1", codes);
}
```

*Key insight:* Merge the two least-frequent symbols first. This pushes rare characters deeper in the tree (longer codes) and frequent characters nearer the root (shorter codes).

*Prefix-free property:* No code is a prefix of another, so decoding is unambiguous -- traverse the tree bit by bit from root to leaf.

*Optimality:* Proven via exchange argument. If any two sibling leaves at the deepest level are not the two least-frequent characters, swapping them in reduces total cost.

== Jump Game Patterns

=== Can Reach End

*Problem:* Each element is max jump length. Can you reach the last index?

*Approach - Forward greedy:* $O(n)$ time, $O(1)$ space

```cpp
bool can_jump(vector<int>& nums) {
    int max_reach = 0;

    for (int i = 0; i <= max_reach && i < (int)nums.size(); i++) {
        max_reach = max(max_reach, i + nums[i]);
    }

    return max_reach >= (int)nums.size() - 1;
}
```

*Key insight:* Track the farthest reachable index. If current index exceeds `max_reach`, we're stuck.

=== Minimum Jumps

*Problem:* Minimum number of jumps to reach the last index (guaranteed reachable).

*Approach - BFS-style greedy:* $O(n)$ time, $O(1)$ space

```cpp
int min_jumps(vector<int>& nums) {
    int jumps = 0;
    int cur_end = 0;    // End of current BFS level
    int farthest = 0;  // Farthest we can reach from this level

    for (int i = 0; i < (int)nums.size() - 1; i++) {
        farthest = max(farthest, i + nums[i]);
        if (i == cur_end) {      // Exhausted current level
            jumps++;
            cur_end = farthest;
            if (cur_end >= (int)nums.size() - 1) break;
        }
    }

    return jumps;
}
```

*Key insight:* Treat as BFS over index ranges. Each "level" represents positions reachable with one more jump. `cur_end` marks the boundary; when we reach it, we must jump.

== Task Scheduling -- Minimizing Lateness

*Problem:* $n$ jobs each with processing time $t_i$ and deadline $d_i$. Schedule on one machine to minimize maximum lateness $max(0, f_i - d_i)$.

*Approach - Earliest deadline first (EDF):* $O(n log n)$ time

```cpp
int minimize_lateness(vector<int>& durations, vector<int>& deadlines) {
    int n = durations.size();
    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);

    // Sort by deadline ascending
    sort(idx.begin(), idx.end(), [&](int a, int b) {
        return deadlines[a] < deadlines[b];
    });

    int time = 0;
    int max_lateness = 0;

    for (int i : idx) {
        time += durations[i];
        max_lateness = max(max_lateness, time - deadlines[i]);
    }

    return max_lateness;
}
```

*Key insight:* Process jobs in deadline order. No benefit to delaying an urgent job.

*Proof (exchange argument):* If two adjacent jobs $i, j$ are out of deadline order ($d_i > d_j$ but $i$ before $j$), swapping them doesn't increase max lateness. Repeated swaps reach the sorted order without increasing cost.

== Gas Station (Circular Tour)

*Problem:* $n$ gas stations in a circle. Station $i$ has `gas[i]` fuel and costs `cost[i]` to reach the next station. Find a starting station for a complete circuit, or $-1$ if impossible.

*Approach - Single pass:* $O(n)$ time, $O(1)$ space

```cpp
int can_complete_circuit(vector<int>& gas, vector<int>& cost) {
    int total_surplus = 0;
    int current_surplus = 0;
    int start = 0;

    for (int i = 0; i < (int)gas.size(); i++) {
        int diff = gas[i] - cost[i];
        total_surplus += diff;
        current_surplus += diff;

        if (current_surplus < 0) {
            // Can't start from 'start' or anything before i
            start = i + 1;
            current_surplus = 0;
        }
    }

    return total_surplus >= 0 ? start : -1;
}
```

*Key insight:* If total gas $>=$ total cost, a solution exists. If starting at `s` fails at station `i`, then no station in $[s, i]$ can be a valid start (they would fail even sooner because they skip early surplus). So jump start to $i + 1$.

== Greedy vs DP Decision Guide

*Use greedy when:*
- Problem has the greedy choice property (local optimum leads to global optimum)
- You can prove correctness via exchange argument or greedy-stays-ahead
- Common signals: sorting + single pass, interval problems, minimum/maximum under a simple constraint

*Use DP when:*
- Greedy choice fails (local optimum misses global optimum)
- Choices have non-trivial dependencies on future decisions
- Problem has overlapping subproblems

*Classic greedy failures:*

#table(
  columns: (auto, auto, auto),
  [*Problem*], [*Greedy attempt*], [*Why it fails*],
  [0/1 Knapsack], [Pick highest value/weight], [Can't take fractions; wastes capacity],
  [Coin change (arbitrary)], [Pick largest coin first], [e.g., coins {1, 3, 4}, amount 6: greedy gives 4+1+1=3 coins, optimal 3+3=2],
  [Longest path in graph], [Pick heaviest edge], [May lead to dead end or cycle],
  [Edit distance], [Match greedily left-to-right], [Misses non-obvious alignments],
)

*Rule of thumb:* If you can sort by one criterion and make irrevocable choices in a single pass with a correctness proof, it's greedy. If you need to "try both options" at each step, it's DP.

== Complexity Reference

#table(
  columns: (auto, auto, auto),
  [*Problem*], [*Time*], [*Space*],
  [Kadane's (max subarray)], [$O(n)$], [$O(1)$],
  [Interval scheduling], [$O(n log n)$], [$O(1)$],
  [Interval partitioning], [$O(n log n)$], [$O(n)$],
  [Fractional knapsack], [$O(n log n)$], [$O(n)$],
  [Huffman coding], [$O(n log n)$], [$O(n)$],
  [Jump game (can reach)], [$O(n)$], [$O(1)$],
  [Jump game (min jumps)], [$O(n)$], [$O(1)$],
  [Minimize lateness (EDF)], [$O(n log n)$], [$O(n)$],
  [Gas station / circular tour], [$O(n)$], [$O(1)$],
)

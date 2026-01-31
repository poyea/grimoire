= Stack

*Stacks enforce LIFO (Last-In, First-Out) ordering. Key patterns: matching/nesting, monotonic stacks for next-greater/smaller queries, and expression evaluation. $O(1)$ push/pop, $O(n)$ total across all operations for monotonic patterns.*

*See also:* Arrays (for cache-friendly linear scans), Trees (for DFS traversal using implicit stack), Backtracking (for explicit state stack)

== Valid Parentheses

*Problem:* Determine if string of parentheses is valid (properly opened and closed in correct order).

*Approach - Stack:* $O(n)$ time, $O(n)$ space

```cpp
bool isValid(string s) {
    // Static lookup: branch predictor friendly, no cache misses
    constexpr array<char, 128> pairs = []() {
        array<char, 128> a{};
        a[')'] = '('; a[']'] = '['; a['}'] = '{';
        return a;
    }();

    vector<char> stack;
    stack.reserve(s.length() / 2);  // Avoid reallocation

    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') {
            stack.push_back(c);
        } else {
            if (stack.empty() || stack.back() != pairs[c]) {
                return false;
            }
            stack.pop_back();
        }
    }
    return stack.empty();
}
```

*Memory optimization:*
- `vector` uses contiguous memory: excellent cache locality vs `std::stack` wrapper
- `reserve()` eliminates reallocation overhead (2x amortized growth = wasted copies)
- Stack grows/shrinks at end: optimal for CPU cache (hot data stays in L1)

*Alternative - Static array stack:*
```cpp
bool isValid(string s) {
    char stack[10000];  // Stack-allocated, zero malloc overhead
    int top = -1;

    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') {
            stack[++top] = c;
        } else {
            char expected = c == ')' ? '(' : (c == ']' ? '[' : '{');
            if (top < 0 || stack[top--] != expected) return false;
        }
    }
    return top == -1;
}
```

*Performance comparison:*
- `std::stack<char>`: Uses `std::deque` by default (chunked $#sym.tilde.op$4KB blocks per implementation). $#sym.tilde.op$10-20% overhead vs vector due to indirection [libstdc++ and libc++ default chunk size]
- `vector<char>`: Contiguous memory, excellent prefetcher efficiency
- Static array: Fastest option (no bounds checking, no heap allocation). *Warning:* Large stack arrays (10KB+) risk stack overflow on some platforms

*Hardware insight:* Stack operations exhibit temporal locality - recently pushed data is immediately popped. Modern CPUs keep stack top in L1 cache (< 4 cycles access).

#pagebreak()

== Min Stack

*Problem:* Design a stack supporting push, pop, top, and retrieving the minimum element, all in $O(1)$ time.

*Approach - Paired stack:* $O(1)$ all operations, $O(n)$ space

```cpp
class MinStack {
    vector<pair<int, int>> data;  // (value, current_min)

public:
    void push(int val) {
        int curMin = data.empty() ? val : min(val, data.back().second);
        data.push_back({val, curMin});
    }

    void pop() { data.pop_back(); }

    int top() { return data.back().first; }

    int getMin() { return data.back().second; }
};
```

*Key insight:* Store the running minimum alongside each element. When we pop, the previous minimum is already recorded in the element below.

*Space optimization - Single stack with encoding:*
```cpp
class MinStack {
    vector<long long> data;
    long long minVal;

public:
    void push(int val) {
        if (data.empty()) {
            data.push_back(0);
            minVal = val;
        } else {
            data.push_back((long long)val - minVal);  // Store diff
            if (val < minVal) minVal = val;
        }
    }

    void pop() {
        long long top = data.back();
        data.pop_back();
        if (top < 0) minVal -= top;  // Restore previous min
    }

    int top() {
        long long top = data.back();
        return top > 0 ? (int)(minVal + top) : (int)minVal;
    }

    int getMin() { return (int)minVal; }
};
```

*Trade-off:* Saves $#sym.tilde.op$50% memory (one value per entry vs two) but uses `long long` for overflow safety and adds arithmetic overhead per operation.

*Cache behavior:* Both approaches use contiguous `vector` storage. The paired approach has better spatial locality (no arithmetic) but 2x the memory footprint, meaning fewer elements fit in L1 cache ($#sym.tilde.op$4K pairs vs $#sym.tilde.op$4K singles in 32KB L1).

== Daily Temperatures

*Problem:* Given daily temperatures, find how many days until a warmer temperature. Return 0 if no warmer day exists.

*Approach - Monotonic decreasing stack:* $O(n)$ time, $O(n)$ space

```cpp
vector<int> dailyTemperatures(vector<int>& temps) {
    int n = temps.size();
    vector<int> result(n, 0);
    vector<int> stack;  // Indices of unresolved days

    for (int i = 0; i < n; i++) {
        while (!stack.empty() && temps[i] > temps[stack.back()]) {
            int prev = stack.back();
            stack.pop_back();
            result[prev] = i - prev;
        }
        stack.push_back(i);
    }

    return result;
}
```

*Key insight:* Maintain a stack of indices in decreasing temperature order. When a warmer day arrives, it resolves all colder days waiting on the stack.

*Amortized analysis:* Each index is pushed once and popped at most once. Total operations = $2n$ across the loop, so $O(n)$ despite the inner `while`.

*Cache behavior:*
- Forward iteration over `temps[]`: excellent spatial locality, hardware prefetcher engaged
- Stack access at back: temporal locality, stays in L1
- `result[]` writes are scattered (indices from stack), but array is contiguous so prefetcher helps

#pagebreak()

== Monotonic Stack Pattern

The monotonic stack is a general technique for "next greater/smaller element" queries.

*Template - Next Greater Element:*
```cpp
// For each element, find the next element to the right that is strictly greater.
// Returns -1 if no such element exists.
vector<int> nextGreater(vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, -1);
    vector<int> stack;  // Monotonically decreasing

    for (int i = 0; i < n; i++) {
        while (!stack.empty() && nums[i] > nums[stack.back()]) {
            result[stack.back()] = nums[i];
            stack.pop_back();
        }
        stack.push_back(i);
    }
    return result;
}
```

*Variants:*
- *Next smaller:* Change `>` to `<` in the while condition
- *Previous greater:* Iterate right-to-left
- *Circular array:* Loop `2n` times using `i % n`

*Next Greater in Circular Array:*
```cpp
vector<int> nextGreaterCircular(vector<int>& nums) {
    int n = nums.size();
    vector<int> result(n, -1);
    vector<int> stack;

    for (int i = 0; i < 2 * n; i++) {
        while (!stack.empty() && nums[i % n] > nums[stack.back()]) {
            result[stack.back()] = nums[i % n];
            stack.pop_back();
        }
        if (i < n) stack.push_back(i);
    }
    return result;
}
```

*Performance:* Still $O(n)$ amortized. The second pass only pops remaining elements; at most $n$ additional pops total.

== Largest Rectangle in Histogram

*Problem:* Find the largest rectangular area in a histogram.

*Approach - Monotonic increasing stack:* $O(n)$ time, $O(n)$ space

```cpp
int largestRectangleArea(vector<int>& heights) {
    int n = heights.size();
    vector<int> stack;
    int maxArea = 0;

    for (int i = 0; i <= n; i++) {
        int h = (i == n) ? 0 : heights[i];  // Sentinel to flush stack

        while (!stack.empty() && h < heights[stack.back()]) {
            int height = heights[stack.back()];
            stack.pop_back();
            int width = stack.empty() ? i : (i - stack.back() - 1);
            maxArea = max(maxArea, height * width);
        }
        stack.push_back(i);
    }

    return maxArea;
}
```

*Key insight:* For each bar, find how far it can extend left and right. A bar extends until it hits a shorter bar. The monotonic increasing stack naturally tracks the left boundary, and the current index provides the right boundary when we pop.

*Why the sentinel:* Appending a height of 0 at position $n$ forces all remaining bars off the stack, so every bar gets its area computed without a separate cleanup loop.

*Width calculation:*
- If stack is empty after pop: the popped bar was the shortest so far, width = $i$
- Otherwise: width = $i - "stack.back()" - 1$ (bars between current left boundary and right boundary)

*Extension - Maximal Rectangle in Binary Matrix:*
```cpp
int maximalRectangle(vector<vector<char>>& matrix) {
    if (matrix.empty()) return 0;
    int cols = matrix[0].size();
    vector<int> heights(cols, 0);
    int maxArea = 0;

    for (auto& row : matrix) {
        for (int j = 0; j < cols; j++) {
            heights[j] = (row[j] == '1') ? heights[j] + 1 : 0;
        }
        maxArea = max(maxArea, largestRectangleArea(heights));
    }
    return maxArea;
}
```

Reduces 2D problem to repeated 1D histogram problems. Each row builds cumulative heights, then applies the histogram algorithm. $O(r #sym.times c)$ time.

#pagebreak()

== Expression Evaluation

*Problem:* Evaluate arithmetic expression with `+`, `-`, `*`, `/`, and parentheses.

*Approach - Two stacks (operators + operands):* $O(n)$ time, $O(n)$ space

```cpp
int calculate(string s) {
    vector<long long> nums;
    vector<char> ops;

    auto precedence = [](char op) -> int {
        if (op == '+' || op == '-') return 1;
        if (op == '*' || op == '/') return 2;
        return 0;
    };

    auto applyOp = [&]() {
        long long b = nums.back(); nums.pop_back();
        long long a = nums.back(); nums.pop_back();
        char op = ops.back(); ops.pop_back();
        switch (op) {
            case '+': nums.push_back(a + b); break;
            case '-': nums.push_back(a - b); break;
            case '*': nums.push_back(a * b); break;
            case '/': nums.push_back(a / b); break;
        }
    };

    for (int i = 0; i < s.size(); i++) {
        if (s[i] == ' ') continue;

        if (isdigit(s[i])) {
            long long num = 0;
            while (i < s.size() && isdigit(s[i])) {
                num = num * 10 + (s[i++] - '0');
            }
            i--;
            nums.push_back(num);
        } else if (s[i] == '(') {
            ops.push_back('(');
        } else if (s[i] == ')') {
            while (ops.back() != '(') applyOp();
            ops.pop_back();  // Remove '('
        } else {
            while (!ops.empty() && ops.back() != '(' &&
                   precedence(ops.back()) >= precedence(s[i])) {
                applyOp();
            }
            ops.push_back(s[i]);
        }
    }
    while (!ops.empty()) applyOp();
    return (int)nums.back();
}
```

*Operator precedence:* Process higher-precedence operators before pushing lower-precedence ones. This implements the shunting-yard algorithm (Dijkstra, 1961) inline.

*Parentheses:* `(` acts as a barrier on the operator stack. `)` triggers evaluation until matching `(` is found.

== Stack Deep Dive

*`std::stack` vs `vector` as stack:*
#table(
  columns: 4,
  align: (left, right, right, right),
  table.header([Implementation], [Push (ns)], [Pop (ns)], [Memory Overhead]),
  [std::stack\<int\> (deque)], [15-25], [10-15], [$#sym.tilde.op$64 bytes/chunk metadata],
  [vector\<int\>], [5-10], [3-5], [1.5x capacity on average],
  [static array], [2-3], [1-2], [fixed allocation],
)

*Why `std::stack` defaults to `std::deque`:*
- Deque never invalidates references to other elements on push (vector may reallocate)
- Deque allocates in fixed-size chunks (typically 512 bytes), avoiding large contiguous allocations
- In practice, `vector` is faster for most competitive programming and interview use cases due to simpler memory layout

*When to use each:*
- *`vector`*: Default choice. Best cache locality, fastest for small-to-medium stacks
- *`std::stack<T, vector<T>>`*: When you want stack semantics enforced at compile time (no random access)
- *Static array*: When maximum size is known and performance is critical
- *`std::stack` (default deque)*: When you need reference stability across push operations

*Monotonic stack applications summary:*
#table(
  columns: 3,
  align: (left, left, left),
  table.header([Problem Type], [Stack Order], [Lookup Direction]),
  [Next greater element], [Decreasing], [Forward],
  [Next smaller element], [Increasing], [Forward],
  [Previous greater element], [Decreasing], [Backward],
  [Previous smaller element], [Increasing], [Backward],
  [Largest rectangle (histogram)], [Increasing], [Forward + sentinel],
  [Trapping rain water], [Decreasing], [Forward],
  [Stock span], [Decreasing], [Forward],
)

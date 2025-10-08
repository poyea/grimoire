= Stack

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
    // ⚠️ Warning: Large stack arrays risk overflow. Increase stack size if needed.

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
- `std::stack<char>`: Uses `std::deque` by default (chunked $#sym.tilde.op$4KB blocks). $#sym.tilde.op$10-20% overhead vs vector due to indirection
- `vector<char>`: Contiguous memory, excellent prefetcher efficiency
- Static array: Fastest option (no bounds checking, no heap allocation). *Warning:* Large stack arrays (10KB+) risk stack overflow on some platforms

*Hardware insight:* Stack operations exhibit temporal locality - recently pushed data is immediately popped. Modern CPUs keep stack top in L1 cache (< 4 cycles access).

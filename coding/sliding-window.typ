= Sliding Window

== Best Time to Buy and Sell Stock

*Problem:* Find maximum profit from buying and selling stock once (buy before sell).

*Approach - Sliding Window:* $O(n)$ time, $O(1)$ space

```cpp
int maxProfit(vector<int>& prices) {
    int maxProfit = 0;
    int minPrice = prices[0];

    for (int price : prices) {
        maxProfit = max(maxProfit, price - minPrice);
        minPrice = min(minPrice, price);
    }
    return maxProfit;
}
```

*Cache behavior:* Sequential array access = optimal prefetching. CPU fetches 64-byte cache line ($#sym.tilde.op$16 ints), amortizing memory latency across elements.

*Compiler optimization:* Modern compilers already optimize `max()`/`min()` to conditional moves (CMOV) when beneficial. Manual arithmetic tricks are unnecessary and often slower.

== Longest Substring Without Repeating Characters

*Problem:* Find length of longest substring without repeating characters.

*Approach - Sliding Window + Hash:* $O(n)$ time, $O(min(n, "charset"))$ space

```cpp
int lengthOfLongestSubstring(string s) {
    array<int, 128> lastSeen;  // ASCII charset
    lastSeen.fill(-1);

    int maxLen = 0;
    int left = 0;

    for (int right = 0; right < s.length(); right++) {
        // If char seen in current window, shrink from left
        if (lastSeen[s[right]] >= left) {
            left = lastSeen[s[right]] + 1;
        }
        lastSeen[s[right]] = right;
        maxLen = max(maxLen, right - left + 1);
    }
    return maxLen;
}
```

*Memory hierarchy:*
- `array<int, 128>` = 512 bytes fits in L1 cache (32-48KB typical)
- Stack allocation avoids heap overhead ($#sym.tilde.op$20-100 cycles for malloc in modern allocators)
- `unordered_set<char>` requires heap + pointer chasing = 3-5x slower

*Cache miss analysis:* String scan is sequential (good). Array lookups have random access pattern but data fits in L1 (< 4 cycle latency).

== Longest Repeating Character Replacement

*Problem:* Find length of longest substring with same character after replacing at most k characters.

*Approach - Sliding Window:* $O(n)$ time, $O(26) = O(1)$ space

```cpp
int characterReplacement(string s, int k) {
    array<int, 26> count = {};
    int maxLen = 0;
    int maxFreq = 0;
    int left = 0;

    for (int right = 0; right < s.length(); right++) {
        count[s[right] - 'A']++;
        maxFreq = max(maxFreq, count[s[right] - 'A']);

        // Window invalid: need to replace more than k chars
        int windowLen = right - left + 1;
        if (windowLen - maxFreq > k) {
            count[s[left] - 'A']--;
            left++;
        }
        maxLen = max(maxLen, right - left + 1);
    }
    return maxLen;
}
```

*Key insight:* Window is valid if `windowLength - maxFrequency <= k` (replace all non-frequent chars).

*Optimization:* No need to recompute maxFreq when shrinking window. It only matters for expanding window. maxFreq is monotonic upper bound.

== Minimum Window Substring

*Problem:* Find minimum window in string s that contains all characters from string t.

*Approach - Sliding Window:* $O(n + m)$ time, $O("charset")$ space

```cpp
string minWindow(string s, string t) {
    if (t.length() > s.length()) return "";

    array<int, 128> required = {}, window = {};
    int need = 0;

    for (char c : t) {
        if (required[c]++ == 0) need++;
    }

    int have = 0;
    int minLen = INT_MAX, minStart = 0;
    int left = 0;

    for (int right = 0; right < s.length(); right++) {
        char c = s[right];
        window[c]++;

        if (window[c] == required[c]) have++;

        // Contract window while valid
        while (have == need) {
            if (right - left + 1 < minLen) {
                minLen = right - left + 1;
                minStart = left;
            }

            char leftChar = s[left];
            if (window[leftChar] == required[leftChar]) have--;
            window[leftChar]--;
            left++;
        }
    }

    return minLen == INT_MAX ? "" : s.substr(minStart, minLen);
}
```

*Key insight:* Expand window until valid, then contract to find minimum valid window.

*Performance notes:*
- `array<int, 128>` vs `unordered_map<char, int>`: array is 10-20x faster (no hashing, no collisions, cache-friendly)
- `substr()` copies data. For read-only: return `string_view(s.data() + minStart, minLen)` (C++17, zero-copy)
- Two-pass algorithm allows SIMD vectorization of counting phase

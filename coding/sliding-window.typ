= Sliding Window

== Best Time to Buy and Sell Stock

*Problem:* Find maximum profit from buying and selling stock once (buy before sell).

*Approach - Sliding Window:* $O(n)$ time, $O(1)$ space
- Initialize `maxProfit = 0`, `left = 0` (buy day), `right = 1` (sell day)
- While `right < len(prices)`:
  + Calculate profit: `prices[right] - prices[left]`
  + Update `maxProfit` if current profit is greater
  + If `prices[left] > prices[right]`: move `left = right` (found better buy day)
  + Increment `right`
- Return `maxProfit`

*Key insight:* Always buy at lowest price seen so far and check profit at each subsequent day.

== Longest Substring Without Repeating Characters

*Problem:* Find length of longest substring without repeating characters.

*Approach - Sliding Window:* $O(n)$ time, $O(n)$ space
- Initialize `maxLen = 0`, `left = 0`
- For `right` in range(len(s)):
  + Get current window: `s[left:right+1]`
  + If `s[right]` is in window (repeating character):
    - Move `left` to position after first occurrence of `s[right]`
  + Update `maxLen = max(maxLen, right - left + 1)`
- Return `maxLen`

*Key Python concepts:*
- String slicing: `s[left:right]`
- Finding character in string: `char in substring`

== Longest Repeating Character Replacement

*Problem:* Find length of longest substring with same character after replacing at most k characters.

*Approach - Sliding Window:* $O(n)$ time, $O(26)$ = $O(1)$ space
- Create hashmap to count characters in window
- Initialize `maxLen = 0`, `left = 0`
- For `right` in range(len(s)):
  + Add `s[right]` to hashmap
  + Get `maxFreq` = most frequent character count in window
  + If `windowLength - maxFreq > k`: window invalid
    - Decrement `count[s[left]]` and increment `left`
  + Update `maxLen = max(maxLen, right - left + 1)`
- Return `maxLen`

*Key insight:* Window is valid if `windowLength - maxFrequency <= k` (we can replace all non-frequent chars).

== Minimum Window Substring

*Problem:* Find minimum window in string s that contains all characters from string t.

*Approach - Sliding Window:* $O(n)$ time, $O(m)$ space where m is unique chars in t
- Create frequency map `countT` for string t
- Initialize `have = 0`, `need = len(countT)` (unique chars needed)
- Initialize `countS = {}`, `left = 0`, `result = ""`, `minLen = infinity`
- For `right` in range(len(s)):
  + Add `s[right]` to `countS`
  + If `s[right]` in t and counts match: increment `have`
  + While `have == need` (valid window):
    - Update result if current window is smaller
    - Remove `s[left]` from window, decrement `have` if needed
    - Increment `left`
- Return `result`

*Key insight:* Expand window until valid, then contract to find minimum valid window.

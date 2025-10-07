= Hashing

== Contains Duplicate

*Problem:* Check if array has duplicates.

*Hash Set:* $O(n)$ time, $O(n)$ space

```cpp
bool containsDuplicate(vector<int>& nums) {
    unordered_set<int> seen;
    for (int num : nums) {
        if (seen.count(num)) return true;
        seen.insert(num);
    }
    return false;
}
```

*Alternative:* Sort + adjacent check for $O(1)$ space but $O(n log n)$ time.

== Valid Anagram

*Problem:* Check if two strings are anagrams.

*Frequency Map:* $O(n)$ time, $O(1)$ space (26 chars)

```cpp
bool isAnagram(string s, string t) {
    if (s.length() != t.length()) return false;

    unordered_map<char, int> count;
    for (char c : s) count[c]++;
    for (char c : t) {
        if (--count[c] < 0) return false;
    }
    return true;
}
```

*Optimization:* Use `array<int, 26>` for O(1) space if only lowercase letters.

== Two Sum

*Problem:* Find indices `i, j` where `nums[i] + nums[j] == target`.

*Hash Map:* $O(n)$ time, $O(n)$ space

```cpp
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> seen;

    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];
        if (seen.count(complement)) {
            return {seen[complement], i};
        }
        seen[nums[i]] = i;
    }
    return {};
}
```

*Critical:* Check complement before inserting to avoid using same element twice.

== Group Anagrams

*Problem:* Group anagrams together.

*Sorted Key:* $O(n k log k)$ time where k = max string length

```cpp
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> groups;

    for (string& s : strs) {
        string key = s;
        sort(key.begin(), key.end());
        groups[key].push_back(s);
    }

    vector<vector<string>> result;
    for (auto& [_, group] : groups) {
        result.push_back(move(group));
    }
    return result;
}
```

*Optimization:* Use char count array as key for $O(n k)$ time:

```cpp
string getKey(const string& s) {
    array<int, 26> count = {};
    for (char c : s) count[c - 'a']++;

    string key;
    for (int i = 0; i < 26; i++) {
        if (count[i]) {
            key += string(count[i], 'a' + i);
        }
    }
    return key;
}
```

== Top K Frequent Elements

*Problem:* Return k most frequent elements.

*Bucket Sort:* $O(n)$ time, $O(n)$ space

```cpp
vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> freq;
    for (int num : nums) freq[num]++;

    int n = nums.size();
    vector<vector<int>> buckets(n + 1);

    for (auto& [num, count] : freq) {
        buckets[count].push_back(num);
    }

    vector<int> result;
    for (int i = n; i >= 0 && result.size() < k; i--) {
        for (int num : buckets[i]) {
            result.push_back(num);
            if (result.size() == k) return result;
        }
    }
    return result;
}
```

*Alternative - Min Heap:* $O(n log k)$ time, better for streaming data:

```cpp
// Use priority_queue with custom comparator
priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> minHeap;
// Keep heap size ≤ k, top k elements have highest frequency
```

== Longest Consecutive Sequence

*Problem:* Find length of longest consecutive sequence in unsorted array.

*Hash Set:* $O(n)$ time, $O(n)$ space

```cpp
int longestConsecutive(vector<int>& nums) {
    unordered_set<int> numSet(nums.begin(), nums.end());
    int maxLen = 0;

    for (int num : numSet) {
        // Only start counting from sequence starts
        if (!numSet.count(num - 1)) {
            int length = 1;
            while (numSet.count(num + length)) {
                length++;
            }
            maxLen = max(maxLen, length);
        }
    }
    return maxLen;
}
```

*Key optimization:* Check `num - 1` to identify sequence starts. Prevents redundant work, ensures O(n).

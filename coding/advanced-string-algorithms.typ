= Advanced String Algorithms

*Beyond basic pattern matching:* Suffix arrays, suffix automata, and multiple pattern matching enable sophisticated string processing with optimal complexity. Critical for bioinformatics, text indexing, and data compression [Gusfield 1997].

*See also:* String Algorithms (for KMP, Rabin-Karp, Z-algorithm), Tries (for prefix trees), Hashing (for rolling hash techniques)

== Boyer-Moore Algorithm

*Key insight:* Compare pattern right-to-left. On mismatch, skip ahead using precomputed shift tables. Average case $O(n\/m)$ --- sublinear, faster for larger alphabets.

*Two heuristics:*
1. *Bad character rule:* On mismatch at text[i], shift pattern so last occurrence of text[i] in pattern aligns. If not in pattern, skip entire pattern length.
2. *Good suffix rule:* On mismatch after matching a suffix, shift to align with next occurrence of that suffix in pattern. (More complex, omitted for brevity.)

*Bad character implementation:*

```cpp
vector<int> boyer_moore_search(const string& text, const string& pattern) {
    int n = text.size(), m = pattern.size();
    vector<int> matches;
    if (m == 0 || m > n) return matches;

    // Build bad character table
    array<int, 256> bad_char;
    bad_char.fill(-1);
    for (int j = 0; j < m; j++) {
        bad_char[(unsigned char)pattern[j]] = j;
    }

    int i = 0;  // Offset in text
    while (i <= n - m) {
        int j = m - 1;  // Compare right-to-left

        while (j >= 0 && pattern[j] == text[i + j]) {
            j--;
        }

        if (j < 0) {
            matches.push_back(i);
            // Shift: align next char in text after match
            i += (i + m < n) ? m - bad_char[(unsigned char)text[i + m]] : 1;
        } else {
            // Shift by bad character rule
            int shift = j - bad_char[(unsigned char)text[i + j]];
            i += max(1, shift);
        }
    }

    return matches;
}
```

*Boyer-Moore-Horspool (simplified variant):*

Uses only the bad character rule applied to the *last* character of the window. Simpler, nearly as fast in practice.

```cpp
vector<int> horspool_search(const string& text, const string& pattern) {
    int n = text.size(), m = pattern.size();
    vector<int> matches;
    if (m == 0 || m > n) return matches;

    // Shift table: distance from rightmost occurrence to end of pattern
    array<int, 256> shift;
    shift.fill(m);
    for (int j = 0; j < m - 1; j++) {
        shift[(unsigned char)pattern[j]] = m - 1 - j;
    }

    int i = 0;
    while (i <= n - m) {
        if (text.compare(i, m, pattern) == 0) {
            matches.push_back(i);
        }
        i += shift[(unsigned char)text[i + m - 1]];
    }

    return matches;
}
```

*Complexity:*
- Preprocessing: $O(m + |Sigma|)$ where $|Sigma|$ = alphabet size
- Best/average case: $O(n\/m)$ --- sublinear, skips large portions of text
- Worst case (bad char only): $O(n m)$. With good suffix rule: $O(n)$.

*When to prefer Boyer-Moore:* Large alphabets (ASCII, Unicode) where mismatches cause large skips. Less effective on small alphabets (DNA: only 4 chars).

== Suffix Array Construction

*Definition:* Array of starting indices of all suffixes, sorted lexicographically.

*Example:* "banana" has suffixes:
- 0: "banana"
- 1: "anana"
- 2: "nana"
- 3: "ana"
- 4: "na"
- 5: "a"

Suffix array: [5, 3, 1, 0, 4, 2] (sorted: "a", "ana", "anana", "banana", "na", "nana")

*Efficient Construction (Prefix Doubling):* $O(n log n)$

```cpp
class SuffixArray {
    string s;
    vector<int> sa, rank_arr, lcp;
    int n;

    void buildSA() {
        sa.resize(n);
        rank_arr.resize(n);
        vector<int> tmp(n);

        // Initial ranking by first character
        for (int i = 0; i < n; i++) {
            sa[i] = i;
            rank_arr[i] = s[i];
        }

        // Prefix doubling
        for (int k = 1; k < n; k *= 2) {
            // Sort by (rank[i], rank[i+k])
            auto cmp = [&](int a, int b) {
                if (rank_arr[a] != rank_arr[b]) {
                    return rank_arr[a] < rank_arr[b];
                }
                int ra = (a + k < n) ? rank_arr[a + k] : -1;
                int rb = (b + k < n) ? rank_arr[b + k] : -1;
                return ra < rb;
            };

            sort(sa.begin(), sa.end(), cmp);

            // Recompute ranks
            tmp[sa[0]] = 0;
            for (int i = 1; i < n; i++) {
                tmp[sa[i]] = tmp[sa[i - 1]] + cmp(sa[i - 1], sa[i]);
            }
            rank_arr = tmp;

            // Early termination if all ranks unique
            if (rank_arr[sa[n - 1]] == n - 1) break;
        }
    }

    void buildLCP() {
        // Kasai's algorithm: O(n)
        lcp.resize(n - 1);
        vector<int> inv(n);
        for (int i = 0; i < n; i++) {
            inv[sa[i]] = i;
        }

        int k = 0;
        for (int i = 0; i < n; i++) {
            if (inv[i] == 0) {
                k = 0;
                continue;
            }

            int j = sa[inv[i] - 1];
            while (i + k < n && j + k < n && s[i + k] == s[j + k]) {
                k++;
            }

            lcp[inv[i] - 1] = k;
            if (k > 0) k--;
        }
    }

public:
    SuffixArray(const string& str) : s(str), n(str.size()) {
        buildSA();
        buildLCP();
    }

    // Binary search for pattern
    pair<int, int> search(const string& pattern) {
        int m = pattern.size();

        // Find first occurrence
        int lo = 0, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (s.compare(sa[mid], m, pattern) < 0) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        int first = lo;

        // Find last occurrence
        hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (s.compare(sa[mid], m, pattern) <= 0) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        return {first, lo};  // Range [first, lo) contains all matches
    }

    int longestRepeatedSubstring() {
        return *max_element(lcp.begin(), lcp.end());
    }

    const vector<int>& getSA() const { return sa; }
    const vector<int>& getLCP() const { return lcp; }
};
```

*Complexity:*
- Construction: $O(n log n)$ with comparison sort, $O(n log n)$ or $O(n)$ with radix sort
- Pattern search: $O(m log n)$ where m = pattern length
- LCP construction (Kasai): $O(n)$

*SA-IS Algorithm:* Linear time $O(n)$ construction [Nong et al. 2009]. More complex but optimal.

== Suffix Array Applications

*Longest Common Substring:*

```cpp
string longestCommonSubstring(const string& a, const string& b) {
    string combined = a + "#" + b;  // # not in alphabet
    SuffixArray sa(combined);

    auto lcpArr = sa.getLCP();
    auto saArr = sa.getSA();
    int n = combined.size();
    int lenA = a.size();

    int maxLen = 0, startPos = 0;

    for (int i = 0; i < n - 1; i++) {
        // Check if adjacent suffixes come from different strings
        bool fromA1 = saArr[i] < lenA;
        bool fromA2 = saArr[i + 1] < lenA;

        if (fromA1 != fromA2 && lcpArr[i] > maxLen) {
            maxLen = lcpArr[i];
            startPos = saArr[i];
        }
    }

    return combined.substr(startPos, maxLen);
}
```

*Count Distinct Substrings:*

```cpp
int64_t countDistinctSubstrings(const string& s) {
    SuffixArray sa(s);
    auto lcpArr = sa.getLCP();
    int n = s.size();

    // Total substrings - LCP overlaps
    int64_t total = (int64_t)n * (n + 1) / 2;
    for (int lcp : lcpArr) {
        total -= lcp;
    }
    return total;
}
```

== Z-Algorithm (Extended)

*Builds Z-array:* Z[i] = length of longest substring starting at i that matches prefix.

```cpp
vector<int> zFunction(const string& s) {
    int n = s.size();
    vector<int> z(n);
    int l = 0, r = 0;

    for (int i = 1; i < n; i++) {
        if (i < r) {
            z[i] = min(r - i, z[i - l]);
        }

        while (i + z[i] < n && s[z[i]] == s[i + z[i]]) {
            z[i]++;
        }

        if (i + z[i] > r) {
            l = i;
            r = i + z[i];
        }
    }

    return z;
}
```

*Pattern Matching with Z:*

```cpp
vector<int> zSearch(const string& text, const string& pattern) {
    string combined = pattern + "$" + text;
    vector<int> z = zFunction(combined);

    vector<int> matches;
    int m = pattern.size();

    for (int i = m + 1; i < z.size(); i++) {
        if (z[i] == m) {
            matches.push_back(i - m - 1);
        }
    }

    return matches;
}
```

*Complexity:* $O(n + m)$ time, $O(n + m)$ space

== Aho-Corasick Automaton

*Problem:* Multiple pattern matching. Find all occurrences of k patterns in text.

*Key idea:* Build trie + failure links (like KMP for multiple patterns).

```cpp
class AhoCorasick {
    struct Node {
        int children[26];
        int fail;
        vector<int> output;  // Pattern IDs ending here

        Node() : fail(0) {
            fill(begin(children), end(children), -1);
        }
    };

    vector<Node> nodes;
    vector<int> patternLengths;

    void buildTrie(const vector<string>& patterns) {
        nodes.push_back(Node());  // Root

        for (int id = 0; id < patterns.size(); id++) {
            int curr = 0;
            for (char c : patterns[id]) {
                int ch = c - 'a';
                if (nodes[curr].children[ch] == -1) {
                    nodes[curr].children[ch] = nodes.size();
                    nodes.push_back(Node());
                }
                curr = nodes[curr].children[ch];
            }
            nodes[curr].output.push_back(id);
            patternLengths.push_back(patterns[id].size());
        }
    }

    void buildAutomaton() {
        queue<int> q;

        // Initialize depth-1 nodes
        for (int c = 0; c < 26; c++) {
            if (nodes[0].children[c] != -1) {
                nodes[nodes[0].children[c]].fail = 0;
                q.push(nodes[0].children[c]);
            } else {
                nodes[0].children[c] = 0;  // Point back to root
            }
        }

        // BFS to build failure links
        while (!q.empty()) {
            int u = q.front();
            q.pop();

            for (int c = 0; c < 26; c++) {
                if (nodes[u].children[c] != -1) {
                    int v = nodes[u].children[c];
                    int f = nodes[u].fail;

                    while (f && nodes[f].children[c] == -1) {
                        f = nodes[f].fail;
                    }

                    nodes[v].fail = nodes[f].children[c];
                    if (nodes[v].fail == v) nodes[v].fail = 0;

                    // Merge output from failure link
                    auto& out = nodes[nodes[v].fail].output;
                    nodes[v].output.insert(nodes[v].output.end(),
                                           out.begin(), out.end());

                    q.push(v);
                } else {
                    // Optimization: direct jump
                    nodes[u].children[c] = nodes[nodes[u].fail].children[c];
                }
            }
        }
    }

public:
    AhoCorasick(const vector<string>& patterns) {
        buildTrie(patterns);
        buildAutomaton();
    }

    // Returns (position, pattern_id) pairs
    vector<pair<int, int>> search(const string& text) {
        vector<pair<int, int>> matches;
        int curr = 0;

        for (int i = 0; i < text.size(); i++) {
            int c = text[i] - 'a';
            curr = nodes[curr].children[c];

            for (int id : nodes[curr].output) {
                matches.push_back({i - patternLengths[id] + 1, id});
            }
        }

        return matches;
    }
};
```

*Complexity:*
- Build: $O(sum |p_i| times Sigma)$ where $Sigma$ = alphabet size
- Search: $O(|T| + |M|)$ where $T$ = text, $M$ = matches

*Use cases:*
- Malware signature detection
- DNA sequence analysis
- Spam filtering (keyword matching)

== Suffix Automaton

*Most space-efficient:* Recognizes all substrings in $O(n)$ space.

*Property:* Minimum state automaton accepting all suffixes.

```cpp
class SuffixAutomaton {
    struct State {
        int len;      // Longest string in this equivalence class
        int link;     // Suffix link
        int firstPos; // First occurrence position
        map<char, int> next;
    };

    vector<State> st;
    int last;

    void extend(char c) {
        int cur = st.size();
        st.push_back({st[last].len + 1, -1, st[last].len, {}});

        int p = last;
        while (p != -1 && !st[p].next.count(c)) {
            st[p].next[c] = cur;
            p = st[p].link;
        }

        if (p == -1) {
            st[cur].link = 0;
        } else {
            int q = st[p].next[c];
            if (st[p].len + 1 == st[q].len) {
                st[cur].link = q;
            } else {
                // Clone state
                int clone = st.size();
                st.push_back({st[p].len + 1, st[q].link, st[q].firstPos, st[q].next});
                while (p != -1 && st[p].next[c] == q) {
                    st[p].next[c] = clone;
                    p = st[p].link;
                }
                st[q].link = st[cur].link = clone;
            }
        }

        last = cur;
    }

public:
    SuffixAutomaton(const string& s) {
        st.push_back({0, -1, -1, {}});  // Initial state
        last = 0;

        for (char c : s) {
            extend(c);
        }
    }

    bool contains(const string& pattern) {
        int curr = 0;
        for (char c : pattern) {
            if (!st[curr].next.count(c)) return false;
            curr = st[curr].next[c];
        }
        return true;
    }

    // Count occurrences of pattern
    int countOccurrences(const string& pattern) {
        int curr = 0;
        for (char c : pattern) {
            if (!st[curr].next.count(c)) return 0;
            curr = st[curr].next[c];
        }

        // Count paths to terminal states (need preprocessing)
        // Each state represents an equivalence class
        return countPaths(curr);
    }

    // Count distinct substrings
    int64_t countDistinct() {
        int64_t total = 0;
        for (int i = 1; i < st.size(); i++) {
            total += st[i].len - st[st[i].link].len;
        }
        return total;
    }

private:
    // DFS to count paths (preprocessing needed for occurrence counting)
    int countPaths(int v) {
        // Implementation depends on use case
        return 0;
    }
};
```

*Complexity:*
- Construction: $O(n)$ time
- Space: $O(n)$ states (at most $2n - 1$ states)
- Substring check: $O(|P|)$ where $P$ = pattern

*Key properties:*
- Number of states $<= 2n - 1$
- Number of transitions $<= 3n - 4$
- Each equivalence class = set of substrings with same right contexts

== Manacher's Algorithm (Palindrome Finding)

*Problem:* Find all palindromic substrings in $O(n)$.

```cpp
class Manacher {
    string transform(const string& s) {
        string t = "^";
        for (char c : s) {
            t += "#";
            t += c;
        }
        t += "#$";
        return t;
    }

public:
    // Returns array where p[i] = radius of palindrome centered at i (in transformed string)
    vector<int> build(const string& s) {
        string t = transform(s);
        int n = t.size();
        vector<int> p(n, 0);

        int center = 0, right = 0;

        for (int i = 1; i < n - 1; i++) {
            int mirror = 2 * center - i;

            if (i < right) {
                p[i] = min(right - i, p[mirror]);
            }

            // Expand around center
            while (t[i + p[i] + 1] == t[i - p[i] - 1]) {
                p[i]++;
            }

            // Update rightmost palindrome
            if (i + p[i] > right) {
                center = i;
                right = i + p[i];
            }
        }

        return p;
    }

    string longestPalindrome(const string& s) {
        if (s.empty()) return "";

        vector<int> p = build(s);
        int maxLen = 0, centerIdx = 0;

        for (int i = 1; i < p.size() - 1; i++) {
            if (p[i] > maxLen) {
                maxLen = p[i];
                centerIdx = i;
            }
        }

        // Convert back to original string indices
        int start = (centerIdx - maxLen) / 2;
        return s.substr(start, maxLen);
    }

    int countPalindromes(const string& s) {
        vector<int> p = build(s);
        int count = 0;
        for (int i = 1; i < p.size() - 1; i++) {
            count += (p[i] + 1) / 2;  // Number of palindromes centered here
        }
        return count;
    }
};
```

*Complexity:* $O(n)$ time and space

*Key insight:* Reuse previously computed palindrome information via the mirror property.

== Lyndon Factorization (Duval's Algorithm)

*Lyndon word:* Lexicographically smallest rotation of itself.

*Lyndon factorization:* Unique decomposition $s = w_1 w_2 ... w_k$ where each $w_i$ is Lyndon and $w_1 >= w_2 >= ... >= w_k$.

```cpp
vector<string> lyndonFactorization(const string& s) {
    vector<string> result;
    int n = s.size();
    int i = 0;

    while (i < n) {
        int j = i, k = i + 1;

        while (k < n && s[j] <= s[k]) {
            if (s[j] < s[k]) {
                j = i;
            } else {
                j++;
            }
            k++;
        }

        // Output Lyndon words of length (k - j)
        while (i <= j) {
            result.push_back(s.substr(i, k - j));
            i += k - j;
        }
    }

    return result;
}

// Find lexicographically smallest rotation
string minRotation(const string& s) {
    string doubled = s + s;
    vector<string> factors = lyndonFactorization(doubled);

    // First Lyndon factor starting at position < n
    int pos = 0;
    for (const string& f : factors) {
        if (pos + f.size() > s.size()) break;
        if (pos < s.size() && pos + f.size() <= 2 * s.size()) {
            return s.substr(pos) + s.substr(0, pos);
        }
        pos += f.size();
    }
    return s;
}
```

*Complexity:* $O(n)$ time, $O(1)$ extra space (for factorization indices)

== Performance Comparison

#table(
  columns: 4,
  align: (left, center, center, left),
  table.header([Algorithm], [Build], [Query], [Use Case]),
  [Suffix Array], [$O(n log n)$], [$O(m log n)$], [Pattern search, LCP queries],
  [Suffix Automaton], [$O(n)$], [$O(m)$], [Substring check, counting],
  [Aho-Corasick], [$O(sum |p|)$], [$O(n + z)$], [Multiple pattern matching],
  [Boyer-Moore], [$O(m + |Sigma|)$], [$O(n\/m)$ avg], [Single pattern, large alphabets],
  [Z-Algorithm], [$O(n)$], [N/A], [Single pattern, period finding],
  [Manacher], [$O(n)$], [N/A], [Palindrome finding],
)

== References

*Primary Sources:*

*Gusfield, D. (1997)*. Algorithms on Strings, Trees, and Sequences: Computer Science and Computational Biology. Cambridge University Press. ISBN 0-521-58519-8.

*Nong, G., Zhang, S., & Chan, W.H. (2009)*. Linear Suffix Array Construction by Almost Pure Induced-Sorting. DCC 2009, pp. 193-202.

*Aho, A.V. & Corasick, M.J. (1975)*. Efficient String Matching: An Aid to Bibliographic Search. Communications of the ACM 18(6): 333-340.

*Algorithms & Theory:*

*Manacher, G. (1975)*. A New Linear-Time "On-Line" Algorithm for Finding the Smallest Initial Palindrome of a String. Journal of the ACM 22(3): 346-351.

*Blumer, A. et al. (1985)*. The Smallest Automaton Recognizing the Subwords of a Text. Theoretical Computer Science 40: 31-55.

*Duval, J.P. (1983)*. Factorizing Words over an Ordered Alphabet. Journal of Algorithms 4(4): 363-381.

*Kasai, T. et al. (2001)*. Linear-Time Longest-Common-Prefix Computation in Suffix Arrays and Its Applications. CPM 2001, pp. 181-192.

= Tries

*Memory overhead warning:* Naive trie = 26 pointers per node = 208 bytes (64-bit). Empty children waste space. Typical English word datasets: 70-95% of pointers are null due to sparse branching.

== Implement Trie (Prefix Tree)

*Problem:* Implement a trie with insert, search, and startsWith operations.

*Approach - HashMap-based Trie:* $O(m)$ for all operations where m is word/prefix length

```cpp
class TrieNode {
public:
    unordered_map<char, TrieNode*> children;  // Sparse representation
    bool is_end = false;
};

class Trie {
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    void insert(const string& word) {
        TrieNode* curr = root;
        for (char c : word) {
            if (!curr->children.count(c)) {
                curr->children[c] = new TrieNode();
            }
            curr = curr->children[c];
        }
        curr->is_end = true;
    }

    bool search(const string& word) {
        TrieNode* curr = root;
        for (char c : word) {
            if (!curr->children.count(c)) return false;
            curr = curr->children[c];
        }
        return curr->is_end;
    }

    bool starts_with(const string& prefix) {
        TrieNode* curr = root;
        for (char c : prefix) {
            if (!curr->children.count(c)) return false;
            curr = curr->children[c];
        }
        return true;
    }
};
```

*Memory-efficient alternatives:*

*Array-based (lowercase only):*
```cpp
struct TrieNode {
    array<TrieNode*, 26> children{};  // Null-initialized
    bool is_end = false;
};
```
- Fixed 208-byte overhead per node
- O(1) lookup vs O(log 26) for map
- Cache-friendly: children array contiguous

*Compressed trie (Patricia/Radix):*
- Merge single-child chains into edges with string labels
- 10-100x memory reduction for English words
- Complexity: harder to implement

*Cache analysis:*
- Each level traversal = pointer dereference = potential cache miss
- Word length m = m pointer chases = up to m × $#sym.tilde.op$200 cycles worst case (L3 miss to main memory) [Drepper 2007, §3.3]
- `unordered_map` adds hash computation + bucket lookup overhead

== Trie Delete

*Problem:* Delete a word from the trie. Must not break other words sharing prefixes.

```cpp
bool remove(TrieNode* node, const string& word, int depth = 0) {
    if (!node) return false;

    if (depth == (int)word.size()) {
        if (!node->is_end) return false;  // Word not found
        node->is_end = false;
        return node->children.empty();  // Safe to delete if no children
    }

    char c = word[depth];
    if (!node->children.count(c)) return false;

    bool should_delete = remove(node->children[c], word, depth + 1);

    if (should_delete) {
        delete node->children[c];
        node->children.erase(c);
        return !node->is_end && node->children.empty();
    }
    return false;
}
```

*Key insight:* Post-order traversal. Delete leaf nodes bottom-up only if they have no children and aren't end-of-word for another string.

== Autocomplete (Prefix Search)

*Problem:* Given a prefix, return all words in the trie that start with it.

```cpp
void collect_words(TrieNode* node, string& current, vector<string>& results) {
    if (!node) return;
    if (node->is_end) results.push_back(current);

    for (auto& [ch, child] : node->children) {
        current.push_back(ch);
        collect_words(child, current, results);
        current.pop_back();
    }
}

vector<string> autocomplete(Trie& trie, const string& prefix) {
    TrieNode* curr = trie.root;
    for (char c : prefix) {
        if (!curr->children.count(c)) return {};
        curr = curr->children[c];
    }

    vector<string> results;
    string current = prefix;
    collect_words(curr, current, results);
    return results;
}
```

*Top-K autocomplete:* Use a priority queue or pre-store frequency at each node. At query time, DFS with pruning to find top-K results by frequency. Google-style: store top-K results at each node for $O(1)$ query but $O(n K)$ space.

== Word Search II

*Problem:* Find all words from word list that exist in 2D board.

*Approach - Trie + Backtracking DFS:* $O(n m 4^L)$ time, $n$ × $m$ = board size, $L$ = word length

```cpp
class TrieNode {
public:
    unordered_map<char, TrieNode*> children;
    string word;  // Store complete word at leaf (avoids rebuilding)
};

class Solution {
    int rows, cols;
    vector<string> result;
    TrieNode* root;

    void dfs(vector<vector<char>>& board, int r, int c, TrieNode* node) {
        if (r < 0 || r >= rows || c < 0 || c >= cols) return;

        char ch = board[r][c];
        if (ch == '#' || !node->children.count(ch)) return;

        node = node->children[ch];

        // Found word
        if (!node->word.empty()) {
            result.push_back(node->word);
            node->word.clear();  // Avoid duplicates
        }

        // Mark visited
        board[r][c] = '#';

        // Explore all 4 directions
        dfs(board, r + 1, c, node);
        dfs(board, r - 1, c, node);
        dfs(board, r, c + 1, node);
        dfs(board, r, c - 1, node);

        // Backtrack
        board[r][c] = ch;
    }

public:
    vector<string> find_words(vector<vector<char>>& board, vector<string>& words) {
        rows = board.size();
        cols = board[0].size();

        // Build trie
        root = new TrieNode();
        for (const string& word : words) {
            TrieNode* curr = root;
            for (char c : word) {
                if (!curr->children.count(c)) {
                    curr->children[c] = new TrieNode();
                }
                curr = curr->children[c];
            }
            curr->word = word;
        }

        // Search from each cell
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                dfs(board, r, c, root);
            }
        }

        return result;
    }
};
```

*Key insight:* Trie prunes search space - stop exploring if prefix doesn't exist in any word. Without trie: must check each word independently = $O(k n m 4^L)$ where k = number of words.

*Optimization - Trie pruning:*
Remove leaf nodes after finding word to avoid revisiting. Reduces trie size during search.

*In-place marking:* Use `board[r][c] = '#'` instead of `unordered_set<pair<int,int>>` for visited tracking. Eliminates set allocation + hash overhead.

*Cache behavior:*
- Board access: depends on DFS path. Random 2D access = cache unfriendly
- Trie traversal: pointer chasing = cache unfriendly
- Combined: each step = $#sym.tilde.op$2 cache misses (board + trie) = $#sym.tilde.op$400 cycles per character

*Memory:* Trie with k words of avg length m: O(km) nodes worst case (no prefix sharing). Shared prefixes reduce to O(total characters).

== Compressed Trie (Radix Tree / Patricia Trie)

*Problem:* Standard trie wastes memory on chains of single-child nodes (e.g., "apple" and "application" share "appl" as 4 separate nodes with one child each).

*Idea:* Merge single-child chains into single edges with string labels.

```
Standard trie for {"apple", "app", "application"}:
  a → p → p → (end:"app") → l → e (end:"apple")
                            ↘ i → c → a → t → i → o → n (end)

Compressed trie:
  "app" (end) → "le" (end:"apple")
              → "lication" (end:"application")
```

```cpp
struct RadixNode {
    string edge_label;
    unordered_map<char, RadixNode*> children;
    bool is_end = false;

    RadixNode(const string& label = "") : edge_label(label) {}
};

void insert(RadixNode* root, const string& word) {
    RadixNode* curr = root;
    int i = 0;

    while (i < (int)word.size()) {
        char c = word[i];
        if (!curr->children.count(c)) {
            // No matching edge -- create new leaf
            RadixNode* leaf = new RadixNode(word.substr(i));
            leaf->is_end = true;
            curr->children[c] = leaf;
            return;
        }

        RadixNode* child = curr->children[c];
        const string& label = child->edge_label;
        int j = 0;

        // Match as much of the edge label as possible
        while (j < (int)label.size() && i < (int)word.size() && label[j] == word[i]) {
            j++;
            i++;
        }

        if (j == (int)label.size()) {
            // Consumed entire edge -- continue to child
            curr = child;
        } else {
            // Split edge at mismatch point
            RadixNode* split = new RadixNode(label.substr(0, j));
            child->edge_label = label.substr(j);
            split->children[label[j]] = child;
            curr->children[c] = split;

            if (i == (int)word.size()) {
                split->is_end = true;
            } else {
                RadixNode* leaf = new RadixNode(word.substr(i));
                leaf->is_end = true;
                split->children[word[i]] = leaf;
            }
            return;
        }
    }
    curr->is_end = true;
}
```

*Memory savings:* For English dictionary ($#sym.tilde.op$170K words), standard trie uses $#sym.tilde.op$50MB; radix trie uses $#sym.tilde.op$5MB. 10x reduction from edge merging.

*Tradeoff:* Insertion/deletion more complex (edge splitting/merging). Search is same $O(m)$ but with better constants (fewer pointer chases).

== Trie vs HashMap

#table(
  columns: (auto, auto, auto),
  [*Operation*], [*Trie*], [*HashMap*],
  [Exact lookup], [$O(m)$], [$O(m)$ avg, $O(n m)$ worst],
  [Prefix search], [$O(m + k)$, $k$ = results], [Not supported],
  [Sorted iteration], [DFS = lexicographic order], [Not supported],
  [Space (n words, avg len m)], [$O(n m)$ worst, shared prefixes help], [$O(n m)$],
  [Cache behavior], [Poor (pointer chasing)], [Better (contiguous buckets)],
  [Insert], [$O(m)$], [$O(m)$ amortized],
  [Delete], [$O(m)$], [$O(m)$ amortized],
)

*Use trie when:* Prefix queries needed, lexicographic ordering required, or alphabet is small.

*Use HashMap when:* Only exact lookups needed, cache performance matters, or dataset is small.

== Applications

*IP routing (longest prefix match):* Routers store IP prefixes in a trie. Packet forwarding = trie traversal on destination IP bits. Binary trie on 32-bit IPv4 addresses: max 32 levels.

*Spell checking:* Insert dictionary into trie. For each word in text, search with edit distance tolerance (DFS with allowed mismatches/insertions/deletions).

*T9 predictive text:* Map phone digits to characters. Trie keyed by digit sequences, values are words. "4663" maps to "gone", "good", "home", etc.

== Complexity Reference

#table(
  columns: (auto, auto, auto),
  [*Operation*], [*Time*], [*Space*],
  [Insert], [$O(m)$], [$O(m)$ new nodes worst case],
  [Search], [$O(m)$], [$O(1)$],
  [Delete], [$O(m)$], [$O(1)$],
  [Prefix search (all matches)], [$O(m + k)$, $k$ = results], [$O(k)$ for result storage],
  [Build trie ($n$ words)], [$O(n m)$], [$O(n m)$ worst, less with shared prefixes],
  [Word Search II], [$O(r c dot 4^L)$], [$O(n m)$ trie + $O(L)$ recursion],
)

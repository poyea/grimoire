= Tries

*Memory overhead warning:* Naive trie = 26 pointers per node = 208 bytes (64-bit). Empty children waste space. Typical English word datasets: 70-95% of pointers are null due to sparse branching.

== Implement Trie (Prefix Tree)

*Problem:* Implement a trie with insert, search, and startsWith operations.

*Approach - HashMap-based Trie:* $O(m)$ for all operations where m is word/prefix length

```cpp
class TrieNode {
public:
    unordered_map<char, TrieNode*> children;  // Sparse representation
    bool isEnd = false;
};

class Trie {
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    void insert(string word) {
        TrieNode* curr = root;
        for (char c : word) {
            if (!curr->children.count(c)) {
                curr->children[c] = new TrieNode();
            }
            curr = curr->children[c];
        }
        curr->isEnd = true;
    }

    bool search(string word) {
        TrieNode* curr = root;
        for (char c : word) {
            if (!curr->children.count(c)) return false;
            curr = curr->children[c];
        }
        return curr->isEnd;
    }

    bool startsWith(string prefix) {
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
    bool isEnd = false;
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
- Word length m = m pointer chases = m × $#sym.tilde.op$200 cycles worst case
- `unordered_map` adds hash computation + bucket lookup overhead

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
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
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

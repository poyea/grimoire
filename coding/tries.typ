= Tries

== Implement Trie (Prefix Tree)

*Problem:* Implement a trie with insert, search, and startsWith operations.

*Approach - Trie Data Structure:* $O(m)$ for all operations where m is word/prefix length

*`TrieNode class:`*
- `self.children = {}` - dictionary of child nodes
- `self.isEnd = False` - marks end of word

*`__init__()`:*
- `self.root = TrieNode()`

*`insert(word)`:*
- `curr = self.root`
- For each char in word:
  + If `char not in curr.children`: create new node
  + Move to child: `curr = curr.children[char]`
- Mark end: `curr.isEnd = True`

*`search(word)`:*
- `curr = self.root`
- For each char in word:
  + If `char not in curr.children`: return False
  + Move to child: `curr = curr.children[char]`
- Return `curr.isEnd`

*`startsWith(prefix):`*
- Same as search but return True at end (don't check isEnd)

== Design Add and Search Words Data Structure

*Problem:* Design data structure to add words and search with wildcards ('.').

*Approach - Trie with DFS for Wildcards:* $O(m)$ insert, $O(26^m)$ search worst case

*addWord(word):* Same as Trie insert

*search(word):*
- Define `dfs(i, curr)`:
  + For j in range(i, len(word)):
    - `char = word[j]`
    - If `char == '.'`: wildcard
      + Try all children: if any `dfs(j+1, child)` returns True, return True
      + Return False if none match
    - Else: normal char
      + If `char not in curr.children`: return False
      + Move to child: `curr = curr.children[char]`
  + Return `curr.isEnd`
- Return `dfs(0, self.root)`

*Key insight:* Use DFS backtracking to try all possibilities for wildcard character.

== Word Search II

*Problem:* Find all words from word list that exist in 2D board.

*Approach - Trie + Backtracking DFS:* $O(n #sym.times m #sym.times 4^L)$ time
- Build trie from words list
- Create TrieNode class with addWord method
- Initialize `root = TrieNode()`, add all words to trie
- Get `rows`, `cols`, create `result = set()`, `visited = set()`
- Define `dfs(row, col, trieNode, currWord)`:
  + If out of bounds, visited, or `board[row][col] not in trieNode.children`: return
  + Add to visited: `visited.add((row, col))`
  + Move in trie: `trieNode = trieNode.children[board[row][col]]`
  + Build word: `currWord += board[row][col]`
  + If `trieNode.isEnd`: `result.add(currWord)`
  + Try all 4 directions with updated trieNode
  + Backtrack: `visited.remove((row, col))`
- For each cell: call `dfs(r, c, root, "")`
- Return `list(result)`

*Key insight:* Trie prunes search space - stop exploring if prefix not in any word.

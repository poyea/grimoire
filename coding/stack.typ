= Stack

== Valid Parentheses

*Problem:* Determine if string of parentheses is valid (properly opened and closed in correct order).

*Approach - Stack:* $O(n)$ time, $O(n)$ space
- Create hashmap: `brackets = {')': '(', ']': '[', '}': '{'}`
- Initialize empty stack: `stack = []`
- For each character in s:
  + If closing bracket (in hashmap):
    - If stack empty or `stack.pop() != brackets[char]`: return False
  + Else (opening bracket): `stack.append(char)`
- Return `len(stack) == 0`

*Key Python concepts:*
- `.pop()` - removes and returns last element from list
- Stack operations: append (push), pop

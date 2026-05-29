= Parsing

The parser turns a flat token stream into a tree. That tree — the *abstract syntax tree* (AST) — makes the recursive structure of the source explicit: an `if` node has a condition subtree, a then-branch subtree, and an optional else-branch subtree. Everything downstream (type checking, IR generation, optimizers) walks trees, not token streams.

Parsing is the pushdown automaton in practice: the call stack of a recursive-descent parser *is* the PDA stack. Understanding parsing algorithms means understanding what kind of grammar they accept, what conflicts they cannot resolve, and what the trade-offs are for your use case.

_See also: Pushdown Automata and Context-Free Grammars for context-free grammars, pushdown automata, and the Chomsky normal form._

== Recursive Descent

Recursive descent is the most direct translation from grammar to code. For each non-terminal, write a function that consumes tokens and returns a subtree. The grammar below:

```ebnf
stmt    := 'if' expr 'then' stmt ('else' stmt)?
         | 'let' IDENT '=' expr ';'
         | expr ';'
expr    := IDENT | INTEGER | expr '+' expr | ...
```

becomes one C++ function per non-terminal. The key insight: *the call stack represents the parse stack*. When `parse_stmt` calls `parse_expr`, the current position in `stmt`'s rule is saved on the CPU stack.

=== FIRST and FOLLOW Sets

To choose which production to use for a non-terminal without backtracking, the parser needs to know: given the next token, which alternative can possibly start?

- *FIRST(alpha)*: the set of terminal tokens that can begin a string derived from `alpha`. If `alpha` can derive the empty string, add epsilon to FIRST(alpha).
- *FOLLOW(A)*: the set of terminals that can appear immediately after non-terminal `A` in any sentential form. Used to decide when to apply an epsilon production.

A grammar is *LL(1)* if, for every non-terminal and every pair of alternatives, the FIRST sets are disjoint (and if one alternative can derive epsilon, its FIRST set is disjoint from FOLLOW of the non-terminal). LL(1) grammars can be parsed predictively with one token of lookahead.

=== Grammar Transformations for LL(1)

Two common obstacles:

*Left recursion* (`expr := expr '+' term`) makes a recursive-descent parser loop infinitely. Eliminate it by rewriting:

```ebnf
expr      := term expr_rest
expr_rest := '+' term expr_rest | epsilon
```

*Left factoring* handles alternatives sharing a common prefix:

```ebnf
stmt := 'if' expr 'then' stmt
      | 'if' expr 'then' stmt 'else' stmt
```

Factor into:

```ebnf
stmt      := 'if' expr 'then' stmt else_part
else_part := 'else' stmt | epsilon
```

These transformations are mechanical and can be automated, but they make grammars harder to read. Many real parsers tolerate a small amount of lookahead (LL(2) or LL(3)) to avoid transforming the grammar at all.

== LR Parsing

LR parsers work bottom-up. They maintain a stack of (state, symbol) pairs and a current input token, and take one of two actions:

- *Shift*: push the current token onto the stack, advance to the next token.
- *Reduce*: pop $k$ symbols off the stack (matching the right-hand side of some production), push the left-hand side non-terminal.

The decision is driven by a DFA over *LR items*. An LR(0) item is a production with a dot marking how far the parser has read:

```text
expr -> expr . '+' term     (shift: expecting '+' next)
expr -> expr '+' term .     (reduce: right-hand side complete)
```

Sets of items form the states of the LR automaton. The *goto table* records which state to enter after a shift or reduce; the *action table* records shift/reduce/accept for each (state, lookahead) pair.

=== LR(0), SLR, LALR, LR(1)

The variants differ in how much lookahead they use to resolve reduce/reduce and shift/reduce conflicts:

- *LR(0)*: no lookahead; reduce whenever a completed item is present. Very few grammars are LR(0).
- *SLR(1)*: reduce when the lookahead is in FOLLOW(A) for the reduced non-terminal A. Handles most textbook grammars.
- *LALR(1)*: merge states of the LR(1) automaton that have the same core (ignoring lookahead sets). Same table size as SLR but resolves more conflicts. This is what yacc and Bison produce.
- *LR(1)*: full lookahead sets; largest tables but handles all unambiguous deterministic context-free grammars.

LALR(1) is the practical sweet spot: its tables are the same size as SLR but it handles almost everything LR(1) can, and its conflicts (when they arise) correspond to real grammar ambiguities that need fixing anyway. Bison's default is LALR; requesting `%glr-parser` switches to GLR for genuinely ambiguous cases.

=== Shift/Reduce Conflicts

A conflict means two actions are valid for the same (state, lookahead). The classic example is the *dangling else*:

```ebnf
stmt := 'if' expr 'then' stmt
      | 'if' expr 'then' stmt 'else' stmt
```

After parsing `if expr then stmt`, with lookahead `else`, should the parser reduce the inner `stmt` (no else) or shift `else` (associating it with the inner if)? The convention — and Bison's default — is to *shift*, binding `else` to the nearest `if`. This is usually correct.

Bison reports all conflicts. Unexplained conflicts in a production grammar are a red flag: they mean the grammar is doing something unintended and the parser will silently pick one interpretation.

== Operator-Precedence and Pratt Parsing

For expression grammars with many levels of operator precedence, recursive descent gets cumbersome: you need one function per precedence level (`parse_add`, `parse_mul`, `parse_unary`, `parse_primary`). With 12 precedence levels (as in C) this is tedious but manageable.

*Pratt parsing* (top-down operator precedence, Pratt 1973) is the practitioner's preferred technique. Each token kind carries two numbers:

- *nud* (null denotation): how to parse this token when it appears at the start of an expression (prefix position). A number nud returns a literal node. A `(` nud parses a sub-expression and expects `)`.
- *led* (left denotation) and *lbp* (left binding power): how to parse this token when it appears between two expressions (infix position). `+` has lbp 10; `*` has lbp 20. Right-associative operators pass `lbp-1` as the right-binding power.

The main loop:

```cpp
// binding powers
static int bp(TokenKind k) {
    switch (k) {
    case TokenKind::Plus:  case TokenKind::Minus: return 10;
    case TokenKind::Star:  case TokenKind::Slash: return 20;
    case TokenKind::Eq:    case TokenKind::NotEq: return 5;
    case TokenKind::Lt:    case TokenKind::Gt:
    case TokenKind::LtEq:  case TokenKind::GtEq: return 7;
    default: return 0;
    }
}

AstNodePtr parse_expr(Parser& p, int min_bp = 0) {
    auto left = parse_prefix(p);          // nud
    while (bp(p.peek()) > min_bp) {
        auto op = p.consume();            // led
        auto right = parse_expr(p, bp(op.kind));
        left = make_binop(op.kind, std::move(left), std::move(right));
    }
    return left;
}
```

This 10-line kernel handles arbitrary precedence hierarchies, left and right associativity, prefix operators, and postfix operators (by consuming the token and returning immediately in the led step with no right operand). Pratt parsing is used in rustc's expression parser, in V8's early front end, and in most hand-written parsers in production systems.

== PEG and Packrat Parsing

*Parsing Expression Grammars* (Ford 2004) replace the alternation of CFGs with *ordered choice*: `e1 / e2` means try `e1` first; only if it fails, try `e2`. This eliminates ambiguity by definition — PEGs describe deterministic recursive-descent parsers directly as grammars.

The cost: a PEG is not a notation for a language, it is a notation for a *parser*. Two PEGs that look similar may recognize different languages, and the difference can be non-obvious. The dangling-else problem disappears (ordered choice makes the greedy interpretation the only one), but subtle errors can lurk in rules where the first alternative is a prefix of a valid second alternative.

*Packrat parsing* memoizes every call to every rule at every input position, guaranteeing $O(n)$ parse time at the cost of $O(n dot |G|)$ memory where $|G|$ is the grammar size. For most practical grammars this is acceptable. PEG libraries (PEGTL in C++, pest in Rust, lpeg in Lua) are widely used for ad-hoc parsing tasks where grammar-engineering effort matters more than memory.

== GLR Parsing

Generalized LR (GLR, Tomita 1987) extends LR by forking the parse state whenever a conflict is encountered, exploring all possibilities in parallel, and merging branches that produce the same subtree. The result is an $O(n^3)$ parser (worst case) that handles all context-free grammars, including ambiguous ones.

GLR is the right tool for natural-language grammars or legacy languages (COBOL, some dialects of C++) where the grammar is genuinely ambiguous and the ambiguity is resolved by semantic context. For new language design, ambiguous grammars are a code smell; prefer LALR or Pratt.

== Error Recovery

Three strategies, commonly combined:

- *Panic mode*: on error, discard tokens until a synchronization token (`}`, `;`, keyword) is found, then resume parsing. Fast and simple; misses errors inside the discarded region.
- *Error productions*: add explicit error alternatives to the grammar (e.g., `stmt := error ';'`). Bison's `error` pseudo-token implements this. Allows the parser to recover within a construct rather than abandoning it entirely.
- *Sync sets*: each non-terminal carries a set of tokens that delimit its region; on error, skip until one is found. Similar to panic mode but scoped per non-terminal.

Modern production compilers layer all three. rustc's parser, for example, attempts to continue after most errors, accumulates diagnostics, and emits them all at the end. clang does similarly. The goal: a single compilation reports as many independent errors as possible, not just the first one.

== C++ Recursive-Descent Parser

The following parser handles the grammar:

```ebnf
expr   := term   (('+'|'-') term)*
term   := factor (('*'|'/') factor)*
factor := NUMBER | '(' expr ')'
```

It builds a simple AST using `std::variant`.

```cpp
#include <memory>
#include <variant>
#include <vector>
#include <stdexcept>
#include <string>
#include <cassert>

// forward declare Token and TokenKind from lexing chapter
// TokenKind: Integer, Plus, Minus, Star, Slash, LParen, RParen, Eof

// --- AST node types ---
struct NumberNode { int value; };

struct BinopNode {
    char op;
    std::unique_ptr<struct AstNode> left;
    std::unique_ptr<struct AstNode> right;
};

struct AstNode : std::variant<NumberNode, BinopNode> {
    using variant::variant;
};
using AstPtr = std::unique_ptr<AstNode>;

// --- simple token stream ---
struct TokenStream {
    const std::vector<Token>& tokens;
    size_t pos = 0;

    Token peek() const {
        if (pos < tokens.size()) return tokens[pos];
        return {TokenKind::Eof, "", 0, 0};
    }
    Token consume() {
        assert(pos < tokens.size());
        return tokens[pos++];
    }
    Token expect(TokenKind k) {
        Token t = consume();
        if (t.kind != k)
            throw std::runtime_error("unexpected token: " + t.lexeme);
        return t;
    }
};

// --- parser ---
AstPtr parse_expr(TokenStream& ts);
AstPtr parse_term(TokenStream& ts);

AstPtr parse_factor(TokenStream& ts) {
    Token t = ts.peek();
    if (t.kind == TokenKind::Integer) {
        ts.consume();
        return std::make_unique<AstNode>(NumberNode{std::stoi(t.lexeme)});
    }
    if (t.kind == TokenKind::LParen) {
        ts.consume();
        AstPtr inner = parse_expr(ts);
        ts.expect(TokenKind::RParen);
        return inner;
    }
    throw std::runtime_error("expected number or '(', got: " + t.lexeme);
}

AstPtr parse_term(TokenStream& ts) {
    AstPtr left = parse_factor(ts);
    while (ts.peek().kind == TokenKind::Star ||
           ts.peek().kind == TokenKind::Slash) {
        char op = (char)ts.consume().lexeme[0];
        AstPtr right = parse_factor(ts);
        left = std::make_unique<AstNode>(BinopNode{op,
                   std::move(left), std::move(right)});
    }
    return left;
}

AstPtr parse_expr(TokenStream& ts) {
    AstPtr left = parse_term(ts);
    while (ts.peek().kind == TokenKind::Plus ||
           ts.peek().kind == TokenKind::Minus) {
        char op = (char)ts.consume().lexeme[0];
        AstPtr right = parse_term(ts);
        left = std::make_unique<AstNode>(BinopNode{op,
                   std::move(left), std::move(right)});
    }
    return left;
}
```

The parser is 55 lines. Operator precedence is encoded structurally: `parse_expr` calls `parse_term`, which calls `parse_factor`, so `*`/`/` bind tighter than `+`/`-` by construction.

== AST Design

The two dominant C++ patterns for AST nodes:

*`std::variant` (sum type):* All node kinds are listed in the variant. Adding a new node kind is a one-line change to the variant declaration; every `std::visit` that does not handle the new kind is a compile error. Pattern matching via `std::visit` is verbose but exhaustive. Good for small, stable ASTs.

*Class hierarchy:* An abstract `AstNode` base with virtual methods (`accept` for visitors, or direct virtuals for common operations). Adding a new kind requires a new class file; forgetting to override a pure virtual is a link error. More boilerplate, but nodes can carry methods and are easier to extend with per-node data. Good for large, evolving ASTs. LLVM, clang, and GCC all use class hierarchies.

A hybrid: use `std::variant` for expression nodes (frequently visited, closed set) and a class hierarchy for statements (open, extensible set).

== Parsing Algorithm Comparison

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Algorithm*], [*Lookahead*], [*Ambiguity*], [*Error recovery*], [*Typical use*],
  [LL(1)],   [1 token],        [None],          [Moderate],  [ANTLR, hand-written DSLs],
  [LL(k)],   [k tokens],       [None],          [Good],      [ANTLR 4],
  [LALR(1)], [1 token],        [Conflicts reported], [Good], [Bison, Menhir, yacc],
  [LR(1)],   [1 token],        [Same as LALR],  [Good],      [hyacc, rare],
  [PEG],     [Unlimited (memo)],[None by construction], [Poor], [PEGTL, pest, lpeg],
  [GLR],     [Unlimited],      [Handles true ambiguity], [Complex], [Bison --glr, Earley],
  [Pratt],   [1 token],        [Exprs: none; stmts: manual], [Excellent], [Hand-written (rustc)],
)

_See also: Advanced String Algorithms (Coding volume) for related string-matching structures._

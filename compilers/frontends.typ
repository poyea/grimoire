#import "../template.typ": xref

= Frontends <frontends>

A production compiler frontend is judged not by what it accepts but by what it does when input is wrong. This chapter is about lexing, parsing, error recovery, and incremental reparsing as engineered artifacts — the theory of context-free grammars and LR/LL parsing lives in the programming-languages volume.

*See also:* _programming-languages/lexing_, _programming-languages/parsing_, _programming-languages/pushdown-cfg_, #xref("compilers", "ir-design", label: "ir-design")

== Hand-Written vs Generated Parsers

The major compilers all hand-write their frontends: Clang, GCC, Roslyn, rustc, Swift, V8. Generator-based frontends (yacc/bison, ANTLR, tree-sitter) dominate everywhere else — DSL toolchains, IDEs, refactoring tools.

#table(
  columns: 4,
  [*Compiler*], [*Strategy*], [*Reason*], [*Error recovery*],
  [Clang], [recursive descent], [diagnostics, fidelity], [hand-coded heuristics],
  [GCC], [recursive descent (since 4.1)], [diagnostics], [token re-sync],
  [Roslyn], [recursive descent + red/green], [incremental, IDE], [synthetic tokens],
  [rustc], [recursive descent], [diagnostics], [ML-trained suggestions],
  [tree-sitter], [GLR (Bison-style)], [incremental, language-agnostic], [error nodes],
)

The recurring lesson: the parser is the cheapest stage but defines the user experience of every error message downstream. A generated $"LALR"(1)$ parser excels at acceptance but is mediocre at telling you _why_ your code is wrong.

== A Production-Grade Lexer

Real lexers do more than tokenize: they track trivia (whitespace, comments) for refactoring, lazily decode strings, and produce byte-accurate spans for IDEs.

```cpp
struct SourceLoc { uint32_t file_id; uint32_t offset; };
struct Span { SourceLoc begin, end; };

enum class TokenKind : uint8_t {
    Identifier, IntLit, FloatLit, StringLit,
    LParen, RParen, LBrace, RBrace, Semicolon, Comma,
    Plus, Minus, Star, Slash, Eq, EqEq, Bang, BangEq,
    Kw_fn, Kw_let, Kw_if, Kw_else, Kw_while, Kw_return,
    Eof, Error,
};

struct Token {
    TokenKind kind;
    Span      span;
    uint32_t  trivia_before;   // index into trivia table
    union { uint64_t int_val; double float_val; uint32_t str_id; };
};

class Lexer {
    const char* src_;
    size_t      pos_, end_;
    uint32_t    file_id_;
    std::vector<Trivia> trivia_;

public:
    Token next() {
        uint32_t trivia_start = trivia_.size();
        skip_trivia();
        SourceLoc begin{file_id_, (uint32_t)pos_};

        if (pos_ >= end_) return {TokenKind::Eof, {begin, begin}, trivia_start, {}};

        char c = src_[pos_];
        if (is_ident_start(c)) return ident_or_keyword(begin, trivia_start);
        if (is_digit(c))       return number(begin, trivia_start);
        if (c == '"')          return string_literal(begin, trivia_start);
        return punctuation(begin, trivia_start);
    }

private:
    void skip_trivia() {
        while (pos_ < end_) {
            char c = src_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { ++pos_; }
            else if (c == '/' && pos_+1 < end_ && src_[pos_+1] == '/') skip_line_comment();
            else if (c == '/' && pos_+1 < end_ && src_[pos_+1] == '*') skip_block_comment();
            else break;
        }
    }

    Token ident_or_keyword(SourceLoc begin, uint32_t trivia) {
        size_t start = pos_;
        while (pos_ < end_ && is_ident_cont(src_[pos_])) ++pos_;
        std::string_view text(src_ + start, pos_ - start);
        TokenKind kw = lookup_keyword(text);   // perfect hash
        return {kw, {begin, {file_id_, (uint32_t)pos_}}, trivia, {}};
    }
};
```

*Perfect hashing for keywords.* `gperf` and equivalents produce $O(1)$ keyword recognition. Clang's `IdentifierInfo` interns every identifier into a hash table and stores `TokenKind` on the entry, so keyword vs identifier is one pointer compare.

*UTF-8.* Identifiers in Rust, Swift, Java accept Unicode (UAX #31). The lexer decodes one code point at a time and consults a precomputed bitmap of `XID_Start` / `XID_Continue` classes.

== Recursive Descent in Practice

Precedence climbing (Pratt parsing) is the dominant technique in modern frontends because it handles operator precedence in one function and extends naturally to mixfix and postfix operators.

```cpp
Expr* parse_expr(int min_prec) {
    Expr* lhs = parse_unary();
    while (true) {
        Token op = peek();
        int prec = binop_prec(op.kind);
        if (prec < min_prec) break;
        consume();
        Expr* rhs = parse_expr(right_assoc(op.kind) ? prec : prec + 1);
        lhs = new BinaryOp(op.kind, lhs, rhs);
    }
    return lhs;
}

Expr* parse_unary() {
    Token t = peek();
    if (t.kind == TokenKind::Minus || t.kind == TokenKind::Bang) {
        consume();
        return new UnaryOp(t.kind, parse_unary());
    }
    return parse_postfix(parse_primary());
}
```

*Precedence table* lives outside the grammar — easy to extend, easy to dump for documentation.

== Error Recovery

Three strategies dominate:

+ *Panic mode.* On error, discard tokens until a synchronization point (`;`, `}`, statement-start keyword). Cheap, loses subtle errors.
+ *Phrase-level repair.* Insert/delete a token to make local parse succeed. Burke-Fisher's algorithm tries $k$ token insertions/deletions within a sliding window.
+ *Error productions.* Augment the grammar with rules for common mistakes (`x = y +;` $arrow$ "missing operand"). Bison and ANTLR support these directly.

Clang's error recovery is the gold standard for diagnostics: it tracks a _correction candidate_ list, will speculatively re-parse with an inserted `;` to confirm the fix, and emits a fix-it hint.

```cpp
// Clang-style fix-it
DiagnosticBuilder D = diag(loc, diag::err_expected_semi);
D << FixItHint::CreateInsertion(loc, ";");
```

*Tail-tolerant parsing* in rustc lets a missing `,` in an argument list be reported once, with the rest of the call still parsed.

== Concrete vs Abstract Syntax Trees

Compilers care about the $"AST"$; IDEs and formatters need the $"CST"$ (every token, every trivia byte, exact spans). Roslyn's *red-green tree* is the canonical solution:

- *Green tree* (immutable, content-addressed): just kinds, widths, child arrays. No parent pointers, no absolute positions. Shareable across edits.
- *Red tree* (lazily allocated facade): wraps a green node with parent pointer and absolute offset. Recomputed on demand.

```csharp
class GreenNode {
    public SyntaxKind Kind;
    public int FullWidth;          // bytes including trivia
    public GreenNode[] Children;
}

class RedNode {
    private GreenNode green;
    private RedNode  parent;
    public  int      absoluteOffset;
    // children allocated on first access
}
```

An edit to one character invalidates only the green nodes on the path from root to the edit; everything else is reused. Tree-sitter independently invented the same idea for its incremental GLR parser.

== Incremental Parsing

The goal: re-parse a 100k-line file in microseconds after a single-character edit, so an IDE can run analyses on every keystroke.

*Tree-sitter algorithm:*

+ Diff old and new source to find a contiguous edit range $[a, b)$.
+ Walk the old tree, marking nodes whose span intersects the edit as _changed_.
+ Resume the GLR parser at the leftmost changed node, reusing unchanged subtrees as opaque skip tokens.
+ Stop early once the parser's state vector matches the old tree's state at a corresponding boundary.

Result: edits typically reparse $O(log n)$ work proportional to tree depth, not file size.

```rust
// Sketch (real tree-sitter is C, ~20k LOC)
fn reparse(old: &Tree, new_src: &str, edit: Edit) -> Tree {
    let mut cursor = old.walk();
    let mut parser = Parser::resume_state(old.root().state());
    cursor.goto_first_intersecting(edit.range);
    while let Some(node) = cursor.next() {
        if node.span().ends_before(edit.start) {
            parser.reuse_subtree(node);
        } else if node.span().starts_after(edit.end) {
            parser.reuse_subtree(node.shifted(edit.delta));
            break;
        } else {
            parser.reparse_range(node.span().expand(edit));
        }
    }
    parser.finish()
}
```

== Lexer Modes and Context-Sensitivity

Some languages have token classes that depend on parser context: C++ templates (`>>` vs `> >`), JavaScript regex vs divide (`/`), Rust raw string `r#"..."#`. Hand-written parsers handle these by switching lexer modes:

```cpp
Token next_token(LexerMode mode) {
    switch (mode) {
        case LexerMode::Normal:    return lex_normal();
        case LexerMode::Template:  return lex_template_body();
        case LexerMode::Regex:     return lex_regex();
    }
}
```

C++'s `>>` problem was solved in C++11 by special-casing in the parser, not the lexer — the parser asks the lexer to split `>>` if the current context is template-argument-list.

== IDE Integration: Salsa and Query-Based Compilation

Modern IDE-grade compilers (rustc, rust-analyzer, Roslyn) treat the compiler as a database. Every input becomes a query whose result is memoized; edits invalidate only the dependent queries.

```rust
// rust-analyzer-style (Salsa)
#[salsa::query_group(ParseDatabaseStorage)]
trait ParseDatabase {
    #[salsa::input]
    fn file_text(&self, id: FileId) -> Arc<String>;

    fn parse(&self, id: FileId) -> Arc<SyntaxNode>;
    fn ast_id_map(&self, id: FileId) -> Arc<AstIdMap>;
    fn resolve(&self, name: NameRef) -> Option<Definition>;
}

fn parse(db: &dyn ParseDatabase, id: FileId) -> Arc<SyntaxNode> {
    let text = db.file_text(id);
    Arc::new(SourceFile::parse(&text).syntax_node())
}
```

Salsa's _firewall queries_ stop invalidation at semantically meaningful boundaries: if you change a function body, only that function's type-check query re-runs, not the whole crate's.

== Diagnostics as a First-Class Output

Rust's error format set the bar: source snippets with caret ranges, labeled spans, fix-it hints, machine-readable JSON. The core data structure is unsurprising; the engineering is in placement, deduplication, and color.

```rust
struct Diagnostic {
    level: Level,                    // Error, Warning, Help, Note
    message: String,
    code: Option<DiagnosticCode>,    // E0308 etc.
    spans: Vec<LabeledSpan>,
    children: Vec<SubDiagnostic>,    // notes/helps
    suggestions: Vec<Suggestion>,    // fix-its with applicability
}

enum Applicability {
    MachineApplicable,   // safe to auto-apply
    MaybeIncorrect,      // suggest, don't auto-apply
    HasPlaceholders,
    Unspecified,
}
```

== Comparison: Hand-Written vs Generated

#table(
  columns: 3,
  [*Property*], [*Hand-written RD*], [*$"LALR"$ generator*],
  [Acceptance for ambiguous grammar], [easy with backtracking], [conflicts, needs $"GLR"$],
  [Error messages], [excellent], [generic],
  [Debugging], [step into], [opaque tables],
  [Maintenance], [verbose], [terse grammar],
  [Performance], [fast (cache-friendly)], [table-lookup, branchy],
)

== Further Reading

Grune, D., Jacobs, C. (2008). _Parsing Techniques: A Practical Guide_, 2nd ed. Springer.

Pratt, V. (1973). "Top Down Operator Precedence." POPL.

Burke, M., Fisher, G. (1987). "A Practical Method for LR and LL Syntactic Error Diagnosis and Recovery." TOPLAS.

Brunsfeld, M. Tree-sitter design notes. #link("https://tree-sitter.github.io").

Lippert, E. "Persistence, Facades, and Roslyn's Red-Green Trees." Microsoft Dev Blog.

Roslyn Project. _Roslyn Architecture Overview._ #link("https://github.com/dotnet/roslyn").

Matsakis, N. "Salsa: Incremental Computation for Rust." 2018.

Clang Project. _Internals Manual: Diagnostics._ #link("https://clang.llvm.org/docs/InternalsManual.html").

= Lexing

A compiler or interpreter sees its input as a flat stream of bytes. The lexer — also called a scanner or tokenizer — imposes the first layer of structure: it converts that stream into a sequence of *tokens*, each carrying a type (integer literal, identifier, keyword, operator) and the raw text that produced it. Everything downstream works on tokens, never on individual characters.

This separation of concerns is not just aesthetic. It isolates encoding complexity (UTF-8, byte-order marks, line endings) in one place. It lets the parser reason about a much smaller alphabet — a few dozen token types rather than 128+ ASCII codes. And it removes ambiguity that would otherwise force the parser to carry impossible amounts of lookahead: the string `<=` is a single LESS_EQ token before the parser ever sees it.

_See also: Regular Languages and Finite Automata for regular languages and DFA theory. The present chapter is that theory made operational._

== Token Classes and Regex Specifications

Each token class is described by a regular expression over the input alphabet. Typical classes for a C-like language:

```text
INTEGER   [0-9]+
FLOAT     [0-9]+ \. [0-9]* ([eE][+-]?[0-9]+)?
IDENT     [a-zA-Z_][a-zA-Z0-9_]*
STRING    " ([^"\\] | \\.)* "
PLUS      \+
MINUS     -
STAR      \*
SLASH     /
EQ        ==
ASSIGN    =
LPAREN    \(
RPAREN    \)
SEMI      ;
WS        [ \t\r\n]+   (discard)
```

Keywords (`if`, `else`, `let`, `return`, ...) are a subset of `IDENT`. The standard approach is to match them as identifiers first, then look up the lexeme in a keyword table — this avoids adding one regex alternative per keyword and keeps the DFA small.

String literals with escape sequences deserve attention. The pattern `([^"\\] | \\.)*` accepts any non-quote, non-backslash character, or a backslash followed by any character. The escape *interpretation* (`\\n` -> newline, `\\t` -> tab, ...) is a separate pass over the raw lexeme, not part of the DFA itself.

== From Regex to DFA

A flex-style tool takes the list of (regex, action) pairs and compiles them into a single DFA:

1. *Thompson's construction*: convert each regex to an NFA fragment with epsilon transitions.
2. *Union*: connect a fresh start state to each fragment's start via epsilon.
3. *Subset construction (powerset)* : convert the combined NFA to a DFA; each DFA state is a set of NFA states.
4. *Minimization*: merge DFA states with identical futures (Hopcroft's algorithm, $O(n log n)$). This keeps the transition table small enough to fit in L1 cache.

The accepting DFA states each carry a *priority* — the index of the first regex rule that caused this state to accept. This is what resolves the keyword-vs-identifier ambiguity: if both the `if`-keyword rule and the IDENT rule would accept at the same state, lower-priority (earlier-listed) rule wins.

In practice, lexer generators emit a 2-D transition table `table[state][char_class]` where `char_class` collapses the 256 byte values into equivalence classes (typically 20-40 classes). The inner loop is:

```text
state = START
while input not empty:
    c = next char
    state = table[state][eq_class[c]]
    if state == DEAD: emit token, reset
```

The entire table fits in a few kilobytes. On a modern x86 core this loop sustains 100-500 MB/s of input throughput — the bottleneck is memory bandwidth, not computation.

== Maximal Munch and Longest-Match

The *maximal munch* rule (also called *longest match*): the lexer always produces the longest possible token starting at the current position. When the DFA reaches a non-accepting state and the last accepting state was $k$ characters back, it emits the token up to position $k$ and restarts from there.

This is why `count++` scans as `count`, `++`, not `count`, `+`, `+`. It is also why `<=` is one token and why `-->` in C is the famous "goes to" joke.

Implementation: the lexer maintains a *mark* — the input position and DFA state of the last accept seen during the current scan. Whenever it enters an accepting state it updates the mark. When it enters a dead state it backs up to the mark, emits, and restarts.

== Lookahead

Most token boundaries are determined by maximal munch alone. But some tokens require *bounded lookahead*: the lexer must peek at the next character without consuming it to decide what to emit.

The canonical example: seeing `=`, peek one character ahead. If the next char is `=`, emit `EQ` (two chars consumed). Otherwise emit `ASSIGN` (one char consumed, next char returned to stream). This is *1-character lookahead* and handles the vast majority of cases in practice.

A second case: FORTRAN's infamous `DO 10 I = 1,10` vs `DO 10 I = 1.10` — the comma vs period distinguishes a DO loop from an assignment to a variable named `DO10I`. Classic FORTRAN lexers needed unbounded lookahead to resolve this; modern language designers avoid it.

Implementation of bounded lookahead: maintain a small ring buffer (typically 1-4 bytes) in the lexer state. `peek_char()` fills the buffer without advancing the main cursor; `consume_char()` drains it.

== flex and re2c

*flex* (Fast Lexical Analyzer) is the GNU successor to lex. You write a `.l` file with `%%`-separated rules; flex generates a C file implementing the DFA as a switch over states and equivalence-class tables. Its output is competitive with hand-written code for large grammars and handles Unicode input classes when configured with `-8`. The generated file can exceed 10 000 lines for complex grammars, which is fine — you never read it.

*re2c* takes a similar approach but generates more idiomatic C/C++: instead of a monolithic switch, it emits `goto`-based code that compiles to near-optimal machine code. re2c is used in PHP's lexer and in several high-performance network protocol parsers. Its key advantage: the generated code has no function-call overhead per character and avoids the branch mispredictions that plague naive hand-rolled scanners.

*Hand-written switch-table lexers* are warranted in two situations: (1) performance-critical embedded targets where code size matters and a generated 10 KB table is unacceptable; (2) languages with very few token classes where a direct switch is more readable than generated machinery. Both rustc and clang hand-write their lexers. Their custom implementations can exploit language-specific structure (e.g., Rust raw string literals `r###"..."###` with variable delimiter length) that regular expression formalisms handle poorly.

== Source-Position Tracking

Every token should carry its origin: line number, column (in code units), and byte offset from the start of the file. This is the minimum needed to emit useful error messages.

The standard approach: maintain three counters in the lexer — `line`, `col`, and `byte_offset`. On each character consumed: increment `byte_offset`; if the character is `\n`, increment `line` and reset `col` to 1; otherwise increment `col`.

*UTF-8 considerations:* byte offset is unambiguous and cheap to compute. Column in *characters* (Unicode code points) requires decoding multi-byte sequences; column in *grapheme clusters* (what users see as one glyph) requires the Unicode segmentation algorithm. Most compiler error messages report byte offsets or code-unit columns and leave grapheme rendering to the IDE. If the source file declares an encoding other than UTF-8, normalize to UTF-8 at the front door before lexing.

Byte-order marks (U+FEFF at the start of a UTF-8 file) should be consumed silently — they are not part of the source text.

== Error Recovery

When the DFA reaches a dead state and the mark buffer is empty, the lexer has encountered an input sequence that matches no token. Options:

1. *Abort*: emit a fatal error and stop. Simple but makes the compiler useless for finding multiple errors in one pass.
2. *Skip to next whitespace*: consume characters until the next whitespace or known delimiter, then emit a `BAD_TOKEN` carrying the offending text. The parser can be written to tolerate `BAD_TOKEN` and continue parsing, collecting further errors.
3. *Single-character skip*: emit a `BAD_TOKEN` for the single bad character and restart. This recovers faster but can cascade — one bad character may fragment a valid identifier into spurious tokens.

Most production compilers use option 2 for lexer errors and reserve option 1 for truly unrecoverable situations (e.g., a null byte in the middle of a string literal). The key insight: error recovery in the lexer costs nothing if the parser is written to be resilient, and it dramatically improves the user experience for typical typos.

== C++ Lexer Implementation

The following lexer handles integers, identifiers, keywords (`if`, `else`, `let`, `return`), common operators, parentheses, and semicolons. It is a hand-written switch-based scanner representative of what you would write for a small expression language or DSL.

```cpp
#include <cassert>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

enum class TokenKind {
    // literals
    Integer, Ident,
    // keywords
    Kw_if, Kw_else, Kw_let, Kw_return,
    // operators
    Plus, Minus, Star, Slash,
    Eq, Assign, Lt, Gt, LtEq, GtEq, Bang, NotEq,
    // delimiters
    LParen, RParen, LBrace, RBrace, Semi, Comma,
    // special
    Eof, Bad,
};

struct Token {
    TokenKind  kind;
    std::string lexeme;
    int        line;
    int        col;
};

static const std::unordered_map<std::string_view, TokenKind> kKeywords = {
    {"if",     TokenKind::Kw_if},
    {"else",   TokenKind::Kw_else},
    {"let",    TokenKind::Kw_let},
    {"return", TokenKind::Kw_return},
};

class Lexer {
public:
    explicit Lexer(std::string_view src)
        : src_(src), pos_(0), line_(1), col_(1) {}

    Token next_token() {
        skip_whitespace();
        if (pos_ >= src_.size())
            return make_token(TokenKind::Eof, "");

        int tok_line = line_, tok_col = col_;
        char c = peek_char();

        if (is_digit(c))  return read_number(tok_line, tok_col);
        if (is_alpha(c))  return read_ident(tok_line, tok_col);

        consume_char();
        switch (c) {
        case '+': return {TokenKind::Plus,   "+", tok_line, tok_col};
        case '-': return {TokenKind::Minus,  "-", tok_line, tok_col};
        case '*': return {TokenKind::Star,   "*", tok_line, tok_col};
        case '/': return {TokenKind::Slash,  "/", tok_line, tok_col};
        case '(': return {TokenKind::LParen, "(", tok_line, tok_col};
        case ')': return {TokenKind::RParen, ")", tok_line, tok_col};
        case '{': return {TokenKind::LBrace, "{", tok_line, tok_col};
        case '}': return {TokenKind::RBrace, "}", tok_line, tok_col};
        case ';': return {TokenKind::Semi,   ";", tok_line, tok_col};
        case ',': return {TokenKind::Comma,  ",", tok_line, tok_col};
        case '=':
            if (peek_char() == '=') { consume_char();
                return {TokenKind::Eq,    "==", tok_line, tok_col}; }
            return {TokenKind::Assign, "=",  tok_line, tok_col};
        case '!':
            if (peek_char() == '=') { consume_char();
                return {TokenKind::NotEq, "!=", tok_line, tok_col}; }
            return {TokenKind::Bang,  "!",  tok_line, tok_col};
        case '<':
            if (peek_char() == '=') { consume_char();
                return {TokenKind::LtEq, "<=", tok_line, tok_col}; }
            return {TokenKind::Lt,   "<",  tok_line, tok_col};
        case '>':
            if (peek_char() == '=') { consume_char();
                return {TokenKind::GtEq, ">=", tok_line, tok_col}; }
            return {TokenKind::Gt,   ">",  tok_line, tok_col};
        default:
            return {TokenKind::Bad, std::string(1, c), tok_line, tok_col};
        }
    }

private:
    std::string_view src_;
    size_t pos_;
    int line_, col_;

    char peek_char() const {
        return pos_ < src_.size() ? src_[pos_] : '\0';
    }
    char consume_char() {
        char c = src_[pos_++];
        if (c == '\n') { ++line_; col_ = 1; } else { ++col_; }
        return c;
    }
    void skip_whitespace() {
        while (pos_ < src_.size() && is_space(peek_char()))
            consume_char();
    }
    Token read_number(int l, int co) {
        size_t start = pos_;
        while (is_digit(peek_char())) consume_char();
        return {TokenKind::Integer, std::string(src_.substr(start, pos_-start)), l, co};
    }
    Token read_ident(int l, int co) {
        size_t start = pos_;
        while (is_alnum(peek_char())) consume_char();
        std::string lex(src_.substr(start, pos_ - start));
        auto it = kKeywords.find(lex);
        TokenKind kind = (it != kKeywords.end()) ? it->second : TokenKind::Ident;
        return {kind, lex, l, co};
    }
    static bool is_digit(char c) { return c >= '0' && c <= '9'; }
    static bool is_alpha(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    }
    static bool is_alnum(char c) { return is_alpha(c) || is_digit(c); }
    static bool is_space(char c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n';
    }
    Token make_token(TokenKind k, const std::string& lex) {
        return {k, lex, line_, col_};
    }
};
```

The lexer is 90 lines and handles all the features discussed: maximal munch for `==` vs `=`, keyword lookup via hash table, position tracking, and `Bad` emission for unknown characters. Extending it to floats or string literals is mechanical.

== Performance

A well-written DFA lexer — whether generated by flex/re2c or hand-rolled — sustains *100-500 MB/s* throughput on a modern x86 core. The bottleneck is memory bandwidth feeding the transition table, not computation. At 500 MB/s, a 100 000-line source file (roughly 5 MB) lexes in under 10 ms.

*`std::regex` and similar backtracking engines are 10-100x slower* for the same task. The reason is fundamental: POSIX regex semantics require the engine to try all possible matches and return the longest one, which forces NFA simulation or backtracking. For lexer workloads — where the patterns are fixed at compile time and inputs are long — a compiled DFA is the correct choice. Use `<regex>` for ad-hoc string processing; never in a hot lexing loop.

_See also: String Algorithms (Coding volume) for Aho-Corasick and KMP, which answer related questions about multi-pattern search._

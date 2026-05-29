= Build a Compiler

This chapter is the constructive proof of the book's thesis. We build a complete, working compiler for a small language called muC ("micro-C") — from raw source text to executable bytecode running on a stack VM. Every phase corresponds to a concept developed in previous chapters. By the end, the abstraction stack is fully concrete.

_See also: Lexing (ch. 4), Parsing (ch. 5), Type Systems (ch. 7) for the theory behind each phase._

== The muC Language

muC is a small C-like language: integer and boolean values, arithmetic, comparisons, if/else, while loops, function definitions, local variables. Enough to write merge sort or a simple interpreter; small enough that the entire compiler fits in this chapter.

```ebnf
program    := func_def*

func_def   := 'fn' IDENT '(' param_list? ')' '->' type block

param_list := param (',' param)*
param      := IDENT ':' type

type       := 'int' | 'bool'

block      := '{' stmt* '}'

stmt       := 'let' IDENT ':' type '=' expr ';'
            | 'return' expr ';'
            | 'if' '(' expr ')' block ('else' block)?
            | 'while' '(' expr ')' block
            | IDENT '=' expr ';'
            | expr ';'

expr       := expr ('+'|'-'|'*'|'/'|'=='|'!='|'<'|'>'|'<='|'>=') expr
            | '!' expr
            | '-' expr
            | IDENT '(' arg_list? ')'
            | IDENT
            | INTEGER
            | 'true' | 'false'
            | '(' expr ')'

arg_list   := expr (',' expr)*
```

Operator precedence (low to high): `==`, `!=`, `<`, `>`, `<=`, `>=`; then `+`, `-`; then `*`, `/`; then unary `!`, `-`.

== Phase 1: Lexing

We reuse the `Token`, `TokenKind`, and `Lexer` from chapter 4, extended with muC-specific tokens: `Kw_fn`, `Kw_int`, `Kw_bool`, `Kw_true`, `Kw_false`, `Kw_while`, `Arrow` (`->`), `Colon`.

```cpp
// Extensions to TokenKind enum from ch.4:
// Kw_fn, Kw_int, Kw_bool, Kw_true, Kw_false, Kw_while,
// Arrow, Colon, (reuse: Kw_if, Kw_else, Kw_return, Kw_let)

// Extension to kKeywords map:
// {"fn",    TokenKind::Kw_fn},
// {"int",   TokenKind::Kw_int},
// {"bool",  TokenKind::Kw_bool},
// {"true",  TokenKind::Kw_true},
// {"false", TokenKind::Kw_false},
// {"while", TokenKind::Kw_while},

// Additional cases in the switch statement:
// case '-': if peek_char()=='>': consume, return Arrow; else return Minus
// case ':': return Colon
```

The lexer produces a `std::vector<Token>` that the parser consumes. Total lexer code including extensions: ~110 lines.

== Phase 2: Parsing and AST

The AST uses a class hierarchy for statements (open, extensible) and `std::variant` for expressions (closed, frequently visited).

```cpp
#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <optional>

// --- Types ---
enum class TypeKind { Int, Bool };

// --- Expression AST ---
struct IntLit    { int value; };
struct BoolLit   { bool value; };
struct VarExpr   { std::string name; };
struct BinopExpr {
    char op[3];   // "+" "-" "*" "/" "==" "!=" "<" ">" "<=" ">="
    std::unique_ptr<struct Expr> left, right;
};
struct UnaryExpr {
    char op;
    std::unique_ptr<struct Expr> operand;
};
struct CallExpr  {
    std::string func;
    std::vector<std::unique_ptr<struct Expr>> args;
};

struct Expr : std::variant<IntLit, BoolLit, VarExpr,
                           BinopExpr, UnaryExpr, CallExpr> {
    using variant::variant;
    TypeKind inferred_type{TypeKind::Int}; // filled by type checker
};

// --- Statement AST ---
struct Stmt { virtual ~Stmt() = default; };
using StmtPtr = std::unique_ptr<Stmt>;

struct LetStmt   : Stmt {
    std::string name; TypeKind type;
    std::unique_ptr<Expr> init;
};
struct AssignStmt: Stmt {
    std::string name;
    std::unique_ptr<Expr> value;
};
struct ReturnStmt: Stmt { std::unique_ptr<Expr> value; };
struct ExprStmt  : Stmt { std::unique_ptr<Expr> expr; };
struct IfStmt    : Stmt {
    std::unique_ptr<Expr> cond;
    std::vector<StmtPtr> then_body, else_body;
};
struct WhileStmt : Stmt {
    std::unique_ptr<Expr> cond;
    std::vector<StmtPtr> body;
};

// --- Function definition ---
struct Param   { std::string name; TypeKind type; };
struct FuncDef {
    std::string name;
    std::vector<Param> params;
    TypeKind ret_type;
    std::vector<StmtPtr> body;
};
```

The parser uses Pratt parsing for expressions and straightforward recursive descent for statements:

```cpp
class Parser {
    std::vector<Token> tokens_;
    size_t pos_ = 0;

    Token peek() const { return pos_<tokens_.size()?tokens_[pos_]:{TokenKind::Eof,"",0,0}; }
    Token consume()    { return tokens_[pos_++]; }
    Token expect(TokenKind k) {
        auto t = consume();
        if (t.kind != k) throw std::runtime_error("expected token, got: "+t.lexeme);
        return t;
    }
    bool check(TokenKind k) const { return peek().kind == k; }

    TypeKind parse_type() {
        auto t = consume();
        if (t.kind == TokenKind::Kw_int)  return TypeKind::Int;
        if (t.kind == TokenKind::Kw_bool) return TypeKind::Bool;
        throw std::runtime_error("expected type");
    }

    std::unique_ptr<Expr> parse_expr(int min_bp = 0);  // Pratt
    std::unique_ptr<Expr> parse_prefix();

    StmtPtr parse_stmt();
    std::vector<StmtPtr> parse_block();

public:
    explicit Parser(std::vector<Token> toks) : tokens_(std::move(toks)) {}

    FuncDef parse_func() {
        expect(TokenKind::Kw_fn);
        std::string name = consume().lexeme;
        expect(TokenKind::LParen);
        std::vector<Param> params;
        while (!check(TokenKind::RParen)) {
            std::string pname = consume().lexeme;
            expect(TokenKind::Colon);
            TypeKind ptype = parse_type();
            params.push_back({pname, ptype});
            if (check(TokenKind::Comma)) consume();
        }
        expect(TokenKind::RParen);
        expect(TokenKind::Arrow);
        TypeKind ret = parse_type();
        auto body = parse_block();
        return {name, params, ret, std::move(body)};
    }
};
```

The Pratt expression parser follows the same pattern as chapter 5: binding powers for each operator, left-recursive loop consuming operators as long as their binding power exceeds `min_bp`.

== Phase 3: Type Checking

The type checker walks the AST, maintains a symbol table (scope stack of `name -> TypeKind` maps), and decorates each `Expr` node with its inferred type. Errors are accumulated, not thrown immediately.

```cpp
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <vector>

struct TypeError { std::string msg; int line; };

class TypeChecker {
    // function signatures: name -> (param_types, ret_type)
    std::unordered_map<std::string,
        std::pair<std::vector<TypeKind>, TypeKind>> funcs_;
    // scope stack
    std::vector<std::unordered_map<std::string, TypeKind>> scopes_;
    std::vector<TypeError> errors_;
    TypeKind current_ret_{TypeKind::Int};

    void push_scope() { scopes_.push_back({}); }
    void pop_scope()  { scopes_.pop_back(); }
    void define(const std::string& n, TypeKind t) { scopes_.back()[n] = t; }
    TypeKind lookup(const std::string& n) {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto f = it->find(n);
            if (f != it->end()) return f->second;
        }
        throw std::runtime_error("undefined variable: " + n);
    }

    TypeKind check_expr(Expr& e) {
        TypeKind t = std::visit([&](auto& node) -> TypeKind {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, IntLit>)  return TypeKind::Int;
            if constexpr (std::is_same_v<T, BoolLit>) return TypeKind::Bool;
            if constexpr (std::is_same_v<T, VarExpr>) return lookup(node.name);
            if constexpr (std::is_same_v<T, BinopExpr>) {
                auto lt = check_expr(*node.left);
                auto rt = check_expr(*node.right);
                std::string op(node.op);
                if (op=="=="||op=="!="||op=="<"||op==">"||op=="<="||op==">=") {
                    if (lt != rt) errors_.push_back({"type mismatch in comparison", 0});
                    return TypeKind::Bool;
                }
                if (lt != TypeKind::Int || rt != TypeKind::Int)
                    errors_.push_back({"arithmetic requires int operands", 0});
                return TypeKind::Int;
            }
            if constexpr (std::is_same_v<T, CallExpr>) {
                auto& sig = funcs_.at(node.func);
                if (node.args.size() != sig.first.size())
                    errors_.push_back({"wrong argument count for "+node.func, 0});
                for (size_t i = 0; i < node.args.size(); ++i) {
                    auto at = check_expr(*node.args[i]);
                    if (i < sig.first.size() && at != sig.first[i])
                        errors_.push_back({"argument type mismatch", 0});
                }
                return sig.second;
            }
            return TypeKind::Int;
        }, static_cast<Expr::variant&>(e));
        e.inferred_type = t;
        return t;
    }

    void check_stmt(Stmt& s) {
        if (auto* p = dynamic_cast<LetStmt*>(&s)) {
            auto t = check_expr(*p->init);
            if (t != p->type) errors_.push_back({"let type mismatch", 0});
            define(p->name, p->type);
        } else if (auto* p = dynamic_cast<ReturnStmt*>(&s)) {
            auto t = check_expr(*p->value);
            if (t != current_ret_) errors_.push_back({"return type mismatch", 0});
        } else if (auto* p = dynamic_cast<IfStmt*>(&s)) {
            auto ct = check_expr(*p->cond);
            if (ct != TypeKind::Bool) errors_.push_back({"if condition must be bool", 0});
            push_scope(); for (auto& st : p->then_body) check_stmt(*st); pop_scope();
            push_scope(); for (auto& st : p->else_body) check_stmt(*st); pop_scope();
        } else if (auto* p = dynamic_cast<WhileStmt*>(&s)) {
            auto ct = check_expr(*p->cond);
            if (ct != TypeKind::Bool) errors_.push_back({"while condition must be bool", 0});
            push_scope(); for (auto& st : p->body) check_stmt(*st); pop_scope();
        } else if (auto* p = dynamic_cast<ExprStmt*>(&s)) {
            check_expr(*p->expr);
        }
    }

public:
    std::vector<TypeError> check(FuncDef& fn) {
        errors_.clear();
        current_ret_ = fn.ret_type;
        push_scope();
        for (auto& p : fn.params) define(p.name, p.type);
        for (auto& s : fn.body) check_stmt(*s);
        pop_scope();
        return errors_;
    }
};
```

The type checker is 75 lines. It decorates every `Expr` node with `inferred_type`, which the IR lowering phase reads directly — no recomputation.

== Phase 4: Intermediate Representation

The IR is *three-address code* (3AC): each instruction has at most one operator and two source operands, writing to one destination. Variables are either named temporaries (`t0`, `t1`, ...) or user-defined locals.

```text
t0 = a + b
t1 = t0 * c
if t1 goto L1 else goto L2
L1:
  return t1
L2:
  return 0
```

*SSA (Static Single Assignment)* is the standard modern refinement: each temporary is assigned exactly once, and phi-functions at join points merge values from different control-flow paths. SSA makes data-flow analysis trivial (use-def chains are explicit in the variable names) and is the basis of LLVM IR. We do not implement full SSA here — that requires dominator tree computation and phi-insertion — but the 3AC we generate is straightforward to convert.

```cpp
#include <string>
#include <vector>
#include <variant>
#include <sstream>

struct Temp { int id; std::string str() const { return "t"+std::to_string(id); } };

struct IrInstr {
    enum class Op {
        Assign,       // dst = src
        Binop,        // dst = left op right
        Load,         // dst = var
        Store,        // dst (var name) = src1
        JumpIf,       // if src1 goto label else fall through
        Jump,         // goto label
        Label,        // label: (label holds name)
        Call,         // dst = func(args...)
        Return,       // return src1
    } op;
    // dst: destination name (temp or variable) for Assign/Binop/Store/Call/Load
    // src1, src2: source operands
    // label: target label for JumpIf/Jump/Label
    // func: function name for Call
    // binop: operator string for Binop
    std::string dst, src1, src2, func, label;
    std::string binop;
    std::vector<std::string> call_args;
};
using IrBlock = std::vector<IrInstr>;

class IrGen {
    int temp_counter_ = 0;
    IrBlock out_;
    int label_counter_ = 0;

    std::string fresh_temp() { return "t" + std::to_string(temp_counter_++); }
    std::string fresh_label() { return "L" + std::to_string(label_counter_++); }

    std::string lower_expr(const Expr& e) {
        return std::visit([&](const auto& node) -> std::string {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, IntLit>)
                return std::to_string(node.value);
            if constexpr (std::is_same_v<T, BoolLit>)
                return node.value ? "1" : "0";
            if constexpr (std::is_same_v<T, VarExpr>)
                return node.name;
            if constexpr (std::is_same_v<T, BinopExpr>) {
                auto l = lower_expr(*node.left);
                auto r = lower_expr(*node.right);
                auto dst = fresh_temp();
                out_.push_back({IrInstr::Op::Binop, dst, l, r, "", "",
                                std::string(node.op), {}});
                return dst;
            }
            if constexpr (std::is_same_v<T, CallExpr>) {
                std::vector<std::string> args;
                for (auto& a : node.args) args.push_back(lower_expr(*a));
                auto dst = fresh_temp();
                out_.push_back({IrInstr::Op::Call, dst, "", "", node.func,
                                "", "", args});
                return dst;
            }
            return "0";
        }, static_cast<const Expr::variant&>(e));
    }

    void lower_stmt(const Stmt& s) {
        if (auto* p = dynamic_cast<const LetStmt*>(&s)) {
            auto src = lower_expr(*p->init);
            out_.push_back({IrInstr::Op::Store, p->name, src, "", "", "", "", {}});
        } else if (auto* p = dynamic_cast<const ReturnStmt*>(&s)) {
            auto src = lower_expr(*p->value);
            out_.push_back({IrInstr::Op::Return, "", src, "", "", "", "", {}});
        } else if (auto* p = dynamic_cast<const IfStmt*>(&s)) {
            auto cond = lower_expr(*p->cond);
            auto l_then = fresh_label(), l_end = fresh_label();
            // JumpIf: branch to l_then on true; fall through to else on false
            out_.push_back({IrInstr::Op::JumpIf, "", cond, "", "", l_then, "", {}});
            for (auto& st : p->else_body) lower_stmt(*st);
            out_.push_back({IrInstr::Op::Jump,   "", "", "", "", l_end, "", {}});
            out_.push_back({IrInstr::Op::Label,  "", "", "", "", l_then, "", {}});
            for (auto& st : p->then_body) lower_stmt(*st);
            out_.push_back({IrInstr::Op::Label,  "", "", "", "", l_end, "", {}});
        } else if (auto* p = dynamic_cast<const WhileStmt*>(&s)) {
            auto l_top = fresh_label(), l_body = fresh_label(), l_exit = fresh_label();
            out_.push_back({IrInstr::Op::Label,  "", "", "", "", l_top, "", {}});
            auto cond = lower_expr(*p->cond);
            // JumpIf: branch to l_body on true; fall through to l_exit on false
            out_.push_back({IrInstr::Op::JumpIf, "", cond, "", "", l_body, "", {}});
            out_.push_back({IrInstr::Op::Jump,   "", "", "", "", l_exit, "", {}});
            out_.push_back({IrInstr::Op::Label,  "", "", "", "", l_body, "", {}});
            for (auto& st : p->body) lower_stmt(*st);
            out_.push_back({IrInstr::Op::Jump,   "", "", "", "", l_top, "", {}});
            out_.push_back({IrInstr::Op::Label,  "", "", "", "", l_exit, "", {}});
        }
    }

public:
    IrBlock lower(const FuncDef& fn) {
        out_.clear();
        temp_counter_ = 0; label_counter_ = 0;
        for (auto& s : fn.body) lower_stmt(*s);
        return std::move(out_);
    }
};
```

The IR lowering is 70 lines. Expressions become flat sequences of Binop instructions; control flow becomes labeled jumps.

== Phase 5: Code Generation and Stack VM

The VM uses a stack for temporaries and a fixed-size local variable array per frame. Instructions are simple bytecodes.

```cpp
#include <cassert>
#include <functional>
#include <stdexcept>
#include <unordered_map>

enum class Opcode : uint8_t {
    Push,    // push immediate
    Load,    // load local var by index
    Store,   // store to local var by index
    Add, Sub, Mul, Div,
    CmpEq, CmpNe, CmpLt, CmpGt, CmpLe, CmpGe,
    JumpIf,  // pop cond; jump if nonzero
    Jump,
    Call, Ret,
    Halt,
};

struct Instruction {
    Opcode op;
    int    operand = 0;   // index, immediate, or jump target
};

using Bytecode = std::vector<Instruction>;

// --- Codegen ---
class Codegen {
    Bytecode code_;
    std::unordered_map<std::string, int>   local_idx_;
    std::unordered_map<std::string, int>   label_addr_;
    std::vector<std::pair<int,std::string>> fixups_;
    int local_count_ = 0;

    int alloc_local(const std::string& name) {
        if (local_idx_.count(name) == 0)
            local_idx_[name] = local_count_++;
        return local_idx_[name];
    }

    void emit(Opcode op, int operand = 0) {
        code_.push_back({op, operand});
    }
    void emit_label(const std::string& lbl) {
        label_addr_[lbl] = (int)code_.size();
    }
    void emit_jump(Opcode op, const std::string& lbl) {
        fixups_.push_back({(int)code_.size(), lbl});
        emit(op, 0);
    }

    void gen_instr(const IrInstr& ir) {
        switch (ir.op) {
        case IrInstr::Op::Label:
            emit_label(ir.label); break;
        case IrInstr::Op::Store: {
            int idx = alloc_local(ir.dst);   // dst holds the variable name
            // src1 is a temp or immediate
            if (!ir.src1.empty() && std::isdigit((unsigned char)ir.src1[0]))
                emit(Opcode::Push, std::stoi(ir.src1));
            else
                emit(Opcode::Load, alloc_local(ir.src1));
            emit(Opcode::Store, idx); break;
        }
        case IrInstr::Op::Binop: {
            auto push_val = [&](const std::string& s) {
                if (!s.empty() && (std::isdigit((unsigned char)s[0]) || s[0]=='-'))
                    emit(Opcode::Push, std::stoi(s));
                else
                    emit(Opcode::Load, alloc_local(s));
            };
            push_val(ir.src1); push_val(ir.src2);
            std::string op = ir.binop;
            if      (op=="+")  emit(Opcode::Add);
            else if (op=="-")  emit(Opcode::Sub);
            else if (op=="*")  emit(Opcode::Mul);
            else if (op=="/")  emit(Opcode::Div);
            else if (op=="==") emit(Opcode::CmpEq);
            else if (op=="!=") emit(Opcode::CmpNe);
            else if (op=="<")  emit(Opcode::CmpLt);
            else if (op==">")  emit(Opcode::CmpGt);
            else if (op=="<=") emit(Opcode::CmpLe);
            else if (op==">=") emit(Opcode::CmpGe);
            emit(Opcode::Store, alloc_local(ir.dst)); break;
        }
        case IrInstr::Op::Return:
            if (!ir.src1.empty() && std::isdigit((unsigned char)ir.src1[0]))
                emit(Opcode::Push, std::stoi(ir.src1));
            else
                emit(Opcode::Load, alloc_local(ir.src1));
            emit(Opcode::Ret); break;
        case IrInstr::Op::JumpIf:
            emit(Opcode::Load, alloc_local(ir.src1));
            emit_jump(Opcode::JumpIf, ir.label);   // jump to then on true; fall through on false
            break;
        case IrInstr::Op::Jump:
            emit_jump(Opcode::Jump, ir.label); break;
        default: break;
        }
    }

public:
    Bytecode compile(const IrBlock& ir) {
        code_.clear(); local_idx_.clear(); label_addr_.clear();
        fixups_.clear(); local_count_ = 0;
        for (auto& instr : ir) gen_instr(instr);
        emit(Opcode::Halt);
        for (auto& [addr, lbl] : fixups_)
            code_[addr].operand = label_addr_.at(lbl);
        return code_;
    }
    int local_count() const { return local_count_; }
};

// --- Stack VM ---
struct Frame {
    std::vector<int> locals;
    int              ret_addr = 0;
};

class VM {
    std::vector<int>   stack_;
    std::vector<Frame> call_stack_;

public:
    int run(const Bytecode& code, int local_count) {
        call_stack_.push_back(Frame{std::vector<int>(local_count, 0), 0});
        int ip = 0;
        for (;;) {
            assert(ip < (int)code.size());
            auto& ins = code[ip++];
            auto& locs = call_stack_.back().locals;
            switch (ins.op) {
            case Opcode::Push:  stack_.push_back(ins.operand); break;
            case Opcode::Load:  stack_.push_back(locs[ins.operand]); break;
            case Opcode::Store: locs[ins.operand] = stack_.back(); stack_.pop_back(); break;
            case Opcode::Add: { auto b=stack_.back(); stack_.pop_back();
                                stack_.back() += b; break; }
            case Opcode::Sub: { auto b=stack_.back(); stack_.pop_back();
                                stack_.back() -= b; break; }
            case Opcode::Mul: { auto b=stack_.back(); stack_.pop_back();
                                stack_.back() *= b; break; }
            case Opcode::Div: { auto b=stack_.back(); stack_.pop_back();
                                if (b==0) throw std::runtime_error("div by zero");
                                stack_.back() /= b; break; }
            case Opcode::CmpEq: { auto b=stack_.back(); stack_.pop_back();
                                  stack_.back() = (stack_.back()==b); break; }
            case Opcode::CmpNe: { auto b=stack_.back(); stack_.pop_back();
                                  stack_.back() = (stack_.back()!=b); break; }
            case Opcode::CmpLt: { auto b=stack_.back(); stack_.pop_back();
                                  stack_.back() = (stack_.back()<b); break; }
            case Opcode::CmpGt: { auto b=stack_.back(); stack_.pop_back();
                                  stack_.back() = (stack_.back()>b); break; }
            case Opcode::CmpLe: { auto b=stack_.back(); stack_.pop_back();
                                  stack_.back() = (stack_.back()<=b); break; }
            case Opcode::CmpGe: { auto b=stack_.back(); stack_.pop_back();
                                  stack_.back() = (stack_.back()>=b); break; }
            case Opcode::JumpIf: { auto c=stack_.back(); stack_.pop_back();
                                   if (c) ip = ins.operand; break; }
            case Opcode::Jump:   ip = ins.operand; break;
            case Opcode::Ret:  { int v=stack_.back(); stack_.pop_back();
                                 call_stack_.pop_back();
                                 if (call_stack_.empty()) return v;
                                 ip = call_stack_.back().ret_addr; break; }
            case Opcode::Halt: return stack_.empty() ? 0 : stack_.back();
            default: break;
            }
        }
    }
};
```

The codegen is 90 lines; the VM is 45 lines. Together they implement a complete execute path from IR to integer result.

== Phase 6: Optimizer Pass

A constant-folding pass over the IR eliminates computations whose operands are both compile-time constants. Dead-code elimination removes instructions that write to temporaries never subsequently read.

```cpp
#include <optional>

// Constant folding: replace Binop(const, const) with a Push.
IrBlock fold_constants(IrBlock ir) {
    std::unordered_map<std::string, int> const_map;
    IrBlock out;
    for (auto& ins : ir) {
        if (ins.op == IrInstr::Op::Store) {
            // track constant assignments: var = literal (dst holds the variable name)
            try { const_map[ins.dst] = std::stoi(ins.src1); }
            catch (...) { const_map.erase(ins.dst); }
            out.push_back(ins);
        } else if (ins.op == IrInstr::Op::Binop) {
            auto lit = [&](const std::string& s) -> std::optional<int> {
                auto it = const_map.find(s);
                if (it != const_map.end()) return it->second;
                try { return std::stoi(s); } catch (...) { return std::nullopt; }
            };
            auto lv = lit(ins.src1), rv = lit(ins.src2);
            if (lv && rv) {
                int result = 0;
                std::string op = ins.binop;
                if      (op=="+") result = *lv + *rv;
                else if (op=="-") result = *lv - *rv;
                else if (op=="*") result = *lv * *rv;
                else if (op=="/") result = (*rv!=0) ? *lv / *rv : 0;
                else if (op=="==") result = (*lv == *rv);
                else if (op=="<")  result = (*lv < *rv);
                else { out.push_back(ins); continue; }
                const_map[ins.dst] = result;
                // emit a direct store of the folded constant
                IrInstr folded{IrInstr::Op::Store};
                folded.src1 = std::to_string(result);
                folded.dst = ins.dst;
                out.push_back(folded);
            } else {
                const_map.erase(ins.dst);
                out.push_back(ins);
            }
        } else {
            out.push_back(ins);
        }
    }
    return out;
}
```

This 40-line pass can eliminate entire loops over constant bounds when combined with branch folding (not shown). It demonstrates the key property of the 3AC IR: each instruction is independent, and optimizations are local rewrites.

== Scaling Up

The compiler above is ~400 lines. Production compilers scale the same architecture:

*LLVM IR / MLIR:* Replace the 3AC with SSA form and a rich type system. LLVM's pass manager runs hundreds of optimization passes (inlining, loop unrolling, alias analysis, vectorization) on the IR before handing off to a register allocator and backend. MLIR (Multi-Level IR) allows multiple IR dialects at different levels of abstraction — useful for ML compilers targeting GPUs and tensor hardware.

*Cranelift:* A fast single-pass register allocator and code generator developed for WebAssembly (Wasmtime). It prioritizes compile speed over optimization quality, making it suitable for JIT compilation.

*Register allocation:* Our stack VM avoids register allocation entirely. Real backends must map IR temporaries to machine registers (typically 16 on x86-64). Graph-coloring allocation (Chaitin 1982) is the classical approach; linear scan (Poletto, Sarkar 1999) is faster and used in most JITs.

*SSA construction:* Converting our 3AC to SSA requires computing the dominance tree of the control-flow graph and inserting phi-functions at dominance frontiers (Cytron et al. 1991). SSA makes most dataflow analyses $O(n)$ instead of $O(n^2)$.

=== Where This Approach Breaks Down

The muC compiler handles the happy path. Real languages add hard problems:

- *Heap-allocated data and GC:* the VM needs a garbage collector, which requires knowing which stack slots hold pointers (a *stack map* or *root set*). Precise GC requires cooperation between the compiler and the runtime.
- *Closures with capture:* a closure that captures a local variable requires that variable to live on the heap after the enclosing function returns. The compiler must perform *escape analysis* to determine which locals escape and allocate them accordingly.
- *Optimizing compilation:* aggressive optimization is essentially graph rewriting on the IR. Alias analysis, loop-invariant code motion, and auto-vectorization all require sophisticated analyses that dwarf the compiler above in complexity.

== Epilogue: Sub-Turing-Complete by Design

Several widely deployed languages are intentionally not Turing-complete. The payoff is *decidable analysis* — properties the tooling can prove or refute in finite time.

*Regular languages:* Regular expressions, `grep`, lexers. Finite-state. Every property (emptiness, equivalence, intersection) is decidable. Query optimizers use regex-like patterns for index selection.

*JSON without recursive CTEs:* A JSON schema validator can decide in $O(n)$ whether any document matches a schema. Add recursive `$ref` and the schema language becomes Turing-complete; schema validation becomes undecidable in general.

*SQL without recursive CTEs:* Standard SQL SELECT is equivalent to relational algebra — first-order logic over finite structures, which is decidable. The query optimizer can rewrite, reorder, and index-select queries with provable correctness. Recursive CTEs (`WITH RECURSIVE`) add a fixed-point operator and lift SQL into the territory of Datalog; most optimizers treat recursive queries conservatively.

*HTML and CSS:* Vanilla HTML is not TC. CSS3 with the general sibling combinator (`~`) and `:checked` pseudo-class can simulate cellular automata (Ligatti 2015) — but browsers do not expose the output of those computations as data, so the TC-ness is inert.

*Coq's terminating fragment:* Every Coq function must terminate. The termination checker decides whether recursive calls are on structurally smaller arguments. This makes every Coq function's termination decidable. The trade is that you cannot define a general fixed-point combinator.

== Epilogue: Turing-Completeness by Accident

Some systems became TC without their designers intending it.

*TeX:* Knuth's typesetting system (1978) is Turing-complete via its macro expansion system. TeX macros can simulate a Turing machine (Turing completeness was demonstrated by various authors). Consequence: a TeX document can loop forever. `pdflatex` has no way to detect this; it just hangs.

*C++ templates:* Veldhuizen (2003) demonstrated that C++ template instantiation is Turing-complete. A template metaprogram can compute any computable function at compile time. Consequence: template instantiation can run forever, the compiler's memory usage is unbounded, and error messages are famously incomprehensible. C++20 `consteval` and `constexpr` provide a TC but bounded alternative.

*CSS3 sibling selectors:* The combination of `:checked`, `~`, and `input` elements can encode state machines in CSS alone (no JavaScript). While this is a curiosity rather than a deployment concern, it means that CSS layout engines cannot, in general, decide statically whether a stylesheet will produce a fixed-point layout.

*What these cost in tooling:* Linters for TC languages cannot decide common questions (dead code, unused variables) in general — they must be heuristic or conservative. IDE features like "go to definition" and "find all references" become best-effort rather than exact. Testing must replace proof: you cannot prove a property of all inputs, so you test a sample. The accidental TC systems above each carry this tax.

== Closing Thought

Characters became tokens via DFA. Tokens became trees via PDA. Trees became typed trees via attribute grammars and decidable Hindley-Milner inference. Typed trees became three-address IR and bytecode via syntax-directed translation. Every layer was an automaton, and the compiler is the proof — not of a theorem on paper, but of a working system you can run.

The hierarchy is not a taxonomy of historical curiosity. It is an engineering guide: choose the weakest computational model that solves your problem. Stay regular when you can. Go context-free when you must. Reach for Turing-completeness only when the problem genuinely requires it — and when you do, accept the tooling costs that come with it.

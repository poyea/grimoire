= Macros and Metaprogramming: Formal Foundations

== Hygiene Formalized: Macros That Work (Clinger--Rees)

(Clinger--Rees 1991, "Macros That Work") gave the first rigorous *proof* that the
`syntax-rules` expander satisfies hygiene, via the painting/timestamp algorithm
below. The argument works by induction on the derivation of the expansion relation.

*Setting.* Fix a `syntax-rules` macro $m$ with rules
$[(m p_1) t_1], dots, [(m p_n) t_n]$. Let $E$ be the expansion of a use
$u = (m e_1 dots e_k)$ in the context produced by pattern $p_i$ matching against
template $t_i$, where $sigma$ is the substitution mapping pattern variables to
syntax objects (pieces of the input with their use-site hygiene metadata attached).

*Definition.* An identifier $x$ occurring in the output $E$ is called *macro-introduced*
if it originated in $t_i$ (the template), and *use-site* if it originated in some
$e_j$ (an argument). A mixed identifier is one produced by a template escape
`(datum->syntax ...)` and is treated as use-site for hygiene purposes.

*Theorem (Clinger--Rees 1991; paraphrase).* For any `syntax-rules` expansion,
no macro-introduced identifier in the output $E$ free-captures a use-site binding,
and no use-site identifier free-captures a macro-introduced binding.

*Proof (sketch).* By structural induction on the template $t_i$.

- *Atomic case.* A template atom $x$ is macro-introduced; by the painting algorithm,
  it is painted with the fresh colour of the current invocation. A use-site variable
  $v$ (a pattern variable substituted in) carries its own colour from the call site.
  Since the two colours are distinct fresh tokens, no identifier resolution can
  confuse them.
- *List template* $(t t')$. By the inductive hypothesis, neither $t$ nor $t'$
  contains cross-capture. Their concatenation therefore inherits this property,
  since identifier resolution is name-plus-colour comparison: concatenation does
  not merge colours.
- *Ellipsis* $(t dots)$. Each expansion of $t$ in the ellipsis sequence is
  painted with the *same* invocation colour (the one stamp is global to the
  expansion, not re-minted per ellipsis iteration). Use-site pieces substituted
  through a pattern variable carry their own colour. Again no confusion.
- *Nested macro.* A macro call $m'$ inside $t_i$ triggers a recursive expansion;
  by the inductive hypothesis on the sub-expansion (with a *fresh* paint for $m'$),
  the sub-expansion is hygienic. Since each invocation uses a distinct colour, the
  outer and inner expansions do not cross-capture. $square$

The colour-based argument translates directly to the set-of-scopes model: "colour"
becomes "the scope introduced at the current macro boundary", and "distinct colours"
becomes "disjoint scope sets outside the shared use context".

== The Ellipsis Pattern Language: Algorithm

The ellipsis sub-language of `syntax-rules` is its own small rewrite system. We
describe the matching and expansion algorithms precisely.

*Syntax.* A pattern is one of:
- A literal `(literal ...)` keyword — matched exactly.
- A pattern variable `var` — matches any single datum.
- `(p ...)` — matches a (possibly empty) sequence of matches of $p$.
- `(p q ...)` — matches a list whose head matches $p$ and tail matches $q dots$.
- A nested list `(p1 p2 ...)`.
- The underscore `_` — matches anything, binds nothing.

An ellipsis variable is a pattern variable that appears under at least one `...` in
the pattern. It is *multi-valued* after matching: it holds a vector of matched syntax
objects.

*Matching algorithm.* `match(pattern, form)` returns a substitution $sigma$ or failure:

```text
match(var, stx)          = { var |-> stx }
match(lit, stx)          = {} if lit = datum(stx), else fail
match(_, stx)            = {}
match((p ...), (s1...sn))= union_i match(p, si)    ; si's must be same-length
match((p1 p2 ...), (s . rest))
    = union(match(p1, s), match((p2 ...), rest))
match((p1 p2), (s1 s2)) = union(match(p1, s1), match(p2, s2))
```

For the ellipsis case, each pattern variable $v$ appearing in $p$ becomes
ellipsis-bound: $sigma(v) = [|v_1, dots, v_n|]$ where $v_i$ is the value of $v$
in the $i$-th repetition.

*Template expansion.* `expand(template, sigma)`:

```text
expand(var, sigma)       = sigma(var)       ; single value
expand(lit, sigma)       = lit              ; painted with current colour
expand((t ...), sigma)   =                  ; t contains ellipsis vars E
    let n = |sigma(e)| for some e in E      ; all e in E must give same n
    [expand(t, sigma_i) for i in 1..n]      ; sigma_i = sigma with e |-> sigma(e)[i]
expand((t1 t2), sigma)   = (expand(t1, sigma)  expand(t2, sigma))
```

*Theorem (Termination).* The matching algorithm terminates because each recursive
call reduces the size of the pattern. The expansion algorithm terminates because
the template is a fixed finite tree; ellipsis expansion produces a finite sequence
(bounded by the input's matched length). $square$

*Example.* Consider
```racket
(define-syntax my-let
  (syntax-rules ()
    [(_ ((v e) ...) body ...)
     ((lambda (v ...) body ...) e ...)]))
```

For the call `(my-let ((x 1) (y 2)) (+ x y))`:

- Pattern: `((v e) ...)` matches `((x 1) (y 2))` giving
  $sigma = { v |-> [x, y], quad e |-> [1, 2] }$.
- `body ...` matches `(+ x y)` giving $sigma("body") = [|(+ x y)|]$.
- Template expansion: `(v ...)` expands to `x y`; `body ...` to `(+ x y)`;
  `e ...` to `1 2`.
- Result: `((lambda (x y) (+ x y)) 1 2)`.

The entire substitution is transparent to the user; hygiene is maintained by the
colour assignment.

== Procedural Macros and Phase-1 Computation

`syntax-case` macros in Racket are Scheme procedures that run at *phase 1*, i.e.,
during the expansion of phase-0 code. The procedure receives a *syntax object* and
returns a syntax object. Inside the body, arbitrary Scheme computation is permitted.

*Example: a `while` loop.*

```racket
(define-syntax while
  (lambda (stx)
    (syntax-case stx ()
      [(_ condition body ...)
       (with-syntax ([loop-id (datum->syntax stx 'loop)])
         #'(let loop-id ()
             (when condition
               body ...
               (loop-id))))])))
```

The `datum->syntax stx 'loop` call *injects* the identifier `loop` into the user's
scope (the scope of `stx`, the use site), so that recursive calls `(loop-id)` refer
to the `let`-binding. The `condition` and `body ...` are use-site code and keep
their original scope. The `#'(let loop-id ...)` template is macro-introduced (carries
the current macro's scope).

*Phase-1 computation example: compile-time lookup.*

```racket
(define-syntax fetch-constant
  (lambda (stx)
    (syntax-case stx ()
      [(_ key)
       (let ([val (hash-ref compile-time-table (syntax->datum #'key) #f)])
         (if val
             (datum->syntax stx val)
             (raise-syntax-error #f "unknown key" stx)))])))
```

Here `compile-time-table` is a hash table available at phase 1 (perhaps defined in
a `(for-syntax ...)` block). The macro looks up `key` at expand time and substitutes
the result directly into the code. The *run-time* program sees a literal value; the
lookup cost is paid once, at compile time.

*Theorem (Phase Soundness, Flatt 2002).* In a module system with phase separation,
a module can be safely instantiated at different phases without interference: the
bindings available at phase $n$ are independent of those at phase $m eq.not n$. In
particular, a module's phase-1 instantiation does not share mutable state with its
phase-0 instantiation.

*Proof sketch.* Each phase instantiation is a separate environment; import resolution
is indexed by both module identity and phase number. Mutation of a phase-1 binding
cannot affect the phase-0 binding because they are distinct cells. $square$

== Zig `comptime`, Nim Macros, and the Spectrum of Compile-Time Evaluation

The design space of "things that run at compile time" has four main points:

1. *Constant folding (all languages):* the compiler reduces constant expressions.
   Scope: limited to values the compiler can verify are pure and bounded.

2. *Typed compile-time evaluation (Zig `comptime`, Rust `const fn`):* arbitrary
   computations at compile time, typed by the host type system, but *value*-level
   only. No new syntax is generated.

3. *AST macros (Nim, Elixir, Lean 4):* a function from AST to AST, with full
   host-language power at macro-expand time. Both values and syntax can be generated.

4. *Full multi-stage programming (MetaOCaml, Scala 3 quoted, Template Haskell):*
   typed quotation and splicing across multiple stages, with a formal type-level
   distinction between stages.

*Zig `comptime`.* In Zig, any function may be called at compile time by annotating
arguments `comptime`. Types are first-class `comptime` values:

```zig
fn makeArray(comptime T: type, comptime n: usize) [n]T {
    return [_]T{0} ** n;
}

const arr = makeArray(u32, 8);  // [8]u32, computed at compile time
```

Zig's entire generic system is built on `comptime`: a "generic" function is just a
function that takes a `comptime` type argument. There is no separate template
language. The compile-time evaluator is a full interpreter for the Zig language.

*Nim macros.* Nim provides three levels of compile-time facilities:

- `template`: hygienic textual substitution.
- `macro`: a Nim function from `NimNode` (the AST type) to `NimNode`, evaluated
  at compile time by the Nim compiler's built-in interpreter.
- `static`: blocks evaluated at compile time (value-only, no AST).

```nim
macro debug(args: varargs[typed]): untyped =
  result = newNimNode(nnkStmtList)
  for arg in args:
    result.add(newCall("echo",
      newLit(arg.repr & " = "),
      arg))
```

The `repr` function gives the source-text representation of the AST node; the macro
builds a statement list that echoes each argument's name and value. This is Nim's
analogue of `printf`-debugging as a macro.

*Rust `const fn` and `const` generics.* Rust's `const fn` evaluates functions at
compile time when all arguments are constants:

```rust
const fn fib(n: u64) -> u64 {
    match n {
        0 | 1 => n,
        _ => fib(n - 1) + fib(n - 2),
    }
}

const FIB10: u64 = fib(10);
```

*Const generics* parameterise types by values:

```rust
struct Matrix<const M: usize, const N: usize> {
    data: [[f64; N]; M],
}
```

The type system ensures at compile time that matrix operations are dimensionally
correct. This is the typed-DSL use of compile-time computation without full macros.

*Comparison.* Macros (Nim, Lisp) let the programmer generate *arbitrary new syntax*;
`comptime`/`const fn` let the programmer compute *values* at compile time. The
boundary is the ability to traverse and construct ASTs. Systems with only value-level
compile-time evaluation cannot implement `while`-loop syntax, pattern matching, or
monadic do-notation — these require syntactic transformation. The convergence of
the two is the "macro that only generates values" pattern ubiquitous in Rust `macro_rules!`.

== Monadic do-Notation Desugaring

Haskell's `do`-notation is a macro in the technical sense: the compiler desugars it
to calls to `(>>=)` and `(>>)` before type-checking. The desugar rules:

```haskell
do { e }           =  e
do { e; stmts }    =  e >> do { stmts }
do { x <- e; stmts }  =  e >>= \x -> do { stmts }
do { let x = e; stmts }  =  let x = e in do { stmts }
```

*Example.* The program

```haskell
do
  name <- getLine
  let greeting = "Hello, " ++ name
  putStrLn greeting
```

desugars to:

```haskell
getLine >>= (\name ->
  let greeting = "Hello, " ++ name in
  putStrLn greeting)
```

The desugar is purely syntactic: the compiler does not know what `>>=` means; it
just rewrites the syntax. The *type checker* then resolves `>>=` to whatever `Monad`
instance is in scope.

*Applicative-do* (Marlow et al. 2016) refines the desugaring: when consecutive
`<-` actions do not depend on each other, they are desugared to `(<*>)` rather than
`(>>=)`, enabling parallelism in effects interpreters that can distinguish applicative
structure from monadic structure.

```haskell
-- Instead of sequential >>= chain:
do { x <- a; y <- b; return (f x y) }

-- Applicative-do produces:
liftA2 f a b
```

The applicative desugaring is a *data-flow analysis* on the `do`-block: build a
dependency graph of the bound variables, identify independent groups, and emit
`(<*>)` for independent groups and `(>>=)` for dependent ones.

== Async/Await Desugaring as a Macro

The `async/await` pattern can be understood as a *macro* that transforms a
direct-style coroutine into a state machine. The precise desugar depends on the
language, but the common structure is:

*Input (direct style):*
```rust
async fn read_two(r: &mut Reader) -> (Bytes, Bytes) {
    let a = r.read().await;
    let b = r.read().await;
    (a, b)
}
```

*Output (state machine):*
```rust
enum ReadTwo { S0, S1 { a: Bytes }, Done }

impl Future for ReadTwo {
    type Output = (Bytes, Bytes);
    fn poll(&mut self, cx: &mut Context) -> Poll<(Bytes, Bytes)> {
        match self {
            S0 => { /* start first read, transition to S1 */ },
            S1 { a } => { /* second read done, return (a, b) */ },
            Done => panic!("polled after completion"),
        }
    }
}
```

Each `.await` point becomes a *state transition*. Live variables at the suspension
point (here `a` between the two reads) must be stored in the state enum. The macro
transformation is in fact a form of *continuation-passing style* (CPS) transformation
followed by defunctionalisation of the continuation type.

*Theorem.* The CPS transformation of a direct-style program $P$ with $n$ `await`
points produces a state machine with at most $n + 1$ states. Each state corresponds
to the continuation of $P$ from a particular `await` point. $square$

This is directly analogous to the *Futamura first projection*: specialising the
"async interpreter" (the executor) to a particular coroutine `read_two` yields the
state machine — the compiled form.

== Macro Hygiene Comparison across Languages

The following table summarises the hygiene discipline of the major macro systems.

#table(
  columns: (auto, auto, auto, auto),
  [*System*], [*Hygiene*], [*Break hygiene*], [*Mechanism*],
  [Scheme `syntax-rules`], [Full, automatic], [`datum->syntax`], [Painting / scopes],
  [Racket `syntax-case`], [Full, automatic], [`datum->syntax`], [Set-of-scopes],
  [Common Lisp `defmacro`], [None], [n/a], [`gensym` required],
  [Clojure `defmacro`], [Partial (ns-qualified)], [`gensym`], [Namespace resolution],
  [Rust `macro_rules!`], [Full, span-based], [None (use `proc_macro`)], [Token spans],
  [Rust `proc_macro`], [None by default], [n/a], [Manual `Span` management],
  [Nim `macro`], [Full, `gensym` provided], [`bindSym`, `genSym`], [AST identity],
  [Template Haskell], [Opt-in (`newName`)], [`mkName`], [Two-tier Name API],
  [Scala 3 `quoted`], [Full, via `Quotes`], [Explicit `Symbol` forge], [Quotes capability],
  [MetaOCaml], [Full, by type system], [None — cannot break], [Type tracks binding],
)

The right column shows the *mechanism* by which hygiene is implemented. The
set-of-scopes and span-based mechanisms are operationally equivalent for most
programs; they differ in how they handle *macro-generating macros* (macros that
produce other macros) and *separately compiled* code.

== Template Haskell: Full Q Monad Walkthrough

Template Haskell's `Q` monad supports: `newName :: String -> Q Name` (fresh name
generation), `reify :: Name -> Q Info` (compile-time reflection), `runIO :: IO a -> Q a`
(I/O at compile time, for reading files, querying databases, etc.), and `recover ::
Q a -> Q a -> Q a` (error recovery).

*Full example: automatic lens generation.*

Given the record

```haskell
data Person = Person { _name :: String, _age :: Int }
```

`makeLenses ''Person` is a TH splice that:

1. `reify ''Person` yields `TyConI (DataD ... [RecC 'Person [(mkName "_name", ...,
   ConT ''String), (mkName "_age", ..., ConT ''Int)]] ...)`.
2. For each field `_name`, generate:
   ```haskell
   name :: Lens' Person String
   name f (Person n a) = fmap (\n' -> Person n' a) (f n)
   ```
   by constructing the corresponding `Q [Dec]`.
3. Splice the generated declarations.

The `Q [Dec]` value is built by quotation:

```haskell
makeLens :: Name -> Name -> Q [Dec]
makeLens fname recname = do
  f <- newName "f"
  s <- newName "s"
  let body = [| \$(varE f) ($(conE recname) ...) = ... |]
  return [FunD lensName [Clause [...] (NormalB body) []]]
```

The generated names `f` and `s` are fresh (from `newName`), ensuring that even if
the user has a variable `f` in scope at the splice site, it is not captured.

== Error Messages, IDE Support, and the Cost of Macros

*Error messages.* When expanded code fails to type-check, compilers report the
error at the *expanded location*, not the *use site*. This is the most common
complaint about macro systems. Three engineering responses exist:

1. *Source maps.* Every token in the expanded output carries a provenance pointer
   to where it came from in the source (use site or template). Rust's `proc_macro`
   carries `Span` information on every token; error messages thread back through
   the span.
2. *Syntax objects as errors.* Racket's `raise-syntax-error` takes the *use-site
   syntax object* and uses its location for the error report, so the user sees the
   macro call in the error, not the template.
3. *Typed quotation.* MetaOCaml catches type errors at the *meta-stage*: the code
   value `.<e>.` must be well-typed in the *embedding context*, so a type mismatch
   inside the bracket is reported when the bracket is written, not when it is run.

*IDE support.* A language server for a macro-heavy language must either:
- *Expand macros* before semantic analysis (Racket's DrRacket, Rust's rust-analyzer).
  This is correct but slow: expansion must succeed before features like
  jump-to-definition work.
- *Approximate* the expansion by pattern-matching known macros (IntelliJ for Scala
  macros). Fast but incomplete.
- *Require macros to declare their shape* (proposed for Lean 4 and some Haskell
  proposals): the macro author provides a *stub* declaration that the IDE uses for
  semantic analysis, with the macro filling in the body.

*Debuggability.* Stepping through generated code in a debugger is disorienting.
The standard remedy is *macro stepper* tooling (DrRacket's Macro Stepper, introduced
by Clements--Culpepper--Flatt 2007): a display that shows each expansion step with
before/after views, allows the programmer to see which template produced each piece
of output, and connects IDE features to the pre-expansion source.

*Compile time.* Macro expansion adds a pass to compilation. In Rust, procedural
macros are compiled as separate crates and invoked via IPC; the overhead is several
hundred milliseconds per crate in addition to the cost of running the macro itself.
Heavy `derive` usage (e.g., `serde`'s `Serialize`/`Deserialize` on large types)
is a major contributor to Rust compile times; incremental compilation and
*declarative macro* alternatives (`macro_rules!` versions of `serde`) are common
mitigations.

== MetaML Type System: Full Presentation

MetaML's type system extends simply-typed $lambda$-calculus with three new forms.
Let $Gamma tack.r_n e : tau$ denote "$e$ has type $tau$ at stage $n$".

*Typing rules for the three staging operations:*

$
  (Gamma tack.r_n e : tau) / (Gamma tack.r_(n) angle.l e angle.r : angle.l tau angle.r) quad quad ("BRACKET")
$

$
  (Gamma tack.r_n e : angle.l tau angle.r) / (Gamma tack.r_(n+1) tilde e : tau) quad quad ("ESCAPE")
$

$
  (Gamma tack.r_0 e : angle.l tau angle.r) / (Gamma tack.r_0 "run" e : tau) quad quad ("RUN")
$

BRACKET says: if $e$ is well-typed at stage $n$, then the code value $angle.l e angle.r$
has type $angle.l tau angle.r$ at stage $n$. ESCAPE says: inside a stage-$(n+1)$
bracket, an escape $tilde e$ at stage $n$ splices the code value $e : angle.l tau angle.r$
as a $tau$ at stage $n+1$. RUN only applies at stage 0 and removes a bracket.

*Cross-stage persistence (CSP).* A value at stage 0 used inside a stage-1 bracket
is a *cross-stage use*. MetaML handles this with a special form `lift : tau -> angle.l tau angle.r`
that serialises a value into a code constant. The type rule:

$
  (Gamma tack.r_0 v : tau) / (Gamma tack.r_1 "lift" v : angle.l tau angle.r) quad quad ("LIFT")
$

Not all types support CSP: closures that refer to mutable state cannot be lifted.
The *liftable* types form a sub-kind.

*Theorem (Taha 2000; Subject Reduction).* If $Gamma tack.r_n e : tau$ and
$e arrow.r e'$ (by the staged reduction relation), then $Gamma tack.r_n e' : tau$.

*Proof.* By induction on the typing derivation and the reduction rule. The key case
is ESCAPE under BRACKET: reducing $angle.l tilde e angle.r arrow.r e$ when
$e : angle.l tau angle.r$ at stage $n$ yields $e$ itself, which has type $angle.l tau angle.r$
at stage $n$. All other cases follow from the standard preservation argument for
$lambda$-calculus extended with fresh constants. $square$

The significance: MetaML's staging cannot produce an ill-typed code value. Code
generation is type-safe end-to-end. This is qualitatively different from all
unhygienic or untyped macro systems.

== Cross-Language Synthesis: Lessons from Forty Years

The history of macros yields a short list of lessons, each earned by a painful
implementation experience.

1. *Hygiene must be the default.* Every macro system that requires the programmer
   to manually ensure hygiene (Common Lisp, Clojure) produces bugs in practice.
   The bugs are subtle, hard to find, and occur only when user and macro names happen
   to collide. Hygiene-by-default (Scheme, Rust `macro_rules!`) eliminates this
   class of bug.

2. *The escape from hygiene must be explicit and typed.* `datum->syntax`, `mkName`,
   and `bindSym` are explicit choices. A macro that breaks hygiene accidentally is
   a bug; one that breaks it deliberately (to introduce an anaphor) is a feature —
   and the programmer can see the difference in the code.

3. *Phases are necessary for large programs.* A module system that does not separate
   compile-time and run-time dependencies cannot be compiled independently. Phase
   separation (Flatt 2002) is the enabling technology for macro-using modules at
   scale.

4. *Types recover what hygiene cannot guarantee.* Hygiene prevents capture; types
   prevent ill-typed generation. MetaOCaml's typed brackets guarantee that every
   generated program is well-typed in the target language — a guarantee that
   untyped macro systems cannot offer and that Template Haskell achieves only
   partially (TH splices are type-checked after generation, not during).

5. *IDE tooling is not optional.* A macro system without a macro stepper, without
   span-aware error messages, and without IDE support is not usable by ordinary
   programmers. The investment in tooling (DrRacket's macro stepper, rust-analyzer's
   expansion, Scala's presentation compiler) is as important as the theoretical
   correctness of the system.

6. *Macros converge with partial evaluation.* The most powerful macro uses —
   Lightweight Modular Staging, typed Template Haskell, MetaOCaml — are in fact
   partial evaluators. The Futamura projections, applied to DSL interpreters, yield
   macro-based compilers competitive with hand-written code generators.

== Further Reading

Clinger, W., Rees, J. (1991). "Macros That Work." POPL. Formalises hygienic, referentially transparent macro expansion for Scheme; the key theoretical model behind the scoped-substitution semantics presented in this chapter.

Kohlbecker, E., Wand, M. (1987). "Macro-by-Example: Deriving Syntactic Transformations from Their Specifications." POPL. The pattern-based macro specification model; introduces example-driven macro definitions that are the basis of syntax-rules.

Culpepper, R., Felleisen, M. (2010). "Fortifying Macros." ICFP. Introduces macro contracts and blame for macro-generated code; the formal treatment of robust macros that produce well-formed output.

Taha, W., Sheard, T. (1997). "Multi-Stage Programming with Explicit Annotations." PEPM. Introduces MetaML's type-directed staging; the formal foundation for typed quotation, splicing, and the bracket-escape discipline.

Herman, D., Wand, M. (2008). "A Theory of Hygienic Macros." ESOP. The first complete formal theory of macro hygiene using nominal logic; proves correctness of hygienic expansion relative to alpha-equivalence.

Flatt, M. (2012). "Binding as Sets of Scopes." POPL. The definitive formal model of Racket's scope-set macro system; resolves subtleties of local-expand and module-level hygiene in a single unified framework.
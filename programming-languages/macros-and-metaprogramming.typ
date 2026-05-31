= Macros and Metaprogramming

A macro is a program that runs at compile time and emits source code. The idea is older than the high-level languages: assembler macros expanded mnemonic sequences before the assembler proper saw them. McCarthy's Lisp made the macro a *first-class* facility by making program text and runtime data the same type — *s-expressions* — so that a program that produces programs is just a program that produces lists. Three generations of language designers have since shown that this simple idea, when combined with a serious binding theory, gives a discipline that supplants nearly every other "syntactic extension" mechanism: pattern matching, async/await, do-notation, contract systems, ORM DSLs, and entire embedded languages are macros.

*See also:* _Type Systems_, _Effects and Handlers_

The technical challenges of doing macros *well* are unobvious and were not understood for two decades after macros were invented. Variable capture, *referential transparency* of identifiers introduced by the macro, *phase separation* between compile-time and run-time computations, and the *hygiene* problem occupied the Scheme community from the late 1980s through 2016. The set-of-scopes model (Flatt 2016) is the current synthesis. Multi-stage programming (Taha–Sheard 1997) is a *typed* alternative that runs in parallel.

== The Original Sin: Variable Capture

The naïve macro of *Common Lisp* `defmacro` is a function from s-expressions to s-expressions, run at compile time:

```lisp
(defmacro swap (a b)
  `(let ((tmp ,a))
     (setf ,a ,b)
     (setf ,b tmp)))
```

The macro is plausible: it expands `(swap x y)` to a let-binding that swaps two places. But consider:

```lisp
(let ((tmp 42))
  (swap tmp other))
```

Expansion:

```lisp
(let ((tmp 42))
  (let ((tmp tmp))      ; the macro's tmp
    (setf tmp other)
    (setf other tmp)))  ; oops — refers to macro's tmp, which is now `other`
```

The macro introduced a binding `tmp` that *captured* the user's `tmp`. The expansion is syntactically valid, but the meaning is wrong: the user's `42` is overwritten by `other`, and the swap is no swap.

The defect is that the macro's `tmp` should have been a *fresh* identifier, distinct from any identifier the user might use. The Common Lisp idiom is to generate a fresh name with `gensym`:

```lisp
(defmacro swap (a b)
  (let ((tmp (gensym)))
    `(let ((,tmp ,a))
       (setf ,a ,b)
       (setf ,b ,tmp))))
```

This works, but the discipline is *manual*: every introduced binding must be a gensym, and every reference to a free identifier from the macro's expansion (say a call to a helper function `f`) is at risk of being shadowed by a user binding of `f`. The macro writer must continuously think about identifiers from two different *worlds* — the macro's world (the *unfolding* context) and the user's world (the *use* context) — and ensure they do not collide.

A macro system that handles this automatically is called *hygienic*. The pursuit of hygiene is the central technical narrative of macros for the past forty years.

== Hygienic Macros: The Painting Algorithm

(Kohlbecker–Friedman–Felleisen–Duba 1986) introduced *syntactic hygiene* and the first algorithm to achieve it. The algorithm — sometimes called the *painting* algorithm — works as follows:

1. Each macro invocation is tagged with a fresh *time-stamp* (a *paint colour*).
2. During expansion, every identifier that *appears in the macro template* is given that time-stamp.
3. Identifiers from the *macro's arguments* (substituted into the template) retain their original colour.
4. When resolving identifier references after expansion, identifiers with the same name but *different colours* are treated as distinct.

The result: identifiers introduced by the macro are guaranteed not to collide with identifiers from the user's context — they have different paint. Identifiers shared between macro and user (intentionally, as when the macro inserts a call to a helper `f` that the user is expected to have in scope) require explicit machinery to *break* hygiene.

In `syntax-rules` notation:

```racket
(define-syntax swap
  (syntax-rules ()
    [(swap a b)
     (let ([tmp a])
       (set! a b)
       (set! b tmp))]))

(let ([tmp 42])
  (swap tmp other))    ; works: macro's tmp is painted, user's tmp is not
```

The `syntax-rules` form is *pattern-based*: the left-hand side is a pattern (with pattern variables `a`, `b`); the right-hand side is a template. The expander does the painting transparently. No `gensym` is needed.

== Syntax-Case and Procedural Macros

`syntax-rules` is limited: the right-hand side is a template, not arbitrary code. Many macros need to inspect the input syntactically, perform compile-time computation, or call helper procedures. (Dybvig–Hieb–Bruggeman 1992) extended the system with `syntax-case`:

```racket
(define-syntax my-if
  (lambda (stx)
    (syntax-case stx ()
      [(_ c t e)
       #'(cond [c t] [else e])])))
```

The `lambda (stx)` says the right-hand side is *arbitrary compile-time code* that takes a syntax object and returns a syntax object. The `#'` reader macro produces a *syntax object* (a piece of code with hygiene metadata attached); a *template* expands pattern variables in scope. Within `syntax-case`, the macro author can:

- Run arbitrary Scheme code at compile time.
- Inspect the input syntactically via `syntax->datum` (extract the s-expression, losing hygiene metadata) and `datum->syntax` (attach a context's hygiene to a raw s-expression).
- Construct new syntax via `with-syntax` (a let-like form that binds pattern variables) and `quasisyntax` (`#`) with `unsyntax` (`#,`).

The `datum->syntax` form is the *escape hatch* for breaking hygiene deliberately. A macro that introduces a binding the user is expected to *see* (the *anaphoric* `it` of Paul Graham's `aif`) calls `(datum->syntax stx 'it)` to attach the user's hygiene context to the introduced name. The user's references to `it` then resolve to the macro's binding. Hygiene is the default; un-hygiene is explicit.

== The Set-of-Scopes Model

Despite syntax-case's success, the *semantics* of hygiene remained subtle. Different implementations disagreed in corner cases, and the algorithms required to maintain hygiene across module boundaries, nested macros, and recursive expansion grew rococo. (Flatt 2016) gave the current synthesis: the *set-of-scopes* model.

The idea is brutally simple:

- Every binding introduction creates a fresh *scope*, an opaque token.
- Every identifier carries a *set* of scopes.
- An identifier *use* refers to a *binding* iff the use's scope set *contains* the binding's scope set.

The expander's job is then to *add scopes* to the identifiers it traverses:

- When the expander enters a `(lambda (x) e)` form, it creates a fresh scope $s$, adds $s$ arrow.r $x$, adds $s$ arrow.r every identifier in $e$, and recursively expands $e$.
- When the expander expands a macro use, it creates a fresh scope $s_"macro"$ for the macro's introduced bindings, adds $s_"macro"$ arrow.r identifiers that come *from the macro template*, and leaves identifiers that come *from the macro's arguments* untouched (those keep their original scope set from the *use* site).

The result: an identifier introduced by the macro has $s_"macro"$ in its scope set; the user's identifier of the same name does not. A binding of the user's identifier is *outside* the macro's introduced binding (it has a *different* scope set), so the two do not capture each other.

The set-of-scopes model is *equational*: two identifiers are *the same* <==> they have the same name and the same scope set. There is no painting, no marking, no renaming — just set membership. The model handles macros, modules, separately compiled code, and recursive expansion uniformly, and it is the basis of the current Racket macro expander.

*Theorem (Hygiene, Flatt 2016).* In the set-of-scopes model, a macro's expansion never causes an identifier from the macro template to capture an identifier from the macro's use site, and never causes an identifier from the use site to capture an identifier from the template.

*Proof sketch.* The macro template's identifiers acquire $s_"macro"$ on expansion; the use-site identifiers do not. A binding introduced by the template binds only identifiers with $s_"macro"$ in their scope set; a use-site reference (without $s_"macro"$) cannot resolve to it. Symmetrically, a use-site binding cannot capture a template reference, because the template reference has $s_"macro"$ extra. $square$

== Phase Separation and Modules

Macros run *before* runtime, but a macro's expansion may itself contain macro uses, and a macro implementation may *itself call* helper functions and use modules. The compile-time computations form their own little universe. (Flatt 2002) introduced *phase separation* to organise this:

- *Phase 0:* run-time code.
- *Phase 1:* code that runs to expand phase-0 code (the body of `define-syntax`).
- *Phase 2:* code that runs to expand phase-1 code (a macro used inside a macro implementation).
- ... and so on, with negative phases for *template* binding.

A module imports definitions *for a particular phase*. `(require racket/list)` imports at phase 0; `(require (for-syntax racket/list))` imports the same module *at phase 1*, making `racket/list`'s functions available *inside* macro implementations. The two imports are *independent* — phase 0 and phase 1 of `racket/list` are different instantiations of the module.

The phase structure makes *separate compilation* of macro-using modules possible: when compiling module $A$ that uses a macro from module $B$, the compiler must instantiate $B$ at phase 1 *only*; it does not need $B$'s phase-0 code. This is the engineering breakthrough that lets Racket's macros scale to large programs.

== MetaML and Multi-Stage Programming

A radically different tradition is *multi-stage programming* (MSP), founded by (Taha–Sheard 1997, 2000) with the language *MetaML*. MSP adds three syntactic constructs:

- *Brackets* `<. e .>` (or `.<e>.`): produce a *code value* representing the expression $e$ without evaluating it. Type: $angle.l tau angle.r$ if $e : tau$.
- *Escape* `~e` (or `.~e`): inside brackets, insert a code value computed by $e$ into the surrounding code. Type: $angle.l tau angle.r$ produces $tau$ in the brackets.
- *Run* `!e` (`.! e`): take a code value and execute it. Type: $angle.l tau angle.r arrow.r tau$.

The classical example is the *power* function staged on its exponent:

```ocaml
let rec power n x =
  if n = 0 then .<1>.
  else .<.~x * .~(power (n-1) x)>.

let power3 = .! .<fun x -> .~(power 3 .<x>.)>.
(* power3 : int -> int, equivalent to fun x -> x * x * x * 1 *)
```

The expression `power 3 .<x>.` runs at the meta-stage and *builds* the AST `x * (x * (x * 1))`; the surrounding `.<fun x -> ...>.` produces a code value for the whole function; `.! ` runs the code at the next stage, producing the runtime function. The optimization-by-staging idiom is: *partial evaluation* with explicit annotations.

MSP differs from Lisp macros in two ways:

1. *Typed.* `<. e .>` is a *type*, $angle.l tau angle.r$ or $"code"(tau)$, and the type system tracks which subexpressions live in which stage. A type error at the *meta* stage is reported as a type error; a code value of the wrong type cannot be inserted.
2. *Hygienic by construction.* Bound variables in code values cannot be captured because the type-checker tracks the *binding environment* of each stage. A code value carries enough information to know which free variables it depends on.

MetaOCaml (Calcagno–Taha–Huang–Leroy 2003; BER MetaOCaml, Kiselyov 2014) is the production implementation, integrated into the OCaml type checker. The brackets become `.< ... >.` and the escape `.~`; the run is `.!`. The compiler generates native code at runtime, often at significant speedup over interpreted alternatives.

A particular concern in MSP is *cross-stage persistence* (CSP): a value of one stage that flows into a code value of a later stage. MetaML handles CSP by *serialising* the value into the code AST; well-typed CSP requires that the value's type be CSP-able (no closures over local stage variables, etc.).

== Template Haskell

Template Haskell (TH, Sheard–Peyton Jones 2002) brings MSP-style staging to Haskell:

- *Quotation* `[| e |]`: produce a value of type `Q Exp` (a *quoted expression* in the `Q` monad).
- *Splice* `$(e)`: insert the result of an expression of type `Q Exp` as a piece of code into the surrounding program.
- *Quotation variants* `[d| ... |]`, `[t| ... |]`, `[p| ... |]` for declarations, types, and patterns.
- *Reify* `reify :: Name -> Q Info`: query the compile-time environment for information about a name (its type, constructors, etc.).

TH's `Q` monad threads two effects: *fresh-name generation* and *I/O during compilation* (controlled, but present). The combination of quotation, splicing, and reification enables generic-programming derivations (`derive Eq`, `derive ToJSON`), DSLs, and bytecode compilation at compile time.

```haskell
makeLenses ''MyRecord    -- TH splice: synthesises lenses for the record
```

Behind the scenes, `makeLenses` is a function `Name -> Q [Dec]` that uses `reify` to inspect the record, generates source for the lenses, and splices it. The user writes one line; the compiler sees a hundred.

TH's hygiene is *opt-in*: `newName :: String -> Q Name` generates a fresh `Name` that does not capture, while `mkName :: String -> Name` produces a name that *does* refer to the binding in scope at the splice site. The two-tier API gives the macro writer explicit control.

F$hash$ has *code quotations* (`<@ ... @>` and `%`) in the same spirit but with the additional twist that the quoted expressions are also available *at runtime* as `Expr` values, enabling LINQ-style provider patterns.

== Scala 3 inline and quoted

Scala 3 has both *inline definitions* (a syntactic facility) and *quoted programming* (a typed multi-stage facility), described in (Stucki–Biboudis–Odersky 2018).

```scala
inline eq.def power(inline n: Int, x: Double): Double =
  inline if n == 0 then 1.0
  else x * power(n - 1, x)
```

The `inline` keyword instructs the compiler arrow.r *inline* the body at the use site; `inline if` and `inline match` are reduced at compile time when the scrutinee is a compile-time constant. The result is a *macro* expressed in ordinary Scala syntax, with the compiler responsible for the staging.

For more sophisticated needs, Scala 3's `quoted` API gives full multi-stage programming:

```scala
import scala.quoted.*

eq.def powerCode(n: Int, x: Expr[Double])(using Quotes): Expr[Double] =
  if n == 0 then '{ 1.0 }
  else '{ ${x} * ${powerCode(n - 1, x)} }
```

The `'{ ... }` is quotation, `${ ... }` is splice, `Expr[T]` is the type of code of type `T`. The `Quotes` capability is required to perform splicing — a more disciplined version of TH's `Q` monad.

== Rust Macros

Rust has *two* macro systems:

1. *Declarative macros* (`macro_rules!`). Pattern-based, hygienic-by-default, no compile-time arbitrary computation. The pattern language is its own little DSL with metavariables (`$x:expr`, `$t:ty`) and repetitions (`$(...)*`).

```rust
macro_rules! vec_of {
    ($($x:expr),*) => {{
        let mut v = Vec::new();
        $( v.push($x); )*
        v
    }};
}

let v = vec_of!(1, 2, 3);
```

Hygiene is *span-based*: each token carries a *span* identifying where it came from (user code vs macro body), and the compiler treats spans like the scopes of the set-of-scopes model. Identifiers introduced inside the macro have a span that the user's identifiers do not, preventing capture.

2. *Procedural macros* (`proc_macro`). The macro is a *Rust function* of type `TokenStream -> TokenStream` invoked at compile time. The function can do arbitrary computation — file I/O, calling out to other tools, parsing the input with `syn`, generating output with `quote`.

```rust
#[proc_macro_derive(MyTrait)]
pub fn derive_my_trait(input: TokenStream) -> TokenStream {
    let ast = syn::parse(input).unwrap();
    let name = &ast.ident;
    let expanded = quote! {
        impl MyTrait for #name {
            fn name(&self) -> &'static str { stringify!(#name) }
        }
    };
    expanded.into()
}
```

Procedural macros come in three flavors: `derive` (extend the `#[derive(...)]` syntax), `attribute` (introduce custom `#[my_attr]` attributes), and `function-like` (use `my_macro!(...)` like a declarative macro). They are *not hygienic by default* — the macro writer is responsible for generating fresh names where needed.

Rust's split — hygienic-by-default declarative macros for the common case, hygiene-as-discipline procedural macros for the heavy lifting — is the inherited wisdom of the Scheme experience.

== C++ Templates and `constexpr`

C++ templates are *accidentally* a Turing-complete compile-time language. Templates were designed for parametric polymorphism (`vector<int>`), but their substitution rules, partial specialisation, and overload resolution conspire to make template instantiation a (deeply painful) functional programming language:

```cpp
template<int N> struct Factorial { static const int value = N * Factorial<N-1>::value; };
template<> struct Factorial<0> { static const int value = 1; };
constexpr int f5 = Factorial<5>::value;   // computed at compile time
```

The discipline survived for two decades because nothing else could do compile-time work in C++. *Modern C++* (since C++11, expanding through C++23) has principled replacements:

- `constexpr` functions: ordinary functions usable at compile time.
- `consteval` functions: *must* be evaluated at compile time.
- Concepts (C++20): named predicates on types, replacing SFINAE.
- Reflection (C++26, in progress): structured access to the type system at compile time.

The `constexpr`/`consteval` machinery is *not* a macro system — it does not let you generate code from arbitrary input syntactically; it only lets you compute values. For code generation, C++ still defers to templates or to external tools.

Rust's `const fn`, Zig's `comptime`, and Nim's `static` are points in the same design space — *compile-time evaluation* without the syntactic-extension power of full macros. Zig's `comptime` is particularly aggressive: any function can be evaluated at compile time, and types are first-class values manipulable by `comptime` code. The boundary between metaprogramming and ordinary programming dissolves.

== Macros as Language Extension: Major Examples

The reason macro systems matter is the *cumulative* power of extensions implemented as macros rather than baked into the compiler. A partial census:

- *Pattern matching* in Lisp, Racket, Clojure (`match`).
- *Async/await*. In Rust, `async fn` is a macro-like desugar to a state machine implementing the `Future` trait; in Scala, async libraries used `@async` as a macro until coroutines were native; in JavaScript, the original ES6 generator-based async was a Babel macro.
- *do-notation* in Haskell, generalised by *applicative-do* and *idiom brackets*.
- *List comprehensions* in Python were originally a macro-style desugar; they are now built-in but the *generator expression* desugar is unchanged.
- *ORM DSLs.* Active Record in Ruby uses `method_missing` (a runtime variant of metaprogramming); Slick in Scala, Diesel in Rust, sqlx in Rust use macros to type-check SQL at compile time.
- *Contract systems.* Racket's `contract-out` is a macro that wraps exported bindings in dynamic contracts.
- *Serialisation*. `derive Serialize/Deserialize` in Rust, `deriving Generic` in Haskell with generic-deriving libraries — all macros.
- *Property-based testing*. QuickCheck-style libraries use macros to generate test boilerplate.
- *Embedded DSLs.* Halide (image processing), Accelerate (parallel arrays), TensorFlow's `tf.function`, JAX's `jit` — all macro-flavoured tools that compile a subset of the host language to a target.

The pattern is consistent: a feature that *looks* like a language extension is implemented as a library + macro, freeing the language designer from baking it in and letting the community iterate.

== Pitfalls

Macros are *brittle* in distinctive ways:

1. *Error messages.* When a macro expansion fails to type-check, the compiler reports the error at the expanded code's location, not the macro use site. The user sees a cryptic message about generated code they never wrote. Tooling responses: *source maps* attach use-site provenance to every emitted token; Racket's error messages thread back through the expansion history.

2. *IDE support.* Auto-completion, jump-to-definition, and refactoring tools must understand the macros to give correct answers. The Racket *DrRacket* IDE re-runs the macro expander to track bindings; *rust-analyzer* has a special macro expansion engine; *IntelliJ Scala* runs the compiler's macro engine. The cost is significant.

3. *Debuggability.* A bug in the macro expansion versus a bug in the macro's output presents very differently. Stepping through generated code is a poor experience.

4. *Phase confusion.* Code that runs at compile time is *not* the same as code that runs at run time. A macro that accidentally tries to compute a runtime value at compile time, or vice versa, produces baffling errors. Racket's phase system makes this explicit and catches the errors early.

5. *Compile time.* Heavy macro use slows the compiler. Template-heavy C++ and macro-heavy Rust can have compile times measured in minutes. Incremental compilation, persistent caches, and parallel expansion are necessary engineering responses.

6. *Cross-cutting concerns.* A macro that subtly changes evaluation order, or that doesn't compose with adjacent macros, breaks reasoning. The discipline is to write macros that *commute* with surrounding code.

== Connection to Partial Evaluation

Macros and *partial evaluation* (PE, Jones–Gomard–Sestoft 1993) solve overlapping problems with different tools. PE takes a program $p$ and some *static* inputs $x$, and produces a *residual* program $p_x$ specialised to those inputs:

$ "PE"(p, x) = p_x quad "such that" quad p(x, y) = p_x(y) "for all dynamic" y $

The *Futamura projections* (Futamura 1971) give the program-construction implications:

- *First projection.* $"PE"("interp", "source")$ specialises an interpreter arrow.r a particular source program, yielding a *compiled* version of the source.
- *Second projection.* $"PE"("PE", "interp")$ specialises the PE arrow.r an interpreter, yielding a *compiler*.
- *Third projection.* $"PE"("PE", "PE")$ specialises the PE to itself, yielding a *compiler generator*.

Multi-stage programming with explicit stage annotations *recovers* the Futamura projections in a typed setting: the brackets tell the system which subexpressions are static, the staging operations correspond to PE's binding-time analysis, and *running* the staged program is exactly specialisation. The MSP discipline is sometimes described as "PE with the programmer doing the binding-time analysis".

Macro systems, by contrast, do *not* in general perform partial evaluation: they perform *syntactic* expansion, but the expanded code is not specialised in the sense of PE. A macro can mimic PE for hand-chosen examples (compute a `Factorial<5>::value`), but it does not do *automatic* specialisation of an arbitrary interpreter.

The convergence point of macros and PE is *typed metaprogramming with reflection*: Template Haskell, Scala 3 quoted, MetaOCaml. In these systems, the macro author can write a partial evaluator *as a macro*, applying it to source programs to produce specialised versions. This is the route by which `staged` libraries in Scala 3, *typed Template Haskell*, and *MetaOCaml's BER* compile DSLs to native code.

== Tools and Implementations

- *Racket.* The gold-standard hygienic macro system; set-of-scopes; phases; modules. The expander itself is written in Racket.
- *Scheme R6RS/R7RS.* `syntax-rules` and `syntax-case` in the standard.
- *Common Lisp.* `defmacro` (unhygienic); the *GENSYM* discipline; *reader macros* for syntactic extension.
- *Clojure.* `defmacro` with *namespace-qualified* symbols providing partial hygiene; *syntax-quote* `${"`"}` auto-resolves symbols.
- *Rust.* `macro_rules!` (declarative, hygienic) and `proc_macro` (procedural, unhygienic but practical).
- *Scala 3.* `inline` + `quoted`; full multi-stage with typed quotation.
- *Haskell.* Template Haskell (procedural); typed Template Haskell (`[|| ... ||]`, `$$( ... )`).
- *OCaml.* MetaOCaml (BER); `ppx` (a separate preprocessor extension system with hooks into the AST).
- *Nim.* AST macros (`macro`); compile-time evaluation (`static`); template macros (`template`); the most expressive macro system in a mainstream language by some measures.
- *Zig.* `comptime` for compile-time evaluation; types as first-class values; minimal AST manipulation.
- *Crystal.* `macro` blocks with template-like syntax; mostly compile-time string-style.
- *Elixir.* `defmacro` (built on Erlang's AST); the entire language has macro-driven sugar.

== An Aside on Reader Macros

*Reader macros* are a Common Lisp facility separate from `defmacro`: they let the user extend the *lexer*, not just the macro expander. A character such as `#` can be associated with a reader-macro function that reads input and returns an s-expression. The pattern enables `${"`"} ... ${"`"}` quasiquotation, `#'fun` for function references, `#(1 2 3)` for vectors. Reader macros are *brittle* and *global* — they change how the entire program is parsed — and have not been imitated in most modern languages.

== Multi-Stage Programming: Soundness

The type system of MetaML guarantees that staged code is *well-typed at every stage*. A staged expression that produces a value of type $angle.l "int" angle.r$ can be *run* arrow.r produce an `int`; the operational semantics of `.!` is to take a code value and run it in the next stage, and the type system's *staging invariants* ensure the code value is closed and type-correct.

*Theorem (Type Safety for MetaML, Taha 2000).* Well-typed MetaML programs do not produce ill-typed code values or unbound-variable errors when run.

*Proof sketch.* By a strengthened progress-and-preservation argument that tracks the *level* of each subexpression (the stage at which it will run). At each stage, the standard arguments apply; the cross-stage operations preserve well-typedness because the type of brackets explicitly records the stage. $square$

The result is *macro hygiene* in a typed setting: capture is impossible because the type system tracks free variables, and ill-typed expansions are caught at the meta-stage rather than at the object stage. This is, in a sense, the *promised land* of macros — a hygienic, typed, expressive metaprogramming facility — and it is realised in MetaOCaml and Scala 3's quoted API.

== Where the Field Stands

Macros are unevenly distributed across modern languages. The *Lisp family* (Racket, Clojure, Common Lisp) has decades-deep macro discipline. The *ML family* (Haskell, Scala, OCaml) has typed multi-stage programming and quotation systems. *Rust* has both declarative and procedural macros, with the latter widely used in the ecosystem. *Python* has no macros, by deliberate language design (Guido's "macros encourage tribal sublanguages"). *Java* and *C$hash$* have *annotation processors* — a constrained form of compile-time code generation. *C++* has templates and `constexpr`. *Go* has neither (and the `go generate` tool is an external preprocessor).

The trend over the past decade has been toward *typed*, *hygienic*, *introspective* macro systems with first-class IDE support. Rust's procedural macros, Scala 3's quoted API, Lean 4's macro system, and Nim's AST macros are all of this character. The C++ reflection proposal (P2996) is moving in the same direction. The endpoint — a language in which compile-time and run-time computations are unified, with first-class code values, typed quotation, and full IDE support — is approached but not yet attained.

The grand bet of macros — that the right way to extend a language is to write a library, not to wait for the next compiler — has been *substantially* vindicated: a non-trivial fraction of the syntax used in every working Racket, Rust, Scala, or Haskell program comes from macros rather than from the core language. The cost has been the disciplined investment in hygiene, phases, and IDE infrastructure that the last forty years have constructed.

The connection to *partial evaluation* and the *Futamura projections* points toward the future: macros that are not merely syntactic transformations but *staged compilers* for embedded languages, capable of fusing user-written DSLs with the host language's optimiser. The work of *Lightweight Modular Staging* (Rompf–Odersky 2010) in Scala and the *futhark* / *accelerate* / *halide* lineage of DSL compilers in Haskell and C++ are concrete instances of this convergence — macro-flavoured tools that produce code competitive with hand-written low-level implementations.

== Hygiene Formalized: The Hilsdale--Friedman Proof

(Hilsdale--Friedman 2000) gave the first published *proof* that the `syntax-rules`
expander satisfies hygiene. The argument works by induction on the derivation of
the expansion relation.

*Setting.* Fix a `syntax-rules` macro $m$ with rules
$[(m p_1) t_1], dots, [(m p_n) t_n]$. Let $E$ be the expansion of a use
$u = (m e_1 dots e_k)$ in the context produced by pattern $p_i$ matching arrow.r
template $t_i$, where $sigma$ is the substitution mapping pattern variables arrow.r
syntax objects (pieces of the input with their use-site hygiene metadata attached).

*Definition.* An identifier $x$ occurring in the output $E$ is called *macro-introduced*
if it originated in $t_i$ (the template), and *use-site* if it originated in some
$e_j$ (an argument). A mixed identifier is one produced by a template escape
`(datum->syntax ...)` and is treated as use-site for hygiene purposes.

*Theorem (Hilsdale--Friedman 2000; paraphrase).* For any `syntax-rules` expansion,
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
arrow.r the continuation of $P$ from a particular `await` point. $square$

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

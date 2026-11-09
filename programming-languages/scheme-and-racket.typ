#import "../template.typ": definition, theorem, xref

= Scheme and Racket <scheme-and-racket>

Scheme is the untyped $lambda$-calculus made into a practical language with as little added as possible.
Sussman and Steele built it in 1975 to understand Hewitt's actor model, and found that their actors and their lexically scoped $lambda$-expressions were the same thing.
Its design principle, stated in the introduction of the Revised Reports, is that "programming languages should be designed not by piling feature on top of feature, but by removing the weaknesses and restrictions that make additional features appear necessary."
Racket grew out of Scheme in the PLT group (Felleisen, Findler, Flatt, Krishnamurthi and others; first released as PLT Scheme in 1995, renamed in 2010) and took the opposite bet on the same foundation: a language for building languages.

*See also:* #xref("programming-languages", "untyped-lambda", label: "The Untyped Lambda Calculus"), #xref("programming-languages", "macros-and-metaprogramming", label: "Macros and Metaprogramming"), #xref("programming-languages", "effects-and-handlers", label: "Effects and Handlers"), #xref("programming-languages", "gradual-and-hybrid", label: "Gradual and Hybrid Type Systems").

This chapter covers the core language and its data, the two features that define Scheme (proper tail calls and first-class continuations), a metacircular evaluator, the standards and implementations, and then Racket's additions: `#lang`, modules and phases, contracts, Typed Racket and the language-oriented style they support.
Macros are covered in depth in the macros chapter and appear here only where Racket depends on them.

== The Core Language

A Scheme program is a sequence of *s-expressions*: atoms (numbers, strings, symbols, booleans, characters) and parenthesised lists.
The same notation writes both programs and data, so a program is a value the language can read, build and transform; this is the property usually called *homoiconicity*.
The evaluation rule is uniform: a symbol is looked up in the lexical environment, a literal evaluates to itself, and a list $(f space a_1 ... a_n)$ evaluates $f$ and the arguments and applies the procedure, *unless* $f$ names a special form or a macro.

A handful of special forms is primitive.

#table(
  columns: (auto, auto),
  [*Form*], [*Meaning*],
  [`(lambda (x ...) body)`], [Procedure with lexically scoped parameters; a closure over the defining environment],
  [`(if c t e)`], [Conditional; only `#f` counts as false],
  [`(quote d)` or `'d`], [The datum `d` itself, not evaluated],
  [`(set! x e)`], [Assignment to an existing binding],
  [`(define x e)`], [Binding at top level or at the start of a body],
  [`(define-syntax k transformer)`], [Macro definition],
)

Everything else is derived.
`let` is an immediately applied `lambda`: `(let ((x 1) (y 2)) (+ x y))` means `((lambda (x y) (+ x y)) 1 2)`.
`let*`, `letrec`, `cond`, `case`, `and`, `or`, `when`, `unless` and `do` are macros over these forms, and the R7RS report gives their definitions as `syntax-rules` macros.
A Scheme compiler therefore only needs to understand a small core, which is why Scheme has been the vehicle for so much compiler research (Steele's Rabbit, Kranz's Orbit, Dybvig's Chez).

*Lexical scope and closures.*
Earlier Lisps used *dynamic* scope: a free variable in a function body referred to the most recent binding at run time, wherever the call came from.
Passing a function as an argument then gave the wrong answer whenever its free variables were rebound between definition and call, the *funarg problem*.
Scheme was the first widely used Lisp with lexical scope throughout: a `lambda` captures the environment where it is written.

```scheme
(define (make-counter)
  (let ((count 0))
    (lambda ()
      (set! count (+ count 1))
      count)))

(define c1 (make-counter))
(define c2 (make-counter))
(c1) (c1) (c2)   ; => 1, 2, 1: each closure owns its own count
```

Closures with assignable variables are enough to build objects, which is how Sussman and Steele recognised that actors were closures.
Common Lisp later adopted lexical scope by default, keeping dynamic scope for `special` variables; Racket provides dynamic binding as *parameters* (`parameterize`), which are thread-safe and interact correctly with continuations.

*One namespace.*
Scheme is a *Lisp-1*: functions and variables share a namespace, so `(f x)` evaluates `f` exactly like `x`, and a procedure is passed as `f`, not `#'f`.
Common Lisp is a Lisp-2 with separate function and value namespaces; Gabriel and Pitman (1988) give the trade-offs, the main one being that a Lisp-1 makes accidental capture of function names by local variables more likely, which is one more reason Scheme needed hygienic macros.

== Data

*Pairs and lists.* `(cons a d)` builds a pair with fields `car` and `cdr`; a list is a chain of pairs ending in the empty list `'()`.
Racket makes pairs immutable (mutable pairs are a separate type, `mcons`), which lets the compiler and the contract system assume that a list stays a list.

*Symbols* are interned strings: two symbols with the same name are the same object, so comparison is a pointer comparison.
They are the natural representation of identifiers when programs are data.

*Equality* comes in three strengths: `eq?` is object identity, `eqv?` additionally compares numbers and characters by value, and `equal?` compares structure recursively.

*The numeric tower.* Numbers are mathematical rather than machine-oriented.
Integers are unbounded, division of integers is exact, and inexactness is a property of the value, not of the type.

```scheme
(expt 2 100)          ; => 1267650600228229401496703205376, exact
(/ 1 3)               ; => 1/3, an exact rational
(+ 1/3 2/3)           ; => 1
(exact->inexact 1/3)  ; => 0.3333333333333333
(sqrt -4)             ; => +2i in implementations with exact complex numbers
```

The cost is that every arithmetic operation must check its operands' representation: a fixnum addition is a tag test, an add, and an overflow check that falls back to bignums.
Chez Scheme and Racket CS keep the fast path to a few instructions by giving fixnums a zero tag in the low bits, so that two tagged fixnums can be added directly, and Racket offers `fx+` and `fl+` for code that can promise small integers or floats.

Strings, characters (full Unicode since R6RS), vectors, bytevectors, hash tables and records complete the data types; records were standardised only in R6RS and R7RS, after decades of ad hoc encodings with vectors.

== Proper Tail Calls

#definition(name: "Tail position")[
An expression is in *tail position* in a `lambda` body if its value is returned directly as the value of the procedure: the body itself, both branches of an `if` in tail position, the last expression of a `begin`, `let` or `cond` clause in tail position, and so on.
A call in tail position is a *tail call*.
]

Scheme requires implementations to be *properly tail recursive*: an unbounded number of active tail calls must run in bounded space.
A tail call does not push a return frame; it reuses the caller's frame and jumps.
Clinger (1998) made the requirement precise as a space-complexity property of an abstract machine, which matters because "does not grow the stack" is not enough: an implementation must also avoid retaining unbounded data in the environments and frames it keeps.

The consequence is that Scheme needs no loop construct.
Iteration is a procedure that calls itself in tail position:

```scheme
(define (sum-list lst)
  (let loop ((xs lst) (acc 0))       ; named let: a local recursive procedure
    (if (null? xs)
        acc
        (loop (cdr xs) (+ acc (car xs))))))   ; tail call: constant space

(define (my-even? n) (if (= n 0) #t (my-odd? (- n 1))))
(define (my-odd? n)  (if (= n 0) #f (my-even? (- n 1))))
(my-even? 100000000)   ; constant space, although the calls alternate
```

The mutual recursion is the case loops cannot express directly: two procedures, possibly in different modules, calling each other indefinitely.
State machines, interpreters dispatching on instructions, and continuation-passing-style code all depend on it.
Steele (1977) argued that a procedure call compiled this way is no more expensive than a `goto`, "debunking the expensive procedure call myth".

*Implementing tail calls on a host that lacks them.*
A native-code compiler emits a jump.
A compiler targeting C, the JVM or JavaScript cannot rely on the host, and has three options:
- *Trampolining*: each procedure returns the next call to make instead of making it, and a driver loop performs the calls.
- *Cheney on the M.T.A.* (Baker 1995, used by Chicken Scheme): compile to C in continuation-passing style, let the C stack grow, and when it reaches a limit, copy live data to the heap and `longjmp` back to the bottom.
- *Whole-program transformation* into a single function with a `switch`, which is how some compilers to JavaScript handle it.

A trampoline in C++ shows the idea.

```cpp
#include <cstdint>

struct Bounce;
using StepFn = Bounce (*)(std::int64_t);

struct Bounce {
    StepFn next;          // nullptr when the computation is finished
    std::int64_t value;   // argument for next, or the final result
};

Bounce even_step(std::int64_t n);
Bounce odd_step(std::int64_t n) {
    if (n == 0) return {nullptr, 0};
    return {even_step, n - 1};
}
Bounce even_step(std::int64_t n) {
    if (n == 0) return {nullptr, 1};
    return {odd_step, n - 1};
}

std::int64_t trampoline(Bounce b) {
    while (b.next != nullptr) b = b.next(b.value);   // one indirect call per bounce
    return b.value;
}
// trampoline({even_step, 1'000'000'000}) == 1, using constant stack space
```

Each bounce costs a return and an indirect call, a few nanoseconds instead of the single jump a native tail call needs, and it defeats inlining across the bounce.
That overhead is why Scheme compilers prefer native code, and why Clang added `[[clang::musttail]]` for interpreters and language runtimes written in C++.

== First-Class Continuations

At any point in a computation, the *continuation* is the rest of the computation: what will happen to the value currently being computed.
`call-with-current-continuation`, usually `call/cc`, passes the current continuation to a procedure as an ordinary value $k$.
Calling $k$ with a value $v$ abandons whatever is running and returns $v$ from the original `call/cc` expression, even if that expression has already returned once.

*Escape.* The simplest use is an early exit.

```scheme
(define (find-first pred lst)
  (call/cc
   (lambda (return)
     (for-each (lambda (x) (when (pred x) (return x))) lst)
     #f)))
```

*Re-entry.* Because a continuation can be invoked after its `call/cc` has returned, continuations can also suspend and resume computations.
A generator turns any procedure that calls a `yield` callback into a stream of values:

```scheme
(define (make-generator walk)       ; walk : (value -> any) -> any
  (define return #f)
  (define resume #f)
  (define (yield v)
    (call/cc (lambda (k) (set! resume k) (return v))))
  (lambda ()
    (call/cc
     (lambda (r)
       (set! return r)
       (if resume
           (resume 'go)
           (begin (walk yield) (return 'done)))))))

(define gen (make-generator (lambda (yield) (for-each yield '(a b c)))))
(gen) (gen) (gen) (gen)   ; => a, b, c, done
```

Each call to `gen` records where to deliver the next value (`return`), and each `yield` records where to resume the walk (`resume`) before jumping out.
The same two-continuation dance implements coroutines, cooperative threads, backtracking search (the `amb` operator), and exceptions; Scheme implementations historically built their thread systems on `call/cc`.

*`dynamic-wind`.* Jumping in and out of a computation breaks code that acquires and releases resources.
`(dynamic-wind before thunk after)` runs `before` every time control enters `thunk`, by any route, and `after` every time it leaves, so that a continuation jump out of a block that opened a file still closes the file, and a jump back in can reopen it.

*Semantics.* `call/cc` has a precise reduction semantics in terms of evaluation contexts: $E["call/cc" space v] arrow.r E[v space k_E]$, where $k_E$ is a value that, when applied to $w$ anywhere, replaces the whole current context by $E[w]$ (#xref("programming-languages", "operational-semantics", label: "Operational Semantics")).
Under Curry–Howard, the type of `call/cc`, $((A arrow.r B) arrow.r A) arrow.r A$, is Peirce's law, so adding `call/cc` to a typed language moves its logic from intuitionistic to classical (Griffin 1990).

*Implementation cost.* If frames live on a stack, capturing a continuation copies the stack, or, in segmented-stack designs such as Chez Scheme's (Hieb, Dybvig and Bruggeman 1990), splits the current stack segment so that capture is constant time and the copy happens lazily on reinstatement.
Compilers that convert the whole program to continuation-passing style get `call/cc` for free, since every continuation is already a closure, but pay for heap-allocating every frame.

== Delimited Continuations

`call/cc` captures the *entire* rest of the program, including the REPL or the operating system's view of it, which makes it hard to compose: a library that uses `call/cc` internally can capture and re-run code that belongs to its caller.
*Delimited* continuations capture only up to a marked point.
Felleisen (1988) introduced the idea with *prompts*; Danvy and Filinski (1990) gave the `shift`/`reset` pair:

```racket
(require racket/control)
(+ 1 (reset (* 2 (shift k (k (k 10))))))   ; => 41
```

`reset` marks a boundary; `shift` captures the continuation up to it, here $k = lambda v . 2 times v$, as an ordinary function that *returns*.
The body applies $k$ twice to get $2 times (2 times 10) = 40$, which becomes the value of the `reset`, and the addition outside is untouched.
Because a delimited continuation returns, it composes like a function, and every computational effect expressible as a monad can be implemented with `shift` and `reset` (Filinski 1994).
This is the same structure as algebraic effect handlers, where `perform` captures the continuation up to the nearest handler (#xref("programming-languages", "effects-and-handlers", label: "Effects and Handlers")).

Racket's control layer is built on tagged prompts (`call-with-continuation-prompt`, `abort-current-continuation`, composable continuations) and *continuation marks*, key-value annotations on frames that a program can inspect without capturing the continuation (Clements, Flatt and Felleisen 2001).
Exceptions, parameters, `dynamic-wind`, the debugger's stack traces and the stepper are all implemented on these primitives.
The broader lesson, argued by Kiselyov among others, is that undelimited `call/cc` is the wrong primitive: it is costly, leaks memory by retaining whole stacks, and is subsumed by delimited control.

== A Metacircular Evaluator

An interpreter for Scheme written in Scheme is short enough to read in one sitting, and it is the central example of _Structure and Interpretation of Computer Programs_ (Abelson and Sussman 1985; 2nd ed. 1996).

```scheme
(define (evaluate expr env)
  (cond ((symbol? expr) (lookup expr env))
        ((not (pair? expr)) expr)                    ; self-evaluating
        ((eq? (car expr) 'quote) (cadr expr))
        ((eq? (car expr) 'if)
         (if (evaluate (cadr expr) env)
             (evaluate (caddr expr) env)
             (evaluate (cadddr expr) env)))
        ((eq? (car expr) 'lambda)
         (list 'closure (cadr expr) (caddr expr) env))
        (else
         (apply-procedure (evaluate (car expr) env)
                          (map (lambda (e) (evaluate e env)) (cdr expr))))))

(define (apply-procedure proc args)
  (if (procedure? proc)
      (apply proc args)                              ; a host primitive
      (evaluate (caddr proc)                         ; closure body
                (cons (map cons (cadr proc) args)    ; new frame
                      (cadddr proc)))))              ; captured environment

(define (lookup name env)
  (if (null? env)
      (error "unbound variable" name)
      (let ((binding (assq name (car env))))
        (if binding (cdr binding) (lookup name (cdr env))))))

(define global-env (list (list (cons '+ +) (cons '* *) (cons '< <))))
(evaluate '((lambda (x) (* x x)) 7) global-env)   ; => 49
```

A closure is a list holding the parameters, the body and the environment at the point of the `lambda`: lexical scope is the decision to store `env` there rather than use the caller's environment in `apply-procedure`.
Reynolds (1972) observed that such a *definitional interpreter* silently inherits features from its host.
This one is properly tail recursive only because the host is (the recursive calls to `evaluate` and `apply-procedure` are in tail position), and its argument evaluation order is whatever order the host's `map` uses, which the Scheme standard leaves unspecified.
Making these properties explicit, by converting the interpreter to continuation-passing style and then defunctionalising the continuations, turns it step by step into an abstract machine; this derivation is the standard bridge from interpreters to compilers.

== Standards and Implementations

#table(
  columns: (auto, auto, auto),
  [*Report*], [*Year*], [*Main additions*],
  [Original memo], [1975], [Lexical scope, closures, tail calls, actors as closures],
  [R2RS, R3RS], [1985, 1986], [Shared standard across MIT, Indiana, Yale and others],
  [R4RS], [1991], [Macro appendix (hygienic macros not yet required)],
  [R5RS], [1998], [`syntax-rules`, `dynamic-wind`, multiple values, `eval`],
  [R6RS], [2007], [Libraries, records, exceptions, Unicode, `syntax-case`; contested],
  [R7RS-small], [2013], [Smaller language close to R5RS, with libraries and records],
)

R6RS grew the standard to several times the size of R5RS, and several major implementations declined to adopt it; the steering committee responded by splitting the work into R7RS-small, a compact core suitable for teaching and embedding, and R7RS-large, an extended library still in progress.
In between, the *Scheme Requests for Implementation* (SRFI) process, running since 1998, standardises libraries one proposal at a time and is how most portable Scheme code shares functionality.

Implementations differ widely in goals:
- *Chez Scheme* (Dybvig, commercial from 1985, open-sourced by Cisco in 2016) is an optimising native-code compiler built from many small passes (the nanopass framework, Keep and Dybvig 2013).
- *Racket* replaced its own virtual machine with Chez Scheme as its back end; Racket CS became the default in Racket 8.0 (2021).
- *Guile* is the GNU extension language; *Chicken* compiles to C with Cheney on the M.T.A.; *Gambit* compiles to C and JavaScript; *MIT/GNU Scheme* descends from the original MIT implementation.

== Racket: A Language for Making Languages

Racket's thesis is that programmers should build a small language for each problem and that the host language should make this cheap (Felleisen et al. 2018).
Every Racket module starts with a `#lang` line that names the language the module is written in:

```racket
#lang racket            ; the full language
#lang typed/racket      ; statically typed Racket
#lang scribble/manual   ; documentation: text-first syntax
#lang datalog           ; a logic language
```

A language is itself a module.
It provides a *reader*, which turns the source text into syntax objects (a language may use non-parenthesised syntax), and the bindings that the module body sees, including `#%module-begin`, which wraps the whole body, and `#%app`, which is inserted around every function application.
Rebinding these hooks with macros changes the meaning of entire programs.
A language whose programs print each top-level expression with its index takes a few lines:

```racket
#lang racket
;; counting.rkt
(provide (except-out (all-from-out racket) #%module-begin)
         (rename-out [counting-module-begin #%module-begin]))

(define-syntax-rule (counting-module-begin form ...)
  (#%module-begin
   (for ([v (list form ...)] [i (in-naturals 1)])
     (printf "~a: ~v\n" i v))))
```

A module written in `#lang s-exp "counting.rkt"` is now a sequence of expressions whose values are numbered.
Production languages built this way include Typed Racket, the teaching languages of _How to Design Programs_, the Scribble documentation system and the Redex semantics workbench; each reuses Racket's expander, module system, compiler and IDE support rather than reimplementing them.

*Modules and phases.* Macros run at compile time and may call ordinary Racket code.
Flatt's module system (2002) separates the code a module needs at run time (phase 0) from the code its macros need at expansion time (phase 1 and higher), instantiating each module separately per phase.
This guarantees that compiling a program gives the same result whether it is compiled all at once or module by module, and that compile-time state never leaks into run time.
The mechanics, and the set-of-scopes hygiene model that underlies Racket's expander since version 6.3, are in #xref("programming-languages", "macros-and-metaprogramming", label: "Macros and Metaprogramming").

== Contracts and Blame

A *contract* is a run-time specification attached to a module boundary.

```racket
#lang racket
(provide (contract-out
          [safe-div (-> number? (and/c number? (not/c zero?)) number?)]
          [apply-twice (-> (-> integer? integer?) integer? integer?)]))

(define (safe-div a b) (/ a b))
(define (apply-twice f x) (f (f x)))
```

First-order contracts such as `number?` are checked immediately.
Higher-order contracts cannot be: whether `f` maps integers to integers is undecidable, so the contract system wraps `f` in a proxy that checks each argument and result when `f` is actually called (Findler and Felleisen 2002).

#theorem(name: "Blame correctness, informal")[
When a contract check fails, the error names the party that broke the agreement: the caller, if it supplied a bad argument, or the module that exported the value, if that value returned a bad result.
For a function argument the roles reverse, so a caller that passes `apply-twice` a function returning strings is blamed, not `apply-twice`.
]

Tracking the swap of roles at each function arrow is what makes the blame accurate, and it is the same contravariance that function subtyping has.
The cost is the proxy: every call through a contract-wrapped function pays for the wrapper and the checks, which is why contracts sit on module boundaries rather than on every internal call.

== Typed Racket

Typed Racket (Tobin-Hochstadt and Felleisen 2008) adds a static type system to Racket without changing how idiomatic Racket code is written.
Its key device is *occurrence typing*: a type predicate in a conditional refines the type of a variable in each branch.

```racket
#lang typed/racket
(: describe (-> (U String Number) String))
(define (describe x)
  (if (string? x)
      (string-append "string of length " (number->string (string-length x)))
      (number->string (+ x 1))))
```

In the true branch `x` has type `String`, in the false branch `Number`, without casts, because the type checker tracks which predicates each test establishes (Tobin-Hochstadt and Felleisen 2010).
TypeScript's narrowing and Python checkers' `isinstance` refinement follow the same idea.

Typed and untyped modules interoperate.
When a typed module exports a value to untyped code, or imports one from it, Typed Racket turns the types at the boundary into contracts, so untyped code cannot violate the typed module's assumptions, and blame points at the untyped side.
This is *sound gradual typing* at module granularity; the cost of the boundary contracts can be large on fine-grained boundaries, and #xref("programming-languages", "gradual-and-hybrid", label: "Gradual and Hybrid Type Systems") discusses the measurements and the alternatives.

== Scheme in the Lisp Family

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Feature*], [*Scheme*], [*Racket*], [*Common Lisp*], [*Clojure*],
  [Namespaces], [Lisp-1], [Lisp-1], [Lisp-2], [Lisp-1],
  [Proper tail calls], [Required], [Yes], [Not required], [No (`recur` for self-loops)],
  [Continuations], [`call/cc`], [Delimited and undelimited], [None], [None],
  [Macros], [Hygienic], [Hygienic, procedural], [`defmacro`, not hygienic], [`defmacro`-style with auto-gensym],
  [Default data], [Mutable pairs], [Immutable pairs], [Mutable], [Persistent immutable],
  [Typical use], [Teaching, embedding, research], [Language design, teaching], [Large systems], [JVM applications],
)

The comparison shows the design pressure on each line.
Clojure runs on the JVM, which has no tail calls, so it offers `recur` for the self-recursive case and trampolines for the rest.
Common Lisp standardised existing practice from large commercial systems in 1984 and 1994, when hygiene was still a research topic.
Scheme kept the core small and the semantics exact, and Racket added the infrastructure for building languages on top of that core.

== Further Reading

Sussman, G. J. and Steele, G. L. (1975). "Scheme: An Interpreter for Extended Lambda Calculus." MIT AI Memo 349. The original report, including the discovery that actors and closures coincide.

Steele, G. L. (1977). "Debunking the 'Expensive Procedure Call' Myth, or, Procedure Call Implementations Considered Harmful, or, Lambda: The Ultimate GOTO." MIT AI Memo 443. The case for compiling tail calls as jumps.

Abelson, H. and Sussman, G. J. (1996). _Structure and Interpretation of Computer Programs_, 2nd ed. MIT Press. Programming with closures and streams, the metacircular evaluator, and compilation to a register machine.

Reynolds, J. C. (1972). "Definitional Interpreters for Higher-Order Programming Languages." ACM Annual Conference. How interpreters inherit evaluation order from their host, and how CPS and defunctionalisation remove the dependence.

Clinger, W. D. (1998). "Proper Tail Recursion and Space Efficiency." PLDI. The precise definition of proper tail recursion as a space-complexity property.

Shinn, A., Cowan, J. and Gleckler, A. A. (eds.) (2013). _Revised^7 Report on the Algorithmic Language Scheme_. The current small standard.

Dybvig, R. K. (2009). _The Scheme Programming Language_, 4th ed. MIT Press. The R6RS language, with `syntax-case`, from the author of Chez Scheme.

Danvy, O. and Filinski, A. (1990). "Abstracting Control." LISP and Functional Programming. Introduces `shift` and `reset` and their CPS semantics.

Findler, R. B. and Felleisen, M. (2002). "Contracts for Higher-Order Functions." ICFP. Higher-order contracts and blame assignment.

Tobin-Hochstadt, S. and Felleisen, M. (2008). "The Design and Implementation of Typed Scheme." POPL. Occurrence typing and sound interoperation between typed and untyped modules.

Felleisen, M., Findler, R. B., Flatt, M., Krishnamurthi, S., Barzilay, E., McCarthy, J. and Tobin-Hochstadt, S. (2018). "A Programmable Programming Language." _Communications of the ACM_ 61(3). The language-oriented programming thesis behind Racket.

Felleisen, M., Findler, R. B., Flatt, M. and Krishnamurthi, S. (2018). _How to Design Programs_, 2nd ed. MIT Press. The design-recipe curriculum and the teaching languages built on Racket.

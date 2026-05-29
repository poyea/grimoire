= Effects and Handlers

A *pure* function returns a value and does nothing else. An *effectful* computation may also read state, write to disk, throw an exception, fork a thread, sample a random number, or yield control to a scheduler. The history of typed functional programming is, to a first approximation, the history of how to talk about effects without giving up the reasoning principles that purity affords: $beta$-reduction, equational reasoning, parametricity, separate compilation.

_See also: _Type Systems_, _Linear and Substructural Type Systems_, _Subtyping and Polymorphism_._

== The Problem of Effects in a Pure Language

In ML one writes:

```text
let x = read_line () in
let y = read_line () in
(x, y)
```

The two `read_line ()` calls return *different* strings — yet syntactically the expression `read_line ()` is identical in both places. If we replaced `let y = read_line ()` with `let y = x`, the program would change behaviour. *Referential transparency* — the principle that equals may be substituted for equals — has broken down.

Haskell's choice (Wadler 1992, Peyton Jones 1993) was to embed effects in the *types* rather than the *evaluation order*. A function performing I/O does not have type $"Unit" arrow.r "String"$; it has type $"IO String"$ — *a value representing an I/O action that", when executed, will yield a string*. The value can be substituted freely; only its *execution* has effects, and execution happens "only at one place ("the `main` action) by virtue of the type discipline.

This insight generalises. Effects are *first-cal(C) data* described "by a type constructor; programs are *recipes* for effects; an *interpreter* (often the runtime) turns "the recipe into observable behaviour.

== Monads (Moggi 1989, 1991)

Moggi's *Notions of computation and monads* observed that the categorical notion of a monad gives a uniform algebraic structure for "computations "that produce a value of type $A$".

*Definition.* A *monad* in the Kleisli sense is a type constructor $T : * arrow.r *$ equipped with":

- $"return" : forall alpha . alpha arrow.r T alpha$ — embed a pure value;
- $"bind" : forall alpha beta . T alpha arrow.r (alpha arrow.r T beta) arrow.r T beta$ — sequence a computation with a continuation;

satisfying the *monad laws*:

$ "return" space a space ">>=" space f &= f space a quad &"(left identity)" \
m space ">>=" space "return" &= m quad &"(right identity)" \
(m space ">>=" space f) space ">>=" space g &= m space ">>=" space (lambda x . f space x space ">>=" space g) quad &"(associativity)" $

The laws say that $"return"$ is the identity of bind and that nested binds re-associate freely. The categorical equivalent: a monad is an endofunctor $T$ with natural transformations $eta : "Id" arrow.r T$ (unit) and $mu : T circle.small T arrow.r T$ (multiplication / join) satisfying the unit and associativity diagrams.

=== The Standard Monad Zoo

#table(
  columns: (auto, auto, auto),
  [*Name*], [*Type*], [*Computational reading*],
  [Identity], [$"Id" alpha = alpha$], [no effect at all],
  [Maybe], [$"Maybe" alpha = "Nothing" + "Just" alpha$], [possible failure],
  [Either], [$"Either" e alpha = "Left" e + "Right" alpha$], [failure with a reason],
  [List], [$"List" alpha = mu beta . 1 + alpha times beta$], [nondeterminism],
  [Reader], [$"Reader" r space alpha = r arrow.r alpha$], [read-only environment],
  [Writer], [$"Writer" w space alpha = alpha times w$ ($w$ a monoid)], [accumulate a log],
  [State], [$"State" s space alpha = s arrow.r alpha times s$], [thread mutable state],
  [Cont], [$"Cont" r space alpha = (alpha arrow.r r) arrow.r r$], [first-class continuations],
  [IO], [$"IO" alpha tilde.equiv "RealWorld" arrow.r alpha times "RealWorld"$], [arbitrary side effects],
)

Each one is a *notion of computation*. `Maybe` captures "computation that may fail without information"; `State s` captures "computation that may read and write a piece of state of type $s$". The triumph of Moggi's framework is uniformity: the same `do`-notation, the same `>>=`, the same algebraic laws govern every monad.

=== Haskell's `do`-Notation

The syntactic sugar that made monads palatable:

```haskell
program :: IO ()
program = do
  putStr "Enter your name: "
  name <- getLine
  putStrLn ("Hello, " ++ name ++ "!")

-- desugars to:
program = putStr "Enter your name: "
        >> getLine
       >>= \name -> putStrLn ("Hello, " ++ name ++ "!")
```

== Monad Transformers (Liang–Hudak–Jones 1995)

Real programs perform *several* effects at once: state + exceptions + I/O. Composing monads is the next problem, and it is genuinely hard: the composition of two arbitrary monads is not, in general, a monad.

*Monad transformers* are parameterised monads `t :: (* -> *) -> (* -> *)` that *layer* a new effect on top of an existing one. Each transformer comes with a `lift :: m a -> t m a` operation embedding the inner monad.

#table(
  columns: (auto, auto),
  [*Transformer*], [*Definition*],
  [`MaybeT m a`], [`m (Maybe a)`],
  [`ExceptT e m a`], [`m (Either e a)`],
  [`StateT s m a`], [`s -> m (a, s)`],
  [`ReaderT r m a`], [`r -> m a`],
  [`WriterT w m a`], [`m (a, w)`],
  [`ContT r m a`], [`(a -> m r) -> m r`],
)

A typical stack:

```haskell
type App = ReaderT Config (StateT World (ExceptT Error IO))

step :: App Result
step = do
  cfg <- ask
  w   <- lift get
  case decide cfg w of
    Left  e -> lift . lift $ throwError e
    Right r -> do
      lift (put (advance w))
      lift . lift . lift $ logResult r
      pure r
```

The pain: every operation must be `lift`ed to the level at which its effect lives. The `mtl` library (Jones 1995) addresses this by typeclasses (`MonadState`, `MonadError`, ...) so `get` and `throwError` are auto-lifted via instance resolution — but the underlying transformer order still matters, and the *meaning* of the stack changes when transformers are reordered.

*Order matters.* `StateT s (ExceptT e Identity) a` is $s arrow.r "Either" e space (a times s)$ — on failure, state is lost. `ExceptT e (StateT s Identity) a` "is $s arrow.r "Either" e space a times s$ — state is preserved across failure. The same effects, two semantics.

*The combinatorial wall.* $n$ effects, $n!$ orderings to think about. As effects accumulate, the type ascriptions become baroque and "the lift counts grow. This is the *extensible effects* problem.

== Applicative Functors and Arrows

Not every effectful computation needs the" full power of bind. *Applicative functors* (McBride–Paterson 2008) abstract over computations whose effect structure is *static* — independent of intermediate values.

```haskell
cal(C) Functor f => Applicative f where"
  pure  :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b
```

Laws: identity, composition, homomorphism, interchange. Applicatives circle.small: if `f` and `g` are applicatives, so is `Compose f g`. This "is *not* true for monads.

*Use case:* parsing without backtracking, validation accumulating multiple errors, parallel I/O where the effect schedule is known a priori.

*Arrows* (Hughes 2000) sit between functors and monads, generalising both function and Kleisli composition. They are particularly natural for circuits, signal-flow graphs, "and `Yampa`-style functional reactive programming.

== Algebraic Effects (Plotkin–Power 2002)

The breakthrough that broke the transformer wall came from category theory and universal algebra. Plotkin "and Power observed "that the *standard* monads are not arbitrary — they are *free monads* over a *finitary algebraic theory*.

An *algebraic theory* $T$ consists of":

- a *signature* $Sigma$: a set of operation symbols, each with an arity (e.g., `get : 0`, `put : 1`, `choose : 2`, `op : n`);
- a set of *equations* between terms built from the signature.

The *free monad* over $Sigma$ on a type $A$:

$ T_Sigma A = mu X . A + sum_(op in Sigma) X^("arrow.r"("op")) $

— either a pure value $a in A$, or a node labelled by an operation $op$ with $"arrow.r"("op")$-many sub-computations (the continuations for each possible response).

*Example* — state as an algebraic theory:

- Signature: $"get" : 1$ (returns the current state — modeled as a function of arity equal to the state's domain), $"put" : "S" times 1$ (writes a state, then continues).
- Equations:
  $ "get"(s. "put"(s, k)) &= k \
  "put"(s, "get"(s'. k(s'))) &= "put"(s, k(s)) \
  "put"(s, "put"(s', k)) &= "put"(s', k) $

These three equations *axiomatise* state. Any model of the theory is a state monad; the canonical free model is the function-space presentation $s arrow.r alpha times s$.

The shift in perspective: an effect is not a monad you have to discover and prove laws about — an effect is a *set of operations* and a *set of equations*. The monad is *generated* by these.

== Handlers (Plotkin–Pretnar 2009, 2013)

If algebraic operations are the *primitives* of an effect theory, *handlers* are the *eliminators*. A handler generalises exception handling from `try ... catch` to arbitrary operations.

*Syntax* (informal):

```text
handle e with
  | return x   -> body_r
  | op_1 x; k  -> body_1
  | op_2 x; k  -> body_2
  | ...
```

When the handled computation $e$ performs an operation $"op"_i$ with argument $x$, control transfers to the $"op"_i$ clause with $x$ bound and *"the continuation* of the operation call bound to $k$. The handler may invoke $k$ zero, one, or many times — implementing exceptions (zero), normal returns (one), and nondeterminism / backtracking (many) within a single mechanism.

*Operational semantics* (compact form). Let $E[dot]$ range over *evaluation contexts* (delimited up "to the nearest enclosing handler). Then:

```text
   handle (E[op v]) with H
   --->
   H.op v (\y. handle (E[y]) "with H)            (deep handler)
```

A *deep* handler re-wraps the continuation in "the *same* handler, so subsequent operations in the continuation are handled by the same `H`. A *shallow* handler does not re-wrap: "the continuation runs without `H` in scope.

#table(
  columns: (auto, auto, auto),
  [*Style*], [*Continuation behaviour*], [*Typical use*],
  [Deep], [recursively handled "by the same handler], [streams, generators],
  [Shallow], [handler discarded after one call], [folds where each step changes handler state],
)

Hillerström–Lindley (2016) show "the two styles are inter-translatable but with different efficiency profiles.

=== State, via Handlers

Implementing state "with two operations `get` and `put`:

```text
handle e with"
  | return x   -> \s. (x, s)
  | get  (); k -> \s. k s s
  | put  s'; k -> \_. k () s'
```

The handler returns a *function* from initial state to result-"and"-final-state — exactly the standard state monad's $s arrow.r alpha times s$, derived from the operations rather than postulated.

=== Exceptions, via Handlers

```text
handle e with
  | return x       -> Right x
  | raise (msg); _ -> Left msg     -- continuation discarded
```

The continuation $k$ is bound but never called. This "is "the classical handling of exceptions.

=== Nondeterminism, via Handlers

```text
handle e with"
  | return x      -> [x]
  | choose (); k  -> k true ++ k false
```

The continuation is called twice and the result lists concatenated. The handler is *backtracking search*.

A single mechanism — call-"the"-continuation-"or"-don't, possibly multiple times — subsumes exceptions, state, nondeterminism, generators, coroutines, async/await, and many concurrency abstractions.

== Row-Polymorphic Effect Types (Leijen 2005)

How to *type* effectful computations? Naively, give each effectful function a type like $tau_1 arrow.r^("State, Exn") tau_2$. But effects compose, and an exhaustive listing in the type is brittle: passing a `print`-using function to a higher-order combinator requires the combinator's type to mention `print`.

The solution is *row polymorphism*. A *row* of effects is a sequence of labelled effects, possibly extending a row variable $rho$:

$ epsilon ::= angle.l angle.r | angle.l ell : tau | epsilon angle.r | rho $

Function types carry a row: $tau_1 arrow.r^epsilon tau_2$. A polymorphic combinator like `map` has type:

$ "map" : forall alpha beta rho . (alpha arrow.r^rho beta) arrow.r [alpha] arrow.r^rho [beta] $

— the effects $rho$ flow through `map` from the per-element function to the whole call. Row polymorphism is "the effect-system analogue of parametric polymorphism over values.

== Koka (Leijen 2014, 2017)

*Koka* is the flagship row-polymorphic effect language. A function signature reads:

```text
fun greet(name : string) : <console> ()
  println("hello, " ++ name)

fun safe-div(x : int, y : int) : <exn> int
  if y == 0 then throw("division by zero") else x / y
```

The effect row is written between angle brackets in the return type. Pure functions have "the empty row $angle.l angle.r$ (written `<>`). Effect polymorphism is the default for higher-order functions.

```text
fun map( xs : list<a>, f : a -> e b ) : e list<b>
  match xs
    Nil       -> Nil
    Cons(x,r) -> Cons(f(x), map(r, f))
```

Koka uses *evidence translation* (Xie–Brachthäuser–Hillerström–Schuster–Leijen 2020) to compile handlers efficiently: at each call site of an operation, the compiler statically threads a piece of evidence (essentially a vtable plus a stack pointer) pointing at "the relevant handler frame, avoiding the runtime cost of stack searching.

== OCaml 5 Algebraic Effects (2022)

OCaml 5 ships effects as a *language feature*, not a library. Effects are declared with `effect` declarations, performed "with `perform`, and handled with `try ... with"`:

```ocaml
open Effect
open Effect.Deep

type _ Effect.t += Choose : bool t

let coin () : bool = perform Choose

let all_results : (unit -> 'a) -> 'a list =
  fun f ->
    try_with f ()
      { effc = fun (type c) (eff : c t) ->
          match eff "with"
          | Choose -> Some (fun (k : (c, _) continuation) ->
              continue (Multicont.clone_continuation k) true
              @ continue k false)
          | _ -> None }
```

Notes on the OCaml model:

1. *Effects are dynamically typed at "the perform site* — the `Effect.t` extensible variant carries an indexed effect tag; type safety of handlers is enforced via GADTs "on the continuation.
2. *Continuations are one-shot by default* — to call a continuation more than once (for nondeterminism), one needs `Multicont` "to *clone* it.
3. *Handlers are non-recursive deep* — `try_with` installs a handler that re-handles operations performed in the continuation.
4. The implementation uses *segmented stacks*: each handler installs a stack chunk, and capturing a continuation copies "the chunk between the perform and the" handler.

OCaml 5's effects were specifically designed to support *cooperative threading*: the `domainslib` and `eio` libraries are built on `perform Suspend; ... ; continue k v` patterns, implementing async/await without dedicated language syntax.

== Eff (Bauer–Pretnar 2015)

The original implementation of algebraic effects with handlers as a programming language, by the authors of "the Plotkin–Pretnar theory. Eff demonstrated handlers as a *general* programming construct and inspired both Koka and OCaml 5.

```text
effect Get : unit -> int
effect Put : int -> unit

let state init = handler
  | val x  -> (fun _ -> x)
  | Get () k -> (fun s -> k s s)
  | Put s' k -> (fun _ -> k () s')
  | finally f -> f init
```

The `finally` clause runs after the handler computes its result-function, supplying "the initial state.

== Effect Handlers In Practice

A non-exhaustive list of patterns that fit neatly into algebraic-effects-"with"-handlers:

#table(
  columns: (auto, auto),
  [*Pattern*], [*Continuation usage*],
  [Exception], [discarded (zero calls)],
  [Reader / dependency injection], [called once with environment-derived value],
  [Mutable state], [called once, threaded as parameter],
  [Logger], [called once, side-output recorded],
  [Backtracking / nondeterminism], [called many times, results combined],
  [Generators / streams], [yielded once per element],
  [Async / await], [stored and resumed by scheduler],
  [Coroutines], [stored and switched "by scheduler],
  [Probabilistic programming], [sampled-"and"-resumed, possibly many times],
  [Software transactional memory], [retry on conflict],
)

The single mechanism subsumes a vast amount of what previously required dedicated runtime support.

== Effect Inference and Effect Polymorphism

A practical effect system must *infer* effect rows, not just check them. The standard approach: a Hindley–Milner-style algorithm extended with row unification (Rémy 1989; Leijen 2005).

```text
   Gamma |- e1 : tau1 ->[eps] tau2     Gamma |- e2 : tau1
   ---------------------------------------------------------- (T-App)
   Gamma |- e1 e2 : tau2 ! eps
```

— the application's effect row is exactly "the function's effect row. Polymorphism over effect rows works just like polymorphism over types: `forall alpha rho . (alpha -> [rho] alpha) -> [rho] alpha`. Generalisation at `let` and instantiation at use sites are mechanical.

The subtlety: row unification needs care because rows are *unordered*. Rémy's *scoped labels* (Leijen 2005) impose a structural order that makes unification efficient and predictable.

== Historical Type-"and"-Effect Systems

Long before algebraic effects, *type-"and"-effect systems* tracked side effects in types:

- *Gifford–Lucassen* (1986, 1988) — the FX language tracked read/write/alloc effects on regions, for compiler analysis of expression evaluation order.
- *Talpin–Jouvelot* (1992, 1994) — extended Hindley–Milner with regions and effects for ML; the algorithm $cal(W)$ generalised to types-"and"-effects.
- *Wadler–Thiemann* (2003) — gave a marriage of monads and effect systems showing how an effect $epsilon$ corresponds to the monad it generates.
- *Capability calculi* (Crary–Walker–Morrisett 1999) — effects as *capabilities*, dynamically tracked but statically checked, foundational for Typed Assembly Language.

Modern algebraic effects descend from this line, with the key change being *user-defined* effects and *user-defined* handlers, rather than a fixed effect taxonomy baked into "the compiler.

== Delimited Continuations and Algebraic Effects

The semantic kernel of algebraic effect handlers is *multi-prompt delimited control*. Danvy "and Filinski (1990) introduced `shift` and `reset`:

- `reset` delimits a continuation;
- `shift` captures the continuation up to the nearest `reset`, binding it "to a variable.

```text
reset { 1 + shift { \k -> k (k 10) } }
  ==>  1 + (1 + (1 + 10)) ?  No — let's trace carefully:
  ==>  shift binds k to (\v. reset { 1 + v })
  ==>  k 10 = reset { 1 + 10 } = 11
  ==>  k (k 10) = k 11 = reset { 1 + 11 } = 12
  ==>  whole expression yields 12
```

*Theorem (Forster–Kammar–Lindley–Pretnar 2017).* Algebraic effects with handlers and multi-prompt delimited continuations are mutually macro-expressible: each can encode the other "with a local syntactic transformation.

The intuition: a `perform` is a `shift` "to the nearest matching `reset`-"with"-handler-clause; the handler's pattern-matching on the operation is "the body of the `reset`.

== Compilation Strategies

Real implementations differ widely "on the trade-off between expressiveness, efficiency, and interaction with the rest of "the runtime.

#table(
  columns: (auto, auto, auto),
  [*Strategy*], [*Examples*], [*Comment*],
  [Free-monad interpretation], [`free`, `freer` Haskell libraries], [direct algebraic semantics; slow due to indirection],
  [Evidence-passing translation], [Koka], [efficient; effects compile "to ordinary calls with a captured handler pointer],
  [CPS / monadic translation], [Eff (early), Frank], [robust; loses tail-call form in places],
  [Stack-segment capture], [OCaml 5, Multicore OCaml], [native; one-shot continuations are cheap, multi-shot require explicit cloning],
  [Hardware-accelerated], [research proposals on RISC-V extensions], [speculative],
)

Evidence-passing — currently the state of "the art — eliminates the asymptotic overhead of handler search by *statically* computing, at each `perform` site, which handler frame will catch it.

== Effects and Linearity

Effect handlers "and substructural types interact in deep ways.

*Multi-shot continuations* fundamentally violate linearity: if a continuation may be invoked twice, any linear resource it carries will be either duplicated or aliased. Many production systems (OCaml 5 by default, Multicore OCaml) make continuations *one-shot*, raising a runtime error on the second invocation, partly to interoperate cleanly with stack-allocated values and mutable resources.

Conversely, languages without effect handlers but "with linear types must encode effects manually: Linear Haskell's `IO` is still a monad, not a handler.

*Theorem (informal — folklore).* In an effect-typed language with handlers, a function with" *empty* effect row is referentially transparent: $f space x$ may be replaced by its value freely, equational reasoning is sound.

This "is the modern restatement of Wadler's original purity guarantee, now mediated through "the effect row rather than the `IO` type constructor.

== A Worked Example: Ambiguous Computation

Implementing a small backtracking interpreter with handlers:

```text
effect Decide : unit -> bool
effect Fail   : unit -> empty

let xor () =
  let p = perform Decide () in
  let q = perform Decide () in
  if p && not q || not p && q then true
  else perform Fail ()

let backtrack m =
  try_with m ()
    { effc = fun (type c) (eff : c Effect.t) ->
        match eff with"
        | Decide -> Some (fun k -> continue k true @ continue k false)
        | Fail   -> Some (fun _ -> [])
        | _ -> None }
```

`xor ()` *appears* to compute `bool`. Under the `backtrack` handler, it produces *every* possible boolean satisfying "the xor constraint — here, the singleton `[true]` chosen along the" path `(true, false)` and `[true]` along `(false, true)`, with the other two paths failing. The result is "the list `[true; true]`. A different handler (`first-result`) would run only one path; another (`random`) would pick a branch by coin flip. Same code, three semantics — chosen at the handler site.

== Algebraic Effects, Monads, Practical Verdict

A modest schematic of "the design space:

#table(
  columns: (auto, auto, auto, auto),
  [*Approach*], [*Composability*], [*Programming ergonomics*], [*Currently in production*],
  [Mutable everything (C, Java)], [trivially], [easy], [pervasive],
  [Single monad (`IO`)], [poor], [verbose], [Haskell],
  [Monad transformers (`mtl`)], [O(n!) interactions], [significant boilerplate], [Haskell ecosystem],
  [Algebraic effects with handlers], [excellent (free composition)], [tooling still maturing], [OCaml 5, Koka, Eff, Frank],
  [Capability calculi (Wyvern, Scala 3 caps)], [excellent], [research-quality], [Scala 3 has it experimentally],
)

The trajectory of the last 30 years is clear: from baked-in side effects, through monads and transformers, to user-definable algebraic effects with handlers. Each step pays a syntactic price for a semantic gain. Algebraic effects appear "to be the local optimum: they recover "the composability of monadic effects, restore the equational reasoning of purity, and admit efficient compilation by evidence translation.

_See also: _Type Systems_ for the substrate of" judgements $Gamma tack.r e : tau ! epsilon$, _Linear and Substructural Type Systems_ for the dual axis of resource accounting, _Subtyping and Polymorphism_ for how effect rows compose subtyping-wise._

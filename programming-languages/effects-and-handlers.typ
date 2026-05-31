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

Haskell's choice (Wadler 1992, Peyton Jones 1993) was to embed effects in the *types* rather than the *evaluation order*. A function performing I/O does not have type $"Unit" arrow.r "String"$; it has type $"IO String"$ — *a value representing an I/O action that, when executed, will yield a string*. The value can be substituted freely; only its *execution* has effects, and execution happens only at one place (the `main` action) by virtue of the type discipline.

This insight generalises. Effects are *first-cal(C) data* described by a type constructor; programs are *recipes* for effects; an *interpreter* (often the runtime) turns the recipe into observable behaviour.

== Monads (Moggi 1989, 1991)

Moggi's *Notions of computation and monads* observed that the categorical notion of a monad gives a uniform algebraic structure for computations that produce a value of type $A$.

*Definition.* A *monad* in the Kleisli sense is a type constructor $T : * arrow.r *$ equipped with:

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

*Order matters.* `StateT s (ExceptT e Identity) a` is $s arrow.r "Either" e space (a times s)$ — on failure, state is lost. `ExceptT e (StateT s Identity) a` is $s arrow.r "Either" e space a times s$ — state is preserved across failure. The same effects, two semantics.

*The combinatorial wall.* $n$ effects, $n!$ orderings to think about. As effects accumulate, the type ascriptions become baroque and the lift counts grow. This is the *extensible effects* problem.

== Applicative Functors and Arrows

Not every effectful computation needs the full power of bind. *Applicative functors* (McBride–Paterson 2008) abstract over computations whose effect structure is *static* — independent of intermediate values.

```haskell
cal(C) Functor f => Applicative f where"
  pure  :: a -> f a
  (<*>) :: f (a -> b) -> f a -> f b
```

Laws: identity, composition, homomorphism, interchange. Applicatives circle.small: if `f` and `g` are applicatives, so is `Compose f g`. This is *not* true for monads.

*Use case:* parsing without backtracking, validation accumulating multiple errors, parallel I/O where the effect schedule is known a priori.

*Arrows* (Hughes 2000) sit between functors and monads, generalising both function and Kleisli composition. They are particularly natural for circuits, signal-flow graphs, and `Yampa`-style functional reactive programming.

== Algebraic Effects (Plotkin–Power 2002)

The breakthrough that broke the transformer wall came from category theory and universal algebra. Plotkin and Power observed that the *standard* monads are not arbitrary — they are *free monads* over a *finitary algebraic theory*.

An *algebraic theory* $T$ consists of:

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

When the handled computation $e$ performs an operation $"op"_i$ with argument $x$, control transfers to the $"op"_i$ clause with $x$ bound and *the continuation* of the operation call bound arrow.r $k$. The handler may invoke $k$ zero, one, or many times — implementing exceptions (zero), normal returns (one), and nondeterminism / backtracking (many) within a single mechanism.

*Operational semantics* (compact form). Let $E[dot]$ range over *evaluation contexts* (delimited up to the nearest enclosing handler). Then:

```text
   handle (E[op v]) with H
   --->
   H.op v (y. handle (E[y]) "with H)            (deep handler)
```

A *deep* handler re-wraps the continuation in the *same* handler, so subsequent operations in the continuation are handled by the same `H`. A *shallow* handler does not re-wrap: the continuation runs without `H` in scope.

#table(
  columns: (auto, auto, auto),
  [*Style*], [*Continuation behaviour*], [*Typical use*],
  [Deep], [recursively handled by the same handler], [streams, generators],
  [Shallow], [handler discarded after one call], [folds where each step changes handler state],
)

Hillerström–Lindley (2016) show the two styles are inter-translatable but with different efficiency profiles.

=== State, via Handlers

Implementing state with two operations `get` and `put`:

```text
handle e with"
  | return x   -> s. (x, s)
  | get  (); k -> s. k s s
  | put  s'; k -> \_. k () s'
```

The handler returns a *function* from initial state to result-and-final-state — exactly the standard state monad's $s arrow.r alpha times s$, derived from the operations rather than postulated.

=== Exceptions, via Handlers

```text
handle e with
  | return x       -> Right x
  | raise (msg); _ -> Left msg     -- continuation discarded
```

The continuation $k$ is bound but never called. This is the classical handling of exceptions.

=== Nondeterminism, via Handlers

```text
handle e with"
  | return x      -> [x]
  | choose (); k  -> k true ++ k false
```

The continuation is called twice and the result lists concatenated. The handler is *backtracking search*.

A single mechanism — call-the-continuation-or-don't, possibly multiple times — subsumes exceptions, state, nondeterminism, generators, coroutines, async/await, and many concurrency abstractions.

== Row-Polymorphic Effect Types (Leijen 2005)

How arrow.r *type* effectful computations? Naively, give each effectful function a type like $tau_1 arrow.r^("State, Exn") tau_2$. But effects compose, and an exhaustive listing in the type is brittle: passing a `print`-using function to a higher-order combinator requires the combinator's type to mention `print`.

The solution is *row polymorphism*. A *row* of effects is a sequence of labelled effects, possibly extending a row variable $rho$:

$ epsilon ::= angle.l angle.r | angle.l ell : tau | epsilon angle.r | rho $

Function types carry a row: $tau_1 arrow.r^epsilon tau_2$. A polymorphic combinator like `map` has type:

$ "map" : forall alpha beta rho . (alpha arrow.r^rho beta) arrow.r [alpha] arrow.r^rho [beta] $

— the effects $rho$ flow through `map` from the per-element function to the whole call. Row polymorphism is the effect-system analogue of parametric polymorphism over values.

== Koka (Leijen 2014, 2017)

*Koka* is the flagship row-polymorphic effect language. A function signature reads:

```text
fun greet(name : string) : <console> ()
  println("hello, " ++ name)

fun safe-div(x : int, y : int) : <exn> int
  if y == 0 then throw("division by zero") else x / y
```

The effect row is written between angle brackets in the return type. Pure functions have the empty row $angle.l angle.r$ (written `<>`). Effect polymorphism is the default for higher-order functions.

```text
fun map( xs : list<a>, f : a -> e b ) : e list<b>
  match xs
    Nil       -> Nil
    Cons(x,r) -> Cons(f(x), map(r, f))
```

Koka uses *evidence translation* (Xie–Brachthäuser–Hillerström–Schuster–Leijen 2020) arrow.r compile handlers efficiently: at each call site of an operation, the compiler statically threads a piece of evidence (essentially a vtable plus a stack pointer) pointing at the relevant handler frame, avoiding the runtime cost of stack searching.

== OCaml 5 Algebraic Effects (2022)

OCaml 5 ships effects as a *language feature*, not a library. Effects are declared with `effect` declarations, performed with `perform`, and handled with `try ... with`:

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

1. *Effects are dynamically typed at the perform site* — the `Effect.t` extensible variant carries an indexed effect tag; type safety of handlers is enforced via GADTs on the continuation.
2. *Continuations are one-shot by default* — arrow.r call a continuation more than once (for nondeterminism), one needs `Multicont` arrow.r *clone* it.
3. *Handlers are non-recursive deep* — `try_with` installs a handler that re-handles operations performed in the continuation.
4. The implementation uses *segmented stacks*: each handler installs a stack chunk, and capturing a continuation copies the chunk between the perform and the handler.

OCaml 5's effects were specifically designed to support *cooperative threading*: the `domainslib` and `eio` libraries are built on `perform Suspend; ... ; continue k v` patterns, implementing async/await without dedicated language syntax.

== Eff (Bauer–Pretnar 2015)

The original implementation of algebraic effects with handlers as a programming language, by the authors of the Plotkin–Pretnar theory. Eff demonstrated handlers as a *general* programming construct and inspired both Koka and OCaml 5.

```text
effect Get : unit -> int
effect Put : int -> unit

let state init = handler
  | val x  -> (fun _ -> x)
  | Get () k -> (fun s -> k s s)
  | Put s' k -> (fun _ -> k () s')
  | finally f -> f init
```

The `finally` clause runs after the handler computes its result-function, supplying the initial state.

== Effect Handlers In Practice

A non-exhaustive list of patterns that fit neatly into algebraic-effects-with-handlers:

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
  [Coroutines], [stored and switched by scheduler],
  [Probabilistic programming], [sampled-and-resumed, possibly many times],
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

— the application's effect row is exactly the function's effect row. Polymorphism over effect rows works just like polymorphism over types: `forall alpha rho . (alpha -> [rho] alpha) -> [rho] alpha`. Generalisation at `let` and instantiation at use sites are mechanical.

The subtlety: row unification needs care because rows are *unordered*. Rémy's *scoped labels* (Leijen 2005) impose a structural order that makes unification efficient and predictable.

== Historical Type-and-Effect Systems

Long before algebraic effects, *type-and-effect systems* tracked side effects in types:

- *Gifford–Lucassen* (1986, 1988) — the FX language tracked read/write/alloc effects on regions, for compiler analysis of expression evaluation order.
- *Talpin–Jouvelot* (1992, 1994) — extended Hindley–Milner with regions and effects for ML; the algorithm $cal(W)$ generalised to types-and-effects.
- *Wadler–Thiemann* (2003) — gave a marriage of monads and effect systems showing how an effect $epsilon$ corresponds to the monad it generates.
- *Capability calculi* (Crary–Walker–Morrisett 1999) — effects as *capabilities*, dynamically tracked but statically checked, foundational for Typed Assembly Language.

Modern algebraic effects descend from this line, with the key change being *user-defined* effects and *user-defined* handlers, rather than a fixed effect taxonomy baked into the compiler.

== Delimited Continuations and Algebraic Effects

The semantic kernel of algebraic effect handlers is *multi-prompt delimited control*. Danvy and Filinski (1990) introduced `shift` and `reset`:

- `reset` delimits a continuation;
- `shift` captures the continuation up to the nearest `reset`, binding it to a variable.

```text
reset { 1 + shift { k -> k (k 10) } }
  ==>  1 + (1 + (1 + 10)) ?  No — let's trace carefully:
  ==>  shift binds k arrow.r (v. reset { 1 + v })
  ==>  k 10 = reset { 1 + 10 } = 11
  ==>  k (k 10) = k 11 = reset { 1 + 11 } = 12
  ==>  whole expression yields 12
```

*Theorem (Forster–Kammar–Lindley–Pretnar 2017).* Algebraic effects with handlers and multi-prompt delimited continuations are mutually macro-expressible: each can encode the other with a local syntactic transformation.

The intuition: a `perform` is a `shift` arrow.r the nearest matching `reset`-with-handler-clause; the handler's pattern-matching on the operation is the body of the `reset`.

== Compilation Strategies

Real implementations differ widely on the trade-off between expressiveness, efficiency, and interaction with the rest of the runtime.

#table(
  columns: (auto, auto, auto),
  [*Strategy*], [*Examples*], [*Comment*],
  [Free-monad interpretation], [`free`, `freer` Haskell libraries], [direct algebraic semantics; slow due to indirection],
  [Evidence-passing translation], [Koka], [efficient; effects compile to ordinary calls with a captured handler pointer],
  [CPS / monadic translation], [Eff (early), Frank], [robust; loses tail-call form in places],
  [Stack-segment capture], [OCaml 5, Multicore OCaml], [native; one-shot continuations are cheap, multi-shot require explicit cloning],
  [Hardware-accelerated], [research proposals on RISC-V extensions], [speculative],
)

Evidence-passing — currently the state of the art — eliminates the asymptotic overhead of handler search by *statically* computing, at each `perform` site, which handler frame will catch it.

== Effects and Linearity

Effect handlers and substructural types interact in deep ways.

*Multi-shot continuations* fundamentally violate linearity: if a continuation may be invoked twice, any linear resource it carries will be either duplicated or aliased. Many production systems (OCaml 5 by default, Multicore OCaml) make continuations *one-shot*, raising a runtime error on the second invocation, partly to interoperate cleanly with stack-allocated values and mutable resources.

Conversely, languages without effect handlers but with linear types must encode effects manually: Linear Haskell's `IO` is still a monad, not a handler.

*Theorem (informal — folklore).* In an effect-typed language with handlers, a function with *empty* effect row is referentially transparent: $f space x$ may be replaced by its value freely, equational reasoning is sound.

This is the modern restatement of Wadler's original purity guarantee, now mediated through the effect row rather than the `IO` type constructor.

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

`xor ()` *appears* arrow.r compute `bool`. Under the `backtrack` handler, it produces *every* possible boolean satisfying the xor constraint — here, the singleton `[true]` chosen along the path `(true, false)` and `[true]` along `(false, true)`, with the other two paths failing. The result is the list `[true; true]`. A different handler (`first-result`) would run only one path; another (`random`) would pick a branch by coin flip. Same code, three semantics — chosen at the handler site.

== Algebraic Effects, Monads, Practical Verdict

A modest schematic of the design space:

#table(
  columns: (auto, auto, auto, auto),
  [*Approach*], [*Composability*], [*Programming ergonomics*], [*Currently in production*],
  [Mutable everything (C, Java)], [trivially], [easy], [pervasive],
  [Single monad (`IO`)], [poor], [verbose], [Haskell],
  [Monad transformers (`mtl`)], [O(n!) interactions], [significant boilerplate], [Haskell ecosystem],
  [Algebraic effects with handlers], [excellent (free composition)], [tooling still maturing], [OCaml 5, Koka, Eff, Frank],
  [Capability calculi (Wyvern, Scala 3 caps)], [excellent], [research-quality], [Scala 3 has it experimentally],
)

The trajectory of the last 30 years is clear: from baked-in side effects, through monads and transformers, arrow.r user-definable algebraic effects with handlers. Each step pays a syntactic price for a semantic gain. Algebraic effects appear to be the local optimum: they recover the composability of monadic effects, restore the equational reasoning of purity, and admit efficient compilation by evidence translation.

_See also: _Type Systems_ for the substrate of judgements $Gamma tack.r e : tau ! epsilon$, _Linear and Substructural Type Systems_ for the dual axis of resource accounting, _Subtyping and Polymorphism_ for how effect rows compose subtyping-wise._

== Free Monads in Detail

The *free monad* over a functor $F$ is the initial algebra of the monad equations for $F$-shaped computations. Concretely, in Haskell:

```haskell
data Free f a
  = Pure a
  | Free (f (Free f a))
```

Every `Free f a` is either a pure result or an `f`-shaped tree of continuations. The monad instance:

```haskell
instance Functor f => Monad (Free f) where
  return = Pure
  Pure x   >>= k = k x
  Free ffx >>= k = Free (fmap (>>= k) ffx)
```

The bind "pushes" the continuation $k$ down into every leaf of the tree, grafting $k$'s tree onto each pure value. The monad laws hold because grafting distributes over tree structure.

*Universality.* For any monad $M$ and any natural transformation $iota : F arrow.r M$, there is a unique monad morphism $"interpret"_iota : "Free" F arrow.r M$:

```haskell
interpret :: (Functor f, Monad m) => (forall a. f a -> m a) -> Free f a -> m a
interpret alg (Pure x)   = return x
interpret alg (Free ffx) = alg ffx >>= interpret alg
```

This is the *universal property* of the free monad: "Free f" is the *initial* monad equipped with an operation $F arrow.r M$. An *algebra* for `Free f` is exactly a *handler*: a specification of how to interpret each $F$-shaped operation node.

*Freer monads.* The *freer monad* (Kiselyov–Ishii 2015) generalises to a non-functor $F$ by adding explicit continuation storage:

```haskell
data Freer f a where
  Pure :: a -> Freer f a
  Impure :: f x -> (x -> Freer f a) -> Freer f a
```

The `Impure` constructor stores the operation $"op" : f x$ and a continuation $k : x arrow.r "Freer" f a$ separately. This avoids the `Functor` requirement and makes the continuation explicit, enabling *efficient fusion* by a sequence of handlers without intermediate tree allocations.

The *extensible effects* library (Kiselyov–Sabry–Swords 2013) uses the Freer monad with an *open union* of effects:

```haskell
data Union (r :: [* -> *]) a where
  UNow  :: t a -> Union (t ': r) a
  UNext :: Union r a -> Union (t ': r) a

type Eff r a = Freer (Union r) a
```

Each effect in the list `r` corresponds to one `UNow` injection. Handlers peel off the head of the list, reducing the effect set:

```haskell
run :: Eff '[] a -> a
run (Pure x) = x

handleRelay :: (a -> Eff r b) -> (forall v. t v -> (v -> Eff r b) -> Eff r b)
            -> Eff (t ': r) a -> Eff r b
```

== Algebraic Effects: Theory Presentation

An *algebraic effect theory* $cal(T)$ consists of:

1. A *signature* $Sigma$: a set of operation symbols, each with a *parameter type* $P_op$ and a *return type* $A_op$. For example:
   - $"Get" : P = 1, A = S$ (get the current state, no parameter)
   - $"Put" : P = S, A = 1$ (set the state, returns unit)
   - $"Choose" : P = 1, A = 2$ (boolean nondeterminism)
   - $"Raise"(E) : P = E, A = 0$ (raise an exception, never returns)

2. A set of *equations* between terms built from the signature and ordinary function application.

The *term model* over signature $Sigma$ and value type $A$:

$ T_Sigma(A) = mu X . A + sum_{op in Sigma} P_op arrow.r (A_op arrow.r X) $

— either a *return* of a value, or an *operation application* with a continuation. This is the free $Sigma$-algebra on $A$.

*Example — state.* With $"Get" : 1 arrow.r S$ and $"Put" : S arrow.r 1$:

$ T_{"Get","Put"}(A) = mu X . A + (S arrow.r X) + (S times X) $

The equations:

$ "Get"(s |-> "Put"(s, k)) &= k quad "(get then put the same state = identity)" \
"Put"(s, "Get"(s' |-> k(s'))) &= "Put"(s, k(s)) quad "(put then get = put then use the written state)" \
"Put"(s, "Put"(s', k)) &= "Put"(s', k) quad "(second put wins)" $

A *model* of the theory is a set $X$ with an $S$-indexed family of maps $X^S arrow.r X$ (for Get) and $S times X arrow.r X$ (for Put) satisfying the equations. The *initial model* (free model) is $T_Sigma(A)$; the *canonical model* is the state monad $S arrow.r A times S$, and the unique morphism from the free model to the state monad is precisely the standard interpreter.

*Free model construction.* Given $e : T_Sigma(A)$ and an initial state $s_0 : S$:

```text
interpret(return v, s)    = (v, s)
interpret(Get(k), s)      = interpret(k(s), s)
interpret(Put(s', k), s)  = interpret(k(()), s')
```

This gives a morphism of $Sigma$-algebras. Uniqueness follows because any other morphism must satisfy the same equations — there is no choice.

*Theorem (Plotkin–Power 2001).* Every finitary monad on $"Set"$ is the free monad of some algebraic theory. The theory is given by: operations = $Sigma$-algebra generators, equations = the monad's equations.

*Corollary.* Exception, state, reader, writer, nondeterminism, and finite distribution monads are all algebraic. The continuation monad $"Cont" r$ is *not* algebraic for non-trivial $r$ (it cannot be presented by a finitary theory).

== Plotkin–Pretnar Handler Semantics: Deep vs Shallow

We work in a core language $lambda_"eff"$ with the following syntax:

$ v ::= x | lambda x . e | "Inl" v | "Inr" v | () \
e ::= v | e_1 e_2 | "op"(v; y . e) | "return" v | "handle" e "with" h \
h ::= {"return" x |-> e_r; "op"_i(x; k) |-> e_i}_{i in I} $

The handler $h$ describes what to do for the `return` case and for each operation case, with $k$ bound to the delimited continuation.

=== Deep Handler Semantics

*Big-step operational rules:*

```text
  e -->* return v
  ----------------------- (H-Return)
  handle e with h -->* [v/x] e_r

  e -->* E[op(v; y.e')]    (op in h)
  ------------------------------------------------------------------- (H-Op-Deep)
  handle e with h -->* [v/x, (lam w. handle [w/y]e' with h)/k] e_op
```

The key: in the op case, the continuation $k$ re-wraps the remainder of the computation in the *same handler* $h$. This means subsequent operations performed in the continuation will also be handled by $h$. Deep handlers naturally implement *stateful* and *recursive* protocols.

=== Shallow Handler Semantics

```text
  e -->* E[op(v; y.e')]    (op in h)
  ------------------------------------------------- (H-Op-Shallow)
  handle e with h -->* [v/x, (lam w. [w/y]e')/k] e_op
```

The continuation $k$ does *not* re-wrap in $h$: subsequent operations in the continuation run *without* $h$ in scope. Shallow handlers implement *single-pass* protocols — the handler sees each operation at most once.

=== Concrete Example: Stateful Counter

```text
-- Deep handler: count total operations
counter_deep = handler
  | return x   -> (x, 0)
  | tick  (); k -> let (r, n) = k () in (r, n + 1)

-- Shallow handler: reset count per recursive call
counter_shallow = handler
  | return x   -> (x, 0)
  | tick  (); k -> let (r, n) = k () in (r, n + 1)
```

For the computation `tick(); tick(); return 42`:

- Under the *deep* handler: each `tick` re-handles subsequent ticks with the same counter. Result: `(42, 2)`.
- Under the *shallow* handler: the first `tick` handles the remaining computation, which has a fresh (inner) handler. Each `tick` sees only the ticks that come *after* it in the continuation. Result is the same `(42, 2)` for this example, but diverges for recursive computations.

The difference becomes clear with a recursive computation:

```text
loop n = if n = 0 then return 0
         else tick(); loop (n-1)
```

Under the *deep* handler `counter_deep`, `loop 5` yields `(0, 5)`. Under a *shallow* handler, the continuation of the first `tick` is `loop 4` without a handler, so subsequent ticks escape — the shallow handler would need to be re-applied manually to get the same effect.

*Theorem (Hillerström–Lindley 2016).* Deep and shallow handlers are mutually encodable. The encoding of deep in terms of shallow uses a fixed-point combinator to re-apply the handler; the encoding of shallow in terms of deep extracts only the "first layer" of the deep handler via a one-shot continuation.

== Row Polymorphism and Effect Inference

In Koka's row-based effect system, types and effects are:

$ tau ::= "Int" | "Bool" | tau_1 ->^epsilon tau_2 | "list"(tau) | dots \
epsilon ::= angle.l angle.r | angle.l op : sigma | epsilon angle.r | rho $

where $rho$ is a *row variable* and $sigma$ is the *type scheme* of operation $op$ (parameter and return type).

Typing rules:

```text
  Gamma |- f : tau1 ->^eps tau2     Gamma |- e : tau1
  ------------------------------------------------------ (T-App)
  Gamma |- f e : tau2 ! eps

  Gamma, x : tau1 |- e : tau2 ! eps
  ------------------------------------------- (T-Abs)
  Gamma |- (lam x. e) : tau1 ->^eps tau2

  Gamma |- e : tau ! <op : P -> A | eps>
  op(v; y.e') in h     Gamma, x : P, k : A ->^eps' tau |- e_op : tau' ! eps'
  --------------------------------------------------------------------------- (T-Handle)
  Gamma |- handle e with h : tau' ! eps
```

The handle rule *removes* operation $op$ from the effect row: if $e$ can perform $op$ and the handler handles $op$, the result has a row without $op$. Unhandled effects *flow through* as the row variable $"eps"$.

*Effect inference algorithm.* Extend Hindley–Milner with row unification:

1. Generate constraints of the form $epsilon_1 = epsilon_2$ from typing rules.
2. Solve by *row unification* (Rémy 1989): rows are equated by *label matching*, with the row variable absorbing remaining labels.
3. Generalize at `let`-bindings over both type and row variables.

Row unification uses the *scoped labels* representation:

$ angle.l l_1 : sigma_1 | angle.l l_2 : sigma_2 | rho angle.r angle.r $

Two rows are unified by finding corresponding labels and unifying their schemes, with the remaining row variable absorbing unmatched labels. The key invariant: labels appear in alphabetical order, so unification terminates.

*Effect polymorphism example.* Inferred type for `map`:

$ "map" : forall alpha beta . forall rho . (alpha ->^{rho} beta) -> "list"(alpha) ->^{rho} "list"(beta) $

The row variable $rho$ is the *effect parameter*: `map` introduces exactly the same effects as the function it applies. At each call site, $rho$ is instantiated to the particular row of the function argument.

== Koka's Evidence Translation

Koka compiles row-polymorphic effect handlers to efficient code by an *evidence translation* (Xie–Brachthäuser–Hillerström–Schuster–Leijen 2020). The key insight: at each `perform` call site, the *handler frame* can be identified *statically* from the effect row, rather than discovered by a runtime stack walk.

The translation works in two phases:

1. *Handler allocation.* When a handler $h$ is installed, an *evidence vector* $"ev" : "Eff"$ is allocated. The vector maps each operation to a function pointer (the handler clause) and a *stack marker* (the address of the handler frame).

2. *Evidence threading.* Every function with effect row $epsilon$ receives an implicit evidence argument $"ev" : "Evidence"(epsilon)$:

```text
-- Source (Koka surface)
fun map(xs : list<a>, f : a -> <exn|e> b) : <exn|e> list<b>
  ...f(x)...

-- After evidence translation (pseudo-code)
fun map(xs : list<a>, f : a -> b, ev_exn : Ev(exn), ev_e : Ev(e))
  ...f(x, ev_exn, ev_e)...
```

3. *Perform compilation.* A call `perform op(v)` becomes a direct call to the handler through the evidence pointer — no stack walking, no dynamic dispatch on an operation tag:

```text
-- Source
perform Raise(msg)

-- After translation
ev_exn.raise(msg, current_stack_pointer)
```

The evidence pointer contains the stack pointer of the handler frame. Resuming the continuation is a *stack switch* to that frame.

*Theorem (Xie et al. 2020, Efficiency).* The evidence translation is $O(1)$ per operation perform: the evidence pointer is known at compile time, and the call is a single indirect function call through the evidence vector.

Compare with the alternative approaches:
- *Free monad*: each `Bind` allocates a heap node; $O(n)$ allocations for $n$ operations.
- *CPS translation*: each handler boundary converts the continuation arrow.r CPS; the continuation is a closure chain of depth $O(n)$.
- *Stack capture*: capturing a continuation copies a stack segment; cost proportional to the segment depth.

The evidence translation has none of these overheads for the common case (single-shot continuations that do not escape). Only *multi-shot* continuations (for nondeterminism) require stack capture.

== OCaml 5 Effects: Stack Segment Implementation

OCaml 5 uses *fiber*-based implementation of effects. Each computation runs in a *fiber* — a stack segment distinct from the main OS stack. When `perform` is called:

1. The current fiber is *suspended*: the machine registers and stack pointer are saved into the fiber's continuation record.
2. The *parent fiber* (the handler) is resumed.
3. The handler's `effc` function is invoked with the effect tag and the continuation handle.

Resuming a continuation with `continue k v`:

1. The parent fiber saves its state.
2. The child fiber is resumed from its saved state with `v` as the result of `perform`.

The fiber stack is a *linked list of segments*. Each segment is $8"KB"$ by default. When a segment is exhausted, a new one is allocated and linked. Capturing a continuation copies all segments from the `perform` point to the handler — this is the cost for multi-shot continuations.

*One-shot continuations* are the common case and are free to resume: the segment is simply resumed in place without copying.

*Example: cooperative threading.* The `eio` library implements async I/O on top of OCaml effects:

```ocaml
effect Suspend : (unit -> unit) -> unit
(* Suspend the current fiber, passing a callback that resumes it. *)

let yield () = perform (Suspend (fun k -> enqueue_continuation k))

let scheduler fibers =
  (* Main loop: dequeue a continuation and resume it *)
  while not (Queue.is_empty fibers) do
    let k = Queue.pop fibers in
    continue k ()
  done
```

Each `yield` suspends the current computation and enqueues the continuation for later. The scheduler is itself written in ordinary OCaml with no explicit threading primitives.

== Forster–Kammar–Lindley–Pretnar 2017: Mutual Encoding

The mutual expressibility between *algebraic effects with handlers* and *multi-prompt delimited control* is formalized by Forster–Kammar–Lindley–Pretnar (2017). The encodings are *modular* (local, not whole-program transforms).

=== Effects arrow.r Delimited Control

Each effect $"op" : P arrow.r A$ is encoded using a *prompt* $p_"op"$:

```text
-- perform op(v) translates to:
shift p_op (fun k -> k_op_handler v (fun a -> push_prompt p_op (k a)))

-- handle e with {op(x; k) -> e_op} translates to:
push_prompt p_op e
```

The `push_prompt` installs the handler (analogous to `handle`); `shift` captures the continuation up to the nearest matching prompt (analogous to `perform`).

=== Delimited Control to Effects

Conversely, `reset { e }` and `shift { k -> e }` are encoded using two effects:

```text
effect DelimReset : unit -> answer
effect DelimShift : (unit -> answer) -> answer

reset { e } = handle e with
  | return x -> x
  | DelimShift f; _ -> f()   (* continuation discarded — "abort" *)

shift { k -> e } = perform DelimShift (fun () ->
  let resume v = perform DelimReset (fun () -> ... v ...)
  in e[k |-> resume])
```

The encoding is *macro-expressible*: each source construct translates to a fixed-size target term (no whole-program transformation). Both directions of the encoding preserve typing.

*Corollary.* Any language with multi-prompt delimited control can simulate algebraic effects, and vice versa. The two mechanisms are *equipotent*.

== Concrete Effect Examples

=== Generators and Iteration

```text
effect Yield : 'a -> unit

let gen_range lo hi =
  let rec loop i =
    if i >= hi then ()
    else (perform (Yield i); loop (i+1))
  in loop lo

let collect gen =
  try_with gen ()
    { effc = fun (type c) (eff : c Effect.t) ->
        match eff with
        | Yield v -> Some (fun k ->
            v :: continue k ())    (* resume after yielding v *)
        | _ -> None }

(* collect (gen_range 0 5) = [0; 1; 2; 3; 4] *)
```

The `Yield` effect is used once per element; the handler resumes with unit, allowing the generator to proceed to the next element.

=== Async/Await

```text
effect Await : 'a promise -> 'a

let await p = perform (Await p)

let async_handler scheduler =
  { effc = fun (type c) eff ->
      match eff with
      | Await promise ->
          Some (fun k ->
            (* Suspend: register k to resume when promise resolves *)
            promise_on_resolve promise (fun v -> scheduler.enqueue (fun () -> continue k v));
            scheduler.run_next ())
      | _ -> None }
```

The `Await` effect captures the rest of the computation as a continuation. When the promise resolves, the continuation is enqueued and the scheduler runs it. This implements cooperative async without dedicated runtime support — only algebraic effects.

=== Probabilistic Programming

```text
effect Sample : distribution -> float

let bernoulli p = perform (Sample (Bernoulli p))
let normal mu sigma = perform (Sample (Normal (mu, sigma)))

(* Exact inference via enumeration (for discrete distributions) *)
let exact_handler =
  { effc = fun (type c) eff ->
      match eff with
      | Sample (Bernoulli p) ->
          Some (fun k ->
            let outcomes = [(true, p); (false, 1. -. p)] in
            List.concat_map (fun (v, prob) ->
              List.map (fun (r, q) -> (r, prob *. q)) (continue k v)
            ) outcomes)
      | _ -> None }
(* Returns a list of (value, probability) pairs *)
```

The same probabilistic program can be run under different handlers: exact enumeration (exponential time), Monte Carlo sampling (linear time with statistical error), or symbolic differentiation (for gradient-based inference). The choice of handler is orthogonal to the model code.

== Linear Handlers: One-Shot Continuations

Hillerström–Lindley (2016, 2020) studied *linear handlers* — handlers in which the continuation is used *at most once* (affine) or *exactly once* (linear).

*Motivation.* Multi-shot continuations (calling $k$ multiple times) require *heap-allocated* continuation closures and potentially *copying* the stack. One-shot continuations can be implemented as *unboxed* stack segments that are moved rather than copied.

*Typing rule (one-shot handler):*

```text
  Gamma, x : tau_op, k : tau_ret ->_1 tau |- e_op : tau   (k used linearly)
  -------------------------------------------------------------------------- (T-Handle-Linear)
  handle_linear e with {op(x; k) -> e_op}
```

The continuation $k$ has a *linear* arrow $->_1$: it must be called exactly once in the handler clause. This enables:

- *Stack-allocated* continuations (no heap allocation for the captured segment).
- *In-place* resumption (no copying of the stack segment).

*Scoped effects* (Piróg–Polesiuk–Sieczkowski 2019; Yang–Paviotti–Bauer–Pretnar–Birkedal 2022) further restrict handlers to *structured* scope: effects are handled in a LIFO order, matching the structure of the program's call stack. This enables efficient implementation using ordinary function calls (no stack capture at all) while still supporting the expressiveness of handlers for state, exceptions, and generators.

== Eff, Frank, Helium: Language Comparison

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Language*], [*Effect syntax*], [*Handler style*], [*Typing*], [*Key features*],
  [Eff (Bauer–Pretnar 2012)], [`effect E : P -> A`], [deep by default, shallow via manual re-wrap], [HM + effects], [first implementation; algebraic semantics faithful to theory],
  [Frank (Levy–Lindley–McBride 2017)], [`<Op : P -> A>`], [shallow; pattern-matching on effects], [bidirectional + effect rows], [handlers are functions; uniform call-by-push-value],
  [Helium (Biernacki–Piróg–Polesiuk–Sieczkowski 2019)], [`let effect E = ...`], [deep, with scoping restrictions], [effect rows + row polymorphism], [scoped effects with algebraic semantics],
  [Koka (Leijen 2014+)], [`effect E { fun op(p : P) : A }`], [multi-clause; deep or shallow], [full row poly + evidence], [production-quality; evidence translation for performance],
  [OCaml 5 (2022)], [`effect E : P -> A` (extensible GADT)], [deep via `try_with`; shallow via `match_with`], [runtime GADT check], [native continuations; segmented stacks; production use],
)

*Frank's distinctive design.* In Frank, a function that handles an effect is written as a *pattern-matching function on its effectful argument*:

```text
-- A function handling the exception effect:
catch : {<Raise E|P> A} -> {E -> A} -> A
catch prog handler =
  prog! -- call prog; if Raise fires, match the argument
  | <Raise e -> _>  -> handler e
  | return x        -> x
```

The `!` operator "calls" an effectful computation; pattern matching on the result either extracts the pure value or catches the operation. This unifies function definition and handler definition into a single syntactic form.

*Helium's scoped effects.* Helium restricts effects to *scoped* protocols: every `perform` must be handled by the *immediately enclosing* handler in the dynamic scope. This makes the implementation straightforward (no continuation capture; the handler stack is the call stack), but forbids multi-shot continuations and some forms of nondeterminism.

The practical upshot: Koka and OCaml 5 are the production systems; Eff and Frank are research vehicles exploring the design space; Helium explores the efficient restricted fragment. All share the same algebraic foundation.

_See also: _Type Systems_ for the substrate of judgements $Gamma tack.r e : tau ! epsilon$, _Linear and Substructural Type Systems_ for the dual axis of resource accounting, _Subtyping and Polymorphism_ for how effect rows compose subtyping-wise._

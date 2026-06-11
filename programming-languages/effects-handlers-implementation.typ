= Effects and Handlers: Implementation and Theory

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

$ T_Sigma(A) = mu X . A + sum_(op in Sigma) (P_op times (A_op arrow.r X)) $

— either a *return* of a value, or an *operation application* with a continuation. This is the free $Sigma$-algebra on $A$.

*Example — state.* With $"Get" : 1 arrow.r S$ and $"Put" : S arrow.r 1$:

$ T_({"Get", "Put"})(A) = mu X . A + (S arrow.r X) + (S times X) $

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

*Theorem (Plotkin–Power 2002).* Every finitary monad on $"Set"$ is the free monad of some algebraic theory. The theory is given by: operations = $Sigma$-algebra generators, equations = the monad's equations.

*Corollary.* Exception, state, reader, writer, nondeterminism, and finite distribution monads are all algebraic. The continuation monad $"Cont" r$ is *not* algebraic for non-trivial $r$ (it cannot be presented by a finitary theory).

== Plotkin–Pretnar Handler Semantics: Deep vs Shallow

We work in a core language $lambda_"eff"$ with the following syntax:

$ v ::= x | lambda x . e | "Inl" v | "Inr" v | () \
e ::= v | e_1 e_2 | "op"(v; y . e) | "return" v | "handle" e "with" h \
h ::= {"return" x |-> e_r; "op"_i(x; k) |-> e_i}_(i in I) $

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

$ "map" : forall alpha beta . forall rho . (alpha ->^(rho) beta) -> "list"(alpha) ->^(rho) "list"(beta) $

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
  [Eff (Bauer–Pretnar 2015)], [`effect E : P -> A`], [deep by default, shallow via manual re-wrap], [HM + effects], [first implementation; algebraic semantics faithful to theory],
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

_See also: _Effects and Handlers_ for the algebraic theory and equational semantics this chapter implements, _Operational Semantics_ for the small-step machinery underlying continuation capture, _Concurrency Models_ for handlers as schedulers._

== Further Reading

Plotkin, G., Pretnar, M. (2009). "Handlers of Algebraic Effects." ESOP. Introduces effect handlers as the eliminators for algebraic effects, showing they subsume exceptions, state, nondeterminism, and coroutines.

Leijen, D. (2017). "Type Directed Compilation of Row-Polymorphic Effects for Practical Generic Programming." POPL. Describes the evidence-passing translation in Koka that compiles row-polymorphic effects to efficient native code without continuation capture.

Sivaramakrishnan, K. C. et al. (2021). "Retrofitting Effect Handlers onto OCaml." PLDI. Details the segmented-stack implementation of algebraic effects in OCaml 5, including the interaction with the multicore memory model.

Bauer, A., Pretnar, M. (2015). "Programming with Algebraic Effects and Handlers." JLAMP 80(7). Presents the Eff language as a practical realisation of the Plotkin–Pretnar theory, with concrete examples of handlers as a programming construct.

Hillerstrom, D., Lindley, S. (2016). "Liberating Effects with Rows and Handlers." TyDe. Introduces the Links effect system with row types, showing how effect rows compose and how handlers interact with polymorphism.

Dolan, S. et al. (2017). "Concurrent System Programming with Effect Handlers." WS-FM. Applies effect handlers to systems programming, demonstrating asynchronous I/O, cooperative threading, and backtracking as user-defined effects in a single language.
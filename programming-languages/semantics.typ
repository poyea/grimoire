= Program Semantics

A grammar tells you which strings are syntactically valid programs. It says nothing about
what those programs *mean*. Two programs with identical syntax trees may have entirely
different behaviors depending on the values of variables, the order of effects, and the
scoping rules in force. Semantics formalizes meaning; without it, a compiler writer has
only intuition, and a type-safety proof has no theorem to prove.

_See also: programming-languages/type-systems.typ for how operational semantics provides
the substrate for the type-soundness theorem (progress + preservation)._

== Why Syntax Is Not Enough

Context-free grammars enforce *syntactic* well-formedness. The following program is
syntactically valid in most CF grammars for C-like languages but semantically ill-formed:

```cpp
int main() {
    int x = y + 1;   // y not declared
    return x;
}
```

Similarly, a type mismatch, use of a variable before initialization, or calling a function
with the wrong number of arguments may be perfectly grammatical yet wrong. These
*context-sensitive* constraints require information that flows across the parse tree --
they are beyond context-free.

*Why not use context-sensitive grammars?* Type 1 (context-sensitive) grammars describe
exactly the languages recognized by *linear-bounded automata*. Membership for a CSG is
PSPACE-complete -- exponential in practice. No language implementation uses a CSG parser.
Instead, compilers apply a *two-phase* approach: a CF parser builds a tree, then semantic
analysis passes enforce the context-sensitive constraints. The semantic analysis phase is
the subject of this chapter.

== Attribute Grammars

*Attribute grammars* (Knuth 1968) annotate each grammar symbol $X$ with a set of
*attributes* $"Attrs"(X)$, partitioned into:

- *Synthesized attributes:* computed from the attributes of children. Information flows
  *up* the tree (bottom-up), as in the type of an expression.
- *Inherited attributes:* computed from the attributes of parent and siblings. Information
  flows *down* the tree (top-down), as in the scope environment.

Each production $A -> X_1 dots X_k$ carries *semantic rules* that define attribute values.
An attribute grammar is *well-defined* if, for every parse tree, there is a unique
assignment of attribute values consistent with all semantic rules. The *dependency graph*
of a parse tree is acyclic for well-defined grammars; *strongly non-circular* (SNC)
grammars guarantee this without examining each parse tree.

*Example -- type inference for arithmetic:*

```
Expr -> Expr1 '+' Expr2
    Expr.type = lub(Expr1.type, Expr2.type)   // synthesized
    Expr1.env = Expr.env                      // inherited (propagated down)
    Expr2.env = Expr.env
```

Here `lub` is the least upper bound in the type lattice (e.g., `int` $subset$ `float`).

== Symbol Tables, Lexical and Dynamic Scope

A *symbol table* maps names to their declarations. The relevant questions are: which
declaration does a use of name $x$ refer to, and at what point in execution does the
binding hold?

- *Lexical (static) scope:* a name refers to the innermost enclosing declaration at the
  point of *definition* in the source text. Most modern languages (C++, Rust, Haskell,
  Python) use lexical scope. Closures capture their lexical environment.
- *Dynamic scope:* a name refers to the most recent binding in the *call stack* at the
  point of *use* during execution. Early Lisp dialects used dynamic scope; Emacs Lisp
  retains it. Dynamic scope makes programs harder to reason about because a function's
  behavior depends on its callers.

Symbol tables are typically implemented as a stack of hash maps: entering a new scope
pushes a fresh map; exiting pops it. For type checking and IR generation, the symbol table
stores the type, storage class, and IR value (register or memory address) for each name.

== Operational Semantics

*Operational semantics* defines meaning by describing how computations *step*. There are
two flavors:

=== Small-Step (Structural) Semantics

The *small-step* relation $e -> e'$ (also written $e |-> e'$) defines a single reduction
step. The meaning of $e$ is the value $v$ to which it reduces after zero or more steps:
$e ->^* v$.

*Rules for arithmetic expressions* (integers, variables, addition):

```
[E-Int]       n -> n         (already a value, no step)

[E-Var]       x -> sigma(x)  (look up x in store sigma)

[E-Add-L]     e1 -> e1'
              ─────────────────
              e1 + e2 -> e1' + e2

[E-Add-R]     e2 -> e2'
              ─────────────────
              v1 + e2 -> v1 + e2'

[E-Add]       n1 + n2 -> n             (where n is the numeral for n1 plus n2)
```

Rules are applied by pattern-matching on the outermost form of the expression. The
*determinism* of small-step semantics (at most one applicable rule for each configuration)
corresponds to deterministic evaluation order. Non-determinism would correspond to
unspecified evaluation order.

*Mutable state.* Extend the configuration to a pair $angle.l e, sigma angle.r$ where
$sigma : "Var" -> "Val"$ is the store:

```
[E-Assign]    x := v , sigma  ->  skip , sigma[x |-> v]

[E-Seq-1]     <skip ; c , sigma>  ->  <c , sigma>

[E-Seq-2]     <c1 , sigma>  ->  <c1' , sigma'>
              ──────────────────────────────────
              <c1 ; c2 , sigma>  ->  <c1' ; c2 , sigma'>
```

Small-step semantics is well-suited to concurrent languages (interleaving), lazy evaluation
(reduction strategies), and compiler intermediate representations (rewriting passes).

=== Big-Step (Natural) Semantics

The *big-step* relation $e arrow.b.double v$ (also written $angle.l e, sigma angle.r arrow.b.double v$ when threading a store) relates an expression directly to its final value, skipping intermediate steps:

```
[B-Int]       n => n

[B-Add]       e1 => n1    e2 => n2
              ─────────────────────
              e1 + e2 => n1 + n2

[B-Var]       ─────────────
              x => sigma(x)

[B-If-T]      e => true    c1 => v
              ──────────────────────
              if e then c1 else c2 => v

[B-If-F]      e => false   c2 => v
              ──────────────────────
              if e then c1 else c2 => v

[B-While]     e => true    c => sigma'    (while e do c, sigma') => sigma''
              ──────────────────────────────────────────────────────────────
              (while e do c, sigma) => sigma''

[B-While-F]   e => false
              ─────────────────────────
              (while e do c, sigma) => sigma
```

Big-step semantics is natural for interpreters (each rule is one case in a recursive
`eval` function) and for reasoning about terminating computations. It does not model
non-termination gracefully -- a diverging program simply has no derivation -- whereas
small-step makes divergence explicit ($e$ reduces forever).

== Denotational Semantics

*Denotational semantics* assigns to each program a *mathematical object* -- its denotation
-- in a compositional way: the meaning of a compound expression is a function of the
meanings of its parts.

For a simple imperative language, the denotation of a command is a *state transformer*:
$[[ c ]] : "State" -> "State"$ (or to a lifted domain including divergence).

*Loops via fixpoints.* The `while` loop $W = "while" e "do" c$ satisfies the unfolding
equation $[[ W ]] = [[ "if" e "then" (c ; W) "else" "skip" ]]$. This is a recursive
equation in the domain of state transformers. To solve it, note that:

- State transformers form an *omega-complete partial order* ($omega$-CPO): the "less defined"
  order is pointwise ($f <= g$ iff for all states $s$, $f(s) = g(s)$ or $f(s) = bot$).
- The functional $Phi(f) = [[ "if" e "then" (c ; f) "else" "skip" ]]$ is *continuous*
  (monotone and preserving of $omega$-chains).
- By the *Kleene fixpoint theorem*, the least fixpoint $mu Phi = sup_n Phi^n(bot)$ exists.
  $Phi^0(bot)$ is the function that diverges on everything; $Phi^n(bot)$ handles loops
  that terminate in at most $n$ iterations.

The denotational meaning of $W$ is this least fixpoint. Denotational semantics is the
right tool for *compositional reasoning*: a library function can be replaced by anything
with the same denotation without affecting the meaning of any calling program. This is
the semantic foundation of the *Liskov substitution principle*.

== Axiomatic Semantics (Hoare Logic)

*Axiomatic semantics* (Hoare 1969) characterizes program behavior via *Hoare triples*
${ P } , c , { Q }$: if $P$ holds in the initial state and $c$ terminates, then $Q$ holds
in the final state. $P$ is the *precondition*, $Q$ the *postcondition*.

=== Inference Rules

```
[H-Skip]
    ─────────────────────
    {P}  skip  {P}

[H-Assign]
    ─────────────────────────────────
    {P[e/x]}  x := e  {P}

[H-Seq]
    {P} c1 {R}     {R} c2 {Q}
    ──────────────────────────
    {P}  c1 ; c2  {Q}

[H-If]
    {P /\ b} c1 {Q}     {P /\ ~b} c2 {Q}
    ──────────────────────────────────────
    {P}  if b then c1 else c2  {Q}

[H-While]
    {I /\ b} c {I}
    ──────────────────────────────────────
    {I}  while b do c  {I /\ ~b}

[H-Conseq]
    P' => P     {P} c {Q}     Q => Q'
    ──────────────────────────────────
    {P'}  c  {Q'}
```

The assignment rule runs *backwards*: to establish $P$ after `x := e`, the precondition
is $P$ with every free occurrence of $x$ replaced by $e$. The while rule uses an
*invariant* $I$: a predicate maintained by each iteration. Finding $I$ is the creative
step; verifying the premises is mechanical.

*Partial vs total correctness.* The rules above give *partial correctness*: the postcondition
holds *if* the program terminates. *Total correctness* additionally requires a proof of
termination (usually via a *variant* -- a natural-number expression that decreases each
iteration and is bounded below by zero).

=== Soundness with Respect to Operational Semantics

Hoare logic is *sound* if every provable triple ${ P } c { Q }$ is true under the
operational semantics:

$forall sigma. P(sigma) -> angle.l c, sigma angle.r ->^* angle.l "skip", sigma' angle.r => Q(sigma')$

The proof proceeds by induction on the derivation of the Hoare triple. Soundness connects
the proof system to the machine model; without it, a provable triple might be false.

*Completeness* (relative completeness, Cook 1978) holds as well: any true partial-correctness
triple is provable in Hoare logic, provided we have access to an oracle for the assertion
language. The incompleteness of arithmetic means we cannot always verify the oracle
predicates, but the logic itself misses nothing.

== Choosing a Semantics

#table(
  columns: (auto, auto),
  [*Semantics*], [*Best fit*],
  [Operational (small-step)],  [Compiler IR passes, concurrent languages, reduction-based PLs],
  [Operational (big-step)],    [Certified interpreters, language standard prose translation],
  [Denotational],              [Compositional library reasoning, abstract interpretation domains],
  [Axiomatic],                 [Verification tools: Dafny, Frama-C/WP, VeriFast, Iris],
)

In practice the choice is not exclusive. CompCert (the verified C compiler) uses
operational semantics for its language specifications and correctness proofs. Frama-C
uses axiomatic (Hoare) verification for user-facing annotations but builds on an
operational model internally. Denotational models underlie abstract interpretation
frameworks (Cousot and Cousot 1977) that power industrial static analyzers.

*Type soundness as a semantic theorem.* A type system is not meaningful without a
semantics. The standard formulation (Wright and Felleisen 1994) proves two properties
over the small-step relation:

- *Progress:* a well-typed, non-value expression can always take a step.
- *Preservation:* if $e$ has type $tau$ and $e -> e'$, then $e'$ also has type $tau$.

Together these guarantee that a well-typed program never reaches a *stuck* configuration
(no applicable rule, not a value) -- which is the formal definition of "no runtime type
errors".

_See also: programming-languages/type-systems.typ for the full Progress + Preservation
proof for the simply-typed lambda calculus and the statement of type-soundness for
Hindley-Milner._

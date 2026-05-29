= Linear and Substructural Type Systems

Substructural type systems control *how often* a variable may be used. Classical logic and the simply-typed lambda calculus treat the typing context $Gamma$ as a "set": any hypothesis may be duplicated, discarded, or reordered at will. Substructural systems drop one or more of those liberties — and in doing so gain the ability to talk about resources, capabilities, file handles, channel endpoints, and memory ownership inside the type system itself.

_See also: Type Systems, Effects and Handlers, Subtyping and Polymorphism._

== The Structural Rules

In a Gentzen-style sequent calculus, three rules govern the bookkeeping of the context $Gamma$ rather than any logical connective. Call them the *structural rules*:

```text
  Gamma, A, B, Delta |- C                 Gamma |- C
  ----------------------- (Exchange)      ---------------- (Weakening)
  Gamma, B, A, Delta |- C                 Gamma, A |- C

  Gamma, A, A |- C
  -------------------- (Contraction)
  Gamma, A |- C
```

- *Exchange* lets you reorder hypotheses.
- *Weakening* lets you ignore a hypothesis.
- *Contraction* lets you use a hypothesis twice (under two names).

Dropping subsets gives a lattice of substructural logics:

#table(
  columns: (auto, auto, auto, auto, auto),
  [*Logic*], [*Exchange*], [*Weakening*], [*Contraction*], [*Reading of $A tack.r B$*],
  [Classical / Intuitionistic], [yes], [yes], [yes], [from $A$ you may derive $B$],
  [Affine], [yes], [yes], [no], [from $A$ you may derive $B$, using $A$ at most once],
  [Relevant], [yes], [no], [yes], [from $A$ you derive $B$, using $A$ at least once],
  [Linear], [yes], [no], [no], [from $A$ you derive $B$, using $A$ exactly once],
  [Ordered (non-commutative)], [no], [no], [no], [order of hypotheses matters],
)

This taxonomy is not merely syntactic curiosity. Each row corresponds to a programming discipline that has shipped in production: affine $tilde$ Rust's ownership; linear $tilde$ Linear Haskell and uniqueness types in Clean; relevant $tilde$ certain capability calculi; ordered $tilde$ stack discipline and concatenative languages.

== Linear Logic (Girard 1987)

Girard's *linear logic* (LL) is the foundational system. It refines intuitionistic logic by splitting conjunction into two and disjunction into two, exposing the "resource" content of proofs.

*Formulae* of intuitionistic linear logic (ILL):

$ tau ::= A | tau_1 multimap tau_2 | tau_1 times.circle tau_2 | tau_1 amp tau_2 | tau_1 plus.circle tau_2 | !tau | 1 | 0 | top $

(The classical version adds par $tau_1 #h(0.3em) "par" #h(0.3em) tau_2$, the dual $bot$, and the why-not modality $?tau$. ILL omits these; intuitionistic linear logic is the one most directly relevant to type systems.) The connectives split into two flavours:

- *Multiplicatives* share their context. To prove $A times.circle B$ the context splits into two disjoint pieces, one proving $A$ and one proving $B$.
- *Additives* share their context. To prove $A amp B$ a single context proves both $A$ and $B$ — but only one can be chosen at elimination time.

#table(
  columns: (auto, auto, auto, auto),
  [*Connective*], [*Symbol*], [*Flavour*], [*Reading*],
  [Linear implication], [$multimap$], [(multiplicative)], [consume an $A$, produce a $B$],
  [Tensor], [$times.circle$], [multiplicative conjunction], [both $A$ and $B$, simultaneously],
  [With], [$amp$], [additive conjunction], [your choice of $A$ or $B$],
  [Plus], [$plus.circle$], [additive disjunction], [either $A$ or $B$, decided by the prover],
  [Of-course], [$!tau$], [exponential], [unlimited copies of $tau$ available],
  [One], [$1$], [tensor unit], [empty multiplicative resource],
  [Top], [$top$], [with unit], [trivially satisfiable in any context],
)

The slogan: *multiplicative connectives split the context, additive connectives share it*.

=== Sequent Calculus for ILL

Sequents are $Gamma tack.r tau$ where $Gamma$ is a *multiset* of formulae (exchange is retained; contraction and weakening are not). A few key rules:

```text
  ---------------- (Id)
  A |- A

  Gamma |- A      Delta, A |- B
  ----------------------------------- (Cut)
  Gamma, Delta |- B

  Gamma, A |- B
  ------------------------ (-o R)
  Gamma |- A -o B

  Gamma |- A     Delta, B |- C
  ----------------------------------- (-o L)
  Gamma, Delta, A -o B |- C

  Gamma |- A     Delta |- B
  ----------------------------------- (tensor R)
  Gamma, Delta |- A (x) B

  Gamma, A, B |- C
  --------------------- (tensor L)
  Gamma, A (x) B |- C

  Gamma |- A     Gamma |- B
  ----------------------------- ("with R)
  Gamma |- A & B

  Gamma, A |- C
  --------------------- ("with" L1)
  Gamma, A & B |- C
```

The shape of the rules encodes the slogan. Tensor-right splits the ambient context $Gamma, Delta$ into a piece proving $A$ and a piece proving $B$. With-right requires both sub-proofs to use the *same* $Gamma$, because the eliminator picks only one of the two branches and only its resources will end up consumed.

The exponential $!tau$ marks formulae on which contraction and weakening are permitted:

```text
  !Gamma |- A
  ----------------- (Promotion / !R)
  !Gamma |- !A

  Gamma, A |- B                    Gamma |- B
  --------------------- (!L)       ------------------- (Weakening)
  Gamma, !A |- B                   Gamma, !A |- B

  Gamma, !A, !A |- B
  ----------------------- (Contraction)
  Gamma, !A |- B
```

Promotion has the side condition that *all* hypotheses are exponential — otherwise we could smuggle weakening past a linear assumption. The exponential is a *comonad* on the category of linear maps: there is a counit $!A multimap A$ (derelict the modality and obtain a usable $A$), a comultiplication $!A multimap !!A$ (a permission to use $A$ unboundedly can itself be used unboundedly), and the bang lifts maps $A multimap B$ to $!A multimap !B$.

=== Cut Elimination

*Theorem (Girard 1987, cut elimination).* Every derivation in linear logic can be transformed into a cut-free derivation of the same sequent.

The proof is the usual triple induction (on the cut formula, on the height of the right derivation, on the height of the left derivation), but linear logic gives sharper data: a cut between $A times.circle B$ on one side and its left-rule on the other reduces *without duplication*; the reduction strictly decreases a resource measure rather than merely a proof height. Cut elimination on the exponentials, in contrast, may duplicate sub-derivations precisely when contraction is invoked — locating the irreducible source of complexity in the modality $!$, not the linear core.

*Computational reading.* Cut is the typing rule for substitution; cut elimination is normalisation of programs; the linear core normalises in polynomial time, while the exponentials are where the combinatorial blow-up of $beta$-reduction lives (Girard, Scedrov, Scott; *Bounded linear logic* 1992).

== Proof Nets

Sequent derivations have spurious bureaucracy: the order in which independent rules are applied is arbitrary, generating many derivations of the same theorem. *Proof nets* (Girard 1987) are a geometric, parallel proof syntax that quotients out this bureaucracy.

A proof net for a multiplicative formula is a graph whose nodes are the formula's connectives and whose edges encode the axiom links between dual atoms. Not every such graph corresponds to a sequent derivation; one needs a *correctness criterion* to identify the "proof-like" graphs.

*Theorem (Danos–Regnier 1989, switching criterion).* A multiplicative proof structure is a proof net iff for every choice of one premise at each par node, the resulting graph is acyclic and connected.

Girard's original *long-trip* criterion was equivalent but combinatorially heavier. Danos–Regnier reduced correctness checking to a check over $2^k$ switchings ($k$ par nodes); subsequent work (Murawski–Ong, Guerrini) gave polynomial-time algorithms. Proof nets are the basis for *game semantics* and the *geometry of interaction* (Girard 1989).

== The Linear Lambda Calculus

The Curry–Howard image of ILL is the *linear lambda calculus*. The syntax:

$ e ::= x | lambda x : tau . e | e_1 space e_2 | (e_1, e_2) | "let" (x, y) = e_1 "in" e_2 | "!" e | "let" "!" x = e_1 "in" e_2 $

Typing uses *two* contexts, $Gamma; Delta$: an *unrestricted* context $Gamma$ (variables behaving classically, originating from $!$ types) and a *linear* context $Delta$ (each variable must be used exactly once).

```text
  ------------------------- (lin-var)
  Gamma; x : tau |- x : tau

  ------------------------- (un-var)
  Gamma, x : tau; . |- x : tau

  Gamma; Delta, x : tau_1 |- e : tau_2
  --------------------------------------------- (lin-abs)
  Gamma; Delta |- (lam x : tau_1. e) : tau_1 -o tau_2

  Gamma; Delta_1 |- e_1 : tau_1 -o tau_2     Gamma; Delta_2 |- e_2 : tau_1
  -------------------------------------------------------------------------- (lin-app)
  Gamma; Delta_1, Delta_2 |- e_1 e_2 : tau_2

  Gamma; Delta_1 |- e_1 : tau_1     Gamma; Delta_2 |- e_2 : tau_2
  ----------------------------------------------------------------- (tensor-I)
  Gamma; Delta_1, Delta_2 |- (e_1, e_2) : tau_1 (x) tau_2

  Gamma; Delta_1 |- e_1 : tau_1 (x) tau_2     Gamma; Delta_2, x:tau_1, y:tau_2 |- e_2 : tau
  ------------------------------------------------------------------------------------------- (tensor-E)
  Gamma; Delta_1, Delta_2 |- (let (x, y) = e_1 in e_2) : tau

  Gamma; . |- e : tau
  ------------------------- (!-I)
  Gamma; . |- !e : !tau

  Gamma; Delta_1 |- e_1 : !tau_1     Gamma, x : tau_1; Delta_2 |- e_2 : tau
  --------------------------------------------------------------------------- (!-E)
  Gamma; Delta_1, Delta_2 |- (let !x = e_1 in e_2) : tau
```

The application rule splits the linear context $Delta_1, Delta_2$ between function and argument: no linear resource is used twice. The introduction rule for $!$ requires an empty linear context — you cannot package a one-shot resource into something that can be used many times.

*Theorem (linearity invariant).* If $Gamma; Delta tack.r e : tau$, then each variable bound in $Delta$ appears free in $e$ exactly once; each variable bound in $Gamma$ may appear zero or more times.

The proof is a straightforward induction on derivations. The economic consequence: a function $f : tau_1 multimap tau_2$ cannot duplicate or discard its argument. Equivalently, $f$ must *transform* its argument into the result without losing information about whether the argument has been consumed.

== Affine, Relevant, and Ordered Calculi

*Affine* type systems drop contraction but retain weakening: every variable is used *at most* once. Rust's ownership discipline is essentially affine — values may be dropped silently (running their destructor) but cannot be aliased without an explicit borrow.

*Relevant* type systems drop weakening but retain contraction: every variable is used *at least* once. Useful when one wishes to guarantee that resources are not silently discarded — e.g., that every transaction is either committed or explicitly rolled back, never forgotten.

*Ordered* (or *non-commutative*) systems drop exchange as well. The Lambek calculus (Lambek 1958), originally proposed for natural-language syntax, sits in this corner; concatenative languages like Forth and Joy implement an ordered discipline at the level of the runtime stack.

== Bunched Logic and Separation Logic

*Bunched implications* (BI; O'Hearn–Pym 1999) treat the context as a *tree of bunches* with two kinds of comma:

$ Gamma ::= Delta | Gamma, Gamma | Gamma; Gamma $

— *additive* semicolon $;$ allows weakening and contraction, *multiplicative* comma $,$ does not. BI has two implications, two conjunctions, and one disjunction. The crucial connective is *separating conjunction* $*$, with the introduction rule

```text
  Gamma |- A     Delta |- B
  ---------------------------- (* I)
  Gamma, Delta |- A * B
```

*Separation logic* (Reynolds 2002; O'Hearn 2007) instantiates BI with the heap model: $A * B$ holds in a heap that splits into two disjoint sub-heaps, one satisfying $A$ and the other $B$. Hoare triples become the basis for *Iris*, *VST*, and modern verification of concurrent and unsafe code. Separation logic is the leading explanation of why Rust's ownership discipline is sound: it is a separation-logic invariant compiled into the type system.

== Uniqueness Types vs Linear Types

Two superficially similar disciplines, with different semantic intent.

*Uniqueness types* (Barendsen–Smetsers 1996, in Clean) annotate a *reference* with the promise that no other reference to the same object exists. The type $"*"tau$ is read "the unique pointer to a $tau$". A unique reference can be safely mutated in place.

*Linear types* (Wadler 1990) annotate a *binding* with the promise that the variable is used exactly once in the program text. A linear value need not be the only handle to its referent — but the program promises to forget its handle once it is used.

#table(
  columns: (auto, auto, auto),
  [*Aspect*], [*Uniqueness (Clean)*], [*Linearity (Linear Haskell)*],
  [Annotates], [a value at a point in time], [a binding in the syntax],
  [Allows aliasing?], [no — the type forbids it], [yes — only single use is required],
  [Use case], [in-place update of arrays], [resource discipline, protocols],
  [Substitution interacts with], [the heap], [the typing context],
)

The two disciplines coincide on the *non-aliased* fragment; the divergence appears once one allows sharing under $!$.

== Linear Haskell (Bernardy–Boespflug–Newton–Peyton Jones–Spiwack 2018)

Linear Haskell adds a *multiplicity-annotated arrow*: `a %1 -> b` is a function that uses its argument exactly once; `a %m -> b` is multiplicity-polymorphic. Crucially, the same data type can be constructed and consumed in either a linear or unrestricted style — a single `Maybe a` works both for ordinary `Maybe` and for linear `Maybe a`, parameterised by the multiplicity at which `a` is held.

```haskell
-- Linear function: consumes its argument exactly once
withFile :: FilePath -> (Handle %1 -> Ur a) %1 -> IO a

-- A linear identity function
linId :: a %1 -> a
linId x = x

-- The unrestricted wrapper
data Ur a where
  Ur :: a -> Ur a

-- Multiplicity polymorphism
($) :: forall {m} a b. (a %m -> b) %1 -> a %m -> b
f $ x = f x
```

The design constraints were severe: Haskell already had a vast ecosystem, so retrofitting linear types had to be *non-invasive*. The trick was *multiplicity polymorphism*: a function arrow now carries a multiplicity $m in {1, omega}$ (and is polymorphic over it by default in many positions). Existing code, written with implicit $omega$-arrows, type-checks unchanged.

*Use cases* in Linear Haskell:

- *Mutable arrays without GC chatter:* `Array.alloc :: Int -> (Array a %1 -> Ur b) -> Ur b` — allocate, mutate linearly, return a non-linear result.
- *Safe file handles:* a handle that *must* be closed.
- *Session-typed channels:* a channel endpoint that must be used according to its protocol.

== Rust as an Affine Type System

Rust's ownership and borrowing discipline is best understood as an affine type system with *region inference* and *bidirectional borrowing*. Rough correspondence:

#table(
  columns: (auto, auto),
  [*Rust feature*], [*Type-theoretic counterpart*],
  [`T` (owned value)], [affine assumption $T$],
  [`&'a T` (shared reference)], [borrowed assumption within region $'a$],
  [`&'a mut T` (unique reference)], [exclusive linear borrow within $'a$],
  [`Drop` impl], [explicit weakening rule],
  [`Copy` trait], [unrestricted ($omega$) assumption],
  [`Clone::clone`], [explicit contraction operator],
  [Lifetime subtyping `'a : 'b`], [region inclusion],
  [Non-Lexical Lifetimes (NLL)], [region inference via dataflow],
)

The *borrow checker* enforces two structural invariants at every program point: (1) every value has exactly one owner, (2) references are either one exclusive `&mut` *or* any number of shared `&` — never both. Reading the rules of the borrow checker as a sequent calculus, exclusive borrows are linear, shared borrows are affine within their region, and ownership transfer (move semantics) is precisely the linear-function application rule.

```rust
fn consume(s: String) {
    println!("{}", s);
}                       // s is dropped here

fn main() {
    let s = String::from("hello");
    consume(s);         // ownership of s moves into consume
    // println!("{}", s); // would not compile: s has been moved
}
```

The error message "value used here after move" is, transliterated into proof theory, "linear hypothesis used twice". Rust's affine nature (rather than strict linear) shows in the *implicit drop*: a value that is never consumed is silently destructed at the end of its scope, which is weakening.

=== The Region Calculus (Tofte–Talpin 1994)

Rust's lifetimes did not arise ex nihilo. Tofte–Talpin's region calculus introduced explicit *region* annotations on values:

$ e ::= x | (e_1, e_2)^rho | "letregion" rho "in" e | dots $

Each allocation specifies the region $rho$ where the value lives; `letregion` introduces and deallocates a region in a stack-discipline manner. The region inference algorithm computes regions automatically from a textually unannotated program. ML Kit (Tofte–Birkedal–Elsman–Hallenberg 1997) compiles SML to native code using region inference *instead of* a garbage collector.

Rust's lifetimes are a generalisation: regions are inferred where possible, but the programmer can annotate function signatures to express region polymorphism. The borrow checker is a region-and-affinity inference engine working over the *MIR* (mid-level intermediate representation) of the program.

== Session Types

A *session type* (Honda 1993; Honda–Vasconcelos–Kubo 1998) is a type assigned to a *channel endpoint* that describes the protocol that endpoint must follow over its lifetime. Where ordinary types classify values, session types classify *interactions*.

The basic syntax:

$ S ::= !tau . S | ?tau . S | S_1 plus.circle S_2 | S_1 amp S_2 | "end" $

with the dyadic reading:

- $!tau . S$ — send a value of type $tau$, then continue as $S$.
- $?tau . S$ — receive a value of type $tau$, then continue as $S$.
- $S_1 plus.circle S_2$ — internal choice: *we* choose to continue as $S_1$ or $S_2$.
- $S_1 amp S_2$ — external choice: *the peer* picks the branch.
- $"end"$ — protocol complete.

Duality: $overline(!tau . S) = ?tau . overline(S)$, etc. Two endpoints of a single channel must have dual types — what one sends, the other receives.

A simple arithmetic-server protocol:

$ S_"server" = ?"Int" . ?"Int" . !"Int" . "end" $
$ S_"client" = overline(S_"server") = !"Int" . !"Int" . ?"Int" . "end" $

If session types are *linear*, a channel endpoint cannot be split, duplicated, or forgotten mid-protocol. This rules out the classical errors of protocol implementation: *receiving when the peer expects to receive*, *forgetting to send the reply*, *closing a channel mid-conversation*.

```text
  Gamma; Delta, c : ?tau.S |- e : tau'
  --------------------------------------------- (T-Recv)
  Gamma; Delta, c : ?tau.S |- recv c >>= (\x:tau. e) : tau'

  Gamma; Delta, c : !tau.S |- v : tau
  --------------------------------------------- (T-Send)
  Gamma; Delta, c : S |- send c v ; e : tau'
```

Each rule *advances* the session type of the endpoint: after receiving, the endpoint has type $S$, not $?tau . S$. The type system thereby tracks *protocol state*.

=== Multiparty Session Types (Honda–Yoshida–Carbone 2008)

Two-party sessions generalise to many participants via *global types* describing the entire choreography:

$ G ::= "p" -> "q" : tau . G | "p" -> "q" : { l_i : G_i }_i | mu t. G | t | "end" $

A global type is *projected* onto each participant to obtain a *local* session type. Well-formedness of the global type (no two participants both make an internal choice on the same branch, etc.) implies *deadlock freedom* and *protocol fidelity* of the projected local types running concurrently.

== The Caires–Pfenning Correspondence

*Theorem (Caires–Pfenning 2010, Wadler 2012 "Propositions as Sessions").* Intuitionistic linear logic proofs and $pi$-calculus processes are in bijection, under the dictionary:

#table(
  columns: (auto, auto),
  [*Linear logic*], [*$pi$-calculus*],
  [Proposition $A$], [session type],
  [Sequent $Gamma tack.r x : A$], [process providing channel $x : A$ given $Gamma$],
  [Cut rule], [parallel composition with hidden channel],
  [Identity], [forwarder process $x <-> y$],
  [$A multimap B$], [input a session of type $A$, behave as $B$],
  [$A times.circle B$], [output a fresh session of type $A$, then behave as $B$],
  [$A amp B$], [offer external choice between $A$ and $B$],
  [$A plus.circle B$], [make internal choice],
  [Cut elimination], [reduction (communication) in the $pi$-calculus],
)

Two consequences fall out of the correspondence: every well-typed process is *deadlock-free* (cut elimination terminates), and every well-typed process is *race-free* (linearity of channels prevents two senders racing on the same endpoint).

Wadler's CP language (*Classical Processes*, 2014) gives a syntactic embodiment: process syntax that *is* linear logic proof syntax, with a one-to-one mapping between reduction and cut elimination steps.

== Implementations

- *Linear Haskell* (GHC 9.0+, 2021) — `LinearTypes` extension, `linear-base` library.
- *Idris 2* (Brady 2021) — quantitative type theory with multiplicities $0$, $1$, $omega$.
- *Granule* (Orchard–Liepelt–Eades 2019) — research language with graded modal types, generalising linear/affine/relevant.
- *ATS* and *F\** support linear-like resource tracking.
- *Session types in OCaml* via GADTs (Padovani 2017): the indexed type of channels encodes the remaining protocol; mis-uses become type errors at compile time.
- *Rust* — the only mainstream production language with an affine type system by default.

== A Brief Worked Example: Safe File Handles

A classical resource-leak bug is "file opened but not closed". A linear type prevents it.

```haskell
{-# LANGUAGE LinearTypes #-}

withFile :: FilePath
         -> (Handle %1 -> Ur a)  %1
         -> IO a

readFirstLine :: FilePath -> IO String
readFirstLine fp = withFile fp $ \h ->
  case hGetLine h of
    (line, h') -> case hClose h' of
      () -> Ur line   -- must close h' before returning
```

The handle `h` has linear type `Handle`; the only operations on it return a *new* `Handle` (after `hGetLine`) or consume it entirely (`hClose`). The continuation must *return* its `Handle` to be closed, lest the linear-arrow constraint be violated. The type system rules out: forgetting to close; closing twice; using after close; smuggling the handle outside the continuation.

== Theoretical Coda: What Substructurality Buys

The discipline of dropping structural rules is not free — it makes the typing context heavier and the" rules subtler. What is gained:

1. *Resource tracking* without runtime reference counting or GC pauses.
2. *Protocol enforcement* — the type system understands "must be done in order".
3. *Memory safety* without a managed runtime (Rust).
4. *In-place update* of arrays in a pure language (Linear Haskell, Clean).
5. *Deadlock freedom* for concurrent processes (Caires–Pfenning).
6. *Capability discipline* — possession of a value *is* permission to act on it.

The cost is a steeper learning curve and a more verbose surface syntax. Forty years on from Girard's original paper, the wager that linear logic captures something fundamental about computation has paid off: every modern systems language has, by name or by structure, absorbed a portion of the substructural toolbox.

== Pierce's Critique and the Pragmatic Middle

Pierce remarks in *Types and Programming Languages* (2002) that "linear types are a powerful tool, but in practice their pervasive use can make programs hard to write". The pragmatic resolution adopted by Linear Haskell, Idris 2, and Granule is *graded modal types*: a single language allows variables to carry any multiplicity from a chosen semiring (e.g., $0$, $1$, $omega$ for irrelevance / linear / unrestricted, or the natural numbers for exact usage counts). Programmers pay the syntactic cost only at the boundaries where it matters.

*Theorem (Atkey 2018, quantitative type theory).* Multiplicities form a semiring; the typing rules instantiate uniformly to recover linear, affine, relevant, dependent-irrelevant, and ordinary type systems as special cases.

This is the modern face of substructural type theory: not five competing logics, but one *graded* framework parameterised by a semiring of resource accounting.

_See also: Effects and Handlers for the dual question — not 'how many times is a value used' but 'what operations does a computation perform' — and Type Systems for the substrate on top of which substructurality is layered._

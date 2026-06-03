= Axiomatic Semantics

Axiomatic semantics describes program meaning by what programs *prove*. Where operational semantics says how a program executes and denotational semantics says what mathematical object it is, axiomatic semantics gives *assertions* about its behavior and inference rules for deriving those assertions from program text. The discipline begins with Floyd's flowchart proofs (1967) and Hoare's *An Axiomatic Basis for Computer Programming* (1969); it has matured into program logics powerful enough to verify operating-system kernels, cryptographic libraries, and concurrent data structures.

*See also:* _Operational Semantics_, _Denotational Semantics_, _Type Systems_, _Categorical Semantics_

== Floyd: Assertions on Flowcharts

Floyd (1967) annotated flowchart programs with *assertions*: predicates over program variables attached to edges. A program is correct if, for every edge $e$ with assertion $P_e$, executing the basic block on the incoming edge $e'$ with assertion $P_(e')$ holding establishes $P_e$. The *verification condition* at edge $e$ is a logical implication.

Floyd's framework already contains the central ideas: the loop *invariant* (an assertion attached to the back-edge of a loop), the proof obligation at each transition, and the reduction of *program correctness* to a finite collection of *logical* implications -- formulae in arithmetic with free variables denoting program state.

The shift from flowcharts to structured programs (Dijkstra 1968) made the assertions follow the syntax of the program rather than the topology of a flowchart. This shift, formalized by Hoare, is the basis of every modern program logic.

== Hoare Triples for IMP

A *Hoare triple* ${P} c {Q}$ asserts: if the precondition $P$ holds before executing command $c$ and $c$ *terminates*, then the postcondition $Q$ holds afterwards. This is *partial correctness*; we discuss total correctness below.

Assertions $P$, $Q$ range over a first-order language with arithmetic that mentions program variables and (typically) logical variables. We assume an assertion language rich enough to express all relevant invariants -- specifically, *expressive* in the sense of Cook (1978).

*Hoare's rules for IMP.*

```text
[H-Skip]
    ----------------
    {P}  skip  {P}

[H-Assign]
    ----------------------------    (substitution P[e/x]: replace free x by e in P)
    {P[e/x]}  x := e  {P}

[H-Seq]
    {P} c1 {R}    {R} c2 {Q}
    -------------------------
    {P}  c1 ; c2  {Q}

[H-If]
    {P /\ b} c1 {Q}    {P /\ ~b} c2 {Q}
    ----------------------------------
    {P}  if b then c1 else c2  {Q}

[H-While]
    {I /\ b} c {I}
    -----------------------------------    (I is the loop invariant)
    {I}  while b do c  {I /\ ~b}

[H-Conseq]
    P' => P    {P} c {Q}    Q => Q'
    --------------------------------
    {P'}  c  {Q'}
```

The assignment rule looks *backwards* -- the precondition is computed from the postcondition by substitution. This is the right direction logically: to ensure $P$ holds after `x := e`, demand that $P[e/x]$ holds before. Forward-style assignment ("$x := e$ from precondition $P$ yields postcondition $exists x_0. x = e[x_0 /x] and P[x_0 / x]$") is logically equivalent but harder to use mechanically.

*Loop invariants* are the creative core. Finding $I$ is undecidable in general; verification *given* $I$ is mechanical (modulo the assertion-language oracle).

=== Example: Sum to N

```text
{n >= 0}
i := 0;
s := 0;
while i < n do
  i := i + 1;
  s := s + i
{s = n*(n+1)/2}
```

Invariant: $I equiv s = i (i+1)/2 and i <= n$.

*Verification.* (1) Establishment: after `i := 0; s := 0`, $s = 0 = 0 dot 1 / 2$ and $0 <= n$. (2) Preservation: assume $I and i < n$; after `i := i+1`, the predicate $s = (i-1)i/2$ holds (since the new $i$ is the old $i$ plus one), then after `s := s + i`, $s = (i-1)i/2 + i = i(i+1)/2$, and $i <= n$ since old $i < n$. (3) On loop exit: $I and "not" (i < n)$ gives $i = n$ and $s = i(i+1)/2 = n(n+1)/2$.

The verification reduces to *three logical implications* over arithmetic. A theorem prover discharges them; the human discovers $I$.

== Partial vs. Total Correctness

*Partial correctness*: ${P} c {Q}$ holds <==> for every state $sigma$ with $sigma tack.r P$ and every $sigma'$ with $angle.l c, sigma angle.r arrow.r^* angle.l "skip", sigma' angle.r$, $sigma' tack.r Q$. Divergence is permitted.

*Total correctness*: $[P] c [Q]$ holds <==> additionally $c$ terminates from every state in which $P$ holds.

To prove termination of a loop, augment the invariant with a *variant* (or *ranking function*) $V$ -- an expression whose value lies in a well-founded order (typically $NN$) and strictly decreases each iteration.

```text
[H-While-Total]
    [I /\ b /\ V = z] c [I /\ V < z]    {I /\ b} V >= 0
    ---------------------------------------------------
    [I]  while b do c  [I /\ ~b]
```

For loops over arbitrary well-founded orders one uses an order $<_W$ and writes $V <_W z$. Termination of recursive procedures requires a similar well-founded measure on call arguments.

The boundary between partial and total correctness is the boundary between *liveness* and *safety* properties (Lamport): preservation rules are syntactic, termination requires a semantic well-foundedness argument.

== Soundness and Relative Completeness

Hoare logic is *sound* with respect to the operational semantics:

*Theorem (Soundness).* If $tack.r {P} c {Q}$ then for every $sigma$ with $sigma tack.r P$ and every $sigma'$ with $angle.l c, sigma angle.r arrow.r^* angle.l "skip", sigma' angle.r$, $sigma' tack.r Q$.

*Proof sketch.* Induction on the derivation of the triple, using the standard operational rules of IMP. The `H-While` case requires induction on the number of loop iterations, using the invariant. $square$

Completeness is more delicate. The natural statement -- "every true triple is provable" -- fails because the assertion language might be too weak to express the necessary invariants. Cook (1978) gave the right formulation.

*Theorem (Cook 1978, Relative Completeness).* Let $cal(L)$ be an *expressive* assertion language (one in which the strongest postcondition of every command can be expressed). Then for every triple ${P} c {Q}$ that is *true* under the operational semantics, $tack.r {P} c {Q}$ in Hoare logic, *relative* to an oracle for the validity of formulae in $cal(L)$.

*Proof sketch.* By induction on $c$, building a derivation whose loop invariants are *strongest* (or *weakest*) postconditions expressible in $cal(L)$. For `while b do c` with given $P$, set $I = "sp"(P, "while" b "do" c) or "the disjunction of all states reachable after some number of iterations"$; expressivity ensures $I$ is in $cal(L)$, and the loop rule discharges with weakening. The reliance on the assertion-language oracle is *unavoidable*: by Tarski's undefinability and Gödel's incompleteness, no recursive proof system for arithmetic is complete. $square$

The slogan: Hoare logic is *as complete as logic itself*. Any incompleteness lies in the underlying theory, not the program logic.

== Dijkstra's Weakest Preconditions

Dijkstra (1975, _Guarded Commands, Nondeterminacy, and Formal Derivation of Programs_) recast program logic as a *predicate transformer* calculus. Define

$ "wp"(c, Q) = "the weakest predicate" P "such that" {P} c {Q} "holds for total correctness." $

Then $tack.r {P} c {Q}$ <==> $P => "wp"(c, Q)$. The triple disappears; the verification condition is a single implication.

*Computing wp.*

$ "wp"("skip", Q) &= Q $
$ "wp"(x := e, Q) &= Q[e / x] $
$ "wp"(c_1 ; c_2, Q) &= "wp"(c_1, "wp"(c_2, Q)) $
$ "wp"("if" b "then" c_1 "else" c_2, Q) &= (b => "wp"(c_1, Q)) and ("not" b => "wp"(c_2, Q)) $
$ "wp"("while" b "do" c, Q) &= exists k. H_k (Q) $

where $H_0(Q) = Q and "not" b$, $H_(k+1)(Q) = H_0(Q) or (b and "wp"(c, H_k(Q)))$ enumerates the states from which the loop terminates in at most $k$ iterations. For partial correctness one writes $"wlp"$ (weakest liberal precondition) instead.

*Healthiness conditions* (Dijkstra). A predicate transformer $T : "Pred" arrow.r "Pred"$ is a $"wp"$ of some command <==> it satisfies:

1. *Law of Excluded Miracle*: $T("false") = "false"$ -- no command can establish the impossible.
2. *Monotonicity*: $Q_1 => Q_2$ => $T(Q_1) => T(Q_2)$.
3. *Conjunctivity*: $T(Q_1 and Q_2) = T(Q_1) "and" T(Q_2)$ -- for *deterministic* commands.
4. *Continuity*: $T(limits(exists)_i Q_i) = limits(exists)_i T(Q_i)$ for increasing chains.

Conjunctivity is the deterministic case; *demonic* nondeterminism (the adversary picks the branch) is captured by weaker conjunctivity (only on $forall$); *angelic* nondeterminism (the program picks) by disjunctivity.

*Program derivation.* Dijkstra's research programme was to *derive* programs from specifications by manipulating $"wp"$ equations: start with the postcondition, fix the invariant, compute $"wp"$ of a candidate program, and let the algebra suggest the program text. The methodology underlies the *refinement calculus*.

== Predicate Transformers, Demonic and Angelic

Nondeterminism splits the calculus.

- *Demonic*: $"wp"(c_1 [] c_2, Q) = "wp"(c_1, Q) and "wp"(c_2, Q)$. Both branches must establish $Q$ because the adversary chooses.
- *Angelic*: $"wp"^"A"(c_1 [] c_2, Q) = "wp"^"A"(c_1, Q) or "wp"^"A"(c_2, Q)$. Some branch must establish $Q$ because the program chooses.

The two are dual under De Morgan. The *refinement order* on commands is $c_1 subset.eq c_2$ <==> $"wp"(c_2, Q) => "wp"(c_1, Q)$ for all $Q$ -- $c_2$ refines $c_1$ if every postcondition guaranteed by $c_1$ is guaranteed by $c_2$. Refinement is reflexive and transitive; specifications are simply non-deterministic programs.

== The Refinement Calculus

Back's *refinement calculus* (1978, monograph 1998 with von Wright) and Morgan's *Programming from Specifications* (1990) made program derivation a formal activity. A *specification statement* $[P, Q]$ denotes any command that, from $P$, establishes $Q$. Refinement rules transform specifications into executable code:

```text
{P, Q}   sqsubseteq   if b then {P /\ b, Q} else {P /\ ~b, Q}    (case split)
{P, Q}   sqsubseteq   c1 ; {P', Q}   when {P, P'} c1 = id        (sequential)
{I /\ b, I /\ ~b}  sqsubseteq  while b do {I /\ b, I}             (loop introduction)
```

A program is *derived* by a sequence of refinement steps from the initial specification, each step justified by a refinement law. The end product is correct *by construction*; correctness is a property of the derivation, not of the result.

This methodology is implemented in tools such as Refine and the B-Method (Abrial), used in safety-critical software (Paris Metro Line 14 signaling, Airbus A380 flight control).

== Separation Logic

Hoare logic models the state as a single *map* and the assertion language has no good way to express *disjoint* updates. Two threads writing to disjoint regions, or a recursive procedure operating on a sublist of a linked list, force the verifier to write ugly aliasing conditions $x eq."not" y$ that cripple compositional reasoning.

*Separation logic* (Reynolds 2002; O'Hearn-Reynolds-Yang 2001) introduced two new connectives:

- *Separating conjunction* $P * Q$: "the heap splits into two disjoint pieces, one satisfying $P$ and the other $Q$." Formally, $sigma, h tack.r P * Q$ <==> $exists h_1, h_2. h = h_1 union.plus h_2 and sigma, h_1 tack.r P "and" sigma, h_2 tack.r Q$, where $union.plus$ is disjoint union.
- *Separating implication* (magic wand) $P "-*" Q$: "for every disjoint heap satisfying $P$, the combined heap satisfies $Q$."

Atomic assertions are *points-to*: $x |-> v$ means "the heap is exactly the singleton mapping address $x$ to value $v$."

=== The Frame Rule

The triumph of separation logic is *local reasoning* through the *frame rule*:

```text
[H-Frame]
    {P} c {Q}      (modifies(c) cap fv(R) = empty)
    -----------------------------------------------
    {P * R}  c  {Q * R}
```

A command $c$ that *operates* on the part of the heap described by $P$ leaves any disjoint frame $R$ untouched. The side condition is a syntactic check on free variables -- the *modifies* clause -- and is automatic.

The frame rule replaces page-long aliasing arguments with a single hypothesis: $R$ is disjoint, hence preserved. Verifications of linked-list manipulation that took dozens of pages in Hoare logic shrink to a single page in separation logic.

=== Example: List Reversal

The predicate $"list"(x, l)$ (defined inductively) says "from address $x$ the heap contains a null-terminated linked list whose values are the sequence $l$.

```text
{list(x, l)}
y := null;
while x != null do
  t := [x.next];
  [x.next] := y;
  y := x;
  x := t
{list(y, reverse(l))}
```

Loop invariant: $exists l_1, l_2. l = "reverse"(l_1) ++ l_2 and "list"(y, l_1) * "list"(x, l_2)$. The $*$ between the two list predicates is the key: it asserts the two list segments are disjoint in memory. The loop body manipulates only the head cell of the second segment, transferring it to the first; the frame rule lets us ignore the (potentially unbounded) rest.

=== Ramification and Hypothetical Frame

*Ramification*: a command operating on a *substructure* (a sublist, a subtree) of a larger structure can be verified locally, with the surrounding structure as a frame. The wand $"-*"$ encodes the *ramification operator*: $(P "-*" Q)$ specifies "what we promise to restore after running."

*Hypothetical frame*: in concurrent settings, the frame may not be physically separate but rather *protected by an invariant* held by the environment. This generalizes the frame rule to *resource invariants*.

== Concurrent Separation Logic

O'Hearn (2007, _Resources, Concurrency, and Local Reasoning_) extended separation logic to shared-memory concurrency. Each *resource* (a lock, a channel) is associated with a *resource invariant* $R$: a heap assertion that holds whenever the resource is *free*.

```text
[Par-Disjoint]
    {P1} c1 {Q1}    {P2} c2 {Q2}     (disjoint heaps and variables)
    --------------------------------------
    {P1 * P2}  c1 || c2  {Q1 * Q2}

[Acquire]
    --------------------------
    {emp}  acquire(L)  {R_L}

[Release]
    --------------------
    {R_L}  release(L)  {emp}
```

*Race-freedom* is a theorem: any program verifiable in CSL is race-free, because the only way to share heap is through a resource, and resources are mutually exclusive.

CSL handles bounded fine-grained synchronization (locks, semaphores) cleanly. *Higher-order* features (closures, callbacks), *atomic* operations (compare-and-swap), and *invariants* depending on protocol history demand more.

== Iris

*Iris* (Jung-Krebbers-Birkedal-Bizjak et al., 2015-) is a modern unifying separation logic. It is *higher-order* (assertions are propositions in a higher-order logic), *step-indexed* (to handle impredicative invariants and recursive types), and *user-extensible* (users define new resource algebras to fit the problem).

Key constructs:

- *Ghost state*. Auxiliary state not present in the program, used to track logical history. Resource algebras (PCMs, *partial commutative monoids*) provide a vocabulary for ghost state: counting permissions, fractional ownership, history tokens.
- *Invariants* $"Inv"(N)(I)$. An assertion $I$ shared by all threads, named by $N$, openable for a single atomic step.
- *View shifts* $P => ? Q$ ("or $|=> $). Logical updates that change ghost state without affecting the physical heap; analogous to weakening in classical logic but with resource-algebra structure.
- *Atomic triples* $angle.l P angle.r c angle.l Q angle.r$ (Svendsen-Birkedal). The pre and postcondition hold *atomically* around the linearization point of a concurrent operation.

Iris is fully formalized in Coq, with the *Iris Proof Mode* providing an interactive separation-logic-aware tactic language. It has been used to verify concurrent data structures (Michael-Scott queue, RDCSS), Rust's `Cell` and `RefCell` semantics (RustBelt, Jung et al. 2018), "and weak-memory models (iGPS).

== Verifying Linked Lists and Concurrent Stacks

A *concurrent stack* using compare-and-swap:

```text
push(x):
  loop:
    h := head;
    x.next := h;
    if CAS(&head, h, x) then return
    else goto loop
```

The linearization point is the successful CAS. In Iris, the resource algebra is the *authoritative-fragmental* construction `Auth(List(Val))`: the authority holds the true list, fragments hold ownership claims. The invariant says the heap representation matches the authoritative list. `push` proceeds by: (i) opening the invariant before CAS to access `head`; (ii) on success, atomically updating the ghost authoritative list and closing the invariant; (iii) on failure, retrying.

Iris formalizes such proofs to the point of machine-checked completeness; the same pattern verifies Treiber stacks, Michael-Scott queues, and hazard-pointer-based reclamation.

== Connection to Type Systems

Hoare logic and type systems are different presentations of *static reasoning*. Two convergence points:

- *Refinement types* (Freeman-Pfenning 1991; Vazou et al. for Liquid Haskell). A type ${nu : "Int" | nu > 0}$ refines `Int` with a logical predicate. Type checking discharges verification conditions to an SMT solver. This is *axiomatic semantics integrated into the type checker*.
- *Hoare Type Theory* (HTT, Nanevski-Morrisett-Birkedal 2008). The type $"Hoare"(P, A, Q)$ classifies effectful computations producing $A$ with precondition $P$ and postcondition $Q$. Sequencing is monadic bind; the whole apparatus is a *monad* (cf. _Categorical Semantics_). $F^*$ (Microsoft Research) is the production heir of HTT.

== Tool Ecosystem

Verification tools have moved from research artifact to industrial deployment.

- *Why3* (Bobot, Filliâtre et al.). A platform that translates first-order specifications to dozens of SMT solvers and proof assistants. Used as a back-end for many front-ends.
- *Dafny* (Leino, Microsoft Research). An imperative language with built-in pre/postcondition annotations, loop invariants, and an SMT-backed verifier. Used in production at AWS and Microsoft.
- *VeriFast* (Jacobs et al.). Separation-logic-based verifier for C and Java; used to verify embedded systems and Linux kernel modules.
- *F\** (FStarLang). Dependent types with a refinement-type sub-language, an effects system, and SMT-backed proof. Used in HACL\* (the verified cryptographic library powering Mozilla NSS and Linux kernel).
- *CN* (Pulte et al., Cambridge). Separation logic for C, with a focus on systems code; in active development for OpenSSL and PKVM (the hypervisor used in Android Pixel).
- *Steel* (Microsoft Research). Concurrent separation logic embedded in $F^*$, verifies concurrent data structures and lock-free algorithms.

== Beyond Functional Correctness

Modern program logics target properties beyond input-output correctness:

- *Cost reasoning*. *RAML* (Hoffmann et al.), *AARA*, and Iris's *time credits* attach resource bounds to Hoare triples: ${P * "time"(n)} c {Q * "time"(m)}$ proves $c$ uses at most $n - m$ time units.
- *Information flow*. *Relational Hoare logic* (Benton 2004) reasons about pairs of executions, expressing non-interference: $"hi"_1 tilde.equiv_"low" "hi"_2 => "exec"(c, "hi"_1) tilde.equiv_"low" "exec"(c, "hi"_2)$.
- *Probabilistic correctness*. *pRHL* and *EasyCrypt* extend separation logic with probabilistic assertions, used to verify cryptographic protocols.
- *Weak memory*. Logics for x86-TSO, ARM, and the C11 memory model (iGPS, Cosmo, RustBelt) extend separation logic with views, modalities, and protocol invariants.

== Summary

Axiomatic semantics has travelled from Floyd's annotated flowcharts to industrial verification of system kernels. The arc is: Hoare triples + relative completeness (Hoare 1969, Cook 1978); predicate transformers (Dijkstra 1975); refinement calculus (Back 1978, Morgan 1990); separation logic (Reynolds, O'Hearn, Yang 2001-2002) with its frame rule for local reasoning; concurrent separation logic (O'Hearn 2007); higher-order step-indexed separation logic with ghost state and invariants in Iris (2015-). Throughout, the methodology is the same: associate program text with logical claims; derive claims by syntactic rules; discharge any unavoidable mathematical content to a theorem prover. The reward, demonstrated in seL4, CompCert, HACL\*, and Iris-verified concurrent libraries, is software whose correctness is a *machine-checked theorem*.

_See also: Operational Semantics for the operational model over which soundness is stated; Type Systems for refinement types and Hoare type theory; Categorical Semantics for the monadic structure of effectful computation; Denotational Semantics for the predicate-transformer view as a model in the category of complete lattices._

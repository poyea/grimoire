= Separation Logic

Heap-manipulating programs are notoriously hard to verify with classical Hoare logic because *aliasing* — two pointers referring to the same memory location — breaks frame reasoning. A postcondition that describes one pointer's cell may silently invalidate a hypothesis about another that happens to alias it. Separation logic, introduced by Reynolds (2002) building on O'Hearn and Pym's logic of bunched implications (1999), extends Hoare logic with a *spatial conjunction* $P * Q$ and an empty-heap predicate $"emp"$ that let you assert ownership of disjoint heap regions. The result is a calculus in which the frame rule holds unconditionally: a proof about a small piece of heap composes freely with reasoning about the rest.

*See also:* _Axiomatic Semantics_, _Theorem Proving and Proof Assistants_, _Model Checking_

== The Heap Model

A *heap* is a finite partial function $h : "Addr" harpoon.rt "Val"$ mapping addresses to values; the domain $"dom"(h)$ is the set of live cells. Two heaps $h_1$ and $h_2$ are *disjoint*, written $h_1 bot h_2$, when $"dom"(h_1) inter "dom"(h_2) = emptyset$. Their *union* $h_1 * h_2$ is defined only when they are disjoint.

Heap assertions are predicates over pairs $(s, h)$ where $s$ is a store (variable environment) and $h$ is a heap:

- $"emp"$ — the heap is empty: $"dom"(h) = emptyset$.
- $p |-> v$ — the heap contains exactly the cell at address $p$ holding value $v$: $h = {p mapsto v}$.
- $p |->$ — shorthand $exists v. p |-> v$; the value is irrelevant.
- $P * Q$ (*separating conjunction*) — the heap splits into disjoint parts satisfying $P$ and $Q$ respectively: $exists h_1, h_2.\ h_1 bot h_2 and h = h_1 * h_2 and (s, h_1) models P and (s, h_2) models Q$.
- $P -* Q$ (*separating implication*, "magic wand") — for every disjoint heap satisfying $P$, the combined heap satisfies $Q$: $forall h'.\ h bot h' and (s, h') models P => (s, h * h') models Q$.

The magic wand $P -* Q$ expresses "if you hand me a disjoint $P$-heap, I can produce a $Q$-heap." It appears in specifications of procedures that consume a resource, and in the soundness proof of the frame rule.

=== Singly-Linked List Predicate

Inductive predicates capture unbounded data structures. The list predicate $"ls"(x)$ asserts that $x$ is the head of a well-formed null-terminated singly-linked list:

$ "ls"(x) &equiv (x = "null" and "emp") \
           &quad or (exists v, n.\ x |-> (v, n) * "ls"(n)) $

Each recursive case owns exactly the cells reachable from $x$; $*$ ensures no sharing. This is impossible to express cleanly in classical first-order logic without auxiliary length or reachability predicates.

== Hoare Logic Recap and the Aliasing Problem

Standard Hoare logic writes ${ P }\ C\ { Q }$ to mean: if $P$ holds in the initial state and $C$ terminates, then $Q$ holds in the final state. The classical *frame rule* is:

$ frac({ P }\ C\ { Q }, { P and F }\ C\ { Q and F }) $

This rule is *unsound* in the presence of aliasing. Consider:

```text
x := 1; [y] := 2
```

With precondition $P = (x |-> 1)$ and frame $F = (y |-> 1)$, if $x$ and $y$ alias the same address then the mutation `[y] := 2` destroys the cell that $F$ describes. The classical frame rule silently carries $F$ through, yielding the false postcondition $x |-> 1 and y |-> 2$ even though the program wrote $2$ to the shared cell.

Separation logic restores soundness by requiring that $C$ *does not touch* the frame's footprint. The frame rule becomes:

$ frac({ P }\ C\ { Q }, { P * R }\ C\ { Q * R }) quad (text("free variables of ") R text(" not modified by ") C) $

Because $*$ guarantees disjointness, $C$ cannot reach the $R$ portion of the heap. The rule is unconditionally sound.

== Core Inference Rules

The small axiom set covers the four primitive heap operations:

*Allocation.* A fresh cell is created and returned:

$ { "emp" }\ x := "alloc"(v)\ { x |-> v } $

*Lookup.* Reading a cell leaves it intact:

$ { x |-> v }\ y := [x]\ { x |-> v and y = v } $

*Mutation.* Writing replaces the old value; the wildcard $\_$ abbreviates existential quantification over the old value:

$ { x |-> \_ }\ [x] := v\ { x |-> v } $

*Deallocation.* Freeing a cell consumes it entirely:

$ { x |-> \_ }\ "free"(x)\ { "emp" } $

These four rules combined with the frame rule, the conjunction rule, and standard consequence give a complete basis for sequential heap reasoning. Every rule has a minimal *footprint*: it mentions only the cells it actually touches. The frame rule then scales any small proof to arbitrary surrounding heaps.

#table(
  columns: (auto, auto, auto),
  [*Rule*], [*Precondition*], [*Postcondition*],
  [Alloc], [$"emp"$], [$x |-> v$],
  [Lookup], [$x |-> v$], [$x |-> v and y = v$],
  [Mutation], [$x |-> \_$], [$x |-> v$],
  [Dealloc], [$x |-> \_$], [$"emp"$],
  [Frame], [$P * R$], [$Q * R$],
)

== Recursive Data Structures

=== List Segments

The *list segment* predicate $"lseg"(x, y)$ describes a linked list from $x$ up to (but not including) $y$:

$ "lseg"(x, y) equiv (x = y and "emp") or (exists v, n.\ x |-> (v, n) * "lseg"(n, y)) $

Key lemmas: $"lseg"(x, "null") equiv "ls"(x)$ and the composition $"lseg"(x, z) * "lseg"(z, y) => "lseg"(x, y)$.

=== Trees

$ "tree"(x) equiv (x = "null" and "emp") or (exists v, l, r.\ x |-> (v, l, r) * "tree"(l) * "tree"(r)) $

Each $*$ enforces that the left subtree, right subtree, and root cell occupy disjoint heap regions — ruling out DAG-sharing or back-edges without an explicit annotation.

=== In-Place List Reversal: Proof Sketch

The standard in-place reversal accumulates a reversed list:

```c
Node* rev(Node* x) {
    Node* y = NULL;
    while (x != NULL) {
        Node* t = x->next;
        x->next = y;
        y = x;
        x = t;
    }
    return y;
}
```

The loop invariant is $"ls"(x) * "ls"(y)$: $x$ holds the unreversed tail and $y$ the reversed prefix, on disjoint heap regions. At each iteration the frame rule isolates the head cell $x |-> (v, t)$; mutation and the list axiom reassemble the invariant. On exit $x = "null"$ collapses the left conjunct to $"emp"$, leaving $"ls"(y)$ — the full reversed list on the heap.

== Concurrent Separation Logic (O'Hearn 2004)

Classical separation logic is sequential. O'Hearn's *concurrent separation logic* (CSL) extends it to shared-memory concurrency by associating a *resource invariant* $I$ with each lock. Locking transfers ownership of $I$ to the thread; unlocking transfers it back. The key rules:

*Lock/unlock:*

$ { R * I }\ "lock"(m)\ { R * I } $

$ { R * I }\ "unlock"(m)\ { R } $

where $m$ protects invariant $I$ and $R$ is the thread's private state.

*Parallel composition:*

$ frac({ P_1 }\ C_1\ { Q_1 } quad { P_2 }\ C_2\ { Q_2 }, { P_1 * P_2 }\ C_1 parallel C_2\ { Q_1 * Q_2 }) $

The $*$ in the precondition ensures that $C_1$ and $C_2$ start with disjoint private heaps; shared state is mediated entirely through locked resource invariants. Data races are impossible to express in a valid proof: any access to shared state requires holding the relevant lock.

=== Connection to Rust's Borrow Checker

The spatial conjunction and ownership transfer of separation logic correspond directly to Rust's affine type system. A Rust `Box<T>` is $x |-> v$; moving it is the frame rule operating on a one-element heap; a mutable reference `&mut T` is the magic wand $P -* Q$. The borrow checker enforces, at compile time, the same disjointness invariants that CSL proves at the level of verification. This connection was made precise by Jung et al.'s RustBelt project (2018), which gave the first machine-checked soundness proof of a Rust-like type system using the Iris framework.

== Bi-Abduction (Calcagno et al. 2009)

Classical separation logic requires the analyst to supply loop invariants and procedure specifications by hand. *Bi-abduction* automates this by solving, for each call site, the inference problem:

$ H * X tack.r "footprint" * Y $

simultaneously discovering the *anti-frame* $X$ (the smallest precondition needed to make the call succeed) and the *frame* $Y$ (the leftover heap not consumed by the callee). The solution is found by a unification procedure over symbolic heaps; when multiple solutions exist, the procedure heuristically picks the smallest.

Bi-abduction enables *compositional interprocedural analysis*: procedures are analysed in isolation, their inferred summaries are cached, and callers compose summaries without re-analysing the callee. This scales to millions of lines of code where exhaustive whole-program analysis is infeasible.

*Facebook Infer* (open-sourced 2015) implements bi-abduction and runs as a continuous-integration check on every diff at Meta, finding null dereferences, resource leaks, and memory errors in Android (Java) and iOS (Objective-C/C) code before the change lands. Infer has been deployed at Amazon, Mozilla, Spotify, and hundreds of other organizations.

== Tools

#table(
  columns: (auto, auto, auto, auto),
  [*Tool*], [*Logic*], [*Backend*], [*Distinguishing feature*],
  [Smallfoot], [first-order SL], [symbolic execution], [original prototype; decision procedure for list/tree predicates],
  [VeriFast], [permission-based SL], [symbolic execution], [fractional permissions; Java and C; annotation-driven],
  [Facebook Infer], [bi-abduction], [abstract interpretation], [fully automatic; production CI at Meta scale],
  [Iris], [higher-order SL], [Coq proof assistant], [ghost state, invariants, Löb induction; machine-checked],
)

=== Iris in More Detail

Iris (Jung et al. 2015–2018) is a *higher-order* concurrent separation logic embedded in Coq. Its three distinguishing features are:

- *Ghost state* — abstract resources tracked only in proofs, not at runtime, modeled as a *resource algebra* (a partial commutative monoid with a validity predicate).
- *Invariants* — propositions that hold at all times, accessible by paying a "later" modality $triangle.r P$; this prevents the invariant from being assumed while proving it.
- *Löb induction* — the rule $( triangle.r P -> P) -> P$ provides a general fixpoint principle that gives sound inductive reasoning about recursive functions and protocols without requiring a separate termination argument.

RustBelt used Iris to prove that the Rust standard library's unsafe code (channels, `Mutex`, `Arc`) cannot cause undefined behaviour when called from safe Rust.

== Further Reading

Reynolds, J. C. (2002). "Separation Logic: A Logic for Shared Mutable Data Structures." #emph[LICS].

O'Hearn, P., Pym, D. (1999). "The Logic of Bunched Implications." #emph[Bulletin of Symbolic Logic].

O'Hearn, P. (2004). "Resources, Concurrency and Local Reasoning." #emph[CONCUR]. (Full version: #emph[TCS] 375, 2007.)

Calcagno, C., Distefano, D., O'Hearn, P., Yang, H. (2009). "Compositional Shape Analysis by Means of Bi-Abduction." #emph[POPL].

Jung, R., Krebbers, R., Jourdan, J.-H., Bizjak, A., Birkedal, L., Dreyer, D. (2018). "Iris from the Ground Up." #emph[Journal of Functional Programming].

Berdine, J., Calcagno, C., O'Hearn, P. (2005). "Smallfoot: Modular Automatic Assertion Checking with Separation Logic." #emph[FMCO].

Jacobs, B., Smans, J., Philippaerts, P., Vogels, F., Penninckx, W., Piessens, F. (2011). "VeriFast: A Powerful, Sound, Predictable, Fast Verifier for C and Java." #emph[NASA Formal Methods].

Gregg, B. (2020). #emph[Systems Performance: Enterprise and the Cloud, 2nd ed.] Pearson. (Chapter 1 discusses Infer's deployment at scale.)

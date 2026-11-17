#import "../template.typ": definition, xref

= Alloy and Relational Modelling <alloy>

Most design flaws are not deep. They are missing cases, wrong assumptions about how structures relate, and interactions nobody drew on the whiteboard. *Lightweight formal methods* aim at these flaws with the least possible ceremony: write a small, precise model of the design, then let a tool search exhaustively for counterexamples within a bounded size. Alloy, designed by Daniel Jackson at MIT, is the canonical example. Its language is first-order relational logic, its analyser translates bounded problems to SAT, and its working assumption is that most bugs have small counterexamples. This chapter covers the relational logic, the language, how the Kodkod engine reduces a model to SAT, the small scope hypothesis, temporal models in Alloy 6, a C++ implementation of the relational core with a bounded checker, and how Alloy compares with TLA+.

*See also:* #xref("formal-methods", "tla-plus", label: "TLA+"), #xref("formal-methods", "sat-and-smt", label: "SAT and SMT"), #xref("formal-methods", "propositional-and-fol", label: "Propositional and First-Order Logic"), #xref("formal-methods", "model-checking", label: "Model Checking").

== Everything Is a Relation

Alloy has one kind of value: a *relation*, a set of tuples of atoms, all of the same arity. A set is a unary relation, a scalar is a singleton unary relation, and a field `f: A -> B` is a binary relation. Atoms are uninterpreted and indivisible. This uniformity is what lets one small set of operators express navigation, set algebra, and graph properties.

#definition(name: "Relational join")[
For relations $p$ of arity $m$ and $q$ of arity $n$ with $m + n > 2$, the *join* $p . q$ is
$ {(a_1, dots, a_(m-1), b_2, dots, b_n) | exists c. (a_1, dots, a_(m-1), c) in p and (c, b_2, dots, b_n) in q}. $
]

Join generalises both function application and relational composition. If `contents: Dir -> Object` and `d` is a directory, `d.contents` is the set of its children, and `contents.o` is the set of directories that contain `o`. Navigation in either direction is the same operator with arguments swapped.

#table(
  columns: 3,
  [*Operator*], [*Meaning*], [*Example*],
  [`p.q`], [Join], [`d.contents`: children of `d`],
  [`~r`], [Transpose of a binary relation], [`~contents`: parent relation],
  [`^r`], [Transitive closure], [`d.^contents`: all descendants],
  [`*r`], [Reflexive transitive closure], [`Root.*contents`: everything reachable, including `Root`],
  [`+`, `&`, `-`], [Union, intersection, difference], [`Object - Root`],
  [`->`], [Cartesian product], [`Dir -> File`],
  [`<:`, `:>`], [Domain and range restriction], [`Dir <: contents`],
  [`in`, `=`], [Subset, equality], [`Object in Root.*contents`],
)

Multiplicity keywords state cardinalities: `one` (exactly one), `lone` (at most one), `some` (at least one), `no` (none), and `set` (any). They appear both as formula quantifiers (`one contents.o`) and in field declarations (`parent: lone Dir`).

== A First Model

A model declares *signatures* (sets of atoms) and their fields, constrains them with *facts*, and then either asks for an instance satisfying a predicate (`run`) or for a counterexample to an assertion (`check`).

```text
abstract sig Object {}
sig File extends Object {}
sig Dir extends Object { contents: set Object }
one sig Root extends Dir {}

fact Structure {
  Object in Root.*contents              -- everything is reachable from the root
  all o: Object - Root | one contents.o -- every non-root object has exactly one parent
  all d: Dir | d not in d.^contents     -- no directory is its own descendant
}

assert NoSelfContainment { no d: Dir | d in d.contents }
check NoSelfContainment for 5

pred Deep { some d: Dir | #d.^contents >= 3 }
run Deep for 4
```

The `check` command searches every instance with up to five atoms in each signature and finds no counterexample, which is expected since acyclicity implies the assertion. Deleting the third fact makes the analyser report a counterexample at once with a single atom: `Root` containing itself. The second fact applies only to non-root objects, so nothing forbade it. Such gaps are typical of informal designs, and the analyser finds them in milliseconds.

The `run` command is equally important. A model whose facts are contradictory satisfies every assertion vacuously, because it has no instances at all. Running predicates that should be satisfiable, and looking at the instances the visualiser draws, is how modellers check that the facts are not *over-constrained*.

== Bounded Analysis and the Small Scope Hypothesis

First-order relational logic with transitive closure is undecidable, so Alloy does not try to prove assertions for all sizes. A *scope* bounds the number of atoms in each signature (default 3), and within that bound the analysis is exhaustive: if a counterexample exists with at most that many atoms, Alloy finds it.

The *small scope hypothesis* is the empirical claim that justifies this: most bugs in designs and code have counterexamples within small scopes. Experiments that mutated programs and specifications and checked how large the smallest counterexamples were found that the great majority appear with scopes of three to five. The hypothesis is not a theorem. Bugs that need many atoms (counter overflows, long rings, deep queues) are missed, and Alloy users raise scopes for the signatures where size matters.

Integers are bounded too. Alloy's `Int` is a two's complement bit-vector of configurable width, 4 bits by default, so integers range from $-8$ to $7$ and arithmetic wraps. Models that rely on counting should either raise the bitwidth or, better, avoid integers in favour of relational structure.

== From Relations to SAT: Kodkod

Alloy's engine, Kodkod, reduces a bounded relational problem to boolean satisfiability.

+ *Bounds.* Each relation $r$ gets a lower bound $L_r$ (tuples it must contain) and an upper bound $U_r$ (tuples it may contain), as sets of tuples over the finite universe of atoms.
+ *Boolean matrices.* Each relation becomes a matrix with one entry per tuple in $U_r$: constant true for tuples in $L_r$, a fresh boolean variable for tuples in $U_r - L_r$, and constant false elsewhere. A binary relation over $n$ atoms needs at most $n^2$ variables.
+ *Operators as matrix operations.* Union is entrywise OR, intersection is AND, join is boolean matrix multiplication ($(p . q)_(i k) = or.big_j p_(i j) and q_(j k)$), and closure is computed by $ceil(log_2 n)$ rounds of squaring.
+ *Formulas as circuits.* Quantifiers over bounded domains expand to finite conjunctions and disjunctions. The resulting boolean circuit is shared aggressively (identical subterms become one gate) and converted to CNF.
+ *Symmetry breaking.* Atoms of the same signature are interchangeable, so every instance has many isomorphic copies. Kodkod adds lex-leader constraints that rule out most copies, which can shrink the search by orders of magnitude.

The SAT solver's model maps back to an instance, which the visualiser draws as a graph. Partial instances, where the user fixes some tuples in advance, become tighter lower bounds and cost nothing to support.

== The Relational Core in C++

The engine's semantics fit in a few lines over bit matrices. The code below implements join, transpose, closure, and the relational properties used in models, and uses them to check a classic false conjecture exhaustively: "a symmetric and transitive relation is reflexive".

```cpp
#include <bitset>
#include <cstdint>
#include <iostream>
#include <optional>

template <int N>
struct relation {
    std::bitset<N * N> bits;

    bool has(int i, int j) const { return bits[i * N + j]; }
    void set(int i, int j) { bits.set(i * N + j); }

    static relation identity() {
        relation r;
        for (int i = 0; i < N; ++i) r.set(i, i);
        return r;
    }

    relation join(const relation& q) const {
        relation r;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (has(i, j))
                    for (int k = 0; k < N; ++k)
                        if (q.has(j, k)) r.set(i, k);
        return r;
    }

    relation transpose() const {
        relation r;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                if (has(i, j)) r.set(j, i);
        return r;
    }

    relation closure() const {   // ^r, by Warshall's algorithm
        relation r = *this;
        for (int k = 0; k < N; ++k)
            for (int i = 0; i < N; ++i)
                if (r.has(i, k))
                    for (int j = 0; j < N; ++j)
                        if (r.has(k, j)) r.set(i, j);
        return r;
    }

    bool subset_of(const relation& q) const { return (bits & ~q.bits).none(); }
};

template <int N> bool symmetric(const relation<N>& r) { return r.transpose().bits == r.bits; }
template <int N> bool transitive(const relation<N>& r) { return r.join(r).subset_of(r); }
template <int N> bool reflexive(const relation<N>& r) { return relation<N>::identity().subset_of(r); }

// check  all r: A -> A | symmetric[r] and transitive[r] implies reflexive[r]  for exactly N atoms
template <int N>
std::optional<relation<N>> find_counterexample() {
    constexpr std::uint64_t count = std::uint64_t{1} << (N * N);
    for (std::uint64_t m = 0; m < count; ++m) {
        relation<N> r;
        r.bits = std::bitset<N * N>(m);
        if (symmetric(r) && transitive(r) && !reflexive(r)) return r;
    }
    return std::nullopt;
}

int main() {
    if (auto r = find_counterexample<2>())
        std::cout << "counterexample: " << r->bits << '\n';
}
```

The first counterexample is the empty relation, for which symmetry and transitivity hold vacuously and reflexivity fails. The informal "proof" of the conjecture (from $a R b$ get $b R a$ by symmetry, then $a R a$ by transitivity) assumes each element is related to something, and the smallest counterexample shows exactly that missing assumption. This is the small scope hypothesis at work: a scope of one atom already refutes the claim. Brute force covers $2^(n^2)$ relations and stops being feasible around $n = 6$; Kodkod's SAT encoding handles scopes well beyond that, because the solver prunes the space instead of enumerating it.

== Dynamic Models and Alloy 6

Classic Alloy has no built-in notion of state change. Dynamic systems were modelled by adding an explicit `Time` or `State` signature and threading it through every mutable field, with predicates relating consecutive states. This works but is verbose and makes traces awkward to read.

Alloy 6 (2021) absorbed the Electrum extension and added *mutable* signatures and fields, marked `var`, together with linear temporal logic: `always`, `eventually`, `after`, `until`, past operators such as `once` and `historically`, and primed variables for the next state. A `check` with temporal formulas searches for a counterexample trace. Bounded analysis unrolls traces up to a configurable number of steps using SAT, and Alloy can also delegate to the nuXmv or Electrod back ends for unbounded model checking of finite instances.

```text
sig Process { var holds: lone Lock }
sig Lock {}

fact Init { no holds }
fact Steps {
  always (stutter or some p: Process, l: Lock | acquire[p, l] or release[p, l])
}
pred stutter { holds' = holds }
pred acquire[p: Process, l: Lock] {
  no p.holds and no holds.l
  holds' = holds + p -> l
}
pred release[p: Process, l: Lock] {
  p.holds = l
  holds' = holds - p -> l
}

assert MutualExclusion { always all l: Lock | lone holds.l }
check MutualExclusion for 3 but 10 steps
```

Each step is one acquire, one release, or a stutter step that changes nothing; without the stutter option, a state where no action is enabled would have no infinite continuation and would silently drop out of the analysis. The assertion says no lock is ever held by two processes. Removing the guard `no holds.l` from `acquire` produces a two-step counterexample trace.

== Alloy in Practice

The best-known Alloy result is Pamela Zave's analysis of the Chord distributed hash table. Modelling Chord's ring-maintenance protocol in Alloy, she showed that the published protocol was not correct: under failures and joins, the ring could break or split in ways the original paper said could not happen. She then developed and verified a corrected version. The analysis used scopes of only a few nodes, and every counterexample was small enough to read and understand.

Other uses include analysing access-control and network configurations, checking database schemas and object models, exploring the consistency of hardware memory models (where litmus tests are small relational structures), and teaching. Forge, a language derived from Alloy's ideas and built on a Kodkod-style engine, is used to teach formal methods at Brown.

== Alloy and TLA+

#table(
  columns: 3,
  [*Aspect*], [*Alloy*], [*TLA+*],
  [Core logic], [First-order relational logic with closure], [Set theory with temporal logic of actions],
  [Strength], [Structure: relations, configurations, object models], [Behaviour: state machines, concurrency, liveness],
  [Analysis], [Bounded SAT search over all instances in a scope], [Explicit-state model checking of a finite instance (TLC); proofs (TLAPS)],
  [Output], [Instances and counterexamples drawn as graphs], [Error traces as sequences of states],
  [Typical use], [Exploring a design's data model early], [Validating a concurrent or distributed protocol],
)

The two overlap more since Alloy 6 added temporal logic, and many teams use both: Alloy to get the structure of a design right, TLA+ to get its concurrency right.

== Pitfalls

- Over-constrained facts that make every assertion vacuously true. Always `run` predicates that should have instances.
- Scopes too small for the property, such as rings of two nodes or queues of one element.
- Integer arithmetic at the default 4-bit width silently wrapping.
- Treating "no counterexample found" as a proof. It is evidence for the stated scope only.
- Modelling too much. Alloy is most useful on an abstract model of the essential structure, not on a transcription of the implementation.

== Further Reading

Jackson, D. (2012). _Software Abstractions: Logic, Language, and Analysis_ (revised edition). MIT Press.

Jackson, D. (2002). "Alloy: A Lightweight Object Modelling Notation." ACM Transactions on Software Engineering and Methodology.

Torlak, E. and Jackson, D. (2007). "Kodkod: A Relational Model Finder." TACAS.

Jackson, D. (2019). "Alloy: A Language and Tool for Exploring Software Designs." Communications of the ACM.

Andoni, A. et al. (2002). "Evaluating the Small Scope Hypothesis." MIT CSAIL technical report.

Macedo, N. et al. (2016). "Lightweight Specification and Analysis of Dynamic Systems with Rich Configurations." FSE.

Zave, P. (2012). "Using Lightweight Modeling to Understand Chord." ACM SIGCOMM Computer Communication Review.

Wickerson, J. et al. (2017). "Automatically Comparing Memory Consistency Models." POPL.

Nelson, T. et al. (2024). "Forge: A Tool and Language for Teaching Formal Methods." OOPSLA.

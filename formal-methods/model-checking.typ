#import "../template.typ": xref

= Model Checking

Model checking is the algorithmic verification of finite-state (or finitely abstractable) systems against temporal specifications. Given a Kripke structure $cal(K)$ and a property $phi$ in LTL, CTL, or the modal $mu$-calculus, the model checker either certifies $cal(K) models phi$ or returns a concrete counterexample trace. Three decades of research have produced explicit-state, symbolic (BDD-based), bounded ($"SAT"$-based), and IC3/PDR algorithms — each tuned for a different state-space regime.

*See also:* #xref("formal-methods", "sat-and-smt", label: "SAT and SMT"), #xref("formal-methods", "propositional-and-fol", label: "Propositional and First-Order Logic"), #xref("formal-methods", "tla-plus", label: "TLA+"), _Omega-Automata_ (programming-languages)

== Kripke Structures and Transition Systems

A *Kripke structure* over atomic propositions $"AP"$ is $cal(K) = (S, S_0, R, L)$ with states $S$, initial states $S_0 subset.eq S$, total transition relation $R subset.eq S times S$, and labeling $L: S -> 2^("AP")$. A *path* is an infinite sequence $pi = s_0 s_1 s_2 ...$ with $(s_i, s_(i+1)) in R$. The *trace* of $pi$ is the infinite word $L(s_0) L(s_1) ... in (2^("AP"))^omega$.

Model checking asks: does every initial path satisfy $phi$? Equivalently, is the set of traces of $cal(K)$ contained in the language $cal(L)(phi)$?

== Linear and Branching Temporal Logic

*LTL* (linear-time) describes single paths:

#table(
  columns: (auto, auto, auto),
  [*Operator*], [*Meaning*], [*Read as*],
  [$bold(X) phi$], [next: $pi_1 models phi$], [neXt],
  [$bold(F) phi$], [$exists i. pi_i models phi$], [Finally / Eventually],
  [$bold(G) phi$], [$forall i. pi_i models phi$], [Globally / Always],
  [$phi bold(U) psi$], [$phi$ holds until $psi$ (which must eventually hold)], [Until],
  [$phi bold(R) psi$], [release: dual of U], [Release],
)

*CTL* (branching-time) quantifies over paths from each state with $bold(A)$ (all paths) and $bold(E)$ (exists path), paired with a path operator. $bold(A G) phi$ means "$phi$ holds along every path from here." $bold(E F) phi$ means "some reachable state satisfies $phi$."

LTL and CTL are incomparable in expressivity. CTL\* subsumes both. The modal $mu$-calculus $L_mu$ subsumes CTL\*: $mu Z. phi$ (least fixpoint) and $nu Z. phi$ (greatest) encode reachability and safety directly. See `programming-languages/omega-automata.typ` for the language-theoretic side.

== Automata-Theoretic LTL Model Checking

The classical Vardi-Wolper construction reduces LTL model checking to language emptiness:

+ Translate $not phi$ to a Büchi automaton $cal(A)_(not phi)$ accepting traces violating $phi$.
+ Form the product $cal(K) times cal(A)_(not phi)$.
+ Check whether the product has a *fair cycle* reachable from an initial state.

The check is a *nested DFS* (Courcoubetis-Vardi-Wolper-Yannakakis):

```python
# Nested DFS for accepting cycle in a Büchi automaton (Holzmann/SPIN style)
def nested_dfs(init, succ, is_accept):
    visited_outer = set()
    visited_inner = set()
    on_stack = []          # DFS stack for outer search

    def outer(s):
        visited_outer.add(s)
        on_stack.append(s)
        for t in succ(s):
            if t not in visited_outer:
                if outer(t): return True
        if is_accept(s):
            if inner(s, s): return True
        on_stack.pop()
        return False

    def inner(s, seed):
        visited_inner.add(s)
        for t in succ(s):
            if t == seed:                  # closed an accepting cycle
                return True
            if t not in visited_inner:
                if inner(t, seed): return True
        return False

    return any(outer(s) for s in init)
```

Memory: $O(|S| dot 2^(|phi|))$ for the product and two bits per state.

== Symbolic Model Checking with BDDs

For systems whose state space is too large to enumerate ($10^(20)$ states or more), McMillan's *symbolic* approach (1992) represents *sets of states* and the transition relation as *Reduced Ordered Binary Decision Diagrams* (ROBDDs).

A BDD over Boolean variables $x_1 < x_2 < ... < x_n$ is a canonical DAG: each internal node tests one variable, with two children. Reduction rules (eliminate redundant tests, share isomorphic subgraphs) give canonicity *given an ordering*. The size depends critically on the ordering — finding the optimum is NP-hard, but heuristics (sifting, group sifting in CUDD) work well in practice.

```text
BDD for f(x1,x2,x3) = (x1 /\ x2) \/ x3,  ordering x1 < x2 < x3

         x1
        /  \
      0/    \1
      v      v
      x3     x2
     /  \   /  \
   0/    \ /    \1
   v      X      v
   F      |      x3
          v     /  \
                F   T
```

*Image computation.* Reachable states from $S$ under $R$ are $"Img"(S) = exists overline(x). (S(overline(x)) and R(overline(x), overline(x)'))$, computable as a BDD existential quantification. Fixpoint iteration $"Reach" = mu Z. (S_0 or "Img"(Z))$ terminates because BDDs over finite variables form a finite lattice. EFG and similar CTL fixpoints reduce to BDD operations.

#table(
  columns: (auto, auto),
  [*CTL operator*], [*Symbolic computation*],
  [$bold(E X) phi$], [$exists overline(x)'. (R(overline(x), overline(x)') and phi(overline(x)'))$],
  [$bold(E F) phi$], [$mu Z. (phi or bold(E X) Z)$],
  [$bold(E G) phi$], [$nu Z. (phi and bold(E X) Z)$],
  [$bold(E)(phi bold(U) psi)$], [$mu Z. (psi or (phi and bold(E X) Z))$],
  [$bold(A G) phi$], [$not bold(E F)(not phi)$],
)

== Partial-Order Reduction

For concurrent systems, many interleavings of independent transitions yield equivalent traces. *Partial-order reduction* (POR) explores only one representative per equivalence class. The two standard frameworks:

- *Ample sets* (Peled): at each state pick a non-empty subset of enabled transitions to explore, subject to four conditions ensuring no LTL_X property is missed. Implemented in SPIN.
- *Stubborn sets* (Valmari): a dual formulation focusing on transitions that must fire to enable a target.

POR can reduce state spaces by exponential factors and is mandatory for verifying real concurrent code at scale.

== Bounded Model Checking

BMC (Biere-Cimatti-Clarke-Zhu 1999) unrolls the transition relation $k$ steps and asks a $"SAT"$ solver whether a counterexample of length $<= k$ exists:

$ I(s_0) and product_(i=0)^(k-1) R(s_i, s_(i+1)) and product_(i=0)^k not phi(s_i) $

If $"SAT"$, the assignment gives a concrete trace. If UNSAT for all $k$ up to the *completeness threshold* (the diameter of the system), the property holds. Computing tight thresholds is hard; in practice BMC excels at bug-finding, not full proofs.

```cpp
// Sketch: BMC unrolling in C++ with a SAT solver API
#include <vector>

struct State { std::vector<int> vars; };  // SAT variables for one step

State unroll_step(SatSolver& s, const State& prev) {
    State cur{ s.fresh_vars(prev.vars.size()) };
    encode_transition(s, prev, cur);      // R(prev, cur) clauses
    return cur;
}

bool bmc_check(SatSolver& s, int k, Formula bad) {
    State s0 = s.fresh_state();
    encode_initial(s, s0);
    State cur = s0;
    for (int i = 0; i <= k; ++i) {
        s.push();
        encode_assert(s, bad, cur);       // assert violation at step i
        auto res = s.solve();
        if (res == SAT) return true;      // counterexample of length i
        s.pop();
        if (i < k) cur = unroll_step(s, cur);
    }
    return false;                         // no cex of length <= k
}
```

== IC3 / PDR: Property-Directed Reachability

Bradley's IC3 (2011), refined to PDR by Een-Mishchenko-Brayton, was the breakthrough that let $"SAT"$-based engines do *unbounded* model checking. IC3 maintains a sequence of *frames* $F_0, F_1, ..., F_k$ — over-approximations of states reachable in at most $i$ steps. Each frame is a CNF formula; the algorithm:

+ Push lemmas (clauses) from $F_i$ to $F_(i+1)$ when they remain inductive.
+ When a $"SAT"$ query finds a counterexample-to-induction (CTI), recursively block its predecessors.
+ Property proved when $F_(i+1) subset.eq F_i$ for some $i$ (a fixpoint of strengthening).

IC3/PDR with localized $"SAT"$ queries scales to hardware designs with millions of latches and is the engine inside ABC, Vampire's avatar, and the Symbiotic-EVA stack. Strengths: incremental, learns small inductive invariants. Weakness: bad at deep counterexamples, where BMC wins.

== CEGAR: Abstraction-Refinement

Counterexample-Guided Abstraction Refinement (Clarke-Grumberg-Jha-Lu-Veith 2000) is the dominant paradigm for software model checking:

+ Build an abstraction $cal(K)^#h(0.1em)$ (predicate abstraction over a finite set of predicates).
+ Model-check $cal(K)^#h(0.1em) models phi$.
+ If the abstract counterexample $pi^*$ is spurious (no concrete realization), use Craig interpolation or an unsat core to discover new predicates that rule it out.
+ Refine and repeat.

SLAM, BLAST, CPAchecker, and SeaHorn all implement variants. The chapter on _Program Verification_ revisits CEGAR for heap-manipulating code.

== State-Space Explosion: Sizes in Practice

#table(
  columns: (auto, auto, auto),
  [*Domain*], [*Reachable states*], [*Best technique*],
  [Cache-coherence protocol (4 cores)], [$10^4$], [Explicit-state (Mur$phi$)],
  [Hardware control (small)], [$10^(20)$], [BDDs (NuSMV, VIS)],
  [Hardware datapath], [$10^(100)+$], [BMC, IC3 ($"SAT"$)],
  [Distributed protocol], [unbounded], [TLA+ TLC w/ symmetry],
  [Concurrent code], [unbounded heap], [Stateful (CBMC, SeaHorn), CEGAR],
  [Cryptographic protocol], [symbolic], [ProVerif, Tamarin],
)

== Symmetry Reduction and Data Abstraction

Many systems exhibit symmetry: process identifiers are interchangeable, addresses can be permuted. Quotienting by the symmetry group $G$ reduces the state space by up to $|G|$. TLC (TLA+'s checker) and Mur$phi$ exploit symmetry annotations.

*Data abstraction* replaces a concrete domain with a finite one preserving the property of interest: e.g., abstract integers to ${-, 0, +}$ for a sign-related property. Combined with predicate abstraction this gives the foundation for software model checking.

== Liveness vs Safety, Fairness

*Safety* properties ("nothing bad happens") have finite-prefix counterexamples; equivalent to checking reachability of bad states. *Liveness* ("something good eventually happens") requires infinite (lasso) counterexamples and Büchi-style emptiness. Liveness without *fairness* is usually trivially violated by a process that never executes — fairness constraints (weak fairness: if continuously enabled, eventually fires; strong fairness: if infinitely often enabled, eventually fires) restrict the path quantifier to fair paths.

== Tool Landscape

#table(
  columns: (auto, auto, auto),
  [*Tool*], [*Niche*], [*Specification*],
  [SPIN], [explicit-state, concurrent protocols], [LTL, Promela],
  [NuSMV / nuXmv], [symbolic (BDD + $"SAT"$ + IC3)], [LTL, CTL, SMV],
  [TLC], [TLA+ specs], [TLA+, PlusCal],
  [ABC], [hardware (AIG, IC3, BMC)], [property files],
  [CBMC], [bounded MC of C/C++], [assertions],
  [Kani], [bounded MC of Rust], [Rust properties],
  [JBMC], [bounded MC of Java bytecode], [JML, asserts],
  [SeaHorn], [Horn-clause based for LLVM IR], [LLVM assertions],
  [CPAchecker], [configurable software MC], [SV-COMP properties],
  [ProB], [B / Event-B], [B method specs],
  [mCRL2], [process algebra], [$mu$-calculus],
)

== Worked Example: Mutual Exclusion

A two-process Peterson-style flag protocol, model-checked for safety ($bold(A G) not (text("crit")_1 and text("crit")_2)$) and liveness ($bold(A G)(text("try")_1 -> bold(A F) text("crit")_1)$).

```text
-- NuSMV input fragment
MODULE proc(other_flag, my_flag, turn, me)
VAR
  pc : {idle, try, crit};
ASSIGN
  init(pc) := idle;
  next(pc) := case
    pc = idle              : {idle, try};
    pc = try & (!other_flag | turn = me) : crit;
    pc = crit              : idle;
    TRUE                   : pc;
  esac;
  next(my_flag) := (next(pc) = try) | (next(pc) = crit);

MODULE main
VAR
  flag1 : boolean;
  flag2 : boolean;
  turn  : {1,2};
  p1    : proc(flag2, flag1, turn, 1);
  p2    : proc(flag1, flag2, turn, 2);
SPEC AG !(p1.pc = crit & p2.pc = crit)              -- safety
LTLSPEC G (p1.pc = try -> F p1.pc = crit)           -- liveness (needs fairness)
FAIRNESS running
```

NuSMV's BDD engine proves the safety property in milliseconds; the liveness property requires fairness on each process.

== Software Model Checking vs Hardware Model Checking

#table(
  columns: (auto, auto, auto),
  [*Aspect*], [*Hardware*], [*Software*],
  [State], [bit-vectors, latches], [heap, stack, globals],
  [Concurrency], [synchronous clocks], [interleaving, weak memory],
  [Abstraction], [bit-level, often complete], [predicate / shape, lossy],
  [Spec], [LTL, assertions, equivalence], [assertions, contracts],
  [Engines], [BDDs, IC3, BMC], [CEGAR, symbolic execution, BMC],
  [Result], [near-routine], [research frontier for unbounded],
)

== Further Reading

Clarke, E., Grumberg, O., Peled, D., et al. (2018). #emph[Model Checking, 2nd ed.] MIT Press.

Baier, C., Katoen, J.-P. (2008). #emph[Principles of Model Checking.] MIT Press.

McMillan, K. (1993). #emph[Symbolic Model Checking.] Kluwer.

Biere, A., Cimatti, A., Clarke, E., Zhu, Y. (1999). "Symbolic Model Checking without BDDs." #emph[TACAS].

Bradley, A. (2011). "SAT-Based Model Checking without Unrolling." #emph[VMCAI].

Clarke, E., Grumberg, O., Jha, S., Lu, Y., Veith, H. (2000). "Counterexample-Guided Abstraction Refinement." #emph[CAV].

Holzmann, G. (2003). #emph[The SPIN Model Checker.] Addison-Wesley.

Cimatti, A. et al. (2014). "The nuXmv Symbolic Model Checker." #emph[CAV].

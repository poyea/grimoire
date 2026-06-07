= SAT and SMT

Boolean satisfiability and satisfiability modulo theories are the computational engines behind nearly every modern formal-methods tool. A $"SAT"$ solver decides whether a propositional formula in CNF has a satisfying assignment; an $"SMT"$ solver lifts this to formulas containing arithmetic, arrays, bit-vectors, and other rich theories. Together they power bounded model checking, symbolic execution, program synthesis, and compiler verification at industrial scale.

*See also:* _Propositional and First-Order Logic_, _Model Checking_, _Theorem Proving_, _TLA+_

== The SAT Problem

Given a CNF formula $phi = C_1 and C_2 and ... and C_m$ over variables $x_1, ..., x_n$, *SAT* asks for an assignment $nu: {x_1, ..., x_n} -> {0,1}$ satisfying every clause, or a proof that none exists. Cook's theorem (1971) established SAT as NP-complete, yet modern *Conflict-Driven Clause Learning* (CDCL) solvers routinely handle instances with $10^7$ variables — a gap between worst-case theory and engineering practice unmatched in computing.

== DPLL: The Foundation

The *Davis-Putnam-Logemann-Loveland* algorithm (1960-62) is a recursive backtracking search augmented by two simplification rules:

- *Unit propagation:* if a clause is a unit (single unassigned literal), that literal must be true; set it and simplify.
- *Pure literal elimination:* if a variable appears only positively (or only negatively), set it to satisfy all those clauses and remove them.

```text
DPLL(F):
  F := unit_propagate(F)
  if F = {} then return SAT
  if {} in F then return UNSAT
  x := choose_variable(F)
  return DPLL(F[x=1]) or DPLL(F[x=0])
```

Unit propagation is the key inner loop — a single application is $O(|F|)$, and cascading *unit propagation chains* prune huge portions of the search tree.

== CDCL: Conflict-Driven Clause Learning

Modern solvers extend DPLL with *non-chronological backjumping* and *clause learning* (Marques-Silva & Sakallah, 1996; Moskewicz et al., 2001). When a conflict (empty clause under the current partial assignment) is detected, the solver:

+ Computes a *conflict clause* from an *implication graph* using the *first unique implication point* (1-UIP) cut.
+ *Learns* that clause by adding it permanently to the formula.
+ *Backjumps* non-chronologically to the second-highest decision level in the learned clause.
+ *Restarts* periodically (Luby or geometric schedules) to escape bad regions of the search space.

```python
# CDCL skeleton (simplified)
def cdcl(clauses, n_vars):
    db = list(clauses)           # clause database (grows via learning)
    trail = []                   # (literal, reason_clause | None)
    level = 0
    assignment = {}

    while True:
        conflict = unit_propagate(db, assignment, trail)
        if conflict is not None:
            if level == 0:
                return "UNSAT"
            learned, back_level = analyze_conflict(conflict, trail, assignment)
            db.append(learned)
            backjump(trail, assignment, back_level)
            level = back_level
        else:
            unassigned = pick_unassigned(assignment, n_vars)
            if unassigned is None:
                return "SAT", assignment
            level += 1
            lit = decide(unassigned)   # heuristic
            trail.append((lit, None))
            assignment[abs(lit)] = (lit > 0)
```

The implication graph records which propagations triggered which — the 1-UIP cut yields the asserting clause that immediately propagates after backjumping, maintaining unit propagation momentum.

== VSIDS and Other Heuristics

*Variable State Independent Decaying Sum* (VSIDS), introduced in Chaff (2001), assigns each variable an *activity score* incremented whenever it appears in a conflict clause. Scores decay by a multiplicative factor periodically, so recent conflicts dominate. Modern variants (EVSIDS in Glucose, LRB in MapleSAT) refine the decay dynamics. Complements include:

- *Phase saving:* when re-deciding a variable, reuse the last assigned polarity. Empirically reduces restarts by 30-50% on structured instances.
- *VSIDS bump propagation:* bump all literals in the reason chain, not just the conflict clause.

== Preprocessing and In-processing

Industrial solvers spend significant time *before and during* search on preprocessing:

#table(
  columns: (auto, auto),
  [*Technique*], [*Effect*],
  [Variable elimination (BVE)], [resolve away a variable; exponential in worst case, often effective],
  [Subsumption], [remove clauses subsumed by shorter ones],
  [Self-subsuming resolution (strengthening)], [shorten clauses],
  [Bounded variable addition (BVA)], [introduce new variables to compress the formula],
  [Probing / failed literal], [detect forced assignments via unit propagation],
  [Vivification], [shorten clauses by propagation-based simplification],
)

CaDiCaL and Kissat implement *in-processing*: these transformations interleave with CDCL search rather than running only at the start, adapting to what the solver has learned.

== From SAT to SMT

*Satisfiability Modulo Theories* extends SAT to formulas over richer domains. An *SMT formula* is a Boolean combination of *theory atoms* — equalities, inequalities, array reads, bit-vector expressions. The *SMT problem* asks: is the formula satisfiable in some interpretation respecting the background theory $T$?

The dominant architecture is *DPLL(T)* (Nieuwenhuis, Oliveras, Tinelli, 2006): a CDCL SAT solver acts as the *Boolean engine*, assigning truth values to theory atoms; a *theory solver* $"T-solver"$ checks consistency of the current partial assignment in $T$ and returns a *theory conflict clause* when inconsistent.

```text
DPLL(T) interaction loop:
  SAT solver proposes partial assignment A over Boolean abstraction
  T-solver checks A for T-consistency
  if T-consistent: extend or declare SAT
  if T-inconsistent: T-solver returns conflict clause C (theory lemma)
                     SAT solver learns C and backtracks
```

== Theory Combination: Nelson-Oppen

Most SMT formulas involve *multiple theories* simultaneously — e.g., arrays with linear arithmetic indices and bit-vector element types. *Nelson-Oppen combination* (1979) combines two or more stably-infinite, signature-disjoint theories:

+ Each theory solver reasons independently over its atoms.
+ Shared terms are given *purified* proxy variables.
+ Solvers *propagate equalities* between shared variables: when $T_1$ deduces $x = y$, it notifies $T_2$, and vice versa.
+ The combined formula is $T_1 union T_2$-satisfiable iff both solvers reach a jointly consistent state.

For non-convex theories (e.g., integer arithmetic), the combination requires *case-splitting* on equalities, typically handled by the DPLL layer.

== Equality and Uninterpreted Functions (EUF)

The *theory of equality with uninterpreted functions* (EUF) adds no axioms beyond the equality axioms and *congruence*: $a_1 = b_1, ..., a_n = b_n => f(a_1, ..., a_n) = f(b_1, ..., b_n)$. The EUF solver maintains a *congruence closure* data structure — a union-find augmented with a *use list* per equivalence class, refreshed on each merge to propagate congruences in near-linear time. EUF is essential for reasoning about function calls in program verification.

== Linear Arithmetic

The *theory of linear arithmetic over the rationals* ($"QF_LRA"$) is decided by the *Simplex method* adapted for incremental SMT use. The Simplex tableau represents the current linear constraints; the solver pivots to satisfy them or derives a Farkas certificate of infeasibility. For integers ($"QF_LIA"$), *branch-and-bound* and *cutting-plane* methods (Gomory cuts) complete the decision procedure, though integer linear arithmetic is NP-complete.

The Simplex implementation in Z3 (de Moura-Dutertre, 2006) uses *bland's rule* for termination and *bound tightening* to propagate integer constraints eagerly.

== Bit-Vectors

The *theory of fixed-width bit-vectors* ($"QF_BV"$) models machine arithmetic exactly: addition wraps modulo $2^n$, shifts are defined bitwise, and overflow is explicit. Two main decision strategies:

- *Bit-blasting:* encode each bit-vector operation as a Boolean circuit and hand the result to the SAT solver. Complete and efficient for small widths or when the SAT solver is very fast.
- *Word-level reasoning:* propagate bounds and equalities at the word level before bit-blasting; used in Bitwuzla's preprocessing.

Bit-vectors underlie virtually all hardware verification and compiler correctness tools.

== The Theory of Arrays

McCarthy's *theory of arrays* (1962) has two axioms:

$ "read"("write"(a, i, v), i) = v $
$ i eq.not j => "read"("write"(a, i, v), j) = "read"(a, j) $

The SMT $"QF_AX"$ solver implements *array congruence closure*: an extension of EUF that handles read-over-write lemmas lazily, instantiating them only when a potential conflict exists. Arrays combine seamlessly with linear arithmetic via Nelson-Oppen.

== SMT Solvers in Practice

#table(
  columns: (auto, auto, auto),
  [*Solver*], [*Strengths*], [*Key theories*],
  [Z3 (Microsoft Research)], [widest theory coverage, Python/C API, tactics], [$"EUF"$, $"LIA"$, $"LRA"$, $"BV"$, $"FP"$, strings, ADTs],
  [CVC5 (Stanford/Iowa/NYU)], [quantifier instantiation, SyGuS synthesis], [$"EUF"$, $"LIA"$, $"BV"$, sequences, sets],
  [Bitwuzla (U Freiburg)], [best-in-class bit-vector solving], [$"BV"$, $"FP"$, arrays],
  [Yices 2 (SRI)], [fast for $"QF"$ fragments], [$"EUF"$, $"LIA"$, $"BV"$],
  [MathSAT 5 (FBK)], [interpolation, optimization], [$"EUF"$, $"LIA"$, $"BV"$],
  [OpenSMT2 (USI)], [proof production, interpolation], [$"EUF"$, $"LRA"$],
)

== A Z3 Python Example

The following snippet encodes a small bounded model-checking query directly in Z3's Python API: it checks whether a simple loop `x = x + 3` starting at 0 can reach a value divisible by 5 within 4 steps.

```python
from z3 import Int, Solver, And, Or, sat

def bmc_divisible_by_5(bound=4):
    s = Solver()
    xs = [Int(f"x_{i}") for i in range(bound + 1)]

    # Initial condition
    s.add(xs[0] == 0)

    # Transition: x_{i+1} = x_i + 3
    for i in range(bound):
        s.add(xs[i + 1] == xs[i] + 3)

    # Property violation: some x_i is divisible by 5
    s.add(Or(*[xs[i] % 5 == 0 for i in range(1, bound + 1)]))

    if s.check() == sat:
        m = s.model()
        trace = [m[xs[i]] for i in range(bound + 1)]
        print("Counterexample:", trace)
    else:
        print("No violation within bound", bound)

bmc_divisible_by_5()
# bound=4: trace [0,3,6,9,12] — no multiple of 5 => UNSAT within 4 steps.
# bound=5: trace [0,3,6,9,12,15] — 15 % 5 == 0 at step 5 => SAT (counterexample found).
```

Z3's `Int` sort maps to the theory of integers ($"LIA"$); the modulo constraint is handled by the arithmetic solver. The Python API translates directly to SMT-LIB2 internally.

== Applications

=== Bounded Model Checking

BMC (see _model-checking.typ_) unrolls a transition relation $k$ steps and asks an SMT solver whether a bad state is reachable. SMT (rather than pure SAT) allows the transition relation to be expressed in linear arithmetic or bit-vectors without bit-blasting the full datapath.

=== Symbolic Execution

*Symbolic execution* (King 1976; KLEE, SAGE, S2E) executes a program with symbolic rather than concrete inputs. At each branch, the *path condition* — a conjunction of SMT constraints — is extended and checked for satisfiability. SMT solvers decide which branches are feasible and generate concrete test inputs. $"QF_BV"$ handles machine integers; $"QF_AX"$ handles memory arrays.

=== Compiler Verification and Optimization

*Alive2* (Lopes et al., 2021) encodes LLVM optimization correctness as SMT queries: given a source and target LLVM IR peephole pattern, it constructs a $"QF_BV"$ formula asserting the target is not a refinement of the source, then checks satisfiability. Hundreds of LLVM bugs have been found this way.

=== Synthesis

*Syntax-Guided Synthesis* (SyGuS) asks: given a specification $phi(x, P(x))$, find a program $P$ satisfying it. The dominant approach is a CEGIS (Counterexample-Guided Inductive Synthesis) loop using an SMT oracle for both verification and counterexample generation.

== SMT-LIB2

The *SMT-LIB2* standard (Barrett, Fontaine, Tinelli, 2015) defines a common input format and theory library. Solvers compete on SMT-COMP benchmarks organized by logic (combination of theories):

#table(
  columns: (auto, auto),
  [*Logic tag*], [*Meaning*],
  [$"QF_UF"$], [quantifier-free EUF],
  [$"QF_LIA"$], [quantifier-free linear integer arithmetic],
  [$"QF_BV"$], [quantifier-free bit-vectors],
  [$"QF_AUFBV"$], [arrays + EUF + bit-vectors, no quantifiers],
  [$"LIA"$], [linear integer arithmetic with quantifiers],
  [$"NIA"$], [nonlinear integer arithmetic],
  [$"FP"$], [floating-point arithmetic (IEEE 754)],
)

Quantified fragments remain challenging — most practical tools keep quantifiers out of the hot path through instantiation heuristics (e-matching, model-based quantifier instantiation).

== Further Reading

Biere, A., Heule, M., van Maaren, H., Walsh, T. (eds.) (2021). _Handbook of Satisfiability, 2nd ed._ IOS Press.

Nieuwenhuis, R., Oliveras, A., Tinelli, C. (2006). "Solving SAT and SAT Modulo Theories." _JACM_ 53(6).

de Moura, L., Bjørner, N. (2008). "Z3: An Efficient SMT Solver." _TACAS_.

Barbosa, H. et al. (2022). "cvc5: A Versatile and Industrial-Strength SMT Solver." _TACAS_.

Lopes, N. P. et al. (2021). "Alive2: Bounded Translation Validation for LLVM." _PLDI_.

Barrett, C., Sebastiani, R., Seshia, S., Tinelli, C. (2021). "Satisfiability Modulo Theories." In _Handbook of Satisfiability, 2nd ed._

Moskewicz, M. et al. (2001). "Chaff: Engineering an Efficient SAT Solver." _DAC_.

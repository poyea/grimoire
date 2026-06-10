= TLA+

TLA+ (Temporal Logic of Actions, Lamport 1994) is a formal specification language designed for describing and reasoning about concurrent and distributed systems at the level of their state machines rather than their code. A TLA+ specification is a mathematical formula — a predicate over behaviors (infinite sequences of states) — that any correct implementation must satisfy. The same formalism supports both automated model checking (TLC) and machine-assisted deductive proof (TLAPS), and it has been adopted by Amazon Web Services, Microsoft Azure, MongoDB, and others to find subtle bugs in complex distributed protocols.

*See also:* _Model Checking_, _Propositional and First-Order Logic_

== The Mathematical Foundation

A *behavior* is an infinite sequence of states $s_0, s_1, s_2, ...$. A *state* is a function from variable names to values. TLA+ reasons about behaviors using:

- *State formulas:* ordinary predicates over the current state (no priming).
- *Actions:* predicates over pairs of consecutive states, written with primed variables for the next state. The *stuttering action* $"UNCHANGED" << v_1, ..., v_n >>$ asserts all listed variables are unchanged.
- *Temporal formulas:* Boolean combinations of state formulas, actions, and temporal operators.

The temporal operators are those of *Linear Temporal Logic* (LTL), written in TLA+ notation:

#table(
  columns: (auto, auto, auto),
  [*Operator*], [*TLA+ syntax*], [*Meaning*],
  [Always], [`[]P`], [P holds in every state of the behavior],
  [Eventually], [`<>P`], [P holds in at least one state],
  [Leads-to], [`P ~> Q`], [whenever P holds, Q eventually holds afterward],
  [Weak fairness], [`WF_v(A)`], [if A is continuously enabled, it eventually occurs],
  [Strong fairness], [`SF_v(A)`], [if A is infinitely often enabled, it eventually occurs],
)

The central concept is *stuttering equivalence*: two behaviors are equivalent if one can be obtained from the other by duplicating states (adding stuttering steps). TLA+ formulas are stuttering-invariant — adding or removing stuttering steps does not change truth. This makes it natural to relate specifications at different abstraction levels.

== Specification Structure

A TLA+ module has the form:

```tla
---- MODULE ModuleName ----
EXTENDS Naturals, Sequences, FiniteSets   \* import standard modules
CONSTANTS C1, C2, ...                     \* uninterpreted constants
VARIABLES v1, v2, ...                     \* state variables

Init == ...           \* initial predicate
Next == ...           \* next-state action
Spec == Init /\ [][Next]_vars /\ Fairness  \* complete specification
====
```

The *spec* `Init /\ [][Next]_vars` asserts: the initial state satisfies `Init`, and every step is either a `Next` step or a stuttering step (the `[...]_vars` notation). Fairness assumptions (`WF` or `SF`) are layered on top to rule out stuttering-forever behaviors.

== Actions and Primed Variables

Actions describe transitions. Primed variable `v'` refers to the value of `v` in the *next* state. An action holds of a step $(s, t)$ if the action formula is true when unprimed variables take values from $s$ and primed variables take values from $t$.

```tla
\* A counter that increments by 1, staying below N
Increment == /\ counter < N
             /\ counter' = counter + 1
             /\ UNCHANGED other_vars
```

Multiple actions are combined with disjunction (`\/`) to form `Next`:

```tla
Next == Increment \/ Reset \/ Skip
```

== Temporal Operators in Detail

*Safety* properties have the form `[][Next]_vars` or `[](P => Q)` and can be violated by a finite prefix. *Liveness* properties use `<>` and `~>` and require infinite behaviors.

The *leads-to* operator $P tilde.op Q$ is defined as $[](P => <> Q)$ — it is not a primitive but a derived temporal operator. It composes: if $P tilde.op Q$ and $Q tilde.op R$ then $P tilde.op R$ (transitivity holds).

*Weak fairness* `WF_v(A)` means: it is not the case that `A` is continuously enabled and yet never taken. Formally:

$ "WF"_v(A) = []([]"Enabled"(angle.l A angle.r_v) => <>angle.l A angle.r_v) $

where $angle.l A angle.r_v$ is the action $A$ ignoring stuttering on $v$. *Strong fairness* `SF_v(A)` strengthens this to: if `A` is enabled infinitely often, it must fire infinitely often.

== PlusCal

*PlusCal* is an algorithm language that compiles to TLA+. It provides an imperative syntax closer to pseudocode while retaining TLA+'s mathematical semantics.

```text
--algorithm Mutex
variables flag = [p \in {1,2} |-> FALSE], turn = 1;

process (Proc \in {1,2})
variables pc_local = "idle";
begin
  Trying:
    flag[self] := TRUE;
  WaitTurn:
    await ~flag[3 - self] \/ turn = self;
  Critical:
    skip;
  Exit:
    flag[self] := FALSE;
    turn := 3 - self;
    goto Trying;
end process;
end algorithm;
```

The PlusCal compiler generates a TLA+ module from this, creating one action per labeled step. Safety properties (mutual exclusion) and liveness properties (no starvation) can then be checked by TLC or proved by TLAPS.

== Worked Example: Producer-Consumer with a Bounded Buffer

```tla
---- MODULE ProducerConsumer ----
EXTENDS Naturals, Sequences

CONSTANT N   \* buffer capacity
VARIABLE buf, produced, consumed

TypeInvariant ==
  /\ buf \in Seq(Nat)
  /\ Len(buf) <= N

Init ==
  /\ buf = << >>
  /\ produced = 0
  /\ consumed = 0

Produce(item) ==
  /\ Len(buf) < N
  /\ buf' = Append(buf, item)
  /\ produced' = produced + 1
  /\ UNCHANGED consumed

Consume ==
  /\ Len(buf) > 0
  /\ buf' = Tail(buf)
  /\ consumed' = consumed + 1
  /\ UNCHANGED produced

Next == \/ \E item \in Nat : Produce(item)
        \/ Consume

Fairness ==
  /\ WF_buf(Consume)
  /\ SF_buf(\E item \in Nat : Produce(item))

Spec == Init /\ [][Next]_<<buf, produced, consumed>> /\ Fairness

\* Safety: buffer never overflows (follows from TypeInvariant)
Safety == Len(buf) <= N

\* Liveness: every produced item is eventually consumed
\* (approximate: consumed eventually catches up to produced)
Progress == produced > 0 ~> consumed > 0

====
```

TLC can model-check this with `N = 3` and a finite set of items to verify `Safety` and `Progress`. The fairness conditions ensure the consumer always eventually runs and the producer eventually produces.

== TLC: The TLA+ Model Checker

*TLC* is an explicit-state model checker for TLA+ specifications. It:

- Evaluates TLA+ over finite models (finite constant instantiations, bounded integers).
- Explores the reachable state space using BFS or DFS.
- Checks invariants (safety properties) and liveness (via acceptance cycle detection with fairness).
- Supports *symmetry reduction* for specifications with symmetric constant sets (e.g., process IDs).
- Supports *simulation mode* (random walk) for large state spaces where exhaustive search is infeasible.

TLC configuration:

```text
SPECIFICATION Spec
INVARIANT TypeInvariant
INVARIANT Safety
PROPERTY Progress
CONSTANTS N <- 3
SYMMETRY Symmetry
```

TLC has found bugs in the Paxos, Raft, Zookeeper ZAB, and ViewStamped Replication protocols — bugs that had survived years of code review and testing.

=== Performance Characteristics

#table(
  columns: (auto, auto, auto),
  [*State space size*], [*TLC behavior*], [*Mitigation*],
  [$< 10^6$ states], [seconds to minutes], [default settings],
  [$10^6$-$10^9$ states], [minutes to hours; distributed TLC], [symmetry, view, action constraints],
  [$> 10^9$ states], [exhaustive infeasible], [simulation mode, abstraction],
)

Distributed TLC can spread the state graph across many workers on a cluster, enabling verification of state spaces with tens of billions of states.

== TLAPS: The TLA+ Proof System

*TLAPS* (Chaudhuri, Doligez, Lamport, Merz, 2010) is a proof system for TLA+. Proofs are written as hierarchical *proof outlines* inside the TLA+ module. TLAPS calls backend provers (Isabelle/HOL, Zenon, Z3, CVC5) to discharge leaf obligations.

```tla
THEOREM Inv == Spec => []TypeInvariant
<1>1. Init => TypeInvariant
  BY DEF Init, TypeInvariant
<1>2. TypeInvariant /\ [Next]_vars => TypeInvariant'
  BY DEF TypeInvariant, Next, Produce, Consume, vars
<1>3. QED
  BY <1>1, <1>2, PTL DEF Spec
```

The proof structure mirrors the standard inductive invariant method: show the invariant holds initially (`<1>1`), is preserved by every step (`<1>2`), and conclude by temporal reasoning (`PTL` = propositional temporal logic, handled by the TLAPS temporal reasoner).

TLAPS has been used to verify:

- Safety and liveness of Paxos (Lamport's own proof).
- The Mojave distributed storage protocol.
- Memory model properties for hardware specifications.

== Real-World Adoption

*Amazon Web Services* has used TLA+ since 2011 to verify S3, DynamoDB, EBS, and internal distributed components. Their 2015 experience report (Newcombe et al.) describes finding 14 critical bugs — including one that would have caused data loss under a legal but unusual sequence of network partitions — bugs that passed code review and extensive testing. AWS now has hundreds of engineers trained in TLA+.

*Microsoft Azure Cosmos DB* uses TLA+ to verify the consistency protocols underlying its five consistency models (strong, bounded staleness, session, consistent prefix, eventual). The specifications were instrumental in finding protocol errors before implementation.

*MongoDB* uses TLA+ to verify the Raft implementation in its replication protocol, including snapshot installation and log compaction edge cases.

*PGo* (UBC) compiles PlusCal to Go, enabling specifications to be refined to runnable implementations with partial automation.

== Refinement and Specification Hierarchies

TLA+ supports *specification refinement*: a concrete specification $S$ *implements* an abstract specification $A$ if every behavior of $S$ (projected to the variables of $A$) is a behavior of $A$.

Formally, $S "implements" A$ means $S => A$ as temporal formulas, after appropriate substitution. Lamport's *refinement mappings* define how concrete state variables map to abstract ones. This lets you:

+ Prove a high-level protocol correct against a simple invariant.
+ Prove a low-level implementation correct against the high-level protocol.
+ Compose the two to get end-to-end correctness.

The two-phase commit specification in Lamport's TLA+ book demonstrates this: a simple atomic-transaction spec is refined by an explicit two-phase commit protocol, then further refined by a fault-tolerant recovery extension.

== Tool Ecosystem

#table(
  columns: (auto, auto),
  [*Tool*], [*Purpose*],
  [TLC], [explicit-state model checker],
  [TLAPS], [interactive proof assistant for TLA+],
  [PlusCal], [algorithm language compiling to TLA+],
  [Toolbox / VS Code extension], [IDE support, TLC integration],
  [Apalache], [SMT-based symbolic model checker for TLA+ (TypedTLA+)],
  [PGo], [PlusCal-to-Go compiler for executable refinement],
  [tla2json / tla2tex], [export tools for specs and counterexamples],
)

*Apalache* (Konnov et al.) extends TLC with an SMT-based backend, handling infinite-state properties that TLC's enumeration cannot: it translates bounded-length TLA+ executions into SMT queries, finding bugs in protocols with unbounded integer domains.

== Further Reading

Lamport, L. (2002). #emph[Specifying Systems: The TLA+ Language and Tools for Hardware and Software Engineers.] Addison-Wesley (free PDF at lamport.org).

Newcombe, C. et al. (2015). "How Amazon Web Services Uses Formal Methods." #emph[CACM] 58(4).

Chaudhuri, K., Doligez, D., Lamport, L., Merz, S. (2010). "Verifying Safety Properties with the TLA+ Proof System." #emph[IJCAR].

Lamport, L. (1994). "The Temporal Logic of Actions." #emph[TOPLAS] 16(3).

Konnov, I. et al. (2019). "TLA+ Model Checking Made Symbolic." #emph[OOPSLA].

Kuppe, M., Lamport, L., Schulz, D. (2019). "The TLA+ Toolbox." #emph[TLAPM Workshop].

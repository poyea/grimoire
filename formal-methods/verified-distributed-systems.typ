#import "../template.typ": definition, theorem, proof, xref

= Verifying Distributed Systems <verified-distributed-systems>

Distributed protocols are short, and their bugs are notoriously hard to find. A few hundred lines of consensus or replication logic interact with message reordering, loss, duplication, crashes, restarts, and partitions, and the failing executions are rare interleavings that tests almost never produce and that are nearly impossible to reproduce in production. They are also the components where a bug means lost or inconsistent data. This makes distributed systems the area where formal methods have paid off most clearly in industry. This chapter covers the main techniques in order of increasing assurance: deterministic simulation and history checking, model checking of specifications and implementations, inductive invariants and how to find them (with a C++ checker for inductiveness), decidable fragments as in Ivy, and full refinement proofs of implementations as in IronFleet and Verdi.

*See also:* #xref("formal-methods", "tla-plus", label: "TLA+"), #xref("formal-methods", "model-checking", label: "Model Checking"), #xref("formal-methods", "deductive-verification", label: "Auto-Active Deductive Verification"), #xref("distributed-systems", "consensus-deep-dive", label: "Consensus Deep Dive"), #xref("distributed-systems", "introduction", label: "Introduction: Models and Impossibilities"), #xref("observability-and-sre", "chaos-engineering", label: "Chaos Engineering").

== Systems as Transition Systems

All the techniques share one model. A distributed system is a state machine whose state is the product of each node's local state and the *network*, a multiset of messages in flight. A step is one node receiving one message (or a timer firing), updating its local state, and adding messages to the network. Failures are more steps: a message is dropped or duplicated, a node crashes and loses volatile state, a node restarts. The asynchronous model allows any enabled step at any time, which is the source of the interleavings that make these systems hard.

Properties come in two kinds. *Safety* says nothing bad ever happens: two nodes never both believe they are leader of the same term, a committed write is never lost, reads are linearisable. *Liveness* says something good eventually happens: a client request eventually completes, given a stable leader and timely delivery. Safety violations have finite counterexamples, which is why most verification effort targets safety. Liveness requires fairness assumptions, and by the FLP result it cannot be guaranteed for deterministic consensus in a fully asynchronous system with even one crash failure.

== Testing That Takes Interleavings Seriously

=== Deterministic Simulation

FoundationDB's developers built the database so that the whole cluster (many nodes, the network, disks, and clocks) runs in a single-threaded simulator under one random seed. Every source of nondeterminism goes through interfaces the simulator controls. The simulator injects faults far more often than reality would (network partitions, disk failures, machine reboots, clock jumps) and checks invariants continuously. Any failure is reproducible exactly from its seed. The team credits simulation with catching the large majority of their bugs before they reached production. TigerBeetle and commercial systems such as Antithesis apply the same idea, the latter to unmodified software by making a whole virtual machine deterministic.

=== History Checking

Jepsen tests real clusters from the outside: clients run operations while a nemesis process partitions the network, kills nodes, and skews clocks. Every invocation and completion is logged into a *history*, and a checker decides whether the history is consistent with the claimed model. For linearisability, the Knossos and Porcupine checkers search for a sequential order that respects real-time precedence. For transactional isolation, Elle infers dependency edges (write-write, write-read, read-write) between transactions from the values observed and searches for cycles that each isolation level forbids. Elle's insight is that designing workloads so each read reveals which writes it saw (for example, appending unique elements to lists) makes the dependency graph recoverable in linear time instead of requiring an exponential search.

=== Stateless Model Checking of Implementations

For concurrency within one process, tools like CHESS, Coyote, Loom, and Shuttle take control of the scheduler and systematically explore thread interleavings of real code, bounding the number of preemptions to keep the search tractable. Most concurrency bugs need only one or two preemptions to appear, so a small bound covers a large fraction of real bugs. AWS's ShardStore team combined this with property-based tests against executable reference models of each component, a style they call lightweight formal methods.

== Model Checking Specifications

TLA+ with TLC explores all behaviours of a specification for a small configuration (three nodes, two values, a few messages) and returns a shortest counterexample trace when a property fails. The P language (Microsoft, later AWS) instead writes the system as communicating state machines in a programming-language syntax and explores schedules systematically; it was used for the Windows USB stack and for validating the protocols behind S3's move to strong consistency. MongoDB used TLA+ for its replication protocol and then checked conformance by tracing implementation executions against the specification.

Model checking a specification gives high confidence for the configurations checked and finds most design bugs, which are typically small. It does not prove the protocol for all cluster sizes, and it does not say anything about the implementation unless a link between code and specification is established.

== Inductive Invariants

To prove safety for every configuration, verification turns to invariants. Let $"Init"$ be the initial states, $"Next"$ the transition relation, and $P$ the safety property.

#definition(name: "Inductive invariant")[
A predicate $I$ is an *inductive invariant* for $P$ if
(1) $"Init" => I$, (2) $I and "Next" => I'$, and (3) $I => P$.
]

#theorem[
If $I$ is an inductive invariant for $P$, then $P$ holds in every reachable state.
]

#proof[
By induction on the length of an execution. The initial state satisfies $I$ by (1); each step preserves $I$ by (2); so every reachable state satisfies $I$ and therefore $P$ by (3).
]

The difficulty is that the property itself is almost never inductive. A state that satisfies $P$ but is not reachable may step to a state that violates $P$. Such a state is a *counterexample to induction* (CTI). The verifier's job is to strengthen $P$ with additional conjuncts until no CTI remains, while keeping (1) true.

Consider a lock service with a server and $n$ clients. The server holds a flag `server_free`. A client asks for the lock; when the server is free it clears the flag and sends a `grant` message to the client. A client that receives `grant` sets `holds`. A holder can release by clearing `holds` and sending `release` to the server, which then sets `server_free` again. The safety property is mutual exclusion: at most one client holds the lock.

```cpp
#include <bitset>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

constexpr int clients = 3;

// State layout: bit 0 = server_free; for client c:
//   bit 1 + 3c = holds[c], bit 2 + 3c = grant in flight to c, bit 3 + 3c = release in flight from c.
constexpr int state_bits = 1 + 3 * clients;
using state = std::bitset<state_bits>;

int holds_bit(int c) { return 1 + 3 * c; }
int grant_bit(int c) { return 2 + 3 * c; }
int release_bit(int c) { return 3 + 3 * c; }

std::vector<state> successors(const state& s) {
    std::vector<state> out;
    for (int c = 0; c < clients; ++c) {
        if (s[0]) {                                    // server grants the lock to c
            state t = s; t.reset(0); t.set(grant_bit(c)); out.push_back(t);
        }
        if (s[grant_bit(c)]) {                         // c receives the grant
            state t = s; t.reset(grant_bit(c)); t.set(holds_bit(c)); out.push_back(t);
        }
        if (s[holds_bit(c)]) {                         // c releases
            state t = s; t.reset(holds_bit(c)); t.set(release_bit(c)); out.push_back(t);
        }
        if (s[release_bit(c)]) {                       // server receives the release
            state t = s; t.reset(release_bit(c)); t.set(0); out.push_back(t);
        }
    }
    return out;
}

bool mutual_exclusion(const state& s) {
    int holders = 0;
    for (int c = 0; c < clients; ++c) holders += s[holds_bit(c)];
    return holders <= 1;
}

bool one_token(const state& s) { return s.count() == 1; }

bool initial(const state& s) { return s.count() == 1 && s[0]; }

using predicate = std::function<bool(const state&)>;

// Checks conditions (1) to (3) by enumerating every state of the bounded instance.
bool check_inductive(const predicate& inv, const predicate& prop, const char* name) {
    for (std::uint32_t m = 0; m < (1u << state_bits); ++m) {
        const state s(m);
        if (initial(s) && !inv(s)) { std::cout << name << ": fails initiation\n"; return false; }
        if (!inv(s)) continue;
        if (!prop(s)) { std::cout << name << ": does not imply the property\n"; return false; }
        for (const state& t : successors(s))
            if (!inv(t)) {
                std::cout << name << ": CTI " << s << " -> " << t << '\n';
                return false;
            }
    }
    std::cout << name << ": inductive\n";
    return true;
}

int main() {
    check_inductive(mutual_exclusion, mutual_exclusion, "mutual exclusion");
    check_inductive([](const state& s) { return one_token(s); }, mutual_exclusion, "one token");
}
```

The first check reports a CTI: a state where one client holds the lock *and* a grant is in flight to another client. Delivering the grant creates a second holder. That state is unreachable, but nothing in the bare property rules it out. The same holds for a holder coexisting with a free server, which reaches two holders in two steps. The strengthened invariant says that exactly one "token" exists among the server flag, the grant messages, the holders, and the release messages. It is preserved by every step (each step moves the token), it holds initially, and it implies mutual exclusion. This is typical: the strengthening captures *why* the protocol is correct, and finding it is the intellectual content of the proof.

The enumeration checks one bounded instance. To prove the invariant for any number of clients, conditions (1) to (3) must be checked symbolically, as validity of formulas with quantifiers over clients. That is where solvers and decidable logics come in.

== Decidable Fragments: Ivy

Ivy models protocols in a restricted first-order logic designed so that checking inductiveness is decidable. Its core is the *effectively propositional* fragment (EPR): formulas whose quantifier prefix is $exists^* forall^*$ with no function symbols that create unbounded terms in the wrong direction. Validity of such formulas reduces to SAT over a finite set of ground terms, so the verifier always terminates and, when an invariant is not inductive, it returns a finite CTI that can be displayed graphically.

Not every protocol fits EPR directly. Padon and colleagues showed that Paxos and several of its variants can be verified in EPR after rewriting the model: introducing derived relations that abstract the troublesome quantifier alternations, and proving a few lemmas outside the fragment. The payoff is predictability: verification never times out, and failures always come with a concrete explanation.

Ivy's interactive workflow mirrors the C++ example: show a CTI, let the user generalise it into a new conjunct ("no client holds the lock while the server is free"), and repeat. Research systems (I4, DistAI, IC3PO, DuoAI) automate the search for invariants by checking small instances, collecting candidate predicates from reachable states, and generalising them to all sizes, and they now infer invariants for many textbook protocols automatically.

== Refinement Proofs of Implementations

A verified protocol says nothing about the code that runs in production. Closing that gap needs a proof that the implementation *refines* the protocol: every behaviour of the implementation corresponds to a behaviour allowed by the protocol specification.

*IronFleet* (Microsoft Research) structured this in Dafny as three layers. A high-level specification states what the service does (a linearisable key-value store, a replicated state machine). A protocol layer describes the distributed algorithm over abstract data structures with an explicit network. An implementation layer is executable Dafny code, compiled and run as a real system. Proofs connect each layer to the next, and the implementation-to-protocol proof is done one node and one step at a time, using a *reduction* argument to treat each handler as atomic. IronFleet proved both safety and liveness for a Paxos-based replicated state machine and a sharded key-value store, with performance within a small factor of unverified implementations.

*Verdi* (University of Washington) took a different structure in Coq. Systems are written and proved correct against an idealised network model (no drops, no duplicates, no crashes) and then passed through *verified system transformers* that add fault-tolerance mechanisms, such as retransmission or Raft-based replication, while preserving the proven properties. The Raft proof in Verdi was one of the first full machine-checked proofs of a practical consensus implementation and took on the order of 50,000 lines of Coq.

More recent work in Verus and Grove (on Perennial and Iris in Coq) verifies distributed and crash-safe storage systems in languages closer to production (Rust and Go), using separation logic to reason about crashes and concurrent access to local state.

== Choosing a Level of Assurance

#table(
  columns: 4,
  [*Technique*], [*What it covers*], [*Typical effort*], [*Finds or proves*],
  [Deterministic simulation], [The real implementation under injected faults], [Architectural: the system must be built for it], [Finds bugs, reproducibly],
  [Jepsen-style history checking], [The deployed system, black box], [Days per system], [Finds consistency violations],
  [Model checking a specification], [The design, for small configurations], [Days to weeks per protocol], [Finds design bugs; proves nothing for larger sizes],
  [Inductive invariant proofs (TLAPS, Ivy, Dafny)], [The design, for all configurations], [Weeks to months], [Proves design safety],
  [Refinement proofs of code], [The implementation, for all configurations], [Person-years], [Proves implementation correctness against the specification],
)

Most organisations get the best return from the first three rows and reserve proofs for the core protocols where a bug means data loss: consensus, replication, transaction commit, and storage durability. The specifications written for model checking are reusable as the top layer of later proofs, so the levels form a path rather than competing choices.

== Pitfalls

- Specifications that model the network as more reliable than it is (no duplication, FIFO delivery) and miss the bugs that matter.
- Checking only configurations with every node live, so recovery paths are never explored.
- Treating a passing model check at three nodes as a proof.
- Specification and implementation that drift apart because nothing checks conformance; trace validation or refinement is needed to keep them linked.
- Liveness claims without stated fairness and timing assumptions.
- Trusting environment assumptions in proofs (clocks, disks that honour `fsync`, the network stack) that production violates.

== Further Reading

Fischer, M. J., Lynch, N. A. and Paterson, M. S. (1985). "Impossibility of Distributed Consensus with One Faulty Process." Journal of the ACM.

Newcombe, C. et al. (2015). "How Amazon Web Services Uses Formal Methods." Communications of the ACM.

Hawblitzel, C. et al. (2015). "IronFleet: Proving Practical Distributed Systems Correct." SOSP.

Wilcox, J. R. et al. (2015). "Verdi: A Framework for Implementing and Formally Verifying Distributed Systems." PLDI.

Woos, D. et al. (2016). "Planning for Change in a Formal Verification of the Raft Consensus Protocol." CPP.

Padon, O. et al. (2016). "Ivy: Safety Verification by Interactive Generalization." PLDI.

Padon, O. et al. (2017). "Paxos Made EPR: Decidable Reasoning about Distributed Protocols." OOPSLA.

Ma, H. et al. (2019). "I4: Incremental Inference of Inductive Invariants for Verification of Distributed Protocols." SOSP.

Yao, J. et al. (2021). "DistAI: Data-Driven Automated Invariant Learning for Distributed Protocols." OSDI.

Desai, A. et al. (2013). "P: Safe Asynchronous Event-Driven Programming." PLDI.

Zhou, J. et al. (2021). "FoundationDB: A Distributed Unbundled Transactional Key Value Store." SIGMOD.

Kingsbury, K. and Alvaro, P. (2020). "Elle: Inferring Isolation Anomalies from Experimental Observations." VLDB.

Musuvathi, M. et al. (2008). "Finding and Reproducing Heisenbugs in Concurrent Programs." OSDI.

Davis, A. J. J., Hirschhorn, M. and Schvimer, J. (2020). "eXtreme Modelling in Practice." VLDB.

Bornholt, J. et al. (2021). "Using Lightweight Formal Methods to Validate a Key-Value Storage Node in Amazon S3." SOSP.

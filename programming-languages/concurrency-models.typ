= Concurrency Models

When a program runs on more than one thread of control, the question "what does this program mean?" is no longer answered by a single trace of states. Concurrent execution multiplies the possible behaviours combinatorially; weak memory hardware multiplies them again; failures multiply them still further. The discipline of *concurrency semantics* is the study of which behaviours are admitted, which are excluded, and what guarantees the programmer may rely on.

*See also:* _Process Calculi_, _Program Semantics_

The previous chapter studied algebraic theories — CCS, CSP, the $pi$-calculus — that take *interaction* as primitive. This chapter takes a different vantage: rather than designing a calculus, we ask how the operational semantics of an existing programming language is to be extended when multiple threads execute concurrently, what mathematical structures underlie the resulting behaviours, and how the dominant programming models — actors, channels, transactions, async/await, dataflow — fit into the semantic picture. The treatment is deliberately language-pluralist: Erlang, Go, Rust, Haskell, OCaml 5, and Scala each embody a distinct point in the design space.

== Interleaving Semantics

The simplest model of concurrency takes a multi-threaded program to be a set of sequential threads, and a concurrent execution to be an arbitrary *interleaving* of their atomic steps. A configuration is a tuple $angle.l T_1, dots, T_n; sigma angle.r$ where $T_i$ are the per-thread control states and $sigma$ is a shared store. The transition rule:

```text
            T_i  ;  sigma   -->_seq   T_i'  ;  sigma'
        --------------------------------------------------       (STEP-i)
          <T_1, ..., T_i, ..., T_n; sigma>
            --> <T_1, ..., T_i', ..., T_n; sigma'>
```

The premise uses the single-thread small-step relation $arrow.r_"seq"$ inherited from the sequential semantics of the host language. The scheduler is *demonic*: it chooses any thread that can take a step, and the program must be correct under every choice.

This *interleaving model* is the implicit semantics of all introductory concurrent-programming texts. It is also the precise content of *sequential consistency*.

=== Sequential Consistency

*Definition (Lamport 1979).* A multiprocessor is *sequentially consistent* if the result of any execution is the same as if the operations of all processors were executed in some sequential order, and the operations of each processor appear in that sequence in the order specified by its program.

Equivalently: there exists a total order on memory operations such that (i) each processor's operations appear in program order, and (ii) every read returns the value written by the most recent write in the total order.

SC is the strongest reasonable concurrency model and the one most programmers intuit. It is also unachievable on modern hardware without prohibitive cost: store buffers, write-back caches, speculative execution, and instruction reordering all violate SC. The PL semantics community responded by building weaker models — and then attempting to recover SC for *well-behaved* programs (the data-race-freedom theorem below).

== True Concurrency: Event Structures

Interleaving identifies two concurrent actions $a$ and $b$ with the two-element set ${a b, b a}$ of interleavings. A *true-concurrency* semantics records that $a$ and $b$ were *concurrent* — neither preceded the other — without flattening to a choice between orders.

*Event structures* (Winskel 1986) are the canonical such model. A *prime event structure* is a triple $cal(E) = (E, lt.eq, sharp)$ where:

- $E$ is a set of *events*.
- $lt.eq subset.eq E times E$ is a partial order — the *causality* relation.
- $sharp subset.eq E times E$ is a symmetric irreflexive relation — the *conflict* relation.

with two axioms:

1. *Finite causes:* for every $e in E$, the down-set $arrow.b e = { e' : e' lt.eq e }$ is finite.
2. *Conflict heredity:* if $e_1 sharp e_2$ and $e_2 lt.eq e_3$, then $e_1 sharp e_3$.

A *configuration* of $cal(E)$ is a subset $X subset.eq E$ that is

- *causally closed:* $e in X and e' lt.eq e => e' in X$;
- *conflict-free:* $forall e_1, e_2 in X. "not" (e_1 sharp e_2)$.

The set $cal(C)(cal(E))$ of configurations, ordered by inclusion, is a *Scott domain*; computations are interpreted as paths through this domain. The crucial point: $X union { e_1 } in cal(C)(cal(E))$ and $X union { e_2 } in cal(C)(cal(E))$ with $X union { e_1, e_2 } in cal(C)(cal(E))$ encodes that $e_1$ and $e_2$ are *concurrent at $X$* — both enabled, neither dependent on the other, neither in conflict.

Variants in the literature: *flow event structures* (Boudol–Castellani 1988) relax the conflict-heredity axiom; *stable event structures* (Winskel 1986) replace conflict with a stability condition allowing multiple possible "histories" for each event; *asymmetric event structures* (Baldan–Corradini–Montanari 1999) replace symmetric conflict with a precedence relation suited to read–write conflicts in concurrent systems.

Event structures are the canonical semantic universe for processes whose concurrent independence must be preserved. They underlie the standard "true-concurrency" semantics of Petri nets, CCS, and the $pi$-calculus, and form the basis of unfolding-based model checking.

== Pomsets

*Partial-order multisets* (pomsets; Pratt 1986, Gischer 1988) take a complementary view: a single execution is a *labelled partial order* of events. A pomset is a tuple $p = (E, lt.eq, lambda)$ with $E$ an event set, $lt.eq$ a partial order, $lambda : E arrow.r Sigma$ a labelling. Pomsets are taken up to label-preserving order-isomorphism.

The pomset language of a program is the set of pomsets it can perform. Pomset *prefix*, *concatenation*, and *parallel composition* generalise their string analogues; pomset languages form a closed algebraic structure (Gischer's *concurrent semiring*). Pomsets are weaker than event structures (they do not record conflict) and incomparable with traces (they record concurrency but flatten internal nondeterminism).

Pomsets enjoy a revival in the analysis of *weak memory*: they form the substrate of the *promising semantics* (Kang et al. 2017) and several axiomatic memory models, because the partial order naturally expresses happens-before without committing to an interleaving.

== Petri Nets

*Petri nets* (Petri 1962) are perhaps the oldest formal model of concurrency. A net is a tuple $N = (P, T, F, M_0)$ with $P$ a finite set of *places*, $T$ a finite set of *transitions*, $F subset.eq (P times T) union (T times P)$ the flow relation, and $M_0 : P arrow.r NN$ the initial *marking* (tokens per place).

A transition $t$ is *enabled* at marking $M$ if every input place $p$ (with $(p, t) in F$) holds at least one token: $M(p) gt.eq 1$. Firing $t$ removes one token from each input place and adds one to each output place. The reachable markings form a transition system; the *reachability set* $cal(R)(N) = { M : M_0 arrow.r^* M }$ is the central object of analysis.

=== Decidability and Complexity

*Theorem (Mayr 1981, Kosaraju 1982, Lambert 1992).* Reachability for Petri nets is decidable.

The proof — a multi-decade effort — uses the *Karp–Miller tree* generalised to a coverability graph, but reachability itself requires a more sophisticated argument. The latest complexity bounds are striking.

*Theorem (Czerwiński–Orlikowski 2021, Leroux 2021).* Reachability for Petri nets is Ackermann-complete.

*Theorem (Lipton 1976).* Reachability requires at least $2^(Omega(sqrt(n)))$ space — i.e., is EXPSPACE-hard.

*Theorem (Rackoff 1978).* *Coverability* (given $M$, is some $M' gt.eq M$ reachable?) is EXPSPACE-complete.

The gap between coverability (EXPSPACE) and reachability (Ackermann) is one of the most dramatic in computational complexity. Coverability suffices for *safety* properties phrased as "a bad marking is never reached", which is why most Petri-net model checkers analyse coverability rather than full reachability.

=== Workflow Nets

A *workflow net* (van der Aalst 1997) is a Petri net with a designated source place, a designated sink place, and the property that every node lies on some path from source to sink. Soundness of a workflow net — every reachable marking can reach the terminal marking with exactly one token on the sink — is decidable in polynomial time, and is the standard correctness criterion for business-process models.

=== Coloured and High-Level Nets

*Coloured Petri nets* (Jensen 1981) attach data values ("colours") to tokens and arc inscriptions that determine which tokens flow where. They are not more expressive than ordinary nets (any coloured net unfolds to an ordinary net, possibly infinite) but are exponentially more concise. Coloured nets are the basis of the CPN Tools verification environment.

== Mazurkiewicz Traces

A *trace alphabet* (Mazurkiewicz 1977) is a pair $(Sigma, I)$ with $Sigma$ a finite alphabet and $I subset.eq Sigma times Sigma$ a symmetric irreflexive *independence* relation. The *trace equivalence* $tilde.equiv_I$ on $Sigma^*$ is the smallest congruence with $a b tilde.equiv_I b a$ for every $(a, b) in I$. A *Mazurkiewicz trace* is an equivalence class $[w]_I$.

Traces are isomorphic to labelled pomsets (the underlying partial order is generated by dependence). They are the precise mathematical content of "we don't care in which order independent actions occurred". This is the basis of *partial-order reduction* (Peled 1993, Godefroid 1996), a model-checking technique that explores only a representative interleaving from each trace equivalence class, often reducing state-space size by orders of magnitude.

A *Foata normal form* of a trace decomposes it into a sequence of mutually independent *steps*; this is the standard representation in implementations.

== The Actor Model

The *actor model* (Hewitt 1973; Agha 1986) discards shared state entirely. The world is a population of *actors*; each actor has a mailbox, an address, a current behaviour, and a set of acquaintances (the addresses it knows). An actor's only actions are:

1. *Send* a message to a known address (asynchronous; the sender does not wait).
2. *Create* a new actor with a specified initial behaviour and obtain its address.
3. *Become* a new behaviour for future messages.

=== Operational Semantics

A *configuration* is a multiset of actors and undelivered messages. Each actor is $angle.l a, b, q angle.r$ with address $a$, behaviour $b$, and mailbox $q$. A message is $a triangle.l v$ ("$v$ for $a$"). The transition rules:

```text
        a triangle.l v   on the wire
        actor at a is ready (mailbox empty? — or, append to mailbox)
        --------------------------------------------------       (DELIVER)
        ... | <a, b, q> | (a triangle.l v) -->
        ... | <a, b, q ++ [v]>

        actor at a is ready, head of q is v, b(v) computes
        actions: send m1...mk, create c1...cj, become b'
        --------------------------------------------------       (DISPATCH)
        ... | <a, b, v :: q> -->
        ... | <a, b', q> | (a_1 triangle.l m_1) | ... |
                          <c_1, b_{c_1}, []> | ...
```

The semantics is *asynchronous* (sends do not block), *fair* (every message is eventually delivered — usually formalised as a fairness constraint on the scheduler), and *location-transparent* (the address $a$ uniquely identifies an actor regardless of physical location).

Address creation gives the actor model dynamic topology equivalent to the $pi$-calculus's name passing. The faithful encoding of asynchronous $pi$ into actors and vice versa (Agha–Mason–Smith–Talcott 1997) makes the two models intertranslatable as theories of concurrency, though the engineering tradeoffs differ: actors emphasise *locality* (one mailbox per actor, scheduled by one thread) while asynchronous $pi$ is symmetric in inputs and outputs.

=== Type Systems for Actors

Agha–Mason–Smith–Talcott (1997) developed a process algebra-style semantics for actors and proved equational properties. Type systems for actors (Colaço–Pantel–Sallé 2000, He–Pradat-Peyre–Salaün 2008) typically combine a mailbox protocol (the types of messages an actor accepts) with a behaviour signature (the message types each behaviour handles), giving the actor analogue of session types.

=== Erlang

Erlang (Armstrong et al. 1986, OTP 1998) is the canonical actor-model language: lightweight processes (each with its own heap), asynchronous message passing, no shared memory, supervision trees for fault tolerance. A minimal example:

```erlang
-module(counter).
-export([start/0, loop/1]).

start() ->
    spawn(?MODULE, loop, [0]).

loop(N) ->
    receive
        {inc, From} ->
            From ! {ok, N + 1},
            loop(N + 1);
        {get, From} ->
            From ! {value, N},
            loop(N);
        stop ->
            ok
    end.
```

The `receive` construct is selective: an Erlang process inspects its mailbox for a message matching any of the patterns, and the first matching message is consumed (in mailbox order among matching messages). Selective receive substantially complicates the semantics (a star.op message may not be the head of the queue when consumed) but is essential for the protocol patterns common in OTP applications.

=== Akka, Pony, Orleans

*Akka* (JVM, Scala/Java) and *Pony* (statically typed actors with capability-based data races elimination, Clebsch et al. 2015) carry the model into systems languages. *Microsoft Orleans* (Bernstein et al. 2014) introduces *virtual actors* ("grains") whose lifecycle is managed by the runtime; an inactive grain may be garbage-collected and resurrected on next access. Orleans's design makes the actor model serviceable for cloud-scale applications.

== Channels: Go and CSP-Style Concurrency

Go's concurrency primitives are *goroutines* (lightweight threads scheduled by a runtime, typically tens of kilobytes of stack each) and *channels* (typed synchronisation/communication conduits). The design follows the CSP tradition: communicate by sharing memory, not the converse.

```go
func producer(ch chan<- int) {
    for i := 0; i < 10; i++ {
        ch <- i
    }
    close(ch)
}

func consumer(ch <-chan int, done chan<- struct{}) {
    for v := range ch {
        fmt.Println(v)
    }
    done <- struct{}{}
}

func main() {
    ch := make(chan int)
    done := make(chan struct{})
    go producer(ch)
    go consumer(ch, done)
    <-done
}
```

The `select` statement nondeterministically chooses among ready channel operations:

```go
select {
case v := <-ch1:
    handle1(v)
case ch2 <- x:
    sent()
case <-time.After(time.Second):
    timeout()
}
```

Go's channels are *bounded* (the buffer size is fixed at creation; unbuffered channels enforce rendezvous). The semantics is *synchronous* for unbuffered channels (matching CSP) and *asynchronous-with-bound* for buffered ones. Go's `chan` types $T arrow.l$, $arrow.l T$ encode IO capabilities in a manner reminiscent of Pierce–Sangiorgi IO subtyping, though without the variance subtyping.

=== Pitfalls

Go's concurrency is widely loved but the operational semantics has sharp edges. Sending on a closed channel panics. Receiving from a closed channel yields the zero value (silently). Goroutine leaks (a goroutine blocked forever on a channel that no one will send to) are a frequent source of memory leaks in long-running systems. The race detector (Vyukov 2013, based on Eraser-style happens-before tracking) is essential.

== Rust: Ownership for Concurrency

Rust's contribution to the concurrency landscape is the use of an affine type system — *ownership* — to statically eliminate data races. The two marker traits `Send` (a type whose values may safely transfer between threads) and `Sync` (a type whose `&T` references may safely be shared between threads) are auto-derived for all types whose fields are themselves `Send`/`Sync`. Primitive types are `Send + Sync`; `Rc<T>` (non-atomic reference counting) is neither; `Arc<T>` is `Send + Sync` when $T$ is.

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Result: {}", *counter.lock().unwrap());
}
```

The type checker enforces that the closure passed to `thread::spawn` captures only `Send` values, that `Mutex<T>` requires `T: Send`, and that the lock guard's lifetime statically prevents use of the protected data outside the critical section. The *soundness* theorem — well-typed Rust programs are data-race-free — is non-trivial because the type system is enforced over a language with mutable references and unsafe escape hatches. *RustBelt* (Jung–Jourdan–Krebbers–Dreyer 2018) gives a machine-checked semantic safety proof using the Iris separation logic.

== Software Transactional Memory

*Software Transactional Memory* (STM; Shavit–Touitou 1995, Herlihy–Moss 1993 for hardware) replaces locking with a transactional model: a block of code marked `atomic` executes as if instantaneous, with the runtime detecting conflicts and retrying.

Harris–Marlow–Peyton Jones–Herlihy (2005) gave STM a uniquely clean design in Haskell, using the type system to enforce that transactional code may only invoke transactional operations:

```haskell
import Control.Concurrent.STM

transfer :: TVar Int -> TVar Int -> Int -> STM ()
transfer from to amt = do
    f <- readTVar from"
    when (f < amt) retry
    writeTVar from (f - amt)
    t <- readTVar to"
    writeTVar to (t + amt)

main = do
    a <- atomically (newTVar 100)
    b <- atomically (newTVar 0)
    atomically (transfer a b 30)
```

The `STM` monad type prevents arbitrary IO from leaking into a transaction (an aborted IO action would be impossible to roll back). The `retry` combinator blocks the current transaction until at least one of the `TVar`s read so far has been modified — implementing *condition synchronisation* without explicit wait queues. `orElse` composes two alternatives: `t1 ` then `t2` runs $t_1$ and, if it retries, runs $t_2$ instead.

=== Opacity

A correctness condition for STM, finer than serialisability:

*Definition (Guerraoui–Kapalka 2008).* A history of an STM is *opaque* <==> there is an equivalent serial history $S$ such that (i) committed transactions appear in $S$ in real-time order, and (ii) every aborted transaction in the original history reads values consistent with some prefix of $S$.

The second clause is the strengthening over serialisability: aborted transactions must not observe inconsistent states. Without opacity, an aborted transaction may, e.g., divide by zero or enter an infinite loop because it observed an impossible combination of values. Most production STM implementations are opaque.

=== Composability

The strongest argument for STM is *composability*. Locks do not compose: combining two thread-safe APIs may produce code that is not thread-safe (the two locks may be acquired in inconsistent orders, deadlocking). Transactions compose trivially: $"atomically"(t_1 ; t_2)$ is well-defined whenever $t_1$ and $t_2$ are. The Haskell `STM` monad makes this composability a static guarantee.

The connection to database concurrency control is direct: STM is *optimistic concurrency control* (OCC) for in-memory data, and the read/write set tracking, validation, and abort/retry pattern matches the OCC theory in databases. See the chapter on database concurrency control in the systems volume.

== Monitors

The classical *monitor* (Hoare 1974; Lampson–Redell 1980) is a synchronisation construct combining mutual exclusion with condition variables: a procedure of a monitor holds the monitor's lock for the duration of its execution, and condition variables (`wait`, `signal`, `broadcast`) allow threads to suspend within the monitor and be resumed.

A subtlety: Hoare's original semantics has `signal` *immediately* transfer the lock to the signalled thread, requiring the signaller to be in a state where the monitor's invariant holds. The Lampson–Redell variant (used by Java, POSIX condition variables, and most production systems) has `signal` merely *enqueue* the signalled thread; the signaller continues holding the lock until it exits the procedure. The semantic consequence is that signalled threads must *recheck* the condition (the canonical `while (!cond) cond.wait();` idiom), because the state may have changed between the signal and the wakeup.

== Reactive Programming and Dataflow

A separate concurrency tradition treats time as the primitive and computation as the propagation of values through a network of operators. *Synchronous dataflow* languages — Lustre (Halbwachs–Caspi–Raymond–Pilaud 1991), Esterel (Berry–Gonthier 1992), Signal — model a system as a network of operators that fire in lock-step with a global clock. Each tick, every operator consumes one value on each input and produces one on each output. The semantics is deterministic and verifiable; Lustre/SCADE is certified for use in avionics (Airbus fly-by-wire).

*Functional Reactive Programming* (Elliott–Hudak 1997) treats time-varying values (*signals*) and discrete events as first-cal(C). A signal $s : "Time" arrow.r alpha$ models a continuously varying quantity; an event $e : "Time" arrow.r "Maybe" alpha$ models a sporadic occurrence. Combinators such as `lift`, `integral`, and `accumulate` build complex behaviours from primitive signals. The pure-functional version of FRP suffers from the *space leak* problem (retaining histories that are never used); push–pull FRP (Elliott 2009) and the more recent *arrowised FRP* (Yampa) ameliorate this.

*Reactive Extensions* (Rx; Meijer 2010 and successors) carry the dataflow idea into mainstream languages as libraries: an `Observable<T>` is a (push-based) sequence of values with subscription, transformation, and composition operators. Backpressure protocols (Reactive Streams JEP) negotiate flow rates between producer and consumer.

== Async/Await

The async/await construct, popularised by C\# (2012) and now present in Rust, JavaScript, Python, Swift, Kotlin, and others, is *syntactic sugar* for an underlying continuation-passing transformation. A function

```rust
async fn fetch_and_parse(url: &str) -> Result<Data, Error> {
    let body = http_get(url).await?;
    parse(&body).await
}
```

desugars to a state machine implementing the `Future` trait. Each `.await` is a suspension point at which the function returns control to its caller along with a *continuation* — the remainder of the function — to be resumed when the awaited future completes.

=== Semantic View

In algebraic-effects terms (cross-reference the chapter on effects and handlers), `await` is an *operation* in an `Async` effect, and the async runtime is the *handler* that interprets these operations. The continuation captured at each `await` is the *delimited continuation* from that point to the end of the enclosing `async` block. This view explains why async functions form a *monad* (the *Future monad*, or in Rust the `IntoFuture` trait) and why they compose by sequencing (`?` and `await`) rather than by raw threads.

The desugaring may be made entirely explicit using a *continuation-passing style* (CPS) transformation:

$ "async" { e_1; "await" e_2; e_3 } &= "do" \
&quad x arrow.l e_1; \
&quad y arrow.l e_2; \
&quad e_3 $

with monadic bind interpreting `do`-notation. In Rust the monad is encoded by hand-written `Future` impls and a state-machine compiler pass; in Haskell the equivalent is `async` from `Control.Concurrent.Async`.

=== Cooperative vs Preemptive

Async runtimes are *cooperative*: a task runs until it explicitly suspends at an `await`. A computation that never yields ("CPU-bound") starves all other tasks on the same scheduler. Production async runtimes (Tokio, async-std) recommend explicit yielding (`tokio::task::yield_now`) or offloading to a thread pool. Preemptive runtimes (Erlang/BEAM, Go) avoid this hazard by inserting preemption checks at function calls or back-edges; the cost is a more complex runtime and the inability to expose synchronous-style APIs as native.

== Structured Concurrency

*Structured concurrency* (Sustrik 2016, Smith 2018) is the principle that the *lifetime* of every concurrent task must be bound to a syntactic scope: a function does not return until all the tasks it spawned have finished or been cancelled.

```python
# Python with Trio
async def main():
    async with trio.open_nursery() as nursery:
        nursery.start_soon(task_a)
        nursery.start_soon(task_b)
    # Both tasks are guaranteed complete here.
```

The nursery (Kotlin: `coroutineScope`; Java: `StructuredTaskScope`; Swift: `withTaskGroup`) is the syntactic boundary. Three properties follow:

1. *Cancellation propagation:* if any task in the scope fails, all sibling tasks are cancelled.
2. *Error propagation:* exceptions from child tasks aggregate at the scope's exit.
3. *Resource accounting:* tasks cannot outlive the resources they captured by closure.

The formalisation uses *region-based* or *scope monads*: a typing judgement $Gamma; rho tack.r e : tau$ with $rho$ a region of currently-open scopes; spawned tasks are tagged with a scope identifier, and the scope's exit waits on (and demands acknowledgement of cancellation from) every task tagged with it. Java's JEP 453 (preview in Java 21, refined in subsequent releases) realises this in `java.util.concurrent`.

== Memory Models

The deepest semantic problem in modern concurrent programming is the *memory model*: which values may a read return, given the writes performed by all threads? Sequential consistency is the strongest answer; real hardware and compilers provide weaker guarantees. A *memory model* is the contract between the programmer and the implementation specifying these guarantees.

=== Hardware Memory Models

- *Sequential consistency:* the strongest. No commodity hardware provides it.
- *Total store order (TSO; x86):* writes by each thread appear to other threads in program order, but a thread may observe its own writes before they are globally visible (store buffering). Reordering of read-after-write is allowed.
- *Release/acquire (ARM, RISC-V, POWER):* a release-write synchronises with an acquire-read on the same location; without these annotations, all reorderings are possible.
- *Relaxed:* no ordering guarantees beyond per-location coherence (writes to the *same* location appear in a single per-location total order).

The cost of a fence is hardware-dependent: an x86 `mfence` typically costs tens of cycles; an ARM `dmb ish` is similar. Compiler reorderings, separately, may move loads and stores across statement boundaries unless prevented by `volatile` (in C) or atomic operations with appropriate memory orderings.

=== The Data-Race-Freedom Theorem

*Definition.* Two memory operations *conflict* if they access the same location and at least one is a write. A *data race* is a pair of conflicting operations from different threads that are not ordered by happens-before.

*Theorem (DRF–SC, Adve–Hill 1990; refined Boehm–Adve 2008).* In a programming language whose memory model guarantees SC for data-race-free programs, any execution of a DRF program is equivalent to some sequentially consistent execution.

This theorem is the foundational compromise of modern memory models: *programmers who avoid data races see SC; programmers who race see only the weak model's guarantees*. The contract makes a strong language guarantee compatible with weak hardware: the compiler may freely reorder *as long as* the reorderings are invisible to DRF programs.

=== The C11/C++11 Memory Model

The C11 / C++11 memory model (Boehm–Adve 2008) refined into ISO standards is the most influential PL memory model. Six memory orders parameterise atomic operations:

- `memory_order_relaxed`: no ordering; only atomicity.
- `memory_order_consume`: data-dependency ordering (deprecated in practice; treated as `acquire`).
- `memory_order_acquire`: read; subsequent reads and writes do not move *before* this read.
- `memory_order_release`: write; preceding reads and writes do not move *after* this write.
- `memory_order_acq_rel`: both, for read-modify-write.
- `memory_order_seq_cst`: globally totally ordered with all other `seq_cst` operations.

The standard establishes a *happens-before* relation built from program order, release/acquire pairs (synchronizes-with), and `seq_cst` total order. A DRF program is one in which conflicting operations are ordered by happens-before. For DRF programs using only `seq_cst` synchronisation, executions are sequentially consistent.

=== Subtleties and Bugs

*Theorem (Batty–Memarian–Owens–Sarkar–Sewell 2011).* The C11/C++11 memory model as standardised contains several inconsistencies — most notably around "out-of-thin-air" reads — that no compiler or hardware implements correctly and that no straightforward patch resolves.

The OOTA problem: the relaxed memory model permits *self-justifying* executions in which a read returns a value only because a later write produces it. This is theoretically permitted but operationally absurd and outlaws important compiler optimisations. The Java memory model (Manson–Pugh–Adve 2005) attempted a global fix via the *causality* requirement; it was subsequently shown to have its own subtle flaws.

=== Promising Semantics

*Promising semantics* (Kang–Hur–Lahav–Vafeiadis–Dreyer 2017) is the first weak memory model that simultaneously:

- Justifies all reasonable compiler optimisations.
- Allows all hardware reorderings of the major architectures.
- Forbids out-of-thin-air reads.
- Supports DRF–SC.

The key idea: a thread may *promise* to perform a write at some future point; other threads may observe the promised value immediately, but the promising thread is then obliged to fulfill the promise on every possible execution. The set of promises encodes the future commitments of each thread; the model is consistent iff every promise is fulfillable.

Promising semantics is the current state of the art and has been formalised in Coq with mechanical proofs of all the DRF and compilation theorems.

=== OCaml 5 Memory Model

OCaml 5 (2022) introduced shared-memory parallelism with a deliberately small memory model (Dolan–Sivaramakrishnan–Madhavapeddy 2018): every read of a non-atomic location returns *some* previously-written value (or the initial value), and atomics provide release/acquire and seq_cst orderings. The model avoids OOTA by construction and is one of the simplest production-language memory models in existence. The price is that programmers cannot rely on certain optimisations available under the C11 model.

== Echoes of Distributed Systems

The concurrency theory of this chapter is local — multiple threads on one machine, sharing memory. Move to multiple machines and partial failure becomes the dominant constraint. The fundamental impossibility results — *FLP* (Fischer–Lynch–Paterson 1985: no asynchronous deterministic consensus tolerates even one crash) and *CAP* (Gilbert–Lynch 2002: no system simultaneously provides consistency, availability, and partition-tolerance) — are treated in the distributed-systems volume. The PL discipline contributes to that conversation through *causal models* (vector clocks, conflict-free replicated data types), *consensus-aware type systems*, and *deterministic parallel languages* (LVish, FlowPools) that give back determinism by restricting the operations.

== Outlook

The concurrent semantic landscape is unified by a small set of questions repeatedly asked at different scales: which interleavings are observable? Which behaviours does the implementation guarantee? How do we compose programs without losing the guarantees? Event structures, Petri nets, traces, actors, channels, transactions, and memory models each answer a slice of these questions. The discipline's most productive direction now lies in *combining* these models — session types over actors, transactions over channels, structured concurrency over async — into composite frameworks that bring local guarantees into global view. The next chapters on session types, effects and handlers, and distributed systems pursue precisely this synthesis.

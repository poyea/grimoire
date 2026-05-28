= Weakly Consistent Systems

Many distributed databases sacrifice strong consistency for availability and performance. Understanding the formal models — CRDT, causal consistency, eventual consistency, CALM — allows engineers to reason about what guarantees an application actually needs.

*See also:* _database/isolation-and-consistency-models.typ_, _database/consensus-and-replication.typ_, _database/transactions-distributed.typ_

== CAP Theorem

Brewer (2000), formalized by Gilbert & Lynch (2002): a distributed system can provide at most two of three guarantees:

- *Consistency (C):* every read sees the most recent write (linearizability).
- *Availability (A):* every request receives a response (non-error).
- *Partition tolerance (P):* the system continues operating despite network partitions.

*Since network partitions are unavoidable in distributed systems, the real choice is C vs A during a partition.*

```
Network partition occurs:
  ┌──────────────┐    ≠≠≠≠≠    ┌──────────────┐
  │  Node A      │  partition  │  Node B      │
  │  (primary)   │             │  (replica)   │
  └──────────────┘             └──────────────┘

CP system (prefer consistency):
  Node B rejects writes/reads until partition heals.
  "Better to be unavailable than to return stale data."
  Examples: HBase, Zookeeper, etcd, CockroachDB (strict)

AP system (prefer availability):
  Both nodes continue to serve requests.
  May return divergent/stale data.
  Examples: Cassandra, Couchbase, DynamoDB (by default)
```

*PACELC (Abadi 2012):* extends CAP. Even without a partition, there is a tradeoff between *Latency* and *Consistency*. A system must choose EL (low latency, eventual consistency) or EC (more consistent but higher latency).

== Eventual Consistency

All replicas eventually converge to the same state if updates stop. No guarantee about *when* or what intermediate states are visible.

*Monotone read:* once you read value V, you will never read an older value. Cassandra's `QUORUM` reads (R + W > N) provide this.

*Read repair:* on a read, if replicas return different versions, the coordinator repairs the stale replica with the latest version — passive convergence.

```python
# Cassandra-style quorum read with read repair
def quorum_read(key: str, replicas: list, quorum: int) -> bytes:
    responses = []
    for r in replicas[:quorum]:
        responses.append((r, r.read(key)))   # (replica, (version, value))

    # Find most recent version
    latest = max(responses, key=lambda x: x[1][0])   # highest timestamp

    # Read repair: update stale replicas
    for replica, (version, value) in responses:
        if version < latest[1][0]:
            replica.write(key, latest[1][1], latest[1][0])   # async

    return latest[1][1]   # return latest value
```

== Conflict-Free Replicated Data Types (CRDTs)

CRDTs (Shapiro et al. 2011) are data structures that can be updated concurrently without coordination, and will always converge to the same state when replicas are merged.

*Two flavors:*
- *CvRDT (state-based):* replicas periodically exchange full state; merge function is commutative, associative, idempotent.
- *CmRDT (operation-based):* replicas exchange operations; operations must be commutative.

=== G-Counter (Grow-only Counter)

```python
class GCounter:
    """Each node has its own slot; increment only your slot."""
    def __init__(self, node_id: str, nodes: list):
        self.node_id = node_id
        self.counts  = {n: 0 for n in nodes}

    def increment(self):
        self.counts[self.node_id] += 1

    def value(self) -> int:
        return sum(self.counts.values())

    def merge(self, other: "GCounter"):
        for node, val in other.counts.items():
            self.counts[node] = max(self.counts.get(node, 0), val)
        # Merge is: take element-wise max → commutative, associative, idempotent

# Concurrent increments on two nodes:
a = GCounter("A", ["A", "B"]); a.increment(); a.increment()
b = GCounter("B", ["A", "B"]); b.increment()
a.merge(b); b.merge(a)
assert a.value() == b.value() == 3   # ✓ convergence
```

=== PN-Counter (Increment and Decrement)

```python
class PNCounter:
    """Positive increments in P; negative decrements in N."""
    def __init__(self, node_id, nodes):
        self.P = GCounter(node_id, nodes)
        self.N = GCounter(node_id, nodes)

    def increment(self): self.P.increment()
    def decrement(self): self.N.increment()
    def value(self):     return self.P.value() - self.N.value()

    def merge(self, other):
        self.P.merge(other.P)
        self.N.merge(other.N)
```

=== LWW-Register (Last-Write-Wins)

```python
import time

class LWWRegister:
    def __init__(self):
        self.value     = None
        self.timestamp = 0

    def write(self, value, ts=None):
        ts = ts or time.time()
        if ts > self.timestamp:
            self.value     = value
            self.timestamp = ts

    def merge(self, other: "LWWRegister"):
        if other.timestamp > self.timestamp:
            self.value     = other.value
            self.timestamp = other.timestamp

# Risk: clock skew can cause a later-arriving write to "win"
# Mitigate: use hybrid logical clocks (HLC) instead of wall time
```

=== OR-Set (Observed-Remove Set)

Supports add and remove operations with unambiguous semantics: a remove only removes elements *observed* at the time of the remove.

```python
import uuid

class ORSet:
    def __init__(self):
        self.elements = {}   # value → set of unique tags (add tokens)
        self.removed  = {}   # value → set of removed tags

    def add(self, value):
        tag = str(uuid.uuid4())
        self.elements.setdefault(value, set()).add(tag)

    def remove(self, value):
        # Remove all currently observed tags for this value
        if value in self.elements:
            self.removed.setdefault(value, set()).update(self.elements[value])
            self.elements[value] -= self.removed[value]

    def contains(self, value) -> bool:
        return bool(self.elements.get(value, set()) - self.removed.get(value, set()))

    def merge(self, other: "ORSet"):
        for v, tags in other.elements.items():
            self.elements.setdefault(v, set()).update(tags)
        for v, tags in other.removed.items():
            self.removed.setdefault(v, set()).update(tags)
```

_Note: the canonical OR-Set keeps `elements` as a monotone-growing add-tag set and only filters by `removed` at read time. The in-place subtraction in `remove()` above is a local optimization; with it, `contains()` should rely on the tag-difference computation rather than on `elements` alone, and `merge()` must be the only path that grows either set to remain join-commutative._


== Causal Consistency

*Causal consistency* ensures that if event A causally precedes event B, all nodes see A before B. Concurrent events (neither precedes the other) may be seen in any order.

*Vector clocks:* track causal dependencies. Each node maintains a vector of timestamps, one per node.

```python
class VectorClock:
    def __init__(self, node_id: str, nodes: list):
        self.node_id = node_id
        self.clock   = {n: 0 for n in nodes}

    def increment(self) -> dict:
        self.clock[self.node_id] += 1
        return dict(self.clock)

    def update(self, other_clock: dict):
        for node, ts in other_clock.items():
            self.clock[node] = max(self.clock.get(node, 0), ts)
        self.clock[self.node_id] += 1

    def happens_before(self, a: dict, b: dict) -> bool:
        """Does event with clock a happen-before event with clock b?"""
        return all(a.get(n, 0) <= b.get(n, 0) for n in set(a) | set(b)) \
               and any(a.get(n, 0) < b.get(n, 0) for n in set(a) | set(b))

    def concurrent(self, a: dict, b: dict) -> bool:
        return not self.happens_before(a, b) and not self.happens_before(b, a)
```

*Causal+ consistency* (Bolt-on Causal Consistency, Bailis et al. 2013): adds a shim layer on top of an eventually-consistent store that buffers reads until all causal dependencies have been received.

== CALM Theorem

Hellerstein & Alvaro (2020): a program can be implemented in a monotone, coordination-free manner if and only if it is *monotone* (it never needs to retract or revise a previous output).

*Monotone examples:* adding elements to a set, incrementing a counter, computing a union. These operations only add information, never remove it — CRDTs are monotone.

*Non-monotone examples:* computing the minimum of a stream (must wait for all elements), checking "no element satisfies P" (universal negation). These require coordination or can produce wrong answers.

```python
# Monotone (CALM-safe): computing reachable nodes in a graph
# Each discovered edge only adds to the reachable set, never removes
def reachable(edges: set, start: str) -> set:
    found = {start}
    changed = True
    while changed:
        changed = False
        for (u, v) in edges:
            if u in found and v not in found:
                found.add(v)
                changed = True
    return found   # Can be computed with partial edge sets; results only grow

# Non-monotone (requires coordination): "are all orders shipped?"
# New order could arrive after check → must coordinate to get complete view
```

== Hybrid Logical Clocks (HLC)

HLC (Kulkarni et al. 2014) combines physical time with a logical counter, providing both causality tracking and a close approximation to wall clock time.

Used by: CockroachDB, YugabyteDB for causal ordering without GPS.

```python
# See transactions-distributed.typ for HLC implementation
# Key property: HLC.now() is always ≥ wall_clock.now()
#               HLC.recv(msg_ts) > msg_ts → preserves causality
```

== References

Brewer, E. (2000). "Towards Robust Distributed Systems." PODC (keynote). (CAP conjecture)

Gilbert, S., Lynch, N. (2002). "Brewer's Conjecture and the Feasibility of Consistent, Available, Partition-Tolerant Web Services." SIGACT News.

Abadi, D. (2012). "Consistency Tradeoffs in Modern Distributed Database System Design." IEEE Computer.

Shapiro, M., Preguiça, N., Baquero, C., Zawirski, M. (2011). "Conflict-Free Replicated Data Types." SSS.

Bailis, P., Ghodsi, A., Hellerstein, J., Stoica, I. (2013). "Bolt-on Causal Consistency." SIGMOD.

Hellerstein, J., Alvaro, P. (2020). "Keeping CALM: When Distributed Consistency Is Easy." CACM 63(9).

Kulkarni, S. et al. (2014). "Logical Physical Clocks and Consistent Snapshots in Globally Distributed Databases." OPODIS.

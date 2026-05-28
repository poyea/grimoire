= Consensus and Replication

Replication makes data available despite node failures. Consensus protocols coordinate replicas so they agree on a single log of operations. The FLP impossibility result bounds what is achievable; Paxos and Raft define the practical design space.

*See also:* _database/transactions-distributed.typ_, _database/weakly-consistent-systems.typ_, _database/partitioning-and-elasticity.typ_, _Networking volume_ (reliable transport, leader-election timeouts, message reordering)

== FLP Impossibility

Fischer, Lynch, Paterson (1985): in an asynchronous network where even one process may fail by crashing, there is no deterministic protocol that can always reach consensus.

*Intuition:* you cannot distinguish a crashed node from a very slow one. If a slow node would change the outcome, you may block forever waiting for it; if you decide without it, you may decide wrong.

*Practical escape:* real systems use *timeouts* (making the system partially synchronous) or *randomization* (Raft's randomized election timeouts, Ben-Or's protocol). FLP holds in the purely asynchronous model.

== Paxos

Paxos (Lamport 1989/1998) achieves consensus in a partially synchronous model. *Single-decree Paxos* decides on one value:

*Three roles:* Proposers (initiate rounds), Acceptors (vote), Learners (observe decided value).
*Quorum:* any majority of acceptors — two majorities always intersect.

```
Phase 1a: Proposer sends PREPARE(n) to all acceptors  (n = ballot number)
Phase 1b: Acceptor: if n > highest_promised,
            update promised_n = n
            reply PROMISE(n, accepted_n, accepted_val)

Phase 2a: Proposer: upon receiving f+1 PROMISEs,
            value = accepted_val of highest accepted_n seen in promises
                    (or proposer's own value if none)
            send ACCEPT(n, value) to all acceptors

Phase 2b: Acceptor: if n >= promised_n,
            accepted_n = n; accepted_val = value
            reply ACCEPTED(n, value)
            (also notify learners)

Learner: upon f+1 ACCEPTED(n, v) for same (n, v) → value v is chosen
```

```python
# Single-decree Paxos (simplified, in-process simulation)
import random

class Acceptor:
    def __init__(self):
        self.promised_n  = -1
        self.accepted_n  = -1
        self.accepted_v  = None

    def prepare(self, n):
        if n > self.promised_n:
            self.promised_n = n
            return ("PROMISE", self.accepted_n, self.accepted_v)
        return ("NACK", self.promised_n)

    def accept(self, n, v):
        if n >= self.promised_n:
            self.promised_n = n
            self.accepted_n  = n
            self.accepted_v  = v
            return ("ACCEPTED", n, v)
        return ("NACK", self.promised_n)

def run_paxos(acceptors, proposed_value, ballot):
    f     = (len(acceptors) - 1) // 2   # max failures tolerated
    quorum = f + 1

    # Phase 1
    promises = [a.prepare(ballot) for a in acceptors]
    promises  = [p for p in promises if p[0] == "PROMISE"]
    if len(promises) < quorum:
        return None  # no quorum

    # Use value with highest accepted ballot, or our own
    best = max(promises, key=lambda p: p[1])
    value = best[2] if best[2] is not None else proposed_value

    # Phase 2
    accepted = [a.accept(ballot, value) for a in acceptors]
    accepted  = [a for a in accepted if a[0] == "ACCEPTED"]
    if len(accepted) >= quorum:
        return value
    return None

acceptors = [Acceptor() for _ in range(5)]
result    = run_paxos(acceptors, proposed_value="commit_txn_42", ballot=1)
print(result)   # "commit_txn_42"
```

*Multi-Paxos:* elect a stable *leader* (via Paxos on a leader-epoch variable). The leader sends ACCEPT directly without repeated PREPARE rounds. This is what production systems (Chubby, Zookeeper) use.

== Raft

Raft (Ongaro & Ousterhout 2014) was designed for understandability. It decomposes consensus into three sub-problems: leader election, log replication, and safety.

*Terms:* monotonically increasing integers. Each term has at most one leader.

*Leader election:*

```
Initially: all nodes are Followers.
After election timeout (150–300ms randomized):
  Follower becomes Candidate → increments term → sends RequestVote(term, lastLogIndex, lastLogTerm)
  Candidate wins if it receives votes from majority and its log is at least as up-to-date as voters'
  Winner becomes Leader → sends periodic heartbeats (AppendEntries with empty entries)
```

*Log replication:*

```python
# Raft leader: replicate a new log entry
class RaftLeader:
    def append_entry(self, command, followers):
        # Leader appends to its own log
        entry  = LogEntry(term=self.current_term,
                          index=len(self.log),
                          command=command)
        self.log.append(entry)

        # Send AppendEntries RPCs to all followers
        acks = 1   # count self
        for f in followers:
            success = f.append_entries(
                leader_term   = self.current_term,
                prev_log_idx  = entry.index - 1,
                prev_log_term = self.log[entry.index-1].term if entry.index > 0 else 0,
                entries       = [entry],
                leader_commit = self.commit_index,
            )
            if success:
                acks += 1

        # Majority of the full cluster (leader + followers); leader's ack already counted.
        if acks * 2 > len(followers) + 1:
            self.commit_index = entry.index    # majority replicated → committed
            return True
        return False  # retry with backoff
```

*Raft safety property:* once an entry is committed (majority replicated), it will always be present in future leaders' logs. Guaranteed by the *election restriction*: a candidate can only win if its log is at least as up-to-date as a majority of voters' logs (compareterm, then index).

*Raft vs Paxos:*

#table(
  columns: (auto, auto, auto),
  [*Dimension*], [*Paxos*], [*Raft*],
  [Understandability], [Hard (many variants)], [Designed for clarity],
  [Leader election], [Implicit via ballot numbers], [Explicit term-based election],
  [Log ordering], [Gaps possible], [No gaps (must replicate in order)],
  [Membership change], [Not specified], [Joint consensus / single-server changes],
  [Used in], [Chubby, Zookeeper, Spanner], [etcd, CockroachDB, TiKV, Consul],
)

== EPaxos (Egalitarian Paxos)

EPaxos (Moraru et al. 2013): no stable leader — any replica can commit any command. Commands are committed with *dependencies* (what commands must precede them). Non-conflicting commands can commit in parallel.

*Commit latency:* 1 RTT for non-conflicting commands (fast path); 2 RTT for conflicting commands. Paxos with leader requires 1 RTT to leader + 1 RTT broadcast = 2 RTT for WAN deployments.

*Used in:* research and some geo-distributed systems (e.g., experimental modes in MongoDB).

== Flexible Paxos

Flexible Paxos (Howard et al. 2017): the quorum requirement can differ between Phase 1 and Phase 2, as long as any Phase 1 quorum and any Phase 2 quorum intersect.

*Classic Paxos:* Phase 1 quorum = majority, Phase 2 quorum = majority.

*Flexible:* Phase 1 quorum = $n$ (can be large, rare), Phase 2 quorum = $n - F$ (can be small, fast).

```
Example: 5 nodes, F=1 (tolerate 1 failure)
  Classic:  Phase 1 quorum = 3, Phase 2 quorum = 3
  Flexible: Phase 1 quorum = 5 (all), Phase 2 quorum = 1 (!!!)
    → Phase 2 (normal operation) needs only 1 replica — ultra-low latency
    → Phase 1 (leader election) needs all 5 — rare, acceptable slowdown

Correctness: any Phase 1 quorum (5) intersects any Phase 2 quorum (1) → ✓
```

== Replication Modes

#table(
  columns: (auto, auto, auto, auto),
  [*Mode*], [*Latency*], [*Durability*], [*Read options*],
  [Synchronous],        [+RTT to replica], [Strong (data on ≥2 nodes)], [Any replica],
  [Semi-sync (1-of-N)], [+RTT to fastest], [Data on ≥2 nodes],         [Leader or lagging replica],
  [Asynchronous],       [Local only],      [Weak (may lose data)],      [Stale reads on replicas],
)

```sql
-- PostgreSQL synchronous replication
-- In postgresql.conf:
-- synchronous_standby_names = 'FIRST 1 (standby1, standby2)'
-- synchronous_commit = on

-- Per-transaction control:
SET synchronous_commit = local;     -- commit when local WAL flushed (async to replicas)
SET synchronous_commit = remote_write;  -- commit when replica received (not fsynced)
SET synchronous_commit = on;        -- commit when replica fsynced

-- Replication lag monitoring
SELECT client_addr,
       state,
       sent_lsn - write_lsn   AS write_lag_bytes,
       sent_lsn - flush_lsn   AS flush_lag_bytes,
       sent_lsn - replay_lsn  AS replay_lag_bytes
FROM   pg_stat_replication;
```

== Read-Your-Writes and Linearizable Reads

*Problem:* after a write to the leader, reading from a lagging replica may return stale data — violating Read-Your-Writes.

*Solutions:*

```python
# Option 1: Always read from leader (simple, high load on leader)
def read_own_writes(key, my_write_lsn):
    return leader.read(key)

# Option 2: Read from replica with a minimum LSN guarantee
def read_own_writes(key, my_write_lsn):
    for replica in replicas:
        if replica.applied_lsn >= my_write_lsn:
            return replica.read(key)
    return leader.read(key)  # fallback

# Option 3: Quorum reads (reads from majority of replicas, take latest)
def quorum_read(key):
    responses = [r.read_with_ts(key) for r in replicas[:quorum_size]]
    return max(responses, key=lambda r: r.timestamp).value
```

== References

Fischer, M., Lynch, N., Paterson, M. (1985). "Impossibility of Distributed Consensus with One Faulty Process." JACM 32(2).

Lamport, L. (1998). "The Part-Time Parliament." TOCS 16(2). (Paxos)

Ongaro, D., Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm." USENIX ATC. (Raft)

Moraru, I., Andersen, D., Kaminsky, M. (2013). "There Is More Consensus in Egalitarian Parliaments." SOSP. (EPaxos)

Howard, H., Malkhi, D., Spiegelman, A. (2017). "Flexible Paxos: Quorum Intersection Revisited." OPODIS.

van Renesse, R., Altinbuken, D. (2015). "Paxos Made Moderately Complex." ACM CSUR 47(3).

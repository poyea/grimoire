= Coordination Services

Distributed systems need a small, highly-available store for configuration, group membership, and locks — something that tolerates node failures yet delivers the strong consistency required for leader election and distributed locking. This chapter examines the three canonical coordination services: ZooKeeper, etcd, and Google's Chubby.

*See also:* _Consensus Deep Dive_, _Leader Election and Leases_, _Transactions_, `distributed-systems/failure-detection.typ`

== ZooKeeper

ZooKeeper (Hunt et al. 2010) is a coordination kernel used by Hadoop, Kafka, HBase, and many other systems. Its data model is a *znodes* tree (similar to a filesystem), and its consistency model is *linearisable writes, serially ordered reads per session*.

=== ZAB Protocol

ZooKeeper Atomic Broadcast (*ZAB*) is the consensus protocol underlying ZooKeeper. It is similar to Paxos but designed around a stable *leader epoch* model:

```
Leader Election:
    nodes vote for the node with the highest (epoch, zxid)
    a candidate with a quorum of votes becomes leader

Synchronisation (recovery):
    leader sends DIFF/TRUNC/SNAP to bring followers to leader's state
    followers ack; once quorum acks, leader can serve

Broadcast (steady state):
    client write -> leader
    leader assigns zxid = (epoch, counter)
    leader -> all followers: PROPOSAL(zxid, txn)
    followers write to log, reply ACK
    once quorum ACK: leader sends COMMIT(zxid)
    followers apply in zxid order
```

*ZXID* is a 64-bit ID: high 32 bits = epoch, low 32 bits = counter. Epoch advances on each leader election, preventing a partitioned old leader from committing stale proposals.

=== Znodes and the Data Model

The namespace is a tree of *znodes*, each storing up to 1 MB of byte data and metadata (ctime, mtime, version, dataLength, numChildren).

- *Persistent znodes:* survive session expiry.
- *Ephemeral znodes:* deleted when the creating session expires. Used for group membership registers and lock recipes.
- *Sequential znodes:* ZooKeeper appends a monotonically increasing suffix to the name on creation. Combined with ephemeral, they power the *lock recipe* below.
- *Container znodes* (v3.5+): deleted automatically when their last child is deleted.

=== Watches

A client registers a *watch* on a znode at read time: `getData(path, watch=True)`. ZooKeeper delivers a one-shot *WatchEvent* notification when the znode changes. Watches are intentionally one-shot and do not carry the new data, forcing the client to re-read and re-register — avoiding thundering herd while keeping the server stateless about watch payloads.

```python
def watch_handler(event):
    if event.type == EventType.CHANGED:
        data, _ = zk.get("/config/feature_flags", watch=watch_handler)
        update_local_config(data)

zk.get("/config/feature_flags", watch=watch_handler)
```

=== Session Fencing and Epoch Tokens

ZooKeeper sessions carry a *session ID* and an *epoch*. When a client reconnects after a partition, ZooKeeper validates the session. Clients must implement *fencing tokens*: include the current ZooKeeper epoch/zxid in requests to downstream services, which reject any request with a stale token. Without fencing, a client that was paused (GC, OS scheduling) might hold a lock it believes valid but ZooKeeper has already expired.

=== Distributed Lock Recipe

```python
def acquire_lock(zk, path):
    node = zk.create(path + "/lock-", ephemeral=True, sequence=True)
    while True:
        children = sorted(zk.get_children(path))
        if children[0] == node.split("/")[-1]:
            return node  # lock held
        predecessor = children[children.index(node.split("/")[-1]) - 1]
        event = threading.Event()
        if not zk.exists(path + "/" + predecessor,
                         watch=lambda _: event.set()):
            continue  # predecessor gone, retry
        event.wait()
```

The sequential ephemeral node recipe guarantees no herd effect: each waiter watches only its immediate predecessor.

== etcd

etcd (CoreOS 2013, now CNCF) is the backing store for Kubernetes and many cloud-native systems. It implements *Raft* (see _Consensus Deep Dive_) and exposes a versioned key-value API with gRPC.

=== MVCC Store

etcd stores all revisions of every key. The *revision* is a cluster-wide monotone integer that increments on every write. A key's history is indexed by (key, mod_revision). Reads default to the latest revision; historical reads use `WithRev(r)`.

```
etcdctl put /config/db_host 10.0.0.1     # revision 5
etcdctl put /config/db_host 10.0.0.2     # revision 6
etcdctl get /config/db_host --rev=5      # returns 10.0.0.1
```

*Compaction* (`etcdctl compact <rev>`) removes all historical revisions before `<rev>`, reclaiming disk space. Kubernetes runs compaction every 5 minutes by default.

=== Leases

A *lease* is a TTL-bound object. Keys attached to a lease (`WithLease(leaseID)`) are deleted atomically when the lease expires or is revoked. Clients renew leases via `LeaseKeepAlive` streams. Leases replace ZooKeeper's ephemeral nodes.

```go
lease, _ := client.Grant(ctx, 10)  // 10-second TTL
client.Put(ctx, "/services/web/instance-1", addr,
    clientv3.WithLease(lease.ID))
ch, _ := client.KeepAlive(ctx, lease.ID)
go func() { for range ch {} }()  // drain keepalive responses
```

=== Watch API

etcd watches are *streaming and persistent* (unlike ZooKeeper's one-shot watches). A watch stream is a gRPC server-streaming RPC; the server sends `WatchResponse` events containing the key, value, previous value, and revision for every create/update/delete matching the watched prefix. Watches survive leader elections transparently.

```go
watchChan := client.Watch(ctx, "/services/", clientv3.WithPrefix())
for resp := range watchChan {
    for _, ev := range resp.Events {
        fmt.Printf("%s %q : %q\n", ev.Type, ev.Kv.Key, ev.Kv.Value)
    }
}
```

=== Transactions (STM)

etcd supports optimistic concurrency via *compare-and-swap transactions*:

```go
_, err := client.Txn(ctx).
    If(clientv3.Compare(clientv3.Value("/lock"), "=", "")).
    Then(clientv3.OpPut("/lock", myID)).
    Else(clientv3.OpGet("/lock")).
    Commit()
```

The Software Transactional Memory (STM) library in the etcd client wraps this into a retry loop.

== Chubby

Chubby (Burrows 2006) is Google's internal distributed lock service, serving Bigtable, Spanner, and hundreds of other Google systems. Its design choices shaped ZooKeeper.

=== Design Philosophy

Chubby is explicitly a *coarse-grained* lock service: locks are held for hours or days (e.g., "I am the master"), not milliseconds. This contrasts with fine-grained locking (per-row locks in a database). The interface is a filesystem-like namespace of *cells*, each a Paxos group of 5 replicas.

*Advisory locks:* Chubby locks are advisory, not mandatory. A client that crashes while holding a lock simply causes the lock to expire after the session timeout. Chubby does not enforce that only the lock holder modifies the associated data — that is the application's responsibility (using sequencers).

=== Sequencers and Fencing

A client holding a lock can obtain a *sequencer* — an opaque byte string containing the lock name, mode, and lock generation number. When the client makes requests to application servers, it passes the sequencer. Application servers validate it via Chubby's `CheckSequencer()` call, rejecting stale sequencers from clients whose locks have expired.

=== Sessions and Jeopardy

When a Chubby client cannot contact the master for a *grace period* (default 45 s), it enters *jeopardy*: it signals the application that its lock may be invalid. If contact is re-established within the grace period, the session is safe. If not, the session expires and all ephemeral files and locks are released.

== Use Cases and Recipes

=== Leader Election

```
# ZooKeeper recipe
1. All candidates create ephemeral sequential /election/n- nodes
2. Candidate with smallest sequence number is leader
3. Others watch the node immediately preceding them
4. On predecessor deletion, re-check if now smallest -> leader

# etcd recipe (using concurrency.Election)
election := concurrency.NewElection(session, "/election")
election.Campaign(ctx, candidateValue)  // blocks until elected
```

=== Service Discovery

Services register ephemeral (ZooKeeper) or lease-bound (etcd) nodes under a prefix. Clients watch the prefix for membership changes. This is the pattern used by Kubernetes Endpoints, Consul service catalog, and etcd-based service meshes.

=== Distributed Locks

ZooKeeper sequential ephemeral nodes (see §ZooKeeper lock recipe above). etcd mutex via `concurrency.NewMutex`. Both degrade gracefully on client crash by expiring the ephemeral/lease entry.

=== Configuration Distribution

Write configuration to a single key or subtree; all consumers watch for changes. etcd's persistent watch and revision-based history make it easy to replay missed updates after a consumer restart.

== Comparison

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  table.header[*Property*][*ZooKeeper*][*etcd*][*Chubby*],
  [Consensus], [ZAB], [Raft], [Paxos],
  [API], [ZooKeeper client / ZK CLI], [gRPC / HTTP], [Internal C++ client],
  [Watch model], [One-shot, re-register], [Persistent streaming], [Callback (not public)],
  [History/MVCC], [No], [Yes (revision)], [No],
  [Leases/Ephemeral], [Ephemeral znodes], [Leases + TTL], [Sessions + grace period],
  [Max value size], [1 MB per znode], [1.5 MB (default)], [~256 KB],
  [Typical use], [Hadoop, Kafka, HBase], [Kubernetes, CoreDNS], [Bigtable, Spanner],
  [Lock style], [Advisory (sequential)], [Advisory (mutex/election)], [Advisory + sequencer],
)

== Anti-Patterns

*Using ZooKeeper or etcd as a database.* Both systems are designed for small, infrequently-written data (kilobytes, tens of thousands of keys). Writing megabytes of application data, storing large blobs, or using them as a message queue degrades their consensus performance and threatens cluster stability. A Kafka-backed or Redis-backed solution is more appropriate for high-throughput data paths.

*Tight polling instead of watches.* Applications that poll coordination services in a tight loop ($<$ 1 s) create unnecessary load. Use watches/events for real-time updates and fall back to polling only when watch state may be inconsistent (e.g., on reconnect).

*Long-held locks without heartbeating.* A client that holds an etcd lease without renewing it will silently lose the lease after TTL. Use `KeepAlive` and monitor keepalive errors. Set TTL conservatively (30–60 s) rather than aggressively short.

*Ignoring session expiry.* When a ZooKeeper session expires, all ephemeral nodes vanish and the client is disconnected. The application must detect `SESSION_EXPIRED` exceptions and restart its coordination logic from scratch — not merely re-acquire the lock.

== Further Reading

Hunt, P., et al. (2010). "ZooKeeper: Wait-free Coordination for Internet-Scale Systems." USENIX ATC.

Burrows, M. (2006). "The Chubby Lock Service for Loosely-Coupled Distributed Systems." OSDI.

Ongaro, D., Ousterhout, J. (2014). "In Search of an Understandable Consensus Algorithm." USENIX ATC.

Kleppmann, M. (2016). "How to do Distributed Locking." Martin Kleppmann's blog. (Discusses fencing token necessity.)

Junqueira, F., Reed, B. (2013). "ZooKeeper: Distributed Process Coordination." O'Reilly.

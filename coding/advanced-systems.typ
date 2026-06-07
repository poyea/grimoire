= Advanced Algorithms in Modern Systems

*Bridging theory and practice:* Algorithms don't exist in vacuum. Real-world systems - from CPU architectures to distributed databases to LLMs - rely on fundamental data structures and algorithms, optimized for specific hardware and workload characteristics.

#pagebreak()

= CPU Systems & Hardware

== Algorithms in CPU Microarchitecture

=== Branch Prediction & Speculative Execution

*Modern CPUs predict branches:* Skylake/Zen 3 reach $#sym.tilde.op$93-95% accuracy on SPEC-style workloads [Agner Fog, *Microarchitecture* 2023]; Raptor Lake P-cores and Zen 4 use TAGE-style predictors that reach 97-99% on the same suites [Seznec 2016, AMD/Intel optimization guides]. Algorithms still affect prediction quality — adversarial branch patterns (random data, hash-table chasing) drop accuracy to 50-70% on every microarch.

*Branch predictor types:*

1. *Local history:* Pattern-based (e.g., loop iterations)
2. *Global history:* Correlated branches (if-else chains)
3. *Tournament predictor:* Combines multiple predictors [Intel implements]

*Algorithm impact:*

```cpp
// BAD: Random-like branch pattern
for (int i = 0; i < n; i++) {
    if (data[i] % 7 == 0) {  // Random-looking condition
        // Unpredictable: ~50% branch misses = 15-20 cycle penalty each
    }
}

// GOOD: Sorted data enables prediction
sort(data.begin(), data.end());
for (int i = 0; i < n; i++) {
    if (data[i] < threshold) {  // Predictable: <5% branch misses after warmup
        // CPU learns pattern quickly
    }
}
```

*Speculative execution attacks:*
- Spectre: Branch predictor poisoning to leak data across privilege boundaries
- Meltdown: Speculative memory access before permission check
- Defense: Branch target injection barriers, kernel page-table isolation

*Algorithm design for security:* Constant-time algorithms avoid data-dependent branches.

```cpp
// Timing attack vulnerable
int compare_secret(const char* secret, const char* input) {
    for (int i = 0; secret[i]; i++) {
        if (secret[i] != input[i]) return 0;  // Early exit leaks info
    }
    return 1;
}

// Constant-time comparison
int compare_secret_safe(const char* secret, const char* input, int len) {
    int diff = 0;
    for (int i = 0; i < len; i++) {
        diff |= secret[i] ^ input[i];  // No early exit
    }
    return diff == 0;
}
```

=== Cache Replacement Policies

*LRU (Least Recently Used):* Most CPUs approximate with pseudo-LRU for performance.

*Algorithm:* Binary tree of usage bits per cache set.

```cpp
// 8-way set-associative cache: 7 bits per set
// Each bit tracks which half was more recently used
int evict_cache_line(int set, int usage_bits[8]) {
    int way = 0;

    // Traverse binary tree
    if (usage_bits[0]) way += 4;
    if (usage_bits[way ? 2 : 1]) way += 2;
    if (usage_bits[way ? 6 : 5]) way += 1;

    return way;  // Evict this way
}
```

*Performance:* Tree traversal = 3 bits for 8-way, logarithmic in associativity. True LRU requires $O(n log n)$ bits and complex logic.

*Cache-aware programming:* Understanding replacement helps optimize access patterns.

=== Out-of-Order Execution & Dependency Chains

*Modern CPUs reorder instructions:* Skylake executes up to 224 instructions in flight [Intel Optimization Manual 2023, §2.1.3].

*Register Allocation Mapping (RAT):* Renames registers to break false dependencies.

```cpp
// False dependency (same register)
a = x + y;  // eax = esi + edi
b = z + w;  // eax = edx + ecx (must wait for a)

// After register renaming (architectural → physical registers)
a = x + y;  // p1 = p5 + p6
b = z + w;  // p2 = p7 + p8 (executes in parallel)
```

*Algorithm impact - dependency chains:*

```cpp
// LONG dependency chain (serialized)
int sum = 0;
for (int x : data) {
    sum += x;  // sum depends on previous sum
}
// IPC ~1.0: a single sum-carrying dependency chain serialises through the
// adder's latency every iteration — register renaming can't help because
// each ADD's input *is* the previous ADD's output.

// REDUCED dependency (loop unrolling + multiple accumulators)
int sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0;
for (int i = 0; i < n; i += 4) {
    sum1 += data[i];      // Independent
    sum2 += data[i+1];    // Executes in parallel
    sum3 += data[i+2];    // 4x throughput
    sum4 += data[i+3];
}
int sum = sum1 + sum2 + sum3 + sum4;
// IPC ~3.0: saturates ALU units
```

=== Prefetching Algorithms

*Hardware prefetcher:* Detects sequential/stride access patterns.

*Stride prefetcher:* Tracks memory access deltas.

```cpp
// Sequential access (stride = 1): Perfect prefetching
for (int i = 0; i < n; i++) {
    sum += array[i];  // Prefetcher loads ahead
}

// Stride = 2: Also detected
for (int i = 0; i < n; i += 2) {
    sum += array[i];  // Prefetcher adapts to stride
}

// Random access: No prefetching possible
for (int i = 0; i < n; i++) {
    sum += array[random_index[i]];  // Every access = cache miss
}
```

*Software prefetch:* Manually insert prefetch instructions for complex patterns.

```cpp
// Linked list traversal (pointer chasing)
Node* curr = head;
while (curr) {
    if (curr->next) {
        __builtin_prefetch(curr->next);  // Prefetch 2-3 nodes ahead
        if (curr->next->next) {
            __builtin_prefetch(curr->next->next);
        }
    }

    process(curr->data);
    curr = curr->next;
}
```

#pagebreak()

= Operating System & Kernel

== Scheduler Algorithms

=== Linux Completely Fair Scheduler (CFS)

*Red-black tree of tasks:* O(log n) enqueue/dequeue.

```cpp
struct task_struct {
    uint64_t vruntime;  // Virtual runtime (execution time / weight)
    // ...
};

// Pick next task to run
task_struct* pick_next_task() {
    // Leftmost node in RB-tree = minimum vruntime
    return rb_tree_first(cfs_rbtree);  // O(1) after caching
}

// Update task after execution
void update_curr(task_struct* curr, uint64_t delta_exec) {
    curr->vruntime += delta_exec * NICE_0_LOAD / curr->weight;

    // Reinsert into tree if needed
    if (curr->vruntime > threshold) {
        rb_erase(curr, cfs_rbtree);
        rb_insert(curr, cfs_rbtree);  // O(log n)
    }
}
```

*Why RB-tree?* Balanced = guaranteed O(log n) worst-case. Min-heap would be O(1) pick but O(n) for finding task by PID.

*Cache behavior:* Tree traversal = pointer chasing. Small task count (< 100) = tree fits in cache.

=== Real-Time Scheduler (SCHED_FIFO, SCHED_RR)

*Priority queues:* Fixed-priority per queue, O(1) operations.

```cpp
struct rt_prio_array {
    DECLARE_BITMAP(bitmap, MAX_RT_PRIO);  // Bitmap of non-empty queues
    struct list_head queue[MAX_RT_PRIO];  // Linked list per priority
};

// O(1) pick highest priority task
task_struct* pick_next_rt() {
    int idx = find_first_bit(prio_array->bitmap, MAX_RT_PRIO);
    return list_first_entry(&prio_array->queue[idx], task_struct, rt_list);
}
```

*find_first_bit:* Uses TZCNT/BSF instruction (3 cycles) or software fallback.

== Memory Management

=== Page Replacement Algorithms

*LRU approximation (Linux):* Two lists (active/inactive) avoid full LRU overhead.

```cpp
// Simplified 2-list LRU
struct lru_lists {
    list_head active;
    list_head inactive;
};

void page_referenced(struct page* page) {
    if (page->list == INACTIVE) {
        move_to_list(page, ACTIVE);  // Promote
    }
    page->referenced = true;
}

void shrink_memory() {
    // Move unreferenced active pages to inactive
    for (page : active_list) {
        if (!page->referenced) {
            move_to_list(page, INACTIVE);
        }
        page->referenced = false;  // Clear for next scan
    }

    // Evict from inactive list
    evict_pages(inactive_list);
}
```

*Clock algorithm:* Single circular list with "referenced" bit.

```cpp
struct page* page_evict_clock() {
    while (true) {
        if (!clock_hand->referenced) {
            page* victim = clock_hand;
            clock_hand = next_page(clock_hand);
            return victim;
        }

        clock_hand->referenced = false;  // Second chance
        clock_hand = next_page(clock_hand);
    }
}
```

*Performance:* O(1) amortized. Circular list = good cache locality.

=== Slab Allocator

*Kernel object caching:* Avoids repeated allocation/initialization overhead.

```cpp
// Simplified slab allocator
struct kmem_cache {
    size_t object_size;
    list_head slabs_full;
    list_head slabs_partial;
    list_head slabs_empty;
};

void* kmem_cache_alloc(kmem_cache* cache) {
    // Try partial slabs first (has free objects)
    if (!list_empty(&cache->slabs_partial)) {
        slab* s = list_first_entry(&cache->slabs_partial, slab, list);
        void* obj = get_free_object(s);

        if (slab_is_full(s)) {
            move_to_list(s, &cache->slabs_full);
        }

        return obj;
    }

    // Allocate new slab if needed
    slab* s = alloc_new_slab(cache);
    void* obj = get_free_object(s);
    add_to_list(s, &cache->slabs_partial);

    return obj;
}
```

*Benefits:*
- Reduces fragmentation (same-sized objects)
- Cache-line aligned objects = fewer false sharing issues
- Constructor/destructor amortization

== File System Algorithms

=== B+ Tree (ext4, XFS, Btrfs)

*Disk-optimized tree:* Nodes sized to match disk block (4KB).

```cpp
const int B = 4096 / sizeof(key_value_pair);  // ~256 entries per node

struct btree_node {
    int num_keys;
    key_value_pair data[B];
    btree_node* children[B + 1];  // B+1 children
};
```

*Why B+ tree over binary tree?*
- Height $log_B n$ vs $log_2 n$ → fewer disk seeks
- 4KB node = single disk read (no partial reads)
- Sequential scans on leaves (linked list)

*Cache consideration:* Entire node loaded into page cache, amortizing disk I/O.

=== Log-Structured Merge Tree (LSM) - RocksDB, LevelDB

*Write optimization:* Sequential writes instead of random updates.

```cpp
struct lsm_tree {
    memtable memory_table;             // In-memory sorted map
    vector<sstable> level0_tables;     // Immutable sorted files
    vector<vector<sstable>> levels;    // Level 1, 2, ...
};

void put(key k, value v) {
    memory_table.insert(k, v);  // O(log n) in-memory insert

    if (memory_table.size() > threshold) {
        sstable* sst = flush_to_disk(memory_table);  // Sequential write
        level0_tables.push_back(sst);
        memory_table.clear();

        trigger_compaction();  // Merge levels in background
    }
}

value get(key k) {
    // Search memory table first
    if (memory_table.contains(k)) {
        return memory_table[k];
    }

    // Search L0 tables (may have duplicates)
    for (sstable* sst : level0_tables) {
        if (sst->bloom_filter->might_contain(k)) {
            value v = sst->get(k);  // Binary search in file
            if (v) return v;
        }
    }

    // Search deeper levels
    for (level : levels) {
        // Each level has disjoint key ranges
        sstable* sst = find_table_for_key(level, k);
        if (sst && sst->bloom_filter->might_contain(k)) {
            value v = sst->get(k);
            if (v) return v;
        }
    }

    return NOT_FOUND;
}
```

*Bloom filter:* Probabilistic set membership, reduces disk I/O.

*Write amplification:* Data rewritten during compaction. Trade-off: write throughput vs read latency.

#pagebreak()

= Distributed Systems

== Consensus Algorithms

=== Raft (Etcd, Consul)

*Leader election + Log replication:* Simpler than Paxos, same guarantees.

```cpp
enum ServerState { FOLLOWER, CANDIDATE, LEADER };

struct raft_server {
    ServerState state;
    int current_term;
    int voted_for;
    vector<log_entry> log;
    int commit_index;
};

void start_election(raft_server* server) {
    server->state = CANDIDATE;
    server->current_term++;
    server->voted_for = server->id;

    int votes = 1;  // Vote for self

    for (peer : cluster) {
        vote_granted = request_vote(peer, server->current_term, server->log);
        if (vote_granted) votes++;

        if (votes > cluster.size() / 2) {
            server->state = LEADER;
            send_heartbeats();
            return;
        }
    }

    // Election timeout → restart
}

void append_entries(log_entry entry) {
    if (server->state != LEADER) {
        forward_to_leader(entry);
        return;
    }

    server->log.push_back(entry);

    // Replicate to followers
    int acks = 1;  // Self
    for (follower : followers) {
        bool success = replicate_log(follower, entry);
        if (success) acks++;

        if (acks > cluster.size() / 2) {
            server->commit_index++;
            apply_to_state_machine(entry);
            return;
        }
    }
}
```

*Network partitions:* Only majority partition can make progress. Prevents split-brain.

*Algorithm complexity:* O(1) leader append, O(cluster_size) replication latency.

=== Distributed Hash Tables (DHT)

*Chord:* Consistent hashing with $O(log n)$ lookup.

```cpp
struct chord_node {
    uint160_t id;  // SHA-1 hash of IP:port
    chord_node* successor;
    chord_node* finger_table[160];  // Finger[i] = successor of (id + 2^i)
};

chord_node* find_successor(uint160_t key) {
    if (key in (id, successor->id]) {
        return successor;
    }

    // Find closest preceding node
    chord_node* closest = find_closest_preceding_node(key);
    return closest->find_successor(key);  // Recursive RPC
}

chord_node* find_closest_preceding_node(uint160_t key) {
    for (int i = 159; i >= 0; i--) {
        if (finger_table[i]->id in (id, key)) {
            return finger_table[i];
        }
    }
    return this;
}
```

*Routing:* Each hop halves distance → $O(log N)$ hops for N nodes.

*Replication:* Store key at k successors for fault tolerance.


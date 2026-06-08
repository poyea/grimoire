= Distributed and ML Algorithms

Distributed coordination, large-model training, and inference algorithms share a common thread: they push single-machine abstractions to their limits and require explicit reasoning about concurrency, failure, and communication cost. This chapter covers consensus-adjacent coordination primitives, data-parallel and pipeline-parallel training patterns, and LLM inference scheduling — the algorithms a systems engineer encounters when a workload outgrows a single node.

*See also:* _Advanced Algorithms in Modern Systems_ (coding), _Distributed Transactions_ (database), _Inference Optimization_ (llm), _GPU Memory Hierarchy_ (gpu-architecture).

== Distributed Coordination

=== Two-Phase Commit (2PC)

*Atomic commit across nodes:* Coordinator ensures all-or-nothing.

```cpp
enum Vote { COMMIT, ABORT };

// Phase 1: Prepare
bool coordinator_prepare(transaction txn) {
    for (participant : txn.participants) {
        vote = send_prepare(participant, txn);

        if (vote == ABORT) {
            send_abort_all(txn);
            return false;
        }
    }

    return true;  // All voted COMMIT
}

// Phase 2: Commit
void coordinator_commit(transaction txn) {
    for (participant : txn.participants) {
        send_commit(participant, txn);
    }

    log_transaction_complete(txn);
}

// Participant side
Vote participant_prepare(transaction txn) {
    if (can_commit(txn)) {
        write_undo_log(txn);  // Prepare for rollback
        return COMMIT;
    }

    return ABORT;
}

void participant_commit(transaction txn) {
    apply_transaction(txn);
    delete_undo_log(txn);
}
```

*Blocking problem:* If coordinator fails between phases, participants block indefinitely.

*3PC (Three-Phase Commit):* Adds pre-commit phase to avoid blocking, but more complex.

#pagebreak()

= AI & Machine Learning Systems

== Algorithms in LLM Inference

=== Attention Mechanism Optimization

*Transformer attention:* $O(n^2)$ complexity for sequence length n.

```cpp
// Naive attention: O(n^2 d)
Tensor attention_naive(Tensor Q, Tensor K, Tensor V) {
    // Q, K, V: [batch, seq_len, d_model]
    Tensor scores = matmul(Q, K.transpose());  // [batch, n, n]
    scores = softmax(scores / sqrt(d_model));
    return matmul(scores, V);  // [batch, n, d_model]
}
```

*Flash Attention:* Fused kernel + tiling for reduced memory I/O.

```cpp
// Flash Attention: O(n^2 d) time, O(n) memory.
// Naive per-block softmax + sum is WRONG (softmax doesn't decompose over
// key blocks). Real Flash Attention uses an *online softmax*: per query
// block, track running row max m and denom l, and rescale the partial
// output when a new block raises the max. See Dao et al. (2022) §3.
Tensor flash_attention(Tensor Q, Tensor K, Tensor V) {
    const int block_size = 128;  // Tile size (fits in SRAM)
    Tensor output = zeros_like(Q);

    for (int i = 0; i < n; i += block_size) {
        Tensor Q_block = Q.slice(i, i + block_size);     // [B, d]
        Vector m = full(block_size, -INFINITY);          // running row-max
        Vector l = zeros(block_size);                    // running row-denom
        Tensor O = zeros({block_size, d_model});         // partial output

        for (int j = 0; j < n; j += block_size) {
            Tensor K_block = K.slice(j, j + block_size);
            Tensor V_block = V.slice(j, j + block_size);

            Tensor S = matmul(Q_block, K_block.T()) / sqrt(d_model);  // [B, B]
            Vector m_new = max(m, rowmax(S));                          // [B]
            Tensor P = exp(S - m_new.unsqueeze(1));                    // [B, B]
            Vector alpha = exp(m - m_new);                             // rescale factor
            Vector l_new = alpha * l + rowsum(P);                      // [B]

            O = O * alpha.unsqueeze(1) + matmul(P, V_block);           // rescale + accumulate
            m = m_new;
            l = l_new;
        }
        output.slice(i, i + block_size) = O / l.unsqueeze(1);
    }
    return output;
}
```

*HBM bandwidth savings:* Standard attention = O(n^2) HBM reads. Flash = O(n) HBM reads.

*Speedup:* 2-4x for long sequences (n > 2048) on GPUs with fast SRAM.

=== KV Cache Management

*Problem:* Attention requires all previous key/value vectors. Memory grows with sequence length.

*Paged Attention (vLLM):* Virtual memory for KV cache.

```cpp
using seq_id = uint64_t;
struct kv_page;  // PAGE_SIZE tokens worth of K and V tensors
bool is_page_full(int page_id);
void evict_lru_page(struct kv_cache* cache);
void write_to_page(kv_page* page, Tensor k, Tensor v);

struct kv_cache {
    vector<kv_page*> physical_pages;                         // Page pool
    vector<int> free_pages;                                  // Free list (page indices)
    unordered_map<seq_id, vector<int>> logical_to_physical;  // Per-sequence page table
};

int allocate_page(kv_cache* cache) {
    if (cache->free_pages.empty()) {
        evict_lru_page(cache);
    }
    int page_id = cache->free_pages.back();
    cache->free_pages.pop_back();
    return page_id;
}

void append_kv(kv_cache* cache, seq_id id, Tensor k, Tensor v) {
    vector<int>& page_table = cache->logical_to_physical[id];

    // Check if last page has space
    if (page_table.empty() || is_page_full(page_table.back())) {
        int new_page = allocate_page(cache);
        page_table.push_back(new_page);
    }

    int page_id = page_table.back();
    write_to_page(cache->physical_pages[page_id], k, v);
}
```

*Memory efficiency:* Reduces waste from padding, enables batching sequences of different lengths.

*Copy-on-write:* Share KV cache for beam search variants.

=== Quantization Algorithms

*INT8 quantization:* Reduce 32-bit floats to 8-bit integers.

```cpp
// Asymmetric uint8 quantization. zero_point must be in [0, 255], so
// store it as uint8 — int8 would overflow for any range where -min/scale
// falls outside [-128, 127].
struct quantization_params {
    float scale;
    uint8_t zero_point;
};

quantization_params compute_params(Tensor weights) {
    float min_val = weights.min();
    float max_val = weights.max();

    float scale = (max_val - min_val) / 255.0f;
    int zp = (int)round(-min_val / scale);
    uint8_t zero_point = (uint8_t)clamp(zp, 0, 255);
    return {scale, zero_point};
}

Tensor quantize(Tensor weights, quantization_params params) {
    Tensor quantized = empty_like(weights, dtype=UINT8);
    for (int i = 0; i < weights.size(); i++) {
        int q = (int)round(weights[i] / params.scale) + params.zero_point;
        quantized[i] = (uint8_t)clamp(q, 0, 255);
    }
    return quantized;
}

Tensor dequantize(Tensor quantized, quantization_params params) {
    Tensor weights = empty_like(quantized, dtype=FLOAT32);
    for (int i = 0; i < quantized.size(); i++) {
        weights[i] = ((int)quantized[i] - params.zero_point) * params.scale;
    }
    return weights;
}
```

*Memory savings:* 4x reduction (32-bit → 8-bit). Accuracy loss typically < 1% for inference.

*INT4/INT1 (1-bit):* Further compression with minimal accuracy degradation for large models.

== Training Algorithms

=== Data Parallel Training

*Synchronous SGD:* Each GPU computes gradients on subset, then all-reduce.

```cpp
void distributed_sgd_step(model& local_model, Tensor batch, int world_size) {
    // Forward + backward on local batch
    Tensor local_grad = local_model.backward(batch);

    // All-reduce gradients across GPUs
    Tensor global_grad = allreduce_sum(local_grad) / world_size;

    // Update local model
    local_model.update(global_grad);
}

Tensor allreduce_sum(Tensor local_grad) {
    // Ring all-reduce: reduce-scatter then all-gather, each in P-1 steps.
    // Per-step send/recv chunk index depends on rank and iteration —
    // sending the same chunk every step (as a naive sketch does) doesn't
    // propagate the sum.
    int rank = get_rank();
    int size = get_world_size();
    int send_to = (rank + 1) % size;
    int recv_from = (rank - 1 + size) % size;

    Tensor result = copy(local_grad);  // chunked into `size` shards

    // Phase 1 — reduce-scatter: after P-1 steps, rank r holds the sum
    // of shard (r + 1) mod P across all ranks.
    for (int i = 0; i < size - 1; i++) {
        int send_idx = (rank - i + size) % size;
        int recv_idx = (rank - i - 1 + size) % size;
        send_async(result.shard(send_idx), send_to);
        Tensor in = recv_async(recv_from);
        result.shard(recv_idx) += in;
    }
    // Phase 2 — all-gather: each rank's complete shard rotates around.
    for (int i = 0; i < size - 1; i++) {
        int send_idx = (rank - i + 1 + size) % size;
        int recv_idx = (rank - i + size) % size;
        send_async(result.shard(send_idx), send_to);
        result.shard(recv_idx) = recv_async(recv_from);
    }
    return result;
}
```

*Bottleneck:* All-reduce communication. Ring algorithm optimal for bandwidth-bound networks.

=== Gradient Checkpointing (Recomputation)

*Trade compute for memory:* Recompute activations during backward instead of storing.

```cpp
Tensor checkpoint_forward(function<Tensor(Tensor)> layer, Tensor input) {
    // Don't save activations during forward
    Tensor output;

    {
        no_grad_guard guard;
        output = layer(input);
    }

    // Save only input and layer for backward
    save_for_backward(input, layer);

    return output;
}

Tensor checkpoint_backward(Tensor grad_output) {
    auto [input, layer] = retrieve_saved();

    // Recompute forward to get activations
    Tensor output = layer(input);

    // Now compute backward with activations
    Tensor grad_input = layer.backward(grad_output);

    return grad_input;
}
```

*Memory savings:* O(sqrt(n)) memory for n-layer model vs O(n).

*Compute overhead:* 33% extra FLOPs (one additional forward pass).

== References

*CPU & Hardware:*

*Intel Corporation (2023)*. Intel 64 and IA-32 Architectures Optimization Reference Manual.

*Agner Fog (2023)*. Instruction Tables. Technical University of Denmark.

*Hennessy, J.L. & Patterson, D.A. (2017)*. Computer Architecture: A Quantitative Approach (6th ed.). Morgan Kaufmann.

*Operating Systems:*

*Love, R. (2010)*. Linux Kernel Development (3rd ed.). Addison-Wesley. ISBN 978-0672329463.

*Gorman, M. (2004)*. Understanding the Linux Virtual Memory Manager. Prentice Hall.

*Distributed Systems:*

*Ongaro, D. & Ousterhout, J. (2014)*. In Search of an Understandable Consensus Algorithm (Raft). USENIX ATC.

*Stoica, I. et al. (2001)*. Chord: A Scalable Peer-to-peer Lookup Service for Internet Applications. SIGCOMM.

*Bernstein, P.A., Hadzilacos, V., & Goodman, N. (1987)*. Concurrency Control and Recovery in Database Systems. Addison-Wesley.

*AI & Machine Learning:*

*Dao, T. et al. (2022)*. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS.

*Kwon, W. et al. (2023)*. Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP.

*Vaswani, A. et al. (2017)*. Attention Is All You Need. NeurIPS.

*Chen, T. et al. (2016)*. Training Deep Nets with Sublinear Memory Cost. arXiv:1604.06174.

== Further Reading

Lynch, N. A. (1996). _Distributed Algorithms_. Morgan Kaufmann. (The definitive textbook on distributed computing: mutual exclusion, consensus, clock synchronization, and Byzantine fault tolerance.)

Lamport, L. (1978). "Time, Clocks, and the Ordering of Events in a Distributed System." _Communications of the ACM_ 21(7): 558--565. (Logical clocks and happened-before ordering — foundational to all distributed coordination.)

Fischer, M. J., Lynch, N. A., & Paterson, M. S. (1985). "Impossibility of Distributed Consensus with One Faulty Process." _Journal of the ACM_ 32(2): 374--382. (The FLP impossibility result, showing no deterministic algorithm can reach consensus in an asynchronous system with one faulty process.)

Attiya, H., & Welch, J. (2004). _Distributed Computing: Fundamentals, Simulations, and Advanced Topics_, 2nd ed. Wiley. (Rigorous treatment of consistency models, wait-freedom, and linearizability.)

Birman, K. P. (2012). _Guide to Reliable Distributed Systems_. Springer. (Practical perspective on fault-tolerant distributed systems, gossip protocols, and eventual consistency.)

#pagebreak()

== Advanced Java

*Comprehensive coverage of Java from fundamentals to high-performance concurrent systems. Designed for senior developer interviews at low-latency, high-throughput companies.*

=== Overview

This section covers both foundational and advanced Java concepts critical for senior Java developer roles, particularly in:
- High-frequency trading systems
- Low-latency financial applications
- High-throughput transaction processing
- Real-time risk management systems

*Key focus areas:*
- JVM internals & memory model
- Concurrency & thread safety
- Lock-free programming & CAS
- GC tuning & latency optimization
- Modern Java features (8-21)

*Prerequisites:* Basic Java syntax, understanding of OOP concepts

=== Interview Strategy

*Common interview patterns for senior Java roles:*

*Coding rounds:*
- Implement thread-safe data structures
- Fix concurrency bugs
- Optimize for low latency/high throughput
- Design concurrent systems

*System design rounds:*
- Order matching engine
- Real-time pricing system
- Market data feed handler
- High-frequency trading platform

*Behavioral:*
- Production incidents with concurrency issues
- Performance optimization war stories
- GC tuning experiences

#pagebreak()

#include "advanced-java/core-java-oop.typ"
#pagebreak()

#include "advanced-java/jvm-internals.typ"
#pagebreak()

#include "advanced-java/concurrency-primitives.typ"
#pagebreak()

#include "advanced-java/concurrent-utilities.typ"
#pagebreak()

#include "advanced-java/advanced-concurrency.typ"
#pagebreak()

#include "advanced-java/jvm-performance.typ"
#pagebreak()

#include "advanced-java/low-latency.typ"
#pagebreak()

#include "advanced-java/modern-java.typ"
#pagebreak()

#include "advanced-java/design-patterns.typ"

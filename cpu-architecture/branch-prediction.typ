= Branch Prediction

Branch prediction guesses the outcome and target of branches to maintain pipeline flow. Modern CPUs predict with 95-99% accuracy for typical code.

*See also:* Pipelining (for control hazards), Superscalar (for speculative execution)

== Cost of Misprediction

When a branch is mispredicted, the pipeline must be flushed, incurring a significant performance penalty. Modern CPUs have pipeline depths of 16-20 stages, resulting in misprediction penalties of 10-20 cycles. For example, with one branch per five instructions and a 5% mispredict rate, the branch overhead calculates to 0.2 × 0.05 × 15 = 0.15 cycles per instruction (CPI penalty).

```
Pipeline depth: 16-20 stages (modern CPUs)
Misprediction penalty: 10-20 cycles

Example: 1 branch per 5 instructions, 5% mispredict rate
Branch overhead = 0.2 × 0.05 × 15 = 0.15 CPI penalty
```

== Branch Types

Conditional branches evaluate a condition and may or may not be taken depending on the outcome:
```asm
cmp rax, rbx
je  target      ; Taken or not taken?
```

Unconditional branches always alter the control flow. The `jmp` instruction always jumps to the target, `call` jumps and pushes the return address onto the stack, while `ret` returns to the address predicted from the return address stack:
```asm
jmp target      ; Always taken
call func       ; Always taken + push return
ret             ; Return address predicted from stack
```

Indirect branches have targets that are unknown until runtime, commonly used for virtual function calls and switch statements:
```asm
jmp [rax]       ; Target unknown until runtime (virtual functions, switch)
```

== Static Prediction

Static prediction uses compiler hints to guess branch outcomes. Backward branches, typically found in loops, are predicted as taken, while forward branches are predicted as not taken. This legacy approach achieves approximately 60-70% accuracy, which is inadequate for modern CPU performance requirements.

== Dynamic Branch Prediction

A 1-bit predictor remembers only the last branch outcome. This simple approach has a significant problem with loops: on the last iteration, it mispredicts the transition from taken to not taken, and on the first iteration of the next loop invocation, it mispredicts again when transitioning from not taken to taken, resulting in two mispredicts per loop.

```
State: Taken | Not Taken

Problem: Loop with N iterations
- Last iteration: Taken → Not Taken (mispredict)
- First iteration next time: Not Taken → Taken (mispredict)
- 2 mispredicts per loop (poor for tight loops)
```

The 2-bit saturating counter introduces hysteresis, using four states: strongly not taken (00), weakly not taken (01), weakly taken (10), and strongly taken (11). When a branch is taken, the counter increments and saturates at 11; when not taken, it decrements and saturates at 00. The prediction is "taken" for states 10 and 11, and "not taken" for states 00 and 01. This approach provides a key benefit: a single anomalous outcome does not immediately change the prediction, and a loop with N iterations produces only one mispredict at the end.

```
States: 00 (Strongly Not Taken)
        01 (Weakly Not Taken)
        10 (Weakly Taken)
        11 (Strongly Taken)

Transitions:
- Taken: Increment (saturate at 11)
- Not Taken: Decrement (saturate at 00)
- Predict: 10,11 → Taken; 00,01 → Not Taken

Benefit: Single anomaly doesn't change prediction
Loop with N iterations: Only 1 mispredict at end
```

== Branch Target Buffer (BTB)

The Branch Target Buffer (BTB) solves the problem of needing the predicted target address early in the pipeline. It functions as a cache that maps branch program counter values to their target addresses.

Each BTB entry contains the branch PC as a tag, the target address, the branch type (conditional, call, return, or indirect), and prediction bits implementing a 2-bit counter. The lookup occurs in parallel with instruction fetch: the PC is hashed to determine a BTB index, the tag is compared to check for a hit, and if a hit occurs, the predictor decides whether the branch is taken or not taken and fetches from the target address. On a miss, the predictor assumes the branch is not taken and fetches sequentially.

```
BTB Entry:
├─ Branch PC (tag)
├─ Target address
├─ Branch type (conditional, call, return, indirect)
└─ Prediction bits (2-bit counter)

Lookup (parallel with instruction fetch):
1. Hash PC → BTB index
2. Compare tag, check hit
3. If hit: Predict taken/not taken, fetch from target
4. If miss: Predict not taken (fetch sequentially)
```

Typical BTB sizes range from 4096 to 8192 entries, with Intel Skylake featuring 4K entries. When a branch is encountered for the first time and causes a BTB miss, the predictor assumes it is not taken, which results in a misprediction if the branch is actually taken.

== Two-Level Adaptive Predictors

Two-level adaptive predictors exploit the key insight that branch outcomes correlate with recent branch history [Yeh & Patt 1991]. For example, consider two branches where Branch B's outcome depends on Branch A's behavior:

```c
if (x < 0)      // Branch A
    y = -x;
if (y > 100)    // Branch B (correlates with A!)
    z = 100;
```

The Global History Register (GHR) tracks the last N branch outcomes using an N-bit shift register. The Pattern History Table (PHT) is indexed by XORing the GHR with the branch PC to achieve better distribution, with each PHT entry containing a 2-bit counter.

During prediction, the processor reads the GHR, computes the index by XORing GHR with the PC, reads the 2-bit counter from PHT[index], and makes the prediction. After the branch resolves, the GHR is updated by shifting left and ORing with the outcome, and the PHT[index] is updated based on the actual outcome.

```
GHR = 10110  (last 5 branches: TNTTNT)

Pattern History Table (PHT):
Index = GHR ⊕ Branch_PC  (XOR for better distribution)
PHT[index] = 2-bit counter

Prediction:
1. Read GHR
2. Compute index = GHR ⊕ PC
3. Read PHT[index] → 2-bit counter → predict

Update:
1. Shift GHR: GHR = (GHR << 1) | outcome
2. Update PHT[index] based on outcome
```

This approach achieves approximately 90-95% accuracy for integer code.

== TAGE Predictor (Tagged Geometric History)

The TAGE predictor represents the state-of-the-art in branch prediction [Seznec & Michaud 2006]. It uses multiple Pattern History Tables with geometrically increasing history lengths: 2, 4, 8, 16, 32, up to 1024 entries.

During lookup, all PHTs are checked in parallel, prioritizing the longest history first. The prediction comes from the longest matching history entry, which provides the most specific context. If no match is found, the predictor falls back to a base predictor. Updates occur when mispredictions happen: the predictor allocates an entry in a longer-history table to learn the pattern, and unused entries are aged out over time.

```
Multiple PHTs with different history lengths: 2, 4, 8, 16, 32, ..., 1024

Lookup:
1. Check all PHTs in parallel (longest history first)
2. Use prediction from longest matching history (most specific)
3. Fallback to base predictor if no match

Update:
- Allocate entry in longer-history table on mispredict (learn pattern)
- Age out unused entries
```

The TAGE predictor achieves 97-99% accuracy. Intel uses variants of TAGE in modern CPUs, though the specific implementation details remain proprietary.

== Return Address Stack (RAS)

The `ret` instruction presents a prediction problem because its target depends on the call chain and is therefore unpredictable using standard techniques. The solution is a hardware stack that tracks return addresses.

When a `call` instruction executes, the return address is pushed onto the RAS before jumping to the function. When a `ret` instruction executes, the return address is popped from the RAS, and the CPU predicts a return to that popped address. Typical RAS sizes range from 16 to 32 entries.

```
call func:
1. Push return address to RAS
2. Jump to func

ret:
1. Pop return address from RAS
2. Predict return to popped address

RAS size: 16-32 entries typical
```

RAS overflow occurs when deep recursion exceeds the RAS capacity, causing the stack to wrap around and resulting in mispredictions. For example, with a 40-level recursion and a 32-entry RAS, overflows occur at depths greater than 32, and returns at depth 33 and beyond will mispredict due to RAS wraparound:

```c
// 40 levels of recursion, RAS size = 32
void deep_recursion(int n) {
    if (n == 0) return;
    deep_recursion(n - 1);  // RAS overflow at depth > 32
}
// Returns at depth 33+ mispredict (RAS wraparound)
```

== Indirect Branch Prediction

Indirect branches present a significant challenge because their targets are determined at runtime. Common sources include virtual function calls, function pointers, and switch statements. For example, a virtual function call through a vtable has an unknown target until the object type is resolved:

```cpp
class Base {
    virtual void func() = 0;  // Indirect call via vtable
};

void call_virtual(Base* obj) {
    obj->func();  // Call [obj->vtable + offset] - unknown target!
}
```

The target cache addresses this challenge by storing recent targets indexed by the branch PC. Each entry contains the branch PC as a tag and 2-4 recent targets maintained with an LRU replacement policy. The predictor returns the most recent target for a given PC, achieving accuracy of 80-90% depending on the degree of polymorphism.

```
Indirect Target Cache:
├─ Branch PC (tag)
└─ Recent targets (2-4 entries, LRU)

Prediction: Return most recent target for this PC
Accuracy: 80-90% (depends on polymorphism degree)
```

Advanced implementations use tagged target predictors that incorporate call-site history to improve prediction accuracy.

== Predication (Branchless Code)

Predication offers an alternative to branch prediction by eliminating the branch entirely through conditional move instructions. The branching version uses a jump that may be mispredicted, while the predicated version executes a conditional move that avoids branches altogether:

```asm
; Branching version:
cmp  rax, rbx
jle  skip
mov  rcx, rdx
skip:

; Predicated version (branchless):
cmp  rax, rbx
cmovg rcx, rdx   ; Conditional move: if rax > rbx then rcx ← rdx

No branch → no misprediction penalty
Cost: Always execute both paths (wastes work if condition predictable)
```

The benefit of predication is that it eliminates misprediction penalties, but the cost is that both paths always execute, wasting work when the condition is highly predictable.

Predication is most effective for unpredictable branches with less than 90% accuracy, short code sequences of 1-3 instructions, and operations without side effects (since `cmov` is only safe for simple operations). The compiler flag `-fno-if-conversion` can be used to disable automatic predication.

== Software Hints

C++20 provides likely and unlikely attributes that guide the compiler's code generation. The `[[likely]]` attribute suggests that a path is frequently executed, prompting the compiler to place the code inline, while `[[unlikely]]` indicates a cold path that the compiler may outline:

```cpp
if (x > 0) [[likely]] {
    // Hot path - compiler places code inline
} else [[unlikely]] {
    // Cold path - compiler may outline
}
```

GCC and Clang provide builtin hints through `__builtin_expect`, where the second argument indicates the expected value:

```cpp
if (__builtin_expect(ptr != NULL, 1)) {  // Likely true
    // Hot path
}
```

These hints primarily affect code layout optimization rather than directly controlling prediction on modern CPUs.

== Measurement

```bash
perf stat -e branches,branch-misses ./program

# Good:  <5% mispredict rate
# OK:    5-10%
# Poor:  >10% (investigate code patterns)
```

*Per-function analysis:*
```bash
perf record -e branch-misses ./program
perf report

# Detailed branch types (Intel)
perf stat -e br_inst_retired.conditional,\
br_misp_retired.conditional,\
br_inst_retired.near_call,\
br_misp_retired.near_call ./program
```

== Practical Optimization Techniques

*1. Profile-Guided Optimization (PGO):*

```bash
# Step 1: Compile with profiling
gcc -O3 -fprofile-generate code.c -o program

# Step 2: Run with representative workload
./program < typical_input.txt

# Step 3: Compile with profile data
gcc -O3 -fprofile-use code.c -o program_optimized

# Benefits: 5-15% speedup from better:
# - Code layout (hot paths inline, cold paths outlined)
# - Branch prediction hints
# - Inlining decisions
```

*2. Sorting data for better predictability:*

```c
// BAD: Random order (50% mispredicts)
int sum = 0;
for (int i = 0; i < n; i++) {
    if (data[i] > threshold) {
        sum += data[i];
    }
}
// Branch miss rate: ~50% if data random

// GOOD: Sort first (0% mispredicts after initial transient)
qsort(data, n, sizeof(int), compare);
for (int i = 0; i < n; i++) {
    if (data[i] > threshold) {  // Predictable: all false, then all true
        sum += data[i];
    }
}
// Branch miss rate: <1%
// Note: Only worthwhile if data reused multiple times
```

*3. Branch elimination with computation:*

```c
// Instead of:
if (x < 0) x = -x;

// Use:
int mask = x >> 31;      // Arithmetic shift: -1 if negative, 0 if positive
x = (x ^ mask) - mask;   // Branchless absolute value

// Or compiler intrinsic:
x = abs(x);  // Often compiled to branchless code
```

*4. Early exit optimization:*

```c
// BAD: Check condition on every iteration
for (int i = 0; i < 1000000; i++) {
    if (unlikely_condition) break;  // Mispredicted 999,999 times!
    work(i);
}

// GOOD: Hoist check outside hot path
bool should_continue = true;
for (int i = 0; i < 1000000 && should_continue; i++) {
    work(i);
    if (i % 1000 == 0) {  // Check less frequently
        should_continue = !unlikely_condition;
    }
}
```

== References

Yeh, T-Y. & Patt, Y.N. (1991). "Two-Level Adaptive Training Branch Prediction." MICRO-24.

Seznec, A. & Michaud, P. (2006). "A Case for (Partially) TAgged GEometric History Length Branch Prediction." Journal of Instruction-Level Parallelism 8: 1-23.

Smith, J.E. (1981). "A Study of Branch Prediction Strategies." ISCA '81.

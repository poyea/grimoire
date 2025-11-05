= Branch Prediction

Branch prediction guesses the outcome and target of branches to maintain pipeline flow. Modern CPUs predict with 95-99% accuracy for typical code.

*See also:* Pipelining (for control hazards), Superscalar (for speculative execution)

== Cost of Misprediction

*Pipeline flush penalty:*

```
Pipeline depth: 16-20 stages (modern CPUs)
Misprediction penalty: 10-20 cycles

Example: 1 branch per 5 instructions, 5% mispredict rate
Branch overhead = 0.2 × 0.05 × 15 = 0.15 CPI penalty
```

== Branch Types

*Conditional branches:*
```asm
cmp rax, rbx
je  target      ; Taken or not taken?
```

*Unconditional branches:*
```asm
jmp target      ; Always taken
call func       ; Always taken + push return
ret             ; Return address predicted from stack
```

*Indirect branches:*
```asm
jmp [rax]       ; Target unknown until runtime (virtual functions, switch)
```

== Static Prediction

*Compiler hints (legacy):*
- Backward branches (loops): Predict taken
- Forward branches: Predict not taken

*Performance:* ~60-70% accuracy (inadequate for modern CPUs).

== Dynamic Branch Prediction

*1-Bit Predictor:* Remember last outcome.

```
State: Taken | Not Taken

Problem: Loop with N iterations
- Last iteration: Taken → Not Taken (mispredict)
- First iteration next time: Not Taken → Taken (mispredict)
- 2 mispredicts per loop (poor for tight loops)
```

*2-Bit Saturating Counter:* Hysteresis.

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

*Problem:* Need predicted target address early in pipeline.

*BTB:* Cache mapping branch PC → target address.

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

*BTB size:* 4096-8192 entries typical (Intel Skylake: 4K entries).

*BTB miss:* First encounter of branch → predict not taken → mispredict if taken.

== Two-Level Adaptive Predictors

*Key insight:* Branch outcome correlates with recent branch history [Yeh & Patt 1991].

*Example:*
```c
if (x < 0)      // Branch A
    y = -x;
if (y > 100)    // Branch B (correlates with A!)
    z = 100;
```

*Global History Register (GHR):* Track last N branch outcomes (N-bit shift register).

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

*Performance:* ~90-95% accuracy for integer code.

== TAGE Predictor (Tagged Geometric History)

*State-of-the-art predictor [Seznec & Michaud 2006]:*

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

*Performance:* 97-99% accuracy.

*Intel implementation:* Variants of TAGE in modern CPUs (proprietary details).

== Return Address Stack (RAS)

*Problem:* `ret` instruction has unpredictable target (depends on call chain).

*Solution:* Hardware stack tracks return addresses.

```
call func:
1. Push return address to RAS
2. Jump to func

ret:
1. Pop return address from RAS
2. Predict return to popped address

RAS size: 16-32 entries typical
```

*RAS overflow:* Deep recursion exceeds RAS → stack wraps → mispredicts.

```c
// 40 levels of recursion, RAS size = 32
void deep_recursion(int n) {
    if (n == 0) return;
    deep_recursion(n - 1);  // RAS overflow at depth > 32
}
// Returns at depth 33+ mispredict (RAS wraparound)
```

== Indirect Branch Prediction

*Challenge:* Virtual function calls, function pointers, switch statements.

```cpp
class Base {
    virtual void func() = 0;  // Indirect call via vtable
};

void call_virtual(Base* obj) {
    obj->func();  // Call [obj->vtable + offset] - unknown target!
}
```

*Target cache:* Store recent targets indexed by branch PC.

```
Indirect Target Cache:
├─ Branch PC (tag)
└─ Recent targets (2-4 entries, LRU)

Prediction: Return most recent target for this PC
Accuracy: 80-90% (depends on polymorphism degree)
```

*Advanced:* Tagged target predictors using call-site history.

== Predication (Branchless Code)

*Alternative to prediction:* Eliminate branch via conditional move.

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

*When to use:*
- Unpredictable branches (< 90% accuracy)
- Short code sequences (1-3 instructions)
- No side effects (cmov safe only for simple operations)

*Compiler flags:* `-fno-if-conversion` disables automatic predication.

== Software Hints

*Likely/unlikely attributes (C++20):*
```cpp
if (x > 0) [[likely]] {
    // Hot path - compiler places code inline
} else [[unlikely]] {
    // Cold path - compiler may outline
}
```

*Builtin hints (GCC/Clang):*
```cpp
if (__builtin_expect(ptr != NULL, 1)) {  // Likely true
    // Hot path
}
```

*Effect:* Code layout optimization (not direct prediction control on modern CPUs).

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
```

== References

Yeh, T-Y. & Patt, Y.N. (1991). "Two-Level Adaptive Training Branch Prediction." MICRO-24.

Seznec, A. & Michaud, P. (2006). "A Case for (Partially) TAgged GEometric History Length Branch Prediction." Journal of Instruction-Level Parallelism 8: 1-23.

Smith, J.E. (1981). "A Study of Branch Prediction Strategies." ISCA '81.

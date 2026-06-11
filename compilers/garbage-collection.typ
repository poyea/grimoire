= Garbage Collection

Garbage collection (GC) is the subsystem that reclaims heap memory automatically, without explicit `free` calls from the programmer. Every managed runtime --- the JVM, .NET CLR, Python, Go, Swift, JavaScript engines --- ships a GC, and the design of that GC is one of the most consequential choices a language implementer makes. This chapter surveys the principal algorithm families, the compiler infrastructure required to support them, and the trade-offs between pause time, throughput, and space overhead.

*See also:* _JIT Compilation and Runtime Systems_, _IR Design and SSA Form_, and the _Build a Compiler_ chapter in the Languages & Compilers volume.

== Heap Layout and the Root Set

The *heap* is the region of memory from which dynamic allocations are served. For the GC to be correct, it must be able to find every live object and must not reclaim any object reachable from the program.

*Root set.* A root is any pointer that exists outside the heap but refers into it. The GC traces reachability starting from roots:

- *Stack roots:* local variables and temporaries in every thread's call stack.
- *Global and static roots:* module-level variables, static fields.
- *Register roots:* pointer values held in CPU registers at the moment of collection.
- *JNI / FFI handles:* pointers passed to or from native code, recorded in a runtime table.

*Precise vs. conservative scanning.* A *precise* (or exact) GC knows the type of every value in every stack frame and register. It can distinguish a pointer `0x7f3c00` from an integer that happens to have the same bit pattern. Precision requires the compiler to emit *stack maps* (also called *GC maps*) at every safepoint, describing the location and type of each live value.

A *conservative* GC (Boehm-Demers-Weiser) treats any aligned word that looks like a valid heap address as a potential pointer. No compiler support is needed, making it easy to retrofit into C/C++ programs, but it can never move objects (a false pointer would become dangling), and it may retain garbage indefinitely due to false positives.

*Object layout.* Managed runtimes typically store a *header word* before the object payload, encoding the type tag (class pointer or type ID) and GC state bits (mark bit, forwarding bit, age). The GC walks the object graph by reading the type tag and following the pointer fields recorded in the type's descriptor.

== Mark-Sweep Collection

Mark-sweep is the oldest and most widely understood GC algorithm. It operates in two phases.

=== The Mark Phase

Starting from the root set, the collector performs a graph traversal, marking every reachable object. The canonical formulation uses a *tri-colour abstraction*:

- *White:* not yet seen; presumed garbage at the end of marking.
- *Grey:* discovered but not yet fully scanned (its outgoing pointers have not been traced).
- *Black:* fully scanned; all outgoing pointers have been traced to grey or black objects.

At the start, all objects are white and roots are greyed. The mark loop pops a grey object, traces its pointer fields (greying any white referents), then colours it black. Marking terminates when the grey set is empty; remaining white objects are garbage.

*Write barriers* are necessary whenever the mutator (application code) runs concurrently with the marker. Without a barrier, the mutator could:

+ Create a new black-to-white pointer (the black object will not be re-scanned).
+ Delete the only grey-to-white path for that white object.

Together these two conditions violate the *tri-colour invariant* (no black object points directly to a white object without a grey intermediary) and would cause a live object to be collected. The two classical barrier strategies are discussed in the concurrent-collection section below.

*Floating garbage.* Objects that become unreachable after marking begins but before marking ends are not collected in the current cycle; they persist until the next cycle. This is unavoidable in any incremental or concurrent collector and is called floating garbage.

=== The Sweep Phase

After marking, the sweep phase scans the entire heap linearly and reclaims white objects, returning their memory to a free list or a bump-pointer arena. Sweep can be done lazily (on the next allocation request) or eagerly (as a separate pass).

*Fragmentation.* Mark-sweep does not move objects, so the heap can become fragmented: many small free regions scattered between live objects. Fragmentation raises allocation cost (first-fit or best-fit searches) and can cause allocation failure even when total free memory is sufficient.

#table(
  columns: 3,
  [*Property*], [*Mark-Sweep*], [*Notes*],
  [Pause pattern], [Stop-the-world mark + sweep], [Pause proportional to live set + heap size],
  [Throughput], [High (no copying)], [Allocation via free list; fragmentation degrades over time],
  [Space overhead], [Low (1 mark bit/object)], [Free list metadata adds minor overhead],
  [Moving?], [No], [Conservative GC is therefore limited to mark-sweep variants],
  [Floating garbage], [Yes], [Bounded to one GC cycle],
)

== Copying Collection and Semi-Space

Copying collection solves fragmentation by moving all live objects into a contiguous region. The heap is split into two equal semi-spaces: *from-space* (currently in use) and *to-space* (empty). When from-space is exhausted, a collection copies live objects to to-space, compacting them, then swaps the roles of the two spaces.

=== Cheney's Algorithm

Cheney (1970) implements copying collection without a recursion stack, using to-space itself as a queue.

```cpp
// Cheney's breadth-first copying collector (toy implementation)
struct Object {
    uintptr_t header;   // low bit = 1 means forwarding pointer
    size_t    size;     // number of pointer-sized fields (after header)
    uintptr_t fields[]; // flexible array of fields (pointers or scalars)
};

static char* to_start;
static char* scan;      // BFS scan pointer
static char* free_ptr;  // bump pointer into to-space

Object* copy(Object* obj) {
    if (obj == nullptr) return nullptr;
    // Already copied: header holds forwarding pointer
    if (obj->header & 1)
        return reinterpret_cast<Object*>(obj->header & ~1ULL);
    // Copy object to to-space
    size_t bytes = sizeof(Object) + obj->size * sizeof(uintptr_t);
    Object* new_obj = reinterpret_cast<Object*>(free_ptr);
    memcpy(new_obj, obj, bytes);
    free_ptr += bytes;
    // Install forwarding pointer in from-space copy
    obj->header = reinterpret_cast<uintptr_t>(new_obj) | 1;
    return new_obj;
}

void collect(Object** roots, size_t num_roots) {
    scan = free_ptr = to_start;
    // Copy roots
    for (size_t i = 0; i < num_roots; ++i)
        roots[i] = copy(roots[i]);
    // BFS: scan copied objects and copy their referents
    while (scan < free_ptr) {
        Object* obj = reinterpret_cast<Object*>(scan);
        for (size_t i = 0; i < obj->size; ++i) {
            Object* ref = reinterpret_cast<Object*>(obj->fields[i]);
            if (ref) obj->fields[i] = reinterpret_cast<uintptr_t>(copy(ref));
        }
        scan += sizeof(Object) + obj->size * sizeof(uintptr_t);
    }
    // to-space is now the live heap; caller swaps from/to
}
```

After collection, `free_ptr` sits immediately after the last live object. All subsequent allocations are simple *bump-pointer* increments --- one or two instructions --- until the next collection.

*Throughput comparison: bump-pointer vs. free list.*

#table(
  columns: 3,
  [*Allocation strategy*], [*Cost per allocation*], [*Notes*],
  [Bump pointer], [$O(1)$, ~2 instructions], [Requires moving GC to reclaim space compactly],
  [Free list (first fit)], [$O(n)$ worst case], [Fragmentation degrades over time],
  [Free list (segregated)], [~10--30 cycles amortised], [tcmalloc / jemalloc style; reduces fragmentation],
)

*Space cost.* Copying collection uses half the heap for to-space at all times. Space overhead is therefore 2× the live set minimum, compared to mark-sweep's near-zero overhead. This is the principal drawback of pure semi-space collection.

== Generational Garbage Collection

Most objects die young. This *weak generational hypothesis*, empirically observed across many workloads, motivates generational GC: collect the young generation (nursery) frequently and cheaply, and collect the old generation (tenured space) rarely.

=== Structure

- *Nursery (young generation):* a small region (1--64 MB) collected with every minor GC. Allocation is bump-pointer. Minor GC copies survivors to the old generation; objects that survive several minor collections are *tenured*.
- *Old generation (tenured space):* a large region collected by a major GC, which is much less frequent.

=== Remembered Sets and Card Tables

A minor GC must treat old-generation pointers into the nursery as roots, because an old object may hold the only live reference to a young object. Scanning the entire old generation for such pointers would negate the performance advantage of a minor GC.

A *remembered set* records every old-to-young pointer. Write barriers maintain it: whenever the mutator stores a pointer from an old object into a young object, the barrier adds the old object (or the pointer slot) to the remembered set.

*Card tables* are the standard implementation. The heap is divided into *cards* of fixed size (typically 512 bytes). One byte of the card table corresponds to each card. A write barrier dirtying card $i$ is:

```cpp
// Simplified card table write barrier
inline void write_barrier(Object* old_obj, Object** slot, Object* new_val) {
    *slot = new_val;
    // Mark the card containing old_obj as dirty
    card_table[reinterpret_cast<uintptr_t>(old_obj) >> CARD_SHIFT] = DIRTY;
}
```

During minor GC, only dirty cards are scanned for old-to-young pointers. Because most old objects are not written between minor GCs, only a small fraction of cards are dirty at any one time, making the scan fast.

*Promotion.* Objects surviving $N$ minor collections (the *tenuring threshold*) are promoted to the old generation. Survivor spaces (often two small semi-spaces within the young generation) hold objects between nursery and tenured space.

== Incremental and Concurrent Collection

Stop-the-world collection is simple but unacceptable for latency-sensitive applications. Incremental and concurrent collectors allow the mutator to run during most of the GC work.

=== Write Barriers for Concurrent Marking

Two classical write barrier designs maintain the tri-colour invariant under concurrent mutation.

*Dijkstra insertion barrier (incremental update).* When the mutator stores a pointer `p` into a field, grey the referent `p` immediately:

```cpp
// Dijkstra insertion barrier
inline void write_ref(Object** slot, Object* new_val) {
    if (is_white(new_val)) grey(new_val);  // shade the new referent
    *slot = new_val;
}
```

Insertion barriers must re-scan roots at the end of marking (a *remark* pause), because roots themselves may have been modified after initial root scanning.

*Yuasa deletion barrier (snapshot-at-the-beginning, SATB).* When the mutator overwrites a pointer, record the old referent before it is lost:

```cpp
// Yuasa SATB deletion barrier
inline void write_ref(Object** slot, Object* new_val) {
    Object* old_val = *slot;
    if (old_val != nullptr && is_white(old_val)) grey(old_val);  // preserve snapshot
    *slot = new_val;
}
```

SATB traces the object graph as it existed at marking start. Any object reachable at that snapshot is kept alive even if the mutator later drops all references to it --- making floating garbage the price of avoiding a remark pause.

=== Pause-Time Analysis of Production Collectors

#table(
  columns: 4,
  [*Collector*], [*Pause model*], [*Barrier type*], [*Key mechanism*],
  [G1 (JVM)], [Bounded STW ~200 ms target], [SATB write + card table], [Heap divided into equal-size regions; evacuates highest-garbage regions first],
  [Shenandoah (JVM)], [~10 ms target; concurrent evacuation], [SATB write + Brooks load barrier], [Forwarding pointer in object header; load barrier returns new address],
  [ZGC (JVM, Linux)], [Sub-millisecond; concurrent relocation], [Coloured pointer load barrier], [Object state encoded in pointer high bits; load barrier heals stale pointers],
  [Go GC], [Sub-millisecond; concurrent mark-sweep], [Insertion (hybrid) write barrier], [Tri-colour concurrent mark; no compaction; periodic STW remark],
  [.NET Gen2 / BGC], [Background concurrent sweep], [Card table write barrier], [Background major GC concurrent with foreground minor GC],
)

*Coloured pointers (ZGC).* ZGC stores GC metadata in 4 unused high bits of 64-bit pointers (finalizable, remapped, marked0, marked1). Every pointer load passes through a load barrier that tests these bits; if the pointer is stale (refers to the old location of a relocated object), the barrier atomically installs the forwarding pointer and returns the new address. This *self-healing* means each stale pointer is fixed at most once, and subsequent loads find a clean pointer.

*Brooks forwarding pointer (Shenandoah).* Each object carries an extra header word (the forwarding pointer) initialised to point to itself. During concurrent evacuation, the GC atomically updates the forwarding pointer to the new location. A load barrier dereferences the forwarding pointer on every object access. The overhead is one extra indirection per object access.

== Reference Counting

Reference counting (RC) stores a count of incoming references in each object. When the count reaches zero, the object is immediately freed.

=== Naive Reference Counting

```cpp
// Naive intrusive reference counting
struct RcObject {
    size_t ref_count = 1;
    virtual ~RcObject() = default;
};

template <typename T>
struct Rc {
    T* ptr;
    explicit Rc(T* p) : ptr(p) {}
    Rc(const Rc& o) : ptr(o.ptr) { ++ptr->ref_count; }
    Rc& operator=(Rc o) { std::swap(ptr, o.ptr); return *this; }
    ~Rc() { if (--ptr->ref_count == 0) delete ptr; }
};
```

*Advantages:* immediate reclamation, no stop-the-world pauses, low space overhead, simple implementation.

*Disadvantages:*
- *Cycles:* two objects pointing to each other will never reach count zero. Cycle detection is required.
- *Performance:* every pointer assignment increments/decrements a count, causing cache line bouncing in multithreaded code.
- *Atomic overhead:* thread-safe RC requires atomic increments ($3$--$10 times$ slower than non-atomic on modern hardware).

=== Deferred Reference Counting

*Deferred RC* (Deutsch & Bobrow 1976) avoids updating the count for stack-to-heap references, which are the most frequent source of count traffic. Only heap-to-heap reference changes are counted. Periodically, a scan reconciles the stack with the deferred counts.

=== Cycle Detection: CPython

CPython uses naive RC as its primary memory manager but adds a *cyclic garbage collector* (the `gc` module) to handle cycles among container objects. The cycle detector uses a *trial deletion* algorithm:

+ Copy the reference counts of all container objects into a working set.
+ Subtract 1 for each internal (container-to-container) reference found by traversal. This simulates deleting each container.
+ Any object whose adjusted count reaches 0 is unreachable from external roots --- it is part of a cycle and can be collected.

The cycle collector runs periodically (generational thresholds: 700/10/10 objects by default) rather than on every allocation.

=== Swift ARC

Swift uses *Automatic Reference Counting* (ARC) inserted by the compiler rather than the runtime. The compiler inserts `retain` and `release` calls at every use and last-use of a reference-typed value. ARC is *not* a tracing GC: there are no safepoints, no stop-the-world pauses, and no global heap traversal. Cycles must be broken manually using `weak` or `unowned` references.

*Strong, weak, unowned:*

- `strong` reference: increments the retain count; keeps the object alive.
- `weak` reference: does not increment the count; becomes `nil` when the object is deallocated (requires an extra side-table lookup).
- `unowned` reference: does not increment the count; traps if accessed after deallocation (zero overhead compared to weak).

ARC's compile-time nature makes it predictable and suitable for systems programming, but it shifts cycle-breaking responsibility to the programmer.

== The GC-Compiler Contract

A precise GC requires deep cooperation from the compiler. This section describes the key mechanisms.

=== Safepoints

A *safepoint* is a program point at which the runtime can safely inspect or modify thread state. All GC-managed threads must reach a safepoint before a stop-the-world pause begins.

The compiler inserts safepoint polls at:
- Method entries and return points.
- Loop back-edges (to bound the time before a thread reaches a safepoint).
- Allocation sites (an allocation may itself trigger GC).

The standard implementation uses a *polling page*: the JIT emits a load from a designated memory page. When the GC wants to stop threads, it revokes read access to the page; the resulting page fault (SIGSEGV on Linux, access violation on Windows) is caught by a signal handler that suspends the thread at the safepoint.

=== Stack Maps

At each safepoint, the compiler emits a *stack map* entry. A stack map records, for every slot in the activation frame and every live register:

- Is this slot a GC pointer?
- For derived pointers (a pointer into the middle of an object, e.g., an interior array iterator), what is the base pointer?

The GC reads stack maps to find all roots in JIT-compiled frames. Without stack maps, the GC would have to be conservative for those frames.

=== Derived Pointers

A *derived pointer* points into the interior of an object rather than to its base. Derived pointers are common in array iteration:

```cpp
// Iterator into a managed array — derived pointer pattern
Object* base = array;           // GC root: base of array object
int*    ptr  = array->data();   // derived pointer: interior of array
// If GC runs here and moves 'array', 'ptr' becomes dangling.
// Stack map must record: ptr is derived from base.
```

If the GC moves the base object, it must also update every derived pointer. Stack maps record the (derived, base) pairing so the GC can compute the new derived address as `new_base + (derived - old_base)`.

=== Barrier Elision

Write and read barriers add overhead to every pointer store or load. The compiler can elide barriers when it can prove they are unnecessary:

- *Nursery-allocated objects:* a store into an object allocated in the same nursery minor GC cycle cannot be an old-to-young reference; the card table barrier is unnecessary.
- *Immutable objects:* a store into a provably immutable object cannot affect the GC invariant.
- *Local temporaries:* pointers that never escape to the heap do not need barriers.

Escape analysis, type-based alias analysis, and region-based proofs all contribute to barrier elision.

== A Compact Stop-the-World Mark-Sweep Collector

The following is a self-contained $approx 100$-line C++ stop-the-world mark-sweep collector for a toy VM with tagged word-sized values. It demonstrates root scanning, mark-bit management, and sweep. For simplicity the toy never reuses reclaimed memory --- a real allocator would thread dead cells onto a free list in `sweep()`.

```cpp
#include <cstdint>
#include <cstring>
#include <vector>
#include <cassert>

// Tagged value: low 2 bits encode type.  00=pointer, 01=int, 10=nil
using Val = uintptr_t;
static constexpr uintptr_t TAG_PTR = 0, TAG_INT = 1, TAG_NIL = 2;

struct Cell {
    uint8_t  mark;    // 1 = reachable
    uint8_t  nfields; // number of Val fields following the header
    Val      fields[];
};

// ── Heap ──────────────────────────────────────────────────────────────────
static constexpr size_t HEAP_SIZE = 1u << 20; // 1 MiB
static char   heap_buf[HEAP_SIZE];
static size_t heap_top = 0;
static std::vector<Cell*> all_cells; // every allocated cell

Cell* alloc_cell(uint8_t nfields) {
    size_t bytes = sizeof(Cell) + nfields * sizeof(Val);
    assert(heap_top + bytes <= HEAP_SIZE && "heap exhausted");
    Cell* c = reinterpret_cast<Cell*>(heap_buf + heap_top);
    heap_top += bytes;
    c->mark    = 0;
    c->nfields = nfields;
    all_cells.push_back(c);
    return c;
}

// ── Mark phase ────────────────────────────────────────────────────────────
static void mark(Val v) {
    if ((v & 3) != TAG_PTR) return;           // not a pointer
    Cell* c = reinterpret_cast<Cell*>(v);
    if (c == nullptr || c->mark) return;       // null or already marked
    c->mark = 1;
    for (uint8_t i = 0; i < c->nfields; ++i)
        mark(c->fields[i]);                    // recurse into fields
}

// ── Sweep phase ───────────────────────────────────────────────────────────
// Returns the number of cells reclaimed.
static size_t sweep() {
    size_t reclaimed = 0;
    std::vector<Cell*> survivors;
    for (Cell* c : all_cells) {
        if (c->mark) {
            c->mark = 0;               // reset mark for next cycle
            survivors.push_back(c);
        } else {
            ++reclaimed;               // would free here in a real allocator
        }
    }
    all_cells = std::move(survivors);
    return reclaimed;
}

// ── Public interface ──────────────────────────────────────────────────────
void gc_collect(Val* roots, size_t num_roots) {
    for (size_t i = 0; i < num_roots; ++i) mark(roots[i]);
    sweep();
}
```

This collector is stop-the-world (the VM must halt all threads before calling `gc_collect`), precise (the caller supplies an explicit root set), and non-moving (objects remain at their original addresses, so no forwarding pointers or stack map updates are needed). Sweep is $O(|"heap"|)$ and mark is $O(|"live set"|)$; total pause time is therefore $O(|"heap"|)$.

== Collector Family Comparison

#table(
  columns: 5,
  [*Family*], [*Pause time*], [*Throughput*], [*Space overhead*], [*Implementation complexity*],
  [Stop-the-world mark-sweep], [High (proportional to heap)], [High (no copying)], [Low (~1 bit/object)], [Low],
  [Stop-the-world copying], [High (proportional to live set)], [Very high (bump allocation)], [High (2× live set)], [Low],
  [Generational (minor+major)], [Low for minor, high for major], [Very high], [Moderate (survivor spaces)], [Moderate],
  [Incremental mark-sweep], [Medium (bounded increments)], [Moderate (barrier overhead)], [Low], [Moderate],
  [Concurrent mark-sweep (G1, Go)], [Low (~10--200 ms)], [Moderate], [Moderate (region metadata)], [High],
  [Concurrent moving (ZGC, Shenandoah)], [Sub-millisecond], [Moderate (load barrier)], [Moderate--High], [Very high],
  [Reference counting (ARC)], [None (incremental)], [Moderate (atomic ops)], [Low (count per object)], [Low--Moderate],
)

== Further Reading

Cheney, C. J. (1970). "A Nonrecursive List Compacting Algorithm." _CACM._

Dijkstra, E. W. et al. (1978). "On-the-fly Garbage Collection: An Exercise in Cooperation." _CACM._

Wilson, P. R. (1992). "Uniprocessor Garbage Collection Techniques." _IWMM._

Jones, R., Hosking, A., Moss, E. (2011). _The Garbage Collection Handbook._ Chapman & Hall / CRC.

Lins, R. D. (1992). "Cyclic Reference Counting with Lazy Mark-Scan." _IPL._

Yuasa, T. (1990). "Real-time Garbage Collection on General-purpose Machines." _Systems and Software._

Detlefs, D. et al. (2004). "Garbage-First Garbage Collection." _ISMM._

Lidén, P., Karlsson, S. (2018). "ZGC: A Scalable Low-latency Garbage Collector." OpenJDK / FOSDEM 2018.

= Part III: java.util.concurrent - Building Blocks

== Atomic Variables

=== AtomicInteger, AtomicLong, AtomicReference

*Problem with volatile:*
```java
volatile int count = 0;
count++;  // NOT atomic! (read-modify-write race condition)
```

*Solution: Atomic classes*
```java
AtomicInteger count = new AtomicInteger(0);
count.incrementAndGet();  // Atomic increment, returns new value
count.getAndIncrement();  // Atomic increment, returns old value
count.addAndGet(5);       // Atomic add 5
count.compareAndSet(10, 20);  // CAS: if count == 10, set to 20
```

*Common methods:*
```java
AtomicInteger ai = new AtomicInteger(0);

ai.get();                    // Read current value
ai.set(10);                  // Set value
ai.getAndSet(20);            // Set and return old value

ai.incrementAndGet();        // ++i (atomic)
ai.getAndIncrement();        // i++ (atomic)
ai.decrementAndGet();        // --i (atomic)
ai.getAndDecrement();        // i-- (atomic)

ai.addAndGet(5);             // i += 5
ai.getAndAdd(5);             // returns old, then adds

ai.compareAndSet(10, 20);    // CAS operation
ai.getAndUpdate(x -> x * 2); // Atomic functional update
ai.updateAndGet(x -> x * 2);
```

*AtomicReference for objects:*
```java
class Node {
    int value;
    Node next;
}

AtomicReference<Node> head = new AtomicReference<>();

// Lock-free stack push
void push(int value) {
    Node newNode = new Node(value);
    Node oldHead;
    do {
        oldHead = head.get();
        newNode.next = oldHead;
    } while (!head.compareAndSet(oldHead, newNode));
    // CAS: if head still == oldHead, set to newNode
    // Retry if another thread changed head
}
```

=== Compare-And-Swap (CAS)

*CAS semantics:*
```java
boolean compareAndSet(int expect, int update) {
    // Atomically:
    if (this.value == expect) {
        this.value = update;
        return true;
    } else {
        return false;  // Another thread changed it
    }
}
```

*Hardware support:*
- x86: `CMPXCHG` instruction
- ARM: `LDREX`/`STREX` (load-link/store-conditional)
- Atomic at CPU level (memory controller)

*Lock-free counter:*
```java
AtomicInteger counter = new AtomicInteger(0);

void increment() {
    int oldValue, newValue;
    do {
        oldValue = counter.get();
        newValue = oldValue + 1;
    } while (!counter.compareAndSet(oldValue, newValue));
    // Retry if CAS fails (another thread modified counter)
}
```

*ABA problem:*
```java
// Thread 1: Reads A
head.get();  // A

// Thread 2: Changes A → B → A
head.set(B);
head.set(A);

// Thread 1: CAS succeeds (head == A)
head.compareAndSet(A, C);  // Succeeds! But state changed through B
```

*ABA solution:* Use `AtomicStampedReference` (value + version stamp)
```java
AtomicStampedReference<Node> head = new AtomicStampedReference<>(null, 0);

int[] stampHolder = new int[1];
Node current = head.get(stampHolder);
int stamp = stampHolder[0];

head.compareAndSet(current, newNode, stamp, stamp + 1);
// CAS succeeds only if value AND stamp match
```

=== AtomicFieldUpdater & VarHandle

*AtomicFieldUpdater:* Atomic operations on regular (volatile) fields

```java
class Counter {
    private volatile int count = 0;  // Must be volatile

    private static final AtomicIntegerFieldUpdater<Counter> updater =
        AtomicIntegerFieldUpdater.newUpdater(Counter.class, "count");

    public void increment() {
        updater.incrementAndGet(this);
    }
}

// Why use? Saves memory (no AtomicInteger object overhead per instance)
// When to use? Many instances, few atomic updates
```

*VarHandle (Java 9+):* More powerful, flexible, faster

```java
class Counter {
    private volatile int count = 0;

    private static final VarHandle COUNT;
    static {
        try {
            COUNT = MethodHandles.lookup()
                .findVarHandle(Counter.class, "count", int.class);
        } catch (ReflectiveOperationException e) {
            throw new Error(e);
        }
    }

    public void increment() {
        COUNT.getAndAdd(this, 1);
    }
}
```

*VarHandle benefits:*
- Faster than FieldUpdater
- More operations (plain, opaque, release/acquire, volatile)
- Fine-grained memory ordering control

=== LongAdder vs AtomicLong

*AtomicLong:* Single variable, CAS for updates

```java
AtomicLong counter = new AtomicLong();
counter.incrementAndGet();  // CAS loop
```

*LongAdder:* Multiple cells, reduces contention

```java
LongAdder counter = new LongAdder();
counter.increment();  // Updates one of many cells (less contention)
long sum = counter.sum();  // Sum all cells
```

*How LongAdder works:*
- Maintains array of cells (one per thread, approximately)
- Each thread increments its own cell (no contention)
- `sum()` adds all cells

```java
// Simplified LongAdder
class LongAdder {
    Cell[] cells;  // Array of cells
    long base;     // Base value

    void increment() {
        Cell c = cells[hash(Thread.currentThread())];
        if (c != null) {
            c.value++;  // Update thread-local cell (low contention)
        } else {
            base++;  // Fallback to base
        }
    }

    long sum() {
        long sum = base;
        for (Cell c : cells) {
            sum += c.value;
        }
        return sum;
    }
}
```

*When to use:*
- *AtomicLong*: Need exact value frequently, low contention
- *LongAdder*: High contention, only need sum occasionally (e.g., statistics)

*Benchmark:*
```
Threads    AtomicLong    LongAdder
1          10M ops/s     10M ops/s
2          8M ops/s      18M ops/s  (2x faster)
4          4M ops/s      35M ops/s  (8x faster)
8          2M ops/s      60M ops/s  (30x faster!)
```

=== Interview Questions: Atomics

*Q1: How does CAS work and what are its advantages?*

A: CAS (Compare-And-Swap) is atomic operation: `compareAndSet(expected, new)`

*Semantics:*
```java
// Atomically (hardware-level):
if (value == expected) {
    value = new;
    return true;
} else {
    return false;
}
```

*Implementation:*
- x86: `CMPXCHG` instruction (lock prefix for atomicity)
- ARM: Load-Link/Store-Conditional

*Advantages:*
1. *Lock-free*: No OS-level blocking (no context switches)
2. *Wait-free progress*: At least one thread always makes progress
3. *No deadlock*: No locks to acquire
4. *Low latency*: ~1-10ns vs ~100-1000ns for locks

*Usage pattern:*
```java
AtomicInteger counter = new AtomicInteger(0);

void increment() {
    int oldValue, newValue;
    do {
        oldValue = counter.get();
        newValue = oldValue + 1;
    } while (!counter.compareAndSet(oldValue, newValue));
    // Retry if CAS fails (optimistic concurrency)
}
```

*Trade-off:* CAS can livelock under extreme contention (retry loops). Use locks for very high contention.

*Q2: When would you use LongAdder instead of AtomicLong?*

A:

*AtomicLong:*
- Single variable with CAS updates
- *High contention* → performance degrades (CAS retry loops)
- Use when: Exact value needed frequently, low contention

*LongAdder:*
- Multiple cells (per-thread), reduces contention
- `sum()` adds cells (slower than `get()`)
- Use when: High contention, sum needed occasionally

*Benchmark (8 threads):*
```
AtomicLong: 2M ops/s (contention overhead)
LongAdder:  60M ops/s (30x faster!)
```

*Decision:*
- Metrics/counters (high write, rare read) → LongAdder
- Sequence numbers (frequent read) → AtomicLong
- Low contention → AtomicLong (simpler)

*Q3: What is the ABA problem and how do you solve it?*

A: ABA problem: Value changes A → B → A, CAS succeeds but state changed.

*Example:*
```java
// Lock-free stack
AtomicReference<Node> head = new AtomicReference<>(A);

// Thread 1
Node oldHead = head.get();  // A
// ... compute ...

// Thread 2 (interleaved)
head.set(B);  // A → B
head.set(A);  // B → A (same A node!)

// Thread 1
head.compareAndSet(oldHead, newNode);  // Succeeds! But missed B state
```

*Problem:* Stack went A → B → A, but Thread 1 thinks nothing changed.

*Solutions:*

1. *AtomicStampedReference:* Add version stamp
```java
AtomicStampedReference<Node> head = new AtomicStampedReference<>(A, 0);

int[] stamp = new int[1];
Node old = head.get(stamp);  // Read value + stamp

head.compareAndSet(old, newNode, stamp[0], stamp[0] + 1);
// CAS succeeds only if value AND stamp unchanged
```

2. *AtomicMarkableReference:* Boolean mark (single bit)
```java
AtomicMarkableReference<Node> ref = new AtomicMarkableReference<>(node, false);

boolean[] markHolder = new boolean[1];
Node n = ref.get(markHolder);
ref.compareAndSet(n, newNode, markHolder[0], true);
```

3. *Avoid ABA:* Use immutable data structures (functional approach)

*Q4: Explain AtomicInteger internal implementation.*

A: AtomicInteger uses `Unsafe` class for CAS operations.

*Structure:*
```java
public class AtomicInteger {
    private static final Unsafe unsafe = Unsafe.getUnsafe();
    private static final long valueOffset;

    static {
        // Get memory offset of 'value' field
        valueOffset = unsafe.objectFieldOffset
            (AtomicInteger.class.getDeclaredField("value"));
    }

    private volatile int value;  // Volatile for visibility

    public final boolean compareAndSet(int expect, int update) {
        return unsafe.compareAndSwapInt(this, valueOffset, expect, update);
        // Native method: x86 CMPXCHG instruction
    }

    public final int incrementAndGet() {
        int oldValue, newValue;
        do {
            oldValue = value;
            newValue = oldValue + 1;
        } while (!compareAndSet(oldValue, newValue));
        // Retry until CAS succeeds
        return newValue;
    }
}
```

*Key points:*
1. `volatile int value` → memory visibility
2. `Unsafe.compareAndSwapInt` → hardware CAS (x86: `LOCK CMPXCHG`)
3. Retry loop in `incrementAndGet()` → optimistic concurrency

== Explicit Locks

=== ReentrantLock

*Why ReentrantLock when we have synchronized?*

*synchronized limitations:*
- Can't interrupt thread waiting for lock
- Can't try lock with timeout
- Can't try lock without blocking
- No fair/unfair control

*ReentrantLock benefits:*
- `tryLock()`: Non-blocking attempt
- `tryLock(timeout)`: Timed attempt
- `lockInterruptibly()`: Can be interrupted
- Fairness control
- Multiple `Condition` objects

```java
ReentrantLock lock = new ReentrantLock();

// Basic usage (similar to synchronized)
lock.lock();
try {
    // Critical section
} finally {
    lock.unlock();  // Must unlock in finally!
}

// Try lock without blocking
if (lock.tryLock()) {
    try {
        // Got lock
    } finally {
        lock.unlock();
    }
} else {
    // Couldn't acquire lock
}

// Try lock with timeout
if (lock.tryLock(1, TimeUnit.SECONDS)) {
    try {
        // Got lock within 1 second
    } finally {
        lock.unlock();
    }
} else {
    // Timeout
}

// Interruptible lock
try {
    lock.lockInterruptibly();  // Throws InterruptedException
    try {
        // Critical section
    } finally {
        lock.unlock();
    }
} catch (InterruptedException e) {
    // Thread interrupted while waiting
}
```

*Fairness:*
```java
// Unfair (default): Better throughput, may starve threads
ReentrantLock unfair = new ReentrantLock(false);

// Fair: FIFO order, no starvation, lower throughput
ReentrantLock fair = new ReentrantLock(true);
```

*Fairness tradeoff:*
- Unfair: 10-100x higher throughput (threads can "barge" ahead)
- Fair: Predictable latency (no starvation)

=== ReadWriteLock

*Problem:* Multiple readers OK, but writer needs exclusive access.

```java
// With synchronized: Only one reader at a time (inefficient!)
synchronized (lock) {
    int value = map.get(key);  // Read
}
```

*Solution: ReadWriteLock*
```java
ReadWriteLock rwLock = new ReentrantReadWriteLock();
Lock readLock = rwLock.readLock();
Lock writeLock = rwLock.writeLock();

// Multiple readers allowed simultaneously
readLock.lock();
try {
    int value = map.get(key);  // Read
} finally {
    readLock.unlock();
}

// Writer has exclusive access (no readers, no other writers)
writeLock.lock();
try {
    map.put(key, value);  // Write
} finally {
    writeLock.unlock();
}
```

*Lock states:*
- No lock: Readers and writers can acquire
- Read lock held: More readers can acquire, writers blocked
- Write lock held: All readers and writers blocked

*Use case:* Read-heavy workloads (cache, configuration)

*Example: Read-heavy cache*
```java
class Cache {
    private final Map<String, Object> map = new HashMap<>();
    private final ReadWriteLock rwLock = new ReentrantReadWriteLock();

    public Object get(String key) {
        rwLock.readLock().lock();
        try {
            return map.get(key);  // Many readers simultaneously
        } finally {
            rwLock.readLock().unlock();
        }
    }

    public void put(String key, Object value) {
        rwLock.writeLock().lock();
        try {
            map.put(key, value);  // Exclusive access
        } finally {
            rwLock.writeLock().unlock();
        }
    }
}
```

=== StampedLock

*Problem with ReadWriteLock:* Writers can starve if readers keep coming.

*StampedLock (Java 8+):* Three modes

1. *Write lock:* Exclusive (like writeLock)
2. *Read lock:* Shared (like readLock)
3. *Optimistic read:* No lock! Just check version stamp

```java
StampedLock sl = new StampedLock();

// Optimistic read (fastest, no lock!)
long stamp = sl.tryOptimisticRead();  // Get current stamp
int value = this.value;  // Read data
if (sl.validate(stamp)) {  // Check if no write occurred
    // Success: Use value (no lock held!)
} else {
    // Failed: Acquire read lock
    stamp = sl.readLock();
    try {
        value = this.value;
    } finally {
        sl.unlockRead(stamp);
    }
}

// Read lock
long stamp = sl.readLock();
try {
    // Read data
} finally {
    sl.unlockRead(stamp);
}

// Write lock
long stamp = sl.writeLock();
try {
    // Write data
} finally {
    sl.unlockWrite(stamp);
}
```

*Optimistic read pattern:*
```java
class Point {
    private int x, y;
    private final StampedLock sl = new StampedLock();

    public double distanceFromOrigin() {
        long stamp = sl.tryOptimisticRead();  // No lock
        int currentX = x;  // Read
        int currentY = y;  // Read
        if (!sl.validate(stamp)) {  // Validate (check no write)
            stamp = sl.readLock();  // Fallback to read lock
            try {
                currentX = x;
                currentY = y;
            } finally {
                sl.unlockRead(stamp);
            }
        }
        return Math.sqrt(currentX * currentX + currentY * currentY);
    }

    public void move(int deltaX, int deltaY) {
        long stamp = sl.writeLock();
        try {
            x += deltaX;
            y += deltaY;
        } finally {
            sl.unlockWrite(stamp);
        }
    }
}
```

*When to use:*
- Read-heavy workloads (90%+ reads)
- Reads very short (optimistic read overhead matters)

*Warning:* StampedLock NOT reentrant (unlike ReentrantReadWriteLock)!

=== Condition Variables

*Problem with wait/notify:* Only one condition per object.

*Solution: Multiple Condition objects*

```java
Lock lock = new ReentrantLock();
Condition notFull = lock.newCondition();
Condition notEmpty = lock.newCondition();

// Producer
lock.lock();
try {
    while (isFull()) {
        notFull.await();  // Wait for not full
    }
    // Add item
    notEmpty.signal();  // Signal not empty
} finally {
    lock.unlock();
}

// Consumer
lock.lock();
try {
    while (isEmpty()) {
        notEmpty.await();  // Wait for not empty
    }
    // Remove item
    notFull.signal();  // Signal not full
} finally {
    lock.unlock();
}
```

*Bounded buffer with Condition:*
```java
class BoundedBuffer<T> {
    private final T[] buffer;
    private int putIndex, takeIndex, count;
    private final Lock lock = new ReentrantLock();
    private final Condition notFull = lock.newCondition();
    private final Condition notEmpty = lock.newCondition();

    public void put(T item) throws InterruptedException {
        lock.lock();
        try {
            while (count == buffer.length) {
                notFull.await();  // Wait for space
            }
            buffer[putIndex] = item;
            putIndex = (putIndex + 1) % buffer.length;
            count++;
            notEmpty.signal();  // Wake one consumer
        } finally {
            lock.unlock();
        }
    }

    public T take() throws InterruptedException {
        lock.lock();
        try {
            while (count == 0) {
                notEmpty.await();  // Wait for item
            }
            T item = buffer[takeIndex];
            takeIndex = (takeIndex + 1) % buffer.length;
            count--;
            notFull.signal();  // Wake one producer
            return item;
        } finally {
            lock.unlock();
        }
    }
}
```

=== Interview Questions: Locks

*Q1: ReentrantLock vs synchronized - when to use which?*

A:

*synchronized:*
- Simpler syntax (no finally needed)
- JVM optimizations (biased locking)
- Automatic release (even if exception)
- Cannot interrupt waiting thread
- No tryLock without blocking

*ReentrantLock:*
- Advanced features:
  - `tryLock()`: Non-blocking attempt
  - `tryLock(timeout)`: Timed wait
  - `lockInterruptibly()`: Can interrupt
  - Fairness control
  - Multiple conditions
- Manually unlock (must use finally!)

*When to use:*
- *synchronized*: Default choice (99% of cases)
- *ReentrantLock*: When you need advanced features
  - Timeout on lock acquisition
  - Try lock without blocking
  - Fair lock (prevent starvation)
  - Multiple condition variables

*Performance:* Similar (synchronized slightly faster in Java 6+)

*Q2: What is a fair lock and when would you use it?*

A: Fair lock: Threads acquire lock in FIFO order (request order).

```java
// Unfair (default)
ReentrantLock unfair = new ReentrantLock(false);
// Thread can "barge" ahead (better throughput)

// Fair
ReentrantLock fair = new ReentrantLock(true);
// FIFO order (predictable latency, no starvation)
```

*Trade-off:*
- *Unfair*: 10-100x higher throughput (less context switches)
- *Fair*: Bounded wait time (no starvation)

*When to use fair:*
- Starvation is unacceptable
- Predictable latency required (real-time systems)
- Example: Order processing (FIFO order matters)

*Default:* Unfair (better performance)

*Q3: Explain optimistic locking in StampedLock.*

A: Optimistic locking: Assume no conflicts, validate later.

*Pattern:*
1. Read stamp (version)
2. Read data (NO lock held!)
3. Validate stamp (check no write happened)
4. If valid: Success (fast path)
5. If invalid: Acquire read lock (slow path)

```java
StampedLock sl = new StampedLock();

long stamp = sl.tryOptimisticRead();  // Get stamp (no lock)
int value = this.value;  // Read (no lock!)
if (sl.validate(stamp)) {  // Validate
    return value;  // Fast path (no lock held!)
} else {
    // Slow path: Acquire read lock
    stamp = sl.readLock();
    try {
        return this.value;
    } finally {
        sl.unlockRead(stamp);
    }
}
```

*When to use:*
- Read-heavy (90%+ reads)
- Short reads (optimistic overhead small)
- Low write contention (validations usually succeed)

*Benefit:* Readers don't block writers (no lock held during read)

*Downside:* Retry overhead if validation fails

== Concurrent Collections

=== ConcurrentHashMap

*Evolution:*

*Java 7:* Segmented locking
- 16 segments (default)
- Lock per segment (fine-grained)
- Throughput: O(1) with low contention

*Java 8+:* CAS + synchronized
- No segments (buckets like HashMap)
- CAS for single-element buckets
- synchronized for collisions (tree nodes)
- Lock-free reads

*Internal structure (Java 8+):*
```java
// Simplified ConcurrentHashMap
class ConcurrentHashMap<K, V> {
    Node<K,V>[] table;  // Array of buckets

    static class Node<K,V> {
        final int hash;
        final K key;
        volatile V value;  // Volatile for visibility
        volatile Node<K,V> next;  // Volatile
    }

    public V get(Object key) {
        // Lock-free read
        Node<K,V> node = table[hash(key) & (table.length - 1)];
        while (node != null) {
            if (node.key.equals(key)) {
                return node.value;  // Volatile read
            }
            node = node.next;
        }
        return null;
    }

    public V put(K key, V value) {
        int hash = hash(key);
        int bucket = hash & (table.length - 1);

        if (table[bucket] == null) {
            // CAS to add first node (lock-free)
            if (casTabAt(table, bucket, null, newNode)) {
                return null;
            }
        }

        // Collision: Lock bucket
        synchronized (table[bucket]) {
            // Insert in linked list or tree
        }
    }
}
```

*Compute methods (Java 8+):*
```java
ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();

// Atomic compute: mapping = function(key, oldValue)
map.compute("key", (k, oldV) -> oldV == null ? 1 : oldV + 1);

// Atomic computeIfAbsent: add if missing
map.computeIfAbsent("key", k -> expensiveComputation(k));

// Atomic computeIfPresent: update if exists
map.computeIfPresent("key", (k, oldV) -> oldV + 1);

// Atomic merge: combine old and new
map.merge("key", 1, (oldV, newV) -> oldV + newV);  // Increment by 1
```

*Why compute methods:*
```java
// Wrong: Race condition
if (!map.containsKey(key)) {
    map.put(key, value);  // Another thread may have put between check and put!
}

// Right: Atomic
map.computeIfAbsent(key, k -> value);
```

=== CopyOnWriteArrayList

*Strategy:* Every modification creates new array copy.

```java
public class CopyOnWriteArrayList<E> {
    private volatile Object[] array;  // Volatile array reference

    public boolean add(E e) {
        synchronized (lock) {
            Object[] oldArray = array;
            int len = oldArray.length;
            Object[] newArray = Arrays.copyOf(oldArray, len + 1);  // Copy!
            newArray[len] = e;
            array = newArray;  // Atomic switch
        }
        return true;
    }

    public E get(int index) {
        return array[index];  // No lock! Read from current snapshot
    }
}
```

*Iterator:* Snapshot at creation, never throws ConcurrentModificationException

```java
CopyOnWriteArrayList<String> list = new CopyOnWriteArrayList<>();
list.add("a");
list.add("b");

Iterator<String> it = list.iterator();  // Snapshot of current state

// Another thread modifies list
list.add("c");

// Iterator sees old snapshot
while (it.hasNext()) {
    System.out.println(it.next());  // Prints: a, b (not c)
}
```

*When to use:*
- Read-heavy (99%+ reads)
- Small lists (copy overhead acceptable)
- Iteration >> modification

*Anti-pattern:* Frequent modifications (O(n) copy for each!)

=== ConcurrentLinkedQueue / ConcurrentLinkedDeque

*Lock-free queue:* Based on Michael-Scott algorithm (CAS-based)

```java
ConcurrentLinkedQueue<String> queue = new ConcurrentLinkedQueue<>();

queue.offer("a");  // Lock-free enqueue
String item = queue.poll();  // Lock-free dequeue

// Safe for multiple producers and consumers
```

*Weakly consistent iterator:* May not reflect recent modifications

```java
queue.offer("a");
Iterator<String> it = queue.iterator();
queue.offer("b");  // May or may not appear in iterator
```

=== BlockingQueue Implementations

*BlockingQueue:* Queue with blocking operations

```java
BlockingQueue<String> queue = new ArrayBlockingQueue<>(10);

// Producer
queue.put("item");  // Blocks if full

// Consumer
String item = queue.take();  // Blocks if empty

// Timeout versions
boolean added = queue.offer("item", 1, TimeUnit.SECONDS);
String item = queue.poll(1, TimeUnit.SECONDS);
```

*Implementations:*

*1. ArrayBlockingQueue:*
- Bounded, array-backed
- FIFO order
- Single lock (putLock + takeLock in LinkedBlockingQueue)

*2. LinkedBlockingQueue:*
- Optionally bounded, linked nodes
- Two locks (putLock, takeLock) → better concurrency
- FIFO order

*3. PriorityBlockingQueue:*
- Unbounded, heap-backed
- Priority order (not FIFO)
- Lock for modifications

*4. SynchronousQueue:*
- No capacity (rendezvous)
- Producer blocks until consumer takes (and vice versa)

*5. DelayQueue:*
- Unbounded, elements delayed
- Element available after delay expires

*Producer-Consumer with BlockingQueue:*
```java
BlockingQueue<Task> queue = new ArrayBlockingQueue<>(100);

// Producer thread
void produce() {
    while (running) {
        Task task = createTask();
        queue.put(task);  // Blocks if queue full
    }
}

// Consumer thread
void consume() {
    while (running) {
        Task task = queue.take();  // Blocks if queue empty
        process(task);
    }
}
```

=== Interview Questions: Concurrent Collections

*Q1: How does ConcurrentHashMap achieve thread-safety without locking entire map?*

A:

*Java 7:* Segmented locking
- 16 segments (default), each with own lock
- Lock only one segment at a time
- Concurrency level = 16 readers + 16 writers

*Java 8+:* CAS + fine-grained locking
- No segments (buckets like HashMap)
- *Reads*: Lock-free (volatile reads)
- *Writes*:
  - Empty bucket → CAS (lock-free)
  - Collision → Lock only that bucket (synchronized)

*Benefit:*
- Reads never block
- Writes lock only one bucket
- High concurrency (thousands of threads)

*Compare to Hashtable:*
```java
// Hashtable: Lock entire map
synchronized (map) {
    map.get(key);  // Blocks all other threads!
}

// ConcurrentHashMap: Lock-free reads
map.get(key);  // Never blocks
```

*Q2: When would you use CopyOnWriteArrayList?*

A: Use when:
1. *Read-heavy*: 99%+ reads, rare writes
2. *Small list*: Copy overhead acceptable
3. *Iteration >> modification*: Frequently iterate, rarely modify

*Example:* Event listeners
```java
CopyOnWriteArrayList<Listener> listeners = new CopyOnWriteArrayList<>();

// Rare: Add listener
listeners.add(listener);  // O(n) copy (rare, acceptable)

// Frequent: Notify all
for (Listener l : listeners) {  // Lock-free, no ConcurrentModificationException
    l.onEvent(event);
}
```

*Anti-pattern:* Frequent modifications
```java
// BAD: O(n²) for n additions!
for (int i = 0; i < 1000; i++) {
    list.add(i);  // Each add copies entire array!
}
```

*Alternative:* Use `ConcurrentHashMap.newKeySet()` for thread-safe set with better write performance.

*Q3: What's the difference between ArrayBlockingQueue and LinkedBlockingQueue?*

A:

*ArrayBlockingQueue:*
- Fixed capacity (bounded)
- Array-backed (contiguous memory)
- Single lock (both put and take)
- Better cache locality

*LinkedBlockingQueue:*
- Optional capacity (unbounded by default)
- Linked nodes (scattered memory)
- Two locks (putLock + takeLock) → better concurrency
- Higher throughput (producers and consumers don't block each other)

```java
ArrayBlockingQueue<String> abq = new ArrayBlockingQueue<>(100);
// put() and take() share same lock

LinkedBlockingQueue<String> lbq = new LinkedBlockingQueue<>();
// put() and take() use separate locks → higher concurrency
```

*When to use:*
- *ArrayBlockingQueue*: Bounded size required, lower memory overhead
- *LinkedBlockingQueue*: High throughput, unbounded OK

*Performance:*
- ArrayBlockingQueue: Better for low contention
- LinkedBlockingQueue: Better for high contention (separate locks)

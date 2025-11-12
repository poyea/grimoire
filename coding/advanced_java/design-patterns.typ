= Part VIII: Design Patterns for Concurrency

== Thread-Safe Singleton

=== Double-Checked Locking

*Problem: Lazy initialization thread-safe*

*Naive (broken):*
```java
class Singleton {
    private static Singleton instance;

    public static Singleton getInstance() {
        if (instance == null) {  // Check 1 (not synchronized)
            synchronized (Singleton.class) {
                if (instance == null) {  // Check 2 (synchronized)
                    instance = new Singleton();  // Problem: Not atomic!
                }
            }
        }
        return instance;
    }
}

// Why broken:
// instance = new Singleton() is 3 steps:
// 1. Allocate memory
// 2. Initialize object
// 3. Assign to instance
// Can reorder to: 1, 3, 2 → another thread sees non-null but uninitialized!
```

*Correct (with volatile):*
```java
class Singleton {
    private static volatile Singleton instance;  // volatile prevents reordering!

    public static Singleton getInstance() {
        if (instance == null) {  // Check 1 (fast path, no lock)
            synchronized (Singleton.class) {
                if (instance == null) {  // Check 2 (prevent double init)
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}

// volatile ensures:
// 1. Happens-before: initialization complete before assignment
// 2. Visibility: Other threads see fully constructed instance
```

*Why two checks:*
- First check: Fast path (no synchronization if already initialized)
- Second check: Prevent double initialization (two threads pass first check)

=== Initialization-on-Demand Holder

*Better: No synchronization overhead*

```java
class Singleton {
    private Singleton() { }

    // Holder class (loaded lazily)
    private static class Holder {
        static final Singleton INSTANCE = new Singleton();
        // Class initialization is thread-safe (JVM guarantees)
    }

    public static Singleton getInstance() {
        return Holder.INSTANCE;  // Triggers Holder class loading (once)
    }
}

// Why thread-safe:
// JVM guarantees class initialization is:
// 1. Thread-safe (only one thread initializes)
// 2. Lazy (Holder loaded when first accessed)
// 3. No synchronization overhead after init
```

*Benefits:*
- Thread-safe (JVM guarantee)
- Lazy (loaded on first use)
- No synchronization overhead
- Simple, clear

*Preferred approach in most cases.*

=== Enum Singleton

*Simplest: Enum (Joshua Bloch recommendation)*

```java
public enum Singleton {
    INSTANCE;

    public void doSomething() {
        // Business logic
    }
}

// Usage
Singleton.INSTANCE.doSomething();

// Why enum:
// 1. Thread-safe (JVM guarantee)
// 2. Serialization-safe (prevents multiple instances)
// 3. Reflection-proof (can't instantiate enum)
// 4. Simplest code
```

*Benefits:*
- Thread-safe by design
- Prevents reflection attacks
- Handles serialization correctly
- Most concise

*Drawback:* Not lazy (created at class load)

=== Interview Questions: Singleton

*Q1: Implement thread-safe singleton. Explain different approaches.*

A:

*1. Enum (best):*
```java
public enum Singleton {
    INSTANCE;
}
// Thread-safe, serialization-safe, simple
```

*2. Holder pattern (lazy):*
```java
class Singleton {
    private static class Holder {
        static final Singleton INSTANCE = new Singleton();
    }
    public static Singleton getInstance() {
        return Holder.INSTANCE;
    }
}
// Thread-safe (JVM guarantee), lazy, no sync overhead
```

*3. Double-checked locking:*
```java
class Singleton {
    private static volatile Singleton instance;
    public static Singleton getInstance() {
        if (instance == null) {
            synchronized (Singleton.class) {
                if (instance == null) {
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
// Thread-safe (volatile), lazy, fast after init
```

*Comparison:*
- Enum: Simplest, not lazy
- Holder: Lazy, no overhead
- DCL: Lazy, requires volatile

*Q2: Why doesn't double-checked locking work without volatile?*

A: Without volatile, can see partially constructed object.

*Problem:*
```java
instance = new Singleton();
// Not atomic! Compiles to:
// 1. memory = allocate()
// 2. instance = memory  (assign before init!)
// 3. init(memory)
// Thread 2 sees instance != null but object not initialized!
```

*With volatile:*
- Prevents reordering (happens-before guarantee)
- Ensures initialization complete before assignment
- Other threads see fully constructed object

*Happens-before:*
```java
// Thread 1
instance = new Singleton();  // volatile write
// Initialization happens-before assignment

// Thread 2
if (instance != null) {  // volatile read
    // Sees fully initialized instance
}
```

== Producer-Consumer

=== BlockingQueue Implementation

*Modern approach: Use BlockingQueue*

```java
class ProducerConsumer {
    private final BlockingQueue<Task> queue = new ArrayBlockingQueue<>(100);

    // Producer thread
    void produce() throws InterruptedException {
        while (running) {
            Task task = createTask();
            queue.put(task);  // Blocks if queue full
        }
    }

    // Consumer thread
    void consume() throws InterruptedException {
        while (running) {
            Task task = queue.take();  // Blocks if queue empty
            process(task);
        }
    }

    // Start producer and consumer threads
    void start() {
        new Thread(this::produce).start();
        new Thread(this::consume).start();
    }
}

// BlockingQueue handles:
// - Thread safety
// - Blocking (put/take)
// - No busy-waiting
```

*Benefits:*
- Simple (no manual synchronization)
- Correct (battle-tested)
- Efficient (no busy-waiting)

=== Wait/Notify Implementation

*Classic approach (for understanding):*

```java
class BoundedBuffer<T> {
    private final Queue<T> queue = new LinkedList<>();
    private final int capacity;
    private final Object lock = new Object();

    public BoundedBuffer(int capacity) {
        this.capacity = capacity;
    }

    public void put(T item) throws InterruptedException {
        synchronized (lock) {
            while (queue.size() == capacity) {
                lock.wait();  // Full: Wait for space
            }
            queue.add(item);
            lock.notifyAll();  // Wake consumers
        }
    }

    public T take() throws InterruptedException {
        synchronized (lock) {
            while (queue.isEmpty()) {
                lock.wait();  // Empty: Wait for item
            }
            T item = queue.remove();
            lock.notifyAll();  // Wake producers
            return item;
        }
    }
}
```

*Key points:*
- Use `while` (not `if`) for wait (spurious wakeups)
- Use `notifyAll()` (not `notify()`) to avoid missed signals
- Always synchronize on same object

*Modern code: Prefer BlockingQueue*

=== Interview Questions: Producer-Consumer

*Q1: Implement producer-consumer with wait/notify.*

A:

```java
class Buffer<T> {
    private final Queue<T> queue = new LinkedList<>();
    private final int capacity;
    private final Object lock = new Object();

    public void put(T item) throws InterruptedException {
        synchronized (lock) {
            while (queue.size() == capacity) {  // while, not if!
                lock.wait();  // Wait for space
            }
            queue.add(item);
            lock.notifyAll();  // Wake consumers
        }
    }

    public T take() throws InterruptedException {
        synchronized (lock) {
            while (queue.isEmpty()) {  // while, not if!
                lock.wait();  // Wait for data
            }
            T item = queue.remove();
            lock.notifyAll();  // Wake producers
            return item;
        }
    }
}
```

*Why while (not if):*
1. Spurious wakeups (thread wakes without notify)
2. Multiple consumers (one consumes item before another wakes)
3. Condition may change (another thread modified queue)

*Why notifyAll (not notify):*
- notify() wakes one thread (may wake wrong one!)
- Producer and consumer both waiting → may wake producer instead of consumer

*Q2: What are the advantages of using BlockingQueue?*

A:

*BlockingQueue advantages:*

*1. Simplicity:*
```java
// Manual: 20+ lines (synchronized, wait, notify)
// BlockingQueue: 2 lines
queue.put(item);
queue.take();
```

*2. Correctness:*
- No spurious wakeups to handle
- No missed signals
- No deadlocks
- Battle-tested implementation

*3. Features:*
- Timeout: `poll(1, TimeUnit.SECONDS)`
- Non-blocking: `offer(item)` (returns false if full)
- Multiple implementations: ArrayBlockingQueue, LinkedBlockingQueue, etc.

*4. Performance:*
- Optimized (e.g., LinkedBlockingQueue has separate put/take locks)
- No busy-waiting

*Modern code: Always prefer BlockingQueue*

== Immutability Patterns

=== Effective Immutability

*Immutable class:*

```java
public final class ImmutablePoint {
    private final int x;
    private final int y;

    public ImmutablePoint(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public int getX() { return x; }
    public int getY() { return y; }

    // New instance (not mutation)
    public ImmutablePoint move(int dx, int dy) {
        return new ImmutablePoint(x + dx, y + dy);
    }
}

// Thread-safe by design:
// - No mutable state
// - No synchronization needed
// - Safe to share across threads
```

*Rules for immutability:*
1. Class is `final` (can't subclass)
2. All fields are `final`
3. All fields are `private`
4. No setters
5. Deep immutability (fields are immutable or defensive copies)

*Defensive copies for mutable fields:*
```java
public final class ImmutableContainer {
    private final Date date;  // Date is mutable!

    public ImmutableContainer(Date date) {
        this.date = new Date(date.getTime());  // Defensive copy
    }

    public Date getDate() {
        return new Date(date.getTime());  // Defensive copy
    }
}
```

*Better: Use immutable types*
```java
public final class ImmutableContainer {
    private final Instant instant;  // Instant is immutable

    public ImmutableContainer(Instant instant) {
        this.instant = instant;  // No copy needed
    }

    public Instant getInstant() {
        return instant;  // Safe to return (immutable)
    }
}
```

=== Builder Pattern for Immutable Objects

*Problem:* Many constructor parameters

```java
// Bad: Telescoping constructors
public ImmutableUser(String name, int age, String email, String phone, ...) {
    // 10+ parameters!
}
```

*Solution: Builder*

```java
public final class ImmutableUser {
    private final String name;
    private final int age;
    private final String email;
    private final String phone;

    private ImmutableUser(Builder builder) {
        this.name = builder.name;
        this.age = builder.age;
        this.email = builder.email;
        this.phone = builder.phone;
    }

    public static class Builder {
        private String name;
        private int age;
        private String email;
        private String phone;

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder age(int age) {
            this.age = age;
            return this;
        }

        public Builder email(String email) {
            this.email = email;
            return this;
        }

        public Builder phone(String phone) {
            this.phone = phone;
            return this;
        }

        public ImmutableUser build() {
            // Validation
            if (name == null || name.isEmpty()) {
                throw new IllegalStateException("Name required");
            }
            return new ImmutableUser(this);
        }
    }

    // Getters
    public String getName() { return name; }
    public int getAge() { return age; }
    public String getEmail() { return email; }
    public String getPhone() { return phone; }
}

// Usage
ImmutableUser user = new ImmutableUser.Builder()
    .name("Alice")
    .age(30)
    .email("alice@example.com")
    .phone("123-456-7890")
    .build();
```

*Benefits:*
- Readable (named parameters)
- Optional parameters (omit if not needed)
- Validation in `build()`
- Still immutable

*Modern alternative: Records + `@Builder` (Lombok)*
```java
@Builder
public record User(String name, int age, String email, String phone) { }
```

=== Interview Questions: Immutability

*Q1: How do you make a class immutable?*

A:

*Rules:*

*1. Class final:*
```java
public final class Immutable { }  // Can't subclass
```

*2. Fields final:*
```java
private final int value;  // Can't reassign
```

*3. No setters:*
```java
// No setAge(), setName(), etc.
```

*4. Defensive copies for mutable fields:*
```java
public final class Container {
    private final Date date;

    public Container(Date date) {
        this.date = new Date(date.getTime());  // Copy
    }

    public Date getDate() {
        return new Date(date.getTime());  // Copy
    }
}
```

*5. Deep immutability:*
- All fields immutable
- Or defensive copies

*Better: Use immutable types (String, Instant, BigDecimal, etc.)*

*Benefits for concurrency:*
- Thread-safe (no synchronization needed)
- No race conditions (can't change)
- Safe to share (no defensive copies)

*Q2: What are the benefits of immutability for concurrency?*

A:

*Benefits:*

*1. Thread-safe by design:*
```java
// Immutable: No synchronization needed
ImmutablePoint p = new ImmutablePoint(10, 20);
// Safe to share across threads (no mutations)

// Mutable: Needs synchronization
MutablePoint p = new MutablePoint(10, 20);
synchronized (p) {  // Must synchronize!
    p.setX(30);
}
```

*2. No race conditions:*
- Can't change state → no race to modify

*3. No defensive copies:*
```java
// Immutable: Safe to return
public ImmutablePoint getPoint() {
    return this.point;  // No copy needed
}

// Mutable: Must copy
public MutablePoint getPoint() {
    return new MutablePoint(this.point);  // Defensive copy
}
```

*4. Safe publication:*
```java
// final field + immutable object = safe publication
private final ImmutableUser user = new ImmutableUser(...);
// Other threads see fully initialized user (no volatile needed)
```

*5. Simplicity:*
- No synchronization complexity
- Easier to reason about
- Fewer bugs

*Trade-off:* More garbage (create new instead of mutate)
- Usually acceptable (GC is fast)
- Use mutable if ultra-high performance needed

== Thread-Safe Lazy Initialization

=== Various Approaches

*1. Eager initialization (simplest):*
```java
class Resource {
    private static final ExpensiveObject instance = new ExpensiveObject();

    public static ExpensiveObject getInstance() {
        return instance;  // Already initialized
    }
}
// Pro: Thread-safe (static init), simple
// Con: Not lazy (created at class load)
```

*2. Synchronized method:*
```java
class Resource {
    private static ExpensiveObject instance;

    public static synchronized ExpensiveObject getInstance() {
        if (instance == null) {
            instance = new ExpensiveObject();
        }
        return instance;
    }
}
// Pro: Thread-safe, lazy
// Con: Synchronized every call (slow after init)
```

*3. Double-checked locking:*
```java
class Resource {
    private static volatile ExpensiveObject instance;

    public static ExpensiveObject getInstance() {
        if (instance == null) {  // Check 1 (no lock)
            synchronized (Resource.class) {
                if (instance == null) {  // Check 2 (with lock)
                    instance = new ExpensiveObject();
                }
            }
        }
        return instance;
    }
}
// Pro: Thread-safe, lazy, fast after init
// Con: Requires volatile (tricky)
```

*4. Holder pattern (best):*
```java
class Resource {
    private static class Holder {
        static final ExpensiveObject instance = new ExpensiveObject();
    }

    public static ExpensiveObject getInstance() {
        return Holder.instance;
    }
}
// Pro: Thread-safe (JVM guarantee), lazy, simple, no overhead
// Con: None
```

*Recommendation: Use holder pattern*

=== Interview Questions: Lazy Init

*Q1: Implement lazy initialization safely. What are the different approaches?*

A: See "Various Approaches" section above.

*Summary:*
1. Eager: Simple, not lazy
2. Synchronized: Safe, slow (sync every call)
3. DCL: Safe, fast, complex (volatile)
4. Holder: Safe, fast, simple ✓ (best)

*Q2: What are the trade-offs between different approaches?*

A:

*Comparison:*

```
Approach           Thread-Safe  Lazy  Fast After Init  Complexity
Eager              Yes          No    Yes             Low
Synchronized       Yes          Yes   No              Low
DCL                Yes          Yes   Yes             High
Holder             Yes          Yes   Yes             Low
```

*Trade-offs:*

*Eager:*
- Pro: Simple, fast
- Con: Not lazy (may waste memory)

*Synchronized:*
- Pro: Simple, safe
- Con: Slow (synchronized on every call)

*DCL:*
- Pro: Fast (no sync after init)
- Con: Complex (volatile required, easy to get wrong)

*Holder:*
- Pro: Simple, fast, safe
- Con: None (use this!)

*Decision:*
- Default: Holder pattern
- If not lazy OK: Eager
- Never: Synchronized (too slow) or DCL (too complex)

== Summary

*Key patterns for concurrent Java:*

*1. Singleton:*
- Enum (simplest)
- Holder pattern (lazy)
- DCL (complex, avoid)

*2. Producer-Consumer:*
- Use BlockingQueue (modern)
- Manual wait/notify (for understanding)

*3. Immutability:*
- Final class + final fields
- No setters
- Defensive copies for mutable fields
- Builder for many parameters

*4. Lazy initialization:*
- Holder pattern (best)
- DCL (complex, avoid)
- Eager (if lazy not needed)

*General principles:*
- Prefer immutability (thread-safe by design)
- Use java.util.concurrent (don't reinvent)
- Minimize shared mutable state
- Document thread-safety guarantees

*Interview tips:*
- Explain trade-offs (performance, complexity, correctness)
- Know multiple approaches (show depth)
- Prefer simple, correct solutions (not clever)
- Mention modern alternatives (BlockingQueue, atomic classes)

= Part II: Concurrency Primitives

== Synchronization Mechanisms

=== synchronized Keyword

*Two forms:*

1. *Synchronized method:*
```java
public synchronized void method() {
    // Only one thread can execute at a time
    // Locks on 'this' object
}

public static synchronized void staticMethod() {
    // Locks on Class object
}
```

2. *Synchronized block:*
```java
public void method() {
    synchronized (lock) {
        // Only one thread can execute at a time
        // Locks on specific object
    }
}
```

*Semantics:*
- *Mutual exclusion*: Only one thread can hold lock at a time
- *Memory visibility*: Unlock happens-before subsequent lock (changes visible)
- *Reentrant*: Thread can reacquire lock it already holds

```java
synchronized void outer() {
    synchronized void inner() {
        // Same thread can enter (reentrant)
    }
    inner();  // OK (doesn't deadlock)
}
```

=== Lock Implementation (Biased, Thin, Fat)

*Three optimization levels:*

*1. Biased locking:* (Fast path, no contention)
- Assume lock owned by single thread
- Mark object header with thread ID
- Entry/exit: Just check thread ID (no atomic ops)
- 10x faster than thin lock

```java
// Thread T1 repeatedly acquires lock
for (int i = 0; i < 1000; i++) {
    synchronized (obj) {  // Biased to T1 after first acquisition
        // No CAS needed! Just check thread ID
    }
}
```

*2. Thin locking:* (Light contention)
- Use CAS (Compare-And-Swap) to acquire lock
- Store lock record in thread's stack frame
- No OS-level blocking (spin briefly)

*3. Fat locking:* (Heavy contention)
- Inflates to OS-level mutex
- Blocking threads park (OS scheduler manages)
- Expensive: Context switch overhead

*Transition:*
```
No lock → Biased lock (first thread)
       → Thin lock (second thread, low contention)
       → Fat lock (multiple threads, high contention)
```

*Object header (Mark Word):*
```
64-bit object header layout:

Unlocked:
| unused (25 bits) | hash (31 bits) | unused (1) | age (4) | biased (1) | lock (2) |

Biased:
| thread ID (54 bits) | epoch (2) | unused (1) | age (4) | biased (1) | lock (2) |

Thin locked:
| ptr to lock record in thread stack (62 bits) | lock (2) |

Fat locked:
| ptr to monitor (62 bits) | lock (2) |
```

=== volatile Keyword

*Guarantees:*
1. *Visibility*: Writes immediately visible to other threads
2. *Ordering*: Memory barrier (prevents reordering)

*NOT atomic for compound operations:*
```java
volatile int count = 0;

count++;  // NOT atomic!
// Expands to:
// 1. Read count
// 2. Add 1
// 3. Write count
// Race condition!

// Thread 1: read 0, add 1, write 1
// Thread 2: read 0, add 1, write 1
// Result: 1 (not 2!)
```

*When to use volatile:*
1. Single writer, multiple readers
2. Boolean flags
3. Double-checked locking (with additional guarantees)

```java
// Good: Flag (single write, multiple reads)
private volatile boolean shutdownRequested = false;

public void shutdown() {
    shutdownRequested = true;  // Write
}

public void run() {
    while (!shutdownRequested) {  // Read
        // Work
    }
}
```

*Memory barriers:*
- *Write barrier*: Before volatile write, flush all prior writes to memory
- *Read barrier*: After volatile read, invalidate cache, read from memory

```java
int a = 0;
volatile boolean flag = false;

// Thread 1
a = 42;          // (1)
flag = true;     // (2) Volatile write barrier
                 // All writes before (2) visible after (2)

// Thread 2
if (flag) {      // (3) Volatile read barrier
                 // All writes before (2) visible here
    int b = a;   // (4) Sees a = 42
}
```

=== final Semantics

*Guarantee:* Object properly constructed before reference visible to other threads.

```java
class Immutable {
    private final int value;

    public Immutable(int value) {
        this.value = value;  // Final field write
    }  // Memory barrier: All final writes visible before reference escape
}

// Thread 1
Immutable obj = new Immutable(42);
global = obj;  // Safe publication (final field guarantee)

// Thread 2
Immutable o = global;
if (o != null) {
    int v = o.value;  // Always sees 42 (not 0 or garbage)
}
```

*Without final:*
```java
class Mutable {
    private int value;  // NOT final

    public Mutable(int value) {
        this.value = value;
    }
}

// Thread 1
global = new Mutable(42);  // Unsafe! May publish before value set

// Thread 2
Mutable o = global;
if (o != null) {
    int v = o.value;  // May see 0! (partially constructed)
}
```

*Final reference (not deep immutability):*
```java
class Holder {
    private final int[] array;  // Final reference

    public Holder(int[] array) {
        this.array = array;  // Array reference is final
    }
}

Holder h = new Holder(new int[]{1, 2, 3});
// h.array reference is safely published
// But array elements can still be mutated!
h.array[0] = 99;  // Allowed (array itself is mutable)
```

=== Interview Questions: Synchronization

*Q1: Difference between synchronized and volatile?*

A:

*synchronized:*
- *Mutual exclusion*: Only one thread at a time
- *Memory visibility*: Changes visible to next thread acquiring lock
- *Atomicity*: Can make compound operations atomic
- *Reentrant*: Same thread can reacquire
- *Cost*: Heavier (lock acquisition overhead)

```java
synchronized void increment() {
    count++;  // Atomic (read-modify-write)
}
```

*volatile:*
- *No mutual exclusion*: Multiple threads can access simultaneously
- *Memory visibility*: Reads see latest write
- *No atomicity*: Compound operations NOT atomic
- *Cost*: Lighter (memory barriers only)

```java
volatile int count;
count++;  // NOT atomic! (race condition)
```

*When to use:*
- synchronized: Compound operations, mutual exclusion needed
- volatile: Single variable reads/writes, flags, double-checked locking

*Q2: Can volatile make compound actions atomic?*

A: *No*. Volatile only ensures visibility and ordering, not atomicity.

```java
volatile int count = 0;

// Thread 1
count++;  // Read 0, write 1

// Thread 2
count++;  // Read 0, write 1 (race!)

// Result: count = 1 (lost update)
```

Why not atomic:
```java
count++;  // Expands to THREE operations:
// 1. Read count
// 2. Add 1
// 3. Write count
// Another thread can interleave between steps!
```

Solutions:
1. *synchronized*:
```java
synchronized void increment() { count++; }
```

2. *AtomicInteger*:
```java
AtomicInteger count = new AtomicInteger(0);
count.incrementAndGet();  // Atomic CAS operation
```

*Q3: Explain how synchronized works at different contention levels.*

A: JVM optimizes synchronized with three lock types:

*1. Biased lock (no contention):*
- First thread marks object header with thread ID
- Re-entry: Just check thread ID (fastest, ~1ns)
- No CAS, no atomic operations

*2. Thin lock (light contention):*
- CAS to acquire lock
- Lock record stored in thread stack
- Loser spins briefly, then inflates to fat lock
- ~10-50ns

*3. Fat lock (heavy contention):*
- OS-level mutex
- Blocked threads parked (OS scheduler)
- ~100-1000ns (context switch)

*Transition:*
```
Biased (single thread)
  → Thin (2nd thread tries to acquire)
  → Fat (contention detected, multiple threads waiting)
```

*Why three levels?*
- Most locks acquired by single thread (biased is optimal)
- Light contention: Spin briefly (avoid OS overhead)
- Heavy contention: Block (don't waste CPU spinning)

== Intrinsic Locks & Monitors

=== Monitor Lock Implementation

*Monitor:* Synchronization construct with:
- Mutual exclusion (lock)
- Condition variables (wait/notify)

Every Java object has an implicit monitor.

```java
synchronized (obj) {
    // Acquire monitor lock
    while (!condition) {
        obj.wait();  // Release lock, wait for notification
    }
    // Reacquire lock, proceed
    // ...
    obj.notify();  // Wake one waiting thread
}  // Release monitor lock
```

*Internal structure:*
```
Monitor:
  ├─ Owner thread (current lock holder)
  ├─ Entry set (threads trying to acquire lock)
  └─ Wait set (threads that called wait())
```

*Flow:*
1. *Lock acquisition:*
   - If available: Acquire, become owner
   - If held: Add to entry set, block

2. *wait():*
   - Release lock
   - Add to wait set
   - Block

3. *notify()/notifyAll():*
   - Move thread(s) from wait set to entry set
   - They compete for lock again

4. *Lock release:*
   - Wake one thread from entry set
   - That thread becomes owner

=== wait/notify/notifyAll

*Rules:*
1. Must be called inside `synchronized` block on same object
2. `wait()` releases lock, sleeps until notified
3. `notify()` wakes one waiting thread (random)
4. `notifyAll()` wakes all waiting threads (they compete for lock)

*Producer-Consumer example:*
```java
class Queue {
    private final int[] buffer = new int[10];
    private int size = 0;

    public synchronized void put(int value) throws InterruptedException {
        while (size == buffer.length) {
            wait();  // Full: Wait for space
        }
        buffer[size++] = value;
        notifyAll();  // Wake consumers
    }

    public synchronized int take() throws InterruptedException {
        while (size == 0) {
            wait();  // Empty: Wait for data
        }
        int value = buffer[--size];
        notifyAll();  // Wake producers
        return value;
    }
}
```

*Why while loop (not if)?*
- *Spurious wakeups*: Thread can wake without notify
- *Multiple conditions*: Another thread may consume data before this thread runs
```java
// Wrong: if
if (size == 0) {
    wait();
}
// May not be true when thread wakes up!

// Correct: while
while (size == 0) {
    wait();  // Re-check after waking
}
```

*notify() vs notifyAll():*
- *notify()*: Wakes one thread (random). Use if all waiters have same condition.
- *notifyAll()*: Wakes all threads. Use if different conditions or to be safe.

```java
// Safe: notifyAll (always works)
notifyAll();

// Risky: notify (may wake wrong thread)
notify();  // If producer and consumer both waiting, may wake wrong one!
```

=== Lock Coarsening & Elision

*Lock coarsening:* Merge adjacent synchronized blocks (reduce lock overhead)

```java
// Before: Multiple lock acquisitions
synchronized (obj) {
    obj.x++;
}
doSomething();
synchronized (obj) {
    obj.y++;
}

// After coarsening (JIT optimization):
synchronized (obj) {
    obj.x++;
    doSomething();  // If doSomething() doesn't use obj
    obj.y++;
}
// Fewer lock acquisitions → faster
```

*Lock elision:* Remove locks on thread-local objects

```java
public String concat(String a, String b) {
    StringBuffer sb = new StringBuffer();  // Thread-local
    sb.append(a);  // synchronized (but only this thread uses sb)
    sb.append(b);  // synchronized
    return sb.toString();
}

// JIT removes synchronization (escape analysis proves sb doesn't escape)
```

*When JIT can elide locks:*
- Object doesn't escape method
- No other threads can access it

=== Interview Questions: Monitors

*Q1: Why must wait/notify be called inside synchronized block?*

A: To prevent race conditions between checking condition and waiting.

```java
// Without synchronization (broken):
if (queue.isEmpty()) {  // Check
    // Another thread adds item here!
    queue.wait();  // Wait (forever! Missed notification)
}

// With synchronization (correct):
synchronized (queue) {
    while (queue.isEmpty()) {
        queue.wait();  // Atomically: check + wait
    }
}
```

Technical reason:
- `wait()` releases monitor lock
- If not holding lock, can't release it → IllegalMonitorStateException

*Q2: Difference between notify() and notifyAll()?*

A:

*notify():*
- Wakes *one* waiting thread (JVM chooses)
- Faster (wakes fewer threads)
- Risk: May wake wrong thread (if multiple conditions)

*notifyAll():*
- Wakes *all* waiting threads
- Slower (more context switches)
- Safe (all threads re-check condition)

```java
// Producer-Consumer with two conditions
synchronized void put(int x) {
    while (full()) wait();
    // Add item
    notify();  // Risk: May wake another producer (not consumer)!
}

synchronized int take() {
    while (empty()) wait();
    // Remove item
    notify();  // Risk: May wake another consumer (not producer)!
}

// Fix: Use notifyAll() or separate Condition objects
```

*When to use:*
- notify(): All waiters have same condition, any one can proceed
- notifyAll(): Multiple conditions, or to be safe (default choice)

*Q3: What are spurious wakeups and how do you handle them?*

A: Spurious wakeup = thread wakes from `wait()` without `notify()/notifyAll()`.

Causes:
- OS-level signals
- JVM implementation details
- Hardware interrupts

Solution: *Always use while loop*, not if:

```java
// Wrong: if
synchronized (lock) {
    if (!condition) {
        lock.wait();  // Wakes up
    }
    // condition may be false! (spurious wakeup)
}

// Correct: while
synchronized (lock) {
    while (!condition) {
        lock.wait();  // Wakes up, re-checks
    }
    // condition guaranteed true
}
```

*Why while works:*
- After waking, re-checks condition
- If spurious wakeup (condition still false), waits again
- Only proceeds when condition actually true

== Thread Management

=== Thread Lifecycle & States

*Six states (Thread.State enum):*

1. *NEW*: Created but not started
```java
Thread t = new Thread();  // NEW
```

2. *RUNNABLE*: Running or ready to run
```java
t.start();  // NEW → RUNNABLE
```

3. *BLOCKED*: Waiting for monitor lock
```java
synchronized (lock) {  // If lock held by another thread → BLOCKED
    // ...
}
```

4. *WAITING*: Waiting indefinitely for another thread
```java
lock.wait();      // RUNNABLE → WAITING
Thread.join();    // RUNNABLE → WAITING
```

5. *TIMED_WAITING*: Waiting for specified time
```java
Thread.sleep(1000);     // RUNNABLE → TIMED_WAITING
lock.wait(1000);        // RUNNABLE → TIMED_WAITING
Thread.join(1000);      // RUNNABLE → TIMED_WAITING
```

6. *TERMINATED*: Finished execution
```java
// run() completes or throws exception → TERMINATED
```

*State transitions:*
```
NEW
  ↓ start()
RUNNABLE ←→ BLOCKED (waiting for lock)
  ↓ wait()/join()
WAITING
  ↓ notify()/interrupt()
RUNNABLE
  ↓ sleep(ms)/wait(ms)
TIMED_WAITING
  ↓ timeout/interrupt()
RUNNABLE
  ↓ run() completes
TERMINATED
```

=== Thread Operations

*sleep():*
- Pauses current thread for specified time
- Does NOT release locks
- Can be interrupted

```java
try {
    Thread.sleep(1000);  // Sleep 1 second
} catch (InterruptedException e) {
    // Thread interrupted during sleep
}
```

*yield():*
- Hints scheduler to give CPU to other threads
- No guarantees (scheduler may ignore)
- Does NOT release locks

```java
Thread.yield();  // Suggest giving up CPU
```

*join():*
- Wait for thread to die
- Current thread blocks until target thread finishes

```java
Thread t = new Thread(() -> {
    // Long task
});
t.start();
t.join();  // Wait for t to finish
// t is TERMINATED here
```

*interrupt():*
- Sets interrupt flag
- Wakes thread from sleep/wait (throws InterruptedException)
- Does NOT forcibly stop thread (thread must check flag)

```java
// Thread 1
Thread t = new Thread(() -> {
    while (!Thread.interrupted()) {  // Check flag
        // Work
    }
    // Cleanup and exit
});
t.start();

// Thread 2
t.interrupt();  // Set interrupt flag
```

=== ThreadLocal & InheritableThreadLocal

*ThreadLocal:* Per-thread storage

```java
ThreadLocal<Integer> userId = new ThreadLocal<>();

// Thread 1
userId.set(123);
userId.get();  // 123

// Thread 2
userId.set(456);
userId.get();  // 456

// Each thread has its own value!
```

*Implementation:*
```java
class Thread {
    ThreadLocal.ThreadLocalMap threadLocals;  // Per-thread map
}

class ThreadLocal<T> {
    public T get() {
        Thread t = Thread.currentThread();
        ThreadLocalMap map = t.threadLocals;
        if (map != null) {
            return map.get(this);  // 'this' ThreadLocal is key
        }
        return null;
    }
}
```

*Use cases:*
- User context (user ID, session)
- Database connections (one per thread)
- Formatters (SimpleDateFormat is NOT thread-safe)

```java
// Thread-safe date formatting
private static final ThreadLocal<SimpleDateFormat> formatter =
    ThreadLocal.withInitial(() -> new SimpleDateFormat("yyyy-MM-dd"));

public String format(Date date) {
    return formatter.get().format(date);  // Each thread has own formatter
}
```

*Memory leak risk:*
```java
ThreadLocal<byte[]> tl = new ThreadLocal<>();
tl.set(new byte[1024 * 1024]);  // 1 MB

// If thread never dies (thread pool), memory leaked!
// Fix: Always remove in finally
try {
    tl.set(data);
    // Use data
} finally {
    tl.remove();  // Important!
}
```

*InheritableThreadLocal:* Child threads inherit parent's value

```java
InheritableThreadLocal<String> context = new InheritableThreadLocal<>();

// Parent thread
context.set("parent-context");

Thread child = new Thread(() -> {
    String ctx = context.get();  // "parent-context" (inherited)
});
child.start();
```

=== Daemon Threads

*Daemon thread:* Background thread that doesn't prevent JVM exit

```java
Thread daemon = new Thread(() -> {
    while (true) {
        // Background work
    }
});
daemon.setDaemon(true);  // Mark as daemon (before start!)
daemon.start();

// When all non-daemon threads finish, JVM exits
// Daemon threads terminated abruptly (no finally blocks!)
```

*Use cases:*
- Garbage collection (JVM GC threads)
- Background monitoring
- Periodic cleanup

*Warning:* Daemons can be killed mid-operation (no cleanup guarantee)

```java
daemon.setDaemon(true);
daemon.start();
// Main thread exits → JVM exits → daemon killed (even if in critical section!)
```

=== Interview Questions: Thread Management

*Q1: What are thread states? Explain the lifecycle.*

A: Six states (Thread.State):

*1. NEW:* Created, not started
```java
Thread t = new Thread();  // NEW
```

*2. RUNNABLE:* Running or waiting for CPU
```java
t.start();  // NEW → RUNNABLE
```

*3. BLOCKED:* Waiting for monitor lock
```java
synchronized (lock) {  // BLOCKED if lock held
}
```

*4. WAITING:* Waiting indefinitely
```java
lock.wait();  // RUNNABLE → WAITING
t.join();
```

*5. TIMED_WAITING:* Waiting with timeout
```java
Thread.sleep(1000);  // RUNNABLE → TIMED_WAITING
lock.wait(1000);
```

*6. TERMINATED:* Finished
```java
// run() completes → TERMINATED
```

Lifecycle:
```
NEW --start()--> RUNNABLE <--notify()--> WAITING
                    ↕                      ↑
              (scheduler)             wait()
                    ↓
               TERMINATED
```

*Q2: What happens when you call start() twice on a thread?*

A: *IllegalThreadStateException*

```java
Thread t = new Thread();
t.start();  // OK: NEW → RUNNABLE
t.start();  // Exception! (already started)
```

Why: Thread can only be started once. After start(), state is not NEW.

To re-run, create new thread:
```java
Thread t1 = new Thread(task);
t1.start();
t1.join();

Thread t2 = new Thread(task);  // New thread
t2.start();
```

*Q3: How does ThreadLocal work and when can it cause memory leaks?*

A: ThreadLocal provides per-thread storage.

*Mechanism:*
- Each `Thread` has a `ThreadLocalMap` (map of ThreadLocal → value)
- `get()` retrieves value from current thread's map
- `set()` stores value in current thread's map

*Memory leak scenario:*
```java
ThreadLocal<byte[]> tl = new ThreadLocal<>();

// Thread pool thread
tl.set(new byte[1_000_000]);  // 1 MB
// Thread returns to pool (still alive)
// tl value never removed → memory leaked!
```

*Why leak:*
- ThreadLocalMap uses weak reference for *key* (ThreadLocal object)
- But uses *strong* reference for *value*
- If ThreadLocal GC'd but thread lives → value leaked

*Fix:*
```java
try {
    tl.set(data);
    // Use data
} finally {
    tl.remove();  // Always clean up!
}
```

*Detection:* Heap dump, look for `ThreadLocalMap` holding large objects.

*Q4: Difference between daemon and non-daemon threads?*

A:

*Non-daemon (user) thread:*
- JVM waits for them to finish
- JVM exits only when all user threads complete
- Default type

*Daemon thread:*
- Background service
- JVM doesn't wait for them
- Killed abruptly when all user threads finish
- Must set before `start()`

```java
Thread daemon = new Thread(() -> {
    while (true) {
        // Background work
    }
});
daemon.setDaemon(true);  // Mark as daemon
daemon.start();

// Main thread (user) exits → JVM exits → daemon killed
```

*Use cases:*
- Daemon: GC, monitoring, heartbeat
- Non-daemon: Application logic, user requests

*Warning:* Daemons don't execute finally blocks on JVM exit!

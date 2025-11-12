= Part IV: Advanced Concurrency

== Lock-Free Programming

=== CAS-Based Algorithms

*Lock-free:* At least one thread always makes progress (no blocking).

*Lock-free stack (Treiber stack):*
```java
class LockFreeStack<T> {
    private static class Node<T> {
        final T value;
        Node<T> next;

        Node(T value, Node<T> next) {
            this.value = value;
            this.next = next;
        }
    }

    private final AtomicReference<Node<T>> head = new AtomicReference<>();

    public void push(T value) {
        Node<T> newHead = new Node<>(value, null);
        Node<T> oldHead;
        do {
            oldHead = head.get();
            newHead.next = oldHead;
        } while (!head.compareAndSet(oldHead, newHead));
        // CAS: if head unchanged, set to newHead, else retry
    }

    public T pop() {
        Node<T> oldHead, newHead;
        do {
            oldHead = head.get();
            if (oldHead == null) return null;
            newHead = oldHead.next;
        } while (!head.compareAndSet(oldHead, newHead));
        return oldHead.value;
    }
}
```

*Key properties:*
- No locks (CAS only)
- No deadlock
- No blocking (threads spin on CAS)
- At least one thread makes progress

*Tradeoff:* High contention → many CAS retries (CPU waste)

=== Memory Ordering

*Memory ordering:* Constraints on instruction reordering.

*Four levels (weakest to strongest):*

1. *Plain:* No ordering guarantees (can reorder freely)
```java
int x;  // Plain field
x = 42;  // Can be reordered
```

2. *Opaque:* Bitwise atomicity (no tearing), but can reorder
```java
VarHandle.setOpaque(obj, offset, value);  // Atomic write, but can reorder
```

3. *Release/Acquire:* Memory barrier
```java
// Release: All prior writes visible before this write
VarHandle.setRelease(obj, offset, value);

// Acquire: All subsequent reads see prior writes
VarHandle.getAcquire(obj, offset);
```

4. *Volatile:* Full fence (strongest)
```java
volatile int x;
x = 42;  // Full memory barrier (no reordering across volatile)
```

*Release/Acquire pattern:*
```java
class Message {
    int data;  // Plain field
    volatile boolean ready = false;  // Volatile flag

    // Thread 1 (producer)
    void send() {
        data = 42;  // (1)
        ready = true;  // (2) Volatile write (release)
        // (1) happens-before (2)
    }

    // Thread 2 (consumer)
    void receive() {
        if (ready) {  // (3) Volatile read (acquire)
            int value = data;  // (4)
            // (2) happens-before (3), (3) happens-before (4)
            // Therefore: value == 42 (guaranteed)
        }
    }
}
```

=== Lock-Free Algorithms

*Michael-Scott Queue (lock-free FIFO):*
```java
class LockFreeQueue<T> {
    private static class Node<T> {
        final T value;
        final AtomicReference<Node<T>> next = new AtomicReference<>();

        Node(T value) { this.value = value; }
    }

    private final AtomicReference<Node<T>> head;
    private final AtomicReference<Node<T>> tail;

    public LockFreeQueue() {
        Node<T> dummy = new Node<>(null);
        head = new AtomicReference<>(dummy);
        tail = new AtomicReference<>(dummy);
    }

    public void enqueue(T value) {
        Node<T> newNode = new Node<>(value);
        while (true) {
            Node<T> curTail = tail.get();
            Node<T> tailNext = curTail.next.get();

            if (curTail == tail.get()) {  // Consistent read
                if (tailNext == null) {  // Tail pointing to last node
                    if (curTail.next.compareAndSet(null, newNode)) {
                        tail.compareAndSet(curTail, newNode);  // Swing tail
                        return;
                    }
                } else {  // Tail behind, help move it
                    tail.compareAndSet(curTail, tailNext);
                }
            }
        }
    }

    public T dequeue() {
        while (true) {
            Node<T> curHead = head.get();
            Node<T> curTail = tail.get();
            Node<T> headNext = curHead.next.get();

            if (curHead == head.get()) {
                if (curHead == curTail) {  // Empty or tail behind
                    if (headNext == null) {
                        return null;  // Empty
                    }
                    tail.compareAndSet(curTail, headNext);  // Help move tail
                } else {
                    T value = headNext.value;
                    if (head.compareAndSet(curHead, headNext)) {
                        return value;
                    }
                }
            }
        }
    }
}
```

=== Wait-Free vs Lock-Free vs Obstruction-Free

*Three progress guarantees (strongest to weakest):*

*1. Wait-free:* Every thread makes progress in bounded steps
- Strongest guarantee
- Example: `AtomicInteger.get()`, CAS with helping
- Hardest to implement

*2. Lock-free:* *Some* thread always makes progress
- At least one thread succeeds per step
- Example: CAS-based stack/queue
- More practical than wait-free

*3. Obstruction-free:* Thread makes progress if runs in isolation
- Weakest guarantee
- Can livelock (all threads retry forever)
- Example: Naive CAS loop

```java
// Obstruction-free (can livelock)
void increment(AtomicInteger counter) {
    int oldValue, newValue;
    do {
        oldValue = counter.get();
        newValue = oldValue + 1;
    } while (!counter.compareAndSet(oldValue, newValue));
    // If many threads contend, all may retry forever (livelock)
}

// Lock-free (at least one succeeds per iteration)
// AtomicInteger.incrementAndGet() is lock-free

// Wait-free (all threads succeed in bounded steps)
AtomicInteger counter = new AtomicInteger();
int value = counter.get();  // Wait-free (never blocks, O(1) time)
```

=== Interview Questions: Lock-Free

*Q1: What is lock-free programming and what are its advantages?*

A: Lock-free = at least one thread always makes progress (no OS-level blocking).

*Characteristics:*
- Uses CAS (Compare-And-Swap) instead of locks
- No deadlock (no locks to acquire)
- No priority inversion
- No blocking (threads spin on CAS)

*Advantages:*
1. *Lower latency*: No context switches (~1-10ns vs ~1000ns for locks)
2. *Scalability*: No lock contention
3. *Progress guarantee*: Some thread always succeeds

*Disadvantages:*
1. *Complexity*: Hard to implement correctly (ABA problem, memory ordering)
2. *Livelock risk*: High contention → many CAS retries
3. *CPU waste*: Spinning consumes CPU

*When to use:*
- Ultra-low latency required (HFT, real-time systems)
- Low-to-medium contention
- Simple data structures (stack, queue, counter)

*Q2: Explain a CAS-based algorithm (e.g., lock-free stack).*

A: Treiber stack (lock-free LIFO):

```java
class LockFreeStack<T> {
    private static class Node<T> {
        T value;
        Node<T> next;
    }

    private final AtomicReference<Node<T>> head = new AtomicReference<>();

    public void push(T value) {
        Node<T> newNode = new Node<>();
        newNode.value = value;

        Node<T> oldHead;
        do {
            oldHead = head.get();  // Read current head
            newNode.next = oldHead;  // Link to old head
        } while (!head.compareAndSet(oldHead, newNode));
        // CAS: if head still == oldHead, set to newNode
        // Retry if another thread changed head
    }

    public T pop() {
        Node<T> oldHead, newHead;
        do {
            oldHead = head.get();
            if (oldHead == null) return null;  // Empty
            newHead = oldHead.next;
        } while (!head.compareAndSet(oldHead, newHead));
        return oldHead.value;
    }
}
```

*Key idea:*
- Read current state (head)
- Compute new state (newNode.next = head)
- CAS to update (retry if state changed)

*Properties:*
- Lock-free (at least one push/pop succeeds per iteration)
- No deadlock
- ABA problem possible (use AtomicStampedReference if needed)

*Q3: What are the challenges of lock-free programming?*

A:

*1. ABA problem:*
```java
// Thread 1: Read A
head.get();  // A

// Thread 2: Change A → B → A
head.set(B);
head.set(A);  // Same A object!

// Thread 1: CAS succeeds (head == A)
head.compareAndSet(A, C);  // Succeeds, but state changed through B
```
Solution: `AtomicStampedReference` (version stamp)

*2. Memory ordering:*
- Must use volatile/VarHandle for visibility
- Incorrect ordering → race conditions

*3. Complexity:*
- Hard to reason about all interleavings
- Requires deep understanding of memory models

*4. Livelock:*
- High contention → all threads retry forever
- May need backoff strategy

*5. Linearizability:*
- Ensuring operations appear atomic
- Complex correctness proofs

*When NOT to use:*
- High contention (locks may be faster)
- Complex operations (hard to make lock-free)
- No latency requirements (locks simpler)

== Thread Pools & Executors

=== ThreadPoolExecutor

*Constructor parameters:*
```java
ThreadPoolExecutor pool = new ThreadPoolExecutor(
    int corePoolSize,       // Min threads (kept alive)
    int maximumPoolSize,    // Max threads
    long keepAliveTime,     // Idle thread timeout
    TimeUnit unit,
    BlockingQueue<Runnable> workQueue,  // Task queue
    ThreadFactory threadFactory,         // Thread creation
    RejectedExecutionHandler handler     // Rejection policy
);
```

*Execution flow:*
1. If threads < corePoolSize → Create thread, execute task
2. If threads >= corePoolSize → Enqueue task
3. If queue full && threads < maximumPoolSize → Create thread
4. If threads == maximumPoolSize && queue full → Reject task

*Example:*
```java
ThreadPoolExecutor pool = new ThreadPoolExecutor(
    5,   // corePoolSize (always kept alive)
    10,  // maximumPoolSize
    60, TimeUnit.SECONDS,  // keepAliveTime for extra threads
    new LinkedBlockingQueue<>(100)  // Queue capacity: 100
);

// Submit 150 tasks:
// - Tasks 1-5: Execute immediately (create 5 core threads)
// - Tasks 6-105: Enqueue (queue capacity 100)
// - Tasks 106-110: Create 5 more threads (up to maximumPoolSize)
// - Tasks 111-150: Rejected (queue full, max threads reached)
```

*Queue strategies:*

1. *Unbounded:* `LinkedBlockingQueue()` (no max capacity)
   - Never rejects
   - maximumPoolSize has no effect
   - Risk: OOM if tasks accumulate

2. *Bounded:* `ArrayBlockingQueue(N)`
   - Fixed capacity
   - Rejects when full

3. *Direct handoff:* `SynchronousQueue()`
   - No queuing (capacity 0)
   - Creates thread immediately (up to max)
   - Rejects if max threads reached

4. *Priority:* `PriorityBlockingQueue()`
   - Tasks executed by priority

*Rejection policies:*
```java
// 1. AbortPolicy (default): Throw RejectedExecutionException
new ThreadPoolExecutor.AbortPolicy()

// 2. CallerRunsPolicy: Caller thread executes task (throttling)
new ThreadPoolExecutor.CallerRunsPolicy()

// 3. DiscardPolicy: Silently discard task
new ThreadPoolExecutor.DiscardPolicy()

// 4. DiscardOldestPolicy: Discard oldest queued task, retry
new ThreadPoolExecutor.DiscardOldestPolicy()
```

*Preconfigured pools:*
```java
// Fixed size (corePoolSize == maximumPoolSize)
ExecutorService pool = Executors.newFixedThreadPool(10);

// Single thread (sequential execution)
ExecutorService pool = Executors.newSingleThreadExecutor();

// Cached (create threads on demand, reuse idle)
ExecutorService pool = Executors.newCachedThreadPool();

// Scheduled (for periodic tasks)
ScheduledExecutorService pool = Executors.newScheduledThreadPool(5);
```

=== ForkJoinPool & Work Stealing

*ForkJoinPool:* Designed for recursive divide-and-conquer tasks.

*Work stealing:*
- Each thread has own deque of tasks
- Thread takes tasks from head of own deque (LIFO)
- Idle thread *steals* from tail of another thread's deque (FIFO)
- Reduces contention (different ends of deque)

```
Thread 1 deque:  [T1, T2, T3, T4]
                  ↑head        ↑tail
Thread 1 takes from head (LIFO)
Thread 2 steals from tail (FIFO, if idle)
```

*RecursiveTask example:*
```java
class SumTask extends RecursiveTask<Long> {
    private final long[] array;
    private final int start, end;
    private static final int THRESHOLD = 1000;

    @Override
    protected Long compute() {
        if (end - start <= THRESHOLD) {
            // Base case: Compute directly
            long sum = 0;
            for (int i = start; i < end; i++) {
                sum += array[i];
            }
            return sum;
        } else {
            // Recursive case: Fork
            int mid = (start + end) / 2;
            SumTask left = new SumTask(array, start, mid);
            SumTask right = new SumTask(array, mid, end);

            left.fork();  // Async execute left
            long rightResult = right.compute();  // Execute right in current thread
            long leftResult = left.join();  // Wait for left result

            return leftResult + rightResult;
        }
    }
}

// Usage:
ForkJoinPool pool = new ForkJoinPool();
long sum = pool.invoke(new SumTask(array, 0, array.length));
```

*When to use ForkJoinPool:*
- Recursive divide-and-conquer
- Many small subtasks
- Tasks spawn more tasks
- Example: Parallel merge sort, parallel stream

=== ScheduledThreadPoolExecutor

*For periodic or delayed tasks:*

```java
ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(5);

// Execute once after delay
scheduler.schedule(() -> {
    System.out.println("Task executed");
}, 5, TimeUnit.SECONDS);

// Execute periodically (fixed rate)
scheduler.scheduleAtFixedRate(() -> {
    System.out.println("Tick");
}, 0, 1, TimeUnit.SECONDS);
// Initial delay: 0s, period: 1s
// Executes at t=0, t=1, t=2, ... (regardless of execution time)

// Execute periodically (fixed delay)
scheduler.scheduleWithFixedDelay(() -> {
    System.out.println("Tick");
}, 0, 1, TimeUnit.SECONDS);
// Delay: 1s *after* previous execution completes
// If task takes 2s: t=0, t=3, t=6, ...
```

*Fixed rate vs fixed delay:*
- *Fixed rate*: Start every N seconds (can overlap if task > period)
- *Fixed delay*: Wait N seconds after completion (no overlap)

=== Virtual Threads (Java 21+)

*Traditional threads:* 1:1 mapping to OS threads (expensive, limited)

*Virtual threads:* Millions of lightweight threads (cheap, M:N mapping)

```java
// Old: Platform thread (expensive)
Thread t = new Thread(() -> {
    // Task
});
t.start();

// New: Virtual thread (cheap)
Thread vt = Thread.startVirtualThread(() -> {
    // Task (can have millions!)
});

// ExecutorService with virtual threads
ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
executor.submit(() -> {
    // Each task gets own virtual thread
});
```

*Key benefits:*
1. *Cheap creation:* ~1KB vs ~1MB for platform thread
2. *Blocking friendly:* Blocking I/O doesn't waste OS thread
3. *Millions of threads:* Can have 1M+ virtual threads

*How it works:*
- Virtual threads run on *carrier* threads (platform threads)
- When virtual thread blocks (I/O, sleep), carrier is freed
- Carrier can run another virtual thread (M:N scheduling)

```
1M virtual threads → ~10 carrier threads (platform threads)
Blocking doesn't waste carrier (scheduler switches to another virtual thread)
```

*Migration impact:*
```java
// Old: Thread pool limits concurrency
ExecutorService pool = Executors.newFixedThreadPool(100);
// Max 100 concurrent tasks (blocking I/O wastes threads)

// New: Virtual threads (no limit)
ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor();
// Millions of tasks OK (blocking doesn't waste resources)
```

=== Interview Questions: Thread Pools

*Q1: Explain ThreadPoolExecutor parameters and execution flow.*

A:

*Parameters:*
```java
ThreadPoolExecutor(
    corePoolSize,      // Min threads (always kept)
    maximumPoolSize,   // Max threads
    keepAliveTime,     // Idle timeout for extra threads
    unit,
    workQueue,         // Task queue
    threadFactory,
    handler            // Rejection policy
)
```

*Execution flow:*
1. threads < core → Create thread
2. threads >= core → Enqueue
3. Queue full && threads < max → Create thread
4. Queue full && threads == max → Reject

*Example:*
```java
pool = new ThreadPoolExecutor(5, 10, 60, SECONDS,
    new LinkedBlockingQueue<>(100));

// 5 tasks: Create 5 core threads
// 100 tasks: Enqueue (queue has capacity 100)
// 5 tasks: Create 5 more threads (up to max 10)
// 1 task: Rejected (queue full, max threads)
```

*Q2: What is work stealing in ForkJoinPool?*

A: Work stealing = idle threads *steal* tasks from busy threads.

*Mechanism:*
- Each thread has own deque of tasks
- Thread takes tasks from *head* of own deque (LIFO, locality)
- Idle thread steals from *tail* of another's deque (FIFO, less contention)

```
Thread 1: [T1, T2, T3, T4]
           ↑head     ↑tail
Takes from head (own tasks)

Thread 2 (idle): Steals from Thread 1's tail
Less contention (different ends)
```

*Benefits:*
1. Load balancing (idle threads help busy ones)
2. Less contention (steal from opposite end)
3. Cache locality (own tasks from head)

*Use case:* Recursive tasks that spawn subtasks (ForkJoinPool ideal)

*Q3: How do virtual threads change concurrent programming?*

A: Virtual threads (Java 21+) remove thread limits.

*Traditional (platform) threads:*
- 1:1 mapping to OS threads
- Expensive (~1MB stack, OS overhead)
- Limited (thousands at most)
- Blocking I/O wastes thread

*Virtual threads:*
- M:N mapping (millions of virtual → few carrier threads)
- Cheap (~1KB, no OS overhead)
- Millions possible
- Blocking doesn't waste carrier (scheduler switches)

*Impact:*
```java
// Old: Limited by thread pool size
ExecutorService pool = Executors.newFixedThreadPool(100);
// Blocking I/O → only 100 concurrent requests

// New: Unlimited concurrency
ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor();
// Millions of requests OK (blocking releases carrier)
```

*Use case:*
- High-concurrency servers (handle 1M+ connections)
- Blocking I/O (JDBC, file I/O) without async complexity
- Simplify async code (write blocking-style, get async performance)

== Synchronizers

=== CountDownLatch

*Use case:* Wait for N events to complete (one-shot).

```java
CountDownLatch latch = new CountDownLatch(3);  // Wait for 3 events

// Worker threads
for (int i = 0; i < 3; i++) {
    new Thread(() -> {
        // Do work
        latch.countDown();  // Signal completion
    }).start();
}

// Main thread
latch.await();  // Wait for all 3 to complete
System.out.println("All workers done");
```

*Methods:*
```java
latch.countDown();  // Decrement count
latch.await();      // Block until count reaches 0
latch.await(1, TimeUnit.SECONDS);  // Timeout
```

*Example: Parallel initialization*
```java
CountDownLatch latch = new CountDownLatch(3);

// Initialize database
new Thread(() -> {
    initDatabase();
    latch.countDown();
}).start();

// Initialize cache
new Thread(() -> {
    initCache();
    latch.countDown();
}).start();

// Load config
new Thread(() -> {
    loadConfig();
    latch.countDown();
}).start();

latch.await();  // Wait for all initialization
System.out.println("System ready");
```

=== CyclicBarrier

*Use case:* Threads wait for each other at barrier (reusable).

```java
CyclicBarrier barrier = new CyclicBarrier(3, () -> {
    System.out.println("All arrived, barrier action");
});

// Worker threads
for (int i = 0; i < 3; i++) {
    new Thread(() -> {
        while (true) {
            // Phase 1
            doWork();
            barrier.await();  // Wait for all
            // Phase 2
            doMoreWork();
            barrier.await();  // Wait for all (reusable)
        }
    }).start();
}
```

*Difference from CountDownLatch:*
- *CountDownLatch*: One-shot (count → 0, done)
- *CyclicBarrier*: Reusable (resets after all arrive)

=== Semaphore

*Use case:* Limit concurrent access to resource (permits).

```java
Semaphore semaphore = new Semaphore(3);  // 3 permits

// Thread
semaphore.acquire();  // Acquire permit (blocks if none available)
try {
    // Access resource (max 3 threads concurrently)
} finally {
    semaphore.release();  // Release permit
}
```

*Example: Connection pool*
```java
class ConnectionPool {
    private final Semaphore permits = new Semaphore(10);  // Max 10 connections
    private final List<Connection> connections = new ArrayList<>();

    public Connection getConnection() throws InterruptedException {
        permits.acquire();  // Wait for available connection
        synchronized (connections) {
            return connections.remove(0);
        }
    }

    public void releaseConnection(Connection conn) {
        synchronized (connections) {
            connections.add(conn);
        }
        permits.release();  // Free permit
    }
}
```

=== Phaser

*Use case:* Flexible barrier with dynamic parties (advanced CyclicBarrier).

```java
Phaser phaser = new Phaser(3);  // 3 parties

// Worker
for (int phase = 0; phase < 5; phase++) {
    doWork(phase);
    phaser.arriveAndAwaitAdvance();  // Wait for all at this phase
}
phaser.arriveAndDeregister();  // Deregister when done
```

*Dynamic parties:*
```java
Phaser phaser = new Phaser();

// Add parties dynamically
phaser.register();  // Add party
phaser.bulkRegister(5);  // Add 5 parties

// Remove party
phaser.arriveAndDeregister();
```

*Difference from CyclicBarrier:*
- *CyclicBarrier*: Fixed parties
- *Phaser*: Dynamic parties (can register/deregister)

=== Exchanger

*Use case:* Two threads exchange data (rendezvous).

```java
Exchanger<String> exchanger = new Exchanger<>();

// Thread 1
String data1 = "Data from Thread 1";
String received = exchanger.exchange(data1);  // Blocks until Thread 2 arrives

// Thread 2
String data2 = "Data from Thread 2";
String received = exchanger.exchange(data2);  // Swap with Thread 1

// Thread 1 receives data2
// Thread 2 receives data1
```

*Example: Producer-Consumer buffer swap*
```java
Exchanger<List<String>> exchanger = new Exchanger<>();

// Producer
List<String> fullBuffer = new ArrayList<>();
while (true) {
    fillBuffer(fullBuffer);  // Fill buffer
    fullBuffer = exchanger.exchange(fullBuffer);  // Swap with consumer
    fullBuffer.clear();  // Reuse empty buffer
}

// Consumer
List<String> emptyBuffer = new ArrayList<>();
while (true) {
    emptyBuffer = exchanger.exchange(emptyBuffer);  // Get full buffer
    process(emptyBuffer);  // Process
    emptyBuffer.clear();  // Return empty
}
```

=== Interview Questions: Synchronizers

*Q1: CountDownLatch vs CyclicBarrier - when to use which?*

A:

*CountDownLatch:*
- *One-shot*: Count down to 0, then done (not reusable)
- *N→0 pattern*: Wait for N events
- *Asymmetric*: Some threads count down, others wait

```java
CountDownLatch latch = new CountDownLatch(3);
// Workers count down
latch.countDown();
// Main waits
latch.await();
// Cannot reuse (count stays at 0)
```

*CyclicBarrier:*
- *Reusable*: Resets after all parties arrive
- *Synchronization point*: All threads wait for each other
- *Symmetric*: All threads wait at barrier

```java
CyclicBarrier barrier = new CyclicBarrier(3);
// All threads wait
barrier.await();
// Resets, can reuse
```

*When to use:*
- *CountDownLatch*: Wait for initialization, start signal
- *CyclicBarrier*: Iterative algorithms, phase synchronization

*Q2: When would you use Semaphore?*

A: Semaphore limits concurrent access to resource (permit-based).

*Use cases:*

1. *Resource pool*: Limit connections, threads, etc.
```java
Semaphore pool = new Semaphore(10);  // Max 10 connections
pool.acquire();
try {
    useConnection();
} finally {
    pool.release();
}
```

2. *Rate limiting*:
```java
Semaphore rateLimiter = new Semaphore(100);  // 100 requests/period
rateLimiter.acquire();
processRequest();
// Separate thread releases permits periodically
```

3. *Mutual exclusion*:
```java
Semaphore mutex = new Semaphore(1);  // Binary semaphore
mutex.acquire();
try {
    // Critical section
} finally {
    mutex.release();
}
```

*Difference from Lock:*
- Semaphore: Can be released by different thread
- Lock: Must be released by same thread

*Q3: Give a real-world use case for Phaser.*

A: Phaser = flexible barrier with dynamic parties.

*Use case:* Parallel iterative algorithm with varying workers

```java
Phaser phaser = new Phaser(1);  // 1 = main thread

// Create workers (dynamic)
for (int i = 0; i < numWorkers; i++) {
    phaser.register();  // Add party
    new Thread(() -> {
        for (int phase = 0; phase < MAX_PHASES; phase++) {
            computePartialResult(phase);
            phaser.arriveAndAwaitAdvance();  // Wait for all

            if (shouldStop()) {
                phaser.arriveAndDeregister();  // Remove self
                return;
            }
        }
        phaser.arriveAndDeregister();
    }).start();
}

// Main thread coordinates phases
for (int phase = 0; phase < MAX_PHASES; phase++) {
    phaser.arriveAndAwaitAdvance();  // Wait for workers
    if (allWorkersDone()) break;
}

phaser.arriveAndDeregister();  // Main done
```

*Benefits over CyclicBarrier:*
- Dynamic parties (register/deregister)
- Phase number tracking
- Termination handling

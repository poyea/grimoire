= Part I: Java Memory Model & JVM Internals

== Java Memory Model (JMM)

The Java Memory Model defines how threads interact through memory and what behaviors are allowed in concurrent execution.

=== Happens-Before Relationship

*Happens-before:* If action A happens-before action B, then changes made by A are visible to B.

*Rules:*

1. *Program order*: Each action in a thread happens-before every subsequent action in same thread

2. *Monitor lock*: Unlock happens-before every subsequent lock on same monitor
```java
synchronized (lock) {
    x = 1;  // Write
}  // Unlock

// Later, another thread:
synchronized (lock) {  // Lock
    int y = x;  // Sees x = 1 (guaranteed)
}
```

3. *Volatile*: Write to volatile happens-before every subsequent read of same variable
```java
volatile boolean flag = false;

// Thread 1
x = 1;
flag = true;  // Volatile write

// Thread 2
if (flag) {  // Volatile read
    int y = x;  // Sees x = 1 (guaranteed)
}
```

4. *Thread start*: `thread.start()` happens-before any action in started thread

5. *Thread termination*: All actions in thread happen-before `thread.join()` returns

6. *Transitivity*: A happens-before B, B happens-before C → A happens-before C

=== Memory Visibility & Reordering

*Problem without synchronization:*
```java
class UnsafeFlag {
    private boolean flag = false;  // NOT volatile
    private int value = 0;

    // Thread 1
    public void writer() {
        value = 42;
        flag = true;  // May be reordered before value = 42!
    }

    // Thread 2
    public void reader() {
        if (flag) {  // May see flag = true
            System.out.println(value);  // But value = 0! (stale read)
        }
    }
}
```

*Why reordering happens:*
1. *Compiler optimization*: Reorders instructions for performance
2. *CPU out-of-order execution*: Executes instructions in different order
3. *CPU cache*: Each core has private cache, writes not immediately visible

*Solution: volatile*
```java
class SafeFlag {
    private volatile boolean flag = false;  // Volatile
    private int value = 0;

    public void writer() {
        value = 42;
        flag = true;  // Volatile write: barrier
        // All writes before this are flushed to main memory
    }

    public void reader() {
        if (flag) {  // Volatile read: barrier
            // All subsequent reads see latest values
            System.out.println(value);  // Sees value = 42 (guaranteed)
        }
    }
}
```

=== Safe Publication

*How to safely publish objects to other threads:*

1. *Initialize in static initializer*
```java
public static final Object instance = new Object();  // Thread-safe
```

2. *Store in volatile field*
```java
private volatile Object instance;

public void init() {
    instance = new Object();  // Safe publication
}
```

3. *Store in AtomicReference*
```java
private AtomicReference<Object> ref = new AtomicReference<>();

public void init() {
    ref.set(new Object());  // Safe publication
}
```

4. *Store in properly locked field*
```java
private Object instance;
private final Object lock = new Object();

public void init() {
    synchronized (lock) {
        instance = new Object();  // Safe publication
    }
}
```

5. *Initialize in final field*
```java
class Holder {
    private final Object instance;

    public Holder() {
        instance = new Object();  // Safe publication (final field)
    }
}
```

*Unsafe publication:*
```java
class Unsafe {
    private Object instance;  // NOT volatile, NOT synchronized

    public void init() {
        instance = new Object();  // Unsafe!
        // Other threads may see partially constructed object!
    }
}
```

=== Interview Questions: JMM

*Q1: What is the happens-before relationship?*

A: Happens-before defines memory visibility guarantees in concurrent programs. If action A happens-before action B, then:
1. Changes made by A are visible to B
2. A is ordered before B (from B's perspective)

Key rules:
- Program order: Sequential statements in same thread
- Monitor lock: Unlock happens-before subsequent lock
- Volatile: Write happens-before subsequent read
- Thread start/join: Parent thread actions visible to child

Example:
```java
volatile boolean ready = false;
int value = 0;

// Thread 1
value = 42;        // (1)
ready = true;      // (2) Volatile write

// Thread 2
if (ready) {       // (3) Volatile read
    print(value);  // (4) Sees value = 42
}
// (1) happens-before (2) [program order]
// (2) happens-before (3) [volatile rule]
// (3) happens-before (4) [program order]
// Therefore: (1) happens-before (4) [transitivity]
```

*Q2: How does volatile guarantee visibility?*

A: Volatile provides two guarantees:

1. *Visibility*: Write to volatile variable immediately flushed to main memory. Read from volatile always reads from main memory (not cache).

2. *Ordering*: Acts as memory barrier
   - Writes before volatile write cannot be reordered after it
   - Reads after volatile read cannot be reordered before it

```java
// Without volatile
private boolean flag = false;
thread1: flag = true;  // May stay in CPU cache
thread2: if (flag) ... // May not see update

// With volatile
private volatile boolean flag = false;
thread1: flag = true;  // Flushed to main memory immediately
thread2: if (flag) ... // Reads from main memory
```

But volatile does NOT make compound actions atomic:
```java
volatile int count = 0;
count++;  // NOT atomic! (read-modify-write)
// Thread 1: read 0
// Thread 2: read 0
// Thread 1: write 1
// Thread 2: write 1 (lost update!)
```

For atomic updates, use `AtomicInteger` or `synchronized`.

*Q3: What is safe publication and why is it important?*

A: Safe publication ensures that when you share an object with other threads, they see it fully constructed (not partially initialized).

Unsafe:
```java
class Holder {
    private Object value;

    public Holder(Object value) {
        this.value = value;
    }
}

// Thread 1
Holder holder = new Holder(new Object());
global = holder;  // Unsafe!

// Thread 2
if (global != null) {
    // May see holder.value = null! (partially constructed)
}
```

Safe publication techniques:
1. Final field: `private final Object value;` (constructor guarantee)
2. Volatile: `volatile Holder global;`
3. Synchronized: `synchronized { global = holder; }`
4. Concurrent collection: `ConcurrentHashMap.put(key, holder);`

== JVM Architecture

=== Runtime Data Areas

*Per-JVM:*
1. *Method Area (Metaspace in Java 8+)*
   - Class metadata, static variables, constant pool
   - Shared across all threads
   - Garbage collected (unload unused classes)

2. *Heap*
   - All objects and arrays
   - Shared across all threads
   - Garbage collected

*Per-Thread:*
3. *JVM Stack*
   - Stack frames (one per method call)
   - Local variables, operand stack, frame data
   - Fixed or dynamic size (typically 1MB)
   - StackOverflowError if exceeded

4. *PC Register*
   - Current instruction address
   - One per thread

5. *Native Method Stack*
   - For native (JNI) methods
   - Implementation-dependent

```
┌─────────────────────────────────────┐
│          Method Area (Metaspace)    │
│  Class metadata, static variables   │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│              Heap                   │
│    Objects, arrays (GC managed)     │
└─────────────────────────────────────┘
┌──────────┐ ┌──────────┐ ┌──────────┐
│Thread 1  │ │Thread 2  │ │Thread 3  │
│          │ │          │ │          │
│ Stack    │ │ Stack    │ │ Stack    │
│ PC       │ │ PC       │ │ PC       │
│ Native   │ │ Native   │ │ Native   │
└──────────┘ └──────────┘ └──────────┘
```

=== Stack Frame Structure

Each method call creates a stack frame:

```
┌────────────────────────┐
│   Return Address       │
│   (where to return)    │
├────────────────────────┤
│   Local Variable Table │
│   (method parameters,  │
│    local variables)    │
├────────────────────────┤
│   Operand Stack        │
│   (JVM workspace)      │
├────────────────────────┤
│   Frame Data           │
│   (constant pool ref,  │
│    exception table)    │
└────────────────────────┘
```

Example:
```java
public int add(int a, int b) {
    int sum = a + b;
    return sum;
}

// Bytecode:
0: iload_1      // Load local var 1 (a) → operand stack
1: iload_2      // Load local var 2 (b) → operand stack
2: iadd         // Pop two, add, push result
3: istore_3     // Pop result, store in local var 3 (sum)
4: iload_3      // Load sum → operand stack
5: ireturn      // Return top of stack

// Local variable table:
// 0: this
// 1: a
// 2: b
// 3: sum
```

=== Class Loading Mechanism

*Three phases:*

1. *Loading:* Find and read .class file into memory
   - Bootstrap ClassLoader: Core Java classes (rt.jar)
   - Extension ClassLoader: Extensions (lib/ext)
   - Application ClassLoader: Classpath
   - Custom ClassLoaders: User-defined

2. *Linking:*
   - *Verification*: Check bytecode validity (security)
   - *Preparation*: Allocate memory for static variables, set default values
   - *Resolution*: Convert symbolic references to direct references (optional)

3. *Initialization:* Execute static initializers and static blocks

```java
class Example {
    static int x = 10;           // Preparation: x = 0, Initialization: x = 10
    static {
        System.out.println("Static block");  // Initialization phase
    }

    public static void main(String[] args) {
        System.out.println(x);
    }
}
```

*Class initialization triggers:*
1. Creating instance: `new Example()`
2. Accessing static field: `Example.x`
3. Calling static method: `Example.method()`
4. Reflection: `Class.forName("Example")`
5. Initializing subclass (initialize superclass first)
6. Main class (JVM startup)

*Lazy initialization:*
```java
class Lazy {
    static {
        System.out.println("Lazy initialized");
    }
}

// Not printed yet (class not loaded)
Lazy.class;  // Reference to Class object (no initialization)

// Printed here (class initialized)
new Lazy();  // or Lazy.staticMethod();
```

=== Method Dispatch

*Three types:*

1. *invokestatic:* Static methods (compile-time binding)
```java
Math.max(a, b);  // invokestatic
```

2. *invokevirtual:* Instance methods (runtime binding via vtable)
```java
object.method();  // invokevirtual
```

3. *invokeinterface:* Interface methods (runtime binding via itable)
```java
List<String> list = ...;
list.add("x");  // invokeinterface
```

4. *invokespecial:* Constructors, private methods, super calls
```java
super.method();  // invokespecial
```

5. *invokedynamic:* Dynamic languages, lambdas (Java 7+)
```java
Runnable r = () -> {};  // invokedynamic
```

*Virtual method table (vtable):*
```java
class Animal {
    void eat() { }   // vtable[0]
    void sleep() { } // vtable[1]
}

class Dog extends Animal {
    void eat() { }   // vtable[0] (overridden)
    void bark() { }  // vtable[2] (new method)
}

Animal a = new Dog();
a.eat();  // invokevirtual
// 1. Load vtable pointer from object header
// 2. Load vtable[0] (eat method address)
// 3. Call method (Dog's eat, not Animal's)
```

=== Interview Questions: JVM Architecture

*Q1: Explain the class loading process in Java.*

A: Class loading has three phases:

*1. Loading:*
- Find .class file (via classloader hierarchy)
- Read bytecode into memory
- Create `Class` object

Classloader hierarchy (delegation model):
```
Bootstrap (native, loads rt.jar)
    ↑
Extension (loads lib/ext)
    ↑
Application (loads classpath)
    ↑
Custom (user-defined)
```

*2. Linking:*
- *Verification*: Check bytecode is valid and safe
- *Preparation*: Allocate memory for static fields, set default values (0, null, false)
- *Resolution*: Resolve symbolic references to concrete references

*3. Initialization:*
- Execute static initializers in order
- Run static blocks

Example:
```java
class Example {
    static int x = computeX();  // Initialization
    static {
        System.out.println("Init");  // Initialization
    }
}

// Load: Read Example.class
// Link: Verify bytecode, allocate x (set to 0), resolve references
// Initialize: x = computeX(), run static block
```

*Q2: What are the different memory areas in JVM?*

A:

*Shared (all threads):*
1. *Heap*: Objects and arrays. GC-managed. `-Xmx` sets max size.
2. *Method Area (Metaspace)*: Class metadata, static variables, constant pool. GC-managed (class unloading).

*Per-thread:*
3. *JVM Stack*: Stack frames (local variables, operand stack). `-Xss` sets size. `StackOverflowError` if exceeded.
4. *PC Register*: Current instruction pointer.
5. *Native Stack*: For JNI native methods.

```
Heap (-Xmx): Objects, arrays
Metaspace (-XX:MetaspaceSize): Class metadata
Stack (-Xss, default 1MB): Method frames, local vars
```

Errors:
- `OutOfMemoryError: Java heap space` → Increase `-Xmx`
- `OutOfMemoryError: Metaspace` → Increase `-XX:MaxMetaspaceSize`
- `StackOverflowError` → Increase `-Xss` or reduce recursion depth

*Q3: What is metaspace and how is it different from PermGen?*

A:

*PermGen (Java 7 and earlier):*
- Part of heap
- Fixed size (`-XX:MaxPermSize`)
- Stores: Class metadata, interned strings, static variables
- Common error: `OutOfMemoryError: PermGen space`

*Metaspace (Java 8+):*
- Native memory (not part of heap)
- Dynamically sized (grows automatically)
- Stores: Only class metadata
- Interned strings moved to heap
- Rarely see metaspace OOM

Benefits:
1. No more PermGen OOM for applications loading many classes
2. Better memory utilization (native memory auto-sized)
3. Simplified GC (no need to tune PermGen size)

Flags:
```bash
# PermGen (Java 7)
-XX:PermSize=128m -XX:MaxPermSize=256m

# Metaspace (Java 8+)
-XX:MetaspaceSize=128m -XX:MaxMetaspaceSize=256m
```

== Memory Management

=== Heap Structure

*Generational hypothesis:* Most objects die young.

*Heap layout (Generational GC):*
```
┌─────────────────────────────────────────────┐
│           Young Generation (1/3)            │
│  ┌──────┬──────┬──────┐                     │
│  │ Eden │  S0  │  S1  │                     │
│  └──────┴──────┴──────┘                     │
├─────────────────────────────────────────────┤
│           Old Generation (2/3)              │
│  (Tenured)                                  │
│                                             │
└─────────────────────────────────────────────┘
```

*Young generation (Eden + 2 Survivor spaces):*
- New objects allocated in Eden
- Minor GC: Copy live objects to Survivor (S0 ↔ S1)
- After N minor GCs (default 15), promote to Old Gen

*Old generation:*
- Long-lived objects
- Major GC: Full collection (slower)

*Allocation:*
1. Allocate in Eden (fast: bump-the-pointer)
2. Eden full → Minor GC (copy live to S0)
3. Next minor GC: Copy live from Eden + S0 → S1 (swap survivor spaces)
4. After 15 minor GCs → Promote to Old Gen
5. Old Gen full → Major GC (expensive!)

*Why two survivor spaces?*
- Avoid fragmentation (compacting copy)
- One always empty (copy target)

Example lifecycle:
```
new Object()          → Eden
Minor GC (age=1)      → S0
Minor GC (age=2)      → S1
Minor GC (age=3)      → S0
...
Minor GC (age=15)     → Old Gen
Major GC              → Freed (if unreachable)
```

=== Stack vs Heap

*Stack:*
- Primitives (local variables)
- Object references (pointer to heap)
- Method call frames
- LIFO (Last In First Out)
- Fast allocation (bump stack pointer)
- Automatic deallocation (method return)

*Heap:*
- Objects (instances)
- Arrays
- Shared across threads
- Slower allocation (find space, thread-safe)
- GC deallocates (when unreachable)

```java
void method() {
    int x = 10;              // Stack: primitive
    String s = new String("hello");
    // Stack: reference 's' (8 bytes)
    // Heap: String object (~40 bytes)

    int[] arr = new int[100];
    // Stack: reference 'arr' (8 bytes)
    // Heap: array object (400 bytes + overhead)
}
// Method returns: Stack frame popped
// Objects on heap eligible for GC
```

=== Reference Types

*Four types (from strong to weak):*

1. *Strong (default):*
```java
Object obj = new Object();  // Strong reference
// obj is NEVER garbage collected while reachable
```

2. *Soft:* GC'd before OutOfMemoryError (good for caches)
```java
SoftReference<byte[]> cache = new SoftReference<>(new byte[1024]);
byte[] data = cache.get();  // May return null if GC'd

// Use case: Cache that clears under memory pressure
Map<String, SoftReference<Image>> imageCache = new HashMap<>();
```

3. *Weak:* GC'd at next GC (good for metadata)
```java
WeakReference<Object> weak = new WeakReference<>(new Object());
Object obj = weak.get();  // May return null after GC

// Use case: WeakHashMap (auto-remove when key no longer used)
WeakHashMap<Key, Value> map = new WeakHashMap<>();
```

4. *Phantom:* Can't retrieve object, used for cleanup (advanced)
```java
PhantomReference<Object> phantom = new PhantomReference<>(obj, queue);
phantom.get();  // Always returns null

// Use case: More reliable than finalize() for cleanup
```

*Reference queue:* Notifies when reference is GC'd
```java
ReferenceQueue<Object> queue = new ReferenceQueue<>();
WeakReference<Object> ref = new WeakReference<>(obj, queue);

// When obj is GC'd, ref is enqueued
Reference<?> r = queue.poll();  // Get notification
```

=== Memory Leaks in Java

*Common causes:*

1. *Unclosed resources:*
```java
void leak() {
    FileInputStream fis = new FileInputStream("file.txt");
    // Forgot to close! File handle leaked
}

// Fix: Use try-with-resources
try (FileInputStream fis = new FileInputStream("file.txt")) {
    // ...
}
```

2. *Static collections:*
```java
class Cache {
    private static final List<Object> cache = new ArrayList<>();

    public void add(Object obj) {
        cache.add(obj);  // Never removed! Grows forever
    }
}

// Fix: Use weak references or eviction policy
private static final Map<Key, SoftReference<Value>> cache = new HashMap<>();
```

3. *ThreadLocal not removed:*
```java
ThreadLocal<byte[]> tl = new ThreadLocal<>();
tl.set(new byte[1024 * 1024]);  // 1 MB
// If thread lives forever (thread pool), memory leaked!

// Fix: Always remove
try {
    tl.set(data);
    // Use data
} finally {
    tl.remove();  // Important!
}
```

4. *Listeners not unregistered:*
```java
button.addActionListener(listener);
// Forgot to remove! button holds reference to listener

// Fix: Remove when done
button.removeActionListener(listener);
```

5. *Inner classes holding outer reference:*
```java
class Outer {
    private byte[] data = new byte[1024 * 1024];  // 1 MB

    class Inner {
        // Implicitly holds reference to Outer!
    }

    public Inner getInner() {
        return new Inner();
    }
}

Inner inner = new Outer().getInner();
// Outer can't be GC'd! Inner holds reference

// Fix: Use static nested class
static class Inner { }
```

=== Interview Questions: Memory Management

*Q1: How can memory leaks happen in Java? Give examples.*

A: Despite automatic GC, leaks occur when objects are reachable but not used.

Common causes:

*1. Static collections:*
```java
private static List<Object> cache = new ArrayList<>();
cache.add(obj);  // Never cleared → leak
```

*2. ThreadLocal:*
```java
ThreadLocal<Data> tl = new ThreadLocal<>();
tl.set(data);  // Never removed → leak (thread pool)
// Fix: tl.remove() in finally
```

*3. Unclosed resources:*
```java
Connection conn = getConnection();
// Forgot close() → leak
// Fix: try-with-resources
```

*4. Listeners:*
```java
source.addListener(listener);
// Forgot removeListener() → source holds listener
```

*5. Inner classes:*
```java
class Outer {
    byte[] data = new byte[1_000_000];
    class Inner { }  // Holds reference to Outer!
}
// Fix: static class Inner
```

Detection: Heap dump analysis (VisualVM, MAT), look for growing collections.

*Q2: What's the difference between stack and heap? What's stored in each?*

A:

*Stack (per-thread):*
- Local variables (primitives + references)
- Method call frames
- Fast (LIFO, no GC needed)
- Small (typically 1MB per thread)
- StackOverflowError if exceeded

*Heap (shared):*
- Objects, arrays
- Slower (GC overhead)
- Large (GB of memory)
- OutOfMemoryError if full

```java
void method() {
    int x = 10;        // Stack: 4 bytes
    Object o = new Object();
    // Stack: reference (8 bytes)
    // Heap: object (~16 bytes)
}
```

*Q3: Explain the generational hypothesis and how it influences GC design.*

A: Generational hypothesis: *Most objects die young*.

Empirical observation:
- 90%+ of objects die shortly after allocation
- Few objects survive long-term

GC design:
1. *Young generation*: Small, frequently collected (minor GC)
   - New objects allocated here
   - Fast collection (most already dead)

2. *Old generation*: Large, infrequently collected (major GC)
   - Only long-lived objects promoted here
   - Slow collection (but rare)

Benefits:
- Focus effort on young gen (most garbage)
- Avoid scanning old gen frequently (expensive)
- ~10x faster than scanning entire heap

Trade-off: Occasional full GC still needed (slow).

*Q4: When would you use WeakReference vs SoftReference?*

A:

*WeakReference:*
- GC'd at next collection (regardless of memory)
- Use for: Canonical mappings, metadata, event listeners

```java
WeakHashMap<Image, Metadata> metadata = new WeakHashMap<>();
// When Image no longer used, metadata auto-removed
```

*SoftReference:*
- GC'd only before OutOfMemoryError (memory-sensitive)
- Use for: Caches (keep if memory available, clear if needed)

```java
Map<String, SoftReference<Image>> cache = new HashMap<>();
// Images kept if memory available, cleared under pressure
```

*Decision:*
- Cache (keep as long as possible) → SoftReference
- Metadata (remove when key gone) → WeakReference

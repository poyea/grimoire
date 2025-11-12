= Part VI: Low-Latency Patterns

== Memory Management for Low Latency

=== Off-Heap Memory

*DirectByteBuffer:* Allocate outside JVM heap (no GC)

```java
// On-heap (GC managed)
ByteBuffer heap = ByteBuffer.allocate(1024);  // GC overhead

// Off-heap (no GC)
ByteBuffer direct = ByteBuffer.allocateDirect(1024);  // Native memory

// Access
direct.putInt(0, 42);
int value = direct.getInt(0);

// Must manually release (or wait for GC of buffer object)
// Use try-with-resources or finally
```

*Benefits:*
- No GC overhead (not in heap)
- Faster I/O (no copy to native memory)
- Persistent across GC

*Drawbacks:*
- Slower allocation (~100x than heap)
- Manual cleanup needed
- Counts toward MaxDirectMemorySize

*Use cases:*
- Network I/O (NIO channels)
- File-mapped buffers
- Long-lived buffers (> seconds)

*Example: Network I/O*
```java
ByteBuffer buffer = ByteBuffer.allocateDirect(8192);
SocketChannel channel = ...;

// Read without extra copy
channel.read(buffer);  // Direct transfer (OS → buffer, no heap copy)
```

=== Memory-Mapped Files

*MappedByteBuffer:* Map file to memory (OS manages)

```java
RandomAccessFile file = new RandomAccessFile("data.bin", "rw");
FileChannel channel = file.getChannel();

// Map file to memory (offset 0, size 1GB)
MappedByteBuffer buffer = channel.map(
    FileChannel.MapMode.READ_WRITE, 0, 1024 * 1024 * 1024);

// Access like array (OS handles paging)
buffer.putLong(0, 123456789L);
long value = buffer.getLong(0);  // Fast (no system call if in RAM)
```

*Benefits:*
- Very fast access (if in OS page cache)
- No GC (not in Java heap)
- Shared between processes
- OS handles paging (virtual memory)

*Use cases:*
- Large data files (GB+)
- Inter-process communication (IPC)
- Memory-mapped databases
- Log files (append-only)

=== Object Pooling & Reuse

*Problem:* Allocation causes GC pressure

*Solution:* Reuse objects instead of allocating new

```java
class ObjectPool<T> {
    private final Queue<T> pool = new ConcurrentLinkedQueue<>();
    private final Supplier<T> factory;
    private final int maxSize;

    public T acquire() {
        T obj = pool.poll();
        return obj != null ? obj : factory.get();
    }

    public void release(T obj) {
        reset(obj);  // Clear state
        if (pool.size() < maxSize) {
            pool.offer(obj);
        }
    }

    private void reset(T obj) {
        // Reset object state (clear fields, etc.)
    }
}

// Usage
ObjectPool<ByteBuffer> bufferPool = new ObjectPool<>(
    () -> ByteBuffer.allocate(8192), 100);

ByteBuffer buffer = bufferPool.acquire();
try {
    // Use buffer
} finally {
    bufferPool.release(buffer);  // Return to pool
}
```

*Trade-offs:*
- Pros: No allocation/GC overhead
- Cons: Memory usage (pooled objects not freed), complexity

*When to use:*
- High allocation rate (millions/sec)
- Short object lifetime (< seconds)
- Predictable pool size

=== Allocation-Free Coding Patterns

*Goal:* Zero allocations in hot path (no GC)

*Pattern 1: Primitive arrays (not objects)*
```java
// Bad: Allocates Integer objects
List<Integer> list = new ArrayList<>();
for (int i = 0; i < 1000; i++) {
    list.add(i);  // Autoboxing → allocation!
}

// Good: Primitive array
int[] array = new int[1000];
for (int i = 0; i < 1000; i++) {
    array[i] = i;  // No allocation
}
```

*Pattern 2: Reuse buffers*
```java
// Bad: Allocate new buffer each time
byte[] serialize(Object obj) {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();  // Allocation!
    ObjectOutputStream oos = new ObjectOutputStream(baos);
    oos.writeObject(obj);
    return baos.toByteArray();
}

// Good: Reuse buffer (thread-local)
private static final ThreadLocal<ByteBuffer> bufferTL =
    ThreadLocal.withInitial(() -> ByteBuffer.allocate(8192));

byte[] serialize(Object obj) {
    ByteBuffer buffer = bufferTL.get();
    buffer.clear();  // Reuse
    // Serialize into buffer
    return buffer.array();
}
```

*Pattern 3: Avoid String concat*
```java
// Bad: Creates many String objects
String result = "";
for (int i = 0; i < 100; i++) {
    result += i;  // O(n²) allocations!
}

// Good: StringBuilder (one allocation)
StringBuilder sb = new StringBuilder(1000);
for (int i = 0; i < 100; i++) {
    sb.append(i);
}
String result = sb.toString();
```

*Pattern 4: Object pools*
```java
// Bad: Allocate per request
void handle(Request req) {
    Response resp = new Response();  // Allocation per request
    // Process
}

// Good: Pool responses
void handle(Request req) {
    Response resp = responsePool.acquire();
    try {
        // Process (reuse response object)
    } finally {
        responsePool.release(resp);
    }
}
```

=== Interview Questions: Memory Management

*Q1: When would you use off-heap memory?*

A: Off-heap = allocate outside JVM heap (DirectByteBuffer, Unsafe)

*When to use:*
1. *Large, long-lived buffers*: Avoid GC overhead
   ```java
   ByteBuffer buffer = ByteBuffer.allocateDirect(1024 * 1024 * 1024);  // 1GB
   // No GC impact (not in heap)
   ```

2. *High-throughput I/O*: Avoid copy to native memory
   ```java
   channel.read(directBuffer);  // OS → buffer directly
   channel.read(heapBuffer);    // OS → temp → heap (extra copy!)
   ```

3. *Shared memory*: IPC via memory-mapped files

4. *GC-sensitive applications*: Reduce heap size, lower pause times

*Trade-offs:*
- Pros: No GC, faster I/O
- Cons: Slower allocation, manual cleanup, harder to debug

*Configuration:*
```bash
-XX:MaxDirectMemorySize=2g  # Limit off-heap allocation
```

*Q2: Explain object pooling and its trade-offs.*

A: Object pooling = reuse objects instead of allocate/GC.

*Pattern:*
```java
// Acquire from pool
ByteBuffer buffer = pool.acquire();
try {
    // Use buffer
} finally {
    pool.release(buffer);  // Return to pool
}
```

*Benefits:*
- No allocation overhead
- No GC pressure
- Predictable latency (no GC pauses)

*Drawbacks:*
- Memory usage (pooled objects not freed)
- Complexity (reset state, thread-safety)
- Risk of leaks (forgot to release)
- Contention (pool access)

*When to use:*
- Ultra-low latency (HFT, real-time systems)
- High allocation rate (millions/sec)
- Short-lived objects (< 1 second)

*Anti-pattern:*
- Low allocation rate (premature optimization)
- Long-lived objects (better to let GC handle)

== CPU Cache Optimization

=== False Sharing

*Problem:* Multiple threads modify different variables in same cache line → cache invalidation

*Cache line:* 64 bytes (x86). If two variables in same line, updates cause coherency traffic.

```java
// Bad: False sharing (x and y in same cache line)
class Counter {
    volatile long x;  // Offset 0
    volatile long y;  // Offset 8 (same cache line!)
}

Counter c = new Counter();

// Thread 1
while (true) { c.x++; }  // Invalidates cache line

// Thread 2
while (true) { c.y++; }  // Invalidates cache line (even though x and y independent!)

// Result: 10-100x slower due to cache coherency protocol (MESI)
```

*Solution: Pad to separate cache lines*

```java
// Good: Padding (x and y in different cache lines)
class Counter {
    volatile long x;
    long p1, p2, p3, p4, p5, p6, p7;  // 7*8 = 56 bytes padding
    volatile long y;  // Now in different cache line (64+ bytes away)
}

// Thread 1
while (true) { c.x++; }  // Updates only cache line containing x

// Thread 2
while (true) { c.y++; }  // Updates only cache line containing y (no interference!)

// Result: 10-100x faster
```

*Cache line size: 64 bytes (8 longs)*

=== \@Contended Annotation

*Java 8+:* `@Contended` annotation for automatic padding

```java
import jdk.internal.vm.annotation.Contended;

class Counter {
    @Contended
    volatile long x;  // JVM adds padding

    @Contended
    volatile long y;  // JVM adds padding (separate cache line)
}
```

*Enable:*
```bash
-XX:-RestrictContended  # Allow @Contended
```

*When to use:*
- High contention on independent variables
- Variables updated by different threads
- Performance-critical (measure first!)

*Example: LongAdder*
```java
// LongAdder uses \@Contended to avoid false sharing between cells
class LongAdder {
    @Contended
    static final class Cell {
        volatile long value;  // Each cell in separate cache line
    }
}
```

=== Data Structure Layout

*Goal:* Maximize cache locality (sequential access)

*Bad: Pointer chasing (random access)*
```java
class Node {
    int value;
    Node next;  // Pointer to next node (likely in different cache line)
}

// Traverse: Each node access → cache miss
Node n = head;
while (n != null) {
    sum += n.value;  // Cache miss (node in random location)
    n = n.next;
}
```

*Good: Array (sequential access)*
```java
int[] values = new int[1000];

// Traverse: Sequential access → cache hits
for (int i = 0; i < values.length; i++) {
    sum += values[i];  // Cache hit (prefetcher loads next cache lines)
}

// 10-100x faster due to cache locality!
```

*Structure of Arrays (SoA) vs Array of Structures (AoS):*

*AoS (poor locality if accessing single field):*
```java
class Point {
    float x, y, z;
}

Point[] points = new Point[1000];

// Sum x coordinates (y and z also loaded but not used)
for (int i = 0; i < points.length; i++) {
    sum += points[i].x;  // Loads x, y, z (wastes bandwidth)
}
```

*SoA (better locality):*
```java
class Points {
    float[] x = new float[1000];
    float[] y = new float[1000];
    float[] z = new float[1000];
}

// Sum x coordinates (only x array loaded)
for (int i = 0; i < x.length; i++) {
    sum += x[i];  // Sequential access, only x loaded
}
```

*When to use SoA:*
- Operate on single field across many elements
- SIMD vectorization (better alignment)

=== Sequential vs Random Access

*Memory hierarchy:*
```
L1 cache:  ~1ns,    32-64 KB
L2 cache:  ~3ns,    256 KB - 1 MB
L3 cache:  ~10ns,   8-64 MB (shared)
RAM:       ~100ns,  GB
```

*Sequential access:* Hardware prefetcher loads ahead
```java
int[] array = new int[1_000_000];

// Sequential: ~1-2ns per element (L1 hits after prefetch)
for (int i = 0; i < array.length; i++) {
    sum += array[i];
}
```

*Random access:* Cache misses
```java
// Random: ~100ns per element (RAM access)
for (int i = 0; i < array.length; i++) {
    int index = random.nextInt(array.length);
    sum += array[index];  // Cache miss (random location)
}
```

*Benchmark:*
```
Sequential access:  1 ns/element (L1 cache)
Random access:      100 ns/element (RAM) → 100x slower!
```

*Design for sequential access:*
- Use arrays (not linked lists)
- Process in order (not random)
- Batch operations (locality)

=== Interview Questions: CPU Cache

*Q1: What is false sharing and how do you prevent it?*

A: False sharing = threads modify different variables in same cache line → invalidation.

*Cause:*
- Cache line = 64 bytes (x86)
- Variables in same line → updates invalidate entire line
- MESI protocol forces coherency traffic

*Example:*
```java
class Bad {
    volatile long x;  // Offset 0-7
    volatile long y;  // Offset 8-15 (same 64-byte cache line!)
}

// Thread 1: Update x → invalidates cache line
// Thread 2: Update y → invalidates cache line (even though x and y independent!)
// Result: 10-100x slower
```

*Solutions:*

*1. Padding:*
```java
class Good {
    volatile long x;
    long p1, p2, p3, p4, p5, p6, p7;  // Padding (56 bytes)
    volatile long y;  // Different cache line (64+ bytes away)
}
```

*2. `@Contended`:*
```java
class Better {
    @Contended volatile long x;
    @Contended volatile long y;
}
// Enable: -XX:-RestrictContended
```

*Prevention:*
- Measure first (perf stat -d)
- Pad contended variables
- Group read-only data together

*Q2: Explain cache locality and its performance impact.*

A: Cache locality = data accessed together stored together.

*Memory hierarchy:*
```
L1:  ~1ns    (fast, small)
L2:  ~3ns
L3:  ~10ns
RAM: ~100ns  (slow, large)
```

*Sequential access:* Prefetcher loads ahead
```java
for (int i = 0; i < array.length; i++) {
    sum += array[i];  // Sequential → L1 hits (1ns)
}
```

*Random access:* Cache misses
```java
sum += array[random.nextInt(n)];  // Random → RAM access (100ns)
```

*Impact:* 100x difference!

*Optimization:*
1. *Array > LinkedList*: Sequential memory
2. *SoA > AoS*: Access single field
3. *Batch operations*: Process together

*Example:*
```java
// Bad: Pointer chasing
Node n = head;
while (n != null) {
    sum += n.value;  // Random memory → cache miss
    n = n.next;
}

// Good: Array
for (int i = 0; i < array.length; i++) {
    sum += array[i];  // Sequential → cache hit
}
// 10-100x faster!
```

== Lock-Free Data Structures

=== Disruptor Pattern (Ring Buffer)

*LMAX Disruptor:* Ultra-low latency queue (6M msgs/sec, single thread)

*Key ideas:*
1. *Ring buffer*: Preallocated array (no allocation)
2. *Cache-line padding*: Avoid false sharing
3. *Sequence numbers*: Lock-free (CAS)
4. *Wait strategies*: Busy-spin (lowest latency)

*Structure:*
```java
class RingBuffer<T> {
    private final T[] buffer;  // Preallocated (power of 2 size)
    private final int mask;    // size - 1 (for modulo)

    @Contended
    private volatile long writeSequence = -1;  // Producer cursor

    @Contended
    private volatile long readSequence = -1;   // Consumer cursor

    public void publish(T event) {
        long nextSeq = writeSequence + 1;
        // Wait for space (read caught up)
        while (nextSeq - readSequence > buffer.length) {
            // Busy-spin (lowest latency) or yield
        }
        buffer[(int) nextSeq & mask] = event;  // Write
        writeSequence = nextSeq;  // Publish (volatile write)
    }

    public T consume() {
        long nextSeq = readSequence + 1;
        // Wait for data (write ahead)
        while (nextSeq > writeSequence) {
            // Busy-spin or yield
        }
        T event = buffer[(int) nextSeq & mask];  // Read
        readSequence = nextSeq;  // Advance (volatile write)
        return event;
    }
}
```

*Benefits:*
- No allocation (preallocated buffer)
- No locks (CAS on sequence numbers)
- Cache-friendly (sequential access, padding)
- Mechanical sympathy (understand hardware)

*Use cases:*
- Ultra-low latency (HFT, gaming)
- High throughput (millions of events/sec)
- Producer-consumer pattern

=== LMAX Architecture Principles

*Principles:*

*1. Mechanical sympathy:* Understand hardware
- CPU cache lines (64 bytes)
- Cache hierarchy (L1/L2/L3)
- Branch prediction
- Memory barriers

*2. Single-writer principle:* One thread writes (no contention)
- Ring buffer: Single producer, single consumer
- No locks, no CAS contention

*3. Pre-allocation:* Avoid allocation in hot path
- Ring buffer: Preallocated array
- Object pooling: Reuse events

*4. Cache-line awareness:* Avoid false sharing
- Pad sequence numbers to separate cache lines
- `@Contended` annotation

*5. Wait strategies:* Trade latency vs CPU
- Busy-spin: Lowest latency, high CPU (100%)
- Yield: Low latency, medium CPU
- Block: High latency, low CPU

*Architecture:*
```
Input → Disruptor (Ring Buffer) → Business Logic → Output Disruptor → Network
        ↑                                           ↓
        Single producer thread                      Single consumer thread
```

=== Mechanical Sympathy

*Definition:* Design software with hardware in mind (Martin Thompson)

*Key concepts:*

*1. CPU cache:*
- Sequential access >> random access (100x)
- False sharing kills performance (10-100x)
- Pad to cache line boundaries

*2. Branch prediction:*
- Predictable branches fast (`<1` cycle)
- Misprediction expensive (~10-20 cycles)
- Avoid unpredictable branches in hot path

*3. Memory ordering:*
- Volatile/synchronized = memory barriers (expensive)
- Use minimally (e.g., Disruptor sequence numbers)

*4. SIMD:*
- Process multiple elements (4-8x speedup)
- Requires alignment, simple loops

*5. Allocation:*
- Heap allocation expensive (~10-100ns)
- GC pauses (milliseconds)
- Pre-allocate or pool objects

*Disruptor applies all:*
- Ring buffer: Sequential access (cache-friendly)
- Padding: Avoid false sharing
- CAS: Minimal synchronization
- Pre-allocation: No GC in hot path

=== Interview Questions: Lock-Free Structures

*Q1: Explain the Disruptor pattern and why it's fast.*

A: Disruptor = ultra-low latency queue (ring buffer + lock-free)

*Design:*
1. *Ring buffer*: Preallocated array (power-of-2 size)
2. *Sequence numbers*: Producer/consumer cursors (lock-free CAS)
3. *Cache-line padding*: Separate sequence numbers (`@Contended`)
4. *Busy-spin*: Lowest latency wait strategy

*Why fast:*
- *No allocation*: Preallocated → no GC
- *No locks*: CAS on sequences → no contention
- *Sequential access*: Ring buffer → cache-friendly
- *No false sharing*: Padding → no cache invalidation

*Performance:*
- 6M messages/sec (single thread)
- `<100ns` latency (99th percentile)

*Use case:* HFT, low-latency messaging

*Q2: What is mechanical sympathy?*

A: Mechanical sympathy = design software with hardware in mind

*Key principles:*

*1. CPU cache:*
- Use sequential access (arrays > linked lists)
- Avoid false sharing (pad to cache lines)

*2. Branch prediction:*
- Make branches predictable
- Avoid random conditionals in hot path

*3. Allocation:*
- Pre-allocate objects (avoid GC)
- Use primitives (not wrappers)

*4. Memory ordering:*
- Minimize volatile/synchronized (memory barriers expensive)
- Use CAS sparingly

*Example: Disruptor*
- Ring buffer: Sequential access (cache hits)
- Padding: No false sharing
- CAS: Minimal synchronization
- Pre-allocation: No GC

*Benefit:* 10-100x faster than naive implementations

== Latency Measurement

=== HdrHistogram

*Problem:* Coordinated omission (missing slow samples)

```java
// Bad: Coordinated omission
while (running) {
    long start = System.nanoTime();
    sendRequest();
    long latency = System.nanoTime() - start;
    histogram.record(latency);  // Misses queued requests!
}
```

*Solution: HdrHistogram*

```java
import org.HdrHistogram.Histogram;

Histogram histogram = new Histogram(3600_000_000_000L, 3);  // 1 hour, 3 digits

// Record latency
long start = System.nanoTime();
sendRequest();
long latency = System.nanoTime() - start;
histogram.recordValue(latency);

// Analyze
System.out.println("50th: " + histogram.getValueAtPercentile(50));
System.out.println("95th: " + histogram.getValueAtPercentile(95));
System.out.println("99th: " + histogram.getValueAtPercentile(99));
System.out.println("99.9th: " + histogram.getValueAtPercentile(99.9));
System.out.println("Max: " + histogram.getMaxValue());

// Output histogram
histogram.outputPercentileDistribution(System.out, 1000.0);  // In microseconds
```

*Correcting coordinated omission:*
```java
long expectedInterval = 1_000_000;  // 1ms expected interval

histogram.recordValueWithExpectedInterval(latency, expectedInterval);
// Fills in missing samples (if latency > interval)
```

=== Percentiles (p50, p95, p99, p999, p9999)

*Why percentiles (not average):*
- Average hides outliers
- Latency not normally distributed (long tail)

*Percentiles:*
- *p50 (median)*: 50% of requests faster
- *p95*: 95% faster (1 in 20 slower)
- *p99*: 99% faster (1 in 100 slower)
- *p999*: 99.9% faster (1 in 1000 slower)
- *p9999*: 99.99% faster (1 in 10000 slower)

*Example:*
```
p50:   10ms
p95:   25ms
p99:   50ms
p999:  200ms
p9999: 500ms
```

*Interpretation:*
- Most requests: `<10ms`
- 1 in 20: 25ms
- 1 in 100: 50ms
- 1 in 1000: 200ms (GC pause?)

*SLA:* Usually based on p99 or p999

=== Warmup & Steady-State Measurement

*JVM warmup:*
1. *Interpretation*: Slow (bytecode)
2. *C1 compilation*: Fast (basic optimizations)
3. *Profiling*: Collect data
4. *C2 compilation*: Optimized (aggressive)
5. *Steady state*: Peak performance

*Measurement phases:*

```java
// 1. Warmup phase (discard results)
for (int i = 0; i < 10000; i++) {
    benchmark();  // Let JIT compile
}

// 2. Measurement phase
Histogram histogram = new Histogram(...);
for (int i = 0; i < 100000; i++) {
    long start = System.nanoTime();
    benchmark();
    long latency = System.nanoTime() - start;
    histogram.recordValue(latency);
}

// 3. Report
histogram.outputPercentileDistribution(System.out, 1.0);
```

*JMH handles warmup automatically:*
```java
@Warmup(iterations = 5, time = 1)  // 5 iterations, 1 sec each
@Measurement(iterations = 10, time = 1)
@Benchmark
public void benchmark() {
    // Code to measure
}
```

=== Interview Questions: Latency

*Q1: What is coordinated omission and how do you avoid it?*

A: Coordinated omission = missing slow samples in latency measurement

*Cause:*
```java
// Sends requests serially
while (true) {
    long start = System.nanoTime();
    sendRequest();  // Blocks until response
    long latency = System.nanoTime() - start;
    histogram.record(latency);
}
```

*Problem:*
- If request slow (100ms), next request delayed
- Misses queued requests that would have experienced same latency
- Underestimates latency (optimistic)

*Solution 1: Fixed rate*
```java
ScheduledExecutorService scheduler = ...;
scheduler.scheduleAtFixedRate(() -> {
    long start = System.nanoTime();
    sendRequest();
    long latency = System.nanoTime() - start;
    histogram.recordValue(latency);
}, 0, 1, TimeUnit.MILLISECONDS);  // Send every 1ms
```

*Solution 2: HdrHistogram correction*
```java
long expectedInterval = 1_000_000;  // 1ms
histogram.recordValueWithExpectedInterval(latency, expectedInterval);
// Fills in missing samples
```

*Q2: Why are percentiles better than averages for latency?*

A:

*Average problems:*
1. Hides outliers (one 1s request + 999 1ms → avg 2ms, misleading!)
2. Not representative (most users don't experience average)
3. Skewed by long tail

*Percentiles:*
- *p50 (median)*: Typical user experience
- *p99*: 1 in 100 users (SLA target)
- *p999*: 1 in 1000 (detect rare issues)

*Example:*
```
100 requests:
99 requests: 10ms
1 request: 1000ms

Average: (99*10 + 1*1000) / 100 = 20ms (misleading!)
p50: 10ms (typical)
p99: 1000ms (shows outlier)
```

*SLA:* Use p99 or p999 (not average)
- "99% of requests < 50ms"

*Q3: How do you measure tail latency?*

A:

*Tools:*
1. *HdrHistogram*: High-resolution percentiles
```java
Histogram h = new Histogram(3600_000_000_000L, 3);
h.recordValue(latency);
long p999 = h.getValueAtPercentile(99.9);
long p9999 = h.getValueAtPercentile(99.99);
```

2. *JMH*: Microbenchmark framework
```java
@Benchmark
@BenchmarkMode(Mode.SampleTime)  // Measures latency distribution
public void test() { ... }
```

3. *Production monitoring*: Prometheus, Datadog, etc.

*Best practices:*
- Measure p99, p999, p9999 (not just average)
- Long measurement window (minutes, not seconds)
- Avoid coordinated omission (fixed rate)
- Separate warmup and measurement phases

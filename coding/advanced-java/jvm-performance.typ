= Part V: JVM Performance & Tuning

== Garbage Collection

=== GC Algorithms

*Serial GC:* Single-threaded, stop-the-world
```bash
-XX:+UseSerialGC
```
- Young: Single thread, copying
- Old: Single thread, mark-sweep-compact
- Use: Single-core, small heap (`<100MB`), batch jobs

*Parallel GC (Throughput):* Multi-threaded, stop-the-world
```bash
-XX:+UseParallelGC
```
- Young: Multiple threads, copying
- Old: Multiple threads, mark-sweep-compact
- Goal: Maximize throughput (minimize GC time %)
- Use: Batch processing, scientific computing

*G1 GC (Balanced):* Region-based, mostly concurrent
```bash
-XX:+UseG1GC (default since Java 9)
```
- Heap divided into regions (~2048 regions)
- Young + Old collected incrementally
- Goal: Predictable pause times
- Target: `-XX:MaxGCPauseMillis=200` (default 200ms)
- Use: Large heaps (>4GB), balanced latency/throughput

*ZGC (Low-latency):* Concurrent, colored pointers
```bash
-XX:+UseZGC
```
- Pause times `<10ms` (even for TB heaps!)
- Concurrent marking, compaction, relocation
- Uses colored pointers (metadata in pointer bits)
- Trade-off: Higher CPU usage (~15%)
- Use: Low-latency requirements (`<10ms` pauses)

*Shenandoah:* Concurrent, brooks pointers
```bash
-XX:+UseShenandoahGC
```
- Similar to ZGC (`<10ms` pauses)
- Uses indirection pointers (forwarding)
- Trade-off: Memory overhead (~10%)
- Use: Low-latency alternative to ZGC

*Comparison:*
```
Algorithm    Pause Time    Throughput    Heap Size    Use Case
Serial       100-1000ms    High          <100MB       Small apps
Parallel     100-1000ms    Highest       GB           Batch jobs
G1           10-200ms      Good          GB           General purpose
ZGC          <10ms         Good          TB           Low-latency
Shenandoah   <10ms         Good          GB-TB        Low-latency
```

=== Generational Hypothesis & Object Lifecycle

*Generational hypothesis:* Most objects die young (~90%).

*Object lifecycle:*
```
1. Allocate in Eden
2. Survive minor GC → S0 (age 1)
3. Survive minor GC → S1 (age 2)
4. ...
5. Survive minor GC (age 15) → Old Gen (tenured)
6. Major GC → Free (if unreachable)
```

*Young generation collection (minor GC):*
- Fast (~10-50ms)
- Frequent (Eden fills quickly)
- Copy live objects (Eden + S0 → S1, swap survivors)
- Dead objects implicitly collected (not copied)

*Old generation collection (major/full GC):*
- Slow (~100-1000ms)
- Infrequent (old gen fills slowly)
- Mark-sweep-compact (scan all live objects)
- Stop-the-world (application paused)

*Promotion:*
- Object survives 15 minor GCs → promoted to old gen
- Threshold: `-XX:MaxTenuringThreshold=15`
- Premature promotion: Large objects go directly to old gen

=== GC Tuning Flags

*Heap sizing:*
```bash
-Xms4g          # Initial heap size
-Xmx8g          # Maximum heap size
-Xmn2g          # Young generation size

# Or ratio-based:
-XX:NewRatio=2  # Old:Young = 2:1 (Young = 1/3 of heap)
```

*GC selection:*
```bash
-XX:+UseG1GC                # G1 (default Java 9+)
-XX:+UseZGC                 # ZGC (low-latency)
-XX:+UseParallelGC          # Parallel (throughput)
```

*G1 tuning:*
```bash
-XX:MaxGCPauseMillis=200    # Target max pause (default 200ms)
-XX:G1HeapRegionSize=16m    # Region size (1-32MB)
-XX:InitiatingHeapOccupancyPercent=45  # Start concurrent marking at 45%
```

*ZGC tuning:*
```bash
-XX:+UseZGC
-XX:ZCollectionInterval=5   # Min interval between GCs (seconds)
-XX:ZAllocationSpikeTolerance=2  # Handle allocation spikes
```

*GC logging:*
```bash
# Java 9+
-Xlog:gc*:file=gc.log:time,uptime,level,tags

# Java 8
-XX:+PrintGCDetails -XX:+PrintGCTimeStamps -Xloggc:gc.log
```

=== Low-Latency GC Strategies

*Goal:* Minimize GC pauses (`<10ms` for HFT)

*Strategy 1: Use ZGC/Shenandoah*
```bash
-XX:+UseZGC -Xmx16g -Xms16g
# Pauses <10ms even with 16GB heap
```

*Strategy 2: Reduce allocation rate*
- Object pooling (reuse objects)
- Primitive arrays (not wrapper objects)
- StringBuilder (not String concat)

*Strategy 3: Promote less to old gen*
- Increase young gen size
- Reduce object lifespan (short-lived OK)
- Avoid long-lived caches

*Strategy 4: Tune heap size*
```bash
# Set Xms == Xmx (avoid resize overhead)
-Xms8g -Xmx8g

# Increase young gen (more minor GCs, fewer major)
-Xmn4g
```

*Strategy 5: Off-heap memory*
- DirectByteBuffer (no GC)
- Memory-mapped files (OS manages)
- Unsafe allocations (advanced)

=== GC Logging & Analysis

*Enable logging:*
```bash
java -Xlog:gc*:file=gc.log:time,uptime,level,tags \
     -XX:+UseG1GC -Xmx4g MyApp
```

*Key metrics:*
- *Pause time*: Application stopped (minimize!)
- *Throughput*: % time not in GC (maximize)
- *Frequency*: How often GC runs
- *Heap usage*: Live data size

*Example log:*
```
[0.234s][info][gc] GC(0) Pause Young (Normal) 24M->3M(256M) 2.345ms
[0.456s][info][gc] GC(1) Pause Young (Normal) 27M->4M(256M) 2.891ms
[5.123s][info][gc] GC(15) Pause Full (Ergonomics) 200M->100M(256M) 345.678ms
```

*Analysis tools:*
- GCViewer: Visual analysis (pause time, throughput)
- GCeasy: Online analysis (upload log)
- jstat: Real-time monitoring

```bash
jstat -gcutil PID 1000  # GC stats every 1 second
```

=== Interview Questions: GC

*Q1: Explain how G1 GC works and when to use it.*

A: G1 (Garbage-First) divides heap into regions (~2048 equal-sized).

*Key features:*
1. *Region-based*: Heap = young regions + old regions + humongous (large objects)
2. *Incremental*: Collects regions with most garbage first
3. *Concurrent marking*: Mark live objects while app runs
4. *Predictable pauses*: Target pause time (`-XX:MaxGCPauseMillis=200`)

*Algorithm:*
1. Young GC: Evacuate young regions (stop-the-world, ~10-50ms)
2. Concurrent marking: Mark live objects in old gen (concurrent)
3. Mixed GC: Evacuate young + some old regions (incremental)
4. Full GC: Fallback if heap exhausted (avoid!)

*When to use:*
- Large heaps (>4GB)
- Predictable pause times required
- Balanced latency and throughput

*Configuration:*
```bash
-XX:+UseG1GC -Xmx8g -XX:MaxGCPauseMillis=100
```

*Q2: When would you use ZGC over G1?*

A:

*ZGC:*
- Ultra-low latency (`<10ms` pauses)
- Scales to TB heaps
- Concurrent (mark, relocate, compact)
- Trade-off: Higher CPU usage (~15%)

*G1:*
- Moderate latency (10-200ms pauses)
- Good throughput
- Simpler tuning

*Decision:*
- *ZGC*: Latency critical (`<10ms` requirement), large heap
  - Example: HFT, real-time pricing, online gaming
- *G1*: General purpose, balanced requirements

*Performance:*
```
Heap   G1 Pause    ZGC Pause
4GB    10-50ms     <5ms
16GB   50-100ms    <5ms
64GB   100-200ms   <10ms
```

*Q3: What is the generational hypothesis and how does it influence GC?*

A: *Generational hypothesis*: Most objects die young (~90% within seconds).

*Evidence:*
- Temporary objects (local variables, iterators)
- Short request processing
- Transient computation results

*GC design:*
1. *Separate young and old gen*
   - Young: Small, frequently collected
   - Old: Large, infrequently collected

2. *Minor GC (young):*
   - Fast (copy live objects only)
   - Frequent (Eden fills quickly)
   - Most objects already dead (efficient)

3. *Major GC (old):*
   - Slow (scan all live objects)
   - Rare (only long-lived objects)
   - Expensive but infrequent

*Performance benefit:*
- Minor GC: 90% of objects dead → only copy 10%
- Don't scan old gen frequently (expensive)
- 10x faster than scanning entire heap

*Q4: How do you diagnose performance issues in production Java app?*

A:

*Step 1: Identify bottleneck*
```bash
# CPU profiling
async-profiler -d 60 -f flamegraph.html PID

# Heap analysis
jmap -dump:live,format=b,file=heap.bin PID

# GC analysis
jstat -gcutil PID 1000
```

*Step 2: Analyze*

*High CPU:*
- Check flame graph (hot methods)
- Look for inefficient algorithms, excessive allocations

*High memory:*
- Heap dump analysis (MAT, VisualVM)
- Look for memory leaks (retained objects)

*Frequent GC:*
- GC logs (pause time, frequency)
- Reduce allocation rate or increase heap

*Long GC pauses:*
- Switch to low-latency GC (ZGC)
- Reduce live set (object pooling)

*Step 3: Fix*
- Optimize hot paths (inline, reduce allocations)
- Cache expensive computations
- Use primitives (not wrappers)
- Tune GC (heap size, collector)

*Tools:*
- JFR (Java Flight Recorder): Low-overhead profiling
- async-profiler: CPU/allocation profiling
- jstack: Thread dumps (deadlocks)
- MAT: Heap dump analysis (memory leaks)

== JIT Compilation

=== C1 vs C2 Compiler

*C1 (Client compiler):*
- Fast compilation (~10ms)
- Basic optimizations
- Lower peak performance
- Use: Short-running apps, startup time critical

*C2 (Server compiler):*
- Slow compilation (~100-1000ms)
- Aggressive optimizations
- Higher peak performance
- Use: Long-running apps, throughput critical

*Tiered compilation (default):*
```
Level 0: Interpreter
Level 1: C1 (no profiling)
Level 2: C1 (limited profiling)
Level 3: C1 (full profiling)
Level 4: C2 (optimized based on profiling)
```

*Flow:*
1. Interpret method (collect profile data)
2. Compile with C1 (fast, some optimizations)
3. Profile compiled code
4. Recompile with C2 (aggressive optimizations)

*Flags:*
```bash
-XX:+TieredCompilation     # Enable (default)
-XX:TieredStopAtLevel=1    # Stop at C1 (fast startup)
-XX:TieredStopAtLevel=4    # Use C2 (default)
-XX:CompileThreshold=10000 # C2 after 10K invocations
```

=== Inlining & Devirtualization

*Inlining:* Replace method call with method body

```java
// Before
int add(int a, int b) { return a + b; }
int result = add(x, y);

// After inlining
int result = x + y;  // No method call overhead!
```

*Benefits:*
- Eliminate call overhead (~5-10ns)
- Enable further optimizations (constant folding, etc.)

*Limits:*
```bash
-XX:MaxInlineSize=35         # Max method size to inline (bytes)
-XX:FreqInlineSize=325       # Hot method size limit
-XX:InlineSmallCode=1000     # Inline if compiled code < 1000 bytes
```

*Devirtualization:* Convert virtual call to direct call (then inline)

```java
Animal a = new Dog();  // Compiler proves a is always Dog
a.speak();  // Virtual call

// JIT sees only Dog instances
// Devirtualizes to:
Dog.speak();  // Direct call (can inline!)
```

*Speculative optimization:*
- Assume one implementation (monomorphic)
- Inline that implementation
- Insert guard (check assumption)
- Deoptimize if assumption breaks (bimorphic/polymorphic)

```java
// JIT sees only ArrayList so far
List<String> list = ... ;  // Runtime type: ArrayList
list.get(0);  // Inline ArrayList.get()

// Guard: if (list.getClass() == ArrayList.class) { inlined code } else { slow path }
```

=== Escape Analysis & Scalar Replacement

*Escape analysis:* Determine if object escapes method/thread.

*Three cases:*
1. *NoEscape:* Object local to method (optimize!)
2. *ArgEscape:* Passed to other method, but doesn't escape caller
3. *GlobalEscape:* Stored in field, returned, published

*Optimizations:*

*1. Scalar replacement:* Allocate object fields on stack (not heap)
```java
Point p = new Point(x, y);  // Object allocation
int sum = p.x + p.y;

// After scalar replacement (p doesn't escape):
int p_x = x;  // Scalars on stack
int p_y = y;
int sum = p_x + p_y;  // No heap allocation!
```

*2. Lock elision:* Remove synchronization on thread-local object
```java
StringBuffer sb = new StringBuffer();  // Thread-local
sb.append("a");  // synchronized
sb.append("b");  // synchronized

// After lock elision (sb doesn't escape):
// Remove synchronization (no other thread can access)
```

*3. Stack allocation:* Allocate object on stack (not heap)
- Avoided in HotSpot (scalar replacement instead)
- C2 uses scalar replacement for better optimization

*Enable:*
```bash
-XX:+DoEscapeAnalysis      # Enable (default)
-XX:+EliminateAllocations  # Scalar replacement (default)
-XX:+EliminateLocks        # Lock elision (default)
```

=== Loop Optimizations

*Loop unrolling:* Repeat loop body to reduce overhead

```java
// Before
for (int i = 0; i < 100; i++) {
    sum += array[i];
}

// After unrolling (factor 4)
for (int i = 0; i < 100; i += 4) {
    sum += array[i];
    sum += array[i+1];
    sum += array[i+2];
    sum += array[i+3];
}
// Benefits: Fewer branches, better instruction-level parallelism
```

*Loop vectorization (auto-vectorization):* Use SIMD instructions

```java
// Before
for (int i = 0; i < 1000; i++) {
    c[i] = a[i] + b[i];
}

// After vectorization (AVX: 8 ints at once)
for (int i = 0; i < 1000; i += 8) {
    __m256i va = _mm256_loadu_si256(&a[i]);
    __m256i vb = _mm256_loadu_si256(&b[i]);
    __m256i vc = _mm256_add_epi32(va, vb);
    _mm256_storeu_si256(&c[i], vc);
}
// 8x throughput!
```

*Loop invariant code motion:* Move constant computation out of loop

```java
// Before
for (int i = 0; i < n; i++) {
    int x = a * b;  // Constant within loop
    array[i] = x + i;
}

// After
int x = a * b;  // Moved outside loop
for (int i = 0; i < n; i++) {
    array[i] = x + i;
}
```

=== Intrinsics & CPU-Specific Optimizations

*Intrinsics:* JVM recognizes methods and replaces with optimized code

*Examples:*
```java
System.arraycopy()      → memcpy (native)
Math.sin/cos/sqrt()     → x87 FPU instructions
String.equals()         → SSE4.2 string compare
Arrays.equals()         → Vectorized compare
Integer.bitCount()      → POPCNT instruction (x86)
Long.numberOfLeadingZeros() → BSR instruction
```

*String operations (SSE4.2):*
```java
// JVM intrinsic for String.equals()
"hello".equals("world")  // Uses SSE4.2 PCMPESTRI (16 bytes at once)
```

*AES encryption (AES-NI):*
```java
Cipher cipher = Cipher.getInstance("AES");  // Uses AES-NI instructions
// 10x faster than software AES
```

*SHA hashing:*
```java
MessageDigest.getInstance("SHA-256")  // Uses SHA extensions (Intel/AMD)
```

*BigInteger:*
```java
BigInteger.multiply()  // Uses CPU multiply instructions (optimized)
```

*Check intrinsics:*
```bash
-XX:+PrintCompilation -XX:+PrintInlining
# Look for "intrinsic" in output
```

=== Interview Questions: JIT

*Q1: What is JIT compilation and how does it work?*

A: JIT (Just-In-Time) = compile bytecode to native code at runtime.

*How it works:*
1. *Interpreter*: Start with interpretation (collect profile data)
2. *Profile*: Track hot methods (frequently executed)
3. *Compile*: Compile hot methods to native code
   - C1 (fast): Quick compilation, basic optimizations
   - C2 (optimized): Aggressive optimizations
4. *Execute*: Run native code (10-100x faster than interpreter)
5. *Deoptimize*: Revert to interpreter if assumptions invalidated

*Tiered compilation:*
```
Interpreter → C1 (profile) → C2 (optimize)
```

*Benefits:*
- Fast startup (interpretation)
- Peak performance (native code)
- Profile-guided optimizations (runtime data)

*Q2: Explain escape analysis and its optimizations.*

A: Escape analysis = determine if object escapes method/thread.

*Optimizations:*

*1. Scalar replacement:* Object doesn't escape → fields on stack
```java
Point p = new Point(x, y);
int sum = p.x + p.y;
// Optimized to:
int p_x = x;  // Stack (no heap allocation)
int p_y = y;
int sum = p_x + p_y;
```

*2. Lock elision:* Thread-local object → remove synchronization
```java
StringBuffer sb = new StringBuffer();  // Thread-local
sb.append("a");  // synchronized (removed!)
```

*3. Stack allocation:* Allocate on stack (not heap)
- Faster allocation
- No GC overhead

*Impact:*
- Eliminate allocation overhead
- Reduce GC pressure
- Remove synchronization cost

*Example:*
```java
public int compute() {
    List<Integer> temp = new ArrayList<>();  // NoEscape
    temp.add(1);
    temp.add(2);
    return temp.get(0) + temp.get(1);
    // JIT: Scalar replace (no heap allocation!)
}
```

*Q3: How does the JIT compiler optimize loops?*

A:

*1. Loop unrolling:* Reduce branch overhead
```java
// Before
for (int i = 0; i < 100; i++) { sum += a[i]; }

// After (unroll factor 4)
for (int i = 0; i < 100; i += 4) {
    sum += a[i] + a[i+1] + a[i+2] + a[i+3];
}
```

*2. Loop vectorization:* SIMD instructions
```java
for (int i = 0; i < 1000; i++) {
    c[i] = a[i] + b[i];  // Processes 8 elements at once (AVX)
}
```

*3. Loop invariant code motion:* Hoist constant computation
```java
for (int i = 0; i < n; i++) {
    x = a * b;  // Move outside loop
    array[i] = x + i;
}
```

*4. Loop peeling:* Optimize first iteration separately
```java
// Handle first iteration (may eliminate null check in loop)
if (n > 0) { process(array[0]); }
for (int i = 1; i < n; i++) { process(array[i]); }
```

*Impact:* 2-10x speedup for loop-heavy code.

== JVM Diagnostics & Profiling

=== Critical JVM Flags

*Performance flags:*
```bash
# GC
-XX:+UseZGC                 # Low-latency GC
-Xms8g -Xmx8g               # Heap size (min == max)
-XX:MaxGCPauseMillis=100    # GC pause target

# JIT
-XX:+TieredCompilation      # C1 + C2 (default)
-XX:CompileThreshold=1500   # C2 threshold (lower = faster warmup)
-XX:+AggressiveOpts         # Experimental optimizations

# Allocation
-XX:+AlwaysPreTouch         # Touch heap pages at startup (avoid page faults)
-XX:+UseLargePages          # Use huge pages (reduce TLB misses)
```

*Diagnostic flags:*
```bash
# Unlock diagnostic options
-XX:+UnlockDiagnosticVMOptions

# Print compilation
-XX:+PrintCompilation       # Print JIT compilation
-XX:+PrintInlining          # Print inlining decisions

# Print assembly
-XX:+PrintAssembly          # Print generated assembly (requires hsdis)
-XX:CompileCommand=print,*ClassName.methodName
```

*Monitoring:*
```bash
# GC logging
-Xlog:gc*:file=gc.log

# JFR (low-overhead profiling)
-XX:+FlightRecorder
-XX:StartFlightRecording=duration=60s,filename=recording.jfr
```

=== JFR (Java Flight Recorder)

*Enable:*
```bash
java -XX:+FlightRecorder \
     -XX:StartFlightRecording=duration=60s,filename=app.jfr \
     MyApp
```

*Or runtime:*
```bash
jcmd PID JFR.start duration=60s filename=app.jfr
jcmd PID JFR.dump filename=app.jfr
jcmd PID JFR.stop
```

*Analyze:*
- JMC (JDK Mission Control): GUI for JFR files
- jfr: CLI tool (Java 11+)

*Data collected:*
- CPU usage (methods, threads)
- Allocations (where objects created)
- GC events (pause times, heap usage)
- Lock contention (synchronized blocks)
- I/O operations (file, network)

*Low overhead:* `<2%` typical (always-on in production)

=== async-profiler (Flame Graphs)

*CPU profiling:*
```bash
async-profiler -d 60 -f flamegraph.html PID
# Profile for 60 seconds, generate flame graph
```

*Allocation profiling:*
```bash
async-profiler -d 60 -e alloc -f flamegraph.html PID
# Where allocations happen
```

*Lock profiling:*
```bash
async-profiler -d 60 -e lock -f flamegraph.html PID
# Lock contention
```

*Reading flame graph:*
- X-axis: Proportion of samples (wider = more time)
- Y-axis: Call stack depth
- Color: Random (no meaning)
- Hover: See method + percentage

*Find hot spots:*
- Wide bars at top = CPU-intensive methods
- Optimize those first (biggest impact)

=== JMH (Java Microbenchmark Harness)

*Purpose:* Accurate microbenchmarking (avoids pitfalls)

*Example:*
```java
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
public class MyBenchmark {

    @Param({"10", "100", "1000"})
    int size;

    int[] array;

    @Setup
    public void setup() {
        array = new int[size];
        for (int i = 0; i < size; i++) {
            array[i] = i;
        }
    }

    @Benchmark
    public int sumLoop() {
        int sum = 0;
        for (int i = 0; i < array.length; i++) {
            sum += array[i];
        }
        return sum;
    }

    @Benchmark
    public int sumStream() {
        return Arrays.stream(array).sum();
    }
}
```

*Run:*
```bash
mvn clean install
java -jar target/benchmarks.jar
```

*JMH handles:*
- Warmup (JIT compilation)
- Dead code elimination
- Constant folding
- Statistical analysis

=== PrintCompilation, PrintInlining, PrintAssembly

*PrintCompilation:*
```bash
-XX:+PrintCompilation
```

*Output:*
```
115   1       3       java.lang.String::charAt (29 bytes)
120   2       3       java.lang.String::length (6 bytes)
125   3       4       java.lang.String::charAt (29 bytes)  # C2 compilation
```

*Format:* `timestamp id tier method (size) status`

*PrintInlining:*
```bash
-XX:+PrintInlining -XX:+UnlockDiagnosticVMOptions
```

*Output:*
```
@ 10   java.lang.String::length (6 bytes)   inline (hot)
@ 15   java.lang.Math::min (10 bytes)       inline
@ 20   com.example.Helper::compute (50 bytes)   too big
```

*PrintAssembly:*
```bash
-XX:+PrintAssembly -XX:CompileCommand=print,*MyClass.myMethod
# Requires hsdis library
```

*Output:* Generated assembly code
```assembly
mov rax, [rbp+0x10]
add rax, [rbp+0x18]
ret
```

*Use:* Verify JIT optimizations (inlining, vectorization, etc.)

=== Interview Questions: Diagnostics

*Q1: How do you diagnose performance issues in production?*

A:

*Step-by-step:*

*1. Identify symptoms:*
- High CPU → CPU profiling
- High memory → Heap dump
- Slow response → Thread dumps
- Frequent GC → GC logs

*2. Tools:*
```bash
# CPU profiling
async-profiler -d 60 -f flamegraph.html PID

# Memory
jmap -dump:live,format=b,file=heap.bin PID
# Analyze with MAT (Eclipse Memory Analyzer)

# Threads
jstack PID > threads.txt

# GC
jstat -gcutil PID 1000  # Every 1 second
```

*3. Analyze:*
- Flame graph → hot methods (optimize)
- Heap dump → memory leaks (fix)
- Thread dump → deadlocks, blocked threads
- GC logs → pause times (tune GC)

*4. Fix:*
- Optimize hot paths (reduce allocations, cache)
- Fix memory leaks (remove references)
- Tune GC (collector, heap size)
- Use profiler-guided optimization

*Q2: What tools would you use to profile a Java application?*

A:

*Production (low overhead):*
1. *JFR (Java Flight Recorder)*: ~1% overhead, always-on
   - CPU, allocations, GC, locks, I/O
   ```bash
   jcmd PID JFR.start duration=60s filename=app.jfr
   ```

2. *async-profiler*: Sampling profiler, `<5%` overhead
   - CPU flame graphs, allocation profiling
   ```bash
   async-profiler -d 60 -f flamegraph.html PID
   ```

3. *jstat*: GC monitoring, real-time
   ```bash
   jstat -gcutil PID 1000
   ```

*Development (higher overhead OK):*
4. *VisualVM*: GUI, heap/thread dumps, sampling
5. *YourKit*: Commercial, detailed profiling
6. *JProfiler*: Commercial, CPU/memory/threads

*Microbenchmarking:*
7. *JMH*: Accurate microbenchmarks
   - Avoids JIT pitfalls (warmup, dead code elimination)

*Decision:*
- Production → JFR, async-profiler (low overhead)
- Development → VisualVM, YourKit (detailed)
- Benchmarking → JMH (accurate)

*Q3: How do you avoid microbenchmark pitfalls?*

A:

*Common pitfalls:*

*1. No warmup:*
```java
// Wrong: Cold start
long start = System.nanoTime();
method();  // Not JIT-compiled yet!
long time = System.nanoTime() - start;
```

*2. Dead code elimination:*
```java
// Wrong: JIT removes entire benchmark!
int sum = 0;
for (int i = 0; i < 1000; i++) {
    sum += i;
}
// sum not used → JIT eliminates loop!
```

*3. Constant folding:*
```java
// Wrong: JIT computes at compile time
int result = 2 + 3;  // Becomes: int result = 5;
```

*4. Non-steady state:*
- JIT compilation during measurement
- GC during measurement

*Solution: Use JMH*
```java
@Benchmark
public int sum(Blackhole bh) {  // Blackhole prevents DCE
    int sum = 0;
    for (int i = 0; i < 1000; i++) {
        sum += i;
    }
    return sum;  // JMH consumes result (no DCE)
}
```

*JMH handles:*
- Warmup iterations (JIT warmup)
- Blackhole (prevent dead code elimination)
- State objects (prevent constant folding)
- Statistical analysis (report p-values)

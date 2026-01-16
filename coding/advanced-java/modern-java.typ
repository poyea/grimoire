= Part VII: Modern Java Features (Java 8-21)

== Functional Programming

=== Lambdas & Method References

*Lambda syntax:*
```java
// Before Java 8
Runnable r = new Runnable() {
    @Override
    public void run() {
        System.out.println("Running");
    }
};

// Java 8+ Lambda
Runnable r = () -> System.out.println("Running");

// With parameters
BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;

// Block lambda
Consumer<String> print = (s) -> {
    System.out.println("Processing: " + s);
    // Multiple statements
};
```

*Method references:*
```java
// Static method reference
Function<String, Integer> parser = Integer::parseInt;
// Equivalent to: s -> Integer.parseInt(s)

// Instance method reference
String str = "hello";
Supplier<Integer> lengthGetter = str::length;
// Equivalent to: () -> str.length()

// Arbitrary object method reference
Function<String, Integer> lengthFunc = String::length;
// Equivalent to: s -> s.length()

// Constructor reference
Supplier<List<String>> listFactory = ArrayList::new;
// Equivalent to: () -> new ArrayList<>()

BiFunction<Integer, Integer, Point> pointFactory = Point::new;
// Equivalent to: (x, y) -> new Point(x, y)
```

*Functional interfaces (single abstract method):*
```java
@FunctionalInterface
interface Processor<T> {
    void process(T t);

    // Default methods allowed
    default void preProcess() { }
}

// Common functional interfaces:
Predicate<T>          // T → boolean
Function<T, R>        // T → R
Consumer<T>           // T → void
Supplier<T>           // () → T
BiFunction<T, U, R>   // (T, U) → R
UnaryOperator<T>      // T → T
BinaryOperator<T>     // (T, T) → T
```

=== Streams API

*Stream pipeline:* source → intermediate ops → terminal op

```java
List<String> names = Arrays.asList("Alice", "Bob", "Charlie", "David");

// Filter, map, collect
List<String> result = names.stream()
    .filter(s -> s.length() > 3)        // Intermediate: keep if length > 3
    .map(String::toUpperCase)           // Intermediate: convert to uppercase
    .collect(Collectors.toList());      // Terminal: collect to list

// Count
long count = names.stream()
    .filter(s -> s.startsWith("A"))
    .count();

// Reduce
int sum = IntStream.range(1, 101)
    .reduce(0, (a, b) -> a + b);  // Sum 1 to 100

// Group by
Map<Integer, List<String>> byLength = names.stream()
    .collect(Collectors.groupingBy(String::length));
```

*Intermediate operations (lazy):*
- `filter()`, `map()`, `flatMap()`, `sorted()`, `distinct()`, `limit()`, `skip()`

*Terminal operations (triggers execution):*
- `collect()`, `forEach()`, `reduce()`, `count()`, `anyMatch()`, `allMatch()`, `findFirst()`

*Short-circuiting:*
```java
// findFirst() stops after finding first match
Optional<String> first = names.stream()
    .filter(s -> s.startsWith("B"))
    .findFirst();  // Stops after "Bob" (doesn't process "Charlie", "David")
```

=== Performance Characteristics

*Streams overhead:*
- Allocation: Stream object, spliterator, internal iterators
- Indirection: Lambda calls (virtual dispatch until JIT inlines)
- Autoboxing: Primitive streams help but not always used

*Benchmark:*
```java
int[] array = new int[1_000_000];

// For loop: ~500 μs
int sum = 0;
for (int i = 0; i < array.length; i++) {
    sum += array[i];
}

// Stream: ~700 μs (40% slower, cold start)
int sum = Arrays.stream(array).sum();

// After JIT warmup: ~500 μs (same speed)
```

*When streams slower:*
1. Small collections (`<100` elements) → overhead dominates
2. Simple operations (single filter/map) → for loop simpler
3. Primitive operations with autoboxing → use IntStream/LongStream/DoubleStream

*When streams faster/better:*
1. Complex pipelines (multiple operations) → more readable
2. Parallel operations → automatic parallelization
3. Large collections + parallelizable operations

=== Primitive Streams

*Problem: Autoboxing overhead*
```java
// Bad: Stream<Integer> (autoboxing)
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
int sum = numbers.stream()
    .reduce(0, (a, b) -> a + b);  // Boxes/unboxes each element!
```

*Solution: IntStream/LongStream/DoubleStream*
```java
// Good: IntStream (no boxing)
int sum = IntStream.range(1, 6).sum();  // Primitive operations

// Convert to primitive stream
int sum = numbers.stream()
    .mapToInt(Integer::intValue)  // Convert to IntStream
    .sum();

// Primitive operations
IntSummaryStatistics stats = IntStream.range(1, 101)
    .summaryStatistics();
System.out.println("Average: " + stats.getAverage());
System.out.println("Max: " + stats.getMax());
```

*Performance:*
```java
// Stream<Integer>: ~1000 μs (autoboxing)
Stream.of(1, 2, 3, ..., 1000).reduce(0, Integer::sum);

// IntStream: ~100 μs (10x faster)
IntStream.range(1, 1001).sum();
```

=== Parallel Streams

*Automatic parallelization:*
```java
// Sequential
long count = list.stream()
    .filter(s -> s.length() > 5)
    .count();

// Parallel (uses ForkJoinPool)
long count = list.parallelStream()
    .filter(s -> s.length() > 5)
    .count();
```

*Pitfalls:*

*1. Shared mutable state:*
```java
// Wrong: Race condition
List<Integer> results = new ArrayList<>();
IntStream.range(0, 1000).parallel()
    .forEach(i -> results.add(i));  // Not thread-safe!

// Right: Use collect
List<Integer> results = IntStream.range(0, 1000).parallel()
    .boxed()
    .collect(Collectors.toList());  // Thread-safe collector
```

*2. Wrong workload:*
```java
// Bad: I/O bound (blocking wastes threads)
list.parallelStream()
    .map(id -> database.query(id))  // Blocking I/O (wrong for parallel)
    .collect(Collectors.toList());

// Good: CPU bound
list.parallelStream()
    .map(this::expensiveComputation)  // CPU-intensive (benefits from parallel)
    .collect(Collectors.toList());
```

*3. Overhead:*
```java
// Small data: Overhead > benefit
List.of(1, 2, 3).parallelStream().sum();  // Slower than sequential!

// Large data: Benefit > overhead
IntStream.range(0, 10_000_000).parallel().sum();  // Faster
```

*When to use parallel streams:*
- Large datasets (>10K elements)
- CPU-intensive operations (not I/O)
- Stateless, independent operations
- Associative operations (order doesn't matter)

=== Interview Questions: Functional Programming

*Q1: When should you avoid streams?*

A:

*Avoid streams when:*

*1. Small collections:*
```java
// Overhead > benefit for 5 elements
List.of(1, 2, 3, 4, 5).stream().filter(...).collect(...);
// Better: for loop
```

*2. Simple operations:*
```java
// Single operation → for loop simpler/faster
for (String s : list) {
    if (s.startsWith("A")) { return s; }
}
```

*3. Debugging needed:*
- Streams harder to debug (no breakpoints inside lambda)
- For loop: Step through each iteration

*4. Early termination complex:*
```java
// Multiple break conditions → for loop clearer
for (Item item : items) {
    if (condition1) break;
    if (condition2) break;
}
```

*5. Index needed:*
```java
// Need index → for loop natural
for (int i = 0; i < list.size(); i++) {
    process(list.get(i), i);
}
```

*Use streams for:*
- Large collections
- Complex pipelines (multiple operations)
- Parallel processing
- Readability (declarative style)

*Q2: Parallel stream pitfalls?*

A:

*Pitfall 1: Shared mutable state*
```java
// Wrong: Race condition
List<Integer> results = new ArrayList<>();
stream.parallel().forEach(x -> results.add(x));  // Not thread-safe!

// Right: Thread-safe collector
List<Integer> results = stream.parallel().collect(Collectors.toList());
```

*Pitfall 2: Blocking operations*
```java
// Wrong: I/O blocks threads
stream.parallel().map(id -> db.query(id));  // Wastes threads on blocking

// Right: Use CompletableFuture for I/O
```

*Pitfall 3: Small data*
```java
// Overhead > benefit
List.of(1, 2, 3).parallelStream();  // Slower than sequential!
```

*Pitfall 4: Order-dependent operations*
```java
// findFirst() on parallel stream: Less efficient (must coordinate threads)
// Use findAny() if order doesn't matter
```

*Pitfall 5: False sharing*
- Multiple threads updating nearby memory locations
- Use primitive arrays or careful data layout

*When to use:*
- Large data (>10K elements)
- CPU-intensive operations
- Stateless, independent operations
- No shared mutable state

*Q3: What's the overhead of lambdas?*

A:

*Overhead sources:*

*1. Allocation:*
- Non-capturing lambda: Singleton (zero allocation)
- Capturing lambda: Allocates closure object

```java
// Non-capturing: No allocation (singleton)
Function<String, Integer> f = String::length;

// Capturing: Allocates closure (captures x)
int x = 10;
Function<Integer, Integer> f = y -> x + y;  // Allocates
```

*2. Invocation:*
- Initially: Virtual call (~2-5ns overhead)
- After JIT: Inlined (zero overhead)

```java
// Cold start: Virtual call
list.forEach(x -> process(x));  // ~5ns overhead per call

// After warmup: Inlined
// JIT inlines lambda body → no overhead
```

*Benchmark:*
```
For loop:        100 ns (baseline)
Lambda (cold):   150 ns (+50% slower, virtual call)
Lambda (warm):   100 ns (same speed, inlined)
```

*Minimizing overhead:*
1. Use method references (simpler for JIT)
2. Avoid capturing variables (allocation)
3. Warmup (let JIT compile)
4. Primitive streams (avoid autoboxing)

*Conclusion:* After warmup, overhead negligible for most code.

== Asynchronous Programming

=== CompletableFuture

*Basic usage:*
```java
// Async computation
CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> {
    // Runs in ForkJoinPool.commonPool()
    return expensiveComputation();
});

// Block and get result
String result = future.get();  // Blocks until complete

// Or with timeout
String result = future.get(1, TimeUnit.SECONDS);  // Timeout after 1 second
```

*Composition:*
```java
CompletableFuture<Integer> future = CompletableFuture
    .supplyAsync(() -> fetchUserId())
    .thenApply(id -> fetchUserName(id))        // Transform result
    .thenApply(name -> name.toUpperCase())
    .exceptionally(ex -> "Unknown");           // Handle exception

// Non-blocking
future.thenAccept(name -> System.out.println(name));

// Combine two futures
CompletableFuture<Integer> f1 = CompletableFuture.supplyAsync(() -> 10);
CompletableFuture<Integer> f2 = CompletableFuture.supplyAsync(() -> 20);

CompletableFuture<Integer> combined = f1.thenCombine(f2, (a, b) -> a + b);
```

*Parallel execution:*
```java
List<CompletableFuture<String>> futures = users.stream()
    .map(user -> CompletableFuture.supplyAsync(() -> fetchData(user)))
    .collect(Collectors.toList());

// Wait for all to complete
CompletableFuture<Void> allOf = CompletableFuture.allOf(
    futures.toArray(new CompletableFuture[0]));

allOf.join();  // Block until all complete

// Collect results
List<String> results = futures.stream()
    .map(CompletableFuture::join)
    .collect(Collectors.toList());
```

*Exception handling:*
```java
CompletableFuture<String> future = CompletableFuture
    .supplyAsync(() -> {
        if (condition) throw new RuntimeException("Error");
        return "Success";
    })
    .exceptionally(ex -> {
        log.error("Failed", ex);
        return "Fallback";  // Recover from exception
    })
    .thenApply(result -> result.toUpperCase());

// Or handle both success and failure
future.handle((result, ex) -> {
    if (ex != null) {
        return "Error: " + ex.getMessage();
    } else {
        return "Success: " + result;
    }
});
```

=== Async vs Traditional Threading

*Traditional:*
```java
ExecutorService executor = Executors.newFixedThreadPool(10);

Future<String> future = executor.submit(() -> {
    return expensiveComputation();
});

String result = future.get();  // Blocks (no composition)
```

*CompletableFuture:*
```java
CompletableFuture<String> future = CompletableFuture
    .supplyAsync(() -> expensiveComputation())
    .thenApply(result -> transform(result))  // Composition
    .thenAccept(System.out::println);  // Non-blocking callback

// No blocking (completely asynchronous)
```

*Benefits of CompletableFuture:*
1. *Composition*: Chain operations (thenApply, thenCompose)
2. *Non-blocking*: Callbacks (thenAccept, thenRun)
3. *Exception handling*: exceptionally, handle
4. *Combination*: thenCombine, allOf, anyOf

=== Interview Questions: Async

*Q1: How does CompletableFuture work?*

A: CompletableFuture = async computation with composition.

*Basic pattern:*
```java
// 1. Start async task
CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> {
    // Runs in thread pool (ForkJoinPool.commonPool)
    return compute();
});

// 2. Transform result (non-blocking)
future.thenApply(result -> transform(result));

// 3. Consume result (non-blocking)
future.thenAccept(result -> System.out.println(result));

// Or block
String result = future.get();
```

*Composition:*
```java
CompletableFuture.supplyAsync(() -> fetchUser(id))
    .thenApply(user -> user.getName())       // Transform
    .thenCompose(name -> fetchAddress(name)) // Flatten nested future
    .thenAccept(addr -> print(addr))         // Consume
    .exceptionally(ex -> { log(ex); return null; });  // Handle error
```

*Thread pool:*
- Default: ForkJoinPool.commonPool()
- Custom: Pass executor to supplyAsync()

*Benefits:*
- Non-blocking (callbacks)
- Composable (chain operations)
- Exception handling (exceptionally)

*Q2: Explain thenApply vs thenCompose.*

A:

*thenApply:* Transform result (1:1 mapping)
```java
CompletableFuture<Integer> f1 = CompletableFuture.supplyAsync(() -> 10);

CompletableFuture<Integer> f2 = f1.thenApply(x -> x * 2);  // 10 → 20
// Type: CompletableFuture<T> → CompletableFuture<U>
```

*thenCompose:* Flatten nested future (flatMap)
```java
CompletableFuture<Integer> f1 = CompletableFuture.supplyAsync(() -> 10);

CompletableFuture<Integer> f2 = f1.thenCompose(x -> {
    return CompletableFuture.supplyAsync(() -> x * 2);  // Returns future
});
// Type: CompletableFuture<T> → CompletableFuture<U>
//       (not CompletableFuture<CompletableFuture<U>>)
```

*Analogy to Streams:*
- `map()` ↔ `thenApply()` (1:1 transform)
- `flatMap()` ↔ `thenCompose()` (flatten)

*Example:*
```java
// thenApply: Simple transform
future.thenApply(user -> user.getName());  // String

// thenCompose: Async transform
future.thenCompose(user -> fetchAddress(user));  // Async operation
```

== Recent Features (Java 9-21)

=== Records (Java 14+)

*Problem:* Boilerplate for data classes

```java
// Old: Boilerplate
public final class Point {
    private final int x;
    private final int y;

    public Point(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public int x() { return x; }
    public int y() { return y; }

    @Override
    public boolean equals(Object o) { /* ... */ }

    @Override
    public int hashCode() { /* ... */ }

    @Override
    public String toString() { /* ... */ }
}
```

*New: Record*
```java
public record Point(int x, int y) { }

// Generates:
// - Constructor: Point(int x, int y)
// - Accessors: x(), y()
// - equals(), hashCode(), toString()
// - Final class, final fields
```

*Custom methods:*
```java
public record Point(int x, int y) {
    // Custom constructor
    public Point {
        if (x < 0 || y < 0) {
            throw new IllegalArgumentException("Negative coordinates");
        }
    }

    // Custom method
    public double distanceFromOrigin() {
        return Math.sqrt(x * x + y * y);
    }

    // Static factory
    public static Point origin() {
        return new Point(0, 0);
    }
}
```

*When to use:*
- Immutable data carriers
- DTOs (Data Transfer Objects)
- Value objects
- Return types for multiple values

=== Sealed Classes (Java 17+)

*Purpose:* Restrict which classes can extend/implement

```java
// Only Circle, Rectangle can extend Shape
public sealed class Shape permits Circle, Rectangle { }

public final class Circle extends Shape {
    private final double radius;
    // ...
}

public final class Rectangle extends Shape {
    private final double width, height;
    // ...
}

// Compile error: Square not permitted
public final class Square extends Shape { }  // Error!
```

*With records:*
```java
public sealed interface Expr permits ConstExpr, AddExpr, MulExpr { }

public record ConstExpr(int value) implements Expr { }
public record AddExpr(Expr left, Expr right) implements Expr { }
public record MulExpr(Expr left, Expr right) implements Expr { }
```

*Pattern matching:*
```java
int eval(Expr expr) {
    return switch (expr) {
        case ConstExpr(int value) -> value;
        case AddExpr(Expr left, Expr right) -> eval(left) + eval(right);
        case MulExpr(Expr left, Expr right) -> eval(left) * eval(right);
        // No default needed (compiler knows all cases)
    };
}
```

*Benefits:*
- Compiler ensures exhaustiveness (no missing cases)
- Domain modeling (algebraic data types)
- Safer than inheritance (controlled hierarchy)

=== Pattern Matching & Switch Expressions (Java 17+)

*Old switch:*
```java
String result;
switch (day) {
    case MONDAY:
    case TUESDAY:
        result = "Weekday";
        break;
    case SATURDAY:
    case SUNDAY:
        result = "Weekend";
        break;
    default:
        result = "Unknown";
}
```

*New switch expression:*
```java
String result = switch (day) {
    case MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY -> "Weekday";
    case SATURDAY, SUNDAY -> "Weekend";
};  // No default needed if exhaustive
```

*Pattern matching for switch:*
```java
String describe(Object obj) {
    return switch (obj) {
        case String s -> "String of length " + s.length();
        case Integer i -> "Integer: " + i;
        case Long l -> "Long: " + l;
        case null -> "null";
        default -> "Unknown";
    };
}
```

*Guarded patterns:*
```java
String classify(int n) {
    return switch (n) {
        case int i when i < 0 -> "Negative";
        case int i when i == 0 -> "Zero";
        case int i when i > 0 -> "Positive";
    };
}
```

=== Virtual Threads (Java 21+, Project Loom)

*Problem:* Platform threads expensive, limited scalability

*Solution:* Virtual threads (cheap, millions possible)

```java
// Old: Platform thread (expensive)
Thread thread = new Thread(() -> {
    // Task
});
thread.start();

// New: Virtual thread (cheap)
Thread vThread = Thread.startVirtualThread(() -> {
    // Task (can create millions!)
});

// Or with executor
try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
    executor.submit(() -> {
        // Each task gets own virtual thread
    });
}
```

*Blocking is OK:*
```java
// Old: Blocking wastes platform thread
Future<String> future = executor.submit(() -> {
    Thread.sleep(1000);  // Wastes expensive platform thread!
    return "Result";
});

// New: Blocking doesn't waste carrier
executor.submit(() -> {
    Thread.sleep(1000);  // Virtual thread parks, carrier freed
    return "Result";
});
```

*Structured concurrency (Preview):*
```java
void handle() throws InterruptedException {
    try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
        Future<String> user = scope.fork(() -> fetchUser());
        Future<String> order = scope.fork(() -> fetchOrder());

        scope.join();           // Wait for all
        scope.throwIfFailed();  // Propagate exceptions

        // Both completed successfully
        process(user.resultNow(), order.resultNow());
    }
}
```

*Impact:*
- Simplifies concurrent I/O (no async/await complexity)
- Scales to millions of connections
- Write blocking-style code, get async performance

=== Interview Questions: Modern Features

*Q1: When would you use records?*

A: Records = immutable data carriers (Java 14+)

*Use when:*
1. *Immutable DTOs:*
```java
public record UserDTO(String name, int age, String email) { }
```

2. *Multiple return values:*
```java
public record Result(int value, String message) { }

public Result compute() {
    return new Result(42, "Success");
}
```

3. *Value objects:*
```java
public record Point(int x, int y) { }
public record Money(BigDecimal amount, String currency) { }
```

*Don't use when:*
- Need mutability (use regular class)
- Complex behavior (records for data, not behavior)
- Need inheritance (records are final)

*Benefits:*
- Less boilerplate (no getters, equals, hashCode, toString)
- Immutable by default (thread-safe)
- Clear intent (data carrier)

*Q2: What are sealed classes and when to use them?*

A: Sealed classes = restrict which classes can extend/implement

```java
public sealed interface Expr permits Const, Add, Mul { }

record Const(int value) implements Expr { }
record Add(Expr left, Expr right) implements Expr { }
record Mul(Expr left, Expr right) implements Expr { }
```

*Benefits:*

*1. Exhaustiveness checking:*
```java
int eval(Expr expr) {
    return switch (expr) {
        case Const(int v) -> v;
        case Add(Expr l, Expr r) -> eval(l) + eval(r);
        case Mul(Expr l, Expr r) -> eval(l) * eval(r);
        // No default needed! Compiler knows all cases
    };
}
```

*2. Domain modeling:*
- Payment: Cash, CreditCard, PayPal (sealed)
- Shape: Circle, Rectangle, Triangle (sealed)
- Result: Success, Failure (sealed)

*3. Safety:*
- Controlled hierarchy (no unknown subclasses)
- Compiler-enforced exhaustiveness

*When to use:*
- Finite set of subtypes
- Algebraic data types (sum types)
- Pattern matching with exhaustiveness

*Q3: How do virtual threads change concurrent programming?*

A: Virtual threads (Java 21+) = cheap threads (millions possible)

*Old problem:*
```java
// Platform threads: Expensive (1MB stack, OS overhead)
ExecutorService pool = Executors.newFixedThreadPool(100);
// Limited to 100 concurrent tasks
// Blocking I/O wastes threads
```

*New solution:*
```java
// Virtual threads: Cheap (1KB, M:N mapping)
ExecutorService pool = Executors.newVirtualThreadPerTaskExecutor();
// Millions of concurrent tasks OK
// Blocking I/O releases carrier thread
```

*Impact:*

*1. Simplify async code:*
```java
// Old: Callback hell
CompletableFuture.supplyAsync(() -> fetchUser())
    .thenCompose(user -> fetchOrders(user))
    .thenAccept(orders -> process(orders));

// New: Blocking style (but async performance)
Thread.startVirtualThread(() -> {
    User user = fetchUser();      // Blocks virtual thread (OK!)
    Orders orders = fetchOrders(user);
    process(orders);
});
```

*2. Scale to millions:*
- Handle 1M+ connections (web servers)
- One virtual thread per request (simple model)

*3. No async complexity:*
- Write blocking code (easier)
- JVM handles async (under the hood)

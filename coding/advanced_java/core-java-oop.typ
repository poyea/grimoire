= Part 0: Core Java & Object-Oriented Programming

== Java Fundamentals

=== Primitive Types vs Reference Types

*Primitives (8 types):* `byte`, `short`, `int`, `long`, `float`, `double`, `char`, `boolean`
- Stored directly on stack (in method frames)
- No object overhead (no header, no GC)
- Pass by value (copy of value)

*Reference types:* Objects, arrays, interfaces
- Reference stored on stack, object on heap
- 12-16 byte object header (mark word + class pointer)
- Pass by reference value (copy of reference)

```java
int x = 42;              // Stack: 4 bytes
Integer y = 42;          // Stack: 8 bytes (reference) + Heap: ~16 bytes (object)
```

*Autoboxing cost:*
```java
// Bad: creates 1,000,000 Integer objects
Integer sum = 0;
for (int i = 0; i < 1_000_000; i++) {
    sum += i;  // Unbox, add, box for each iteration!
}

// Good: primitive
int sum = 0;
for (int i = 0; i < 1_000_000; i++) {
    sum += i;
}
```

*Wrapper class caching:*
```java
Integer a = 127;
Integer b = 127;
System.out.println(a == b);  // true (cached)

Integer c = 128;
Integer d = 128;
System.out.println(c == d);  // false (not cached)
```

*Cache ranges:*
- `Integer`, `Byte`, `Short`, `Long`: -128 to 127
- `Character`: 0 to 127
- `Boolean`: true and false (always cached)

=== String Immutability & String Pool

*Why String is immutable:*
1. Security: Can't change after creation (URLs, file paths, credentials)
2. Thread-safety: No synchronization needed for read-only data
3. String pooling: Safe to share references
4. Hash code caching: Computed once, used for HashMap keys

```java
public final class String {
    private final byte[] value;  // Java 9+: compact strings
    private int hash;            // Cached hash code

    // No setters - immutable!
}
```

*String pool (interning):*
```java
String s1 = "hello";           // String literal → pool
String s2 = "hello";           // Reuses from pool
System.out.println(s1 == s2);  // true (same object)

String s3 = new String("hello");  // Heap object (not pooled)
System.out.println(s1 == s3);     // false

String s4 = s3.intern();       // Add to pool, return pooled reference
System.out.println(s1 == s4);  // true
```

*String concatenation:*
```java
// Bad: Creates multiple intermediate String objects
String result = "";
for (int i = 0; i < 10000; i++) {
    result += i;  // O(n²) time!
}

// Good: StringBuilder for mutable operations
StringBuilder sb = new StringBuilder();
for (int i = 0; i < 10000; i++) {
    sb.append(i);  // O(n) time
}
String result = sb.toString();
```

*Compiler optimization (Java 9+):*
```java
String s = "a" + "b" + "c";
// Compiled to:
String s = "abc";  // Constant folding

String x = s1 + s2 + s3;
// Compiled to (Java 9+):
String x = invokedynamic(s1, s2, s3);  // Uses StringConcatFactory
```

=== equals() vs ==

*`==` operator:*
- Primitives: Compare values
- References: Compare memory addresses

*`equals()` method:*
- Compare logical equality (content)

```java
String s1 = new String("hello");
String s2 = new String("hello");

s1 == s2;        // false (different objects)
s1.equals(s2);   // true (same content)
```

*equals() contract (must override together with hashCode):*
1. Reflexive: `x.equals(x)` is `true`
2. Symmetric: `x.equals(y)` ⟺ `y.equals(x)`
3. Transitive: `x.equals(y) && y.equals(z)` ⟹ `x.equals(z)`
4. Consistent: Multiple invocations return same result
5. `x.equals(null)` is `false`

*hashCode() contract:*
1. Consistent: Same object → same hash code (during execution)
2. Equal objects: `x.equals(y)` ⟹ `x.hashCode() == y.hashCode()`
3. Unequal objects: Not required to have different hash codes (but better)

```java
class Person {
    private String name;
    private int age;

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Person person = (Person) o;
        return age == person.age && Objects.equals(name, person.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, age);  // Combines hash codes
    }
}
```

*What happens if you don't override hashCode():*
```java
Person p1 = new Person("Alice", 30);
Person p2 = new Person("Alice", 30);

p1.equals(p2);  // true (overridden equals)

Map<Person, String> map = new HashMap<>();
map.put(p1, "Engineer");
map.get(p2);  // null! (different hash codes → different buckets)
```

=== Interview Questions: Java Fundamentals

*Q1: Why is String immutable in Java?*

A: Multiple reasons:
1. *Security*: String used for sensitive data (passwords, URLs, file paths). If mutable, could be changed after security check.
2. *Thread-safety*: Immutable objects are inherently thread-safe. No synchronization needed.
3. *String pool*: JVM maintains pool of string literals. Immutability allows safe sharing.
4. *Hash code caching*: Hash computed once, safe to cache. Critical for HashMap performance.
5. *Class loading*: Class names are strings. If mutable, could load wrong class.

*Q2: What happens when you don't override hashCode() when overriding equals()?*

A: Violates the hashCode-equals contract. Consequences:
- Hash-based collections (HashMap, HashSet) won't work correctly
- Equal objects may have different hash codes → stored in different buckets
- `map.get(key)` may return null even if equal key exists

Example:
```java
Set<Person> set = new HashSet<>();
Person p1 = new Person("Alice", 30);
set.add(p1);
Person p2 = new Person("Alice", 30);
set.contains(p2);  // false (should be true!)
```

*Q3: Difference between String, StringBuilder, and StringBuffer?*

A:
- *String*: Immutable. Thread-safe. Use for small, read-only strings.
- *StringBuilder*: Mutable. NOT thread-safe. Use for string manipulation in single thread. (Fastest)
- *StringBuffer*: Mutable. Thread-safe (synchronized methods). Use for string manipulation in multi-threaded context. (Slower due to synchronization)

Modern code: Prefer StringBuilder unless thread-safety needed.

*Q4: What is autoboxing and what are its performance implications?*

A: Autoboxing = automatic conversion between primitives and wrapper objects.

```java
Integer x = 42;  // Autoboxing: int → Integer
int y = x;       // Unboxing: Integer → int
```

Performance costs:
1. Object creation overhead (heap allocation, GC)
2. Memory overhead (16 bytes vs 4 bytes for int)
3. Indirection (extra pointer dereference)
4. NullPointerException risk (wrapper can be null)

Critical in tight loops:
```java
// Creates 1M objects!
Integer sum = 0;
for (int i = 0; i < 1_000_000; i++) {
    sum += i;  // Unbox → add → box
}
```

*Q5: Explain the String pool. When does a String go into the pool?*

A: String pool = special memory region in heap where JVM stores unique string literals.

*Enters pool:*
1. String literals: `String s = "hello";`
2. Compile-time constants: `String s = "hel" + "lo";`
3. Explicit interning: `s.intern();`

*Does NOT enter pool:*
1. `new String("hello")` - creates heap object
2. Runtime concatenation: `s1 + s2`
3. String from char array: `new String(chars)`

Benefits:
- Memory savings (one copy of each unique string)
- Fast equality check with `==` for pooled strings
- Reduces string duplication

Tradeoff: Pool is permanent (can't be GC'd easily), so don't intern dynamic strings.

== Object-Oriented Principles

=== Four Pillars of OOP

*1. Encapsulation:* Bundle data (fields) and methods that operate on data. Hide internal state.

```java
public class BankAccount {
    private double balance;  // Hidden from outside

    public void deposit(double amount) {
        if (amount > 0) {
            balance += amount;  // Controlled access
        }
    }

    public double getBalance() {
        return balance;  // Read-only access
    }
}
```

*Benefits:*
- Data validation
- Flexibility to change implementation
- Reduced coupling

*2. Inheritance:* "Is-a" relationship. Subclass extends superclass.

```java
public class Animal {
    public void eat() { }
}

public class Dog extends Animal {
    public void bark() { }  // Additional behavior
}
```

*3. Polymorphism:* Many forms. Same interface, different implementations.

```java
Animal a1 = new Dog();
Animal a2 = new Cat();
a1.eat();  // Dog's eat()
a2.eat();  // Cat's eat()
```

*4. Abstraction:* Hide complexity, show only essential features.

```java
interface PaymentProcessor {
    void processPayment(double amount);  // What, not how
}

class CreditCardProcessor implements PaymentProcessor {
    public void processPayment(double amount) {
        // Implementation details hidden
    }
}
```

=== Method Overloading vs Overriding

*Overloading (compile-time polymorphism):*
- Same method name, different parameters
- Resolved at compile time (static binding)

```java
class Calculator {
    int add(int a, int b) { return a + b; }
    double add(double a, double b) { return a + b; }
    int add(int a, int b, int c) { return a + b + c; }
}
```

*Overriding (runtime polymorphism):*
- Subclass provides specific implementation of superclass method
- Same signature (name + parameters)
- Resolved at runtime (dynamic binding)

```java
class Animal {
    void sound() { System.out.println("Animal sound"); }
}

class Dog extends Animal {
    @Override
    void sound() { System.out.println("Bark"); }  // Override
}

Animal a = new Dog();
a.sound();  // "Bark" (runtime decision)
```

*Rules for overriding:*
1. Same method signature
2. Return type: Same or covariant (subtype)
3. Access modifier: Same or more permissive
4. Cannot override: `final`, `static`, `private` methods
5. Can throw: Same or fewer checked exceptions

*Covariant return types:*
```java
class Animal {
    Animal reproduce() { return new Animal(); }
}

class Dog extends Animal {
    @Override
    Dog reproduce() { return new Dog(); }  // Covariant (Dog is subtype of Animal)
}
```

=== Static vs Dynamic Binding

*Static binding (early binding):*
- Resolved at compile time
- Applied to: `static`, `private`, `final` methods
- Also: Method overloading

```java
class Parent {
    static void display() { System.out.println("Parent"); }
}

class Child extends Parent {
    static void display() { System.out.println("Child"); }
}

Parent p = new Child();
p.display();  // "Parent" (resolved at compile time based on reference type)
```

*Dynamic binding (late binding):*
- Resolved at runtime
- Applied to: Overridden instance methods
- Uses virtual method table (vtable)

```java
class Parent {
    void display() { System.out.println("Parent"); }
}

class Child extends Parent {
    @Override
    void display() { System.out.println("Child"); }
}

Parent p = new Child();
p.display();  // "Child" (resolved at runtime based on actual object type)
```

=== Interview Questions: OOP

*Q1: Can you override static methods? Why or why not?*

A: No, you cannot override static methods. You can *hide* them, but it's not true overriding.

Reason: Static methods belong to the class, not instances. They're resolved at compile time based on reference type (static binding), not runtime based on object type (dynamic binding).

```java
class Parent {
    static void display() { System.out.println("Parent"); }
}

class Child extends Parent {
    static void display() { System.out.println("Child"); }  // Method hiding, not overriding
}

Parent p = new Child();
p.display();  // "Parent" (compile-time resolution)

Child c = new Child();
c.display();  // "Child"
```

This is method *hiding*, not *overriding*. Always use class name for static methods: `Parent.display()`.

*Q2: What's the difference between abstract class and interface? When to use which?*

A:

*Abstract class:*
- Can have state (instance variables)
- Can have constructors
- Can have concrete methods
- Single inheritance only
- Can have any access modifiers

*Interface (Java 8+):*
- No state (only `public static final` constants)
- No constructors
- Can have: abstract, default, static, private methods
- Multiple inheritance supported
- All methods implicitly `public`

```java
// Abstract class: Partial implementation + state
abstract class Vehicle {
    protected int speed;  // State

    public Vehicle(int speed) { this.speed = speed; }  // Constructor

    public abstract void start();  // Abstract
    public void stop() { speed = 0; }  // Concrete
}

// Interface: Contract only (Java 8+: with default implementations)
interface Flyable {
    void fly();  // Abstract

    default void land() {  // Default implementation
        System.out.println("Landing...");
    }
}
```

*When to use:*
- *Abstract class*: When subclasses share common state/behavior ("is-a" relationship)
- *Interface*: When unrelated classes share common capability ("can-do" relationship)

Example:
- `Animal` (abstract class): `Dog`, `Cat` share common state (age, weight)
- `Comparable` (interface): `String`, `Integer`, `Date` can all be compared

*Q3: Can interfaces have state in Java?*

A:
- Pre-Java 8: No instance state. Only `public static final` constants.
- Java 8+: Still no instance state, but can have `default` and `static` methods.
- Java 9+: Can have `private` methods (for code reuse in default methods).

```java
interface MyInterface {
    int CONSTANT = 42;  // public static final (implicit)

    void abstractMethod();  // public abstract (implicit)

    default void defaultMethod() {  // Java 8+
        System.out.println(CONSTANT);
    }

    static void staticMethod() {  // Java 8+
        System.out.println("Static");
    }

    private void helperMethod() {  // Java 9+
        // Used by default methods
    }
}
```

Key: Interfaces define *behavior*, not state. For state, use abstract class.

== Classes & Interfaces

=== Abstract Classes vs Interfaces (Deep Dive)

*Evolution of interfaces:*

```java
// Pre-Java 8: Pure contract
interface OldInterface {
    void method1();
    void method2();
}

// Java 8+: Default methods (for API evolution)
interface ModernInterface {
    void method1();

    default void method2() {  // New method without breaking existing implementations
        System.out.println("Default implementation");
    }

    static void utility() {  // Static utility methods
        System.out.println("Utility");
    }
}

// Java 9+: Private methods (for code reuse)
interface Java9Interface {
    default void method1() {
        commonLogic();  // Reuse private method
    }

    default void method2() {
        commonLogic();
    }

    private void commonLogic() {  // Avoid duplication
        System.out.println("Common logic");
    }
}
```

*Diamond problem resolution:*
```java
interface A {
    default void method() { System.out.println("A"); }
}

interface B {
    default void method() { System.out.println("B"); }
}

class C implements A, B {
    @Override
    public void method() {
        A.super.method();  // Explicitly choose which default to use
        // Or provide own implementation
    }
}
```

=== Marker Interfaces vs Annotations

*Marker interface:* Empty interface to mark a class (e.g., `Serializable`, `Cloneable`, `Remote`)

```java
public interface Serializable {
    // Empty - just marks the class
}

class Person implements Serializable {
    // Now eligible for serialization
}
```

*Annotation:* Modern alternative (more flexible)

```java
@Entity
@Table(name = "users")
class User {
    @Id
    @GeneratedValue
    private Long id;
}
```

*Marker interface advantages:*
1. Type safety: `if (obj instanceof Serializable)` - compile-time check
2. Part of type hierarchy

*Annotation advantages:*
1. Can have parameters: `@Table(name = "users")`
2. Can be applied to methods, fields, parameters (not just classes)
3. Retained at runtime: Reflection can read them
4. No pollution of class hierarchy

*Modern preference:* Annotations for new code. Marker interfaces for backward compatibility.

=== Inner Classes

*Four types:*

*1. Static nested class:*
```java
class Outer {
    private static int staticField = 10;

    static class StaticNested {
        void display() {
            System.out.println(staticField);  // Can access static members
        }
    }
}

Outer.StaticNested obj = new Outer.StaticNested();  // No outer instance needed
```

*2. Non-static inner class:*
```java
class Outer {
    private int instanceField = 20;

    class Inner {
        void display() {
            System.out.println(instanceField);  // Can access instance members
            System.out.println(Outer.this.instanceField);  // Explicit outer reference
        }
    }
}

Outer outer = new Outer();
Outer.Inner inner = outer.new Inner();  // Needs outer instance
```

*3. Local inner class (inside method):*
```java
class Outer {
    void method() {
        final int localVar = 30;  // Must be effectively final

        class LocalInner {
            void display() {
                System.out.println(localVar);  // Can access final local variables
            }
        }

        LocalInner obj = new LocalInner();
        obj.display();
    }
}
```

*4. Anonymous inner class:*
```java
Runnable r = new Runnable() {
    @Override
    public void run() {
        System.out.println("Running");
    }
};

// Modern alternative: Lambda
Runnable r2 = () -> System.out.println("Running");
```

*Memory implications:*
- Non-static inner class holds reference to outer instance (potential memory leak!)
- Static nested class: No outer reference (preferred for performance)

```java
// Memory leak risk
class Outer {
    private byte[] data = new byte[1024 * 1024];  // 1 MB

    class Inner {
        // Implicitly holds reference to Outer (even if not used!)
    }

    Inner getInner() {
        return new Inner();
    }
}

Inner inner = new Outer().getInner();  // Outer can't be GC'd! (Inner holds reference)
```

*Best practice:* Use static nested class unless you need access to outer instance state.

=== Interview Questions: Classes & Interfaces

*Q1: What's the difference between abstract class and interface? When would you use each?*

A: (See detailed answer in previous section)

Short version:
- *Abstract class*: Shared state + partial implementation. Single inheritance. Use for "is-a" relationship.
- *Interface*: Contract + optional default methods. Multiple inheritance. Use for "can-do" capability.

*Q2: Can interfaces have constructors?*

A: No. Interfaces cannot be instantiated, so constructors would serve no purpose. Only classes (including abstract classes) can have constructors.

*Q3: What's the difference between static nested class and inner class?*

A:
- *Static nested class*: No reference to outer instance. Can be instantiated without outer instance. More memory efficient.
- *Inner class*: Holds implicit reference to outer instance. Requires outer instance to create. Can access outer instance members.

```java
Outer.StaticNested s = new Outer.StaticNested();  // OK

Outer outer = new Outer();
Outer.Inner i = outer.new Inner();  // Needs outer instance
```

Performance: Static nested class is preferred (no extra reference, no potential memory leak).

*Q4: Explain the diamond problem and how Java resolves it.*

A: Diamond problem occurs when a class implements two interfaces with the same default method.

```java
interface A { default void m() { } }
interface B { default void m() { } }
class C implements A, B { }  // Compilation error!
```

Resolution:
1. *Provide own implementation*: Override the method in class C
2. *Explicitly choose*: Call `A.super.m()` or `B.super.m()`

```java
class C implements A, B {
    @Override
    public void m() {
        A.super.m();  // Explicitly choose A's implementation
    }
}
```

Why no problem with classes? Java doesn't support multiple inheritance of classes (only interfaces).

== Exception Handling

=== Checked vs Unchecked Exceptions

*Checked exceptions (compile-time checking):*
- Must be declared in `throws` clause or caught
- Extend `Exception` (but not `RuntimeException`)
- Examples: `IOException`, `SQLException`, `ClassNotFoundException`

```java
// Must handle checked exception
public void readFile(String path) throws IOException {
    FileReader fr = new FileReader(path);  // Throws IOException
}

// Or catch
public void readFile(String path) {
    try {
        FileReader fr = new FileReader(path);
    } catch (IOException e) {
        // Handle
    }
}
```

*Unchecked exceptions (runtime checking):*
- No compile-time enforcement
- Extend `RuntimeException` or `Error`
- Examples: `NullPointerException`, `ArrayIndexOutOfBoundsException`, `IllegalArgumentException`

```java
// No need to declare or catch
public void divide(int a, int b) {
    return a / b;  // May throw ArithmeticException (unchecked)
}
```

*Exception hierarchy:*
```
Throwable
├── Error (unchecked)
│   ├── OutOfMemoryError
│   ├── StackOverflowError
│   └── ...
└── Exception
    ├── RuntimeException (unchecked)
    │   ├── NullPointerException
    │   ├── IllegalArgumentException
    │   └── ...
    ├── IOException (checked)
    ├── SQLException (checked)
    └── ...
```

*When to use:*
- *Checked*: Recoverable conditions caller can handle (file not found, network error)
- *Unchecked*: Programming errors (null pointer, invalid argument) or unrecoverable conditions

=== Try-with-Resources & AutoCloseable

*Old way (verbose):*
```java
FileReader fr = null;
try {
    fr = new FileReader("file.txt");
    // Use fr
} catch (IOException e) {
    // Handle
} finally {
    if (fr != null) {
        try {
            fr.close();  // Must close manually
        } catch (IOException e) {
            // Handle close exception
        }
    }
}
```

*New way (Java 7+):*
```java
try (FileReader fr = new FileReader("file.txt")) {
    // Use fr
} catch (IOException e) {
    // Handle
}
// fr.close() called automatically (even if exception occurs)
```

*AutoCloseable interface:*
```java
public interface AutoCloseable {
    void close() throws Exception;
}

public class MyResource implements AutoCloseable {
    @Override
    public void close() throws Exception {
        System.out.println("Closing resource");
        // Cleanup logic
    }
}

try (MyResource resource = new MyResource()) {
    // Use resource
}  // close() called automatically
```

*Multiple resources:*
```java
try (FileReader fr = new FileReader("input.txt");
     FileWriter fw = new FileWriter("output.txt")) {
    // Use both
}  // Both closed in reverse order (fw first, then fr)
```

*Suppressed exceptions:*
```java
try (MyResource r = new MyResource()) {
    throw new Exception("Primary");
} catch (Exception e) {
    System.out.println(e.getMessage());  // "Primary"

    Throwable[] suppressed = e.getSuppressed();
    // Contains exceptions from close() if any
}
```

=== Exception Best Practices

*1. Be specific:*
```java
// Bad
public void processFile(String path) throws Exception { }

// Good
public void processFile(String path) throws IOException, FileNotFoundException { }
```

*2. Don't swallow exceptions:*
```java
// Bad
try {
    // ...
} catch (Exception e) {
    // Ignored - silently fails!
}

// Good
try {
    // ...
} catch (IOException e) {
    logger.error("Failed to process file", e);
    throw new ProcessingException("Failed to process file", e);
}
```

*3. Use unchecked for programming errors:*
```java
public void setAge(int age) {
    if (age < 0) {
        throw new IllegalArgumentException("Age cannot be negative");  // Unchecked
    }
    this.age = age;
}
```

*4. Clean up resources:*
```java
// Always use try-with-resources for AutoCloseable
try (Connection conn = getConnection()) {
    // Use conn
}
```

=== Performance Cost of Exceptions

*Exceptions are expensive:*
1. Stack trace creation (walks call stack)
2. Object creation (exception object)
3. Control flow disruption

*Benchmark:*
```java
// Normal flow: ~1 ns
int result = divide(10, 2);

// Exception flow: ~1000 ns (1000x slower!)
try {
    int result = divide(10, 0);
} catch (ArithmeticException e) {
    // ...
}
```

*Why expensive:*
- `fillInStackTrace()` walks the entire call stack
- Creates array of `StackTraceElement` objects
- Disrupts JIT optimizations (exception paths not optimized)

*Optimization (when stack trace not needed):*
```java
public class MyException extends Exception {
    @Override
    public synchronized Throwable fillInStackTrace() {
        return this;  // Don't fill stack trace (10-100x faster)
    }
}
```

*Best practice:* Use exceptions for exceptional conditions only, not for control flow.

```java
// Bad: Exception for control flow
try {
    int i = 0;
    while (true) {
        array[i++];
    }
} catch (ArrayIndexOutOfBoundsException e) {
    // End of array
}

// Good: Explicit condition
for (int i = 0; i < array.length; i++) {
    // Process array[i]
}
```

=== Interview Questions: Exceptions

*Q1: When should you use checked vs unchecked exceptions?*

A:
- *Checked*: Recoverable conditions that caller can reasonably handle
  - File not found → prompt user for different file
  - Network timeout → retry
  - Database connection failure → use fallback

- *Unchecked*: Programming errors or unrecoverable conditions
  - Null pointer → bug in code
  - Invalid argument → caller violated contract
  - Out of memory → can't recover

Rule of thumb: If caller can do something about it → checked. If it's a bug → unchecked.

*Q2: What's the performance impact of exception handling?*

A: Exceptions are expensive (~1000x slower than normal code) due to:
1. Stack trace creation (walks entire call stack)
2. Object allocation
3. JIT optimization disruption

*Mitigation:*
1. Use exceptions for exceptional conditions only (not control flow)
2. Override `fillInStackTrace()` if stack trace not needed (rare)
3. Catch specific exceptions (not generic `Exception`)

*Anti-pattern:*
```java
// DON'T use exceptions for flow control!
try {
    while (true) {
        process(iterator.next());
    }
} catch (NoSuchElementException e) {
    // Done
}
```

*Q3: What is try-with-resources and why is it better than finally?*

A: Try-with-resources (Java 7+) automatically closes `AutoCloseable` resources.

Benefits over `finally`:
1. *Less boilerplate*: No manual close() calls
2. *Exception safety*: Close called even if exception thrown
3. *Suppressed exceptions*: Exceptions from close() preserved
4. *Multiple resources*: Handles multiple resources correctly (closes in reverse order)

```java
// Old way: Verbose, error-prone
finally {
    if (resource != null) {
        try {
            resource.close();
        } catch (Exception e) {
            // Handle close exception
        }
    }
}

// New way: Concise, safe
try (Resource r = new Resource()) {
    // Use r
}  // close() called automatically
```

*Q4: Can you throw an exception from a finally block? What happens?*

A: Yes, but it *shadows* the original exception!

```java
try {
    throw new Exception("Original");
} finally {
    throw new Exception("Finally");  // Shadows original!
}
// Result: "Finally" exception propagates, "Original" is lost!
```

Best practice: *Never throw from finally*. If unavoidable, preserve original exception:

```java
Exception original = null;
try {
    // ...
} catch (Exception e) {
    original = e;
} finally {
    try {
        // Cleanup
    } catch (Exception e) {
        if (original != null) {
            e.addSuppressed(original);
        }
        throw e;
    }
}
```

Or better: Use try-with-resources (handles this automatically).

== Generics & Type System

=== Generic Classes and Methods

*Generic class:*
```java
public class Box<T> {
    private T value;

    public void set(T value) { this.value = value; }
    public T get() { return value; }
}

Box<Integer> intBox = new Box<>();
intBox.set(42);
Integer value = intBox.get();  // No cast needed!
```

*Generic method:*
```java
public class Utils {
    public static <T> void swap(T[] array, int i, int j) {
        T temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

String[] names = {"Alice", "Bob"};
Utils.<String>swap(names, 0, 1);  // Explicit type argument
Utils.swap(names, 0, 1);          // Type inference
```

*Bounded type parameters:*
```java
// Upper bound: T must be Number or subclass
public class NumberBox<T extends Number> {
    private T value;

    public double doubleValue() {
        return value.doubleValue();  // Can call Number methods
    }
}

NumberBox<Integer> box = new NumberBox<>();  // OK
NumberBox<String> box = new NumberBox<>();   // Compile error!

// Multiple bounds
public class MyClass<T extends Number & Comparable<T>> {
    // T must be Number AND Comparable
}
```

=== Type Erasure

*Type erasure:* Generics are compile-time only. At runtime, generic type information is *erased*.

```java
List<String> strings = new ArrayList<>();
List<Integer> integers = new ArrayList<>();

// At runtime, both are just List (type info erased)
strings.getClass() == integers.getClass();  // true!
```

*Why type erasure:*
1. Backward compatibility with pre-generics code (Java 5)
2. Single class file works for all type parameters

*Bridge methods:*
```java
class Node<T> {
    public void setValue(T value) { }
}

class IntNode extends Node<Integer> {
    @Override
    public void setValue(Integer value) { }  // Overrides setValue(T)
}

// After type erasure:
class Node {
    public void setValue(Object value) { }  // T → Object
}

class IntNode extends Node {
    public void setValue(Integer value) { }  // Doesn't override setValue(Object)!

    // Compiler generates bridge method:
    public void setValue(Object value) {
        setValue((Integer) value);  // Delegates to setValue(Integer)
    }
}
```

*Limitations due to type erasure:*
```java
// Cannot create generic array
T[] array = new T[10];  // Compile error!

// Cannot use instanceof with type parameter
if (obj instanceof T) { }  // Compile error!

// Cannot create instance of type parameter
T obj = new T();  // Compile error!

// Cannot have static field of type parameter
class MyClass<T> {
    private static T value;  // Compile error!
}
```

=== Wildcards

*Three types:*

*1. Unbounded wildcard: `?`*
```java
public void printList(List<?> list) {
    for (Object item : list) {
        System.out.println(item);
    }
}

printList(List.of("a", "b"));  // OK
printList(List.of(1, 2, 3));   // OK
```

*2. Upper bounded wildcard: `? extends T`*
```java
// Can read as T (or its subtypes)
public double sum(List<? extends Number> list) {
    double sum = 0;
    for (Number n : list) {  // Can read as Number
        sum += n.doubleValue();
    }
    return sum;
}

sum(List.of(1, 2, 3));           // List<Integer> OK
sum(List.of(1.5, 2.5));          // List<Double> OK
// list.add(42);                 // Compile error! Can't write
```

*3. Lower bounded wildcard: `? super T`*
```java
// Can write T (or its subtypes)
public void addNumbers(List<? super Integer> list) {
    list.add(42);        // Can write Integer
    list.add(100);       // OK
    // Integer x = list.get(0);  // Compile error! Can only read as Object
}

addNumbers(new ArrayList<Integer>());  // OK
addNumbers(new ArrayList<Number>());   // OK
addNumbers(new ArrayList<Object>());   // OK
```

=== PECS Principle

*PECS: Producer Extends, Consumer Super*

*Producer (you read from it):* Use `? extends T`
```java
public void processNumbers(List<? extends Number> numbers) {
    for (Number n : numbers) {  // Read (produce) Number
        System.out.println(n.doubleValue());
    }
}
```

*Consumer (you write to it):* Use `? super T`
```java
public void fillList(List<? super Integer> list) {
    list.add(1);  // Write (consume) Integer
    list.add(2);
    list.add(3);
}
```

*Real-world example (Collections.copy):*
```java
public static <T> void copy(
    List<? super T> dest,      // Consumer: we write to it
    List<? extends T> src      // Producer: we read from it
) {
    for (T item : src) {
        dest.add(item);
    }
}

List<Number> dest = new ArrayList<>();
List<Integer> src = List.of(1, 2, 3);
Collections.copy(dest, src);  // Works!
```

*Mnemonic:*
- GET (read) → Extends
- PUT (write) → Super

=== Interview Questions: Generics

*Q1: What is type erasure and why does Java use it?*

A: Type erasure = generic type information removed at runtime. Compiler replaces type parameters with bounds (or Object) and inserts casts.

```java
// Before erasure
List<String> list = new ArrayList<String>();
list.add("hello");
String s = list.get(0);

// After erasure
List list = new ArrayList();
list.add("hello");
String s = (String) list.get(0);  // Cast inserted by compiler
```

Why:
1. *Backward compatibility*: Pre-Java 5 code works with generic collections
2. *Single implementation*: One class file for all type arguments (not like C++ templates)

Downside: Type info unavailable at runtime (can't do `new T[]`, `instanceof T`, etc.)

*Q2: Why can't you create a generic array? `new T[10]`*

A: Due to type erasure + array covariance, would break type safety.

```java
// If this were allowed:
T[] array = new T[10];  // After erasure: Object[] array = new Object[10];

// Then this would compile but fail at runtime:
Box<String>[] boxes = new Box<String>[10];  // Suppose allowed
Object[] objects = boxes;  // Array covariance
objects[0] = new Box<Integer>();  // ArrayStoreException at runtime!
// But compiler thinks boxes[0] is Box<String>!
```

Workaround:
```java
@SuppressWarnings("unchecked")
T[] array = (T[]) new Object[10];  // Cast (unsafe but necessary)
```

Better: Use `ArrayList<T>` instead of `T[]`.

*Q3: Explain PECS principle with examples.*

A: PECS = Producer Extends, Consumer Super.

- *Producer* (you read from): `? extends T`
  - Can read as T
  - Can't write (don't know exact type)

- *Consumer* (you write to): `? super T`
  - Can write T
  - Can only read as Object (don't know exact type)

Example:
```java
// Producer: Read numbers, compute sum
public double sum(List<? extends Number> numbers) {
    double total = 0;
    for (Number n : numbers) {  // OK: read as Number
        total += n.doubleValue();
    }
    // numbers.add(42);  // Error: can't write
    return total;
}

// Consumer: Write integers
public void fill(List<? super Integer> list) {
    list.add(42);   // OK: write Integer
    list.add(100);
    // Integer x = list.get(0);  // Error: can only read as Object
}
```

Real use: `Collections.copy(List<? super T> dest, List<? extends T> src)`

*Q4: What's the difference between `List<?>` and `List<Object>`?*

A:
- `List<?>`: Unknown type. Can read as `Object`, but can't write (except null).
- `List<Object>`: Specifically a list of `Object`. Can read and write `Object`.

```java
// List<?>
List<?> wildcardList = List.of("a", "b");
Object obj = wildcardList.get(0);  // OK: read as Object
wildcardList.add("c");  // Error: unknown type
wildcardList.add(null); // OK: null is compatible with any type

// List<Object>
List<Object> objectList = new ArrayList<>();
objectList.add("string");  // OK
objectList.add(42);        // OK
objectList.add(new Dog()); // OK

// Assignment
List<?> wc = objectList;       // OK
List<Object> obj = wildcardList;  // Error: incompatible types
```

Key: `List<String>` is NOT a subtype of `List<Object>`, but IS a subtype of `List<?>`.

== Collections Framework

=== List Implementations

*ArrayList:*
- Backed by dynamic array
- Random access: O(1)
- Insert/delete at end: O(1) amortized
- Insert/delete at middle: O(n)
- Not thread-safe

```java
List<String> list = new ArrayList<>();
list.add("a");          // O(1) amortized
list.get(5);            // O(1)
list.add(2, "x");       // O(n) - shift elements
list.remove(2);         // O(n) - shift elements
```

*LinkedList:*
- Doubly-linked list
- Random access: O(n)
- Insert/delete at ends: O(1)
- Insert/delete in middle: O(n) to find + O(1) to modify
- Not thread-safe
- More memory per element (node overhead)

```java
LinkedList<String> list = new LinkedList<>();
list.addFirst("x");     // O(1)
list.addLast("y");      // O(1)
list.get(100);          // O(n) - traverse list
list.add(50, "z");      // O(n) - find position
```

*When to use:*
- *ArrayList*: Default choice. Random access, iterations, small modifications
- *LinkedList*: Frequent insertions/deletions at ends (use as Deque), rarely used in practice

*CopyOnWriteArrayList:*
- Thread-safe variant
- All modifications create new underlying array
- Ideal for read-heavy workloads with rare writes
- Iterators never throw `ConcurrentModificationException`

```java
CopyOnWriteArrayList<String> list = new CopyOnWriteArrayList<>();
list.add("x");  // Creates new array (expensive!)

// Safe concurrent iteration (no ConcurrentModificationException)
for (String s : list) {
    // Even if another thread modifies list
}
```

=== Set Implementations

*HashSet:*
- Backed by HashMap
- O(1) add, remove, contains (average)
- No ordering guarantees
- Allows one null element

```java
Set<String> set = new HashSet<>();
set.add("a");           // O(1) average
set.contains("a");      // O(1) average
set.remove("a");        // O(1) average
```

*LinkedHashSet:*
- HashSet + doubly-linked list
- Maintains insertion order
- Slightly slower than HashSet (linked list overhead)

```java
Set<String> set = new LinkedHashSet<>();
set.add("c");
set.add("a");
set.add("b");
// Iteration order: c, a, b (insertion order)
```

*TreeSet:*
- Backed by TreeMap (Red-Black tree)
- O(log n) add, remove, contains
- Sorted order (natural or custom comparator)
- No null elements

```java
Set<Integer> set = new TreeSet<>();
set.add(5);
set.add(1);
set.add(3);
// Iteration order: 1, 3, 5 (sorted)

set.first();  // 1
set.last();   // 5
set.headSet(3);  // [1]
set.tailSet(3);  // [3, 5]
```

*When to use:*
- *HashSet*: Default. Fast, no ordering needed
- *LinkedHashSet*: Need insertion order
- *TreeSet*: Need sorted order, range queries

=== Map Implementations

*HashMap:*
- Hash table (array of buckets)
- O(1) average put, get, remove
- Since Java 8: Buckets become trees when too many collisions (8+ elements)
- Not thread-safe
- Allows one null key, multiple null values

```java
Map<String, Integer> map = new HashMap<>();
map.put("Alice", 30);   // O(1) average
map.get("Alice");       // O(1) average
map.remove("Alice");    // O(1) average
```

*Internal structure:*
```java
// Simplified HashMap internals
class HashMap<K, V> {
    Node<K,V>[] table;  // Array of buckets (default size: 16)
    int size;
    float loadFactor = 0.75f;  // Resize when 75% full

    static class Node<K,V> {
        final int hash;
        final K key;
        V value;
        Node<K,V> next;  // Linked list for collisions
    }

    public V get(Object key) {
        int hash = hash(key);
        int bucket = hash & (table.length - 1);  // Modulo (fast bitwise)
        Node<K,V> node = table[bucket];

        // Search in bucket (linked list or tree)
        while (node != null) {
            if (node.hash == hash && Objects.equals(node.key, key)) {
                return node.value;
            }
            node = node.next;
        }
        return null;
    }
}
```

*Why capacity is power of 2:*
- Fast modulo: `hash % capacity` → `hash & (capacity - 1)` (bitwise AND)
- Example: `hash % 16` → `hash & 15`

*Hash collision handling (Java 8+):*
- Bucket with < 8 entries: Linked list
- Bucket with 8+ entries: Red-black tree (O(log n) worst case)
- Converts back to list when size drops to 6

*LinkedHashMap:*
- HashMap + doubly-linked list
- Maintains insertion order (or access order)
- Useful for LRU cache

```java
// Insertion order
Map<String, Integer> map = new LinkedHashMap<>();
map.put("c", 3);
map.put("a", 1);
// Iteration: c, a (insertion order)

// Access order (for LRU cache)
Map<String, Integer> lru = new LinkedHashMap<>(16, 0.75f, true);
lru.put("a", 1);
lru.put("b", 2);
lru.get("a");  // Access "a"
// Iteration: b, a (a moved to end)
```

*TreeMap:*
- Red-black tree (self-balancing BST)
- O(log n) put, get, remove
- Sorted by keys (natural order or comparator)
- No null keys (but null values OK)

```java
Map<String, Integer> map = new TreeMap<>();
map.put("c", 3);
map.put("a", 1);
map.put("b", 2);
// Iteration: a, b, c (sorted by key)

map.firstKey();  // "a"
map.lastKey();   // "c"
map.headMap("b");  // {a=1}
map.tailMap("b");  // {b=2, c=3}
```

*When to use:*
- *HashMap*: Default. Fast, no ordering
- *LinkedHashMap*: Need insertion/access order, LRU cache
- *TreeMap*: Need sorted keys, range queries
- *ConcurrentHashMap*: Thread-safe (see Part III)

=== Queue & Deque

*Queue interface:*
```java
Queue<String> queue = new LinkedList<>();
queue.offer("a");  // Add (returns false if fails)
queue.poll();      // Remove and return head (null if empty)
queue.peek();      // Return head without removing (null if empty)

queue.add("b");    // Add (throws exception if fails)
queue.remove();    // Remove head (throws exception if empty)
queue.element();   // Return head (throws exception if empty)
```

*Deque (double-ended queue):*
```java
Deque<String> deque = new ArrayDeque<>();

// Queue operations
deque.offer("a");
deque.poll();

// Stack operations
deque.push("b");  // Add to front
deque.pop();      // Remove from front

// Both ends
deque.offerFirst("x");
deque.offerLast("y");
deque.pollFirst();
deque.pollLast();
```

*PriorityQueue:*
- Min-heap (by default)
- O(log n) insert, remove
- O(1) peek
- Not thread-safe

```java
PriorityQueue<Integer> pq = new PriorityQueue<>();
pq.offer(5);
pq.offer(1);
pq.offer(3);
pq.poll();  // 1 (min element)
pq.poll();  // 3
pq.poll();  // 5

// Custom comparator (max-heap)
PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Comparator.reverseOrder());
```

=== Interview Questions: Collections

*Q1: How does HashMap work internally?*

A: HashMap uses array of buckets (default 16). Each bucket is linked list (or tree if 8+ collisions).

Steps:
1. *Hash key*: `hash = key.hashCode()` (then rehash for better distribution)
2. *Find bucket*: `index = hash & (capacity - 1)` (fast modulo using power-of-2 capacity)
3. *Search bucket*: Traverse linked list/tree, compare hash and equals()
4. *Insert/update*: Add to bucket or update value

*Collision handling:*
- Java 7: Linked list (O(n) worst case)
- Java 8+: Tree when 8+ entries (O(log n) worst case)

*Resizing:*
- When `size > capacity * loadFactor` (0.75), double capacity
- Rehash all entries (expensive!)

Key requirements:
- Override `equals()` AND `hashCode()`
- Immutable keys (don't change after inserting)

*Q2: What happens on hash collision in HashMap?*

A: Multiple strategies:

*Java 7 and earlier:* Linked list in bucket
- Insert: O(1) (add to front)
- Search: O(n) (traverse list)
- Worst case: All keys in one bucket → O(n) lookup

*Java 8+:* Tree-ification when too many collisions
- Bucket with < 8 entries: Linked list
- Bucket with 8+ entries: Convert to Red-Black tree → O(log n)
- Convert back to list when drops to 6 entries

Why 8? Balance between memory (tree uses more) and performance.

*Q3: Why is HashMap capacity always a power of 2?*

A: Fast modulo operation using bitwise AND.

```java
// Slow: Modulo operation
int bucket = hash % capacity;

// Fast: Bitwise AND (when capacity is power of 2)
int bucket = hash & (capacity - 1);
```

Example (capacity = 16):
- `hash % 16` → `hash & 15`
- `15 = 0b1111` → keeps last 4 bits → same result as modulo
- Bitwise AND is much faster than division

Also ensures good distribution of keys across buckets.

*Q4: What's the difference between ArrayList and LinkedList? When would you use each?*

A:

*ArrayList:*
- Backed by array
- Random access: O(1)
- Insert/remove at end: O(1) amortized
- Insert/remove in middle: O(n) (shift elements)
- Memory: Compact (just array + small overhead)

*LinkedList:*
- Doubly-linked list
- Random access: O(n) (must traverse)
- Insert/remove at ends: O(1)
- Insert/remove in middle: O(n) to find, O(1) to modify
- Memory: Higher overhead (node objects, pointers)

*When to use:*
- *ArrayList*: 99% of the time. Default choice. Fast iteration and random access.
- *LinkedList*: Frequent insertions/deletions at ends (better use ArrayDeque). Rarely used in practice.

Myth: "Use LinkedList for frequent insertions" → Usually false! ArrayList is faster due to cache locality.

*Q5: Explain the internal structure of HashMap in Java 8+.*

A: HashMap = array of buckets + collision handling.

*Structure:*
```java
Node<K,V>[] table;  // Array (default size 16)

static class Node<K,V> {
    int hash;
    K key;
    V value;
    Node<K,V> next;  // For linked list
}
```

*Bucket contents:*
- 0 entries: null
- 1 entry: Single node
- 2-7 entries: Linked list
- 8+ entries: Red-Black tree (TreeNode)

*Why tree-ification?*
- Defense against poor hash functions or malicious input
- Worst case: O(log n) instead of O(n)

*Process:*
1. Compute hash: `hash(key.hashCode())`
2. Find bucket: `table[hash & (table.length - 1)]`
3. Search bucket:
   - If list: Traverse, compare hash + equals()
   - If tree: Binary search in tree
4. Resize if needed: When `size > capacity * 0.75`

*Load factor (0.75):*
- Trade-off between space and time
- Higher load → less space, more collisions
- Lower load → more space, fewer collisions
- 0.75 is empirically optimal

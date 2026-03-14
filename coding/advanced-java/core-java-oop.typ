= Part 0: Core C++ & Object-Oriented Programming

== C++ Fundamentals

=== Value Types vs Reference/Pointer Types

*Fundamental types:* `bool`, `char`, `short`, `int`, `long`, `long long`, `float`, `double`
- Stored directly on stack (in function frames)
- No object overhead (no vtable pointer unless polymorphic)
- Pass by value (copy of value) by default

*Pointer/reference types:* Pointers (`T*`), references (`T&`), smart pointers (`unique_ptr`, `shared_ptr`)
- Raw pointer: 8 bytes on 64-bit (stores address)
- Reference: alias for existing object (no separate storage conceptually)
- Objects can live on stack _or_ heap (programmer controls placement)

*Performance note:* Unlike Java, C++ gives full control over allocation. Stack allocation is essentially free (just a pointer bump), while heap allocation (`new`) costs ~50--100 ns due to allocator overhead.

```cpp
int x = 42;                          // Stack: 4 bytes, zero overhead
auto y = std::make_unique<int>(42);  // Heap: 4 bytes + allocator metadata
// No "boxing" concept -- int is always int
```

*No autoboxing in C++:*
```cpp
// C++ has no wrapper classes -- primitives are always primitives
// Templates work directly with fundamental types
std::vector<int> nums;  // Stores int directly (no boxing!)
nums.push_back(42);     // No object creation overhead

// Unlike Java's ArrayList<Integer> which boxes every element,
// std::vector<int> stores contiguous int values (cache-friendly)
```

*Integral promotions (analogous to Java's widening):*
```cpp
short a = 10;
int b = a;       // Implicit widening -- safe
double c = b;    // Implicit int to double
int d = c;       // Warning: narrowing conversion (data loss possible)
int e = static_cast<int>(c);  // Explicit cast -- suppresses warning
```

=== std::string and String Views

*Why `std::string` is mutable (unlike Java's String):*
1. Performance: In-place modification avoids allocation
2. Small String Optimization (SSO): Strings up to ~22 chars stored inline (no heap)
3. No string pool needed -- mutability means no shared references to worry about

```cpp
#include <string>

// std::string is mutable
std::string s = "hello";
s[0] = 'H';       // Direct modification -- OK
s += " world";    // Append in place
```

*`std::string_view` (C++17) -- non-owning, read-only view:*
```cpp
#include <string_view>

void print_name(std::string_view name) {  // No copy!
    std::cout << name << "\n";
}

std::string full = "Hello, World";
std::string_view sv = full;       // No allocation
std::string_view sub = sv.substr(0, 5);  // "Hello" -- still no allocation!
```

*String concatenation:*
```cpp
// Bad: repeated allocations
std::string result;
for (int i = 0; i < 10000; i++) {
    result += std::to_string(i);  // May reallocate each time
}

// Better: reserve capacity upfront
std::string result;
result.reserve(50000);  // Pre-allocate
for (int i = 0; i < 10000; i++) {
    result += std::to_string(i);  // No reallocation
}

// Alternative: std::ostringstream
std::ostringstream oss;
for (int i = 0; i < 10000; i++) {
    oss << i;
}
std::string result = oss.str();

// C++20: std::format
auto s = std::format("{} + {} = {}", 1, 2, 3);  // "1 + 2 = 3"
```

*Small String Optimization (SSO):*
```cpp
// Strings <= ~22 chars (implementation-defined) stored inline
std::string short_str = "hello";    // No heap allocation (SSO)
std::string long_str(100, 'x');     // Heap allocation required

// sizeof(std::string) is typically 32 bytes (stores inline buffer + metadata)
```

=== operator== and Comparison Operators

*`==` operator:*
- Fundamental types: Compare values directly
- Pointers: Compare addresses
- Objects: Uses `operator==` (must be defined)

*Defining equality:*
```cpp
#include <string>
#include <functional>

class Person {
    std::string name_;
    int age_;
public:
    Person(std::string name, int age) : name_(std::move(name)), age_(age) {}

    // C++20 defaulted comparison
    bool operator==(const Person&) const = default;

    // Or manual definition:
    // bool operator==(const Person& other) const {
    //     return name_ == other.name_ && age_ == other.age_;
    // }

    // For use in unordered containers (equivalent to Java's hashCode)
    friend struct std::hash<Person>;
    const std::string& name() const { return name_; }
    int age() const { return age_; }
};

// std::hash specialization (equivalent to Java's hashCode())
template<>
struct std::hash<Person> {
    size_t operator()(const Person& p) const {
        size_t h1 = std::hash<std::string>{}(p.name_);
        size_t h2 = std::hash<int>{}(p.age_);
        return h1 ^ (h2 << 1);  // Combine hashes
    }
};
```

*The spaceship operator (C++20) -- three-way comparison:*
```cpp
#include <compare>

class Point {
    int x_, y_;
public:
    Point(int x, int y) : x_(x), y_(y) {}

    // Generates ==, !=, <, <=, >, >= automatically
    auto operator<=>(const Point&) const = default;
};

Point a{1, 2}, b{3, 4};
bool eq = (a == b);   // false
bool lt = (a < b);    // true (lexicographic by x_, then y_)
```

*What happens without `std::hash` specialization:*
```cpp
Person p1{"Alice", 30};
Person p2{"Alice", 30};

p1 == p2;  // true (if operator== defined)

std::unordered_map<Person, std::string> map;
// Compile error! No std::hash<Person> specialization
// Must provide hash to use Person as key in unordered containers
```

=== Interview Questions: C++ Fundamentals

*Q1: What is Small String Optimization (SSO)?*

A: SSO stores short strings (typically up to 22 chars) directly inside the `std::string` object itself, avoiding heap allocation. This is possible because `std::string` is typically 32 bytes and has enough internal space. Benefits:
1. *No heap allocation*: Short strings are stack-allocated
2. *Cache-friendly*: Data is inline with the string object
3. *No allocator overhead*: Avoids `malloc`/`free` cost

*Q2: What is the difference between `std::string` and `std::string_view`?*

A:
- *`std::string`*: Owning, mutable, manages its own memory. Copies on assignment.
- *`std::string_view`*: Non-owning, read-only view into existing string data. No allocation on construction or substr. Must ensure the underlying data outlives the view (dangling risk).

```cpp
std::string owned = "hello world";
std::string_view view = owned;  // No copy
// view is valid only as long as 'owned' is alive and unmodified
```

*Q3: How does C++ handle equality vs identity comparison?*

A:
- *Identity*: Compare addresses with `&a == &b` (like Java's `==` for references)
- *Equality*: Use `operator==` (like Java's `equals()`)
- C++20 `operator<=>` generates all comparison operators from a single definition
- For hash containers, specialize `std::hash<T>` (like Java's `hashCode()`)

*Q4: Stack vs heap allocation -- when to use which?*

A:
- *Stack*: Default choice. Automatic lifetime, zero allocation cost, cache-friendly. Limited by stack size (~1--8 MB).
- *Heap*: For large objects, runtime-determined sizes, or objects that must outlive the current scope. Use `std::make_unique` / `std::make_shared` (never raw `new`).

```cpp
// Stack: fast, automatic cleanup
Person p{"Alice", 30};

// Heap: when needed for polymorphism or lifetime management
auto p = std::make_unique<Person>("Alice", 30);
```

Performance: Stack allocation is a single pointer decrement (~1 cycle). Heap allocation is ~50--100 ns (malloc + bookkeeping).

*Q5: What are the differences between `std::string` concatenation approaches?*

A:
- *`operator+=`*: Good for appending. May reallocate if capacity exceeded.
- *`reserve()` + `+=`*: Best for known-size loops. Pre-allocates capacity.
- *`std::ostringstream`*: Good for complex formatting with mixed types.
- *`std::format` (C++20)*: Type-safe, Python-style formatting. Best for readability.
- *`std::string::append()`*: Equivalent to `+=`, can append substrings efficiently.

Modern code: Prefer `std::format` (C++20) for readability, `reserve` + `+=` for performance.

== Object-Oriented Principles

=== Four Pillars of OOP

*1. Encapsulation:* Bundle data (fields) and methods that operate on data. Hide internal state.

```cpp
class BankAccount {
    double balance_ = 0.0;  // Private by default in class
public:
    void deposit(double amount) {
        if (amount > 0) {
            balance_ += amount;  // Controlled access
        }
    }

    double balance() const {  // Read-only access (const method)
        return balance_;
    }
};
```

*Benefits:*
- Data validation
- Flexibility to change implementation
- Reduced coupling
- `const` methods enforce read-only access at compile time

*2. Inheritance:* "Is-a" relationship. Derived class extends base class.

```cpp
class Animal {
public:
    virtual void eat() { }         // virtual for polymorphism
    virtual ~Animal() = default;   // Always virtual destructor in base class!
};

class Dog : public Animal {
public:
    void bark() { }  // Additional behavior
};
```

*Performance note:* Each class with virtual functions has a vtable pointer (8 bytes on 64-bit). Virtual dispatch costs ~2--5 ns per call due to indirect function call (pointer chase through vtable).

*3. Polymorphism:* Many forms. Same interface, different implementations.

```cpp
Animal* a1 = new Dog();
Animal* a2 = new Cat();
a1->eat();  // Dog's eat() -- virtual dispatch at runtime
a2->eat();  // Cat's eat() -- virtual dispatch at runtime

// Prefer smart pointers:
std::unique_ptr<Animal> a3 = std::make_unique<Dog>();
a3->eat();
```

*4. Abstraction:* Hide complexity, show only essential features.

```cpp
// Abstract class (equivalent to Java interface)
class PaymentProcessor {
public:
    virtual void process_payment(double amount) = 0;  // Pure virtual
    virtual ~PaymentProcessor() = default;
};

class CreditCardProcessor : public PaymentProcessor {
public:
    void process_payment(double amount) override {
        // Implementation details hidden
    }
};
```

=== Function Overloading vs Overriding

*Overloading (compile-time polymorphism):*
- Same function name, different parameters
- Resolved at compile time (static binding)

```cpp
class Calculator {
public:
    int add(int a, int b) { return a + b; }
    double add(double a, double b) { return a + b; }
    int add(int a, int b, int c) { return a + b + c; }
};
```

*Overriding (runtime polymorphism):*
- Derived class provides specific implementation of base class virtual method
- Same signature (name + parameters)
- Resolved at runtime (dynamic binding via vtable)

```cpp
class Animal {
public:
    virtual void sound() { std::cout << "Animal sound\n"; }
    virtual ~Animal() = default;
};

class Dog : public Animal {
public:
    void sound() override { std::cout << "Bark\n"; }  // override keyword
};

Animal* a = new Dog();
a->sound();  // "Bark" (runtime decision via vtable)
delete a;    // Virtual destructor ensures proper cleanup
```

*Rules for overriding:*
1. Must be `virtual` in base class
2. Same function signature
3. Return type: Same or covariant (pointer/reference to derived)
4. `override` keyword (C++11) catches errors at compile time
5. Cannot override: non-virtual, `static`, or `final` methods

*Covariant return types:*
```cpp
class Animal {
public:
    virtual Animal* reproduce() { return new Animal(); }
    virtual ~Animal() = default;
};

class Dog : public Animal {
public:
    Dog* reproduce() override { return new Dog(); }  // Covariant return
};
```

=== Static vs Dynamic Binding

*Static binding (early binding):*
- Resolved at compile time
- Applied to: non-virtual methods, `static` methods, overloaded functions
- Also: templates (compile-time polymorphism)

```cpp
class Parent {
public:
    void display() { std::cout << "Parent\n"; }  // Non-virtual!
};

class Child : public Parent {
public:
    void display() { std::cout << "Child\n"; }  // Hides, not overrides
};

Parent* p = new Child();
p->display();  // "Parent" (resolved at compile time -- not virtual)
delete p;
```

*Dynamic binding (late binding):*
- Resolved at runtime
- Applied to: virtual methods
- Uses virtual method table (vtable)

```cpp
class Parent {
public:
    virtual void display() { std::cout << "Parent\n"; }
    virtual ~Parent() = default;
};

class Child : public Parent {
public:
    void display() override { std::cout << "Child\n"; }
};

Parent* p = new Child();
p->display();  // "Child" (resolved at runtime via vtable)
delete p;
```

*CRTP -- Static polymorphism (compile-time, zero-cost):*
```cpp
template<typename Derived>
class Base {
public:
    void interface_method() {
        static_cast<Derived*>(this)->implementation();
    }
};

class Concrete : public Base<Concrete> {
public:
    void implementation() { std::cout << "Concrete\n"; }
};

// No vtable, no virtual dispatch cost -- resolved at compile time
Concrete c;
c.interface_method();  // "Concrete"
```

=== Interview Questions: OOP

*Q1: Why should base class destructors be virtual?*

A: Without a virtual destructor, deleting a derived object through a base pointer causes undefined behavior (only base destructor runs, derived part leaked).

```cpp
class Base {
public:
    ~Base() { std::cout << "~Base\n"; }  // Non-virtual!
};

class Derived : public Base {
    int* data_ = new int[100];
public:
    ~Derived() { delete[] data_; std::cout << "~Derived\n"; }
};

Base* p = new Derived();
delete p;  // Only ~Base runs! data_ leaked! Undefined behavior!
```

Fix: Always declare virtual destructor in polymorphic base classes: `virtual ~Base() = default;`

*Q2: What is the difference between abstract class and class with pure virtual functions?*

A: In C++, an abstract class _is_ a class with at least one pure virtual function (`= 0`). There is no separate `interface` keyword.

*Abstract class (like Java abstract class):*
- Can have data members (state)
- Can have constructors
- Can have concrete (non-pure) methods
- Can have pure virtual methods

*Pure interface idiom (like Java interface):*
- Only pure virtual methods + virtual destructor
- No data members
- Multiple inheritance supported (no diamond data problem)

```cpp
// Pure interface (Java interface equivalent)
class Drawable {
public:
    virtual void draw() const = 0;
    virtual ~Drawable() = default;
};

// Abstract class with state (Java abstract class equivalent)
class Shape : public Drawable {
protected:
    double x_, y_;
public:
    Shape(double x, double y) : x_(x), y_(y) {}
    double x() const { return x_; }
    // draw() still pure virtual -- Shape is abstract
};

class Circle : public Shape {
    double radius_;
public:
    Circle(double x, double y, double r) : Shape(x, y), radius_(r) {}
    void draw() const override { /* ... */ }
};
```

*Q3: Can you override non-virtual methods?*

A: No. Non-virtual methods use static binding (resolved by declared type at compile time). You can _hide_ them, but it is not true overriding. Always use `virtual` for polymorphic behavior, and `override` to catch mistakes.

```cpp
class Parent {
public:
    void display() { std::cout << "Parent\n"; }  // Non-virtual
};

class Child : public Parent {
public:
    void display() { std::cout << "Child\n"; }  // Hides, not overrides
};

Parent* p = new Child();
p->display();  // "Parent" -- static binding

Child* c = new Child();
c->display();  // "Child"
```

*Q4: Explain the diamond problem and how C++ resolves it.*

A: Diamond problem occurs when a class inherits from two classes that share a common base.

```cpp
class Animal {
public:
    int age_ = 0;
    virtual void speak() { std::cout << "...\n"; }
    virtual ~Animal() = default;
};

class Dog : public Animal { };
class Cat : public Animal { };
class DogCat : public Dog, public Cat { };  // Two copies of Animal!
```

Resolution: *Virtual inheritance*
```cpp
class Dog : virtual public Animal { };
class Cat : virtual public Animal { };
class DogCat : public Dog, public Cat { };  // Single copy of Animal

DogCat dc;
dc.age_ = 5;  // Unambiguous -- single Animal subobject
```

Virtual inheritance has a cost: extra indirection through vbase pointer (~1--2 ns per access).

== Classes & Interfaces

=== Abstract Classes vs Pure Interfaces (Deep Dive)

*C++ has no `interface` keyword -- use abstract classes:*

```cpp
// Pure interface: only pure virtual functions + virtual destructor
class Printable {
public:
    virtual void print(std::ostream& os) const = 0;
    virtual ~Printable() = default;
};

// Abstract class with default behavior:
class Shape : public Printable {
protected:
    double x_, y_;
public:
    Shape(double x, double y) : x_(x), y_(y) {}
    virtual double area() const = 0;  // Still abstract

    void print(std::ostream& os) const override {
        os << "Shape at (" << x_ << ", " << y_ << ")";
    }
};

// C++20 Concepts as interface constraints:
template<typename T>
concept Hashable = requires(T a) {
    { std::hash<T>{}(a) } -> std::convertible_to<size_t>;
};

template<Hashable T>
void use_in_hash_map(const T& key) {
    // T must satisfy Hashable concept
}
```

*Multiple inheritance (C++ supports it directly):*
```cpp
class Flyable {
public:
    virtual void fly() = 0;
    virtual ~Flyable() = default;
};

class Swimmable {
public:
    virtual void swim() = 0;
    virtual ~Swimmable() = default;
};

class Duck : public Flyable, public Swimmable {
public:
    void fly() override { std::cout << "Flying\n"; }
    void swim() override { std::cout << "Swimming\n"; }
};
```

*Mixin pattern using CRTP:*
```cpp
template<typename Derived>
class Serializable {
public:
    std::string serialize() const {
        return static_cast<const Derived*>(this)->to_string();
    }
};

template<typename Derived>
class Loggable {
public:
    void log() const {
        std::cout << "[LOG] " << static_cast<const Derived*>(this)->to_string() << "\n";
    }
};

class User : public Serializable<User>, public Loggable<User> {
    std::string name_;
public:
    User(std::string name) : name_(std::move(name)) {}
    std::string to_string() const { return "User: " + name_; }
};
```

=== Tag Types and Type Traits (C++ Equivalent of Marker Interfaces)

*Tag types (equivalent to Java marker interfaces):*

```cpp
// Tag struct (empty -- like Java's Serializable marker interface)
struct SerializableTag {};

class Person : public SerializableTag {
    // Marked as serializable
};

// Check at compile time (like Java's instanceof)
static_assert(std::is_base_of_v<SerializableTag, Person>);
```

*Modern alternative: Concepts (C++20) and type traits:*

```cpp
// Type trait
template<typename T>
struct is_serializable : std::false_type {};

template<>
struct is_serializable<Person> : std::true_type {};

// C++20 Concept
template<typename T>
concept Serializable = requires(T t) {
    { t.serialize() } -> std::convertible_to<std::string>;
};

// Use with constexpr if
template<typename T>
void maybe_serialize(const T& obj) {
    if constexpr (Serializable<T>) {
        auto data = obj.serialize();
        // ...
    }
}
```

*Advantages of concepts over tag types:*
1. Checked at compile time with clear error messages
2. Can express requirements (methods, operators, expressions)
3. No class hierarchy pollution
4. Works with any type (including fundamental types)

=== Nested Classes

*Static nested class (most common in C++):*
```cpp
class Outer {
    static inline int static_field = 10;
public:
    class Nested {  // Does NOT have implicit reference to Outer
    public:
        void display() {
            std::cout << static_field << "\n";  // Can access private static
        }
    };
};

Outer::Nested obj;  // No outer instance needed
```

*Inner class pattern (holding reference to outer):*
```cpp
class Outer {
    int instance_field_ = 20;
public:
    class Inner {
        Outer& outer_;  // Explicit reference to outer (unlike Java's implicit)
    public:
        explicit Inner(Outer& outer) : outer_(outer) {}
        void display() {
            std::cout << outer_.instance_field_ << "\n";
        }
    };

    Inner make_inner() { return Inner{*this}; }
};

Outer outer;
Outer::Inner inner = outer.make_inner();
```

*Lambda as anonymous class replacement:*
```cpp
// Java anonymous inner class equivalent:
auto task = []() {
    std::cout << "Running\n";
};
std::thread t(task);
t.join();

// Capturing outer variables:
int local_var = 30;
auto lambda = [local_var]() {      // Capture by value
    std::cout << local_var << "\n";
};
auto lambda2 = [&local_var]() {    // Capture by reference
    std::cout << local_var << "\n";
};
```

*Memory implications:*
- C++ nested classes do NOT implicitly capture outer instance (unlike Java inner classes)
- No hidden reference means no memory leak risk from this pattern
- Lambdas capture only what you specify (by value or reference)

*Best practice:* Prefer nested classes without outer reference. Use lambdas for short-lived function objects.

=== Interview Questions: Classes & Interfaces

*Q1: How does C++ achieve the equivalent of Java interfaces?*

A: Abstract classes with only pure virtual functions and a virtual destructor serve as interfaces. C++20 concepts provide a compile-time alternative.

Short version:
- *Abstract class with pure virtuals*: Runtime polymorphism. Like Java interface with default methods.
- *Concepts (C++20)*: Compile-time constraints. Like Java interface but resolved statically.
- Multiple inheritance of pure interface classes is safe and common.

*Q2: Can abstract classes have constructors?*

A: Yes. Abstract classes can have constructors (called by derived class constructors). They cannot be instantiated directly, but constructors initialize the base portion.

```cpp
class Shape {
protected:
    double x_, y_;
public:
    Shape(double x, double y) : x_(x), y_(y) {}
    virtual double area() const = 0;
    virtual ~Shape() = default;
};

// Shape s(0, 0);  // Error: cannot instantiate abstract class
class Circle : public Shape {
    double r_;
public:
    Circle(double x, double y, double r) : Shape(x, y), r_(r) {}  // Calls Shape ctor
    double area() const override { return 3.14159 * r_ * r_; }
};
```

*Q3: What is the difference between a nested class and an inner class in C++?*

A: In C++, all nested classes are "static" by default (no implicit reference to enclosing instance). There is no Java-style "inner class" that automatically captures the enclosing `this`. To achieve that, explicitly store a reference.

Performance: No hidden outer reference means no accidental memory retention.

*Q4: Explain the diamond problem and how C++ resolves it.*

A: Diamond problem: Class D inherits from B and C, which both inherit from A. D gets two copies of A.

Resolution:
1. *Virtual inheritance*: `class B : virtual public A` ensures single A subobject
2. *Explicit qualification*: `d.B::method()` to disambiguate
3. *Override in D*: Provide D's own implementation

```cpp
class A { public: virtual void m() {} virtual ~A() = default; };
class B : virtual public A { public: void m() override {} };
class C : virtual public A { public: void m() override {} };
class D : public B, public C {
public:
    void m() override { B::m(); }  // Explicitly choose
};
```

== Exception Handling

=== C++ Exception Model

*C++ exceptions:*
- Any type can be thrown (but prefer `std::exception` hierarchy)
- No checked/unchecked distinction (all are "unchecked")
- `noexcept` specifier declares functions that do not throw
- Stack unwinding calls destructors (RAII ensures cleanup)

```cpp
#include <stdexcept>

void read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    // Use file... destructor closes it automatically (RAII)
}

void caller() {
    try {
        read_file("data.txt");
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
}
```

*Exception hierarchy:*
```
std::exception
├── std::logic_error
│   ├── std::invalid_argument
│   ├── std::out_of_range
│   ├── std::domain_error
│   └── std::length_error
├── std::runtime_error
│   ├── std::overflow_error
│   ├── std::underflow_error
│   ├── std::range_error
│   └── std::system_error
└── std::bad_alloc
    └── std::bad_array_new_length
```

*When to use:*
- *`std::logic_error`*: Programming errors (precondition violations)
- *`std::runtime_error`*: Errors detectable only at runtime (file not found, network error)
- *`std::invalid_argument`*: Bad function argument
- *`noexcept`*: Mark functions that never throw (enables optimizations)

=== RAII: The C++ Answer to try-with-resources

*RAII (Resource Acquisition Is Initialization):*
- Resources tied to object lifetime
- Constructor acquires, destructor releases
- Stack unwinding guarantees cleanup
- Superior to Java's try-with-resources (automatic, no special syntax)

```cpp
// RAII file handle (equivalent to Java's try-with-resources)
{
    std::ifstream file("data.txt");  // Opens file
    if (!file) throw std::runtime_error("open failed");
    std::string line;
    while (std::getline(file, line)) {
        // Process line
    }
}  // file destructor called automatically -- closes file

// No need for try-with-resources syntax!
```

*Custom RAII wrapper:*
```cpp
class DatabaseConnection {
    Connection* conn_;
public:
    explicit DatabaseConnection(const std::string& url)
        : conn_(connect(url)) {
        if (!conn_) throw std::runtime_error("Connection failed");
    }

    ~DatabaseConnection() {
        if (conn_) disconnect(conn_);  // Always cleaned up
    }

    // Non-copyable, moveable
    DatabaseConnection(const DatabaseConnection&) = delete;
    DatabaseConnection& operator=(const DatabaseConnection&) = delete;
    DatabaseConnection(DatabaseConnection&& other) noexcept
        : conn_(std::exchange(other.conn_, nullptr)) {}
    DatabaseConnection& operator=(DatabaseConnection&& other) noexcept {
        if (this != &other) {
            if (conn_) disconnect(conn_);
            conn_ = std::exchange(other.conn_, nullptr);
        }
        return *this;
    }

    void query(const std::string& sql) { /* ... */ }
};

// Usage: automatic cleanup
{
    DatabaseConnection db("localhost:5432");
    db.query("SELECT * FROM users");
}  // db destructor closes connection
```

*Smart pointers as RAII:*
```cpp
// unique_ptr: exclusive ownership
auto ptr = std::make_unique<Widget>();
// No need to delete -- cleaned up automatically

// shared_ptr: shared ownership (reference counted)
auto shared = std::make_shared<Widget>();
auto copy = shared;  // Reference count = 2
// Destroyed when last shared_ptr goes out of scope

// lock_guard: RAII mutex locking
std::mutex mtx;
{
    std::lock_guard<std::mutex> lock(mtx);
    // Critical section
}  // Automatically unlocked
```

=== Exception Best Practices

*1. Catch by const reference:*
```cpp
// Bad: catches by value (slicing risk)
try { /* ... */ } catch (std::exception e) { }

// Good: catch by const reference
try { /* ... */ } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
}
```

*2. Use noexcept for functions that cannot throw:*
```cpp
// Enables compiler optimizations (no unwind tables needed)
void swap(int& a, int& b) noexcept {
    int tmp = a;
    a = b;
    b = tmp;
}

// Move operations should be noexcept (enables optimizations in containers)
class Widget {
public:
    Widget(Widget&& other) noexcept;
    Widget& operator=(Widget&& other) noexcept;
};
```

*3. Use RAII instead of try-finally:*
```cpp
// Bad: manual cleanup
Widget* w = new Widget();
try {
    w->do_something();
} catch (...) {
    delete w;
    throw;
}
delete w;

// Good: RAII
auto w = std::make_unique<Widget>();
w->do_something();  // Automatically cleaned up on exception or normal exit
```

*4. Validate arguments with exceptions:*
```cpp
void set_age(int age) {
    if (age < 0) {
        throw std::invalid_argument("Age cannot be negative");
    }
    age_ = age;
}
```

=== Performance Cost of Exceptions

*C++ exceptions use zero-cost model (when not thrown):*
1. Normal path: Zero overhead (no try/catch cost)
2. Throw path: Very expensive (stack unwinding, RTTI)

*Performance characteristics:*
```cpp
// Normal flow: 0 ns overhead (zero-cost exception model)
int result = divide(10, 2);

// Exception flow: ~1000-5000 ns (stack unwinding + RTTI)
try {
    int result = divide(10, 0);
} catch (const std::exception& e) {
    // ...
}
```

*Why throwing is expensive:*
- Stack unwinding: Must call destructors for all stack objects
- RTTI: Runtime type identification to match catch clauses
- Memory allocation: Exception object typically heap-allocated
- Binary size: Exception tables increase code size

*Alternative: Error codes or `std::expected` (C++23):*
```cpp
// std::expected (C++23): value-or-error without exceptions
std::expected<int, std::string> divide(int a, int b) {
    if (b == 0) return std::unexpected("Division by zero");
    return a / b;
}

auto result = divide(10, 0);
if (result) {
    std::cout << *result << "\n";
} else {
    std::cerr << result.error() << "\n";
}

// std::optional for "might not have a value"
std::optional<int> find_index(const std::vector<int>& v, int val) {
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i] == val) return static_cast<int>(i);
    }
    return std::nullopt;
}
```

*Best practice:* Use exceptions for truly exceptional conditions. Use return values (`std::optional`, `std::expected`, error codes) for expected failure cases.

=== Interview Questions: Exceptions

*Q1: How do C++ exceptions differ from Java exceptions?*

A:
- *No checked exceptions*: C++ has no compile-time exception checking. Use `noexcept` to declare non-throwing functions.
- *Zero-cost model*: Normal path has zero overhead (unlike Java where try blocks have some cost). Throw path is more expensive.
- *RAII replaces finally*: Destructors guarantee cleanup. No `finally` block needed.
- *Any type can be thrown*: Not just `std::exception` subclasses (though best practice is to use them).

*Q2: What is the performance impact of exception handling in C++?*

A: C++ uses zero-cost exception handling:
1. *Normal path*: Zero overhead -- no extra instructions
2. *Throw path*: Very expensive (1000--5000 ns) due to stack unwinding and RTTI
3. *Binary size*: Exception tables increase executable size

*When to avoid exceptions:*
1. Hot paths where failures are expected (use error codes / `std::expected`)
2. Real-time systems (unpredictable latency)
3. Embedded systems (binary size constraints)

*Q3: What is RAII and why is it better than Java's try-with-resources?*

A: RAII (Resource Acquisition Is Initialization) ties resource lifetime to object lifetime. Destructor releases the resource.

Benefits over try-with-resources:
1. *Automatic*: No special syntax needed -- just use objects on the stack
2. *Composable*: Works with any resource (memory, files, locks, sockets)
3. *Exception-safe*: Stack unwinding calls all destructors
4. *No forget risk*: Cannot accidentally forget to close -- destructor always runs

```cpp
// C++ RAII: automatic, no special syntax
{
    std::lock_guard lock(mtx);
    std::ifstream file("data.txt");
    auto conn = std::make_unique<Connection>(url);
    // All three resources released at scope exit
}
```

*Q4: What is `noexcept` and when should you use it?*

A: `noexcept` declares that a function does not throw exceptions. If it does throw, `std::terminate()` is called.

When to use:
1. *Move constructors/assignment*: Enables container optimizations (`vector::push_back` uses move if noexcept)
2. *Destructors*: Implicitly noexcept (throwing from destructor is almost always wrong)
3. *Swap functions*: Should never throw
4. *Simple getters/setters*: Obvious non-throwing functions

```cpp
class Widget {
public:
    Widget(Widget&& other) noexcept;  // Enables vector optimization
    ~Widget();                         // Implicitly noexcept
    int value() const noexcept { return value_; }
};
```

== Templates & Type System

=== Class Templates and Function Templates

*Class template:*
```cpp
template<typename T>
class Box {
    T value_;
public:
    void set(const T& value) { value_ = value; }
    const T& get() const { return value_; }
};

Box<int> int_box;
int_box.set(42);
int value = int_box.get();  // No cast needed -- type safe
```

*Function template:*
```cpp
template<typename T>
void swap_values(T* array, int i, int j) {
    T temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}

std::string names[] = {"Alice", "Bob"};
swap_values<std::string>(names, 0, 1);  // Explicit type
swap_values(names, 0, 1);               // Type deduction
```

*Constrained templates (C++20 concepts):*
```cpp
// Upper bound equivalent: T must be arithmetic
template<typename T>
    requires std::is_arithmetic_v<T>
class NumberBox {
    T value_;
public:
    double double_value() const {
        return static_cast<double>(value_);
    }
};

NumberBox<int> box;      // OK
// NumberBox<std::string> box;  // Compile error! string is not arithmetic

// Multiple constraints
template<typename T>
    requires std::is_arithmetic_v<T> && std::totally_ordered<T>
class ComparableNumber {
    T value_;
    // T must be arithmetic AND comparable
};
```

*C++20 abbreviated function templates:*
```cpp
// Shorthand with auto
void print(const auto& value) {
    std::cout << value << "\n";
}

// Constrained with concept
void print_number(const std::integral auto& value) {
    std::cout << value << "\n";
}
```

=== Template Specialization (vs Java's Type Erasure)

*C++ templates are NOT erased -- each instantiation is a separate type:*

```cpp
Box<int> int_box;
Box<std::string> str_box;

// These are completely different types at runtime!
// sizeof(Box<int>) may differ from sizeof(Box<std::string>)
// typeid(int_box) != typeid(str_box)
```

*Unlike Java's type erasure:*
1. Full type information available at runtime
2. Each instantiation generates separate code (code bloat tradeoff)
3. Can specialize for specific types
4. Can use `sizeof(T)`, `new T()`, `T[]` -- all things Java can't do

*Template specialization:*
```cpp
// Primary template
template<typename T>
class Serializer {
public:
    static std::string serialize(const T& value) {
        return std::to_string(value);
    }
};

// Full specialization for std::string
template<>
class Serializer<std::string> {
public:
    static std::string serialize(const std::string& value) {
        return "\"" + value + "\"";
    }
};

// Partial specialization for pointers
template<typename T>
class Serializer<T*> {
public:
    static std::string serialize(T* value) {
        if (!value) return "null";
        return Serializer<T>::serialize(*value);
    }
};

Serializer<int>::serialize(42);          // "42"
Serializer<std::string>::serialize("hi"); // "\"hi\""
```

*Compile-time capabilities (no Java equivalent):*
```cpp
// Can create arrays of template type
template<typename T, size_t N>
class FixedArray {
    T data_[N];  // Stack-allocated array of T
public:
    T& operator[](size_t i) { return data_[i]; }
    constexpr size_t size() const { return N; }
};

FixedArray<int, 10> arr;  // 10 ints on the stack, zero heap allocation
```

=== Concepts and Constraints (C++20)

*Concepts replace Java's bounded wildcards with compile-time constraints:*

*Defining concepts:*
```cpp
// Equivalent to Java's <T extends Number>
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

// Equivalent to Java's <T extends Comparable<T>>
template<typename T>
concept Comparable = std::totally_ordered<T>;

// Complex concept
template<typename T>
concept Printable = requires(T t, std::ostream& os) {
    { os << t } -> std::same_as<std::ostream&>;
};
```

*Using concepts:*
```cpp
// Constrained template (like Java's bounded type parameter)
template<Numeric T>
double sum(const std::vector<T>& numbers) {
    double total = 0;
    for (const auto& n : numbers) {
        total += static_cast<double>(n);
    }
    return total;
}

sum(std::vector<int>{1, 2, 3});      // OK
sum(std::vector<double>{1.5, 2.5});  // OK
// sum(std::vector<std::string>{});  // Compile error! string is not Numeric
```

*No PECS needed in C++:*

Java needs PECS (`? extends T`, `? super T`) because of type erasure and invariant generics. C++ templates are structurally typed -- if the operations compile, the type works.

```cpp
// C++ equivalent of Collections.copy -- no wildcards needed
template<typename DestIter, typename SrcIter>
void copy_range(DestIter dest, SrcIter src_begin, SrcIter src_end) {
    while (src_begin != src_end) {
        *dest++ = *src_begin++;
    }
}

// Works as long as types are assignment-compatible
std::vector<double> dest(3);
std::vector<int> src = {1, 2, 3};
copy_range(dest.begin(), src.begin(), src.end());  // int -> double: OK
```

=== Interview Questions: Templates

*Q1: How do C++ templates differ from Java generics?*

A: Fundamental differences:

1. *No type erasure*: C++ generates separate code per instantiation. Full type info at runtime.
2. *Can use value types*: `vector<int>` stores actual ints (no boxing).
3. *Template specialization*: Can provide optimized implementations for specific types.
4. *Compile-time computation*: `constexpr`, `if constexpr`, template metaprogramming.
5. *Code bloat*: Each instantiation generates code (tradeoff for zero-cost abstraction).

```cpp
// Things possible in C++ but not Java:
template<typename T>
T* create() { return new T(); }    // Can construct T

template<typename T, size_t N>
struct Array { T data[N]; };        // Can create arrays of T with size N

template<typename T>
constexpr size_t type_size = sizeof(T);  // Can get size of T
```

*Q2: What are concepts (C++20) and how do they improve templates?*

A: Concepts are named constraints on template parameters. They replace SFINAE and `enable_if` with readable syntax.

```cpp
// Before C++20: SFINAE (ugly)
template<typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
T add(T a, T b) { return a + b; }

// C++20: Concepts (clean)
template<std::integral T>
T add(T a, T b) { return a + b; }
```

Benefits:
1. *Readable error messages*: "T does not satisfy Integral" vs cryptic SFINAE errors
2. *Self-documenting*: Concept name describes the requirement
3. *Composable*: Combine with `&&`, `||`
4. *Subsumption*: More constrained overloads are preferred

*Q3: What is template specialization? When would you use it?*

A: Template specialization provides a custom implementation for specific type arguments.

Use cases:
1. *Optimization*: `std::vector<bool>` is specialized to use 1 bit per element
2. *Different algorithms*: Sort strings differently from integers
3. *Type traits*: `is_integral<int>` is specialized to `true_type`

```cpp
// Primary template
template<typename T>
struct Hash {
    size_t operator()(const T& v) const;
};

// Specialization for const char*
template<>
struct Hash<const char*> {
    size_t operator()(const char* s) const {
        size_t hash = 0;
        while (*s) hash = hash * 31 + *s++;
        return hash;
    }
};
```

*Q4: Why doesn't C++ need PECS (Producer Extends, Consumer Super)?*

A: PECS is a Java workaround for invariant generics + type erasure. C++ templates use structural typing -- if the code compiles with a given type, it works. No variance annotations needed.

```cpp
// This just works -- no wildcards required
template<typename T>
void process(const std::vector<T>& input, std::vector<T>& output) {
    for (const auto& item : input) {
        output.push_back(item);
    }
}

// Implicit conversions work naturally
std::vector<int> ints = {1, 2, 3};
std::vector<double> doubles;
// Just use std::copy with appropriate iterators
std::transform(ints.begin(), ints.end(), std::back_inserter(doubles),
               [](int i) { return static_cast<double>(i); });
```

== STL Containers

=== Sequence Containers

*`std::vector` (equivalent to Java's ArrayList):*
- Dynamic array, contiguous memory
- Random access: $O(1)$
- Insert/delete at end: $O(1)$ amortized
- Insert/delete at middle: $O(n)$
- Not thread-safe

```cpp
std::vector<std::string> vec;
vec.push_back("a");         // O(1) amortized
vec[5];                     // O(1) -- no bounds check
vec.at(5);                  // O(1) -- bounds checked (throws std::out_of_range)
vec.insert(vec.begin() + 2, "x");  // O(n) -- shift elements
vec.erase(vec.begin() + 2);        // O(n) -- shift elements
```

*Performance note:* `std::vector` stores elements contiguously in memory. This means excellent cache locality -- iterating a vector is ~10x faster than iterating a linked list due to CPU cache line prefetching.

*`std::deque` (double-ended queue):*
- Random access: $O(1)$
- Insert/delete at both ends: $O(1)$
- Insert/delete in middle: $O(n)$
- Not contiguous memory (blocks of arrays)

```cpp
std::deque<std::string> dq;
dq.push_front("x");     // O(1) -- unlike vector!
dq.push_back("y");      // O(1)
dq[100];                 // O(1)
```

*`std::list` (equivalent to Java's LinkedList):*
- Doubly-linked list
- Random access: $O(n)$
- Insert/delete anywhere (with iterator): $O(1)$
- Not thread-safe
- More memory per element (two pointers + node overhead)

```cpp
std::list<std::string> lst;
lst.push_front("x");    // O(1)
lst.push_back("y");     // O(1)
// No operator[] -- must iterate
auto it = std::next(lst.begin(), 50);
lst.insert(it, "z");    // O(1) insertion (but O(n) to find position)
```

*`std::array` (fixed-size, stack-allocated):*
```cpp
std::array<int, 5> arr = {1, 2, 3, 4, 5};
arr[0];           // O(1)
arr.size();       // constexpr 5
// No push_back -- fixed size
```

*When to use:*
- *`std::vector`*: Default choice (~99% of cases). Fast, cache-friendly, contiguous.
- *`std::deque`*: Need fast push_front. Use as queue.
- *`std::list`*: Need stable iterators during insertion/deletion. Rarely used.
- *`std::array`*: Fixed size known at compile time. Zero overhead.

=== Associative Containers (Sets)

*`std::unordered_set` (equivalent to Java's HashSet):*
- Hash table
- $O(1)$ average insert, find, erase
- No ordering guarantees
- Requires `std::hash<T>` and `operator==`

```cpp
std::unordered_set<std::string> set;
set.insert("a");          // O(1) average
set.count("a");           // O(1) average (0 or 1)
set.contains("a");        // O(1) average (C++20)
set.erase("a");           // O(1) average
```

*`std::set` (equivalent to Java's TreeSet):*
- Red-Black tree (self-balancing BST)
- $O(log n)$ insert, find, erase
- Sorted order (uses `operator<` by default)
- Requires `operator<` or custom comparator

```cpp
std::set<int> sorted_set;
sorted_set.insert(5);
sorted_set.insert(1);
sorted_set.insert(3);
// Iteration order: 1, 3, 5 (sorted)

auto it = sorted_set.begin();   // Points to 1 (smallest)
auto rit = sorted_set.rbegin(); // Points to 5 (largest)
auto lb = sorted_set.lower_bound(3);  // Iterator to 3
auto ub = sorted_set.upper_bound(3);  // Iterator past 3 (to 5)
```

*When to use:*
- *`std::unordered_set`*: Default. Fast, no ordering needed.
- *`std::set`*: Need sorted order, range queries, or ordered iteration.

=== Associative Containers (Maps)

*`std::unordered_map` (equivalent to Java's HashMap):*
- Hash table (array of buckets)
- $O(1)$ average insert, find, erase
- Not thread-safe
- Requires `std::hash<K>` and `operator==` for key type

```cpp
std::unordered_map<std::string, int> map;
map["Alice"] = 30;           // O(1) average (insert or update)
map.at("Alice");             // O(1) average (throws if not found)
map.erase("Alice");          // O(1) average
map.contains("Alice");       // O(1) average (C++20)
```

*Internal structure (simplified):*
```cpp
// Simplified unordered_map internals
template<typename K, typename V>
class UnorderedMap {
    struct Node {
        size_t hash;
        K key;
        V value;
        Node* next;  // Chaining for collisions
    };

    Node** buckets_;       // Array of bucket pointers
    size_t bucket_count_;  // Number of buckets
    size_t size_;
    float max_load_factor_ = 1.0f;  // Rehash threshold

    V& get(const K& key) {
        size_t hash = std::hash<K>{}(key);
        size_t bucket = hash % bucket_count_;
        Node* node = buckets_[bucket];
        while (node) {
            if (node->hash == hash && node->key == key)
                return node->value;
            node = node->next;
        }
        throw std::out_of_range("Key not found");
    }
};
```

*Collision handling:*
- Separate chaining (linked list per bucket)
- Load factor threshold triggers rehash (default max_load_factor = 1.0)
- Rehash doubles bucket count and redistributes elements

*`std::map` (equivalent to Java's TreeMap):*
- Red-Black tree
- $O(log n)$ insert, find, erase
- Sorted by keys (uses `operator<`)

```cpp
std::map<std::string, int> sorted_map;
sorted_map["c"] = 3;
sorted_map["a"] = 1;
sorted_map["b"] = 2;
// Iteration order: a, b, c (sorted by key)

sorted_map.begin()->first;   // "a"
sorted_map.rbegin()->first;  // "c"
auto lb = sorted_map.lower_bound("b");  // Iterator to {"b", 2}
```

*When to use:*
- *`std::unordered_map`*: Default. Fast $O(1)$ operations.
- *`std::map`*: Need sorted keys, range queries, ordered iteration.
- Thread-safe map: Use `std::shared_mutex` + `std::map`, or a concurrent hash map library.

=== Queue, Stack, and Priority Queue

*`std::queue` (adapter over deque):*
```cpp
std::queue<std::string> queue;
queue.push("a");       // Enqueue
queue.front();         // Peek front (no remove)
queue.pop();           // Dequeue (void return!)
queue.empty();         // Check if empty
```

*`std::stack` (adapter over deque):*
```cpp
std::stack<std::string> stack;
stack.push("a");       // Push
stack.top();           // Peek top (no remove)
stack.pop();           // Pop (void return!)
```

*`std::priority_queue` (max-heap by default):*
- $O(log n)$ push, pop
- $O(1)$ top
- Not thread-safe

```cpp
// Max-heap (default)
std::priority_queue<int> max_pq;
max_pq.push(5);
max_pq.push(1);
max_pq.push(3);
max_pq.top();   // 5 (max element) -- NOTE: opposite of Java (min-heap)
max_pq.pop();   // Remove 5

// Min-heap (like Java's default PriorityQueue)
std::priority_queue<int, std::vector<int>, std::greater<int>> min_pq;
min_pq.push(5);
min_pq.push(1);
min_pq.push(3);
min_pq.top();   // 1 (min element)

// Custom comparator
auto cmp = [](const auto& a, const auto& b) { return a.priority < b.priority; };
std::priority_queue<Task, std::vector<Task>, decltype(cmp)> task_pq(cmp);
```

=== Interview Questions: Containers

*Q1: How does `std::unordered_map` work internally?*

A: `std::unordered_map` uses a hash table with separate chaining (linked list per bucket).

Steps:
1. *Hash key*: `std::hash<K>{}(key)`
2. *Find bucket*: `hash % bucket_count`
3. *Search bucket*: Traverse linked list, compare hash and `operator==`
4. *Insert/update*: Add to bucket or update value

*Collision handling:*
- Separate chaining with linked lists
- Rehash when `size > bucket_count * max_load_factor` (default 1.0)
- Rehash is $O(n)$

Key requirements:
- `std::hash<K>` specialization must exist
- `operator==` must be defined for key type
- Keys should be effectively immutable after insertion

*Q2: What is the difference between `std::map` and `std::unordered_map`?*

A:

*`std::unordered_map`:*
- Hash table
- $O(1)$ average, $O(n)$ worst case
- No ordering
- Requires hash + equality
- Better for most use cases

*`std::map`:*
- Red-Black tree
- $O(log n)$ guaranteed
- Sorted by key
- Requires `operator<`
- Better for ordered iteration, range queries

*Q3: Why is `std::vector` almost always preferred over `std::list`?*

A: Cache locality. `std::vector` stores elements contiguously in memory, which means:
1. CPU cache prefetching works perfectly
2. Iteration is ~10x faster than linked list
3. Less memory overhead (no per-node pointers)
4. Even insertion in the middle is often faster than list due to cache effects

Only use `std::list` when you need stable iterators (iterators remain valid after insertion/deletion elsewhere) or guaranteed $O(1)$ splice operations.

Myth: "Use linked list for frequent insertions" -- Usually false due to cache effects.

*Q4: What is the difference between `std::set` and `std::unordered_set`?*

A:
- *`std::unordered_set`*: Hash table. $O(1)$ average. No ordering. Default choice.
- *`std::set`*: Red-Black tree. $O(log n)$ guaranteed. Sorted. Use for range queries.

*Q5: How does `std::unordered_map` handle collisions and resizing?*

A:
*Collisions:* Separate chaining -- each bucket contains a linked list of key-value pairs with the same hash bucket index.

*Resizing:*
- When `size() > bucket_count() * max_load_factor()`, triggers rehash
- All elements re-bucketed into new, larger table
- Rehash is $O(n)$ -- use `reserve()` to avoid rehashes if size is known

```cpp
std::unordered_map<std::string, int> map;
map.reserve(1000);  // Pre-allocate buckets for 1000 elements
// No rehash until 1000+ elements inserted
```

*Load factor:*
- Trade-off between space and time
- Higher load factor: less space, more collisions
- Lower load factor: more space, fewer collisions
- Default max_load_factor is 1.0

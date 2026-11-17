#import "../template.typ": xref

= Memory Safety, Exploitation, and Mitigations <memory-safety>

The strongest cryptography is irrelevant if an attacker can read the key out of memory or redirect the program that holds it. For decades, the largest single class of serious vulnerabilities in systems software has been memory corruption in C and C++: writing past the end of a buffer, using memory after it was freed, confusing one type for another. Microsoft and the Chromium project both reported that about 70 percent of their severe security bugs were memory-safety issues. This chapter explains the bug classes, how attackers turn them into control of a process in general terms, the layered mitigations that make exploitation harder (stack protectors, non-executable memory, address randomisation, control-flow integrity, hardened allocators, memory tagging and capabilities), the tools that find these bugs before attackers do, and the shift toward memory-safe languages. It describes defences and the reasoning behind them, not exploitation recipes.

*See also:* #xref("cryptography-and-security", "side-channel-attacks", label: "Side-Channel Attacks"), #xref("formal-methods", "symbolic-execution-and-fuzzing", label: "Symbolic Execution and Fuzzing") (sanitizers and fuzzing), #xref("compilers", "compiler-correctness", label: "Compiler Correctness: Undefined Behaviour, Testing, and Validation"), #xref("web-and-browsers", "browser-architecture", label: "Browser Architecture") (sandboxing), #xref("operating-systems", "security-models", label: "Security Models").

== Bug Classes

#table(
  columns: 3,
  [*Class*], [*What goes wrong*], [*Typical cause*],
  [Spatial: out-of-bounds write or read], [Access outside an object's bounds], [Unchecked length, off-by-one, wrong size calculation],
  [Temporal: use-after-free], [Access through a pointer to freed memory, which may now hold a different object], [Unclear ownership, callbacks that free their caller's object],
  [Double free, invalid free], [Allocator metadata corrupted], [Duplicate cleanup paths],
  [Type confusion], [Memory interpreted as the wrong type], [Unchecked downcasts, unions, JIT bugs],
  [Uninitialised read], [Stale data (possibly secrets or pointers) used], [Missing initialisation on some path],
  [Integer overflow or truncation], [A size wraps, then an allocation is too small], [Arithmetic on attacker-controlled lengths],
)

Two short examples show how ordinary code produces them:

```cpp
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

// Spatial: if len > 64, memcpy writes past the end of buf on the stack.
void parse_unsafe(const char* src, std::size_t len) {
    char buf[64];
    std::memcpy(buf, src, len);
}

// Integer overflow: count * sizeof(T) can wrap, so the allocation is too small
// and the loop writes past it.
struct record { char data[32]; };
record* alloc_records_unsafe(std::size_t count) {
    auto* p = static_cast<record*>(std::malloc(count * sizeof(record)));
    for (std::size_t i = 0; i < count; ++i) p[i] = record{};
    return p;
}

// Temporal: the reference into the vector dangles after push_back reallocates.
void use_after_reallocation(std::vector<int>& v) {
    int& first = v.front();
    v.push_back(42);          // may reallocate and free the old storage
    first = 7;                // write to freed memory
}

// Safer forms: bounds and overflow checks, owning containers, no dangling references.
bool parse_safe(const char* src, std::size_t len, std::vector<char>& out) {
    if (len > 64) return false;
    out.assign(src, src + len);
    return true;
}
```

== From Bug to Compromise

An attacker who controls a memory-corruption bug usually works toward *primitives*: the ability to read memory at chosen addresses (an *information leak*) and to write chosen values at chosen addresses. With those, the classic goals were to overwrite a code pointer (a return address on the stack, a function pointer, a vtable pointer) so that the program jumps where the attacker wants.

Defences have pushed attacks along a progression:

- *Code injection*: write machine code into a buffer and jump to it. Defeated by non-executable data memory.
- *Code reuse*: when injected code cannot run, chain together existing instruction sequences that end in returns or indirect jumps (*return-oriented programming*), which Shacham showed to be Turing-complete given a large enough binary. Defeated in part by address randomisation and control-flow integrity.
- *Data-only attacks*: leave control flow intact and corrupt data instead, such as a permission flag, a length field, or a pointer to a security policy. These are hardest to stop with control-flow defences, which is why eliminating the underlying bug class matters.

In modern targets such as browsers, a single bug is rarely enough. Real attacks chain several: an information leak to defeat randomisation, a corruption bug in the sandboxed renderer, and a second bug to escape the sandbox into the operating system. Each layer of mitigation raises the number and quality of bugs an attacker needs.

== Mitigations Built into Toolchains and Operating Systems

#table(
  columns: 3,
  [*Mitigation*], [*Mechanism*], [*Limitation*],
  [Stack protector (canary)], [A random value between local buffers and the saved return address, checked before return], [Bypassed by leaks or by writes that skip the canary],
  [Non-executable memory (NX, DEP, W^X)], [Pages are writable or executable, not both], [Does not stop code reuse; JIT compilers need careful handling],
  [Address space layout randomisation (ASLR)], [Randomises base addresses of executables, libraries, stack and heap], [One information leak usually defeats it; limited entropy on 32-bit],
  [RELRO, bind-now], [Makes relocation tables read-only after start-up], [Covers only those tables],
  [`_FORTIFY_SOURCE`], [Replaces `memcpy`, `strcpy` and similar with checked versions when sizes are known at compile time], [Only where the compiler can see object sizes],
  [Automatic variable initialisation], [`-ftrivial-auto-var-init=zero` or `pattern` initialises stack variables], [Small cost; removes a whole bug class],
  [Library hardening], [Bounds-checked standard library operations (libc++ hardening modes, `_GLIBCXX_ASSERTIONS`)], [Checks only library containers, not raw arrays],
)

These are cheap, mostly free in performance, and enabled by default in modern distributions and toolchains. Their limitation is shared: each blocks a technique, not the bug, so attackers adapt.

== Control-Flow Integrity

*Control-flow integrity* (CFI), introduced by Abadi and colleagues, restricts indirect control transfers to targets that the program's control-flow graph allows. It has two halves:

- *Forward edges* (indirect calls and jumps): an indirect call may only go to a function of the expected type, or only to addresses marked as valid targets. Implementations include Clang CFI (type-based checks on virtual and indirect calls), Microsoft Control Flow Guard (a bitmap of valid call targets), and hardware landing-pad instructions: Intel's indirect branch tracking (`endbr64` must begin every indirect-call target) and Arm's Branch Target Identification (BTI).
- *Backward edges* (returns): a *shadow stack* keeps a second copy of return addresses where ordinary writes cannot reach, and a return to an address that does not match the shadow copy faults. Intel CET provides a hardware shadow stack, supported by Windows and by Linux since 6.6 for user space. Arm's *pointer authentication* (PAC) takes a different approach: it signs return addresses and other pointers with a key held in registers and stores a short cryptographic MAC in the unused upper pointer bits; a corrupted pointer fails authentication when used.

Coarse-grained CFI (any function entry is a valid target) leaves many usable targets; fine-grained, type-based CFI leaves far fewer. None of it stops data-only attacks.

== Hardened Allocators and Temporal Safety

Use-after-free is the most common exploitable bug class in browsers, and it is hard to mitigate without changing how memory is reused. Hardened allocators make reuse less useful to an attacker:

- *Randomised and isolated heaps*: objects of different types or sizes are allocated from separate regions (Chrome's PartitionAlloc, iOS's `kalloc_type`), so freed memory is reused only for objects of the same kind, and an attacker cannot easily place a controlled object of a different type where the freed one was.
- *Quarantine*: freed memory is not reused immediately, so dangling pointers keep pointing at dead, unchanged memory for longer.
- *Metadata protection*: allocator bookkeeping is kept out of line or checksummed (Scudo, hardened malloc implementations).
- *Reference-counted dangling-pointer protection*: Chrome's MiraclePtr (BackupRefPtr) keeps a reference count for each allocation in a raw pointer wrapper; memory freed while references remain is quarantined and poisoned rather than reused, turning many use-after-free bugs into non-exploitable crashes.

== Memory Tagging and Capabilities

Hardware can check memory safety directly.

*Memory tagging* (Arm MTE, part of Armv8.5) assigns a 4-bit tag to every 16-byte granule of memory and stores a tag in the unused top bits of each pointer. Every load and store compares the two, and a mismatch faults (synchronously, for precise reports, or asynchronously, for lower overhead). The allocator gives each new allocation a random tag and changes it on free, so out-of-bounds accesses into a neighbouring allocation and accesses through stale pointers are caught with probability about 15 in 16. It is deployed on recent Android devices and in Apple's Memory Integrity Enforcement.

A software model shows the idea:

```cpp
#include <cstdint>
#include <cstdlib>
#include <random>
#include <unordered_map>

// Toy model of memory tagging: tags in the top byte of 64-bit pointers,
// one tag per 16-byte granule, checked on every access.
class tagged_heap {
    std::unordered_map<std::uintptr_t, std::uint8_t> granule_tags_;
    std::mt19937 rng_{12345};

    static constexpr std::uintptr_t granule = 16;
    static constexpr int tag_shift = 56;

    static std::uintptr_t strip(std::uintptr_t p) { return p & ((std::uintptr_t{1} << tag_shift) - 1); }
    static std::uint8_t tag_of(std::uintptr_t p) { return static_cast<std::uint8_t>(p >> tag_shift); }

    std::uint8_t random_tag() { return static_cast<std::uint8_t>(rng_() & 0xF); }

    void set_tags(std::uintptr_t base, std::size_t size, std::uint8_t tag) {
        for (std::uintptr_t g = base; g < base + size; g += granule) granule_tags_[g] = tag;
    }

public:
    std::uintptr_t allocate(std::size_t size) {
        const std::size_t rounded = (size + granule - 1) / granule * granule;
        const auto base = reinterpret_cast<std::uintptr_t>(std::aligned_alloc(granule, rounded));
        const std::uint8_t tag = random_tag();
        set_tags(base, rounded, tag);
        return base | (std::uintptr_t{tag} << tag_shift);
    }

    void release(std::uintptr_t tagged, std::size_t size) {
        const std::uintptr_t base = strip(tagged);
        const std::size_t rounded = (size + granule - 1) / granule * granule;
        std::uint8_t fresh;
        do fresh = random_tag(); while (fresh == tag_of(tagged));  // retag on free
        set_tags(base, rounded, fresh);
        // Real allocators also quarantine and later reuse the memory; omitted here.
    }

    // The check that hardware performs on each load and store.
    bool access_allowed(std::uintptr_t tagged) const {
        const std::uintptr_t addr = strip(tagged);
        const auto it = granule_tags_.find(addr & ~(granule - 1));
        return it != granule_tags_.end() && it->second == tag_of(tagged);
    }
};
```

After `release`, the old pointer's tag no longer matches its memory, so a use-after-free is detected; an index that runs past an allocation into a neighbouring one hits a granule with a different random tag in 15 of 16 cases. Real implementations guarantee that adjacent allocations get different tags, which makes linear overflows into the next object deterministic to detect.

*Capabilities* go further. The CHERI architecture (Cambridge and SRI) replaces pointers with unforgeable *capabilities*: 128-bit values carrying bounds and permissions, with a hidden validity bit that hardware clears if the capability is modified by ordinary data operations. Every access is checked against the capability's bounds, giving deterministic spatial safety, and with allocator support (revocation of capabilities to freed memory) temporal safety as well. Arm's Morello prototype demonstrated CHERI on a server-class processor, and large C and C++ code bases have been ported with modest source changes.

== Finding Bugs First

The tools that detect memory errors during testing are the same family that fuzzing relies on:

- *AddressSanitizer* (ASan) instruments every load and store with a check against *shadow memory* (one shadow byte for every 8 bytes of application memory) that marks red zones around objects and freed memory in quarantine. It catches out-of-bounds accesses and use-after-free with about a 2 times slowdown.
- *MemorySanitizer* detects reads of uninitialised memory; *UndefinedBehaviorSanitizer* detects integer overflow, invalid shifts, misaligned access and more; *HWASan* uses top-byte tagging, like MTE in software, with lower memory overhead than ASan.
- *Continuous fuzzing* with these sanitizers (#xref("formal-methods", "symbolic-execution-and-fuzzing", label: "Symbolic Execution and Fuzzing")) finds most memory bugs in parsers and protocol code before release.
- *Static analysis* finds some classes cheaply but with false positives; it is most useful when tuned to a project's patterns.

== Memory-Safe Languages

The durable fix is to remove the bug classes from the language. Garbage-collected languages (Java, C\#, Go) prevent temporal errors and check bounds; *Rust* provides both without a garbage collector, through ownership and borrowing rules checked at compile time, with `unsafe` blocks marking where those guarantees are suspended.

Rewriting everything is not feasible, and it is not necessary. Most vulnerabilities occur in *new* or recently changed code, because old code has had its bugs found and fixed. Google reported that after Android began writing new code in memory-safe languages (mainly Rust and Kotlin) while leaving most existing C and C++ in place, the share of memory-safety vulnerabilities fell from 76 percent in 2019 to 24 percent in 2024. The strategy that follows is to write new components in memory-safe languages, harden the existing C and C++ (bounds-checked library modes, safer buffer APIs, compiler extensions such as Clang's `-fbounds-safety` for C), and isolate what remains in sandboxes with minimal privileges.

== Defence in Depth

No single layer is sufficient, and the effective posture combines them:

+ Prevent bugs: memory-safe languages for new code, safe abstractions in C++, code review for ownership.
+ Find bugs: sanitizers, fuzzing, static analysis in CI.
+ Make bugs hard to exploit: compiler and OS mitigations, CFI, hardened allocators, memory tagging.
+ Contain exploitation: sandboxing, least privilege, process isolation, so a compromised component cannot reach keys or other users' data.
+ Respond quickly: fast patching and update mechanisms, because exploited bugs will still occur.

== Pitfalls

- Treating mitigations as fixes and leaving known bugs unpatched because "ASLR covers it".
- Shipping without the standard hardening flags, or disabling them for small performance gains.
- `unsafe` code in Rust or unchecked casts in C++ that quietly reintroduce the bug classes.
- Sanitizer and fuzzing coverage that excludes the code that parses untrusted input.
- Keeping secrets in the same process as complex parsers of untrusted data.

== Further Reading

Szekeres, L., Payer, M., Wei, T. and Song, D. (2013). "SoK: Eternal War in Memory." IEEE Symposium on Security and Privacy.

Aleph One (1996). "Smashing the Stack for Fun and Profit." Phrack.

Cowan, C. et al. (1998). "StackGuard: Automatic Adaptive Detection and Prevention of Buffer-Overflow Attacks." USENIX Security.

Shacham, H. (2007). "The Geometry of Innocent Flesh on the Bone: Return-into-libc without Function Calls (on the x86)." CCS.

Abadi, M., Budiu, M., Erlingsson, Ú. and Ligatti, J. (2005). "Control-Flow Integrity." CCS.

Serebryany, K. et al. (2012). "AddressSanitizer: A Fast Address Sanity Checker." USENIX ATC.

Serebryany, K. et al. (2018). "Memory Tagging and How It Improves C/C++ Memory Safety." arXiv:1802.09517.

Watson, R. N. M. et al. (2015). "CHERI: A Hybrid Capability-System Architecture for Scalable Software Compartmentalization." IEEE Symposium on Security and Privacy.

Liljestrand, H. et al. (2019). "PAC it up: Towards Pointer Integrity using ARM Pointer Authentication." USENIX Security.

Jung, R., Jourdan, J.-H., Krebbers, R. and Dreyer, D. (2018). "RustBelt: Securing the Foundations of the Rust Programming Language." POPL.

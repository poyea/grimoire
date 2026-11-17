#import "../template.typ": definition, theorem, xref

= Symbolic Execution and Fuzzing <symbolic-execution-and-fuzzing>

Testing runs a program on chosen inputs and checks what happens. Symbolic execution and fuzzing are two ways of choosing inputs automatically, and they sit at opposite ends of a spectrum. Symbolic execution runs the program on *symbols* instead of values, collects the branch conditions along each path as logical formulas, and asks an SMT solver for inputs that drive execution down paths not yet explored. Fuzzing runs the program on millions of concrete, randomly mutated inputs per hour, guided only by cheap feedback such as which branches were covered. Symbolic execution is precise and slow; fuzzing is shallow and fast; modern tools combine them. Both are *under-approximate*: every bug they report is real (up to the fidelity of the environment model), and they never prove absence of bugs. This chapter covers the formal model of symbolic execution, a working executor on top of Z3, path explosion and its mitigations, concolic testing, coverage-guided fuzzing with a C++ core loop, hybrid fuzzers, and the theory of under-approximate reasoning.

*See also:* #xref("formal-methods", "sat-and-smt", label: "SAT and SMT"), #xref("formal-methods", "abstract-interpretation", label: "Abstract Interpretation") (the over-approximate dual), #xref("formal-methods", "model-checking", label: "Model Checking") (bounded model checking), #xref("cryptography-and-security", "side-channel-attacks", label: "Side-Channel Attacks").

== Symbolic States and Path Conditions

King's 1976 formulation still describes every symbolic executor. Program inputs are bound to fresh symbols $alpha_1, alpha_2, dots$. A *symbolic state* has three parts:

- a program counter;
- a *symbolic store* $sigma$ mapping each variable to an expression over the input symbols;
- a *path condition* $pi$, a formula over the input symbols that inputs must satisfy to reach this state.

Assignments update $sigma$ by evaluating the right-hand side symbolically. At a branch on condition $c$, the executor evaluates $c$ to a formula $sigma(c)$ and asks the solver whether $pi and sigma(c)$ and $pi and not sigma(c)$ are satisfiable. Each satisfiable side continues as its own state with the conjunct added to its path condition. When a state terminates or reaches an error, a model of its path condition is a concrete input that follows exactly that path.

#definition(name: "Symbolic execution tree")[
The *symbolic execution tree* of a program has one node per symbolic state reached, with children for each feasible successor. Each leaf's path condition characterises the set of inputs that follow the corresponding path, and the path conditions of distinct leaves are pairwise unsatisfiable together.
]

#theorem(name: "Path soundness")[
If a symbolic executor's transfer functions are exact for the program's semantics, then for every reached state with path condition $pi$ and every model $m models pi$, running the program concretely on $m$ follows that state's path and ends in the state obtained by evaluating $sigma$ under $m$.
]

The theorem is what makes symbolic execution bug-finding precise: a reachable error state comes with a witness input. It holds only where the executor models the semantics exactly. Machine integers must be modelled as bit-vectors (so `x * 3` wraps), memory as arrays, and system calls by models of the environment. Each place where these are approximated is a place where reports can be spurious or bugs missed.

== A Symbolic Executor on Z3

The executor below handles a small statement language (assignment, `if`, and `check` assertions) over 32-bit integers. It explores paths depth-first, prunes infeasible branches with the solver, and for each `check` asks whether the assertion can fail on the current path.

```cpp
#include <z3++.h>

#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

struct expr_node;
using expr_ptr = std::shared_ptr<const expr_node>;

struct expr_node {
    enum class kind { var, constant, add, mul, lt, eq, ne } k;
    std::string name;
    std::int32_t value = 0;
    expr_ptr lhs, rhs;
};

struct stmt;
using stmt_ptr = std::shared_ptr<const stmt>;

struct stmt {
    enum class kind { assign, branch, check } k;
    std::string target;
    expr_ptr e;
    std::vector<stmt_ptr> then_body, else_body;
};

struct sym_state {
    std::map<std::string, z3::expr> store;
    std::vector<z3::expr> path;
};

class executor {
    z3::context ctx_;
    std::vector<std::string> inputs_;

    z3::expr eval(const expr_ptr& e, const sym_state& s) {
        using k = expr_node::kind;
        switch (e->k) {
        case k::var:      return s.store.at(e->name);
        case k::constant: return ctx_.bv_val(e->value, 32);
        case k::add:      return eval(e->lhs, s) + eval(e->rhs, s);
        case k::mul:      return eval(e->lhs, s) * eval(e->rhs, s);
        case k::lt:       return z3::slt(eval(e->lhs, s), eval(e->rhs, s));
        case k::eq:       return eval(e->lhs, s) == eval(e->rhs, s);
        case k::ne:       return eval(e->lhs, s) != eval(e->rhs, s);
        }
        throw std::logic_error("unknown expression kind");
    }

    std::optional<z3::model> solve(const std::vector<z3::expr>& path) {
        z3::solver s(ctx_);
        for (const auto& c : path) s.add(c);
        if (s.check() != z3::sat) return std::nullopt;
        return s.get_model();
    }

    void print_input(const z3::model& m) {
        for (const auto& n : inputs_) {
            const auto v = m.eval(ctx_.bv_const(n.c_str(), 32), true).get_numeral_uint64();
            std::cout << n << '=' << static_cast<std::int32_t>(v) << ' ';
        }
        std::cout << '\n';
    }

public:
    void run(const std::vector<stmt_ptr>& program, std::vector<std::string> inputs) {
        inputs_ = std::move(inputs);
        sym_state init;
        for (const auto& n : inputs_)
            init.store.insert_or_assign(n, ctx_.bv_const(n.c_str(), 32));

        struct item {
            std::vector<stmt_ptr> todo;   // statements still to run, last one first
            sym_state st;
        };
        std::vector<item> work;
        work.push_back({{program.rbegin(), program.rend()}, init});

        while (!work.empty()) {
            item it = std::move(work.back());
            work.pop_back();
            if (it.todo.empty()) {
                if (auto m = solve(it.st.path)) { std::cout << "path test: "; print_input(*m); }
                continue;
            }
            const stmt_ptr s = it.todo.back();
            it.todo.pop_back();

            switch (s->k) {
            case stmt::kind::assign:
                it.st.store.insert_or_assign(s->target, eval(s->e, it.st));
                work.push_back(std::move(it));
                break;
            case stmt::kind::check: {
                const z3::expr c = eval(s->e, it.st);
                auto violated = it.st.path;
                violated.push_back(!c);
                if (auto m = solve(violated)) { std::cout << "CHECK FAILS: "; print_input(*m); }
                it.st.path.push_back(c);
                work.push_back(std::move(it));
                break;
            }
            case stmt::kind::branch: {
                const z3::expr c = eval(s->e, it.st);
                for (const bool taken : {true, false}) {
                    item next = it;
                    next.st.path.push_back(taken ? c : !c);
                    if (!solve(next.st.path)) continue;
                    const auto& body = taken ? s->then_body : s->else_body;
                    next.todo.insert(next.todo.end(), body.rbegin(), body.rend());
                    work.push_back(std::move(next));
                }
                break;
            }
            }
        }
    }
};
```

On the program

```text
if (x * 3 + 7 == y) {
    if (y < 100) check(x != 13);
}
```

the executor finds three feasible paths and reports `CHECK FAILS: x=13 y=46`. A random tester would need to hit the equality `x * 3 + 7 == y` by chance, a probability of about $2^(-32)$ per try. The solver finds it in milliseconds, which is the core advantage of symbolic reasoning: it inverts conditions instead of guessing.

Because arithmetic is on bit-vectors, `x * 3 + 7` is computed modulo $2^32$ exactly as the hardware computes it. When the solver pairs a large `x` with a `y`, it is the wrapped value, and the test behaves on the real machine as it did in the model. An executor over mathematical integers would produce tests whose `y` does not fit in an `int`, and would miss paths that exist only because of overflow.

== Path Explosion

Every branch can double the number of paths, and a loop whose trip count depends on input creates a path per iteration count. A program with $n$ independent input-dependent branches in sequence has $2^n$ paths; a parser loop over a symbolic buffer has more paths than any search can visit. All practical techniques fight this:

#table(
  columns: 3,
  [*Technique*], [*Idea*], [*Cost*],
  [Search heuristics], [Prioritise states likely to reach new code (KLEE's coverage-optimised search, random path selection)], [Completeness is given up],
  [State merging], [Join states at merge points with $"ite"$ expressions in the store, trading paths for harder queries], [Larger, slower solver formulas],
  [Veritesting], [Switch between dynamic exploration and static merging of loop-free regions], [Static analysis of the region],
  [Function summaries], [Analyse a function once, reuse its input-output relation at each call], [Summaries can be imprecise or large],
  [Loop bounding], [Unroll loops a fixed number of times], [Misses deeper iterations],
  [Selective symbolic inputs], [Keep only some inputs symbolic, others concrete], [Explores fewer behaviours],
)

The other bottleneck is the solver. Most of an executor's time goes to SMT queries, so engines cache query results, split path conditions into independent subsets of constraints, and reuse models: if an earlier model still satisfies a new query, no solver call is needed. KLEE's counterexample cache and constraint independence optimisations each cut solving time several-fold on its benchmarks.

== Memory and the Environment

Symbolic pointers are the hardest part of the memory model. A read through a pointer whose value is symbolic could touch any object. Options range from forking the state once per object the pointer might reference (KLEE's approach), to modelling all of memory as a single SMT array (simple, but every access becomes an array-theory term), to concretising the pointer to one feasible value (fast, incomplete).

The environment, meaning the file system, network, clock, and system calls, must be modelled or concretised. KLEE ships models of a POSIX file system with symbolic files and of common libc functions. Binary-level executors (angr, S2E, Mayhem) face the same problem at the instruction-set level and must also model or execute the kernel. Environment modelling is the main source of both false reports and missed bugs in practice.

== Concolic Testing

*Concolic* (concrete plus symbolic) execution runs the program on a concrete input while also tracking symbolic expressions alongside. At the end of the run it has one path condition $c_1 and c_2 and dots and c_n$. To explore a new path it negates one conjunct and keeps the prefix, solving $c_1 and dots and c_(i-1) and not c_i$ for a new input. DART and CUTE introduced the approach in 2005.

The concrete values solve two problems of pure symbolic execution. Calls to code the engine cannot reason about (system calls, hash functions, native libraries) simply execute concretely, and their results enter the symbolic state as constants. And when a constraint is outside the solver's theories (nonlinear arithmetic, floating point), the engine can substitute the concrete value and keep going, losing completeness but not soundness of the reports. Microsoft's SAGE applied *generational search*, negating every conjunct of a long trace in turn to produce a whole generation of new inputs per run, to file parsers in Windows and Office. It found roughly a third of the file-fuzzing bugs during Windows 7 development.

== Coverage-Guided Fuzzing

Fuzzing generates inputs by mutation. Its modern form, *coverage-guided greybox fuzzing* (AFL, libFuzzer, honggfuzz), keeps a *corpus* of interesting inputs and adds a mutated input to it only if it exercises new behaviour, measured by branch coverage. The instrumentation is cheap: at compile time, each basic block gets a random ID, and each edge between blocks increments a counter in a shared bitmap.

```cpp
#include <array>
#include <cstdint>
#include <random>
#include <vector>

constexpr std::size_t map_size = 1 << 16;
std::array<std::uint8_t, map_size> coverage_map;
thread_local std::uint32_t prev_location;

// Inserted by the compiler at every basic block with a random block id.
inline void on_block(std::uint32_t block_id) {
    ++coverage_map[(block_id ^ prev_location) & (map_size - 1)];
    prev_location = block_id >> 1;
}

std::uint8_t bucket(std::uint8_t hits) {
    if (hits <= 3) return hits;
    if (hits <= 7) return 4;
    if (hits <= 15) return 8;
    if (hits <= 31) return 16;
    if (hits <= 127) return 32;
    return 128;
}

class fuzzer {
    std::vector<std::vector<std::uint8_t>> corpus_;
    std::array<std::uint8_t, map_size> seen_{};   // union of bucketed hits so far
    std::mt19937_64 rng_{0x5eed};

    bool has_new_coverage() {
        bool fresh = false;
        for (std::size_t i = 0; i < map_size; ++i) {
            const std::uint8_t b = coverage_map[i] ? bucket(coverage_map[i]) : 0;
            if (b & ~seen_[i]) { seen_[i] |= b; fresh = true; }
        }
        return fresh;
    }

    std::vector<std::uint8_t> mutate(std::vector<std::uint8_t> in) {
        static constexpr std::uint8_t interesting[] = {0, 1, 0x7f, 0x80, 0xff};
        const int rounds = 1 + static_cast<int>(rng_() % 8);
        for (int r = 0; r < rounds; ++r) {
            if (in.empty()) { in.push_back(0); continue; }
            const std::size_t pos = rng_() % in.size();
            switch (rng_() % 4) {
            case 0: in[pos] ^= static_cast<std::uint8_t>(1u << (rng_() % 8)); break;
            case 1: in[pos] = interesting[rng_() % std::size(interesting)]; break;
            case 2: in.insert(in.begin() + pos, static_cast<std::uint8_t>(rng_())); break;
            case 3: if (in.size() > 1) in.erase(in.begin() + pos); break;
            }
        }
        return in;
    }

public:
    template <class Target>
    void run(Target&& target, std::vector<std::uint8_t> seed, std::size_t iterations) {
        corpus_.push_back(std::move(seed));
        for (std::size_t i = 0; i < iterations; ++i) {
            auto input = mutate(corpus_[rng_() % corpus_.size()]);
            coverage_map.fill(0);
            prev_location = 0;
            target(input.data(), input.size());   // crashes are caught by sanitizers
            if (has_new_coverage()) corpus_.push_back(std::move(input));
        }
    }
};
```

Hashing the edge as `block ^ (prev >> 1)` distinguishes $A -> B$ from $B -> A$ and records edges rather than blocks. Bucketing hit counts makes "the loop ran 9 times instead of 2" count as new behaviour without making every iteration count distinct. Real fuzzers add many refinements: deterministic mutation stages, splicing two corpus entries, dictionaries of tokens, corpus minimisation, and *power schedules* that spend more mutations on inputs exercising rarely visited paths (AFLFast models this as a Markov chain).

Fuzzing finds bugs only when the program *notices* them. Compiling the target with AddressSanitizer, UndefinedBehaviorSanitizer, and MemorySanitizer turns silent memory corruption into immediate crashes, and is responsible for most of the bugs fuzzers report. Google's OSS-Fuzz runs this combination continuously on over a thousand open-source projects and has found tens of thousands of bugs.

== Hybrid Fuzzing

Fuzzers get stuck on *magic values*: a branch like `if (header == 0x89504E47)` is a 32-bit lottery for random mutation. Symbolic execution solves such branches easily but explores slowly. Hybrid tools use each where it is strong:

- *Driller* runs AFL until coverage stalls, then uses concolic execution on corpus inputs to solve the branches that block progress, and hands the new inputs back to the fuzzer.
- *QSYM* makes the concolic side fast enough to run continuously, by executing natively with dynamic binary translation and by solving only the last constraint of a path when the full path condition is too hard (optimistic solving).
- *Input-to-state correspondence* (Redqueen, AFL++ CmpLog) avoids a solver entirely for the common case: it logs the operands of comparisons at run time, finds where those operand bytes appear in the input, and patches the input with the value the comparison expects.

Input-to-state matching handles most magic-byte and checksum-style obstacles at fuzzing speed, which is why it is now enabled by default in AFL++, while full symbolic reasoning remains valuable for arithmetic relations between fields.

== Under-Approximate Reasoning

Static analysis and deductive verification over-approximate: they reason about a superset of behaviours and can raise false alarms. Symbolic execution and fuzzing under-approximate: they reason about a subset of behaviours and every reported bug is reachable. O'Hearn's *incorrectness logic* gives the latter a proof theory. Its triples $[p] space C space [q]$ mean that every state in $q$ is reachable by running $C$ from some state in $p$, the reverse of Hoare logic's inclusion. Under-approximate triples justify dropping paths (the essential move of symbolic execution) without losing the guarantee that reports are true positives. Meta's Pulse analyser is built on this logic, so each bug it reports is reachable under its model of the program.

== Results in Practice

- KLEE, on the GNU Coreutils, generated tests with over 90 percent line coverage on average, more than the developers' own suites, and found bugs that had been present for over a decade.
- In the 2016 DARPA Cyber Grand Challenge, fully automated systems found, exploited, and patched vulnerabilities in unseen binaries. The winning system, Mayhem, combined symbolic execution with fuzzing.
- Continuous fuzzing services (OSS-Fuzz, ClusterFuzz, Microsoft's OneFuzz) are now standard practice for parsers, codecs, network protocol stacks, and browsers.

The rule of thumb from these deployments: start with coverage-guided fuzzing and sanitizers because they are cheap and find the most bugs per CPU hour; add input-to-state or concolic help when coverage plateaus on comparisons; reach for full symbolic execution when inputs must satisfy precise arithmetic or structural constraints, or when the goal is a test suite with systematic path coverage.

== Further Reading

King, J. C. (1976). "Symbolic Execution and Program Testing." Communications of the ACM.

Godefroid, P., Klarlund, N. and Sen, K. (2005). "DART: Directed Automated Random Testing." PLDI.

Sen, K., Marinov, D. and Agha, G. (2005). "CUTE: A Concolic Unit Testing Engine for C." ESEC/FSE.

Cadar, C., Dunbar, D. and Engler, D. (2008). "KLEE: Unassisted and Automatic Generation of High-Coverage Tests for Complex Systems Programs." OSDI.

Godefroid, P., Levin, M. Y. and Molnar, D. (2012). "SAGE: Whitebox Fuzzing for Security Testing." Communications of the ACM.

Avgerinos, T. et al. (2014). "Enhancing Symbolic Execution with Veritesting." ICSE.

Böhme, M., Pham, V.-T. and Roychoudhury, A. (2016). "Coverage-based Greybox Fuzzing as Markov Chain." CCS.

Stephens, N. et al. (2016). "Driller: Augmenting Fuzzing Through Selective Symbolic Execution." NDSS.

Yun, I. et al. (2018). "QSYM: A Practical Concolic Execution Engine Tailored for Hybrid Fuzzing." USENIX Security.

Aschermann, C. et al. (2019). "REDQUEEN: Fuzzing with Input-to-State Correspondence." NDSS.

O'Hearn, P. W. (2020). "Incorrectness Logic." POPL.

Baldoni, R. et al. (2018). "A Survey of Symbolic Execution Techniques." ACM Computing Surveys.

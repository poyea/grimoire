#import "../template.typ": definition, xref

= E-Graphs and Equality Saturation <e-graphs>

A conventional optimiser is a sequence of destructive rewrites. Each pass replaces code with code it believes is better, and the original is gone. The result depends on the order of passes, and a rewrite that looks bad locally (expanding a product, introducing a temporary) may be the one that enables a large win later. This is the *phase-ordering problem*, and compilers fight it with hand-tuned pass pipelines and repeated passes. *Equality saturation* sidesteps it. Rewrites add equivalent forms to a data structure, the *e-graph*, without removing anything. When no rewrite adds anything new, or a budget runs out, the e-graph compactly represents a very large set of equivalent programs, and an extraction step picks the best one under a cost model. This chapter covers the e-graph data structure and its invariants, the rebuild technique that made it fast, e-matching, extraction, a complete small implementation in C++, and the systems that use the approach, from Herbie to Cranelift.

*See also:* #xref("compilers", "optimisation-passes", label: "Optimisation Passes"), #xref("compilers", "ir-design", label: "IR Design"), #xref("compilers", "backend-and-codegen", label: "Backend and Code Generation") (instruction selection), #xref("formal-methods", "sat-and-smt", label: "SAT and SMT") (congruence closure in EUF).

== The Data Structure

An e-graph stores a set of terms and an equivalence relation over them that is closed under congruence: if $a equiv b$ then $f(a) equiv f(b)$.

#definition(name: "E-graph")[
An *e-graph* consists of *e-classes*, each identified by an id and containing a set of *e-nodes*. An e-node is an operator applied to a list of e-class ids, $f(c_1, dots, c_n)$, not to other nodes. Ids are managed by a union-find structure, so merged classes share a canonical id. A *hashcons* maps each canonical e-node to the e-class that contains it.
]

Because an e-node's children are classes, one e-node stands for every combination of terms in its child classes. An e-graph with a few hundred nodes can represent more equivalent terms than could ever be listed. For example, if a class contains both `a * 2` and `a << 1`, every parent of that class represents both forms at once without duplication.

Two invariants make the structure correct:

- *Hashcons invariant.* Every canonical e-node appears in the hashcons, mapped to its class. Adding a node that already exists returns the existing class, so equal terms are shared.
- *Congruence invariant.* If two e-nodes have the same operator and equivalent children, they are in the same class.

Merging two classes can break both: parents of the merged classes may now be congruent with each other, and their stored child ids are no longer canonical. Restoring the invariants can cascade, since merging those parents can make *their* parents congruent. This is congruence closure, the same algorithm that decides the theory of equality with uninterpreted functions in SMT solvers.

== Rebuilding

Early e-graph implementations restored the invariants immediately after every merge, which made each merge expensive and repeated work when many merges touch the same parents. The *egg* library introduced *rebuilding*: during a phase of rewriting, merges only update the union-find and record the merged class on a worklist. Afterwards, a single `rebuild` processes the worklist, re-canonicalises each affected parent, and merges parents that have become congruent, deduplicating work across all the merges of the phase. Read operations such as e-matching run only between rebuilds, when the invariants hold. On equality saturation workloads this gave order-of-magnitude speedups and made the technique practical for large problems.

== E-Matching

A rewrite rule $l -> r$ has patterns with variables, for example $(x times y) \/ z -> x times (y \/ z)$. *E-matching* finds every substitution of e-classes for pattern variables such that the instantiated left-hand side is represented in the e-graph. The classic algorithm walks the pattern top-down: for each e-node in each class whose operator matches the pattern root, it recursively matches child patterns against child classes, backtracking over the alternative e-nodes in each class. Applying a rule adds the instantiated right-hand side and merges its class with the matched class.

E-matching is the dominant cost in equality saturation. *Relational e-matching* treats the e-graph as a relational database (one table per operator, columns for children and the result class) and compiles each pattern to a conjunctive query evaluated with worst-case optimal joins, which is asymptotically faster for patterns with shared variables. *egglog* takes this further and unifies e-graphs with Datalog, so rules can derive facts and analyses incrementally as well as equalities.

== The Saturation Loop and Extraction

```text
repeat until saturated, or a node, time, or iteration limit is hit:
    matches := e-match every rule against the e-graph      (read-only)
    for each match: add the right-hand side, merge with the matched class
    rebuild
extract the cheapest term from the root class
```

If no rule adds a node or merges two classes, the e-graph is *saturated*: it contains every term reachable from the input by the rules, and extraction is guaranteed to find the best term the rules can produce, regardless of order. In practice, rules such as associativity and commutativity generate exponentially many terms, so saturation is rarely reached and the limits matter. Rule *scheduling* (for example, backing off rules that match too often) keeps such rules from crowding out the useful ones.

*Extraction* picks one e-node per class, starting from the root, to form a term. With a cost function where a term's cost is its operator's cost plus the costs of its children (counting shared subterms each time they appear), the best choice for each class can be computed bottom-up by a fixpoint, like a shortest-path computation. If shared subterms should be paid for once (DAG cost, which matches the real cost of computed values), extraction becomes NP-hard, and tools use integer linear programming or heuristics.

== A Complete E-Graph in C++

The implementation below follows egg's design: union-find, hashcons, parent lists, deferred rebuilding, and bottom-up extraction. Rules are written as small matching functions over the e-graph rather than as a general pattern language, which keeps the code short.

```cpp
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using eclass_id = std::uint32_t;

struct enode {
    std::string op;
    std::vector<eclass_id> kids;
    bool operator==(const enode&) const = default;
};

struct enode_hash {
    std::size_t operator()(const enode& n) const {
        std::size_t h = std::hash<std::string>{}(n.op);
        for (eclass_id k : n.kids) h = h * 1000003u ^ k;
        return h;
    }
};

class egraph {
    struct eclass {
        std::vector<enode> nodes;
        std::vector<std::pair<enode, eclass_id>> parents;
    };

    std::vector<eclass_id> uf_;
    std::unordered_map<enode, eclass_id, enode_hash> memo_;
    std::unordered_map<eclass_id, eclass> classes_;
    std::vector<eclass_id> pending_;

    enode canonical(enode n) {
        for (auto& k : n.kids) k = find(k);
        return n;
    }

    void repair(eclass_id c) {
        auto parents = std::move(classes_[c].parents);
        classes_[c].parents.clear();
        for (auto& [n, p] : parents) {
            memo_.erase(n);
            n = canonical(n);
            memo_[n] = find(p);
        }
        std::unordered_map<enode, eclass_id, enode_hash> unique;
        for (const auto& [n, p] : parents) {
            const enode cn = canonical(n);
            if (auto it = unique.find(cn); it != unique.end()) merge(it->second, p);
            unique[cn] = find(p);
        }
        auto& cls = classes_[find(c)];
        for (auto& [n, p] : unique) cls.parents.emplace_back(n, find(p));
    }

public:
    eclass_id find(eclass_id a) {
        while (uf_[a] != a) a = uf_[a] = uf_[uf_[a]];
        return a;
    }

    eclass_id add(enode n) {
        n = canonical(std::move(n));
        if (auto it = memo_.find(n); it != memo_.end()) return find(it->second);
        const auto id = static_cast<eclass_id>(uf_.size());
        uf_.push_back(id);
        for (eclass_id k : n.kids) classes_[k].parents.emplace_back(n, id);
        classes_[id].nodes.push_back(n);
        memo_[n] = id;
        return id;
    }

    eclass_id leaf(const std::string& op) { return add({op, {}}); }

    bool merge(eclass_id a, eclass_id b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        if (classes_[a].nodes.size() < classes_[b].nodes.size()) std::swap(a, b);
        uf_[b] = a;
        auto& from = classes_[b];
        auto& into = classes_[a];
        into.nodes.insert(into.nodes.end(), from.nodes.begin(), from.nodes.end());
        into.parents.insert(into.parents.end(), from.parents.begin(), from.parents.end());
        classes_.erase(b);
        pending_.push_back(a);
        return true;
    }

    void rebuild() {
        while (!pending_.empty()) {
            auto todo = std::move(pending_);
            pending_.clear();
            for (eclass_id c : todo) repair(find(c));
        }
    }

    bool has_leaf(eclass_id c, const std::string& op) {
        for (const auto& n : classes_[find(c)].nodes)
            if (n.op == op && n.kids.empty()) return true;
        return false;
    }

    template <class F>
    void for_each_node(F&& f) {
        std::vector<std::pair<eclass_id, enode>> all;
        for (const auto& [id, cls] : classes_)
            for (const auto& n : cls.nodes) all.emplace_back(id, n);
        for (auto& [id, n] : all) f(id, canonical(n));
    }

    const std::vector<enode>& nodes_of(eclass_id c) { return classes_[find(c)].nodes; }

    std::string extract(eclass_id root, const std::function<double(const std::string&)>& cost) {
        constexpr double inf = std::numeric_limits<double>::infinity();
        std::unordered_map<eclass_id, std::pair<double, enode>> best;
        for (bool changed = true; changed;) {
            changed = false;
            for (const auto& [id, cls] : classes_) {
                for (const auto& n : cls.nodes) {
                    double c = cost(n.op);
                    for (eclass_id k : n.kids) {
                        auto it = best.find(find(k));
                        c = it == best.end() ? inf : c + it->second.first;
                    }
                    auto it = best.find(id);
                    if (c < inf && (it == best.end() || c < it->second.first)) {
                        best.insert_or_assign(id, std::pair{c, canonical(n)});
                        changed = true;
                    }
                }
            }
        }
        std::function<std::string(eclass_id)> show = [&](eclass_id c) {
            const enode& n = best.at(find(c)).second;
            if (n.kids.empty()) return n.op;
            return "(" + show(n.kids[0]) + " " + n.op + " " + show(n.kids[1]) + ")";
        };
        return show(root);
    }
};

using rule = std::function<bool(egraph&)>;

rule mul_two_to_shift() {      // x * 2  =>  x << 1
    return [](egraph& g) {
        std::vector<std::pair<eclass_id, eclass_id>> hits;
        g.for_each_node([&](eclass_id c, const enode& n) {
            if (n.op == "*" && g.has_leaf(n.kids[1], "2")) hits.emplace_back(c, n.kids[0]);
        });
        bool changed = false;
        for (auto [c, x] : hits) changed |= g.merge(c, g.add({"<<", {x, g.leaf("1")}}));
        return changed;
    };
}

rule reassociate_div() {       // (x * y) / z  =>  x * (y / z)
    return [](egraph& g) {
        std::vector<std::pair<eclass_id, enode>> hits;
        g.for_each_node([&](eclass_id c, const enode& n) {
            if (n.op != "/") return;
            for (const auto& m : g.nodes_of(n.kids[0]))
                if (m.op == "*") hits.emplace_back(c, enode{"", {m.kids[0], m.kids[1], n.kids[1]}});
        });
        bool changed = false;
        for (auto& [c, xyz] : hits) {
            const auto quotient = g.add({"/", {xyz.kids[1], xyz.kids[2]}});
            changed |= g.merge(c, g.add({"*", {xyz.kids[0], quotient}}));
        }
        return changed;
    };
}

rule div_self() {              // x / x  =>  1   (sound only where x != 0)
    return [](egraph& g) {
        std::vector<eclass_id> hits;
        g.for_each_node([&](eclass_id c, const enode& n) {
            if (n.op == "/" && g.find(n.kids[0]) == g.find(n.kids[1])) hits.push_back(c);
        });
        bool changed = false;
        for (auto c : hits) changed |= g.merge(c, g.leaf("1"));
        return changed;
    };
}

rule mul_one() {               // x * 1  =>  x
    return [](egraph& g) {
        std::vector<std::pair<eclass_id, eclass_id>> hits;
        g.for_each_node([&](eclass_id c, const enode& n) {
            if (n.op == "*" && g.has_leaf(n.kids[1], "1")) hits.emplace_back(c, n.kids[0]);
        });
        bool changed = false;
        for (auto [c, x] : hits) changed |= g.merge(c, x);
        return changed;
    };
}

int main() {
    egraph g;
    const auto a = g.leaf("a");
    const auto two = g.leaf("2");
    const auto root = g.add({"/", {g.add({"*", {a, two}}), two}});   // (a * 2) / 2

    const std::vector<rule> rules = {mul_two_to_shift(), reassociate_div(), div_self(), mul_one()};
    for (int iter = 0; iter < 10; ++iter) {
        bool changed = false;
        for (const auto& r : rules) changed |= r(g);
        g.rebuild();
        if (!changed) break;
    }

    auto cost = [](const std::string& op) {
        if (op == "*") return 4.0;
        if (op == "/") return 8.0;
        return 1.0;
    };
    std::cout << g.extract(root, cost) << '\n';   // prints: a
}
```

On `(a * 2) / 2`, the rules add `a << 1` to the class of `a * 2`, add `a * (2 / 2)` to the root class, merge `2 / 2` with `1`, and then merge `a * 1` (now the same e-node as `a * (2 / 2)`) with `a`. The root class now contains `a`, and extraction returns it. A destructive rewriter that applied the attractive strength reduction `a * 2 -> a << 1` first would have destroyed the `a * 2` that the cancellation needs; the e-graph keeps both and lets the cost model decide at the end.

The `div_self` rule shows a real hazard: `x / x = 1` is false when `x` is zero. Rewrite systems in compilers must be sound under the IR's semantics, including undefined behaviour and poison, and tools such as Alive2 are used to check rules before they are admitted.

== E-Class Analyses

Many rewrites need facts about a class, not just its syntax: its constant value, its sign, whether it can be zero, its type. An *e-class analysis* attaches a value from a semilattice to each class, computes it from the e-nodes (`make`), combines values when classes merge (`join`), and can modify the class in response (`modify`, for example adding a constant leaf when constant folding succeeds). Because classes only merge, never split, the analysis value can only grow up the lattice and converges. Constant folding, interval analysis for guarding rules like `div_self`, and type information are all implemented this way.

== Applications

#table(
  columns: 3,
  [*System*], [*Domain*], [*Use of e-graphs*],
  [Denali (2002)], [Superoptimisation of machine code], [Equivalent instruction sequences, with a SAT-based search for the shortest],
  [Peggy (Tate et al.)], [Java bytecode optimisation], [Introduced "equality saturation", with loops represented as program expression graphs],
  [Herbie], [Floating-point accuracy], [Rewrites real-number identities to find expressions with less rounding error],
  [Cranelift], [Production code generator for Wasmtime], [Mid-end optimiser on an *acyclic e-graph* (aegraph) over SSA, with ISLE rules],
  [Tensat], [Deep-learning graph rewriting], [Equality saturation over tensor operator graphs, ILP extraction],
  [Ruler, Enumo], [Rule synthesis], [Enumerate terms in an e-graph to discover and verify new rewrite rules],
  [Szalinski], [CAD programs], [Shrinks flattened CAD meshes into parametric programs with loops],
)

Cranelift's design shows how the idea adapts to a production compiler with tight compile-time budgets. Rather than saturating, it builds an e-graph from the SSA of each function, applies rewrite rules eagerly as nodes are created (each rewrite runs once, in a bounded way), keeps the e-graph acyclic so that it maps back to SSA cleanly, and then extracts and schedules instructions with loop-invariant code motion and rematerialisation folded into the placement step. It recovers most of the phase-ordering benefit at a compile time comparable to a conventional pass pipeline.

== Limits

- *Blow-up.* Associativity and commutativity make the number of represented terms grow exponentially and can make the e-graph itself grow very large. Node limits, rule scheduling, and specialised AC-matching are standard countermeasures.
- *Extraction quality.* Tree-cost extraction overcounts shared subterms; DAG-cost extraction is NP-hard. Real cost depends on scheduling and register pressure, which the e-graph does not model.
- *Control flow.* E-graphs represent pure expressions naturally. Loops, memory, and side effects need encodings (Peggy's PEGs, Cranelift's side-effecting "skeleton" of instructions that the e-graph does not reorder, or explicit state tokens).
- *Soundness.* Each rule must be correct under the full semantics of the IR, including overflow, NaN, and undefined behaviour.

== Further Reading

Nelson, G. and Oppen, D. C. (1980). "Fast Decision Procedures Based on Congruence Closure." Journal of the ACM.

Downey, P. J., Sethi, R. and Tarjan, R. E. (1980). "Variations on the Common Subexpression Problem." Journal of the ACM.

Joshi, R., Nelson, G. and Randall, K. (2002). "Denali: A Goal-Directed Superoptimizer." PLDI.

Tate, R. et al. (2009). "Equality Saturation: A New Approach to Optimization." POPL.

Panchekha, P. et al. (2015). "Automatically Improving Accuracy for Floating Point Expressions." PLDI.

Willsey, M. et al. (2021). "egg: Fast and Extensible Equality Saturation." POPL.

Zhang, Y. et al. (2022). "Relational E-Matching." POPL.

Zhang, Y. et al. (2023). "Better Together: Unifying Datalog and Equality Saturation." PLDI.

Yang, Y. et al. (2021). "Equality Saturation for Tensor Graph Superoptimization." MLSys.

Nandi, C. et al. (2021). "Rewrite Rule Inference Using Equality Saturation." OOPSLA.

Nandi, C. et al. (2020). "Synthesizing Structured CAD Models with Equality Saturation and Inverse Transformations." PLDI.

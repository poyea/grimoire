#import "../template.typ": definition, lemma, overbar, proof, theorem, xref

= The Untyped Lambda Calculus <untyped-lambda>

The untyped $lambda$-calculus has three syntactic forms and one computation rule, and it is Turing complete.
Church introduced it in the early 1930s as part of a foundation for logic; that larger system turned out to be inconsistent (Kleene and Rosser 1935), but its purely functional core survived and became the first precise definition of an effectively computable function (Church 1936).
Every functional language is this calculus with constants, types and an evaluation strategy added, and every typed calculus in this book refines it by ruling some terms out.

*See also:* #xref("programming-languages", "simply-typed-lambda", label: "Simply-Typed Lambda Calculus"), #xref("programming-languages", "operational-semantics", label: "Operational Semantics"), #xref("programming-languages", "denotational-semantics", label: "Denotational Semantics"), #xref("programming-languages", "computability", label: "Computability and Recursion Theory").

The typed restriction, $lambda^arrow.r$, is the subject of the next chapter.

== Syntax

#definition(name: "Terms")[
Given a countably infinite set of variables $x, y, z, ...$, the $lambda$-terms are
$ M, N ::= x | lambda x . M | M space N $
]

Application associates to the left and binds tighter than abstraction, so $lambda x . M space N space P$ means $lambda x . ((M space N) space P)$.
Nested abstractions are abbreviated: $lambda x y z . M$ means $lambda x . lambda y . lambda z . M$.
Every function takes one argument; a function of several arguments is *curried* into a chain of one-argument functions.

*Free variables.*
$ "FV"(x) &= {x} \
"FV"(lambda x . M) &= "FV"(M) \\ {x} \
"FV"(M space N) &= "FV"(M) union "FV"(N) $

A term with no free variables is *closed*, also called a *combinator*.

*$alpha$-equivalence.* Terms that differ only in the names of bound variables are identified: $lambda x . x equiv lambda y . y$.
From here on a "term" means an $alpha$-equivalence class, and we adopt *Barendregt's variable convention*: in any statement, the bound variables are chosen distinct from each other and from all free variables in play.

#definition(name: "Capture-avoiding substitution")[
$ x[x := N] &= N \
y[x := N] &= y && (y eq.not x) \
(M_1 space M_2)[x := N] &= M_1[x := N] space M_2[x := N] \
(lambda y . M)[x := N] &= lambda y . M[x := N] && (y eq.not x, space y in.not "FV"(N)) $
]

The side condition on the last clause is always satisfiable by renaming $y$ first.
Without it, $(lambda y . x)[x := y]$ would become $lambda y . y$, *capturing* the free $y$ and changing the meaning of the term.
Getting substitution right is the main implementation burden of the calculus, and the reason for the nameless representation later in the chapter.

== Reduction

#definition(name: "$beta$ and $eta$")[
$ (lambda x . M) space N &arrow.r_beta M[x := N] \
lambda x . M space x &arrow.r_eta M && (x in.not "FV"(M)) $
]

A term of the form $(lambda x . M) space N$ is a *$beta$-redex* and $M[x := N]$ is its *contractum*.
One-step reduction $arrow.r_beta$ is the compatible closure of the rule (a redex may be contracted anywhere in a term, including under $lambda$); $arrow.r_beta^*$ is its reflexive-transitive closure and $=_beta$, *$beta$-convertibility*, its reflexive, symmetric and transitive closure.
The $eta$-rule makes the theory *extensional*: if $M space x =_beta N space x$ for a fresh $x$, then $M =_(beta eta) N$.

#definition(name: "Normal forms")[
- $M$ is in *$beta$-normal form* if it contains no $beta$-redex. Normal forms have the shape $lambda x_1 ... x_n . y space N_1 ... N_k$ with each $N_i$ normal.
- $M$ is in *head normal form* if it has the shape $lambda x_1 ... x_n . y space M_1 ... M_k$, with the $M_i$ arbitrary. The variable $y$ is the *head variable*.
- $M$ is in *weak head normal form* if it is an abstraction or of the shape $y space M_1 ... M_k$.
]

Each class contains the previous one: $lambda x . (lambda y . y) space x$ is in weak head normal form but not head normal form, and $x space ((lambda y . y) space z)$ is in head normal form but not normal form.
Functional-language evaluators stop at weak head normal form; proof assistants comparing types may need full normal forms.

Not every term has a normal form.
With $omega = lambda x . x space x$ and $Omega = omega space omega$,
$ Omega = (lambda x . x space x) space (lambda x . x space x) arrow.r_beta (lambda x . x space x) space (lambda x . x space x) = Omega $
so $Omega$ reduces only to itself.
Some terms have a normal form along one path and diverge along another: $(lambda x . y) space Omega$ reduces to $y$ in one step, or to itself forever.
Whether we find the normal form depends on which redex we choose, which is the subject of the next two sections.

== Confluence

#theorem(name: "Church–Rosser 1936")[
If $M arrow.r_beta^* N_1$ and $M arrow.r_beta^* N_2$, there is a $P$ with $N_1 arrow.r_beta^* P$ and $N_2 arrow.r_beta^* P$.
]

The obvious attempt, showing that one-step reduction has the diamond property, fails: contracting a redex can duplicate another redex, so closing the diamond can take several steps on one side.
The standard proof, due to Tait and Martin-Löf, replaces one-step reduction by a relation that contracts many redexes at once.

#definition(name: "Parallel reduction")[
$M arrow.r.double N$ is defined inductively:
$ x arrow.r.double x, quad
(M arrow.r.double M') / (lambda x . M arrow.r.double lambda x . M'), quad
(M arrow.r.double M' quad N arrow.r.double N') / (M space N arrow.r.double M' space N'), quad
(M arrow.r.double M' quad N arrow.r.double N') / ((lambda x . M) space N arrow.r.double M'[x := N']) $
]

#lemma[
$arrow.r_beta subset.eq arrow.r.double subset.eq arrow.r_beta^*$, so the reflexive-transitive closure of $arrow.r.double$ is $arrow.r_beta^*$.
]

#lemma(name: "Takahashi 1995")[
Define the *complete development* $M^*$, which contracts every redex present in $M$:
$ x^* = x, quad (lambda x . M)^* = lambda x . M^*, quad ((lambda x . M) space N)^* = M^*[x := N^*], quad (M space N)^* = M^* space N^* "otherwise". $
If $M arrow.r.double N$ then $N arrow.r.double M^*$.
]

#proof[
Induction on the derivation of $M arrow.r.double N$, using the substitution lemma: if $M arrow.r.double M'$ and $N arrow.r.double N'$ then $M[x := N] arrow.r.double M'[x := N']$.
]

The lemma gives the diamond property for $arrow.r.double$ directly, since any two parallel reducts of $M$ both reduce in parallel to $M^*$.
A relation with the diamond property has a confluent reflexive-transitive closure (tile the diamonds), and that closure is $arrow.r_beta^*$, which proves the theorem.
The same argument works for $beta eta$.

*Consequences.*
- A term has at most one normal form, so a normal form is a well-defined *value* of a term.
- If $M =_beta N$, then $M$ and $N$ have a common reduct (the conversion zig-zag can be straightened out).
- The theory $lambda beta$ is *consistent*: distinct normal forms such as $lambda x y . x$ and $lambda x y . y$ are not convertible, so not all equations are provable.

== Reduction Strategies

A *strategy* picks which redex to contract.
The four classical choices differ in whether they evaluate arguments first and whether they reduce under $lambda$.

#table(
  columns: (auto, auto, auto, auto),
  [*Strategy*], [*Redex chosen*], [*Under $lambda$?*], [*Stops at*],
  [Normal order], [Leftmost-outermost], [Yes], [Normal form],
  [Applicative order], [Leftmost-innermost], [Yes], [Normal form],
  [Call-by-name], [Leftmost-outermost, not in arguments], [No], [Weak head normal form],
  [Call-by-value], [Leftmost redex whose argument is a value], [No], [Weak head normal form],
)

On $(lambda x . y) space Omega$, normal order contracts the outer redex and returns $y$; applicative order first reduces $Omega$, and never finishes.
That example is the whole story in miniature: a strategy that evaluates arguments before knowing whether they are needed can diverge on a term that has a normal form.
The converse cannot happen.

#theorem(name: "Standardisation, Curry–Feys 1958")[
If $M arrow.r_beta^* N$, there is a *standard* reduction from $M$ to $N$: one that contracts redexes from left to right, never contracting a redex to the left of one already contracted (or one that descends from such a redex).
]

#theorem(name: "Normalisation")[
If $M$ has a normal form, normal-order reduction reaches it.
]

#proof[
Let $M arrow.r_beta^* N$ with $N$ normal, and take a standard reduction from $M$ to $N$.
A standard reduction that ends in a normal form contracts the leftmost-outermost redex at each step: a step that skipped it could never come back to it, and a redex that is never contracted would remain in $N$.
So the standard reduction is the normal-order reduction.
]

The weak-head analogue holds for call-by-name: if $M$ has a weak head normal form, CBN reaches one.
Programming languages nevertheless mostly use call-by-value, because it is cheaper (each argument is evaluated once rather than once per use) and because it makes the order of side effects predictable.
Call-by-need, or lazy evaluation, recovers the termination behaviour of call-by-name with the cost of call-by-value by evaluating each argument at most once and sharing the result (Wadsworth 1971).
#xref("programming-languages", "operational-semantics", label: "Operational Semantics") gives the strategies as structural operational semantics and abstract machines.

== Fixed Points and Recursion

The calculus has no names for functions, so a recursive definition such as
$ "fact" = lambda n . "if" space (n = 0) space 1 space (n times "fact" space (n - 1)) $
is not directly a term.
It is, however, an *equation* $"fact" = F space "fact"$ with $F = lambda f . lambda n . "if" space (n = 0) space 1 space (n times f space (n - 1))$, and every such equation has a solution.

#theorem(name: "First fixed-point theorem")[
For every term $F$ there is a term $X$ with $F space X =_beta X$.
]

#proof[
Let $W = lambda x . F space (x space x)$ and $X = W space W$.
Then $X = (lambda x . F space (x space x)) space W arrow.r_beta F space (W space W) = F space X$.
]

Abstracting over $F$ gives Curry's *paradoxical combinator*
$ Y = lambda f . (lambda x . f space (x space x)) space (lambda x . f space (x space x)) $
with $Y space F =_beta F space (Y space F)$.
The equation holds only up to conversion: $Y space F$ and $F space (Y space F)$ have a common reduct, but $Y space F$ does not reduce *to* $F space (Y space F)$.
Turing's combinator does:
$ Theta = (lambda x y . y space (x space x space y)) space (lambda x y . y space (x space x space y)), quad Theta space F arrow.r_beta^* F space (Theta space F). $

Under call-by-value both combinators loop, because $x space x$ is evaluated before $f$ is called.
The fix is to $eta$-expand the self-application so it is delayed until an argument arrives:
$ Z = lambda f . (lambda x . f space (lambda v . x space x space v)) space (lambda x . f space (lambda v . x space x space v)) $
$Z$ is the standard fixed-point combinator for strict languages, and the usual way to write recursion in a call-by-value language that lacks a recursive binding form.
The construction is also the reason the type systems of the following chapters reject self-application: once $x space x$ is typable, a fixed-point combinator is typable, every type is inhabited, and (as #xref("programming-languages", "simply-typed-lambda-extensions", label: "Simply-Typed Lambda Calculus: Extensions and Advanced Topics") explains for $"fix"$) the type system stops being a sound logic.

== Church Encodings

Pure $lambda$-terms can represent data: a value is encoded as the function that performs case analysis or iteration over it.

*Booleans.* A boolean chooses between two continuations.
$ "true" = lambda t f . t, quad "false" = lambda t f . f, quad "if" = lambda b space t space f . b space t space f $
$ "and" = lambda p q . p space q space "false", quad "or" = lambda p q . p space "true" space q, quad "not" = lambda p . p space "false" space "true" $

*Pairs.* A pair waits for a selector.
$ "pair" = lambda a b s . s space a space b, quad "fst" = lambda p . p space "true", quad "snd" = lambda p . p space "false" $

*Numerals.* The Church numeral $overbar(n)$ iterates a function $n$ times.
$ overbar(n) = lambda f x . f^n space x, quad "succ" = lambda n f x . f space (n space f space x) $
$ "add" = lambda m n f x . m space f space (n space f space x), quad "mul" = lambda m n f . m space (n space f), quad "exp" = lambda m n . n space m, quad "iszero" = lambda n . n space (lambda x . "false") space "true" $
Here $"exp" space overbar(m) space overbar(n) arrow.r_beta^* overbar(m^n)$.

*Predecessor.* Iteration only ever moves forward, so subtracting one is not obvious; Kleene found the solution in 1932, which convinced Church that $lambda$-definability might capture all computable functions.
The trick is to iterate on pairs: start from $(overbar(0), overbar(0))$, step $(a, b) |-> (b, b + 1)$, and take the first component after $n$ steps:
$ "pred" = lambda n . "fst" space (n space (lambda p . "pair" space ("snd" space p) space ("succ" space ("snd" space p))) space ("pair" space overbar(0) space overbar(0))) $
After $n$ steps the pair is $(overbar(n - 1), overbar(n))$ for $n >= 1$, and $(overbar(0), overbar(0))$ for $n = 0$.
Predecessor costs $O(n)$ steps, and subtraction $"sub" = lambda m n . n space "pred" space m$ costs $O(m n)$; this inefficiency is inherent to the Church encoding, which supports iteration but not direct case analysis.

*Scott encoding.* The alternative encodes a number by its case analysis:
$ overbar(0)_S = lambda z s . z, quad overbar(n + 1)_S = lambda z s . s space overbar(n)_S $
Predecessor is now $lambda n . n space overbar(0)_S space (lambda m . m)$, a constant number of steps, but iteration requires a fixed-point combinator.
The two encodings correspond to two eliminators: Church numerals are folds, Scott numerals are pattern matches.

*Lists* follow the same pattern.
The Church list $[a, b, d]$ is its own right fold, $lambda c space n . c space a space (c space b space (c space d space n))$, so `map`, `sum` and `length` are one-line folds; this is the encoding behind *foldr/build fusion* in GHC.

== $lambda$-Definability and Undecidability

#definition[
A partial function $f : NN^k harpoon.rt NN$ is *$lambda$-definable* if there is a term $F$ such that $F space overbar(n_1) ... overbar(n_k) =_beta overbar(f(n_1, ..., n_k))$ whenever $f(n_1, ..., n_k)$ is defined, and $F space overbar(n_1) ... overbar(n_k)$ has no normal form otherwise.
]

#theorem(name: "Kleene 1936; Turing 1937")[
The $lambda$-definable functions are exactly the partial recursive functions, and exactly the Turing-computable functions.
]

#proof[
Recursive functions are $lambda$-definable: the initial functions are $overbar(0)$, $"succ"$ and projections, composition is application, primitive recursion is iteration on pairs as in $"pred"$, and minimisation is a search loop built with a fixed-point combinator.
Conversely, $beta$-reduction on Gödel-numbered terms is a recursive relation, so the normal-order reducer is a partial recursive function.
Turing (1937) proved the equivalence with his machines directly.
]

This equivalence is the evidence behind the *Church–Turing thesis*: three independently motivated definitions of computability (recursive functions, $lambda$-definability, Turing machines) define the same class.

Terms can be represented inside the calculus.
Fix a Gödel numbering $\#M$ of terms and write $ceil(M) = overbar(\#M)$ for the numeral of the code.

#theorem(name: "Second fixed-point theorem, Kleene")[
For every term $F$ there is a term $X$ with $F space ceil(X) =_beta X$.
]

A term can therefore compute with its own code, which is the $lambda$-calculus version of Kleene's recursion theorem (see #xref("programming-languages", "computability", label: "Computability and Recursion Theory")) and the basis of the undecidability results.

#theorem(name: "Scott–Curry")[
Let $A$ and $B$ be nonempty sets of terms, each closed under $=_beta$.
Then $A$ and $B$ are not recursively separable: there is no decidable set containing $A$ and disjoint from $B$.
]

#proof[
Suppose $C$ is decidable, $A subset.eq C$ and $B inter C = emptyset$.
Pick $M_A in A$ and $M_B in B$.
Since $C$ is decidable, there is a term $F$ with $F space ceil(X) =_beta M_B$ if $X in C$ and $F space ceil(X) =_beta M_A$ otherwise.
By the second fixed-point theorem, take $X$ with $F space ceil(X) =_beta X$.
If $X in C$ then $X =_beta M_B in B$, so $X in B$ and $X in.not C$; if $X in.not C$ then $X =_beta M_A in A subset.eq C$.
Both cases contradict.
]

#theorem(name: "Church 1936")[
It is undecidable whether a term has a normal form, and whether two terms are $beta$-convertible.
]

#proof[
For the first, apply Scott–Curry with $A$ the terms that have a normal form and $B$ the terms that do not; both are closed under conversion by confluence.
For the second, fix $N$: the set ${M | M =_beta N}$ is nonempty, closed under conversion, and its complement is also nonempty, so it is not decidable.
]

Church's paper appeared a few months before Turing's halting-problem paper.
Scott–Curry also yields Rice's theorem for the calculus at once: every nontrivial property of terms that respects $=_beta$ is undecidable.

== Solvability and Meaning

What should the *value* of a term without a normal form be?
Treating every such term as "undefined" fails.
The terms $M = lambda x . x space "true" space Omega$ and $N = lambda x . x space "false" space Omega$ both lack normal forms, yet $M space (lambda a b . a)$ reduces to $"true"$ and $N space (lambda a b . a)$ to $"false"$, so equating $M$ with $N$ would equate $"true"$ with $"false"$ and make every equation provable.
These terms produce useful output even though they never finish.
The right notion of "undefined" is narrower.

#definition(name: "Solvability")[
A closed term $M$ is *solvable* if there are terms $N_1, ..., N_k$ with $M space N_1 ... N_k =_beta lambda x . x$.
An open term is solvable if its closure $lambda arrow(x) . M$ is.
]

#theorem(name: "Wadsworth 1971")[
A term is solvable if and only if it has a head normal form.
]

A term in head normal form $lambda arrow(x) . y space M_1 ... M_k$ has already committed to its outermost structure; no further reduction can change $y$ or the number of arguments.
Unsolvable terms such as $Omega$ never commit to anything, and they are the right model of the undefined value.
The theory $cal(H)$ that equates all unsolvable terms is consistent (Barendregt 1984), and its maximal consistent extension $cal(H)^*$ is the equational theory of Scott's $D_infinity$ model: two terms are equal there exactly when no context can tell them apart by making one solvable and the other unsolvable.

#theorem(name: "Böhm 1968")[
Let $M$ and $N$ be closed terms with $beta eta$-normal forms that are not $beta eta$-convertible.
Then there are terms $L_1, ..., L_k$ with $M space L_1 ... L_k =_beta "true"$ and $N space L_1 ... L_k =_beta "false"$.
]

Böhm's theorem says that distinct normal forms can be *separated* by a context.
It has a strong corollary: adding any equation $M = N$ between distinct $beta eta$-normal forms to $lambda beta eta$ is inconsistent, since from it $"true" = "false"$ follows.
So $lambda beta eta$ is maximal on normalising terms, and *observational equivalence* (no context distinguishes them) coincides with $beta eta$-convertibility there.
Böhm's proof inspects the *Böhm trees* of $M$ and $N$, the possibly infinite trees of their iterated head normal forms, which are the standard way to give a value to every term, normalising or not.

== Models

A model of the calculus needs a domain $D$ whose elements are also functions on $D$, so that self-application makes sense.
By Cantor's theorem the full function space $D arrow.r D$ is strictly larger than $D$ whenever $D$ has more than one element, so no set-theoretic model with all functions can exist.
Scott (1969) resolved this by restricting to *continuous* functions on complete partial orders: he constructed a domain $D_infinity$ with an isomorphism $D_infinity tilde.equiv [D_infinity arrow.r D_infinity]$ as the limit of $D_0$, $D_1 = [D_0 arrow.r D_0]$, $D_2 = [D_1 arrow.r D_1]$, and so on.
This was the first model of the untyped calculus and the origin of domain theory.
Simpler *graph models* ($P omega$, Plotkin 1972 and Scott 1976; Engeler's model) interpret a term as the set of finite input-output pairs it realises.
#xref("programming-languages", "denotational-semantics", label: "Denotational Semantics") builds these domains and uses them for programming languages; #xref("programming-languages", "categorical-semantics", label: "Categorical Semantics") describes the models abstractly as reflexive objects in a Cartesian closed category.

== Typing the Untyped Calculus

Type systems can be read as tools for *classifying* untyped terms.
Simple types assign a type to $lambda x . x$ but not to $omega = lambda x . x space x$, since $x$ would need both a type $sigma$ and a type $sigma arrow.r tau$; consequently every simply typable term is strongly normalising, which the next chapter proves.
The converse fails: $lambda x . x space x$ is in normal form but not simply typable.

*Intersection types* (Coppo and Dezani-Ciancaglini 1980; Pottinger 1980) close the gap.
Allowing a variable to have several types at once, $x : sigma inter (sigma arrow.r tau)$, makes $omega$ typable, and the result is a *characterisation*: a term is typable in the intersection type system without a universal type if and only if it is strongly normalising.
Since strong normalisation is undecidable, so is intersection typability; the system is a precise logic of termination, not a practical type checker.
System F lies between: it types $lambda x . x space x$, but still only strongly normalising terms (see #xref("programming-languages", "system-f-and-parametricity", label: "System F and Parametricity")).

== Nameless Terms and a Normaliser

Implementing substitution with names requires generating fresh variables and renaming on every capture risk.
*De Bruijn indices* (de Bruijn 1972) remove names altogether: a variable is replaced by the number of binders between its occurrence and its own binder.
$ lambda x . lambda y . x space (lambda z . z space y) quad arrow.r.squiggly quad lambda . lambda . 1 space (lambda . 0 space 1) $
$alpha$-equivalent terms are now syntactically identical, so equality is structural, and capture is impossible.
The price is that substitution must *shift* indices when a term moves under a binder.

#definition(name: "Shifting and substitution")[
$arrow.t^d_c$ adds $d$ to every index $>= c$ (the free ones, at cutoff $c$):
$ arrow.t^d_c (k) = cases(k & "if" k < c, k + d & "if" k >= c), quad arrow.t^d_c (lambda . M) = lambda . arrow.t^d_(c+1) M, quad arrow.t^d_c (M space N) = arrow.t^d_c M space arrow.t^d_c N $
Substitution of $N$ for index $j$:
$ k[j := N] = cases(N & "if" k = j, k & "if" k eq.not j), quad (lambda . M)[j := N] = lambda . M[j + 1 := arrow.t^1_0 N], quad (M_1 space M_2)[j := N] = M_1[j := N] space M_2[j := N] $
The $beta$-rule becomes $(lambda . M) space N arrow.r_beta arrow.t^(-1)_0 (M[0 := arrow.t^1_0 N])$.
]

The following C++ normaliser implements normal-order reduction on de Bruijn terms.
Terms are immutable and shared through `std::shared_ptr`, so unchanged subterms are reused rather than copied.

```cpp
#include <cstddef>
#include <memory>
#include <optional>

struct Term;
using TermPtr = std::shared_ptr<const Term>;

struct Term {
    enum class Kind { var, lam, app };
    Kind kind;
    int index = 0;      // var: de Bruijn index
    TermPtr body;       // lam: body
    TermPtr fn, arg;    // app: function and argument
};

TermPtr make_var(int i) {
    return std::make_shared<const Term>(Term{Term::Kind::var, i, nullptr, nullptr, nullptr});
}
TermPtr make_lam(TermPtr b) {
    return std::make_shared<const Term>(Term{Term::Kind::lam, 0, std::move(b), nullptr, nullptr});
}
TermPtr make_app(TermPtr f, TermPtr a) {
    return std::make_shared<const Term>(
        Term{Term::Kind::app, 0, nullptr, std::move(f), std::move(a)});
}

// Add d to every index >= cutoff.
TermPtr shift(const TermPtr& t, int d, int cutoff) {
    switch (t->kind) {
    case Term::Kind::var:
        return t->index >= cutoff ? make_var(t->index + d) : t;
    case Term::Kind::lam:
        return make_lam(shift(t->body, d, cutoff + 1));
    case Term::Kind::app:
        return make_app(shift(t->fn, d, cutoff), shift(t->arg, d, cutoff));
    }
    return t;
}

// t[j := s]
TermPtr subst(const TermPtr& t, int j, const TermPtr& s) {
    switch (t->kind) {
    case Term::Kind::var:
        return t->index == j ? s : t;
    case Term::Kind::lam:
        return make_lam(subst(t->body, j + 1, shift(s, 1, 0)));
    case Term::Kind::app:
        return make_app(subst(t->fn, j, s), subst(t->arg, j, s));
    }
    return t;
}

TermPtr beta(const TermPtr& body, const TermPtr& arg) {
    return shift(subst(body, 0, shift(arg, 1, 0)), -1, 0);
}

// One normal-order step, or nullopt if t is in normal form.
std::optional<TermPtr> step(const TermPtr& t) {
    switch (t->kind) {
    case Term::Kind::var:
        return std::nullopt;
    case Term::Kind::lam:
        if (auto b = step(t->body)) return make_lam(*b);
        return std::nullopt;
    case Term::Kind::app:
        if (t->fn->kind == Term::Kind::lam) return beta(t->fn->body, t->arg);
        if (auto f = step(t->fn)) return make_app(*f, t->arg);
        if (auto a = step(t->arg)) return make_app(t->fn, *a);
        return std::nullopt;
    }
    return std::nullopt;
}

// Normalise with a step budget; nullopt means the budget ran out.
std::optional<TermPtr> normalise(TermPtr t, std::size_t fuel) {
    for (std::size_t i = 0; i < fuel; ++i) {
        auto next = step(t);
        if (!next) return t;
        t = std::move(*next);
    }
    return std::nullopt;
}
```

The step function encodes the strategy in its case order: an application whose function is an abstraction is the leftmost-outermost redex; otherwise the function is searched before the argument.
Searching the function and the argument for redexes *before* contracting the outer redex would give applicative order instead.
The step budget is not optional: by Church's theorem no implementation can decide in advance whether normalisation terminates.

*Cost.* Each step walks the term to find the redex and rebuilds the path to it, so a step costs $O(|t|)$ time, and the shifts inside $"beta"$ add a pass over the argument for every substituted occurrence.
Serious implementations avoid substitution entirely: environment machines such as Krivine's machine for call-by-name and the CEK machine for call-by-value keep a pointer to the body and an environment of closures, and normalisation by evaluation (see #xref("programming-languages", "simply-typed-lambda-extensions", label: "Simply-Typed Lambda Calculus: Extensions and Advanced Topics")) reuses the host language's own closures.

== The Cost of Reduction

Is the number of $beta$-steps a fair measure of running time?
Not obviously: a term can double in size with each step, so after $n$ steps its size can be $2^n$, a phenomenon called *size explosion*.
Accattoli and Dal Lago (2016) nevertheless proved that leftmost-outermost reduction is an *invariant* cost model: a Turing machine can simulate $n$ normal-order steps in time polynomial in $n$ and the size of the initial term, provided the reducer works with *shared* representations (the linear substitution calculus) and never writes out the exploded term.
So the calculus is a reasonable machine model for complexity theory, not only for computability.

Sharing can go further.
Lévy (1978) defined *optimal* reduction, which never duplicates a redex: all copies of a redex that arise from the same origin are contracted in one step.
Lamping (1990) found the first implementation, using graph rewriting with sharing nodes.
Optimal does not mean fast: Asperti and Mairson (1998) showed that the bookkeeping needed to keep reduction optimal can cost non-elementary time relative to the number of Lévy-optimal steps, so the count of optimal steps is not an invariant cost model.

== Summary

- Three term forms and one rule; capture-avoiding substitution is the only subtle operation.
- Reduction is confluent, so normal forms are unique, and normal order finds one whenever it exists; call-by-value may not.
- Every term has a fixed point, so recursion and data (Church and Scott encodings) are definable.
- The $lambda$-definable functions are the computable ones, and every nontrivial property closed under conversion is undecidable.
- Unsolvable terms are the meaningless ones; Böhm's theorem separates distinct normal forms.
- With sharing, normal-order step counts are an invariant cost model.

== Further Reading

Barendregt, H. P. (1984). _The Lambda Calculus: Its Syntax and Semantics_, revised ed. North-Holland. The standard reference: confluence, standardisation, solvability, Böhm trees, models, and the theories $cal(H)$ and $cal(H)^*$.

Hindley, J. R. and Seldin, J. P. (2008). _Lambda-Calculus and Combinators: An Introduction_. Cambridge University Press. A gentler introduction covering the calculus, combinatory logic, representability of recursive functions and simple types side by side.

Church, A. (1936). "An Unsolvable Problem of Elementary Number Theory." _American Journal of Mathematics_ 58(2). Defines $lambda$-definability as effective calculability and proves convertibility undecidable.

Church, A. and Rosser, J. B. (1936). "Some Properties of Conversion." _Transactions of the AMS_ 39(3). The confluence theorem that gives normal forms their meaning.

Kleene, S. C. (1936). "$lambda$-Definability and Recursiveness." _Duke Mathematical Journal_ 2(2). Proves that the $lambda$-definable functions are the recursive functions.

Böhm, C. (1968). "Alcune proprietà delle forme $beta$-$eta$-normali nel $lambda$-K-calcolo." _Pubblicazioni dell'IAC_ 696. The separability theorem for normal forms.

de Bruijn, N. G. (1972). "Lambda Calculus Notation with Nameless Dummies." _Indagationes Mathematicae_ 34(5). Introduces the index representation used by most implementations and proof assistants.

Takahashi, M. (1995). "Parallel Reductions in $lambda$-Calculus." _Information and Computation_ 118(1). The complete-development proof of confluence given in this chapter.

Accattoli, B. and Dal Lago, U. (2016). "(Leftmost-Outermost) Beta Reduction is Invariant, Indeed." _Logical Methods in Computer Science_ 12(1). Shows that counting normal-order steps is a reasonable cost model despite size explosion.

Asperti, A. and Guerrini, S. (1998). _The Optimal Implementation of Functional Programming Languages_. Cambridge University Press. Lévy optimality, Lamping's algorithm, and the cost of sharing graphs.

Pierce, B. C. (2002). _Types and Programming Languages_. MIT Press. Chapters 5–7 present the untyped calculus, de Bruijn indices and an OCaml implementation as the starting point for the typed calculi.

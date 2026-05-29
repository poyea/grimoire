#import "template.typ": project

#project("Formal Foundations of Programming Languages")[
  #align(center)[
  #block(
    fill: luma(245),
    inset: 10pt,
    width: 80%,
    text(size: 9pt)[
      #align(center)[
        #text(size: 24pt, weight: "bold")[Formal Foundations of Programming Languages]
        #linebreak()
        #text(size: 11pt, style: "italic")[Automata, Semantics, Type Theory, and Compilers]
      ]
    ]
  )
]

#outline(
  title: "Table of Contents",
  depth: 2,
)

#emph[Enjoy.]

#pagebreak()

// ============================================================================
// PART I: Foundations
// ============================================================================

#include "programming-languages/foundations.typ"
#pagebreak()

#include "programming-languages/lexing.typ"
#pagebreak()

#include "programming-languages/parsing.typ"
#pagebreak()

#include "programming-languages/pushdown-cfg.typ"
#pagebreak()

#include "programming-languages/turing-computability.typ"
#pagebreak()

// ============================================================================
// PART II: Automata, Languages, Computability
// ============================================================================

#include "programming-languages/regular-languages.typ"
#pagebreak()

#include "programming-languages/context-free-languages.typ"
#pagebreak()

#include "programming-languages/pushdown-and-beyond.typ"
#pagebreak()

#include "programming-languages/omega-automata.typ"
#pagebreak()

#include "programming-languages/infinite-trees-and-games.typ"
#pagebreak()

#include "programming-languages/tree-automata.typ"
#pagebreak()

#include "programming-languages/timed-and-hybrid-automata.typ"
#pagebreak()

#include "programming-languages/weighted-and-probabilistic-automata.typ"
#pagebreak()

#include "programming-languages/automata-learning.typ"
#pagebreak()

#include "programming-languages/computability.typ"
#pagebreak()

#include "programming-languages/complexity.typ"
#pagebreak()

// ============================================================================
// PART III: Semantics
// ============================================================================

#include "programming-languages/semantics.typ"
#pagebreak()

#include "programming-languages/operational-semantics.typ"
#pagebreak()

#include "programming-languages/denotational-semantics.typ"
#pagebreak()

#include "programming-languages/axiomatic-semantics.typ"
#pagebreak()

#include "programming-languages/categorical-semantics.typ"
#pagebreak()

// ============================================================================
// PART IV: Type Theory
// ============================================================================

#include "programming-languages/type-systems.typ"
#pagebreak()

#include "programming-languages/simply-typed-lambda.typ"
#pagebreak()

#include "programming-languages/system-f-and-parametricity.typ"
#pagebreak()

#include "programming-languages/dependent-types.typ"
#pagebreak()

#include "programming-languages/linear-and-substructural.typ"
#pagebreak()

#include "programming-languages/effects-and-handlers.typ"
#pagebreak()

#include "programming-languages/subtyping-and-polymorphism.typ"
#pagebreak()

#include "programming-languages/gradual-and-hybrid.typ"
#pagebreak()

#include "programming-languages/homotopy-type-theory.typ"
#pagebreak()

// ============================================================================
// PART V: Meta-programming and Concurrency
// ============================================================================

#include "programming-languages/macros-and-metaprogramming.typ"
#pagebreak()

#include "programming-languages/process-calculi.typ"
#pagebreak()

#include "programming-languages/concurrency-models.typ"
#pagebreak()

// ============================================================================
// PART VI: Practice
// ============================================================================

#include "programming-languages/build-a-compiler.typ"
#pagebreak()

= Conclusion

Every layer of a programming language is recognized or executed by an automaton, and the Chomsky hierarchy is the map. Lexers are deterministic finite automata; parsers are pushdown automata over context-free grammars; full evaluators are Turing machines. Beyond the hierarchy, semantics tells you what programs *mean*, type systems prove they cannot go wrong, and category theory explains why the two views agree. The proof of the thesis is a working compiler — not the abstraction.

*Key synthesis:*

- *Regular languages* are closed, decidable, and fast — every layer of a system that can stay in this class (configuration, log filters, JSON shape) gets cheap analysis for free.
- *Context-free grammars* are the universal language of programming-language syntax; LL(k), LR(k), and PEG are engineering trade-offs over the same theory.
- *$omega$-regular languages and parity games* are the language-theoretic engine of model checking: LTL → Büchi → emptiness, modal $mu$-calculus → parity game solving.
- *Turing-completeness* is a ceiling, not a goal — sub-TC languages (regex, SQL, Coq's terminating fragment) trade expressiveness for decidable analysis, and that trade is often the right one.
- *Operational, denotational, and axiomatic semantics* are three lenses on the same artifact; adequacy and full abstraction theorems relate them, and category theory unifies all three via CCC / monad / topos structure.
- *Type systems* via Curry–Howard *are* logics: STLC ↔ IPC, System F ↔ second-order logic, MLTT ↔ first-order intuitionistic predicate calculus, CIC ↔ higher-order constructive mathematics, HoTT ↔ univalent foundations.
- *Hindley–Milner* is the sweet spot of inference power vs decidability; System F gives up the former, dependent types give up the latter, gradual types interpolate.
- *Substructural type systems* (linear, affine, ordered, bunched) encode resource discipline; Rust's borrow checker, session types, and separation logic are all reflections of Girard's linear logic.
- *Algebraic effects and handlers* generalize monads with composable equations; they are the modern operational reading of computational effects.
- *Process calculi* ($pi$, CSP, CCS) and *concurrency models* (event structures, actors, STM, weak memory) formalize what concurrent programs mean before any implementation choice is made.
- *A compiler* is the constructive proof that grammars, semantics, types, and machines compose into something useful.

== Further Reading

Sipser, M. (2012). _Introduction to the Theory of Computation_, 3rd ed. Cengage.

Hopcroft, J., Motwani, R. and Ullman, J. (2006). _Introduction to Automata Theory, Languages, and Computation_, 3rd ed. Addison-Wesley.

Eilenberg, S. (1974, 1976). _Automata, Languages and Machines_, vols A and B. Academic Press.

Sakarovitch, J. (2009). _Elements of Automata Theory_. Cambridge.

Perrin, D. and Pin, J.-É. (2004). _Infinite Words: Automata, Semigroups, Logic and Games_. Elsevier.

Grädel, E., Thomas, W. and Wilke, T. (eds.) (2002). _Automata, Logics, and Infinite Games_. LNCS 2500, Springer.

Arora, S. and Barak, B. (2009). _Computational Complexity: A Modern Approach_. Cambridge.

Aho, A., Lam, M., Sethi, R. and Ullman, J. (2006). _Compilers: Principles, Techniques, and Tools_ (the Dragon Book), 2nd ed. Addison-Wesley.

Pierce, B. (2002). _Types and Programming Languages_. MIT Press.

Pierce, B. (ed.) (2005). _Advanced Topics in Types and Programming Languages_. MIT Press.

Harper, R. (2016). _Practical Foundations for Programming Languages_, 2nd ed. Cambridge.

Winskel, G. (1993). _The Formal Semantics of Programming Languages_. MIT Press.

Mitchell, J. (1996). _Foundations for Programming Languages_. MIT Press.

Streicher, T. (1991). _Semantics of Type Theory_. Birkhäuser.

The Univalent Foundations Program (2013). _Homotopy Type Theory: Univalent Foundations of Mathematics_. IAS.

Mac Lane, S. (1998). _Categories for the Working Mathematician_, 2nd ed. Springer.

Awodey, S. (2010). _Category Theory_, 2nd ed. Oxford.

Sangiorgi, D. and Walker, D. (2001). _The $pi$-Calculus: A Theory of Mobile Processes_. Cambridge.

Milner, R. (1999). _Communicating and Mobile Systems: The $pi$-Calculus_. Cambridge.

Hoare, C. A. R. (1985). _Communicating Sequential Processes_. Prentice Hall.

Appel, A. (2002). _Modern Compiler Implementation in ML_, revised ed. Cambridge.

Nielson, F., Nielson, H. and Hankin, C. (2005). _Principles of Program Analysis_. Springer.
]

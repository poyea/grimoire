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
        #text(size: 11pt, style: "italic")[Grammars, Machines, and Compilers]
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

#include "programming-languages/foundations.typ"
#pagebreak()

#include "programming-languages/pushdown-cfg.typ"
#pagebreak()

#include "programming-languages/turing-computability.typ"
#pagebreak()

#include "programming-languages/lexing.typ"
#pagebreak()

#include "programming-languages/parsing.typ"
#pagebreak()

#include "programming-languages/semantics.typ"
#pagebreak()

#include "programming-languages/type-systems.typ"
#pagebreak()

#include "programming-languages/build-a-compiler.typ"
#pagebreak()

= Conclusion

Every layer of a programming language is recognized or executed by an automaton, and the Chomsky hierarchy is the map. Lexers are deterministic finite automata; parsers are pushdown automata over context-free grammars; full evaluators are Turing machines. Beyond the hierarchy, semantics tells you what programs *mean* and type systems prove they cannot go wrong. The proof of the thesis is a working compiler — not the abstraction.

*Key synthesis:*

- *Regular languages* are closed, decidable, and fast — every layer of a system that can stay in this class (configuration, log filters, JSON shape) gets cheap analysis for free.
- *Context-free grammars* are the universal language of programming-language syntax; LL(k), LR(k), and PEG are engineering trade-offs over the same theory.
- *Turing-completeness* is a ceiling, not a goal — sub-TC languages (regex, SQL, Coq's terminating fragment) trade expressiveness for decidable analysis, and that trade is often the right one.
- *Semantics before types*: type soundness is a theorem about an operational semantics, not a property of a syntax tree.
- *Hindley-Milner* is the sweet spot of inference power vs decidability; System F gives up the former, dependent types give up the latter.
- *A compiler* is the constructive proof that grammars and machines compose into something useful.

== Further Reading

Sipser, M. (2012). _Introduction to the Theory of Computation_, 3rd ed. Cengage.

Hopcroft, J., Motwani, R. and Ullman, J. (2006). _Introduction to Automata Theory, Languages, and Computation_, 3rd ed. Addison-Wesley.

Aho, A., Lam, M., Sethi, R. and Ullman, J. (2006). _Compilers: Principles, Techniques, and Tools_ (the Dragon Book), 2nd ed. Addison-Wesley.

Pierce, B. (2002). _Types and Programming Languages_. MIT Press.

Pierce, B. (ed.) (2005). _Advanced Topics in Types and Programming Languages_. MIT Press.

Winskel, G. (1993). _The Formal Semantics of Programming Languages_. MIT Press.

Appel, A. (2002). _Modern Compiler Implementation in ML_, revised ed. Cambridge.

Nielson, F., Nielson, H. and Hankin, C. (2005). _Principles of Program Analysis_. Springer.
]

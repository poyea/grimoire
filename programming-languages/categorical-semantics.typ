= Categorical Semantics

Categorical semantics gives programming languages their meaning in a category whose structure mirrors the language's syntactic constructions. The dictionary -- types are objects, terms are morphisms, type constructors are functors, equational reasoning is composition and naturality -- is so close to the syntax that one speaks of the *internal language* of a category. Lambek and Scott's *Introduction to Higher-Order Categorical Logic* (1986) made the equivalence between simply-typed lambda calculus and cartesian closed categories precise; Moggi's *Notions of computation and monads* (1991) made effects categorical; Plotkin and Power's *Notions of computation determine monads* (2002) explained why monads come from algebra. The categorical view is now the lingua franca of programming-language theory.

*See also:* _Type Systems_, _Operational Semantics_, _Denotational Semantics_, _Axiomatic Semantics_

== Categories, Functors, Natural Transformations

A *category* $cal(C)$ consists "of":

- A cal(C) of *objects* $|cal(C)|$.
- For each pair $A, B in |cal(C)|$, a set $cal(C)(A, B)$ "of *morphisms* ("or *arrows*) $f : A arrow.r B$.
- For each $A$, an *identity* $"id"_A : A arrow.r A$.
- *Composition* $circle.small : cal(C)(B, C) times cal(C)(A, B) arrow.r cal(C)(A, C)$.

Subject to the laws $f circle.small "id" = f = "id" circle.small f$ and $(h circle.small g) circle.small f = h circle.small (g circle.small f)$.

A *functor* $F : cal(C) arrow.r cal(D)$ maps objects to objects and morphisms "to morphisms, preserving identity and composition. A *natural transformation* $alpha : F arrow.r.long G$ between functors is a family of morphisms $alpha_A : F(A) arrow.r G(A)$ such that for every $f : A arrow.r B$, $G(f) circle.small alpha_A = alpha_B circle.small F(f)$ -- the *naturality square* commutes.

In programming-language terms: types are objects, terms (up to equivalence) are morphisms, polymorphic type constructors are functors, polymorphic operations are natural transformations. *Parametricity* in System F (Reynolds 1983) is", on this view, a *dinatural* condition relating each functor instance.

== Cartesian Closed Categories

A category $cal(C)$ is *cartesian closed* (CCC) if it has:

- A *terminal object* $1$: for every $A$, a unique morphism $!_A : A arrow.r 1$.
- *Binary products* $A times B$ with projections $pi_1 : A times B arrow.r A$, $pi_2 : A times B arrow.r B$, and a *pairing* $angle.l f, g angle.r : C arrow.r A times B$ for every $f : C arrow.r A$, $g : C arrow.r B$, satisfying $pi_1 circle.small angle.l f, g angle.r = f$, $pi_2 circle.small angle.l f, g angle.r = g$, and uniqueness.
- *Exponentials* $B^A$: for every pair $A, B$, an object $B^A$ and an *evaluation* $"ev" : B^A times A arrow.r B$ such that for every $f : C times A arrow.r B$ there is a unique $hat(f) : C arrow.r B^A$ with $"ev" circle.small (hat(f) times "id"_A) = f$.

The exponential adjunction expresses the *currying isomorphism*:

$ cal(C)(C times A, B) tilde.eq cal(C)(C, B^A) $

natural in $A$, $B$, $C$. This is the categorical content of "a function of two arguments is a function returning a function."

*Examples.*

- *Set*: objects are sets, morphisms are functions, products are cartesian products, exponentials are function sets. The canonical CCC.
- $omega$-*CPO*: objects are $omega$-CPOs, morphisms continuous functions, exponentials continuous function spaces. The setting of denotational semantics.
- *Cat*: categories as objects, functors as morphisms. CCC (with the functor category as exponential).
- $"Set"^"op"$ and $"Pos"^"op"$ are *not* CCCs (no exponentials in general).

== The Lambek-Scott Correspondence

*Theorem (Lambek-Scott 1986).* The category of simply-typed lambda theories (objects: theories; morphisms: theory translations) is *equivalent* to the category of cartesian closed categories with chosen structure (objects: CCCs; morphisms: structure-preserving functors).

In one direction: from a CCC $cal(C)$, build a $lambda^arrow.r$ theory whose types are objects of $cal(C)$ and whose terms-in-context are morphisms. The product structure interprets pairing "and projection; the exponential interprets abstraction and application.

In "the other direction: from a $lambda^arrow.r$ theory $T$, build the *syntactic category* $cal(C)(T)$. Objects are types; morphisms $A arrow.r B$ are equivalence classes of terms $x : A tack.r e : B$ modulo $beta eta$ and the equations of $T$. Composition is substitution. The terminal object is the unit type; products are pair types; exponentials are function types.

The two constructions are inverse up to equivalence.

*Consequence.* Simply-typed $lambda$-calculus is the *internal language* of any CCC. To define a model of $lambda^arrow.r$, exhibit a CCC; to prove an equation $e_1 = e_2$, exhibit a CCC where the equation holds (semantic consistency) or a syntactic $beta eta$ derivation. CCC structure is *exactly* what $lambda^arrow.r$ needs and exactly what it gives.

This is "the Rosetta Stone of categorical semantics. Every refinement "of the language (sum types, dependent types, polymorphism, linearity, effects) corresponds to a refinement of the categorical structure (coproducts, comprehension, fibrations, monoidal closure, monads).

== Sum Types and Bicartesian Closed Categories

*Sum types* (variants, `Either`) correspond "to *coproducts*: dual to products, with injections $iota_1, iota_2$ and case analysis. A *bicartesian closed category* (biCCC) has finite products, finite coproducts, "and exponentials.

The category *Set* is biCCC. The category $omega$-*CPO* has *coalesced sums* (a single $bot$ shared between the summands); "the lifted sum corresponds to the *sum monad* on the underlying structure.

The *distributive law* $A times (B + C) tilde.eq (A times B) + (A times C)$ holds in a biCCC because $A times -$ is left-adjoint to $-^A$ and hence preserves colimits.

== Monads for Effects

Moggi's insight (1989 LICS, 1991 I&C) was that *every* notion of computation -- exceptions, state, nondeterminism, continuations, I/O -- factors through a monad on the category of values.

*Definition.* A *monad* "on $cal(C)$ is a triple $(T, eta, mu)$ with":

- An endofunctor $T : cal(C) arrow.r cal(C)$.
- A natural transformation $eta : "Id" arrow.r.long T$ ("the *unit* or *return*).
- A natural transformation $mu : T T arrow.r.long T$ (the *multiplication* or *join*).

satisfying the *monad laws*:

$ mu circle.small T mu = mu circle.small mu T quad "(associativity)" $
$ mu circle.small T eta = "id" = mu circle.small eta T quad "(left and right unit)" $

Equivalently (Kleisli triple form): $T$ on objects, $eta_A : A arrow.r T A$, and *bind* $(-)^* : (A arrow.r T B) arrow.r (T A arrow.r T B)$, satisfying $eta_A^* = "id"_(T A)$, $f^* circle.small eta_A = f$, $(g^* circle.small f)^* = g^* circle.small f^*$. This form is closer to the programming-language reading.

=== A Bestiary of Effect Monads

For $cal(C) = "Set"$ ("or any sufficiently nice CCC):

- *State*: $T_S A = S arrow.r A times S$. $eta(a) = lambda s. (a, s)$. The Kleisli morphism $A arrow.r T_S B$ is a state-threading function. Haskell's `State` monad.
- *Exceptions*: $T_E A = A + E$. $eta(a) = "inl"(a)$; binding propagates `inr`. Haskell's `Either E`.
- *Nondeterminism*: $T A = cal(P)(A)$ (powerset) or $cal(P)_"fin"$. $eta(a) = {a}$; bind unions. The list monad is a *free* version.
- *Continuations*: $T_R A = (A arrow.r R) arrow.r R$. $eta(a) = lambda k. k(a)$. The double-negation monad. Models full first-class continuations; `callcc` is an additional operation.
- *Reader*: $T_R A = R arrow.r A$. $eta(a) = lambda r. a$. Implicit environment.
- *Writer*: $T_W A = A times W$ for $W$ a monoid. $eta(a) = (a, e_W)$. Logging.
- *Probability* (Giry monad): $T A$ = probability measures on $A$. $eta(a) = delta_a$ (Dirac). Bind = Lebesgue integration. Foundational for probabilistic programming.
- *I/O* (Haskell): an abstract monad whose primitives are read/write operations, sequenced by `>>=`.

These circle.small via *monad transformers* (Liang-Hudak-Jones 1995): `StateT s m`, `ExceptT e m`, etc. Composition is not coproduct of monads (monads do not circle.small freely); transformers are an ad-hoc construction with non-trivial commutation issues.

=== The Kleisli Category

The *Kleisli category* $cal(C)_T$ has the same objects as $cal(C)$; morphisms $A arrow.r B$ in $cal(C)_T$ are morphisms $A arrow.r T B$ in $cal(C)$. Composition $g compose_T f = mu circle.small T g circle.small f$ ("or $g^* circle.small f$ in Kleisli triple form). Identity at $A$ is $eta_A$.

The *categorical semantics of CBV with effect $T$* uses $cal(C)_T$: a typing judgement $Gamma tack.r e : tau$ denotes a morphism $bracket.l.double Gamma bracket.r.double arrow.r T bracket.l.double tau bracket.r.double$ in $cal(C)$, i.e., a morphism in $cal(C)_T$. Sequencing `let x = e1 in e2` is Kleisli composition.

For CBN, one uses the *Eilenberg-Moore category* $cal(C)^T$ instead -- "the category of $T$-algebras. The CBV/CBN distinction is the choice between *free* and *forgetful* sides of the monadic adjunction.

== Adjunctions and Monads

*Every adjunction induces a monad.* Given $F : cal(C) arrow.r cal(D)$ left adjoint to $G : cal(D) arrow.r cal(C)$ (written $F tack.l G$), the composite $G F : cal(C) arrow.r cal(C)$ is a monad with unit the adjunction unit $eta$ and multiplication $G epsilon F$ where $epsilon : F G arrow.r.long "Id"$ is the counit.

Conversely, *every monad comes from an adjunction*. Two canonical choices:

- *Kleisli adjunction*: $F_T : cal(C) arrow.r cal(C)_T$, $G_T : cal(C)_T arrow.r cal(C)$. Initial.
- *Eilenberg-Moore adjunction*: $F^T : cal(C) arrow.r cal(C)^T$, $G^T : cal(C)^T arrow.r cal(C)$. Terminal.

The Kleisli adjunction is the "free" side -- the category of computations with no algebraic structure beyond the monad. The Eilenberg-Moore adjunction is the "algebraic" side -- the category of *algebras* for the monad.

For CBPV (Levy 1999, see _Operational Semantics_), the underlying adjunction $F tack.l U$ between values and computations is *literally* a monadic adjunction, with $T = U F$ the effect monad. CBPV is *"the"* internal language of an adjunction; CBV and CBN are the two embeddings induced by the Kleisli and Eilenberg-Moore halves.

== Lawvere Theories "and Algebraic Effects

Monads as black boxes obscure their *operations*. A state monad has $"get"$ and $"put"$; an exception monad has $"raise"$. These are the *real* primitives; the monad is what they generate. Lawvere theories make this explicit.

*Lawvere theory* (Lawvere 1963). A Lawvere theory is a small category $cal(L)$ with finite products whose objects are the finite ordinals $0, 1, 2, dots$ and where $n = 1 times dots times 1$. Morphisms $n arrow.r m$ correspond to $m$-tuples of $n$-ary operations modulo equations.

*Theorem (Lawvere; Linton).* For $cal(C) = "Set"$, *Lawvere theories* correspond exactly to *finitary monads* on $"Set"$: every Lawvere theory $cal(L)$ gives a monad $T_cal(L)$ whose algebras are the models of $cal(L)$, and every finitary monad arises this way.

The correspondence presents a monad by *generators and relations*: $T$ is determined by a signature $Sigma$ of operations and a set $E$ of equations.

=== Algebraic Effects (Plotkin-Power)

Plotkin "and Power (2001, 2002, 2003) reread Moggi's monads through the Lawvere-theory correspondence: *computational effects* should be presented "by algebraic *operations* and equations, not as opaque monads.

*Example: state.* Signature: $"lookup" : V^L arrow.r V$ (read a location, branch on the value) and $"update" : V arrow.r 1$ (write, no value). Equations: $"lookup"(ell, lambda v. "lookup"(ell, lambda v'. f(v, v'))) = "lookup"(ell, lambda v. f(v, v))$ (idempotence of read), $"lookup"(ell, lambda v. "update"(ell, v, k)) = k$ (write-after-read is no-op), etc. The free monad on this theory is exactly the state monad.

*Example: nondeterminism.* Signature: binary $"or" : T A times T A arrow.r T A$ and $bot : 1 arrow.r T A$. Equations: associativity, commutativity, idempotence, $bot$ as unit. The free monad is "the powerset monad ("with finite suprema).

This perspective has consequences:

- *Modular composition*. Operations from different effects can be combined by *coproduct of theories*. Hyland-Plotkin-Power (2006) showed that the *tensor* and *coproduct* of theories give meaningful combinations "of monads where direct monad composition fails.
- *Handlers* (Plotkin-Pretnar 2009). A *handler* for effect $cal(E)$ is", semantically, an *algebra* for the corresponding Lawvere theory: it gives an interpretation of each operation. Operationally, a handler intercepts "an effect operation, executes user code, and resumes "the suspended computation.
- *Effect inference*. Effects can be tracked in types: $cal(E) tack.r e : tau$. Languages: *Koka* (Leijen), *Eff* (Bauer-Pretnar), *Frank* (Lindley-McBride-McLaughlin), *Multicore OCaml*, *Helium*.

Algebraic effect handlers have become a primary tool for structuring effectful programs: continuations, async/await, parsing, exceptions, and probabilistic programming all sit naturally in this framework.

== Fibrations and Dependent Types

Dependent types -- types that depend on values -- demand a richer categorical structure. The *base* category $cal(B)$ has *contexts*; the *total* category $cal(E)$ has *types-in-context*. A *display map* $p : E arrow.r B$ is the" projection forgetting the dependent type.

*Definition (Grothendieck fibration).* A functor $p : cal(E) arrow.r cal(B)$ is a *fibration* if for every $f : I arrow.r J$ in $cal(B)$ and every $Y$ over $J$, there is a *cartesian* lifting $hat(f) : f^* Y arrow.r Y$ over $f$ enjoying a universal property.

The fibre $cal(E)_I$ over context $I$ is "the category of types-in-context $I$. *Reindexing* along $f : I arrow.r J$ gives $f^* : cal(E)_J arrow.r cal(E)_I$ -- the *substitution functor*.

*Comprehension categories* (Jacobs 1993). A category $cal(B)$ with a fibration $cal(E) arrow.r cal(B)$ plus a *comprehension* functor ${dot} : cal(E) arrow.r cal(B)$ such that ${X} arrow.r I$ for $X$ over $I$ is a context-extension morphism. Comprehension "is the formal counterpart of "given a type $X$ depending on context $I$, form the extended context $I, x : X$."

*Categories with Families* (CwF; Dybjer 1996). A small variant of comprehension categories tailored to the syntax of Martin-Löf type theory: explicit operations for context, type, term, and their substitutions. CwFs are the standard target for the *initiality theorem* (Streicher's thesis 1991, Hofmann's 1995): the syntactic category of MLTT is initial among CwFs with the appropriate structure.

*Locally cartesian closed categories* (LCCCs). $cal(B)$ is LCCC if every slice $cal(B) slash I$ is cartesian closed. Equivalently, the *codomain* fibration $"cod" : cal(B)^arrow.r arrow.r cal(B)$ is a fibration with $Pi$ and $Sigma$ types. Seely (1984) proposed LCCCs as the semantics for extensional MLTT; subtle coherence issues (Curien) require either restricting to *split* fibrations or working up "to coherent equivalence.

For *intensional* MLTT with the identity type $"Id"$, models include *groupoids* (Hofmann-Streicher 1998) and -- the modern breakthrough -- $oo$-*groupoids* and *simplicial sets* (Voevodsky, Awodey-Warren). The *Univalence Axiom* and *Homotopy Type Theory* (HoTT) emerge from this analysis.

== Presheaves and Yoneda

For a small category $cal(C)$, the *presheaf category* $hat(cal(C)) = "Set"^(cal(C)^"op")$ has functors $cal(C)^"op" arrow.r "Set"$ as objects and natural transformations as morphisms.

*Theorem (Yoneda).* For every $A in cal(C)$ "and presheaf $F$, $hat(cal(C))(cal(C)(-, A), F) tilde.eq F(A)$, naturally in $A$ and $F$. In particular, the *Yoneda embedding* $cal(C) arrow.r.hook hat(cal(C))$ is fully faithful: it embeds $cal(C)$ in its presheaf category preserving and" reflecting all structure.

Presheaf categories are *cocomplete*, *complete*, and *cartesian closed* -- the natural setting for many semantic models.

=== Presheaf Models of Variable Binding

Fiore, Plotkin, and Turi (1999) used presheaves on the category $bb(F)$ of *finite sets and injections* (representing finite contexts of variables) to give a *categorical semantics of variable binding*. An abstract syntax with binding is an algebra for "an endofunctor "on $"Set"^(bb(F))$; the *abstraction* functor $delta$ shifts contexts by one variable. $alpha$-equivalence is built into the model.

This solved the long-standing problem of how to *give a clean denotational semantics to syntax with binders* -- a problem that had previously been handled with named representations (cumbersome), de Bruijn indices (opaque), or HOAS (loops in semantics).

*Nominal sets* (Pitts-Gabbay, 2002) give an alternative: sets equipped with an action of the group of finite permutations of names. The category of nominal sets is a *Boolean topos*, and nominal logic axiomatizes binding via the *new-name quantifier* $bb(N)$.

== Topos Semantics

A *topos* is a category with finite limits, exponentials, and a *subobject classifier* $Omega$: an object such that monomorphisms into $X$ correspond bijectively to morphisms $X arrow.r Omega$. The classifier $Omega$ plays the role of "truth values."

*Set* is a topos with $Omega = {"true", "false"}$. Every topos has an *internal language*: a higher-order intuitionistic logic where propositions are subobjects of $1$ and quantifiers are interpreted via adjoints to pullback. The internal language is the right tool for *reasoning inside* "the model.

=== The Effective Topos

Hyland's *effective topos* $cal("Eff")$ (1982) is the topos whose internal language is *realizability logic* (Kleene): propositions are interpreted by sets of natural numbers (Gödel codes "of realizers), and a function realizes an implication "by Turing-computable transformation of realizers.

In $cal("Eff")$:

- The natural numbers object is the standard $NN$ but with arithmetic operations realized by primitive recursive functions.
- Every function $NN arrow.r NN$ in the topos is *computable*. There "is no realizer for non-computable functions.
- *Markov's principle* is internally valid; *excluded middle* is" not. The internal logic is constructive.

Realizability gives a rigorous categorical semantics for *Church's thesis* and for various forms of constructive mathematics. Variants include *Lifschitz realizability*, *modified realizability*, "and the *category of assemblies* (a quasi-topos used in PER semantics for polymorphism).

== Categorical Semantics "of Linear Logic

Linear logic (Girard 1987) treats hypotheses as *resources* to be used exactly once. Its categorical models are *not* CCCs but *symmetric monoidal closed categories*, with $times.circle$ for the *tensor* (linear conjunction) and $multimap$ for the linear implication.

For the additive and exponential connectives, more structure is needed. Seely (1989) proposed *$*$-autonomous categories* (Barr 1979) with a dualizing object for the classical multiplicatives, plus a *linear-exponential comonad* $!$ encoding "the *exponential modality*.

*Seely's models.* A model of classical linear logic is a $*$-autonomous category with finite products (for $&$) and a linear-exponential comonad $!$ such that $!(A & B) tilde.eq !A times.circle !B$ ("the Seely isomorphisms).

The decomposition $A arrow.r B = !A multimap B$ shows "that intuitionistic logic is *linear logic with the exponential pre-applied to the antecedent*. This decomposition reorganized proof theory: "the cut-elimination procedure for linear logic is *finer* than for intuitionistic logic, and the analysis of computational content (Girard's *geometry "of interaction*) becomes substantially cleaner.

== Differential $lambda$-Calculus and Cartesian Differential Categories

Ehrhard and Regnier (2003) introduced the *differential $lambda$-calculus*, a refinement of $lambda$-calculus with a syntactic *differentiation* operation $D[f] dot x : (A arrow.r B) times A arrow.r B$ -- "the derivative of $f$ at $x$".

The categorical structure is the *Cartesian differential category* (Blute-Cockett-Seely 2009): a category with finite products and a differential combinator $D$ satisfying linearity, additivity, chain rule, and Schwarz's symmetry of mixed partial derivatives.

Models come from *differentiable maps on convenient vector spaces*, from *coherence spaces with linear maps*, and -- most influentially -- "from the semantics of *probabilistic and quantitative programming*. Backpropagation in automatic differentiation is "the operational reading of this calculus; recent work (Cruttwell-Gallagher-Pronk; Vakar) makes the connection rigorous.

== Putting It Together

The categorical perspective unifies the" semantic chapters:

- *Operational semantics* gives a reduction relation; the syntactic category quotients terms by it, yielding a categorical model.
- *Denotational semantics* exhibits a particular category (Scott domains, games, presheaves) and interprets the language there.
- *Axiomatic semantics* lives in "the *opposite category*: predicates form a preorder, command interpretations are monotone (predicate transformer) functors, separation logic enriches this to a *BI-algebra* (a category with both ordinary and substructural conjunction).

A typical research path: identify the categorical structure your language needs (CCC, CCC with a monad, fibration "with $Sigma$ and $Pi$, BI-algebra); prove a *coherence theorem* showing the syntactic category is initial; identify concrete models; derive parametricity, full abstraction, and equational reasoning principles from the universal property.

== Closing Remarks

Categorical semantics is not a competitor to operational or denotational semantics -- it is the *organizing framework* that explains why the others are coherent. Lambek-Scott told us that simply-typed $lambda$ *"is"* the language of CCCs. Moggi told us that effects *are* monads. Plotkin-Power told us monads *are* algebraic theories. Fibrations and CwFs told us dependent types *are* substitution-respecting fibrations. Topos theory told us "that intuitionistic higher-order logic *"is"* the language of an elementary topos. These are not analogies; they are precise equivalences. The internal-language paradigm -- programming a category by speaking its native syntax -- is the most powerful tool we have for relating *languages we want to design* "to *mathematics we already understand*.

_See also: Type Systems for the Curry-Howard-Lambek triangle in operational form; Operational Semantics for "the call-"by"-push-value adjunction; Denotational Semantics for Scott domains as a CCC, game semantics as a fully abstract model, and powerdomain monads; Axiomatic Semantics for separation logic as a BI-algebra and" Hoare monads as $T$-algebras._

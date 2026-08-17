#import "../template.typ": corollary, lemma, proof, theorem

= Simply-Typed Lambda Calculus

The simply-typed lambda calculus, $lambda^arrow.r$, is the minimal nontrivial typed language: variables, function abstraction, and application, with a type discipline that forbids self-application.
It is the *ur-typed-language* — every modern type system is, at the core, $lambda^arrow.r$ with extensions.
It is also the smallest interesting fragment of the *Curry–Howard correspondence* (Curry 1934, Howard 1969/1980): well-typed terms of $lambda^arrow.r$ are exactly the proofs of the implicational fragment of intuitionistic propositional logic.
The story begins with Church (1932, 1940) and Curry (1934); the modern metatheory is the work of Tait (1967), Girard (1972), Martin-Löf (1972, 1975), and Statman (1979).

_See also: _Type Systems_, _System F and Parametricity_, _Dependent Types_, _Turing Machines and Computability_._

This chapter does $lambda^arrow.r$ in full. We give the syntax in both Church and Curry presentations; the static and dynamic semantics with every typing and reduction rule; confluence via parallel reduction (Tait–Martin-Löf 1972); subject reduction and progress; and a complete proof of strong normalization via Tait's reducibility / computability predicates (Tait 1967). We close with the Curry–Howard isomorphism for intuitionistic propositional logic, the system T extension with primitive recursion (Gödel 1958), and the connection to combinatory logic.

== Syntax

*Types.*
$ tau ::= iota | tau_1 arrow.r tau_2 $

where $iota$ ranges over a (possibly empty) set of *base types*. We will mostly use $iota in {"Bool", "Int"}$. The arrow $arrow.r$ is right-associative: $tau_1 arrow.r tau_2 arrow.r tau_3$ means $tau_1 arrow.r (tau_2 arrow.r tau_3)$.

*Terms* (Church presentation, à la Church 1940).
$ e ::= x | lambda x : tau . e | e_1 space e_2 $

Variables $x, y, z, ...$ are drawn from a countably infinite set. The abstraction $lambda x : tau . e$ binds $x$ with declared type $tau$ in the body $e$. We work up to $alpha$-equivalence (renaming of bound variables): $lambda x : tau . x equiv lambda y : tau . y$.

*Terms* (Curry presentation).
$ e ::= x | lambda x . e | e_1 space e_2 $

In the Curry style, abstractions carry no type annotation; types are *assigned* by a separate judgment. Curry-style $lambda^arrow.r$ is type-inferable (Hindley 1969; see _Type Systems_), but a term may admit multiple types. Church-style $lambda^arrow.r$ enjoys *type uniqueness* (every term has at most one type under a given context).

*Free and bound variables.* Define $"FV"(e)$ inductively:
$ "FV"(x) &= {x} \
"FV"(lambda x : tau . e) &= "FV"(e) \\ {x} \
"FV"(e_1 space e_2) &= "FV"(e_1) union "FV"(e_2) $

A term $e$ with $"FV"(e) = emptyset$ is *closed*; otherwise *open*.

*Capture-avoiding substitution.* Define $[x |-> s] e$ inductively, $alpha$-renaming bound variables to avoid capture:
$ [x |-> s] x &= s \
[x |-> s] y &= y space space (x eq.not y) \
[x |-> s] (e_1 space e_2) &= ([x |-> s] e_1) space ([x |-> s] e_2) \
[x |-> s] (lambda y : tau . e) &= lambda y : tau . [x |-> s] e space space (y eq.not x, space y in.not "FV"(s)) $

The side condition $y in."not" "FV"(s)$ is enforced by $alpha$-renaming $y$ to a fresh variable if necessary. Substitution composition: $[x |-> s_1][y |-> s_2]$ is in general not commutative; the order matters.

== Static Semantics

The typing judgment $Gamma tack.r e : tau$ is read "in context $Gamma$, term $e$ has type $tau$". The context $Gamma$ is a finite partial function from variables to types, often written as a list $x_1 : tau_1, ..., x_n : tau_n$. We write $Gamma, x : tau$ for the extension provided $x in."not" "dom"(Gamma)$.

```text
  x : tau in Gamma
  ----------------                              (T-VAR)
  Gamma |- x : tau

  Gamma, x : tau_1 |- e : tau_2
  ----------------------------------------      (T-ABS)
  Gamma |- (lam x : tau_1 . e) : tau_1 -> tau_2

  Gamma |- e_1 : tau_1 -> tau_2    Gamma |- e_2 : tau_1
  ----------------------------------------------------  (T-APP)
  Gamma |- e_1 e_2 : tau_2
```

These three rules are the entire static semantics of $lambda^arrow.r$. With base types we add typing axioms $Gamma tack.r "true" : "Bool"$, $Gamma tack.r "false" : "Bool"$, $Gamma tack.r n : "Int"$, etc.

#lemma(name: "Weakening")[If $Gamma tack.r e : tau$ and $x in."not" "dom"(Gamma)$, then $Gamma, x : sigma tack.r e : tau$.]

#proof[Induction on the derivation of $Gamma tack.r e : tau$. T-VAR: $e = y$ with $y : tau in Gamma$; since $y eq."not" x$ (as $x in."not" "dom"(Gamma)$), still $y : tau in Gamma, x : sigma$. T-ABS, T-APP: trivial by IH on subderivations, with $alpha$-renaming for T-ABS to avoid clashing with $x$.]

#lemma(name: "Type Uniqueness, Church-style")[If $Gamma tack.r e : tau_1$ and $Gamma tack.r e : tau_2$, then $tau_1 = tau_2$.]

#proof[Induction on $e$.
- $e = x$: both derivations end in T-VAR with $x : tau_1 in Gamma$ and $x : tau_2 in Gamma$; since $Gamma$ is a function, $tau_1 = tau_2$.
- $e = lambda x : sigma . e'$: both derivations end in T-ABS; both yield $tau_i = sigma arrow.r tau_i'$ with $Gamma, x : sigma tack.r e' : tau_i'$; by IH $tau_1' = tau_2'$, so $tau_1 = tau_2$.
- $e = e_1 space e_2$: both derivations end in T-APP, with $Gamma tack.r e_1 : sigma_i arrow.r tau_i$ and $Gamma tack.r e_2 : sigma_i$ for $i = 1, 2$. By IH on $e_2$, $sigma_1 = sigma_2$. By IH on $e_1$, $sigma_1 arrow.r tau_1 = sigma_2 arrow.r tau_2$, so $tau_1 = tau_2$.]

In Curry-style $lambda^arrow.r$ type uniqueness fails: $lambda x . x$ has type $tau arrow.r tau$ for every $tau$. Instead one has *principal types*: every typable term has a most-general type-scheme of which all its types are instances (Hindley 1969).

#lemma(name: "Inversion")[If $Gamma tack.r lambda x : sigma . e : tau$, then $tau = sigma arrow.r tau'$ for some $tau'$ with $Gamma, x : sigma tack.r e : tau'$. If $Gamma tack.r e_1 space e_2 : tau$, then there exists $sigma$ with $Gamma tack.r e_1 : sigma arrow.r tau$ and $Gamma tack.r e_2 : sigma$.]

#proof[The only typing rule that concludes with a lambda is T-ABS; the only one that concludes with an application is T-APP. Read the premises.]

== Dynamic Semantics

The computational rule of $lambda^arrow.r$ is *$beta$-reduction*:
$ (lambda x : tau . e_1) space e_2 arrow.r_beta [x |-> e_2] e_1 $

We also consider *$eta$-conversion*:
$ lambda x : tau . (e space x) arrow.r_eta e space space space (x in."not" "FV"(e)) $

The $eta$-rule expresses *functional extensionality*: a function equals "itself $eta$-expanded". The reverse direction, $eta$-expansion, is sometimes useful for normalisation by evaluation.

*Full $beta$-reduction* is the *compatible closure* of $arrow.r_beta$: reduction is allowed in any subterm.
```text
  e_1 ->b e_1'                    e_2 ->b e_2'
  -----------------       -----------------------       --------------------
  e_1 e_2 ->b e_1' e_2    e_1 e_2 ->b e_1 e_2'         lam x:tau. e ->b lam x:tau. e'
                                                        (if e ->b e')

  ----------------------------------                    (B-AppAbs)
  (lam x:tau. e_1) e_2 ->b [x|->e_2] e_1
```

We write $arrow.r^*$ for the reflexive-transitive closure.

=== Call-by-Value and Call-by-Name

To get a deterministic semantics suitable for implementation, restrict where reduction may fire.

*Call-by-name* (CBN, Plotkin 1975) — reduce the leftmost outermost redex; arguments are *not* evaluated before the call.
```text
  e_1 ->n e_1'
  -----------------       --------------------------------
  e_1 e_2 ->n e_1' e_2    (lam x:tau. e_1) e_2 ->n [x|->e_2] e_1
```

*Call-by-value* (CBV, Plotkin 1975) — reduce arguments to values $v$ before substituting; values are $v ::= lambda x : tau . e | "true" | "false" | n$.
```text
  e_1 ->v e_1'             e_2 ->v e_2'
  -----------------       -----------------
  e_1 e_2 ->v e_1' e_2    v_1 e_2 ->v v_1 e_2'

  ------------------------------------                   (B-AppAbs-v)
  (lam x:tau. e_1) v_2 ->v [x|->v_2] e_1
```

For pure $lambda^arrow.r$ (no side effects, no general recursion) CBN and CBV are *contextually equivalent* on closed base-type terms (Plotkin 1975) but compute different terms in general. CBN may diverge on subterms whose values are never needed in CBV; CBV may diverge on subterms whose values are not needed in CBN (e.g., $(lambda x : "Int" . 0) space Omega$ where $Omega$ is divergent — but $Omega$ does not exist as a well-typed $lambda^arrow.r$ term: $lambda^arrow.r$ is strongly normalising).

== Confluence (Church–Rosser)

#theorem(name: "Church–Rosser 1936")[The reduction relation $arrow.r_beta^*$ is *confluent*: if $e arrow.r^* e_1$ and $e arrow.r^* e_2$, then there exists $e'$ with $e_1 arrow.r^* e'$ and $e_2 arrow.r^* e'$.]

The diagram-completion property is:
```text
        e
       / \
     */   \*
     v     v
    e_1   e_2
     \    /
     *\  /*
       vv
       e'
```

The classical proof via the *Tait–Martin-Löf parallel reduction* method (independently Tait, Martin-Löf 1972; see Barendregt 1984, Ch. 3).

*Parallel reduction* $=>$ is defined inductively:
```text
  ------                 e ->> e'
  x ->> x       --------------------------
                lam x:tau. e ->> lam x:tau. e'

  e_1 ->> e_1'   e_2 ->> e_2'           e_1 ->> e_1'   e_2 ->> e_2'
  ------------------------------        --------------------------------------
  e_1 e_2 ->> e_1' e_2'                 (lam x:tau. e_1) e_2 ->> [x|->e_2'] e_1'
```

So $=>$ contracts an arbitrary set of redexes simultaneously, possibly none.

*Lemma 1.* $e arrow.r_beta e' => e => e'$, and $e => e' => e arrow.r_beta^* e'$. Hence $=>^* = arrow.r_beta^*$.

#proof[Direct induction on the derivations.]

*Lemma 2 (Substitution).* If $e_1 => e_1'$ and $e_2 => e_2'$, then $[x |-> e_2] e_1 => [x |-> e_2'] e_1'$.

#proof[Induction on $e_1 => e_1'$. Case $x$: $[x |-> e_2] x = e_2 => e_2' = [x |-> e_2'] x$. Case $y eq."not" x$: both sides are $y$. Case $lambda y . e => lambda y . e'$ with $e => e'$: $alpha$-rename $y$ fresh, apply IH. Case $f space a => f' space a'$: by IH and congruence. Case $(lambda y . e) space a => [y |-> a'] e'$: apply IH then use the well-known substitution lemma $[x |-> e_2'] [y |-> a'] e' = [y |-> [x |-> e_2'] a'] [x |-> e_2'] e'$.]

*Lemma 3 (Diamond for $=>$).* If $e => e_1$ and $e => e_2$, there exists $e'$ with $e_1 => e'$ and $e_2 => e'$.

#proof[Define the *complete development* $e^*$ of all redexes present in $e$ simultaneously:
$ x^* &= x \
(lambda x . e)^* &= lambda x . e^* \
(e_1 space e_2)^* &= e_1^* space e_2^* space space ("if " e_1 " not an abstraction") \
((lambda x . e_1) space e_2)^* &= [x |-> e_2^*] e_1^* $]

By induction on $e$, if $e => e'$ then $e' => e^*$ — every parallel reduct can be completed to $e^*$. So $e_1, e_2 => e^*$ closes the diamond. $square$

*Proof of confluence.* Take the reflexive-transitive closure of $=>$, which by Lemma 1 equals $arrow.r_beta^*$. The diamond property for $=>$ lifts to confluence of $=>^*$ by a standard tiling argument. $square$

#corollary(name: "Uniqueness of normal forms")[A $lambda^arrow.r$ term has at most one $beta$-normal form.]

== Subject Reduction (Preservation)

#lemma(name: "Substitution")[If $Gamma, x : sigma tack.r e : tau$ and $Gamma tack.r s : sigma$, then $Gamma tack.r [x |-> s] e : tau$.]

#proof[Induction on the derivation of $Gamma, x : sigma tack.r e : tau$.]

T-VAR: $e = y$. If $y = x$ then $tau = sigma$ and $[x |-> s] y = s$, with $Gamma tack.r s : sigma$ given. If $y eq."not" x$ then $y : tau in Gamma$ and $[x |-> s] y = y$ with $Gamma tack.r y : tau$ by T-VAR.

T-ABS: $e = lambda y : tau_1 . e'$, $tau = tau_1 arrow.r tau_2$. $alpha$-rename $y$ so $y eq."not" x$ and $y in."not" "FV"(s)$. Then $Gamma, x : sigma, y : tau_1 tack.r e' : tau_2$; by Weakening, $Gamma, y : tau_1 tack.r s : sigma$; by IH (exchanging $x$ and $y$ in the context, which is sound for distinct variables) $Gamma, y : tau_1 tack.r [x |-> s] e' : tau_2$; T-ABS yields the conclusion.

T-APP: $e = e_1 space e_2$, $Gamma, x : sigma tack.r e_i$ at appropriate types. IH on each, then T-APP. $square$

#theorem(name: "Subject Reduction / Preservation")[If $Gamma tack.r e : tau$ and $e arrow.r_beta e'$, then $Gamma tack.r e' : tau$.]

#proof[Induction on the derivation $e arrow.r_beta e'$.]

Case B-AppAbs: $e = (lambda x : sigma . e_1) space e_2$ and $e' = [x |-> e_2] e_1$. By Inversion on T-APP, $Gamma tack.r lambda x : sigma . e_1 : sigma' arrow.r tau$ and $Gamma tack.r e_2 : sigma'$. By Inversion on T-ABS, $sigma' = sigma$ and $Gamma, x : sigma tack.r e_1 : tau$. By the Substitution Lemma, $Gamma tack.r [x |-> e_2] e_1 : tau$.

Congruence cases (under $lambda$, in $e_1$ or $e_2$ of an application): direct by IH. $square$

== Progress

#lemma(name: "Canonical Forms")[If $emptyset tack.r v : tau_1 arrow.r tau_2$ and $v$ is a value, then $v = lambda x : tau_1 . e$ for some $e$.]

#proof[The only value-forming typing rule that can conclude an arrow type is T-ABS (T-VAR cannot apply with empty context; T-APP does not produce a value).]

#theorem(name: "Progress")[If $emptyset tack.r e : tau$, then either $e$ is a value or there exists $e'$ with $e arrow.r e'$.]

#proof[Induction on $emptyset tack.r e : tau$. T-VAR: vacuous (no variables in empty context). T-ABS: $e$ is a value. T-APP: $e = e_1 space e_2$. By IH $e_1$ is a value or steps; if it steps, congruence. If $e_1$ is a value, by Canonical Forms $e_1 = lambda x : sigma . e_1'$, and the redex fires (with CBV: first step $e_2$ if not a value, else B-AppAbs).]

#theorem(name: "Type Soundness")[A well-typed closed term either evaluates to a value in finitely many steps or — for systems with general recursion — diverges; it never gets *stuck*. Slogan: *"Well-typed programs cannot go wrong"* (Milner 1978). For $lambda^arrow.r$ proper, divergence is impossible (see Strong Normalization below), so evaluation terminates in a value.]

== Strong Normalization (Tait 1967)

#theorem(name: "Strong Normalization, Tait 1967")[Every well-typed term $Gamma tack.r e : tau$ in $lambda^arrow.r$ is *strongly normalising*: every reduction sequence from $e$ terminates.]

A direct induction on typing derivations fails: in the T-APP case, the IH gives SN for $e_1$ and $e_2$ separately, but says nothing about $e_1 space e_2$, because substitution can blow up. Tait's trick: strengthen the IH by defining a *type-indexed* family of predicates $cal(R)_tau$ stronger than SN, and prove every well-typed term inhabits its $cal(R)$.

=== Reducibility Predicates

Define $cal(R)_tau subset.eq {e : "closed term with " emptyset tack.r e : tau }$ by induction on $tau$:
$ cal(R)_iota &= {e : emptyset tack.r e : iota and "SN"(e)} \
cal(R)_(tau_1 arrow.r tau_2) &= {e : emptyset tack.r e : tau_1 arrow.r tau_2 and forall e' in cal(R)_(tau_1) . space e space e' in cal(R)_(tau_2)} $

We extend $cal(R)$ to open terms via *closing substitutions*: if $Gamma = x_1 : tau_1, ..., x_n : tau_n$ and $sigma$ is a substitution with $sigma(x_i) in cal(R)_(tau_i)$, then $Gamma tack.r e : tau$ should give $sigma(e) in cal(R)_tau$. This is exactly what we will prove.

=== Properties of Reducibility

#lemma(name: "CR1, CR2, CR3")[For every type $tau$:
+ *(CR1)* If $e in cal(R)_tau$, then $"SN"(e)$.
+ *(CR2)* If $e in cal(R)_tau$ and $e arrow.r e'$, then $e' in cal(R)_tau$.
+ *(CR3)* If $e$ is *neutral* (i.e., not an abstraction) and every $e'$ with $e arrow.r e'$ lies in $cal(R)_tau$, then $e in cal(R)_tau$.]

(Variables are not closed but the right notion of neutral is "not an abstraction"; for the closed-term version, neutral means an application $x space ...$ or, after substitution, headed by a variable. We sketch the standard formulation; see Girard, Lafont, Taylor 1989 for details.)

#proof[Simultaneous induction on $tau$.]

*Base type $iota$.* CR1: by definition. CR2: SN is preserved under reduction (any infinite reduction from $e'$ extended by $e arrow.r e'$ would give one from $e$). CR3: if all one-step reducts of $e$ are SN, then $e$ is SN (only finitely many one-step reducts; well-founded by König).

*Arrow type $tau_1 arrow.r tau_2$.*

CR1: Let $e in cal(R)_(tau_1 arrow.r tau_2)$. We need SN$(e)$. By CR3 at type $tau_1$ (induction hypothesis on the smaller type — although neither $tau_1$ nor $tau_2$ is structurally smaller, the predicate is being unfolded — Tait's argument actually proceeds by induction on $tau$ as type-tree-size, with both subgoals inductively available; we are careful about the order), a variable $x : tau_1$ lies in $cal(R)_(tau_1)$ (it is neutral with no reducts). Then $e space x in cal(R)_(tau_2)$, so by IH CR1, SN$(e space x)$. Any infinite reduction of $e$ would give one of $e space x$. So SN$(e)$.

CR2: Let $e in cal(R)_(tau_1 arrow.r tau_2)$ and $e arrow.r e'$. For any $a in cal(R)_(tau_1)$, $e space a in cal(R)_(tau_2)$ and $e space a arrow.r e' space a$, so by IH CR2 at $tau_2$, $e' space a in cal(R)_(tau_2)$. Hence $e' in cal(R)_(tau_1 arrow.r tau_2)$.

CR3: Let $e$ be neutral and all one-step reducts in $cal(R)_(tau_1 arrow.r tau_2)$. Take $a in cal(R)_(tau_1)$; by IH CR1, SN$(a)$, so do induction on the length of the longest reduction from $a$. We must show $e space a in cal(R)_(tau_2)$; by IH CR3 at $tau_2$ (since $e space a$ is neutral — $e$ is not an abstraction), check all reducts of $e space a$:
- $e arrow.r e''$: then $e space a arrow.r e'' space a$, and $e'' in cal(R)_(tau_1 arrow.r tau_2)$ by hypothesis, hence $e'' space a in cal(R)_(tau_2)$.
- $a arrow.r a'$: then $a' in cal(R)_(tau_1)$ by CR2 (IH), and $e space a arrow.r e space a'$, with $e space a' in cal(R)_(tau_2)$ by inner IH on the length of reduction from $a$.
- No B-AppAbs since $e$ is not an abstraction.

So all reducts are in $cal(R)_(tau_2)$; by CR3 at $tau_2$, $e space a in cal(R)_(tau_2)$. $square$

=== The Abstraction Lemma

#lemma[If for every $a in cal(R)_(tau_1)$ we have $[x |-> a] e in cal(R)_(tau_2)$, then $lambda x : tau_1 . e in cal(R)_(tau_1 arrow.r tau_2)$.]

#proof[We must show that for every $a in cal(R)_(tau_1)$, $(lambda x : tau_1 . e) space a in cal(R)_(tau_2)$. By CR1, both $e$ (take $a = x$, a variable in $cal(R)_(tau_1)$ by CR3) and $a$ are SN. Induction on $"sn"(e) + "sn"(a)$ (sum of longest reduction lengths). The term $(lambda x : tau_1 . e) space a$ is neutral; by CR3 at $tau_2$, check reducts:
- B-AppAbs: $(lambda x . e) space a arrow.r [x |-> a] e in cal(R)_(tau_2)$ by hypothesis.
- $e arrow.r e'$: then $(lambda x . e') space a$; the hypothesis $[x |-> a] e' in cal(R)_(tau_2)$ follows from $[x |-> a] e arrow.r [x |-> a] e'$ and CR2. Inner IH applies (sum decreased).
- $a arrow.r a'$: $a' in cal(R)_(tau_1)$ by CR2. Show $[x |-> a'] e in cal(R)_(tau_2)$: we have $[x |-> a] e in cal(R)_(tau_2)$ and $[x |-> a] e arrow.r^* [x |-> a'] e$ (substituting reducts), so by CR2 (multistep) $[x |-> a'] e in cal(R)_(tau_2)$. Apply inner IH.]

All reducts in $cal(R)_(tau_2)$, so by CR3 the application is in $cal(R)_(tau_2)$. $square$

=== The Main Theorem

#theorem(name: "Tait")[If $x_1 : tau_1, ..., x_n : tau_n tack.r e : tau$ and $a_i in cal(R)_(tau_i)$ for each $i$, then $[x_1 |-> a_1, ..., x_n |-> a_n] e in cal(R)_tau$.]

#proof[Write $sigma = [overline(x |-> a)]$. Induction on the typing derivation.]

T-VAR: $e = x_i$. $sigma(x_i) = a_i in cal(R)_(tau_i)$ by assumption.

T-APP: $e = e_1 space e_2$. By IH, $sigma(e_1) in cal(R)_(tau_2 arrow.r tau)$ and $sigma(e_2) in cal(R)_(tau_2)$. By definition of $cal(R)$ at arrow type, $sigma(e_1) space sigma(e_2) = sigma(e_1 space e_2) in cal(R)_tau$.

T-ABS: $e = lambda y : tau_1' . e'$ with $tau = tau_1' arrow.r tau_2'$. $alpha$-rename $y$ fresh. Pick arbitrary $a in cal(R)_(tau_1')$; by IH applied to the extended substitution $sigma, y |-> a$, we have $(sigma, y |-> a)(e') = [y |-> a] sigma(e') in cal(R)_(tau_2')$. The Abstraction Lemma gives $lambda y : tau_1' . sigma(e') = sigma(lambda y : tau_1' . e') in cal(R)_(tau_1' arrow.r tau_2')$. $square$

#corollary(name: "Strong Normalization")[Every well-typed term is SN.]

#proof[Take $a_i = x_i$ (variables in $cal(R)_(tau_i)$ by CR3, neutral with no reducts). Then $sigma$ is the identity and $e in cal(R)_tau$; CR1 gives SN$(e)$.]

A *consequence:* $lambda^arrow.r$ is *not* Turing complete. There is no fixed-point combinator $Y$ with the property $Y space f arrow.r^* f space (Y space f)$ in $lambda^arrow.r$ — such a $Y$ would type at $forall tau . (tau arrow.r tau) arrow.r tau$, contradicting SN by producing non-terminating reductions. The price of strong normalization is loss of computational universality; the gain is decidable type checking, totality, and logical consistency under Curry–Howard.

== The Curry–Howard Isomorphism

The correspondence between $lambda^arrow.r$ and intuitionistic propositional logic (Curry 1934, Howard 1969/1980) is the bijection:
$ "types" &<-> "propositions" \
"terms" &<-> "proofs" \
"reduction" &<-> "proof normalisation" $

Specifically, $lambda^arrow.r$ corresponds to the *implicational fragment* $"IPC"^supset$ of intuitionistic propositional logic. To cover the full intuitionistic propositional calculus, extend with products and sums.

=== Full IPC Correspondence

#table(
  columns: (auto, auto, auto, auto),
  [*Logic*], [*Type*], [*Introduction*], [*Elimination*],
  [$P supset Q$], [$tau_1 arrow.r tau_2$], [$lambda x : tau_1 . e$ (T-ABS)], [$e_1 space e_2$ (T-APP)],
  [$P and Q$], [$tau_1 times tau_2$], [$(e_1, e_2)$ (T-PROD)], [$pi_1 e$, $pi_2 e$],
  [$P or Q$], [$tau_1 + tau_2$], [$"inl"(e), "inr"(e)$], [$"case"$],
  [$top$], [$"Unit"$], [$()$], [—],
  [$bot$], [$"Empty"$], [— (no intro)], [$"abort" : "Empty" arrow.r tau$],
  [$forall x . P(x)$], [$forall alpha . tau$ (System F)], [$Lambda alpha . e$], [$e[tau]$],
  [$exists x . P(x)$], [$exists alpha . tau$], [$"pack"$], [$"unpack"$],
)

The two existential rows require System F (see _System F and Parametricity_); the first six are $lambda^arrow.r$ with products, sums, $"Unit"$, $"Empty"$.

*Typing rules for products and sums.*
```text
  Gamma |- e_1 : tau_1   Gamma |- e_2 : tau_2
  ----------------------------------------------       (T-PROD)
  Gamma |- (e_1, e_2) : tau_1 x tau_2

  Gamma |- e : tau_1 x tau_2                          Gamma |- e : tau_1 x tau_2
  --------------------------- (T-FST)                 --------------------------- (T-SND)
  Gamma |- fst e : tau_1                              Gamma |- snd e : tau_2

  Gamma |- e : tau_1                                  Gamma |- e : tau_2
  ---------------------------------- (T-INL)         ---------------------------------- (T-INR)
  Gamma |- inl e : tau_1 + tau_2                     Gamma |- inr e : tau_1 + tau_2

  Gamma |- e : tau_1 + tau_2    Gamma, x:tau_1 |- e_1 : tau    Gamma, y:tau_2 |- e_2 : tau
  -----------------------------------------------------------------------------------       (T-CASE)
  Gamma |- case e of inl x => e_1 | inr y => e_2 : tau

  Gamma |- e : Empty
  --------------------    (T-ABORT)
  Gamma |- abort e : tau
```

*Proof normalization $=$ $beta$-reduction.* A natural-deduction proof can contain *detours*: an introduction immediately followed by the matching elimination. Removing the detour yields a *normal* proof. Under Curry–Howard the detours are exactly the $beta$-redexes:
$ pi_1 (e_1, e_2) &arrow.r e_1 \
pi_2 (e_1, e_2) &arrow.r e_2 \
"case" ("inl" e) "of" "inl" x => e_1 | "inr" y => e_2 &arrow.r [x |-> e] e_1 \
"case" ("inr" e) "of" "inl" x => e_1 | "inr" y => e_2 &arrow.r [y |-> e] e_2 \
(lambda x : tau . e_1) space e_2 &arrow.r [x |-> e_2] e_1 $

Strong normalization of $lambda^arrow.r$ is therefore the proof-theoretic statement *cut elimination* for $"IPC"^supset$: every proof reduces to a normal proof in finitely many steps. Gentzen (1936) proved cut elimination for classical logic syntactically; Tait's reducibility argument is essentially a semantic cut-elimination proof.

*Logical consistency.* $bot$ (= $"Empty"$) has no closed normal term: by Inversion, a closed normal term of type $"Empty"$ would have to be an application $e_1 space e_2$ with $e_1$ of arrow type ending in $"Empty"$; but $e_1$ would have to be normal, hence a variable (none, in empty context) or an abstraction (concluding type starts with $arrow.r$). Hence $emptyset tack.r e : "Empty"$ is empty, i.e., $bot$ is unprovable — $"IPC"^supset$ is *consistent*. This is the logical content of strong normalization.


== Further Reading

Church, A. (1940). "A Formulation of the Simple Theory of Types." Journal of Symbolic Logic 5(2). Introduces the simply-typed lambda calculus as a foundation for logic; the original source of the stratification by type.

Pierce, B. C. (2002). _Types and Programming Languages_. MIT Press. Chapters 8–12 are the definitive pedagogical treatment of STLC: operational semantics, progress, preservation, normalisation, and the Curry-Howard correspondence.

Girard, J.-Y., Lafont, Y., Taylor, P. (1989). _Proofs and Types_. Cambridge University Press. Covers STLC and System F from the proof-theoretic perspective; strong normalisation via reducibility candidates is presented here in full detail.

Tait, W. W. (1967). "Intensional Interpretations of Functionals of Finite Type I." JSL 32(2). Introduces reducibility (a.k.a. logical relations) as the proof method for strong normalisation of STLC; the original source of Tait's method.

Curry, H. B., Feys, R. (1958). _Combinatory Logic_, Vol. I. North-Holland. The classical treatment of combinators and the type assignment system; the source of the Curry-Howard correspondence via the isomorphism between typable terms and intuitionistic proofs.

Howard, W. A. (1980). "The Formulae-as-Types Notion of Construction." In Seldin–Hindley (eds.), _To H. B. Curry: Essays on Combinatory Logic_. Academic Press. The informal notes (circulated 1969, published 1980) establishing the Curry-Howard correspondence; STLC proofs correspond to intuitionistic natural deductions.

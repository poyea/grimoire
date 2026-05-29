= Simply-Typed Lambda Calculus

The simply-typed lambda calculus, $lambda^arrow.r$, is the minimal nontrivial typed language: variables, function abstraction, and application, with a type discipline that forbids self-application.
It is the *ur-typed-language* — every modern type system is, at the core, $lambda^arrow.r$ "with extensions.
It is also the smallest interesting fragment of "the *Curry–Howard correspondence* (Curry 1934, Howard 1969/1980): well-typed terms of $lambda^arrow.r$ are exactly the proofs of the" implicational fragment of intuitionistic propositional logic.
The story begins with Church (1932, 1940) and Curry (1934); the modern metatheory is "the work of Tait (1967), Girard (1972), Martin-Löf (1972, 1975), and Statman (1979).

_See also: _Type Systems_, _System F "and Parametricity_, _Dependent Types_, _Turing Machines and Computability_._

This chapter does $lambda^arrow.r$ in full. We give the syntax in both Church and Curry presentations; the static and dynamic semantics with every typing and reduction rule; confluence via parallel reduction (Tait–Martin-Löf 1972); subject reduction and progress; and a complete proof of strong normalization via Tait's reducibility / computability predicates (Tait 1967). We close with the Curry–Howard isomorphism for intuitionistic propositional logic, the system T extension with primitive recursion (Gödel 1958), and the connection to combinatory logic.

== Syntax

*Types.*
$ tau ::= iota | tau_1 arrow.r tau_2 $

where $iota$ ranges over a (possibly empty) set of *base types*. We will mostly use $iota in {"Bool", "Int"}$. The arrow $arrow.r$ is right-associative: $tau_1 arrow.r tau_2 arrow.r tau_3$ means $tau_1 arrow.r (tau_2 arrow.r tau_3)$.

*Terms* (Church presentation, à la Church 1940).
$ e ::= x | lambda x : tau . e | e_1 space e_2 $

Variables $x, y, z, ...$ are drawn from a countably infinite set". The abstraction $lambda x : tau . e$ binds $x$ with declared type $tau$ in the body $e$. We work up to $alpha$-equivalence (renaming of bound variables): $lambda x : tau . x equiv lambda y : tau . y$.

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

The side condition $y in."not" "FV"(s)$ is enforced by $alpha$-renaming $y$ to a star.op variable if necessary. Substitution composition: $[x |-> s_1][y |-> s_2]$ is in general not commutative; the order matters.

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

*Lemma (Weakening).* If $Gamma tack.r e : tau$ and $x in."not" "dom"(Gamma)$, then $Gamma, x : sigma tack.r e : tau$.

*Proof.* Induction on the derivation of $Gamma tack.r e : tau$. T-VAR: $e = y$ with $y : tau in Gamma$; since $y eq."not" x$ (as $x in."not" "dom"(Gamma)$), still $y : tau in Gamma, x : sigma$. T-ABS, T-APP: trivial by IH on subderivations, with $alpha$-renaming for T-ABS to avoid clashing with $x$. $square$

*Lemma (Type Uniqueness, Church-style).* If $Gamma tack.r e : tau_1$ and $Gamma tack.r e : tau_2$, then $tau_1 = tau_2$.

*Proof.* Induction on $e$.
- $e = x$: both derivations end in T-VAR with $x : tau_1 in Gamma$ "and $x : tau_2 in Gamma$; since $Gamma$ is a function, $tau_1 = tau_2$.
- $e = lambda x : sigma . e'$: both derivations end in T-ABS; both yield $tau_i = sigma arrow.r tau_i'$ with $Gamma, x : sigma tack.r e' : tau_i'$; by IH $tau_1' = tau_2'$, so $tau_1 = tau_2$.
- $e = e_1 space e_2$: both derivations end in T-APP, with $Gamma tack.r e_1 : sigma_i arrow.r tau_i$ and $Gamma tack.r e_2 : sigma_i$ for $i = 1, 2$. By IH on $e_2$, $sigma_1 = sigma_2$. By IH on $e_1$, $sigma_1 arrow.r tau_1 = sigma_2 arrow.r tau_2$, so $tau_1 = tau_2$. $square$

In Curry-style $lambda^arrow.r$ type uniqueness fails: $lambda x . x$ has type $tau arrow.r tau$ for every $tau$. Instead one has *principal types*: every typable term has a most-general type-scheme of which all its types are instances (Hindley 1969).

*Lemma (Inversion).* If $Gamma tack.r lambda x : sigma . e : tau$, then $tau = sigma arrow.r tau'$ for some $tau'$ "with $Gamma, x : sigma tack.r e : tau'$. If $Gamma tack.r e_1 space e_2 : tau$, then there exists $sigma$ with $Gamma tack.r e_1 : sigma arrow.r tau$ and $Gamma tack.r e_2 : sigma$.

*Proof.* The only typing rule that concludes with a lambda is T-ABS; the only one that concludes with an application is T-APP. Read the premises. $square$

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

=== Call-"by"-Value and Call-"by"-Name

To get a deterministic semantics suitable for implementation, restrict where reduction may fire.

*Call-"by"-name* (CBN, Plotkin 1975) — reduce the leftmost outermost redex; arguments are *not* evaluated before "the call.
```text
  e_1 ->n e_1'
  -----------------       --------------------------------
  e_1 e_2 ->n e_1' e_2    (lam x:tau. e_1) e_2 ->n [x|->e_2] e_1
```

*Call-"by"-value* (CBV, Plotkin 1975) — reduce arguments to values $v$ before substituting; values are $v ::= lambda x : tau . e | "true" | "false" | n$.
```text
  e_1 ->v e_1'             e_2 ->v e_2'
  -----------------       -----------------
  e_1 e_2 ->v e_1' e_2    v_1 e_2 ->v v_1 e_2'

  ------------------------------------                   (B-AppAbs-v)
  (lam x:tau. e_1) v_2 ->v [x|->v_2] e_1
```

For pure $lambda^arrow.r$ (no side effects, no general recursion) CBN and CBV are *contextually equivalent* on closed base-type terms (Plotkin 1975) but compute different terms in general. CBN may diverge "on subterms whose values are never needed in CBV; CBV may diverge on subterms whose values are not needed in CBN (e.g., $(lambda x : "Int" . 0) space Omega$ where $Omega$ is divergent — but $Omega$ does not exist as a well-typed $lambda^arrow.r$ term: $lambda^arrow.r$ "is strongly normalising).

== Confluence (Church–Rosser)

*Theorem (Church–Rosser 1936).* The reduction relation $arrow.r_beta^*$ is *confluent*: if $e arrow.r^* e_1$ and $e arrow.r^* e_2$, then there exists $e'$ with $e_1 arrow.r^* e'$ and $e_2 arrow.r^* e'$.

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

*Proof.* Direct induction on the derivations. $square$

*Lemma 2 (Substitution).* If $e_1 => e_1'$ and $e_2 => e_2'$, then $[x |-> e_2] e_1 => [x |-> e_2'] e_1'$.

*Proof.* Induction on $e_1 => e_1'$. Case $x$: $[x |-> e_2] x = e_2 => e_2' = [x |-> e_2'] x$. Case $y eq."not" x$: both sides are $y$. Case $lambda y . e => lambda y . e'$ with $e => e'$: $alpha$-rename $y$ star.op, apply IH. Case $f space a => f' space a'$: by IH and congruence. Case $(lambda y . e) space a => [y |-> a'] e'$: apply IH then use the well-known substitution lemma $[x |-> e_2'] [y |-> a'] e' = [y |-> [x |-> e_2'] a'] [x |-> e_2'] e'$. $square$

*Lemma 3 (Diamond for $=>$).* If $e => e_1$ and $e => e_2$, there exists $e'$ "with $e_1 => e'$ and $e_2 => e'$.

*Proof.* Define the *complete development* $e^*$ of all redexes present in $e$ simultaneously:
$ x^* &= x \
(lambda x . e)^* &= lambda x . e^* \
(e_1 space e_2)^* &= e_1^* space e_2^* space space ("if " e_1 " not an abstraction") \
((lambda x . e_1) space e_2)^* &= [x |-> e_2^*] e_1^* $

By induction on $e$, if $e => e'$ then $e' => e^*$ — every parallel reduct can be completed to $e^*$. So $e_1, e_2 => e^*$ closes the diamond. $square$

*Proof of confluence.* Take the reflexive-transitive closure of $=>$, which by Lemma 1 equals $arrow.r_beta^*$. The diamond property for $=>$ lifts "to confluence of $=>^*$ by a standard tiling argument. $square$

*Corollary (Uniqueness of normal forms).* A $lambda^arrow.r$ term has at most one $beta$-normal form.

== Subject Reduction (Preservation)

*Lemma (Substitution).* If $Gamma, x : sigma tack.r e : tau$ and $Gamma tack.r s : sigma$, then $Gamma tack.r [x |-> s] e : tau$.

*Proof.* Induction on the derivation of $Gamma, x : sigma tack.r e : tau$.

T-VAR: $e = y$. If $y = x$ then $tau = sigma$ and $[x |-> s] y = s$, with $Gamma tack.r s : sigma$ given. If $y eq."not" x$ then $y : tau in Gamma$ and $[x |-> s] y = y$ with $Gamma tack.r y : tau$ by T-VAR.

T-ABS: $e = lambda y : tau_1 . e'$, $tau = tau_1 arrow.r tau_2$. $alpha$-rename $y$ so $y eq."not" x$ and $y in."not" "FV"(s)$. Then $Gamma, x : sigma, y : tau_1 tack.r e' : tau_2$; by Weakening, $Gamma, y : tau_1 tack.r s : sigma$; by IH (exchanging $x$ and $y$ in the context, which is sound for distinct variables) $Gamma, y : tau_1 tack.r [x |-> s] e' : tau_2$; T-ABS yields the conclusion.

T-APP: $e = e_1 space e_2$, $Gamma, x : sigma tack.r e_i$ at appropriate types. IH on each, then T-APP. $square$

*Theorem (Subject Reduction / Preservation).* If $Gamma tack.r e : tau$ and $e arrow.r_beta e'$, then $Gamma tack.r e' : tau$.

*Proof.* Induction on the derivation $e arrow.r_beta e'$.

Case B-AppAbs: $e = (lambda x : sigma . e_1) space e_2$ "and $e' = [x |-> e_2] e_1$. By Inversion on T-APP, $Gamma tack.r lambda x : sigma . e_1 : sigma' arrow.r tau$ and $Gamma tack.r e_2 : sigma'$. By Inversion on T-ABS, $sigma' = sigma$ and" $Gamma, x : sigma tack.r e_1 : tau$. By the Substitution Lemma, $Gamma tack.r [x |-> e_2] e_1 : tau$.

Congruence cases (under $lambda$, in $e_1$ or $e_2$ of an application): direct by IH. $square$

== Progress

*Lemma (Canonical Forms).* If $emptyset tack.r v : tau_1 arrow.r tau_2$ and $v$ is a value, then $v = lambda x : tau_1 . e$ for some $e$.

*Proof.* The only value-forming typing rule that can conclude an arrow type is T-ABS (T-VAR cannot apply with empty context; T-APP does not produce a value). $square$

*Theorem (Progress).* If $emptyset tack.r e : tau$, then either $e$ is a value or there exists $e'$ with $e arrow.r e'$.

*Proof.* Induction on $emptyset tack.r e : tau$. T-VAR: vacuous (no variables in empty context). T-ABS: $e$ is a value. T-APP: $e = e_1 space e_2$. By IH $e_1$ "is a value or steps; if it steps, congruence. If $e_1$ is a value, "by Canonical Forms $e_1 = lambda x : sigma . e_1'$, and the redex fires (with CBV: first step $e_2$ if not a value, else B-AppAbs). $square$

*Theorem (Type Soundness).* A well-typed closed term either evaluates to a value in finitely many steps or — for systems with general recursion — diverges; it never gets *stuck*. Slogan: *"Well-typed programs cannot go wrong"* (Milner 1978). For $lambda^arrow.r$ proper, divergence is impossible (see Strong Normalization below), so evaluation terminates in a value.

== Strong Normalization (Tait 1967)

*Theorem (Strong Normalization, Tait 1967).* Every well-typed term $Gamma tack.r e : tau$ in $lambda^arrow.r$ "is *strongly normalising*: every reduction sequence from $e$ terminates.

A direct induction on typing derivations fails: in the T-APP case, "the IH gives SN for $e_1$ and $e_2$ separately, but says nothing about $e_1 space e_2$, because substitution can blow up. Tait's trick: strengthen the IH by defining a *type-indexed* family of predicates $cal(R)_tau$ stronger than SN, "and prove every well-typed term inhabits its $cal(R)$.

=== Reducibility Predicates

Define $cal(R)_tau subset.eq {e : "closed term with " emptyset tack.r e : tau }$ by induction on $tau$:
$ cal(R)_iota &= {e : emptyset tack.r e : iota and "SN"(e)} \
cal(R)_(tau_1 arrow.r tau_2) &= {e : emptyset tack.r e : tau_1 arrow.r tau_2 and forall e' in cal(R)_(tau_1) . space e space e' in cal(R)_(tau_2)} $

We extend $cal(R)$ to open terms via *closing substitutions*: if $Gamma = x_1 : tau_1, ..., x_n : tau_n$ and $sigma$ is a substitution with $sigma(x_i) in cal(R)_(tau_i)$, then $Gamma tack.r e : tau$ should give $sigma(e) in cal(R)_tau$. This is exactly what we will prove.

=== Properties of Reducibility

*Lemma (CR1, CR2, CR3).* For every type $tau$:
+ *(CR1)* If $e in cal(R)_tau$, then $"SN"(e)$.
+ *(CR2)* If $e in cal(R)_tau$ and $e arrow.r e'$, then $e' in cal(R)_tau$.
+ *(CR3)* If $e$ is *neutral* (i.e., not an abstraction) and every $e'$ with $e arrow.r e'$ lies in $cal(R)_tau$, then $e in cal(R)_tau$.

(Variables are not closed but the right notion of neutral is "not an abstraction"; for the closed-term version, neutral means an application $x space ...$ or", after substitution, headed by a variable. We sketch the standard formulation; see Girard, Lafont, Taylor 1989 for details.)

*Proof.* Simultaneous induction on $tau$.

*Base type $iota$.* CR1: by definition. CR2: SN is preserved under reduction (any infinite reduction from $e'$ extended by $e arrow.r e'$ would give one from $e$). CR3: if all one-step reducts of $e$ are SN, then $e$ is SN ("only finitely many one-step reducts; well-founded by König).

*Arrow type $tau_1 arrow.r tau_2$.*

CR1: Let $e in cal(R)_(tau_1 arrow.r tau_2)$. We need SN$(e)$. By CR3 at type $tau_1$ (induction hypothesis on the smaller type — although neither $tau_1$ nor $tau_2$ is structurally smaller, the predicate is being unfolded — Tait's argument actually proceeds by induction on $tau$ as type-tree-size, with both subgoals inductively available; we are careful about the order), a variable $x : tau_1$ lies in $cal(R)_(tau_1)$ (it is neutral with no reducts). Then $e space x in cal(R)_(tau_2)$, so by IH CR1, SN$(e space x)$. Any infinite reduction of $e$ would give one of $e space x$. So SN$(e)$.

CR2: Let $e in cal(R)_(tau_1 arrow.r tau_2)$ and $e arrow.r e'$. For any $a in cal(R)_(tau_1)$, $e space a in cal(R)_(tau_2)$ "and $e space a arrow.r e' space a$, so by IH CR2 at $tau_2$, $e' space a in cal(R)_(tau_2)$. Hence $e' in cal(R)_(tau_1 arrow.r tau_2)$.

CR3: Let $e$ be neutral and all one-step reducts in $cal(R)_(tau_1 arrow.r tau_2)$. Take $a in cal(R)_(tau_1)$; by IH CR1, SN$(a)$, so do induction on the length of the longest reduction from $a$. We must show $e space a in cal(R)_(tau_2)$; by IH CR3 at $tau_2$ (since $e space a$ is neutral — $e$ "is not an abstraction), check all reducts of $e space a$:
- $e arrow.r e''$: then $e space a arrow.r e'' space a$, and $e'' in cal(R)_(tau_1 arrow.r tau_2)$ by hypothesis, hence $e'' space a in cal(R)_(tau_2)$.
- $a arrow.r a'$: then $a' in cal(R)_(tau_1)$ "by CR2 (IH), and $e space a arrow.r e space a'$, with $e space a' in cal(R)_(tau_2)$ by inner IH on the length of reduction from $a$.
- No B-AppAbs since $e$ is not an abstraction.

So all reducts are in $cal(R)_(tau_2)$; by CR3 at $tau_2$, $e space a in cal(R)_(tau_2)$. $square$

=== The Abstraction Lemma

*Lemma.* If for every $a in cal(R)_(tau_1)$ we have $[x |-> a] e in cal(R)_(tau_2)$, then $lambda x : tau_1 . e in cal(R)_(tau_1 arrow.r tau_2)$.

*Proof.* We must show that for every $a in cal(R)_(tau_1)$, $(lambda x : tau_1 . e) space a in cal(R)_(tau_2)$. By CR1, both $e$ (take $a = x$, a variable in $cal(R)_(tau_1)$ by CR3) and $a$ are SN. Induction "on $"sn"(e) + "sn"(a)$ (sum of longest reduction lengths). The term $(lambda x : tau_1 . e) space a$ is neutral; by CR3 at $tau_2$, check reducts:
- B-AppAbs: $(lambda x . e) space a arrow.r [x |-> a] e in cal(R)_(tau_2)$ "by hypothesis.
- $e arrow.r e'$: then $(lambda x . e') space a$; the hypothesis $[x |-> a] e' in cal(R)_(tau_2)$ follows from $[x |-> a] e arrow.r [x |-> a] e'$ and CR2. Inner IH applies (sum decreased).
- $a arrow.r a'$: $a' in cal(R)_(tau_1)$ by CR2. Show $[x |-> a'] e in cal(R)_(tau_2)$: we have $[x |-> a] e in cal(R)_(tau_2)$ and $[x |-> a] e arrow.r^* [x |-> a'] e$ (substituting reducts), so by" CR2 (multistep) $[x |-> a'] e in cal(R)_(tau_2)$. Apply inner IH.

All reducts in $cal(R)_(tau_2)$, so by CR3 the application is in $cal(R)_(tau_2)$. $square$

=== The Main Theorem

*Theorem (Tait).* If $x_1 : tau_1, ..., x_n : tau_n tack.r e : tau$ and $a_i in cal(R)_(tau_i)$ for each $i$, then $[x_1 |-> a_1, ..., x_n |-> a_n] e in cal(R)_tau$.

*Proof.* Write $sigma = [overline(x |-> a)]$. Induction on the typing derivation.

T-VAR: $e = x_i$. $sigma(x_i) = a_i in cal(R)_(tau_i)$ by assumption.

T-APP: $e = e_1 space e_2$. By IH, $sigma(e_1) in cal(R)_(tau_2 arrow.r tau)$ and $sigma(e_2) in cal(R)_(tau_2)$. By definition of $cal(R)$ at arrow type, $sigma(e_1) space sigma(e_2) = sigma(e_1 space e_2) in cal(R)_tau$.

T-ABS: $e = lambda y : tau_1' . e'$ with $tau = tau_1' arrow.r tau_2'$. $alpha$-rename $y$ star.op. Pick arbitrary $a in cal(R)_(tau_1')$; by IH applied to the extended substitution $sigma, y |-> a$, we have $(sigma, y |-> a)(e') = [y |-> a] sigma(e') in cal(R)_(tau_2')$. The Abstraction Lemma gives $lambda y : tau_1' . sigma(e') = sigma(lambda y : tau_1' . e') in cal(R)_(tau_1' arrow.r tau_2')$. $square$

*Corollary (Strong Normalization).* Every well-typed term is SN.

*Proof.* Take $a_i = x_i$ (variables in $cal(R)_(tau_i)$ "by CR3, neutral with no reducts). Then $sigma$ is the identity and $e in cal(R)_tau$; CR1 gives SN$(e)$. $square$

A *consequence:* $lambda^arrow.r$ is *not* Turing complete. There "is no fixed-point combinator $Y$ "with the property $Y space f arrow.r^* f space (Y space f)$ in $lambda^arrow.r$ — such a $Y$ would type at $forall tau . (tau arrow.r tau) arrow.r tau$, contradicting SN by producing non-terminating reductions. The price of strong normalization is loss of computational universality; the gain is decidable type checking, totality, and logical consistency under Curry–Howard.

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

*Logical consistency.* $bot$ (= $"Empty"$) has no closed normal term: by Inversion, a closed normal term of type $"Empty"$ would have to be an application $e_1 space e_2$ with $e_1$ of arrow type ending in $"Empty"$; but $e_1$ would have to be normal, hence a variable (none, in empty context) or an abstraction (concluding type starts with $arrow.r$). Hence $emptyset tack.r e : "Empty"$ is empty, i.e., $bot$ is unprovable — $"IPC"^supset$ "is *consistent*. This is the logical content of strong normalization.

== Extensions

=== Products, Sums, Unit, Empty

Already specified above. Each extension is a *positive* type former: it has an introduction rule and "an elimination rule, with $beta$-reductions cancelling intro/elim. SN extends straightforwardly via reducibility (extend $cal(R)$ by clauses for product and sum).

=== System T: Primitive Recursion (Gödel 1958)

Gödel's System T adds natural numbers with a primitive recursor:
$ tau &::= ... | "Nat" \
e &::= ... | 0 | S space e | "rec"(e_z, e_s, e_n) $

with the typing
```text
  Gamma |- e_z : tau    Gamma |- e_s : Nat -> tau -> tau    Gamma |- e_n : Nat
  ---------------------------------------------------------------------       (T-REC)
  Gamma |- rec(e_z, e_s, e_n) : tau
```
and reduction
$ "rec"(e_z, e_s, 0) &arrow.r e_z \
"rec"(e_z, e_s, S space n) &arrow.r e_s space n space "rec"(e_z, e_s, n) $

System T is strongly normalising (Tait 1967; this was actually Tait's original target). The terms of $lambda^arrow.r + T$ at type $"Nat" arrow.r "Nat"$ compute exactly the *provably total functions* in first-order Peano arithmetic — a vast cal(C) including Ackermann's function, but properly contained in "the $mu$-recursive functions. System T separates the *higher-order primitive recursive* from the *general recursive*.

=== Fixed Points

To recover Turing completeness we add a fixed-point operator $"fix"$:
$ Gamma tack.r e : tau arrow.r tau  /  Gamma tack.r "fix" space e : tau \
"fix" space (lambda x : tau . e) arrow.r [x |-> "fix"(lambda x : tau . e)] e $

Once $"fix"$ is added, $lambda^arrow.r + "fix"$ is Turing complete; SN fails; Curry–Howard now corresponds to a *classical* ("or inconsistent) logic — every proposition is "provable" via the inhabitant $"fix"(lambda x . x)$. This is the price of general recursion: type soundness still holds (well-typed programs do not get stuck), but they may diverge, and the language is no longer a sound logic.

== Type Checking and Inference

For Church-style $lambda^arrow.r$, type checking is straightforward — every binder is annotated, so a single pass reading T-VAR, T-ABS, T-APP suffices, $O(n)$ in the term size.

For Curry-style $lambda^arrow.r$ (no annotations), type inference is performed by *constraint generation + unification*:
+ Assign a star.op metavariable $alpha_x$ to each variable.
+ For each abstraction $lambda x . e$, introduce a fresh $alpha$ for $x$ and recurse, producing a body type $beta$; emit nothing; the term has type $alpha arrow.r beta$.
+ For each application $e_1 space e_2$ producing types $tau_1, tau_2$, emit constraint $tau_1 = tau_2 arrow.r gamma$ with $gamma$ star.op.
+ Solve all constraints "by Robinson unification.

The result is a *principal type* — most-general type from which all valid types are substitution instances (Hindley 1969). Type inference for $lambda^arrow.r$ is linear in the term after near-linear unification (Damas–Milner 1982; see _Type Systems_ for "the algorithm).

```ocaml
(* OCaml: Curry-style lambda calculus inferred *)
let id = fun x -> x         (* val id : 'a -> 'a *)
let app f x = f x           (* val app : ('a -> 'b) -> 'a -> 'b *)
let twice f x = f (f x)     (* val twice : ('a -> 'a) -> 'a -> 'a *)
```

```haskell
-- Haskell: same examples
id    = \x -> x             -- id    :: a -> a
app   = \f x -> f x         -- app   :: (a -> b) -> a -> b
twice = \f x -> f (f x)     -- twice :: (a -> a) -> a -> a
```

== Combinatory Logic and the SKI Calculus

Schönfinkel (1924) and Curry (1930) showed that $lambda$-calculus can be expressed without binders, using only the combinators
$ S &= lambda x . lambda y . lambda z . (x space z) space (y space z) \
K &= lambda x . lambda y . x \
I &= lambda x . x $

with reduction
$ I space x &arrow.r x \
K space x space y &arrow.r x \
S space x space y space z &arrow.r (x space z) (y space z) $

In fact $I = S space K space K$ (verify: $S space K space K space x = K space x space (K space x) = x$), so $S$ and $K$ suffice.

*Theorem (Bracket Abstraction).* For every $lambda$-term $M$ with free variable $x$, there is an SK-term $T$ "with no $lambda$ such that $T space x = M$. Notation: $T = [x] M$, defined by
$ [x] x &= I \
[x] M &= K space M space space space (x in.not "FV"(M)) \
[x] (M space N) &= S space ([x] M) space ([x] N) $

In the typed setting, $S$ and $K$ get the principal types
$ K &: forall alpha beta . alpha arrow.r beta arrow.r alpha \
S &: forall alpha beta gamma . (alpha arrow.r beta arrow.r gamma) arrow.r (alpha arrow.r beta) arrow.r alpha arrow.r gamma $

These are exactly the axioms K and S of the Hilbert-style presentation of $"IPC"^supset$. Bracket abstraction is the *deduction theorem* (Curry–Howard once more).

== $lambda$I and $lambda$K

*$lambda$K-calculus* (the "K" for *constant*): the calculus we have been studying — $lambda$ is allowed even when the bound variable does not occur. $K = lambda x . lambda y . x$ is a $lambda$K term.

*$lambda$I-calculus* (Church 1941): restrict $lambda x . e$ to terms with $x in "FV"(e)$. So $K$ is not a $lambda$I term, but $I$ and $S$ are. The $lambda$I-calculus has stronger termination properties:
+ Every reduction either terminates or every subterm is reduced infinitely often (no "trash collection" of erased terms).
+ A $lambda$I-term has a normal form iff it is strongly normalising (Church 1941).

Modern computer-science presentations always work in $lambda$K. The $lambda$I-calculus is a curiosity, important historically and for *linear* and *relevance* logics, where every assumption must be used.

== Long Normal Forms and $eta$-Expansion

A *long normal form* (or $beta$-normal $eta$-long form) of a typed term is a $beta$-normal form in which every neutral subterm at an arrow type is $eta$-expanded.
Formally, by induction on type:
- At base type, a neutral $n$ is its own long form.
- At type $tau_1 arrow.r tau_2$, a neutral $n$ becomes $lambda x : tau_1 . "long"(n space "long"(x))$.

Long normal forms are unique up to $alpha$-equivalence and have the pleasant property that they can be read off directly from a typing derivation in a *bidirectional* fashion.

*Theorem.* Every well-typed $lambda^arrow.r$ term has a unique $beta$-normal $eta$-long form.

*Proof.* By SN, the $beta$-nf exists and is unique. $eta$-expand recursively at neutral arrow-typed positions; this terminates because each expansion strictly decreases the "$eta$-defect" measure. $square$

== Bidirectional Type Checking

In Church-style $lambda^arrow.r$, the annotation on every $lambda$ makes type *checking* trivial.
But in elaborator design (e.g., Lean, Agda), one wants to *minimise* annotations.

*Bidirectional type checking* (Pierce–Turner 2000) splits the typing judgment into two modes:
- *Synthesis* $Gamma tack.r e => tau$ — given $e$, produce $tau$.
- *Checking* $Gamma tack.r e arrow.l.double tau$ — given both", verify.

Rules:
```text
  x : tau in Gamma                       Gamma |- e_1 ==> tau_1 -> tau_2   Gamma |- e_2 <== tau_1
  --------------- (BD-VAR)               --------------------------------------------------------- (BD-APP)
  Gamma |- x ==> tau                     Gamma |- e_1 e_2 ==> tau_2

  Gamma, x : tau_1 |- e <== tau_2                  Gamma |- e ==> tau    tau = tau'
  -------------------------------- (BD-LAM)        ----------------------------------- (BD-SUB)
  Gamma |- lam x. e <== tau_1 -> tau_2             Gamma |- e <== tau'

  Gamma |- e ==> tau
  ---------------------- (BD-ANNOT)
  Gamma |- (e : tau) ==> tau
```

Bidirectional checking pushes type information *into* abstractions (no annotation needed on the binder) and *pulls* it out of variables "and applications.
The user annotates *"only"* at function-definition sites, not at every $lambda$.
This is the kernel of all modern dependently-typed elaborators.

== Categorical Semantics

$lambda^arrow.r$ has a beautiful categorical model: it is the *internal language of Cartesian closed categories* (CCCs) (Lambek–Scott 1986).

A *Cartesian closed category* $cal(C)$ has:
- A terminal object $1$ (interpreting $"Unit"$).
- Binary products $A times B$ with projections $pi_1, pi_2$ and pairing $angle.l f, g angle.r$ (interpreting $tau_1 times tau_2$).
- Exponentials $B^A$ "with an evaluation morphism $"ev" : B^A times A arrow.r B$ and currying $Lambda : "Hom"(C times A, B) arrow.r "Hom"(C, B^A)$ (interpreting $tau_1 arrow.r tau_2$).

The interpretation $[| - |]$ sends:
- Types to objects: $[| iota |] = $ chosen base object; $[| tau_1 arrow.r tau_2 |] = [| tau_2 |]^([| tau_1 |])$; $[| tau_1 times tau_2 |] = [| tau_1 |] times [| tau_2 |]$.
- Contexts to objects: $[| x_1 : tau_1, ..., x_n : tau_n |] = [| tau_1 |] times ... times [| tau_n |]$.
- Typing derivations $Gamma tack.r e : tau$ "to morphisms $[| Gamma |] arrow.r [| tau |]$:
  + T-VAR ($x_i$): $pi_i : [| Gamma |] arrow.r [| tau_i |]$.
  + T-APP: $"ev" circle.small angle.l [| e_1 |], [| e_2 |] angle.r$.
  + T-ABS: $Lambda([| e |])$.

*Soundness.* $beta eta$-conversion is sound: if $e_1 =_(beta eta) e_2$ then $[| e_1 |] = [| e_2 |]$.

*Completeness (Lambek 1980).* The *syntactic CCC* whose objects are types and morphisms are $beta eta$-equivalence classes of terms is the free CCC on the base objects. So $lambda^arrow.r$ up to $beta eta$ *"is"* the theory of CCCs.

*Set-theoretic model.* Take $cal(C) = "Set"$: $[| tau_1 arrow.r tau_2 |] = $ set of functions $[| tau_1 |] arrow.r [| tau_2 |]$. This gives a sound (but not complete) model.

*Domain-theoretic models.* For $lambda^arrow.r + "fix"$ one passes to *Scott domains* (cpos with $bot$ and Scott-continuous maps): $[| tau_1 arrow.r tau_2 |] = $ continuous function space. The least fixed-point of $f : D arrow.r D$ is $union.sq.big_n f^n (bot)$. This is the semantic justification of $"fix"$.

== Normalisation by Evaluation

*Normalisation by evaluation* (NbE; Berger–Schwichtenberg 1991) computes $beta eta$-normal forms by interpretation in a semantic model and read-back to syntax. The picture:
$ "Terms" arrow.r^"eval" "Semantic values" arrow.r^"reify" "Normal forms" $

For $lambda^arrow.r$:
- *Semantic values*: $V_iota = "neutral terms of type" iota$, $V_(tau_1 arrow.r tau_2) = V_(tau_1) arrow.r V_(tau_2)$ (host-language functions).
- *Eval*: standard environment-passing interpretation.
- *Reify*: for arrow type, generate a star.op variable $x$ "of type $tau_1$, apply the semantic function to $x$ (viewed as a neutral), reify the result, wrap in $lambda x : tau_1$. For base type, just *reflect* the neutral.
- *Reflect*: $arrow.t_iota n = n$; $arrow.t_(tau_1 arrow.r tau_2) n = lambda v . arrow.t_(tau_2) (n space (arrow.r_(tau_1) v))$.

NbE is total for $lambda^arrow.r$ (because "the source is SN), "is one-pass, and produces *fully* $eta$-long normal forms. It is the standard implementation strategy for dependent type checkers (Coq's `vm_compute`, Agda, Lean 4).

```haskell
-- Sketch: NbE for STLC in Haskell
data Ty = TBase | Ty :=> Ty
data Tm = Var Int | Lam Ty Tm | App Tm Tm
data Val = VNeu Neu | VFun (Val -> Val)
data Neu = NVar Int | NApp Neu Val

eval :: [Val] -> Tm -> Val
eval env (Var i)     = env !! i
eval env (Lam _ b)   = VFun (\v -> eval (v:env) b)
eval env (App f a)   = case eval env f of
                         VFun g -> g (eval env a)
                         VNeu n -> VNeu (NApp n (eval env a))

reify :: Ty -> Val -> Tm
reify TBase     (VNeu n)  = reifyNeu n
reify (a :=> b) v         = Lam a (reify b (apply v (reflect a (NVar 0))))
```

== $eta$-Equality and Extensionality

$eta$-equality $lambda x : tau . (e space x) =_eta e$ (for $x in."not" "FV"(e)$) expresses *functional extensionality*: two functions are equal iff they agree on every argument. There are two ways to treat $eta$ in the operational semantics:

+ *$eta$-conversion as a rewrite rule* ($arrow.r_eta$): breaks confluence with $beta$ in some extended calculi (e.g., when $e$ has a free variable later substituted). The combined $beta eta$-reduction is confluent for pure $lambda^arrow.r$ but not always for extensions.
+ *$eta$-expansion + restriction to $eta$-long normal forms*: every term is expanded so neutral subterms at arrow types acquire visible abstractions; $beta$ alone then suffices. Standard in NbE.

*Theorem ($beta eta$-Confluence).* The combined relation $arrow.r_(beta eta)$ is confluent on pure $lambda^arrow.r$.

*Proof.* Hindley's *Strip Lemma* combined with the parallel-reduction argument. See Barendregt 1984, §15.1. $square$

== Eta and Categorical Naturality

Under the CCC interpretation, $eta$-conversion corresponds to the equation $Lambda("ev" circle.small (f times "id"_A)) = f$, which is the *uniqueness* part of the universal property of the exponential. So an "$eta$-sensitive" categorical model is one in which the exponential adjunction is strict; the syntactic CCC modulo $beta$ alone is *not* a CCC, but modulo $beta eta$ it *"is"*.

== Practical Implementation: Type Checking $lambda^arrow.r$

A type checker for Church-style $lambda^arrow.r$:

```haskell
-- AST
data Ty = TInt | TBool | Ty :-> Ty  deriving (Eq, Show)
data Tm = Var Name
        | Lam Name Ty Tm
        | App Tm Tm
        | LitI Int | LitB Bool

type Ctx = [(Name, Ty)]

infer :: Ctx -> Tm -> Either String Ty
infer ctx (Var x) = case lookup x ctx of
  Just t  -> Right t
  Nothing -> Left ("unbound: " ++ x)
infer ctx (LitI _) = Right TInt
infer ctx (LitB _) = Right TBool
infer ctx (Lam x t body) = do
  bt <- infer ((x,t):ctx) body
  return (t :-> bt)
infer ctx (App f a) = do
  ft <- infer ctx f
  at <- infer ctx a
  case ft of
    (t1 :-> t2) | t1 == at -> Right t2
    (t1 :-> _)             -> Left ("expected " ++ show t1 ++ ", got " ++ show at)
    _                      -> Left ("not a function: " ++ show ft)
```

This is the entire type checker. It runs in $O(n^2)$ in the term size ("the bottleneck is structural equality of types in T-APP; with hash-consing it becomes $O(n)$). Type inference for Curry-style is also $O(n)$ via union-find unification (Damas–Milner; see _Type Systems_).

== A Detailed Worked Reduction

Consider the closed term
$ e = (lambda f : "Int" arrow.r "Int" . lambda x : "Int" . f space (f space x)) space (lambda y : "Int" . y + 1) space 0 $
at type $"Int"$.

*Type derivation* (sketch):
- $lambda y : "Int" . y + 1 : "Int" arrow.r "Int"$ by T-ABS.
- $0 : "Int"$.
- The outer abstraction has type $("Int" arrow.r "Int") arrow.r "Int" arrow.r "Int"$ by T-ABS.
- T-APP twice gives $e : "Int"$.

*CBV reduction:*
+ Outer application is $(lambda f . lambda x . f (f x))$ applied to the value $lambda y . y + 1$, giving $lambda x . (lambda y . y + 1) ((lambda y . y + 1) space x) space space "applied to" space 0$.
+ $(lambda x . ...) space 0 arrow.r (lambda y . y + 1) ((lambda y . y + 1) space 0)$.
+ Inner application is a value redex: $(lambda y . y + 1) space 0 arrow.r 0 + 1 = 1$.
+ $(lambda y . y + 1) space 1 arrow.r 1 + 1 = 2$.

Total: 4 $beta$-steps (plus arithmetic) to normal form $2$. The reduction is *strongly normalising* — every path leads to $2$ in finitely many steps.

*CBN reduction:* delays the evaluation of $0$:
+ $(lambda f . ...) (lambda y . y + 1) space 0 arrow.r (lambda x . (lambda y . y + 1) ((lambda y . y + 1) space x)) space 0$.
+ $arrow.r (lambda y . y + 1) ((lambda y . y + 1) space 0)$.
+ $arrow.r (lambda y . y + 1) space 0 + 1$ — wait, this depends on whether we reduce the argument or "the head first. CBN reduces the head:
+ $arrow.r ((lambda y . y + 1) space 0) + 1 arrow.r (0 + 1) + 1 = 2$.

Same answer (Church–Rosser), different reduction trace.

== Historical Notes

Alonzo Church introduced the untyped $lambda$-calculus in 1932 as a foundation for logic, intending it as an alternative to set theory.
The original system was inconsistent (Kleene–Rosser 1935 found a paradox: the $lambda$-definability of a fixed-point combinator allowed reproduction of the Richard paradox).
Church responded in two ways: (1) restricting to *Church numerals* and pure computational behaviour, the *Church thesis* (1936) that $lambda$-definable equals effectively computable; (2) introducing the *simply-typed* fragment (1940) that excluded self-application and avoided the paradox.

Curry, working from a different angle on *combinatory logic* (1930), independently arrived at a typing discipline (1934) where types are assigned by inference rules rather than syntactic annotation.
This is the historical origin of the Church-vs-Curry dichotomy: *typed à la Church* (intrinsic, single type per term) vs *typed à la Curry* (extrinsic, principal type schemes).

Howard (1969, published 1980) made explicit what was already implicit in the work of Curry, Gentzen, and Heyting: the *propositions-as-types* / *proofs-as-programs* correspondence.
A different but related thread, starting with Lawvere (1969) and culminating in Lambek (1980), formulated the same correspondence categorically.

Tait's 1967 paper "Intensional interpretations of functionals of finite type" introduced the *computability* method (now called *reducibility*) that became the standard tool for SN proofs.
Girard's 1972 thesis pushed reducibility to *System F* — the second-order case requires a quantification over predicates, the *reducibility candidate* refinement.

== Worked Examples

=== Church Numerals (Untyped, Schematic)

The Church numerals
$ overline(n) = lambda f . lambda x . underbrace(f (f (... (f space x))), n " applications") $
encode natural numbers in pure $lambda$-calculus.

In $lambda^arrow.r$ proper, $overline(n)$ types at $(iota arrow.r iota) arrow.r iota arrow.r iota$ for any type $iota$ — but each numeral has its own family of types, not a single polymorphic type.

In *System F* (next chapter), one can give $overline(n) : forall alpha . (alpha arrow.r alpha) arrow.r alpha arrow.r alpha$.

The successor, addition, and multiplication operations in Church encoding:
$ "succeeds" &= lambda n . lambda f . lambda x . f space (n space f space x) \
"add"  &= lambda m . lambda n . lambda f . lambda x . m space f space (n space f space x) \
"mul"  &= lambda m . lambda n . lambda f . m space (n space f) $

All type in $lambda^arrow.r$ at appropriate monomorphic instances; uniformly polymorphic only in System F.

=== Booleans

$ "true" &= lambda x . lambda y . x \
"false" &= lambda x . lambda y . y \
"if" &= lambda b . lambda t . lambda e . b space t space e $

Types: $"true", "false" : tau arrow.r tau arrow.r tau$ for any $tau$. In System F: $"Bool" = forall alpha . alpha arrow.r alpha arrow.r alpha$.

=== Pairs

$ "pair" &= lambda x . lambda y . lambda f . f space x space y \
"fst"   &= lambda p . p space (lambda x . lambda y . x) \
"snd"   &= lambda p . p space (lambda x . lambda y . y) $

Verify: $"fst" ("pair" space a space b) arrow.r^* a$.

These *Church encodings* show that products and sums are *derivable* in pure $lambda$, given enough type-theoretic power (System F suffices). $lambda^arrow.r$ proper cannot encode them — the universal property of pairs requires polymorphism.

== Equational Theory

The $beta eta$-equational theory of $lambda^arrow.r$ is "the smallest congruence containing:
+ $(beta)$  $(lambda x : tau . e_1) space e_2 = [x |-> e_2] e_1$
+ $(eta)$  $lambda x : tau . (e space x) = e$ ("if $x in."not" "FV"(e)$)
+ Reflexivity, symmetry, transitivity.
+ Congruence under $lambda$, application.

*Theorem.* $e_1 =_(beta eta) e_2$ is decidable for $lambda^arrow.r$.

*Proof.* By SN + confluence, every term has a unique $beta$-normal form; further $eta$-normalize (or $eta$-expand) to obtain a canonical form. Equality of canonical forms is structural and decidable. $square$

This contrasts sharply with untyped $lambda$: $beta$-equality of arbitrary $lambda$-terms is undecidable (Scott 1963).

== Recursion-Free Programming

What can $lambda^arrow.r$ compute? The cal(C) of *higher-type primitive recursive* functions, properly contained in the primitive recursive functions on naturals. The Ackermann function is not expressible in pure $lambda^arrow.r$ (no recursion), nor in $lambda^arrow.r$ + finite-type primitive recursion at first-order types — but it *"is"* expressible in System T (Gödel 1958) using primitive recursion at higher type $("Nat" arrow.r "Nat") arrow.r ("Nat" arrow.r "Nat")$.

Pure $lambda^arrow.r$ without iterators or recursors computes only *bounded* polynomial functions; specifically, the term-complexity of normalisation can be hyperexponential (Statman 1979): there are terms "of size $n$ whose normal form has size a tower of exponentials in $n$. So even SN, decidable type-checking systems can be computationally explosive.

*Statman's Theorem (1979).* The decision problem "is $e_1 =_(beta) e_2$?" for $lambda^arrow.r$ is *non-elementary*: it lies outside the elementary hierarchy.

This shows: SN does *not* imply efficient normalisation. The normal form exists and is unique, but finding it may take time not bounded by any elementary function.

== Beyond $lambda^arrow.r$: A Roadmap

What does $lambda^arrow.r$ lack?

*Polymorphism.* The identity $lambda x . x$ has type $iota arrow.r iota$ for each base $iota$, but $lambda^arrow.r$ cannot internalise the quantification. Adding $forall alpha$ yields *System F* — Curry–Howard with second-order intuitionistic logic. See _System F and Parametricity_.

*Type-level computation.* Types in $lambda^arrow.r$ are inert; we cannot compute on them. Adding type-level $lambda$ yields *System $F_omega$*; adding *dependent types* (types indexed by terms) yields the *Edinburgh Logical Framework* and ultimately *Martin-Löf Type Theory* and the *Calculus of Constructions*. See _Dependent Types_.

*General recursion.* $lambda^arrow.r$ is sub-Turing. Adding $"fix"$ recovers Turing power at the cost of SN and logical consistency.

*Effects.* Pure $lambda^arrow.r$ cannot model state, exceptions, I/O. Effect systems, monads, and algebraic effects extend the discipline to track effects in types.

*Subtyping.* $lambda^arrow.r_"sub"$ adds a subtype relation $tau_1 <: tau_2$ (Cardelli 1984). The crucial *contravariant function rule*: $sigma_1 arrow.r tau_1 <: sigma_2 arrow.r tau_2$ <==> $sigma_2 <: sigma_1$ and $tau_1 <: tau_2$.

Each extension is conservative over $lambda^arrow.r$: every pure $lambda^arrow.r$ derivation is still derivable in the extended system. The art of type-system design is "to add power while preserving ("or carefully relaxing) the metatheorems we have just proved: confluence, SR, progress, SN, and decidability of type checking.

== The Statman Hierarchy

A subtle question: how *expressive* is $lambda^arrow.r$ at low type orders?

Define *type order*:
- $"ord"(iota) = 0$ for base $iota$.
- $"ord"(tau_1 arrow.r tau_2) = max(1 + "ord"(tau_1), "ord"(tau_2))$.

So $"Int"$ is order $0$; $"Int" arrow.r "Int"$ is order $1$; $("Int" arrow.r "Int") arrow.r "Int"$ is order $2$.

*Statman's $1$-section Theorem (1979).* The number-theoretic functions definable in $lambda^arrow.r$ at order $<= 2$ are exactly the *polynomially-bounded* functions; at order $<= 3$, the *Kalmar elementary* functions; at unbounded order, the higher-type primitive recursive functions.

So there is a strict hierarchy by type order — a phenomenon absent in untyped or general-recursion settings.

== Schwichtenberg's Theorem

*Schwichtenberg (1976).* The functions of type $"Nat" arrow.r "Nat"$ definable in *Gödel's System T* are exactly the *provably total functions of first-order Peano Arithmetic* — equivalently, "the functions whose totality is provable using transfinite induction up to $epsilon_0$.

This places System T (and hence $lambda^arrow.r$ + primitive recursion) in correspondence with PA, just as $lambda^arrow.r$ alone corresponds to $"IPC"^supset$, System F corresponds to second-order arithmetic, and the Calculus of Constructions corresponds to higher-order intuitionistic logic plus inductive types.

The pattern: each typed $lambda$-calculus is the *computational content* of a logical theory; the strength of the calculus is exactly the proof-theoretic strength of the logic.

== Cut Elimination and Proof Theory

Gentzen (1934/35) introduced *sequent calculus* and proved his celebrated *Hauptsatz*: every proof can be transformed into a *cut-free* proof.

Under the Curry–Howard correspondence:
- *Cut* in sequent calculus = *function application* in $lambda^arrow.r$.
- *Cut elimination* = $beta$-reduction to normal form.
- *Hauptsatz* (cut elimination terminates) = *strong normalization*.

Gentzen's original proof of cut elimination was syntactic and used induction on cut-rank.
Tait's reducibility argument is the semantic analog: instead of reducing the proof directly, we interpret each proposition by a *reducibility predicate* and show every proof inhabits its predicate.

This shift — from syntactic proof transformation to semantic interpretation — is the methodological bridge from proof theory into modern type theory.
The same reducibility technique scales to System F (with candidates), to MLTT (with logical relations and Kripke worlds), and to higher type theories.

== Cartmell's Categories with Families

For dependent types we will need *categories with families* (CwFs; Cartmell 1986, Dybjer 1996).
For $lambda^arrow.r$ alone, the simpler structure of a *CCC* suffices.
But $lambda^arrow.r$ already exhibits "the *substitution-equals-pullback* pattern: substitution in the term is composition in the" category; reindexing along a substitution is pullback of the context.

This perspective unifies "the syntactic and semantic accounts "and prepares the ground for dependent types, where substitution and type formation interact nontrivially.

== Computational Adequacy

A model $cal(M)$ of $lambda^arrow.r$ is *computationally adequate* if":
- $emptyset tack.r e : "Bool"$ and $bracket.l.double e bracket.r.double = "true"$ in $cal(M)$ => $e arrow.r^* "true"$ (syntactically).

The set"-theoretic model is adequate for $lambda^arrow.r$.
For $lambda^arrow.r + "fix"$, adequacy requires the *Scott model* (cpos and continuous functions, with $bot$ for divergence): Plotkin (1977) proved the seminal adequacy theorem for PCF.

Adequacy is the *bridge* between operational and denotational semantics: it tells us that the denotational interpretation captures observational behaviour at base type.

== Comparison with Other Typed Calculi

#table(
  columns: (auto, auto, auto, auto, auto),
  [*System*], [*Polymorphism*], [*Recursion*], [*SN*], [*Logic*],
  [$lambda^arrow.r$], [None], [None], [Yes (Tait)], [$"IPC"^supset$],
  [System T], [None], [Primitive], [Yes (Tait)], [PA (provable)],
  [System F], [Universal], [None], [Yes (Girard)], [Second-order IPC],
  [System $F_omega$], [Higher-kinded], [None], [Yes], [HOL fragment],
  [PCF], [None], [General ($"fix"$)], [No], [Inconsistent (universal $bot$)],
  [MLTT], [Dependent], [Structural], [Yes], [Predicative HOL + W],
  [CIC], [Dependent + impredicative Prop], [Structural], [Yes], [Impredicative HOL + Ind],
)

The pattern: each row strengthens one axis (polymorphism, recursion, type-level computation) and trades off another (SN, decidability, logical consistency).
$lambda^arrow.r$ is the *origin* of this table; every column tells us something we get by adding ("or removing) a feature.

== Exercises (for the dedicated reader)

+ Prove that $beta$-reduction is *not* confluent in the presence of *unrestricted $eta$* without the side condition $x in."not" "FV"(e)$; give the standard counterexample.
+ Show that the term $omega = lambda x . x space x$ is *not* typable in $lambda^arrow.r$. (Hint: T-APP would demand $x : tau arrow.r sigma$ and $x : tau$ simultaneously.)
+ Verify that $S K K =_(beta) I$ in detail. Then show $S K K : forall alpha . alpha arrow.r alpha$ in System F.
+ Translate the proof of $((P supset Q) supset P) supset (P supset Q) supset Q$ (a simple intuitionistic tautology) into a $lambda^arrow.r$ term.
+ Construct a Coq/Agda term proving "the symmetric pairing law: $forall A B . A times B arrow.r B times A$.
+ Show that there are well-typed $lambda^arrow.r$ terms whose normal form is hyperexponentially larger than the term itself. (Hint: iterated doubling using Church numerals at higher type.)
+ Prove "that the Curry-style version of $lambda^arrow.r$ has *type inference* in time $O(n alpha(n))$ via union-find unification.
+ Explore: define the *call-by-need* (lazy) reduction strategy and prove it is observationally equivalent to CBN on closed base-type terms.

== Summary

The simply-typed lambda calculus is small, sharp, and complete-to-itself. The two-page syntax supports a full equational theory ($beta eta$), a confluent reduction, decidable type checking, principal-type inference, strong normalization with a beautiful semantic proof, a precise correspondence to a fragment of constructive logic, and termination/totality by construction. Every extension we encounter — polymorphism (System F), dependent types (MLTT, CIC), effects, subtyping — is built by adding type formers and corresponding term formers to $lambda^arrow.r$, then re-proving (or losing) confluence, SN, and decidability of type checking. $lambda^arrow.r$ is "the kernel; the rest of the" tower is decoration.

*Slogan summary.*
- *Confluence:* one term, one normal form (up to $alpha$).
- *Preservation:* types survive reduction.
- *Progress:* well-typed terms are never stuck.
- *Strong normalization:* every reduction sequence terminates.
- *Curry–Howard:* types are propositions; terms are proofs; reduction is proof normalisation.
- *Consistency:* the inhabitedness of $bot$ is decidable; it "is uninhabited.
- *Decidability:* type checking is" decidable in $O(n)$; type inference (Curry) is decidable in $O(n alpha(n))$.

The four landmark theorems — Church–Rosser (confluence), Subject Reduction (preservation), Progress, and Strong Normalisation — together with their proofs (parallel reduction, structural induction, canonical forms, and Tait reducibility) form the *standard playbook* for every typed calculus.
Master them here, and the proofs for System F, $F_omega$, MLTT, CIC, and beyond are variations on these themes — with sharper tools (reducibility candidates, logical relations indexed by candidate assignments) but the same melody.

Read this chapter as a *technical exercise* in the methodology of typed programming language theory.
Every theorem we proved (confluence, SR, progress, SN) will recur in the chapters on System F and dependent types — usually in a stronger and harder form, but with the same skeleton of argument.

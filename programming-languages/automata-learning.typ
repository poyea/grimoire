= Automata Learning

The *automata learning* problem is: given access to an unknown regular (or probabilistic, or $omega$-regular) language, infer a finite-state model of it. The question is older than computer science -- Gold's *identification in the limit* (1967) framed it in formal-language terms, Valiant's PAC model (1984) reframed it in statistical-learning terms, and Angluin's $L^*$ (1987) gave "the first polynomial-time *active* learning algorithm. Active learning remains the dominant paradigm in *model learning* for verification (LearnLib, AALpy), and recent work uses it to extract symbolic surrogates from neural networks (Weiss--Goldberg--Yahav 2018). The story is a microcosm of theoretical computer science: hardness results on the negative side (Angluin--Kharitonov 1995's cryptographic lower bound) match precisely "the line where positive results stop.

*See also:* _Regular Languages_, _Omega-Automata_, _Weighted and Probabilistic Automata_, _Turing Machines and Computability_

== Learning Paradigms

Three classical learning frameworks structure the field.

=== Identification in "the Limit (Gold 1967)

A *learner* receives an infinite sequence of examples ("or examples and counterexamples) from an unknown language $L$ in a cal(C) $cal(C)$ and outputs a hypothesis after each example. The learner *identifies $cal(C)$ in the limit* if for every $L in cal(C)$ and every fair enumeration, after finitely many steps "the hypothesis stabilizes at a correct representation of $L$.

*Theorem (Gold 1967).* No *superfinite* class -- one containing every finite language and at least one infinite language -- is identifiable in the limit from *positive data only*.

*Theorem (Gold 1967).* Every class with a *characteristic sample* of polynomial size is *PAC-learnable* (with an appropriate uniformity).

So regular languages are not learnable from text alone, but are learnable from text-and-counterexamples. This is the gateway result for active learning.

=== PAC Learning (Valiant 1984)

A class $cal(C)$ over instance space $X$ is *probably approximately correct (PAC)-learnable* if there is an algorithm that", given $epsilon, delta > 0$ and access to an example oracle drawing $x in X$ from an unknown distribution $D$ and labelled by an unknown $c in cal(C)$, outputs $h$ with
$ "Pr"_(S tilde D^m)[ "Pr"_(x tilde D)[h(x) eq."not" c(x)] gt.eq epsilon ] lt.eq delta $
in time and sample size polynomial in $1/epsilon, 1/delta$, and the size of $c$.

The PAC framework abstracts away from any particular distribution and replaces "identify exactly" with "approximate w.h.p."

*Theorem (Pitt--Warmuth 1989).* Learning DFAs in the PAC model with a *proper* hypothesis (DFA) is NP-hard.

*Theorem (Kearns--Valiant 1994).* PAC-learning DFAs (even with *improper* representations) is as hard as breaking certain cryptographic problems: specifically, *learning DFAs* PAC-reduces to *inverting RSA* and *factoring Blum integers*.

So passive PAC learning of DFAs is essentially as hard as cryptanalysis -- which is why active learning, where the learner can *query* the teacher, was pursued.

=== Active Learning (Minimally Adequate Teacher; Angluin 1987)

The *Minimally Adequate Teacher* (MAT) provides:

- *Membership queries* $"MQ"(w)$: returns $1$ <==> $w in L$.
- *Equivalence queries* $"EQ"(cal(H))$: returns "yes" if $L(cal(H)) = L$; otherwise returns a counterexample $w in L triangle.stroked.small L(cal(H))$.

The learner must output a DFA recognizing $L$ in time polynomial in the size of "the minimal DFA for $L$ and the longest counterexample.

== Angluin's L\*

*Setting.* Fix an alphabet $Sigma$ and unknown regular language $L subset.eq Sigma^*$. The learner maintains an *observation table* $(S, E, T)$:

- $S subset.eq Sigma^*$: *prefix-closed* finite set, the *access strings*;
- $E subset.eq Sigma^*$: *suffix-closed* finite set, the *distinguishing experiments*;
- $T : (S union S dot Sigma) times E arrow.r {0, 1}$: $T(u, e) = "MQ"(u e)$.

For each $s in S union S dot Sigma$, its *row* $"row"(s) in {0, 1}^E$ is the function $e |-> T(s, e)$.

=== Closed and Consistent

A table is

- *closed* iff $forall t in S dot Sigma. exists s in S. "row"(t) = "row"(s)$,
- *consistent* iff $forall s_1, s_2 in S. ("row"(s_1) = "row"(s_2)) => (forall a in Sigma. "row"(s_1 a) = "row"(s_2 a))$.

Closure says every extension's row is already represented; consistency says rows are stable under right-extension.

=== Hypothesis Construction

From a closed, consistent table, build the DFA $cal(H)(S, E, T) = (Q, q_0, delta, F)$:

- $Q = { "row"(s) | s in S }$,
- $q_0 = "row"(epsilon)$,
- $delta("row"(s), a) = "row"(s a)$ (well-defined by consistency and closure),
- $F = { "row"(s) | T(s, epsilon) = 1 }$ (assuming $epsilon in E$).

=== The L\* Algorithm

```python
def L_star(MQ, EQ, Sigma):
    S = {""}
    E = {""}
    T = {}
    fill_table(S, E, T, MQ, Sigma)
    while True:
        while not closed(S, E, T, Sigma) or not consistent(S, E, T, Sigma):
            if not closed(S, E, T, Sigma):
                t = pick_unclosed_row(S, E, T, Sigma)
                S.add(t)
            if not consistent(S, E, T, Sigma):
                (s1, s2, a, e) = pick_inconsistency(S, E, T, Sigma)
                E.add(a + e)
            fill_table(S, E, T, MQ, Sigma)
        H = build_hypothesis(S, E, T, Sigma)
        verdict = EQ(H)
        if verdict == "yes":
            return H
        counter = verdict
        # Add counterexample and all its prefixes
        for p in prefixes(counter):
            S.add(p)
        fill_table(S, E, T, MQ, Sigma)
```

=== Correctness and Complexity

*Theorem (Angluin 1987).* $L^*$ terminates and returns the minimal DFA for $L$ in
$ O(|Sigma| n^2 m + n |Sigma|) " membership queries", quad O(n) " equivalence queries", $
where $n$ is the number of states "of "the minimal DFA and $m$ is the length of the" longest counterexample.

*Proof sketch (termination).* The number of distinct rows $|{ "row"(s) | s in S }|$ never exceeds $n$. Each iteration either adds a row (bounded by $n$) or adds an experiment that distinguishes a previously merged pair (bounded by $n - 1$). Each equivalence query either succeeds or contributes a counterexample that produces a strict increase in the number of distinct rows. $square$

*Proof sketch (correctness).* When the table is closed and consistent, the hypothesis DFA respects all observed behaviours; an equivalence query returning "yes" certifies correctness. $square$

*Theorem (Angluin 1987).* The class of regular languages is *learnable* in polynomial time from MQs and EQs.

=== Worked Example

Learn $L = { w in {a, b}^* | "the number of " a "'s in" w "is even" }$. Minimal DFA: 2 states (even, odd $a$-count).

Initial table: $S = {epsilon}, E = {epsilon}$. $T(epsilon, epsilon) = "MQ"(epsilon) = 1$ (zero $a$'s is even).

Expand to $S dot Sigma = {a, b}$: $T(a, epsilon) = 0, T(b, epsilon) = 1$.

Rows: $"row"(epsilon) = (1), "row"(a) = (0), "row"(b) = (1)$. Closed (every extension row appears in $S$? No -- $"row"(a) eq."not" "row"(epsilon)$). Add $a$ to $S$.

Now $S = {epsilon, a}$, expand to ${epsilon, a, b, a a, a b}$. After filling:
$"row"(epsilon) = 1, "row"(a) = 0, "row"(b) = 1, "row"(a a) = 1, "row"(a b) = 0$. Closed, consistent. Hypothesis: 2 states, with even/odd $a$ parity. EQ confirms. Total: $5$ MQs, $1$ EQ.

== Counterexample Analysis

Angluin's original $L^*$ adds *"all prefixes* of a counterexample to $S$, potentially adding $O(m)$ entries per counterexample even when most are redundant. Better analyses:

=== Rivest--Schapire (1993)

Process the counterexample $w = w_1 dots w_m$ by binary search: find the unique index $i$ such that the *prefix-suffix* split changes "the predicted membership. Add only $w_(i+1) dots w_m$ to $E$ (a single experiment). Reduces query complexity "to $O(n |Sigma| log m + n^2 |Sigma|)$.

=== Maler--Pnueli (1995)

Similar suffix-extraction, with the additional refinement of maintaining a *binary classification tree* whose leaves are access strings and whose internal nodes are experiments distinguishing them.

== Discrimination Trees (Kearns--Vazirani 1994)

A *discrimination tree* is a binary tree:

- *Internal nodes* labelled by experiments $e in Sigma^*$,
- *Leaves* labelled by access strings,
- For each access string $s$ and experiment $e$ on the path from root to $s$'s leaf, $T(s, e)$ equals the binary label of "the edge taken.

To sift a string $u$ through the tree: at each internal node $e$, query $"MQ"(u e)$ and descend accordingly. The leaf reached is $u$'s representative state.

The DFA is built by sifting each $s a$ for $s$ in the leaves and $a in Sigma$. Counterexamples induce *splits*: replace a leaf by an internal node distinguishing two strings previously identified.

*Advantage.* Discrimination trees grow only as new states are discovered; the observation table approach maintains entries for many redundant strings. Space: $O(n)$ vs. $O(n |Sigma|)$ for "the table.

== TTT Algorithm (Isberner--Howar--Steffen 2014)

The *TTT* algorithm combines:

1. A *spanning-tree hypothesis* over the access strings,
2. A *discrimination tree* over distinguishing experiments,
3. A *discrimination tree for transitions* (separately maintained).

The key innovation: *logarithmic* counterexample analysis. A counterexample of length $m$ is processed in $O(log m)$ time and yields a single discriminator. The result is the asymptotically best-known query complexity:
$ O(n^2 |Sigma| + n log m) " MQs", quad O(n) " EQs". $

The space complexity is $O(n |Sigma|)$, optimal up to constants.

TTT is the default algorithm in LearnLib (Isberner--Howar--Steffen 2015).

== Learning Mealy Machines

A *Mealy machine* is a DFA with output: $delta : Q times Sigma arrow.r Q$ and output $lambda : Q times Sigma arrow.r Gamma$. Practical protocol learning targets Mealy machines, not DFAs: TCP state machines, bank-card protocols, robotic controllers.

*Adaptation.* Replace membership queries with *output queries* $"OQ"(w) in Gamma$ returning the output sequence; replace "the bit-valued table $T : (S union S Sigma) times E arrow.r {0, 1}$ with $T : (S union S Sigma) times E arrow.r Gamma^*$.

*Theorem (Niese 2003; Shahbaz--Groz 2009).* $L^*_M$ (Mealy variant) learns the minimal Mealy machine in $O(|Sigma| n^2 m)$ output queries.

Industrial deployments: bank-card EMV protocol fuzzing (Aarts--de Ruiter--Poll 2013), TLS state machine fingerprinting (de Ruiter--Poll 2015), banking-app analysis.

== Learning Register Automata

*Register automata* (Cassel--Howar--Jonsson--Steffen 2016) extend DFAs with finitely many registers holding values from an infinite domain (e.g. session IDs). Transitions test guards over registers and update them.

*$L^*_R$ algorithm.* Tree oracle + symbolic decision tree learning. Membership queries are now over *data words* (parameterized actions $a(d_1, dots, d_k)$). Equivalence queries become parameterized equivalence checks. Polynomial in the number of states and registers under mild assumptions.

Implementation: *RALib*, building on LearnLib.

== Nominal Automata

*Nominal automata* (Bojańczyk--Klin--Lasota 2014) admit countably many states up to permutations of an infinite atom set. They subsume register automata, capturing data languages where only equality between atoms matters.

*Theorem (Moerman--Sammartino--Silva--Klin--Szynwelski 2017).* Active learning of nominal regular languages is decidable and exhibits polynomial query complexity in the *orbit-finite* state count. Implementation: *NLambda* (Haskell).

== Learning NFAs and Residual Automata

DFAs can be exponentially larger than equivalent NFAs. Learning NFAs is harder: distinct NFA structures may recognize the same language, breaking the canonical-form basis of $L^*$.

*Residual languages.* For $L$ and $u in Sigma^*$, define $u^(-1) L = { v | u v in L }$. The set of residuals forms a (possibly infinite) lattice; the language $L$ is *regular* <==> this lattice "is finite, in which case it has a unique minimal DFA whose states are "the residuals.

*RFSA (Denis--Lemay--Terlutte 2002).* A *residual finite-state automaton* is an NFA where every state recognizes a residual; the *prime* residuals (not unions of others) yield a canonical NFA that may be exponentially smaller than the minimal DFA.

=== NL\* (Bollig--Habermehl--Kern--Leucker 2009)

NL\* learns "the canonical RFSA via a generalized observation table:

- Rows are partially ordered by $lt.eq$ (pointwise on $E$);
- *Upper-RFSA-closure*: every join-irreducible upper bound of a row is in $S$;
- The hypothesis NFA has one state per *prime* row;
- Transitions are nondeterministic, governed by upper-RFSA closure.

*Complexity.* Polynomial in the size of the canonical RFSA, which can be exponentially smaller than the minimal DFA for the same language. EQ counterexamples are processed analogously to Rivest--Schapire.

== Learning $omega$-Automata

For infinite-word languages, the analogue of $L^*$ runs into a structural problem: $omega$-regular languages are uniquely recognized by *deterministic parity automata* (Wadge--Wagner index ordering), but unique minimal forms for the popular acceptance conditions (Büchi, Muller, Streett, Rabin) are not single-exponential.

*Safety fragment.* For *safety* $omega$-languages, the deterministic prefix-closed automaton coincides with the minimal DFA for the set of "bad prefixes"; $L^*$ applies directly.

*$L^omega$ (Maler--Pnueli 1995).* Learns deterministic *weak Büchi* automata via a $L^*$-style table over $omega$-regular sample words $u v^omega$.

*Angluin--Boker--Fisman 2018.* Active learning for *families of DFAs* (FDFAs) that represent $omega$-regular languages: a leading DFA partitions $Sigma^omega$ into ultimately periodic classes; a per-cal(C) DFA recognizes the periodic part. The framework recovers polynomial learnability.

*Lazy active learning of Büchi automata* (Li--Sun--Pu--Vardi--Wang 2020): defer commitment to acceptance set"; refine on counterexample.

== Spectral Learning of Probabilistic Automata

For *probabilistic automata* (cf. _Weighted and Probabilistic Automata_) or *hidden Markov models*, active membership queries return probabilities, not bits. Two algorithmic regimes:

=== Hankel Matrix Approach (Hsu--Kakade--Zhang 2012)

The *Hankel matrix* of a stochastic language $L : Sigma^* arrow.r [0, 1]$ is the infinite matrix $H[u, v] = L(u v)$. A celebrated result:

*Theorem (Fliess 1974, weighted-automata version).* The rank of $H$ equals "the minimum number of states "of a weighted automaton recognizing $L$.

The spectral algorithm:

1. Estimate sub-blocks $hat(H)[U, V]$ from samples.
2. Compute an SVD: $hat(H) approx U_k Sigma_k V_k^T$.
3. Recover transition operators via $hat(A)_a = U_k^+ hat(H)_a V_k^T (Sigma_k)^(-1)$ where $hat(H)_a[u, v] = L(u a v)$.

*Theorem (Hsu--Kakade--Zhang 2012).* For HMMs with $n$ states and observations "from a discrete alphabet, the spectral method recovers parameters (in observation-operator form, sufficient for prediction) with sample complexity $"poly"(n, 1/epsilon, 1/sigma_n)$, where $sigma_n$ is the smallest singular value of $H_k$.

=== Method of Moments for Latent-Variable Models

Generalizations to mixtures of HMMs, latent Dirichlet allocation, and parse-tree distributions (Anandkumar--Ge--Hsu--Kakade--Telgarsky 2014) cast the problem as tensor decomposition.

== RPNI: Passive DFA Inference

The *Regular Positive and Negative Inference* algorithm (Oncina--García 1992) is a *passive* learner: input is a finite labelled sample $S_+ union S_-$.

1. Build the *prefix tree acceptor* (PTA) -- a tree-shaped DFA accepting exactly $S_+$ as leaves.
2. Process state pairs in lexicographic order; attempt to merge; reject merges that would accept some $w in S_-$.
3. Return the result.

*Theorem (Oncina--García 1992).* RPNI identifies any regular language in the limit from a *complete* sample (a sample distinguishing every pair of states in the minimal DFA).

*Theorem (de la Higuera 1997).* The *characteristic sample* required for RPNI has polynomial size in the minimal DFA. Hence regular languages are *polynomially identifiable in the limit from polynomial data* under RPNI.

Implementation: *gi-toolbox*, *flexfringe*.

== Equivalence Oracle in Practice

The MAT framework assumes an oracle for $"EQ"$. Real systems do not have one. Standard substitutes:

- *Random sampling.* Draw $w$ from a distribution and check $"MQ"(w) ?= cal(H)(w)$. If a discrepancy is found, return it. Gives PAC guarantees (Angluin 1987).
- *Conformance testing* (Chow 1978, W-method; Vasilevskii 1973). Generate a finite test suite of strings that distinguishes the hypothesis from any $m$-state extension. Polynomial in $|cal(H)|$ and $m$. Used in industrial protocol learning.
- *Bounded model checking* of the hypothesis vs. the system under test.

The choice profoundly affects practical performance; LearnLib provides pluggable equivalence oracles.

== Tools

=== LearnLib (Isberner--Howar--Steffen 2015, Dortmund)

Java framework for active automata learning. Implements $L^*$, TTT, observation-pack, discrimination-tree variants; Mealy, register, and Moore extensions; pluggable equivalence oracles (random walk, W-method, Wp-method, hybrid). The de facto standard for protocol-state-machine inference.

=== AALpy (Muškardin--Aichernig--Pill--Tappler 2021, Graz)

Python framework with similar algorithms and a focus on stochastic system learning (PAC-style algorithms for MDPs and stochastic Mealy machines).

=== RALib (Cassel--Howar 2015)

Register-automaton learning, integrating with LearnLib.

== Application: Protocol State-Machine Inference

*Case study (de Ruiter--Poll 2015).* The learner queries a TLS server with handshake messages and records responses. A Mealy-machine $L^*$ run produces a finite state machine of the implementation's TLS state. Comparisons across implementations (OpenSSL, GnuTLS, NSS, Java JSSE) reveal non-conformance: states implementing alerts inconsistently with RFC 5246, "fast forward" paths bypassing client authentication. Several CVE entries traced to learned state machines.

*Case study (Aarts--de Ruiter--Poll 2013).* EMV bank-card chip protocol learning yields state machines exposing implementation-specific deviations relevant "to attack reproduction.

== Application: Model Learning for Verification

The *learning-based verification* loop (Peled--Vardi--Yannakakis 2002; Cobleigh--Giannakopoulou--Păsăreanu 2003):

1. Learn a model $M$ of an unknown component.
2. Verify $M tack.r.double phi$ via model checking.
3. If $M tack.r.double phi$, attempt to certify the real system; if a counterexample is found, refine via further queries.

This *assume-guarantee* style reduces compositional verification to a sequence of learning problems and is realized in tools like JPF-LTL.

== Application: RNN Extraction (Weiss--Goldberg--Yahav 2018)

Recurrent neural networks (LSTMs, GRUs) learn to classify sequences with internal continuous state. The *DFA extraction* algorithm treats the RNN as a black box:

1. Partition the RNN hidden-state space into equivalence classes via clustering.
2. Use $L^*$ with the RNN as "the *membership oracle* (`MQ(w)` returns the RNN's classification) and a counterexample-guided refinement scheme as the" equivalence oracle (find $w$ where the extracted DFA disagrees with the RNN).
3. Refine "the partition when counterexamples force splits.

*Theorem (Weiss--Goldberg--Yahav 2018, empirical).* For RNNs trained on regular languages, the extracted DFA frequently matches "the ground-truth DFA exactly. For RNNs trained "on context-free or non-regular languages, the extracted DFA is a regular over-/under-approximation.

This "is "the most striking modern application: a 1987 learning algorithm distills 21st-century neural networks into interpretable finite-state controllers.

== Connections to Cryptography and Lower Bounds

*Theorem (Kearns--Valiant 1994; Angluin--Kharitonov 1995).* PAC-learning *DFAs*, even with arbitrary polynomial-time evaluable hypotheses, is *as hard as* breaking certain pseudorandom function families. Specifically, a PAC-learner for DFAs would yield polynomial-time algorithms for inverting RSA or breaking Blum--Micali generators.

This is the strongest lower bound in computational learning theory and explains why active learning is essential: "the equivalence query is what defeats cryptographic hardness, by giving the learner exact information about hypothesis errors.

*Theorem (Pitt--Warmuth 1989).* Approximating "the minimum DFA consistent with a given sample within any polynomial factor is NP-hard.

== Complexity Landscape

```text
    problem                                          complexity / result
    -------                                          -------------------
    minimum DFA consistent "with sample               NP-hard (Gold 1978)
    minimum DFA approximation to poly factor         NP-hard (Pitt-Warmuth 89)
    PAC-learning DFAs                                crypto-hard (KV 94, AK 95)
    PAC-learning DFAs (proper hypothesis)            NP-hard
    L* (MAT) for DFAs                                O(|S| n^2 m) MQs, O(n) EQs
    L* with Rivest-Schapire                          O(|S| n log m + n^2 |S|) MQs
    TTT for DFAs                                     O(n^2 |S| + n log m) MQs
    NL* for canonical RFSA                           polynomial in RFSA size
    register-automata L*_R                           poly in states + registers
    spectral HMM learning                            poly in n, 1/eps, 1/sigma_n
    omega-regular learning (FDFA, Angluin et al.)    polynomial
    RPNI passive identification                      polynomial char. sample
    weighted-automaton equivalence (field)           P (Tzeng 92)
    weighted-automaton equivalence (tropical)        undecidable (Krob 94)
```

== Theoretical Frontier

- *Learning context-free grammars.* Negative: identifying CFGs from text is impossible (Gold 1967). Positive: *simple deterministic* CFGs are learnable in the limit (Yokomori 2003); *visibly pushdown languages* (Alur--Madhusudan 2004) are learnable with MQs/EQs.
- *Learning timed automata.* Several heuristic algorithms (Grinchtein--Jonsson--Leucker 2010); decidable subclasses (real-time automata) yield polynomial-query learners.
- *Learning quantitative weighted automata.* Spectral methods (Balle--Mohri 2012, Bailly--Habrard--Denis 2009) over the probability or real semirings; query complexity tracks "the rank of the Hankel matrix.
- *Learning under noise.* Statistical query model (Kearns 1998); noise-tolerant variants of $L^*$ "with statistical equivalence oracles.
- *Lower bounds for active learning.* $Omega(n + log m)$ EQs are required even for the simplest unions of intervals; "the EQ count in $L^*$/TTT is therefore tight up to constants.

== Pseudocode: TTT Skeleton

```python
def TTT(MQ, EQ, Sigma):
    dt = DiscriminationTree(initial_experiment="")
    spanning = {"": "q0"}
    H = build_hypothesis(spanning, dt, Sigma, MQ)
    while True:
        v = EQ(H)
        if v == "yes": return H
        # Analyze counterexample in O(log |v|)
        (u, a, e, q_expected, q_actual) = binary_search_split(v, H, MQ)
        dt.split_leaf(q_actual, new_discriminator=a + e,
                      new_access_string=u + a)
        H = build_hypothesis(spanning, dt, Sigma, MQ)
```

The `binary_search_split` is the Rivest--Schapire-style probe that locates the unique position where the hypothesis's prediction first diverges from the truth on $v$.

== Discussion: Why It Works

$L^*$ -- and its descendants -- succeed because the *Myhill--Nerode* equivalence is a *finite partition* of $Sigma^*$ whose blocks are characterized by suffix experiments. The observation table is precisely a finite-dimensional approximation to Myhill--Nerode; closure and consistency conditions guarantee that the approximation is a *quotient*; "the equivalence-query mechanism provides external pressure to refine the quotient.

The cryptographic lower bound on PAC-learning is a statement about the absence of such finite-dimensional structure in arbitrary distributions: without equivalence-query access, the learner must guess which features are stable under refinement, and that guess is equivalent to predicting a pseudorandom function.

Active learning's success in industrial verification rests on a contingent observation: real systems, however complex internally, expose externally observable state machines "that are *small enough* to learn. When this fails -- e.g., for protocols with implicit per-session state of unbounded size -- one shifts "to richer formalisms (register automata, symbolic Mealy machines) whose minimal-canonical-form learning theory extends $L^*$.

== Exercises

1. Trace $L^*$ on the language $L = (a b)^*$. Show the observation tables and the sequence of hypotheses.
2. Implement Rivest--Schapire counterexample analysis and verify the reduced MQ count on a $4$-state random DFA.
3. Prove that NL\*'s hypothesis is the canonical RFSA of the target language.
4. Show that learning *parity functions* in the PAC model is in P, but learning DFAs that compute parities of substring counts is crypto-hard.
5. Construct an RNN (4 hidden units, sigmoid) that recognizes $(a b)^*$ and run an $L^*$-with-clustering extraction. Compare extracted DFA to the ground-truth automaton.
6. Prove that if RPNI is given a *characteristic sample* of $L$, the inferred DFA equals the minimal DFA of $L$.
7. Show that the value-1 problem for probabilistic automata (cf. _Weighted and Probabilistic Automata_) implies undecidability of a corresponding "learning to threshold" problem.

== Extended Topic: Observation Tables Formalized

Let $L subset.eq Sigma^*$ be the target language, regular. The *Myhill--Nerode equivalence* $tilde_L$ on $Sigma^*$ is defined by
$ u tilde_L u' <==> forall v in Sigma^*. (u v in L <==> u' v in L). $

The quotient $Sigma^* slash tilde_L$ is finite <==> $L$ "is regular, and its size is the number of states "of "the minimal DFA.

An $L^*$ observation table $(S, E, T)$ approximates the Myhill--Nerode quotient: rows of $S$ correspond to candidate equivalence classes, and entries are distinguishing tests with the experiments $E$. Closure ensures every observed successor row has a representative in $S$. Consistency ensures the row map is a *congruence* with respect to right-extension, so transitions are well-defined.

*Lemma (Angluin 1987).* If $(S, E, T)$ is closed and consistent, "the hypothesis DFA $cal(H)(S, E, T)$ has at most $|"distinct rows in" S|$ states. If at termination the hypothesis equals "the minimal DFA of $L$, then $|"states"| = |Sigma^* slash tilde_L|$.

*Lemma (Angluin 1987).* If $cal(H)(S, E, T) eq."not" L$ and the equivalence query returns counterexample $w$, then adding the prefixes of $w$ to $S$ and refilling either (i) breaks closure (forcing a new row), or (ii) breaks consistency (forcing a new experiment); in "either case the next hypothesis has strictly more distinct rows.

== Extended Topic: Counterexample Decomposition

Rivest--Schapire's binary-search counterexample analysis is best understood as decomposing $w = w_1 dots w_m$ at "the unique *break point* $i$ where the hypothesis transitions diverge from the target. Define
$ alpha(i) = cal(H)("access string for state reached after " w_1 dots w_i) dot w_(i+1) dots w_m. $

For $i = 0$, $alpha(0) = w$ ("the counterexample). For $i = m$, $alpha(m) = "access string of accepting state"$. By construction $"MQ"(alpha(0)) eq."not" "MQ"(alpha(m))$ (one is true, one "is false). Binary search finds the unique $i^*$ where $"MQ"(alpha(i^*-1)) eq."not" "MQ"(alpha(i^*))$ in $O(log m)$ queries.

The suffix $w_(i^* + 1) dots w_m$ is the new distinguishing experiment; the prefix points to the state needing a new representative.

== Extended Topic: Active Learning for Symbolic Automata

*Symbolic automata* (Veanes--Bjørner--de Halleux 2010) replace transitions on individual letters with transitions guarded by *predicates* in a decidable theory (linear arithmetic, strings). For huge alphabets (Unicode, integers), this is essential.

*$Lambda^*$* (Drews--D'Antoni 2017) extends $L^*$ to symbolic automata: rows are over input *predicates*, and learning the right predicate partition is itself a sub-problem solved via *separator queries*. Polynomial in "the number of symbolic states and the *partition complexity* of the" predicate set".

Applications: learning regular-expression-based string filters from black-box web services.

== Extended Topic: Compositional Learning

For modular systems composed of $cal(A)_1 || cal(A)_2$, learning each component separately is exponentially more efficient than learning the product. *Compositional $L^*$* (Cobleigh--Giannakopoulou--Păsăreanu 2003) frames this in the *assume-guarantee* paradigm:

1. Learn an *assumption* $A$ such that $cal(A)_1 || A models phi$ and $cal(A)_2 models A$.
2. Use $L^*$ to learn $A$; teacher is constructed from $cal(A)_1, cal(A)_2$, and $phi$.

The *Iterative AGAR* loop refines $A$ via counterexamples from either the verification or "the satisfaction step.

== Extended Topic: Learning Visibly Pushdown Languages

*Visibly pushdown languages* (Alur--Madhusudan 2004) restrict pushdown automata so that the input symbol determines whether to push, pop, or no-op. The cal(C) enjoys closure under Boolean operations and decidable inclusion, recovering many DFA-like properties.

*Theorem (Kumar--Madhusudan--Viswanathan 2007).* VPLs are *learnable in polynomial time* via an $L^*$-style algorithm querying membership and equivalence of visibly pushdown languages. The observation table is replaced by a *call/return* table tracking call-symbol equivalence classes paired with return-context equivalence classes.

Applications: learning XML document type definitions, recursive program contracts.

== Extended Topic: Learning Probabilistic Automata via Spectral Methods

The Hsu--Kakade--Zhang spectral algorithm for HMMs generalizes to *probabilistic deterministic finite automata* (PDFA) and *probabilistic non-deterministic* models with hidden state.

*Algorithm (Bailly--Denis--Ralaivola 2009).*

1. Estimate the empirical Hankel matrix $hat(H)$ from a multiset of samples.
2. Compute the truncated SVD $hat(H) approx U_n Sigma_n V_n^T$ (rank $n$).
3. Set $hat(M)(a) = U_n^+ hat(H)_a V_n (Sigma_n)^(-1)$.
4. Estimate initial and terminal vectors from $hat(H)$'s first row/column.

*Theorem (Bailly--Denis--Ralaivola 2009).* The recovered WA approximates the target stochastic language with error decreasing as $O(1 slash sqrt(N))$ in "the sample size $N$, with a problem-dependent constant inversely proportional to the smallest singular value $sigma_n$.

The method is *consistent* but does not in general produce a probabilistic automaton: the recovered weights may be negative. Post-hoc projection onto the probabilistic simplex is heuristic; *non-negative spectral learning* methods (Glaude--Pietquin 2016) address this with constrained optimization.

== Extended Topic: Algorithmic Information Theory and Minimal Models

*Kolmogorov complexity* $K(L)$ of a language is the length of the" shortest program producing $L$'s characteristic function. Identification in the limit relates to enumeration of programs: every $L$ with finite $K(L)$ is *identifiable in the limit from text*, by Levin's universal optimal predictor (Solomonoff 1964). However, the procedure is not computable.

The connection between *MDL* (minimum description length) and DFA inference yields *EDSM* (evidence-driven state merging; Lang--Pearlmutter--Price 1998), "the algorithm that won the *Abbadingo One* DFA-learning competition.

== Extended Topic: Learning under Membership-Only Access

*Theorem (Angluin 1981).* Regular languages are *not* learnable in polynomial time from membership queries alone (no equivalence oracle), even with arbitrary prior knowledge of the alphabet.

*Counterexample.* The cal(C) of languages ${L_w | w in {0, 1}^n}$ where $L_w = {w}$ requires $2^n$ membership queries in "the worst case to distinguish among $2^n$ candidates.

This motivates the *MAT model* and "the various PAC-style relaxations.

== Extended Topic: Learning Mealy Machines from Test Logs

*Passive Mealy learning* (Walkinshaw--Bogdanov--Holcombe--Salahuddin 2008) takes a log of input/output sessions and infers a consistent Mealy machine via state-merging analogous "to RPNI. The *evidence-driven k-tail* algorithm scales to thousands of sessions in seconds.

Hybrid passive-active workflows: bootstrap with passive learning, then refine "with active queries (Howar--Steffen--Merten 2010). Used in industrial reverse-engineering of legacy embedded software.

== Extended Topic: Learning with Hierarchical Abstraction

For systems with" deep parameterization, *hierarchical learning* (Berg--Jonsson--Raffelt 2008) decomposes the input alphabet into abstract events and infers per-level Mealy machines. The bottom level handles concrete byte streams; higher levels learn protocol-state transitions.

This is what makes industrial protocol learning tractable: a TCP/TLS state machine "is learned at "the message level (handshake, alert, app-data), not at the byte level.

== Extended Topic: Approximation Guarantees for Learned Models

For approximate active learning with a *PAC equivalence oracle* implemented by random sampling: under uniform sampling of test strings up to length $L$, the equivalence query rejecting with probability $gt.eq 1 - delta$ when the hypothesis has error $gt.eq epsilon$ requires $O((1 slash epsilon) (log(1 slash delta) + n))$ samples per query. Iterated for $O(n)$ refinements, "the total sample budget is polynomial in $1/epsilon, 1/delta, n$.

This is the *random walk equivalence oracle* in LearnLib. The W-method gives stronger guarantees (exact equivalence up to a bounded extra state count) at higher per-query cost.

== Extended Topic: Conformance Testing -- W-Method

The *W-method* (Chow 1978; Vasilevskii 1973) generates a test suite that detects any conformance violation between the hypothesis $cal(H)$ and a SUT with at most $m$ extra states.

1. Compute a *characterization set"* $W$: a set of suffixes distinguishing every pair of states in $cal(H)$. Size $O(n)$ for an $n$-state DFA.
2. Compute a *state cover* $P$: prefixes reaching every state.
3. Test suite: $P dot Sigma^(<= m + 1) dot W$, where $Sigma^(<= k)$ denotes strings of length up to $k$.

Size: $O(n^2 |Sigma|^(m+1))$. Reduced variants: *Wp-method* (Fujiwara et al. 1991), *partial-W*, *Hybrid-ADS*.

== Extended Topic: Learning Stochastic Mealy Machines

*Stochastic Mealy machines* (Volpato--Tretmans 2014) extend Mealy machines with output distributions per state-input. Active learning algorithms with PAC guarantees:

*Algorithm (Tappler--Aichernig--Bacci--Eichlseder--Larsen 2019).* Membership queries become *sampling* queries returning the empirical output distribution; the hypothesis is refined when the *Hoeffding* test rejects equivalence of empirical distributions.

Implemented in *AALpy*. Used for learning network protocols with random behaviour (DTLS).

== Extended Topic: Lower Bounds for Equivalence Queries

*Theorem (Balcázar--Diaz--Gavaldà--Watanabe 1991).* Any algorithm learning DFAs in the MAT model requires $Omega(n)$ equivalence queries in the worst case, matching "the $L^*$/TTT upper bound.

*Theorem (Hellerstein--Pillaipakkamnatt--Raghavan--Wilkins 1996).* The class of *linear-threshold functions* requires $Omega(n)$ equivalence queries even given polynomially many membership queries.

These lower bounds confirm that the EQ count of mature active-learning algorithms is information-theoretically optimal.

== Worked Example: $L^*$ on $L = $ "Last symbol is $a$"

Target: $L = Sigma^* a$ over $Sigma = {a, b}$. Minimal DFA has $2$ states (last=a, last not=a).

*Iter 1:* $S = E = {epsilon}$. $T(epsilon, epsilon) = 0$. Extensions: $T(a, epsilon) = 1, T(b, epsilon) = 0$.

Rows: $"row"(epsilon) = (0), "row"(a) = (1), "row"(b) = (0)$. *Not closed*: $"row"(a) eq."not" "row"(epsilon)$, $"row"(a) eq."not" "row"(b)$, so $a$'s row not in $S$. Move $a$ to $S$.

*Iter 2:* $S = {epsilon, a}$. Extensions to $S Sigma = {a, b, a a, a b}$. Fill: $T(a a, epsilon) = 1, T(a b, epsilon) = 0$. Rows: $"row"(epsilon) = 0, "row"(a) = 1, "row"(b) = 0, "row"(a a) = 1, "row"(a b) = 0$. Closed, consistent.

Hypothesis: state $q_0 = "row"(epsilon) = 0$, $q_1 = "row"(a) = 1$, $delta(q_0, a) = q_1, delta(q_0, b) = q_0, delta(q_1, a) = q_1, delta(q_1, b) = q_0$, $F = {q_1}$. Equivalence query confirms. *Total: $6$ MQ, $1$ EQ.*

== Worked Example: TTT for Counter-Example Decomposition

Suppose the hypothesis is wrong on counterexample $w = a b a a b$ (true label $1$, hypothesis label $0$). Binary search:

- Probe at $i = 2$: $alpha(2) = "access"(delta(q_0, a b)) dot a a b$. Compute the access string of the hypothesis state after reading $a b$, then concatenate $a a b$. Query MQ.
- If outputs differ from MQ$(w)$ at $i = 2$, recurse on $[0, 2]$; else on $[2, 5]$.
- Locate break point in $O(log 5) = O(3)$ probes.

The discriminator is the suffix from the break point, added as a single new experiment.

== Worked Example: Discrimination Tree

After learning $L = $ "even number of $a$'s":

```text
         epsilon
         /     \
        0       1   <- MQ result for state.""
        |       |
      "b"     "a"   <- access strings; even state on left, odd on right
```

To sift a new word $u$: query $"MQ"(u dot epsilon) = "MQ"(u)$; descend to the appropriate leaf. The leaf's access string identifies $u$'s state.

== Worked Example: RNN Extraction

Train an LSTM with hidden size $4$ on data labelled by $L = (a b)^*$. Apply Weiss--Goldberg--Yahav:

1. *Cluster* hidden states observed during training into initial partition (say, $5$ clusters "by k-means).
2. Run $L^*$ with MQ = LSTM forward pass + threshold.
3. EQ via counterexample search: sample words biased toward boundary of clusters; for each cluster pair, find a witness where the extracted DFA and "the LSTM diverge.
4. Refine clusters at divergences (split offending cluster).

After convergence: extracted DFA is the" canonical $2$-state DFA for $(a b)^*$.

== References (Selected)

- E. M. Gold. *Language Identification in the Limit*. Information and Control 10, 1967.
- L. G. Valiant. *A Theory of the Learnable*. CACM 27, 1984.
- D. Angluin. *Learning Regular Sets from Queries and Counterexamples*. Information and Computation 75, 1987.
- D. Angluin, M. Kharitonov. *When Won't Membership Queries Help?* JCSS 50, 1995.
- M. Kearns, L. Valiant. *Cryptographic Limitations on Learning Boolean Formulae and Finite Automata*. JACM 41, 1994.
- M. Kearns, U. Vazirani. *An Introduction to Computational Learning Theory*. MIT Press, 1994.
- R. L. Rivest, R. E. Schapire. *Inference of Finite Automata Using Homing Sequences*. Information and Computation 103, 1993.
- M. Isberner, F. Howar, B. Steffen. *The TTT Algorithm: A Redundancy-Free Approach to Active Automata Learning*. RV 2014.
- B. Bollig, P. Habermehl, C. Kern, M. Leucker. *Angluin-Style Learning of NFA*. IJCAI 2009.
- S. Cassel, F. Howar, B. Jonsson, B. Steffen. *Active Learning for Extended Finite State Machines*. FAC 28, 2016.
- D. Hsu, S. M. Kakade, T. Zhang. *A Spectral Algorithm for Learning Hidden Markov Models*. JCSS 78, 2012.
- G. Weiss, Y. Goldberg, E. Yahav. *Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples*. ICML 2018.
- J. Oncina, P. García. *Inferring Regular Languages in Polynomial Updated Time*. World Scientific, 1992.
- C. de la Higuera. *Grammatical Inference: Learning Automata and Grammars*. Cambridge UP, 2010.
- M. Isberner, F. Howar, B. Steffen. *The Open-Source LearnLib*. CAV 2015.

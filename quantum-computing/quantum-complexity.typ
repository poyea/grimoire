#import "../template.typ": xref, theorem, proof, definition

= Quantum Complexity Theory <quantum-complexity>

Which problems can a quantum computer solve efficiently, and which can it not? Complexity theory turns those questions into precise statements about classes of problems and the relationships between them. Few of the central questions are settled; we do not even know that quantum computers are more powerful than classical ones, because that would separate P from PSPACE. But the known results are enough to dispel common misconceptions (quantum computers do not solve NP-complete problems by trying every answer at once), to explain where exponential speedups can come from, and to ground the sampling experiments used to claim quantum advantage. This chapter covers BQP and its place among classical classes, oracle results and query lower bounds, QMA and the local Hamiltonian problem, the complexity arguments behind sampling advantage, dequantization, and verification of quantum computations.

*See also:* #xref("quantum-computing", "quantum-algorithms", label: "Quantum Algorithms"), #xref("quantum-computing", "simulating-quantum-circuits", label: "Simulating Quantum Circuits Classically"), #xref("programming-languages", "complexity", label: "Complexity Theory") (classical classes), #xref("quantum-computing", "hamiltonian-simulation", label: "Quantum Simulation of Physics and Chemistry")

== BQP

#definition[
*BQP* (bounded-error quantum polynomial time) is the class of decision problems $L$ for which there is a uniform family of quantum circuits $\{C_n\}$, of size polynomial in $n$, such that for every input $x$ of length $n$, measuring the first output qubit of $C_n$ applied to $|x chevron.r |0 dots.c 0 chevron.r$ gives 1 with probability at least $2 \/ 3$ if $x in L$ and at most $1 \/ 3$ if $x in.not L$.
]

*Uniform* means a classical machine can write down the circuit $C_n$ in time polynomial in $n$, which stops hard problems from being hidden in the circuit description. The constants $2 \/ 3$ and $1 \/ 3$ are arbitrary: repeating the computation and taking a majority vote drives the error down exponentially, so any constant gap works. The definition is robust to the choice of universal gate set (by the Solovay-Kitaev theorem) and equivalent to definitions based on quantum Turing machines.

=== Where BQP Sits

The known inclusions are

$ "P" subset.eq "BPP" subset.eq "BQP" subset.eq "PP" subset.eq "PSPACE" $

A quantum computer can simulate a probabilistic classical one by using measured superpositions as random coins, giving $"BPP" subset.eq "BQP"$ (Bernstein and Vazirani, 1997). The upper bound $"BQP" subset.eq "PSPACE"$ shows that quantum computers cannot solve problems beyond what classical computers solve with polynomial memory and unlimited time.

#theorem[
$"BQP" subset.eq "PSPACE"$.
]

#proof[
The acceptance probability of a circuit $C = U_T dots.c U_1$ on input $x$ is $sum_y |chevron.l y|U_T dots.c U_1|x, 0 chevron.r|^2$ over outputs $y$ whose first bit is 1. Inserting a complete basis between consecutive gates writes each amplitude as a sum over *paths* $x = z_0, z_1, dots, z_T = y$ of products $product_t chevron.l z_t|U_t|z_(t - 1) chevron.r$. Each factor involves a gate on a constant number of qubits and is computable in polynomial time. A classical machine can enumerate the exponentially many paths one at a time, keeping only the current path and a running sum, in polynomial space. Summing $|"amplitude"|^2$ over the relevant outputs, again one at a time, decides whether the acceptance probability is above $2 \/ 3$ or below $1 \/ 3$.
]

The sharper bound $"BQP" subset.eq "PP"$ (Adleman, DeMarrais and Huang, 1997) follows from the same path-sum idea with a counting argument. Since $"P" = "PSPACE"$ is not ruled out, no unconditional proof that $"BQP" != "BPP"$ is possible without a breakthrough in classical complexity.

=== BQP and NP

The relationship between BQP and NP is unknown, and the evidence suggests neither contains the other:

- *Factoring* and the discrete logarithm are in BQP (Shor) and in $"NP" inter "coNP"$, and are not believed to be NP-complete. Shor's algorithm exploits a hidden periodic structure that generic NP-complete problems lack.
- No known quantum algorithm solves an NP-complete problem in polynomial time. Grover's search gives only a quadratic speedup, and that is optimal for unstructured search.
- Conversely, BQP contains problems (such as simulating quantum dynamics) that are not known to be in NP: there may be no short classical certificate of the answer.

== Oracles and Query Lower Bounds

Without proofs of class separations, much of the evidence about quantum power comes from the *query model*: an algorithm accesses its input only through an oracle (a black box) and is charged per query.

#theorem(name: "Bennett, Bernstein, Brassard, Vazirani")[
Any quantum algorithm that finds a marked item among $N$ with bounded error, using only oracle access, needs $Omega(sqrt(N))$ queries.
]

Grover's algorithm matches this bound, so for unstructured search the quadratic speedup is the best possible. As a corollary, there is an oracle relative to which $"NP" subset.eq.not "BQP"$: a quantum computer cannot solve NP-complete problems by brute force "in parallel".

Exponential speedups in the query model require structure, a *promise* about the input:

- *Simon's problem* (1994): given $f$ with $f(x) = f(y)$ exactly when $x plus.o y in \{0, s\}$, find $s$. Quantum algorithms need $O(n)$ queries; classical ones need $Omega(2^(n \/ 2))$. Simon's algorithm directly inspired Shor's.
- *Forrelation* and related problems give the largest possible separations, and Raz and Tal (2019) used one to construct an oracle relative to which BQP is not contained in the entire *polynomial hierarchy* PH, strong evidence that quantum computation can solve problems beyond any constant number of NP-style quantifier alternations.

For *total* functions (no promise), the separation is at most polynomial: deterministic query complexity $D(f)$ is at most a constant power of quantum query complexity $Q(f)$, and the best bound, $D(f) = O(Q(f)^4)$, is tight (Aaronson, Ben-David, Kothari, Rao and Tal, 2021, building on Huang's sensitivity theorem). Exponential quantum speedups come from structure in the problem, never from the input space alone.

The main lower-bound techniques are the *polynomial method* (the acceptance probability of a $T$-query algorithm is a polynomial of degree at most $2 T$ in the input bits, so a function needing high approximate degree needs many queries) and the *adversary method* (bounding how fast queries can distinguish inputs that must give different answers).

== QMA and the Local Hamiltonian Problem

NP is the class of problems whose "yes" instances have short certificates a deterministic machine can check. Its quantum analogue allows a quantum certificate and a quantum verifier.

#definition[
*QMA* (quantum Merlin-Arthur) is the class of decision problems for which a polynomial-time quantum verifier, given an input $x$ and a quantum *witness* state of polynomially many qubits, accepts with probability at least $2 \/ 3$ for some witness when $x in L$, and with probability at most $1 \/ 3$ for every witness when $x in.not L$.
]

The canonical QMA-complete problem is the quantum analogue of satisfiability.

#theorem(name: "Kitaev")[
The $k$-local Hamiltonian problem is QMA-complete: given $H = sum_j H_j$ on $n$ qubits, where each term acts on at most $k$ qubits, and thresholds $a < b$ with $b - a >= 1 \/ "poly"(n)$, decide whether the ground-state energy of $H$ is at most $a$ or at least $b$.
]

Kitaev proved this for $k = 5$, encoding a verifier's computation history in the ground state of a Hamiltonian (a quantum version of the Cook-Levin construction); later work brought it to $k = 2$ (Kempe, Kitaev and Regev, 2006) and to physically natural models on two-dimensional lattices. The consequence is sobering for quantum chemistry and materials science: computing ground-state energies is hard *in general* even for quantum computers. Practical quantum simulation relies on the physical systems of interest being easier than the worst case, typically because good approximate ground states can be prepared (#xref("quantum-computing", "hamiltonian-simulation", label: "Quantum Simulation of Physics and Chemistry")).

Variants probe what makes quantum proofs powerful: *QCMA* allows only classical witnesses with a quantum verifier, and *QMA(2)* allows two unentangled quantum witnesses, which appears to be more powerful, though this is unproven.

== Complexity of Sampling

The quantum advantage experiments of 2019 onwards did not solve decision problems. They *sampled* from the output distribution of a quantum process: random circuits, boson sampling with photons, or IQP circuits of commuting gates. The complexity argument for why classical computers cannot do the same runs through *postselection*:

1. If quantum circuits could postselect on (condition on) measurement outcomes of exponentially small probability, they could solve PP-complete problems: $"PostBQP" = "PP"$ (Aaronson, 2005).
2. If a classical randomised algorithm could sample *exactly* (or within a small multiplicative error) from the output distribution of these restricted circuits, then postselected classical computation would equal PP as well, which by Toda's theorem would collapse the polynomial hierarchy to its third level (Bremner, Jozsa and Shepherd, 2011; Aaronson and Arkhipov, 2011).

The polynomial hierarchy is widely believed not to collapse, so exact efficient classical sampling is believed impossible. Experiments, however, sample *approximately* (with additive error, from noisy devices), and hardness of approximate sampling relies on further conjectures about the average-case hardness of computing output probabilities and their anti-concentration (Bouland, Fefferman, Nirkhe and Vazirani, 2019). Noise weakens the argument further: for random circuits with a constant error rate per gate, the output approaches the uniform distribution as depth grows, and polynomial-time classical algorithms exist for noisy random circuit sampling in some regimes (#xref("quantum-computing", "simulating-quantum-circuits", label: "Simulating Quantum Circuits Classically")). The advantage claimed by sampling experiments is therefore real but conditional, and it does not by itself imply useful computation.

== Dequantization and Input Assumptions

Several proposed exponential speedups for linear algebra and machine learning assumed the input was available in quantum superposition, through *quantum random access memory* (QRAM). Tang (2019) showed that for recommendation systems, a classical algorithm with comparable *sampling* access to the input (the ability to sample entries with probability proportional to their squared magnitude) runs in time polynomial in the same parameters, removing the exponential speedup. Similar *quantum-inspired* classical algorithms followed for low-rank regression, principal component analysis and more. The general lesson is to compare quantum and classical algorithms under the same input model. Linear-system solvers such as HHL also come with conditions (sparse, well-conditioned matrices, efficient state preparation, and outputs that are expectation values rather than full solution vectors) that must all hold for an exponential advantage to survive (Aaronson, 2015).

== Verifying Quantum Computation

If a quantum computer solves a problem no classical computer can, how can a classical user check the answer? For problems in NP, such as factoring, the answer is easy to verify. For general BQP computations, the question is subtler:

- With an *interactive proof* in which the verifier has a small quantum device (preparing single qubits), blind and verifiable delegation protocols exist.
- *Mahadev* (2018) showed that a purely classical verifier can check any BQP computation, assuming the prover cannot break the learning with errors (LWE) problem, a post-quantum cryptographic assumption.
- With two non-communicating quantum provers sharing entanglement, the power of interactive proofs becomes immense: $"MIP"^* = "RE"$ (Ji, Natarajan, Vidick, Wright and Yuen, 2020), meaning such verifiers can be convinced of the answer to the halting problem. The result also resolved Connes' embedding problem in operator algebras negatively.

== A Map of the Classes

#table(
  columns: 3,
  [*Class*], [*Informal meaning*], [*Example*],
  [BPP], [Efficient classical randomised computation], [Primality testing (also in P)],
  [BQP], [Efficient quantum computation], [Factoring, quantum dynamics simulation],
  [NP], [Classical certificate, classical verifier], [SAT, graph colouring],
  [QMA], [Quantum certificate, quantum verifier], [Local Hamiltonian ground energy],
  [PP], [Majority-vote counting], [Exact output probabilities of quantum circuits],
  [PostBQP], [Quantum computation with postselection], [Equals PP],
  [PSPACE], [Polynomial memory, unlimited time], [Contains all of the above],
)

== Further Reading

Bernstein, E., Vazirani, U. (1997). "Quantum Complexity Theory." SIAM Journal on Computing.

Adleman, L., DeMarrais, J., Huang, M.-D. (1997). "Quantum Computability." SIAM Journal on Computing.

Bennett, C., Bernstein, E., Brassard, G., Vazirani, U. (1997). "Strengths and Weaknesses of Quantum Computing." SIAM Journal on Computing.

Kitaev, A., Shen, A., Vyalyi, M. (2002). "Classical and Quantum Computation." American Mathematical Society.

Kempe, J., Kitaev, A., Regev, O. (2006). "The Complexity of the Local Hamiltonian Problem." SIAM Journal on Computing.

Aaronson, S. (2005). "Quantum Computing, Postselection, and Probabilistic Polynomial-Time." Proceedings of the Royal Society A.

Aaronson, S., Arkhipov, A. (2011). "The Computational Complexity of Linear Optics." STOC.

Bremner, M., Jozsa, R., Shepherd, D. (2011). "Classical Simulation of Commuting Quantum Computations Implies Collapse of the Polynomial Hierarchy." Proceedings of the Royal Society A.

Raz, R., Tal, A. (2019). "Oracle Separation of BQP and PH." STOC.

Tang, E. (2019). "A Quantum-Inspired Classical Algorithm for Recommendation Systems." STOC.

Mahadev, U. (2018). "Classical Verification of Quantum Computations." FOCS.

Ji, Z., Natarajan, A., Vidick, T., Wright, J., Yuen, H. (2020). "MIP\* = RE." arXiv:2001.04383.

Aaronson, S. (2015). "Read the Fine Print." Nature Physics.

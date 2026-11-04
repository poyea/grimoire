#import "../template.typ": xref

= Quantum Simulation of Physics and Chemistry <hamiltonian-simulation>

Feynman's original argument for quantum computers (1982) was not about factoring but about physics: nature is quantum, describing a quantum system of $n$ particles classically takes resources exponential in $n$, so a machine that is itself quantum should simulate it efficiently. Simulating molecules, materials and nuclear matter remains the application with the clearest scientific value and, for many problems, an exponential gap between the best known quantum and classical methods. This chapter covers how a Hamiltonian is turned into a circuit (product formulas, randomised methods, and the block-encoding and qubitization framework), how ground-state energies are extracted, how fermions are mapped onto qubits, what resource estimates say about useful chemistry, the classical methods any quantum approach must beat, and analog simulators that already explore physics beyond classical reach.

*See also:* #xref("quantum-computing", "quantum-algorithms", label: "Quantum Algorithms") (phase estimation, VQE), #xref("quantum-computing", "error-correction", label: "Quantum Error Correction") (resource estimates), #xref("quantum-computing", "quantum-compilation", label: "Compilation and the Quantum Software Stack"), #xref("numerical-computing", "eigenvalue-problems", label: "Eigenvalue Problems") (the classical problem)

== The Simulation Problem

A closed quantum system evolves under the Schrödinger equation, so its state at time $t$ is $|psi(t) chevron.r = e^(-i H t) |psi(0) chevron.r$ for a Hermitian *Hamiltonian* $H$. The *Hamiltonian simulation problem*: given a description of $H$, a time $t$ and a precision $epsilon$, build a circuit that implements an operator within $epsilon$ of $e^(-i H t)$.

Physical Hamiltonians are structured. They are typically sums of many simple terms, $H = sum_(j = 1)^L h_j P_j$, where each $P_j$ is a Pauli string acting on a few qubits (for lattice models) or on many (for molecules after fermion encoding). Each $e^(-i h_j P_j tau)$ is easy to implement: change basis so that $P_j$ becomes a product of $Z$s, compute the parity of the affected qubits into one qubit with a CNOT ladder, rotate it with $R_z(2 h_j tau)$, and undo. The difficulty is that the terms do not commute, so $e^(-i H t)$ is not the product of the term exponentials.

== Product Formulas

The *Lie-Trotter* formula splits the evolution into $r$ short steps and applies each term separately in every step:

$ e^(-i H t) approx (product_(j = 1)^L e^(-i h_j P_j t \/ r))^r $

The error comes from the commutators of the terms. For the first-order formula it is bounded by roughly $(t^2 \/ (2 r)) sum_(j < k) norm([h_j P_j, h_k P_k])$, so $r = O(t^2 \/ epsilon)$ steps suffice. The symmetric second-order formula (apply the terms forwards with half steps, then backwards) has error $O(t^3 \/ r^2)$, and Suzuki's recursive construction gives formulas of order $2k$ with error $O(t^(2k + 1) \/ r^(2k))$, at a constant factor that grows as $5^(k - 1)$.

Product formulas are simple, need no ancilla qubits, and in practice perform much better than their worst-case bounds suggest. Their error depends on *commutators* rather than on the norms of the terms (Childs et al., 2021), so Hamiltonians with mostly commuting terms, such as local lattice models where distant terms commute, need few steps. For near-term and early fault-tolerant hardware they are the default method, and their gate counts for lattice models such as the Fermi-Hubbard model are among the smallest of any useful quantum computation.

=== Randomised Methods

*qDRIFT* (Campbell, 2019) samples terms at random with probability proportional to $|h_j|$ and applies each for a fixed short time. Its cost depends on $lambda = sum_j |h_j|$ and not on the number of terms $L$, requiring $O(lambda^2 t^2 \/ epsilon)$ gates. For molecular Hamiltonians, with millions of terms of widely varying size, that independence from $L$ can make it much cheaper than deterministic product formulas. Hybrids apply the large terms deterministically and sample the long tail of small ones.

== Block Encodings and Qubitization

Asymptotically optimal methods treat the Hamiltonian as a matrix to be embedded in a larger unitary.

- A *block encoding* of $H$ is a unitary $U$ on the system plus some ancilla qubits whose top-left block (ancillas in $|0 chevron.r$) equals $H \/ lambda$. For a Pauli sum, the *linear combination of unitaries* (LCU) construction builds one with a *PREPARE* circuit, which loads the coefficients $|h_j| \/ lambda$ into the ancilla amplitudes, and a *SELECT* circuit, which applies $P_j$ controlled on the ancilla holding $j$.
- *Qubitization* (Low and Chuang, 2019) turns one block-encoding call into a walk operator whose eigenphases are $plus.minus arccos(E_k \/ lambda)$ for each eigenvalue $E_k$ of $H$. Combined with *quantum signal processing* and its generalisation, the *quantum singular value transformation* (Gilyén et al., 2019), the walk can implement essentially any polynomial of $H$, including $e^(-i H t)$, with

$ O(lambda t + log(1 \/ epsilon)) $

block-encoding calls, which matches known lower bounds. The same framework yields matrix inversion, Gibbs-state preparation and eigenvalue filtering, and much of the recent literature on fault-tolerant algorithms is phrased in it.

For chemistry, qubitization is often used directly for energy estimation: running phase estimation on the walk operator reads out $arccos(E \/ lambda)$ without ever implementing $e^(-i H t)$, which saves a constant factor. The total cost is dominated by $lambda$, so much research goes into *factorising* the Hamiltonian (double factorisation, tensor hypercontraction) to reduce $lambda$ and the cost of PREPARE.

== Finding Ground-State Energies

Most chemistry questions (reaction energies, barrier heights, binding affinities) reduce to the lowest eigenvalue of $H$ for a set of molecular geometries. *Quantum phase estimation* (QPE) applied to $e^(-i H t)$ or to a qubitized walk returns an eigenvalue $E_k$ with probability $|chevron.l E_k|psi_0 chevron.r|^2$, the overlap of the input state with the corresponding eigenstate, and costs $O(1 \/ epsilon)$ applications for precision $epsilon$ (the *Heisenberg limit*). Chemical accuracy is conventionally 1.6 millihartree (1 kcal/mol).

The weak point is the *initial state*. QPE succeeds only if the prepared state has non-negligible overlap with the ground state. For small molecules a Hartree-Fock state or a cheap classical approximation suffices, but for the strongly correlated systems where quantum computers would help most, there is no guarantee, and for large systems the overlap of simple product states with the true ground state can decay exponentially with system size (the *orthogonality catastrophe*). Finding the ground state of a general local Hamiltonian is QMA-complete, so no algorithm, quantum or classical, is expected to solve the problem in general. Quantum advantage in chemistry rests on the belief that physically relevant molecules admit good initial states while still being hard classically, a belief that is plausible but debated (Lee et al., 2023).

The *variational quantum eigensolver* (VQE, #xref("quantum-computing", "quantum-algorithms", label: "Quantum Algorithms")) was proposed for noisy hardware: prepare a parameterised state, measure the energy, and let a classical optimiser adjust the parameters. Its measurement cost (the energy is a sum of many Pauli expectation values, each needing many shots), the difficulty of optimising noisy and flat landscapes, and the lack of accuracy guarantees have so far prevented it from beating classical methods on any problem of chemical interest.

== From Fermions to Qubits

Electrons are fermions. In *second quantisation*, a molecule is described in a basis of $N$ spin-orbitals, and its Hamiltonian is

$ H = sum_(p q) h_(p q) a_p^dagger a_q + 1/2 sum_(p q r s) h_(p q r s) a_p^dagger a_q^dagger a_r a_s $

where $a_p^dagger$ and $a_p$ create and annihilate an electron in orbital $p$. The coefficients come from classical integrals over the orbitals, and there are $O(N^4)$ of them. Each orbital becomes one qubit (occupied or empty), but fermionic operators *anticommute*, and qubit operators on different qubits commute, so the encoding must restore the signs.

The *Jordan-Wigner* transformation does so with strings of $Z$ operators that count the parity of all lower-numbered orbitals:

$ a_j = (product_(k < j) Z_k) (X_j + i Y_j) / 2 $

A hopping term between orbitals $p < q$ becomes $1/2 (X_p Z_(p+1) dots.c Z_(q-1) X_q + Y_p Z_(p+1) dots.c Z_(q-1) Y_q)$:

```cpp
#include <array>
#include <string>

// Jordan-Wigner image of the hopping term a_p^dagger a_q + a_q^dagger a_p (p < q)
// on n qubits: 1/2 (X Z...Z X + Y Z...Z Y). Character k of each string acts on qubit k.
std::array<std::string, 2> jordan_wigner_hopping(int p, int q, int n) {
    std::string xx(static_cast<std::size_t>(n), 'I');
    std::string yy(static_cast<std::size_t>(n), 'I');
    for (int k = p + 1; k < q; ++k) {
        xx[static_cast<std::size_t>(k)] = 'Z';   // parity of the orbitals in between
        yy[static_cast<std::size_t>(k)] = 'Z';
    }
    xx[static_cast<std::size_t>(p)] = 'X';
    xx[static_cast<std::size_t>(q)] = 'X';
    yy[static_cast<std::size_t>(p)] = 'Y';
    yy[static_cast<std::size_t>(q)] = 'Y';
    return {xx, yy};                             // each with coefficient 1/2
}

// jordan_wigner_hopping(1, 4, 6) -> {"IXZZXI", "IYZZYI"}
```

The $Z$ strings make terms act on up to $N$ qubits. The *Bravyi-Kitaev* encoding stores parities in a tree so that every fermionic operator acts on $O(log N)$ qubits, and other encodings trade extra qubits for strictly local operators on lattices. The choice affects gate counts and how errors in the hardware map back to physical errors in the simulated system.

== How Big Must the Computer Be?

Resource estimates for chemistry have fallen by orders of magnitude as algorithms improved, and they are the clearest guide to when quantum simulation becomes useful. The benchmark molecule is *FeMoco*, the iron-molybdenum cofactor of nitrogenase, the enzyme that fixes nitrogen at room temperature where the industrial Haber-Bosch process needs high temperature and pressure.

- Reiher et al. (2017) gave the first detailed estimate using Trotterised phase estimation, of order $10^(14)$ to $10^(15)$ $T$ gates.
- Qubitization with tensor hypercontraction (Lee et al., 2021) reduced this to a few billion Toffoli gates, which under standard surface-code assumptions (physical error rate $10^(-3)$, microsecond cycles) corresponds to about four million physical qubits running for a few days.

Hubbard-model simulations beyond classical reach need far fewer resources, on the order of $10^6$ to $10^7$ $T$ gates and a few hundred logical qubits, which is why condensed-matter dynamics is a likely first application of early fault-tolerant machines.

== The Classical Competition

Any claim of quantum advantage must beat strong classical methods, and in chemistry those methods are very good:

- *Coupled cluster* (CCSD(T), the "gold standard") is highly accurate for weakly correlated molecules and scales as $O(N^7)$.
- *Density matrix renormalisation group* (DMRG) and tensor-network methods handle strongly correlated systems with limited entanglement, including active spaces of FeMoco's size.
- *Quantum Monte Carlo* methods are accurate for many systems but suffer from the fermionic sign problem.
- *Density functional theory* is approximate but cheap enough for thousands of atoms, and it is what most practising chemists use.

The problems where quantum computers should help most are those combining strong correlation with large size and with a need for high accuracy: transition-metal catalysts, some excited states, and real-time dynamics, where classical methods struggle because entanglement grows over time.

== Analog Quantum Simulation

A *digital* quantum computer simulates any Hamiltonian by compiling it to gates. An *analog* simulator instead engineers a controllable physical system whose own Hamiltonian matches the model of interest, and lets it evolve:

- *Ultracold atoms in optical lattices* realise Hubbard models directly, with quantum gas microscopes imaging individual atoms.
- *Rydberg atom arrays* (the same platform as neutral-atom quantum computers) realise Ising-type spin models in arbitrary two-dimensional geometries and have been used to observe quantum phases, including signatures of a topological spin liquid.
- *Trapped ions* simulate long-range interacting spin chains, and *quantum annealers* explore transverse-field Ising dynamics.

Analog simulators already reach system sizes and evolution times beyond exact classical simulation for some dynamics questions. Their limitations are a restricted set of Hamiltonians, imperfect calibration (the simulated Hamiltonian is only approximately the intended one), and no error correction, so results need careful validation against theory and cross-platform comparisons.

== Further Reading

Feynman, R. (1982). "Simulating Physics with Computers." International Journal of Theoretical Physics.

Lloyd, S. (1996). "Universal Quantum Simulators." Science.

Childs, A., Su, Y., Tran, M., Wiebe, N., Zhu, S. (2021). "Theory of Trotter Error with Commutator Scaling." Physical Review X.

Campbell, E. (2019). "Random Compiler for Fast Hamiltonian Simulation." Physical Review Letters.

Low, G. H., Chuang, I. (2019). "Hamiltonian Simulation by Qubitization." Quantum.

Gilyén, A., Su, Y., Low, G. H., Wiebe, N. (2019). "Quantum Singular Value Transformation and Beyond: Exponential Improvements for Quantum Matrix Arithmetics." STOC.

Reiher, M., Wiebe, N., Svore, K., Wecker, D., Troyer, M. (2017). "Elucidating Reaction Mechanisms on Quantum Computers." Proceedings of the National Academy of Sciences.

Lee, J. et al. (2021). "Even More Efficient Quantum Computations of Chemistry Through Tensor Hypercontraction." PRX Quantum.

McArdle, S., Endo, S., Aspuru-Guzik, A., Benjamin, S., Yuan, X. (2020). "Quantum Computational Chemistry." Reviews of Modern Physics.

Lee, S. et al. (2023). "Evaluating the Evidence for Exponential Quantum Advantage in Ground-State Quantum Chemistry." Nature Communications.

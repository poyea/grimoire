#import "../template.typ": xref, theorem, proof, definition

= Density Matrices, Channels, and Quantum Information <quantum-information>

State vectors describe isolated quantum systems in perfectly known states. Real qubits are neither isolated nor perfectly known: they interact with their surroundings, they are prepared imperfectly, and we often care about one part of a larger entangled system. The density-matrix formalism handles all three, and quantum channels describe every physically allowed process, including noise. This chapter develops that formalism and the basic results of quantum information built on it: partial trace and purification, generalised measurements, channels and their Kraus representation, distance measures, entropy and the Holevo bound, the no-cloning theorem, teleportation and superdense coding, and Bell inequalities.

*See also:* #xref("quantum-computing", "qubits-and-gates", label: "Qubits and Gates"), #xref("quantum-computing", "notation", label: "Notation and Conventions"), #xref("quantum-computing", "nisq-and-benchmarking", label: "NISQ Devices and Benchmarking") (noise models), #xref("machine-learning-foundations", "information-theory", label: "Information Theory") (classical counterpart)

== Density Matrices

Suppose a source emits state $|psi_i chevron.r$ with probability $p_i$. Every measurement statistic is then determined by the *density matrix*

$ rho = sum_i p_i |psi_i chevron.r chevron.l psi_i| $

since the probability of outcome $m$ for a projector $P_m$ is $sum_i p_i chevron.l psi_i|P_m|psi_i chevron.r = op("tr")(P_m rho)$.

#definition[
A *density matrix* (or density operator) on a $d$-dimensional system is a $d times d$ matrix $rho$ that is Hermitian, positive semidefinite ($rho >= 0$), and has unit trace ($op("tr") rho = 1$). It is *pure* if $rho = |psi chevron.r chevron.l psi|$ for some unit vector, which holds exactly when $op("tr") rho^2 = 1$; otherwise it is *mixed*.
]

The *purity* $op("tr") rho^2$ ranges from $1 \/ d$ (the maximally mixed state $II \/ d$) to 1. For a qubit, $rho = (II + arrow(r) dot arrow(sigma)) \/ 2$ with Bloch vector $|arrow(r)| <= 1$: pure states lie on the sphere, mixed states inside, and the maximally mixed state at the centre.

Different ensembles can produce the same $rho$. An equal mixture of $|0 chevron.r$ and $|1 chevron.r$ and an equal mixture of $|+ chevron.r$ and $|- chevron.r$ are both $II \/ 2$, and no experiment can tell them apart. The density matrix is therefore the complete physical description; the ensemble is not.

== Composite Systems and the Partial Trace

For a state $rho_(A B)$ of two systems, the state of $A$ alone is obtained by the *partial trace* over $B$:

$ rho_A = op("tr")_B rho_(A B) = sum_j (II_A times.o chevron.l j|_B) rho_(A B) (II_A times.o |j chevron.r_B) $

It is the unique description of $A$ that reproduces every measurement on $A$ alone. Applied to the Bell state $|Phi^+ chevron.r = (|00 chevron.r + |11 chevron.r) \/ sqrt(2)$, it gives $rho_A = II \/ 2$: a *pure* joint state whose parts are *maximally mixed*. That is the signature of entanglement. For product states $rho_A times.o rho_B$, the parts are as pure as the whole.

Every pure bipartite state has a *Schmidt decomposition* $|psi chevron.r = sum_k sqrt(lambda_k) |a_k chevron.r |b_k chevron.r$ with orthonormal sets $\{|a_k chevron.r\}$ and $\{|b_k chevron.r\}$. The *Schmidt coefficients* $lambda_k$ are the eigenvalues of both reduced states, and the state is entangled exactly when more than one is non-zero. Conversely, every mixed state $rho_A$ has a *purification*: a pure state on a larger system $A R$ whose partial trace over $R$ is $rho_A$. Mixedness can always be viewed as entanglement with something we are not looking at, the starting point for understanding decoherence.

== Generalised Measurements

Projective measurements are not the most general. A *POVM* (positive operator-valued measure) is a set of positive operators $\{E_m\}$ with $sum_m E_m = II$, giving outcome $m$ with probability $op("tr")(E_m rho)$. POVMs can have more outcomes than the dimension of the system, and every POVM can be realised as a projective measurement on the system plus an ancilla. They describe real detectors (with inefficiency and dark counts) and are needed for optimal state discrimination: two non-orthogonal states can be distinguished *unambiguously* some of the time, with an extra "inconclusive" outcome, but never perfectly.

== Quantum Channels

#definition[
A *quantum channel* is a linear map $cal(E)$ from density matrices to density matrices that is *completely positive* (it maps positive operators to positive operators even when applied to part of a larger system, $cal(E) times.o II$) and *trace preserving*. Every such map has a *Kraus representation*
$ cal(E)(rho) = sum_k K_k rho K_k^dagger, quad sum_k K_k^dagger K_k = II. $
]

Unitary evolution is the one-Kraus-operator case. Every channel can also be realised as a unitary interaction with an environment followed by discarding the environment (the *Stinespring dilation*), which is how noise arises physically. Standard single-qubit noise channels:

#table(
  columns: 3,
  [*Channel*], [*Action*], [*Physical picture*],
  [Bit flip], [$(1 - p) rho + p X rho X$], [Random $X$ errors],
  [Dephasing], [$(1 - p) rho + p Z rho Z$], [Loss of phase coherence ($T_2$ processes); shrinks the Bloch vector towards the $z$ axis],
  [Depolarising], [$(1 - p) rho + p II \/ 2$], [Isotropic noise; shrinks the Bloch vector uniformly],
  [Amplitude damping], [$K_0 = mat(1, 0; 0, sqrt(1 - gamma))$, $K_1 = mat(0, sqrt(gamma); 0, 0)$], [Energy relaxation $|1 chevron.r -> |0 chevron.r$ ($T_1$ processes)],
)

Applying a channel to a density matrix is a few small matrix products:

```cpp
#include <array>
#include <cmath>
#include <complex>
#include <span>

using cplx = std::complex<double>;
using mat2 = std::array<std::array<cplx, 2>, 2>;

mat2 multiply(const mat2& a, const mat2& b) {
    mat2 r{};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) r[i][j] += a[i][k] * b[k][j];
    return r;
}

mat2 dagger(const mat2& a) {
    return {{{std::conj(a[0][0]), std::conj(a[1][0])},
             {std::conj(a[0][1]), std::conj(a[1][1])}}};
}

// rho -> sum_k K_k rho K_k^dagger
mat2 apply_channel(const mat2& rho, std::span<const mat2> kraus) {
    mat2 out{};
    for (const mat2& k : kraus) {
        const mat2 term = multiply(multiply(k, rho), dagger(k));
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) out[i][j] += term[i][j];
    }
    return out;
}

double purity(const mat2& rho) {   // tr(rho^2)
    const mat2 sq = multiply(rho, rho);
    return (sq[0][0] + sq[1][1]).real();
}

// Amplitude damping with decay probability gamma = 1 - exp(-t / T1).
std::array<mat2, 2> amplitude_damping(double gamma) {
    return {mat2{{{1.0, 0.0}, {0.0, std::sqrt(1.0 - gamma)}}},
            mat2{{{0.0, std::sqrt(gamma)}, {0.0, 0.0}}}};
}
```

Starting from $|1 chevron.r$, repeated amplitude damping drives the state to $|0 chevron.r$ (purity dips, then returns to 1); starting from $|+ chevron.r$, the off-diagonal *coherences* decay as $sqrt(1 - gamma)$ per application. Channels compose, and the noise models used to simulate NISQ devices are built from these components.

== Distance Between States

Two measures dominate:

- *Trace distance* $D(rho, sigma) = 1/2 norm(rho - sigma)_1$, half the sum of the absolute eigenvalues of $rho - sigma$. It has a direct operational meaning: given one copy of either state with equal prior probability, the best possible measurement guesses correctly with probability $1/2 (1 + D(rho, sigma))$.
- *Fidelity* $F(rho, sigma) = (op("tr") sqrt(sqrt(rho) sigma sqrt(rho)))^2$, which reduces to $|chevron.l psi|phi chevron.r|^2$ for pure states. (Some texts define fidelity without the square; check the convention before comparing numbers.)

They are related by $1 - sqrt(F) <= D <= sqrt(1 - F)$. Neither increases under any channel: processing cannot make states more distinguishable. Gate and state fidelities quoted for hardware use these definitions or the closely related *average gate fidelity* over all input states.

== Entropy and the Holevo Bound

The *von Neumann entropy* $S(rho) = -op("tr")(rho log_2 rho) = -sum_k lambda_k log_2 lambda_k$ is the Shannon entropy of $rho$'s eigenvalues. It is zero for pure states and $log_2 d$ for the maximally mixed state. For a pure bipartite state, $S(rho_A) = S(rho_B)$ is the *entanglement entropy*, the standard measure of how entangled the two parts are; a Bell pair has exactly 1 bit of it, called one *ebit*.

How much classical information can be stored in a quantum system? A qubit's state space is a continuum, but measurement extracts only a finite amount:

#theorem(name: "Holevo bound")[
If a sender encodes classical message $x$ with probability $p_x$ as the state $rho_x$, then for any measurement the receiver performs, the mutual information between message and outcome satisfies
$ I(X : Y) <= chi = S(sum_x p_x rho_x) - sum_x p_x S(rho_x). $
]

Since $chi <= log_2 d$, $n$ qubits can carry at most $n$ bits of classical information. Quantum states are not a way to compress classical data beyond its size, even though describing a general $n$-qubit state takes $2^n$ complex numbers. (Superdense coding, below, doubles the rate only by consuming prior entanglement.)

== No-Cloning

#theorem(name: "No-cloning")[
There is no unitary $U$ and fixed blank state $|e chevron.r$ such that $U(|psi chevron.r times.o |e chevron.r) = |psi chevron.r times.o |psi chevron.r$ for every state $|psi chevron.r$.
]

#proof[
Suppose such a $U$ copies two states $|psi chevron.r$ and $|phi chevron.r$. Unitaries preserve inner products, so
$ chevron.l psi|phi chevron.r = chevron.l psi|phi chevron.r chevron.l e|e chevron.r = (chevron.l psi| chevron.l psi|)(|phi chevron.r |phi chevron.r) = chevron.l psi|phi chevron.r^2. $
Hence $chevron.l psi|phi chevron.r$ is 0 or 1: a cloner can copy only states that are identical or orthogonal, never an arbitrary unknown state. Allowing an ancilla and a final measurement does not help, because any such process is a unitary on a larger system.
]

No-cloning underlies quantum key distribution (an eavesdropper cannot copy the transmitted qubits to measure later), rules out simple repetition codes for quantum error correction (which instead spread information into entanglement, #xref("quantum-computing", "error-correction", label: "Quantum Error Correction")), and is consistent with relativity: cloning plus entanglement would allow faster-than-light signalling.

== Teleportation and Superdense Coding

Entanglement and classical communication can trade for each other in two complementary protocols.

*Teleportation* (Bennett et al., 1993) moves an unknown qubit state $|psi chevron.r$ from Alice to Bob using one shared Bell pair and two classical bits. Alice performs a Bell measurement on $|psi chevron.r$ and her half of the pair, obtaining one of four outcomes. Writing out the joint state shows that Bob's qubit is then $|psi chevron.r$ up to one of $II$, $X$, $Z$ or $X Z$, determined by Alice's outcome. She sends the two bits, and Bob applies the corresponding correction. The original is destroyed by Alice's measurement (consistent with no-cloning), and nothing travels faster than light, because Bob's qubit is maximally mixed until the classical bits arrive. Teleportation is a basic primitive of quantum networks (with entanglement swapping) and of some fault-tolerant architectures (gate teleportation).

*Superdense coding* runs the resource exchange in the other direction: Alice applies one of $II$, $X$, $Z$, $X Z$ to her half of a shared Bell pair, sends that single qubit to Bob, and Bob's Bell measurement on both qubits recovers which of four operations she chose. One qubit plus one ebit carry two classical bits, in agreement with the Holevo bound once the entanglement is counted.

== Bell Inequalities

Could quantum correlations be explained by hidden classical variables carried by each particle? Bell (1964) showed that any *local hidden-variable* theory obeys inequalities that quantum mechanics violates. The most used is *CHSH*: Alice measures one of two $plus.minus 1$-valued observables $A_0, A_1$, Bob one of $B_0, B_1$, and

$ S = chevron.l A_0 B_0 chevron.r + chevron.l A_0 B_1 chevron.r + chevron.l A_1 B_0 chevron.r - chevron.l A_1 B_1 chevron.r $

Every local hidden-variable model has $|S| <= 2$. A Bell pair with suitably rotated measurements gives $S = 2 sqrt(2) approx 2.83$, which is the maximum quantum mechanics allows (*Tsirelson's bound*). Experiments violating CHSH while closing the detection and locality loopholes simultaneously were performed in 2015 (Hensen et al. and others), work recognised by the 2022 Nobel Prize in Physics. Beyond foundations, Bell violations are a resource: *device-independent* protocols certify randomness and secret keys from the observed statistics alone, without trusting the devices.

== Detecting Entanglement

A mixed state is *separable* if it can be written as $sum_i p_i rho_i^A times.o rho_i^B$, and *entangled* otherwise. Deciding separability is NP-hard in general, but practical criteria exist:

- *Peres-Horodecki (PPT) criterion*: if the *partial transpose* of $rho$ (transposing only $B$'s indices) has a negative eigenvalue, $rho$ is entangled. For two qubits, and for a qubit and a qutrit, PPT is also sufficient for separability.
- *Entanglement witnesses*: an observable $W$ with $op("tr")(W sigma) >= 0$ for every separable $sigma$; measuring $op("tr")(W rho) < 0$ proves entanglement. This is how experiments certify multi-qubit entanglement without full state tomography, whose cost grows exponentially with the number of qubits.
- *Quantitative measures*: entanglement entropy for pure states; for mixed states, entanglement of formation, concurrence (closed form for two qubits) and negativity (from the partial transpose).

== Further Reading

Nielsen, M., Chuang, I. (2010). "Quantum Computation and Quantum Information." Cambridge University Press, 10th anniversary ed.

Wilde, M. (2017). "Quantum Information Theory." Cambridge University Press, 2nd ed.

Watrous, J. (2018). "The Theory of Quantum Information." Cambridge University Press.

Holevo, A. (1973). "Bounds for the Quantity of Information Transmitted by a Quantum Communication Channel." Problems of Information Transmission.

Wootters, W., Zurek, W. (1982). "A Single Quantum Cannot Be Cloned." Nature.

Bennett, C. et al. (1993). "Teleporting an Unknown Quantum State via Dual Classical and Einstein-Podolsky-Rosen Channels." Physical Review Letters.

Clauser, J., Horne, M., Shimony, A., Holt, R. (1969). "Proposed Experiment to Test Local Hidden-Variable Theories." Physical Review Letters.

Hensen, B. et al. (2015). "Loophole-free Bell Inequality Violation Using Electron Spins Separated by 1.3 Kilometres." Nature.

Peres, A. (1996). "Separability Criterion for Density Matrices." Physical Review Letters.

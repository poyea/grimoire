#import "../template.typ": xref, theorem

= Simulating Quantum Circuits Classically <simulating-quantum-circuits>

Classical simulation of quantum circuits is how quantum algorithms are developed and debugged, how hardware results are checked, how error-correcting codes are designed, and how claims of quantum advantage are tested. Every simulation method exploits some structure that keeps the cost below the naive exponential, and every one fails on circuits that lack that structure. This chapter covers the main methods (state vectors, stabilizer tableaux, tensor networks and path sums), how each is engineered for performance, how noise is simulated, and what the limits of classical simulation say about quantum advantage.

*See also:* #xref("quantum-computing", "qubits-and-gates", label: "Qubits and Gates"), #xref("quantum-computing", "quantum-information", label: "Density Matrices, Channels, and Quantum Information"), #xref("quantum-computing", "nisq-and-benchmarking", label: "NISQ Devices and Benchmarking") (supremacy claims), #xref("gpu-architecture", "memory-hierarchy", label: "Memory Hierarchy") (bandwidth-bound kernels)

== State-Vector Simulation

The direct approach stores all $2^n$ complex amplitudes of an $n$-qubit state and applies each gate as a sparse linear map. With double-precision complex numbers (16 bytes each), memory is the binding constraint:

#table(
  columns: 2,
  [*Qubits*], [*State vector size*],
  [20], [16 MiB],
  [30], [16 GiB],
  [36], [1 TiB],
  [40], [16 TiB],
  [45], [512 TiB],
)

A laptop simulates about 30 qubits, a large GPU node about 35 to 38, and the largest supercomputer runs have reached the mid-40s (Häner and Steiger simulated 45 qubits with 0.5 PB of memory in 2017). Every additional qubit doubles memory and time.

=== Applying Gates

A single-qubit gate $U$ on qubit $k$ acts independently on each pair of amplitudes whose indices differ only in bit $k$. A controlled gate acts only on pairs where the control bit is 1. In C++:

```cpp
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

using amplitude = std::complex<double>;
using gate_1q = std::array<amplitude, 4>;   // row-major 2x2 unitary

class state_vector {
public:
    explicit state_vector(unsigned num_qubits)
        : amplitudes_(std::size_t{1} << num_qubits) {
        amplitudes_[0] = 1.0;   // |00...0>
    }

    // Apply a 2x2 unitary to `target`. Qubit 0 is the least significant bit.
    void apply_1q(const gate_1q& u, unsigned target) {
        const std::size_t stride = std::size_t{1} << target;
        for (std::size_t base = 0; base < amplitudes_.size(); base += 2 * stride) {
            for (std::size_t i = base; i < base + stride; ++i) {
                const amplitude a0 = amplitudes_[i];
                const amplitude a1 = amplitudes_[i + stride];
                amplitudes_[i] = u[0] * a0 + u[1] * a1;
                amplitudes_[i + stride] = u[2] * a0 + u[3] * a1;
            }
        }
    }

    void apply_cnot(unsigned control, unsigned target) {
        const std::size_t c = std::size_t{1} << control;
        const std::size_t t = std::size_t{1} << target;
        for (std::size_t i = 0; i < amplitudes_.size(); ++i) {
            if ((i & c) != 0 && (i & t) == 0) {
                std::swap(amplitudes_[i], amplitudes_[i | t]);
            }
        }
    }

    // Sample a full computational-basis measurement without collapsing.
    std::size_t sample(std::mt19937_64& rng) const {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        double r = uniform(rng);
        for (std::size_t i = 0; i < amplitudes_.size(); ++i) {
            r -= std::norm(amplitudes_[i]);   // |amplitude|^2
            if (r <= 0.0) return i;
        }
        return amplitudes_.size() - 1;        // guard against rounding
    }

private:
    std::vector<amplitude> amplitudes_;
};

// Bell state: H on qubit 0, then CNOT(0 -> 1). Samples are 0 (|00>) or 3 (|11>).
std::size_t bell_sample(std::mt19937_64& rng) {
    const double s = 1.0 / std::sqrt(2.0);
    state_vector psi(2);
    psi.apply_1q({s, s, s, -s}, 0);
    psi.apply_cnot(0, 1);
    return psi.sample(rng);
}
```

Each gate touches the whole vector with only a few floating-point operations per amplitude, so state-vector simulation is *memory-bandwidth bound*. At 30 qubits, one gate reads and writes 16 GiB, about 0.1 s at 300 GB/s. The engineering therefore focuses on reducing passes over memory:

- *Gate fusion*: multiply consecutive gates acting on the same few qubits into one $2^k times 2^k$ matrix (for $k$ up to about 5), so that one pass applies many gates. The extra arithmetic is free because the kernel was memory-bound anyway.
- *Cache blocking*: reorder gates so that runs of gates on low-order qubits are applied to cache-sized blocks of the vector before moving on.
- *SIMD and GPUs*: the inner loops vectorise well, and GPU libraries (NVIDIA cuStateVec, Qiskit Aer's GPU backend, qsim) achieve close to peak memory bandwidth.
- *Distribution*: across nodes, the high-order qubits are "global" (they index the node) and the rest are "local". Gates on local qubits need no communication; gates on global qubits are handled by swapping a global qubit with a local one, which is an all-to-all exchange of half the state. Scheduling to minimise those swaps dominates performance at scale.

Measurement in a state-vector simulator is cheap and *exact*: the full probability distribution is available, which hardware never provides. Sampling many shots costs one pass to build cumulative probabilities and then a binary search per shot.

== Stabilizer Simulation

A large and important class of circuits can be simulated in polynomial time.

#theorem(name: "Gottesman-Knill")[
A quantum circuit that starts in a computational basis state and uses only Clifford gates (Hadamard, phase $S$, CNOT), Pauli measurements, and classical control on measurement outcomes can be simulated efficiently on a classical computer.
]

The key is that such states are *stabilizer states*: each is uniquely determined by $n$ commuting Pauli operators that leave it unchanged (#xref("quantum-computing", "qubits-and-gates", label: "Qubits and Gates") introduces the stabilizer formalism). Instead of $2^n$ amplitudes, the simulator stores a *tableau* of $2n$ Pauli strings (the stabilizers plus "destabilizers") in about $4n^2$ bits. A Clifford gate maps Paulis to Paulis, so applying it updates two columns of the tableau in $O(n)$ bit operations; measurement costs $O(n^2)$ in the Aaronson-Gottesman (CHP) algorithm.

Stabilizer simulation is indispensable for quantum error correction, because syndrome extraction circuits are Clifford and Pauli noise maps Paulis to Paulis. *Stim* (Gidney, 2021) samples noisy error-correction circuits with millions of operations at very high rates by tracking how Pauli errors propagate through the circuit (a *Pauli frame*) relative to one noiseless reference sample, using bit-parallel operations over many shots at once. Nearly all surface-code threshold and decoder studies run on simulators of this kind.

Clifford circuits are not universal; adding the $T$ gate makes them so. Simulation cost then grows exponentially in the number of $T$ gates rather than in the number of qubits. *Stabilizer-rank* methods write the state as a sum of stabilizer states, and the number of terms needed for $t$ $T$ gates grows roughly as $2^(0.23 t)$ (Bravyi and Gosset, 2016), so circuits with many qubits and a few dozen $T$ gates remain tractable.

== Tensor Network Simulation

A circuit is itself a *tensor network*: each gate is a tensor with one index per input and output qubit wire, and computing an amplitude $chevron.l x|U|0 chevron.r$ means contracting the whole network. The cost depends on the *order* of contraction, and the best order's cost is exponential in the network's *treewidth* rather than in the number of qubits. Shallow circuits, circuits on lines, and circuits with limited entanglement can be contracted cheaply even with hundreds of qubits.

- *Matrix product states* (MPS) represent the state as a chain of tensors connected by *bond* indices of dimension $chi$. Memory is $O(n chi^2)$, and $chi$ must grow as $2^S$ where $S$ is the entanglement entropy across a cut. Low-entanglement circuits (and one-dimensional physics) are cheap; generic deep circuits make $chi$ exponential. Truncating small singular values gives an approximate simulation with controllable fidelity.
- *General contraction* of the circuit network, with orderings found by graph-partitioning heuristics and *slicing* (fixing some indices and summing over independent sub-contractions) to fit in memory, is how random-circuit-sampling experiments are challenged.

The history of Google's 2019 Sycamore experiment illustrates the method's power: the original claim that sampling its 53-qubit circuits would take a classical supercomputer 10 000 years was followed by tensor-network simulations that produced comparable samples in days and then hours on GPU clusters (Pan and Zhang, 2022, among others). Later experiments responded with more qubits, deeper circuits and lower error rates, moving beyond the reach of known contraction methods. Claims of quantum advantage are claims about the best *known* classical algorithm, and they move as algorithms improve.

== Path Sums and Hybrid Methods

The *Feynman path* approach writes an output amplitude as a sum over all computational-basis paths through the circuit, each a product of gate matrix entries. Memory is linear in circuit size, but the number of paths is exponential in the number of branching gates. *Schrödinger-Feynman hybrids* split the qubits into two halves, simulate each half's state vector, and sum over paths only for the gates that cross the cut. Their cost is exponential in the number of cross-cut gates, giving a dial between memory and time that suits circuits with few gates between the halves.

== Simulating Noise

Real hardware is noisy, and simulating noise is essential for predicting device behaviour and testing error mitigation:

- *Density-matrix simulation* applies channels directly to $rho$, which has $4^n$ entries: a 15-qubit density matrix needs the memory of a 30-qubit state vector.
- *Quantum trajectories* (Monte Carlo wavefunction) keep a state vector and, at each noise location, randomly apply one Kraus operator with the appropriate probability (renormalising afterwards). Averaging many trajectories reproduces the density-matrix result with memory $2^n$, at the cost of statistical error and repetition.
- *Pauli noise in stabilizer simulators*: for Clifford circuits with Pauli noise, error sampling stays polynomial, as in Stim. Non-Pauli noise is often approximated by its *Pauli twirl*, which keeps the average error rates but discards coherence between errors.

Device noise models are fitted from calibration data: $T_1$ and $T_2$ times per qubit, gate error rates from randomised benchmarking, readout confusion matrices, and crosstalk terms. They capture average behaviour well and rare correlated events (cosmic-ray bursts, leakage out of the qubit subspace) poorly.

== Choosing a Method

#table(
  columns: 3,
  [*Circuit structure*], [*Method*], [*Scales with*],
  [Small, arbitrary gates, exact amplitudes needed], [State vector], [$2^n$ memory and time],
  [Clifford gates and Pauli noise (error correction)], [Stabilizer tableau, Pauli frames], [Polynomial in $n$ and circuit size],
  [Clifford plus few $T$ gates], [Stabilizer rank], [Exponential in $T$ count],
  [Low entanglement, 1D, shallow], [Matrix product states], [Bond dimension $chi$],
  [Large but shallow or structured], [Tensor-network contraction], [Treewidth of the circuit graph],
  [Few gates across a qubit partition], [Schrödinger-Feynman hybrid], [Cross-cut gate count],
  [Small noisy circuits], [Density matrix or trajectories], [$4^n$, or $2^n$ times trajectories],
)

Common tools: Qiskit Aer and Cirq/qsim for state vectors and noise, NVIDIA cuQuantum for GPU state-vector and tensor-network kernels, Stim for stabilizer circuits, and quimb or ITensor for tensor networks.

== What Simulation Says About Advantage

A quantum computer can only offer an advantage on circuits that are hard for *every* classical method at once: enough qubits for state vectors to fail, enough non-Clifford gates for stabilizer methods to fail, enough entanglement and depth for tensor networks to fail, and low enough noise that the output is not approximable by a classically simulable noisy process. Noise is subtle here: as circuit depth grows at a fixed error rate, the output of random circuits approaches the uniform distribution, which is trivial to sample, and efficient classical algorithms have been proposed for noisy random circuit sampling in some regimes. Demonstrations of advantage are therefore races against classical simulation, and the durable path is fault tolerance, after which the quantum computer can run circuits of any depth.

== Further Reading

Aaronson, S., Gottesman, D. (2004). "Improved Simulation of Stabilizer Circuits." Physical Review A.

Gidney, C. (2021). "Stim: A Fast Stabilizer Circuit Simulator." Quantum.

Häner, T., Steiger, D. (2017). "0.5 Petabyte Simulation of a 45-Qubit Quantum Circuit." SC17.

Bravyi, S., Gosset, D. (2016). "Improved Classical Simulation of Quantum Circuits Dominated by Clifford Gates." Physical Review Letters.

Markov, I., Shi, Y. (2008). "Simulating Quantum Computation by Contracting Tensor Networks." SIAM Journal on Computing.

Vidal, G. (2003). "Efficient Classical Simulation of Slightly Entangled Quantum Computations." Physical Review Letters.

Pan, F., Zhang, P. (2022). "Simulation of Quantum Circuits Using the Big-Batch Tensor Network Method." Physical Review Letters.

Arute, F. et al. (2019). "Quantum Supremacy Using a Programmable Superconducting Processor." Nature.

Dalibard, J., Castin, Y., Mølmer, K. (1992). "Wave-Function Approach to Dissipative Processes in Quantum Optics." Physical Review Letters.

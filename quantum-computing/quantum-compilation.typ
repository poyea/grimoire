#import "../template.typ": xref

= Compilation and the Quantum Software Stack <quantum-compilation>

A quantum algorithm on paper uses arbitrary gates between arbitrary qubits. A real device offers a small set of calibrated native gates, connections between only some pairs of qubits, error rates that differ from qubit to qubit and from day to day, and a limited time before the qubits decohere. The quantum compiler bridges the two, and on noisy hardware its quality often decides whether a computation succeeds at all. This chapter covers the software stack and its intermediate representations, gate synthesis from single-qubit Euler angles to fault-tolerant Clifford+T sequences, circuit optimisation, qubit mapping and routing, scheduling and pulses, dynamic circuits, and the error-mitigation techniques that stretch noisy hardware further.

*See also:* #xref("quantum-computing", "qubits-and-gates", label: "Qubits and Gates") (universal gate sets, OpenQASM), #xref("quantum-computing", "hardware-architectures", label: "Hardware Architectures") (native gates and connectivity), #xref("quantum-computing", "nisq-and-benchmarking", label: "NISQ Devices and Benchmarking"), #xref("compilers", "ir-design", label: "IR Design") (classical compiler counterpart)

== The Stack

#table(
  columns: 2,
  [*Layer*], [*Examples*],
  [Applications and algorithm libraries], [Chemistry and optimisation packages, algorithm primitives],
  [Programming frameworks], [Qiskit, Cirq, CUDA-Q, Q\#, Guppy, PennyLane],
  [Circuit intermediate representations], [OpenQASM 3, QIR (LLVM-based), MLIR quantum dialects],
  [Optimisation, mapping, routing], [Qiskit transpiler, TKET, BQSKit, vendor compilers],
  [Native gates, scheduling, pulses], [Calibrated gate sets, OpenPulse-style pulse programs],
  [Control system], [FPGA-based waveform generation and real-time feedback],
)

The IRs reflect a shift from static circuits to programs. OpenQASM 3 adds classical types, control flow and timing to the circuit language. QIR expresses quantum programs as LLVM IR with calls to quantum intrinsics, so that classical compiler infrastructure can optimise the classical parts and hybrid programs can be compiled as a unit. As in classical compilers, most optimisation happens on a graph IR (a directed acyclic graph of gates on qubit wires) rather than on the source text.

== Gate Synthesis

=== Single-Qubit Gates

Any single-qubit unitary can be written, up to a global phase, as three rotations about two fixed axes. The *ZYZ decomposition* is

$ U = e^(i alpha) R_z(beta) R_y(gamma) R_z(delta) $

Superconducting devices make this especially cheap: $R_z$ is implemented *virtually* by shifting the phase reference of all later pulses on that qubit (McKay et al., 2017), so it is exact and takes no time. Every single-qubit gate then costs at most two physical $sqrt(X)$ pulses with virtual $Z$ rotations between them.

```cpp
#include <cmath>
#include <complex>

using cplx = std::complex<double>;

struct zyz_angles {
    double alpha, beta, gamma, delta;   // U = e^{i alpha} Rz(beta) Ry(gamma) Rz(delta)
};

// Decompose a 2x2 unitary [[u00, u01], [u10, u11]].
// Conventions: Rz(t) = diag(e^{-it/2}, e^{it/2}), Ry(t) = [[c, -s], [s, c]] with
// c = cos(t/2), s = sin(t/2).
zyz_angles decompose_zyz(cplx u00, cplx u01, cplx u10, cplx u11) {
    const cplx det = u00 * u11 - u01 * u10;
    const double alpha = std::arg(det) / 2.0;
    const cplx phase = std::polar(1.0, -alpha);           // V = e^{-i alpha} U is in SU(2)
    const cplx v00 = u00 * phase;
    const cplx v10 = u10 * phase;
    const cplx v11 = u11 * phase;

    const double gamma = 2.0 * std::atan2(std::abs(v10), std::abs(v00));
    constexpr double eps = 1e-12;
    double sum;    // beta + delta
    double diff;   // beta - delta
    if (std::abs(v10) < eps) {            // diagonal: only beta + delta is defined
        sum = 2.0 * std::arg(v11);
        diff = 0.0;
    } else if (std::abs(v00) < eps) {     // anti-diagonal: only beta - delta is defined
        sum = 0.0;
        diff = 2.0 * std::arg(v10);
    } else {
        sum = 2.0 * std::arg(v11);        // v11 = e^{i(beta+delta)/2} cos(gamma/2)
        diff = 2.0 * std::arg(v10);       // v10 = e^{i(beta-delta)/2} sin(gamma/2)
    }
    return {alpha, (sum + diff) / 2.0, gamma, (sum - diff) / 2.0};
}
```

Choosing $alpha$ from the determinant leaves a sign ambiguity ($V$ and $-V$ are both in SU(2)), which the decomposition absorbs into $beta$ and $delta$ by $2 pi$ shifts; either choice reproduces $U$ exactly.

=== Two-Qubit Gates

The *KAK* (Cartan) decomposition writes any two-qubit unitary as single-qubit gates on each side of an interaction $exp(i(a X X + b Y Y + c Z Z))$. It follows that every two-qubit gate needs at most *three* CNOTs plus single-qubit gates, and that three are necessary in general (Vatan and Williams, 2004). The coefficients $(a, b, c)$ also tell the compiler how many entangling gates a particular unitary needs (zero, one, two or three), which is the basis of resynthesis: collect a block of gates acting on the same two qubits, compute its unitary, and re-emit it with the minimum number of native entangling gates. Different hardware has different native entanglers (CNOT, CZ, iSWAP-family gates, Mølmer-Sørensen $X X$ interactions on trapped ions), and the decomposition targets whichever is native.

Arbitrary $n$-qubit unitaries need on the order of $4^n$ two-qubit gates, so efficient algorithms are always expressed in structured gates, never as generic unitaries.

=== Fault-Tolerant Gate Sets

Error-corrected machines implement Clifford gates cheaply and the $T$ gate expensively, through magic-state distillation (#xref("quantum-computing", "error-correction", label: "Quantum Error Correction")). Compilation for them therefore minimises *T-count* and *T-depth*:

- An arbitrary rotation $R_z(theta)$ must be *approximated* with Clifford and $T$ gates. The Solovay-Kitaev theorem guarantees polylogarithmic length in the precision $epsilon$; number-theoretic methods (Ross and Selinger, 2016) find optimal sequences with about $3 log_2(1 \/ epsilon)$ $T$ gates, around 60 to 100 $T$ gates for precisions algorithms typically need.
- *Phase-polynomial* and *ZX-calculus* optimisations (PyZX, for example) rewrite circuits of CNOT and phase gates to merge and cancel $T$ gates, often reducing T-count by tens of percent.
- Resource estimates for fault-tolerant algorithms are dominated by these counts, which is why compilers report them alongside qubit counts.

== Circuit Optimisation

Before mapping to hardware, compilers apply local rewrites familiar from classical peephole optimisation:

- *Cancellation* of inverse pairs ($H H$, CNOT CNOT, $S S^dagger$), including across gates that commute with them.
- *Rotation merging*: consecutive rotations about the same axis add; rotations commuting through CNOT controls (for $Z$) or targets (for $X$) can be moved to meet and merge.
- *Template matching*: replace a subcircuit with a cheaper equivalent from a library of identities.
- *Block resynthesis*: collect runs of gates on one or two qubits and resynthesise them optimally, as above.

Optimisation must preserve the circuit's semantics exactly, and *equivalence checking* (comparing unitaries for small circuits, or using decision diagrams and ZX rewriting for large ones) verifies compiler output in testing.

== Mapping and Routing

Most hardware connects each qubit to only a few neighbours: IBM's heavy-hexagon lattice gives each qubit two or three neighbours, Google's devices use a square grid with up to four. A two-qubit gate between non-adjacent qubits requires moving quantum states together with SWAP gates, each costing three CNOTs, and every extra gate adds error. (Trapped-ion and some neutral-atom systems offer all-to-all or reconfigurable connectivity, which largely removes this problem at the cost of slower gates.)

The compiler must:

1. Choose an *initial placement* of logical qubits on physical qubits, preferring well-calibrated qubits and placing frequently interacting pairs close together.
2. *Route*: insert SWAPs so that every two-qubit gate acts on adjacent qubits when it executes.

Finding the minimum number of SWAPs is NP-hard, so compilers use heuristics. *SABRE* (Li, Ding and Xie, 2019) is the most widely used: it processes the *front layer* of gates whose dependencies are satisfied, scores each candidate SWAP by how much it reduces the distance of front-layer gates (with a look-ahead term for the next gates and a decay term that discourages repeatedly swapping the same qubits), applies the best, and repeats. Running it forwards and backwards over the circuit refines the initial placement. *Noise-aware* variants weight distances by gate error rates, so routes avoid poor couplers. On typical benchmark circuits, routing adds tens of percent to the two-qubit gate count, and bad routing can easily double it.

== Scheduling, Pulses, and Idle Time

After routing, gates are scheduled in time. Qubits that sit idle still decohere, and on superconducting devices they also accumulate coherent phase errors from residual couplings. Two techniques reduce the damage:

- *Scheduling* gates as late as possible (ALAP) shortens the time qubits spend initialised but unused, and avoiding simultaneous gates on neighbouring qubits reduces crosstalk.
- *Dynamical decoupling* inserts sequences of $pi$ pulses (such as $X X$ or $X Y X Y$) into idle periods. Like a spin echo, they refocus slowly varying noise and residual interactions, and compilers add them automatically.

Below the gate level, each native gate is a calibrated microwave or laser pulse. Pulse-level access lets experts shape pulses (DRAG pulses suppress leakage to higher energy levels in transmons), implement custom gates directly, or stretch pulses for error mitigation. Most users never see this layer, and several vendors have narrowed public pulse access in favour of fractional and parameterised native gates.

== Dynamic Circuits

Error correction, teleportation-based protocols, and some algorithms need *mid-circuit measurement* and *feed-forward*: measure some qubits, compute on the results classically, and choose later gates accordingly, all within the qubits' coherence time. That makes the control system part of the computer: results must travel from the readout electronics to a classical processor and back to the pulse generators in well under a microsecond on superconducting hardware. Compilers for dynamic circuits must schedule classical computation alongside quantum operations, and the latency of that loop limits what can be done. QIR and OpenQASM 3 exist largely to express such programs.

== Error Mitigation

Error *correction* removes errors at the cost of many physical qubits per logical qubit. Error *mitigation* instead estimates what the noiseless result would have been, from many runs of noisy circuits. It reduces bias in expectation values, never corrects individual runs, and always costs extra samples, often exponentially many in circuit size:

- *Readout mitigation*: measure the confusion matrix (the probability of reading each bit string given each prepared one, usually assumed to factor per qubit) and invert its effect on measured distributions.
- *Zero-noise extrapolation* (ZNE): run the circuit at amplified noise levels (stretching pulses, or *gate folding*, replacing $G$ by $G G^dagger G$), measure the expectation value at each level, and extrapolate to zero noise. IBM's 2023 "utility" experiment on 127 qubits relied on ZNE with a learned noise model.
- *Probabilistic error cancellation* (PEC): learn the noise model, write the inverse of each noisy gate as a quasi-probability mixture of implementable operations, and sample from it with signs. It is unbiased under an accurate noise model, but the sampling overhead grows as $gamma^2$ per gate with $gamma > 1$, which is exponential in circuit size.
- *Pauli twirling* (randomised compiling): surround gates with random Pauli gates chosen so that the ideal circuit is unchanged. Averaging over the randomisations converts coherent errors into stochastic Pauli noise, which accumulates more slowly and is what ZNE and PEC models assume.
- *Symmetry verification*: discard runs that violate a known conserved quantity, such as particle number in chemistry simulations.

Mitigation extends what NISQ devices can do, but its exponential costs mean it cannot replace error correction for large computations.

== Further Reading

Nielsen, M., Chuang, I. (2010). "Quantum Computation and Quantum Information." Cambridge University Press, 10th anniversary ed., chapter 4.

Vatan, F., Williams, C. (2004). "Optimal Quantum Circuits for General Two-Qubit Gates." Physical Review A.

Dawson, C., Nielsen, M. (2006). "The Solovay-Kitaev Algorithm." Quantum Information and Computation.

Ross, N., Selinger, P. (2016). "Optimal Ancilla-Free Clifford+T Approximation of z-Rotations." Quantum Information and Computation.

Li, G., Ding, Y., Xie, Y. (2019). "Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices." ASPLOS.

Cross, A. et al. (2022). "OpenQASM 3: A Broader and Deeper Quantum Assembly Language." ACM Transactions on Quantum Computing.

McKay, D., Wood, C., Sheldon, S., Chow, J., Gambetta, J. (2017). "Efficient Z Gates for Quantum Computing." Physical Review A.

Kissinger, A., van de Wetering, J. (2020). "PyZX: Large Scale Automated Diagrammatic Reasoning." QPL, EPTCS 318.

Temme, K., Bravyi, S., Gambetta, J. (2017). "Error Mitigation for Short-Depth Quantum Circuits." Physical Review Letters.

Wallman, J., Emerson, J. (2016). "Noise Tailoring for Scalable Quantum Computation via Randomized Compiling." Physical Review A.

Kim, Y. et al. (2023). "Evidence for the Utility of Quantum Computing before Fault Tolerance." Nature.

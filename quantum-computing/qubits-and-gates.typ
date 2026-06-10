= Qubits and Gates

A qubit is a unit vector in $bb(C)^2$, the simplest non-trivial quantum system. Where a classical bit chooses one of two values, a qubit superposes them — and when you compose $n$ qubits via tensor product, the state space grows to $bb(C)^(2^n)$, the source of every quantum speedup and every implementation headache. This chapter develops the linear-algebraic foundation: state vectors, the Bloch sphere, single- and multi-qubit gates, measurement, and the universal gate sets that quantum software stacks compile to.

*See also:* _Quantum Algorithms_, _Quantum Software Stack_, _Hardware Architectures_, _Quantum Error Correction_, _Post-Quantum Cryptography_ (cryptography-and-security), _Complexity Theory_ (programming-languages)

== State Vectors and the Bloch Sphere

A single-qubit pure state is

$ |psi angle.r = alpha |0 angle.r + beta |1 angle.r, quad alpha, beta in bb(C), quad |alpha|^2 + |beta|^2 = 1. $

Global phase is unobservable: $e^(i phi) |psi angle.r$ and $|psi angle.r$ give identical measurement statistics. Modulo global phase, any pure single-qubit state is parameterized by two real angles $(theta, phi)$:

$ |psi angle.r = cos(theta/2) |0 angle.r + e^(i phi) sin(theta/2) |1 angle.r. $

This is the *Bloch sphere* parameterization.

```
                  |0>
                   |
                   |   .  |psi>
                   |  /
                   | / theta
                   |/____________ y
                  /
                 / phi
                /
               x
                   |
                  |1>
```

The poles are $|0 angle.r$ and $|1 angle.r$. The equator carries the superposition states $(|0 angle.r + e^(i phi)|1 angle.r) / sqrt(2)$ — including $|+ angle.r$, $|- angle.r$, $|"+i" angle.r$, $|"-i" angle.r$. Mixed states (statistical ensembles) live *inside* the ball as density matrices $rho = (II + arrow(r) dot arrow(sigma)) / 2$ with $|arrow(r)| <= 1$.

== Single-Qubit Gates

Unitary $2 times 2$ matrices act on the Bloch sphere as rotations of $S^2$ (SU(2) double-covers SO(3)).

#table(
  columns: (auto, auto, auto),
  [*Gate*], [*Matrix*], [*Action*],
  [$X$ (NOT)], [$mat(0, 1; 1, 0)$], [bit flip; $pi$ about $hat(x)$],
  [$Y$], [$mat(0, -i; i, 0)$], [$pi$ about $hat(y)$],
  [$Z$], [$mat(1, 0; 0, -1)$], [phase flip; $pi$ about $hat(z)$],
  [$H$ (Hadamard)], [$1/sqrt(2) mat(1, 1; 1, -1)$], [basis change $Z arrow.l.r X$],
  [$S$], [$mat(1, 0; 0, i)$], [$pi/2$ about $hat(z)$],
  [$T$], [$mat(1, 0; 0, e^(i pi / 4))$], [$pi/4$ about $hat(z)$],
)

Parameterized rotations:

$ R_x(theta) = e^(-i theta X / 2), quad R_y(theta) = e^(-i theta Y / 2), quad R_z(theta) = e^(-i theta Z / 2). $

Any single-qubit unitary decomposes (up to global phase) as $U = R_z(alpha) R_y(beta) R_z(gamma)$ — the *ZYZ decomposition*, central to gate synthesis.

```python
# Qiskit: build single-qubit states and inspect them
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
import numpy as np

qc = QuantumCircuit(1)
qc.h(0)            # |+>
qc.t(0)            # rotate to magic-state direction
sv = Statevector(qc)
print(sv.data)     # [0.7071, 0.5+0.5j]

# Verify HTH equals an Rx rotation up to global phase
U = Operator(QuantumCircuit(1).compose(qc)).data
print(np.round(U, 3))
```

== Multi-Qubit States and Entanglement

For $n$ qubits the state is a unit vector in $bb(C)^(2^n)$ written in the computational basis:

$ |psi angle.r = sum_(x in {0,1}^n) c_x |x angle.r, quad sum_x |c_x|^2 = 1. $

A state is *product* if $|psi angle.r = |a angle.r times.circle |b angle.r$ and *entangled* otherwise. The canonical Bell pair

$ |Phi^+ angle.r = (|00 angle.r + |11 angle.r) / sqrt(2) $

has no product decomposition: measuring the first qubit instantly fixes the second, regardless of separation. The four Bell states form an orthonormal basis of $bb(C)^4$ used in teleportation, superdense coding, and entanglement-based QKD (BB84/E91).

== Two-Qubit Gates

The *controlled-NOT* (CNOT, CX) is the workhorse:

$ "CNOT" = mat(1, 0, 0, 0; 0, 1, 0, 0; 0, 0, 0, 1; 0, 0, 1, 0). $

It flips the target iff the control is $|1 angle.r$. Equivalently, in the $|+ angle.r, |- angle.r$ basis the roles swap (CNOT in $Z$-basis equals CZ-with-Hadamards). CZ, iSWAP, and $sqrt("SWAP")$ are the other common natives — your hardware picks one. CNOT plus arbitrary single-qubit gates is *universal*: any unitary on $n$ qubits can be approximated to arbitrary precision.

```python
# Bell pair preparation in Qiskit
from qiskit import QuantumCircuit
bell = QuantumCircuit(2, 2)
bell.h(0)
bell.cx(0, 1)
bell.measure([0,1], [0,1])
# After many shots: ~50% '00', ~50% '11', never '01' or '10'.
```

== Measurement and the Born Rule

A projective measurement in the computational basis projects $|psi angle.r = sum c_x |x angle.r$ onto outcome $x$ with probability $|c_x|^2$, collapsing the state to $|x angle.r$. More generally, a *POVM* ${E_m}$ with $sum_m E_m = II$ yields outcome $m$ with $P(m) = angle.l psi | E_m | psi angle.r$.

The *no-cloning theorem* (Wootters & Zurek 1982) forbids a unitary $U$ with $U(|psi angle.r times.circle |0 angle.r) = |psi angle.r times.circle |psi angle.r$ for arbitrary $|psi angle.r$ — measurement is destructive in a fundamental way, which is why classical debugging tools do not port to quantum directly.

== Universal Gate Sets and Solovay-Kitaev

A *universal* gate set generates a dense subgroup of $U(2^n)$. Common choices:

#table(
  columns: (auto, auto),
  [*Set*], [*Notes*],
  [${H, T, "CNOT"}$], [Clifford+T; standard for fault-tolerant compilation],
  [${H, S, "CNOT"} + T$], [Clifford group is *not* universal alone — Gottesman-Knill],
  [${"Toffoli", H}$], [Classical reversible + one quantum gate],
  [${R_x(theta), R_z(theta), "CNOT"}$], [Continuous, used in NISQ variational circuits],
)

The *Solovay-Kitaev theorem* says any unitary can be approximated to error $epsilon$ using $O(log^c (1/epsilon))$ gates from a universal discrete set, with $c approx 3.97$ classically and $c approx 1$ with Ross-Selinger optimal $T$-count synthesis.

```python
# Cirq: decompose a parameterized rotation into Clifford+T
import cirq
q = cirq.LineQubit.range(1)
circuit = cirq.Circuit(cirq.rz(0.1)(q[0]))
decomposed = cirq.decompose(circuit, keep=lambda op: op.gate in
    {cirq.H, cirq.T, cirq.T**-1, cirq.S, cirq.X, cirq.Y, cirq.Z})
```

== Clifford Group and Stabilizer Formalism

The *Clifford group* ${H, S, "CNOT"}$ maps Pauli operators to Pauli operators under conjugation. The *Gottesman-Knill theorem* shows Clifford circuits with stabilizer-state inputs and Pauli measurements can be simulated in polynomial time classically — they are not the source of quantum speedup. The non-Clifford $T$ gate (or any magic state) is what crosses the boundary into BQP-hard regimes.

Stabilizer states are described compactly by their *stabilizer group*: an abelian subgroup of the Pauli group whose joint $+1$-eigenspace is the state. This formalism underpins error correction (Chapter on _Error Correction_) and lets simulators handle thousands of qubits when restricted to Cliffords.

== OpenQASM 3 Example

```
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;

h q[0];
cx q[0], q[1];
rz(pi/4) q[1];
cx q[0], q[1];
h q[0];

c = measure q;
```

OpenQASM 3 adds classical control, timing, and gate definitions over 2.x — every major stack (Qiskit, t|ket angle.r, Cirq via interop) ingests it.

== Further Reading

Nielsen, M., Chuang, I. (2010). _Quantum Computation and Quantum Information_, 10th anniv. ed. Cambridge.

Preskill, J. _Lecture Notes on Quantum Computation_, Ph229, Caltech.

Wootters, W., Zurek, W. (1982). "A Single Quantum Cannot Be Cloned." Nature.

Gottesman, D. (1998). "The Heisenberg Representation of Quantum Computers." arXiv:quant-ph/9807006.

Dawson, C., Nielsen, M. (2006). "The Solovay-Kitaev Algorithm." Quantum Inf. Comput.

Ross, N., Selinger, P. (2016). "Optimal Ancilla-Free Clifford+T Approximation of $z$-Rotations." Quantum Inf. Comput.

Cross, A. et al. (2022). "OpenQASM 3: A Broader and Deeper Quantum Assembly Language." ACM TQC.

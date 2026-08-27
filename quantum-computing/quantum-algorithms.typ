#import "../template.typ": xref

= Quantum Algorithms <quantum-algorithms>

Quantum algorithms exploit interference — amplitudes can cancel, not just add — to make wrong answers destructive and right answers constructive. The canonical examples (Deutsch-Jozsa, Grover, Shor) demonstrate this on structured problems; the variational family (VQE, QAOA) trades provable speedup for near-term tractability on noisy hardware. This chapter walks through each with explicit circuits and complexity bounds.

*See also:* #xref("quantum-computing", "qubits-and-gates", label: "Qubits and Gates"), #xref("quantum-computing", "nisq-and-benchmarking", label: "NISQ Devices and Benchmarking"), #xref("cryptography-and-security", "post-quantum", label: "Post-Quantum Cryptography") (cryptography-and-security), #xref("programming-languages", "complexity", label: "Complexity Theory") (programming-languages)

== Deutsch-Jozsa: The First Speedup

Given $f: {0,1}^n -> {0,1}$ promised to be either *constant* or *balanced* (half zeros, half ones), determine which. Classically, the worst case requires $2^(n-1) + 1$ queries; Deutsch-Jozsa answers with a *single* quantum query.

Circuit: prepare $|0 chevron.r^(times.o n) |1 chevron.r$, apply $H^(times.o n+1)$, query the phase oracle $U_f: |x chevron.r |y chevron.r -> |x chevron.r |y plus.o f(x) chevron.r$, apply $H^(times.o n)$ to the first register, measure. The output is $|0 chevron.r^n$ iff $f$ is constant.

```python
# Qiskit Deutsch-Jozsa for a balanced oracle f(x) = x_0
from qiskit import QuantumCircuit
n = 3
qc = QuantumCircuit(n + 1, n)
qc.x(n); qc.h(range(n + 1))
qc.cx(0, n)                  # phase oracle: f(x) = x_0
qc.h(range(n))
qc.measure(range(n), range(n))
```

The lesson: phase kickback turns a function-evaluation oracle into a *phase* oracle, and a final Hadamard converts phases to amplitudes the computational-basis measurement can read off.

== Grover Search

Grover (1996) finds a marked element in an unstructured list of $N$ items with $O(sqrt(N))$ oracle queries — a *quadratic* speedup, provably tight (BBBV lower bound, 1997).

The algorithm iterates the *Grover operator* $G = (2|s chevron.r chevron.l s| - II) O_f$, where $|s chevron.r$ is the uniform superposition and $O_f$ phase-flips marked states. Geometrically, $G$ rotates $|s chevron.r$ toward the target by $2 theta$ per step where $sin theta = sqrt(M/N)$. Optimal iteration count: $k^* = floor(pi/4 sqrt(N/M))$.

```python
# Grover for one marked item in n=3 qubits (N=8), marked = |101>
from qiskit import QuantumCircuit
from math import pi, sqrt, floor
n = 3
qc = QuantumCircuit(n, n)
qc.h(range(n))

iters = floor(pi/4 * sqrt(2**n))
for _ in range(iters):
    # Oracle: flip phase of |101>
    qc.x(1)                      # flip middle bit so |101> -> |111>
    qc.h(n-1); qc.mcx(list(range(n-1)), n-1); qc.h(n-1)
    qc.x(1)
    # Diffuser: 2|s><s| - I
    qc.h(range(n)); qc.x(range(n))
    qc.h(n-1); qc.mcx(list(range(n-1)), n-1); qc.h(n-1)
    qc.x(range(n)); qc.h(range(n))

qc.measure(range(n), range(n))
```

*Amplitude amplification* generalizes Grover: any quantum subroutine producing the right answer with probability $p$ can be boosted to near 1 in $O(1/sqrt(p))$ repetitions, vs $O(1/p)$ classically. This is the engine behind quantum walks, collision finding, and many NP-search heuristics.

== Quantum Fourier Transform

The QFT on $n$ qubits implements

$ "QFT" |x chevron.r = 1 / sqrt(2^n) sum_(y=0)^(2^n - 1) e^(2 pi i x y / 2^n) |y chevron.r. $

The classical FFT costs $O(N log N)$ time; the QFT uses $O(n^2) = O(log^2 N)$ gates. The catch: the output is encoded in *amplitudes*, not values you can read out — QFT is useful as a subroutine that feeds into phase estimation, not as a black-box DFT replacement.

== Phase Estimation

Given a unitary $U$ and an eigenstate $|u chevron.r$ with $U |u chevron.r = e^(2 pi i phi) |u chevron.r$, *quantum phase estimation* (QPE) extracts $phi$ to $t$ bits using $t$ ancillae, $2^t - 1$ controlled-$U$ applications, and an inverse QFT.

QPE is the workhorse behind Shor, HHL, quantum chemistry energy estimation, and quantum-enhanced Monte Carlo.

== Shor's Factoring Algorithm

Shor (1994) factors an $n$-bit integer $N$ in $tilde(O)(n^3)$ quantum time by reducing factoring to *order finding*: find the smallest $r$ with $a^r equiv 1 mod N$. Classical reduction handles the rest (via $gcd(a^(r/2) plus.minus 1, N)$).

Order-finding uses QPE on $U_a |y chevron.r = |a y mod N chevron.r$. The output bits, post-processed with continued fractions, yield $r$ with constant probability.

```
        [t ancillae] -- H^t ----o------o----o----- QFT^-1 -- measure
                                |      |    |
        [n target]   --|1>----- U^1 -- U^2--U^4 -- ...  (modular exp.)
```

Cost: $O(n^2 log n log log n)$ modular multiplications via fast arithmetic, total $tilde(O)(n^3)$. Compare best-known classical: GNFS at $exp(O(n^(1/3) log^(2/3) n))$. A cryptographically relevant 2048-bit RSA key requires *millions* of physical qubits at current error rates (Gidney & Ekerå 2021: 20 million noisy qubits, 8 hours) — see _Error Correction_.

== HHL: Quantum Linear Systems

Harrow-Hassidim-Lloyd (2009): given a sparse $s$-sparse, well-conditioned (condition number $kappa$) $N times N$ matrix $A$ and $|b chevron.r$ encoded as a state, produce $|x chevron.r prop A^(-1) |b chevron.r$ in $tilde(O)(log(N) s^2 kappa^2 / epsilon)$ time vs classical $O(N s sqrt(kappa) log(1/epsilon))$ with conjugate gradient.

*Caveats* (Aaronson 2015): (1) state preparation of $|b chevron.r$ may itself be expensive; (2) you cannot read out $|x chevron.r$ entry-wise without paying $O(N)$; (3) only specific functionals $chevron.l x | M | x chevron.r$ are extractable cheaply. HHL gives exponential speedup only when these match the application.

== Variational Quantum Eigensolver (VQE)

NISQ-era algorithm: minimize $E(theta) = chevron.l psi(theta) | H | psi(theta) chevron.r$ over a parameterized ansatz $|psi(theta) chevron.r$ using a classical optimizer. Decompose $H = sum_i c_i P_i$ into Pauli strings, measure each term, aggregate.

```python
# Qiskit VQE for H2 (sketch)
from qiskit.primitives import Estimator
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA

ansatz = EfficientSU2(num_qubits=4, reps=2)
vqe = VQE(estimator=Estimator(), ansatz=ansatz, optimizer=SPSA(maxiter=100))
# result = vqe.compute_minimum_eigenvalue(operator=H_pauli)
```

Open problems: *barren plateaus* (McClean 2018) — gradients vanish exponentially in qubit count for random ansätze — and the absence of provable speedup over classical heuristics like DMRG for most chemistry instances.

== QAOA: Quantum Approximate Optimization

Farhi-Goldstone-Gutmann (2014): for a combinatorial cost $C(z) = sum_alpha C_alpha (z)$ encoded as $H_C$, alternate $e^(-i gamma H_C)$ and $e^(-i beta H_M)$ with mixing $H_M = sum_i X_i$ for $p$ layers; optimize $(arrow(gamma), arrow(beta))$ classically.

$p = 1$ QAOA on MaxCut achieves approximation ratio 0.6924 on 3-regular graphs — beaten by Goemans-Williamson's 0.878. Increasing $p$ improves the bound; the open question is whether QAOA at constant $p$ provably beats classical for *any* natural problem. The Stilck França-Garcia-Patrón (2021) results suggest noise quickly destroys the advantage on near-term hardware.

== Algorithm Cost Comparison

#table(
  columns: (auto, auto, auto, auto),
  [*Algorithm*], [*Quantum*], [*Classical*], [*Speedup*],
  [Deutsch-Jozsa], [$1$ query], [$Theta(2^n)$ worst case], [Exponential (oracle)],
  [Grover], [$O(sqrt(N))$], [$Theta(N)$], [Quadratic (provably tight)],
  [Shor], [$tilde(O)(n^3)$], [$exp(O(n^(1/3) log^(2/3) n))$], [Superpolynomial],
  [HHL], [$tilde(O)(log N kappa^2 s^2 / epsilon)$], [$O(N sqrt(kappa) s)$], [Exp. (with caveats)],
  [Simon], [$O(n)$], [$Omega(2^(n/2))$], [Exponential (oracle)],
  [VQE/QAOA], [Heuristic], [Heuristic], [Empirical / unproven],
)

== Quantum Walks and Element Distinctness

Ambainis (2004): element distinctness on $N$ items needs $Theta(N^(2/3))$ quantum queries via a quantum walk on the Johnson graph — improving on the earlier $O(N^(3/4))$ Grover-based bound. The same quantum-walk framework relates to BHT collision finding ($O(N^(1/3))$, Brassard-Høyer-Tapp 1997) and triangle finding speedups; relevant to symmetric-cryptography post-quantum security margins.

== Further Reading

Deutsch, D., Jozsa, R. (1992). "Rapid Solution of Problems by Quantum Computation." Proc. Roy. Soc.

Grover, L. (1996). "A Fast Quantum Mechanical Algorithm for Database Search." STOC.

Bennett, C., Bernstein, E., Brassard, G., Vazirani, U. (1997). "Strengths and Weaknesses of Quantum Computing." SIAM J. Comput.

Shor, P. (1994). "Algorithms for Quantum Computation: Discrete Logarithms and Factoring." FOCS.

Harrow, A., Hassidim, A., Lloyd, S. (2009). "Quantum Algorithm for Linear Systems of Equations." Phys. Rev. Lett.

Aaronson, S. (2015). "Read the Fine Print." Nature Physics (HHL caveats).

Peruzzo, A. et al. (2014). "A Variational Eigenvalue Solver on a Photonic Quantum Processor." Nat. Commun.

Farhi, E., Goldstone, J., Gutmann, S. (2014). "A Quantum Approximate Optimization Algorithm." arXiv:1411.4028.

McClean, J. et al. (2018). "Barren Plateaus in Quantum Neural Network Training Landscapes." Nat. Commun.

Gidney, C., Ekerå, M. (2021). "How to Factor 2048-bit RSA Integers in 8 Hours Using 20 Million Noisy Qubits." Quantum.

Ambainis, A. (2007). "Quantum Walk Algorithm for Element Distinctness." SIAM J. Comput.

Stilck França, D., García-Patrón, R. (2021). "Limitations of Optimization Algorithms on Noisy Quantum Devices." Nat. Phys.

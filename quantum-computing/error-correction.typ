= Quantum Error Correction

Physical qubits decohere — typical superconducting $T_1, T_2$ sit at 100-300 $mu$s, two-qubit gate errors at $10^(-3)$. No useful algorithm fits in that window. *Quantum error correction* (QEC) encodes one logical qubit across many physical qubits so errors can be detected and reversed without measuring the encoded data. This chapter covers the stabilizer formalism, the surface code, magic-state distillation, and the resource estimates that determine when fault-tolerant quantum computing becomes practical.

*See also:* _Qubits and Gates_, _Hardware Architectures_, _Complexity_, _Post-Quantum Cryptography_ (cryptography-and-security)

== Why Classical Codes Do Not Port

You cannot copy a qubit (no-cloning) and you cannot measure it without collapsing it. Yet QEC works, via three observations:

1. *Discretize errors:* any single-qubit error decomposes as $a II + b X + c Y + d Z$ — measuring an error syndrome projects to one of these Paulis, which you then undo.
2. *Measure stabilizers, not data:* a stabilizer is an operator that leaves codewords invariant; measuring it reveals the error without disturbing the encoded state.
3. *Continuous to discrete:* tiny rotations get projected to either "no error" or a full Pauli flip — analog noise becomes digital.

== Stabilizer Codes

A $[[n, k, d]]$ stabilizer code encodes $k$ logical qubits in $n$ physical qubits with distance $d$. The codespace is the joint $+1$ eigenspace of an abelian subgroup $S subset cal(P)_n$ of the Pauli group, with $|S| = n - k$ independent generators. Errors $E$ with $E S E^dagger != S$ are detectable via syndrome measurement.

The *3-qubit bit-flip code* (encodes one logical against $X$ errors, not a true QEC):

$ |0 chevron.r_L = |000 chevron.r, quad |1 chevron.r_L = |111 chevron.r, $

stabilizers $S_1 = Z_1 Z_2, S_2 = Z_2 Z_3$.

The *Shor [[9,1,3]] code* (1995) combines bit- and phase-flip codes — first true QEC code. The *Steane [[7,1,3]] code* (1996) is a CSS code from the classical $[7,4,3]$ Hamming code, with the elegant property that transversal Cliffords work for free.

== The Surface Code

Kitaev (1997) and Bravyi-Kitaev (1998) introduced topological codes on a 2D lattice. The *surface code* places data qubits on edges (or vertices, in the rotated layout) of an $L times L$ patch; stabilizers are 4-body (weight-4) $X$ and $Z$ checks. Distance $d = L$, encoding 1 logical qubit in $approx d^2$ physical qubits.

```
   d---X---d---X---d
   |   |   |   |   |
   Z   d   Z   d   Z
   |   |   |   |   |
   d---X---d---X---d        d = data qubit
   |   |   |   |   |        X = X-stabilizer ancilla
   Z   d   Z   d   Z        Z = Z-stabilizer ancilla
   |   |   |   |   |
   d---X---d---X---d
```

*Threshold theorem:* below a physical error rate $p_("th") approx 0.7%$ (circuit-level), arbitrarily long computation is possible by increasing $d$. Logical error per round scales as $(p / p_("th"))^((d+1)/2)$ — exponential suppression in distance.

The surface code dominates because:

- *Planar 2D layout* matches superconducting hardware.
- *Local stabilizers* (weight 4) are realistic to measure.
- *Highest known threshold* among practical codes.
- *Lattice surgery* enables logical operations without long-range moves.

Google's 2023-2024 demonstrations (Acharya et al. 2023, Nature 614; Acharya et al. 2024, Nature 638) showed the first sub-threshold suppression: $d=3 -> d=5 -> d=7$ each cut logical error rate by $approx 2.1 times$ on Willow.

== Decoders

Syndrome data ${s_t}$ arrives every QEC cycle ($approx 1 mu$s on superconducting). A *decoder* infers the most likely error from the syndrome history, in real time.

#table(
  columns: (auto, auto, auto),
  [*Decoder*], [*Complexity*], [*Notes*],
  [Minimum-Weight Perfect Matching (MWPM)], [$O(n^3)$ naive, $O(n)$ amortized (sparse blossom)], [Standard baseline; optimal for independent $X$/$Z$],
  [Union-Find (Delfosse-Nickerson)], [Near-linear], [Sub-optimal but FPGA-friendly],
  [Belief-Propagation + OSD], [Polynomial], [Best for biased noise, qLDPC codes],
  [Neural-network decoders], [Inference-time], [Match BP-OSD on small codes],
  [Tensor-network (Bravyi et al.)], [Exponential in width], [Near-optimal benchmark],
)

Decoding latency must beat the syndrome cycle, or the backlog grows unboundedly — Terhal's *backlog problem*. Real-time MWPM on $d=11$ surface codes was demonstrated by Riverlane and Google in 2024.

== Color Codes

Bombin (2006). Triangular tiling with 3-colorable faces; each face hosts both $X$ and $Z$ stabilizers (higher weight, typically 6 or 8). Tradeoffs vs surface code:

#table(
  columns: (auto, auto, auto),
  [*Property*], [*Surface code*], [*Color code*],
  [Stabilizer weight], [4], [6 (2D) or 8 (3D)],
  [Threshold (circuit-level)], [$tilde 0.7%$], [$tilde 0.3%$],
  [Transversal Cliffords], [$"CNOT"$ only], [Full Clifford in 2D],
  [Transversal $T$], [No (in 2D)], [Yes (in 3D)],
  [Lattice surgery], [Mature], [More involved],
  [Hardware fits], [Superconducting], [Trapped-ion (Quantinuum 2024 [[7,1,3]])],
)

Quantinuum demonstrated repeated logical error correction on a [[7,1,3]] color code (2024) and a [[12,2,4]] code, leveraging trapped-ion all-to-all connectivity.

== qLDPC Codes

Surface codes need $d^2$ physical qubits per logical qubit; constant-overhead asymptotics are impossible in 2D (Bravyi-Poulin-Terhal 2010). *Quantum LDPC codes* break this barrier in higher-dimensional / nonlocal layouts:

- *Bicycle codes* (MacKay 2004): early constructions.
- *Hypergraph product* (Tillich-Zémor 2014): $[[n, k, d]]$ with $k, d = Theta(sqrt(n))$.
- *Lifted product / balanced product* (Panteleev-Kalachev 2022): achieve $k, d = Theta(n)$ — *good qLDPC codes*, settling a 20-year open problem.
- *Bivariate bicycle* (IBM 2024, Nature): $[[144, 12, 12]]$ code using $approx 12 times$ fewer qubits than the equivalent-distance surface code, at the cost of non-planar connectivity.

These require long-range couplers (photonic interconnects, reconfigurable atom arrays) — actively investigated as a way to cut overhead from millions to hundreds of thousands of physical qubits for cryptanalysis.

== Magic-State Distillation

Transversal gates are limited by Eastin-Knill (2009): no QEC code admits a *universal* set of transversal gates. The surface code has transversal Clifford only — non-Clifford $T$ requires *magic states*

$ |T chevron.r = (|0 chevron.r + e^(i pi / 4) |1 chevron.r) / sqrt(2). $

Inject a noisy $|T chevron.r$, then distill: Bravyi-Kitaev's 15-to-1 protocol takes 15 noisy magic states with error $p$ to one with error $approx 35 p^3$. Hierarchies of distillation factories dominate the resource cost of fault-tolerant quantum algorithms — typically 50-90% of qubits in a Shor-factoring layout.

```python
# Stim: simulate the 15-to-1 distillation circuit (Clifford-only, fast)
import stim
circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    distance=5, rounds=5,
    after_clifford_depolarization=0.001,
    after_reset_flip_probability=0.001,
    before_measure_flip_probability=0.001,
)
sampler = circuit.compile_detector_sampler()
syndromes = sampler.sample(shots=10_000)
```

Litinski's *game of surface code* (2019) reduced magic-state factory cost dramatically using lattice surgery and improved distillation routines; further improved by Gidney's *cultivation* protocol (2024).

== Resource Estimates

Cryptographically relevant Shor (2048-bit RSA), at physical error $10^(-3)$, code cycle $1 mu$s, surface-code distance $d=27$:

- $approx 20$M physical qubits.
- $approx 8$ hours wall-clock.
- Dominated by magic-state factories ($approx 80%$ qubits) and modular exponentiation.

Recent improvements (Gidney 2024, Litinski 2024) drop this by $3-5 times$ via cultivation and better factoring circuits. Bivariate-bicycle qLDPC alternatives drop physical-qubit overhead by another $5-10 times$ if photonic interconnects materialize.

== Further Reading

Shor, P. (1995). "Scheme for Reducing Decoherence in Quantum Computer Memory." Phys. Rev. A.

Steane, A. (1996). "Multiple-Particle Interference and Quantum Error Correction." Proc. Roy. Soc.

Calderbank, A., Shor, P. (1996); Steane, A. (1996). CSS codes.

Kitaev, A. (2003). "Fault-Tolerant Quantum Computation by Anyons." Ann. Phys. (surface code).

Bravyi, S., Kitaev, A. (1998). "Quantum Codes on a Lattice with Boundary." arXiv:quant-ph/9811052.

Fowler, A., Mariantoni, M., Martinis, J., Cleland, A. (2012). "Surface Codes: Towards Practical Large-Scale Quantum Computation." Phys. Rev. A.

Bombin, H. (2006). "Topological Quantum Distillation." Phys. Rev. Lett. (color codes).

Bravyi, S., Poulin, D., Terhal, B. (2010). "Tradeoffs for Reliable Quantum Information Storage in 2D Systems." Phys. Rev. Lett.

Panteleev, P., Kalachev, G. (2022). "Asymptotically Good Quantum and Locally Testable Classical LDPC Codes." STOC.

Bravyi, S. et al. (2024). "High-Threshold and Low-Overhead Fault-Tolerant Quantum Memory." Nature (IBM bivariate bicycle).

Google Quantum AI (2024). "Quantum Error Correction Below the Surface Code Threshold." Nature 638.

Bravyi, S., Kitaev, A. (2005). "Universal Quantum Computation with Ideal Clifford Gates and Noisy Ancillas." Phys. Rev. A (magic states).

Litinski, D. (2019). "A Game of Surface Codes." Quantum.

Gidney, C., Ekerå, M. (2021). "How to Factor 2048-bit RSA Integers in 8 Hours Using 20 Million Noisy Qubits." Quantum.

Gidney, C. (2024). "Magic-State Cultivation." arXiv:2409.17595.

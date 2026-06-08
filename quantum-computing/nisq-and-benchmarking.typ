= NISQ Devices and Benchmarking

John Preskill coined *NISQ* (Noisy Intermediate-Scale Quantum) in 2018 to describe the 50--1000 qubit devices available today: large enough that classical simulation is hard, small enough that error correction is impractical. Understanding what NISQ devices can and cannot do requires precise benchmarking and honest assessment of noise.

*See also:* _Hardware Architectures_, _Quantum Algorithms_, _Error Correction_

== The NISQ Regime

Preskill's definition draws a practical boundary: 50--1000 physical qubits, operated without full quantum error correction. This range is interesting precisely because it sits in a hard zone: representing an arbitrary state on $n > 50$ qubits requires $2^n$ complex amplitudes, making exact classical simulation intractable, yet the qubit count is too small to run the deep fault-tolerant circuits that would yield provable algorithmic advantage.

Key device parameters on leading superconducting platforms (2023--2025):

- *Single-qubit gate error:* $10^(-4)$--$10^(-3)$ (0.01--0.1%)
- *Two-qubit gate error:* $10^(-3)$--$10^(-2)$ (0.1--1%)
- *Relaxation time $T_1$:* 50--500 $mu$s (energy decay to ground state)
- *Dephasing time $T_2$:* 10--200 $mu$s (coherence loss without energy loss; $T_2 <= 2 T_1$ always)
- *Gate time:* 20--100 ns for two-qubit gates

A rough circuit depth limit before errors dominate follows from $d_max approx T_2 \/ t_"gate"$, giving $tilde 10^3$--$10^4$ layers in principle — but correlated noise, leakage, and crosstalk reduce this severely in practice. Useful NISQ circuits today are typically shallower than 100 layers.

Trapped-ion platforms achieve lower error rates ($tilde 10^(-4)$ for two-qubit gates) at the cost of slower gate times ($tilde 10$--$100$ $mu$s), giving comparable depth budgets.

== Noise Models

Real noise is complex; tractable models approximate it.

*Depolarizing channel.* With probability $p$, apply a uniformly random Pauli ($I, X, Y, Z$); with probability $1-p$ do nothing. For a single qubit:

$ cal(E)(rho) = (1 - p) rho + (p/4)(rho + X rho X + Y rho Y + Z rho Z). $

This is the standard model for gate benchmarks: it is symmetric, has no preferred axis, and is parameterized by a single number $p$.

*Amplitude damping.* Models spontaneous emission ($T_1$ decay). Kraus operators:

$ K_0 = mat(1, 0; 0, sqrt(1-gamma)), quad K_1 = mat(0, sqrt(gamma); 0, 0), quad gamma = 1 - e^(-t\/T_1). $

The excited state $|1 chevron.r$ decays to $|0 chevron.r$ with probability $gamma$; ground state is unaffected.

*Phase damping.* Models dephasing ($T_2$ process, without energy exchange):

$ K_0 = mat(1, 0; 0, sqrt(1-lambda)), quad K_1 = mat(0, 0; 0, sqrt(lambda)), quad lambda = 1 - e^(-t\/T_2). $

Off-diagonal elements of $rho$ (coherences) decay as $e^(-t\/T_2)$.

*Crosstalk.* Always-on $Z Z$ coupling between neighboring qubits causes idle qubits to accumulate conditional phase. This is especially problematic during long two-qubit gate sequences and manifests as context-dependent error rates — gate fidelity depends on what adjacent qubits are doing simultaneously.

*Measurement error.* Readout is typically the dominant single-shot error source, at 1--5% per qubit. A *confusion matrix* $M$ with $M_(i j) = Pr("read" i | "prepared" j)$ characterizes it; inverting $M$ (readout error mitigation) is cheap classically but limited by shot noise.

== Quantum Volume

IBM's Quantum Volume (Cross et al. 2019) is a single-number metric that captures gate fidelity, qubit connectivity, circuit compilation quality, and coherence time together.

*Definition.* For a device with $n$ qubits, run random $n$-qubit, $n$-layer square circuits (each layer is a random permutation followed by random $"SU"(4)$ gates on paired qubits). A circuit is *heavy* if its output bitstring has probability above the median of the ideal output distribution. The device achieves $"QV" = 2^n$ if the fraction of heavy outputs exceeds $2/3$ with at least $2 sigma$ confidence over many random circuits.

The threshold $2/3$ is non-trivial: for a perfect device the heavy-output probability is $approx 0.854$; noise degrades this toward $1/2$.

*Progress.* IBM reported QV~4 in 2019, QV~32 in 2020, QV~128 in 2021, and QV~512 in 2023 on Falcon/Eagle/Heron processor families.

*Criticism.* QV measures square circuits on the best $n$ qubits the device can identify. It does not capture performance on large circuits that span all qubits, and heavy connectivity (all-to-all or high degree) inflates QV without reflecting what sparse-connectivity hardware achieves on real workloads.

== Other Benchmarks

*Randomized Benchmarking (RB).* Apply random sequences of Clifford gates of increasing length $m$, followed by the inverse; measure survival probability. The decay

$ p(m) = A lambda^m + B $

gives average Clifford error per gate $r = (1 - lambda)(d-1)/d$ where $d = 2^n$. RB is insensitive to state preparation and measurement errors, and is the de facto standard for reporting gate fidelities. Extensions include Interleaved RB (error rate of a specific gate) and Character RB (non-Clifford gates).

*CLOPS (Circuit Layer Operations Per Second).* IBM's throughput metric: how many parametrized circuit layers can be executed and updated per second, including classical overhead. Relevant for variational algorithms where the optimizer must update parameters between runs.

*Mirror circuits (Proctor et al. 2021).* A scalable benchmark that constructs circuits with a built-in known answer: run a random circuit $C$, then its mirror $C^dagger$, and check that the output is the input state. Unlike RB, mirror circuits scale to large $n$ and catch *coherent* errors (systematic rotations) that depolarizing-noise models miss.

*Application-level benchmarks.* QASMbench (Li et al.) provides a suite of real algorithm circuits from chemistry, optimization, and machine learning. The QED-C benchmarks (Lubinski et al. 2023) define success conditions for application-relevant circuits and have been run across IBM, IonQ, Quantinuum, and Rigetti hardware.

== Quantum Supremacy Claims

*Google Sycamore (2019).* Arute et al. ran random circuit sampling on a 53-qubit device with 20 cycles of two-qubit gates. They estimated that sampling from the output distribution to error $epsilon$ would take 10,000 years on Summit (the then-fastest classical supercomputer), claiming "quantum supremacy."

*Classical counterarguments.* IBM researchers (2019) immediately noted that storing the full $2^{53}$ amplitude vector on disk avoids memory bottlenecks and cuts the estimate to $tilde 2.5$ days. Subsequent work by Pan, Chen, and Zhang (2022) using improved tensor-network contraction on classical clusters reduced the simulation cost further, to hours on a large HPC cluster with acceptable approximation error.

*Honest assessment.* Random circuit sampling is an artificial task with no known practical application. Classical simulation of NISQ-scale random circuits is a moving target as algorithms improve. The 2019 result demonstrated that a quantum device can execute circuits that are *hard to verify* classically, which is meaningful for device characterization but does not constitute practical quantum advantage.

== NISQ Algorithms and Their Limitations

*VQE.* The variational quantum eigensolver minimizes $E(arrow(theta)) = chevron.l psi(arrow(theta)) | H | psi(arrow(theta)) chevron.r$ over a parameterized state $|psi(arrow(theta)) chevron.r$. Two fundamental problems arise on NISQ hardware:

1. *Barren plateaus (McClean et al. 2018):* For hardware-efficient ansätze, the gradient $partial E \/ partial theta_k$ has variance that decays exponentially in $n$. Training requires exponentially many shots to resolve the signal from noise.

2. *Noise accumulation:* Even moderate circuit depth renders the energy estimate unreliable. Error mitigation techniques (zero-noise extrapolation, probabilistic error cancellation) reduce bias but increase shot overhead exponentially.

*QAOA.* At depth $p$, QAOA on MaxCut achieves approximation ratio that provably improves with $p$ but requires $p = Omega(n)$ to match classical Goemans-Williamson (ratio 0.878) on worst-case instances. Shallow ($p tilde 1$--$3$) QAOA is tractable on NISQ hardware but achieves ratios below 0.75 on hard instances — below what classical algorithms can guarantee.

*Quantum machine learning.* Claims of exponential speedup via quantum kernels or quantum neural networks typically depend on: (1) efficient quantum RAM (does not exist), (2) efficient state preparation (exponentially hard in general), and (3) the ability to read out results (sampling overhead). The *dequantization* results of Tang (2019) showed that classical algorithms with *sampling access* to data can match the quantum linear-algebra speedups of Kerenidis-Prakash and related quantum ML proposals.

== Path to Fault Tolerance

The surface code (Fowler et al. 2012) achieves a threshold of $tilde 1\%$ physical gate error: below this, increasing the code distance $d$ exponentially suppresses the logical error rate:

$ p_L approx (p \/ p_"th")^((d+1)\/2). $

At physical error rate $p = 0.1\%$ and threshold $p_"th" = 1\%$, distance $d = 7$ gives $p_L approx 10^{-9}$. The overhead is severe: each logical qubit requires $tilde 2 d^2$ physical qubits, so $d = 7$ costs $tilde 100$ physical qubits per logical qubit; for the $tilde 1000$ logical qubits and deep circuits needed to run Shor's algorithm on a 2048-bit key (Gidney & Ekerå 2021), the total physical qubit count is $tilde 20$ million.

Leading labs (IBM, Google, Microsoft, IonQ, Quantinuum) have announced roadmaps targeting $10^4$--$10^6$ physical qubits in the 2030--2035 timeframe. The first realistic fault-tolerant application is widely expected to be quantum simulation of strongly correlated chemistry (100+ orbitals) where classical methods such as DMRG and CCSD(T) hit polynomial walls; provable classical hardness for optimization remains undemonstrated.

Near-term (2025--2030) realistic expectations: error-mitigated VQE may provide marginal improvements over classical heuristics on small molecule problems; NISQ devices remain useful as experimental platforms for studying physics, not as practical compute accelerators.

== Further Reading

Preskill, J. (2018). "Quantum Computing in the NISQ Era and Beyond." Quantum 2, 79.

Cross, A. et al. (2019). "Validating Quantum Computers Using Randomized Model Circuits." Phys. Rev. A 100, 032328.

Arute, F. et al. (Google) (2019). "Quantum Supremacy Using a Programmable Superconducting Processor." Nature 574, 505--510.

Pan, F., Chen, K., Zhang, P. (2022). "Solving the Sampling Problem of the Sycamore Quantum Circuits." Phys. Rev. Lett. 129, 090502.

Tang, E. (2019). "A Quantum-Inspired Classical Algorithm for Recommendation Systems." STOC. (Dequantization.)

Proctor, T. et al. (2022). "Measuring the Capabilities of Quantum Computers." Nature Physics 18, 75--79. (Mirror circuits.)

Cerezo, M. et al. (2021). "Variational Quantum Algorithms." Nature Reviews Physics 3, 625--644. (Barren plateau review.)

Lubinski, T. et al. (2023). "Application-Oriented Performance Benchmarks for Quantum Computing." IEEE Trans. Quantum Eng. 4, 3102824.

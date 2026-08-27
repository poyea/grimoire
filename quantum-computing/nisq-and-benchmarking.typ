#import "../template.typ": xref

= NISQ Devices and Benchmarking <nisq-and-benchmarking>

John Preskill coined *NISQ* (Noisy Intermediate-Scale Quantum) in 2018 to describe the 50--1000 qubit devices available today: large enough that classical simulation is hard, small enough that error correction is impractical. Understanding what NISQ devices can and cannot do requires precise benchmarking and honest assessment of noise.

*See also:* #xref("quantum-computing", "hardware-architectures", label: "Hardware Architectures"), #xref("quantum-computing", "quantum-algorithms", label: "Quantum Algorithms"), #xref("quantum-computing", "error-correction", label: "Error Correction")

== The NISQ Regime

Preskill's definition draws a practical boundary: 50--1000 physical qubits, operated without full quantum error correction. This range is interesting precisely because it sits in a hard zone: representing an arbitrary state on $n > 50$ qubits requires $2^n$ complex amplitudes, making exact classical simulation intractable, yet the qubit count is too small to run the deep fault-tolerant circuits that would yield provable algorithmic advantage.

Key device parameters on leading superconducting platforms (2023--2025):

- *Single-qubit gate error:* $10^(-4)$--$10^(-3)$ (0.01--0.1%)
- *Two-qubit gate error:* $10^(-3)$--$10^(-2)$ (0.1--1%)
- *Relaxation time $T_1$:* 50--500 $mu$s (energy decay to ground state)
- *Dephasing time $T_2$:* 10--200 $mu$s (coherence loss without energy loss; $T_2 <= 2 T_1$ always)
- *Gate time:* 20--100 ns for two-qubit gates

A rough circuit depth limit before errors dominate follows from $d_max approx T_2 \/ t_"gate"$, giving $tilde 10^3$--$10^4$ layers in principle — but correlated noise, leakage, and crosstalk reduce this severely in practice. Useful NISQ circuits today are typically shallower than 100 layers.

A useful fidelity estimate for a circuit with $n_1$ single-qubit gates and $n_2$ two-qubit gates is

$ F_"circuit" approx (1 - epsilon_1)^(n_1) (1 - epsilon_2)^(n_2), $

treating each gate error as independent (optimistic, since crosstalk introduces correlations). For $n_1 = 100$, $n_2 = 50$, $epsilon_1 = 10^{-3}$, $epsilon_2 = 10^{-2}$: $F approx 0.90 times 0.61 approx 0.55$. The circuit output is essentially random noise at this depth — motivating the practical 20--50 two-qubit gate budget for circuits where fidelity must remain above $tilde 90\%$.

Trapped-ion platforms achieve lower error rates ($tilde 10^(-4)$ for two-qubit gates) at the cost of slower gate times ($tilde 10$--$100$ $mu$s), giving comparable depth budgets. Photonic processors (PsiQuantum, Xanadu) and neutral-atom arrays (QuEra, Pasqal) have emerged as alternatives: neutral atoms offer reconfigurable all-to-all connectivity within a zone, enabling high-QV circuits on moderate qubit counts.

== Noise Models

Real noise is complex; tractable models approximate it. The standard framework is the *Kraus representation*: any physically valid quantum channel (completely positive, trace-preserving map) on a state $rho$ can be written as

$ cal(E)(rho) = sum_k K_k rho K_k^dagger, quad sum_k K_k^dagger K_k = II, $

where the $K_k$ are *Kraus operators*. The completeness relation $sum_k K_k^dagger K_k = II$ guarantees trace preservation ($"tr"[cal(E)(rho)] = 1$). Each noise model below is a specific choice of Kraus operators.

*Depolarizing channel.* With probability $p$, apply a uniformly random Pauli ($I, X, Y, Z$); with probability $1-p$ do nothing. For a single qubit:

$ cal(E)(rho) = (1 - p) rho + (p/4)(rho + X rho X + Y rho Y + Z rho Z). $

The Kraus operators are $K_0 = sqrt(1 - 3p/4)\, II$, $K_1 = sqrt(p/4)\, X$, $K_2 = sqrt(p/4)\, Y$, $K_3 = sqrt(p/4)\, Z$. One can verify $sum_k K_k^dagger K_k = (1 - 3p/4 + 3 dot p/4) II = II$. This is the standard model for gate benchmarks: it is symmetric, has no preferred axis, and is parameterized by a single number $p$.

*Amplitude damping — worked example.* Models spontaneous emission ($T_1$ decay). Kraus operators:

$ K_0 = mat(1, 0; 0, sqrt(1-gamma)), quad K_1 = mat(0, sqrt(gamma); 0, 0), quad gamma = 1 - e^(-t\/T_1). $

Apply to a general state $rho = mat(a, b; b^*, 1-a)$ (with $0 <= a <= 1$):

$ K_0 rho K_0^dagger = mat(a, b sqrt(1-gamma); b^* sqrt(1-gamma), (1-a)(1-gamma)), $

$ K_1 rho K_1^dagger = mat(gamma(1-a), 0; 0, 0). $

Summing: $cal(E)_"AD"(rho) = mat(a + gamma(1-a), quad b sqrt(1-gamma); b^* sqrt(1-gamma), quad (1-a)(1-gamma))$. The excited-state population $1-a$ shrinks by factor $1-gamma$; the ground state $a$ grows by $gamma(1-a)$; coherences $b$ decay by $sqrt(1-gamma)$. In the limit $t -> infinity$ ($gamma -> 1$), all population flows to $|0 chevron.r$ and coherences vanish.

*Phase damping — worked example.* Models dephasing ($T_2$ process, no energy exchange):

$ K_0 = mat(1, 0; 0, sqrt(1-lambda)), quad K_1 = mat(0, 0; 0, sqrt(lambda)), quad lambda = 1 - e^(-t\/T_2). $

Applying to $rho = mat(a, b; b^*, 1-a)$:

$ K_0 rho K_0^dagger + K_1 rho K_1^dagger = mat(a, b sqrt(1-lambda); b^* sqrt(1-lambda), 1-a). $

Populations on the diagonal are unchanged; coherences decay as $b -> b sqrt(1-lambda) approx b\, e^(-t\/(2 T_2))$. This captures the physical picture of random phase kicks: energy is conserved, but superpositions wash out. Combined amplitude and phase damping gives $T_2^(-1) = (2 T_1)^(-1) + T_phi^(-1)$ where $T_phi$ is the pure dephasing time.

*Combining amplitude and phase damping — a worked example.* Consider a qubit initialized in $|+ chevron.r = (|0 chevron.r + |1 chevron.r)/sqrt(2)$, so $rho_0 = mat(1/2, 1/2; 1/2, 1/2)$. After time $t$ under both processes:

$ rho(t) = mat(1/2 + gamma/2, quad (1/2) sqrt(1-gamma) sqrt(1-lambda); (1/2) sqrt(1-gamma) sqrt(1-lambda), quad (1/2)(1-gamma)). $

Setting $t = T_1 = T_2/2$ (a common superconducting-qubit approximation) and $gamma = 1 - e^{-1} approx 0.63$, $lambda approx 0.86$: the excited-state population drops from $1/2$ to $approx 0.19$, while the coherence $rho_(01)$ decays from $1/2$ to $approx 0.083$. Even at one $T_1$, the qubit has lost most of its coherence — motivating why NISQ circuits must finish well within $T_2$.

*Crosstalk.* Always-on $Z Z$ coupling between neighboring qubits causes idle qubits to accumulate conditional phase. This is especially problematic during long two-qubit gate sequences and manifests as context-dependent error rates — gate fidelity depends on what adjacent qubits are doing simultaneously.

*Measurement error.* Readout is typically the dominant single-shot error source, at 1--5% per qubit. A *confusion matrix* $M$ with $M_(i j) = Pr("read" i | "prepared" j)$ characterizes it; inverting $M$ (readout error mitigation) is cheap classically but limited by shot noise.

== Quantum Volume

IBM's Quantum Volume (Cross et al. 2019) is a single-number metric that captures gate fidelity, qubit connectivity, circuit compilation quality, and coherence time together.

*Definition.* For a device with $n$ qubits, run random $n$-qubit, $n$-layer square circuits (each layer is a random permutation followed by random $"SU"(4)$ gates on paired qubits). A circuit is *heavy* if its output bitstring has probability above the median of the ideal output distribution. The device achieves $"QV" = 2^n$ if the fraction of heavy outputs exceeds $2/3$ with at least $2 sigma$ confidence over many random circuits.

*Why "heavy output probability" and why $2/3$.* For an ideal device, the output distribution of a Haar-random circuit is approximately a Porter-Thomas distribution: probabilities $p_x$ follow $Pr(p_x > t) = e^{-2^n t}$. The median of this distribution is $m = (ln 2)/2^n$. A bitstring $x$ is *heavy* when $p_x > m$. Under the Porter-Thomas model, the probability that a randomly drawn sample is heavy is

$ Pr(p_x > m) = integral_m^infinity 2^n e^(-2^n t) dif t = e^(-2^n m) = e^(-ln 2) = 1/2. $

That integral gives $1/2$, which is the probability that a *uniformly random* bitstring (drawn without regard to the circuit) is heavy — as expected, since half of all bitstrings are above the median by definition. For a *perfect* device that samples according to the ideal output distribution $p_x$, the relevant quantity is instead $sum_{x : p_x > m} p_x$, the total probability mass concentrated on heavy bitstrings. This is not $1/2$: because the Porter-Thomas distribution is heavy-tailed, the high-probability bitstrings (the ones an ideal device will frequently output) are disproportionately the same bitstrings that are above the median. The 0.854 threshold follows from the Porter-Thomas distribution: for an $n$-qubit ideal circuit, the probability that an output bitstring is "heavy" (above the median) is exactly $1 - 1/e approx 0.632$ in expectation under a uniform draw, and the precise value 0.854 follows from the distribution of the median itself over random circuits — the ideal device's weighted heavy-output probability (weighting each bitstring by $p_x$) averages to approximately 0.854 over the ensemble of Haar-random circuits. So for a *perfect* device sampling exactly from the ideal distribution, heavy outputs occur with probability $approx 0.854$ — substantially above $1/2$ because there are many rare high-probability bitstrings, and an ideal device will frequently land on them. A *uniformly random* device (one that ignores the circuit entirely and samples uniformly) achieves exactly $1/2$.

The threshold $2/3$ sits halfway between the ideal value $0.854$ and the random baseline $0.5$: $ 2/3 approx (0.854 + 0.5)/2. $ Passing the $2/3$ threshold therefore means the device is closer to ideal behavior than to random noise — it is running the circuit in a meaningful sense. The $2 sigma$ confidence requirement means we need enough circuit samples (typically $tilde 100$ circuits, each run many times) to distinguish $2/3$ from $1/2$ statistically.

*Progress.* IBM reported QV~4 in 2019, QV~32 in 2020, QV~128 in 2021, and QV~512 in 2023 on Falcon/Eagle/Heron processor families.

*Criticism.* QV measures square circuits on the best $n$ qubits the device can identify. It does not capture performance on large circuits that span all qubits, and heavy connectivity (all-to-all or high degree) inflates QV without reflecting what sparse-connectivity hardware achieves on real workloads. Furthermore, QV is bounded above by the native qubit count: a 27-qubit device can achieve at most QV $= 2^{27}$, regardless of gate fidelity — so comparing QV across devices with very different qubit counts is misleading. IBM retired QV as a primary public metric after 2023 in favor of CLOPS and application-level benchmarks.

== Other Benchmarks

*Randomized Benchmarking (RB).* Apply random sequences of Clifford gates of increasing length $m$, followed by the inverse; measure survival probability. The decay

$ p(m) = A lambda^m + B $

gives average Clifford error per gate $r = (1 - lambda)(d-1)/d$ where $d = 2^n$. RB is insensitive to state preparation and measurement errors, and is the de facto standard for reporting gate fidelities. Extensions include Interleaved RB (error rate of a specific gate) and Character RB (non-Clifford gates).

*CLOPS (Circuit Layer Operations Per Second).* IBM's throughput metric: how many parametrized circuit layers can be executed and updated per second, including classical overhead. Relevant for variational algorithms where the optimizer must update parameters between runs. IBM's Falcon r5.11 achieves $tilde 1400$ CLOPS; Heron r1 reaches $tilde 5000$ CLOPS. A VQE run with 1000 optimizer iterations and 100 parameter-shift gradient evaluations per step requires $10^5$ circuit executions: at 5000 CLOPS with 50-layer circuits, that is $tilde 1000$ seconds of pure quantum execution time, ignoring classical overhead and queue wait.

*Mirror circuits (Proctor et al. 2021).* A scalable benchmark that constructs circuits with a built-in known answer: run a random circuit $C$, then its mirror $C^dagger$, and check that the output is the input state. Unlike RB, mirror circuits scale to large $n$ and catch *coherent* errors (systematic rotations) that depolarizing-noise models miss.

*Application-level benchmarks.* QASMbench (Li et al.) provides a suite of real algorithm circuits from chemistry, optimization, and machine learning. The QED-C benchmarks (Lubinski et al. 2023) define success conditions for application-relevant circuits and have been run across IBM, IonQ, Quantinuum, and Rigetti hardware.

== Quantum Supremacy Claims

*Google Sycamore (2019).* Arute et al. ran random circuit sampling on a 53-qubit device with 20 cycles of two-qubit gates. They estimated that sampling from the output distribution to error $epsilon$ would take 10,000 years on Summit (the then-fastest classical supercomputer), claiming "quantum supremacy."

*Classical counterarguments.* IBM researchers (2019) immediately noted that storing the full $2^{53}$ amplitude vector on disk avoids memory bottlenecks and cuts the estimate to $tilde 2.5$ days. Subsequent work by Pan, Chen, and Zhang (2022) using improved tensor-network contraction on classical clusters reduced the simulation cost further, to hours on a large HPC cluster with acceptable approximation error.

*Honest assessment.* Random circuit sampling is an artificial task with no known practical application. Classical simulation of NISQ-scale random circuits is a moving target as algorithms improve. The 2019 result demonstrated that a quantum device can execute circuits that are *hard to verify* classically, which is meaningful for device characterization but does not constitute practical quantum advantage.

*Xanadu Borealis (2022).* Madsen et al. performed Gaussian boson sampling on a 216-mode photonic chip, claiming $9000 times$ faster than the best classical algorithm at the time. As with Sycamore, subsequent classical algorithm improvements narrowed the gap. Both results are best understood as demonstrations of device capability in an adversarial benchmarking context, not as proofs of practical quantum utility.

The recurring pattern across supremacy claims is instructive: (1) quantum team identifies a sampling problem with no classical algorithm known to be efficient; (2) classicists improve tensor-network or polynomial methods; (3) the advantage shrinks but rarely disappears entirely. This dynamic is expected to continue until NISQ devices reach sufficient scale that classical simulation becomes thermodynamically implausible.

== NISQ Algorithms and Their Limitations

*VQE.* The variational quantum eigensolver minimizes $E(arrow(theta)) = chevron.l psi(arrow(theta)) | H | psi(arrow(theta)) chevron.r$ over a parameterized state $|psi(arrow(theta)) chevron.r$. Two fundamental problems arise on NISQ hardware:

1. *Barren plateaus (McClean et al. 2018):* For hardware-efficient ansätze drawn from the unitary $n$-design family, the gradient variance vanishes exponentially. Precisely, for any parameter $theta_k$ in a sufficiently expressive (2-design) ansatz on $n$ qubits:

$ "Var"[partial E / partial theta_k] = O(1/2^n). $

The variance shrinks by half for each additional qubit. This means that for $n = 50$ qubits, a gradient estimate requires roughly $2^{50} approx 10^{15}$ shots to achieve signal-to-noise ratio 1 — far beyond any practical device. The intuition is that a 2-design ansatz explores a region of Hilbert space that is exponentially large; almost everywhere in that space, the landscape is flat. Structured ansätze (chemistry-inspired, problem-specific) can avoid this, but at the cost of restricting the expressibility, creating a circuit-depth vs. trainability tradeoff.

2. *Noise accumulation:* Even moderate circuit depth renders the energy estimate unreliable. Error mitigation techniques (zero-noise extrapolation, probabilistic error cancellation) reduce bias but increase shot overhead exponentially.

The barren plateau and noise problems compound: a deeper ansatz needed to escape a plateau accumulates more noise; a shallower ansatz avoids noise but may sit in a barren plateau or be classically simulable. This is sometimes called the "NISQ expressibility-trainability-noise trilemma." Practical VQE implementations currently address it by using chemistry-guided ansätze (UCCSD, $k$-UpCCGSD) that restrict the search space to physically motivated excitations, accepting lower expressibility in exchange for trainable gradients.

*QAOA.* At depth $p$, QAOA on MaxCut achieves approximation ratio that provably improves with $p$ but requires $p = Omega(n)$ to match classical Goemans-Williamson (ratio 0.878) on worst-case instances. Shallow ($p tilde 1$--$3$) QAOA is tractable on NISQ hardware but achieves ratios below 0.75 on hard instances — below what classical algorithms can guarantee.

*Quantum machine learning and dequantization.* Several quantum linear-algebra algorithms (HHL for linear systems, Kerenidis-Prakash (KP) for recommendation systems) claimed exponential speedups over classical methods. These relied on *quantum RAM* (qRAM) to prepare quantum states encoding vectors in $O(text("polylog") N)$ time, and on the ability to extract useful information from the resulting quantum state.

Tang (2019) introduced the concept of *sampling access* (also called $ell^2$-sampling access) and used it to dequantize the KP recommendation-system algorithm. Sampling access to a vector $v in bb(R)^N$ means: (1) query any entry $v_i$ in $O(1)$; (2) draw an index $i$ with probability proportional to $v_i^2$ in $O(1)$. This is a classical analogue of what a qRAM state $sum_i v_i |i chevron.r$ allows a quantum algorithm to do.

Tang showed that given sampling access to input matrices $A$ and $B$, a classical algorithm can solve the same low-rank matrix approximation problem the KP algorithm solved, with runtime $O(text("poly")(k, 1/epsilon) dot text("polylog")(m, n))$ — the same asymptotic dependence on $k$ (rank) and $epsilon$ (error), and only polylogarithmic in the matrix dimensions $m, n$. The quantum speedup therefore evaporated: it was not a property of quantum mechanics but of the structured data access model (qRAM or sampling access).

The broader lesson: many QML speedup claims assume efficient state preparation, which is equivalent to sampling access. If the classical algorithm is also given sampling access, the quantum advantage disappears. Genuine quantum ML speedups, if they exist, require problems where quantum interference beyond data loading provides an advantage — and no such provable example is known for practical datasets.

Following Tang's result, similar dequantization was applied to quantum principal component analysis (Lloyd et al.), quantum support vector machines (Rebentrost et al.), and several other quantum linear-algebra primitives. The pattern in each case: the quantum algorithm's speedup came from qRAM-assisted state preparation, not from quantum interference or entanglement. This does not mean quantum ML is hopeless, but it does mean that provable advantage requires either (a) a problem with structure that quantum mechanics exploits directly (e.g., learning properties of a quantum system), or (b) a separation result showing sampling access is insufficient classically — which for the dequantized problems it demonstrably is not.

== Path to Fault Tolerance

The surface code (Fowler et al. 2012) achieves a threshold of $tilde 1\%$ physical gate error: below this, increasing the code distance $d$ exponentially suppresses the logical error rate:

$ p_L approx (p \/ p_"th")^((d+1)\/2). $

At physical error rate $p = 0.1\%$ and threshold $p_"th" = 1\%$, distance $d = 7$ gives $p_L approx 10^{-9}$. The overhead is severe: each logical qubit requires $tilde 2 d^2$ physical qubits, so $d = 7$ costs $tilde 100$ physical qubits per logical qubit; for the $tilde 1000$ logical qubits and deep circuits needed to run Shor's algorithm on a 2048-bit key (Gidney & Ekerå 2021), the total physical qubit count is $tilde 20$ million.

Leading labs (IBM, Google, Microsoft, IonQ, Quantinuum) have announced roadmaps targeting $10^4$--$10^6$ physical qubits in the 2030--2035 timeframe. The first realistic fault-tolerant application is widely expected to be quantum simulation of strongly correlated chemistry (100+ orbitals) where classical methods such as DMRG and CCSD(T) hit polynomial walls; provable classical hardness for optimization remains undemonstrated.

Near-term (2025--2030) realistic expectations: error-mitigated VQE may provide marginal improvements over classical heuristics on small molecule problems; NISQ devices remain useful as experimental platforms for studying physics, not as practical compute accelerators.

The clearest near-term milestone is the demonstration of a *logical qubit* with lower error rate than the best physical qubit on the same device. Google announced this milestone in 2023 (Nature 614, 676--681) for a distance-7 surface code logical qubit, achieving $p_L < p_"physical"$ for the first time. IBM's roadmap targets $100$ logical qubits with error rates below $10^{-6}$ by 2029 ("Quantum Centric Supercomputing"). Whether these milestones translate into practical algorithmic advantage before 2035 remains the central open question in the field.

== Further Reading

Preskill, J. (2018). "Quantum Computing in the NISQ Era and Beyond." Quantum 2, 79.

Cross, A. et al. (2019). "Validating Quantum Computers Using Randomized Model Circuits." Phys. Rev. A 100, 032328.

Arute, F. et al. (Google) (2019). "Quantum Supremacy Using a Programmable Superconducting Processor." Nature 574, 505--510.

Pan, F., Chen, K., Zhang, P. (2022). "Solving the Sampling Problem of the Sycamore Quantum Circuits." Phys. Rev. Lett. 129, 090502.

Tang, E. (2019). "A Quantum-Inspired Classical Algorithm for Recommendation Systems." STOC. (Dequantization.)

Proctor, T. et al. (2022). "Measuring the Capabilities of Quantum Computers." Nature Physics 18, 75--79. (Mirror circuits.)

Cerezo, M. et al. (2021). "Variational Quantum Algorithms." Nature Reviews Physics 3, 625--644. (Barren plateau review.)

Lubinski, T. et al. (2023). "Application-Oriented Performance Benchmarks for Quantum Computing." IEEE Trans. Quantum Eng. 4, 3102824.

McClean, J. R. et al. (2018). "Barren Plateaus in Quantum Neural Network Training Landscapes." Nature Communications 9, 4812.

Kerenidis, I., Prakash, A. (2017). "Quantum Recommendation Systems." ITCS. (The QML algorithm dequantized by Tang.)

Gidney, C., Ekerå, M. (2021). "How to Factor 2048-bit RSA Integers in 8 Hours Using 20 Million Noisy Qubits." Quantum 5, 433. (Resource estimate for fault-tolerant Shor's algorithm.)

Aharonov, D., Ben-Or, M. (1997). "Fault-Tolerant Quantum Computation with Constant Error Rate." STOC. (Early threshold theorem.)

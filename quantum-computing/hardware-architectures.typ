#import "../template.typ": xref

= Hardware Architectures <hardware-architectures>

A qubit is whatever two-level quantum system you can isolate, control, and read out faster than it decoheres. Engineering tradeoffs — coherence time, gate fidelity, connectivity, scalability, operating temperature — split the industry into four credible modalities: superconducting circuits, trapped ions, neutral atoms, and photonics. Topological qubits remain an open research bet. This chapter surveys the physical implementations, their native gate sets, and how the choice cascades up the stack into compilers and codes.

*See also:* #xref("quantum-computing", "qubits-and-gates", label: "Qubits and Gates"), #xref("quantum-computing", "error-correction", label: "Error Correction"), #xref("quantum-computing", "nisq-and-benchmarking", label: "NISQ Devices and Benchmarking"), `cpu-architecture/`, `linux-kernel/`

== DiVincenzo's Criteria

Any scalable quantum computer must satisfy:

1. Well-characterized, scalable qubits.
2. Reliable initialization to a fiducial state.
3. Long coherence relative to gate times.
4. A universal gate set.
5. Qubit-specific measurement.

Plus two for communication: stationary-to-flying qubit interconversion and faithful flying-qubit transmission.

== Superconducting Qubits

Aluminum Josephson junctions on a silicon/sapphire substrate, cooled to $approx 10-20$ mK in a dilution refrigerator. The *transmon* (Koch 2007) suppresses charge noise by operating in the regime $E_J / E_C >> 1$; today's variants (fluxonium, unimon, tunable couplers) chase higher anharmonicity and lower crosstalk.

Native operations: microwave drives at 4-8 GHz on individual qubits; two-qubit gates via tunable couplers (Google), cross-resonance (IBM), parametric drives, or capacitive coupling.

#table(
  columns: (auto, auto),
  [*Parameter*], [*Typical 2024 value*],
  [$T_1$], [100-300 $mu$s (transmon), 1-2 ms (fluxonium)],
  [$T_2$ (echo)], [100-300 $mu$s],
  [1Q gate time / fidelity], [20-40 ns / 99.95%],
  [2Q gate time / fidelity], [30-80 ns / 99.5-99.7%],
  [Readout time / fidelity], [100-500 ns / 99%],
  [Connectivity], [Nearest-neighbor on heavy-hex (IBM) or square (Google)],
  [Qubit count], [IBM Condor 1121, Google Willow 105, Rigetti Ankaa 84],
)

*Strengths:* fastest gates, most mature fabrication (leverages CMOS-adjacent processes), highest qubit counts.
*Weaknesses:* fixed couplers force SWAP-heavy compilation; dilution refrigerators are expensive; calibration drift requires constant retuning.

== Trapped Ions

Single $""^171$Yb$""^+$, $""^138$Ba$""^+$, $""^40$Ca$""^+$, or $""^9$Be$""^+$ ions held in linear Paul traps or surface traps (HOA). Qubit encoded in hyperfine clock states ($approx 12.6$ GHz for Yb) or optical transitions.

Native gates: laser- or microwave-driven single-qubit rotations; two-qubit Mølmer-Sørensen gate via shared motional modes — *all-to-all* connectivity within a chain.

#table(
  columns: (auto, auto),
  [*Parameter*], [*Typical 2024 value*],
  [$T_1$], [Hours (hyperfine)],
  [$T_2$], [Seconds to minutes (memory qubits)],
  [1Q gate time / fidelity], [1-20 $mu$s / 99.99%],
  [2Q gate time / fidelity], [50-300 $mu$s / 99.9%],
  [Readout fidelity], [99.97% (Quantinuum H2)],
  [Connectivity], [All-to-all in a chain; QCCD shuttling between zones],
  [Qubit count], [Quantinuum H2 56, IonQ Forte 32],
)

*QCCD architecture* (Kielpinski 2002): partition the trap into memory, interaction, and readout zones; physically shuttle ions between them. Quantinuum's H-series implements this and holds records for two-qubit fidelity and logical-qubit benchmarks (2024: 12 logical qubits, [[7,1,3]] color-code memory below threshold).

*Strengths:* highest fidelity, all-to-all connectivity, long coherence.
*Weaknesses:* slow gates limit clock speed; shuttling adds latency; scaling beyond $tilde 100$ ions per trap requires photonic interconnects between modules.

== Neutral Atoms

$""^87$Rb or $""^133$Cs atoms in optical tweezer arrays. Qubits in hyperfine ground states; entanglement via excitation to Rydberg states ($n approx 60$-$70$) where strong dipole-dipole interactions enable controlled-Z gates.

Pioneers: QuEra (Aquila 256-atom analog), Atom Computing, Pasqal. 2023-2024 saw a step change with *reconfigurable arrays*: tweezers reposition atoms mid-circuit, giving programmable connectivity.

#table(
  columns: (auto, auto),
  [*Parameter*], [*Typical 2024 value*],
  [$T_1$], [Seconds],
  [$T_2$ (dynamical decoupling)], [10-100 ms],
  [2Q (Rydberg) gate], [200-400 ns / 99.5%],
  [Connectivity], [Reconfigurable via atom movement],
  [Qubit count], [Atom Computing 1180, QuEra Gemini 256, Pasqal 100+],
)

Lukin et al. (Harvard/QuEra/MIT, Nature 626, 2024) demonstrated 48 logical qubits from 280 physical atoms running surface-code and [[7,1,3]] color-code logic — a leap enabled by mid-circuit atom shuttling and transversal logical operations.

*Strengths:* massive scalability (cold-atom physics scales to thousands), flexible connectivity, room-temperature optics (laser cooling is involved but no dilution fridge).
*Weaknesses:* slower gates than superconducting; atom loss / reloading introduces non-Markovian errors; Rydberg blockade range limits density.

== Photonic Qubits

Encoded in dual-rail photons (polarization, time-bin, or path), generated by spontaneous parametric down-conversion or solid-state emitters (quantum dots, defect centers). Gates: linear optics + measurement-induced nonlinearity (KLM 1999) or *measurement-based* / *fusion-based* quantum computing (FBQC) on resource states.

PsiQuantum, Xanadu, ORCA, Quandela. Xanadu's Borealis (2022) demonstrated Gaussian boson sampling at $> 200$ modes — quantum advantage on a sampling task, not a universal computer.

*Strengths:* room-temperature operation (for many stages), naturally networked (photons fly through fiber), no decoherence in flight.
*Weaknesses:* probabilistic gates demand massive multiplexing; loss is the dominant error and has no error-correction analog as friendly as Pauli — needs *photon-loss codes*; single-photon sources and detectors remain limiting.

== Topological Qubits

Majorana zero modes in semiconductor-superconductor nanowires (Microsoft/Station Q) or fractional quantum Hall systems. Logical states protected by topology — exponentially suppressed errors. Microsoft (Nature 638, 2025) reported "topoconductor" devices with claimed Majorana signatures; community remains cautious pending reproducible braiding demonstrations.

== Cross-Modality Comparison

#table(
  columns: (auto, auto, auto, auto),
  [], [*Superconducting*], [*Trapped ion*], [*Neutral atom*],
  [Operating temp], [10-20 mK], [Room (laser-cooled)], [Room (laser-cooled)],
  [2Q fidelity], [99.5-99.7%], [99.9%], [99.5%],
  [Gate time], [30-80 ns], [50-300 $mu$s], [200-400 ns],
  [Coherence], [100-300 $mu$s], [Seconds-minutes], [10-100 ms],
  [Connectivity], [Nearest-neighbor], [All-to-all (chain)], [Reconfigurable],
  [Scaling path], [Tile chips], [Photonic interconnect (modules)], [Larger arrays],
  [Best-fit QEC], [Surface code], [Color / qLDPC], [Surface / qLDPC],
  [Lead vendors], [IBM, Google, Rigetti], [Quantinuum, IonQ], [QuEra, Atom Computing, Pasqal],
)

== Control Stack

A modern quantum control stack looks roughly like:

```
   User program (Qiskit / Cirq / OpenQASM)
        |
        v
   Transpiler (gate decomposition, routing, scheduling)
        |
        v
   Pulse-level IR (OpenPulse / Qiskit Pulse / Quil-T)
        |
        v
   FPGA-based controller (Quantum Machines OPX, Zurich Instruments,
   Keysight QCS) -- runs in DSP-friendly real-time loops
        |
        v
   Analog electronics (DAC, microwave / laser drives, AWGs)
        |
        v
   Physical qubits (in cryostat / vacuum chamber)
        |
        v
   Readout (ADC, integrated digital signal processing, decoder)
```

Real-time syndrome decoding for QEC (see _Error Correction_) lives on the FPGA layer — Riverlane's Deltaflow, Google's custom ASICs, and Q-CTRL's Boulder Opal close that loop.

== Cryogenic Control and Cabling Crisis

A 1000-qubit superconducting chip needs $approx 4000$ control lines through a dilution fridge — thermal load and physical space become limiting. Solutions:

- *Cryo-CMOS:* Intel Horse Ridge II, Google Quantum AI custom 4 K control chips.
- *Frequency multiplexing:* one line drives many qubits at different frequencies.
- *Photonic readout / control:* optical fibers carry far less heat than coax.

Trapped ions and atoms sidestep this — laser beams steer optically — but acquire their own optical-engineering burden (acousto-optic deflectors, beam stability).

== Further Reading

DiVincenzo, D. (2000). "The Physical Implementation of Quantum Computation." Fortschr. Phys.

Koch, J. et al. (2007). "Charge-Insensitive Qubit Design Derived from the Cooper Pair Box." Phys. Rev. A (transmon).

Arute, F. et al. (2019). "Quantum Supremacy Using a Programmable Superconducting Processor." Nature (Sycamore).

Google Quantum AI (2024). "Quantum Error Correction Below the Surface Code Threshold." Nature 638 (Willow).

Kielpinski, D., Monroe, C., Wineland, D. (2002). "Architecture for a Large-Scale Ion-Trap Quantum Computer." Nature (QCCD).

Moses, S. et al. (2023). "A Race-Track Trapped-Ion Quantum Processor." Phys. Rev. X (Quantinuum H2).

da Silva, M. et al. (2024). "Demonstration of Logical Qubits and Repeated Error Correction with Better-than-Physical Error Rates." arXiv:2404.02280 (Quantinuum).

Bluvstein, D. et al. (2024). "Logical Quantum Processor Based on Reconfigurable Atom Arrays." Nature 626 (Harvard/QuEra).

Knill, E., Laflamme, R., Milburn, G. (2001). "A Scheme for Efficient Quantum Computation with Linear Optics." Nature (KLM).

Bartolucci, S. et al. (2023). "Fusion-Based Quantum Computation." Nat. Commun. (PsiQuantum FBQC).

Madsen, L. et al. (2022). "Quantum Computational Advantage with a Programmable Photonic Processor." Nature (Borealis).

Microsoft Quantum (2025). "Interferometric Single-Shot Parity Measurement in InAs-Al Hybrid Devices." Nature 638.

Krinner, S. et al. (2019). "Engineering Cryogenic Setups for 100-Qubit Scale Superconducting Circuit Systems." EPJ Quantum Tech.

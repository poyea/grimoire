= Notation and Conventions

This chapter fixes the Dirac notation, gate symbols, and error-correction conventions used throughout the volume.

== States and Dirac Notation

A pure state of $n$ qubits is a unit vector in $bb(C)^(2^n)$. We write kets as $|psi chevron.r$ and bras as $chevron.l psi|$.

#table(
  columns: (auto, auto),
  [*Symbol*], [*Meaning*],
  [$|0 chevron.r$, $|1 chevron.r$], [computational basis of $bb(C)^2$],
  [$|psi chevron.r = alpha |0 chevron.r + beta |1 chevron.r$], [single-qubit state, $|alpha|^2 + |beta|^2 = 1$],
  [$chevron.l phi | psi chevron.r$], [inner product],
  [$|psi chevron.r chevron.l psi|$], [outer product (projector for unit $|psi chevron.r$)],
  [$|+ chevron.r$, $|- chevron.r$], [$(|0 chevron.r plus.minus |1 chevron.r) \/ sqrt(2)$],
  [$|psi chevron.r times.o |phi chevron.r$], [tensor product; often abbreviated $|psi phi chevron.r$],
  [$|0 chevron.r^(times.o n)$], [$n$-fold tensor power],
  [$rho$], [density matrix; pure iff $rho = |psi chevron.r chevron.l psi|$, i.e. $"tr"(rho^2) = 1$],
)

Global phase is unobservable: $e^(i phi) |psi chevron.r$ and $|psi chevron.r$ are physically identical. A single-qubit pure state is parameterized on the *Bloch sphere* by $(theta, phi)$ as $cos(theta\/2) |0 chevron.r + e^(i phi) sin(theta\/2) |1 chevron.r$; mixed states fill the interior of the ball via $rho = (II + arrow(r) dot arrow(sigma)) \/ 2$.

== Operators and Gates

#table(
  columns: (auto, auto),
  [*Symbol*], [*Meaning*],
  [$II$], [identity operator],
  [$X, Y, Z$], [Pauli matrices; $arrow(sigma) = (X, Y, Z)$],
  [$H$], [Hadamard; $S$, $T$ are the $pi\/2$ and $pi\/4$ phase gates],
  [$R_x (theta) = e^(-i theta X \/ 2)$], [rotation; similarly $R_y$, $R_z$],
  [$"CNOT"$, $"CZ"$], [controlled-$X$, controlled-$Z$ (control listed first)],
  [$U^dagger$], [conjugate transpose; unitarity means $U^dagger U = II$],
  [$X_i$], [Pauli $X$ acting on qubit $i$ (identity elsewhere)],
  [$cal(P)_n$], [$n$-qubit Pauli group],
)

Qubits are indexed from 0 in code (Qiskit little-endian: qubit 0 is the least significant bit of the measured bitstring) and from 1 in stabilizer formulas like $Z_1 Z_2$. Circuits read left to right; matrix products apply right to left.

== Measurement

Measurement in the computational basis yields outcome $x$ with probability $|chevron.l x | psi chevron.r|^2$ (Born rule) and collapses the state. Expectation values are $chevron.l A chevron.r = chevron.l psi | A | psi chevron.r = "tr"(rho A)$. "Measuring in the $X$ basis" means rotating with $H$ then measuring in the computational basis.

== Error Correction

#table(
  columns: (auto, auto),
  [*Symbol*], [*Meaning*],
  [$[[n, k, d]]$], [stabilizer code: $n$ physical qubits, $k$ logical, distance $d$],
  [$S subset cal(P)_n$], [stabilizer group; codespace is its joint $+1$ eigenspace],
  [$|0 chevron.r_L$, $X_L$], [logical state, logical operator],
  [$p$, $p_("th")$], [physical error rate, threshold],
  [$T_1$, $T_2$], [relaxation and dephasing times],
)

== Complexity and Units

$O(dot)$ is standard asymptotic notation; $N = 2^n$ converts between qubit count and search-space size. Oracle algorithms are measured in *queries*. Gate fidelities are quoted as error rates (e.g. $10^(-3)$ per two-qubit gate); times in $mu$s or ns. Hardware-relevant classes: BQP (quantum polynomial time) vs P, NP — see _Complexity Theory_ (programming-languages).

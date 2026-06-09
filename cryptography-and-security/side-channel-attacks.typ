= Side-Channel Attacks

A side-channel attack extracts secrets not from mathematical weaknesses but from the physical or behavioural traces of computation: timing, cache state, power draw, electromagnetic emission, or induced faults. The algorithm can be provably secure while its implementation leaks the key. This chapter covers timing and cache attacks, power analysis, fault injection, and the engineering discipline of constant-time code.

*See also:* _Symmetric Primitives_ (AES T-tables), _Asymmetric Cryptography_ (RSA/ECC implementations), _Digital Signatures_ (nonce leakage), and the CPU Architecture volume's _Speculative Execution Security_ chapter (Spectre/Meltdown).

== Timing Attacks

Kocher (1996) showed that the running time of square-and-multiply RSA leaks the private exponent bit by bit: the data-dependent time of each modular operation can be correlated against per-bit hypotheses across many measurements. Remote exploitation is practical: Brumley & Boneh (2003) recovered an OpenSSL RSA key across a network by timing decryption, exploiting Montgomery reduction's extra-reduction step.

Two recurring non-RSA examples:
- *Non-constant-time comparison*: `memcmp`-style early exit on MAC or token verification lets an attacker forge byte by byte.
- *Padding-dependent processing time*: Lucky Thirteen (AlFardan & Paterson, 2013) exploited the fact that TLS CBC MAC verification time depends on the plaintext padding length.
- *Table lookups indexed by secrets*: the classic AES T-table implementation indexes memory with key-dependent bytes — directly observable through the cache.

== Cache Attacks

The cache is a shared, stateful resource whose latency difference (hit $approx 1$–$4$ ns depending on level, miss $approx 100$ ns) is easily measurable. Canonical techniques:

- *Prime+Probe* (Osvik, Shamir & Tromer, 2006): fill a cache set with attacker data, let the victim run, then measure which of the attacker's lines were evicted. Works without shared memory; demonstrated against AES (2006), and in later work across VMs and from JavaScript.
- *Flush+Reload* (Yarom & Falkner, 2014): with shared memory (deduplicated pages, shared libraries), `clflush` a line, wait, then time a reload — a fast hit means the victim touched it. Line-granularity spying; recovered GnuPG RSA keys across VM boundaries.
- *Evict+Time, Flush+Flush, occupancy channels*: variants trading noise, speed, and stealth.

Cache attacks matter beyond cryptography: they are the *exfiltration channel* for transient-execution attacks. Spectre and Meltdown (2018) use speculative execution to touch secret-dependent cache lines, then read the secret out with Flush+Reload — see the CPU Architecture volume for the microarchitectural detail and mitigations (KPTI, retpolines, fence-based hardening).

Mitigations: no secret-indexed memory access (bitsliced or vector-permute AES, dedicated AES-NI instructions), cache partitioning (Intel CAT), disabling page deduplication across trust domains, and reduced-resolution timers in browsers (post-Spectre, along with site isolation).

== Power and Electromagnetic Analysis

With physical access (smart cards, embedded devices, HSMs), the power trace is a rich signal:

- *Simple Power Analysis (SPA)*: read the operation sequence directly off one trace — square vs. multiply in RSA, point double vs. add in ECC.
- *Differential Power Analysis* (Kocher, Jaffe & Jun, 1999): statistical attack over many traces. Guess a key byte, predict an intermediate bit (e.g., an S-box output), partition traces by the prediction, and subtract averaged groups: the correct guess produces a visible difference spike. *Correlation Power Analysis* (CPA) refines this with a Hamming-weight power model and Pearson correlation.
- *Template and deep-learning attacks*: profile a controlled identical device, then classify single traces from the target; neural networks now defeat many masked implementations with surprisingly few traces.
- *EM analysis*: the same statistics on electromagnetic emission — non-contact, localisable to specific chip regions. Related acoustic and power-LED channels have extracted RSA keys from laptop sounds (Genkin, Shamir & Tromer, 2014).

Countermeasures:
- *Masking*: split every secret intermediate into $d + 1$ random shares so any $d$ probes are statistically independent of the secret; cost grows quadratically with order, and composition is subtle enough that verification tools (maskVerif et al.) are standard.
- *Hiding*: shuffling operation order, random delays, dual-rail logic, on-chip noise — raises trace counts rather than providing proofs.
- Certification regimes (Common Criteria, FIPS 140-3) mandate resistance levels for payment cards and secure elements.

== Fault Injection

Rather than observing, the attacker perturbs: voltage glitches, clock glitches, laser pulses, or electromagnetic pulses push the device outside its operating envelope at a chosen instant.

- *The Bellcore attack / RSA-CRT* (Boneh, DeMillo & Lipton, 1997): a single faulty CRT-RSA signature factors $N$ — from a correct signature $s$ and faulty $s'$, $gcd(s - s', N)$ reveals a prime factor. Mandatory countermeasure: verify every signature before release.
- *Differential Fault Analysis on AES* (Piret & Quisquater, 2003): one byte fault injected before the final MixColumns narrows the key to a small brute-forceable set.
- *Instruction skipping*: glitching a conditional branch bypasses signature checks entirely — the basis of many game-console and bootloader jailbreaks (e.g., the Nintendo Switch Tegra bootROM exploit chain combined glitching with software bugs).
- *Rowhammer*: software-only fault injection — rapid DRAM row activation flips bits in neighbouring rows; demonstrated for privilege escalation and (as Rowhammer-based fault attacks) against co-located cryptographic keys.

Defences: redundant computation with comparison, infective countermeasures, sensors (voltage/clock/light detectors), and sign-then-verify.

== Constant-Time Programming

The portable software defence is a discipline: *no secret may influence a branch, a memory address, or the operand of a variable-time instruction* (division, some multipliers).

Patterns:
- Branchless selection: `mask = -(condition); result = (a & mask) | (b & ~mask);`
- Constant-time comparison: OR the XOR of all byte pairs; compare the accumulator to zero once.
- Table lookups: read _every_ entry and select with masks, or use bitslicing/vector permutes.
- For ECC: complete addition formulas and Montgomery ladders rather than data-dependent branching.

Hazards: compilers may re-introduce branches from masked code (increasingly documented as "constant-time correctness" failures; `ct_select` intrinsics and language support are emerging), and CPUs may add data-dependent timing (big.LITTLE, DVFS — the *Hertzbleed* attack, 2022, turned frequency scaling into a remote power side channel). Verification tooling: `dudect` (statistical), ctgrind/Memcheck taint tricks, formal tools (Binsec/Rel, Jasmin/EasyCrypt pipelines used for HACL\* and libjade).

== Design Checklist

- Use hardware primitives (AES-NI, SHA extensions) — they are constant-time by design and faster.
- Choose algorithms designed for constant-time implementation (X25519, Ed25519, ChaCha20-Poly1305, ML-KEM).
- Never early-exit on secret comparison; never index tables with secrets.
- On physical-access threat models, assume DPA and faults: mask, add sensors, verify before output.
- Test: run `dudect`-style measurements in CI; pin crypto to constant-frequency cores where Hertzbleed-class channels matter.

== Further Reading

- Kocher, P. (1996). Timing attacks on implementations of Diffie-Hellman, RSA, DSS, and other systems. _CRYPTO_.
- Kocher, P., Jaffe, J., & Jun, B. (1999). Differential power analysis. _CRYPTO_.
- Boneh, D., DeMillo, R., & Lipton, R. (1997). On the importance of checking cryptographic protocols for faults. _EUROCRYPT_.
- Yarom, Y., & Falkner, K. (2014). FLUSH+RELOAD: a high resolution, low noise, L3 cache side-channel attack. _USENIX Security_.
- Osvik, D. A., Shamir, A., & Tromer, E. (2006). Cache attacks and countermeasures: the case of AES. _CT-RSA_.
- Wang, Y. et al. (2022). Hertzbleed: turning power side-channel attacks into remote timing attacks on x86. _USENIX Security_.

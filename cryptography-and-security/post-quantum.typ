= Post-Quantum Cryptography

Shor's algorithm (1994) factors integers and computes discrete logs in polynomial time on a sufficiently large quantum computer, breaking $"RSA"$, finite-field $"DH"$, and elliptic-curve cryptography. NIST began standardizing post-quantum (PQ) replacements in 2016; FIPS 203/204/205 (2024) finalized ML-KEM, ML-DSA, and SLH-DSA. Real-world deployment is well underway in *hybrid* mode (classical + PQ KEM combined).

*See also:* _Asymmetric Cryptography_, _Protocols_, _Key Management_, _TLS (Transport Layer Security)_ (networking).

== Quantum Threats and Timelines

- *Shor:* factoring $n$-bit integers takes $O(n^3)$ qubit operations on a fault-tolerant quantum computer. 2048-bit $"RSA"$ requires $approx 4000$ logical qubits, conservatively $approx 20$ million physical qubits (Gidney-Ekerå 2021).
- *Grover:* unstructured search square-roots, so 256-bit symmetric keys become 128-bit. *128-bit AES is no longer enough* if quantum-relevant; use $"AES"$-256.
- *Harvest-now-decrypt-later:* adversaries record traffic today to decrypt when quantum is available. Long-lived secrets (medical, government, identity) need PQ confidentiality *now*.

NSA's CNSA 2.0 mandates PQ for national-security systems by 2030–2035. Chrome, Firefox, Cloudflare, Apple iMessage, and Signal have already deployed hybrid PQ KEMs.

== Lattice-Based Foundations

The hardness assumption is *Module Learning With Errors* (M-LWE): given $(A, A s + e)$ where $A$ is a uniform matrix over $R_q = ZZ_q [X] slash (X^n + 1)$, $s$ a secret module element, $e$ a small error, distinguishing from uniform is hard. Closely related: *Module Short Integer Solution* (M-SIS).

These are believed quantum-hard. Best known attack is BKZ lattice reduction; cost is exponential in dimension.

== ML-KEM (Kyber)

ML-KEM (FIPS 203, formerly CRYSTALS-Kyber) is a *key-encapsulation mechanism*: it outputs an encapsulated symmetric key, not arbitrary ciphertext.

#table(
  columns: 5,
  [*Param set*], [*Pub key*], [*Ciphertext*], [*Shared secret*], [*Security*],
  [ML-KEM-512], [800 B], [768 B], [32 B], [\~AES-128],
  [ML-KEM-768], [1184 B], [1088 B], [32 B], [\~AES-192],
  [ML-KEM-1024], [1568 B], [1568 B], [32 B], [\~AES-256],
)

Interface:

```python
# Using liboqs Python binding
from oqs import KeyEncapsulation
kem = KeyEncapsulation("ML-KEM-768")
pk = kem.generate_keypair()
# Sender
ct, ss_sender = kem.encap_secret(pk)
# Receiver
ss_receiver = kem.decap_secret(ct)
assert ss_sender == ss_receiver   # 32-byte shared secret
```

ML-KEM achieves *IND-CCA2* security via the Fujisaki-Okamoto transform: an inner IND-CPA scheme is wrapped with re-encryption check and rejection of malformed ciphertexts.

== ML-DSA (Dilithium)

ML-DSA (FIPS 204, formerly CRYSTALS-Dilithium) is a lattice signature scheme using the *Fiat-Shamir with aborts* paradigm.

#table(
  columns: 4,
  [*Param set*], [*Pub key*], [*Signature*], [*Security*],
  [ML-DSA-44], [1312 B], [2420 B], [\~AES-128],
  [ML-DSA-65], [1952 B], [3309 B], [\~AES-192],
  [ML-DSA-87], [2592 B], [4627 B], [\~AES-256],
)

```rust
use pqcrypto_dilithium::dilithium3::*;
let (pk, sk) = keypair();
let sig = detached_sign(b"message", &sk);
verify_detached_signature(&sig, b"message", &pk).unwrap();
```

Signatures are $approx 50 times$ larger than Ed25519, but verification is fast (millions/sec on a CPU). Performance is competitive with $"ECDSA"$ for signing.

== SLH-DSA (SPHINCS+)

SLH-DSA (FIPS 205) is a *hash-based* stateless signature. Its security reduces to the security of the underlying hash function — no number-theoretic assumption. Signatures are large (8 KB to 50 KB) and signing is slow, but the scheme is *the most conservative PQ signature* known.

Use SLH-DSA when:

- You need maximum confidence (no lattice assumption).
- Long-term archival signatures.
- Firmware / boot signing where size is acceptable.

== FN-DSA (Falcon)

FN-DSA (formerly Falcon) is a lattice signature based on NTRU lattices with smaller signatures than ML-DSA ($approx 666$ B for Falcon-512) but harder constant-time implementation due to floating-point Gaussian sampling. FN-DSA is slated for standardization as FIPS 206, still in draft as of 2026 (only FIPS 203/204/205 were finalized in August 2024).

== Other Families

- *Code-based:* Classic McEliece and HQC. NIST finalized three PQ standards in August 2024: FIPS 203 (ML-KEM/Kyber), FIPS 204 (ML-DSA/Dilithium), and FIPS 205 (SLH-DSA/SPHINCS+). HQC is under evaluation as a potential backup KEM (code-based, complementing the lattice-based ML-KEM) but has not been standardized as a FIPS. McEliece has a 60-year track record; both McEliece and HQC have very large public keys (\~ 1 MB for McEliece).
- *Isogeny-based:* SIDH/SIKE broken by Castryck-Decru (2022) via torsion-point attack. CSIDH and SQIsign remain candidates.
- *Multivariate:* Rainbow broken by Beullens (2022). $"GeMSS"$ and $"UOV"$ variants survive.

== Hybrid Deployments

The transition strategy is *hybrid* — combine a classical and a PQ KEM so the result is secure if *either* survives.

```text
X25519 shared secret: s1  (32 B)
ML-KEM-768 shared secret: s2 (32 B)
combined = HKDF-Extract(salt=transcript, ikm = s1 ‖ s2)
```

TLS 1.3 deployments use the `x25519_kyber768` hybrid group (codepoint 0x6399); Chrome enabled it by default in 2024. Cloudflare reports < 1 ms median handshake overhead over X25519 alone.

```python
# Conceptual hybrid KEM
ec_priv = X25519PrivateKey.generate()
ec_pub = ec_priv.public_key().public_bytes_raw()
mlkem_pub = pk
combined_pub = ec_pub + mlkem_pub
# Sender
ec_eph = X25519PrivateKey.generate()
s1 = ec_eph.exchange(load_pub(ec_pub))
ct_mlkem, s2 = kem.encap_secret(mlkem_pub)
shared = hkdf_combine(s1, s2, transcript)
```

== Migration Practicalities

#table(
  columns: 3,
  [*System*], [*Status (2025–2026)*], [*Path*],
  [TLS 1.3 (web)], [hybrid X25519+ML-KEM-768 default in major browsers], [add PQ groups, drop after CNSA 2.0],
  [SSH], [OpenSSH 9.9 added sntrup761x25519 + ML-KEM], [upgrade],
  [WireGuard], [no PQ in core; Rosenpass adds PQ pre-shared keys], [overlay],
  [IPsec], [draft RFCs for hybrid IKEv2], [vendor specific],
  [JWT / OIDC], [ML-DSA in JOSE registry], [add alg],
  [Code signing], [SLH-DSA for firmware; ML-DSA for builds], [transition],
)

CNSA 2.0 explicit requirements (2024): ML-KEM-1024, ML-DSA-87 or SLH-DSA-256, $"AES"$-256, $"SHA"$-384, X.509 certificates with PQ algorithms.

== Implementation Pitfalls

- *Side-channels:* ML-KEM decapsulation must be constant-time over the entire ciphertext, including rejection sampling; Kyber-specific timing attacks on naive implementations have been published (Ravi et al. 2023).
- *Decryption failure rate:* lattice schemes have nonzero failure probability ($approx 2^(-150)$ for ML-KEM-768). Implementations must not leak failure as a timing oracle.
- *Floating-point in Falcon:* deterministic FP behavior is hard; constant-time Gaussian sampler is non-trivial.
- *Large messages:* HTTP/3 and QUIC handshakes hit MTU limits with PQ certs; chained delivery and KEMTLS (Schwabe-Stebila-Wiggers 2020) are research topics.
- *State pitfalls:* stateful hash-based signatures (XMSS, LMS — NIST SP 800-208) require *never reusing a one-time key*; reuse breaks everything. SLH-DSA is stateless and safe for general use.

== Comparison Summary

#table(
  columns: 5,
  [*Scheme*], [*Type*], [*Sig / CT size*], [*Pub key*], [*Assumption*],
  [Ed25519], [classical], [64 B], [32 B], [ECDLP],
  [X25519 ECDH], [classical], [32 B], [32 B], [ECDLP],
  [ML-KEM-768], [PQ KEM], [1088 B], [1184 B], [M-LWE],
  [ML-DSA-65], [PQ sig], [3309 B], [1952 B], [M-LWE/SIS],
  [FN-DSA-512], [PQ sig], [666 B], [897 B], [NTRU],
  [SLH-DSA-128s], [PQ sig], [7856 B], [32 B], [hash only],
  [Classic McEliece], [PQ KEM], [188 B], [1 MB+], [code-based],
)

== Further Reading

NIST FIPS 203 (2024). "Module-Lattice-Based Key-Encapsulation Mechanism Standard."

NIST FIPS 204 (2024). "Module-Lattice-Based Digital Signature Standard."

NIST FIPS 205 (2024). "Stateless Hash-Based Digital Signature Standard."

NIST SP 800-208 (2020). "Recommendation for Stateful Hash-Based Signature Schemes."

Bos, J. et al. (2018). "CRYSTALS-Kyber: A CCA-secure Module-Lattice-based KEM." EuroS&P.

Ducas, L. et al. (2018). "CRYSTALS-Dilithium: A Lattice-based Digital Signature Scheme." TCHES.

Bernstein, D. J. et al. (2019). "$"SPHINCS"$+: a signature scheme." CCS.

Castryck, W., Decru, T. (2022). "An efficient key recovery attack on $"SIDH"$." EUROCRYPT.

Beullens, W. (2022). "Breaking Rainbow Takes a Weekend on a Laptop." CRYPTO.

Gidney, C., Ekerå, M. (2021). "How to factor 2048-bit $"RSA"$ integers in 8 hours using 20 million noisy qubits." Quantum.

Schwabe, P., Stebila, D., Wiggers, T. (2020). "Post-Quantum $"TLS"$ Without Handshake Signatures." CCS (KEMTLS).

Ravi, P. et al. (2023). "Side-channel attacks on lattice-based KEMs." ePrint.

NSA (2022). "Commercial National Security Algorithm Suite 2.0."

Westerbaan, B., Stebila, D. (2024). "X25519 Kyber768 Draft Hybrid Post-Quantum Key Agreement." IETF draft.

= Digital Signatures

A digital signature binds a message to a public key such that anyone can verify the binding but only the holder of the private key can produce it. Signatures provide authenticity, integrity, and non-repudiation, and they underpin TLS certificates, software updates, blockchains, and secure boot. This chapter covers security definitions, the RSA and discrete-log signature families, modern deterministic schemes, and aggregate and threshold constructions.

*See also:* _Asymmetric Cryptography_ (RSA, elliptic curves), _Hashing and MACs_ (hash-then-sign, collision resistance), _Key Exchange and PKI_ (certificates), _Post-Quantum Cryptography_ (Dilithium, SPHINCS+).

== Security Definitions

A signature scheme is a triple of algorithms:
- $"KeyGen"() -> ("sk", "pk")$: generate a key pair.
- $"Sign"("sk", m) -> sigma$: produce a signature.
- $"Verify"("pk", m, sigma) -> {0, 1}$: accept or reject.

The standard security notion is *existential unforgeability under chosen-message attack* (EUF-CMA): an adversary with access to a signing oracle must not produce a valid signature on any message it did not query. A stronger notion, *strong unforgeability* (SUF-CMA), additionally forbids producing a _new_ signature on a previously signed message — relevant when signatures are malleable (ECDSA is malleable: $(r, s)$ and $(r, n - s)$ both verify).

*Hash-then-sign*: practical schemes sign $H(m)$ rather than $m$ itself. Security then also rests on the collision resistance of $H$: a collision $H(m_1) = H(m_2)$ lets an attacker transfer a signature from $m_1$ to $m_2$. This is why MD5 and SHA-1 certificates were catastrophic (the Flame malware forged a Microsoft code-signing certificate via an MD5 chosen-prefix collision).

== RSA Signatures

=== Textbook RSA and Its Failure

Textbook RSA signing ($sigma = m^d mod N$) is insecure: it is malleable ($sigma_1 sigma_2$ signs $m_1 m_2$) and allows existential forgery (pick $sigma$, compute $m = sigma^e$). All practical schemes pad first.

=== PKCS#1 v1.5

The legacy padding encodes $"0x00 0x01 0xFF" dots "0xFF 0x00" || "DigestInfo" || H(m)$ and signs the result. It has no security proof but no practical attacks when implemented correctly. The famous *Bleichenbacher 2006 forgery* exploited verifiers that did not check the padding extended to the full modulus length, allowing forgeries against small exponents ($e = 3$) by cube-rooting a crafted value. Variants of this bug have recurred for over a decade (BERserk, recent JavaScript library CVEs).

=== RSA-PSS

*Probabilistic Signature Scheme* (Bellare & Rogaway, 1996) is randomised padding with a tight security proof in the random-oracle model: forging PSS is provably as hard as inverting RSA. PSS is the recommended RSA mode (mandatory in TLS 1.3 for RSA signatures). Salt length is typically equal to the hash length.

== Discrete-Log Signatures

=== Schnorr Signatures

The conceptual ancestor of modern schemes. With generator $g$ of a prime-order group, secret $x$, public $y = g^x$:

1. Pick random nonce $k$; compute $R = g^k$.
2. Compute challenge $e = H(R || m)$.
3. Compute $s = k + e x mod q$. Signature: $(R, s)$ or $(e, s)$.

Verification: check $g^s = R y^e$. Schnorr is derived from the Schnorr identification protocol via the *Fiat-Shamir transform* (see _Zero-Knowledge Proofs_) and has a clean security proof under discrete log in the random-oracle model. Schnorr signatures are *linear*, which enables multi-signatures and aggregation (MuSig2, Bitcoin Taproot).

=== DSA and ECDSA

DSA (proposed by NIST in 1991, standardised as FIPS 186 in 1994) is a Schnorr variant designed around the then-active Schnorr patent. ECDSA is its elliptic-curve form, ubiquitous in TLS, Bitcoin, and code signing. Signing with key $d$, nonce $k$, and base point $G$ of order $n$:

$ r = (k G)_x mod n, quad s = k^(-1) (H(m) + r d) mod n. $

*The nonce is the Achilles heel.* If $k$ repeats across two messages, the private key follows from two linear equations: this broke the Sony PS3 firmware signing key (2010, constant $k$) and numerous Bitcoin wallets. Even _partial_ nonce bias is fatal: lattice attacks (Howgrave-Graham & Smart; the Minerva and TPM-Fail attacks, 2019) recover keys from a few hundred signatures with only a few bits of bias per nonce. The standard mitigation is *RFC 6979 deterministic ECDSA*, which derives $k$ from the private key and $H(m)$ via HMAC-DRBG, removing the RNG from the hot path.

=== EdDSA and Ed25519

EdDSA (Bernstein et al., 2011) is a deterministic Schnorr-style scheme over twisted Edwards curves. Ed25519 (Curve25519, SHA-512) is its dominant instantiation:

- *Deterministic nonces*: $k = H("prefix" || m)$, where prefix is derived from the secret seed — no RNG failures.
- *Complete addition formulas*: no exceptional cases, easing constant-time implementation.
- *Fast and small*: 64-byte signatures, 32-byte keys; batch verification supported.

Ed25519 is the default for SSH keys, Signal, and most modern protocols. One caveat: deterministic nonces make the scheme more vulnerable to *fault attacks* (glitch one of two identical signings and the key leaks); hardened implementations add randomness to the nonce derivation ("hedged" signing, as in XEdDSA and the CFRG hedged-signature drafts).

== Pairing-Based and Aggregate Signatures

*BLS signatures* (Boneh, Lynn & Shacham, 2001) use bilinear pairings $e: G_1 times G_2 -> G_T$. Signing is one scalar multiplication: $sigma = x H(m)$, verified by $e(sigma, g_2) = e(H(m), "pk")$. The killer feature is *aggregation*: $n$ signatures on $n$ messages combine into a single group element verified in one (multi-)pairing. Ethereum's beacon chain aggregates tens of thousands of validator signatures per slot with BLS over BLS12-381. Cost: pairings are slower than ECDSA verification, and the scheme requires careful defence against *rogue-key attacks* (proofs of possession or message augmentation).

== Threshold and Multi-Signatures

- *Multi-signature*: $n$ signers jointly produce one signature. MuSig2 (Nick, Ruffing & Seurin, 2021) achieves two-round Schnorr multi-signatures indistinguishable from single-key signatures.
- *Threshold signature* ($t$-of-$n$): any $t$ shareholders can sign; fewer learn nothing. FROST (Komlo & Goldberg, 2020) is the standard Schnorr threshold scheme; GG18/GG20 and CGGMP21 provide threshold ECDSA (used by custody providers and MPC wallets). The private key never exists in one place — key generation itself is distributed (DKG).

== Post-Quantum Signatures

Shor's algorithm breaks RSA, DSA, and all elliptic-curve schemes. NIST standardised (2024):
- *ML-DSA (Dilithium)*: lattice-based (module-LWE), the general-purpose default; signatures $approx 2.4$–$4.6$ KB depending on security level.
- *SLH-DSA (SPHINCS+)*: stateless hash-based; conservative security, large signatures ($approx 8$–$50$ KB).
- *FN-DSA (Falcon)*: lattice (NTRU); compact but requires floating-point Gaussian sampling that is hard to make constant-time.

See _Post-Quantum Cryptography_ for the underlying problems and migration strategy.

== Implementation Pitfalls

- *Nonce reuse or bias* (ECDSA): use RFC 6979 or Ed25519.
- *Signature malleability*: enforce low-$s$ ECDSA (Bitcoin BIP-62) or use Schnorr/Ed25519; check SUF-CMA if signatures are used as identifiers.
- *Verification laxity*: always check $r, s in [1, n-1]$; reject the identity point; validate that the public key is on the curve (invalid-curve attacks).
- *Cross-protocol reuse*: never use one key for both signing and decryption, or across protocol contexts without domain separation in the hash.
- *Faults*: verify the signature after signing (sign-then-verify) in high-assurance settings; randomise deterministic nonces in fault-prone environments.

== Further Reading

- Bellare, M., & Rogaway, P. (1996). The exact security of digital signatures — how to sign with RSA and Rabin. _EUROCRYPT_.
- Pornin, T. (2013). Deterministic usage of DSA and ECDSA. _RFC 6979_.
- Bernstein, D. J. et al. (2012). High-speed high-security signatures (Ed25519). _Journal of Cryptographic Engineering_, 2(2).
- Boneh, D., Lynn, B., & Shacham, H. (2001). Short signatures from the Weil pairing. _ASIACRYPT_.
- Komlo, C., & Goldberg, I. (2020). FROST: flexible round-optimized Schnorr threshold signatures. _SAC_.
- Nick, J., Ruffing, T., & Seurin, Y. (2021). MuSig2: simple two-round Schnorr multi-signatures. _CRYPTO_.

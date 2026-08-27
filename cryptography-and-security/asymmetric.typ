#import "../template.typ": rfc, xref

= Asymmetric Cryptography <asymmetric>

Public-key cryptography rests on number-theoretic problems believed hard for classical computers: integer factorization (RSA), discrete log in $(ZZ slash p)^*$ and elliptic curve groups (DH, ECDSA, EdDSA), and bilinear pairings (BLS signatures, identity-based encryption). Modern deployments are dominated by elliptic curves with classical hardness arguments, in transition to post-quantum.

*See also:* #xref("cryptography-and-security", "symmetric-primitives", label: "Symmetric Primitives"), #xref("cryptography-and-security", "hashing-and-macs", label: "Hashing and MACs"), #xref("cryptography-and-security", "post-quantum", label: "Post-Quantum Cryptography"), _Protocols_, _TLS (Transport Layer Security)_ (networking).

== RSA

$"RSA"$ (Rivest-Shamir-Adleman 1977) picks primes $p, q$, sets $n = p q$, public exponent $e$ (usually $65537$), private exponent $d = e^(-1) "mod" lambda(n)$. Encryption: $c = m^e mod n$; decryption: $m = c^d mod n$.

Textbook $"RSA"$ is *malleable* and deterministic. Use $"RSA"$-$"OAEP"$ for encryption (PKCS#1 v2) and $"RSA"$-$"PSS"$ for signatures. Never use $"PKCS"$#1 v1.5 in new designs (Bleichenbacher 1998 attack, still exploited via Bleichenbacher's CAT 2018, $"ROBOT"$ 2017).

```python
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization

priv = rsa.generate_private_key(public_exponent=65537, key_size=3072)
pub  = priv.public_key()

# Encryption with OAEP
ct = pub.encrypt(b"payload", padding.OAEP(
    mgf=padding.MGF1(hashes.SHA256()),
    algorithm=hashes.SHA256(), label=None))

# Signing with PSS
sig = priv.sign(b"doc", padding.PSS(
    mgf=padding.MGF1(hashes.SHA256()),
    salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
```

NIST recommends $|n| >= 3072$ for new keys ($approx 128$-bit security). $"RSA"$-$"OAEP"$ is rarely used for bulk data — encrypt a random $"AES"$ key with $"RSA"$, then $"AES"$-$"GCM"$ the payload (KEM/DEM split, see hybrid encryption below).

== Diffie–Hellman Key Agreement

The original public-key idea (Diffie-Hellman 1976): given group $G$ with generator $g$, two parties exchange $g^a$ and $g^b$ and compute shared $g^(a b)$. Hardness: Computational Diffie-Hellman (CDH) and Decisional Diffie-Hellman (DDH).

*Finite-field DH* (#rfc(7919)) requires 3072-bit groups. *Elliptic-curve DH* dominates: 256-bit curves give equivalent security at far lower cost.

== Elliptic-Curve Cryptography

An elliptic curve over $"GF"(p)$: $y^2 = x^3 + a x + b$. Points form an abelian group under chord-tangent addition with identity at infinity. Scalar multiplication $[k] P$ replaces modular exponentiation.

ECDLP (discrete log on $E("GF"(p))$) is believed exponentially hard; best generic attack is Pollard rho at $sqrt(n)$ work. 256-bit curves give $approx 128$-bit security.

=== Standard Curves

#table(
  columns: 4,
  [*Curve*], [*Field*], [*Designer*], [*Use*],
  [NIST P-256 / secp256r1], [256-bit prime], [NIST/NSA], [TLS, FIPS],
  [NIST P-384], [384-bit], [NIST], [CNSA top secret],
  [secp256k1], [256-bit Koblitz], [Certicom], [Bitcoin, Ethereum],
  [Curve25519 / Ed25519], [$2^255 - 19$], [Bernstein 2006], [TLS, SSH, Signal],
  [Curve448 / Ed448], [$2^448 - 2^224 - 1$], [Hamburg 2015], [high-security],
  [BLS12-381], [pairing-friendly], [Boneh-Lynn-Shacham], [Ethereum 2, ZK],
)

Curve25519 (used by X25519 ECDH) and Ed25519 (EdDSA signatures) are designed for *misuse resistance*: deterministic signatures, complete addition formulas (no special cases), all 32-byte strings are valid points after clamping. They are the modern default.

```rust
use ed25519_dalek::{SigningKey, Signature, Signer, Verifier};
use rand_core::OsRng;

let sk = SigningKey::generate(&mut OsRng);
let pk = sk.verifying_key();
let sig: Signature = sk.sign(b"message");
pk.verify(b"message", &sig).unwrap();
```

=== $"ECDSA"$ vs EdDSA

$"ECDSA"$ requires a per-signature nonce $k$; *nonce reuse leaks the private key instantly* (Sony PS3 famously did this in 2010). #rfc(6979) specifies deterministic ECDSA via $"HMAC"$-derived nonces. EdDSA (Ed25519/Ed448) is deterministic by design: $k = H("hash"("sk") parallel m)$ — recommended for all new code.

== Hybrid (KEM/DEM) Encryption

The standard pattern for encrypting arbitrary-length data:

1. *KEM:* generate ephemeral $(x, g^x)$; recipient public key $g^y$; shared secret $s = g^(x y)$.
2. Derive $K = "HKDF"(s parallel g^x parallel g^y)$.
3. *DEM:* $"AEAD"_K (N, m)$.

Encapsulated form is $(g^x, "ct")$. $"ECIES"$ (SECG SEC 1) formalizes this; modern HPKE (#rfc(9180)) is the cleaner, more general version used in $"TLS"$ Encrypted Client Hello, $"MLS"$, Apple Private Relay.

```python
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

eph = X25519PrivateKey.generate()
shared = eph.exchange(recipient_pub)
key = HKDF(hashes.SHA256(), 32, None, b"hybrid-v1").derive(shared)
ct = ChaCha20Poly1305(key).encrypt(nonce, msg, aad)
```

== Signatures

#table(
  columns: 4,
  [*Scheme*], [*Pub*], [*Sig*], [*Notes*],
  [$"RSA"$-$"PSS"$-3072], [384 B], [384 B], [FIPS, slow verify wrt EC],
  [$"ECDSA"$-P256], [64 B], [64 B], [must use det. nonce or RNG],
  [Ed25519], [32 B], [64 B], [deterministic, fast, batch verify],
  [Ed448], [57 B], [114 B], [higher security margin],
  [BLS12-381], [48 B / 96 B], [96 B / 48 B], [aggregatable, pairing-based],
  [Schnorr (BIP-340)], [32 B], [64 B], [Bitcoin Taproot, MuSig2],
)

=== BLS Signatures

BLS (Boneh-Lynn-Shacham 2001) uses pairings $e : G_1 times G_2 -> G_T$ on pairing-friendly curves (BLS12-381). Signatures are *aggregatable*: $sigma_("agg") = sum_i sigma_i$ verifies against $sum_i "pk"_i$. Used in Ethereum 2 consensus (validators sign attestations; aggregator combines into one signature per slot).

=== Schnorr and MuSig2

Schnorr signatures (revived in Bitcoin Taproot, BIP-340) enable interactive multi-signatures (MuSig2, Nick-Ruffing-Seurin 2021) that produce a single signature indistinguishable from a single-party signature.

== Pairing-Based Cryptography

A bilinear pairing $e : G_1 times G_2 -> G_T$ enables:

- *Identity-based encryption* (Boneh-Franklin 2001): public key = email address.
- *Short signatures* (BLS).
- *KZG polynomial commitments*: constant-size openings, foundational to PLONK and Ethereum Danksharding.
- *Pairing-based SNARKs* (Groth16).

Curves: BLS12-381 (most common), BN254 (legacy), BLS12-377 (recursion-friendly). Pairing computation is expensive (~1 ms) but produces tiny verification artifacts.

== Key Encoding

Standard formats:

- *PKCS#8* / SPKI: DER/PEM encoding for private/public keys.
- *X.509*: certificates (see _Protocols_).
- *JWK* (#rfc(7517)): JSON Web Keys for OIDC/JWT.
- *OpenSSH*: `ssh-ed25519 AAAA... user@host`.
- *COSE* (RFC 9052/9053, obsoletes #rfc(8152)): CBOR-encoded for $"WebAuthn"$, $"FIDO2"$.

```python
priv_pem = priv.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption())
```

== Common Pitfalls

- *Nonce reuse in ECDSA* exposes the private key after two signatures. Always use #rfc(6979) or Ed25519.
- *Small subgroup attacks* on Curve25519 if cofactor is not handled — X25519 clamping (clear low 3 bits, set bit 254) avoids this.
- *Invalid-curve attacks*: validate that received points are on the curve and not the identity.
- *Bleichenbacher and Manger* padding oracles on $"PKCS"$#1 v1.5 and old $"OAEP"$ implementations — use modern libraries that constant-time the padding check.
- *RSA common-modulus / low-exponent / Coppersmith*: never use $e = 3$ with raw $"RSA"$; never share modulus across users.
- *Cross-protocol attacks*: a key used for both signing and encryption can leak via decryption oracle queries (use separate keys, or HPKE-style domain separation).

== Practical Recommendations

#table(
  columns: 2,
  [*Need*], [*Use*],
  [Modern signature], [Ed25519; Ed448 for ultra-high security],
  [FIPS-required signature], [$"ECDSA"$-P256 with #rfc(6979) or $"RSA"$-$"PSS"$-3072],
  [Hybrid public-key encryption], [HPKE (#rfc(9180))],
  [Aggregate signatures], [BLS12-381],
  [Bitcoin / Taproot multisig], [Schnorr + MuSig2],
  [WebAuthn / FIDO2], [$"ECDSA"$-P256 or Ed25519 via COSE],
)

All modern public-key cryptography listed here is broken by a sufficiently large quantum computer (Shor 1994). The transition is covered in _Post-Quantum_.

== Further Reading

Rivest, R., Shamir, A., Adleman, L. (1978). "A Method for Obtaining Digital Signatures and Public-Key Cryptosystems." CACM.

Diffie, W., Hellman, M. (1976). "New Directions in Cryptography." IEEE TIT.

Bernstein, D. J. (2006). "Curve25519: new Diffie-Hellman speed records." PKC.

Bernstein, D. J., Duif, N., Lange, T., Schwabe, P., Yang, B-Y. (2012). "High-speed high-security signatures." CHES (Ed25519).

Pornin, T. (2013). "Deterministic Usage of the Digital Signature Algorithm ($"DSA"$) and Elliptic Curve $"DSA"$ ($"ECDSA"$)." #rfc(6979).

Wahby, R. S., Boneh, D. (2019). "Fast and simple constant-time hashing to the BLS12-381 elliptic curve."

Boneh, D., Lynn, B., Shacham, H. (2001). "Short Signatures from the Weil Pairing." ASIACRYPT.

Nick, J., Ruffing, T., Seurin, Y. (2021). "MuSig2: Simple Two-Round Schnorr Multi-Signatures." CRYPTO.

Barnes, R. et al. (2022). "Hybrid Public Key Encryption." #rfc(9180).

Bleichenbacher, D. (1998). "Chosen Ciphertext Attacks Against Protocols Based on the $"RSA"$ Encryption Standard $"PKCS"$#1." CRYPTO.

Coppersmith, D. (1996). "Finding a Small Root of a Univariate Modular Equation." EUROCRYPT.

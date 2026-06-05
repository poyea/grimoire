= Hashing and MACs

A cryptographic hash function $H : {0,1}^* -> {0,1}^n$ is the workhorse of modern cryptography: it underpins digital signatures, password storage, blockchain identifiers, key derivation, and Merkle proofs. MACs add a key, turning the unkeyed hash into an authenticated integrity tag.

*See also:* _Symmetric Primitives_, _Asymmetric_, _Key Management_, #emph[distributed-systems\/consensus-and-replication.typ].

== Security Properties

A hash function should satisfy:

- *Preimage resistance:* given $y$, finding $x$ with $H(x) = y$ takes $approx 2^n$ work.
- *Second-preimage resistance:* given $x_1$, finding $x_2 != x_1$ with $H(x_1) = H(x_2)$ takes $approx 2^n$ work.
- *Collision resistance:* finding any $(x_1, x_2)$ with $H(x_1) = H(x_2)$ takes $approx 2^(n / 2)$ work (birthday bound).

Therefore 256-bit hashes offer 128-bit collision security. $"SHA"$-1 (160 bits) was broken in 2017 (SHAttered, Google + CWI) for $approx 2^63$ work. $"MD5"$ is dead — chosen-prefix collisions in seconds.

== $"SHA"$-2 Family

$"SHA"$-256 / $"SHA"$-512 use the Merkle–Damgård construction over a Davies–Meyer compression function. Block sizes are 512 / 1024 bits; outputs 256 / 512 bits. $"SHA"$-384 and $"SHA"$-512/256 are truncations of $"SHA"$-512 with different IVs.

```c
// Simplified core round of SHA-256 (illustrative)
static void sha256_round(uint32_t W[64], uint32_t H[8], const uint32_t K[64]) {
    uint32_t a=H[0], b=H[1], c=H[2], d=H[3];
    uint32_t e=H[4], f=H[5], g=H[6], h=H[7];
    for (int t = 0; t < 64; t++) {
        uint32_t T1 = h + Sigma1(e) + Ch(e,f,g) + K[t] + W[t];
        uint32_t T2 = Sigma0(a) + Maj(a,b,c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }
    H[0]+=a; H[1]+=b; H[2]+=c; H[3]+=d;
    H[4]+=e; H[5]+=f; H[6]+=g; H[7]+=h;
}
```

Intel $"SHA"$-NI and ARMv8 $"SHA2"$ extensions accelerate to $approx 1.5$ cycles/byte. $"SHA"$-256 is vulnerable to *length-extension*: given $H(M)$ and $|M|$, an attacker can compute $H(M parallel "pad" parallel M')$ without knowing $M$. Defense: $"HMAC"$, $"SHA"$-512/256, or $"SHA"$-3.

== $"SHA"$-3 / Keccak

$"SHA"$-3 (FIPS 202) uses the *sponge construction* over the Keccak-$f$[1600] permutation: absorb input into rate $r$, squeeze output. $"SHAKE"$-128 / $"SHAKE"$-256 are extendable-output functions (XOFs) — produce any output length. $"cSHAKE"$, $"KMAC"$, $"TupleHash"$, $"ParallelHash"$ (SP 800-185) build on the same permutation.

```python
import hashlib
h = hashlib.sha3_256(b"data").hexdigest()

# SHAKE: arbitrary-length output
shake = hashlib.shake_256(b"seed")
ks_64 = shake.digest(64)   # 64-byte output
```

The sponge naturally resists length extension. No retrofit needed.

== $"BLAKE3"$

$"BLAKE3"$ (Aumasson, O'Connor, Schmidt, Wilcox-O'Hearn 2020) is a single function with PRF, hash, MAC, KDF, and XOF roles. Internally a Merkle tree of compression chunks; parallel within and across chunks; SIMD-friendly. Throughput on Skylake-X is $approx 6.8$ GB/s single-threaded, $> 30$ GB/s with AVX-512 and threads.

```rust
let hash = blake3::hash(b"hello world");           // 32-byte
let keyed = blake3::keyed_hash(&[0u8; 32], data);  // MAC
let mut xof = blake3::Hasher::new().update(b"x").finalize_xof();
let mut out = [0u8; 1024];
xof.fill(&mut out);
```

#table(
  columns: 4,
  [*Property*], [$"SHA"$-256], [$"SHA3"$-256], [$"BLAKE3"$],
  [Construction], [Merkle–Damgård], [sponge], [Merkle tree (BLAKE2 compress)],
  [Output], [256 bits fixed], [256 bits + $"SHAKE"$ XOF], [arbitrary XOF],
  [Length extension], [vulnerable], [immune], [immune],
  [SIMD friendly], [moderate ($"SHA"$-NI)], [poor], [excellent],
  [Throughput], [\~1.5 cpb HW], [\~10 cpb], [\~1 cpb SIMD],
  [Tree mode], [no], [no], [native],
  [Standard], [FIPS 180-4], [FIPS 202], [community spec],
)

== Message Authentication Codes

A MAC is a keyed function $T = "MAC"_K (M)$ such that no adversary without $K$ can produce a valid $(M, T)$ on a fresh $M$.

=== $"HMAC"$

$"HMAC"$ (RFC 2104) wraps any Merkle–Damgård hash and is provably secure under PRF assumption on the compression function. Pad key to block size, XOR with $"ipad"$/$"opad"$, hash twice:

$ "HMAC"_K (M) = H((K plus.o "opad") parallel H((K plus.o "ipad") parallel M)) $

```python
import hmac, hashlib
tag = hmac.new(key, msg, hashlib.sha256).digest()
assert hmac.compare_digest(tag, expected)   # constant-time compare
```

=== $"KMAC"$ and $"BLAKE3"$ Keyed Hash

$"SHA"$-3 supports a native $"KMAC"$ mode (SP 800-185) that prepends the key inside the sponge — no $"HMAC"$ wrapping needed. $"BLAKE3"$'s `keyed_hash` is equally direct.

=== Polynomial MACs

$"Poly1305"$, $"GMAC"$ are *one-time* MACs over a prime field: the key must be unique per message (typically derived from a long-term key + nonce). Used inside AEAD constructions ($"ChaCha20"$-$"Poly1305"$, $"AES"$-$"GCM"$).

== Password Hashing and KDFs

Generic hashes are too fast for passwords. Use *memory-hard* functions that resist GPU/ASIC parallelism.

#table(
  columns: 3,
  [*Function*], [*Year*], [*Use*],
  [$"PBKDF2"$], [2000], [legacy, FIPS-required only],
  [$"bcrypt"$], [1999], [moderate, fixed 72-byte input limit],
  [$"scrypt"$], [2009], [memory-hard, CPU/RAM tradeoff],
  [$"Argon2id"$], [2015], [recommended (OWASP, RFC 9106)],
)

```python
from argon2 import PasswordHasher
ph = PasswordHasher(time_cost=3, memory_cost=64*1024, parallelism=4)
hash_str = ph.hash("correct horse battery staple")
ph.verify(hash_str, "correct horse battery staple")
```

OWASP 2023 recommends $"Argon2id"$ $(t=2, m = 19 "MiB", p=1)$ minimum, or $"scrypt"$ $(N=2^17, r=8, p=1)$.

*KDFs* expand or extract entropy. $"HKDF"$ (RFC 5869) is the standard: extract then expand.

```python
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
k = HKDF(algorithm=hashes.SHA256(), length=32,
         salt=salt, info=b"tls13 c hs traffic").derive(shared_secret)
```

== Merkle Trees

A Merkle tree commits to a sequence of leaves with logarithmic-size membership proofs. Foundational to Git, Bitcoin, Certificate Transparency, IPFS, BitTorrent, Ethereum state.

```text
        root
       /    \
     h01     h23
    /  \    /  \
   h0  h1  h2  h3
   |   |   |   |
   L0  L1  L2  L3
```

Proof that $L_2$ is in the tree: $(h_3, h_(01))$, $O(log n)$ size. Verify by recomputing.

```python
def verify_proof(leaf, proof, root, index):
    h = sha256(b"\x00" + leaf).digest()    # domain separation for leaves
    for sib in proof:
        if index & 1:
            h = sha256(b"\x01" + sib + h).digest()
        else:
            h = sha256(b"\x01" + h + sib).digest()
        index >>= 1
    return h == root
```

*Second-preimage on Merkle trees*: without domain separation between leaf and internal nodes, an attacker can swap subtrees. The Bitcoin Merkle root has this flaw; Certificate Transparency (RFC 6962) and Signed Merkle Trees fix it with leaf/node tags.

*Sparse Merkle Trees* (SMTs) and *Verkle trees* (Kuszmaul 2018; using vector commitments) provide more compact proofs for stateful systems like Ethereum.

== Commitment Schemes

A commitment $c = "Com"(m, r)$ is *hiding* (reveals nothing about $m$) and *binding* (cannot be opened to two different $m$). $H(m parallel r)$ for random $r$ is a generic, computationally binding, computationally hiding commitment. Pedersen commitments give information-theoretic hiding (see _Zero-Knowledge_).

== Universal Hashing

Carter–Wegman universal hashing is used inside polynomial MACs and for hash tables resistant to collision DoS (SipHash, used in Python `dict`, Rust `HashMap`, $"FreeBSD"$ DNS).

```c
// SipHash-2-4 (Aumasson-Bernstein) — 128-bit key, 64-bit output
uint64_t siphash_2_4(const uint8_t *in, size_t inlen, const uint8_t key[16]);
```

== Practical Recommendations

#table(
  columns: 2,
  [*Need*], [*Use*],
  [General hash, FIPS], [$"SHA"$-256 or $"SHA"$-384],
  [General hash, max speed], [$"BLAKE3"$],
  [Hash with XOF], [$"SHAKE"$-256 or $"BLAKE3"$],
  [MAC, FIPS], [$"HMAC"$-$"SHA"$-256],
  [MAC, modern], [$"BLAKE3"$ keyed or $"KMAC"$],
  [Password hashing], [$"Argon2id"$ (preferred) or $"scrypt"$],
  [KDF from high-entropy secret], [$"HKDF"$-$"SHA"$-256],
  [Hash-table keying], [SipHash or $"HighwayHash"$],
)

Never use $"MD5"$ or $"SHA"$-1 for security. Never use a raw hash for passwords. Never roll your own MAC by concatenation ($H(K parallel M)$ enables length-extension on Merkle–Damgård hashes).

== Further Reading

NIST FIPS 180-4 (2015). "Secure Hash Standard."

NIST FIPS 202 (2015). "$"SHA"$-3 Standard: Permutation-Based Hash and Extendable-Output Functions."

NIST SP 800-185 (2016). "$"SHA"$-3 Derived Functions: $"cSHAKE"$, $"KMAC"$, $"TupleHash"$, $"ParallelHash"$."

Krawczyk, H., Bellare, M., Canetti, R. (1997). "$"HMAC"$: Keyed-Hashing for Message Authentication." RFC 2104.

Krawczyk, H., Eronen, P. (2010). "$"HKDF"$: $"HMAC"$-based Extract-and-Expand Key Derivation Function." RFC 5869.

Biryukov, A., Dinu, D., Khovratovich, D. (2016). "$"Argon2"$." RFC 9106.

Percival, C. (2009). "Stronger Key Derivation via Sequential Memory-Hard Functions." (scrypt.)

Aumasson, J-P., O'Connor, J., Schmidt, S., Wilcox-O'Hearn, Z. (2020). "$"BLAKE3"$: one function, fast everywhere."

Stevens, M. et al. (2017). "The first collision for full $"SHA"$-1." CRYPTO (SHAttered).

Laurie, B., Langley, A., Kasper, E. (2013). "Certificate Transparency." RFC 6962.

Aumasson, J-P., Bernstein, D. J. (2012). "$"SipHash"$: a fast short-input PRF." INDOCRYPT.

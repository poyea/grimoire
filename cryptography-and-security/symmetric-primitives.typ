= Symmetric Primitives

Symmetric cryptography is built from a small set of well-studied primitives: block ciphers (AES), stream ciphers (ChaCha20), polynomial MACs (Poly1305, GMAC), and the AEAD modes that combine them. Choosing the right mode, using nonces correctly, and writing constant-time code matter far more than the underlying algorithm.

*See also:* _Hashing and MACs_, _Protocols_, _Key Exchange and PKI_, _TLS (Transport Layer Security)_ (networking), _Database Security and Privacy_ (database).

== Block Ciphers

A *block cipher* is a keyed permutation $E_k : {0,1}^n -> {0,1}^n$ that is indistinguishable from a uniformly random permutation (PRP security). $"AES"$ has $n = 128$ and key sizes 128, 192, 256 bits.

=== $"AES"$ Internals

$"AES"$-128 has 10 rounds; each round applies $"SubBytes"$ (S-box), $"ShiftRows"$, $"MixColumns"$ (in $"GF"(2^8)$), and $"AddRoundKey"$. The final round skips $"MixColumns"$.

```c
// Round body (illustrative; production uses AES-NI intrinsics)
static void aes_round(uint8_t state[16], const uint8_t rk[16]) {
    sub_bytes(state);     // 16 parallel S-box lookups
    shift_rows(state);
    mix_columns(state);
    for (int i = 0; i < 16; i++) state[i] ^= rk[i];
}
```

The S-box is the only non-linear step: it is $x mapsto x^(-1)$ in $"GF"(2^8)$ followed by an affine map. Lookup-table implementations are vulnerable to cache-timing attacks (Bernstein 2005, Osvik-Shamir-Tromer 2006); use $"AES"$-NI on x86 or the bitsliced reference (Käsper-Schwabe) when hardware support is absent.

=== $"AES"$-NI Intrinsics

```c
#include <wmmintrin.h>
// One AES round using hardware instruction
__m128i aes_round_ni(__m128i state, __m128i round_key) {
    return _mm_aesenc_si128(state, round_key);
}
// Final round uses _mm_aesenclast_si128 (no MixColumns)
```

$"ARMv8"$ provides $"AESE"$/$"AESMC"$; Apple Silicon, Graviton, modern Snapdragon all accelerate $"AES"$ to roughly 1 cycle/byte.

== Modes of Operation

A block cipher alone encrypts only one block. Modes turn it into something usable for arbitrary-length messages.

#table(
  columns: 4,
  [*Mode*], [*Type*], [*Properties*], [*Use*],
  [$"ECB"$], [det.], [leaks structure], [never],
  [$"CBC"$], [confid.], [needs IV + MAC], [legacy TLS],
  [$"CTR"$], [stream], [parallel; nonce must be unique], [building block],
  [$"GCM"$], [AEAD], [CTR + GHASH], [TLS 1.3, IPsec],
  [$"CCM"$], [AEAD], [CTR + CBC-MAC], [802.11i WPA2],
  [$"GCM"$-$"SIV"$], [AEAD], [nonce-misuse-resistant], [logs, deterministic],
  [$"OCB3"$], [AEAD], [single-pass, patented (now free)], [rare],
  [$"XTS"$], [tweakable], [length-preserving, sector enc.], [disk encryption],
)

*ECB is broken*: identical plaintext blocks produce identical ciphertext blocks (the "Tux penguin" demonstration). Never use ECB.

== Authenticated Encryption (AEAD)

AEAD primitives provide *confidentiality + integrity + binding to associated data* in one call. The interface:

$ "Enc"_K (N, A, P) -> C, quad "Dec"_K (N, A, C) -> P "or" bot $

where $N$ is a nonce, $A$ associated data (authenticated but not encrypted), $P$ plaintext, $C$ ciphertext (including tag).

=== $"AES"$-$"GCM"$

$"GCM"$ = $"CTR"$ mode for encryption + $"GHASH"$ (polynomial MAC over $"GF"(2^128)$) for authentication. Hardware-accelerated via $"PCLMULQDQ"$.

```python
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

key = AESGCM.generate_key(bit_length=256)
aesgcm = AESGCM(key)

nonce = os.urandom(12)          # 96 bits, MUST be unique per (key, message)
aad = b"header-v1"
plaintext = b"secret payload"

ct = aesgcm.encrypt(nonce, plaintext, aad)
pt = aesgcm.decrypt(nonce, ct, aad)   # raises InvalidTag if AAD/ct modified
```

*Critical pitfall:* $"GCM"$ catastrophically fails on nonce reuse — two messages with the same $(K, N)$ reveal $"GHASH"$ key and allow arbitrary forgeries (Joux's forbidden attack). Counter-based nonces or $"AES"$-$"GCM"$-$"SIV"$ (which is nonce-misuse-resistant) are the practical defenses.

=== $"ChaCha20"$-$"Poly1305"$

$"ChaCha20"$ is a 20-round ARX stream cipher; $"Poly1305"$ is a one-time polynomial MAC in $"GF"(2^130 - 5)$. Together they form RFC 8439 AEAD — the default for TLS on mobile (no $"AES"$-NI required, no cache-timing risk).

```rust
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce, aead::{Aead, KeyInit}};

let key = Key::from_slice(&[0u8; 32]);
let cipher = ChaCha20Poly1305::new(key);
let nonce = Nonce::from_slice(&[0u8; 12]);
let ct = cipher.encrypt(nonce, b"hello".as_ref()).unwrap();
let pt = cipher.decrypt(nonce, ct.as_ref()).unwrap();
```

#table(
  columns: 4,
  [*Property*], [$"AES"$-$"GCM"$], [$"ChaCha20"$-$"Poly1305"$], [$"AES"$-$"GCM"$-$"SIV"$],
  [Cipher type], [block + CTR], [ARX stream], [block + CTR],
  [HW accel.], [$"AES"$-NI, $"PCLMULQDQ"$], [SSSE3/AVX2/NEON], [$"AES"$-NI],
  [Nonce size], [96 bits], [96 bits], [96 bits],
  [Nonce reuse], [catastrophic], [catastrophic], [graceful (reveals only equality)],
  [Throughput], [\~4 GB/s with HW], [\~2 GB/s], [\~3 GB/s with HW],
  [Standard], [NIST SP 800-38D], [RFC 8439], [RFC 8452],
)

=== Nonce-Misuse-Resistant: $"AES"$-$"GCM"$-$"SIV"$

$"GCM"$-$"SIV"$ derives the encryption nonce from the plaintext (synthetic IV), so reusing an external nonce only leaks plaintext equality, not the key. Suitable for deterministic encryption, log-tail re-encryption, and systems where uniqueness is hard.

== Stream Ciphers

A stream cipher generates a pseudorandom keystream $Z = "PRF"(K, N)$ that is XORed with plaintext.

```c
// Constant-time XOR of equal-length buffers
static void xor_block(uint8_t *out, const uint8_t *a, const uint8_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] ^ b[i];
}
```

$"ChaCha20"$ is now the standard. $"RC4"$ is broken (biased keystream, used in old $"WEP"$/$"TLS"$). $"Salsa20"$ is $"ChaCha20"$'s predecessor.

== Constant-Time Programming

Side-channel attacks recover keys from timing, cache behavior, branch prediction, or power. Code that handles secrets must be *constant-time*: control flow and memory access patterns must not depend on secret data.

```c
// WRONG: data-dependent branch leaks key bit timing
if (key[i] == 0) do_thing();

// WRONG: data-dependent table access leaks via cache
return sbox[secret_byte];

// RIGHT: constant-time conditional move
static inline uint32_t ct_select(uint32_t mask, uint32_t a, uint32_t b) {
    // mask is all-1s or all-0s; selects a if mask, else b
    return (mask & a) | (~mask & b);
}

// RIGHT: constant-time memcmp (returns 0 iff equal)
int ct_memcmp(const uint8_t *a, const uint8_t *b, size_t n) {
    uint8_t diff = 0;
    for (size_t i = 0; i < n; i++) diff |= a[i] ^ b[i];
    return diff;   // 0 iff equal; do not branch on intermediate values
}
```

Tag comparison in AEAD decryption must use constant-time compare; an early-exit compare is the classic Lucky 13 / CRIME family. In Python use `hmac.compare_digest`; in Go `crypto/subtle.ConstantTimeCompare`; in Rust `subtle::ConstantTimeEq`.

== Tweakable and Length-Preserving Encryption

$"XTS"$-$"AES"$ (IEEE 1619) is used for full-disk encryption ($"BitLocker"$, $"FileVault"$, LUKS, dm-crypt) because sector size must equal ciphertext size. $"XTS"$ provides no integrity — it stops bulk decryption but not chosen-ciphertext attacks; pair with a per-sector MAC or use $"AES"$-$"GCM"$ on a higher layer if integrity matters.

Format-preserving encryption ($"FF1"$, $"FF3-1"$, NIST SP 800-38G) encrypts a credit-card number to another 16-digit number; used for tokenization. $"FF3-1"$ was attacked in 2020 (Durak-Vaudenay), reducing recommended domain sizes.

== Memory Hygiene

Wipe key material after use; compilers will optimize away naive `memset`. Use platform primitives:

```c
#ifdef _WIN32
SecureZeroMemory(key, sizeof(key));
#else
explicit_bzero(key, sizeof(key));  // glibc, *BSD
// or memset_s on C11 Annex K
#endif
```

Pin pages with `mlock` to prevent swap leakage; on Linux 5.14+ use `memfd_secret` for keys never mapped into userspace.

== Practical Recommendations

#table(
  columns: 2,
  [*Situation*], [*Recommended*],
  [General-purpose AEAD], [$"AES"$-256-$"GCM"$ if $"AES"$-NI; else $"ChaCha20"$-$"Poly1305"$],
  [Long-lived keys, many messages], [$"AES"$-$"GCM"$-$"SIV"$ or rekey via $"HKDF"$],
  [Disk sector encryption], [$"AES"$-$"XTS"$-256 + optional dm-integrity],
  [Tokenization], [$"FF3-1"$ with sufficiently large domain],
  [Embedded / no $"AES"$ HW], [$"ChaCha20"$-$"Poly1305"$ or $"Ascon"$ (NIST LWC winner)],
)

NIST's Lightweight Cryptography competition (2023) selected $"Ascon"$ for constrained devices; its permutation is also a foundation for hash and MAC variants.

== Further Reading

Daemen, J., Rijmen, V. (2002). "The Design of Rijndael: $"AES"$ - The Advanced Encryption Standard." Springer.

NIST FIPS 197 (2001). "Advanced Encryption Standard."

NIST SP 800-38D (2007). "Recommendation for Block Cipher Modes of Operation: Galois/Counter Mode."

Bernstein, D. J. (2008). "$"ChaCha"$, a variant of $"Salsa20"$."

Nir, Y., Langley, A. (2018). "$"ChaCha20"$ and $"Poly1305"$ for IETF Protocols." RFC 8439.

Gueron, S., Langley, A., Lindell, Y. (2017). "$"AES"$-$"GCM"$-$"SIV"$: Specification and Analysis." RFC 8452.

Joux, A. (2006). "Authentication Failures in NIST version of $"GCM"$." (Forbidden attack.)

Bernstein, D. J. (2005). "Cache-timing attacks on $"AES"$."

Käsper, E., Schwabe, P. (2009). "Faster and Timing-Attack Resistant $"AES"$-$"GCM"$." CHES.

Dobraunig, C. et al. (2021). "$"Ascon"$ v1.2." NIST Lightweight Cryptography Standard.

Durak, F. B., Vaudenay, S. (2020). "Breaking the $"FF3"$ Format-Preserving Encryption Standard." CRYPTO.

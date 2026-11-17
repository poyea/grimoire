#import "../template.typ": xref

= Randomness and Key Management <randomness-and-key-management>

Cryptographic algorithms are rarely what fails in practice. What fails is the material they are given: keys generated from too little entropy, nonces repeated, secrets stored in source code, keys never rotated or rotated so carelessly that data becomes unreadable. Every security proof assumes keys are uniformly random and secret, and every deployed system has to make that assumption true over years of operation. This chapter covers where randomness comes from and the real failures caused by its absence, operating-system random number generators and how to use them, generating random values without bias in C++, deterministic alternatives for nonces, key derivation, the key lifecycle, key hierarchies and envelope encryption, hardware security modules and key management services, and rotation.

*See also:* #xref("cryptography-and-security", "symmetric-primitives", label: "Symmetric Primitives") (nonces and memory hygiene), #xref("cryptography-and-security", "hashing-and-macs", label: "Hashing and MACs") (HMAC and KDFs), #xref("cryptography-and-security", "digital-signatures", label: "Digital Signatures"), #xref("cloud-and-infrastructure", "cloud-storage", label: "Cloud Storage Services") (envelope encryption in practice), #xref("numerical-computing", "monte-carlo-methods", label: "Monte Carlo Methods and Random Numbers") (non-cryptographic generators).

== Why Randomness Fails

Cryptographic randomness has a stricter requirement than statistical randomness: outputs must be *unpredictable* to an adversary who knows the algorithm and may have seen earlier outputs. A generator that passes every statistical test can be completely predictable. Failures have been frequent and severe:

- *Debian OpenSSL (2008).* A patch meant to silence a memory-checking warning removed the code that mixed entropy into OpenSSL's pool on Debian and Ubuntu. For nearly two years, keys were seeded essentially only by the process ID, so only about 32,768 SSH and TLS keys of each type and size could be generated. Every one could be enumerated.
- *Weak keys at boot.* Heninger and colleagues (2012) collected public keys from across the internet and found that a fraction of a percent of TLS hosts shared RSA prime factors with other hosts, allowing their private keys to be computed with a GCD. The cause was headless devices (routers, firewalls) generating keys at first boot, before their kernels had gathered entropy.
- *Repeated ECDSA nonces.* An ECDSA signature uses a fresh secret value $k$. If two signatures use the same $k$, the private key follows from two equations in two unknowns. Sony's PlayStation 3 used a constant $k$ (2010), exposing its signing key; a 2013 Android random number generator bug led to repeated nonces and stolen Bitcoin.
- *Backdoored generators.* The Dual_EC_DRBG generator, standardised by NIST, was shown to allow anyone who knew a secret relation between its two curve points to predict outputs; it was withdrawn in 2014. In 2015, Juniper's ScreenOS was found to use Dual_EC with points changed by an unknown party.
- *Virtual machine snapshots.* Restoring the same snapshot twice, or cloning a VM, duplicates the random generator's state, so two machines produce identical "random" keys and nonces unless the generator is reseeded on resume.

== Operating-System Generators

The right source of cryptographic randomness is the operating system's generator, which collects entropy from hardware events and from hardware random number instructions, mixes it into a pool, and expands it with a cryptographic algorithm (a *deterministic random bit generator*, DRBG).

#table(
  columns: 2,
  [*Platform*], [*Interface*],
  [Linux], [`getrandom(2)` (blocks only until the pool is first initialised, then never), or `/dev/urandom`],
  [macOS, BSDs], [`getentropy`, `arc4random_buf`],
  [Windows], [`BCryptGenRandom` (or `ProcessPrng`)],
  [Browsers], [`crypto.getRandomValues`],
)

Linux's generator was rewritten around 2022 to use BLAKE2s for entropy extraction and ChaCha20 for output, and `/dev/random` and `/dev/urandom` now behave the same once initialised. The long-standing advice to read from `/dev/random` because it "blocks when entropy runs out" is obsolete: entropy does not run out once a cryptographic generator is properly seeded, and blocking only caused outages. The one real hazard is very early boot, before seeding, which `getrandom` handles by blocking until the pool is ready.

Hardware instructions (`RDRAND`, `RDSEED` on x86, `RNDR` on Arm) are useful entropy inputs but should be mixed with other sources rather than trusted alone: they are not auditable, and some implementations have had bugs (certain AMD processors returned constant values after suspend and resume). The OS generator mixes them in.

Standards describe the construction: NIST SP 800-90A specifies approved DRBGs (Hash_DRBG, HMAC_DRBG, CTR_DRBG), 800-90B the assessment of entropy sources, and 800-90C their combination. Fortuna (Ferguson and Schneier) is a design that recovers from state compromise by spreading incoming entropy across pools that are used at exponentially decreasing frequencies.

Rules for application code:

- Use the OS generator, through `getrandom` or the platform equivalent, or through a well-reviewed library that wraps it.
- Never use `rand()`, `std::mt19937` or other statistical generators for keys, tokens, nonces, salts or session identifiers. Their state can be reconstructed from a few outputs.
- Avoid user-space generators that cache state across `fork()`: a child process that inherits the parent's generator state produces the same outputs as its sibling.

== Uniform Values Without Bias

Turning random bits into a value in a range is a frequent source of subtle bias. `random() % n` favours small values whenever $n$ does not divide the generator's range: with a 32-bit source and $n = 3 dot 10^9$, values below $2^32 - 3 dot 10^9 approx 1.29 dot 10^9$ are twice as likely as the others. For a key or a lottery that matters. *Rejection sampling* removes the bias:

```cpp
#include <sys/random.h>   // getrandom (Linux, glibc 2.25+)
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

void fill_random(void* buffer, std::size_t length) {
    auto* p = static_cast<unsigned char*>(buffer);
    while (length > 0) {
        const ssize_t n = getrandom(p, length, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("getrandom failed");
        }
        p += n;
        length -= static_cast<std::size_t>(n);
    }
}

std::uint64_t random_u64() {
    std::uint64_t v;
    fill_random(&v, sizeof v);
    return v;
}

// Uniform integer in [0, bound), bound > 0, with no modulo bias.
std::uint64_t uniform_below(std::uint64_t bound) {
    // 2^64 mod bound, computed without overflow: values below it are rejected,
    // leaving a range whose size is a multiple of bound.
    const std::uint64_t threshold = (0 - bound) % bound;
    for (;;) {
        const std::uint64_t r = random_u64();
        if (r >= threshold) return r % bound;
    }
}
```

The expected number of draws is below 2 for any bound and close to 1 for small bounds. Random tokens for URLs or passwords are generated the same way, one uniform index into the alphabet per character. A token needs enough entropy for its purpose: 128 bits (for example 22 characters from a 64-symbol alphabet) makes guessing infeasible for any online or offline attacker.

== Avoiding the Need for Randomness

Where randomness is risky, deterministic constructions remove it:

- *Deterministic signatures*: RFC 6979 derives the ECDSA nonce $k$ from the private key and the message hash with HMAC_DRBG, so the same message always gets the same signature and a bad generator cannot leak the key. EdDSA (Ed25519) builds this in. Hedged variants mix in fresh randomness as well, to resist fault attacks while remaining safe if the randomness fails.
- *Counter nonces*: AEAD modes such as AES-GCM need unique, not random, nonces. A message counter within one key is safe as long as the counter never repeats, including across restarts and replicas, which is where counters fail. With random 96-bit nonces, the collision probability after $2^32$ messages under one key reaches about $2^(-33)$, the limit NIST recommends; long-lived high-volume keys need rotation or nonce-misuse-resistant modes (#xref("cryptography-and-security", "symmetric-primitives", label: "Symmetric Primitives")).
- *Key derivation instead of generation*: deriving many keys from one well-generated master key reduces the number of times fresh randomness is needed.

== Key Derivation

A *key derivation function* turns one secret into keys for specific purposes. HKDF (RFC 5869) works in two steps:

- *Extract*: $"PRK" = "HMAC"("salt", "IKM")$ concentrates the entropy of input keying material that may be non-uniform (such as a Diffie-Hellman shared secret) into a pseudorandom key.
- *Expand*: output blocks $T_i = "HMAC"("PRK", T_(i-1) || "info" || i)$, concatenated to the required length.

The `info` parameter provides *domain separation*: deriving the encryption key with `info = "app v1 encryption"` and the MAC key with `info = "app v1 mac"` yields independent keys, so a weakness or misuse in one purpose cannot compromise another. Using the same key for two algorithms or two purposes (signing and encryption, two protocols, two directions of a connection) is a classic design error that domain separation prevents.

Password-derived keys need deliberately slow, memory-hard functions (Argon2id, scrypt) rather than HKDF, because passwords have low entropy (#xref("cryptography-and-security", "hashing-and-macs", label: "Hashing and MACs")).

== The Key Lifecycle

NIST SP 800-57 organises key management around a lifecycle:

+ *Generation* with an approved generator, inside the system that will use or protect the key, preferably an HSM.
+ *Registration and distribution* to the parties that need it, through authenticated channels or key agreement.
+ *Storage*, encrypted under another key or inside hardware, never in source code, container images or configuration repositories.
+ *Use* for one purpose only, by the fewest components possible.
+ *Rotation* at the end of a *cryptoperiod*, set by how much data the key protects, how exposed it is, and algorithm limits.
+ *Revocation* if compromise is suspected, with the ability to identify everything the key protected.
+ *Archival or destruction*: old decryption keys may be needed for old data; destruction must cover backups and copies.

Most failures happen at storage and rotation: secrets committed to Git (secret scanning on every push catches many), long-lived credentials that nobody dares rotate because nobody knows what uses them, and decryption keys deleted while data encrypted under them still exists.

== Key Hierarchies and Envelope Encryption

Large systems never encrypt bulk data directly with a master key. They use a *hierarchy*:

```text
root key (in HSM, never exported)
   └── key-encryption keys (KEKs), per tenant or service, stored wrapped by the root
         └── data-encryption keys (DEKs), per object or file, stored wrapped by a KEK
               └── data, encrypted with the DEK using an AEAD
```

*Envelope encryption* follows the hierarchy. To encrypt an object: generate a fresh DEK, encrypt the data with it locally, ask the key service to *wrap* (encrypt) the DEK under a KEK, store the wrapped DEK beside the ciphertext, and discard the plaintext DEK. To decrypt: send the wrapped DEK to the key service, which checks authorisation, unwraps it, and returns it; decrypt locally. The benefits:

- Bulk data never leaves the application, and the key service handles only small keys, so it scales.
- Every decryption passes through the key service, where access control and audit logging apply.
- *Rotation* of a KEK only re-wraps DEKs, not terabytes of data. New writes use the new KEK; old wrapped DEKs remain decryptable until re-wrapped in the background.
- *Crypto-shredding*: destroying a tenant's KEK makes all its data unreadable without finding and deleting every copy, which is how many systems implement deletion guarantees across backups.

== Hardware Security Modules and Key Services

A *hardware security module* (HSM) is a tamper-resistant device that generates and stores keys and performs operations with them, so that the keys never exist in plaintext outside it. Applications send data in and get signatures or ciphertexts out. HSMs are certified under FIPS 140-3 (levels 1 to 4, with physical tamper resistance and response from level 3), and certification authorities, payment networks and banks are required to use them for root keys.

Cloud *key management services* (AWS KMS, Google Cloud KMS, Azure Key Vault) put HSM-backed root keys behind an API with identity-based access control, audit logs, automatic rotation and multi-region replication. *Secret managers* (HashiCorp Vault, AWS Secrets Manager) store application secrets such as database passwords and API tokens, and can issue *dynamic secrets*: short-lived credentials created on demand for each client and revoked automatically, so that a leaked credential expires within minutes.

Trusted execution environments and *confidential computing* (Intel TDX, AMD SEV-SNP, Arm CCA) extend protection to data in use, with *attestation* proving to a key service what code is running before it releases keys. They reduce the trust placed in the cloud operator but have their own history of side-channel vulnerabilities.

== Rotation Without Outages

Rotation fails in predictable ways: a new key is deployed to writers before readers can use it, or an old key is removed while data or tokens encrypted under it still exist. Safe rotation is staged:

+ *Add* the new key everywhere as a valid key for *verification and decryption*.
+ *Switch* writers to sign and encrypt with the new key once every reader accepts it.
+ *Re-encrypt or let expire* everything protected by the old key.
+ *Remove* the old key only when nothing depends on it.

Carrying a *key identifier* with every ciphertext and signature (the `kid` in JSON Web Tokens, the key version in a wrapped DEK) makes this possible: readers select the right key rather than trying each one. *Crypto agility*, the ability to change algorithms as well as keys, uses the same mechanism and is now pressing for the migration to post-quantum algorithms (#xref("cryptography-and-security", "post-quantum", label: "Post-Quantum Cryptography")).

== Pitfalls

- Using non-cryptographic generators for tokens, keys or nonces.
- Generating long-term keys at first boot of embedded devices or freshly cloned VMs.
- Modulo bias when mapping random bits to a range.
- Reusing a key across purposes or protocols without domain separation.
- Secrets in source code, images or logs.
- Rotation procedures that switch writers before readers, or delete keys still needed for old data.

== Further Reading

Heninger, N., Durumeric, Z., Wustrow, E. and Halderman, J. A. (2012). "Mining Your Ps and Qs: Detection of Widespread Weak Keys in Network Devices." USENIX Security.

Yilek, S. et al. (2009). "When Private Keys Are Public: Results from the 2008 Debian OpenSSL Vulnerability." IMC.

Checkoway, S. et al. (2014). "On the Practical Exploitability of Dual EC in TLS Implementations." USENIX Security.

Checkoway, S. et al. (2016). "A Systematic Analysis of the Juniper Dual EC Incident." CCS.

Barker, E. and Kelsey, J. (2015). "Recommendation for Random Number Generation Using Deterministic Random Bit Generators." NIST SP 800-90A Rev. 1.

Barker, E. (2020). "Recommendation for Key Management: Part 1 – General." NIST SP 800-57 Part 1 Rev. 5.

Krawczyk, H. and Eronen, P. (2010). "HMAC-based Extract-and-Expand Key Derivation Function (HKDF)." RFC 5869.

Krawczyk, H. (2010). "Cryptographic Extraction and Key Derivation: The HKDF Scheme." CRYPTO.

Pornin, T. (2013). "Deterministic Usage of the Digital Signature Algorithm (DSA) and Elliptic Curve Digital Signature Algorithm (ECDSA)." RFC 6979.

Ferguson, N., Schneier, B. and Kohno, T. (2010). _Cryptography Engineering: Design Principles and Practical Applications_. Wiley.

Donenfeld, J. A. (2022). "Random Number Generator Enhancements for Linux 5.17 and 5.18." zx2c4 blog, reference.

#import "../template.typ": definition, xref

= Homomorphic Encryption <homomorphic-encryption>

Ordinary encryption protects data at rest and in transit, but to compute on it, someone must decrypt it. *Homomorphic encryption* (HE) removes that step: a server can add and multiply encrypted values and return an encrypted result that only the key holder can read, without ever seeing the inputs, the intermediate values or the output. The idea is as old as RSA, which happens to be multiplicatively homomorphic, but encryption supporting arbitrary computation was an open problem until Gentry's construction in 2009. Since then, *fully homomorphic encryption* (FHE) has gone from a theoretical existence proof running at roughly a trillion times the cost of plaintext computation to libraries used in production for password breach checking, caller-ID lookup and private search. This chapter covers partially homomorphic schemes, the learning-with-errors foundation and its noise, a C++ toy LWE scheme that shows noise growth, Gentry's bootstrapping, the BGV, BFV, CKKS and TFHE scheme families, packing and performance, security subtleties, applications, and the tooling.

*See also:* #xref("cryptography-and-security", "post-quantum", label: "Post-Quantum Cryptography") (lattice assumptions), #xref("cryptography-and-security", "secure-multiparty-computation", label: "Secure Multi-Party Computation"), #xref("cryptography-and-security", "asymmetric", label: "Asymmetric Cryptography"), #xref("cryptography-and-security", "zero-knowledge-proofs", label: "Zero-Knowledge Proofs").

== Partially Homomorphic Schemes

Several classic schemes support one operation on ciphertexts:

#table(
  columns: 3,
  [*Scheme*], [*Homomorphism*], [*Notes*],
  [Textbook RSA], [$E(m_1) E(m_2) = E(m_1 m_2)$], [Unpadded RSA is insecure; secure padding destroys the property],
  [ElGamal], [Multiplicative], [Exponential ElGamal ($g^m$) is additive but decryption needs a small discrete log],
  [Paillier], [$E(m_1) E(m_2) = E(m_1 + m_2)$, and $E(m)^k = E(k m)$], [Additive over $ZZ_N$; used for encrypted tallies and private aggregation],
)

Additive homomorphism already enables useful protocols: e-voting tallies where only the total is decrypted, private aggregation of statistics, and private information retrieval. What it cannot do is evaluate a general function, which needs both addition and multiplication, an unbounded number of times.

== Learning With Errors and Noise

All practical FHE schemes rest on the *learning with errors* (LWE) problem or its ring variant (RLWE), the same lattice assumptions behind post-quantum key encapsulation.

#definition(name: "Learning with errors")[
For a secret $s in ZZ_q^n$, an LWE sample is $(a, b)$ with $a$ uniform in $ZZ_q^n$ and $b = chevron.l a, s chevron.r + e mod q$, where $e$ is a small error drawn from a narrow distribution. The search problem is to find $s$ from many samples; the decision problem is to distinguish samples from uniform pairs. Both are believed hard for classical and quantum computers for suitable parameters.
]

An LWE sample hides a message by adding it, suitably scaled, to $b$. In the simplest symmetric scheme for a bit $m$, the ciphertext is $(a, chevron.l a, s chevron.r + e + m floor(q \/ 2))$. Decryption computes $b - chevron.l a, s chevron.r = e + m floor(q \/ 2)$ and rounds: near 0 means 0, near $q \/ 2$ means 1. The scheme is additively homomorphic: adding two ciphertexts adds the messages (modulo 2, here) and adds the errors. Multiplying by a public integer multiplies the error. Decryption is correct only while the accumulated error stays below $q \/ 4$. That *noise budget* is the central resource of homomorphic computation.

```cpp
#include <cstdint>
#include <random>
#include <vector>

// Toy symmetric LWE encryption of bits with q = 2^32 (arithmetic wraps naturally).
// Parameters are illustrative only and NOT secure.
class toy_lwe {
public:
    struct ciphertext {
        std::vector<std::uint32_t> a;
        std::uint32_t b;
    };

    explicit toy_lwe(std::size_t n, std::uint64_t seed) : n_(n), rng_(seed), s_(n) {
        for (auto& x : s_) x = static_cast<std::uint32_t>(rng_());
    }

    ciphertext encrypt(int bit) {
        ciphertext c{std::vector<std::uint32_t>(n_), 0};
        std::uniform_int_distribution<int> error(-16, 16);
        for (auto& x : c.a) x = static_cast<std::uint32_t>(rng_());
        c.b = dot(c.a) + static_cast<std::uint32_t>(error(rng_)) +
              (bit ? half_q : 0u);
        return c;
    }

    int decrypt(const ciphertext& c) const {
        const std::uint32_t phase = c.b - dot(c.a);       // = e + m * q/2  (mod q)
        return ((phase + quarter_q) / half_q) & 1;         // round to nearest multiple of q/2
    }

    // Signed error of a ciphertext that should encrypt `bit`, to watch the budget.
    std::int64_t noise(const ciphertext& c, int bit) const {
        const std::uint32_t v = c.b - dot(c.a) - (bit ? half_q : 0u);
        return static_cast<std::int32_t>(v);
    }

    static ciphertext add(const ciphertext& x, const ciphertext& y) {   // XOR of bits
        ciphertext r{x.a, x.b + y.b};
        for (std::size_t i = 0; i < r.a.size(); ++i) r.a[i] += y.a[i];
        return r;
    }

    static ciphertext scale(const ciphertext& x, std::uint32_t k) {    // k * bit mod 2
        ciphertext r{x.a, x.b * k};
        for (auto& v : r.a) v *= k;
        return r;
    }

private:
    static constexpr std::uint32_t half_q = 1u << 31;
    static constexpr std::uint32_t quarter_q = 1u << 30;

    std::uint32_t dot(const std::vector<std::uint32_t>& a) const {
        std::uint32_t acc = 0;
        for (std::size_t i = 0; i < n_; ++i) acc += a[i] * s_[i];
        return acc;
    }

    std::size_t n_;
    std::mt19937_64 rng_;
    std::vector<std::uint32_t> s_;
};
```

In an experiment with $n = 512$, summing 1,000 fresh encryptions of random bits still decrypts to their XOR, with accumulated noise in the thousands, far below the budget of $2^30$. Scaling a ciphertext whose error is 4 by $k$ multiplies the error to $4k$: correct at $k = 2^20$, close to the limit near $k = 2^28$, wrong beyond. Additions are cheap in noise; multiplications, in schemes that support them, roughly multiply the noise of their inputs, so the budget limits the *multiplicative depth* of the computation, not its size.

== Bootstrapping

A scheme that can evaluate circuits only up to a fixed depth is *somewhat* (or *levelled*) homomorphic. Gentry's insight turned this into fully homomorphic encryption. Suppose the scheme can homomorphically evaluate its own decryption circuit, with some depth to spare. Encrypt the secret key under itself (a *bootstrapping key*). Given a noisy ciphertext, homomorphically run decryption on it, using the encrypted key: the result is a fresh encryption of the same message, whose noise depends only on the depth of the decryption circuit, not on the history of the input. Bootstrapping resets the noise budget, so computation can continue indefinitely.

This requires *circular security* (encrypting the key under itself must remain safe), an assumption that is widely believed for these schemes but not implied by LWE alone. Bootstrapping is also by far the most expensive operation, and the main engineering goal of FHE research since 2009 has been to make it cheaper or to avoid it.

== The Scheme Families

#table(
  columns: 4,
  [*Scheme*], [*Plaintext*], [*Strengths*], [*Typical use*],
  [BGV, BFV], [Vectors of integers modulo a plaintext modulus $t$], [Exact arithmetic; SIMD batching of thousands of slots], [Exact integer computation, private set operations, PIR],
  [CKKS], [Vectors of approximate real or complex numbers], [Fixed-point arithmetic with rescaling; efficient for linear algebra], [Machine-learning inference, statistics],
  [TFHE, FHEW], [Bits or small integers], [Bootstrapping after every gate in milliseconds; *programmable* bootstrapping evaluates a lookup table while refreshing noise], [Arbitrary Boolean or integer programs, comparisons, non-linear functions],
)

The second-generation schemes (BGV, BFV) and CKKS work over polynomial rings $ZZ_q [X] \/ (X^N + 1)$ with $N$ from $2^12$ to $2^17$. One ciphertext holds a polynomial, and the Chinese remainder theorem lets one polynomial encode a vector of $N$ (or $N \/ 2$) independent values, so each homomorphic operation acts on all slots at once, like a SIMD instruction. Rotations of the slot vector are available through *Galois automorphisms*, at the cost of key-switching. Multiplication doubles the ciphertext size, and *relinearisation* (a key-switching operation using an evaluation key) brings it back. *Modulus switching* scales the ciphertext down to a smaller modulus after each multiplication, reducing the noise proportionally, which is what makes levelled computation efficient.

CKKS treats the noise as part of the approximation error of fixed-point arithmetic. It encodes values multiplied by a scale factor $Delta$, and after each multiplication *rescales* by dividing by $Delta$, consuming one level of the modulus chain. It is the natural fit for neural network inference and statistics, where results are approximate anyway, but non-polynomial functions such as ReLU, comparison and division must be approximated by polynomials, which costs depth.

TFHE takes the opposite approach: small ciphertexts encrypting one bit or a few bits, and a fast bootstrapping that runs after every operation, so there is no depth limit and no need to plan the noise budget. *Programmable bootstrapping* evaluates any function of the encrypted small integer during the refresh, so comparisons, activations and table lookups cost one bootstrap each. Each costs on the order of ten milliseconds on a CPU core, so large computations need massive parallelism or hardware acceleration.

Schemes can be combined: scheme switching moves data between CKKS (for arithmetic) and TFHE-like representations (for comparisons and non-linear steps).

== Performance

FHE is still expensive. Rough orders of magnitude on a modern CPU:

- A ciphertext of a large ring holds hundreds of kilobytes to a few megabytes, many times larger than the data it encrypts, though packing amortises this across thousands of slots.
- Homomorphic multiplication with relinearisation takes milliseconds; a bootstrapping of a packed CKKS or BFV ciphertext takes a fraction of a second to seconds but refreshes thousands of slots at once.
- End-to-end, computations run thousands to millions of times slower than on plaintext, depending heavily on depth and the amount of non-linear computation.

The dominant operations are number-theoretic transforms (the FFT over finite fields used to multiply polynomials) and key switching, both memory-bandwidth hungry. GPU implementations give large speedups, and dedicated accelerators (developed in DARPA's DPRIVE programme and by several companies) aim at further orders of magnitude.

Programming FHE well requires choosing algorithms with low multiplicative depth and data layouts that suit slot rotations, which compilers (for example HEIR and Concrete) increasingly automate.

== Security Subtleties

- *Parameters*: security depends on the ring dimension, the modulus size and the error distribution. The HomomorphicEncryption.org security standard tabulates the largest modulus for each ring dimension at 128-bit security; larger moduli for deeper computation require larger rings.
- *IND-CPA only*: FHE ciphertexts are malleable by design, so FHE cannot be secure against chosen-ciphertext attacks. Protocols must authenticate inputs and results separately.
- *Leaking decryptions of approximate schemes*: Li and Micciancio showed that in CKKS, the decrypted result includes the noise, and an attacker who sees decryptions of chosen computations can recover the secret key (an attack on IND-CPA-D security). Implementations now add noise ("noise flooding") before releasing decryptions, and protocols should not reveal decrypted CKKS outputs to untrusted parties without it.
- *Circuit privacy*: a ciphertext's noise may reveal something about the computation that produced it, which matters when the server's function is itself secret; noise flooding or re-randomisation hides it.
- *Integrity*: FHE hides data from the server but does not prove the server computed the right function. Verifiable computation or zero-knowledge proofs are needed for that.

== Applications

Deployed uses favour narrow, well-defined computations with a client who holds the key and a server who holds data or a model:

- *Private information retrieval and lookup*: a client retrieves an entry from a server's database without the server learning which. Apple's Live Caller ID Lookup uses BFV-based PIR so that the service does not learn which numbers are calling, and Microsoft Edge's password monitor uses homomorphic encryption to check passwords against breach data.
- *Private inference*: a client sends an encrypted input (an image, a medical record) to a model server and gets an encrypted prediction; latency is seconds to minutes for small networks.
- *Private set intersection and aggregation*, often in combination with MPC.
- *Encrypted analytics in regulated settings*, such as cross-institution genomic or financial statistics where data cannot legally be pooled.

== Tooling

#table(
  columns: 2,
  [*Library*], [*Schemes and focus*],
  [Microsoft SEAL], [BFV, BGV, CKKS; C++ with wrappers],
  [OpenFHE], [BGV, BFV, CKKS, FHEW/TFHE, scheme switching, threshold FHE],
  [TFHE-rs, Concrete (Zama)], [TFHE with programmable bootstrapping; compiler from Python],
  [Lattigo], [Go implementation of BFV, BGV, CKKS, multiparty variants],
  [HElib], [BGV and CKKS, early reference implementation],
)

Choosing parameters by hand is error-prone; libraries offer parameter generators tied to the security standard, and production systems should use them.

== Pitfalls

- Choosing ring dimension and modulus without checking the security tables.
- Exposing decrypted CKKS results to an adversary without noise flooding.
- Assuming FHE provides integrity or protection against malicious servers.
- Designing deep computations without accounting for the noise budget, so results decrypt to garbage.
- Using FHE where MPC, trusted hardware or simply not collecting the data would be cheaper and sufficient.

== Further Reading

Gentry, C. (2009). "Fully Homomorphic Encryption Using Ideal Lattices." STOC.

Regev, O. (2005). "On Lattices, Learning with Errors, Random Linear Codes, and Cryptography." STOC.

Paillier, P. (1999). "Public-Key Cryptosystems Based on Composite Degree Residuosity Classes." EUROCRYPT.

Brakerski, Z., Gentry, C. and Vaikuntanathan, V. (2012). "(Leveled) Fully Homomorphic Encryption without Bootstrapping." ITCS.

Fan, J. and Vercauteren, F. (2012). "Somewhat Practical Fully Homomorphic Encryption." IACR ePrint 2012/144.

Cheon, J. H., Kim, A., Kim, M. and Song, Y. (2017). "Homomorphic Encryption for Arithmetic of Approximate Numbers." ASIACRYPT.

Chillotti, I., Gama, N., Georgieva, M. and Izabachène, M. (2020). "TFHE: Fast Fully Homomorphic Encryption over the Torus." Journal of Cryptology.

Li, B. and Micciancio, D. (2021). "On the Security of Homomorphic Encryption on Approximate Numbers." EUROCRYPT.

Albrecht, M. et al. (2018). "Homomorphic Encryption Security Standard." HomomorphicEncryption.org standard.

Marcolla, C. et al. (2022). "Survey on Fully Homomorphic Encryption, Theory, and Applications." Proceedings of the IEEE.

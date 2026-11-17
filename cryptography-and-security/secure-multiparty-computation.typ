#import "../template.typ": definition, theorem, xref

= Secure Multi-Party Computation <secure-multiparty-computation>

Several hospitals want to know whether a treatment works across all their patients without revealing any patient's record to the others. Two companies want to know which customers they have in common without exchanging customer lists. A wallet should sign transactions only if two of three devices agree, without the full private key ever existing in one place. *Secure multi-party computation* (MPC) makes all of these possible: parties holding private inputs jointly compute a function of them and learn the output, and nothing else, without trusting each other or a third party. Once a theoretical curiosity, MPC now runs in production for threshold signing, private set intersection, advertising measurement and privacy-preserving statistics. This chapter covers the security definitions, secret sharing with C++ Shamir sharing, computing on shares and Beaver multiplication triples, oblivious transfer, Yao's garbled circuits and their optimisations, protocols for malicious adversaries, threshold cryptography, private set intersection, and the costs and deployments.

*See also:* #xref("cryptography-and-security", "zero-knowledge-proofs", label: "Zero-Knowledge Proofs"), #xref("cryptography-and-security", "homomorphic-encryption", label: "Homomorphic Encryption"), #xref("cryptography-and-security", "digital-signatures", label: "Digital Signatures"), #xref("cryptography-and-security", "randomness-and-key-management", label: "Randomness and Key Management").

== What Security Means

The definition compares the real protocol with an *ideal world* in which a trusted party receives every input, computes the function, and returns each party its output. A protocol is secure if whatever an adversary can achieve by attacking the real protocol, it could also achieve in the ideal world, where it can only choose the inputs of the parties it corrupts.

#definition(name: "Simulation-based security")[
A protocol $pi$ securely computes $f$ if for every efficient adversary $cal(A)$ attacking $pi$ there is an efficient *simulator* $cal(S)$ in the ideal world such that the joint distribution of $cal(S)$'s view and the honest parties' outputs is indistinguishable from that of $cal(A)$ in the real protocol.
]

Intuitively, anything the corrupted parties see in the real protocol could have been generated from their own inputs and outputs alone, so the protocol leaks nothing more. The output itself may reveal information (the average of two salaries reveals the other salary to each party), which is a property of the function, not the protocol.

Adversaries are classified by behaviour and number:

#table(
  columns: 2,
  [*Model*], [*Meaning*],
  [Semi-honest (passive, honest-but-curious)], [Follows the protocol but tries to learn from what it sees],
  [Malicious (active)], [Deviates arbitrarily: sends wrong values, aborts],
  [Honest majority], [Fewer than half the parties are corrupted; information-theoretic security possible],
  [Dishonest majority], [Any number corrupted, including all but one; requires computational assumptions, and fairness (everyone gets the output or nobody does) is impossible in general],
)

Two classic feasibility results anchor the field: any function can be computed securely against a semi-honest or malicious dishonest majority under standard cryptographic assumptions (Yao for two parties; Goldreich, Micali and Wigderson for many), and with an honest majority of two thirds, any function can be computed with information-theoretic security, no assumptions at all (Ben-Or, Goldwasser and Wigderson; Chaum, Crépeau and Damgård).

== Secret Sharing

*Additive sharing* splits a secret $s$ in a field $FF_p$ among $n$ parties: choose random $s_1, dots, s_(n-1)$ and set $s_n = s - sum s_i$. All $n$ shares reconstruct $s$; any $n - 1$ are uniformly random and reveal nothing.

*Shamir's threshold scheme* allows any $t$ of $n$ parties to reconstruct. The dealer picks a random polynomial $f$ of degree $t - 1$ with $f(0) = s$ and gives party $i$ the share $f(i)$. Any $t$ shares determine $f$ by interpolation, and hence $s = f(0)$; any $t - 1$ shares are consistent with every possible secret, so they reveal nothing, unconditionally.

```cpp
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

constexpr std::uint64_t P = (std::uint64_t{1} << 61) - 1;   // Mersenne prime field

std::uint64_t add_mod(std::uint64_t a, std::uint64_t b) { const auto s = a + b; return s >= P ? s - P : s; }
std::uint64_t sub_mod(std::uint64_t a, std::uint64_t b) { return a >= b ? a - b : a + P - b; }
std::uint64_t mul_mod(std::uint64_t a, std::uint64_t b) {
    return static_cast<std::uint64_t>(static_cast<unsigned __int128>(a) * b % P);
}
std::uint64_t pow_mod(std::uint64_t b, std::uint64_t e) {
    std::uint64_t r = 1;
    for (; e; e >>= 1, b = mul_mod(b, b))
        if (e & 1) r = mul_mod(r, b);
    return r;
}
std::uint64_t inv_mod(std::uint64_t a) { return pow_mod(a, P - 2); }   // Fermat

using share = std::pair<std::uint64_t, std::uint64_t>;   // (x, f(x))

// Split secret into n shares, any t of which reconstruct it.
// The coefficients must come from a cryptographic generator in real use.
std::vector<share> shamir_split(std::uint64_t secret, int t, int n, std::mt19937_64& rng) {
    std::vector<std::uint64_t> coeff{secret};
    std::uniform_int_distribution<std::uint64_t> uniform(0, P - 1);
    for (int k = 1; k < t; ++k) coeff.push_back(uniform(rng));
    std::vector<share> shares;
    for (std::uint64_t x = 1; x <= static_cast<std::uint64_t>(n); ++x) {
        std::uint64_t y = 0;
        for (auto it = coeff.rbegin(); it != coeff.rend(); ++it) y = add_mod(mul_mod(y, x), *it);
        shares.emplace_back(x, y);
    }
    return shares;
}

// Lagrange interpolation at 0 from any t shares.
std::uint64_t shamir_reconstruct(const std::vector<share>& shares) {
    std::uint64_t secret = 0;
    for (std::size_t i = 0; i < shares.size(); ++i) {
        std::uint64_t num = 1, den = 1;
        for (std::size_t j = 0; j < shares.size(); ++j) {
            if (i == j) continue;
            num = mul_mod(num, sub_mod(0, shares[j].first));
            den = mul_mod(den, sub_mod(shares[i].first, shares[j].first));
        }
        secret = add_mod(secret, mul_mod(shares[i].second, mul_mod(num, inv_mod(den))));
    }
    return secret;
}
```

Splitting a secret into 5 shares with threshold 3 and reconstructing from shares 1, 2, 3 or from 1, 3, 5 both return the original. (The example uses `std::mt19937_64` for brevity; real shares must use a cryptographic generator, or the "random" coefficients are predictable.) Shamir sharing is used directly for backing up master keys (split among several officers, any quorum restores it) and is the basis of honest-majority MPC protocols.

== Computing on Shares

Linear operations on shared values need no communication. If parties hold shares $[x]$ and $[y]$, then each adding its shares of $x$ and $y$ gives shares of $x + y$, and multiplying its share by a public constant gives shares of $c x$. Multiplication of two shared values is the hard part, because the product of shares is not a share of the product in a usable form.

*Beaver triples* make multiplication cheap online. In an input-independent *preprocessing* phase, the parties obtain shares of random $a$, $b$ and $c = a b$. To multiply $[x]$ and $[y]$ online:

+ Each party computes and broadcasts its share of $d = x - a$ and $e = y - b$; the parties reconstruct $d$ and $e$. These reveal nothing, because $a$ and $b$ are uniformly random masks.
+ Each party computes its share of $z = c + d [b] + e [a] + d e$, with the public constant $d e$ added by one designated party.

Since $c + d b + e a + d e = a b + (x - a) b + (y - b) a + (x - a)(y - b) = x y$, the result is a sharing of the product. Online multiplication costs one round and two field elements of communication per party; the expensive work of generating triples (with oblivious transfer or homomorphic encryption) happens beforehand. A function is then evaluated as an arithmetic circuit, one round per layer of multiplications, which makes the *multiplicative depth* of the circuit the latency driver over wide-area networks.

== Oblivious Transfer

*1-out-of-2 oblivious transfer* (OT) is a two-party primitive: a sender has two messages $m_0, m_1$; a receiver has a choice bit $b$ and learns $m_b$, while the sender learns nothing about $b$ and the receiver nothing about $m_(1-b)$. OT is *complete* for MPC: any function can be computed securely from OT alone (Kilian, 1988).

OT requires public-key operations, which are slow, but *OT extension* (Ishai, Kilian, Nissim and Petrank, 2003) turns a small number of base OTs (about 128) into millions of OTs using only symmetric operations such as AES and hashing. Later *silent OT* constructions based on learning-parity-with-noise assumptions generate huge batches of correlated OTs with very little communication. These advances are why OT-based MPC is practical.

== Garbled Circuits

Yao's *garbled circuits* evaluate any Boolean circuit between two parties in a constant number of rounds.

+ The *garbler* chooses two random keys (labels) for each wire, one meaning 0 and one meaning 1. For each gate, it encrypts the output label for each combination of input labels, so that knowing one label per input wire lets you decrypt exactly one output label, the correct one, and learn nothing else. It sends the encrypted gate tables and its own input labels.
+ The *evaluator* obtains the labels for its own input bits through oblivious transfer, so the garbler does not learn its inputs, then evaluates the circuit gate by gate, holding one meaningless label per wire.
+ The output wires' labels are mapped back to bits.

Engineering has made garbling fast:

- *Point-and-permute* attaches a permutation bit to each label so the evaluator knows which table row to decrypt, instead of trying all four.
- *Free-XOR* (Kolesnikov and Schneider) chooses all label pairs with a global offset $Delta$ ($L^1 = L^0 xor Delta$), so XOR gates need no table and no encryption at all: the output label is the XOR of the input labels.
- *Half-gates* (Zahur, Rosulek and Evans) reduce each AND gate to two ciphertexts, which was shown to be optimal for a broad class of schemes.
- Fixed-key AES as the encryption, with hardware AES instructions, allows tens of millions of gates per second per core.

Garbled circuits have a constant number of rounds, which suits high-latency networks, but communication proportional to the circuit size. Secret-sharing protocols use less bandwidth but more rounds. Mixed protocols switch between Boolean and arithmetic representations depending on the operation.

== Against Malicious Adversaries

Semi-honest security is not enough when parties may cheat. Upgrades include:

- *Cut-and-choose* for garbled circuits: the garbler prepares many circuits, the evaluator opens and checks a random subset and evaluates the rest; more recent *authenticated garbling* avoids most of the overhead.
- *Information-theoretic MACs on shares*: in SPDZ (Damgård, Pastro, Smart and Zakarias), every shared value carries a share of a MAC under a secret global key, so any party that changes a share is detected when outputs are opened, with probability $1 - 1 \/ p$. Preprocessing generates authenticated triples using homomorphic encryption or OT. Malicious security with a dishonest majority then costs only a small constant factor over semi-honest in the online phase.

== Threshold Cryptography

An important special case of MPC computes cryptographic operations with a distributed key:

- *Distributed key generation* (DKG) lets $n$ parties create a key pair such that the private key is Shamir-shared among them and never exists in one place.
- *Threshold signatures* let any $t$ of the $n$ parties produce a signature verifiable with the single public key. For Schnorr and EdDSA-style signatures, FROST gives an efficient two-round protocol. Threshold ECDSA is harder, because the nonce inversion in ECDSA is not linear, and needs MPC multiplication; several practical protocols exist.
- *Threshold decryption* similarly requires a quorum to decrypt.

These are widely deployed: cryptocurrency custody services and wallets split keys across devices and servers so that no single breach steals funds, and certificate authorities and distributed randomness beacons (such as drand) use threshold schemes to avoid single points of compromise.

== Private Set Intersection

*Private set intersection* (PSI) lets two parties learn the intersection of their sets (or only its size, or an aggregate over it) without revealing other elements. A classic construction uses commutative exponentiation in a group where Diffie-Hellman is hard: each party hashes its elements to group elements and raises them to its secret exponent; after exchanging and re-exponentiating each other's values, both sides hold $H(x)^(a b)$ for their elements and can compare. Faster protocols use OT extension and hashing into bins, handling sets of millions of elements in seconds.

PSI is used in contact discovery for messaging apps, in password breach checking (Google and Microsoft check users' passwords against breach corpora without seeing the passwords), and in advertising measurement, where an advertiser and a platform compute aggregate conversion statistics over their joined data (Google's Private Join and Compute).

== Costs and Deployments

MPC's costs are dominated by communication and rounds rather than computation:

- Garbled circuits: about 32 bytes per AND gate with half-gates and 128-bit labels, so a circuit with $10^9$ AND gates transfers about 32 GB.
- Secret-sharing protocols: a few field elements per multiplication per party, but one round per multiplicative layer.
- Preprocessing (triples, OTs) often dominates total cost but can run before inputs are known.

Representative deployments: the Danish sugar beet auction (2008), which computed market-clearing prices from farmers' private bids; the Boston Women's Workforce Council's pay-gap study, which aggregated salary data from employers who would not share it directly; threshold-signing custody systems; and federated statistics and machine learning in which aggregation of client updates is computed so that the server sees only the sum.

MPC, homomorphic encryption, zero-knowledge proofs and trusted execution environments overlap. MPC distributes trust among parties; homomorphic encryption (#xref("cryptography-and-security", "homomorphic-encryption", label: "Homomorphic Encryption")) lets one untrusted party compute on encrypted data; zero-knowledge proofs prove facts about private data; trusted hardware relies on the manufacturer instead of cryptography. Real systems often combine them.

== Pitfalls

- Using semi-honest protocols where participants have incentives to cheat.
- Forgetting that the output itself may reveal private inputs; combine MPC with output restrictions or differential privacy.
- Weak randomness in shares, masks or labels.
- Ignoring the network: protocols with many rounds perform poorly across continents.
- Assuming threshold schemes remove all risk; a quorum of compromised parties still has the key, and key refresh is needed after any suspected compromise.

== Further Reading

Yao, A. C. (1986). "How to Generate and Exchange Secrets." FOCS.

Goldreich, O., Micali, S. and Wigderson, A. (1987). "How to Play Any Mental Game." STOC.

Ben-Or, M., Goldwasser, S. and Wigderson, A. (1988). "Completeness Theorems for Non-Cryptographic Fault-Tolerant Distributed Computation." STOC.

Shamir, A. (1979). "How to Share a Secret." Communications of the ACM.

Beaver, D. (1991). "Efficient Multiparty Protocols Using Circuit Randomization." CRYPTO.

Ishai, Y., Kilian, J., Nissim, K. and Petrank, E. (2003). "Extending Oblivious Transfers Efficiently." CRYPTO.

Kolesnikov, V. and Schneider, T. (2008). "Improved Garbled Circuit: Free XOR Gates and Applications." ICALP.

Zahur, S., Rosulek, M. and Evans, D. (2015). "Two Halves Make a Whole: Reducing Data Transfer in Garbled Circuits Using Half Gates." EUROCRYPT.

Damgård, I., Pastro, V., Smart, N. and Zakarias, S. (2012). "Multiparty Computation from Somewhat Homomorphic Encryption." CRYPTO.

Komlo, C. and Goldberg, I. (2020). "FROST: Flexible Round-Optimized Schnorr Threshold Signatures." SAC.

Bogetoft, P. et al. (2009). "Secure Multiparty Computation Goes Live." Financial Cryptography.

Evans, D., Kolesnikov, V. and Rosulek, M. (2018). _A Pragmatic Introduction to Secure Multi-Party Computation_. Now Publishers.

Lindell, Y. (2020). "Secure Multiparty Computation." Communications of the ACM.

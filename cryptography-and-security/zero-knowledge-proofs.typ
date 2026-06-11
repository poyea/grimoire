= Zero-Knowledge Proofs

A zero-knowledge proof lets a prover convince a verifier that a statement is true while revealing nothing beyond its truth. Once a theoretical curiosity, ZK proofs now run in production at scale: blockchain rollups prove the correct execution of millions of transactions, and privacy systems prove properties of hidden data. This chapter builds from interactive proofs and sigma protocols to the modern SNARK and STARK families.

*See also:* _Asymmetric Cryptography_ (discrete log, pairings), _Digital Signatures_ (Fiat-Shamir, Schnorr), _Hashing and MACs_ (Merkle trees, random oracles).

== Interactive Proofs and the Three Properties

An interactive proof system for a language $L$ involves a prover $P$ and verifier $V$ exchanging messages. Goldwasser, Micali & Rackoff (1985; journal version 1989) defined zero knowledge via three properties:

- *Completeness*: if $x in L$, an honest prover convinces the verifier.
- *Soundness*: if $x in.not L$, no cheating prover succeeds except with negligible probability. (*Proof of knowledge* strengthens this: a successful prover must _know_ a witness, formalised via an extractor.)
- *Zero knowledge*: the verifier learns nothing — formalised by a *simulator* that, given only the statement, produces transcripts indistinguishable from real interactions. Whatever the verifier could compute after the protocol, it could have computed alone.

The classic example: proving graph isomorphism by sending a random re-labelling and answering a coin flip with the isomorphism to one side or the other. Each round halves a cheater's success probability.

== Sigma Protocols

A *sigma protocol* is a three-move structure — commitment $a$, random challenge $e$, response $z$ — with completeness, *special soundness* (two accepting transcripts with the same $a$ and different challenges yield the witness), and honest-verifier zero knowledge.

*Schnorr's protocol* proves knowledge of $x$ with $y = g^x$:

1. $P$ sends $a = g^k$ for random $k$.
2. $V$ sends challenge $e$.
3. $P$ sends $z = k + e x mod q$; $V$ checks $g^z = a y^e$.

Sigma protocols compose: AND/OR combinations (proving one of two statements without revealing which — the basis of ring signatures), and proofs of linear relations among committed values.

=== The Fiat-Shamir Transform

Replacing the verifier's challenge with a hash of the transcript, $e = H(a || "statement")$, makes the proof *non-interactive* in the random-oracle model. Schnorr signatures are exactly Fiat-Shamir applied to Schnorr identification with the message in the hash. A recurring class of real vulnerabilities ("weak Fiat-Shamir") omits the statement from the hash — exploited against several deployed SNARK and voting implementations (Frozen Heart, 2022).

== Commitments

A *commitment scheme* lets one bind to a value now and reveal it later — *hiding* (commitment reveals nothing) and *binding* (cannot open to a different value). The *Pedersen commitment* $C = g^m h^r$ is perfectly hiding, computationally binding, and *additively homomorphic*: $C_1 C_2$ commits to $m_1 + m_2$. Homomorphic commitments are the glue of most ZK constructions; *polynomial commitments* (KZG, 2010) extend this to committing to a polynomial and proving evaluations $p(z) = v$ with constant-size openings — the core of most modern SNARKs and of Ethereum's EIP-4844 blob commitments.

== zk-SNARKs

A *SNARK* — succinct non-interactive argument of knowledge — proves arbitrary computation with a proof that is tiny (hundreds of bytes) and verifiable in milliseconds, essentially independent of computation size. The pipeline:

1. *Arithmetisation*: express the computation as constraints over a finite field — R1CS (rank-1 constraint systems, triples enforcing $chevron.l a, w chevron.r dot chevron.l b, w chevron.r = chevron.l c, w chevron.r$) or the more flexible *Plonkish* / AIR formats with custom gates and lookup tables.
2. *Polynomial IOP*: encode constraint satisfaction as polynomial identities checked at random points.
3. *Cryptographic compiler*: a polynomial commitment scheme plus Fiat-Shamir collapses the interaction into one proof.

=== Groth16

Groth (2016) remains the smallest: 2 $G_1$ + 1 $G_2$ elements ($approx 192$ bytes on BLS12-381), one pairing equation to verify. Cost: a *trusted setup per circuit* — the "toxic waste" from the setup ceremony would allow forging proofs, so multi-party ceremonies (Zcash's Sapling MPC, built on a circuit-independent powers-of-tau phase) ensure security if any one participant is honest.

=== PLONK and Universal Setups

PLONK (Gabizon, Williamson & Ciobotaru, 2019) uses one *universal* trusted setup for all circuits up to a size bound, with KZG commitments. Descendants dominate practice: lookup arguments (plookup, LogUp) make non-arithmetic operations (range checks, XOR) cheap, and *folding schemes* (Nova, 2021) make incremental verifiable computation efficient by folding many steps into one instance before a final SNARK.

=== STARKs

STARKs (Ben-Sasson et al., 2018) replace elliptic-curve commitments with hash-based ones (Merkle trees + the *FRI* low-degree test):
- *Transparent*: no trusted setup at all.
- *Post-quantum*: security reduces to hash functions.
- Cost: proofs of tens to hundreds of KB (vs. bytes), though recursion compresses them.

StarkWare's production systems and most zkVMs (RISC Zero, SP1) are STARK-based, often wrapped in a final Groth16 proof for cheap on-chain verification.

=== Bulletproofs

Bulletproofs (Bünz et al., 2018) need no trusted setup and give logarithmic-size *range proofs* (proving a committed value lies in $[0, 2^n)$) — used in Monero's confidential transactions (Bulletproofs from 2018, Bulletproofs+ since 2022). Verification is linear in the statement, so they suit small statements rather than general computation.

== Applications

- *Rollups*: zkSync, Starknet, Scroll, Polygon zkEVM prove batches of EVM execution; Ethereum verifies one proof instead of re-executing.
- *Private transactions*: Zcash (Groth16 over the Sapling circuit) proves a transaction is valid — inputs exist in the note commitment tree, values balance — without revealing amounts or addresses.
- *Identity and credentials*: prove "over 18" or "EU passport holder" from a signed credential without revealing the document (BBS+ signatures, anonymous credentials, zk-mDL work).
- *Proof of correct ML inference and verifiable computation*: zkVMs prove arbitrary RISC-V execution; cloud results can be checked without re-running.

== Practical Considerations

- *Prover cost is the bottleneck*: proving is $10^3$–$10^6$ times slower than native execution; hardware acceleration (GPU MSM/NTT) is an active industry.
- *Circuit bugs are soundness bugs*: an under-constrained circuit lets a malicious prover "prove" false statements — the most common and most severe vulnerability class in deployed ZK systems (numerous audited incidents); formal circuit verification is nascent.
- *Fiat-Shamir hygiene*: hash the entire statement and all commitments.
- *Recursion*: proofs that verify other proofs enable aggregation and unbounded computation with constant verification.

== Further Reading

- Goldwasser, S., Micali, S., & Rackoff, C. (1989). The knowledge complexity of interactive proof systems. _SIAM Journal on Computing_, 18(1).
- Groth, J. (2016). On the size of pairing-based non-interactive arguments. _EUROCRYPT_.
- Gabizon, A., Williamson, Z., & Ciobotaru, O. (2019). PLONK. ePrint 2019/953.
- Ben-Sasson, E. et al. (2018). Scalable, transparent, and post-quantum secure computational integrity (STARKs). ePrint 2018/046.
- Bünz, B. et al. (2018). Bulletproofs: short proofs for confidential transactions and more. _IEEE S&P_.
- Thaler, J. (2022). _Proofs, Arguments, and Zero-Knowledge_. (freely available online)

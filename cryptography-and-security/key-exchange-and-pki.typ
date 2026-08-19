#import "../template.typ": rfc, xref

= Key Exchange and PKI

Key exchange lets two parties who share no secret establish one over a public channel; public-key infrastructure (PKI) solves the remaining problem of knowing _whose_ key you are exchanging with. Together they form the trust backbone of TLS, SSH, VPNs, and messaging. This chapter covers Diffie-Hellman and its modern instantiations, authenticated key exchange, the Signal protocol stack, and the X.509 certificate ecosystem.

*See also:* #xref("cryptography-and-security", "asymmetric", label: "Asymmetric Cryptography") (discrete log, elliptic curves), #xref("cryptography-and-security", "tls", label: "TLS: Transport Layer Security") (the deployed handshake), #xref("cryptography-and-security", "digital-signatures", label: "Digital Signatures") (certificate signing), #xref("cryptography-and-security", "post-quantum", label: "Post-Quantum Cryptography") (ML-KEM, hybrid key exchange).

== Diffie-Hellman Key Exchange

The 1976 Diffie-Hellman protocol (modern practice uses a prime-order subgroup; the original paper worked in $ZZ_p^*$): over a group of prime order $q$ with generator $g$, Alice picks $a$ and sends $g^a$; Bob picks $b$ and sends $g^b$; both compute the shared secret $g^(a b)$.

Security rests on the *computational Diffie-Hellman* (CDH) assumption — given $(g, g^a, g^b)$, computing $g^(a b)$ is hard — and in practice the stronger *decisional* variant (DDH). The raw shared secret is never used directly: a *key derivation function* (HKDF, #rfc(5869)) extracts uniform keys and binds them to the protocol transcript.

=== Finite-Field vs. Elliptic-Curve DH

- *FFDHE*: classic $ZZ_p^*$ groups. Require $|p| >= 2048$ bits. The *Logjam* attack (2015) exploited export-grade 512-bit groups and showed that precomputation against a single popular 1024-bit prime could break a large fraction of deployed servers.
- *ECDH*: elliptic-curve groups give equivalent security at 256 bits with far cheaper operations. *X25519* (Bernstein's Curve25519 in Montgomery form) is the de facto standard: fast, constant-time by construction, and resistant to invalid-curve attacks by design (x-only Montgomery ladder over a twist-secure curve; every 32-byte string is accepted, though low-order inputs map to a zero output and contributory protocols must reject them). X448 is its higher-security sibling.

*Ephemeral vs. static*: ephemeral DH (fresh keys per session, "DHE"/"ECDHE") provides *forward secrecy* — compromise of long-term keys does not reveal past session keys. TLS 1.3 removed all non-forward-secret key exchange.

== Authenticated Key Exchange

Unauthenticated DH falls to an active man-in-the-middle who runs separate exchanges with each side. Authentication options:

- *Signed DH* (TLS 1.3, SSH): each party signs the transcript with a certified long-term key — the SIGMA family of designs (Krawczyk, 2003), which TLS 1.3 follows.
- *Static-DH / implicit authentication* (Noise KK/IK patterns, WireGuard): mix long-term DH shares into the key schedule; authenticity follows from the ability to compute the shared secret.
- *PAKE* (password-authenticated KE): derive authentication from a low-entropy password without exposing it to offline guessing. SRP is the legacy scheme; *OPAQUE* (2018) is the modern asymmetric PAKE; *CPace* the symmetric one.

A well-designed AKE binds identities, transcript, and keys together to prevent *unknown key-share* attacks and provide *channel binding* — formalised in the eCK and related models.

=== The Noise Framework and WireGuard

The *Noise protocol framework* (Perrin) is a small algebra of handshake patterns over DH, HKDF, and AEAD. Each pattern (NN, NK, IK, XX, ...) specifies which static and ephemeral keys are exchanged and when. WireGuard uses Noise IK with X25519, ChaCha20-Poly1305, and BLAKE2s: a 1-RTT handshake, mandatory key rotation every 2 minutes, and a tiny verifiable state machine — a deliberate contrast with the complexity of IKE/IPsec.

== The Signal Protocol

The de facto standard for end-to-end encrypted messaging (Signal, WhatsApp, Google Messages):

- *X3DH* (extended triple DH): asynchronous initial key agreement. The initiator combines DH operations among identity keys, a signed prekey, and (when available) a one-time prekey fetched from the server, so sessions start while the recipient is offline.
- *Double Ratchet*: each message advances a symmetric KDF chain (forward secrecy per message), and each round trip mixes in a fresh DH exchange (*post-compromise security*: an attacker who steals state is healed out after one honest round trip).
- *Sealed sender, sender keys*: metadata reduction and group fan-out.

Successor work: *MLS* (Messaging Layer Security, #rfc(9420)) scales post-compromise-secure group keying to thousands of members via the TreeKEM ratcheting tree; *PQXDH* adds a Kyber (now ML-KEM) ciphertext to X3DH against harvest-now-decrypt-later adversaries.

== Key Encapsulation Mechanisms

Post-quantum schemes are KEMs rather than DH: $"Encap"("pk") -> (c, K)$ and $"Decap"("sk", c) -> K$. A KEM is non-interactive in one direction only, which changes protocol design (no contributory behaviour, no static-static patterns). *ML-KEM* (Kyber) is the NIST standard; deployed hybrids combine it with X25519 (X25519MLKEM768 in TLS 1.3, as shipped by Chrome and Cloudflare) so that security holds if _either_ component survives.

== X.509 and the Web PKI

=== Certificates and Chains

An X.509 certificate binds a subject (DNS name, organisation) to a public key, signed by an issuer. Verification builds a *chain* from the end-entity certificate through intermediates to a *root CA* in the client's trust store. Standard checks: signature validity, validity period, name constraints, key usage / extended key usage, basic constraints (CA flag, path length), and hostname matching against subjectAltName.

Chain building is genuinely hard: cross-signatures, expired intermediates, and multiple valid paths caused the May 2020 *AddTrust root expiry* breakage across countless clients with naive path builders.

=== Certificate Issuance and ACME

*ACME* (#rfc(8555), the Let's Encrypt protocol) automated domain-validated issuance: the CA challenges the requester to prove control via HTTP-01 (well-known URL), DNS-01 (TXT record), or TLS-ALPN-01. Automation moved the web from yearly manual renewal to 90-day (now trending toward 47-day) certificate lifetimes. *Multi-perspective validation* counters BGP-hijack attacks on the validation path itself.

=== Revocation

The weakest part of the PKI:
- *CRLs*: signed lists of revoked serials; grow large, fetched rarely.
- *OCSP*: per-certificate status queries; leaks browsing history to the CA and fails open in practice (browsers soft-fail).
- *OCSP stapling*: the server attaches a short-lived signed OCSP response in the handshake; with the `must-staple` extension this becomes sound, but deployment stalled.
- Browsers now ship aggregated revocation sets (CRLite, OneCRL) and rely on short certificate lifetimes as the real mitigation.

=== Certificate Transparency

After the DigiNotar (2011) and Symantec mis-issuance incidents, *Certificate Transparency* (#rfc(9162), obsoletes #rfc(6962)) made issuance publicly auditable: CAs submit certificates to append-only Merkle-tree logs and receive *signed certificate timestamps* (SCTs); Chrome and Safari reject certificates without SCTs from independent logs. CT does not prevent mis-issuance — it guarantees detection, which has proven sufficient to discipline the CA ecosystem (mass distrust of Symantec, 2018).

== Beyond the Web PKI

- *SSH*: trust-on-first-use host keys, optionally SSH certificates (small-scale CA) — pragmatic where global naming is unnecessary.
- *DANE/TLSA*: pin keys in DNSSEC-signed DNS; deployed for SMTP, not browsers.
- *Internal PKI*: service meshes (SPIFFE/SPIRE, Istio) issue short-lived workload identities (X.509 SVIDs) from private CAs — certificate lifetimes of minutes to hours replace revocation entirely.

== Design Checklist

- Always use ephemeral exchange (ECDHE or hybrid KEM); never ship static-RSA key transport.
- Derive keys through HKDF bound to the full transcript and protocol labels.
- Authenticate both directions or document exactly which side is anonymous.
- Plan for post-quantum hybrids now: key exchange is the urgent half (harvest-now-decrypt-later), signatures can lag.
- Treat revocation as unreliable; prefer short lifetimes and automation.

== Further Reading

- Diffie, W., & Hellman, M. (1976). New directions in cryptography. _IEEE Transactions on Information Theory_, 22(6).
- Krawczyk, H. (2003). SIGMA: the 'SIGn-and-MAc' approach to authenticated Diffie-Hellman. _CRYPTO_.
- Adrian, D. et al. (2015). Imperfect forward secrecy: how Diffie-Hellman fails in practice (Logjam). _CCS_.
- Perrin, T. The Noise protocol framework. noiseprotocol.org.
- Marlinspike, M., & Perrin, T. (2016). The X3DH key agreement protocol and the Double Ratchet algorithm. Signal specifications.
- Laurie, B. (2014). Certificate transparency. _ACM Queue_, 12(8).

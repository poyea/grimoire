#import "../template.typ": rfc, xref

= TLS: Transport Layer Security

TLS (formerly SSL) is the cryptographic protocol securing nearly all internet traffic. Every HTTPS connection, gRPC call, and database connection uses TLS. Understanding TLS means understanding how the cryptographic primitives from earlier chapters compose into a real protocol: asymmetric cryptography for authentication and key exchange, symmetric ciphers for bulk encryption, and MACs for integrity.

*See also:* #xref("cryptography-and-security", "asymmetric", label: "Asymmetric Cryptography"), #xref("cryptography-and-security", "symmetric-primitives", label: "Symmetric Primitives"), #xref("cryptography-and-security", "hashing-and-macs", label: "Hashing and MACs"), #xref("cryptography-and-security", "post-quantum", label: "Post-Quantum Cryptography"), #xref("networking", "application-protocols", label: "Application Protocols") (networking), _TLS_ (networking, full protocol detail); #xref("networking", "tls", label: "TLS") (the protocol and performance view of the same handshake).

== Protocol Versions

TLS has gone through several iterations, each retiring broken constructions:

- *SSL 3.0* — broken by the POODLE attack (Möller et al. 2014); padding oracle on CBC mode. #rfc(7568) prohibits SSL 3.0.
- *TLS 1.0 / 1.1* — deprecated by #rfc(8996) (2021); both share CBC-mode vulnerabilities (BEAST, Lucky Thirteen). Browsers dropped them in 2020.
- *TLS 1.2* (#rfc(5246), 2008) — still widely deployed; supports AEAD ciphers when configured correctly; vulnerable if misconfigured (RC4, static RSA).
- *TLS 1.3* (#rfc(8446), 2018) — current standard; removed all legacy constructions; mandatory forward secrecy; simplified handshake; faster.

The practical enforcement baseline is TLS 1.2 minimum, TLS 1.3 preferred. NIST SP 800-52r2 prohibits SSL and TLS 1.0/1.1 in federal systems.

== TLS 1.3 Handshake (1-RTT)

TLS 1.3 completes authentication and key exchange in a single round trip:

```text
Client                                    Server
  │                                          │
  │──── ClientHello ───────────────────────► │
  │     supported_versions: [TLS 1.3]        │
  │     cipher_suites: [AES_128_GCM_SHA256…] │
  │     key_share: X25519 pubkey             │
  │     server_name: example.com (SNI)       │
  │                                          │
  │◄─── ServerHello ────────────────────────  │
  │     key_share: X25519 pubkey             │
  │     cipher_suite: AES_128_GCM_SHA256     │
  │                                          │
  │  [Server derives handshake keys]         │
  │                                          │
  │◄─── EncryptedExtensions ─────────────── │
  │◄─── Certificate ────────────────────── │
  │◄─── CertificateVerify ──────────────── │
  │     (sig over transcript hash)           │
  │◄─── Finished ───────────────────────── │
  │     (HMAC over transcript)               │
  │                                          │
  │  [Client verifies cert chain + sig]      │
  │                                          │
  │──── Finished ──────────────────────────► │
  │                                          │
  │  [Both derive application traffic keys]  │
  │◄════ Application Data ════════════════► │
```

*Key schedule* — TLS 1.3 uses an HKDF-based extract-and-expand construction. Starting from the ephemeral shared secret and optional PSK:

```text
0                                    PSK (or 0)
│                                      │
▼                                      ▼
HKDF-Extract(0, 0) ──── Early Secret ──── HKDF-Extract(early_secret, DHE)
                                              │
                                         Handshake Secret
                                              │
                              HKDF-Expand-Label(hs, "c hs traffic", …) → client_handshake_key
                              HKDF-Expand-Label(hs, "s hs traffic", …) → server_handshake_key
                                              │
                              HKDF-Extract(handshake_secret, 0)
                                              │
                                         Master Secret
                              HKDF-Expand-Label(ms, "c ap traffic", …) → client_app_key
                              HKDF-Expand-Label(ms, "s ap traffic", …) → server_app_key
```

The `CertificateVerify` message signs the entire transcript hash up to that point, binding the authentication to the exact handshake sequence and preventing transcript substitution attacks.

== 0-RTT Session Resumption

TLS 1.3 supports 0-RTT data using a Pre-Shared Key (PSK) from a previous session. The server issues a `NewSessionTicket` post-handshake; the client reuses it in the next connection's `ClientHello`:

```text
Client                          Server
  │── ClientHello + early_data ──► │
  │   (PSK from previous ticket)   │
  │   (0-RTT application data)     │
  │                                │
  │◄── ServerHello ──────────────  │
  │◄── ... Finished ─────────────  │
  │──── Finished ────────────────► │
```

*Replay risk:* 0-RTT data is not forward-secret relative to the ticket key, and the server cannot distinguish a replayed first flight from a genuine one. Safe uses: idempotent GET requests, read-only RPCs. Unsafe: state-mutating requests, payments. Defenses include single-use tickets (server-side state), time-window rejection (tickets carry a timestamp; server rejects outside window), and application-layer idempotency tokens.

== Cipher Suites in TLS 1.3

TLS 1.3 reduced the cipher suite list to five, all using AEAD:

#table(
  columns: 3,
  [*Suite*], [*AEAD*], [*PRF hash*],
  [`TLS_AES_128_GCM_SHA256`], [AES-128-GCM], [SHA-256],
  [`TLS_AES_256_GCM_SHA384`], [AES-256-GCM], [SHA-384],
  [`TLS_CHACHA20_POLY1305_SHA256`], [ChaCha20-Poly1305], [SHA-256],
  [`TLS_AES_128_CCM_SHA256`], [AES-128-CCM], [SHA-256],
  [`TLS_AES_128_CCM_8_SHA256`], [AES-128-CCM-8 (short tag)], [SHA-256],
)

Key exchange is always ephemeral: ECDHE (typically X25519 or P-256) or DHE. Static RSA key exchange — which allowed passive decryption of recorded traffic if the private key was later obtained — was eliminated entirely.

TLS 1.2 still supports hundreds of legacy suites. Configuration guidance: disable all non-AEAD suites (CBC, RC4, 3DES), require ECDHE or DHE, prefer SHA-256+.

== Forward Secrecy

*Forward secrecy* (or perfect forward secrecy) means that compromise of a server's long-term private key does not enable decryption of previously recorded sessions.

In TLS 1.3, the shared secret comes from ephemeral key exchange (the client and server each generate a fresh DH keypair per handshake). The long-term private key is only used in `CertificateVerify` to authenticate — not to derive the session keys. Recording traffic and later obtaining the server's certificate private key yields only the identity proof, not the session keys.

TLS 1.2 with static RSA key exchange had no forward secrecy: the pre-master secret was encrypted directly with the server's RSA public key. Decrypting old recordings required only the server's RSA private key.

Session tickets are encrypted with a server-side ticket key. Compromise of the ticket key breaks session resumption privacy but does not retroactively expose sessions that used fresh ECDHE.

== Certificate Validation and PKI

TLS authentication rests on the Web PKI:

+ *Chain validation:* leaf certificate → one or more intermediates → root CA trusted by the client.
+ *Validity period:* `notBefore`/`notAfter` fields; browsers enforce hard maximum (398 days since 2020).
+ *Subject Alternative Names (SANs):* the `subjectAltName` extension lists the hostnames the certificate covers. The CN field is deprecated for hostname matching.
+ *Revocation:* CRL (Certificate Revocation List) — signed list of serial numbers; large and infrequently fetched. OCSP (Online Certificate Status Protocol) — real-time query but introduces latency and a privacy leak (the CA learns which sites you visit).
+ *OCSP stapling:* the server fetches and caches a signed OCSP response, stapling it into the TLS handshake. The client does not need to query the CA; latency and privacy problem solved.
+ *Certificate Transparency (CT):* since 2018, all publicly trusted certificates must be logged in CT logs before issuance; browsers reject certificates without two signed certificate timestamps (SCTs). Provides post-hoc detection of mis-issuance.
+ *Certificate pinning:* an application hardcodes the expected certificate or SPKI hash. Protects against rogue CAs; brittle — pin rotation requires app updates. HPKP (HTTP Public Key Pinning) was deprecated; modern practice uses CT monitoring instead.

== SNI and Encrypted Client Hello

*Server Name Indication (SNI)* is a TLS extension in the ClientHello that carries the target hostname in plaintext. This lets servers hosting multiple certificates (and middleboxes doing traffic routing) see the destination before the handshake completes.

Privacy problem: SNI is visible to any network observer, revealing the destination even when the content is encrypted.

*Encrypted Client Hello (ECH, draft-ietf-tls-esni)* solves this by splitting the ClientHello into an outer (visible) and inner (encrypted). The client fetches the server's ECH public key from a DNS HTTPS record and uses HPKE to encrypt the inner ClientHello:

```text
DNS HTTPS record → ech_config (ECH public key + metadata)
ClientHello outer: visible SNI = "cloudflare-esni.com" (cover domain)
ClientHello inner: real SNI = "private-site.example.com" (encrypted)
```

ECH is supported by Chrome and Firefox and deployed by Cloudflare, though DNS HTTPS record rollout is still in progress.

== mTLS (Mutual TLS)

Standard TLS authenticates the server to the client. *Mutual TLS* requires the client to also present a certificate, authenticating both parties:

```text
Handshake addition:
  ◄── CertificateRequest ──────── (server requests client cert)
  ──── Certificate ────────────► (client sends its cert)
  ──── CertificateVerify ───────► (client signs transcript)
```

Use cases:

- *Service meshes* (Istio, Linkerd): each sidecar proxy holds a workload certificate; all inter-service traffic uses mTLS. The mesh control plane automates certificate rotation.
- *Internal APIs* and microservices: service-to-service authentication without API keys.
- *Client-authenticated VPNs:* mutual auth before tunnel establishment.
- *SPIFFE/SPIRE:* a workload identity standard; issues short-lived X.509 SVIDs (SPIFFE Verifiable Identity Documents) to workloads; Kubernetes pod identity maps to SPIFFE URIs.

== Performance

TLS 1.3 is faster than TLS 1.2 in two important ways:

#table(
  columns: 3,
  [*Metric*], [*TLS 1.2*], [*TLS 1.3*],
  [Full handshake], [2 RTT], [1 RTT],
  [Resumption], [1 RTT (session ticket)], [0 RTT (PSK early data)],
  [CPU — AES-NI], [≈1–3% overhead], [same],
  [CPU — no AES-NI], [≈10–15% overhead], [comparable; ChaCha20 preferred],
)

AES-GCM with AES-NI hardware instructions (present on all x86 since ~2011, ARM since Cortex-A57) costs roughly 1–3% CPU overhead for typical HTTPS workloads. Without AES-NI (embedded, older ARMv7), ChaCha20-Poly1305 is significantly faster and is the preferred suite.

Session tickets avoid repeating the asymmetric key exchange on returning connections; the cost is symmetric encryption of the ticket (negligible).

== Post-Quantum TLS

Harvest-now-decrypt-later attacks motivate deploying post-quantum key exchange before quantum computers are practically available. The threat: an adversary records TLS sessions today and decrypts them once a sufficiently large quantum computer exists.

*NIST post-quantum standards* (2024):
- *ML-KEM* (FIPS 203, formerly Kyber) — module lattice-based KEM for key exchange.
- *ML-DSA* (FIPS 204, formerly Dilithium) — module lattice-based signatures.

*Hybrid key exchange:* combine classical and post-quantum in a single handshake. If either is unbroken, the session is secure. The `X25519MLKEM768` key share group (X25519 + ML-KEM-768) is already deployed:

- Chrome enabled X25519MLKEM768 by default in 2024.
- Cloudflare deploys it for all TLS 1.3 connections.
- The IETF draft `draft-ietf-tls-hybrid-design` specifies the combination mechanism.

*Size overhead:* ML-KEM-768 has a 1184-byte public key and 1088-byte ciphertext, versus 32 bytes for X25519. The hybrid key_share extension adds ~2.2 KB to the ClientHello. TCP slow start and MTU constraints mean the first few packets may require an extra round trip for large handshakes.

Post-quantum signatures (ML-DSA) are much larger (~2.5 KB) and not yet deployed in TLS by default; certificate chain size would increase substantially. Research on signature compression and KEMTLS (authentication via KEM rather than signatures) is ongoing.

== Further Reading

Rescorla, E. (2026). "The Transport Layer Security (TLS) Protocol Version 1.3." #rfc(9846). (Obsoletes #rfc(8446), 2018.)

Rescorla, E., Oku, K., Sullivan, N., Wood, C. A. "TLS Encrypted Client Hello." draft-ietf-tls-esni (IETF work in progress).

Bhargavan, K., Blanchet, B., Kobeissi, N. (2017). "Verified Models and Reference Implementations for the TLS 1.3 Standard Candidate." IEEE S&P. (Formal analysis of TLS 1.3.)

Kobeissi, N., Bhargavan, K., Blanchet, B. (2017). "Automated Verification for Secure Messaging Protocols and Their Implementations." IEEE EuroS&P.

Mozilla. "Mozilla SSL Configuration Generator." https://ssl-config.mozilla.org — recommended cipher and protocol configuration for nginx, Apache, HAProxy.

NIST SP 800-52r2. "Guidelines for the Selection, Configuration, and Use of Transport Layer Security (TLS) Implementations." NIST, 2019.

Langley, A. (2015). "ImperialViolet: How to Build a TLS-Terminating Load Balancer." https://www.imperialviolet.org — session ticket design and rotation.

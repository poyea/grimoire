= TLS (Transport Layer Security)

TLS provides confidentiality, integrity, and authentication for network communications. It operates above TCP, encrypting application data before transmission.

*See also:* Transport Layer (for TCP foundations), Application Protocols (for HTTP/2, HTTP/3), Kernel Bypass (for hardware acceleration)

== Protocol Evolution

*TLS history:*
- SSL 2.0 (1995): Netscape, deprecated due to cryptographic flaws
- SSL 3.0 (1996): Major redesign, deprecated 2015 [RFC 7568]
- TLS 1.0 (1999): IETF standardization [RFC 2246]
- TLS 1.1 (2006): Explicit IV, deprecated 2021 [RFC 8996]
- TLS 1.2 (2008): AEAD ciphers, SHA-256 [RFC 5246]
- TLS 1.3 (2018): Simplified handshake, 0-RTT [RFC 8446]

*Current recommendation:* TLS 1.3 required, TLS 1.2 acceptable for legacy systems.

== TLS 1.2 Full Handshake (2-RTT)

```
Client                                               Server
  │                                                    │
  │  ClientHello                                       │
  │  (version, random, cipher_suites, extensions)      │
  ├───────────────────────────────────────────────────►│
  │                                                    │
  │                                       ServerHello  │
  │                          (version, random, cipher) │
  │                                        Certificate │
  │                                    ServerKeyExchange│
  │                                    ServerHelloDone │
  │◄───────────────────────────────────────────────────┤
  │                                                    │  1 RTT
  │  ClientKeyExchange                                 │
  │  ChangeCipherSpec                                  │
  │  Finished                                          │
  ├───────────────────────────────────────────────────►│
  │                                                    │
  │                                  ChangeCipherSpec  │
  │                                           Finished │
  │◄───────────────────────────────────────────────────┤
  │                                                    │  2 RTT
  │  [Application Data encrypted]                      │
  ├───────────────────────────────────────────────────►│
```

*Latency cost:* 2 RTT before application data.
- Localhost: ~1ms
- Same datacenter: 2-4ms
- Cross-country (50ms RTT): 100-200ms
- Intercontinental (150ms RTT): 300-600ms

== TLS 1.3 Full Handshake (1-RTT)

*Key improvement:* Client sends key share in first message, eliminating one round trip [RFC 8446].

```
Client                                               Server
  │                                                    │
  │  ClientHello                                       │
  │  + key_share (ECDHE public key)                    │
  │  + supported_versions (TLS 1.3)                    │
  │  + signature_algorithms                            │
  ├───────────────────────────────────────────────────►│
  │                                                    │
  │                                       ServerHello  │
  │                                       + key_share  │
  │                             {EncryptedExtensions}  │
  │                                      {Certificate} │
  │                               {CertificateVerify}  │
  │                                         {Finished} │
  │◄───────────────────────────────────────────────────┤
  │                                                    │  1 RTT
  │  {Finished}                                        │
  ├───────────────────────────────────────────────────►│
  │                                                    │
  │  [Application Data encrypted]                      │
  ├───────────────────────────────────────────────────►│
```

*Notation:* `{}` indicates encrypted with handshake keys.

*Latency savings:* 1 RTT vs 2 RTT = 50% reduction in handshake time.

== 0-RTT Session Resumption

*TLS 1.3 allows sending application data in the first flight using pre-shared keys (PSK) [RFC 8446 Section 2.3]:*

```
Client                                               Server
  │                                                    │
  │  ClientHello                                       │
  │  + early_data (0-RTT indicator)                    │
  │  + pre_shared_key (PSK identity)                   │
  │  + key_share                                       │
  │  [Application Data] (encrypted with PSK)           │
  ├───────────────────────────────────────────────────►│
  │                                                    │  0 RTT!
  │                                       ServerHello  │
  │                                       + key_share  │
  │                             {EncryptedExtensions}  │
  │                                         {Finished} │
  │◄───────────────────────────────────────────────────┤
  │                                                    │
  │  {Finished}                                        │
  ├───────────────────────────────────────────────────►│
```

*0-RTT limitations:*
- *Replay vulnerability:* No forward secrecy for 0-RTT data; attacker can replay first flight
- *Idempotent operations only:* Safe for GET requests, unsafe for POST (non-idempotent)
- *Limited data:* Server may reject or limit early data size

*Mitigation:* Servers should implement replay protection (single-use tickets, strike registers).

== Cipher Suites

*TLS 1.2 cipher suite format:*
```
TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
     │      │        │    │    │
     │      │        │    │    └── PRF hash
     │      │        │    └─────── Key size
     │      │        └──────────── Bulk cipher + mode
     │      └───────────────────── Authentication
     └──────────────────────────── Key exchange
```

*TLS 1.3 simplified cipher suites (only AEAD ciphers):*
- `TLS_AES_128_GCM_SHA256` (mandatory)
- `TLS_AES_256_GCM_SHA384`
- `TLS_CHACHA20_POLY1305_SHA256`

*Key exchange separated:* Negotiated via `supported_groups` extension (ECDHE only).

*Removed in TLS 1.3:*
- RSA key exchange (no forward secrecy)
- Static DH (no forward secrecy)
- CBC-mode ciphers (padding oracle attacks)
- RC4, DES, 3DES (weak ciphers)
- MD5, SHA-1 for signatures

== Cipher Suite Performance

*Throughput comparison (modern x86-64 with AES-NI):*

#table(
  columns: 4,
  align: (left, right, right, left),
  table.header([Cipher], [Encrypt], [Decrypt], [Notes]),
  [AES-128-GCM], [~6 GB/s], [~6 GB/s], [Hardware accelerated],
  [AES-256-GCM], [~5 GB/s], [~5 GB/s], [Hardware accelerated],
  [ChaCha20-Poly1305], [~2 GB/s], [~2 GB/s], [Better without AES-NI],
)

*Key exchange performance (operations per second, single core):*

#table(
  columns: 3,
  align: (left, right, right),
  table.header([Algorithm], [Key Gen], [Shared Secret]),
  [ECDHE P-256], [~40,000/s], [~15,000/s],
  [ECDHE X25519], [~70,000/s], [~25,000/s],
  [RSA-2048 sign], [~1,500/s], [N/A],
)

*Recommendation:* Use X25519 for key exchange, AES-256-GCM or ChaCha20-Poly1305 for bulk encryption.

== Certificate Chains and Validation

*Certificate chain structure:*
```
┌─────────────────────────────────────────┐
│  End-Entity Certificate                  │
│  (Subject: www.example.com)              │
│  (Issuer: Intermediate CA)               │
│  (Public Key: RSA-2048 or ECDSA P-256)   │
└────────────────────┬────────────────────┘
                     │ signed by
┌────────────────────▼────────────────────┐
│  Intermediate CA Certificate             │
│  (Subject: Intermediate CA)              │
│  (Issuer: Root CA)                       │
└────────────────────┬────────────────────┘
                     │ signed by
┌────────────────────▼────────────────────┐
│  Root CA Certificate (Trust Anchor)      │
│  (Subject: Root CA)                      │
│  (Self-signed)                           │
│  [Stored in OS/browser trust store]      │
└─────────────────────────────────────────┘
```

*Validation steps [RFC 5280]:*
1. Verify each signature in chain (bottom to top)
2. Check validity period (notBefore, notAfter)
3. Verify certificate purposes (Key Usage, Extended Key Usage)
4. Check revocation status (CRL, OCSP)
5. Validate hostname matches certificate (Subject CN or SAN)

*OCSP Stapling [RFC 6066]:*
- Server fetches OCSP response and includes in TLS handshake
- Avoids client-side OCSP lookup latency (50-200ms)
- *Benefit:* Faster handshake, improved privacy

== Session Resumption Mechanisms

*1. Session IDs (TLS 1.2):*
```cpp
// Server stores session state
struct SessionCache {
    uint8_t session_id[32];
    uint8_t master_secret[48];
    CipherSuite cipher;
    time_t created;
};

// Client sends previous session_id in ClientHello
// Server looks up, skips full handshake if found
```

*Problem:* Requires server-side storage, doesn't scale across server farms.

*2. Session Tickets [RFC 5077]:*
- Server encrypts session state, sends to client as opaque ticket
- Client stores ticket, presents in next ClientHello
- Server decrypts ticket, resumes session without lookup

*Security:* Ticket encryption key must be rotated frequently; shared key across servers enables resumption.

*3. TLS 1.3 PSK Resumption:*
- Server sends `NewSessionTicket` after handshake completes
- Contains PSK identity and ticket age
- Client uses PSK for 0-RTT or 1-RTT resumption

== Handshake Latency Analysis

*Total connection time = TCP handshake + TLS handshake:*

#table(
  columns: 4,
  align: (left, right, right, right),
  table.header([Scenario], [TCP], [TLS 1.2], [TLS 1.3]),
  [Localhost], [\<1ms], [~2ms], [~1ms],
  [Datacenter (1ms RTT)], [1ms], [4ms], [2ms],
  [Regional (20ms RTT)], [20ms], [80ms], [40ms],
  [Cross-country (50ms RTT)], [50ms], [200ms], [100ms],
  [Intercontinental (150ms RTT)], [150ms], [600ms], [300ms],
)

*With 0-RTT resumption (TLS 1.3):*
- Cross-country: 50ms TCP + 0ms TLS = 50ms (vs 200ms for TLS 1.2)
- *Savings:* 150ms, 75% reduction

== CPU Cost Analysis

*Handshake CPU cost (RSA-2048 vs ECDSA P-256):*
- RSA-2048 signing: ~1.5ms per operation
- ECDSA P-256 signing: ~0.1ms per operation
- *Recommendation:* Use ECDSA certificates for high-traffic servers

*Bulk encryption overhead:*
- AES-GCM with AES-NI: 1-3% CPU overhead
- Without hardware acceleration: 10-30% overhead

*Optimizations:*
```bash
# Verify AES-NI support
grep -o aes /proc/cpuinfo | head -1

# OpenSSL speed test
openssl speed -evp aes-128-gcm
openssl speed -evp chacha20-poly1305
```

== Common Pitfalls

*1. Protocol downgrade attacks:*
- Attacker forces older, weaker TLS version
- *Mitigation:* `downgrade_sentinel` in TLS 1.3 ServerHello random bytes

*2. Certificate validation bypass:*
```cpp
// WRONG: Disabling certificate validation
SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, NULL);

// CORRECT: Always verify with proper callback
SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, verify_callback);
```

*3. Weak cipher suites:*
```bash
# Check server cipher suites
openssl s_client -connect example.com:443 -cipher 'ALL' 2>/dev/null | \
    grep "Cipher is"

# Recommended OpenSSL cipher string
"ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS"
```

*4. Missing HSTS (HTTP Strict Transport Security):*
```http
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
```

*5. TLS compression (CRIME attack):*
- Compression leaks plaintext length
- *Solution:* Disable TLS compression (disabled by default in TLS 1.3)

== Security Considerations

*Forward secrecy:*
- Ephemeral key exchange (ECDHE) ensures past sessions cannot be decrypted if long-term key is compromised
- TLS 1.3 mandates forward secrecy; static RSA key exchange removed

*Replay protection (0-RTT):*
```cpp
// Server-side single-use ticket implementation
struct TicketStore {
    std::unordered_set<std::string> used_tickets;
    std::mutex lock;

    bool try_use(const std::string& ticket_id) {
        std::lock_guard<std::mutex> guard(lock);
        auto [it, inserted] = used_tickets.insert(ticket_id);
        return inserted;  // false if replay
    }
};
```

*Key rotation:*
- Session ticket keys: Rotate every 24-48 hours
- Server certificates: Rotate annually or upon compromise
- Private keys: Secure storage (HSM for high-security environments)

== Implementation Example (OpenSSL)

```cpp
#include <openssl/ssl.h>
#include <openssl/err.h>

SSL_CTX* create_tls13_context() {
    // Create TLS 1.3 context
    SSL_CTX* ctx = SSL_CTX_new(TLS_server_method());

    // Set minimum version to TLS 1.2, prefer TLS 1.3
    SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
    SSL_CTX_set_max_proto_version(ctx, TLS1_3_VERSION);

    // Load certificate chain
    SSL_CTX_use_certificate_chain_file(ctx, "server.crt");
    SSL_CTX_use_PrivateKey_file(ctx, "server.key", SSL_FILETYPE_PEM);

    // Configure cipher suites
    // TLS 1.3 ciphers
    SSL_CTX_set_ciphersuites(ctx,
        "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256");

    // TLS 1.2 ciphers (fallback)
    SSL_CTX_set_cipher_list(ctx,
        "ECDHE+AESGCM:ECDHE+CHACHA20:!aNULL:!MD5");

    // Enable session tickets for resumption
    SSL_CTX_set_session_cache_mode(ctx, SSL_SESS_CACHE_SERVER);

    // Enable OCSP stapling
    SSL_CTX_set_tlsext_status_type(ctx, TLSEXT_STATUSTYPE_ocsp);

    return ctx;
}
```

== References

*Primary sources:*

RFC 8446: The Transport Layer Security (TLS) Protocol Version 1.3. Rescorla, E. (2018).

RFC 5246: The Transport Layer Security (TLS) Protocol Version 1.2. Dierks, T. & Rescorla, E. (2008).

RFC 5280: Internet X.509 PKI Certificate and CRL Profile. Cooper, D., et al. (2008).

RFC 5077: Transport Layer Security (TLS) Session Resumption without Server-Side State. Salowey, J., et al. (2008).

RFC 6066: Transport Layer Security (TLS) Extensions: Extension Definitions. Eastlake, D. (2011).

RFC 7568: Deprecating Secure Sockets Layer Version 3.0. Barnes, R., et al. (2015).

RFC 8996: Deprecating TLS 1.0 and TLS 1.1. Moriarty, K. & Farrell, S. (2021).

Langley, A., Modadugu, N., & Moeller, B. (2016). "Transport Layer Security (TLS) False Start." RFC 7918.

Sullivan, N. (2016). "Keyless SSL: The Nitty Gritty Technical Details." Cloudflare Blog.

Sy, E., et al. (2018). "Tracking Users across the Web via TLS Session Resumption." ACSAC '18.

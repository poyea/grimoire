= Network Security

Network security encompasses protocols, architectures, and practices that protect data in transit across untrusted networks. This section covers encryption protocols (IPSec, TLS), VPN architectures, and modern solutions like WireGuard.

*See also:* Transport Layer (for TCP/UDP foundations), Application Protocols (for TLS/HTTP), Kernel Bypass (for high-performance security appliances)

== IPSec (Internet Protocol Security)

*Suite of protocols providing authentication, integrity, and encryption at the IP layer [RFC 4301-4309].*

IPSec operates transparently to applications - no code changes required. Two core protocols:

*1. AH (Authentication Header) [RFC 4302]:*
- Provides integrity and authentication
- No encryption (data visible to inspection)
- Covers entire IP packet including headers (except mutable fields)
- Use case: Authentication-only environments, debugging

*AH header structure:*
```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Next Header  |  Payload Len  |          Reserved             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                 Security Parameters Index (SPI)               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Sequence Number                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|                    Integrity Check Value (ICV)                |
|                          (variable)                           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

*2. ESP (Encapsulating Security Payload) [RFC 4303]:*
- Provides confidentiality, integrity, and authentication
- Encrypts payload (data hidden from inspection)
- Does not protect outer IP header (enables NAT traversal)
- *Preferred for most deployments*

*ESP packet structure:*
```
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                 Security Parameters Index (SPI)               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Sequence Number                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|                    Payload Data (encrypted)                   |
~                                                               ~
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|     Padding (0-255 bytes)     |  Pad Length   | Next Header   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|         Integrity Check Value (ICV) (variable)                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

== IPSec Modes: Transport vs Tunnel

*Transport Mode:*
- Encrypts only payload, original IP header preserved
- Used for host-to-host communication
- Lower overhead (no additional IP header)

```
Original:  [IP Header][TCP/UDP][Payload]
Transport: [IP Header][ESP Header][TCP/UDP][Payload][ESP Trailer]
```

*Tunnel Mode:*
- Encrypts entire original packet, adds new IP header
- Used for gateway-to-gateway (site-to-site VPN)
- Hides internal network topology

```
Original:  [IP Header][TCP/UDP][Payload]
Tunnel:    [New IP Header][ESP Header][IP Header][TCP/UDP][Payload][ESP Trailer]
```

*Visual comparison:*
```
Transport Mode (host-to-host):
  Host A ────────────────────────────────────────── Host B
    │              Encrypted payload                   │
    └──────────────────────────────────────────────────┘

Tunnel Mode (site-to-site):
  [Site A]              [Internet]              [Site B]
  Host ── Gateway ═══════════════════════ Gateway ── Host
             │      Encrypted tunnel           │
             └─────────────────────────────────┘
```

== IKE (Internet Key Exchange)

*Protocol for establishing IPSec Security Associations (SAs) [RFC 7296].*

IKEv2 establishes shared keys and negotiates algorithms in two phases:

*Phase 1 (IKE_SA_INIT):* Establish secure channel for negotiation
*Phase 2 (IKE_AUTH):* Authenticate peers and create Child SAs

*IKEv2 exchange (4 messages):*
```
Initiator                           Responder
    │                                   │
    │ IKE_SA_INIT (DH, nonce, proposal) │
    ├──────────────────────────────────►│
    │                                   │
    │ IKE_SA_INIT (DH, nonce, proposal) │
    │◄──────────────────────────────────┤
    │                                   │
    │ IKE_AUTH (ID, AUTH, SA, TS)      │  ─┐
    ├──────────────────────────────────►│   ├─ Encrypted
    │                                   │   │
    │ IKE_AUTH (ID, AUTH, SA, TS)      │   │
    │◄──────────────────────────────────┤  ─┘
    │                                   │
    │    Child SA established           │
```

*Cryptographic algorithms (recommended):*
- Key exchange: ECDH (Curve25519, P-256)
- Encryption: AES-256-GCM (AEAD)
- Integrity: SHA-256, SHA-384
- PRF: HMAC-SHA-256

== VPN Architectures

*1. Site-to-Site VPN:*
- Connects entire networks through gateway devices
- Transparent to end users
- Uses tunnel mode IPSec

```
┌─────────────────┐           ┌─────────────────┐
│   Site A        │           │   Site B        │
│  10.0.1.0/24    │           │  10.0.2.0/24    │
│  ┌───────────┐  │           │  ┌───────────┐  │
│  │  Hosts    │  │           │  │  Hosts    │  │
│  └─────┬─────┘  │           │  └─────┬─────┘  │
│        │        │           │        │        │
│  ┌─────┴─────┐  │  IPSec    │  ┌─────┴─────┐  │
│  │  Gateway  │──┼───Tunnel──┼──│  Gateway  │  │
│  └───────────┘  │           │  └───────────┘  │
└─────────────────┘           └─────────────────┘
```

*2. Remote Access VPN:*
- Individual clients connect to corporate network
- Client software required (IKEv2, L2TP/IPSec, OpenVPN)
- Uses tunnel mode with client-assigned virtual IP

*3. Hub-and-Spoke VPN:*
- Central hub connects multiple remote sites
- All traffic routes through hub
- Simplified management, single point of failure

*4. Full Mesh VPN:*
- Direct tunnels between all sites
- Optimal routing, complex configuration
- N sites require N(N-1)/2 tunnels

== WireGuard

*Modern VPN protocol designed for simplicity and performance [Donenfeld 2017].*

*Design principles:*
- Minimal attack surface: ~4,000 lines of code (vs 100,000+ for OpenVPN)
- Cryptographically opinionated: No cipher negotiation, single suite
- Stateless: Connections appear/disappear seamlessly

*Cryptographic primitives:*
- Key exchange: Curve25519 (ECDH)
- Encryption: ChaCha20 (stream cipher)
- Authentication: Poly1305 (MAC)
- Hashing: BLAKE2s
- Key derivation: HKDF

*Noise Protocol Framework [Perrin 2018]:*
```
WireGuard uses Noise_IKpsk2:
- I: Initiator sends static key
- K: Responder static key known to initiator
- psk2: Pre-shared key mixed in at message 2

Handshake (1-RTT):
Initiator ──── Noise_IKpsk2 message 1 ────► Responder
         ◄──── Noise_IKpsk2 message 2 ────
         ──── Encrypted application data ──►
```

*WireGuard packet structure:*
```
┌──────────────────────────────────────────────────────┐
│ Type (1) │ Reserved (3) │ Receiver Index (4)         │
├──────────────────────────────────────────────────────┤
│                    Counter (8)                       │
├──────────────────────────────────────────────────────┤
│              Encrypted Payload (variable)            │
│              (ChaCha20-Poly1305)                     │
├──────────────────────────────────────────────────────┤
│                   Auth Tag (16)                      │
└──────────────────────────────────────────────────────┘
```

*Performance advantage:*
- Kernel-space implementation (Linux, Windows, macOS, BSD)
- Single round-trip handshake
- No TCP-over-TCP meltdown (UDP only)
- Hardware acceleration for ChaCha20 on ARM

== SSH Tunneling and Port Forwarding

*SSH provides encrypted tunnels without VPN infrastructure [RFC 4251-4254].*

*1. Local Port Forwarding (-L):*
```bash
# Forward local:8080 → remote:3306 through SSH
ssh -L 8080:database.internal:3306 user@bastion

# Access database as localhost:8080
mysql -h 127.0.0.1 -P 8080
```

```
┌──────────┐        ┌──────────┐        ┌──────────┐
│  Client  │──SSH──►│ Bastion  │───────►│ Database │
│ :8080    │        │          │        │ :3306    │
└──────────┘        └──────────┘        └──────────┘
```

*2. Remote Port Forwarding (-R):*
```bash
# Expose local service to remote network
ssh -R 8080:localhost:3000 user@remote

# Remote users access localhost:8080 → your :3000
```

*3. Dynamic Port Forwarding (-D):*
```bash
# SOCKS5 proxy through SSH
ssh -D 1080 user@remote

# Configure browser/application to use SOCKS5 proxy at localhost:1080
```

*4. SSH VPN (tun device):*
```bash
# Create point-to-point tunnel (requires root)
ssh -w 0:0 root@remote

# Configure routing through tun0 interface
```

== Network Segmentation and Zero Trust

*Traditional perimeter security:* Trust internal network, firewall at boundary.

*Problem:* Once attacker breaches perimeter, lateral movement is trivial.

*Zero Trust Architecture [NIST SP 800-207]:*
- Never trust, always verify
- Assume breach mentality
- Least privilege access

*Key principles:*
1. *Microsegmentation:* Fine-grained network boundaries
2. *Identity-centric:* Authentication for every access
3. *Continuous verification:* Re-authenticate frequently
4. *Encrypted everywhere:* mTLS for all internal communication

*Implementation patterns:*

```
Traditional (flat network):
┌─────────────────────────────────────────┐
│  Trusted Internal Network               │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐    │
│  │ Web │──│ App │──│ DB  │──│Admin│    │
│  └─────┘  └─────┘  └─────┘  └─────┘    │
└─────────────────────────────────────────┘

Zero Trust (microsegmented):
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Web DMZ │   │ App Seg │   │ DB Seg  │
│ ┌─────┐ │   │ ┌─────┐ │   │ ┌─────┐ │
│ │ Web │─┼mTLS│ App │─┼mTLS│ DB  │ │
│ └─────┘ │   │ └─────┘ │   │ └─────┘ │
└─────────┘   └─────────┘   └─────────┘
     ▲             ▲             ▲
     └─────────────┴─────────────┘
           Identity Provider
```

== Common Attack Vectors and Mitigations

*1. Man-in-the-Middle (MITM):*
- Attack: Intercept and modify traffic between parties
- Mitigation: TLS certificate validation, certificate pinning, mTLS

*2. Replay Attacks:*
- Attack: Capture and re-send valid encrypted packets
- Mitigation: Sequence numbers (IPSec), nonces, timestamps

*3. Downgrade Attacks:*
- Attack: Force weaker cipher negotiation
- Mitigation: Strict cipher suites, TLS 1.3 (no downgrade)

*4. Key Compromise:*
- Attack: Stolen private key enables decryption
- Mitigation: Perfect Forward Secrecy (PFS), ephemeral keys

*Perfect Forward Secrecy:*
```
Without PFS:
- Static RSA key pair
- Compromise key → decrypt all past traffic

With PFS (ECDHE):
- New ephemeral key per session
- Compromise long-term key → cannot decrypt past sessions
```

*5. DNS Hijacking:*
- Attack: Redirect VPN endpoint resolution
- Mitigation: DNSSEC, DoH/DoT, IP-based configuration

*6. Traffic Analysis:*
- Attack: Infer communication patterns from metadata
- Mitigation: Padding, traffic shaping, onion routing (Tor)

== Performance Overhead of Encryption

*Encryption adds CPU cost and latency. Modern hardware mitigates much of this.*

*CPU overhead (single core, AES-NI enabled):*
```
AES-128-GCM:    ~5 GB/s  (0.2 cycles/byte)
AES-256-GCM:    ~4 GB/s  (0.25 cycles/byte)
ChaCha20-Poly1305: ~3 GB/s  (0.3 cycles/byte, no special hardware)
ChaCha20-Poly1305: ~6 GB/s  (ARM with crypto extensions)
```

*Latency overhead:*
- TLS handshake: 1-2 RTT (50-200ms cross-country)
- IPSec IKEv2: 2 RTT initial, 0 RTT rekeying
- WireGuard: 1 RTT handshake

*VPN throughput comparison (typical benchmarks):*

#table(
  columns: 5,
  align: (left, right, right, right, left),
  table.header([Protocol], [Throughput], [Latency], [CPU], [Notes]),
  [IPSec (AES-GCM)], [5-10 Gbps], [+0.5ms], [10-20%], [Hardware offload common],
  [OpenVPN (UDP)], [200-500 Mbps], [+2-5ms], [50-100%], [Userspace, single-threaded],
  [WireGuard], [3-5 Gbps], [+0.3ms], [5-15%], [Kernel-space, modern crypto],
  [No encryption], [10+ Gbps], [baseline], [baseline], [Reference],
)

*Hardware acceleration:*
- AES-NI (x86): 10x improvement for AES operations
- ARM Crypto Extensions: Native AES/SHA support
- NIC offload: IPSec in hardware (Intel, Mellanox)

*Performance recommendations:*
1. Use AES-GCM with AES-NI for x86 servers
2. Use ChaCha20-Poly1305 for mobile/ARM devices
3. Enable hardware offload where available
4. Avoid OpenVPN for high-throughput requirements
5. Consider WireGuard for best latency-to-security ratio

== Configuration Examples

*Linux IPSec (strongSwan):*
```bash
# /etc/ipsec.conf
conn site-to-site
    left=192.0.2.1
    leftsubnet=10.0.1.0/24
    right=198.51.100.1
    rightsubnet=10.0.2.0/24
    ike=aes256gcm16-sha384-ecp384!
    esp=aes256gcm16-ecp384!
    keyexchange=ikev2
    auto=start
```

*WireGuard:*
```bash
# /etc/wireguard/wg0.conf
[Interface]
PrivateKey = <base64_private_key>
Address = 10.0.0.1/24
ListenPort = 51820

[Peer]
PublicKey = <peer_base64_public_key>
AllowedIPs = 10.0.0.2/32, 192.168.1.0/24
Endpoint = peer.example.com:51820
PersistentKeepalive = 25
```

```bash
# Bring up interface
wg-quick up wg0
```

== References

*IPSec:*

RFC 4301: Security Architecture for the Internet Protocol. Kent, S. & Seo, K. (2005).

RFC 4302: IP Authentication Header. Kent, S. (2005).

RFC 4303: IP Encapsulating Security Payload (ESP). Kent, S. (2005).

RFC 7296: Internet Key Exchange Protocol Version 2 (IKEv2). Kaufman, C., et al. (2014).

*WireGuard:*

Donenfeld, J.A. (2017). "WireGuard: Next Generation Kernel Network Tunnel." NDSS '17.

Perrin, T. (2018). "The Noise Protocol Framework." https://noiseprotocol.org/

*Zero Trust:*

NIST SP 800-207: Zero Trust Architecture. Rose, S., et al. (2020).

*SSH:*

RFC 4251: The Secure Shell (SSH) Protocol Architecture. Ylonen, T. & Lonvick, C. (2006).

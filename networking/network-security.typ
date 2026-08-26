#import "../template.typ": rfc, xref

= Network Security

Network security encompasses protocols, architectures, and practices that protect data in transit across untrusted networks. This section covers encryption protocols (IPSec, TLS), VPN architectures, and modern solutions like WireGuard.

*See also:* #xref("networking", "transport-layer", label: "Transport Layer") (for TCP/UDP foundations), #xref("networking", "application-protocols", label: "Application Protocols") (for TLS/HTTP), #xref("networking", "kernel-bypass", label: "Kernel Bypass") (for high-performance security appliances)

== IPSec (Internet Protocol Security)

*Suite of protocols providing authentication, integrity, and encryption at the IP layer [RFC 4301-4309].*

IPSec operates transparently to applications - no code changes required. Two core protocols:

*1. AH (Authentication Header) [#rfc(4302)]:*
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

*2. ESP (Encapsulating Security Payload) [#rfc(4303)]:*
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

*Protocol for establishing IPSec Security Associations (SAs) [#rfc(7296)].*

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

== SSH Protocol Internals

*SSH (Secure Shell) is a layered protocol stack defined in #rfc(4251)–4254. It provides encrypted remote login, command execution, file transfer, and arbitrary TCP tunneling.*

=== Protocol Stack

```
┌─────────────────────────────────────────────────────┐
│  SSH Connection Protocol  [RFC 4254]                │
│  channels: session / direct-tcpip / forwarded-tcpip │
├─────────────────────────────────────────────────────┤
│  SSH Authentication Protocol  [RFC 4252]            │
│  methods: publickey / password / keyboard-interactive│
├─────────────────────────────────────────────────────┤
│  SSH Transport Layer Protocol  [RFC 4253]           │
│  key exchange, encryption, MAC, compression         │
├─────────────────────────────────────────────────────┤
│  TCP (port 22)                                      │
└─────────────────────────────────────────────────────┘
```

=== Transport Layer: Handshake and Key Exchange

*Full handshake message sequence:*

```
Client                                          Server
  │                                               │
  │──── TCP SYN ────────────────────────────────►│
  │◄─── TCP SYN-ACK ───────────────────────────── │
  │                                               │
  │◄─── SSH-2.0-OpenSSH_9.x\r\n ───────────────── │  version banner
  │──── SSH-2.0-OpenSSH_9.x\r\n ────────────────►│
  │                                               │
  │──── SSH_MSG_KEXINIT (20) ───────────────────►│  algorithm lists
  │◄─── SSH_MSG_KEXINIT (20) ───────────────────── │
  │                                               │
  │  ──── ECDH key exchange (curve25519) ────────  │
  │──── SSH_MSG_KEX_ECDH_INIT (30) ─────────────►│  client ephemeral pub key
  │◄─── SSH_MSG_KEX_ECDH_REPLY (31) ────────────── │  host key + server ephemeral pub key + sig
  │                                               │
  │  both sides derive: K = ECDH(client_priv, server_pub)
  │  session_id = H = hash(V_C || V_S || I_C || I_S || K_S || e || f || K)
  │                                               │
  │──── SSH_MSG_NEWKEYS (21) ───────────────────►│  switch to negotiated cipher
  │◄─── SSH_MSG_NEWKEYS (21) ───────────────────── │
  │                                               │
  │──── SSH_MSG_SERVICE_REQUEST "ssh-userauth" ──►│
  │◄─── SSH_MSG_SERVICE_ACCEPT ─────────────────── │
  │                                               │
  │  [Authentication Protocol begins]             │
```

*Key derivation (from shared secret K and hash H):*
```
IV_c→s  = HASH(K || H || "A" || session_id)
IV_s→c  = HASH(K || H || "B" || session_id)
key_c→s = HASH(K || H || "C" || session_id)   # cipher key, client→server
key_s→c = HASH(K || H || "D" || session_id)
mac_c→s = HASH(K || H || "E" || session_id)
mac_s→c = HASH(K || H || "F" || session_id)
```
Each direction gets independent IV, cipher key, and MAC key; compromise of one direction does not expose the other.

*OpenSSH source:* `kex_derive_keys()` implements #rfc(4253) §7.2 key material expansion: #link("https://github.com/openssh/openssh-portable/blob/master/kex.c")[`kex.c`]

=== Packet Wire Format

Every SSH message after `NEWKEYS` is encrypted:
```
uint32   packet_length    (covers payload + padding, NOT itself)
byte     padding_length
byte[n]  payload          (n = packet_length - padding_length - 1)
byte[p]  random padding   (p = padding_length, min 4, rounds to cipher block)
byte[m]  MAC              (m = mac length, over sequence_number + plaintext)
```

Sequence number is implicit (uint32, wraps at 2³²) and prevents replay within a session. MAC covers the plaintext packet, not the ciphertext (Encrypt-then-MAC in modern ciphers like AES-GCM where AEAD subsumes the MAC field).

*OpenSSH source:* `ssh_packet_send2_wrapped()` for the encryption path: #link("https://github.com/openssh/openssh-portable/blob/master/packet.c")[`packet.c`]

=== Authentication Protocol

Three methods in order of security preference:

*1. publickey (#rfc(4252) §7):*
```
Client sends: SSH_MSG_USERAUTH_REQUEST
  username, service="ssh-connection", method="publickey",
  want_reply=TRUE, algorithm, public_key_blob, signature

Signature over:
  session_id || SSH_MSG_USERAUTH_REQUEST (without sig)

Server verifies against ~/.ssh/authorized_keys
```

*OpenSSH `authorized_keys` line format:*
```
options keytype base64key comment
# example:
restrict,command="rsync --server ..." ssh-ed25519 AAAA... deploy@ci
```
Options: `no-pty`, `no-port-forwarding`, `from="IP"`, `command="..."`, `restrict`

*OpenSSH source:* `auth_key_is_revoked()` + `user_key_allowed2()`: #link("https://github.com/openssh/openssh-portable/blob/master/auth2-pubkey.c")[`openssh-portable/auth2-pubkey.c`]

*2. password:* plaintext password sent inside encrypted channel. Vulnerable to server compromise; avoid in hardened deployments.

*3. keyboard-interactive (#rfc(4256)):* challenge-response, used for TOTP/PAM integration (e.g., Google Authenticator via `libpam-google-authenticator`).

=== Host Key Verification and `known_hosts`

On first connect, client receives server's host public key. Client checks `~/.ssh/known_hosts`:
- If absent: TOFU (Trust On First Use): user prompted, key saved
- If present but changed: *WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED* (potential MITM)
- SSHFP DNS records (#rfc(4255)) provide a CA-independent second channel for verification

```bash
# Add SSHFP record to DNS for verified hosts:
ssh-keygen -r hostname.example.com
# → hostname.example.com IN SSHFP 4 2 <sha256-of-ed25519-key>
```

*`known_hosts` format:*
```
# plain:
github.com ssh-ed25519 AAAA...
# hashed (ssh-keygen -H or HashKnownHosts yes in ssh_config):
|1|base64salt|base64hash ssh-ed25519 AAAA...
```
Hashed form prevents `known_hosts` from leaking server inventory on client compromise.

=== Channel Multiplexing (#rfc(4254))

SSH multiplexes arbitrary bidirectional streams over a single encrypted TCP connection. Each channel has a local and remote number and independent flow-control windows.

```
Client                                          Server
  │──── SSH_MSG_CHANNEL_OPEN "session" ────────►│   open shell/exec channel
  │◄─── SSH_MSG_CHANNEL_OPEN_CONFIRMATION ────── │   remote channel number + window
  │──── SSH_MSG_CHANNEL_REQUEST "exec" ─────────►│   run a command
  │◄─── SSH_MSG_CHANNEL_DATA ─────────────────── │   stdout bytes
  │◄─── SSH_MSG_CHANNEL_EXTENDED_DATA type=1 ─── │   stderr bytes
  │◄─── SSH_MSG_CHANNEL_EOF ─────────────────── │
  │◄─── SSH_MSG_CHANNEL_REQUEST "exit-status" ── │
  │◄─── SSH_MSG_CHANNEL_CLOSE ─────────────────── │
```

*Channel types:*
- `session`: remote shell / exec / subsystem (sftp)
- `direct-tcpip`: local port forward (client→server→destination)
- `forwarded-tcpip`: remote port forward (server→client→destination)
- `tun@openssh.com`: layer-3 VPN tun device

*Flow control:* each side advertises a window (bytes it can receive without acknowledgement). Sender must stop when window is 0. Prevents fast sender from overwhelming slow receiver without per-packet ACKs.

*OpenSSH source:* `channel_output_poll()`, `channel_post_open()`: #link("https://github.com/openssh/openssh-portable/blob/master/channels.c")[`openssh-portable/channels.c`]

=== ssh-agent and Key Forwarding

`ssh-agent` holds decrypted private keys in memory, responds to signing requests over a Unix socket (`SSH_AUTH_SOCK`). Private key material never leaves the agent process.

```
Client process          ssh-agent (separate process)
     │                         │
     │── sign(session_id) ────►│   agent holds ed25519 private key
     │◄── signature ─────────── │   signs, returns bytes
     │                         │
     │── signs SSH auth msg ──► server
```

Agent forwarding (`-A` flag / `ForwardAgent yes`): the remote `sshd` proxies signing requests back through the tunnel to the local agent. Lets you `ssh` from a bastion to internal hosts using your local key.

*Security warning:* forwarding exposes your agent socket on the remote host. Anyone with root on the remote can use your agent. Use `ForwardAgent` only to trusted bastions; prefer `ProxyJump` (`-J`) which does not require agent forwarding.

```bash
# ProxyJump (preferred over ForwardAgent + manual hop):
ssh -J bastion.example.com internal.host

# Equivalent in ~/.ssh/config:
Host internal.host
    ProxyJump bastion.example.com
```

=== Algorithm Negotiation and Modern Defaults

`SSH_MSG_KEXINIT` carries name-lists for: kex, host-key, cipher c→s, cipher s→c, mac c→s, mac s→c, compression. First match wins.

*OpenSSH 9.x defaults (2024):*
```
KEX:       sntrup761x25519-sha512@openssh.com  ← post-quantum hybrid
           curve25519-sha256                   ← standard ECDH
Host key:  ssh-ed25519, ecdsa-sha2-nistp256, rsa-sha2-512
Cipher:    chacha20-poly1305@openssh.com       ← preferred (AEAD, no IV reuse risk)
           aes128-gcm@openssh.com, aes256-gcm@openssh.com
MAC:       (unused when AEAD cipher selected)
```

*sntrup761x25519* is a hybrid KEM: classical X25519 ECDH (Curve25519, non-NIST) combined with NTRU Prime lattice KEM (sntrup761). The post-quantum resistance comes from NTRU Prime; X25519 provides classical fallback so the hybrid is at least as secure as today's X25519 even if the lattice scheme is broken. Mitigates harvest-now-decrypt-later attacks.

*Disable legacy algorithms in `sshd_config`:*
```
KexAlgorithms sntrup761x25519-sha512@openssh.com,curve25519-sha256
HostKeyAlgorithms ssh-ed25519
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com
MACs hmac-sha2-512-etm@openssh.com
```

=== `sshd` Privilege Separation

OpenSSH splits the server into two processes to limit damage from a vulnerability:

```
sshd master (root)
  └── pre-auth Monitor (root, minimal surface)
        └── pre-auth child (unprivileged, sandboxed via seccomp/pledge)
              │  [after auth succeeds]
              └── post-auth child (user UID, handles session)
```

The unprivileged child handles all network I/O during authentication. It communicates with the Monitor only through a socketpair. If an attacker exploits a memory corruption bug in packet parsing, they land in the sandbox, not root.

*OpenSSH source:* #link("https://github.com/openssh/openssh-portable/blob/master/monitor.c")[`openssh-portable/monitor.c`] (Monitor process), #link("https://github.com/openssh/openssh-portable/blob/master/sandbox-seccomp-filter.c")[`openssh-portable/sandbox-seccomp-filter.c`] (Linux seccomp filter)

=== SSH Tunneling and Port Forwarding

*1. Local Port Forwarding (-L):*
```bash
# Forward local:8080 → remote:3306 through SSH
ssh -L 8080:database.internal:3306 user@bastion
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
ssh -R 8080:localhost:3000 user@remote
# Remote users reach your :3000 via remote:8080
```

*3. Dynamic Port Forwarding (-D):*
```bash
ssh -D 1080 user@remote   # SOCKS5 proxy at localhost:1080
```

*4. SSH VPN (tun device):*
```bash
ssh -w 0:0 root@remote    # layer-3 tunnel via tun0
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
  columns: (auto, auto, auto, auto, auto),
  [*Protocol*], [*Throughput*], [*Latency*], [*CPU*], [*Notes*],
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

#rfc(4301): Security Architecture for the Internet Protocol. Kent, S. & Seo, K. (2005).

#rfc(4302): IP Authentication Header. Kent, S. (2005).

#rfc(4303): IP Encapsulating Security Payload (ESP). Kent, S. (2005).

#rfc(7296): Internet Key Exchange Protocol Version 2 (IKEv2). Kaufman, C., et al. (2014).

*WireGuard:*

Donenfeld, J.A. (2017). "WireGuard: Next Generation Kernel Network Tunnel." NDSS '17.

Perrin, T. (2018). "The Noise Protocol Framework." https://noiseprotocol.org/

*Zero Trust:*

NIST SP 800-207: Zero Trust Architecture. Rose, S., et al. (2020).

*SSH:*

#rfc(4251): The Secure Shell (SSH) Protocol Architecture. Ylonen, T. & Lonvick, C. (2006).

#rfc(4252): The Secure Shell (SSH) Authentication Protocol. Ylonen, T. & Lonvick, C. (2006).

#rfc(4253): The Secure Shell (SSH) Transport Layer Protocol. Ylonen, T. & Lonvick, C. (2006).

#rfc(4254): The Secure Shell (SSH) Connection Protocol. Ylonen, T. & Lonvick, C. (2006).

#rfc(4255): Using DNS to Securely Publish Secure Shell (SSH) Key Fingerprints (SSHFP). Schlyter, J. & Griffin, W. (2006).

#rfc(4256): Generic Message Exchange Authentication for SSH (keyboard-interactive). Cusack, F. & Forssen, M. (2006).

Friedl, M., Provos, N., & Simpson, W. (2006). "Diffie-Hellman Group Exchange for the Secure Shell (SSH) Transport Layer Protocol." #rfc(4419).

OpenSSH portable source: #link("https://github.com/openssh/openssh-portable")[`openssh/openssh-portable`]:
- #link("https://github.com/openssh/openssh-portable/blob/master/kex.c")[`kex.c`]: key exchange and session key derivation
- #link("https://github.com/openssh/openssh-portable/blob/master/packet.c")[`packet.c`]: packet encryption and MAC
- #link("https://github.com/openssh/openssh-portable/blob/master/channels.c")[`channels.c`]: channel multiplexing
- #link("https://github.com/openssh/openssh-portable/blob/master/auth2-pubkey.c")[`auth2-pubkey.c`]: public key authentication
- #link("https://github.com/openssh/openssh-portable/blob/master/monitor.c")[`monitor.c`]: privilege separation monitor
- #link("https://github.com/openssh/openssh-portable/blob/master/sandbox-seccomp-filter.c")[`sandbox-seccomp-filter.c`]: Linux seccomp sandbox

Bellovin, S.M. & Blaze, M. (2000). "Cryptographic Modes of Operation for the Internet." NDSS.

Provos, N. & Mazieres, D. (1999). "Preventing Privilege Escalation." USENIX Security.

Bernstein, D.J. et al. (2020). "NTRU Prime: Reducing Attack Surface at Low Cost." CHES 2020. (sntrup761 post-quantum KEM)

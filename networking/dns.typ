= DNS (Domain Name System)

The Domain Name System is a hierarchical, distributed naming system that maps human-readable domain names to IP addresses and other resource records.

*See also:* Application Protocols (for HTTP which depends on DNS), Transport Layer (UDP/TCP transport), Internet Layer (IP addressing)

== DNS Architecture and Hierarchy

*DNS provides [RFC 1034, RFC 1035]:*
- Hierarchical namespace (root → TLD → domain → subdomain)
- Distributed database (no single point of failure)
- Caching at multiple levels (resolver, OS, application)
- UDP transport (port 53) with TCP fallback for large responses

*DNS hierarchy:*

```
                        . (root)
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
      .com               .org              .net        (TLD - Top Level Domain)
        │                  │                  │
   ┌────┴────┐        ┌────┴────┐        ┌────┴────┐
   │         │        │         │        │         │
example   google   wikipedia  apache   cloudflare ...  (Second-level domains)
   │
┌──┴──┐
│     │
www  mail                                              (Subdomains)
```

*13 root server clusters:* A through M (anycast, hundreds of physical servers globally).

== DNS Resolution Process

*Recursive vs iterative queries:*

```
Client                Recursive Resolver        Root NS      .com NS     example.com NS
  │                         │                     │            │              │
  │  "www.example.com A?"  │                     │            │              │
  ├───────────────────────▶│                     │            │              │
  │                         │  "example.com?"    │            │              │
  │                         ├────────────────────▶            │              │
  │                         │  "go ask .com NS"  │            │              │
  │                         │◀────────────────────            │              │
  │                         │                                 │              │
  │                         │       "example.com NS?"        │              │
  │                         ├────────────────────────────────▶│              │
  │                         │       "go ask ns1.example.com" │              │
  │                         │◀────────────────────────────────│              │
  │                         │                                                │
  │                         │              "www.example.com A?"              │
  │                         ├───────────────────────────────────────────────▶│
  │                         │              "93.184.216.34"                   │
  │                         │◀───────────────────────────────────────────────│
  │  "93.184.216.34"       │                                                │
  │◀───────────────────────│                                                │
```

*Latency breakdown (cold cache):*
- Root query: 10-50ms (anycast, geographically distributed)
- TLD query: 10-50ms
- Authoritative query: 10-100ms (depends on server location)
- *Total cold cache:* 50-200ms typical

*Warm cache:* 0-5ms (local resolver cache hit)

== DNS Record Types

*Common record types:*

| Type  | Value | Purpose                                    | Example                              |
|-------|-------|--------------------------------------------|--------------------------------------|
| A     | 1     | IPv4 address                              | example.com → 93.184.216.34         |
| AAAA  | 28    | IPv6 address                              | example.com → 2606:2800:220:1:...   |
| CNAME | 5     | Canonical name (alias)                    | www → example.com                   |
| MX    | 15    | Mail exchanger (priority, host)          | 10 mail.example.com                 |
| NS    | 2     | Name server for zone                      | ns1.example.com                     |
| TXT   | 16    | Arbitrary text (SPF, DKIM, verification) | "v=spf1 include:\_spf.google.com"  |
| SOA   | 6     | Start of authority (zone metadata)        | Serial, refresh, retry, expire      |
| PTR   | 12    | Reverse lookup (IP → name)               | 34.216.184.93.in-addr.arpa → ...   |
| SRV   | 33    | Service location (port, priority, weight) | \_ldap.\_tcp.example.com           |
| CAA   | 257   | Certificate authority authorization       | 0 issue "letsencrypt.org"          |

*CNAME restrictions [RFC 1034]:*
- CNAME cannot coexist with other records at same name
- Cannot be used at zone apex (use ALIAS/ANAME instead, non-standard)

*MX priority:* Lower value = higher priority. Mail servers try lowest first.

== DNS Message Format

*DNS message structure [RFC 1035]:*

```
┌────────────────────────────────────────┐
│             Header (12 bytes)          │
├────────────────────────────────────────┤
│             Question Section           │  Query name, type, class
├────────────────────────────────────────┤
│             Answer Section             │  Resource records
├────────────────────────────────────────┤
│           Authority Section            │  NS records for zone
├────────────────────────────────────────┤
│           Additional Section           │  Glue records (NS IP addresses)
└────────────────────────────────────────┘
```

*Header flags:*
- QR: Query (0) or Response (1)
- OPCODE: Standard query (0), inverse query (1), status (2)
- AA: Authoritative answer
- TC: Truncated (response > 512 bytes, use TCP)
- RD: Recursion desired
- RA: Recursion available
- RCODE: Response code (0=NOERROR, 3=NXDOMAIN, 2=SERVFAIL)

*Size limits:*
- UDP: 512 bytes traditional, 4096 bytes with EDNS0 [RFC 6891]
- TCP: 65535 bytes (2-byte length prefix)

== DNS Caching and TTL

*TTL (Time To Live):* Seconds a record may be cached.

*Typical TTL values:*
- Short (60-300s): Dynamic content, failover scenarios
- Medium (3600s): Standard websites
- Long (86400s+): Stable infrastructure records

*Caching layers:*

```
Application Cache (browser)
        │
        ▼
   OS Resolver Cache (nscd, systemd-resolved)
        │
        ▼
   Local Recursive Resolver (router, ISP)
        │
        ▼
   Public Recursive Resolver (8.8.8.8, 1.1.1.1)
        │
        ▼
   Authoritative Server (source of truth)
```

*Negative caching [RFC 2308]:*
- NXDOMAIN responses cached based on SOA minimum TTL
- Prevents repeated queries for non-existent domains
- Typical negative TTL: 300-3600s

*Cache poisoning risk:* Low TTL = more queries = more attack surface. Balance security vs performance.

== DNS over HTTPS (DoH) and DNS over TLS (DoT)

*Traditional DNS:* Plaintext UDP, no authentication. Vulnerable to eavesdropping and manipulation.

*DoT (DNS over TLS) [RFC 7858]:*
- Port 853
- TLS 1.2+ encryption
- Server authentication via certificates
- *Latency overhead:* +1-2 RTT for TLS handshake (first query)

*DoH (DNS over HTTPS) [RFC 8484]:*
- Port 443 (same as HTTPS)
- HTTP/2 or HTTP/3 transport
- Harder to block (indistinguishable from web traffic)
- *Latency overhead:* Similar to DoT, benefits from HTTP/2 connection reuse

*Performance comparison:*

| Protocol | First Query | Subsequent Queries | Privacy |
|----------|-------------|-------------------|---------|
| UDP      | 10-50ms     | 10-50ms           | None    |
| DoT      | 50-150ms    | 10-50ms           | High    |
| DoH      | 50-150ms    | 10-50ms           | High    |

*Public resolvers supporting DoH/DoT:*
- Cloudflare: 1.1.1.1 (https://cloudflare-dns.com/dns-query)
- Google: 8.8.8.8 (https://dns.google/dns-query)
- Quad9: 9.9.9.9 (https://dns.quad9.net/dns-query)

== DNSSEC (DNS Security Extensions)

*Problem:* DNS responses can be forged (cache poisoning, man-in-the-middle).

*DNSSEC provides [RFC 4033, 4034, 4035]:*
- Data origin authentication (cryptographic signatures)
- Data integrity verification
- Authenticated denial of existence (NSEC/NSEC3)

*DNSSEC record types:*
- *RRSIG:* Signature over resource record set
- *DNSKEY:* Public key for zone
- *DS:* Delegation signer (hash of child DNSKEY, in parent zone)
- *NSEC/NSEC3:* Proof that name does not exist

*Chain of trust:*

```
Root Zone (trust anchor)
    │
    │  DS record for .com
    ▼
.com Zone
    │  DNSKEY + RRSIG
    │  DS record for example.com
    ▼
example.com Zone
    │  DNSKEY + RRSIG
    │  A record + RRSIG
    ▼
Validator verifies all signatures up to root
```

*Performance impact:*
- Larger responses: +500-2000 bytes for signatures
- Validation CPU cost: ~0.1-1ms per query
- Additional queries for DNSKEY/DS records (usually cached)

*Adoption:* ~30% of domains signed, ~25% of resolvers validate [APNIC 2023].

== DNS Performance Optimization

*Resolver selection matters:*

| Resolver Type        | Typical Latency | Cache Hit Rate |
|---------------------|-----------------|----------------|
| ISP default         | 20-100ms        | Medium         |
| Public (1.1.1.1)    | 10-30ms         | High           |
| Local (Pi-hole)     | 1-5ms           | Configurable   |

*Optimization techniques:*

1. *Prefetching:* Resolve domains before user clicks
   ```html
   <link rel="dns-prefetch" href="//cdn.example.com">
   ```

2. *Connection reuse:* HTTP/2 multiplexing reduces DNS lookups per page

3. *Minimize unique domains:* Each domain = 1 DNS lookup

4. *Appropriate TTLs:* Longer TTL = fewer queries (but slower failover)

*DNS lookup in page load:*
- Average webpage: 10-20 unique domains (CDN, analytics, ads)
- Cold cache penalty: 10-20 \* 50ms = 500-1000ms
- Warm cache: \<50ms total

== DNS Security Concerns

*Common attacks:*

1. *DNS spoofing/cache poisoning:*
   - Attacker injects false records into resolver cache
   - Mitigated by: DNSSEC, randomized source ports, 0x20 encoding

2. *DNS amplification DDoS:*
   - Small query → large response (amplification factor 50-100x)
   - ANY queries especially dangerous (now deprecated [RFC 8482])
   - Mitigated by: Response rate limiting, BCP38 (source address validation)

3. *DNS tunneling:*
   - Encodes data in DNS queries/responses
   - Bypasses firewalls (DNS usually allowed)
   - Detection: Query length analysis, entropy measurement

4. *DNS hijacking:*
   - Attacker compromises authoritative server or registrar
   - Mitigated by: Registry lock, DNSSEC, certificate transparency

*Best practices:*
- Use DNSSEC validation
- Enable DoH/DoT for privacy
- Monitor for DNS anomalies
- Use reputable resolvers

== References

*Primary sources:*

RFC 1034: Domain Names - Concepts and Facilities. Mockapetris, P. (1987).

RFC 1035: Domain Names - Implementation and Specification. Mockapetris, P. (1987).

RFC 2308: Negative Caching of DNS Queries. Andrews, M. (1998).

RFC 4033: DNS Security Introduction and Requirements. Arends, R. et al. (2005).

RFC 4034: Resource Records for DNS Security Extensions. Arends, R. et al. (2005).

RFC 4035: Protocol Modifications for DNS Security Extensions. Arends, R. et al. (2005).

RFC 6891: Extension Mechanisms for DNS (EDNS(0)). Damas, J., Graff, M., & Vixie, P. (2013).

RFC 7858: Specification for DNS over Transport Layer Security (TLS). Hu, Z. et al. (2016).

RFC 8484: DNS Queries over HTTPS (DoH). Hoffman, P. & McManus, P. (2018).

RFC 8482: Providing Minimal-Sized Responses to DNS Queries That Have QTYPE=ANY. Abley, J. et al. (2019).

Mockapetris, P. & Dunlap, K. (1988). "Development of the Domain Name System." SIGCOMM '88.

Liu, C. & Albitz, P. (2006). DNS and BIND, 5th Edition. O'Reilly Media.

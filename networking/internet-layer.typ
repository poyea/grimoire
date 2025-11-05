= Internet Layer

The Internet layer provides host-to-host routing across networks using IP addresses.

*See also:* Link Layer (for MAC-to-IP mapping), Transport Layer (for end-to-end communication)

== IPv4 Addressing

*Address format:* 32-bit, dot-decimal notation (e.g., 192.168.1.1).

*Classes (historical):*
- Class A: 0.0.0.0 - 127.255.255.255 (16M hosts per network)
- Class B: 128.0.0.0 - 191.255.255.255 (65K hosts per network)
- Class C: 192.0.0.0 - 223.255.255.255 (256 hosts per network)

*CIDR (Classless Inter-Domain Routing) [RFC 4632]:* Variable-length subnet masks.
- Example: 192.168.1.0/24 = 192.168.1.0 - 192.168.1.255 (256 addresses)
- Subnet mask: 255.255.255.0 (24 bits of network, 8 bits of host)

*Private address ranges [RFC 1918]:*
- 10.0.0.0/8 (10.0.0.0 - 10.255.255.255)
- 172.16.0.0/12 (172.16.0.0 - 172.31.255.255)
- 192.168.0.0/16 (192.168.0.0 - 192.168.255.255)

== IPv4 Packet Structure

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
├───┬───────┬─────────────────────────────────────────────────────┤
│Ver│  IHL  │      ToS      │          Total Length               │
├───┴───────┴───────────────┴─────────────────────────────────────┤
│         Identification    │Flags│      Fragment Offset          │
├───────────────────────────┴─────┴───────────────────────────────┤
│  Time to Live │  Protocol │         Header Checksum             │
├───────────────┴───────────┴─────────────────────────────────────┤
│                       Source Address                            │
├─────────────────────────────────────────────────────────────────┤
│                    Destination Address                          │
├─────────────────────────────────────────────────────────────────┤
│                    Options (if IHL > 5)                         │
└─────────────────────────────────────────────────────────────────┘
```

*Minimum header:* 20 bytes.

*Key fields:*
- *TTL (Time To Live):* Hop count, decremented at each router. Prevents routing loops.
- *Protocol:* 6 = TCP, 17 = UDP, 1 = ICMP
- *Checksum:* Header only (not payload)

== Routing

*Routing table lookup:* Longest prefix match.

```bash
# View routing table
ip route show

# Example:
default via 192.168.1.1 dev eth0      # Gateway
192.168.1.0/24 dev eth0 scope link    # Local subnet
```

*Routing decision:* Match destination IP against prefixes, select longest match.

== IPv6

*Address format:* 128-bit, hex notation (e.g., 2001:0db8:0000:0000:0000:8a2e:0370:7334).

*Simplifications:*
- Omit leading zeros: 2001:db8:0:0:0:8a2e:370:7334
- Replace consecutive zeros: 2001:db8::8a2e:370:7334

*Advantages:*
- Larger address space (2^128 vs 2^32)
- No NAT required
- Simpler header (fewer fields)
- Built-in IPsec

*Adoption:* ~40% of internet traffic as of 2023 [Google IPv6 statistics].

== References

RFC 791: Internet Protocol. Postel, J. (1981).

RFC 4632: Classless Inter-domain Routing (CIDR). Fuller, V. & Li, T. (2006).

RFC 8200: Internet Protocol, Version 6 (IPv6) Specification. Deering, S. & Hinden, R. (2017).

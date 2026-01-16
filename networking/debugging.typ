= Network Debugging

Systematic approaches to diagnosing network issues: packet capture, socket inspection, latency analysis, and advanced tracing.

*See also:* Transport Layer (for TCP/UDP internals), Sockets API (for socket programming), Kernel Bypass (for XDP/BPF context), Performance Reference (for baseline latency numbers)

== tcpdump Essentials

*tcpdump* is the standard command-line packet analyzer. Runs in kernel space via libpcap, captures packets before/after kernel processing.

=== Basic Capture

```bash
# Capture all traffic on eth0
tcpdump -i eth0

# Capture with verbose output (-v, -vv, -vvv for increasing verbosity)
tcpdump -i eth0 -vvv

# Capture without DNS resolution (-n) and port resolution (-nn)
tcpdump -i eth0 -nn

# Capture specific count of packets
tcpdump -i eth0 -c 100

# Capture and save to file (pcap format)
tcpdump -i eth0 -w capture.pcap

# Read from saved capture
tcpdump -r capture.pcap
```

=== Capture Filters (BPF Syntax)

Capture filters use Berkeley Packet Filter (BPF) syntax, applied in kernel before packets reach userspace. More efficient than display filters.

```bash
# Filter by host
tcpdump -i eth0 host 192.168.1.100
tcpdump -i eth0 src host 192.168.1.100
tcpdump -i eth0 dst host 192.168.1.100

# Filter by network
tcpdump -i eth0 net 192.168.1.0/24

# Filter by port
tcpdump -i eth0 port 80
tcpdump -i eth0 src port 443
tcpdump -i eth0 dst port 22
tcpdump -i eth0 portrange 8000-9000

# Filter by protocol
tcpdump -i eth0 tcp
tcpdump -i eth0 udp
tcpdump -i eth0 icmp
tcpdump -i eth0 arp

# Compound filters (and, or, not)
tcpdump -i eth0 'host 192.168.1.100 and port 80'
tcpdump -i eth0 'tcp and (port 80 or port 443)'
tcpdump -i eth0 'not port 22'  # Exclude SSH

# TCP flags
tcpdump -i eth0 'tcp[tcpflags] & tcp-syn != 0'      # SYN packets
tcpdump -i eth0 'tcp[tcpflags] & tcp-rst != 0'      # RST packets
tcpdump -i eth0 'tcp[tcpflags] == tcp-syn'          # SYN only (no ACK)
tcpdump -i eth0 'tcp[tcpflags] & (tcp-syn|tcp-fin) != 0'  # SYN or FIN
```

=== Advanced tcpdump Options

```bash
# Capture full packet (-s 0 = snaplen unlimited)
tcpdump -i eth0 -s 0 -w full_capture.pcap

# Print packet contents in ASCII
tcpdump -i eth0 -A

# Print packet contents in hex and ASCII
tcpdump -i eth0 -X

# Rotate capture files (100MB each, keep 10 files)
tcpdump -i eth0 -w capture.pcap -C 100 -W 10

# Add timestamps (microsecond precision)
tcpdump -i eth0 -tttt

# Buffer size (reduce packet drops under load)
tcpdump -i eth0 -B 4096  # 4MB buffer
```

*Performance note:* tcpdump adds ~5-10us latency per packet. For high-speed capture (>1Gbps), consider DPDK-based tools or hardware TAPs.

== Wireshark Filters

Wireshark provides GUI-based analysis with powerful display filters. Capture filters (BPF) apply during capture; display filters apply post-capture.

=== Display Filters vs Capture Filters

*Capture filters (BPF):*
- Applied at capture time in kernel
- Cannot be changed without restarting capture
- Use primitive syntax: `host`, `port`, `tcp`, `and`, `or`

*Display filters:*
- Applied post-capture in Wireshark GUI
- Can be changed dynamically
- Rich protocol-aware syntax: `tcp.port`, `http.request`, `dns.qry.name`

=== Common Display Filters

```
# IP address filtering
ip.addr == 192.168.1.100
ip.src == 10.0.0.1
ip.dst == 10.0.0.2

# Port filtering
tcp.port == 80
tcp.srcport == 443
udp.dstport == 53

# Protocol filtering
http
dns
tls
tcp
udp

# TCP analysis
tcp.flags.syn == 1                    # SYN packets
tcp.flags.reset == 1                  # RST packets
tcp.analysis.retransmission           # Retransmissions
tcp.analysis.duplicate_ack            # Duplicate ACKs
tcp.analysis.zero_window              # Zero window events
tcp.analysis.window_update            # Window updates

# HTTP filtering
http.request.method == "GET"
http.request.uri contains "/api"
http.response.code == 500
http.host == "example.com"

# DNS filtering
dns.qry.name == "example.com"
dns.qry.type == 1                     # A record
dns.flags.response == 1               # Responses only

# TLS/SSL filtering
tls.handshake.type == 1               # Client Hello
tls.handshake.type == 2               # Server Hello
tls.record.version == 0x0303          # TLS 1.2
ssl.alert_message                     # TLS alerts

# Compound filters
tcp.port == 80 and ip.src == 10.0.0.1
http or dns
not arp and not icmp
```

=== Wireshark Statistics

*Useful analysis views:*
- *Statistics > Conversations:* Top talkers by bytes/packets
- *Statistics > Protocol Hierarchy:* Protocol distribution
- *Statistics > IO Graph:* Throughput over time
- *Analyze > Expert Information:* Anomalies and warnings
- *Statistics > TCP Stream Graphs:* RTT, throughput, window scaling

== Socket Inspection (ss and netstat)

=== ss (Socket Statistics)

Modern replacement for netstat. Faster, more information, direct kernel access via netlink.

```bash
# List all TCP connections
ss -t

# List all listening sockets
ss -l

# Show process info (-p requires root)
ss -tlp

# Numeric output (no DNS resolution)
ss -tln

# Show timer information
ss -to

# Extended information (memory, congestion)
ss -tie

# Filter by state
ss -t state established
ss -t state time-wait
ss -t state syn-sent

# Filter by address/port
ss -t dst 192.168.1.100
ss -t src :80
ss -t 'dport == :443'
ss -t '( dport == :80 or sport == :80 )'

# Show socket memory usage
ss -tm

# Show TCP internal info (congestion window, RTT)
ss -ti
```

*Example output interpretation:*
```
$ ss -ti dst :443
State   Recv-Q  Send-Q  Local Address:Port  Peer Address:Port
ESTAB   0       0       10.0.0.5:54321      93.184.216.34:443
         cubic wscale:7,7 rto:204 rtt:25.5/12.7 ato:40 mss:1448
         pmtu:1500 rcvmss:1448 advmss:1448 cwnd:10 bytes_sent:1024
         bytes_acked:1024 bytes_received:4096 segs_out:15 segs_in:20
```

*Key metrics:*
- `cwnd`: Congestion window (segments)
- `rtt`: Round-trip time / variance (ms)
- `rto`: Retransmission timeout (ms)
- `mss`: Maximum segment size
- `wscale`: Window scale factors (send,recv)

=== netstat (Legacy)

```bash
# TCP connections with process info
netstat -tlnp

# All connections with state
netstat -an

# Statistics by protocol
netstat -s

# Routing table
netstat -r
```

*Note:* Prefer `ss` on modern Linux; netstat reads `/proc` which is slower.

== Latency and Connection Debugging

=== ping and mtr

```bash
# Basic ICMP ping
ping -c 10 192.168.1.1

# Specify interval (requires root for <0.2s)
ping -i 0.1 192.168.1.1

# TCP ping (when ICMP blocked)
# Using hping3
hping3 -S -p 443 -c 10 example.com

# mtr - combines ping and traceroute
mtr --report example.com
mtr -T -P 443 example.com  # TCP mode, port 443
```

=== traceroute Variants

```bash
# ICMP traceroute
traceroute example.com

# TCP traceroute (bypasses ICMP filters)
traceroute -T -p 443 example.com

# UDP traceroute
traceroute -U example.com

# Paris traceroute (consistent path discovery)
paris-traceroute example.com
```

=== TCP Connection Timing

```bash
# Measure TCP handshake time with curl
curl -w "DNS: %{time_namelookup}s\nConnect: %{time_connect}s\nTLS: %{time_appconnect}s\nTotal: %{time_total}s\n" -o /dev/null -s https://example.com

# Continuous monitoring with httping
httping -c 100 -s https://example.com
```

== Network Stack Profiling (perf)

*perf* profiles kernel and userspace code. Useful for identifying CPU bottlenecks in network path.

```bash
# Profile network syscalls
perf record -e 'syscalls:sys_enter_send*' -e 'syscalls:sys_enter_recv*' -a -- sleep 10
perf report

# Profile TCP stack functions
perf record -g -a -e 'net:*' -- sleep 10
perf report --call-graph

# Trace packet receive path
perf trace -e 'net:netif_receive_skb' -a -- sleep 5

# CPU cycles in network functions
perf top -e cycles -g --call-graph dwarf

# Socket buffer allocation
perf stat -e 'kmem:kmalloc' -a -- sleep 10
```

*Key events for networking:*
- `net:netif_receive_skb` - Packet received by kernel
- `net:net_dev_xmit` - Packet transmitted
- `tcp:tcp_probe` - TCP congestion events
- `sock:inet_sock_set_state` - Socket state changes

== eBPF/bpftrace for Advanced Tracing

eBPF allows custom tracing programs in kernel without modules. bpftrace provides high-level scripting.

=== bpftrace One-Liners

```bash
# Count syscalls by process
bpftrace -e 'tracepoint:syscalls:sys_enter_* { @[comm] = count(); }'

# Trace TCP connections
bpftrace -e 'kprobe:tcp_connect { printf("%s -> %s\n", comm, str(arg0)); }'

# TCP retransmit tracing
bpftrace -e 'kprobe:tcp_retransmit_skb { @[comm] = count(); }'

# Socket accept latency histogram (us)
bpftrace -e 'kprobe:inet_csk_accept { @start[tid] = nsecs; }
             kretprobe:inet_csk_accept /@start[tid]/ {
               @us = hist((nsecs - @start[tid]) / 1000);
               delete(@start[tid]);
             }'

# Trace DNS queries (port 53)
bpftrace -e 'kprobe:udp_sendmsg /arg2 == 53/ { printf("%s DNS query\n", comm); }'

# Network device queue length
bpftrace -e 'tracepoint:net:net_dev_xmit { @qlen[str(args->name)] = hist(args->len); }'
```

=== BCC Tools

BCC (BPF Compiler Collection) provides pre-built tools:

```bash
# TCP connection latency
tcpconnlat

# TCP retransmissions
tcpretrans

# TCP connection tracing (connect/accept/close)
tcptracer

# Socket statistics by process
sockstat

# Network throughput by process
nethogs

# Trace TCP state changes
tcpstates
```

== Common Debugging Scenarios

=== Connection Refused

*Symptoms:* `connect()` returns ECONNREFUSED.

*Diagnosis:*
```bash
# Check if port is listening
ss -tln | grep :8080

# Check firewall rules
iptables -L -n -v
nft list ruleset

# Trace connection attempt
tcpdump -i any port 8080 -nn
```

*Common causes:* Service not running, wrong port, firewall blocking.

=== Connection Timeout

*Symptoms:* `connect()` hangs, eventually returns ETIMEDOUT.

*Diagnosis:*
```bash
# Check route to host
traceroute -T -p 443 target.com

# Check for SYN packets leaving
tcpdump -i eth0 'tcp[tcpflags] == tcp-syn and dst host target.com'

# Check for SYN-ACK response
tcpdump -i eth0 'tcp[tcpflags] & (tcp-syn|tcp-ack) == (tcp-syn|tcp-ack) and src host target.com'
```

*Common causes:* Firewall dropping packets silently, routing issue, target host unreachable.

=== High Latency

*Symptoms:* Slow response times, high RTT.

*Diagnosis:*
```bash
# Measure RTT along path
mtr --report target.com

# Check TCP RTT for existing connections
ss -ti dst target.com | grep rtt

# Profile kernel time spent in networking
perf record -g -e 'net:*' -- sleep 10

# Check for retransmissions (indicates packet loss)
ss -ti | grep retrans
netstat -s | grep retransmit
```

*Common causes:* Geographic distance, congestion, packet loss causing retransmissions.

=== Packet Loss

*Symptoms:* Missing responses, retransmissions, degraded throughput.

*Diagnosis:*
```bash
# Interface statistics
ip -s link show eth0
ethtool -S eth0 | grep -i drop
ethtool -S eth0 | grep -i error

# Kernel network statistics
netstat -s | grep -i drop
netstat -s | grep -i error
cat /proc/net/softnet_stat

# Ring buffer overruns
ethtool -g eth0  # Show ring size
dmesg | grep -i "ring buffer"
```

*Common causes:* Ring buffer overflow, CPU saturation, NIC errors, network congestion.

=== TCP Reset (RST) Debugging

*Symptoms:* Connections terminated unexpectedly with RST.

*Diagnosis:*
```bash
# Capture RST packets
tcpdump -i eth0 'tcp[tcpflags] & tcp-rst != 0' -nn

# Check for half-open connections
ss -t state syn-recv

# Firewall connection tracking
conntrack -L | grep target
```

*Common causes:* Firewall timeout, application crash, load balancer health check failure, TCP keepalive timeout.

== Performance Checklist

*Before debugging:*
1. Baseline: Know normal latency/throughput for your environment
2. Isolate: Is it client, server, or network?
3. Reproduce: Can you trigger the issue consistently?

*Quick checks:*
```bash
# NIC link status and errors
ethtool eth0
ip -s link show eth0

# Socket buffer sizes
sysctl net.core.rmem_max
sysctl net.core.wmem_max
sysctl net.ipv4.tcp_rmem
sysctl net.ipv4.tcp_wmem

# Connection tracking table
cat /proc/sys/net/netfilter/nf_conntrack_count
cat /proc/sys/net/netfilter/nf_conntrack_max

# TCP statistics
netstat -s | head -50
ss -s
```

== References

*Primary sources:*

McCanne, S. & Jacobson, V. (1993). "The BSD Packet Filter: A New Architecture for User-level Packet Capture." USENIX Winter.

Gregg, B. (2019). BPF Performance Tools: Linux System and Application Observability. Addison-Wesley.

Wireshark Foundation (2025). Wireshark User's Guide. https://www.wireshark.org/docs/

tcpdump/libpcap maintainers. tcpdump man page. https://www.tcpdump.org/manpages/

Gregg, B. (2025). bpftrace Reference Guide. https://github.com/iovisor/bpftrace

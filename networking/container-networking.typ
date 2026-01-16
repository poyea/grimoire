= Container Networking

Containers rely on Linux kernel primitives to create isolated network environments. This section covers the building blocks (namespaces, veth, bridges) and orchestration layers (Docker, Kubernetes, CNI).

*See also:* Link Layer (for bridge/veth internals), Internet Layer (for IP routing), Kernel Bypass (for high-performance container networking)

== Linux Network Namespaces

*Network namespace:* Isolated network stack with its own interfaces, routing tables, iptables rules, and sockets [man 7 network_namespaces].

```bash
# Create namespace
ip netns add container1

# List namespaces
ip netns list

# Execute command in namespace
ip netns exec container1 ip link show
# Output: Only loopback interface (lo)

# Each namespace has:
# - Separate routing table
# - Separate iptables rules
# - Separate /proc/net
# - Separate socket namespace
```

*Namespace isolation:*

```
┌─────────────────────────────────────────────────────────────────┐
│                        Host Network Stack                        │
│  eth0: 10.0.0.1/24    docker0: 172.17.0.1/16                   │
│  Routing table: default via 10.0.0.254                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────┐    ┌─────────────────────────┐
│  Container1 Namespace    │    │  Container2 Namespace    │
│  eth0: 172.17.0.2/16    │    │  eth0: 172.17.0.3/16    │
│  lo: 127.0.0.1          │    │  lo: 127.0.0.1          │
│  Gateway: 172.17.0.1    │    │  Gateway: 172.17.0.1    │
└─────────────────────────┘    └─────────────────────────┘
```

*Performance:* Namespace operations are fast (~1μs for context switch between namespaces). No packet copying overhead for isolation itself.

== veth Pairs and Bridges

*veth (Virtual Ethernet):* Paired virtual NICs - packets sent to one end appear at the other [man 4 veth].

```bash
# Create veth pair
ip link add veth0 type veth peer name veth1

# Move one end into container namespace
ip link set veth1 netns container1

# Configure host side
ip addr add 172.17.0.1/16 dev veth0
ip link set veth0 up

# Configure container side
ip netns exec container1 ip addr add 172.17.0.2/16 dev veth1
ip netns exec container1 ip link set veth1 up
ip netns exec container1 ip link set lo up
```

*Linux bridge:* Software switch connecting multiple veth pairs.

```
┌──────────────────────────────────────────────────────────────┐
│                         Host                                  │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    docker0 (bridge)                     │ │
│  │                     172.17.0.1/16                       │ │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐               │ │
│  │   │ vethA   │  │ vethB   │  │ vethC   │               │ │
│  └───┼─────────┼──┼─────────┼──┼─────────┼───────────────┘ │
│      │         │  │         │  │         │                  │
└──────┼─────────┼──┼─────────┼──┼─────────┼──────────────────┘
       │         │  │         │  │         │
┌──────┴───┐ ┌───┴──────┐ ┌───┴──────┐
│Container1│ │Container2│ │Container3│
│172.17.0.2│ │172.17.0.3│ │172.17.0.4│
└──────────┘ └──────────┘ └──────────┘
```

*Bridge creation:*
```bash
# Create bridge
ip link add docker0 type bridge
ip addr add 172.17.0.1/16 dev docker0
ip link set docker0 up

# Attach veth to bridge
ip link set veth0 master docker0

# Enable IP forwarding
echo 1 > /proc/sys/net/ipv4/ip_forward

# NAT for outbound traffic
iptables -t nat -A POSTROUTING -s 172.17.0.0/16 ! -o docker0 -j MASQUERADE
```

*Performance [Suo et al. 2018]:*
- veth pair latency: ~5-10μs additional vs native
- Bridge forwarding: ~2-5μs per hop
- Throughput: ~90-95% of native for large packets

== Docker Networking Modes

Docker provides four network drivers, each with different isolation and performance characteristics.

*1. Bridge mode (default):*

```bash
docker run --network bridge nginx  # Default
# Container gets IP from docker0 subnet (172.17.0.0/16)
# NAT for external access via iptables MASQUERADE
```

*Architecture:* Container → veth → docker0 bridge → iptables NAT → eth0

*2. Host mode:*

```bash
docker run --network host nginx
# Container shares host network namespace
# No isolation, no NAT overhead
# Port conflicts possible
```

*Performance:* Native speed (no veth/bridge overhead). Use for latency-sensitive applications.

*3. None mode:*

```bash
docker run --network none alpine
# Only loopback interface
# Complete network isolation
# Must configure manually if networking needed
```

*4. Overlay mode:*

```bash
docker network create --driver overlay my_overlay
docker run --network my_overlay nginx
# Multi-host networking via VXLAN encapsulation
# Used by Docker Swarm
```

*Performance comparison:*

#table(
  columns: 4,
  align: (left, right, right, left),
  table.header([Mode], [Latency], [Throughput], [Use Case]),
  [Host], [Native], [100%], [Maximum performance, trusted workloads],
  [Bridge], [+5-15μs], [90-95%], [Single-host isolation (default)],
  [Overlay], [+20-50μs], [70-85%], [Multi-host, Swarm/Compose],
  [None], [N/A], [N/A], [Security isolation, manual config],
)

== Kubernetes Networking Model

Kubernetes defines a flat network model with three requirements [Kubernetes Networking SIG]:

1. *Pod-to-Pod:* All pods can communicate without NAT
2. *Pod-to-Service:* Services provide stable endpoints
3. *External-to-Service:* External traffic routed to services

*Pod networking:*

```
┌──────────────────────────────────────────────────────────────────┐
│                              Node                                 │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      Pod (Network Namespace)                 ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        ││
│  │  │ Container A │  │ Container B │  │ Container C │        ││
│  │  │   :8080     │  │   :3000     │  │   :5432     │        ││
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        ││
│  │         │                │                │                ││
│  │         └────────────────┼────────────────┘                ││
│  │                          │                                  ││
│  │                    eth0 (10.244.1.5)                       ││
│  └──────────────────────────┼──────────────────────────────────┘│
│                             │ veth                              │
│  ┌──────────────────────────┴───────────────────────────────┐  │
│  │                     cbr0 (bridge) / CNI                   │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

*Key insight:* Containers in same pod share network namespace (communicate via localhost).

*Service types:*

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-service
spec:
  type: ClusterIP  # Internal only (default)
  # type: NodePort   # Expose on node port 30000-32767
  # type: LoadBalancer  # Cloud provider LB
  selector:
    app: my-app
  ports:
    - port: 80
      targetPort: 8080
```

*kube-proxy modes:*

- *iptables (default):* O(n) rule matching, ~10K services practical limit
- *IPVS:* O(1) hash-based lookup, scales to 100K+ services [Kubernetes docs]
- *eBPF (Cilium):* Kernel bypass for service routing, lowest latency

== CNI (Container Network Interface)

*CNI:* Specification for container runtime to configure networking [containernetworking/cni].

*Plugin execution:*

```
┌─────────────────┐    ┌────────────┐    ┌─────────────────┐
│  Container      │───▶│  kubelet   │───▶│  CNI Plugin     │
│  Runtime        │    │            │    │  (bridge/calico)│
│  (containerd)   │◀───│            │◀───│                 │
└─────────────────┘    └────────────┘    └─────────────────┘
                              │
                              ▼
                       /etc/cni/net.d/
                       10-bridge.conflist
```

*CNI configuration:*

```json
{
  "cniVersion": "1.0.0",
  "name": "my-network",
  "plugins": [
    {
      "type": "bridge",
      "bridge": "cni0",
      "isGateway": true,
      "ipMasq": true,
      "ipam": {
        "type": "host-local",
        "subnet": "10.244.0.0/16"
      }
    },
    {
      "type": "portmap",
      "capabilities": {"portMappings": true}
    }
  ]
}
```

*Popular CNI plugins:*

#table(
  columns: 4,
  align: (left, left, left, left),
  table.header([Plugin], [Encapsulation], [Network Policy], [Performance]),
  [Flannel], [VXLAN/host-gw], [No (basic)], [Good (host-gw native)],
  [Calico], [VXLAN/BGP/eBPF], [Yes (rich)], [Excellent (eBPF mode)],
  [Cilium], [VXLAN/eBPF], [Yes (L7 aware)], [Excellent (eBPF)],
  [Weave], [VXLAN/sleeve], [Yes], [Moderate],
  [Canal], [Flannel+Calico], [Yes], [Good],
)

== Overlay Networks

*Problem:* Pod IPs must be routable across nodes. Solutions: L2 extension, L3 routing, or encapsulation.

*VXLAN (Virtual Extensible LAN) [RFC 7348]:*

```
┌────────────────────────────────────────────────────────────────┐
│                    Original Ethernet Frame                      │
│  ┌──────────┬──────────┬──────┬────────────────────┐          │
│  │ Dst MAC  │ Src MAC  │ Type │      IP Packet      │          │
│  └──────────┴──────────┴──────┴────────────────────┘          │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼ VXLAN Encapsulation
┌────────────────────────────────────────────────────────────────┐
│ Outer      │ Outer  │ UDP    │ VXLAN  │   Inner Ethernet       │
│ Ethernet   │ IP     │ (4789) │ Header │   Frame (original)     │
│ (14B)      │ (20B)  │ (8B)   │ (8B)   │                        │
└────────────────────────────────────────────────────────────────┘
  50 bytes overhead
```

*VXLAN operation:*

```
Node A (10.0.0.1)                     Node B (10.0.0.2)
┌──────────────┐                      ┌──────────────┐
│ Pod1         │                      │ Pod2         │
│ 10.244.1.5   │─────────────────────▶│ 10.244.2.7   │
└──────────────┘                      └──────────────┘
       │                                     ▲
       ▼                                     │
  ┌─────────┐    VXLAN Tunnel (UDP 4789)  ┌─────────┐
  │  VTEP   │═══════════════════════════▶│  VTEP   │
  │ flannel │    Outer: 10.0.0.1→10.0.0.2│ flannel │
  └─────────┘    Inner: 10.244.1.5→2.7   └─────────┘
```

*Flannel backends:*

- *VXLAN:* Encapsulation, works anywhere, ~50B overhead
- *host-gw:* Direct L3 routing, requires L2 adjacency, native performance
- *WireGuard:* Encrypted tunnel, ~60B overhead + crypto cost

*Calico modes:*

- *VXLAN:* Like Flannel, for cloud environments
- *IP-in-IP:* Lighter encapsulation (20B overhead)
- *BGP:* No encapsulation, native routing, datacenter use
- *eBPF:* Bypass iptables, highest performance

== Performance Overhead

*Latency overhead [Amaral et al. 2021, Suo et al. 2018]:*

#table(
  columns: 3,
  align: (left, right, left),
  table.header([Configuration], [Added Latency], [Notes]),
  [Native (host network)], [0], [Baseline],
  [Bridge (Docker default)], [5-15μs], [veth + bridge + iptables],
  [Calico BGP], [2-5μs], [Native routing, no encap],
  [Calico eBPF], [1-3μs], [Bypass iptables],
  [Flannel host-gw], [5-10μs], [Direct routing, L2 required],
  [Flannel VXLAN], [15-30μs], [Encap/decap overhead],
  [Weave (encrypted)], [30-50μs], [Encryption cost],
)

*Throughput impact:*

```
Benchmark: iperf3, 10Gbps NIC, TCP, jumbo frames disabled

Native:           9.4 Gbps  (100%)
Host network:     9.4 Gbps  (100%)
Calico eBPF:      9.2 Gbps  (98%)
Calico BGP:       9.0 Gbps  (96%)
Bridge (Docker):  8.8 Gbps  (94%)
Flannel VXLAN:    7.5 Gbps  (80%)
Weave encrypted:  5.0 Gbps  (53%)
```

*CPU overhead:* VXLAN encapsulation costs ~1000-2000 cycles per packet. At 1M PPS, this consumes ~1 CPU core.

== Optimization Techniques

*1. Use host network for latency-critical pods:*

```yaml
apiVersion: v1
kind: Pod
spec:
  hostNetwork: true  # Share host network namespace
```

*2. Enable eBPF-based CNI (Cilium):*

```bash
# Replace kube-proxy with Cilium
helm install cilium cilium/cilium --set kubeProxyReplacement=true
# Benefit: iptables bypass, ~50% latency reduction for services
```

*3. Use IPVS for kube-proxy:*

```bash
# In kube-proxy config
mode: ipvs  # vs iptables
# Benefit: O(1) service lookup, scales to 100K services
```

*4. MTU tuning for overlays:*

```bash
# Account for encapsulation overhead
# VXLAN: 50 bytes, IP-in-IP: 20 bytes
# If host MTU = 1500, container MTU = 1450 (VXLAN)
# Or use jumbo frames: host MTU = 9000, container = 8950
```

*5. Disable iptables connection tracking:*

```bash
# For stateless workloads (massive connection counts)
# Cilium: enable BPF-based connection tracking bypass
```

*6. CPU pinning for network-intensive pods:*

```yaml
resources:
  limits:
    cpu: "2"
  requests:
    cpu: "2"
# Combined with CPU manager static policy
```

== Debugging Container Networks

*Namespace inspection:*

```bash
# Find container network namespace
docker inspect <container> | jq '.[0].NetworkSettings'
pid=$(docker inspect -f '{{.State.Pid}}' <container>)
nsenter -t $pid -n ip addr

# Kubernetes pod namespace
kubectl exec -it <pod> -- ip addr
crictl inspect <container-id> | jq '.info.runtimeSpec.linux.namespaces'
```

*Traffic capture:*

```bash
# Capture on veth (host side)
tcpdump -i veth1234abc -nn

# Capture inside container
nsenter -t <pid> -n tcpdump -i eth0 -nn

# Capture on overlay (VXLAN)
tcpdump -i flannel.1 -nn
```

*Connectivity testing:*

```bash
# Pod-to-pod
kubectl exec pod1 -- curl pod2-ip:8080

# Service resolution
kubectl exec pod1 -- nslookup my-service.namespace.svc.cluster.local

# Check kube-proxy rules
iptables -t nat -L KUBE-SERVICES -n
ipvsadm -Ln  # If using IPVS mode
```

== References

*Specifications:*

RFC 7348: Virtual eXtensible Local Area Network (VXLAN). Mahalingam, M. et al. (2014).

CNI Specification v1.0.0. containernetworking/cni. https://github.com/containernetworking/cni

*Research:*

Suo, K., Zhao, Y., Chen, W., & Rao, J. (2018). "An Analysis and Empirical Study of Container Networks." IEEE INFOCOM 2018.

Amaral, M., Polo, J., Carrera, D., et al. (2021). "Performance Evaluation of Microservices Architectures using Containers." IEEE MASCOTS.

*Documentation:*

Docker Networking Overview. https://docs.docker.com/network/

Kubernetes Networking Concepts. https://kubernetes.io/docs/concepts/cluster-administration/networking/

Cilium Documentation. https://docs.cilium.io/

Calico Documentation. https://docs.projectcalico.org/

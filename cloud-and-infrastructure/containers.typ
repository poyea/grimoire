#import "../template.typ": xref

= Containers: OCI, runc, containerd, Image Layers

Containers are not a kernel feature but a *user-space packaging convention* over kernel features — namespaces, cgroups, seccomp, capabilities, mount propagation, and union filesystems. The OCI standards (runtime-spec, image-spec, distribution-spec) define the contract between image builders, registries, and runtimes so that an image produced by Docker, Buildah, or Bazel runs unchanged under runc, crun, youki, or gVisor. This chapter follows a container from `docker push` through registry, kubelet pull, image unpacking, runc invocation, and finally the `clone()` that creates the container's first process.

*See also:* #xref("cloud-and-infrastructure", "kubernetes-internals", label: "Kubernetes Internals"), #xref("linux-kernel", "cgroups-namespaces", label: "cgroups and Namespaces") (linux-kernel), #xref("linux-kernel", "containers-in-the-kernel", label: "Containers in the Kernel") (linux-kernel), #xref("cloud-and-infrastructure", "serverless", label: "Serverless Computing"), #xref("networking", "container-networking", label: "Container Networking") (networking).

== The OCI Stack

```text
   ┌─────────────────────────────────────────────────────┐
   │ User: docker / podman / nerdctl / kubectl           │
   ├─────────────────────────────────────────────────────┤
   │ High-level runtime: containerd / CRI-O              │
   │   (image mgmt, snapshots, CRI)                      │
   ├─────────────────────────────────────────────────────┤
   │ OCI runtime: runc / crun / youki / kata / gVisor    │
   │   (clone + setns + cgroup setup)                    │
   ├─────────────────────────────────────────────────────┤
   │ Linux kernel: namespaces, cgroups v2, seccomp, LSM  │
   └─────────────────────────────────────────────────────┘
```

The crucial division is between the *high-level* runtime (containerd) which manages images, networks, and the lifecycle of many containers, and the *low-level* OCI runtime (runc) which is a one-shot process that creates a single container and exits.

== OCI Image Format

An OCI image is a content-addressed bundle:

```text
manifest.json (SHA256:abcd...)
   ├── config.json   (env, cmd, entrypoint, rootfs layers)
   └── layers[]      (each a tar.gz, identified by digest)
       layer0: SHA256:1111... (base OS)
       layer1: SHA256:2222... (apt-get install)
       layer2: SHA256:3333... (COPY app/)
```

```json
// manifest.json (simplified)
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.oci.image.manifest.v1+json",
  "config": {"digest": "sha256:abcd...", "size": 7023},
  "layers": [
    {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
     "digest": "sha256:1111...", "size": 32654000},
    {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
     "digest": "sha256:2222...", "size": 1843},
    {"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
     "digest": "sha256:3333...", "size": 251}
  ]
}
```

*Content addressing implications:* identical layers are deduplicated across images on a host and across blobs in a registry. A base image of 200 MB is stored once even if 50 derived images reuse it.

== Image Layers and Union Filesystems

Each layer is a *changeset* expressing the diff from its parent: added files, modified files (whole-file replacement), and deletions (whiteouts). At runtime, layers are stacked by a union filesystem (overlayfs on modern Linux):

```bash
# Manual overlayfs equivalent of what containerd does
mount -t overlay overlay \
  -o lowerdir=/layers/l1:/layers/l0,upperdir=/upper,workdir=/work \
  /merged
```

- *lowerdirs* are read-only layers (rightmost is bottom).
- *upperdir* is the writable layer for this container.
- File deletions in upper are represented by character device whiteouts.
- Modifications copy-up: first write to a lower file copies it to upper.

*Snapshot drivers* in containerd:
- `overlayfs` — default, ~zero overhead read, copy-up cost on first write.
- `btrfs`, `zfs` — native copy-on-write at block level, fewer copy-up stalls.
- `devmapper` — block-level thin provisioning, used on RHEL family.
- `stargz` / `nydus` — lazy-pull formats; container starts before image is fully downloaded.

== Registry Protocol (Distribution Spec)

```text
GET  /v2/                                    # version probe
HEAD /v2/<name>/manifests/<reference>        # exists?
GET  /v2/<name>/manifests/<reference>        # fetch manifest
GET  /v2/<name>/blobs/<digest>               # fetch layer
POST /v2/<name>/blobs/uploads/               # start push (returns Location)
PATCH <Location>                             # chunked upload
PUT   <Location>?digest=<sha256>             # finalize
```

Pushes are content-addressed: the client computes SHA256 first and the registry accepts only if the digest matches. This makes pushes idempotent and immune to corruption.

== runc Lifecycle

```bash
# What runc actually does, in pseudocode
config = parse("config.json")
fd = open("/proc/self/ns/pid", O_RDONLY)
clone(CLONE_NEWPID|CLONE_NEWNS|CLONE_NEWUTS|CLONE_NEWIPC|CLONE_NEWUSER|CLONE_NEWNET|CLONE_NEWCGROUP, child)

# Inside child:
sethostname(config.hostname)
pivot_root("/bundle/rootfs", "/bundle/rootfs/.old_root")
umount2("/.old_root", MNT_DETACH)
setresuid(config.uid); setresgid(config.gid)
prctl(PR_SET_NO_NEW_PRIVS, 1)
seccomp_load(config.seccomp_profile)
capset(config.capabilities)
execve(config.process.args[0], config.process.args, config.process.env)
```

The OCI runtime-spec's `config.json` is the input to this process: it lists every namespace to enter, every mount to perform, the cgroup limits to apply, the seccomp filter, and the entrypoint to exec.

```json
{
  "ociVersion": "1.1.0",
  "process": {
    "terminal": false,
    "user": {"uid": 1000, "gid": 1000},
    "args": ["/bin/sh"],
    "env": ["PATH=/usr/bin", "TERM=xterm"],
    "cwd": "/",
    "capabilities": {"bounding": ["CAP_NET_BIND_SERVICE"]},
    "noNewPrivileges": true
  },
  "root": {"path": "rootfs", "readonly": true},
  "linux": {
    "namespaces": [
      {"type": "pid"}, {"type": "network"}, {"type": "mount"},
      {"type": "ipc"}, {"type": "uts"}, {"type": "user"}
    ],
    "resources": {
      "memory": {"limit": 536870912},
      "cpu": {"shares": 1024, "quota": 50000, "period": 100000}
    },
    "seccomp": {"defaultAction": "SCMP_ACT_ERRNO", "syscalls": []}
  }
}
```

== containerd Architecture

containerd is a daemon exposing a gRPC API. Internally it is composed of pluggable services:

```text
   client (kubelet via CRI, ctr, nerdctl)
        │ gRPC
        ▼
   ┌───────────────────────────────────────┐
   │ containerd                            │
   │  ┌─────────────┐  ┌────────────────┐ │
   │  │ images svc  │  │ content svc    │ │
   │  └─────────────┘  └────────────────┘ │
   │  ┌─────────────┐  ┌────────────────┐ │
   │  │ snapshots   │  │ tasks (shim)   │ │
   │  └─────────────┘  └────────────────┘ │
   └────────────┬──────────────────────────┘
                │ exec
                ▼
        containerd-shim-runc-v2
                │ fork
                ▼
              runc create → container init → exec
```

The *shim* is the parent of the container process so that containerd can restart without orphaning workloads (the shim reparents to PID 1 and keeps the container alive).

== CRI Runtime Comparison

#table(
  columns: 5,
  [*Runtime*], [*Isolation*], [*Startup*], [*Use Case*], [*Notes*],
  [runc], [namespaces+cgroups], [~50 ms], [default], [reference impl, Go],
  [crun], [namespaces+cgroups], [~10 ms], [low-latency], [C, lighter],
  [youki], [namespaces+cgroups], [~30 ms], [research], [Rust],
  [Kata], [QEMU/Firecracker VM], [~150-500 ms], [hostile multi-tenant], [hardware-level],
  [gVisor (runsc)], [user-space kernel], [~300 ms], [defense in depth], [syscall interception],
  [Firecracker (via Kata)], [microVM], [~125 ms], [Lambda, Fargate], [see serverless.typ],
)

The isolation/startup tradeoff is fundamental: stronger isolation requires a heavier setup. For trusted, single-tenant workloads runc is appropriate; for executing untrusted code (Lambda, online judges, CI runners running PR code) microVMs are the only safe choice.

== Networking Setup (CNI)

A container starts in a fresh network namespace with only `lo`. The CNI plugin (called by containerd via the CRI flow) wires it in:

```bash
# What bridge CNI does, equivalent shell
ip netns add ctr1
ip link add veth0 type veth peer name veth1
ip link set veth1 netns ctr1
ip link set veth0 master cni0       # bridge on host
ip link set veth0 up
ip netns exec ctr1 ip link set veth1 name eth0 up
ip netns exec ctr1 ip addr add 10.244.0.5/24 dev eth0
ip netns exec ctr1 ip route add default via 10.244.0.1
```

CNI plugins are detailed in #xref("cloud-and-infrastructure", "kubernetes-internals", label: "Kubernetes Internals").

== Resource Limits via Cgroups v2

```bash
# Hierarchical limit: parent cap that children share
cd /sys/fs/cgroup
mkdir mygroup && cd mygroup
echo "+cpu +memory +io +pids" > cgroup.subtree_control
echo "50000 100000" > cpu.max          # 50ms quota / 100ms period → 0.5 CPU
echo "1073741824" > memory.max          # 1 GiB
echo "1000" > pids.max
echo "8:0 wbps=10485760" > io.max       # 10 MB/s writes
echo $$ > cgroup.procs                  # move current shell in
```

OOM behaviour is governed by `memory.oom.group` (kill the whole cgroup vs a single process) and `memory.swap.max`. Production containers should set both `requests` (cgroup `cpu.weight`) and `limits` (`cpu.max`).

== Rootless Containers and User Namespaces

User namespaces remap UIDs so that root (0) inside the container is an unprivileged UID outside. `newuidmap`/`newgidmap` configure the mapping via `/etc/subuid`:

```bash
cat /etc/subuid
# alice:100000:65536
# Means: alice may map container UIDs 0-65535 → host UIDs 100000-165535.
```

Rootless mode is what makes `podman` safe to run as a normal user and underpins user-namespace remapping in modern Kubernetes (KEP-127).

== Build Tooling: Dockerfile, BuildKit, Bazel

BuildKit (the default Docker engine since 23.x) implements a DAG of LLB ops, enabling concurrent stage execution, content-addressed cache mounts, and remote cache import/export.

```dockerfile
# syntax=docker/dockerfile:1.7
FROM golang:1.22 AS build
WORKDIR /src
COPY go.* ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download
COPY . .
RUN --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -o /out/app ./cmd/server

FROM gcr.io/distroless/static:nonroot
COPY --from=build /out/app /app
USER nonroot:nonroot
ENTRYPOINT ["/app"]
```

Multi-stage builds keep the runtime image minimal (distroless is ~2 MB). `--mount=type=cache` makes the module and build caches persist across builds without bloating the image.

== Image Security

- *Signing:* cosign (Sigstore) signs the manifest digest, verifies on pull via admission policy.
- *SBOM:* Syft generates SPDX/CycloneDX; stored alongside the image (referrers API).
- *Vulnerability scanning:* Trivy, Grype map CVE feeds onto package metadata in layers.
- *Provenance:* SLSA attestations bind a build to a signed VCS commit.

== Worked Example: From Build to Run

```bash
# Build with BuildKit, sign, push, run.
docker buildx build --sbom=true --provenance=true -t reg.example.com/app:v1.4 --push .
cosign sign reg.example.com/app:v1.4
# On host:
ctr image pull reg.example.com/app:v1.4
ctr run --rm -t reg.example.com/app:v1.4 inst1 /app --port 8080
```

containerd resolves the manifest, fetches missing blobs (parallelized), unpacks layers into the snapshotter, materialises the rootfs via overlayfs, generates a runtime-spec, and invokes runc through the shim. Total cold-start time on a warm host: 30-100 ms; on a fresh host with a 200 MB image: 5-15 s dominated by pull.

== Further Reading

OCI: Runtime Specification 1.1, Image Specification 1.1, Distribution Specification.

Sigstore Project: "Sigstore: Software Signing for Everybody." USENIX Security 2022.

Harter, T. et al. (2016). "Slacker: Fast Distribution with Lazy Docker Containers." FAST.

Anwar, A. et al. (2018). "Improving Docker Registry Design Based on Production Workload Analysis." FAST.

containerd documentation; BuildKit LLB reference; CNI specification 1.0.

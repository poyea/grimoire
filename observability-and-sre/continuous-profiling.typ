#import "../template.typ": xref

= Continuous Profiling <continuous-profiling>

A distributed trace tells you which service is slow; a profile tells you which line of code inside that service is consuming the CPU or memory. Continuous profiling (always-on, low-overhead profiling in production) closes the gap between aggregate metrics and individual request traces by providing a third dimension: resource attribution at the code level. This chapter covers the mechanics of CPU and memory profiling, the flame graph as the canonical visualization, eBPF-based always-on profilers, and the emerging practice of correlating profiles with traces.

*See also:* #xref("observability-and-sre", "metrics-systems", label: "Metrics Systems"), #xref("observability-and-sre", "distributed-tracing", label: "Distributed Tracing"), #xref("observability-and-sre", "the-three-pillars-and-beyond", label: "The Three Pillars and Beyond")

== CPU Profiling: Sampling vs Instrumentation

Two fundamentally different approaches produce CPU profiles.

*Instrumentation-based profiling* inserts probes at every function entry and exit, recording exact call counts and time spent. It is precise but incurs 2–10× overhead, making it impractical in production at continuous rates.

*Sampling-based profiling* sends a signal (SIGPROF on POSIX systems) to the process at a fixed frequency (typically 99 or 100 Hz) and captures the current call stack at each interruption. Statistical inference yields fractional CPU time per stack frame. At 99 Hz the overhead is typically under 1 % of a single CPU core.

The choice of 99 Hz rather than 100 Hz is deliberate: it avoids aliasing with processes that run at exactly 100 Hz (e.g., the Linux scheduler tick), which would cause systematic oversampling of specific code paths.

=== Call Stack Capture

On Linux, sampling profilers capture stacks either through kernel signals (uprobes, perf_events) or via a language runtime. Languages with managed runtimes (JVM, Go) expose safe-point stack walks that avoid tearing mid-GC. Native profilers must handle frame pointer omission, since compilers can elide the frame pointer register for performance, breaking naive stack unwinding. Solutions include:

- Compiling with `-fno-omit-frame-pointer` (adds 1–3 % overhead).
- Using DWARF CFI (Call Frame Information) unwinding: slower but always correct.
- LBR (Last Branch Record) hardware support on Intel CPUs: fast, limited to 32 frames.
- eBPF with `bpf_get_stackid`: kernel-assisted, does not require frame pointers.

== Flame Graphs

Brendan Gregg invented the *flame graph* at Netflix in 2011. The x-axis is alphabetically sorted stack frames (not time); the y-axis is stack depth. Width represents the fraction of samples containing that frame. A wide frame at the top of a tower is a hot function.

Reading rules:
- The *widest towers* identify the dominant CPU consumers.
- *Plateau shapes* (wide frames with thin children) indicate a function that is itself slow, not its callees.
- *Tower shapes* (narrow at top) indicate deep call chains where a leaf function dominates.

Flame graphs are generated from a folded stack file:

```
main;http.ListenAndServe;net.Accept;crypto/tls.Handshake 42
main;http.ListenAndServe;ServeHTTP;json.Marshal;encoding.json.encode 317
main;http.ListenAndServe;ServeHTTP;db.Query;pq.Exec 891
```

Each line is a semicolon-separated stack with a sample count. The `flamegraph.pl` script (Gregg, GitHub) or the Go `github.com/google/pprof` package renders SVG from this format.

=== Differential Flame Graphs

A *differential flame graph* subtracts one profile from another, coloring frames red (regression) or blue (improvement). This is the standard technique for comparing profiles before and after a deploy:

```bash
# Capture baseline profile (30 s)
go tool pprof -seconds 30 http://service:6060/debug/pprof/profile > base.pb.gz

# Deploy change, capture comparison
go tool pprof -seconds 30 http://service:6060/debug/pprof/profile > new.pb.gz

# Generate differential
pprof -diff_base base.pb.gz new.pb.gz -http=:8080 new.pb.gz
```

The differential view makes regressions visible even when the absolute magnitude is small: a 2 % regression in a hot path is invisible on an absolute flame graph but vivid in red on a differential.

== pprof Format

*pprof* is the de facto standard profile format, originally from Google's performance tools. It is a Protocol Buffer encoding of stacks with location, function, line number, and sample type metadata.

```
Profile {
  sample_type: [{ type: "cpu", unit: "nanoseconds" }]
  sample: [{ location_id: [3, 2, 1], value: [15000000] }]
  location: [{ id: 1, line: [{ function_id: 10, line: 42 }] }]
  function: [{ id: 10, name: "json.Marshal", filename: "encoding/json/encode.go" }]
}
```

Go's `runtime/pprof` and `net/http/pprof` packages emit this format natively. Rust's `pprof-rs` crate emits it. Java agents (async-profiler, JFR converter) output it. The format supports multiple sample types in a single file; CPU samples, heap allocations, and mutex contention can coexist.

```go
// Expose all pprof endpoints in a Go HTTP server
import _ "net/http/pprof"

// Endpoints available at runtime:
// GET /debug/pprof/profile?seconds=30   CPU profile
// GET /debug/pprof/heap                 heap snapshot
// GET /debug/pprof/goroutine            goroutine stacks
// GET /debug/pprof/allocs               allocation profile
// GET /debug/pprof/mutex                mutex contention
// GET /debug/pprof/block                blocking profile
```

== eBPF-Based Continuous Profilers

Traditional sampling profilers require per-language agents or runtime cooperation. *eBPF* (extended Berkeley Packet Filter) runs sandboxed programs in the Linux kernel that can attach to perf events, capture stack traces across language boundaries, and stream data to userspace with minimal overhead (typically 0.5–2 % CPU).

eBPF-based profilers are *always-on* by design: they capture profiles continuously, store them in a time-series backend, and expose them through a query interface. The three leading open-source options:

#table(
  columns: (auto, auto, auto, auto),
  align: left,
  table.header[*Profiler*][*Vendor*][*Backend*][*Notable feature*],
  [Parca], [Parca (OSS)], [Columnar in-process], [Pull model; pprof-native],
  [Pyroscope], [Grafana Labs], [Object storage], [Multi-language; push model],
  [Polar Signals Cloud], [Polar Signals], [Managed], [DWARF unwinding; native symbols],
)

=== Pyroscope Architecture

Pyroscope (now Grafana Pyroscope) uses a push model: language SDKs or the eBPF agent send profiles to a central server every 10–15 seconds. The server stores profiles in a columnar format optimized for flamegraph aggregation over time ranges.

```python
import pyroscope

pyroscope.configure(
    application_name = "checkout-service",
    server_address   = "http://pyroscope:4040",
    tags             = {"region": "us-east-1", "version": "v2.3.1"},
)
# From this point, CPU and memory profiles are sent automatically
```

=== Parca Architecture

Parca uses a pull model: the Parca server scrapes `/debug/pprof/profile` endpoints on a configurable interval, stores profiles as compressed columnar data (using Apache Parquet internally), and exposes a gRPC query API. It integrates with Kubernetes service discovery via the same annotations Prometheus uses.

```yaml
# parca.yaml — scrape configuration
scrape_configs:
  - job_name: go-services
    scrape_interval: 1m
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_parca_io_scrape]
        action: keep
        regex: "true"
```

== Heap and Allocation Profiling

CPU profiles miss memory pressure. Two complementary profile types address memory:

*Heap profiles* show live heap allocations at the time of sampling. They identify which call paths are responsible for the current live object graph, useful for diagnosing steady-state memory growth.

*Allocation profiles* record every allocation (sampled at a byte threshold) regardless of whether the object is still live. They reveal allocation hot spots that stress the garbage collector even if objects are short-lived. In Go, the `allocs` pprof endpoint provides this; in Java, async-profiler's `--alloc` flag does.

```go
// Go: take a heap profile programmatically
f, _ := os.Create("heap.pb.gz")
defer f.Close()
runtime.GC()  // force GC so dead objects are excluded from live count
pprof.WriteHeapProfile(f)
```

In JVM languages, heap profiling additionally captures *object histograms* (the distribution of live bytes by class):

```
 num     #instances         #bytes  class name
   1:      15234123     1830494760  byte[]
   2:       2341234      374597440  java.lang.String
   3:        891234      213895816  com.example.Request
```

=== Memory Leak Detection

A steady upward slope in the heap profile over hours is the signature of a memory leak. Differential heap profiles (comparing two snapshots) identify which allocation sites grew:

```bash
# Go: two heap snapshots, compare with pprof
go tool pprof -diff_base heap_t0.pb.gz heap_t1.pb.gz
```

== Language-Specific Profiling

=== Go

Go ships a production-safe profiler in the standard library. The `runtime` package controls sampling rates:

```go
runtime.SetCPUProfileRate(99)        // samples/sec; 0 = disable
runtime.MemProfileRate = 512 * 1024  // sample every N bytes allocated
runtime.SetMutexProfileFraction(5)   // sample 1-in-5 mutex contentions
runtime.SetBlockProfileRate(1000)    // sample goroutine blocks > 1 µs
```

=== Rust

Rust has no runtime profiler, but the `pprof-rs` crate provides a sampling profiler that registers a SIGPROF handler:

```rust
let guard = pprof::ProfilerGuardBuilder::default()
    .frequency(99)
    .blocklist(&["libc", "pthread"])
    .build()
    .unwrap();

// ... do work ...

if let Ok(report) = guard.report().build() {
    let file = File::create("profile.pb").unwrap();
    report.pprof().unwrap().encode(&mut BufWriter::new(file)).unwrap();
}
```

Alternatively, `cargo flamegraph` wraps `perf record` and `flamegraph.pl` for one-command flame graph generation in development.

=== JVM (Java, Kotlin, Scala)

*async-profiler* is the standard choice for JVM profiling in production. It uses AsyncGetCallTrace (a non-safepoint JVM API) to avoid safe-point bias, the distortion that occurs when traditional JVMTI profilers only sample at GC safe points, causing hot loops to appear cold.

```bash
# Attach to running JVM process (PID 12345), CPU profile for 30s
./asprof -d 30 -f profile.jfr 12345

# Allocation profile
./asprof -e alloc -d 30 -f allocs.jfr 12345

# Convert JFR to pprof
asprof convert --pprof profile.jfr -o profile.pb.gz
```

== Profile-Trace Correlation

The most powerful observability pattern is linking a distributed trace span to the profile captured during that span's execution. When a P99 latency spike is visible in traces, engineers can click through to the profile of the hot span and see the exact stack frame responsible, without having to reproduce the issue.

The mechanism: the profiler records the goroutine/thread ID alongside each stack sample. The tracing SDK tags spans with the thread ID and a time range. The profiling backend can then filter the profile to samples that occurred during the span's time range on the matching thread.

In Go with OpenTelemetry and Pyroscope:

```go
import "github.com/grafana/otel-profiling-go"

// Wrap the tracer provider to enable profile-trace linking
tp = otelpyroscope.NewTracerProvider(tp,
    otelpyroscope.WithAppName("checkout"),
    otelpyroscope.WithProfileURL("http://pyroscope:4040"),
    otelpyroscope.WithProfileBaselineLabels(map[string]string{
        "env": "production",
    }),
    otelpyroscope.WithRootSpanOnly(), // only link root spans
)
```

The Grafana Explore UI renders a split panel: the trace waterfall on top, the aggregated flame graph for the selected span below. This collapses the debug loop from hours (reproduce issue, attach profiler, analyze) to minutes (click span, read flame graph).

== Overhead Budget

A continuous profiler must be safe to leave running in production indefinitely. Published overhead measurements:

#table(
  columns: (auto, auto, auto),
  align: left,
  table.header[*Profiler*][*Typical CPU overhead*][*Memory overhead*],
  [Go `pprof` at 99 Hz], [$< 1%$], [Negligible],
  [async-profiler (JVM)], [$1-2%$], [$approx 10$ MB],
  [Pyroscope eBPF agent], [$0.5-1.5%$], [$< 50$ MB],
  [Parca eBPF], [$0.5-1%$], [$< 30$ MB],
  [Instrumentation profiler], [$200-1000%$], [High],
)

The eBPF profilers pay a higher fixed cost (kernel BPF programs, ring buffer) but amortize it across all processes on the host. Language-native profilers pay per-process but have lower per-host overhead when few services run per machine.

== Further Reading

Gregg, B. (2016). "The Flame Graph." _Communications of the ACM_ 59(6):48–57.

Gregg, B. (2020). _Systems Performance: Enterprise and the Cloud, 2nd ed._ Addison-Wesley. Chapters 6 and 13 (CPU and Memory profiling).

Gregg, B. (2019). _BPF Performance Tools._ Addison-Wesley. Chapters 4–6 (eBPF profiling).

Pangin, A. "async-profiler: low-overhead sampling profiler for the JVM." https://github.com/async-profiler/async-profiler

Polar Signals. "Parca: Open Source Continuous Profiling." https://www.parca.dev

Grafana Labs. "Pyroscope: Open Source Continuous Profiling." https://pyroscope.io/docs/

Google. "pprof: a tool for visualization and analysis of profiling data." https://github.com/google/pprof

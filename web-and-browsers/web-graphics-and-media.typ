#import "../template.typ": xref

= Graphics and Media in the Browser <web-graphics-and-media>

Browsers expose the GPU, video decoders, cameras, microphones and the audio hardware to untrusted web pages. Every one of these APIs has to be fast enough for games, video calls and machine learning, yet safe enough that a malicious page cannot read other processes' memory, crash the GPU driver, or fingerprint the user. This chapter covers how the browser mediates each: Canvas 2D, WebGL and its translation layer, WebGPU, the video pipeline from `<video>` through Media Source Extensions, DRM and WebCodecs, real-time communication with WebRTC, and the Web Audio API.

*See also:* #xref("web-and-browsers", "rendering-pipeline", label: "Rendering Pipeline") (compositing), #xref("web-and-browsers", "browser-architecture", label: "Browser Architecture") (the GPU process), #xref("graphics-and-rendering", "explicit-gpu-apis", label: "Explicit GPU APIs: Vulkan, Direct3D 12, and Metal"), #xref("networking", "realtime-protocols", label: "Real-Time Protocols")

== The GPU Process

No renderer talks to the GPU driver directly. Graphics drivers are large, written in unsafe languages, and historically a rich source of exploitable bugs, so browsers isolate them in a separate *GPU process*. Renderers serialise their graphics commands into a shared-memory *command buffer*; the GPU process validates and executes them. If the driver crashes, only the GPU process dies and is restarted, and pages see a "context lost" event instead of the whole browser failing. The cost is an extra hop and validation for every command, which shapes the performance characteristics of every graphics API below.

== Canvas 2D

The `<canvas>` 2D context is an immediate-mode drawing API (paths, fills, strokes, images, text) originally designed for software rasterisation. In Chromium it is backed by Skia and usually accelerated on the GPU. Its performance traps follow from that hidden GPU:

- *Readback*: `getImageData` must wait for all pending GPU work and copy pixels back to the CPU, a pipeline stall. Canvases that are read often are better kept in software (the `willReadFrequently` context option).
- *State changes*: thousands of small draws with different styles are slower than batching same-styled shapes into one path.
- *Text* rendering is expensive; cache rendered text in an offscreen canvas.

`OffscreenCanvas` lets a worker own a canvas and draw without touching the main thread, so rendering continues smoothly while the page runs other script.

== WebGL

WebGL 1 and 2 expose OpenGL ES 2.0 and 3.0 semantics to JavaScript. Native OpenGL is poorly supported on Windows and deprecated on Apple platforms, so browsers translate WebGL through *ANGLE*, which implements OpenGL ES on top of Direct3D 11, Metal or Vulkan. Every WebGL call therefore passes through JavaScript bindings, the command buffer, validation in the GPU process, ANGLE's translation, and the native driver. Shaders are parsed, validated, rewritten for safety (for example inserting bounds checks on array indexing and initialising all variables, so a shader cannot read uninitialised GPU memory left by another process) and translated into HLSL, MSL or SPIR-V.

Two consequences matter for developers. Draw calls are expensive, often several times the cost of native OpenGL, so WebGL applications batch aggressively with instancing and texture atlases. And shader compilation happens on first use, which causes visible hitches unless programs are compiled and linked during loading (the `KHR_parallel_shader_compile` extension lets the page poll for completion instead of blocking).

== WebGPU

WebGPU is the web's modern GPU API, modelled on Vulkan, Direct3D 12 and Metal (#xref("graphics-and-rendering", "explicit-gpu-apis", label: "Explicit GPU APIs: Vulkan, Direct3D 12, and Metal")): explicit pipelines, bind groups, command encoders and first-class compute shaders, with its own shading language, *WGSL*. It shipped in Chrome in 2023, and Safari and Firefox followed in 2025. Chromium implements it with *Dawn* and Firefox with *wgpu*; both also serve as portable native graphics libraries.

Unlike the native explicit APIs, WebGPU must be safe:

- *Validation* of every call, with errors reported asynchronously rather than undefined behaviour.
- *Automatic synchronisation*: the implementation tracks resource usage and inserts the barriers that native APIs require the application to write by hand.
- *Robust access*: out-of-bounds buffer reads in shaders return a value in bounds (or zero) instead of reading arbitrary memory.
- *Uniformity analysis* in WGSL rejects shaders that use derivatives or barriers in non-uniform control flow, where native behaviour is undefined.
- Limits and features are reported in coarse buckets, and timestamp queries are quantised, to reduce fingerprinting and timing side channels.

A minimal compute example doubles every element of a buffer:

```js
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

const module = device.createShaderModule({
  code: `
    @group(0) @binding(0) var<storage, read_write> data: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
      if (id.x < arrayLength(&data)) {
        data[id.x] = data[id.x] * 2.0;
      }
    }`,
});

const input = new Float32Array(1024).map((_, i) => i);
const storage = device.createBuffer({
  size: input.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(storage, 0, input);

const readback = device.createBuffer({
  size: input.byteLength,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

const pipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module, entryPoint: "main" },
});
const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: storage } }],
});

const encoder = device.createCommandEncoder();
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(0, bindGroup);
pass.dispatchWorkgroups(Math.ceil(input.length / 64));
pass.end();
encoder.copyBufferToBuffer(storage, 0, readback, 0, input.byteLength);
device.queue.submit([encoder.finish()]);

await readback.mapAsync(GPUMapMode.READ);
const result = new Float32Array(readback.getMappedRange().slice(0));
readback.unmap();
```

Compute shaders make WebGPU a platform for machine learning in the browser: libraries such as ONNX Runtime Web, Transformers.js and WebLLM run vision models and small language models locally on the user's GPU, with no server round trip and no data leaving the device.

== The Video Pipeline

Playing a `<video>` is a pipeline spread across processes: the network stack fetches media segments, a *demuxer* splits the container (MP4, WebM) into audio and video streams, a *decoder* (hardware where available, in a sandboxed process) produces frames, and the frames, typically already in GPU memory as YUV textures, go straight to the compositor without passing through JavaScript or the renderer's main thread.

Power efficiency depends on that last step. If the video frame can be handed to the operating system's display hardware as an *overlay* plane (DirectComposition on Windows, CoreAnimation on macOS), the GPU need not composite it with the page at all, and a laptop can play video for hours. CSS effects on the video element (rounded corners, filters, some transforms) can force full GPU compositing and noticeably raise power use.

=== Adaptive Streaming

Streaming services do not point `<video>` at a single file. With *Media Source Extensions* (MSE), a JavaScript player (hls.js, dash.js, Shaka Player) fetches segments of a few seconds each, chooses a bitrate per segment from measured throughput and buffer level, and appends the segments to a `SourceBuffer`. Adaptive bitrate algorithms balance three goals that conflict: high quality, few quality switches, and no rebuffering stalls. Buffer-based algorithms (such as BOLA) have largely replaced purely throughput-based ones because throughput estimates are noisy.

=== Encrypted Media

*Encrypted Media Extensions* (EME) connect MSE to a *content decryption module* (CDM): Widevine (Chrome, Firefox, Android), FairPlay (Safari) or PlayReady (Edge, Windows). The player obtains a licence from the service's licence server through the CDM, and the CDM decrypts and often decodes the media in a protected path that script cannot read. The highest robustness levels keep decrypted frames inside hardware-protected memory, which is why screenshots of protected video come out black. CDMs are closed-source components, a long-running point of contention with the open web.

=== WebCodecs

Applications such as video editors, cloud gaming clients and conferencing tools need direct access to encoders and decoders, without the `<video>` element's buffering and timing. *WebCodecs* exposes `VideoDecoder`, `VideoEncoder`, `AudioDecoder` and `AudioEncoder` with `VideoFrame` objects that can wrap GPU textures, be drawn to canvas or WebGPU, and be transferred between workers without copying. It leaves containers, network transport and synchronisation to the application.

== Real-Time Communication

*WebRTC* provides peer-to-peer audio, video and data with latency low enough for conversation, which rules out the buffering that streaming relies on:

- `getUserMedia` captures camera and microphone (with permission), and `getDisplayMedia` captures the screen.
- `RTCPeerConnection` sets up the connection. Peers exchange *session descriptions* (SDP) and *ICE candidates* through an application-provided signalling channel. *ICE* tries direct paths first, uses *STUN* servers to discover public addresses behind NAT, and falls back to relaying through *TURN* servers when no direct path works.
- Media is encrypted with *DTLS-SRTP*. Codecs include Opus for audio and VP8, VP9, H.264 and AV1 for video.
- *Congestion control* (Google Congestion Control, using transport-wide feedback) estimates available bandwidth from packet delay trends and loss, and adjusts encoder bitrate continuously. A *jitter buffer* on the receiver trades a little latency for smooth playout, and forward error correction and retransmission recover losses.
- *Data channels* carry arbitrary messages over SCTP inside DTLS, reliable or unreliable.

Multi-party calls rarely use a full mesh, because each participant would upload a separate stream to every other. A *selective forwarding unit* (SFU) receives each sender's stream once and forwards it. With *simulcast* (several resolutions encoded at once) or *scalable video coding* (layered streams), the SFU forwards to each receiver the layer that fits its bandwidth and screen layout, without transcoding.

*WebTransport*, built on HTTP/3 and QUIC, offers client-server streams and unreliable datagrams with less setup complexity than WebRTC, and is the basis of emerging *Media over QUIC* work for live streaming with latency between that of WebRTC and of segment-based streaming.

== Web Audio

The Web Audio API builds an *audio graph* of nodes (sources, gains, filters, convolution reverbs, analysers, spatial panners) that is processed on a dedicated real-time audio thread in blocks of 128 sample frames (the *render quantum*). Custom processing runs in an `AudioWorklet`, whose `process` method executes on that real-time thread and must finish within the quantum's deadline (under 3 ms at 48 kHz) without allocating memory or blocking, or the output glitches. Heavy DSP is typically compiled to WebAssembly and run inside the worklet, with `SharedArrayBuffer` ring buffers (requiring cross-origin isolation) to exchange data with other threads. Output latency depends on the platform's audio stack and ranges from about 10 ms on good desktop systems to considerably more on some mobile devices.

== Further Reading

W3C (2024). "WebGPU." W3C Candidate Recommendation specification.

W3C (2024). "WebGPU Shading Language." W3C Candidate Recommendation specification.

Khronos Group (2024). "WebGL 2.0 Specification." Khronos specification.

W3C (2024). "Media Source Extensions." W3C Recommendation specification.

W3C (2024). "Encrypted Media Extensions." W3C Recommendation specification.

W3C (2024). "WebCodecs." W3C Working Draft specification.

Jennings, C. et al. (2021). "WebRTC: Real-Time Communication in Browsers." W3C Recommendation specification.

Holmer, S., Lundin, H., Carlucci, G., De Cicco, L., Mascolo, S. (2016). "A Google Congestion Control Algorithm for Real-Time Communication." IETF draft-ietf-rmcat-gcc.

Spiteri, K., Urgaonkar, R., Sitaraman, R. (2016). "BOLA: Near-Optimal Bitrate Adaptation for Online Videos." IEEE INFOCOM.

W3C (2024). "Web Audio API." W3C Recommendation specification.

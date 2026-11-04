#import "../template.typ": xref

= Explicit GPU APIs: Vulkan, Direct3D 12, and Metal <explicit-gpu-apis>

OpenGL and Direct3D 11 presented the GPU as a state machine: set some state, issue a draw, and let the driver work out memory placement, synchronisation, resource transitions and shader compilation behind your back. That convenience cost CPU time on every draw, made performance unpredictable (a draw could stall for tens of milliseconds while the driver compiled a shader variant), and confined submission to essentially one thread. Vulkan, Direct3D 12 and Metal hand those responsibilities to the application. This chapter covers what the application now owns: command recording and submission, pipeline state objects, resource binding, memory management, synchronisation, render passes on tiled GPUs, presentation, and debugging when the GPU stops responding.

*See also:* #xref("graphics-and-rendering", "rasterization-pipeline", label: "The Rasterization Pipeline"), #xref("graphics-and-rendering", "shaders", label: "Shaders and Shading Languages") (binding models, compilation), #xref("graphics-and-rendering", "realtime-engines", label: "Real-Time Engine Architecture") (render graphs), #xref("gpu-architecture", "gpu-fundamentals", label: "GPU Fundamentals")

== Why Explicit

A Direct3D 11 or OpenGL driver had to infer, on every draw, which resources might be read or written, insert barriers and cache flushes accordingly, track which memory each resource lived in, and compile or patch shaders when state combinations changed. All of that was hidden work on the CPU thread that issued the draw. Explicit APIs move the knowledge the driver was guessing at into the application, which generally already knows it:

- *Which state combinations exist* is declared up front in pipeline objects, compiled once.
- *Which resources a pass reads and writes* is declared with barriers.
- *Where resources live* is chosen by the application's memory allocator.
- *Command recording* happens on any number of threads into independent command buffers.

The result is draw-call overhead an order of magnitude lower and much more predictable frame times, at the price of a great deal more code and new classes of bugs (missing barriers, use-after-free of GPU resources) that the driver used to prevent.

== Devices, Queues, and Command Buffers

The application selects a physical GPU, creates a logical *device*, and obtains *queues* from queue families: graphics (which can also do compute and copies), compute-only (*async compute*, which can overlap with graphics work), and transfer-only (copy engines that move data over PCIe without occupying the shader cores).

Work is recorded into *command buffers* (Vulkan), *command lists* (Direct3D 12) or *command buffers* with encoders (Metal). Recording is cheap and thread-local, so an engine typically splits a frame into chunks (shadow cascades, opaque geometry, transparency, post-processing) and records them on several worker threads in parallel. Recorded buffers are then *submitted* to a queue in order. Submission is comparatively expensive, so engines batch many command buffers into few submissions.

Command buffers are allocated from *pools* (Vulkan) or *allocators* (Direct3D 12) that are not thread-safe. The standard pattern is one pool per thread per frame in flight, reset wholesale once the GPU has finished that frame.

== Pipeline State Objects

A *pipeline state object* (PSO) bundles the shaders with nearly all fixed-function state: vertex input layout, rasteriser state, depth and stencil tests, blending, render target formats. The driver compiles the whole package into machine code when the PSO is created, not when it is first used.

That moves the hitching problem rather than eliminating it. A game with many materials and permutations can need tens of thousands of PSOs, each taking milliseconds to compile. Creating them lazily at first use reproduces the stutter of older APIs, which is the "shader compilation stutter" familiar from many PC releases. The countermeasures are:

- *Precompile* known PSOs during loading screens, from lists recorded during playtesting.
- *Cache* compiled pipelines on disk (Vulkan pipeline caches, Direct3D 12 pipeline libraries, Metal binary archives), and ship caches where the platform allows.
- *Reduce permutations*: fewer, more general shaders, with dynamic branching where it is cheap.
- *Use dynamic and partial state*: Vulkan's dynamic state and graphics pipeline libraries (and the shader-object extension) let parts of the pipeline be compiled independently and linked quickly.

Consoles avoid most of this because the hardware is fixed and shaders are compiled to final machine code at build time.

== Resource Binding

Shaders access buffers and textures through *descriptors*, small hardware records that describe a resource.

- *Vulkan* groups descriptors into *descriptor sets*, laid out according to *descriptor set layouts* collected in a *pipeline layout*.
- *Direct3D 12* places descriptors in *descriptor heaps* and describes how the shader sees them with a *root signature*, which can also hold small constants and raw buffer addresses directly.
- *Metal* uses *argument buffers*, which are ordinary buffers containing resource references.

The classic model binds a small set of resources per draw, which costs CPU work and limits how many different materials can be drawn together. Modern engines go *bindless*: every texture and buffer in the scene lives in one huge descriptor array (Vulkan descriptor indexing, Direct3D 12 Shader Model 6.6 `ResourceDescriptorHeap`), and each draw passes only integer indices, often through *push constants* or root constants. Bindless is a prerequisite for GPU-driven rendering, where compute shaders decide what to draw, and for ray tracing, where any ray can hit any material.

== Memory Management

Explicit APIs expose GPU memory as a few *memory types* with different properties, typically:

- *Device-local*: fast video memory, not visible to the CPU (or visible only through a small window).
- *Host-visible, coherent*: system memory the GPU reads over PCIe, used for per-frame upload data.
- *Device-local and host-visible*: video memory mapped into the CPU address space. Historically a 256 MB window; with *resizable BAR* the whole of video memory can be mapped, which is useful for streaming but must be written sequentially, since CPU reads from it are very slow.

Allocating a separate block per resource is slow and some drivers limit the number of allocations (Vulkan guarantees only 4096). Applications therefore allocate large blocks and *sub-allocate* resources from them, respecting alignment rules, usually through a library such as AMD's Vulkan Memory Allocator or D3D12MA. Two further techniques matter:

- *Aliasing*: render targets used only within part of a frame (a blur intermediate, a downsampled buffer) can share the same memory if their lifetimes do not overlap. Render graphs compute lifetimes automatically and can save hundreds of megabytes.
- *Residency*: when total demand exceeds video memory, the application (Direct3D 12) or the OS decides what to evict. Exceeding the budget silently pages resources over PCIe and collapses performance.

== Synchronisation

Synchronisation is where explicit APIs are most demanding and most error-prone. The GPU runs commands in a deep pipeline and executes them concurrently; nothing guarantees that a write by one command is visible to a later read unless a dependency says so.

=== Barriers

A *barrier* declares that commands before it in some pipeline stages, with some kinds of memory access, must complete and their writes be made visible before commands after it, in some stages, perform some kinds of access. Images also have *layouts* (optimised internal formats for rendering, sampling, copying or presenting), and transitions between layouts happen inside barriers. Vulkan's synchronization2 form makes the two halves explicit:

```cpp
#include <vulkan/vulkan.h>

// After rendering to `image`, make it readable by fragment shaders as a texture.
void transition_attachment_to_sampled(VkCommandBuffer cmd, VkImage image) {
    const VkImageMemoryBarrier2 barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    const VkDependencyInfo dependency{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dependency);
}
```

Every barrier is a trade-off. Too few and the program has a data race that may only show up on another vendor's GPU or driver version. Too broad (waiting for all stages, flushing all caches) and the GPU drains its pipeline and idles. Direct3D 12's *enhanced barriers* adopted a model close to Vulkan's, replacing the older resource-state model that tracked one "state" per resource.

Barriers are the main reason engines adopt *render graphs*: each pass declares the resources it reads and writes, and the graph compiler derives the minimal barriers, layout transitions, and memory aliasing automatically.

=== CPU-GPU Synchronisation and Frames in Flight

The CPU records frame $n + 1$ while the GPU renders frame $n$. Resources the CPU writes every frame (uniform buffers, command pools, descriptor pools) must therefore exist once per frame in flight, and the CPU must not reuse a copy until the GPU has finished the frame that used it. A *timeline semaphore* (Vulkan) or *fence* (Direct3D 12), a 64-bit counter the GPU increments as work completes, expresses this directly:

```cpp
#include <array>
#include <cstdint>
#include <vulkan/vulkan.h>

constexpr std::uint32_t frames_in_flight = 2;

struct frame_resources {
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    // per-frame upload buffers, descriptor pools, ...
};

class frame_ring {
public:
    frame_ring(VkDevice device, VkQueue queue, VkSemaphore timeline)
        : device_(device), queue_(queue), timeline_(timeline) {}

    // Blocks until the GPU has finished the frame that last used this slot.
    frame_resources& begin_frame() {
        if (frame_number_ >= frames_in_flight) {
            const std::uint64_t wait_value = frame_number_ - frames_in_flight + 1;
            const VkSemaphoreWaitInfo wait{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                .semaphoreCount = 1,
                .pSemaphores = &timeline_,
                .pValues = &wait_value,
            };
            vkWaitSemaphores(device_, &wait, UINT64_MAX);
        }
        frame_resources& f = frames_[frame_number_ % frames_in_flight];
        vkResetCommandPool(device_, f.command_pool, 0);
        return f;
    }

    // Frame k signals the timeline value k + 1 when the GPU completes it.
    void submit(const frame_resources& f) {
        const VkCommandBufferSubmitInfo cmd_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = f.command_buffer,
        };
        const VkSemaphoreSubmitInfo signal{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .semaphore = timeline_,
            .value = frame_number_ + 1,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        };
        const VkSubmitInfo2 submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &cmd_info,
            .signalSemaphoreInfoCount = 1,
            .pSignalSemaphoreInfos = &signal,
        };
        vkQueueSubmit2(queue_, 1, &submit, VK_NULL_HANDLE);
        ++frame_number_;
    }

private:
    VkDevice device_;
    VkQueue queue_;
    VkSemaphore timeline_;
    std::uint64_t frame_number_ = 0;
    std::array<frame_resources, frames_in_flight> frames_{};
};
```

A real frame also waits on the swapchain's "image acquired" semaphore and signals a "render finished" semaphore for presentation. Two frames in flight is the usual choice: one more lets the CPU absorb hitches, but each extra frame adds a frame of input latency.

Resources must also outlive their last GPU use. Deleting a buffer the moment the CPU stops needing it is a use-after-free if a frame still in flight references it. Engines queue deletions and perform them once the timeline value of the last frame that used the resource has been reached.

== Render Passes and Tiled GPUs

Mobile GPUs (and Apple silicon) render in tiles held in on-chip memory (#xref("graphics-and-rendering", "rasterization-pipeline", label: "The Rasterization Pipeline")). The expensive operations are loading a render target from DRAM into tile memory at the start of a pass and storing it back at the end. Explicit APIs expose this through *load and store operations* on each attachment:

- Load `CLEAR` or `DONT_CARE` instead of `LOAD` when the previous contents are not needed.
- Store `DONT_CARE` for attachments needed only within the pass, such as depth used only for testing, or the multisampled buffer after resolving.
- Mark such attachments *transient* (lazily allocated memory in Vulkan, memoryless storage in Metal), so they never get backing memory at all.

On a tiler, getting these right can save gigabytes per second of bandwidth and a corresponding amount of power. Vulkan's subpasses (and the newer dynamic rendering with local read) let a deferred renderer read the G-buffer from tile memory instead of DRAM. On desktop GPUs the same settings matter less but still help, especially for clears and multisample resolves.

== Presentation

The *swapchain* holds the images that are shown on screen. Its *present mode* sets the latency and tearing trade-off:

- *FIFO* (vsync): every frame is shown, with no tearing; when rendering is slower than the display, the frame rate drops to a divisor of the refresh rate unless variable refresh rate is available.
- *Mailbox*: the newest finished frame replaces any queued one; no tearing and lower latency, at the cost of wasted rendering.
- *Immediate*: frames are shown as soon as they are ready; lowest latency, visible tearing.

Frame pacing (presenting at even intervals) matters as much as average frame rate. Latency-reduction systems (NVIDIA Reflex, AMD Anti-Lag) delay the start of CPU work so that input is sampled as late as possible before the GPU needs the frame.

== Debugging

Explicit APIs do little validation at runtime for speed. Development builds therefore enable the *validation layers* (Vulkan) or the *debug layer* (Direct3D 12), which check every call against the specification, and *synchronisation validation*, which detects missing barriers by tracking accesses. GPU-assisted validation instruments shaders to catch out-of-bounds descriptor indexing in bindless code.

Frame debuggers (RenderDoc, PIX, Xcode's Metal debugger, Nsight Graphics) capture a frame and let you inspect every command, resource and pipeline state. The hardest failures are *device lost* (Vulkan) or *device removed* (Direct3D 12): the GPU hung or faulted, and the driver reset it. Write *breadcrumbs* (markers written into a buffer by the GPU as it passes points in the command stream, using Direct3D 12's DRED or Vulkan's vendor checkpoint extensions) so that after a crash you know which command the GPU was executing. Without them, a hang in a shipped game is nearly impossible to diagnose.

== Choosing an API

#table(
  columns: 3,
  [*API*], [*Platforms*], [*Character*],
  [Vulkan], [Windows, Linux, Android, Switch; macOS and iOS through MoltenVK], [Most explicit; extension-driven; verbose],
  [Direct3D 12], [Windows, Xbox], [Explicit; strong tooling (PIX); DirectX Raytracing and mesh shaders first],
  [Metal], [Apple platforms], [Somewhat less explicit (optional automatic hazard tracking); designed for tiled GPUs],
  [WebGPU], [Browsers, and natively through wgpu and Dawn], [Explicit object model with validation and automatic barriers; safe for the web],
)

Engines typically write a *render hardware interface* over two or three of these, so that rendering code is written once and the backends deal with each API's specifics.

== Further Reading

The Khronos Group (2024). "Vulkan 1.3 Specification." Synchronization and Cache Control chapter.

Microsoft (2024). "Direct3D 12 Programming Guide." Microsoft Learn documentation.

Apple (2024). "Metal Programming Guide." Apple Developer documentation.

Sellers, G., Kessenich, J. (2016). "Vulkan Programming Guide." Addison-Wesley.

O'Donnell, Y. (2017). "FrameGraph: Extensible Rendering Architecture in Frostbite." Game Developers Conference.

AMD (2024). "Vulkan Memory Allocator Documentation." GPUOpen reference.

Arm (2024). "Arm GPU Best Practices Developer Guide." Arm Developer documentation.

W3C (2024). "WebGPU." W3C Candidate Recommendation specification.

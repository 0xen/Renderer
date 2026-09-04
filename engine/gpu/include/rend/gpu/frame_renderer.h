#pragma once

#include "rend/core/result.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

typedef struct VkCommandPool_T* VkCommandPool;
typedef struct VkCommandBuffer_T* VkCommandBuffer;
typedef struct VkSemaphore_T* VkSemaphore;
typedef struct VkFence_T* VkFence;
typedef struct VkBuffer_T* VkBuffer;
typedef struct VkDescriptorSet_T* VkDescriptorSet;

namespace rend::gpu {

class Device;
class Image;
class Pipeline;
class Swapchain;

// Mirrors VkDrawIndexedIndirectCommand so callers can fill indirect
// buffers without Vulkan headers.
struct DrawIndexedIndirect {
    std::uint32_t indexCount = 0;
    std::uint32_t instanceCount = 0;
    std::uint32_t firstIndex = 0;
    std::int32_t vertexOffset = 0;
    std::uint32_t firstInstance = 0;
};
static_assert(sizeof(DrawIndexedIndirect) == 20);

// How a DrawBatch reaches the GPU, best first. The caller picks the best
// mode the device's enabled features allow (Device::isEnabled) — the gpu
// layer executes what it is told and never chooses policy.
enum class DrawSubmitMode {
    // vkCmdDrawIndexedIndirectCount: the GPU also reads the draw count, so
    // a compacted entry list needs no CPU round trip. Needs
    // Feature::DrawIndirectCount (+ the Indirect requirements).
    IndirectCount,
    // vkCmdDrawIndexedIndirect over all drawCount entries; hidden objects
    // are instanceCount-0 entries. Needs Feature::MultiDrawIndirect and
    // Feature::DrawIndirectFirstInstance.
    Indirect,
    // One vkCmdDrawIndexed per entry, read from cpuDraws on the CPU at
    // record time. Works on any Vulkan device; visibility changes need a
    // re-record (static recordings must be invalidated), so this is the
    // fallback of last resort.
    Direct,
};

// One indirect-draw submission over the geometry pool: the pool's buffer
// is bound once as vertex + index source, then every entry in the indirect
// buffer draws by offset. Adding/removing objects only rewrites entries.
struct DrawBatch {
    VkBuffer geometry = nullptr; // bound at offset 0 as VB and IB (uint32 indices)
    VkBuffer indirect = nullptr; // DrawIndexedIndirect[drawCount]
    std::uint32_t drawCount = 0;
    DrawSubmitMode mode = DrawSubmitMode::Indirect;
    // IndirectCount mode: buffer holding the uint32 draw count, one region
    // per frame slot (countRegionStride apart); drawCount caps it.
    VkBuffer count = nullptr;
    std::uint64_t countRegionStride = 0;
    // Direct mode: CPU-side copy of the drawCount entries.
    const DrawIndexedIndirect* cpuDraws = nullptr;
    // Shadow pass (optional): the same draw stream rendered depth-only
    // from the light's point of view into shadowMap before the main pass,
    // which then samples it (descriptor bindings 7-9). The light matrix
    // comes from the per-slot light buffer, so static recordings survive
    // a moving sun.
    const Pipeline* shadowPipeline = nullptr;
    const Image* shadowMap = nullptr; // depth attachment + sampled, caller-owned
    // GPU compaction (IndirectCount mode only): a compute pipeline whose
    // shader reads the draw templates (descriptor binding 3), appends
    // visible entries to `indirect` (binding 4) and counts them into
    // `count` (binding 5). Recorded before the render pass: zero the
    // slot's count, dispatch one thread per template, barrier to the
    // indirect read. Null = no compaction pass.
    const Pipeline* cullPipeline = nullptr;
    // Byte distance between per-frame-slot copies of the indirect array
    // inside `indirect`. Non-zero lets the CPU rewrite the slot's region
    // (host-visible buffer) while the other slot's region is in flight —
    // the instanceCount 0/1 toggle path. 0 = one shared region.
    std::uint64_t indirectRegionStride = 0;
    // The camera matrix lives in the bindless table's per-slot camera
    // buffer (binding 6); only the slot index is pushed, so a moving
    // camera never invalidates static recordings.
    VkDescriptorSet descriptors = nullptr;  // bindless table set, bound once if set
};

// Per-frame-recorded baseline frame loop: acquire, record, submit, present,
// with two frames in flight. Deliberately the naive re-record-every-frame
// path — the static command buffer model (docs/ARCHITECTURE.md) is measured
// against this in a later milestone.
class FrameRenderer {
public:
    static constexpr std::uint32_t kFramesInFlight = 2;

    static Result<std::unique_ptr<FrameRenderer>> create(const Device& device, Swapchain& swapchain);
    ~FrameRenderer();

    FrameRenderer(const FrameRenderer&) = delete;
    FrameRenderer& operator=(const FrameRenderer&) = delete;

    // Records and submits one frame. With a batch, the frame renders it
    // depth-tested via indirect draws; without one it draws the pipeline's
    // own geometry (the milestone-6 triangle) with no depth attachment —
    // the pipeline's depthFormat must match. Out-of-date/suboptimal
    // swapchains are recreated transparently.
    Result<void> drawFrame(const Pipeline& pipeline, const DrawBatch* batch = nullptr);

    // Static recording (the milestone-7 experiment): command buffers are
    // recorded once per (frame slot, swapchain image) and reused every
    // frame — per-frame CPU work shrinks to buffer writes + submit. The
    // recordings are invalidated (and lazily rebuilt on the next drawFrame)
    // by a swapchain recreate; visibility changes must go through the
    // indirect buffer, never through re-recording.
    void setStaticRecording(bool enabled);
    bool staticRecording() const { return staticEnabled_; }

    // Per-frame overlay (UI) recorded into its own small command buffer
    // after the scene: the callback runs inside an active dynamic rendering
    // pass on the swapchain image (loadOp LOAD, viewport/scissor set) and
    // records e.g. ImGui draw data. Recorded every frame regardless of
    // static mode — overlay content is inherently dynamic; the prerecorded
    // scene buffers stay untouched. Null disables the pass.
    using OverlayRecorder = std::function<void(VkCommandBuffer)>;
    void setOverlayRecorder(OverlayRecorder recorder) { overlayRecorder_ = std::move(recorder); }

    // Blocks until the current frame slot's previous submission finished,
    // making the slot's per-frame regions (see DrawBatch::
    // indirectRegionStride) safe to write. drawFrame's own wait then
    // returns immediately.
    Result<void> waitFrameSlot();
    std::uint32_t frameSlot() const { return frameIndex_; }

    // CPU cost counters since the last take; the record/reset time is the
    // number the static-vs-rerecord experiment compares.
    struct Stats {
        std::uint64_t frames = 0;
        std::uint64_t recordMicros = 0; // per-frame reset+record CPU time
        std::uint64_t prerecords = 0;   // static recordings built (amortized)
    };
    Stats takeStats();

    // Requests a swapchain rebuild at the next frame; a zero size parks the
    // loop until a real size arrives (minimized window).
    void resize(std::uint32_t width, std::uint32_t height);

    // The destructor only touches device-owned objects, so the swapchain may
    // be destroyed first (it retires presents still waiting on the per-image
    // semaphores destroyed here).
    void waitIdle() const;

    void setClearColor(float r, float g, float b, float a = 1.0f) { clearColor_ = {r, g, b, a}; }

private:
    FrameRenderer() = default;

    Result<void> createSyncObjects();
    Result<void> createImageSemaphores();
    void destroyImageSemaphores();
    Result<void> createDepthBuffer();
    Result<void> recreateSwapchain();
    Result<void> waitForFence(VkFence fence, const char* what) const;
    Result<void> record(VkCommandBuffer cmd, std::uint32_t imageIndex, std::uint32_t slot,
                        const Pipeline& pipeline, const DrawBatch* batch, bool reusable) const;
    Result<void> prerecordStatic(const Pipeline& pipeline, const DrawBatch* batch);
    void invalidateStatic();
    Result<void> recordOverlay(VkCommandBuffer cmd, std::uint32_t imageIndex) const;

    struct FrameData {
        VkCommandBuffer commandBuffer = nullptr;
        // The per-frame UI/present-transition tail after the scene buffer.
        VkCommandBuffer overlayCommandBuffer = nullptr;
        VkSemaphore imageAvailable = nullptr;
        VkFence inFlight = nullptr;
    };

    const Device* device_ = nullptr;
    Swapchain* swapchain_ = nullptr;
    VkCommandPool commandPool_ = nullptr;
    std::array<FrameData, kFramesInFlight> frames_{};
    // One per swapchain image, not per frame in flight: presentation may
    // still be reading an image's semaphore when its frame slot comes round.
    std::vector<VkSemaphore> renderFinished_;
    // Depth buffer at swapchain extent; recreated with it. One is enough
    // for both frames in flight: rendering is serialized by the barriers.
    std::unique_ptr<Image> depth_;

    // Static-mode recordings, indexed [slot * imageCount + imageIndex];
    // empty while invalid. A (slot, image) pair is never in flight twice,
    // so the buffers need no simultaneous-use flag.
    std::vector<VkCommandBuffer> staticBuffers_;
    bool staticEnabled_ = false;
    bool staticValid_ = false;

    Stats stats_{};
    OverlayRecorder overlayRecorder_;

    std::array<float, 4> clearColor_{0.02f, 0.02f, 0.04f, 1.0f};
    std::uint32_t frameIndex_ = 0;
    std::uint32_t pendingWidth_ = 0;
    std::uint32_t pendingHeight_ = 0;
    bool resizeRequested_ = false;
};

} // namespace rend::gpu

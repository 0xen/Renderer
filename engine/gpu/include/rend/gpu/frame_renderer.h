#pragma once

#include "rend/core/result.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

typedef struct VkCommandPool_T* VkCommandPool;
typedef struct VkCommandBuffer_T* VkCommandBuffer;
typedef struct VkSemaphore_T* VkSemaphore;
typedef struct VkFence_T* VkFence;
typedef struct VkBuffer_T* VkBuffer;

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

// One indirect-draw submission over the geometry pool: the pool's buffer
// is bound once as vertex + index source, then every entry in the indirect
// buffer draws by offset. Adding/removing objects only rewrites entries.
struct DrawBatch {
    VkBuffer geometry = nullptr; // bound at offset 0 as VB and IB (uint32 indices)
    VkBuffer indirect = nullptr; // DrawIndexedIndirect[drawCount]
    std::uint32_t drawCount = 0;
    std::array<float, 16> viewProj{}; // pushed to the pipeline (64 bytes)
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
    Result<void> record(VkCommandBuffer cmd, std::uint32_t imageIndex, const Pipeline& pipeline,
                        const DrawBatch* batch) const;

    struct FrameData {
        VkCommandBuffer commandBuffer = nullptr;
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

    std::array<float, 4> clearColor_{0.02f, 0.02f, 0.04f, 1.0f};
    std::uint32_t frameIndex_ = 0;
    std::uint32_t pendingWidth_ = 0;
    std::uint32_t pendingHeight_ = 0;
    bool resizeRequested_ = false;
};

} // namespace rend::gpu

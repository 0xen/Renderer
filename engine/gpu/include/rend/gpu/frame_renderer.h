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

namespace rend::gpu {

class Device;
class Pipeline;
class Swapchain;

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

    // Records and submits one frame drawing the pipeline's geometry.
    // Out-of-date/suboptimal swapchains are recreated transparently.
    Result<void> drawFrame(const Pipeline& pipeline);

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
    Result<void> recreateSwapchain();
    Result<void> waitForFence(VkFence fence, const char* what) const;
    Result<void> record(VkCommandBuffer cmd, std::uint32_t imageIndex, const Pipeline& pipeline) const;

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

    std::array<float, 4> clearColor_{0.02f, 0.02f, 0.04f, 1.0f};
    std::uint32_t frameIndex_ = 0;
    std::uint32_t pendingWidth_ = 0;
    std::uint32_t pendingHeight_ = 0;
    bool resizeRequested_ = false;
};

} // namespace rend::gpu

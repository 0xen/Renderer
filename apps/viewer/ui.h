#pragma once

#include <cstdint>
#include <memory>

typedef struct VkCommandBuffer_T* VkCommandBuffer;

namespace rend::gpu {
class Device;
class Instance;
class Swapchain;
} // namespace rend::gpu

namespace viewer {

// Dear ImGui debug UI: lives entirely in the viewer app (the engine only
// provides the overlay hook on FrameRenderer). Owns the ImGui context and
// the Vulkan backend; styled black/red.
class Ui {
public:
    static std::unique_ptr<Ui> create(const rend::gpu::Instance& instance,
                                      const rend::gpu::Device& device,
                                      const rend::gpu::Swapchain& swapchain);
    ~Ui();

    Ui(const Ui&) = delete;
    Ui& operator=(const Ui&) = delete;

    // Starts the ImGui frame and lays out the overlay widgets (currently
    // the FPS counter). Call once per frame before FrameRenderer::drawFrame.
    void buildFrame(std::uint32_t width, std::uint32_t height, float deltaSeconds);

    // The FrameRenderer overlay recorder: finalizes the ImGui frame and
    // records its draw data. Runs inside an active rendering pass.
    void render(VkCommandBuffer cmd);

private:
    Ui() = default;

    float smoothedFrameSeconds_ = 0.0f;
    bool frameBuilt_ = false;
};

} // namespace viewer

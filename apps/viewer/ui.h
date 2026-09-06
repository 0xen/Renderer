#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

typedef struct VkCommandBuffer_T* VkCommandBuffer;

namespace rend::gpu {
class Device;
class Instance;
class Swapchain;
} // namespace rend::gpu

namespace rend::platform {
struct Event;
} // namespace rend::platform

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

    // Feeds a platform event into ImGui (mouse position, buttons, wheel)
    // so future interactive widgets work. Call for every pumped event.
    void handleEvent(const rend::platform::Event& event);

    // Scene asset-loading progress for the loading bar. hideScene = wait
    // mode: a fullscreen cover with a centered bar replaces the scene
    // until loading completes; otherwise a small inline bar overlays the
    // (already visible, streaming) scene.
    struct LoadingStatus {
        bool hideScene = false;
        std::uint32_t done = 0;
        std::uint32_t total = 0;
    };

    // Starts the ImGui frame and lays out the debug panel (FPS counter,
    // frame-rate history graph, and — when vsync is non-null — a VSync
    // checkbox bound to it). loading non-null draws the loading bar (see
    // LoadingStatus). Call once per frame before FrameRenderer::drawFrame;
    // the caller reacts to a toggled *vsync.
    void buildFrame(std::uint32_t width, std::uint32_t height, float deltaSeconds,
                    bool* vsync = nullptr, const LoadingStatus* loading = nullptr);

    // The FrameRenderer overlay recorder: finalizes the ImGui frame and
    // records its draw data. Runs inside an active rendering pass.
    void render(VkCommandBuffer cmd);

private:
    Ui() = default;

    float smoothedFrameSeconds_ = 0.0f;
    // Instantaneous FPS ring buffer feeding the debug panel's graph.
    std::array<float, 180> fpsHistory_{};
    std::size_t fpsHistoryOffset_ = 0;
    // Total tracked GPU memory (MiB) per frame — the memory graph's ring.
    std::array<float, 180> memoryHistory_{};
    std::size_t memoryHistoryOffset_ = 0;
    bool frameBuilt_ = false;
};

} // namespace viewer

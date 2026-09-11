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

    // Cull-pass counters for the debug panel's culling section (from the
    // viewer's fenced count-buffer readback — kFramesInFlight frames
    // late, stats only). Null = no GPU culling active (lower draw tiers).
    struct CullStats {
        std::uint32_t drawsInView = 0;  // opaque + transparent emitted
        std::uint32_t drawsLive = 0;    // shadow stream = every live entry
        std::uint32_t triangles = 0;    // post-LOD, both scene streams
        std::uint32_t occluded = 0;     // dropped by the occlusion test
    };

    // Starts the ImGui frame and lays out the debug panel (FPS counter,
    // frame-rate history graph, and — when vsync is non-null — a VSync
    // checkbox bound to it). cull non-null adds the culling section
    // (draws, triangle count + history graph, occluded). loading non-null
    // draws the loading bar (see LoadingStatus). Call once per frame
    // before FrameRenderer::drawFrame; the caller reacts to a toggled
    // *vsync.
    void buildFrame(std::uint32_t width, std::uint32_t height, float deltaSeconds,
                    bool* vsync = nullptr, const LoadingStatus* loading = nullptr,
                    const CullStats* cull = nullptr);

    // The FrameRenderer overlay recorder: finalizes the ImGui frame and
    // records its draw data. Runs inside an active rendering pass.
    void render(VkCommandBuffer cmd);

private:
    Ui() = default;

    float smoothedFrameSeconds_ = 0.0f;
    // The numeric readouts refresh at 4 Hz (kTextRefreshSeconds) so they
    // are readable instead of flickering every frame; these hold the
    // values displayed between refreshes. The history GRAPHS stay
    // per-frame — their whole point is raw variation.
    static constexpr float kTextRefreshSeconds = 0.25f;
    float textRefreshTimer_ = kTextRefreshSeconds; // expired: first frame fills
    float shownFrameSeconds_ = 0.0f;
    double shownMemoryMiB_ = 0.0;
    std::uint32_t shownMemoryCount_ = 0;
    double shownDeviceMiB_ = 0.0, shownHostMiB_ = 0.0, shownImageMiB_ = 0.0;
    std::uint32_t shownDeviceCount_ = 0, shownHostCount_ = 0, shownImageCount_ = 0;
    CullStats shownCull_{};
    // Instantaneous FPS ring buffer feeding the debug panel's graph.
    std::array<float, 180> fpsHistory_{};
    std::size_t fpsHistoryOffset_ = 0;
    // Total tracked GPU memory (MiB) per frame — the memory graph's ring.
    std::array<float, 180> memoryHistory_{};
    std::size_t memoryHistoryOffset_ = 0;
    // Rendered triangles (millions) per frame — the culling graph's ring.
    std::array<float, 180> triangleHistory_{};
    std::size_t triangleHistoryOffset_ = 0;
    bool frameBuilt_ = false;
};

} // namespace viewer

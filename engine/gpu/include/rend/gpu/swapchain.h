#pragma once

#include "rend/core/result.h"
#include "rend/gpu/format.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace rend::gpu {

class Device;
class Image;
class Instance;

struct SwapchainDesc {
    // The presentation surface the platform layer created for this
    // backend (VkSurfaceKHR under Vulkan); ownership transfers to the
    // Swapchain.
    void* nativeSurface = nullptr;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    // Request a composite-alpha mode that lets the desktop show through
    // (BorderlessTransparent targets); falls back to opaque with a warning
    // when the driver doesn't offer one.
    bool transparent = false;
    bool vsync = true;
};

// Owns the surface, the API swapchain and the per-image views. Recreated
// on resize via recreate(); images/format stay queryable. The images are
// exposed as non-owning gpu::Image wrappers so the frame renderer and
// frame passes never see API handles.
class Swapchain {
public:
    static Result<std::unique_ptr<Swapchain>> create(const Instance& instance, const Device& device,
                                                     const SwapchainDesc& desc);
    virtual ~Swapchain() = default;

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    virtual Result<void> recreate(std::uint32_t width, std::uint32_t height) = 0;

    // Present-mode preference; takes effect on the next recreate() (the
    // caller triggers one, e.g. via FrameRenderer::resize).
    bool vsync() const { return vsync_; }
    void setVsync(bool vsync) { vsync_ = vsync; }

    Format imageFormat() const { return format_; }
    std::uint32_t width() const { return width_; }
    std::uint32_t height() const { return height_; }
    std::uint32_t imageCount() const { return static_cast<std::uint32_t>(wrapped_.size()); }
    // Swapchain image `index` as a (non-owning) Image: what frame passes
    // and the frame renderer render into and barrier.
    const Image& image(std::uint32_t index) const { return *wrapped_[index]; }

protected:
    Swapchain() = default;

    std::vector<std::unique_ptr<Image>> wrapped_;
    Format format_ = Format::Undefined;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    bool transparent_ = false;
    bool vsync_ = true;
};

} // namespace rend::gpu

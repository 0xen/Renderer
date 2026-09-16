#pragma once

#include "rend/core/result.h"
#include "rend/gpu/format.h"

#include <cstdint>
#include <memory>
#include <vector>

typedef struct VkSurfaceKHR_T* VkSurfaceKHR;
typedef struct VkSwapchainKHR_T* VkSwapchainKHR;
typedef struct VkImage_T* VkImage;
typedef struct VkImageView_T* VkImageView;

namespace rend::gpu {

class Instance;
class Device;

struct SwapchainDesc {
    VkSurfaceKHR surface = nullptr; // ownership transfers to the Swapchain
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    // Request a composite-alpha mode that lets the desktop show through
    // (BorderlessTransparent targets); falls back to opaque with a warning
    // when the driver doesn't offer one.
    bool transparent = false;
    bool vsync = true;
};

// Owns the surface, the VkSwapchainKHR, and the per-image views.
// Recreated on resize via recreate(); images/format stay queryable.
class Swapchain {
public:
    static Result<std::unique_ptr<Swapchain>> create(const Instance& instance, const Device& device,
                                                     const SwapchainDesc& desc);
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    Result<void> recreate(std::uint32_t width, std::uint32_t height);

    // Present-mode preference; takes effect on the next recreate() (the
    // caller triggers one, e.g. via FrameRenderer::resize).
    bool vsync() const { return vsync_; }
    void setVsync(bool vsync) { vsync_ = vsync; }

    VkSwapchainKHR handle() const { return swapchain_; }
    Format imageFormat() const { return format_; }
    std::uint32_t width() const { return width_; }
    std::uint32_t height() const { return height_; }
    const std::vector<VkImage>& images() const { return images_; }
    const std::vector<VkImageView>& imageViews() const { return views_; }

private:
    Swapchain() = default;
    Result<void> build(std::uint32_t width, std::uint32_t height, VkSwapchainKHR old);
    void destroyViews();

    const Instance* instance_ = nullptr;
    const Device* device_ = nullptr;
    VkSurfaceKHR surface_ = nullptr;
    VkSwapchainKHR swapchain_ = nullptr;
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
    Format format_ = Format::Undefined;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    bool transparent_ = false;
    bool vsync_ = true;
};

} // namespace rend::gpu

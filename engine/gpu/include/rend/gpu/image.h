#pragma once

#include "rend/core/result.h"
#include "rend/gpu/format.h"

#include <array>
#include <cstdint>
#include <memory>

typedef struct VkImage_T* VkImage;
typedef struct VkImageView_T* VkImageView;
typedef struct VkDeviceMemory_T* VkDeviceMemory;

namespace rend::gpu {

class Device;

// VkImageUsageFlags bits callers need without including Vulkan headers.
inline constexpr std::uint32_t kImageUsageTransferSrc = 0x1;
inline constexpr std::uint32_t kImageUsageTransferDst = 0x2;
inline constexpr std::uint32_t kImageUsageSampled = 0x4;
inline constexpr std::uint32_t kImageUsageColorAttachment = 0x10;
inline constexpr std::uint32_t kImageUsageDepthAttachment = 0x20;

struct ImageDesc {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    Format format = Format::Undefined;
    std::uint32_t usage = 0;  // VkImageUsageFlags
    std::uint32_t mipLevels = 1;
    bool depth = false; // view aspect: depth instead of color
    // Cubemap: six square layers with a cube sampled view plus one 2D
    // render view per face (faceView) — reflection probes render into the
    // faces and shaders sample the cube.
    bool cube = false;
};

// One VkImage plus its allocation and a single full view — the image
// sibling of Buffer (see ARCHITECTURE.md base object classes). 2D,
// single-layer, or a six-face cube (desc.cube).
class Image {
public:
    static Result<std::unique_ptr<Image>> create(const Device& device, const ImageDesc& desc);
    // Backend-internal: a non-owning view of an image the backend does
    // not allocate itself (swapchain images). Destroys nothing.
    static std::unique_ptr<Image> wrapExternal(VkImage image, VkImageView view, Format format,
                                               std::uint32_t width, std::uint32_t height);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    VkImage handle() const { return image_; }
    VkImageView view() const { return view_; }
    // Cube images only: 2D render view of one face's mip 0.
    VkImageView faceView(std::uint32_t face) const { return faceViews_[face]; }
    Format format() const { return format_; }
    std::uint32_t width() const { return width_; }
    std::uint32_t height() const { return height_; }
    std::uint32_t mipLevels() const { return mipLevels_; }
    std::uint32_t layerCount() const { return layerCount_; }

private:
    Image() = default;

    const Device* device_ = nullptr;
    VkImage image_ = nullptr;
    VkDeviceMemory memory_ = nullptr;
    VkImageView view_ = nullptr;
    std::array<VkImageView, 6> faceViews_{};
    Format format_ = Format::Undefined;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uint32_t mipLevels_ = 1;
    std::uint32_t layerCount_ = 1;
    // Alignment-padded allocation cost reported to the MemoryTracker; the
    // destructor releases the same figure.
    std::uint64_t allocatedBytes_ = 0;
    bool owned_ = true;
};

} // namespace rend::gpu

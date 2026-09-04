#pragma once

#include "rend/core/result.h"

#include <cstdint>
#include <memory>

typedef struct VkImage_T* VkImage;
typedef struct VkImageView_T* VkImageView;
typedef struct VkDeviceMemory_T* VkDeviceMemory;

namespace rend::gpu {

class Device;

struct ImageDesc {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t format = 0; // VkFormat
    std::uint32_t usage = 0;  // VkImageUsageFlags
    std::uint32_t mipLevels = 1;
    bool depth = false; // view aspect: depth instead of color
};

// One VkImage plus its allocation and a single full view — the image
// sibling of Buffer (see ARCHITECTURE.md base object classes). 2D,
// single-mip, single-layer for now; extended when textures land.
class Image {
public:
    static Result<std::unique_ptr<Image>> create(const Device& device, const ImageDesc& desc);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    VkImage handle() const { return image_; }
    VkImageView view() const { return view_; }
    std::uint32_t format() const { return format_; }
    std::uint32_t width() const { return width_; }
    std::uint32_t height() const { return height_; }
    std::uint32_t mipLevels() const { return mipLevels_; }

private:
    Image() = default;

    const Device* device_ = nullptr;
    VkImage image_ = nullptr;
    VkDeviceMemory memory_ = nullptr;
    VkImageView view_ = nullptr;
    std::uint32_t format_ = 0;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uint32_t mipLevels_ = 1;
};

} // namespace rend::gpu

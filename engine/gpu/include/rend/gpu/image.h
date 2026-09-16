#pragma once

#include "rend/core/result.h"
#include "rend/gpu/format.h"

#include <cstdint>
#include <memory>

namespace rend::gpu {

class Device;

// Image usage bits (backend-neutral vocabulary; each backend maps them).
inline constexpr std::uint32_t kImageUsageTransferSrc = 0x1;
inline constexpr std::uint32_t kImageUsageTransferDst = 0x2;
inline constexpr std::uint32_t kImageUsageSampled = 0x4;
inline constexpr std::uint32_t kImageUsageColorAttachment = 0x10;
inline constexpr std::uint32_t kImageUsageDepthAttachment = 0x20;

struct ImageDesc {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    Format format = Format::Undefined;
    std::uint32_t usage = 0; // kImageUsage* bits
    std::uint32_t mipLevels = 1;
    bool depth = false; // view aspect: depth instead of color
    // Cubemap: six square layers with a cube sampled view plus one 2D
    // render view per face — reflection probes render into the faces and
    // shaders sample the cube.
    bool cube = false;
};

// One GPU image plus its allocation and a single full view — the image
// sibling of Buffer (see ARCHITECTURE.md base object classes). 2D,
// single-layer, or a six-face cube (desc.cube).
class Image {
public:
    static Result<std::unique_ptr<Image>> create(const Device& device, const ImageDesc& desc);
    virtual ~Image() = default;

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Format format() const { return format_; }
    std::uint32_t width() const { return width_; }
    std::uint32_t height() const { return height_; }
    std::uint32_t mipLevels() const { return mipLevels_; }
    std::uint32_t layerCount() const { return layerCount_; }

protected:
    Image() = default;

    Format format_ = Format::Undefined;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::uint32_t mipLevels_ = 1;
    std::uint32_t layerCount_ = 1;
};

} // namespace rend::gpu

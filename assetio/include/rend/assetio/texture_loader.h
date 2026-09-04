#pragma once

#include "rend/core/result.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace rend::assetio {

// Decoded pixels, always tightly-packed RGBA8. Color-space handling (sRGB
// vs linear) is the consumer's choice of GPU format, not encoded here.
struct TextureData {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> rgba;
};

// Decodes an image file (PNG/JPEG/TGA/BMP... — whatever stb_image reads).
Result<TextureData> loadTexture(const std::filesystem::path& file);

} // namespace rend::assetio

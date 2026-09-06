#pragma once

#include "rend/core/result.h"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace rend::assetio {

// How TextureData.bytes is encoded. Backend-agnostic vocabulary: assetio
// describes what the file held, and each renderer backend maps it to its
// own format enum (e.g. Bc7 -> VK_FORMAT_BC7_*_BLOCK on Vulkan).
enum class TextureEncoding {
    Rgba8, // decoded, tightly-packed RGBA8, single level (mips empty)
    Bc7,   // BC7 blocks, pre-generated mip chain described by mips
};

// One stored mip level inside TextureData.bytes.
struct TextureMipLevel {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint64_t byteOffset = 0;
    std::uint64_t byteLength = 0;
};

// Texture payload plus enough metadata for any consumer to upload it.
// - Rgba8 (stb-decoded files): bytes = one tightly-packed level, mips
//   empty; color space is the CONSUMER's choice (the file doesn't say).
// - Block-compressed (DDS): bytes = every stored level back to back, mips
//   filled, srgb = the color space the FILE declares (authoritative).
struct TextureData {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    TextureEncoding encoding = TextureEncoding::Rgba8;
    bool srgb = false;
    std::vector<TextureMipLevel> mips;
    std::vector<std::uint8_t> bytes;
};

// Loads an image file. Dispatches on extension: .dds parses the DDS
// container (BC7 with a DX10 header only, the modern compressor output);
// everything else decodes through stb_image (PNG/JPEG/TGA/BMP...).
Result<TextureData> loadTexture(const std::filesystem::path& file);

} // namespace rend::assetio

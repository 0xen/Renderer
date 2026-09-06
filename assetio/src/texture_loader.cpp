#include "rend/assetio/texture_loader.h"

#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STBI_WINDOWS_UTF8
#include <stb_image.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cstring>
#include <format>
#include <fstream>

namespace rend::assetio {

namespace {

// DDS container layout: 4-byte magic, 124-byte DDS_HEADER, and (for
// modern compressors) a 20-byte DX10 extension header, then the payload
// levels back to back, largest first.
constexpr std::size_t kDdsHeaderSize = 4 + 124;
constexpr std::size_t kDx10HeaderSize = 20;
constexpr std::size_t kDdsDataOffset = kDdsHeaderSize + kDx10HeaderSize;
constexpr std::uint32_t kDxgiBc7Unorm = 98;
constexpr std::uint32_t kDxgiBc7UnormSrgb = 99;

std::uint32_t readU32(const std::vector<std::uint8_t>& b, std::size_t offset) {
    std::uint32_t v = 0;
    std::memcpy(&v, b.data() + offset, sizeof(v));
    return v;
}

// BC7: 4x4 texel blocks, 16 bytes each; dimensions round up to whole
// blocks at every level.
std::uint64_t bc7LevelBytes(std::uint32_t width, std::uint32_t height) {
    return std::uint64_t{(width + 3) / 4} * ((height + 3) / 4) * 16;
}

Result<TextureData> loadDds(const std::filesystem::path& file) {
    std::ifstream in(file, std::ios::binary | std::ios::ate);
    if (!in) {
        return Error{std::format("Failed to open '{}'", file.string())};
    }
    const auto fileSize = static_cast<std::size_t>(in.tellg());
    if (fileSize < kDdsDataOffset) {
        return Error{std::format("'{}' too small for a DX10 DDS", file.string())};
    }
    std::vector<std::uint8_t> raw(fileSize);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(fileSize));

    if (std::memcmp(raw.data(), "DDS ", 4) != 0 || readU32(raw, 4) != 124) {
        return Error{std::format("'{}' is not a DDS file", file.string())};
    }
    const std::uint32_t height = readU32(raw, 12);
    const std::uint32_t width = readU32(raw, 16);
    const std::uint32_t mipCount = std::max(readU32(raw, 28), 1u);
    if (std::memcmp(raw.data() + 84, "DX10", 4) != 0) {
        return Error{std::format("'{}': only DX10-header DDS is supported (legacy fourCC "
                                 "containers are not)",
                                 file.string())};
    }
    const std::uint32_t dxgiFormat = readU32(raw, 128);
    if (dxgiFormat != kDxgiBc7Unorm && dxgiFormat != kDxgiBc7UnormSrgb) {
        return Error{std::format("'{}': unsupported DXGI format {} (BC7 only)", file.string(),
                                 dxgiFormat)};
    }
    const std::uint32_t arraySize = readU32(raw, 136);
    if (arraySize > 1) {
        return Error{std::format("'{}': DDS arrays/cubes not supported", file.string())};
    }

    TextureData out;
    out.width = width;
    out.height = height;
    out.encoding = TextureEncoding::Bc7;
    out.srgb = dxgiFormat == kDxgiBc7UnormSrgb;
    out.mips.reserve(mipCount);
    std::uint64_t offset = 0;
    for (std::uint32_t level = 0; level < mipCount; ++level) {
        const std::uint32_t w = std::max(width >> level, 1u);
        const std::uint32_t h = std::max(height >> level, 1u);
        const std::uint64_t size = bc7LevelBytes(w, h);
        out.mips.push_back({.width = w, .height = h, .byteOffset = offset, .byteLength = size});
        offset += size;
    }
    if (kDdsDataOffset + offset > fileSize) {
        return Error{std::format("'{}': truncated DDS ({} payload bytes, {} expected)",
                                 file.string(), fileSize - kDdsDataOffset, offset)};
    }
    out.bytes.assign(raw.begin() + kDdsDataOffset,
                     raw.begin() + static_cast<std::ptrdiff_t>(kDdsDataOffset + offset));
    return out;
}

} // namespace

Result<TextureData> loadTexture(const std::filesystem::path& file) {
    std::filesystem::path ext = file.extension();
    if (ext == ".dds" || ext == ".DDS") {
        return loadDds(file);
    }

    int width = 0, height = 0, channels = 0;
    stbi_uc* pixels =
        stbi_load(file.string().c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!pixels) {
        return Error{std::format("Texture decode failed for '{}': {}", file.string(),
                                 stbi_failure_reason())};
    }

    TextureData out;
    out.width = static_cast<std::uint32_t>(width);
    out.height = static_cast<std::uint32_t>(height);
    out.bytes.resize(static_cast<std::size_t>(width) * height * 4);
    std::memcpy(out.bytes.data(), pixels, out.bytes.size());
    stbi_image_free(pixels);
    return out;
}

} // namespace rend::assetio

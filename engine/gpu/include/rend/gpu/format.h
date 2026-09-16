#pragma once

#include <cstdint>

namespace rend::gpu {

// Backend-neutral pixel/attribute format vocabulary. This is the ONLY
// format type that crosses the gpu-layer boundary: the app, the pipeline
// XML and the frame renderer speak Format; each backend maps it to its
// own enum (VkFormat, DXGI_FORMAT) internally. The numeric values are
// arbitrary identifiers — never pass them to a graphics API directly.
enum class Format : std::uint32_t {
    Undefined = 0,
    R8G8B8A8Unorm = 37,
    R8G8B8A8Srgb = 43,
    B8G8R8A8Unorm = 44,
    B8G8R8A8Srgb = 50,
    R16G16B16A16Sfloat = 97,
    R32Sfloat = 100,
    R32G32Sfloat = 103,
    R32G32B32Sfloat = 106,
    D32Sfloat = 126,
    Bc7Unorm = 145,
    Bc7Srgb = 146,
};

inline constexpr bool isDepthFormat(Format format) {
    return format == Format::D32Sfloat;
}

// Names for logs.
inline constexpr const char* formatName(Format format) {
    switch (format) {
    case Format::Undefined: return "Undefined";
    case Format::R8G8B8A8Unorm: return "R8G8B8A8Unorm";
    case Format::R8G8B8A8Srgb: return "R8G8B8A8Srgb";
    case Format::B8G8R8A8Unorm: return "B8G8R8A8Unorm";
    case Format::B8G8R8A8Srgb: return "B8G8R8A8Srgb";
    case Format::R16G16B16A16Sfloat: return "R16G16B16A16Sfloat";
    case Format::R32Sfloat: return "R32Sfloat";
    case Format::R32G32Sfloat: return "R32G32Sfloat";
    case Format::R32G32B32Sfloat: return "R32G32B32Sfloat";
    case Format::D32Sfloat: return "D32Sfloat";
    case Format::Bc7Unorm: return "Bc7Unorm";
    case Format::Bc7Srgb: return "Bc7Srgb";
    }
    return "?";
}

// Short constant aliases kept for call sites that predate the enum.
inline constexpr Format kFormatR8G8B8A8Unorm = Format::R8G8B8A8Unorm;
inline constexpr Format kFormatR8G8B8A8Srgb = Format::R8G8B8A8Srgb;
inline constexpr Format kFormatR16G16B16A16Sfloat = Format::R16G16B16A16Sfloat;
inline constexpr Format kFormatR32Sfloat = Format::R32Sfloat;
inline constexpr Format kFormatR32G32Sfloat = Format::R32G32Sfloat;
inline constexpr Format kFormatR32G32B32Sfloat = Format::R32G32B32Sfloat;
inline constexpr Format kFormatD32Sfloat = Format::D32Sfloat;

} // namespace rend::gpu

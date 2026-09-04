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

namespace rend::assetio {

Result<TextureData> loadTexture(const std::filesystem::path& file) {
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
    out.rgba.resize(static_cast<std::size_t>(width) * height * 4);
    std::memcpy(out.rgba.data(), pixels, out.rgba.size());
    stbi_image_free(pixels);
    return out;
}

} // namespace rend::assetio

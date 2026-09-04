#include "rend/gpu/shader.h"

#include "rend/core/log.h"
#include "rend/gpu/device.h"

#include <volk.h>

#include <cstdint>
#include <fstream>
#include <vector>

namespace rend::gpu {

Result<std::unique_ptr<Shader>> Shader::createFromFile(const Device& device,
                                                       const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return Error{std::format("Shader file not found: {}", path.string())};
    }
    const std::streamoff size = file.tellg();
    if (size <= 0 || (size % 4) != 0) {
        return Error{std::format("Shader file is not valid SPIR-V (size {}): {}",
                                 static_cast<long long>(size), path.string())};
    }

    std::vector<std::uint32_t> code(static_cast<std::size_t>(size) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(code.data()), size);
    if (!file) {
        return Error{std::format("Failed to read shader file: {}", path.string())};
    }

    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = static_cast<std::size_t>(size);
    info.pCode = code.data();

    VkShaderModule handle = VK_NULL_HANDLE;
    if (VkResult r = vkCreateShaderModule(device.handle(), &info, nullptr, &handle); r != VK_SUCCESS) {
        return Error{std::format("vkCreateShaderModule failed ({}) for {}", static_cast<int>(r),
                                 path.string())};
    }

    auto shader = std::unique_ptr<Shader>(new Shader());
    shader->device_ = &device;
    shader->module_ = handle;
    log::trace("Shader loaded: {} ({} bytes)", path.filename().string(), static_cast<long long>(size));
    return shader;
}

Shader::~Shader() {
    if (device_ && module_ != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device_->handle(), module_, nullptr);
    }
}

} // namespace rend::gpu

#pragma once

#include "rend/core/result.h"

#include <filesystem>
#include <memory>

typedef struct VkShaderModule_T* VkShaderModule;

namespace rend::gpu {

class Device;

// One SPIR-V module, nothing more. Shaders are compiled offline (dxc, see
// assets/CMakeLists.txt); this class only uploads the bytes and owns the
// handle. It knows nothing about stages, pipelines or materials.
class Shader {
public:
    static Result<std::unique_ptr<Shader>> createFromFile(const Device& device,
                                                          const std::filesystem::path& path);
    ~Shader();

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    VkShaderModule handle() const { return module_; }

private:
    Shader() = default;

    const Device* device_ = nullptr;
    VkShaderModule module_ = nullptr;
};

} // namespace rend::gpu

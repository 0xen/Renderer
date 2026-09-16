#pragma once

#include "rend/core/result.h"

#include <filesystem>
#include <memory>

namespace rend::gpu {

class Device;

// One compiled shader module, nothing more. Shaders are compiled offline
// (dxc, see assets/CMakeLists.txt) into the backend's binary format; this
// class only uploads the bytes and owns the handle. It knows nothing
// about stages, pipelines or materials.
class Shader {
public:
    static Result<std::unique_ptr<Shader>> createFromFile(const Device& device,
                                                          const std::filesystem::path& path);
    virtual ~Shader() = default;

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

protected:
    Shader() = default;
};

} // namespace rend::gpu

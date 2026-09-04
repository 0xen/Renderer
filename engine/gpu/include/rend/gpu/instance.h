#pragma once

#include "rend/core/result.h"

#include <memory>
#include <string>
#include <vector>

// Forward declarations so gpu headers never force Vulkan headers on consumers.
typedef struct VkInstance_T* VkInstance;
typedef struct VkDebugUtilsMessengerEXT_T* VkDebugUtilsMessengerEXT;

namespace rend::gpu {

struct InstanceDesc {
    std::string appName = "Renderer";
    // Validation layers + debug messenger; silently skipped if the layer
    // is not installed (logged as a warning).
    bool enableValidation = true;
    // Platform surface extensions etc., supplied by the caller (the
    // platform layer's Vulkan seam feeds this from milestone 5 on).
    std::vector<const char*> extraExtensions;
};

// Owns volk initialization, the VkInstance, and the debug messenger.
class Instance {
public:
    static Result<std::unique_ptr<Instance>> create(const InstanceDesc& desc);
    ~Instance();

    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;

    VkInstance handle() const { return instance_; }
    bool validationEnabled() const { return messenger_ != nullptr; }
    std::uint32_t apiVersion() const { return apiVersion_; }

private:
    Instance() = default;

    VkInstance instance_ = nullptr;
    VkDebugUtilsMessengerEXT messenger_ = nullptr;
    std::uint32_t apiVersion_ = 0;
};

} // namespace rend::gpu

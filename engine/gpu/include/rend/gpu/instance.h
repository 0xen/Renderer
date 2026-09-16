#pragma once

#include "rend/core/result.h"
#include "rend/gpu/api.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rend::gpu {

struct InstanceDesc {
    std::string appName = "Renderer";
    // Validation layers + debug messenger; silently skipped if the layer
    // is not installed (logged as a warning).
    bool enableValidation = true;
    // Synchronization validation (hazard detection across submits/presents).
    // Much slower; meant for debug runs chasing sync bugs. Requires
    // enableValidation and the installed layer to support it.
    bool enableSyncValidation = false;
    // Platform surface extensions etc., supplied by the caller (the
    // platform layer's Vulkan seam feeds this). Vulkan only.
    std::vector<const char*> extraExtensions;
};

// The root of a backend's object tree: the API loader/instance/factory and
// its debug hookup. Everything else is created from the Device built on it.
class Instance {
public:
    static Result<std::unique_ptr<Instance>> create(Api api, const InstanceDesc& desc);
    virtual ~Instance() = default;

    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;

    Api api() const { return api_; }
    // Backend object behind this instance (VkInstance under Vulkan). For
    // integrations that must talk to the API directly (surface creation,
    // the ImGui backend); everything else stays API-neutral.
    virtual void* nativeHandle() const = 0;
    virtual bool validationEnabled() const = 0;
    virtual std::uint32_t apiVersion() const = 0;

protected:
    explicit Instance(Api api) : api_(api) {}

private:
    Api api_;
};

} // namespace rend::gpu

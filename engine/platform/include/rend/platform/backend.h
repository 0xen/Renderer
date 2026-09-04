#pragma once

#include "rend/core/result.h"
#include "rend/platform/events.h"
#include "rend/platform/presentation_target.h"
#include "rend/platform/types.h"

#include <memory>
#include <vector>

// Vulkan forward declarations (x64 handles) — platform headers never pull
// in Vulkan itself; this two-function seam is the only crossing point.
typedef struct VkInstance_T* VkInstance;
typedef struct VkSurfaceKHR_T* VkSurfaceKHR;

namespace rend::platform {

enum class BackendKind {
    SDL3,
    // Win32, — future second backend proving the abstraction
};

// One implementation per windowing system. The interface never leaks
// backend types (no SDL/Win32 headers cross this boundary).
class IPlatformBackend {
public:
    virtual ~IPlatformBackend() = default;

    virtual Result<void> initialize() = 0;
    virtual void shutdown() = 0;

    virtual Result<std::unique_ptr<PresentationTarget>> createTarget(const TargetDesc& desc) = 0;

    // Drains pending OS events, translated to portable Event values.
    virtual std::vector<Event> pumpEvents() = 0;

    // Relative mouse mode (mouselook): hides the cursor and delivers
    // unbounded MouseMoved deltas until disabled.
    virtual void setRelativeMouseMode(PresentationTarget& target, bool enabled) = 0;

    // --- Vulkan seam: the only place platform touches Vulkan. ---
    // Instance extensions required to present to this backend's targets.
    virtual std::vector<const char*> requiredVulkanInstanceExtensions() const = 0;
    // Creates a presentable surface for a target this backend created.
    virtual Result<VkSurfaceKHR> createVulkanSurface(VkInstance instance,
                                                     PresentationTarget& target) = 0;
};

Result<std::unique_ptr<IPlatformBackend>> createBackend(BackendKind kind);

} // namespace rend::platform
